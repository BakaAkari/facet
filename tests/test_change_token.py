"""The count / hidden-summary cache key: a sentinel-backed change token.

The old key hashed size+mtime of the database and its ``-wal``. A pure read moves
the ``-wal`` mtime, so every request produced a new key and the cache never hit.
``api.change_token.db_change_token`` reads ``PRAGMA data_version`` from one
long-lived read-only connection instead: it moves on every commit by any other
connection or process and on nothing else.

Every test builds its own DB with ``init_database`` (never a hand-rolled
``CREATE TABLE photos``) and patches ``api.database.DEFAULT_DB_PATH``, the global
the request connections read.
"""

import logging
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from unittest import mock

import pytest

from db.schema import init_database


@pytest.fixture()
def ct():
    """The module under test, imported lazily so route tests below stay runnable."""
    from api import change_token
    return change_token


def _make_db(tmp_path, rows=1, name="tok.db"):
    path = str(tmp_path / name)
    init_database(path)
    if not rows:
        return path
    conn = sqlite3.connect(path)
    for i in range(rows):
        conn.execute("INSERT INTO photos (path, filename) VALUES (?, ?)",
                     (f"/{name}/p{i}.jpg", f"p{i}.jpg"))
    conn.commit()
    conn.close()
    return path


def _commit_row(path, n):
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO photos (path, filename) VALUES (?, ?)",
                 (f"/extra/{n}.jpg", f"{n}.jpg"))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Unit: the token itself
# ---------------------------------------------------------------------------

def test_pure_read_requests_share_one_token(ct, tmp_path):
    """Six request-style open/read/close cycles: the token is read while the
    request connection is open and must not move (the old function gave six)."""
    from api.database import get_db_connection
    path = _make_db(tmp_path, rows=0)  # no writer left a WAL behind: the pre-fix shape
    tokens = set()
    with mock.patch("api.database.DEFAULT_DB_PATH", path):
        for _ in range(6):
            conn = get_db_connection()  # the pragma set a real request opens with
            conn.execute("SELECT COUNT(*) FROM photos").fetchone()
            tokens.add(ct.db_change_token(path))
            conn.close()
            time.sleep(0.05)  # filesystem mtime granularity: space the requests out
    assert len(tokens) == 1


def test_commit_from_another_connection_changes_token(ct, tmp_path):
    path = _make_db(tmp_path)
    before = ct.db_change_token(path)
    assert ct.db_change_token(path) == before
    _commit_row(path, 1)
    after = ct.db_change_token(path)
    assert after != before
    assert ct.db_change_token(path) == after


def test_hundred_child_process_commits_each_change_token(ct, tmp_path):
    """A size+mtime token missed 169 of 599 commits from a long-lived external
    writer; the data_version sentinel must miss none."""
    path = _make_db(tmp_path)
    script = tmp_path / "child_writer.py"
    script.write_text(textwrap.dedent("""
        import sqlite3, sys
        conn = sqlite3.connect(sys.argv[1])
        conn.execute("PRAGMA wal_autocheckpoint = 5")
        n = 0
        for line in sys.stdin:
            n += 1
            conn.execute("INSERT INTO photos (path, filename) VALUES (?, ?)",
                         ("/child/%d.jpg" % n, "%d.jpg" % n))
            conn.commit()
            print("ok", flush=True)
    """))
    proc = subprocess.Popen(
        [sys.executable, "-I", str(script), path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        seen = {ct.db_change_token(path)}
        for _ in range(100):
            proc.stdin.write("go\n")
            proc.stdin.flush()
            assert proc.stdout.readline().strip() == "ok"
            token = ct.db_change_token(path)
            assert token not in seen
            seen.add(token)
        assert len(seen) == 101
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


def _truncate_checkpoint(path):
    """Commit a row on a writer that stays open (so the WAL is non-empty), then
    TRUNCATE-checkpoint and return ``(busy, wal_size_after)``."""
    writer = sqlite3.connect(path)
    writer.execute("INSERT INTO photos (path, filename) VALUES ('/ckpt.jpg', 'ckpt.jpg')")
    writer.commit()
    assert os.path.getsize(path + "-wal") > 0
    busy = writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
    size = os.path.getsize(path + "-wal")
    writer.close()
    return busy, size


def test_sentinel_does_not_block_truncate_checkpoint(ct, tmp_path):
    """``wal_checkpoint`` never raises on contention, it returns busy=1, so
    asserting that it "succeeds" proves nothing: assert busy == 0 and an empty WAL."""
    path = _make_db(tmp_path)
    ct.db_change_token(path)
    busy, size = _truncate_checkpoint(path)
    assert busy == 0
    assert size == 0


def test_pinned_reader_makes_truncate_checkpoint_busy(tmp_path):
    """Control for the test above: a reader holding a statement open DOES turn
    the checkpoint busy, so ``busy == 0`` above can fail."""
    path = _make_db(tmp_path, rows=5)
    reader = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
    cur = reader.execute("SELECT path FROM photos")
    cur.fetchone()
    try:
        busy, _ = _truncate_checkpoint(path)
        assert busy == 1
    finally:
        cur.close()
        reader.close()


def test_nonexistent_path_gives_miss_tokens_and_one_warning(ct, tmp_path, caplog):
    missing = str(tmp_path / "does_not_exist.db")
    with caplog.at_level(logging.WARNING, logger="facet.api.change_token"):
        first = ct.db_change_token(missing)
        second = ct.db_change_token(missing)
    assert first != second
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert missing in warnings[0].getMessage()


def test_close_is_idempotent_and_reopen_never_repeats_a_token(ct, tmp_path):
    """The token before the first commit is read on purpose: a reopened
    sentinel restarts ``data_version``, so without the generation it would
    reproduce that earliest token."""
    path = _make_db(tmp_path)
    t0 = ct.db_change_token(path)
    _commit_row(path, 1)
    t1 = ct.db_change_token(path)
    assert t1 != t0
    ct.close_change_token_sentinels()
    ct.close_change_token_sentinels()
    t2 = ct.db_change_token(path)
    assert t2 not in {t0, t1}


def test_close_releases_every_sentinel(ct, tmp_path):
    path = _make_db(tmp_path)
    ct.db_change_token(path)
    ct.close_change_token_sentinels()
    assert ct._sentinels == {}


def test_busy_pragma_returns_miss_token_and_keeps_sentinel(ct, tmp_path):
    path = _make_db(tmp_path)
    stable = ct.db_change_token(path)
    with mock.patch.object(ct, "_read_data_version",
                           side_effect=sqlite3.OperationalError("database is locked")):
        busy_a = ct.db_change_token(path)
        busy_b = ct.db_change_token(path)
    assert busy_a != busy_b
    assert busy_a != stable
    # Reader restored: the very same token comes back, so no reopen happened
    # (a reopen bumps the generation and would give a new token).
    assert ct.db_change_token(path) == stable


def test_other_pragma_failure_drops_sentinel_and_misses(ct, tmp_path, caplog):
    path = _make_db(tmp_path)
    stable = ct.db_change_token(path)
    with mock.patch.object(ct, "_read_data_version",
                           side_effect=sqlite3.DatabaseError("disk image is malformed")):
        with caplog.at_level(logging.WARNING, logger="facet.api.change_token"):
            miss = ct.db_change_token(path)
    assert miss != stable
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    # The sentinel was reopened for the next call: a new generation.
    assert ct.db_change_token(path) != stable


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows refuses to replace a file the sentinel holds open")
def test_replaced_db_file_reopens_sentinel(ct, tmp_path):
    path = _make_db(tmp_path, rows=2)
    other = _make_db(tmp_path, rows=7, name="other.db")
    before = ct.db_change_token(path)
    os.replace(other, path)
    assert ct.db_change_token(path) != before


# ---------------------------------------------------------------------------
# Integration: the cache key as the routes use it
# ---------------------------------------------------------------------------

def test_second_identical_count_request_hits_the_cache(edition_client, seed_photos_prefix):
    import api.db_helpers as db_helpers
    with mock.patch.object(db_helpers, "_count_cache_store",
                           wraps=db_helpers._count_cache_store) as store:
        first = edition_client.get("/api/photos/count").json()["total"]
        second = edition_client.get("/api/photos/count").json()["total"]
        assert first == second
        assert store.call_count == 1

        seed_photos_prefix("/tokencount/", [
            {"path": "/tokencount/a.jpg", "filename": "a.jpg", "aggregate": 5.0},
        ])
        third = edition_client.get("/api/photos/count").json()["total"]
    assert third == first + 1


def test_cache_key_follows_the_patched_database_path(edition_client, tmp_path):
    """Connections read ``api.database.DEFAULT_DB_PATH``; the cache key must too,
    at call time. Reading ``db_helpers``' import-time binding would hand the
    second database the first one's count (no cache clearing in between)."""
    first_db = _make_db(tmp_path, rows=3, name="first.db")
    second_db = _make_db(tmp_path, rows=5, name="second.db")
    with mock.patch("api.database.DEFAULT_DB_PATH", first_db):
        assert edition_client.get("/api/photos/count").json()["total"] == 3
    with mock.patch("api.database.DEFAULT_DB_PATH", second_db):
        assert edition_client.get("/api/photos/count").json()["total"] == 5
    with mock.patch("api.database.DEFAULT_DB_PATH", first_db):
        assert edition_client.get("/api/photos/count").json()["total"] == 3
