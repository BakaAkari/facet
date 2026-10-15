"""Viewer start-up and shutdown: the library-locked schema step and the final checkpoint.

Viewer-API speed spec, Step 11 (``api.schema_startup.init_schema_at_startup`` and the
lifespan call to it) and Step 12 (the shutdown checkpoint no longer depends on
``performance.wal_checkpoint_minutes``, and the change-token sentinels close first).

Every test builds its own DB with ``init_database`` (``.claude/patterns/test-fixtures.md``)
and patches ``api.database.DEFAULT_DB_PATH``, the global the lifespan and the request
connections read, plus ``api.config._FULL_CONFIG`` so no test depends on the operator's
``scoring_config.json``. No auth dependency is patched: the lifespan needs none.
"""

import gc
import logging
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import api.config as api_config
from api import create_app
from api.change_token import db_change_token
from api.schema_startup import init_schema_at_startup
from db.schema import init_database
from facet import LIBRARY_JOB_MAINTENANCE, LibraryLock

# The eight gallery indexes the viewer build step is responsible for, written out on
# purpose (the registry test lives in tests/test_gallery_indexes.py).
GALLERY_INDEXES = (
    'idx_gallery_aggregate', 'idx_gallery_date_desc', 'idx_gallery_date_asc',
    'idx_gallery_category', 'idx_gallery_camera', 'idx_star_rating_path',
    'idx_gallery_iaa_asc', 'idx_stats_categories',
)


def _make_db(tmp_path, rows=300, name="life.db", drop_gallery_indexes=False):
    path = str(tmp_path / name)
    init_database(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        "INSERT INTO photos (path, filename, aggregate) VALUES (?, ?, ?)",
        [(f"/{name}/p{i}.jpg", f"p{i}.jpg", float(i % 10)) for i in range(rows)])
    if drop_gallery_indexes:
        for idx in GALLERY_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {idx}")
    conn.commit()
    conn.close()
    return path


def _present_indexes(path):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
    finally:
        conn.close()


def _config(wal_minutes):
    return {
        "performance": {"wal_checkpoint_minutes": wal_minutes},
        "viewer": {"features": {"show_capsules": False}},
    }


@contextmanager
def _patched_server(db_path, wal_minutes=0):
    """Point the server globals at a tmp DB and a config the test owns."""
    with mock.patch("api.database.DEFAULT_DB_PATH", db_path), \
            mock.patch.object(api_config, "_FULL_CONFIG", _config(wal_minutes)):
        yield


def _wait_for_wal_gone(db_path, seconds=5.0):
    wal = db_path + "-wal"
    deadline = time.monotonic() + seconds
    while os.path.exists(wal) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not os.path.exists(wal)


def _row_count_in_db_file_alone(db_path, path):
    """Rows visible when only the main ``.db`` file is read (no WAL, no locks)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM photos WHERE path = ?", (path,)).fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 11: init_schema_at_startup
# ---------------------------------------------------------------------------

def test_lock_held_defers_then_builds_after_release(tmp_path, caplog):
    """Verify 4: a held library lock defers the index build; releasing it allows it."""
    db = _make_db(tmp_path, drop_gallery_indexes=True)
    holder = LibraryLock(db, kind=LIBRARY_JOB_MAINTENANCE).acquire()
    # An earlier test can leave the "facet" logger at INFO, so _make_db's own
    # "Creating index" records may already be captured; only this call's count.
    caplog.clear()
    try:
        with caplog.at_level(logging.INFO):
            result = init_schema_at_startup(db)
        assert result == 'deferred'
        assert not set(GALLERY_INDEXES) & _present_indexes(db)
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING and "Library lock" in r.getMessage()]
        assert len(warnings) == 1
        # The warning names the holder: its kind and pid come from the lock payload.
        assert f"pid {os.getpid()}" in warnings[0]
        assert "maintenance is already running" in warnings[0]
        assert not [r for r in caplog.records if "Creating index" in r.getMessage()]
    finally:
        holder.release()

    assert init_schema_at_startup(db) == 'built'
    assert set(GALLERY_INDEXES) <= _present_indexes(db)


def test_lock_held_by_another_process_defers(tmp_path):
    """The real case: a different process (a scan) holds the lock."""
    db = _make_db(tmp_path, drop_gallery_indexes=True)
    script = tmp_path / "hold_lock.py"
    script.write_text(textwrap.dedent("""
        import sys, time
        sys.path.insert(0, sys.argv[1])
        from facet import LibraryLock, LIBRARY_JOB_SCAN
        lock = LibraryLock(sys.argv[2], kind=LIBRARY_JOB_SCAN).acquire()
        print("held", flush=True)
        sys.stdin.readline()
        lock.release()
    """))
    root = str(Path(__file__).resolve().parent.parent)
    child = subprocess.Popen(
        [sys.executable, "-I", str(script), root, db],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        assert init_schema_at_startup(db) == 'deferred'
        assert not set(GALLERY_INDEXES) & _present_indexes(db)
    finally:
        child.stdin.write("\n")
        child.stdin.flush()
        child.wait(timeout=30)
    assert init_schema_at_startup(db) == 'built'
    assert set(GALLERY_INDEXES) <= _present_indexes(db)


def test_deferred_still_migrates_tables_and_columns(tmp_path):
    """A deferred start is not a no-op: the schema the endpoints read is still current."""
    db = _make_db(tmp_path, drop_gallery_indexes=True)
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE iqa_declined")
    conn.commit()
    conn.close()
    holder = LibraryLock(db, kind=LIBRARY_JOB_MAINTENANCE).acquire()
    try:
        assert init_schema_at_startup(db) == 'deferred'
    finally:
        holder.release()
    conn = sqlite3.connect(db)
    try:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'iqa_declined'").fetchone()
    finally:
        conn.close()


def test_unexpected_failure_returns_failed_and_releases_lock(tmp_path, caplog):
    db = _make_db(tmp_path)
    with mock.patch("db.schema.init_database", side_effect=RuntimeError("boom")), \
            caplog.at_level(logging.WARNING):
        assert init_schema_at_startup(db) == 'failed'
    assert "Schema migration on startup failed" in caplog.text
    # The lock was released in the finally: it can be taken again straight away.
    LibraryLock(db, kind=LIBRARY_JOB_MAINTENANCE).acquire().release()


def test_concurrent_starts(tmp_path):
    """Verify 5: two simultaneous starts on an upgraded DB; nobody raises, someone builds."""
    db = _make_db(tmp_path, rows=3000, drop_gallery_indexes=True)
    barrier = threading.Barrier(2)
    results, errors = [], []

    def start():
        try:
            barrier.wait(timeout=10)
            results.append(init_schema_at_startup(db))
        except BaseException as ex:  # noqa: BLE001 - the test reports whatever escapes
            errors.append(ex)

    threads = [threading.Thread(target=start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert len(results) == 2
    assert set(results) <= {'built', 'deferred'}
    assert 'built' in results
    # A start that deferred leaves the build to the next start: run it.
    assert init_schema_at_startup(db) == 'built'
    assert set(GALLERY_INDEXES) <= _present_indexes(db)


# ---------------------------------------------------------------------------
# Step 11: the lifespan calls it
# ---------------------------------------------------------------------------

def test_lifespan_calls_schema_startup_once(tmp_path):
    """Verify 6: the app still starts, and the lifespan runs the helper exactly once."""
    db = _make_db(tmp_path, drop_gallery_indexes=True)
    with _patched_server(db), \
            mock.patch("api.schema_startup.init_schema_at_startup",
                       wraps=init_schema_at_startup) as spy:
        with TestClient(create_app()):
            pass
    spy.assert_called_once_with(db)
    assert set(GALLERY_INDEXES) <= _present_indexes(db)


# ---------------------------------------------------------------------------
# Step 12: the shutdown checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wal_minutes", [0, 30])
def test_shutdown_checkpoint_leaves_db_file_complete(tmp_path, wal_minutes):
    """Verify 1 (wal_checkpoint_minutes = 0) and 2 (= 30, the regression guard).

    The change-token sentinel is a read-only connection; if it were the last closer
    the ``-wal`` would stay behind holding commits the ``.db`` alone lacks.
    """
    db = _make_db(tmp_path)
    marker = "/life.db/committed-while-running.jpg"
    with _patched_server(db, wal_minutes):
        with TestClient(create_app()):
            # ``init_database`` leaves its ``with sqlite3.connect(...)`` connection
            # unclosed (a ``with`` commits, it does not close), so it lingers until
            # the cyclic collector runs and, as the last read-write closer, would
            # delete the -wal itself and hide a missing shutdown checkpoint. Free it
            # now. The sentinel is held by the registry, so collection cannot free it.
            gc.collect()
            db_change_token(db)  # opens the sentinel on the tmp DB
            other = sqlite3.connect(db)
            other.execute("INSERT INTO photos (path, filename) VALUES (?, ?)",
                          (marker, "committed-while-running.jpg"))
            other.commit()
            other.close()
            # While the app runs, the commit lives in the WAL.
            assert os.path.exists(db + "-wal")
    assert _wait_for_wal_gone(db), "the -wal outlived shutdown"
    assert _row_count_in_db_file_alone(db, marker) == 1


def test_shutdown_closes_sentinels_before_the_final_checkpoint(tmp_path):
    """Verify 3: ordering."""
    db = _make_db(tmp_path)
    calls = []
    with _patched_server(db, wal_minutes=0), \
            mock.patch("api.change_token.close_change_token_sentinels",
                       side_effect=lambda: calls.append("close_sentinels")), \
            mock.patch("api._final_wal_checkpoint",
                       side_effect=lambda: calls.append("checkpoint")):
        with TestClient(create_app()):
            assert calls == []
    assert calls == ["close_sentinels", "checkpoint"]
