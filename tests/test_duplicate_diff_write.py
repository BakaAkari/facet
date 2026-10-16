"""``detect_duplicates``: covering read, Python sort, diff-only write, native popcount
(post-scan speed spec, Step 4).

* ``_hamming_to_all`` returns the same int32 distances on the ``np.bitwise_count`` fast path as
  on the byte-table fallback, and both agree with a plain ``int.bit_count`` reference.
* ``detect_duplicates`` leaves the same ``(duplicate_group_id, is_duplicate_lead)`` on every
  row as HEAD's algorithm (``tests/_post_scan_reference_dup.py``, which takes its row order from
  HEAD's own ``ORDER BY path`` statement, not from the Python sort under test), writes ONLY the
  rows whose stored pair differs (counted by an AFTER UPDATE trigger, so a no-op rewrite shows),
  and a second run writes nothing.
* the two stored-state reads plan ``USING COVERING INDEX idx_post_scan``.

Every database is private (``tmp_path`` + ``init_database``): these passes touch EVERY row, so
they must not run against the shared session database (``.claude/patterns/test-fixtures.md``).
"""

import logging
import random
import re
import sqlite3
from contextlib import closing

import numpy as np
import pytest

import utils.duplicate as dup
from config_resolve import defaults_path
from db.schema import init_database
from tests._post_scan_reference_dup import reference_duplicate_state
from tests.test_post_scan_indexes import MIN_COVERING_SQLITE
from utils.duplicate import (
    _DUP_NULL_PHASH_SQL,
    _DUP_ROWS_SQL,
    _hamming_to_all,
    detect_duplicates,
)

CONFIG = defaults_path()
DIM = 16

_STALE = [(None, None), (None, 0), (99, 1), (3, 0), (1, 1), (7, None)]


# ---------------------------------------------------------------------------
# (a) _hamming_to_all: fast path == table path == plain popcount
# ---------------------------------------------------------------------------

def _hash_array():
    rng = np.random.default_rng(20260716)
    edge = [0, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000, 1, 0x00000000000000FF,
            0xFF00000000000000, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA]
    edge += [1 << b for b in range(64)]
    rand = rng.integers(0, 2**64, size=4000, dtype=np.uint64, endpoint=False)
    return np.concatenate([np.array(edge, dtype=np.uint64), rand])


def _reference_distances(query, hashes):
    q = int(query)
    return np.array([(q ^ int(h)).bit_count() for h in hashes], dtype=np.int32)


@pytest.mark.parametrize("fast", [
    pytest.param(True, marks=pytest.mark.skipif(
        not hasattr(np, "bitwise_count"), reason="numpy < 2: no np.bitwise_count")),
    False,
], ids=["bitwise_count", "byte_table"])
def test_hamming_to_all_matches_reference(monkeypatch, fast):
    monkeypatch.setattr(dup, "_HAS_BITWISE_COUNT", fast)
    hashes = _hash_array()
    for query in (np.uint64(0), np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(1),
                  np.uint64(0x0123456789ABCDEF), hashes[len(hashes) // 2]):
        got = _hamming_to_all(query, hashes)
        assert got.dtype == np.int32
        assert np.array_equal(got, _reference_distances(query, hashes))


@pytest.mark.skipif(not hasattr(np, "bitwise_count"), reason="numpy < 2: no np.bitwise_count")
def test_hamming_fast_and_table_paths_are_identical(monkeypatch):
    hashes = _hash_array()
    query = np.uint64(0xDEADBEEFCAFEF00D)
    monkeypatch.setattr(dup, "_HAS_BITWISE_COUNT", True)
    fast = _hamming_to_all(query, hashes)
    monkeypatch.setattr(dup, "_HAS_BITWISE_COUNT", False)
    table = _hamming_to_all(query, hashes)
    assert fast.dtype == table.dtype == np.int32
    assert np.array_equal(fast, table)


# ---------------------------------------------------------------------------
# a database designed to bite
# ---------------------------------------------------------------------------

def _vec(rng, dim=DIM):
    v = np.array([rng.gauss(0, 1) for _ in range(dim)], dtype=np.float32)
    return v / np.linalg.norm(v)


def _blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


def _near(rng, v):
    w = v + np.array([rng.gauss(0, 0.01) for _ in range(len(v))], dtype=np.float32)
    return w / np.linalg.norm(w)


def _orthogonal(v):
    w = np.roll(v, 1) * np.array([(-1) ** i for i in range(len(v))], dtype=np.float32)
    w = w - float(np.dot(w, v)) * v
    return w / np.linalg.norm(w)


def _flip(h, positions):
    for p in positions:
        h ^= 1 << p
    return h


def _build_rows(kind, seed=7):
    """``[dict]`` of photo rows for ``kind`` in grouped / noemb / nodups / nophash.

    Returns ``(rows, probes)``: ``probes`` names paths the assertions look at.
    """
    rng = random.Random(seed)
    rows = []
    probes = {}

    def add(path, h, agg=5.0, emb=None, **extra):
        rows.append({"path": path, "phash": None if h is None else f"{h:016x}",
                     "aggregate": agg, "clip_embedding": emb, **extra})

    def hash64():
        return rng.getrandbits(64)

    if kind in ("grouped", "noemb"):
        e = (lambda v: _blob(v)) if kind == "grouped" else (lambda v: None)

        # 1. strict pair, no embeddings, hamming 3: the higher aggregate leads.
        h = hash64()
        add("/lib/a/strict_lo.jpg", h, 7.0)
        add("/lib/a/strict_hi.jpg", _flip(h, rng.sample(range(64), 3)), 8.0)
        # 2. strict pair with embeddings that agree.
        h, v = hash64(), _vec(rng)
        add("/lib/a/embed_ok_1.jpg", h, 6.0, e(v))
        add("/lib/a/embed_ok_2.jpg", _flip(h, rng.sample(range(64), 2)), 6.5, e(_near(rng, v)))
        # 3. strict-close pHash but orthogonal embeddings: a collision the cosine gate rejects.
        h, v = hash64(), _vec(rng)
        add("/lib/a/collision_1.jpg", h, 6.0, e(v))
        add("/lib/a/collision_2.jpg", _flip(h, rng.sample(range(64), 2)), 6.0, e(_orthogonal(v)))
        # 4. loose-only (hamming 9) with agreeing embeddings: merged only on the loose gate.
        h, v = hash64(), _vec(rng)
        add("/lib/a/loose_ok_1.jpg", h, 6.0, e(v))
        add("/lib/a/loose_ok_2.jpg", _flip(h, rng.sample(range(64), 9)), 6.2, e(_near(rng, v)))
        # 5. loose-only with orthogonal embeddings: rejected.
        h, v = hash64(), _vec(rng)
        add("/lib/a/loose_bad_1.jpg", h, 6.0, e(v))
        add("/lib/a/loose_bad_2.jpg", _flip(h, rng.sample(range(64), 9)), 6.0, e(_orthogonal(v)))
        # 6. loose-only, one side has no embedding: the strict gate applies, not merged.
        h, v = hash64(), _vec(rng)
        add("/lib/a/loose_half_1.jpg", h, 6.0, e(v))
        add("/lib/a/loose_half_2.jpg", _flip(h, rng.sample(range(64), 9)), 6.0, None)
        # 7. a chain: A-B 5 bits, B-C 5 other bits, A-C 10: one group of three via B.
        bits = rng.sample(range(64), 10)
        a = hash64()
        b = _flip(a, bits[:5])
        c = _flip(b, bits[5:])
        add("/lib/a/chain_a.jpg", a, 5.0)
        add("/lib/a/chain_b.jpg", b, 5.5)
        add("/lib/a/chain_c.jpg", c, 9.0)
        # 8. a lead tie: three identical frames with the same aggregate -> the first path.
        h = hash64()
        for name in ("tie_z", "tie_B", "tie_m"):
            add(f"/lib/b/{name}.jpg", h, 6.0)
        # 9. the global learned score decides an aggregate tie; a per-user row on the other
        #    frame (which the subquery must not read) must not.
        h = hash64()
        add("/lib/b/learned_1.jpg", h, 6.0, learned=100.0, learned_user="someone")
        add("/lib/b/learned_2.jpg", h, 6.0, learned=9.0)
        # 10. eyes-open decides an aggregate tie.
        h = hash64()
        add("/lib/b/eyes_1.jpg", h, 6.0, face_count=1, eyes_open_score=9.0, expression_score=5.0)
        add("/lib/b/eyes_2.jpg", h, 6.0, face_count=1, eyes_open_score=1.0, expression_score=5.0)
        # 11. a minority embedding dimension is treated as absent (strict gate at hamming 9).
        h, v = hash64(), _vec(rng)
        add("/lib/c/dim_1.jpg", h, 6.0, e(v))
        add("/lib/c/dim_2.jpg", _flip(h, rng.sample(range(64), 9)), 6.0, e(_vec(rng, 8)))
        # 12. a zero-norm embedding decodes to nothing (strict gate at hamming 9).
        h, v = hash64(), _vec(rng)
        add("/lib/c/zero_1.jpg", h, 6.0, e(v))
        add("/lib/c/zero_2.jpg", _flip(h, rng.sample(range(64), 9)), 6.0,
            e(np.zeros(DIM, dtype=np.float32)))
        # 13/14. the strict boundary: 6 bits merge, 7 do not (no embeddings).
        h = hash64()
        add("/lib/c/edge6_1.jpg", h, 6.0)
        add("/lib/c/edge6_2.jpg", _flip(h, rng.sample(range(64), 6)), 6.0)
        h = hash64()
        add("/lib/c/edge7_1.jpg", h, 6.0)
        add("/lib/c/edge7_2.jpg", _flip(h, rng.sample(range(64), 7)), 6.0)
        # Paths that sort differently by case, by accent and by script.
        h = hash64()
        add("/Lib/Upper.jpg", h, 6.0)
        add("/lib/été.jpg", h, 6.0)
        add("/lib/日本/photo.jpg", h, 6.0)
        # Singletons, half of them with an embedding.
        for i in range(80):
            add(f"/lib/s/single_{i:03d}.jpg", hash64(), rng.choice([4.0, 5.0, 6.0, 7.0]),
                e(_vec(rng)) if i % 2 else None)
    elif kind == "nodups":
        for i in range(60):
            add(f"/lib/n/p_{i:03d}.jpg", hash64(), 6.0, _blob(_vec(rng)) if i % 3 else None)
    elif kind == "nophash":
        pass
    else:
        raise AssertionError(kind)

    # Rows that take no part in grouping. A real library has some (unscanned phash), and
    # HEAD's blanket reset cleared whatever they carried.
    for i in range(12):
        add(f"/lib/z/nophash_{i:02d}.jpg", None, 6.0,
            _blob(_vec(rng)) if i % 2 else None)
    probes["nophash"] = [f"/lib/z/nophash_{i:02d}.jpg" for i in range(12)]

    rng.shuffle(rows)  # insertion order must not be path order
    for r in rows:
        r["stored"] = rng.choice(_STALE)
    return rows, probes


def _make_db(tmp_path, kind, name="dup.db"):
    db_path = str(tmp_path / name)
    init_database(db_path)
    rows, probes = _build_rows(kind)
    with sqlite3.connect(db_path) as conn:
        for r in rows:
            conn.execute(
                "INSERT INTO photos (path, filename, phash, aggregate, clip_embedding, "
                "face_count, eyes_open_score, expression_score, tech_sharpness, "
                "duplicate_group_id, is_duplicate_lead) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (r["path"], r["path"].rsplit("/", 1)[-1], r["phash"], r["aggregate"],
                 r["clip_embedding"], r.get("face_count"), r.get("eyes_open_score"),
                 r.get("expression_score"), r.get("tech_sharpness"), *r["stored"]))
            if "learned" in r:
                user_id = r.get("learned_user")
                conn.execute(
                    "INSERT INTO learned_scores (photo_path, learned_score, user_id, category) "
                    "VALUES (?, ?, ?, NULL)", (r["path"], r["learned"], user_id))
        conn.commit()
    return db_path, probes


def _state(db_path):
    with sqlite3.connect(db_path) as conn:
        return {p: (g, lead) for p, g, lead in conn.execute(
            "SELECT path, duplicate_group_id, is_duplicate_lead FROM photos")}


def _set_state(db_path, updates):
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE photos SET duplicate_group_id = ?, is_duplicate_lead = ? WHERE path = ?",
            [(g, lead, p) for p, (g, lead) in updates.items()])
        conn.commit()


def _install_write_counter(db_path):
    """An AFTER UPDATE trigger that logs every row UPDATE touches, even a same-value one."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE write_log (path TEXT)")
        conn.execute("CREATE TRIGGER write_log_t AFTER UPDATE ON photos "
                     "BEGIN INSERT INTO write_log VALUES (new.path); END")
        conn.commit()


def _writes(db_path, reset=False):
    with sqlite3.connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM write_log").fetchone()[0]
        if reset:
            conn.execute("DELETE FROM write_log")
            conn.commit()
    return n


def _align_some_rows_with_the_answer(db_path, fraction=0.4, seed=11):
    """Make a share of the rows already correct, so the diff is partial, not total."""
    expected = reference_duplicate_state(db_path, CONFIG)
    rng = random.Random(seed)
    _set_state(db_path, {p: s for p, s in expected.items() if rng.random() < fraction})
    return expected


def _members(state):
    return {p for p, (g, _lead) in state.items() if g is not None}


# ---------------------------------------------------------------------------
# (b) parity with HEAD, diff-only write, idempotence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["grouped", "noemb", "nodups", "nophash"])
def test_detect_duplicates_matches_head_and_writes_only_the_diff(tmp_path, kind, caplog):
    db_path, _ = _make_db(tmp_path, kind)
    expected = _align_some_rows_with_the_answer(db_path)
    before = _state(db_path)
    must_change = {p for p in before if before[p] != expected[p]}

    if kind in ("grouped", "noemb"):
        assert _members(expected), "the fixture must produce duplicate groups"
        assert must_change and len(must_change) < len(before), "the diff must be partial"
    if kind == "nodups":
        assert not _members(expected)
        assert must_change, "stale markings must exist to be cleared"

    _install_write_counter(db_path)
    caplog.set_level(logging.INFO, logger="facet.duplicate")
    detect_duplicates(db_path, CONFIG)

    after = _state(db_path)
    assert after == expected
    # Exactly the differing rows were written, each once.
    assert _writes(db_path) == len(must_change)
    if kind == "nophash":
        assert "No photos with pHash found." in caplog.text
    else:
        assert f"Duplicate state: {len(must_change)} rows changed" in caplog.text


def test_grouped_fixture_exercises_every_gate(tmp_path):
    """The fixture is only worth its parity check if each gate actually decides something."""
    db_path, _ = _make_db(tmp_path, "grouped")
    expected = reference_duplicate_state(db_path, CONFIG)

    def grouped(a, b):
        ga, gb = expected[a][0], expected[b][0]
        return ga is not None and ga == gb

    assert grouped("/lib/a/strict_lo.jpg", "/lib/a/strict_hi.jpg")
    assert expected["/lib/a/strict_hi.jpg"][1] == 1 and expected["/lib/a/strict_lo.jpg"][1] == 0
    assert grouped("/lib/a/embed_ok_1.jpg", "/lib/a/embed_ok_2.jpg")
    assert not grouped("/lib/a/collision_1.jpg", "/lib/a/collision_2.jpg")
    assert grouped("/lib/a/loose_ok_1.jpg", "/lib/a/loose_ok_2.jpg")
    assert not grouped("/lib/a/loose_bad_1.jpg", "/lib/a/loose_bad_2.jpg")
    assert not grouped("/lib/a/loose_half_1.jpg", "/lib/a/loose_half_2.jpg")
    assert grouped("/lib/a/chain_a.jpg", "/lib/a/chain_b.jpg")
    assert grouped("/lib/a/chain_b.jpg", "/lib/a/chain_c.jpg")
    assert expected["/lib/a/chain_c.jpg"][1] == 1
    assert not grouped("/lib/c/dim_1.jpg", "/lib/c/dim_2.jpg")
    assert not grouped("/lib/c/zero_1.jpg", "/lib/c/zero_2.jpg")
    assert grouped("/lib/c/edge6_1.jpg", "/lib/c/edge6_2.jpg")
    assert not grouped("/lib/c/edge7_1.jpg", "/lib/c/edge7_2.jpg")
    # Lead ties: path order, learned score, eyes-open.
    tie = sorted(["/lib/b/tie_z.jpg", "/lib/b/tie_B.jpg", "/lib/b/tie_m.jpg"])
    assert [expected[p][1] for p in tie] == [1, 0, 0]
    assert expected["/lib/b/learned_2.jpg"][1] == 1
    assert expected["/lib/b/eyes_1.jpg"][1] == 1
    # Case / accent / script paths share a hash and sort differently from insertion order.
    three = ["/Lib/Upper.jpg", "/lib/été.jpg", "/lib/日本/photo.jpg"]
    assert len({expected[p][0] for p in three}) == 1
    assert [expected[p][1] for p in sorted(three)] == [1, 0, 0]
    # Rows without a pHash end at (NULL, 0) whatever they carried.
    assert {expected[p] for p in expected if p.startswith("/lib/z/")} == {(None, 0)}


def test_second_run_writes_nothing(tmp_path, caplog):
    db_path, _ = _make_db(tmp_path, "grouped")
    detect_duplicates(db_path, CONFIG)
    first = _state(db_path)
    _install_write_counter(db_path)
    caplog.set_level(logging.INFO, logger="facet.duplicate")
    detect_duplicates(db_path, CONFIG)
    assert _state(db_path) == first
    assert _writes(db_path) == 0
    assert "Duplicate state: 0 rows changed" in caplog.text


def test_only_perturbed_rows_are_rewritten(tmp_path, caplog):
    db_path, probes = _make_db(tmp_path, "grouped")
    detect_duplicates(db_path, CONFIG)
    settled = _state(db_path)
    _install_write_counter(db_path)

    member = "/lib/a/strict_hi.jpg"
    other = "/lib/a/chain_a.jpg"
    loner = "/lib/s/single_000.jpg"
    stale_nophash = probes["nophash"][0]
    assert settled[member][0] is not None and settled[loner] == (None, 0)
    _set_state(db_path, {
        member: (settled[member][0], 1 - settled[member][1]),   # flipped lead
        other: (None, 0),                                       # lost its group
        loner: (5, 1),                                          # phantom group
        stale_nophash: (4, 1),                                  # phantom on a no-pHash row
    })
    _writes(db_path, reset=True)  # the perturbation above is not the code under test
    caplog.set_level(logging.INFO, logger="facet.duplicate")
    detect_duplicates(db_path, CONFIG)
    assert _state(db_path) == settled
    assert _writes(db_path) == 4
    assert "Duplicate state: 4 rows changed" in caplog.text


def test_null_phash_rows_with_stale_markings_are_cleared(tmp_path):
    db_path, probes = _make_db(tmp_path, "grouped")
    updates = {probes["nophash"][0]: (7, 1), probes["nophash"][1]: (None, None),
               probes["nophash"][2]: (2, 0)}
    _set_state(db_path, updates)
    detect_duplicates(db_path, CONFIG)
    after = _state(db_path)
    for path in probes["nophash"]:
        assert after[path] == (None, 0)


def test_no_phash_anywhere_leaves_the_table_untouched(tmp_path, caplog):
    db_path, _ = _make_db(tmp_path, "nophash")
    before = _state(db_path)
    _install_write_counter(db_path)
    caplog.set_level(logging.INFO, logger="facet.duplicate")
    detect_duplicates(db_path, CONFIG)
    assert _state(db_path) == before
    assert _writes(db_path) == 0
    assert "No photos with pHash found." in caplog.text


# ---------------------------------------------------------------------------
# (c) the stored-state reads are index-only
# ---------------------------------------------------------------------------

def _plan(db_path, sql):
    # A fresh connection per plan: a cached EXPLAIN statement is not re-planned after a
    # DROP INDEX on the same connection, which would make the control below vacuous.
    with closing(sqlite3.connect(db_path)) as conn:
        return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]


def _reads_photos_from_covering_index(details):
    photos = [d for d in details if re.search(r"\b(SCAN|SEARCH) photos\b", d)]
    return bool(photos) and all("USING COVERING INDEX idx_post_scan" in d for d in photos)


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < MIN_COVERING_SQLITE,
    reason="SQLite %s is older than the measured covering-index floor %s"
           % (sqlite3.sqlite_version, ".".join(map(str, MIN_COVERING_SQLITE))))
@pytest.mark.parametrize("sql", [_DUP_ROWS_SQL, _DUP_NULL_PHASH_SQL],
                         ids=["scalar_read", "null_phash_read"])
def test_stored_state_reads_plan_a_covering_index(tmp_path, sql):
    db_path, _ = _make_db(tmp_path, "grouped")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("ANALYZE photos")
        conn.commit()
    details = _plan(db_path, sql)
    assert _reads_photos_from_covering_index(details), details

    # Control: without the index the same SQL does not plan a covering read, so the
    # assertion above can fail.
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("DROP INDEX idx_post_scan")
        conn.commit()
    details = _plan(db_path, sql)
    assert not _reads_photos_from_covering_index(details), details


def test_scalar_read_has_no_order_by():
    """ORDER BY path made the planner walk the path autoindex plus the table."""
    assert "ORDER BY" not in _DUP_ROWS_SQL.upper()
