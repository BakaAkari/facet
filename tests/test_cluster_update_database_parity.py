"""``FaceClusterer._update_database`` writes only the faces whose owner changes.

HEAD blanket-NULLed every face's ``person_id`` and re-wrote it one UPDATE at a
time, committing in 10% chunks (145,677 faces: 250 s warm). The phase now reads
the stored owner map once, builds the wanted one in memory and writes the
difference in a single compare-and-set ``executemany`` inside ONE transaction.

HDBSCAN is not run: parity is proven by calling ``_update_database`` directly
with a hand-made ``face_to_cluster`` -- exactly the part that changed -- on a
synthetic library, against ``tests/fixtures/faces_update_database_golden.json``,
which was produced by the code BEFORE the change (see ``tests/fixtures/README.md``).

Every database here is a private ``init_database`` file under ``tmp_path``. No
photo file is opened: the legacy-thumbnail decode (``_generate_face_thumbnail``,
which reads the ORIGINAL photo) is stubbed everywhere, and the synthetic paths
do not exist.
"""

import base64
import contextlib
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from db import init_database
from db.embedding_space import ARCFACE_MODEL, SFACE_MODEL
from db.person_centroids import recompute_person_centroid
from faces.clusterer import FaceClusterer

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "faces_update_database_golden.json"

DIM = 512
SFACE_DIM = 128
CASES = ("incremental", "stable", "incremental_new_faces", "named_only", "force")

# Blob labels. 0-4 are owned going in; 5 is claimed by the centroid rule; 6 and 7
# are new people, and 6 has no stored face thumbnail (the legacy decode path).
BLOB_SIZES = (36, 34, 32, 30, 36, 34, 38, 32)
LEGACY_BLOB = 6
NOISE_LABEL = -1


# ---------------------------------------------------------------------------
# The synthetic library. Everything below the golden driver is also run, by hand,
# against the pre-change tree to produce the golden file -- so it may only call
# API that exists in both trees (init_database, FaceClusterer._update_database).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _db(path):
    """Commit on success, always close: ``with sqlite3.connect`` leaves the file open."""
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    return (vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)).astype(np.float32)


def _axis(seed, dim=DIM):
    return _unit(np.random.default_rng(seed).normal(size=dim))


def _points(axis, n, seed, scale=0.02):
    rng = np.random.default_rng(seed)
    return _unit(axis + rng.normal(scale=scale, size=(n, axis.shape[0])))


def _mean_centroid(vectors):
    centroid = np.mean(vectors, axis=0).astype(np.float32)
    return (centroid / (np.linalg.norm(centroid) + 1e-10)).astype(np.float32)


def _blur(centroid, away, seed):
    """A unit vector at a chosen distance from ``centroid`` (``away`` = noise scale)."""
    noise = _axis(seed, centroid.shape[0])
    return _unit(centroid + away * noise)


def _add_face(conn, label_of, n, embedding, model, person_id, thumbnail, label, confidence):
    path = f"/synthetic/face{n:04d}.jpg"
    conn.execute("INSERT OR IGNORE INTO photos(path) VALUES (?)", (path,))
    cur = conn.execute(
        "INSERT INTO faces(photo_path, face_index, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
        "confidence, person_id, face_thumbnail, embedding_model) "
        "VALUES (?, 0, ?, 10, 10, 60, 60, ?, ?, ?, ?)",
        (path, sqlite3.Binary(embedding.astype(np.float32).tobytes()), confidence,
         person_id, thumbnail, model))
    if label is not None:
        label_of[cur.lastrowid] = label
    return cur.lastrowid


def build_library(db_path):
    """Create the library and return ``label_of``: face id -> cluster label.

    Persons going in (id: what it exercises):
      1 Alice (named, centroid)      owns 2/3 of blob 0 -> centroid rule, plus 4 noise
                                     faces restored; her representative is stale
      2 Bob   (named, centroid)      owns blob 1 -> but 3 of its faces are Alice's
      3 unnamed auto, centroid       owns blob 2
      4 Carol (named, NULL centroid) owns blob 3 and 3 noise faces -> history rule only
      5 unnamed auto, unrelated      owns blob 4 -> history rule, no centroid match
      6 sface person                 another embedding space: never touched
      7 unnamed auto, unrelated      owns 2 faces of blob 5 -> ends empty, deleted
      8, 9 unnamed auto, centroids   both near blob 5, 9 is the nearer and the LATER
                                     one, so first-above-threshold != best
      10 Dan (named, centroid)       a centroid just under merge_threshold of blob 7
    """
    init_database(db_path)
    label_of = {}
    rng = np.random.default_rng(20260101)
    axes = [_axis(100 + i) for i in range(len(BLOB_SIZES))]
    blobs = [_points(axes[i], BLOB_SIZES[i], 200 + i) for i in range(len(BLOB_SIZES))]

    persons = {
        1: ("Alice", _mean_centroid(blobs[0]), 0),
        2: ("Bob", _mean_centroid(blobs[1]), 0),
        3: (None, _mean_centroid(blobs[2]), 1),
        4: ("Carol", None, 0),
        5: (None, _axis(301), 1),
        7: (None, _axis(302), 1),
        8: (None, _blur(_mean_centroid(blobs[5]), 0.9, 303), 1),
        9: (None, _blur(_mean_centroid(blobs[5]), 0.35, 304), 1),
        10: ("Dan", _blur(_mean_centroid(blobs[7]), 1.4, 305), 0),
    }
    # The scenario's premises, so a changed seed cannot silently stop exercising them.
    cent5, cent7 = _mean_centroid(blobs[5]), _mean_centroid(blobs[7])
    sim = {pid: float(np.dot(cent5, persons[pid][1])) for pid in (8, 9)}
    assert 0.6 < sim[8] < sim[9], sim
    assert float(np.dot(cent7, persons[10][1])) < 0.6

    with _db(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for pid, (name, centroid, auto) in persons.items():
            conn.execute(
                "INSERT INTO persons(id, name, centroid, centroid_model, auto_clustered, face_count) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (pid, name, None if centroid is None else centroid.tobytes(),
                 None if centroid is None else ARCFACE_MODEL, auto))
        sface_centroid = _axis(306, SFACE_DIM)
        conn.execute(
            "INSERT INTO persons(id, name, centroid, centroid_model, auto_clustered, face_count) "
            "VALUES (6, NULL, ?, ?, 1, 0)", (sface_centroid.tobytes(), SFACE_MODEL))

        owner_of_blob = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        n = 0
        face_numbers = {}
        for blob, vectors in enumerate(blobs):
            for i, vec in enumerate(vectors):
                owner = owner_of_blob.get(blob)
                if blob == 0 and i % 3 == 2:
                    owner = None                       # a face the person does not own yet
                if blob == 1 and i < 3:
                    owner = 1                          # stale: Bob's cluster, Alice's row
                if blob == 5 and i < 2:
                    owner = 7                          # a minority that makes person 7 a husk
                thumb = None if blob == LEGACY_BLOB else b"stored-thumb-%d" % n
                # every fifth blob-0 face carries the legacy NULL model tag
                model = None if (blob == 0 and i % 5 == 0) else ARCFACE_MODEL
                fid = _add_face(conn, label_of, n, vec, model, owner, thumb, blob,
                                float(rng.uniform(0.5, 1.0)))
                face_numbers.setdefault(blob, []).append(fid)
                n += 1

        noise = _unit(rng.normal(size=(12, DIM)))
        noise_owner = [1, 1, 1, 1, 4, 4, 4, None, None, None, None, None]
        for vec, owner in zip(noise, noise_owner):
            _add_face(conn, label_of, n, vec, ARCFACE_MODEL, owner, b"stored-thumb-%d" % n,
                      NOISE_LABEL, float(rng.uniform(0.5, 1.0)))
            n += 1

        # A second embedding space: loaded by nobody here, owned by person 6.
        for i, vec in enumerate(_points(sface_centroid, 10, 400)):
            _add_face(conn, {}, n, vec, SFACE_MODEL, 6 if i < 6 else None,
                      b"stored-thumb-%d" % n, None, 0.9)
            n += 1

        # Stale representatives: Alice's points at a face that is Bob's after the run.
        conn.execute("UPDATE persons SET representative_face_id = ? WHERE id = 1",
                     (face_numbers[1][0],))
        conn.execute("UPDATE persons SET representative_face_id = ? WHERE id = 2",
                     (face_numbers[1][5],))
        conn.execute("UPDATE persons SET representative_face_id = ? WHERE id = 3",
                     (face_numbers[2][0],))
        conn.commit()
    return label_of


def add_new_faces(db_path, label_of):
    """Unowned faces joining an existing library: 6 near blob 0, 5 near the legacy
    blob, 4 noise. Returns nothing; ``label_of`` is extended in place."""
    with _db(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        n = 5000
        for vec in _points(_axis(100), 6, 500):
            _add_face(conn, label_of, n, vec, ARCFACE_MODEL, None, b"stored-thumb-%d" % n, 0, 0.8)
            n += 1
        for vec in _points(_axis(100 + LEGACY_BLOB), 5, 501):
            _add_face(conn, label_of, n, vec, ARCFACE_MODEL, None, None, LEGACY_BLOB, 0.8)
            n += 1
        for vec in _unit(np.random.default_rng(502).normal(size=(4, DIM))):
            _add_face(conn, label_of, n, vec, ARCFACE_MODEL, None, b"stored-thumb-%d" % n,
                      NOISE_LABEL, 0.8)
            n += 1
        conn.commit()


def make_clusterer(db_path):
    return FaceClusterer(db_path, embedding_model=ARCFACE_MODEL, merge_threshold=0.6)


def run_phase(db_path, label_of, **mode):
    """What ``cluster_faces`` does around HDBSCAN, with the labels supplied."""
    clusterer = make_clusterer(db_path)
    face_ids, embeddings = clusterer.load_embeddings()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    mapping = {fid: label_of[fid] for fid in face_ids}
    clusterer._update_database(mapping, normalized, face_ids, **mode)
    return clusterer


def _legacy_decoder_stub(self, conn, face_id):
    return b"legacy-thumb-%d" % face_id


@contextlib.contextmanager
def stubbed_decoder(stub=_legacy_decoder_stub):
    """Replace the legacy face-thumbnail decode: it opens the ORIGINAL photo file."""
    with mock.patch.object(FaceClusterer, "_generate_face_thumbnail", stub):
        yield


def dump_state(db_path):
    """Everything the phase is allowed to change, in a JSON-able form."""
    conn = sqlite3.connect(db_path)
    try:
        faces = [list(r) for r in conn.execute("SELECT id, person_id FROM faces ORDER BY id")]
        persons = []
        for (pid, name, count, rep, auto, model, centroid, thumb) in conn.execute(
                "SELECT id, name, face_count, representative_face_id, auto_clustered, "
                "centroid_model, centroid, face_thumbnail FROM persons ORDER BY id"):
            persons.append({
                "id": pid, "name": name, "face_count": count,
                "representative_face_id": rep, "auto_clustered": auto,
                "centroid_model": model,
                "centroid": None if centroid is None else base64.b64encode(centroid).decode("ascii"),
                "face_thumbnail_sha256": None if thumb is None else hashlib.sha256(thumb).hexdigest(),
            })
    finally:
        conn.close()
    return {"faces": faces, "persons": persons}


def run_all_cases(root):
    """Run the five cases under ``root`` and return ``{case: dump_state}``."""
    root = Path(root)
    out = {}
    with stubbed_decoder():
        inc_db = str(root / "incremental.db")
        label_of = build_library(inc_db)
        run_phase(inc_db, label_of)
        out["incremental"] = dump_state(inc_db)

        run_phase(inc_db, label_of)
        out["stable"] = dump_state(inc_db)

        add_new_faces(inc_db, label_of)
        run_phase(inc_db, label_of)
        out["incremental_new_faces"] = dump_state(inc_db)

        named_db = str(root / "named_only.db")
        label_named = build_library(named_db)
        run_phase(named_db, label_named, preserve_named_only=True)
        out["named_only"] = dump_state(named_db)

        force_db = str(root / "force.db")
        label_force = build_library(force_db)
        run_phase(force_db, label_force, force=True)
        out["force"] = dump_state(force_db)
    return out


# ---------------------------------------------------------------------------
# Parity against the golden file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    return run_all_cases(tmp_path_factory.mktemp("update_database_parity"))


@pytest.mark.parametrize("case", CASES)
def test_face_owners_match_the_golden(case, actual, golden):
    assert actual[case]["faces"] == golden[case]["faces"]


@pytest.mark.parametrize("case", CASES)
def test_person_rows_match_the_golden(case, actual, golden):
    def plain(dump):
        return [{k: v for k, v in p.items() if k != "centroid"} for p in dump["persons"]]
    assert plain(actual[case]) == plain(golden[case])


@pytest.mark.parametrize("case", CASES)
def test_centroids_match_the_golden(case, actual, golden):
    """``ORDER BY id`` and the unchanged mean pin the summation order, so on the machine that
    produced the golden the bytes are identical. The normalisation goes through the BLAS
    build, which differs between platforms in the last float32 bit, so a tolerance is compared
    here; it is far below anything a face-to-person distance can see."""
    def vec(b64):
        return None if b64 is None else np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    got = [vec(p["centroid"]) for p in actual[case]["persons"]]
    want = [vec(p["centroid"]) for p in golden[case]["persons"]]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        if w is None:
            assert g is None
        else:
            np.testing.assert_allclose(g, w, atol=1e-6)


def test_the_scenario_exercises_what_it_claims(actual):
    """The golden proves nothing if a case degenerates into a no-op."""
    incremental = {p["id"]: p for p in actual["incremental"]["persons"]}
    assert 7 not in incremental, "person 7 must end empty and be deleted"
    assert 9 in incremental and incremental[9]["face_count"] >= BLOB_SIZES[5] - 2
    auto_new = [p for pid, p in incremental.items() if pid > 10]
    assert len(auto_new) >= 2, "blobs 6 and 7 become new persons"
    assert any(p["name"] == "Dan" for p in incremental.values())
    assert any(p["name"] == "Carol" and p["face_count"] >= BLOB_SIZES[3] + 3
               for p in incremental.values())
    legacy = [p for p in auto_new if p["face_thumbnail_sha256"] ==
              hashlib.sha256(b"legacy-thumb-%d" % p["representative_face_id"]).hexdigest()]
    assert legacy, "a new person must have received a legacy-decoded thumbnail"
    force = {p["id"] for p in actual["force"]["persons"]}
    assert not force & {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "force deletes every person"
    named = {p["id"] for p in actual["named_only"]["persons"]}
    assert {1, 2, 4, 10} <= named and not named & {3, 5, 7, 8, 9}


def test_stable_run_changes_no_face_and_no_person(actual):
    """A second run over the first run's output moves no face and no person row.

    Centroids are the one thing allowed to move, by float noise only: a person the
    first run CREATED carries the cluster's mean, and the second run recomputes it
    from the stored embeddings (``recompute_person_centroid``). HEAD does the same.
    """
    first, second = actual["incremental"], actual["stable"]
    assert second["faces"] == first["faces"]

    def plain(dump):
        return [{k: v for k, v in p.items() if k != "centroid"} for p in dump["persons"]]
    assert plain(second) == plain(first)

    def vec(b64):
        return None if b64 is None else np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    for a, b in zip(first["persons"], second["persons"]):
        if a["centroid"] is None:
            assert b["centroid"] is None
        else:
            np.testing.assert_allclose(vec(b["centroid"]), vec(a["centroid"]), atol=1e-6)


def test_stable_run_reports_zero_changed_owners(tmp_path, caplog):
    db = str(tmp_path / "stable.db")
    with stubbed_decoder():
        label_of = build_library(db)
        run_phase(db, label_of)
        with caplog.at_level(logging.INFO, logger="facet.face_cluster"):
            run_phase(db, label_of)
    assert any("Face assignment: 0 of" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# load_embeddings: id order
# ---------------------------------------------------------------------------

def test_load_embeddings_returns_ids_ascending_despite_a_covering_index(tmp_path):
    """Cluster discovery order -- the ids of new persons and the float summation
    order of their centroids -- follows load order. A covering index over
    (person_id, embedding_model, embedding) would otherwise serve the scan in
    person order."""
    db = str(tmp_path / "order.db")
    label_of = build_library(db)
    with _db(db) as conn:
        conn.execute("CREATE INDEX test_faces_cover ON faces(person_id, embedding_model, embedding)")
        # make person order and id order disagree for the model-tagged rows
        conn.execute("UPDATE faces SET person_id = 100 - (id % 7) WHERE embedding_model IS NOT NULL")
        conn.commit()
    face_ids, embeddings = make_clusterer(db).load_embeddings()
    assert face_ids == sorted(face_ids)
    assert len(face_ids) == len([f for f in label_of]) and embeddings.shape == (len(face_ids), DIM)


def test_load_embeddings_plan_has_no_temp_btree(tmp_path, monkeypatch):
    db = str(tmp_path / "plan.db")
    build_library(db)
    seen = []

    from faces import clusterer as clusterer_module
    real = clusterer_module.get_connection

    @contextlib.contextmanager
    def tracing(path, row_factory=True):
        with real(path, row_factory=row_factory) as conn:
            conn.set_trace_callback(seen.append)
            yield conn

    monkeypatch.setattr(clusterer_module, "get_connection", tracing)
    make_clusterer(db).load_embeddings()
    load_sql = [s for s in seen if "FROM faces" in s and "embedding IS NOT NULL" in s][0]
    assert "ORDER BY id" in load_sql

    def details(sql):
        with _db(db) as conn:
            return " | ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql))

    assert "TEMP B-TREE" not in details(load_sql), details(load_sql)
    # control: an ordering the table does not provide DOES need a sort, so the
    # assertion above can fail
    assert "TEMP B-TREE" in details(
        "SELECT id, embedding FROM faces WHERE embedding IS NOT NULL ORDER BY confidence")


# ---------------------------------------------------------------------------
# recompute_person_centroid: id-ordered mean
# ---------------------------------------------------------------------------

def test_recompute_person_centroid_is_the_id_ordered_mean(tmp_path):
    db = str(tmp_path / "centroid.db")
    init_database(db)
    rng = np.random.default_rng(7)
    vectors = (rng.normal(size=(60, DIM)) * rng.uniform(0.5, 3.0, size=(60, 1))).astype(np.float32)
    with _db(db) as conn:
        conn.execute("INSERT INTO persons(id, name) VALUES (1, 'P')")
        for i, vec in enumerate(vectors):
            path = f"/synthetic/c{i}.jpg"
            conn.execute("INSERT INTO photos(path) VALUES (?)", (path,))
            conn.execute(
                "INSERT INTO faces(photo_path, face_index, embedding, person_id, embedding_model) "
                "VALUES (?, 0, ?, 1, ?)", (path, vec.tobytes(), ARCFACE_MODEL))
        # an index that would serve the read in embedding-byte order if no ORDER BY pinned it
        conn.execute("CREATE INDEX test_faces_cover ON faces(person_id, embedding_model, embedding)")
        conn.commit()

        recompute_person_centroid(conn, 1, ARCFACE_MODEL)
        stored = conn.execute("SELECT centroid, centroid_model FROM persons WHERE id = 1").fetchone()

    normalised = [v / (np.linalg.norm(v) + 1e-10) for v in vectors]      # id order == insertion order
    expected = np.mean(normalised, axis=0).astype(np.float32)
    expected = expected / (np.linalg.norm(expected) + 1e-10)
    assert stored[1] == ARCFACE_MODEL
    assert stored[0] == expected.tobytes()


# ---------------------------------------------------------------------------
# Atomicity: an exception inside the phase changes nothing
# ---------------------------------------------------------------------------

ATOMIC_CASES = [
    ({}, "_update_person_centroids"),
    ({"preserve_named_only": True}, "_update_person_centroids"),
    ({}, "_apply_owner_changes"),
    ({"preserve_named_only": True}, "_apply_owner_changes"),
    ({"force": True}, "_apply_owner_changes"),   # force keeps no preserved person to recompute
]


@pytest.mark.parametrize("mode, fails_at", ATOMIC_CASES,
                         ids=[f"{sorted(m) or 'incremental'}-{h}" for m, h in ATOMIC_CASES])
def test_an_exception_inside_the_phase_rolls_everything_back(tmp_path, mode, fails_at):
    """One transaction: person inserts, deletes and the owner write all vanish together."""
    db = str(tmp_path / "atomic.db")
    label_of = build_library(db)
    before = dump_state(db)
    real = getattr(FaceClusterer, fails_at)

    def boom(self, conn, arg):
        real(self, conn, arg)             # the write has been issued, uncommitted
        raise RuntimeError("interrupted")

    with stubbed_decoder(), mock.patch.object(FaceClusterer, fails_at, boom):
        with pytest.raises(RuntimeError, match="interrupted"):
            run_phase(db, label_of, **mode)
    assert dump_state(db) == before


# ---------------------------------------------------------------------------
# G4: no photo-file decode while a write transaction is open
# ---------------------------------------------------------------------------

def test_legacy_thumbnails_are_decoded_with_no_transaction_open(tmp_path):
    """Twenty-five new people with no stored thumbnail: past 20 the thumbnail loop
    batches its commits, and a decode between a batch's first UPDATE and its commit
    would hold the write lock across the read of an original (RAW) file."""
    db = str(tmp_path / "legacy.db")
    init_database(db)
    label_of = {}
    n_clusters, per = 25, 3
    with _db(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        n = 0
        for c in range(n_clusters):
            for vec in _points(_axis(800 + c), per, 900 + c):
                _add_face(conn, label_of, n, vec, ARCFACE_MODEL, None, None, c, 0.9)
                n += 1
        conn.commit()

    calls = []

    def stub(self, conn, face_id):
        calls.append((face_id, conn.in_transaction))
        return b"legacy-thumb-%d" % face_id

    with stubbed_decoder(stub):
        run_phase(db, label_of)

    assert len(calls) == n_clusters, "one decode per new person, so the assertion below is not vacuous"
    assert not any(in_tx for _, in_tx in calls), calls
    with _db(db) as conn:
        thumbs = conn.execute("SELECT face_thumbnail, representative_face_id FROM persons").fetchall()
    assert len(thumbs) == n_clusters
    assert all(t == b"legacy-thumb-%d" % rep for t, rep in thumbs)


# ---------------------------------------------------------------------------
# G5: compare-and-set owner write
# ---------------------------------------------------------------------------

def _three_face_db(tmp_path):
    db = str(tmp_path / "cas.db")
    init_database(db)
    with _db(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for pid in (1, 2, 3):
            conn.execute("INSERT INTO persons(id, name) VALUES (?, ?)", (pid, f"p{pid}"))
        ids = [_add_face(conn, {}, i, _axis(i), ARCFACE_MODEL, owner, b"t", None, 0.9)
               for i, owner in enumerate((None, 1, 2))]
        conn.commit()
    return db, ids


def test_apply_owner_changes_does_not_overwrite_a_concurrent_change(tmp_path, caplog):
    db, (f_null, f_one, f_two) = _three_face_db(tmp_path)
    clusterer = make_clusterer(db)
    changes = [(3, f_null, None), (3, f_one, 1), (3, f_two, 2)]
    # the viewer reassigns f_one between this run's read and its write
    with _db(db) as other:
        other.execute("UPDATE faces SET person_id = 2 WHERE id = ?", (f_one,))
        other.commit()
    with _db(db) as conn, caplog.at_level(logging.WARNING, logger="facet.face_cluster"):
        conflicts = clusterer._apply_owner_changes(conn, changes)
        conn.commit()
    with _db(db) as conn:
        owners = dict(conn.execute("SELECT id, person_id FROM faces"))
    assert conflicts == 1
    assert owners == {f_null: 3, f_one: 2, f_two: 3}
    assert any("1" in r.getMessage() and "conflict" in r.getMessage().lower()
               for r in caplog.records if r.levelno >= logging.WARNING), \
        [r.getMessage() for r in caplog.records]


def test_apply_owner_changes_matches_a_null_old_owner(tmp_path):
    db, (f_null, _, _) = _three_face_db(tmp_path)
    with _db(db) as conn:
        assert make_clusterer(db)._apply_owner_changes(conn, [(3, f_null, None)]) == 0
        conn.commit()
        assert conn.execute("SELECT person_id FROM faces WHERE id = ?", (f_null,)).fetchone()[0] == 3


def test_apply_owner_changes_with_nothing_to_write(tmp_path):
    db, _ = _three_face_db(tmp_path)
    with _db(db) as conn:
        assert make_clusterer(db)._apply_owner_changes(conn, []) == 0
        assert not conn.in_transaction


def test_a_viewer_change_between_the_read_and_the_write_survives_the_phase(tmp_path, caplog):
    """End to end. The stable library has no other write before the final
    compare-and-set, so a second connection can commit in front of it."""
    def stable_library(name):
        path = str(tmp_path / name)
        label_of = build_library(path)
        run_phase(path, label_of)
        return path, label_of

    with stubbed_decoder():
        db, label_of = stable_library("viewer.db")
        oracle_db, oracle_labels = stable_library("oracle.db")

        # exactly one face changes owner: an unowned noise face joins blob 0 (Alice's)
        with _db(db) as conn:
            target = conn.execute(
                "SELECT id FROM faces WHERE person_id IS NULL AND embedding_model = ? "
                "AND id IN (%s) ORDER BY id LIMIT 1" % ",".join(
                    str(f) for f, lab in label_of.items() if lab == NOISE_LABEL),
                (ARCFACE_MODEL,)).fetchone()[0]
        label_of[target] = 0
        oracle_labels[target] = 0

        real_apply = FaceClusterer._apply_owner_changes

        def viewer_gets_in_first(self, conn, changes):
            assert [c[1] for c in changes] == [target], changes
            with _db(db) as other:
                other.execute("UPDATE faces SET person_id = 2 WHERE id = ?", (target,))
                other.commit()
            return real_apply(self, conn, changes)

        with mock.patch.object(FaceClusterer, "_apply_owner_changes", viewer_gets_in_first), \
                caplog.at_level(logging.WARNING, logger="facet.face_cluster"):
            run_phase(db, label_of)
        run_phase(oracle_db, oracle_labels)

    mine = dict(dump_state(db)["faces"])
    oracle = dict(dump_state(oracle_db)["faces"])
    assert oracle[target] == 1, "premise: undisturbed, the face joins Alice"
    assert mine[target] == 2, "the viewer's change must survive"
    assert {k: v for k, v in mine.items() if k != target} == \
           {k: v for k, v in oracle.items() if k != target}
    assert any("conflict" in r.getMessage().lower() for r in caplog.records)
