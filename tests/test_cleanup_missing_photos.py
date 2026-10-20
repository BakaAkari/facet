"""cleanup_missing_photos must distinguish genuine deletion from inaccessible
paths (F33): a permission-denied directory or an unmounted/absent scan root
must be preserved, never cascade-deleted, unless --force is given.
"""

import os
import sqlite3

import pytest

from db.maintenance import cleanup_missing_photos, delete_photo_rows
from db.schema import init_database
from db.stats_cache import PERCENTILES_SNAPSHOT_KEY

_IS_ROOT = hasattr(os, 'geteuid') and os.geteuid() == 0


def _make_db(path, photo_paths):
    init_database(path)
    conn = sqlite3.connect(path)
    for p in photo_paths:
        conn.execute(
            "INSERT INTO photos (path, filename) VALUES (?, ?)", (p, p.rsplit('/', 1)[-1])
        )
    conn.commit()
    conn.close()


def _paths_in_db(path):
    conn = sqlite3.connect(path)
    rows = {r[0] for r in conn.execute("SELECT path FROM photos").fetchall()}
    conn.close()
    return rows


@pytest.mark.skipif(_IS_ROOT, reason="root bypasses directory permissions, so a chmod-000 dir is still readable")
def test_inaccessible_dir_preserved_but_deleted_file_removed(tmp_path):
    present = tmp_path / 'present.jpg'
    present.write_bytes(b'x')

    deleted = tmp_path / 'deleted.jpg'  # created then removed → truly gone

    locked_dir = tmp_path / 'locked'
    locked_dir.mkdir()
    locked = locked_dir / 'locked.jpg'
    locked.write_bytes(b'x')

    ghost = tmp_path / 'ghost_root' / 'sub' / 'ghost.jpg'  # root never created

    db = str(tmp_path / 'scores.db')
    _make_db(db, [str(present), str(deleted), str(locked), str(ghost)])

    os.chmod(locked_dir, 0o000)
    try:
        removed = cleanup_missing_photos(db, dry_run=False, force=False, verbose=False)
    finally:
        os.chmod(locked_dir, 0o700)

    remaining = _paths_in_db(db)
    assert removed == 1
    assert str(deleted) not in remaining
    assert str(present) in remaining
    assert str(locked) in remaining
    assert str(ghost) in remaining


def test_unmounted_root_preserves_everything_without_force(tmp_path):
    root = tmp_path / 'mnt' / 'share'  # never created → simulates an unmounted volume
    photo_paths = [str(root / f'{i}.jpg') for i in range(3)]
    db = str(tmp_path / 'scores.db')
    _make_db(db, photo_paths)

    removed = cleanup_missing_photos(db, dry_run=False, force=False, verbose=False)

    assert removed == 0
    assert _paths_in_db(db) == set(photo_paths)


def test_force_removes_inaccessible_paths(tmp_path):
    root = tmp_path / 'mnt' / 'share'
    photo_paths = [str(root / f'{i}.jpg') for i in range(3)]
    db = str(tmp_path / 'scores.db')
    _make_db(db, photo_paths)

    removed = cleanup_missing_photos(db, dry_run=False, force=True, verbose=False)

    assert removed == 3
    assert _paths_in_db(db) == set()


def test_client_picks_removed_for_deleted_photo(tmp_path):
    present = tmp_path / 'present.jpg'
    present.write_bytes(b'x')
    deleted = tmp_path / 'deleted.jpg'
    db = str(tmp_path / 'scores.db')
    _make_db(db, [str(present), str(deleted)])

    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO albums (name) VALUES ('proof')")
    album_id = conn.execute("SELECT id FROM albums").fetchone()[0]
    conn.executemany(
        "INSERT INTO album_client_picks (album_id, photo_path) VALUES (?, ?)",
        [(album_id, str(present)), (album_id, str(deleted))],
    )
    conn.commit()
    conn.close()

    removed = cleanup_missing_photos(db, dry_run=False, force=False, verbose=False)

    conn = sqlite3.connect(db)
    remaining_picks = {r[0] for r in conn.execute("SELECT photo_path FROM album_client_picks").fetchall()}
    conn.close()
    assert removed == 1
    assert str(deleted) not in remaining_picks
    assert str(present) in remaining_picks


def test_delete_photo_rows_refreshes_face_count_without_its_own_commit(tmp_path):
    """`delete_photo_rows` is the row+cascade portion `cleanup_missing_photos`
    extracted so the new `POST /api/photo/delete` endpoint can run it on a
    connection THE REQUEST already opened and control its own commit timing
    (B1: the sequence-lead re-pick and the row delete must land in one
    transaction). It must not secretly depend on `cleanup_missing_photos`'s
    own `conn.commit()` to take effect -- the persons.face_count refresh has
    to be visible on the SAME uncommitted connection immediately."""
    path = str(tmp_path / 'linked.jpg')
    db = str(tmp_path / 'scores.db')
    init_database(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("INSERT INTO photos (path, filename) VALUES (?, ?)", (path, 'linked.jpg'))
    conn.execute("INSERT INTO persons (name, face_count) VALUES ('Someone', 1)")
    person_id = conn.execute("SELECT id FROM persons").fetchone()[0]
    conn.execute(
        "INSERT INTO faces (photo_path, face_index, embedding, person_id) VALUES (?, 0, ?, ?)",
        (path, b'x', person_id),
    )
    conn.commit()

    result = delete_photo_rows(conn, [path])

    assert result == {"deleted": 1, "emptied_persons": 1}
    # No conn.commit() was called by delete_photo_rows itself, yet the face
    # cascade and the persons.face_count refresh must already be visible on
    # THIS SAME connection -- proving the helper does its own work rather
    # than depending on cleanup_missing_photos's commit to take effect.
    face_count = conn.execute("SELECT face_count FROM persons WHERE id = ?", (person_id,)).fetchone()[0]
    assert face_count == 0
    remaining_faces = conn.execute("SELECT COUNT(*) FROM faces WHERE photo_path = ?", (path,)).fetchone()[0]
    assert remaining_faces == 0

    # The "without its own commit" half of the claim above cannot be proven
    # by reading back through the SAME connection: every read there uses the
    # writing connection, which sees its own uncommitted writes regardless
    # of whether delete_photo_rows ever committed -- so that assertion alone
    # passes whether or not the fix under test is in place. A SECOND
    # connection is SQLite's actual test for "not yet committed": it must
    # still see the pre-delete state.
    other = sqlite3.connect(db)
    try:
        other_photo_count = other.execute(
            "SELECT COUNT(*) FROM photos WHERE path = ?", (path,)).fetchone()[0]
        other_face_count = other.execute(
            "SELECT face_count FROM persons WHERE id = ?", (person_id,)).fetchone()[0]
    finally:
        other.close()
    assert other_photo_count == 1  # row still visible elsewhere -- not committed yet
    assert other_face_count == 1  # pre-delete face_count still visible elsewhere

    conn.commit()
    conn.close()
    assert path not in _paths_in_db(db)


def _seed_stats_cache(db):
    conn = sqlite3.connect(db)
    conn.executemany(
        "INSERT INTO stats_cache (key, value, updated_at) VALUES (?, ?, 0)",
        [(PERCENTILES_SNAPSHOT_KEY, '{"percentiles": {"raw_sharpness_variance": 100.0}}'),
         ('total_photos', '2'),
         ('auto_retrain_pending:global', '3')],
    )
    conn.commit()
    conn.close()


def _stats_cache_keys(db):
    conn = sqlite3.connect(db)
    # A NULL value is an eviction tombstone (invalidate_metric_ranges), not a kept entry.
    keys = {r[0] for r in conn.execute("SELECT key FROM stats_cache WHERE value IS NOT NULL").fetchall()}
    conn.close()
    return keys


@pytest.mark.parametrize('preserve_counters', [True, False])
def test_delete_photo_rows_keeps_the_percentile_snapshot(tmp_path, preserve_counters):
    """The snapshot is the distribution the last recompute normalized every stored
    aggregate with; deleting a photo invalidates the derived counts, not that."""
    db = str(tmp_path / 'scores.db')
    _make_db(db, ['/lib/a.jpg', '/lib/b.jpg'])
    _seed_stats_cache(db)

    conn = sqlite3.connect(db)
    delete_photo_rows(conn, ['/lib/b.jpg'], preserve_auto_retrain_counters=preserve_counters)
    conn.commit()
    conn.close()

    expected = {PERCENTILES_SNAPSHOT_KEY}
    if preserve_counters:
        expected.add('auto_retrain_pending:global')
    assert _stats_cache_keys(db) == expected


def test_cleanup_missing_photos_keeps_the_percentile_snapshot(tmp_path):
    present = tmp_path / 'present.jpg'
    present.write_bytes(b'x')
    db = str(tmp_path / 'scores.db')
    _make_db(db, [str(present), str(tmp_path / 'deleted.jpg')])
    _seed_stats_cache(db)

    assert cleanup_missing_photos(db, dry_run=False, force=False, verbose=False) == 1

    assert _stats_cache_keys(db) == {PERCENTILES_SNAPSHOT_KEY}


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
