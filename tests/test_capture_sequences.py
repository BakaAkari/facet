"""Capture sequence integration: production schema, atomic metadata and guards."""
import sqlite3
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from db.schema import init_database
from utils.capture_sequences import (
    detect_capture_sequences, persist_capture_metadata,
    protected_paths, require_unprotected_files,
)


@pytest.fixture
def library(tmp_path):
    path = str(tmp_path / 'capture.db')
    init_database(path)
    return path


def seed(path, rows):
    with sqlite3.connect(path) as conn:
        for row in rows:
            keys = list(row)
            conn.execute(f"INSERT INTO photos ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                         [row[k] for k in keys])


def test_partial_fuji_protected_during_save(library):
    seed(library, [{'path': '/partial.raf', 'burst_group_id': 3}])
    with sqlite3.connect(library) as conn:
        persist_capture_metadata(conn, {'path': '/partial.raf', 'pixel_shift_shots': 16})
        assert protected_paths(conn, ['/partial.raf']) == {'/partial.raf'}
        assert conn.execute('SELECT burst_group_id FROM photos').fetchone()[0] is None
        with pytest.raises(HTTPException) as error:
            require_unprotected_files(conn, ['/partial.raf'])
        assert error.value.status_code == 409


def test_sony_grouping_and_disabled_value(library):
    seed(library, [dict(path=f'/sony/{i}.arw', camera_model='ILCE-7RM3A',
                        date_taken='2026:01:01 00:00:00',
                        pixel_shift_info=f'01020304 {i} 4 0x0') for i in (1, 3)] +
         [dict(path='/sony/normal.arw', pixel_shift_info='00000000 0 0 0x0')])
    detect_capture_sequences(library)
    with sqlite3.connect(library) as conn:
        rows = conn.execute('SELECT sequence_kind, sequence_group_id FROM photos WHERE path != ?',
                            ('/sony/normal.arw',)).fetchall()
        assert rows[0] == rows[1]
        assert rows[0][0] == 'pixel_shift'
        assert conn.execute('SELECT sequence_kind FROM photos WHERE path=?',
                            ('/sony/normal.arw',)).fetchone()[0] is None


def test_timelapse_dry_run_and_idempotence(library):
    seed(library, [dict(path=f'/time/{i:03}.raf', camera_model='GFX', camera_serial='body',
                        focal_length=50, phash='abcdef', burst_group_id=5,
                        date_taken=(datetime(2026, 1, 1)+timedelta(seconds=2*i)).isoformat())
                   for i in range(16)])
    report = detect_capture_sequences(library, dry_run=True)
    assert len(report['groups']) == 1
    with sqlite3.connect(library) as conn:
        assert conn.execute('SELECT COUNT(*) FROM photos WHERE sequence_kind IS NOT NULL').fetchone()[0] == 0
    first = detect_capture_sequences(library)
    assert detect_capture_sequences(library) == first
    with sqlite3.connect(library) as conn:
        assert conn.execute("SELECT COUNT(*) FROM photos WHERE sequence_kind='timelapse' AND burst_group_id IS NULL").fetchone()[0] == 16


def test_companion_guard_and_ordinary_photo(library):
    seed(library, [{'path': '/photo.jpg'}, {'path': '/photo.raf', 'sequence_kind': 'pixel_shift'}])
    with sqlite3.connect(library) as conn:
        require_unprotected_files(conn, ['/photo.jpg'])
        with pytest.raises(HTTPException):
            require_unprotected_files(conn, ['/photo.jpg', '/photo.raf'])
