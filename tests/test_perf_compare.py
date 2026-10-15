"""Tests for scripts/perf/compare.py, the cell-exact comparator behind the scan-pass performance gates.

CPU only. Databases come from ``db.schema.init_database`` (never a hand-rolled ``CREATE TABLE photos``),
except for the storage-class test: SQLite's column affinity converts an integer written to a REAL
column (and an integral REAL written to an INTEGER one), so the same numeric value can only be stored
as two storage classes in a column without affinity -- that one test builds minimal untyped tables.
"""
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from db.schema import FACES_COLUMNS, PHOTOS_COLUMNS, SCAN_FAILURES_COLUMNS, init_database

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'scripts' / 'perf' / 'compare.py'

_spec = importlib.util.spec_from_file_location('perf_compare', SCRIPT)
compare = importlib.util.module_from_spec(_spec)
sys.modules['perf_compare'] = compare
_spec.loader.exec_module(compare)

PATHS = ['/p/1.jpg', '/p/2.jpg', '/p/3.jpg']


# ----------------------------------------------------------------------------- helpers
def _value(typedef, variant):
    """A schema-valid value for a column: variant 0 is the baseline's, 1 is the patched one's."""
    kind = typedef.split()[0]
    return {'TEXT': ('a', 'b'), 'REAL': (0.25, 0.5), 'INTEGER': (0, 1),
            'BLOB': (b'\x01\x02', b'\x01\x03')}[kind][variant]


def _backup(src, dst):
    s = sqlite3.connect(str(src))
    d = sqlite3.connect(str(dst))
    s.backup(d)
    d.close()
    s.close()
    return dst


def _insert(conn, table, columns, values):
    names = [c for c, _ in columns if c in values]
    conn.execute('INSERT INTO %s (%s) VALUES (%s)' % (table, ', '.join(names), ', '.join('?' * len(names))),
                 [values[n] for n in names])


@pytest.fixture
def pair(tmp_path):
    """(A, B): a schema-initialised baseline with 3 photos / 2 faces / 2 tags / 1 failure, and its backup copy."""
    a = tmp_path / 'a.db'
    init_database(str(a))
    conn = sqlite3.connect(str(a))
    for p in PATHS:
        vals = {c: _value(t, 0) for c, t in PHOTOS_COLUMNS}
        vals['path'] = p
        _insert(conn, 'photos', PHOTOS_COLUMNS, vals)
    for idx in (0, 1):
        vals = {c: _value(t, 0) for c, t in FACES_COLUMNS if c != 'id'}
        vals.update(photo_path=PATHS[0], face_index=idx)
        _insert(conn, 'faces', FACES_COLUMNS, vals)
    conn.executemany('INSERT INTO photo_tags (photo_path, tag) VALUES (?, ?)',
                     [(PATHS[0], 'dog'), (PATHS[1], 'beach')])
    conn.execute("INSERT INTO scan_runs (status) VALUES ('completed')")
    conn.execute('INSERT INTO scan_failures (scan_run_id, path, stage, error) VALUES (1, ?, ?, ?)',
                 (PATHS[2], 'load', 'boom at 0x7f00aa11'))
    conn.commit()
    conn.close()
    return a, _backup(a, tmp_path / 'b.db')


def _cmp(a, b, **kw):
    lines = []
    code = compare.run(str(a), str(b), out=lines.append, **kw)
    return code, '\n'.join(lines)


def _exec(db, sql, params=()):
    conn = sqlite3.connect(str(db))
    conn.execute(sql, params)
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------- identical / values
def test_identical_databases_have_no_difference(pair):
    code, out = _cmp(*pair)
    assert code == 0, out
    assert 'differing cells/rows: 0' in out
    assert 'cells_compared:' in out and 'cells_compared: 0' not in out
    assert 'photo_tags' in out and 'rows A=2' in out


def test_run_directories_are_accepted(pair, tmp_path):
    a, b = pair
    for name, db in (('ra', a), ('rb', b)):
        (tmp_path / name).mkdir()
        _backup(db, tmp_path / name / 'db.sqlite')
    code, out = _cmp(tmp_path / 'ra', tmp_path / 'rb', profile_name='16gb')
    assert code == 0, out
    assert '[profile 16gb]' in out


def test_flipped_thumbnail_byte_is_found(pair):
    a, b = pair
    _exec(b, 'UPDATE photos SET thumbnail = ? WHERE path = ?', (b'\x01\x03', PATHS[1]))
    code, out = _cmp(a, b)
    assert code == 1
    assert 'DIFF photos.thumbnail' in out and PATHS[1] in out


def test_one_ulp_class_change_in_a_non_tolerance_real_fails(pair):
    a, b = pair
    _exec(a, 'UPDATE photos SET f_stop = 2.5')
    _exec(b, 'UPDATE photos SET f_stop = ?', (2.5 + 1e-15,))
    assert 2.5 + 1e-15 != 2.5
    code, out = _cmp(a, b)
    assert code == 1
    assert 'DIFF photos.f_stop' in out


@pytest.mark.parametrize('column', ['aggregate', 'raw_sharpness_variance', 'isolation_bonus'])
def test_tolerance_columns_accept_1e15_and_refuse_1e9(pair, column):
    a, b = pair
    _exec(a, 'UPDATE photos SET %s = 5.0' % column)
    close, far = 5.0 * (1 + 1e-15), 5.0 * (1 + 1e-9)
    assert close != 5.0
    _exec(b, 'UPDATE photos SET %s = ?' % column, (close,))
    code, out = _cmp(a, b)
    assert code == 0, out
    assert column in out and 'tolerance' in out
    _exec(b, 'UPDATE photos SET %s = ?' % column, (far,))
    code, out = _cmp(a, b)
    assert code == 1
    assert 'DIFF photos.%s' % column in out


def test_tolerance_has_an_absolute_floor_when_the_baseline_is_zero(pair):
    a, b = pair
    _exec(a, 'UPDATE photos SET aggregate = 0.0')
    _exec(b, 'UPDATE photos SET aggregate = 5e-13')
    assert _cmp(a, b)[0] == 0
    _exec(b, 'UPDATE photos SET aggregate = 5e-11')
    assert _cmp(a, b)[0] == 1


def test_floats_are_compared_by_bit_pattern():
    """SQLite itself stores -0.0 as 0.0, so this cannot be shown through a database: the cell key is checked."""
    import struct
    assert compare.canon('real', -0.0) != compare.canon('real', 0.0)
    nan_a = struct.unpack('<d', struct.pack('<Q', 0x7ff8000000000000))[0]
    nan_b = struct.unpack('<d', struct.pack('<Q', 0x7ff8000000000001))[0]
    assert compare.canon('real', nan_a) != compare.canon('real', nan_b)
    assert compare.canon('real', 1.5) == compare.canon('real', 1.5)
    assert compare.canon('integer', 3) != compare.canon('real', 3.0)
    # inside the tolerance list a non-finite value is never "close"
    assert not compare.within_tolerance('aggregate', compare.canon('real', nan_a), compare.canon('real', nan_a * 1))
    assert compare.within_tolerance('aggregate', compare.canon('real', 5.0), compare.canon('real', 5.0 * (1 + 1e-15)))
    assert not compare.within_tolerance('f_stop', compare.canon('real', 5.0), compare.canon('real', 5.0 * (1 + 1e-15)))


def test_tolerance_never_hides_a_null(pair):
    a, b = pair
    _exec(b, 'UPDATE photos SET aggregate = NULL WHERE path = ?', (PATHS[0],))
    assert _cmp(a, b)[0] == 1


def test_storage_class_change_fails_even_when_the_numeric_value_matches(tmp_path):
    """3 (INTEGER) vs 3.0 (REAL): equal in Python, different in SQLite. Untyped columns keep both."""
    def mini(path, val):
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE photos (path TEXT PRIMARY KEY, val);
            CREATE TABLE faces (photo_path TEXT, face_index INTEGER, val);
            CREATE TABLE photo_tags (photo_path TEXT, tag TEXT);
            CREATE TABLE scan_failures (scan_run_id INTEGER, path TEXT, stage TEXT, error TEXT, timestamp TEXT);
        """)
        conn.execute('INSERT INTO photos VALUES (?, ?)', ('/p/1.jpg', val))
        conn.commit()
        conn.close()
        return path
    a, b = mini(tmp_path / 'a.db', 3), mini(tmp_path / 'b.db', 3.0)
    code, out = _cmp(a, b)
    assert code == 1, out
    assert 'DIFF photos.val' in out and 'integer' in out and 'real' in out
    assert _cmp(a, mini(tmp_path / 'c.db', 3))[0] == 0


# ----------------------------------------------------------------------------- tables
def test_empty_photos_is_an_invalid_comparison(tmp_path):
    a, b = tmp_path / 'a.db', tmp_path / 'b.db'
    init_database(str(a))
    init_database(str(b))
    code, out = _cmp(a, b)
    assert code == 2
    assert 'photos is empty' in out


def test_photos_key_sets_must_match(pair):
    a, b = pair
    _exec(b, 'DELETE FROM photos WHERE path = ?', (PATHS[2],))
    code, out = _cmp(a, b)
    assert code == 2 and 'key sets differ' in out


def test_faces_lost_by_the_patched_run_is_a_regression(pair):
    a, b = pair
    _exec(b, 'DELETE FROM faces')
    code, out = _cmp(a, b)
    assert code == 1
    assert 'DIFF faces.<row>' in out


def test_face_cell_difference_is_found(pair):
    a, b = pair
    _exec(b, 'UPDATE faces SET confidence = 0.75 WHERE face_index = 1')
    code, out = _cmp(a, b)
    assert code == 1 and 'DIFF faces.confidence' in out


def test_faces_autoincrement_id_is_not_compared(pair):
    a, b = pair
    _exec(b, 'UPDATE faces SET id = id + 100')
    assert _cmp(a, b)[0] == 0


def test_zero_face_baseline_is_invalid_when_faces_are_required(pair):
    a, b = pair
    _exec(a, 'DELETE FROM faces')
    _exec(b, 'DELETE FROM faces')
    code, out = _cmp(a, b, require_faces=True)
    assert code == 2 and '0 faces' in out
    code, out = _cmp(a, b)
    assert code == 0
    assert 'faces: 0 rows in both' in out and 'UNTESTED' in out


def test_photo_tags_empty_in_both_is_expected_for_a_scan(pair):
    a, b = pair
    for db in (a, b):
        _exec(db, 'DELETE FROM photo_tags')
    code, out = _cmp(a, b)
    assert code == 0, out
    assert 'photo_tags: 0 rows in both, expected-empty for a scan' in out
    assert 'a scan never writes it' in out


def test_photo_tags_present_in_one_arm_only_fails_both_ways(pair):
    a, b = pair
    _exec(a, 'DELETE FROM photo_tags')
    assert _cmp(a, b)[0] == 1          # baseline empty, patched has rows
    assert _cmp(b, a)[0] == 1          # baseline has rows, patched empty


def test_photo_tags_set_difference_fails(pair):
    a, b = pair
    _exec(b, "UPDATE photo_tags SET tag = 'cat' WHERE tag = 'dog'")
    code, out = _cmp(a, b)
    assert code == 1 and 'photo_tags' in out


def test_scan_failures_row_in_one_arm_fails(pair):
    a, b = pair
    _exec(b, 'DELETE FROM scan_failures')
    assert _cmp(a, b)[0] == 1
    assert _cmp(b, a)[0] == 1


def test_scan_failures_changed_stage_fails(pair):
    a, b = pair
    _exec(b, "UPDATE scan_failures SET stage = 'composition'")
    code, out = _cmp(a, b)
    assert code == 1 and 'scan_failures' in out


def test_scan_failures_address_in_the_message_is_normalised_but_text_is_not(pair):
    a, b = pair
    _exec(b, "UPDATE scan_failures SET error = 'boom at 0x1234ffee'")
    assert _cmp(a, b)[0] == 0
    _exec(b, "UPDATE scan_failures SET error = 'bang at 0x1234ffee'")
    assert _cmp(a, b)[0] == 1


def test_scan_failures_run_id_and_timestamp_are_not_compared(pair):
    a, b = pair
    _exec(b, "UPDATE scan_failures SET scan_run_id = 9, timestamp = '1999-01-01'")
    assert _cmp(a, b)[0] == 0


def test_missing_table_is_invalid(pair):
    a, b = pair
    _exec(b, 'DROP TABLE scan_failures')
    code, out = _cmp(a, b)
    assert code == 2 and 'scan_failures' in out


def test_column_set_mismatch_is_invalid(pair):
    a, b = pair
    _exec(b, 'ALTER TABLE photos ADD COLUMN surprise TEXT')
    code, out = _cmp(a, b)
    assert code == 2 and 'surprise' in out


def test_missing_database_file_is_invalid(pair, tmp_path):
    code, out = _cmp(pair[0], tmp_path / 'nope.db')
    assert code == 2 and 'does not exist' in out


# ----------------------------------------------------------------------------- columns compared by default
DOCUMENTED_EXCLUDED = {
    'photos': {'scanned_at'},
    'faces': {'id'},
    'scan_failures': {'scan_run_id', 'timestamp'},
    'photo_tags': set(),
}


def test_excluded_set_equals_the_documented_list():
    assert {t: set(c) for t, c in compare.EXCLUDED.items()} == DOCUMENTED_EXCLUDED
    assert all(reason.strip() for cols in compare.EXCLUDED.values() for reason in cols.values())
    assert set(compare.TOLERANCE) == {'raw_sharpness_variance', 'isolation_bonus', 'aggregate'}
    assert set(compare.TOLERANCE.values()) == {1e-12}


def test_excluded_and_tolerance_names_exist_in_the_schema():
    photos = dict(PHOTOS_COLUMNS)
    assert set(compare.EXCLUDED['photos']) <= set(photos)
    assert set(compare.EXCLUDED['faces']) <= {c for c, _ in FACES_COLUMNS}
    assert set(compare.EXCLUDED['scan_failures']) <= {c for c, _ in SCAN_FAILURES_COLUMNS}
    assert all(photos[c].startswith('REAL') for c in compare.TOLERANCE)


@pytest.mark.parametrize('column', [c for c, _ in PHOTOS_COLUMNS if c != 'path'])
def test_every_photos_column_is_compared_unless_excluded(pair, column):
    a, b = pair
    typedef = dict(PHOTOS_COLUMNS)[column]
    _exec(b, 'UPDATE photos SET %s = ? WHERE path = ?' % column, (_value(typedef, 1), PATHS[1]))
    code, out = _cmp(a, b)
    if column in compare.EXCLUDED['photos']:
        assert code == 0, out
    else:
        assert code == 1, '%s was not compared' % column
        assert 'DIFF photos.%s ' % column in out


@pytest.mark.parametrize('column', [c for c, _ in FACES_COLUMNS if c not in ('id', 'photo_path', 'face_index')])
def test_every_faces_column_is_compared(pair, column):
    a, b = pair
    typedef = dict(FACES_COLUMNS)[column]
    _exec(b, 'UPDATE faces SET %s = ? WHERE face_index = 0' % column, (_value(typedef, 1),))
    code, out = _cmp(a, b)
    assert code == 1, '%s was not compared' % column
    assert 'DIFF faces.%s ' % column in out


@pytest.mark.parametrize('column', [c for c, _ in SCAN_FAILURES_COLUMNS
                                    if c not in compare.EXCLUDED['scan_failures']])
def test_every_scan_failures_column_is_compared(pair, column):
    a, b = pair
    _exec(b, "UPDATE scan_failures SET %s = 'changed'" % column)
    assert _cmp(a, b)[0] == 1


# ----------------------------------------------------------------------------- ledgers
def _run_dir(root, name, db, ledger):
    d = root / name
    d.mkdir()
    _backup(db, d / 'db.sqlite')
    if ledger is not None:
        (d / 'batch_sizes.json').write_text(json.dumps(ledger))
    return d


LEDGER = {'chunks': {'0': [['saliency', 8, None], ['topiq', 16, None]], '1': [['saliency', 8, None]]}}


def test_matching_ledgers_pass_and_are_reported(pair, tmp_path):
    a, b = pair
    da, db_ = _run_dir(tmp_path, 'ra', a, LEDGER), _run_dir(tmp_path, 'rb', b, LEDGER)
    code, out = _cmp(da, db_)
    assert code == 0, out
    assert 'sub-batch sequences identical (3 entries over 2 chunks)' in out


def test_different_ledger_sequences_fail_even_when_every_cell_matches(pair, tmp_path):
    a, b = pair
    other = json.loads(json.dumps(LEDGER))
    other['chunks']['1'][0][1] = 4
    da, db_ = _run_dir(tmp_path, 'ra', a, LEDGER), _run_dir(tmp_path, 'rb', b, other)
    code, out = _cmp(da, db_)
    assert code == 1
    assert 'DIFF batch_sizes.chunk 1' in out
    reordered = {'chunks': {'0': list(reversed(LEDGER['chunks']['0'])), '1': LEDGER['chunks']['1']}}
    dc = _run_dir(tmp_path, 'rc', b, reordered)
    assert _cmp(da, dc)[0] == 1          # same entries, different order: a different sequence


def test_a_ledger_on_one_side_only_is_invalid(pair, tmp_path):
    a, b = pair
    da, db_ = _run_dir(tmp_path, 'ra', a, LEDGER), _run_dir(tmp_path, 'rb', b, None)
    code, out = _cmp(da, db_)
    assert code == 2 and 'only one side' in out


def test_halving_recorded_in_one_arm_only_fails(pair, tmp_path):
    a, b = pair
    halved = json.loads(json.dumps(LEDGER))
    halved['chunks']['0'][1][2] = 8
    da, db_ = _run_dir(tmp_path, 'ra', a, LEDGER), _run_dir(tmp_path, 'rb', b, halved)
    assert _cmp(da, db_)[0] == 1


# ----------------------------------------------------------------------------- plain-path open, CLI
def test_a_live_wal_database_opens_by_plain_path(tmp_path):
    """The prototype's `file:...?mode=ro` open failed on WAL run databases; the plain path must not."""
    a = tmp_path / 'a.db'
    init_database(str(a))
    writer = sqlite3.connect(str(a))
    writer.execute('PRAGMA journal_mode = WAL')
    vals = {c: _value(t, 0) for c, t in PHOTOS_COLUMNS}
    vals['path'] = PATHS[0]
    _insert(writer, 'photos', PHOTOS_COLUMNS, vals)
    writer.commit()                      # committed to the WAL, writer stays open
    assert (tmp_path / 'a.db-wal').exists()
    try:
        b = _backup(a, tmp_path / 'b.db')
        code, out = _cmp(a, b)
        assert code == 0, out
    finally:
        writer.close()


def test_command_line_exit_codes(pair, tmp_path):
    a, b = pair
    base = [sys.executable, '-I', str(SCRIPT)]
    ok = subprocess.run(base + [str(a), str(b), '--no-report'], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    _exec(b, 'UPDATE photos SET f_stop = 9.5 WHERE path = ?', (PATHS[0],))
    bad = subprocess.run(base + [str(a), str(b), '--no-report'], capture_output=True, text=True)
    assert bad.returncode == 1 and 'DIFF photos.f_stop' in bad.stdout
    inv = subprocess.run(base + [str(a), str(tmp_path / 'missing.db')], capture_output=True, text=True)
    assert inv.returncode == 2


def test_report_lists_tag_sets_and_face_counts(pair):
    code, out = _cmp(*pair)
    assert code == 0
    assert '1.jpg' in out and 'face_rows=2' in out and 'photo_tags=dog' in out


# ----------------------------------------------------------------------------- a REAL run database
REAL_DB = Path(os.environ.get('PERF_REAL_DB', r'D:\facet-scratch\perf2\runs\B16a\db.sqlite'))


@pytest.mark.skipif(not REAL_DB.is_file(), reason='no real scan run database (set PERF_REAL_DB)')
def test_real_run_database_self_test(tmp_path):
    """A backup-API copy of a real run DB compares identical; one flipped thumbnail byte does not."""
    a = _backup(REAL_DB, tmp_path / 'real_a.db')
    b = _backup(REAL_DB, tmp_path / 'real_b.db')
    code, out = _cmp(a, b, require_faces=False)
    assert code == 0, out
    conn = sqlite3.connect(str(b))
    path, blob = conn.execute('SELECT path, thumbnail FROM photos WHERE thumbnail IS NOT NULL LIMIT 1').fetchone()
    flipped = bytearray(blob)
    flipped[len(flipped) // 2] ^= 0x01
    conn.execute('UPDATE photos SET thumbnail = ? WHERE path = ?', (bytes(flipped), path))
    conn.commit()
    conn.close()
    code, out = _cmp(a, b)
    assert code == 1
    assert 'DIFF photos.thumbnail' in out
