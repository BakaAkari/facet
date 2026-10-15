"""Cell-exact comparator for two scan databases (replaces the prototype parity.py).

usage: compare.py A B [--profile-name NAME] [--ledger-a FILE --ledger-b FILE] [--require-faces]
                  [--max-diffs N] [--no-report]
  A, B       baseline and patched: a run directory (uses <dir>/db.sqlite and <dir>/batch_sizes.json)
             or a database file (a batch_sizes.json beside it is picked up when present)
  --require-faces   the baseline MUST hold at least one face (the pinned 49-photo set holds 7); a
                    baseline without faces makes the run invalid (exit 2) because the face path
                    would be untested

Exit codes
  0  every compared cell is identical (tolerance list aside), every required table is non-empty,
     cells_compared > 0 and the batch-size ledgers match
  1  at least one differing cell / row, or the ledgers differ (the first --max-diffs are printed)
  2  the comparison itself is invalid: missing table, column sets differ, `photos` empty or its key
     sets differ, a baseline with no faces under --require-faces, a ledger on one side only

What is compared
  photos      by `path`; every column as a (sqlite typeof, value) pair, BLOBs by sha256, floats by bit
              pattern (so -0.0 vs 0.0 and NaN payloads cannot hide). Excluded: see EXCLUDED.
  faces       by (photo_path, face_index), the same way. A face key present in one DB only is a
              difference (exit 1), not exit 2: a patched scan that lost its faces must read as a regression.
  photo_tags  as a multiset of full rows. Empty is allowed only when the baseline is empty too.
  scan_failures  as a multiset of (path, stage, error) with 0x... addresses normalised; the exception
              CLASS is not stored, only str(error)[:500].

The only tolerance (TOLERANCE) is 1e-12 relative on three unrounded `photos` columns that inherit
S3's <= 1e-15 variance differences. Nothing else is tolerated.

Both databases are opened with a PLAIN path plus `PRAGMA query_only = ON`, never a `file:` URI (a
`file:...?mode=ro` open failed on WAL run databases on this box, root cause not established).
"""
import argparse
import collections
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import sys

# table -> {column: reason}. A test asserts this equals the documented list, so a column added to the
# schema is compared by default.
EXCLUDED = {
    'photos': {'scanned_at': 'timestamp of the scan'},
    'faces': {'id': 'autoincrement row id'},
    'scan_failures': {'scan_run_id': 'run id', 'timestamp': 'timestamp of the failure'},
    'photo_tags': {},
}

# photos column -> relative tolerance (absolute floor of the same size when the baseline value is 0)
TOLERANCE = {
    'raw_sharpness_variance': 1e-12,
    'isolation_bonus': 1e-12,
    'aggregate': 1e-12,
}

REQUIRED_TABLES = ('photos', 'faces', 'photo_tags', 'scan_failures')
_ADDR = re.compile(r'0x[0-9a-fA-F]+')


class Invalid(Exception):
    """The comparison cannot be made (exit 2)."""


def resolve(arg):
    """Return (db_path, ledger_path_or_None) for a run directory or a database file."""
    if os.path.isdir(arg):
        db = os.path.join(arg, 'db.sqlite')
        base = arg
    else:
        db = arg
        base = os.path.dirname(os.path.abspath(arg))
    led = os.path.join(base, 'batch_sizes.json')
    return db, (led if os.path.isfile(led) else None)


def open_ro(path):
    if not os.path.isfile(path):
        raise Invalid('database does not exist: %s' % path)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA query_only = ON')
    conn.text_factory = lambda b: b.decode('utf-8', 'surrogateescape')
    return conn


def table_columns(conn, table):
    cols = [r[1] for r in conn.execute('PRAGMA table_info("%s")' % table)]
    return cols


def canon(t, v):
    if v is None:
        return (t, None)
    if t == 'real':
        return (t, struct.pack('<d', v))
    if t == 'blob':
        return (t, hashlib.sha256(v).hexdigest(), len(v))
    return (t, v)


def show(c):
    t = c[0]
    if c[1] is None:
        return 'NULL'
    if t == 'real':
        return 'real %r (%s)' % (struct.unpack('<d', c[1])[0], c[1].hex())
    if t == 'blob':
        return 'blob sha256=%s.. len=%d' % (c[1][:12], c[2])
    if t == 'text':
        s = c[1]
        return 'text %r' % (s if len(s) <= 60 else s[:57] + '...')
    return '%s %r' % (t, c[1])


def read_keyed(conn, table, key_cols, cols):
    """{key tuple: {col: canon cell}} for `cols`, with SQLite's own typeof() on each cell."""
    sel = ', '.join('"%s"' % k for k in key_cols)
    sel += ''.join(', typeof("%s"), "%s"' % (c, c) for c in cols)
    out = {}
    nk = len(key_cols)
    for row in conn.execute('SELECT %s FROM "%s"' % (sel, table)):
        key = tuple(row[:nk])
        cells = {}
        for i, c in enumerate(cols):
            cells[c] = canon(row[nk + 2 * i], row[nk + 2 * i + 1])
        out[key] = cells
    return out


def within_tolerance(col, a, b):
    """True when the two cells are both REAL, finite, and within TOLERANCE[col] of each other."""
    if col not in TOLERANCE or a[0] != 'real' or b[0] != 'real' or a[1] is None or b[1] is None:
        return False
    x, y = struct.unpack('<d', a[1])[0], struct.unpack('<d', b[1])[0]
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    tol = TOLERANCE[col]
    err = abs(x - y)
    return err <= tol if x == 0 else err / abs(x) <= tol


class Report:
    def __init__(self, max_diffs):
        self.diffs = []
        self.max_diffs = max_diffs
        self.n_diffs = 0
        self.tol_used = collections.Counter()
        self.tol_max_rel = collections.defaultdict(float)
        self.cells = collections.Counter()
        self.rows = {}
        self.notes = []
        self.invalid = []

    def diff(self, table, column, key, a, b):
        self.n_diffs += 1
        if len(self.diffs) < self.max_diffs:
            self.diffs.append((table, column, key, a, b))


def compare_table(rep, table, key_cols, a_conn, b_conn, a_cols, b_cols):
    excl = set(EXCLUDED.get(table, {}))
    if a_cols != b_cols:
        only_a = [c for c in a_cols if c not in b_cols]
        only_b = [c for c in b_cols if c not in a_cols]
        raise Invalid('%s: column sets differ (only in A: %s; only in B: %s)' % (table, only_a, only_b))
    cols = [c for c in a_cols if c not in excl and c not in key_cols]
    a = read_keyed(a_conn, table, key_cols, cols)
    b = read_keyed(b_conn, table, key_cols, cols)
    rep.rows[table] = (len(a), len(b))
    ka, kb = set(a), set(b)
    only_a, only_b = sorted(ka - kb, key=repr), sorted(kb - ka, key=repr)
    if table == 'photos' and (only_a or only_b):
        raise Invalid('photos: key sets differ (%d only in A, %d only in B; e.g. %s)'
                      % (len(only_a), len(only_b), (only_a + only_b)[:3]))
    for k in only_a:
        rep.diff(table, '<row>', k, 'present', 'MISSING')
    for k in only_b:
        rep.diff(table, '<row>', k, 'MISSING', 'present')
    for k in sorted(ka & kb, key=repr):
        for c in cols:
            x, y = a[k][c], b[k][c]
            rep.cells[table] += 1
            if x == y:
                continue
            if table == 'photos' and within_tolerance(c, x, y):
                rep.tol_used[c] += 1
                fx, fy = struct.unpack('<d', x[1])[0], struct.unpack('<d', y[1])[0]
                rel = abs(fx - fy) / abs(fx) if fx else abs(fx - fy)
                rep.tol_max_rel[c] = max(rep.tol_max_rel[c], rel)
                continue
            rep.diff(table, c, k, show(x), show(y))
    return a, b


def read_rows(conn, table, cols):
    sel = ', '.join('"%s"' % c for c in cols)
    return [tuple(r) for r in conn.execute('SELECT %s FROM "%s"' % (sel, table))]


def compare_multiset(rep, table, a_conn, b_conn, a_cols, b_cols, norm=None):
    excl = set(EXCLUDED.get(table, {}))
    if a_cols != b_cols:
        raise Invalid('%s: column sets differ (A %s, B %s)' % (table, a_cols, b_cols))
    cols = [c for c in a_cols if c not in excl]
    ra, rb = read_rows(a_conn, table, cols), read_rows(b_conn, table, cols)
    if norm:
        ra, rb = [norm(cols, r) for r in ra], [norm(cols, r) for r in rb]
    ca, cb = collections.Counter(ra), collections.Counter(rb)
    rep.rows[table] = (len(ra), len(rb))
    rep.cells[table] += len(set(ra) | set(rb)) * len(cols)
    for r in sorted((ca - cb), key=repr):
        rep.diff(table, '<row>', r, 'present x%d' % ca[r], 'x%d' % cb[r] if cb[r] else 'MISSING')
    for r in sorted((cb - ca), key=repr):
        rep.diff(table, '<row>', r, 'x%d' % ca[r] if ca[r] else 'MISSING', 'present x%d' % cb[r])
    return ra, rb, cols


def norm_failure(cols, row):
    d = dict(zip(cols, row))
    err = d.get('error')
    if isinstance(err, str):
        d['error'] = _ADDR.sub('0x?', err)
    return tuple(d[c] for c in cols)


def compare_ledgers(rep, la, lb):
    if (la is None) != (lb is None):
        raise Invalid('batch_sizes.json exists for only one side (A: %s, B: %s)' % (la, lb))
    if la is None:
        rep.notes.append('ledger: not compared (no batch_sizes.json beside either database)')
        return
    A, B = json.load(open(la)), json.load(open(lb))
    if A == B:
        n = sum(len(v) for v in A.get('chunks', {}).values())
        rep.notes.append('ledger: sub-batch sequences identical (%d entries over %d chunks)'
                         % (n, len(A.get('chunks', {}))))
        return
    for ch in sorted(set(A.get('chunks', {})) | set(B.get('chunks', {})), key=lambda s: int(s)):
        x, y = A.get('chunks', {}).get(ch), B.get('chunks', {}).get(ch)
        if x != y:
            rep.diff('batch_sizes', 'chunk %s' % ch, (ch,), json.dumps(x), json.dumps(y))
            return
    rep.diff('batch_sizes', '<other>', (), json.dumps(A)[:200], json.dumps(B)[:200])


def run(a_arg, b_arg, profile_name=None, ledger_a=None, ledger_b=None, require_faces=False,
        max_diffs=20, report=True, out=print):
    rep = Report(max_diffs)
    a_db, a_led = resolve(a_arg)
    b_db, b_led = resolve(b_arg)
    a_led, b_led = ledger_a or a_led, ledger_b or b_led
    out('compare: A=%s' % a_db)
    out('         B=%s%s' % (b_db, '   [profile %s]' % profile_name if profile_name else ''))
    a_conn = b_conn = None
    try:
        a_conn, b_conn = open_ro(a_db), open_ro(b_db)
        cols = {}
        for t in REQUIRED_TABLES:
            ca, cb = table_columns(a_conn, t), table_columns(b_conn, t)
            if not ca or not cb:
                raise Invalid('table %r is missing in %s' % (t, 'A' if not ca else 'B'))
            cols[t] = (ca, cb)
        pa, pb = compare_table(rep, 'photos', ('path',), a_conn, b_conn, *cols['photos'])
        if not pa or not pb:
            raise Invalid('photos is empty in %s: nothing was compared' % ('A' if not pa else 'B'))
        fa, fb = compare_table(rep, 'faces', ('photo_path', 'face_index'), a_conn, b_conn, *cols['faces'])
        if require_faces and not fa:
            raise Invalid('the baseline holds 0 faces: the face path would be untested (--require-faces)')
        if not fa and not fb:
            rep.notes.append('faces: 0 rows in both - the face path is UNTESTED by this comparison')
        ta, tb, _ = compare_multiset(rep, 'photo_tags', a_conn, b_conn, *cols['photo_tags'])
        if not ta and not tb:
            rep.notes.append('photo_tags: 0 rows in both, expected-empty for a scan: a scan never writes it, '
                             'only --migrate-tags and tag_existing do; per-photo `tags` is covered as a `photos` column')
        compare_multiset(rep, 'scan_failures', a_conn, b_conn, *cols['scan_failures'], norm=norm_failure)
        if not rep.rows['scan_failures'][0] and not rep.rows['scan_failures'][1]:
            rep.notes.append('scan_failures: 0 rows in both (the normal case)')
        compare_ledgers(rep, a_led, b_led)
        if report:
            report_lines(out, a_conn, b_conn, pb, fb)
    except Invalid as e:
        out('INVALID COMPARISON: %s' % e)
        out('RESULT: exit 2')
        return 2
    finally:
        for c in (a_conn, b_conn):
            if c is not None:
                c.close()
    total_cells = sum(rep.cells.values())
    for t in REQUIRED_TABLES:
        ra, rb = rep.rows[t]
        out('  %-14s rows A=%-6d B=%-6d cells_compared=%d' % (t, ra, rb, rep.cells[t]))
    out('cells_compared: %d' % total_cells)
    if rep.tol_used:
        out('tolerance (<= 1e-12 relative, closed list): ' + ', '.join(
            '%s: %d cells, max rel %.3g' % (c, n, rep.tol_max_rel[c]) for c, n in sorted(rep.tol_used.items())))
    else:
        out('tolerance: no cell needed it (all bit-identical)')
    for n in rep.notes:
        out(n)
    if total_cells <= 0:
        out('INVALID COMPARISON: cells_compared == 0')
        out('RESULT: exit 2')
        return 2
    out('differing cells/rows: %d' % rep.n_diffs)
    for table, col, key, x, y in rep.diffs:
        out('  DIFF %s.%s %s: A=%s B=%s' % (table, col, key, x, y))
    if rep.n_diffs > len(rep.diffs):
        out('  ... %d more not shown' % (rep.n_diffs - len(rep.diffs)))
    if rep.n_diffs:
        out('RESULT: exit 1')
        return 1
    out('RESULT: identical, exit 0')
    return 0


def report_lines(out, a_conn, b_conn, photos_b, faces_b):
    """Tag sets per photo and face counts (patched side), for the S0/wave reports."""
    tags = collections.defaultdict(list)
    for p, t in b_conn.execute('SELECT photo_path, tag FROM photo_tags ORDER BY photo_path, tag'):
        tags[p].append(t)
    faces = collections.Counter(k[0] for k in faces_b)
    out('report (side B): %d photos, %d faces on %d photos, photo_tags rows on %d photos'
        % (len(photos_b), len(faces_b), len(faces), len(tags)))
    for (path,), cells in sorted(photos_b.items()):
        t = cells.get('tags')
        tag_text = t[1] if t and t[0] == 'text' else None
        tag_set = sorted(x for x in (tag_text or '').split(',') if x)
        out('  %s: tags=%s face_rows=%d photo_tags=%s' % (os.path.basename(path), ','.join(tag_set) or '-',
                                                          faces.get(path, 0), ','.join(tags.get(path, [])) or '-'))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Cell-exact comparator for two scan databases')
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--profile-name')
    ap.add_argument('--ledger-a')
    ap.add_argument('--ledger-b')
    ap.add_argument('--require-faces', action='store_true')
    ap.add_argument('--max-diffs', type=int, default=20)
    ap.add_argument('--no-report', action='store_true')
    ns = ap.parse_args(argv)
    return run(ns.a, ns.b, ns.profile_name, ns.ledger_a, ns.ledger_b, ns.require_faces, ns.max_diffs,
               not ns.no_report)


if __name__ == '__main__':
    sys.exit(main())
