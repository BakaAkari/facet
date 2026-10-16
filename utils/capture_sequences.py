"""Conservative capture-sequence evidence and shared protection policy.

Hardware evidence protects incomplete sets too. Detection never deletes files.
"""
import hashlib
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PIXEL_SHIFT = 'pixel_shift'
TIMELAPSE = 'timelapse'
PROTECTED_KINDS = (PIXEL_SHIFT, TIMELAPSE)


def _tag(data, name):
    return next((v for k, v in data.items() if k.rsplit(':', 1)[-1] == name), None)


def sony_pixel_shift(value):
    """ExifTool Sony.pm 0x202f raw (-n) and formatted representations."""
    text = str(value or '').strip()
    match = re.fullmatch(r'(\d{8}) (\d+) (4|16) (0x[0-9a-fA-F]+)', text)
    if match is None:
        match = re.fullmatch(r'Group (\d{8}), Shot (\d+)/(4|16) \((0x[0-9a-fA-F]+)\)', text)
    if match is None:
        return None
    group, frame, total, suffix = match.groups()
    if not 1 <= int(frame) <= int(total):
        return None  # composed output (frame 0) is not a source frame
    return group + ':' + suffix, int(frame), int(total)


def pixel_shift_evidence(data):
    """Return validated mode/offset; unknown Sony layouts are not guessed."""
    shots = _tag(data, 'PixelShiftShots')
    try:
        mode = int(shots)
        if float(shots) != mode or mode not in (4, 16):
            mode = None
    except (TypeError, ValueError, OverflowError):
        mode = None
    return {'pixel_shift_shots': mode,
            'pixel_shift_offset': str(_tag(data, 'PixelShiftOffset'))
            if _tag(data, 'PixelShiftOffset') is not None else None,
            'pixel_shift_info': str(_tag(data, 'PixelShiftInfo'))
            if _tag(data, 'PixelShiftInfo') is not None else None,
            'camera_serial': str(_tag(data, 'SerialNumber') or
                                 _tag(data, 'InternalSerialNumber') or '') or None}


def _time(value):
    if not value:
        return None
    try:
        text = str(value)
        if len(text) >= 10 and text[4] == ':':
            text = text[:10].replace(':', '-') + text[10:]
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _distance(a, b):
    try:
        if not a or not b:
            return 999
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except (ValueError, TypeError):
        return 999


def stable_group_id(kind, paths):
    """Positive SQLite int64, independent of unrelated library additions."""
    key = kind + '\0' + min(paths)
    # 48 bits also round-trip exactly through JavaScript Number / JSON clients.
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:6], 'big')


def find_timelapses(rows, *, min_frames=15, min_interval=1.0,
                    max_interval=3600.0, tolerance=0.15, max_hamming=8):
    """Detect near-uniform cadence AND stable framing, never cadence alone.

    Unknown focal length/hash/time is insufficient evidence. Same-model bodies
    without serial numbers are separated by directory as a conservative fallback.
    Compare every frame with the run anchor to avoid slowly drifting sequences.
    """
    min_frames = 15 if min_frames is None else min_frames
    if min_frames < 15 or not 0 <= tolerance < 0.5:
        raise ValueError('unsafe timelapse settings')
    buckets = defaultdict(list)
    for row in rows:
        p = dict(row)
        p['_time'] = _time(p.get('date_taken'))
        focal = p.get('focal_length')
        if p['_time'] is None or not p.get('camera_model') or not p.get('phash'):
            continue
        try:
            focal = float(focal)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(focal) or focal <= 0:
            continue
        key = (p['camera_model'], p.get('camera_serial'),
               str(Path(p['path']).parent), focal)
        buckets[key].append(p)
    groups = []
    for bucket in buckets.values():
        bucket.sort(key=lambda p: (p['_time'], p['path']))
        run, interval = [], None
        for p in bucket:
            if run:
                try:
                    gap = (p['_time'] - run[-1]['_time']).total_seconds()
                except TypeError:
                    gap = -1
                continues = (min_interval <= gap <= max_interval
                             and _distance(run[0]['phash'], p['phash']) <= max_hamming
                             and (interval is None or
                                  abs(gap - interval) <= max(0.05, interval * tolerance)))
                if not continues:
                    if len(run) >= min_frames:
                        groups.append([r['path'] for r in run])
                    run, interval = [], None
                elif interval is None:
                    interval = gap
            run.append(p)
        if len(run) >= min_frames:
            groups.append([r['path'] for r in run])
    return sorted(groups, key=lambda group: min(group))


def detect_capture_sequences(db_path, *, dry_run=False, settings=None):
    """Whole-library metadata pass. Only own labels are updated, never files.

    Existing protected labels are sticky: missing metadata on a later scan must
    not silently revoke protection. Explicit review is needed to clear evidence.
    """
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            'SELECT path, date_taken, camera_model, camera_serial, focal_length, '
            'phash, pixel_shift_shots, pixel_shift_offset, pixel_shift_info, '
            'sequence_kind FROM photos ORDER BY date_taken, path')]
        pixel_groups = []
        sony_groups = defaultdict(list)
        fuji_runs = defaultdict(list)
        for p in rows:
            sony = sony_pixel_shift(p['pixel_shift_info'])
            identity = (p['camera_model'], p['camera_serial'], str(Path(p['path']).parent))
            if sony:
                # Date isolates group ids that cameras may reuse on later days.
                sony_groups[(*identity, str(p['date_taken'])[:10], sony[0])].append(p['path'])
            elif p['pixel_shift_shots'] in (4, 16):
                fuji_runs[(*identity, p['pixel_shift_shots'])].append(p)
        pixel_groups.extend(sony_groups.values())
        for bucket in fuji_runs.values():
            run, offsets = [], set()
            for p in bucket:
                offset = p['pixel_shift_offset']
                gap = None
                if run:
                    a, b = _time(run[-1]['date_taken']), _time(p['date_taken'])
                    if a is not None and b is not None:
                        try:
                            gap = (b - a).total_seconds()
                        except TypeError:
                            pass
                # Missing offset cannot establish a set boundary: protect as a
                # singleton rather than inventing a complete synthesis group.
                boundary = (offset is None or offset in offsets or
                            (run and (gap is None or not 0 <= gap <= 60)) or
                            len(run) >= p['pixel_shift_shots'])
                if boundary and run:
                    pixel_groups.append([r['path'] for r in run])
                    run, offsets = [], set()
                run.append(p)
                offsets.add(offset)
                if offset is None:
                    pixel_groups.append([p['path']])
                    run, offsets = [], set()
            if run:
                pixel_groups.append([r['path'] for r in run])
        pixel_paths = {p for group in pixel_groups for p in group}
        time_groups = find_timelapses(
            [r for r in rows if r['path'] not in pixel_paths
             and r['sequence_kind'] != PIXEL_SHIFT], **(settings or {}))
        groups = [(PIXEL_SHIFT, g) for g in pixel_groups] + [(TIMELAPSE, g) for g in time_groups]
        if not dry_run:
            for kind, paths in groups:
                group_id = stable_group_id(kind, paths)
                conn.executemany(
                    'UPDATE photos SET sequence_kind=?, sequence_group_id=?, '
                    'sequence_ev_offset=NULL, burst_group_id=NULL, is_burst_lead=0, '
                    'is_sequence_lead=? WHERE path=?',
                    [(kind, group_id, int(p == min(paths)), p) for p in paths])
            conn.execute('UPDATE photos SET burst_group_id=NULL, is_burst_lead=0 '
                         'WHERE sequence_kind IN (?, ?)', PROTECTED_KINDS)
        return {'groups': [{'kind': kind, 'paths': paths} for kind, paths in groups],
                'dry_run': dry_run}


def persist_capture_metadata(conn, photo):
    """Store extraction evidence and protect tagged frames in the scan transaction."""
    fields = ('pixel_shift_shots', 'pixel_shift_offset', 'pixel_shift_info', 'camera_serial')
    if not any(field in photo for field in fields):
        return
    conn.execute('UPDATE photos SET ' + ', '.join(f'{f}=COALESCE(?, {f})' for f in fields)
                 + ' WHERE path=?', (*[photo.get(f) for f in fields], photo['path']))
    sony = sony_pixel_shift(photo.get('pixel_shift_info'))
    if photo.get('pixel_shift_shots') in (4, 16) or sony:
        # Keep valid set identity across rescans until the grouping pass updates it.
        group = stable_group_id(PIXEL_SHIFT, [photo['path']])
        conn.execute('UPDATE photos SET sequence_group_id=CASE WHEN sequence_kind=? '
                     'THEN COALESCE(sequence_group_id, ?) ELSE ? END, sequence_kind=?, '
                     'burst_group_id=NULL, is_burst_lead=0, sequence_ev_offset=NULL '
                     'WHERE path=?', (PIXEL_SHIFT, group, group, PIXEL_SHIFT, photo['path']))


def require_unprotected_files(conn, paths):
    """Refuse the entire destructive operation, including resolved companions."""
    if protected_paths(conn, paths):
        from fastapi import HTTPException
        raise HTTPException(status_code=409,
                            detail='Protected Pixel Shift/Timelapse source frames cannot be rejected or removed')


def protected_paths(conn, paths):
    """Bounded queries shared by reject and physical-delete boundaries."""
    paths = list(dict.fromkeys(paths))
    result = set()
    for start in range(0, len(paths), 400):
        batch = paths[start:start + 400]
        marks = ','.join('?' for _ in batch)
        result.update(row[0] for row in conn.execute(
            f'SELECT path FROM photos WHERE path IN ({marks}) '
            'AND sequence_kind IN (?, ?)', (*batch, *PROTECTED_KINDS)))
    return result
