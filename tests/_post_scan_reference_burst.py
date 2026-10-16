"""The HEAD burst grouping, kept verbatim as the oracle for ``tests/test_burst_diff_write.py``.

Not a test module (the leading underscore keeps pytest from collecting it). It is the body of
``processing.scorer.process_bursts`` as it stood before the diff-only rewrite (post-scan speed
spec, Step 2), with the database writes replaced by writes to an in-memory ``state`` so the
result can be compared with what the new function stored.

The oracle must not be circular. Two things keep it independent of the code under test:

* the ROW ORDER comes from HEAD's own statement (``HEAD_BURST_SQL``, byte for byte
  ``... WHERE phash IS NOT NULL ORDER BY date_taken``) executed by SQLite against the same
  database, never from the Python sort key the new code pins. The order of equal dates is
  whatever the planner's index walk yields; ``head_order_plan`` returns the plan so the test
  can assert it is the one the pin was derived from (``idx_date_taken_desc``);
* the date of every pair is parsed per pair, as HEAD did, and nothing is precomputed.

``reference_bursts`` returns ``(state, emitted, group_count)``: ``state`` maps path to
``(is_burst_lead, burst_group_id)`` after HEAD's resets and updates, ``emitted`` is the list of
``on_burst_detected`` payloads in emission order, and ``group_count`` is the number HEAD's final
log line printed.
"""

import sqlite3

# Byte for byte HEAD's statement (processing/scorer.py before this change).
HEAD_BURST_SQL = (
    "SELECT path, date_taken, aggregate, phash, "
    "face_count, eyes_open_score, expression_score, tech_sharpness, "
    "(SELECT ls.learned_score FROM learned_scores ls "
    " WHERE ls.photo_path = photos.path AND ls.user_id IS NULL "
    " AND ls.category IS NULL) AS learned_score "
    "FROM photos WHERE phash IS NOT NULL ORDER BY date_taken"
)


def head_order_plan(db_path):
    """The ``EXPLAIN QUERY PLAN`` detail lines of HEAD's statement on this database."""
    conn = sqlite3.connect(db_path)
    try:
        return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + HEAD_BURST_SQL)]
    finally:
        conn.close()


def reference_bursts(db_path, max_hamming_distance, time_window_minutes, rapid_burst_seconds):
    """Run HEAD's grouping against ``db_path`` in memory. Never writes to the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        photos = conn.execute(HEAD_BURST_SQL).fetchall()

        stored = {
            r['path']: (r['phash'], r['is_burst_lead'], r['burst_group_id'])
            for r in conn.execute("SELECT path, phash, is_burst_lead, burst_group_id FROM photos")
        }
        photo_persons = {}
        has_faces = conn.execute("SELECT 1 FROM faces LIMIT 1").fetchone()
        if has_faces:
            face_data = conn.execute(
                "SELECT photo_path, person_id FROM faces WHERE person_id IS NOT NULL"
            ).fetchall()
            for row in face_data:
                photo_persons.setdefault(row['photo_path'], set()).add(row['person_id'])
    finally:
        conn.close()

    # state[path] = [is_burst_lead, burst_group_id]
    state = {path: [lead, gid] for path, (_ph, lead, gid) in stored.items()}

    def run_update(sql_kind, path=None, value=None):
        """The UPDATE statements HEAD issued, applied to ``state``."""
        if sql_kind == 'reset_lead_phash':          # UPDATE ... SET is_burst_lead = 0 WHERE phash IS NOT NULL
            for p, (ph, _lead, _gid) in stored.items():
                if ph is not None:
                    state[p][0] = 0
        elif sql_kind == 'lead_nophash':            # UPDATE ... SET is_burst_lead = 1 WHERE phash IS NULL
            for p, (ph, _lead, _gid) in stored.items():
                if ph is None:
                    state[p][0] = 1
        elif sql_kind == 'reset_gid_phash':         # UPDATE ... SET burst_group_id = NULL WHERE phash IS NOT NULL
            for p, (ph, _lead, _gid) in stored.items():
                if ph is not None:
                    state[p][1] = None
        elif sql_kind == 'lead':                    # UPDATE ... SET is_burst_lead = 1 WHERE path = ?
            state[path][0] = 1
        elif sql_kind == 'gid':                     # UPDATE ... SET burst_group_id = ? WHERE path = ?
            state[path][1] = value

    if not photos:
        return {p: tuple(v) for p, v in state.items()}, [], 0

    # --- verbatim from HEAD from here on -------------------------------------------------
    run_update('reset_lead_phash')
    run_update('lead_nophash')

    def phash_distance(hash1, hash2):
        """Compute hamming distance between two hex phash strings."""
        if not hash1 or not hash2:
            return 999
        return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

    from utils.date_utils import parse_date
    from utils.selection import pick_lead

    def shares_person(path1, path2):
        """Check if two photos share at least one identified person."""
        persons1 = photo_persons.get(path1, set())
        persons2 = photo_persons.get(path2, set())
        # If either has no identified faces, allow grouping
        if not persons1 or not persons2:
            return True
        # Otherwise require at least one shared person
        return bool(persons1 & persons2)

    def is_similar_to_burst(photo, burst, threshold, time_limit, rapid_seconds):
        """Check if photo belongs to burst - rapid shots require face consistency."""
        photo_date = parse_date(photo['date_taken'])
        if photo_date is None:
            return False

        for b in burst:
            b_date = parse_date(b['date_taken'])
            if b_date is None:
                continue

            time_diff = abs((photo_date - b_date).total_seconds())

            # Rapid burst: within N seconds AND face-consistent AND visually similar
            # Use relaxed phash threshold (2x normal) for rapid bursts
            if time_diff <= rapid_seconds:
                if shares_person(photo['path'], b['path']):
                    if phash_distance(photo['phash'], b['phash']) <= threshold * 2:
                        return True

            # Slow burst: within time window AND visually similar
            if time_diff <= time_limit * 60:
                if phash_distance(photo['phash'], b['phash']) <= threshold:
                    return True

        return False

    run_update('reset_gid_phash')

    current_burst = [photos[0]]
    group_id = 0
    burst_groups_for_emit = []
    for i in range(1, len(photos)):
        if is_similar_to_burst(photos[i], current_burst, max_hamming_distance, time_window_minutes, rapid_burst_seconds):
            current_burst.append(photos[i])
        else:
            # Finalize previous burst
            if current_burst:
                winner = pick_lead([dict(p) for p in current_burst])
                run_update('lead', winner['path'])
                if len(current_burst) >= 2:
                    for p in current_burst:
                        run_update('gid', p['path'], group_id)
                    burst_groups_for_emit.append({
                        'burst_group_id': group_id,
                        'photo_count': len(current_burst),
                        'best_path': winner['path'],
                        'paths': [p['path'] for p in current_burst],
                    })
                    group_id += 1
            current_burst = [photos[i]]

    # Handle final burst
    if current_burst:
        winner = pick_lead([dict(p) for p in current_burst])
        run_update('lead', winner['path'])
        if len(current_burst) >= 2:
            for p in current_burst:
                run_update('gid', p['path'], group_id)
            burst_groups_for_emit.append({
                'burst_group_id': group_id,
                'photo_count': len(current_burst),
                'best_path': winner['path'],
                'paths': [p['path'] for p in current_burst],
            })

    group_count = group_id + 1 if current_burst and len(current_burst) >= 2 else group_id
    return {p: tuple(v) for p, v in state.items()}, burst_groups_for_emit, group_count
