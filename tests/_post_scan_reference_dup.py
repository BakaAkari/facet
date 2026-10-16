"""HEAD's duplicate-detection algorithm, kept verbatim as the oracle for Step 4 of the
post-scan speed spec (``tests/test_duplicate_diff_write.py``).

This is a helper module, not a test file. It holds the grouping logic of
``utils.duplicate.detect_duplicates`` as it was before the change (commit 8a2bb440), with
two edits and no others:

* the final write is not performed; the state the blanket reset plus the per-row UPDATEs
  would have left is RETURNED, as ``{path: (duplicate_group_id, is_duplicate_lead)}`` for
  every row of ``photos``;
* the helpers the change modified (``_hamming_to_all``) or that it leans on
  (``_two_stage_union``, ``_build_embedding_matrix``) are copied here, so the oracle does
  not call into the module under test. ``_hamming_to_all`` is the byte-table loop.

The row ORDER comes from HEAD's statement, ``... ORDER BY path``, executed against the same
database -- never from a Python sort -- so a tie-order divergence cannot hide behind a
shared sort key.
"""

import sqlite3
from collections import Counter

import numpy as np

from utils.embedding import bytes_to_normalized_embedding
from utils.union_find import UnionFind as _UnionFind

# HEAD's read, byte for byte (``git show 8a2bb440:utils/duplicate.py``).
HEAD_ROWS_SQL = (
    "SELECT path, phash, aggregate, clip_embedding, "
    "face_count, eyes_open_score, expression_score, tech_sharpness, "
    "(SELECT ls.learned_score FROM learned_scores ls "
    " WHERE ls.photo_path = photos.path AND ls.user_id IS NULL "
    " AND ls.category IS NULL) AS learned_score "
    "FROM photos WHERE phash IS NOT NULL ORDER BY path"
)

_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def _hex_to_uint64(hex_str):
    """Convert a hex pHash string to uint64."""
    return int(hex_str, 16)


def _hamming_to_all(query_hash, hashes):
    """Vectorized Hamming distance of one uint64 hash against an array of hashes."""
    xor_result = np.bitwise_xor(query_hash, hashes)
    distances = np.zeros(len(hashes), dtype=np.int32)
    for byte_idx in range(8):
        byte_vals = (xor_result >> np.uint64(byte_idx * 8)) & np.uint64(0xFF)
        distances += _POPCOUNT_TABLE[byte_vals.astype(np.int32)]
    return distances


def _build_embedding_matrix(emb_blobs):
    decoded = [bytes_to_normalized_embedding(b) for b in emb_blobs]
    dims = [e.shape[0] for e in decoded if e is not None]
    n = len(decoded)
    has_emb = np.zeros(n, dtype=bool)
    if not dims:
        return None, has_emb
    target_dim = Counter(dims).most_common(1)[0][0]
    matrix = np.zeros((n, target_dim), dtype=np.float32)
    for idx, e in enumerate(decoded):
        if e is not None and e.shape[0] == target_dim:
            matrix[idx] = e
            has_emb[idx] = True
    return matrix, has_emb


def _two_stage_union(hashes, matrix, has_emb, max_distance, prefilter_hamming, cosine_threshold):
    n = len(hashes)
    uf = _UnionFind(n)
    for i in range(n):
        start_j = i + 1
        if start_j >= n:
            continue
        remaining = hashes[start_j:]
        distances = _hamming_to_all(hashes[i], remaining)

        if matrix is not None and has_emb[i]:
            both_emb = has_emb[start_j:]
            gate = np.where(both_emb, prefilter_hamming, max_distance)
            cand = np.where(distances <= gate)[0]
            for mi in cand:
                j = start_j + int(mi)
                if has_emb[j]:
                    if float(np.dot(matrix[i], matrix[j])) >= cosine_threshold:
                        uf.union(i, j)
                else:
                    uf.union(i, j)
        else:
            cand = np.where(distances <= max_distance)[0]
            for mi in cand:
                uf.union(i, start_j + int(mi))
    return uf


def reference_duplicate_state(db_path, config_path):
    """The ``(duplicate_group_id, is_duplicate_lead)`` HEAD's ``detect_duplicates`` leaves.

    Returns ``{path: (group_id, is_lead)}`` for every row of ``photos``. Writes nothing.
    ``config_path`` must be given (the shipped defaults file, in tests).
    """
    from config import ScoringConfig
    from utils.selection import composite_lead_score

    config = ScoringConfig(config_path, validate=False)
    settings = config.get_duplicate_detection_settings()
    similarity_pct = settings.get('similarity_threshold_percent', 90)
    max_distance = int(64 * (1 - similarity_pct / 100))
    prefilter_hamming = max(int(settings.get('prefilter_hamming', 12)), max_distance)
    cosine_threshold = float(settings.get('embedding_cosine_threshold', 0.90))

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(HEAD_ROWS_SQL).fetchall()
        everything = {
            r['path']: (r['duplicate_group_id'], r['is_duplicate_lead'])
            for r in conn.execute("SELECT path, duplicate_group_id, is_duplicate_lead FROM photos")
        }
    finally:
        conn.close()

    if not rows:
        # "No photos with pHash found." -- HEAD returns before writing anything.
        return everything

    paths = [r['path'] for r in rows]
    lead_data = [
        {
            'aggregate': r['aggregate'] or 0.0,
            'face_count': r['face_count'],
            'eyes_open_score': r['eyes_open_score'],
            'expression_score': r['expression_score'],
            'tech_sharpness': r['tech_sharpness'],
            'learned_score': r['learned_score'],
        }
        for r in rows
    ]
    n = len(paths)

    hashes = np.array([_hex_to_uint64(r['phash']) for r in rows], dtype=np.uint64)
    matrix, has_emb = _build_embedding_matrix([r['clip_embedding'] for r in rows])

    uf = _two_stage_union(hashes, matrix, has_emb, max_distance, prefilter_hamming, cosine_threshold)

    groups = {}
    for idx in range(n):
        root = uf.find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(idx)
    dup_groups = {root: members for root, members in groups.items() if len(members) >= 2}

    # The blanket ``UPDATE photos SET duplicate_group_id = NULL, is_duplicate_lead = 0``
    # (both HEAD branches run it), then the per-member UPDATEs.
    state = {path: (None, 0) for path in everything}
    group_id = 1
    for _root, members in sorted(dup_groups.items()):
        best_idx = max(members, key=lambda idx: composite_lead_score(lead_data[idx]))
        for idx in members:
            state[paths[idx]] = (group_id, 1 if idx == best_idx else 0)
        group_id += 1
    return state
