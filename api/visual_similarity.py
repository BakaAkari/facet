"""Visual comparison using only valid evidence shared by each image pair."""
import math
import re

import numpy as np


def _phash(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{1,16}', value) is None:
        return None
    return int(value, 16)


def _embedding(value):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return None
    try:
        vector = np.frombuffer(value, dtype=np.float32).astype(np.float64)
    except (ValueError, TypeError):
        return None
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-10:
        return None
    return vector / norm


def find_visual(conn, source, photo_path, min_similarity, vis_sql, vis_params, result):
    """Return the complete ranked visible population, then let the API paginate.

    No shared valid feature means unknown, including at threshold zero. Weights
    are normalized over shared features: 70/30 when both exist, 100% otherwise.
    The exact scan deliberately avoids ANN/candidate caps: total/has_more must
    describe the full thresholded set, including embedding-only candidates.
    """
    if not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
        raise ValueError('Similarity threshold must be finite and between zero and one')
    reference_hash = _phash(source.get('phash'))
    reference_embedding = _embedding(source.get('clip_embedding'))
    if reference_hash is None and reference_embedding is None:
        return [], 'Visual similarity is unavailable: this photo has no valid visual fingerprint or embedding. Run image indexing first.'
    cursor = conn.execute(f'''
        SELECT path, filename, phash, clip_embedding, date_taken, aggregate, aesthetic
        FROM photos WHERE path != ? AND (phash IS NOT NULL OR clip_embedding IS NOT NULL) AND {vis_sql}
    ''', [photo_path] + vis_params)
    matches = []
    while batch := cursor.fetchmany(256):
        for raw in batch:
            row = dict(raw)
            candidate_hash = _phash(row['phash'])
            weighted = weight = 0.0
            if reference_hash is not None and candidate_hash is not None:
                weighted += .7 * (1 - (reference_hash ^ candidate_hash).bit_count() / 64)
                weight += .7
            if reference_embedding is not None:
                candidate_embedding = _embedding(row['clip_embedding'])
                if candidate_embedding is not None and candidate_embedding.shape == reference_embedding.shape:
                    cosine = float(np.dot(reference_embedding, candidate_embedding))
                    # Float64 normalization/dot rounding must not exclude an
                    # identical vector at 100%. Bound only roundoff near one,
                    # proportional to the dot-product dimension (no score floor).
                    roundoff = 8 * np.finfo(np.float64).eps * reference_embedding.size
                    if abs(1.0 - cosine) <= roundoff:
                        cosine = 1.0
                    weighted += .3 * min(1.0, max(0.0, cosine))
                    weight += .3
            if not weight:
                continue
            similarity = weighted / weight
            if similarity >= min_similarity:
                matches.append(result(row, similarity))
    matches.sort(key=lambda item: (-item['similarity'], item['path']))
    return matches, None
