"""Embedding-only visual similarity ranks by plain cosine.

This file once covered the sqlite-vec KNN branch of ``_find_similar_visual``.
That branch was removed when visual similarity moved to an exact scan over the
shared evidence (``api/visual_similarity.py``): an ANN candidate cap cannot
report the complete thresholded total the endpoint now paginates. What remains
is the brute-force reference that branch was checked against: with no pHash
anywhere, the ranking is the cosine of the embeddings at full weight.
"""

import sqlite3

import numpy as np
import pytest

from api.routers.gallery import _similar_result
from api.visual_similarity import find_visual_page
from db.schema import init_database
from utils.embedding import embedding_to_bytes

DIM = 16


@pytest.fixture
def embeddings():
    rng = np.random.RandomState(42)
    return {f"/p{i}.jpg": rng.randn(DIM).astype(np.float32) for i in range(60)}


def _cosine_expected(embeddings, source_path):
    src = embeddings[source_path] / np.linalg.norm(embeddings[source_path])
    scored = []
    for p, emb in embeddings.items():
        if p == source_path:
            continue
        sim = max(0.0, float(np.dot(src, emb / np.linalg.norm(emb))))
        scored.append((p, round(sim, 4)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def test_embedding_only_ranking_is_the_cosine_order(tmp_path, embeddings):
    db = str(tmp_path / "embeddings.db")
    init_database(db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    for p, emb in embeddings.items():
        conn.execute("INSERT INTO photos (path, filename, clip_embedding) VALUES (?, ?, ?)",
                     [p, p.lstrip('/'), embedding_to_bytes(emb)])
    conn.commit()
    source_path = "/p0.jpg"
    source = dict(conn.execute("SELECT phash, clip_embedding FROM photos WHERE path = ?", [source_path]).fetchone())

    page, total, message = find_visual_page(conn, source, source_path, 0.0, "1=1", [], 0, 100, _similar_result)

    expected = _cosine_expected(embeddings, source_path)
    assert message is None
    assert total == len(expected)
    assert [(r['path'], r['similarity']) for r in page] == expected
    conn.close()
