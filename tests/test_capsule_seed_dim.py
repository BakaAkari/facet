"""``_generate_seeded`` must survive a seed whose embedding dimension differs from the
candidate matrix.

``filter_uniform_embeddings`` keeps only the majority dimension (CLIP 768 vs SigLIP 1152)
for ``emb_matrix``, but the seed vector is decoded straight from its row, so a minority
seed made ``emb_matrix @ seed_emb`` raise ``ValueError: matmul ... 1152 is different from
768``. On a real library about one 10-seed draw in four hit it.
"""

import sqlite3

import numpy as np

import analyzers.capsule_generator as cg
from db.schema import init_database


def _embedding(dim, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32).tobytes()  # distinct bytes per row


def test_mixed_dimension_seeds_do_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr("api.db_helpers.is_photo_tags_available", lambda *a, **k: False)

    db_path = str(tmp_path / "cap.db")
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = []
    for i in range(6):  # the majority dimension
        rows.append((f"/p/clip{i}.jpg", f"clip{i}.jpg", 8.0 - i * 0.01,
                     f"2026:03:0{1 + i} 10:00:00", _embedding(768, i)))
    for i in range(2):  # the minority dimension, still picked as seeds
        rows.append((f"/p/siglip{i}.jpg", f"siglip{i}.jpg", 7.9 - i * 0.01,
                     f"2026:03:0{3 + i} 11:00:00", _embedding(1152, 100 + i)))
    conn.executemany(
        "INSERT INTO photos (path, filename, aggregate, date_taken, clip_embedding) "
        "VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()

    # num_seeds >= row count makes all 8 rows seeds (min(num_seeds, len(rows))), so the
    # 1152-d rows are always seeds regardless of the RNG draw.
    config = {"seeded": {"num_seeds": 10, "min_photos": 4}}
    capsules = cg._generate_seeded(
        conn, config, min_aggregate=6.0, vis=("1=1", []), user_id=None)
    conn.close()

    # Non-empty: the time axis still runs for the dimension-mismatched seeds, so this
    # cannot pass by an early ``return []``.
    assert capsules, "expected at least one seeded capsule"
