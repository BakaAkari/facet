"""Populate the ``aesthetic_clip`` column from cached embeddings + text projection.

Loads the same CLIP/SigLIP model used at scan time (via the canonical text
encoder in ``api.routers.search``), builds the aesthetic axis from the prompts
in ``analyzers/aesthetic_clip.py``, then scores every photo with a cached
``clip_embedding`` BLOB.

The ``aesthetic_clip`` column is part of the canonical schema
(``db/schema.py:PHOTOS_COLUMNS``); fresh DBs already have it. This script only
needs to UPDATE rows.

Usage::

    python scripts/compute_aesthetic_clip.py --db D:/photo-llm/ava_test.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# Make project root importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzers.aesthetic_clip import (
    build_aesthetic_axis,
    load_prompts_from_config,
    score_embeddings,
)
from utils.embedding import bytes_to_embedding


def _make_text_encoder(model_override: str | None = None, backend_override: str | None = None):
    """Build a text encoder matching the active CLIP/SigLIP profile.

    Reads ``model_name`` / ``backend`` / ``pretrained`` from ``scoring_config.json``
    via ``ScoringConfig.get_clip_config()`` so the encoder always matches the
    image pipeline. Pass ``model_override`` / ``backend_override`` to score
    DBs whose cached embeddings came from a different profile than the active
    config (e.g., benchmarking SigLIP-2 embeddings while running the 8gb profile).
    Uses ``AutoProcessor`` for the transformers backend; ``open_clip.get_tokenizer``
    for ``open_clip``.
    """
    import torch
    from config import ScoringConfig

    cfg = ScoringConfig(validate=False).get_clip_config()
    model_name = model_override or cfg.get("model_name")
    backend = backend_override or cfg.get("backend", "open_clip")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "transformers":
        from transformers import AutoProcessor

        from utils.siglip import load_siglip_model

        model = load_siglip_model(model_name, device)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        def encode(texts):
            inputs = processor(text=list(texts), padding="max_length", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.get_text_features(**inputs)
            # SigLIP/SigLIP-2 NaFlex (trust_remote_code) returns BaseModelOutputWithPooling;
            # vanilla SigLIP returns a tensor directly.
            if isinstance(out, torch.Tensor):
                feats = out
            elif getattr(out, "pooler_output", None) is not None:
                feats = out.pooler_output
            else:
                raise RuntimeError(
                    f"Unsupported get_text_features output: {type(out).__name__} "
                    "(no pooler_output and not a tensor). SigLIP-2 NaFlex with "
                    "trust_remote_code is expected to expose pooler_output."
                )
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu().numpy().astype(np.float32)

        return encode

    import open_clip

    pretrained = cfg.get("pretrained", "laion2b_s32b_b82k")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    if device == "cuda":
        model = model.half()
    tokenizer = open_clip.get_tokenizer(model_name)

    def encode(texts):
        tokens = tokenizer(list(texts)).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().astype(np.float32)

    return encode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--photo-dir", default=None, help="Only score photos whose path contains this substring")
    p.add_argument("--model", default=None,
                   help="Override CLIP/SigLIP model name from config (use when DB embeddings don't match active profile)")
    p.add_argument("--backend", default=None, choices=("open_clip", "transformers"),
                   help="Override CLIP/SigLIP backend from config")
    p.add_argument("--dry-run", action="store_true", help="Compute scores but don't write to DB")
    args = p.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    # Probe the cached-embedding dim from the DB. The text encoder must agree.
    with sqlite3.connect(os.fspath(args.db)) as probe:
        row = probe.execute(
            "SELECT clip_embedding FROM photos WHERE clip_embedding IS NOT NULL LIMIT 1"
        ).fetchone()
    if row is None:
        print(f"No photos with cached embeddings in {args.db}", file=sys.stderr)
        return 1
    cached_emb = bytes_to_embedding(row[0])
    cached_dim = cached_emb.shape[0]
    print(f"Cached image embeddings: {cached_dim}-dim")

    from config import ScoringConfig
    full_cfg = ScoringConfig(validate=False).config
    pos_prompts, neg_prompts = load_prompts_from_config(full_cfg)
    print(f"Building aesthetic axis from {len(pos_prompts)}+{len(neg_prompts)} prompts ...")
    encode = _make_text_encoder(model_override=args.model, backend_override=args.backend)
    axis = build_aesthetic_axis(encode, positive_prompts=pos_prompts, negative_prompts=neg_prompts)
    print(f"  axis shape: {axis.shape}")
    if axis.shape[0] != cached_dim:
        print(
            f"Axis dim {axis.shape[0]} does not match cached embedding dim {cached_dim}. "
            "The active CLIP/SigLIP model differs from the one that produced the cached "
            "embeddings — set the matching profile in scoring_config.json before re-running.",
            file=sys.stderr,
        )
        return 1

    with sqlite3.connect(os.fspath(args.db)) as conn:
        conn.row_factory = sqlite3.Row

        where_clauses = ["clip_embedding IS NOT NULL"]
        params: list = []
        if args.photo_dir:
            where_clauses.append("path LIKE ?")
            params.append(f"%{args.photo_dir}%")
        where_sql = " AND ".join(where_clauses)
        rows = conn.execute(
            f"SELECT path, clip_embedding FROM photos WHERE {where_sql}", params,
        ).fetchall()
        print(f"Found {len(rows):,} photos with cached embeddings")

        t0 = time.time()
        BATCH = 1024
        n_written = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            embs_list = []
            valid_batch = []
            for r in batch:
                emb = bytes_to_embedding(r["clip_embedding"], dim=cached_dim)
                if emb is None:
                    continue  # silently skip rows with malformed/short BLOBs
                embs_list.append(emb)
                valid_batch.append(r)
            if not embs_list:
                continue
            embs = np.stack(embs_list)
            scores = score_embeddings(embs, axis)
            if not args.dry_run:
                conn.executemany(
                    "UPDATE photos SET aesthetic_clip = ? WHERE path = ?",
                    [(float(s), r["path"]) for s, r in zip(scores, valid_batch)],
                )
            n_written += len(valid_batch)
        if not args.dry_run:
            conn.commit()
        elapsed = time.time() - t0
        suffix = " (dry-run)" if args.dry_run else ""
        print(f"Scored {n_written:,} photos in {elapsed:.1f}s{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
