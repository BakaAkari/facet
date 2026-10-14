"""Shared plumbing for the read-only model-evaluation harnesses.

Used by ``optimization/model_eval.py`` (``--evaluate-models``) and
``scripts/qrealign_ab.py``: stratified sampling, resume bookkeeping for slow GPU
runs, the production input bound, and the pairwise-agreement statistics.
Nothing here writes to the library DB.
"""

import contextlib
import json
import os
import random
import sqlite3
import time

import numpy as np

N_DECILES = 10


def stratified_sample_rows(rows, n, seed):
    """Return up to ``n`` rows from ``rows`` spread evenly across aggregate deciles.

    ``rows`` is a sequence of ``(path, aggregate)`` pairs. Rows are ranked by
    aggregate and split into ``N_DECILES`` equal-population buckets, then a
    seeded, roughly even number is drawn from each bucket so the sample covers
    the full quality range rather than clustering near the mean. If a bucket
    has fewer candidates than its share, it is simply under-filled (never
    backfilled from another bucket) so deciles stay honestly representative.
    """
    rows = list(rows)
    if not rows or n <= 0:
        return []
    if n >= len(rows):
        # Taking the whole population — stratification is moot, and the
        # per-bucket remainder math below can under-fill when population
        # doesn't divide evenly across N_DECILES buckets.
        rng = random.Random(seed)
        sample = list(rows)
        rng.shuffle(sample)
        return sample

    order = sorted(range(len(rows)), key=lambda i: rows[i][1])
    n_rows = len(order)
    buckets = [[] for _ in range(N_DECILES)]
    for rank, i in enumerate(order):
        bucket = min(N_DECILES - 1, rank * N_DECILES // n_rows)
        buckets[bucket].append(rows[i])

    rng = random.Random(seed)
    target = min(n, n_rows)
    per_bucket, remainder = divmod(target, N_DECILES)
    sample = []
    for b, bucket in enumerate(buckets):
        take = per_bucket + (1 if b < remainder else 0)
        take = min(take, len(bucket))
        sample.extend(rng.sample(bucket, take))
    rng.shuffle(sample)
    return sample


def partial_path(out_path):
    return f"{out_path}.partial.jsonl"


def ensure_fresh_partial_file(partial_file, resume):
    """Remove a stale ``.partial.jsonl`` sidecar unless ``--resume`` was passed.

    Without ``--resume``, ``score_model`` treats every sample photo as "to
    do" (its ``done`` set is empty) and appends a fresh record for each one
    regardless of what is already on disk. Leftover rows from a prior run —
    possibly against a different ``--db``/``--sample``/``--seed`` that
    happened to share ``--out`` — would then sit alongside the new run's
    rows: ``analyze()`` folds both in (inflating ``n_scored`` and skewing
    ``mean_latency_s``) and repeated non-resume reruns double-count skips.
    Called once, before the model loop, so a non-resume run always starts
    from an empty sidecar. Resuming leaves the file untouched.
    """
    if not resume and os.path.exists(partial_file):
        os.remove(partial_file)


def load_partial_records(partial_file):
    """Parse the partial JSONL sidecar into a list of dicts, skipping bad lines."""
    records = []
    if not os.path.exists(partial_file):
        return records
    with open(partial_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def done_keys(records):
    """{(model, path)} already recorded (scored or skipped) — resume skips these."""
    return {(r.get("model"), r.get("path")) for r in records}


def append_partial(partial_file, record):
    with open(partial_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def filter_todo(sample, model_id, done):
    """Sample paths not yet recorded (scored or skipped) for ``model_id`` — the resume filter."""
    return [path for path, _agg in sample if (model_id, path) not in done]


def ro_connection(db_path):
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def bounded_size(w, h, max_side):
    """Target (w, h) after the production long-edge bound; unchanged if within it."""
    long_edge = max(w, h)
    if long_edge <= max_side:
        return w, h
    scale = max_side / long_edge
    return int(w * scale), int(h * scale)


def bound_to_inference_size(img):
    """Mirror the production input contract before scoring.

    PyIQAScorer bounds every image to _MAX_INFERENCE_SIZE on the long edge
    before any pyiqa metric sees it (models/pyiqa_scorer.py). Feeding this
    harness native-resolution originals instead measured qrealign encoding
    24MP frames whole — 23s/img and ~13GB of activations for a 0.8B model —
    a regime no shipped code path ever enters. Both arms get the same bound
    so the latency half of the ship gate compares what production runs.
    """
    from models.pyiqa_scorer import PyIQAScorer
    from PIL import Image
    w, h = bounded_size(*img.size, PyIQAScorer._MAX_INFERENCE_SIZE)
    if (w, h) == img.size:
        return img
    return img.resize((w, h), Image.LANCZOS)


def load_original(db_path_str):
    """Load a library photo's original from its stored path, or None if unreadable.

    ``map_disk_path`` applies ``viewer.path_mapping`` so a library scanned on a
    NAS can be evaluated from this machine. Imported lazily: ``api.config``
    mints a server secret at import (precedent: facet.py's ``--export-*``).
    """
    from api.config import map_disk_path
    from utils.image_loading import load_image_from_path
    pil_img, _ = load_image_from_path(map_disk_path(db_path_str))
    return pil_img


def pairwise_agreement(pairs, scores):
    """Fraction of pairwise picks a score ranks the same way the user did.

    ``pairs``: iterable of ``(a_path, b_path, winner)``, winner in {'a', 'b'}.
    ``scores``: {path: float|None}. A pair counts 1.0 when the winner scores
    strictly higher, 0.0 when lower, 0.5 on an equal score. A pair with either
    score missing is excluded and shows up as lost coverage.

    Returns ``{'per_pair', 'agreement', 'n_pairs', 'n_total', 'coverage'}``;
    ``agreement`` is None when no pair is scoreable.
    """
    pairs = list(pairs)
    per_pair = []
    for a, b, winner in pairs:
        sa, sb = scores.get(a), scores.get(b)
        if sa is None or sb is None:
            continue
        win, lose = (sa, sb) if winner == 'a' else (sb, sa)
        per_pair.append(1.0 if win > lose else 0.0 if win < lose else 0.5)
    n_total = len(pairs)
    return {
        'per_pair': per_pair,
        'agreement': float(np.mean(per_pair)) if per_pair else None,
        'n_pairs': len(per_pair),
        'n_total': n_total,
        'coverage': len(per_pair) / n_total if n_total else 0.0,
    }


def bootstrap_ci(per_pair, n=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI of the mean of ``per_pair``; (None, None) if empty.

    Uses a local generator, so the result is deterministic per seed and the
    global numpy RNG is never touched.
    """
    values = np.asarray(per_pair, dtype=float)
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@contextlib.contextmanager
def measure():
    """Record wall latency, peak CUDA VRAM and peak RSS of the enclosed block.

    Yields a dict filled on exit: ``latency_s``, ``peak_vram_gb`` (None without
    CUDA/torch) and ``peak_rss_gb`` (None without psutil). RSS is sampled at
    entry and exit, so it is a lower bound on the true peak.
    """
    stats = {'latency_s': None, 'peak_vram_gb': None, 'peak_rss_gb': None}
    torch = None
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            torch = _torch
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    proc = None
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        pass
    rss = [proc.memory_info().rss] if proc else []
    t0 = time.monotonic()
    try:
        yield stats
    finally:
        stats['latency_s'] = time.monotonic() - t0
        if torch is not None:
            stats['peak_vram_gb'] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if proc:
            rss.append(proc.memory_info().rss)
            stats['peak_rss_gb'] = max(rss) / (1024 ** 3)


# ---------------------------------------------------------------------------
# Face-slot helpers (detector match rate, pairwise same-person F1)
# ---------------------------------------------------------------------------

# Bucket edges are the bbox area as a share of the frame.
SIZE_BUCKETS = ('<1%', '1-5%', '>5%')


def size_bucket(bbox, frame_w, frame_h):
    """``'<1%'`` / ``'1-5%'`` / ``'>5%'`` by the box's share of the frame area.

    Lower edges are inclusive: exactly 1% falls in ``1-5%``, exactly 5% in ``>5%``.
    """
    x1, y1, x2, y2 = bbox[:4]
    share = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(frame_w * frame_h)
    if share < 0.01:
        return SIZE_BUCKETS[0]
    if share < 0.05:
        return SIZE_BUCKETS[1]
    return SIZE_BUCKETS[2]


def iou(a, b):
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(reference, detected, threshold=0.5):
    """One-to-one greedy IoU matching; returns ``[(ref_idx, det_idx, iou)]``.

    Candidate pairs are taken in descending IoU (ties broken by lower indices, so
    the result is deterministic), each reference and each detection used once.
    """
    cands = []
    for i, r in enumerate(reference):
        for j, d in enumerate(detected):
            v = iou(r, d)
            if v >= threshold:
                cands.append((-v, i, j))
    cands.sort()
    used_r, used_d, out = set(), set(), []
    for neg, i, j in cands:
        if i in used_r or j in used_d:
            continue
        used_r.add(i)
        used_d.add(j)
        out.append((i, j, -neg))
    return out


def pairwise_f1_grid(sim, labels, grid):
    """Same-person precision / recall / F1 at each cosine threshold in ``grid``.

    ``sim`` is an NxN similarity matrix, ``labels`` the reference person id per
    row. Only the upper triangle (i < j) is scored; a pair is predicted "same"
    when ``sim >= threshold``. F1 is 0.0 when precision + recall is 0.
    """
    sim = np.asarray(sim, dtype=np.float32)
    labels = np.asarray(labels)
    iu, ju = np.triu_indices(len(labels), k=1)
    s = sim[iu, ju]
    same = labels[iu] == labels[ju]
    n_same = int(same.sum())
    rows = []
    for t in grid:
        pred = s >= t
        tp = int((pred & same).sum())
        n_pred = int(pred.sum())
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_same if n_same else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({'threshold': round(float(t), 6), 'precision': p, 'recall': r, 'f1': f1,
                     'tp': tp, 'n_pred': n_pred, 'n_same': n_same})
    return rows
