"""Read-only model evaluation against the user's own pairwise picks.

    python facet.py --evaluate-models aesthetic --candidates stored:topiq_score,qrealign-mini:aesthetic

For each candidate in a *slot* (aesthetic, quality, face-quality, end-to-end) this
measures how often the candidate's score ranks the winner of a decisive
``comparisons`` row above the loser. A candidate either reads a stored ``photos``
column (``stored:<column>``) or scores the original files live; live scores are
cached to a resumable JSONL sidecar next to the report, so a long GPU run
survives interruption and a rerun only scores what is missing.

The DB is opened ``mode=ro`` and never written. Other waves add candidates with
:func:`register_candidate`.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

from optimization.eval_common import (
    append_partial,
    bootstrap_ci,
    bound_to_inference_size,
    done_keys,
    load_original,
    load_partial_records,
    measure,
    pairwise_agreement,
    partial_path,
    ro_connection,
)

logger = logging.getLogger("facet.model_eval")

DEFAULT_SEED = 42
DEFAULT_OUT_DIR = os.path.join('.facet_cache', 'eval')

# Slot -> the ``photos`` column a candidate's value replaces in the end-to-end
# aggregate recompute. ``quality_score`` carries no aggregate weight (it is
# redistributed into ``aesthetic``), so a quality candidate is a no-op there.
SLOT_TARGET_COLUMNS = {
    'aesthetic': 'aesthetic',
    'quality': 'quality_score',
    'face-quality': 'face_quality',
}
# Slots whose candidates (detectors / embedders) arrive with the permissive face pipeline.
DEFERRED_SLOTS = ('faces-detect', 'faces-recognize')
END_TO_END = 'end-to-end'
SLOTS = ('aesthetic', 'quality', 'faces-detect', 'faces-recognize', 'face-quality', END_TO_END)

STORED_PREFIX = 'stored:'
_BLOB_COLUMNS = frozenset({
    'thumbnail', 'clip_embedding', 'caption_embedding', 'histogram_data', 'face_embedding',
})


@dataclass(frozen=True)
class StoredColumn:
    """A candidate that reads an already-computed ``photos`` column."""
    column: str


@dataclass(frozen=True)
class Candidate:
    slot: str
    name: str
    # StoredColumn, or ``factory(device) -> score(pil) -> float`` on the 0-10 scale.
    # A scorer may expose ``close()`` to release its model after the run.
    factory: object


_REGISTRY: dict[tuple[str, str], Candidate] = {}


def register_candidate(slot: str, name: str, scorer_factory) -> Candidate:
    """Register a candidate for ``slot`` (later waves add theirs here)."""
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}; expected one of {SLOTS}")
    cand = Candidate(slot, name, scorer_factory)
    _REGISTRY[(slot, name)] = cand
    return cand


def registered_candidates(slot: str) -> list[Candidate]:
    return [c for (s, _), c in _REGISTRY.items() if s == slot]


def _pyiqa_qrealign_factory(pyiqa_id: str, task: str) -> Callable:
    """Live Q-ReAlign via pyiqa; raw (0, 1) is scaled to 0-10 like production."""
    def factory(device):
        import torch
        from models.pyiqa_scorer import _ensure_pyiqa
        pyiqa = _ensure_pyiqa()
        metric = pyiqa.create_metric(pyiqa_id, device=torch.device(device), as_loss=False)

        def score(pil):
            with torch.no_grad():
                raw = metric(pil, task_=task)
            return float(raw.item() if hasattr(raw, 'item') else raw) * 10.0

        def close():
            nonlocal metric
            if hasattr(metric, 'cpu'):
                metric.cpu()
            metric = None
            if device == 'cuda':
                torch.cuda.empty_cache()

        score.close = close
        return score
    return factory


def _native_qrealign_factory(size: str, task: str) -> Callable:
    """Q-ReAlign through plain transformers (no pyiqa); scores are already 0-10."""
    def factory(device):
        from models.qrealign_native import QReAlignNativeScorer
        scorer = QReAlignNativeScorer(size=size, device=device).load()

        def score(pil):
            return scorer.score(pil, task=task)

        score.close = scorer.unload
        return score
    return factory


def _register_builtins():
    stored = {
        'aesthetic': ('aesthetic', 'topiq_score', 'aesthetic_iaa', 'qrealign_score',
                      'aesthetic_clip', 'aesthetic_v25', 'deqa_score'),
        'quality': ('quality_score', 'topiq_score', 'liqe_score', 'qrealign_score'),
        'face-quality': ('face_quality', 'face_quality_iqa'),
        END_TO_END: ('aggregate',),
    }
    for slot, columns in stored.items():
        for col in columns:
            register_candidate(slot, f'{STORED_PREFIX}{col}', StoredColumn(col))
    # Both heads are offered in both slots: whether the quality head ranks
    # aesthetic picks better than the aesthetic head is part of the question.
    for size in ('mini', 'lite'):
        for task in ('aesthetic', 'quality'):
            factory = _pyiqa_qrealign_factory(f'qrealign-{size}', task)
            for slot in ('aesthetic', 'quality'):
                register_candidate(slot, f'qrealign-{size}:{task}', factory)
            native = _native_qrealign_factory(size, task)
            for slot in ('aesthetic', 'quality'):
                register_candidate(slot, f'qrealign-native-{size}:{task}', native)


_register_builtins()


def fetch_decisive_pairs(conn):
    """(pairs, n_ties): decisive ``(a_path, b_path, winner)`` rows and the tie count."""
    pairs = [(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT photo_a_path, photo_b_path, winner FROM comparisons WHERE winner IN ('a', 'b')")]
    n_ties = conn.execute("SELECT COUNT(*) FROM comparisons WHERE winner = 'tie'").fetchone()[0]
    return pairs, n_ties


def _chunks(items, size=500):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_stored_column(conn, column, paths):
    """{path: float|None} for one ``photos`` column."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(photos)")}
    if column not in cols or column in _BLOB_COLUMNS:
        raise ValueError(f"stored candidate column {column!r} is not a scalar photos column")
    out = {}
    for chunk in _chunks(list(paths)):
        marks = ','.join('?' * len(chunk))
        for path, val in conn.execute(
                f"SELECT path, {column} FROM photos WHERE path IN ({marks})", chunk):
            out[path] = float(val) if isinstance(val, (int, float)) else None
    return out


def fetch_metric_rows(conn, paths):
    """{path: metrics dict} (blob columns excluded) with scoring overrides applied."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(photos)") if r[1] not in _BLOB_COLUMNS]
    select = ', '.join(cols)
    rows = {}
    for chunk in _chunks(list(paths)):
        marks = ','.join('?' * len(chunk))
        for row in conn.execute(f"SELECT {select} FROM photos WHERE path IN ({marks})", chunk):
            rows[row['path']] = dict(row)
        for row in conn.execute(
                "SELECT photo_path, scoring_context, category_override "
                f"FROM photo_scoring_overrides WHERE photo_path IN ({marks})", chunk):
            if row[0] in rows:
                rows[row[0]]['scoring_context'] = row[1]
                rows[row[0]]['category_override'] = row[2]
    return rows


def resolve_candidate(slot: str, name: str) -> Candidate:
    """Registered candidate, or an ad-hoc ``stored:<column>``."""
    cand = _REGISTRY.get((slot, name))
    if cand is not None:
        return cand
    if name.startswith(STORED_PREFIX):
        return Candidate(slot, name, StoredColumn(name[len(STORED_PREFIX):]))
    known = ', '.join(c.name for c in registered_candidates(slot)) or '(none)'
    raise ValueError(f"unknown candidate {name!r} for slot {slot!r}; registered: {known}")


def score_live(name, factory, paths, partial_file, device):
    """Score ``paths`` from the originals with a live scorer, resuming from the sidecar.

    Returns ``({path: score|None}, run_stats, mean_latency_s)``. Photos that were
    recorded earlier (scored or skipped) are not rescored; ``run_stats`` is
    empty when nothing needed scoring.
    """
    done = done_keys(load_partial_records(partial_file))
    todo = [p for p in paths if (name, p) not in done]
    run_stats = {}
    if todo:
        logger.info("[%s] loading scorer on %s (%d/%d photos to score)", name, device, len(todo), len(paths))
        with measure() as run_stats:
            score = factory(device)
            try:
                for i, path in enumerate(todo, 1):
                    img = load_original(path)
                    if img is None:
                        append_partial(partial_file, {"model": name, "path": path, "score": None, "skipped": True})
                        continue
                    t0 = time.monotonic()
                    try:
                        value = score(bound_to_inference_size(img))
                    except Exception as e:
                        logger.warning("[%s] inference failed on %s: %s", name, path, e)
                        append_partial(partial_file, {
                            "model": name, "path": path, "score": None, "skipped": True, "error": str(e)})
                        continue
                    append_partial(partial_file, {
                        "model": name, "path": path, "score": float(value),
                        "latency_s": time.monotonic() - t0})
                    if i % 50 == 0:
                        logger.info("[%s] %d/%d", name, i, len(todo))
            finally:
                close = getattr(score, 'close', None)
                if close:
                    close()
    wanted = set(paths)
    scores, latencies = {}, []
    for rec in load_partial_records(partial_file):
        if rec.get("model") != name or rec.get("path") not in wanted:
            continue
        scores[rec["path"]] = rec.get("score")
        if rec.get("score") is not None and rec.get("latency_s") is not None:
            latencies.append(rec["latency_s"])
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    return scores, run_stats, mean_latency


def _candidate_scores(conn, cand, paths, partial_file, device):
    """(scores, run_stats, mean_latency_s) for one candidate on 0-10 scale."""
    if isinstance(cand.factory, StoredColumn):
        return fetch_stored_column(conn, cand.factory.column, paths), {}, None
    return score_live(cand.name, cand.factory, paths, partial_file, device)


# Normalized column -> the raw column ``update_all_aggregates`` rebuilds it from.
_NORMALIZED_COLUMNS = {
    'tech_sharpness': 'raw_sharpness_variance',
    'color_score': 'raw_color_entropy',
    'eye_sharpness': 'raw_eye_sharpness',
}


def load_normalizer(conn, cfg):
    """The percentile normalizer the last ``--recompute-average`` used, or None.

    ``update_all_aggregates`` rewrites tech_sharpness / color_score /
    eye_sharpness from their raw columns in memory, never on disk, so the stored
    values alone do not reproduce the stored aggregate. Rebuilt read-only from
    the ``percentiles_snapshot`` it persists to ``stats_cache``.
    """
    from config.percentile_normalizer import PercentileNormalizer
    settings = cfg.get_normalization_settings()
    if settings.get('method') != 'percentile':
        return None
    snap = PercentileNormalizer.load_persisted(conn)
    if not snap:
        logger.warning("No persisted percentile snapshot: the end-to-end recompute runs "
                       "un-normalized and will not reproduce stored aggregates")
        return None
    norm = PercentileNormalizer(None, target_percentile=snap.get('target_percentile', 95),
                                per_category=settings.get('per_category', False))
    norm.percentiles = snap.get('percentiles') or {}
    norm.category_percentiles = snap.get('category_percentiles') or {}
    return norm


def _production_metrics(scorer, row, normalizer):
    """Mirror ``update_all_aggregates``' in-memory rewrites before the aggregate."""
    m = dict(row)
    m.setdefault('category_override', None)
    m.setdefault('scoring_context', None)
    if normalizer is not None:
        category = scorer._determine_photo_category(m, scorer.config)
        for col, raw_col in _NORMALIZED_COLUMNS.items():
            raw = m.get(raw_col)
            if not isinstance(raw, (int, float)):
                continue
            value = (normalizer.normalize_with_category(raw_col, raw, category)
                     if normalizer.per_category else normalizer.normalize(raw_col, raw))
            if value is not None:
                m[col] = value
    min_faces = scorer.config.get_face_detection_settings().get('min_faces_for_group', 4)
    m['is_group_portrait'] = 1 if (m.get('face_count') or 0) >= min_faces else 0
    return m


def baseline_match_rate(conn, scorer, paths, normalizer, tol=0.01):
    """Fraction of photos whose recomputed, unsubstituted aggregate equals the stored one."""
    rows = fetch_metric_rows(conn, paths)
    hits = n = 0
    for row in rows.values():
        if row.get('aggregate') is None:
            continue
        agg = scorer.calculate_aggregate_logic(_production_metrics(scorer, row, normalizer))[0]
        n += 1
        hits += abs(round(agg, 2) - row['aggregate']) <= tol
    return hits / n if n else None


def _end_to_end_scores(conn, base_slot, cand, paths, partial_file, device, scorer, normalizer=None):
    """Aggregate per photo with the candidate's value substituted into its slot's column.

    ``base_slot`` None means the candidate is the stored aggregate itself.
    """
    if base_slot is None:
        return _candidate_scores(conn, cand, paths, partial_file, device)
    values, run_stats, mean_latency = _candidate_scores(conn, cand, paths, partial_file, device)
    rows = fetch_metric_rows(conn, paths)
    target = SLOT_TARGET_COLUMNS[base_slot]
    scores = {}
    for path in paths:
        value, row = values.get(path), rows.get(path)
        if value is None or row is None:
            scores[path] = None
            continue
        metrics = _production_metrics(scorer, row, normalizer)
        metrics[target] = value
        scores[path] = scorer.calculate_aggregate_logic(metrics)[0]
    return scores, run_stats, mean_latency


def _resolve_end_to_end(name):
    """``stored:aggregate`` or ``<slot>:<candidate>`` (e.g. ``aesthetic:qrealign-mini:aesthetic``)."""
    if name == f'{STORED_PREFIX}aggregate':
        return None, resolve_candidate(END_TO_END, name)
    base_slot, _, rest = name.partition(':')
    if base_slot not in SLOT_TARGET_COLUMNS or not rest:
        raise ValueError(
            f"end-to-end candidate {name!r} must be 'stored:aggregate' or '<slot>:<candidate>' "
            f"with slot in {tuple(SLOT_TARGET_COLUMNS)}")
    return base_slot, resolve_candidate(base_slot, rest)


def _default_candidates(slot):
    if slot == END_TO_END:
        names = [f'{STORED_PREFIX}aggregate']
        for base in SLOT_TARGET_COLUMNS:
            if base == 'quality':
                continue  # a no-op substitution: would only re-run live models for the baseline
            names += [f'{base}:{c.name}' for c in registered_candidates(base)]
        return names
    return [c.name for c in registered_candidates(slot)]


def evaluate(db_path, slot, candidates=None, out=None, seed=DEFAULT_SEED, config_path=None, device=None):
    """Run the evaluation and write the JSON report; returns the report dict."""
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}; expected one of {SLOTS}")
    if slot in DEFERRED_SLOTS:
        raise NotImplementedError(
            f"slot {slot!r} candidates are registered in W5 (permissive face pipeline); "
            "nothing to evaluate yet")
    names = list(candidates) if candidates else _default_candidates(slot)
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    if out is None:
        out = os.path.join(DEFAULT_OUT_DIR, f"{slot}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    partial_file = partial_path(os.path.join(os.path.dirname(os.path.abspath(out)), 'live-scores'))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    scorer = None
    if slot == END_TO_END:
        from config import ScoringConfig
        from processing.scorer import Facet
        # Bypass Facet.__init__: it ends in init_database(db_path), which migrates
        # the schema and flips journal_mode to WAL — a write to the evaluated DB.
        # calculate_aggregate_logic only needs ``config``.
        scorer = Facet.__new__(Facet)
        scorer.config = ScoringConfig(config_path)

    conn = ro_connection(db_path)
    try:
        pairs, n_ties = fetch_decisive_pairs(conn)
        paths = sorted({p for a, b, _ in pairs for p in (a, b)})
        logger.info("%s: %d decisive pairs over %d photos (%d ties set aside)",
                    slot, len(pairs), len(paths), n_ties)
        normalizer = match_rate = None
        if scorer is not None:
            normalizer = load_normalizer(conn, scorer.config)
            match_rate = baseline_match_rate(conn, scorer, paths, normalizer)
            logger.info("Recomputed baseline reproduces the stored aggregate for %s of photos "
                        "(the rest drifted with config/category changes since the last recompute)",
                        'n/a' if match_rate is None else f'{match_rate:.1%}')
        results = {}
        for name in names:
            if slot == END_TO_END:
                base_slot, cand = _resolve_end_to_end(name)
                scores, run_stats, mean_latency = _end_to_end_scores(
                    conn, base_slot, cand, paths, partial_file, device, scorer, normalizer)
            else:
                scores, run_stats, mean_latency = _candidate_scores(
                    conn, resolve_candidate(slot, name), paths, partial_file, device)
            agree = pairwise_agreement(pairs, scores)
            lo, hi = bootstrap_ci(agree['per_pair'], seed=seed)
            results[name] = {
                'agreement': agree['agreement'],
                'ci_low': lo,
                'ci_high': hi,
                'n_pairs': agree['n_pairs'],
                'coverage': agree['coverage'],
                'mean_latency_s': mean_latency,
                'peak_vram_gb': run_stats.get('peak_vram_gb'),
                'peak_rss_gb': run_stats.get('peak_rss_gb'),
            }
    finally:
        conn.close()

    report = {
        'slot': slot,
        'db': os.path.abspath(db_path),
        'seed': seed,
        'n_decisive_pairs': len(pairs),
        'n_ties': n_ties,
        'candidates': results,
    }
    if slot == END_TO_END:
        report['baseline_match_rate'] = match_rate
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    log_report(report)
    logger.info("Report written to %s", out)
    return report


def log_report(report):
    def fmt(v, spec):
        return 'n/a' if v is None else format(v, spec)

    logger.info("%-34s %9s %17s %7s %6s %8s %8s %8s", "candidate", "agree", "95% CI", "pairs",
                "cover", "lat(s)", "VRAM", "RSS")
    for name, r in report['candidates'].items():
        ci = f"[{fmt(r['ci_low'], '.3f')}, {fmt(r['ci_high'], '.3f')}]"
        logger.info("%-34s %9s %17s %7d %6s %8s %8s %8s", name, fmt(r['agreement'], '.3f'), ci,
                    r['n_pairs'], fmt(r['coverage'], '.2f'), fmt(r['mean_latency_s'], '.2f'),
                    fmt(r['peak_vram_gb'], '.2f'), fmt(r['peak_rss_gb'], '.2f'))
