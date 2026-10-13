"""Tests for optimization/model_eval.py (--evaluate-models) and eval_common.

Every DB here is a temp file built with init_database; no test touches a
library DB or any photo file.
"""

import gc
import hashlib
import json
import sqlite3
import sys

import numpy as np
import pytest
from PIL import Image

from cli_args import build_parser
from db.schema import init_database
from optimization import eval_common as ec
from optimization import model_eval as me


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


@pytest.fixture
def eval_db(tmp_path):
    """Temp DB: 6 photos, 4 decisive pairs (winner always the higher `aesthetic`) + 1 tie."""
    db = str(tmp_path / 'eval.db')
    init_database(db)
    conn = sqlite3.connect(db)
    for i in range(6):
        conn.execute(
            "INSERT INTO photos (path, filename, aesthetic, topiq_score, aggregate, face_quality) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f'/eval/p{i}.jpg', f'p{i}.jpg', float(i), 5.0, 5.0 + i * 0.1, float(i)))
    # winner has the higher index in every pair
    for a, b, w in [(0, 1, 'b'), (3, 2, 'a'), (4, 5, 'b'), (1, 5, 'b')]:
        conn.execute("INSERT INTO comparisons (photo_a_path, photo_b_path, winner) VALUES (?, ?, ?)",
                     (f'/eval/p{a}.jpg', f'/eval/p{b}.jpg', w))
    conn.execute("INSERT INTO comparisons (photo_a_path, photo_b_path, winner) VALUES ('/eval/p0.jpg','/eval/p2.jpg','tie')")
    conn.execute("INSERT INTO comparisons (photo_a_path, photo_b_path, winner) VALUES ('/eval/p0.jpg','/eval/p3.jpg','skip')")
    conn.commit()
    conn.close()
    # Fold the WAL into the main file so the sha256 checks compare a quiescent DB.
    gc.collect()  # init_database leaves its connection to the garbage collector
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.close()
    return db


PAIRS = [('a', 'b', 'a'), ('c', 'd', 'b'), ('e', 'f', 'a')]


def test_agreement_exactly_one_when_winner_always_higher():
    # winners: a, d, e
    scores = {'a': 2.0, 'b': 1.0, 'c': 1.0, 'd': 2.0, 'e': 9.0, 'f': 0.0}
    res = ec.pairwise_agreement(PAIRS, scores)
    assert res['agreement'] == 1.0
    assert res['coverage'] == 1.0


def test_agreement_exactly_zero_when_winner_always_lower():
    scores = {'a': 1.0, 'b': 2.0, 'c': 2.0, 'd': 1.0, 'e': 0.0, 'f': 9.0}
    assert ec.pairwise_agreement(PAIRS, scores)['agreement'] == 0.0


def test_agreement_half_on_constant_score():
    scores = {k: 3.0 for k in 'abcdef'}
    assert ec.pairwise_agreement(PAIRS, scores)['agreement'] == 0.5


def test_missing_scores_reduce_coverage_not_agreement():
    scores = {'a': 2.0, 'b': 1.0, 'c': None, 'd': 2.0, 'e': 9.0}  # 'f' absent, 'c' None
    res = ec.pairwise_agreement(PAIRS, scores)
    assert res['n_pairs'] == 1 and res['n_total'] == 3
    assert res['coverage'] == pytest.approx(1 / 3)
    assert res['agreement'] == 1.0


def test_no_scoreable_pairs_gives_no_agreement():
    res = ec.pairwise_agreement(PAIRS, {})
    assert res['agreement'] is None and res['coverage'] == 0.0
    assert ec.bootstrap_ci([]) == (None, None)


def test_bootstrap_deterministic_per_seed_and_contains_mean():
    per_pair = [(i * 37 % 70) / 70 for i in range(70)]
    state = np.random.get_state()[1].copy()
    a = ec.bootstrap_ci(per_pair, seed=7)
    assert a == ec.bootstrap_ci(per_pair, seed=7)
    assert a != ec.bootstrap_ci(per_pair, seed=8)
    assert a[0] <= np.mean(per_pair) <= a[1]
    assert (np.random.get_state()[1] == state).all()  # global RNG untouched


def test_fetch_decisive_pairs_separates_ties(eval_db):
    conn = ec.ro_connection(eval_db)
    pairs, n_ties = me.fetch_decisive_pairs(conn)
    conn.close()
    assert len(pairs) == 4 and n_ties == 1  # skip rows are neither


def test_ro_connection_cannot_write(eval_db):
    conn = ec.ro_connection(eval_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM comparisons")
    conn.close()


def test_evaluate_stored_candidates_and_db_unchanged(eval_db, tmp_path):
    before = _sha(eval_db)
    out = str(tmp_path / 'out' / 'r.json')
    report = me.evaluate(eval_db, 'aesthetic',
                         candidates=['stored:aesthetic', 'stored:topiq_score'], out=out, seed=3)
    assert _sha(eval_db) == before
    good = report['candidates']['stored:aesthetic']
    flat = report['candidates']['stored:topiq_score']
    assert good['agreement'] == 1.0 and good['n_pairs'] == 4 and good['coverage'] == 1.0
    assert flat['agreement'] == 0.5
    assert report['n_ties'] == 1 and report['seed'] == 3 and report['slot'] == 'aesthetic'
    assert json.load(open(out))['candidates'].keys() == report['candidates'].keys()


def test_unknown_candidate_and_column_rejected(eval_db, tmp_path):
    out = str(tmp_path / 'r.json')
    with pytest.raises(ValueError):
        me.evaluate(eval_db, 'aesthetic', candidates=['nope'], out=out)
    with pytest.raises(ValueError):
        me.evaluate(eval_db, 'aesthetic', candidates=['stored:thumbnail'], out=out)


def test_face_slots_are_wired_but_deferred(eval_db, tmp_path):
    for slot in ('faces-detect', 'faces-recognize'):
        with pytest.raises(NotImplementedError, match='W5'):
            me.evaluate(eval_db, slot, out=str(tmp_path / 'r.json'))


def test_live_candidate_scores_cached_and_resumed(eval_db, tmp_path, monkeypatch):
    img_dir = tmp_path / 'imgs'
    img_dir.mkdir()
    conn = sqlite3.connect(eval_db)
    for i in range(6):
        p = str(img_dir / f'p{i}.png')
        Image.new('RGB', (32, 32), (i * 10, 0, 0)).save(p)
        conn.execute("UPDATE photos SET path = ? WHERE path = ?", (p, f'/eval/p{i}.jpg'))
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.commit()
    conn.close()
    # the comparisons FK cascade is off for the rewrite above; remap the pair paths too
    conn = sqlite3.connect(eval_db)
    for i in range(6):
        p = str(img_dir / f'p{i}.png')
        for col in ('photo_a_path', 'photo_b_path'):
            conn.execute(f"UPDATE comparisons SET {col} = ? WHERE {col} = ?", (p, f'/eval/p{i}.jpg'))
    conn.commit()
    conn.close()

    monkeypatch.setattr(ec, 'load_original', lambda path: Image.open(path).convert('RGB'))
    monkeypatch.setattr(me, 'load_original', ec.load_original)
    monkeypatch.setattr(ec, 'bound_to_inference_size', lambda img: img)
    monkeypatch.setattr(me, 'bound_to_inference_size', lambda img: img)
    calls = []

    def factory(device):
        def score(pil):
            calls.append(1)
            return float(np.asarray(pil)[0, 0, 0])  # red channel == photo index * 10
        return score

    me.register_candidate('aesthetic', 'stub-red', factory)
    try:
        out = str(tmp_path / 'o' / 'r.json')
        r1 = me.evaluate(eval_db, 'aesthetic', candidates=['stub-red'], out=out, device='cpu')
        assert r1['candidates']['stub-red']['agreement'] == 1.0
        n = len(calls)
        assert n == 6
        r2 = me.evaluate(eval_db, 'aesthetic', candidates=['stub-red'], out=out, device='cpu')
        assert len(calls) == n  # everything came from the sidecar
        assert r2['candidates']['stub-red']['agreement'] == 1.0
        assert r2['candidates']['stub-red']['mean_latency_s'] is not None
    finally:
        me._REGISTRY.pop(('aesthetic', 'stub-red'))


def test_end_to_end_substitutes_candidate_into_aggregate(eval_db, tmp_path):
    out = str(tmp_path / 'r.json')
    before = _sha(eval_db)
    # topiq_score is 5.0 for every photo: substituting it flattens the aggregate
    me.register_candidate('aesthetic', 'stub-flip', me.StoredColumn('topiq_score'))
    try:
        report = me.evaluate(
            eval_db, 'end-to-end',
            candidates=['stored:aggregate', 'aesthetic:stored:aesthetic', 'aesthetic:stub-flip'], out=out)
    finally:
        me._REGISTRY.pop(('aesthetic', 'stub-flip'))
    gc.collect()
    assert _sha(eval_db) == before  # the scorer must not init/migrate the evaluated DB
    cands = report['candidates']
    assert cands['stored:aggregate']['agreement'] == 1.0
    # substituting the photo's own aesthetic reproduces a monotone aggregate
    assert cands['aesthetic:stored:aesthetic']['agreement'] == 1.0
    assert cands['aesthetic:stub-flip']['n_pairs'] == 4
    assert cands['aesthetic:stub-flip']['agreement'] == 0.5


def test_end_to_end_applies_persisted_percentile_snapshot(eval_db):
    """update_all_aggregates normalizes tech_sharpness in memory only; the recompute must too."""
    from config import ScoringConfig
    from config.percentile_normalizer import PercentileNormalizer
    from processing.scorer import Facet
    conn = sqlite3.connect(eval_db)
    conn.execute("UPDATE photos SET raw_sharpness_variance = 400.0, tech_sharpness = 1.0")
    snapshot = {'percentiles': {'raw_sharpness_variance': 200.0}, 'category_percentiles': {},
                'target_percentile': 90, 'photo_count': 6, 'computed_at': 0}
    conn.execute("INSERT INTO stats_cache (key, value, updated_at) VALUES (?, ?, 0)",
                 (PercentileNormalizer.PERSIST_KEY, json.dumps(snapshot)))
    conn.commit()
    conn.close()
    scorer = Facet.__new__(Facet)
    scorer.config = ScoringConfig(None)
    ro = ec.ro_connection(eval_db)
    try:
        norm = me.load_normalizer(ro, scorer.config)
        assert norm is not None
        row = me.fetch_metric_rows(ro, ['/eval/p0.jpg'])['/eval/p0.jpg']
    finally:
        ro.close()
    m = me._production_metrics(scorer, row, norm)
    assert m['tech_sharpness'] == pytest.approx(norm.normalize('raw_sharpness_variance', 400.0))
    assert m['tech_sharpness'] != 1.0
    assert me._production_metrics(scorer, row, None)['tech_sharpness'] == 1.0


def test_end_to_end_rejects_bad_candidate_name(eval_db, tmp_path):
    with pytest.raises(ValueError):
        me.evaluate(eval_db, 'end-to-end', candidates=['bogus'], out=str(tmp_path / 'r.json'))


def test_cli_parses_evaluate_models():
    args = build_parser().parse_args(
        ['--evaluate-models', 'aesthetic', '--candidates', 'a,b', '--eval-out', 'x.json'])
    assert args.evaluate_models == 'aesthetic'
    assert args.candidates == 'a,b' and args.eval_out == 'x.json'
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--evaluate-models', 'nonsense'])


def test_measure_reports_latency_and_rss():
    with ec.measure() as stats:
        sum(range(1000))
    assert stats['latency_s'] >= 0
    assert 'torch' in sys.modules or stats['peak_vram_gb'] is None
