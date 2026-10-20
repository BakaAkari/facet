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
from db.person_carry_over import box_iou, match_boxes
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


def test_face_slots_have_registered_candidates():
    assert {c.name for c in me.registered_candidates('faces-detect')} >= {'yunet', 'blazeface'}
    assert {c.name for c in me.registered_candidates('faces-recognize')} >= {'sface'}
    cand = me.resolve_candidate('faces-detect', 'yunet@0.8')
    assert cand.arg == 0.8 and cand.factory is me.resolve_candidate('faces-detect', 'yunet').factory
    with pytest.raises(ValueError, match='number'):
        me.resolve_candidate('faces-detect', 'yunet@high')
    with pytest.raises(ValueError, match='ofiq'):
        me.resolve_candidate('face-quality', 'ofiq')


def test_iou_and_one_to_one_matching_exact_counts():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    ref = [(0, 0, 10, 10), (100, 100, 110, 110), (200, 200, 210, 210)]
    # two detections hug ref 0 (one exact), one is ref 1 shifted below IoU 0.5, none for ref 2
    det = [(0, 0, 10, 10), (1, 0, 11, 10), (110, 100, 120, 110)]
    m = match_boxes(ref, det, 0.5)
    assert [(i, j) for i, j, _ in m] == [(0, 0)]   # the second det cannot reuse ref 0
    det2 = [(1, 0, 11, 10), (0, 0, 10, 10), (100, 100, 110, 110)]
    assert sorted((i, j) for i, j, _ in match_boxes(ref, det2, 0.5)) == [(0, 1), (1, 2)]


def test_two_references_cannot_share_one_detection():
    ref = [(0, 0, 10, 10), (2, 0, 12, 10)]
    assert len(match_boxes(ref, [(1, 0, 11, 10)], 0.5)) == 1


def test_size_buckets_edges():
    assert ec.size_bucket((0, 0, 9, 10), 100, 100) == '<1%'      # 0.9%
    assert ec.size_bucket((0, 0, 10, 10), 100, 100) == '1-5%'    # exactly 1%
    assert ec.size_bucket((0, 0, 49, 10), 100, 100) == '1-5%'
    assert ec.size_bucket((0, 0, 50, 10), 100, 100) == '>5%'     # exactly 5%


def test_threshold_grid_f1_known_answer():
    # persons [0, 0, 1, 1]; same pairs (0,1) and (2,3)
    sim = np.array([[1, .9, .4, .1], [.9, 1, .1, .4], [.4, .1, 1, .5], [.1, .4, .5, 1]])
    rows = {r['threshold']: r for r in ec.pairwise_f1_grid(sim, [0, 0, 1, 1], [0.3, 0.45, 0.6])}
    assert rows[0.3]['n_pred'] == 4 and rows[0.3]['tp'] == 2     # P .5, R 1
    assert rows[0.3]['f1'] == pytest.approx(2 / 3)
    assert rows[0.45]['f1'] == 1.0
    assert rows[0.6]['precision'] == 1.0 and rows[0.6]['recall'] == 0.5


def test_score_face_detection_counts_buckets_and_false_positives():
    samples = [('/a.jpg', [(10, 10, 60, 60), (200, 10, 210, 20)]),   # 6.25% face + 0.0125% face
               ('/b.jpg', []), ('/c.jpg', []), ('/gone.jpg', [(0, 0, 5, 5)])]

    def loader(path):
        return None if path == '/gone.jpg' else np.zeros((200, 400, 3), np.uint8)

    def finds_big_only(img):
        return [(12, 12, 62, 62, 0.9)]

    def hallucinates(img):
        return [(10, 10, 60, 60, 0.9), (200, 10, 210, 20, 0.8), (300, 100, 340, 140, 0.8)]

    acc, skipped = me.score_face_detection(samples, loader, {'big': finds_big_only, 'all': hallucinates})
    assert skipped == {'unreadable': 1, 'size_mismatch': 0}
    summary = me.summarize_face_detection(acc, seed=1)
    big, every = summary['big'], summary['all']
    assert big['match_rate']['rate'] == 0.5 and big['match_rate']['n'] == 2
    assert big['match_rate_by_size']['1-5%']['rate'] == 1.0      # 3.1% of the frame
    assert big['match_rate_by_size']['<1%']['rate'] == 0.0       # 0.125%
    assert big['match_rate_by_size']['>5%']['n'] == 0
    assert big['false_positive_rate']['rate'] == 1.0 and big['false_positive_rate']['n'] == 2
    assert every['match_rate']['rate'] == 1.0
    assert every['mean_extra_detections_per_face_photo'] == 1.0
    assert every['mean_detections_per_empty_photo'] == 3.0


def test_size_mismatch_photo_is_skipped():
    acc, skipped = me.score_face_detection(
        [('/x.jpg', [(0, 0, 900, 900)])], lambda p: np.zeros((100, 100, 3), np.uint8), {'d': lambda i: []})
    assert skipped['size_mismatch'] == 1 and acc['d']['ref'] == []


def _toy_embeddings():
    rng = np.random.default_rng(0)
    centres = np.eye(3, 8)
    emb, labels = [], []
    for pid in range(3):
        for _ in range(4):
            v = centres[pid] + rng.normal(0, 0.05, 8)
            emb.append(v / np.linalg.norm(v))
            labels.append(pid)
    return np.array(emb, np.float32), np.array(labels)


def test_sweep_recognition_separates_toy_persons_and_is_deterministic():
    emb, labels = _toy_embeddings()
    a = me.sweep_recognition(emb, labels, me.default_threshold_grid(), seed=3)
    b = me.sweep_recognition(emb, labels, me.default_threshold_grid(), seed=3)
    assert a == b
    assert a['best']['f1'] == 1.0 and a['best']['threshold'] == 0.2   # first of the tied maxima
    assert a['nearest_neighbour_same_person_accuracy'] == 1.0
    assert a['cluster_ari_at_best_threshold'] == pytest.approx(1.0)
    grid = me.default_threshold_grid()
    assert len(grid) == 17 and grid[0] == 0.2 and grid[-1] == 0.6


def _seed_faces(db):
    conn = sqlite3.connect(db)
    for i in range(30):
        conn.execute("INSERT INTO photos (path, filename, aggregate, face_count) VALUES (?,?,?,?)",
                     (f'/f/p{i}.jpg', f'p{i}.jpg', float(i % 10), 1 if i < 20 else 0))
    for pid in (1, 2):
        conn.execute("INSERT INTO persons (id, name) VALUES (?, ?)", (pid, 'Ann' if pid == 1 else None))
    for i in range(20):
        conn.execute(
            "INSERT INTO faces (photo_path, face_index, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            "person_id, embedding_model) VALUES (?,?,?,?,?,?,?,?,?)",
            (f'/f/p{i}.jpg', 0, b'x', 10, 10, 60, 60, 1 + i % 2, 'arcface_buffalo_l'))
    conn.commit()
    conn.close()
    gc.collect()
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.close()


def test_faces_detect_slot_is_seeded_and_leaves_db_untouched(eval_db, tmp_path, monkeypatch):
    _seed_faces(eval_db)
    me.register_candidate('faces-detect', 'stubdet', lambda device, arg: (lambda img: [(10, 10, 60, 60, 0.9)]))
    monkeypatch.setattr(me, '_load_bgr', lambda p: np.zeros((200, 400, 3), np.uint8))
    before = _sha(eval_db)
    reports = [me.evaluate(eval_db, 'faces-detect', candidates=['stubdet'], out=str(tmp_path / f'r{i}.json'),
                           seed=5, limit=10) for i in range(2)]
    assert _sha(eval_db) == before
    for r in (reports[0], reports[1]):
        r['candidates']['stubdet'].pop('mean_latency_s')    # wall clock, the only non-seeded field
    assert reports[0]['candidates'] == reports[1]['candidates']
    assert reports[0]['n_photos'] == 10
    r = reports[0]['candidates']['stubdet']
    assert r['match_rate']['rate'] == 1.0 and r['false_positive_rate']['rate'] == 1.0
    assert json.load(open(tmp_path / 'r0.json'))['slot'] == 'faces-detect'


def test_faces_detect_refuses_a_library_without_insightface_reference(eval_db, tmp_path):
    with pytest.raises(ValueError, match='InsightFace reference'):
        me.evaluate(eval_db, 'faces-detect', candidates=['yunet'], out=str(tmp_path / 'r.json'))


def test_faces_recognize_slot_with_stub_embedder_and_threshold_variant(eval_db, tmp_path, monkeypatch):
    _seed_faces(eval_db)
    axes = {1: np.array([1.0, 0.0]), 0: np.array([0.0, 1.0])}

    def loader(path):
        return np.full((4, 4, 3), int(path.split('/p')[-1].split('.')[0]) % 2, np.uint8)

    def factory(device, arg):
        # person 1 = odd photo index (value 1), person 2 = even (value 0)
        return lambda img, boxes: [axes[int(img[0, 0, 0])] for _ in boxes]

    me.register_candidate('faces-recognize', 'stubemb', factory)
    monkeypatch.setattr(me, '_load_bgr', loader)
    before = _sha(eval_db)
    rep = me.evaluate(eval_db, 'faces-recognize', candidates=['stubemb', 'stubemb@0.5'],
                      out=str(tmp_path / 'r.json'), seed=1, limit=20)
    assert _sha(eval_db) == before
    assert rep['n_reference_faces'] == 16     # 8 per person cap
    cand = rep['candidates']['stubemb']
    assert cand['n_embedded'] == 16
    assert cand['all_persons']['best']['f1'] == 1.0
    assert cand['named_persons_only']['n_persons'] == 1
    op = rep['candidates']['stubemb@0.5']['operating_point']
    assert op['threshold'] == 0.5 and op['f1'] == 1.0


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
    from db.stats_cache import PERCENTILES_SNAPSHOT_KEY
    from processing.scorer import Facet
    conn = sqlite3.connect(eval_db)
    conn.execute("UPDATE photos SET raw_sharpness_variance = 400.0, tech_sharpness = 1.0")
    snapshot = {'percentiles': {'raw_sharpness_variance': 200.0}, 'category_percentiles': {},
                'target_percentile': 90, 'photo_count': 6, 'computed_at': 0}
    conn.execute("INSERT INTO stats_cache (key, value, updated_at) VALUES (?, ?, 0)",
                 (PERCENTILES_SNAPSHOT_KEY, json.dumps(snapshot)))
    conn.commit()
    conn.close()
    scorer = Facet.__new__(Facet)
    scorer.config = ScoringConfig(None)
    ro = ec.ro_connection(eval_db)
    try:
        norm = PercentileNormalizer.from_snapshot(ro, scorer.config.get_normalization_settings())
        assert norm is not None
        row = me.fetch_metric_rows(ro, ['/eval/p0.jpg'])['/eval/p0.jpg']
    finally:
        ro.close()
    m = Facet.prepare_scoring_row(dict(row), scorer.config, norm)
    assert m['tech_sharpness'] == pytest.approx(norm.normalize('raw_sharpness_variance', 400.0))
    assert m['tech_sharpness'] != 1.0
    assert Facet.prepare_scoring_row(dict(row), scorer.config, None)['tech_sharpness'] == 1.0


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


def test_blink_sweep_finds_known_best_threshold():
    # blinks score 0.6-0.9, opens 0.1-0.5: the grid optimum is 0.55 (lowest threshold with F1 == 1)
    scores = [0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.35]
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    res = me.sweep_blink(scores, labels, me.blink_threshold_grid(), seed=42, n_boot=50)
    assert res['best']['threshold'] == 0.55 and res['best']['f1'] == 1.0
    assert res['base_rate'] == 0.4 and len(res['grid']) == 11
    assert res['grid'][0]['threshold'] == 0.3 and res['grid'][-1]['threshold'] == 0.8


def test_blink_slot_stub_scorer_and_db_unchanged(eval_db, tmp_path, monkeypatch):
    conn = sqlite3.connect(eval_db)
    for i in range(6):
        conn.execute("UPDATE photos SET is_blink = ?, face_count = 1 WHERE path = ?", (i % 2, f'/eval/p{i}.jpg'))
        conn.execute("INSERT INTO faces (photo_path, face_index, embedding, embedding_model, bbox_x1, bbox_y1, bbox_x2, bbox_y2) "
                     "VALUES (?, 0, x'00', ?, 0, 0, 2, 2)", (f'/eval/p{i}.jpg', me.REFERENCE_EMBEDDING_MODEL))
    conn.commit()
    conn.close()
    gc.collect()
    conn = sqlite3.connect(eval_db)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.close()
    before = _sha(eval_db)
    monkeypatch.setattr(me, '_load_bgr', lambda path: np.full((4, 4, 3), int(path.split('/p')[-1].split('.')[0]) % 2, np.uint8))
    monkeypatch.setattr(me, '_blink_scorer_factory', lambda: (lambda img, bbox: (0.9, 0.7) if img[0, 0, 0] else (0.1, 0.2)))
    rep = me.evaluate(eval_db, 'blink', out=str(tmp_path / 'b.json'), seed=42)
    assert _sha(eval_db) == before
    assert rep['n_sampled'] == 6 and rep['n_mesh_failed'] == 0
    biased = rep['full_sample_selection_biased']['max_blink']
    assert biased['best']['f1'] == 1.0 and biased['base_rate'] == 0.5
    assert rep['held_out']['selection']['split'] == 'A' and rep['held_out']['report']['split'] == 'B'
    assert len(rep['faces']) == 6 and rep['faces'][0]['yaw'] is None


def test_centroid_sweep_known_half_centroid_outcomes():
    # persons 1 and 2 share one axis (confusable), person 3 sits on its own; person 4 has < 4 faces
    e = np.array([[1, 0, 0]] * 4 + [[0.8, 0.6, 0]] * 4 + [[0, 0, 1]] * 4 + [[0, 1, 0]] * 3, np.float32)
    labels = np.array([1] * 4 + [2] * 4 + [3] * 4 + [4] * 3)
    res = me.centroid_sweep(e, labels, [0.5, 0.9], seed=42)
    assert res['n_persons'] == 3 and res['n_diff_pairs'] == 6
    lo, hi = res['grid']
    assert lo['same_recall'] == 1.0 and lo['diff_pairs_above'] == 2    # 1<->2 both directions, cos 0.8
    assert hi['same_recall'] == 1.0 and hi['diff_pairs_above'] == 0
    assert hi['false_merge_rate'] == 0.0 and hi['f1'] == 1.0
    assert lo['precision'] == 3 / 5


def test_choose_rules_apply_pre_registered_criteria():
    arc = [{'threshold': 0.6, 'false_merge_rate': 0.0}]
    sf = [{'threshold': t, 'false_merge_rate': r} for t, r in ((0.45, 0.2), (0.5, 0.1), (0.55, 0.0), (0.6, 0.0))]
    assert me.choose_centroid_threshold(sf, arc)['threshold'] == 0.55
    assert me.choose_centroid_threshold(sf[:2], arc)['threshold'] is None
    arc_d = [{'distance': 0.15, 'false_merge_rate': 0.01}]
    sf_d = [{'distance': d, 'false_merge_rate': r} for d, r in ((0.2, 0.0), (0.3, 0.01), (0.4, 0.2))]
    assert me.choose_auto_merge_distance(sf_d, arc_d)['distance'] == 0.3


def test_faces_centroid_slot_runs_and_leaves_db_untouched(eval_db, tmp_path, monkeypatch):
    _seed_faces(eval_db)
    axes = {1: np.array([1.0, 0.0], np.float32), 0: np.array([0.0, 1.0], np.float32)}
    conn = sqlite3.connect(eval_db)
    for i in range(20):
        conn.execute("UPDATE faces SET embedding = ? WHERE photo_path = ?",
                     (axes[int(i % 2 == 0)].tobytes(), f'/f/p{i}.jpg'))
    conn.commit()
    conn.close()
    gc.collect()

    def loader(path):
        return np.full((4, 4, 3), int(path.split('/p')[-1].split('.')[0]) % 2, np.uint8)

    me.register_candidate('faces-recognize', 'sface', lambda device, arg: (
        lambda img, boxes: [axes[int(img[0, 0, 0])] for _ in boxes]))
    monkeypatch.setattr(me, '_load_bgr', loader)
    before = _sha(eval_db)
    rep = me.evaluate(eval_db, 'faces-centroid', out=str(tmp_path / 'c.json'), seed=1, limit=20)
    assert _sha(eval_db) == before
    sf = rep['results']['all_persons']['sface']['centroid']
    assert sf['n_persons'] == 2 and sf['grid'][0]['same_recall'] == 1.0
    assert rep['full_sample_selection_biased']['chosen']['merge_threshold']['threshold'] == 0.3
    assert rep['results']['named_persons_only']['sface']['centroid']['n_persons'] == 1
    assert json.load(open(tmp_path / 'c.json'))['slot'] == 'faces-centroid'
