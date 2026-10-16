"""The optimizer starts from, and writes, the weights the scorer SERVES (T9b, G1).

``calculate_aggregate_logic`` folds ``quality_percent`` into ``aesthetic``
before weighting (``fold_quality_weight``). ``_load_current_weights`` used to
read the raw ``<key>_percent`` values and not renormalise, so for the ten
categories with ``quality_percent > 0`` the starting weights summed to 0.94-0.98
and carried the wrong aesthetic weight; ``_apply_weights_locked`` kept
``quality_percent``, so the served aesthetic weight after an apply was
``(new + quality) / (100 + quality)`` instead of what the optimizer fitted.
The oracle throughout is the shipped default config.
"""

import json

import pytest

from config import ScoringConfig
from config_resolve import load_resolved
from optimization.weight_optimizer import WeightOptimizer
from processing.scorer import Facet, build_metric_vector, fold_quality_weight

COMPONENTS = WeightOptimizer.SCORE_COMPONENTS


@pytest.fixture()
def cfg_path(tmp_path):
    """An empty override: the shipped defaults."""
    path = tmp_path / 'scoring_config.json'
    path.write_text('{}')
    return str(path)


def _served(cfg, category):
    folded = fold_quality_weight(cfg.get_weights(category))
    return {c: folded.get(c, 0.0) for c in COMPONENTS}


class TestStartsFromServedWeights:
    @pytest.mark.parametrize('category, aesthetic', [('default', 0.640), ('product', 0.380)])
    def test_quality_is_folded_into_aesthetic(self, cfg_path, category, aesthetic):
        cfg = ScoringConfig(cfg_path, validate=False)
        assert cfg.get_weights(category)['quality'] > 0
        got = WeightOptimizer('unused.db', cfg_path)._load_current_weights(category)
        want = _served(cfg, category)
        assert set(got) == set(COMPONENTS)
        for comp in COMPONENTS:
            assert got[comp] == pytest.approx(want[comp], abs=1e-9), comp
        assert sum(got.values()) == pytest.approx(1.0)
        assert got['aesthetic'] == pytest.approx(aesthetic)

    def test_a_category_without_quality_is_unchanged(self, cfg_path):
        cfg = ScoringConfig(cfg_path, validate=False)
        assert cfg.get_weights('portrait').get('quality', 0) == 0
        got = WeightOptimizer('unused.db', cfg_path)._load_current_weights('portrait')
        for comp in COMPONENTS:
            assert got[comp] == pytest.approx(cfg.get_weights('portrait').get(comp, 0.0), abs=1e-9)
        assert sum(got.values()) == pytest.approx(1.0)

    def test_the_v3_fallback_gets_the_same_fold(self, tmp_path):
        path = tmp_path / 'v3.json'
        path.write_text(json.dumps({'categories': [], 'weights': {'others': {
            'aesthetic_percent': 50, 'liqe_percent': 40, 'quality_percent': 10}}}))
        got = WeightOptimizer('unused.db', str(path))._load_current_weights(None)
        assert got['aesthetic'] == pytest.approx(0.60)
        assert got['liqe'] == pytest.approx(0.40)
        assert sum(got.values()) == pytest.approx(1.0)


def _new_weights(category_weights, move=('color', 'aesthetic'), amount=0.10):
    new = dict(category_weights)
    new[move[0]] += amount
    new[move[1]] -= amount
    return new


def _row():
    """A photo with no penalties and no clipping, so score == sum(w * v) + bonus."""
    return {
        'path': '/row.jpg', 'aesthetic': 6.3, 'comp_score': 4.2, 'exposure_score': 7.5,
        'color_score': 8.1, 'contrast_score': 3.3, 'tech_sharpness': 5.5, 'liqe_score': 6.8,
        'aesthetic_iaa': 4.4, 'histogram_spread': 40.0, 'mean_saturation': 0.4,
        'noise_sigma': 0.0, 'isolation_bonus': 1.0, 'face_count': 0, 'face_ratio': 0.0,
        'tags': '', 'mean_luminance': 0.5, 'is_silhouette': 0, 'shadow_clipped': 0,
        'highlight_clipped': 0, 'is_monochrome': 0,
    }


class TestApplyWritesServedWeights:
    def test_the_served_aesthetic_weight_equals_the_fitted_one(self, cfg_path):
        opt = WeightOptimizer('unused.db', cfg_path)
        w_served = opt._load_current_weights('default')
        new_w = _new_weights(w_served)
        opt.apply_optimized_weights(new_w, 'default', backup=False)

        resolved = load_resolved(cfg_path)
        entry = next(c for c in resolved['categories'] if c['name'] == 'default')
        assert entry['weights']['quality_percent'] == 0
        cfg = ScoringConfig(cfg_path, validate=False)
        folded = _served(cfg, 'default')
        for comp in COMPONENTS:
            assert folded[comp] == pytest.approx(new_w[comp], abs=1e-3), comp

    def test_the_aggregate_is_the_hand_computed_one(self, cfg_path):
        opt = WeightOptimizer('unused.db', cfg_path)
        new_w = _new_weights(opt._load_current_weights('default'))
        opt.apply_optimized_weights(new_w, 'default', backup=False)

        cfg = ScoringConfig(cfg_path, validate=False)
        scorer = Facet.__new__(Facet)
        scorer.config = cfg
        row = _row()
        score, category = scorer.calculate_aggregate_logic(dict(row))
        assert category == 'default'
        vec = build_metric_vector(row, cfg, 'default')
        hand = sum(new_w[c] * max(0.0, min(10.0, vec[c])) for c in COMPONENTS)
        hand += cfg.get_weights('default').get('bonus', 0.0)
        assert score == pytest.approx(hand, abs=0.02)

    def test_applying_the_unchanged_served_weights_is_a_numerical_noop(self, cfg_path):
        before_cfg = ScoringConfig(cfg_path, validate=False)
        scorer = Facet.__new__(Facet)
        scorer.config = before_cfg
        before, _ = scorer.calculate_aggregate_logic(_row())

        opt = WeightOptimizer('unused.db', cfg_path)
        opt.apply_optimized_weights(opt._load_current_weights('default'), 'default', backup=False)

        scorer.config = ScoringConfig(cfg_path, validate=False)
        after, category = scorer.calculate_aggregate_logic(_row())
        assert category == 'default'
        assert abs(after - before) <= 0.02

    def test_a_category_without_quality_percent_gains_none(self, tmp_path):
        path = tmp_path / 'cfg.json'
        path.write_text(json.dumps(
            {'categories': [{'name': 'portrait', 'weights': {'aesthetic_percent': 100}}]}))
        WeightOptimizer('unused.db', str(path)).apply_optimized_weights(
            {'aesthetic': 0.5, 'liqe': 0.5}, 'portrait', backup=False)
        weights = json.loads(path.read_text())['categories'][0]['weights']
        assert 'quality_percent' not in weights
        assert weights['liqe_percent'] == 50.0
