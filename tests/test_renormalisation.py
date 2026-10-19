"""W4 of issue #185: commercial-mode renormalisation happens at scoring time only."""
import json
import sqlite3

import pytest

import licensing
from config import ScoringConfig
from db.schema import init_database
from db.weight_snapshots import record_weight_snapshot
from processing.scorer import Facet, _calculate_scoring_penalties, SCORING_METRIC_KEYS, build_metric_vector, renormalise_weights

DEFAULT_CONFIG = str(__import__('pathlib').Path(__file__).resolve().parent.parent
                     / 'config' / 'scoring_config.default.json')

BASE = dict(
    aesthetic=6.3, tech_sharpness=7.1, face_quality=6.0, eye_sharpness=5.5, face_sharpness=6.2,
    exposure_score=7.3, color_score=6.1, comp_score=6.7, contrast_score=5.4, face_count=0,
    face_ratio=0.0, is_silhouette=0, is_group_portrait=0, is_monochrome=0, is_blink=0,
    noise_sigma=1.7, histogram_bimodality=1.0, mean_saturation=0.45, leading_lines_score=0.0,
    isolation_bonus=1.3, mean_luminance=0.5, iso=400, shutter_speed=0.005, focal_length=50,
    f_stop=4.0, shadow_clipped=0, highlight_clipped=0, histogram_spread=300.0, tags='',
    aesthetic_iaa=6.9, face_quality_iqa=4.8, liqe_score=7.7, subject_sharpness=6.6,
    subject_prominence=5.9, subject_placement=6.1, bg_separation=4.4, power_point_score=5.2,
    quality_score=5.0, scoring_model='clip-mlp',
)
VARIANTS = {
    'plain': {},
    'portrait': dict(face_count=1, face_ratio=0.25, tags='portrait'),
    'group': dict(face_count=3, face_ratio=0.2, is_group_portrait=1, tags='portrait'),
    'blink': dict(face_count=1, face_ratio=0.3, is_blink=1),
    'landscape': dict(tags='landscape,mountain', iso=100, f_stop=11.0),
    'night': dict(tags='night', iso=3200, mean_luminance=0.1, noise_sigma=6.0),
    'mono': dict(is_monochrome=1, tags='street'),
    'clip': dict(shadow_clipped=1, highlight_clipped=1, liqe_score=3.3, aesthetic_iaa=8.1),
}
# Captured by running the pre-W4 calculate_aggregate_logic (HEAD 291488f6).
# The default / landscape / night entries (the categories whose quality_percent is
# non-zero) were re-captured when quality_percent became a real aesthetic weight instead
# of a flat constant: (a + q) * aes replaces a * aes + q, so each moves by q * (aes - 1)
# with aes = 6.3 here: default +0.318 (q = 0.06), landscape and night +0.159 (q = 0.03).
# Every category with no quality weight keeps its pre-W4 value to the last digit.
# Numeric-domain correction: night ISO 3200 reaches the existing
# +0.5*log2(3200/800) = +1 sharpness at weight .07, adding .07.
# The scoring formula, category and every other strict golden are unchanged.
GOLDENS = {
    'plain': (6.345999999999998, 'default'),
    'portrait': (6.748999999999999, 'portrait'),
    'group': (6.673999999999999, 'group_portrait'),
    'blink': (3.5839999999999996, 'portrait'),
    'landscape': (6.036, 'landscape'),
    'night': (4.916, 'night'),
    'mono': (7.393, 'urban'),
    'clip': (3.799999999999999, 'default'),
}


def _facet():
    f = Facet.__new__(Facet)
    f.config = ScoringConfig(DEFAULT_CONFIG)
    return f


def _photo(name):
    m = dict(BASE)
    m.update(VARIANTS[name])
    return m


@pytest.fixture()
def default_mode(monkeypatch):
    monkeypatch.delenv(licensing.LICENSING_MODE_ENV, raising=False)


@pytest.fixture()
def commercial_mode(monkeypatch):
    monkeypatch.setenv(licensing.LICENSING_MODE_ENV, 'commercial')


@pytest.mark.parametrize('name', sorted(GOLDENS))
def test_default_mode_aggregate_matches_goldens(default_mode, name):
    assert _facet().calculate_aggregate_logic(_photo(name)) == GOLDENS[name]


def test_commercial_effective_weights_sum_to_one(commercial_mode):
    f = _facet()
    gone = licensing.unavailable_metrics(licensing.resolve_policy(f.config.config))
    assert {'liqe', 'aesthetic_iaa'} <= gone
    for cat in ('default', 'portrait', 'landscape', 'night'):
        raw = f.config.get_weights(cat)
        eff = renormalise_weights(raw, gone)
        from processing.scorer import _WEIGHTED_METRIC_KEYS
        assert abs(sum(v for k, v in eff.items() if k in _WEIGHTED_METRIC_KEYS) - 1.0) < 1e-12
        assert all(eff[k] == 0.0 for k in gone if k in eff)
        # the raw dict is untouched
        assert raw['liqe'] == f.config.get_weights(cat)['liqe'] > 0


def test_commercial_score_with_null_iqa_matches_manual_renormalisation(commercial_mode):
    f = _facet()
    gone = licensing.unavailable_metrics(licensing.resolve_policy(f.config.config))
    m = _photo('plain')
    m.update(aesthetic_iaa=None, liqe_score=None, face_quality_iqa=None, noise_sigma=0.5)
    score, cat = f.calculate_aggregate_logic(m)
    w = dict(f.config.get_weights(cat))
    w['aesthetic'] = w.get('aesthetic', 0.0) + w.get('quality', 0.0)  # quality is folded into aesthetic
    vec = build_metric_vector(m, f.config, cat)
    kept = {k: w.get(k, 0.0) for k in vec if k not in gone and w.get(k, 0.0) > 0}
    rem = sum(kept.values())
    manual = sum(max(0.0, min(10.0, vec[k])) * v / rem for k, v in kept.items()) + w.get('bonus', 0.0)
    pen = _calculate_scoring_penalties(m, f.config)
    assert pen['noise_penalty'] == 0 and pen['oversaturation_penalty'] == 0
    assert not set(kept) & gone
    assert score == pytest.approx(manual, abs=1e-9)
    # and the vector omits (does not neutralise) those metrics
    assert not set(build_metric_vector(m, f.config, cat, unavailable=gone)) & gone


def test_optimizer_vector_keeps_every_metric_even_in_commercial_mode(commercial_mode):
    f = _facet()
    vec = build_metric_vector(_photo('plain'), f.config, 'default')
    assert {'liqe', 'aesthetic_iaa', 'face_quality_iqa'} <= set(vec)
    assert set(SCORING_METRIC_KEYS) <= set(vec)


def test_snapshot_round_trip_keeps_liqe_percent(commercial_mode, tmp_path):
    f = _facet()
    raw_cat = next(c for c in f.config.config['categories'] if c['name'] == 'default')
    assert raw_cat['weights']['liqe_percent'] > 0
    db = str(tmp_path / 'snap.db')
    init_database(db)
    weights = dict(raw_cat['weights'])
    sid = record_weight_snapshot('default', weights, created_by='test', db=db)
    conn = sqlite3.connect(db)
    try:
        stored = json.loads(conn.execute(
            'SELECT weights FROM weight_config_snapshots WHERE id = ?', (sid,)).fetchone()[0])
    finally:
        conn.close()
    assert stored['liqe_percent'] == raw_cat['weights']['liqe_percent']
    assert stored == weights


def test_version_hash_differs_between_modes(monkeypatch):
    monkeypatch.delenv(licensing.LICENSING_MODE_ENV, raising=False)
    default_hash = ScoringConfig(DEFAULT_CONFIG).version_hash
    monkeypatch.setenv(licensing.LICENSING_MODE_ENV, 'commercial')
    assert ScoringConfig(DEFAULT_CONFIG).version_hash != default_hash
