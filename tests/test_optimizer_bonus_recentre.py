"""Applying optimized weights re-centres the category bonus (T9b, defect 5).

The pairwise likelihood is blind to a constant added to every photo of a
category, so an apply moved the category's mean aggregate (+1.03 on the
library's portraits, taking their share of the top 10% from 3.4% to 27.8%).
``apply_optimized_weights`` now writes ``bonus_new = clamp(bonus_old - shift)``
in the same ``write_user_config`` call as the weights, where ``shift`` is the
mean over the category's photos of ``V @ (w_new - w_served)``.
"""

import copy
import json
import logging
from unittest import mock

import numpy as np
import pytest

import api.config as api_config
from config_resolve import load_resolved
from db import get_connection, init_database
from optimization.weight_optimizer import WeightOptimizer
from processing.scorer import build_metric_vector
from config import ScoringConfig

COMPONENTS = WeightOptimizer.SCORE_COMPONENTS


def _seed(db_path, category='portrait', n=12, aesthetic=8.0, liqe=2.0):
    with get_connection(db_path, row_factory=False) as conn:
        for i in range(n):
            conn.execute(
                "INSERT INTO photos (path, filename, category, aesthetic, liqe_score) "
                "VALUES (?, ?, ?, ?, ?)",
                (f'/bonus/{category}{i}.jpg', f'{category}{i}.jpg', category, aesthetic, liqe))
        # photos of another category must not enter the shift
        for i in range(3):
            conn.execute(
                "INSERT INTO photos (path, filename, category, aesthetic, liqe_score) "
                "VALUES (?, ?, 'landscape', 1.0, 9.0)", (f'/bonus/other{i}.jpg', f'other{i}.jpg'))
        conn.commit()


def _config(tmp_path, bonus=0.5, with_modifiers=True):
    entry = {'name': 'portrait', 'weights': {'aesthetic_percent': 50, 'liqe_percent': 50}}
    if with_modifiers:
        entry['modifiers'] = {'bonus': bonus}
    path = tmp_path / 'scoring_config.json'
    path.write_text(json.dumps({'categories': [entry]}))
    return str(path)


def _bonus(path):
    entry = next(c for c in load_resolved(path)['categories'] if c['name'] == 'portrait')
    return entry.get('modifiers', {}).get('bonus')


@pytest.fixture()
def env(tmp_path):
    db_path = str(tmp_path / 'b.db')
    init_database(db_path)
    _seed(db_path)
    return db_path, tmp_path


@pytest.fixture()
def writes():
    """Every write_user_config call, recorded (and still performed)."""
    calls = []
    real = api_config.write_user_config

    def spy(path, data, *args, **kwargs):
        calls.append(copy.deepcopy(data))
        return real(path, data, *args, **kwargs)

    with mock.patch.object(api_config, 'write_user_config', spy):
        yield calls


# aesthetic 8 and liqe 2 on every portrait photo: moving x of weight from liqe to
# aesthetic shifts the mean score by x * (8 - 2)
def test_apply_recentres_category_bonus(env, writes):
    db_path, tmp_path = env
    cfg = _config(tmp_path, bonus=0.5)
    opt = WeightOptimizer(db_path, cfg)
    opt.apply_optimized_weights({'aesthetic': 0.667, 'liqe': 0.333}, 'portrait', backup=False)

    assert len(writes) == 1, 'weights and bonus must land in ONE write'
    entry = next(c for c in writes[0]['categories'] if c['name'] == 'portrait')
    assert entry['weights']['aesthetic_percent'] == pytest.approx(66.7)
    # shift = (0.667 - 0.5) * (8 - 2) = 1.002
    assert entry['modifiers']['bonus'] == pytest.approx(0.5 - 1.0, abs=0.01)
    assert _bonus(cfg) == pytest.approx(-0.5, abs=0.01)


def test_apply_recentres_in_the_other_direction(env, writes):
    db_path, tmp_path = env
    cfg = _config(tmp_path, bonus=0.5)
    WeightOptimizer(db_path, cfg).apply_optimized_weights(
        {'aesthetic': 0.333, 'liqe': 0.667}, 'portrait', backup=False)
    assert _bonus(cfg) == pytest.approx(0.5 + 1.0, abs=0.01)


@pytest.mark.parametrize('old_bonus, new_weights, expected', [
    (4.8, {'aesthetic': 0.333, 'liqe': 0.667}, 5.0),     # shift -1.0 -> 5.8, clamped up
    (-4.8, {'aesthetic': 0.667, 'liqe': 0.333}, -5.0),   # shift +1.0 -> -5.8, clamped down
])
def test_the_bonus_is_clamped_to_the_validated_range(env, writes, caplog, old_bonus, new_weights, expected):
    db_path, tmp_path = env
    cfg = _config(tmp_path, bonus=old_bonus)
    with caplog.at_level(logging.INFO, logger='facet.optimizer'):
        WeightOptimizer(db_path, cfg).apply_optimized_weights(new_weights, 'portrait', backup=False)
    assert len(writes) == 1
    assert _bonus(cfg) == expected
    assert any('residual' in r.getMessage().lower() for r in caplog.records)


def test_apply_without_shift_keeps_bonus(env, writes):
    db_path, tmp_path = env
    cfg = _config(tmp_path, bonus=0.5)
    opt = WeightOptimizer(db_path, cfg)
    opt.apply_optimized_weights(opt._load_current_weights('portrait'), 'portrait', backup=False)
    assert _bonus(cfg) == pytest.approx(0.5, abs=0.005)


def test_an_entry_without_modifiers_is_not_given_one_when_nothing_shifts(env, writes):
    db_path, tmp_path = env
    cfg = _config(tmp_path, with_modifiers=False)
    opt = WeightOptimizer(db_path, cfg)
    opt.apply_optimized_weights(opt._load_current_weights('portrait'), 'portrait', backup=False)
    assert _bonus(cfg) is None


def test_an_entry_without_a_bonus_is_recentred_from_zero(env, writes):
    db_path, tmp_path = env
    cfg = _config(tmp_path, with_modifiers=False)
    WeightOptimizer(db_path, cfg).apply_optimized_weights(
        {'aesthetic': 0.667, 'liqe': 0.333}, 'portrait', backup=False)
    assert _bonus(cfg) == pytest.approx(-1.0, abs=0.01)


def test_a_category_with_quality_percent_is_recentred_against_the_folded_weights(tmp_path, writes):
    """`default` carries quality_percent 6: the shift is measured against the SERVED
    (folded) weights and the bonus lands together with `quality_percent: 0`."""
    db_path = str(tmp_path / 'q.db')
    init_database(db_path)
    _seed(db_path, category='default', aesthetic=9.0, liqe=1.0)
    cfg_path = tmp_path / 'scoring_config.json'
    cfg_path.write_text('{}')
    cfg_path = str(cfg_path)
    opt = WeightOptimizer(db_path, cfg_path)
    w_served = opt._load_current_weights('default')
    assert w_served['aesthetic'] == pytest.approx(0.64)

    # unchanged served weights: nothing shifts, even though the raw aesthetic weight is 0.58
    bonus_old = ScoringConfig(cfg_path, validate=False).get_weights('default').get('bonus', 0.0)
    opt.apply_optimized_weights(dict(w_served), 'default', backup=False)
    assert len(writes) == 1
    entry = next(c for c in writes[0]['categories'] if c['name'] == 'default')
    assert entry['weights']['quality_percent'] == 0
    assert entry['modifiers'].get('bonus', 0.0) == pytest.approx(bonus_old, abs=0.01)

    # moving 10 points aesthetic -> liqe shifts every photo by -0.10 * (9 - 1)
    writes.clear()
    new_w = dict(w_served)
    new_w['aesthetic'] -= 0.10
    new_w['liqe'] += 0.10
    opt2 = WeightOptimizer(db_path, cfg_path)
    served_cfg = ScoringConfig(cfg_path, validate=False)
    opt2.apply_optimized_weights(new_w, 'default', backup=False)
    assert len(writes) == 1
    entry = next(c for c in writes[0]['categories'] if c['name'] == 'default')
    assert entry['weights']['quality_percent'] == 0
    # the 3 'landscape' photos (aesthetic 1, liqe 9) are not part of 'default'
    row = {'aesthetic': 9.0, 'liqe_score': 1.0}
    vec = build_metric_vector(row, served_cfg, 'default')
    shift = sum(vec[c] * (new_w[c] - w_served[c]) for c in COMPONENTS)
    assert shift == pytest.approx(-0.8, abs=0.05)
    assert entry['modifiers']['bonus'] == pytest.approx(bonus_old - shift, abs=0.02)


def test_the_others_category_is_not_recentred(env, writes, caplog):
    """No `categories[]` entry (the v3 fallback): no bonus to adjust, so a logged no-op."""
    db_path, tmp_path = env
    cfg = _config(tmp_path)
    with caplog.at_level(logging.INFO, logger='facet.optimizer'):
        WeightOptimizer(db_path, cfg).apply_optimized_weights(
            {'aesthetic': 0.7, 'liqe': 0.3}, 'others', backup=False)
    data = json.loads(open(cfg).read())
    assert [c['name'] for c in data['categories']] == ['portrait']
    assert 'modifiers' not in data.get('weights', {}).get('others', {})
    assert data['weights']['others']['aesthetic_percent'] == 70.0
    assert _bonus(cfg) == pytest.approx(0.5)
    assert any('not re-centred' in r.getMessage() for r in caplog.records)
    assert np.isclose(sum(v for k, v in data['weights']['others'].items() if k.endswith('_percent')), 100.0)


def test_a_missing_database_applies_the_weights_without_recentring(tmp_path, writes):
    cfg = _config(tmp_path)
    WeightOptimizer(str(tmp_path / 'absent.db'), cfg).apply_optimized_weights(
        {'aesthetic': 0.7, 'liqe': 0.3}, 'portrait', backup=False)
    assert _bonus(cfg) == pytest.approx(0.5)
    assert not (tmp_path / 'absent.db').exists(), 'apply must not create a database'
    assert json.loads(open(cfg).read())['categories'][0]['weights']['aesthetic_percent'] == 70.0
