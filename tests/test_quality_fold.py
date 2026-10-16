"""``quality_percent`` is folded into ``aesthetic`` before the weights are applied.

The documented contract is "redistributed into aesthetic" (docs/SCORING.md). The scorer
used to add a flat ``q / a`` to the aesthetic VALUE instead, so for the ten categories with
a non-zero ``quality_percent`` the weight bought a constant, not a share of the aesthetic
score: aesthetic 5.0 scored 3.40 under 60 / 40 instead of 5.0, and a photo that was a 10
on every metric topped out at 9.40 in ``default``.
"""
import json
from pathlib import Path

import pytest

from config import ScoringConfig
from config_resolve import defaults_path
from processing.scorer import Facet, fold_quality_weight


def _facet(config_path):
    f = Facet.__new__(Facet)
    f.config = ScoringConfig(str(config_path))
    return f


def _sixty_forty(tmp_path):
    """A config whose only category weights aesthetic 60 / quality 40 and nothing else."""
    cfg = json.loads(Path(defaults_path()).read_text(encoding='utf-8'))
    default = next(c for c in cfg['categories'] if c['name'] == 'default')
    weights = {k: 0 for k in default['weights']}
    weights.update(aesthetic_percent=60, quality_percent=40)
    cat = {'name': 'default', 'priority': 100, 'filters': {}, 'weights': weights,
           'modifiers': {'bonus': 0}}
    path = tmp_path / 'scoring_config.json'
    path.write_text(json.dumps({'categories': [cat]}), encoding='utf-8')
    return path


def _row(**over):
    """Every metric at 10, no penalty source, no tags: lands in the default category."""
    row = dict(
        aesthetic=10.0, tech_sharpness=10.0, face_quality=10.0, eye_sharpness=10.0,
        face_sharpness=10.0, exposure_score=10.0, color_score=10.0, comp_score=10.0,
        contrast_score=10.0, face_count=0, face_ratio=0.0, is_silhouette=0,
        is_group_portrait=0, is_monochrome=0, is_blink=0, noise_sigma=0.0,
        mean_saturation=0.5, leading_lines_score=0.0, isolation_bonus=1.0,
        mean_luminance=0.5, iso=100, f_stop=8.0, shadow_clipped=0, highlight_clipped=0,
        histogram_spread=60.0, tags='', aesthetic_iaa=10.0, face_quality_iqa=10.0,
        liqe_score=10.0, subject_sharpness=10.0, subject_prominence=10.0,
        subject_placement=10.0, bg_separation=10.0, power_point_score=10.0,
    )
    row.update(over)
    return row


@pytest.mark.parametrize('aesthetic', [0.0, 2.5, 5.0, 7.5, 10.0])
def test_quality_weight_scales_the_aesthetic_score(tmp_path, aesthetic):
    f = _facet(_sixty_forty(tmp_path))
    score, cat = f.calculate_aggregate_logic(_row(aesthetic=aesthetic))
    assert cat == 'default'
    # (60 + 40)% of the aesthetic score: 5.0 -> 5.0 (was 3.40), 10 -> 10.0 (was 6.00)
    assert score == pytest.approx(aesthetic, abs=1e-9)


def test_shipped_default_tops_out_at_ten_when_every_metric_is_ten():
    f = _facet(defaults_path())
    score, cat = f.calculate_aggregate_logic(_row())
    assert cat == 'default'
    assert f.config.get_weights('default')['quality'] > 0
    assert score == pytest.approx(10.0, abs=1e-9)  # was 9.40


def test_fold_quality_weight_moves_quality_into_aesthetic():
    w = {'aesthetic': 0.5, 'quality': 0.1, 'composition': 0.4, 'bonus': 0.3}
    folded = fold_quality_weight(w)
    assert folded == {'aesthetic': 0.6, 'quality': 0.0, 'composition': 0.4, 'bonus': 0.3}
    assert w['quality'] == 0.1 and w['aesthetic'] == 0.5  # the raw dict is untouched
    assert fold_quality_weight({'composition': 1.0})['aesthetic'] == 0.0
