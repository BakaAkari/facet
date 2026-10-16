"""Every key under ``thresholds`` and every category ``modifiers`` key in the shipped
defaults must be one that code reads. The next dead key fails here instead of sitting in
the config (and the docs, and seven translations) for releases.

Nine such keys, ten instances, shipped for a long time with no reader: five
``thresholds.*``, ``iso_tolerance_multiplier`` (concert and sports),
``min_saturation_bonus``, ``contrast_bonus``, and ``penalties.oversaturation_pixel_percent``.
"""
import json
from pathlib import Path

DEFAULTS = Path(__file__).resolve().parent.parent / 'config' / 'scoring_config.default.json'

# Read by code: ScoringConfig.get_threshold('blink_penalty_percent') in
# Facet.calculate_aggregate_logic; the modifier keys are read off the weights dict there.
READ_THRESHOLD_KEYS = {'blink_penalty_percent'}
READ_MODIFIER_KEYS = {
    'bonus',
    'noise_tolerance_multiplier',
    '_clipping_multiplier',
    '_skip_clipping_penalty',
    '_skip_oversaturation_penalty',
    '_apply_blink_penalty',
}
# Read by _calculate_scoring_penalties
READ_PENALTY_KEYS = {
    'noise_sigma_threshold',
    'noise_max_penalty_points',
    'noise_penalty_per_sigma',
    'leading_lines_blend_percent',
    'oversaturation_threshold',
    'oversaturation_penalty_points',
}


def _defaults():
    return json.loads(DEFAULTS.read_text(encoding='utf-8'))


def test_every_threshold_key_is_read():
    extra = set(_defaults()['thresholds']) - READ_THRESHOLD_KEYS
    assert not extra, f"thresholds keys with no reader: {sorted(extra)}"


def test_every_category_modifier_key_is_read():
    extra = {}
    for cat in _defaults()['categories']:
        bad = set(cat.get('modifiers', {})) - READ_MODIFIER_KEYS
        if bad:
            extra[cat['name']] = sorted(bad)
    assert not extra, f"category modifiers with no reader: {extra}"


def test_every_penalty_key_is_read():
    extra = set(_defaults()['penalties']) - READ_PENALTY_KEYS
    assert not extra, f"penalties keys with no reader: {sorted(extra)}"


def test_get_thresholds_is_gone():
    from config.scoring_config import ScoringConfig
    assert not hasattr(ScoringConfig, 'get_thresholds')
    assert hasattr(ScoringConfig, 'get_threshold')
