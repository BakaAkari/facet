"""Synthetic parity with the real native scoring vector; no models or files."""

from pathlib import Path

import pytest

from config.scoring_config import ScoringConfig
from processing.scorer import Facet
from api.routers.critique import _calculate_breakdown


@pytest.fixture
def config():
    return ScoringConfig(
        str(Path(__file__).parents[1] / "config/scoring_config.default.json"),
        validate=False,
    )


@pytest.mark.parametrize(
    ("metric", "photo", "expected"),
    [
        ("noise", {"noise_sigma": 2.0}, 8.6),
        ("isolation", {"isolation_bonus": 2.0, "f_stop": 8.0}, 5.0),
        ("dynamic_range", {"histogram_spread": 48.0, "dynamic_range_stops": 3.0}, 8.0),
    ],
)
def test_breakdown_uses_native_quality_value(config, metric, photo, expected):
    config.get_weights = lambda category: {metric: 1.0}
    native = Facet.__new__(Facet)
    native.config = config
    aggregate, category = native.calculate_aggregate_logic(photo)
    assert aggregate == pytest.approx(expected)
    breakdown = _calculate_breakdown(photo, config, category)
    assert len(breakdown) == 1
    assert breakdown[0]["value"] == pytest.approx(expected)
    assert breakdown[0]["contribution"] == pytest.approx(aggregate)


def test_quality_weight_folds_into_native_aesthetic(config):
    config.get_weights = lambda category: {"aesthetic": 0.4, "quality": 0.6}
    terms = _calculate_breakdown(
        {"aesthetic": 8.0, "quality_score": 2.0}, config, "default"
    )
    assert [
        (term["metric_key"], term["weight"], term["contribution"]) for term in terms
    ] == [("aesthetic", 1.0, 8.0)]


def test_unavailable_weights_use_native_renormalization(config, monkeypatch):
    monkeypatch.setattr(
        "processing.scorer.unavailable_for_config", lambda cfg: frozenset({"noise"})
    )
    config.get_weights = lambda category: {"aesthetic": 0.4, "noise": 0.6}
    terms = _calculate_breakdown({"aesthetic": 8.0}, config, "default")
    assert len(terms) == 1 and terms[0]["contribution"] == 8.0


def test_normalized_current_calculation_does_not_erase_legacy_stored_score(config):
    from api.routers.critique import _build_rule_critique
    from config.percentile_normalizer import PercentileNormalizer
    from copy import deepcopy

    normalizer = PercentileNormalizer(None)
    normalizer.percentiles = {"raw_sharpness_variance": 100.0}
    config.get_weights = lambda category: {"tech_sharpness": 1.0}
    photo = {
        "raw_sharpness_variance": 20.0,
        "tech_sharpness": 1.0,
        "aggregate": 7.0,
        "config_version": "legacy",
    }
    frozen = deepcopy(photo)
    result = _build_rule_critique(photo, config=config, normalizer=normalizer)
    assert photo == frozen
    assert result["current_formula_score"] == 2.0
    assert result["aggregate"] == 7.0
    assert result["stored_score_state"] == "present"
    assert result["stored_score_matches_current_formula"] is False


@pytest.mark.parametrize("bonus", [4.0, -20.0, 0.007])
def test_net_adjustments_reconcile_exact_displayed_cents(config, bonus):
    from api.routers.critique import _build_rule_critique
    from decimal import Decimal

    config.get_weights = lambda category: {"noise": 1.0, "bonus": bonus}
    result = _build_rule_critique({"noise_sigma": 2.0}, config=config)
    displayed = sum(
        (Decimal(str(term["contribution"])) for term in result["breakdown"]), Decimal(0)
    )
    assert displayed + Decimal(str(result["native_adjustment"])) == Decimal(
        str(result["current_formula_score"])
    )
    assert 0 <= result["current_formula_score"] <= 10


def test_override_is_not_described_as_a_filter_match(config):
    from api.routers.critique import _build_rule_critique

    config.get_weights = lambda category: {"noise": 1.0}
    result = _build_rule_critique(
        {"noise_sigma": 2.0}, config=config, overrides={"category_override": "portrait"}
    )
    assert result["category"] == "portrait"
    assert result["category_reason"]["reason_key"] == "override"


def test_no_constructor_or_database_initialization(config, monkeypatch):
    from api.routers.critique import _build_rule_critique

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A scalar explanation must not initialize Facet or its database"
        )

    monkeypatch.setattr(Facet, "__init__", forbidden)
    monkeypatch.setattr("processing.scorer.init_database", forbidden)
    config.get_weights = lambda category: {"noise": 1.0}
    result = _build_rule_critique({"noise_sigma": 2.0}, config=config)
    assert result["current_formula_score"] == 8.6


@pytest.mark.parametrize(
    ("stored", "state"),
    [(None, "missing"), (float("nan"), "invalid"), (True, "invalid"), (0.0, "present")],
)
def test_stored_presence_is_separate_from_recalculation(
    config, stored, state
):
    import json
    from api.routers.critique import _build_rule_critique

    config.get_weights = lambda category: {"noise": 1.0}
    result = _build_rule_critique(
        {"noise_sigma": 2.0, "aggregate": stored}, config=config
    )
    assert result["stored_score_state"] == state
    assert result["current_formula_score"] == 8.6
    json.dumps(result, allow_nan=False)


def test_noise_quality_uses_higher_is_better_strength_and_prompt(config):
    from api.routers.critique import _build_rule_critique, _build_vlm_prompt

    config.get_weights = lambda category: {"noise": 1.0}
    result = _build_rule_critique({"noise_sigma": 2.0}, config=config)
    assert result["strengths"] == [{"metric_key": "noise_sigma", "value": 8.6}]
    prompt = _build_vlm_prompt(result, {})
    assert "higher quality is better" in prompt and "lower is better" not in prompt
    assert "not verified historical publication provenance" in prompt
    assert "Penalties applied" not in prompt


@pytest.mark.parametrize("format", ["legacy_luminance", "rgb_uint16"])
def test_histogram_repreparation_matches_native_exposure_for_both_formats(
    config, format
):
    import struct
    import numpy as np
    from api.routers.critique import _build_rule_critique
    from utils.histogram import pack_histogram

    counts = np.zeros(256)
    counts[128] = 1.0
    blob = (
        struct.pack("256f", *counts)
        if format == "legacy_luminance"
        else pack_histogram(counts, counts, counts, counts)
    )
    config.get_weights = lambda category: {"exposure": 1.0}
    photo = {"histogram_data": blob, "exposure_score": 1.0}
    result = _build_rule_critique(photo, config=config)
    assert result["current_formula_score"] == 6.98
    assert result["breakdown"][0]["value"] == 6.98
    assert photo["exposure_score"] == 1.0


def test_rejection_trail_uses_effective_context_not_global_category_order(config):
    from api.routers.critique import _build_rule_critique

    config.get_weights = lambda category: {"noise": 1.0}
    result = _build_rule_critique(
        {
            "noise_sigma": 2.0,
            "tags": "sports",
            "shutter_speed": "1/500",
            "scoring_context": "action_stage",
        },
        config=config,
    )
    assert result["category"] == "sports"
    assert result["category_reason"]["rejected"] == []


def test_rejection_trail_follows_the_context_promotion_order(config):
    from api.routers.critique import _build_rule_critique

    config.get_weights = lambda category: {"noise": 1.0}
    result = _build_rule_critique(
        {
            "noise_sigma": 2.0,
            "tags": "sports",
            "shutter_speed": "1/500",
            "scoring_context": "portrait_session",
        },
        config=config,
    )
    assert result["category"] == "sports"
    # The global order would start at astro; portrait_session promotes the
    # portrait categories, so they are the first ones rejected.
    assert [
        (entry["category"], entry["mismatch"]["key"])
        for entry in result["category_reason"]["rejected"]
    ] == [
        ("portrait", "face_ratio_min"),
        ("portrait_bw", "face_ratio_min"),
        ("fashion", "has_face"),
        ("astro", "luminance_max"),
        ("group_portrait", "face_ratio_min"),
    ]


def test_native_nonpositive_comparison_does_not_turn_nan_weight_into_json_nan(config):
    config.get_weights = lambda category: {"noise": float("nan"), "aesthetic": 1.0}
    terms = _calculate_breakdown(
        {"noise_sigma": 2.0, "aesthetic": 8.0}, config, "default"
    )
    assert len(terms) == 1 and terms[0]["contribution"] == 8.0
