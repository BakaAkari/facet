"""Synthetic numeric-domain regressions; no models, images or external config."""
import math
import pytest
from processing import scorer
from config.category_filter import _to_float


@pytest.fixture
def native_scorer():
    from config import ScoringConfig
    from config_resolve import defaults_path
    instance = scorer.Facet.__new__(scorer.Facet)
    instance.config = ScoringConfig(str(defaults_path()), validate=False)
    instance.config.config['models']['vram_profile'] = 'legacy'
    return instance


class Config:
    def __init__(self, compensate=True):
        self.compensate = compensate
        self.received = None

    def get_exif_adjustments(self):
        return {'iso_sharpness_compensation': self.compensate, 'aperture_isolation_boost': True}

    def get_penalty_settings(self):
        return {}

    def get_extended_iqa_settings(self):
        return {}

    def get_categories(self):
        return [{'name': 'portrait'}]

    def determine_category(self, data, context=None):
        self.received = data, context
        return 'default'


@pytest.mark.parametrize('iso,expected', [(800, 5), (1600, 5.5), (3200, 6), (12800, 7)])
def test_iso_compensation_reachable_for_real_exif_values(iso, expected):
    result = scorer.build_metric_vector({'tech_sharpness': 5, 'iso': iso}, Config(), 'default')
    assert result['tech_sharpness'] == expected
    assert scorer.build_metric_vector({'tech_sharpness': 5, 'iso': iso}, Config(False), 'default')['tech_sharpness'] == 5


@pytest.mark.parametrize('spread', [100, 100.0001, 120.0, 127.5])
def test_population_std_keeps_legitimate_high_range(spread):
    assert scorer.build_metric_vector({'histogram_spread': spread}, Config(), 'default')['dynamic_range'] == 10


def test_population_std_refuses_impossible_range():
    assert scorer.build_metric_vector({'histogram_spread': 127.5001}, Config(), 'default')['dynamic_range'] == 0


def test_raw_variance_is_valid_and_existing_aggregate_clamp_unchanged(monkeypatch, native_scorer):
    value = scorer.build_metric_vector({'face_sharpness': 2500.0}, Config(), 'default')['face_sharpness']
    assert value == 2500.0
    monkeypatch.setattr(native_scorer.config, 'get_weights', lambda category: {'face_sharpness': 1.0})
    score, _ = native_scorer.calculate_aggregate_logic({'face_sharpness': 2500.0, 'noise_sigma': 0})
    assert score == 10


def test_unbounded_valid_measurements_keep_native_penalty_and_saturation():
    assert scorer.build_metric_vector({'isolation_bonus': 101}, Config(), 'default')['isolation'] == 10
    penalties = scorer._calculate_scoring_penalties({'noise_sigma': 101}, Config())
    assert penalties['noise_penalty'] == 1.5
    assert scorer.build_metric_vector({'noise_sigma': 101}, Config(), 'default')['noise'] == 0


@pytest.mark.parametrize('bad', [None, b'5', {}, [], 'bad', math.nan, math.inf, -math.inf, 'nan', 'inf', '-inf', 10**400])
def test_invalid_values_return_exact_supplied_default(bad):
    sentinel = object()
    assert scorer._safe_float(bad, sentinel, max_value=None) is sentinel


def test_legacy_default_policy_strings_bools_and_optional_inclusive_bounds():
    assert scorer._safe_float(100) == 100
    assert scorer._safe_float(-100) == -100
    assert scorer._safe_float(100.1) == 5
    assert scorer._safe_float('12.5') == 12.5
    assert scorer._safe_float(False) == 0 and scorer._safe_float(True) == 1
    assert scorer._safe_float(200, None, max_value=None) == 200
    assert scorer._safe_float(-200, None, min_value=None) == -200
    assert scorer._safe_float(-200, None, min_value=None, max_value=None) == -200


def test_legacy_score_clamp_policy_not_replaced_with_strict_score_range():
    assert scorer._calculate_scoring_penalties({'leading_lines_score': 100}, Config())['leading_lines'] == 10
    assert scorer._calculate_scoring_penalties({'leading_lines_score': 200}, Config())['leading_lines'] == 0


def test_category_domains_and_context_without_arbitrary_hundred_cap():
    cfg = Config()
    assert scorer.Facet._determine_photo_category({'face_count': 101, 'face_ratio': 1.2, 'mean_luminance': 1,
                                                  'iso': 3200, 'focal_length': 200, 'f_stop': 128,
                                                  'shutter_speed': '120', 'scoring_context': 'street'}, cfg) == 'default'
    values, context = cfg.received
    assert (values['face_count'], values['face_ratio'], values['iso'], values['focal_length'],
            values['f_stop'], values['shutter_speed'], context) == (101, 1.2, 3200, 200, 128, 120, 'street')


def test_nested_category_numeric_string_policy_preserved_and_override_first():
    cfg = Config()
    scorer.Facet._determine_photo_category({'face_count': '101', 'face_ratio': '1.2', 'mean_luminance': '0.2'}, cfg)
    assert cfg.received[0]['face_count'] == 0 and cfg.received[0]['face_ratio'] == 0
    assert cfg.received[0]['mean_luminance'] == 0.5
    cfg.received = None
    assert scorer.Facet._determine_photo_category({'category_override': 'portrait', 'face_count': math.inf}, cfg) == 'portrait'
    assert cfg.received is None


@pytest.mark.parametrize('value', [0, -1, math.nan, math.inf, 'nan', 'inf', '1/0', '1/2/3'])
def test_nonpositive_or_nonfinite_exposure_is_not_a_real_shutter(value):
    assert scorer.Facet._parse_shutter_speed(value) is None


def test_fractional_and_long_positive_shutters_and_filter_parsing():
    assert scorer.Facet._parse_shutter_speed('1/500') == 0.002
    assert scorer.Facet._parse_shutter_speed('120') == 120
    assert _to_float('1/500') == 0.002
    assert _to_float(b'3200') == 3200  # Existing generic filter parsing remains compatible.
    assert _to_float(-1) == -1
    assert _to_float(math.nan) is None and _to_float('inf') is None


def test_saturation_luminance_and_aperture_boundaries():
    assert scorer.build_metric_vector({'mean_saturation': 1}, Config(), 'default')['saturation'] == 10
    assert scorer.build_metric_vector({'mean_saturation': 0}, Config(), 'default')['saturation'] == 0
    assert scorer.build_metric_vector({'mean_saturation': 1.1}, Config(), 'default')['saturation'] == 5
    assert scorer.build_metric_vector({'isolation_bonus': 1, 'f_stop': 2}, Config(), 'default')['isolation'] == 2.5
    assert scorer.build_metric_vector({'isolation_bonus': 1, 'f_stop': 2.8}, Config(), 'default')['isolation'] == pytest.approx(1.5)


def test_existing_fractional_exif_category_parsing_preserved():
    cfg = Config()
    scorer.Facet._determine_photo_category({'iso': '3200/1', 'f_stop': '28/10', 'focal_length': '200/1'}, cfg)
    values, _ = cfg.received
    assert values['iso'] == 3200
    assert values['f_stop'] == 2.8
    assert values['focal_length'] == 200
    assert scorer.build_metric_vector({'tech_sharpness': 5, 'iso': '3200/1'}, Config(), 'default')['tech_sharpness'] == 6


@pytest.mark.parametrize('field', ['iso', 'f_stop', 'focal_length'])
@pytest.mark.parametrize('bad', [0, -1, math.nan, math.inf, 'nan', '1/0'])
def test_nonpositive_or_nonfinite_exif_stored_as_none_in_category_data(field, bad):
    cfg = Config()
    scorer.Facet._determine_photo_category({field: bad}, cfg)
    assert cfg.received[0][field] is None


@pytest.mark.parametrize('bad', [0, -3200, math.nan])
def test_invalid_iso_gives_no_sharpness_compensation(bad):
    vector = scorer.build_metric_vector({'tech_sharpness': 5, 'iso': bad}, Config(), 'default')
    assert vector['tech_sharpness'] == 5


def test_negative_raw_measurements_fall_back_to_supplied_default():
    vector = scorer.build_metric_vector(
        {'face_sharpness': -1, 'noise_sigma': -1, 'histogram_spread': -1, 'isolation_bonus': 0.5}, Config(), 'default')
    assert vector['face_sharpness'] == 5.0
    assert vector['noise'] == scorer.build_metric_vector({'noise_sigma': 0}, Config(), 'default')['noise']
    assert vector['dynamic_range'] == 0
    assert vector['isolation'] == 0
    assert scorer._calculate_scoring_penalties({'noise_sigma': -1}, Config())['noise_penalty'] == \
        scorer._calculate_scoring_penalties({'noise_sigma': 0}, Config())['noise_penalty']
