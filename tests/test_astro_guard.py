"""The shipped ``astro`` category needs a dark frame as well as an astro tag.

``aurora`` is in the shipped tag vocabulary as the CLIP tagger's junk-tag magnet: 6,506 of a
129,737-photo library carried it without being astro photos (luminance p50 0.388, the library's
own p50 being 0.401), and they all landed in ``astro``. The shipped filter now carries
``luminance_max: 0.30``. These tests name ``config/scoring_config.default.json`` explicitly (a
developer's own ``scoring_config.json`` overrides it) and pin the outcome through both category
paths: ``ScoringConfig.determine_category`` (``CategoryFilter``) and the scorer's
``_determine_photo_category``, which defaults a missing luminance to 0.5.
"""

import pytest

from config.scoring_config import ScoringConfig
from config_resolve import defaults_path
from processing.scorer import Facet


@pytest.fixture(scope="module")
def cfg():
    return ScoringConfig(defaults_path(), validate=False)


def _photo(tags, luminance, **extra):
    return {"tags": tags, "face_count": 0, "face_ratio": 0.0, "is_group_portrait": 0,
            "is_silhouette": 0, "is_monochrome": 0, "mean_luminance": luminance, **extra}


def _both_paths(cfg, photo):
    """The category through the filter engine and through the scorer's row-based wrapper."""
    wrapper = Facet.__new__(Facet)._determine_photo_category(photo, cfg)
    return cfg.determine_category(photo), wrapper


def test_the_shipped_astro_filter_carries_the_guard_and_night_is_untouched(cfg):
    by_name = {c["name"]: c for c in cfg.get_categories()}
    assert by_name["astro"]["filters"]["luminance_max"] == 0.30
    assert by_name["night"]["filters"]["luminance_max"] == 0.15


@pytest.mark.parametrize("tag", ["stars", "aurora", "milky way", "astrophotography"])
def test_a_dark_astro_tagged_photo_is_astro(cfg, tag):
    assert _both_paths(cfg, _photo(tag, 0.10)) == ("astro", "astro")


def test_a_bright_astro_tagged_photo_is_not_astro(cfg):
    for category in _both_paths(cfg, _photo("stars", 0.45)):
        assert category != "astro"


def test_the_bound_is_inclusive(cfg):
    assert _both_paths(cfg, _photo("stars", 0.30)) == ("astro", "astro")
    for category in _both_paths(cfg, _photo("stars", 0.31)):
        assert category != "astro"


def test_a_bright_aurora_tagged_photo_with_a_face_is_a_portrait_not_astro(cfg):
    photo = _photo("aurora", 0.45, face_count=1, face_ratio=0.2)
    assert _both_paths(cfg, photo) == ("portrait", "portrait")


def test_an_astro_tagged_photo_with_no_luminance_is_not_astro_on_either_path(cfg):
    # The numeric luminance_max filter treats a NULL actual as a mismatch, and the scorer's
    # wrapper defaults a missing luminance to 0.5 (> 0.30): the same outcome by two routes.
    photo = _photo("stars", None)
    assert cfg.determine_category(photo) != "astro"
    assert Facet.__new__(Facet)._determine_photo_category(photo, cfg) != "astro"
    assert Facet.__new__(Facet)._determine_photo_category({**photo, "mean_luminance": 0.5}, cfg) != "astro"
