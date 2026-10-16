"""A photo with no subject (NULL saliency / power-point columns) is not matched by a range filter.

The no-subject branch used to store 5.0 in ``subject_sharpness`` / ``subject_placement`` /
``bg_separation`` and the no-box branch 5.0 in ``power_point_score``, so ``min_subject_sharpness=0``
returned a photo whose "sharpness" was a made-up number. They are NULL now, and SQL's
``NULL >= 0`` is not true: the photo drops out of a range filter (intended), while the unfiltered
gallery still lists it, with ``null`` on the wire rather than ``NaN`` or a neutral score.

The no-subject row is built from what the two producers return for a frame with no subject
(``SaliencyScorer._score_from_mask`` and ``CompositionAnalyzer.get_placement_data``), so the test
fails against a producer that stores a neutral number. Rows go into the SHARED session database
through ``seed_photos_prefix``; the auth-overridden ``edition_client`` drives the endpoint.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")

from analyzers import CompositionAnalyzer  # noqa: E402
from models.saliency_scorer import SaliencyScorer  # noqa: E402

PREFIX = "/t8-null-saliency/"
# (column, min filter key, max filter key)
NULLABLE = [
    ("subject_sharpness", "min_subject_sharpness", "max_subject_sharpness"),
    ("subject_placement", "min_subject_placement", "max_subject_placement"),
    ("bg_separation", "min_bg_separation", "max_bg_separation"),
    ("power_point_score", "min_power_point", "max_power_point"),
]


def _no_subject_values():
    """What a scan stores for a frame with no subject: the producers' own output."""
    scorer = SaliencyScorer.__new__(SaliencyScorer)
    scorer.min_subject_pixels = 50
    scorer.mask_threshold = 0.3
    img = np.full((12, 16, 3), 120, dtype=np.uint8)
    saliency = scorer._score_from_mask(np.zeros((12, 16), np.uint8), img)
    assert saliency['subject_prominence'] == 0.0
    composition = CompositionAnalyzer.get_placement_data(None, 16, 12, None, img)
    return {
        "subject_sharpness": saliency['subject_sharpness'],
        "subject_prominence": saliency['subject_prominence'],
        "subject_placement": saliency['subject_placement'],
        "bg_separation": saliency['bg_separation'],
        "power_point_score": composition['power_point_score'],
    }


def _rows():
    measured = {col: 6.0 for col, _, _ in NULLABLE}
    return [
        {"path": PREFIX + "no_subject.jpg", "filename": "no_subject.jpg", "aggregate": 5.0,
         **_no_subject_values()},
        {"path": PREFIX + "subject.jpg", "filename": "subject.jpg", "aggregate": 5.0,
         "subject_prominence": 0.4, **measured},
    ]


def _names(resp):
    assert resp.status_code == 200, resp.text
    return sorted(p["path"].rsplit("/", 1)[1] for p in resp.json()["photos"])


def _get(edition_client, **params):
    return edition_client.get("/api/photos", params={
        "path_prefix": PREFIX, "per_page": 50, "hide_bursts": "0", "hide_duplicates": "0",
        "hide_blinks": "0", "hide_brackets": "0", "hide_panoramas": "0", **params})


@pytest.mark.parametrize("column, min_key, max_key", NULLABLE)
def test_a_min_filter_of_zero_drops_the_null_photo(column, min_key, max_key, seed_photos_prefix, edition_client):
    seed_photos_prefix(PREFIX, _rows())
    assert _names(_get(edition_client, **{min_key: "0"})) == ["subject.jpg"]


@pytest.mark.parametrize("column, min_key, max_key", NULLABLE)
def test_a_max_filter_drops_the_null_photo_too(column, min_key, max_key, seed_photos_prefix, edition_client):
    seed_photos_prefix(PREFIX, _rows())
    assert _names(_get(edition_client, **{max_key: "10"})) == ["subject.jpg"]


def test_the_unfiltered_gallery_lists_both_and_sends_null_not_nan(seed_photos_prefix, edition_client):
    seed_photos_prefix(PREFIX, _rows())
    resp = _get(edition_client)
    assert _names(resp) == ["no_subject.jpg", "subject.jpg"]
    by_name = {p["path"].rsplit("/", 1)[1]: p for p in resp.json()["photos"]}
    for column, _, _ in NULLABLE:
        assert by_name["no_subject.jpg"][column] is None, column
        assert by_name["subject.jpg"][column] == 6.0, column
    assert by_name["no_subject.jpg"]["subject_prominence"] == 0.0   # a real measurement stays a number
    assert "NaN" not in resp.text
