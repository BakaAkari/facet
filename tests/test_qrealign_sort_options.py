"""Synthetic API/UI-contract coverage for the existing qrealign_score metric."""

from pathlib import Path

import pytest

import licensing
from api.types import VALID_SORT_COLS

COLUMN = 'qrealign_score'
LOADER = 'qrealign'
LABEL = 'Q-ReAlign (Aesthetic)'
PREFIX = "/qrealign_score-sort/"


def test_metric_is_advertised_and_allowed(anonymous_client):
    response = anonymous_client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    assert [COLUMN, LABEL] in body["sort_options"]
    assert any(option["column"] == COLUMN
               for options in body["sort_options_grouped"].values() for option in options)
    assert COLUMN in VALID_SORT_COLS


@pytest.mark.parametrize("lang", ["en", "fr", "de", "it", "es", "pt", "zh"])
def test_metric_label_is_translated(lang, anonymous_client):
    response = anonymous_client.get(f"/api/i18n/{lang}")
    assert response.status_code == 200
    assert response.json()["sort_options"][COLUMN].strip()
    keys = Path(__file__).resolve().parents[1] / "client/src/app/core/i18n/keys.ts"
    assert f'sort_options.{COLUMN}' in keys.read_text()


@pytest.mark.parametrize("direction,expected", [
    ("ASC", ["b_zero", "c_tie", "d_tie", "e_high", "a_null"]),
    ("DESC", ["e_high", "c_tie", "d_tie", "b_zero", "a_null"]),
])
def test_gallery_numeric_order_null_ties_and_pages(
    direction, expected, seed_photos_prefix, anonymous_client,
):
    values = [("a_null", None), ("b_zero", 0.0), ("c_tie", 3.0),
              ("d_tie", 3.0), ("e_high", 8.0)]
    seed_photos_prefix(PREFIX, [
        {"path": PREFIX + name + ".jpg", "filename": name + ".jpg",
         "aggregate": 5.0, COLUMN: score} for name, score in values
    ])
    params = {"sort": COLUMN, "sort_direction": direction, "path_prefix": PREFIX}
    response = anonymous_client.get("/api/photos", params=params)
    assert response.status_code == 200, response.text
    photos = response.json()["photos"]
    assert [photo["filename"] for photo in photos] == [name + ".jpg" for name in expected]
    assert photos[-1][COLUMN] is None
    assert {photo[COLUMN] for photo in photos[:-1]} == {0.0, 3.0, 8.0}
    paged = []
    for page in (1, 2, 3):
        response = anonymous_client.get("/api/photos", params={**params, "page": page, "per_page": 2})
        assert response.status_code == 200, response.text
        paged.extend(photo["filename"] for photo in response.json()["photos"])
    assert paged == [name + ".jpg" for name in expected]


@pytest.mark.parametrize("direction", ["ASC", "DESC"])
def test_all_unscored_rows_remain_visible(direction, seed_photos_prefix, anonymous_client):
    seed_photos_prefix(PREFIX, [
        {"path": PREFIX + name, "filename": name, "aggregate": 5.0}
        for name in ("a.jpg", "b.jpg")
    ])
    response = anonymous_client.get("/api/photos", params={
        "sort": COLUMN, "sort_direction": direction, "path_prefix": PREFIX,
    })
    assert response.status_code == 200, response.text
    assert [photo["filename"] for photo in response.json()["photos"]] == ["a.jpg", "b.jpg"]


@pytest.mark.parametrize("mode,allow_unclear", [
    ("default", False), ("commercial", False), ("commercial", True),
])
def test_sort_visibility_follows_existing_loader_license(mode, allow_unclear, anonymous_client):
    licensing.set_policy(mode, allow_unclear=allow_unclear)
    allowed = licensing.loader_allowed(LOADER)
    response = anonymous_client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    flat = {column for column, _ in body["sort_options"]}
    grouped = {option["column"] for options in body["sort_options_grouped"].values()
               for option in options}
    assert (COLUMN in flat) is allowed
    assert (COLUMN in grouped) is allowed
    assert (COLUMN in body["unavailable_metrics"]) is not allowed
