"""Ascending gallery sort on the nullable supplementary IQA columns sinks NULLs.

PyIQA scorers now store NULL (never a neutral 5.0) for a photo they could not
score -- a faceless photo's ``face_quality_iqa``, a sub-224px ``liqe_score``.
SQLite orders NULLs FIRST in ASC, so without the ``(col IS NULL)`` guard an
ascending sort would open with every unscored photo. ``learned_score`` and
``narrative_moment_confidence`` already carry the guard; these columns join them.
DESC needs no guard (SQLite already puts NULLs last), so it must stay unguarded
to keep ``(col DESC, path)`` indexes usable.
"""

import pytest

from api.routers.gallery import _resolve_order_by

PREFIX = "/null-sort/"
NULLABLE_SORT_COLUMNS = ("aesthetic_iaa", "liqe_score", "face_quality_iqa", "qrealign_score",
                         "deqa_score", "aesthetic_v25")


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_asc_sort_puts_null_row_last(column, seed_photos_prefix, anonymous_client):
    seed_photos_prefix(PREFIX, [
        {"path": PREFIX + "a_null.jpg", "filename": "a_null.jpg", "aggregate": 5.0},
        {"path": PREFIX + "b_low.jpg", "filename": "b_low.jpg", "aggregate": 5.0, column: 3.0},
        {"path": PREFIX + "c_high.jpg", "filename": "c_high.jpg", "aggregate": 5.0, column: 8.0},
    ])
    resp = anonymous_client.get(
        f"/api/photos?sort={column}&sort_direction=ASC&path_prefix={PREFIX}")
    assert resp.status_code == 200, resp.text
    paths = [p["path"].rsplit("/", 1)[1] for p in resp.json()["photos"]]
    assert paths == ["b_low.jpg", "c_high.jpg", "a_null.jpg"]


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_desc_sort_unchanged_and_null_last(column, seed_photos_prefix, anonymous_client):
    # PIN: DESC already sank NULLs; it is unchanged by the ASC guard.
    seed_photos_prefix(PREFIX, [
        {"path": PREFIX + "a_null.jpg", "filename": "a_null.jpg", "aggregate": 5.0},
        {"path": PREFIX + "b_low.jpg", "filename": "b_low.jpg", "aggregate": 5.0, column: 3.0},
        {"path": PREFIX + "c_high.jpg", "filename": "c_high.jpg", "aggregate": 5.0, column: 8.0},
    ])
    resp = anonymous_client.get(
        f"/api/photos?sort={column}&sort_direction=DESC&path_prefix={PREFIX}")
    assert resp.status_code == 200, resp.text
    paths = [p["path"].rsplit("/", 1)[1] for p in resp.json()["photos"]]
    assert paths == ["c_high.jpg", "b_low.jpg", "a_null.jpg"]


@pytest.mark.parametrize("column", (*NULLABLE_SORT_COLUMNS, "aesthetic_v25"))
def test_order_by_clause_guards_asc_only(column):
    asc = _resolve_order_by({"sort": column, "dir": "ASC"})
    desc = _resolve_order_by({"sort": column, "dir": "DESC"})
    assert asc == f"({column} IS NULL) ASC, {column} ASC, path ASC"
    assert desc == f"{column} DESC, path ASC"
