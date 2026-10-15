"""Ascending album sort on the nullable supplementary IQA columns sinks NULLs.

The gallery (``/api/photos``) already puts a NULL ``face_quality_iqa`` /
``aesthetic_iaa`` / ``liqe_score`` last on an ascending sort (see
``test_gallery_null_sort.py``). Albums build their own ORDER BY, so they need the
same guard: the album-photos endpoint defaults to ASC, which would otherwise open a
smart or shared album with every unscored photo, and a smart album's cover (the top
photo of its sort) would be a photo the model could not score.

Real album rows and photo rows go into the SHARED session database (via
``seed_photos_prefix``); the auth-overridden ``edition_client`` drives the endpoints.
"""

import json
import sqlite3

import pytest

from api.routers.albums import _compute_smart_album_cover
from tests.conftest import _TEST_DB_FILE

PREFIX = "/album-null-sort/"
OWNER = "test"  # the user_id the edition_client fixture authenticates as
TOKEN = "null-sort-share-token"
NULLABLE_SORT_COLUMNS = ("aesthetic_iaa", "liqe_score", "face_quality_iqa")


def _rows(column):
    return [
        {"path": PREFIX + "a_null.jpg", "filename": "a_null.jpg", "aggregate": 5.0},
        {"path": PREFIX + "b_low.jpg", "filename": "b_low.jpg", "aggregate": 5.0, column: 3.0},
        {"path": PREFIX + "c_high.jpg", "filename": "c_high.jpg", "aggregate": 5.0, column: 8.0},
    ]


@pytest.fixture()
def make_album():
    """Insert an album row into the shared DB; every one is deleted on teardown."""
    ids = []

    def make(*, smart_filter=None, member_names=()):
        with sqlite3.connect(_TEST_DB_FILE.name) as conn:
            cur = conn.execute(
                "INSERT INTO albums (user_id, name, is_smart, smart_filter_json, share_token) "
                "VALUES (?, 'null sort', ?, ?, ?)",
                (OWNER, 1 if smart_filter is not None else 0,
                 json.dumps(smart_filter) if smart_filter is not None else None, TOKEN),
            )
            album_id = cur.lastrowid
            ids.append(album_id)
            for pos, name in enumerate(member_names):
                conn.execute(
                    "INSERT INTO album_photos (album_id, photo_path, position) VALUES (?, ?, ?)",
                    (album_id, PREFIX + name, pos),
                )
        return album_id

    yield make
    with sqlite3.connect(_TEST_DB_FILE.name) as conn:
        for album_id in ids:
            conn.execute("DELETE FROM album_photos WHERE album_id = ?", (album_id,))
            conn.execute("DELETE FROM albums WHERE id = ?", (album_id,))


def _names(resp):
    assert resp.status_code == 200, resp.text
    return [p["path"].rsplit("/", 1)[1] for p in resp.json()["photos"]]


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_smart_album_asc_puts_null_row_last(column, seed_photos_prefix, make_album, edition_client):
    seed_photos_prefix(PREFIX, _rows(column))
    album_id = make_album(smart_filter={"path_prefix": PREFIX})
    resp = edition_client.get(f"/api/albums/{album_id}/photos?sort={column}&sort_direction=ASC")
    assert _names(resp) == ["b_low.jpg", "c_high.jpg", "a_null.jpg"]


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_regular_album_asc_puts_null_row_last(column, seed_photos_prefix, make_album, edition_client):
    seed_photos_prefix(PREFIX, _rows(column))
    album_id = make_album(member_names=("a_null.jpg", "b_low.jpg", "c_high.jpg"))
    resp = edition_client.get(f"/api/albums/{album_id}/photos?sort={column}&sort_direction=ASC")
    assert _names(resp) == ["b_low.jpg", "c_high.jpg", "a_null.jpg"]


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_shared_smart_album_asc_puts_null_row_last(column, seed_photos_prefix, make_album, edition_client):
    seed_photos_prefix(PREFIX, _rows(column))
    album_id = make_album(smart_filter={"path_prefix": PREFIX, "sort": column, "sort_direction": "ASC"})
    resp = edition_client.get(f"/api/shared/album/{album_id}?token={TOKEN}")
    assert _names(resp) == ["b_low.jpg", "c_high.jpg", "a_null.jpg"]


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_shared_regular_album_asc_puts_null_row_last(column, seed_photos_prefix, make_album, edition_client):
    seed_photos_prefix(PREFIX, _rows(column))
    album_id = make_album(member_names=("a_null.jpg", "b_low.jpg", "c_high.jpg"))
    resp = edition_client.get(
        f"/api/shared/album/{album_id}?token={TOKEN}&sort={column}&sort_direction=ASC")
    assert _names(resp) == ["b_low.jpg", "c_high.jpg", "a_null.jpg"]


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_album_desc_unchanged_and_null_last(column, seed_photos_prefix, make_album, edition_client):
    # PIN: DESC already sank NULLs; the ASC guard must not disturb it.
    seed_photos_prefix(PREFIX, _rows(column))
    album_id = make_album(smart_filter={"path_prefix": PREFIX})
    resp = edition_client.get(f"/api/albums/{album_id}/photos?sort={column}&sort_direction=DESC")
    assert _names(resp) == ["c_high.jpg", "b_low.jpg", "a_null.jpg"]


def _cover(album_id):
    with sqlite3.connect(_TEST_DB_FILE.name) as conn:
        conn.row_factory = sqlite3.Row
        album = conn.execute("SELECT * FROM albums WHERE id = ?", (album_id,)).fetchone()
        return _compute_smart_album_cover(conn, album, None)


@pytest.mark.parametrize("column", NULLABLE_SORT_COLUMNS)
def test_smart_album_cover_asc_skips_null_row(column, seed_photos_prefix, make_album):
    seed_photos_prefix(PREFIX, _rows(column))
    album_id = make_album(
        smart_filter={"path_prefix": PREFIX, "sort": column, "sort_direction": "ASC"})
    assert _cover(album_id) == PREFIX + "b_low.jpg"


def test_smart_album_cover_asc_with_only_null_rows_still_picks_one(seed_photos_prefix, make_album):
    # The guard reorders, it never filters: an album whose photos are ALL unscored
    # still gets a cover.
    seed_photos_prefix(PREFIX, [
        {"path": PREFIX + "a_null.jpg", "filename": "a_null.jpg", "aggregate": 5.0},
        {"path": PREFIX + "b_null.jpg", "filename": "b_null.jpg", "aggregate": 5.0},
    ])
    album_id = make_album(
        smart_filter={"path_prefix": PREFIX, "sort": "face_quality_iqa", "sort_direction": "ASC"})
    assert _cover(album_id) == PREFIX + "a_null.jpg"


def test_smart_album_cover_top_picks_expression_still_sorts(seed_photos_prefix, make_album):
    # PIN: the computed top-picks expression is not a nullable column and keeps
    # its plain ORDER BY (a ``(expr IS NULL)`` guard would be wasted work).
    seed_photos_prefix(PREFIX, [
        {"path": PREFIX + "a_low.jpg", "filename": "a_low.jpg", "aggregate": 2.0},
        {"path": PREFIX + "b_high.jpg", "filename": "b_high.jpg", "aggregate": 9.0},
    ])
    album_id = make_album(
        smart_filter={"path_prefix": PREFIX, "sort": "top_picks_score", "sort_direction": "DESC"})
    assert _cover(album_id) == PREFIX + "b_high.jpg"
