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
from api.types import VALID_SORT_COLS
from tests.conftest import _TEST_DB_FILE

PREFIX = "/album-null-sort/"
OWNER = "test"  # the user_id the edition_client fixture authenticates as
TOKEN = "null-sort-share-token"
NULLABLE_SORT_COLUMNS = ("aesthetic_iaa", "liqe_score", "face_quality_iqa", "qrealign_score",
                         "deqa_score", "aesthetic_v25")


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


def test_smart_album_cover_top_picks_asc_picks_lowest(seed_photos_prefix, make_album):
    # The ASC twin of the pin above: the same expression, the other direction.
    seed_photos_prefix(PREFIX, [
        {"path": PREFIX + "a_low.jpg", "filename": "a_low.jpg", "aggregate": 2.0},
        {"path": PREFIX + "b_high.jpg", "filename": "b_high.jpg", "aggregate": 9.0},
    ])
    album_id = make_album(
        smart_filter={"path_prefix": PREFIX, "sort": "top_picks_score", "sort_direction": "ASC"})
    assert _cover(album_id) == PREFIX + "a_low.jpg"


# --- sort=top_picks_score: a computed SELECT alias, not a column --------------------
#
# ``top_picks_score`` is in VALID_SORT_COLS but is an on-the-fly weighted expression,
# so an ORDER BY on it needs the same ``(expr) AS top_picks_score`` alias the gallery
# selects. Every input is set on every row and strictly ordered together (the filenames
# are shuffled against the score order so neither direction can pass by accident), so
# the score is strictly high > mid > low under any positive weights, whichever of the
# face / no-face branches of the expression a row takes.

def _top_picks_rows():
    return [
        {"path": PREFIX + "a_mid.jpg", "filename": "a_mid.jpg", "aggregate": 6.0,
         "aesthetic": 6.0, "comp_score": 6.0, "face_ratio": 0.05, "face_quality": 6.0},
        {"path": PREFIX + "b_high.jpg", "filename": "b_high.jpg", "aggregate": 9.0,
         "aesthetic": 9.0, "comp_score": 9.0, "face_ratio": 0.5, "face_quality": 9.0},
        {"path": PREFIX + "c_low.jpg", "filename": "c_low.jpg", "aggregate": 3.0,
         "aesthetic": 3.0, "comp_score": 3.0, "face_ratio": 0.0, "face_quality": 3.0},
    ]


TOP_PICKS_MEMBERS = ("a_mid.jpg", "b_high.jpg", "c_low.jpg")
TOP_PICKS_DESC = ["b_high.jpg", "a_mid.jpg", "c_low.jpg"]


def _assert_top_picks_order(resp, expected_names, direction):
    names = _names(resp)
    assert names == expected_names
    scores = [p["top_picks_score"] for p in resp.json()["photos"]]
    assert all(s is not None for s in scores), scores
    assert scores == sorted(scores, reverse=(direction == "DESC")), scores
    assert len(set(scores)) == len(scores), scores  # strict, so the order is not a tie-break


@pytest.mark.parametrize("direction,expected", [
    ("DESC", TOP_PICKS_DESC),
    ("ASC", TOP_PICKS_DESC[::-1]),
])
def test_smart_album_sorts_by_top_picks_score(direction, expected, seed_photos_prefix,
                                              make_album, edition_client):
    seed_photos_prefix(PREFIX, _top_picks_rows())
    album_id = make_album(smart_filter={"path_prefix": PREFIX})
    resp = edition_client.get(
        f"/api/albums/{album_id}/photos?sort=top_picks_score&sort_direction={direction}")
    _assert_top_picks_order(resp, expected, direction)


@pytest.mark.parametrize("direction,expected", [
    ("DESC", TOP_PICKS_DESC),
    ("ASC", TOP_PICKS_DESC[::-1]),
])
def test_regular_album_sorts_by_top_picks_score(direction, expected, seed_photos_prefix,
                                                make_album, edition_client):
    seed_photos_prefix(PREFIX, _top_picks_rows())
    album_id = make_album(member_names=TOP_PICKS_MEMBERS)
    resp = edition_client.get(
        f"/api/albums/{album_id}/photos?sort=top_picks_score&sort_direction={direction}")
    _assert_top_picks_order(resp, expected, direction)


@pytest.mark.parametrize("direction,expected", [
    ("DESC", TOP_PICKS_DESC),
    ("ASC", TOP_PICKS_DESC[::-1]),
])
def test_shared_smart_album_sorts_by_top_picks_score(direction, expected, seed_photos_prefix,
                                                     make_album, edition_client):
    seed_photos_prefix(PREFIX, _top_picks_rows())
    album_id = make_album(smart_filter={
        "path_prefix": PREFIX, "sort": "top_picks_score", "sort_direction": direction})
    resp = edition_client.get(f"/api/shared/album/{album_id}?token={TOKEN}")
    _assert_top_picks_order(resp, expected, direction)


@pytest.mark.parametrize("direction,expected", [
    ("DESC", TOP_PICKS_DESC),
    ("ASC", TOP_PICKS_DESC[::-1]),
])
def test_shared_regular_album_sorts_by_top_picks_score(direction, expected, seed_photos_prefix,
                                                       make_album, edition_client):
    seed_photos_prefix(PREFIX, _top_picks_rows())
    album_id = make_album(member_names=TOP_PICKS_MEMBERS)
    resp = edition_client.get(
        f"/api/shared/album/{album_id}?token={TOKEN}"
        f"&sort=top_picks_score&sort_direction={direction}")
    _assert_top_picks_order(resp, expected, direction)


def test_smart_album_top_picks_filter_carries_the_score(seed_photos_prefix, make_album,
                                                        edition_client):
    # The gallery selects the alias for the top-picks FILTER too, not only the sort;
    # a smart album saved with that filter must return the same field.
    seed_photos_prefix(PREFIX, _top_picks_rows())
    album_id = make_album(smart_filter={"path_prefix": PREFIX, "top_picks_filter": "1"})
    resp = edition_client.get(f"/api/albums/{album_id}/photos")
    assert _names(resp) == ["b_high.jpg"]
    assert resp.json()["photos"][0]["top_picks_score"] is not None


@pytest.mark.parametrize("kind", ["smart", "regular"])
@pytest.mark.parametrize("sort_col", VALID_SORT_COLS)
def test_every_valid_sort_column_works_on_albums(sort_col, kind, seed_photos_prefix,
                                                 make_album, edition_client):
    # Structural guard: a future computed sort alias (like top_picks_score) that the
    # album SELECT does not carry would 500 here instead of in a user's browser.
    seed_photos_prefix(PREFIX, _top_picks_rows())
    if kind == "smart":
        album_id = make_album(smart_filter={"path_prefix": PREFIX})
    else:
        album_id = make_album(member_names=TOP_PICKS_MEMBERS)
    for direction in ("ASC", "DESC"):
        resp = edition_client.get(
            f"/api/albums/{album_id}/photos?sort={sort_col}&sort_direction={direction}")
        assert resp.status_code == 200, (sort_col, kind, direction, resp.text)
        assert len(resp.json()["photos"]) == len(TOP_PICKS_MEMBERS)
