"""Numeric sorting of the TEXT ``shutter_speed`` column through the pure-SQL expression.

The expression itself is checked on a bare in-memory connection; ordering is checked
through the real endpoints against rows in the SHARED session database (seeded with
``seed_photos_prefix``), driven by the auth-overridden client fixtures.
"""

import json
import sqlite3

import pytest

from api.exposure import exposure_seconds_sql
from api.routers.albums import _compute_smart_album_cover
from api.routers.gallery import _resolve_order_by
from tests.conftest import _TEST_DB_FILE

PREFIX = "/shutter-sort/"
OWNER = "test"  # the user_id the edition_client fixture authenticates as
TOKEN = "shutter-sort-share-token"
EXPOSURES = [
    ("long", "120"), ("medium", "30"), ("short", "0.25"), ("fraction", "1/500"),
    ("tie_a", "1/4"), ("tie_b", "0.25"), ("alias_a", "1/250"), ("alias_b", "0.004"),
    ("bad", "garbage"), ("bad_suffix", "120foo"), ("zero", "0"),
    ("divide_zero", "1/0"), ("missing", None), ("infinity", "inf"),
]
INVALID = ["bad", "bad_suffix", "divide_zero", "infinity", "missing", "zero"]  # path tie-break
ASCENDING = ["fraction", "alias_a", "alias_b", "short", "tie_a", "tie_b", "medium", "long", *INVALID]
DESCENDING = ["long", "medium", "short", "tie_a", "tie_b", "alias_a", "alias_b", "fraction", *INVALID]
ORDERS = [("ASC", ASCENDING), ("DESC", DESCENDING)]


def _rows():
    # aggregate runs opposite to the expected ASC order, so an album sort that
    # silently fell back to the saved/default aggregate order cannot pass.
    return [
        {"path": PREFIX + name, "filename": name, "aggregate": 1.0 + i * 0.5, "shutter_speed": value}
        for i, (name, value) in enumerate(reversed(EXPOSURES))
    ]


def _seconds(value):
    with sqlite3.connect(":memory:") as conn:
        return conn.execute(
            f"SELECT {exposure_seconds_sql()} FROM (SELECT ? AS shutter_speed)", (value,)
        ).fetchone()[0]


@pytest.mark.parametrize("value,expected", [
    ("120", 120), ("30", 30), ("0.25", 0.25), ("1/4", 0.25), ("1/250", 0.004),
    ("0.004", 0.004), (" 1 / 250 ", 0.004), ("1e-3", 0.001), (120, 120),
    (0.004, 0.004), ("1/500", 0.002),
])
def test_expression_valid_values(value, expected):
    assert _seconds(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [
    None, "", "garbage", "None", "120foo", "0", "-1", "1/0", "0/1", "1/-2", "-1/-2",
    "1/2/3", "inf", "nan", "1e999", "1; DROP TABLE photos",
])
def test_expression_invalid_values_are_null(value):
    assert _seconds(value) is None


def test_expression_uses_the_given_column():
    with sqlite3.connect(":memory:") as conn:
        assert conn.execute(
            f"SELECT {exposure_seconds_sql('ss')} FROM (SELECT '1/4' AS ss)").fetchone()[0] == 0.25


def _names(resp):
    assert resp.status_code == 200, resp.text
    return [p["path"].rsplit("/", 1)[1] for p in resp.json()["photos"]]


@pytest.mark.parametrize("direction,expected", ORDERS)
def test_gallery_orders_numerically_with_pagination(direction, expected, seed_photos_prefix,
                                                    anonymous_client):
    seed_photos_prefix(PREFIX, _rows())
    actual = []
    for page in range(1, 6):
        resp = anonymous_client.get(
            f"/api/photos?sort=shutter_speed&sort_direction={direction}"
            f"&path_prefix={PREFIX}&per_page=3&page={page}")
        actual += _names(resp)
    assert actual == expected


@pytest.mark.parametrize("direction,expected", ORDERS)
def test_select_bottom_percent_follows_the_order(direction, expected, seed_photos_prefix,
                                                 anonymous_client):
    seed_photos_prefix(PREFIX, _rows())
    resp = anonymous_client.get(
        "/api/photos/select_bottom_percent",
        params={"keep_percent": 50, "sort": "shutter_speed", "sort_direction": direction,
                "path_prefix": PREFIX})
    assert resp.status_code == 200, resp.text
    assert [p.rsplit("/", 1)[1] for p in resp.json()["paths"]] == expected[7:]


def test_stored_exif_text_is_unchanged_by_sorting(seed_photos_prefix, anonymous_client):
    seed_photos_prefix(PREFIX, _rows())
    _names(anonymous_client.get(
        f"/api/photos?sort=shutter_speed&sort_direction=ASC&path_prefix={PREFIX}"))
    with sqlite3.connect(_TEST_DB_FILE.name) as conn:
        stored = dict(conn.execute(
            "SELECT filename, shutter_speed FROM photos WHERE path LIKE ?", (PREFIX + "%",)))
    assert stored == dict(EXPOSURES)


def test_sort_injection_falls_back_safely():
    sort_payload = _resolve_order_by({"sort": "shutter_speed); DROP TABLE photos;--", "dir": "ASC"})
    dir_payload = _resolve_order_by({"sort": "shutter_speed", "dir": "ASC; DROP TABLE photos"})
    assert sort_payload == "aggregate ASC, path ASC"
    assert "DROP" not in dir_payload and ";" not in dir_payload
    assert dir_payload.endswith("DESC, path ASC")


@pytest.fixture()
def make_album():
    """Insert an album row into the shared DB; every one is deleted on teardown."""
    ids = []

    def make(*, smart_filter=None, member_names=()):
        with sqlite3.connect(_TEST_DB_FILE.name) as conn:
            cur = conn.execute(
                "INSERT INTO albums (user_id, name, is_smart, smart_filter_json, share_token) "
                "VALUES (?, 'shutter sort', ?, ?, ?)",
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


@pytest.mark.parametrize("direction,expected", ORDERS)
def test_regular_album_and_shared_album_order(direction, expected, seed_photos_prefix, make_album,
                                              edition_client):
    seed_photos_prefix(PREFIX, _rows())
    album_id = make_album(member_names=[name for name, _ in EXPOSURES])
    query = f"sort=shutter_speed&sort_direction={direction}"
    assert _names(edition_client.get(f"/api/albums/{album_id}/photos?{query}")) == expected
    assert _names(edition_client.get(
        f"/api/shared/album/{album_id}?token={TOKEN}&{query}")) == expected


@pytest.mark.parametrize("direction,expected", ORDERS)
def test_smart_album_request_sort_overrides_saved_filter_sort(direction, expected,
                                                              seed_photos_prefix, make_album,
                                                              edition_client):
    seed_photos_prefix(PREFIX, _rows())
    album_id = make_album(smart_filter={"path_prefix": PREFIX, "sort": "aggregate",
                                        "sort_direction": "ASC"})
    resp = edition_client.get(
        f"/api/albums/{album_id}/photos?sort=shutter_speed&sort_direction={direction}")
    assert _names(resp) == expected


@pytest.mark.parametrize("direction,expected", ORDERS)
def test_smart_album_cover_follows_saved_filter_sort(direction, expected, seed_photos_prefix,
                                                     make_album):
    seed_photos_prefix(PREFIX, _rows())
    album_id = make_album(smart_filter={"path_prefix": PREFIX, "sort": "shutter_speed",
                                        "sort_direction": direction})
    with sqlite3.connect(_TEST_DB_FILE.name) as conn:
        conn.row_factory = sqlite3.Row
        album = conn.execute("SELECT * FROM albums WHERE id = ?", (album_id,)).fetchone()
        assert _compute_smart_album_cover(conn, album, None) == PREFIX + expected[0]
