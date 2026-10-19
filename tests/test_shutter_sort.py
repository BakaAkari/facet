"""Numeric exposure sorting through real SQLite and both API connection factories."""
import math
import json
import sqlite3
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api import database
from api.exposure import SQL_FUNCTION, exposure_seconds
from api.routers.gallery import _resolve_order_by
from api.routers.albums import warm_smart_album_covers, _cover_cache_key
from tests.test_gallery import (
    _VIEWER_CONFIG, _create_app_no_auth, _existing_columns, _make_db, _photo,
)

EXPOSURES = [
    ('/long', '120'), ('/medium', '30'), ('/short', '0.25'),
    ('/fraction', '1/500'), ('/tie_a', '1/4'), ('/tie_b', '0.25'),
    ('/alias_a', '1/250'), ('/alias_b', '0.004'),
    ('/bad', 'garbage'), ('/bad_suffix', '120foo'), ('/zero', '0'),
    ('/divide_zero', '1/0'), ('/missing', None), ('/infinity', 'inf'),
]
ASCENDING = [
    '/fraction', '/alias_a', '/alias_b', '/short', '/tie_a', '/tie_b',
    '/medium', '/long', '/bad', '/bad_suffix', '/divide_zero', '/infinity',
    '/missing', '/zero',
]
DESCENDING = [
    '/long', '/medium', '/short', '/tie_a', '/tie_b', '/alias_a', '/alias_b',
    '/fraction', '/bad', '/bad_suffix', '/divide_zero', '/infinity', '/missing', '/zero',
]


@pytest.mark.parametrize('direction,expected', [('ASC', ASCENDING), ('DESC', DESCENDING)])
def test_real_sqlite_text_numeric_order_and_invalids_last(direction, expected):
    with sqlite3.connect(':memory:') as connection:
        connection.create_function(SQL_FUNCTION, 1, exposure_seconds, deterministic=True)
        connection.execute('CREATE TABLE photos(path TEXT, shutter_speed TEXT)')
        connection.executemany('INSERT INTO photos VALUES(?, ?)', EXPOSURES)
        clause = _resolve_order_by({'sort': 'shutter_speed', 'dir': direction})
        actual = [row[0] for row in connection.execute('SELECT path FROM photos ORDER BY ' + clause)]
        assert actual == expected
        assert connection.execute('SELECT path, shutter_speed FROM photos ORDER BY rowid').fetchall() == EXPOSURES


@pytest.mark.parametrize('value', [
    None, 'garbage', '120foo', '1/0', '0', '-1', '-1/-2', 'inf', 'nan',
    '1e999', '1/2/3', b'30', True, False, '1/inf', 'inf/1', '1e-999',
    '1; DROP TABLE photos', '0/1', '1/-2', float('inf'), float('nan'),
])
def test_invalid_values_do_not_become_zero_or_partial_numbers(value):
    assert exposure_seconds(value) is None


@pytest.mark.parametrize('value,expected', [
    (120, 120), (0.004, 0.004), (' 1 / 250 ', 0.004), ('1e-3', 0.001),
    ('1/250', 0.004), ('0.004', 0.004), ('1/4', 0.25),
])
def test_valid_numeric_and_rational_exif_values(value, expected):
    assert exposure_seconds(value) == expected


@pytest.mark.parametrize('direction,expected', [('ASC', ASCENDING), ('DESC', DESCENDING)])
def test_real_api_connections_and_pagination(tmp_path, direction, expected):
    path = str(tmp_path / 'photos.db')
    _make_db(path, [_photo(p, '2020:01:01 00:00:00', shutter_speed=value) for p, value in EXPOSURES])
    app = _create_app_no_auth()
    with (
        mock.patch.object(database, 'DEFAULT_DB_PATH', path),
        mock.patch('api.routers.gallery.VIEWER_CONFIG', _VIEWER_CONFIG),
        mock.patch('api.db_helpers._existing_columns_cache', _existing_columns(path)),
        TestClient(app) as client,
    ):
        connection = database.get_db_connection()
        try:
            assert connection.execute(f'SELECT {SQL_FUNCTION}(?)', ('1/500',)).fetchone()[0] == .002
        finally:
            connection.close()
        actual = []
        for page in range(1, math.ceil(len(EXPOSURES) / 3) + 1):
            response = client.get('/api/photos', params={
                'sort': 'shutter_speed', 'sort_direction': direction, 'per_page': 3, 'page': page,
            })
            assert response.status_code == 200, response.text
            actual += [photo['path'] for photo in response.json()['photos']]
        assert actual == expected
        response = client.get('/api/photos/select_bottom_percent', params={
            'keep_percent': 50, 'sort': 'shutter_speed', 'sort_direction': direction,
        })
        assert response.status_code == 200, response.text
        assert response.json()['paths'] == expected[7:]


def test_sort_validation_preserves_sql_boundary():
    assert _resolve_order_by({'sort': 'shutter_speed); DROP TABLE photos;--', 'dir': 'ASC'}) == 'aggregate ASC, path ASC'
    assert _resolve_order_by({'sort': 'shutter_speed', 'dir': 'ASC; DROP TABLE photos'}) == (
        f'({SQL_FUNCTION}(shutter_speed) IS NULL) ASC, {SQL_FUNCTION}(shutter_speed) DESC, path ASC'
    )


@pytest.mark.parametrize('direction,expected', [('ASC', ASCENDING), ('DESC', DESCENDING)])
@pytest.mark.parametrize('smart', [False, True])
def test_album_sort_and_cover_use_same_exposure_order(tmp_path, direction, expected, smart):
    path = str(tmp_path / 'albums.db')
    _make_db(path, [_photo(p, '2020:01:01 00:00:00', shutter_speed=value) for p, value in EXPOSURES])
    filters = json.dumps({'sort': 'shutter_speed', 'sort_direction': direction}) if smart else None
    with sqlite3.connect(path) as connection:
        connection.execute(
            'INSERT INTO albums(id, user_id, name, is_smart, smart_filter_json, share_token) '
            'VALUES(1, NULL, ?, ?, ?, ?)', ('exposures', smart, filters, 'exposure-test'),
        )
        connection.executemany(
            'INSERT INTO album_photos(album_id, photo_path, position) VALUES(1, ?, ?)',
            [(p, i) for i, (p, _) in enumerate(EXPOSURES)],
        )
    with (
        mock.patch.object(database, 'DEFAULT_DB_PATH', path),
        mock.patch('api.routers.albums.VIEWER_CONFIG', _VIEWER_CONFIG),
        mock.patch('api.db_helpers._existing_columns_cache', _existing_columns(path)),
        TestClient(_create_app_no_auth()) as client,
    ):
        response = client.get('/api/albums/1/photos', params={
            'sort': 'shutter_speed', 'sort_direction': direction,
        })
        assert response.status_code == 200, response.text
        assert [p['path'] for p in response.json()['photos']] == expected
        response = client.get('/api/shared/album/1', params={
            'token': 'exposure-test', 'sort': 'shutter_speed', 'sort_direction': direction,
        })
        assert response.status_code == 200, response.text
        assert [p['path'] for p in response.json()['photos']] == expected
        if smart:
            assert warm_smart_album_covers(path) == 1
            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                album = connection.execute('SELECT * FROM albums WHERE id = 1').fetchone()
                cover = connection.execute(
                    'SELECT value FROM stats_cache WHERE key = ?', (_cover_cache_key(album),),
                ).fetchone()[0]
                assert cover == expected[0]
