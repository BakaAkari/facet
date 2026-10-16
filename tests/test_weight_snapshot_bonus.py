"""Weight snapshots carry the category bonus, so undo restores it too (T9b, G3).

``apply_optimized_weights`` now re-centres ``modifiers.bonus`` together with the
weights. A snapshot that kept only the weights could not undo that: restoring
it would leave the shifted bonus behind and re-level the category the other way.
``weight_config_snapshots.category_bonus`` (NULL = not recorded, as for every row
written before the column existed and every ``calibrate.py`` snapshot) holds it.
"""

import json
from contextlib import contextmanager
from unittest import mock

import pytest

from config import ScoringConfig
from config_resolve import load_resolved
from db import get_connection, init_database, record_weight_snapshot
from optimization.weight_optimizer import WeightOptimizer

_AUTH_MODULE = 'api.auth'
_CMP_MODULE = 'api.routers.comparison'

_VIEWER_CONFIG = {
    'password': '',
    'edition_password': 'secret',
    'features': {},
    'display': {'image_jpeg_quality': 96},
}


@pytest.fixture(autouse=True)
def _patch_config():
    with (
        mock.patch(f'{_AUTH_MODULE}.VIEWER_CONFIG', _VIEWER_CONFIG),
        mock.patch(f'{_AUTH_MODULE}.is_multi_user_enabled', return_value=False),
    ):
        yield


@pytest.fixture()
def env(tmp_path):
    db = str(tmp_path / 't.db')
    init_database(db)
    with get_connection(db, row_factory=False) as conn:
        for i in range(10):
            conn.execute(
                "INSERT INTO photos (path, filename, category, aesthetic, liqe_score) "
                "VALUES (?, ?, 'portrait', 8.0, 2.0)", (f'/s/p{i}.jpg', f'p{i}.jpg'))
        conn.commit()
    cfg = tmp_path / 'scoring_config.json'
    cfg.write_text(json.dumps({'categories': [{
        'name': 'portrait',
        'weights': {'aesthetic_percent': 50, 'liqe_percent': 50},
        'modifiers': {'bonus': 0.5},
    }]}))
    return {'db': db, 'cfg': str(cfg)}


def _real_db_cm(db_path):
    @contextmanager
    def _ctx():
        with get_connection(db_path) as conn:
            yield conn
    return _ctx


@contextmanager
def _routes(env):
    with (
        mock.patch(f'{_CMP_MODULE}._CONFIG_PATH', env['cfg']),
        mock.patch(f'{_CMP_MODULE}.reload_config', lambda: None),
        mock.patch(f'{_CMP_MODULE}.get_db', _real_db_cm(env['db'])),
        mock.patch(f'{_CMP_MODULE}.server_scoring_config',
                   lambda validate=False: ScoringConfig(env['cfg'], validate=False)),
    ):
        yield


def _entry(env):
    return next(c for c in load_resolved(env['cfg'])['categories'] if c['name'] == 'portrait')


def _snapshot_rows(env, created_by=None):
    sql = 'SELECT id, category, weights, created_by, category_bonus FROM weight_config_snapshots'
    params = ()
    if created_by:
        sql += ' WHERE created_by = ?'
        params = (created_by,)
    with get_connection(env['db']) as conn:
        return [dict(r) for r in conn.execute(sql + ' ORDER BY id', params).fetchall()]


def _restore(client, env, snapshot_id):
    with _routes(env):
        return client.post('/api/config/restore_weights', json={'snapshot_id': snapshot_id})


def _apply(env, weights=None, category='portrait'):
    return WeightOptimizer(env['db'], env['cfg']).apply_optimized_weights(
        weights or {'aesthetic': 0.667, 'liqe': 0.333}, category)


def test_restore_returns_the_weights_and_the_bonus(env, edition_client):
    snapshot_id = _apply(env)
    assert _entry(env)['modifiers']['bonus'] == pytest.approx(-0.5, abs=0.01)
    assert _snapshot_rows(env)[0]['category_bonus'] == pytest.approx(0.5)

    resp = _restore(edition_client, env, snapshot_id)

    assert resp.status_code == 200, resp.text
    entry = _entry(env)
    assert entry['weights']['aesthetic_percent'] == 50
    assert entry['modifiers']['bonus'] == pytest.approx(0.5)


def test_restoring_a_snapshot_without_a_bonus_leaves_the_bonus_alone(env, edition_client):
    snapshot_id = record_weight_snapshot(
        'portrait', {'aesthetic_percent': 80, 'liqe_percent': 20}, created_by='manual', db=env['db'])
    assert _snapshot_rows(env)[0]['category_bonus'] is None
    cfg = json.loads(open(env['cfg']).read())
    cfg['categories'][0]['modifiers']['bonus'] = 2.0
    open(env['cfg'], 'w').write(json.dumps(cfg))

    resp = _restore(edition_client, env, snapshot_id)

    assert resp.status_code == 200, resp.text
    entry = _entry(env)
    assert entry['weights']['aesthetic_percent'] == 80
    assert entry['modifiers']['bonus'] == pytest.approx(2.0)


def test_restore_then_restore_the_pre_restore_snapshot_round_trips_the_bonus(env, edition_client):
    snapshot_id = _apply(env)                      # bonus 0.5 -> about -0.5
    shifted = _entry(env)['modifiers']['bonus']

    assert _restore(edition_client, env, snapshot_id).status_code == 200
    assert _entry(env)['modifiers']['bonus'] == pytest.approx(0.5)

    pre_restore = _snapshot_rows(env, 'auto:pre_restore')
    assert len(pre_restore) == 1
    assert pre_restore[0]['category_bonus'] == pytest.approx(shifted)

    assert _restore(edition_client, env, pre_restore[0]['id']).status_code == 200
    entry = _entry(env)
    assert entry['modifiers']['bonus'] == pytest.approx(shifted)
    assert entry['weights']['aesthetic_percent'] == pytest.approx(66.7)


def test_save_snapshot_stores_the_current_bonus(env, edition_client):
    with _routes(env):
        resp = edition_client.post('/api/config/save_snapshot', json={'category': 'portrait'})
    assert resp.status_code == 200, resp.text
    rows = _snapshot_rows(env)
    assert [r['category_bonus'] for r in rows] == [pytest.approx(0.5)]


def test_a_category_without_a_bonus_key_snapshots_and_restores_a_zero_bonus(env, edition_client):
    """street carries no ``modifiers.bonus`` (the scorer reads that as 0), so the
    manual snapshot stores 0 like the optimizer and viewer-edit writers do, and a
    restore after a bonus was re-centred puts it back to 0 rather than leaving it."""
    cfg = json.loads(open(env['cfg']).read())
    cfg['categories'].append({'name': 'street', 'weights': {'aesthetic_percent': 50, 'liqe_percent': 50}})
    open(env['cfg'], 'w').write(json.dumps(cfg))

    with _routes(env):
        resp = edition_client.post('/api/config/save_snapshot', json={'category': 'street'})
    assert resp.status_code == 200, resp.text
    snapshot_id = resp.json()['snapshot_id']
    assert _snapshot_rows(env)[0]['category_bonus'] == pytest.approx(0.0)

    cfg = json.loads(open(env['cfg']).read())
    street = next(c for c in cfg['categories'] if c['name'] == 'street')
    street['modifiers'] = {'bonus': -0.4}
    open(env['cfg'], 'w').write(json.dumps(cfg))

    resp = _restore(edition_client, env, snapshot_id)

    assert resp.status_code == 200, resp.text
    entry = next(c for c in load_resolved(env['cfg'])['categories'] if c['name'] == 'street')
    assert entry['modifiers']['bonus'] == pytest.approx(0.0)


def test_the_viewers_own_weight_edit_records_the_bonus(env, edition_client):
    with _routes(env):
        resp = edition_client.post('/api/config/update_weights', json={
            'category': 'portrait', 'weights': {'aesthetic_percent': 60, 'liqe_percent': 40}})
    assert resp.status_code == 200, resp.text
    rows = _snapshot_rows(env, 'auto:edit')
    assert len(rows) == 1 and rows[0]['category_bonus'] == pytest.approx(0.5)


def test_the_others_category_records_no_bonus(env):
    _apply(env, {'aesthetic': 0.7, 'liqe': 0.3}, category='others')
    rows = _snapshot_rows(env, 'auto:optimizer')
    assert len(rows) == 1
    assert rows[0]['category'] == 'others'
    assert rows[0]['category_bonus'] is None


def test_the_list_response_still_validates_with_the_column_present(env, edition_client):
    _apply(env)
    with _routes(env):
        resp = edition_client.get('/api/config/weight_snapshots')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body['snapshots']) == 1
    assert 'category_bonus' not in body['snapshots'][0]


def test_an_existing_database_gets_the_column(tmp_path):
    """A database created before the column existed is migrated by init_database."""
    import sqlite3
    db = str(tmp_path / 'old.db')
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE weight_config_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TEXT DEFAULT (datetime('now')), category TEXT, weights TEXT NOT NULL, "
        "description TEXT, accuracy_before REAL, accuracy_after REAL, comparisons_used INTEGER, "
        "created_by TEXT)")
    conn.commit()
    conn.close()
    init_database(db)
    with get_connection(db) as c:
        cols = {r[1] for r in c.execute('PRAGMA table_info(weight_config_snapshots)')}
    assert 'category_bonus' in cols
