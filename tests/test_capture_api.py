"""HTTP bypass tests: use production schema and real temporary source files."""
import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from db.schema import init_database


@pytest.mark.parametrize('kind', ['pixel_shift', 'timelapse'])
@pytest.mark.parametrize('operation', ['move', 'delete', 'reject', 'confirm', 'override'])
def test_protected_http_writes_refused(tmp_path, edition_client, kind, operation):
    source = tmp_path / 'source.raf'
    source.write_bytes(b'fixture-original-do-not-remove')
    db = str(tmp_path / 'library.db')
    init_database(db)
    with sqlite3.connect(db) as conn:
        conn.execute('INSERT INTO photos(path, filename, sequence_kind, sequence_group_id, '
                     'is_rejected) VALUES (?, ?, ?, 1, ?)',
                     (str(source), source.name, kind, int(operation == 'move')))

    @contextmanager
    def connection():
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    from api.auth import CurrentUser, require_auth
    edition_client.app.dependency_overrides[require_auth] = lambda: CurrentUser(
        user_id='test', role='admin', edition_authenticated=True)
    with (patch('api.routers.export.get_db', connection),
          patch('api.routers.faces.get_db', connection),
          patch('api.routers.burst_culling.get_db', connection),
          patch('api.routers.export._require_trash_available') as trash):
        if operation == 'move':
            response = edition_client.post('/api/cull/apply', json={
                'paths': [str(source)], 'action': 'move_rejects',
                'target_dir': str(tmp_path / 'rejects'), 'dry_run': False})
        elif operation == 'delete':
            response = edition_client.post('/api/photo/delete', json={
                'paths': [str(source)], 'dry_run': False,
                'include_sequence_siblings': True})
        elif operation == 'reject':
            response = edition_client.post('/api/photo/toggle_rejected',
                                           json={'photo_path': str(source)})
        elif operation == 'confirm':
            response = edition_client.post('/api/culling-groups/confirm', json={
                'group_id': 1, 'type': kind, 'paths': [str(source)], 'keep_paths': []})
        else:
            response = edition_client.post('/api/culling-groups/override_sequence', json={
                'paths': [str(source)], 'kind': None})
        assert response.status_code == 409, response.text
        trash.return_value.send2trash.assert_not_called()
    assert source.read_bytes() == b'fixture-original-do-not-remove'
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT sequence_kind FROM photos').fetchone()[0] == kind
