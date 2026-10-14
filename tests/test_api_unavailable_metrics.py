"""``/api/config`` ``unavailable_metrics`` and the commercial-mode sort filter."""

import licensing

_HIDDEN = {'liqe_score', 'aesthetic_iaa'}


def _sort_columns(body):
    flat = {c for c, _ in body['sort_options']}
    grouped = {o['column'] for opts in (body.get('sort_options_grouped') or {}).values() for o in opts}
    return flat, grouped


def test_default_mode_reports_nothing_unavailable(edition_client):
    body = edition_client.get('/api/config').json()
    assert body['unavailable_metrics'] == []
    flat, grouped = _sort_columns(body)
    assert _HIDDEN <= flat and _HIDDEN <= grouped


def test_commercial_mode_lists_column_names_and_drops_sorts(edition_client):
    licensing.set_policy(licensing.MODE_COMMERCIAL)
    body = edition_client.get('/api/config').json()
    assert _HIDDEN <= set(body['unavailable_metrics'])
    assert 'liqe' not in body['unavailable_metrics']  # column name, not weight key
    flat, grouped = _sort_columns(body)
    hidden = set(body['unavailable_metrics'])
    assert not hidden & flat
    assert not hidden & grouped
    assert 'aggregate' in flat
