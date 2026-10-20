"""``viewer.trusted_local_editing``: the real auth dependencies against the shared session DB.

The opt-in grants edition to a direct loopback browser on a single-user install
that has NEITHER a browse password nor an edition password. Every other request
-- forwarded, cross-origin, non-loopback, carrying a token on a locked install --
fails closed, and a configured ``viewer.edition_password`` is always required.

The fixture install is OPEN on both passwords, so every refusal below is the
trusted-local guard refusing, never an edition password doing the work.
"""

import sqlite3
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import auth, db_helpers
from api.routers import auth as auth_routes, comparison, faces, scan
from db import DEFAULT_DB_PATH

PREFIX = "/trusted-local-editing/"
PHOTO = PREFIX + "p.jpg"
RATE = ("/api/photo/set_rating", {"photo_path": PHOTO, "rating": 4})
FAVORITE = ("/api/photo/toggle_favorite", {"photo_path": PHOTO})
COMPARE = ("/api/comparison/submit", {"photo_a": "a", "photo_b": "b", "winner": "a"})
OPEN_DETAIL = auth.OPEN_INSTALL_EDIT_DETAIL


@pytest.fixture()
def _flush_rating_sync():
    # Requested before ``seed_photos_prefix`` so it tears down AFTER the seeded
    # rows are deleted: the debounced rating sync then rebuilds the shared
    # comparisons table without this module's photo.
    yield
    faces.flush_rating_comparisons()


@pytest.fixture()
def local(monkeypatch, _flush_rating_sync, seed_photos_prefix):
    cfg = {
        "trusted_local_editing": True,
        "password": "",
        "edition_password": "",
        "features": {},
    }
    monkeypatch.setattr(auth, "VIEWER_CONFIG", cfg)
    monkeypatch.setattr(auth_routes, "VIEWER_CONFIG", cfg)
    monkeypatch.setattr(auth, "config_load_failed", lambda: False)
    for module in (auth, auth_routes, faces, db_helpers):
        monkeypatch.setattr(module, "is_multi_user_enabled", lambda: False)
    monkeypatch.setattr("api.raw_processing.get_darktable_profiles", lambda: [])
    monkeypatch.delenv("FACET_VIEWER_READ_ONLY", raising=False)
    seed_photos_prefix(PREFIX, [
        {"path": PHOTO, "filename": "p.jpg", "star_rating": 0, "is_favorite": 0, "is_rejected": 0},
    ])
    manager = Mock()
    manager.submit_comparison.return_value = True
    manager.get_statistics.return_value = {}
    monkeypatch.setattr("comparison.ComparisonManager", lambda *a: manager)
    app = FastAPI()
    for router in (auth_routes.router, faces.router, comparison.router, scan.router):
        app.include_router(router)
    yield cfg, app, manager


def _client(app, host="http://localhost", peer="127.0.0.1"):
    return TestClient(app, base_url=host, client=(peer, 12345))


def _stored():
    with sqlite3.connect(DEFAULT_DB_PATH) as conn:
        return conn.execute(
            "SELECT star_rating, is_favorite FROM photos WHERE path = ?", (PHOTO,)
        ).fetchone()


def _user_token(role="user", **claims):
    return "Bearer " + auth.create_access_token({"sub": "x", "role": role, **claims})


def test_trusted_local_request_rates_favorites_and_compares(local):
    cfg, app, manager = local
    c = _client(app)
    status = c.get("/api/auth/status").json()
    assert status["authenticated"] and status["edition_authenticated"] and status["edition_enabled"]
    assert not status["edition_password_required"]
    for path, body in (RATE, FAVORITE, COMPARE):
        assert c.post(path, json=body).status_code == 200, path
    assert _stored() == (4, 1)
    manager.submit_comparison.assert_called_once()


def test_trusted_local_request_with_a_user_token_on_an_open_install_is_granted(local):
    cfg, app, manager = local
    headers = {"Authorization": _user_token()}
    c = _client(app)
    assert c.get("/api/auth/status", headers=headers).json()["edition_authenticated"]
    assert c.post(RATE[0], json=RATE[1], headers=headers).status_code == 200
    assert _stored() == (4, 0)


def test_configured_edition_password_is_still_required(local):
    """An operator who set an edition password expects it to be asked for."""
    cfg, app, manager = local
    cfg["edition_password"] = "existing"
    c = _client(app)
    status = c.get("/api/auth/status").json()
    assert not status["edition_authenticated"]
    assert status["edition_password_required"]
    for path, body in (RATE, FAVORITE, COMPARE):
        response = c.post(path, json=body)
        assert response.status_code == 403, path
        assert response.json()["detail"] != OPEN_DETAIL
    assert _stored() == (0, 0)
    manager.submit_comparison.assert_not_called()


@pytest.mark.parametrize(
    "edition_password,detail",
    [
        pytest.param("", "Set viewer.edition_password to enable the scan trigger", id="open-install"),
        pytest.param("existing", "Edition access required", id="edition-password-set"),
    ],
)
def test_trusted_local_request_cannot_trigger_a_scan(local, edition_password, detail):
    cfg, app, manager = local
    cfg["edition_password"] = edition_password
    response = _client(app).get("/api/scan/status")
    assert response.status_code == 403
    assert response.json()["detail"] == detail


def _refused(app, host, peer, headers):
    c = _client(app, host, peer)
    assert not c.get("/api/auth/status", headers=headers).json()["edition_authenticated"]
    assert c.get("/api/comparison/next_pair", headers=headers).status_code == 403
    for path, body in (RATE, FAVORITE, COMPARE):
        response = c.post(path, json=body, headers=headers)
        assert response.status_code == 403, path
        assert response.json()["detail"] == OPEN_DETAIL


@pytest.mark.parametrize(
    "host,peer,headers",
    [
        pytest.param("http://localhost", "192.0.2.1", [], id="non-loopback-peer"),
        pytest.param("http://localhost", "10.0.0.1", [], id="private-peer"),
        pytest.param("http://localhost", "::ffff:192.0.2.1", [], id="ipv4-mapped-non-loopback-peer"),
        pytest.param("http://localhost", "testclient", [], id="non-ip-peer"),
        pytest.param("http://evil.example", "127.0.0.1", [], id="foreign-host"),
        pytest.param("http://localhost.evil.example", "127.0.0.1", [], id="suffixed-host"),
        pytest.param("http://127.0.0.2", "127.0.0.1", [], id="other-loopback-host"),
        pytest.param("http://localhost", "127.0.0.1", [("Host", "localhost"), ("Host", "localhost")],
                     id="duplicate-host"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "https://evil.example")], id="foreign-origin"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "null")], id="null-origin"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "http://localhost:9999")], id="port-mismatch"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "https://localhost:80")], id="scheme-mismatch"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "http://u@localhost")], id="origin-username"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "http://u:p@localhost")], id="origin-userinfo"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "http://localhost/x")], id="origin-path"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "http://localhost?q=1")], id="origin-query"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "http://localhost#f")], id="origin-fragment"),
        pytest.param("http://localhost", "127.0.0.1",
                     [("Origin", "http://localhost"), ("Origin", "http://localhost")], id="duplicate-origin"),
        pytest.param("http://localhost", "127.0.0.1", [("Referer", "http://evil.example/page")],
                     id="foreign-referer"),
        pytest.param("http://localhost", "127.0.0.1", [("Referer", "http://u@localhost/page")],
                     id="referer-userinfo"),
        pytest.param("http://localhost", "127.0.0.1",
                     [("Referer", "http://localhost/a"), ("Referer", "http://localhost/b")],
                     id="duplicate-referer"),
        pytest.param("http://localhost", "127.0.0.1", [("Sec-Fetch-Site", "cross-site")], id="cross-site"),
        pytest.param("http://localhost", "127.0.0.1", [("Sec-Fetch-Site", "same-site")], id="same-site"),
        pytest.param("http://localhost", "127.0.0.1",
                     [("Sec-Fetch-Site", "same-origin"), ("Sec-Fetch-Site", "same-origin")],
                     id="duplicate-sec-fetch-site"),
        pytest.param("http://localhost", "127.0.0.1", [("X-Forwarded-For", "127.0.0.1")], id="x-forwarded-for"),
        pytest.param("http://localhost", "127.0.0.1", [("X-Forwarded-Host", "localhost")], id="x-forwarded-host"),
        pytest.param("http://localhost", "127.0.0.1", [("Forwarded", "for=127.0.0.1")], id="forwarded"),
        pytest.param("http://localhost", "127.0.0.1", [("Via", "1.1 local-proxy")], id="via"),
        pytest.param("http://localhost", "127.0.0.1", [("X-Real-IP", "192.0.2.1")], id="x-real-ip"),
        pytest.param("http://localhost", "127.0.0.1", [("X-Client-IP", "192.0.2.1")], id="x-client-ip"),
        pytest.param("http://localhost", "127.0.0.1", [("True-Client-IP", "192.0.2.1")], id="true-client-ip"),
        pytest.param("http://localhost", "127.0.0.1", [("CF-Connecting-IP", "192.0.2.1")],
                     id="cf-connecting-ip"),
        pytest.param("http://localhost", "127.0.0.1", [("X-Original-Forwarded-For", "192.0.2.1")],
                     id="x-original-forwarded-for"),
    ],
)
def test_untrusted_requests_cannot_edit_including_get(local, host, peer, headers):
    cfg, app, manager = local
    _refused(app, host, peer, headers)
    assert _stored() == (0, 0)
    manager.submit_comparison.assert_not_called()


@pytest.mark.parametrize(
    "host,peer,headers",
    [
        pytest.param("http://127.0.0.1:9000", "127.0.0.1",
                     [("Origin", "http://127.0.0.1:9000"), ("Sec-Fetch-Site", "same-origin")],
                     id="ipv4-same-origin"),
        pytest.param("http://localhost", "::1", [("Host", "[::1]:9000"), ("Origin", "http://[::1]:9000")],
                     id="ipv6-same-origin"),
        pytest.param("http://localhost", "127.0.0.1", [("Origin", "http://localhost:80")], id="default-port"),
        pytest.param("http://localhost", "127.0.0.1", [("Referer", "http://localhost/gallery?sort=date")],
                     id="same-origin-referer"),
        pytest.param("http://localhost", "127.0.0.1", [("Sec-Fetch-Site", "none")], id="typed-url"),
        pytest.param("http://localhost", "::ffff:127.0.0.1", [], id="ipv4-mapped-loopback-peer"),
    ],
)
def test_direct_local_origins(local, host, peer, headers):
    status = _client(local[1], host, peer).get("/api/auth/status", headers=headers)
    assert status.json()["edition_authenticated"]


def _disabled(cfg, monkeypatch):
    cfg.pop("trusted_local_editing")
    return {}


def _non_bool(cfg, monkeypatch):
    cfg["trusted_local_editing"] = "true"
    return {}


def _browse_password(cfg, monkeypatch):
    cfg["password"] = "locked"
    return {}


def _browse_password_with_user_token(cfg, monkeypatch):
    cfg["password"] = "locked"
    return {"Authorization": _user_token()}


def _config_load_failed(cfg, monkeypatch):
    monkeypatch.setattr(auth, "config_load_failed", lambda: True)
    return {}


def _multi_user(cfg, monkeypatch):
    monkeypatch.setattr(auth, "is_multi_user_enabled", lambda: True)
    return {}


def _share_token(cfg, monkeypatch):
    return {"Authorization": _user_token("share_client", trusted_local_editing=True)}


def _forged_claim(cfg, monkeypatch):
    cfg["trusted_local_editing"] = False
    return {"Authorization": _user_token(trusted_local_editing=True)}


@pytest.mark.parametrize(
    "setup,expected",
    [
        pytest.param(_disabled, 403, id="disabled"),
        pytest.param(_non_bool, 403, id="non-bool"),
        pytest.param(_browse_password, 401, id="browse-password"),
        pytest.param(_browse_password_with_user_token, 403, id="browse-password-user-token"),
        pytest.param(_config_load_failed, 401, id="config-load-failed"),
        pytest.param(_multi_user, 401, id="multi-user"),
        pytest.param(_share_token, 401, id="share-token"),
        pytest.param(_forged_claim, 403, id="forged-token-claim"),
    ],
)
def test_default_fail_closed_and_tokens(local, monkeypatch, setup, expected):
    cfg, app, manager = local
    headers = setup(cfg, monkeypatch)
    c = _client(app)
    assert not c.get("/api/auth/status", headers=headers).json()["edition_authenticated"]
    assert c.post(RATE[0], json=RATE[1], headers=headers).status_code == expected
    assert _stored() == (0, 0)
