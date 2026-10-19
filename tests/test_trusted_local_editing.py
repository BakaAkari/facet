"""Real auth dependencies/routes, synthetic in-memory data only."""

from contextlib import contextmanager
from unittest.mock import Mock
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import auth
from api.routers import auth as auth_routes, faces, comparison


@pytest.fixture
def local(monkeypatch):
    cfg = {
        "trusted_local_editing": True,
        "password": "",
        "edition_password": "existing",
        "features": {},
    }
    monkeypatch.setattr(auth, "VIEWER_CONFIG", cfg)
    monkeypatch.setattr(auth_routes, "VIEWER_CONFIG", cfg)
    monkeypatch.setattr(auth, "config_load_failed", lambda: False)
    monkeypatch.setattr(auth, "is_multi_user_enabled", lambda: False)
    monkeypatch.setattr(auth_routes, "is_multi_user_enabled", lambda: False)
    monkeypatch.setattr(faces, "is_multi_user_enabled", lambda: False)
    monkeypatch.setattr("api.raw_processing.get_darktable_profiles", lambda: [])
    monkeypatch.delenv("FACET_VIEWER_READ_ONLY", raising=False)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE photos(path TEXT PRIMARY KEY,star_rating INTEGER,is_favorite INTEGER,is_rejected INTEGER)"
    )
    conn.execute("INSERT INTO photos VALUES('synthetic',0,0,0)")
    conn.commit()

    @contextmanager
    def db():
        yield conn

    monkeypatch.setattr(faces, "get_db", db)
    monkeypatch.setattr(faces, "assert_photo_visible", lambda *a: None)
    monkeypatch.setattr(faces, "_mint_rating_comparisons", lambda *a: None)
    manager = Mock()
    manager.submit_comparison.return_value = True
    manager.get_statistics.return_value = {}
    monkeypatch.setattr("comparison.ComparisonManager", lambda *a: manager)
    app = FastAPI()
    for router in (auth_routes.router, faces.router, comparison.router):
        app.include_router(router)
    yield cfg, app, conn, manager
    conn.close()


def client(app, host="http://localhost", peer="127.0.0.1", headers=None):
    return TestClient(app, base_url=host, client=(peer, 12345), headers=headers or {})


@pytest.mark.parametrize("edition_password", ["existing", ""])
def test_status_and_actual_rating_favorite_comparison(local, edition_password):
    cfg, app, conn, manager = local
    cfg["edition_password"] = edition_password
    c = client(app)
    status = c.get("/api/auth/status").json()
    assert (
        status["authenticated"]
        and status["edition_authenticated"]
        and status["edition_enabled"]
    )
    assert not status["edition_password_required"]
    assert (
        c.post(
            "/api/photo/set_rating", json={"photo_path": "synthetic", "rating": 4}
        ).status_code
        == 200
    )
    assert (
        c.post(
            "/api/photo/toggle_favorite", json={"photo_path": "synthetic"}
        ).status_code
        == 200
    )
    assert tuple(
        conn.execute("SELECT star_rating,is_favorite FROM photos").fetchone()
    ) == (4, 1)
    assert (
        c.post(
            "/api/comparison/submit",
            json={"photo_a": "a", "photo_b": "b", "winner": "a"},
        ).status_code
        == 200
    )
    manager.submit_comparison.assert_called_once()


@pytest.mark.parametrize(
    "host,peer,headers",
    [
        ("http://localhost", "192.0.2.1", {}),
        ("http://evil.example", "127.0.0.1", {}),
        ("http://localhost.evil.example", "127.0.0.1", {}),
        ("http://127.0.0.2", "127.0.0.1", {}),
        ("http://localhost", "127.0.0.1", {"Origin": "https://evil.example"}),
        ("http://localhost", "127.0.0.1", {"Origin": "null"}),
        ("http://localhost", "127.0.0.1", {"Origin": "http://localhost:9999"}),
        ("http://localhost", "127.0.0.1", {"Referer": "http://evil.example/page"}),
        ("http://localhost", "127.0.0.1", {"Sec-Fetch-Site": "cross-site"}),
        ("http://localhost", "127.0.0.1", {"Sec-Fetch-Site": "same-site"}),
        ("http://localhost", "127.0.0.1", {"X-Forwarded-For": "127.0.0.1"}),
        ("http://localhost", "127.0.0.1", {"Forwarded": "for=127.0.0.1"}),
        ("http://localhost", "127.0.0.1", {"Via": "1.1 local-proxy"}),
        ("http://localhost", "127.0.0.1", {"X-Real-IP": "192.0.2.1"}),
        ("http://localhost", "127.0.0.1", {"X-Forwarded-Host": "localhost"}),
    ],
)
def test_untrusted_requests_cannot_edit_including_get(local, host, peer, headers):
    cfg, app, conn, manager = local
    c = client(app, host, peer, headers)
    assert not c.get("/api/auth/status").json()["edition_authenticated"]
    assert c.get("/api/comparison/next_pair").status_code == 403
    assert (
        c.post(
            "/api/photo/set_rating", json={"photo_path": "synthetic", "rating": 5}
        ).status_code
        == 403
    )
    assert (
        c.post(
            "/api/photo/toggle_favorite", json={"photo_path": "synthetic"}
        ).status_code
        == 403
    )
    assert (
        c.post(
            "/api/comparison/submit",
            json={"photo_a": "a", "photo_b": "b", "winner": "a"},
        ).status_code
        == 403
    )
    assert tuple(
        conn.execute("SELECT star_rating,is_favorite FROM photos").fetchone()
    ) == (0, 0)
    manager.submit_comparison.assert_not_called()


@pytest.mark.parametrize(
    "host,peer,headers",
    [
        (
            "http://127.0.0.1:9000",
            "127.0.0.1",
            {"Origin": "http://127.0.0.1:9000", "Sec-Fetch-Site": "same-origin"},
        ),
        (
            "http://localhost",
            "::1",
            {"Host": "[::1]:9000", "Origin": "http://[::1]:9000"},
        ),
        ("http://localhost", "127.0.0.1", {"Sec-Fetch-Site": "none"}),
    ],
)
def test_direct_local_origins(local, host, peer, headers):
    assert (
        client(local[1], host, peer, headers)
        .get("/api/auth/status")
        .json()["edition_authenticated"]
    )


@pytest.mark.parametrize(
    "case",
    ["disabled", "nonbool", "browse", "failed", "multi", "share", "forged_claim"],
)
def test_default_failclosed_and_tokens(local, monkeypatch, case):
    cfg, app, conn, manager = local
    headers = {}
    if case == "disabled":
        cfg.pop("trusted_local_editing")
    if case == "nonbool":
        cfg["trusted_local_editing"] = "true"
    if case == "browse":
        cfg["password"] = "locked"
    if case == "failed":
        monkeypatch.setattr(auth, "config_load_failed", lambda: True)
    if case == "multi":
        monkeypatch.setattr(auth, "is_multi_user_enabled", lambda: True)
    if case in ("share", "forged_claim"):
        if case == "forged_claim":
            cfg["trusted_local_editing"] = False
        token = auth.create_access_token(
            {
                "sub": "x",
                "role": "share_client" if case == "share" else "user",
                "trusted_local_editing": True,
            }
        )
        headers["Authorization"] = "Bearer " + token
    status = client(app, headers=headers).get("/api/auth/status").json()
    assert not status["edition_authenticated"]
    assert client(app, headers=headers).post(
        "/api/photo/set_rating", json={"photo_path": "synthetic", "rating": 5}
    ).status_code in (401, 403)
    assert conn.execute("SELECT star_rating FROM photos").fetchone()[0] == 0
