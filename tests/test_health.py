"""Tests for the health and readiness check endpoints (api/routers/health.py)."""

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api import create_app


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_liveness_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        # No build was started in this process (no lifespan): nothing pending.
        assert resp.json() == {
            "status": "ok",
            "gallery_indexes": {"state": "ready", "done": 0, "total": 0},
        }

    def test_liveness_reports_a_running_gallery_index_build(self, client, monkeypatch):
        from api import schema_startup
        monkeypatch.setattr(schema_startup, "_status",
                            {"state": "building", "done": 3, "total": 8})
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["gallery_indexes"] == {"state": "building", "done": 3, "total": 8}

    def test_liveness_never_opens_the_database(self, client):
        """The app-wide schema warm-up skips /health: with every schema cache cold,
        liveness still opens no connection, while any other route does (control)."""
        import sqlite3
        from tests.test_async_routes_do_not_block import _cold_caches
        real_connect = sqlite3.connect
        opened = []

        def recording_connect(*args, **kwargs):
            opened.append(args[0] if args else kwargs.get("database"))
            return real_connect(*args, **kwargs)

        _cold_caches()
        with mock.patch.object(sqlite3, "connect", recording_connect):
            resp = client.get("/health")
            assert resp.status_code == 200
            assert opened == []
            client.get("/ready")
        assert opened, "control: a cold warm-up on any other route opens the database"


class TestSecurityHeaders:
    def test_safe_headers_always_present(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_default_csp_present_and_spa_safe(self, client):
        resp = client.get("/health")
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        # SPA needs inline script (theme bootstrap) + Google Fonts + OSM tiles.
        assert "'unsafe-inline'" in csp
        assert "https://fonts.googleapis.com" in csp
        assert "img-src 'self' data: blob: https:" in csp

    def test_hsts_off_by_default(self, client):
        resp = client.get("/health")
        assert "Strict-Transport-Security" not in resp.headers


class TestReadyEndpoint:
    """Tests target the async /ready endpoint that uses get_async_db()."""

    @staticmethod
    def _async_cm(conn):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _ctx():
            yield conn
        return _ctx

    @staticmethod
    def _make_async_conn(execute_side_effect=None):
        class _Cursor:
            async def fetchone(self): return (1,)
            async def close(self): pass
        class _Conn:
            async def execute(self, *a, **kw):
                if execute_side_effect is not None:
                    raise execute_side_effect
                return _Cursor()
        return _Conn()

    def test_ready_when_database_accessible(self, client):
        conn = self._make_async_conn()
        with mock.patch("api.routers.health.get_async_db", self._async_cm(conn)):
            resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["checks"]["database"] == "ok"

    def test_not_ready_when_database_unavailable(self, client):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _broken():
            raise Exception("connection refused")
            yield  # unreachable
        with mock.patch("api.routers.health.get_async_db", _broken):
            resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["database"] == "unavailable"

    def test_not_ready_when_query_fails(self, client):
        conn = self._make_async_conn(execute_side_effect=Exception("disk I/O error"))
        with mock.patch("api.routers.health.get_async_db", self._async_cm(conn)):
            resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["database"] == "unavailable"
