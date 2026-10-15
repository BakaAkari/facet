"""Tests for cross-cutting error handling and request logging (api/__init__.py).

These exercise RequestLoggingMiddleware and the global exception handlers on a
throwaway app — no database, auth, or shared fixtures involved.
"""

import logging
import time

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api import (
    RequestLoggingMiddleware,
    unhandled_exception_handler,
    validation_exception_handler,
)


def _make_app(slow_request_ms=1000):
    """Build a minimal app wired with the logging middleware + exception handlers."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, slow_request_ms=slow_request_ms)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/fast")
    def fast():
        return {"ok": True}

    @app.get("/slow")
    def slow():
        time.sleep(0.1)
        return {"ok": True}

    @app.get("/typed")
    def typed(n: int):
        return {"n": n}

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    return app


class TestRequestLogging:
    def test_slow_request_logs_warning(self, caplog):
        client = TestClient(_make_app(slow_request_ms=10))
        with caplog.at_level(logging.INFO, logger="api"):
            resp = client.get("/slow")
        assert resp.status_code == 200
        slow = [r for r in caplog.records
                if r.levelno == logging.WARNING and "SLOW" in r.getMessage()]
        assert slow, "a slow request should log a WARNING with a SLOW marker"

    def test_fast_request_does_not_log_warning(self, caplog):
        client = TestClient(_make_app(slow_request_ms=5000))
        with caplog.at_level(logging.INFO, logger="api"):
            resp = client.get("/fast")
        assert resp.status_code == 200
        assert not [r for r in caplog.records
                    if r.name == "api" and r.levelno >= logging.WARNING]

    def test_failed_request_is_access_logged(self, caplog):
        # A request whose handler raises must still produce an access-log line —
        # the middleware logs in a finally block, not after a successful return.
        client = TestClient(_make_app(), raise_server_exceptions=False)
        with caplog.at_level(logging.INFO, logger="api"):
            resp = client.get("/boom")
        assert resp.status_code == 500
        access = [r for r in caplog.records
                  if r.name == "api" and "/boom" in r.getMessage() and "500" in r.getMessage()]
        assert access, "a failed request must still produce an access-log line"

    def test_static_asset_request_is_not_logged(self, caplog):
        # The middleware deliberately skips access-logging for static assets to
        # cut noise — a request under /assets/ must produce no access-log line.
        app = _make_app()

        @app.get("/assets/app.js")
        def asset():
            return {"ok": True}

        client = TestClient(app)
        with caplog.at_level(logging.INFO, logger="api"):
            resp = client.get("/assets/app.js")
        assert resp.status_code == 200
        assert not [r for r in caplog.records
                    if r.name == "api" and "/assets/app.js" in r.getMessage()]

    def test_zero_threshold_disables_slow_warning(self, caplog):
        # slow_request_ms=0 disables the SLOW escalation (documented behaviour):
        # even a genuinely slow request stays at INFO, never WARNING.
        client = TestClient(_make_app(slow_request_ms=0))
        with caplog.at_level(logging.INFO, logger="api"):
            resp = client.get("/slow")
        assert resp.status_code == 200
        assert not [r for r in caplog.records
                    if r.name == "api" and r.levelno >= logging.WARNING]
        assert [r for r in caplog.records
                if r.name == "api" and r.levelno == logging.INFO and "/slow" in r.getMessage()]


class TestExceptionHandlers:
    def test_validation_error_returns_422_and_logs(self, caplog):
        client = TestClient(_make_app())
        with caplog.at_level(logging.WARNING, logger="api"):
            resp = client.get("/typed", params={"n": "not-an-int"})
        assert resp.status_code == 422
        assert "detail" in resp.json()
        assert [r for r in caplog.records
                if r.levelno == logging.WARNING and "422" in r.getMessage()]

    def test_unhandled_error_returns_clean_json_500(self, caplog):
        client = TestClient(_make_app(), raise_server_exceptions=False)
        with caplog.at_level(logging.ERROR, logger="api"):
            resp = client.get("/boom")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
        assert [r for r in caplog.records if r.levelno == logging.ERROR]


class TestLibraryBusy:
    """A busy/locked database escaping ANY route is a 503 ``library_busy``.

    Driven through the real app against the shared session database while a second
    connection holds its write lock, with the request connections' busy_timeout cut
    to 0.1 s so the lock error arrives quickly. The default TestClient re-raises an
    exception that reaches Starlette's ServerErrorMiddleware (the path that logs
    "Exception in ASGI application"), so a 503 here also proves it never got there.
    """

    @staticmethod
    def _client(monkeypatch):
        import api.database as api_database
        from api import create_app
        from api.auth import (
            CurrentUser, get_optional_user, require_auth, require_authenticated, require_edition,
        )
        real_apply = api_database.apply_pragmas

        def short_busy_timeout(conn, *args, **kwargs):
            real_apply(conn, *args, **kwargs)
            conn.execute("PRAGMA busy_timeout = 100")

        monkeypatch.setattr(api_database, "apply_pragmas", short_busy_timeout)
        app = create_app()
        user = CurrentUser(user_id="test", role="admin", edition_authenticated=True)
        for dep in (require_edition, require_authenticated, require_auth, get_optional_user):
            app.dependency_overrides[dep] = lambda u=user: u
        return TestClient(app)

    @staticmethod
    def _write_locked():
        import contextlib
        import os
        import sqlite3

        @contextlib.contextmanager
        def held():
            conn = sqlite3.connect(os.environ["DB_PATH"], isolation_level=None)
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            finally:
                conn.execute("ROLLBACK")
                conn.close()
        return held()

    @staticmethod
    def _assert_busy(resp):
        assert resp.status_code == 503, resp.text
        assert resp.json() == {"detail": "library_busy"}
        assert resp.headers["Retry-After"] == "5"

    def test_an_undecorated_write_route_answers_503_with_one_warning(self, monkeypatch, caplog):
        """POST /api/albums has no retry decorator and no try: the raw lock error."""
        client = self._client(monkeypatch)
        client.get("/health")  # warm the schema caches before the lock is taken
        caplog.clear()
        with self._write_locked(), caplog.at_level(logging.INFO, logger="api"):
            resp = client.post("/api/albums", json={"name": "busy-probe"})
        self._assert_busy(resp)
        busy = [r for r in caplog.records if "library_busy" in r.getMessage()]
        assert len(busy) == 1 and busy[0].levelno == logging.WARNING
        assert busy[0].exc_info is None, "a refused write is not a fault: no traceback"
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_a_route_that_wraps_the_lock_error_in_a_500_answers_503(self, monkeypatch):
        """POST /api/persons catches sqlite3.Error and raises HTTPException(500)."""
        client = self._client(monkeypatch)
        with self._write_locked():
            resp = client.post("/api/persons", json={"name": "busy-probe"})
        self._assert_busy(resp)

    def test_a_retried_write_route_answers_503(self, monkeypatch, seeded_photos):
        client = self._client(monkeypatch)
        with self._write_locked():
            resp = client.post("/api/photo/set_rating",
                               json={"photo_path": seeded_photos[0]["path"], "rating": 4})
        self._assert_busy(resp)

    def test_a_non_lock_operational_error_is_still_a_500(self, caplog):
        import sqlite3
        from api import sqlite_operational_error_handler
        app = _make_app()
        app.add_exception_handler(sqlite3.OperationalError, sqlite_operational_error_handler)

        @app.get("/schema-fault")
        def schema_fault():
            raise sqlite3.OperationalError("no such column: sharpness")

        with caplog.at_level(logging.ERROR, logger="api"):
            resp = TestClient(app).get("/schema-fault")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
        assert [r for r in caplog.records if r.levelno == logging.ERROR]
