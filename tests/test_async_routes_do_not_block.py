"""Blocking work never runs on the event-loop thread.

A blocking call inside an ``async def`` handler stalls the one event loop, and with it
every other request in flight: on a 130k-photo library ``/api/ranker/status`` (a full
``COUNT(*)`` over ``photos`` on a sync connection) held ``/health`` for 0.3 to 0.45 s
per call, and three cold calls finished at the same instant after 13 s. Sync work
belongs in a plain ``def`` handler (FastAPI runs it in the threadpool) or behind
``asyncio.to_thread``.

The guard is a runtime one: the blocking openers below raise when they are called on
a thread that runs an event loop, and the route-coverage request set of
``tests/test_route_coverage.py`` is driven through the app with every process-wide
cache that hides an opener (column list, photo_tags availability, change-token
sentinels, stats cache) emptied before each request. A source scan cannot see a
connection opened three calls down (``get_cached_stat``, ``is_photo_tags_available``),
this can.
"""

import asyncio
import os
import sqlite3
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from api import create_app
from api.auth import (
    CurrentUser, get_optional_user, require_authenticated, require_edition,
    require_superadmin,
)
from tests.test_route_coverage import (
    AUTH_ROUTES, PARAMETERISED_ROUTES, POST_ROUTES, PUBLIC_ROUTES, SUPERADMIN_ROUTES,
)


class BlockingCallOnEventLoop(AssertionError):
    """A blocking opener ran on the event-loop thread."""


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _guarded_openers():
    """(owner, attribute) pairs whose call blocks: a DB open, an image decode, a child."""
    openers = [(sqlite3, "connect"), (subprocess, "run"), (subprocess, "check_output")]
    from PIL import Image
    openers.append((Image, "open"))
    try:
        import rawpy
    except ImportError:  # pragma: no cover - rawpy is a hard dependency of the viewer
        pass
    else:
        openers.append((rawpy, "imread"))
    return openers


@contextmanager
def loop_guard():
    """Make every guarded opener raise (and record itself) when called on the loop.

    It records as well as raises because a caller may swallow the exception (the
    photo_tags probe falls back to "unavailable" on any error); the record is what
    the tests assert on.
    """
    hits = []
    lock = threading.Lock()
    patches = []
    for owner, name in _guarded_openers():
        real = getattr(owner, name)

        def guarded(*args, _real=real, _label=f"{owner.__name__}.{name}", **kwargs):
            if _on_event_loop():
                frames = [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}"
                          for f in traceback.extract_stack()[:-1]
                          if "site-packages" not in f.filename
                          and "test_async_routes" not in f.filename][-6:]
                with lock:
                    hits.append(f"{_label} <- " + " <- ".join(reversed(frames)))
                raise BlockingCallOnEventLoop(f"{_label} called on the event loop")
            return _real(*args, **kwargs)

        patches.append(mock.patch.object(owner, name, guarded))
    for patch in patches:
        patch.start()
    try:
        yield hits
    finally:
        for patch in reversed(patches):
            patch.stop()


def _cold_caches():
    """Empty every process-wide cache that lets a handler skip its opener."""
    from api import config as api_config
    from api.change_token import close_change_token_sentinels
    from api.db_helpers import invalidate_existing_columns_cache
    invalidate_existing_columns_cache()
    with api_config._photo_tags_lock:
        api_config._photo_tags_available = None
        api_config._photo_tags_checked_at = 0.0
    import api.db_helpers as db_helpers
    db_helpers._photo_tags_available = None
    with api_config._stats_cache_lock:
        api_config._stats_cache.clear()
    with api_config._count_cache_lock:
        api_config._count_cache.clear()
    close_change_token_sentinels()


def _user_app(user):
    app = create_app()
    for dep in (require_edition, require_authenticated, require_superadmin, get_optional_user):
        app.dependency_overrides[dep] = lambda u=user: u
    return app


EDITION = CurrentUser(user_id="test", role="admin", edition_authenticated=True)
SUPERADMIN = CurrentUser(user_id="root", role="superadmin", edition_authenticated=True)


# Async GET routes the route-coverage lists do not drive (they list a route once
# whatever its handler kind; these are the ``async def`` ones, where a blocking
# call stalls the loop). Left out: /api/updates/check (calls the network),
# /api/download (its RAW branch has its own test below), the POSTs that spawn a
# real facet.py subprocess (/api/recalculate, /api/stats/categories/recompute),
# /api/config/update_weights (writes the config file), /api/scan/stream, /dav and
# the Immich webhook.
ASYNC_EXTRA_ROUTES = [
    ('/api/photo/set', {'path': '/nonexistent.jpg'}),
    ('/api/photos/select_bottom_percent', {'keep_percent': 50}),
    ('/api/photos/count', {}),
    ('/api/photos/paths', {}),
    ('/api/filter_options/colors', {}),
    ('/api/filter_options/junk_kinds', {}),
    ('/api/person/1/faces', {}),
    ('/api/persons', {}),
    ('/api/persons/needs_naming', {}),
    ('/api/stats/categories/breakdown', {}),
    ('/api/stats/categories/correlations', {}),
    ('/api/stats/categories/overlap', {}),
    ('/api/albums/1/photos', {}),
    ('/api/shared/album/1', {'token': 'x'}),
    ('/api/memories/check', {}),
    ('/api/capsules/x/photos', {}),
    ('/api/similar_photos/nonexistent.jpg', {}),
]


def _requests():
    yield from (("GET", path, {}, None, None) for path in PUBLIC_ROUTES)
    yield from (("GET", path, {}, None, EDITION) for path in AUTH_ROUTES)
    yield from (("GET", path, params, None, EDITION) for path, params in PARAMETERISED_ROUTES)
    yield from (("GET", path, params, None, EDITION) for path, params in ASYNC_EXTRA_ROUTES)
    yield from (("GET", path, {}, None, SUPERADMIN) for path in SUPERADMIN_ROUTES)
    yield from (("POST", path, {}, body, EDITION) for path, body in POST_ROUTES)


REQUESTS = list(_requests())


def _drive(requests):
    """Run ``requests`` with cold caches under the guard; return the hits per request."""
    apps = {}
    found = []
    for method, path, params, body, user in requests:
        key = user.user_id if user else None
        if key not in apps:
            apps[key] = TestClient(_user_app(user) if user else create_app(),
                                   raise_server_exceptions=False)
        client = apps[key]
        _cold_caches()
        with loop_guard() as hits:
            client.request(method, path, params=params, json=body)
        found.extend(f"{method} {path}: {hit}" for hit in hits)
    return found


@pytest.mark.usefixtures("seeded_photos")
def test_no_route_runs_a_blocking_opener_on_the_event_loop():
    assert _drive(REQUESTS) == []


def test_a_raw_download_converts_off_the_event_loop(tmp_path, seed_photos_prefix):
    """``/api/download`` of a standalone RAW decodes it with rawpy (seconds of CPU)."""
    raw = tmp_path / "frame.cr2"
    raw.write_bytes(b"not really a raw file")
    seed_photos_prefix(str(tmp_path), [{"path": str(raw), "filename": raw.name}])
    assert _drive([("GET", "/api/download", {"path": str(raw)}, None, EDITION)]) == []


def test_the_guard_flags_a_blocking_opener_on_the_loop():
    """Control: an async handler that opens a connection directly is caught, and the
    same open on a worker thread is not."""
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/on-loop")
    async def on_loop():
        try:
            sqlite3.connect(":memory:").close()
        except BlockingCallOnEventLoop:
            pass
        return {}

    @app.get("/off-loop")
    async def off_loop():
        await asyncio.to_thread(lambda: sqlite3.connect(":memory:").close())
        return {}

    client = TestClient(app)
    with loop_guard() as hits:
        client.get("/off-loop")
    assert hits == []
    with loop_guard() as hits:
        client.get("/on-loop")
    assert len(hits) == 1 and hits[0].startswith("sqlite3.connect")
    assert len(REQUESTS) > 80


@contextmanager
def _slow_db(seconds):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE photos (path TEXT PRIMARY KEY, clip_embedding BLOB);
        CREATE TABLE learned_scores (photo_path TEXT, learned_score REAL,
            comparison_count INTEGER, category TEXT, updated_at TEXT, user_id TEXT);
        CREATE TABLE stats_cache (key TEXT PRIMARY KEY, value TEXT, updated_at REAL);
    """)
    time.sleep(seconds)  # a cold full-table COUNT(*) on a large library
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_a_slow_ranker_status_does_not_hold_other_requests():
    app = create_app()
    app.dependency_overrides[get_optional_user] = lambda: CurrentUser(
        user_id="test", edition_authenticated=True)
    transport = httpx.ASGITransport(app=app)
    try:
        with mock.patch("api.routers.ranker.get_db", lambda: _slow_db(1.5)):
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                started = time.monotonic()
                slow = asyncio.create_task(client.get("/api/ranker/status"))
                await asyncio.sleep(0.05)
                health = await client.get("/health")
                health_seconds = time.monotonic() - started
                status = await slow
    finally:
        app.dependency_overrides.clear()
    assert health.status_code == 200
    assert status.status_code == 200, status.text
    assert health_seconds < 0.75, f"/health waited {health_seconds:.2f}s behind the ranker"
