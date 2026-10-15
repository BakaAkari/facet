"""Async route handlers never run blocking SQLite work on the event loop.

A blocking call inside an ``async def`` handler stalls the one event loop, and with it
every other request in flight: on a 130k-photo library ``/api/ranker/status`` (a full
``COUNT(*)`` over ``photos`` on a sync connection) held ``/health`` for 0.3 to 0.45 s
per call, and three cold calls finished at the same instant after 13 s. Sync work
belongs in a plain ``def`` handler (FastAPI runs it in the threadpool) or behind
``asyncio.to_thread``.
"""

import ast
import asyncio
import pathlib
import sqlite3
import time
from contextlib import contextmanager
from unittest import mock

import httpx
import pytest

from api import create_app
from api.auth import CurrentUser, get_optional_user

ROUTERS = pathlib.Path(__file__).resolve().parent.parent / "api" / "routers"
# Calls that open a sync SQLite connection.
SYNC_DB_OPENERS = {"get_db", "get_db_connection", "connect"}


def _called_name(call):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else None


def _off_loop_ids(handler):
    """Nodes that do not run on the event loop: awaited expressions, ``to_thread`` /
    ``run_in_threadpool`` arguments and the bodies of nested functions."""
    ids = set()
    for node in ast.walk(handler):
        if isinstance(node, ast.Await):
            ids.update(id(sub) for sub in ast.walk(node.value))
        elif isinstance(node, ast.Call) and _called_name(node) in (
                "to_thread", "run_in_threadpool", "run_in_executor"):
            ids.update(id(sub) for sub in ast.walk(node))
        elif node is not handler and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            ids.update(id(sub) for sub in ast.walk(node))
    return ids


def _blocking_calls_in_async_handlers():
    found = []
    for path in sorted(ROUTERS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.AsyncFunctionDef):
                continue
            off_loop = _off_loop_ids(handler)
            for node in ast.walk(handler):
                if (isinstance(node, ast.Call) and id(node) not in off_loop
                        and _called_name(node) in SYNC_DB_OPENERS):
                    found.append(f"{path.name}:{node.lineno} {handler.name} -> "
                                 f"{_called_name(node)}()")
    return found


def test_no_async_handler_opens_a_sync_connection_on_the_event_loop():
    assert _blocking_calls_in_async_handlers() == []


def test_the_guard_sees_a_blocking_call():
    """Control: the scan is not vacuous (it parses handlers and flags a direct call)."""
    tree = ast.parse(
        "async def h():\n"
        "    with get_db() as conn:\n"
        "        conn.execute('SELECT 1')\n"
        "    def off():\n"
        "        return get_db()\n"
        "    await asyncio.to_thread(get_db)\n")
    handler = tree.body[0]
    off_loop = _off_loop_ids(handler)
    flagged = [n.lineno for n in ast.walk(handler)
               if isinstance(n, ast.Call) and id(n) not in off_loop
               and _called_name(n) in SYNC_DB_OPENERS]
    assert flagged == [2]
    assert len(list(ROUTERS.glob("*.py"))) > 20


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
