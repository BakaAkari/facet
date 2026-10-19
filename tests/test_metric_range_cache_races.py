"""Deterministic WAL readers vs an external score publisher; synthetic rows only."""

import contextlib
import json
import sqlite3
import time
from threading import Event, Lock, Thread
from unittest import mock

import pytest

from api.routers import filter_options
from db.schema import init_database
from db.stats_cache import METRIC_RANGES_KEY, invalidate_metric_ranges


@pytest.fixture
def database(tmp_path):
    path = str(tmp_path / "ranges.db")
    init_database(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executemany(
            "INSERT INTO photos(path,filename,aggregate) VALUES(?,?,?)",
            [("a", "a", 2.0), ("b", "b", 5.0), ("c", "c", 8.0)],
        )
        conn.execute(
            "INSERT OR REPLACE INTO stats_cache VALUES('percentiles_snapshot','keep-me',123)"
        )

    @contextlib.contextmanager
    def factory():
        conn = sqlite3.connect(path, timeout=2)
        try:
            yield conn
        finally:
            conn.close()

    return path, factory


def publish(path):
    """An external score update that invalidates the ranges in the same transaction."""
    with sqlite3.connect(path, timeout=2) as conn:
        conn.execute(
            "UPDATE photos SET aggregate=CASE path WHEN 'a' THEN 0 WHEN 'b' THEN 1 ELSE 10 END"
        )
        invalidate_metric_ranges(conn)


def unrelated_commit(path):
    """A rating save / scan batch: touches neither the scores nor the ranges row."""
    with sqlite3.connect(path, timeout=2) as conn:
        conn.execute("UPDATE photos SET filename = filename WHERE path = 'a'")
        conn.execute("INSERT OR REPLACE INTO stats_cache VALUES('unrelated', '1', 1)")


def stored(path):
    """The persisted ranges row as ``(exists, parsed value or None)``."""
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT value FROM stats_cache WHERE key=?", (METRIC_RANGES_KEY,)
        ).fetchone()
        assert conn.execute(
            "SELECT value FROM stats_cache WHERE key='percentiles_snapshot'"
        ).fetchone() == ("keep-me",)
    if row is None:
        return False, None
    return True, json.loads(row[0]) if row[0] is not None else None


def compute_across(factory, name, commit, pause_if=lambda *args: True):
    """Run one ``_compute_metric_ranges`` in a thread, and run ``commit()`` while
    it is paused right after ``filter_options.<name>`` returns (when
    ``pause_if(*args)``). Returns what the read returned."""
    reached = Event()
    committed = Event()
    out = {}
    original = getattr(filter_options, name)

    def paused(*args):
        result = original(*args)
        if pause_if(*args):
            reached.set()
            assert committed.wait(5), "the concurrent commit did not complete"
        return result

    def reader():
        try:
            out["value"] = filter_options._compute_metric_ranges()
        except BaseException as error:
            out["error"] = error

    with (
        mock.patch.object(filter_options, "get_db", factory),
        mock.patch.object(filter_options, name, paused),
    ):
        thread = Thread(target=reader)
        thread.start()
        try:
            assert reached.wait(5), f"reader never reached {name}"
            commit()
        finally:
            committed.set()
        thread.join(5)
        assert not thread.is_alive()
    assert "error" not in out, out.get("error")
    return out["value"]


def test_old_reader_cannot_repopulate_ranges_after_external_eviction(database):
    path, factory = database
    # The read begun before publication returns its old coherent snapshot...
    old = compute_across(factory, "compute_metric_ranges", lambda: publish(path))
    assert old["min_score"]["min"] == 2
    # ...but never persists it over the publisher's tombstone.
    assert stored(path) == (True, None)
    with mock.patch.object(filter_options, "get_db", factory):
        assert filter_options._compute_metric_ranges()["min_score"]["min"] == 0
    assert stored(path)[1]["min_score"]["min"] == 0


def test_unrelated_commit_during_compute_still_persists_ranges(database):
    path, factory = database
    ranges = compute_across(factory, "compute_metric_ranges", lambda: unrelated_commit(path))
    assert stored(path) == (True, ranges)


def test_external_eviction_does_not_leave_process_ttl_serving_old_ranges(database):
    path, factory = database
    with mock.patch.object(filter_options, "get_db", factory):
        assert (
            filter_options.metric_ranges(user=None)["ranges"]["min_score"]["min"] == 2
        )
        publish(path)
        assert (
            filter_options.metric_ranges(user=None)["ranges"]["min_score"]["min"] == 0
        )


def test_one_read_never_mixes_precommit_bounds_with_postcommit_buckets(database):
    path, factory = database
    with factory() as conn:
        old = filter_options.compute_metric_ranges(conn)
    value = compute_across(
        factory, "_metric_bounds", lambda: publish(path),
        pause_if=lambda conn, column: column == "aggregate",
    )
    assert value == old
    assert stored(path) == (True, None)


def test_concurrent_cold_requests_compute_once(database):
    path, factory = database
    entered = Event()
    release = Event()
    calls = []
    calls_lock = Lock()
    original = filter_options.compute_metric_ranges

    def counted(conn):
        with calls_lock:
            calls.append(1)
        entered.set()
        assert release.wait(5)
        return original(conn)

    results = []

    def request():
        results.append(filter_options.metric_ranges(user=None)["ranges"])

    with (
        mock.patch.object(filter_options, "get_db", factory),
        mock.patch.object(filter_options, "compute_metric_ranges", counted),
    ):
        leader = Thread(target=request)
        leader.start()
        assert entered.wait(5)
        followers = [Thread(target=request) for _ in range(3)]
        for thread in followers:
            thread.start()
        # Give the followers time to reach the cache read while the leader is
        # still computing: without single-flight, each would start its own walk.
        time.sleep(0.3)
        release.set()
        for thread in [leader, *followers]:
            thread.join(5)
            assert not thread.is_alive()
    assert len(calls) == 1
    assert len(results) == 4 and all(r == results[0] for r in results)
    assert stored(path)[1] == results[0]


def test_follower_does_not_accept_a_result_whose_write_was_skipped_as_busy(database):
    path, factory = database
    entered = Event()
    release = Event()
    original = filter_options.compute_metric_ranges
    state = {"first": True}

    def paused(conn):
        result = original(conn)
        if state["first"]:
            state["first"] = False
            entered.set()
            assert release.wait(5)
        return result

    out = {}

    def request(name):
        out[name] = filter_options._compute_metric_ranges()

    blocker = sqlite3.connect(path, timeout=2)
    with (
        mock.patch.object(filter_options, "get_db", factory),
        mock.patch.object(filter_options, "compute_metric_ranges", paused),
        mock.patch("db.stats_cache.CACHE_WRITE_BUSY_TIMEOUT_MS", 50),
    ):
        leader = Thread(target=request, args=("leader",))
        leader.start()
        assert entered.wait(5)
        publish(path)
        follower = Thread(target=request, args=("follower",))
        follower.start()
        time.sleep(0.3)
        # Hold the write lock so the leader's compare-and-set is skipped as busy.
        blocker.execute("BEGIN IMMEDIATE")
        release.set()
        time.sleep(0.5)
        blocker.rollback()
        for thread in (leader, follower):
            thread.join(10)
            assert not thread.is_alive()
    blocker.close()
    assert out["follower"]["min_score"]["min"] == 0
    exists, value = stored(path)
    assert exists and (value is None or value["min_score"]["min"] == 0)


def test_stats_readers_treat_the_eviction_tombstone_as_a_miss(database):
    from db.stats_cache import get_cached_stat, get_stats_cache_info
    path, _factory = database
    with sqlite3.connect(path) as conn:
        invalidate_metric_ranges(conn)
    assert get_cached_stat(path, METRIC_RANGES_KEY) == (None, False)
    assert METRIC_RANGES_KEY not in get_cached_stat(path)
    assert METRIC_RANGES_KEY not in get_stats_cache_info(path)
