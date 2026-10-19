"""Deterministic WAL readers vs an external score publisher; synthetic rows only."""

import contextlib
import json
import sqlite3
from threading import Event, Thread
from unittest import mock

import pytest

from api import config as api_config
from api.routers import filter_options
from db.schema import init_database
from db.stats_cache import METRIC_RANGES_KEY


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

    api_config.invalidate_stats_cache()
    yield path, factory
    api_config.invalidate_stats_cache()


def publish(path):
    with sqlite3.connect(path, timeout=2) as conn:
        conn.execute(
            "UPDATE photos SET aggregate=CASE path WHEN 'a' THEN 0 WHEN 'b' THEN 1 ELSE 10 END"
        )
        conn.execute("DELETE FROM stats_cache WHERE key=?", (METRIC_RANGES_KEY,))


def cached(path):
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT value FROM stats_cache WHERE key=?", (METRIC_RANGES_KEY,)
        ).fetchone()
        assert conn.execute(
            "SELECT value FROM stats_cache WHERE key='percentiles_snapshot'"
        ).fetchone() == ("keep-me",)
        return json.loads(row[0]) if row else None


def test_old_reader_cannot_repopulate_ranges_after_external_eviction(database):
    path, factory = database
    computed = Event()
    committed = Event()
    out = {}
    original = filter_options.cache_stats_best_effort

    def pause(conn, entries):
        computed.set()
        assert committed.wait(5), "external commit did not complete"
        return original(conn, entries)

    def reader():
        try:
            out["value"] = filter_options._compute_metric_ranges()
        except BaseException as error:
            out["error"] = error

    with (
        mock.patch.object(filter_options, "get_db", factory),
        mock.patch.object(filter_options, "cache_stats_best_effort", pause),
    ):
        thread = Thread(target=reader)
        thread.start()
        try:
            assert computed.wait(5), "reader never reached writeback"
            publish(path)
        finally:
            committed.set()
        thread.join(5)
        assert not thread.is_alive()
        assert "error" not in out, out.get("error")
    # A read begun before publication may return its old coherent snapshot,
    # but it must never persist that snapshot after the publisher deleted it.
    row = cached(path)
    assert row is None or row["min_score"]["min"] == 0
    with mock.patch.object(filter_options, "get_db", factory):
        assert filter_options._compute_metric_ranges()["min_score"]["min"] == 0


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
    read_bounds = Event()
    committed = Event()
    out = {}
    with factory() as conn:
        old = filter_options.compute_metric_ranges(conn)
    original = filter_options._metric_bounds

    def pause(conn, column):
        result = original(conn, column)
        if column == "aggregate":
            read_bounds.set()
            assert committed.wait(5)
        return result

    def reader():
        try:
            out["value"] = filter_options._compute_metric_ranges()
        except BaseException as error:
            out["error"] = error

    with (
        mock.patch.object(filter_options, "get_db", factory),
        mock.patch.object(filter_options, "_metric_bounds", pause),
    ):
        thread = Thread(target=reader)
        thread.start()
        try:
            assert read_bounds.wait(5)
            publish(path)
        finally:
            committed.set()
        thread.join(5)
        assert not thread.is_alive()
        assert "error" not in out, out.get("error")
    assert out["value"] == old
