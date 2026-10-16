"""Plan test for ``/api/stats/categories``: the averages statement must be answered
from ``idx_stats_categories`` alone.

Without the index the GROUP BY reads every table page (59 s and 6.8 GB on a 127k
library). The statement is the module constant ``CATEGORY_AVERAGES_SQL`` itself, never
a hand copy, so a rewrite of the route's SQL changes what this test plans.

The pin guards the silent regression: a column added to the SELECT but not to the
index still returns correct rows, it just stops being a covering scan. Both column
lists are derived (``PRAGMA index_info`` and a token scan checked against
``PRAGMA table_info(photos)``), never typed out.
"""

import re
import sqlite3

import pytest

from api.routers import stats
from db.schema import init_database

INDEX = "idx_stats_categories"


@pytest.fixture()
def conn(tmp_path):
    db_path = tmp_path / "plan.db"
    init_database(str(db_path))
    c = sqlite3.connect(str(db_path))
    yield c
    c.close()


def _plan(c, sql):
    return " | ".join(r[3] for r in c.execute("EXPLAIN QUERY PLAN " + sql).fetchall())


def _index_columns(c):
    # Expression columns report name None; this index has none, and a None would
    # simply never match a photos column.
    return {r[2].lower() for r in c.execute(f"PRAGMA index_info({INDEX})") if r[2]}


def _statement_columns(c, sql):
    """The ``photos`` columns a statement names: identifier tokens that are columns."""
    photos_cols = {r[1].lower() for r in c.execute("PRAGMA table_info(photos)")}
    tokens = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)}
    return tokens & photos_cols


def _statement(sql=None):
    return (sql or stats.CATEGORY_AVERAGES_SQL).format(vis="")


def test_index_is_built_by_init_database(conn):
    assert _index_columns(conn), f"{INDEX} missing from the schema"


def test_statement_uses_covering_index(conn):
    plan = _plan(conn, _statement())
    assert f"USING COVERING INDEX {INDEX}" in plan, plan


def test_control_without_index_scans_the_table(conn):
    conn.execute(f"DROP INDEX {INDEX}")
    plan = _plan(conn, _statement())
    assert INDEX not in plan, plan
    assert "SCAN photos" in plan, plan


def test_statement_columns_are_all_in_the_index(conn):
    named = _statement_columns(conn, stats.CATEGORY_AVERAGES_SQL)
    assert named, "token scan found no photos columns: the pin would pass vacuously"
    missing = named - _index_columns(conn)
    assert not missing, f"columns read by the statement but absent from {INDEX}: {sorted(missing)}"


def test_pin_detects_a_column_added_without_the_index(conn):
    """Fault, in-process: the same pin logic rejects a non-indexed column in the SELECT."""
    indexed = _index_columns(conn)
    photos_cols = [r[1].lower() for r in conn.execute("PRAGMA table_info(photos)")]
    extra = next(c for c in photos_cols if c not in indexed and c != "path")
    faulty = stats.CATEGORY_AVERAGES_SQL.replace(
        "ROUND(AVG(contrast_score), 2)",
        f"ROUND(AVG(contrast_score), 2), MAX({extra})", 1)
    assert faulty != stats.CATEGORY_AVERAGES_SQL, "fault anchor did not match"
    assert extra in _statement_columns(conn, faulty)
    assert _statement_columns(conn, faulty) - indexed == {extra}
    # ... and the plan it would produce is no longer a covering scan, so the plan
    # test above is the second line of defence.
    assert f"USING COVERING INDEX {INDEX}" not in _plan(conn, _statement(faulty))
