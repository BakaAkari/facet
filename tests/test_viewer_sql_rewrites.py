"""Viewer API speed: the tags dropdown rewrite and the deterministic tie order.

Two unrelated SQL changes share this file because they share a reason: both
replace a statement whose result an index change could otherwise reorder.

* ``/api/filter_options/tags`` (single-user, no visibility fragment) counts
  ``photo_tags`` with an index-only ``GROUP BY`` and anti-joins the manual tags,
  instead of de-duping every ``(photo, tag)`` pair through a ``UNION`` temp b-tree.
  The answer must not change.
* Album pages and the two map queries ordered by ``aggregate`` alone, so which tied
  row came first (and therefore which rows page 2 held) depended on the plan the
  indexes happened to give. They now end in ``path ASC``.

Rows go into the SHARED session database (``seed_photos_prefix``) and the
auth-overridden ``edition_client`` drives the real routes. Tie fixtures insert their
paths in DESCENDING order, so rowid order is the opposite of path order and an
un-tiebroken statement returns the wrong answer rather than the right one by luck.
"""

import json
import sqlite3
import uuid

import pytest

from tests.conftest import _TEST_DB_FILE

TAG_PREFIX = "/vsr-tags/"
ALBUM_PREFIX = "/vsr-album/"
MAP_PREFIX = "/vsr-map/"
OWNER = "test"  # the user_id the edition_client fixture authenticates as

# The statement this change replaced, kept verbatim as the reference for the
# "same answer" test.
OLD_TAGS_SQL = """
    SELECT tag, COUNT(*) as cnt
    FROM (SELECT photo_path, tag FROM photo_tags
          UNION SELECT photo_path, tag FROM photo_manual_tags)
    WHERE 1=1
    GROUP BY tag
    ORDER BY cnt DESC, tag ASC
    LIMIT ?
"""


def _connect():
    return sqlite3.connect(_TEST_DB_FILE.name)


# --- tags dropdown -----------------------------------------------------------------


@pytest.fixture()
def tag_rows(seed_photos_prefix, monkeypatch):
    """Insert ``photo_tags`` / ``photo_manual_tags`` rows; delete them on teardown.

    ``seed_photos_prefix`` deletes only ``photos`` rows, and its raw connection does
    not enforce the ``ON DELETE CASCADE``, so the side tables are cleaned here.

    ``is_photo_tags_available`` means "the table exists AND has rows" and caches the
    answer process-wide for five minutes. Letting it run while these rows exist would
    cache True after they are deleted and send every later tag filter in the session
    to an empty table, so the route sees it answer True without caching anything.
    """
    monkeypatch.setattr("api.routers.filter_options.is_photo_tags_available", lambda *a, **k: True)
    seed_photos_prefix(TAG_PREFIX, [
        {"path": f"{TAG_PREFIX}{n}.jpg", "filename": f"{n}.jpg", "aggregate": 5.0}
        for n in "abcdef"
    ])

    def add(ai=(), manual=()):
        with _connect() as conn:
            conn.executemany(
                "INSERT INTO photo_tags (photo_path, tag) VALUES (?, ?)",
                [(TAG_PREFIX + p, t) for p, t in ai])
            conn.executemany(
                "INSERT INTO photo_manual_tags (photo_path, tag, source) VALUES (?, ?, 'user')",
                [(TAG_PREFIX + p, t) for p, t in manual])

    yield add
    with _connect() as conn:
        conn.execute("DELETE FROM photo_tags WHERE photo_path LIKE ?", (TAG_PREFIX + "%",))
        conn.execute("DELETE FROM photo_manual_tags WHERE photo_path LIKE ?", (TAG_PREFIX + "%",))


@pytest.fixture()
def no_cached_tags():
    """Drop the ``stats_cache`` 'tags' row so the endpoint cannot answer from it.

    ``tags()`` returns the cached list for a single-user caller with no visibility
    fragment; an earlier test in the session may have filled it, and an oracle that
    compares the cache with itself proves nothing.
    """
    def clear():
        with _connect() as conn:
            conn.execute("DELETE FROM stats_cache WHERE key = 'tags'")
    clear()
    yield clear
    clear()


@pytest.fixture()
def wide_tag_limit(monkeypatch):
    from api.config import VIEWER_CONFIG
    monkeypatch.setitem(VIEWER_CONFIG["dropdowns"], "max_tags", 100000)


def _python_oracle():
    """(tag, distinct photo count) from the raw rows, ordered count DESC then tag ASC."""
    pairs = set()
    with _connect() as conn:
        pairs.update(conn.execute("SELECT photo_path, tag FROM photo_tags"))
        pairs.update(conn.execute("SELECT photo_path, tag FROM photo_manual_tags"))
    counts = {}
    for _path, tag in pairs:
        counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _endpoint_tags(client):
    resp = client.get("/api/filter_options/tags")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cached"] is False
    return [tuple(t) for t in body["tags"]]


def _seed_mixed_tags(tag_rows):
    u = uuid.uuid4().hex[:8]
    tags = {k: f"{u}-{k}" for k in ("ai_only", "manual_only", "both", "pop", "tie_a", "tie_b")}
    tag_rows(
        ai=[("a.jpg", tags["ai_only"]),
            ("a.jpg", tags["both"]), ("b.jpg", tags["both"]),
            ("a.jpg", tags["pop"]), ("b.jpg", tags["pop"]), ("c.jpg", tags["pop"]),
            ("d.jpg", tags["tie_b"]), ("e.jpg", tags["tie_b"]),
            ("d.jpg", tags["tie_a"]), ("e.jpg", tags["tie_a"])],
        manual=[("f.jpg", tags["manual_only"]),
                # the same (photo, tag) as an AI row: must count that photo once
                ("a.jpg", tags["both"]),
                # a manual tag on a photo that does not carry the AI tag: counts
                ("d.jpg", tags["pop"])],
    )
    return tags


def test_tags_match_independent_python_count(edition_client, tag_rows, no_cached_tags, wide_tag_limit):
    tags = _seed_mixed_tags(tag_rows)
    got = _endpoint_tags(edition_client)
    assert got == _python_oracle()
    by_tag = dict(got)
    assert by_tag[tags["both"]] == 2          # manual duplicate of an AI pair counted once
    assert by_tag[tags["pop"]] == 4           # 3 AI photos + 1 manual-only photo
    assert by_tag[tags["manual_only"]] == 1
    assert by_tag[tags["ai_only"]] == 1
    names = [t for t, _ in got]
    assert names.index(tags["tie_a"]) < names.index(tags["tie_b"])  # equal counts: by name


def test_tags_equal_the_old_union_query_as_a_set(edition_client, tag_rows, no_cached_tags, wide_tag_limit):
    _seed_mixed_tags(tag_rows)
    with _connect() as conn:
        old = {tuple(r) for r in conn.execute(OLD_TAGS_SQL, [100000])}
    assert {t for t in _endpoint_tags(edition_client)} == old


def test_tags_max_tags_cut_keeps_the_same_prefix(edition_client, tag_rows, no_cached_tags,
                                                 wide_tag_limit, monkeypatch):
    from api.config import VIEWER_CONFIG
    _seed_mixed_tags(tag_rows)
    full = _endpoint_tags(edition_client)
    assert len(full) > 3
    monkeypatch.setitem(VIEWER_CONFIG["dropdowns"], "max_tags", 3)
    assert _endpoint_tags(edition_client) == full[:3]


def test_tags_without_manual_table_still_counts_ai_pairs(edition_client, tag_rows, no_cached_tags,
                                                         wide_tag_limit, monkeypatch):
    # manual_ready=False must drop the manual half of the statement, not break it.
    tags = _seed_mixed_tags(tag_rows)
    monkeypatch.setattr("api.routers.filter_options.is_manual_tags_available", lambda *a, **k: False)
    by_tag = dict(_endpoint_tags(edition_client))
    assert by_tag[tags["both"]] == 2
    assert by_tag[tags["pop"]] == 3           # the manual-only photo is not counted
    assert tags["manual_only"] not in by_tag


def _capture_tags_sql(client, monkeypatch):
    """Run the real route and return the ``(sql, params)`` of its tag-count statement."""
    from api.routers import filter_options
    seen = []
    real = filter_options._fetch_all

    async def spy(conn, sql, params=None):
        seen.append((sql, list(params or [])))
        return await real(conn, sql, params)

    monkeypatch.setattr(filter_options, "_fetch_all", spy)
    assert client.get("/api/filter_options/tags").status_code == 200
    statements = [s for s in seen if "photo_tags" in s[0]]
    assert len(statements) == 1, seen
    return statements[0]


def _plan(sql, params):
    with _connect() as conn:
        return "\n".join(r[-1] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params))


def test_tags_statement_is_an_index_only_group_by(edition_client, tag_rows, no_cached_tags, monkeypatch):
    _seed_mixed_tags(tag_rows)
    sql, params = _capture_tags_sql(edition_client, monkeypatch)
    plan = _plan(sql, params)
    assert "COVERING INDEX idx_photo_tags_tag" in plan, plan
    assert not [ln for ln in plan.splitlines() if ln.strip() == "SCAN photo_tags"], plan
    # Control: the statement this replaced de-dupes pairs through a temp b-tree.
    old_plan = _plan(OLD_TAGS_SQL, [100000])
    assert "TEMP B-TREE" in old_plan or "COMPOUND QUERY" in old_plan, old_plan
    assert "UNION" not in sql.upper().replace("UNION ALL", "")


# --- deterministic tie order: albums -----------------------------------------------


def _tie_names(count=10):
    # Descending, so rowid (insertion) order is the reverse of path order.
    return [f"t{n:02d}.jpg" for n in range(count - 1, -1, -1)]


def _tie_rows(count=10):
    return [{"path": ALBUM_PREFIX + n, "filename": n, "aggregate": 6.0} for n in _tie_names(count)]


@pytest.fixture()
def make_album():
    ids = []

    def make(*, smart_filter=None, member_names=()):
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO albums (user_id, name, is_smart, smart_filter_json) "
                "VALUES (?, 'vsr ties', ?, ?)",
                (OWNER, 1 if smart_filter is not None else 0,
                 json.dumps(smart_filter) if smart_filter is not None else None))
            album_id = cur.lastrowid
            ids.append(album_id)
            for pos, name in enumerate(member_names):
                conn.execute(
                    "INSERT INTO album_photos (album_id, photo_path, position) VALUES (?, ?, ?)",
                    (album_id, ALBUM_PREFIX + name, pos))
        return album_id

    yield make
    with _connect() as conn:
        for album_id in ids:
            conn.execute("DELETE FROM album_photos WHERE album_id = ?", (album_id,))
            conn.execute("DELETE FROM albums WHERE id = ?", (album_id,))


def _paged_names(client, album_id, direction, per_page=3):
    names = []
    page = 1
    while True:
        resp = client.get(
            f"/api/albums/{album_id}/photos?sort=aggregate&sort_direction={direction}"
            f"&per_page={per_page}&page={page}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        names.extend(p["path"].rsplit("/", 1)[1] for p in body["photos"])
        if not body["has_more"]:
            return names
        page += 1


@pytest.mark.parametrize("direction", ["DESC", "ASC"])
def test_smart_album_pages_through_equal_scores_in_path_order(direction, seed_photos_prefix,
                                                              make_album, edition_client):
    seed_photos_prefix(ALBUM_PREFIX, _tie_rows())
    album_id = make_album(smart_filter={"path_prefix": ALBUM_PREFIX})
    assert _paged_names(edition_client, album_id, direction) == sorted(_tie_names())


@pytest.mark.parametrize("direction", ["DESC", "ASC"])
def test_regular_album_pages_through_equal_scores_in_path_order(direction, seed_photos_prefix,
                                                                make_album, edition_client):
    seed_photos_prefix(ALBUM_PREFIX, _tie_rows())
    album_id = make_album(member_names=_tie_names())
    assert _paged_names(edition_client, album_id, direction) == sorted(_tie_names())


@pytest.mark.parametrize("kind", ["smart", "regular"])
def test_album_page_statement_ends_in_a_path_tiebreak(kind, seed_photos_prefix, make_album,
                                                      edition_client, monkeypatch):
    # The regular album's join scans ``album_photos`` through its (album_id, photo_path)
    # unique index, which hands ties over in path order by accident, so the behavioural
    # tests above cannot see its tiebreak go missing. Read the statement the route ran.
    from contextlib import asynccontextmanager
    from api.routers import albums

    real = albums.get_async_db
    statements = []

    @asynccontextmanager
    async def traced():
        async with real() as conn:
            await conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(albums, "get_async_db", traced)
    seed_photos_prefix(ALBUM_PREFIX, _tie_rows(4))
    if kind == "smart":
        album_id = make_album(smart_filter={"path_prefix": ALBUM_PREFIX})
    else:
        album_id = make_album(member_names=_tie_names(4))
    resp = edition_client.get(f"/api/albums/{album_id}/photos?sort=aggregate&sort_direction=DESC")
    assert resp.status_code == 200, resp.text
    pages = [s for s in statements if "ORDER BY" in s and "LIMIT" in s]
    assert len(pages) == 1, statements
    order_by = pages[0].split("ORDER BY", 1)[1].split("LIMIT", 1)[0].strip()
    assert order_by == "aggregate DESC, photos.path ASC", order_by


def test_album_ties_are_ordered_by_score_first_then_path(seed_photos_prefix, make_album, edition_client):
    # The tiebreak only breaks ties: a higher score still wins whatever its path.
    rows = _tie_rows(4)
    rows[0]["aggregate"] = 9.0  # t03, the LAST path
    seed_photos_prefix(ALBUM_PREFIX, rows)
    album_id = make_album(smart_filter={"path_prefix": ALBUM_PREFIX})
    assert _paged_names(edition_client, album_id, "DESC") == ["t03.jpg", "t00.jpg", "t01.jpg", "t02.jpg"]


# --- deterministic tie order: map --------------------------------------------------

MAP_LAT, MAP_LNG = 71.5, 25.5
MAP_BOUNDS = f"{MAP_LAT - 0.5},{MAP_LNG - 0.5},{MAP_LAT + 0.5},{MAP_LNG + 0.5}"


def _map_rows(count=5):
    # A score (the column caps at 10) no other test seeds, so only these rows can be the representative.
    return [{"path": MAP_PREFIX + n, "filename": n, "aggregate": 9.97,
             "gps_latitude": MAP_LAT, "gps_longitude": MAP_LNG}
            for n in _tie_names(count)]


@pytest.fixture()
def fixed_cluster_threshold(monkeypatch):
    monkeypatch.setattr("api.routers.map._get_cluster_zoom_threshold", lambda: 10)


def test_map_cluster_representative_is_the_lowest_path_among_ties(
        seed_photos_prefix, edition_client, fixed_cluster_threshold):
    seed_photos_prefix(MAP_PREFIX, _map_rows())
    resp = edition_client.get(f"/api/photos/map?bounds={MAP_BOUNDS}&zoom=2")
    assert resp.status_code == 200, resp.text
    clusters = resp.json()["clusters"]
    assert len(clusters) == 1 and clusters[0]["count"] == 5
    assert clusters[0]["representative_path"] == MAP_PREFIX + "t00.jpg"


def test_map_individual_photos_order_equal_scores_by_path(
        seed_photos_prefix, edition_client, fixed_cluster_threshold):
    seed_photos_prefix(MAP_PREFIX, _map_rows())
    resp = edition_client.get(f"/api/photos/map?bounds={MAP_BOUNDS}&zoom=15")
    assert resp.status_code == 200, resp.text
    assert [p["path"] for p in resp.json()["photos"]] == sorted(r["path"] for r in _map_rows())


def test_map_individual_photos_limit_keeps_the_lowest_paths(
        seed_photos_prefix, edition_client, fixed_cluster_threshold):
    # Under a LIMIT the tiebreak decides which rows are returned at all.
    seed_photos_prefix(MAP_PREFIX, _map_rows())
    resp = edition_client.get(f"/api/photos/map?bounds={MAP_BOUNDS}&zoom=15&limit=2")
    assert resp.status_code == 200, resp.text
    assert [p["path"] for p in resp.json()["photos"]] == [MAP_PREFIX + "t00.jpg", MAP_PREFIX + "t01.jpg"]
