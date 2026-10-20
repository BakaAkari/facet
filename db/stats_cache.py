"""
Statistics cache functions for Facet.

Precomputed aggregations for viewer performance.
"""

import json
import logging
import sqlite3
import time as time_module

logger = logging.getLogger("facet.stats_cache")

from db.connection import DEFAULT_DB_PATH, get_connection
from db.render_version import count_pending_render
from db.schema import _build_create_table_sql, STATS_CACHE_COLUMNS

PENDING_RENDER_KEY = 'pending_render_count'
# The persisted gallery metric ranges. Defined here so the viewer endpoint, the
# offline refresh and the neutral-IQA repair all name one key. Its evictors write
# a tombstone (see invalidate_metric_ranges) instead of deleting the row.
METRIC_RANGES_KEY = 'metric_ranges'
# The percentile distribution the last --recompute-average / --recompute-category
# normalized every stored aggregate with (PercentileNormalizer.save_to_stats_cache).
# Library data rather than a derived count, so the stats_cache invalidations that
# follow a photo delete keep it. Defined here because db cannot import config.
PERCENTILES_SNAPSHOT_KEY = 'percentiles_snapshot'

# The count only moves when a scan or a --refresh-thumbnails run writes, and the
# banner it drives is advisory. An hour keeps the scan off every SPA startup
# without letting a finished migration keep nagging for long.
PENDING_RENDER_TTL_SECONDS = 3600

# How long a read-path cache write may wait for the write lock (best_effort_write).
CACHE_WRITE_BUSY_TIMEOUT_MS = 500


def refresh_stats_cache(db_path='photo_scores_pro.db', verbose=True):
    """Refresh all cached statistics for performance optimization.

    Args:
        db_path: Path to SQLite database
        verbose: If True, print progress

    Returns:
        Dict of cached statistics
    """
    from api.db_helpers import HIDE_BURSTS_SQL

    with get_connection(db_path) as conn:
        # Ensure stats_cache table exists
        conn.execute(_build_create_table_sql('stats_cache', STATS_CACHE_COLUMNS))

        now = time_module.time()
        stats = {}

        if verbose:
            logger.info("Refreshing statistics cache...")

        # 1. Total photo count
        total = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        stats['total_photos'] = total
        _cache_stat(conn, 'total_photos', total, now)
        if verbose:
            logger.info("  Total photos: %d", total)

        # 2. Photo count by blink/burst status (for filtered counts)
        try:
            hide_blinks_count = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE is_blink = 0 OR is_blink IS NULL"
            ).fetchone()[0]
            stats['count_hide_blinks'] = hide_blinks_count
            _cache_stat(conn, 'count_hide_blinks', hide_blinks_count, now)

            hide_bursts_count = conn.execute(
                f"SELECT COUNT(*) FROM photos WHERE {HIDE_BURSTS_SQL}"
            ).fetchone()[0]
            stats['count_hide_bursts'] = hide_bursts_count
            _cache_stat(conn, 'count_hide_bursts', hide_bursts_count, now)

            hide_both_count = conn.execute(
                f"""SELECT COUNT(*) FROM photos
                   WHERE (is_blink = 0 OR is_blink IS NULL)
                   AND {HIDE_BURSTS_SQL}"""
            ).fetchone()[0]
            stats['count_hide_both'] = hide_both_count
            _cache_stat(conn, 'count_hide_both', hide_both_count, now)
        except sqlite3.OperationalError:
            pass

        # 3. Camera model counts
        try:
            cameras = conn.execute("""
                SELECT camera_model, COUNT(*) as cnt
                FROM photos
                WHERE camera_model IS NOT NULL
                GROUP BY camera_model
                ORDER BY cnt DESC
            """).fetchall()
            camera_data = [(r[0], r[1]) for r in cameras]
            stats['cameras'] = camera_data
            _cache_stat(conn, 'cameras', json.dumps(camera_data), now)
            if verbose:
                logger.info("  Camera models: %d", len(camera_data))
        except sqlite3.OperationalError:
            pass

        # 4. Lens model counts
        try:
            lenses = conn.execute("""
                SELECT lens_model, COUNT(*) as cnt
                FROM photos
                WHERE lens_model IS NOT NULL
                GROUP BY lens_model
                ORDER BY cnt DESC
            """).fetchall()
            lens_data = [(r[0], r[1]) for r in lenses]
            stats['lenses'] = lens_data
            _cache_stat(conn, 'lenses', json.dumps(lens_data), now)
            if verbose:
                logger.info("  Lens models: %d", len(lens_data))
        except sqlite3.OperationalError:
            pass

        # 5. Person counts (for face recognition dropdown)
        # Mirror the live /api/filter_options/persons query exactly: exclude
        # hidden persons and apply the same min_photos / max_persons bounds.
        # Without this the cached list resurrects hidden persons in the filter
        # for the cache TTL after --refresh-stats.
        try:
            from api.config import VIEWER_CONFIG
            from db.schema import person_not_hidden_clause
            min_photos = VIEWER_CONFIG['dropdowns'].get('min_photos_for_person', 1)
            max_persons = VIEWER_CONFIG['dropdowns']['max_persons']
            persons = conn.execute(f"""
                SELECT p.id, p.name, COUNT(DISTINCT f.photo_path) as photo_count
                FROM persons p
                JOIN faces f ON f.person_id = p.id
                WHERE {person_not_hidden_clause('p')}
                GROUP BY p.id
                HAVING photo_count >= ?
                ORDER BY photo_count DESC
                LIMIT ?
            """, (min_photos, max_persons)).fetchall()
            person_data = [(r[0], r[1], r[2]) for r in persons]
            stats['persons'] = person_data
            _cache_stat(conn, 'persons', json.dumps(person_data), now)
            if verbose:
                logger.info("  Persons: %d", len(person_data))
        except sqlite3.OperationalError:
            pass

        # 6. Category counts
        try:
            categories = conn.execute("""
                SELECT category, COUNT(*) as cnt
                FROM photos
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY cnt DESC
            """).fetchall()
            category_data = [(r[0], r[1]) for r in categories]
            stats['categories'] = category_data
            _cache_stat(conn, 'categories', json.dumps(category_data), now)
            if verbose:
                logger.info("  Categories: %d", len(category_data))
        except sqlite3.OperationalError:
            pass

        # 7. Composition pattern counts
        try:
            patterns = conn.execute("""
                SELECT composition_pattern, COUNT(*) as cnt
                FROM photos
                WHERE composition_pattern IS NOT NULL AND composition_pattern != ''
                GROUP BY composition_pattern
                ORDER BY cnt DESC
            """).fetchall()
            pattern_data = [(r[0], r[1]) for r in patterns]
            stats['composition_patterns'] = pattern_data
            _cache_stat(conn, 'composition_patterns', json.dumps(pattern_data), now)
            if verbose:
                logger.info("  Composition patterns: %d", len(pattern_data))
        except sqlite3.OperationalError:
            pass

        # 8. Tag counts: AI tags (photo_tags) UNION the user's manual tags.
        # UNION on (photo, tag) pairs so a manual tag equal to an AI tag counts
        # that photo once. Either table being non-empty is enough to build it,
        # otherwise a manual-only library would never cache.
        try:
            sources = []
            for table in ('photo_tags', 'photo_manual_tags'):
                try:
                    if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                        sources.append(f"SELECT photo_path, tag FROM {table}")
                except sqlite3.OperationalError:
                    continue
            if sources:
                tags = conn.execute(f"""
                    SELECT tag, COUNT(*) as cnt
                    FROM ({' UNION '.join(sources)})
                    GROUP BY tag
                    ORDER BY cnt DESC
                    LIMIT 100
                """).fetchall()
                tag_data = [(r[0], r[1]) for r in tags]
                stats['tags'] = tag_data
                _cache_stat(conn, 'tags', json.dumps(tag_data), now)
                if verbose:
                    logger.info("  Tags: %d (photo_tags + photo_manual_tags)", len(tag_data))
            else:
                if verbose:
                    logger.info("  Tags: skipped (photo_tags and photo_manual_tags empty - run --migrate-tags)")
        except sqlite3.OperationalError:
            pass

        # 9. Aperture (rounded f_stop) counts
        try:
            apertures = conn.execute("""
                SELECT ROUND(f_stop, 1) as ap, COUNT(*) as cnt
                FROM photos
                WHERE f_stop IS NOT NULL AND f_stop > 0 AND f_stop < 1000
                GROUP BY ap
                ORDER BY ap ASC
            """).fetchall()
            aperture_data = [(r[0], r[1]) for r in apertures]
            stats['apertures'] = aperture_data
            _cache_stat(conn, 'apertures', json.dumps(aperture_data), now)
            if verbose:
                logger.info("  Apertures: %d", len(aperture_data))
        except sqlite3.OperationalError:
            pass

        # 10. Focal length (rounded) counts
        try:
            focals = conn.execute("""
                SELECT CAST(ROUND(focal_length) AS INTEGER) as fl, COUNT(*) as cnt
                FROM photos
                WHERE focal_length IS NOT NULL AND focal_length > 0
                GROUP BY fl
                ORDER BY fl ASC
            """).fetchall()
            focal_data = [(r[0], r[1]) for r in focals]
            stats['focal_lengths'] = focal_data
            _cache_stat(conn, 'focal_lengths', json.dumps(focal_data), now)
            if verbose:
                logger.info("  Focal lengths: %d", len(focal_data))
        except sqlite3.OperationalError:
            pass

        # 11. Metric ranges + sparkline histograms for the gallery filter sidebar.
        # Precomputing offline keeps the first sidebar open off the critical path:
        # the endpoint walks one covering index per metric column, which is cheap
        # but not free on a large library.
        try:
            from api.routers.filter_options import compute_metric_ranges
            metric_ranges = compute_metric_ranges(conn)
            stats[METRIC_RANGES_KEY] = metric_ranges
            _cache_stat(conn, METRIC_RANGES_KEY, json.dumps(metric_ranges), now)
            if verbose:
                logger.info("  Metric ranges: %d", len(metric_ranges))
        except sqlite3.OperationalError:
            pass

        pending_render = refresh_pending_render_stat(conn)
        stats[PENDING_RENDER_KEY] = pending_render
        if verbose:
            logger.info("  RAW photos awaiting a thumbnail refresh: %d", pending_render)

        conn.commit()

    if verbose:
        logger.info("Statistics cache refreshed.")

    return stats


def refresh_pending_render_stat(conn):
    """Recount RAW rows awaiting a thumbnail refresh and re-cache the total."""
    count = count_pending_render(conn)
    _cache_stat(conn, PENDING_RENDER_KEY, count, time_module.time())
    return count


def get_pending_render_count(db_path=None, max_age_seconds=PENDING_RENDER_TTL_SECONDS):
    """Cached count of RAW rows whose thumbnail predates the current render.

    Served to ``/api/config``, which the SPA calls on every startup, so the
    underlying scan must never run per request. A stale entry is recomputed once
    and re-cached; a miss on a read-only database still answers, it just cannot
    persist the result.
    """
    path = db_path or DEFAULT_DB_PATH
    value, is_fresh = get_cached_stat(path, PENDING_RENDER_KEY, max_age_seconds)
    if is_fresh and isinstance(value, int):
        return value
    with get_connection(path) as conn:
        count = count_pending_render(conn)
        try:
            cache_stats_best_effort(conn, [(PENDING_RENDER_KEY, count)])
        except sqlite3.OperationalError:
            logger.debug("Could not persist %s (read-only database?)", PENDING_RENDER_KEY)
    return count


def _cache_stat(conn, key, value, timestamp):
    """Store a value in the stats_cache table."""
    conn.execute(
        "INSERT OR REPLACE INTO stats_cache (key, value, updated_at) VALUES (?, ?, ?)",
        (key, str(value), timestamp)
    )


def best_effort_write(conn, write, what="cache rows"):
    """Run ``write(conn)`` and commit it from a READ path, without queueing behind a writer.

    A GET handler that caches its answer must not turn into a write that waits
    out ``busy_timeout`` (5 s) behind the gallery index build, whose CREATE INDEX
    holds the write lock for up to minutes: that would answer 503 ``library_busy``
    on a read or add seconds to it. So the write is skipped outright while this
    process builds the indexes, and otherwise waits at most
    ``CACHE_WRITE_BUSY_TIMEOUT_MS`` -- long enough to land between a scan's
    continuous commits, short enough never to hold a request. The connection's own
    timeout is restored afterwards. The cache is only an optimisation: the caller
    returns the value it computed either way. Only a busy/locked error is
    swallowed (the transaction is rolled back); any other raises.

    Returns True when the write was committed.
    """
    from api.db_helpers import gallery_index_build_running, is_locked_error
    if gallery_index_build_running():
        return False
    previous = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    conn.execute(f"PRAGMA busy_timeout = {CACHE_WRITE_BUSY_TIMEOUT_MS}")
    try:
        write(conn)
        conn.commit()
        return True
    except sqlite3.OperationalError as ex:
        if not is_locked_error(ex):
            raise
        conn.rollback()
        logger.debug("Skipped writing %s: the database is busy", what)
        return False
    finally:
        conn.execute(f"PRAGMA busy_timeout = {int(previous)}")


def invalidate_metric_ranges(conn):
    """Mark the persisted metric ranges stale, inside the caller's transaction.

    The row is overwritten with a tombstone -- ``value`` NULL and an
    ``updated_at`` strictly later than the row it replaces -- rather than
    deleted. The viewer computes the ranges outside any write lock and persists
    them with a compare-and-set against the row it read first
    (``cache_stat_if_unchanged``); a DELETE would turn "absent, then evicted"
    back into "absent", so a compute begun before the eviction would repopulate
    the pre-eviction ranges. Every reader treats a NULL value as a miss.
    """
    conn.execute(
        "INSERT OR REPLACE INTO stats_cache (key, value, updated_at) VALUES (?, NULL, "
        "MAX(?, COALESCE((SELECT updated_at FROM stats_cache WHERE key = ?), 0) + 0.001))",
        (METRIC_RANGES_KEY, time_module.time(), METRIC_RANGES_KEY))


def cache_stat_if_unchanged(conn, key, value, seen):
    """Best-effort write of one ``stats_cache`` row, only if it is still ``seen``.

    ``seen`` is the ``(value, updated_at)`` pair read before the value was
    computed, or None when the row was absent. Any writer that touched the row
    since -- an eviction tombstone, an offline refresh, another viewer process --
    wins, and this write is dropped; commits to OTHER rows or tables do not
    matter, unlike a read-to-write upgrade of the snapshot the value was computed
    in. Returns True when the row was written, False when it had changed since
    ``seen``, and None when the write was skipped because the database was busy.
    """
    landed = []

    def write(c):
        now = time_module.time()
        if seen is None:
            cur = c.execute(
                "INSERT OR IGNORE INTO stats_cache (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now))
        else:
            cur = c.execute(
                "UPDATE stats_cache SET value = ?, updated_at = ? "
                "WHERE key = ? AND value IS ? AND updated_at IS ?",
                (value, now, key, seen[0], seen[1]))
        landed.append(cur.rowcount == 1)

    return landed[0] if best_effort_write(conn, write, key) else None


def cache_stats_best_effort(conn, entries):
    """``best_effort_write`` of ``(key, value)`` rows into ``stats_cache``."""
    def write(c):
        now = time_module.time()
        for key, value in entries:
            _cache_stat(c, key, value, now)
    return best_effort_write(conn, write, ", ".join(key for key, _ in entries))


def get_cached_stat(db_path='photo_scores_pro.db', key=None, max_age_seconds=300):
    """Get cached statistics from the database.

    Args:
        db_path: Path to SQLite database
        key: Specific key to fetch (None = all)
        max_age_seconds: Maximum age of cached data before considered stale

    Returns:
        If key specified: (value, is_fresh) tuple
        If key is None: dict of all cached stats with freshness info
    """
    with get_connection(db_path) as conn:
        now = time_module.time()

        try:
            if key:
                row = conn.execute(
                    "SELECT value, updated_at FROM stats_cache WHERE key = ?",
                    (key,)
                ).fetchone()

                if row is None or row['value'] is None:  # absent, or an eviction tombstone
                    return None, False

                value = row['value']
                updated_at = row['updated_at']
                is_fresh = (now - updated_at) < max_age_seconds

                # Try to parse JSON values
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if not JSON
                    pass

                return value, is_fresh

            else:
                rows = conn.execute(
                    "SELECT key, value, updated_at FROM stats_cache WHERE value IS NOT NULL"
                ).fetchall()

                result = {}
                for row in rows:
                    key_name = row['key']
                    value = row['value']
                    updated_at = row['updated_at']
                    is_fresh = (now - updated_at) < max_age_seconds

                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    result[key_name] = {'value': value, 'fresh': is_fresh, 'age': now - updated_at}

                return result

        except sqlite3.OperationalError:
            return (None, False) if key else {}


def get_stats_cache_info(db_path='photo_scores_pro.db'):
    """Get information about the stats cache.

    Returns:
        Dict with cache info: {key: {age_seconds, fresh}}
    """
    with get_connection(db_path) as conn:
        now = time_module.time()

        try:
            rows = conn.execute(
                "SELECT key, updated_at FROM stats_cache WHERE value IS NOT NULL ORDER BY key"
            ).fetchall()
            info = {}
            for row in rows:
                age = now - row['updated_at']
                info[row['key']] = {
                    'age_seconds': int(age),
                    'age_human': _format_age(age),
                    'fresh': age < 300  # 5 minute threshold
                }
            return info
        except sqlite3.OperationalError:
            return {}


def _format_age(seconds):
    """Format age in seconds to human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"
