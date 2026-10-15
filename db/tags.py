"""
Tag migration functions for Facet.

Populates photo_tags lookup table from tags column.
"""

import logging
import sqlite3

from db.connection import get_connection

logger = logging.getLogger("facet.db_tags")
from db.schema import (
    _build_create_table_sql, PHOTO_TAGS_COLUMNS, PHOTO_TAGS_INDEXES,
)


# Paths per DELETE/SELECT probe: under SQLite's 999-variable cap on older builds.
_SYNC_CHUNK = 900


def iter_tag_pairs(path, tags_str):
    """Yield the ``(path, tag)`` lookup rows a ``photos.tags`` value stands for.

    The one parser for the photo_tags table: split on ``,``, strip, drop
    empties. Duplicates are left to the table's primary key
    (``INSERT OR IGNORE``).
    """
    if not tags_str:
        return
    for tag in tags_str.split(','):
        tag = tag.strip()
        if tag:
            yield path, tag


def _photo_tags_populated(conn):
    """True only when photo_tags exists AND holds a row (what readers key on)."""
    try:
        return conn.execute("SELECT 1 FROM photo_tags LIMIT 1").fetchone() is not None
    except sqlite3.OperationalError:
        return False


def sync_photo_tags(conn, paths):
    """Re-derive the photo_tags rows of ``paths`` from their current ``photos.tags``.

    A no-op unless the table exists and already holds at least one row:
    readers prefer the lookup as soon as it is non-empty
    (``api.db_helpers.is_photo_tags_available``), so filling an absent or empty
    table here would hand them a partial index and hide every photo never
    synced. Bootstrapping is ``database.py --migrate-tags``'s job.

    Rows are read back from ``photos`` rather than taken from the caller, so a
    rescan that writes NULL tags simply leaves the path with no rows. Runs on
    the caller's connection and inside its transaction; the caller commits.

    Returns:
        Number of lookup rows inserted (0 when the table is not in use).
    """
    paths = list(dict.fromkeys(paths))
    if not paths or not _photo_tags_populated(conn):
        return 0
    inserted = 0
    for start in range(0, len(paths), _SYNC_CHUNK):
        chunk = paths[start:start + _SYNC_CHUNK]
        placeholders = ','.join('?' * len(chunk))
        conn.execute(f"DELETE FROM photo_tags WHERE photo_path IN ({placeholders})", chunk)
        rows = conn.execute(
            f"SELECT path, tags FROM photos WHERE path IN ({placeholders}) "
            "AND tags IS NOT NULL AND tags != ''",
            chunk,
        ).fetchall()
        pairs = [pair for path, tags in rows for pair in iter_tag_pairs(path, tags)]
        conn.executemany(
            "INSERT OR IGNORE INTO photo_tags (photo_path, tag) VALUES (?, ?)", pairs)
        inserted += len(pairs)
    return inserted


def write_photo_tags(conn, rows):
    """Set ``photos.tags`` for ``rows`` and keep the photo_tags lookup in step.

    The only place allowed to issue a raw tag UPDATE (a test pins it), so no
    re-tag path can forget the lookup again.

    Args:
        conn: Open SQLite connection; the caller commits.
        rows: Iterable of ``(path, tags_str_or_None)``.
    """
    rows = list(rows)
    if not rows:
        return
    conn.executemany("UPDATE photos SET tags = ? WHERE path = ?",
                     [(tags, path) for path, tags in rows])
    sync_photo_tags(conn, [path for path, _ in rows])


def migrate_tags_to_lookup(db_path='photo_scores_pro.db', batch_size=10000):
    """
    (Re)build the photo_tags lookup table from the current tags column.

    This enables fast exact-match tag queries instead of slow LIKE '%tag%' scans.
    It is a full resync: the table is cleared first and repopulated from the live
    photos.tags values, so re-running it after tags change (rescan, XMP import,
    re-tag) drops tags no longer present and adds new ones. Appending only, as an
    earlier version did, left readers preferring a stale table with removed tags
    still in it.

    No DB backup is taken: photo_tags is a derived index fully reconstructible
    from photos.tags, and this runs under the library lock (database.py holds it
    for --migrate-tags). The previous naive ``shutil.copy2`` of the whole DB was
    WAL-unsafe (dropped the -wal/-shm sidecars) and copied the entire multi-GB
    file on every run with no disk check or rotation.

    Args:
        db_path: Path to the SQLite database file
        batch_size: Number of photos to process per batch

    Returns:
        Tuple of (total_tags_inserted, total_photos_processed)
    """
    with get_connection(db_path, row_factory=False) as conn:
        # Ensure table exists
        conn.execute(_build_create_table_sql(
            'photo_tags',
            PHOTO_TAGS_COLUMNS,
            constraints=['PRIMARY KEY (photo_path, tag)']
        ))

        # Create indexes
        for idx_name, table, column_expr in PHOTO_TAGS_INDEXES:
            conn.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column_expr})')

        # Full resync: clear the derived table so removed tags don't linger.
        conn.execute("DELETE FROM photo_tags")
        conn.commit()

        # Get total count
        total = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE tags IS NOT NULL AND tags != ''"
        ).fetchone()[0]
        logger.info("Processing %d photos with tags...", total)

        total_tags = 0
        processed = 0

        # Process in batches to avoid memory issues
        cursor = conn.execute(
            "SELECT path, tags FROM photos WHERE tags IS NOT NULL AND tags != ''"
        )

        batch = []
        for path, tags in cursor:
            batch.extend(iter_tag_pairs(path, tags))
            processed += 1

            # Insert batch
            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT OR IGNORE INTO photo_tags (photo_path, tag) VALUES (?, ?)",
                    batch
                )
                conn.commit()
                total_tags += len(batch)
                batch = []
                logger.info("  Processed %d/%d photos (%d tags)...", processed, total, total_tags)

        # Final batch
        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO photo_tags (photo_path, tag) VALUES (?, ?)",
                batch
            )
            conn.commit()
            total_tags += len(batch)

    logger.info("Migration complete: %d tags from %d photos", total_tags, processed)
    return total_tags, processed


def get_photo_tags_count(db_path='photo_scores_pro.db'):
    """Return the number of entries in the photo_tags lookup table."""
    with get_connection(db_path, row_factory=False) as conn:
        try:
            count = conn.execute("SELECT COUNT(*) FROM photo_tags").fetchone()[0]
        except sqlite3.OperationalError:
            count = 0
        return count
