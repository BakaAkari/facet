"""Schema step of the viewer's start-up, run under the library lock.

``db.schema.init_database`` builds the gallery covering indexes, which on a large
library is minutes of write-locked work per index. Every CLI job takes
``facet.LibraryLock`` before its own schema step; the viewer is a writer of the same
library, so it takes it too. A scan started meanwhile then fails fast with the clear
"a maintenance is already running" message instead of timing out on
``database is locked``, and with ``--workers N`` exactly one worker holds the lock and
builds while the others defer and serve.
"""

import logging

logger = logging.getLogger(__name__)

BUILT = 'built'
DEFERRED = 'deferred'
FAILED = 'failed'


def init_schema_at_startup(db_path) -> str:
    """Bring the schema up to date; build the photos indexes only if we hold the lock.

    Returns:
        ``'built'``: this process held the library lock and ran the full
        ``init_database``. ``'deferred'``: another job holds the lock, so tables and
        columns were migrated but the photos indexes were not built (the next start,
        or ``python database.py``, builds them). ``'failed'``: an unexpected error,
        logged; the viewer carries on, as it did when the bare call failed.
    """
    # Lazy, like api/routers/scan.py: importing facet pulls in the scoring stack,
    # which the API process otherwise avoids until a route needs it.
    from facet import LIBRARY_JOB_MAINTENANCE, LibraryLock, LibraryLockError
    from db.schema import init_database

    try:
        lock = LibraryLock(db_path, kind=LIBRARY_JOB_MAINTENANCE)
        try:
            lock.acquire()
        except LibraryLockError as ex:
            # str(ex) is library_job_conflict_message(holder, lock_path), or the
            # reason the lock file cannot be used at all.
            logger.warning(
                "Library lock not available, skipping the photos index build at "
                "start-up: %s. Tables and columns are still migrated; the indexes "
                "are built on the next start or by `python database.py`.", ex)
            init_database(db_path, build_photo_indexes=False)
            return DEFERRED
        try:
            init_database(db_path)
        finally:
            lock.release()
        return BUILT
    except Exception:
        logger.warning("Schema migration on startup failed", exc_info=True)
        return FAILED
