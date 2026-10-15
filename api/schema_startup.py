"""Schema step of the viewer's start-up: a synchronous migration, then a background index build.

``db.schema.init_database(build_photo_indexes=False)`` migrates tables, columns, the
photos_fts table and every index group but the photos ``INDEXES``; it is quick, and on
a current schema it takes no write lock, so it runs before the viewer serves.

The gallery covering indexes are minutes of write-locked work per index on a large
library (13 min in all on a 130k-photo library with a cold page cache), so
``PhotoIndexBuild`` builds the missing ones on a daemon thread while the viewer
already answers requests. The thread takes ``facet.LibraryLock`` (origin ``viewer``)
like every CLI job: a scan started meanwhile fails fast with "a maintenance is already
running" instead of timing out on ``database is locked``, and with ``--workers N``
exactly one worker builds while the others serve. Shutdown calls ``stop``, which
interrupts the CREATE INDEX in progress (``sqlite3.Connection.interrupt``) and joins
the thread; each index is its own transaction, so only the one in progress is rolled
back and the next start resumes from there.
"""

import logging
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

BUILT = 'built'
CURRENT = 'current'
DEFERRED = 'deferred'
INTERRUPTED = 'interrupted'
FAILED = 'failed'

# Re-interrupt period while stop() waits: sqlite3_interrupt() is a no-op when issued
# while no statement runs, so a CREATE INDEX that starts right after it is caught by
# the next one.
_INTERRUPT_PERIOD_SECONDS = 0.1


def migrate_schema_at_startup(db_path) -> bool:
    """Synchronous part: everything but the photos ``INDEXES``. False on failure (logged)."""
    from db.schema import init_database
    try:
        init_database(db_path, build_photo_indexes=False)
        return True
    except Exception:
        logger.warning("Schema migration on startup failed", exc_info=True)
        return False


def _describe(names):
    return ', '.join(names)


class PhotoIndexBuild:
    """Build the missing photos ``INDEXES`` (and their ANALYZE) under the library lock.

    ``run`` is the synchronous body and returns one of ``BUILT``, ``CURRENT``,
    ``DEFERRED``, ``INTERRUPTED`` or ``FAILED``; ``start`` runs it on a daemon thread
    and ``stop`` interrupts and joins that thread.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.result = None
        self._stop = threading.Event()
        self._conn = None
        self._conn_lock = threading.Lock()
        self._thread = None

    # -- thread control --------------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._thread_main, daemon=True,
                                        name="photo-index-build")
        self._thread.start()
        return self

    def _thread_main(self):
        self.result = self.run()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout=30.0) -> bool:
        """Interrupt the build and wait for its thread; True when the thread has ended."""
        self._stop.set()
        if self._thread is None:
            return True
        deadline = time.monotonic() + timeout
        while self._thread.is_alive():
            with self._conn_lock:
                if self._conn is not None:
                    self._conn.interrupt()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._thread.join(min(_INTERRUPT_PERIOD_SECONDS, remaining))
        if self._thread.is_alive():
            logger.warning(
                "The gallery index build did not stop within %.0f s; exiting without it "
                "(the index in progress is rolled back, the next start resumes)", timeout)
            return False
        return True

    # -- body ------------------------------------------------------------------------

    def _open(self):
        from db.connection import apply_pragmas
        # isolation_level=None: every CREATE INDEX / ANALYZE is its own autocommit
        # statement. check_same_thread=False: stop() interrupts it from the lifespan.
        conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        try:
            apply_pragmas(conn)
        except BaseException:
            conn.close()
            raise
        with self._conn_lock:
            self._conn = conn
        return conn

    def _close(self, conn):
        with self._conn_lock:
            self._conn = None
        conn.close()

    def _work_pending(self) -> bool:
        from db.schema import missing_photo_indexes, photos_need_analyze
        conn = self._open()
        try:
            return bool(missing_photo_indexes(conn)) or photos_need_analyze(conn)
        finally:
            self._close(conn)

    def run(self) -> str:
        # Lazy, like api/routers/scan.py: importing facet pulls in the scoring stack,
        # which the API process otherwise avoids until a route needs it.
        from facet import (
            JOB_ORIGIN_VIEWER, LIBRARY_JOB_MAINTENANCE, LibraryLock, LibraryLockError,
        )
        try:
            # Read-only first: on a current library (every start but the first after an
            # upgrade) the lock is never taken, so a start never makes a scan launched
            # at the same moment fail on "already running" for nothing.
            if not self._work_pending():
                return CURRENT
            lock = LibraryLock(self.db_path, kind=LIBRARY_JOB_MAINTENANCE,
                               origin=JOB_ORIGIN_VIEWER)
            try:
                lock.acquire()
            except LibraryLockError as ex:
                return self._deferred(ex, lock.lock_path)
            try:
                return self._build()
            finally:
                lock.release()
        except Exception:
            logger.warning("Gallery index build failed", exc_info=True)
            return FAILED

    def _build(self) -> str:
        from db.schema import (
            analyze_photos_if_unanalyzed, create_photo_indexes, missing_photo_indexes,
            photos_need_analyze,
        )
        conn = self._open()
        missing = []
        try:
            try:
                missing = [name for name, _, _ in missing_photo_indexes(conn)]
                if not missing and not photos_need_analyze(conn):
                    return CURRENT
                started = time.monotonic()
                if missing:
                    logger.info("Building %d missing gallery index(es) in the background "
                                "while the viewer serves: %s", len(missing),
                                _describe(missing))
                built = create_photo_indexes(conn, should_stop=self._stop.is_set)
                if self._stop.is_set():
                    raise sqlite3.OperationalError("interrupted")
                analyzed = analyze_photos_if_unanalyzed(conn, should_stop=self._stop.is_set)
                if self._stop.is_set():
                    raise sqlite3.OperationalError("interrupted")
            except sqlite3.OperationalError as ex:
                if self._stop.is_set() and "interrupt" in str(ex).lower():
                    # From the database: an interrupt inside create_photo_indexes never
                    # returns its list of finished indexes.
                    finished = max(len(missing) - len(missing_photo_indexes(conn)), 0)
                    logger.info(
                        "Gallery index build stopped by shutdown after %d of %d index(es); "
                        "the next start resumes", finished, len(missing))
                    return INTERRUPTED
                raise
            logger.info("Gallery indexes ready: %d built in %.1f s%s", len(built),
                        time.monotonic() - started, ", photos analyzed" if analyzed else "")
            return BUILT
        finally:
            self._close(conn)

    def _deferred(self, ex, lock_path) -> str:
        """The lock is held elsewhere or unusable: no build; keep the planner statistics."""
        from facet import JOB_ORIGIN_VIEWER, LibraryLockUnusableError
        from db.schema import analyze_photos_if_unanalyzed, missing_photo_indexes
        holder = getattr(ex, 'holder', None) or {}
        sibling_viewer = holder.get('origin') == JOB_ORIGIN_VIEWER
        conn = self._open()
        try:
            missing = [name for name, _, _ in missing_photo_indexes(conn)]
            if isinstance(ex, LibraryLockUnusableError):
                if missing:
                    logger.warning(
                        "Cannot use the library lock file %s (%s), so the viewer does not "
                        "build the %d missing gallery index(es): %s. Make that path usable "
                        "by the viewer and restart it; until then the gallery reads without "
                        "them (slower).", lock_path, ex.__cause__ or ex, len(missing),
                        _describe(missing))
            elif missing and sibling_viewer:
                logger.info(
                    "Another viewer process (pid %s) holds the library lock and builds "
                    "the gallery indexes; this one serves meanwhile.", holder.get('pid'))
            elif missing:
                logger.warning(
                    "Library lock not available, the %d missing gallery index(es) are not "
                    "built by this start: %s. %s A facet.py job builds them at its own start; "
                    "otherwise restart the viewer once that job ends, or run "
                    "`python database.py`. Until then the gallery reads without them "
                    "(slower).", len(missing), _describe(missing), ex)
            if sibling_viewer:
                # The building worker analyzes once its indexes are in.
                return DEFERRED
            # Statistics need no library lock: one short write per index, so a scan
            # holding the lock waits at most about 0.2 s per statement.
            try:
                if analyze_photos_if_unanalyzed(conn, should_stop=self._stop.is_set):
                    logger.info("Photos analyzed for the query planner")
            except sqlite3.OperationalError as err:
                text = str(err).lower()
                if self._stop.is_set() and "interrupt" in text:
                    return INTERRUPTED
                if "locked" not in text and "busy" not in text:
                    raise
                logger.info("Planner statistics not refreshed at start-up: the database "
                            "is busy (%s); the next start retries", err)
            return DEFERRED
        finally:
            self._close(conn)
