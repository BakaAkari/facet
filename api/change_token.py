"""Cross-process database-change token for the count / hidden-summary cache keys.

``PRAGMA data_version`` changes on every commit made by any OTHER connection of
the database, whether it lives in this process or in another one (the
``--detect-junk`` / ``--recompute-*`` CLI jobs), and on nothing else. It only
means something on a connection that outlives the commit, so this module keeps
one lazily-opened read-only "sentinel" connection per database path. A token
built from the on-disk size and mtime of the database and its ``-wal`` cannot do
this job: a pure read moves the ``-wal`` mtime, so every request got a fresh
key and the cache never hit, while a long-lived writer's commits slip past
size+mtime.

The token is ``generation:device:inode:data_version``:

* ``generation`` comes from a process-wide counter bumped on every (re)open, so
  a reopened sentinel (whose ``data_version`` restarts) can never reproduce an
  earlier token.
* ``device:inode`` are re-read from ``os.stat`` on every call; a replaced file
  (rsync-rename, restore) would otherwise leave the sentinel reading the old
  inode forever. A change closes and reopens the sentinel.

Any failure returns a never-repeating "miss" token, which is the pre-existing
behaviour (the cache simply does not hit). The sentinel is host-local, like
``facet.LibraryLock``: ``data_version`` rides the shared-memory WAL index and
does not see a commit made by another host on a network share.

The sentinel keeps a handle on the database for the life of the process, so the
lifespan must call :func:`close_change_token_sentinels` BEFORE its final
``wal_checkpoint(TRUNCATE)``: a read-only connection cannot delete the WAL at
last close.
"""

import itertools
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("facet.api.change_token")

# Short enough that a shm-lock wait cannot stall the event loop (the token is
# computed synchronously inside async routes), long enough to ride out the
# microseconds a checkpoint holds the WAL index.
_BUSY_TIMEOUT_SECONDS = 0.1

_lock = threading.Lock()
_generation = itertools.count(1)
_misses = itertools.count(1)


@dataclass
class _Sentinel:
    conn: sqlite3.Connection
    generation: int
    device: int
    inode: int


_sentinels: dict[str, _Sentinel] = {}
_warned: set[str] = set()


def _miss_token() -> str:
    return f"x{next(_misses)}"


def _warn_once(key: str, reason: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning("DB change token unavailable for %s (%s); the count cache "
                   "will not hit until it recovers", key, reason)


def _read_data_version(conn: sqlite3.Connection) -> int:
    # fetchall() finishes the cursor, so the sentinel never pins a read
    # transaction (which would turn a TRUNCATE checkpoint busy).
    return int(conn.execute("PRAGMA data_version").fetchall()[0][0])


def _open_sentinel(key: str, st: os.stat_result) -> _Sentinel:
    uri = Path(key).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None,
                           check_same_thread=False, timeout=_BUSY_TIMEOUT_SECONDS)
    return _Sentinel(conn, next(_generation), st.st_dev, st.st_ino)


def _drop(key: str) -> None:
    sentinel = _sentinels.pop(key, None)
    if sentinel is not None:
        try:
            sentinel.conn.close()
        except sqlite3.Error:
            logger.debug("closing the change-token sentinel for %s failed", key, exc_info=True)


def db_change_token(db_path) -> str:
    """Return a string that differs after any commit to ``db_path``."""
    key = os.fspath(db_path)
    try:
        st = os.stat(key)
    except OSError as ex:
        with _lock:
            _drop(key)
            _warn_once(key, f"stat failed: {ex}")
        return _miss_token()

    with _lock:
        sentinel = _sentinels.get(key)
        if sentinel is not None and (sentinel.device, sentinel.inode) != (st.st_dev, st.st_ino):
            _drop(key)
            sentinel = None
        if sentinel is None:
            try:
                sentinel = _open_sentinel(key, st)
            except (sqlite3.Error, OSError, ValueError) as ex:
                _warn_once(key, f"open failed: {ex}")
                return _miss_token()
            _sentinels[key] = sentinel
        try:
            data_version = _read_data_version(sentinel.conn)
        except sqlite3.OperationalError as ex:
            message = str(ex).lower()
            if "locked" in message or "busy" in message:
                logger.debug("change-token read busy for %s: %s", key, ex)
                return _miss_token()
            _drop(key)
            _warn_once(key, f"read failed: {ex}")
            return _miss_token()
        except (sqlite3.Error, IndexError, TypeError, ValueError) as ex:
            _drop(key)
            _warn_once(key, f"read failed: {ex}")
            return _miss_token()
        return f"{sentinel.generation}:{sentinel.device}:{sentinel.inode}:{data_version}"


def close_change_token_sentinels() -> None:
    """Close every sentinel and forget it. Idempotent."""
    with _lock:
        for key in list(_sentinels):
            _drop(key)
        _warned.clear()
