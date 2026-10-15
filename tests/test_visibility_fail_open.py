"""Regression tests for the photo visibility clause fail-open (whole-project review Finding 1).

An unauthenticated request (user_id=None) must see NOTHING on a protected
deployment — multi-user mode, or a single-user viewer password — and everything
only on a fully open single-user install. Authenticated users are unchanged.
"""

import sqlite3
import sys
import types
from unittest import mock

import pytest

from api import db_helpers
from api.db_helpers import get_visibility_clause
from db.schema import init_database


def test_anonymous_open_single_user_sees_all():
    """No multi-user, no password → world-readable (1=1)."""
    with (
        mock.patch("api.db_helpers.is_multi_user_enabled", return_value=False),
        mock.patch.dict("api.db_helpers.VIEWER_CONFIG", {"password": ""}, clear=False),
    ):
        sql, params = get_visibility_clause(None)
    assert sql == "1=1"
    assert params == []


def test_anonymous_blocked_when_password_set():
    """Single-user with a viewer password → anonymous sees nothing (0=1)."""
    with (
        mock.patch("api.db_helpers.is_multi_user_enabled", return_value=False),
        mock.patch.dict("api.db_helpers.VIEWER_CONFIG", {"password": "secret"}, clear=False),
    ):
        sql, params = get_visibility_clause(None)
    assert sql == "0=1"
    assert params == []


def test_anonymous_blocked_in_multi_user():
    """Multi-user mode → anonymous sees nothing (0=1)."""
    with (
        mock.patch("api.db_helpers.is_multi_user_enabled", return_value=True),
        mock.patch.dict("api.db_helpers.VIEWER_CONFIG", {"password": ""}, clear=False),
    ):
        sql, params = get_visibility_clause(None)
    assert sql == "0=1"
    assert params == []


def test_authenticated_single_user_sees_all():
    """A logged-in legacy user (sub='_legacy') in single-user mode sees all (1=1)."""
    with mock.patch("api.db_helpers.is_multi_user_enabled", return_value=False):
        sql, params = get_visibility_clause("_legacy")
    assert sql == "1=1"
    assert params == []


def test_authenticated_multi_user_scoped_to_directories(monkeypatch):
    """A multi-user user is scoped to their directories, not fail-open.

    The LIKE shape is what a case-insensitive host (Windows, macOS) emits; a
    case-sensitive host emits the range pinned in ``TestPosixCase``. The two
    assertions below are the original pin, moved here deliberately.
    """
    monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", True)
    with (
        mock.patch("api.db_helpers.is_multi_user_enabled", return_value=True),
        mock.patch("api.db_helpers.get_user_directories", return_value=["/photos/alice"]),
    ):
        sql, params = get_visibility_clause("alice")
    assert sql == "(photos.path LIKE ? ESCAPE '\\')"
    assert params == ["/photos/alice/%"]


def test_authenticated_multi_user_no_dirs_sees_nothing():
    """A multi-user user with no assigned directories sees nothing (0=1)."""
    with (
        mock.patch("api.db_helpers.is_multi_user_enabled", return_value=True),
        mock.patch("api.db_helpers.get_user_directories", return_value=[]),
    ):
        sql, params = get_visibility_clause("bob")
    assert sql == "0=1"
    assert params == []


# ---------------------------------------------------------------------------
# Directory matching: the stored form, and the host's case rules
# ---------------------------------------------------------------------------
#
# The scanner stores ``str(Path(p).resolve())``: backslashes on Windows, native
# case on both. The clause used to build ``d.rstrip('/\\') + '/'`` and match it
# with a bare ``LIKE``, which (a) never prefixed a Windows-stored path and
# (b) folded ASCII case on a case-sensitive host, so a user scoped to
# ``/mnt/photos`` also saw ``/mnt/Photos``. These cases run against the real
# schema (``init_database``), never a hand-rolled ``photos`` table.

RANGE_SQL = "(photos.path >= ? AND photos.path < ?)"
LIKE_SQL = "(photos.path LIKE ? ESCAPE '\\')"


@pytest.fixture()
def photos_db(tmp_path):
    db = str(tmp_path / "v.db")
    init_database(db)
    conn = sqlite3.connect(db)
    yield conn
    conn.close()


def _clause(directories):
    with (
        mock.patch("api.db_helpers.is_multi_user_enabled", return_value=True),
        mock.patch("api.db_helpers.get_user_directories", return_value=directories),
    ):
        return get_visibility_clause("alice")


def _visible(conn, directories, stored):
    """The stored paths (a subset of ``stored``) the clause lets through."""
    conn.execute("DELETE FROM photos")
    conn.executemany("INSERT INTO photos (path) VALUES (?)", [(p,) for p in stored])
    sql, params = _clause(directories)
    rows = conn.execute(f"SELECT path FROM photos WHERE {sql}", params).fetchall()
    return {r[0] for r in rows}


class TestWindowsStoredPaths:
    """Windows-shaped entries are matched in the backslash form, on every host."""

    @pytest.mark.parametrize("entry, stored_root", [
        ("C:\\photos\\2024", "C:\\photos\\2024"),
        ("C:/photos/2024", "C:\\photos\\2024"),
        ("C:\\photos\\2024\\", "C:\\photos\\2024"),
        ("\\\\nas\\share\\2024", "\\\\nas\\share\\2024"),
        ("\\\\?\\C:\\photos\\2024", "C:\\photos\\2024"),
        ("\\\\?\\UNC\\nas\\share\\2024", "\\\\nas\\share\\2024"),
    ])
    def test_every_spelling_of_the_directory_sees_what_the_scanner_stored(
            self, photos_db, monkeypatch, entry, stored_root):
        monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", False)
        inside = stored_root + "\\sub\\a.jpg"
        outside = [stored_root + "5\\a.jpg", stored_root + "_x\\a.jpg",
                   "D:\\elsewhere\\a.jpg"]
        sql, params = _clause([entry])
        assert len(params) == 1 and "LIKE" in sql
        assert _visible(photos_db, [entry], [inside, *outside]) == {inside}

    def test_a_bare_drive_letter_is_the_drive_root(self, photos_db):
        inside = ["D:\\a.jpg", "D:\\sub\\b.jpg"]
        assert _visible(photos_db, ["D:"], [*inside, "E:\\a.jpg", "DD:\\a.jpg"]) == set(inside)

    def test_wildcards_in_a_directory_name_match_only_itself(self, photos_db):
        entry = "C:\\ph_to\\100%"
        inside = "C:\\ph_to\\100%\\a.jpg"
        stored = [inside, "C:\\phxto\\100%\\a.jpg", "C:\\ph_to\\1000\\a.jpg"]
        assert _visible(photos_db, [entry], stored) == {inside}

    @pytest.mark.parametrize("flag", [False, True])
    def test_case_is_ignored_because_ntfs_wrote_the_path(self, photos_db, monkeypatch, flag):
        monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", flag)
        stored = "C:\\photos\\2024\\a.jpg"
        assert _visible(photos_db, ["c:\\PHOTOS\\2024"], [stored]) == {stored}

    def test_a_forward_slash_stored_row_is_not_matched(self, photos_db):
        """One stored form only: a Windows scan never writes ``C:/...`` rows."""
        assert _visible(photos_db, ["C:\\photos\\2024"], ["C:/photos/2024/a.jpg"]) == set()

    def test_two_directories_are_alternatives(self, photos_db):
        stored = ["\\\\nas\\a\\x.jpg", "\\\\nas\\b\\y.jpg", "\\\\nas\\c\\z.jpg"]
        got = _visible(photos_db, ["\\\\nas\\a", "\\\\nas\\b"], stored)
        assert got == set(stored[:2])


class TestPosixCase:
    """A POSIX-shaped entry follows the host: exact on a case-sensitive one."""

    def test_a_case_sensitive_host_gets_a_binary_range(self, photos_db, monkeypatch):
        monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", False)
        sql, params = _clause(["/mnt/photos"])
        assert sql == RANGE_SQL
        assert params == ["/mnt/photos/", "/mnt/photos0"]
        stored = ["/mnt/photos/a.jpg", "/mnt/photos/sub/b.jpg"]
        other = ["/mnt/Photos/a.jpg", "/mnt/photos2/a.jpg", "/mnt/photos.jpg"]
        assert _visible(photos_db, ["/mnt/photos"], [*stored, *other]) == set(stored)

    def test_a_case_insensitive_host_keeps_the_like(self, photos_db, monkeypatch):
        monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", True)
        sql, params = _clause(["/photos/alice"])
        assert sql == LIKE_SQL
        assert params == ["/photos/alice/%"]
        stored = ["/mnt/photos/a.jpg", "/mnt/Photos/a.jpg"]
        assert _visible(photos_db, ["/mnt/photos"], stored) == set(stored)

    def test_a_backslash_inside_a_posix_name_is_not_a_separator(self, photos_db, monkeypatch):
        monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", False)
        sql, params = _clause(["/mnt/we\\ird"])
        assert sql == RANGE_SQL and len(params) == 2
        inside = "/mnt/we\\ird/a.jpg"
        assert _visible(photos_db, ["/mnt/we\\ird"], [inside, "/mnt/we/ird/a.jpg"]) == {inside}

    @pytest.mark.parametrize("directories", [["/mnt/a"], ["/mnt/a", "/mnt/b"]])
    def test_the_range_is_an_index_search(self, photos_db, monkeypatch, directories):
        monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", False)
        sql, params = _clause(directories)
        plan = photos_db.execute(f"EXPLAIN QUERY PLAN SELECT 1 FROM photos WHERE {sql}", params).fetchall()
        details = [row[-1] for row in plan]
        assert any("SEARCH" in d for d in details), details
        assert not any("SCAN" in d for d in details), details


class TestStoredPathForm:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        db_helpers.stored_path_form.cache_clear()
        yield
        db_helpers.stored_path_form.cache_clear()

    @pytest.mark.parametrize("entry, expected", [
        ("C:/photos/2024/", "C:\\photos\\2024"),
        ("\\\\?\\C:\\photos\\2024", "C:\\photos\\2024"),
        ("\\\\?\\UNC\\nas\\share\\2024", "\\\\nas\\share\\2024"),
        ("D:", "D:\\"),
    ])
    def test_a_non_windows_host_normalises_lexically(self, monkeypatch, entry, expected):
        monkeypatch.setattr(db_helpers, "sys", types.SimpleNamespace(platform="linux"))
        assert db_helpers.stored_path_form(entry) == expected

    def test_a_posix_path_is_returned_untouched(self):
        assert db_helpers.stored_path_form("/mnt//photos/../x/") == "/mnt//photos/../x/"

    @pytest.mark.skipif(sys.platform != "win32", reason="canonical on-disk case is an NTFS property")
    def test_a_windows_host_takes_the_case_the_disk_holds(self, tmp_path):
        (tmp_path / "MixedCase").mkdir()
        got = db_helpers.stored_path_form(str(tmp_path / "mixedcase"))
        assert got == str((tmp_path / "MixedCase").resolve())
