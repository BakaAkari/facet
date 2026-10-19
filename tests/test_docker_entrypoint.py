"""Tests for docker-entrypoint.sh's config-seeding logic (issue #127).

Drives the real script rather than re-implementing its shell logic in Python:
a `SEEDED_CONFIG` / `IMAGE_CONFIG` env-var seam (see the top of
docker-entrypoint.sh) lets these tests point ``seed_config()`` at a temp
directory instead of ``/config`` and ``/app``, with the container command
replaced by ``true`` so nothing besides the seeding logic actually runs.

The root branch's privilege drop and readability probe are covered by the
stub-PATH ``TestDropPrivileges`` class below (stub ``id``/``chown``/``setpriv``
on PATH); the real ownership takeover still needs uid 0 and is not covered.

The NON-ROOT tail (``mkdir -p /config; seed_config; exec "$@"``) is covered by
the real-script tests: the module is skipped outright under uid 0 (see
``_IS_ROOT`` below) so that `id -u` is always non-zero and the script falls
through to that tail, which calls the exact same ``seed_config()`` the root
branch calls first. What the root branch adds on top (chowning a freshly
seeded file unconditionally, chowning a pre-existing one only as a last resort
behind a readability probe) is covered through the stubs in
``TestDropPrivileges``, which assert the ``chown``/``setpriv`` calls; the real
ownership change still needs uid 0 and is not exercised by a real-uid-0 test.
The intended home for one is a uid-0 container smoke step in
.github/workflows/docker-publish.yml that bind-mounts a config unreadable by
the ``facet`` uid.

This is the regression suite for issue #127: under rootless Podman, an
operator's pre-existing ``/config/scoring_config.json`` was re-chmod'd 0600
(and, in the root branch, re-chowned) on every single container start,
locking them out of editing it from the host without root.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parent.parent / "docker-entrypoint.sh"


def _mode_of(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _run_entrypoint(
    seeded_config, image_config, python_bin=None, facet_config="seeded", extra_env=None, cwd=None
):
    """Run docker-entrypoint.sh's non-root tail against a temp SEEDED_CONFIG.

    ``facet_config="seeded"`` mirrors docker-compose.yml (FACET_CONFIG names the
    seed); ``None`` leaves it unset, as a plain ``docker run`` does.
    """
    env = dict(os.environ)
    env.pop("FACET_CONFIG", None)
    env.pop("GENERATED_PASSWORD", None)
    if python_bin is not None:
        env["FACET_ENTRYPOINT_PYTHON"] = python_bin
    if facet_config == "seeded":
        env["FACET_CONFIG"] = str(seeded_config)
    elif facet_config is not None:
        env["FACET_CONFIG"] = facet_config
    env.update(extra_env or {})
    env["SEEDED_CONFIG"] = str(seeded_config)
    env["IMAGE_CONFIG"] = str(image_config)
    return subprocess.run(
        ["sh", str(ENTRYPOINT), "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )


# os.geteuid does not exist on Windows and skipif decorators evaluate at
# import time, so it is read the same guarded way the sibling suites read it
# (tests/test_config_writes.py, tests/test_api_config.py).
_IS_ROOT = os.geteuid() == 0 if hasattr(os, "geteuid") else False

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or _IS_ROOT,
    reason="POSIX permission bits, symlinks and sh do not apply on Windows; "
    "and under uid 0 the script's root branch touches absolute host paths "
    "(/app/data, /app/storage, /config, /home/facet/*) and requires uid 0 and the "
    "image's setpriv (the stub-PATH class fakes both)",
)


class TestSeedConfigNonRoot:
    """seed_config() driven through the real script, SEEDED_CONFIG in tmp_path."""

    def test_fresh_seed_carries_a_generated_edition_password_printed_once(self, tmp_path):
        seeded = tmp_path / "scoring_config.json"
        image = tmp_path / "no_such_image_config.json"  # deliberately absent

        result = _run_entrypoint(seeded, image)

        assert result.returncode == 0, result.stderr
        password = json.loads(seeded.read_text())["viewer"]["edition_password"]
        assert len(password) >= 24
        assert _mode_of(seeded) == 0o600
        assert result.stdout.count(password) == 1
        assert password not in result.stderr

    def test_failed_password_generation_seeds_empty_and_prints_nothing(self, tmp_path):
        seeded = tmp_path / "scoring_config.json"

        result = _run_entrypoint(
            seeded, tmp_path / "no_such_image_config.json", python_bin="/nonexistent/python3"
        )

        assert result.returncode == 0, result.stderr
        assert seeded.read_text() == "{}\n"
        assert _mode_of(seeded) == 0o600
        assert "edition_password" not in result.stdout
        assert "no password was set" in result.stderr

    def test_no_facet_config_seeds_empty_and_prints_nothing(self, tmp_path):
        """A plain `docker run`: the server reads another file, so a password could never work."""
        seeded = tmp_path / "scoring_config.json"

        result = _run_entrypoint(
            seeded, tmp_path / "no_such_image_config.json", facet_config=None
        )

        assert result.returncode == 0, result.stderr
        assert seeded.read_text() == "{}\n"
        assert result.stdout == ""
        assert "password" not in result.stderr

    def test_facet_config_elsewhere_seeds_empty_and_prints_nothing(self, tmp_path):
        seeded = tmp_path / "scoring_config.json"

        result = _run_entrypoint(
            seeded,
            tmp_path / "no_such_image_config.json",
            facet_config=str(tmp_path / "other.json"),
        )

        assert result.returncode == 0, result.stderr
        assert seeded.read_text() == "{}\n"
        assert result.stdout == ""

    def test_planted_secrets_module_in_cwd_is_not_imported(self, tmp_path):
        """The generator runs as root in the image with /app (facet-writable) as CWD."""
        seeded = tmp_path / "scoring_config.json"
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        marker = tmp_path / "pwned"
        (cwd / "secrets.py").write_text(
            f"open({str(marker)!r}, 'w').write('x')\n"
            "def token_urlsafe(n):\n    return 'planted'\n"
        )

        result = _run_entrypoint(
            seeded,
            tmp_path / "no_such_image_config.json",
            extra_env={"PYTHONPATH": str(cwd)},
            cwd=cwd,
        )

        assert result.returncode == 0, result.stderr
        assert not marker.exists()
        assert json.loads(seeded.read_text())["viewer"]["edition_password"] != "planted"

    def test_inherited_generated_password_is_never_announced(self, tmp_path):
        seeded = tmp_path / "scoring_config.json"

        result = _run_entrypoint(
            seeded,
            tmp_path / "no_such_image_config.json",
            python_bin="/nonexistent/python3",
            extra_env={"GENERATED_PASSWORD": "inherited-secret"},
        )

        assert result.returncode == 0, result.stderr
        assert "inherited-secret" not in result.stdout + result.stderr
        assert seeded.read_text() == "{}\n"

    def test_inherited_generated_password_not_announced_on_image_copy(self, tmp_path):
        image = tmp_path / "image_config.json"
        image.write_text("{}\n")

        result = _run_entrypoint(
            tmp_path / "scoring_config.json",
            image,
            extra_env={"GENERATED_PASSWORD": "inherited-secret"},
        )

        assert "inherited-secret" not in result.stdout + result.stderr

    def test_unwritable_seed_prints_no_password(self, tmp_path):
        seeded = tmp_path / "missing_dir" / "scoring_config.json"

        result = _run_entrypoint(seeded, tmp_path / "no_such_image_config.json")

        assert result.returncode == 0, result.stderr
        assert not seeded.exists()
        assert "edition_password" not in result.stdout

    def test_preexisting_file_keeps_mode_inode_and_content(self, tmp_path):
        """Regression test for #127: an operator-owned config is left alone."""
        seeded = tmp_path / "scoring_config.json"
        seeded.write_text('{"real": "config"}\n')
        seeded.chmod(0o644)
        inode_before = seeded.stat().st_ino
        mode_before = _mode_of(seeded)
        assert mode_before == 0o644

        result = _run_entrypoint(seeded, tmp_path / "no_such_image_config.json")

        assert result.returncode == 0, result.stderr
        assert seeded.stat().st_ino == inode_before
        assert _mode_of(seeded) == mode_before
        assert seeded.read_text() == '{"real": "config"}\n'
        assert result.stdout == ""

    def test_symlinked_seeded_config_is_refused(self, tmp_path):
        target = tmp_path / "target.json"
        target.write_text('{"real": 1}\n')
        target.chmod(0o640)
        link = tmp_path / "scoring_config.json"
        link.symlink_to(target)

        result = _run_entrypoint(link, tmp_path / "no_such_image_config.json")

        assert result.returncode == 0, result.stderr
        assert link.is_symlink()
        assert os.readlink(link) == str(target)
        assert _mode_of(target) == 0o640
        assert target.read_text() == '{"real": 1}\n'
        assert "edition_password" not in result.stdout

    def test_image_config_present_is_copied_verbatim(self, tmp_path):
        image = tmp_path / "image_config.json"
        image.write_text('{"weights": {"x": 1}}\n')
        seeded = tmp_path / "scoring_config.json"

        result = _run_entrypoint(seeded, image)

        assert result.returncode == 0, result.stderr
        assert seeded.read_text() == image.read_text()
        assert _mode_of(seeded) == 0o600
        assert result.stdout == ""


def _stub_bin(tmp_path, id_groups):
    """A PATH dir of stubs that lets the ROOT branch run on a non-root host.

    ``id`` claims uid 0, primary gid 1000 for ``facet`` and ``id_groups`` for
    ``-G``; ``mkdir`` is inert (the branch names absolute host paths); ``chown``
    and ``setpriv`` append their arguments to a log. ``setpriv``'s exit status is
    ``$STUB_SETPRIV_RC``. There is deliberately no ``gosu``.
    """
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    stubs = {
        "id": (
            'case "$1" in\n'
            "  -u) echo 0 ;;\n"
            "  -g) echo 1000 ;;\n"
            f"  -G) echo '{id_groups}' ;;\n"
            "esac\n"
        ),
        "mkdir": "exit 0\n",
        "chown": f'echo "chown $*" >> "{log}"\n',
        "setpriv": f'echo "setpriv $*" >> "{log}"\nexit "${{STUB_SETPRIV_RC:-0}}"\n',
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\n" + body)
        stub.chmod(0o755)
    return bin_dir, log


def _run_root_branch(
    tmp_path, id_groups, setpriv_rc=0, preexisting=True, image_config=None
):
    bin_dir, log = _stub_bin(tmp_path, id_groups)
    seeded = tmp_path / "scoring_config.json"
    if preexisting:
        seeded.write_text("{}\n")
    env = dict(os.environ)
    env.pop("FACET_CONFIG", None)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["SEEDED_CONFIG"] = str(seeded)
    env["IMAGE_CONFIG"] = str(image_config or tmp_path / "no_such_image_config.json")
    env["STUB_SETPRIV_RC"] = str(setpriv_rc)
    result = subprocess.run(
        ["sh", str(ENTRYPOINT), "myserver", "--flag"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


class TestDropPrivileges:
    """Issue #208: the root branch keeps compose ``group_add`` gids across the drop."""

    def test_extra_group_survives_final_exec_and_probe(self, tmp_path):
        result, calls = _run_root_branch(tmp_path, "0 101")
        privs = [c for c in calls if c.startswith("setpriv")]
        assert len(privs) == 2, (result.stderr, calls)
        for line in privs:
            assert "--reuid=facet --regid=facet --groups=1000,101 --" in line
        assert privs[-1].endswith("-- myserver --flag")

    def test_no_group_add_still_passes_a_nonempty_group_list(self, tmp_path):
        _, calls = _run_root_branch(tmp_path, "0")
        privs = [c for c in calls if c.startswith("setpriv")]
        assert privs and all("--groups=1000 --" in line for line in privs), calls

    def test_root_and_nobody_gids_are_filtered_several_extras_kept(self, tmp_path):
        _, calls = _run_root_branch(tmp_path, "0 101 65534 102")
        privs = [c for c in calls if c.startswith("setpriv")]
        assert privs and all("--groups=1000,101,102 --" in line for line in privs), calls

    def test_group_readable_config_is_not_taken_over(self, tmp_path):
        result, calls = _run_root_branch(tmp_path, "0 101", setpriv_rc=0)
        assert not any(c.startswith("chown facet:facet") for c in calls), calls
        assert "cannot read" not in result.stderr

    def test_probe_failure_is_treated_as_unreadable(self, tmp_path):
        result, calls = _run_root_branch(tmp_path, "0 101", setpriv_rc=1)
        assert any(c.startswith("chown facet:facet") for c in calls), calls
        assert "cannot read" in result.stderr

    def test_freshly_seeded_config_is_chowned_without_a_probe(self, tmp_path):
        image = tmp_path / "image_config.json"
        image.write_text("{}\n")
        result, calls = _run_root_branch(
            tmp_path, "0 101", preexisting=False, image_config=image
        )
        assert result.returncode == 0, result.stderr
        seeded = str(tmp_path / "scoring_config.json")
        assert f"chown facet:facet {seeded}" in calls, calls
        assert len([c for c in calls if c.startswith("setpriv")]) == 1, calls
        assert "cannot read" not in result.stderr
