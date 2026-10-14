"""Tests for scripts/check_packaging.py restricted-package gate."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_packaging", Path(__file__).resolve().parents[1] / "scripts" / "check_packaging.py"
)
check_packaging = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_packaging)


def test_requirement_names_normalises_and_skips_comments():
    names = check_packaging._requirement_names(
        ["# insightface>=1", "Open_CLIP_Torch==3", "pyiqa>=0.1.16  # note", "", "  numpy>=1"]
    )
    assert names == {"open-clip-torch", "pyiqa", "numpy"}


def test_repo_base_files_contain_no_restricted_packages():
    assert check_packaging._restricted_in_base() == []


def test_restricted_in_core_dependencies_is_reported(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname="x"\ndependencies = ["numpy", "insightface>=0.7"]\n', encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\npyiqa>=0.1.16\n", encoding="utf-8")
    monkeypatch.setattr(check_packaging, "PYPROJECT", pyproject)
    monkeypatch.setattr(check_packaging, "REPO_ROOT", tmp_path)
    failures = check_packaging._restricted_in_base()
    assert any("insightface" in f and "pyproject" in f for f in failures)
    assert any("pyiqa" in f and "requirements.txt" in f for f in failures)
