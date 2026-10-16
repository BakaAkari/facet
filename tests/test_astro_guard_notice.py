"""``validate_astro_guard``: an operator's own ``categories`` array silently drops the astro guard.

``config_resolve`` replaces a list wholesale, so a ``scoring_config.json`` that carries a
``categories`` array (every file a viewer weight/priority/filter save or an optimizer apply
wrote, and every pre-1.14.3 full config) keeps the OLD ``astro`` filter and never receives the
shipped ``luminance_max``. It is not migrated (no shim); ``--doctor`` names it.
"""

import copy
import json
import logging

import pytest

from config.scoring_config import ScoringConfig
from config_resolve import defaults_path, load_defaults

FRAGMENT = "categories.astro has no luminance_max"


@pytest.fixture(autouse=True)
def _no_env_profile(monkeypatch):
    monkeypatch.delenv("FACET_VRAM_PROFILE", raising=False)


def _categories(with_guard):
    categories = copy.deepcopy(load_defaults()["categories"])
    astro = next(c for c in categories if c["name"] == "astro")
    assert astro["filters"]["luminance_max"] == 0.30   # the shipped default carries the guard
    if not with_guard:
        del astro["filters"]["luminance_max"]
    return categories


def _write(tmp_path, name, override):
    path = tmp_path / name
    path.write_text(json.dumps(override), encoding="utf-8")
    return str(path)


@pytest.fixture()
def old_array(tmp_path):
    """An operator file whose categories array was copied from the OLD defaults."""
    return _write(tmp_path, "old.json", {"categories": _categories(with_guard=False)})


@pytest.fixture()
def quiet_configs(tmp_path):
    return {
        "array_with_guard": _write(tmp_path, "with.json", {"categories": _categories(with_guard=True)}),
        "no_categories": _write(tmp_path, "none.json", {"viewer": {"defaults": {"hide_bursts": True}}}),
    }


def test_an_old_categories_array_is_named_once(old_array):
    messages = ScoringConfig(old_array, validate=False).validate_astro_guard()
    assert len(messages) == 1
    for word in ("astro", "luminance_max", "0.30", "astro.filters"):
        assert word in messages[0]
    assert messages[0].startswith(FRAGMENT)


def test_the_shipped_defaults_carry_no_notice():
    assert ScoringConfig(defaults_path(), validate=False).validate_astro_guard() == []


def test_an_array_that_carries_the_key_and_an_override_without_one_carry_no_notice(quiet_configs):
    for path in quiet_configs.values():
        assert ScoringConfig(path, validate=False).validate_astro_guard() == []


def test_an_array_without_astro_at_all_is_not_flagged(tmp_path):
    categories = [c for c in _categories(with_guard=False) if c["name"] != "astro"]
    path = _write(tmp_path, "noastro.json", {"categories": categories})
    assert ScoringConfig(path, validate=False).validate_astro_guard() == []


def test_a_different_bound_in_the_operators_array_is_their_choice(tmp_path):
    categories = _categories(with_guard=True)
    next(c for c in categories if c["name"] == "astro")["filters"]["luminance_max"] = 0.5
    path = _write(tmp_path, "custom.json", {"categories": categories})
    assert ScoringConfig(path, validate=False).validate_astro_guard() == []


def test_it_is_not_logged_at_load(old_array, caplog):
    with caplog.at_level(logging.WARNING, logger="facet.config"):
        ScoringConfig(old_array)   # a validating load logs config_notices()
    assert not [r for r in caplog.records if FRAGMENT in r.getMessage()]


def _doctor_output(path, tmp_path, caplog):
    from diagnostics import run_doctor
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="facet.diagnostics"):
        run_doctor(config_path=path, db_path=str(tmp_path / "no.db"))
    return "\n".join(r.getMessage() for r in caplog.records if r.name == "facet.diagnostics")


def test_doctor_prints_it_for_the_old_array_and_not_for_the_others(old_array, quiet_configs, tmp_path, caplog):
    out = _doctor_output(old_array, tmp_path, caplog)
    assert "  Config notices" in out
    assert f"[!!] Config notice: {FRAGMENT}" in out
    for path in quiet_configs.values():
        out = _doctor_output(path, tmp_path, caplog)
        assert "  Config notices" in out
        assert FRAGMENT not in out
