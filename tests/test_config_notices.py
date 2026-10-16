"""Config notices: obsolete keys that are ignored must be NAMED, never aliased.

`models.clip.similarity_threshold_percent` / `models.clip_legacy.similarity_threshold_percent`
were the old tag-gate keys. The gate is now `tag_threshold_percent` in the active
profile's clip block; an operator file still carrying the old key keeps it inert
(no alias, repo rule) and the load log plus `--doctor` say so.
"""

import json
import logging

import pytest

MESSAGE_FRAGMENT = "similarity_threshold_percent is no longer read"


def _write(tmp_path, models, name="scoring_config.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def _no_env_profile(monkeypatch):
    monkeypatch.delenv("FACET_VRAM_PROFILE", raising=False)


def test_shipped_defaults_carry_no_notice():
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path

    cfg = ScoringConfig(defaults_path(), validate=False)
    assert cfg.validate_obsolete_clip_tag_gate_keys() == []


def test_obsolete_tag_gate_keys_warn_once_naming_the_new_key(tmp_path, caplog):
    from config.scoring_config import ScoringConfig
    from diagnostics import run_doctor

    path = _write(tmp_path, {
        "vram_profile": "legacy",
        "clip": {"similarity_threshold_percent": 8},
        "clip_legacy": {"similarity_threshold_percent": 22},
    })

    messages = ScoringConfig(path, validate=False).validate_obsolete_clip_tag_gate_keys()
    assert len(messages) == 2
    for block, message in zip(("clip", "clip_legacy"), sorted(messages)):
        assert message.startswith(f"models.{block}.similarity_threshold_percent ")
        assert f"models.{block}.tag_threshold_percent" in message

    # Only the block that still carries the key is named.
    only_legacy = _write(tmp_path, {"clip_legacy": {"similarity_threshold_percent": 22}},
                         name="only_legacy.json")
    messages = ScoringConfig(only_legacy, validate=False).validate_obsolete_clip_tag_gate_keys()
    assert len(messages) == 1 and "models.clip_legacy." in messages[0]

    # Logged exactly once per block on a validating load.
    with caplog.at_level(logging.WARNING, logger="facet.config"):
        ScoringConfig(path)
    logged = [r for r in caplog.records if MESSAGE_FRAGMENT in r.getMessage()]
    assert len(logged) == 2

    # And --doctor prints it, in its own section.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="facet.diagnostics"):
        run_doctor(config_path=path, db_path=str(tmp_path / "no.db"))
    out = "\n".join(r.getMessage() for r in caplog.records if r.name == "facet.diagnostics")
    assert "  Config notices" in out
    assert "[!!] Config notice: models.clip.similarity_threshold_percent" in out
    assert "models.clip_legacy.similarity_threshold_percent is no longer read" in out
    assert "models.clip_legacy.tag_threshold_percent" in out
