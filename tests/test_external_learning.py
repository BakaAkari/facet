"""Temporary SQL and recorded calls only; never trains/evaluates a model."""

import importlib
import sqlite3
import sys
from types import SimpleNamespace

import pytest

import optimization.auto_retrain as ar
import optimization.learning_policy as lp
real_backend = lp.learning_backend


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE stats_cache(key TEXT PRIMARY KEY,value TEXT,updated_at REAL)"
        )
    monkeypatch.setattr(lp, "learning_backend", lambda *args: "external")
    with ar._retrain_lock:
        ar._retrain_running = False
        ar._pending_timers.clear()
    yield path
    ar.cancel_pending_retrains()


def count(path, scope):
    with sqlite3.connect(path) as conn:
        return ar._read_counter(conn, scope)


@pytest.mark.parametrize("scope", [None, "owner"])
@pytest.mark.parametrize("total", [25, 30, 100])
def test_external_retains_every_count_without_dispatch(db, monkeypatch, scope, total):
    monkeypatch.setattr(
        ar,
        "_dispatch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("NO_DISPATCH")),
    )
    monkeypatch.setattr(
        ar,
        "_schedule",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("NO_TIMER")),
    )
    for _ in range(total):
        assert (
            ar.maybe_retrain(db, scope, added=1, threshold=25, idle_seconds=0) is False
        )
    assert count(db, scope) == total
    assert not ar._retrain_running


def test_old_timer_external_dispatch_never_consumes_pending(db, monkeypatch):
    with sqlite3.connect(db) as conn:
        ar._bump_counter(conn, None, 35)
        conn.commit()
    monkeypatch.setattr(
        ar,
        "_consume_counter",
        lambda *args: (_ for _ in ()).throw(AssertionError("NO_CONSUME")),
    )
    assert ar._dispatch(db, None, 25) is False
    assert count(db, None) == 35


def test_old_consumed_worker_restores_once_before_any_model_or_librarylock(
    db, monkeypatch
):
    monkeypatch.setattr(
        ar,
        "_hold_library_lock",
        lambda *args: (_ for _ in ()).throw(AssertionError("NO_LOCAL_TRAIN")),
    )
    ar._retrain_running = True
    ar._run_retrain(db, None, pending=25)
    assert count(db, None) == 25
    assert ar._retrain_running is False


def test_core_deferred_race_restores_once_and_does_not_call_keeper(db, monkeypatch):
    monkeypatch.setattr(lp, "learning_backend", lambda *args: "native")
    released = []
    monkeypatch.setattr(
        ar,
        "_hold_library_lock",
        lambda *args: SimpleNamespace(release=lambda: released.append(True)),
    )
    pr = importlib.import_module("optimization.personal_ranker")
    keeper = importlib.import_module("optimization.keeper_head")
    monkeypatch.setattr(
        pr, "train_ranker", lambda **kwargs: lp.external_learning_result()
    )
    monkeypatch.setattr(
        keeper,
        "train_keeper_head",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("NO_KEEPER")),
    )
    ar._run_retrain(db, "owner", pending=30)
    assert count(db, "owner") == 30
    assert released == [True]


@pytest.mark.parametrize(
    "document,expected",
    [
        ({}, "native"),
        ({"auto_retrain": {"backend": "native"}}, "native"),
        ({"auto_retrain": {"backend": "external"}}, "external"),
        ({"auto_retrain": {"backend": "typo"}}, "external"),
        ({"auto_retrain": []}, "external"),
        ({"auto_retrain": {"backend": []}}, "external"),
    ],
)
def test_default_native_and_invalid_policy_failclosed(monkeypatch, document, expected):
    import config_resolve

    monkeypatch.setattr(config_resolve, "load_resolved", lambda *args: document)
    assert real_backend() == expected


def test_unreadable_policy_cannot_enable_cpu(monkeypatch):
    import config_resolve

    monkeypatch.setattr(
        config_resolve,
        "load_resolved",
        lambda *args: (_ for _ in ()).throw(ValueError("broken config")),
    )
    assert real_backend() == "external"


def function(name, module_name, class_name=None):
    module = importlib.import_module(module_name)
    target = getattr(module, class_name) if class_name else module
    return getattr(target, name)


@pytest.mark.parametrize(
    "name,module_name",
    [
        ("train_ranker", "optimization.personal_ranker"),
        ("train_keeper_head", "optimization.keeper_head"),
    ],
)
def test_actual_manual_train_core_defers_before_any_fit(db, name, module_name):
    result = function(name, module_name)(db_path="/must-not-open")
    assert result["deferred"] is True
    assert result["written"] == 0
    assert "error" in result


@pytest.mark.parametrize(
    "name",
    [
        "optimize_weights_direct",
        "optimize_weights_with_cv",
        "compute_weight_confidence",
    ],
)
def test_weight_cli_methods_have_no_validation_inference_fallback(db, name):
    fn = function(name, "optimization.weight_optimizer", "WeightOptimizer")
    assert fn(SimpleNamespace(config_path=None))["deferred"] is True


@pytest.mark.parametrize(
    "name", ["api_comparison_learned_weights", "api_comparison_confidence"]
)
def test_actual_compare_manual_json_contract_is_unavailable_not_servererror(db, name):
    fn = function(name, "api.routers.comparison")
    kwargs = {"category": None}
    if name == "api_comparison_learned_weights":
        kwargs.update(include_ties="true", use_cv="false")
    result = fn(**kwargs)
    assert result["available"] is False
    assert "external GPU learning" in result["message"]
    from api.models.comparison import LearnedWeightsResponse

    if name == "api_comparison_learned_weights":
        assert LearnedWeightsResponse.model_validate(result).available is False


@pytest.mark.parametrize(
    "name,path",
    [
        ("learned_weights", "/api/comparison/learned_weights"),
        ("confidence", "/api/comparison/confidence"),
    ],
)
def test_actual_compare_http_json_contract_external_no_cpu(db, edition_client, name, path):
    response = edition_client.get(path)
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "external GPU learning" in response.json()["message"]


def test_backend_uses_authoritative_relocated_server_config(monkeypatch):
    import config_resolve

    seen = []
    monkeypatch.setitem(
        sys.modules,
        "api.config",
        SimpleNamespace(
            server_config_path=lambda: "/synthetic/library/scoring_config.json"
        ),
    )
    monkeypatch.setattr(
        config_resolve,
        "load_resolved",
        lambda path: seen.append(path) or {"auto_retrain": {"backend": "external"}},
    )
    assert real_backend() == "external"
    assert seen == ["/synthetic/library/scoring_config.json"]


def test_invalid_backend_warns_once_per_distinct_value(monkeypatch, caplog):
    import config_resolve

    monkeypatch.setattr(lp, "_warned_values", set())
    monkeypatch.setattr(
        config_resolve, "load_resolved", lambda *a: {"auto_retrain": {"backend": "typo"}}
    )
    with caplog.at_level("WARNING", logger=lp.logger.name):
        assert real_backend() == "external"
        assert real_backend() == "external"
    assert len([r for r in caplog.records if "typo" in r.getMessage()]) == 1
    monkeypatch.setattr(
        config_resolve, "load_resolved", lambda *a: {"auto_retrain": ["x"]}
    )
    with caplog.at_level("WARNING", logger=lp.logger.name):
        assert real_backend() == "external"
    assert len([r for r in caplog.records if "not an object" in r.getMessage()]) == 1


def test_flip_to_external_cancels_and_deregisters_armed_timer(db, monkeypatch):
    class FakeTimer:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    timer = FakeTimer()
    with ar._retrain_lock:
        ar._pending_timers[None] = (timer, db, 25)
    assert ar.maybe_retrain(db, None, added=1, threshold=25, idle_seconds=0) is False
    assert timer.cancelled is True
    assert None not in ar._pending_timers


def test_dispatch_external_deregisters_only_the_firing_timer(db):
    firing, newer = SimpleNamespace(cancel=lambda: None), SimpleNamespace(cancel=lambda: None)
    with ar._retrain_lock:
        ar._pending_timers[None] = (firing, db, 25)
    assert ar._dispatch(db, None, 25, firing_timer=firing) is False
    assert None not in ar._pending_timers
    with ar._retrain_lock:
        ar._pending_timers[None] = (newer, db, 25)
    assert ar._dispatch(db, None, 25, firing_timer=firing) is False
    assert ar._pending_timers[None][0] is newer


def test_flip_to_external_between_ranker_and_keeper_skips_keeper(db, monkeypatch):
    backends = iter(["native", "external"])
    monkeypatch.setattr(lp, "learning_backend", lambda *a: next(backends))
    monkeypatch.setattr(
        ar, "_hold_library_lock", lambda *a: SimpleNamespace(release=lambda: None)
    )
    pr = importlib.import_module("optimization.personal_ranker")
    keeper = importlib.import_module("optimization.keeper_head")
    monkeypatch.setattr(pr, "train_ranker", lambda **kw: {"written": 3, "cv_accuracy": 70.0})
    keeper_calls = []
    monkeypatch.setattr(
        keeper, "train_keeper_head", lambda **kw: keeper_calls.append(kw) or {}
    )
    ar._retrain_running = True
    ar._run_retrain(db, "owner", pending=30)
    assert ar._retrain_running is False
    assert keeper_calls == []


def test_train_ranker_gates_on_the_resolved_config_path(db, monkeypatch):
    pr = importlib.import_module("optimization.personal_ranker")
    seen = []
    monkeypatch.setattr(pr, "resolve_scoring_config_path", lambda p: "/resolved/cfg.json")
    monkeypatch.setattr(
        lp, "learning_backend", lambda path=None: seen.append(path) or "external"
    )
    assert pr.train_ranker(db_path="/must-not-open")["deferred"] is True
    assert seen == ["/resolved/cfg.json"]


def test_external_keeper_head_gate_skips_every_per_group_query(db, monkeypatch):
    bc = importlib.import_module("api.routers.burst_culling")
    keeper = importlib.import_module("optimization.keeper_head")
    monkeypatch.setattr(
        keeper, "load_keeper_head", lambda *a: (_ for _ in ()).throw(AssertionError("NO_LOAD"))
    )
    monkeypatch.setattr(
        bc, "_keeper_rows", lambda *a: (_ for _ in ()).throw(AssertionError("NO_ROWS"))
    )
    assert bc._keeper_head_for(None, "owner") is None
    groups = [{"photos": [{"path": "/a.jpg"}]}]
    bc._for_each_group_probs(None, groups, "owner", lambda *a: pytest.fail("NO_APPLY"))
