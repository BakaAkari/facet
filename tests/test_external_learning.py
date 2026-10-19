"""Temporary SQL and recorded calls only; never trains/evaluates a model."""

import importlib
import sqlite3
import sys
from types import SimpleNamespace

import pytest

import optimization.auto_retrain as ar
real_backend = ar.learning_backend


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "learning.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE stats_cache(key TEXT PRIMARY KEY,value TEXT,updated_at REAL)"
        )
    monkeypatch.setattr(ar, "learning_backend", lambda *args: "external")
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
    assert count(db, scope) == total and not ar._retrain_running


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
    assert count(db, None) == 25 and ar._retrain_running is False


def test_core_deferred_race_restores_once_and_does_not_call_keeper(db, monkeypatch):
    monkeypatch.setattr(ar, "learning_backend", lambda *args: "native")
    released = []
    monkeypatch.setattr(
        ar,
        "_hold_library_lock",
        lambda *args: SimpleNamespace(release=lambda: released.append(True)),
    )
    pr = importlib.import_module("optimization.personal_ranker")
    keeper = importlib.import_module("optimization.keeper_head")
    monkeypatch.setattr(
        pr, "train_ranker", lambda **kwargs: ar.external_learning_result()
    )
    monkeypatch.setattr(
        keeper,
        "train_keeper_head",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("NO_KEEPER")),
    )
    ar._run_retrain(db, "owner", pending=30)
    assert count(db, "owner") == 30 and released == [True]


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


def function(relative, name, module_name, class_name=None):
    module = importlib.import_module(module_name)
    target = getattr(module, class_name) if class_name else module
    return getattr(target, name)


@pytest.mark.parametrize(
    "relative,name,module_name",
    [
        (
            "optimization/personal_ranker.py",
            "train_ranker",
            "optimization.personal_ranker",
        ),
        (
            "optimization/keeper_head.py",
            "train_keeper_head",
            "optimization.keeper_head",
        ),
    ],
)
def test_actual_manual_train_core_defers_before_any_fit(
    db, relative, name, module_name
):
    result = function(relative, name, module_name)(db_path="/must-not-open")
    assert result["deferred"] is True and result["written"] == 0 and "error" in result


def test_existing_keeper_cannot_evaluate_on_cpu_in_externalmode(db):
    fn = function(
        "optimization/keeper_head.py",
        "keeper_probs_for_group",
        "optimization.keeper_head",
    )
    assert fn(object(), SimpleNamespace(config_path=None), [object()]) is None


@pytest.mark.parametrize(
    "name",
    [
        "optimize_weights_direct",
        "optimize_weights_with_cv",
        "compute_weight_confidence",
    ],
)
def test_weight_cli_methods_have_no_validation_inference_fallback(db, name):
    fn = function(
        "optimization/weight_optimizer.py",
        name,
        "optimization.weight_optimizer",
        "WeightOptimizer",
    )
    assert fn(SimpleNamespace(config_path=None))["deferred"] is True


@pytest.mark.parametrize(
    "name", ["api_comparison_learned_weights", "api_comparison_confidence"]
)
def test_actual_compare_manual_json_contract_is_unavailable_not_servererror(db, name):
    fn = function("api/routers/comparison.py", name, "api.routers.comparison")
    kwargs = {"category": None}
    if name == "api_comparison_learned_weights":
        kwargs.update(include_ties="true", use_cv="false")
    result = fn(**kwargs)
    assert result["available"] is False and "external GPU learning" in result["message"]
    from api.models.comparison import LearnedWeightsResponse

    if name == "api_comparison_learned_weights":
        assert LearnedWeightsResponse.model_validate(result).available is False


@pytest.mark.parametrize(
    "name,path",
    [
        ("api_comparison_learned_weights", "/api/comparison/learned_weights"),
        ("api_comparison_confidence", "/api/comparison/confidence"),
    ],
)
def test_actual_compare_http_json_contract_external_no_cpu(db, name, path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import auth
    from api.models.comparison import LearnedWeightsResponse

    fn = function("api/routers/comparison.py", name, "api.routers.comparison")
    app = FastAPI()
    app.dependency_overrides[auth.require_edition] = lambda: SimpleNamespace()
    options = (
        {"response_model": LearnedWeightsResponse, "response_model_exclude_unset": True}
        if name == "api_comparison_learned_weights"
        else {}
    )
    app.add_api_route(path, fn, methods=["GET"], **options)
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 200 and response.json()["available"] is False
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
    assert real_backend() == "external" and seen == [
        "/synthetic/library/scoring_config.json"
    ]
