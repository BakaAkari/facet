"""A NULL face_ratio (failed face pass) must categorise like face_ratio 0.

Item 9 stores face_count / face_ratio / is_group_portrait as NULL after a face-pass
failure. ``CategoryFilter`` deliberately fails a ``*_max`` filter on NULL, so every
reader that re-derives a category from a stored row must coalesce through
``config.category_filter.with_face_defaults`` exactly as the scan does.
"""

from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from config import ScoringConfig
from config.category_filter import with_face_defaults
from optimization.weight_optimizer import WeightOptimizer

from tests.test_stats import (
    _create_app_no_auth,
    _make_stats_db,
    _photo,
    _stats_mocks,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "scoring_config.default.json"

# Tags whose default-config category carries a face_ratio_max filter.
_TAGS = "landscape, mountain"


def _cfg():
    return ScoringConfig(str(_DEFAULT_CONFIG), validate=False)


def test_premise_null_changes_category_without_the_helper():
    cfg = _cfg()
    base = {"tags": _TAGS, "face_count": 0, "face_ratio": 0, "is_group_portrait": 0}
    with_zero = cfg.determine_category(base)
    raw_null = cfg.determine_category({**base, "face_count": None, "face_ratio": None,
                                      "is_group_portrait": None})
    assert with_zero != raw_null, "premise: NULL must diverge on the raw row"
    assert cfg.determine_category(with_face_defaults(
        {**base, "face_count": None, "face_ratio": None, "is_group_portrait": None}
    )) == with_zero


def test_helper_does_not_mutate_and_keeps_real_values():
    row = {"face_ratio": None, "face_count": 2, "is_group_portrait": None}
    out = with_face_defaults(row)
    assert row["face_ratio"] is None
    assert out == {"face_ratio": 0, "face_count": 2, "is_group_portrait": 0}


def test_weight_optimizer_categorises_null_like_zero():
    opt = WeightOptimizer(":memory:")
    opt._cfg = _cfg()
    seen = []

    def fake_vector(row, cfg, cat, penalties=None, unavailable=()):
        seen.append(cat)
        return {}

    base = {"tags": _TAGS, "face_count": 0, "face_ratio": 0, "is_group_portrait": 0}
    nulls = {**base, "face_count": None, "face_ratio": None, "is_group_portrait": None}
    with mock.patch("optimization.weight_optimizer.build_metric_vector", fake_vector):
        opt._metric_vector(base, None)
        opt._metric_vector(nulls, None)
    assert seen[0] == seen[1]
    assert seen[0] != "default"


def _overlap(tmp_path, name, **face):
    db_path = str(tmp_path / f"{name}.db")
    _make_stats_db(db_path, [_photo("/a.jpg", category="landscape", tags=_TAGS, **face)])
    app = _create_app_no_auth()
    with _stats_mocks(db_path):
        resp = TestClient(app).get("/api/stats/categories/overlap")
    assert resp.status_code == 200
    return {c["name"]: c["matched"] for c in resp.json()["per_category"]}


def test_stats_overlap_categorises_null_like_zero(tmp_path):
    zero = _overlap(tmp_path, "zero", face_ratio=0.0, face_count=0, is_group_portrait=0)
    null = _overlap(tmp_path, "null", face_ratio=None, face_count=None, is_group_portrait=None)
    assert null == zero
    assert zero.get("landscape") == 1
