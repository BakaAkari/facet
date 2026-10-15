"""Rule-based composition runs in the load workers (scan-pass performance, S1).

``get_placement_data`` and ``detect_leading_lines`` are pure functions of the decoded
pixels plus the config (the face bbox is always ``None`` where they used to run), so
``_load_images`` now computes them inside the worker pool and ``_compute_aggregates``
only reads the result.  This file pins:

* the hoisted values equal a verbatim copy of the old inline logic, in both branches
  (SAMP-Net supplied ``comp_score`` / it did not);
* a restricted (``--pass ...``) processor does no composition work at all;
* a composition error fails ONE photo, recorded under stage ``composition``, and the
  rest of the chunk loads (it used to abort the whole scan).
"""

from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from analyzers import CompositionAnalyzer  # noqa: E402
from processing.multi_pass import ChunkedMultiPassProcessor  # noqa: E402


class _FakeModelManager:
    def detect_vram(self):
        return 8.0


class _SpyScorer:
    def __init__(self, db_path, config):
        self.db_path = db_path
        self.config = config
        self.calls = []

    def calculate_aggregate_logic(self, metrics):
        self.calls.append(metrics)
        return 7.5, "spy"

    def get_exif_data(self, path):
        return {}


@pytest.fixture(scope="module")
def real_config():
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path
    return ScoringConfig(defaults_path(), validate=False)


@pytest.fixture()
def scorer(tmp_path, real_config):
    from db.schema import init_database
    db_path = str(tmp_path / "chunk.db")
    init_database(db_path)
    return _SpyScorer(db_path, real_config)


def _processor(scorer, restricted=False, load_workers=1, on_error=None):
    with mock.patch("processing.multi_pass._ensure_imports"):
        proc = ChunkedMultiPassProcessor(
            scorer=scorer, model_manager=_FakeModelManager(),
            config={"processing": {"load_workers": load_workers}}, on_error=on_error,
        )
    proc.restricted = restricted
    return proc


def _structured_image(seed, size=(320, 240)):
    """Noise plus a bright rectangle and diagonal lines: a subject region and Hough lines exist."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 60, (size[1], size[0], 3), dtype=np.uint8)
    cv2.rectangle(arr, (size[0] // 4, size[1] // 3), (size[0] // 2, size[1] * 3 // 4), (230, 200, 40), -1)
    cv2.line(arr, (0, size[1] - 1), (size[0] - 1, 0), (255, 255, 255), 3)
    cv2.line(arr, (10, 0), (size[0] - 20, size[1] - 1), (255, 255, 255), 3)
    return arr


def _save(tmp_path, name, arr):
    path = tmp_path / name
    Image.fromarray(arr).save(path, quality=95)
    return str(path)


def _images(tmp_path, count=3):
    arrays = [_structured_image(seed) for seed in range(count)]
    if count > 1:
        arrays[1] = _structured_image(1, size=(300, 220))  # distinct size: tests single this one out
    arrays.append(np.random.default_rng(99).integers(0, 256, (240, 320, 3), dtype=np.uint8))
    return [_save(tmp_path, f"img{i}.jpg", a) for i, a in enumerate(arrays)]


def _load(proc, paths):
    with mock.patch("exiftool.get_exif_batch", return_value={}):
        return proc._load_images(paths)


def _reference_inline(img_data, data, config):
    """Verbatim copy of the pre-hoist logic in ``_compute_aggregates``."""
    img_cv = img_data["cv"]
    img_h, img_w = img_data["height"], img_data["width"]
    if "comp_score" not in data or data.get("comp_score") is None:
        face_bbox = None
        comp_data = CompositionAnalyzer.get_placement_data(face_bbox, img_w, img_h, config, img_cv)
        data["comp_score"] = comp_data["score"]
        data["power_point_score"] = comp_data.get("power_point_score", 5.0)
        cache = img_data.get("cache")
        leading_lines_data = CompositionAnalyzer.detect_leading_lines(img_cv, cache=cache)
        data["leading_lines_score"] = leading_lines_data.get("leading_lines_score", 0)
    else:
        if "power_point_score" not in data:
            comp_data = CompositionAnalyzer.get_placement_data(None, img_w, img_h, config, img_cv)
            data["power_point_score"] = comp_data.get("power_point_score", 5.0)
        if "leading_lines_score" not in data:
            cache = img_data.get("cache")
            leading_lines_data = CompositionAnalyzer.detect_leading_lines(img_cv, cache=cache)
            data["leading_lines_score"] = leading_lines_data.get("leading_lines_score", 0)


class TestHoistedValuesEqualTheOldInlineLogic:
    def test_load_stores_the_three_composition_values(self, tmp_path, scorer):
        paths = _images(tmp_path)
        images = _load(_processor(scorer), paths)
        assert len(images) == len(paths)
        for data in images.values():
            assert set(data["composition"]) == {"score", "power_point_score", "leading_lines_score"}

    @pytest.mark.parametrize("samp_net_comp_score", [None, 6.25])
    def test_compute_aggregates_matches_the_reference(self, tmp_path, scorer, real_config, samp_net_comp_score):
        paths = _images(tmp_path)
        proc = _processor(scorer)
        images = _load(proc, paths)
        assert len(images) == len(paths)

        results, expected = {}, {}
        for path in paths:
            seed = {"aesthetic": 5.0}  # an empty result row is skipped by _compute_aggregates
            if samp_net_comp_score is not None:
                seed["comp_score"] = samp_net_comp_score
            results[path] = dict(seed)
            expected[path] = dict(seed)
            _reference_inline(images[path], expected[path], real_config)

        proc._compute_aggregates(results, images)

        for path in paths:
            for key in ("comp_score", "power_point_score", "leading_lines_score"):
                assert type(results[path][key]) is type(expected[path][key]), (path, key)
                assert results[path][key] == expected[path][key], (path, key)

    def test_preset_values_are_never_overwritten_and_composition_is_not_read(self, tmp_path, scorer):
        # SAMP-Net supplied comp_score and a previous pass supplied the other two: the
        # hoisted dict need not exist (fixtures preset all three values).
        path = str((tmp_path / "a.jpg").resolve())
        proc = _processor(scorer)
        results = {path: {"comp_score": 4.0, "power_point_score": 3.0, "leading_lines_score": 2.0}}
        images = {path: {"height": 10, "width": 10, "sharpness": {}, "color": {}, "histogram": {},
                         "mono": {}, "noise": {}, "contrast": {}, "form": {}, "exif": {}}}
        proc._compute_aggregates(results, images)
        assert (results[path]["comp_score"], results[path]["power_point_score"],
                results[path]["leading_lines_score"]) == (4.0, 3.0, 2.0)

    def test_samp_net_score_is_kept_and_only_the_missing_values_are_filled(self, tmp_path, scorer):
        path = str((tmp_path / "a.jpg").resolve())
        proc = _processor(scorer)
        results = {path: {"comp_score": 4.0, "power_point_score": 3.0}}
        images = {path: {"height": 10, "width": 10, "sharpness": {}, "color": {}, "histogram": {},
                         "mono": {}, "noise": {}, "contrast": {}, "form": {}, "exif": {},
                         "composition": {"score": 9.0, "power_point_score": 8.0,
                                         "leading_lines_score": 7.0}}}
        proc._compute_aggregates(results, images)
        assert results[path]["comp_score"] == 4.0
        assert results[path]["power_point_score"] == 3.0
        assert results[path]["leading_lines_score"] == 7.0

    def test_missing_comp_score_takes_all_three_from_the_hoisted_dict(self, tmp_path, scorer):
        path = str((tmp_path / "a.jpg").resolve())
        proc = _processor(scorer)
        results = {path: {"aesthetic": 5.0}}
        images = {path: {"height": 10, "width": 10, "sharpness": {}, "color": {}, "histogram": {},
                         "mono": {}, "noise": {}, "contrast": {}, "form": {}, "exif": {},
                         "composition": {"score": 9.0, "power_point_score": 8.0,
                                         "leading_lines_score": 7.0}}}
        proc._compute_aggregates(results, images)
        assert (results[path]["comp_score"], results[path]["power_point_score"],
                results[path]["leading_lines_score"]) == (9.0, 8.0, 7.0)

    def test_compute_aggregates_no_longer_reads_the_cv_array(self, tmp_path, scorer):
        path = str((tmp_path / "a.jpg").resolve())
        proc = _processor(scorer)
        results = {path: {"comp_score": 4.0, "power_point_score": 3.0, "leading_lines_score": 2.0}}
        images = {path: {"height": 10, "width": 10, "sharpness": {}, "color": {}, "histogram": {},
                         "mono": {}, "noise": {}, "contrast": {}, "form": {}, "exif": {}}}
        assert "cv" not in images[path]
        proc._compute_aggregates(results, images)  # KeyError('cv') before the hoist
        assert len(scorer.calls) == 1


class TestRestrictedProcessorDoesNoCompositionWork:
    def test_no_composition_call_and_no_key(self, tmp_path, scorer):
        paths = _images(tmp_path, count=2)
        proc = _processor(scorer, restricted=True)
        with mock.patch.object(CompositionAnalyzer, "get_placement_data") as placement, \
                mock.patch.object(CompositionAnalyzer, "detect_leading_lines") as lines:
            images = _load(proc, paths)
        assert len(images) == len(paths)
        placement.assert_not_called()
        lines.assert_not_called()
        assert all(data["composition"] is None for data in images.values())

    def test_a_full_processor_calls_each_function_once_per_photo(self, tmp_path, scorer):
        paths = _images(tmp_path, count=2)
        proc = _processor(scorer)
        with mock.patch.object(CompositionAnalyzer, "get_placement_data",
                               wraps=CompositionAnalyzer.get_placement_data) as placement, \
                mock.patch.object(CompositionAnalyzer, "detect_leading_lines",
                                  wraps=CompositionAnalyzer.detect_leading_lines) as lines:
            _load(proc, paths)
        assert placement.call_count == len(paths)
        assert lines.call_count == len(paths)


class TestCompositionFailureFailsOnePhoto:
    @pytest.mark.parametrize("load_workers", [1, 4])
    @pytest.mark.parametrize("target", ["get_placement_data", "detect_leading_lines"])
    def test_error_is_recorded_under_stage_composition(self, tmp_path, scorer, load_workers, target):
        paths = _images(tmp_path, count=3)
        bad = paths[1]
        errors = []
        proc = _processor(scorer, load_workers=load_workers,
                          on_error=lambda path, stage, err: errors.append((path, stage, err)))
        real = getattr(CompositionAnalyzer, target)

        def flaky(*args, **kwargs):
            img = args[-1] if target == "get_placement_data" else args[0]
            if img.shape[:2] == (220, 300):  # only the odd-sized photo fails
                raise ValueError("composition boom")
            return real(*args, **kwargs)

        with mock.patch.object(CompositionAnalyzer, target, side_effect=flaky):
            images = _load(proc, paths)

        assert bad not in images
        assert set(images) == set(paths) - {bad}
        assert [(p, s) for p, s, _ in errors] == [(bad, "composition")]
        assert "composition boom" in str(errors[0][2])

    def test_the_stage_is_not_load(self, tmp_path, scorer):
        paths = _images(tmp_path, count=2)
        errors = []
        proc = _processor(scorer, on_error=lambda p, s, e: errors.append(s))
        with mock.patch.object(CompositionAnalyzer, "get_placement_data", side_effect=RuntimeError("x")):
            # RuntimeError is the decode-abandonment signal for the loaders; a composition
            # RuntimeError must still be a per-photo failure, not a scan abort.
            images = _load(proc, paths)
        assert images == {}
        assert errors == ["composition"] * len(paths)
