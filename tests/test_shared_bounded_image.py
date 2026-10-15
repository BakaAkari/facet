"""One shared LANCZOS-1024 image per photo (scan-pass performance, S1).

The pyiqa models and Q-ReAlign each resize the full-resolution decode to a 1024 px
long edge themselves.  ``ChunkedMultiPassProcessor._load_images`` now does that resize
once in the load worker and the passes hand the result to those consumers.  Sharing is
only correct for consumers that provably preprocess to the same pixels, so this file
pins four things:

* the shared image yields the SAME tensor / image the consumer would have built itself;
* ``aesthetic_v25`` and ``deqa`` keep the full-resolution image;
* a photo whose resize fails degrades to the full-resolution image;
* a plan that never reads the bounded image does not pay for it.
"""

from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from processing.multi_pass import ChunkedMultiPassProcessor  # noqa: E402
from utils.image_bounds import (  # noqa: E402
    MAX_INFERENCE_SIZE, bound_to_inference_size, liqe_input_too_small,
)


class _FakeModelManager:
    def detect_vram(self):
        return 8.0


def _processor(pass_groups=None, load_workers=1):
    # A real config: the composition analysis that now runs in the worker reads it.
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path
    scorer = mock.MagicMock()
    scorer.config = ScoringConfig(defaults_path(), validate=False)
    scorer.get_exif_data.return_value = {}
    with mock.patch("processing.multi_pass._ensure_imports"):
        proc = ChunkedMultiPassProcessor(
            scorer=scorer, model_manager=_FakeModelManager(),
            config={"processing": {"load_workers": load_workers}},
        )
    proc.pass_groups = pass_groups
    return proc


def _noise_jpeg(path, size, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)
    return str(path)


@pytest.fixture(scope="module")
def jpegs(tmp_path_factory):
    root = tmp_path_factory.mktemp("bounded")
    return [
        _noise_jpeg(root / "landscape.jpg", (4000, 3000), 1),
        _noise_jpeg(root / "portrait.jpg", (3000, 4000), 2),
        _noise_jpeg(root / "small.jpg", (1024, 700), 3),
    ]


def _load(proc, paths):
    with mock.patch("exiftool.get_exif_batch", return_value={}):
        images = proc._load_images(paths)
    assert len(images) == len(paths)
    return images


def _pyiqa_scorer():
    from models.pyiqa_scorer import PyIQAScorer
    scorer = PyIQAScorer.__new__(PyIQAScorer)
    scorer.device = "cpu"
    return scorer


def _qrealign_scorer():
    from models.qrealign_native import QReAlignNativeScorer
    return QReAlignNativeScorer.__new__(QReAlignNativeScorer)


class TestSharedImageEqualsWhatTheConsumerBuilds:
    def test_pyiqa_tensor_is_identical(self, jpegs):
        images = _load(_processor(), jpegs)
        scorer = _pyiqa_scorer()
        for path in jpegs:
            data = images[path]
            assert data["bounded"] is not None
            assert torch.equal(
                scorer._preprocess_image(data["bounded"]),
                scorer._preprocess_image(data["pil"]),
            ), path

    def test_qrealign_input_is_identical(self, jpegs):
        images = _load(_processor(), jpegs)
        q = _qrealign_scorer()
        for path in jpegs:
            data = images[path]
            assert np.array_equal(
                np.asarray(q._prepare(data["bounded"])), np.asarray(q._prepare(data["pil"])),
            ), path

    def test_the_bounded_image_is_the_inference_bound(self, jpegs):
        images = _load(_processor(), jpegs)
        sizes = {p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]: images[p]["bounded"].size for p in jpegs}
        assert sizes["landscape.jpg"] == (1024, 768)
        assert sizes["portrait.jpg"] == (768, 1024)
        assert sizes["small.jpg"] == (1024, 700)

    def test_a_photo_within_the_bound_shares_the_full_image_object(self, jpegs):
        small = jpegs[2]
        images = _load(_processor(), [small])
        assert images[small]["bounded"] is images[small]["pil"]

    def test_liqe_size_check_agrees_for_both_images(self, jpegs):
        images = _load(_processor(), jpegs)
        for path in jpegs:
            data = images[path]
            assert liqe_input_too_small(*data["bounded"].size) == liqe_input_too_small(*data["pil"].size)

    def test_worker_pool_produces_the_same_bounded_image(self, jpegs):
        sequential = _load(_processor(load_workers=1), jpegs)
        parallel = _load(_processor(load_workers=4), jpegs)
        for path in jpegs:
            assert np.array_equal(np.asarray(sequential[path]["bounded"]), np.asarray(parallel[path]["bounded"]))


class TestDegenerateAspect:
    def test_a_failed_resize_stores_none_and_the_pass_falls_back_to_pil(self, tmp_path):
        # 20000x1: int(1 * 0.0512) == 0, so the LANCZOS resize raises.
        path = _noise_jpeg(tmp_path / "strip.jpg", (20000, 1), 4)
        with pytest.raises(ValueError):
            bound_to_inference_size(Image.new("RGB", (20000, 1)))

        proc = _processor()
        scorer = mock.MagicMock()
        scorer.score_batch.side_effect = lambda imgs: [5.0 for _ in imgs]
        with mock.patch("exiftool.get_exif_batch", return_value={}):
            images = proc._load_images([path])
        assert len(images) == 1
        assert images[path]["bounded"] is None

        results = {path: {}}
        proc._pass_pyiqa(scorer, "topiq", images, results)
        handed = scorer.score_batch.call_args[0][0]
        assert handed[0] is images[path]["pil"]


class _Recorder:
    """Stub scorer that remembers the objects it was handed."""

    def __init__(self):
        self.batch = None
        self.single = []

    def score_batch(self, pil_imgs):
        self.batch = list(pil_imgs)
        return [5.0] * len(pil_imgs)

    def score_tasks(self, pil, tasks):
        self.single.append(pil)
        return {task: 5.0 for task in tasks}

    def score(self, pil, task="aesthetic"):
        self.single.append(pil)
        return 5.0


def _entry():
    return {"pil": object(), "bounded": object()}


class TestPassesHandTheRightImage:
    @pytest.mark.parametrize("model", ["topiq", "topiq_iaa", "topiq_nr_face", "liqe"])
    def test_pyiqa_models_get_the_bounded_image(self, model):
        proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
        images = {"a": _entry(), "b": _entry()}
        stub = _Recorder()
        proc._pass_pyiqa(stub, model, images, {"a": {}, "b": {}})
        assert [x is images[p]["bounded"] for x, p in zip(stub.batch, "ab")] == [True, True]

    @pytest.mark.parametrize("model", ["aesthetic_v25", "deqa"])
    def test_extended_models_keep_the_full_resolution_image(self, model):
        proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
        images = {"a": _entry(), "b": _entry()}
        stub = _Recorder()
        proc._pass_pyiqa(stub, model, images, {"a": {}, "b": {}})
        assert [x is images[p]["pil"] for x, p in zip(stub.batch, "ab")] == [True, True]

    def test_a_none_bounded_image_falls_back_to_pil(self):
        proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
        images = {"a": {"pil": object(), "bounded": None}, "b": _entry()}
        stub = _Recorder()
        proc._pass_pyiqa(stub, "topiq", images, {"a": {}, "b": {}})
        assert stub.batch[0] is images["a"]["pil"]
        assert stub.batch[1] is images["b"]["bounded"]

    @pytest.mark.parametrize("model", ["qrealign", "qrealign_quality"])
    def test_qrealign_gets_the_bounded_image(self, model):
        proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
        proc._qrealign_is_primary = lambda: False
        images = {"a": _entry()}
        stub = _Recorder()
        proc._pass_qrealign(stub, model, images, {"a": {}})
        assert stub.single == [images["a"]["bounded"]]

    def test_qrealign_falls_back_to_pil_when_bounded_is_none(self):
        proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
        proc._qrealign_is_primary = lambda: False
        images = {"a": {"pil": object(), "bounded": None}}
        stub = _Recorder()
        proc._pass_qrealign(stub, "qrealign", images, {"a": {}})
        assert stub.single == [images["a"]["pil"]]

    def test_the_full_resolution_set_is_named_once(self):
        assert set(ChunkedMultiPassProcessor.FULL_RES_IQA_MODELS) == {"aesthetic_v25", "deqa"}
        assert set(ChunkedMultiPassProcessor.FULL_RES_IQA_MODELS) <= set(ChunkedMultiPassProcessor.PYIQA_MODELS)


class TestOnlyComputedWhenThePlanNeedsIt:
    @pytest.mark.parametrize("groups, expected", [
        (None, True),
        ([["clip", "topiq"]], True),
        ([["clip"], ["liqe"]], True),
        ([["qrealign"]], True),
        ([["qrealign_quality"]], True),
        ([["saliency"]], False),
        ([["clip", "samp_net"], ["insightface"]], False),
        ([["aesthetic_v25"], ["deqa"]], False),
        ([["aesthetic_v25"], ["topiq_nr_face"]], True),
    ])
    def test_needs_bounded_image(self, groups, expected):
        assert _processor(pass_groups=groups)._needs_bounded_image() is expected

    def test_a_saliency_only_plan_stores_no_resize(self, jpegs):
        proc = _processor(pass_groups=[["saliency"]])
        with mock.patch("processing.multi_pass.bound_to_inference_size") as bound:
            images = _load(proc, jpegs[:1])
        bound.assert_not_called()
        assert images[jpegs[0]]["bounded"] is None

    def test_the_inference_bound_is_the_shared_constant(self):
        assert MAX_INFERENCE_SIZE == 1024
