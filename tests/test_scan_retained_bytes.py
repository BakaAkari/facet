"""A multi-pass chunk no longer retains the BGR array, the gray plane or the mask list (S9, S8b).

``_load_images`` used to keep, per photo, the PIL image (4 B/px), the BGR array (3 B/px) and the
gray plane of its ``ImageCache`` (1 B/px) for the whole chunk, and the saliency pass held every
mask of the chunk at once.  Both planes are pure functions of the retained PIL image (the BGR is
``pil_to_bgr(pil)``, the gray ``bgr_to_gray`` of that), so now:

* ``_load_images`` returns neither ``'cv'`` nor ``cache.gray``;
* the face pass rebuilds ONE BGR array per photo and drops it before the next;
* the saliency pass hands the scorer zero-argument callables that rebuild the gray plane for one
  photo at a time;
* SAMP-Net scores the PIL image (its preprocess gives the identical tensor);
* ``ChunkedMultiPassProcessor.RETAINED_BYTES_PER_PX`` is a measured upper bound of what is left.

Every rebuild must give exactly the arrays the load path used to retain, and no pass may mutate the
PIL image the rebuilds depend on.
"""
import hashlib
import time
import weakref
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from analyzers import ImageCache  # noqa: E402
from processing import multi_pass  # noqa: E402
from processing.multi_pass import (  # noqa: E402
    FACE_FAILED_KEY, SALIENCY_FAILED_KEY, ChunkedMultiPassProcessor,
)
from utils.image_loading import bgr_to_gray, load_image_from_path, pil_to_bgr  # noqa: E402

SIZES = [(1, 1), (7, 5), (64, 48), (333, 211), (48, 640), (1024, 700)]


def _rgb(size, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)


def _reference_bgr(pil):
    """The expression the load path used before the helpers existed, spelled out."""
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _reference_gray(pil):
    return cv2.cvtColor(_reference_bgr(pil), cv2.COLOR_BGR2GRAY)


class _Manager:
    device = "cpu"

    def detect_vram(self):
        return 8.0


class _Scorer:
    def __init__(self, config, db_path=None):
        self.config = config
        self.db_path = db_path

    def get_exif_data(self, path):
        return {}


@pytest.fixture(scope="module")
def real_config():
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path
    return ScoringConfig(defaults_path(), validate=False)


def _processor(real_config, load_workers=1, errors=None):
    with mock.patch("processing.multi_pass._ensure_imports"):
        proc = ChunkedMultiPassProcessor(
            scorer=_Scorer(real_config), model_manager=_Manager(),
            config={"processing": {"load_workers": load_workers}},
            on_error=(lambda *a: errors.append(a)) if errors is not None else None,
        )
    return proc


def _entry(pil, **extra):
    """What the passes read from one ``_load_images`` entry (no cv, no gray)."""
    data = {
        "pil": pil, "bounded": None, "path": "/p.jpg", "width": pil.width, "height": pil.height,
        "cache": SimpleNamespace(laplacian_variance=40.0, gray=None, hsv=None),
    }
    data.update(extra)
    return data


def _load(proc, paths):
    with mock.patch("exiftool.get_exif_batch", return_value={}):
        return proc._load_images(paths)


def _save(tmp_path, name, arr, fmt="PNG"):
    path = tmp_path / name
    Image.fromarray(arr).save(path, **({"quality": 95} if fmt == "JPEG" else {}))
    return str(path)


def _structured(seed, size=(320, 240)):
    arr = np.random.default_rng(seed).integers(0, 60, (size[1], size[0], 3), dtype=np.uint8)
    cv2.rectangle(arr, (size[0] // 4, size[1] // 3), (size[0] // 2, size[1] * 3 // 4), (230, 200, 40), -1)
    cv2.line(arr, (0, size[1] - 1), (size[0] - 1, 0), (255, 255, 255), 3)
    return arr


# ---------------------------------------------------------------------------
# The rebuild helpers equal what the load path used to retain
# ---------------------------------------------------------------------------

class TestRebuildsEqualTheRetainedArrays:
    @pytest.mark.parametrize("size", SIZES)
    def test_pil_to_bgr_equals_the_array_the_loader_returns(self, tmp_path, size):
        path = _save(tmp_path, "a.png", _rgb(size, seed=size[0]))
        pil, img_cv = load_image_from_path(path)
        rebuilt = pil_to_bgr(pil)
        assert rebuilt.dtype == img_cv.dtype and rebuilt.shape == img_cv.shape
        assert np.array_equal(rebuilt, img_cv)
        assert np.array_equal(rebuilt, _reference_bgr(pil))

    @pytest.mark.parametrize("size", SIZES)
    def test_bgr_to_gray_equals_the_image_cache_plane(self, tmp_path, size):
        path = _save(tmp_path, "a.png", _rgb(size, seed=size[1]))
        pil, img_cv = load_image_from_path(path)
        gray = bgr_to_gray(pil_to_bgr(pil))
        cache_gray = ImageCache(img_cv).gray
        assert gray.dtype == cache_gray.dtype and gray.shape == cache_gray.shape
        assert np.array_equal(gray, cache_gray)

    def test_gray_is_not_a_direct_rgb2gray_of_the_pil(self):
        # The exact chain matters: RGB2GRAY of the RGB array rounds differently on some pixels.
        pil = Image.fromarray(_rgb((333, 211), seed=3))
        assert np.array_equal(bgr_to_gray(pil_to_bgr(pil)), _reference_gray(pil))


# ---------------------------------------------------------------------------
# _load_images retains neither plane
# ---------------------------------------------------------------------------

class TestLoadImagesRetainsNoPlanes:
    def _paths(self, tmp_path):
        sizes = [(320, 240), (300, 220), (64, 48)]
        return [_save(tmp_path, f"img{i}.jpg", _structured(i, s), fmt="JPEG") for i, s in enumerate(sizes)]

    def test_entries_have_no_cv_and_no_gray(self, tmp_path, real_config):
        paths = self._paths(tmp_path)
        images = _load(_processor(real_config), paths)
        assert len(images) == len(paths)  # a silent per-photo load failure must not pass
        for entry in images.values():
            assert "cv" not in entry
            assert entry["cache"].gray is None
            assert entry["cache"].hsv is None

    def test_the_cache_keeps_the_laplacian_variance_the_face_pass_reads(self, tmp_path, real_config):
        paths = self._paths(tmp_path)
        images = _load(_processor(real_config, load_workers=2), paths)
        assert len(images) == len(paths)
        for entry in images.values():
            variance = entry["cache"].laplacian_variance
            assert isinstance(variance, float) and variance > 0
            assert variance == ImageCache(pil_to_bgr(entry["pil"])).laplacian_variance

    def test_the_hoisted_composition_still_sees_the_gray_plane(self, tmp_path, real_config):
        # Dropping the gray plane must come AFTER the composition hoist that reads it.
        from analyzers import CompositionAnalyzer
        paths = self._paths(tmp_path)
        images = _load(_processor(real_config), paths)
        assert len(images) == len(paths)
        for entry in images.values():
            img_cv = pil_to_bgr(entry["pil"])
            cache = ImageCache(img_cv)
            lines = CompositionAnalyzer.detect_leading_lines(img_cv, cache=cache)
            assert entry["composition"]["leading_lines_score"] == lines.get("leading_lines_score", 0)
            placement = CompositionAnalyzer.get_placement_data(
                None, entry["width"], entry["height"], real_config, img_cv)
            assert entry["composition"]["score"] == placement["score"]


# ---------------------------------------------------------------------------
# Face pass: one BGR rebuild at a time
# ---------------------------------------------------------------------------

class _Result(dict):
    pass


def _face_result():
    return {
        "face_count": 1, "face_quality": 7.5, "eye_sharpness": 120.0, "face_sharpness": 200.0,
        "is_blink": False, "is_group_portrait": False, "max_face_confidence": 0.9,
        "raw_eye_sharpness": 130.0, "face_details": [{}], "face_area": 400,
    }


class _RecordingAnalyzer:
    def __init__(self):
        self.copies = []

    def analyze_faces(self, img_cv):
        self.copies.append(img_cv.copy())
        return _face_result()


class TestFacePassRebuildsBgr:
    def _images(self, count=3):
        sizes = [(40, 30), (31, 47), (64, 64), (5, 9)]
        pils = [Image.fromarray(_rgb(sizes[i % len(sizes)], seed=i)) for i in range(count)]
        return {f"/p{i}.jpg": _entry(p, path=f"/p{i}.jpg") for i, p in enumerate(pils)}

    def test_the_analyzer_sees_the_array_the_loader_used_to_retain(self, real_config):
        proc = _processor(real_config)
        images = self._images()
        results = {p: {} for p in images}
        analyzer = _RecordingAnalyzer()
        proc._pass_insightface(analyzer, images, results)
        assert len(analyzer.copies) == len(images)
        for seen, entry in zip(analyzer.copies, images.values()):
            assert seen.dtype == np.uint8 and seen.shape == (entry["height"], entry["width"], 3)
            assert np.array_equal(seen, _reference_bgr(entry["pil"]))
        assert all(r["face_count"] == 1 and FACE_FAILED_KEY not in r for r in results.values())

    def test_face_ratio_uses_the_stored_dimensions(self, real_config):
        proc = _processor(real_config)
        images = self._images(1)
        path = next(iter(images))
        results = {path: {}}
        proc._pass_insightface(_RecordingAnalyzer(), images, results)
        entry = images[path]
        assert results[path]["face_ratio"] == 400 / (entry["height"] * entry["width"])

    def test_no_second_bgr_copy_is_alive_when_the_next_one_is_built(self, real_config, monkeypatch):
        proc = _processor(real_config)
        images = self._images(4)
        refs, alive_at_build = [], []
        real = multi_pass.pil_to_bgr

        def spy(pil):
            alive_at_build.append(sum(1 for r in refs if r() is not None))
            arr = real(pil)
            refs.append(weakref.ref(arr))
            return arr

        monkeypatch.setattr(multi_pass, "pil_to_bgr", spy)
        proc._pass_insightface(_RecordingAnalyzer(), images, {p: {} for p in images})
        assert len(alive_at_build) == len(images)
        assert alive_at_build == [0] * len(images), (
            "a previous photo's BGR array was still alive while the next one was rebuilt")
        assert sum(1 for r in refs if r() is not None) == 0, "the last BGR array outlived the pass"

    def test_the_array_is_dropped_even_when_the_analyzer_raises(self, real_config, monkeypatch, caplog):
        # The pass logs the exception object, and pytest's capture handler keeps the record
        # (and so the traceback and the analyzer frame's array); silence that logger.
        caplog.set_level("CRITICAL", logger="facet.multi_pass")
        proc = _processor(real_config)
        images = self._images(3)
        refs = []
        real = multi_pass.pil_to_bgr

        def spy(pil):
            arr = real(pil)
            refs.append(weakref.ref(arr))
            return arr

        class Boom:
            def analyze_faces(self, img_cv):
                raise ValueError("detector crashed")

        monkeypatch.setattr(multi_pass, "pil_to_bgr", spy)
        results = {p: {} for p in images}
        proc._pass_insightface(Boom(), images, results)
        assert all(r[FACE_FAILED_KEY] for r in results.values())
        assert len(refs) == 3 and all(r() is None for r in refs)

    def test_a_rebuild_failure_is_a_face_failure(self, real_config, monkeypatch):
        proc = _processor(real_config)
        images = self._images(2)
        results = {p: {} for p in images}

        def broken(pil):
            raise MemoryError("no room for a BGR plane")

        monkeypatch.setattr(multi_pass, "pil_to_bgr", broken)
        analyzer = _RecordingAnalyzer()
        proc._pass_insightface(analyzer, images, results)
        assert analyzer.copies == []
        for res in results.values():
            assert res[FACE_FAILED_KEY] is True
            assert res["face_count"] is None and res["face_ratio"] is None
            assert res["face_details"] is None and res["isolation_bonus"] == 1.0


# ---------------------------------------------------------------------------
# Saliency pass: callables, one gray plane at a time, one sub-batch of masks
# ---------------------------------------------------------------------------

class _CallableRecorder:
    def __init__(self):
        self.pil = None
        self.cv = None

    def score_batch(self, pil_images, cv_images):
        self.pil, self.cv = list(pil_images), list(cv_images)
        return [{"subject_sharpness": 5.0, "subject_prominence": 0.1, "subject_placement": 5.0,
                 "bg_separation": 5.0, "subject_bbox": None} for _ in pil_images]


class TestSaliencyPassPassesRebuilders:
    def _images(self, count=3):
        sizes = [(40, 30), (31, 47), (64, 64)]
        return {f"/p{i}.jpg": _entry(Image.fromarray(_rgb(sizes[i % 3], seed=i + 10)))
                for i in range(count)}

    def test_each_cv_element_is_a_zero_arg_callable_returning_the_old_gray_plane(self, real_config):
        proc = _processor(real_config)
        images = self._images()
        scorer = _CallableRecorder()
        results = {p: {} for p in images}
        proc._pass_saliency(scorer, images, results)
        assert scorer.pil == [e["pil"] for e in images.values()]
        assert len(scorer.cv) == len(images)
        for rebuild, entry in zip(scorer.cv, images.values()):
            assert callable(rebuild) and not isinstance(rebuild, np.ndarray)
            gray = rebuild()
            assert gray.ndim == 2 and gray.dtype == np.uint8
            assert np.array_equal(gray, _reference_gray(entry["pil"]))
        assert all(SALIENCY_FAILED_KEY not in r for r in results.values())

    def test_entries_gain_no_plane(self, real_config):
        proc = _processor(real_config)
        images = self._images(2)
        proc._pass_saliency(_CallableRecorder(), images, {p: {} for p in images})
        for entry in images.values():
            assert "cv" not in entry and entry["cache"].gray is None

    def test_real_scorer_gives_the_same_values_as_the_old_bgr_route(self, real_config):
        from models.saliency_scorer import SaliencyScorer
        scorer = SaliencyScorer(device="cpu")
        scorer._loaded = True
        scorer._affordable_batch_size = lambda requested: 2

        def mask_for(w, h):
            m = np.zeros((h, w), np.uint8)
            m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
            return m

        scorer._run_chunk = lambda chunk, sizes: [mask_for(w, h) for (w, h) in sizes]
        proc = _processor(real_config)
        pils = [Image.fromarray(_structured(i, (96 + 8 * i, 80 + 6 * i))) for i in range(5)]
        images = {f"/p{i}.jpg": _entry(p) for i, p in enumerate(pils)}
        results = {p: {} for p in images}
        proc._pass_saliency(scorer, images, results)
        for path, pil in zip(images, pils):
            expected = scorer._score_from_mask(mask_for(pil.width, pil.height), _reference_bgr(pil))
            assert SALIENCY_FAILED_KEY not in results[path]
            for key in ("subject_sharpness", "subject_prominence", "subject_placement", "bg_separation"):
                assert results[path][key] == expected[key], (path, key)
            assert results[path]["subject_bbox"] == expected["subject_bbox"]

    def test_at_most_one_gray_plane_and_one_sub_batch_of_masks_are_alive(self, real_config, monkeypatch):
        from models.saliency_scorer import SaliencyScorer
        scorer = SaliencyScorer(device="cpu")
        scorer._loaded = True
        sub_batch = 2
        scorer._affordable_batch_size = lambda requested: sub_batch
        live = {"masks": 0, "max_masks": 0}

        class Mask(np.ndarray):
            pass

        def release():
            live["masks"] -= 1

        def forward(chunk, sizes):
            out = []
            for (w, h) in sizes:
                m = np.zeros((h, w), np.uint8)
                m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
                t = m.view(Mask)
                live["masks"] += 1
                live["max_masks"] = max(live["max_masks"], live["masks"])
                weakref.finalize(t, release)
                out.append(t)
            return out

        scorer._run_chunk = forward
        grays, bgrs = [], []
        real_gray, real_bgr = multi_pass.bgr_to_gray, multi_pass.pil_to_bgr
        peak = {"gray": 0, "bgr": 0}

        def gray_spy(arr):
            peak["gray"] = max(peak["gray"], sum(1 for r in grays if r() is not None))
            out = real_gray(arr)
            grays.append(weakref.ref(out))
            return out

        def bgr_spy(pil):
            peak["bgr"] = max(peak["bgr"], sum(1 for r in bgrs if r() is not None))
            out = real_bgr(pil)
            bgrs.append(weakref.ref(out))
            return out

        monkeypatch.setattr(multi_pass, "bgr_to_gray", gray_spy)
        monkeypatch.setattr(multi_pass, "pil_to_bgr", bgr_spy)
        proc = _processor(real_config)
        images = {f"/p{i}.jpg": _entry(Image.fromarray(_rgb((64, 48), seed=i))) for i in range(7)}
        results = {p: {} for p in images}
        proc._pass_saliency(scorer, images, results)
        assert len(grays) == 7 and len(bgrs) == 7
        assert live["max_masks"] <= sub_batch
        assert peak["gray"] == 0, "a gray plane was still alive when the next one was built"
        assert peak["bgr"] == 0, "a BGR plane was still alive when the next one was built"
        assert not any(SALIENCY_FAILED_KEY in r for r in results.values())


# ---------------------------------------------------------------------------
# S8b: SAMP-Net scores the PIL image
# ---------------------------------------------------------------------------

def _preprocessing_scorer():
    """A SAMPNetScorer without weights: only ``transform``/``device``, which ``preprocess`` reads."""
    from torchvision import transforms
    from models.samp_net import SAMPNetScorer
    scorer = SAMPNetScorer.__new__(SAMPNetScorer)
    scorer.device = "cpu"
    scorer.transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return scorer


class _SampStub:
    """Records the tensor the real SAMP-Net preprocess builds from whatever the pass hands over."""

    def __init__(self):
        self.real = _preprocessing_scorer()
        self.tensors = []

    def score(self, image):
        self.tensors.append(self.real.preprocess(image))
        return {"comp_score": 6.5, "pattern": "rule_of_thirds"}


class TestSampNetScoresThePilImage:
    @pytest.mark.parametrize("size", [(64, 48), (333, 211), (48, 640)])
    def test_pil_and_bgr_routes_give_the_identical_tensor(self, size):
        scorer = _preprocessing_scorer()
        pil = Image.fromarray(_rgb(size, seed=size[0]))
        assert torch.equal(scorer.preprocess(pil), scorer.preprocess(pil_to_bgr(pil)))

    def test_the_pass_input_gives_the_tensor_the_bgr_route_gave(self, real_config):
        proc = _processor(real_config)
        pils = [Image.fromarray(_rgb(s, seed=i)) for i, s in enumerate([(64, 48), (333, 211), (48, 640)])]
        images = {f"/p{i}.jpg": _entry(p) for i, p in enumerate(pils)}
        results = {p: {} for p in images}
        stub = _SampStub()
        proc._pass_samp_net(stub, images, results)
        assert len(stub.tensors) == 3
        reference = _preprocessing_scorer()
        for tensor, pil in zip(stub.tensors, pils):
            assert torch.equal(tensor, reference.preprocess(_reference_bgr(pil)))
        assert all(r["comp_score"] == 6.5 and r["composition_pattern"] == "rule_of_thirds"
                   for r in results.values())

    def test_a_grayscale_input_would_differ(self):
        # What the fault in the pass looks like: guards the tensor comparison itself.
        scorer = _preprocessing_scorer()
        pil = Image.fromarray(_rgb((64, 48), seed=1))
        assert not torch.equal(scorer.preprocess(pil), scorer.preprocess(pil.convert("L")))


# ---------------------------------------------------------------------------
# The bound the memory budget reads
# ---------------------------------------------------------------------------

def _walk_bytes(entry):
    """Retained bytes of one entry: PIL counted at its 4 B/px in-memory size, arrays by nbytes."""
    total = 0
    seen = set()

    def add(obj):
        nonlocal total
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, Image.Image):
            total += obj.width * obj.height * 4
        elif isinstance(obj, np.ndarray):
            total += obj.nbytes
        elif isinstance(obj, (bytes, bytearray)):
            total += len(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                add(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                add(v)
        elif hasattr(obj, "__slots__"):
            for s in obj.__slots__:
                add(getattr(obj, s, None))

    add(entry)
    return total


class TestRetainedBytesPerPx:
    def test_the_constant_is_a_quarter_step_far_below_the_old_retention(self):
        value = ChunkedMultiPassProcessor.RETAINED_BYTES_PER_PX
        assert value > 0 and (value * 4) == int(value * 4)  # a multiple of 0.25
        assert value < 5.0  # PIL 4 + cv 3 + gray 1 was ~8.1 B/px before S9

    @pytest.mark.timeout(300)
    def test_it_covers_the_walked_bytes_of_a_synthetic_chunk(self, tmp_path, real_config):
        # A smooth 20 MP image: cheap to encode and analyse, and large enough that the shared
        # LANCZOS-1024 image is the small extra it is for a real photo.
        w, h = 5000, 4000
        xs = np.linspace(0, 255, w, dtype=np.float32)
        ys = np.linspace(0, 255, h, dtype=np.float32)[:, None]
        arr = np.stack([xs[None, :] + 0 * ys, 0 * xs[None, :] + ys, (xs[None, :] + ys) / 2], axis=-1)
        arr = arr.astype(np.uint8)
        cv2.rectangle(arr, (w // 4, h // 4), (w // 2, h // 2), (230, 200, 40), -1)
        path = _save(tmp_path, "big.jpg", arr, fmt="JPEG")
        del arr
        images = _load(_processor(real_config), [path])
        assert len(images) == 1
        entry = next(iter(images.values()))
        assert entry["bounded"] is not None and entry["bounded"] is not entry["pil"]
        walked = _walk_bytes(entry) / (w * h)
        assert 3.9 < walked < 4.6, walked  # sanity: PIL 4 B/px plus the small bounded image
        assert ChunkedMultiPassProcessor.RETAINED_BYTES_PER_PX >= walked


# ---------------------------------------------------------------------------
# R9: no pass mutates the retained PIL image
# ---------------------------------------------------------------------------

def _digest(pil):
    return hashlib.sha256(pil.tobytes()).hexdigest()


class _Clip:
    def encode_image(self, inputs):
        return torch.ones((inputs.shape[0], 4))


class _Tagger:
    def tag_batch(self, imgs, max_tags=None):
        return [["tag"] for _ in imgs]


class _Quality:
    def score_batch(self, imgs):
        return [5.0 for _ in imgs]


class _Aesthetic:
    def score(self, pil, task="aesthetic"):
        return 5.0

    def score_tasks(self, pil, tasks):
        return {t: 5.0 for t in tasks}


class _Preprocess:
    def __call__(self, pil):
        return torch.zeros((3, 4, 4))


class TestPassesNeverMutateThePilImage:
    def _images(self, count=3):
        return {f"/p{i}.jpg": _entry(Image.fromarray(_rgb((48 + 5 * i, 36), seed=i)),
                                     bounded=Image.fromarray(_rgb((16, 12), seed=i)))
                for i in range(count)}

    def _plan(self):
        tagger_key = multi_pass.VLM_TAGGER_KEYS[0]
        return [
            ("clip", {"model": _Clip(), "preprocess": _Preprocess(), "backend": "open_clip"}),
            ("samp_net", _SampStub()),
            ("insightface", _RecordingAnalyzer()),
            ("qrealign", _Aesthetic()),
            (tagger_key, _Tagger()),
            ("saliency", _CallableRecorder()),
            ("topiq", _Quality()),
            ("aesthetic_v25", _Quality()),
        ]

    def test_pil_bytes_are_identical_after_every_pass(self, real_config, monkeypatch):
        proc = _processor(real_config)
        monkeypatch.setattr(proc, "_qrealign_is_primary", lambda: False)
        images = self._images()
        results = {p: {} for p in images}
        for name, model in self._plan():
            before = {p: _digest(e["pil"]) for p, e in images.items()}
            proc._run_model_pass(name, model, images, results)
            after = {p: _digest(e["pil"]) for p, e in images.items()}
            assert after == before, f"pass {name!r} changed a retained PIL image"
        assert results and all(results.values())  # the passes really ran

    def test_the_guard_detects_a_mutating_pass(self, real_config):
        class Mutator:
            def score_batch(self, imgs):
                for im in imgs:
                    im.putpixel((0, 0), (1, 2, 3))
                return [5.0 for _ in imgs]

        proc = _processor(real_config)
        images = {p: _entry(Image.new("RGB", (8, 8), (9, 9, 9))) for p in ("/a.jpg",)}
        before = {p: _digest(e["pil"]) for p, e in images.items()}
        proc._run_model_pass("topiq", Mutator(), images, {p: {} for p in images})
        assert {p: _digest(e["pil"]) for p, e in images.items()} != before


# ---------------------------------------------------------------------------
# The rebuild time is accounted
# ---------------------------------------------------------------------------

class TestRebuildTimeMetric:
    def test_starts_at_zero(self, real_config):
        assert _processor(real_config).metrics["rebuild_time"] == 0

    def test_face_pass_accumulates_the_bgr_rebuilds(self, real_config, monkeypatch):
        proc = _processor(real_config)
        images = {f"/p{i}.jpg": _entry(Image.fromarray(_rgb((40, 30), seed=i))) for i in range(3)}
        real = multi_pass.pil_to_bgr

        def slow(pil):
            time.sleep(0.05)
            return real(pil)

        monkeypatch.setattr(multi_pass, "pil_to_bgr", slow)
        proc._pass_insightface(_RecordingAnalyzer(), images, {p: {} for p in images})
        assert 3 * 0.05 * 0.9 <= proc.metrics["rebuild_time"] < 3 * 0.05 + 5

    def test_saliency_pass_accumulates_the_gray_rebuilds(self, real_config):
        proc = _processor(real_config)
        images = {f"/p{i}.jpg": _entry(Image.fromarray(_rgb((40, 30), seed=i))) for i in range(3)}
        scorer = _CallableRecorder()
        proc._pass_saliency(scorer, images, {p: {} for p in images})
        assert proc.metrics["rebuild_time"] == 0  # nothing rebuilt until the scorer asks
        for rebuild in scorer.cv:
            rebuild()
        assert proc.metrics["rebuild_time"] > 0
