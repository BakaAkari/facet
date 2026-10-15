"""PyIQAScorer failure contract (issue #184 follow-up).

A PyIQA scorer used to answer 5.0 for every image it failed to score, which the
writers stored as a real measurement. These tests pin the replacement contract:

* an image that fails yields ``None`` in its slot, never 5.0;
* a CUDA out-of-memory error is retried at half the forward batch down to one
  image, with the cache cleared OUTSIDE the ``except`` block (the live
  exception's traceback pins the failed activations);
* ``topiq_nr_face`` is scored one image at a time (pyiqa asserts batch size 1);
* unexpected failures are logged at ERROR at the end of every call, expected
  skips ("No face detected", "Short side is less than") stay quiet.

The scorers wrap a stub model, so nothing here loads pyiqa or touches a GPU.
"""

import sys

import numpy as np
import pytest
from PIL import Image

# PyIQAScorer imports torch at module load; skip (do not error) where torch is
# absent, e.g. the lightweight CI test env.
pytest.importorskip("torch")

from models.pyiqa_scorer import PyIQAScorer  # noqa: E402

_NO_FACE = "No face detected in the input image."
_SHORT_SIDE = "Short side is less than 224, try upsampling the original image"
_OOM = "CUDA out of memory. Tried to allocate 1.50 GiB (GPU 0; 15.99 GiB total capacity)"


def _img(i, size=(64, 64)):
    """A uniform image whose mean intensity identifies it: (i + 1) * 20 / 255."""
    return Image.fromarray(np.full((size[1], size[0], 3), (i + 1) * 20, dtype=np.uint8), 'RGB')


def _expected(i):
    """The 0-10 score of ``_img(i)`` on a (0, 1) score_range model."""
    return (i + 1) * 20 / 255 * 10


def _ids(batch):
    """Which ``_img`` indices a stacked batch tensor holds."""
    means = batch.mean(dim=[1, 2, 3]).tolist()
    return [round(m * 255 / 20) - 1 for m in means]


class _Model:
    """Per-sample-independent stub. ``fail(ids)`` returns an exception or None."""

    def __init__(self, fail=None):
        self.fail = fail or (lambda ids: None)
        self.batch_sizes = []
        self.kwargs_seen = []

    def __call__(self, batch, **kwargs):
        self.batch_sizes.append(batch.shape[0])
        self.kwargs_seen.append(kwargs)
        err = self.fail(_ids(batch))
        if err is not None:
            raise err
        return batch.mean(dim=[1, 2, 3], keepdim=False).unsqueeze(1)


def _scorer(name, fail=None):
    s = PyIQAScorer(name, device='cpu')
    s.model = _Model(fail)
    s._loaded = True
    return s


@pytest.fixture()
def clears(monkeypatch):
    """Record every clear_device_cache call and whether an exception was live."""
    seen = []

    def fake_clear(device=None):
        seen.append(sys.exc_info()[1])

    monkeypatch.setattr("utils.device.clear_device_cache", fake_clear)
    return seen


def _oom_above(n, exc=None):
    return lambda ids: (exc or RuntimeError(_OOM)) if len(ids) > n else None


# ---------------------------------------------------------------- topiq_nr_face


class TestFaceModelIsSerial:
    def test_topiq_nr_face_is_not_batchable(self):
        assert _scorer('topiq_nr_face').supports_batching is False

    def test_each_image_is_one_model_call_and_a_miss_is_none(self):
        bad = {1, 3}

        def fail(ids):
            return AssertionError(_NO_FACE) if bad & set(ids) else None

        s = _scorer('topiq_nr_face', fail)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert s.model.batch_sizes == [1, 1, 1, 1]
        assert scores[1] is None and scores[3] is None
        assert scores[0] == pytest.approx(_expected(0), abs=1e-3)
        assert scores[2] == pytest.approx(_expected(2), abs=1e-3)
        assert 5.0 not in scores


# ------------------------------------------------------------------- None slots


class TestFailedImageIsNone:
    def test_liqe_short_side_is_none_not_five(self):
        def fail(ids):
            return RuntimeError(_SHORT_SIDE) if 1 in ids else None

        s = _scorer('liqe', fail)
        scores = s.score_batch([_img(i) for i in range(3)])

        assert scores[1] is None
        assert scores[0] == pytest.approx(_expected(0) / 5, abs=1e-3)   # liqe range (0, 5)
        assert scores[2] == pytest.approx(_expected(2) / 5, abs=1e-3)

    def test_single_image_failure_is_none(self):
        s = _scorer('topiq', lambda ids: ValueError("bad tensor"))
        assert s.score_batch([_img(0)]) == [None]

    def test_empty_list_is_empty(self):
        assert _scorer('topiq').score_batch([]) == []


# ------------------------------------------------------------------ OOM halving


class TestOomRetry:
    @pytest.mark.parametrize("exc_factory", [
        lambda: RuntimeError(_OOM),
        lambda: __import__("torch").cuda.OutOfMemoryError("allocation failed"),
    ], ids=["message", "cuda-type"])
    def test_group_is_halved_until_it_fits(self, exc_factory):
        try:
            exc_factory()
        except AttributeError:
            pytest.skip("this torch has no torch.cuda.OutOfMemoryError")
        s = _scorer('topiq', _oom_above(2, exc_factory()))

        scores = s.score_batch([_img(i) for i in range(4)])

        assert s.model.batch_sizes == [4, 2, 2]
        assert scores == pytest.approx([_expected(i) for i in range(4)], abs=1e-3)

    def test_cache_is_cleared_outside_the_handler(self, clears):
        s = _scorer('topiq', _oom_above(2))
        s.score_batch([_img(i) for i in range(4)])
        assert clears, "an OOM must clear the device cache"
        assert all(exc is None for exc in clears)

    def test_oom_at_every_size_is_all_none_and_reported(self, caplog):
        s = _scorer('topiq', lambda ids: RuntimeError(_OOM))
        with caplog.at_level("DEBUG", logger="facet.pyiqa"):
            scores = s.score_batch([_img(i) for i in range(4)])

        assert scores == [None] * 4
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert "4 image(s)" in errors[0]
        assert "out of memory" in errors[0]
        assert "Tried to allocate" not in errors[0]      # allocator sizes are normalised away

    def test_a_lone_failure_fails_alone(self):
        def fail(ids):
            if len(ids) > 1 or 2 in ids:
                return RuntimeError(_OOM)

        s = _scorer('topiq', fail)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert scores[2] is None
        assert [scores[i] for i in (0, 1, 3)] == pytest.approx(
            [_expected(i) for i in (0, 1, 3)], abs=1e-3)

    def test_reduction_warning_omits_the_exception_text(self, caplog):
        s = _scorer('topiq', _oom_above(2))
        with caplog.at_level("WARNING", logger="facet.pyiqa"):
            s.score_batch([_img(i) for i in range(4)])
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "reduced to 2" in warnings[0]
        assert "Tried to allocate" not in warnings[0] and "out of memory CUDA" not in warnings[0]

    def test_non_oom_group_error_falls_back_per_image_without_halving(self, clears):
        s = _scorer('topiq', lambda ids: ValueError("boom") if len(ids) > 1 else None)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert s.model.batch_sizes == [4, 1, 1, 1, 1]
        assert scores == pytest.approx([_expected(i) for i in range(4)], abs=1e-3)
        assert clears == []
        assert s.forward_batch == 4

    def test_group_error_then_per_image_oom_clears_with_no_live_exception(self, clears):
        """The per-image fallback also runs after the group's handler has exited."""
        def fail(ids):
            if len(ids) > 1:
                return ValueError("boom")
            if 1 in ids:
                return RuntimeError(_OOM)

        s = _scorer('topiq', fail)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert s.model.batch_sizes == [4, 1, 1, 1, 1]
        assert scores[1] is None
        assert clears and all(exc is None for exc in clears)

    def test_serial_model_oom_is_none_and_clears_outside_the_handler(self, clears):
        s = _scorer('liqe', lambda ids: RuntimeError(_OOM) if 1 in ids else None)
        scores = s.score_batch([_img(i) for i in range(3)])

        assert scores[1] is None and scores[0] is not None and scores[2] is not None
        assert clears and all(exc is None for exc in clears)

    def test_halved_batch_is_sticky_and_reset_by_unload(self):
        s = _scorer('topiq', _oom_above(2))
        s.score_batch([_img(i) for i in range(4)])
        assert s.model.batch_sizes == [4, 2, 2]
        assert s.forward_batch == 2

        s.score_batch([_img(i) for i in range(4)])
        assert s.model.batch_sizes[3:] == [2, 2]          # no second 4-forward

        s.unload()
        assert s.forward_batch == PyIQAScorer.MAX_FORWARD_BATCH == 4

    def test_slice_loop_rereads_the_batch_within_one_group(self):
        """An 8-image group whose first 4-forward OOMs continues at the halved size."""
        s = _scorer('topiq', _oom_above(2))
        scores = s.score_batch([_img(i) for i in range(8)])

        assert s.model.batch_sizes == [4, 2, 2, 2, 2]
        assert scores == pytest.approx([_expected(i) for i in range(8)], abs=1e-3)

    def test_every_recursive_forward_keeps_the_forward_kwargs(self):
        s = _scorer('topiq', _oom_above(2))
        s._forward_kwargs = {'task_': 'aesthetic'}
        s.score_batch([_img(i) for i in range(4)])
        assert s.model.batch_sizes == [4, 2, 2]
        assert s.model.kwargs_seen == [{'task_': 'aesthetic'}] * 3


# ------------------------------------------------------------ preprocess failure


class TestPreprocessFailure:
    @pytest.mark.parametrize("exc, is_oom", [
        (RuntimeError(_OOM), True),
        (ValueError("bad image"), False),
    ], ids=["oom", "non-oom"])
    def test_preprocess_failure_degrades_to_the_serial_path(self, monkeypatch, clears, exc, is_oom):
        s = _scorer('topiq')
        original = s._preprocess_image

        def flaky(image):
            if image.getpixel((0, 0))[0] == 3 * 20:       # _img(2)
                raise exc
            return original(image)

        monkeypatch.setattr(s, '_preprocess_image', flaky)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert scores[2] is None
        assert [scores[i] for i in (0, 1, 3)] == pytest.approx(
            [_expected(i) for i in (0, 1, 3)], abs=1e-3)
        assert s.model.batch_sizes == [1, 1, 1]            # the serial path, no stacked forward
        if is_oom:
            assert clears and all(e is None for e in clears)
        else:
            assert clears == []


# ---------------------------------------------------------------------- logging


class TestFailureLogging:
    def test_an_unexpected_failure_is_an_error_at_the_end_of_the_call(self, caplog):
        def fail(ids):
            if len(ids) > 1:
                return RuntimeError("batch broke")
            if ids == [1]:
                return ValueError("bad tensor")

        s = _scorer('topiq', fail)
        with caplog.at_level("DEBUG", logger="facet.pyiqa"):
            s.score_batch([_img(i) for i in range(3)])
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert errors and "failed for 1 image(s)" in errors[0] and "bad tensor" in errors[0]

    def test_an_expected_skip_logs_nothing_above_debug_until_the_total(self, caplog):
        s = _scorer('topiq_nr_face',
                    lambda ids: AssertionError(_NO_FACE) if 1 in ids else None)
        with caplog.at_level("DEBUG", logger="facet.pyiqa"):
            s.score_batch([_img(i) for i in range(3)])
        assert [r for r in caplog.records if r.levelno > 10] == []

        caplog.clear()
        with caplog.at_level("DEBUG", logger="facet.pyiqa"):
            s.unload()
        totals = [r.getMessage() for r in caplog.records if "in total" in r.getMessage()]
        assert len(totals) == 1 and "skipped 1 image(s)" in totals[0] and "No face detected" in totals[0]

    def test_report_skips_logs_the_totals_once_and_clears_them(self, caplog):
        s = _scorer('topiq_nr_face',
                    lambda ids: AssertionError(_NO_FACE) if ids == [0] else None)
        s.score_batch([_img(0), _img(1)])
        with caplog.at_level("WARNING", logger="facet.pyiqa"):
            s.report_skips()
            s.report_skips()
        totals = [r for r in caplog.records if "in total" in r.getMessage()]
        assert len(totals) == 1
        assert s._skipped_totals == {}


# --------------------------------------------------------------- last_declined


class TestLastDeclined:
    """``last_declined``: positions whose ``None`` is the model's own, deliberate answer.

    ``--recompute-iqa`` remembers those and retries everything else, so an OOM or
    an unexpected error must never land in the set.
    """

    def test_it_starts_empty(self):
        assert _scorer('topiq_nr_face').last_declined == frozenset()

    def test_no_face_detected_positions_are_exactly_the_declined_ones(self):
        def fail(ids):
            return AssertionError(_NO_FACE) if {1, 3} & set(ids) else None

        s = _scorer('topiq_nr_face', fail)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert scores[1] is None and scores[3] is None
        assert s.last_declined == frozenset({1, 3})
        assert isinstance(s.last_declined, frozenset)

    def test_liqe_short_side_is_declined(self):
        s = _scorer('liqe', lambda ids: RuntimeError(_SHORT_SIDE) if 1 in ids else None)
        s.score_batch([_img(i) for i in range(3)])
        assert s.last_declined == frozenset({1})

    def test_an_oom_is_none_but_not_declined(self):
        s = _scorer('liqe', lambda ids: RuntimeError(_OOM) if 1 in ids else None)
        scores = s.score_batch([_img(i) for i in range(3)])

        assert scores[1] is None
        assert s.last_declined == frozenset()

    def test_an_unexpected_error_is_none_but_not_declined(self):
        s = _scorer('topiq_nr_face', lambda ids: RuntimeError('boom') if 1 in ids else None)
        scores = s.score_batch([_img(i) for i in range(3)])

        assert scores[1] is None
        assert s.last_declined == frozenset()

    def test_a_mixed_batch_yields_only_the_expected_positions(self):
        def fail(ids):
            if 0 in ids:
                return AssertionError(_NO_FACE)
            if 1 in ids:
                return RuntimeError(_OOM)
            if 2 in ids:
                return RuntimeError('boom')
            if 3 in ids:
                return RuntimeError(_SHORT_SIDE)

        s = _scorer('liqe', fail)
        scores = s.score_batch([_img(i) for i in range(5)])

        assert [i for i, v in enumerate(scores) if v is None] == [0, 1, 2, 3]
        assert s.last_declined == frozenset({0, 3})

    def test_it_is_reset_on_every_call(self):
        s = _scorer('topiq_nr_face', lambda ids: AssertionError(_NO_FACE) if 0 in ids else None)
        s.score_batch([_img(0), _img(1)])
        assert s.last_declined == frozenset({0})

        s.score_batch([_img(1), _img(2)])           # a clean second call
        assert s.last_declined == frozenset()

    def test_it_is_reset_even_for_an_empty_batch(self):
        s = _scorer('topiq_nr_face', lambda ids: AssertionError(_NO_FACE) if 0 in ids else None)
        s.score_batch([_img(0)])
        assert s.last_declined == frozenset({0})
        s.score_batch([])
        assert s.last_declined == frozenset()

    def test_a_batched_model_keeps_the_original_positions_through_the_fallback(self):
        """A stacked forward that fails falls back to per-image scoring; the position
        recorded is the image's place in the caller's list, not in a sub-group."""
        def fail(ids):
            return AssertionError(_NO_FACE) if 2 in ids else None

        s = _scorer('topiq', fail)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert scores[2] is None
        assert [scores[i] for i in (0, 1, 3)] == pytest.approx(
            [_expected(i) for i in (0, 1, 3)], abs=1e-3)
        assert s.last_declined == frozenset({2})

    def test_a_batched_model_oom_is_not_declined(self):
        s = _scorer('topiq', lambda ids: RuntimeError(_OOM))
        assert s.score_batch([_img(i) for i in range(4)]) == [None] * 4
        assert s.last_declined == frozenset()

    def test_a_preprocess_failure_falls_back_and_keeps_the_positions(self, monkeypatch):
        s = _scorer('topiq', lambda ids: AssertionError(_NO_FACE) if 3 in ids else None)
        original = s._preprocess_image

        def flaky(image):
            if image.getpixel((0, 0))[0] == 2 * 20:       # _img(1)
                raise ValueError("bad image")
            return original(image)

        monkeypatch.setattr(s, '_preprocess_image', flaky)
        scores = s.score_batch([_img(i) for i in range(4)])

        assert scores[1] is None and scores[3] is None
        assert s.last_declined == frozenset({3})

    def test_the_native_qrealign_scorer_declares_the_attribute(self):
        from models.qrealign_native import QReAlignNativeScorer
        assert QReAlignNativeScorer.last_declined == frozenset()


# ----------------------------------------------------------------- aesthetic_v25


class TestAestheticV25:
    def test_a_failed_image_is_none_not_five(self):
        pytest.importorskip("PIL")
        try:
            from models.aesthetic_v25_scorer import AestheticV25Scorer
        except ImportError:
            pytest.skip("aesthetic_v25 scorer needs a package this env lacks")

        s = AestheticV25Scorer.__new__(AestheticV25Scorer)
        s._loaded = True

        def broken_processor(*a, **k):                # forces the per-image fallback
            raise RuntimeError("batch broke")

        s.processor = broken_processor

        def score_image(image):
            if image.getpixel((0, 0))[0] == 2 * 20:   # _img(1)
                raise RuntimeError("bad image")
            return 7.5

        s.score_image = score_image
        assert s.score_batch([_img(0), _img(1), _img(2)]) == [7.5, None, 7.5]
