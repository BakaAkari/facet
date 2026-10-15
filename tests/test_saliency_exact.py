"""Exactness tests for the S2 rewrite of ``SaliencyScorer._score_from_mask`` and ``bbox_from_mask``.

The rewrite swaps a float64 numpy formulation for cv2 integer arithmetic. The contract is that the
returned dict is *identical* to the old one: same keys, same Python types, same float bit patterns.
``tests/_saliency_reference.py`` holds the old code verbatim; every case here compares against it,
never against a route taken inside the new code.

Also covers the streaming contract ``score_batch`` gives A2: per-image elements may be an ndarray or a
zero-argument callable, and at most one sub-batch of masks is alive at a time.
"""
import functools
import gc
import logging
import struct
import weakref

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

import models.saliency_scorer as ss  # noqa: E402
from models.saliency_scorer import SaliencyScorer, bbox_from_mask  # noqa: E402
from tests._saliency_reference import ReferenceSaliency, reference_bbox_from_mask  # noqa: E402

KEYS = ['subject_sharpness', 'subject_prominence', 'subject_placement', 'bg_separation', 'subject_bbox']


# ---------------------------------------------------------------------------
# helpers

def _scorer(min_subject_pixels=50):
    sc = SaliencyScorer.__new__(SaliencyScorer)
    sc.min_subject_pixels = min_subject_pixels
    sc.mask_threshold = 0.3
    return sc


def _bits(x):
    return struct.pack('<d', x)


def _diffs(a, b, where=''):
    """Strict structural differences: type identity first, floats by bit pattern."""
    if type(a) is not type(b):
        return ['%s: type %s != %s (%r vs %r)' % (where, type(a).__name__, type(b).__name__, a, b)]
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return ['%s: keys %s != %s' % (where, list(a), list(b))]
        out = []
        for k in a:
            out += _diffs(a[k], b[k], '%s.%s' % (where, k))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return ['%s: len %d != %d' % (where, len(a), len(b))]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diffs(x, y, '%s[%d]' % (where, i))
        return out
    if isinstance(a, float):
        return [] if _bits(a) == _bits(b) else ['%s: %r != %r' % (where, a, b)]
    return [] if a == b else ['%s: %r != %r' % (where, a, b)]


def assert_same(ref, new):
    d = _diffs(ref, new)
    assert not d, '\n'.join(d)


def check(mask, img, min_pix=50):
    """New vs the verbatim reference for one (mask, image); returns the (identical) dict."""
    ref = ReferenceSaliency(min_pix)._score_from_mask(mask, img)
    new = _scorer(min_pix)._score_from_mask(mask, img)
    assert_same(ref, new)
    assert list(new) == KEYS
    return new


def _textured(h, w, seed=0, channels=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w, channels) if channels == 3 else (h, w), dtype=np.uint8)
    # a smooth ramp + noise so the Laplacian has both flat and busy areas
    ramp = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    mix = (base.astype(np.float32) * 0.15 + (ramp if channels == 1 else ramp[..., None]) * 0.85)
    return mix.clip(0, 255).astype(np.uint8)


def _rect(h, w, y0, y1, x0, x1, value=255):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = value
    return m


H, W = 60, 80


@pytest.fixture(scope='module')
def img():
    return _textured(H, W, seed=1)


# ---------------------------------------------------------------------------
# binary masks

class TestBinaryMasks:
    def test_blob_in_the_middle(self, img):
        res = check(_rect(H, W, 15, 40, 20, 55), img)
        assert res['subject_bbox'] is not None

    def test_empty_mask_returns_the_neutral_dict(self, img):
        res = check(np.zeros((H, W), np.uint8), img)
        assert res['subject_sharpness'] == 5.0 and res['subject_bbox'] is None

    @pytest.mark.parametrize('n', [49, 50, 51])
    def test_around_min_subject_pixels(self, img, n):
        m = np.zeros((H, W), np.uint8)
        m.reshape(-1)[:n] = 255
        res = check(m, img)
        assert (res['subject_bbox'] is None) == (n < 50)

    def test_full_frame(self, img):
        res = check(np.full((H, W), 255, np.uint8), img)
        assert res['bg_separation'] == 10.0          # no background pixels at all
        assert res['subject_bbox'] == [0.0, 0.0, 1.0, 1.0]

    def test_one_pixel_wide_row(self, img):
        check(_rect(H, W, 30, 31, 0, W), img)

    def test_one_pixel_wide_column(self, img):
        check(_rect(H, W, 0, H, 40, 41), img)

    def test_checkerboard(self, img):
        yy, xx = np.indices((H, W))
        check((((yy + xx) % 2) * 255).astype(np.uint8), img)

    @pytest.mark.parametrize('box', [
        (0, 20, 0, 30), (0, 20, 50, W), (40, H, 0, 30), (40, H, 50, W), (0, H, 0, 10), (0, 10, 0, W),
    ])
    def test_subject_touching_each_border(self, img, box):
        check(_rect(H, W, *box), img)

    def test_mask_smaller_than_image(self):
        big = _textured(120, 160, seed=2)
        check(_rect(H, W, 10, 40, 10, 60), big)

    def test_mask_larger_than_image(self):
        small = _textured(30, 40, seed=3)
        check(_rect(H, W, 10, 40, 10, 60), small)

    @pytest.mark.parametrize('value', [0, 7, 128, 255])
    def test_constant_image(self, value):
        flat = np.full((H, W, 3), value, np.uint8)
        res = check(_rect(H, W, 10, 40, 10, 60), flat)
        # a constant region must give exactly zero variance, as np.var does
        assert res['subject_sharpness'] == 0.0 and res['bg_separation'] == 10.0

    def test_constant_subject_over_a_busy_background(self):
        noisy = _textured(H, W, seed=4)
        noisy[10:40, 10:60] = 90
        check(_rect(H, W, 12, 38, 12, 58), noisy)

    def test_busy_subject_over_a_constant_background(self):
        flat = np.full((H, W, 3), 40, np.uint8)
        flat[10:40, 10:60] = _textured(30, 50, seed=5)
        check(_rect(H, W, 10, 40, 10, 60), flat)

    def test_saturated_image(self):
        rng = np.random.default_rng(6)
        harsh = (rng.integers(0, 2, (H, W, 3)) * 255).astype(np.uint8)
        check(_rect(H, W, 5, 55, 5, 75), harsh)

    def test_gray_and_bgr_inputs_give_the_same_dict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = _rect(H, W, 15, 40, 20, 55)
        assert_same(check(mask, img), check(mask, gray))

    @pytest.mark.parametrize('min_pix', [1, 50, 400, 10_000])
    def test_min_subject_pixels_setting_is_honoured(self, img, min_pix):
        check(_rect(H, W, 15, 40, 20, 55), img, min_pix)

    def test_non_contiguous_mask_and_image(self, img):
        mask = _rect(2 * H, 2 * W, 20, 80, 30, 110)[::2, ::2]
        assert not mask.flags['C_CONTIGUOUS']
        check(mask, img[:, ::-1])

    def test_random_blobs_of_many_shapes(self):
        rng = np.random.default_rng(7)
        for k in range(40):
            h = int(rng.integers(8, 220))
            w = int(rng.integers(8, 220))
            noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), float(rng.uniform(1, 8)))
            mask = ((noise > np.quantile(noise, rng.uniform(0.2, 0.9))).astype(np.uint8)) * 255
            check(mask, _textured(h, w, seed=100 + k), min_pix=int(rng.integers(1, 80)))

    def test_zero_sized_mask(self):
        check(np.zeros((0, 5), np.uint8), np.zeros((0, 5, 3), np.uint8))


# ---------------------------------------------------------------------------
# non-binary uint8 masks: asserted against the REFERENCE (three different thresholds, spec D22)

class TestNonBinaryUint8Masks:
    def test_values_only_in_1_to_128_pass_the_gate_with_an_empty_subject_region(self, img):
        mask = _rect(H, W, 15, 40, 20, 55, value=128)
        ref = ReferenceSaliency()._score_from_mask(mask, img)
        # the reference's own behaviour: non-zero count gates, > 128 builds the region
        assert ref['subject_bbox'] is None and ref['subject_placement'] == 5.0
        assert ref['subject_prominence'] > 0.0
        assert_same(ref, _scorer()._score_from_mask(mask, img))

    @pytest.mark.parametrize('value', [1, 64, 127, 128, 129, 200])
    def test_flat_levels(self, img, value):
        check(_rect(H, W, 15, 40, 20, 55, value=value), img)

    def test_exactly_128_versus_129(self, img):
        a = check(_rect(H, W, 15, 40, 20, 55, value=128), img)
        b = check(_rect(H, W, 15, 40, 20, 55, value=129), img)
        assert a['subject_bbox'] is None and b['subject_bbox'] is not None

    def test_ramp_0_to_255(self):
        mask = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
        check(mask, _textured(64, 256, seed=8))

    def test_gt_128_count_below_min_while_nonzero_count_is_above(self, img):
        mask = np.zeros((H, W), np.uint8)
        mask[10:20, 10:20] = 100          # 100 px, non-zero but not > 128
        mask[30, 30:50] = 255             # 20 px > 128, below min_subject_pixels=50
        ref = ReferenceSaliency()._score_from_mask(mask, img)
        assert ref['subject_bbox'] is None
        assert ref['subject_prominence'] > 0.0
        assert_same(ref, _scorer()._score_from_mask(mask, img))

    def test_random_levels(self):
        rng = np.random.default_rng(9)
        for k in range(10):
            h, w = int(rng.integers(20, 120)), int(rng.integers(20, 120))
            check(rng.integers(0, 256, (h, w), dtype=np.uint8), _textured(h, w, seed=200 + k))


# ---------------------------------------------------------------------------
# masks that are not 2-D uint8 keep the numpy code

class TestOtherMaskTypes:
    @pytest.mark.parametrize('make', [
        lambda m: m.astype(np.float32),
        lambda m: m.astype(np.float64) / 255.0,
        lambda m: m.astype(bool),
        lambda m: m.astype(np.int32),
        lambda m: m.astype(np.uint16) * 200,
    ], ids=['float32', 'float64-unit', 'bool', 'int32', 'uint16'])
    def test_matches_the_reference(self, img, make):
        check(make(_rect(H, W, 15, 40, 20, 55)), img)

    def test_float_mask_takes_the_numpy_code(self, img, monkeypatch):
        calls = []
        real = np.var
        monkeypatch.setattr(np, 'var', lambda *a, **k: calls.append(1) or real(*a, **k))
        _scorer()._score_from_mask(_rect(H, W, 15, 40, 20, 55).astype(np.float32), img)
        assert calls

    def test_float_gray_plane_with_a_uint8_mask_still_matches(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        check(_rect(H, W, 15, 40, 20, 55), gray)

    def test_uint8_mask_does_not_build_index_arrays_or_float_variances(self, img, monkeypatch):
        """The speed-up IS the absence of np.nonzero / np.var over the full frame."""
        def boom(*a, **k):
            raise AssertionError('numpy full-frame path used for a uint8 mask')
        real_nonzero = np.nonzero

        def nonzero_1d_only(a):
            # np.flatnonzero over a w- or h-length projection is fine; a 2-D index build is the old path
            if getattr(a, 'ndim', 1) >= 2:
                boom()
            return real_nonzero(a)
        monkeypatch.setattr(np, 'nonzero', nonzero_1d_only)
        monkeypatch.setattr(np, 'var', boom)
        monkeypatch.setattr(np, 'mean', boom)
        res = _scorer()._score_from_mask(_rect(H, W, 15, 40, 20, 55), img)
        assert res['subject_bbox'] is not None
        assert bbox_from_mask(_rect(H, W, 15, 40, 20, 55)) is not None


# ---------------------------------------------------------------------------
# bbox_from_mask on its own

class TestBboxFromMask:
    @pytest.mark.parametrize('mask', [
        np.zeros((10, 10), np.uint8),
        np.full((50, 50), 255, np.uint8),
        _rect(40, 60, 8, 24, 15, 45),
        _rect(40, 60, 0, 40, 59, 60),
        _rect(40, 60, 10, 20, 10, 20, value=128),
        _rect(40, 60, 10, 20, 10, 20, value=129),
        np.zeros((0, 7), np.uint8),
        np.zeros((7, 0), np.uint8),
    ])
    @pytest.mark.parametrize('min_pix', [1, 50])
    def test_matches_the_reference(self, mask, min_pix):
        assert_same(reference_bbox_from_mask(mask, min_pix), bbox_from_mask(mask, min_pix))

    def test_returns_python_floats(self):
        box = bbox_from_mask(_rect(40, 60, 8, 24, 15, 45))
        assert type(box) is list and all(type(v) is float for v in box)

    def test_float_mask_uses_the_old_path(self):
        mask = _rect(40, 60, 8, 24, 15, 45).astype(np.float32)
        assert_same(reference_bbox_from_mask(mask), bbox_from_mask(mask))


# ---------------------------------------------------------------------------
# streaming and the callable element form of score_batch

class _Tracked(np.ndarray):
    """ndarray subclass whose live instances are counted (weakref.finalize)."""


class _Live:
    def __init__(self):
        self.n = 0
        self.peak = 0

    def track(self, arr):
        t = arr.view(_Tracked)
        self.n += 1
        self.peak = max(self.peak, self.n)
        weakref.finalize(t, self._drop)
        return t

    def _drop(self):
        self.n -= 1


def _mask_for(w, h):
    m = np.zeros((h, w), np.uint8)
    m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    return m


def _stub_scorer(batch, live=None, forwards=None, fail_on=None, oom_sizes=()):
    sc = SaliencyScorer(device='cpu')
    sc._loaded = True
    sc._affordable_batch_size = lambda requested: batch

    def run_chunk(chunk, sizes):
        if forwards is not None:
            forwards.append(len(chunk))
        if fail_on is not None and len(forwards) == fail_on:
            raise ValueError('forward exploded')
        if set(sizes) & set(oom_sizes):
            raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 3.75 GiB')
        out = [_mask_for(w, h) for (w, h) in sizes]
        return [live.track(m) for m in out] if live is not None else out
    sc._run_chunk = run_chunk
    return sc


def _pils(n, size=(80, 60)):
    from PIL import Image
    return [Image.new('RGB', (size[0] + i, size[1]), (i, i, i)) for i in range(n)]


def _grays(pils):
    return [_textured(p.size[1], p.size[0], seed=300 + i, channels=1) for i, p in enumerate(pils)]


class TestStreamingScoreBatch:
    def test_peak_live_masks_never_exceeds_the_sub_batch_size(self):
        live = _Live()
        forwards = []
        sc = _stub_scorer(2, live=live, forwards=forwards)
        pils = _pils(7)
        res = sc.score_batch(pils, _grays(pils))
        gc.collect()
        assert len(res) == 7 and all(r['subject_sharpness'] is not None for r in res)
        assert forwards == [2, 2, 2, 1]
        assert live.peak <= 2, 'peak %d masks alive for sub-batch size 2' % live.peak
        assert live.n == 0

    def test_get_saliency_masks_is_still_the_whole_aligned_list(self):
        forwards = []
        sc = _stub_scorer(3, forwards=forwards)
        masks = sc.get_saliency_masks(_pils(7))
        assert len(masks) == 7 and forwards == [3, 3, 1]
        assert [m.shape for m in masks] == [(60, 80 + i) for i in range(7)]

    def test_sub_batch_size_is_computed_once_per_call(self):
        sc = _stub_scorer(2)
        calls = []
        sc._affordable_batch_size = lambda requested: calls.append(requested) or 2
        pils = _pils(5)
        sc.score_batch(pils, _grays(pils))
        assert len(calls) == 1

    def test_results_equal_the_one_shot_scoring(self):
        sc = _stub_scorer(2)
        pils = _pils(5)
        grays = _grays(pils)
        res = sc.score_batch(pils, grays)
        ref = [ReferenceSaliency()._score_from_mask(_mask_for(*p.size), g) for p, g in zip(pils, grays)]
        for r, e in zip(res, ref):
            assert_same(e, r)

    def test_callable_elements_give_the_same_dicts_as_arrays(self):
        pils = _pils(5)
        grays = _grays(pils)
        by_array = _stub_scorer(2).score_batch(pils, grays)
        by_call = _stub_scorer(2).score_batch(pils, [functools.partial(lambda g: g.copy(), g) for g in grays])
        for a, b in zip(by_array, by_call):
            assert_same(a, b)

    def test_at_most_one_callable_result_is_alive_at_a_time(self):
        live = _Live()
        pils = _pils(6)
        grays = _grays(pils)
        elems = [functools.partial(lambda g: live.track(g.copy()), g) for g in grays]
        res = _stub_scorer(3).score_batch(pils, elems)
        gc.collect()
        assert all(r['subject_sharpness'] is not None for r in res)
        assert live.peak == 1, 'peak %d rebuilt planes alive' % live.peak

    def test_callable_is_not_invoked_for_an_image_without_a_mask(self):
        pils = _pils(3)
        poison = pils[1].size
        invoked = []

        def elem(i, g):
            invoked.append(i)
            return g
        grays = _grays(pils)
        res = _stub_scorer(3, oom_sizes=[poison]).score_batch(
            pils, [functools.partial(elem, i, g) for i, g in enumerate(grays)])
        assert [r['subject_sharpness'] is None for r in res] == [False, True, False]
        assert invoked == [0, 2]

    def test_a_raising_callable_fails_only_its_own_image(self, caplog):
        pils = _pils(4)
        grays = _grays(pils)

        def boom():
            raise RuntimeError('rebuild failed')
        elems = [grays[0], boom, grays[2], grays[3]]
        with caplog.at_level(logging.ERROR, logger='facet.saliency'):
            res = _stub_scorer(2).score_batch(pils, elems)
        assert [r['subject_sharpness'] is None for r in res] == [False, True, False, False]
        assert any('Saliency scoring failed for batch size 4' in r.getMessage() for r in caplog.records)

    def test_a_forward_failure_mid_call_fails_the_whole_call(self, caplog):
        forwards = []
        sc = _stub_scorer(2, forwards=forwards, fail_on=2)
        pils = _pils(5)
        with caplog.at_level(logging.ERROR, logger='facet.saliency'):
            res = sc.score_batch(pils, _grays(pils))
        assert len(res) == 5
        assert all(all(v is None for v in r.values()) for r in res)
        assert any('batch size 5' in r.getMessage() for r in caplog.records)

    def test_unusable_oom_image_still_logs_the_lost_count(self, caplog):
        pils = _pils(3)
        with caplog.at_level(logging.ERROR, logger='facet.saliency'):
            res = _stub_scorer(3, oom_sizes=[pils[1].size]).score_batch(pils, _grays(pils))
        assert sum(r['subject_sharpness'] is None for r in res) == 1
        assert any('Saliency failed for 1 of 3 image(s)' in r.getMessage() for r in caplog.records)

    def test_module_still_exposes_the_helpers(self):
        assert callable(ss.bbox_from_mask)
