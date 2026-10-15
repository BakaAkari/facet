"""The histogram / CV_16S replacements in analyzers.technical and analyzers.image_cache.

Every metric is compared against a frozen copy of the code it replaced
(``tests/_technical_reference.py``) on synthetic and random images. The contract:

* dynamic range, noise and contrast come out EXACTLY equal (same value, same type, same
  bit pattern) after the functions' own rounding;
* ``ImageCache.laplacian_variance`` is the one tolerance (<= 1e-12 relative);
* none of the replaced paths builds a full-frame float64 copy again.
"""

import math
import struct
import tracemalloc

import cv2
import numpy as np
import pytest

from analyzers import technical
from analyzers.image_cache import ImageCache
from analyzers.technical import TechnicalAnalyzer
from tests import _technical_reference as ref


def _bits(value):
    return struct.pack('<d', float(value))


def assert_identical(new, old, where=''):
    """Same keys, same types, same bit pattern for every float (NaN payloads and -0.0 included)."""
    assert list(new) == list(old), where
    for key in old:
        a, b = new[key], old[key]
        assert type(a) is type(b), f'{where}{key}: {type(a).__name__} vs {type(b).__name__}'
        if isinstance(b, float):
            assert _bits(a) == _bits(b), f'{where}{key}: {a!r} vs {b!r}'
        else:
            assert a == b, f'{where}{key}: {a!r} vs {b!r}'


def _outcome(fn):
    """Result of ``fn()`` or the exception class it raised, so error behaviour is compared too."""
    with np.errstate(all='ignore'):
        try:
            return ('ok', fn())
        except Exception as exc:  # noqa: BLE001 - the comparison is the point
            return ('error', type(exc))


def _bgr(gray_or_bgr):
    arr = np.asarray(gray_or_bgr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return np.ascontiguousarray(arr)


def _synthetic_images():
    rng = np.random.default_rng(20260715)
    images = {
        'noise_64x64': rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
        'noise_37x53': rng.integers(0, 256, (37, 53, 3), dtype=np.uint8),
        'noise_480x640': rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        'gradient_h': _bgr(np.tile(np.linspace(0, 255, 300).astype(np.uint8), (200, 1))),
        'gradient_v': _bgr(np.tile(np.linspace(0, 255, 200).astype(np.uint8)[:, None], (1, 300))),
        'const_0': np.zeros((50, 50, 3), np.uint8),
        'const_255': np.full((50, 50, 3), 255, np.uint8),
        'const_128': np.full((50, 50, 3), 128, np.uint8),
        'const_1': np.full((40, 60, 3), 1, np.uint8),
        'row_1xN': rng.integers(0, 256, (1, 40, 3), dtype=np.uint8),
        'col_Nx1': rng.integers(0, 256, (40, 1, 3), dtype=np.uint8),
        'one_pixel': rng.integers(0, 256, (1, 1, 3), dtype=np.uint8),
        '2x2': rng.integers(0, 256, (2, 2, 3), dtype=np.uint8),
        '2x2_flat': np.full((2, 2, 3), 77, np.uint8),
        '3x3': rng.integers(0, 256, (3, 3, 3), dtype=np.uint8),
        '2xN': rng.integers(0, 256, (2, 30, 3), dtype=np.uint8),
        'Nx2': rng.integers(0, 256, (30, 2, 3), dtype=np.uint8),
    }
    checker = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    images['checkerboard'] = _bgr(checker)
    halves = np.zeros((60, 80), np.uint8)
    halves[:, 40:] = 255
    images['saturated_halves'] = _bgr(halves)
    two_tone = np.full((70, 70), 5, np.uint8)
    two_tone[:7] = 250
    images['two_tone_5_250'] = _bgr(two_tone)
    colour = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    colour[:, :, 0] = 255
    images['saturated_blue'] = colour
    # Sparse histogram: a handful of grey levels so percentile ranks land on bin boundaries.
    sparse = rng.choice(np.array([3, 60, 61, 200, 201, 255], np.uint8), size=(101, 99))
    images['sparse_levels'] = _bgr(sparse)
    return images


IMAGES = _synthetic_images()


def _both(name):
    img = IMAGES[name]
    return img, ImageCache(img), ref.OldImageCache(img)


# ----------------------------------------------------------------------------------------------
# dynamic range / noise / contrast: exactly equal
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(IMAGES))
@pytest.mark.parametrize('use_cache', [True, False])
def test_dynamic_range_is_identical(name, use_cache):
    img, cache, old_cache = _both(name)
    new = _outcome(lambda: TechnicalAnalyzer.get_dynamic_range(img, cache=cache if use_cache else None))
    old = _outcome(lambda: ref.old_dynamic_range(img, cache=old_cache if use_cache else None))
    assert new[0] == old[0]
    if old[0] == 'ok':
        assert_identical(new[1], old[1], f'{name}: ')
    else:
        assert new[1] is old[1]


@pytest.mark.parametrize('name', sorted(IMAGES))
@pytest.mark.parametrize('use_cache', [True, False])
def test_noise_estimate_is_identical(name, use_cache):
    img, cache, old_cache = _both(name)
    new = _outcome(lambda: TechnicalAnalyzer.get_noise_estimate(img, cache=cache if use_cache else None))
    old = _outcome(lambda: ref.old_noise_estimate(img, cache=old_cache if use_cache else None))
    assert new[0] == old[0]
    if old[0] == 'ok':
        assert_identical(new[1], old[1], f'{name}: ')
    else:
        assert new[1] is old[1]


@pytest.mark.parametrize('name', sorted(IMAGES))
@pytest.mark.parametrize('use_cache', [True, False])
def test_contrast_score_is_identical_after_its_own_rounding(name, use_cache):
    img, cache, old_cache = _both(name)
    new = _outcome(lambda: TechnicalAnalyzer.get_contrast_score(img, cache=cache if use_cache else None))
    old = _outcome(lambda: ref.old_contrast_score(img, cache=old_cache if use_cache else None))
    assert new[0] == old[0]
    if old[0] == 'ok':
        assert_identical(new[1], old[1], f'{name}: ')
    else:
        assert new[1] is old[1]


def test_two_pixel_wide_image_yields_inf_with_a_numpy_warning_not_zero_division():
    """The noise tail must stay numpy float64: a Python-float tail would raise ZeroDivisionError."""
    img = IMAGES['2x2']
    with pytest.warns(RuntimeWarning):
        out = TechnicalAnalyzer.get_noise_estimate(img, cache=ImageCache(img))
    assert out['noise_sigma'] == math.inf
    assert type(out['noise_sigma']) is np.float64


def test_none_input_short_circuits_like_before():
    assert TechnicalAnalyzer.get_dynamic_range(None) == {'dynamic_range_stops': 0}
    assert TechnicalAnalyzer.get_noise_estimate(None) == {'noise_sigma': 0}
    assert TechnicalAnalyzer.get_contrast_score(None) == {
        'contrast_score': 0, 'percentile_contrast': 0, 'rms_contrast': 0}
    assert TechnicalAnalyzer.get_sharpness_data(None) == {'raw_variance': 0, 'normalized': 0}


def test_random_images_of_assorted_sizes_stay_identical():
    rng = np.random.default_rng(7)
    for _ in range(40):
        h, w = (int(v) for v in rng.integers(3, 120, 2))
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        # A random image shifted into a narrow band exercises the low-count percentiles.
        lo = int(rng.integers(0, 200))
        img = (lo + (img.astype(np.int32) * int(rng.integers(1, 56)) // 255)).astype(np.uint8)
        cache, old_cache = ImageCache(img), ref.OldImageCache(img)
        assert_identical(TechnicalAnalyzer.get_dynamic_range(img, cache=cache),
                         ref.old_dynamic_range(img, cache=old_cache), f'{h}x{w}: ')
        assert_identical(TechnicalAnalyzer.get_noise_estimate(img, cache=cache),
                         ref.old_noise_estimate(img, cache=old_cache), f'{h}x{w}: ')
        assert_identical(TechnicalAnalyzer.get_contrast_score(img, cache=cache),
                         ref.old_contrast_score(img, cache=old_cache), f'{h}x{w}: ')


# ----------------------------------------------------------------------------------------------
# the histogram helpers themselves
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('n', [1, 2, 3, 5, 10, 99, 100, 101, 1000, 4097])
def test_histogram_percentile_matches_numpy_linear_for_every_q(n):
    rng = np.random.default_rng(n)
    for spread in (3, 40, 256):
        data = rng.integers(0, spread, n).astype(np.uint8)
        hist = np.bincount(data, minlength=256).astype(np.int64)
        for q in (0, 1, 2, 5, 10, 25, 33.3, 50, 66.6, 75, 90, 95, 98, 99, 100):
            got = technical.histogram_percentile(hist, q)
            want = np.percentile(data, q)
            assert type(got) is np.float64
            assert _bits(got) == _bits(want), (n, spread, q, got, want)
            want64 = np.percentile(data.astype(np.float64), q)
            assert _bits(got) == _bits(want64), (n, spread, q, got, want64)


def test_histogram_percentile_uses_the_upper_branch_of_the_lerp_when_gamma_is_a_half_or_more():
    # 6 values: q=90 -> virtual index 4.5, gamma exactly 0.5; q=95 -> 4.75. A lerp that
    # always interpolates from the lower neighbour differs from numpy in the last bit
    # for some value pairs, so pick pairs where it does.
    data = np.array([0, 7, 13, 29, 101, 173], np.uint8)
    hist = np.bincount(data, minlength=256).astype(np.int64)
    for q in (10, 30, 50, 70, 90, 95, 99):
        assert _bits(technical.histogram_percentile(hist, q)) == _bits(np.percentile(data, q))


@pytest.mark.parametrize('seed', range(8))
def test_histogram_std_within_a_few_ulp_of_numpy(seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, int(rng.integers(2, 257)), int(rng.integers(1, 5000))).astype(np.uint8)
    hist = np.bincount(data, minlength=256).astype(np.int64)
    got = technical.histogram_std(hist)
    want = float(np.std(data.astype(np.float64)))
    assert type(got) is np.float64
    assert abs(got - want) <= 1e-13 * max(want, 1e-12) or got == want


def test_histogram_std_of_a_constant_image_is_exactly_zero():
    hist = np.zeros(256, np.int64)
    hist[137] = 123457
    assert technical.histogram_std(hist) == 0.0


def test_gray_histogram_is_an_exact_int64_count():
    gray = np.random.default_rng(3).integers(0, 256, (123, 77), dtype=np.uint8)
    hist = technical.gray_histogram(gray)
    assert hist.shape == (256,) and hist.dtype == np.int64
    assert np.array_equal(hist, np.bincount(gray.ravel(), minlength=256))


def test_gray_histogram_is_exact_for_a_single_bin_above_the_float32_integer_limit():
    """A float32 counter saturates at 2**24 pixels in one bin; a 17 MP flat frame is realistic."""
    # 4099**2 = 16_801_801 > 2**24 = 16_777_216; with two pixels moved out of bin 0 the bin holds an
    # ODD count, which float32 cannot represent at that magnitude (the spacing there is 2).
    side = 4099
    gray = np.zeros((side, side), np.uint8)
    gray[0, :2] = 200
    hist = technical.gray_histogram(gray)
    assert int(hist.sum()) == side * side
    assert int(hist[200]) == 2
    assert int(hist[0]) == side * side - 2
    assert int(hist[0]) % 2 == 1
    assert float(np.float32(hist[0])) != int(hist[0])  # the probe really is beyond float32


def test_gray_histogram_handles_a_non_contiguous_view():
    gray = np.random.default_rng(5).integers(0, 256, (90, 120), dtype=np.uint8)
    view = gray[10:80, 5:100]
    assert np.array_equal(technical.gray_histogram(view), np.bincount(view.ravel(), minlength=256))


def test_big_flat_frame_matches_the_reference_end_to_end():
    side = 4099
    img = np.zeros((side, side, 3), np.uint8)
    img[:10] = 255
    cache, old_cache = ImageCache(img), ref.OldImageCache(img)
    assert_identical(TechnicalAnalyzer.get_dynamic_range(img, cache=cache),
                     ref.old_dynamic_range(img, cache=old_cache))
    assert_identical(TechnicalAnalyzer.get_contrast_score(img, cache=cache),
                     ref.old_contrast_score(img, cache=old_cache))
    assert_identical(TechnicalAnalyzer.get_noise_estimate(img, cache=cache),
                     ref.old_noise_estimate(img, cache=old_cache))


# ----------------------------------------------------------------------------------------------
# ImageCache.laplacian_variance: the one tolerance
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(IMAGES))
def test_laplacian_variance_within_1e12_relative(name):
    img = IMAGES[name]
    new = _outcome(lambda: ImageCache(img).laplacian_variance)
    old = _outcome(lambda: ref.OldImageCache(img).laplacian_variance)
    assert new[0] == old[0]
    if old[0] != 'ok':
        assert new[1] is old[1]
        return
    assert isinstance(new[1], float)
    assert abs(new[1] - old[1]) <= 1e-12 * abs(old[1])


def test_laplacian_variance_of_a_flat_image_is_exactly_zero():
    assert ImageCache(IMAGES['const_128']).laplacian_variance == 0.0


def test_laplacian_variance_is_a_python_float():
    assert type(ImageCache(IMAGES['noise_64x64']).laplacian_variance) is float


def test_laplacian_variance_does_not_saturate_on_the_hardest_pattern():
    """CV_16S holds +-1020 at most for a 3x3 Laplacian of uint8; a checkerboard reaches it."""
    img = IMAGES['checkerboard']
    new = ImageCache(img).laplacian_variance
    old = ref.OldImageCache(img).laplacian_variance
    assert old > 0
    assert abs(new - old) <= 1e-12 * old


@pytest.mark.parametrize('name', ['noise_64x64', 'noise_480x640', 'gradient_h', 'checkerboard', 'sparse_levels'])
def test_sharpness_data_is_equal_after_its_own_rounding(name):
    img = IMAGES[name]
    cache, old_cache = ImageCache(img), ref.OldImageCache(img)
    for new, old in (
        (TechnicalAnalyzer.get_sharpness_data(img, cache=cache), ref.old_sharpness_data(img, cache=old_cache)),
        (TechnicalAnalyzer.get_sharpness_data(img), ref.old_sharpness_data(img)),
    ):
        assert round(new['normalized'], 2) == round(old['normalized'], 2)
        assert abs(new['raw_variance'] - old['raw_variance']) <= 1e-12 * max(old['raw_variance'], 1e-12)
    assert round(TechnicalAnalyzer.get_sharpness_score(img), 2) == round(ref.old_sharpness_score(img), 2)


def test_cache_and_no_cache_paths_agree_exactly():
    """One implementation behind two entry points: the no-cache path must give the cache path's value."""
    for name in ('noise_480x640', 'gradient_v', 'sparse_levels', 'two_tone_5_250'):
        img = IMAGES[name]
        cache = ImageCache(img)
        assert_identical(TechnicalAnalyzer.get_dynamic_range(img), TechnicalAnalyzer.get_dynamic_range(img, cache=cache))
        assert_identical(TechnicalAnalyzer.get_noise_estimate(img), TechnicalAnalyzer.get_noise_estimate(img, cache=cache))
        assert_identical(TechnicalAnalyzer.get_contrast_score(img), TechnicalAnalyzer.get_contrast_score(img, cache=cache))
        assert TechnicalAnalyzer.get_sharpness_data(img)['raw_variance'] == cache.laplacian_variance
        assert TechnicalAnalyzer.get_sharpness_score(img) == pytest.approx(
            TechnicalAnalyzer.get_sharpness_data(img)['normalized'], abs=0, rel=0)


def test_cache_slots_and_size_are_unchanged():
    cache = ImageCache(IMAGES['noise_37x53'])
    assert (cache.height, cache.width) == (37, 53)
    assert cache.gray.shape == (37, 53) and cache.gray.dtype == np.uint8
    assert cache.hsv.shape == (37, 53, 3)
    with pytest.raises(AttributeError):
        cache.extra = 1


# ----------------------------------------------------------------------------------------------
# structural: no float64 full-frame copies, no np.percentile / np.std, 16-bit kernels
# ----------------------------------------------------------------------------------------------

def test_replaced_metrics_do_not_call_np_percentile_or_np_std(monkeypatch):
    img = IMAGES['noise_480x640']
    cache = ImageCache(img)

    def boom(*args, **kwargs):
        raise AssertionError('full-frame numpy statistic called')

    monkeypatch.setattr(np, 'percentile', boom)
    monkeypatch.setattr(np, 'std', boom)
    TechnicalAnalyzer.get_dynamic_range(img, cache=cache)
    TechnicalAnalyzer.get_contrast_score(img, cache=cache)
    TechnicalAnalyzer.get_dynamic_range(img)
    TechnicalAnalyzer.get_contrast_score(img)


def test_noise_and_laplacian_use_16_bit_signed_outputs(monkeypatch):
    seen = {}
    real_filter2d, real_laplacian = cv2.filter2D, cv2.Laplacian

    def spy_filter2d(src, ddepth, *args, **kwargs):
        seen['filter2D'] = ddepth
        return real_filter2d(src, ddepth, *args, **kwargs)

    def spy_laplacian(src, ddepth, *args, **kwargs):
        seen['Laplacian'] = ddepth
        return real_laplacian(src, ddepth, *args, **kwargs)

    img = IMAGES['noise_64x64']
    monkeypatch.setattr(cv2, 'filter2D', spy_filter2d)
    monkeypatch.setattr(cv2, 'Laplacian', spy_laplacian)
    TechnicalAnalyzer.get_noise_estimate(img, cache=ImageCache(img))
    assert seen == {'filter2D': cv2.CV_16S, 'Laplacian': cv2.CV_16S}


def _peak_bytes(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_replaced_metrics_allocate_no_full_frame_float64_buffers():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (1500, 2000, 3), dtype=np.uint8)
    pixels = img.shape[0] * img.shape[1]
    cache = ImageCache(img)
    # The old paths peaked at 16-24 bytes per pixel (float64 copy + float64 filter output + abs).
    assert _peak_bytes(lambda: TechnicalAnalyzer.get_dynamic_range(img, cache=cache)) < 0.25 * pixels
    assert _peak_bytes(lambda: TechnicalAnalyzer.get_contrast_score(img, cache=cache)) < 0.25 * pixels
    assert _peak_bytes(lambda: TechnicalAnalyzer.get_noise_estimate(img, cache=cache)) < 3 * pixels
    # Reference peaks prove the probe itself can see a float64 copy.
    old_cache = ref.OldImageCache(img)
    assert _peak_bytes(lambda: ref.old_contrast_score(img, cache=old_cache)) > 8 * pixels
