"""Tests for RAW decode concurrency, timeout, and the parallel chunk loader."""

import sqlite3
import sys
import threading
import time
import types
from pathlib import Path
from unittest import mock

import tracemalloc

import numpy as np
import pytest
from PIL import Image

from utils import image_loading, system_memory
from utils.image_loading import (
    bgr_to_gray, configure_raw_decoding, load_image_from_path, pil_to_bgr,
)
from utils.system_memory import EffectiveMemory

GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _reset_decode_state():
    """Restore module decode state after each test."""
    yield
    image_loading._abandoned_decodes = 0
    configure_raw_decoding(concurrency=image_loading._auto_decode_concurrency(),
                           timeout_seconds=0)
    image_loading._hdr_pq_tonemap_settings = None


def _make_jpegs(tmp_path, count=3):
    paths = []
    for i in range(count):
        arr = np.full((32, 48, 3), i * 60 + 20, dtype=np.uint8)
        arr[:, : (i + 1) * 10] = 255 - i * 40
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(arr).save(p, quality=95)
        paths.append(str(p))
    return paths


class TestConfigureRawDecoding:
    def test_concurrency_one_serializes(self):
        configure_raw_decoding(concurrency=1)
        assert image_loading._raw_semaphore._value == 1

    def test_auto_concurrency_narrowed_by_tight_memory(self, monkeypatch):
        monkeypatch.setattr(image_loading.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(
            system_memory, "effective_memory",
            lambda: EffectiveMemory(total=8 * GIB, used=5 * GIB, available=3 * GIB, percent=62.5),
        )
        assert image_loading._auto_decode_concurrency() == 1

    def test_auto_concurrency_reaches_cpu_ceiling_when_roomy(self, monkeypatch):
        monkeypatch.setattr(image_loading.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(
            system_memory, "effective_memory",
            lambda: EffectiveMemory(total=128 * GIB, used=8 * GIB, available=120 * GIB, percent=6.25),
        )
        assert image_loading._auto_decode_concurrency() == 4

    def test_zero_keeps_current_concurrency(self):
        configure_raw_decoding(concurrency=3)
        configure_raw_decoding(concurrency=0, timeout_seconds=5)
        assert image_loading._decode_concurrency == 3
        assert image_loading._decode_timeout == 5.0


class TestColourPlaneHelpers:
    """``pil_to_bgr`` / ``bgr_to_gray`` are the one expression every rebuild of the planes uses."""

    @pytest.mark.parametrize("size", [(1, 1), (7, 3), (48, 32), (3, 40)])
    def test_pil_to_bgr_is_a_channel_reversal(self, size):
        rng = np.random.default_rng(size[0] * 100 + size[1])
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        bgr = pil_to_bgr(Image.fromarray(arr))
        assert bgr.dtype == np.uint8 and bgr.shape == arr.shape
        assert np.array_equal(bgr, arr[:, :, ::-1])

    def test_bgr_to_gray_matches_opencv(self):
        import cv2
        rng = np.random.default_rng(5)
        bgr = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        gray = bgr_to_gray(bgr)
        assert gray.shape == (20, 30)
        assert np.array_equal(gray, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

    def test_load_image_from_path_returns_the_helper_result(self, tmp_path):
        path = _make_jpegs(tmp_path, 1)[0]
        pil_img, img_cv = load_image_from_path(path)
        assert np.array_equal(img_cv, pil_to_bgr(pil_img))


class TestJpegLoading:
    def test_jpeg_loads_without_semaphore(self, tmp_path):
        path = _make_jpegs(tmp_path, 1)[0]
        pil_img, img_cv = load_image_from_path(path)
        assert pil_img is not None
        assert img_cv.shape == (32, 48, 3)

    def test_missing_file_returns_none_tuple(self, tmp_path):
        pil_img, img_cv = load_image_from_path(str(tmp_path / "nope.jpg"))
        assert pil_img is None and img_cv is None

    def test_concurrent_jpeg_loads_match_sequential(self, tmp_path):
        paths = _make_jpegs(tmp_path, 4)
        reference = {p: load_image_from_path(p)[1] for p in paths}
        results = {}
        errors = []

        def _load(p):
            try:
                results[p] = load_image_from_path(p)[1]
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=_load, args=(p,)) for p in paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        for p in paths:
            assert np.array_equal(results[p], reference[p])


class _StubRaw:
    """rawpy.imread() stand-in whose postprocess sleeps then returns pixels."""

    def __init__(self, delay):
        self.delay = delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_thumb(self):
        raise RuntimeError("no thumb")

    def postprocess(self, **kwargs):
        time.sleep(self.delay)
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _stub_rawpy(delay):
    stub = types.ModuleType("rawpy")
    stub.imread = lambda path: _StubRaw(delay)
    stub.ThumbFormat = types.SimpleNamespace(JPEG="jpeg")
    stub.ColorSpace = types.SimpleNamespace(sRGB="srgb")
    return stub


class TestDecodeTimeout:
    def test_timeout_returns_none_tuple(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "slow.dng"
        raw_path.write_bytes(b"fake")
        monkeypatch.setitem(sys.modules, "rawpy", _stub_rawpy(delay=1.5))
        image_loading._abandoned_decodes = 0
        configure_raw_decoding(concurrency=2, timeout_seconds=0.2)

        start = time.time()
        pil_img, img_cv = load_image_from_path(str(raw_path))
        elapsed = time.time() - start

        assert pil_img is None and img_cv is None
        assert elapsed < 1.0
        assert image_loading._abandoned_decodes == 1

    def test_budget_exhaustion_raises(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "slow.nef"
        raw_path.write_bytes(b"fake")
        monkeypatch.setitem(sys.modules, "rawpy", _stub_rawpy(delay=1.5))
        image_loading._abandoned_decodes = 0
        configure_raw_decoding(concurrency=2, timeout_seconds=0.1)

        for _ in range(image_loading._ABANDON_BUDGET):
            assert load_image_from_path(str(raw_path)) == (None, None)
        with pytest.raises(RuntimeError, match="storage likely stalled"):
            load_image_from_path(str(raw_path))

    def test_queue_wait_excluded_from_timeout(self, tmp_path, monkeypatch):
        # The timeout has to sit between one decode and two: below _DECODE it would
        # abandon the decode this test needs to succeed, above 2x it would pass even
        # if the queue wait were wrongly counted, proving nothing. Within that band
        # the only slack the decode has against scheduling jitter is
        # _TIMEOUT - _DECODE, so the pair is scaled to leave half a second of it --
        # at 0.3/0.5 the margin was 0.2s and the test failed under full-suite load.
        _DECODE = 1.0
        _TIMEOUT = 1.5
        raw_path = tmp_path / "queued.cr2"
        raw_path.write_bytes(b"fake")
        monkeypatch.setitem(sys.modules, "rawpy", _stub_rawpy(delay=_DECODE))
        image_loading._abandoned_decodes = 0
        configure_raw_decoding(concurrency=1, timeout_seconds=_TIMEOUT)

        results = {}

        def _load(key):
            results[key] = load_image_from_path(str(raw_path))

        threads = [threading.Thread(target=_load, args=(k,)) for k in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["a"][0] is not None and results["b"][0] is not None
        assert image_loading._abandoned_decodes == 0

    def test_fast_decode_succeeds_within_timeout(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "fast.arw"
        raw_path.write_bytes(b"fake")
        monkeypatch.setitem(sys.modules, "rawpy", _stub_rawpy(delay=0.0))
        configure_raw_decoding(concurrency=2, timeout_seconds=5)

        pil_img, img_cv = load_image_from_path(str(raw_path))
        assert pil_img is not None
        assert img_cv.shape == (4, 4, 3)


class TestMultiPassParallelLoader:
    def _make_processor(self, load_workers):
        # The parallel loader itself is torch-free, but constructing the
        # processor runs multi_pass._ensure_imports(), which eagerly imports
        # torch. CI installs no torch, so skip there; runs locally where torch
        # is present.
        pytest.importorskip("torch")
        from processing.multi_pass import ChunkedMultiPassProcessor
        scorer = mock.MagicMock()
        scorer.config.get_exposure_settings.return_value = {}
        scorer.config.get_monochrome_settings.return_value = {}
        scorer.get_exif_data.return_value = {"iso": 100}
        model_manager = mock.MagicMock()
        model_manager.detect_vram.return_value = 0
        config = {"processing": {"load_workers": load_workers}}
        return ChunkedMultiPassProcessor(scorer, model_manager, config)

    def test_parallel_output_matches_sequential(self, tmp_path):
        paths = _make_jpegs(tmp_path, 4)
        sequential = self._make_processor(1)._load_images(paths)
        parallel = self._make_processor(4)._load_images(paths)

        assert len(sequential) == len(parallel) == len(paths)
        assert list(parallel.keys()) == list(sequential.keys()) == paths
        for path in paths:
            assert parallel[path]["phash"] == sequential[path]["phash"]
            assert parallel[path]["exif"] is not None
            assert (
                parallel[path]["sharpness"]["raw_variance"]
                == sequential[path]["sharpness"]["raw_variance"]
            )

    def test_failed_image_skipped(self, tmp_path):
        paths = _make_jpegs(tmp_path, 2)
        bad = str(tmp_path / "corrupt.jpg")
        with open(bad, "wb") as f:
            f.write(b"not an image")
        images = self._make_processor(4)._load_images(paths + [bad])
        assert len(images) == len(paths)
        assert set(images.keys()) == set(paths)

    def test_load_workers_capped_at_eight(self):
        processor = self._make_processor(32)
        assert processor.load_workers == 8


class TestFilterUnscannedPaths:
    @pytest.fixture()
    def scorer_db(self, tmp_path):
        from db.schema import init_database
        from processing.scorer import Facet
        db_path = str(tmp_path / "scan.db")
        init_database(db_path)
        conn = sqlite3.connect(db_path)
        conn.executemany(
            "INSERT INTO photos (path, filename) VALUES (?, ?)",
            [(f"/lib/p{i}.jpg", f"p{i}.jpg") for i in range(25)],
        )
        conn.commit()
        conn.close()
        return Facet(db_path=db_path, lightweight=True)

    def test_filters_known_paths(self, scorer_db):
        candidates = [f"/lib/p{i}.jpg" for i in range(20)] + ["/new/a.jpg", "/new/b.jpg"]
        result = scorer_db.filter_unscanned_paths(candidates)
        assert result == {"/new/a.jpg", "/new/b.jpg"}

    def test_chunk_boundaries(self, scorer_db):
        candidates = [f"/lib/p{i}.jpg" for i in range(25)] + ["/new/x.jpg"]
        result = scorer_db.filter_unscanned_paths(candidates, chunk=10)
        assert result == {"/new/x.jpg"}

    def test_empty_input(self, scorer_db):
        assert scorer_db.filter_unscanned_paths([]) == set()

    def test_all_new(self, scorer_db):
        candidates = ["/other/1.jpg", "/other/2.jpg"]
        assert scorer_db.filter_unscanned_paths(candidates, chunk=1) == set(candidates)


class TestExifPrefetch:
    def _make_processor(self):
        from processing.batch_processor import BatchProcessor
        scorer = mock.MagicMock()
        return BatchProcessor(scorer, batch_size=4, num_workers=2)

    def test_cache_hit_avoids_sync_fetch(self, monkeypatch):
        import exiftool
        processor = self._make_processor()
        from pathlib import Path
        resolved = str(Path("/x/a.jpg").resolve())
        processor._exif_cache[resolved] = {"iso": 200}
        sync_calls = []
        monkeypatch.setattr(
            exiftool, "get_exif_batch",
            lambda paths, **kw: sync_calls.append(paths) or {},
        )
        result = processor._get_batch_exif(["/x/a.jpg"])
        assert result[resolved] == {"iso": 200}
        assert sync_calls == []
        assert processor._exif_cache == {}

    def test_miss_falls_back_to_sync_fetch(self, monkeypatch):
        import exiftool
        processor = self._make_processor()
        from pathlib import Path
        resolved = str(Path("/x/b.jpg").resolve())
        monkeypatch.setattr(
            exiftool, "get_exif_batch",
            lambda paths, **kw: {resolved: {"iso": 400}},
        )
        result = processor._get_batch_exif(["/x/b.jpg"])
        assert result[resolved] == {"iso": 400}

    def test_prefetch_thread_fills_cache(self, monkeypatch):
        import exiftool
        processor = self._make_processor()
        from pathlib import Path
        resolved = str(Path("/x/c.jpg").resolve())
        monkeypatch.setattr(
            exiftool, "get_exif_batch",
            lambda paths, **kw: {resolved: {"iso": 800}},
        )
        processor._start_exif_prefetch(["/x/c.jpg"])
        processor._exif_prefetch_thread.join(timeout=5)
        assert processor._exif_cache[resolved] == {"iso": 800}

    def test_prefetch_disabled_is_noop(self):
        processor = self._make_processor()
        processor.exif_prefetch_enabled = False
        processor._start_exif_prefetch(["/x/d.jpg"])
        assert processor._exif_prefetch_thread is None


# --- Canon .HIF HDR PQ -> sRGB tone mapping -----------------------------------

def test_scannable_extensions_include_hif():
    # SCANNABLE_IMAGE_EXTENSIONS is the single allow-list consumed by both the
    # scan collector (facet.py) and watch mode (processing/watcher.py).
    # pillow-heif is a soft dependency: the CI test job installs the minimal
    # dependency set without it, so HEIF_EXTENSIONS is empty and .HIF is neither
    # scanned nor watched. Assert the correct behaviour for both states rather
    # than assuming the decoder is present.
    if image_loading._heif_available:
        assert '.hif' in image_loading.HEIF_EXTENSIONS
        assert '.hif' in image_loading.SCANNABLE_IMAGE_EXTENSIONS
    else:
        assert image_loading.HEIF_EXTENSIONS == set()
        assert '.hif' not in image_loading.SCANNABLE_IMAGE_EXTENSIONS
    # AVIF mirrors the HEIF gating exactly: decodable only when this Pillow
    # build has the AVIF codec (native only from Pillow >= 11.3).
    if image_loading._avif_available:
        assert '.avif' in image_loading.AVIF_EXTENSIONS
        assert '.avif' in image_loading.SCANNABLE_IMAGE_EXTENSIONS
    else:
        assert image_loading.AVIF_EXTENSIONS == set()
        assert '.avif' not in image_loading.SCANNABLE_IMAGE_EXTENSIONS
    assert image_loading.SCANNABLE_IMAGE_EXTENSIONS == (
        image_loading.JPEG_EXTENSIONS
        | image_loading.HEIF_EXTENSIONS
        | image_loading.RAW_EXTENSIONS
        | image_loading.PNG_EXTENSIONS
        | image_loading.GIF_EXTENSIONS
        | image_loading.WEBP_EXTENSIONS
        | image_loading.BMP_EXTENSIONS
        | image_loading.TIFF_EXTENSIONS
        | image_loading.AVIF_EXTENSIONS
    )


def test_pq_eotf_reference_points():
    """SMPTE ST 2084:2014 EOTF: S=1 -> 10000 nits, S=0 -> 0, monotonic."""
    assert image_loading._pq_eotf(1.0) == pytest.approx(10000.0, abs=1e-3)
    assert image_loading._pq_eotf(0.0) == pytest.approx(0.0, abs=1e-6)
    s = np.linspace(0.0, 1.0, 21)
    L = image_loading._pq_eotf(s)
    assert np.all(np.diff(L) >= 0)  # monotonic non-decreasing


def test_heif_is_pq_gates_on_transfer_not_extension():
    """Gating keys off NCLX transfer==16 only; SDR/HLG/missing-NCLX all False."""
    class _Img:
        def __init__(self, info):
            self.info = info

    assert image_loading._heif_is_pq(_Img(
        {'nclx_profile': {'transfer_characteristics': 16}})) is True   # PQ
    assert image_loading._heif_is_pq(_Img(
        {'nclx_profile': {'transfer_characteristics': 13}})) is False  # SDR sRGB
    assert image_loading._heif_is_pq(_Img(
        {'nclx_profile': {'transfer_characteristics': 18}})) is False  # HLG
    assert image_loading._heif_is_pq(_Img({})) is False              # no NCLX
    assert image_loading._heif_is_pq(_Img({'nclx_profile': None})) is False


def test_srgb_oetf_reference_points():
    """IEC 61966-2-1 sRGB OETF endpoints: linear 0 -> 0, 1 -> 1."""
    assert image_loading._srgb_oetf(0.0) == pytest.approx(0.0)
    assert image_loading._srgb_oetf(1.0) == pytest.approx(1.0)


def test_srgb_oetf_keeps_the_linear_toe():
    """Below the 0.0031308 threshold the curve is the 12.92 linear segment.

    The in-place form computes the toe mask and the scaled toe BEFORE np.power
    overwrites the buffer; getting that order wrong returns the shoulder for
    every toe pixel, which no endpoint check would catch.
    """
    toe = np.array([0.0, 0.001, 0.003], dtype=np.float32)
    assert image_loading._srgb_oetf(toe) == pytest.approx(toe * 12.92, rel=1e-5)


def test_tonemap_pq_output_shape_and_range():
    """tone map returns an 8-bit RGB PIL image with pixels in [0,255]."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    out = image_loading._tonemap_pq_to_srgb(Image.fromarray(arr, 'RGB'))
    assert out.mode == 'RGB'
    assert out.size == (24, 16)
    a = np.asarray(out)
    assert a.dtype == np.uint8
    assert a.min() >= 0 and a.max() <= 255


def test_open_nonraw_image_tonemaps_only_pq():
    """open_nonraw_image tone-maps PQ HEIF but passes SDR / no-NCLX through."""
    dark = np.full((8, 8, 3), 60, dtype=np.uint8)

    def _img(transfer):
        im = Image.fromarray(dark, 'RGB')
        if transfer is not None:
            im.info['nclx_profile'] = {'transfer_characteristics': transfer}
        return im

    pq, sdr, plain = _img(16), _img(13), _img(None)
    with mock.patch.object(Image, 'open', side_effect=[pq, sdr, plain]):
        out_pq = image_loading.open_nonraw_image('pq.heif')
        out_sdr = image_loading.open_nonraw_image('sdr.heif')
        out_plain = image_loading.open_nonraw_image('plain.jpg')

    # PQ dark frame is lifted by tone mapping; SDR and no-NCLX pass through.
    assert np.asarray(out_pq).mean() > np.asarray(out_sdr).mean()
    assert np.asarray(out_sdr).mean() == 60.0
    assert np.asarray(out_plain).mean() == 60.0


# --- HDR PQ tone mapping: configurable white point / highlight roll-off ------

def _pq_oetf(nits):
    """Inverse of image_loading._pq_eotf: absolute nits -> PQ signal [0,1]."""
    L = np.asarray(nits, dtype=np.float64) / image_loading._PQ_PEAK_NITS
    Lm = L ** image_loading._PQ_M1
    return ((image_loading._PQ_C1 + image_loading._PQ_C2 * Lm)
            / (1.0 + image_loading._PQ_C3 * Lm)) ** image_loading._PQ_M2


def _full_settings(**overrides):
    """A complete hdr_pq_tonemap settings dict with optional overrides."""
    from config.scoring_config import merge_hdr_pq_tonemap_settings
    return merge_hdr_pq_tonemap_settings(overrides)


def test_hdr_pq_default_settings():
    s = image_loading.configure_hdr_pq_tonemap_profile()
    assert s['enabled'] is True
    assert s['method'] == 'hable'
    assert s['chroma_preserve'] == 'per_channel'
    wp = s['white_point']
    assert wp['mode'] == 'percentile'
    assert wp['percentile'] == 99.99
    assert wp['min_nits'] == 100.0
    assert wp['max_nits'] == 1200.0


def test_hand_edited_null_white_point_leaf_falls_back_to_its_default():
    """A null LEAF must not reach the decode path, as a null block already cannot.

    The nested block is merged key-by-key, so a null under white_point replaces
    a float the tone map multiplies. The decode path reads the config without
    re-validating it, so the TypeError lands per photo and surfaces as the still
    silently missing from the scan rather than as a config error.
    """
    s = image_loading.configure_hdr_pq_tonemap_profile(
        {'white_point': {'min_nits': None, 'percentile': 'not a number'}})
    assert s['white_point']['min_nits'] == 100.0
    assert s['white_point']['percentile'] == 99.99
    # A string that IS a number is still usable, so it is coerced, not dropped.
    s = image_loading.configure_hdr_pq_tonemap_profile({'white_point': {'max_nits': '900'}})
    assert s['white_point']['max_nits'] == 900.0


def test_configure_hdr_pq_profile_merges_nested_white_point():
    # Overriding one white_point sub-key keeps the others at defaults.
    s = image_loading.configure_hdr_pq_tonemap_profile(
        {'white_point': {'percentile': 99.9}, 'method': 'clip'})
    assert s['method'] == 'clip'
    assert s['white_point']['percentile'] == 99.9
    assert s['white_point']['mode'] == 'percentile'      # untouched default
    assert s['white_point']['max_nits'] == 1200.0        # untouched default
    assert s['chroma_preserve'] == 'per_channel'         # untouched default


def test_configure_hdr_pq_profile_tolerates_malformed_block():
    # The decode path reads config with validate=False, so a hand-edited file
    # must not crash the merge: non-dict blocks fall back to defaults, a
    # non-dict white_point is ignored (valid siblings still apply), and unknown
    # keys at either level are discarded.
    from config.scoring_config import (
        HDR_PQ_TONEMAP_DEFAULTS,
        merge_hdr_pq_tonemap_settings,
    )
    for bad in (None, "x", 42, []):
        assert merge_hdr_pq_tonemap_settings(bad) == HDR_PQ_TONEMAP_DEFAULTS
    s = merge_hdr_pq_tonemap_settings(
        {'white_point': None, 'method': 'clip'})
    assert s['white_point'] == HDR_PQ_TONEMAP_DEFAULTS['white_point']
    assert s['method'] == 'clip'
    s2 = merge_hdr_pq_tonemap_settings(
        {'bogus': 1, 'white_point': {'bogus': 2}})
    assert 'bogus' not in s2 and 'bogus' not in s2['white_point']


def test_hdr_pq_white_point_modes_and_clamps():
    # Brightest-channel field: values 20 / 300 / 800 nits.
    nits = np.zeros((4, 4, 3), dtype=np.float32)
    nits[:2] = 20.0
    nits[2:, :2] = 300.0
    nits[2:, 2:] = 800.0
    wp = image_loading._hdr_pq_white_point
    assert wp(nits, {'mode': 'percentile', 'percentile': 99.99,
                     'min_nits': 100, 'max_nits': 1200}) == pytest.approx(800.0)
    assert wp(nits, {'mode': 'max', 'min_nits': 100}) == pytest.approx(800.0)
    assert wp(nits, {'mode': 'fixed', 'fixed_nits': 1000.0,
                     'min_nits': 100}) == pytest.approx(1000.0)
    # Upper clamp: a very bright percentile is capped at max_nits.
    hot = np.full((4, 4, 3), 4000.0, dtype=np.float32)
    assert wp(hot, {'mode': 'percentile', 'percentile': 99.99,
                    'min_nits': 100, 'max_nits': 1200}) == pytest.approx(1200.0)
    # Lower clamp: a dark frame never sets white below the SDR floor.
    dark = np.full((4, 4, 3), 5.0, dtype=np.float32)
    assert wp(dark, {'mode': 'percentile', 'percentile': 99.99,
                     'min_nits': 100, 'max_nits': 1200}) == pytest.approx(100.0)


def test_hdr_pq_white_point_percentile_ignores_hot_pixels():
    # A night scene at ~11 nits with one 3402-nit specular pixel. Percentile
    # keeps the white at the floor; max mode chases the hot pixel and would
    # darken the whole frame.
    night = np.full((100, 100, 3), 11.0, dtype=np.float32)
    night[0, 0] = 3402.0
    wp = image_loading._hdr_pq_white_point
    assert wp(night, {'mode': 'percentile', 'percentile': 99.99,
                      'min_nits': 100, 'max_nits': 1200}) == pytest.approx(100.0)
    assert wp(night, {'mode': 'max', 'min_nits': 100}) == pytest.approx(3402.0)


def test_hdr_pq_tone_map_preserves_highlight_texture():
    # Regression: normalising at a fixed 100-nit white clipped 300 and 800 nit
    # regions to identical pure white. The per-image white keeps them apart.
    nits = np.zeros((4, 4, 3), dtype=np.float32)
    nits[:2] = 20.0
    nits[2:, :2] = 300.0
    nits[2:, 2:] = 800.0
    cfg = _full_settings()
    white = image_loading._hdr_pq_white_point(nits, cfg['white_point'])
    out = image_loading._hdr_pq_tone_map(nits, white, cfg)
    assert out[2, 0].mean() < out[2, 2].mean()      # 300 nits darker than 800
    assert out[2, 2].mean() == pytest.approx(1.0)    # white maps to 1
    assert out[0, 0].mean() < out[2, 0].mean()       # shadows still darker


def test_hdr_pq_tone_map_methods_and_chroma_preserve_in_range():
    rng = np.random.default_rng(1)
    nits = (rng.random((16, 16, 3)) * 1500.0).astype(np.float32) + 1.0
    for method in ('hable', 'clip'):
        for chroma in ('per_channel', 'max_channel'):
            cfg = _full_settings(method=method, chroma_preserve=chroma)
            white = image_loading._hdr_pq_white_point(nits, cfg['white_point'])
            out = image_loading._hdr_pq_tone_map(nits, white, cfg)
            assert np.isfinite(out).all()
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_tonemap_pq_disabled_returns_input_unchanged():
    im = Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8), 'RGB')
    out = image_loading._tonemap_pq_to_srgb(im, {'enabled': False})
    assert out is im


def test_tonemap_pq_end_to_end_bright_gradient_keeps_steps():
    # A PQ gradient spanning ~150-1000 nits must not collapse to flat white.
    cols = 64
    nits = np.tile(np.linspace(150.0, 1000.0, cols, dtype=np.float32), (8, 1))
    nits = np.repeat(nits[:, :, None], 3, axis=2)
    signal = _pq_oetf(nits).astype(np.float32)
    arr = np.clip(signal * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = np.asarray(image_loading._tonemap_pq_to_srgb(
        Image.fromarray(arr, 'RGB'), _full_settings()))
    col_means = out.mean(axis=(0, 2))
    assert np.all(np.diff(col_means) >= -1e-6)          # monotonic non-decreasing
    assert (col_means < 255).sum() > cols // 2          # most steps not clipped
    assert col_means.max() >= 250                        # bright end reaches white


def test_non_pq_passes_through_regardless_of_enabled():
    # Gating is on NCLX transfer, not the enabled switch: SDR HEIF is never
    # tone-mapped whether the HDR block is enabled or not.
    def _open_sdr():
        sdr = Image.fromarray(np.full((4, 4, 3), 60, dtype=np.uint8), 'RGB')
        sdr.info['nclx_profile'] = {'transfer_characteristics': 13}
        with mock.patch.object(Image, 'open', return_value=sdr):
            return np.asarray(image_loading.open_nonraw_image('sdr.heif')).mean()

    image_loading.configure_hdr_pq_tonemap_profile({'enabled': True})
    assert _open_sdr() == 60.0
    image_loading.configure_hdr_pq_tonemap_profile({'enabled': False})
    assert _open_sdr() == 60.0

@pytest.mark.parametrize('mode', ['percentile', 'max', 'fixed'])
@pytest.mark.parametrize('shape', [(97, 61), (1, 40), (40, 1), (64, 64)])
def test_banded_white_point_matches_the_whole_frame_form(shape, mode):
    """Banding the white point is an allocation strategy, not a different number.

    Three rows per band divides none of these shapes evenly, which is the case
    an equal-split implementation gets wrong. The clamps are opened right out on
    purpose: at the shipped 100/1200 nit bounds random PQ codes saturate
    max_nits, and the two forms then agree on a constant instead of on the value
    they computed.
    """
    rng = np.random.default_rng(3)
    src = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    wp = dict(_full_settings()['white_point'],
              mode=mode, min_nits=1.0, max_nits=1e6, percentile=99.0)
    whole = image_loading._hdr_pq_white_point(image_loading._band_nits(src), wp)
    assert image_loading._banded_white_point(src, wp, 3) == pytest.approx(whole, rel=1e-6)


def test_pq_tonemap_banding_does_not_change_the_result():
    """Row banding is an allocation strategy, not a different transform."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (97, 61, 3), dtype=np.uint8)
    settings = _full_settings()
    whole = np.asarray(image_loading._tonemap_pq_to_srgb(Image.fromarray(arr, 'RGB'), settings))
    with mock.patch.object(image_loading, '_TONEMAP_BAND_PIXELS', 61 * 8):
        banded = np.asarray(image_loading._tonemap_pq_to_srgb(Image.fromarray(arr, 'RGB'), settings))
    assert np.array_equal(whole, banded)


def test_pq_eotf_lut_is_exact_over_every_8bit_code():
    """The table is a tabulation, not an approximation.

    The decoder hands back 8-bit RGB, so the EOTF has exactly 256 possible
    inputs per channel and the lookup must equal the closed form on all of
    them -- otherwise the "exact" in its comment is a claim, not a fact.
    """
    codes = np.arange(256, dtype=np.float32) / 255.0
    assert np.array_equal(image_loading._PQ_EOTF_LUT, image_loading._pq_eotf(codes).astype(np.float32))


def test_tonemap_pq_float32_intermediates_are_band_bound():
    """Growing the frame must not grow the float32 working set.

    The peak itself is NOT frame-invariant and the name does not claim it is:
    the uint8 output and the per-pixel channel maxima are inherently frame-sized,
    which measures ~6.6 bytes/pixel. What must not scale is the float32
    intermediates -- the whole-array form built eight frame-sized copies, so a
    48 MP still peaked at 1327 MiB against this one's 107. The bound below is
    3 bytes/pixel of headroom over the measurement, which one more full-size
    float32 temporary (4 bytes/pixel) would exceed.
    """
    rng = np.random.default_rng(11)
    settings = _full_settings()

    def _peak(side):
        arr = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
        img = Image.fromarray(arr, 'RGB')
        tracemalloc.start()
        image_loading._tonemap_pq_to_srgb(img, settings)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak, arr.nbytes

    small_peak, small_bytes = _peak(512)
    large_peak, large_bytes = _peak(2048)

    # At 8 float32 copies the whole-array form grew ~29x the frame delta.
    assert large_peak - small_peak < 3 * (large_bytes - small_bytes), (
        f"{small_bytes / 2**20:.0f} MiB frame -> {small_peak / 2**20:.0f} MiB peak; "
        f"{large_bytes / 2**20:.0f} MiB frame -> {large_peak / 2**20:.0f} MiB peak")


# --- Real camera files --------------------------------------------------------
# See tests/fixtures/README.md for provenance and the exact NCLX values.

FIXTURES = Path(__file__).parent / 'fixtures'
CANON_PQ_HIF = FIXTURES / 'canon_eos_r8_hdr_pq.hif'
SONY_SDR_HIF = FIXTURES / 'sony_a7sm3_sdr.hif'
APPLE_GAINMAP_HEIC = FIXTURES / 'apple_iphone13pro_gainmap.heic'
GRADIENT_PQ_HIF = FIXTURES / 'pq_gradient_ramp.hif'

requires_heif = pytest.mark.skipif(
    not image_loading._heif_available, reason='pillow-heif not installed')


def _pure_white_pixels(rgb):
    return int((rgb == 255).all(axis=-1).sum())


@requires_heif
def test_real_canon_hif_reports_pq_through_its_nclx_profile():
    """A genuine in-camera HDR PQ still must be recognised from its metadata."""
    Image_, _ = image_loading._ensure_pil()
    with Image_.open(CANON_PQ_HIF) as img:
        nclx = img.info['nclx_profile']
        assert image_loading._heif_is_pq(img) is True
        assert nclx['color_primaries'] == 9          # BT.2020
        assert nclx['transfer_characteristics'] == 16  # SMPTE ST 2084 (PQ)
        # An HDR PQ HEIF need not carry BT.2020 non-constant luminance: this
        # Canon body writes matrix 1 (BT.709). Nothing may gate on that field.
        assert nclx['matrix_coefficients'] == 1


@requires_heif
def test_real_canon_hif_still_decodes_with_a_null_white_point_leaf():
    """The end of the path the merge guard protects.

    Asserted on the real PQ file rather than on the merge alone: what the null
    actually costs is this decode returning None, and load_image_from_path
    turning that into a dropped photo.
    """
    image_loading.configure_hdr_pq_tonemap_profile({'white_point': {'min_nits': None}})
    assert image_loading.open_nonraw_image(CANON_PQ_HIF) is not None


@requires_heif
def test_real_canon_hif_is_dark_when_decoded_as_plain_srgb():
    """Pins the bug: pillow-heif hands back PQ code values, not sRGB ones."""
    Image_, _ = image_loading._ensure_pil()
    with Image_.open(CANON_PQ_HIF) as img:
        decoded = np.asarray(img.convert('RGB'))
    assert decoded.mean() < 100          # measured 92.6/255
    assert np.percentile(decoded, 99) < 170  # measured 153
    assert _pure_white_pixels(decoded) == 0


@requires_heif
def test_real_canon_hif_tone_map_invents_no_clipping():
    """The regression guard for the tone curve's white point.

    The source frame has no pure-white pixel at all. Normalising the Hable
    operator at diffuse white instead of at the signal peak turned 21 648 of
    its 240 000 pixels pure white, which would have driven highlight_clipped
    and channel_clip_highlight_pct on a frame that clips nowhere.
    """
    Image_, _ = image_loading._ensure_pil()
    with Image_.open(CANON_PQ_HIF) as img:
        decoded = np.asarray(img.convert('RGB'))
    mapped = np.asarray(image_loading.open_nonraw_image(str(CANON_PQ_HIF)))

    assert _pure_white_pixels(mapped) == 0
    # The transform ran, and it opened the highlights up the scale.
    assert not np.array_equal(mapped, decoded)
    assert np.percentile(mapped, 99) > np.percentile(decoded, 99)


@requires_heif
def test_real_sony_sdr_hif_passes_through_untouched():
    """.HIF is neither Canon-only nor HDR: Sony writes SDR stills under it.

    An extension-based gate would tone-map this file. The NCLX transfer (13,
    sRGB) is what keeps it byte-identical to a plain decode.
    """
    Image_, _ = image_loading._ensure_pil()
    with Image_.open(SONY_SDR_HIF) as img:
        assert img.info['nclx_profile']['transfer_characteristics'] == 13
        assert image_loading._heif_is_pq(img) is False
        decoded = np.asarray(img.convert('RGB'))
    loaded = np.asarray(image_loading.open_nonraw_image(str(SONY_SDR_HIF)))
    assert np.array_equal(loaded, decoded)


@requires_heif
def test_real_apple_heic_has_no_nclx_box_at_all():
    """Apple HDR is a gain map, not PQ — and it ships no NCLX box whatsoever.

    The Sony fixture covers "NCLX present, transfer != 16". This one covers the
    other falsy branch of _heif_is_pq: `img.info` has no 'nclx_profile' key, so
    the guard must read it with .get() and not KeyError. An iPhone HDR still is
    an 8-bit SDR base image plus a `...aux:hdrgainmap` auxiliary, so there is
    nothing here to tone map.
    """
    Image_, _ = image_loading._ensure_pil()
    with Image_.open(APPLE_GAINMAP_HEIC) as img:
        assert 'nclx_profile' not in img.info
        assert image_loading._heif_is_pq(img) is False


@requires_heif
def test_real_apple_heic_passes_through_untouched():
    """The base image is already displayable SDR — tone mapping it would be a bug."""
    Image_, _ = image_loading._ensure_pil()
    with Image_.open(APPLE_GAINMAP_HEIC) as img:
        decoded = np.asarray(img.convert('RGB'))
    loaded = np.asarray(image_loading.open_nonraw_image(str(APPLE_GAINMAP_HEIC)))
    assert np.array_equal(loaded, decoded)


@requires_heif
@pytest.mark.parametrize(
    'fixture', [CANON_PQ_HIF, SONY_SDR_HIF, APPLE_GAINMAP_HEIC])
def test_real_hif_files_are_scannable_and_load_as_rgb(fixture):
    assert fixture.suffix.lower() in image_loading.SCANNABLE_IMAGE_EXTENSIONS
    pil_img, img_cv = image_loading.load_image_from_path(str(fixture))
    assert pil_img is not None and pil_img.mode == 'RGB'
    assert img_cv is not None and img_cv.shape[2] == 3


def _empty_interior_luma_bins(rgb, lo=1, hi=254):
    """Count zero-count 8-bit luma bins strictly inside the black/white ends.

    Posterisation from crushing a PQ frame to 8 bits before the EOTF shows up
    as gaps in the output luma histogram before it shows up anywhere else --
    exactly the comb the native-depth decode in issue #154 closes.
    """
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = np.clip(np.round(0.2126 * r + 0.7152 * g + 0.0722 * b), 0, 255).astype(np.int32)
    counts = np.bincount(y.ravel(), minlength=256)
    return int((counts[lo:hi + 1] == 0).sum())


@requires_heif
def test_pq_gradient_ramp_native_decode_has_no_posterisation_gaps():
    """The regression this issue exists to fix.

    tests/fixtures/pq_gradient_ramp.hif is a 1024-step, 10-bit PQ ramp, one
    code per column. Through the 8-bit decode path only ~130 of PQ's codes
    cover the SDR range, so a 1024-step ramp collapses several columns onto
    the same output luma and the histogram gets holes (measured 86). Decoded
    at its native 10-bit depth every step keeps its own code and the holes
    close (measured 0) -- see
    .claude/specs/issue-154-native-bit-depth-pq-decode.md.
    """
    Image_, _ = image_loading._ensure_pil()
    with Image_.open(GRADIENT_PQ_HIF) as img:
        eight_bit_pil = image_loading._tonemap_pq_to_srgb(img.convert('RGB'))
    eight_bit = np.asarray(eight_bit_pil)

    native_pil = image_loading.open_nonraw_image(str(GRADIENT_PQ_HIF))
    native = np.asarray(native_pil)

    assert eight_bit_pil.mode == 'RGB' and eight_bit.dtype == np.uint8
    assert native_pil.mode == 'RGB' and native.dtype == np.uint8
    assert eight_bit.shape == native.shape

    # The 8-bit path floors rather than pins the exact count: an encoder
    # revision of the fixture must not make this test brittle.
    assert _empty_interior_luma_bins(eight_bit) >= 40
    assert _empty_interior_luma_bins(native) == 0
