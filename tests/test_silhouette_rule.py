"""The face-aware silhouette rule (``utils.detection.detect_silhouette``) and its scan sites.

A silhouette is a dark patch (the largest detected face) against a clearly brighter
surround. The rule reads pixels and face boxes only: no tag, no face count, no whole-frame
histogram. The first half pins the rule itself on synthetic gray planes, the second half the
three scan sites that call it (single image, ``BatchProcessor``, multi-pass
``_compute_aggregates``).
"""

import warnings
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from utils import detect_silhouette

H = W = 200
BRIGHT = 204          # 0.80
FACE = [80, 80, 120, 120]


def _frame(surround=BRIGHT, h=H, w=W):
    return np.full((h, w), surround, dtype=np.uint8)


def _scene(core=13, surround=BRIGHT, box=FACE, h=H, w=W):
    """A uniform ``surround`` with ``core`` painted over the whole face box."""
    gray = _frame(surround, h, w)
    x1, y1, x2, y2 = box
    gray[max(0, y1):y2, max(0, x1):x2] = core
    return gray


def _face(*bbox):
    return {'bbox': list(bbox)}


class TestTheRule:
    def test_dark_face_in_a_bright_surround_is_a_silhouette(self):
        assert detect_silhouette(_scene(core=13), [_face(*FACE)]) == 1      # core 0.05, ring 0.80

    def test_a_brighter_face_is_not(self):
        assert detect_silhouette(_scene(core=51), [_face(*FACE)]) == 0      # core 0.20

    def test_core_threshold_is_strict_below_0_12(self):
        assert detect_silhouette(_scene(core=30), [_face(*FACE)]) == 1      # 30 / 255 = 0.1176
        assert detect_silhouette(_scene(core=31), [_face(*FACE)]) == 0      # 31 / 255 = 0.1216

    def test_dark_face_in_a_dark_scene_is_underexposure_not_a_silhouette(self):
        # core 0.05, surround 0.20: contrast 0.15 < 0.20
        assert detect_silhouette(_scene(core=13, surround=51), [_face(*FACE)]) == 0

    def test_contrast_threshold_is_strict_above_0_20(self):
        assert detect_silhouette(_scene(core=0, surround=52), [_face(*FACE)]) == 1    # 0.2039
        assert detect_silhouette(_scene(core=0, surround=51), [_face(*FACE)]) == 0    # 0.2000

    def test_only_the_core_of_the_box_is_judged(self):
        # A bright rim in the outer 20% of the box does not matter; a bright centre does.
        gray = _scene(core=204)
        gray[88:112, 88:112] = 13               # the central 60% of FACE
        assert detect_silhouette(gray, [_face(*FACE)]) == 1
        gray[95:105, 95:105] = 204              # a bright patch inside the core, mean > 0.12
        assert detect_silhouette(gray, [_face(*FACE)]) == 0

    @pytest.mark.parametrize('face_details', [None, [], [{}], [{'bbox': None}], [{'bbox': [1, 2, 3]}]])
    def test_no_usable_face_is_not_a_silhouette(self, face_details):
        assert detect_silhouette(_scene(core=0), face_details) == 0

    def test_no_gray_plane_is_not_a_silhouette(self):
        assert detect_silhouette(None, [_face(*FACE)]) == 0

    def test_a_core_under_2x2_px_is_not_judged(self):
        assert detect_silhouette(_scene(core=0, box=[100, 100, 101, 101]), [_face(100, 100, 101, 101)]) == 0
        assert detect_silhouette(_scene(core=0, box=[100, 100, 101, 140]), [_face(100, 100, 101, 140)]) == 0
        # a 2 px wide strip has a 2 px core and is judged; a 2x2 box has a 2x2 core but its
        # ring is 20 px, one short of the guard
        assert detect_silhouette(_scene(core=0, box=[100, 100, 102, 140]), [_face(100, 100, 102, 140)]) == 1
        assert detect_silhouette(_scene(core=0, box=[100, 100, 102, 102]), [_face(100, 100, 102, 102)]) == 0
        assert detect_silhouette(_scene(core=0, box=[100, 100, 103, 103]), [_face(100, 100, 103, 103)]) == 1

    def test_a_ring_under_21_px_is_not_judged(self):
        # 10 rows x 13 cols, the box spans the frame's full height and 10 of 13 columns: the
        # ring is one 10 px column, bright, and the face is black.
        gray = np.full((10, 13), BRIGHT, dtype=np.uint8)
        gray[:, :10] = 0
        assert detect_silhouette(gray, [_face(0, 0, 10, 10)]) == 0
        # the same face with room around it is a silhouette
        assert detect_silhouette(_scene(core=0, box=[0, 0, 10, 10]), [_face(0, 0, 10, 10)]) == 1
        # a box that fills the frame has no ring at all
        assert detect_silhouette(_frame(0, 10, 10), [_face(0, 0, 10, 10)]) == 0

    def test_only_the_largest_face_is_judged(self):
        small = [10, 10, 30, 30]
        big = [100, 60, 160, 140]
        # dark small face, bright big face: judged on the big one -> 0
        gray = _scene(core=204, box=big)
        gray[small[1]:small[3], small[0]:small[2]] = 0
        assert detect_silhouette(gray, [_face(*small), _face(*big)]) == 0
        assert detect_silhouette(gray, [_face(*big), _face(*small)]) == 0
        # dark big face, bright small face -> 1, whatever the order
        gray = _scene(core=0, box=big)
        assert detect_silhouette(gray, [_face(*small), _face(*big)]) == 1
        assert detect_silhouette(gray, [_face(*big), _face(*small)]) == 1

    def test_the_old_three_argument_form_is_gone(self):
        with pytest.raises(TypeError):
            detect_silhouette({'is_silhouette': 1}, 'portrait, silhouette', 1)
        with pytest.raises(TypeError):
            detect_silhouette(_scene(core=0), [_face(*FACE)], 'silhouette')

    def test_float_boxes_from_a_scaled_backfill_are_judged(self):
        assert detect_silhouette(_scene(core=0), [_face(80.4, 79.6, 119.7, 120.2)]) == 1


class TestBoxesOutsideTheFrame:
    """InsightFace boxes can be negative or run past the edge. They are clipped to the
    frame FIRST: a negative slice start would wrap and read the far side of the image."""

    @staticmethod
    def _unclipped_reference(gray, bbox):
        """The rule without the clip, as a naive slice would apply it (negative starts wrap)."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        g = gray / 255.0
        core = g[int(y1 + 0.2 * h):int(y2 - 0.2 * h), int(x1 + 0.2 * w):int(x2 - 0.2 * w)].mean()
        mask = np.zeros(g.shape, bool)
        mask[max(0, int(y1 - .75 * h)):int(y2 + .75 * h), max(0, int(x1 - .75 * w)):int(x2 + .75 * w)] = True
        mask[max(0, int(y1 - .2 * h)):int(y2 + .2 * h), max(0, int(x1 - .2 * w)):int(x2 + .2 * w)] = False
        return int(core < 0.12 and g[mask].mean() - core > 0.20)

    def test_a_box_with_a_negative_x1_does_not_read_the_far_edge(self):
        # Bright frame, dark band across the right 40 px. The box hangs off the left edge,
        # so the unclipped core slice [-38:-2] wraps onto that dark band.
        gray = _frame()
        gray[:, 160:] = 13
        bbox = [-50, 80, 10, 140]
        assert self._unclipped_reference(gray, bbox) == 1          # the fault, shown
        assert detect_silhouette(gray, [_face(*bbox)]) == 0

    def test_a_box_clipped_to_a_valid_dark_face_still_flags(self):
        gray = _frame()
        gray[60:140, 0:60] = 13                                    # the visible part of the face
        assert detect_silhouette(gray, [_face(-20, 60, 60, 140)]) == 1

    @pytest.mark.parametrize('bbox', [
        [300, 300, 350, 350],            # wholly outside
        [-80, -80, -10, -10],            # wholly outside, negative
        [80, -40, 120, 40],              # negative y1
        [160, 160, 260, 260],            # past W and H
        [-1000, -1000, 1000, 1000],      # swallows the frame
        [100, 100, 100, 140],            # zero width
        [120, 80, 80, 120],              # inverted
    ])
    def test_any_box_off_the_frame_is_judged_or_zero_without_warnings(self, bbox):
        for gray in (_scene(core=13), _frame(), np.zeros((H, W), np.uint8)):
            with warnings.catch_warnings():
                warnings.simplefilter('error')         # what ``pytest -W error`` turns on
                result = detect_silhouette(gray, [_face(*bbox)])
            assert result in (0, 1)

    @pytest.mark.parametrize('bbox', [
        [float('nan'), 80, 120, 120], [80, 80, float('inf'), 120], [float('-inf'), 80, 120, 120],
    ])
    def test_a_non_finite_box_is_zero_without_warnings(self, bbox):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert detect_silhouette(_scene(core=0), [_face(*bbox)]) == 0

    def test_a_dark_face_cut_by_the_bottom_right_corner_still_flags(self):
        gray = _frame()
        gray[100:, 100:] = 13                                      # the visible part of the face
        assert detect_silhouette(gray, [_face(100, 100, 260, 260)]) == 1


# ---------------------------------------------------------------------------
# Scan sites
# ---------------------------------------------------------------------------

SITE_SIZE = 64
SITE_FACE = [20, 20, 44, 44]


def _site_image(dark_face):
    """64x64 mid-bright frame, optionally with a near-black face square at SITE_FACE."""
    arr = np.full((SITE_SIZE, SITE_SIZE, 3), 230, dtype=np.uint8)
    if dark_face:
        x1, y1, x2, y2 = SITE_FACE
        arr[y1:y2, x1:x2] = 10
    return arr


def _save_png(tmp_path, name, arr):
    path = tmp_path / name
    Image.fromarray(arr).save(path, 'PNG')
    return str(path)


class _OneFaceAnalyzer:
    """A face analyzer reporting one face at SITE_FACE, whatever the pixels are."""

    def analyze_faces(self, img_cv):
        x1, y1, x2, y2 = SITE_FACE
        return {'face_count': 1, 'face_quality': 5.0, 'eye_sharpness': 5.0, 'face_sharpness': 5.0,
                'face_area': (x2 - x1) * (y2 - y1), 'bbox': np.array(SITE_FACE),
                'raw_eye_sharpness': 1.0, 'is_group_portrait': 0, 'max_face_confidence': 0.9,
                'face_details': [{'bbox': list(SITE_FACE)}]}


@pytest.fixture()
def _scan_stack():
    pytest.importorskip('torch')
    pytest.importorskip('cv2')


@pytest.mark.usefixtures('_scan_stack')
class TestBatchProcessorSite:
    @pytest.mark.timeout(30)
    def test_the_batch_path_flags_a_dark_face_and_only_that(self, tmp_path):
        from tests.test_batch_processor_e2e import FakeScorer, _make_processor
        dark = _save_png(tmp_path, 'dark.png', _site_image(True))
        light = _save_png(tmp_path, 'light.png', _site_image(False))
        scorer = FakeScorer(tmp_path)
        scorer.face_analyzer = _OneFaceAnalyzer()
        _make_processor(scorer).process_files([dark, light], show_metrics=False)
        flags = {r['filename']: r['is_silhouette'] for r in scorer.saved}
        assert flags == {'dark.png': 1, 'light.png': 0}

    @pytest.mark.timeout(30)
    def test_a_photo_without_faces_is_never_a_silhouette_whatever_its_histogram(self, tmp_path):
        from tests.test_batch_processor_e2e import FakeScorer, _make_processor
        half = np.zeros((SITE_SIZE, SITE_SIZE, 3), np.uint8)
        half[:, SITE_SIZE // 2:] = 255              # the old histogram rule's silhouette
        path = _save_png(tmp_path, 'half.png', half)
        scorer = FakeScorer(tmp_path)               # its face analyzer reports no face
        _make_processor(scorer).process_files([path], show_metrics=False)
        assert [r['is_silhouette'] for r in scorer.saved] == [0]


@pytest.mark.usefixtures('_scan_stack')
class TestSingleImageSite:
    def _scorer(self, tmp_path):
        from analyzers.technical import TechnicalAnalyzer
        from processing.scorer import Facet, _load_image_modules
        from tests.test_batch_processor_e2e import FakeScorer

        _load_image_modules()

        class Scorer(FakeScorer):
            score_photo_from_pil = Facet.score_photo_from_pil

            def get_aesthetic_and_quality(self, pil):
                return 5.0, b'emb', 5.0, 'topiq'

        scorer = Scorer(tmp_path)
        scorer.tech_analyzer = TechnicalAnalyzer()
        scorer.face_analyzer = _OneFaceAnalyzer()
        scorer.config.version_hash = 'v'
        return scorer

    def _score(self, tmp_path, dark_face):
        from utils import load_image_from_path
        scorer = self._scorer(tmp_path)
        path = _save_png(tmp_path, 'single.png', _site_image(dark_face))
        pil, cv = load_image_from_path(path)
        return scorer.score_photo_from_pil(pil, cv, path)

    def test_the_single_image_path_flags_a_dark_face(self, tmp_path):
        assert self._score(tmp_path, dark_face=True)['is_silhouette'] == 1

    def test_the_single_image_path_leaves_a_bright_face_alone(self, tmp_path):
        assert self._score(tmp_path, dark_face=False)['is_silhouette'] == 0


class _SpyScorer:
    def __init__(self, db_path, config):
        self.db_path = db_path
        self.config = config
        self.calls = []

    def calculate_aggregate_logic(self, metrics):
        self.calls.append(metrics)
        return 7.5, 'spy'

    def get_exif_data(self, path):
        return {}


@pytest.mark.usefixtures('_scan_stack')
class TestMultiPassSite:
    @pytest.fixture()
    def processor(self, tmp_path):
        from config.scoring_config import ScoringConfig
        from config_resolve import defaults_path
        from db.schema import init_database
        from processing.multi_pass import ChunkedMultiPassProcessor

        db_path = str(tmp_path / 'chunk.db')
        init_database(db_path)
        scorer = _SpyScorer(db_path, ScoringConfig(defaults_path(), validate=False))

        class _ModelManager:
            def detect_vram(self):
                return 8.0

        with mock.patch('processing.multi_pass._ensure_imports'):
            proc = ChunkedMultiPassProcessor(scorer=scorer, model_manager=_ModelManager(),
                                             config={'processing': {'load_workers': 1}})
        proc.restricted = False
        return proc

    @staticmethod
    def _images(path, dark_face):
        pil = Image.fromarray(_site_image(dark_face))
        return {path: {'height': SITE_SIZE, 'width': SITE_SIZE, 'sharpness': {}, 'color': {},
                       'histogram': {}, 'mono': {}, 'noise': {}, 'contrast': {}, 'form': {}, 'exif': {},
                       'pil': pil, 'composition': {'score': 5.0, 'power_point_score': 5.0,
                                                   'leading_lines_score': 0}}}

    def test_a_failed_face_pass_leaves_face_details_none_and_the_flag_zero(self, processor, tmp_path):
        path = str((tmp_path / 'a.jpg').resolve())
        results = {path: {'aesthetic': 5.0, 'face_count': None, 'face_details': None}}
        processor._compute_aggregates(results, self._images(path, dark_face=True))
        assert results[path]['is_silhouette'] == 0
        assert processor.scorer.calls[0]['is_silhouette'] == 0

    def test_no_face_pass_at_all_is_a_zero_too(self, processor, tmp_path):
        path = str((tmp_path / 'a.jpg').resolve())
        results = {path: {'aesthetic': 5.0}}
        processor._compute_aggregates(results, self._images(path, dark_face=True))
        assert results[path]['is_silhouette'] == 0

    def test_a_dark_face_is_flagged_and_reaches_the_aggregate(self, processor, tmp_path):
        path = str((tmp_path / 'a.jpg').resolve())
        results = {path: {'aesthetic': 5.0, 'face_count': 1,
                          'face_details': [{'bbox': list(SITE_FACE)}]}}
        processor._compute_aggregates(results, self._images(path, dark_face=True))
        assert results[path]['is_silhouette'] == 1
        assert processor.scorer.calls[0]['is_silhouette'] == 1

    def test_a_bright_face_is_not(self, processor, tmp_path):
        path = str((tmp_path / 'a.jpg').resolve())
        results = {path: {'aesthetic': 5.0, 'face_count': 1,
                          'face_details': [{'bbox': list(SITE_FACE)}]}}
        processor._compute_aggregates(results, self._images(path, dark_face=False))
        assert results[path]['is_silhouette'] == 0

    def test_multi_pass_ignores_the_clip_silhouette_tag(self, processor, tmp_path):
        path = str((tmp_path / 'a.jpg').resolve())
        results = {path: {'aesthetic': 5.0, 'face_count': 1, 'tags': 'silhouette, portrait',
                          'face_details': [{'bbox': list(SITE_FACE)}]}}
        processor._compute_aggregates(results, self._images(path, dark_face=False))
        assert results[path]['is_silhouette'] == 0
