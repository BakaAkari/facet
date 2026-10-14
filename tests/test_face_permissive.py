"""Tests for the permissive (YuNet + MediaPipe + SFace) face analyzer, issue #185 W5."""

import glob
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from analyzers import face_permissive
from analyzers.face import FaceAnalyzer
from analyzers.face_permissive import PermissiveFaceAnalyzer

NAMES_OPEN = {'eyeBlinkLeft': 0.1, 'eyeBlinkRight': 0.1}
NAMES_BLINK = {'eyeBlinkLeft': 0.9, 'eyeBlinkRight': 0.2}


def _row(x, y, w, h, score):
    """YuNet row: box, 5 keypoints (eyes at 30%/70% of width), score."""
    kps = [x + 0.3 * w, y + 0.35 * h, x + 0.7 * w, y + 0.35 * h,
           x + 0.5 * w, y + 0.55 * h, x + 0.35 * w, y + 0.8 * h, x + 0.65 * w, y + 0.8 * h]
    return [x, y, w, h, *kps, score]


class _StubDetector:
    def __init__(self, rows_fullres, scale_ref):
        self.rows_fullres = rows_fullres
        self.scale_ref = scale_ref
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, img):
        # Report coordinates in the (possibly downscaled) space actually given.
        scale = img.shape[1] / self.scale_ref
        if not self.rows_fullres:
            return 0, None
        rows = np.array(self.rows_fullres, dtype=np.float32)
        rows[:, :14] *= scale
        return len(rows), rows


class _StubRecognizer:
    def __init__(self):
        self.aligned_rows = []

    def alignCrop(self, img, row):
        self.aligned_rows.append(np.asarray(row).copy())
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def feature(self, aligned):
        return np.full((1, 128), 3.0, dtype=np.float32)  # un-normalised on purpose


class _StubBlendshapes:
    def __init__(self, names_per_call):
        self.names = list(names_per_call)
        self.crops = []

    def analyze_face_crop(self, crop):
        self.crops.append(crop.shape)
        names = self.names.pop(0)
        return 8.0, 6.0, dict(names), None


def _analyzer(monkeypatch, rows, img_width, blendshapes=None, **kwargs):
    detector = _StubDetector(rows, img_width)
    recognizer = _StubRecognizer()
    monkeypatch.setattr(face_permissive, '_ensure_weights', lambda: ('y.onnx', 's.onnx'))
    monkeypatch.setattr(face_permissive, '_create_detector', lambda *a, **k: detector)
    monkeypatch.setattr(face_permissive, '_create_recognizer', lambda *a, **k: recognizer)
    analyzer = PermissiveFaceAnalyzer(**kwargs)
    analyzer._blendshape_scorer = blendshapes
    analyzer.enable_blendshapes = blendshapes is not None
    return analyzer, detector, recognizer


def _textured(w=640, h=480):
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _insightface_contract():
    """Result of a FaceAnalyzer run against a stubbed InsightFace app."""
    rng = np.random.default_rng(0)
    face = SimpleNamespace(
        det_score=0.99, bbox=np.array([100., 100., 220., 240.]),
        embedding=rng.random(512).astype(np.float32),
        landmark_2d_106=(rng.random((106, 2)) * 100 + 100).astype(np.float32))
    analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
    analyzer.available = True
    analyzer.min_confidence = 0.7
    analyzer.min_face_size = 30
    analyzer.thumbnail_size = 64
    analyzer.thumbnail_quality = 80
    analyzer.blink_ear_threshold = 0.21
    analyzer.min_faces_for_group = 4
    analyzer.enable_blendshapes = False
    analyzer.enable_3d_landmarks = False
    analyzer.face_app = SimpleNamespace(get=lambda img: [face])
    return analyzer.analyze_faces(_textured())


class TestContract:
    def test_key_sets_match_face_analyzer(self, monkeypatch):
        reference = _insightface_contract()
        analyzer, *_ = _analyzer(monkeypatch, [_row(100, 100, 120, 140, 0.95)], 640)
        result = analyzer.analyze_faces(_textured())
        # face_quality_iqa (OFIQ) is the one result key only this analyzer reports
        assert set(result) == set(reference) | {'face_quality_iqa'}
        assert set(result['face_details'][0]) == (
            set(reference['face_details'][0]) | {'embedding_model'})

    def test_zero_contract_matches_face_analyzer(self, monkeypatch):
        reference = FaceAnalyzer.__new__(FaceAnalyzer)
        reference.available = False
        expected = reference.analyze_faces(_textured())
        analyzer, *_ = _analyzer(monkeypatch, [], 640)
        assert analyzer.analyze_faces(_textured()) == {**expected, 'face_quality_iqa': None}

    def test_class_attributes(self):
        assert PermissiveFaceAnalyzer.embedding_model == 'sface'
        assert PermissiveFaceAnalyzer.embedding_dim == 128


class TestAnalyze:
    def test_none_image_returns_zero_contract(self, monkeypatch):
        analyzer, *_ = _analyzer(monkeypatch, [], 640)
        assert analyzer.analyze_faces(None)['face_count'] == 0

    def test_low_confidence_and_tiny_faces_filtered(self, monkeypatch):
        rows = [_row(50, 50, 100, 100, 0.5), _row(300, 300, 20, 20, 0.99)]
        analyzer, *_ = _analyzer(monkeypatch, rows, 640)
        result = analyzer.analyze_faces(_textured())
        assert result['face_count'] == 0
        assert result['max_face_confidence'] == pytest.approx(0.99)

    def test_two_faces_union_area_and_embeddings(self, monkeypatch):
        rows = [_row(100, 100, 100, 120, 0.9), _row(300, 150, 80, 100, 0.8)]
        analyzer, _, recognizer = _analyzer(monkeypatch, rows, 640)
        result = analyzer.analyze_faces(_textured())
        assert result['face_count'] == 2
        assert result['bbox'].tolist() == [100, 100, 380, 250]
        assert result['face_area'] == 100 * 120 + 80 * 100
        assert result['is_group_portrait'] == 0
        assert result['face_quality'] == pytest.approx(round(0.7 * 8.0 + 0.3 * 8.5, 2), abs=0.02)
        # no mesh -> blink was not measured: unknown (None), never "eyes open" (0)
        assert result['is_blink'] is None
        for detail in result['face_details']:
            emb = np.frombuffer(detail['embedding'], dtype=np.float32)
            assert emb.shape == (128,)
            assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
            assert detail['landmark_2d_106'] is None
            assert detail['embedding_model'] == 'sface'
            assert isinstance(detail['bbox'], list)
            assert detail['thumbnail'][:2] == b'\xff\xd8'
        assert len(recognizer.aligned_rows) == 2

    def test_group_portrait_threshold(self, monkeypatch):
        rows = [_row(20 + 150 * i, 50, 100, 100, 0.9) for i in range(4)]
        analyzer, *_ = _analyzer(monkeypatch, rows, 640, min_faces_for_group=4)
        assert analyzer.analyze_faces(_textured())['is_group_portrait'] == 1
        analyzer.min_faces_for_group = 5
        assert analyzer.analyze_faces(_textured())['is_group_portrait'] == 0

    def test_downscaled_detection_is_rescaled_to_fullres(self, monkeypatch):
        big = _textured(2560, 1920)
        rows = [_row(1000, 800, 400, 500, 0.95)]
        analyzer, detector, recognizer = _analyzer(monkeypatch, rows, 2560)
        result = analyzer.analyze_faces(big)
        assert detector.input_size == (1280, 960)  # detection ran at half size
        assert result['face_details'][0]['bbox'] == [1000, 800, 1400, 1300]
        assert recognizer.aligned_rows[0][:4] == pytest.approx([1000, 800, 400, 500], abs=0.5)

    def test_blink_threshold_behaviour(self, monkeypatch):
        rows = [_row(100, 100, 120, 140, 0.9)]
        scorer = _StubBlendshapes([NAMES_OPEN, NAMES_BLINK, NAMES_BLINK])
        analyzer, *_ = _analyzer(monkeypatch, rows, 640, blendshapes=scorer)
        img = _textured()
        assert analyzer.analyze_faces(img)['is_blink'] == 0
        assert analyzer.analyze_faces(img)['is_blink'] == 1
        analyzer.blink_threshold = 0.95  # 0.9 activation no longer counts
        assert analyzer.analyze_faces(img)['is_blink'] == 0

    def test_blendshape_scores_and_upscaled_crop(self, monkeypatch):
        rows = [_row(100, 100, 60, 60, 0.9)]
        scorer = _StubBlendshapes([NAMES_OPEN])
        analyzer, *_ = _analyzer(monkeypatch, rows, 640, blendshapes=scorer)
        detail = analyzer.analyze_faces(_textured())['face_details'][0]
        assert (detail['eyes_open_score'], detail['smile_score']) == (8.0, 6.0)
        assert min(scorer.crops[0][:2]) >= 192

    def test_no_blendshapes_leaves_scores_none(self, monkeypatch):
        rows = [_row(100, 100, 120, 140, 0.9)]
        analyzer, *_ = _analyzer(monkeypatch, rows, 640)
        result = analyzer.analyze_faces(_textured())
        detail = result['face_details'][0]
        assert detail['eyes_open_score'] is None and detail['smile_score'] is None
        assert result['is_blink'] is None
        assert result['raw_eye_sharpness'] > 0  # keypoint fallback patches


class TestAvailability:
    def test_missing_cv2_api_is_unavailable_and_logged(self, monkeypatch, caplog):
        monkeypatch.delattr(cv2, 'FaceDetectorYN')
        with caplog.at_level('WARNING', logger='facet.face_permissive'):
            analyzer = PermissiveFaceAnalyzer()
        assert analyzer.available is False
        assert 'FaceDetectorYN' in caplog.text
        assert analyzer.analyze_faces(_textured())['face_count'] == 0

    def test_weight_load_failure_marks_unavailable(self, monkeypatch, caplog):
        def boom():
            raise ConnectionError("offline")
        monkeypatch.setattr(face_permissive, '_ensure_weights', boom)
        analyzer = PermissiveFaceAnalyzer()
        with caplog.at_level('WARNING', logger='facet.face_permissive'):
            assert analyzer.analyze_faces(_textured())['face_count'] == 0
        assert analyzer.available is False
        assert 'offline' in caplog.text


@pytest.mark.skipif(os.environ.get('FACET_FACE_PERMISSIVE_REAL') != '1',
                    reason='set FACET_FACE_PERMISSIVE_REAL=1 to run with real models')
def test_real_models_smoke():
    photos = sorted(glob.glob(r'D:\photo-llm\photos\*.jpg'))[:3]
    if not photos:
        pytest.skip('no sample photos')
    analyzer = PermissiveFaceAnalyzer()
    found = 0
    for path in photos:
        img = cv2.imread(path)  # read-only
        result = analyzer.analyze_faces(img)
        print(os.path.basename(path), result['face_count'], result['max_face_confidence'],
              [(d['eyes_open_score'], d['smile_score']) for d in result['face_details']])
        for detail in result['face_details']:
            emb = np.frombuffer(detail['embedding'], dtype=np.float32)
            assert emb.shape == (128,)
            assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-4)
        found += result['face_count']
    assert found >= 1


def test_default_blink_threshold_is_the_w8_measured_optimum():
    from analyzers.face_permissive import DEFAULT_BLINK_THRESHOLD
    # D33: base-rate-weighted choice (was 0.35, the unweighted max-F1 point)
    assert DEFAULT_BLINK_THRESHOLD == 0.55
