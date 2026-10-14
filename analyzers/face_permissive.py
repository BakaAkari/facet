"""Permissively-licensed face pipeline for commercial mode (issue #185, W5).

Drop-in for :class:`analyzers.face.FaceAnalyzer` without InsightFace (its
buffalo_l weights are non-commercial):

- detection: YuNet (MIT) via ``cv2.FaceDetectorYN``, run on a downscaled copy;
- per-face mesh + blendshapes: MediaPipe Face Landmarker (Apache-2.0), giving
  eyes-open / smile / blink and the eye regions used for eye sharpness;
- recognition: SFace (Apache-2.0) via ``cv2.FaceRecognizerSF``, 128-d embeddings.

Embeddings are 128-d and live in a different space from the 512-d ArcFace ones,
so every face dict carries ``embedding_model='sface'``. ``landmark_2d_106`` is
always ``None`` (no 106-point model here).
"""

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from analyzers.face_blendshapes import CROP_PADDING, crop_face_region, get_blendshape_scorer
from models.weights import (
    SFACE_WEIGHTS_FILENAME,
    SFACE_WEIGHTS_SHA256,
    SFACE_WEIGHTS_URL,
    YUNET_WEIGHTS_FILENAME,
    YUNET_WEIGHTS_SHA256,
    YUNET_WEIGHTS_URL,
    download_weights,
    pretrained_model_path,
)
from utils.image_transforms import crop_face_with_padding, padded_face_bbox

logger = logging.getLogger("facet.face_permissive")

EMBEDDING_MODEL = 'sface'
EMBEDDING_DIM = 128

# Detection runs on a copy whose long side is at most this; boxes and keypoints
# are scaled back to full resolution. YuNet cost is linear in pixels, and faces
# large enough to matter survive the downscale.
DETECT_MAX_SIDE = 1280
# The detector's own floor is below any sensible min_confidence so
# max_face_confidence still reflects rejected faces, as FaceAnalyzer's does.
DETECTOR_SCORE_FLOOR = 0.3
DETECTOR_NMS = 0.3
DETECTOR_TOP_K = 5000

# Why 0.5: eyeBlink* is a 0-1 activation; 0.5 is the midpoint where the lid
# covers about half the eye. Provisional default, W8 tunes it against data.
DEFAULT_BLINK_THRESHOLD = 0.5

# MediaPipe canonical 478-point mesh eye contours (subject's right / left eye).
MESH_RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
MESH_LEFT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
EYE_PATCH_PADDING = 0.25
MESH_MIN_POINTS = 400

# YuNet row layout: x, y, w, h, then (rx, ry), (lx, ly), nose, mouth-r, mouth-l, score.
_KPS_RIGHT_EYE = slice(4, 6)
_KPS_LEFT_EYE = slice(6, 8)
_SCORE_COL = 14


def _zero_result(max_confidence=0):
    return {
        'face_count': 0, 'face_quality': 0, 'eye_sharpness': 0,
        'is_blink': 0, 'face_area': 0, 'bbox': None,
        'face_sharpness': 0, 'raw_eye_sharpness': 0,
        'is_group_portrait': 0, 'max_face_confidence': max_confidence,
        'face_details': [],
    }


def _create_detector(model_path, input_size, score_threshold, nms, top_k):
    return cv2.FaceDetectorYN.create(model_path, "", input_size, score_threshold, nms, top_k)


def _create_recognizer(model_path):
    return cv2.FaceRecognizerSF.create(model_path, "")


def _ensure_weights():
    """Download (once, SHA-checked) and return ``(yunet_path, sface_path)``."""
    paths = []
    for filename, url, sha in (
        (YUNET_WEIGHTS_FILENAME, YUNET_WEIGHTS_URL, YUNET_WEIGHTS_SHA256),
        (SFACE_WEIGHTS_FILENAME, SFACE_WEIGHTS_URL, SFACE_WEIGHTS_SHA256),
    ):
        path = pretrained_model_path(filename)
        if not path.exists():
            download_weights(url, Path(path), sha)
        paths.append(str(path))
    return tuple(paths)


def _laplacian_ratio(img, x1, y1, x2, y2):
    """Laplacian variance / (mean + 1) of a patch, or None if it is empty."""
    h, w = img.shape[:2]
    roi = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() / (np.mean(gray) + 1)


def _crop_sharpness(img, bbox):
    """Laplacian variance of the face box (same measure as FaceAnalyzer)."""
    h, w = img.shape[:2]
    y1, y2, x1, x2 = max(0, bbox[1]), min(h, bbox[3]), max(0, bbox[0]), min(w, bbox[2])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    return cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


class PermissiveFaceAnalyzer:
    """YuNet + MediaPipe + SFace analyzer with the ``FaceAnalyzer`` contract."""

    embedding_model = EMBEDDING_MODEL
    embedding_dim = EMBEDDING_DIM

    def __init__(self, device='cpu', min_confidence=0.7, min_face_size=30,
                 thumbnail_size=128, thumbnail_quality=85,
                 blink_threshold=DEFAULT_BLINK_THRESHOLD, min_faces_for_group=4,
                 enable_blendshapes=True, blendshape_min_crop=192):
        # `device` is accepted for constructor parity; YuNet/SFace run on CPU via cv2.
        self.available = False
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.thumbnail_size = thumbnail_size
        self.thumbnail_quality = thumbnail_quality
        self.blink_threshold = blink_threshold
        self.min_faces_for_group = min_faces_for_group
        self.enable_blendshapes = enable_blendshapes
        self.blendshape_min_crop = blendshape_min_crop
        self._blendshape_scorer = None
        self._detector = None
        self._recognizer = None
        self._lock = threading.Lock()
        if not (hasattr(cv2, 'FaceDetectorYN') and hasattr(cv2, 'FaceRecognizerSF')):
            logger.warning(
                "Permissive face pipeline unavailable: this OpenCV build (%s) lacks "
                "FaceDetectorYN/FaceRecognizerSF (needs opencv-python >= 4.5.4)",
                getattr(cv2, '__version__', '?'))
            return
        self.available = True

    def _ensure_models(self):
        if self._detector is not None:
            return True
        with self._lock:
            if self._detector is not None:
                return True
            try:
                yunet, sface = _ensure_weights()
                self._detector = _create_detector(
                    yunet, (320, 320), DETECTOR_SCORE_FLOOR, DETECTOR_NMS, DETECTOR_TOP_K)
                self._recognizer = _create_recognizer(sface)
                return True
            except Exception as e:
                self.available = False
                logger.warning("Permissive face models failed to load: %s", e)
                return False

    def _detect(self, img_cv):
        """YuNet rows rescaled to full-res coordinates (N x 15), or an empty list."""
        h, w = img_cv.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        small = img_cv
        if scale < 1.0:
            small = cv2.resize(img_cv, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        with self._lock:
            self._detector.setInputSize((sw, sh))
            _, rows = self._detector.detect(small)
        if rows is None or len(rows) == 0:
            return []
        rows = np.asarray(rows, dtype=np.float32).copy()
        rows[:, :14] /= scale
        return rows

    def _mesh_analysis(self, img_cv, bbox):
        """Landmarker on a padded crop upscaled to ``blendshape_min_crop``.

        Returns ``(eyes_open, smile, blendshapes, mesh_in_full_res)`` or None.
        """
        if not self.enable_blendshapes:
            return None
        if self._blendshape_scorer is None:
            self._blendshape_scorer = get_blendshape_scorer(min_crop_size=self.blendshape_min_crop)
        crop = crop_face_region(img_cv, bbox)
        if crop is None:
            return None
        ch, cw = crop.shape[:2]
        scale = max(1.0, self.blendshape_min_crop / min(ch, cw))
        if scale > 1.0:
            crop = cv2.resize(crop, (round(cw * scale), round(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        analysis = self._blendshape_scorer.analyze_face_crop(crop)
        if analysis is None:
            return None
        eyes_open, smile, names, mesh = analysis
        if mesh is not None:
            ox, oy, _, _ = padded_face_bbox(img_cv.shape, bbox, CROP_PADDING)
            mesh = mesh / scale + np.array([ox, oy], dtype=np.float32)
        return eyes_open, smile, names, mesh

    def _eye_sharpness(self, img_cv, row, mesh):
        """Raw eye sharpness: max Laplacian ratio over the two eye patches.

        Patches come from the mesh eye contours when available, otherwise from a
        window around the YuNet eye keypoints sized like FaceAnalyzer's.
        """
        patches = []
        if mesh is not None and len(mesh) >= MESH_MIN_POINTS:
            for idx in (MESH_RIGHT_EYE, MESH_LEFT_EYE):
                pts = mesh[list(idx)]
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                pad = max(x2 - x1, y2 - y1) * EYE_PATCH_PADDING
                patches.append((int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad)))
        else:
            r_eye, l_eye = row[_KPS_RIGHT_EYE], row[_KPS_LEFT_EYE]
            offset = int(np.linalg.norm(r_eye - l_eye) * 0.15)
            for ex, ey in (r_eye, l_eye):
                patches.append((int(ex - offset), int(ey - offset),
                                int(ex + offset), int(ey + offset)))
        vals = [v for v in (_laplacian_ratio(img_cv, *p) for p in patches) if v is not None]
        return max(vals) if vals else 0

    def _embed(self, img_cv, row):
        aligned = self._recognizer.alignCrop(img_cv, row)
        feat = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 0 else feat

    def analyze_faces(self, img_cv):
        """Same return contract as :meth:`FaceAnalyzer.analyze_faces`."""
        if not self.available or img_cv is None or not self._ensure_models():
            return _zero_result()

        h, w = img_cv.shape[:2]
        max_confidence = 0
        faces = []
        for row in self._detect(img_cv):
            confidence = float(row[_SCORE_COL])
            max_confidence = max(max_confidence, confidence)
            if confidence < self.min_confidence:
                continue
            bbox = np.array([row[0], row[1], row[0] + row[2], row[1] + row[3]]).astype(int)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h)
            if (bbox[2] - bbox[0]) < self.min_face_size or (bbox[3] - bbox[1]) < self.min_face_size:
                continue
            faces.append((row, bbox, confidence))

        if not faces:
            return _zero_result(max_confidence)

        qualities, eye_scores, raw_eyes, sharpness = [], [], [], []
        any_blink = False
        total_area = 0
        min_x, min_y, max_x, max_y = w, h, 0, 0
        face_details = []
        for idx, (row, bbox, confidence) in enumerate(faces):
            min_x, min_y = min(min_x, bbox[0]), min(min_y, bbox[1])
            max_x, max_y = max(max_x, bbox[2]), max(max_y, bbox[3])
            total_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            qualities.append(confidence * 10)

            mesh_result = self._mesh_analysis(img_cv, bbox)
            eyes_open = smile = mesh = None
            if mesh_result is not None:
                eyes_open, smile, names, mesh = mesh_result
                if max(names['eyeBlinkLeft'], names['eyeBlinkRight']) >= self.blink_threshold:
                    any_blink = True

            raw_eye = self._eye_sharpness(img_cv, row, mesh)
            raw_eyes.append(raw_eye)
            eye_scores.append(min(10.0, raw_eye / 2.0))
            sharpness.append(_crop_sharpness(img_cv, bbox))

            face_details.append({
                'index': idx,
                'bbox': bbox.tolist(),
                'confidence': confidence,
                'embedding': self._embed(img_cv, row).tobytes(),
                'landmark_2d_106': None,
                'thumbnail': crop_face_with_padding(
                    img_cv, bbox, 0.3, self.thumbnail_size, self.thumbnail_quality),
                'eyes_open_score': eyes_open,
                'smile_score': smile,
                'embedding_model': EMBEDDING_MODEL,
            })

        face_quality = round(0.7 * min(qualities) + 0.3 * (sum(qualities) / len(qualities)), 2)
        return {
            'face_obj': None,  # InsightFace object in FaceAnalyzer; no equivalent here
            'face_count': len(faces),
            'face_quality': face_quality,
            'eye_sharpness': round(sum(eye_scores) / len(eye_scores), 2),
            'raw_eye_sharpness': sum(raw_eyes) / len(raw_eyes),
            'face_sharpness': sum(sharpness) / len(sharpness),
            'is_blink': 1 if any_blink else 0,
            'face_area': total_area,
            'bbox': np.array([min_x, min_y, max_x, max_y]),
            'is_group_portrait': 1 if len(faces) >= self.min_faces_for_group else 0,
            'max_face_confidence': max_confidence,
            'face_details': face_details,
        }
