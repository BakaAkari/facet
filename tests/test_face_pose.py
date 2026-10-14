"""Head-pose derivation from MediaPipe's facial transformation matrix (issue #185 item 6)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from analyzers import face_blendshapes as fb


def _rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _matrix(rot, scale=1.0, translate=(0.0, 0.0, -40.0)):
    m = np.eye(4)
    m[:3, :3] = rot * scale
    m[:3, 3] = translate
    return m


def test_frontal_face_is_zero():
    yaw, pitch, roll = fb.pose_from_matrix(_matrix(np.eye(3)))
    assert (abs(yaw), abs(pitch), abs(roll)) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("angle", [10, 35, 60, 80])
def test_yaw_only_lands_on_yaw_axis(angle):
    # rotation about the vertical axis (a head turn): yaw == angle, pitch and roll untouched
    for sign in (1, -1):
        yaw, pitch, roll = fb.pose_from_matrix(_matrix(_rot_y(sign * angle), scale=1.7))
        assert yaw == pytest.approx(sign * angle, abs=1e-6)
        assert abs(pitch) < 1e-6 and abs(roll) < 1e-6


@pytest.mark.parametrize("angle", [10, 35, 60])
def test_pitch_only_lands_on_pitch_axis(angle):
    # rotation about the horizontal axis (a nod): |pitch| == angle, yaw and roll untouched
    for sign in (1, -1):
        yaw, pitch, roll = fb.pose_from_matrix(_matrix(_rot_x(sign * angle)))
        assert abs(pitch) == pytest.approx(angle, abs=1e-6)
        assert abs(yaw) < 1e-6 and abs(roll) < 1e-6


def test_roll_is_not_confused_with_yaw_or_pitch():
    yaw, pitch, roll = fb.pose_from_matrix(_matrix(_rot_z(25)))
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6 and abs(roll) == pytest.approx(25, abs=1e-6)


def test_axes_are_not_swapped():
    # the property that fails if yaw and pitch are assigned to each other's axis
    yaw_m = fb.pose_from_matrix(_matrix(_rot_y(50)))
    pitch_m = fb.pose_from_matrix(_matrix(_rot_x(50)))
    assert abs(yaw_m[0]) > 49 and abs(yaw_m[1]) < 1
    assert abs(pitch_m[1]) > 49 and abs(pitch_m[0]) < 1


def test_combined_rotation_recovers_both_angles():
    yaw, pitch, _ = fb.pose_from_matrix(_matrix(_rot_y(30) @ _rot_x(20)))
    # Ry applied last: forward = (sin30 cos20, -sin20, cos30 cos20) -> yaw exactly 30, |pitch| exactly 20
    assert yaw == pytest.approx(30, abs=1e-6)
    assert abs(pitch) == pytest.approx(20, abs=1e-6)


@pytest.mark.parametrize("bad", [None, np.zeros((4, 4)), [[1, 2], [3, 4]], "x"])
def test_degenerate_matrix_is_none(bad):
    assert fb.pose_from_matrix(bad) is None


class _FakeLandmarker:
    def __init__(self, matrixes):
        self.matrixes = matrixes

    def detect(self, image):
        cats = [SimpleNamespace(category_name=n, score=0.1) for n in (
            fb.EYE_BLINK_LEFT, fb.EYE_BLINK_RIGHT, fb.MOUTH_SMILE_LEFT, fb.MOUTH_SMILE_RIGHT,
            fb.MOUTH_FROWN_LEFT, fb.MOUTH_FROWN_RIGHT)]
        return SimpleNamespace(
            face_blendshapes=[cats], face_landmarks=[[SimpleNamespace(x=0.5, y=0.5)] * 478],
            facial_transformation_matrixes=self.matrixes)


def _scorer(matrixes):
    s = fb.BlendshapeScorer(model_path='x', min_crop_size=8)
    s._landmarker = _FakeLandmarker(matrixes)
    s._mp = SimpleNamespace(Image=lambda **k: None, ImageFormat=SimpleNamespace(SRGB=0))
    return s


def test_scorer_keeps_four_tuple_and_adds_pose_variant():
    crop = np.zeros((16, 16, 3), np.uint8)
    s = _scorer([_matrix(_rot_y(40))])
    assert len(s.analyze_face_crop(crop)) == 4
    full = s.analyze_face_crop_pose(crop)
    assert len(full) == 5 and full[4][0] == pytest.approx(40, abs=1e-6)
    assert s.analyze_face_crop(crop)[:3] == full[:3]


def test_scorer_without_matrix_gives_none_pose():
    crop = np.zeros((16, 16, 3), np.uint8)
    assert _scorer(None).analyze_face_crop_pose(crop)[4] is None
    assert _scorer([]).analyze_face_crop_pose(crop)[4] is None
