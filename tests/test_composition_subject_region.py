"""detect_subject_region is Canny-only: a box when a contour qualifies, else None."""

import numpy as np

from analyzers.composition import CompositionAnalyzer


def test_canny_contour_returns_bounding_box():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[15:45, 25:55] = 255
    assert CompositionAnalyzer.detect_subject_region(image) == [24, 14, 55, 45]


def test_uniform_image_returns_none():
    assert CompositionAnalyzer.detect_subject_region(np.zeros((64, 96, 3), dtype=np.uint8)) is None


def test_none_image_returns_none():
    assert CompositionAnalyzer.detect_subject_region(None) is None


def test_larger_contour_near_thirds_wins():
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    image[10:30, 10:30] = 255
    image[40:80, 50:110] = 255
    assert CompositionAnalyzer.detect_subject_region(image) == [49, 39, 110, 80]


def test_contour_below_min_area_is_ignored():
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image[500:505, 500:505] = 255
    assert CompositionAnalyzer.detect_subject_region(image) is None
