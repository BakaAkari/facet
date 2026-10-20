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
