"""Distinguish unavailable saliency from a successful empty subject result."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from analyzers.composition import CompositionAnalyzer


@pytest.fixture
def image():
    # Uniform input skips the Canny branch; saliency below is synthetic.
    return np.zeros((64, 96, 3), dtype=np.uint8)


def install_saliency(monkeypatch, callback):
    detector = SimpleNamespace(computeSaliency=callback)
    monkeypatch.setattr(
        cv2, "saliency",
        SimpleNamespace(StaticSaliencySpectralResidual_create=lambda: detector),
        raising=False,
    )


def test_missing_capability_preserves_default_and_strict_reports_failure(monkeypatch, image):
    monkeypatch.delattr(cv2, "saliency", raising=False)
    assert CompositionAnalyzer.detect_subject_region(image) is None
    with pytest.raises(RuntimeError, match="capability or execution") as error:
        CompositionAnalyzer.detect_subject_region(image, strict_errors=True)
    assert isinstance(error.value.__cause__, AttributeError)


def test_failed_status_preserves_default_and_strict_reports_failure(monkeypatch, image):
    install_saliency(monkeypatch, lambda _: (False, None))
    assert CompositionAnalyzer.detect_subject_region(image) is None
    with pytest.raises(RuntimeError, match="computeSaliency failed"):
        CompositionAnalyzer.detect_subject_region(image, strict_errors=True)


def test_opencv_error_preserves_default_and_strict_preserves_cause(monkeypatch, image):
    def fail(_):
        raise cv2.error("synthetic spectral failure")

    install_saliency(monkeypatch, fail)
    assert CompositionAnalyzer.detect_subject_region(image) is None
    with pytest.raises(RuntimeError, match="capability or execution") as error:
        CompositionAnalyzer.detect_subject_region(image, strict_errors=True)
    assert isinstance(error.value.__cause__, cv2.error)


@pytest.mark.parametrize("kind", ["nan", "inf", "shape", "string", "complex", "none", "negative", "overflow"])
def test_strict_rejects_invalid_successful_maps(monkeypatch, image, kind):
    value = np.zeros(image.shape[:2], dtype=np.float32)
    if kind == "nan":
        value[0, 0] = np.nan
    elif kind == "inf":
        value[0, 0] = np.inf
    elif kind == "shape":
        value = value[:2, :2]
    elif kind == "string":
        value = value.astype(str)
    elif kind == "complex":
        value = value.astype(complex)
    elif kind == "negative":
        value[10:20, 20:40] = -1
    elif kind == "overflow":
        value[10:20, 20:40] = 256
    else:
        value = None
    install_saliency(monkeypatch, lambda _: (True, value))
    with pytest.raises(RuntimeError, match="Invalid spectral"):
        CompositionAnalyzer.detect_subject_region(image, strict_errors=True)


@pytest.mark.parametrize("has_subject", [False, True])
def test_successful_spectral_geometry_unchanged(monkeypatch, image, has_subject):
    value = np.zeros(image.shape[:2], dtype=np.float32)
    if has_subject:
        value[10:20, 20:40] = 1
    install_saliency(monkeypatch, lambda _: (True, value))
    expected = [20, 10, 40, 20] if has_subject else None
    assert CompositionAnalyzer.detect_subject_region(image) == expected
    assert CompositionAnalyzer.detect_subject_region(image, strict_errors=True) == expected


def test_canny_success_does_not_require_saliency(monkeypatch, image):
    monkeypatch.delattr(cv2, "saliency", raising=False)
    image[15:45, 25:55] = 255
    ordinary = CompositionAnalyzer.detect_subject_region(image)
    assert ordinary is not None
    assert CompositionAnalyzer.detect_subject_region(image, strict_errors=True) == ordinary
    assert CompositionAnalyzer.get_placement_data(ordinary, 96, 64) == (
        CompositionAnalyzer.get_placement_data(
            CompositionAnalyzer.detect_subject_region(image, strict_errors=True), 96, 64
        )
    )
