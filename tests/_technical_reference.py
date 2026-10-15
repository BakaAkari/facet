"""Verbatim copies of the technical metrics as they were before the exact replacements.

``tests/test_technical_exact.py`` and ``scripts/perf/verify_technical_exact.py`` compare the
shipped code against these. They are frozen on purpose: do not "tidy" or speed them up, and
do not import anything from ``analyzers`` here, or the reference would follow the code under
test instead of checking it.
"""

import cv2
import numpy as np


class OldImageCache:
    """``analyzers.image_cache.ImageCache`` before the CV_16S Laplacian variance."""

    __slots__ = ['gray', 'hsv', 'laplacian_variance', 'height', 'width']

    def __init__(self, img_cv):
        self.height, self.width = img_cv.shape[:2]
        self.gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        self.laplacian_variance = cv2.Laplacian(self.gray, cv2.CV_64F).var()


def old_sharpness_score(image_cv):
    if image_cv is None:
        return 0
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(10.0, variance / 50.0)


def old_sharpness_data(image_cv, cache=None):
    if image_cv is None:
        return {'raw_variance': 0, 'normalized': 0}

    if cache is not None:
        variance = cache.laplacian_variance
    else:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        'raw_variance': variance,
        'normalized': float(min(10.0, variance / 50.0))
    }


def old_dynamic_range(image_cv, cache=None):
    if image_cv is None:
        return {'dynamic_range_stops': 0}

    if cache is not None:
        gray = cache.gray
    else:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    p2 = np.percentile(gray, 2)
    p98 = np.percentile(gray, 98)

    if p2 < 1:
        p2 = 1

    dynamic_range = np.log2(max(p98, 1) / p2)

    return {'dynamic_range_stops': round(dynamic_range, 2)}


def old_noise_estimate(image_cv, cache=None):
    if image_cv is None:
        return {'noise_sigma': 0}

    if cache is not None:
        gray = cache.gray.astype(np.float64)
    else:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY).astype(np.float64)

    h, w = gray.shape

    M = np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]])

    sigma = np.sum(np.abs(cv2.filter2D(gray, -1, M)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))

    return {'noise_sigma': round(sigma, 2)}


def old_contrast_score(image_cv, cache=None):
    if image_cv is None:
        return {'contrast_score': 0, 'percentile_contrast': 0, 'rms_contrast': 0}

    if cache is not None:
        gray = cache.gray.astype(np.float64)
    else:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY).astype(np.float64)

    p5, p95 = np.percentile(gray, [5, 95])
    percentile_contrast = (p95 - p5) / 255.0

    rms = np.std(gray) / 255.0

    contrast_score = min(10.0, (percentile_contrast * 5.0) + (rms * 20.0))

    return {
        'contrast_score': round(contrast_score, 2),
        'percentile_contrast': round(percentile_contrast, 4),
        'rms_contrast': round(rms, 4)
    }
