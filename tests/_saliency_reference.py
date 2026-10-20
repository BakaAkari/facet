"""Verbatim copy of the pre-S2 saliency post-processing, kept as the exactness oracle.

``ReferenceSaliency`` holds the bodies of ``SaliencyScorer._score_from_mask``,
``SaliencyScorer._compute_placement_score`` and ``models.saliency_scorer.bbox_from_mask`` as they were
at 07280e7a (float64 numpy formulation). Nothing here may be "improved": it exists so a faster rewrite
is compared against what shipped, not against itself. The one deliberate deviation is the no-subject
branch of ``_score_from_mask``: it stores None (not a made-up 5.0) for the three metrics that need a
subject, because the production method now does; the other is the ``int()`` around
``np.count_nonzero`` (see there). ``tests/test_saliency_exact.py`` and
``scripts/perf/verify_saliency_exact.py`` both import it.
"""
import cv2
import numpy as np


def reference_bbox_from_mask(mask, min_subject_pixels: int = 50):
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return None
    ys, xs = np.nonzero(mask > 128)
    if xs.size < min_subject_pixels:
        return None
    x0 = float(xs.min()) / w
    y0 = float(ys.min()) / h
    x1 = float(xs.max() + 1) / w
    y1 = float(ys.max() + 1) / h
    return [round(x0, 4), round(y0, 4), round(min(x1, 1.0), 4), round(min(y1, 1.0), 4)]


class ReferenceSaliency:
    """The old scoring methods, bound to a ``min_subject_pixels`` setting."""

    def __init__(self, min_subject_pixels: int = 50):
        self.min_subject_pixels = min_subject_pixels

    def _score_from_mask(self, mask, img_cv):
        h, w = mask.shape[:2]
        total_pixels = h * w

        # Subject area ratio. int() is the one departure from the old code: it ran
        # under NumPy < 2.3, where np.count_nonzero returned a Python int; from 2.3
        # it returns np.int64, which would make this reference's prominence an
        # np.float64 the old code never produced.
        subject_pixels = int(np.count_nonzero(mask))
        subject_prominence = subject_pixels / total_pixels if total_pixels > 0 else 0

        # If no subject detected, return defaults (None, not 5.0: see the module docstring)
        if subject_pixels < self.min_subject_pixels:  # Minimum subject size
            return {
                'subject_sharpness': None,
                'subject_prominence': 0.0,
                'subject_placement': None,
                'bg_separation': None,
                'subject_bbox': None,
            }

        # Convert to grayscale for Laplacian
        if img_cv.ndim == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv

        # Resize gray to mask dimensions if needed
        if gray.shape[:2] != mask.shape[:2]:
            gray = cv2.resize(gray, (w, h))

        # Compute Laplacian (edge/sharpness detector)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Subject sharpness: Laplacian variance on subject region
        subject_mask_bool = mask > 128
        bg_mask_bool = ~subject_mask_bool

        subject_laplacian = laplacian[subject_mask_bool]
        subject_variance = float(np.var(subject_laplacian)) if len(subject_laplacian) > 0 else 0

        # Background sharpness for separation metric
        bg_laplacian = laplacian[bg_mask_bool]
        bg_variance = float(np.var(bg_laplacian)) if len(bg_laplacian) > 0 else 0

        # Normalize subject sharpness to 0-10 (typical range 0-5000)
        subject_sharpness = min(10.0, (subject_variance ** 0.5) / 7.0)

        # Background separation: ratio of subject to background sharpness
        if bg_variance > 0:
            separation_ratio = subject_variance / (bg_variance + 1e-6)
            bg_separation = min(10.0, separation_ratio * 2.0)
        else:
            bg_separation = 10.0  # Perfect separation (no background detail)

        # Subject placement: rule-of-thirds scoring for subject centroid
        subject_placement = self._compute_placement_score(mask, h, w)

        # Normalize prominence to 0-10 scale
        prominence_score = min(10.0, subject_prominence * 20.0)  # 50% coverage = 10.0

        return {
            'subject_sharpness': round(subject_sharpness, 2),
            'subject_prominence': round(prominence_score, 2),
            'subject_placement': round(subject_placement, 2),
            'bg_separation': round(bg_separation, 2),
            'subject_bbox': reference_bbox_from_mask(mask, self.min_subject_pixels),
        }

    def _compute_placement_score(self, mask, h, w):
        # Find subject centroid
        ys, xs = np.nonzero(mask > 128)
        if len(xs) == 0:
            return 5.0

        cx = float(np.mean(xs)) / w
        cy = float(np.mean(ys)) / h

        # Rule-of-thirds power points
        thirds_x = [1/3, 2/3]
        thirds_y = [1/3, 2/3]

        # Find minimum distance to any power point
        min_dist = float('inf')
        for tx in thirds_x:
            for ty in thirds_y:
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                min_dist = min(min_dist, dist)

        # Max possible distance from a power point is ~0.47 (corner to center third)
        max_dist = 0.47
        score = max(0.0, 10.0 * (1.0 - min_dist / max_dist))

        return score
