"""
Detection utilities for Facet.

Silhouette detection and shared constants.
"""

import math

import numpy as np

# Silhouette rule: the subject (the largest detected face) is a dark patch against a
# clearly brighter surround. Policy thresholds, not a natural gap in the data -- the
# face-core luma and ring contrast distributions are continuous (see docs/SCORING.md).
SILHOUETTE_CORE_MAX = 0.12          # mean luma (0-1) of the central 60% of the face box
SILHOUETTE_CONTRAST_MIN = 0.20      # ring mean minus core mean
SILHOUETTE_CORE_INSET = 0.2         # the core drops this fraction of w/h from each side
SILHOUETTE_RING_INNER = 0.2         # the ring starts this far outside the box (per side, x w/h)
SILHOUETTE_RING_OUTER = 0.75        # ... and ends this far outside it
SILHOUETTE_MIN_CORE_PX = 2          # the core must span at least this many px on both axes
SILHOUETTE_MIN_RING_PX = 21         # the ring must hold at least this many pixels


def _span(lo, hi, limit):
    """Pixel index range covering ``[lo, hi]`` (floor / ceil), clipped to ``[0, limit]``."""
    return max(0, int(math.floor(lo))), min(limit, int(math.ceil(hi)))


def _largest_box(face_details):
    """``(x1, y1, x2, y2)`` floats of the face with the largest bbox area, or ``None``."""
    best, best_area = None, -1.0
    for face in face_details:
        try:
            x1, y1, x2, y2 = (float(v) for v in face['bbox'])
        except (KeyError, TypeError, ValueError):
            continue
        area = (x2 - x1) * (y2 - y1)
        if all(math.isfinite(v) for v in (x1, y1, x2, y2)) and math.isfinite(area) and area > best_area:
            best, best_area = (x1, y1, x2, y2), area
    return best


def _clip_box(box, height, width):
    """``box`` clipped to the frame as ``(x1, y1, x2, y2)``, or ``None`` when nothing is left.

    Clipping comes first: a box with a negative corner would otherwise turn the core
    slice into a wrapped negative-index slice that reads the far edge of the frame.
    """
    x1, y1 = max(0.0, box[0]), max(0.0, box[1])
    x2, y2 = min(float(width), box[2]), min(float(height), box[3])
    box_w, box_h = x2 - x1, y2 - y1
    if not (math.isfinite(box_w) and math.isfinite(box_h)) or box_w <= 0 or box_h <= 0:
        return None
    return x1, y1, x2, y2


def _core_luma(gray, clipped):
    """Mean luma (0-1) of the central 60% of a clipped box, or ``None`` below 2x2 px."""
    x1, y1, x2, y2 = clipped
    height, width = gray.shape
    box_w, box_h = x2 - x1, y2 - y1
    inset = SILHOUETTE_CORE_INSET
    cx1, cx2 = _span(x1 + inset * box_w, x2 - inset * box_w, width)
    cy1, cy2 = _span(y1 + inset * box_h, y2 - inset * box_h, height)
    if cx2 - cx1 < SILHOUETTE_MIN_CORE_PX or cy2 - cy1 < SILHOUETTE_MIN_CORE_PX:
        return None
    core = gray[cy1:cy2, cx1:cx2]
    return float(core.sum(dtype=np.float64)) / core.size / 255.0


def silhouette_core_luma(gray, bbox):
    """Mean luma (0-1) of the central 60% of ``bbox`` (clipped to the frame), or ``None``.

    The same core the silhouette rule judges; ``None`` for a box that is empty or outside
    the frame after clipping, or a core under 2x2 px. Exposed so the thumbnail backfill can
    calibrate a stored box against the same measurement the rule makes.
    """
    if gray is None or getattr(gray, 'ndim', 0) != 2:
        return None
    try:
        box = tuple(float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in box):
        return None
    clipped = _clip_box(box, *gray.shape)
    return None if clipped is None else _core_luma(gray, clipped)


def detect_silhouette(gray, face_details):
    """
    Determine whether the subject is a silhouette: the largest detected face is a dark
    patch against a clearly brighter surround.

    The box is clipped to the frame first (InsightFace boxes can be negative or run past
    the edge), so a core slice can never wrap around via a negative index. Then, with
    ``G = gray / 255`` and ``w, h`` the clipped box size::

        core = mean(G over the central 60% of the box)
        ring = mean(G over the box grown by 0.75 w/h per side, minus the box grown by
                    0.2 w/h per side), clipped to the frame
        silhouette = core < 0.12 and ring - core > 0.20

    Returns 0 for no faces (None or empty ``face_details``), a box that is empty or
    outside the frame after clipping, a core under 2x2 px, a ring under 21 px, or any
    non-finite value. Tags and the face count are not inputs: the CLIP ``silhouette``
    tag and the whole-frame histogram are not evidence about the subject.

    Args:
        gray: 2-D uint8 luma plane of the image the boxes were measured on
        face_details: List of dicts with a ``bbox`` of ``[x1, y1, x2, y2]`` pixels in
            ``gray``'s coordinate space (or None)

    Returns:
        int: 1 if silhouette detected, 0 otherwise
    """
    if gray is None or not face_details or getattr(gray, 'ndim', 0) != 2:
        return 0
    box = _largest_box(face_details)
    if box is None:
        return 0
    height, width = gray.shape
    clipped = _clip_box(box, height, width)
    if clipped is None:
        return 0
    core_mean = _core_luma(gray, clipped)
    # A face this bright cannot pass, so skip the ring.
    if core_mean is None or core_mean >= SILHOUETTE_CORE_MAX:
        return 0

    x1, y1, x2, y2 = clipped
    box_w, box_h = x2 - x1, y2 - y1

    def _grown(frac):
        gx1, gx2 = _span(x1 - frac * box_w, x2 + frac * box_w, width)
        gy1, gy2 = _span(y1 - frac * box_h, y2 + frac * box_h, height)
        region = gray[gy1:gy2, gx1:gx2]
        return float(region.sum(dtype=np.float64)), region.size

    outer_sum, outer_n = _grown(SILHOUETTE_RING_OUTER)
    inner_sum, inner_n = _grown(SILHOUETTE_RING_INNER)
    ring_n = outer_n - inner_n
    if ring_n < SILHOUETTE_MIN_RING_PX:
        return 0
    ring_mean = (outer_sum - inner_sum) / ring_n / 255.0

    if not (math.isfinite(core_mean) and math.isfinite(ring_mean)):
        return 0
    return 1 if (core_mean < SILHOUETTE_CORE_MAX
                 and ring_mean - core_mean > SILHOUETTE_CONTRAST_MIN) else 0


# Shared constants from config (with defaults)
DEFAULT_PHOTO_THUMBNAIL_SIZE = 640
DEFAULT_PHOTO_THUMBNAIL_QUALITY = 80
DEFAULT_FACE_PADDING_RATIO = 0.3
DEFAULT_FACE_THUMBNAIL_SIZE = 128
DEFAULT_FACE_THUMBNAIL_QUALITY = 85


def get_shared_constants(config=None):
    """
    Get shared constants from config or use defaults.

    Args:
        config: Optional ScoringConfig instance

    Returns:
        dict: Shared constants for thumbnails and face cropping
    """
    constants = {
        'photo_thumbnail_size': DEFAULT_PHOTO_THUMBNAIL_SIZE,
        'photo_thumbnail_quality': DEFAULT_PHOTO_THUMBNAIL_QUALITY,
        'face_padding_ratio': DEFAULT_FACE_PADDING_RATIO,
        'face_thumbnail_size': DEFAULT_FACE_THUMBNAIL_SIZE,
        'face_thumbnail_quality': DEFAULT_FACE_THUMBNAIL_QUALITY,
    }

    if config is not None:
        proc_settings = config.get_processing_settings()
        thumbnails = proc_settings.get('thumbnails', {})
        constants.update({
            'photo_thumbnail_size': thumbnails.get('photo_size', constants['photo_thumbnail_size']),
            'photo_thumbnail_quality': thumbnails.get('photo_quality', constants['photo_thumbnail_quality']),
            'face_padding_ratio': thumbnails.get('face_padding_ratio', constants['face_padding_ratio']),
        })

        # Face processing settings
        face_proc = config.get_face_processing_settings()
        constants.update({
            'face_thumbnail_size': face_proc.get('face_thumbnail_size', constants['face_thumbnail_size']),
            'face_thumbnail_quality': face_proc.get('face_thumbnail_quality', constants['face_thumbnail_quality']),
        })

    return constants
