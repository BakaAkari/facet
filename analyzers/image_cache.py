"""
Image cache for pre-computed transformations.
"""

import cv2


def laplacian_variance(gray):
    """Population variance of the 3x3 Laplacian of a uint8 grey image.

    Computed from a 16-bit signed Laplacian (exact: +-1020 at most) and ``cv2.meanStdDev``,
    instead of a float64 Laplacian plus ``.var()``: ~5x faster and no 8 byte/pixel buffer.
    Agrees with the float64 pipeline to ~6e-15 relative; it is not bit-identical.
    Returns a Python ``float``.
    """
    _, std = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
    return float(std[0][0]) ** 2


class ImageCache:
    """
    Pre-computes and caches commonly needed image transformations.

    Creating grayscale/HSV conversions and Laplacian variance once per image
    avoids redundant computation across multiple analyzer methods.

    Usage:
        cache = ImageCache(img_cv)
        sharpness_data = tech_analyzer.get_sharpness_data(img_cv, cache=cache)
        color_data = tech_analyzer.get_color_harmony_data(img_cv, cache=cache)
    """
    __slots__ = ['gray', 'hsv', 'laplacian_variance', 'height', 'width']

    def __init__(self, img_cv):
        """
        Initialize cache with pre-computed transformations.

        Args:
            img_cv: OpenCV BGR image array
        """
        self.height, self.width = img_cv.shape[:2]
        self.gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        self.laplacian_variance = laplacian_variance(self.gray)

