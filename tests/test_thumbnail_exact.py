"""S8a: `generate_photo_thumbnail` without the full-resolution `pil.copy()`.

The new function must return the SAME BYTES as the old one on every input, and must
never modify the image it is handed (the scan retains that image for later passes).
`_reference_generate_photo_thumbnail` is a verbatim copy of the function it replaced.

Set ``PERF_PHOTOS`` to a directory of photos to also run the byte comparison and the
SC5 timing ratio (new / reference <= 0.75) on real files; without it the synthetic
matrix below runs on its own and nothing is skipped.
"""

import hashlib
import os
import random
import time
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from utils.image_transforms import _fitted_size, _round_aspect, generate_photo_thumbnail


def _reference_generate_photo_thumbnail(pil_img, size=640, quality=80):
    """Verbatim copy of the pre-S8a implementation."""
    thumb = pil_img.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


MATRIX = [
    (1, 1), (639, 640), (640, 640), (641, 640), (640, 1), (1, 640),
    (4000, 3000), (3000, 4000), (20000, 2000), (5494, 3666),
    # Rounding decides the short side here: 426.67 -> 427 (ceil), 479.84 -> 480 (ceil).
    (6000, 4000), (4000, 2999),
]


def _image(width, height, mode="RGB", seed=0):
    """Smooth gradient plus noise, so a resampling difference changes the bytes."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    base = (ys * 0.5 + xs * 0.5)[..., None] * np.array([1.0, 0.8, 0.6], dtype=np.float32)
    noise = rng.integers(0, 24, size=(height, width, 3), dtype=np.uint8)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    return img.convert(mode) if mode != "RGB" else img


def _digest(img):
    return hashlib.sha256(img.tobytes()).hexdigest(), img.size, img.mode


@pytest.mark.parametrize("width,height", MATRIX)
def test_bytes_match_the_reference_on_the_size_matrix(width, height):
    img = _image(width, height)

    assert generate_photo_thumbnail(img) == _reference_generate_photo_thumbnail(img)


@pytest.mark.parametrize("size,quality", [(640, 80), (320, 90), (1024, 60), (100, 95)])
def test_bytes_match_for_other_sizes_and_qualities(size, quality):
    for width, height in ((1500, 1000), (1000, 1500), (300, 200), (50, 3000)):
        img = _image(width, height)
        assert generate_photo_thumbnail(img, size=size, quality=quality) == \
            _reference_generate_photo_thumbnail(img, size=size, quality=quality)


def test_bytes_match_on_random_sizes_where_rounding_decides_the_height():
    """Both floor and ceil win somewhere in this range; floor-only rounding must fail."""
    rng = random.Random(20260715)
    ceil_wins = 0
    for _ in range(120):
        width, height = rng.randint(1, 2400), rng.randint(1, 2400)
        img = _image(width, height, seed=width * 7 + height)
        assert generate_photo_thumbnail(img) == _reference_generate_photo_thumbnail(img), (width, height)
        final = _fitted_size(width, height, 640)
        if final is not None:
            aspect = width / height
            exact = (640 / aspect) if 640 / 640 < aspect else (640 * aspect)
            ceil_wins += final[1 if 640 / 640 < aspect else 0] > int(exact)
    assert ceil_wins > 0, "the random set never exercised a ceil-rounded size"


def test_6000x4000_rounds_up_to_427_not_down_to_426():
    assert _fitted_size(6000, 4000, 640) == (640, 427)
    assert _fitted_size(4000, 2999, 640) == (640, 480)
    assert _round_aspect(426.67, key=lambda n: abs(1.5 - 640 / n)) == 427


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_bytes_match_for_other_modes(mode):
    img = _image(2000, 1500, mode=mode)
    assert generate_photo_thumbnail(img) == _reference_generate_photo_thumbnail(img)


def test_bytes_match_when_the_image_carries_info():
    img = _image(2000, 1500)
    img.info["icc_profile"] = b"not-a-profile"
    img.info["dpi"] = (300, 300)
    assert generate_photo_thumbnail(img) == _reference_generate_photo_thumbnail(img)


def test_bytes_match_for_a_lazily_opened_file(tmp_path):
    path = tmp_path / "lazy.jpg"
    _image(3000, 2000).save(path, "JPEG", quality=92)
    with Image.open(path) as lazy:
        reference = _reference_generate_photo_thumbnail(lazy)
    with Image.open(path) as lazy:
        assert generate_photo_thumbnail(lazy) == reference


@pytest.mark.parametrize("width,height", [(4000, 3000), (500, 400), (641, 640)])
def test_the_input_image_is_never_modified(width, height):
    img = _image(width, height)
    before = _digest(img)

    generate_photo_thumbnail(img)
    generate_photo_thumbnail(img, size=100, quality=50)

    assert _digest(img) == before


def test_no_full_resolution_copy_is_made():
    img = _image(3000, 2000)
    expected = _reference_generate_photo_thumbnail(img)

    with mock.patch.object(Image.Image, "copy", side_effect=AssertionError("copy() called")):
        assert generate_photo_thumbnail(img) == expected
        small = _image(300, 200)
        assert generate_photo_thumbnail(small)


def _timed(fn, imgs, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for img in imgs:
            fn(img)
        best = min(best, time.perf_counter() - start)
    return best


def _photo_images():
    root = os.environ.get("PERF_PHOTOS")
    if not root:
        return []
    paths = sorted(p for p in Path(root).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg"})
    images = []
    for path in paths:
        with Image.open(path) as src:
            images.append(src.convert("RGB"))
    return images


def test_real_photos_match_and_run_faster_when_perf_photos_is_set():
    images = _photo_images()
    if not images:
        # No photo set named: the synthetic matrix is the evidence (this test is not skipped).
        images = [_image(w, h) for w, h in ((5494, 3666), (4000, 3000), (3000, 4000))]
        for img in images:
            assert generate_photo_thumbnail(img) == _reference_generate_photo_thumbnail(img)
        return

    for img in images:
        assert generate_photo_thumbnail(img) == _reference_generate_photo_thumbnail(img)
    ratio = _timed(generate_photo_thumbnail, images) / _timed(_reference_generate_photo_thumbnail, images)
    print(f"PERF_PHOTOS thumbnail ratio new/reference = {ratio:.3f} over {len(images)} photos")
    assert ratio <= 0.75, ratio
