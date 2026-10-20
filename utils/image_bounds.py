"""Input-size contracts every quality scorer shares: the long-edge bound and LIQE's floor.

Kept apart from ``models.pyiqa_scorer`` and ``optimization`` so the production
Q-ReAlign scorer, ``--recompute-iqa`` and the evaluation harnesses can share one
definition without either importing the other (or anything heavy).
"""

# PyIQA models are trained on <=1024px images; larger adds no benefit but explodes
# intermediate activation memory. ``PyIQAScorer`` imports this constant.
MAX_INFERENCE_SIZE = 1024

# LIQE asserts a short side of at least this many pixels on the tensor it is given
# ("Short side is less than 224", pyiqa/archs/liqe_arch.py).
LIQE_MIN_SHORT_SIDE = 224


def bounded_size(w, h, max_side):
    """Target (w, h) after the long-edge bound; unchanged if within it."""
    long_edge = max(w, h)
    if long_edge <= max_side:
        return w, h
    scale = max_side / long_edge
    return int(w * scale), int(h * scale)


def liqe_input_too_small(w, h):
    """True when LIQE would refuse an image of this size, decided from its dimensions alone.

    LIQE sees the image after ``PyIQAScorer._preprocess_image`` bounded its long
    edge to ``MAX_INFERENCE_SIZE`` with ``int(w * scale)`` truncation, which is
    exactly ``bounded_size``. Applying the same bound first makes this check
    EXACT rather than a heuristic: a 2000x230 image is refused because it is
    scored at 1024x117, and a 1100x241 one is not because it is scored at 1024x224.
    """
    bw, bh = bounded_size(w, h, MAX_INFERENCE_SIZE)
    return min(bw, bh) < LIQE_MIN_SHORT_SIDE


def bound_to_inference_size(img):
    """Mirror the production input contract before scoring.

    PyIQAScorer bounds every image to ``MAX_INFERENCE_SIZE`` on the long edge
    before any pyiqa metric sees it (models/pyiqa_scorer.py). Feeding the
    Q-ReAlign scorer native-resolution originals instead measured it encoding
    24MP frames whole -- 23s/img and ~13GB of activations for a 0.8B model --
    a regime no shipped code path ever enters. The evaluation harnesses apply
    the same bound to both arms so the latency half of the ship gate compares
    what production runs.
    """
    from PIL import Image
    w, h = bounded_size(*img.size, MAX_INFERENCE_SIZE)
    if (w, h) == img.size:
        return img
    return img.resize((w, h), Image.Resampling.LANCZOS)
