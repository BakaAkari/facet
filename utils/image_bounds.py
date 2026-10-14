"""The long-edge bound every quality scorer applies to its input image.

Kept apart from ``models.pyiqa_scorer`` and ``optimization`` so the production
Q-ReAlign scorer and the evaluation harnesses can share one definition without
either importing the other (or anything heavy).
"""

# PyIQA models are trained on <=1024px images; larger adds no benefit but explodes
# intermediate activation memory. ``PyIQAScorer`` imports this constant.
MAX_INFERENCE_SIZE = 1024


def bounded_size(w, h, max_side):
    """Target (w, h) after the long-edge bound; unchanged if within it."""
    long_edge = max(w, h)
    if long_edge <= max_side:
        return w, h
    scale = max_side / long_edge
    return int(w * scale), int(h * scale)


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
    return img.resize((w, h), Image.LANCZOS)
