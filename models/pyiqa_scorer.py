"""PyIQA-based image quality assessment models.

Wrapper for pyiqa library models including TOPIQ, TOPIQ-IAA, TOPIQ-NR-Face,
HyperIQA, DBCNN, MUSIQ, and LIQE.
These models provide excellent quality assessment with low VRAM usage.
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import Any, Optional, TypedDict

import licensing
from utils.image_bounds import MAX_INFERENCE_SIZE

logger = logging.getLogger("facet.pyiqa")

# Lazy import to avoid loading pyiqa unless needed
pyiqa = None


def _ensure_pyiqa():
    """Lazy load pyiqa library (refused in commercial licensing mode)."""
    licensing.require('pyiqa')
    global pyiqa
    if pyiqa is None:
        try:
            # imgaug (pyiqa dep) uses np.sctypes removed in NumPy 2.0
            if not hasattr(np, 'sctypes'):
                np.sctypes = {
                    'float': [np.float16, np.float32, np.float64],
                    'int': [np.int8, np.int16, np.int32, np.int64],
                    'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                    'complex': [np.complex64, np.complex128],
                    'others': [bool, object, bytes, str, np.void],
                }
            import pyiqa as _pyiqa
            pyiqa = _pyiqa
        except ImportError:
            raise ImportError(
                "pyiqa is required for TOPIQ/HyperIQA/DBCNN/MUSIQ models. "
                "Install with: pip install pyiqa"
            )
    return pyiqa


# Model info: name, pyiqa_id, vram_gb, lower_is_better, score_range, and the
# optional 'task' (forwarded per call as task_= for multi-head metrics).
class _PyIQARequired(TypedDict):
    pyiqa_id: str
    vram_gb: float
    lower_better: bool
    score_range: tuple[float, float]
    description: str


class PyIQAModelInfo(_PyIQARequired, total=False):
    task: str
    has_distortion: bool


PYIQA_MODELS: dict[str, PyIQAModelInfo] = {
    'topiq': {
        'pyiqa_id': 'topiq_nr',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (0, 1),  # Normalized 0-1
        'description': 'TOPIQ NR - Best accuracy, ResNet50 backbone',
    },
    'hyperiqa': {
        'pyiqa_id': 'hyperiqa',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (0, 1),
        'description': 'HyperIQA - Very efficient, good accuracy',
    },
    'dbcnn': {
        'pyiqa_id': 'dbcnn',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (0, 1),
        'description': 'DBCNN - Dual-branch CNN',
    },
    'musiq': {
        'pyiqa_id': 'musiq',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (0, 100),  # MUSIQ outputs 0-100
        'description': 'MUSIQ - Multi-scale, handles any resolution',
    },
    'musiq-koniq': {
        'pyiqa_id': 'musiq-koniq',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (0, 100),
        'description': 'MUSIQ trained on KonIQ-10k',
    },
    'clipiqa+': {
        'pyiqa_id': 'clipiqa+',
        'vram_gb': 4,
        'lower_better': False,
        'score_range': (0, 1),
        'description': 'CLIP-IQA+ with learned prompts',
    },
    'topiq_iaa': {
        'pyiqa_id': 'topiq_iaa',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (1, 10),  # AVA MOS scale (1-10)
        'description': 'TOPIQ IAA - AVA-trained aesthetic merit scoring',
    },
    'topiq_nr_face': {
        'pyiqa_id': 'topiq_nr-face',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (0, 1),
        'description': 'TOPIQ NR-Face - Purpose-built face quality scoring',
    },
    'liqe': {
        'pyiqa_id': 'liqe',
        'vram_gb': 2,
        'lower_better': False,
        'score_range': (0, 5),  # LIQE outputs 0-5 quality score
        'description': 'LIQE - Quality score + distortion type diagnosis',
        'has_distortion': True,
    },
}


# Models whose forward pass is per-sample independent in eval mode and accept a
# stacked same-size batch tensor, so real batching gives results bit-identical to
# per-image scoring. Resolution-sensitive multi-scale models (musiq*), the
# variable-output LIQE stay serial (forced fixed-size batching would shift their
# scores). topiq_nr_face is serial too: pyiqa's face metric asserts a batch size
# of 1, so a stacked forward could only ever fail and fall back.
_BATCHABLE_MODELS = {'topiq', 'hyperiqa', 'dbcnn', 'topiq_iaa', 'clipiqa+'}

# Failures that are the model's own, legitimate answer for some photos rather than
# a fault: topiq_nr_face declines a photo its detector finds no face in, LIQE a
# photo whose short side is under 224 px. Matched by substring against pyiqa
# 0.1.x text; a future pyiqa that rewords them is reported as a failure (louder,
# never silent).
_EXPECTED_SKIPS = ('No face detected', 'Short side is less than')

# Every out-of-memory error is counted under one key: the raw text embeds
# allocator sizes that differ per run and would defeat the counting.
_OOM_SKIP_KEY = 'out of memory'


def _is_expected_skip(msg: str) -> bool:
    """True for a failure text that is the model's own, deliberate answer."""
    return any(known in msg for known in _EXPECTED_SKIPS)


def _is_oom(e: BaseException) -> bool:
    from utils.device import is_out_of_memory_error
    return is_out_of_memory_error(e)


def _clear_cache(device) -> None:
    # Imported at call time so a test (or caller) patching utils.device is honoured.
    from utils.device import clear_device_cache
    clear_device_cache(device)


class PyIQAScorer:
    """Wrapper for pyiqa image quality assessment models."""

    def __init__(self, model_name: str = 'topiq', device: Optional[str] = None):
        """Initialize PyIQA scorer.

        Args:
            model_name: Model identifier (topiq, hyperiqa, dbcnn, musiq, etc.)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if model_name not in PYIQA_MODELS:
            available = ', '.join(PYIQA_MODELS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        self.model_name = model_name
        self.model_info = PYIQA_MODELS[model_name]
        # Per-call forward kwargs. A spec that pins a 'task' (Q-ReAlign's
        # 'aesthetic' head) must pass it on EVERY inference call — see the
        # CONTRACT note on the qrealign spec above.
        task = self.model_info.get('task')
        self._forward_kwargs = {'task_': task} if task else {}
        if device is None:
            from utils.device import get_device
            device = get_device()
        self.device = device
        self.model: Any = None
        self._loaded = False
        # Skips are counted for the whole pass and reported once by
        # report_skips() when the scorer leaves the process (unload, or a
        # ModelManager dropping it from the RAM cache). Per-batch they are
        # noise: topiq_nr_face legitimately declines every photo without a
        # face, so a landscape-heavy scan logged a warning per chunk that said
        # nothing a single total does not. Failures are different: they are
        # reported at ERROR at the end of every score_batch call.
        self._skipped_totals: dict[str, int] = {}
        # Positions, in the list the last score_batch() call was given, whose
        # None is the model's own deliberate answer (an expected skip) rather
        # than an out-of-memory error or a fault. --recompute-iqa remembers
        # those so it stops re-attempting them; everything else it retries.
        self.last_declined: frozenset[int] = frozenset()
        # Images per stacked forward. Starts at the ceiling and is halved by an
        # out-of-memory error; it stays halved for the rest of this load so a
        # RAM-cached scorer does not re-OOM (and flush the allocator) on every
        # slice of every chunk.
        self.forward_batch = self.MAX_FORWARD_BATCH

    def load(self):
        """Load model to GPU/CPU."""
        if self._loaded:
            return

        _ensure_pyiqa()

        pyiqa_id = self.model_info['pyiqa_id']

        self.model = pyiqa.create_metric(
            pyiqa_id,
            device=torch.device(self.device)
        )
        self.forward_batch = self.MAX_FORWARD_BATCH
        self._loaded = True

    def report_skips(self):
        """Log the skip totals accumulated so far, once, then clear them.

        Idempotent. Called from :meth:`unload` and by ``ModelManager`` for a
        cached scorer it drops without unloading: a RAM-cached model never
        reaches ``unload()``, so its totals would otherwise be lost.
        """
        totals, self._skipped_totals = self._skipped_totals, {}
        for msg, count in totals.items():
            logger.warning("  %s skipped %d image(s) in total: %s",
                           self.model_name, count, msg)

    def unload(self):
        """Unload model to free VRAM."""
        self.report_skips()
        if not self._loaded:
            return

        if self.model is not None:
            # Move to CPU first if on GPU
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None

        self._loaded = False
        self.forward_batch = self.MAX_FORWARD_BATCH
        from utils.device import clear_device_cache
        clear_device_cache(self.device)
        logger.info("  %s unloaded", self.model_name)

    # Max long edge for inference is utils.image_bounds.MAX_INFERENCE_SIZE
    # (ResNet50 at 5496x3670 uses ~10GB per image in FP32).

    # Max images stacked into a single forward pass, even when a same-shape
    # group spans the whole chunk (up to ram_chunk_size, commonly 128). A
    # single-forward group's activation memory scales with the WHOLE group,
    # not just the model's per-image footprint — on a unified-memory device
    # (MPS) that has no separate VRAM ceiling to hit an OOM against, the
    # allocator just grows the process's resident set instead of failing.
    # Slicing each shape group into forwards of this size bounds peak
    # activation memory to a small, constant multiple of one image's cost,
    # independent of chunk size.
    MAX_FORWARD_BATCH = 4

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor for pyiqa.

        Args:
            image: PIL Image (RGB)

        Returns:
            Tensor of shape (1, 3, H, W), normalized to [0, 1]
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if larger than max inference size to prevent OOM
        w, h = image.size
        long_edge = max(w, h)
        if long_edge > MAX_INFERENCE_SIZE:
            scale = MAX_INFERENCE_SIZE / long_edge
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        # Convert to tensor: (H, W, C) -> (C, H, W)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)

    def _normalize_score(self, raw_score) -> float:
        """Normalize score to 0-10 range.

        Args:
            raw_score: Raw score from model (may be tensor, numpy, or scalar)

        Returns:
            Normalized score in 0-10 range as Python float
        """
        # Ensure we have a Python float
        if hasattr(raw_score, 'item'):
            raw_score = raw_score.item()
        raw_score = float(raw_score)

        min_val, max_val = self.model_info['score_range']

        # Clamp to expected range
        raw_score = max(float(min_val), min(float(max_val), raw_score))

        # Normalize to 0-1
        if max_val > min_val:
            normalized = (raw_score - min_val) / (max_val - min_val)
        else:
            normalized = raw_score

        # Scale to 0-10 and ensure Python float
        result = float(normalized * 10.0)

        # Clamp final result to valid range
        return max(0.0, min(10.0, result))

    def score_image(self, image: Image.Image) -> float:
        """Score a single image.

        Args:
            image: PIL Image

        Returns:
            Quality score normalized to 0-10 as Python float
        """
        if not self._loaded:
            self.load()

        img_tensor = self._preprocess_image(image)

        with torch.no_grad():
            raw_score = self.model(img_tensor, **self._forward_kwargs)

        # Extract scalar from tensor - handle various return types
        if isinstance(raw_score, torch.Tensor):
            if raw_score.numel() == 1:
                raw_score = raw_score.item()
            else:
                raw_score = raw_score.mean().item()
        elif hasattr(raw_score, 'item'):
            raw_score = raw_score.item()

        return self._finalize_raw(raw_score)

    def _finalize_raw(self, raw_score) -> float:
        """Apply lower_better inversion then normalize a raw scalar to 0-10."""
        raw_score = float(raw_score)
        if self.model_info['lower_better']:
            min_val, max_val = self.model_info['score_range']
            raw_score = float(max_val) - raw_score + float(min_val)
        return self._normalize_score(raw_score)

    @property
    def supports_batching(self) -> bool:
        """Whether this model can be scored with a stacked same-size batch tensor."""
        return self.model_name in _BATCHABLE_MODELS

    def _extract_batch_raw(self, out, n: int) -> list:
        """Turn a batched model output into a list of n raw scalars.

        Raises ValueError if the output doesn't have n elements so the caller
        falls back to serial scoring rather than silently mis-aligning scores.
        """
        if isinstance(out, torch.Tensor):
            flat = out.detach().cpu().flatten()
            if flat.numel() != n:
                raise ValueError(f"batched output has {flat.numel()} elements, expected {n}")
            return [float(x) for x in flat.tolist()]
        if isinstance(out, (list, tuple)):
            if len(out) != n:
                raise ValueError(f"batched output has {len(out)} elements, expected {n}")
            return [float(x.item()) if hasattr(x, 'item') else float(x) for x in out]
        raise ValueError(f"unbatchable output type: {type(out)}")

    def _score_one(self, image: Image.Image, skipped: dict, declined: set,
                   position: int) -> Optional[float]:
        """Score one image, or ``None`` when it fails (counted into ``skipped``).

        An out-of-memory failure clears the device cache AFTER the handler has
        exited: the live exception's traceback holds the failed activations, so
        freeing from inside the ``except`` would release nothing. An expected
        skip (never an out-of-memory error) also adds ``position`` to ``declined``.
        """
        try:
            return float(self.score_image(image))
        except Exception as e:
            oom = _is_oom(e)
            msg = _OOM_SKIP_KEY if oom else (str(e) or type(e).__name__)
        if oom:
            _clear_cache(self.device)
        elif _is_expected_skip(msg):
            declined.add(position)
        skipped[msg] = skipped.get(msg, 0) + 1
        return None

    def _score_batch_serial(self, images: list[Image.Image], skipped: dict,
                            declined: set) -> list[Optional[float]]:
        """Per-image scoring; an image that fails yields ``None`` in its slot."""
        return [self._score_one(image, skipped, declined, i) for i, image in enumerate(images)]

    def score_batch(self, images: list[Image.Image]) -> list[Optional[float]]:
        """Score a batch of images, normalized to 0-10.

        For batchable models, same-size preprocessed images are stacked into a
        single tensor and scored in one forward pass (bit-identical to per-image
        in eval mode). Non-batchable models and any group that errors fall back
        to per-image scoring.

        An image that cannot be scored yields ``None`` in its slot, never a
        neutral number: a stored 5.0 is indistinguishable from a measurement.
        A CUDA out-of-memory error halves the forward batch (down to one image)
        before giving up on an image.

        After the call, ``last_declined`` holds the positions in ``images`` whose
        ``None`` is an expected skip (the model's own answer: no face found, short
        side under 224 px). An out-of-memory error or any other failure is a
        ``None`` that is NOT in it, so a caller can retry those and remember these.
        """
        self.last_declined = frozenset()
        if not self._loaded:
            self.load()

        skipped: dict[str, int] = {}
        reduced_to: list[int] = []
        declined: set[int] = set()
        if not self.supports_batching or len(images) <= 1:
            scores = self._score_batch_serial(images, skipped, declined)
        else:
            scores = self._score_batched(images, skipped, reduced_to, declined)

        self._record_skips(skipped)
        self._report_failures(skipped)
        if reduced_to:
            logger.warning("  %s forward batch reduced to %d after out-of-memory",
                           self.model_name, min(reduced_to))
        self.last_declined = frozenset(declined)
        return scores

    def _score_batched(self, images, skipped: dict, reduced_to: list,
                       declined: set) -> list[Optional[float]]:
        from collections import defaultdict

        # Preprocess once; group by tensor shape so each forward is a clean stack.
        try:
            tensors = [self._preprocess_image(img) for img in images]
        except Exception as e:
            tensors = None
            oom = _is_oom(e)
        if tensors is None:
            # Serial fallback runs outside the handler so the partial tensors
            # are not pinned by the traceback.
            if oom:
                _clear_cache(self.device)
            return self._score_batch_serial(images, skipped, declined)

        groups: dict = defaultdict(list)
        for idx, t in enumerate(tensors):
            groups[tuple(t.shape[2:])].append(idx)

        scores: list = [None] * len(images)
        for idxs in groups.values():
            # Slice each same-shape group into forwards of at most
            # forward_batch so a chunk-sized group cannot stack the whole chunk
            # into one forward's activation memory. forward_batch is re-read
            # per slice: an out-of-memory halving applies to the rest of it.
            pos = 0
            while pos < len(idxs):
                sub_idxs = idxs[pos:pos + self.forward_batch]
                pos += len(sub_idxs)
                self._forward_group(sub_idxs, tensors, images, scores, skipped, reduced_to,
                                    declined)
        return scores

    def _forward_group(self, sub_idxs, tensors, images, scores, skipped, reduced_to, declined):
        """One stacked forward over ``sub_idxs``; recovers from a failure in place.

        Out-of-memory: halve and retry both halves (a lone image that still does
        not fit is ``None``). Anything else: score each image on its own. Both
        recoveries run after the ``except`` block has exited.
        """
        try:
            with torch.no_grad():
                out = self.model(
                    torch.cat([tensors[i] for i in sub_idxs], dim=0),
                    **self._forward_kwargs,
                )
            raws = self._extract_batch_raw(out, len(sub_idxs))
            for k, i in enumerate(sub_idxs):
                scores[i] = self._finalize_raw(raws[k])
            return
        except Exception as e:
            oom = _is_oom(e)
            msg = str(e)

        if oom:
            _clear_cache(self.device)
            if len(sub_idxs) == 1:
                scores[sub_idxs[0]] = None
                skipped[_OOM_SKIP_KEY] = skipped.get(_OOM_SKIP_KEY, 0) + 1
                return
            half = len(sub_idxs) // 2
            self.forward_batch = min(self.forward_batch, half)
            reduced_to.append(half)
            self._forward_group(sub_idxs[:half], tensors, images, scores, skipped, reduced_to,
                                declined)
            self._forward_group(sub_idxs[half:], tensors, images, scores, skipped, reduced_to,
                                declined)
            return

        # Per-group serial fallback keeps a single bad shape from failing all.
        for i in sub_idxs:
            scores[i] = self._score_one(images[i], skipped, declined, i)
        logger.debug("  %s batch group fell back to serial: %s", self.model_name, msg)

    def _record_skips(self, skipped: dict):
        """Fold this batch's skips into the pass total, logging only at DEBUG."""
        for msg, count in skipped.items():
            self._skipped_totals[msg] = self._skipped_totals.get(msg, 0) + count
            logger.debug("  %s skipped %d image(s): %s", self.model_name, count, msg)

    def _report_failures(self, skipped: dict):
        """Log every skip that is not an expected one, at ERROR, once per call."""
        for msg, count in skipped.items():
            if not _is_expected_skip(msg):
                logger.error("  %s failed for %d image(s) (%s)", self.model_name, count, msg)

    @property
    def vram_gb(self) -> float:
        """Get estimated VRAM requirement in GB."""
        return self.model_info['vram_gb']

    @property
    def description(self) -> str:
        """Get model description."""
        return self.model_info['description']


def get_available_models() -> dict:
    """Get dict of available pyiqa models with their info."""
    return PYIQA_MODELS.copy()


def select_best_model(available_vram_gb: float) -> str:
    """Select best quality model based on available VRAM.

    Args:
        available_vram_gb: Available GPU VRAM in GB

    Returns:
        Model name to use
    """
    # Priority order: best accuracy first
    priority = ['topiq', 'hyperiqa', 'dbcnn', 'musiq-koniq', 'musiq', 'clipiqa+']

    for model_name in priority:
        info = PYIQA_MODELS[model_name]
        if info['vram_gb'] <= available_vram_gb:
            return model_name

    # Fallback to most lightweight
    return 'topiq'
