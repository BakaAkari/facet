"""Q-ReAlign scorer loaded directly through ``transformers`` (no pyiqa).

The model is asked to rate a picture; the logits it puts on the five level words
(excellent / good / fair / poor / bad) at the last position are softmaxed and
collapsed with fixed weights into a score in [0, 1], reported here on 0-10.
"""

from __future__ import annotations

LEVELS = ("excellent", "good", "fair", "poor", "bad")
WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)

MODELS = {
    'mini': 'q-future/Q-ReAlign-Mini-0.8B',
    'lite': 'q-future/Q-ReAlign-Lite-4B',
}

# task -> (user question, assistant answer stem)
TASKS = {
    'quality': ('Can you rate the quality of this picture?', 'The quality of the image is'),
    'aesthetic': ('Can you rate the aesthetics of this picture?', 'The aesthetics of the image is'),
}


def level_token_ids(tokenizer, words=LEVELS):
    """First token id of each level word as it appears after the stem (space-prefixed)."""
    ids = []
    for word in words:
        toks = tokenizer(' ' + word, add_special_tokens=False)['input_ids']
        if not toks:
            raise ValueError(f'level word {word!r} did not tokenize to any id')
        ids.append(int(toks[0]))
    return ids


def levels_to_score(level_logits) -> float:
    """Softmax over the five level logits, weighted sum, scaled to 0-10."""
    import torch
    logits = torch.as_tensor(level_logits).float().reshape(-1)
    probs = torch.softmax(logits, dim=-1)
    weights = torch.tensor(WEIGHTS, dtype=probs.dtype, device=probs.device)
    return float((probs * weights).sum().item()) * 10.0


class QReAlignNativeScorer:
    """Q-ReAlign Mini/Lite scorer on plain transformers."""

    # Same contract as ``PyIQAScorer.last_declined`` (positions the model refused on
    # purpose). This scorer has no expected skip and never returns None, so it is
    # always empty; declaring it keeps --recompute-iqa free of a getattr.
    last_declined = frozenset()

    def __init__(self, size='mini', device='cuda', dtype='auto'):
        if size not in MODELS:
            raise ValueError(f'unknown size {size!r}; expected one of {tuple(MODELS)}')
        self.size = size
        self.repo = MODELS[size]
        self.device = device
        # 'auto' loads the checkpoint's bf16, which a CPU without AMX/AVX512-BF16
        # runs far slower than fp32 -- so the CPU default is fp32.
        if dtype == 'auto' and str(device).split(':')[0] == 'cpu':
            dtype = 'float32'
        self.dtype = dtype
        self.processor = None
        self.model = None
        self._level_ids = None

    @staticmethod
    def _check_task(task):
        if task not in TASKS:
            raise ValueError(f'unknown task {task!r}; expected one of {tuple(TASKS)}')

    def load(self):
        if self.model is not None:
            return self
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.repo)
        model = AutoModelForImageTextToText.from_pretrained(self.repo, dtype=self.dtype)
        self.model = model.to(self.device).eval()
        self._level_ids = level_token_ids(self.processor.tokenizer)
        return self

    def _prepare(self, pil):
        from utils.image_bounds import bound_to_inference_size
        return bound_to_inference_size(pil.convert('RGB'))

    def _score_one(self, img, task):
        import torch
        question, stem = TASKS[task]
        messages = [{'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': question},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True) + stem
        inputs = self.processor(text=[text], images=[img], return_tensors='pt').to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        ids = torch.as_tensor(self._level_ids, dtype=torch.long, device=logits.device)
        return levels_to_score(logits[ids])

    def score(self, pil, task='aesthetic') -> float:
        return self.score_tasks(pil, (task,))[task]

    def score_tasks(self, pil, tasks=('aesthetic', 'quality')) -> dict:
        for task in tasks:
            self._check_task(task)
        self.load()
        img = self._prepare(pil)
        return {task: self._score_one(img, task) for task in tasks}

    def score_batch(self, pil_images, task='aesthetic') -> list:
        """Scores for several images (same interface as ``PyIQAScorer.score_batch``)."""
        return [self.score(pil, task) for pil in pil_images]

    def unload(self):
        self.model = None
        self.processor = None
        self._level_ids = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
