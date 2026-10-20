"""Shared SigLIP (transformers) model loader.

One place builds the model so the scan's ModelManager, the single-pass
scorer, ``tag_existing.py`` and ``scripts/compute_aesthetic_clip.py`` produce
identical embeddings. Not covered: ``api/routers/search.py`` loads fp32 on
purpose. On CUDA the
weights are materialised straight into fp16, avoiding the fp32 transient
(~5 GB) that ``from_pretrained`` + ``.half()`` allocated on the card.
CPU and MPS keep fp32 (and need no ``torch`` import here).
"""

from __future__ import annotations

from typing import Any


def load_siglip_model(model_name: str, device: str):
    """Load a transformers SigLIP model onto ``device`` in eval mode.

    Requests ``dtype=torch.float16`` only when ``device`` is ``'cuda'``.
    """
    from transformers import AutoModel

    kwargs: dict[str, Any] = {'trust_remote_code': True}
    if device == 'cuda':
        import torch
        kwargs['dtype'] = torch.float16
    model = AutoModel.from_pretrained(model_name, **kwargs)
    return model.to(device).eval()
