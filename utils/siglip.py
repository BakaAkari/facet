"""Shared SigLIP (transformers) model loader.

One place builds the model so every load site (ModelManager, the legacy
single-pass scorer, scripts) produces identical embeddings. On CUDA the
weights are materialised straight into fp16, avoiding the fp32 transient
(~5 GB) that ``from_pretrained`` + ``.half()`` allocated on the card.
CPU and MPS keep fp32.
"""

from __future__ import annotations


def load_siglip_model(model_name: str, device: str):
    """Load a transformers SigLIP model onto ``device`` in eval mode.

    Requests ``dtype=torch.float16`` only when ``device`` is ``'cuda'``.
    """
    import torch
    from transformers import AutoModel

    kwargs = {'trust_remote_code': True}
    if device == 'cuda':
        kwargs['dtype'] = torch.float16
    model = AutoModel.from_pretrained(model_name, **kwargs)
    return model.to(device).eval()
