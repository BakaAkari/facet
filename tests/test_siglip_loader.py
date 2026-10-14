"""The shared SigLIP loader requests fp16 only on CUDA, and every site uses it."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utils.siglip import load_siglip_model


@pytest.fixture
def fake_auto(monkeypatch):
    pytest.importorskip('torch', reason='load_siglip_model imports torch')
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    auto = MagicMock()
    auto.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoModel=auto))
    return auto, model


def test_cuda_requests_fp16(fake_auto):
    torch = pytest.importorskip('torch')
    auto, model = fake_auto
    out = load_siglip_model('m', 'cuda')
    assert auto.from_pretrained.call_args.kwargs['dtype'] is torch.float16
    model.to.assert_called_once_with('cuda')
    model.half.assert_not_called()
    assert out is model


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_non_cuda_keeps_fp32(fake_auto, device):
    auto, _ = fake_auto
    load_siglip_model('m', device)
    assert 'dtype' not in auto.from_pretrained.call_args.kwargs


def test_cpu_path_needs_no_torch(monkeypatch):
    """The CPU/MPS path must not import torch (CI's Python job has none).

    ``tag_existing.build_clip_tagger`` reaches the helper on CPU in tests that
    run where torch is absent; only the CUDA branch needs ``torch.float16``.
    """
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    auto = MagicMock()
    auto.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoModel=auto))
    monkeypatch.setitem(sys.modules, 'torch', None)  # `import torch` -> ImportError
    out = load_siglip_model('m', 'cpu')
    assert out is model
    assert 'dtype' not in auto.from_pretrained.call_args.kwargs


# Every file allowed to call ``AutoModel.from_pretrained`` for SigLIP itself.
# Anything else must go through ``utils.siglip.load_siglip_model``.
SIGLIP_DIRECT_LOAD_ALLOW_LIST = [
    # text-only query encoder, deliberately fp32 (see the comment there)
    'api/routers/search.py',
    # the shared loader itself
    'utils/siglip.py',
]


def test_all_siglip_sites_use_helper():
    """No file in the repo loads a transformers model directly, bar the allow-list.

    ``tests/`` is excluded (this file and ``test_tagging.py`` contain the
    string); ``conda/`` is excluded because it is untracked and ``--untracked``
    would otherwise scan the whole local environment.
    """
    repo_root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        ['git', 'grep', '--untracked', '-l', 'AutoModel.from_pretrained', '--', '*.py',
         ':(exclude)tests', ':(exclude)conda'],
        capture_output=True, text=True, cwd=repo_root, check=True).stdout.split()
    assert sorted(out) == sorted(SIGLIP_DIRECT_LOAD_ALLOW_LIST)
