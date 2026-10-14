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


def test_all_siglip_sites_use_helper():
    repo_root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        ['git', 'grep', '--untracked', '-ln', 'AutoModel.from_pretrained', '--', 'models/', 'processing/',
         'scripts/', 'utils/'],
        capture_output=True, text=True, cwd=repo_root).stdout.split()
    assert out == ['utils/siglip.py']
