"""Tests for the pyiqa-free Q-ReAlign scorer (stub logits; optional pyiqa parity)."""
import os

import pytest
import torch
from PIL import Image

from models import qrealign_native as qn


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        # " excellent" -> [100 + idx, 7]: only the first token matters
        idx = {' ' + w: i for i, w in enumerate(qn.LEVELS)}[text]
        return {'input_ids': [100 + idx, 7]}


class _Inputs(dict):
    def to(self, device):
        return self


class _Proc:
    tokenizer = _Tok()

    def __init__(self):
        self.texts, self.images = [], []

    def apply_chat_template(self, messages, add_generation_prompt):
        assert add_generation_prompt
        assert messages[0]['content'][0] == {'type': 'image'}
        return 'CHAT[' + messages[0]['content'][1]['text'] + ']'

    def __call__(self, text, images, return_tensors):
        self.texts.append(text[0])
        self.images.append(images[0])
        return _Inputs(x=1)


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Model:
    def __init__(self, level_logits):
        self.level_logits = level_logits

    def __call__(self, **kw):
        out = torch.zeros(1, 3, 200)
        out[0, -1, 100:105] = torch.tensor(self.level_logits)
        out[0, 0, 100:105] = 99.0  # earlier positions must be ignored
        return _Out(out)


def _scorer(level_logits):
    s = qn.QReAlignNativeScorer('mini', device='cpu')
    s.processor = _Proc()
    s.model = _Model(level_logits)
    s._level_ids = qn.level_token_ids(s.processor.tokenizer)
    return s


def test_first_token_of_space_prefixed_word():
    assert qn.level_token_ids(_Tok()) == [100, 101, 102, 103, 104]


def test_all_mass_on_excellent_is_ten():
    assert _scorer([50, 0, 0, 0, 0]).score(Image.new('RGB', (8, 8))) == pytest.approx(10.0, abs=1e-4)


def test_all_mass_on_bad_is_zero():
    assert _scorer([0, 0, 0, 0, 50]).score(Image.new('RGB', (8, 8))) == pytest.approx(0.0, abs=1e-4)


def test_uniform_is_five():
    assert _scorer([0, 0, 0, 0, 0]).score(Image.new('RGB', (8, 8))) == pytest.approx(5.0)


def test_known_vector_exact():
    logits = [1.0, 2.0, 0.5, -1.0, 0.0]
    p = torch.softmax(torch.tensor(logits), -1)
    expected = float((p * torch.tensor([1, .75, .5, .25, 0])).sum()) * 10
    assert _scorer(logits).score(Image.new('RGB', (8, 8))) == pytest.approx(expected, abs=1e-5)


def test_task_prompt_selection_and_shared_preprocessing():
    s = _scorer([0, 0, 0, 0, 0])
    res = s.score_tasks(Image.new('RGB', (8, 8)), ('aesthetic', 'quality'))
    assert set(res) == {'aesthetic', 'quality'}
    assert s.processor.texts == [
        'CHAT[Can you rate the aesthetics of this picture?]The aesthetics of the image is',
        'CHAT[Can you rate the quality of this picture?]The quality of the image is',
    ]
    assert s.processor.images[0] is s.processor.images[1]


def test_invalid_task_and_size():
    with pytest.raises(ValueError):
        _scorer([0] * 5).score(Image.new('RGB', (8, 8)), task='bogus')
    with pytest.raises(ValueError):
        qn.QReAlignNativeScorer('huge')


def test_image_bounded_to_inference_size():
    s = _scorer([0] * 5)
    s.score(Image.new('RGB', (4000, 2000)))
    assert max(s.processor.images[0].size) == 1024


@pytest.mark.skipif(os.environ.get('FACET_QREALIGN_PARITY') != '1', reason='set FACET_QREALIGN_PARITY=1')
def test_parity_with_pyiqa():
    pyiqa = pytest.importorskip('pyiqa')
    imgs = [Image.effect_noise((64, 48), 40).convert('RGB'),
            Image.new('RGB', (80, 60), (200, 30, 30)),
            Image.linear_gradient('L').resize((72, 72)).convert('RGB')]
    metric = pyiqa.create_metric('qrealign', device=torch.device('cpu'), as_loss=False)
    native = qn.QReAlignNativeScorer('mini', device='cpu').load()
    from torchvision.transforms.functional import to_tensor
    worst = 0.0
    for img in imgs:
        for task in ('aesthetic', 'quality'):
            ref = float(metric(to_tensor(img).unsqueeze(0), task_=task).item()) * 10
            worst = max(worst, abs(ref - native.score(img, task)))
    print('PARITY max abs diff (0-10 scale):', worst)
    assert worst < 1e-2  # 1e-3 on the 0-1 scale
