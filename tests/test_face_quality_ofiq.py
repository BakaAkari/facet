import sys
import types
import zipfile

import numpy as np
import pytest

from analyzers import face_quality_ofiq as fq


class _QualityError(Exception):
    pass


def _stub(scalar=None, seen=None):
    """Mirrors python-ofiq 0.3.1 as observed: OFIQ(config_dir).scalar_quality(rgb uint8)
    -> float in [0, 100], raising a QualityError subclass when no face is found."""
    mod = types.ModuleType("ofiq")

    class OFIQ:
        def __init__(self, config_dir="", config_file="ofiq_config.jaxn"):
            if seen is not None:
                seen["root"] = config_dir

        def scalar_quality(self, image_rgb):
            assert image_rgb.dtype == np.uint8 and image_rgb.flags["C_CONTIGUOUS"]
            if seen is not None:
                seen["pixel"] = tuple(int(v) for v in image_rgb[0, 0])
            if scalar is None:
                raise _QualityError("FaceDetectionError: No faces were detected")
            return scalar

    mod.OFIQ = OFIQ
    return mod


CROP = np.zeros((32, 32, 3), np.uint8)
CROP[..., 0] = 10  # B
CROP[..., 2] = 30  # R


@pytest.fixture
def ready(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / fq.OFIQ_CONFIG_NAME).write_text("{}")
    return tmp_path


def test_stub_score_mapping_model_path_and_rgb_order(monkeypatch, ready):
    seen = {}
    monkeypatch.setitem(sys.modules, "ofiq", _stub(73.0, seen))
    s = fq.OFIQScorer(data_root=ready)
    assert s.available
    assert s.score_face(CROP) == pytest.approx(7.3)
    assert seen["root"] == str(ready)
    assert seen["pixel"] == (30, 0, 10)  # BGR crop handed over as RGB


def test_no_face_returns_none_not_zero(monkeypatch, ready):
    monkeypatch.setitem(sys.modules, "ofiq", _stub(None))
    assert fq.OFIQScorer(data_root=ready).score_face(CROP) is None


def test_setup_failure_cached(monkeypatch, tmp_path):
    calls = []

    def boom(root):
        calls.append(root)
        raise OSError("offline")

    monkeypatch.setitem(sys.modules, "ofiq", _stub(50.0))
    monkeypatch.setattr(fq, "_fetch_models", boom)
    s = fq.OFIQScorer(data_root=tmp_path)
    assert s.score_face(CROP) is None
    assert s.score_face(CROP) is None
    assert calls == [tmp_path]


def test_absent_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "ofiq", None)  # forces ImportError
    s = fq.OFIQScorer()
    assert s.available is False
    assert s.score_face(CROP) is None


@pytest.mark.parametrize("raw,expected", [(150.0, 10.0), (-5.0, 0.0)])
def test_scale_clamped(monkeypatch, ready, raw, expected):
    monkeypatch.setitem(sys.modules, "ofiq", _stub(raw))
    assert fq.OFIQScorer(data_root=ready).score_face(CROP) == expected


def test_fetch_pins_hashes_and_installs_atomically(monkeypatch, tmp_path):
    pins = {}

    def fake_dl(url, dest, sha256=None):
        pins[dest.name] = sha256
        if dest.name == fq.OFIQ_CONFIG_NAME:
            dest.write_text("{}")
        else:
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("models/a.onnx", "x")

    monkeypatch.setattr(fq, "download_weights", fake_dl)
    root = tmp_path / "ofiq"
    fq._fetch_models(root)
    assert fq._ready(root)
    assert pins == {fq.OFIQ_CONFIG_NAME: fq.OFIQ_CONFIG_SHA256, "OFIQ-MODELS.zip": fq.OFIQ_MODELS_SHA256}
    assert [p.name for p in tmp_path.iterdir()] == ["ofiq"]  # staging dir cleaned up
