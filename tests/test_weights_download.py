"""download_weights must refuse a body shorter than the declared Content-Length."""

import hashlib
import io

import pytest

from models import weights


class _Resp(io.BytesIO):
    def __init__(self, body, content_length):
        super().__init__(body)
        self.headers = {} if content_length is None else {'Content-Length': str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch(monkeypatch, resp):
    monkeypatch.setattr("models.weights.urllib.request.urlopen",
                        lambda url, timeout=None: resp)


@pytest.mark.parametrize('pin_sha', [False, True])
def test_truncated_body_is_refused_with_or_without_sha(tmp_path, monkeypatch, pin_sha):
    dest = tmp_path / "w.bin"
    body = b"x" * 100
    sha = hashlib.sha256(body).hexdigest() if pin_sha else None
    _patch(monkeypatch, _Resp(body, 500))
    with pytest.raises(ValueError, match="truncated"):
        weights.download_weights("http://example/w.bin", dest, sha)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_complete_body_is_installed(tmp_path, monkeypatch):
    dest = tmp_path / "w.bin"
    _patch(monkeypatch, _Resp(b"y" * 64, 64))
    weights.download_weights("http://example/w.bin", dest)
    assert dest.read_bytes() == b"y" * 64


def test_missing_content_length_is_still_accepted(tmp_path, monkeypatch):
    dest = tmp_path / "w.bin"
    _patch(monkeypatch, _Resp(b"z" * 8, None))
    weights.download_weights("http://example/w.bin", dest)
    assert dest.exists()
