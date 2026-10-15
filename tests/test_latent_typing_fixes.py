"""Regressions for the four latent defects the mypy-zero pass surfaced.

Each one was wrong in principle but unreachable from any caller that exists today,
so none of them had a failing test until the type checker pointed at them:

* ``Facet.rescan_samp_composition`` used the lazy ``cv2`` / ``Image`` globals
  without loading them (its sibling ``recompute_composition_scores`` does).
* ``create_samp_scorer`` returns ``None`` on failure but was annotated as
  always returning a scorer.
* ``write_metadata`` re-probed ``exiftool -ver`` to build its error message, and
  a second probe that returned ``None`` raised ``TypeError`` instead of the
  ``RuntimeError`` ``export_sidecars`` catches.
* ``_build_safe_url`` returned a ``None`` hostname as the ``Host`` header.
"""

import sys
import types
import typing

import pytest

from db import get_connection
from db.schema import init_database


# ---------------------------------------------------------------------------
# rescan_samp_composition loads its own image modules
# ---------------------------------------------------------------------------

class _StubSampScorer:
    def __init__(self, *args, **kwargs):
        pass

    def score_batch(self, images):
        return [{"comp_score": 7.5, "pattern": "rule_of_thirds"} for _ in images]


@pytest.mark.timeout(60)
def test_rescan_samp_composition_loads_image_modules_itself(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    import processing.scorer as scorer_mod

    ok, buf = cv2.imencode(".jpg", np.full((400, 400, 3), 128, dtype=np.uint8))
    assert ok

    db_path = str(tmp_path / "scores.db")
    init_database(db_path)
    with get_connection(db_path, row_factory=False) as conn:
        conn.execute(
            "INSERT INTO photos (path, filename, thumbnail, comp_score) VALUES (?, ?, ?, ?)",
            ("/photos/a.jpg", "a.jpg", buf.tobytes(), 0.0),
        )
        conn.commit()

    # Stub the GPU scorer so the test needs neither torch nor SAMP-Net weights.
    stub = types.ModuleType("models.samp_net")
    stub.SAMPNetScorer = _StubSampScorer
    monkeypatch.setitem(sys.modules, "models.samp_net", stub)

    # Put the lazy globals back to their never-loaded state, as in a fresh
    # process running Facet(lightweight=True) with no prior loader call. A test
    # that ran earlier in the session may have loaded them.
    monkeypatch.setattr(scorer_mod, "cv2", None)
    monkeypatch.setattr(scorer_mod, "Image", None)

    facet = scorer_mod.Facet(db_path=db_path, config_path=None, lightweight=True)
    facet.rescan_samp_composition(batch_size=4)

    with get_connection(db_path, row_factory=False) as conn:
        row = conn.execute(
            "SELECT comp_score, composition_pattern FROM photos WHERE path = ?",
            ("/photos/a.jpg",),
        ).fetchone()
    assert tuple(row) == (7.5, "rule_of_thirds")


# ---------------------------------------------------------------------------
# create_samp_scorer: None on failure is part of the contract
# ---------------------------------------------------------------------------

def test_create_samp_scorer_none_is_declared_and_returned(monkeypatch):
    pytest.importorskip("torch")
    import models.samp_net as samp_net

    ret = typing.get_type_hints(samp_net.create_samp_scorer)["return"]
    assert type(None) in typing.get_args(ret)

    class _Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights unavailable")

    monkeypatch.setattr(samp_net, "SAMPNetScorer", _Broken)
    assert samp_net.create_samp_scorer({}) is None


# ---------------------------------------------------------------------------
# write_metadata probes the exiftool version once
# ---------------------------------------------------------------------------

def test_write_metadata_error_reuses_the_first_version_probe(monkeypatch):
    import processing.xmp_export as xe

    probes = []

    def _flaky_version(exe):
        probes.append(exe)
        # The first probe saw the defective 13.25; any re-probe would see nothing.
        return (13, 25) if len(probes) == 1 else None

    monkeypatch.setattr(xe, "_resolve_exiftool", lambda: "exiftool")
    monkeypatch.setattr(xe, "_exiftool_version", _flaky_version)

    # export_sidecars catches (OSError, RuntimeError); a TypeError would escape it.
    with pytest.raises(RuntimeError, match=r"exiftool 13\.25 .*13\.28"):
        xe.write_metadata("a.jpg", xe.XmpRating(star_rating=3))
    assert len(probes) == 1


def test_exiftool_defective_version_bounds(monkeypatch):
    import processing.xmp_export as xe

    monkeypatch.setattr(xe, "_resolve_exiftool", lambda: "exiftool")
    for raw, expected in [((13, 22), None), ((13, 23), (13, 23)), ((13, 27), (13, 27)),
                          ((13, 28), None), (None, None)]:
        monkeypatch.setattr(xe, "_exiftool_version", lambda exe, v=raw: v)
        assert xe.exiftool_defective_version() == expected

    monkeypatch.setattr(xe, "_resolve_exiftool", lambda: None)
    assert xe.exiftool_defective_version() is None


# ---------------------------------------------------------------------------
# _build_safe_url never hands back a None Host header
# ---------------------------------------------------------------------------

def test_build_safe_url_rejects_a_hostless_url():
    from plugins import PluginManager

    with pytest.raises(ValueError, match="No hostname"):
        PluginManager._build_safe_url("http:///hook", "93.184.216.34")


def test_build_safe_url_returns_the_hostname_for_a_normal_url():
    from plugins import PluginManager

    safe_url, hostname = PluginManager._build_safe_url("https://example.com/hook?a=1", "93.184.216.34")
    assert safe_url == "https://93.184.216.34:443/hook?a=1"
    assert hostname == "example.com"
