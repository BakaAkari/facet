"""Optional OFIQ face quality (ISO/IEC 29794-5) for commercial mode.

Fills ``face_quality_iqa`` when the ``python-ofiq`` package (UNICEF's pybind11
bindings over BSI's OFIQ reference implementation, MIT; import name ``ofiq``) is
installed. Its wheels exist for Linux x86-64 and macOS arm64 only. The OFIQ
models are never bundled: they are fetched on demand into
``pretrained_models/ofiq/`` with a pinned SHA-256. Absent package -> ``available``
is False and ``score_face`` returns None.
"""

import importlib
import logging
import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from models.weights import PRETRAINED_MODELS_DIR, download_weights

log = logging.getLogger(__name__)

OFIQ_MODULE = "ofiq"
OFIQ_VERSION = "1.1.2"
OFIQ_MODELS_URL = "https://standards.iso.org/iso-iec/29794/-5/ed-1/en/OFIQ-MODELS.zip"
OFIQ_MODELS_SHA256 = "8ceb6cec30c95056044e9191bbf8aa26e3e29a639b40a7636a60e13063987f36"
OFIQ_CONFIG_URL = (
    f"https://raw.githubusercontent.com/BSI-OFIQ/OFIQ-Project/v{OFIQ_VERSION}/data/ofiq_config.jaxn"
)
OFIQ_CONFIG_SHA256 = "e117286706d799a1e23130db01cfdff7ef36d157b22ecf68dd196052a8a8d0b3"
OFIQ_CONFIG_NAME = "ofiq_config.jaxn"
DEFAULT_DATA_ROOT = PRETRAINED_MODELS_DIR / "ofiq"
# python-ofiq's scalar_quality() returns the unified score in [0, 100]; Facet is 0-10.
OFIQ_SCALE_MAX = 100.0

_fetch_lock = threading.Lock()


def _ready(root: Path) -> bool:
    return (root / OFIQ_CONFIG_NAME).is_file() and (root / "models").is_dir()


def _fetch_models(root: Path) -> None:
    """Download config and models into ``root``; installed only on completion."""
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=root.parent, prefix=".ofiq-"))
    try:
        download_weights(OFIQ_CONFIG_URL, staging / OFIQ_CONFIG_NAME, OFIQ_CONFIG_SHA256)
        archive = staging / "OFIQ-MODELS.zip"
        download_weights(OFIQ_MODELS_URL, archive, OFIQ_MODELS_SHA256)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        archive.unlink()
        if not (staging / "models").is_dir():
            raise RuntimeError("OFIQ models archive has no models/ directory")
        root.mkdir(parents=True, exist_ok=True)
        old_models = root / "models"
        if old_models.is_dir():
            shutil.rmtree(old_models)
        for item in staging.iterdir():
            os.replace(item, root / item.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class OFIQScorer:
    def __init__(self, data_root=None):
        self.data_root = Path(data_root or os.environ.get("OFIQ_DATA_DIR") or DEFAULT_DATA_ROOT)
        self._engine = None
        self._setup_failed = False
        self._module = None
        self._lock = threading.Lock()  # OFIQ instances are not thread-safe
        try:
            self._module = importlib.import_module(OFIQ_MODULE)
        except ImportError:
            log.info("OFIQ face quality unavailable: '%s' is not installed", OFIQ_MODULE)
        self.available = self._module is not None

    def _get_engine(self):
        if self._engine is None and not self._setup_failed:
            with _fetch_lock:
                if self._engine is None and not self._setup_failed:
                    try:
                        if not _ready(self.data_root):
                            _fetch_models(self.data_root)
                        self._engine = self._module.OFIQ(str(self.data_root))
                    except Exception as exc:
                        self._setup_failed = True  # do not retry once per face
                        log.warning("OFIQ setup failed, face_quality_iqa stays NULL: %s", exc)
        return self._engine

    def score_face(self, crop_bgr) -> float | None:
        """Return the OFIQ unified score mapped to 0-10, or None (no face / unavailable)."""
        if not self.available or crop_bgr is None:
            return None
        engine = self._get_engine()
        if engine is None:
            return None
        try:
            import numpy as np
            rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1], dtype=np.uint8)
            with self._lock:
                raw = engine.scalar_quality(rgb)
        except Exception as exc:  # FaceDetectionError and other QualityError: no score
            log.debug("OFIQ scoring failed: %s", exc)
            return None
        if raw is None:
            return None
        return max(0.0, min(10.0, float(raw) / OFIQ_SCALE_MAX * 10.0))


_scorer: OFIQScorer | None = None


def get_ofiq_scorer() -> OFIQScorer:
    global _scorer
    if _scorer is None:
        _scorer = OFIQScorer()
    return _scorer
