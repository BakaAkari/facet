"""A photo with no subject box stores a NULL ``power_point_score``, never a made-up 5.0.

``CompositionAnalyzer.get_placement_data`` has no power-point distance to measure when
``detect_subject_region`` finds no box, and used to answer a neutral 5.0 that was stored as
if measured (range filters, ``AVG`` and the sort could not tell it from a real score). It now
answers ``None``. This pins the three scan sites that persist it (``BatchProcessor``, the
single-image ``score_photo_from_pil`` and the multi-pass ``_compute_aggregates`` +
``_save_results``) and the aggregate reader, which still scores a NULL as the neutral 5.0 so
no aggregate moves.
"""

from unittest import mock

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("cv2")

from analyzers import CompositionAnalyzer  # noqa: E402
from db import get_connection  # noqa: E402
from tests import test_single_pass_preserves as _sp  # noqa: E402
from tests.test_composition_hoist import (  # noqa: E402
    _SpyScorer, _load, _processor as _hoist_processor, _save, _structured_image,
)

seeded = _sp.seeded  # the fixture, shared rather than copied
_images, _processor = _sp._images, _sp._processor

SIZE = 160


def _flat_image():
    """A uniform grey frame: ``detect_subject_region`` finds nothing to box."""
    return np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)


def _flat_path(tmp_path, name='flat.png'):
    path = tmp_path / name
    Image.fromarray(_flat_image()).save(path)
    return str(path)


def _structured_path(tmp_path, name='structured.jpg'):
    return _save(tmp_path, name, _structured_image(3))


class TestAnalyzer:
    def test_a_flat_image_has_no_box_and_no_power_point_score(self):
        import cv2
        img = cv2.cvtColor(_flat_image(), cv2.COLOR_RGB2BGR)
        assert CompositionAnalyzer.detect_subject_region(img) is None
        data = CompositionAnalyzer.get_placement_data(None, SIZE, SIZE, None, img)
        assert data['power_point_score'] is None
        assert data['score'] == 7.0   # the centre assumption is unchanged
        assert data['line_score'] == 5.0

    def test_a_structured_image_keeps_a_measured_power_point_score(self):
        import cv2
        img = cv2.cvtColor(_structured_image(3), cv2.COLOR_RGB2BGR)
        data = CompositionAnalyzer.get_placement_data(None, img.shape[1], img.shape[0], None, img)
        assert isinstance(data['power_point_score'], float)


class TestBatchProcessorSite:
    """The batch path hands ``get_placement_data`` no pixels, so a faceless photo has no box."""

    @pytest.mark.timeout(30)
    def test_a_faceless_photo_saves_null(self, tmp_path):
        from tests.test_batch_processor_e2e import FakeScorer, _make_processor
        flat = _flat_path(tmp_path)
        structured = _structured_path(tmp_path)
        scorer = FakeScorer(tmp_path)      # its face analyzer reports no face
        _make_processor(scorer).process_files([flat, structured], show_metrics=False)
        saved = {r['filename']: r['power_point_score'] for r in scorer.saved}
        assert saved == {'flat.png': None, 'structured.jpg': None}

    @pytest.mark.timeout(30)
    def test_a_photo_with_a_face_box_saves_a_measured_float(self, tmp_path):
        from tests.test_batch_processor_e2e import FakeScorer, _make_processor
        from tests.test_silhouette_rule import _OneFaceAnalyzer, _save_png, _site_image
        path = _save_png(tmp_path, 'face.png', _site_image(False))
        scorer = FakeScorer(tmp_path)
        scorer.face_analyzer = _OneFaceAnalyzer()
        _make_processor(scorer).process_files([path], show_metrics=False)
        assert [type(r['power_point_score']) for r in scorer.saved] == [float]


class TestSingleImageSite:
    """``score_photo_from_pil`` hands the pixels over, so only a flat frame has no box."""

    def _score(self, tmp_path, with_face):
        from tests.test_batch_processor_e2e import _FakeFaceAnalyzer
        from tests.test_silhouette_rule import TestSingleImageSite as _Site
        from utils import load_image_from_path
        scorer = _Site()._scorer(tmp_path)
        if not with_face:
            scorer.face_analyzer = _FakeFaceAnalyzer()
        pil, cv = load_image_from_path(_flat_path(tmp_path, 'single.png'))
        return scorer.score_photo_from_pil(pil, cv, str(tmp_path / 'single.png'))

    def test_a_flat_faceless_photo_returns_null(self, tmp_path):
        assert self._score(tmp_path, with_face=False)['power_point_score'] is None

    def test_a_photo_with_a_face_box_returns_a_measured_float(self, tmp_path):
        assert isinstance(self._score(tmp_path, with_face=True)['power_point_score'], float)


class TestMultiPassSite:
    def test_load_and_compute_aggregates_carry_the_none_through(self, tmp_path):
        scorer = _SpyScorer(str(tmp_path / 'chunk.db'), mock.MagicMock())
        from config.scoring_config import ScoringConfig
        from config_resolve import defaults_path
        from db.schema import init_database
        init_database(scorer.db_path)
        scorer.config = ScoringConfig(defaults_path(), validate=False)
        proc = _hoist_processor(scorer)
        path = _flat_path(tmp_path)
        images = _load(proc, [path])
        assert images[path]['composition']['power_point_score'] is None
        results = {path: {'aesthetic': 5.0}}
        proc._compute_aggregates(results, images)
        assert results[path]['power_point_score'] is None
        # the aggregate reader still gets the metric (None) and scores it neutral
        assert scorer.calls[0]['power_point_score'] is None

    def test_the_full_save_stores_null(self, seeded):
        path, facet, db = seeded
        with get_connection(db, row_factory=True) as conn:
            assert conn.execute("SELECT power_point_score FROM photos WHERE path=?",
                                (path,)).fetchone()[0] is not None   # the seed row has a number
        proc = _processor(facet, [['saliency']], restricted=False)
        proc._save_results({path: {'power_point_score': None, 'aesthetic': 7.0}}, _images(path))
        with get_connection(db, row_factory=True) as conn:
            assert conn.execute("SELECT power_point_score FROM photos WHERE path=?",
                                (path,)).fetchone()[0] is None

    def test_the_full_save_stores_a_measured_value_as_a_float(self, seeded):
        path, facet, db = seeded
        proc = _processor(facet, [['saliency']], restricted=False)
        proc._save_results({path: {'power_point_score': 6, 'aesthetic': 7.0}}, _images(path))
        with get_connection(db, row_factory=True) as conn:
            assert conn.execute("SELECT power_point_score FROM photos WHERE path=?",
                                (path,)).fetchone()[0] == 6.0


class TestReaderIsNeutral:
    """The aggregate is unchanged by the NULL: ``build_metric_vector`` scores it 5.0, as before."""

    def test_a_null_power_point_scores_like_the_old_stored_five(self):
        from config.scoring_config import ScoringConfig
        from config_resolve import defaults_path
        from processing.scorer import build_metric_vector
        cfg = ScoringConfig(defaults_path(), validate=False)
        base = {'aesthetic': 7.0, 'comp_score': 6.0, 'tech_sharpness': 6.0, 'leading_lines_score': 3.0}
        null_vec = build_metric_vector({**base, 'power_point_score': None}, cfg, 'default')
        five_vec = build_metric_vector({**base, 'power_point_score': 5.0}, cfg, 'default')
        assert null_vec == five_vec
