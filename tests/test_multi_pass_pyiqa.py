"""Tests for the PyIQA pass -> DB column mapping in the multi-pass scorer.

Locks in:

1. When TOPIQ is the *primary* aesthetic model (16gb/24gb profiles), the pass
   fills the dedicated ``topiq_score`` column in addition to
   ``aesthetic``/``quality_score``. The viewer's TOPIQ range filter
   (``min_topiq``/``max_topiq``) reads ``topiq_score``; without this it is
   always empty after a normal scan (only ``--score-topiq`` would fill it).
2. Supplementary PyIQA models map to their own dedicated columns and never
   touch ``topiq_score``.
3. (#184 follow-up) A ``None`` score means "this image failed": it is never
   written as a value. A failed supplementary model leaves its key unset; a
   failed PRIMARY model flags the photo (``SCAN_FAILED_KEY``) so it is recorded
   in ``scan_failures`` and retried, instead of being saved with a neutral 5.0.
   An exception that escapes ``score_batch`` is swallowed for a supplementary
   model and re-raised for a primary one (``_process_chunk`` marks the chunk
   retryable).

Needs no torch/pyiqa: the scorers are fakes. The DB-level tests reuse the
single-pass seeder and skip (alone) when imagehash/cv2 are absent.
"""

import logging

import pytest

from processing.multi_pass import SCAN_FAILED_KEY, ChunkedMultiPassProcessor

try:
    # The DB-level tests reuse the single-pass seeder (imported, not copied).
    # That module skips itself when imagehash/cv2 are absent (CI's minimal job);
    # only the DB tests below then skip, never the torch-free writer tests.
    from tests import test_single_pass_preserves as _sp
    seeded = _sp.seeded
except pytest.skip.Exception:  # pragma: no cover - only without imagehash/cv2
    _sp = None

    @pytest.fixture()
    def seeded():
        pytest.skip("imagehash/cv2 not installed")


class _StubScorer:
    def __init__(self, scores):
        self._scores = scores

    def score_batch(self, pil_imgs):
        return self._scores[:len(pil_imgs)]


class _RaisingScorer:
    def score_batch(self, pil_imgs):
        raise RuntimeError("chunk-wide boom")


def _run_pass(model_name, scores):
    # _pass_pyiqa only needs the class-level PYIQA_COLUMN_MAP, so bypass __init__.
    proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
    images = {f"p{i}.jpg": {"pil": object()} for i in range(len(scores))}
    results = {path: {} for path in images}
    proc._pass_pyiqa(_StubScorer(scores), model_name, images, results)
    return results


def _run_with_errors(model_name, scores, with_callback=True):
    """Run one pass; returns (results, on_error calls)."""
    proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
    calls = []
    if with_callback:
        proc.on_error = lambda path, stage, msg: calls.append((path, stage, msg))
    images = {f"p{i}.jpg": {"pil": object()} for i in range(len(scores))}
    results = {path: {} for path in images}
    proc._pass_pyiqa(_StubScorer(scores), model_name, images, results)
    return results, calls


class TestPyiqaColumnMapping:
    def test_topiq_primary_fills_topiq_score(self):
        r = _run_pass("topiq", [6.25, 4.5])["p0.jpg"]
        assert r["aesthetic"] == 6.25
        assert r["quality_score"] == 6.25
        assert r["scoring_model"] == "topiq"
        assert r["topiq_score"] == 6.25  # the fix: dedicated column is filled

    def test_supplementary_models_map_to_dedicated_columns(self):
        assert _run_pass("topiq_iaa", [7.0])["p0.jpg"] == {"aesthetic_iaa": 7.0}
        assert _run_pass("topiq_nr_face", [8.0])["p0.jpg"] == {"face_quality_iqa": 8.0}
        assert _run_pass("liqe", [3.0])["p0.jpg"] == {"liqe_score": 3.0}

    def test_supplementary_does_not_set_topiq_score(self):
        assert "topiq_score" not in _run_pass("liqe", [3.0])["p0.jpg"]

    def test_non_topiq_primary_does_not_set_topiq_score(self):
        r = _run_pass("clip_aesthetic", [5.5])["p0.jpg"]
        assert r["aesthetic"] == 5.5
        assert "topiq_score" not in r


_SUPPLEMENTARY = {
    "topiq_iaa": "aesthetic_iaa",
    "topiq_nr_face": "face_quality_iqa",
    "liqe": "liqe_score",
    "aesthetic_v25": "aesthetic_v25",
    "deqa": "deqa_score",
}
_PRIMARY_KEYS = ("aesthetic", "quality_score", "topiq_score", "scoring_model")


class TestNoneScoreIsNeverWritten:
    @pytest.mark.parametrize("model_name,column", sorted(_SUPPLEMENTARY.items()))
    def test_supplementary_none_leaves_the_key_unset(self, model_name, column):
        results = _run_pass(model_name, [None, 8.0])
        assert results["p0.jpg"] == {}
        assert results["p1.jpg"] == {column: 8.0}

    def test_supplementary_none_is_not_a_scan_failure(self):
        results, calls = _run_with_errors("liqe", [None, 4.0])
        assert SCAN_FAILED_KEY not in results["p0.jpg"]
        assert calls == []


class TestFailedPrimaryMarksThePhotoFailed:
    def test_topiq_none_sets_scan_failed_and_no_score_keys(self):
        results, _ = _run_with_errors("topiq", [None, 6.0])
        p0 = results["p0.jpg"]
        assert p0[SCAN_FAILED_KEY] == "topiq"
        for key in _PRIMARY_KEYS:
            assert key not in p0
        # the sibling photo is complete and unflagged
        p1 = results["p1.jpg"]
        assert SCAN_FAILED_KEY not in p1
        assert p1["topiq_score"] == 6.0
        assert p1["aesthetic"] == 6.0
        assert p1["scoring_model"] == "topiq"

    def test_on_error_is_called_exactly_once_for_the_failed_photo(self):
        _, calls = _run_with_errors("topiq", [None, 6.0])
        assert len(calls) == 1
        path, stage, message = calls[0]
        assert (path, stage) == ("p0.jpg", "topiq")
        assert "required model pass failed: topiq" in message

    def test_non_topiq_primary_none_is_also_flagged(self):
        results, calls = _run_with_errors("clip_aesthetic", [None])
        assert results["p0.jpg"] == {SCAN_FAILED_KEY: "clip_aesthetic"}
        assert len(calls) == 1

    def test_a_processor_without_on_error_does_not_crash(self):
        """``--pass quality`` builds the processor with no ``on_error`` (spec A5): the
        photo is still flagged (so the restricted save skips it and keeps the stored
        value) but there is no scan_failures row to write."""
        results, calls = _run_with_errors("topiq", [None], with_callback=False)
        assert results["p0.jpg"] == {SCAN_FAILED_KEY: "topiq"}
        assert calls == []


class TestScoreBatchExceptions:
    @staticmethod
    def _run(model_name):
        proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
        images = {"p0.jpg": {"pil": object()}}
        results = {"p0.jpg": {}}
        proc._pass_pyiqa(_RaisingScorer(), model_name, images, results)
        return results

    def test_supplementary_exception_is_swallowed_and_logged(self, caplog):
        with caplog.at_level(logging.ERROR):
            results = self._run("liqe")
        assert results == {"p0.jpg": {}}
        assert "PyIQA liqe pass failed" in caplog.text

    def test_primary_exception_propagates_to_the_retryable_branch(self):
        with pytest.raises(RuntimeError, match="chunk-wide boom"):
            self._run("topiq")


_STORED = ("aesthetic", "quality_score", "topiq_score", "scoring_model",
           "aesthetic_iaa", "face_quality_iqa", "liqe_score")


def _stored_row(db, path):
    from db import get_connection
    with get_connection(db, row_factory=True) as conn:
        return dict(conn.execute(
            f"SELECT {', '.join(_STORED)} FROM photos WHERE path=?", (path,)).fetchone())


def _db_pass(proc, model_name, path, scores):
    results = {path: {}}
    proc._pass_pyiqa(_StubScorer(scores), model_name, _sp._images(path), results)
    return results


class TestSavePath:
    def test_restricted_save_keeps_stored_supplementary_values_over_a_none(self, seeded):
        path, facet, db = seeded
        proc = _sp._processor(facet, [["topiq_nr_face", "liqe", "topiq_iaa"]])
        before = _stored_row(db, path)
        assert (before["aesthetic_iaa"], before["face_quality_iqa"],
                before["liqe_score"]) == (6.0, 7.0, 3.7)

        merged = {path: {}}
        for model in ("topiq_iaa", "topiq_nr_face", "liqe"):
            merged[path].update(_db_pass(proc, model, path, [None])[path])
        assert merged[path] == {}
        _sp._persist(proc, merged, _sp._images(path))

        assert _stored_row(db, path) == before

    def test_restricted_save_leaves_a_primary_failed_photo_untouched(self, seeded):
        path, facet, db = seeded
        proc = _sp._processor(facet, [["topiq"]])
        before = _stored_row(db, path)
        assert (before["aesthetic"], before["quality_score"],
                before["topiq_score"]) == (8.2, 8.0, 8.2)

        results = _db_pass(proc, "topiq", path, [None])
        assert results[path] == {SCAN_FAILED_KEY: "topiq"}
        _sp._persist(proc, results, _sp._images(path))

        assert _stored_row(db, path) == before

    def test_full_save_keeps_the_old_row_of_a_primary_failed_photo(self, seeded):
        path, facet, db = seeded
        full = _sp._processor(facet, [["topiq"]], restricted=False)
        before = _stored_row(db, path)

        results = _db_pass(full, "topiq", path, [None])
        full._save_results(results, _sp._images(path))

        assert _stored_row(db, path) == before

    def test_pin_full_save_stores_null_for_absent_supplementary_keys(self, seeded):
        """PIN (passes before and after): a full rescan with the keys absent writes
        NULL for the three supplementary columns, never a neutral 5.0."""
        path, facet, db = seeded
        full = _sp._processor(facet, [["liqe"]], restricted=False)
        full._save_results({path: {"aesthetic": 7.0}}, _sp._images(path))
        row = _stored_row(db, path)
        assert row["aesthetic_iaa"] is None
        assert row["face_quality_iqa"] is None
        assert row["liqe_score"] is None


def test_pin_null_supplementary_scores_equal_neutral_in_the_aggregate_vector():
    """PIN (passes before and after): a NULL scores as the neutral 5.0 in the metric
    vector, so the new NULLs do not move any aggregate."""
    pytest.importorskip("imagehash")
    pytest.importorskip("cv2")
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path
    from processing.scorer import build_metric_vector

    cfg = ScoringConfig(defaults_path(), validate=False)
    base = dict(face_count=0, face_ratio=0.0, aesthetic=7, comp_score=6, tech_sharpness=7,
                color_score=6, exposure_score=7, tags='', is_monochrome=0, mean_saturation=0.4)
    cols = ("aesthetic_iaa", "face_quality_iqa", "liqe_score", "aesthetic_v25")
    nulls = dict(base, **{c: None for c in cols})
    neutral = dict(base, **{c: 5.0 for c in cols})
    assert (build_metric_vector(nulls, cfg, 'default')
            == build_metric_vector(neutral, cfg, 'default'))
