"""Issue #184: a failed saliency run stores NULL (or keeps the stored value), never a neutral score.

Covers the three writers (``_pass_saliency`` + restricted save, ``--recompute-saliency``'s
flush) and the aggregate reader that scores a NULL as the neutral 5.0.
"""

import pytest
from PIL import Image

pytest.importorskip("imagehash")
pytest.importorskip("cv2")

from db import get_connection  # noqa: E402
from tests import test_single_pass_preserves as _sp  # noqa: E402
from processing.multi_pass import (  # noqa: E402
    SALIENCY_FAILED_KEY, SALIENCY_PASS_COLUMNS,
)

_COLS = ('subject_sharpness', 'subject_prominence', 'subject_placement', 'bg_separation',
         'subject_bbox')
_FAILED = {k: None for k in _COLS}
_SALIENCY, _images, _persist, _processor = _sp._SALIENCY, _sp._images, _sp._persist, _sp._processor
seeded = _sp.seeded  # the fixture, shared rather than copied


class _Scorer:
    def __init__(self, scores):
        self.scores = scores

    def score_batch(self, pil, cv):
        return self.scores[:len(pil)]


def _saliency_row(db, path):
    with get_connection(db, row_factory=True) as conn:
        return dict(conn.execute(
            f"SELECT {', '.join(_COLS)} FROM photos WHERE path=?", (path,)).fetchone())


def _run_pass(proc, path, scores):
    results = {path: {}}
    proc._pass_saliency(_Scorer(scores), {path: {'pil': Image.new('RGB', (8, 8))}}, results)
    return results


def test_failed_photo_is_flagged_and_its_five_keys_stay_unset(seeded):
    path, facet, _ = seeded
    results = _run_pass(_processor(facet, [['saliency']]), path, [dict(_FAILED)])
    assert results[path] == {SALIENCY_FAILED_KEY: True}
    assert not (SALIENCY_PASS_COLUMNS & results[path].keys())


def test_successful_photo_keeps_its_values_including_a_legit_no_subject_result(seeded):
    path, facet, _ = seeded
    neutral = {'subject_sharpness': 5.0, 'subject_prominence': 0.0, 'subject_placement': 5.0,
               'bg_separation': 5.0, 'subject_bbox': None}
    results = _run_pass(_processor(facet, [['saliency']]), path, [neutral])
    assert SALIENCY_FAILED_KEY not in results[path]
    assert results[path]['subject_sharpness'] == 5.0
    assert results[path]['subject_bbox'] is None


def test_restricted_save_keeps_stored_values_over_a_failure(seeded):
    path, facet, db = seeded
    proc = _processor(facet, [['saliency']])
    _persist(proc, {path: dict(_SALIENCY)}, _images(path))
    before = _saliency_row(db, path)
    assert before['subject_sharpness'] == 7.0

    results = _run_pass(proc, path, [dict(_FAILED)])
    _persist(proc, results, _images(path))

    assert _saliency_row(db, path) == before


def test_full_save_stores_null_over_a_failure_never_a_neutral_score(seeded):
    """A full rescan (restricted=False, e.g. --force) is the one writer that overwrites.

    The photo carries stored saliency; the rescan's saliency fails, so the full
    UPSERT writes the honest NULL in all five columns -- never the old 5.0.
    """
    path, facet, db = seeded
    _persist(_processor(facet, [['saliency']]), {path: dict(_SALIENCY)}, _images(path))
    assert _saliency_row(db, path)['subject_sharpness'] == 7.0

    full = _processor(facet, [['saliency']], restricted=False)
    results = _run_pass(full, path, [dict(_FAILED)])
    assert results[path] == {SALIENCY_FAILED_KEY: True}
    full._save_results(results, _images(path))

    assert _saliency_row(db, path) == _FAILED


def test_a_photo_that_never_had_saliency_stays_null_after_a_failure(seeded):
    path, facet, db = seeded
    proc = _processor(facet, [['saliency']])
    results = _run_pass(proc, path, [dict(_FAILED)])
    _persist(proc, results, _images(path))
    assert _saliency_row(db, path) == _FAILED


def test_null_saliency_scores_neutral_in_the_aggregate():
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path
    from processing.scorer import Facet, build_metric_vector

    cfg = ScoringConfig(defaults_path(), validate=False)
    base = dict(face_count=0, face_ratio=0.0, aesthetic=7, comp_score=6, tech_sharpness=7,
                color_score=6, exposure_score=7, tags='', is_monochrome=0, mean_saturation=0.4)
    nulls = dict(base, **{k: None for k in _COLS})
    neutral = dict(base, subject_sharpness=5.0, subject_prominence=5.0,
                   subject_placement=5.0, bg_separation=5.0)

    vector = build_metric_vector(nulls, cfg, 'default')
    for key in ('subject_sharpness', 'subject_prominence', 'subject_placement', 'bg_separation'):
        assert vector[key] == 5.0

    facet = Facet.__new__(Facet)
    facet.config = cfg
    original = cfg.get_weights

    def weighted(category):
        w = dict(original(category))
        w.update(subject_sharpness=0.2, subject_prominence=0.1,
                 subject_placement=0.1, bg_separation=0.1)
        return w

    cfg.get_weights = weighted
    score_null, _ = facet.calculate_aggregate_logic(nulls)
    score_neutral, _ = facet.calculate_aggregate_logic(neutral)
    assert score_null == score_neutral
    assert 0.0 < score_null < float('inf')


@pytest.fixture()
def recompute_db(seeded):
    """The seeded row (populated) plus a second row that never had saliency."""
    path, facet, db = seeded
    with get_connection(db, row_factory=False) as conn:
        conn.execute("UPDATE photos SET subject_sharpness=7.0, subject_prominence=5.0, "
                     "subject_placement=6.0, bg_separation=4.0, "
                     "subject_bbox='[0.1, 0.1, 0.9, 0.9]' WHERE path=?", (path,))
        conn.execute("INSERT INTO photos (path, filename, thumbnail) VALUES ('/x/never.jpg', "
                     "'never.jpg', x'00')")
        conn.commit()
    return path, db


def test_recompute_flush_skips_a_failed_photo_and_does_not_count_it(recompute_db):
    import facet as facet_cli
    path, db = recompute_db
    good = {'subject_sharpness': 3.0, 'subject_prominence': 2.0, 'subject_placement': 1.0,
            'bg_separation': 9.0, 'subject_bbox': [0.2, 0.2, 0.4, 0.4]}
    scorer = _Scorer([dict(_FAILED), good])
    with get_connection(db, row_factory=False) as conn:
        written = facet_cli._flush_saliency_batch(
            conn, scorer, [path, '/x/never.jpg'], [object(), object()], [object(), object()])
        conn.commit()
    assert written == 1
    # --force over a populated row: the failure leaves the stored value in place
    assert _saliency_row(db, path)['subject_sharpness'] == 7.0
    assert _saliency_row(db, path)['subject_bbox'] == '[0.1, 0.1, 0.9, 0.9]'
    assert _saliency_row(db, '/x/never.jpg')['subject_sharpness'] == 3.0


def test_recompute_flush_leaves_a_never_scored_row_null_and_reselectable(recompute_db):
    import facet as facet_cli
    _, db = recompute_db
    scorer = _Scorer([dict(_FAILED)])
    with get_connection(db, row_factory=False) as conn:
        written = facet_cli._flush_saliency_batch(
            conn, scorer, ['/x/never.jpg'], [object()], [object()])
        conn.commit()
    assert written == 0
    assert _saliency_row(db, '/x/never.jpg') == _FAILED
    with get_connection(db, row_factory=False) as conn:
        selected = [r[0] for r in conn.execute(
            "SELECT path FROM photos WHERE thumbnail IS NOT NULL AND subject_bbox IS NULL")]
    assert '/x/never.jpg' in selected


def test_restricted_save_skips_saliency_columns_of_a_flagged_photo_even_if_present(seeded):
    """The flag itself guards the save, not only the absence of the keys."""
    path, facet, db = seeded
    proc = _processor(facet, [['saliency']])
    _persist(proc, {path: dict(_SALIENCY)}, _images(path))
    before = _saliency_row(db, path)

    _persist(proc, {path: {SALIENCY_FAILED_KEY: True, **_FAILED, 'subject_sharpness': 5.0}},
             _images(path))

    assert _saliency_row(db, path) == before
