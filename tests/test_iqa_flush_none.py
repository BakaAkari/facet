"""Issue #184 follow-up: the two recompute writers skip a ``None`` PyIQA score.

``PyIQAScorer.score_batch`` returns ``None`` for an image it could not score.
``--score-topiq`` (``facet._flush_topiq_batch``) and ``--recompute-iqa``
(``Facet._run_iqa_pass._flush``) must leave that photo's column alone (no UPDATE,
not counted) so a rerun retries it. ``--recompute-iqa`` must also never treat a
faceless photo's NULL ``face_quality_iqa`` as work, nor decode its thumbnail.

Torch-free: the scorers are fakes and the schema comes from ``init_database``.
"""

import logging
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from db import get_connection
from db.schema import init_database


def _jpeg():
    buf = BytesIO()
    Image.new('RGB', (32, 32), (120, 130, 140)).save(buf, 'JPEG')
    return buf.getvalue()


_THUMB = _jpeg()
_NOT_AN_IMAGE = b'this is not a jpeg'


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "iqa.db")
    init_database(path)
    return path


def _insert(db_path, rows):
    with get_connection(db_path, row_factory=False) as conn:
        for row in rows:
            cols = list(row)
            conn.execute(
                f"INSERT INTO photos ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [row[c] for c in cols])
        conn.commit()


def _column(db_path, column):
    with get_connection(db_path, row_factory=False) as conn:
        return dict(conn.execute(f"SELECT path, {column} FROM photos").fetchall())


class _FakeScorer:
    """Returns the queued scores batch by batch; records every batch size."""

    # PyIQAScorer.last_declined: positions of the last batch the model refused on
    # purpose. None of these tests declines, so it stays empty.
    last_declined = frozenset()

    def __init__(self, *batches):
        self.batches = list(batches)
        self.sizes = []
        self.unloaded = False

    def score_batch(self, images):
        self.sizes.append(len(images))
        scores = self.batches.pop(0)
        assert len(scores) == len(images)
        return scores

    def unload(self):
        self.unloaded = True


class _MustNotScore:
    last_declined = frozenset()

    def score_batch(self, images):
        raise AssertionError("scored a photo that should have been skipped")

    def unload(self):
        pass


# ---------------------------------------------------------------------------
# --score-topiq
# ---------------------------------------------------------------------------

class TestScoreTopiqFlush:
    def test_none_leaves_the_column_and_is_not_counted(self, db):
        import facet as facet_cli
        _insert(db, [
            {'path': '/a.jpg', 'filename': 'a.jpg', 'thumbnail': b'x'},
            {'path': '/b.jpg', 'filename': 'b.jpg', 'thumbnail': b'x', 'topiq_score': 5.5},
            {'path': '/c.jpg', 'filename': 'c.jpg', 'thumbnail': b'x'},
        ])
        scorer = _FakeScorer([6.126, None, 7.0])
        with get_connection(db, row_factory=False) as conn:
            written = facet_cli._flush_topiq_batch(
                conn, scorer, ['/a.jpg', '/b.jpg', '/c.jpg'], [object()] * 3)
            conn.commit()

        assert written == 2
        assert _column(db, 'topiq_score') == {'/a.jpg': 6.13, '/b.jpg': 5.5, '/c.jpg': 7.0}

    def test_an_all_none_batch_writes_nothing(self, db):
        import facet as facet_cli
        _insert(db, [{'path': '/a.jpg', 'filename': 'a.jpg', 'thumbnail': b'x'}])
        with get_connection(db, row_factory=False) as conn:
            written = facet_cli._flush_topiq_batch(
                conn, _FakeScorer([None]), ['/a.jpg'], [object()])
            conn.commit()
        assert written == 0
        assert _column(db, 'topiq_score') == {'/a.jpg': None}

    def test_pin_scored_batch_is_rounded_and_counted(self, db):
        """PIN of the clean-batch behaviour the hoist must carry over (it errors before
        only because the function was nested inside the ``--score-topiq`` block)."""
        import facet as facet_cli
        _insert(db, [{'path': '/a.jpg', 'filename': 'a.jpg', 'thumbnail': b'x'},
                     {'path': '/b.jpg', 'filename': 'b.jpg', 'thumbnail': b'x'}])
        with get_connection(db, row_factory=False) as conn:
            written = facet_cli._flush_topiq_batch(
                conn, _FakeScorer([1.234, 9.876]), ['/a.jpg', '/b.jpg'], [object()] * 2)
            conn.commit()
        assert written == 2
        assert _column(db, 'topiq_score') == {'/a.jpg': 1.23, '/b.jpg': 9.88}


# ---------------------------------------------------------------------------
# --recompute-iqa
# ---------------------------------------------------------------------------

def _facet(db_path, scorer=None):
    from processing.scorer import Facet
    facet = Facet.__new__(Facet)
    facet.db_path = db_path
    facet.config = SimpleNamespace(get_extended_iqa_settings=lambda: {})
    if scorer is not None:
        facet._make_iqa_scorer = lambda name: scorer
    return facet


def _counters():
    return {'updated': {'face_quality_iqa': 0}, 'errors': 0, 'declined': 0, 'liqe_too_small': 0}


def _face_rows():
    return [
        {'path': f'/{n}.jpg', 'filename': f'{n}.jpg', 'thumbnail': _THUMB, 'face_count': 1}
        for n in ('a', 'b', 'c')
    ]


class TestRecomputeIqaFlush:
    def test_none_leaves_that_photo_null_counts_only_written_and_is_retried(self, db):
        _insert(db, _face_rows())
        facet = _facet(db, _FakeScorer([6.126, None, 8.0], [4.0]))
        counters = _counters()

        facet._run_iqa_pass(['topiq_nr_face'], 64, counters)

        assert _column(db, 'face_quality_iqa') == {'/a.jpg': 6.13, '/b.jpg': None, '/c.jpg': 8.0}
        assert counters['updated']['face_quality_iqa'] == 2
        assert counters['errors'] == 0

        # A rerun selects exactly the photo that stayed NULL (WHERE col IS NULL).
        counters = _counters()
        facet._run_iqa_pass(['topiq_nr_face'], 64, counters)
        assert facet._make_iqa_scorer('x').sizes == [3, 1]
        assert _column(db, 'face_quality_iqa')['/b.jpg'] == 4.0
        assert counters['updated']['face_quality_iqa'] == 1

    def test_a_faceless_photo_is_never_selected_nor_decoded(self, db):
        _insert(db, [
            {'path': '/faceless.jpg', 'filename': 'f.jpg', 'thumbnail': _NOT_AN_IMAGE,
             'face_count': 0},
            {'path': '/unknown.jpg', 'filename': 'u.jpg', 'thumbnail': _NOT_AN_IMAGE},
        ])
        facet = _facet(db, _MustNotScore())
        counters = _counters()

        facet._run_iqa_pass(['topiq_nr_face'], 64, counters)

        assert counters['errors'] == 0  # the BLOB was never decoded
        assert counters['updated']['face_quality_iqa'] == 0
        assert _column(db, 'face_quality_iqa') == {'/faceless.jpg': None, '/unknown.jpg': None}


class TestRecomputeIqaNothingToDo:
    @staticmethod
    def _recompute(db_path, monkeypatch):
        from models.model_manager import ModelManager
        from processing.scorer import Facet

        executed = []
        monkeypatch.setattr(ModelManager, "detect_vram", staticmethod(lambda: 24.0))
        monkeypatch.setattr(ModelManager, "detect_accelerator", staticmethod(lambda: "cuda"))
        monkeypatch.setattr(ModelManager, "detect_system_ram_gb", staticmethod(lambda: 32.0))
        monkeypatch.setattr(Facet, "_iqa_model_vram_gb", lambda self, name: 1.0)
        monkeypatch.setattr(
            Facet, "_run_iqa_pass",
            lambda self, names, batch_size, counters: executed.append(list(names)))
        _facet(db_path).recompute_iqa_from_thumbnails()
        return executed

    def test_a_faceless_null_face_quality_is_not_work(self, db, monkeypatch, caplog):
        _insert(db, [{
            'path': '/faceless.jpg', 'filename': 'f.jpg', 'thumbnail': _THUMB, 'face_count': 0,
            'aesthetic_iaa': 6.0, 'liqe_score': 4.0,  # face_quality_iqa stays NULL
        }])
        with caplog.at_level(logging.INFO):
            executed = self._recompute(db, monkeypatch)

        assert executed == []
        assert "nothing to do" in caplog.text
        assert "face_quality_iqa: 0 NULL" in caplog.text

    def test_only_photos_that_can_have_the_score_are_counted(self, db, monkeypatch, caplog):
        done = {'aesthetic_iaa': 6.0, 'liqe_score': 4.0}
        _insert(db, [
            {'path': '/faceless.jpg', 'filename': 'f.jpg', 'thumbnail': _THUMB,
             'face_count': 0, **done},
            {'path': '/face.jpg', 'filename': 'g.jpg', 'thumbnail': _THUMB,
             'face_count': 2, **done},
        ])
        with caplog.at_level(logging.INFO):
            executed = self._recompute(db, monkeypatch)

        assert "face_quality_iqa: 1 NULL" in caplog.text
        assert "1 photos to score" in caplog.text
        assert len(executed) >= 1
