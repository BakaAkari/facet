"""``--recompute-iqa`` stops re-attempting photos a model cannot score.

Two deterministic outcomes used to be retried on every run:

* LIQE asserts a short side of at least 224 px. The outcome is a pure function of
  the stored thumbnail's geometry, so the pass skips it from the header alone
  (before any decode) and records nothing.
* TOPIQ NR-Face answers "No face detected" for a thumbnail its own detector finds
  no face in, even though InsightFace found one on the full image. That is
  remembered in the ``iqa_declined`` side table, keyed by a fingerprint
  (``scanned_at || '|' || length(thumbnail)``) that a rescan invalidates.

An out-of-memory error or any unexpected failure is not a decline and is still
retried.

Torch-free: the scorers are fakes and every schema comes from ``init_database``.
Names the production code adds are imported inside the tests so a missing one
fails that test rather than the whole module.
"""

import logging
import random
import sqlite3
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from db import get_connection
from db.schema import init_database

_SCANNED = '2026-01-01 00:00:00'


def _jpeg(size=(32, 32), colour=(120, 130, 140)):
    buf = BytesIO()
    Image.new('RGB', size, colour).save(buf, 'JPEG')
    return buf.getvalue()


def _noisy_jpeg(size=(100, 50)):
    """A JPEG with an incompressible body, so truncating it leaves a valid header."""
    rng = random.Random(0)
    raw = bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3))
    buf = BytesIO()
    Image.frombytes('RGB', size, raw).save(buf, 'JPEG', quality=95)
    return buf.getvalue()


_THUMB = _jpeg()
_NOT_AN_IMAGE = b'this is not a jpeg'


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "iqa.db")
    init_database(path)
    return path


def _photo(path='/a.jpg', **extra):
    row = {'path': path, 'filename': path.rsplit('/', 1)[-1], 'thumbnail': _THUMB,
           'face_count': 1, 'scanned_at': _SCANNED}
    row.update(extra)
    return row


def _insert(db_path, rows):
    with get_connection(db_path, row_factory=False) as conn:
        for row in rows:
            cols = list(row)
            conn.execute(
                f"INSERT INTO photos ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [row[c] for c in cols])
        conn.commit()


def _exec(db_path, sql, params=()):
    with get_connection(db_path, row_factory=False) as conn:
        conn.execute(sql, params)
        conn.commit()


def _column(db_path, column):
    with get_connection(db_path, row_factory=False) as conn:
        return dict(conn.execute(f"SELECT path, {column} FROM photos").fetchall())


def _declines(db_path):
    with get_connection(db_path, row_factory=False) as conn:
        return conn.execute(
            "SELECT photo_path, model, scan_fingerprint FROM iqa_declined "
            "ORDER BY photo_path, model").fetchall()


def _fingerprint(db_path, path):
    """The fingerprint spelled out here, independent of the production constant."""
    with get_connection(db_path, row_factory=False) as conn:
        return conn.execute(
            "SELECT scanned_at || '|' || length(thumbnail) FROM photos WHERE path = ?",
            (path,)).fetchone()[0]


def _decline(db_path, path, model='topiq_nr_face'):
    """Record a decline directly, as the writer would, at the photo's CURRENT fingerprint."""
    _exec(db_path,
          "INSERT INTO iqa_declined (photo_path, model, scan_fingerprint) "
          "SELECT path, ?, scanned_at || '|' || length(thumbnail) FROM photos WHERE path = ?",
          (model, path))


class _FakeScorer:
    """Returns the queued scores batch by batch and exposes ``last_declined`` like PyIQAScorer.

    ``declined`` is one set of positions per call; a call past the end declines
    nothing. ``seen`` records the (width, height) of every image of every call.
    """

    def __init__(self, *batches, declined=()):
        self.batches = list(batches)
        self.declined = [frozenset(d) for d in declined]
        self.seen = []
        self.last_declined = frozenset()
        self.unloaded = False

    def score_batch(self, images):
        self.seen.append([img.size for img in images])
        scores = self.batches.pop(0)
        assert len(scores) == len(images)
        self.last_declined = self.declined.pop(0) if self.declined else frozenset()
        return scores

    def unload(self):
        self.unloaded = True


class _ByWidth:
    """Scores an image as width / 10, so a result can be matched to its photo."""

    last_declined = frozenset()

    def __init__(self):
        self.seen = []

    def score_batch(self, images):
        self.seen.append([img.size for img in images])
        return [img.size[0] / 10 for img in images]

    def unload(self):
        pass


class _MustNotScore:
    last_declined = frozenset()

    def score_batch(self, images):
        raise AssertionError("scored a photo that should have been skipped")

    def unload(self):
        pass


def _facet(db_path, scorers=None):
    from processing.scorer import Facet
    facet = Facet.__new__(Facet)
    facet.db_path = db_path
    facet.config = SimpleNamespace(get_extended_iqa_settings=lambda: {})
    if scorers is not None:
        facet._make_iqa_scorer = lambda name: scorers[name]
    return facet


def _counters(*columns):
    return {'updated': {c: 0 for c in columns}, 'errors': 0, 'declined': 0, 'liqe_too_small': 0}


def _pass(facet, models, counters=None):
    from processing.scorer import Facet
    col_map = dict(Facet._BASE_IQA_MODELS)
    counters = counters or _counters(*(col_map[m] for m in models))
    facet._run_iqa_pass(models, 64, counters)
    return counters


def _recompute(db_path, monkeypatch, fake_pass=None):
    """Run ``recompute_iqa_from_thumbnails`` with the model passes replaced; return the passes."""
    from models.model_manager import ModelManager
    from processing.scorer import Facet

    executed = []

    def _record(self, names, batch_size, counters):
        executed.append(list(names))
        if fake_pass is not None:
            fake_pass(counters)

    monkeypatch.setattr(ModelManager, "detect_vram", staticmethod(lambda: 24.0))
    monkeypatch.setattr(ModelManager, "detect_accelerator", staticmethod(lambda: "cuda"))
    monkeypatch.setattr(ModelManager, "detect_system_ram_gb", staticmethod(lambda: 32.0))
    monkeypatch.setattr(Facet, "_iqa_model_vram_gb", lambda self, name: 1.0)
    monkeypatch.setattr(Facet, "_run_iqa_pass", _record)
    _facet(db_path).recompute_iqa_from_thumbnails()
    return executed


# ---------------------------------------------------------------------------
# the side table
# ---------------------------------------------------------------------------

class TestIqaDeclinedTable:
    def test_exists_after_init_database_with_the_declared_columns(self, db):
        with get_connection(db, row_factory=False) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(iqa_declined)")]
        assert cols == ['photo_path', 'model', 'scan_fingerprint']

    def test_primary_key_rejects_a_duplicate(self, db):
        _insert(db, [_photo()])
        _decline(db, '/a.jpg')
        with pytest.raises(sqlite3.IntegrityError):
            _decline(db, '/a.jpg')

    def test_a_decline_for_an_unknown_photo_is_rejected(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            _exec(db, "INSERT INTO iqa_declined (photo_path, model, scan_fingerprint) "
                      "VALUES ('/ghost.jpg', 'topiq_nr_face', 'x')")

    def test_deleting_the_photo_deletes_its_declines(self, db):
        _insert(db, [_photo('/a.jpg'), _photo('/b.jpg')])
        _decline(db, '/a.jpg')
        _decline(db, '/b.jpg')
        _exec(db, "DELETE FROM photos WHERE path = '/a.jpg'")
        assert [r[0] for r in _declines(db)] == ['/b.jpg']

    def test_is_registered_in_the_migrated_tables(self):
        from db import schema
        assert ('iqa_declined', schema.IQA_DECLINED_COLUMNS) in schema._MIGRATED_TABLES

    def test_init_database_recreates_a_dropped_table(self, db):
        """The upgrade path: a library that predates the table gets it on the next init."""
        _exec(db, "DROP TABLE iqa_declined")
        init_database(db)
        assert _declines(db) == []

    def test_a_second_init_database_keeps_the_rows(self, db):
        _insert(db, [_photo()])
        _decline(db, '/a.jpg')
        init_database(db)
        assert len(_declines(db)) == 1


# ---------------------------------------------------------------------------
# NR-Face decline
# ---------------------------------------------------------------------------

class TestNrFaceDecline:
    def test_a_decline_is_recorded_at_the_current_fingerprint(self, db):
        _insert(db, [_photo()])
        facet = _facet(db, {'topiq_nr_face': _FakeScorer([None], declined=[{0}])})

        counters = _pass(facet, ['topiq_nr_face'])

        assert _declines(db) == [('/a.jpg', 'topiq_nr_face', f"{_SCANNED}|{len(_THUMB)}")]
        assert _declines(db)[0][2] == _fingerprint(db, '/a.jpg')
        assert counters['declined'] == 1
        assert counters['updated']['face_quality_iqa'] == 0
        assert _column(db, 'face_quality_iqa') == {'/a.jpg': None}

    def test_a_declined_photo_is_not_scored_again(self, db):
        _insert(db, [_photo()])
        _pass(_facet(db, {'topiq_nr_face': _FakeScorer([None], declined=[{0}])}),
              ['topiq_nr_face'])

        counters = _pass(_facet(db, {'topiq_nr_face': _MustNotScore()}), ['topiq_nr_face'])

        assert counters['declined'] == 0
        assert counters['errors'] == 0

    def test_a_declined_photo_is_not_work_for_recompute_iqa(self, db, monkeypatch, caplog):
        _insert(db, [_photo(aesthetic_iaa=6.0, liqe_score=4.0)])
        _decline(db, '/a.jpg')

        with caplog.at_level(logging.INFO):
            executed = _recompute(db, monkeypatch)

        assert executed == []
        assert "nothing to do" in caplog.text
        assert "face_quality_iqa: 0 NULL" in caplog.text

    def test_a_photo_without_a_decline_is_still_work(self, db, monkeypatch, caplog):
        _insert(db, [_photo(aesthetic_iaa=6.0, liqe_score=4.0)])

        with caplog.at_level(logging.INFO):
            executed = _recompute(db, monkeypatch)

        assert executed != []
        assert "face_quality_iqa: 1 NULL" in caplog.text

    def test_a_none_without_a_decline_flag_is_retried(self, db):
        """An OOM (or any unexpected failure) leaves ``last_declined`` empty: not remembered."""
        _insert(db, [_photo()])
        first = _FakeScorer([None])
        counters = _pass(_facet(db, {'topiq_nr_face': first}), ['topiq_nr_face'])

        assert _declines(db) == []
        assert counters['declined'] == 0

        second = _FakeScorer([8.0])
        _pass(_facet(db, {'topiq_nr_face': second}), ['topiq_nr_face'])
        assert len(second.seen) == 1 and len(second.seen[0]) == 1
        assert _column(db, 'face_quality_iqa') == {'/a.jpg': 8.0}

    def test_a_decline_is_recorded_per_position_in_a_batch(self, db):
        _insert(db, [_photo('/a.jpg', thumbnail=_jpeg((32, 32))),
                     _photo('/b.jpg', thumbnail=_jpeg((48, 48)))])

        class _DeclinesTheWide(_ByWidth):
            def score_batch(self, images):
                scores = super().score_batch(images)
                self.last_declined = frozenset(
                    i for i, img in enumerate(images) if img.size[0] == 48)
                return [None if i in self.last_declined else s for i, s in enumerate(scores)]

        _pass(_facet(db, {'topiq_nr_face': _DeclinesTheWide()}), ['topiq_nr_face'])

        assert [r[0] for r in _declines(db)] == ['/b.jpg']
        assert _column(db, 'face_quality_iqa') == {'/a.jpg': 3.2, '/b.jpg': None}

    def test_only_a_declining_model_records_a_decline(self, db):
        """A pass over aesthetic_iaa never writes ``iqa_declined`` (NOTE 17), whatever the scorer says."""
        _insert(db, [_photo()])
        scorer = _FakeScorer([None], declined=[{0}])

        counters = _pass(_facet(db, {'topiq_iaa': scorer}), ['topiq_iaa'])

        assert _declines(db) == []
        assert counters['declined'] == 0


# ---------------------------------------------------------------------------
# a rescan invalidates a decline
# ---------------------------------------------------------------------------

class TestDeclineIsInvalidatedByARescan:
    @staticmethod
    def _declined_once(db):
        _insert(db, [_photo()])
        _pass(_facet(db, {'topiq_nr_face': _FakeScorer([None], declined=[{0}])}),
              ['topiq_nr_face'])
        assert len(_declines(db)) == 1

    @staticmethod
    def _selected_again(db):
        scorer = _FakeScorer([8.0])
        _pass(_facet(db, {'topiq_nr_face': scorer}), ['topiq_nr_face'])
        return len(scorer.seen) == 1

    def test_a_new_scanned_at_selects_the_photo_again(self, db):
        self._declined_once(db)
        _exec(db, "UPDATE photos SET scanned_at = '2026-02-02 00:00:00'")
        assert self._selected_again(db)

    def test_a_thumbnail_of_a_different_length_selects_the_photo_again(self, db):
        self._declined_once(db)
        replacement = _jpeg((64, 64))
        assert len(replacement) != len(_THUMB)
        _exec(db, "UPDATE photos SET thumbnail = ?", (replacement,))
        assert self._selected_again(db)

    def test_an_unchanged_photo_is_not_selected_again(self, db):
        self._declined_once(db)
        assert not self._selected_again(db)

    def test_null_scanned_at_fails_open(self, db):
        """``NULL = NULL`` is not true, so a photo with no ``scanned_at`` is never 'declined'."""
        _insert(db, [_photo(scanned_at=None)])
        _pass(_facet(db, {'topiq_nr_face': _FakeScorer([None], declined=[{0}])}),
              ['topiq_nr_face'])

        assert _declines(db) == [('/a.jpg', 'topiq_nr_face', None)]
        assert self._selected_again(db)

    def test_redeclining_after_invalidation_leaves_one_row_with_the_new_fingerprint(self, db):
        self._declined_once(db)
        _exec(db, "UPDATE photos SET scanned_at = '2026-03-03 00:00:00'")

        _pass(_facet(db, {'topiq_nr_face': _FakeScorer([None], declined=[{0}])}),
              ['topiq_nr_face'])

        assert _declines(db) == [('/a.jpg', 'topiq_nr_face', f"2026-03-03 00:00:00|{len(_THUMB)}")]


# ---------------------------------------------------------------------------
# a pass over several columns
# ---------------------------------------------------------------------------

class TestMultiModelPassHonoursTheDecline:
    def test_a_photo_declined_for_nr_face_is_still_scored_for_iaa(self, db):
        _insert(db, [_photo('/a.jpg', thumbnail=_jpeg((32, 32))),
                     _photo('/b.jpg', thumbnail=_jpeg((48, 48)))])
        _decline(db, '/a.jpg')
        iaa, nr_face = _ByWidth(), _ByWidth()
        facet = _facet(db, {'topiq_iaa': iaa, 'topiq_nr_face': nr_face})

        counters = _pass(facet, ['topiq_iaa', 'topiq_nr_face'])

        assert sorted(s for call in iaa.seen for s in call) == [(32, 32), (48, 48)]
        assert [s for call in nr_face.seen for s in call] == [(48, 48)]
        assert _column(db, 'aesthetic_iaa') == {'/a.jpg': 3.2, '/b.jpg': 4.8}
        assert _column(db, 'face_quality_iqa') == {'/a.jpg': None, '/b.jpg': 4.8}
        assert counters['updated'] == {'aesthetic_iaa': 2, 'face_quality_iqa': 1}


# ---------------------------------------------------------------------------
# LIQE short side
# ---------------------------------------------------------------------------

class TestLiqeShortSide:
    @pytest.mark.parametrize("size, too_small", [
        ((224, 224), False),
        ((223, 640), True),
        ((640, 223), True),
        ((640, 224), False),
        ((224, 640), False),
        ((2000, 230), True),      # the long-edge bound makes it 1024x117
        ((1100, 241), False),     # ... and this one 1024x224
        ((1100, 240), True),      # ... and this one 1024x223
    ])
    def test_liqe_input_too_small(self, size, too_small):
        from utils.image_bounds import liqe_input_too_small
        assert liqe_input_too_small(*size) is too_small

    def test_the_constant_is_pyiqas_assertion(self):
        from utils.image_bounds import LIQE_MIN_SHORT_SIDE
        assert LIQE_MIN_SHORT_SIDE == 224

    def test_a_too_small_thumbnail_never_reaches_liqe_and_records_nothing(self, db):
        _insert(db, [_photo(thumbnail=_jpeg((300, 100)), face_count=None)])

        counters = _pass(_facet(db, {'liqe': _MustNotScore()}), ['liqe'])

        assert counters['liqe_too_small'] == 1
        assert counters['errors'] == 0
        assert counters['updated']['liqe_score'] == 0
        assert _column(db, 'liqe_score') == {'/a.jpg': None}
        assert _declines(db) == []

    def test_a_thumbnail_on_the_boundary_is_scored(self, db):
        _insert(db, [_photo(thumbnail=_jpeg((640, 224)), face_count=None)])
        scorer = _FakeScorer([3.3])

        counters = _pass(_facet(db, {'liqe': scorer}), ['liqe'])

        assert scorer.seen == [[(640, 224)]]
        assert counters['liqe_too_small'] == 0
        assert _column(db, 'liqe_score') == {'/a.jpg': 3.3}

    def test_a_too_small_thumbnail_is_still_decoded_for_the_other_columns(self, db):
        _insert(db, [_photo(thumbnail=_jpeg((300, 100)), face_count=None)])
        iaa = _FakeScorer([6.5])
        facet = _facet(db, {'topiq_iaa': iaa, 'liqe': _MustNotScore()})

        counters = _pass(facet, ['topiq_iaa', 'liqe'])

        assert iaa.seen == [[(300, 100)]]
        assert counters['liqe_too_small'] == 1
        assert _column(db, 'aesthetic_iaa') == {'/a.jpg': 6.5}
        assert _column(db, 'liqe_score') == {'/a.jpg': None}

    def test_header_check_does_not_decode(self):
        """Premise of the skip: ``Image.open`` reads the header only."""
        full = _noisy_jpeg()
        truncated = full[:len(full) * 6 // 10]
        assert Image.open(BytesIO(truncated)).size == (100, 50)
        with pytest.raises(OSError):
            Image.open(BytesIO(truncated)).convert('RGB')

    def test_a_truncated_too_small_thumbnail_is_skipped_without_an_error(self, db):
        full = _noisy_jpeg()
        _insert(db, [_photo(thumbnail=full[:len(full) * 6 // 10], face_count=None)])

        counters = _pass(_facet(db, {'liqe': _MustNotScore()}), ['liqe'])

        assert counters['liqe_too_small'] == 1
        assert counters['errors'] == 0

    def test_a_corrupt_thumbnail_is_still_an_error(self, db):
        """The lazy PIL calls stay inside the existing try (NOTE 12)."""
        _insert(db, [_photo(thumbnail=_NOT_AN_IMAGE, face_count=None)])

        counters = _pass(_facet(db, {'liqe': _MustNotScore()}), ['liqe'])

        assert counters['errors'] == 1
        assert counters['liqe_too_small'] == 0


# ---------------------------------------------------------------------------
# the summary
# ---------------------------------------------------------------------------

class TestSummaryLines:
    @staticmethod
    def _seed(db):
        _insert(db, [_photo()])

    def test_both_lines_are_logged_with_their_counts(self, db, monkeypatch, caplog):
        self._seed(db)

        def fake_pass(counters):
            counters['declined'] = 2
            counters['liqe_too_small'] = 3

        with caplog.at_level(logging.INFO):
            _recompute(db, monkeypatch, fake_pass)

        assert "  liqe_score: 3 photo(s) skipped, short side under 224 px (stay NULL)" in caplog.text
        assert ("  face_quality_iqa: 2 photo(s) the model declined (no face found in the "
                "thumbnail); not retried until the photo is rescanned") in caplog.text

    def test_neither_line_is_logged_when_the_counts_are_zero(self, db, monkeypatch, caplog):
        self._seed(db)
        with caplog.at_level(logging.INFO):
            _recompute(db, monkeypatch)

        assert "short side under 224 px" not in caplog.text
        assert "the model declined" not in caplog.text


# ---------------------------------------------------------------------------
# the two writers that replace a thumbnail without touching scanned_at
# ---------------------------------------------------------------------------

class TestThumbnailWritersClearDeclines:
    def test_refresh_thumbnails_clears_the_declines_of_a_rewritten_photo(self, db, monkeypatch):
        import facet as facet_cli
        from config import ScoringConfig

        _insert(db, [_photo('/photos/a.CR2', thumbnail=b'old-a'),
                     _photo('/photos/b.CR2', thumbnail=b'old-b')])
        _decline(db, '/photos/a.CR2')
        _decline(db, '/photos/b.CR2')
        # a.CR2 re-renders; b.CR2 does not (None keeps its old thumbnail and its decline).
        monkeypatch.setattr(
            facet_cli, '_display_thumbnail_bytes',
            lambda path, size, quality, sequence_kind: b'new-a' if path.endswith('a.CR2') else None)

        updated, failed, _ = facet_cli.refresh_thumbnails(
            db, ScoringConfig(validate=False), workers=1)

        assert (updated, failed) == (1, 1)
        assert _column(db, 'thumbnail') == {'/photos/a.CR2': b'new-a', '/photos/b.CR2': b'old-b'}
        assert [r[0] for r in _declines(db)] == ['/photos/b.CR2']

    def test_fix_thumbnail_rotation_clears_the_declines_of_a_rotated_photo(
            self, db, tmp_path, monkeypatch):
        import processing.scorer as scorer_module

        rotated_file = tmp_path / 'rotated.jpg'
        rotated_file.write_bytes(b'original')
        upright_file = tmp_path / 'upright.jpg'
        upright_file.write_bytes(b'original')
        thumb = _jpeg((64, 32))
        _insert(db, [_photo(str(rotated_file), thumbnail=thumb),
                     _photo(str(upright_file), thumbnail=thumb)])
        _decline(db, str(rotated_file))
        _decline(db, str(upright_file))
        monkeypatch.setattr(
            scorer_module, '_get_exif_orientation',
            lambda p: 6 if p == str(rotated_file) else 1)

        scorer_module.fix_thumbnail_rotation(db)

        thumbs = _column(db, 'thumbnail')
        assert Image.open(BytesIO(thumbs[str(rotated_file)])).size == (32, 64)   # it was rotated
        assert thumbs[str(upright_file)] == thumb                               # it was not
        assert [r[0] for r in _declines(db)] == [str(upright_file)]
