"""``--recompute-silhouette``: re-derive ``photos.is_silhouette`` from stored thumbnails + faces.

The backfill is database-only. It maps each photo's stored face boxes into the stored 640 px
thumbnail (the box space is not reliably the photo's own pixel space) and then calls the very
same ``detect_silhouette`` the scan calls. These tests build a real schema with
``db.schema.init_database`` and synthetic JPEG thumbnails.

The backfill rewrites every row of the database it is given, so these tests run it on a private
database rather than on the session database the shared ``seeded_photos`` fixture writes into,
where it would also rewrite other suites' rows.
"""

import io
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from db.schema import init_database
from processing import silhouette_backfill as backfill

THUMB_W, THUMB_H = 640, 426
TRUE_BOX = (250, 150, 350, 250)         # the face, in the thumbnail's own grid
LOGGER = 'facet.silhouette_backfill'
REPO = Path(__file__).resolve().parent.parent


def _jpeg(array, quality=95):
    out = io.BytesIO()
    Image.fromarray(array).save(out, 'JPEG', quality=quality)
    return out.getvalue()


def _thumbnail(dark_box=TRUE_BOX, seed=0):
    """Bright frame with a near-black patch at ``dark_box``; ``seed`` makes the bytes distinct."""
    gray = np.full((THUMB_H, THUMB_W), 230, dtype=np.uint8)
    if dark_box is not None:
        x1, y1, x2, y2 = dark_box
        gray[y1:y2, x1:x2] = 10
    gray[0, :8] = 200 + seed % 40           # distinct bytes per row
    return _jpeg(np.stack([gray] * 3, axis=-1))


def _face_thumbnail(dark):
    arr = np.full((128, 128, 3), 10 if dark else 220, dtype=np.uint8)
    return _jpeg(arr)


def _bbox_at(scale, box=TRUE_BOX):
    return [int(round(v * scale)) for v in box]


def _face_ratio(bboxes, width, height):
    return sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes) / (width * height)


def _add_photo(conn, name, *, width, height, face_count, face_ratio, flag, thumbnail, faces=()):
    path = f'/silhouette/{name}.jpg'
    conn.execute(
        "INSERT INTO photos (path, filename, image_width, image_height, face_count, face_ratio, "
        "is_silhouette, thumbnail, aggregate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 5.0)",
        (path, f'{name}.jpg', width, height, face_count, face_ratio, flag, thumbnail))
    for index, (bbox, face_thumbnail) in enumerate(faces):
        conn.execute(
            "INSERT INTO faces (photo_path, face_index, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            "face_thumbnail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (path, index, b'\x00' * 8, *bbox, face_thumbnail))
    return path


def _populate(conn):
    """One photo per case, in an order the resumability test relies on. Returns ``{name: path}``."""
    paths = {}
    seed = iter(range(1, 100))

    # exact: the box space is image_width x image_height (3200 x 2130 -> a 5x thumbnail)
    exact_box = _bbox_at(5)
    exact_ratio = _face_ratio([exact_box], 3200, 2130)
    paths['exact_dark_was_0'] = _add_photo(
        conn, 'exact_dark_was_0', width=3200, height=2130, face_count=1, face_ratio=exact_ratio,
        flag=0, thumbnail=_thumbnail(seed=next(seed)), faces=[(exact_box, _face_thumbnail(True))])
    paths['exact_bright_was_1'] = _add_photo(
        conn, 'exact_bright_was_1', width=3200, height=2130, face_count=1, face_ratio=exact_ratio,
        flag=1, thumbnail=_thumbnail(None, seed=next(seed)), faces=[(exact_box, _face_thumbnail(False))])
    paths['exact_bright_was_null'] = _add_photo(
        conn, 'exact_bright_was_null', width=3200, height=2130, face_count=1, face_ratio=exact_ratio,
        flag=None, thumbnail=_thumbnail(None, seed=next(seed)), faces=[(exact_box, _face_thumbnail(False))])

    # scaled: scanned on the 640 px preview, faces re-extracted at 5200 px (8.125x)
    scaled_box = _bbox_at(8.125)
    scaled_ratio = _face_ratio([scaled_box], 5200, 5200 * THUMB_H / THUMB_W)
    paths['scaled_dark_was_0'] = _add_photo(
        conn, 'scaled_dark_was_0', width=THUMB_W, height=THUMB_H, face_count=1, face_ratio=scaled_ratio,
        flag=0, thumbnail=_thumbnail(seed=next(seed)), faces=[(scaled_box, _face_thumbnail(True))])
    # unresolved: the stored box points at bright pixels at every candidate width, while the
    # stored face crop says the face is dark -> no evidence, written 0
    paths['unresolved_was_1'] = _add_photo(
        conn, 'unresolved_was_1', width=THUMB_W, height=THUMB_H, face_count=1, face_ratio=scaled_ratio,
        flag=1, thumbnail=_thumbnail((20, 20, 120, 120), seed=next(seed)),
        faces=[(scaled_box, _face_thumbnail(True))])
    # a bright face under scaled geometry is cleared without a width search
    paths['scaled_bright_was_1'] = _add_photo(
        conn, 'scaled_bright_was_1', width=THUMB_W, height=THUMB_H, face_count=1, face_ratio=scaled_ratio,
        flag=1, thumbnail=_thumbnail(None, seed=next(seed)), faces=[(scaled_box, _face_thumbnail(False))])

    # count_mismatch: two face rows for one counted face, boxes in a 4000 px space
    cm_box = _bbox_at(6.25)
    paths['count_mismatch_dark_was_0'] = _add_photo(
        conn, 'count_mismatch_dark_was_0', width=THUMB_W, height=THUMB_H, face_count=1, face_ratio=0.02,
        flag=0, thumbnail=_thumbnail(seed=next(seed)),
        faces=[(cm_box, _face_thumbnail(True)), ([10, 10, 60, 60], _face_thumbnail(False))])

    # no faces: a stale flag is cleared, whatever the thumbnail
    paths['no_faces_was_1'] = _add_photo(
        conn, 'no_faces_was_1', width=THUMB_W, height=THUMB_H, face_count=0, face_ratio=0.0,
        flag=1, thumbnail=_thumbnail(seed=next(seed)))

    # G5: the face evidence is unknown / the thumbnail is absent -> the flag is left alone
    paths['face_count_null_was_1'] = _add_photo(
        conn, 'face_count_null_was_1', width=THUMB_W, height=THUMB_H, face_count=None, face_ratio=None,
        flag=1, thumbnail=_thumbnail(seed=next(seed)))
    paths['no_thumbnail_was_1'] = _add_photo(
        conn, 'no_thumbnail_was_1', width=THUMB_W, height=THUMB_H, face_count=1, face_ratio=exact_ratio,
        flag=1, thumbnail=None, faces=[(exact_box, _face_thumbnail(True))])
    return paths


EXPECTED = {
    'exact_dark_was_0': 1,                  # set
    'exact_bright_was_1': 0,                # cleared
    'exact_bright_was_null': None,          # stays NULL: nothing to change
    'scaled_dark_was_0': 1,                 # mapped through the refined width, set
    'unresolved_was_1': 0,                  # cleared
    'scaled_bright_was_1': 0,               # cleared
    'count_mismatch_dark_was_0': 1,         # wide search, set
    'no_faces_was_1': 0,                    # cleared
    'face_count_null_was_1': 1,             # skipped
    'no_thumbnail_was_1': 1,                # skipped
}


@pytest.fixture()
def library(tmp_path):
    db_path = str(tmp_path / 'silhouette.db')
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    paths = _populate(conn)
    conn.commit()
    conn.close()
    return db_path, paths


def _flags(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return dict(conn.execute("SELECT path, is_silhouette FROM photos").fetchall())
    finally:
        conn.close()


def _as_names(flags, paths):
    return {name: flags[path] for name, path in paths.items()}


class TestBackfill:
    def test_every_case_lands_on_its_expected_flag(self, library):
        db_path, paths = library
        backfill.recompute_silhouette(db_path)
        assert _as_names(_flags(db_path), paths) == EXPECTED

    def test_the_stats_account_for_every_row(self, library):
        db_path, _ = library
        stats = backfill.recompute_silhouette(db_path)
        assert stats['rows'] == 10
        assert stats['classes'] == {'exact': 3, 'scaled': 3, 'count_mismatch': 1, 'other': 0}
        assert stats['flagged'] == {'exact': 1, 'scaled': 1, 'count_mismatch': 1, 'other': 0}
        assert stats['unresolved'] == {'exact': 0, 'scaled': 1, 'count_mismatch': 0, 'other': 0}
        assert stats['negative_by_core'] == {'exact': 0, 'scaled': 1, 'count_mismatch': 0, 'other': 0}
        assert stats['face_count_zero'] == 1
        assert stats['flagged_total'] == 3
        assert (stats['set'], stats['cleared']) == (3, 4)
        assert stats['unchanged'] == 1                      # the NULL flag that stays NULL
        assert stats['skipped_face_count_null'] == 1
        assert stats['skipped_no_thumbnail'] == 1

    def test_a_face_count_of_null_keeps_a_stale_flag_and_is_counted(self, library):
        db_path, paths = library
        stats = backfill.recompute_silhouette(db_path)
        assert _flags(db_path)[paths['face_count_null_was_1']] == 1
        assert stats['skipped_face_count_null'] == 1

    def test_a_missing_thumbnail_keeps_the_flag_and_is_counted(self, library):
        db_path, paths = library
        stats = backfill.recompute_silhouette(db_path)
        assert _flags(db_path)[paths['no_thumbnail_was_1']] == 1
        assert stats['skipped_no_thumbnail'] == 1

    def test_both_skips_are_logged_by_a_real_run(self, library, caplog):
        db_path, _ = library
        with caplog.at_level(logging.INFO, logger=LOGGER):
            backfill.recompute_silhouette(db_path)
        text = caplog.text
        assert 'skipped_face_count_null=1' in text
        assert 'skipped_no_thumbnail=1' in text
        assert '--recompute-average' in text

    def test_dry_run_reports_everything_and_writes_nothing(self, library, caplog):
        db_path, paths = library
        before = _flags(db_path)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            dry = backfill.recompute_silhouette(db_path, dry_run=True)
        assert _flags(db_path) == before
        assert 'skipped_face_count_null=1' in caplog.text
        assert 'skipped_no_thumbnail=1' in caplog.text
        assert 'nothing was written' in caplog.text
        real = backfill.recompute_silhouette(db_path)
        assert dry == real                                  # the report is the real run's, unwritten
        assert _as_names(_flags(db_path), paths) == EXPECTED

    def test_a_second_run_changes_nothing(self, library):
        db_path, _ = library
        backfill.recompute_silhouette(db_path)
        after_first = _flags(db_path)
        again = backfill.recompute_silhouette(db_path)
        assert _flags(db_path) == after_first
        assert (again['set'], again['cleared']) == (0, 0)
        assert again['unchanged'] == again['rows'] - 2      # the two skipped rows

    def test_only_is_silhouette_is_written(self, library):
        db_path, _ = library
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        before = [dict(r) for r in conn.execute("SELECT * FROM photos ORDER BY rowid")]
        conn.close()
        backfill.recompute_silhouette(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        after = [dict(r) for r in conn.execute("SELECT * FROM photos ORDER BY rowid")]
        faces_n = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        conn.close()
        assert faces_n == 9
        for old, new in zip(before, after):
            assert {k for k in old if old[k] != new[k]} <= {'is_silhouette'}

    def test_an_interrupted_run_keeps_its_finished_batches_and_a_rerun_converges(self, library, monkeypatch):
        db_path, paths = library
        real_decode = backfill._decode_gray
        calls = {'n': 0}

        def flaky(blob):
            calls['n'] += 1
            if calls['n'] == 3:                             # the first row of the second batch
                raise RuntimeError('interrupted')
            return real_decode(blob)

        monkeypatch.setattr(backfill, '_decode_gray', flaky)
        with pytest.raises(RuntimeError):
            backfill.recompute_silhouette(db_path, batch_size=2)
        partial = _as_names(_flags(db_path), paths)
        assert partial['exact_dark_was_0'] == 1             # batch 1 was committed
        assert partial['exact_bright_was_1'] == 0
        assert partial['scaled_dark_was_0'] == 0            # batch 2 was not
        assert partial['unresolved_was_1'] == 1

        monkeypatch.setattr(backfill, '_decode_gray', real_decode)
        backfill.recompute_silhouette(db_path, batch_size=2)
        assert _as_names(_flags(db_path), paths) == EXPECTED

    def test_batch_size_does_not_change_the_result(self, library):
        db_path, paths = library
        backfill.recompute_silhouette(db_path, batch_size=1)
        assert _as_names(_flags(db_path), paths) == EXPECTED


class TestGeometry:
    def test_classes(self):
        face = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}
        photo = {'image_width': 1000.0, 'image_height': 1000.0, 'face_ratio': 0.01, 'face_count': 1.0}
        assert backfill.classify_geometry(photo, [face]) == ('exact', None)
        # boxes 8x too large for the stored image: scaled, implied width 8000
        scaled = {**photo, 'face_ratio': 0.01 / 64}
        assert backfill.classify_geometry(scaled, [face]) == ('scaled', pytest.approx(8000.0))
        # a face row more than the counted faces: count_mismatch
        assert backfill.classify_geometry(scaled, [face, face])[0] == 'count_mismatch'
        # no usable face_ratio / image size: nothing to classify against
        assert backfill.classify_geometry({**photo, 'face_ratio': None}, [face])[0] == 'other'
        assert backfill.classify_geometry({**photo, 'image_width': None}, [face, face])[0] == 'count_mismatch'

    def test_a_scaled_set_whose_boxes_do_not_fit_the_implied_frame_is_other(self):
        photo = {'image_width': 640.0, 'image_height': 426.0, 'face_ratio': 0.01 / 64, 'face_count': 2.0}
        boxes = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}, {'x1': 999000, 'y1': 0, 'x2': 1000000, 'y2': 10}]
        assert backfill.classify_geometry(photo, boxes)[0] == 'other'
        assert backfill.classify_geometry(photo, boxes[:1] * 2)[0] == 'scaled'


class TestRegistration:
    def test_the_flag_takes_the_library_lock(self):
        from facet import LIBRARY_JOB_ARGS

        assert 'recompute_silhouette' in LIBRARY_JOB_ARGS

    def test_the_job_args_stay_alphabetical_around_it(self):
        from facet import LIBRARY_JOB_ARGS

        names = list(LIBRARY_JOB_ARGS)
        index = names.index('recompute_silhouette')
        assert names[index - 1] == 'recompute_saliency'
        assert names[index + 1] == 'recompute_skin_tone'

    def test_the_cli_honours_dry_run_and_exits_clean(self, library):
        db_path, _ = library
        before = _flags(db_path)
        result = subprocess.run(
            [sys.executable, str(REPO / 'facet.py'), '--recompute-silhouette', '--dry-run', '--db', db_path],
            capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert 'skipped_face_count_null=1' in output
        assert 'skipped_no_thumbnail=1' in output
        assert _flags(db_path) == before

    def test_the_cli_writes_without_dry_run(self, library):
        db_path, paths = library
        result = subprocess.run(
            [sys.executable, str(REPO / 'facet.py'), '--recompute-silhouette', '--db', db_path],
            capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _as_names(_flags(db_path), paths) == EXPECTED
