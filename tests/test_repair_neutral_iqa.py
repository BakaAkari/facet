"""`--repair-neutral-iqa`: clear the fake 5.0 pre-1.20 versions stored in the
supplementary IQA columns.

Before 1.20 a metric a model could not score was stored as a neutral 5.0 in
`aesthetic_iaa`, `face_quality_iqa` and `liqe_score`. The repair NULLs exactly
those three columns where the value is exactly 5.0, in ONE transaction, and
touches nothing else -- `topiq_score` / `aesthetic` / `aesthetic_v25` /
`quality_score` can hold a genuine 5.0 (and `aesthetic` is the CLIP-MLP column
on legacy/8gb), so they are deliberately out of scope.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys

import pytest

import db.maintenance as maintenance
from db.maintenance import NEUTRAL_IQA_COLUMNS, repair_neutral_iqa
from db.schema import init_database
from db.stats_cache import METRIC_RANGES_KEY

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FACET_SCRIPT = os.path.join(_REPO_ROOT, "facet.py")

_THREE = ('aesthetic_iaa', 'face_quality_iqa', 'liqe_score')
_UNTOUCHED = ('aesthetic_v25', 'topiq_score', 'aesthetic', 'quality_score')


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "neutral.db")
    init_database(path)
    return path


def _insert(path, photo, **cols):
    conn = sqlite3.connect(path)
    names = ", ".join(["path", "filename", *cols])
    marks = ", ".join("?" for _ in range(2 + len(cols)))
    conn.execute(f"INSERT INTO photos ({names}) VALUES ({marks})",
                 [photo, photo.rsplit('/', 1)[-1], *cols.values()])
    conn.commit()
    conn.close()


def _get(path, photo, *cols):
    conn = sqlite3.connect(path)
    row = conn.execute(
        f"SELECT {', '.join(cols)} FROM photos WHERE path = ?", (photo,)).fetchone()
    conn.close()
    return row


def test_the_constant_names_the_three_columns():
    assert tuple(NEUTRAL_IQA_COLUMNS) == _THREE


class TestRepair:
    def test_clears_exactly_the_three_columns(self, db_path):
        _insert(db_path, '/r/a.jpg', aesthetic_iaa=5.0, face_quality_iqa=5.0, liqe_score=5.0,
                aesthetic_v25=5.0, topiq_score=5.0, aesthetic=5.0, quality_score=5.0)
        counts = repair_neutral_iqa(db_path, verbose=False)
        assert counts == {'aesthetic_iaa': 1, 'face_quality_iqa': 1, 'liqe_score': 1}
        assert _get(db_path, '/r/a.jpg', *_THREE) == (None, None, None)
        assert _get(db_path, '/r/a.jpg', *_UNTOUCHED) == (5.0, 5.0, 5.0, 5.0)

    def test_leaves_other_values_alone(self, db_path):
        values = [5.01, 4.99, 4.995, 0, 10, None]
        for i, v in enumerate(values):
            _insert(db_path, f'/r/v{i}.jpg', aesthetic_iaa=v, face_quality_iqa=v, liqe_score=v)
        assert repair_neutral_iqa(db_path, verbose=False) == {c: 0 for c in _THREE}
        for i, v in enumerate(values):
            stored = _get(db_path, f'/r/v{i}.jpg', *_THREE)
            assert stored == (v, v, v)

    def test_counts_are_per_column(self, db_path):
        _insert(db_path, '/r/1.jpg', aesthetic_iaa=5.0, face_quality_iqa=5.0, liqe_score=5.0)
        _insert(db_path, '/r/2.jpg', aesthetic_iaa=5.0, face_quality_iqa=7.0, liqe_score=5.0)
        _insert(db_path, '/r/3.jpg', aesthetic_iaa=5.0, face_quality_iqa=7.0, liqe_score=6.5)
        assert repair_neutral_iqa(db_path, verbose=False) == {
            'aesthetic_iaa': 3, 'face_quality_iqa': 1, 'liqe_score': 2}

    def test_idempotent(self, db_path):
        _insert(db_path, '/r/a.jpg', aesthetic_iaa=5.0, liqe_score=5.0, face_quality_iqa=6.0)
        repair_neutral_iqa(db_path, verbose=False)
        before = _get(db_path, '/r/a.jpg', *_THREE)
        assert repair_neutral_iqa(db_path, verbose=False) == {c: 0 for c in _THREE}
        assert _get(db_path, '/r/a.jpg', *_THREE) == before == (None, 6.0, None)

    def test_is_atomic(self, db_path, monkeypatch):
        """The third UPDATE raises: the first two must be rolled back, not committed."""
        _insert(db_path, '/r/a.jpg', aesthetic_iaa=5.0, face_quality_iqa=5.0, liqe_score=5.0)
        monkeypatch.setattr(
            maintenance, 'NEUTRAL_IQA_COLUMNS',
            ('aesthetic_iaa', 'face_quality_iqa', 'no_such_column'))
        with pytest.raises(sqlite3.OperationalError):
            repair_neutral_iqa(db_path, verbose=False)
        assert _get(db_path, '/r/a.jpg', *_THREE) == (5.0, 5.0, 5.0)

    def test_uses_the_column_indexes(self, db_path):
        """An index seek, not a BLOB-walking table scan (thumbnail/embedding sit inline)."""
        conn = sqlite3.connect(db_path)
        for col in _THREE:
            plan = " ".join(r[3] for r in conn.execute(
                f"EXPLAIN QUERY PLAN UPDATE photos SET {col} = NULL WHERE {col} = 5.0"))
            assert f"idx_{col}" in plan, plan
        conn.close()


class TestSelectedForRecompute:
    def test_repaired_rows_are_selected_by_recompute_iqa(self, db_path):
        from processing.scorer import Facet
        _insert(db_path, '/r/face.jpg', aesthetic_iaa=5.0, face_quality_iqa=5.0,
                liqe_score=5.0, face_count=1)
        _insert(db_path, '/r/faceless.jpg', aesthetic_iaa=5.0, face_quality_iqa=5.0,
                liqe_score=5.0, face_count=0)
        repair_neutral_iqa(db_path, verbose=False)

        def selected(col):
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                f"SELECT path FROM photos WHERE {Facet._iqa_needs_score_sql(col)}").fetchall()
            conn.close()
            return {r[0] for r in rows}

        both = {'/r/face.jpg', '/r/faceless.jpg'}
        assert selected('aesthetic_iaa') == both
        assert selected('liqe_score') == both
        # face_quality_iqa is scored only for a photo with a face: the faceless row
        # stays out, by design.
        assert selected('face_quality_iqa') == {'/r/face.jpg'}


class TestMetricRangesCache:
    def _seed_cache(self, path):
        conn = sqlite3.connect(path)
        conn.execute("INSERT OR REPLACE INTO stats_cache (key, value, updated_at) VALUES (?, ?, ?)",
                     (METRIC_RANGES_KEY, json.dumps({'ranges': {}}), 1.0))
        conn.execute("INSERT OR REPLACE INTO stats_cache (key, value, updated_at) VALUES (?, ?, ?)",
                     ('other_key', '1', 1.0))
        conn.commit()
        conn.close()

    def _keys(self, path):
        """Keys holding a value; an invalidated metric_ranges row is a NULL tombstone."""
        conn = sqlite3.connect(path)
        keys = {r[0] for r in conn.execute("SELECT key FROM stats_cache WHERE value IS NOT NULL")}
        conn.close()
        return keys

    def test_invalidates_metric_ranges(self, db_path):
        """The gallery ranges are computed from these columns; a stale cache would lie."""
        _insert(db_path, '/r/a.jpg', liqe_score=5.0)
        self._seed_cache(db_path)
        repair_neutral_iqa(db_path, verbose=False)
        assert self._keys(db_path) == {'other_key'}
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT value FROM stats_cache WHERE key='metric_ranges'").fetchone()
        conn.close()
        assert row == (None,)

    def test_keeps_the_cache_when_nothing_changed(self, db_path):
        _insert(db_path, '/r/a.jpg', liqe_score=6.0)
        self._seed_cache(db_path)
        repair_neutral_iqa(db_path, verbose=False)
        assert self._keys(db_path) == {METRIC_RANGES_KEY, 'other_key'}


def test_aggregate_is_unchanged():
    """A NULL scores neutral 5.0 at full weight, so the repair moves no aggregate."""
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path
    from processing.scorer import Facet
    facet = Facet.__new__(Facet)
    facet.config = ScoringConfig(defaults_path(), validate=False)
    metrics = dict(face_count=1, face_ratio=0.3, face_quality=8, eye_sharpness=7,
                   aesthetic=7, comp_score=6, tech_sharpness=7, color_score=6,
                   exposure_score=7, tags='', is_monochrome=0, mean_saturation=0.4,
                   is_blink=0)
    neutral, _ = facet.calculate_aggregate_logic(
        dict(metrics, aesthetic_iaa=5.0, face_quality_iqa=5.0, liqe_score=5.0))
    cleared, _ = facet.calculate_aggregate_logic(
        dict(metrics, aesthetic_iaa=None, face_quality_iqa=None, liqe_score=None))
    assert neutral == cleared


def _run_facet(*argv):
    return subprocess.run([sys.executable, _FACET_SCRIPT, *argv],
                          capture_output=True, text=True, timeout=300)


class TestCli:
    def test_repairs_and_prints_the_per_column_lines_and_next_step(self, db_path):
        _insert(db_path, '/r/a.jpg', aesthetic_iaa=5.0, face_quality_iqa=5.0, liqe_score=5.0)
        proc = _run_facet('--repair-neutral-iqa', '--db', db_path)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        for col in _THREE:
            assert f"Cleared 1 neutral 5.0 value(s) in {col}" in out
        assert "--recompute-iqa" in out and "--recompute-average" in out
        assert _get(db_path, '/r/a.jpg', *_THREE) == (None, None, None)

    def test_refuses_dry_run_before_touching_the_database(self, db_path):
        _insert(db_path, '/r/a.jpg', liqe_score=5.0)
        digest = hashlib.sha256(open(db_path, 'rb').read()).hexdigest()
        proc = _run_facet('--repair-neutral-iqa', '--dry-run', '--db', db_path)
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "no dry-run" in proc.stderr
        assert hashlib.sha256(open(db_path, 'rb').read()).hexdigest() == digest
