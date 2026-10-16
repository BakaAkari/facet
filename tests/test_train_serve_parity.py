"""Train == serve (T9b, defect 1): the optimizer fits the rows the scorer serves.

``Facet.update_all_aggregates`` rewrites ``tech_sharpness`` / ``color_score`` /
``eye_sharpness`` from their ``raw_*`` columns with the percentile normalizer,
and ``exposure_score`` from the stored histogram, in memory only. The optimizer
used to read the stored columns, so it trained on features the scorer never
used (eye/tech/colour means 1.1 / 1.1 / 7.5 stored vs 2.5 / 2.4 / 9.0 served).
"""

from unittest import mock

import numpy as np
import pytest

from config import ScoringConfig
from config.percentile_normalizer import PercentileNormalizer
from db import get_connection, init_database
from optimization.weight_optimizer import WeightOptimizer
from processing.scorer import Facet, build_metric_vector
from utils.histogram import HISTOGRAM_BINS, pack_histogram


def _histogram(rng):
    counts = np.zeros(HISTOGRAM_BINS)
    centre = int(rng.integers(40, 220))
    counts[max(0, centre - 30):centre + 30] = rng.uniform(50, 500)
    return pack_histogram(counts, counts, counts, counts)


def _seed(db_path, n=60, seed=5):
    """Mixed face / no-face photos whose stored (un-normalized) columns differ from the served ones."""
    rng = np.random.default_rng(seed)
    cols = [
        'path', 'filename', 'aesthetic', 'face_count', 'face_ratio', 'face_quality',
        'face_sharpness', 'eye_sharpness', 'tech_sharpness', 'color_score',
        'raw_sharpness_variance', 'raw_color_entropy', 'raw_eye_sharpness',
        'exposure_score', 'comp_score', 'power_point_score', 'contrast_score',
        'histogram_data', 'histogram_spread', 'mean_luminance', 'mean_saturation',
        'noise_sigma', 'isolation_bonus', 'leading_lines_score', 'tags',
        'is_silhouette', 'is_monochrome', 'liqe_score',
    ]
    rows = []
    for i in range(n):
        faces = int(i % 3 == 0)
        rows.append((
            f'/parity/p{i:03d}.jpg', f'p{i:03d}.jpg', float(rng.uniform(2, 9)),
            faces, 0.12 if faces else 0.0, float(rng.uniform(3, 9)) if faces else None,
            float(rng.uniform(1, 9)) if faces else None,
            # stored eye/tech/colour are the OLD un-normalized values (small), the raw_* are what serves
            float(rng.uniform(0.5, 2)) if faces else None, float(rng.uniform(0.5, 2)),
            float(rng.uniform(5, 8)),
            float(rng.uniform(50, 900)), float(rng.uniform(2, 7)),
            float(rng.uniform(5, 80)) if faces else None,
            3.0, float(rng.uniform(3, 8)), 5.0, float(rng.uniform(3, 8)),
            _histogram(rng), float(rng.uniform(30, 60)), float(rng.uniform(0.2, 0.7)),
            float(rng.uniform(0.2, 0.7)), float(rng.uniform(0.5, 3)), 1.0, 0.0, '',
            0, 0, float(rng.uniform(3, 9)),
        ))
    with get_connection(db_path, row_factory=False) as conn:
        conn.executemany(
            f"INSERT INTO photos ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", rows)
        conn.commit()


@pytest.fixture()
def cfg_path(tmp_path):
    """An empty override: the shipped defaults, never the operator's own config."""
    path = tmp_path / 'scoring_config.json'
    path.write_text('{}')
    return str(path)


@pytest.fixture()
def served(tmp_path, cfg_path):
    """(db_path, {path: row as calculate_aggregate_logic received it}, {path: category})."""
    db_path = str(tmp_path / 'parity.db')
    init_database(db_path)
    _seed(db_path)
    facet = Facet(db_path=db_path, config_path=cfg_path, lightweight=True)
    norm = PercentileNormalizer(db_path, target_percentile=90, per_category=True, category_min_samples=3)
    norm.compute_percentiles()
    # the first pass stores categories, which the per-category percentiles group by
    facet.update_all_aggregates(use_embeddings=False, normalizer=norm)

    captured = {}
    real = Facet.calculate_aggregate_logic

    def spy(self, m, config=None):
        result = real(self, m, config)
        captured[m['path']] = (dict(m), result[1])
        return result

    with mock.patch.object(Facet, 'calculate_aggregate_logic', spy):
        facet.update_all_aggregates(use_embeddings=False, normalizer=norm)
    with get_connection(db_path) as conn:
        norm.save_to_stats_cache(conn)
        conn.commit()
    return db_path, captured, cfg_path


class TestMetricVectorIsTheServedVector:
    def test_every_component_matches_the_served_row(self, served):
        db_path, captured, cfg_path = served
        opt = WeightOptimizer(db_path, cfg_path)
        cfg = opt.cfg
        categories = {cat for _, cat in captured.values()}
        assert len(categories) >= 2, "the fixture must serve at least two categories"
        checked = 0
        with get_connection(db_path) as conn:
            for path, (served_row, category) in captured.items():
                stored = dict(conn.execute("SELECT * FROM photos WHERE path = ?", (path,)).fetchone())
                got = opt._metric_vector(stored, category)
                vec = build_metric_vector(served_row, cfg, category)
                want = [max(0.0, min(10.0, vec.get(c, 0.0))) for c in opt.SCORE_COMPONENTS]
                for comp, g, w in zip(opt.SCORE_COMPONENTS, got, want):
                    assert g == pytest.approx(w, abs=1e-3), f'{comp} on {path} ({category})'
                checked += 1
        assert checked == len(captured) > 0

    def test_eye_tech_and_colour_are_the_normalized_values(self, served):
        """The three rewritten columns do move (otherwise the test above proves nothing)."""
        db_path, captured, cfg_path = served
        opt = WeightOptimizer(db_path, cfg_path)
        faces = [(p, r) for p, (r, _) in captured.items() if r.get('face_count')]
        assert faces
        path, served_row = faces[0]
        with get_connection(db_path) as conn:
            stored = dict(conn.execute("SELECT * FROM photos WHERE path = ?", (path,)).fetchone())
        assert served_row['eye_sharpness'] != pytest.approx(stored['eye_sharpness'])
        vec = opt._metric_vector(stored, captured[path][1])
        idx = opt.SCORE_COMPONENTS.index
        assert vec[idx('eye_sharpness')] == pytest.approx(served_row['eye_sharpness'], abs=1e-3)
        assert vec[idx('tech_sharpness')] == pytest.approx(
            build_metric_vector(served_row, opt.cfg, captured[path][1])['tech_sharpness'], abs=1e-3)
        assert vec[idx('color')] == pytest.approx(served_row['color_score'], abs=1e-3)

    def test_without_a_snapshot_the_stored_columns_are_what_is_served(self, tmp_path, cfg_path):
        """No percentile recompute ever ran: nothing is rewritten, nothing breaks."""
        db_path = str(tmp_path / 'nosnap.db')
        init_database(db_path)
        _seed(db_path, n=6)
        opt = WeightOptimizer(db_path, cfg_path)
        with get_connection(db_path) as conn:
            row = dict(conn.execute("SELECT * FROM photos WHERE face_count = 1 LIMIT 1").fetchone())
        vec = opt._metric_vector(row, 'portrait')
        assert vec[opt.SCORE_COMPONENTS.index('eye_sharpness')] == pytest.approx(row['eye_sharpness'])

    def test_category_none_uses_the_photos_own_category(self, served):
        db_path, captured, cfg_path = served
        opt = WeightOptimizer(db_path, cfg_path)
        with get_connection(db_path) as conn:
            for path, (served_row, category) in list(captured.items())[:12]:
                stored = dict(conn.execute("SELECT * FROM photos WHERE path = ?", (path,)).fetchone())
                assert opt._metric_vector(stored, None) == pytest.approx(
                    opt._metric_vector(stored, category), abs=1e-3)


class TestPrepareScoringRow:
    def test_is_a_static_method_and_returns_the_same_dict(self, cfg_path):
        row = {'path': '/x.jpg', 'face_count': 5}
        cfg = ScoringConfig(cfg_path, validate=False)
        out = Facet.prepare_scoring_row(row, cfg, None)
        assert out is row
        assert row['is_group_portrait'] == 1

    def test_determine_photo_category_needs_no_instance(self, cfg_path):
        cfg = ScoringConfig(cfg_path, validate=False)
        assert Facet._determine_photo_category({'tags': '', 'face_count': 0}, cfg)
