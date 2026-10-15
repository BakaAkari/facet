"""The weight optimizer's apply gate (optimization/weight_optimizer.py).

``optimize_weights_with_cv`` is the one statistical gate both the CLI apply
and the viewer's ``suggest_changes`` go through: photo-grouped folds, pooled
held-out predictions over decisive pairs, and a one-sided exact McNemar test
on the pairs the fitted and the current weights disagree about. These tests
pin its calibration (random winners must rarely pass, a planted signal must),
the fold hygiene, the decisive-pair guards, and the cost contract of the
direct path (CV only when the in-sample fit already clears the threshold).
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from db.schema import init_database
from optimization import weight_optimizer as wo
from optimization.weight_optimizer import WeightOptimizer, run_weight_optimization

REPO_CONFIG = Path(__file__).resolve().parent.parent / 'config' / 'scoring_config.default.json'

# Photo columns build_metric_vector reads; each gets an independent random value
NOISE_COLUMNS = (
    'aesthetic', 'face_quality', 'eye_sharpness', 'tech_sharpness', 'comp_score',
    'exposure_score', 'color_score', 'contrast_score', 'liqe_score', 'aesthetic_iaa',
    'face_quality_iqa', 'subject_sharpness',
)


def build_db(db_path, *, n_decisive, n_ties=0, seed=0, agree=None, category='portrait',
             only_aesthetic=False):
    """Disjoint photo pairs (one component each) with random metric columns.

    agree=None: every decisive winner is a coin flip (no weight vector is better
    than another). agree=p: the higher-aesthetic photo wins with probability p.
    only_aesthetic: every other metric stays NULL (scored as a constant 5.0).
    """
    import sqlite3
    rng = np.random.default_rng(seed)
    init_database(db_path)
    columns = ('aesthetic',) if only_aesthetic else NOISE_COLUMNS
    n_pairs = n_decisive + n_ties
    conn = sqlite3.connect(db_path)
    values = {}
    for i in range(2 * n_pairs):
        path = f'/g/p{i:04d}.jpg'
        row = {c: float(rng.uniform(1, 9)) for c in columns}
        values[path] = row
        cols = ', '.join(('path', 'filename') + tuple(row))
        marks = ', '.join('?' * (2 + len(row)))
        conn.execute(f"INSERT INTO photos ({cols}) VALUES ({marks})",
                     (path, path.rsplit('/', 1)[1], *row.values()))
    for i in range(n_pairs):
        pa, pb = f'/g/p{2 * i:04d}.jpg', f'/g/p{2 * i + 1:04d}.jpg'
        if i >= n_decisive:
            winner = 'tie'
        elif agree is None:
            winner = 'a' if rng.random() < 0.5 else 'b'
        else:
            better = 'a' if values[pa]['aesthetic'] > values[pb]['aesthetic'] else 'b'
            worse = 'b' if better == 'a' else 'a'
            winner = better if rng.random() < agree else worse
        conn.execute(
            "INSERT INTO comparisons (photo_a_path, photo_b_path, winner, category, source) "
            "VALUES (?, ?, ?, ?, 'vote')", (pa, pb, winner, category))
    conn.commit()
    conn.close()
    return db_path


def _optimizer(db_path):
    return WeightOptimizer(db_path, str(REPO_CONFIG))


class TestNoiseCalibration:
    """(5) Under pure noise every apply is a false apply."""

    SEEDS = range(20)

    def test_cli_false_apply_rate_under_random_winners(self, tmp_path):
        applied = []
        for seed in self.SEEDS:
            db = build_db(str(tmp_path / f'noise{seed}.db'), n_decisive=80, n_ties=10, seed=seed)
            cfg = tmp_path / f'cfg{seed}.json'
            shutil.copy(REPO_CONFIG, cfg)
            before = cfg.read_text()
            run_weight_optimization(db_path=db, config_path=str(cfg), category='portrait',
                                    min_comparisons=30)
            applied.append(cfg.read_text() != before)
        rate = sum(applied) / len(applied)
        # The pre-fix gate (mean pair-fold CV - accuracy_before >= 2 pp) false-applied
        # 30-45% of the time; the McNemar gate's design rate is <= 5%.
        assert rate <= 0.15, f'false-apply rate {rate:.2f} under random winners'

    def test_planted_signal_with_150_decisive_pairs_applies(self, tmp_path):
        db = build_db(str(tmp_path / 'signal.db'), n_decisive=150, seed=7, agree=0.85)
        cfg = tmp_path / 'cfg.json'
        shutil.copy(REPO_CONFIG, cfg)
        before = cfg.read_text()
        run_weight_optimization(db_path=db, config_path=str(cfg), category='portrait',
                                min_comparisons=30)
        assert cfg.read_text() != before

    def test_gate_is_deterministic(self, tmp_path):
        db = build_db(str(tmp_path / 'det.db'), n_decisive=80, n_ties=10, seed=3)
        first = _optimizer(db).optimize_weights_with_cv(category='portrait', min_comparisons=30)
        second = _optimizer(db).optimize_weights_with_cv(category='portrait', min_comparisons=30)
        for key in ('p_value', 'passed', 'discordant_new', 'discordant_old', 'heldout_gain_pp',
                    'new_weights'):
            assert first[key] == second[key]


class TestGateReturnsSuggestChanges:
    """(11) ``?use_cv=true`` reads ``suggest_changes`` off the CV return."""

    @pytest.mark.parametrize('agree, expected', [(None, False), (0.85, True)])
    def test_suggest_changes_equals_passed(self, tmp_path, agree, expected):
        db = build_db(str(tmp_path / 'cv.db'), n_decisive=150, seed=7, agree=agree)
        result = _optimizer(db).optimize_weights_with_cv(category='portrait', min_comparisons=30)
        assert 'error' not in result
        assert result['suggest_changes'] is result['passed']
        assert result['passed'] is expected
        assert result['discordant_new'] >= 0 and result['discordant_old'] >= 0
        assert 0.0 <= result['p_value'] <= 1.0


class TestDirectPathRunsTheGateOnlyWhenNeeded:
    """(12) The viewer GET runs CV only when the in-sample fit clears the bar."""

    @staticmethod
    def _spy(monkeypatch):
        calls = []
        real = WeightOptimizer.optimize_weights_with_cv

        def spy(self, *args, **kwargs):
            calls.append(kwargs)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(WeightOptimizer, 'optimize_weights_with_cv', spy)
        return calls

    def test_in_sample_miss_skips_cv(self, tmp_path, monkeypatch):
        # Only aesthetic varies and it decides every vote: the current weights
        # already rank every pair right, so the in-sample gain is 0 pp.
        db = build_db(str(tmp_path / 'perfect.db'), n_decisive=60, seed=5, agree=1.0,
                      only_aesthetic=True)
        calls = self._spy(monkeypatch)
        result = _optimizer(db).optimize_weights_direct(category='portrait', min_comparisons=30)
        assert result['improvement'] < 2.0
        assert calls == []
        assert result['suggest_changes'] is False

    def test_in_sample_pass_runs_cv_exactly_once(self, tmp_path, monkeypatch):
        db = build_db(str(tmp_path / 'signal.db'), n_decisive=150, seed=7, agree=0.85)
        calls = self._spy(monkeypatch)
        result = _optimizer(db).optimize_weights_direct(category='portrait', min_comparisons=30)
        assert result['improvement'] >= 2.0
        assert len(calls) == 1
        assert result['suggest_changes'] is True


class TestDecisiveGuards:
    """(7) Ties carry no ranking evidence, so they do not count toward the minimum."""

    def test_ties_do_not_satisfy_min_comparisons(self, tmp_path):
        db = build_db(str(tmp_path / 'ties.db'), n_decisive=10, n_ties=40, seed=2, agree=0.9)
        opt = _optimizer(db)
        assert 'error' in opt.optimize_weights_direct(category='portrait', min_comparisons=30)
        assert 'error' in opt.optimize_weights_with_cv(category='portrait', min_comparisons=30)
        assert 'error' in opt.compute_weight_confidence(
            category='portrait', min_comparisons=30, n_bootstrap=10)


class TestPhotoGroupedFolds:
    """(8) No photo is on both sides of any split."""

    @staticmethod
    def _assert_disjoint(photo_a, photo_b, fold_a, fold_b, n_folds):
        for f in range(n_folds):
            test = (fold_a == f) & (fold_b == f)
            train = (fold_a != f) & (fold_b != f)
            test_photos = set(photo_a[test]) | set(photo_b[test])
            train_photos = set(photo_a[train]) | set(photo_b[train])
            assert not (test_photos & train_photos), f'fold {f} leaks photos'

    def test_component_folds_keep_every_pair_testable(self):
        photo_a = np.array([f'a{i}' for i in range(40)])
        photo_b = np.array([f'b{i}' for i in range(40)])
        fold_a, fold_b, k = wo._photo_folds(photo_a, photo_b, 5)
        assert k == 5
        assert (fold_a == fold_b).all()  # a component never straddles folds
        self._assert_disjoint(photo_a, photo_b, fold_a, fold_b, k)

    def test_a_dense_graph_is_split_without_leaks(self):
        rng = np.random.default_rng(0)
        idx = np.array([rng.choice(30, size=2, replace=False) for _ in range(120)])
        photo_a = np.array([f'p{i}' for i in idx[:, 0]])
        photo_b = np.array([f'p{i}' for i in idx[:, 1]])
        fold_a, fold_b, k = wo._photo_folds(photo_a, photo_b, 5)
        assert k == 5
        assert (fold_a == fold_b).any()  # some pairs are held out
        self._assert_disjoint(photo_a, photo_b, fold_a, fold_b, k)

    def test_cv_splits_with_photo_grouped_folds(self, tmp_path, monkeypatch):
        seen = []
        real = wo._photo_folds

        def spy(photo_a, photo_b, n_folds):
            out = real(photo_a, photo_b, n_folds)
            seen.append((np.asarray(photo_a), np.asarray(photo_b), *out))
            return out

        monkeypatch.setattr(wo, '_photo_folds', spy)
        db = build_db(str(tmp_path / 'f.db'), n_decisive=60, seed=4, agree=0.8)
        result = _optimizer(db).optimize_weights_with_cv(category='portrait', min_comparisons=30)
        assert 'error' not in result
        assert len(seen) == 1
        self._assert_disjoint(*seen[0])
