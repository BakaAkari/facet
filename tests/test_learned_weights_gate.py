"""``GET /api/comparison/learned_weights`` honours the optimizer's gate.

The Suggestions tab offers Apply only when ``suggest_changes`` is true, so the
route must answer with the held-out McNemar decision rather than the in-sample
gain, count only decisive comparisons of the requested category before it
fits, build the optimizer on the server's config file, and report the
weights it fitted under the config keys the client saves.
"""

import os
import sqlite3

import numpy as np
import pytest

from api.config import get_comparison_mode_settings, server_config_path
from tests.test_weight_optimizer_gate import NOISE_COLUMNS as METRICS

PREFIX = '/b4-learned-gate/'
CATEGORY = 'portrait'


@pytest.fixture()
def seed_votes(seed_photos_prefix):
    """Seed disjoint photo pairs plus their votes into the shared session DB."""
    conn = sqlite3.connect(os.environ['DB_PATH'])
    pre = conn.execute(
        "SELECT COUNT(*) FROM comparisons WHERE category = ?", (CATEGORY,)).fetchone()[0]
    assert pre == 0, 'another test left comparisons in the shared session DB'
    runs_before = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM weight_optimization_runs").fetchone()[0]

    def seed(*, n_decisive, n_ties=0, agree=None, seed=0):
        rng = np.random.default_rng(seed)
        n_pairs = n_decisive + n_ties
        rows = []
        for i in range(2 * n_pairs):
            row = {'path': f'{PREFIX}p{i:04d}.jpg', 'filename': f'p{i:04d}.jpg'}
            row.update({m: float(rng.uniform(1, 9)) for m in METRICS})
            rows.append(row)
        seed_photos_prefix(PREFIX, rows)
        for i in range(n_pairs):
            a, b = rows[2 * i], rows[2 * i + 1]
            if i >= n_decisive:
                winner = 'tie'
            elif agree is None:
                winner = 'a' if rng.random() < 0.5 else 'b'
            else:
                better = 'a' if a['aesthetic'] > b['aesthetic'] else 'b'
                winner = better if rng.random() < agree else ('b' if better == 'a' else 'a')
            conn.execute(
                "INSERT INTO comparisons (photo_a_path, photo_b_path, winner, category, source) "
                "VALUES (?, ?, ?, ?, 'vote')", (a['path'], b['path'], winner, CATEGORY))
        conn.commit()

    try:
        yield seed
    finally:
        conn.execute("DELETE FROM comparisons WHERE photo_a_path LIKE ?", (PREFIX + '%',))
        conn.execute("DELETE FROM weight_optimization_runs WHERE id > ?", (runs_before,))
        conn.commit()
        conn.close()


def _min_required():
    return get_comparison_mode_settings()['min_comparisons_for_optimization']


class TestSuggestChangesIsTheHeldOutGate:
    """(6) The viewer's Apply follows the held-out decision, not the in-sample gain."""

    def test_random_winners_do_not_suggest_changes(self, edition_client, seed_votes):
        seed_votes(n_decisive=max(150, _min_required()), seed=7)
        body = edition_client.get('/api/comparison/learned_weights',
                                  params={'category': CATEGORY}).json()
        assert body['available'] is True
        assert body['suggest_changes'] is False

    def test_planted_signal_suggests_changes(self, edition_client, seed_votes):
        seed_votes(n_decisive=max(150, _min_required()), agree=0.85, seed=7)
        body = edition_client.get('/api/comparison/learned_weights',
                                  params={'category': CATEGORY}).json()
        assert body['available'] is True
        assert body['suggest_changes'] is True

    def test_use_cv_reports_the_same_decision(self, edition_client, seed_votes):
        seed_votes(n_decisive=max(150, _min_required()), agree=0.85, seed=7)
        body = edition_client.get('/api/comparison/learned_weights',
                                  params={'category': CATEGORY, 'use_cv': 'true'}).json()
        assert body['suggest_changes'] is True


class TestPreCheckCountsDecisiveVotesOfTheCategory:
    def test_ties_do_not_count(self, edition_client, seed_votes):
        need = _min_required()
        seed_votes(n_decisive=need - 1, n_ties=need, agree=0.9, seed=3)
        body = edition_client.get('/api/comparison/learned_weights',
                                  params={'category': CATEGORY}).json()
        assert body['available'] is False
        assert body['comparisons'] == need - 1

    def test_other_categories_do_not_count(self, edition_client, seed_votes):
        seed_votes(n_decisive=_min_required(), agree=0.9, seed=3)
        body = edition_client.get('/api/comparison/learned_weights',
                                  params={'category': 'landscape'}).json()
        assert body['available'] is False
        assert body['comparisons'] == 0


class TestSuggestedWeightsUseConfigKeys:
    def test_every_fitted_component_reaches_the_response(self, edition_client, seed_votes):
        # The optimizer's weights are keyed by config metric names; looking them up
        # under DB column names (comp_score, liqe_score, ...) reported 0 for them,
        # and an Apply would then have written those zeros into the config.
        from optimization import WeightOptimizer
        from config import ScoringConfig
        seed_votes(n_decisive=max(150, _min_required()), agree=0.85, seed=7)
        body = edition_client.get('/api/comparison/learned_weights',
                                  params={'category': CATEGORY}).json()
        current = ScoringConfig(server_config_path(), validate=False).get_weights(CATEGORY)
        for key in ('composition', 'liqe', 'color', 'contrast', 'exposure'):
            if current.get(key, 0) * 100 >= 1:
                assert body['current_weights'][f'{key}_percent'] >= 1, key
        assert sum(body['suggested_weights'].values()) == pytest.approx(100, abs=len(
            WeightOptimizer.SCORE_COMPONENTS))


class TestRouterUsesTheServerConfig:
    """(9) The optimizer scores with the file the server scores and authenticates with."""

    @pytest.fixture()
    def recorded(self, monkeypatch):
        import optimization
        built = []

        class Recorder:
            def __init__(self, db_path, config_path=None):
                built.append(config_path)

            def optimize_weights_direct(self, **kwargs):
                return {'error': 'recorded'}

            def compute_weight_confidence(self, **kwargs):
                return {'error': 'recorded'}

        monkeypatch.setattr(optimization, 'WeightOptimizer', Recorder)
        return built

    def test_learned_weights(self, edition_client, seed_votes, recorded):
        seed_votes(n_decisive=_min_required(), agree=0.9, seed=3)
        edition_client.get('/api/comparison/learned_weights', params={'category': CATEGORY})
        assert recorded == [server_config_path()]

    def test_confidence(self, edition_client, recorded):
        edition_client.get('/api/comparison/confidence', params={'category': CATEGORY})
        assert recorded == [server_config_path()]
