"""Held-out split, pose-gate rule and wording fixes of optimization/model_eval.py (issue #185 items 6, 7, 12)."""

import importlib.util

import numpy as np
import pytest

from optimization import eval_common as ec
from optimization import model_eval as me


class TestSplitHalves:
    def test_disjoint_covering_and_deterministic(self):
        a, b = ec.split_halves(101, 42)
        assert not set(a.tolist()) & set(b.tolist())
        assert sorted(a.tolist() + b.tolist()) == list(range(101))
        a2, b2 = ec.split_halves(101, 42)
        assert a.tolist() == a2.tolist() and b.tolist() == b2.tolist()
        assert ec.split_halves(101, 43)[0].tolist() != a.tolist()
        assert abs(len(a) - len(b)) <= 1

    def test_does_not_touch_global_rng(self):
        np.random.seed(5)
        before = np.random.random()
        np.random.seed(5)
        ec.split_halves(50, 1)
        assert np.random.random() == before

    def test_stratified_keeps_label_share(self):
        labels = [1] * 21 + [0] * 79
        a, b = ec.split_halves(100, 7, strata=labels)
        pos_a = sum(labels[i] for i in a)
        pos_b = sum(labels[i] for i in b)
        assert abs(pos_a - pos_b) <= 1 and pos_a + pos_b == 21

    def test_groups_never_straddle(self):
        groups = [i // 5 for i in range(100)]
        a, b = ec.split_halves(100, 3, groups=groups)
        assert not {groups[i] for i in a} & {groups[i] for i in b}
        assert len(a) + len(b) == 100

    def test_person_split_has_no_shared_person(self):
        labels = np.repeat(np.arange(20), np.arange(1, 21) % 7 + 2)
        a, b = me.split_persons(labels, 11)
        assert not set(labels[a].tolist()) & set(labels[b].tolist())
        assert len(a) + len(b) == len(labels)
        assert me.split_persons(labels, 11)[0].tolist() == a.tolist()


class TestWeighting:
    def test_weighted_prf_known_value(self):
        # base rate 0.1, 10 pos / 10 neg in the sample -> each negative stands for 9
        p, r, f1 = me.weighted_prf(tp=10, fp=2, fn=0, n_pos=10, n_neg=10, base_rate=0.1)
        assert p == pytest.approx(10 / 28) and r == 1.0
        assert f1 == pytest.approx(2 * p / (p + 1))

    def test_no_base_rate_gives_none(self):
        assert me.weighted_prf(1, 1, 1, 5, 5, None) is None


def _planted(seed=42, n=80):
    """Scores where split A is separable at 0.6 and split B only rewards 0.8 (its negatives sit at 0.5-0.75)."""
    labels = np.array([1, 0] * (n // 2))
    a, b = ec.split_halves(n, seed, strata=labels.tolist())
    rng = np.random.default_rng(0)
    scores = np.zeros(n)
    for idx in a:
        scores[idx] = rng.uniform(0.6, 0.9) if labels[idx] else rng.uniform(0.1, 0.4)
    for idx in b:
        scores[idx] = rng.uniform(0.85, 0.95) if labels[idx] else rng.uniform(0.5, 0.75)
    return scores, labels, a, b


NO_POSE = lambda n: np.full(n, np.nan)  # noqa: E731


class TestSelectionNeverSeesB:
    def test_threshold_chosen_on_a_and_reported_on_b(self):
        scores, labels, a, b = _planted()
        held = me.blink_heldout(scores, labels, NO_POSE(len(labels)), 0.5, 42, me.blink_threshold_grid())
        chosen = held['selection']['chosen']['threshold']
        assert chosen == 0.6                      # ties on A go to the larger threshold; B would prefer 0.8
        assert held['selection']['chosen']['row']['f1'] == 1.0   # perfect on A ...
        assert held['report']['split'] == 'B' and held['report']['threshold'] == chosen
        assert held['report']['metric']['f1'] < 1.0               # ... but B's own number is reported

    def test_selection_is_invariant_to_b_scores(self):
        scores, labels, a, b = _planted()
        grid = me.blink_threshold_grid()
        base = me.blink_heldout(scores, labels, NO_POSE(len(labels)), 0.5, 42, grid)
        mutated = scores.copy()
        mutated[b] = np.random.default_rng(9).uniform(0, 1, len(b))
        other = me.blink_heldout(mutated, labels, NO_POSE(len(labels)), 0.5, 42, grid)
        assert other['selection'] == base['selection']
        assert other['report']['metric'] != base['report']['metric']

    def test_deterministic_per_seed(self):
        scores, labels, _, _ = _planted()
        args = (scores, labels, NO_POSE(len(labels)), 0.5, 42, me.blink_threshold_grid())
        assert me.blink_heldout(*args) == me.blink_heldout(*args)

    def test_bootstrap_ci_brackets_ordering(self):
        scores, labels, _, b = _planted()
        ci = me.blink_bootstrap_ci(scores[b], labels[b], 0.6, 0.5, 1, n_boot=200)
        assert ci['f1']['ci_low'] <= ci['f1']['ci_high'] and ci['n_boot'] == 200


class TestPoseGate:
    def test_gated_true_positive_is_a_lost_recall(self):
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        labels = np.array([1, 1, 0, 0])
        poses = np.array([50.0, 5.0, 5.0, 5.0])
        table = {r['cutoff']: r for r in me.pose_gate_table(scores, labels, poses, 0.55, 0.5)}
        assert table[None]['recall'] == 1.0 and table[None]['n_gated'] == 0
        assert table[45]['recall'] == 0.5 and table[45]['n_gated'] == 1 and table[45]['gated_share'] == 0.25
        assert table[45]['precision'] == 1.0

    def test_nan_pose_is_never_gated(self):
        table = me.pose_gate_table(np.array([0.9]), np.array([1]), np.array([np.nan]), 0.55, 0.5)
        assert all(r['n_gated'] == 0 for r in table)

    @staticmethod
    def _table(rows):
        return [dict(cutoff=c, gated_share=g, weighted_f1=f, f1=f) for c, g, f in rows]

    def test_rule_selects_on_a_and_confirms_on_b(self):
        a = self._table([(None, 0, .2), (45, .02, .3), (40, .04, .3), (35, .06, .25),
                         (30, .3, .9), (25, .4, .9), (20, .5, .9)])
        b = self._table([(None, 0, .2), (45, .02, .25), (40, .04, .31), (35, .06, .2),
                         (30, .3, .9), (25, .4, .9), (20, .5, .9)])
        d = me.choose_pose_cutoff(a, b)
        assert d['selected_on_a'] == 45                    # 30/25/20 gate > 15%; 45 vs 40 tie -> larger
        assert d['adopt'] is True and d['cutoff'] == 45    # B: .25 > .2
        b[1]['weighted_f1'] = b[1]['f1'] = 0.2             # no gain on B
        d = me.choose_pose_cutoff(a, b)
        assert d['adopt'] is False and d['cutoff'] is None and 'null result' in d['verdict']

    def test_none_wins_ties_and_b_must_stay_under_share(self):
        a = self._table([(None, 0, .3), (45, .01, .3), (40, .02, .1), (35, .03, .1),
                         (30, .04, .1), (25, .05, .1), (20, .06, .1)])
        assert me.choose_pose_cutoff(a, a)['selected_on_a'] is None
        a2 = self._table([(None, 0, .3), (45, .01, .5)] + [(c, .02, .1) for c in (40, 35, 30, 25, 20)])
        b2 = self._table([(None, 0, .3), (45, .20, .6)] + [(c, .02, .1) for c in (40, 35, 30, 25, 20)])
        assert me.choose_pose_cutoff(a2, b2)['adopt'] is False

    def test_rule_text_is_in_every_report(self):
        scores, labels, _, _ = _planted()
        poses = np.abs(np.random.default_rng(1).normal(20, 15, len(labels)))
        held = me.blink_heldout(scores, labels, poses, 0.5, 42, me.blink_threshold_grid())
        assert held['rule_pose'] == me.PRE_REGISTERED_POSE_RULE
        assert held['pose_gate']['decision']['verdict']


class TestCentroidHeldOut:
    @staticmethod
    def _emb(seed=0, persons=24, per=6, dim=12, noise=0.5):
        rng = np.random.default_rng(seed)
        centres = rng.normal(size=(persons, dim))
        e = np.repeat(centres, per, 0) + rng.normal(scale=noise, size=(persons * per, dim))
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return e.astype(np.float32), np.repeat(np.arange(persons), per)

    def test_report_is_scored_on_b_persons_only(self):
        e, lab = self._emb()
        arc, _ = self._emb(noise=0.3)
        sel, held = me.centroid_heldout(e, arc, lab, 42, n_boot=50)
        a, b = me.split_persons(lab, 42)
        assert sel['split'] == 'A' and held['split'] == 'B'
        assert sel['n_persons'] + held['n_persons'] == 24
        thr = held['merge_threshold']
        assert thr is not None
        expected = me.centroid_sweep(e[b], lab[b], [thr['threshold']], 42)['grid'][0]
        assert thr['row'] == expected
        full = me.centroid_sweep(e, lab, [thr['threshold']], 42)['grid'][0]
        assert thr['row'] != full
        assert 'same_recall' in thr['ci'] and thr['ci']['n_boot'] == 50

    def test_deterministic_per_seed(self):
        e, lab = self._emb()
        arc, _ = self._emb(noise=0.3)
        assert me.centroid_heldout(e, arc, lab, 3, n_boot=20) == me.centroid_heldout(e, arc, lab, 3, n_boot=20)

    def test_selection_ignores_b_embeddings(self):
        e, lab = self._emb()
        arc, _ = self._emb(noise=0.3)
        sel1, _ = me.centroid_heldout(e, arc, lab, 42, n_boot=10)
        _, b = me.split_persons(lab, 42)
        e2 = e.copy()
        e2[b] = np.random.default_rng(4).normal(size=e2[b].shape)
        e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
        arc2 = arc.copy()
        arc2[b] = e2[b]
        sel2, _ = me.centroid_heldout(e2, arc2, lab, 42, n_boot=10)
        assert sel1 == sel2


def test_ofiq_message_is_accurate(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a, **k: None)
    msg = me._ofiq_unavailable_message()
    assert 'pip install python-ofiq' in msg and 'not on PyPI' not in msg
    assert 'Linux x86-64' in msg and 'WSL2' in msg
