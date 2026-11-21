"""The per-turn credit (vagen/turn_credit.py, its estimator, the nominal filter, the rl-config switch).  Synthetic data
and temporary directories only (no real paths)."""

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vagen import turn_credit as TC

HAVE_TRAINING_ENV = all(importlib.util.find_spec(m) is not None for m in ("ray", "omegaconf", "tensordict"))
K, G = 10.0 / 3.0, 0.99


def rec(phi0, turns, d0=None, ds=None):
    """turns = [(start, end, reward, phi_after, success)]; ds the table distances after each turn"""
    ds = ds or [None] * len(turns)
    return {"phi0": phi0, "d0": d0,
            "turns": [{"span": [s, e], "reward": r, "phi_after": p, "d_after": d, "success": ok, "valid": True}
                      for (s, e, r, p, ok), d in zip(turns, ds)]}


# the 11-01 plan's TC-mix (the reward in the turn's value, turn-index baseline, clip and protection, raw episode term)
PLAN_1101 = dict(TC.DEFAULTS, turn_reward=True, baseline="turn_index", turn_weight=1.0, episode_norm="none",
                 neg_clip=3.0, min_positive_mass=0.4)


class CreditTest(unittest.TestCase):
    def test_credit_formula_and_success_turn(self):
        r = rec(0.2, [(0, 3, -0.15, 0.4, False), (5, 8, 5.0, 0.4, True)])  # the env's score is stale on success
        c = TC.turn_credits(r, K, G)
        self.assertAlmostEqual(c[0], -0.15 + K * (G * 0.4 - 0.2))
        self.assertAlmostEqual(c[1], 5.0 + K * (G * 1.0 - 0.4))
        # phase 2's turn value: the shaping alone
        v = TC.turn_values(r, K, G)
        self.assertAlmostEqual(v[0], K * (G * 0.4 - 0.2))
        self.assertAlmostEqual(v[1], K * (G * 1.0 - 0.4))

    def test_stall_penalty(self):
        # distances 6 -> 5 -> 5 -> 5 -> 4: turn 2 (index) is the second turn in a row with no change
        r = rec(0.0, [(0, 1, 0.0, 0.1, False)] * 4, d0=6, ds=[5, 5, 5, 4])
        self.assertEqual(TC.stalled(r), [False, False, True, False])
        base = TC.turn_values(r, K, G)
        pen = TC.turn_values(r, K, G, stall_penalty=0.05)
        self.assertEqual([round(a - b, 6) for a, b in zip(base, pen)], [0.0, 0.0, 0.05, 0.0])
        # an unknown distance never counts as a stall
        self.assertEqual(TC.stalled(rec(0.0, [(0, 1, 0.0, 0.0, False)] * 3, d0=None, ds=[5, 5, 5])), [False, False, True])
        self.assertEqual(TC.stalled(rec(0.0, [(0, 1, 0.0, 0.0, False)] * 3, d0=5, ds=[None, 5, 5])), [False, False, False])

    def test_baselines(self):
        b = TC.baselines([[1.0, 2.0], [3.0], [5.0, 6.0, 7.0]], ["a", "a", "b"], "turn_index")
        self.assertEqual(b, [[2.0, 2.0], [2.0], [5.0, 6.0, 7.0]])
        self.assertEqual(TC.baselines([[1.0], [2.0]], ["a", "a"], "none"), [[0.0], [0.0]])
        self.assertEqual(TC.baselines([[1.0, 2.0], [6.0]], ["a", "b"], "batch_mean"), [[3.0, 3.0], [3.0]])

    def test_phase2_default_is_the_standardised_sum(self):
        # two tasks: a mixed group and a group of equal returns (its episode term is 0 and it stays in the batch)
        records = [rec(0.2, [(0, 2, 0.0, 0.5, False), (2, 4, 5.0, 0.5, True)]),
                   rec(0.2, [(0, 2, 0.0, 0.1, False), (2, 4, 0.0, 0.1, False)]),
                   rec(0.3, [(0, 2, 0.0, 0.6, False)]), rec(0.3, [(0, 2, 0.0, 0.2, False)])]
        uids, returns = ["a", "a", "b", "b"], [5.0, 0.0, 0.0, 0.0]
        cfg = TC.settings({})
        mask = torch.ones(4, 4)
        adv, m = TC.token_advantages(records, uids, mask, returns, cfg)
        vals = [TC.turn_values(r, K, G) for r in records]
        flat = np.array([x for v in vals for x in v])
        z = (flat - flat.mean()) / (flat.std() + 1e-6)
        raw = np.array(TC.episode_advantages(returns, uids))
        e = (raw - raw.mean()) / (raw.std() + 1e-6)
        self.assertAlmostEqual(float(adv[0, 0]), e[0] + 0.5 * z[0], places=5)
        self.assertAlmostEqual(float(adv[0, 3]), e[0] + 0.5 * z[1], places=5)
        self.assertAlmostEqual(float(adv[2, 1]), 0.5 * z[4], places=5)  # tied group: the turn term alone
        self.assertEqual(float(adv[2, 3]), 0.5 * 0.0 + 0.0)  # past the episode's last span: the episode term, 0 here
        self.assertEqual((m["turn_credit/groups"], m["turn_credit/tied_groups"], m["turn_credit/turns"]), (2.0, 1.0, 6.0))
        self.assertAlmostEqual(m["turn_credit/turn_only_share"], 2 / 6)
        self.assertGreaterEqual(m["turn_credit/sign_flip_share"], 0.0)
        self.assertTrue(0.0 < m["turn_credit/turn_magnitude_share"] < 1.0)
        # all returns equal: the episode term is 0 everywhere, the turn term still moves the update
        adv, m = TC.token_advantages(records, uids, mask, [0.0] * 4, cfg)
        self.assertEqual(m["turn_credit/episode_zero_share"], 1.0)
        self.assertGreater(float(adv.abs().sum()), 0.0)

    def test_episode_advantages_match_verl_grpo(self):
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        returns = [5.0, 0.0, -0.1, 5.2, 1.0, 0.0]
        uids = np.array(["a", "a", "a", "b", "b", "c"], dtype=object)
        rewards = torch.zeros(6, 4)
        rewards[:, -1] = torch.tensor(returns)
        ref, _ = compute_grpo_outcome_advantage(rewards, torch.ones(6, 4), uids)
        mine = TC.episode_advantages(returns, list(uids))
        for i in range(5):  # verl gives a lone episode (uid c) mean 0, std 1: its raw return; this estimator 0
            self.assertAlmostEqual(mine[i], float(ref[i, 0]), places=5)
        self.assertEqual(mine[5], 0.0)

    def test_clip_and_positive_mass(self):
        records = [rec(0.0, [(0, 1, -1.0, 0.0, False)] * 9 + [(0, 1, 0.1, 0.0, False)])]
        cfg = dict(PLAN_1101, baseline="none")
        z, m = TC.turn_advantages(records, ["a"], cfg)
        flat = z[0]
        pos = sum(a for a in flat if a > 0)
        neg = -sum(a for a in flat if a < 0)
        self.assertLess(m["turn_credit/neg_scale"], 1.0)
        self.assertAlmostEqual(pos, 0.4 * neg, places=6)
        self.assertGreaterEqual(min(flat), -3.0)
        # a credit far below the rest is clipped at -3 standard deviations
        records = [rec(0.0, [(0, 1, 1.0, 0.0, False)] * 30 + [(0, 1, -100.0, 0.0, False)])]
        z, m = TC.turn_advantages(records, ["a"], dict(cfg, min_positive_mass=0.0))
        self.assertAlmostEqual(min(z[0]), -3.0)
        self.assertGreater(m["turn_credit/clipped_share"], 0.0)

    def test_token_advantages_spans_mask_and_episode_term(self):
        mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 1, 1, 1]], dtype=torch.float32)
        records = [rec(0.1, [(0, 2, 0.0, 0.3, False), (4, 7, 5.0, 0.3, True)]),
                   rec(0.1, [(0, 3, -0.1, 0.1, False), (5, 9, 0.0, 0.2, False)])]  # the last span runs past the end
        cfg = dict(PLAN_1101, baseline="none", min_positive_mass=0.0, neg_clip=0.0)
        returns = [5.0, -0.1]
        adv, m = TC.token_advantages(records, ["a", "a"], mask, returns, cfg)
        z, _ = TC.turn_advantages(records, ["a", "a"], cfg, [[2, 3], [3, 3]])
        ep = TC.episode_advantages(returns, ["a", "a"])
        self.assertAlmostEqual(float(adv[0, 0]), z[0][0] + ep[0], places=5)
        self.assertAlmostEqual(float(adv[0, 5]), z[0][1] + ep[0], places=5)
        self.assertAlmostEqual(float(adv[1, 7]), z[1][1] + ep[1], places=5)
        self.assertEqual(float(adv[0, 2]), 0.0)  # observation tokens keep 0
        self.assertEqual(float(adv[1, 3]), 0.0)
        self.assertAlmostEqual(m["turn_credit/span_token_share"], 1.0)
        pure, _ = TC.token_advantages(records, ["a", "a"], mask, returns, dict(cfg, episode_weight=0.0))
        self.assertAlmostEqual(float(pure[0, 0]), z[0][0], places=5)

    def test_records_from_the_agent_loop(self):
        info = {"current_potential_score": 0.42, "current_region_metrics": {"optimal_distance": 4}}
        t = TC.turn_record((3, 9), -0.15, info, False, True)
        self.assertEqual(t, {"span": [3, 9], "reward": -0.15, "phi_after": 0.42, "d_after": 4, "success": False,
                             "valid": True})
        self.assertEqual(TC.turn_record(None, 0.0, {}, False, False)["span"], [0, 0])
        self.assertIsNone(TC.turn_record(None, 0.0, {}, False, False)["d_after"])
        # the turn-end pose's table phi wins over a stale current_potential_score (the step budget's last turn)
        stale = {"current_potential_score": 0.5, "current_region_metrics": {"optimal_distance": 6, "optimal_progress": 0.4}}
        self.assertEqual(TC.turn_record(None, 0.0, stale, False, True)["phi_after"], 0.4)
        # experiment J (丙): an env with V4's progress has no table score; phi is its turn-end shaping_score, not the
        # stale potential, and the table distance is unknown
        v4 = {"current_potential_score": 0.2, "current_region_metrics": {"progress_version": "grouped_gated",
                                                                         "shaping_score": 0.31, "score": 0.9}}
        t4 = TC.turn_record((0, 4), 0.0, v4, False, True)
        self.assertEqual((t4["phi_after"], t4["d_after"]), (0.31, None))
        # unchanged: a metric of another version (no progress_version key) still falls back to the potential, and the
        # table's score still wins when both are there
        prod = {"current_potential_score": 0.2, "current_region_metrics": {"shaping_score": 0.31}}
        self.assertEqual(TC.turn_record(None, 0.0, prod, False, True)["phi_after"], 0.2)
        both = {"current_potential_score": 0.2, "current_region_metrics": {"progress_version": "grouped_gated",
                                                                           "shaping_score": 0.31, "optimal_progress": 0.6}}
        self.assertEqual(TC.turn_record(None, 0.0, both, False, True)["phi_after"], 0.6)
        # experiment K: the env's step count is kept when the info has it (and only then)
        self.assertEqual(TC.turn_record(None, 0.0, dict(info, env_step=5), False, True)["steps"], 5)
        self.assertNotIn("steps", t)
        e = TC.episode_record(0.2, [t, dict(t, span=[10, 20])], response_length=12, d0=5)
        self.assertEqual([x["span"] for x in e["turns"]], [[3, 9], [10, 12]])
        self.assertEqual((e["phi0"], e["d0"]), (0.2, 5))
        self.assertEqual(TC.reset_distance({"initial_region_metrics": {"optimal_distance": 7}}), 7)
        self.assertIsNone(TC.reset_distance({"initial_region_metrics": {"optimal_distance": None}}))
        self.assertIsNone(TC.reset_distance(None))

    def test_success_table_phi(self):
        # experiment K (戊): phi = Phi(d, b) from the success table; 1 on success, 0 for an unknown d
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            phi = [[1, 1, 1, 1, 1], [0, 0.3, 0.5, 0.7, 0.8], [0, 0, 0.2, 0.4, 0.6], [0, 0, 0, 0.1, 0.3],
                   [0, 0, 0, 0, 0.1]]
            p.write_text(json.dumps({"version": "success_table_v1", "budget": 4, "phi": phi}))
            tab = TC.load_success_table(str(p))
            self.assertEqual(TC.success_phi(tab, None, 3), 0.0)
            self.assertEqual(TC.success_phi(tab, 2, 9), 0.6)  # b clipped to the budget
            self.assertEqual(TC.success_phi(tab, 9, 4), 0.0)  # beyond the table
            self.assertEqual(TC.success_phi(tab, 3, 2, success=True), 1.0)
            r = rec(0.9, [(0, 2, 0.0, 0.7, False), (2, 4, 0.0, 0.8, False), (4, 6, 0.0, 0.9, True)], d0=3,
                    ds=[2, None, 0])
            r["turns"][0]["steps"] = 1  # b 3: Phi(2, 3) = 0.4
            r["turns"][1]["steps"] = 3  # b 1, d unknown: 0
            t = TC.table_record(r, tab)  # third turn: no step count (previous b), success: 1
            self.assertEqual(t["phi0"], 0.3)  # Phi(3, 4)
            self.assertEqual([x["phi_after"] for x in t["turns"]], [0.4, 0.0, 1.0])
            self.assertEqual(r["phi0"], 0.9)  # the input record is not changed
            cfg = TC.settings({"turn_credit": {"phi_version": "success_table", "phi_table": str(p), "baseline": "none",
                                               "episode_weight": 0}})
            z, m = TC.turn_advantages([r], ["u"], cfg)
            self.assertEqual(m["turn_credit/steps_missing_share"], 0.5)  # of the turns with a known d, one has none
            self.assertEqual(m["turn_credit/phi0_mean"], 0.3)
            vals = [K * (G * 0.4 - 0.3), K * (G * 0.0 - 0.4), K * (G * 1.0 - 0.0)]
            self.assertTrue(np.allclose(TC.turn_values(t, K, G), vals))
            self.assertTrue(np.allclose(z[0], np.asarray(vals) / (np.std(vals) + cfg["eps"])))
            z0, _ = TC.turn_advantages([r], ["u"], dict(cfg, phi_version="optimal", phi_table=None))
            self.assertFalse(np.allclose(z0[0], z[0]))  # the default reads the record's own phi
            bad = Path(tmp) / "b.json"
            bad.write_text(json.dumps({"version": "x", "budget": 4, "phi": phi}))
            with self.assertRaises(ValueError):
                TC.load_success_table(str(bad))
            # experiment M: v2p is read like v1; the unprojected b12 refit is not
            v2p = Path(tmp) / "v2p.json"
            v2p.write_text(json.dumps({"version": "success_table_v2p", "budget": 4, "phi": phi}))
            self.assertTrue(np.allclose(TC.load_success_table(str(v2p))["phi"], np.asarray(phi, float)))
            b12 = Path(tmp) / "b12.json"
            b12.write_text(json.dumps({"version": "success_table_v2_b12", "budget": 4, "phi": phi}))
            with self.assertRaises(ValueError):
                TC.load_success_table(str(b12))
            bad2 = Path(tmp) / "b2.json"
            bad2.write_text(json.dumps({"version": "success_table_v1", "budget": 3, "phi": phi}))  # 5 columns, budget 3
            with self.assertRaises(ValueError):
                TC.load_success_table(str(bad2))

    def test_unreadable_as_wrong(self):
        # experiment L (庚): an unreadable turn (valid false) gets Phi(d + 1, b - 1) of the state before it; 0 off the table
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            phi = [[1, 1, 1, 1, 1], [0, 0.3, 0.5, 0.7, 0.8], [0, 0, 0.2, 0.4, 0.6], [0, 0, 0, 0.1, 0.3],
                   [0, 0, 0, 0, 0.1]]
            p.write_text(json.dumps({"version": "success_table_v1", "budget": 4, "phi": phi}))
            tab = TC.load_success_table(str(p))
            self.assertEqual(TC.wrong_step_phi(tab, 2, 4), 0.1)  # Phi(3, 3)
            self.assertEqual(TC.wrong_step_phi(tab, 2, 9), 0.1)  # b capped at the budget first
            self.assertEqual(TC.wrong_step_phi(tab, 4, 4), 0.0)  # d + 1 beyond the table
            self.assertEqual(TC.wrong_step_phi(tab, 1, 0), 0.0)  # b - 1 < 0
            self.assertEqual(TC.wrong_step_phi(tab, None, 4), 0.0)  # unknown d
            # d0 2, b 4: unreadable (d, b unchanged) / a step to d 1 (b 3) / unreadable / to d 1 at b 0 / unreadable
            r = rec(0.0, [(0, 2, -0.1, 0.0, False), (2, 4, 0.0, 0.0, False), (4, 6, -0.1, 0.0, False),
                          (6, 8, 0.0, 0.0, False), (8, 9, -0.1, 0.0, False)], d0=2, ds=[2, 1, 1, 1, 1])
            for t, steps, ok in zip(r["turns"], [0, 1, 1, 4, 4], [False, True, False, True, False]):
                t["steps"], t["valid"] = steps, ok
            off = TC.table_record(r, tab)
            self.assertEqual([x["phi_after"] for x in off["turns"]], [0.6, 0.7, 0.7, 0.0, 0.0])
            self.assertFalse(any("wrong_step" in x for x in off["turns"]))
            on = TC.table_record(r, tab, True)
            self.assertEqual(on["phi0"], 0.6)
            # Phi(3, 3), Phi(1, 3), Phi(2, 2), Phi(1, 0), b - 1 < 0
            self.assertEqual([x["phi_after"] for x in on["turns"]], [0.1, 0.7, 0.2, 0.0, 0.0])
            self.assertEqual([bool(x.get("wrong_step")) for x in on["turns"]], [True, False, True, False, True])
            self.assertTrue(r["turns"][0]["valid"] is False and "wrong_step" not in r["turns"][0])  # input unchanged
            # reading B (the user's 11-14 night): the next turn starts from the real Phi(2, 4) = 0.6, not from 0.1
            self.assertEqual([x.get("phi_from") for x in on["turns"]], [None, 0.6, None, 0.7, None])  # Phi(2, 4), Phi(1, 3)
            self.assertTrue(np.allclose(TC.turn_values(on, K, G)[:2], [K * (G * 0.1 - 0.6), K * (G * 0.7 - 0.6)]))
            # a success turn is never a wrong step; a record without "valid" counts as readable
            s_rec = rec(0.0, [(0, 2, 1.0, 0.0, True)], d0=1, ds=[0])
            s_rec["turns"][0].update(steps=1, valid=False)
            self.assertEqual(TC.table_record(s_rec, tab, True)["turns"][0]["phi_after"], 1.0)
            n_rec = rec(0.0, [(0, 2, 0.0, 0.0, False)], d0=2, ds=[2])
            del n_rec["turns"][0]["valid"]
            n_rec["turns"][0]["steps"] = 0
            self.assertEqual(TC.table_record(n_rec, tab, True)["turns"][0]["phi_after"], 0.6)
            cfg = TC.settings({"turn_credit": {"phi_version": "success_table", "phi_table": str(p), "baseline": "none",
                                               "episode_weight": 0, "unreadable_as_wrong": True}})
            z, m = TC.turn_advantages([r], ["u"], cfg)
            self.assertEqual(m["turn_credit/unreadable_wrong_share"], 0.6)
            vals = TC.turn_values(on, K, G)
            self.assertTrue(np.allclose(z[0], np.asarray(vals) / (np.std(vals) + cfg["eps"])))
            z_off, m_off = TC.turn_advantages([r], ["u"], dict(cfg, unreadable_as_wrong=False))
            self.assertNotIn("turn_credit/unreadable_wrong_share", m_off)
            self.assertFalse(np.allclose(z_off[0], z[0]))

    def test_unreadable_as_wrong_reading_b(self):
        # the user's 11-14 night 一.2: two unreadable turns in a row get the same penalty; a wrong or a right step
        # after an unreadable turn gets the value it gets in 戊 (the switch off)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            phi = [[1, 1, 1, 1, 1], [0, 0.3, 0.5, 0.7, 0.8], [0, 0, 0.2, 0.4, 0.6], [0, 0, 0, 0.1, 0.3],
                   [0, 0, 0, 0, 0.1]]
            p.write_text(json.dumps({"version": "success_table_v1", "budget": 4, "phi": phi}))
            tab = TC.load_success_table(str(p))

            def episode(steps, ds, valid, d0=2):
                r = rec(0.0, [(2 * i, 2 * i + 2, 0.0, 0.0, False) for i in range(len(ds))], d0=d0, ds=ds)
                for x, k, v in zip(r["turns"], steps, valid):
                    x["steps"], x["valid"] = k, v
                return r

            cases = {  # (steps, d after, valid): from (2, 4)
                "two unreadable, then right": ([0, 0, 1], [2, 2, 1], [False, False, True]),
                "unreadable, then wrong": ([0, 1], [2, 3], [False, True]),
                "unreadable, then right": ([0, 1], [2, 1], [False, True]),
                "three unreadable": ([0, 0, 0], [2, 2, 2], [False, False, False]),
                "right, unreadable, right": ([1, 1, 2], [1, 1, 0 + 1], [True, False, True]),
            }
            pen = K * (G * 0.1 - 0.6)  # Phi(3, 3) from Phi(2, 4)
            for name, (steps, ds, valid) in cases.items():
                r = episode(steps, ds, valid)
                on = TC.turn_values(TC.table_record(r, tab, True), K, G)
                off = TC.turn_values(TC.table_record(r, tab), K, G)
                for i, ok in enumerate(valid):
                    if ok and i > 0 and not valid[i - 1]:  # the turn after an unreadable one: as in 戊
                        self.assertAlmostEqual(on[i], off[i], msg=name)
                    if ok and (i == 0 or valid[i - 1]):  # an ordinary readable turn: as in 戊
                        self.assertAlmostEqual(on[i], off[i], msg=name)
                if name in ("two unreadable, then right", "three unreadable", "unreadable, then wrong",
                            "unreadable, then right"):
                    first = valid.index(False)
                    self.assertAlmostEqual(on[first], pen, msg=name)
                    for i in range(first, len(valid)):
                        if not valid[i]:
                            self.assertAlmostEqual(on[i], on[first], msg=name)  # every unreadable turn: the same
            # the explicit numbers of the two follow-ups
            w = TC.turn_values(TC.table_record(episode([0, 1], [2, 3], [False, True]), tab, True), K, G)
            self.assertTrue(np.allclose(w, [pen, K * (G * 0.1 - 0.6)]))  # the wrong step: Phi(3, 3) from Phi(2, 4)
            g = TC.turn_values(TC.table_record(episode([0, 1], [2, 1], [False, True]), tab, True), K, G)
            self.assertTrue(np.allclose(g, [pen, K * (G * 0.7 - 0.6)]))  # the right step: Phi(1, 3) from Phi(2, 4)
            # unknown d after an unreadable turn: the next turn starts from 0 (the table's rule), the turn itself 0 - phi
            u_ = TC.table_record(episode([0, 1], [None, 1], [False, True], d0=None), tab, True)
            self.assertEqual([x.get("phi_from") for x in u_["turns"]], [None, 0.0])

    def test_unreadable_floor(self):
        # experiment N (辛, the user's 2026-11-16 一.1): an unreadable turn's value is min(v_B, -0.3); only those turns
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            phi = [[1, 1, 1, 1, 1], [0, 0.3, 0.5, 0.7, 0.8], [0, 0, 0.2, 0.4, 0.6], [0, 0, 0, 0.1, 0.3],
                   [0, 0, 0, 0, 0.1]]
            p.write_text(json.dumps({"version": "success_table_v1", "budget": 4, "phi": phi}))
            tab = TC.load_success_table(str(p))

            def episode(steps, ds, valid, d0):
                r = rec(0.0, [(2 * i, 2 * i + 2, 0.0, 0.0, False) for i in range(len(ds))], d0=d0, ds=ds)
                for x, k, v in zip(r["turns"], steps, valid):
                    x["steps"], x["valid"] = k, v
                return r

            # from (3, 4): three steps without getting closer (b 1, the dead cell d 3 > b 1), unreadable there
            # (Phi(4, 0) = 0, v_B = 0 -> -0.3), then a step to d 2 at b 0
            dead = TC.table_record(episode([3, 3, 4], [3, 3, 2], [True, False, True], 3), tab, True)
            off = TC.turn_values(dead, K, G)
            on = TC.turn_values(dead, K, G, unreadable_floor=-0.3)
            self.assertTrue(np.allclose(off, [K * (G * 0.0 - 0.3), 0.0, 0.0]))
            self.assertTrue(np.allclose(on, [K * (G * 0.0 - 0.3), -0.3, 0.0]))
            # from (2, 4): unreadable, v_B = K (0.99 Phi(3, 3) - Phi(2, 4)) = -1.67 is below the floor: unchanged;
            # the next turn starts from the real Phi(2, 4) (phi_from) as without the floor
            low = TC.table_record(episode([0, 1], [2, 1], [False, True], 2), tab, True)
            self.assertTrue(np.allclose(TC.turn_values(low, K, G, unreadable_floor=-0.3), TC.turn_values(low, K, G)))
            self.assertEqual([x.get("phi_from") for x in low["turns"]], [None, 0.6])
            # without unreadable_as_wrong no turn is marked wrong_step: the floor changes nothing (settings refuse it)
            plain = TC.table_record(episode([3, 3, 4], [3, 3, 2], [True, False, True], 3), tab)
            self.assertEqual(TC.turn_values(plain, K, G, unreadable_floor=-0.3), TC.turn_values(plain, K, G))
            # settings: a negative number, only with unreadable_as_wrong (and so with the success table)
            base = {"phi_version": "success_table", "phi_table": str(p), "baseline": "none", "episode_weight": 0,
                    "unreadable_as_wrong": True}
            self.assertEqual(TC.settings({"turn_credit": dict(base, unreadable_floor=-0.3)})["unreadable_floor"], -0.3)
            self.assertIsNone(TC.settings({"turn_credit": base})["unreadable_floor"])
            for bad in (0, 0.1, "x", True, float("nan"), float("-inf")):
                with self.assertRaises(ValueError, msg=repr(bad)):
                    TC.settings({"turn_credit": dict(base, unreadable_floor=bad)})
            with self.assertRaises(ValueError):
                TC.settings({"turn_credit": dict(base, unreadable_as_wrong=False, unreadable_floor=-0.3)})
            # the estimator: the floored values, and the share of unreadable turns the floor lowered
            cfg = TC.settings({"turn_credit": dict(base, unreadable_floor=-0.3)})
            raw = [episode([3, 3, 4], [3, 3, 2], [True, False, True], 3), episode([0, 1], [2, 1], [False, True], 2)]
            z, m = TC.turn_advantages(raw, ["a", "b"], cfg)
            self.assertEqual(m["turn_credit/unreadable_floor_share"], 0.5)
            vals = [x for r in raw for x in TC.turn_values(TC.table_record(r, tab, True), K, G, unreadable_floor=-0.3)]
            flat = [x for xs in z for x in xs]
            self.assertTrue(np.allclose(flat, np.asarray(vals) / (np.std(vals) + cfg["eps"])))
            z0, m0 = TC.turn_advantages(raw, ["a", "b"], TC.settings({"turn_credit": base}))
            self.assertNotIn("turn_credit/unreadable_floor_share", m0)
            self.assertFalse(np.allclose([x for xs in z0 for x in xs], flat))

    def test_unreadable_floor_random_and_switch_off_matches_p3l2(self):
        # 辛 (the user's 11-16 一.2): on random records the floor moves only the unreadable turns, to min(v_B, floor);
        # with the switch off the estimator is 庚's frozen code (p3l2, reading B) record by record
        rng = np.random.default_rng(20261116)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            phi = np.clip(rng.random((13, 13)), 0, 1)
            phi[0] = 1.0
            phi[:, 0] = np.where(np.arange(13) == 0, 1.0, 0.0)
            p.write_text(json.dumps({"version": "success_table_v1", "budget": 12, "phi": phi.tolist()}))
            tab = TC.load_success_table(str(p))
            recs = []
            for _ in range(3000):
                n = int(rng.integers(1, 13))
                steps, turns = 0, []
                for i in range(n):
                    valid = bool(rng.random() > 0.3)
                    steps += int(rng.integers(1, 3)) if valid else 0
                    d = None if rng.random() < 0.05 else int(rng.integers(0, 14))
                    turns.append({"span": [i, i + 1], "reward": float(rng.normal()), "phi_after": float(rng.random()),
                                  "d_after": d, "success": bool(i == n - 1 and rng.random() < 0.3), "valid": valid,
                                  **({"steps": min(steps, 12)} if rng.random() > 0.02 else {})})
                recs.append({"phi0": float(rng.random()), "d0": int(rng.integers(0, 13)), "turns": turns})
            n_floor = n_dead = 0
            for r in recs:
                t = TC.table_record(r, tab, True)
                off, on = TC.turn_values(t, K, G), TC.turn_values(t, K, G, unreadable_floor=-0.3)
                for x, a, b in zip(t["turns"], off, on):
                    if x.get("wrong_step"):
                        self.assertEqual(b, min(a, -0.3))
                        n_floor += a > -0.3
                        n_dead += a > -0.3 and x["phi_after"] == 0.0
                    else:
                        self.assertEqual(a, b)
            self.assertGreater(n_floor, 0)
            self.assertGreater(n_dead, 0)
            src = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-reward-p3l2/vagen/turn_credit.py")
            if not src.is_file():
                self.skipTest("no p3l2 frozen copy here")
            spec = importlib.util.spec_from_file_location("turn_credit_p3l2", src)
            old = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old)
            conf = {"turn_credit": {"phi_version": "success_table", "phi_table": str(p), "baseline": "none",
                                    "episode_weight": 0, "unreadable_as_wrong": True}}
            cfg_new, cfg_old = TC.settings(conf), old.settings(conf)
            tab_old = old.load_success_table(str(p))
            for r in recs:
                a, b = TC.table_record(r, tab, True), old.table_record(r, tab_old, True)
                self.assertEqual(a, b)
                self.assertEqual(TC.turn_values(a, K, G), old.turn_values(b, K, G))
            za, ma = TC.turn_advantages(recs, list(range(len(recs))), cfg_new)
            zb, mb = old.turn_advantages(recs, list(range(len(recs))), cfg_old)
            self.assertEqual(za, zb)
            self.assertEqual(ma, mb)

    def test_switch_off_matches_p3k(self):
        # the user's 11-14 night 一.2: with the switch off, the same as 戊's frozen code (p3k) record by record
        import importlib.util

        src = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-reward-p3k/vagen/turn_credit.py")
        if not src.is_file():
            self.skipTest("no p3k frozen copy here")
        spec = importlib.util.spec_from_file_location("turn_credit_p3k", src)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        rng = np.random.default_rng(20261114)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            phi = np.clip(rng.random((13, 13)), 0, 1)
            phi[0] = 1.0
            p.write_text(json.dumps({"version": "success_table_v1", "budget": 12, "phi": phi.tolist()}))
            tab_new, tab_old = TC.load_success_table(str(p)), old.load_success_table(str(p))
            recs = []
            for _ in range(3000):
                n = int(rng.integers(1, 13))
                steps, ds, turns = 0, [], []
                for i in range(n):
                    valid = bool(rng.random() > 0.3)
                    steps += int(rng.integers(1, 3)) if valid else 0
                    d = None if rng.random() < 0.05 else int(rng.integers(0, 14))
                    turns.append({"span": [i, i + 1], "reward": float(rng.normal()), "phi_after": float(rng.random()),
                                  "d_after": d, "success": bool(i == n - 1 and rng.random() < 0.3), "valid": valid,
                                  **({"steps": min(steps, 12)} if rng.random() > 0.02 else {})})
                recs.append({"phi0": float(rng.random()), "d0": int(rng.integers(0, 13)), "turns": turns})
            cfg_new = TC.settings({"turn_credit": {"phi_version": "success_table", "phi_table": str(p),
                                                   "baseline": "none", "episode_weight": 0}})
            cfg_old = old.settings({"turn_credit": {"phi_version": "success_table", "phi_table": str(p),
                                                    "baseline": "none", "episode_weight": 0}})
            for r in recs:
                a, b = TC.table_record(r, tab_new), old.table_record(r, tab_old)
                self.assertEqual(a, b)
                self.assertEqual(TC.turn_values(a, K, G), old.turn_values(b, K, G))
            za, ma = TC.turn_advantages(recs, list(range(len(recs))), cfg_new)
            zb, mb = old.turn_advantages(recs, list(range(len(recs))), cfg_old)
            self.assertEqual(za, zb)
            self.assertEqual(ma, mb)

    def test_settings_and_selection(self):
        self.assertEqual(TC.settings({"turn_credit": {"episode_weight": 0}})["episode_weight"], 0.0)
        s = TC.settings({})
        self.assertEqual((s["baseline"], s["turn_weight"], s["episode_norm"], s["turn_reward"], s["neg_clip"],
                          s["min_positive_mass"], s["stall_penalty"]), ("batch_mean", 0.5, "batch", False, 0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            TC.settings({"turn_credit": {"baseline": "phi_bin"}})
        with self.assertRaises(ValueError):
            TC.settings({"turn_credit": {"episode_norm": "group"}})
        with self.assertRaises(ValueError):
            TC.settings({"turn_credit": {"turn_wieght": 0.5}})  # a misspelt key is refused, not ignored
        self.assertEqual(s["phi_version"], "optimal")  # experiment J: the table's score unless asked
        self.assertEqual(TC.settings({"turn_credit": {"phi_version": "grouped_gated"}})["phi_version"], "grouped_gated")
        with self.assertRaises(ValueError):
            TC.settings({"turn_credit": {"phi_version": "product"}})
        with self.assertRaises(ValueError):  # experiment K: the table goes with success_table, and only with it
            TC.settings({"turn_credit": {"phi_version": "success_table"}})
        with self.assertRaises(ValueError):
            TC.settings({"turn_credit": {"phi_table": "/x.json"}})
        self.assertIsNone(s["phi_table"])
        self.assertIs(s["unreadable_as_wrong"], False)  # experiment L: off unless asked
        with self.assertRaises(ValueError):  # experiment L: only with the success table
            TC.settings({"turn_credit": {"unreadable_as_wrong": True}})
        with self.assertRaises(ValueError):
            TC.settings({"turn_credit": {"phi_version": "success_table", "phi_table": "/x.json",
                                         "unreadable_as_wrong": "yes"}})
        self.assertIs(TC.settings({"turn_credit": {"phi_version": "success_table", "phi_table": "/x.json",
                                                   "unreadable_as_wrong": True}})["unreadable_as_wrong"], True)
        self.assertTrue(TC.selected({"algorithm": {"adv_estimator": "turn_credit"}}))
        self.assertFalse(TC.selected({"algorithm": {"adv_estimator": "grpo"}}))
        self.assertFalse(TC.selected(None))


class NeighbourTableTest(unittest.TestCase):
    """distance_table_near: a missing contract key takes its 26 neighbours' smallest reachable distance."""

    def test_neighbour_fallback(self):
        from data_gen.interiorgs_v10 import distance_table as DT
        from data_gen.interiorgs_v10 import distance_table_near as NT
        from data_gen.interiorgs_v10.difficulty import _split, contract_keys

        c2w = np.eye(4)
        c2w[:3, 3] = [1.234, 2.345, 0.0]
        k = int(contract_keys(*_split(c2w))[0])
        kx, ky, kyaw = NT.unpack(k)
        self.assertEqual(NT.pack(kx, ky, kyaw), k)
        near = [NT.pack(kx + 1, ky, kyaw), NT.pack(kx, ky - 1, kyaw + 1), NT.pack(kx - 1, ky - 1, kyaw)]
        far = NT.pack(kx + 2, ky, kyaw)
        keys = np.array(sorted(near + [far]), dtype=np.int64)
        dist = {near[0]: 5, near[1]: 3, near[2]: -1, far: 1}  # -1: no success reachable
        table = {"keys": keys, "dist": np.array([dist[int(x)] for x in keys], dtype=np.int64)}
        self.assertIsNone(DT.lookup(table, c2w))
        self.assertEqual(NT.lookup_near(table, c2w), {"distance": 3, "near": True})
        self.assertEqual(NT.neighbour_distance({"keys": np.array([far], dtype=np.int64), "dist": np.array([1])}, k), None)
        # a key in the table is looked up as before
        table2 = {"keys": np.array(sorted([k, near[0]]), dtype=np.int64), "dist": None}
        table2["dist"] = np.array([2 if int(x) == k else 5 for x in table2["keys"]], dtype=np.int64)
        self.assertEqual(NT.lookup_near(table2, c2w), {"distance": 2, "near": False})
        # the cache: phi from the neighbour's distance
        with tempfile.TemporaryDirectory() as tmp:
            DT.save(Path(tmp) / "t1.npz", dict(table, depth_of=np.zeros(len(keys), dtype=np.int64), task_id="t1"))
            item = {"task_id": "t1", "difficulty": {"d_star": 4}}
            plain = DT.TableCache(tmp).progress(item, c2w, False)
            fb = NT.NearTableCache(tmp).progress(item, c2w, False)
        self.assertEqual((plain["optimal_distance"], plain["optimal_progress"]), (None, 0.0))
        self.assertEqual((fb["optimal_distance"], fb["optimal_near"]), (3, True))
        self.assertAlmostEqual(fb["optimal_progress"], 1 - 3 / 8)


@unittest.skipUnless(HAVE_TRAINING_ENV, "needs the training env (ray, omegaconf, tensordict)")
class EstimatorAndFilterTest(unittest.TestCase):
    def test_registered_estimator(self):
        import vagen.custom_advantage  # noqa: F401  (registers the estimators)
        from omegaconf import OmegaConf
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn

        fn = get_adv_estimator_fn("turn_credit")
        mask = torch.ones(2, 4)
        rewards = torch.zeros(2, 4)
        rewards[0, -1] = 5.0
        records = [rec(0.0, [(0, 4, 5.0, 0.0, True)]), rec(0.0, [(0, 4, 0.0, 0.1, False)])]
        data = SimpleNamespace(batch={"response_mask": mask, "token_level_rewards": rewards},
                               non_tensor_batch={"turn_records": np.array(records, dtype=object),
                                                 "uid": np.array(["a", "a"], dtype=object)}, meta_info={})
        cfg = OmegaConf.create({"adv_estimator": "turn_credit", "turn_credit": {"baseline": "turn_index"}})
        adv, ret = fn(data=data, config=cfg, gamma=1.0, lam=1.0)
        self.assertTrue(torch.equal(adv, ret))
        self.assertGreater(float(adv[0, 0]), 0.0)
        self.assertLess(float(adv[1, 0]), 0.0)
        self.assertIn("turn_credit/pos_share", data.meta_info["turn_credit_metrics"])
        self.assertTrue(TC.selected(OmegaConf.create({"algorithm": {"adv_estimator": "turn_credit"}})))
        # the smoke's dump: spans, ids, mask and token advantages of each sample
        with tempfile.TemporaryDirectory() as tmp:
            data.batch["responses"] = torch.arange(8).reshape(2, 4)
            dcfg = OmegaConf.create({"adv_estimator": "turn_credit",
                                     "turn_credit": {"dump_dir": tmp, "dump_updates": 1000}})
            adv, _ = fn(data=data, config=dcfg, gamma=1.0, lam=1.0)
            files = sorted(Path(tmp).glob("update_*.jsonl"))
            self.assertEqual(len(files), 1)
            lines = [json.loads(l) for l in files[0].read_text().splitlines()]
            self.assertEqual([l["ids"] for l in lines], [[0, 1, 2, 3], [4, 5, 6, 7]])
            self.assertAlmostEqual(lines[0]["adv"][0], float(adv[0, 0]), places=4)
            self.assertEqual(lines[1]["record"]["turns"][0]["span"], [0, 4])
        del data.non_tensor_batch["turn_records"]
        with self.assertRaises(KeyError):
            fn(data=data, config=cfg, gamma=1.0, lam=1.0)

    @unittest.skip("q38 port (p2): the 8B entry vagen/interiorgs_v10_ppo.py stays unchanged (DESIGN_ASYNC 2.4); the "
                   "q38 stack runs turn_credit through vagen.q38.trainer (vagen/q38/tests/test_turn_arms.py)")
    def test_ppo_entry_point_accepts_the_estimator(self):
        # the runner checks (adv_estimator, concat) before training: the smoke of 11-01 failed here
        from omegaconf import OmegaConf

        from vagen.interiorgs_v10_ppo import history_mode
        from verl.trainer.ppo.utils import need_critic

        def cfg(est, concat):
            return OmegaConf.create({"algorithm": {"adv_estimator": est}, "trainer": {"concat_multi_turn": concat},
                                     "critic": {"enable": None}})

        self.assertEqual(history_mode(cfg("turn_credit", True)), "concat")
        self.assertEqual(history_mode(cfg("grpo", True)), "concat")  # unchanged
        with self.assertRaises(ValueError):
            history_mode(cfg("turn_credit", False))
        self.assertFalse(need_critic(cfg("turn_credit", True)))  # no value network

    def test_nominal_filter_keeps_every_outcome(self):
        from verl import DataProto

        from vagen.custom_filter.filter import FILTER_REGISTRY

        outcomes = [[1, 1], [0, 0], [1, 0], [0, 0], [1, 1]]  # t0 all success, t1 all fail, t2 mixed, ...
        uids, succ = [], []
        for t, o in enumerate(outcomes):
            uids += [f"t{t}"] * len(o)
            succ += [float(x) for x in o]
        d = DataProto.from_dict(tensors={"advantages": torch.ones(len(uids), 3)},
                                non_tensors={"uid": np.array(uids, dtype=object), "traj_success": np.array(succ)})
        d.meta_info["uid_draw_order"] = {"t3": 0, "t0": 1, "t1": 2, "t2": 3, "t4": 4}
        out, m = FILTER_REGISTRY["nominal"](d, {}, target_groups=3)
        self.assertEqual(sorted(set(out.non_tensor_batch["uid"])), ["t0", "t1", "t3"])
        self.assertEqual((m["filter/nominal/kept"], m["filter/nominal/all_fail"], m["filter/nominal/all_success"],
                          m["filter/nominal/mixed"]), (3, 2, 1, 0))


class RlConfigTest(unittest.TestCase):
    """rl-config --turn-credit / --group-filter nominal on synthetic scenes (the training-prep tests' fixture)."""

    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import training_prep as TP
        from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
        from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

        cls.TP = TP
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        rows = _jsonable(generate_scene(paths["two_rooms"], "two_rooms", RunConfig(target_tasks_per_scene=4)))["rows"]
        val = [dict(r, scene_id="two_rooms_val", task_id=r["task_id"] + "_val") for r in rows[:2]]
        run_dir = cls.root / "run"
        run_dir.mkdir()
        (run_dir / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + val))
        manifest = cls.root / "split.json"
        manifest.write_text(json.dumps({"version": 2, "train_scenes": ["two_rooms"], "val_scenes": ["two_rooms_val"],
                                        "test_scenes": []}))
        cls.sft = cls.root / "sft"
        TP.cmd_sft_frames(Namespace(run=str(run_dir), split_manifest=str(manifest), splits="train,val", limit=0,
                                    out=str(cls.sft)))
        rs = [json.loads(l) for l in (cls.sft / "rows.jsonl").read_text().splitlines() if l.strip()]
        for r in rs:
            r["reserve"] = r["split"] == "train"
        (cls.sft / "rows_reserve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        cls.tables = cls.root / "tables"
        cls.tables.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def config(self, out_name, **kw):
        import yaml

        args = dict(rows=str(self.sft / "rows_reserve.jsonl"), gs_root="/ply", model="/m", name="t",
                    run_root=str(self.root / f"run_{out_name}"), val_limit=1, trial=False, out=str(self.root / out_name))
        args.update(kw)
        self.TP.cmd_rl_config(Namespace(**args))
        out = self.root / out_name
        return (yaml.safe_load((out / "ppo.yaml").read_text()),
                yaml.safe_load((out / "train_env.yaml").read_text())["envs"][0]["config"],
                json.loads((out / "rl_config_report.json").read_text()))

    def test_turn_credit_switch(self):
        tc = {"episode_weight": 1.0, "baseline": "turn_index"}
        ppo, env, report = self.config("tc", standard="v10_p1_S0", seed=7, progress_version="optimal",
                                       distance_tables=str(self.tables), turn_credit=json.dumps(tc),
                                       group_filter="nominal")
        self.assertEqual(ppo["algorithm"]["adv_estimator"], "turn_credit")
        self.assertEqual(ppo["algorithm"]["turn_credit"], tc)
        self.assertEqual(ppo["filter"], {"enable": True, "name": "nominal", "filter_kwargs": {"target_groups": 12}})
        self.assertEqual((env["progress_version"], env["distance_tables_dir"]), ("optimal", str(self.tables.resolve())))
        self.assertFalse(env.get("enable_potential_shaping_reward", False))  # the S0 reward
        self.assertEqual((report["turn_credit"], report["adv_estimator"], report["group_filter"]), (tc, "turn_credit", "nominal"))
        # without the switch nothing changes: S0 is GRPO with mixed_outcome
        ppo, env, report = self.config("s0", standard="v10_p1_S0", seed=7)
        self.assertEqual((ppo["algorithm"]["adv_estimator"], ppo["filter"]["name"]), ("grpo", "mixed_outcome"))
        self.assertNotIn("turn_credit", ppo["algorithm"])
        # refused: a dense reward (the credit adds the progress term itself), no table, a bad setting
        with self.assertRaises(SystemExit):
            self.config("tc_dense", standard="v10_p1_Dcond", seed=7, progress_version="optimal",
                        distance_tables=str(self.tables), turn_credit=json.dumps(tc))
        with self.assertRaises(SystemExit):
            self.config("tc_notable", standard="v10_p1_S0", seed=7, turn_credit=json.dumps(tc))
        with self.assertRaises(SystemExit):
            self.config("tc_bad", standard="v10_p1_S0", seed=7, progress_version="optimal",
                        distance_tables=str(self.tables), turn_credit=json.dumps({"baseline": "phi_bin"}))
        # the S0 reward with the table's score but no turn credit is still refused (the progress options are dense)
        with self.assertRaises(SystemExit):
            self.config("s0_optimal", standard="v10_p1_S0", seed=7, progress_version="optimal",
                        distance_tables=str(self.tables))

    def test_turn_credit_v4_phi(self):
        # experiment J (丙): the S0 reward, the env on V4's progress version, no tables, phi_version grouped_gated
        tc = {"baseline": "none", "episode_weight": 0, "phi_version": "grouped_gated"}
        ppo, env, report = self.config("tcv4", standard="v10_p1_S0", seed=7, progress_version="grouped_gated",
                                       turn_credit=json.dumps(tc), group_filter="nominal", target_groups=24)
        self.assertEqual((ppo["algorithm"]["adv_estimator"], ppo["algorithm"]["turn_credit"]), ("turn_credit", tc))
        self.assertEqual(env["progress_version"], "grouped_gated")
        self.assertNotIn("distance_tables_dir", env)
        self.assertFalse(env.get("optimal_progress_near", False))
        self.assertFalse(env.get("enable_potential_shaping_reward", False))  # still the S0 reward
        self.assertNotIn("progress_weights", env)
        self.assertEqual(report["turn_credit"], tc)
        # the same env without the setting is refused as before (the default reads the table)
        with self.assertRaises(SystemExit):
            self.config("tcv4_noset", standard="v10_p1_S0", seed=7, progress_version="grouped_gated",
                        turn_credit=json.dumps({"baseline": "none"}))
        # the setting with the table's env, with tables, or with V4's dense reward is refused
        with self.assertRaises(SystemExit):
            self.config("tcv4_opt", standard="v10_p1_S0", seed=7, progress_version="optimal",
                        distance_tables=str(self.tables), turn_credit=json.dumps(tc))
        with self.assertRaises(SystemExit):
            self.config("tcv4_tables", standard="v10_p1_S0", seed=7, progress_version="grouped_gated",
                        distance_tables=str(self.tables), turn_credit=json.dumps(tc))
        with self.assertRaises(SystemExit):
            self.config("tcv4_dense", standard="v10_p1_Dcond", seed=7, turn_credit=json.dumps(tc))
        with self.assertRaises(SystemExit):  # no env version given: the S0 env's product score is not V4's
            self.config("tcv4_product", standard="v10_p1_S0", seed=7, turn_credit=json.dumps(tc))
        with self.assertRaises(SystemExit):  # no table distance: a stall penalty would silently do nothing
            self.config("tcv4_stall", standard="v10_p1_S0", seed=7, progress_version="grouped_gated",
                        turn_credit=json.dumps({**tc, "stall_penalty": 0.1}))

    def test_turn_credit_success_table(self):
        # experiment K (戊): the table's env (optimal, tables) with phi from the success table of the env's budget
        tab = self.root / "success_table.json"
        tab.write_text(json.dumps({"version": "success_table_v1", "budget": 12,
                                   "phi": [[1.0] * 13] + [[0.0] * 13 for _ in range(12)]}))
        tc = {"baseline": "none", "episode_weight": 0, "phi_version": "success_table", "phi_table": str(tab)}
        ppo, env, report = self.config("tcst", standard="v10_p1_S0", seed=7, progress_version="optimal",
                                       distance_tables=str(self.tables), turn_credit=json.dumps(tc),
                                       group_filter="nominal", target_groups=24)
        self.assertEqual(ppo["algorithm"]["adv_estimator"], "turn_credit")
        self.assertEqual(ppo["algorithm"]["turn_credit"], dict(tc, phi_table=str(tab.resolve())))
        self.assertEqual((env["progress_version"], env["distance_tables_dir"]), ("optimal", str(self.tables.resolve())))
        self.assertFalse(env.get("enable_potential_shaping_reward", False))
        with self.assertRaises(SystemExit):  # no tables
            self.config("tcst_notab", standard="v10_p1_S0", seed=7, turn_credit=json.dumps(tc))
        # experiment L (庚): the switch passes through to the PPO config; refused without the success table
        tcw = dict(tc, unreadable_as_wrong=True)
        ppo_w, _, _ = self.config("tcst_wrong", standard="v10_p1_S0", seed=7, progress_version="optimal",
                                  distance_tables=str(self.tables), turn_credit=json.dumps(tcw),
                                  group_filter="nominal", target_groups=24)
        self.assertEqual(ppo_w["algorithm"]["turn_credit"], dict(tcw, phi_table=str(tab.resolve())))
        with self.assertRaises(SystemExit):
            self.config("tcst_wrong_opt", standard="v10_p1_S0", seed=7, progress_version="optimal",
                        distance_tables=str(self.tables),
                        turn_credit=json.dumps({"baseline": "none", "episode_weight": 0, "unreadable_as_wrong": True}))
        four = self.root / "success_table_b4.json"
        four.write_text(json.dumps({"version": "success_table_v1", "budget": 4, "phi": [[1.0] * 5, [0.5] * 5]}))
        with self.assertRaises(SystemExit):  # the table's budget is not the env's 12
            self.config("tcst_budget", standard="v10_p1_S0", seed=7, progress_version="optimal",
                        distance_tables=str(self.tables), turn_credit=json.dumps(dict(tc, phi_table=str(four))))
        with self.assertRaises(SystemExit):  # no such table
            self.config("tcst_nofile", standard="v10_p1_S0", seed=7, progress_version="optimal",
                        distance_tables=str(self.tables),
                        turn_credit=json.dumps(dict(tc, phi_table=str(self.root / "missing.json"))))
        # the table's source must be this env's: the same tables and neighbour lookup (code review 11-14)
        other = self.root / "other_tables"
        other.mkdir(exist_ok=True)
        for name, src, ok in (("src_ok", {"distance_tables_dir": str(self.tables), "optimal_progress_near": False}, True),
                              ("src_dir", {"distance_tables_dir": str(other), "optimal_progress_near": False}, False),
                              ("src_near", {"distance_tables_dir": str(self.tables), "optimal_progress_near": True}, False)):
            f = self.root / f"success_table_{name}.json"
            f.write_text(json.dumps({"version": "success_table_v1", "budget": 12, "source": src,
                                     "phi": [[1.0] * 13] + [[0.0] * 13 for _ in range(12)]}))
            kw = dict(standard="v10_p1_S0", seed=7, progress_version="optimal", distance_tables=str(self.tables),
                      turn_credit=json.dumps(dict(tc, phi_table=str(f))))
            if ok:
                self.config(f"tcst_{name}", **kw)
            else:
                with self.assertRaises(SystemExit):
                    self.config(f"tcst_{name}", **kw)

    def test_phase2_first_round(self):
        # the brief's direction 1: phase-2 defaults, neighbour fallback, all 24 candidates in the update
        ppo, env, report = self.config("tc2", standard="v10_p1_S0", seed=7, progress_version="optimal",
                                       distance_tables=str(self.tables), turn_credit=json.dumps({}),
                                       group_filter="nominal", table_near=True, target_groups=24)
        self.assertEqual(ppo["algorithm"]["adv_estimator"], "turn_credit")
        self.assertEqual(ppo["filter"]["filter_kwargs"], {"target_groups": 24})
        self.assertEqual((ppo["data"]["train_batch_size"], ppo["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"]),
                         (24, 24))
        self.assertTrue(env["optimal_progress_near"])
        # without --table-near the env keeps the plain lookup; --table-near needs the optimal progress
        _, env, _ = self.config("tc2_plain", standard="v10_p1_S0", seed=7, progress_version="optimal",
                                distance_tables=str(self.tables), turn_credit=json.dumps({}))
        self.assertFalse(env.get("optimal_progress_near", False))
        with self.assertRaises(SystemExit):
            self.config("near_alone", standard="v10_p1_S0", seed=7, table_near=True, turn_credit=json.dumps({}))


if __name__ == "__main__":
    unittest.main()
