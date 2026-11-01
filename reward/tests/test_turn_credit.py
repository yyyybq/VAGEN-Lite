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
        e = TC.episode_record(0.2, [t, dict(t, span=[10, 20])], response_length=12, d0=5)
        self.assertEqual([x["span"] for x in e["turns"]], [[3, 9], [10, 12]])
        self.assertEqual((e["phi0"], e["d0"]), (0.2, 5))
        self.assertEqual(TC.reset_distance({"initial_region_metrics": {"optimal_distance": 7}}), 7)
        self.assertIsNone(TC.reset_distance({"initial_region_metrics": {"optimal_distance": None}}))
        self.assertIsNone(TC.reset_distance(None))

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
