"""Guided sampling (vagen/jsrl_guide.py, phase 2 direction 2): the schedule, the turns, the greedy guide on synthetic
scenes and their tables, the agent loop's guide turns, and the rl-config switch.  Temporary directories only."""

import asyncio
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from vagen import jsrl_guide as J

HAVE_TRAINING_ENV = all(importlib.util.find_spec(m) is not None for m in ("ray", "omegaconf", "tensordict"))
BRIEF = [[1, 50, 0.5], [51, 100, 0.25]]


class ScheduleTest(unittest.TestCase):
    def test_settings(self):
        s = J.settings({"tables": "/t"})
        self.assertEqual((s["stages"], s["min_d_star"], s["families"], s["max_per_turn"]),
                         (BRIEF, 6, ["equidistance"], 5))
        for bad in ({}, {"tables": "/t", "k": 3}, {"tables": "/t", "stages": [[1, 50, 1.0]]},
                    {"tables": "/t", "stages": [[60, 50, 0.5]]}, {"tables": "/t", "max_per_turn": 0}):
            with self.assertRaises(ValueError):
                J.settings(bad)

    def test_guide_count_follows_the_brief(self):
        # updates 1-50 ceil(d*/2), 51-100 ceil(d*/4), 101 on 0; always below d*
        self.assertEqual([J.guide_count(8, s, BRIEF) for s in (1, 50, 51, 100, 101, 150)], [4, 4, 2, 2, 0, 0])
        self.assertEqual([J.guide_count(7, s, BRIEF) for s in (1, 51)], [4, 2])
        self.assertEqual([J.guide_count(6, s, BRIEF) for s in (1, 51)], [3, 2])
        self.assertEqual(J.guide_count(1, 1, BRIEF), 0)
        self.assertEqual(J.guide_count(12, 1, [[1, None, 0.5]]), 6)

    def test_eligible_turns_and_text(self):
        cfg = J.settings({"tables": "/t"})
        row = lambda d, fam: {"difficulty": {"d_star": d}, "task_type": fam}  # noqa: E731
        self.assertEqual([J.eligible(row(6, "fov_inclusion"), cfg), J.eligible(row(5, "fov_inclusion"), cfg),
                          J.eligible(row(3, "equidistance"), cfg)], [True, False, True])
        self.assertEqual(J.chunk(["a", "a", "b", "a"], 5), [["a", "a"], ["b"], ["a"]])
        self.assertEqual(J.chunk(["a"] * 7, 5), [["a"] * 5, ["a"] * 2])
        self.assertEqual(J.action_text(["move_forward", "move_forward"]), "<action>move_forward|move_forward|</action>")

    def test_from_config(self):
        self.assertIsNone(J.from_config({"algorithm": {"adv_estimator": "grpo"}}))
        self.assertEqual(J.from_config({"algorithm": {"jsrl_guide": {"tables": "/t"}}})["stages"], BRIEF)
        if HAVE_TRAINING_ENV:
            from omegaconf import OmegaConf

            cfg = OmegaConf.create({"algorithm": {"jsrl_guide": {"tables": "/t", "min_d_star": 4}}})
            self.assertEqual(J.from_config(cfg)["min_d_star"], 4)


class GuideOnSceneTest(unittest.TestCase):
    """The guide on the synthetic two-room scene with depth tables (distance_table's fixture)."""

    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import distance_table as DT
        from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
        from data_gen.interiorgs_v10.scene import load_scene
        from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

        cls.DT = DT
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.scene_dir = write_synthetic_scenes(root / "scenes")["two_rooms"]
        scene = load_scene(cls.scene_dir, "two_rooms")
        rows = _jsonable(generate_scene(cls.scene_dir, cls.scene_dir.name, RunConfig(target_tasks_per_scene=8)))["rows"]
        # tasks whose table agrees with the generator's d* (the guide's normal case), and at most one whose table puts
        # the start closer (the cap)
        cls.rows, cls.odd = [], None
        cls.tables = root / "tables"
        for row in sorted(rows, key=lambda r: -int(r["difficulty"]["d_star"])):
            d = int(row["difficulty"]["d_star"])
            if d < 3 or (len(cls.rows) >= 3 and cls.odd is not None):
                continue
            t = DT.build(scene, row, depth=d + 2)
            if int(t["d_init"]) == d and len(cls.rows) < 3:
                cls.rows.append(row)
            elif int(t["d_init"]) < d and cls.odd is None:
                cls.odd = row
            else:
                continue
            DT.save(DT.table_path(cls.tables, row["task_id"]), t)
        cls.tasks = root / "tasks.jsonl"
        cls.tasks.write_text("".join(json.dumps(r) + "\n" for r in cls.rows + ([cls.odd] if cls.odd else [])))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def env_config(self):
        from data_gen.interiorgs_v10.replay_env import ENV_CONFIG

        return dict(jsonl_path=str(self.tasks), gs_root=str(self.scene_dir.parent), **ENV_CONFIG,
                    progress_version="optimal", distance_tables_dir=str(self.tables))

    def env(self):
        from vagen.envs.active_spatial.env import ActiveSpatialEnv
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**self.env_config()))
        env._render = lambda init_obs=True, task_prompt="", env_feedback="": {"obs_str": "", "multi_modal_data": {}}
        return env

    def test_guide_walks_a_shortest_path_and_never_ends_the_task(self):
        # k = d*: capped at d* - 1 (the guide never ends a task); the turns, stepped in the env, put each pose where
        # planned, one table step closer each, and the one action left (the plan's own next step) ends the task
        self.assertTrue(self.rows)
        env = self.env()
        for i, row in enumerate(self.rows):
            env.reset(seed=i)
            self.assertEqual(env.current_item["task_id"], row["task_id"])
            d = int(row["difficulty"]["d_star"])
            table = self.DT.load(self.DT.table_path(self.tables, row["task_id"]))
            actions, keys, stop = J.plan(env, table, d)
            self.assertEqual((len(actions), stop), (d - 1, "capped"), row["task_id"])
            n = 0
            for turn in J.chunk(actions, 5):
                _, _, done, info = env.step(J.action_text(turn))
                n += len(turn)
                self.assertFalse(done)
                self.assertEqual(J.pose_key(env), keys[n - 1])
                self.assertEqual(env.current_region_metrics["optimal_distance"], d - n)
            last, _, _ = J.plan(env, table, 1)
            self.assertEqual(last, [])  # one action from success: the cap leaves it to the policy
            from data_gen.interiorgs_v10.contract import ACTIONS

            ok = []
            for a in ACTIONS:  # some single action from here succeeds (the table's d = 1)
                probe = self.env()
                probe.reset(seed=i)
                for turn in J.chunk(actions, 5):
                    probe.step(J.action_text(turn))
                _, _, _, info = probe.step(J.action_text([a]))
                ok.append(bool(info.get("success")))
            self.assertTrue(any(ok), row["task_id"])

    def test_table_closer_than_d_star_caps_the_guide(self):
        if self.odd is None:
            self.skipTest("no synthetic task whose table disagrees with d*")
        env = self.env()
        env.reset(seed=len(self.rows))
        table = self.DT.load(self.DT.table_path(self.tables, self.odd["task_id"]))
        d0 = J._distance(table, J.pose_key(env))
        actions, _, stop = J.plan(env, table, int(self.odd["difficulty"]["d_star"]))
        self.assertEqual((len(actions), stop), (max(0, d0 - 1), "capped"))

    def test_partial_guide_leaves_the_rest(self):
        env = self.env()
        for i, row in enumerate(self.rows):
            env.reset(seed=i)
            d = int(row["difficulty"]["d_star"])
            k = J.guide_count(d, 1, BRIEF)
            table = self.DT.load(self.DT.table_path(self.tables, row["task_id"]))
            actions, keys, stop = J.plan(env, table, k)
            self.assertEqual(len(actions), k)
            for turn in J.chunk(actions, 5):
                _, _, done, info = env.step(J.action_text(turn))
                self.assertFalse(done)
            if k:
                self.assertEqual(env.current_region_metrics["optimal_distance"], d - k)
            self.assertFalse(info.get("success", False) if k else False)

    @unittest.skipUnless(HAVE_TRAINING_ENV, "needs the training env (the agent loop's imports)")
    def test_run_guide_writes_history_only(self):
        from vagen.envs.active_spatial.active_spatial_env import ActiveSpatialGymEnv

        cfg = J.settings({"tables": str(self.tables), "min_d_star": 1, "families": []})
        for i, row in enumerate(self.rows):
            gym = ActiveSpatialGymEnv(self.env_config())
            gym._inner._render = lambda init_obs=True, task_prompt="", env_feedback="": {"obs_str": "", "multi_modal_data": {}}
            asyncio.run(gym.reset(seed=i))
            d = int(row["difficulty"]["d_star"])
            data = SimpleNamespace(env=gym, messages=[{"role": "system", "content": "s"}], image_data=[], env_rewards=[])
            rec = asyncio.run(J.run_guide(cfg, data, 1))
            k = J.guide_count(d, 1, BRIEF)
            turns = J.chunk(J.plan(self.env_at_reset(i), self.DT.load(self.DT.table_path(self.tables, row["task_id"])), k)[0], 5)
            self.assertEqual((rec["guide_eligible"], rec["guide_k"], rec["guide_actions"], rec["guide_turns"],
                              rec["guide_stop"]), (1, k, k, len(turns), ""))
            # one assistant answer and one observation per guide turn; nothing for the return
            roles = [m["role"] for m in data.messages[1:]]
            self.assertEqual(roles, ["assistant", "user"] * len(turns))
            self.assertEqual([m["content"] for m in data.messages[1::2]], [J.action_text(t) for t in turns])
            self.assertEqual(data.env_rewards, [])
            if k:
                self.assertEqual(gym._inner.current_region_metrics["optimal_distance"], d - k)
            # the stage after the brief's last one: no guide turn
            gym2 = ActiveSpatialGymEnv(self.env_config())
            gym2._inner._render = gym._inner._render
            asyncio.run(gym2.reset(seed=i))
            data2 = SimpleNamespace(env=gym2, messages=[], image_data=[], env_rewards=[])
            rec2 = asyncio.run(J.run_guide(cfg, data2, 101))
            self.assertEqual((rec2["guide_k"], rec2["guide_turns"], data2.messages), (0, 0, []))

    def env_at_reset(self, i):
        env = self.env()
        env.reset(seed=i)
        return env


class RlConfigTest(unittest.TestCase):
    """rl-config --jsrl-guide on synthetic scenes (the training-prep tests' fixture)."""

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

    def test_jsrl_guide_switch(self):
        guide = {"tables": str(self.tables)}
        ppo, env, report = self.config("g", standard="v10_p1_S0", seed=7, jsrl_guide=json.dumps(guide))
        self.assertEqual(ppo["algorithm"]["jsrl_guide"], guide)
        self.assertEqual((ppo["algorithm"]["adv_estimator"], ppo["filter"]["name"]), ("grpo", "mixed_outcome"))
        self.assertEqual(report["jsrl_guide"], guide)
        self.assertFalse(env.get("enable_potential_shaping_reward", False))  # the S0 reward
        # without the switch nothing changes
        ppo, _, report = self.config("s0", standard="v10_p1_S0", seed=7)
        self.assertNotIn("jsrl_guide", ppo["algorithm"])
        self.assertIsNone(report["jsrl_guide"])
        # refused: an unknown setting, a missing tables directory
        with self.assertRaises(SystemExit):
            self.config("g_bad", standard="v10_p1_S0", seed=7, jsrl_guide=json.dumps(dict(guide, k=3)))
        with self.assertRaises(SystemExit):
            self.config("g_notab", standard="v10_p1_S0", seed=7,
                        jsrl_guide=json.dumps({"tables": str(self.root / "none")}))


if __name__ == "__main__":
    unittest.main()
