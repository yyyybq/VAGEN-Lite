"""distance_table.py (round 13, the optimal dense reward): tables agree with the generator's d*, the env's optimal
progress reads them along a shortest path, and the prefilter never drops a successful state."""

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import difficulty as D
from data_gen.interiorgs_v10 import distance_table as DT
from data_gen.interiorgs_v10.replay_env import ENV_CONFIG
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class _Disk:
    def __init__(self, cx, cy, r, yaw_deg, tol_deg):
        self.cx, self.cy, self.r, self.yaw, self.tol = cx, cy, r, yaw_deg, tol_deg

    def __call__(self, c2w):
        c2w = np.asarray(c2w)
        dy = abs((math.degrees(C.yaw_of(c2w)) - self.yaw + 180.0) % 360.0 - 180.0)
        return math.hypot(c2w[0, 3] - self.cx, c2w[1, 3] - self.cy) <= self.r and dy <= self.tol


class DistanceTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        paths = write_synthetic_scenes(Path(cls.tmp.name))
        cls.scene_dir = paths["two_rooms"]
        cls.scene = load_scene(cls.scene_dir, "two_rooms")
        res = _jsonable(generate_scene(cls.scene_dir, cls.scene_dir.name, RunConfig(target_tasks_per_scene=5)))
        cls.rows = res["rows"][:4]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tables_agree_with_dstar_and_prefilter_drops_no_success(self):
        self.assertGreaterEqual(len(self.rows), 3)
        for row in self.rows:
            d = int(row["difficulty"]["d_star"])
            t = DT.build(self.scene, row, depth=d + 2, check_prefilter=True)
            self.assertEqual(t["d_init"], d, row["task_id"])
            self.assertTrue(t["agree"])
            self.assertEqual(t["prefilter_missed_success"], 0)
            self.assertTrue(np.all(np.diff(t["keys"]) > 0))  # sorted, unique: searchsorted lookups

    def test_disk_region_distances_match_bfs_and_fall_along_the_path(self):
        rng = np.random.default_rng(3)
        oracle = _Disk(7.5, 2.5, 0.35, 90.0, 25.0)
        checked = 0
        while checked < 4:
            x, y = rng.uniform(0.6, 9.4), rng.uniform(0.6, 4.4)
            init = C.pose_from_xy_yaw(x, y, math.radians(float(rng.choice(np.arange(0, 360, 20)))))
            if not self.scene.is_pose_legal(init)[0]:
                continue
            cert = D.certify_d_star(self.scene, oracle, init, d_max=6, key_mode="contract")
            if cert["d_star"] is None or cert["d_star"] == 0:
                continue
            row = {"init_camera": {"extrinsics": init.tolist(), "intrinsics": [[320, 0, 320], [0, 320, 240], [0, 0, 1]]},
                   "difficulty": {"d_star": cert["d_star"]}}
            t = DT.build(self.scene, row, depth=cert["d_star"] + 2, oracle=oracle, prefilter=False)
            self.assertEqual(t["d_init"], cert["d_star"])
            pose = init
            for k, a in enumerate(cert["certificate"]):  # the scalar runtime step, as the env moves
                self.assertEqual(DT.lookup(t, pose), cert["d_star"] - k)
                pose = C.apply_action(pose, a)
            self.assertEqual(DT.lookup(t, pose), 0)
            checked += 1

    def test_phi_save_load_and_cache(self):
        self.assertEqual(DT.phi(None, 3), 0.0)
        self.assertEqual(DT.phi(5, 3, success=True), 1.0)
        self.assertAlmostEqual(DT.phi(2, 3), 1 - 2 / 7)
        self.assertEqual(DT.phi(9, 3), 0.0)  # clipped
        self.assertEqual(DT.phi(0, 3), 1.0)
        row = self.rows[0]
        t = DT.build(self.scene, row, depth=int(row["difficulty"]["d_star"]) + 1)
        with tempfile.TemporaryDirectory() as tmp:
            DT.save(DT.table_path(Path(tmp), row["task_id"]), t)
            back = DT.load(DT.table_path(Path(tmp), row["task_id"]))
            np.testing.assert_array_equal(back["keys"], t["keys"])
            np.testing.assert_array_equal(back["dist"], t["dist"])
            self.assertEqual(back["d_init"], t["d_init"])
            cache = DT.TableCache(tmp)
            init = np.asarray(row["init_camera"]["extrinsics"], float)
            got = cache.progress(row, init, success=False)
            d = int(row["difficulty"]["d_star"])
            self.assertEqual(got["optimal_distance"], d)
            self.assertAlmostEqual(got["optimal_progress"], DT.phi(d, d))
            self.assertFalse(cache.progress(dict(row, task_id="missing"), init, False)["optimal_table"])

    def test_may_succeed_is_conservative(self):
        row = {"init_camera": {"intrinsics": [[320, 0, 320], [0, 320, 240], [0, 0, 1]]},
               "target_object": {"objects": [{"bbox_min": [4.8, -0.2, 0.0], "bbox_max": [5.2, 0.2, 1.0]}]}}
        ahead = C.pose_from_xy_yaw(0.0, 0.0, 0.0)  # facing +x, the object 5 m ahead
        behind = C.pose_from_xy_yaw(0.0, 0.0, math.pi)
        inside = C.pose_from_xy_yaw(5.0, 0.0, math.pi)  # within the bounding sphere
        got = DT.may_succeed(row, np.stack([ahead, behind, inside]))
        self.assertEqual(got.tolist(), [True, False, True])

    def test_env_optimal_progress_along_the_shortest_path(self):
        from vagen.envs.active_spatial.env import ActiveSpatialEnv
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tasks = tmp / "tasks.jsonl"
            tasks.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
            for row in self.rows:
                DT.save(DT.table_path(tmp / "tables", row["task_id"]),
                        DT.build(self.scene, row, depth=int(row["difficulty"]["d_star"]) + 2))
            cfg = ActiveSpatialEnvConfig(jsonl_path=str(tasks), gs_root=str(self.scene_dir.parent), **ENV_CONFIG,
                                         progress_version="optimal", distance_tables_dir=str(tmp / "tables"))
            env = ActiveSpatialEnv(cfg)
            env._render = lambda init_obs=True, task_prompt="", env_feedback="": {"obs_str": "", "multi_modal_data": {}}
            for i, row in enumerate(self.rows):
                env.reset(seed=i)
                self.assertEqual(env.current_task_id, row["task_id"])
                d = int(row["difficulty"]["d_star"])
                actions = [a for turn in row["turns_audit_only"] for a in turn]
                for k, a in enumerate(actions):
                    _, _, done, info = env.step(f"<think>t</think><action>{a}|</action>")
                    m = env.current_region_metrics
                    self.assertEqual(m.get("progress_version"), "optimal")
                    if k + 1 < d:
                        self.assertEqual(m["optimal_distance"], d - k - 1)
                        self.assertAlmostEqual(m["optimal_progress"], DT.phi(d - k - 1, d))
                    else:
                        self.assertTrue(info.get("success"))
                        self.assertEqual(m["optimal_progress"], 1.0)
                        break

    def test_depth_rule_and_shards(self):
        row = {"difficulty": {"d_star": 3}}
        self.assertEqual(DT.depth_for(row, 8, None), 8)
        self.assertEqual(DT.depth_for(row, 8, 3), 6)
        self.assertEqual(DT.depth_for({"difficulty": {"d_star": 7}}, 8, 3), 8)
        ids = [f"v10_{i:06d}" for i in range(400)]
        parts = [[t for t in ids if DT.in_shard(t, f"{i}/4")] for i in range(4)]
        self.assertEqual(sorted(sum(parts, [])), ids)  # every task in exactly one shard
        self.assertTrue(all(60 < len(p) < 140 for p in parts))
        self.assertTrue(DT.in_shard("anything", None))

    def test_rl_config_needs_both_options(self):
        from data_gen.interiorgs_v10.training_prep import design_variant_overrides

        env = {"enable_potential_shaping_reward": True, "progress_version": "grouped_gated"}
        with tempfile.TemporaryDirectory() as tmp:
            ok = design_variant_overrides(argparse.Namespace(progress_version="optimal", distance_tables=tmp), env)
            self.assertEqual(ok["progress_version"], "optimal")
            self.assertEqual(ok["distance_tables_dir"], str(Path(tmp).resolve()))
            for bad in (argparse.Namespace(progress_version="optimal", distance_tables=None),
                        argparse.Namespace(progress_version=None, distance_tables=tmp),
                        argparse.Namespace(progress_version="optimal", distance_tables=str(Path(tmp) / "nope"))):
                with self.assertRaises(SystemExit):
                    design_variant_overrides(bad, env)


if __name__ == "__main__":
    unittest.main()
