"""evaluation.py: the fixed test subset and the metrics table (the episodes themselves need a GPU)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import evaluation as E


class _Img:  # a frame stand-in that can be saved as a JPEG
    def convert(self, mode):
        from PIL import Image

        return Image.new(mode, (4, 4))


def _row(i, fam, d):
    return {"task_id": f"t{i:04d}", "task_type": fam, "scene_id": f"s{i % 7}", "difficulty": {"d_star": d}}


class EvaluationTest(unittest.TestCase):
    def test_select_is_stratified_and_fixed(self):
        rows = [_row(i, "projective_relations", 1 + i % 8) for i in range(200)]
        rows += [_row(1000 + i, "occlusion_alignment", 1 + i % 3) for i in range(20)]   # short of tasks
        rows += [_row(2000 + i, "equidistance", 7 + i % 2) for i in range(80)]          # one band only
        a = E.select_tasks(rows, 30, seed=5)
        b = E.select_tasks(list(reversed(rows)), 30, seed=5)
        self.assertEqual([r["task_id"] for r in a], [r["task_id"] for r in b])  # order of the input is irrelevant
        fam = {f: [r for r in a if r["task_type"] == f] for f in ("projective_relations", "occlusion_alignment",
                                                                   "equidistance")}
        self.assertEqual({f: len(v) for f, v in fam.items()},
                         {"projective_relations": 30, "occlusion_alignment": 20, "equidistance": 30})
        bands = [E.band_name(r["difficulty"]["d_star"]) for r in fam["projective_relations"]]
        self.assertEqual(sorted(bands.count(b) for b in ("1-3", "4-6", "7-8")), [10, 10, 10])
        self.assertNotEqual([r["task_id"] for r in a], [r["task_id"] for r in E.select_tasks(rows, 30, seed=6)])

    def test_metrics_and_table(self):
        rows = [{"task_type": "fov_inclusion", "d_star": 2, "success": True, "actions_to_success": 4, "turns": 2,
                 "actions": 4, "collisions": 1, "invalid_turns": 0},
                {"task_type": "fov_inclusion", "d_star": 8, "success": False, "actions_to_success": None, "turns": 12,
                 "actions": 12, "collisions": 0, "invalid_turns": 3},
                {"index": 2, "error": "renderer"}]
        m = E.metrics(rows)
        self.assertEqual((m["episodes"], m["errors"], m["success"]), (2, 1, 0.5))
        self.assertEqual(m["success_by_band"], {"1-3": 1.0, "7-8": 0.0})
        self.assertEqual(m["actions_over_dstar"], 2.0)
        self.assertEqual((m["collision_rate"], m["invalid_rate"]), (0.0625, round(3 / 14, 4)))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "sft"
            d.mkdir()
            (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            E.cmd_table(type("A", (), {"eval": tmp})())
            table = (Path(tmp) / "table.md").read_text(encoding="utf-8")
        self.assertIn("| sft | 2 | 0.500 |", table)

    def test_run_episode_records_the_trajectory(self):
        from PIL import Image

        class Env:  # two turns: a blocked move, then success at the third action
            current_item = {"task_id": "t1", "scene_id": "s1", "task_type": "fov_inclusion", "task_description": "Go.",
                            "difficulty": {"d_star": 2}}
            _current_step = 0

            def _obs(self, shade):
                return {"obs_str": f"<image>\nturn {self._current_step}",
                        "multi_modal_data": {"<image>": [Image.new("RGB", (8, 8), (shade, 0, 0))]}}

            def reset(self, seed):
                return self._obs(10), {}

            def system_prompt(self):
                return "sys"

            def step(self, text):
                acts = text.split("|")
                self._current_step += len(acts)
                done = self._current_step >= 3
                info = {"actions": acts, "metrics": {"turn_metrics": {"collision_count": int(not done),
                                                                      "action_is_valid": True}},
                        "termination_reason": "success_auto" if done else "continuing"}
                return self._obs(200 if done else 90), (4.99 if done else -0.16), done, info

            def _calculate_canonical_metric(self):
                ok = self._current_step >= 3
                return {"success": ok, "gates": {"distance_ok": ok, "target_a_clearly_visible": True},
                        "continuous_gate_progress": min(1.0, self._current_step / 3)}

        answers = iter(["move_forward|turn_left", "move_forward"])
        policy = type("P", (), {"generate": lambda self, s, u, im: next(answers)})()
        with tempfile.TemporaryDirectory() as tmp:
            row = E.run_episode(Env(), policy, 7, 12, frames=Path(tmp) / "frames")
            files = sorted(p.name for p in (Path(tmp) / "frames").iterdir())
        self.assertEqual(files, ["007_00.jpg", "007_01.jpg", "007_02.jpg"])
        self.assertEqual((row["success"], row["actions_to_success"], row["turns"], row["collisions"]), (True, 3, 2, 1))
        t0, t1 = row["trajectory"]
        self.assertEqual((t0["frame"], t0["user"], t0["answer"]), ("frames/007_00.jpg", "<image>\nturn 0",
                                                                    "move_forward|turn_left"))
        self.assertEqual((t0["actions"], t0["collisions"], t0["reward"], t0["met"]),
                         (["move_forward", "turn_left"], 1, -0.16, False))
        self.assertEqual((t1["met"], row["final_frame"], row["end_reason"], row["task_description"]),
                         (True, "frames/007_02.jpg", "success_auto", "Go."))
        self.assertEqual(row["start"], {"failed_gates": ["distance_ok"], "progress": 0.0})  # before the first turn
        self.assertEqual((t0["failed_gates"], t0["progress"], t1["failed_gates"], t1["progress"]),
                         (["distance_ok"], 0.6667, [], 1.0))
        answers = iter(["move_forward|turn_left", "move_forward"])
        self.assertNotIn("trajectory", E.run_episode(Env(), policy, 7, 12))  # without a frames folder: counts only
        # round 11: a sampled episode passes a fixed seed per (task, sample, turn) and records its sample number
        seeds = []

        def sampled(self, s, u, im, seed=None):
            seeds.append(seed)
            return next(answers)

        answers = iter(["move_forward|turn_left", "move_forward"])
        with tempfile.TemporaryDirectory() as tmp:
            row = E.run_episode(Env(), type("S", (), {"generate": sampled})(), 7, 12, frames=Path(tmp) / "frames",
                                sample=2, seed=11)
            files = sorted(p.name for p in (Path(tmp) / "frames").iterdir())
        self.assertEqual(row["sample"], 2)
        self.assertEqual(files, ["007_s2_00.jpg", "007_s2_01.jpg", "007_s2_02.jpg"])
        self.assertEqual(seeds, [E.call_seed(11, 7, 2, 0), E.call_seed(11, 7, 2, 1)])
        self.assertEqual(len({E.call_seed(11, 7, k, 0) for k in range(4)}), 4)  # the samples of a task differ
        self.assertEqual(E.call_seed(11, 7, 2, 0), E.call_seed(11, 7, 2, 0))   # and are reproducible

    def test_sampled_metrics_and_table(self):
        """A sampled run: the success column is the mean over the episodes, plus the share of tasks with at least one
        success; task-by-task comparison refuses it."""
        rows = []
        for i, outcomes in enumerate(([True, False, False, False], [False] * 4, [True] * 4)):
            for k, ok in enumerate(outcomes):
                rows.append({"index": i, "sample": k, "task_type": "fov_inclusion", "d_star": 2, "success": ok,
                             "actions_to_success": 2 if ok else None, "turns": 1, "actions": 2, "collisions": 0,
                             "invalid_turns": 0})
        m = E.metrics(rows)
        self.assertEqual((m["samples"], m["tasks"], m["episodes"]), (4, 3, 12))
        self.assertEqual(m["pass_at_n"], round(2 / 3, 4))
        self.assertEqual(m["success_mean"], round((0.25 + 0 + 1) / 3, 4))
        self.assertEqual(m["success"], round(5 / 12, 4))
        self.assertNotIn("pass_at_n", E.metrics([{k: v for k, v in rows[0].items() if k != "sample"}]))  # greedy
        with self.assertRaises(ValueError):
            E.paired(rows, rows)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "m_s4"
            d.mkdir()
            (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            E.cmd_table(type("A", (), {"eval": tmp})())
            table = (Path(tmp) / "table.md").read_text(encoding="utf-8")
        self.assertIn("| m_s4 | 3×4 | 0.417 |", table)
        self.assertIn("0.667（4 条）", table)
        import argparse

        self.assertFalse(E.sampling_mode(argparse.Namespace(samples=1, temperature=0.0)))
        self.assertTrue(E.sampling_mode(argparse.Namespace(samples=4, temperature=0.8)))
        self.assertTrue(E.sampling_mode(argparse.Namespace(samples=1, temperature=0.8)))

    def test_history_gives_every_earlier_turn_in_the_sft_layout(self):
        part = lambda im: {"type": "image", "id": im}  # noqa: E731
        msgs = E.chat_messages("sys", [("[Initial Observation]:\n<image>\nTask A", "f0", "<action>turn_left|</action>")],
                               "[Observation]:\n<image>\nEnvironment Feedback: Action executed.", "f1", part)
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertEqual(msgs[1]["content"], [{"type": "text", "text": "[Initial Observation]:\n"},
                                              {"type": "image", "id": "f0"}, {"type": "text", "text": "\nTask A"}])
        self.assertEqual(msgs[2]["content"], "<action>turn_left|</action>")
        self.assertEqual(msgs[3]["content"][1], {"type": "image", "id": "f1"})
        self.assertEqual(len(E.chat_messages("sys", [], "<image>\nx", "f", part)), 2)  # no history: system + user

        class Env:
            current_item = {"task_id": "t", "task_type": "fov_inclusion", "difficulty": {"d_star": 2}}
            _current_step = 0

            def reset(self, seed):
                return {"obs_str": "<image>\nturn 0", "multi_modal_data": {"<image>": ["frame0"]}}, {}

            def system_prompt(self):
                return "sys"

            def step(self, text):
                self._current_step += 1
                obs = {"obs_str": f"<image>\nturn {self._current_step}",
                       "multi_modal_data": {"<image>": [f"frame{self._current_step}"]}}
                return obs, -0.01, self._current_step >= 3, {"actions": ["turn_left"], "metrics": {}}

            def _calculate_canonical_metric(self):
                return {"success": False}

        calls, flags = [], []

        class Policy:
            last_prompt_tokens = None

            def generate(self, system, user_text, image, earlier=(), as_sft=False):
                calls.append((user_text, image, list(earlier)))
                flags.append(as_sft)
                self.last_prompt_tokens = 100 * (len(earlier) + 1)
                return f"<action>a{len(calls)}|</action>"

        row = E.run_episode(Env(), Policy(), 0, 12, history=True)
        self.assertEqual(row["turns"], 3)
        self.assertEqual(calls[0][2], [])
        self.assertEqual(calls[2][2], [("<image>\nturn 0", "frame0", "<action>a1|</action>"),
                                       ("<image>\nturn 1", "frame1", "<action>a2|</action>")])
        calls.clear()
        E.run_episode(Env(), Policy(), 0, 12)  # without history the earlier turns are not passed
        self.assertEqual([c[2] for c in calls], [[], [], []])
        flags.clear()
        E.run_episode(Env(), Policy(), 0, 12, as_sft=True)
        self.assertEqual(flags, [True, True, True])
        class FrameEnv(Env):  # frames that can be saved, for the recorded trajectory
            def reset(self, seed):
                obs, info = super().reset(seed)
                return dict(obs, multi_modal_data={"<image>": [_Img()]}), info

            def step(self, text):
                obs, reward, done, info = super().step(text)
                return dict(obs, multi_modal_data={"<image>": [_Img()]}), reward, done, info

        with tempfile.TemporaryDirectory() as tmp:
            rec = E.run_episode(FrameEnv(), Policy(), 0, 12, frames=Path(tmp) / "frames", history=True)
        self.assertEqual([t["prompt_tokens"] for t in rec["trajectory"]], [100, 200, 300])  # the input length per call

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "the SFT export package needs scipy (the cluster env)")
    def test_messages_as_the_sft_data_has_them(self):
        part = lambda im: {"type": "image", "id": im}  # noqa: E731
        # the env's trailing newlines removed (as the LLaMA-Factory export did), the rest unchanged
        sft = E.chat_messages("sys line.\n", [("<image>\nTask A\n", "f0", "<action>a|</action>")],
                              "[Observation]:\n<image>\nEnvironment Feedback: Action executed.\n", "f1", part, as_sft=True)
        self.assertEqual(sft[0]["content"], "sys line.")
        self.assertEqual(sft[1]["content"], [{"type": "image", "id": "f0"}, {"type": "text", "text": "\nTask A"}])
        self.assertEqual(sft[3]["content"][-1], {"type": "text", "text": "\nEnvironment Feedback: Action executed."})
        self.assertEqual(E.sft_text("a\n\n\n\nb\n"), "a\n\nb")

    def test_switches_reach_the_env_config(self):
        import argparse
        import os
        from unittest import mock

        configs = Path(E.__file__).with_name("configs")
        template = configs / "rl_env_v10_S0_r6.yaml"  # the switches are defined against the rounds 6-9 rules
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"R1_RENDER_URL": "http://x/render"}):
            tasks = Path(tmp) / "tasks.jsonl"
            tasks.write_text("")
            # round 10: the live template has switches 1, 3 and 4 on; the flags then change nothing
            live = E.env_config(configs / "rl_env_v10_S0.yaml", tasks, "/gs", None, {})
            self.assertEqual((live["success_check_each_action"], live["max_consecutive_collisions"],
                              live["collision_feedback_detail"], live["wall_model"]), (True, 0, True, "thick"))
            self.assertEqual(E.env_config(configs / "rl_env_v10_S0.yaml", tasks, "/gs", None, E.switch_overrides(
                argparse.Namespace(check_each_action=True, no_collision_end=True, collision_feedback=True))), live)
            plain = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace()))
            self.assertEqual(E.switch_overrides(argparse.Namespace()), {})  # no switch: the template's rules
            self.assertNotIn("success_check_each_action", plain)
            on = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace(check_each_action=True)))
            self.assertTrue(on["success_check_each_action"])
            self.assertEqual({k: v for k, v in on.items() if k != "success_check_each_action"}, plain)
            limit = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace(max_actions=20)))
            self.assertEqual((plain["max_episode_steps"], limit["max_episode_steps"]), (12, 20))
            free = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace(no_collision_end=True)))
            self.assertEqual((plain.get("max_consecutive_collisions", 3), free["max_consecutive_collisions"]), (3, 0))
            told = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace(collision_feedback=True)))
            self.assertTrue(told["collision_feedback_detail"])
            self.assertNotIn("collision_feedback_detail", plain)

    def test_end_reasons_and_paired_comparison(self):
        def r(i, ok, end):
            return {"index": i, "task_type": "fov_inclusion", "d_star": 2, "success": ok, "actions_to_success": 2 if ok else None,
                    "turns": 2, "actions": 4, "collisions": 0, "invalid_turns": 0, "end_reason": end}

        base = [r(0, True, "success_auto"), r(1, False, "max_primitive_steps"), r(2, False, "consecutive_collision"),
                r(3, True, "max_primitive_steps_success"), r(4, False, "max_turns"), {"index": 5, "error": "x"}]
        m = E.metrics(base)
        self.assertEqual(m["used_limit"], 3)  # the action limit (with and without success at the last action) or turn limit
        self.assertEqual(m["collision_ended"], 1)
        self.assertEqual(m["ended_by"]["success_auto"], 1)
        self.assertIsNone(E.metrics([dict(base[0], end_reason=None)])["used_limit"])  # eval v1 rows: not recorded
        other = [r(0, True, "success_auto"), r(1, True, "success_auto"), r(2, True, "success_auto"),
                 r(3, False, "max_primitive_steps"), r(4, False, "max_turns"), r(5, True, "success_auto")]
        self.assertEqual(E.paired(base, other), {"tasks": 5, "both": 1, "only_base": 1, "only_other": 2, "neither": 1,
                                                 "p": 1.0})
        self.assertEqual((round(E.sign_test(23, 18), 3), round(E.sign_test(6, 19), 3), E.sign_test(0, 0)),
                         (0.533, 0.015, 1.0))  # round 8: SFT e3 vs RL 250, SFT e2 vs e3
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in (("D0", base), ("D1", other)):
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            out = Path(tmp) / "cmp.json"
            E.cmd_compare(type("A", (), {"base": str(Path(tmp) / "D0"), "other": [str(Path(tmp) / "D1")], "out": str(out)})())
            self.assertEqual(json.loads(out.read_text())["compare"]["D1"]["only_other"], 2)
            self.assertIn("| D1 | 5 | 1 | 1 | 2 | 1 | 1.000 |", out.with_suffix(".md").read_text(encoding="utf-8"))

    def test_select_from_a_split_without_flagged_tasks(self):
        """Round 10: select --split val --exclude depth_check_summary.json (the fixed validation set)."""
        from argparse import Namespace

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "output"
            run.mkdir()
            rows = []
            for i in range(60):
                scene = f"v{i % 6}" if i % 2 else f"t{i % 6}"
                rows.append({"task_id": f"k{i:03d}", "scene_id": scene, "task_type": "equidistance",
                             "difficulty": {"d_star": 1 + i % 8}, "target_object": {}, "target_region": {},
                             "init_camera": {}, "camera_model_version": "m", "canonical_task_metric_version": "c",
                             "action_protocol_version": "a"})
            (run / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = Path(tmp) / "split.json"
            manifest.write_text(json.dumps({"version": 2, "train_scenes": [], "val_scenes": [f"v{k}" for k in range(6)],
                                            "test_scenes": [f"t{k}" for k in range(6)]}))
            flagged = Path(tmp) / "depth.json"
            flagged.write_text(json.dumps({"flagged_task_ids": [f"k{i:03d}" for i in range(1, 60, 2)][:10]}))
            out = Path(tmp) / "val200"
            E.cmd_select(Namespace(run=str(run), split_manifest=str(manifest), per_family=12, seed=1, split="val",
                                   exclude=str(flagged), out=str(out)))
            picked = [json.loads(l) for l in (out / "tasks.jsonl").read_text().splitlines()]
            sel = json.loads((out / "selection.json").read_text())
            self.assertEqual(len(picked), 12)
            self.assertTrue(all(r["scene_id"].startswith("v") and r["split"] == "val" for r in picked))
            self.assertFalse({r["task_id"] for r in picked} & set(json.loads(flagged.read_text())["flagged_task_ids"]))
            self.assertEqual((sel["split"], sel["val_tasks"], sel["excluded_flagged"]), ("val", 30, 10))
            # the default is the test split, nothing excluded, as before
            out2 = Path(tmp) / "test"
            E.cmd_select(Namespace(run=str(run), split_manifest=str(manifest), per_family=12, seed=1, out=str(out2)))
            picked2 = [json.loads(l) for l in (out2 / "tasks.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["scene_id"].startswith("t") and r["split"] == "test" for r in picked2))
            self.assertEqual(json.loads((out2 / "selection.json").read_text())["test_tasks"], 30)

    def test_replace_flagged(self):
        picked = [_row(i, "projective_relations", 2) for i in range(4)] + [_row(10, "occlusion_alignment", 8)]
        pool = picked + [_row(20 + i, "projective_relations", 1 + i % 3) for i in range(3)] + [_row(30, "projective_relations", 7)]
        res = E.replace_flagged(picked, ["t0001", "t0010", "t0021"], pool, seed=3)
        ids = [r["task_id"] for r in res["tasks"]]
        self.assertEqual(len(ids), 4)                         # the OA task had no substitute
        self.assertNotIn("t0001", ids)
        self.assertNotIn("t0021", ids)                        # flagged tasks never come back in
        swap = next(s for s in res["swaps"] if s["out"] == "t0001")
        self.assertIn(swap["in"], {"t0020", "t0022"})          # same family and d* band (1-3), unflagged
        self.assertIsNone(next(s for s in res["swaps"] if s["out"] == "t0010")["in"])
        self.assertEqual(res, E.replace_flagged(picked, ["t0001", "t0010", "t0021"], list(reversed(pool)), seed=3))


if __name__ == "__main__":
    unittest.main()
