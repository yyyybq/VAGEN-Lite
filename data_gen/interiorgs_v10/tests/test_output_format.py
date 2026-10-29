"""rl-config --output-format (docs/transition_qa, step 6): think_action shows the free_think prompt to RL, everything
else as action_only; the RL prompt differs from the SFT one only in its output-format text; the env parser reads both."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import yaml

from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.admission import load_policy
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class OutputFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        rows = _jsonable(generate_scene(paths["two_rooms"], "two_rooms", RunConfig(target_tasks_per_scene=4)))["rows"]
        val = [dict(r, scene_id="two_rooms_val", task_id=r["task_id"] + "_val") for r in rows[:2]]
        run = cls.root / "run"
        run.mkdir()
        (run / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + val))
        manifest = cls.root / "split.json"
        manifest.write_text(json.dumps({"version": 2, "train_scenes": ["two_rooms"], "val_scenes": ["two_rooms_val"],
                                        "test_scenes": []}))
        cls.sft = cls.root / "sft"
        TP.cmd_sft_frames(Namespace(run=str(run), split_manifest=str(manifest), splits="train,val", limit=0,
                                    out=str(cls.sft)))
        cls.template = cls.root / "ppo.yaml"
        cls.template.write_text(yaml.safe_dump({
            "algorithm": {"adv_estimator": "no_concat_gae"}, "data": {"train_batch_size": 12},
            "trainer": {"n_gpus_per_node": 8, "concat_multi_turn": False, "save_freq": 25, "test_freq": 25},
            "actor_rollout_ref": {"model": {"path": "old"}, "actor": {"ppo_mini_batch_size": 8},
                                  "rollout": {"n": 4, "tensor_model_parallel_size": 4, "agent": {"num_workers": 8}}},
            "critic": {"model": {"path": "old", "tokenizer_path": "old"}}}))
        cls.env_template = Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"
        cls.out = {}
        for fmt in (None, "action_only", "think_action"):
            out = cls.root / f"rl_{fmt}"
            args = dict(rows=str(cls.sft / "rows.jsonl"), env_template=str(cls.env_template), ppo_template=str(cls.template),
                        gs_root="/ply", model="/m", name="t", run_root=str(cls.root / f"run_{fmt}"), val_limit=1,
                        trial=False, out=str(out), history_mode="concat")
            if fmt:
                args["output_format"] = fmt
            TP.cmd_rl_config(Namespace(**args))
            cls.out[fmt] = out
        # matrix group E: the free_think prompt and a 6,144-token answer budget (user, 2026-10-19)
        out = cls.root / "rl_e"
        TP.cmd_rl_config(Namespace(rows=str(cls.sft / "rows.jsonl"), env_template=str(cls.env_template),
                                   ppo_template=str(cls.template), gs_root="/ply", model="/m", name="t",
                                   run_root=str(cls.root / "run_think_action"), val_limit=1, trial=False, out=str(out),
                                   history_mode="concat", output_format="think_action", max_response_length=6144))
        cls.out["e"] = out

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def env(self, fmt, split="train"):
        return yaml.safe_load((self.out[fmt] / f"{split}_env.yaml").read_text())["envs"][0]["config"]

    def test_default_is_unchanged(self):
        template = yaml.safe_load(self.env_template.read_text())["envs"][0]["config"]
        for split in ("train", "val"):
            self.assertEqual(self.env(None, split)["prompt_format"], template["prompt_format"])
            self.assertEqual(self.env("action_only", split), {**self.env(None, split),
                                                               "jsonl_path": self.env("action_only", split)["jsonl_path"]})
        rep = json.loads((self.out[None] / "rl_config_report.json").read_text())
        self.assertEqual((rep["output_format"], rep["rl_prompt_format"]), ("action_only", "no_think"))

    def test_think_action_changes_only_the_prompt_format(self):
        for split in ("train", "val"):
            a, t = self.env("action_only", split), self.env("think_action", split)
            self.assertEqual(t["prompt_format"], "free_think")
            self.assertEqual(sorted(k for k in set(a) | set(t) if a.get(k) != t.get(k) and k != "jsonl_path"),
                             ["prompt_format"])
        ppo = {f: yaml.safe_load((self.out[f] / "ppo.yaml").read_text()) for f in ("action_only", "think_action")}
        for p in ppo.values():
            p["data"].pop("train_files"), p["data"].pop("val_files")
            for k in ("default_local_dir", "rollout_data_dir", "validation_data_dir"):
                p["trainer"].pop(k)
        self.assertEqual(ppo["action_only"], ppo["think_action"])
        rep = json.loads((self.out["think_action"] / "rl_config_report.json").read_text())
        self.assertEqual((rep["output_format"], rep["rl_prompt_format"]), ("think_action", "free_think"))

    def test_rl_prompt_differs_from_sft_only_in_the_format_text(self):
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
        from vagen.envs.active_spatial.prompt import FORMAT_CONFIGS, _response_format

        policy = load_policy()
        types = ["projective_relations", "equidistance", "occlusion_alignment"]
        self.assertEqual(TP.prompt_mismatches(self.env("action_only"), types, policy), [])
        self.assertEqual(TP.prompt_mismatches(self.env("think_action"), types, policy), sorted(types))
        # the "Response format" line as the env shows it, with the run's format example (round 12 freeze, 3.1.1)
        cfg = self.env("think_action")
        a = ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                      if k in ActiveSpatialEnvConfig.__dataclass_fields__}).system_prompt_arguments()
        shown = {f: dict(FORMAT_CONFIGS[f], format=_response_format(FORMAT_CONFIGS[f], a["action_sep"],
                                                                    a["max_actions_per_step"], a["format_example"]))
                 for f in ("free_think", "no_think")}
        for t in types:
            think = TP.env_system_prompt(self.env("think_action"), t)
            for key in ("description", "format", "example"):
                think = think.replace(shown["free_think"][key], shown["no_think"][key])
            self.assertEqual(think, TP.env_system_prompt(self.env("action_only"), t))

    def test_unknown_format_is_refused(self):
        with self.assertRaises(SystemExit):
            TP.cmd_rl_config(Namespace(rows=str(self.sft / "rows.jsonl"), env_template=str(self.env_template),
                                       ppo_template=str(self.template), gs_root="/ply", model="/m", name="t",
                                       run_root=str(self.root / "x"), val_limit=1, trial=False,
                                       out=str(self.root / "x_out"), output_format="think"))

    def test_env_parser_reads_both_formats(self):
        from types import SimpleNamespace

        from vagen.envs.active_spatial.env import ActiveSpatialEnv

        stub = SimpleNamespace(_allowed_actions={"move_forward", "move_backward", "move_left", "move_right",
                                                 "turn_left", "turn_right"})
        parse = lambda text: ActiveSpatialEnv._default_parse_func(stub, text, action_sep="|", max_actions=5)  # noqa: E731
        a = parse("<action>turn_left|move_forward|</action>")
        t = parse("<think>The sofa is on the left.</think><action>turn_left|move_forward|</action>")
        self.assertEqual(a["actions"], ["turn_left", "move_forward"])
        self.assertEqual(t["actions"], a["actions"])
        self.assertTrue(a["format_correct"] and t["format_correct"])
        self.assertEqual(t["think"], "The sofa is on the left.")
        cut = parse("<think>a long thought that never ends")
        self.assertFalse(cut["format_correct"])
        self.assertEqual(cut["parse_error"], "missing_action_tag")
        # the share of answers with thinking text (group E): a non-empty think block, closed or cut off
        self.assertEqual([a["has_think"], t["has_think"], cut["has_think"]], [False, True, True])
        self.assertFalse(parse("<think> </think><action>turn_left|</action>")["has_think"])

    def test_answer_budget_only_where_asked(self):
        ppo = {f: yaml.safe_load((self.out[f] / "ppo.yaml").read_text()) for f in ("think_action", "e")}
        self.assertEqual(ppo["think_action"]["data"]["max_response_length"], TP.CONCAT_RESPONSE_LENGTH)
        self.assertEqual(ppo["e"]["data"]["max_response_length"], 6144)
        for p in ppo.values():
            p["data"].pop("train_files"), p["data"].pop("val_files"), p["data"].pop("max_response_length")
            for k in ("default_local_dir", "rollout_data_dir", "validation_data_dir"):
                p["trainer"].pop(k)
        self.assertEqual(ppo["think_action"], ppo["e"])
        rep = json.loads((self.out["e"] / "rl_config_report.json").read_text())
        self.assertEqual(rep["max_response_length"], 6144)
        with self.assertRaises(SystemExit):  # the budget is the concat mode's one-sequence budget
            TP.cmd_rl_config(Namespace(rows=str(self.sft / "rows.jsonl"), env_template=str(self.env_template),
                                       ppo_template=str(self.template), gs_root="/ply", model="/m", name="t",
                                       run_root=str(self.root / "y"), val_limit=1, trial=False,
                                       out=str(self.root / "y_out"), history_mode="no_concat", max_response_length=6144))

    def test_agent_loop_counts_turns_with_thinking_text(self):
        from vagen.agent_loop.gym_agent_loop import turn_has_think

        self.assertTrue(turn_has_think({"metrics": {"turn_metrics": {"has_think": True}}}))
        self.assertFalse(turn_has_think({"metrics": {"turn_metrics": {"action_is_valid": True}}}))
        self.assertFalse(turn_has_think(None))


if __name__ == "__main__":
    unittest.main()
