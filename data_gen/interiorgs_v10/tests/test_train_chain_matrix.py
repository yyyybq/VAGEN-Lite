"""train_chain.py, the experiment-matrix fields (branch transition-qa, docs/transition_qa/CHANGES.md section 5):
sft_lf_dirs, start_model, rl output_format / max_response_length, guards invalid_until, eval rl_prompt_format."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from data_gen.interiorgs_v10 import train_chain as T


def write_lf(folder: Path, epochs=10, output_dir=None, save="epoch") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "train.yaml").write_text(yaml.safe_dump({"output_dir": str(output_dir or folder / "model"),
                                                       "num_train_epochs": epochs, "save_strategy": save,
                                                       "dataset": "P0_train,P1_train,P2_train"}))
    return folder


class MatrixFieldsTest(unittest.TestCase):
    def test_sft_lf_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lf = write_lf(root / "mix_D")
            plan = {"tr": str(root / "tr"), "sft_dir": str(root / "sft"), "endpoint": 300, "sft_epochs": 10,
                    "sft_groups": {"mix": "sft_trajectories.jsonl"}, "sft_lf_dirs": {"mix": str(lf)}}
            chain = T.Chain(plan, sleep=lambda s: None)
            self.assertEqual(chain.lf_dir("mix"), lf)
            self.assertTrue(chain.lf_given())
            calls = []
            chain.sh = lambda args, **k: calls.append(args)
            chain.render()  # no frames.jsonl needed, no render job
            chain.sft_data()  # no sft-records (it would rewrite sft_dir), no prepare script
            self.assertEqual(calls, [])
            steps = chain.state["steps"]
            self.assertIn("sft_lf_dirs", steps["render"]["note"])
            self.assertEqual(steps["sft_data"]["lf_dirs"]["mix"]["dir"], str(lf))
            for bad in ({"epochs": 5}, {"output_dir": root / "elsewhere"}, {"save": "steps"}):
                write_lf(lf, **bad)
                with self.assertRaises(T.ChainError):
                    chain.sft_data()

    def test_start_model_skips_the_sft_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"tr": str(root / "tr"), "sft_dir": str(root), "endpoint": 300, "start_model": "/models/base_7b"}
            chain = T.Chain(plan, sleep=lambda s: None)
            calls = []
            for step in T.STEPS:
                setattr(chain, step, lambda step=step: calls.append(step))
            self.assertEqual(chain.run(), 0)
            self.assertEqual(calls, ["check_code", "rl_config", "rl", "rl_extend", "evaluate"])
            for s in T.SFT_STEPS:
                self.assertEqual(chain.state["steps"][s]["note"], "not planned (start_model)")
            self.assertEqual(chain.sft_model(), Path("/models/base_7b"))

    def test_start_model_from_another_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d_tr = root / "m2_d"
            plan = {"tr": str(root / "m2_e"), "sft_dir": str(root), "endpoint": 300,
                    "start_model": {"from_chain": str(d_tr)}}
            chain = T.Chain(plan, sleep=lambda s: None)
            with self.assertRaises(T.ChainError):  # D has not run
                chain.sft_model()
            d_tr.mkdir()
            (d_tr / "progress.json").write_text(json.dumps({"steps": {"pick_sft": {"state": "running"}}}))
            with self.assertRaises(T.ChainError):  # D has not picked yet
                chain.sft_model()
            (d_tr / "progress.json").write_text(json.dumps(
                {"steps": {"pick_sft": {"state": "done", "decision": {"best": "sft_mix_e7", "model": "/d/ckpt-7"}}}}))
            self.assertEqual(chain.sft_model(), Path("/d/ckpt-7"))
            chain.state["rl_model_override"] = "/fallback"  # a signal-check fallback still wins
            self.assertEqual(chain.sft_model(), Path("/fallback"))

    def test_group_e_options_reach_rl_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = T.Chain({"tr": tmp, "sft_dir": str(root / "sft"), "endpoint": 300, "gs_root": "/ply",
                             "start_model": "/d/ckpt-7",
                             "rl": {"standard": "v10_p1_S0", "seed": 20261019, "output_format": "think_action",
                                    "max_response_length": 6144},
                             "rl_runs": {"p1s0": {}, "p1d": {"rl": {"standard": "v10_p1_D"}}}}, sleep=lambda s: None)
            calls = {}
            chain.sh = lambda args, **k: calls.__setitem__(args[args.index("--out") + 1], args)
            chain.rl_config()
            for args in calls.values():
                self.assertEqual(args[args.index("--output-format") + 1], "think_action")
                self.assertEqual(args[args.index("--max-response-length") + 1], "6144")
                self.assertEqual(args[args.index("--model") + 1], "/d/ckpt-7")
            self.assertEqual(chain.state["steps"]["rl_config"]["start_model"], "/d/ckpt-7")
            plain = T.Chain({"tr": tmp, "sft_dir": str(root / "sft"), "endpoint": 300, "gs_root": "/ply",
                             "rl": {"standard": "v10_p1_S0"}}, sleep=lambda s: None)
            plain.state["steps"]["pick_sft"]["decision"] = {"model": "/lf/e9"}
            got = []
            plain.sh = lambda args, **k: got.append(args)
            plain.rl_config()
            self.assertNotIn("--output-format", got[0])
            self.assertNotIn("--max-response-length", got[0])

    def test_invalid_until(self):
        guards = {"invalid_max": 0.20, "invalid_window": 10, "invalid_until": 50, "drop_window": 1000}
        line = lambda s, inv: f"(Runner) step:{s} - episode/success_rate:0.5 - transition/invalid_action_rate/mean:{inv}\n"  # noqa: E731
        early = "".join(line(s, 0.3 if s > 40 else 0.0) for s in range(1, 51))
        self.assertIn("unparseable answers 0.300", T.guard_reason(early, guards))   # updates 41-50: inside the window
        late = "".join(line(s, 0.3 if s > 50 else 0.0) for s in range(1, 61))
        self.assertIsNone(T.guard_reason(late, guards))                               # after update 50: not checked
        self.assertIn("unparseable answers", T.guard_reason(late, {**guards, "invalid_until": None}))

    def test_rl_checkpoints_evaluated_with_the_prompt_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"tr": tmp, "sft_dir": tmp, "endpoint": 100, "eval_sets": {"val": str(root / "val")},
                    "rl_runs": {"p1s0": {}}, "eval": {"every": 50, "rl_prompt_format": "free_think",
                                                      "models": {"base": {"model": "/m/base"}}}}
            chain = T.Chain(plan, sleep=lambda s: None)
            got = {}
            chain.eval_jobs = lambda step, j: got.update(j)
            chain.tables = lambda: None
            chain.evaluate()
            self.assertEqual(got["val__p1s0_050"][3], "free_think")
            self.assertEqual(got["val__p1s0_100"][3], "free_think")
            self.assertIsNone(got["val__base"][3])  # a listed model keeps its own format


if __name__ == "__main__":
    unittest.main()
