"""The model line's formal runs: the plans and the SFT acceptance (step 3, the user's decisions of 2026-10-26)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def episode(answers, success=False, sample=None):
    traj = [{"answer": a, "valid": a.startswith("<action>move") or a.startswith("<action>turn"), "entropy": 0.5,
             "tokens": 4} for a in answers]
    r = {"index": 0, "task_id": "t", "task_type": "fov_inclusion", "d_star": 3, "success": success,
         "actions_to_success": 3 if success else None, "turns": len(answers), "actions": len(answers), "collisions": 0,
         "invalid_turns": sum(not t["valid"] for t in traj), "trajectory": traj, "end_reason": "max_turns"}
    if sample is not None:
        r["sample"] = sample
    return r


class AcceptanceTest(unittest.TestCase):
    def test_both_runs_against_the_rule(self):
        from model_matrix.formal import acceptance as A

        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp)
            good = [episode(["<action>move_forward|turn_left|</action>"] * 50, sample=s) for s in range(4)]
            bad = [episode(["<action>move_forward|</action>"] * 98 + ["<action>look_down|</action>", "no tag"])]
            for name, rows in (("m_s4", good), ("m_greedy", bad)):
                (ev / name).mkdir()
                (ev / name / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                (ev / name / "done.json").write_text("{}")
            out = A.accept(ev, "m")
            s, g = out["sampled"], out["greedy"]
            self.assertEqual((s["turns"], s["invalid_rate"], s["names_outside"], s["passes"], s["samples"]),
                             (200, 0.0, 0, True, 4))
            self.assertEqual((g["turns"], g["invalid_rate"], g["names_outside"], g["names_outside_top"]),
                             (100, 0.02, 1, {"look_down": 1}))
            self.assertEqual((g["readable_ok"], g["names_ok"], g["passes"]), (False, False, False))
            self.assertEqual(g["unreadable_forms"], {"made_up_name": 1, "no_action_tag": 1})
            self.assertEqual(s["entropy"], 0.5)
            self.assertTrue((ev / "acceptance_m.md").is_file())
            (ev / "m_greedy" / "done.json").unlink()
            with self.assertRaises(SystemExit):  # incomplete results are not judged
                A.accept(ev, "m")


class PlansTest(unittest.TestCase):
    def test_formal_sft_plan(self):
        from model_matrix.formal import plans as P

        plan = P.sft_plan("t", "/m", "qwen3_vl", "q3formal", "abc")
        self.assertEqual((plan["sft_only"], plan["sft_epochs"], plan["select_by"], plan["model_type"]),
                         (True, 10, "val", "qwen3_vl"))
        self.assertEqual(plan["sft_groups"], {"q3formal": "sft_trajectories.jsonl"})  # C0's 11,095 conversations
        self.assertNotIn("retain", plan)  # no checkpoint deleted
        s4, greedy = plan["eval"]["variants"]
        self.assertEqual((s4["suffix"], s4["sets"], s4["models"]), ("_s4", ["accept"], ["sft_pick"]))
        self.assertEqual(s4["options"][:6], ["--samples", "4", "--temperature", "0.8", "--top-p", "0.92"])
        for v in (s4, greedy):  # variants pass options as given: the format example and the entropy are explicit
            self.assertIn("--entropy", v["options"])
            self.assertEqual(v["options"][-2:], ["--format-example", "action_names"])
        self.assertNotIn("--samples", greedy["options"])
        self.assertEqual(plan["eval"]["extra_sets"], {"accept": str(P.ACCEPT)})

    def test_accept_epoch_plans(self):
        from model_matrix.formal import plans as P

        with tempfile.TemporaryDirectory() as tmp:
            chain = Path(tmp) / "m_x_formal_sft_d"
            for s in (154, 308, 1540):
                (chain / "lf_g" / "model" / f"checkpoint-{s}").mkdir(parents=True)
            ck = P.epoch_checkpoints(chain, "g")
        self.assertEqual({e: p.name for e, p in ck.items()}, {1: "checkpoint-154", 2: "checkpoint-308", 3: "checkpoint-1540"})
        plans = P.accept_epoch_plans("d", "abc", {"m_q_accept_epochs_d": ("qwen3_vl", "g", 2, ck)})
        q = plans["m_q_accept_epochs_d"]
        self.assertEqual((q["eval_only"], q["model_type"], q["eval_sets"]), (True, "qwen3_vl", {"accept": str(P.ACCEPT)}))
        self.assertEqual(sorted(q["eval"]["models"]), ["sft_g_e1_greedy", "sft_g_e3_greedy"])  # not the picked e2
        self.assertEqual(q["eval"]["models"]["sft_g_e1_greedy"]["options"], P.ACCEPT_GREEDY)
        c0 = plans["m_qwen25vl7b_c0_accept_d"]
        self.assertNotIn("model_type", c0)
        self.assertNotIn("--format-example", " ".join(c0["eval"]["models"]["c0_sft_formal_e9_s4"]["options"]))
        self.assertIn("--entropy", c0["eval"]["models"]["c0_sft_formal_e9_greedy"]["options"])

    def test_sdpa_control_is_the_smoke_with_sdpa_and_token_logs(self):
        from model_matrix.formal import plans as P
        from model_matrix.smoke import plans as S

        smoke = S.smoke("m_qwen3vl8b_smoke_20261024", S.QWEN3VL, "qwen3_vl", "q3smoke", "q3v4smoke", "abc")
        ctl = P.sdpa_control_plan("ctl", "abc", "/sft/checkpoint-7")
        self.assertEqual(ctl["rl_runs"], {"q3v4sdpa": smoke["rl_runs"]["q3v4smoke"]})  # the same V4 arm
        self.assertEqual({k: v for k, v in ctl["rl"].items() if k != "logprob_tokens"}, smoke["rl"])  # same seed
        self.assertEqual((ctl["rl"]["logprob_tokens"], ctl["trainer_attn"], ctl["rl_start_model"], ctl["endpoint"]),
                         (True, "sdpa", "/sft/checkpoint-7", 10))
        self.assertEqual(ctl["sft_groups"], smoke["sft_groups"])
        diag = P.diagnostic_plans("d", "abc", "/q25/checkpoint-7", "/ctl/global_step_10/actor/huggingface")
        q25, q3 = diag["m_qwen25vl7b_tokens_d"], diag["m_qwen3vl8b_restart_d"]
        self.assertEqual((q25["rl_start_model"], q25["endpoint"], q25["rl"]["checkpoint_every"]),
                         ("/q25/checkpoint-7", 3, 3))
        self.assertNotIn("trainer_attn", q25)  # eager, the main line's
        self.assertNotIn("model_type", q25)
        self.assertEqual((q3["rl_start_model"], q3["endpoint"], q3["trainer_attn"], q3["model_type"]),
                         ("/ctl/global_step_10/actor/huggingface", 2, "sdpa", "qwen3_vl"))
        for p in (q25, q3):
            self.assertNotIn("eval", p)  # RL only
            self.assertTrue(p["rl"]["logprob_tokens"] and p["rl"]["logprob_audit"])
            self.assertEqual(p["rl"]["seed"], smoke["rl"]["seed"])
            self.assertEqual(list(p["rl_runs"].values())[0], smoke["rl_runs"]["q3v4smoke"])  # the V4 arm
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "progress.json").write_text(json.dumps(
                {"steps": {"pick_sft": {"decision": {"model": "/sft/checkpoint-7"}}}}))
            self.assertEqual(P.picked_model(Path(tmp)), "/sft/checkpoint-7")


if __name__ == "__main__":
    unittest.main()
