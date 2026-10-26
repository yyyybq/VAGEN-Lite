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


class QualityTest(unittest.TestCase):
    def test_answers_and_blocks(self):
        from model_matrix.formal import quality as Q

        obs = ("<|im_start|>user\n[Observation]:\n<image>\nRespond with only the action:\n"
               "<action>move_forward|turn_left|</action>\n<|im_end|>\n")
        out = ("<action>move_backward|</action><|im_end|>" + obs + "<|im_start|>assistant\n<action>action|</action><|im_end|>"
               + obs + "<|im_start|>assistant\n<action>turn_left|turn_left|turn_left|turn_left|turn_left|turn_left|</action>"
               "<|im_end|>")
        self.assertEqual(Q.answers(out), ["<action>move_backward|</action>", "<action>action|</action>",
                                          "<action>" + "turn_left|" * 6 + "</action>"])  # the example is not an answer
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "rl_run_q3d"
            (run / "rollout_data").mkdir(parents=True)
            for u in (1, 2, 26):
                (run / "rollout_data" / f"{u}.jsonl").write_text(json.dumps({"output": out}) + "\n")
            got = Q.block_quality(run)
        self.assertEqual(sorted(got), ["1-25", "26-50"])
        b = got["1-25"]
        self.assertEqual((b["updates"], b["answers"], b["unreadable"], b["names_outside"]), (2, 6, 4, 2))
        self.assertEqual(b["names_outside_top"], {"action": 2})
        self.assertEqual(b["forms"], {"format_example_copied": 2, "too_many_or_other": 2})  # evaluation's own forms
        self.assertIn("| q3d | 1-25（2 次） | 6 |", Q.report({"q3d": got}))


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

    def test_rl_plan_is_the_main_lines_formal_pair(self):
        from model_matrix.formal import plans as P

        plan = P.rl_plan("t", "/m", "qwen3_vl", "/sft/e8", "q3formal", "q3", "abc", "V4")
        self.assertEqual((plan["rl_start_model"], plan["endpoint"], plan["run_pools"]),
                         ("/sft/e8", 300, {"q3s0": "h800", "q3d": "zoetrope"}))
        rl = plan["rl"]
        self.assertEqual((rl["standard"], rl["seed"], rl["task_pool"], rl["format_example"]),
                         ("v10_p1_S0", 20261019, "reserve", "action_names"))
        self.assertEqual({k: rl["guards"][k] for k in ("kl_max", "invalid_max", "invalid_window")}, {"kl_max": 0.2,
                         "invalid_max": 0.1, "invalid_window": 10})
        self.assertEqual(rl["guards"]["keep_fill_min"], 1.0)
        self.assertEqual(plan["rl_runs"]["q3s0"], {})
        self.assertEqual(plan["rl_runs"]["q3d"]["rl"]["keep_std_threshold"], 0.3318)
        self.assertEqual(plan["eval"], {"every": 50, "options": [], "stop_invalid": {"set": "val", "max": 0.10}})
        self.assertNotIn("retain", plan)
        self.assertNotIn("trainer_attn", plan)  # eager, as the main line (decision 20)
        self.assertNotIn("logprob_audit", rl)
        with self.assertRaises(SystemExit):
            P.rl_plan("t", "/m", "qwen3_vl", "/s", "g", "q3", "abc", "V9")
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "v.json"
            f.write_text(json.dumps({"variants": {"dv1": {"variant": "V1", "summary": {"success_pooled": 0.46}},
                                                  "dv4": {"variant": "V4", "summary": {"success_pooled": 0.48}}}}))
            self.assertEqual(P.best_variant(f), "V4")

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
