"""The QA RL pilot (the user's instruction of 2026-11-03, section 2): the reward (P0's scoring, unreadable answers),
the single-turn P0QA env, and qa_rl's rl-config."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from data_gen.interiorgs_v10 import qa_rl as R


def question(bench, answer="B", answer_type="mc", options=("x", "y", "z", "w"), family="orig"):
    return {"qid": f"q_{bench}", "benchmark": bench, "group": "S1", "answer_type": answer_type, "scene_id": "0001_1",
            "variants": {family: {"answer": answer, "options": list(options)}}}


def row(bench, answer="B", reply_gold="B", answer_type="mc", images=(), prompt=None, family="orig"):
    q = question(bench, answer, answer_type, family=family)
    return {"qid": q["qid"], "benchmark": bench, "group": "S1", "family": family, "answer_type": answer_type,
            "scene_id": "0001_1", "prompt": prompt if prompt is not None else "<image>" * len(images) + "\nQ?",
            "images": list(images), "answer": reply_gold, "question": q}


class QARewardTest(unittest.TestCase):
    def test_reward_and_readable(self):
        cases = [  # (benchmark, reply, score, readable)
            ("MMSI-Bench", "B", 1.0, True), ("MMSI-Bench", "C", 0.0, True), ("MMSI-Bench", "no idea", 0.0, False),
            ("MindCube", "<answer>B. Behind</answer>", 1.0, True), ("MindCube", "Behind the sofa", 0.0, False),
            ("BLINK", "(B)", 1.0, True), ("EmbSpatial-Bench", "B) basket", 1.0, True),
            ("ViewSpatial-Bench", "A. right", 0.0, True), ("3DSRBench", "", 0.0, False),
        ]
        for bench, reply, value, ok in cases:
            self.assertEqual(R.reward(row(bench), reply), (value, ok), (bench, reply))
        numeric = row("VSI-Bench", answer="2.0", answer_type="numeric")
        self.assertEqual(R.reward(numeric, "2.0"), (1.0, True))
        self.assertEqual(R.reward(numeric, "about two"), (0.0, False))
        self.assertTrue(0.0 < R.reward(numeric, "2.3")[0] < 1.0)
        self.assertEqual(R.check([row("MMSI-Bench"), row("BLINK", reply_gold="(B)")])["problems"], {})
        self.assertIn("gold", R.check([row("MMSI-Bench", reply_gold="C")])["problems"])

    def test_env_and_rl_config(self):
        from vagen.envs.p0_qa import P0QAEnv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "v.png"
            Image.new("RGB", (32, 32)).save(img)
            rows = root / "rows.jsonl"
            rows.write_text("".join(json.dumps(r) + "\n" for r in (row("MMSI-Bench", images=[str(img)]),
                                                                   row("BLINK", images=[str(img), str(img)]))))
            env = P0QAEnv({"jsonl_path": str(rows)})
            obs, info = asyncio.run(env.reset(1))
            self.assertEqual(obs["obs_str"], "<image><image>\nQ?")
            self.assertEqual(len(obs["multi_modal_input"]["<image>"]), 2)
            self.assertEqual((info["task_type"], info["task_id"]), ("BLINK", "q_BLINK"))
            self.assertEqual(asyncio.run(env.system_prompt()), {})  # no system message, as in the SFT records
            _, reward, done, info = asyncio.run(env.step("(B)"))
            self.assertEqual((reward, done, info["success"], info["metrics"]["turn_metrics"]["action_is_valid"]),
                             (1.0, True, True, True))
            asyncio.run(env.reset(0))
            _, reward, done, info = asyncio.run(env.step("I cannot tell"))
            self.assertEqual((reward, info["success"], info["metrics"]["turn_metrics"]["action_is_valid"]),
                             (0.0, False, False))
            rep = R.rl_config(rows, rows, str(root / "model"), "qrl_t", root / "run", root / "rl", seed=7)
            ppo = yaml.safe_load((root / "rl" / "ppo.yaml").read_text())
            self.assertEqual((ppo["data"]["train_batch_size"], ppo["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"],
                              ppo["data"]["max_response_length"], ppo["actor_rollout_ref"]["rollout"]["n"],
                              ppo["algorithm"]["adv_estimator"], ppo["filter"]["name"], ppo["data"]["seed"],
                              ppo["trainer"]["concat_multi_turn"], ppo["trainer"]["save_freq"]),
                             (64, 32, 256, 8, "grpo", "mixed_outcome", 7, True, 50))
            env_yaml = yaml.safe_load((root / "rl" / "train_env.yaml").read_text())["envs"][0]
            self.assertEqual((env_yaml["name"], env_yaml["max_turns"], env_yaml["n_envs"], env_yaml["seed_list"]),
                             ("P0QA", 1, 2, [0, 1]))
            self.assertEqual(rep["train"], 2)


if __name__ == "__main__":
    unittest.main()
