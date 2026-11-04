"""Decision 48 甲 of 2026-11-04: the next-frame prediction head's statistics on the held-out SFT conversations
(model_matrix/formal/nfp_heldout.py): the expansion as the concat path's, one conversation as compact ids, the
summary, the weight checks, the job script.  The scoring against the adapter's training forward is in
test_cambrian_nfp (AdapterForwardTest.test_heldout_scoring_matches_training)."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from model_matrix.formal import nfp_heldout as H

TOKENIZER = Path("/mnt/umm/users/yinbaiqiao/models_runs/chains/m_cambrian7b_formal_sft_20261026/lf_cformal/model/"
                 "checkpoint-770")


class ExpandTest(unittest.TestCase):
    def test_as_the_concat_path(self):
        from vagen.agent_loop.cambrian_concat import IMAGE_PAD_ID, IMAGE_TOKEN_ID, TOKENS_PER_IMAGE, _expand

        self.assertEqual((H.IMAGE_TOKEN_ID, H.IMAGE_PAD_ID, H.TOKENS_PER_IMAGE),
                         (IMAGE_TOKEN_ID, IMAGE_PAD_ID, TOKENS_PER_IMAGE))
        ids = [1, H.IMAGE_TOKEN_ID, 2, 3, H.IMAGE_TOKEN_ID, 4]
        self.assertEqual(H.expand(ids), _expand(ids, IMAGE_TOKEN_ID, IMAGE_PAD_ID, TOKENS_PER_IMAGE)[0])
        self.assertEqual(len(H.expand(ids)), 4 + 2 * 756)


@unittest.skipUnless((TOKENIZER / "tokenizer_config.json").exists(), "needs the SFT start's tokenizer")
class EpisodeTest(unittest.TestCase):
    def test_one_conversation(self):
        from transformers import AutoTokenizer

        from vagen.models.cambrian_processor import CambrianProcessorWrapper

        proc = CambrianProcessorWrapper(AutoTokenizer.from_pretrained(TOKENIZER))
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "[Initial Observation]:\n<image>\nTask"},
                {"role": "assistant", "content": "<action>turn_left|</action>"},
                {"role": "user", "content": "[Observation]:\n<image>\nAgain"},
                {"role": "assistant", "content": "<action>move_forward|</action>"}]
        ids = H.episode_ids(proc, msgs)
        img = [i for i, t in enumerate(ids) if t == H.IMAGE_TOKEN_ID]
        self.assertEqual(len(img), 2)
        between = proc.tokenizer.decode(ids[img[0] + 1:img[1]])
        self.assertIn("turn_left", between)  # the first answer lies between the two frames
        self.assertNotIn("move_forward", between)
        self.assertIn("move_forward", proc.tokenizer.decode(ids[img[1]:]))
        self.assertTrue(proc.tokenizer.decode(ids).rstrip().endswith("<|im_end|>"))  # no generation prompt
        self.assertEqual(H.answers(msgs), ["<action>turn_left|</action>", "<action>move_forward|</action>"])


class SummaryTest(unittest.TestCase):
    ROWS = [{"episode": 0, "k": 0, "mse": 1.0, "cos": 0.5, "copy_mse": 2.0, "copy_cos": 0.4, "loss": 0.15,
             "copy_loss": 0.24},
            {"episode": 0, "k": 1, "mse": 3.0, "cos": 0.7, "copy_mse": 0.0, "copy_cos": 0.0, "loss": 0.37,
             "copy_loss": 0.0},
            {"episode": 2, "k": 0, "mse": 2.0, "cos": 0.6, "copy_mse": 1.0, "copy_cos": 0.9, "loss": 0.26,
             "copy_loss": 0.19}]

    def test_summary(self):
        s = H.summarize(self.ROWS, episodes=4, one_frame=1)
        self.assertEqual((s["episodes"], s["one_frame_episodes"], s["pairs"], s["unchanged_frame_pairs"]),
                         (4, 1, 3, 1))
        self.assertAlmostEqual(s["mean"]["mse"], 2.0)
        self.assertAlmostEqual(s["mean"]["copy_loss"], 0.43 / 3)
        self.assertAlmostEqual(s["episode_mean"]["loss"], ((0.15 + 0.37) / 2 + 0.26) / 2)  # episode 0 first
        self.assertEqual({k: round(v, 6) for k, v in s["head_below_copy"].items()},
                         {"loss": round(1 / 3, 6), "mse": round(1 / 3, 6), "cos": round(1 / 3, 6)})
        self.assertEqual((s["quantiles"]["loss"]["q10"], s["quantiles"]["loss"]["q50"], s["quantiles"]["loss"]["q90"]),
                         (0.15, 0.26, 0.37))
        self.assertEqual(H.summarize([], 2, 2), {"episodes": 2, "one_frame_episodes": 2, "pairs": 0})


class WeightCheckTest(unittest.TestCase):
    def test_equal_different_missing(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            a, b = torch.randn(3, 2), torch.randn(4)
            save_file({"x.a": a}, str(d / "m-1.safetensors"))
            save_file({"x.b": b}, str(d / "m-2.safetensors"))
            (d / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x.a": "m-1.safetensors", "x.b": "m-2.safetensors"}}))
            params = {"x.a": torch.nn.Parameter(a.to(torch.bfloat16)), "x.b": torch.nn.Parameter(b + 1),
                      "x.c": torch.nn.Parameter(b)}
            self.assertEqual(H.weight_checks(params, d, ("x.a", "x.b", "x.c", "x.d")),
                             {"x.a": True, "x.b": False, "x.c": None, "x.d": None})

    def test_the_checked_keys_are_in_csnfps_checkpoint_index(self):
        index = Path("/mnt/umm/users/yinbaiqiao/models_runs/chains/m_cambrian7b_formal_rl_20261103_nfp/rl_run_csnfp/"
                     "checkpoints/global_step_50/actor/huggingface/model.safetensors.index.json")
        if not index.exists():
            self.skipTest("needs csnfp's checkpoint")
        self.assertTrue(set(H.CHECKED) <= set(json.loads(index.read_text())["weight_map"]))


class JobScriptTest(unittest.TestCase):
    def test_parses(self):
        sh = Path(H.__file__).with_suffix(".sh")
        self.assertEqual(subprocess.run(["bash", "-n", str(sh)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
