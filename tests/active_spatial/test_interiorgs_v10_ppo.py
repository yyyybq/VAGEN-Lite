"""vagen/interiorgs_v10_ppo.py: which PPO checkpoints are kept (decision of 2026-10-14)."""

import importlib.util
import unittest

HAVE_RAY = importlib.util.find_spec("ray") is not None and importlib.util.find_spec("omegaconf") is not None


@unittest.skipUnless(HAVE_RAY, "needs ray and omegaconf (the training env)")
class RetainedStepsTest(unittest.TestCase):
    def test_every_50th_and_the_latest_two_others(self):
        from vagen.interiorgs_v10_ppo import retained_steps

        saved = list(range(25, 151, 25))  # a checkpoint every 25 updates up to 150
        self.assertEqual(retained_steps(saved, 50, 2), {50, 100, 150, 75, 125})
        self.assertEqual(retained_steps(saved[:3], 50, 2), {25, 50, 75})
        self.assertEqual(retained_steps(range(25, 251, 25), 50, 2), {50, 100, 150, 200, 250, 175, 225})
        self.assertEqual(retained_steps([1, 2, 3], 2, 1), {2, 3})  # the one-card resume check
        self.assertEqual(retained_steps([1, 2, 3], 0, 1), {3})

    def test_prune_to_hf_keeps_only_the_weights(self):
        import tempfile
        from pathlib import Path

        from vagen.interiorgs_v10_ppo import prune_to_hf

        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "global_step_50"
            for rel in ("actor/huggingface/model.safetensors", "actor/model_world_size_8_rank_0.pt",
                        "actor/optim_world_size_8_rank_0.pt", "critic/model_world_size_8_rank_0.pt", "data.pt"):
                (ck / rel).parent.mkdir(parents=True, exist_ok=True)
                (ck / rel).write_text("x")
            (ck / "COMPLETE").write_text("")
            removed = prune_to_hf(ck)
            self.assertEqual(sorted(removed), ["actor/model_world_size_8_rank_0.pt", "actor/optim_world_size_8_rank_0.pt",
                                               "critic", "data.pt"])
            left = sorted(str(p.relative_to(ck)).replace("\\", "/") for p in ck.rglob("*") if p.is_file())
            self.assertEqual(left, ["COMPLETE", "actor/huggingface/model.safetensors"])
            bare = Path(tmp) / "global_step_75"
            (bare / "critic").mkdir(parents=True)
            self.assertEqual(prune_to_hf(bare), [])  # no actor/huggingface: left alone
            self.assertTrue((bare / "critic").is_dir())

    def test_history_mode_from_the_config_pair(self):
        from omegaconf import OmegaConf

        from vagen.interiorgs_v10_ppo import history_mode

        def cfg(adv, concat):
            return OmegaConf.create({"algorithm": {"adv_estimator": adv}, "trainer": {"concat_multi_turn": concat}})

        self.assertEqual(history_mode(cfg("no_concat_gae", False)), "no_concat")  # rounds 6-9 and no_concat history
        self.assertEqual(history_mode(cfg("gae", True)), "concat")               # one sequence per episode
        for bad in (("gae", False), ("no_concat_gae", True), ("grpo", True)):
            with self.assertRaises(ValueError):
                history_mode(cfg(*bad))


if __name__ == "__main__":
    unittest.main()
