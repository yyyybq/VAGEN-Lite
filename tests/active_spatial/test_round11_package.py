"""Round 11, the optimisation package of InteriorGS v10 RL: the mixed-outcome group filter with refill from extra
candidates, the per-episode share of unparseable answers, and GRPO in the concat (one sequence per episode) mode."""

import importlib.util
import unittest

HAVE_TRAINING_ENV = all(importlib.util.find_spec(m) is not None for m in ("ray", "omegaconf", "torch", "tensordict"))


@unittest.skipUnless(HAVE_TRAINING_ENV, "needs the training env (ray, omegaconf, torch, tensordict)")
class MixedOutcomeFilterTest(unittest.TestCase):
    def batch(self, outcomes, order=None):
        """One row per episode; outcomes = [[ok, ok, ...] per task]; uids t0, t1, ... in draw order."""
        import numpy as np
        import torch
        from verl import DataProto

        uids, succ = [], []
        for t, oks in enumerate(outcomes):
            for ok in oks:
                uids.append(f"t{t}")
                succ.append(float(ok))
        n = len(uids)
        d = DataProto.from_dict(tensors={"advantages": torch.ones(n, 3), "ids": torch.arange(n).unsqueeze(1)},
                                non_tensors={"uid": np.array(uids, dtype=object), "traj_success": np.array(succ)})
        d.meta_info["uid_draw_order"] = order or {f"t{t}": t for t in range(len(outcomes))}
        return d

    def test_nominal_batch_refilled_from_extra_candidates(self):
        from vagen.custom_filter.filter import FILTER_REGISTRY

        f = FILTER_REGISTRY["mixed_outcome"]
        mixed, all_ok, all_fail = [True, False, True, False], [True] * 4, [False] * 4
        # 3 nominal tasks (one mixed, one all success, one all fail), 3 extra candidates (two mixed)
        d = self.batch([mixed, all_ok, all_fail, all_fail, mixed, mixed])
        metrics = {}
        out, metrics = f(d, metrics, target_groups=3)
        kept = sorted(set(out.non_tensor_batch["uid"]))
        self.assertEqual(kept, ["t0", "t4", "t5"])  # t1, t2 dropped; t4, t5 refill them
        self.assertEqual(len(out), 12)
        self.assertEqual({k: metrics[f"filter/mixed/{k}"] for k in ("candidates", "nominal", "kept", "dropped_all_success",
                                                                     "dropped_all_fail", "refilled", "shortfall")},
                         {"candidates": 6, "nominal": 3, "kept": 3, "dropped_all_success": 1, "dropped_all_fail": 1,
                          "refilled": 2, "shortfall": 0})
        self.assertAlmostEqual(metrics["filter/mixed/mixed_share"], 0.5)
        # the draw order decides what is nominal, not the row order (rows get reordered for balancing)
        d = self.batch([mixed, mixed, all_ok, mixed], order={"t3": 0, "t2": 1, "t1": 2, "t0": 3})
        out, metrics = f(d, {}, target_groups=2)
        self.assertEqual(sorted(set(out.non_tensor_batch["uid"])), ["t1", "t3"])  # t3 nominal, t2 dropped, t1 refills
        # too few mixed groups: a shortfall, and the advantages of what is kept stay as they are
        out, metrics = f(self.batch([all_ok, mixed, all_fail]), {}, target_groups=2)
        self.assertEqual((metrics["filter/mixed/kept"], metrics["filter/mixed/shortfall"]), (1, 1))
        self.assertEqual(float(out.batch["advantages"].sum()), 12.0)
        # no mixed group at all: the nominal batch with zero advantages (only the entropy and KL terms act)
        out, metrics = f(self.batch([all_ok, all_fail, all_ok]), {}, target_groups=2)
        self.assertEqual(sorted(set(out.non_tensor_batch["uid"])), ["t0", "t1"])
        self.assertEqual(float(out.batch["advantages"].abs().sum()), 0.0)
        self.assertEqual((metrics["filter/mixed/empty_fallback"], metrics["filter/mixed/kept"]), (1.0, 0))


@unittest.skipUnless(HAVE_TRAINING_ENV, "needs the training env")
class InvalidTurnAndGrpoModeTest(unittest.TestCase):
    def test_turn_invalid(self):
        from vagen.agent_loop.gym_agent_loop import turn_invalid

        self.assertTrue(turn_invalid({"metrics": {"turn_metrics": {"action_is_valid": False}}}))
        self.assertFalse(turn_invalid({"metrics": {"turn_metrics": {"action_is_valid": True}}}))
        self.assertFalse(turn_invalid({"traj_success": False}))  # no turn metrics (e.g. an env error): not counted
        self.assertFalse(turn_invalid(None))

    def test_grpo_in_the_concat_mode(self):
        from omegaconf import OmegaConf

        from vagen.interiorgs_v10_ppo import history_mode

        cfg = lambda adv, concat: OmegaConf.create({"algorithm": {"adv_estimator": adv}, "trainer": {"concat_multi_turn": concat}})  # noqa: E731
        self.assertEqual(history_mode(cfg("grpo", True)), "concat")
        self.assertEqual(history_mode(cfg("gae", True)), "concat")
        with self.assertRaises(ValueError):
            history_mode(cfg("grpo", False))  # GRPO in the per-turn mode is not supported


if __name__ == "__main__":
    unittest.main()
