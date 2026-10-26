"""Round 12, the dense-reward first batch (D-keep): the group filter keep_by_return_std."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAVE_TRAINING_ENV = all(importlib.util.find_spec(m) is not None for m in ("ray", "omegaconf", "torch", "tensordict"))


@unittest.skipUnless(HAVE_TRAINING_ENV, "needs the training env (ray, omegaconf, torch, tensordict)")
class KeepByReturnStdTest(unittest.TestCase):
    def batch(self, tasks):
        """tasks = [(family, [(success, return) per episode])]; uids t0, t1, ... in draw order, task ids id0, ..."""
        import numpy as np
        import torch
        from verl import DataProto

        uids, succ, fam, ids, rets = [], [], [], [], []
        for t, (family, episodes) in enumerate(tasks):
            for ok, ret in episodes:
                uids.append(f"t{t}")
                succ.append(float(ok))
                fam.append(family)
                ids.append(f"id{t}")
                rets.append(ret)
        n = len(uids)
        rewards = torch.zeros(n, 3)
        rewards[:, -1] = torch.tensor(rets)
        d = DataProto.from_dict(
            tensors={"advantages": torch.ones(n, 3), "token_level_rewards": rewards},
            non_tensors={"uid": np.array(uids, dtype=object), "traj_success": np.array(succ),
                         "task_type": np.array(fam, dtype=object), "task_id": np.array(ids, dtype=object)})
        d.meta_info["uid_draw_order"] = {f"t{t}": t for t in range(len(tasks))}
        return d

    def test_fill_with_spread_all_fail_tasks(self):
        import torch

        from vagen.custom_filter.filter import FILTER_REGISTRY

        f = FILTER_REGISTRY["keep_by_return_std"]
        mixed = ("fov_inclusion", [(1, 5.0), (0, 0.0), (1, 5.0), (0, 0.0)])
        wide = ("equidistance", [(0, 0.6), (0, 0.0), (0, 0.6), (0, 0.0)])        # spread 0.346
        wider = ("occlusion_alignment", [(0, 1.0), (0, 0.0), (0, 1.0), (0, 0.0)])  # spread 0.577
        narrow = ("fov_inclusion", [(0, 0.0), (0, -0.01), (0, 0.0), (0, -0.01)])  # spread 0.006: penalty noise
        all_ok = ("screen_occupancy", [(1, 5.0)] * 4)
        with tempfile.TemporaryDirectory() as tmp:
            rows = Path(tmp) / "train.jsonl"
            rows.write_text("".join(json.dumps({"task_id": f"id{t}", "difficulty": {"d_star": d}}) + "\n"
                                    for t, d in ((1, 6), (4, 9), (2, 2))))
            # 3 places: the mixed task, then the two all-fail tasks above the threshold, the wider one first
            d = self.batch([mixed, wide, narrow, all_ok, wider])
            out, m = f(d, {}, target_groups=3, std_threshold=0.2, std_floor=1.0, rows=str(rows))
        kept = list(dict.fromkeys(out.non_tensor_batch["uid"]))
        self.assertEqual(sorted(kept), ["t0", "t1", "t4"])
        self.assertEqual((m["filter/mixed/kept"], m["filter/keep/filled"], m["filter/keep/kept_total"]), (1, 2, 3))
        self.assertEqual((m["filter/keep/all_fail"], m["filter/keep/all_fail_above_threshold"]), (3, 2))
        self.assertEqual(m["filter/keep/filled_below_floor"], 2)  # spreads 0.35 and 0.58 under the floor 1.0
        self.assertAlmostEqual(m["filter/keep/filled_std_max"], float(torch.std(torch.tensor([1.0, 0, 1, 0]))), places=5)
        self.assertEqual((m["filter/keep/filled_family/equidistance"], m["filter/keep/filled_family/occlusion_alignment"]),
                         (1, 1))
        self.assertAlmostEqual(m["filter/keep/filled_min_steps_mean"], 7.5)  # d* 6 and 9; the narrow task is not filled
        # the mixed task keeps its advantages; a filled one is scaled by spread / max(spread, floor)
        adv = {u: float(out.batch["advantages"][i, 0]) for i, u in enumerate(out.non_tensor_batch["uid"])}
        self.assertEqual(adv["t0"], 1.0)
        self.assertAlmostEqual(adv["t1"], (0.34641 + 1e-6) / (1.0 + 1e-6), places=4)
        self.assertAlmostEqual(adv["t4"], (0.57735 + 1e-6) / (1.0 + 1e-6), places=4)
        # a spread above the floor is left as it is
        out, m = f(self.batch([mixed, wider]), {}, target_groups=2, std_threshold=0.2, std_floor=0.5)
        self.assertEqual(float(out.batch["advantages"].min()), 1.0)
        self.assertEqual(m["filter/keep/filled_below_floor"], 0)

    def test_enough_mixed_tasks_and_fallback(self):
        from vagen.custom_filter.filter import FILTER_REGISTRY

        f = FILTER_REGISTRY["keep_by_return_std"]
        mixed = ("fov_inclusion", [(1, 5.0), (0, 0.0)])
        wide = ("equidistance", [(0, 1.0), (0, 0.0)])
        narrow = ("equidistance", [(0, 0.0), (0, -0.01)])
        all_ok = ("fov_inclusion", [(1, 5.0), (1, 5.0)])
        out, m = f(self.batch([mixed, wide, mixed]), {}, target_groups=2, std_threshold=0.2, std_floor=1.0)
        self.assertEqual(sorted(set(out.non_tensor_batch["uid"])), ["t0", "t2"])  # the refill comes before any fill
        self.assertEqual((m["filter/keep/filled"], m["filter/mixed/refilled"]), (0, 1))
        # nothing mixed, nothing above the threshold: the nominal batch with zero advantages, as mixed_outcome
        out, m = f(self.batch([all_ok, narrow, all_ok]), {}, target_groups=2, std_threshold=0.2, std_floor=1.0)
        self.assertEqual(sorted(set(out.non_tensor_batch["uid"])), ["t0", "t1"])
        self.assertEqual(float(out.batch["advantages"].abs().sum()), 0.0)
        self.assertEqual((m["filter/mixed/empty_fallback"], m["filter/keep/kept_total"]), (1.0, 0))
        # nothing mixed but one task above the threshold: the update is that task alone
        out, m = f(self.batch([all_ok, wide, narrow]), {}, target_groups=2, std_threshold=0.2, std_floor=1.0)
        self.assertEqual(sorted(set(out.non_tensor_batch["uid"])), ["t1"])
        self.assertEqual((m["filter/mixed/empty_fallback"], m["filter/keep/filled"]), (0.0, 1))


if __name__ == "__main__":
    unittest.main()
