"""Validation in no-concat mode: the padding copies must not merge into the original trajectories (round 7).

The val batch (8 tasks x 4 rollouts = 32 rows) is padded to the 48 rollout workers with copies of its first 16
rows.  The copies kept their uid, passed the "uid in original_uids" filter and were concatenated into the
original trajectories: the summed reward counted two rollouts, the success flag came from the longer one.
"""

import importlib.util
import unittest

HAVE_VERL = importlib.util.find_spec("verl") is not None and importlib.util.find_spec("tensordict") is not None


@unittest.skipUnless(HAVE_VERL, "needs verl and tensordict (the training env)")
class ValPaddingTest(unittest.TestCase):
    def _batches(self):
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor

        uid = np.array(["a", "a", "b", "b"], dtype=object)  # 2 tasks x 2 rollouts
        gen = DataProto(batch=TensorDict({"x": torch.zeros(4, 1)}, batch_size=(4,)),
                        non_tensor_batch={"uid": uid, "group_idx": uid.copy(),
                                          "traj_idx": np.array([0, 1, 0, 1], dtype=object)}, meta_info={})
        padded, pad_size = pad_dataproto_to_divisor(gen, 6)
        return gen, padded, pad_size

    def test_padding_copies_get_their_own_uids(self):
        from vagen.utils.concat_val_multi_turn import mark_padding_uids

        gen, padded, pad_size = self._batches()
        self.assertEqual(pad_size, 2)
        self.assertEqual(list(padded.non_tensor_batch["uid"][4:]), ["a", "a"])  # the copies, uid included
        mark_padding_uids(padded, pad_size)
        for key in ("uid", "group_idx"):
            self.assertEqual(list(padded.non_tensor_batch[key][:4]), ["a", "a", "b", "b"])
            self.assertEqual(len(set(padded.non_tensor_batch[key][4:]) | {"a", "b"}), 4)
            self.assertEqual(list(gen.non_tensor_batch[key]), ["a", "a", "b", "b"])  # the unpadded batch keeps its own
        mark_padding_uids(padded, 0)  # nothing padded, nothing changed
        self.assertEqual(list(padded.non_tensor_batch["uid"][:4]), ["a", "a", "b", "b"])

    def test_filtered_validation_counts_each_rollout_once(self):
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto

        from vagen.utils.concat_val_multi_turn import concat_val_multi_turn, mark_padding_uids

        gen, padded, pad_size = self._batches()
        mark_padding_uids(padded, pad_size)
        original_uids = set(gen.non_tensor_batch["uid"])
        # one turn per padded input row, as the agent loop returns them; every rollout earns 4.99
        n = len(padded)
        rm = torch.zeros(n, 3)
        rm[:, -1] = 4.99
        out = DataProto(
            batch=TensorDict({"prompts": torch.ones(n, 2, dtype=torch.long), "responses": torch.ones(n, 3, dtype=torch.long),
                              "response_mask": torch.ones(n, 3, dtype=torch.long), "rm_scores": rm}, batch_size=(n,)),
            non_tensor_batch={"group_idx": padded.non_tensor_batch["group_idx"],
                              "traj_idx": padded.non_tensor_batch["traj_idx"],
                              "turn_idx": np.zeros(n, dtype=object),
                              "reward_extra_info": np.array([{"traj_success": 1.0} for _ in range(n)], dtype=object)},
            meta_info={})
        keep = [i for i, g in enumerate(out.non_tensor_batch["group_idx"]) if g in original_uids]  # ray_trainer's filter
        self.assertEqual(keep, [0, 1, 2, 3])
        merged = concat_val_multi_turn(out.select_idxs(keep), gen, type("Tok", (), {"pad_token_id": 0})())
        self.assertEqual([round(s, 2) for s in merged.batch["rm_scores"].sum(-1).tolist()], [4.99] * 4)


if __name__ == "__main__":
    unittest.main()
