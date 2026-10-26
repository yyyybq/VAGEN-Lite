"""The per-token log-probability files (vagen/logprob_tokens.py) and their breakdown (model_matrix/logprob)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

IMG = 151655


def fake_batch():
    # prompt (left-padded, 6): pad pad sys <img> <img> txt | responses (8): a0 a1 | obs <img> obs | a2 a3 a4 (mask 1 on a*)
    prompt = [0, 0, 11, IMG, IMG, 12]
    responses = [21, 22, 31, IMG, 32, 23, 24, 25]
    mask = [1, 1, 0, 0, 0, 1, 1, 1]
    input_ids = torch.tensor([prompt + responses, [0] * 6 + [0] * 8])
    resp = torch.tensor([responses, [0] * 8])
    m = torch.tensor([mask, [0] * 8])
    rollout = torch.tensor([[-1.0, -2.0, 0, 0, 0, -0.5, -0.1, -0.2], [0.0] * 8])
    trainer = rollout + torch.tensor([[0.1, 0.0, 0, 0, 0, -0.06, 0.0, 0.01], [0.0] * 8])
    return SimpleNamespace(batch={"rollout_log_probs": rollout, "old_log_probs": trainer, "responses": resp,
                                  "response_mask": m, "input_ids": input_ids})


class TokenRecordsTest(unittest.TestCase):
    def test_turns_places_and_image_distance(self):
        from vagen.logprob_tokens import token_records, write_logprob_tokens

        rec = token_records(fake_batch())
        self.assertEqual(rec["sample"].tolist(), [0] * 5)  # the empty row is skipped
        self.assertEqual(rec["token"].tolist(), [21, 22, 23, 24, 25])
        self.assertEqual(rec["turn"].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(rec["pos_in_turn"].tolist(), [0, 1, 0, 1, 2])
        # a0 at sequence position 6, the last image before it at 4; a2 at 11, the image in the observation at 9
        self.assertEqual(rec["image_distance"].tolist(), [2, 3, 2, 3, 4])
        self.assertTrue(np.allclose(rec["trainer"] - rec["rollout"], [0.1, 0.0, -0.06, 0.0, 0.01], atol=1e-6))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(write_logprob_tokens(fake_batch(), tmp, 3), {"logprob_tokens/tokens": 5.0})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["step_0003.npz"])
            from model_matrix.logprob import breakdown as B

            got = B.breakdown(B.load(Path(tmp)), is_format=np.array([True, False, False, False, True]))
            self.assertEqual(got["all"]["tokens"], 5)
            self.assertEqual(got["turn: first answer"]["tokens"], 2)
            self.assertEqual(got["place: first token"]["tokens"], 2)
            self.assertEqual(got["image distance: 0-49"]["tokens"], 5)
            self.assertEqual(got["kind: format tokens"]["tokens"], 2)
            # |exp(d) - 1| > 5%: d = 0.1 and -0.06 (exp(-0.06) = 0.942)
            self.assertAlmostEqual(got["all"]["share_ratio_off_5pct"], 0.4)
            self.assertAlmostEqual(got["turn: later answers"]["mean_abs"], round((0.06 + 0 + 0.01) / 3, 5), places=5)
            self.assertEqual(B.parse_steps("2-10"), list(range(2, 11)))
            self.assertIn("| all |", B.report({"eager": got, "sdpa": got}))

    def test_nothing_without_the_audit(self):
        from vagen.logprob_tokens import write_logprob_tokens

        b = fake_batch()
        b.batch.pop("rollout_log_probs")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(write_logprob_tokens(b, str(Path(tmp) / "tokens"), 1), {})
            self.assertFalse((Path(tmp) / "tokens").exists())


if __name__ == "__main__":
    unittest.main()
