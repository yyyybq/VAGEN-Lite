"""Cambrian-S's next-frame prediction loss on the concat path (the model line, the user's decision 47 乙 of
2026-11-03): the frame pairs, the loss against Cambrian-S's definition, the detached targets, the one-frame episode,
the actor's metrics, and the adapter's forward with it (vagen/models/cambrian_nfp.py, cambrian_register.py)."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from vagen.models import cambrian_nfp as N

I = N.IMAGE_TOKEN_INDEX
CAMBRIAN_SRC = "/mnt/umm/users/yinbaiqiao/cambrian-s"


def grid_by_hand(x, side, d):
    """Cambrian-S's MIV resize of one frame's (side * side, d) features to (64, d)."""
    y = F.interpolate(x.view(side, side, d).permute(2, 0, 1)[None], size=(8, 8), mode="bilinear", align_corners=False)
    return y[0].permute(1, 2, 0).reshape(64, d)


class PairsTest(unittest.TestCase):
    def test_patch_offsets(self):
        self.assertEqual(N.patch_offsets(3).tolist(), [0, 1, 2, 4, 5, 6, 8, 9, 10])  # a newline after every row
        self.assertEqual(N.patch_offsets(3, newline=False).tolist(), list(range(9)))
        self.assertEqual(len(N.patch_offsets(27)), 729)
        self.assertEqual(int(N.patch_offsets(27)[-1]), 26 * 28 + 26)

    def test_pairs(self):
        row = [1] + [I] * 4 + [2] + [I] * 4 + [3, 4] + [I] * 4
        self.assertEqual(N.concat_nfp_pairs(torch.tensor([row]), None, 4), [(0, 1, 0, 1), (0, 6, 1, 2)])
        # two rows: the global frame indices run on, as the adapter scatters the frames
        two = torch.tensor([row, [I] * 4 + [5] + [I] * 4 + [0] * 7])
        self.assertEqual(N.concat_nfp_pairs(two, None, 4), [(0, 1, 0, 1), (0, 6, 1, 2), (1, 0, 3, 4)])
        # one frame, no frame: no pair
        self.assertEqual(N.concat_nfp_pairs(torch.tensor([[1] + [I] * 4 + [2]]), None, 4), [])
        self.assertEqual(N.concat_nfp_pairs(torch.tensor([[1, 2, 3]]), None, 4), [])
        with self.assertRaises(ValueError):  # a cut frame
            N.concat_nfp_pairs(torch.tensor([[1] + [I] * 3]), None, 4)

    def test_no_pair_across_sequences(self):
        """After verl's padding removal a row may hold two episodes; their positions restart at 0."""
        ep = [7] + [I] * 4 + [8] + [I] * 4
        ids = torch.tensor([ep + ep])
        pos = torch.tensor([list(range(10)) + list(range(10))])
        self.assertEqual(N.concat_nfp_pairs(ids, pos, 4), [(0, 1, 0, 1), (0, 11, 2, 3)])
        # a left-padded row (padding at position 0, as compute_position_id_with_mask gives it): one sequence
        ids = torch.tensor([[0, 0] + ep])
        pos = torch.tensor([[0, 0] + list(range(10))])
        self.assertEqual(N.concat_nfp_pairs(ids, pos, 4), [(0, 3, 0, 1)])
        # positions of another shape (e.g. none given): the row as one sequence
        self.assertEqual(N.concat_nfp_pairs(torch.tensor([ep + ep]), torch.arange(5)[None], 4),
                         [(0, 1, 0, 1), (0, 6, 1, 2), (0, 11, 2, 3)])


class LossTest(unittest.TestCase):
    SIDE, H, D = 27, 6, 5

    def episode(self, frames):
        per = self.SIDE * (self.SIDE + 1)
        row = []
        for k in range(frames):
            row += [100 + k] + [I] * per
        return torch.tensor([row + [99]]), per

    def test_head(self):
        torch.manual_seed(0)
        head = N.build_head(64, 16)
        self.assertEqual([type(m) for m in head], [nn.Linear, nn.GELU, nn.Linear])
        self.assertEqual((head[0].in_features, head[0].out_features, head[2].out_features), (64, 64, 16))
        for i in (0, 2):
            self.assertEqual(float(head[i].bias.abs().sum()), 0.0)
            self.assertAlmostEqual(float(head[i].weight.std()), 0.02, delta=0.003)

    def test_loss_is_cambrian_s_definition(self):
        torch.manual_seed(0)
        side, h, d = self.SIDE, self.H, self.D
        ids, per = self.episode(3)
        hs = torch.randn(1, ids.shape[1], h, requires_grad=True)
        raw = torch.randn(3, side * side, d, requires_grad=True)
        head = N.build_head(h, d)
        pairs = N.concat_nfp_pairs(ids, None, per)
        loss, stats = N.concat_nfp_loss(hs, pairs, raw, head)
        offs = N.patch_offsets(side)
        preds, targets, copies = [], [], []
        for b, s, c, n in pairs:
            g = hs[b, s + offs].view(side, side, h).permute(2, 0, 1)
            preds.append(head(F.adaptive_avg_pool2d(g, 8).permute(1, 2, 0).reshape(64, h)))
            targets.append(grid_by_hand(raw[n].detach(), side, d))
            copies.append(grid_by_hand(raw[c].detach(), side, d))
        P, T, C = (torch.stack(x) for x in (preds, targets, copies))
        mse = ((P - T) ** 2).mean()
        cos = (1 - F.cosine_similarity(P, T, dim=-1)).mean()
        self.assertAlmostEqual(float(loss), float(0.1 * mse + 0.1 * cos), places=5)
        self.assertEqual(stats["pairs"], 2.0)
        self.assertAlmostEqual(stats["mse"], float(mse), places=5)
        self.assertAlmostEqual(stats["cos"], float(cos), places=5)
        self.assertAlmostEqual(stats["copy_mse"], float(((C - T) ** 2).mean()), places=5)
        self.assertAlmostEqual(stats["copy_cos"], float((1 - F.cosine_similarity(C, T, dim=-1)).mean()), places=5)
        w, _ = N.concat_nfp_loss(hs, pairs, raw, head, mse_weight=1.0, cos_weight=0.0)
        self.assertAlmostEqual(float(w), float(mse), places=5)

        loss.backward()
        self.assertIsNone(raw.grad)  # the targets give no gradient
        g = hs.grad[0].abs().sum(-1)
        start = [p[1] for p in pairs]
        for s in start:  # every patch position of a predicting frame, no newline position
            self.assertTrue(bool((g[s + offs] > 0).all()))
            self.assertEqual(float(g[s + side::side + 1][:side].sum()), 0.0)
        last = start[-1] + per + 1  # the last frame predicts nothing
        self.assertEqual(float(g[last:last + per].sum()), 0.0)
        self.assertEqual(float(g[[0, per + 1, 2 * per + 2, ids.shape[1] - 1]].sum()), 0.0)  # text
        self.assertTrue(all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in head.parameters()))

    def test_one_frame_runs_the_head_with_weight_zero(self):
        torch.manual_seed(0)
        ids, per = self.episode(1)
        hs = torch.randn(1, ids.shape[1], self.H, requires_grad=True)
        head = N.build_head(self.H, self.D)
        loss, stats = N.concat_nfp_loss(hs, N.concat_nfp_pairs(ids, None, per), torch.randn(1, 729, self.D), head)
        self.assertEqual((float(loss), stats), (0.0, {"pairs": 0.0}))
        loss.backward()
        for p in head.parameters():  # in the graph (FSDP's reduction), with zero gradient
            self.assertIsNotNone(p.grad)
            self.assertEqual(float(p.grad.abs().sum()), 0.0)

    def test_copy_baseline_of_a_still_camera(self):
        torch.manual_seed(0)
        ids, per = self.episode(2)
        frame = torch.randn(1, 729, self.D)
        _, stats = N.concat_nfp_loss(torch.randn(1, ids.shape[1], self.H), N.concat_nfp_pairs(ids, None, per),
                                     frame.repeat(2, 1, 1), N.build_head(self.H, self.D))
        self.assertAlmostEqual(stats["copy_mse"], 0.0, places=6)
        self.assertAlmostEqual(stats["copy_cos"], 0.0, places=5)

    def test_metrics(self):
        # the LFP path (no statistics): the metrics as before
        self.assertEqual(N.nfp_metrics(0.5, None, 1.0, 0.25),
                         {"actor/nfp_loss": 0.5, "actor/nfp_loss_scaled": 0.125, "actor/nfp_loss_coef": 1.0,
                          "actor/image_gen_loss_coef": 1.0, "actor/nfp_loss_active": 1.0})
        # an episode without a pair: not a loss
        self.assertEqual(N.nfp_metrics(0.0, {"pairs": 0.0}, 1.0, 0.25),
                         {"actor/nfp_loss_active": 0.0, "actor/nfp_pairs": 0.0})
        m = N.nfp_metrics(0.2, {"pairs": 3.0, "mse": 1.0, "cos": 0.5, "copy_mse": 2.0, "copy_cos": 0.7}, 1.0, 1.0)
        self.assertEqual((m["actor/nfp_loss"], m["actor/nfp_pairs"], m["actor/nfp_mse"], m["actor/nfp_copy_cos"],
                          m["actor/nfp_loss_active"]), (0.2, 3.0, 1.0, 0.7, 1.0))


class FakeInner(nn.Module):
    """The adapter's inner model, small: embeddings, projector, newline, a one-layer stand-in for the decoder and the
    model line's head."""

    def __init__(self, h, d, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, h)
        self.mm_projector = nn.Sequential(nn.Linear(d, h))
        self.image_newline = nn.Parameter(torch.randn(h))
        self.mix = nn.Linear(h, h)
        self.nfp_head = N.build_head(h, d)

    def forward(self, inputs_embeds=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(last_hidden_state=torch.tanh(self.mix(inputs_embeds)))


@unittest.skipUnless(Path(CAMBRIAN_SRC).is_dir(), "needs the Cambrian source")
class AdapterForwardTest(unittest.TestCase):
    """CambrianForCausalLMAdapter.forward with config model_line_nfp, on a small stand-in model (the real forward,
    alias mapping, image embedding and loss wiring; the 7B model and SigLIP are not loaded)."""

    @classmethod
    def setUpClass(cls):
        cls._env = mock.patch.dict(os.environ, {"CAMBRIAN_SRC": CAMBRIAN_SRC})
        cls._env.start()
        from vagen.models import cambrian_register as R

        cls.R = R

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()

    H, D, VOCAB, ALIAS = 8, 6, 160, 9

    def make(self, nfp=True):
        R = self.R
        torch.manual_seed(0)
        a = R.CambrianForCausalLMAdapter.__new__(R.CambrianForCausalLMAdapter)
        nn.Module.__init__(a)
        a.config = SimpleNamespace(connector_only=True, image_token_alias=self.ALIAS, si_token_len=729,
                                   mm_use_im_newline_token=True, model_line_nfp=nfp, output_attentions=False,
                                   output_hidden_states=False, use_return_dict=True, vocab_size=self.VOCAB)
        a.model = FakeInner(self.H, self.D, self.VOCAB)
        a.lm_head = nn.Linear(self.H, self.VOCAB, bias=False)
        a.vision = nn.Linear(3, self.D)
        a.encode_images = lambda xs: [a.vision(xs[0].flatten(2).transpose(1, 2))]  # (N, 729, D)
        return a

    def batch(self, frames):
        row = []
        for k in range(frames):
            row += [100 + k] + [self.ALIAS] * 756
        row += [99]
        ids = torch.tensor([row])
        return ids, torch.arange(ids.shape[1])[None], torch.randn(frames, 3, 27, 27)

    def test_training_forward_returns_the_loss(self):
        a = self.make()
        ids, pos, pix = self.batch(3)
        out = a(input_ids=ids, position_ids=pos, pixel_values=pix)
        self.assertEqual(out.nfp_stats["pairs"], 2.0)
        # the same loss by hand from the adapter's own pieces
        mapped = ids.masked_fill(ids == self.ALIAS, I)
        embeds, raw = a._embed_multimodal_batch(mapped, pix, return_raw=True)
        self.assertTrue(torch.equal(raw, a.encode_images([pix])[0]))
        hs = a.model(inputs_embeds=embeds)[0]
        want, _ = N.concat_nfp_loss(hs, N.concat_nfp_pairs(mapped, pos, 756), raw, a.model.nfp_head)
        self.assertAlmostEqual(float(out.loss), float(want), places=5)
        out.loss.backward()
        self.assertTrue(all(p.grad is not None for p in a.model.nfp_head.parameters()))
        self.assertTrue(float(a.model.mix.weight.grad.abs().sum()) > 0)

    def test_no_loss_in_eval_or_without_the_switch(self):
        ids, pos, pix = self.batch(3)
        a = self.make()
        a.eval()  # the reference model, the old log-probabilities
        out = a(input_ids=ids, position_ids=pos, pixel_values=pix)
        self.assertIsNone(out.loss)
        self.assertFalse(hasattr(out, "nfp_stats"))
        off = self.make(nfp=False)
        out = off(input_ids=ids, position_ids=pos, pixel_values=pix)
        self.assertIsNone(out.loss)
        self.assertTrue(torch.allclose(out.logits, a(input_ids=ids, position_ids=pos, pixel_values=pix).logits))

    def test_one_frame_episode(self):
        a = self.make()
        ids, pos, pix = self.batch(1)
        out = a(input_ids=ids, position_ids=pos, pixel_values=pix)
        self.assertEqual((float(out.loss), out.nfp_stats), (0.0, {"pairs": 0.0}))


if __name__ == "__main__":
    unittest.main()
