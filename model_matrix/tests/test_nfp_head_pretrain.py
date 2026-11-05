"""Decision 54 丙 of 2026-11-05: the next-frame prediction head trained alone on the SFT frame pairs
(model_matrix/formal/nfp_head_pretrain.py): the pairs' tensors against the training loss, the held-back split, the
training loop, the held-out rows and the gate, the RL start directory, the job script."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from model_matrix.formal import nfp_head_pretrain as P
from model_matrix.formal import nfp_heldout as H
from vagen.models import cambrian_nfp as N

I = N.IMAGE_TOKEN_INDEX
CAMBRIAN_SRC = "/mnt/umm/users/yinbaiqiao/cambrian-s"


class FakeInner(nn.Module):
    """The adapter's inner model, small, as test_cambrian_nfp's: embeddings, projector, newline, a one-layer stand-in
    for the decoder and the model line's head."""

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


class PairTensorsTest(unittest.TestCase):
    def test_as_the_training_loss(self):
        """The head on pair_tensors' input against pair_tensors' target is concat_nfp_loss, the copy baseline too."""
        torch.manual_seed(0)
        h, d = 8, 5
        row = [1] + [I] * 756 + [2, 3] + [I] * 756 + [4] + [I] * 756 + [5]
        ids = torch.tensor([row, [6] + [I] * 756 + [7] + [I] * 756 + [8] * (len(row) - 2 * 756 - 2)])
        pos = torch.arange(ids.shape[1])[None].expand(2, -1)
        pairs = N.concat_nfp_pairs(ids, pos, 756)
        self.assertEqual(len(pairs), 3)
        hidden, raw, head = torch.randn(2, ids.shape[1], h), torch.randn(5, 729, d), N.build_head(h, d)
        want, stats = N.concat_nfp_loss(hidden, pairs, raw, head)
        x, y, c = P.pair_tensors(hidden, pairs, raw)
        self.assertEqual([tuple(t.shape) for t in (x, y, c)], [(3, 64, h), (3, 64, d), (3, 64, d)])
        self.assertEqual((x.dtype, y.dtype, c.dtype), (hidden.dtype, torch.float32, torch.float32))
        mse, cos = N._losses(head(x).float(), y)
        self.assertAlmostEqual(float(0.1 * mse + 0.1 * cos), float(want), places=5)
        cmse, ccos = N._losses(c, y)
        self.assertAlmostEqual(float(cmse), stats["copy_mse"], places=5)
        self.assertAlmostEqual(float(ccos), stats["copy_cos"], places=5)
        self.assertFalse(y.requires_grad)  # the target is detached, as in training


@unittest.skipUnless(Path(CAMBRIAN_SRC).is_dir(), "needs the Cambrian source")
class AdapterPathTest(unittest.TestCase):
    """nfp_heldout.episode_forward + pair_tensors on the small stand-in adapter (test_cambrian_nfp's): the same rows
    as score_episode, the scoring every RL checkpoint's head had."""

    H, D, VOCAB, ALIAS = 8, 6, 160, 9

    @classmethod
    def setUpClass(cls):
        cls._env = mock.patch.dict(os.environ, {"CAMBRIAN_SRC": CAMBRIAN_SRC})
        cls._env.start()
        from vagen.models import cambrian_register as R

        cls.R = R

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()

    def make(self):
        torch.manual_seed(0)
        a = self.R.CambrianForCausalLMAdapter.__new__(self.R.CambrianForCausalLMAdapter)
        nn.Module.__init__(a)
        a.config = SimpleNamespace(connector_only=True, image_token_alias=self.ALIAS, si_token_len=729,
                                   mm_use_im_newline_token=True, model_line_nfp=True, output_attentions=False,
                                   output_hidden_states=False, use_return_dict=True, vocab_size=self.VOCAB)
        a.model = FakeInner(self.H, self.D, self.VOCAB)
        a.lm_head = nn.Linear(self.H, self.VOCAB, bias=False)
        a.vision = nn.Linear(3, self.D)
        a.encode_images = lambda xs: [a.vision(xs[0].flatten(2).transpose(1, 2))]
        return a.eval()

    def test_rows_as_score_episode(self):
        a = self.make()
        row = []
        for k in range(4):
            row += [100 + k] + [self.ALIAS] * 756
        ids, pix = torch.tensor([row + [99]]), torch.randn(4, 3, 27, 27)
        with torch.no_grad():
            want = H.score_episode(a, ids, pix)
            hidden, raw, pairs, side, newline = H.episode_forward(a, ids, pix)
            x, y, c = P.pair_tensors(hidden, pairs, raw, side, newline)
        meta = [{"episode": 0, "k": k, "answer": ""} for k in range(len(pairs))]
        got = P.heldout_rows(a.model.nfp_head, x, y, c, meta)
        self.assertEqual(len(got), 3)
        for g, w in zip(got, want):
            for k in ("mse", "cos", "copy_mse", "copy_cos", "loss", "copy_loss"):
                self.assertAlmostEqual(g[k], w[k], places=5, msg=k)


class HeldBackTest(unittest.TestCase):
    def test_share_and_fixed(self):
        backs = [P.held_back(i) for i in range(20000)]
        self.assertAlmostEqual(sum(backs) / len(backs), 0.05, delta=0.005)
        self.assertEqual(backs[:100], [P.held_back(i) for i in range(100)])
        self.assertEqual(P.held_back(7, 0.0), False)
        self.assertEqual(P.held_back(7, 1.0), True)


class TrainHeadTest(unittest.TestCase):
    @staticmethod
    def data(n, w):
        x = torch.randn(n, 64, w.shape[0])
        return x, torch.tanh(x @ w)

    def test_learns_and_keeps_the_best_epoch(self):
        torch.manual_seed(0)
        w = torch.randn(6, 4)
        train, pick = self.data(256, w), self.data(64, w)
        head = N.build_head(6, 4)
        before = P.mean_loss(head, *pick)
        best, hist = P.train_head(head, train, pick, epochs=30, lr=1e-2, batch=32, patience=100, warmup=1, seed=0,
                                  log=lambda s: None)
        self.assertEqual([h["epoch"] for h in hist], list(range(1, 31)))
        head.load_state_dict(best)
        after = P.mean_loss(head, *pick)
        self.assertLess(after, 0.5 * before)
        self.assertAlmostEqual(after, min(h["pick_loss"] for h in hist), places=6)
        self.assertEqual(set(best), {"0.weight", "0.bias", "2.weight", "2.bias"})

    def test_stops_after_patience(self):
        torch.manual_seed(0)
        w = torch.randn(6, 4)
        head = N.build_head(6, 4)
        start = {k: v.clone() for k, v in head.state_dict().items()}
        best, hist = P.train_head(head, self.data(64, w), self.data(16, w), epochs=50, lr=0.0, batch=16, patience=3,
                                  seed=0, log=lambda s: None)
        self.assertEqual(len(hist), 4)  # epoch 1 is the best, three more without a better pick loss
        self.assertTrue(all(torch.equal(best[k], start[k]) for k in start))  # a zero rate moves nothing

    def test_reproducible(self):
        w = torch.randn(6, 4)
        train, pick = self.data(64, w), self.data(16, w)
        runs = []
        for _ in range(2):
            torch.manual_seed(1)
            head = N.build_head(6, 4)
            runs.append(P.train_head(head, train, pick, epochs=3, lr=1e-2, batch=16, patience=5, seed=3,
                                     log=lambda s: None)[1])
        self.assertEqual(runs[0], runs[1])


class HeldoutRowsTest(unittest.TestCase):
    def test_rows_summary_and_gate(self):
        torch.manual_seed(0)
        x, y, c = torch.randn(10, 64, 6), torch.randn(10, 64, 4), torch.randn(10, 64, 4)
        head = N.build_head(6, 4)
        meta = [{"episode": i // 2, "k": i % 2, "answer": "<action>turn_left|</action>"} for i in range(10)]
        rows = P.heldout_rows(head, x, y, c, meta)
        for r, xi, yi, ci, m in zip(rows, x, y, c, meta):
            mse, cos = N._losses(head(xi[None]).float(), yi[None])
            cmse, ccos = N._losses(ci[None], yi[None])
            self.assertEqual({k: r[k] for k in m}, m)
            self.assertAlmostEqual(r["loss"], 0.1 * float(mse) + 0.1 * float(cos), places=6)
            self.assertAlmostEqual(r["copy_loss"], 0.1 * float(cmse) + 0.1 * float(ccos), places=6)
        H.avg_baseline(rows, list(y))
        s = H.summarize(rows, 5, 0)
        self.assertEqual(s["pairs"], 10)
        self.assertEqual(P.gate(s), s["mean"]["loss"] < s["mean"]["avg_loss"])
        self.assertTrue(P.gate({"mean": {"loss": 0.1, "avg_loss": 0.2}}))
        self.assertFalse(P.gate({"mean": {"loss": 0.2, "avg_loss": 0.2}}))
        self.assertFalse(P.gate({"mean": {"loss": 0.1}}))  # no average-frame baseline: not passed
        self.assertFalse(P.gate({}))


class StartDirTest(unittest.TestCase):
    def test_links_config_index_and_head(self):
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ckpt"
            src.mkdir()
            save_file({"model.embed.weight": torch.ones(2, 3)}, str(src / "model-00001-of-00001.safetensors"))
            (src / "model.safetensors.index.json").write_text(json.dumps(
                {"metadata": {"total_size": 24}, "weight_map": {"model.embed.weight": "model-00001-of-00001.safetensors"}}))
            (src / "config.json").write_text(json.dumps({"model_type": "cambrian_qwen", "hidden_size": 6}))
            (src / "tokenizer.json").write_text("{}")
            before = {p.name: p.read_bytes() for p in src.iterdir()}
            head = N.build_head(6, 4)
            out = P.start_dir(src, head.state_dict(), Path(tmp) / "start")
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             sorted(["config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors",
                                     "tokenizer.json", P.HEAD_SHARD]))
            for name in ("model-00001-of-00001.safetensors", "tokenizer.json"):
                self.assertTrue((out / name).is_symlink())
                self.assertEqual((out / name).resolve(), (src / name).resolve())
            for name in ("config.json", "model.safetensors.index.json", P.HEAD_SHARD):
                self.assertFalse((out / name).is_symlink())
            self.assertEqual(json.loads((out / "config.json").read_text()),
                             {"model_type": "cambrian_qwen", "hidden_size": 6, "model_line_nfp": True})
            index = json.loads((out / "model.safetensors.index.json").read_text())
            keys = {P.HEAD_PREFIX + k for k in ("0.weight", "0.bias", "2.weight", "2.bias")}
            self.assertEqual({k for k, v in index["weight_map"].items() if v == P.HEAD_SHARD}, keys)
            self.assertEqual(index["weight_map"]["model.embed.weight"], "model-00001-of-00001.safetensors")
            self.assertEqual(index["metadata"]["total_size"], 24 + sum(v.numel() * 4 for v in head.state_dict().values()))
            with safe_open(str(out / P.HEAD_SHARD), framework="pt") as f:
                self.assertEqual(set(f.keys()), keys)
                for k, v in head.state_dict().items():
                    self.assertTrue(torch.equal(f.get_tensor(P.HEAD_PREFIX + k), v.float()))
            self.assertEqual({p.name: p.read_bytes() for p in src.iterdir()}, before)  # the source untouched
            with self.assertRaises(FileExistsError):
                P.start_dir(src, head.state_dict(), out)
            with self.assertRaises(ValueError):  # a source that has a head already
                P.start_dir(out, head.state_dict(), Path(tmp) / "again")
            self.assertFalse((Path(tmp) / "again").exists())

    @unittest.skipUnless((P.SFT / "model.safetensors.index.json").exists(), "needs the SFT start")
    def test_on_the_sft_start(self):
        """On checkpoint-770 itself: links only, and nfp_heldout's weight check finds the head on the new shard."""
        head = N.build_head(3584, 1152)
        with tempfile.TemporaryDirectory() as tmp:
            out = P.start_dir(P.SFT, head.state_dict(), Path(tmp) / "start")
            big = [p for p in out.iterdir() if p.name.endswith(".safetensors") and p.name != P.HEAD_SHARD]
            self.assertEqual(len(big), 7)
            self.assertTrue(all(p.is_symlink() for p in big))
            self.assertLess((out / P.HEAD_SHARD).stat().st_size, 100e6)  # only the head is written (17M float32)
            checks = H.weight_checks({"model.nfp_head.2.weight": head[2].weight}, out)
            self.assertIs(checks["model.nfp_head.2.weight"], True)
            self.assertIsNone(checks[H.CHECKED[1]])  # not among the given parameters
            self.assertTrue(json.loads((out / "config.json").read_text())["model_line_nfp"])
        self.assertTrue((P.SFT / "config.json").is_file())  # the start's own files untouched
        self.assertNotIn("model_line_nfp", json.loads((P.SFT / "config.json").read_text()))


class JobScriptTest(unittest.TestCase):
    def test_parses(self):
        script = Path(P.__file__).with_suffix(".sh")
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        self.assertIn("-m model_matrix.formal.nfp_head_pretrain", text)
        self.assertIn("-m model_matrix.formal.nfp_heldout --model \"${OUT}/run/start\"", text)


if __name__ == "__main__":
    unittest.main()
