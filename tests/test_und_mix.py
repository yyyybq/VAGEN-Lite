"""vagen/und_mix.py (the reward line, 2026-11-05): the pool and its held-out checks, the draws, the PPO block, the
gradient mixing, the mixer's step on a toy actor, and -- when UND_MIX_TEST_MODEL names a Qwen3-VL model folder (read
only) -- the matrix SFT's LLaMA-Factory path on synthetic rows.  Everything written goes to a temporary folder."""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

from vagen import und_mix as U

ATTRS = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images", "videos": "videos"},
         "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                  "system_tag": "system"}}
RECIPE = ("template: qwen3_vl_nothink\ncutoff_len: 16384\nmask_history: false\nimage_max_pixels: 65536\n"
          "video_max_pixels: 65536\nbf16: true\nlearning_rate: 1.0e-05\n")


def _png(path: Path, color, size=(64, 48)) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return str(path)


def make_sft(root: Path):
    """A small SFT folder like the matrix's: P0 rows (one multi-turn, one video of frames), P1 rows."""
    img = lambda name, c: _png(root / "img" / f"{name}.png", c)  # noqa: E731
    p0 = [
        {"messages": [{"role": "user", "content": "<image><image>\nWhich is closer?\nChoices: A. sofa B. desk"},
                      {"role": "assistant", "content": "B"}],
         "images": [img("a0", (200, 10, 10)), img("a1", (10, 200, 10))], "videos": [], "pool": "P0",
         "source_id": "p0_aaaa0000", "scene": "0001_000001", "stratum": "M8",
         "meta": json.dumps({"qids": ["aaaa0000"], "questions": 1})},
        {"messages": [{"role": "user", "content": "<video>\nHow many chairs?\nAnswer with a single number."},
                      {"role": "assistant", "content": "3"},
                      {"role": "user", "content": "Floor area in m^2?\nAnswer with a single number."},
                      {"role": "assistant", "content": "12.5"}],
         "images": [], "videos": [[img(f"v{i}", (10 * i, 50, 90)) for i in range(4)]], "pool": "P0",
         "source_id": "p0_bbbb1111", "scene": "0002_000002", "stratum": "V",
         "meta": json.dumps({"qids": ["bbbb1111", "bbbb2222"], "questions": 2})},
        {"messages": [{"role": "user", "content": "<image>\nIs the lamp on?\nAnswer Yes or No."},
                      {"role": "assistant", "content": "Yes"}],
         "images": [img("held_image", (90, 90, 10))], "videos": [], "pool": "P0", "source_id": "p0_cccc0000",
         "scene": "0003_000003", "stratum": "S", "meta": json.dumps({"qids": ["cccc0000"], "questions": 1})},
        {"messages": [{"role": "user", "content": "No picture here. Say A.\nChoices: A. a B. b"},
                      {"role": "assistant", "content": "A"}],
         "images": [], "videos": [], "pool": "P0", "source_id": "p0_dddd0000", "scene": "0004_000004",
         "stratum": "T", "meta": json.dumps({"qids": ["dddd0000"], "questions": 1})},
        {"messages": [{"role": "user", "content": "<image>\n<video>\nSame room?\nAnswer Yes or No."},
                      {"role": "assistant", "content": "No"}],
         "images": [img("both_i", (5, 5, 5))], "videos": [[img(f"both_v{i}", (9, 9, 9)) for i in range(2)]],
         "pool": "P0", "source_id": "p0_eeee0000", "scene": "0005_000005", "stratum": "IV",
         "meta": json.dumps({"qids": ["eeee0000"], "questions": 1})},
    ]
    p1 = [
        {"messages": [{"role": "user", "content": "First image: <image>\nWhat did the camera do?\n(A) left (B) right"},
                      {"role": "assistant", "content": "A" if i % 2 else "B"}],
         "images": [img(f"p1_{i}", (i * 20, 100, 100))], "videos": [], "pool": "P1",
         "source_id": f"p1_{i:012x}", "scene": f"01{i:02d}_000100", "stratum": "action_between",
         "meta": json.dumps({"task_id": f"v10_task{i}", "questions": 1})} for i in range(6)]
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("P0_train", p0), ("P1_train", p1)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "dataset_info.json").write_text(json.dumps({
        "P0_train": dict(ATTRS, file_name="P0_train.jsonl"), "P1_train": dict(ATTRS, file_name="P1_train.jsonl"),
        "P2q_train": dict(ATTRS, file_name="P2q_train.jsonl")}))
    (root / "train.yaml").write_text(RECIPE)
    # held-out sets: a probe sharing a scene (P1 row 0), a task (P1 row 1) and a question id (P1 row 2) with the
    # training rows, and a RoboCasa question set sharing an image with a P0 row
    probe = [{"sample_id": "p1_probe0", "scene": "0100_000100", "task_id": "v10_other", "images": ["/nowhere/x.png"]},
             {"sample_id": "p1_probe1", "scene": "0999_000999", "task_id": "v10_task1", "images": []},
             {"sample_id": "p1_000000000002", "scene": "0998_000998", "task_id": "v10_other2", "images": []}]
    questions = [{"id": "x0_0001", "task": "OpenDrawer", "seed": 1000, "camera": "left",
                  "image_path": str(root / "img" / "held_image.png")},
                 {"id": "x0_0002", "task": "OpenDrawer", "seed": 1001, "camera": "left", "image_path": "/x0/1.png"}]
    (root / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probe))
    (root / "x0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in questions))
    return root


class PoolTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sft = make_sft(self.root / "sft")
        self.pool = self.root / "pool"
        self.manifest = U.prepare(self.sft, ["P0_train", "P1_train"],
                                  [("P1_probe", self.sft / "probe.jsonl"), ("X0", self.sft / "x0.jsonl")], self.pool)

    def tearDown(self):
        self._tmp.cleanup()

    def test_prepare_drops_every_overlap_and_rows_without_media(self):
        rows, manifest = U.load_pool(self.pool)
        kept = [r["source_id"] for r in rows]
        # P0: c shares an image with X0, d has no picture, e has both pictures and a video; P1: rows 0 (scene),
        # 1 (task), 2 (question id) held out
        self.assertEqual(kept, ["p0_aaaa0000", "p0_bbbb1111", "p1_000000000003", "p1_000000000004", "p1_000000000005"])
        self.assertEqual([r["und_index"] for r in rows], list(range(5)))
        self.assertEqual([r["und_dataset"] for r in rows], ["P0_train"] * 2 + ["P1_train"] * 3)
        self.assertEqual(manifest["per_dataset"], {"P0_train": {"rows": 5, "kept": 2, "excluded": 3},
                                                   "P1_train": {"rows": 6, "kept": 3, "excluded": 3}})
        self.assertEqual(manifest["excluded"], {"P1_probe": {"scenes": 1, "tasks": 1, "ids": 1},
                                                "X0": {"images": 1}, "no_media": {"rows": 1},
                                                "image_and_video": {"rows": 1}})
        self.assertEqual(manifest["videos"], 1)
        self.assertEqual(json.loads((self.pool / "dataset_info.json").read_text()),
                         {U.POOL_DATASET: dict(ATTRS, file_name="pool.jsonl")})
        self.assertEqual((self.pool / "lf_recipe.yaml").read_text(), RECIPE)
        self.assertEqual(manifest["held_out"]["X0"]["questions"], 2)
        self.assertEqual(sorted(manifest["files_sha256"]), ["dataset_info.json", "lf_recipe.yaml"])

    def test_prepare_never_overwrites_a_pool_and_edits_are_caught(self):
        with self.assertRaises(ValueError):
            U.prepare(self.sft, ["P0_train"], [], self.pool)
        lines = (self.pool / "pool.jsonl").read_text().splitlines()
        lines[0] = lines[0].replace('"B"', '"A"')  # same row count, another answer
        (self.pool / "pool.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            U.load_pool(self.pool)
        with self.assertRaises(ValueError):
            U.ppo_block({"pool": str(self.pool), "batch_size": 4}, model="/m", run_root=self.root)

    def test_prepare_refuses_datasets_with_other_columns(self):
        info = json.loads((self.sft / "dataset_info.json").read_text())
        info["P1_train"]["columns"] = {"messages": "conversations"}
        (self.sft / "dataset_info.json").write_text(json.dumps(info))
        with self.assertRaises(ValueError):
            U.prepare(self.sft, ["P0_train", "P1_train"], [], self.root / "pool2")

    def test_ppo_block(self):
        with self.assertRaises(ValueError):  # the default 32 samples from a 5-row pool
            U.ppo_block({"pool": str(self.pool)}, model="/m", run_root=self.root)
        block = U.ppo_block({"pool": str(self.pool), "batch_size": 4}, model="/models/start", run_root=self.root / "run")
        self.assertEqual(block, {"enable": True, "pool": str(self.pool.resolve()), "interval": 2, "weight": 0.1,
                                 "batch_size": 4, "micro_batch_size": 1, "seed": 20261105, "tokenizer": "/models/start",
                                 "log_dir": str((self.root / "run").resolve() / "und_mix"), "pool_rows": 5,
                                 "pool_sha256": self.manifest["pool_sha256"]})
        # a JSON text, and a JSON file
        text = json.dumps({"pool": str(self.pool), "batch_size": 4, "weight": 0.2, "interval": 3, "seed": 7})
        block = U.ppo_block(text, model="/m", run_root=self.root)
        self.assertEqual((block["batch_size"], block["weight"], block["interval"], block["seed"]), (4, 0.2, 3, 7))
        (self.root / "req.json").write_text(text)
        self.assertEqual(U.ppo_block(str(self.root / "req.json"), model="/m", run_root=self.root), block)
        for bad in ({"pool": str(self.pool), "batch_size": 4, "lr": 1}, {"batch_size": 4},
                    {"pool": str(self.root / "nowhere"), "batch_size": 4}, {"pool": str(self.pool), "batch_size": 4,
                                                                            "weight": 0},
                    {"pool": str(self.pool), "batch_size": 2.5}, {"pool": str(self.pool), "batch_size": True},
                    {"pool": str(self.pool), "batch_size": 4, "micro_batch_size": 2},
                    {"pool": str(self.pool), "batch_size": 4, "seed": -1}):
            with self.assertRaises(ValueError, msg=bad):
                U.ppo_block(bad, model="/m", run_root=self.root)

    def test_counts(self):
        log = self.root / "steps.jsonl"
        recs = [{"opt_step": 0, "inserted": True, "draws": [0, 1]}, {"opt_step": 1, "inserted": False},
                {"opt_step": 2, "inserted": True, "draws": [1, 2]},
                {"opt_step": 2, "inserted": True, "draws": [3, 4]}]  # step 2 again after a resume: the last counts
        log.write_text("".join(json.dumps(r) + "\n" for r in recs))
        got = U.counts(self.pool, log)
        self.assertEqual((got["batches"], got["draws"], got["rows_drawn"]), (2, 4, 4))
        self.assertEqual(got["draws_per_row_histogram"], {"0": 1, "1": 4})
        self.assertEqual(got["draws_by_dataset"], {"P0_train": 2, "P1_train": 2})
        # a log of another pool: its rows' ids do not match
        log.write_text(json.dumps({"opt_step": 0, "inserted": True, "draws": [0, 1],
                                   "source_ids": ["p0_aaaa0000", "p1_other"]}) + "\n")
        with self.assertRaises(ValueError):
            U.counts(self.pool, log)


class DrawTest(unittest.TestCase):
    def test_draws_are_fixed_distinct_and_uniform(self):
        a = U.draws(20261105, 4, 100, 32)
        self.assertEqual(a, U.draws(20261105, 4, 100, 32))
        self.assertNotEqual(a, U.draws(20261105, 6, 100, 32))
        self.assertEqual(len(set(a)), 32)
        seen = set()
        for k in range(0, 200, 2):
            seen |= set(U.draws(1, k, 100, 32))
        self.assertEqual(seen, set(range(100)))

    def test_rank_slices(self):
        batch = U.draws(3, 0, 50, 32)
        parts = [U.rank_slice(batch, r, 8) for r in range(8)]
        self.assertEqual([x for p in parts for x in p], batch)
        self.assertEqual({len(p) for p in parts}, {4})
        with self.assertRaises(ValueError):
            U.rank_slice(batch, 0, 5)


class MixGradientsTest(unittest.TestCase):
    def test_the_sum_and_its_measures(self):
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(5)), torch.nn.Parameter(torch.randn(3, 2)),
                  torch.nn.Parameter(torch.randn(4))]
        g_p = [torch.randn_like(p) for p in params]
        g_u = [torch.randn_like(p) for p in params]
        g_u[2] = None  # a parameter the understanding loss does not reach keeps the policy gradient
        for p, g in zip(params, g_p):
            p.grad = g.clone()

        def und_backward():
            for p, g in zip(params, g_u):
                self.assertIsNone(p.grad)  # the policy gradient is set aside meanwhile
                if g is not None:
                    p.grad = g.clone()
            return 7.5

        stats = U.mix_gradients(params, und_backward, lambda values: [float(v) for v in values])
        for p, gp, gu in zip(params, g_p, g_u):
            self.assertTrue(torch.allclose(p.grad, gp + gu if gu is not None else gp))
        flat_p = torch.cat([g.flatten() for g in g_p])
        flat_u = torch.cat([g.flatten() for g in g_u if g is not None])
        flat_pu = torch.cat([g.flatten() for g in g_p[:2]])
        total = torch.cat([(gp + gu).flatten() if gu is not None else gp.flatten() for gp, gu in zip(g_p, g_u)])
        self.assertAlmostEqual(stats["grad_norm_policy"], float(flat_p.norm()), places=5)
        self.assertAlmostEqual(stats["grad_norm_und"], float(flat_u.norm()), places=5)
        self.assertAlmostEqual(stats["grad_norm_total"], float(total.norm()), places=5)
        self.assertAlmostEqual(stats["grad_cos_policy_und"],
                               float(flat_pu @ flat_u / (flat_p.norm() * flat_u.norm())), places=5)
        self.assertAlmostEqual(stats["grad_share_und"], float(flat_u.norm() / total.norm()), places=5)
        self.assertEqual(stats["loss_sum"], 7.5)


class _ToyLM(torch.nn.Module):
    def __init__(self, vocab=11, dim=6):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def log_probs(self, input_ids):
        logits = self.head(self.emb(input_ids))  # token t predicts token t + 1
        return torch.log_softmax(logits[:, :-1], -1).gather(-1, input_ids[:, 1:, None])[..., 0]


class _ToyActor:
    """What UndMixer uses of verl's actor: the module, the optimizer, _forward_micro_batch, no scaler, no SP."""

    def __init__(self):
        torch.manual_seed(1)
        self.actor_module = _ToyLM()
        self.actor_optimizer = torch.optim.AdamW(self.actor_module.parameters(), lr=1e-2)
        self.scaler = None
        self.use_ulysses_sp = False
        self.calls = []

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        self.calls.append(micro_batch)
        lp = self.actor_module.log_probs(micro_batch["input_ids"])
        return None, lp[:, -micro_batch["responses"].shape[1]:]


class _ToyLF:
    """A stand-in for LFPath: a row's tokens are fixed, its last two are the answer."""

    class tokenizer:
        @staticmethod
        def decode(ids, skip_special_tokens=False):
            return " ".join(str(i) for i in ids)

    def features(self, rows):
        return [{"input_ids": [1, 2, 3, 4 + r["und_index"] % 5, 5], "labels": [-100, -100, -100, 4 + r["und_index"] % 5, 5],
                 "row": r["und_index"]} for r in rows]

    def collate(self, features):
        n = max(len(f["input_ids"]) for f in features)
        pad = lambda v, x: v + [x] * (n - len(v))  # noqa: E731
        ids = torch.tensor([pad(f["input_ids"], 0) for f in features])
        att = torch.tensor([pad([1] * len(f["input_ids"]), 0) for f in features])
        lab = torch.tensor([pad(f["labels"], -100) for f in features])
        pos = torch.arange(n).expand(3, len(features), n).clone()
        return {"input_ids": ids, "attention_mask": att, "labels": lab, "position_ids": pos,
                "pixel_values": torch.zeros(1, 2)}


class MixerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        sft = make_sft(self.root / "sft")
        self.pool = self.root / "pool"
        U.prepare(sft, ["P0_train", "P1_train"], [], self.pool)  # 9 rows (the one without media dropped)

    def tearDown(self):
        self._tmp.cleanup()

    def _mixer(self, actor, **cfg):
        block = U.ppo_block({"pool": str(self.pool), "batch_size": 4, **cfg}, model="/m", run_root=self.root / "run")
        return U.UndMixer(block, actor, lf=_ToyLF())

    def test_steps_insert_on_even_counts_and_add_the_weighted_token_mean(self):
        actor = _ToyActor()
        mixer = self._mixer(actor, weight=0.5)
        params = list(actor.actor_module.parameters())
        # step 0: the policy gradient is some fixed tensor; the understanding batch adds 0.5 x its token-mean loss
        policy = [torch.randn_like(p) for p in params]
        for p, g in zip(params, policy):
            p.grad = g.clone()
        out = mixer.before_optimizer_step()
        self.assertEqual((out["actor/und_inserted"], out["actor/und_opt_step"], out["actor/und_samples"]), (1.0, 0.0, 4.0))
        self.assertEqual(out["actor/und_tokens"], 8.0)  # four samples, two answer tokens each
        rows = U.draws(20261105, 0, 9, 4)
        ref = _ToyLM()
        ref.load_state_dict(actor.actor_module.state_dict())
        ids = torch.tensor([[1, 2, 3, 4 + i % 5, 5] for i in rows])
        nll = -ref.log_probs(ids)[:, -2:].sum()
        (0.5 * nll / 8).backward()
        for p, g, r in zip(params, policy, ref.parameters()):
            self.assertTrue(torch.allclose(p.grad, g + r.grad, atol=1e-6))
        self.assertAlmostEqual(out["actor/und_loss"], float(nll) / 8, places=5)
        self.assertEqual(len(actor.calls), 4)  # micro-batches of one sample
        for call in actor.calls:
            self.assertEqual(call["position_ids"].shape, (1, 4, 5))
            self.assertEqual(call["position_ids"][0, 0].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(call["responses"].tolist(), call["input_ids"][:, 3:].tolist())  # from the first answer token
        # the answer-mask check file and the step log
        first = json.loads((self.root / "run" / "und_mix" / "first_batch_rank0.json").read_text())
        self.assertEqual([s["und_index"] for s in first["samples"]], rows)
        self.assertEqual({s["answer_tokens"] for s in first["samples"]}, {2})
        per_row = -ref.log_probs(ids)[:, -2:].sum(dim=1)
        self.assertEqual(len(first["samples"]), 4)
        for s, want in zip(first["samples"], per_row.tolist()):  # each row's loss, for the reference check
            self.assertAlmostEqual(s["nll_sum"], want, places=4)
        self.assertAlmostEqual(out["actor/grad_mag_share_und"] + out["actor/grad_mag_share_policy"], 1.0, places=6)
        self.assertAlmostEqual(out["actor/grad_proj_share_und"] + out["actor/grad_proj_share_policy"], 1.0, places=6)
        actor.actor_optimizer.step()
        actor.actor_optimizer.zero_grad()
        # step 1: no understanding batch, only the policy norm
        for p in params:
            p.grad = torch.ones_like(p)
        out = mixer.before_optimizer_step()
        self.assertEqual((out["actor/und_inserted"], out["actor/und_opt_step"]), (0.0, 1.0))
        self.assertAlmostEqual(out["actor/grad_norm_policy"], math.sqrt(sum(p.numel() for p in params)), places=5)
        for p in params:
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
        self.assertEqual(mixer.end_of_update(), {"actor/und_batches": 1.0, "actor/und_update_opt_steps": 2.0})
        self.assertEqual(mixer.end_of_update(), {"actor/und_batches": 0.0, "actor/und_update_opt_steps": 0.0})
        actor.actor_optimizer.step()
        out = mixer.before_optimizer_step()  # step 2 draws again, other rows
        self.assertEqual(out["actor/und_inserted"], 1.0)
        log = [json.loads(l) for l in (self.root / "run" / "und_mix" / "steps.jsonl").read_text().splitlines()]
        self.assertEqual([(r["opt_step"], r["inserted"]) for r in log], [(0, True), (1, False), (2, True)])
        self.assertEqual(log[0]["draws"], rows)
        self.assertEqual(log[2]["draws"], U.draws(20261105, 2, 9, 4))

    def test_a_nonfinite_understanding_term_is_dropped_and_the_count_moves_on(self):
        actor = _ToyActor()
        mixer = self._mixer(actor)
        params = list(actor.actor_module.parameters())
        policy = [torch.randn_like(p) for p in params]
        for p, g in zip(params, policy):
            p.grad = g.clone()
        real = actor._forward_micro_batch
        actor._forward_micro_batch = lambda mb, temperature, calculate_entropy=False: (
            None, real(mb, temperature)[1] * float("inf"))
        out = mixer.before_optimizer_step()
        self.assertEqual((out["actor/und_inserted"], out["actor/und_dropped"]), (0.0, 1.0))
        for p, g in zip(params, policy):  # the policy gradient alone, untouched
            self.assertTrue(torch.equal(p.grad, g))
        actor.actor_optimizer.step()  # the step happens, so the next insertion (k = 2) draws other rows
        actor._forward_micro_batch = real
        for p in params:
            p.grad = torch.zeros_like(p)
        self.assertEqual(mixer.before_optimizer_step()["actor/und_inserted"], 0.0)  # k = 1: not its turn
        log = [json.loads(l) for l in (self.root / "run" / "und_mix" / "steps.jsonl").read_text().splitlines()]
        self.assertEqual([(r["opt_step"], r["inserted"], r["reason"]) for r in log], [(0, False, "nonfinite"),
                                                                                   (1, False, "interval")])

    def test_a_row_that_fails_skips_the_batch_on_every_rank(self):
        actor = _ToyActor()
        lf = _ToyLF()
        bad = U.draws(20261105, 0, 9, 4)[2]
        real = lf.features
        lf.features = lambda rows: [(_ for _ in ()).throw(ValueError("unreadable"))
                                    if r["und_index"] == bad else real([r])[0] for r in rows]
        block = U.ppo_block({"pool": str(self.pool), "batch_size": 4}, model="/m", run_root=self.root / "run")
        mixer = U.UndMixer(block, actor, lf=lf)
        params = list(actor.actor_module.parameters())
        for p in params:
            p.grad = torch.ones_like(p)
        out = mixer.before_optimizer_step()
        self.assertEqual((out["actor/und_inserted"], out["actor/und_dropped"]), (0.0, 1.0))
        self.assertEqual(actor.calls, [])  # no understanding forward at all
        for p in params:
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
        errors = (self.root / "run" / "und_mix" / "errors_rank0.jsonl").read_text()
        self.assertIn("unreadable", errors)
        log = json.loads((self.root / "run" / "und_mix" / "steps.jsonl").read_text().splitlines()[-1])
        self.assertEqual((log["reason"], log["failed_ranks"]), ("data", 1))

    def test_interval_three_and_refusals(self):
        actor = _ToyActor()
        mixer = self._mixer(actor, interval=3)
        got = []
        for _ in range(6):
            for p in actor.actor_module.parameters():
                p.grad = torch.zeros_like(p)
            got.append(mixer.before_optimizer_step()["actor/und_inserted"])
            actor.actor_optimizer.step()
        self.assertEqual(got, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        actor.scaler = object()
        with self.assertRaises(ValueError):
            self._mixer(actor)
        actor.scaler, actor.use_ulysses_sp = None, True
        with self.assertRaises(ValueError):
            self._mixer(actor)

    def test_a_changed_pool_is_refused(self):
        block = U.ppo_block({"pool": str(self.pool), "batch_size": 4}, model="/m", run_root=self.root / "run")
        with (self.pool / "manifest.json").open("r+") as f:
            m = json.load(f)
            m["pool_sha256"] = "0" * 64
            f.seek(0)
            f.truncate()
            json.dump(m, f)
        with self.assertRaises(ValueError):
            U.UndMixer(block, _ToyActor(), lf=_ToyLF())


@unittest.skipUnless(os.environ.get("UND_MIX_TEST_MODEL"), "UND_MIX_TEST_MODEL: a Qwen3-VL model folder (read only)")
class LFPathTest(unittest.TestCase):
    """The real LLaMA-Factory path with the model's tokenizer and processor: answer tokens only, the RL position rows,
    images and the video's frames."""

    def test_rows_through_the_sft_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft = make_sft(root / "sft")
            pool = root / "pool"
            U.prepare(sft, ["P0_train", "P1_train"], [], pool)
            rows, _ = U.load_pool(pool)
            lf = U.LFPath(pool, os.environ["UND_MIX_TEST_MODEL"])
            self.assertEqual(lf.recipe["template"], "qwen3_vl_nothink")
            for r in rows:
                feat = lf.features([r])[0]
                inputs, mask = U.rl_micro_batch(lf.collate([feat]))
                self.assertEqual(U.answer_text(lf.tokenizer, inputs, mask), U.expected_answer_text(r), r["source_id"])
                self.assertEqual(int(mask.sum()), sum(1 for x in feat["labels"] if x != U.IGNORE_INDEX))
                n = int(inputs["attention_mask"][0].sum())
                self.assertEqual(inputs["position_ids"].shape[:2], (1, 4))
                self.assertEqual(inputs["position_ids"][0, 0, :n].tolist(), list(range(n)))
                media = inputs["multi_modal_inputs"][0]
                if r["images"]:
                    self.assertEqual(int(media["image_grid_thw"].shape[0]), len(r["images"]))
                if r["videos"]:
                    self.assertIn("video_grid_thw", media)
                    self.assertIn("pixel_values_videos", media)
            # two samples in one micro-batch: the response part starts at the earlier first answer token
            feats = lf.features(rows[:2])
            inputs, mask = U.rl_micro_batch(lf.collate(feats))
            self.assertEqual(inputs["input_ids"].shape[0], 2)
            for i, r in enumerate(rows[:2]):
                self.assertEqual(U.answer_text(lf.tokenizer, inputs, mask, i), U.expected_answer_text(r))


if __name__ == "__main__":
    unittest.main()
