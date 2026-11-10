"""Per-reply loss weights in vpd/train/n2_train_vpd.py (round 3, 2026-11-10; docstring item 11).

What is proved here (login node, 1 core, CPU, no GPU, no network, no rendering):
- the weight rules: a reply's own "loss_weight", else the demo's, else 1.0; finite and >= 0; replies only; not all 0;
- with every weight 1.0 (absent or explicit) a demo takes the unweighted path, and that path gives bit-identical
  losses and gradients to the per-demo code of the trainer before loss weights (a verbatim copy of its loop body,
  itself checked against the git blob of n2_train_vpd.py at 68db16b4), on a small fake causal LM;
- the weighted path equals sum(w * CE) / sum(w) computed independently (full logits, per-span weights), weight-0
  spans contribute nothing and a window whose reply weighs 0 is not run;
- DemoData with the real Q3S processor puts each reply's weight on exactly its span tokens;
- end to end on the tiny Qwen3-VL of test_train.py: a default run of the current trainer and a run of the trainer
  at 68db16b4 (exec'd from git) give bit-identical adapters and the same log values; explicit all-1.0 weights
  change nothing; a weighted run records its weights in meta.json and log.jsonl.
On a GPU the claim is the same sequence of operations as before, so a default run is exactly as reproducible as
before the change (GPU kernels themselves need not be bit-deterministic).

    cd $WT && env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 nice -n 19 $PY_Q3 -B -m pytest -q -p no:cacheprovider \
      vpd/tests/test_train_weights.py
Scratch under eval_external_assets/vpd/scratch/train/ (env VPD_TRAIN_TEST_DIR overrides), removed at the end.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import types
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest  # noqa: E402

from vpd.tests import test_train as TT  # noqa: E402  (helpers only: tiny model, demos, skips)
from vpd.train import n2_train_vpd as T  # noqa: E402

WT = Path(__file__).resolve().parents[2]
# the trainer before loss weights: the commit that added it (unchanged until round 3) and its blob sha256
REF_COMMIT = "68db16b4881112620874f07fad45c741ef98183c"
REF_SHA256 = "b91da67ca138365b9a24d2ff8c7ffe64cfd91fe846fad0fa8d18af9967cc9c31"
# the per-demo loss of train() at REF_COMMIT, verbatim (test_reference_snippet_is_verbatim checks it)
REF_SNIPPET = '''        n_reply = sum(e["reply_tokens"] for e in item["encs"])
        demo_ce = 0.0
        for e in item["encs"]:
            inputs = {kk: v.to(model.device) for kk, v in e["inputs"].items()}
            labels = e["labels"].to(model.device)
            with torch.autocast(device_type, dtype=torch.bfloat16):
                ce_sum = response_loss(model, inputs, labels, reduction="sum")
            (ce_sum / n_reply / args.accum).backward()
            demo_ce += float(ce_sum.detach())
'''


@pytest.fixture(scope="module")
def workdir():
    d = TT._make_workdir()
    yield d
    shutil.rmtree(d)  # only what this module created


def _torch():
    return TT._torch()


# ------------------------------------------------------------------------------------------ fake model
def _fake_lm_class():
    torch = _torch()

    class FakeLM(torch.nn.Module):
        """A tiny causal LM with the trainer's model interface: model(input_ids=..., logits_to_keep=idx).logits is
        [1, len(idx), V] and model.device exists.  Elementwise ops only, so bf16 autocast leaves it in float32."""

        def __init__(self, vocab=13, dim=6, seed=0):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.emb = torch.nn.Parameter(torch.randn(vocab, dim, generator=g))
            self.mix = torch.nn.Parameter(torch.randn(dim, dim, generator=g) / math.sqrt(dim))
            self.head = torch.nn.Parameter(torch.randn(dim, vocab, generator=g))

        @property
        def device(self):
            return self.emb.device

        def forward(self, input_ids, logits_to_keep, **_):
            h = self.emb[input_ids[0]]  # [L, d]
            h = torch.cumsum(h, 0) / torch.arange(1, h.shape[0] + 1, dtype=h.dtype).unsqueeze(1)  # causal mean
            h = torch.tanh((h.unsqueeze(2) * self.mix).sum(1))
            logits = (h.unsqueeze(2) * self.head).sum(1)  # [L, V]
            return types.SimpleNamespace(logits=logits[logits_to_keep].unsqueeze(0))

    return FakeLM


def _ids(n: int, seed: int, vocab: int = 13) -> list:
    rnd = random.Random(seed)
    return [rnd.randrange(vocab) for _ in range(n)]


def _enc(ids: list, spans: list, target: str = "all", weights: list = None) -> dict:
    """One encoded sample as DemoData builds it: labels on the trained spans; with weights, span_weights' fields."""
    torch = _torch()
    x = torch.tensor([ids])
    labels = torch.full_like(x, -100)
    for a, b in (spans if target == "all" else spans[-1:]):
        labels[0, a:b] = x[0, a:b]
    e = {"inputs": {"input_ids": x}, "labels": labels, "tokens": len(ids), "reply_tokens": int((labels != -100).sum())}
    if weights is not None:
        e.update(T.span_weights(labels, spans, {"target": target, "weights": weights}))
    return e


# two samples (e.g. two windows of one demo) plus a single-sample demo
SAMPLES = [(_ids(22, 1), [(4, 7), (10, 14), (18, 22)]), (_ids(15, 2), [(3, 6), (9, 15)])]


def _reference_demo(model, item: dict, accum: int, device_type: str = "cpu") -> float:
    """REF_SNIPPET (the trainer's per-demo loss before loss weights) as a function; args.accum -> accum."""
    import torch

    response_loss = T.response_loss  # unmodified since REF_COMMIT
    n_reply = sum(e["reply_tokens"] for e in item["encs"])
    demo_ce = 0.0
    for e in item["encs"]:
        inputs = {kk: v.to(model.device) for kk, v in e["inputs"].items()}
        labels = e["labels"].to(model.device)
        with torch.autocast(device_type, dtype=torch.bfloat16):
            ce_sum = response_loss(model, inputs, labels, reduction="sum")
        (ce_sum / n_reply / accum).backward()
        demo_ce += float(ce_sum.detach())
    return demo_ce / n_reply


def _grads(model) -> dict:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def _formula(model, parts: list, accum: int) -> float:
    """sum(w * CE) / sum(w) / accum from full logits, independently of span_weights: parts = [(ids, spans, ws)]."""
    import torch
    import torch.nn.functional as F

    num, den = 0.0, 0.0
    for ids, spans, ws in parts:
        x = torch.tensor([ids])
        logits = model(input_ids=x, logits_to_keep=torch.arange(len(ids) - 1)).logits[0].float()
        ce = F.cross_entropy(logits, x[0, 1:], reduction="none")  # ce[p] predicts token p + 1
        for (a, b), w in zip(spans, ws):
            num = num + w * ce[a - 1:b - 1].sum()
            den += w * (b - a)
    (num / den / accum).backward()
    return float(num.detach()) / den


# ------------------------------------------------------------------------------------------ pure python
def test_reply_weights_rules():
    rec = TT._demo("w", 3)  # system + 3 x (user, assistant): replies at 2, 4, 6
    assert T.reply_weights(rec) == {2: 1.0, 4: 1.0, 6: 1.0}
    assert [s["weights"] for s in T.samples(rec)] == [[1.0, 1.0, 1.0]]
    r2 = copy.deepcopy(rec)
    r2["loss_weight"] = 2  # the demo default (an int is fine)
    r2["messages"][4]["loss_weight"] = 0  # an injected turn
    r2["messages"][6]["loss_weight"] = 5.0
    assert T.reply_weights(r2) == {2: 2.0, 4: 0.0, 6: 5.0}
    assert T.samples(r2)[0]["weights"] == [2.0, 0.0, 5.0]
    assert [s["weights"] for s in T.samples(r2, 1)] == [[2.0], [0.0], [5.0]]
    assert T.samples(r2, 0, blind=True)[0]["weights"] == [2.0, 0.0, 5.0]
    # the record itself and the chat-template messages are unaffected by the extra key
    assert "loss_weight" not in str(T.chat_messages(T.samples(r2)[0]["messages"]))
    r5 = copy.deepcopy(r2)
    r5["messages"][4]["loss_weight"] = -0.0
    assert str(T.reply_weights(r5)[4]) == "0.0"  # -0.0 is 0, stored as 0.0
    for bad in (-1, -1e-9, float("nan"), float("inf"), "1", True, None, [1]):
        r3 = copy.deepcopy(rec)
        r3["messages"][2]["loss_weight"] = bad
        with pytest.raises(ValueError, match="loss_weight"):
            T.reply_weights(r3)
        with pytest.raises(ValueError, match="loss_weight"):
            T.reply_weights(dict(rec, loss_weight=bad))
    r4 = copy.deepcopy(rec)
    r4["messages"][1]["loss_weight"] = 1.0  # a user turn
    with pytest.raises(ValueError, match="only replies"):
        T.reply_weights(r4)
    with pytest.raises(ValueError, match="every reply"):
        T.reply_weights(dict(rec, loss_weight=0.0))
    assert T.is_uniform([1.0, 1, 1.0]) and T.is_uniform([]) and not T.is_uniform([1.0, 1.0000001])


def test_weight_stats():
    rec = TT._demo("u", 3)
    ones = copy.deepcopy(rec)
    for m in ones["messages"]:
        if m["role"] == "assistant":
            m["loss_weight"] = 1.0
    w = copy.deepcopy(rec)
    w["messages"][2]["loss_weight"] = 5.0
    w["messages"][4]["loss_weight"] = 0.0
    st = T.weight_stats([rec, ones])
    assert st["mode"] == "uniform" and st["demos_weighted"] == 0 and st["replies_by_weight"] == {"1.0": 6}
    st = T.weight_stats([rec, w])
    assert st["mode"] == "weighted" and st["demos"] == 2 and st["demos_weighted"] == 1 and st["replies"] == 6
    assert st["replies_by_weight"] == {"0.0": 1, "1.0": 4, "5.0": 1} and st["zero_weight_replies"] == 1
    assert st["reply_weight_mean"] == pytest.approx(9 / 6)
    with pytest.raises(ValueError):
        T.weight_stats([dict(rec, loss_weight=-2)])


def test_span_weights_labels_and_tensor():
    torch = _torch()
    ids, spans = SAMPLES[0]
    e = _enc(ids, spans)
    r = T.span_weights(e["labels"], spans, {"target": "all", "weights": [1.0, 0.0, 3.0]})
    lab, wt = r["labels"], r["weights"]
    assert wt.dtype == torch.float32 and wt.shape == e["labels"].shape
    assert bool((lab[0, 10:14] == -100).all()) and bool((wt[0, 10:14] == 0).all())  # the weight-0 span
    assert torch.equal(lab[0, 4:7], e["labels"][0, 4:7]) and torch.equal(lab[0, 18:22], e["labels"][0, 18:22])
    assert bool((wt[0, 4:7] == 1).all()) and bool((wt[0, 18:22] == 3).all())
    assert float(wt.sum()) == r["weight_sum"] == 3 * 1 + 4 * 3
    assert r["reply_tokens"] == 7 and r["masked_tokens"] == 4
    assert bool((wt[lab == -100] == 0).all()) and bool((wt[lab != -100] > 0).all())
    assert torch.equal(e["labels"], _enc(ids, spans)["labels"])  # input labels not modified
    last = T.span_weights(_enc(ids, spans, "last")["labels"], spans, {"target": "last", "weights": [0.5]})
    assert last["weight_sum"] == 0.5 * 4 and bool((last["weights"][0, :18] == 0).all())
    with pytest.raises(ValueError, match="reply weights"):
        T.span_weights(e["labels"], spans, {"target": "all", "weights": [1.0, 2.0]})


def test_reference_snippet_is_verbatim():
    """REF_SNIPPET is the per-demo loss of n2_train_vpd.py at REF_COMMIT (git blob, sha256 pinned)."""
    src = _ref_source()
    assert REF_SNIPPET in src
    assert src.count('response_loss(model, inputs, labels, reduction="sum")') == 1  # the only loss call in train()


# ------------------------------------------------------------------------------------------ fake-model steps
def test_uniform_step_bitwise_equals_reference():
    """The default path of demo_step == the trainer's code before loss weights, bit for bit (losses, gradients),
    for a 2-sample demo and a 1-sample demo, with weights absent and with every weight explicitly 1.0."""
    torch = _torch()
    FakeLM = _fake_lm_class()
    for parts in (SAMPLES, SAMPLES[1:]):
        for explicit in (False, True):
            ref, new = FakeLM(), FakeLM()
            encs = [_enc(ids, sp) for ids, sp in parts]
            item = {"encs": encs}
            if explicit:  # DemoData's flag for all-1.0 weights
                item["weighted"] = not T.is_uniform([1.0] * sum(len(sp) for _, sp in parts))
            loss_ref = _reference_demo(ref, {"encs": [_enc(ids, sp) for ids, sp in parts]}, 8)
            r = T.demo_step(new, item, 8, "cpu")
            assert r["loss"] == loss_ref and r["loss_unweighted"] == loss_ref
            gr, gn = _grads(ref), _grads(new)
            assert all(torch.equal(gr[n], gn[n]) for n in gr)
            assert r["reply_tokens"] == sum(e["reply_tokens"] for e in encs) == r["weight"]
            assert r["tokens"] == sum(len(ids) for ids, _ in parts) and r["masked_tokens"] == 0


def test_weighted_step_equals_formula():
    """sum(w * CE) / sum(w) over the demo's reply tokens, all samples together; weight 0 contributes nothing."""
    torch = _torch()
    FakeLM = _fake_lm_class()
    cases = [
        [[1.0, 5.0, 3.0], [2.0, 1.0]],  # STOP / turn style weights
        [[1.0, 0.0, 1.0], [0.0, 1.0]],  # injected (weight-0) turns
        [[1.0, 1.0, 1.0], [1.0, 1.0]],  # the weighted path with all ones (not used by the trainer for such demos)
        [[2.5, 2.5, 2.5], [2.5, 2.5]],  # a constant weight: the normalisation gives the unweighted loss
    ]
    for ws in cases:
        a, b = FakeLM(), FakeLM()
        item = {"weighted": True, "encs": [_enc(ids, sp, weights=w) for (ids, sp), w in zip(SAMPLES, ws)]}
        r = T.demo_step(a, item, 8, "cpu")
        want = _formula(b, [(ids, sp, w) for (ids, sp), w in zip(SAMPLES, ws)], 8)
        assert r["loss"] == pytest.approx(want, rel=1e-6, abs=1e-7)
        ga, gb = _grads(a), _grads(b)
        assert all(torch.allclose(ga[n], gb[n], rtol=1e-5, atol=1e-8) for n in ga), ws
        n_pos = sum(b_ - a_ for (_, sp), w in zip(SAMPLES, ws) for (a_, b_), x in zip(sp, w) if x > 0)
        assert r["reply_tokens"] == n_pos and r["weight"] == pytest.approx(
            sum(x * (b_ - a_) for (_, sp), w in zip(SAMPLES, ws) for (a_, b_), x in zip(sp, w)))
    # all ones / a constant weight on the weighted path == the reference (unweighted) loss, up to rounding
    ref = FakeLM()
    loss_ref = _reference_demo(ref, {"encs": [_enc(ids, sp) for ids, sp in SAMPLES]}, 8)
    for c in (1.0, 2.5):
        m = FakeLM()
        item = {"weighted": True, "encs": [_enc(ids, sp, weights=[c] * len(sp)) for ids, sp in SAMPLES]}
        r = T.demo_step(m, item, 8, "cpu")
        assert r["loss"] == pytest.approx(loss_ref, rel=1e-6)
        gr, gm = _grads(ref), _grads(m)
        assert all(torch.allclose(gr[n], gm[n], rtol=1e-5, atol=1e-8) for n in gr)
    # a weight-0 span == the same demo with that span unlabelled on the unweighted path (gradients and loss)
    z, u = FakeLM(), FakeLM()
    item = {"weighted": True, "encs": [_enc(SAMPLES[0][0], SAMPLES[0][1], weights=[1.0, 0.0, 1.0])]}
    r = T.demo_step(z, item, 8, "cpu")
    loss_u = _reference_demo(u, {"encs": [_enc(SAMPLES[0][0], [SAMPLES[0][1][0], SAMPLES[0][1][2]])]}, 8)
    assert r["loss"] == pytest.approx(loss_u, rel=1e-6) and r["masked_tokens"] == 4
    gz, gu = _grads(z), _grads(u)
    assert all(torch.allclose(gz[n], gu[n], rtol=1e-5, atol=1e-8) for n in gz)


def test_weighted_step_skips_zero_weight_window():
    """A window (history-k sample) whose trained reply weighs 0 is not run: same result as the demo without it."""
    torch = _torch()
    FakeLM = _fake_lm_class()
    (ids0, sp0), (ids1, sp1) = SAMPLES
    calls = []

    a, b = FakeLM(), FakeLM()
    orig = a.forward
    a.forward = lambda *x, **k: calls.append(1) or orig(*x, **k)
    item = {"weighted": True, "encs": [_enc(ids0, sp0, "last", [3.0]), _enc(ids1, sp1, "last", [0.0])]}
    r = T.demo_step(a, item, 8, "cpu")
    assert len(calls) == 1 and r["samples_not_run"] == 1 and r["masked_tokens"] == sp1[-1][1] - sp1[-1][0]
    want = _formula(b, [(ids0, sp0[-1:], [3.0])], 8)
    assert r["loss"] == pytest.approx(want, rel=1e-6)
    ga, gb = _grads(a), _grads(b)
    assert all(torch.allclose(ga[n], gb[n], rtol=1e-5, atol=1e-8) for n in ga)
    with pytest.raises(ValueError, match="positive loss weight"):
        T.demo_step(FakeLM(), {"weighted": True, "id": "z", "encs": [_enc(ids1, sp1, "last", [0.0])]}, 8, "cpu")


# ------------------------------------------------------------------------------------------ real processor
def test_demodata_weighted_encoding(workdir):
    """DemoData (Q3S processor, real images): unweighted demos come back exactly as before (same keys and labels
    as encode); a weighted demo carries each reply's weight on exactly its span; history-k windows too."""
    torch = _torch()
    TT._transformers()
    from vpd import starts

    proc = starts.load_processor("Q3S")
    rec = json.loads(TT._demo_file(workdir).read_text().splitlines()[1])  # d_b, 3 turns, 3 images
    w = copy.deepcopy(rec)
    w["messages"][2]["loss_weight"] = 1.0
    w["messages"][4]["loss_weight"] = 0.0
    w["messages"][6]["loss_weight"] = 5.0
    ds = T.DemoData(proc, [rec, w])
    plain = ds[(0, 0)]
    assert plain["weighted"] is False and plain["skip"] is None and len(plain["encs"]) == 1
    e0 = plain["encs"][0]
    assert set(e0) == {"inputs", "labels", "tokens", "reply_tokens"}
    inputs, labels, spans, _ = T.encode(proc, T.samples(rec)[0])
    assert torch.equal(e0["labels"], labels) and torch.equal(e0["inputs"]["input_ids"], inputs["input_ids"])
    assert e0["reply_tokens"] == int((labels != -100).sum())
    item = ds[(0, 1)]
    assert item["weighted"] is True
    e = item["encs"][0]
    (a1, b1), (a2, b2), (a3, b3) = spans
    assert torch.equal(e["inputs"]["input_ids"], inputs["input_ids"])  # same tokens, same images
    assert bool((e["labels"][0, a2:b2] == -100).all()) and torch.equal(e["labels"][0, a1:b1], labels[0, a1:b1])
    assert bool((e["weights"][0, a1:b1] == 1).all()) and bool((e["weights"][0, a3:b3] == 5).all())
    assert float(e["weights"].sum()) == e["weight_sum"] == (b1 - a1) + 5 * (b3 - a3)
    assert e["reply_tokens"] == (b1 - a1) + (b3 - a3) and e["masked_tokens"] == b2 - a2
    tok = proc.tokenizer
    assert tok.decode(inputs["input_ids"][0, a3:b3]) == w["messages"][6]["content"] + T.END  # span = that reply
    wk = T.DemoData(proc, [w], history_k=1)[(0, 0)]
    assert [x["weight_sum"] > 0 for x in wk["encs"]] == [True, False, True]
    assert wk["encs"][2]["weight_sum"] == 5 * int((T.encode(proc, T.samples(w, 1)[2])[1] != -100).sum())


# ------------------------------------------------------------------------------------------ end to end (tiny)
_REF = {}


def _ref_source() -> str:
    if "src" not in _REF:
        if shutil.which("git") is None or not (WT / ".git").exists():
            pytest.skip("no git checkout")
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(WT), "show",
                            f"{REF_COMMIT}:vpd/train/n2_train_vpd.py"], capture_output=True, timeout=60)
        if r.returncode != 0:
            pytest.skip(f"git show {REF_COMMIT[:8]} failed: {r.stderr.decode()[-200:]}")
        assert hashlib.sha256(r.stdout).hexdigest() == REF_SHA256
        _REF["src"] = r.stdout.decode("utf-8")
    return _REF["src"]


def _ref_trainer():
    """The trainer at REF_COMMIT as a module (exec'd with the real file path, so it finds the same worktree)."""
    if "mod" not in _REF:
        src = _ref_source()
        TT._lora()
        importlib.import_module("vpd.starts")
        saved = list(sys.path)
        try:
            mod = types.ModuleType("n2_train_vpd_ref68db16b4")
            mod.__file__ = T.__file__
            exec(compile(src, f"n2_train_vpd.py@{REF_COMMIT[:8]}", "exec"), mod.__dict__)  # noqa: S102
        finally:
            sys.path[:] = saved
        _REF["mod"] = mod
    return _REF["mod"]


def _log(out: Path) -> list:
    timing = {"seconds", "step_seconds", "wait_seconds"}
    return [{k: v for k, v in json.loads(x).items() if k not in timing} for x in (out / "log.jsonl").read_text().splitlines()]


def _same_adapter(a: Path, b: Path) -> bool:
    torch = _torch()
    sa, sb = torch.load(a / "adapter.pt"), torch.load(b / "adapter.pt")
    return set(sa) == set(sb) and all(torch.equal(sa[n]["A"], sb[n]["A"]) and torch.equal(sa[n]["B"], sb[n]["B"])
                                      for n in sa)


def _with_weights(src: Path, dst: Path, fn) -> Path:
    recs = [json.loads(x) for x in src.read_text().splitlines()]
    for r in recs:
        fn(r)
    dst.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return dst


def _common(workdir: Path, data: Path) -> list:
    return ["--start", "Q3S", "--model-path", str(TT._tiny_saved(workdir)), "--data", str(data), "--device", "cpu",
            "--threads", "1", "--accum", "2", "--epochs", "2", "--seed", "1001", "--phase", "t"]


def test_default_run_identical_to_reference_trainer(workdir):
    """Default runs: the current trainer and the trainer at 68db16b4 give the same adapter (torch.equal) and log
    values, full history and --history-k 1; explicit all-1.0 weights give the same adapter again."""
    _torch()
    TT._transformers()
    ref = _ref_trainer()
    data = TT._demo_file(workdir)
    for extra, tag in (([], "k0"), (["--history-k", "1"], "k1")):
        outs = {}
        for name, mod in (("ref", ref), ("new", T)):
            out = workdir / f"e2e_{tag}_{name}"
            assert mod.main(_common(workdir, data) + ["--out", str(out), "--num-workers", "0"] + extra) == 0
            outs[name] = out
        assert _same_adapter(outs["ref"], outs["new"]), tag
        assert _log(outs["ref"]) == _log(outs["new"]), tag  # same keys too: a default run logs nothing new
        mr, mn = (json.loads((outs[k] / "meta.json").read_text()) for k in ("ref", "new"))
        for k in ("optimizer_steps", "micro_steps", "loss_first", "loss_last", "loss_last10_mean", "tokens_total",
                  "reply_tokens_total", "max_len", "skipped", "data_sha256"):
            assert mr[k] == mn[k], (tag, k)
        assert "loss_weights" not in mr
        lw = mn["loss_weights"]
        assert lw["mode"] == "uniform" and lw["demos_weighted"] == 0 and lw["replies"] == 6
        assert lw["reply_weight_total"] == mn["reply_tokens_total"] and lw["masked_reply_tokens_total"] == 0
    ones = _with_weights(data, workdir / "demos3_ones.jsonl", lambda r: [m.__setitem__("loss_weight", 1.0)
                                                                         for m in r["messages"] if m["role"] == "assistant"])
    out = workdir / "e2e_k0_ones"
    assert T.main(_common(workdir, ones) + ["--out", str(out), "--num-workers", "0"]) == 0
    assert _same_adapter(out, workdir / "e2e_k0_new") and _log(out) == _log(workdir / "e2e_k0_new")


def test_weighted_run_tiny(workdir):
    """A weighted run (STOP-style x5 on one reply, an injected weight-0 reply): trains, records its weights, logs
    loss_unweighted / reply_weight, inline == DataLoader; differs from the default run.  A constant weight on every
    reply is the unweighted objective again (adapter equal up to float rounding)."""
    torch = _torch()
    TT._transformers()
    data = TT._demo_file(workdir)
    base = workdir / "e2e_k0_new"
    if not (base / "adapter.pt").is_file():
        assert T.main(_common(workdir, data) + ["--out", str(base), "--num-workers", "0"]) == 0

    def weigh(r):
        a = [m for m in r["messages"] if m["role"] == "assistant"]
        if r["id"] == "d_a":
            a[-1]["loss_weight"] = 5.0
        if r["id"] == "d_b":
            a[1]["loss_weight"] = 0.0

    wdata = _with_weights(data, workdir / "demos3_w.jsonl", weigh)
    outs = {}
    for nw in (0, 1):
        out = workdir / f"e2e_w_{nw}"
        assert T.main(_common(workdir, wdata) + ["--out", str(out), "--num-workers", str(nw)]) == 0
        outs[nw] = out
    assert _same_adapter(outs[0], outs[1]) and _log(outs[0]) == _log(outs[1])
    assert not _same_adapter(outs[0], base)
    meta = json.loads((outs[0] / "meta.json").read_text())
    lw = meta["loss_weights"]
    assert lw["mode"] == "weighted" and lw["demos_weighted"] == 2 and lw["zero_weight_replies"] == 1
    assert lw["replies_by_weight"] == {"0.0": 1, "1.0": 4, "5.0": 1}
    assert lw["masked_reply_tokens_total"] > 0 and lw["samples_not_run"] == 0
    assert lw["reply_weight_total"] > meta["reply_tokens_total"]  # the x5 reply outweighs the masked one
    logs = _log(outs[0])
    assert all({"loss_unweighted", "reply_weight"} <= set(x) for x in logs) and len(logs) == 3
    base_meta = json.loads((base / "meta.json").read_text())
    assert meta["reply_tokens_total"] < base_meta["reply_tokens_total"]  # d_b's weight-0 reply, in both epochs
    # constant weight 3.0 everywhere: same objective as the default run
    const = _with_weights(data, workdir / "demos3_c.jsonl", lambda r: r.__setitem__("loss_weight", 3.0))
    out = workdir / "e2e_c"
    assert T.main(_common(workdir, const) + ["--out", str(out), "--num-workers", "0"]) == 0
    a, b = torch.load(out / "adapter.pt"), torch.load(base / "adapter.pt")
    diff = math.sqrt(sum(float((a[n][x] - b[n][x]).pow(2).sum()) for n in a for x in "AB"))
    norm = math.sqrt(sum(float(b[n][x].pow(2).sum()) for n in b for x in "AB"))
    assert diff / norm < 1e-3, diff / norm  # bf16 autocast: rounding only, not a different objective
    lc, lb = _log(out), _log(base)
    assert lc[0]["loss"] == pytest.approx(lb[0]["loss"], rel=1e-5)  # step 1: same model, same objective
    assert all(x["loss"] == pytest.approx(y["loss"], rel=1e-3) for x, y in zip(lc, lb))
    assert all(x["loss_unweighted"] == pytest.approx(x["loss"], rel=1e-5) for x in lc)
    assert json.loads((out / "meta.json").read_text())["loss_weights"]["mode"] == "weighted"


def test_dry_run_reports_weights(workdir):
    TT._transformers()
    wdata = _with_weights(TT._demo_file(workdir), workdir / "demos3_dry.jsonl",
                          lambda r: r["messages"][2].__setitem__("loss_weight", 2.0))
    out = workdir / "dry_w"
    assert T.main(["--start", "Q3S", "--data", str(wdata), "--out", str(out), "--dry-run", "--limit", "3"]) == 0
    lw = json.loads((out / "dry_run.json").read_text())["loss_weights"]
    assert lw["mode"] == "weighted" and lw["demos_weighted"] == 3 and lw["replies_by_weight"] == {"1.0": 3, "2.0": 3}
    bad = _with_weights(TT._demo_file(workdir), workdir / "demos3_bad.jsonl",
                        lambda r: r["messages"][1].__setitem__("loss_weight", 2.0))
    with pytest.raises(ValueError, match="only replies"):  # refused before any processor or model is loaded
        T.main(["--start", "Q3S", "--data", str(bad), "--out", str(workdir / "dry_bad"), "--dry-run"])
