#!/usr/bin/env python3
"""VPD trainer: LoRA SFT of one VPD start (vpd/starts.json) on WalkerBench demonstrations, one GPU.

Source: a copy of downstream/n/n2_train.py at bf4b2553 (VAGEN-Lite-vpd, branch viewpoint-downstream), changed for the
VPD line (WP-TRAIN; plan section 3.4; DECISIONS B1, B7, B10, B16).  downstream/train/lora.py is imported read-only
from the worktree (WORKTREE/downstream/train); downstream/x/q0_ask.py is no longer needed (B1: starts come from
vpd/starts.json).

    PY_TRAIN vpd/train/n2_train_vpd.py --start Q3S --data demos_50.jsonl --out ADAPTER_DIR --epochs auto \\
        --seed 1001 [--init-adapter 0W_DIR] [--history-k 2] [--blind] [--attn sdpa|fa2] [--num-workers 3]
    PY_TRAIN vpd/train/n2_train_vpd.py --start Q3S --data demos.jsonl --out DIR --dry-run --limit 3 --expect-grid 24x40

Recipe (unchanged from N2): LoRA rank 16, alpha 32, dropout 0.05 on the language model's attention and MLP
projections (lora.py), everything else frozen including the vision tower; AdamW (no weight decay), lr 1e-4, 3% linear
warm-up then cosine to 0; one conversation per micro-step, gradients accumulated over 8, clip 1.0; bf16 autocast with
float32 LoRA masters; gradient checkpointing; every conversation through processor.apply_chat_template with all its
turns and images; loss on the assistant replies only (each reply's tokens and its <|im_end|>).

Changes against n2_train.py:
 1. --start NAME (vpd/starts.json) and --model-path (weights override); loaded with vpd.starts.load_model(attn=--attn,
    default sdpa), which asserts the attention implementation on the model, its language model and its vision tower
    (B10) and bf16 everywhere; the processor is capped at max_pixels 1,024,000 (B16).  meta: start, code, path, attn.
 2. Response-only lm_head: logits only at the positions that predict a reply token (logits_to_keep=idx), the CE
    computed here; equal to the HF labels= loss (tests).
 3. DataLoader prefetch (--num-workers, default 3; prefetch 2; batch_size None) over the main process's seeded
    order (random.Random(seed), one shuffle per epoch, as before); workers run 1 torch thread, the main process
    --threads (default 4).  --num-workers 0 encodes inline.
 4. --epochs auto = max(3, ceil(240 / N)), N = conversations; meta['epochs_rule'].
 5. --init-adapter DIR: DIR/adapter.pt copied into the fresh LoRA fp32 masters (names and shapes must match
    exactly); fresh optimizer and schedule; meta records its sha256.
 6. --history-k K: each demo expands to vpd.walkerbench.history.train_windows(messages, K); each window trains its
    last reply only; per demo the summed token-CE over its windows / its reply tokens; one demo = one micro-step;
    each window carries only the images of its own user turns.
 7. --blind: '<image>' stripped, no images (as eval_external run.py --blind).
 8. --max-seq-tokens (default 40000): longer samples are skipped and listed in meta['skipped'].
 9. log.jsonl per optimizer step (step, epoch, phase, loss, lr, tokens, seconds, max_memory_allocated reset each
    step); meta: max_len, tokens_per_s, skipped, code_sha256 (B7), versions.
10. --dry-run --limit N: processor only (no model, CPU): per conversation the total, image and reply tokens, the
    image grids (--expect-grid 24x40 asserts [1,24,40] per image) and the span check; dry_run.json carries, for the
    first 3 demos, the sha256 of the rendered chat text, input_ids, labels and pixel_values (and per-prefix prompt
    token counts) for vpd/train/parity_check.py.
11. Optional loss weights (round 3, 2026-11-10; off unless the data carries them).  An assistant message may carry a
    float "loss_weight" >= 0; a demo may carry a record-level "loss_weight" used for its replies without one; absent
    = 1.0.  A demo whose replies all weigh exactly 1.0 trains on the unweighted code path above, operation for
    operation (tests: vpd/tests/test_train_weights.py, against the trainer at commit 68db16b4).  Any other demo:
    loss = sum over its reply tokens of w * CE / sum of w (each reply span's tokens, its <|im_end|> included, carry
    that reply's weight; all its samples together); weight-0 replies are left unlabelled, and a --history-k window
    whose trained reply weighs 0 is not run.  meta / dry_run.json 'loss_weights': the weight counts; a weighted run
    also logs loss_unweighted (the plain CE mean over the w > 0 tokens) and reply_weight per optimizer step.
Save format unchanged: OUT/adapter.pt ({module: {"A", "B"}} float32), OUT/meta.json, OUT/log.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent  # WORKTREE/vpd/train
WORKTREE = HERE.parent.parent
# B1: the read-only N2 modules, from the worktree root (not relative to vpd/train)
sys.path.insert(0, str(WORKTREE / "downstream" / "x"))
sys.path.insert(0, str(WORKTREE / "downstream" / "train"))
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from vpd import paths  # noqa: E402
from vpd.walkerbench import history  # noqa: E402

RANK, ALPHA, DROPOUT = 16, 32.0, 0.05
LR, WARMUP, ACCUM, EPOCHS, SEED = 1e-4, 0.03, 8, 3, 1
EPOCHS_AUTO_TARGET, EPOCHS_AUTO_MIN = 240, 3
MAX_SEQ_TOKENS = 40000
NUM_WORKERS, PREFETCH, MAIN_THREADS = 3, 2, 4
PARITY_DEMOS = 3
REPLY_START = "<|im_start|>assistant\n"
END = "<|im_end|>"
IMAGE = "<image>"
LOSS_WEIGHT = "loss_weight"  # optional, per assistant message or per demo (docstring item 11)
# files whose bytes define the training code (B7: pilot_gate asserts equal code_sha256 across starts)
CODE_FILES = ("vpd/train/n2_train_vpd.py", "downstream/train/lora.py", "vpd/starts.py", "vpd/starts.json",
              "vpd/paths.py", "vpd/walkerbench/history.py")


# ------------------------------------------------------------------------------------------ conversations
def chat_messages(messages: list) -> list:
    """sharegpt messages as chat-template messages, each "<image>" an image part (as n2_train.py)."""
    out = []
    for m in messages:
        parts = []
        for k, piece in enumerate(m["content"].split(IMAGE)):
            if k:
                parts.append({"type": "image"})
            if piece:
                parts.append({"type": "text", "text": piece})
        out.append({"role": m["role"], "content": parts})
    return out


def image_slots(messages: list) -> tuple[list, int]:
    """Per message, the indices into rec['images'] of its '<image>' placeholders (in order); and their total."""
    out, n = [], 0
    for m in messages:
        c = m["content"].count(IMAGE)
        out.append(list(range(n, n + c)))
        n += c
    return out, n


def samples(rec: dict, history_k: int = 0, blind: bool = False) -> list:
    """The training samples of one demo: [{"messages", "images", "target": "all"|"last", "turn", "weights"}].

    history_k 0: the whole conversation, every reply trained.  history_k K > 0: one sample per assistant turn i, the
    clean window history.window(messages[:i], K) + messages[i], only the last reply trained, only the images of the
    window's user turns.  blind: '<image>' removed from every message and no images.  weights: the loss weight of
    each trained reply, in order (reply_weights)."""
    msgs = rec["messages"]
    imgs = list(rec.get("images") or [])
    slots, n = image_slots(msgs)
    if n != len(imgs):
        raise ValueError(f"{rec.get('id')}: {n} '<image>' placeholders for {len(imgs)} images")
    rw = reply_weights(rec)
    if history_k == 0:
        out = [{"messages": list(msgs), "images": imgs, "target": "all", "turn": None, "weights": list(rw.values())}]
    else:
        pos = {id(m): j for j, m in enumerate(msgs)}
        out = []
        for ctx, i in history.train_windows(msgs, history_k):
            kept = [pos[id(m)] for m in ctx] + [i]
            out.append({"messages": [msgs[j] for j in kept], "images": [imgs[s] for j in kept for s in slots[j]],
                        "target": "last", "turn": i, "weights": [rw[i]]})
    if blind:
        for s in out:
            s["messages"] = [dict(m, content=m["content"].replace(IMAGE, "")) for m in s["messages"]]
            s["images"] = []
    return out


def reply_spans(ids, start_ids, end_id) -> list:
    """[(first, last + 1)] of every assistant reply in a token list: after each REPLY_START, up to and including the
    next end-of-turn token."""
    ids, n, spans, i = list(ids), len(start_ids), [], 0
    while i <= len(ids) - n:
        if ids[i:i + n] == list(start_ids):
            j = i + n
            while j < len(ids) and ids[j] != end_id:
                j += 1
            spans.append((i + n, min(j + 1, len(ids))))
            i = j
        else:
            i += 1
    return spans


def load_images(files: list) -> list:
    from PIL import Image

    out = []
    for p in files:
        with Image.open(p) as im:
            out.append(im.convert("RGB"))
    return out


def encode(processor, sample: dict, add_generation_prompt: bool = False):
    """(model inputs, labels, spans, text) of one sample.  target 'all': every reply span labelled; 'last': only
    the final one.  With add_generation_prompt (a prompt prefix ending with a user turn) nothing is labelled."""
    text = processor.apply_chat_template(chat_messages(sample["messages"]), tokenize=False,
                                         add_generation_prompt=add_generation_prompt)
    images = load_images(sample["images"])
    inputs = processor(text=[text], images=images or None, return_tensors="pt")
    tok = processor.tokenizer
    spans = reply_spans(inputs["input_ids"][0].tolist(), tok.encode(REPLY_START, add_special_tokens=False),
                        tok.convert_tokens_to_ids(END))
    labels = inputs["input_ids"].clone().fill_(-100)
    if not add_generation_prompt:
        for a, b in (spans if sample["target"] == "all" else spans[-1:]):
            labels[0, a:b] = inputs["input_ids"][0, a:b]
    return inputs, labels, spans, text


def check_spans(rec_id, messages: list, spans: list) -> None:
    turns = sum(m["role"] == "assistant" for m in messages)
    if len(spans) != turns:
        raise ValueError(f"{rec_id}: {len(spans)} reply spans for {turns} assistant turns")


# ------------------------------------------------------------------------------------------ loss weights
def _weight(v, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: {LOSS_WEIGHT} must be a number, got {v!r}")
    w = float(v)
    if not math.isfinite(w) or w < 0:
        raise ValueError(f"{where}: {LOSS_WEIGHT} must be finite and >= 0, got {v!r}")
    return 0.0 if w == 0 else w  # -0.0 -> 0.0


def reply_weights(rec: dict) -> dict:
    """{message index: loss weight} of every assistant message of a demo, in message order: the message's own
    "loss_weight", else the demo's "loss_weight", else 1.0.  Weights are finite numbers >= 0; only assistant
    messages may carry one; a demo with replies needs at least one weight > 0."""
    rid = rec.get("id")
    default = _weight(rec[LOSS_WEIGHT], f"{rid}") if LOSS_WEIGHT in rec else 1.0
    out = {}
    for j, m in enumerate(rec["messages"]):
        if m.get("role") != "assistant":
            if LOSS_WEIGHT in m:
                raise ValueError(f"{rid}: message {j} ({m.get('role')}) has a {LOSS_WEIGHT}; only replies may")
            continue
        out[j] = _weight(m[LOSS_WEIGHT], f"{rid} message {j}") if LOSS_WEIGHT in m else default
    if out and not any(w > 0 for w in out.values()):
        raise ValueError(f"{rid}: every reply has {LOSS_WEIGHT} 0")
    return out


def is_uniform(weights) -> bool:
    """True iff every weight is exactly 1.0 (such a demo trains on the unweighted code path)."""
    return all(w == 1.0 for w in weights)


def span_weights(labels, spans: list, sample: dict) -> dict:
    """The weighted form of one encoded sample: each trained reply span's tokens carry that reply's weight
    (sample["weights"], in span order), at the label positions; a weight-0 span is unlabelled (-100), so the lm_head
    never runs on it.  {"labels", "weights" (float32, labels' shape, 0 off the labelled tokens), "weight_sum"
    (sum of w over the labelled tokens), "reply_tokens" (labelled tokens), "masked_tokens"}."""
    import torch

    trained = spans if sample["target"] == "all" else spans[-1:]
    ws = list(sample["weights"])
    if len(ws) != len(trained):
        raise ValueError(f"{len(ws)} reply weights for {len(trained)} trained reply spans")
    lab = labels.clone()
    wt = torch.zeros(labels.shape, dtype=torch.float32)
    total, masked = 0.0, 0
    for (a, b), w in zip(trained, ws):
        if w > 0:
            wt[0, a:b] = w
            total += w * (b - a)
        else:
            lab[0, a:b] = -100
            masked += b - a
    return {"labels": lab, "weights": wt, "weight_sum": total, "reply_tokens": int((lab != -100).sum()),
            "masked_tokens": masked}


def weight_stats(records: list) -> dict:
    """The loss-weight facts of a demo file (validates every weight): meta / dry_run.json 'loss_weights'."""
    by, weighted = Counter(), 0
    for rec in records:
        ws = list(reply_weights(rec).values())
        weighted += not is_uniform(ws)
        by.update(ws)
    n = sum(by.values())
    return {"mode": "weighted" if weighted else "uniform", "demos": len(records), "demos_weighted": weighted,
            "replies": n, "replies_by_weight": {str(w): by[w] for w in sorted(by)},
            "zero_weight_replies": by.get(0.0, 0),
            "reply_weight_mean": (sum(w * c for w, c in by.items()) / n) if n else None,
            "rule": "per demo: sum(w * CE) / sum(w) over its reply tokens; w = the reply's loss_weight, else the "
                    "demo's, else 1.0; weight-0 replies unlabelled; all-1.0 demos on the unweighted path"}


def lr_at(step: int, total: int, lr: float = LR, warmup: float = WARMUP) -> float:
    """Linear warm-up over the first warmup share of the steps (at least one), then cosine to 0."""
    w = max(1, int(math.ceil(warmup * total)))
    if step < w:
        return lr * (step + 1) / w
    return lr * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def lr_phase(step: int, total: int, warmup: float = WARMUP) -> str:
    return "warmup" if step < max(1, int(math.ceil(warmup * total))) else "cosine"


def epochs_for(spec, n: int) -> tuple[int, str]:
    """(epochs, rule).  'auto' = max(3, ceil(240 / n)) (plan 3.4 item 4); an int is used as given."""
    if str(spec) == "auto":
        if n <= 0:
            raise ValueError("--epochs auto needs at least one conversation")
        e = max(EPOCHS_AUTO_MIN, int(math.ceil(EPOCHS_AUTO_TARGET / n)))
        return e, f"auto: max({EPOCHS_AUTO_MIN}, ceil({EPOCHS_AUTO_TARGET}/N)), N={n} -> {e}"
    e = int(spec)
    if e < 1:
        raise ValueError(f"--epochs must be >= 1 or 'auto', got {spec!r}")
    return e, f"fixed: {e}"


def parse_grid(spec):
    if not spec:
        return None
    h, _, w = str(spec).lower().partition("x")
    return [1, int(h), int(w)]


# ------------------------------------------------------------------------------------------------ hashes
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_ids(t) -> str:
    return sha256_bytes(",".join(str(int(x)) for x in t.reshape(-1).tolist()).encode())


def sha256_tensor(t) -> str:
    import numpy as np

    return sha256_bytes(np.ascontiguousarray(t.detach().cpu().float().numpy()).tobytes())


def code_fingerprint(root: Path = WORKTREE) -> dict:
    """sha256 of every CODE_FILES file and one sha over all of them (B7)."""
    files = {}
    for rel in CODE_FILES:
        p = Path(root) / rel
        files[rel] = sha256_file(p) if p.is_file() else None
    total = sha256_bytes("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode())
    return {"code_sha256": total, "code_files": files}


def git_head(root: Path = WORKTREE) -> dict:
    """Read-only git facts of the worktree (None where git is unavailable)."""
    def run(*a):
        try:
            r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), *a], capture_output=True,
                               text=True, timeout=30)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    status = run("status", "--porcelain", "--", "vpd")
    return {"git_head": run("rev-parse", "HEAD"), "vpd_dirty": None if status is None else bool(status)}


def versions() -> dict:
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("torch", "transformers", "PIL", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - provenance only
            out[mod] = None
    return out


# ------------------------------------------------------------------------------------------------ adapter
def adapter_state(model) -> dict:
    import lora

    return {n: {"A": m.lora_A.detach().float().cpu(), "B": m.lora_B.detach().float().cpu()}
            for n, m in model.named_modules() if isinstance(m, lora.LoRALinear)}


def merge_adapter(model, path, scale: float = None) -> int:
    """Add scale * B A (scale = alpha / rank from the adapter's meta.json next to it) to each named base weight, in
    memory; the module names are the trained model's (same class, same loader).  Returns the modules changed.
    (Copied unchanged from n2_train.py.)"""
    import torch

    path = Path(path)
    if scale is None:
        meta = json.loads((path.parent / "meta.json").read_text())
        scale = float(meta["alpha"]) / int(meta["rank"])
    state = torch.load(path, map_location="cpu")
    with torch.no_grad():
        for name, ab in state.items():
            lin = model.get_submodule(name)
            w = lin.weight
            w.add_((ab["B"].float() @ ab["A"].float() * scale).to(device=w.device, dtype=w.dtype))
    return len(state)


def load_init_adapter(model, adapter_dir, rank: int = None, alpha: float = None) -> dict:
    """Copy DIR/adapter.pt into the wrapped LoRA modules' A/B (their float32 masters).  The module name set and every
    shape must match exactly; rank/alpha must equal DIR/meta.json's when it exists.  Returns provenance."""
    import torch

    import lora

    d = Path(adapter_dir)
    p = d / "adapter.pt"
    state = torch.load(p, map_location="cpu")
    mods = {n: m for n, m in model.named_modules() if isinstance(m, lora.LoRALinear)}
    if set(state) != set(mods):
        only_a, only_m = sorted(set(state) - set(mods)), sorted(set(mods) - set(state))
        raise ValueError(f"{p}: module names differ: {len(only_a)} only in the adapter (e.g. {only_a[:2]}), "
                         f"{len(only_m)} only in the model (e.g. {only_m[:2]})")
    for n, m in mods.items():
        a, b = state[n]["A"], state[n]["B"]
        if tuple(a.shape) != tuple(m.lora_A.shape) or tuple(b.shape) != tuple(m.lora_B.shape):
            raise ValueError(f"{p}: {n}: A {tuple(a.shape)} / B {tuple(b.shape)} vs model "
                             f"{tuple(m.lora_A.shape)} / {tuple(m.lora_B.shape)}")
    src_meta = {}
    if (d / "meta.json").is_file():
        src_meta = json.loads((d / "meta.json").read_text())
        for key, want in (("rank", rank), ("alpha", alpha)):
            if want is not None and key in src_meta and float(src_meta[key]) != float(want):
                raise ValueError(f"{d}/meta.json: {key} {src_meta[key]} != {want}")
    with torch.no_grad():
        for n, m in mods.items():
            m.lora_A.copy_(state[n]["A"].to(device=m.lora_A.device, dtype=m.lora_A.dtype))
            m.lora_B.copy_(state[n]["B"].to(device=m.lora_B.device, dtype=m.lora_B.dtype))
    return {"dir": str(d), "sha256": sha256_file(p), "modules": len(mods), "start": src_meta.get("start"),
            "data": src_meta.get("data"), "code_sha256": src_meta.get("code_sha256")}


# ------------------------------------------------------------------------------------------------- loss
def response_loss(model, inputs, labels, reduction: str = "mean"):
    """CE on the reply tokens only, with the lm_head run only where it predicts one (plan 3.4 item 2):
    idx = positions p with labels[p + 1] != -100; logits_to_keep=idx; CE(logits, labels[idx + 1])."""
    import torch.nn.functional as F

    idx = (labels[0, 1:] != -100).nonzero().squeeze(1)
    if idx.numel() == 0:
        raise ValueError("no reply tokens in this sample")
    out = model(**inputs, logits_to_keep=idx)
    return F.cross_entropy(out.logits[0].float(), labels[0, idx + 1], reduction=reduction)


def weighted_response_loss(model, inputs, labels, weights):
    """(sum of w * CE, sum of CE) over the reply tokens: response_loss(reduction='none') on the same positions, each
    token's CE times weights[0, p + 1] (span_weights' tensor, aligned with labels)."""
    ce = response_loss(model, inputs, labels, reduction="none")
    idx = (labels[0, 1:] != -100).nonzero().squeeze(1)
    w = weights.to(device=ce.device, dtype=ce.dtype)[0, idx + 1]
    return (ce * w).sum(), ce.detach().sum()


def demo_step(model, item: dict, accum: int, device_type: str) -> dict:
    """Forward and backward of one encoded demo (one micro-step); returns its numbers for the log.

    Unweighted demo (item["weighted"] false; all its reply weights 1.0): the trainer's code before loss weights,
    operation for operation (n2_train_vpd.py at 68db16b4, the loop body of train()): per sample ce_sum =
    response_loss(reduction='sum'), backward of ce_sum / n_reply / accum, loss = sum(ce_sum) / n_reply.
    Weighted demo: W = sum of w over the labelled tokens of all its samples; per sample with a positive weight,
    backward of sum(w * CE) / W / accum; a sample whose weight sum is 0 is not run; loss = sum(w * CE) / W,
    loss_unweighted = sum(CE) / (labelled tokens)."""
    import torch

    encs = item["encs"]
    if not item.get("weighted"):
        n_reply = sum(e["reply_tokens"] for e in encs)
        demo_ce = 0.0
        tokens, max_len = 0, 0
        for e in encs:
            inputs = {kk: v.to(model.device) for kk, v in e["inputs"].items()}
            labels = e["labels"].to(model.device)
            with torch.autocast(device_type, dtype=torch.bfloat16):
                ce_sum = response_loss(model, inputs, labels, reduction="sum")
            (ce_sum / n_reply / accum).backward()
            demo_ce += float(ce_sum.detach())
            tokens += e["tokens"]
            max_len = max(max_len, e["tokens"])
        loss = demo_ce / n_reply
        return {"loss": loss, "loss_unweighted": loss, "weight": float(n_reply), "reply_tokens": n_reply,
                "tokens": tokens, "max_len": max_len, "masked_tokens": 0, "samples_not_run": 0}
    run = [e for e in encs if e["weight_sum"] > 0]
    big_w = sum(e["weight_sum"] for e in run)
    n_reply = sum(e["reply_tokens"] for e in run)
    if not run or big_w <= 0 or n_reply <= 0:
        raise ValueError(f"{item.get('id')}: no reply with a positive loss weight")
    wce, ce_all, tokens, max_len = 0.0, 0.0, 0, 0
    for e in run:
        inputs = {kk: v.to(model.device) for kk, v in e["inputs"].items()}
        labels = e["labels"].to(model.device)
        with torch.autocast(device_type, dtype=torch.bfloat16):
            wsum, csum = weighted_response_loss(model, inputs, labels, e["weights"])
        (wsum / big_w / accum).backward()
        wce += float(wsum.detach())
        ce_all += float(csum)
        tokens += e["tokens"]
        max_len = max(max_len, e["tokens"])
    return {"loss": wce / big_w, "loss_unweighted": ce_all / n_reply, "weight": big_w, "reply_tokens": n_reply,
            "tokens": tokens, "max_len": max_len, "masked_tokens": sum(e["masked_tokens"] for e in encs),
            "samples_not_run": len(encs) - len(run)}


# ------------------------------------------------------------------------------------------------- data
class DemoData:
    """Map-style dataset over (epoch, record index) keys: one item = one demo, already encoded (all its samples).
    Runs in the DataLoader workers; an over-long demo comes back as {"skip": tokens}.  item["weighted"]: some reply
    weight of the demo is not 1.0; its encs then carry span_weights' labels, weights and sums."""

    def __init__(self, processor, records: list, history_k: int = 0, blind: bool = False,
                 max_seq_tokens: int = MAX_SEQ_TOKENS):
        self.processor, self.records = processor, records
        self.history_k, self.blind, self.max_seq_tokens = history_k, blind, max_seq_tokens

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        epoch, k = key
        rec = self.records[k]
        smp = samples(rec, self.history_k, self.blind)
        weighted = not is_uniform(w for s in smp for w in s["weights"])
        encs = []
        for s in smp:
            inputs, labels, spans, _ = encode(self.processor, s)
            check_spans(rec.get("id"), s["messages"], spans)
            n = int(inputs["input_ids"].shape[1])
            if n > self.max_seq_tokens:
                return {"key": key, "id": rec.get("id"), "skip": n, "encs": []}
            e = {"inputs": dict(inputs), "labels": labels, "tokens": n, "reply_tokens": int((labels != -100).sum())}
            if weighted:
                e.update(span_weights(labels, spans, s))
            encs.append(e)
        return {"key": key, "id": rec.get("id"), "skip": None, "encs": encs, "weighted": weighted}


def _identity(x):
    return x


def _worker_init(_):
    import torch

    torch.set_num_threads(1)


def epoch_keys(n: int, epochs: int, seed: int) -> list:
    """The training order: one random.Random(seed), a fresh shuffle of range(n) per epoch (as n2_train.py)."""
    rng = random.Random(seed)
    keys = []
    for epoch in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        keys.extend((epoch, k) for k in order)
    return keys


def items(ds: DemoData, keys: list, num_workers: int, prefetch: int = PREFETCH, seed: int = SEED):
    """The encoded demos in `keys` order: inline with 0 workers, else a prefetching DataLoader.  The DataLoader
    gets its own generator: its worker base seed would otherwise be drawn from the global torch RNG and shift the
    LoRA dropout masks, so 0 and N workers train identically (tested)."""
    if num_workers <= 0:
        for key in keys:
            yield ds[key]
        return
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=None, sampler=keys, num_workers=num_workers, prefetch_factor=prefetch,
                    worker_init_fn=_worker_init, collate_fn=_identity, persistent_workers=False,
                    generator=torch.Generator().manual_seed(int(seed)))
    yield from dl


# ------------------------------------------------------------------------------------------------- train
def load_records(path: Path, limit: int = None) -> list:
    recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return recs[:limit] if limit else recs


def start_entry(args) -> dict:
    from vpd import starts

    e = starts.resolve(args.start)
    return {"start": args.start, "code": e["code"], "path": str(args.model_path or e["path"]),
            "registry_path": e["path"], "model_path_override": None if args.model_path is None else str(args.model_path)}


def image_token_id(processor):
    tok = getattr(processor, "image_token", None) or "<|image_pad|>"
    return processor.tokenizer.convert_tokens_to_ids(tok)


def describe(processor, rec: dict, args, expect_grid) -> dict:
    """Dry-run numbers of one demo (every sample encoded)."""
    img_id = image_token_id(processor)
    out = {"id": rec.get("id"), "samples": 0, "tokens": [], "image_tokens": 0, "reply_tokens": 0, "images": 0,
           "grids": [], "over_limit": []}
    for s in samples(rec, args.history_k, args.blind):
        inputs, labels, spans, _ = encode(processor, s)
        check_spans(rec.get("id"), s["messages"], spans)
        ids = inputs["input_ids"][0]
        n = int(ids.shape[0])
        out["samples"] += 1
        out["tokens"].append(n)
        out["image_tokens"] += int((ids == img_id).sum())
        out["reply_tokens"] += int((labels != -100).sum())
        out["images"] += len(s["images"])
        grid = inputs.get("image_grid_thw")
        rows = [] if grid is None else [list(map(int, r)) for r in grid.tolist()]
        if len(rows) != len(s["images"]):
            raise ValueError(f"{rec.get('id')}: {len(rows)} grids for {len(s['images'])} images")
        if expect_grid is not None:
            bad = [r for r in rows if r != expect_grid]
            if bad:
                raise AssertionError(f"{rec.get('id')}: image grid {bad[0]} != expected {expect_grid}")
        out["grids"].extend(rows)
        if n > args.max_seq_tokens:
            out["over_limit"].append({"turn": s["turn"], "tokens": n})
    out["grid_set"] = sorted({tuple(r) for r in out["grids"]})
    del out["grids"]
    return out


def parity_record(processor, rec: dict, history_k: int, blind: bool, out_dir: Path = None, save_px0: bool = False):
    """Hashes of the first sample of one demo, and of every prompt prefix (context + generation prompt) of it, for
    parity_check.py (and the smoke job's vLLM prompt-token comparison, B5)."""
    s = samples(rec, history_k, blind)[0]
    inputs, labels, spans, text = encode(processor, s)
    pv = inputs.get("pixel_values")
    entry = {"id": rec.get("id"), "sample": 0, "turn": s["turn"], "n_images": len(s["images"]),
             "image_files": list(s["images"]),
             "image_rgb_sha256": [sha256_bytes(im.tobytes()) for im in load_images(s["images"])],
             "text_sha256": sha256_bytes(text.encode("utf-8")), "text_chars": len(text),
             "input_ids_sha256": sha256_ids(inputs["input_ids"]), "n_tokens": int(inputs["input_ids"].shape[1]),
             "labels_sha256": sha256_ids(labels), "spans": [list(x) for x in spans],
             "grid": None if inputs.get("image_grid_thw") is None else inputs["image_grid_thw"].tolist(),
             "pixel_values_sha256": None if pv is None else sha256_tensor(pv),
             "pixel_values_shape": None if pv is None else list(pv.shape)}
    if save_px0 and pv is not None and out_dir is not None:
        import torch

        g = inputs["image_grid_thw"][0]
        rows = int(g[0] * g[1] * g[2])
        torch.save(pv[:rows].detach().float().clone(), out_dir / "dry_run_px0.pt")
        entry["px0_file"] = "dry_run_px0.pt"
    entry["prefixes"] = [dict(p, input_ids_sha256=sha256_ids(pin["input_ids"]),
                              text_sha256=sha256_bytes(ptext.encode("utf-8")),
                              prompt_tokens=int(pin["input_ids"].shape[1]))
                         for p, pin, ptext in prompt_prefixes(processor, rec, history_k, blind)]
    return entry


def prefix_contexts(rec: dict, history_k: int = 0, blind: bool = False) -> list:
    """[(assistant turn i, context messages ending with the user turn, their images)]: what the eval agent sends
    at each step (full history for K 0, the clean K-window otherwise)."""
    if history_k:
        return [(x["turn"], x["messages"][:-1], x["images"]) for x in samples(rec, history_k, blind)]
    s = samples(rec, 0, blind)[0]
    msgs = s["messages"]
    slots, _ = image_slots(msgs)
    return [(i, msgs[:i], [s["images"][q] for j in range(i) for q in slots[j]])
            for i, m in enumerate(msgs) if m["role"] == "assistant"]


def prompt_prefixes(processor, rec: dict, history_k: int = 0, blind: bool = False):
    """Yield ({"turn", "n_images"}, inputs, text) of every step's prompt: the context + the generation prompt, as
    the training processor tokenizes it (B5: the smoke job compares prompt_tokens with vLLM's usage)."""
    for i, msgs, imgs in prefix_contexts(rec, history_k, blind):
        pin, _, _, ptext = encode(processor, {"messages": msgs, "images": imgs, "target": "all"},
                                  add_generation_prompt=True)
        yield {"turn": i, "n_images": len(imgs)}, pin, ptext


def dry_run(args, records: list) -> int:
    import torch

    from vpd import starts

    torch.set_num_threads(args.threads or 1)
    processor = starts.load_processor(args.start, path=args.model_path, max_pixels=args.max_pixels)
    expect = parse_grid(args.expect_grid)
    per, lens, replies = [], [], []
    for k, rec in enumerate(records):
        d = describe(processor, rec, args, expect)
        per.append(d)
        lens.extend(d["tokens"])
        replies.append(d["reply_tokens"])
        print(f"[n2_train_vpd] dry-run {rec.get('id')}: {d['samples']} sample(s), tokens max {max(d['tokens'])}, "
              f"images {d['images']}, image tokens {d['image_tokens']}, reply tokens {d['reply_tokens']}, "
              f"grids {d['grid_set']}", flush=True)
        if k == 0:
            s = samples(rec, args.history_k, args.blind)[0]
            inputs, _, spans, _ = encode(processor, s)
            ids = inputs["input_ids"][0].tolist()
            tk = processor.tokenizer
            marked = "".join(("[[" + tk.decode(ids[a:b]) + "]]" + tk.decode(
                ids[b:(spans[i + 1][0] if i + 1 < len(spans) else len(ids))])) for i, (a, b) in enumerate(spans))
            (args.out / "dry_run_sample.txt").write_text(tk.decode(ids[:spans[0][0]]) + marked + "\n",
                                                         encoding="utf-8")
    parity = [parity_record(processor, rec, args.history_k, args.blind, args.out, save_px0=(k == 0))
              for k, rec in enumerate(records[:PARITY_DEMOS])]
    ent = start_entry(args)
    rep = {"start": args.start, "code": ent["code"], "model_path": ent["model_path_override"], "path": ent["path"],
           "processor_dir": processor.vpd_processor_dir, "max_pixels": args.max_pixels,
           "data": str(args.data), "data_sha256": sha256_file(args.data), "limit": args.limit,
           "history_k": args.history_k, "blind": args.blind, "expect_grid": expect,
           "max_seq_tokens": args.max_seq_tokens, "conversations": len(records),
           "tokens_max": max(lens), "tokens_mean": sum(lens) / len(lens), "trained_tokens": sum(replies),
           "over_limit": {d["id"]: d["over_limit"] for d in per if d["over_limit"]},
           "per_conversation": per, "parity": parity,
           "loss_weights": getattr(args, "loss_weights", None) or weight_stats(records),
           "versions": versions(), **code_fingerprint()}
    (args.out / "dry_run.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("start", "conversations", "tokens_max", "tokens_mean", "trained_tokens",
                                          "processor_dir", "max_pixels")}), flush=True)
    return 0


def train(args, records: list) -> int:
    import torch

    import lora
    from vpd import starts

    torch.set_num_threads(args.threads or MAIN_THREADS)
    epochs, rule = epochs_for(args.epochs, len(records))
    cuda = str(args.device).startswith("cuda")
    t0 = time.time()
    torch.manual_seed(args.seed)
    model, processor = starts.load_model(args.start, device=args.device, attn=args.attn, path=args.model_path,
                                         max_pixels=args.max_pixels)
    load_info = dict(model.vpd_load_info)
    print(f"[n2_train_vpd] attention {load_info['attn_sites']}", flush=True)
    load_s = time.time() - t0
    model.config.use_cache = False
    info = lora.wrap(model, args.rank, args.alpha, args.dropout, args.seed)
    for m in model.modules():
        if isinstance(m, lora.LoRALinear):  # float32 masters; autocast runs the products in bf16
            m.lora_A.data, m.lora_B.data = m.lora_A.data.float(), m.lora_B.data.float()
    init = None
    if args.init_adapter:
        init = load_init_adapter(model, args.init_adapter, args.rank, args.alpha)
        print(f"[n2_train_vpd] init adapter {init['dir']} ({init['modules']} modules, sha256 {init['sha256'][:12]})",
              flush=True)
    frozen = lora.freeze(model, trainable_prefixes=())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = int(math.ceil(len(records) * epochs / args.accum))
    keys = epoch_keys(len(records), epochs, args.seed)
    ds = DemoData(processor, records, args.history_k, args.blind, args.max_seq_tokens)
    wstats = getattr(args, "loss_weights", None) or weight_stats(records)
    weighted_run = wstats["mode"] == "weighted"
    step, micro, acc, acc_n, losses = 0, 0, 0.0, 0, []
    step_tok, step_reply, step_wait, tok_total, reply_total, max_len = 0, 0, 0.0, 0, 0, 0
    step_unw, step_weight = 0.0, 0.0  # logged in weighted runs only
    wtot = {"reply_weight_total": 0.0, "masked_reply_tokens_total": 0, "samples_not_run": 0}
    skipped, peaks = [], []
    log = open(args.out / "log.jsonl", "w")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t1 = t_step = time.time()
    device_type = "cuda" if cuda else "cpu"

    def optimizer_step(epoch):
        nonlocal step, acc, acc_n, step_tok, step_reply, step_wait, t_step, step_unw, step_weight
        lr = lr_at(step, total, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        now = time.time()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
        peaks.append(peak or 0.0)
        rec = {"step": step + 1, "epoch": epoch + 1, "phase": args.phase, "lr_phase": lr_phase(step, total),
               "loss": round(acc / acc_n, 5), "lr": lr, "demos": acc_n, "tokens": step_tok,
               "reply_tokens": step_reply, "seconds": round(now - t1, 1), "step_seconds": round(now - t_step, 2),
               "wait_seconds": round(step_wait, 2),
               "max_memory_allocated_gib": None if peak is None else round(peak, 3)}
        if weighted_run:
            rec["loss_unweighted"] = round(step_unw / acc_n, 5)
            rec["reply_weight"] = round(step_weight, 4)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        losses.append(rec["loss"])
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        step, acc, acc_n, step_tok, step_reply, step_wait, t_step = step + 1, 0.0, 0, 0, 0, 0.0, now
        step_unw, step_weight = 0.0, 0.0
        if step % 10 == 0 or step == total:
            print(f"[n2_train_vpd] step {step}/{total} loss {rec['loss']:.4f} ({rec['seconds']} s, "
                  f"{rec['tokens']} tok in {rec['step_seconds']} s)", flush=True)

    epoch = 0
    t_wait = time.time()
    for item in items(ds, keys, args.num_workers, args.prefetch, args.seed):
        step_wait += time.time() - t_wait
        epoch, k = item["key"]
        if item["skip"] is not None:
            skipped.append({"id": item["id"], "epoch": epoch + 1, "tokens": item["skip"]})
            print(f"[n2_train_vpd] skipped {item['id']} ({item['skip']} tokens > {args.max_seq_tokens})", flush=True)
            t_wait = time.time()
            continue
        r = demo_step(model, item, args.accum, device_type)
        step_tok += r["tokens"]
        max_len = max(max_len, r["max_len"])
        tok_total += r["tokens"]
        reply_total += r["reply_tokens"]
        step_reply += r["reply_tokens"]
        acc, acc_n = acc + r["loss"], acc_n + 1
        step_unw, step_weight = step_unw + r["loss_unweighted"], step_weight + r["weight"]
        wtot["reply_weight_total"] += r["weight"]
        wtot["masked_reply_tokens_total"] += r["masked_tokens"]
        wtot["samples_not_run"] += r["samples_not_run"]
        micro += 1
        if micro % args.accum == 0:
            optimizer_step(epoch)
        t_wait = time.time()
    if acc_n:
        optimizer_step(epoch)
    log.close()
    train_s = time.time() - t1
    torch.save(adapter_state(model), args.out / "adapter.pt")
    ent = start_entry(args)
    meta = {"model": args.start, "start": args.start, "code": ent["code"], "path": ent["path"],
            "model_path": ent["path"], "registry_path": ent["registry_path"],
            "model_path_override": ent["model_path_override"], "attn": load_info["attn"],
            "attn_sites": load_info["attn_sites"], "processor": load_info["processor_dir"],
            "max_pixels": load_info["max_pixels"], "phase": args.phase,
            "data": str(args.data), "data_sha256": sha256_file(args.data), "conversations": len(records),
            "epochs": epochs, "epochs_rule": rule, "accum": args.accum, "lr": args.lr, "warmup": WARMUP,
            "schedule": "linear warm-up, cosine to 0", "seed": args.seed, "rank": args.rank, "alpha": args.alpha,
            "dropout": args.dropout, "lora": info, "frozen": frozen, "init_adapter": init,
            "history_k": args.history_k, "blind": args.blind, "max_seq_tokens": args.max_seq_tokens,
            "skipped": skipped, "num_workers": args.num_workers, "prefetch_factor": args.prefetch,
            "threads": args.threads or MAIN_THREADS, "device": str(args.device),
            "optimizer_steps": step, "optimizer_steps_planned": total, "micro_steps": micro,
            "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
            "loss_last10_mean": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
            "tokens_total": tok_total, "reply_tokens_total": reply_total, "max_len": max_len,
            "tokens_per_s": round(tok_total / train_s, 1) if train_s > 0 else None,
            "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1),
            "peak_mem_gib": round(max(peaks), 2) if (cuda and peaks) else None,
            "loss_weights": {**wstats, "reply_weight_total": round(wtot["reply_weight_total"], 4),
                             "masked_reply_tokens_total": wtot["masked_reply_tokens_total"],
                             "samples_not_run": wtot["samples_not_run"]},
            "versions": versions(), **code_fingerprint(), **git_head()}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[n2_train_vpd] done: {step} steps in {train_s:.0f} s, loss {meta['loss_first']} -> {meta['loss_last']}, "
          f"{meta['tokens_per_s']} tok/s, skipped {len(skipped)}", flush=True)
    return 0 if step else 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="a start of vpd/starts.json")
    ap.add_argument("--model-path", type=Path, default=None, help="weights folder overriding the start's path")
    ap.add_argument("--attn", default="sdpa", choices=("sdpa", "fa2"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", default=str(EPOCHS), help="an integer or 'auto' = max(3, ceil(240/N))")
    ap.add_argument("--accum", type=int, default=ACCUM)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--init-adapter", type=Path, default=None, help="DIR with adapter.pt (and meta.json)")
    ap.add_argument("--history-k", type=int, default=0)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--max-seq-tokens", type=int, default=MAX_SEQ_TOKENS)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--prefetch", type=int, default=PREFETCH)
    ap.add_argument("--threads", type=int, default=None,
                    help=f"torch threads of the main process (default {MAIN_THREADS}; dry run 1)")
    ap.add_argument("--max-pixels", type=int, default=paths.MAX_PIXELS)
    ap.add_argument("--phase", default="train", help="a label written to every log line (e.g. 0W, tier50)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N conversations")
    ap.add_argument("--expect-grid", default=None, help="dry run: assert every image grid is [1,H,W], e.g. 24x40")
    return ap


def main(argv=None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args(argv)
    if args.history_k < 0:
        raise SystemExit("--history-k must be >= 0")
    records = load_records(args.data, args.limit)
    if not records:
        raise SystemExit(f"{args.data}: no conversations")
    args.loss_weights = weight_stats(records)  # validates every loss_weight before any model is loaded
    args.out.mkdir(parents=True, exist_ok=True)
    return dry_run(args, records) if args.dry_run else train(args, records)


if __name__ == "__main__":
    sys.exit(main())
