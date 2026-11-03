#!/usr/bin/env python3
"""N2 (few-shot; the user's decisions 2 of 2026-11-02 and 1 of 2026-11-03): LoRA on one of the eight models from
N2's demonstrations (n2_data.py), in the StarVLA environment's Python on one GPU.

    python downstream/n/n2_train.py --model B --data DATA/qwen25/demos_100.jsonl --out ADAPTER [--epochs 3]
    python downstream/n/n2_train.py --model Q3S --data DATA/qwen3/demos_100.jsonl --out DIR --dry-run

Recipe (N_DESIGN section 4, N2_PLAN): LoRA rank 16, alpha 32, dropout 0.05 on the language model's attention and
feed-forward projections (downstream/train/lora.py, as T2), everything else frozen, the vision tower included; AdamW
(no weight decay), learning rate 1e-4 with a 3% linear warm-up then cosine to 0; one conversation per forward pass,
gradients accumulated over 8; 3 epochs, the conversations reshuffled each epoch (seeded); bf16 autocast with the
LoRA matrices kept in float32; gradient checkpointing.  The model and processor are loaded as the evaluation's
server loads them (q0_ask.load), and each conversation goes through the same chat template
(processor.apply_chat_template) with all its turns and images; the loss is on the assistant replies only (each
reply's tokens and its end-of-turn token), every earlier turn kept in the input, as the main line's SFT
(LLaMA-Factory, mask_history false).  Not LLaMA-Factory itself: the Qwen3-VL runs need the model line's own
environment and registrations, and this way training and evaluation share one environment and one template.
--dry-run: no model and no GPU; the processor only: every conversation's token count and reply spans, the first
  one decoded with the trained tokens marked (OUT/dry_run.json, OUT/dry_run_sample.txt).
Writes OUT/adapter.pt ({module name: {"A", "B"}} float32), OUT/meta.json (recipe, data, steps, seconds, losses,
peak memory), OUT/log.jsonl (one line per optimizer step).  merge_adapter(model, path) adds alpha / rank * B A to
the base weights in memory (the evaluation's server, n_policy_server.py --adapter; nothing is written).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "x"))
sys.path.insert(0, str(HERE.parent / "train"))

RANK, ALPHA, DROPOUT = 16, 32.0, 0.05
LR, WARMUP, ACCUM, EPOCHS, SEED = 1e-4, 0.03, 8, 3, 1
REPLY_START = "<|im_start|>assistant\n"
END = "<|im_end|>"


# ------------------------------------------------------------------------------------------ conversations
def chat_messages(rec: dict) -> list:
    """sharegpt messages (n2_data.py) as chat-template messages, each "<image>" an image part (n_episode.chat)."""
    out = []
    for m in rec["messages"]:
        parts = []
        for k, piece in enumerate(m["content"].split("<image>")):
            if k:
                parts.append({"type": "image"})
            if piece:
                parts.append({"type": "text", "text": piece})
        out.append({"role": m["role"], "content": parts})
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


def encode(processor, rec: dict):
    """(model inputs, labels, spans) of one conversation."""
    from PIL import Image

    text = processor.apply_chat_template(chat_messages(rec), tokenize=False, add_generation_prompt=False)
    images = [Image.open(p).convert("RGB") for p in rec["images"]]
    inputs = processor(text=[text], images=images or None, return_tensors="pt")
    tok = processor.tokenizer
    spans = reply_spans(inputs["input_ids"][0].tolist(), tok.encode(REPLY_START, add_special_tokens=False),
                        tok.convert_tokens_to_ids(END))
    labels = inputs["input_ids"].clone().fill_(-100)
    for a, b in spans:
        labels[0, a:b] = inputs["input_ids"][0, a:b]
    return inputs, labels, spans, text


def check_spans(rec: dict, spans: list) -> None:
    turns = sum(m["role"] == "assistant" for m in rec["messages"])
    if len(spans) != turns:
        raise ValueError(f"{rec['id']}: {len(spans)} reply spans for {turns} assistant turns")


def lr_at(step: int, total: int, lr: float = LR, warmup: float = WARMUP) -> float:
    """Linear warm-up over the first warmup share of the steps (at least one), then cosine to 0."""
    w = max(1, int(math.ceil(warmup * total)))
    if step < w:
        return lr * (step + 1) / w
    return lr * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


# ------------------------------------------------------------------------------------------------ adapter
def adapter_state(model) -> dict:
    import lora

    return {n: {"A": m.lora_A.detach().float().cpu(), "B": m.lora_B.detach().float().cpu()}
            for n, m in model.named_modules() if isinstance(m, lora.LoRALinear)}


def merge_adapter(model, path, scale: float = None) -> int:
    """Add scale * B A (scale = alpha / rank from the adapter's meta.json next to it) to each named base weight, in
    memory; the module names are the trained model's (same class, same loader).  Returns the modules changed."""
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


# ------------------------------------------------------------------------------------------------- train
def dry_run(args, records: list) -> int:
    import q0_ask
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(q0_ask.processor_dir(args.model))
    lens, replies = [], []
    for k, rec in enumerate(records):
        inputs, labels, spans, text = encode(processor, rec)
        check_spans(rec, spans)
        lens.append(int(inputs["input_ids"].shape[1]))
        replies.append(int((labels != -100).sum()))
        if k == 0:
            ids = inputs["input_ids"][0].tolist()
            marked = "".join(("[[" + processor.tokenizer.decode(ids[a:b]) + "]]" + processor.tokenizer.decode(
                ids[b:(spans[i + 1][0] if i + 1 < len(spans) else len(ids))])) for i, (a, b) in enumerate(spans))
            (args.out / "dry_run_sample.txt").write_text(
                processor.tokenizer.decode(ids[:spans[0][0]]) + marked + "\n", encoding="utf-8")
    rep = {"model": args.model, "data": str(args.data), "conversations": len(records), "tokens_max": max(lens),
           "tokens_mean": sum(lens) / len(lens), "trained_tokens": sum(replies)}
    (args.out / "dry_run.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep), flush=True)
    return 0


def train(args, records: list) -> int:
    import torch

    import lora
    import q0_ask

    t0 = time.time()
    torch.manual_seed(args.seed)
    processor, model, proc_dir = q0_ask.load(args.model)
    load_s = time.time() - t0
    model.config.use_cache = False
    info = lora.wrap(model, args.rank, args.alpha, args.dropout, args.seed)
    for m in model.modules():
        if isinstance(m, lora.LoRALinear):  # float32 masters; autocast runs the products in bf16
            m.lora_A.data, m.lora_B.data = m.lora_A.data.float(), m.lora_B.data.float()
    frozen = lora.freeze(model, trainable_prefixes=())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = int(math.ceil(len(records) * args.epochs / args.accum))
    rng = random.Random(args.seed)
    step, micro, acc, acc_n, losses = 0, 0, 0.0, 0, []
    log = open(args.out / "log.jsonl", "w")
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(records)))
        rng.shuffle(order)
        for k in order:
            inputs, labels, spans, _ = encode(processor, records[k])
            check_spans(records[k], spans)
            inputs = inputs.to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**inputs, labels=labels.to(model.device))
            (out.loss / args.accum).backward()
            acc, acc_n = acc + float(out.loss.detach()), acc_n + 1
            micro += 1
            if micro % args.accum == 0 or (epoch == args.epochs - 1 and k == order[-1]):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total, args.lr)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                rec = {"step": step + 1, "epoch": epoch + 1, "loss": round(acc / acc_n, 5),
                       "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t1, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                losses.append(rec["loss"])
                step, acc, acc_n = step + 1, 0.0, 0
                if step % 10 == 0:
                    print(f"[n2_train] step {step}/{total} loss {rec['loss']:.4f} ({rec['seconds']} s)", flush=True)
    log.close()
    train_s = time.time() - t1
    torch.save(adapter_state(model), args.out / "adapter.pt")
    meta = {"model": args.model, "model_path": q0_ask.model_path(args.model), "processor": proc_dir,
            "data": str(args.data), "conversations": len(records), "epochs": args.epochs, "accum": args.accum,
            "lr": args.lr, "warmup": WARMUP, "schedule": "linear warm-up, cosine to 0", "seed": args.seed,
            "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "lora": info, "frozen": frozen,
            "optimizer_steps": step, "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_last10_mean": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
            "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[n2_train] done: {step} steps in {train_s:.0f} s, loss {meta['loss_first']} -> {meta['loss_last']}",
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--accum", type=int, default=ACCUM)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    records = [json.loads(x) for x in args.data.read_text().splitlines() if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    return dry_run(args, records) if args.dry_run else train(args, records)


if __name__ == "__main__":
    sys.exit(main())
