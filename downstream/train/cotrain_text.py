#!/usr/bin/env python3
"""C1's text samples (the user's decision of 2026-10-30: lambda 0.1, 8 text samples per step; the main line's QA
mixed 1:1 with the trajectories to cut repetition, its records being the same chat format).

Sources (read by path, never copied; both are LLaMA-Factory sharegpt records {"messages", "images"[, "videos"]} with
one <image> per image, 256 x 256 frames):
    traj  the main line's round-11 SFT training trajectories (interiorgs_v10_train_r11_20261016/lf_formal/
          trajectory_train.jsonl, 9,802; the design's text data)
    qa    the main line's final QA training set (interiorgs_qa_final_20261028/lf_256/qa_train.jsonl, 10,879 records
          from the QA bank interiorgs_qa_bank_20261027's training scenes; the 262 video conversations are left out:
          C1 trains on images only)
Which sample a process takes is a function of the step alone, so a resumed run continues exactly and nothing is
saved for it: with W processes, even ranks read traj and odd ranks qa (four each on eight cards: 1:1); the k-th
process of a source takes position p = step * n_k + k of that source, n_k processes reading it; epoch p // N uses
its own random order (seed, source, epoch), so every record is used once per pass.
Each sample becomes Qwen2.5-VL chat text (x4_ppl.to_chat, the processor of B), and only the answers' tokens (each
answer and its closing <|im_end|>, x4_ppl.answer_spans) carry labels; everything else is -100.  A sample longer than
max_tokens is replaced by the next position's.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "x"))

MAIN = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps"
SOURCES = {"traj": f"{MAIN}/vagen_active_spatial/interiorgs_v10_train_r11_20261016/lf_formal/trajectory_train.jsonl",
           "qa": f"{MAIN}/interiorgs_qa_final_20261028/lf_256/qa_train.jsonl"}
MAX_TOKENS = 8192
IGNORE = -100


def load_records(path, skip_videos: bool = True) -> list:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if skip_videos and r.get("videos"):
            continue
        out.append({"messages": r["messages"], "images": list(r.get("images") or [])})
    return out


def source_of(rank: int, world: int):
    """(source, k, n): this rank's source, its index among the ranks reading it, and how many read it."""
    name = "traj" if rank % 2 == 0 else "qa"
    readers = [r for r in range(world) if (r % 2 == 0) == (name == "traj")]
    return name, readers.index(rank), len(readers)


def order(n: int, seed: int, name: str, epoch: int) -> list:
    idx = list(range(n))
    random.Random(f"{seed}|{name}|{epoch}").shuffle(idx)
    return idx


def pick(step: int, rank: int, world: int, sizes: dict, seed: int, offset: int = 0):
    """(source, record index) of this rank's sample at this step; offset moves to later positions (an over-long
    sample is replaced by the next one)."""
    name, k, n = source_of(rank, world)
    p = step * n + k + offset * n
    size = sizes[name]
    return name, order(size, seed, name, p // size)[p % size]


def labels_for(ids) -> list:
    """Labels for a token list: the answers' tokens keep their ids, everything else is IGNORE."""
    import x4_ppl

    ids = list(ids)
    out = [IGNORE] * len(ids)
    for a, b in x4_ppl.answer_spans(ids):
        for t in range(a, b + 1):
            out[t] = ids[t]
    return out


class TextBatches:
    """One text sample per process per step (see the module text)."""

    def __init__(self, processor, rank: int, world: int, seed: int, sources=None, max_tokens: int = MAX_TOKENS):
        self.processor, self.rank, self.world, self.seed, self.max_tokens = processor, rank, world, seed, max_tokens
        self.records = {k: load_records(v) for k, v in (sources or SOURCES).items()}
        self.sizes = {k: len(v) for k, v in self.records.items()}

    def encode(self, rec: dict):
        import torch
        from PIL import Image

        import x4_ppl

        text = self.processor.apply_chat_template(x4_ppl.to_chat(rec["messages"]), tokenize=False,
                                                  add_generation_prompt=False)
        images = [Image.open(p).convert("RGB") for p in rec["images"]]
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt")
        inputs["labels"] = torch.tensor([labels_for(inputs["input_ids"][0].tolist())])
        return inputs

    def at(self, step: int):
        """(source, index, model inputs with labels) for this process at this step."""
        for offset in range(100):
            name, i = pick(step, self.rank, self.world, self.sizes, self.seed, offset)
            inputs = self.encode(self.records[name][i])
            if inputs["input_ids"].shape[1] <= self.max_tokens and int((inputs["labels"] != IGNORE).sum()) > 0:
                return name, i, inputs
        raise RuntimeError(f"no usable text sample near step {step} for rank {self.rank}")
