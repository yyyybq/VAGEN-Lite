#!/usr/bin/env python3
"""Prompt length of N episodes with and without dialogue history (the user's task book of 2026-11-06, the design of
tier g): each turn's model input is rebuilt from a recorded episode (n_episode.user_text with the turn's pose and the
feedback the turn before left; with history also the earlier turns' user messages and replies, n_episode.chat_history)
and its tokens counted with the model's own processor (q0_ask.processor_dir) and blank 256 x 256 frames (a frame's
token count depends only on its size).  No model, no simulator, CPU only.

    python downstream/n/n_tokens.py --eval DIR [DIR ...] --model M --instances I.jsonl [--val-env V] [--limit 60]
                                    --out OUT.json

The episodes are the recorded ones without history; with history the model may act otherwise, so the lengths are
those of the same conversations, not of the future ones.  OUT.json: per turn number the mean and the largest prompt
with and without history, the largest over all turns, and how many calls were counted.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "x"))
import n_episode as E  # noqa: E402


def turn_texts(inst: dict, row: dict, cfg) -> list:
    """[(user message, reply)] of every recorded turn, as the episode built them."""
    item, out, fb = inst["item"], [], ""
    for t, turn in enumerate(row.get("turn_log") or []):
        user = E.user_text(cfg, t == 0, tuple(turn["pose_before"]), item["task_description"], fb)
        out.append((user, turn.get("reply") or ""))
        fb = turn.get("feedback", "Action executed.") if turn.get("actions") else "Action had no effect."
    return out


def prompts(system: str, texts: list):
    """Per turn: (messages without history, messages with history, the number of frames with history)."""
    for t, (user, _) in enumerate(texts):
        yield E.chat(system, user), E.chat_history(system, texts[:t], user), t + 1


def count(processor, messages, n_images: int) -> int:
    from PIL import Image

    frame = Image.new("RGB", (256, 256), (128, 128, 128))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return int(processor(text=[text], images=[frame] * n_images, return_tensors="pt")["input_ids"].shape[1])


def summarize(by_turn: dict) -> dict:
    out = {"per_turn": {}, "calls": sum(len(v["without"]) for v in by_turn.values())}
    for t in sorted(by_turn):
        v = by_turn[t]
        out["per_turn"][str(t)] = {"calls": len(v["without"]),
                                    "without_mean": statistics.mean(v["without"]), "without_max": max(v["without"]),
                                    "with_mean": statistics.mean(v["with"]), "with_max": max(v["with"])}
    alls = [x for v in by_turn.values() for x in v["with"]]
    out["with_max"], out["without_max"] = max(alls), max(x for v in by_turn.values() for x in v["without"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, nargs="+", required=True, help="evaluation folders with <MODEL>/episodes.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--instances", type=Path, nargs="+", required=True)
    ap.add_argument("--val-env", type=Path, default=E.VAL_ENV)
    ap.add_argument("--limit", type=int, default=60, help="episodes per folder (the longest first)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import q0_ask
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(q0_ask.processor_dir(args.model))
    cfg, _ = E.upstream_config(args.val_env)
    insts = {r["instance_id"]: r for p in args.instances for r in E.load_instances([str(p)], 0)}
    by_turn = {}
    for d in args.eval:
        rows = [json.loads(x) for x in (d / args.model / "episodes.jsonl").read_text().splitlines() if x.strip()]
        rows = sorted((r for r in rows if not r.get("error") and r["instance_id"] in insts),
                      key=lambda r: -len(r.get("turn_log") or []))[:args.limit]
        for row in rows:
            inst = insts[row["instance_id"]]
            system = E.system_text(cfg, inst["item"]["task_type"])
            for t, (plain, hist, n_img) in enumerate(prompts(system, turn_texts(inst, row, cfg)), 1):
                cell = by_turn.setdefault(t, {"without": [], "with": []})
                cell["without"].append(count(processor, plain, 1))
                cell["with"].append(count(processor, hist, n_img))
    rep = dict(summarize(by_turn), model=args.model, val_env=str(args.val_env), eval=[str(d) for d in args.eval],
               limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    last = rep["per_turn"][max(rep["per_turn"], key=int)]
    print(json.dumps({"model": args.model, "calls": rep["calls"], "with_max": rep["with_max"],
                      "without_max": rep["without_max"], "last_turn": last}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
