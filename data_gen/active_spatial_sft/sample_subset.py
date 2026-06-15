#!/usr/bin/env python3
"""
Sample N items from a large pipeline JSONL with reproducible shuffling.

Used to prepare a 5k random subset of output_100scenes/train.jsonl for the
first SFT generation pass.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify_by", default="task_type",
                    help="Optional field to stratify on (e.g. task_type). "
                         "If set, sample N proportionally across values.")
    args = ap.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    print(f"[sample] loaded {len(lines)} items from {args.input}")
    rng = random.Random(args.seed)

    if args.stratify_by:
        buckets = {}
        for ln in lines:
            try:
                key = json.loads(ln).get(args.stratify_by, "__none__")
            except Exception:
                key = "__none__"
            buckets.setdefault(key, []).append(ln)
        print(f"[sample] stratify_by={args.stratify_by} -> "
              f"{ {k: len(v) for k, v in buckets.items()} }")
        total = sum(len(v) for v in buckets.values())
        out = []
        for k, vs in buckets.items():
            quota = max(1, round(args.n * len(vs) / total))
            rng.shuffle(vs)
            out.extend(vs[:quota])
        rng.shuffle(out)
        out = out[: args.n]
    else:
        rng.shuffle(lines)
        out = lines[: args.n]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.writelines(out)
    print(f"[sample] wrote {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
