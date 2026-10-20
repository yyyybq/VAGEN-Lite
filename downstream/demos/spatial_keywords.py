#!/usr/bin/env python3
"""Count spatial-relation words in every instruction of the 34 seen target tasks (input to the spatial-task list).

    python downstream/demos/spatial_keywords.py --out downstream/demos/spatial_keyword_counts.json

Reads meta/tasks.jsonl of each target dataset (all instruction variants, including the subtask annotations of
composite tasks).  A word count is evidence, not the decision: spatial_tasks.json holds the decision and its rule.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from subsets import source_of, task_table  # noqa: E402

PATTERNS = {
    "left/right": r"\b(left|right)\b",
    "front/rear/back": r"\b(front|rear|back|behind|in front of)\b",
    "top/bottom/above/below/under": r"\b(top|bottom|above|below|under|underneath|upper|lower)\b",
    "next to/beside/near/far": r"\b(next to|beside|near|nearby|nearest|closest|far|farthest|adjacent)\b",
    "center/middle": r"\b(center|centre|middle)\b",
    "size order": r"\b(smaller|larger|bigger|smallest|largest)\b",
}


def counts_for(instructions):
    out = {}
    for name, pattern in PATTERNS.items():
        hits = [s for s in instructions if re.search(pattern, s, flags=re.I)]
        if hits:
            out[name] = {"instructions": len(hits), "example": hits[0][:160]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "spatial_keyword_counts.json")
    args = ap.parse_args(argv)
    report = {}
    for task in task_table():
        if task["group"] == "composite_unseen":
            continue
        source = source_of(task)
        instructions = [json.loads(x)["task"] for x in (source / "meta/tasks.jsonl").read_text().splitlines() if x.strip()]
        report[task["task"]] = {"group": task["group"], "instructions": len(instructions),
                                "keywords": counts_for(instructions)}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for task, r in report.items():
        print(f"{r['group'][:9]:9s} {task:26s} {r['instructions']:4d} {sorted(r['keywords'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
