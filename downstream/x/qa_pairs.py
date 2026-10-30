#!/usr/bin/env python3
"""Paired comparisons for the question-answering batch report (2026-10-30) that the automatic tables (qa_report.py)
do not give:

    rl_vs_sft   each RL model against its SFT start, and the two RL models against each other (RS / RD vs S, RD vs
                RS; Q3RS / Q3RD vs Q3S, Q3RD vs Q3RS), on X0, X0b-1, X0b-2 and G0, paired by question (McNemar)
    loosest     X0b-2 under the loosest reading: the lenient one (x0b_qa.parse_tagged_text), plus a reply that opens
                with "<A." / "<B." without the tag's name, as Q3B wrote 461 of its 510 replies ("<B. right>";
                x0_qa.parse_letter drops anything inside angle brackets, so the other readings find no letter there);
                every model against its family's base

    python downstream/x/qa_pairs.py --qa QA_DIR [--x0 X0_OUT] --out QA_DIR/results/pairs.json

QA_DIR is run_qa_batch.sh's OUT (questions in g0/, x0b1/, x0b2/, replies in answers/); X0's questions and the Qwen2.5
models' X0 replies are the 10-26 ones in X0_OUT (answers_<M>.jsonl), the Qwen3 models' in QA_DIR/answers/x0_<M>.jsonl.
Read-only apart from --out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g0_qa  # noqa: E402
import x0_qa  # noqa: E402
import x0b_qa  # noqa: E402

X0_OUT = "/mnt/umm/users/yinbaiqiao/downstream_runs/x/X0/run_20261026/out"
SETS = ("X0", "X0b-1", "X0b-2", "G0")
RL_PAIRS = (("RS", "S"), ("RD", "S"), ("RD", "RS"), ("Q3RS", "Q3S"), ("Q3RD", "Q3S"), ("Q3RD", "Q3RS"))
BASE_PAIRS = (("S", "B"), ("RS", "B"), ("RD", "B"), ("Q3S", "Q3B"), ("Q3RS", "Q3B"), ("Q3RD", "Q3B"), ("Q3B", "B"))


def read_loosest(text, options):
    """X0b-2's loosest reading: the lenient reading, else a reply opening with "<A." / "<B."."""
    letter = x0b_qa.parse_tagged_text(text, options or [])
    if letter:
        return letter
    m = re.match(r"\s*<\s*([AB])\.", str(text or ""))
    return m.group(1) if m else None


def rows(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def questions(qa: Path, x0: Path, name: str) -> dict:
    path = {"X0": x0 / "questions.jsonl", "X0b-1": qa / "x0b1/questions.jsonl", "X0b-2": qa / "x0b2/questions.jsonl",
            "G0": qa / "g0/questions.jsonl"}[name]
    return {q["id"]: q for q in rows(path)}


def reply_path(qa: Path, x0: Path, name: str, model: str) -> Path:
    if name == "X0":
        return qa / f"answers/x0_{model}.jsonl" if model.startswith("Q3") else x0 / f"answers_{model}.jsonl"
    return qa / "answers" / {"X0b-1": f"x0b1_{model}.jsonl", "X0b-2": f"x0b2_{model}.jsonl", "G0": f"g0_{model}.jsonl"}[name]


def right(qs: dict, path: Path, name: str) -> dict:
    """{question id: whether the reply is right} under the set's reading (X0b-2: the loosest)."""
    out = {}
    for r in rows(path):
        q = qs.get(r["id"])
        if q is None:
            continue
        reply = r.get("reply")
        if name == "G0":
            got = g0_qa.parse_reply(q, reply)
        elif name == "X0b-2":
            got = read_loosest(reply, q.get("options"))
        else:
            got = x0_qa.parse_letter(reply)
        out[r["id"]] = got == q["answer"]
    return out


def pair(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    only_a = sum(1 for i in ids if a[i] and not b[i])
    only_b = sum(1 for i in ids if b[i] and not a[i])
    ka, kb = sum(a[i] for i in ids), sum(b[i] for i in ids)
    return {"n": len(ids), "correct": ka, "reference_correct": kb,
            "diff_points": 100.0 * (ka - kb) / len(ids) if ids else None,
            "only_model": only_a, "only_reference": only_b, "p": x0_qa.mcnemar_exact(only_a, only_b)}


def compute(qa: Path, x0: Path) -> dict:
    cache = {}

    def get(name, model):
        if (name, model) not in cache:
            path = reply_path(qa, x0, name, model)
            cache[(name, model)] = right(questions(qa, x0, name), path, name) if path.exists() else None
        return cache[(name, model)]

    out = {"rl_vs_sft": {}, "loosest": {}}
    for name in SETS:
        for m, ref in RL_PAIRS:
            a, b = get(name, m), get(name, ref)
            if a is not None and b is not None:
                out["rl_vs_sft"].setdefault(name, {})[f"{m}-{ref}"] = pair(a, b)
    for m, ref in BASE_PAIRS:
        a, b = get("X0b-2", m), get("X0b-2", ref)
        if a is not None and b is not None:
            out["loosest"][f"{m}-{ref}"] = pair(a, b)
    for m in ("B", "S", "RS", "RD", "Q3B", "Q3S", "Q3RS", "Q3RD"):
        a = get("X0b-2", m)
        if a is not None:
            out["loosest"].setdefault("accuracy", {})[m] = {"n": len(a), "correct": sum(a.values())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--x0", default=X0_OUT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = compute(Path(args.qa), Path(args.x0))
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    for name, pairs in res["rl_vs_sft"].items():
        for k, v in pairs.items():
            print(f"{name:6s} {k:10s} {v['correct']} vs {v['reference_correct']} of {v['n']} "
                  f"({v['diff_points']:+.1f} points) p {v['p']:.3g}")
    for k, v in res["loosest"].items():
        if k != "accuracy":
            print(f"X0b-2 loosest {k:9s} {v['correct']} vs {v['reference_correct']} of {v['n']} p {v['p']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
