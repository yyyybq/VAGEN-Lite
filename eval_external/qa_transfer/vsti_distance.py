"""VSTI-Bench metric questions: does a model order distances right even when its scale is off, and does it answer
with a few fixed numbers?

    python -m eval_external.qa_transfer.vsti_distance --stage step2 --models base-7B ours-SFT --out eval_external/results/vsti_distance

For camera-object distance and camera displacement, per model: the Spearman rank correlation between predicted and
true value (answers with a positive number only), the number of distinct predicted values, and the share of answers
taken by the five most frequent values.  Numbers are read as in scale_bias.py.  Writes OUT.md and OUT.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval_external.qa_transfer.scale_bias import items
from eval_external.qa_transfer.summary import load_results

TYPES = ("camera_obj_abs_dist", "camera_displacement")


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1  # ties share their mean rank
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return round(cov / (vx * vy) ** 0.5, 3) if vx and vy else None


def concentration(preds: list[float]) -> dict:
    c = Counter(preds)
    top = c.most_common(5)
    return {"distinct_values": len(c), "top5": [[v, n] for v, n in top],
            "top5_share": round(sum(n for _, n in top) / len(preds), 3) if preds else None}


def analyse(rows: list[dict]) -> dict:
    out = {}
    for qt in TYPES:
        pairs = [(x["prediction"], x["truth"]) for x in rows
                 if x["question_type"] == qt and x["truth"] and x["prediction"] is not None and x["prediction"] > 0]
        n_all = sum(x["question_type"] == qt for x in rows)
        preds = [p for p, _ in pairs]
        out[qt] = {"n": n_all, "n_positive_number": len(pairs), "spearman": spearman(preds, [g for _, g in pairs]),
                   **concentration(preds)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--bench", default="VSTI-Bench")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    res = load_results(args.stage)
    table = {m: analyse(items(res[(m, args.bench)])) for m in args.models if (m, args.bench) in res}
    lines = ["| 模型 | 题型 | 题数 | 有正数回答 | 排序相关（Spearman） | 不同取值个数 | 前 5 个取值占比 | 前 5 个取值（值×次数） |",
             "|---|---|---|---|---|---|---|---|"]
    for m, by in table.items():
        for qt, s in by.items():
            top = ", ".join(f"{v:g}×{n}" for v, n in s["top5"])
            lines.append(f"| {m} | {qt} | {s['n']} | {s['n_positive_number']} | {s['spearman']} | "
                         f"{s['distinct_values']} | {s['top5_share']} | {top} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(table, indent=1) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
