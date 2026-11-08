"""Metric scale of the numeric answers: for every model, the ratio prediction / ground truth on the questions that
ask for a length in metres (VSI-Bench object distance, object size, room size; VSTI-Bench camera-object distance,
camera displacement).  The benchmarks' MRA score says only how often an answer is close; this says whether a model
answers systematically too small or too large.

    python -m eval_external.qa_transfer.scale_bias --stage step2 --out eval_external/results/scale_bias_step2

Reads the per-item files listed in each run's results.json (VLMEvalKit *_matching.xlsx, lmms-eval samples jsonl).
The number of an answer is the first number in it (VLMEvalKit's pred_extracted when present); an answer with no
number, or 0, is counted apart.  Writes OUT.md and OUT.json.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from eval_external.qa_transfer.summary import MODEL_ORDER, load_results

TYPES = {  # benchmark -> question types whose answer is a length in metres (room size is m^2)
    "VSI-Bench": ("object_abs_distance", "object_size_estimation", "room_size_estimation"),
    "VSTI-Bench": ("camera_obj_abs_dist", "camera_displacement"),
    "VSTI-Bench-prefix": ("camera_obj_abs_dist", "camera_displacement"),
}
_NUM = re.compile(r"[-+]?\d*\.?\d+")


def first_number(text) -> float | None:
    m = _NUM.search(str(text)) if text is not None else None
    return float(m.group()) if m else None


def ratio_stats(pairs: list[tuple[float | None, float]]) -> dict:
    """pairs: (prediction or None, ground truth).  Median and quartiles of prediction / truth over the answers with
    a positive number; shares below half and above double the truth."""
    r = sorted(p / g for p, g in pairs if p is not None and p > 0 and g > 0)
    out = {"n": len(pairs), "n_no_number_or_zero": len(pairs) - len(r)}
    if not r:
        return out
    q = statistics.quantiles(r, n=4) if len(r) > 1 else [r[0]] * 3
    out.update({"median": round(statistics.median(r), 3), "q25": round(q[0], 3), "q75": round(q[2], 3),
                "share_below_half": round(sum(x < 0.5 for x in r) / len(r), 3),
                "share_above_double": round(sum(x > 2 for x in r) / len(r), 3)})
    return out


def items(result: dict) -> list[dict]:
    """[{question_type, prediction, truth}] of one run."""
    rows = []
    if result["backend"] == "vlmevalkit":
        import pandas as pd

        f = [p for p in result["files"] if p.endswith(".xlsx")][0]
        for _, x in pd.read_excel(f).iterrows():
            pred = x.get("pred_extracted")
            pred = first_number(x["prediction"]) if pred is None or pred != pred else first_number(pred)
            rows.append({"question_type": x["question_type"], "prediction": pred, "truth": first_number(x["answer"])})
    else:
        f = [p for p in result["files"] if "samples" in Path(p).name][0]
        for line in open(f):
            x = json.loads(line)
            info = next(v for v in x.values() if isinstance(v, dict) and "question_type" in v)
            rows.append({"question_type": info["question_type"], "prediction": first_number(info["prediction"]),
                         "truth": first_number(info["ground_truth"])})
    return rows


def build(stage: str) -> dict:
    res = load_results(stage)
    table = {}
    for (model, bench), r in res.items():
        if bench not in TYPES:
            continue
        rows = items(r)
        for qt in TYPES[bench]:
            pairs = [(x["prediction"], x["truth"]) for x in rows if x["question_type"] == qt and x["truth"]]
            table.setdefault(model, {})[f"{bench}/{qt}"] = ratio_stats(pairs)
    order = [m for m in MODEL_ORDER if m in table] + sorted(set(table) - set(MODEL_ORDER))
    return {m: table[m] for m in order}


def markdown(table: dict) -> str:
    cols = sorted({c for row in table.values() for c in row})
    head = "| 模型 | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    lines = []
    for m, row in table.items():
        cells = []
        for c in cols:
            s = row.get(c)
            cells.append("—" if not s or "median" not in s else
                         f"{s['median']} [{s['q25']}, {s['q75']}] <½: {round(100 * s['share_below_half'])}%")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return ("**预测值 / 真实值**：中位数 [四分位]，以及不到真实值一半的回答占比\n\n" + head + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    table = build(args.stage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
    md = markdown(table)
    args.out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
