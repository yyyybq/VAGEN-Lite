"""Read a finished QA run's raw outputs into RUN_DIR/results.json (one format for every backend).

    python -m eval_external.qa_transfer.collect RUN_DIR [RUN_DIR ...]

results.json holds:
  overall          the benchmark's own headline score in percent, as the backend computed it
  subscores        per question type, in percent (backend's numbers where it reports them, else recomputed
                   from the per-item scores)
  n_items          items with an answer
  n_correct, ci95  for accuracy-type benchmarks: items scored right and the Wilson 95% interval of overall
                   (for mixed accuracy/MRA benchmarks the interval treats the mean item score as a proportion)
  n_unparseable    answers that name no usable option / number by parse_check's rule (same rule for all models)
  manifest         the run's manifest.json (model, hashes, prompt, frames, decoding)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from eval_external.common.stats import wilson
from eval_external.qa_transfer.benchmarks import BENCHMARKS
from eval_external.qa_transfer.parse_check import mcq_letter, number


def _latest(pattern: str) -> str:
    files = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


def _item_score(rec: dict) -> float | None:
    for key in ("accuracy", "hit", "MRA:.5:.95:.05", "exact_match"):
        if key in rec and rec[key] is not None and not (isinstance(rec[key], float) and math.isnan(rec[key])):
            return float(rec[key])
    return None


def _usable(pred: str, qtype_is_mcq: bool, n_options: int, options=None) -> bool:
    return (mcq_letter(pred, n_options, options) is not None) if qtype_is_mcq else (number(pred) is not None)


def collect_lmms(run_dir: Path, bench: dict) -> dict:
    task = bench["task"]
    res_file = _latest(str(run_dir / "lmms" / "*" / "*_results.json"))
    sam_file = _latest(str(run_dir / "lmms" / "*" / f"*_samples_{task}.jsonl"))
    res = json.load(open(res_file))["results"][task]
    rows = [json.loads(line) for line in open(sam_file)]
    scores, unparse, by_type = [], 0, defaultdict(list)
    overall = None
    if task.startswith("sat"):
        overall = 100 * float(res["exact_match,flexible-extract"])
        for r in rows:
            s = float(r["exact_match"])
            scores.append(s)
            unparse += mcq_letter(str(r["filtered_resps"]), 2) is None
    else:  # vsibench-style: each sample carries the scored doc under its metric name
        metric = next(k.split(",")[0] for k in res if not k.startswith("alias") and "stderr" not in k)
        val = res[f"{metric},none"]
        overall = float(val) if float(val) > 1.5 else 100 * float(val)
        for r in rows:
            doc = r.get(metric)
            if isinstance(doc, str):
                doc = json.loads(doc) if doc.startswith("{") else None
            if not isinstance(doc, dict):
                continue
            s = _item_score(doc)
            if s is None:
                continue
            scores.append(s)
            by_type[doc["question_type"]].append(s)
            opts = doc.get("options")
            is_mcq = bool(opts) and str(opts) not in ("None", "[]", "nan")
            n_opt = len(opts) if isinstance(opts, list) else (str(opts).count("'") // 2 if is_mcq else 0)
            unparse += not _usable(str(doc.get("prediction", "")), is_mcq, max(n_opt, 2))
    return {"overall": overall, "scores": scores, "unparseable": unparse,
            "subscores": {k: 100 * sum(v) / len(v) for k, v in sorted(by_type.items())},
            "files": [res_file, sam_file]}


def collect_vlmeval(run_dir: Path, bench: dict) -> dict:
    data = bench["data"]
    acc_file = _latest(str(run_dir / "vlmeval" / "**" / f"*{data}*_acc*.*sv"))
    sep = "\t" if acc_file.endswith(".tsv") else ","
    import pandas as pd

    acc = pd.read_csv(acc_file, sep=sep)
    item_file = _latest(str(run_dir / "vlmeval" / "**" / f"*{data}*_matching*.xlsx"))
    items = pd.read_excel(item_file)
    scores, unparse, by_type = [], 0, defaultdict(list)
    for _, r in items.iterrows():
        rec = r.to_dict()
        s = _item_score(rec)
        if s is None:
            continue
        scores.append(s)
        qtype = str(rec.get("question_type", rec.get("category", "all")))
        by_type[qtype].append(s)
        cands = rec.get("candidates")
        letters = [c for c in "ABCDEFGH" if c in rec and isinstance(rec[c], str)]
        is_mcq = rec.get("task_type", "MCQ") == "MCQ"
        n_opt = len(letters) or (len(eval(cands)) if isinstance(cands, str) and cands.startswith("[") else 4)
        unparse += not _usable(str(rec.get("prediction", "")), is_mcq, n_opt)
    row = acc.iloc[0].to_dict() if len(acc) == 1 else {}
    overall = row.get("overall", row.get("Overall"))
    if overall is None and "Overall" in acc.columns:
        overall = acc["Overall"].iloc[0]
    overall = float(overall) if overall is not None else 100 * sum(scores) / max(len(scores), 1)
    if overall <= 1.0 and scores and sum(scores) / len(scores) > 0.01:
        overall *= 100
    return {"overall": overall, "scores": scores, "unparseable": unparse,
            "subscores": {k: 100 * sum(v) / len(v) for k, v in sorted(by_type.items())},
            "backend_table": {str(k): (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()},
            "files": [acc_file, item_file]}


def collect(run_dir: Path) -> dict:
    manifest = json.load(open(run_dir / "manifest.json"))
    name = manifest["benchmark"]["name"]
    bench = BENCHMARKS[name]
    got = collect_lmms(run_dir, bench) if bench["backend"] == "lmms_eval" else collect_vlmeval(run_dir, bench)
    n = len(got.pop("scores"))
    out = {"bench": name, "model": manifest["models"][0]["code"], "backend": bench["backend"],
           "overall": round(got["overall"], 2), "subscores": {k: round(v, 2) for k, v in got["subscores"].items()},
           "n_items": n, "n_expected": bench["items"], "n_unparseable": got["unparseable"],
           "unparseable_pct": round(100 * got["unparseable"] / max(n, 1), 2), "files": got["files"],
           "manifest": manifest}
    if "backend_table" in got:
        out["backend_table"] = got["backend_table"]
    k = got["overall"] / 100 * n
    p, lo, hi = wilson(k, n)
    out["ci95"] = [round(100 * lo, 1), round(100 * hi, 1)]
    out["n_correct_equiv"] = round(k, 1)
    with open(run_dir / "results.json", "w", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
        f.write("\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    for run in args.runs:
        r = collect(run.resolve())
        print(f"{r['model']:14s} {r['bench']:18s} {r['overall']:6.2f}  95% [{r['ci95'][0]}, {r['ci95'][1]}]  "
              f"n={r['n_items']}/{r['n_expected']}  unparseable={r['n_unparseable']} ({r['unparseable_pct']}%)")


if __name__ == "__main__":
    main()
