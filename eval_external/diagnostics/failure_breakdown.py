"""Diagnostic 3c-2: what happened in each episode of our env, from the per-turn trajectories (no new runs).

    python -m eval_external.diagnostics.failure_breakdown --out OUT.json NAME=RESULTS_DIR [NAME=RESULTS_DIR ...]

Every turn of evaluation.py's results rows carries ``met`` (the pose satisfies the task after the turn, by the
env's own scoring function), ``progress`` (0..1 toward success) and ``failed_gates`` (the success conditions not
met).  Each episode falls in exactly one class:

  reached            met at some turn and still met at the end (under the default rules the env ends the
                     episode as soon as the task is met: end reason success_auto)
  reached_then_left  met at some turn but not at the end (impossible under the default rules, counted anyway)
  ended_early        never met; the env ended the episode before the action budget (low_info: three low-
                     information frames in a row; this env has no stop action)
  out_of_budget      never met; the action budget ran out (max_primitive_steps)

For the episodes that never met the task it adds the best progress reached and the conditions still failed at
the end.  Rows with an infrastructure error are listed apart.  Shares come with Wilson 95% intervals.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval_external.common.stats import wilson

CLASSES = ("reached", "reached_then_left", "ended_early", "out_of_budget")


def classify(row: dict) -> str:
    traj = row.get("trajectory") or []
    mets = [bool(t.get("met")) for t in traj]
    if any(mets):
        return "reached" if mets[-1] else "reached_then_left"
    return "out_of_budget" if row.get("end_reason") in ("max_primitive_steps", "max_turns") else "ended_early"


def breakdown(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    counts = Counter(classify(r) for r in ok)
    n = len(ok)
    out = {"n": n, "errors": len(rows) - n, "classes": {}}
    for c in CLASSES:
        p, lo, hi = wilson(counts.get(c, 0), n)
        out["classes"][c] = {"n": counts.get(c, 0), "pct": round(100 * p, 1),
                             "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}
    failed = [r for r in ok if classify(r) in ("ended_early", "out_of_budget")]
    peaks = [max((t.get("progress") or 0.0) for t in r["trajectory"]) for r in failed if r.get("trajectory")]
    finals = [(r["trajectory"][-1].get("progress") or 0.0) for r in failed if r.get("trajectory")]
    gates = Counter(g for r in failed if r.get("trajectory") for g in (r["trajectory"][-1].get("failed_gates") or []))
    out["never_met"] = {
        "n": len(failed),
        "end_reasons": dict(Counter(r.get("end_reason") for r in failed)),
        "peak_progress_mean": round(sum(peaks) / len(peaks), 3) if peaks else None,
        "final_progress_mean": round(sum(finals) / len(finals), 3) if finals else None,
        "peak_progress_at_least_0.9": sum(p >= 0.9 for p in peaks),
        "failed_gates_at_end": dict(gates.most_common()),
    }
    return out


def to_markdown(table: dict) -> str:
    head = "| 模型 | 回合数 | 到达 | 到达后走偏 | 提前结束 | 用完步数 | 未到达回合的最高进度均值 |\n|---|---|---|---|---|---|---|\n"
    lines = []
    for name, b in table.items():
        c = b["classes"]
        cell = lambda k: f"{c[k]['pct']}% ({c[k]['n']}) [{c[k]['ci95'][0]}, {c[k]['ci95'][1]}]"  # noqa: E731
        lines.append(f"| {name} | {b['n']} | {cell('reached')} | {cell('reached_then_left')} | {cell('ended_early')} | "
                     f"{cell('out_of_budget')} | {b['never_met']['peak_progress_mean']} |")
    return head + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("runs", nargs="+", help="NAME=RESULTS_DIR (a folder with results.jsonl)")
    args = ap.parse_args()
    table = {}
    for item in args.runs:
        name, folder = item.split("=", 1)
        rows = [json.loads(line) for line in open(Path(folder) / "results.jsonl") if line.strip()]
        table[name] = breakdown(rows)
        table[name]["source"] = folder
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
    args.out.with_suffix(".md").write_text(to_markdown(table))
    print(to_markdown(table))


if __name__ == "__main__":
    main()
