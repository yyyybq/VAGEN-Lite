"""Own-env runs compared task by task with a reference run (default base-7B), and success per task type.

    python -m eval_external.diagnostics.compare_ownenv --ref base-7B --out eval_external/results/ownenv_step2

A run is a folder of ROOT/runs/ownenv/eval_test, or a folder of the round-10 eval_test (written r10:<name>, read
only).  For each run: success k / n, the difference to the reference in points, the tasks only it solved and the
tasks only the reference solved, and the two-sided exact McNemar p (binomial on those two counts).  Writes OUT.md
and OUT.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

from eval_external.diagnostics.ownenv_eval import EVAL, SOURCE_EVAL


def run_path(name: str) -> Path:
    return SOURCE_EVAL / name[4:] if name.startswith("r10:") else EVAL / name


KEEP: set[str] | None = None  # --multi-task-scenes: only tasks whose scene has another test task


def load(name: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in open(run_path(name) / "results.jsonl") if line.strip()]
    return {r["task_id"]: r for r in rows if "error" not in r and (KEEP is None or r["task_id"] in KEEP)}


def multi_task_scene_ids() -> set[str]:
    from collections import Counter

    tasks = [json.loads(line) for line in open(EVAL / "tasks.jsonl")]
    rows = [json.loads(line) for line in open(EVAL / "ours-SFT" / "results.jsonl")]  # index -> task_id
    n = Counter(t["scene_id"] for t in tasks)
    return {r["task_id"] for r in rows if n[tasks[r["index"]]["scene_id"]] > 1}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(ref: dict, run: dict) -> dict:
    common = sorted(set(ref) & set(run))
    only_run = sum(run[t]["success"] and not ref[t]["success"] for t in common)
    only_ref = sum(ref[t]["success"] and not run[t]["success"] for t in common)
    k_run = sum(bool(run[t]["success"]) for t in common)
    k_ref = sum(bool(ref[t]["success"]) for t in common)
    return {"n": len(common), "success": k_run, "ref_success": k_ref,
            "diff_pts": round(100 * (k_run - k_ref) / len(common), 1) if common else None,
            "only_run": only_run, "only_ref": only_ref, "p_mcnemar": round(mcnemar_exact(only_run, only_ref), 4)}


def by_type(run: dict) -> dict[str, list[int]]:
    out = defaultdict(lambda: [0, 0])
    for r in run.values():
        out[r["task_type"]][0] += bool(r["success"])
        out[r["task_type"]][1] += 1
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="base-7B")
    ap.add_argument("--runs", nargs="*", help="default: every finished run in the eval copy")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--multi-task-scenes", action="store_true",
                    help="only tasks whose scene has another test task (replace-history donors from the same scene)")
    args = ap.parse_args()
    global KEEP
    if args.multi_task_scenes:
        KEEP = multi_task_scene_ids()
    names = args.runs or sorted(p.parent.name for p in EVAL.glob("*/results.json"))
    ref = load(args.ref)
    table = {n: {**paired(ref, load(n)), "by_type": by_type(load(n))} for n in names}
    types = sorted({t for v in table.values() for t in v["by_type"]})
    md = [f"**与 {args.ref} 逐题比**（只有它做对 / 只有 {args.ref} 做对；p 为精确 McNemar 双侧检验）\n",
          "| 运行 | 成功 / 题数 | 差（点） | 只有它对 | 只有参照对 | p |", "|---|---|---|---|---|---|"]
    for n, v in table.items():
        p = v["p_mcnemar"]
        md.append(f"| {n} | {v['success']} / {v['n']} | {v['diff_pts']:+.1f} | {v['only_run']} | {v['only_ref']} | "
                  f"{'<0.0001' if p < 1e-4 else p} |")
    md += ["", "**按任务类型的成功数 / 题数**\n", "| 运行 | " + " | ".join(types) + " |",
           "|---|" + "---|" * len(types)]
    for n, v in table.items():
        md.append(f"| {n} | " + " | ".join("{}/{}".format(*v["by_type"].get(t, [0, 0])) for t in types) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({"ref": args.ref, "runs": table}, indent=1) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
