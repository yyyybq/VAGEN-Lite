"""Re-collect every finished QA run with collect.py's checks (predictions = items, no skipped combination, run time
vs base-7B) and list them; for runs collected before the checks existed (2026-10-19).

    python -m eval_external.qa_transfer.check_runs --stages step1 step2 [--out eval_external/results/qa_checks]

A run whose checks fail keeps its old results.json renamed to results_unchecked.json, so the queue no longer
counts it done.  Folders starting with "_" (set-aside runs) are skipped; symlinked runs are checked once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_external.qa_transfer.collect import QA_RUNS, CheckFailed, collect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["step1", "step2"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    seen, rows = set(), []
    for stage in args.stages:
        for rd in sorted((QA_RUNS / stage).iterdir()):
            if rd.name.startswith("_") or not (rd / "results.json").exists() or rd.resolve() in seen:
                continue
            seen.add(rd.resolve())
            try:
                r = collect(rd.resolve())
                c = r["checks"]
                rows.append({"run": f"{stage}/{rd.name}", "ok": True, **c})
            except CheckFailed as e:
                (rd / "results.json").rename(rd / "results_unchecked.json")
                rows.append({"run": f"{stage}/{rd.name}", "ok": False, "error": str(e)})
    lines = ["| 运行 | 预测条数 / 题数 | 后端跳过 | 用时（小时） | base-7B 用时（小时） | 标记 |", "|---|---|---|---|---|---|"]
    for r in rows:
        if not r["ok"]:
            lines.append(f"| {r['run']} | 核对失败 | | | | {r['error'][:120]} |")
            continue
        h = lambda s: "—" if s is None else f"{s / 3600:.2f}"
        flag = "用时过短" if r["short_run"] else ""
        lines.append(f"| {r['run']} | {r['n_predictions']} / {r['n_expected']} | {'是' if r['backend_skipped'] else '否'} "
                     f"| {h(r['duration_s'])} | {h(r['base7b_duration_s'])} | {flag} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".md").write_text(text)
        args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
