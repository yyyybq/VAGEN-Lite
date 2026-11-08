"""WalkerBench tables (decided 2026-10-19): the main table uses the local rule (--lenient) for every model, the
strict parse goes to the appendix; both list the share of unparseable replies the rule turned into an action.

    python -m eval_external.closed_loop.walkerbench.summary --stage step2 --out eval_external/results/walkerbench_step2

A model whose strict run had no unparseable step never triggers the rule, so its strict run is its main-table
run.  Writes OUT.md and OUT.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_external.closed_loop.walkerbench.run import TYPES
from eval_external.qa_transfer.summary import MODEL_ORDER

RUNS = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets/runs/walkerbench")
REPORTED = {"base-7B": {"total": 7.0, "steps": 17.71}, "SI-7B": {"total": 14.0, "steps": 6.52}}  # SI paper


def load(stage: str, name: str) -> dict | None:
    f = RUNS / stage / name / "results.json"
    return json.loads(f.read_text()) if f.exists() else None


def row(res: dict, run: str) -> dict:
    c = res.get("counts") or {}
    tried = c.get("lenient_repairs", 0) + c.get("lenient_no_action", 0)
    return {"run": run, "total_sr": res["total_sr"], "ci95": res["overall_ci95"],
            "per_type": {t: res["per_type"][t]["success"] for t in TYPES}, "mean_steps": res["mean_steps"],
            "unparseable_pct": res["unparseable_pct"], "context_overflow_steps": res.get("context_overflow_steps", 0),
            "rule_rescued": (round(100 * c.get("lenient_repairs", 0) / tried, 1) if tried else None),
            "rule_tried": tried}


def build(stage: str) -> dict:
    models = sorted({p.name.split("__")[0] for p in (RUNS / stage).iterdir() if not p.name.startswith("_")},
                    key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))
    main, strict = {}, {}
    for m in models:
        s, le = load(stage, m), load(stage, f"{m}__lenient")
        if s:
            strict[m] = row(s, m)
        if le:
            main[m] = row(le, f"{m}__lenient")
        elif s and s["unparseable_steps"] == 0:
            main[m] = {**row(s, m), "note": "no unparseable step: the rule never fires, strict run = rule run"}
    return {"main": main, "strict": strict}


def combined(stage: str, main_run: str, extra_run: str) -> dict | None:
    """Standard-100 (main-table run) + extra300 for one model: per-type successes, total over 400 with its Wilson
    interval, and per-task success for paired tests."""
    from eval_external.common.stats import wilson

    a, b = load(stage, main_run), load(stage, extra_run)
    if not (a and b):
        return None
    per = {t: a["per_type"][t]["success"] + b["per_type"][t]["success"] for t in TYPES}
    n = a["n_tasks"] + b["n_tasks"]
    k = sum(per.values())
    p, lo, hi = wilson(k, n)
    succ = {}  # per_task is keyed by the task file's name, unique across both sets
    for res in (a, b):
        for tid, r in res["per_task"].items():
            succ[tid] = bool(r.get("success"))
    return {"n": n, "success": k, "rate": round(100 * p, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
            "per_type": per, "per_task": succ,
            "mean_steps": round((a["mean_steps"] * a["n_scored"] + b["mean_steps"] * b["n_scored"])
                                / (a["n_scored"] + b["n_scored"]), 2)}


def extra_tables(stage: str, main: dict) -> list[str]:
    from eval_external.diagnostics.compare_ownenv import mcnemar_exact

    rows = {}
    for m, r in main.items():
        c = combined(stage, r["run"], f"{m}__extra300__lenient")
        x = load(stage, f"{m}__extra300__lenient")
        if c and x:
            rows[m] = (row(x, f"{m}__extra300__lenient"), c)
    if not rows:
        return []
    out = ["", "**加测 300 题（本地规则；每类 60 题）和合计 400 题**", "",
           "| 模型 | 加测 300 题成功率（95% 区间） | 加测各类成功题数 | 合计 400 题成功率（95% 区间） | 合计各类成功题数（每类 80） | 平均步数（400 题） | 规则救回的无效回答（加测） |",
           "|---|---|---|---|---|---|---|"]
    for m, (x, c) in rows.items():
        out.append(f"| {m} | {x['total_sr']:.1f} [{x['ci95'][0]}, {x['ci95'][1]}] | "
                   + " / ".join(str(x["per_type"][t]) for t in TYPES)
                   + f" | {c['rate']} [{c['ci95'][0]}, {c['ci95'][1]}]（{c['success']}/{c['n']}） | "
                   + " / ".join(str(c["per_type"][t]) for t in TYPES) + f" | {c['mean_steps']} | "
                   + (f"{x['rule_rescued']}%（{x['rule_tried']} 个里）" if x["rule_tried"] else "无无效回答") + " |")
    names = list(rows)
    if len(names) == 2:
        (m1, (_, c1)), (m2, (_, c2)) = rows.items()
        common = set(c1["per_task"]) & set(c2["per_task"])
        only1 = sum(c1["per_task"][t] and not c2["per_task"][t] for t in common)
        only2 = sum(c2["per_task"][t] and not c1["per_task"][t] for t in common)
        out += ["", f"逐题配对（{len(common)} 题）：只有 {m1} 做对 {only1} 题，只有 {m2} 做对 {only2} 题，"
                    f"精确 McNemar p = {mcnemar_exact(only1, only2):.2g}。"]
    return out


def table(rows: dict, rule: bool) -> list[str]:
    head = "| 模型 | 成功率（95% 区间） | 角度 / 距离 / 高度 / 导航 / 可见性（成功题数） | 平均步数 | 解析不出的步 | 上下文溢出的步 |"
    head += " 规则救回的无效回答 |" if rule else ""
    out = [head, "|---" * (7 if rule else 6) + "|"]
    for m, r in rows.items():
        rep = REPORTED.get(m)
        total = f"{r['total_sr']:.1f} [{r['ci95'][0]}, {r['ci95'][1]}]" + (f"（SI 报 {rep['total']}）" if rep else "")
        cells = [m, total, " / ".join(str(r["per_type"][t]) for t in TYPES), f"{r['mean_steps']:.1f}",
                 f"{r['unparseable_pct']:.1f}%", str(r["context_overflow_steps"])]
        if rule:
            cells.append("无无效回答" if r.get("note") else
                         (f"{r['rule_rescued']}%（{r['rule_tried']} 个里）" if r["rule_tried"] else "—"))
        out.append("| " + " | ".join(cells) + " |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    t = build(args.stage)
    md = (["**主表：本地规则（回答不是 JSON 时，取其中最后提到的 MOVE / ROTATION / STOP）**", ""] + table(t["main"], True)
          + extra_tables(args.stage, t["main"])
          + ["", "**附录：严格解析（WalkerBench 自己的 JSON 解析，不修补）**", ""] + table(t["strict"], False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(t, indent=1, ensure_ascii=False) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
