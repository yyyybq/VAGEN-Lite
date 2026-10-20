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
          + ["", "**附录：严格解析（WalkerBench 自己的 JSON 解析，不修补）**", ""] + table(t["strict"], False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(t, indent=1, ensure_ascii=False) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
