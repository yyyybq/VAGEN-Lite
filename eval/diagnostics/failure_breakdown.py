"""Diagnostic 1: what each evaluation episode did, per task family.

    python -m eval.diagnostics.failure_breakdown --eval EVAL_DIR --names sft_formal_e9,rs0_300 --out OUT_PREFIX

Reads the result folders of data_gen/interiorgs_v10/evaluation.py (EVAL_DIR/NAME/results.jsonl, written with
per-turn trajectories since round 8: after every turn whether the task was met, ``met``, and the success
conditions not met, ``failed_gates``).  Each episode gets one class:

  reached          some turn ended with the task met (the evaluation's ``success``);
  drifted          reached, then the last turn no longer met it - cannot happen under the current rules (success
                   is checked after every action and ends the episode at once), counted to show it;
  stopped          not reached, the episode ended early (nearly uniform view or invalid answers three times in a
                   row, blocked moves, a "done" without success);
  budget_one_gate  not reached, the 12 actions / turns ran out, and at its best turn only one condition failed;
  budget_other     not reached, the actions / turns ran out, never closer than two failed conditions.

Writes OUT_PREFIX.json (counts per model, family and class; the single failed condition of budget_one_gate) and
OUT_PREFIX.md (shares per family, one table per model).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

CLASSES = ("reached", "drifted", "stopped", "budget_one_gate", "budget_other")
CLASS_ZH = {"reached": "到过（成功）", "drifted": "到过又走偏", "stopped": "没到就停了",
            "budget_one_gate": "用完步数·最好时只差一个门", "budget_other": "用完步数·其他"}
BUDGET_ENDS = {"max_primitive_steps", "max_turns"}
FAMILY_ORDER = ("projective_relations", "absolute_positioning", "screen_occupancy", "equidistance",
                "occlusion_alignment", "fov_inclusion")


def classify(row: Dict[str, Any]) -> Dict[str, Any]:
    """The class of one episode and, for budget_one_gate, the condition that failed at its best turn."""
    traj = row.get("trajectory")
    if traj is None:
        raise ValueError(f"episode {row.get('index')}: no per-turn trajectory (results written before round 8)")
    met = [bool(t.get("met")) for t in traj]
    if row.get("success") or any(met):
        return {"class": "drifted" if (met and not met[-1]) else "reached"}
    if row.get("end_reason") not in BUDGET_ENDS:
        return {"class": "stopped", "end_reason": row.get("end_reason")}
    counts = [len(t["failed_gates"]) for t in traj if t.get("failed_gates") is not None]
    if counts and min(counts) == 1:
        best = next(t for t in traj if t.get("failed_gates") is not None and len(t["failed_gates"]) == 1)
        start = (row.get("start") or {}).get("failed_gates")
        return {"class": "budget_one_gate", "gate": best["failed_gates"][0],
                "start_one_gate": bool(start is not None and len(start) == 1)}
    return {"class": "budget_other"}


def breakdown(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    gates: collections.Counter = collections.Counter()
    start_close = 0
    errors = 0
    for r in rows:
        if "error" in r:
            errors += 1
            continue
        c = classify(r)
        fam = str(r.get("task_type"))
        by_family[fam][c["class"]] += 1
        by_family["all"][c["class"]] += 1
        if c["class"] == "budget_one_gate":
            gates[c["gate"]] += 1
            start_close += int(c["start_one_gate"])
    return {"by_family": {k: dict(v) for k, v in by_family.items()}, "one_gate": dict(gates.most_common()),
            "one_gate_start_already": start_close, "errors": errors}


def table(name: str, res: Dict[str, Any]) -> List[str]:
    fams = [f for f in FAMILY_ORDER if f in res["by_family"]] + \
        sorted(f for f in res["by_family"] if f not in FAMILY_ORDER and f != "all") + ["all"]
    lines = [f"**{name}**", "", "| 任务族 | 题数 | " + " | ".join(CLASS_ZH[c] for c in CLASSES) + " |",
             "| --- | --- | " + " | ".join("---" for _ in CLASSES) + " |"]
    for f in fams:
        c = res["by_family"][f]
        n = sum(c.values())
        cells = [f"{100.0 * c.get(k, 0) / n:.1f}%" if n else "–" for k in CLASSES]
        lines.append(f"| {'合计' if f == 'all' else f} | {n} | " + " | ".join(cells) + " |")
    if res["one_gate"]:
        top = "，".join(f"{g} {n}" for g, n in list(res["one_gate"].items())[:6])
        lines += ["", f"只差一个门时差的是：{top}；其中 {res['one_gate_start_already']} 道起点就只差一个门。"]
    return lines + [""]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="an evaluation folder (tasks.jsonl and one folder per model)")
    ap.add_argument("--names", default="", help="comma-separated result folders (default: every folder with results)")
    ap.add_argument("--out", required=True, help="output prefix (.json and .md are added)")
    args = ap.parse_args(argv)
    ev = Path(args.eval)
    names = [n for n in args.names.split(",") if n] or sorted(p.parent.name for p in ev.glob("*/results.jsonl"))
    out: Dict[str, Any] = {}
    md = [f"# 失败拆解：{ev.name}", "",
          "每道题一个结论。\"门\"是任务成功判据里的一项条件；\"到过又走偏\"在现行规则下不会出现"
          "（每个动作后都检查成功，满足就立刻结束），这里照样统计。", ""]
    for name in names:
        res = breakdown(read_jsonl(ev / name / "results.jsonl"))
        out[name] = res
        md += table(name, res)
    Path(args.out + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    Path(args.out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({n: out[n]["by_family"]["all"] for n in out}, indent=1))


if __name__ == "__main__":
    main()
