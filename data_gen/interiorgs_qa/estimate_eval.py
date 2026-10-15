"""How many evaluation questions each group can have in 100 val and 100 test scenes, and how precise that is.

    python -m data_gen.interiorgs_qa.estimate_eval --numbers exps/interiorgs_qa_report_20261014/report_numbers.json \
        [--bank-stats BANK/bank_stats.json --bank-scenes N] --plan configs/bank_plan_20261015.json --out OUT.json

Questions per scene of a group:
* rare groups (M3, M6, M8-M11): verified combos per scanned scene (report: 93-150 train scenes, every combo
  rendered) x questions per verified combo (M8: up to 3 question kinds per set, M9: up to 4, others 1);
* other groups: from a bank run when given (questions per scene, already limited by the bank quotas), else
  census combos per scene x pass rate, capped by the bank quota (3 per subtask and scene);
* video groups: census videos per scene x share of videos with a question of the group.
Counts are upper estimates before the answer and object caps of sample.py (the bank trial measures that loss).
Precision: half-width of the 95% interval of an accuracy near 50% (the widest case), 1.96 * sqrt(0.25 / n).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

RARE = ("M3", "M6", "M8", "M9", "M10", "M11")
PER_SET = {"M8": 3, "M9": 4}
BENCH = {"S1": "EmbSpatial / ViewSpatial / BLINK", "S2": "EmbSpatial / 3DSRBench / BLINK", "S3": "EmbSpatial / 3DSRBench / BLINK",
         "S4": "3DSRBench / BLINK", "S5": "BLINK", "S6": "BLINK", "S7": "3DSRBench", "M1": "MMSI / BLINK", "M2": "MMSI",
         "M3": "MMSI", "M4": "BLINK", "M5": "MMSI", "M6": "MMSI", "M7": "MindCube", "M8": "MindCube", "M9": "MindCube",
         "M10": "MindCube", "M11": "ViewSpatial", "V1": "VSI", "V2": "VSI", "V3": "VSI", "V4": "VSI", "V5": "VSI", "V6": "VSI"}


def half_width(n: float) -> float:
    return 1.96 * math.sqrt(0.25 / n) if n > 0 else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bank-stats", default="")
    ap.add_argument("--scenes", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nums = json.loads(Path(a.numbers).read_text())
    plan = json.loads(Path(a.plan).read_text())
    bank = json.loads(Path(a.bank_stats).read_text()) if a.bank_stats else None
    rows: Dict[str, Any] = {}
    for g, v in nums["groups"].items():
        quota = sum(min(n, 999) for n in plan["groups"].get(g, {}).values())
        src = ""
        if g in RARE and g in nums.get("scan", {}):
            per = nums["scan"][g]["verified_per_scene"] * PER_SET.get(g, 1)
            src = f"扫描 {nums['scan'][g]['scenes']} 个场景"
        elif bank and g in bank["groups"] and not g.startswith("V"):
            per = bank["groups"][g].get("questions", 0) / max(bank["scenes_done"], 1)
            src = f"题库试跑 {bank['scenes_done']} 个场景"
        else:
            per = min(v["valid_mean_per_scene_est"] or 0.0, quota if quota < 999 else 1e9)
            src = "普查 × 通过率"
        n = per * a.scenes
        rows[g] = {"benchmark": BENCH.get(g, ""), "per_scene": round(per, 2), "source": src,
                   "per_100_scenes": round(n), "ci95_half_width_at_50pct": round(half_width(n), 3)}
    by_bench: Dict[str, float] = {}
    for g, r in rows.items():
        for b in r["benchmark"].split(" / "):
            by_bench[b] = by_bench.get(b, 0) + r["per_100_scenes"] / len(r["benchmark"].split(" / "))
    out = {"scenes": a.scenes, "groups": rows,
           "by_benchmark_approx": {b: {"per_100_scenes": round(n), "ci95_half_width_at_50pct": round(half_width(n), 3)}
                                   for b, n in by_bench.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for g, r in rows.items():
        print(g, r)
    print(out["by_benchmark_approx"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
