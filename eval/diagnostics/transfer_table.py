"""Diagnostic 4: which QA task families were trained x how the success of each task-family group changed.

    python -m eval.diagnostics.transfer_table --config CONFIG.yaml --out OUT_PREFIX

CONFIG:
    groups:                      # the 3 x 3 table's groups (default below)
      投影: [projective_relations]
      距离类: [absolute_positioning, equidistance, screen_occupancy]
      可见性类: [occlusion_alignment, fov_inclusion]
    baseline: {eval: EVAL_DIR, name: RESULT_FOLDER}          # the model trained without these QA questions
    runs:
      - {eval: EVAL_DIR, name: RESULT_FOLDER, qa_groups: [投影]}   # trained with the QA of these groups

A cell (row: QA group trained, column: task group evaluated) is the run's success rate on that column's tasks
minus the baseline's, in points, averaged over the runs whose ``qa_groups`` contain the row; the tasks are the
evaluation folder's (results of the same tasks, paired by index).  Runs come from the experiment matrix
(configs/matrix_v2); there are none yet (2026-10-18).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

DEFAULT_GROUPS = {"投影": ["projective_relations"],
                  "距离类": ["absolute_positioning", "equidistance", "screen_occupancy"],
                  "可见性类": ["occlusion_alignment", "fov_inclusion"]}


def results(eval_dir: str, name: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for line in (Path(eval_dir) / name / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[int(r["index"])] = {"family": r["task_type"], "success": bool(r["success"])}
    return out


def rate(res: Dict[int, Dict[str, Any]], families: Sequence[str], keep: Sequence[int]) -> Optional[float]:
    sel = [res[i]["success"] for i in keep if res[i]["family"] in families]
    return sum(sel) / len(sel) if sel else None


def table(cfg: Dict[str, Any]) -> Dict[str, Any]:
    groups = cfg.get("groups") or DEFAULT_GROUPS
    base = results(cfg["baseline"]["eval"], cfg["baseline"]["name"])
    cells: Dict[str, Dict[str, List[float]]] = {g: {h: [] for h in groups} for g in groups}
    for run in cfg.get("runs") or []:
        res = results(run["eval"], run["name"])
        keep = sorted(set(res) & set(base))
        for col, fams in groups.items():
            b, r = rate(base, fams, keep), rate(res, fams, keep)
            if b is None or r is None:
                continue
            for row in run["qa_groups"]:
                cells[row][col].append(100.0 * (r - b))
    return {row: {col: (sum(v) / len(v) if v else None, len(v)) for col, v in cols.items()} for row, cols in cells.items()}


def markdown(t: Dict[str, Any]) -> str:
    cols = list(next(iter(t.values())).keys())
    lines = ["# 迁移表：训练用的 QA 任务族 × 各任务族成功率的变化（百分点，相对基线）", "",
             "| 训练用的 QA ＼ 评测的任务族 | " + " | ".join(cols) + " |", "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for row, cs in t.items():
        lines.append(f"| {row} | " + " | ".join("–" if v is None else f"{v:+.1f}（{n} 组）" for v, n in cs.values()) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = table(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")))
    Path(a.out + ".json").write_text(json.dumps(t, indent=1, ensure_ascii=False))
    Path(a.out + ".md").write_text(markdown(t), encoding="utf-8")
    print(markdown(t))


if __name__ == "__main__":
    main()
