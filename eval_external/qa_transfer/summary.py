"""Summary table of QA runs: rows are models, columns are benchmarks (and the sub-items the plan asks for).

    python -m eval_external.qa_transfer.summary --stage step2 --out eval_external/results/qa_step2

Each cell: our score, then in brackets the reported score (reported.yaml) and our minus reported.  A second
table gives the Wilson 95% interval and the share of unparseable answers per cell.  Writes OUT.md and OUT.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval_external.sco.job import ROOT

HERE = Path(__file__).resolve().parent
COLUMNS = [  # (column label, benchmark, sub-item or None)
    ("SAT-Real", "SAT-Real", None), ("SAT-Syn", "SAT-Syn", None), ("ViewSpatial", "ViewSpatial", None),
    ("MindCube-Tiny", "MindCube-Tiny", None), ("MindCube", "MindCube", None),
    ("VSI", "VSI-Bench", None), ("VSI 路线规划", "VSI-Bench", "route_planning"),
    ("VSI 相对距离", "VSI-Bench", "object_rel_distance"),
    ("VSTI", "VSTI-Bench", None), ("VSTI 相机位移", "VSTI-Bench", "camera_displacement"),
    ("VSTI 修正前", "VSTI-Bench-prefix", None), ("VSTI 修正前 相机位移", "VSTI-Bench-prefix", "camera_displacement"),
    ("MMSI", "MMSI", None), ("SPBench-MV", "SPBench-MV", None),
]
MODEL_ORDER = ["base-3B", "base-7B", "ours-SFT", "ours-RL-50", "ours-RL-150", "SI-3B", "SI-7B", "SpaceR-7B",
               "VideoR1-7B", "ViLASR-7B", "VST-7B-RL", "SenseNovaSI-7B"]


def load_results(stage: str) -> dict:
    out = {}
    for f in sorted((ROOT / "runs" / "qa" / stage).glob("*/results.json")):
        r = json.loads(f.read_text())
        out[(r["model"], r["bench"])] = r
    return out


def cell_value(r: dict, sub: str | None) -> float | None:
    if r is None:
        return None
    return r["overall"] if sub is None else r["subscores"].get(sub)


def build(stage: str) -> dict:
    res = load_results(stage)
    reported = yaml.safe_load((HERE / "reported.yaml").read_text())["models"]
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in res)] + sorted(
        {k[0] for k in res} - set(MODEL_ORDER))
    table = {}
    for m in models:
        row = {}
        for label, bench, sub in COLUMNS:
            r = res.get((m, bench))
            ours = cell_value(r, sub)
            rep = reported.get(m, {}).get(bench if sub is None else f"{bench}/{sub}")
            row[label] = {"ours": None if ours is None else round(ours, 1), "reported": rep,
                          "diff": None if ours is None or rep is None else round(ours - rep, 1),
                          "ci95": r["ci95"] if (r and sub is None) else None,
                          "unparseable_pct": r["unparseable_pct"] if (r and sub is None) else None,
                          "n": r["n_items"] if r else None}
        table[m] = row
    return table


def markdown(table: dict) -> str:
    labels = [c[0] for c in COLUMNS if any(table[m][c[0]]["ours"] is not None for m in table)]
    head = "| 模型 | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n"
    rows = []
    for m, row in table.items():
        cells = []
        for lab in labels:
            c = row[lab]
            if c["ours"] is None:
                cells.append("—")
            elif c["reported"] is None:
                cells.append(f"{c['ours']}")
            else:
                cells.append(f"{c['ours']} ({c['reported']}, {c['diff']:+.1f})")
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    second = ["| 模型 | " + " | ".join(l for l in labels if " " not in l) + " |",
              "|---|" + "---|" * len([l for l in labels if " " not in l])]
    for m, row in table.items():
        cells = []
        for lab in labels:
            if " " in lab:
                continue
            c = row[lab]
            cells.append("—" if c["ours"] is None else
                         f"[{c['ci95'][0]}, {c['ci95'][1]}] 无法解析 {c['unparseable_pct']}% (n={c['n']})")
        second.append(f"| {m} | " + " | ".join(cells) + " |")
    return ("**分数**（我们测的数；括号里是论文报的数和差值，差值 = 我们 − 论文）\n\n" + head + "\n".join(rows) +
            "\n\n**95% Wilson 区间、无法解析的回答比例、题数**\n\n" + "\n".join(second) + "\n")


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
