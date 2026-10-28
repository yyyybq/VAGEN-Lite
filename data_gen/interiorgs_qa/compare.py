"""Put evaluations (evaluate.py outputs) side by side, one section per wording family.

    python -m data_gen.interiorgs_qa.compare --eval NAME1=DIR1 --eval NAME2=DIR2 [--res 256] --out OUT.md

For each family ("orig": the benchmark's own format, "h1": the held-out wording): the benchmark scores merged the
benchmark's own way (evaluate.official_merge), then every benchmark category (tasks.py), each with its question
count and 95% interval; with two evaluations, the difference (second minus first). MMSI-Bench and 3DSRBench show the
plain and the circular (rotated options) score. Reads the summaries only, runs in a second.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ORDER = ("VSI-Bench", "MindCube", "MindCube（平移，单列）", "MMSI-Bench", "BLINK", "ViewSpatial-Bench", "EmbSpatial-Bench",
         "3DSRBench")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def official_cells(o: Optional[Dict[str, Any]], name: str) -> Tuple[str, Optional[float]]:
    """Cell text and the headline number (for differences)."""
    if not o:
        return "—", None
    lo, hi = o.get("ci95") or (None, None)
    if name == "VSI-Bench":
        return f"{o['score']:.3f} ({lo:.3f}–{hi:.3f})", o["score"]
    s = f"{_pct(o['score'])} ({_pct(lo)}–{_pct(hi)})"
    if "circular" in o:
        c = o["circular"]
        s += f"；轮换 {_pct(c['mean'])}"
    if "plain" in o:
        s += f"；普通 {_pct(o['plain']['mean'])}"
    return s, o["score"]


def category_cells(b: Optional[Dict[str, Any]]) -> Tuple[str, Optional[float], int]:
    if not b:
        return "—", None, 0
    if "mra" in b:
        m = b["mra"]
        return f"MRA {m['mean']:.3f} ({m['ci95'][0]:.3f}–{m['ci95'][1]:.3f})", m["mean"], m["n"]
    a = b["accuracy"]
    s = f"{_pct(a['mean'])} ({_pct(a['ci95'][0])}–{_pct(a['ci95'][1])})"
    if "circular_accuracy" in b:
        s += f"；轮换 {_pct(b['circular_accuracy']['mean'])}"
    return s, a["mean"], a["n"]


def diff_text(a: Optional[float], b: Optional[float], scale: float) -> str:
    if a is None or b is None:
        return "—"
    d = (b - a) * scale
    return f"{d:+.1f}" if scale == 100 else f"{d:+.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True, help="NAME=evaluate.py output folder")
    ap.add_argument("--res", default="256")
    ap.add_argument("--title", default="同域评测对照")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    evals: List[Tuple[str, Dict[str, Any]]] = []
    for e in a.eval:
        name, d = e.split("=", 1)
        evals.append((name, json.loads((Path(d) / "summary.json").read_text())["summary"]))
    names = [n for n, _ in evals]
    two = len(evals) == 2
    lines = [f"# {a.title}", "", "数值：准确率为百分数（括号里是 95% 区间），VSI-Bench 总分和数值题为 MRA / 得分（0–1）。"
             + ("最后一列是后者减前者（百分点，或得分差）。" if two else ""), ""]
    for fam, fam_name in (("orig", "benchmark 原格式"), ("h1", "留出措辞")):
        key = f"res{a.res}/{fam}"
        if not all(key in s for _, s in evals):
            continue
        lines += [f"## {fam_name}（{key}）", "", "### 按 benchmark（官方合并方式）", "",
                  "| benchmark | 题数 | " + " | ".join(names) + (" | 差 |" if two else " |"),
                  "| --- | --- | " + " | ".join("---" for _ in names) + (" | --- |" if two else " |")]
        for bname in ORDER:
            cells, vals, n = [], [], None
            for _, s in evals:
                o = s[key]["official"].get(bname)
                t, v = official_cells(o, bname)
                cells.append(t)
                vals.append(v)
                if o:
                    n = o.get("questions", o.get("n"))
            if all(v is None for v in vals):
                continue
            row = f"| {bname} | {n} | " + " | ".join(cells)
            if two:
                row += " | " + diff_text(vals[0], vals[1], 1 if bname == "VSI-Bench" else 100)
            lines.append(row + " |")
        lines += ["", "### 按 benchmark 自己的类别", "",
                  "| benchmark | 类别 | 题数 | " + " | ".join(names) + (" | 差 |" if two else " |"),
                  "| --- | --- | --- | " + " | ".join("---" for _ in names) + (" | --- |" if two else " |")]
        cats = list(evals[0][1][key].get("by_category", {}))
        for c in cats:
            cells, vals, n, is_mra = [], [], 0, False
            for _, s in evals:
                b = s[key].get("by_category", {}).get(c)
                t, v, nn = category_cells(b)
                cells.append(t)
                vals.append(v)
                n = n or nn
                is_mra = is_mra or bool(b and "mra" in b)
            bn, cn = c.split(" | ", 1)
            row = f"| {bn} | `{cn}` | {n} | " + " | ".join(cells)
            if two:
                row += " | " + diff_text(vals[0], vals[1], 1 if is_mra else 100)
            lines.append(row + " |")
        lines.append("")
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
