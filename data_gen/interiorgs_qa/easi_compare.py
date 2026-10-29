"""Side-by-side table of EASI-8 results (outputs of easi_job.py), scored by EASI's own adapter.

    python -m data_gen.interiorgs_qa.easi_compare --root OUT --model 基座=base7b --model QA-SFT=qasft256_e3 \
        --model Act-SFT=actsft_r11_e9 --out TABLE.md

For every model folder OUT/<name>/ the lmms-eval result files are read with EASI v0.2.2's
``LmmsEvalAdapter.extract_scores`` (third_party/EASI/scripts/submissions/backends/lmmseval.py), the code that writes
EASI's official payload: headline score per benchmark (VSI overall, MMSI average, MindCube-Tiny overall accuracy,
ViewSpatial overall accuracy, SITE from its image and video tasks, BLINK accuracy, 3DSRBench plain accuracy,
EmbSpatial accuracy; 0-100) and the sub-scores EASI reports. Missing benchmarks show "—"; the EASI-8 mean is given
only when all 8 are present. Differences: every model minus the first, and the second minus the third when three
models are given. Reads files only; runs in a second.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EASI = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/EASI/scripts/submissions")
EASI_8 = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site", "blink", "3dsrbench", "embspatial")
NAMES = {"vsi_bench": "VSI-Bench", "mmsi_bench": "MMSI-Bench", "mindcube_tiny": "MindCube-Tiny",
         "viewspatial": "ViewSpatial", "site": "SITE", "blink": "BLINK", "3dsrbench": "3DSRBench",
         "embspatial": "EmbSpatial"}


def scores(model_root: Path) -> Dict[str, Tuple[Optional[float], Dict[str, Optional[float]]]]:
    sys.path.insert(0, str(EASI))
    from backends.lmmseval import _TASK_MAP, LmmsEvalAdapter   # noqa: E402

    dirs = [d for d in model_root.iterdir() if d.is_dir() and any(d.glob("*_results.json"))]
    if not dirs:
        return {}
    bench = {k: v for k, v in _TASK_MAP.items() if k in EASI_8 or k in ("site_image", "site_video")}
    got = LmmsEvalAdapter().extract_scores(dirs[0], model_root.name, bench)
    return {k: (s.overall, dict(s.sub_scores)) for k, s in got.items()}


def cell(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1f}"


def diff(a: Optional[float], b: Optional[float]) -> str:
    return "—" if a is None or b is None else f"{a - b:+.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", action="append", required=True, help="LABEL=folder under --root")
    ap.add_argument("--title", default="EASI-8 对照")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models: List[Tuple[str, Dict]] = []
    for m in a.model:
        label, folder = m.split("=", 1)
        models.append((label, scores(Path(a.root) / folder)))
    labels = [l for l, _ in models]
    extra = [f"{labels[i]} − {labels[0]}" for i in range(1, len(models))]
    if len(models) == 3:
        extra.append(f"{labels[1]} − {labels[2]}")

    def row(name: str, vals: List[Optional[float]]) -> str:
        ds = [diff(vals[i], vals[0]) for i in range(1, len(vals))]
        if len(vals) == 3:
            ds.append(diff(vals[1], vals[2]))
        return f"| {name} | " + " | ".join(cell(v) for v in vals) + " | " + " | ".join(ds) + " |"

    head = "| | " + " | ".join(labels) + " | " + " | ".join(extra) + " |"
    sep = "| --- " * (1 + len(labels) + len(extra)) + "|"
    lines = [f"# {a.title}", "", "EASI v0.2.2 的官方打分（`LmmsEvalAdapter.extract_scores`），0–100。", "",
             "## 各 benchmark 的主分数", "", head, sep]
    for b in EASI_8:
        lines.append(row(NAMES[b], [s.get(b, (None, {}))[0] for _, s in models]))
    means = []
    for _, s in models:
        v = [s.get(b, (None, {}))[0] for b in EASI_8]
        means.append(statistics.mean(v) if all(x is not None for x in v) else None)
    lines.append(row("**EASI-8 平均**", means))
    present = [b for b in EASI_8 if all(s.get(b, (None, {}))[0] is not None for _, s in models)]
    if len(present) < len(EASI_8):
        sub = []
        for _, s in models:
            sub.append(statistics.mean(s[b][0] for b in present) if present else None)
        lines.append(row(f"已完成的 {len(present)} 个的平均", sub))
    lines += ["", "## 子分数", "", "| benchmark | 子项 | " + " | ".join(labels) + " | " + " | ".join(extra) + " |",
              "| --- " * (2 + len(labels) + len(extra)) + "|"]
    for b in EASI_8:
        keys: List[str] = []
        for _, s in models:
            for k in s.get(b, (None, {}))[1]:
                if k not in keys:
                    keys.append(k)
        for k in keys:
            vals = [s.get(b, (None, {}))[1].get(k) for _, s in models]
            if all(v is None for v in vals):
                continue
            r = row(k, vals)
            lines.append(f"| {NAMES[b]} " + r)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
