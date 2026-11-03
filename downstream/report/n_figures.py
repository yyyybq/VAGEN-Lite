#!/usr/bin/env python3
"""Paper figures of the N, QA and V4 results (the user's decision 4 of 2026-11-03): matplotlib, PNG and SVG, labels in
English (the cluster has no CJK font), 95% Wilson intervals; DIR/figures.json keeps every number plotted.

    python downstream/report/n_figures.py --out DIR [--runs /mnt/umm/users/yinbaiqiao/downstream_runs]

n1_by_type: N1 and its second set merged (600 instances, 100 per task type): success per task type and overall for
  the eight models, one panel per generation (Qwen2.5-VL-7B: B, S, RS, RD; Qwen3-VL-8B: Q3B, Q3S, Q3RS, Q3RD), the
  same colour for the same role in both (n/n1_merged_20261102/report.json, "combined").
qa_heatmap: single-image question answering, the eight models x three question sets (X0: RoboCasa frames, 510
  two-choice spatial questions; X0b-1: InteriorGS frames, 510 two-choice spatial questions; G0: 300 non-spatial
  recognition questions), each cell the accuracy and its k / n (x/QA_20261030/results: q0_vs_B.json, x0b1_vs_B.json,
  g0.json, every model's "all").
v4_curve: the V4 recheck's three checkpoints (150k / 175k / 200k downstream training steps, 943 episodes each): the
  model line's Q3B and Q3RD against the two re-trained dense-reward RL starts V4S3 and DCKS2 (v4_recheck/report.json
  "counts").
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RUNS = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")
FAMILIES = ("projective_relations", "equidistance", "absolute_positioning", "occlusion_alignment", "fov_inclusion",
            "screen_occupancy")
TYPE_LABEL = {"projective_relations": "Left / right", "equidistance": "Equal\ndistance",
              "absolute_positioning": "Absolute\ndistance", "occlusion_alignment": "Occlusion",
              "fov_inclusion": "Both in\nview", "screen_occupancy": "Screen\nshare", "all": "All\n(600)"}
GENERATIONS = (("Qwen2.5-VL-7B", ("B", "S", "RS", "RD")), ("Qwen3-VL-8B", ("Q3B", "Q3S", "Q3RS", "Q3RD")))
ROLE_COLOR = ("#7a8291", "#3569b5", "#d9822b", "#2f9a6a")  # base, SFT, sparse RL, dense RL (figures.START_COLOR)
ROLE_NAME = ("base", "SFT", "RL sparse", "RL dense")
QA_SETS = (("X0", "q0_vs_B.json", "X0\nRoboCasa frames\nspatial, 2-choice"),
           ("X0b-1", "x0b1_vs_B.json", "X0b-1\nInteriorGS frames\nspatial, 2-choice"),
           ("G0", "g0.json", "G0\nnon-spatial\nrecognition"))
V4_LINES = (("Q3B", "Q3B  base (model line)", "#7a8291", "-", "o"),
            ("Q3RD", "Q3RD  V4 dense RL (model line)", "#2f9a6a", "-", "s"),
            ("Q3V4S3", "V4S3  V4 recipe, new seed", "#1f6f8b", "--", "D"),
            ("Q3DCKS2", "DCKS2  DCK recipe", "#8e5bb5", "--", "^"))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": "#c9ced6",
        "axes.labelcolor": "#1f2933", "xtick.color": "#66707c", "ytick.color": "#66707c", "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "svg.fonttype": "none"})
    return plt


def save(fig, out: Path, name: str) -> list:
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    for p in paths:
        fig.savefig(p)
    return [p.name for p in paths]


# ------------------------------------------------------------------------------------------------- data
def n1_data(runs: Path) -> dict:
    comb = json.loads((runs / "n/n1_merged_20261102/report.json").read_text())["combined"]
    out = {}
    for _, models in GENERATIONS:
        for m in models:
            rates = comb["models"][m]["rates"]
            cells = {t: (rates[t]["k"], rates[t]["n"]) for t in FAMILIES}
            cells["all"] = (sum(k for k, _ in cells.values()), sum(n for _, n in cells.values()))
            out[m] = cells
    return out


def qa_data(runs: Path) -> dict:
    base = runs / "x/QA_20261030/results"
    out = {}
    for key, fname, _ in QA_SETS:
        r = json.loads((base / fname).read_text())
        out[key] = {m: (r[m]["all"]["correct"], r[m]["all"]["n"]) for _, models in GENERATIONS for m in models}
    return out


def v4_data(runs: Path) -> dict:
    r = json.loads((runs / "v4_recheck/report.json").read_text())
    return {"points": r["points"], "counts": {k: [tuple(c) for c in v] for k, v in r["counts"].items()}}


# ---------------------------------------------------------------------------------------------- figures
def n1_by_type(data: dict, out: Path) -> list:
    import numpy as np

    plt = plt_setup()
    groups = list(FAMILIES) + ["all"]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), sharey=True)
    x = np.arange(len(groups))
    w = 0.2
    top = max(wilson(*data[m][g])[2] for _, ms in GENERATIONS for m in ms for g in groups)
    for ax, (gen, models) in zip(axes, GENERATIONS):
        for i, m in enumerate(models):
            vals = [wilson(*data[m][g]) for g in groups]
            p = np.array([v[0] for v in vals]) * 100
            lo = p - np.array([v[1] for v in vals]) * 100
            hi = np.array([v[2] for v in vals]) * 100 - p
            ax.bar(x + (i - 1.5) * w, p, w, color=ROLE_COLOR[i], label=f"{m}  {ROLE_NAME[i]}", zorder=2)
            ax.errorbar(x + (i - 1.5) * w, p, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=0.7,
                        capsize=1.6, zorder=3)
        ax.axvline(len(FAMILIES) - 0.5, color="#c9ced6", lw=0.8)
        ax.set_title(gen, loc="left", fontweight="bold")
        ax.set_ylabel("Success (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([TYPE_LABEL[g] for g in groups])
        ax.set_ylim(0, math.ceil(top * 100 / 5) * 5 + 2)
        ax.grid(axis="y", color="#e4e7eb", lw=0.7, zorder=0)
        ax.legend(ncol=4, loc="upper left", fontsize=8, handlelength=1.2, columnspacing=1.2)
    fig.text(0.0, -0.02, "RoboCasa near transfer (N1 + second set): 600 instances, 100 per task type, one greedy "
             "episode per model and instance. Bars: success rate; whiskers: 95% Wilson interval.", fontsize=7.5,
             color="#66707c")
    fig.tight_layout()
    files = save(fig, out, "n1_by_type")
    plt.close(fig)
    return files


def qa_heatmap(data: dict, out: Path) -> list:
    import numpy as np

    plt = plt_setup()
    models = [m for _, ms in GENERATIONS for m in ms]
    acc = np.array([[data[s][m][0] / data[s][m][1] * 100 for s, _, _ in QA_SETS] for m in models])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    im = ax.imshow(acc, cmap="YlGnBu", vmin=50, vmax=90, aspect="auto")
    for i, m in enumerate(models):
        for j, (s, _, _) in enumerate(QA_SETS):
            k, n = data[s][m]
            ax.text(j, i, f"{acc[i, j]:.1f}%\n{k}/{n}", ha="center", va="center", fontsize=8,
                    color="white" if acc[i, j] > 76 else "#1f2933")
    ax.axhline(3.5, color="white", lw=3)
    ax.set_xticks(range(len(QA_SETS)))
    ax.set_xticklabels([lab for _, _, lab in QA_SETS], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)))
    roles = ROLE_NAME * 2
    ax.set_yticklabels([f"{m}  {roles[i]}" for i, m in enumerate(models)])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Accuracy (%)")
    cb.outline.set_visible(False)
    ax.set_xlabel("Rows 1-4: Qwen2.5-VL-7B; rows 5-8: Qwen3-VL-8B. X0 and X0b-1: 510 questions each,\n"
                  "chance 50%; G0: 300 questions. Greedy answers, one per question.", fontsize=7.5, color="#66707c")
    fig.tight_layout()
    files = save(fig, out, "qa_heatmap")
    plt.close(fig)
    return files


def v4_curve(data: dict, out: Path) -> list:
    plt = plt_setup()
    pts = data["points"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, (key, lab, col, ls, mk) in enumerate(V4_LINES):
        vals = [wilson(k, n) for k, n in data["counts"][key]]
        xs = [p / 1000 + (i - 1.5) * 0.9 for p in pts]
        p = [v[0] * 100 for v in vals]
        ax.errorbar(xs, p, yerr=[[a - v[1] * 100 for a, v in zip(p, vals)], [v[2] * 100 - a for a, v in zip(p, vals)]],
                    color=col, ls=ls, marker=mk, ms=4.5, lw=1.4, elinewidth=0.8, capsize=2, label=lab)
    ax.set_xticks([p / 1000 for p in pts])
    ax.set_xticklabels([f"{p // 1000}k" for p in pts])
    ax.set_xlabel("Downstream training step (E1 protocol, 100% demonstrations)")
    ax.set_ylabel("Success (%)")
    ax.grid(axis="y", color="#e4e7eb", lw=0.7)
    ax.legend(fontsize=8, loc="lower left", ncol=2, columnspacing=1.0)
    lo = min(wilson(k, n)[1] for v in data["counts"].values() for k, n in v) * 100
    hi = max(wilson(k, n)[2] for v in data["counts"].values() for k, n in v) * 100
    ax.set_ylim(math.floor(lo) - 4, math.ceil(hi) + 1)
    ax.set_title("RoboCasa, 34 tasks, 943 episodes per point; whiskers: 95% Wilson", fontsize=8.5, color="#66707c",
                 loc="left")
    fig.tight_layout()
    files = save(fig, out, "v4_curve")
    plt.close(fig)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=RUNS)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    n1, qa, v4 = n1_data(args.runs), qa_data(args.runs), v4_data(args.runs)
    files = n1_by_type(n1, args.out) + qa_heatmap(qa, args.out) + v4_curve(v4, args.out)
    rec = {"n1_by_type": {m: {g: {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n)[1:]} for g, (k, n) in c.items()}
                          for m, c in n1.items()},
           "qa_heatmap": {s: {m: {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n)[1:]} for m, (k, n) in c.items()}
                          for s, c in qa.items()},
           "v4_curve": {"points": v4["points"],
                        "counts": {m: [{"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n)[1:]} for k, n in c]
                                   for m, c in v4["counts"].items()}},
           "sources": {"n1_by_type": "n/n1_merged_20261102/report.json",
                       "qa_heatmap": [f"x/QA_20261030/results/{f}" for _, f, _ in QA_SETS],
                       "v4_curve": "v4_recheck/report.json"}}
    (args.out / "figures.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
