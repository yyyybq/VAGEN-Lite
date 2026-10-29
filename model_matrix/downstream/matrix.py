"""The model line's downstream matrix (the user's 3.4 of 2026-10-27: every point in DS-B's form, in one table with the
downstream line's four starts): per start, the training state and the E1 points (150k, 175k, 200k: overall and the
9 spatial tasks, successes of the episodes after the fixed exclusion list), their mean once all three are in, and
each point against the start's reference (the downstream line's B for its own starts, this line's raw start for
the rest), paired by task and episode.  Everything is read with the downstream line's own report code (collect,
stats; read only).

    python -m model_matrix.downstream.matrix --out docs/models/MATRIX.md
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

DS_WORKTREE = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream")
DS_RUNS = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")
OUR_RUNS = Path("/mnt/umm/users/yinbaiqiao/models_runs/downstream")
POINTS = (150000, 175000, 200000)
# (line, code, label, reference)
ROWS = [
    ("下游线", "B", "原始 Qwen2.5-VL-7B", None),
    ("下游线", "S", "Qwen2.5-VL 动作 SFT 第 9 个 epoch", "B"),
    ("下游线", "RS", "Qwen2.5-VL P-S0 第 300 次", "B"),
    ("下游线", "RD", "Qwen2.5-VL P-D 第 150 次", "B"),
    ("本线", "Q3B", "原始 Qwen3-VL-8B", None),
    ("本线", "Q3S", "Qwen3-VL SFT 第 8 个 epoch", "Q3B"),
    ("本线", "Q3RD", "Qwen3-VL 密集第 300 次（种子 20261019）", "Q3B"),
    ("本线", "Q3RS", "Qwen3-VL S0 第 100 次（该臂第 150 次走样停下）", "Q3B"),
    ("本线", "CSB", "原始 Cambrian-S-7B", None),
    ("本线", "CSS", "Cambrian SFT 第 5 个 epoch", "CSB"),
    ("本线", "CSRS", "Cambrian S0 第 100 次（该臂第 122 次走样停下）", "CSB"),
    ("本线", "CSRD", "Cambrian 密集重跑第 50 次（该臂第 129 次走样停下）", "CSB"),
]


def rate(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}%（{k}/{n}）" if n else "—"


def point_cell(groups: dict) -> str:
    """groups: the downstream line's group_counts, {group: [successes, episodes]}."""
    return f"总体 {rate(*groups['all'])}<br>空间 {rate(*groups['spatial'])}"


def diff_cell(cmp: dict) -> str:
    """cmp: the downstream line's compare()['vs_reference'][start]['groups'] for one point."""
    a, s = cmp["all"], cmp["spatial"]
    return (f"总体 {100 * a['diff']:+.1f}（p {a['p']:.2g}）<br>"
            f"空间 {100 * s['diff']:+.1f}（p {s['p']:.2g}）")


def train_state(exp: Path) -> str:
    """The training state from the run's own files: done (its final evaluation summary), the last step in
    train.log, or not started."""
    if (exp / "evals" / "steps_200000" / "out" / "episodes" / "summary.json").is_file():
        return "训完、评完"
    log = exp / "train.log"
    if log.is_file():
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 20000))
            tail = f.read().decode("utf-8", "replace")
        steps = re.findall(r"(\d+)/200000 \[", tail)
        if steps and int(steps[-1]) >= POINTS[-1]:
            return "训完，任务内评测中"
        if steps:
            return f"训练中：第 {int(steps[-1]):,} 步"
        return "训练中"
    return "排队"


def build() -> str:
    sys.path.insert(0, str(DS_WORKTREE))
    from downstream.report import collect as C, stats as S

    colls = {"下游线": C.collect(DS_RUNS), "本线": C.collect(OUR_RUNS)}
    roots = {"下游线": DS_RUNS, "本线": OUR_RUNS}
    lines = [
        "# 模型线下游矩阵（E1 协议）",
        "",
        f"生成于 {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC，"
        "由 `python -m model_matrix.downstream.matrix` 生成，不要手改。",
        "",
        "- **协议**：下游线的 E1，100% 数据、种子 1、20 万步，评第 15 万、17.5 万、20 万步。"
        "每点 944 个回合，已按下游线固定的剔除清单剔除；主数字用三点平均。",
        "- **数字来源**：下游线自己的 `downstream/report/collect.py` 和 `stats.py`，按路径只读。",
        "- **起点**：下游线的四个起点读 `downstream_runs/experiments/DS-*-100-s1`，本线的读 `models_runs/downstream/experiments/DS-*-100-s1`。",
        "- **对参照**：同一点逐回合配对（按任务、评测种子），差 = 本起点 − 参照，p 是 McNemar 精确检验。"
        "下游线三个起点的参照是 B，Qwen3-VL 的是 Q3B，Cambrian 的是 CSB。",
        "- **空间**：下游线的 9 个空间任务，447 个回合。",
        "",
        "| 线 | 起点 | 训练 | 15 万步 | 17.5 万步 | 20 万步 | 三点平均 | 对参照（15 万 / 17.5 万 / 20 万步） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for line, code, label, ref in ROWS:
        coll = colls[line]
        exp = roots[line] / "experiments" / f"DS-{code}-100-s1"
        have = {e.step for e in coll.first() if (e.start, e.ratio, e.seed) == (code, "100", 1)}
        cells, rates = [], {"all": [], "spatial": []}
        for st in POINTS:
            if st in have:
                g = S.group_counts(S.episodes_of(coll, code, "100", st, {1}), coll.tasks)
                cells.append(point_cell(g))
                for k in rates:
                    rates[k].append(g[k][0] / g[k][1])
            else:
                cells.append("—")
        mean = ("—" if len(rates["all"]) < len(POINTS) else
                f"总体 {100 * sum(rates['all']) / 3:.1f}%<br>空间 {100 * sum(rates['spatial']) / 3:.1f}%")
        diffs = []
        if ref:
            ref_have = {e.step for e in coll.first() if (e.start, e.ratio, e.seed) == (ref, "100", 1)}
            for st in POINTS:
                if st in have and st in ref_have:
                    r = S.compare(coll, "100", st, {1}, reference=ref)
                    diffs.append(diff_cell(r["vs_reference"][code]["groups"]))
                else:
                    diffs.append("—")
        state = "训完、评完" if line == "下游线" and len(have) == 3 else train_state(exp)
        lines.append(f"| {line} | **{code}** {label}<br>`DS-{code}-100-s1` | {state} | {' | '.join(cells)} | {mean} | "
                     f"{' ／ '.join(diffs) if ref else '（参照）'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(build(), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
