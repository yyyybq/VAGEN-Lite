#!/usr/bin/env python3
"""The matrix v3 starts on N1's merged 600 instances, in N1's tables (the user, 2026-11-03 evening).

    python downstream/n/nm_report.py --instances INSTANCES_600 --eval NM [--n1 EVAL1 EVAL2] [--starts FILE]
                                     [--exclude SCAN ...] --out DIR

Episodes: the starts of nm_starts.json from NM/<name>/episodes.jsonl (run_nm.sh); Q3B, Q3S, Q3RS and Q3RD from N1's
two evaluation folders (their episodes on the same 600 instances, nothing played again).  Tables as n_report.py
(success per task type with the 95% Wilson interval; paired differences with McNemar's exact p and the scene
resampling interval; near-threshold and the rest; behaviour), with three pairings:
  * versus: every model against the base Q3B;
  * versus_sft: every RL start against the SFT start it grew from (C against C-SFT, D against D-SFT, E against Q3S,
    the main line's formal SFT epoch 8; N1's Q3RS and Q3RD against Q3S);
  * versus_s0: each V4 start against the S0 start of the same group and seed.
An overview first: per group and reward, each seed's success and its difference from the SFT start.
--exclude: instance ids to leave out everywhere (n_frames.py scan reports or id files, as n_report.py).
Writes DIR/report.json and DIR/report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_report as R  # noqa: E402
import nm_starts  # noqa: E402

N1_EVALS = (Path("/mnt/umm/users/yinbaiqiao/downstream_runs/n/n1_eval_20261031"),
            Path("/mnt/umm/users/yinbaiqiao/downstream_runs/n/n1b_eval_20261102"))
N1_MODELS = ("Q3B", "Q3S", "Q3RS", "Q3RD")
PAIRINGS = (("versus", "和底座 Q3B 配对"),
            ("versus_sft", "RL 和它的 SFT 起点配对（C 对 C-SFT，D 对 D-SFT，E 对 Q3S；N1 的 Q3RS、Q3RD 对 Q3S）"),
            ("versus_s0", "同组同种子 V4 对 S0"))
PAIR_NOTE = ("全部对 Q3B；C 的 RL 对 C-SFT，D 的对 D-SFT，E 的对 Q3S（E 的 RL 从主线正式 SFT 第 8 轮起步，就是 N1 的 Q3S）；"
             "N1 的 Q3RS、Q3RD 对 Q3S；V4 对同组同种子的 S0。")


def order(starts) -> list:
    """Group by group, SFT first, then seed, S0 before V4."""
    return sorted(starts, key=lambda e: (e["group"], e["reward"] != "SFT", e.get("seed") or 0, e["reward"]))


def refs(starts, models) -> dict:
    by = {(e["group"], e.get("seed"), e["reward"]): e["name"] for e in starts}
    out = {"versus": {m: "Q3B" for m in models if m != "Q3B"},
           "versus_sft": {"Q3RS": "Q3S", "Q3RD": "Q3S"},
           "versus_s0": {}}
    for e in starts:
        if e.get("sft"):
            out["versus_sft"][e["name"]] = e["sft"]
        if e["reward"] == "V4" and (e["group"], e.get("seed"), "S0") in by:
            out["versus_s0"][e["name"]] = by[(e["group"], e.get("seed"), "S0")]
    return out


def load(instances: Path, nm: Path, starts, n1=N1_EVALS, exclude=()) -> tuple:
    insts = {x["instance_id"]: x for x in R.read_jsonl(instances) if x["instance_id"] not in exclude}
    recs = {m: [r for d in n1 for r in R.read_jsonl(Path(d) / m / "episodes.jsonl") if r.get("instance_id") in insts]
            for m in N1_MODELS}
    for e in starts:
        recs[e["name"]] = [r for r in R.read_jsonl(Path(nm) / e["name"] / "episodes.jsonl")
                           if r.get("instance_id") in insts]
    for m, rs in recs.items():
        ids = [r["instance_id"] for r in rs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{m}: an instance played twice")
    return insts, recs


def build(instances: Path, nm: Path, starts, n1=N1_EVALS, exclude=()) -> dict:
    starts = order(starts)
    insts, recs = load(instances, nm, starts, n1, exclude)
    # N1's models where their folders have episodes; every start, so one without episodes shows as missing
    models = [m for m in N1_MODELS if recs[m]] + [e["name"] for e in starts]
    rep = R.summarize(insts, recs, models, refs(starts, models))
    rep["starts"] = starts
    return rep


def overview(rep: dict) -> list:
    rows = {}
    for e in rep["starts"]:
        rows.setdefault((e["group"], e["reward"]), {})[e.get("seed")] = e["name"]
    seeds = sorted({s for r in rows.values() for s in r if s is not None})
    lines = ["| 组 | 起点 | " + " | ".join(f"种子 {s}" for s in seeds) + " |", "| --- | --- |" + " --- |" * len(seeds)]
    for (g, reward), by_seed in rows.items():
        cells = []
        for s in seeds if reward != "SFT" else seeds[:1]:
            name = by_seed.get(s if reward != "SFT" else None)
            if name is None:
                cells.append("没有这个起点")
                continue
            r = rep["models"][name]
            text = R.cell(r["rates"]["all"])
            if "versus_sft" in r:
                text += f"；对 {r['versus_sft']['base']} {R.diff_cell(r['versus_sft']['all'])}"
            cells.append(text)
        if reward == "SFT":
            cells += [""] * (len(seeds) - 1)
        lines.append(f"| {g} | {reward} | " + " | ".join(cells) + " |")
    return lines


def markdown(rep: dict) -> str:
    head = [f"# 矩阵 v3 起点按 N1 的做法评（{rep['instances']} 个实例，门槛附近 {rep['near_threshold']} 个）", "",
            "## 总览：成功 / 回合（百分比，95% 区间）；RL 对它的 SFT 起点的配对差（百分点，95% 区间；只有 RL 成功对"
            "只有 SFT 成功的回合数，精确 McNemar p）", ""] + overview(rep) + [
            "", "- 权重来自主线挑的检查点，按路径读、不复制；每个模型看它自己那条训练链的验证提示（C、D 只输出动作，"
                "E 先想再输出动作），和 N1 一样只给当前一轮。Q3B、Q3S、Q3RS、Q3RD 用 N1 现成的回合。", ""]
    return "\n".join(head) + "\n" + R.markdown(rep, "各模型（N1 的表）", "##", PAIRINGS, PAIR_NOTE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--n1", type=Path, nargs="+", default=list(N1_EVALS))
    ap.add_argument("--starts", type=Path, default=nm_starts.STARTS)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rep = build(args.instances, args.eval, nm_starts.load(args.starts), args.n1, R.excluded(args.exclude))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (args.out / "report.md").write_text(markdown(rep))
    print(json.dumps({m: r["rates"]["all"] for m, r in rep["models"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
