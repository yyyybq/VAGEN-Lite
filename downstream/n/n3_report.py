#!/usr/bin/env python3
"""N3's report (the transfer ladder; the user's decision 2 of 2026-11-03, design reports/N3_DESIGN_20261102.md): where
the main line's "RL beats SFT" goes away on the way to RoboCasa.

    python downstream/n/n3_report.py --n3 N3 [--exclude SCAN.json ...] [--out DIR]

Tiers (each RL start against the SFT start it grew from, same generation: RS, RD against S; Q3RS, Q3RD against Q3S):
  a  the main line's own v2 validation (413) and test (292) tasks, upstream scenes, renderer and body: X4's per-task
     results for S, RS, RD (orig-*), the model line's for Q3S, Q3RS, Q3RD (sft_q3formal_e8, q3s0_100, q3d_300); paired
     by task.  (The main line's v2 evaluation shows the model every earlier turn; the N series, as the RL validation
     env, only the current one.)
  c  N1's merged instances solvable under the upstream camera sphere (N3/recheck_sphere/solvable.jsonl), that rule:
     N3/c/<model>/episodes.jsonl; paired by instance.
  d  N1 as it is (the merged 600: n1_eval_20261031 and n1b_eval_20261102); also on tier c's instances only, so that
     c and d differ in the body alone.
  e  the robot's own camera (N3/e/<model>/episodes.jsonl, 300 new instances), all eight models.
The user's task book of 2026-11-05 (items 3-5) adds:
  a_nohist  tier a again with only the current turn shown (evaluation.py run --no-history, everything else as a):
            N3/a_nohist/eval_{val,test}_v2/<model>/results.jsonl; paired by task.
  f  tier c's instances with N1's robot body but turns never blocked (n3_env collision robot_free_turn):
     N3/f/<model>/episodes.jsonl.
  RD against RS and Q3RD against Q3RS in every tier (the two RL rewards on one start).
Per tier and model: successes / episodes with the 95% Wilson interval; per pairing the paired difference, its 95%
interval and McNemar's exact p (n_report.compare).  Across tiers, for each pairing: c against d, f against d (both on
the same instances) and a against a_nohist (the same tasks), the difference of the two paired differences with a 95%
interval from resampling instances or tasks (DRAWS, fixed seed); a against c, d and e and a_nohist against c,
independent sets, a z test of the gap (n_report.agree).  Per model: f against d and a_nohist against a, its own paired
difference.  --exclude: n_frames.py scan reports or id files, every instance listed left out of every tier (all
models alike).  Writes DIR/report.json and report.md (DIR defaults to N3).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_report as R  # noqa: E402

RUNS = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")
QWEN3_EVALS = Path("/mnt/umm/users/yinbaiqiao/models_runs/eval_sets")
TIER_A = {"S": ("x4", "orig-S"), "RS": ("x4", "orig-RS"), "RD": ("x4", "orig-RD"),
          "Q3S": ("q3", "sft_q3formal_e8"), "Q3RS": ("q3", "q3s0_100"), "Q3RD": ("q3", "q3d_300")}
LADDER = ("S", "RS", "RD", "Q3S", "Q3RS", "Q3RD")
RL = ("RS", "RD", "Q3RS", "Q3RD")
RL_PAIRS = (("RD", "RS"), ("Q3RD", "Q3RS"))  # the two RL rewards on one start (task book item 5)
TIER_ZH = {"a": "(a) 上游原样", "a_nohist": "(a) 上游原样，不带对话历史", "c": "(c) RoboCasa + 上游身体",
           "d": "(d) N1 原样", "d_on_c": "(d) 只取 (c) 的实例", "f": "(f) N1 的身体，原地转身不算碰撞",
           "e": "(e) 机器人自己的相机"}
SIDE_BY_SIDE = ("a", "a_nohist", "c", "d_on_c", "f", "d", "e")
DRAWS, SEED = 2000, 20261103


def tier_a(runs: Path = RUNS, q3: Path = QWEN3_EVALS) -> dict:
    """{model: {task id: success}} over the v2 validation and test sets."""
    out = {}
    for m, (src, name) in TIER_A.items():
        succ = {}
        for s in ("val", "test"):
            base = runs / "x/X4" if src == "x4" else q3
            for r in R.read_jsonl(base / f"eval_{s}_v2" / name / "results.jsonl"):
                succ[str(r["task_id"])] = bool(r.get("success"))
        if succ:
            out[m] = succ
    return out


def tier_a_nohist(n3: Path) -> dict:
    """{model: {task id: success}} of tier a without the dialogue history (N3/a_nohist, one folder per model)."""
    out = {}
    for m in LADDER:
        succ = {}
        for s in ("val", "test"):
            for r in R.read_jsonl(Path(n3) / "a_nohist" / f"eval_{s}_v2" / m / "results.jsonl"):
                succ[str(r["task_id"])] = bool(r.get("success"))
        if succ:
            out[m] = succ
    return out


def episodes(dirs, models, exclude=()) -> dict:
    """{model: {instance id: success}} from n_episode.py folders (DIR/<model>/episodes.jsonl), errors left out."""
    out = {}
    for m in models:
        succ = {}
        for d in dirs:
            for r in R.read_jsonl(Path(d) / m / "episodes.jsonl"):
                if not r.get("error") and r["instance_id"] not in exclude:
                    succ[r["instance_id"]] = bool(r.get("success"))
        if succ:
            out[m] = succ
    return out


def tier_table(succ: dict) -> dict:
    """Rates per model and the RL-SFT pairings of one tier (on the ids both have)."""
    out = {"models": {m: R.rate(s, list(s)) for m, s in succ.items()}, "pairs": {}}
    for m, ref in [(m, R.sft_of(m)) for m in RL] + list(RL_PAIRS):
        if m in succ and ref in succ:
            ids = sorted(set(succ[m]) & set(succ[ref]))
            out["pairs"][f"{m}-{ref}"] = R.compare(succ[m], succ[ref], ids)
    for m in ("S", "RS", "RD", "Q3S", "Q3RS", "Q3RD"):
        b = R.base_of(m)
        if m in succ and b in succ:
            ids = sorted(set(succ[m]) & set(succ[b]))
            out["pairs"][f"{m}-{b}"] = R.compare(succ[m], succ[b], ids)
    return out


def paired_gap(c: dict, d: dict, rl: str, ref: str, draws: int = DRAWS, seed: int = SEED) -> dict:
    """(rl - ref on c) - (rl - ref on d) on the instances all four have, with a 95% interval from resampling them."""
    ids = sorted(set(c.get(rl, {})) & set(c.get(ref, {})) & set(d.get(rl, {})) & set(d.get(ref, {})))
    if not ids:
        return {"n": 0, "gap": math.nan, "low": math.nan, "high": math.nan}
    per = [(int(c[rl][i]) - int(c[ref][i])) - (int(d[rl][i]) - int(d[ref][i])) for i in ids]
    rng, n, gaps = random.Random(seed), len(per), []
    for _ in range(draws):
        gaps.append(sum(per[rng.randrange(n)] for _ in range(n)) / n)
    gaps.sort()
    return {"n": n, "gap": sum(per) / n, "low": gaps[int(0.025 * draws)], "high": gaps[int(0.975 * draws) - 1]}


def build(n3: Path, exclude=(), runs: Path = RUNS, a=None, d_dirs=None, a_nohist=None) -> dict:
    n3 = Path(n3)
    d_dirs = d_dirs or [runs / "n/n1_eval_20261031", runs / "n/n1b_eval_20261102"]
    succ = {"a": tier_a(runs) if a is None else a,
            "a_nohist": tier_a_nohist(n3) if a_nohist is None else a_nohist,
            "c": episodes([n3 / "c"], LADDER, exclude),
            "d": episodes(d_dirs, R.MODELS, exclude),
            "e": episodes([n3 / "e"], R.MODELS, exclude),
            "f": episodes([n3 / "f"], LADDER, exclude)}
    c_ids = set().union(*[set(s) for s in succ["c"].values()]) if succ["c"] else set()
    succ["d_on_c"] = {m: {i: v for i, v in s.items() if i in c_ids} for m, s in succ["d"].items()}
    rep = {"tiers": {t: tier_table(s) for t, s in succ.items()}, "excluded": sorted(exclude), "across": {},
           "per_model": {"f_minus_d": {}, "a_nohist_minus_a": {}}}
    for m, ref in [(m, R.sft_of(m)) for m in RL] + list(RL_PAIRS):
        key = f"{m}-{ref}"
        row = {"c_vs_d_same_instances": paired_gap(succ["c"], succ["d"], m, ref)}
        if succ["f"]:
            row["f_vs_d_same_instances"] = paired_gap(succ["f"], succ["d"], m, ref)
        if succ["a_nohist"]:
            row["a_vs_a_nohist_same_tasks"] = paired_gap(succ["a"], succ["a_nohist"], m, ref)
        pa = rep["tiers"]["a"]["pairs"].get(key)
        for t in ("c", "d", "e"):
            pt = rep["tiers"][t]["pairs"].get(key)
            if pa and pt:
                row[f"a_vs_{t}"] = R.agree(pa, pt)
        pn, pc = rep["tiers"]["a_nohist"]["pairs"].get(key), rep["tiers"]["c"]["pairs"].get(key)
        if pn and pc:
            row["a_nohist_vs_c"] = R.agree(pn, pc)
        pd, pe = rep["tiers"]["d"]["pairs"].get(key), rep["tiers"]["e"]["pairs"].get(key)
        if pd and pe:
            row["e_vs_d"] = R.agree(pe, pd)
        rep["across"][key] = row
    for m in LADDER:
        for name, x, y in (("f_minus_d", "f", "d"), ("a_nohist_minus_a", "a_nohist", "a")):
            if m in succ[x] and m in succ[y]:
                ids = sorted(set(succ[x][m]) & set(succ[y][m]))
                rep["per_model"][name][m] = R.compare(succ[x][m], succ[y][m], ids)
    return rep


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}"


def diff_cell(c: dict) -> str:
    if not c or not c.get("n"):
        return "—"
    return (f"{pct(c.get('diff'))}（{pct(c.get('low'))} 到 {pct(c.get('high'))}；"
            f"{c.get('only_a')} 对 {c.get('only_b')}；p = {c.get('p', math.nan):.2g}）")


def gap_cell(g: dict) -> str:
    return f"{pct(g['gap'])}（{pct(g['low'])} 到 {pct(g['high'])}，{g['n']} 个）" if g and g.get("n") else "—"


def markdown(rep: dict) -> str:
    lines = ["# N3 迁移梯子（自动生成）", "",
             "差值都是百分点；括号里是 95% 区间、只有前者成功对只有后者成功的个数、精确 McNemar p。", ""]
    tiers = [t for t in SIDE_BY_SIDE if rep["tiers"].get(t, {}).get("models")]
    keys = [f"{m}-{R.sft_of(m)}" for m in RL] + [f"{m}-{r}" for m, r in RL_PAIRS]
    lines += ["## RL 减 SFT 和两种 RL 之间，各档并排（差，p）", "",
              "| 配对 | " + " | ".join(TIER_ZH[t] for t in tiers) + " |", "| --- |" + " ---: |" * len(tiers)]
    for k in keys:
        cells = []
        for t in tiers:
            c = rep["tiers"][t]["pairs"].get(k)
            cells.append(f"{pct(c['diff'])}（p = {c['p']:.2g}）" if c and c.get("n") else "—")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    for t, tab in rep["tiers"].items():
        if not tab["models"]:
            continue
        lines += [f"## {TIER_ZH[t]}", "", "| 模型 | 成功 / 总数 | 成功率（95% 区间） |", "| --- | ---: | ---: |"]
        for m, r in tab["models"].items():
            lines.append(f"| {m} | {r['k']}/{r['n']} | "
                         f"{100 * r['rate']:.1f}%（{100 * r['low']:.1f}–{100 * r['high']:.1f}） |")
        lines += ["", "| 配对 | 差（区间；只前者 对 只后者；p） |", "| --- | --- |"]
        for k, c in tab["pairs"].items():
            lines.append(f"| {k} | {diff_cell(c)} |")
        lines.append("")
    lines += ["## 跨档", "",
              "同一批实例或同一批题的两档，给两个配对差之差和按实例 / 按题重抽的区间；不同实例的两档，给独立样本 z 检验的 p。", "",
              "| 配对 | (c) 减 (d) | (f) 减 (d) | (a) 减 (a 不带历史) | (a) 对 (c) p | (a 不带历史) 对 (c) p | "
              "(a) 对 (d) p | (a) 对 (e) p | (e) 对 (d) p |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for k, row in rep["across"].items():
        cells = [gap_cell(row.get(x)) for x in ("c_vs_d_same_instances", "f_vs_d_same_instances",
                                                "a_vs_a_nohist_same_tasks")]
        cells += [f"{row[x]['p']:.2g}" if x in row and not math.isnan(row[x]["p"]) else "—"
                  for x in ("a_vs_c", "a_nohist_vs_c", "a_vs_d", "a_vs_e", "e_vs_d")]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    pm = rep.get("per_model", {})
    if any(pm.values()):
        lines += ["", "## 每个模型自己：(f) 减 (d)（同一批实例）、(a 不带历史) 减 (a)（同一批题）", "",
                  "| 模型 | (f) 减 (d) | (a 不带历史) 减 (a) |", "| --- | --- | --- |"]
        for m in LADDER:
            lines.append(f"| {m} | {diff_cell(pm.get('f_minus_d', {}).get(m))} | "
                         f"{diff_cell(pm.get('a_nohist_minus_a', {}).get(m))} |")
    if rep["excluded"]:
        lines += ["", f"- 画面损坏、各档都剔除的实例：{len(rep['excluded'])} 个。"]
    return "\n".join(lines) + "\n"


FIG_TIERS = (("a", "(a) upstream as is"), ("a_nohist", "(a) upstream, current turn only"),
             ("c", "(c) RoboCasa, upstream body"), ("d_on_c", "(d) N1, same instances as (c)"),
             ("f", "(f) N1 body, turns never blocked"), ("e", "(e) robot's own camera"))
FIG_PAIRS = (("RS-S", "RS − S", "#d9822b"), ("RD-S", "RD − S", "#2f9a6a"), ("Q3RS-Q3S", "Q3RS − Q3S", "#e8a65c"),
             ("Q3RD-Q3S", "Q3RD − Q3S", "#6cbf96"))


def figure(rep: dict, out: Path) -> list:
    """RL minus SFT per tier with its paired 95% interval (one row per tier, one marker per pairing)."""
    sys.path.insert(0, str(HERE.parent / "report"))
    import n_figures as NF

    plt = NF.plt_setup()
    fig, ax = plt.subplots(figsize=(6.6, 0.75 * len(FIG_TIERS) + 1.0))
    for k, (key, label, color) in enumerate(FIG_PAIRS):
        shown = False
        for row, (tier, _) in enumerate(FIG_TIERS):
            c = rep["tiers"].get(tier, {}).get("pairs", {}).get(key)
            if not c or not c.get("n"):
                continue
            y = row + (k - 1.5) * 0.17
            ax.errorbar([100 * c["diff"]], [y], xerr=[[100 * (c["diff"] - c["low"])], [100 * (c["high"] - c["diff"])]],
                        fmt="o", color=color, ms=4.5, elinewidth=1.1, capsize=2, label=None if shown else label)
            shown = True
    ax.axvline(0, color="#66707c", lw=0.8)
    ax.set_yticks(range(len(FIG_TIERS)))
    ax.set_yticklabels([lab for _, lab in FIG_TIERS])
    ax.invert_yaxis()
    ax.set_xlabel("RL − SFT success (percentage points, paired 95% interval)")
    ax.grid(axis="x", color="#e4e7eb", lw=0.7)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=4)
    fig.tight_layout()
    files = NF.save(fig, out, "n3_ladder")
    plt.close(fig)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n3", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--figure", action="store_true", help="also DIR/n3_ladder.png and .svg")
    args = ap.parse_args(argv)
    rep = build(args.n3, R.excluded(args.exclude))
    out = args.out or args.n3
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(markdown(rep), encoding="utf-8")
    if args.figure:
        figure(rep, out)
    print(json.dumps({t: {k: round(v.get("diff", math.nan), 4) for k, v in tab["pairs"].items()}
                      for t, tab in rep["tiers"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
