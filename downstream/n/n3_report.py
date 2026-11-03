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
Per tier and model: successes / episodes with the 95% Wilson interval; per pairing the paired difference, its 95%
interval and McNemar's exact p (n_report.compare).  Across tiers: c against d on the same instances, the difference of
the two paired differences with a 95% interval from resampling instances (DRAWS, fixed seed); a against c, d and e,
independent sets, a z test of the gap (n_report.agree).  --exclude: n_frames.py scan reports, every instance with a
corrupted frame in any evaluation of the tier left out (all models alike).  Writes DIR/report.json and report.md
(DIR defaults to N3).
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
TIER_ZH = {"a": "(a) 上游原样", "c": "(c) RoboCasa + 上游身体", "d": "(d) N1 原样", "d_on_c": "(d) 只取 (c) 的实例",
           "e": "(e) 机器人自己的相机"}
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
    for m in RL:
        ref = R.sft_of(m)
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


def build(n3: Path, exclude=(), runs: Path = RUNS, a=None, d_dirs=None) -> dict:
    n3 = Path(n3)
    d_dirs = d_dirs or [runs / "n/n1_eval_20261031", runs / "n/n1b_eval_20261102"]
    succ = {"a": tier_a(runs) if a is None else a,
            "c": episodes([n3 / "c"], LADDER, exclude),
            "d": episodes(d_dirs, R.MODELS, exclude),
            "e": episodes([n3 / "e"], R.MODELS, exclude)}
    c_ids = set().union(*[set(s) for s in succ["c"].values()]) if succ["c"] else set()
    succ["d_on_c"] = {m: {i: v for i, v in s.items() if i in c_ids} for m, s in succ["d"].items()}
    rep = {"tiers": {t: tier_table(s) for t, s in succ.items()}, "excluded": sorted(exclude), "across": {}}
    for m in RL:
        ref = R.sft_of(m)
        key = f"{m}-{ref}"
        row = {"c_vs_d_same_instances": paired_gap(succ["c"], succ["d"], m, ref)}
        pa = rep["tiers"]["a"]["pairs"].get(key)
        for t in ("c", "d", "e"):
            pt = rep["tiers"][t]["pairs"].get(key)
            if pa and pt:
                row[f"a_vs_{t}"] = R.agree(pa, pt)
        pd, pe = rep["tiers"]["d"]["pairs"].get(key), rep["tiers"]["e"]["pairs"].get(key)
        if pd and pe:
            row["e_vs_d"] = R.agree(pe, pd)
        rep["across"][key] = row
    return rep


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}"


def markdown(rep: dict) -> str:
    lines = ["# N3 迁移梯子（自动生成）", "",
             "差值都是百分点；括号里是 95% 区间、只有前者成功对只有后者成功的个数、精确 McNemar p。", ""]
    for t, tab in rep["tiers"].items():
        lines += [f"## {TIER_ZH[t]}", "", "| 模型 | 成功 / 总数 | 成功率（95% 区间） |", "| --- | ---: | ---: |"]
        for m, r in tab["models"].items():
            lines.append(f"| {m} | {r['k']}/{r['n']} | "
                         f"{100 * r['rate']:.1f}%（{100 * r['low']:.1f}–{100 * r['high']:.1f}） |")
        lines += ["", "| 配对 | 差（区间；只前者 对 只后者；p） |", "| --- | --- |"]
        for k, c in tab["pairs"].items():
            lines.append(f"| {k} | {pct(c.get('diff'))}（{pct(c.get('low'))} 到 {pct(c.get('high'))}；"
                         f"{c.get('only_a')} 对 {c.get('only_b')}；p = {c.get('p', math.nan):.2g}） |")
        lines.append("")
    lines += ["## 跨档", "",
              "| 配对 | (c) 减 (d)，同一批实例（区间） | (a) 减 (c) p | (a) 减 (d) p | (a) 减 (e) p | (e) 减 (d) p |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for k, row in rep["across"].items():
        g = row["c_vs_d_same_instances"]
        cells = [f"{pct(g['gap'])}（{pct(g['low'])} 到 {pct(g['high'])}，{g['n']} 个）"]
        cells += [f"{row[x]['p']:.2g}" if x in row and not math.isnan(row[x]["p"]) else "—"
                  for x in ("a_vs_c", "a_vs_d", "a_vs_e", "e_vs_d")]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if rep["excluded"]:
        lines += ["", f"- 画面损坏、各档都剔除的实例：{len(rep['excluded'])} 个。"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n3", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rep = build(args.n3, R.excluded(args.exclude))
    out = args.out or args.n3
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(markdown(rep), encoding="utf-8")
    print(json.dumps({t: {k: round(v.get("diff", math.nan), 4) for k, v in tab["pairs"].items()}
                      for t, tab in rep["tiers"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
