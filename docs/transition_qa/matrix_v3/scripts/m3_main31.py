"""Matrix v3, pre-registration 3.1 (main metric): action success on eval_test_v2_screened (292 tasks) at each RL's
pre-registered checkpoint (section 2: update 300; a stopped RL its last checkpoint; m3cs1d, stopped at 47 before its
first checkpoint, uses C's picked SFT: the user, 2026-11-03, "用 SFT 起点").  Pairs: same seed, same side, task by
task (matched by task_id).  Pooled = the four pairs together (4 x 292); each seed's difference = the mean of its S0 and
V4 pairs.  Test: exact two-sided sign test on the tasks only one side solved, pooled; 95% CI from 2,000 bootstrap
draws of the 292 tasks (the same draws for every pair).  Verdict: pooled p < 0.05 and pooled difference >= 5 pp and
the two seeds' differences of the same sign (both > 0 or both < 0).  D - C is the main comparison, D - C0 secondary,
E - C0 exploratory (pre-registration note 4).  Read only; writes OUT.

    python m3_main31.py OUT_DIR
"""
import json
import math
import random
import sys
from pathlib import Path

M = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/eval_test_v2_screened")
C0 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/matrix_inputs_20261029/eval_test_v2_screened")
REPS, SEED, ALPHA, MIN_PP = 2000, 20261103, 0.05, 5.0
PAIRS = [("1", "s0"), ("1", "d"), ("2", "s0"), ("2", "d")]
MODELS = {
    "C": {"1s0": M / "m3cs1s0_050", "1d": M / "sft_m3c_e4", "2s0": M / "m3cs2s0_100", "2d": M / "m3cs2d_050"},
    "D": {"1s0": M / "m3ds1s0_300", "1d": M / "m3ds1d_300", "2s0": M / "m3ds2s0_300", "2d": M / "m3ds2d_300"},
    "E": {"1s0": M / "m3es1s0_300", "1d": M / "m3es1d_200", "2s0": M / "m3es2s0_200", "2d": M / "m3es2d_300"},
    "C0": {"1s0": C0 / "c0q3_s1s0_150", "1d": C0 / "c0q3_s1d_300", "2s0": C0 / "c0q3_s2s0_300", "2d": C0 / "c0q3_s2d_300"},
}
COMPARE = [("D", "C", "main"), ("D", "C0", "secondary"), ("E", "C0", "exploratory")]


def sign_test(b: int, c: int) -> float:  # as data_gen.interiorgs_v10.evaluation.sign_test
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(d: Path) -> dict:
    rows = [json.loads(l) for l in open(d / "results.jsonl", encoding="utf-8") if l.strip()]
    assert not any("error" in r for r in rows), d
    out = {r["task_id"]: bool(r["success"]) for r in rows}
    assert len(out) == len(rows) == 292, (d, len(out), len(rows))
    return out


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {g: {k: load(p) for k, p in m.items()} for g, m in MODELS.items()}
    ids = sorted(res["D"]["1s0"])
    assert all(sorted(v) == ids for m in res.values() for v in m.values())
    rng = random.Random(SEED)
    draws = [[rng.randrange(len(ids)) for _ in ids] for _ in range(REPS)]
    report = {"models": {g: {k: str(p) for k, p in m.items()} for g, m in MODELS.items()},
              "success": {g: {k: sum(v.values()) for k, v in m.items()} for g, m in res.items()},
              "reps": REPS, "seed": SEED, "compare": {}}
    lines = ["# 矩阵第三版：主指标（预登记 3.1）", "", __doc__.split("\n\n")[0], ""]
    for a, b, kind in COMPARE:
        d = {f"{s}{side}": [int(res[a][f"{s}{side}"][t]) - int(res[b][f"{s}{side}"][t]) for t in ids] for s, side in PAIRS}
        diff = {k: 100 * sum(v) / len(ids) for k, v in d.items()}
        seed = {s: (diff[f"{s}s0"] + diff[f"{s}d"]) / 2 for s in "12"}
        pooled = sum(diff.values()) / 4
        only_a = sum(x == 1 for v in d.values() for x in v)
        only_b = sum(x == -1 for v in d.values() for x in v)
        p = sign_test(only_a, only_b)
        boot = {"pooled": [], "1": [], "2": []}
        for idx in draws:
            pd = {k: 100 * sum(v[i] for i in idx) / len(idx) for k, v in d.items()}
            boot["pooled"].append(sum(pd.values()) / 4)
            for s in "12":
                boot[s].append((pd[f"{s}s0"] + pd[f"{s}d"]) / 2)
        ci = {k: [round(sorted(v)[int(0.025 * REPS)], 2), round(sorted(v)[int(0.975 * REPS) - 1], 2)] for k, v in boot.items()}
        per_pair = {k: {"diff_pp": round(diff[k], 2), "only_" + a: sum(x == 1 for x in v), "only_" + b: sum(x == -1 for x in v),
                        "p": round(sign_test(sum(x == 1 for x in v), sum(x == -1 for x in v)), 4)} for k, v in d.items()}
        same = (seed["1"] > 0 and seed["2"] > 0) or (seed["1"] < 0 and seed["2"] < 0)
        verdict = p < ALPHA and pooled >= MIN_PP and same if pooled >= 0 else p < ALPHA and -pooled >= MIN_PP and same
        rec = {"kind": kind, "pairs": per_pair, "seed_diff_pp": {k: round(v, 2) for k, v in seed.items()},
               "pooled_diff_pp": round(pooled, 2), "only_" + a: only_a, "only_" + b: only_b, "p": round(p, 4),
               "ci95": ci, "same_sign": same, "difference": bool(verdict)}
        report["compare"][f"{a}-{b}"] = rec
        lines += [f"## {a} − {b}（{ {'main': '主比较', 'secondary': '次要比较', 'exploratory': '探索性'}[kind] }）", "",
                  "| 配对 | 差（百分点） | 只有 " + a + " 做对 | 只有 " + b + " 做对 | p |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| 种子 {k[0]} {'S0' if k[1:] == 's0' else 'V4'} | {v['diff_pp']:+.1f} | {v['only_' + a]} | {v['only_' + b]} | {v['p']:.3f} |"
                  for k, v in per_pair.items()]
        lines += ["", f"- 种子 1：{seed['1']:+.2f}（95% 区间 {ci['1'][0]:+.1f} 到 {ci['1'][1]:+.1f}）；"
                  f"种子 2：{seed['2']:+.2f}（{ci['2'][0]:+.1f} 到 {ci['2'][1]:+.1f}）；同号：{'是' if same else '否'}",
                  f"- 四个合起来：{pooled:+.2f} 个百分点（{ci['pooled'][0]:+.1f} 到 {ci['pooled'][1]:+.1f}）；"
                  f"只有 {a} 做对 {only_a}，只有 {b} 做对 {only_b}，符号检验 p = {p:.3f}",
                  f"- 判定：{'有差异' if verdict else '没有差异'}（p < 0.05：{'是' if p < ALPHA else '否'}；"
                  f"差 ≥ 5：{'是' if abs(pooled) >= MIN_PP else '否'}；同号：{'是' if same else '否'}）", ""]
    (out_dir / "main_31.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (out_dir / "main_31.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
