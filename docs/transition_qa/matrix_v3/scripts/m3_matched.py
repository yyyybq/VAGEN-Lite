"""Matrix v3, exploratory (not in the pre-registration): D - C with D taken at the same RL update as each C RL's
checkpoint of 3.1 (m3cs1s0 50, m3cs1d 0 = the SFT picks, m3cs2s0 100, m3cs2d 50), so the difference does not
include "how many updates ran"; and the test success of every evaluated checkpoint by update (0 = the SFT start).
Same statistics as 3.1 (m3_main31.py): pairs by seed and side, task by task; sign test pooled; 2,000 bootstrap draws.
Read only; writes OUT/matched.{json,md}.

    python m3_matched.py OUT_DIR
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_main31 as m  # noqa: E402

M, C0 = m.M, m.C0
D_MATCHED = {"1s0": M / "m3ds1s0_050", "1d": M / "sft_m3d_e9", "2s0": M / "m3ds2s0_100", "2d": M / "m3ds2d_050"}
STEP = {"1s0": 50, "1d": 0, "2s0": 100, "2d": 50}
START = {"C": M / "sft_m3c_e4", "D": M / "sft_m3d_e9", "E": C0 / "c0q3_sft_e8"}
RUNS = {"C": ["m3cs1s0", "m3cs1d", "m3cs2s0", "m3cs2d"], "D": ["m3ds1s0", "m3ds1d", "m3ds2s0", "m3ds2d"],
        "E": ["m3es1s0", "m3es1d", "m3es2s0", "m3es2d"]}


def compare(a: dict, b: dict, ids: list, draws: list) -> dict:
    d = {k: [int(a[k][t]) - int(b[k][t]) for t in ids] for k in a}
    diff = {k: 100 * sum(v) / len(ids) for k, v in d.items()}
    seed = {s: (diff[f"{s}s0"] + diff[f"{s}d"]) / 2 for s in "12"}
    pooled = sum(diff.values()) / 4
    only_a = sum(x == 1 for v in d.values() for x in v)
    only_b = sum(x == -1 for v in d.values() for x in v)
    boot = []
    for idx in draws:
        boot.append(sum(100 * sum(v[i] for i in idx) / len(idx) for v in d.values()) / 4)
    boot.sort()
    return {"pairs_pp": {k: round(v, 2) for k, v in diff.items()}, "seed_pp": {k: round(v, 2) for k, v in seed.items()},
            "pooled_pp": round(pooled, 2), "only_a": only_a, "only_b": only_b, "p": round(m.sign_test(only_a, only_b), 4),
            "ci95": [round(boot[int(0.025 * len(boot))], 2), round(boot[int(0.975 * len(boot)) - 1], 2)]}


def main() -> int:
    out = Path(sys.argv[1])
    c = {k: m.load(p) for k, p in m.MODELS["C"].items()}
    dm = {k: m.load(p) for k, p in D_MATCHED.items()}
    d300 = {k: m.load(p) for k, p in m.MODELS["D"].items()}
    ids = sorted(c["1s0"])
    rng = random.Random(m.SEED)
    draws = [[rng.randrange(len(ids)) for _ in ids] for _ in range(m.REPS)]
    rep = {"matched": compare(dm, c, ids, draws), "matched_models": {k: str(p) for k, p in D_MATCHED.items()},
           "step": STEP, "d300_vs_dmatched": compare(d300, dm, ids, draws)}
    curve = {}
    for g, runs in RUNS.items():
        curve[g] = {"0": sum(m.load(START[g]).values())}
        for r in runs:
            for p in sorted(M.glob(f"{r}_[0-9][0-9][0-9]")):
                curve[g].setdefault(str(int(p.name[-3:])), {})
                curve[g][str(int(p.name[-3:]))][r] = sum(m.load(p).values())
    rep["curve"] = curve
    (out / "matched.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    x = rep["matched"]
    lines = ["# 矩阵第三版：D 取和 C 相同的更新次数（探索性，不在预登记里）", "", __doc__.split("\n\n")[0], "",
             "| 配对 | 更新次数 | D − C |", "| --- | --- | --- |"]
    lines += [f"| 种子 {k[0]} {'S0' if k[1:] == 's0' else 'V4'} | {STEP[k]} | {v:+.1f} |" for k, v in x["pairs_pp"].items()]
    lines += ["", f"- 种子 1 {x['seed_pp']['1']:+.2f}，种子 2 {x['seed_pp']['2']:+.2f}；四个合起来 {x['pooled_pp']:+.2f}"
              f"（95% 区间 {x['ci95'][0]:+.1f} 到 {x['ci95'][1]:+.1f}）；只有 D 做对 {x['only_a']}，只有 C 做对 {x['only_b']}，"
              f"p = {x['p']:.3f}",
              f"- D 第 300 次对这几个 D 检查点：四个合起来 {rep['d300_vs_dmatched']['pooled_pp']:+.2f}"
              f"（{rep['d300_vs_dmatched']['ci95'][0]:+.1f} 到 {rep['d300_vs_dmatched']['ci95'][1]:+.1f}），"
              f"p = {rep['d300_vs_dmatched']['p']:.3f}", "", "## 各检查点的测试集成功题数（292 道）", "",
             "| 组 | 更新次数 | 各 RL |", "| --- | --- | --- |"]
    for g, cv in curve.items():
        for s in sorted(cv, key=int):
            v = cv[s]
            lines.append(f"| {g} | {s} | " + ("SFT 起点 " + str(v) if s == "0" else "，".join(f"{r} {n}" for r, n in v.items())) + " |")
    (out / "matched.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
