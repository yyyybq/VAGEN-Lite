"""Matrix v3, the rest of the pre-registration at the checkpoints of m3_main31 (section 2; note 5: m3cs1d counts as C's
picked SFT): 3.1 on the validation set (reference); 3.2 by task family (Holm over the 6 families x 2 comparisons D - C,
D - C0; E - C0 its own 6, exploratory; the occlusion family exploratory, note 1); 3.3 the probe (per-model table and
qa_probe.compare, the same function as `qa_probe compare`); 3.4 the failure breakdown (failure_breakdown.classify:
(1) the relation share of the one-condition-short failures, two-proportion z test pooled over the four RLs; (2) the
share of all tasks that are relation one-short failures, sign test task by task); the SFT-stage comparison (section 2:
secondary, one pair each, uncorrected); section 5's stability table.  Pairs: same seed, same side, task by task.
Read only; writes OUT/rest.{json,md} and OUT/probe/compare_*.{json,md}.

    python m3_prereg_rest.py OUT_DIR
"""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
sys.path.insert(0, str(ROOT))
from eval.diagnostics import qa_probe  # noqa: E402
from eval.diagnostics.failure_breakdown import FAMILY_ORDER, GATE_GROUP, classify  # noqa: E402

M3 = ROOT / "exps/matrix_v3"
DATA = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps")
SETS = {"test": (M3 / "eval_test_v2_screened", DATA / "matrix_inputs_20261029/eval_test_v2_screened", 292),
        "val": (M3 / "eval_val_v2_screened", DATA / "matrix_inputs_20261029/eval_val_v2_screened", 382)}
PROBE = DATA / "transition_qa_20261019b/p1_probe.jsonl"
PROBE_DIRS = [M3 / f"m3_{g}_s{s}/diag/probe_P1" for g in "cde" for s in (20261019, 20261021)] + \
    [DATA / "diagnostics_20261101/probe_v3g_q3"]
REPS, SEED, ALPHA = 2000, 20261103, 0.05
PAIR_ZH = ["种子 1 S0", "种子 1 V4", "种子 2 S0", "种子 2 V4"]
SEEDS = ("20261019", "20261021")
RL = {"C": ["m3cs1s0_050", "sft_m3c_e4", "m3cs2s0_100", "m3cs2d_050"],
      "D": ["m3ds1s0_300", "m3ds1d_300", "m3ds2s0_300", "m3ds2d_300"],
      "E": ["m3es1s0_300", "m3es1d_200", "m3es2s0_200", "m3es2d_300"],
      "C0": ["c0q3_s1s0_150", "c0q3_s1d_300", "c0q3_s2s0_300", "c0q3_s2d_300"]}
SFT = {"C": "sft_m3c_e4", "D": "sft_m3d_e9", "C0": "c0q3_sft_e8"}
COMPARE = [("D", "C", "main"), ("D", "C0", "secondary"), ("E", "C0", "exploratory")]
KIND_ZH = {"main": "主比较", "secondary": "次要比较", "exploratory": "探索性"}
OCC = "occlusion_alignment"
FAM_ZH = {"projective_relations": "投影", "absolute_positioning": "绝对距离", "screen_occupancy": "画面占比",
          "equidistance": "等距", "occlusion_alignment": "遮挡（探索性）", "fov_inclusion": "视野"}
# section 5: PROGRESS 12.35 (the chains' state and logs) and the model line's
# chains/m_qwen3vl8b_formal_rl_20261026/PROGRESS.md (C0 seed 1 S0)
STABILITY = [
    ("C", "m3cs1s0", "96", "熵线（第 92–96 次都高于 0.3）；之前有过一次图像 token 崩溃，从第 50 次接着训", "50"),
    ("C", "m3cs1d", "47", "熵线（第 43–47 次）", "没有（用 C 的 SFT，说明第 5 条）"),
    ("C", "m3cs2s0", "101", "图像 token 崩溃连续 2 次", "100"),
    ("C", "m3cs2d", "50", "图像 token 崩溃连续 2 次", "50"),
    ("D", "m3ds1s0", "300", "跑满", "300"), ("D", "m3ds1d", "300", "跑满", "300"),
    ("D", "m3ds2s0", "300", "跑满", "300"), ("D", "m3ds2d", "300", "跑满", "300"),
    ("E", "m3es1s0", "300", "跑满", "300"), ("E", "m3es1d", "208", "熵线（第 204–208 次）", "200"),
    ("E", "m3es2s0", "227", "熵线（第 223–227 次）", "200"), ("E", "m3es2d", "300", "跑满", "300"),
    ("C0", "种子 1 S0（模型线 `q3s0`）", "150", "读不出线：第 141–150 次平均 10.6%，超过 10%", "150"),
    ("C0", "种子 1 V4（模型线 `q3d`）", "300", "跑满", "300"), ("C0", "种子 2 S0", "300", "跑满", "300"),
    ("C0", "种子 2 V4", "300", "跑满", "300")]


def sign_test(b: int, c: int) -> float:  # as data_gen.interiorgs_v10.evaluation.sign_test
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps: list) -> list:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def fp(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def load(folder: Path, n: int) -> dict:
    rows = [json.loads(l) for l in open(folder / "results.jsonl", encoding="utf-8") if l.strip()]
    assert not any("error" in r for r in rows), folder
    out = {}
    for r in rows:
        c = classify(r)
        one = c["class"] == "budget_one_gate"
        out[r["task_id"]] = {"family": r["task_type"], "success": bool(r["success"]), "one": one,
                             "rel": one and GATE_GROUP.get(c["gate"], "other") == "relation",
                             "stopped": c["class"] == "stopped"}
    assert len(out) == len(rows) == n, (folder, len(out), len(rows))
    return out


DRAWS: dict = {}


def paired(xa: list, xb: list, ids: list, key: str, field: str) -> dict:
    """A - B over 4 pairs (or 1): per pair the difference in pp of the tasks, per seed the mean of its two pairs,
    pooled the mean of all; sign test on the (task, pair)s only one side has; 95% CI from REPS bootstrap draws of the
    tasks (the same draws for every comparison on these tasks)."""
    n = len(ids)
    d = [[int(a[t][field]) - int(b[t][field]) for t in ids] for a, b in zip(xa, xb)]
    diff = [100 * sum(v) / n for v in d]
    oa, ob = [sum(x == 1 for x in v) for v in d], [sum(x == -1 for x in v) for v in d]
    seeds = [(diff[0] + diff[1]) / 2, (diff[2] + diff[3]) / 2] if len(d) == 4 else []
    if key not in DRAWS:
        rng = random.Random(f"{SEED}/{key}")
        DRAWS[key] = [[rng.randrange(n) for _ in range(n)] for _ in range(REPS)]
    boot = sorted(sum(100 * sum(v[i] for i in idx) / n for v in d) / len(d) for idx in DRAWS[key])
    pooled = sum(diff) / len(diff)
    same = len(seeds) == 2 and ((seeds[0] > 0 and seeds[1] > 0) or (seeds[0] < 0 and seeds[1] < 0))
    return {"n": n, "pairs": [{"diff_pp": round(x, 2), "only_a": a, "only_b": b, "p": round(sign_test(a, b), 4)}
                              for x, a, b in zip(diff, oa, ob)],
            "seed_diff_pp": [round(x, 2) for x in seeds], "pooled_diff_pp": round(pooled, 2), "only_a": sum(oa),
            "only_b": sum(ob), "p": sign_test(sum(oa), sum(ob)),
            "ci95": [round(boot[int(0.025 * REPS)], 2), round(boot[int(0.975 * REPS) - 1], 2)], "same_sign": same}


def rate(models: list, ids: list, field: str) -> str:
    k = sum(m[t][field] for m in models for t in ids)
    return f"{100 * k / (len(models) * len(ids)):.1f}%（{k} / {len(models) * len(ids)}）"


def ztest(ra: int, na: int, rb: int, nb: int) -> tuple:
    pa, pb, pp = ra / na, rb / nb, (ra + rb) / (na + nb)
    se = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    z = (pa - pb) / se if se else 0.0
    return 100 * (pa - pb), z, math.erfc(abs(z) / math.sqrt(2))


def probe_row(name: str, probe: dict) -> dict:
    path = next(d / f"{name}.jsonl" for d in PROBE_DIRS if (d / f"{name}.jsonl").exists())
    rows = {r["sample_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}
    assert set(rows) == set(probe), (name, len(rows))

    def acc(ids: list) -> float:
        return sum(rows[s]["correct"] for s in ids) / len(ids)

    by = lambda q, v: [s for s, r in probe.items() if r["qtype"] in q and r.get("answer_value") == v]  # noqa: E731
    occ = {v: acc(by(qa_probe.OCC_TYPES, v)) for v in qa_probe.ANSWER_VALUES["occlusion"]}
    out = {"path": str(path), "all": acc(list(probe)), "occ_balanced": sum(occ.values()) / len(occ), "occ": occ,
           "unreadable": sum(not rows[s]["readable"] for s in probe) / len(probe)}
    for q in qa_probe.GOAL_TYPES:
        out[q] = acc([s for s, r in probe.items() if r["qtype"] == q])
        out[q + "_by"] = {v: acc(by((q,), v)) for v in qa_probe.ANSWER_VALUES["goal"]}
    return out


def probe_name(name: str) -> str:
    return str(next(d / name for d in PROBE_DIRS if (d / f"{name}.jsonl").exists()))


def main() -> int:
    out_dir = Path(sys.argv[1])
    (out_dir / "probe").mkdir(parents=True, exist_ok=True)
    rep: dict = {"rules": {"reps": REPS, "seed": SEED, "alpha": ALPHA}, "sets": {}}
    md = ["# 矩阵第三版：预登记 3.1（验证集）、3.2、3.3、3.4、SFT 阶段、第 5 节", "", __doc__.split("\n\n")[0], ""]
    res = {}
    for s, (m3, c0, n) in SETS.items():
        res[s] = {g: [load((c0 if x.startswith("c0q3") else m3) / x, n) for x in xs] for g, xs in RL.items()}
        res[s]["sft"] = {g: load((c0 if x.startswith("c0q3") else m3) / x, n) for g, x in SFT.items()}
    # the C0 numbers the pre-registration's 3.4 row gives (test): SFT 51 / 103, the four RLs 41/76, 53/105, 53/105, 58/108
    chk = [(sum(m[t]["rel"] for t in m), sum(m[t]["one"] for t in m)) for m in [res["test"]["sft"]["C0"]] + res["test"]["C0"]]
    assert chk == [(51, 103), (41, 76), (53, 105), (53, 105), (58, 108)], chk

    for s in SETS:
        ids = sorted(res[s]["D"][0])
        assert all(sorted(m) == ids for g in RL for m in res[s][g])
        fam = {t: res[s]["D"][0][t]["family"] for t in ids}
        assert all(m[t]["family"] == fam[t] for g in RL for m in res[s][g] for t in ids)
        R = rep["sets"][s] = {}
        # 3.1 (the test set's is m3_main31; here the validation set, reference)
        if s == "val":
            md += ["## 3.1 验证集（382 道，参考）", "",
                   "| 比较 | " + " | ".join(PAIR_ZH) + " | 种子 1 | 种子 2 | 四个合起来（95% 区间） | 只有 A 对 / 只有 B 对 | 符号检验 p | 按 3.1 的规则 |",
                   "| --- | " + " | ".join("---" for _ in PAIR_ZH) + " | --- | --- | --- | --- | --- | --- |"]
            R["3.1"] = {}
            for a, b, kind in COMPARE:
                r = R["3.1"][f"{a}-{b}"] = paired(res[s][a], res[s][b], ids, f"{s}/all", "success")
                ok = r["p"] < ALPHA and abs(r["pooled_diff_pp"]) >= 5 and r["same_sign"]
                r["difference"] = ok
                md.append(f"| {a} − {b}（{KIND_ZH[kind]}） | " + " | ".join(f"{x['diff_pp']:+.1f}" for x in r["pairs"])
                          + f" | {r['seed_diff_pp'][0]:+.2f} | {r['seed_diff_pp'][1]:+.2f} | {r['pooled_diff_pp']:+.2f}"
                          f"（{r['ci95'][0]:+.1f} 到 {r['ci95'][1]:+.1f}） | {r['only_a']} / {r['only_b']} | {fp(r['p'])} | "
                          f"{'有差异' if ok else '没有差异'} |")
            md += ["", "四组的成功率（4 个 RL 合起来）：" + "，".join(f"{g} {rate(res[s][g], ids, 'success')}" for g in RL)
                   + "。", ""]
        # 3.2
        R["3.2"] = {}
        md += [f"## 3.2 按任务族（{'测试集 292 道' if s == 'test' else '验证集 382 道，参考'}）", "",
               "差的单位是百分点；成功率是 4 个 RL 合起来（\"题 × RL\"）。Holm 校正：D − C、D − C0 共 12 个检验一起校正；"
               "E − C0 的 6 个单独校正。判定\"有差异\"：校正后 p < 0.05，且 |差| ≥ 10，且两个种子的差同号。", ""]
        for group in (COMPARE[:2], COMPARE[2:]):
            recs = []
            for a, b, kind in group:
                for f in FAMILY_ORDER:
                    fids = [t for t in ids if fam[t] == f]
                    r = paired(res[s][a], res[s][b], fids, f"{s}/{f}", "success")
                    recs.append((a, b, kind, f, fids, r))
            for (a, b, kind, f, fids, r), adj in zip(recs, holm([r["p"] for *_, r in recs])):
                r["p_holm"] = adj
                r["difference"] = adj < ALPHA and abs(r["pooled_diff_pp"]) >= 10 and r["same_sign"]
                r["exploratory"] = f == OCC or kind == "exploratory"
                R["3.2"][f"{a}-{b}/{f}"] = r
            for a, b, kind in group:
                md += [f"**{a} − {b}（{KIND_ZH[kind]}）**", "",
                       f"| 任务族 | 题数 | {a} | {b} | 种子 1 | 种子 2 | 四个合起来（95% 区间） | 只有 {a} 对 / 只有 {b} 对 | p | Holm 校正后 p | 判定 |",
                       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
                for f in FAMILY_ORDER:
                    r = R["3.2"][f"{a}-{b}/{f}"]
                    fids = [t for t in ids if fam[t] == f]
                    v = "有差异" if r["difference"] else "没有差异"
                    md.append(f"| {FAM_ZH[f]} | {len(fids)} | {rate(res[s][a], fids, 'success')} | {rate(res[s][b], fids, 'success')} | "
                              f"{r['seed_diff_pp'][0]:+.1f} | {r['seed_diff_pp'][1]:+.1f} | {r['pooled_diff_pp']:+.1f}"
                              f"（{r['ci95'][0]:+.1f} 到 {r['ci95'][1]:+.1f}） | {r['only_a']} / {r['only_b']} | {fp(r['p'])} | "
                              f"{fp(r['p_holm'])} | {v + '（探索性）' if r['exploratory'] else v} |")
                md.append("")
        # 3.4
        R["3.4"] = {"models": {}, "compare": {}}
        md += [f"## 3.4 失败拆解（{'测试集' if s == 'test' else '验证集，参考'}）", "",
               "\"只差一个\"：12 个动作用完、最好的回合只差一个条件的题。①差\"关系\"的占只差一个的比例；②差\"关系\"的只差一个的失败占全部题的比例。", "",
               "| 组 | 模型 | 成功 | 只差一个的失败 | 其中差\"关系\" | ① | ② | 没到就停了 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g in RL:
            for name, m in ([(SFT[g] + "（SFT）", res[s]["sft"][g])] if g in SFT else []) + list(zip(RL[g], res[s][g])):
                one, rel = sum(m[t]["one"] for t in ids), sum(m[t]["rel"] for t in ids)
                R["3.4"]["models"][f"{g}/{name}"] = {"one": one, "rel": rel, "n": len(ids)}
                md.append(f"| {g} | `{name}` | {sum(m[t]['success'] for t in ids)} | {one} | {rel} | "
                          f"{100 * rel / one:.0f}% | {100 * rel / len(ids):.1f}% | {sum(m[t]['stopped'] for t in ids)} |")
        md += ["", "| 比较 | ① A（4 个合起来） | ① B | ① 差 | z 检验 p | ① 判定\"下降\" | ② 种子 1 | ② 种子 2 | ② 四个合起来（95% 区间） | "
               "只有 A / 只有 B | 符号检验 p | ② 判定\"下降\" |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for a, b, kind in COMPARE:
            ra, na = (sum(m[t][k] for m in res[s][a] for t in ids) for k in ("rel", "one"))
            rb, nb = (sum(m[t][k] for m in res[s][b] for t in ids) for k in ("rel", "one"))
            d1, z, p1 = ztest(ra, na, rb, nb)
            r2 = paired(res[s][a], res[s][b], ids, f"{s}/all", "rel")
            down1 = p1 < ALPHA and d1 <= -10
            down2 = r2["p"] < ALPHA and all(x < 0 for x in r2["seed_diff_pp"])
            R["3.4"]["compare"][f"{a}-{b}"] = {"kind": kind, "share_a": [ra, na], "share_b": [rb, nb], "diff_pp": round(d1, 2),
                                               "z": round(z, 3), "p": p1, "down": down1, "all_tasks": r2, "all_tasks_down": down2}
            md.append(f"| {a} − {b}（{KIND_ZH[kind]}） | {100 * ra / na:.1f}%（{ra} / {na}） | {100 * rb / nb:.1f}%（{rb} / {nb}） | "
                      f"{d1:+.1f} | {fp(p1)} | {'是' if down1 else '否'} | {r2['seed_diff_pp'][0]:+.2f} | {r2['seed_diff_pp'][1]:+.2f} | "
                      f"{r2['pooled_diff_pp']:+.2f}（{r2['ci95'][0]:+.1f} 到 {r2['ci95'][1]:+.1f}） | {r2['only_a']} / {r2['only_b']} | "
                      f"{fp(r2['p'])} | {'是' if down2 else '否'} |")
        md.append("")
        # SFT stage (secondary, uncorrected)
        R["sft"] = {}
        sft = res[s]["sft"]
        md += [f"## SFT 阶段（{'测试集' if s == 'test' else '验证集，参考'}；次要，每组一个模型，不校正）", "",
               f"D 第 9 个 epoch（`{SFT['D']}`）、C 第 4 个（`{SFT['C']}`）、C0 第 8 个（`{SFT['C0']}`）。差的单位是百分点，p 是逐题符号检验（①是 z 检验）。", "",
               "| 指标 | 题数 | D | C | C0 | D − C（p） | D − C0（p） |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for f in ("all",) + FAMILY_ORDER:
            fids = ids if f == "all" else [t for t in ids if fam[t] == f]
            cells = []
            for b in ("C", "C0"):
                r = R["sft"][f"D-{b}/{f}"] = paired([sft["D"]], [sft[b]], fids, f"{s}/{f}", "success")
                cells.append(f"{r['pooled_diff_pp']:+.1f}（{fp(r['p'])}）")
            md.append(f"| 成功率：{'合计' if f == 'all' else FAM_ZH[f]} | {len(fids)} | " + " | ".join(
                rate([sft[g]], fids, "success") for g in ("D", "C", "C0")) + " | " + " | ".join(cells) + " |")
        cells, cells2 = [], []
        for b in ("C", "C0"):
            ra, na = sum(sft["D"][t]["rel"] for t in ids), sum(sft["D"][t]["one"] for t in ids)
            rb, nb = sum(sft[b][t]["rel"] for t in ids), sum(sft[b][t]["one"] for t in ids)
            d1, z, p1 = ztest(ra, na, rb, nb)
            r2 = R["sft"][f"D-{b}/rel"] = paired([sft["D"]], [sft[b]], ids, f"{s}/all", "rel")
            R["sft"][f"D-{b}/share"] = {"diff_pp": round(d1, 2), "p": p1}
            cells.append(f"{d1:+.1f}（{fp(p1)}）")
            cells2.append(f"{r2['pooled_diff_pp']:+.1f}（{fp(r2['p'])}）")
        md += [f"| 3.4 ① | – | " + " | ".join(f"{100 * sum(sft[g][t]['rel'] for t in ids) / sum(sft[g][t]['one'] for t in ids):.0f}%"
                                              for g in ("D", "C", "C0")) + " | " + " | ".join(cells) + " |",
               f"| 3.4 ② | {len(ids)} | " + " | ".join(rate([sft[g]], ids, "rel") for g in ("D", "C", "C0")) + " | "
               + " | ".join(cells2) + " |", ""]

    # 3.3 probe
    probe = {r["sample_id"]: r for r in (json.loads(l) for l in PROBE.read_text(encoding="utf-8").splitlines() if l.strip())}
    assert len(probe) == 838
    P = rep["probe"] = {"models": {}, "compare": {}}
    md += ["## 3.3 探针（838 道）", "", "| 组 | 模型 | 全部 838 道 | `goal_now`（Yes / No） | `goal_after`（Yes / No） | 遮挡平衡准确率（完全 / 部分 / 没有） | 读不出 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    for g in RL:
        for name in ([SFT[g]] if g in SFT else []) + RL[g]:
            if name in P["models"]:
                continue
            r = P["models"][name] = probe_row(name, probe)
            md.append(f"| {g} | `{name}` | {pct(r['all'])} | " + " | ".join(
                f"{pct(r[q])}（{pct(r[q + '_by']['yes'])} / {pct(r[q + '_by']['no'])}）" for q in qa_probe.GOAL_TYPES)
                + f" | {pct(r['occ_balanced'])}（" + " / ".join(pct(r["occ"][v]) for v in ("full", "partial", "none"))
                + f"） | {pct(r['unreadable'])}% |")
    md += ["", "比较用 `qa_probe.compare`（与 `python -m eval.diagnostics.qa_probe compare` 相同），结果在 `probe/compare_*.md`。"
           "goal 两类判定：四个合起来 McNemar p < 0.05，且 |差| ≥ 10，且两个种子同号；遮挡：|差| ≥ 20 且两个种子同号。", "",
           "| 比较 | 题类 | 种子 1 | 种子 2 | 四个合起来 | p | 95% 区间 | 判定 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    groups = {g: (g, [(SEEDS[i], [probe_name(RL[g][2 * i]), probe_name(RL[g][2 * i + 1])]) for i in range(2)]) for g in RL}
    jobs = [(f"{a}_vs_{b}", groups[a], groups[b], f"{a} − {b}（{KIND_ZH[k]}）") for a, b, k in COMPARE] + \
        [(f"{a}_SFT_vs_{b}_SFT", (f"{a}_SFT", [probe_name(SFT[a])]), (f"{b}_SFT", [probe_name(SFT[b])]), f"SFT：{a} − {b}（次要）")
         for a, b in (("D", "C"), ("D", "C0"))]
    for label, ga, gb, title in jobs:
        r = qa_probe.compare(PROBE, out_dir / "probe", ga, gb, label)
        (out_dir / "probe" / f"compare_{label}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        (out_dir / "probe" / f"compare_{label}.md").write_text(qa_probe.compare_markdown(r), encoding="utf-8")
        P["compare"][label] = {q: r[q]["pooled"] | {"per_seed": [x["diff"] for x in r[q]["per_seed"]]}
                               for q in qa_probe.GOAL_TYPES + ("occlusion",)}
        for q in qa_probe.GOAL_TYPES + ("occlusion",):
            pl, ps = r[q]["pooled"], [x["diff"] for x in r[q]["per_seed"]]
            md.append(f"| {title} | {'遮挡（平衡）' if q == 'occlusion' else '`' + q + '`'} | "
                      + (" | ".join(f"{100 * x:+.1f}" for x in ps) if ps else "– | –") + f" | {100 * pl['diff']:+.1f} | {fp(pl['p'])} | "
                      + (f"{100 * pl['ci'][0]:+.1f} 到 {100 * pl['ci'][1]:+.1f}" if "ci" in pl else "–")
                      + f" | {qa_probe.VERDICT_ZH[pl['verdict']]} |")
    md += ["", "## 第 5 节：稳不稳", "", "| 组 | RL | 停在第几次 | 原因 | 主指标用的检查点 |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {g} | {r if '`' in r or g == 'C0' else '`' + r + '`'} | {u} | {why} | {ck} |" for g, r, u, why, ck in STABILITY]
    rep["stability"] = STABILITY
    md += ["", "C 的 4 个都被停下，D 0 个，E 2 个，C0 1 个。C0 跑的时候没有熵线和图像 token 崩溃线，KL 线是 0.2。", ""]
    (out_dir / "rest.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "rest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
