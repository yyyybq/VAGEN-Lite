"""Matrix v3, the supplementary cells by the methods of 3.1-3.4 and section 5 (pre-registration notes 7-9; all
exploratory, no H1-H3 verdict changes): D - D_noP0 (only P0 differs) and D_noP0 - C (P1 instead of P0), two seeds x
S0 / V4; B' - C (B' ran seed 20261019 only, so its two pairs get no two-seed verdict); Cr - C and Cr - D (3.1 on the
test set is cr_31_20261105, repeated here with the validation set and 3.2-3.4); the SFT stage, one model each:
D_noP0 - D, D_noP0 - C, B' - C, E2 - D, Cr - C.  The tests, bootstrap draws and thresholds are m3_prereg_rest's
(imported); 3.2's Holm correction runs over each comparison's 6 families (as E - C0 in the first batch); the
occlusion family stays exploratory.  Read only; writes OUT/supp_rest.{json,md} and OUT/probe/compare_*.

    python m3_supp_rest.py OUT_DIR
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_prereg_rest as R  # noqa: E402  (also puts the matrix worktree on sys.path)
from eval.diagnostics import qa_probe  # noqa: E402
from eval.diagnostics.failure_breakdown import FAMILY_ORDER  # noqa: E402

M3 = R.M3
N = {"test": 292, "val": 382}
SET_ZH = {"test": "测试集 292 道", "val": "验证集 382 道，参考"}
RL = {"C": [("shared", n) for n in ("m3cs1s0_050", "sft_m3c_e4", "m3cs2s0_100", "m3cs2d_050")],
      "D": [("shared", n) for n in ("m3ds1s0_300", "m3ds1d_300", "m3ds2s0_300", "m3ds2d_300")],
      "D_noP0": [("m3_dn_s20261019", "m3dns1s0_300"), ("m3_dn_s20261019", "m3dns1d_300"),
                 ("m3_dn_s20261021", "m3dns2s0_300"), ("m3_dn_s20261021", "m3dns2d_300")],
      "Cr": [("m3_cr_s20261019", "m3crs1s0_250"), ("m3_cr_s20261019", "m3crs1d_250"),
             ("m3_cr_s20261021", "m3crs2s0_300"), ("m3_cr_s20261021", "m3crs2d_200")],
      "B'": [("m3_b_s20261019", "m3bs1s0_300"), ("m3_b_s20261019", "m3bs1d_300")]}
SFT = {"C": ("shared", "sft_m3c_e4"), "D": ("shared", "sft_m3d_e9"), "D_noP0": ("m3_dn_sft", "sft_m3dn_e8"),
       "B'": ("m3_b_sft", "sft_m3b_e10"), "E2": ("m3_e2_sft", "sft_m3e2_e10"), "Cr": ("m3_cr_sft", "sft_m3cr_e7")}
RL_CMP = [("D", "D_noP0", "只差 P0"), ("D_noP0", "C", "P1 换 P0"),
          ("B'", "C", "格式对话换动作轨迹；只有种子 20261019 的两侧，不下两个种子的判定"),
          ("Cr", "C", "C 的 SFT 重训、从过了采样验收的 epoch 接 RL"), ("Cr", "D", "同上，对 D")]
SFT_CMP = [("D_noP0", "D", "只差 P0"), ("D_noP0", "C", "P1 换 P0"), ("B'", "C", "格式对话换动作轨迹"),
           ("E2", "D", "只差动作轨迹"), ("Cr", "C", "同一配方重训")]
PROBE_DIRS = [M3 / f"m3_{g}_s{s}" / "diag" / "probe_P1" for g in ("c", "d", "e", "dn", "cr", "b") for s in R.SEEDS] + \
    [M3 / "probe_cells_20261104" / "probe"]
STARTS = {"sft_m3dn_e8": M3 / "m3_dn_s20261019/diag/probe_P1/start", "sft_m3b_e10": M3 / "m3_b_s20261019/diag/probe_P1/start"}


def folder(key: str, name: str, s: str) -> Path:
    return (M3 / f"eval_{s}_v2_screened" if key == "shared" else M3 / key / f"eval_{s}_v2_screened") / name


def probe_path(name: str) -> str:
    """The probe answers of a model, without the .jsonl suffix (qa_probe.compare's form)."""
    for d in PROBE_DIRS:
        if (d / f"{name}.jsonl").exists():
            return str(d / name)
    p = STARTS[name]
    assert Path(str(p) + ".jsonl").exists(), name
    return str(p)


def probe_row(name: str, probe: dict) -> dict:
    """As m3_prereg_rest.probe_row, for a model found by probe_path."""
    path = Path(probe_path(name) + ".jsonl")
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


def pairs_of(a: str, b: str, res: dict) -> tuple:
    xa, xb = res[a], res[b]
    return (xa, xb[:len(xa)]) if len(xa) < len(xb) else (xa, xb)


def verdict31(r: dict) -> str:
    if not r["seed_diff_pp"]:
        return "只有一个种子，不下判定"
    return "有差异" if r["p"] < R.ALPHA and abs(r["pooled_diff_pp"]) >= 5 and r["same_sign"] else "没有差异"


def main() -> int:
    out_dir = Path(sys.argv[1])
    (out_dir / "probe").mkdir(parents=True, exist_ok=True)
    rep: dict = {"rules": "m3_prereg_rest (tests, draws, thresholds); 3.2 Holm per comparison over 6 families", "sets": {}}
    md = ["# 矩阵第三版补充格：按预登记 3.1–3.4、第 5 节的方法（探索性）", "", __doc__.split("\n\n")[0], ""]
    res: dict = {}
    for s in N:
        res[s] = {g: [R.load(folder(k, n, s), N[s]) for k, n in xs] for g, xs in RL.items()}
        res[s]["sft"] = {g: R.load(folder(*v, s), N[s]) for g, v in SFT.items()}
    for s in N:
        ids = sorted(res[s]["D"][0])
        fam = {t: res[s]["D"][0][t]["family"] for t in ids}
        allm = [m for g in RL for m in res[s][g]] + list(res[s]["sft"].values())
        assert all(sorted(m) == ids and all(m[t]["family"] == fam[t] for t in ids) for m in allm), s
        S = rep["sets"][s] = {"3.1": {}, "3.2": {}, "3.4": {}, "sft": {}}
        md += [f"## {SET_ZH[s]}", "", "### 3.1 动作成功率（差的单位是百分点）", "",
               "| 比较 | 配对的差 | 种子 1 / 种子 2 | 合起来（95% 区间） | 只有 A 对 / 只有 B 对 | 符号检验 p | 按 3.1 的规则 |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for a, b, why in RL_CMP:
            xa, xb = pairs_of(a, b, res[s])
            r = S["3.1"][f"{a}-{b}"] = R.paired(xa, xb, ids, f"{s}/all", "success")
            r["verdict"] = verdict31(r)
            md.append(f"| {a} − {b}（{why}） | " + " / ".join(f"{x['diff_pp']:+.1f}" for x in r["pairs"]) + " | "
                      + (" / ".join(f"{x:+.2f}" for x in r["seed_diff_pp"]) if r["seed_diff_pp"] else "–")
                      + f" | {r['pooled_diff_pp']:+.2f}（{r['ci95'][0]:+.1f} 到 {r['ci95'][1]:+.1f}） | {r['only_a']} / {r['only_b']} | "
                      f"{R.fp(r['p'])} | {r['verdict']} |")
        md += ["", "成功率（RL 合起来，\"题 × RL\"）：" + "，".join(f"{g} {R.rate(res[s][g], ids, 'success')}" for g in RL) + "。", "",
               "### 3.2 按任务族", "", "判定\"有差异\"：Holm 校正后 p < 0.05，且 |差| ≥ 10，且两个种子同号（B′ − C 只有一个种子，不下判定）。", "",
               "| 比较 | 任务族 | 题数 | A | B | 合起来（95% 区间） | p | Holm 校正后 p | 判定 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for a, b, why in RL_CMP:
            xa, xb = pairs_of(a, b, res[s])
            recs = [(f, [t for t in ids if fam[t] == f]) for f in FAMILY_ORDER]
            rs = [R.paired(xa, xb, fids, f"{s}/{f}", "success") for f, fids in recs]
            for (f, fids), r, adj in zip(recs, rs, R.holm([r["p"] for r in rs])):
                r["p_holm"] = adj
                ok = adj < R.ALPHA and abs(r["pooled_diff_pp"]) >= 10 and r["same_sign"]
                v = "只有一个种子" if not r["seed_diff_pp"] else ("有差异" if ok else "没有差异")
                r["verdict"] = v
                S["3.2"][f"{a}-{b}/{f}"] = r
                md.append(f"| {a} − {b} | {R.FAM_ZH[f]} | {len(fids)} | {R.rate(xa, fids, 'success')} | {R.rate(xb, fids, 'success')} | "
                          f"{r['pooled_diff_pp']:+.1f}（{r['ci95'][0]:+.1f} 到 {r['ci95'][1]:+.1f}） | {R.fp(r['p'])} | {R.fp(adj)} | {v} |")
        md += ["", "### 3.4 失败拆解", "", "①差\"关系\"的占只差一个条件的失败的比例（z 检验）；②差\"关系\"的只差一个的失败占全部题的比例（逐题符号检验）。", "",
               "| 比较 | ① A | ① B | ① 差 | p | ② 合起来（95% 区间） | ② p | ① 下降 | ② 下降 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for a, b, why in RL_CMP:
            xa, xb = pairs_of(a, b, res[s])
            ra, na = (sum(m[t][k] for m in xa for t in ids) for k in ("rel", "one"))
            rb, nb = (sum(m[t][k] for m in xb for t in ids) for k in ("rel", "one"))
            d1, z, p1 = R.ztest(ra, na, rb, nb)
            r2 = R.paired(xa, xb, ids, f"{s}/all", "rel")
            down1 = p1 < R.ALPHA and d1 <= -10
            down2 = bool(r2["seed_diff_pp"]) and r2["p"] < R.ALPHA and all(x < 0 for x in r2["seed_diff_pp"])
            S["3.4"][f"{a}-{b}"] = {"share_a": [ra, na], "share_b": [rb, nb], "diff_pp": round(d1, 2), "p": p1,
                                    "all_tasks": r2, "down": down1, "all_tasks_down": down2}
            md.append(f"| {a} − {b} | {100 * ra / na:.1f}%（{ra} / {na}） | {100 * rb / nb:.1f}%（{rb} / {nb}） | {d1:+.1f} | {R.fp(p1)} | "
                      f"{r2['pooled_diff_pp']:+.2f}（{r2['ci95'][0]:+.1f} 到 {r2['ci95'][1]:+.1f}） | {R.fp(r2['p'])} | "
                      f"{'是' if down1 else '否'} | {'是' if down2 else '否'} |")
        sft = res[s]["sft"]
        md += ["", "### SFT 阶段（每组一个模型，不校正）", "",
               "挑点：" + "，".join(f"{g} `{v[1]}`" for g, v in SFT.items()) + "。", "",
               "| 比较 | 合计（p） | " + " | ".join(R.FAM_ZH[f] for f in FAMILY_ORDER) + " | 3.4 ① 差（p） | 3.4 ② 差（p） |",
               "| --- | --- | " + " | ".join("---" for _ in FAMILY_ORDER) + " | --- | --- |"]
        for a, b, why in SFT_CMP:
            cells = []
            for f in ("all",) + tuple(FAMILY_ORDER):
                fids = ids if f == "all" else [t for t in ids if fam[t] == f]
                r = S["sft"][f"{a}-{b}/{f}"] = R.paired([sft[a]], [sft[b]], fids, f"{s}/{f}", "success")
                cells.append(f"{r['pooled_diff_pp']:+.1f}（{R.fp(r['p'])}）")
            ra, na = sum(sft[a][t]["rel"] for t in ids), sum(sft[a][t]["one"] for t in ids)
            rb, nb = sum(sft[b][t]["rel"] for t in ids), sum(sft[b][t]["one"] for t in ids)
            d1, z, p1 = R.ztest(ra, na, rb, nb)
            r2 = S["sft"][f"{a}-{b}/rel"] = R.paired([sft[a]], [sft[b]], ids, f"{s}/all", "rel")
            S["sft"][f"{a}-{b}/share"] = {"diff_pp": round(d1, 2), "p": p1}
            md.append(f"| {a} − {b}（{why}） | " + " | ".join(cells) + f" | {d1:+.1f}（{R.fp(p1)}） | {r2['pooled_diff_pp']:+.1f}（{R.fp(r2['p'])}） |")
        md += ["", "SFT 成功率：" + "，".join(f"{g} {R.rate([sft[g]], ids, 'success')}" for g in SFT) + "。", ""]

    probe = {r["sample_id"]: r for r in (json.loads(l) for l in R.PROBE.read_text(encoding="utf-8").splitlines() if l.strip())}
    assert len(probe) == 838
    P = rep["probe"] = {"models": {}, "compare": {}}
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    md += ["## 3.3 探针（838 道）", "", "| 组 | 模型 | 全部 | `goal_now`（Yes / No） | `goal_after`（Yes / No） | 遮挡平衡准确率（完全 / 部分 / 没有） | 读不出 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for g in list(SFT):
        for name in [SFT[g][1]] + [n for _, n in RL.get(g, [])]:
            if name in P["models"]:
                continue
            r = P["models"][name] = probe_row(name, probe)
            md.append(f"| {g} | `{name}` | {pct(r['all'])} | " + " | ".join(
                f"{pct(r[q])}（{pct(r[q + '_by']['yes'])} / {pct(r[q + '_by']['no'])}）" for q in qa_probe.GOAL_TYPES)
                + f" | {pct(r['occ_balanced'])}（" + " / ".join(pct(r["occ"][v]) for v in ("full", "partial", "none"))
                + f"） | {pct(r['unreadable'])}% |")
    md += ["", "比较用 `qa_probe.compare`。goal 两类：合起来 McNemar p < 0.05，且 |差| ≥ 10，且两个种子同号；遮挡：|差| ≥ 20 且两个种子同号。", "",
           "| 比较 | 题类 | 各种子 | 合起来 | p | 95% 区间 | 判定 |", "| --- | --- | --- | --- | --- | --- | --- |"]

    def grp(g: str, n_pairs: int) -> tuple:
        names = [n for _, n in RL[g]][:n_pairs]
        return (g, [(R.SEEDS[i], [probe_path(names[2 * i]), probe_path(names[2 * i + 1])]) for i in range(n_pairs // 2)])

    jobs = []
    for a, b, why in RL_CMP:
        n_pairs = min(len(RL[a]), len(RL[b]))
        jobs.append((f"{a}_vs_{b}".replace("'", "p"), grp(a, n_pairs), grp(b, n_pairs), f"{a} − {b}"))
    for a, b, why in SFT_CMP:
        jobs.append((f"{a}_SFT_vs_{b}_SFT".replace("'", "p"), (f"{a}_SFT", [probe_path(SFT[a][1])]),
                     (f"{b}_SFT", [probe_path(SFT[b][1])]), f"SFT：{a} − {b}"))
    for label, ga, gb, title in jobs:
        r = qa_probe.compare(R.PROBE, out_dir / "probe", ga, gb, label)
        (out_dir / "probe" / f"compare_{label}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        (out_dir / "probe" / f"compare_{label}.md").write_text(qa_probe.compare_markdown(r), encoding="utf-8")
        P["compare"][label] = {q: r[q]["pooled"] | {"per_seed": [x["diff"] for x in r[q]["per_seed"]]}
                               for q in qa_probe.GOAL_TYPES + ("occlusion",)}
        for q in qa_probe.GOAL_TYPES + ("occlusion",):
            pl, ps = r[q]["pooled"], [x["diff"] for x in r[q]["per_seed"]]
            md.append(f"| {title} | {'遮挡（平衡）' if q == 'occlusion' else '`' + q + '`'} | "
                      + (" / ".join(f"{100 * x:+.1f}" for x in ps) if ps else "–") + f" | {100 * pl['diff']:+.1f} | {R.fp(pl['p'])} | "
                      + (f"{100 * pl['ci'][0]:+.1f} 到 {100 * pl['ci'][1]:+.1f}" if "ci" in pl else "–")
                      + f" | {qa_probe.VERDICT_ZH[pl['verdict']]} |")
    hc = json.loads((M3 / "handover_cells_20261104" / "handover_cells_20261104.json").read_text())
    md += ["", "## 第 5 节：稳不稳（补充格）", "", "| 格 | 种子 | 侧 | 跑了几次更新 | 原因 | 预登记检查点 |", "| --- | --- | --- | --- | --- | --- |"]
    rep["stability"] = []
    for r in hc["runs"]:
        why = r["stop"]["why"] if r.get("stop") else "跑满"
        rep["stability"].append({k: r.get(k) for k in ("cell", "seed", "side", "run", "updates", "stop")})
        md.append(f"| {r['cell']} | {r['seed']} | {r['side']} | {r['updates']} | {why} | `{r['prereg']['result']}` |")
    md += ["", "B′ 2 个、D_noP0 4 个都跑满 300 次；Cr 4 个里 3 个被停下（见 PROGRESS 12.52、12.55）。", ""]
    (out_dir / "supp_rest.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "supp_rest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
