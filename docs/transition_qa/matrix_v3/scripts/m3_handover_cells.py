"""What the main line asked for on 2026-11-04 (the P1 columns of its matrix summary for the new cells; the user's D of
11-03: results go to the main line in the pre-registration's format): for each cell's picked SFT epoch the test
success (screened test set, 292 tasks), the probe (838 questions), the fail-end steps d (reward.fail_end_steps on the
test set with the main line's settings, handover_cells_20261104/failend_test.json) and the sampling check; for every
finished RL chain the two readings of handover_main_20261103 ("prereg": update 300, or the last checkpoint of a run
stopped early; "best_val": the highest validation success, ties to the earlier) with their probes; the read-only
weight paths.

    python exps/matrix_v3/watch/m3_handover_cells.py OUT_PREFIX
"""
import json
import sys
from pathlib import Path

W = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
M3 = W / "exps" / "matrix_v3"
sys.path.insert(0, str(W))
from eval.diagnostics import qa_probe  # noqa: E402

PROBE = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261019b/p1_probe.jsonl")
FAILEND = M3 / "handover_cells_20261104" / "failend_test.json"
FAILEND_RL = M3 / "handover_cells_20261104" / "failend_rl_test.json"  # the RL runs' prereg checkpoints, once all ran
CELL_PROBES = M3 / "probe_cells_20261104" / "probe"  # the cells without an RL chain (E2, Cr): sco_probe pt-ostoj75m
D_KEYS = ("failed", "d_end_mean", "d_end_median", "d_star_mean", "d_end_le1", "d_end_le2")
CELLS = {"B'": ("b", "sft_m3b_e10", ["m3_b_s20261019"]),
         "D_noP0": ("dn", "sft_m3dn_e8", ["m3_dn_s20261019", "m3_dn_s20261021"]),
         "E2": ("e2", "sft_m3e2_e10", []), "Cr": ("cr", "sft_m3cr_e7", [])}


def success(folder: Path) -> float:
    rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    return sum(bool(r.get("success")) for r in rows) / len(rows)


def model_path(folder: Path) -> str:
    return json.loads((folder / "model.json").read_text()).get("model")


def probe_row(path: Path, probe: dict) -> dict:
    """As m3_prereg_rest.probe_row: overall, occlusion balanced, unreadable, the two goal types and their Yes / No."""
    rows = {r["sample_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}
    assert set(rows) == set(probe), (path, len(rows))

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


def main(out: str) -> None:
    probe = {r["sample_id"]: r for r in (json.loads(l) for l in PROBE.read_text().splitlines() if l.strip())}
    failend = json.loads(FAILEND.read_text())["results"]
    failend_rl = json.loads(FAILEND_RL.read_text())["results"] if FAILEND_RL.exists() else {}
    sft, runs, pending, weights = {}, [], [], {"sft": {}, "rl": {}}
    for cell, (short, name, chains) in CELLS.items():
        tr = M3 / f"m3_{short}_sft"
        test_dir, val_dir = tr / "eval_test_v2_screened" / name, tr / "eval_val_v2_screened" / name
        acc = json.loads((tr / "accept_summary.json").read_text())[f"{name}_s4"]
        fe = failend[str(test_dir)]
        start = next((M3 / c / "diag" / "probe_P1" / "start.jsonl" for c in chains
                      if (M3 / c / "diag" / "probe_P1" / "start.jsonl").exists()), None)
        if start is None and (CELL_PROBES / f"{name}.jsonl").exists():
            start = CELL_PROBES / f"{name}.jsonl"
        row = {"result": name, "test": round(success(test_dir), 4), "val": round(success(val_dir), 4),
               "model": model_path(test_dir), "result_dirs": {"test": str(test_dir), "val": str(val_dir)},
               "accept": {k: acc[k] for k in ("entropy", "unreadable_share", "pass", "success")},
               "d_test": {k: fe[k] for k in D_KEYS}, "probe": probe_row(start, probe) if start else None}
        row["model_on_disk"] = Path(row["model"]).is_dir()
        sft[cell] = row
        weights["sft"][f"{cell} {name}"] = row["model"]
        for c in chains:
            ch = M3 / c
            state = json.loads((ch / "progress.json").read_text()).get("state")
            if state != "done":
                pending.append({"cell": cell, "chain": c, "state": state})
                continue
            plan = json.loads((ch / "plan.json").read_text())
            val_set, test_set = ch / "eval_val_v2_screened", ch / "eval_test_v2_screened"
            for run in plan["rl_runs"]:
                ckpts = sorted((int(p.name[len(run) + 1:]), p.name) for p in val_set.glob(f"{run}_[0-9][0-9][0-9]")
                               if (p / "results.jsonl").is_file())
                val = {n: success(val_set / n) for _, n in ckpts}
                best = max(ckpts, key=lambda x: (val[x[1]], -x[0]))[1]  # ties: the earlier
                r = {"cell": cell, "chain": c, "seed": int(plan["rl"]["seed"]), "run": run,
                     "side": "V4" if run.endswith("d") else "S0", "last_update": ckpts[-1][0]}
                for k, n in (("prereg", ckpts[-1][1]), ("best_val", best)):
                    pj = ch / "diag" / "probe_P1" / f"{n}.jsonl"
                    r[k] = {"result": n, "test": round(success(test_set / n), 4), "val": round(val[n], 4),
                            "model": model_path(test_set / n), "probe": probe_row(pj, probe) if pj.exists() else None}
                    r[k]["model_on_disk"] = Path(r[k]["model"]).is_dir()
                    weights["rl"][n] = r[k]["model"]
                fe = failend_rl.get(str(test_set / r["prereg"]["result"]))
                r["prereg"]["d_test"] = {k: fe[k] for k in D_KEYS} if fe else None
                r["best_val"]["checkpoints_on_val"] = {n: round(v, 4) for n, v in val.items()}
                runs.append(r)
    groups = {}
    for cell in CELLS:
        rs = [r for r in runs if r["cell"] == cell]
        if rs:
            groups[cell] = {k: round(sum(r[k]["test"] for r in rs) / len(rs), 4) for k in ("prereg", "best_val")}
            groups[cell].update(runs=len(rs), complete=not any(p["cell"] == cell for p in pending))
            if all(r["prereg"]["d_test"] for r in rs):  # the matrix summary's RL d: the runs' prereg d averaged
                groups[cell]["d_prereg_mean"] = round(sum(r["prereg"]["d_test"]["d_end_mean"] for r in rs) / len(rs), 3)
            if all(r["prereg"]["probe"] for r in rs):   # and its probe columns: the runs' prereg probes averaged
                groups[cell]["probe_prereg_mean"] = {q: round(sum(r["prereg"]["probe"][q] for r in rs) / len(rs), 4)
                                                     for q in ("goal_now", "goal_after")}
            groups[cell]["stopped_before_300"] = sum(r["last_update"] < 300 for r in rs)
    doc = {"readings": json.loads((Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/docs/transition_qa/matrix_v3")
                                   / "handover_main_20261103.json").read_text())["readings"],
           "test_set": "eval_test_v2_screened (292 tasks)", "val_set": "eval_val_v2_screened (382 tasks)",
           "probe_set": str(PROBE), "d": "reward.fail_end_steps (VAGEN-Lite-r12dense 22a0fc62), test set, tables "
                                         "reward_runs/tables/eval_test_v2_screened, 4 workers; d_end_mean = the column d "
                                         "(SFT picks: failend_test.json; RL prereg checkpoints: failend_rl_test.json; a "
                                         "group's d_prereg_mean = its runs' prereg d averaged, the matrix summary's RL d)",
           "sft": sft, "runs": runs, "groups": groups, "pending": pending, "weights_read_only": weights}
    Path(out + ".json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    md = ["# 矩阵第三版补充格：交主线的结果（2026-11-04）", "",
          "主线 11-04 要的：补充格 SFT 挑点的测试集成功率、探针 goal_now / goal_after、d、采样验收；RL 的检查点路径。"
          "口径同 `handover_main_20261103`（测试集 292 道、验证集 382 道，都去掉边缘场景）。", "",
          "## SFT 挑点", "",
          "| 格 | 挑点 | 测试 % | 验证 % | 探针 goal_now / goal_after % | d 平均（中位），失败局 | 采样验收：熵、读不出、过没过 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for cell, r in sft.items():
        p = r["probe"]
        md.append(f"| {cell} | `{r['result']}` | {pct(r['test'])} | {pct(r['val'])} | "
                  + (f"{pct(p['goal_now'])} / {pct(p['goal_after'])}" if p else "没跑") + " | "
                  + f"{r['d_test']['d_end_mean']:.2f}（{r['d_test']['d_end_median']:g}），{r['d_test']['failed']} | "
                  + f"{r['accept']['entropy']:.3f}、{100 * r['accept']['unreadable_share']:.2f}%、"
                  + ("过" if r["accept"]["pass"] else "没过") + " |")
    probed = [c for c in ("E2", "Cr") if sft[c]["probe"]]
    md += ["", "- 探针：B′、D_noP0 的挑点是 RL 链诊断步骤里的 `start`（RL 的起点就是这个挑点）；"
           + ("E2、Cr 没有 RL 链，探针单独跑（sco_probe `pt-ostoj75m`，两个模型一个任务，结果在 `probe_cells_20261104/probe/`）。"
              if len(probed) == 2 else "E2、Cr 没有 RL，没跑探针。"),
           "- d：失败局终点离成功还差几步，`reward.fail_end_steps`，设置同主线 `failend_m3_20261103`。逐局结果：SFT 挑点在 "
           "`handover_cells_20261104/failend_test.json`，RL 的预登记检查点在 `failend_rl_test.json`。", "",
           "## RL（跑完的链）", "",
           "| 格 | 种子 | 侧 | 预登记口径：检查点 | 测试 | d | 按验证集挑：检查点 | 验证 | 测试 | 权重在盘上 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        d = r["prereg"]["d_test"]
        md.append(f"| {r['cell']} | {r['seed']} | {r['side']} | `{r['prereg']['result']}` | {pct(r['prereg']['test'])} | "
                  + (f"{d['d_end_mean']:.2f}" if d else "—") + " | "
                  + f"`{r['best_val']['result']}` | {pct(r['best_val']['val'])} | {pct(r['best_val']['test'])} | "
                  + ("是" if r["prereg"]["model_on_disk"] and r["best_val"]["model_on_disk"] else "否") + " |")
    md += ["", "合起来（测试集成功率 %；d 和探针是各运行预登记检查点的平均）：", "",
           "| 格 | 运行数 | 预登记口径 | 按验证集挑 | d | 探针 goal_now / goal_after % | 300 次前被停下 | 齐了没有 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {g} | {v['runs']} | {pct(v['prereg'])} | {pct(v['best_val'])} | "
           + (f"{v['d_prereg_mean']:.2f}" if "d_prereg_mean" in v else "—") + " | "
           + (f"{pct(v['probe_prereg_mean']['goal_now'])} / {pct(v['probe_prereg_mean']['goal_after'])}"
              if "probe_prereg_mean" in v else "—")
           + f" | {v['stopped_before_300']}/{v['runs']} | " + ("齐" if v["complete"] else "没齐") + " |"
           for g, v in groups.items()]
    if pending:
        md += ["", "还在跑的链：" + "，".join(f"`{p['chain']}`（{p['state']}）" for p in pending) + "。"]
    md += ["", "## 权重（只读）", ""]
    md += [f"- SFT {k}：`{v}`" for k, v in weights["sft"].items()]
    md += [f"- RL `{k}`：`{v}`" for k, v in sorted(weights["rl"].items())]
    Path(out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
