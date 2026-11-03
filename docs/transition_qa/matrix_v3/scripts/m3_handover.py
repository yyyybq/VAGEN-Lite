"""What the main line asked for (11-03; the user's D: results to the main line in the pre-registration's format, the
main line writes the summary): the action success rate of every matrix v3 RL in two readings, the probe numbers per
model, where the failure breakdowns are, and the read-only weight paths.

    python exps/matrix_v3/watch/m3_handover.py OUT_PREFIX

Two readings of the action success rate (test set, 292 tasks without the edge scenes):
- "prereg" (the pre-registration's section 2, the main metric): each RL's update-300 checkpoint; a run stopped by a
  stop line its last checkpoint; a run stopped before its first checkpoint (m3cs1d) its SFT start;
- "best_val" (the main line's Qwen3 reports; exploratory in the pre-registration): each RL's checkpoint with the
  highest success on the validation set (382 tasks), ties to the earlier, and that checkpoint's test success.
Reads main_31.json and rest.json (the committed analyses) and every result folder's results.jsonl and model.json.
"""
import json
import re
import sys
from pathlib import Path

DOCS = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/docs/transition_qa/matrix_v3")
M3 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3")
SFT_START = {"C": "sft_m3c_e4", "D": "sft_m3d_e9", "E": "c0q3_sft_e8", "C0": "c0q3_sft_e8"}
CHAINS = {"C": "m3_c", "D": "m3_d", "E": "m3_e"}


def success(folder: Path) -> float:
    rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    return sum(bool(r.get("success")) for r in rows) / len(rows)


def model_path(folder: Path) -> str:
    return json.loads((folder / "model.json").read_text()).get("model")


def main(out: str) -> None:
    m31 = json.loads((DOCS / "main_31.json").read_text())
    rest = json.loads((DOCS / "rest.json").read_text())
    runs, groups = [], {}
    for g, side in m31["models"].items():
        for key, test_dir in side.items():
            test_dir = Path(test_dir)
            test_set, val_set = test_dir.parent, test_dir.parent.parent / "eval_val_v2_screened"
            name = test_dir.name
            run = (re.sub(r"_\d{3}$", "", name) if re.search(r"_\d{3}$", name) else
                   {"C": "m3cs1d"}.get(g) if name.startswith("sft_") else name)
            ckpts = sorted((int(p.name[len(run) + 1:]), p.name) for p in val_set.glob(f"{run}_[0-9][0-9][0-9]")
                           if (p / "results.jsonl").is_file())
            val = {n: success(val_set / n) for _, n in ckpts}
            best = max(ckpts, key=lambda c: (val[c[1]], -c[0]))[1] if ckpts else name   # ties: the earlier
            row = {"group": g, "seed": int(key[0]), "side": "S0" if key[1:] == "s0" else "V4", "run": run,
                   "prereg": {"result": name, "test": round(success(test_set / name), 4),
                              "val": round(success(val_set / name), 4), "model": model_path(test_set / name)},
                   "best_val": {"result": best, "val": round(success(val_set / best), 4),
                                "test": round(success(test_set / best), 4), "model": model_path(test_set / best),
                                "checkpoints_on_val": {n: round(v, 4) for n, v in val.items()}}}
            for k in ("prereg", "best_val"):
                row[k]["model_on_disk"] = Path(row[k]["model"]).is_dir()
            runs.append(row)
    for g in m31["models"]:
        rs = [r for r in runs if r["group"] == g]
        groups[g] = {k: round(sum(r[k]["test"] for r in rs) / len(rs), 4) for k in ("prereg", "best_val")}
    probes = {n: {k: v for k, v in d.items() if k in ("all", "goal_now", "goal_now_by", "goal_after", "goal_after_by",
                                                       "occ_balanced", "occ", "unreadable", "path")}
              for n, d in rest["probe"]["models"].items()}
    failure = {"summary": "rest.json sets.test['3.4'] (pre-registration 3.4, test) and sets.val['3.4'] (val)",
               "per_chain": {f"{c}_s{s}": {k: str(M3 / f"{c}_s{s}" / "diag" / f"failure_{k}.json") for k in ("test", "val")}
                             for c in CHAINS.values() for s in (20261019, 20261021)}}
    sft = {g: n for g, n in SFT_START.items()}
    weights = {
        "sft": {"C (epoch 4)": str(M3 / "sft" / "C" / "model" / "checkpoint-1204"),
                "D (epoch 9)": str(M3 / "sft" / "D" / "model" / "checkpoint-3987"),
                "C0 / E start (epoch 8)": "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232"},
        "rl": {r["prereg"]["result"]: r["prereg"]["model"] for r in runs} | {r["best_val"]["result"]: r["best_val"]["model"] for r in runs},
        "new_cells": {"D_noP0 SFT (every epoch kept until the pick)": str(M3 / "sft" / "D_noP0" / "model"),
                      "B' SFT (every epoch kept until the pick)": str(M3 / "sft" / "B" / "model")},
    }
    doc = {"readings": {"prereg": __doc__.split("- \"prereg\"")[1].split("- \"best_val\"")[0].strip(),
                        "best_val": __doc__.split("- \"best_val\"")[1].split("Reads")[0].strip()},
           "test_set": "eval_test_v2_screened (292 tasks)", "val_set": "eval_val_v2_screened (382 tasks)",
           "runs": runs, "groups": groups, "sft_starts": sft, "probe": probes, "failure_breakdown": failure,
           "weights_read_only": weights}
    Path(out + ".json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    md = ["# 矩阵第三版：交主线的结果（2026-11-03）", "",
          "主线 11-03 要的：动作成功率两种口径、探针逐模型的数、失败拆解、权重的只读路径。用户 11-03 定：结果按预登记的格式交主线，汇总由主线写。", "",
          "## 两种口径（测试集 292 道，去掉边缘场景）", "",
          "- **预登记口径**（预登记第 2 节，主指标）：每个 RL 用第 300 次的检查点；被停止线停下的用最后一个检查点；"
          "第一个检查点之前就停下的（`m3cs1d`）用它的 SFT 起点。",
          "- **按验证集挑**（主线 Qwen3 报告的做法；预登记里只作探索性）：每个 RL 在验证集 382 道上成功率最高的检查点"
          "（每 50 次更新一个，打平取更早的），报它在测试集上的成功率。", "",
          "| 组 | 种子 | 侧 | 预登记口径：检查点 | 测试 | 按验证集挑：检查点 | 验证 | 测试 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        md.append(f"| {r['group']} | {r['seed']} | {r['side']} | `{r['prereg']['result']}` | {100 * r['prereg']['test']:.1f} | "
                  f"`{r['best_val']['result']}` | {100 * r['best_val']['val']:.1f} | {100 * r['best_val']['test']:.1f} |")
    md += ["", "四个 RL 合起来（测试集成功率，%）：", "", "| 组 | 预登记口径 | 按验证集挑 |", "| --- | --- | --- |"]
    md += [f"| {g} | {100 * v['prereg']:.1f} | {100 * v['best_val']:.1f} |" for g, v in groups.items()]
    md += ["", "- C0 在去掉边缘场景的两个题集上只评过预登记口径用的检查点和 SFT 第 8 个 epoch（`PROGRESS.md` 12.27），"
           "所以它的两种口径相同。模型线的 C0 报告是在原来的 `eval_val_v2` / `eval_test_v2` 上按验证集挑的，那是另一个题集。",
           "- C 种子 1 的 V4（`m3cs1d`）在第 47 次停下，没有检查点，两种口径都用它的 SFT 起点（用户 11-03 定）。"]
    md += ["", "## 探针（838 道；逐模型）", "",
           "| 模型 | 全部 | `goal_now` | `goal_after` | 遮挡平衡准确率 | 读不出 |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| `{n}` | {100 * d['all']:.1f} | {100 * d['goal_now']:.1f} | {100 * d['goal_after']:.1f} | "
           f"{100 * d['occ_balanced']:.1f} | {100 * d['unreadable']:.1f}% |" for n, d in probes.items()]
    md += ["", "比较和判定见 `rest.md` 第 3.3 节；逐题结果在各链的 `diag/probe_P1/<模型>.jsonl`（JSON 里的 `path`）。", "",
           "## 失败拆解", "", "- 预登记 3.4 的汇总：`rest.json` 的 `sets.test[\"3.4\"]`（测试集）、`sets.val[\"3.4\"]`（验证集），表在 `rest.md`。",
           "- 每条链的拆解：`<链>/diag/failure_test.json`、`failure_val.json`（本文件 JSON 的 `failure_breakdown.per_chain`）。", "",
           "## 权重（只读）", "", "主线按路径只读，不要改、不要删。D_noP0、B′ 挑出 epoch 后再补。", ""]
    md += [f"- {k}：`{v}`" for k, v in weights["sft"].items()]
    md += [f"- RL `{k}`：`{v}`" for k, v in sorted(weights["rl"].items())]
    md += [f"- {k}：`{v}`" for k, v in weights["new_cells"].items()]
    Path(out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    missing = [r for r in runs for k in ("prereg", "best_val") if not r[k]["model_on_disk"]]
    print(json.dumps({"groups": groups, "missing_weights": [(r["run"], r["prereg"]["model"]) for r in missing]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
