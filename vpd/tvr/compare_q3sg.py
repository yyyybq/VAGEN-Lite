"""The EXPLORATORY comparison of the third explore start Q3SG (genc0, sft_q3genc0_e8; PREREG_TVR_PILOT.md section 17,
row 2026-11-13; DECISIONS G20), on compare_rl's method (和 compare_rl 同口径).  Report only: not in the gate, the
decision table, A4 or the confirmatory round; computed by this script and rechecked by an independent one.  Not a
run_tvr CODE_FILE.  A separate output: COMPARE_RL (vpd.tvr.compare_rl) is unchanged.

    nice -n 19 PY_EVAL -B -m vpd.tvr.compare_q3sg [--explore-root OUT/tvr/runs/explore/rl] [--pilot-root
        OUT/tvr/runs/pilot] [--out EXPLORE_ROOT/COMPARE_Q3SG.json] [--tasks-dir REPO/data/tasks] [--prereg
        vpd/PREREG_TVR_PILOT.md]
    -> COMPARE_Q3SG.json (schema vpd.tvr.compare_q3sg/1) and COMPARE_Q3SG.md next to it.
    Exit 0 = every cell computed and every cross-check made and equal; 1 = a cross-check mismatch on any eval read, or
    a cross-check that could not be made on an eval whose numbers enter a cell; 3 = a cell missing (refused, not
    imputed); 2 = refused: no single section-17 row of 2026-11-13 (Q3SG_ROW_NEEDLE) in the prereg, the registered eval
    task set is not EVAL500_N tasks, not all 5 Q3SG evals exist and validate, or the trainer invariant fails (below).

Comparisons, first minus second (b = only the first start succeeds, c = only the second), on compare_rl's 5 cells
(compare_tvr.cells: 0W c, 100 c, TOP c, TOP u, TOP blind; TOP = tiers.n_top, frozen ALL; seed 1001):
    Q3SG - Q3B  registered primary comparison on 100 c and TOP c; the other three cells as description;
    Q3SG - Q3S  registered diagnostic on 100 c and TOP c; the other three cells as description.
Q3SG side: EXPLORE_ROOT/Q3SG/evals/xrl__Q3SG__<tier>__<arm>; Q3B / Q3S: the existing pilot evals PILOT_ROOT/<S>/
evals/pilot__<S>__<tier>__<arm>, not rerun.  As compare_rl: paired by task_id over the registered eval tasks
(REPO/data/tasks/eval.json) with compare_tvr.read_side / pair (success = tasks/*/trajectory.json result.success; the
task_dirs / per_task / results_json cross-checks; the section-12.2 exact two-sided p; k/n and Wilson 95% (z =
gates.z) per side; (k1 - k2) / n), compare_rl.crosscheck, and a start table (compare_rl.start_table) of Q3B, Q3S and
Q3SG on the 5 cells.  No multiplicity correction.  The p values cover task sampling only, not the training-seed
variation (one seed per start and tier), so they are optimistic: descriptive only, not a confirmatory result (推断).
Prediction (written in the row before Q3SG had any TVR eval; judged on each registered cell, 100 c and TOP c,
separately; two different results are reported as they are, never merged):
    遗忘            k(Q3SG) >= k(Q3B) (the gap is mainly forgetting; replay can close it);
    动作任务本身    Q3SG - Q3S p >= 0.05 and Q3SG - Q3B p < 0.05 with k(Q3SG) < k(Q3B);
    介于两者之间，不判   otherwise.
The p thresholds compare the exact fractions (p_exact) with 1/20.  A cell whose Q3SG - Q3B pair is missing, or whose
Q3SG - Q3S pair is missing while k(Q3SG) < k(Q3B), has no verdict (null; the cell is then in missing_cells anyway).
Validation (the refusal; made before any trajectory is read; it reads no success field):
  the 5 Q3SG evals as compare_rl.validate does for the RL starts (results.json valid true, n_tasks EVAL500_N, no
  tasks_missing, mechanism_only false, its own start / tier / arm, the registered harness HARNESS_SHA256_PREFIX,
  official_check.json equal true, it served the adapter as it is on disk; that adapter complete with seed 1001, phase,
  the tiers manifest's data sha and the init chain, trained with Q3SG's registered trainer code: pilot_tvr
  .x_adapter_check);
  the trainer invariant ("5 个 .py 文件 sha 不变、starts.json 只多这一条") on each Q3SG adapter's meta.code_files: the 5
  .py shas == the ones the pilot Q3B / Q3S adapters recorded (the adapters whose evals enter the cells: 0W, 100, TOP;
  they must all agree), starts.json == pilot_tvr.STARTS_JSON_SHA256_Q3SG, and the pilot adapters' recorded
  starts.json == pilot_tvr.STARTS_JSON_SHA256_PILOT (the bytes of the Q3SG file without the Q3SG entry; the jobs check
  that on the live file, pilot_tvr.q3sg_trainer_problems, recorded here as this_tree, not judged).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

from vpd import paths
from vpd.tvr import compare_rl as CR
from vpd.tvr import compare_tvr as C
from vpd.tvr import pilot_cfg, pilot_tvr

SCHEMA = "vpd.tvr.compare_q3sg/1"
ROW_DATE = pilot_tvr.Q3SG_ROW_DATE
ROW_NEEDLE = pilot_tvr.Q3SG_ROW_NEEDLE
Q3SG_STARTS = pilot_tvr.Q3SG_STARTS
COMPARISONS = (("Q3SG", "Q3B", "primary"), ("Q3SG", "Q3S", "diagnostic"))
TABLE_STARTS = ("Q3B", "Q3S", "Q3SG")
PILOT_REF_STARTS = ("Q3B", "Q3S")  # the pilot adapters whose recorded code_files the invariant compares with
ALPHA = Fraction(5, 100)
FORGETTING, TASK, BETWEEN = "遗忘", "动作任务本身", "介于两者之间，不判"
RULES = {FORGETTING: "k(Q3SG) >= k(Q3B)",
         TASK: "Q3SG - Q3S p >= 0.05 and Q3SG - Q3B p < 0.05 with k(Q3SG) < k(Q3B)",
         BETWEEN: "otherwise"}
NOTE_P = CR.NOTE_P
NOTE = ("EXPLORATORY, report only (PREREG section 17, 2026-11-13): not in the gate, the decision table, A4 or the "
        "confirmatory round")
Refused = C.Refused


def registered_cells(cfg=None) -> List[tuple]:
    """The row's registered cells: 100 c and TOP c (TOP = tiers.n_top)."""
    return [("100", "c"), (pilot_cfg.n_top(cfg), "c")]


def prediction(vs_q3b: dict, vs_q3s: dict) -> dict:
    """The row's prediction on one registered cell from the Q3SG - Q3B and Q3SG - Q3S pairs (compare_tvr.pair)."""
    if vs_q3b["missing"]:
        return {"verdict": None, "rule": None, "why": "Q3SG - Q3B missing"}
    k1, k2 = vs_q3b["k_first"], vs_q3b["k_second"]
    out = {"k_Q3SG": k1, "k_Q3B": k2, "n": vs_q3b["n"], "p_vs_Q3B": vs_q3b["p"], "p_vs_Q3B_exact": vs_q3b["p_exact"],
           "k_Q3S": None if vs_q3s["missing"] else vs_q3s["k_second"],
           "p_vs_Q3S": None if vs_q3s["missing"] else vs_q3s["p"],
           "p_vs_Q3S_exact": None if vs_q3s["missing"] else vs_q3s["p_exact"]}
    if k1 >= k2:
        verdict = FORGETTING
    elif vs_q3s["missing"]:
        return dict(out, verdict=None, rule=None,
                    why="Q3SG - Q3S missing (k(Q3SG) < k(Q3B): 动作任务本身 or 不判 needs it)")
    elif Fraction(vs_q3s["p_exact"]) >= ALPHA and Fraction(vs_q3b["p_exact"]) < ALPHA:
        verdict = TASK
    else:
        verdict = BETWEEN
    return dict(out, verdict=verdict, rule=RULES[verdict], why=None)


def pilot_code_files(pilot_root: Path, cfg=None) -> dict:
    """The trainer code_files the pilot Q3B / Q3S adapters recorded (0W, 100, TOP: the adapters of the evals in the
    cells): {"files": their common code_files or None, "problems", "adapters": {S/tier: facts}}."""
    root = Path(pilot_root)
    tiers = list(dict.fromkeys(pilot_tvr.EVAL_ADAPTER[t] for t, _ in C.cells(cfg)))
    seen, problems, adapters = [], [], {}
    for s in PILOT_REF_STARTS:
        for t in tiers:
            p = root / s / "adapters" / t / "meta.json"
            m = pilot_tvr.read_json(p) or {}
            cf = m.get("code_files")
            adapters[f"{s}/{t}"] = {"meta": str(p), "code_sha256": m.get("code_sha256"),
                                    "code_files_ok": isinstance(cf, dict)}
            if isinstance(cf, dict):
                seen.append(cf)
            else:
                problems.append(f"{p}: no code_files")
    files = seen[0] if seen and all(x == seen[0] for x in seen) else None
    if seen and files is None:
        problems.append("the pilot Q3B / Q3S adapters recorded different code_files")
    sj = pilot_tvr.STARTS_JSON
    if files is not None and files.get(sj) != pilot_tvr.STARTS_JSON_SHA256_PILOT:
        problems.append(f"the pilot adapters' {sj} {str(files.get(sj))[:12]} != STARTS_JSON_SHA256_PILOT "
                        f"{pilot_tvr.STARTS_JSON_SHA256_PILOT[:12]}")
    return {"files": files, "problems": problems, "adapters": adapters}


def trainer_invariant(explore_root: Path, pilot_root: Path, cfg=None) -> dict:
    """The invariant (module docstring) on every Q3SG adapter's meta: {"ok", "problems", "pilot_reference",
    "adapters", "this_tree"}."""
    ref = pilot_code_files(pilot_root, cfg)
    sj = pilot_tvr.STARTS_JSON
    problems = list(ref["problems"])
    py = {k: v for k, v in (ref["files"] or {}).items() if k != sj}
    adapters = {}
    for tier, _init in pilot_tvr.train_tiers("Q3SG", cfg):
        p = Path(explore_root) / "Q3SG" / "adapters" / tier / "meta.json"
        m = pilot_tvr.read_json(p) or {}
        cf = m.get("code_files")
        bad = []
        if not isinstance(cf, dict):
            bad.append("no code_files")
        else:
            if ref["files"] is not None:
                bad += [f"code_files {rel} {str(cf.get(rel))[:12]} != the pilot adapters' {want[:12]}"
                        for rel, want in py.items() if cf.get(rel) != want]
                other = sorted(set(cf) - set(ref["files"]))
                if other:
                    bad.append(f"code_files has files the pilot adapters do not: {other}")
            if cf.get(sj) != pilot_tvr.STARTS_JSON_SHA256_Q3SG:
                bad.append(f"code_files {sj} {str(cf.get(sj))[:12]} != STARTS_JSON_SHA256_Q3SG "
                           f"{pilot_tvr.STARTS_JSON_SHA256_Q3SG[:12]}")
        if not str(m.get("code_sha256") or "").startswith(pilot_tvr.Q3SG_TRAINER_PREFIX):
            bad.append(f"code_sha256 {str(m.get('code_sha256'))[:12]} != the Q3SG prefix "
                       f"{pilot_tvr.Q3SG_TRAINER_PREFIX}")
        adapters[tier] = {"meta": str(p), "code_sha256": m.get("code_sha256"), "ok": not bad, "problems": bad}
        problems += [f"Q3SG/{tier} ({p}): {b}" for b in bad]
    try:
        fp = pilot_tvr.trainer_fingerprint()
        tree = {"trainer_code_sha256": fp.get("code_sha256"), "problems": pilot_tvr.q3sg_trainer_problems(fp)}
    except Exception as e:  # noqa: BLE001 - recorded, not judged
        tree = {"trainer_code_sha256": None, "problems": [f"unreadable: {e!r}"]}
    return {"ok": not problems, "problems": problems,
            "rule": "5 .py shas == the pilot Q3B / Q3S adapters' recorded ones; starts.json == "
                    "STARTS_JSON_SHA256_Q3SG; the pilot adapters' starts.json == STARTS_JSON_SHA256_PILOT; code_sha256 "
                    f"starts with {pilot_tvr.Q3SG_TRAINER_PREFIX}",
            "pilot_reference": {"code_files": ref["files"], "adapters": ref["adapters"],
                                "py_equals_pinned": py == pilot_tvr.TRAINER_PY_SHA256 if ref["files"] else None},
            "starts_json_sha256": {"pilot": pilot_tvr.STARTS_JSON_SHA256_PILOT,
                                   "Q3SG": pilot_tvr.STARTS_JSON_SHA256_Q3SG},
            "adapters": adapters, "this_tree": tree}


def compare(explore_root: Path = None, pilot_root: Path = None, cfg=None, tasks_dir: Path = None,
            prereg: Path = None, L: pilot_tvr.Layout = None) -> dict:
    L = L or pilot_tvr.LAYOUT
    xroot = Path(explore_root or L.xroot)
    proot = Path(pilot_root or L.pilot_root)
    prereg = Path(prereg or pilot_tvr.prereg_path())
    try:
        text = prereg.read_text(encoding="utf-8")
    except OSError:
        text = ""
    row = pilot_tvr.xrow(text, ROW_DATE, ROW_NEEDLE)
    if row is None:
        raise Refused(f"refused: {prereg} has no single section-17 row '| {ROW_DATE} | ... {ROW_NEEDLE} ...'")
    try:
        commit = pilot_tvr.prereg_state().get("commit")
    except Exception:  # noqa: BLE001 - git may be absent; the file sha256 is recorded anyway
        commit = None
    reg = C.registered_tasks(Path(tasks_dir or paths.TVR_REPO / "data" / "tasks"))
    val = CR.validate(xroot, L, cfg, starts=Q3SG_STARTS)
    if not val["ok"]:
        raise Refused("refused: not all 5 Q3SG evals exist and validate (" + str(len(val["problems"]))
                      + " problems):\n  - " + "\n  - ".join(val["problems"][:40]))
    inv = trainer_invariant(xroot, proot, cfg)
    if not inv["ok"]:
        raise Refused("refused: the trainer invariant does not hold (5 .py shas unchanged, starts.json only gains "
                      f"Q3SG; {len(inv['problems'])} problems):\n  - " + "\n  - ".join(inv["problems"][:40]))
    ids, cats = reg.pop("ids"), reg.pop("cats")
    rset = set(ids)
    z = float(pilot_cfg.get("gates.z", cfg))
    sides: Dict[str, dict] = {}

    def side(s, tier, arm):
        key = f"{s}/{tier}__{arm}"
        if key not in sides:
            root, prefix = CR.side_root(s, xroot, proot, Q3SG_STARTS)
            sides[key] = C.read_side(root, s, tier, arm, rset, z, cats, prefix=prefix)
        return sides[key]

    regc = registered_cells(cfg)
    comps = {}
    for first, second, role in COMPARISONS:
        cs = {}
        for tier, arm in C.cells(cfg):
            cs[f"{tier}__{arm}"] = dict(C.pair(side(first, tier, arm), side(second, tier, arm), ids), tier=tier,
                                        arm=arm, use=role if (tier, arm) in regc else "description")
        comps[f"{first}-{second}"] = {"role": role, "first": first, "second": second, "first_label": CR.label(first),
                                      "second_label": CR.label(second), "b_is": f"only {CR.label(first)} succeeds",
                                      "c_is": f"only {CR.label(second)} succeeds",
                                      "registered_cells": [f"{t}__{a}" for t, a in regc], "cells": cs}
    preds = {}
    for tier, arm in regc:
        key = f"{tier}__{arm}"
        preds[key] = dict(prediction(comps["Q3SG-Q3B"]["cells"][key], comps["Q3SG-Q3S"]["cells"][key]), tier=tier,
                          arm=arm)
    for s in TABLE_STARTS:
        for tier, arm in C.cells(cfg):
            side(s, tier, arm)
    table = CR.start_table(sides, C.cells(cfg), TABLE_STARTS, Q3SG_STARTS)
    missing = [f"{name} {ck}" for name, comp in comps.items() for ck, c in comp["cells"].items() if c["missing"]]
    missing += [f"start table {r['start']} {r['tier']}__{r['arm']}" for r in table if not r["usable"]]
    harness = sorted({str((pilot_tvr.read_json(Path(sd["dir"]) / "results.json") or {}).get("code_sha256"))
                      for sd in sides.values()})
    return {"schema": SCHEMA, "utc": pilot_tvr.utc(), "exploratory": True, "explore_root": str(xroot),
            "pilot_root": str(proot), "note": NOTE, "note_p": NOTE_P, "multiplicity_correction": "none", "z": z,
            "prereg": {"file": str(prereg), "sha256": C.sha256_file(prereg), "commit": commit, "section": 17,
                       "row_text": row, "row_sha256": C.sha256_bytes(row.encode())},
            "validation": val, "trainer_invariant": inv, "harness_code_sha256": harness, "registered_tasks": reg,
            "cells": [f"{t}__{a}" for t, a in C.cells(cfg)], "registered_cells": [f"{t}__{a}" for t, a in regc],
            "comparisons": comps, "prediction_rules": dict(RULES, alpha="1/20 (exact fractions)"),
            "predictions": preds, "table_starts": list(TABLE_STARTS), "start_table": table,
            "inputs": {sd["dir"]: sd["inputs"] for sd in sides.values()},
            "trajectories_digest_rule": C.TRAJ_DIGEST_RULE, "crosscheck": CR.crosscheck(sides),
            "missing_cells": missing}


# --- COMPARE_Q3SG.md ----------------------------------------------------------------------------------------------

USE = {"primary": "主（登记）", "diagnostic": "诊断（登记）", "description": "描述"}


def render_md(rep: dict) -> str:
    n = rep["registered_tasks"]["n"]
    L = ["# TVR 探索：第三个起点 Q3SG 和 Q3B、Q3S 的比较（EXPLORATORY，只报告）", "",
         f"- 依据：PREREG_TVR_PILOT.md 第 17 节 {ROW_DATE} 的登记行（sha256 {rep['prereg']['row_sha256'][:12]}）；"
         "探索性质，不进闸门、决策表、A4 和确认轮；和 compare_rl 同口径；不做多重比较校正。",
         "- **这些 p 只算了题目抽样，没算训练种子之间的波动（每个起点每档只有一个种子），所以偏乐观，只作描述，"
         "不是确认轮的结论（推断）。**",
         f"- 成对：同一批 {n} 道登记评测题按 task_id 配对；成功取 trajectory.json 的 result.success；"
         "p = min(1, 2·P(X ≤ min(b, c)))，X ~ Bin(b + c, 1/2)（第 12 节第 2 条）；b = 只有前一个起点成功，"
         "c = 只有后一个起点成功。",
         "- Q3SG 的评测在 `runs/explore/rl/Q3SG`（前缀 xrl）；Q3B、Q3S 用已有的试点评测（前缀 pilot），不重跑。",
         f"- 训练器：Q3SG 适配器的指纹以 {pilot_tvr.Q3SG_TRAINER_PREFIX} 开头；5 个 .py 文件的 sha 和试点 Q3B、Q3S "
         "适配器记录的相同，starts.json 只多 Q3SG 这一条（核对通过才算）。", ""]
    L += ["## 预判（登记的两格，分别判，不合并）", "",
          "| 格 | Q3SG k/n | Q3B k/n | Q3SG − Q3B 精确 p | Q3S k/n | Q3SG − Q3S 精确 p | 判定 |",
          "|---|---|---|---|---|---|---|"]
    for key, pr in rep["predictions"].items():
        if pr.get("k_Q3SG") is None:
            L.append(f"| {pr['tier']} {pr['arm']} | 缺 | 缺 | | | | 不判（{pr['why']}） |")
            continue
        kq3s = "缺" if pr["k_Q3S"] is None else f"{pr['k_Q3S']}/{pr['n']}"
        ps = "" if pr["p_vs_Q3S"] is None else C._p(pr["p_vs_Q3S"])
        verdict = pr["verdict"] if pr["verdict"] is not None else f"不判（{pr['why']}）"
        L.append(f"| {pr['tier']} {pr['arm']} | {pr['k_Q3SG']}/{pr['n']} | {pr['k_Q3B']}/{pr['n']} | "
                 f"{C._p(pr['p_vs_Q3B'])} | {kq3s} | {ps} | {verdict} |")
    L += ["", "规则（读数之前登记）：Q3SG 的 k ≥ Q3B 的 k → 遗忘（缺口主要是遗忘，回放能补）；Q3SG − Q3S 不显著"
          "（p ≥ 0.05）且 Q3SG − Q3B 显著为负（p < 0.05）→ 动作任务本身；其他情况 → 介于两者之间，不判。", ""]
    titles = {"primary": "Q3SG 对 Q3B（主：100、ALL 的 c 臂；其余三格作描述）",
              "diagnostic": "Q3SG 对 Q3S（诊断：100、ALL 的 c 臂；其余三格作描述）"}
    for comp in rep["comparisons"].values():
        f, s = comp["first_label"], comp["second_label"]
        L += [f"## {titles[comp['role']]}", "",
              f"| 档 | 臂 | {f} k/n（95% CI） | {s} k/n（95% CI） | 只有 {f} 成功 b | 只有 {s} 成功 c | 精确双侧 p | "
              "(k1−k2)/n | 用途 |", "|---|---|---|---|---|---|---|---|---|"]
        L += [f"{CR._cell_row(c)} {USE[c['use']]} |" for c in comp["cells"].values()]
        L.append("")
    L += ["## 三个起点各档各臂的 k/n 和 95% Wilson 区间", "",
          "| 档 | 臂 | 起点 | k/n | 率 | 95% CI | 来源 |", "|---|---|---|---|---|---|---|"]
    for r in rep["start_table"]:
        if not r["usable"]:
            L.append(f"| {r['tier']} | {r['arm']} | {r['label']} | 缺 | | {'；'.join(r['problems'])[:200]} | "
                     f"{r['source']} |")
        else:
            L.append(f"| {r['tier']} | {r['arm']} | {r['label']} | {r['k']}/{r['n']} | {100 * r['rate']:.1f}% | "
                     f"{C._ci(r['wilson95'])} | {r['source']} |")
    odd = [f"{r['label']} {r['tier']}__{r['arm']}（{r.get('results_file') or '没有结果文件'}）"
           for r in rep["start_table"] if r["usable"] and r.get("results_file") != "results.json"]
    if odd:
        L += ["", "注：这些评测没有 results.json（score_tvr 没判为有效），数字仍取自 trajectory，见交叉核对：" + "；".join(odd)]
    cc = rep["crosscheck"]
    L += ["", "## 交叉核对（trajectory.json 对任务目录、per_task.jsonl、results.json）", "",
          "全部一致。" if cc["ok"] and not cc["not_checked"] else
          "；".join([f"{m['eval']} {m['check']} 不一致（{'；'.join(m.get('problems') or [])}）"
                    for m in cc["mismatches"]] + [f"没核对：{u}" for u in cc["not_checked"]])[:2000],
          "", f"评测框架指纹（results.code_sha256）：{', '.join(h[:12] for h in rep['harness_code_sha256'])}"]
    if rep["missing_cells"]:
        L += ["", "缺的格（拒算，不插补）：" + "；".join(rep["missing_cells"])]
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explore-root", type=Path, default=None, help="default OUT/tvr/runs/explore/rl")
    ap.add_argument("--pilot-root", type=Path, default=None, help="default OUT/tvr/runs/pilot (read only)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default EXPLORE_ROOT/COMPARE_Q3SG.json (COMPARE_Q3SG.md beside it); never under the pilot "
                         "root")
    ap.add_argument("--tasks-dir", type=Path, default=None, help="default the pinned TVRBench data/tasks")
    ap.add_argument("--prereg", type=Path, default=None, help="default vpd/PREREG_TVR_PILOT.md of this tree")
    args = ap.parse_args(argv)
    try:
        rep = compare(args.explore_root, args.pilot_root, tasks_dir=args.tasks_dir, prereg=args.prereg)
        out = args.out or Path(rep["explore_root"]) / "COMPARE_Q3SG.json"
        pr = os.path.realpath(rep["pilot_root"])
        if os.path.realpath(str(out)).startswith(pr.rstrip("/") + "/"):
            raise Refused(f"refused: {out} is under the pilot root {rep['pilot_root']} (explore outputs never go "
                          "there)")
    except Refused as r:
        print(r.message, file=sys.stderr)
        return 2
    C.write_text(out, json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    C.write_text(out.with_suffix(".md"), render_md(rep))
    cc = rep["crosscheck"]
    print(json.dumps({"out": str(out), "crosscheck_ok": cc["ok"], "used_sides_all_checked":
                      cc["used_sides_all_checked"], "not_checked": len(cc["not_checked"]),
                      "missing_cells": len(rep["missing_cells"]),
                      "predictions": {k: v["verdict"] for k, v in rep["predictions"].items()}}, ensure_ascii=False))
    if not cc["ok"] or not cc["used_sides_all_checked"]:
        return 1
    return 3 if rep["missing_cells"] else 0


if __name__ == "__main__":
    sys.exit(main())
