"""The EXPLORATORY comparison of the dose starts Q3S2 and Q3S4 (Q3S's own C0 action SFT run lf_q3formal at epochs 2 and
4: checkpoint-308 / -616, codes c0q3_sft_e2 / c0q3_sft_e4; PREREG_TVR_PILOT.md section 17, row
pilot_tvr.DOSE_ROW_DATE), on compare_rl's method (和 compare_rl、compare_q3sg 同口径).  Report only: not in the gate,
the decision table, A4 or the confirmatory round; computed by this script and rechecked by an independent one.  Not a
run_tvr CODE_FILE.  A separate output: COMPARE_RL and COMPARE_Q3SG are unchanged.

    nice -n 19 PY_EVAL -B -m vpd.tvr.compare_dose [--explore-root OUT/tvr/runs/explore/rl] [--pilot-root
        OUT/tvr/runs/pilot] [--out EXPLORE_ROOT/COMPARE_DOSE.json] [--tasks-dir REPO/data/tasks] [--prereg
        vpd/PREREG_TVR_PILOT.md]
    -> COMPARE_DOSE.json (schema vpd.tvr.compare_dose/1) and COMPARE_DOSE.md next to it.
    Exit 0 = every cell computed and every cross-check made and equal; 1 = a cross-check mismatch on any eval read, or
    a cross-check that could not be made on an eval whose numbers enter a cell; 3 = a cell missing (refused, not
    imputed); 2 = refused: no single section-17 row of DOSE_ROW_DATE (DOSE_ROW_NEEDLE) in the prereg, the registered
    eval task set is not EVAL500_N tasks, not all 10 dose evals exist and validate, or the trainer invariant fails
    (below).

Comparisons, first minus second (b = only the first start succeeds, c = only the second), on compare_rl's 5 cells
(compare_tvr.cells: 0W c, 100 c, TOP c, TOP u, TOP blind; TOP = tiers.n_top, frozen ALL; seed 1001):
    Q3S2 - Q3B, Q3S4 - Q3B   registered on 100 c and TOP c; the other three cells as description;
    Q3S2 - Q3S, Q3S4 - Q3S   registered on 100 c and TOP c; the other three cells as description.
Dose sides: EXPLORE_ROOT/<S>/evals/xrl__<S>__<tier>__<arm>; Q3B / Q3S: the existing pilot evals PILOT_ROOT/<S>/evals/
pilot__<S>__<tier>__<arm>, not rerun.  As compare_rl / compare_q3sg: paired by task_id over the registered eval tasks
(REPO/data/tasks/eval.json) with compare_tvr.read_side / pair (success = tasks/*/trajectory.json result.success; the
task_dirs / per_task / results_json cross-checks; the section-12.2 exact two-sided p; k/n and Wilson 95% (z = gates.z)
per side; (k1 - k2) / n), compare_rl.crosscheck, and a start table (compare_rl.start_table) of Q3B, Q3S, Q3S2, Q3S4 on
the 5 cells.  No multiplicity correction.  The p values cover task sampling only, not the training-seed variation (one
seed per start and tier), so they are optimistic: descriptive only, not a confirmatory result (推断).
Dose table (description): Q3S2, Q3S4 and Q3S = epochs 2, 4 and 8 of the same run, each minus Q3B on TOP c and 100 c
(the pairs above; Q3S - Q3B from the two pilot evals, the same pair as COMPARE.json's primary comparison there).
H1 (the user's prediction, verbatim in the row, written before any Q3S2 / Q3S4 eval), read off TOP c:
    monotone  d(Q3S2 - Q3B) > d(Q3S4 - Q3B) > d(Q3S - Q3B), d = (k1 - k2) / n compared as exact fractions (strict);
    light     Q3S2 - Q3B not significant (exact p >= 1/20) or positive (k(Q3S2) > k(Q3B)): the report then writes, as
              the row requires, "轻量动作训练不损害迁移，重训才损害", a correction to section 3.5, not support for the claim.
    Each is null when a pair it needs is missing.  The three d are reported ("差值从接近 0 走向 −9.2 个百分点"), not judged.
Validation (the refusal; made before any trajectory is read; it reads no success field):
  the 10 dose evals as compare_rl.validate does (results.json valid true, n_tasks EVAL500_N, no tasks_missing,
  mechanism_only false, its own start / tier / arm, the registered harness HARNESS_SHA256_PREFIX (64edd575f36b, as
  every other eval), official_check.json equal true, it served the adapter as it is on disk; that adapter complete
  with seed 1001, phase, the tiers manifest's data sha and the init chain, trained with the dose starts' registered
  trainer code: pilot_tvr.x_adapter_check);
  the trainer invariant ("5 个 .py 文件 sha 不变、starts.json 只多这两条") on each dose adapter's meta.code_files: the 5
  .py shas == the ones the pilot Q3B / Q3S adapters recorded (compare_q3sg.pilot_code_files: 0W, 100, TOP; they must
  all agree), starts.json == pilot_tvr.STARTS_JSON_SHA256_DOSE, code_sha256 starting with DOSE_TRAINER_PREFIX, and
  the pilot adapters' recorded starts.json == STARTS_JSON_SHA256_PILOT; the jobs' rule on the live file
  (pilot_tvr.dose_trainer_problems) is recorded as this_tree, not judged.
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
from vpd.tvr import compare_q3sg as CQ
from vpd.tvr import compare_rl as CR
from vpd.tvr import compare_tvr as C
from vpd.tvr import pilot_cfg, pilot_tvr

SCHEMA = "vpd.tvr.compare_dose/1"
GROUP = "dose"
ROW_DATE = pilot_tvr.DOSE_ROW_DATE
ROW_NEEDLE = pilot_tvr.DOSE_ROW_NEEDLE
DOSE_STARTS = pilot_tvr.DOSE_STARTS
COMPARISONS = tuple((s, ref, "registered") for ref in ("Q3B", "Q3S") for s in DOSE_STARTS)
TABLE_STARTS = ("Q3B", "Q3S") + DOSE_STARTS
EPOCHS = {"Q3S2": 2, "Q3S4": 4, "Q3S": 8}  # of the same 10-epoch run (lf_q3formal, 154 steps per epoch)
DOSE_ORDER = ("Q3S2", "Q3S4", "Q3S")       # the dose table and H1, by epoch
PILOT_REF_STARTS = CQ.PILOT_REF_STARTS     # Q3B, Q3S: the pilot adapters whose code_files the invariant compares
ALPHA = Fraction(5, 100)
H1_TEXT = ("H1（剂量）：ALL c 的 Q3Sn − Q3B 随轮数单调下降（第 2 轮 > 第 4 轮 > 第 8 轮，差值从接近 0 走向 −9.2 个百分点）。"
           "若第 2 轮的 Q3S2 − Q3B 不显著或为正，如实写'轻量动作训练不损害迁移，重训才损害'，这是对第 3.5 节的修正，"
           "不是对命题的支持。")
LIGHT_TEXT = "轻量动作训练不损害迁移，重训才损害"
H1_RULES = {"monotone": "d(Q3S2 - Q3B) > d(Q3S4 - Q3B) > d(Q3S - Q3B) on TOP c, d = (k1 - k2) / n as exact fractions",
            "light": "Q3S2 - Q3B on TOP c: exact p >= 1/20 or k(Q3S2) > k(Q3B)"}
NOTE_P = CR.NOTE_P
NOTE = (f"EXPLORATORY, report only (PREREG section 17, {ROW_DATE}): not in the gate, the decision table, A4 or the "
        "confirmatory round")
Refused = C.Refused


def registered_cells(cfg=None) -> List[tuple]:
    """The row's registered cells: 100 c and TOP c (TOP = tiers.n_top)."""
    return [("100", "c"), (pilot_cfg.n_top(cfg), "c")]


def dose_cells(cfg=None) -> List[tuple]:
    """The dose table's cells: TOP c (H1's), then 100 c."""
    return [(pilot_cfg.n_top(cfg), "c"), ("100", "c")]


def trainer_invariant(explore_root: Path, pilot_root: Path, cfg=None) -> dict:
    """The invariant (module docstring) on every dose adapter's meta: {"ok", "problems", "pilot_reference",
    "adapters", "this_tree"}."""
    ref = CQ.pilot_code_files(pilot_root, cfg)
    sj = pilot_tvr.STARTS_JSON
    problems = list(ref["problems"])
    py = {k: v for k, v in (ref["files"] or {}).items() if k != sj}
    adapters = {}
    for s in DOSE_STARTS:
        for tier, _init in pilot_tvr.train_tiers(s, cfg):
            p = Path(explore_root) / s / "adapters" / tier / "meta.json"
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
                if cf.get(sj) != pilot_tvr.STARTS_JSON_SHA256_DOSE:
                    bad.append(f"code_files {sj} {str(cf.get(sj))[:12]} != STARTS_JSON_SHA256_DOSE "
                               f"{pilot_tvr.STARTS_JSON_SHA256_DOSE[:12]}")
            if not str(m.get("code_sha256") or "").startswith(pilot_tvr.DOSE_TRAINER_PREFIX):
                bad.append(f"code_sha256 {str(m.get('code_sha256'))[:12]} != the dose prefix "
                           f"{pilot_tvr.DOSE_TRAINER_PREFIX}")
            adapters[f"{s}/{tier}"] = {"meta": str(p), "code_sha256": m.get("code_sha256"), "ok": not bad,
                                       "problems": bad}
            problems += [f"{s}/{tier} ({p}): {b}" for b in bad]
    try:
        fp = pilot_tvr.trainer_fingerprint()
        tree = {"trainer_code_sha256": fp.get("code_sha256"), "problems": pilot_tvr.dose_trainer_problems(fp)}
    except Exception as e:  # noqa: BLE001 - recorded, not judged
        tree = {"trainer_code_sha256": None, "problems": [f"unreadable: {e!r}"]}
    return {"ok": not problems, "problems": problems,
            "rule": "5 .py shas == the pilot Q3B / Q3S adapters' recorded ones; starts.json == "
                    "STARTS_JSON_SHA256_DOSE; the pilot adapters' starts.json == STARTS_JSON_SHA256_PILOT; code_sha256 "
                    f"starts with {pilot_tvr.DOSE_TRAINER_PREFIX}",
            "pilot_reference": {"code_files": ref["files"], "adapters": ref["adapters"],
                                "py_equals_pinned": py == pilot_tvr.TRAINER_PY_SHA256 if ref["files"] else None},
            "starts_json_sha256": {"pilot": pilot_tvr.STARTS_JSON_SHA256_PILOT,
                                   "Q3SG": pilot_tvr.STARTS_JSON_SHA256_Q3SG,
                                   "dose": pilot_tvr.STARTS_JSON_SHA256_DOSE},
            "adapters": adapters, "this_tree": tree}


def h1_readout(pairs: Dict[str, dict], tier: str, arm: str) -> dict:
    """H1 on one cell from PAIRS (start -> its pair minus Q3B, compare_tvr.pair; Q3S2, Q3S4, Q3S): the three
    differences, monotone (strict, exact fractions) and the light-training clause (module docstring)."""
    out = {"tier": tier, "arm": arm, "order": list(DOSE_ORDER), "epochs": [EPOCHS[s] for s in DOSE_ORDER],
           "rules": dict(H1_RULES), "text": H1_TEXT}
    got = {s: pairs.get(s) or {"missing": True} for s in DOSE_ORDER}
    out["differences"] = {s: None if p["missing"] else p["rate_difference"] for s, p in got.items()}
    out["differences_exact"] = {s: None if p["missing"] else p["rate_difference_exact"] for s, p in got.items()}
    if any(p["missing"] for p in got.values()):
        out["monotone_decreasing"] = None
    else:
        d = [Fraction(got[s]["rate_difference_exact"]) for s in DOSE_ORDER]
        out["monotone_decreasing"] = d[0] > d[1] > d[2]
    p2 = got["Q3S2"]
    if p2["missing"]:
        out.update(light_clause=None, light_text=None, q3s2_vs_q3b=None)
    else:
        light = Fraction(p2["p_exact"]) >= ALPHA or p2["k_first"] > p2["k_second"]
        out.update(light_clause=light, light_text=LIGHT_TEXT if light else None,
                   q3s2_vs_q3b={"k_Q3S2": p2["k_first"], "k_Q3B": p2["k_second"], "n": p2["n"], "p": p2["p"],
                                "p_exact": p2["p_exact"], "not_significant": Fraction(p2["p_exact"]) >= ALPHA,
                                "positive": p2["k_first"] > p2["k_second"]})
    return out


def compare(explore_root: Path = None, pilot_root: Path = None, cfg=None, tasks_dir: Path = None,
            prereg: Path = None, L: pilot_tvr.Layout = None) -> dict:
    """COMPARE_DOSE (module docstring)."""
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
    val = CR.validate(xroot, L, cfg, starts=DOSE_STARTS)
    if not val["ok"]:
        raise Refused("refused: not all 10 dose evals exist and validate (" + str(len(val["problems"]))
                      + " problems):\n  - " + "\n  - ".join(val["problems"][:40]))
    inv = trainer_invariant(xroot, proot, cfg)
    if not inv["ok"]:
        raise Refused("refused: the trainer invariant does not hold (5 .py shas unchanged, starts.json only gains "
                      f"Q3S2 and Q3S4; {len(inv['problems'])} problems):\n  - " + "\n  - ".join(inv["problems"][:40]))
    ids, cats = reg.pop("ids"), reg.pop("cats")
    rset = set(ids)
    z = float(pilot_cfg.get("gates.z", cfg))
    sides: Dict[str, dict] = {}

    def side(s, tier, arm):
        key = f"{s}/{tier}__{arm}"
        if key not in sides:
            root, prefix = CR.side_root(s, xroot, proot, DOSE_STARTS)
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
    dose, h1 = [], None
    for tier, arm in dose_cells(cfg):
        key, pairs = f"{tier}__{arm}", {}
        for s in DOSE_ORDER:
            pr = (comps[f"{s}-Q3B"]["cells"][key] if f"{s}-Q3B" in comps else
                  dict(C.pair(side(s, tier, arm), side("Q3B", tier, arm), ids), tier=tier, arm=arm))
            pairs[s] = pr
            dose.append(dict({k: v for k, v in pr.items() if k != "use"}, start=s, epoch=EPOCHS[s], cell=key,
                             source="explore" if s in DOSE_STARTS else "pilot"))
        if (tier, arm) == dose_cells(cfg)[0]:
            h1 = h1_readout(pairs, tier, arm)
    for s in TABLE_STARTS:
        for tier, arm in C.cells(cfg):
            side(s, tier, arm)
    table = CR.start_table(sides, C.cells(cfg), TABLE_STARTS, DOSE_STARTS)
    missing = [f"{name} {ck}" for name, comp in comps.items() for ck, c in comp["cells"].items() if c["missing"]]
    missing += [f"dose table Q3S-Q3B {r['cell']}" for r in dose if r["start"] == "Q3S" and r["missing"]]
    missing += [f"start table {r['start']} {r['tier']}__{r['arm']}" for r in table if not r["usable"]]
    harness = sorted({str((pilot_tvr.read_json(Path(sd["dir"]) / "results.json") or {}).get("code_sha256"))
                      for sd in sides.values()})
    return {"schema": SCHEMA, "utc": pilot_tvr.utc(), "exploratory": True, "explore_root": str(xroot),
            "pilot_root": str(proot), "note": NOTE, "note_p": NOTE_P, "multiplicity_correction": "none", "z": z,
            "prereg": {"file": str(prereg), "sha256": C.sha256_file(prereg), "commit": commit, "section": 17,
                       "row_text": row, "row_sha256": C.sha256_bytes(row.encode())},
            "validation": val, "trainer_invariant": inv, "harness_code_sha256": harness, "registered_tasks": reg,
            "cells": [f"{t}__{a}" for t, a in C.cells(cfg)], "registered_cells": [f"{t}__{a}" for t, a in regc],
            "comparisons": comps, "epochs": dict(EPOCHS), "dose_table": dose, "h1": h1,
            "table_starts": list(TABLE_STARTS), "start_table": table,
            "inputs": {sd["dir"]: sd["inputs"] for sd in sides.values()},
            "trajectories_digest_rule": C.TRAJ_DIGEST_RULE, "crosscheck": CR.crosscheck(sides),
            "missing_cells": missing}


# --- COMPARE_DOSE.md ----------------------------------------------------------------------------------------------

USE = {"registered": "登记", "description": "描述"}
YES_NO = {True: "是", False: "否", None: "缺（不判）"}


def _pp(x) -> str:
    return "缺" if x is None else f"{100 * x:+.1f}"


def render_md(rep: dict) -> str:
    n = rep["registered_tasks"]["n"]
    h1 = rep["h1"]
    L = ["# TVR 探索：剂量起点 Q3S2、Q3S4 和 Q3B、Q3S 的比较（EXPLORATORY，只报告）", "",
         f"- 依据：PREREG_TVR_PILOT.md 第 17 节 {ROW_DATE} 的登记行（sha256 {rep['prereg']['row_sha256'][:12]}）；"
         "探索性质，不进闸门、决策表、A4 和确认轮；和 compare_rl、compare_q3sg 同口径；不做多重比较校正。",
         "- **这些 p 只算了题目抽样，没算训练种子之间的波动（每个起点每档只有一个种子），所以偏乐观，只作描述，"
         "不是确认轮的结论（推断）。**",
         f"- 成对：同一批 {n} 道登记评测题按 task_id 配对；成功取 trajectory.json 的 result.success；"
         "p = min(1, 2·P(X ≤ min(b, c)))，X ~ Bin(b + c, 1/2)（第 12 节第 2 条）；b = 只有前一个起点成功，"
         "c = 只有后一个起点成功。",
         "- Q3S2、Q3S4 是 Q3S 同一次 C0 动作 SFT（10 轮，每轮 154 步）的第 2、4 轮检查点（checkpoint-308、-616），"
         "Q3S 是第 8 轮（checkpoint-1232）；Q3S2、Q3S4 的评测在 `runs/explore/rl/<起点>`（前缀 xrl）；Q3B、Q3S 用已有的"
         "试点评测（前缀 pilot），不重跑。",
         f"- 训练器：Q3S2、Q3S4 适配器的指纹以 {pilot_tvr.DOSE_TRAINER_PREFIX} 开头；5 个 .py 文件的 sha 和试点 Q3B、Q3S "
         "适配器记录的相同，starts.json 只多 Q3S2、Q3S4 这两条（核对通过才算）。", ""]
    L += ["## H1（剂量，读数之前登记，原文）", "", f"> {H1_TEXT}", "",
          f"| 起点 | 轮数 | {h1['tier']} {h1['arm']} 起点 k/n | Q3B k/n | 差值（个百分点） | 精确双侧 p |",
          "|---|---|---|---|---|---|"]
    for r in rep["dose_table"]:
        if r["cell"] != f"{h1['tier']}__{h1['arm']}":
            continue
        if r["missing"]:
            L.append(f"| {r['start']} | {r['epoch']} | 缺 | 缺 | | |")
        else:
            L.append(f"| {r['start']} | {r['epoch']} | {r['k_first']}/{r['n']} | {r['k_second']}/{r['n']} | "
                     f"{_pp(r['rate_difference'])} | {C._p(r['p'])} |")
    light = h1["light_clause"]
    L += ["", f"- 单调下降（第 2 轮 > 第 4 轮 > 第 8 轮，差值按精确分数比，严格）：{YES_NO[h1['monotone_decreasing']]}",
          f"- 第 2 轮的 Q3S2 − Q3B 不显著（精确 p ≥ 0.05）或为正（Q3S2 的 k > Q3B 的 k）：{YES_NO[light]}"
          + (f"。按登记如实写：**{LIGHT_TEXT}**——这是对第 3.5 节的修正，不是对命题的支持。" if light else ""),
          "- 差值是否\"从接近 0 走向 −9.2 个百分点\"只照报三个差值，不判。", ""]
    L += ["## 剂量表（描述：同一次训练的第 2、4、8 轮，各减 Q3B）", "",
          "| 格 | 起点 | 轮数 | 起点 k/n（95% CI） | Q3B k/n（95% CI） | 只有起点成功 b | 只有 Q3B 成功 c | 精确双侧 p | "
          "(k1−k2)/n | 来源 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["dose_table"]:
        t, a = r["cell"].split("__")
        if r["missing"]:
            L.append(f"| {t} {a} | {r['start']} | {r['epoch']} | 缺 | 缺 | | | | {'；'.join(r['why'])[:200]} | "
                     f"{r['source']} |")
        else:
            L.append(f"| {t} {a} | {r['start']} | {r['epoch']} | {r['k_first']}/{r['n']}（{C._ci(r['wilson95_first'])}"
                     f"） | {r['k_second']}/{r['n']}（{C._ci(r['wilson95_second'])}） | {r['b']} | {r['c']} | "
                     f"{C._p(r['p'])} | {_pp(r['rate_difference'])} 个百分点 | {r['source']} |")
    L.append("")
    for comp in rep["comparisons"].values():
        f, s = comp["first_label"], comp["second_label"]
        L += [f"## {f} 对 {s}（登记：100、ALL 的 c 臂；其余三格作描述）", "",
              f"| 档 | 臂 | {f} k/n（95% CI） | {s} k/n（95% CI） | 只有 {f} 成功 b | 只有 {s} 成功 c | 精确双侧 p | "
              "(k1−k2)/n | 用途 |", "|---|---|---|---|---|---|---|---|---|"]
        L += [f"{CR._cell_row(c)} {USE[c['use']]} |" for c in comp["cells"].values()]
        L.append("")
    L += ["## 四个起点各档各臂的 k/n 和 95% Wilson 区间", "",
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
                    help="default EXPLORE_ROOT/COMPARE_DOSE.json (COMPARE_DOSE.md beside it); never under the pilot "
                         "root")
    ap.add_argument("--tasks-dir", type=Path, default=None, help="default the pinned TVRBench data/tasks")
    ap.add_argument("--prereg", type=Path, default=None, help="default vpd/PREREG_TVR_PILOT.md of this tree")
    args = ap.parse_args(argv)
    try:
        rep = compare(args.explore_root, args.pilot_root, tasks_dir=args.tasks_dir, prereg=args.prereg)
        out = args.out or Path(rep["explore_root"]) / "COMPARE_DOSE.json"
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
                      "h1": {"monotone_decreasing": rep["h1"]["monotone_decreasing"],
                             "light_clause": rep["h1"]["light_clause"]}}, ensure_ascii=False))
    if not cc["ok"] or not cc["used_sides_all_checked"]:
        return 1
    return 3 if rep["missing_cells"] else 0


if __name__ == "__main__":
    sys.exit(main())
