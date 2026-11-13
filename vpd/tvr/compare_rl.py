"""The EXPLORATORY comparison of the RL starts (PREREG_TVR_PILOT.md section 17, row 2026-11-12; DECISIONS G18).  Report
only: not in the gate, the decision table, A4 or the confirmatory round; computed by this script and rechecked by an
independent one.  Not a run_tvr CODE_FILE.

    nice -n 19 PY_EVAL -B -m vpd.tvr.compare_rl [--explore-root OUT/tvr/runs/explore/rl] [--pilot-root
        OUT/tvr/runs/pilot] [--out EXPLORE_ROOT/COMPARE_RL.json] [--tasks-dir REPO/data/tasks] [--prereg
        vpd/PREREG_TVR_PILOT.md]
    -> COMPARE_RL.json (schema vpd.tvr.compare_rl/1) and COMPARE_RL.md next to it.
    Exit 0 = every cell computed and every cross-check made and equal; 1 = a cross-check mismatch on any eval read, or
    a cross-check that could not be made on an eval whose numbers enter a cell; 3 = a cell missing (refused, not
    imputed); 2 = refused: no single section-17 row of 2026-11-12 in the prereg, the registered eval task set is not
    EVAL500_N tasks, or not all 10 explore evals exist and validate (below).

Comparisons, first minus second (b = only the first start succeeds, c = only the second): Q3RD - Q3B and Q3RS150 -
Q3B (primary), Q3RD - Q3S and Q3RS150 - Q3S (supplement: how much RL recovers relative to SFT).  Reports call Q3RS150
"Q3RS (Q3RS150)" (the starts.json key is kept: renaming it would change the trainer fingerprint).  Cells (compare_tvr
.cells): 0W c, 100 c, TOP c, TOP u, TOP blind (TOP = tiers.n_top, frozen ALL), all seed 1001.  RL sides: the explore
evals EXPLORE_ROOT/<S>/evals/xrl__<S>__<tier>__<arm>; Q3B / Q3S sides: the existing pilot evals PILOT_ROOT/<S>/evals/
pilot__<S>__<tier>__<arm>, not rerun.  Per cell the two evals are paired by task_id over the registered eval tasks
(REPO/data/tasks/eval.json) with compare_tvr's own read_side / pair: success = tasks/*/trajectory.json
result.success; the task_dirs / per_task / results_json cross-checks; the section-12.2 exact two-sided p; k/n and
Wilson 95% (z = gates.z) per side; (k1 - k2) / n.  The five-start table: k/n and Wilson of every start, tier and arm
in the order Q3B, Q3S, E2, Q3RD, Q3RS150 (E2's rows from its pilot evals, absolute k/n only: no comparison with E2).
No multiplicity correction.  The p values cover task sampling only, not the training-seed variation (one seed per
start and tier), so they are optimistic: descriptive only, not a confirmatory result (推断).
Validation (the refusal; made before any trajectory is read, and it reads no success field): each of the 10 explore
evals (Q3RD, Q3RS150 x the 5 cells) has a results.json with valid true, n_tasks EVAL500_N, no tasks_missing,
mechanism_only false, its own start / tier / arm, the registered harness (code_sha256 starting with
pilot_tvr.HARNESS_SHA256_PREFIX) and an official_check.json with equal true; it served the explore adapter as it is on
disk (eval-ok), and that adapter is complete with seed 1001 and the right init chain (pilot_tvr.x_adapter_check:
start, seed, phase, data_sha256 == the tiers manifest's, the RL starts' registered trainer code: recipe prefix
0865dd75f1ee, also on a tree whose starts.json has the Q3SG entry, DECISIONS G20).
validate, side_root, crosscheck and start_table take the start lists as arguments (defaults: this comparison's);
vpd.tvr.compare_q3sg (row 2026-11-13) reuses them for Q3SG.  COMPARE_RL's output does not depend on that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

from vpd import paths
from vpd.tvr import compare_tvr as C
from vpd.tvr import pilot_cfg, pilot_tvr
from vpd.walkerbench.pilot import eval_check

SCHEMA = "vpd.tvr.compare_rl/1"
ROW_DATE = pilot_tvr.XROW_DATE
ROW_NEEDLE = pilot_tvr.XROW_NEEDLE
RL_STARTS = pilot_tvr.RL_STARTS  # not EXPLORE_STARTS: that also holds Q3SG (row 2026-11-13, compare_q3sg)
COMPARISONS = (("Q3RD", "Q3B", "primary"), ("Q3RS150", "Q3B", "primary"), ("Q3RD", "Q3S", "supplement"),
               ("Q3RS150", "Q3S", "supplement"))
FIVE_STARTS = ("Q3B", "Q3S", "E2", "Q3RD", "Q3RS150")
LABEL = {"Q3RS150": "Q3RS (Q3RS150)"}
NOTE_P = ("p covers task sampling only, not the training-seed variation (one seed per start and tier): optimistic, "
          "descriptive only, not a confirmatory result (推断); no multiplicity correction")
NOTE = ("EXPLORATORY, report only (PREREG section 17, 2026-11-12): not in the gate, the decision table, A4 or the "
        "confirmatory round")
Refused = C.Refused


def label(s: str) -> str:
    return LABEL.get(s, s)


def side_root(s: str, explore_root: Path, pilot_root: Path, explore_starts=RL_STARTS) -> tuple:
    """(root, eval dir prefix) of a start's evals: the explore starts (default the RL starts) from the explore root
    (xrl), the others from the pilot (pilot)."""
    return (Path(explore_root), pilot_tvr.XRL_PREFIX) if s in explore_starts else (Path(pilot_root), "pilot")


def validate(explore_root: Path, L: pilot_tvr.Layout = None, cfg=None, starts=RL_STARTS) -> dict:
    """The explore evals of STARTS (default the RL starts: 10 evals) and their adapters (module docstring); {"ok",
    "problems", "evals": {key: facts}}."""
    L = L or pilot_tvr.LAYOUT
    root = Path(explore_root)
    problems, evals = [], {}
    eval_adapter = pilot_tvr.EVAL_ADAPTER
    for s in starts:
        for tier, init in pilot_tvr.train_tiers(s, cfg):
            rc, msg = pilot_tvr.x_adapter_check(s, tier, init, L, cfg, adapter_root=root)
            if rc != 0:
                problems.append(f"adapter {s}/{tier}: {msg}")
        for tier, arm in C.cells(cfg):
            d = C.eval_dir(root, s, tier, arm, pilot_tvr.XRL_PREFIX)
            key = f"{s}/{tier}__{arm}"
            ok, det = pilot_tvr._valid_results(d, pilot_tvr.EVAL500_N)
            r = pilot_tvr.read_json(d / "results.json") or {}
            bad = [] if ok else [f"results.json not valid: {det}"]
            if r:
                if r.get("mechanism_only") is not False:
                    bad.append(f"mechanism_only {r.get('mechanism_only')!r}")
                want = {"start": s, "tier": tier, "arm": arm}
                bad += [f"{k} {r.get(k)!r} != {v!r}" for k, v in want.items() if r.get(k) != v]
                if not str(r.get("code_sha256") or "").startswith(pilot_tvr.HARNESS_SHA256_PREFIX):
                    bad.append(f"harness code_sha256 {str(r.get('code_sha256'))[:12]} != "
                               f"{pilot_tvr.HARNESS_SHA256_PREFIX}")
            ad = root / s / "adapters" / eval_adapter[tier]
            erc, emsg = eval_check(d, ad)
            if erc != 0:
                bad.append(f"served adapter: {emsg}")
            oc = pilot_tvr.read_json(d / "official_check.json")
            if not (oc and oc.get("equal") is True):
                bad.append(f"official_check.json equal {None if not oc else oc.get('equal')!r}")
            evals[key] = {"dir": str(d), "ok": not bad, "problems": bad,
                          "results_json_sha256": C.sha256_file(d / "results.json"),
                          "code_sha256": r.get("code_sha256"),
                          "adapter_sha256": ((r.get("adapter") or {}).get("adapter_sha256"))}
            problems += [f"{key} ({d.name}): {b}" for b in bad]
    return {"ok": not problems, "problems": problems, "evals": evals}


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
    row = pilot_tvr.xrow(text)
    if row is None:
        raise Refused(f"refused: {prereg} has no single section-17 row '| {ROW_DATE} | ... {ROW_NEEDLE} ...'")
    try:
        commit = pilot_tvr.prereg_state().get("commit")
    except Exception:  # noqa: BLE001 - git may be absent; the file sha256 is recorded anyway
        commit = None
    reg = C.registered_tasks(Path(tasks_dir or paths.TVR_REPO / "data" / "tasks"))
    val = validate(xroot, L, cfg)
    if not val["ok"]:
        raise Refused("refused: not all 10 explore evals exist and validate (" + str(len(val["problems"]))
                      + " problems):\n  - " + "\n  - ".join(val["problems"][:40]))
    ids, cats = reg.pop("ids"), reg.pop("cats")
    rset = set(ids)
    z = float(pilot_cfg.get("gates.z", cfg))
    sides: Dict[str, dict] = {}

    def side(s, tier, arm):
        key = f"{s}/{tier}__{arm}"
        if key not in sides:
            root, prefix = side_root(s, xroot, proot)
            sides[key] = C.read_side(root, s, tier, arm, rset, z, cats, prefix=prefix)
        return sides[key]

    comps = {}
    for first, second, role in COMPARISONS:
        cs = {}
        for tier, arm in C.cells(cfg):
            cs[f"{tier}__{arm}"] = dict(C.pair(side(first, tier, arm), side(second, tier, arm), ids), tier=tier,
                                        arm=arm)
        comps[f"{first}-{second}"] = {"role": role, "first": first, "second": second, "first_label": label(first),
                                      "second_label": label(second), "b_is": f"only {label(first)} succeeds",
                                      "c_is": f"only {label(second)} succeeds", "cells": cs}
    for s in FIVE_STARTS:  # E2: absolute k/n only (its sides are read here, no comparison uses them)
        for tier, arm in C.cells(cfg):
            side(s, tier, arm)
    five = start_table(sides, C.cells(cfg), FIVE_STARTS)
    missing = [f"{name} {ck}" for name, comp in comps.items() for ck, c in comp["cells"].items() if c["missing"]]
    missing += [f"five-start table {r['start']} {r['tier']}__{r['arm']}" for r in five if not r["usable"]]
    harness = sorted({str((pilot_tvr.read_json(Path(sd["dir"]) / "results.json") or {}).get("code_sha256"))
                      for sd in sides.values()})
    return {"schema": SCHEMA, "utc": pilot_tvr.utc(), "exploratory": True, "explore_root": str(xroot),
            "pilot_root": str(proot), "note": NOTE, "note_p": NOTE_P, "multiplicity_correction": "none", "z": z,
            "prereg": {"file": str(prereg), "sha256": C.sha256_file(prereg), "commit": commit, "section": 17,
                       "row_text": row, "row_sha256": C.sha256_bytes(row.encode())},
            "validation": val, "harness_code_sha256": harness, "registered_tasks": reg,
            "cells": [f"{t}__{a}" for t, a in C.cells(cfg)], "comparisons": comps, "five_starts": list(FIVE_STARTS),
            "five_start_table": five, "inputs": {sd["dir"]: sd["inputs"] for sd in sides.values()},
            "trajectories_digest_rule": C.TRAJ_DIGEST_RULE, "crosscheck": crosscheck(sides),
            "missing_cells": missing}


def start_table(sides: Dict[str, dict], cells, starts, explore_starts=RL_STARTS) -> list:
    """k/n and Wilson of every start of STARTS on every cell (cell-major), from the sides already read; source
    explore for the starts in explore_starts (default the RL starts), else pilot."""
    rows = []
    for tier, arm in cells:
        for s in starts:
            sd = sides[f"{s}/{tier}__{arm}"]
            rows.append(dict({"start": s, "label": label(s), "tier": tier, "arm": arm, "source":
                              "explore" if s in explore_starts else "pilot", "results_file": sd["results_file"]},
                             **({"usable": True, "k": sd["k"], "n": sd["n"], "rate": sd["rate"],
                                 "wilson95": sd["wilson95"]} if sd["usable"] else
                                {"usable": False, "problems": sd["problems"]})))
    return rows


def crosscheck(sides: Dict[str, dict]) -> dict:
    """The cross-checks of every side read (compare_tvr.read_side): mismatches (named "S/tier__arm" and by the eval dir
    name), checks that could not be made, and whether every usable side (whose numbers enter a cell) was checked."""
    mism, unknown, unknown_used = [], [], []
    for key, sd in sides.items():
        for name, c in sd["crosscheck"].items():
            if c.get("ok") is False:
                mism.append({"eval": key, "eval_dir": Path(sd["dir"]).name, "check": name,
                             **{k: v for k, v in c.items() if k != "ok"}})
            elif c.get("ok") is None:
                unknown.append(f"{key}: {name}: {c.get('why')} ({Path(sd['dir']).name})")
                if sd["usable"]:
                    unknown_used.append(unknown[-1])
    return {"ok": not mism, "mismatches": mism, "not_checked": unknown, "used_sides_all_checked": not unknown_used}


# --- COMPARE_RL.md ------------------------------------------------------------------------------------------------

def _cell_row(c: dict) -> str:
    if c["missing"]:
        return f"| {c['tier']} | {c['arm']} | 缺 | 缺 | | | | {'；'.join(c['why'])[:200]} |"
    return (f"| {c['tier']} | {c['arm']} | {c['k_first']}/{c['n']}（{C._ci(c['wilson95_first'])}） | "
            f"{c['k_second']}/{c['n']}（{C._ci(c['wilson95_second'])}） | {c['b']} | {c['c']} | {C._p(c['p'])} | "
            f"{100 * c['rate_difference']:+.1f} 个百分点 |")


def render_md(rep: dict) -> str:
    L = ["# TVR 探索：RL 起点和 Q3B、Q3S 的比较（EXPLORATORY，只报告）", "",
         f"- 依据：PREREG_TVR_PILOT.md 第 17 节 {ROW_DATE} 的登记行（sha256 {rep['prereg']['row_sha256'][:12]}）；"
         "探索性质，不进闸门、决策表、A4 和确认轮；不做多重比较校正。",
         "- **这些 p 只算了题目抽样，没算训练种子之间的波动（每个起点每档只有一个种子），所以偏乐观，只作描述，"
         "不是确认轮的结论（推断）。**",
         f"- 成对：同一批 {rep['registered_tasks']['n']} 道登记评测题按 task_id 配对；成功取 trajectory.json 的 "
         "result.success；p = min(1, 2·P(X ≤ min(b, c)))，X ~ Bin(b + c, 1/2)（第 12 节第 2 条）；b = 只有前一个起点成功，"
         "c = 只有后一个起点成功。",
         "- RL 起点的评测在 `runs/explore/rl`（前缀 xrl）；Q3B、Q3S、E2 用已有的试点评测（前缀 pilot），不重跑。"
         "Q3RS 即 starts.json 的 Q3RS150。", ""]
    groups = (("RL 对 Q3B（主报告）", "Q3B"), ("RL 对 Q3S（补充：RL 相对 SFT 修回多少）", "Q3S"))
    for title, second in groups:
        L += [f"## {title}", ""]
        for name, comp in rep["comparisons"].items():
            if comp["second"] != second:
                continue
            f, s = comp["first_label"], comp["second_label"]
            L += [f"### {f} 对 {s}", "",
                  f"| 档 | 臂 | {f} k/n（95% CI） | {s} k/n（95% CI） | 只有 {f} 成功 b | 只有 {s} 成功 c | 精确双侧 p | "
                  "(k1−k2)/n |", "|---|---|---|---|---|---|---|---|"]
            L += [_cell_row(c) for c in comp["cells"].values()]
            L.append("")
    L += ["## 五个起点各档各臂的 k/n 和 95% Wilson 区间（E2 只列绝对数，不和 RL 比）", "",
          "| 档 | 臂 | 起点 | k/n | 率 | 95% CI | 来源 |", "|---|---|---|---|---|---|---|"]
    for r in rep["five_start_table"]:
        if not r["usable"]:
            L.append(f"| {r['tier']} | {r['arm']} | {r['label']} | 缺 | | {'；'.join(r['problems'])[:200]} | "
                     f"{r['source']} |")
        else:
            L.append(f"| {r['tier']} | {r['arm']} | {r['label']} | {r['k']}/{r['n']} | {100 * r['rate']:.1f}% | "
                     f"{C._ci(r['wilson95'])} | {r['source']} |")
    odd = [f"{r['label']} {r['tier']}__{r['arm']}（{r.get('results_file') or '没有结果文件'}）"
           for r in rep["five_start_table"] if r["usable"] and r.get("results_file") != "results.json"]
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
                    help="default EXPLORE_ROOT/COMPARE_RL.json (COMPARE_RL.md beside it); never under the pilot root")
    ap.add_argument("--tasks-dir", type=Path, default=None, help="default the pinned TVRBench data/tasks")
    ap.add_argument("--prereg", type=Path, default=None, help="default vpd/PREREG_TVR_PILOT.md of this tree")
    args = ap.parse_args(argv)
    try:
        rep = compare(args.explore_root, args.pilot_root, tasks_dir=args.tasks_dir, prereg=args.prereg)
        out = args.out or Path(rep["explore_root"]) / "COMPARE_RL.json"
        pr = os.path.realpath(rep["pilot_root"])
        if os.path.realpath(str(out)).startswith(pr.rstrip("/") + "/"):
            raise Refused(f"refused: {out} is under the pilot root {rep['pilot_root']} (explore outputs never go there)")
    except Refused as r:
        print(r.message, file=sys.stderr)
        return 2
    C.write_text(out, json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    C.write_text(out.with_suffix(".md"), render_md(rep))
    cc = rep["crosscheck"]
    print(json.dumps({"out": str(out), "crosscheck_ok": cc["ok"], "used_sides_all_checked":
                      cc["used_sides_all_checked"], "not_checked": len(cc["not_checked"]),
                      "missing_cells": len(rep["missing_cells"])}))
    if not cc["ok"] or not cc["used_sides_all_checked"]:
        return 1
    return 3 if rep["missing_cells"] else 0


if __name__ == "__main__":
    sys.exit(main())
