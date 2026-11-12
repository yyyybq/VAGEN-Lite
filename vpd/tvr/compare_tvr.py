"""The registered report-only comparison of the TVR pilot (PREREG_TVR_PILOT.md section 17, row 2026-11-11 "登记主报告的
比较 Q3S − Q3B"; DECISIONS G16 (2)).  Not part of the gate, the decision table or A4; computed only after GATE.json and
POWER.json exist, and rechecked by an independent script.

    nice -n 19 PY_EVAL -B -m vpd.tvr.compare_tvr [--pilot-root OUT/tvr/runs/pilot] [--out P/COMPARE.json]
        [--tasks-dir REPO/data/tasks] [--prereg vpd/PREREG_TVR_PILOT.md]
    -> COMPARE.json (schema vpd.tvr.compare/1) and COMPARE.md next to it.
    Exit 0 = every cell computed and every cross-check made and equal; 1 = a cross-check mismatch on any eval read,
    or a cross-check that could not be made on an eval whose numbers enter a cell (e.g. no results.json, only
    results_invalid.json; written anyway, numbers from the trajectories); 3 = a cell missing (refused, not imputed);
    2 = refused (no P/GATE.json or P/POWER.json, no registered row in the prereg, or the registered eval task set is
    not EVAL500_N tasks).

Cells: 0W c, 100 c, TOP c, TOP u, TOP blind (TOP = pilot_cfg tiers.n_top, frozen ALL), all seed 1001.
Comparisons: Q3S - Q3B (primary) and Q3S - E2 (supplement), first minus second; reference row: Q3B 100 c seed 1001 vs
the replicate 100s1002 (seed 1002).  Per cell the two evals are paired by task_id over the registered eval tasks
(REPO/data/tasks/eval.json): success = tasks/*/trajectory.json result.success (G0(17)'s source; also cross-checked,
every mismatch reported under crosscheck with the eval dir name: task_dirs = every tasks/<dir>/ holds a
trajectory.json whose task_id is <dir>; per_task = one per_task.jsonl row per task (a duplicate row is a mismatch,
whatever its value) whose success equals result.success; results_json = results.json sr k/n and by_category sr k/n
equal the trajectories'; a check that cannot be made, e.g. no results.json, is listed in not_checked).  A side whose
trajectories are not exactly the registered tasks (one file each, result.success a bool) makes the cell missing.
Reported: k/n and Wilson 95% (z = gates.z = 1.96) per side; b = only the first succeeds, c = only the second; the
section-12.2 exact two-sided p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2), 1 if b + c = 0 (also as an exact
fraction); (k1 - k2) / n.  No multiplicity correction.  The p values cover task sampling only, not the training-seed
variation (one seed per start and tier), so they are optimistic: descriptive only, not a confirmatory result (推断).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from vpd import paths
from vpd.tvr import pilot_cfg, pilot_tvr
from vpd.walkerbench.pilot_gate import wilson_lb, wilson_ub

SCHEMA = "vpd.tvr.compare/1"
ROW_DATE = "2026-11-11"
ROW_NEEDLE = "登记主报告的比较 Q3S − Q3B"
COMPARISONS = (("Q3S", "Q3B", "primary"), ("Q3S", "E2", "supplement"))
TRAJ_DIGEST_RULE = ("sha256 of the lines '<path relative to the eval dir, posix> <sha256 of the file>\\n' sorted by "
                    "path, over tasks/*/trajectory.json")
NOTE_P = ("p covers task sampling only, not the training-seed variation (one seed per start and tier): optimistic, "
          "descriptive only, not a confirmatory result (推断); no multiplicity correction")
NOTE = "report only (PREREG section 17, 2026-11-11): not in the gate, the decision table or A4"
OTHER_RESULTS = ("results_invalid.json", "results_partial.json")  # what score_tvr writes instead of results.json


class Refused(SystemExit):
    def __init__(self, message: str):
        super().__init__(2)
        self.message = message


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> Optional[str]:
    try:
        return sha256_bytes(Path(p).read_bytes())
    except OSError:
        return None


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def exact_p(b: int, c: int) -> Fraction:
    """PREREG 12.2: p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2); 1 when b + c = 0."""
    n = b + c
    if n == 0:
        return Fraction(1)
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2 ** n))


def wilson(k: int, n: int, z: float) -> List[float]:
    return [wilson_lb(k, n, z), wilson_ub(k, n, z)]


def cells(cfg=None) -> List[tuple]:
    top = pilot_cfg.n_top(cfg)
    return [("0W", "c"), ("100", "c"), (top, "c"), (top, "u"), (top, "blind")]


def reference(cfg=None) -> dict:
    rep = pilot_tvr.replicate(cfg)
    return {"start": rep["start"], "arm": "c", "first": {"tier": str(rep["tier"]), "seed": int(pilot_cfg.SEED)},
            "second": {"tier": f"{rep['tier']}s{rep['seed']}", "seed": int(rep["seed"])}}


def eval_dir(root: Path, s: str, tier: str, arm: str, prefix: str = "pilot") -> Path:
    """ROOT/<S>/evals/<prefix>__<S>__<tier>__<arm>; prefix pilot (this comparison), xrl (vpd.tvr.compare_rl's RL
    sides under OUT/tvr/runs/explore/rl)."""
    return Path(root) / s / "evals" / f"{prefix}__{s}__{tier}__{arm}"


def prereg_row(text: str) -> Optional[str]:
    rows = [ln for ln in text.splitlines() if ln.startswith(f"| {ROW_DATE} |") and ROW_NEEDLE in ln]
    return rows[0] if len(rows) == 1 else None


def registered_tasks(tasks_dir: Path) -> dict:
    p = Path(tasks_dir) / "eval.json"
    doc = read_json(p)
    tasks = doc if isinstance(doc, list) else (doc or {}).get("tasks") if isinstance(doc, dict) else None
    if not isinstance(tasks, list):
        raise Refused(f"no registered eval task list {p}")
    ids = [t.get("task_id") for t in tasks if isinstance(t, dict)]
    if len(ids) != pilot_tvr.EVAL500_N or len(set(ids)) != len(ids) or None in ids:
        raise Refused(f"{p}: {len(ids)} task ids ({len(set(ids))} distinct), want {pilot_tvr.EVAL500_N}")
    frozen = pilot_cfg.FROZEN["tvrbench"]["files_sha256"].get("data/tasks/eval.json")
    sha = sha256_file(p)
    cats = {}  # task_id -> score_tvr's category (results.json by_category), for the cross-check only
    for t in tasks:
        try:
            cats[t["task_id"]] = pilot_cfg.category_of(t)
        except (KeyError, TypeError):
            pass
    return {"file": str(p), "sha256": sha, "frozen_sha256": frozen, "matches_frozen": sha == frozen,
            "n": len(ids), "ids_sha256": sha256_bytes("\n".join(sorted(ids)).encode()), "ids": ids, "cats": cats}


# --- one eval ---------------------------------------------------------------------------------------------------

def read_side(root: Path, s: str, tier: str, arm: str, registered: set, z: float, cats: dict = None,
              prefix: str = "pilot") -> dict:
    """One eval's successes from its trajectories, its input shas and the cross-checks; usable iff its trajectories
    are exactly the registered tasks, one file each, with a bool result.success.  cats: task_id -> category of the
    registered tasks (results.json by_category cross-check).  prefix: the eval dir prefix (eval_dir)."""
    d = eval_dir(root, s, tier, arm, prefix)
    out = {"start": s, "tier": tier, "arm": arm, "dir": str(d), "problems": [], "usable": False}
    td = d / "tasks"
    files = sorted(td.glob("*/trajectory.json")) if td.is_dir() else []
    success: Dict[str, bool] = {}
    lines, dup, nonbool, unreadable, misnamed = [], [], [], [], []
    for p in files:
        raw = p.read_bytes()
        lines.append(f"{p.relative_to(d).as_posix()} {sha256_bytes(raw)}\n")
        try:
            t = json.loads(raw)
        except ValueError:
            unreadable.append(p.parent.name)
            continue
        tid = t.get("task_id") if isinstance(t, dict) else None
        res = (t.get("result") if isinstance(t, dict) else None) or {}
        ok = res.get("success") if isinstance(res, dict) else None
        if isinstance(tid, str) and tid != p.parent.name:
            misnamed.append(p.parent.name)
        if tid in success:
            dup.append(tid)
        if not isinstance(ok, bool):
            nonbool.append(tid)
        success[tid] = ok is True
    out["inputs"] = {"results_json_sha256": sha256_file(d / "results.json"),
                     **{f"{x.replace('.', '_')}_sha256": sha256_file(d / x) for x in OTHER_RESULTS},
                     "per_task_jsonl_sha256": sha256_file(d / "per_task.jsonl"),
                     "n_trajectory_files": len(files),
                     "trajectories_digest": sha256_bytes("".join(sorted(lines)).encode()) if files else None}
    probs = out["problems"]
    if not files:
        probs.append("no trajectories")
    if unreadable:
        probs.append(f"{len(unreadable)} unreadable trajectory.json, e.g. {unreadable[:3]}")
    if dup:
        probs.append(f"{len(dup)} task ids with two trajectories, e.g. {dup[:3]}")
    if nonbool:
        probs.append(f"{len(nonbool)} trajectories without a bool result.success, e.g. {nonbool[:3]}")
    ids = set(success)
    extra, lack = sorted(map(str, ids - registered)), sorted(registered - ids)
    if files and (extra or lack):
        probs.append(f"not the registered tasks: {len(extra)} unregistered (e.g. {extra[:2]}), {len(lack)} "
                     f"registered missing (e.g. {lack[:2]})")
    if files and len(files) != pilot_tvr.EVAL500_N:
        probs.append(f"{len(files)} trajectory files, want {pilot_tvr.EVAL500_N}")
    out["usable"] = not probs
    k, n = sum(success.values()), len(success)
    if out["usable"]:
        out.update(k=k, n=n, rate=k / n, wilson95=wilson(k, n, z))
    # cross-checks (reported; the numbers above come from the trajectories).  Each check: ok (None = could not be
    # made, with why) and its problems as text
    cc = {}
    if td.is_dir():
        empty = sorted(x.name for x in td.iterdir() if x.is_dir() and not (x / "trajectory.json").is_file())
        tp = ([f"{len(empty)} task dir(s) without a trajectory.json, e.g. {empty[:3]}"] if empty else []) + \
             ([f"{len(misnamed)} trajectory.json whose task_id is not its dir name, e.g. dirs {sorted(misnamed)[:3]}"]
              if misnamed else [])
        cc["task_dirs"] = {"ok": not tp, "n_dirs_without_trajectory": len(empty), "n_misnamed": len(misnamed),
                           "problems": tp}
    r = read_json(d / "results.json")
    other = [x for x in OTHER_RESULTS if (d / x).is_file()]
    out["results_file"] = "results.json" if r is not None else (other[0] if other else None)
    if r is None:
        why = "no results.json" if not (d / "results.json").exists() else "results.json unreadable"
        if other:
            why += f" ({', '.join(other)} present: score_tvr did not score this eval as valid)"
        cc["results_json"] = {"ok": None, "why": why}
    else:
        sr = r.get("sr") if isinstance(r.get("sr"), dict) else {}
        rp = [] if (sr.get("k"), sr.get("n")) == (k, n) else [f"sr {sr.get('k')!r}/{sr.get('n')!r} != the "
                                                               f"trajectories' {k}/{n}"]
        byc = r.get("by_category")
        if isinstance(byc, dict):  # score_tvr: per category k = successes, n = the scored tasks of the category
            mine: Dict[str, list] = {}
            for t, ok in success.items():
                c = (cats or {}).get(t)
                if c is not None:
                    m = mine.setdefault(c, [0, 0])
                    m[0], m[1] = m[0] + int(ok), m[1] + 1
            for c in sorted(set(map(str, byc)) | set(mine)):
                e = byc.get(c)
                esr = e.get("sr") if isinstance(e, dict) and isinstance(e.get("sr"), dict) else {}
                want = mine.get(c, [0, 0])
                if [esr.get("k"), esr.get("n")] != want:
                    rp.append(f"by_category {c} sr {esr.get('k')!r}/{esr.get('n')!r} != the trajectories' "
                              f"{want[0]}/{want[1]}")
        cc["results_json"] = {"ok": not rp, "k": sr.get("k"), "n": sr.get("n"), "trajectories": [k, n],
                              "by_category_checked": isinstance(byc, dict), "problems": rp}
    pt = d / "per_task.jsonl"
    if not pt.is_file():
        cc["per_task"] = {"ok": None, "why": "no per_task.jsonl"}
    else:
        rows: Dict[str, list] = {}  # task_id -> the success value of every row with it, in file order
        bad_lines, n_rows, dups = 0, 0, []
        for ln in pt.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                x = json.loads(ln)
                tid, v = x["task_id"], x.get("success")
                hash(tid)
            except (ValueError, KeyError, TypeError, AttributeError):
                bad_lines += 1
                continue
            n_rows += 1
            if tid in rows:
                dups.append(tid)
            rows.setdefault(tid, []).append(v)
        # a task is a mismatch when it lacks a row or a trajectory, or ANY of its rows disagrees with result.success
        mism = sorted(str(t) for t in set(rows) | ids
                      if t not in rows or t not in success
                      or any(v not in (0, 1, True, False) or bool(v) != success[t] for v in rows[t]))
        dup_ids = sorted(set(map(str, dups)))
        pp = ([f"{len(mism)} task(s) whose per_task.jsonl success differs from result.success or that lack a row "
               f"or a trajectory, e.g. {mism[:3]}"] if mism else []) + \
             ([f"{bad_lines} unreadable per_task.jsonl line(s)"] if bad_lines else []) + \
             ([f"{len(dups)} duplicate per_task.jsonl row(s), task ids e.g. {dup_ids[:3]}"] if dups else [])
        cc["per_task"] = {"ok": not pp, "n_rows": n_rows, "n_task_ids": len(rows), "n_mismatch": len(mism),
                          "bad_lines": bad_lines, "n_duplicate_rows": len(dups), "duplicate_ids": dup_ids[:5],
                          "first": mism[:5], "problems": pp}
    out["crosscheck"] = cc
    out["_success"] = success
    return out


def pair(a: dict, b: dict, registered: List[str]) -> dict:
    """First (a) minus second (b) on the same registered tasks; missing (never imputed) unless both are usable."""
    if not (a["usable"] and b["usable"]):
        why = [f"{x['start']}/{x['tier']}__{x['arm']}: {p}" for x in (a, b) for p in x["problems"]]
        return {"missing": True, "why": why}
    sa, sb = a["_success"], b["_success"]
    n = len(registered)
    k1, k2 = sum(sa[t] for t in registered), sum(sb[t] for t in registered)
    bb = sum(1 for t in registered if sa[t] and not sb[t])
    cc = sum(1 for t in registered if sb[t] and not sa[t])
    p = exact_p(bb, cc)
    d = Fraction(k1 - k2, n)
    return {"missing": False, "n": n, "k_first": k1, "k_second": k2, "wilson95_first": a["wilson95"],
            "wilson95_second": b["wilson95"], "b": bb, "c": cc, "p": float(p), "p_exact": f"{p.numerator}/"
            f"{p.denominator}", "rate_difference": float(d), "rate_difference_exact": f"{d.numerator}/{d.denominator}",
            "both": sum(1 for t in registered if sa[t] and sb[t]),
            "neither": sum(1 for t in registered if not sa[t] and not sb[t])}


# --- the report -------------------------------------------------------------------------------------------------

def compare(pilot_root: Path = None, cfg=None, tasks_dir: Path = None, prereg: Path = None) -> dict:
    root = Path(pilot_root or pilot_tvr.LAYOUT.pilot_root)
    gp, pp = root / "GATE.json", root / "POWER.json"
    if not gp.is_file() or not pp.is_file():
        raise Refused(f"refused: the comparison runs only after GATE.json and POWER.json exist "
                      f"({gp}: {gp.is_file()}, {pp}: {pp.is_file()}; PREREG section 17, 2026-11-11)")
    prereg = Path(prereg or pilot_tvr.prereg_path())
    try:
        text = prereg.read_text(encoding="utf-8")
    except OSError:
        text = ""
    row = prereg_row(text)
    if row is None:
        raise Refused(f"refused: {prereg} has no single section-17 row '| {ROW_DATE} | ... {ROW_NEEDLE} ...'")
    try:
        commit = pilot_tvr.prereg_state().get("commit")
    except Exception:  # noqa: BLE001 - git may be absent; the file sha256 is recorded anyway
        commit = None
    reg = registered_tasks(Path(tasks_dir or paths.TVR_REPO / "data" / "tasks"))
    ids, cats = reg.pop("ids"), reg.pop("cats")
    rset = set(ids)
    z = float(pilot_cfg.get("gates.z", cfg))
    gate, power = read_json(gp) or {}, read_json(pp) or {}
    sides: Dict[str, dict] = {}

    def side(s, tier, arm):
        key = f"{s}/{tier}__{arm}"
        if key not in sides:
            sides[key] = read_side(root, s, tier, arm, rset, z, cats)
        return sides[key]

    comps = {}
    for first, second, role in COMPARISONS:
        cs = {}
        for tier, arm in cells(cfg):
            cs[f"{tier}__{arm}"] = dict(pair(side(first, tier, arm), side(second, tier, arm), ids), tier=tier,
                                        arm=arm)
        comps[f"{first}-{second}"] = {"role": role, "first": first, "second": second,
                                      "b_is": f"only {first} succeeds", "c_is": f"only {second} succeeds",
                                      "cells": cs}
    ref = reference(cfg)
    ref["pair"] = pair(side(ref["start"], ref["first"]["tier"], ref["arm"]),
                       side(ref["start"], ref["second"]["tier"], ref["arm"]), ids)
    # per start, tier and arm: k/n and Wilson of every eval the comparison read
    per_start = {}
    for key, sd in sides.items():
        per_start[key] = dict({"usable": True, "k": sd["k"], "n": sd["n"], "rate": sd["rate"],
                               "wilson95": sd["wilson95"]} if sd["usable"] else
                              {"usable": False, "problems": sd["problems"]}, results_file=sd["results_file"])
    # every cross-check entry names its eval twice: "S/tier__arm" (this report's key) and the eval dir name
    mism = []
    for key, sd in sides.items():
        for name, c in sd["crosscheck"].items():
            if c.get("ok") is False:
                mism.append({"eval": key, "eval_dir": Path(sd["dir"]).name, "check": name,
                             **{k: v for k, v in c.items() if k != "ok"}})
    unknown, unknown_used = [], []
    for key, sd in sides.items():
        for name, c in sd["crosscheck"].items():
            if c.get("ok") is None:
                unknown.append(f"{key}: {name}: {c.get('why')} ({Path(sd['dir']).name})")
                if sd["usable"]:  # its numbers enter a cell unchecked (an unusable side's cells are missing anyway)
                    unknown_used.append(unknown[-1])
    missing = [f"{name} {ck}" for name, comp in comps.items() for ck, c in comp["cells"].items() if c["missing"]]
    if ref["pair"]["missing"]:
        missing.append(f"reference {ref['start']} {ref['first']['tier']} vs {ref['second']['tier']}")
    return {"schema": SCHEMA, "utc": pilot_tvr.utc(), "pilot_root": str(root), "note": NOTE, "note_p": NOTE_P,
            "multiplicity_correction": "none", "z": z,
            "prereg": {"file": str(prereg), "sha256": sha256_file(prereg), "commit": commit, "section": 17,
                       "row_text": row, "row_sha256": sha256_bytes(row.encode())},
            "gate": {"file": str(gp), "sha256": sha256_file(gp), "decision": gate.get("decision"),
                     "G0": (gate.get("G0") or {}).get("status"), "A4": (gate.get("A4") or {}).get("status")},
            "power": {"file": str(pp), "sha256": sha256_file(pp), "pass": power.get("pass"),
                      "fallback_pass": power.get("fallback_pass")},
            "registered_tasks": reg, "cells": [f"{t}__{a}" for t, a in cells(cfg)],
            "comparisons": comps, "reference": ref, "per_start": per_start,
            "inputs": {sd["dir"]: sd["inputs"] for sd in sides.values()}, "trajectories_digest_rule": TRAJ_DIGEST_RULE,
            "crosscheck": {"ok": not mism, "mismatches": mism, "not_checked": unknown,
                           "used_sides_all_checked": not unknown_used},
            "missing_cells": missing}


# --- COMPARE.md -------------------------------------------------------------------------------------------------

def _ci(w) -> str:
    return f"{100 * w[0]:.2f}–{100 * w[1]:.2f}%"


def _p(p: float) -> str:
    return f"{p:.4g}"


def render_md(rep: dict) -> str:
    L = ["# TVR 试点：登记的比较（只报告）", "",
         f"- 依据：PREREG_TVR_PILOT.md 第 17 节 {ROW_DATE} 的登记行（sha256 {rep['prereg']['row_sha256'][:12]}）；"
         f"在 GATE.json（decision {rep['gate']['decision']}）和 POWER.json（pass {rep['power']['pass']}）之后算。",
         "- 不进闸门、决策表和 A4；不做多重比较校正。",
         "- **这些 p 只算了题目抽样，没算训练种子之间的波动（每个起点每档只有一个种子），所以偏乐观，只作描述，"
         "不是确认轮的结论（推断）。**",
         f"- 成对：同一批 {rep['registered_tasks']['n']} 道登记评测题按 task_id 配对；成功取 trajectory.json 的 "
         "result.success；p = min(1, 2·P(X ≤ min(b, c)))，X ~ Bin(b + c, 1/2)（第 12 节第 2 条）。", ""]
    for name, comp in rep["comparisons"].items():
        f, s = comp["first"], comp["second"]
        title = "主报告" if comp["role"] == "primary" else "补充"
        L += [f"## {f} 对 {s}（{title}）", "",
              f"| 档 | 臂 | {f} k/n（95% CI） | {s} k/n（95% CI） | 只有 {f} 成功 b | 只有 {s} 成功 c | 精确双侧 p | "
              f"(k1−k2)/n |", "|---|---|---|---|---|---|---|---|"]
        for c in comp["cells"].values():
            if c["missing"]:
                L.append(f"| {c['tier']} | {c['arm']} | 缺 | 缺 | | | | {'；'.join(c['why'])[:200]} |")
                continue
            L.append(f"| {c['tier']} | {c['arm']} | {c['k_first']}/{c['n']}（{_ci(c['wilson95_first'])}） | "
                     f"{c['k_second']}/{c['n']}（{_ci(c['wilson95_second'])}） | {c['b']} | {c['c']} | {_p(c['p'])} | "
                     f"{100 * c['rate_difference']:+.1f} 个百分点 |")
        L.append("")
    L += ["## 三个起点各档各臂的 k/n 和 95% Wilson 区间", "", "| 起点 | 档 | 臂 | k/n | 率 | 95% CI |",
          "|---|---|---|---|---|---|"]
    for key, r in sorted(rep["per_start"].items()):
        s, rest = key.split("/", 1)
        tier, arm = rest.split("__")
        if not r["usable"]:
            L.append(f"| {s} | {tier} | {arm} | 缺 | | {'；'.join(r['problems'])[:200]} |")
        else:
            L.append(f"| {s} | {tier} | {arm} | {r['k']}/{r['n']} | {100 * r['rate']:.1f}% | {_ci(r['wilson95'])} |")
    odd = [f"{key}（{r.get('results_file') or '没有结果文件'}）" for key, r in sorted(rep["per_start"].items())
           if r["usable"] and r.get("results_file") != "results.json"]
    if odd:
        L += ["", "注：这些评测没有 results.json（score_tvr 没判为有效），数字仍取自 trajectory，见交叉核对：" + "；".join(odd)]
    ref = rep["reference"]
    rp = ref["pair"]
    L += ["", f"## 参照行：{ref['start']} {ref['first']['tier']} 档 c 臂，种子 {ref['first']['seed']} 对 "
          f"{ref['second']['seed']}（{ref['second']['tier']}）", "",
          "| 种子 1001 k/n（95% CI） | 种子 1002 k/n（95% CI） | 只有 1001 成功 b | 只有 1002 成功 c | 精确双侧 p |",
          "|---|---|---|---|---|"]
    if rp["missing"]:
        L.append(f"| 缺 | 缺 | | | {'；'.join(rp['why'])[:200]} |")
    else:
        L.append(f"| {rp['k_first']}/{rp['n']}（{_ci(rp['wilson95_first'])}） | {rp['k_second']}/{rp['n']}"
                 f"（{_ci(rp['wilson95_second'])}） | {rp['b']} | {rp['c']} | {_p(rp['p'])} |")
    cc = rep["crosscheck"]
    L += ["", "## 交叉核对（trajectory.json 对任务目录、per_task.jsonl、results.json）", "",
          "全部一致。" if cc["ok"] and not cc["not_checked"] else
          "；".join([f"{m['eval']} {m['check']} 不一致（{'；'.join(m.get('problems') or [])}）"
                    for m in cc["mismatches"]] + [f"没核对：{u}" for u in cc["not_checked"]])[:2000]]
    if rep["missing_cells"]:
        L += ["", "缺的格（拒算，不插补）：" + "；".join(rep["missing_cells"])]
    L.append("")
    return "\n".join(L)


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def main(argv=None) -> int:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-root", type=Path, default=None, help="default OUT/tvr/runs/pilot")
    ap.add_argument("--out", type=Path, default=None, help="default PILOT_ROOT/COMPARE.json (COMPARE.md beside it)")
    ap.add_argument("--tasks-dir", type=Path, default=None, help="default the pinned TVRBench data/tasks")
    ap.add_argument("--prereg", type=Path, default=None, help="default vpd/PREREG_TVR_PILOT.md of this tree")
    args = ap.parse_args(argv)
    try:
        rep = compare(args.pilot_root, tasks_dir=args.tasks_dir, prereg=args.prereg)
    except Refused as r:
        print(r.message, file=sys.stderr)
        return 2
    out = args.out or Path(rep["pilot_root"]) / "COMPARE.json"
    write_text(out, json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    write_text(out.with_suffix(".md"), render_md(rep))
    cc = rep["crosscheck"]
    print(json.dumps({"out": str(out), "crosscheck_ok": cc["ok"], "used_sides_all_checked":
                      cc["used_sides_all_checked"], "not_checked": len(cc["not_checked"]),
                      "missing_cells": len(rep["missing_cells"])}))
    if not cc["ok"] or not cc["used_sides_all_checked"]:
        return 1
    return 3 if rep["missing_cells"] else 0


if __name__ == "__main__":
    sys.exit(main())
