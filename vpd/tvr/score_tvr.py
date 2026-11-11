"""Scoring of one TVR run dir (CONTRACT_wave2 4.7, 5.2.3, 6.4).  PY_Q3, stdlib only.

score_run(run_dir, task_ids=..., bfs_table=..., reach_index=..., arm=..., meta=...) reads
RUN/tasks/<task_id>/trajectory.json for every task of the set and writes RUN/per_task.jsonl and RUN/results.json
(valid) | results_invalid.json | results_partial.json (soft deadline).  Metrics (frozen in the prereg):
  SR = k/n with Wilson 95% (vpd.walkerbench.pilot_gate.wilson_lb / wilson_ub, z 1.96), per category too;
  n_nonstop = replies before the Stop (all replies without a Stop); SPL_i = success * opt / max(opt, n_nonstop)
  with opt = the BFS minimum of non-Stop actions in the frozen-set pose graph (bfs/<set>_opt.json); SPL = mean;
  within1 = stopped AND kin_env.within_one(final, target, R(task)); reached = some visited pose == target;
  stop_rate, fstop_given_stop, fstop_over_all (eval_va definitions); invalid counts; final errors;
  scene-cluster bootstrap 95% percentile CI of SR (10,000 resamples, random.Random(0); descriptive only);
  spl_official = "eval_va _spl with our BFS opt (Stop counted in steps)" (descriptive).
valid iff every task of the set has a trajectory, the server was alive at the end, 0 API errors and 0 task
failures (regex mismatches are gate G2, not validity), and - an image arm or the replay whose meta carries a
renderer block - some THOR worker's ready reply is in that block (renderer_measured).  Under mechanism_only (the
smoke) SR / SPL / within1 / stop / reached / bootstrap / final-error fields are withheld ("withheld": "mechanism
smoke").

CLI (bash payloads):
    PY_Q3 -B -m vpd.tvr.score_tvr official --run-dir D --served CODE --bfs OUT/tvr/bfs/<set>_opt.json
writes D/official_opt.json, runs TVRBench's own eval_va.aggregate() on D/tasks under TVR_PY (imports ai2thor,
starts no Controller; eval_va writes results.json + summary.txt INTO D/tasks), then official_check ->
D/official_check.json; exit 0 equal, 1 unequal, 2 refused (mechanism-only dir, no results.json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vpd.tvr import kin_env, lvp, pilot_cfg
from vpd.walkerbench import pilot_gate

SCHEMA = "vpd.tvr.results/1"
WITHHELD = "mechanism smoke"
SPL_OFFICIAL_LABEL = "eval_va _spl with our BFS opt (Stop counted in steps)"
RESULT_FILES = ("results.json", "results_invalid.json", "results_partial.json")
MECH_COLUMNS = ("task_id", "category", "scene", "num_steps", "n_nonstop", "n_blocked", "n_unparseable",
                "regex_mismatch")
RENDER_ARMS = ("c", "u", "replay")  # rendered by THOR workers: a valid result needs their measured renderer


def renderer_measured(renderer) -> bool:
    """run_tvr's renderer block holds some worker's ready reply (a ready reply always carries the worker's pid; a
    failed start-up reply has none; no pool start-up -> workers [])."""
    ws = renderer.get("workers") if isinstance(renderer, dict) else None
    return isinstance(ws, list) and any(isinstance(w, dict) and w.get("pid") is not None for w in ws)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, path)


def read_json(p) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- statistics ---------------------------------------------------------------------------------------------------
def wilson(k, n) -> Optional[List[float]]:
    """[lb, ub] of the Wilson 95% interval (pilot_gate.wilson_lb / wilson_ub, z = gates.z)."""
    if not n:
        return None
    z = float(pilot_cfg.FROZEN["gates"]["z"])
    return [round(pilot_gate.wilson_lb(k, n, z), 6), round(pilot_gate.wilson_ub(k, n, z), 6)]


def rate_block(k, n) -> Dict[str, Any]:
    return {"k": k, "n": n, "rate": (round(k / n, 6) if n else None), "wilson95": wilson(k, n)}


def percentile(xs: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy's default)."""
    s = sorted(float(x) for x in xs)
    if not s:
        return None
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def scene_bootstrap(rows, resamples: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Scene-cluster bootstrap 95% percentile CI of SR: resample the scenes (sorted) with replacement; SR =
    successes / tasks of the resampled scenes.  Descriptive only."""
    by: Dict[str, List[int]] = {}
    for r in rows:
        s = by.setdefault(r["scene"], [0, 0])
        s[0] += int(bool(r["success"]))
        s[1] += 1
    scenes = sorted(by)
    rng = random.Random(seed)
    m = len(scenes)
    out = {"resamples": int(resamples), "seed": seed, "n_clusters": m, "ci95": None,
           "use": pilot_cfg.FROZEN["bootstrap"]["use"]}
    if not m:
        return out
    ks = [by[s][0] for s in scenes]
    ns = [by[s][1] for s in scenes]
    vals = []
    for _ in range(int(resamples)):
        k = n = 0
        for _j in range(m):
            i = rng.randrange(m)
            k += ks[i]
            n += ns[i]
        vals.append(k / n if n else 0.0)
    out["ci95"] = [round(percentile(vals, 2.5), 6), round(percentile(vals, 97.5), 6)]
    return out


def spl_official_term(success, num_steps: int, opt) -> float:
    """eval_va._spl (eval_va.py:363-368) with our BFS opt."""
    if not success or opt is None or num_steps <= 0:
        return 0.0
    return min(opt, num_steps) / max(opt, num_steps)


# --- per task -----------------------------------------------------------------------------------------------------
def task_row(traj: Dict[str, Any], *, opt_entry: Optional[dict], reach, have_bfs: bool) -> Dict[str, Any]:
    res, v = traj["result"], traj.get("vpd") or {}
    stopped = bool(res.get("stopped"))
    n = int(res.get("num_steps") or 0)
    n_nonstop = n - 1 if stopped else n
    success = res.get("success")
    succ = int(bool(success)) if success is not None else None
    opt = None
    if have_bfs and opt_entry is not None and opt_entry.get("feasible") and opt_entry.get("opt") is not None:
        opt = int(opt_entry["opt"])
    spl = spl_off = None
    if have_bfs:
        spl = (succ * opt / max(opt, n_nonstop)) if (succ and opt) else 0.0
        spl_off = spl_official_term(bool(succ), n, opt)
    target = kin_env.pose_of(traj["target"])
    states = v.get("states") or []
    poses = [kin_env.pose_of(s) for s in states]
    final = poses[-1] if poses else None
    within1 = None
    if reach is not None and final is not None:
        within1 = bool(stopped and kin_env.within_one(final, target, reach))
    reached = any(kin_env.is_exact(p, target) for p in poses) if poses else None
    cap = int(v.get("cap") or pilot_cfg.cap_of(traj))
    return {"task_id": traj["task_id"], "category": v.get("category") or pilot_cfg.category_of(traj),
            "scene": traj["scene"], "success": succ, "stopped": stopped, "num_steps": n, "n_nonstop": n_nonstop,
            "opt": opt, "spl": spl, "spl_official": spl_off, "within1": within1, "reached": reached,
            "final_pos_error": res.get("final_pos_error"), "final_rot_error": res.get("final_rot_error"),
            "final_hor_error": res.get("final_hor_error"), "n_blocked": int(v.get("n_blocked") or 0),
            "n_unparseable": int(v.get("n_unparseable") or 0), "regex_mismatch": int(v.get("regex_mismatch") or 0),
            "capped": bool(not stopped and n >= cap)}


def _mean(xs) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else None


def block(rows: List[Dict[str, Any]], mechanism_only: bool, have_bfs: bool) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {"n": n, "mean_steps": _mean(r["num_steps"] for r in rows)}
    if mechanism_only:
        return out
    k = sum(r["success"] for r in rows)
    st = sum(r["stopped"] for r in rows)
    out.update(sr=rate_block(k, n), spl=(_mean(r["spl"] for r in rows) if have_bfs else None),
               within1=(rate_block(sum(bool(r["within1"]) for r in rows), n)
                        if rows and all(r["within1"] is not None for r in rows) else None),
               stop_rate=(round(st / n, 6) if n else None),
               fstop_given_stop=(round((st - k) / st, 6) if st else None),
               fstop_over_all=(round((st - k) / n, 6) if n else None))
    return out


def score_run(run_dir, *, task_ids: List[str], bfs_table: Optional[dict], reach_index, arm: str,
              mechanism_only: bool = False, meta: dict, partial: bool = False,
              overwrite_valid: bool = False) -> Tuple[dict, bool]:
    """Score RUN/tasks/*/trajectory.json of `task_ids`; write per_task.jsonl and the result file; (results, valid).
    bfs_table None (the replay, scored before the data job's bfs step): opt / spl / spl_official are null.
    partial (soft deadline): results_partial.json, never results.json.  A valid results.json already on disk is
    never overwritten unless overwrite_valid (rescore).  arm in RENDER_ARMS and meta["renderer"] present: valid also
    needs renderer_measured (recorded as "renderer_measured")."""
    run_dir = Path(run_dir)
    if (run_dir / "results.json").exists() and not overwrite_valid:  # before any write (per_task.jsonl too)
        raise RuntimeError(f"{run_dir}/results.json exists: a valid result is never overwritten")
    have_bfs = bfs_table is not None
    rows, missing = [], []
    for tid in task_ids:
        traj = read_json(run_dir / "tasks" / tid / "trajectory.json")
        if traj is None:
            missing.append(tid)
            continue
        reach = None
        if reach_index is not None:
            try:
                reach = reach_index.reach(tid)
            except (KeyError, OSError, ValueError):
                reach = None
        opt_entry = ((bfs_table or {}).get("tasks") or {}).get(tid) if have_bfs else None
        rows.append(task_row(traj, opt_entry=opt_entry, reach=reach, have_bfs=have_bfs))
    n_tasks = len(task_ids)
    api_errors = int(meta.get("api_errors") or 0)
    task_failures = meta.get("task_failures") or 0
    task_failures = len(task_failures) if isinstance(task_failures, (list, tuple)) else int(task_failures)
    alive = bool(meta.get("server_alive_at_end", True))
    # gate G0(4) judges the renderer block, and a valid results.json is never overwritten: without a measured
    # renderer the arm is invalid (or partial), so a later --resume redoes the finalize with a started pool
    rend_check = arm in RENDER_ARMS and "renderer" in meta
    rend_ok = renderer_measured(meta["renderer"]) if rend_check else True
    valid = (not missing) and alive and api_errors == 0 and task_failures == 0 and rend_ok and not partial
    comp: Dict[str, Any] = {
        "schema": SCHEMA, "arm": arm, "n_tasks": n_tasks, "n_scored": len(rows), "valid": valid,
        "partial": bool(partial), "tasks_missing": missing,
        "invalid": {"unparseable_steps": sum(r["n_unparseable"] for r in rows),
                    "blocked_steps": sum(r["n_blocked"] for r in rows),
                    "regex_mismatch": sum(r["regex_mismatch"] for r in rows),
                    "replies": sum(r["num_steps"] for r in rows)},
        "steps": {"mean": _mean(r["num_steps"] for r in rows), "total": sum(r["num_steps"] for r in rows)},
        "by_category": {c: block([r for r in rows if r["category"] == c], mechanism_only, have_bfs)
                        for c in pilot_cfg.CATEGORIES},
        "api_errors": api_errors, "task_failures": task_failures, "server_alive_at_end": alive,
        # the two conventions agree on a valid run (every task scored); they differ on partial / invalid files
        "denominators": {"overall": "n_tasks (a missing task counts as a failure)",
                         "by_category": "n = the scored tasks of the category"},
        "utc_scored": utc()}
    if rend_check:
        comp["renderer_measured"] = rend_ok
    if mechanism_only:
        comp["withheld"] = WITHHELD
    else:
        k = sum(r["success"] for r in rows)
        st = sum(r["stopped"] for r in rows)
        comp.update(
            sr=rate_block(k, n_tasks),
            spl=(round(sum(r["spl"] for r in rows) / n_tasks, 6) if (have_bfs and n_tasks) else None),
            spl_official=(round(sum(r["spl_official"] for r in rows) / n_tasks, 6) if (have_bfs and n_tasks)
                          else None),
            spl_official_label=SPL_OFFICIAL_LABEL,
            within1=(rate_block(sum(bool(r["within1"]) for r in rows), n_tasks)
                     if rows and all(r["within1"] is not None for r in rows) else None),
            reached=(rate_block(sum(bool(r["reached"]) for r in rows), n_tasks) if rows else None),
            stop={"n_stopped": st, "stop_rate": (round(st / n_tasks, 6) if n_tasks else None),
                  "fstop_given_stop": (round((st - k) / st, 6) if st else None),
                  "fstop_over_all": (round((st - k) / n_tasks, 6) if n_tasks else None)},
            errors_final={"pos": _mean(r["final_pos_error"] for r in rows),
                          "rot": _mean(r["final_rot_error"] for r in rows),
                          "hor": _mean(r["final_hor_error"] for r in rows)},
            bootstrap_scene=scene_bootstrap(rows, pilot_cfg.FROZEN["bootstrap"]["resamples"],
                                            pilot_cfg.FROZEN["bootstrap"]["seed"]))
        comp["steps"]["capped_tasks"] = sum(r["capped"] for r in rows)
    res = dict(meta)
    res.update(comp)
    # per_task.jsonl (mechanism-only: mechanism columns only)
    tmp = run_dir / f"per_task.jsonl.part{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in MECH_COLUMNS} if mechanism_only else r) + "\n")
    os.replace(tmp, run_dir / "per_task.jsonl")
    name = "results_partial.json" if partial else ("results.json" if valid else "results_invalid.json")
    for other in RESULT_FILES:
        if (run_dir / other).exists():
            (run_dir / other).unlink()  # only this run dir's own earlier result file
    write_json_atomic(run_dir / name, res)
    return res, valid


# --- TVRBench's own aggregate (official check) ------------------------------------------------------------------
def official_opt(bfs_table: dict) -> Dict[str, Dict[str, Any]]:
    """eval_va's optimal-steps format {task_id: {"feasible", "optimal_steps"}} from our BFS table."""
    out = {}
    for tid, e in (bfs_table.get("tasks") or {}).items():
        out[tid] = {"feasible": bool(e.get("feasible")), "optimal_steps": e.get("opt")}
    return out


def official_aggregate_cmd(run_dir, served: str, bfs_table: dict, *, py=None, repo=None) -> List[str]:
    """Writes RUN/official_opt.json and returns the argv that runs eval_va.aggregate(RUN/tasks, served,
    optimal_steps_path=RUN/official_opt.json) under the TVR env (eval_va imports ai2thor; no Controller starts).
    eval_va writes results.json + summary.txt INTO RUN/tasks (never RUN/results.json)."""
    run_dir = Path(run_dir).resolve()
    repo = Path(repo if repo is not None else lvp.TVR_REPO)
    py = str(py if py is not None else lvp.TVR_PY)
    write_json_atomic(run_dir / "official_opt.json", official_opt(bfs_table))
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('tvr_eval_va', {str(repo / 'scripts' / 'eval_va.py')!r})\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            f"m.aggregate({str(run_dir / 'tasks')!r}, {served!r}, "
            f"optimal_steps_path={str(run_dir / 'official_opt.json')!r})\n")
    return [py, "-B", "-c", code]


def official_check(run_dir) -> Dict[str, Any]:
    """{"official_sr", "our_sr", "equal", ...}: eval_va's RUN/tasks/results.json against our RUN/results.json."""
    run_dir = Path(run_dir)
    off = read_json(run_dir / "tasks" / "results.json") or {}
    ours = read_json(run_dir / "results.json") or {}
    sr = ours.get("sr") or {}
    ok_n, ok_k = off.get("n_tasks"), off.get("n_success")
    eq = ok_k is not None and ok_k == sr.get("k") and ok_n == sr.get("n")
    out = {"official_sr": off.get("overall_success_rate"), "our_sr": sr.get("rate"),
           "official_k": ok_k, "official_n": ok_n, "our_k": sr.get("k"), "our_n": sr.get("n"), "equal": bool(eq),
           "official_spl": off.get("overall_spl"), "our_spl_official": ours.get("spl_official"),
           "spl_label": SPL_OFFICIAL_LABEL}
    if off.get("overall_spl") is not None and ours.get("spl_official") is not None:
        out["spl_equal"] = abs(float(off["overall_spl"]) - float(ours["spl_official"])) < 1e-6
    return out


def cmd_official(a) -> int:
    run_dir = Path(a.run_dir)
    proto = read_json(run_dir / "protocol.json") or {}
    res = read_json(run_dir / "results.json")
    if res is None:
        print(f"refusing: {run_dir}/results.json missing (only a valid run is checked)", file=sys.stderr)
        return 2
    if proto.get("mechanism_only") or res.get("withheld"):
        print(f"refusing: {run_dir} is a mechanism-only run (success withheld)", file=sys.stderr)
        return 2
    bfs = read_json(a.bfs)
    if bfs is None:
        print(f"refusing: cannot read the BFS table {a.bfs}", file=sys.stderr)
        return 2
    argv = official_aggregate_cmd(run_dir, a.served, bfs, py=a.py, repo=a.repo)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=float(a.timeout_s))
        rc, tail = p.returncode, (p.stdout + p.stderr)[-3000:]
    except subprocess.TimeoutExpired as e:
        rc, tail = None, f"timeout after {a.timeout_s} s: {e!r}"
    chk = official_check(run_dir)
    chk.update(rc=rc, output_tail=tail, seconds=round(time.time() - t0, 2), served=a.served, bfs=str(a.bfs),
               utc=utc())
    if rc != 0:
        chk["equal"] = False
    write_json_atomic(run_dir / "official_check.json", chk)
    print(json.dumps({k: chk.get(k) for k in ("official_sr", "our_sr", "equal", "rc")}))
    return 0 if chk["equal"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("official", help="TVRBench's own eval_va.aggregate on a run's trajectories vs results.json")
    o.add_argument("--run-dir", required=True)
    o.add_argument("--served", required=True, help="the served model name (eval_va's model_name label)")
    o.add_argument("--bfs", required=True, help="OUT/tvr/bfs/<set>_opt.json")
    o.add_argument("--py", default=str(lvp.TVR_PY), help="the TVR env python (default TVR_PY)")
    o.add_argument("--repo", default=str(lvp.TVR_REPO))
    o.add_argument("--timeout-s", type=float, default=600.0)
    a = ap.parse_args(argv)
    return cmd_official(a)


if __name__ == "__main__":
    sys.exit(main())
