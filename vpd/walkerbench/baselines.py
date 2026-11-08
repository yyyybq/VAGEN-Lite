"""Blind floors and the expert ceiling on the 179-task s2test, through WalkerBench's own loop and scorer.

    python -m vpd.walkerbench.baselines      # -> OUT/walkerbench/baselines/{baselines_s2test.json, hard_subsets.json}

Policies (rendering stubbed: the observation generator returns None, no image is made or sent):
- STOP0: STOP("ARRIVED") at once.
- FWD:   always the front-most move (available_moves[0]), never stop; the loop ends at the stock step cap
         round(1.5 x len(visual_path)).  With no move it re-looks at the current heading.
- ODO:   FWD until the walked distance reaches ground_truth.optimal_distance_meters, then stop.  A legitimate blind
         floor for nav only, whose instruction states the total distance; on vis it is descriptive.
- RM:    the expert used for the demos (DECISIONS B6 = scratchpad vpd_demos/b3_proto.py lines 43-60): follow the
         visual_path; rotate first when the next pano is > 45 deg off the current heading and this node has not been
         rotated at yet, then MOVE; STOP("ARRIVED") at the path end, off the path or at a path break.
- RM_SPLICE: RM on vis_1103 along the BFS-spliced path of split/expert_splice.json (side note; file unmodified).
Each task runs through scripts/run_benchmark_parallel.run_single_task in a sandbox (vpd.walkerbench.sandbox) and is
scored by scripts/evaluate_all.evaluate_session (success = final pano within 15 m of a target).  Results are keyed
by the file stem (= the log's task_id, which is the stem run_single_task was given).

Floors written for the gate (PREREG G1, DECISIONS B3): nav ODO rate on the 85 nav tasks, vis FWD rate on the 94 vis
tasks, and the pooled legit floor F = (nav ODO successes + vis FWD successes) / 179.
hard_subsets.json: nav tasks ODO fails, vis tasks FWD fails.
The survey numbers (scratchpad vpd_demos/b2_out.txt) are compared and every difference is printed as WARN.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as _dt
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

from vpd import paths
from vpd.walkerbench import sandbox
from vpd.walkerbench import split as S

SCHEMA = "vpd.walkerbench.baselines/1"
POLICIES = ("STOP0", "FWD", "ODO", "RM")
ROT_THRESHOLD_DEG = 45
# survey (b2_out.txt): successes per kind on s2test; standard-only = the 10 test tasks from tasks/standard
EXPECTED = {
    "STOP0": {"nav": 4, "vis": 3, "standard_only": 0},
    "FWD": {"nav": 6, "vis": 3, "standard_only": 1},
    "ODO": {"nav": 24, "vis": 19, "standard_only": 4},
    "RM": {"nav": 85, "vis": 93, "standard_only": 10},
}
EXPECTED_N = {"nav": 85, "vis": 94, "standard_only": 10}
EXPECTED_RM_FAILS = [S.SPLICE_TASK]
EXPECTED_HARD = {"nav_odo_fail": 61, "vis_fwd_fail": 91}
EXPECTED_POOLED_F = 27


def thread_env() -> None:
    """Login-node / job rules: one thread per process; dummy API settings (no server is ever contacted)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[v] = os.environ.get("VPD_THREADS", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.update({"API_KEY": "x", "API_BASE_URL": "http://127.0.0.1:1", "BENCHMARK_URL": "http://127.0.0.1:1"})


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": [round(lo, 6), round(hi, 6)]}


# ---------------------------------------------------------------- policies (pure) ----------------------------------

def stop(reason: str = "") -> dict:
    return {"type": "stop", "answer": "ARRIVED", "reason": reason}


def rm_expert(session, get_moves, rotated: set, path: list[str] | None = None) -> dict:
    """The RM expert (b3_proto.py lines 43-60).  rotated: path indices already rotated at (per session)."""
    vp = path if path is not None else [p["pano_id"] for p in session.task_config["visual_path"]]
    cur = session.state.pano_id
    idx = max(i for i, p in enumerate(vp) if p == cur) if cur in vp else None
    if idx is None or idx == len(vp) - 1:
        return stop("")
    m = [x for x in get_moves() if x["pano_id"] == vp[idx + 1]]
    if not m:
        return stop("path break")
    m = m[0]
    rel = (m["heading"] - session.state.heading + 360) % 360
    rel = min(rel, 360 - rel)
    if rel > ROT_THRESHOLD_DEG and idx not in rotated:
        rotated.add(idx)
        return {"type": "rotation", "heading": round(m["heading"], 1), "pitch": 0, "reason": ""}
    return {"type": "move", "move_id": m["id"], "reason": ""}


def blind_action(mode: str, session, moves: list[dict], st: dict) -> dict:
    """STOP0 / FWD / ODO (a9_baselines.py, b2_baselines.py).  st: per-session {'walked': metres}."""
    if mode == "STOP0":
        return stop("x")
    if mode == "ODO":
        opt = session.task_config.get("ground_truth", {}).get("optimal_distance_meters", 0)
        if st["walked"] >= opt or not moves:
            return stop("x")
    if not moves:
        return {"type": "rotation", "heading": session.state.heading, "pitch": 0, "reason": "x"}
    m = moves[0]
    st["walked"] += m.get("distance") or 0
    return {"type": "move", "move_id": m["id"], "reason": "x"}


# ---------------------------------------------------------------- WalkerBench in-process ---------------------------

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_walkerbench(copy: Path, tasks_label: str) -> types.SimpleNamespace:
    """Import the sandbox copy's runner, scorer and engine into this process (one sandbox per process)."""
    copy = Path(copy).resolve()
    eng = sys.modules.get("engine")
    if eng is not None and not str(getattr(eng, "__file__", "")).startswith(str(copy)):
        raise RuntimeError("another WalkerBench engine is already imported in this process")
    thread_env()
    os.chdir(copy)
    sys.path.insert(0, str(copy))
    ex = types.ModuleType("examples")
    ex.__path__ = [str(copy / "examples")]
    sys.modules["examples"] = ex
    with contextlib.redirect_stdout(io.StringIO()):
        rbp = _load("wb_rbp", copy / "scripts" / "run_benchmark_parallel.py")
        ev = _load("wb_ev", copy / "scripts" / "evaluate_all.py")
    import engine.observation_generator as og
    from engine.action_executor import action_executor
    from engine.session_manager import session_manager
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

    class _NoRender:
        calls = 0

        def generate_observation(self, *a, **k):
            _NoRender.calls += 1
            return None

    og._observation_generator = _NoRender()
    return types.SimpleNamespace(copy=copy, rbp=rbp, ev=ev, og=og, ax=action_executor, sm=session_manager,
                                 norender=_NoRender, tasks_dir=copy / "tasks" / tasks_label)


def run_policy(wb, mode: str, stems: list[str], logs_dir: Path, paths_override: dict | None = None) -> dict:
    """Run one policy on the stems; returns {stem: evaluate_session result subset} (missing stem = no log)."""
    state = collections.defaultdict(lambda: {"walked": 0.0, "rotated": set()})
    base = "RM" if mode == "RM_SPLICE" else mode

    class BaselineAgent(wb.rbp.VLNAgent):
        def get_image_base64(self, url):
            return None

        def decide_action(self, obs):
            s = wb.sm.get_session(self.session_id)
            self.messages.append(self._build_user_message(obs))  # exercise the stock prompt builder
            st = state[self.session_id]
            if base == "RM":
                path = (paths_override or {}).get(s.task_id)
                return rm_expert(s, lambda: wb.ax._get_available_moves(s), st["rotated"], path)
            return blind_action(base, s, wb.ax._get_available_moves(s), st)

    logs_dir = Path(logs_dir)
    if logs_dir.exists() and any(logs_dir.iterdir()):
        raise FileExistsError(f"{logs_dir} is not empty")
    logs_dir.mkdir(parents=True, exist_ok=True)
    wb.rbp.VLNAgent = BaselineAgent
    wb.rbp.TASKS_DIR = wb.tasks_dir
    wb.rbp.LOGS_DIR = logs_dir
    failed = []
    with contextlib.redirect_stdout(io.StringIO()):
        for t in stems:
            if wb.rbp.run_single_task(mode.lower(), t) is False:
                failed.append(t)
    per = {}
    for log in sorted(logs_dir.glob("*.jsonl")):
        r = wb.ev.evaluate_session(log, wb.tasks_dir)
        if not r:
            continue
        tid = r["task_id"]
        if tid in per:
            raise RuntimeError(f"{mode}: two logs for {tid}")
        per[tid] = {"success": int(r["success"]), "steps": r["steps"], "error": r["error"],
                    "spl": r["spl"], "length": r.get("length"), "log": log.name}
    extra = sorted(set(per) - set(stems))
    if extra:
        raise RuntimeError(f"{mode}: logs for tasks not run: {extra[:5]}")
    return {"per_task": per, "run_failed": failed}


# ---------------------------------------------------------------- summaries ---------------------------------------

def summarise(per: dict, stems: list[str], std_only: set[str]) -> dict:
    c = collections.Counter()
    for t in stems:
        k = S.kind_of(t)
        r = per.get(t)
        ok = bool(r and r["success"])
        c[k + "_n"] += 1
        c[k + "_k"] += ok
        c["nolog"] += r is None
        if t in std_only:
            c["so_n"] += 1
            c["so_k"] += ok
    out = {"nav": rate(c["nav_k"], c["nav_n"]), "vis": rate(c["vis_k"], c["vis_n"]),
           "all": rate(c["nav_k"] + c["vis_k"], c["nav_n"] + c["vis_n"]),
           "standard_only": rate(c["so_k"], c["so_n"]), "tasks_without_log": c["nolog"]}
    tm = [out[k]["rate"] for k in ("nav", "vis") if out[k]["n"]]
    out["type_mean"] = sum(tm) / len(tm) if tm else None
    return out


def floors(res: dict, stems: list[str]) -> dict:
    nav = [t for t in stems if S.kind_of(t) == "nav"]
    vis = [t for t in stems if S.kind_of(t) == "vis"]
    ok = lambda m, t: bool(res[m]["per_task"].get(t, {}).get("success"))  # noqa: E731
    k_nav = sum(ok("ODO", t) for t in nav)
    k_vis = sum(ok("FWD", t) for t in vis)
    return {"nav_ODO": dict(rate(k_nav, len(nav)), tasks="s2test nav", policy="ODO"),
            "vis_FWD": dict(rate(k_vis, len(vis)), tasks="s2test vis", policy="FWD"),
            "pooled_F": dict(rate(k_nav + k_vis, len(stems)), definition="(nav ODO successes + vis FWD successes) / N"),
            "STOP0_pooled": res["STOP0"]["summary"]["all"], "FWD_pooled": res["FWD"]["summary"]["all"]}


def hard_subsets(res: dict, stems: list[str]) -> dict:
    nav = sorted(t for t in stems if S.kind_of(t) == "nav" and not res["ODO"]["per_task"].get(t, {}).get("success"))
    vis = sorted(t for t in stems if S.kind_of(t) == "vis" and not res["FWD"]["per_task"].get(t, {}).get("success"))
    return {"nav_odo_fail": nav, "vis_fwd_fail": vis, "counts": {"nav_odo_fail": len(nav), "vis_fwd_fail": len(vis)},
            "definition": "nav tasks the blind ODO policy fails; vis tasks the blind FWD policy fails (s2test)"}


def compare_survey(res: dict, fl: dict, hs: dict, budget_hits: int, n_test: int) -> list[str]:
    warn = []
    for m, want in EXPECTED.items():
        if m not in res:
            continue
        sm = res[m]["summary"]
        for key, k in want.items():
            got = (sm[key]["k"], sm[key]["n"])
            if got != (k, EXPECTED_N[key]):
                warn.append(f"{m} {key}: {got[0]}/{got[1]} != survey {k}/{EXPECTED_N[key]}")
        if sm["tasks_without_log"]:
            warn.append(f"{m}: {sm['tasks_without_log']} tasks without a log")
    if "RM" in res:
        fails = sorted(t for t, r in res["RM"]["per_task"].items() if not r["success"])
        if fails != EXPECTED_RM_FAILS:
            warn.append(f"RM failures {fails} != survey {EXPECTED_RM_FAILS}")
    if "RM_SPLICE" in res and not res["RM_SPLICE"]["per_task"].get(S.SPLICE_TASK, {}).get("success"):
        warn.append("RM_SPLICE: the spliced expert does not reach vis_1103's target")
    if hs and hs["counts"] != EXPECTED_HARD:
        warn.append(f"hard subsets {hs['counts']} != survey {EXPECTED_HARD}")
    if fl and fl["pooled_F"]["k"] != EXPECTED_POOLED_F:
        warn.append(f"pooled F {fl['pooled_F']['k']}/{fl['pooled_F']['n']} != survey {EXPECTED_POOLED_F}/179")
    if "FWD" in res and budget_hits != n_test:
        warn.append(f"FWD: {budget_hits}/{n_test} tasks end at the step cap (survey: all)")
    return warn


# ---------------------------------------------------------------- main ---------------------------------------------

def _write_json(path: Path, obj, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists (use --force to replace it; no job may be reading it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def run(split_dir: Path, out: Path, work: Path, policies=POLICIES, stems: list[str] | None = None,
        keep_sandbox: bool = False, force: bool = False) -> dict:
    split_dir, out, work = Path(split_dir), Path(out), Path(work)
    sp_path = split_dir / "split_s2.json"
    doc = json.loads(sp_path.read_text())
    test = doc["splits"]["s2test"]
    subset = stems is not None
    stems = list(stems) if subset else list(test)
    if set(stems) - set(test):
        raise ValueError(f"not s2test tasks: {sorted(set(stems) - set(test))[:5]}")
    if subset and out.resolve() == paths.WB_BASELINES.resolve():
        raise ValueError("a task subset must not be written to the frozen baselines dir; pass --out")
    for f in ("baselines_s2test.json", "hard_subsets.json"):
        if (out / f).exists() and not force:
            raise FileExistsError(f"{out / f} exists (use --force to replace it; no job may be reading it)")
    splice = json.loads((split_dir / "expert_splice.json").read_text())
    std_only = set(doc["standard_only_test"])
    stamp = utc_stamp()
    wdir = work / f"baselines_{stamp}"
    copy, diff = sandbox.make_sandbox(wdir, split_dir / "s2test" / "tasks", render_size=(640, 400),
                                      exclude=sandbox.LEAN_EXCLUDE)
    cwd = os.getcwd()
    t_all = time.time()
    try:
        wb = load_walkerbench(copy, sandbox.task_label(split_dir / "s2test" / "tasks"))
        if sorted(p.stem for p in wb.tasks_dir.glob("*.json")) != sorted(test):
            raise RuntimeError("sandbox task dir differs from split_s2.json s2test")
        res = {}
        logs = out / "logs" / stamp
        todo = list(policies) + (["RM_SPLICE"] if "RM" in policies and S.SPLICE_TASK in stems else [])
        for m in todo:
            t0 = time.time()
            if m == "RM_SPLICE":
                r = run_policy(wb, m, [S.SPLICE_TASK], logs / m, {S.SPLICE_TASK: splice["spliced_path"]})
            else:
                r = run_policy(wb, m, stems, logs / m)
            r["seconds"] = round(time.time() - t0, 1)
            if m != "RM_SPLICE":
                r["summary"] = summarise(r["per_task"], stems, std_only)
                s = r["summary"]
                print(f"{m:6s} nav {s['nav']['k']}/{s['nav']['n']} vis {s['vis']['k']}/{s['vis']['n']} "
                      f"all {s['all']['k']}/{s['all']['n']} | standard-only {s['standard_only']['k']}/"
                      f"{s['standard_only']['n']} | no log {s['tasks_without_log']} | {r['seconds']} s", flush=True)
            else:
                pt = r["per_task"].get(S.SPLICE_TASK, {})
                print(f"RM_SPLICE {S.SPLICE_TASK}: success {pt.get('success')} steps {pt.get('steps')} "
                      f"error {pt.get('error')}", flush=True)
            res[m] = r
        if wb.norender.calls == 0 or any((copy / "temp_images").rglob("*.jpg")):
            raise RuntimeError("rendering stub was not in effect")
        render_calls = wb.norender.calls
    finally:
        os.chdir(cwd)
        if not keep_sandbox:
            shutil.rmtree(wdir)  # this run's own scratch copy
    budget_hits = 0
    if "FWD" in res:
        for t in stems:
            n_vp = doc["tasks"][t]["n_path"]
            budget_hits += res["FWD"]["per_task"].get(t, {}).get("steps", -1) >= int(round(1.5 * n_vp))
    full_run = not subset and set(POLICIES) <= set(policies)
    fl = floors(res, stems) if {"ODO", "FWD", "STOP0"} <= set(res) else None
    hs = hard_subsets(res, stems) if {"ODO", "FWD"} <= set(res) else None
    warn = compare_survey(res, fl, hs, budget_hits, len(stems)) if full_run else ["subset run: survey comparison skipped"]
    for w in warn:
        print("WARN", w)
    base = {"schema": SCHEMA, "created_utc": stamp, "split_s2_sha256": S.sha256_file(sp_path),
            "code": {"file": "vpd/walkerbench/baselines.py", "sha256": S.sha256_file(__file__),
                     "sandbox_sha256": S.sha256_file(sandbox.__file__)},
            "walkerbench_commit": sandbox.repo_commit(), "sandbox_settings_diff": diff,
            "rendering": f"stubbed (generate_observation -> None, {render_calls} calls); no image made or sent",
            "loop": "scripts/run_benchmark_parallel.run_single_task + scripts/evaluate_all.evaluate_session",
            "step_cap": "round(1.5 * len(visual_path)) (stock)", "logs": str(logs), "seconds": round(time.time() - t_all, 1),
            "n_tasks": len(stems), "subset": subset, "policies_doc": __doc__.split("Policies")[1].split("Each task")[0],
            "survey_expected": {"per_kind": EXPECTED, "hard": EXPECTED_HARD, "pooled_F": EXPECTED_POOLED_F},
            "warnings": warn}
    bdoc = dict(base, floors=fl, fwd_step_cap_hits=budget_hits,
                expert={"RM": res["RM"]["summary"] if "RM" in res else None,
                        "RM_SPLICE": res.get("RM_SPLICE", {}).get("per_task")},
                policies=res)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "baselines_s2test.json", bdoc, force)
    if hs is not None:
        _write_json(out / "hard_subsets.json",
                    dict(hs, schema="vpd.walkerbench.hard_subsets/1", created_utc=stamp,
                         baselines_sha256=S.sha256_file(out / "baselines_s2test.json"),
                         split_s2_sha256=base["split_s2_sha256"]), force)
    if fl:
        print(f"floors: nav ODO {fl['nav_ODO']['k']}/{fl['nav_ODO']['n']}, vis FWD {fl['vis_FWD']['k']}/"
              f"{fl['vis_FWD']['n']}, pooled F {fl['pooled_F']['k']}/{fl['pooled_F']['n']} "
              f"({100 * fl['pooled_F']['rate']:.1f}%)")
    if hs:
        print(f"hard subsets: nav ODO-fail {hs['counts']['nav_odo_fail']}, vis FWD-fail {hs['counts']['vis_fwd_fail']}")
    print(f"wrote {out / 'baselines_s2test.json'}" + (f" and {out / 'hard_subsets.json'}" if hs else ""))
    return bdoc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-dir", type=Path, default=paths.WB_SPLIT)
    ap.add_argument("--out", type=Path, default=paths.WB_BASELINES)
    ap.add_argument("--work", type=Path, default=paths.scratch("wbdata"), help="sandbox parent (removed afterwards)")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--tasks", default=None, help="comma-separated s2test stems (subset run; needs --out)")
    ap.add_argument("--keep-sandbox", action="store_true")
    ap.add_argument("--force", action="store_true", help="replace existing output files")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any survey number differs")
    a = ap.parse_args(argv)
    pol = [p.strip() for p in a.policies.split(",") if p.strip()]
    if set(pol) - set(POLICIES):
        ap.error(f"unknown policies {set(pol) - set(POLICIES)}")
    stems = [t.strip() for t in a.tasks.split(",")] if a.tasks else None
    doc = run(a.split_dir, a.out, a.work, pol, stems, a.keep_sandbox, a.force)
    return 1 if (a.strict and doc["warnings"]) else 0


if __name__ == "__main__":
    thread_env()
    sys.exit(main())
