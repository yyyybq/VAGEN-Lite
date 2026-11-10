"""EXPLORATORY WalkerBench perception zero-shot (DECISIONS G5; ROUND3 section 3, WalkerBench, the "by the way" run).

Raw starts Q3B, Q3S and E2 (no fine-tuning) on the WalkerBench perception types angle / dis / height.  Numbers only,
absolute per start: nothing here computes a difference between starts, ranks starts, or applies a gate or decision
rule (ROUND3 section 3: outside a frozen prereg, start numbers are listed as absolute values only).  Counted
separately from the WB 30 GPU-h exploration budget.  Nothing is written under OUT/walkerbench/runs/pilot/ (refused).
Root R = OUT/walkerbench/runs/explore/perc.

    PY_EVAL -m vpd.walkerbench.perc build [--check]   # create, or verify, the task set perc294 (--check: verify only)
    PY_Q3   -m vpd.walkerbench.perc eval --start S     # run_vpd.main with perc294 registered in-process (run_vpd.py
                                                        # is not edited) and the fixed protocol below
    PY_EVAL -m vpd.walkerbench.perc analyze             # -> R/PERC.json and R/REPORT_EXPLORATORY.md
    PY_EVAL -m vpd.walkerbench.perc expect [--check] [--out F]   # the outputs to verify before `vpd.sco delete`
    PY_EVAL -m vpd.walkerbench.perc protocol            # print the run_vpd argv of every start
In-job: vpd/jobs/wb_perc.sh RUN (sco job vpd-wb-perc, as_root).

Task set perc294 = OUT/walkerbench/perception/perc294/{tasks/, selection.json}:
- sources: tasks/full and tasks/standard of the WalkerBench checkout (paths.REPO_WB @ b197f51), files angle_* / dis_* /
  height_*; distinct by file name, and a name present in both sources must have identical bytes (else refused);
- a task is kept when its spawn panorama paths.PANOS/<spawn>_z2.jpg exists (spawn = spawn_point or spawn_pano_id, the
  session manager's order; <id>_z<zoom>.jpg is the panorama cache's file name, zoom 2 = settings PANORAMA_ZOOM_LEVEL);
- expected 104 angle / 107 dis / 83 height = 294 (anything else is refused); flags per file: paper_full (in
  tasks/full, the paper's 100 per type) and standard20 (in tasks/standard);
- every copy is sha256-checked against its source; selection.json lists file -> sha256, the flags, the dropped files
  (spawn panorama missing), files with identical bytes under another name (expected none) and the source commit.
Protocol (identical for every start; the argv is built here, never passed through): --render-size 640x400,
--constrained off (the action regex's only STOP is STOP("ARRIVED"), which can never score on these types),
--lenient (the local stand-in for the reference agent's gpt-4o-mini JSON repair: the last action named in a reply),
--max-tokens 1024, --history-turns 0 (full history), --temperature 0, --workers 8, image-token audit on.  Run name
explore__<START>__raw__perc294__640__off_lenient under R/<START>/evals.
Upstream behaviour that applies to every start alike: angle / dis allow 20 steps, height 30 (max_steps null -> the
runner's default); the engine ends a height episode on a MOVE once 180 s have elapsed (max_time_seconds); the log
records no done_reason, so such endings appear as "end_other" below.
analyze: per start and type, success k/n with Wilson 95% on perc294, on the paper_full subset (100 / 100 / 76) and on
the standard20 subset (20 / 20 / 15); per type the logged steps and the parse-valid / repaired / unparseable / crash
steps (run_vpd.step_stats on each task log), how episodes end (STOP with a number, STOP without one, step cap, crash,
other), model calls cut at max_tokens (calls.jsonl); per start the run totals from results.json.  Every k / n and step
total is recomputed from per_task.jsonl and the logs and checked against results.json.  The paper's Qwen3-VL-8B
Original-setting row (Height 39.0, Dis 6.0, Angle 8.0 %) is listed next to the numbers, with no difference taken.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path

from vpd import paths

SCHEMA = "vpd.walkerbench.perc/1"
LABEL = "EXPLORATORY"
TASK_SET = "perc294"
PERC_DIR = paths.WB_OUT / "perception" / TASK_SET
PERC_TASKS = PERC_DIR / "tasks"
ROOT = paths.WB_RUNS / "explore" / "perc"
PILOT_ROOT = paths.WB_RUNS / "pilot"
STARTS = ("Q3B", "Q3S", "E2")  # fixed listing order (not a ranking)
TYPES = ("angle", "dis", "height")
SOURCES = ("full", "standard")  # checkout tasks/<source>
PANO_ZOOM = 2
EXPECTED = {"angle": 104, "dis": 107, "height": 83}
SUBSETS = ("perc294", "paper_full", "standard20")
DEFAULT_CAP = 30  # run_single_task's step cap when a task has neither visual_path nor max_steps
WILSON_Z = 1.959964
PROTOCOL = OrderedDict([("render_size", "640x400"), ("constrained", "off"), ("lenient", True), ("max_tokens", 1024),
                        ("history_turns", 0), ("temperature", 0.0), ("workers", 8)])
PAPER_REFERENCE = {
    "model": "qwen3-vl-8b-instruct", "setting": "Original",
    "success_pct": {"angle": 8.0, "dis": 6.0, "height": 39.0},
    "n": {"angle": 100, "dis": 100, "height": 100},
    "source": "WalkerBench paper (Ge et al., ECCV 2026) Table 3; the full set tasks/full (100 per type); the "
              "checkout's defaults are RENDER_OUTPUT_SIZE 1280x800 and gpt-4o-mini JSON repair",
    "note": "listed for reference only; our paper_full subset has 100 / 100 / 76 tasks (24 height tasks have no "
            "spawn panorama on disk), 640x400 and the local --lenient repair",
}
_NUM = re.compile(r"[-+]?\d*\.?\d+")  # evaluate_all.extract_number's pattern (b197f51), copied (not imported)


# ------------------------------------------------------------------------------------------------- utilities
def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def thread_env(n: int | None = None) -> None:
    """One thread per process unless VPD_THREADS says otherwise (login-node rule)."""
    v = str(n or os.environ.get("VPD_THREADS") or 1)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = v
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(p) -> list:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(p, obj) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def write_text(p, text: str) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return p


def is_under(path, root) -> bool:
    p, r = os.path.realpath(str(path)), os.path.realpath(str(root))
    return p == r or p.startswith(r.rstrip("/") + "/")


def refuse_pilot(path) -> None:
    """Every output of this exploratory run stays out of the pilot's tree."""
    if is_under(path, PILOT_ROOT):
        raise SystemExit(f"[perc] refused: {path} is under {PILOT_ROOT} (the pilot's outputs are read only)")


def wilson(k: float, n: int, z: float = WILSON_Z) -> tuple:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k, n) -> dict:
    if not n:
        return {"k": k, "n": 0, "rate": None, "wilson95": None}
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}


def extract_number(answer):
    """evaluate_all.extract_number (b197f51): float(answer), else the first number in it, else None."""
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    s = str(answer).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = _NUM.findall(s)
    if m:
        try:
            return float(m[0])
        except ValueError:
            return None
    return None


def task_type(name: str) -> str:
    return name.split("_", 1)[0]


def run_name(start: str) -> str:
    return f"explore__{start}__raw__{TASK_SET}__640__off_lenient"


def run_dir_of(start: str, root=ROOT) -> Path:
    return Path(root) / start / "evals" / run_name(start)


def task_cap(task: dict) -> int:
    """run_single_task's step budget: round(1.5 x len(visual_path)), else max_steps, else 30."""
    vp = task.get("visual_path") or []
    return int(round(len(vp) * 1.5)) if vp else int(task.get("max_steps") or DEFAULT_CAP)


# ------------------------------------------------------------------------------------------------- task set
def plan_set(repo=paths.REPO_WB, panos=paths.PANOS, expected=EXPECTED) -> dict:
    """The perc294 selection from the checkout's tasks/full + tasks/standard (refused on a name clash or on counts
    other than `expected`)."""
    from vpd.walkerbench import sandbox

    repo, panos = Path(repo), Path(panos)
    files: "OrderedDict[str, dict]" = OrderedDict()
    clashes = []
    for src in SOURCES:
        d = repo / "tasks" / src
        if not d.is_dir():
            raise SystemExit(f"[perc] {d} missing")
        for p in sorted(d.glob("*.json")):
            t = task_type(p.name)
            if t not in TYPES:
                continue
            data = p.read_bytes()
            sha = sha256_bytes(data)
            e = files.get(p.name)
            if e is None:
                task = json.loads(data)
                spawn = task.get("spawn_point") or task.get("spawn_pano_id")
                e = files[p.name] = {
                    "file": p.name, "type": t, "sha256": sha, "sources": [], "spawn_pano_id": spawn,
                    "panorama": bool(spawn) and (panos / f"{spawn}_z{PANO_ZOOM}.jpg").is_file(),
                    "step_cap": task_cap(task)}
            elif e["sha256"] != sha:
                clashes.append(f"{p.name} ({e['sources'][0]} vs {src})")
                continue
            e["sources"].append(src)
    if clashes:
        raise SystemExit(f"[perc] same file name, different bytes: {clashes[:5]} ({len(clashes)} in all)")
    kept, dropped = [], []
    for name in sorted(files):
        e = files[name]
        e["paper_full"] = "full" in e["sources"]
        e["standard20"] = "standard" in e["sources"]
        (kept if e["panorama"] else dropped).append(e)
    by_sha = defaultdict(list)
    for e in kept:
        by_sha[e["sha256"]].append(e["file"])
    same_bytes = sorted(v for v in by_sha.values() if len(v) > 1)
    counts = OrderedDict()
    for t in TYPES:
        k = [e for e in kept if e["type"] == t]
        counts[t] = {"kept": len(k), "paper_full": sum(e["paper_full"] for e in k),
                     "standard20": sum(e["standard20"] for e in k),
                     "dropped_no_panorama": sum(1 for e in dropped if e["type"] == t),
                     "step_cap": sorted({e["step_cap"] for e in k})}
    got = {t: counts[t]["kept"] for t in TYPES}
    if expected is not None and got != dict(expected):
        raise SystemExit(f"[perc] perc294 counts {got} differ from the expected {dict(expected)} (nothing written)")
    entries = [{k: e[k] for k in ("file", "type", "sha256", "sources", "paper_full", "standard20", "spawn_pano_id",
                                  "step_cap")} for e in kept]
    return {
        "schema": "vpd.walkerbench.perc294/1", "label": LABEL, "exploratory": True, "task_set": TASK_SET,
        "purpose": "EXPLORATORY perception zero-shot (DECISIONS G5): raw Q3B / Q3S / E2 on angle, dis, height; "
                   "numbers only, absolute per start, no between-start differences; not part of any pilot",
        "rule": "tasks/full + tasks/standard of the checkout, types angle/dis/height, distinct by file name (a name "
                "in both sources has identical bytes), kept when PANOS/<spawn>_z2.jpg exists",
        "source": {"repo": str(repo), "commit": sandbox.repo_commit(repo),
                   "dirs": {s: str(repo / "tasks" / s) for s in SOURCES}},
        "panoramas": str(panos), "counts": counts, "n_tasks": len(kept),
        "files": OrderedDict((e["file"], e["sha256"]) for e in kept), "entries": entries,
        "dropped": [{"file": e["file"], "type": e["type"], "sources": e["sources"], "spawn_pano_id": e["spawn_pano_id"],
                     "reason": "spawn panorama missing"} for e in dropped],
        "same_bytes_other_name": same_bytes,
    }


_PLAN_KEYS = ("schema", "task_set", "rule", "source", "panoramas", "counts", "n_tasks", "files", "entries", "dropped",
              "same_bytes_other_name")


def set_problems(dest, plan: dict) -> list:
    dest = Path(dest)
    probs = []
    sel = read_json(dest / "selection.json")
    if not sel:
        return [f"{dest}/selection.json missing or unreadable"]
    for k in _PLAN_KEYS:
        if json.loads(json.dumps(sel.get(k), default=str)) != json.loads(json.dumps(plan[k], default=str)):
            probs.append(f"{dest}/selection.json: {k} differs from the plan")
    have = {p.name: p for p in (dest / "tasks").glob("*.json")} if (dest / "tasks").is_dir() else {}
    if set(have) != set(plan["files"]):
        probs.append(f"{dest}/tasks: files differ (missing {sorted(set(plan['files']) - set(have))[:3]}, "
                     f"extra {sorted(set(have) - set(plan['files']))[:3]})")
    for f in sorted(set(have) & set(plan["files"])):
        if sha256_file(have[f]) != plan["files"][f]:
            probs.append(f"{dest}/tasks/{f}: sha256 differs")
    return probs


def build_set(dest=PERC_DIR, repo=paths.REPO_WB, panos=paths.PANOS, expected=EXPECTED, check_only=False) -> dict:
    """Create dest/{tasks/, selection.json} (atomic rename of a sibling temp dir) or verify an existing one."""
    dest, repo = Path(dest), Path(repo)
    refuse_pilot(dest)
    plan = plan_set(repo, panos, expected)
    counts = {t: plan["counts"][t]["kept"] for t in TYPES}
    if dest.exists() or check_only:
        probs = set_problems(dest, plan) if dest.exists() else [f"{dest} missing"]
        if probs:
            raise SystemExit("[perc] perc294 check failed (nothing overwritten): " + "; ".join(probs))
        return {"state": "verified", "dir": str(dest), "n_tasks": plan["n_tasks"], **counts}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    (tmp / "tasks").mkdir(parents=True)
    try:
        for e in plan["entries"]:
            src = repo / "tasks" / e["sources"][0] / e["file"]
            data = src.read_bytes()
            if sha256_bytes(data) != e["sha256"]:
                raise SystemExit(f"[perc] {src} changed while building")
            (tmp / "tasks" / e["file"]).write_bytes(data)
        write_json(tmp / "selection.json", dict(plan, created_utc=utc(), code_sha256={"perc.py": sha256_file(__file__)}))
        try:
            os.rename(tmp, dest)
            state = "created"
        except OSError:
            state = "raced"
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)  # this call's own temp dir
    probs = set_problems(dest, plan)
    if probs:
        raise SystemExit("[perc] perc294 check failed: " + "; ".join(probs))
    return {"state": state, "dir": str(dest), "n_tasks": plan["n_tasks"], **counts}


def register_task_sets(run_vpd=None, tasks_dir=PERC_TASKS):
    """Add 'perc294' to run_vpd.TASK_SETS in this process only (run_vpd.py is not edited)."""
    if run_vpd is None:
        from vpd.walkerbench import run_vpd
    cur = run_vpd.TASK_SETS.get(TASK_SET)
    if cur is None:
        run_vpd.TASK_SETS[TASK_SET] = Path(tasks_dir)
    elif Path(cur) != Path(tasks_dir):
        raise SystemExit(f"run_vpd.TASK_SETS[{TASK_SET!r}] is {cur}, expected {tasks_dir}")
    return run_vpd


# ------------------------------------------------------------------------------------------------- eval
def protocol_argv(start: str, root=ROOT) -> list:
    """The run_vpd argv of one start: the same flags for every start, only --model / --run-name / --out-root vary."""
    if start not in STARTS:
        raise ValueError(f"start must be one of {STARTS}")
    p = PROTOCOL
    return ["--model", start, "--task-set", TASK_SET, "--render-size", p["render_size"],
            "--constrained", p["constrained"], "--lenient", "--max-tokens", str(p["max_tokens"]),
            "--history-turns", str(p["history_turns"]), "--temperature", "0", "--workers", str(p["workers"]),
            "--run-name", run_name(start), "--out-root", str(Path(root) / start / "evals")]


def cmd_eval(argv: list, run_vpd=None, perc_dir=PERC_DIR, repo=paths.REPO_WB, panos=paths.PANOS,
             expected=EXPECTED) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.walkerbench.perc eval")
    ap.add_argument("--start", required=True, choices=STARTS)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier run of this start (run_vpd)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    refuse_pilot(root)
    perc_dir = Path(perc_dir)
    probs = set_problems(perc_dir, plan_set(repo, panos, expected)) if perc_dir.exists() else [f"{perc_dir} missing"]
    if probs:
        raise SystemExit("[perc] eval refused, perc294 check failed: " + "; ".join(probs))
    R = register_task_sets(run_vpd, perc_dir / "tasks")
    rv = protocol_argv(a.start, root) + (["--overwrite"] if a.overwrite else [])
    run_dir = (root / a.start / "evals" / run_name(a.start)).resolve()
    rc = 1
    try:
        R.main(rv)
        rc = 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if rc and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    finally:
        if run_dir.is_dir():
            code = {"vpd/walkerbench/perc.py": sha256_file(__file__)}
            try:
                code.update(R.code_sha256())
            except Exception as e:  # a fake run_vpd in tests
                code["run_vpd_code_error"] = f"{type(e).__name__}: {e}"[:200]
            write_json(run_dir / "explore.json", {
                "schema": SCHEMA + ":eval", "label": LABEL, "exploratory": True, "utc": utc(),
                "wrapper": "python -m vpd.walkerbench.perc eval", "start": a.start, "argv": argv,
                "run_vpd_argv": rv, "rc": rc, "protocol": dict(PROTOCOL),
                "task_sets_added": {TASK_SET: str(perc_dir / "tasks")},
                "selection_sha256": sha256_file(perc_dir / "selection.json"), "code": code})
    return rc


# ------------------------------------------------------------------------------------------------- analysis
END_KEYS = ("end_stop_number", "end_stop_no_number", "end_cap", "end_crash", "end_other", "no_log")
STEP_KEYS = ("steps_logged", "parse_valid_steps", "repaired_steps", "unparseable_steps", "invalid_move_crash_steps",
             "action_failed_steps", "api_failed_steps", "context_overflow_steps")


def episode_end(events: list, cap: int, R) -> str:
    """How one logged episode ended (descriptive): STOP with / without a number, step cap, crash tag, other."""
    acts = [e for e in events if e.get("event") == "action"]
    if not acts:
        return "end_other"
    last = acts[-1]
    if (last.get("action") or {}).get("type") == "stop":
        return "end_stop_number" if extract_number(last["action"].get("answer")) is not None else "end_stop_no_number"
    if R.step_kind(last.get("reason")) in ("invalid_move_crash", "action_failed"):
        return "end_crash"
    if len(acts) >= cap:
        return "end_cap"
    return "end_other"  # e.g. the engine's max_time (height: 180 s) after a MOVE


def protocol_problems(res: dict, manifest: dict | None, start: str) -> list:
    want = {"start": start, "model": start, "task_set": TASK_SET, "tasks_label": f"vpd_{TASK_SET}",
            "render_size": [640, 400], "constrained": PROTOCOL["constrained"], "lenient": True,
            "max_tokens": PROTOCOL["max_tokens"], "history_turns": PROTOCOL["history_turns"],
            "temperature": PROTOCOL["temperature"], "blind": False, "limit": None, "model_dir": None, "adapter": None,
            "run_name": run_name(start)}
    probs = [f"{k}: {res.get(k)!r} != {v!r}" for k, v in want.items() if res.get(k) != v]
    workers = ((manifest or {}).get("backend") or {}).get("workers")
    if workers != PROTOCOL["workers"]:
        probs.append(f"workers: {workers!r} != {PROTOCOL['workers']}")
    return probs


def analyze_start(run_dir: Path, sel: dict, start: str, R=None) -> dict:
    if R is None:
        from vpd.walkerbench import run_vpd as R
    run_dir = Path(run_dir)
    out = {"run_dir": str(run_dir), "run_name": run_dir.name}
    res_path = next((run_dir / n for n in ("results.json", "results_invalid.json") if (run_dir / n).exists()), None)
    if res_path is None or not (run_dir / "per_task.jsonl").exists():
        out["state"] = "missing"
        return out
    res = read_json(res_path) or {}
    out["state"] = "ok" if res_path.name == "results.json" else "invalid"
    out["results_file"] = res_path.name
    out["protocol_problems"] = protocol_problems(res, read_json(run_dir / "manifest.json"), start)
    out["protocol_ok"] = not out["protocol_problems"]
    flags = {Path(e["file"]).stem: e for e in sel["entries"]}
    rows = read_jsonl(run_dir / "per_task.jsonl")
    by_stem = {r["stem"]: r for r in rows}
    checks = []
    if sorted(by_stem) != sorted(flags) or len(rows) != len(by_stem):
        checks.append(f"per_task.jsonl stems != the {len(flags)} perc294 tasks "
                      f"(missing {sorted(set(flags) - set(by_stem))[:3]}, extra {sorted(set(by_stem) - set(flags))[:3]})")
    # success per type and subset, from per_task.jsonl
    succ = {}
    for subset in SUBSETS:
        block = OrderedDict()
        for t in TYPES:
            stems = [s for s, e in flags.items() if e["type"] == t and (subset == "perc294" or e[subset])]
            k = sum(int((by_stem.get(s) or {}).get("success") or 0) for s in stems)
            block[t] = rate(k, len(stems))
        succ[subset] = block
    out["success"] = succ
    for t in TYPES:
        pt = (res.get("per_type") or {}).get(t) or {}
        if (pt.get("success"), pt.get("n")) != (succ["perc294"][t]["k"], succ["perc294"][t]["n"]):
            checks.append(f"{t}: per_task.jsonl {succ['perc294'][t]['k']}/{succ['perc294'][t]['n']} != results.json "
                          f"per_type {pt.get('success')}/{pt.get('n')}")
    # steps and episode endings per type, from the task logs
    logs = run_dir / "WalkerBench" / "logs"
    steps = {t: OrderedDict((k, 0) for k in STEP_KEYS) for t in TYPES}
    ends = {t: OrderedDict((k, 0) for k in END_KEYS) for t in TYPES}
    for s, e in flags.items():
        t = e["type"]
        row = by_stem.get(s) or {}
        log = logs / row["log"] if row.get("log") else None
        if log is None or not log.exists():
            ends[t]["no_log"] += 1
            continue
        events = R.read_events(log)
        for k, v in R.step_stats(events).items():
            if k in steps[t]:
                steps[t][k] += v
        ends[t][episode_end(events, int(e["step_cap"]), R)] += 1
    for t in TYPES:
        n = steps[t]["steps_logged"]
        steps[t]["parse_valid_pct"] = round(100 * steps[t]["parse_valid_steps"] / n, 2) if n else None
        steps[t]["repaired_pct"] = round(100 * steps[t]["repaired_steps"] / n, 2) if n else None
        steps[t]["mean_steps"] = round(n / max(1, succ["perc294"][t]["n"] - ends[t]["no_log"]), 2)
    out["steps"] = steps
    out["ends"] = ends
    for k in ("steps_logged", "parse_valid_steps", "repaired_steps", "unparseable_steps"):
        tot = sum(steps[t][k] for t in TYPES)
        if res.get(k) is not None and tot != res.get(k):
            checks.append(f"{k}: logs {tot} != results.json {res.get(k)}")
    # model calls cut at max_tokens, per type (calls.jsonl)
    calls = defaultdict(lambda: {"calls": 0, "errors": 0, "finish_length": 0})
    if (run_dir / "calls.jsonl").exists():
        for c in read_jsonl(run_dir / "calls.jsonl"):
            t = task_type(str(c.get("task") or ""))
            if t not in TYPES:
                continue
            calls[t]["calls"] += 1
            calls[t]["errors"] += c.get("error") is not None
            calls[t]["finish_length"] += c.get("finish_reason") == "length"
    out["calls"] = {t: dict(calls[t]) for t in TYPES}
    reg = res.get("regex") or {}
    img = res.get("image_tokens") or {}
    out["totals"] = {
        "n_tasks": res.get("n_tasks"), "tasks_without_log": res.get("tasks_without_log"),
        "steps_logged": res.get("steps_logged"), "parse_valid_steps": res.get("parse_valid_steps"),
        "parse_valid_pct": res.get("parse_valid_pct"), "repaired_steps": res.get("repaired_steps"),
        "unparseable_steps": res.get("unparseable_steps"), "unparseable_pct": res.get("unparseable_pct"),
        "invalid_move_crash": res.get("invalid_move_crash"), "api_failed_steps": res.get("api_failed_steps"),
        "finish_length": reg.get("finish_length"), "model_calls": reg.get("calls"),
        "lenient_repairs": (res.get("counts") or {}).get("lenient_repairs"),
        "lenient_no_action": (res.get("counts") or {}).get("lenient_no_action"),
        "server_alive_at_end": res.get("server_alive_at_end"), "wall_seconds": res.get("wall_seconds"),
        "seconds_per_step": res.get("seconds_per_step"),
        "server_ready_seconds": (res.get("server") or {}).get("ready_seconds"),
        "image_tokens_per_image": img.get("tokens_per_image") if isinstance(img, dict) else None,
        "served_model_name": res.get("served_model_name")}
    out["consistency_problems"] = checks
    out["consistent"] = not checks
    return out


def analyze(root=ROOT, perc_dir=PERC_DIR, starts=STARTS, R=None, write=True) -> dict:
    root, perc_dir = Path(root), Path(perc_dir)
    refuse_pilot(root)
    sel = read_json(perc_dir / "selection.json")
    if not sel:
        raise SystemExit(f"[perc] {perc_dir}/selection.json missing")
    doc = OrderedDict([
        ("schema", SCHEMA + ":analysis"), ("label", LABEL), ("exploratory", True), ("utc", utc()),
        ("decision", "DECISIONS G5: numbers only, absolute per start; no between-start difference, ranking, gate or "
                     "decision rule is computed (ROUND3 section 3)"),
        ("task_set", {"name": TASK_SET, "dir": str(perc_dir), "selection_sha256": sha256_file(perc_dir / "selection.json"),
                      "counts": sel.get("counts"), "n_tasks": sel.get("n_tasks")}),
        ("protocol", dict(PROTOCOL)), ("paper_reference", PAPER_REFERENCE),
        ("starts", OrderedDict((s, analyze_start(run_dir_of(s, root), sel, s, R)) for s in starts)),
        ("code_sha256", {"perc.py": sha256_file(__file__)}),
    ])
    if write:
        write_json(root / "PERC.json", doc)
        write_text(root / "REPORT_EXPLORATORY.md", report_md(doc))
    return doc


def _cell(b: dict) -> str:
    if not b or not b.get("n"):
        return "—"
    lo, hi = b["wilson95"]
    return f"{b['k']}/{b['n']} ({100 * b['rate']:.1f}%, [{100 * lo:.1f}, {100 * hi:.1f}])"


def report_md(doc: dict) -> str:
    ts = doc["task_set"]
    c = ts.get("counts") or {}
    L = [f"# {LABEL}：WalkerBench 感知题零样本（Q3B / Q3S / E2，raw）", "",
         f"生成时间 {doc['utc']}（UTC）。DECISIONS G5。只列各起点的绝对数：不算起点之间的差、不排序、没有闸门。"
         f"不计入 WalkerBench 的 30 卡时探索预算。", "",
         f"- 题集 {ts['name']}（{ts['n_tasks']} 题）：tasks/full + tasks/standard 的 angle / dis / height，按文件名去重，"
         f"只留出生点全景图在盘上的题。angle {c.get('angle', {}).get('kept')}、dis {c.get('dis', {}).get('kept')}、"
         f"height {c.get('height', {}).get('kept')}；其中论文全集（paper_full）"
         f"{c.get('angle', {}).get('paper_full')} / {c.get('dis', {}).get('paper_full')} / "
         f"{c.get('height', {}).get('paper_full')}。selection.json sha256 {ts['selection_sha256'][:12]}。",
         "- 协议（每个起点完全相同）：640x400，--constrained off，--lenient（本地修复规则，代替参考实现的 gpt-4o-mini），"
         "max_tokens 1024，完整历史，temperature 0，8 个 worker。", ""]
    starts = doc["starts"]
    for subset, title in (("perc294", f"成功率，{TASK_SET} 全部"), ("paper_full", "成功率，论文全集子集（tasks/full）"),
                          ("standard20", "成功率，Standard-20 子集（tasks/standard）")):
        L += [f"## {title}（k/n，95% Wilson）", "", "| 起点 | 状态 | angle | dis | height |", "|---|---|---|---|---|"]
        for s, a in starts.items():
            if a.get("state") == "missing":
                L.append(f"| {s} | 缺 | — | — | — |")
                continue
            b = a["success"][subset]
            L.append(f"| {s} | {a['state']} | " + " | ".join(_cell(b[t]) for t in TYPES) + " |")
        if subset == "paper_full":
            ref = doc["paper_reference"]
            L.append(f"| 论文 {ref['model']}（{ref['setting']}，n=100/类，仅作参照） | — | "
                     + " | ".join(f"{ref['success_pct'][t]:.1f}%" for t in TYPES) + " |")
        L.append("")
    L += ["## 格式和步数（每类：记录步数、可解析 %、修复步数、无法解析步数；结束方式）", "",
          "| 起点 | 类型 | 步数 | 可解析 % | 修复步 | 无法解析 | 崩溃步 | STOP+数字 | STOP 无数字 | 到步数上限 | 其他结束 | 无日志 | 截断调用 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, a in starts.items():
        if a.get("state") == "missing":
            continue
        for t in TYPES:
            st, en, ca = a["steps"][t], a["ends"][t], a["calls"][t]
            L.append(f"| {s} | {t} | {st['steps_logged']} | {st['parse_valid_pct']} | {st['repaired_steps']} | "
                     f"{st['unparseable_steps']} | {st['invalid_move_crash_steps'] + st['action_failed_steps']} | "
                     f"{en['end_stop_number']} | {en['end_stop_no_number']} | {en['end_cap']} | {en['end_other']} | "
                     f"{en['no_log']} | {ca['finish_length']} |")
    L += ["", "## 运行", "", "| 起点 | 结果文件 | 协议一致 | 复算一致 | 总步数 | 可解析 % | 修复步 | 秒/步 | 墙钟秒 | 服务器存活 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s, a in starts.items():
        if a.get("state") == "missing":
            L.append(f"| {s} | 缺 | — | — | — | — | — | — | — | — |")
            continue
        tt = a["totals"]
        L.append(f"| {s} | {a['results_file']} | {'是' if a['protocol_ok'] else '否'} | "
                 f"{'是' if a['consistent'] else '否'} | {tt['steps_logged']} | {tt['parse_valid_pct']} | "
                 f"{tt['repaired_steps']} | {tt['seconds_per_step']} | {tt['wall_seconds']} | {tt['server_alive_at_end']} |")
    probs = [(s, p) for s, a in starts.items() for p in (a.get("protocol_problems") or []) + (a.get("consistency_problems") or [])]
    if probs:
        L += ["", "## 问题", ""] + [f"- {s}: {p}" for s, p in probs]
    L += ["", "说明：", "- 「其他结束」包括引擎的 180 秒限时（只有 height 题有 max_time_seconds）：日志里没有 done_reason，"
          "所以只能归到这里（推断）。",
          "- 「修复步」是 --lenient 规则从回复里取出最后一个动作的步；它们不算可解析步。",
          "- 论文那一行是全集 100 题/类、1280x800、gpt-4o-mini 修复的数，只作参照，不与本表相减。",
          "", "---", "本报告只列数字，没有结论，没有闸门。"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------- outputs
def expected_outputs(root=ROOT, perc_dir=PERC_DIR, starts=STARTS) -> list:
    root, perc_dir = Path(root), Path(perc_dir)
    out = [perc_dir / "selection.json"]
    for s in starts:
        d = run_dir_of(s, root)
        out += [d / f for f in ("results.json", "per_task.jsonl", "manifest.json", "explore.json")]
    out += [root / "PERC.json", root / "REPORT_EXPLORATORY.md"]
    return out


def expect(root=ROOT, perc_dir=PERC_DIR, starts=STARTS) -> tuple:
    """(lines, n_problems): every expected file, then PERC.json's per-start state / protocol / consistency."""
    lines, bad = [], 0
    for p in expected_outputs(root, perc_dir, starts):
        ok = p.exists()
        bad += not ok
        lines.append(f"{'ok     ' if ok else 'MISSING'} {p}")
    doc = read_json(Path(root) / "PERC.json")
    for s in starts:
        a = ((doc or {}).get("starts") or {}).get(s)
        good = bool(a) and a.get("state") == "ok" and a.get("protocol_ok") and a.get("consistent")
        bad += not good
        detail = "no PERC.json entry" if not a else (f"state {a.get('state')}, protocol_ok {a.get('protocol_ok')}, "
                                                     f"consistent {a.get('consistent')}")
        lines.append(f"{'ok     ' if good else 'PROBLEM'} PERC.json {s}: {detail}")
    return lines, bad


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        return cmd_eval(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="create or verify OUT/walkerbench/perception/perc294")
    p.add_argument("--dest", type=Path, default=PERC_DIR)
    p.add_argument("--check", action="store_true", help="verify only (exit 1 when missing or different)")
    sub.add_parser("eval", help="run_vpd.main on perc294 with the fixed protocol (see module docstring)")
    for nm in ("analyze", "expect", "protocol"):
        p = sub.add_parser(nm)
        p.add_argument("--root", type=Path, default=ROOT)
        if nm != "protocol":
            p.add_argument("--perc-dir", type=Path, default=PERC_DIR)
        if nm == "expect":
            p.add_argument("--check", action="store_true", help="exit 1 when anything is missing or not ok")
            p.add_argument("--out", type=Path, default=None, help="also write the list to this file (RUN/expect.txt)")
    args = ap.parse_args(argv)
    thread_env()
    if args.cmd == "build":
        try:
            print(json.dumps(build_set(args.dest, check_only=args.check)))
        except SystemExit as e:
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr)
                return 1
            raise
    elif args.cmd == "analyze":
        doc = analyze(args.root, args.perc_dir)
        print(f"[perc] wrote {Path(args.root) / 'PERC.json'} and {Path(args.root) / 'REPORT_EXPLORATORY.md'}")
        for s, a in doc["starts"].items():
            if a.get("state") == "missing":
                print(f"  {s:4s} missing")
                continue
            b = a["success"]["perc294"]
            print(f"  {s:4s} {a['state']:7s} " + "  ".join(f"{t} {b[t]['k']}/{b[t]['n']}" for t in TYPES)
                  + f"  protocol_ok {a['protocol_ok']} consistent {a['consistent']}")
    elif args.cmd == "expect":
        lines, bad = expect(args.root, args.perc_dir)
        lines.append(f"[perc] {bad} missing or not ok")
        print("\n".join(lines))
        if args.out is not None:
            refuse_pilot(args.out)
            write_text(args.out, "\n".join(lines) + "\n")
        return 1 if (args.check and bad) else 0
    elif args.cmd == "protocol":
        for s in STARTS:
            print(s, " ".join(protocol_argv(s, args.root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
