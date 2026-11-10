"""EXPLORATORY WalkerBench perception zero-shot (DECISIONS G5; ROUND3 section 3, WalkerBench, the "by the way" run).

Raw starts Q3B, Q3S and E2 (no fine-tuning) on the WalkerBench perception types angle / dis / height.  Numbers only,
absolute per start: nothing here computes a difference between starts, ranks starts, or applies a gate or decision
rule (ROUND3 section 3: outside a frozen prereg, start numbers are listed as absolute values only).  Counted
separately from the WB 30 GPU-h exploration budget.  Nothing is written under OUT/walkerbench/runs/pilot/ (refused).
Root R = OUT/walkerbench/runs/explore/perc.

    PY_EVAL -m vpd.walkerbench.perc build [--check]   # create, or verify, the task set perc294 (--check: verify only)
    PY_Q3   -m vpd.walkerbench.perc eval --start S     # run_vpd.main with perc294 registered in-process (run_vpd.py
                                                        # is not edited) and the fixed protocol below; in a job only
    PY_EVAL -m vpd.walkerbench.perc analyze             # -> R/PERC.json and R/REPORT_EXPLORATORY.md
    PY_EVAL -m vpd.walkerbench.perc expect [--check] [--out F]   # the outputs to verify before `vpd.sco delete`
    PY_EVAL -m vpd.walkerbench.perc protocol            # print the run_vpd argv of every start
In-job: vpd/jobs/wb_perc.sh RUN [START ...] (sco job vpd-wb-perc, as_root; default all three starts).
eval guards (it starts vLLM):
- it runs only inside a sco job: vpd.snapshot.snapshot_state() must be in mode "snapshot" (the job's code snapshot
  RUN/code, DECISIONS G1) and env RUN must be an absolute path (wb_perc.sh exports it).  Env
  VPD_PERC_TEST_ALLOW_LIVE=1 is the override for the hermetic tests only (never set it on the login node);
- per-start run-dir lock R/<START>/evals/perc_eval.lock, created with O_EXCL and holding {host, pid, utc, run,
  start, token}; removed when the eval returns.  An existing lock is taken over when it names this same RUN, or when
  its RUN has finished (RUN/DONE, or RUN/job/exit_code written by launch.sh after the payload exited, also after a
  timeout).  Otherwise eval is refused, unless env VPD_PERC_TAKEOVER=1: set it only on a resubmission after the
  first job has reached a terminal state (`vpd.sco status`) without writing either marker (deleted job, lost node);
- right after taking the lock: a run dir that already holds results.json (exit 0) or results_invalid.json (exit 1)
  is not re-run and its explore.json is not rewritten unless --overwrite is given (run_vpd would refuse it anyway).

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
Upstream behaviour that applies to every start alike (the engine and the limit are not changed here): angle / dis
allow 20 steps, height 30 (max_steps null -> the runner's default); every height task has max_time_seconds 180 (angle
and dis have none).  That limit is wall-clock time from session start (session_manager elapsed_time), checked only
after a MOVE (action_executor), so it depends on decoding speed and on the 8 workers sharing one vLLM server: a start
with longer replies loses height steps to the clock under identical flags.  The log records no done_reason, so
analyze infers these endings (end_max_time, labelled 推断, END_MAX_TIME_RULE below).
analyze: per start and type, success k/n with Wilson 95% on perc294, on the paper_full subset (100 / 100 / 76) and on
the standard20 subset (20 / 20 / 15); per type the logged steps and the parse-valid / repaired / unparseable / crash
steps (run_vpd.step_stats on each task log), how episodes end (STOP with a number, STOP without one, step cap,
max_time (inferred), crash, other), model calls cut at max_tokens (calls.jsonl), completion_tokens per answered call
(calls.jsonl) and seconds per step and per episode (log timestamps), each as n / mean / p95 / max; per start the run
totals from results.json and the job that produced it (explore.json).  Every k / n and step total is recomputed from
per_task.jsonl and the logs and checked against results.json.  The paper's Qwen3-VL-8B Original-setting row (Height
39.0, Dis 6.0, Angle 8.0 %) is listed next to the numbers, with no difference taken.  analyze and expect always read
all three starts from R/<START>/evals/, whichever job produced each.
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
import socket
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
ALLOW_LIVE_ENV = "VPD_PERC_TEST_ALLOW_LIVE"  # hermetic tests only: eval outside a job's code snapshot
TAKEOVER_ENV = "VPD_PERC_TAKEOVER"  # resubmission after the first job reached a terminal state without a marker
LOCK_NAME = "perc_eval.lock"  # R/<START>/evals/perc_eval.lock
START_TS_FORMAT = "%Y%m%d_%H%M%S"  # session_start "timestamp" (run_benchmark_parallel, b197f51)


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


def job_state() -> dict:
    """The code-snapshot state when perc eval may run, else SystemExit.  eval starts vLLM, so it runs only inside a sco
    job, whose code is the snapshot RUN/code (vpd.snapshot mode "snapshot", DECISIONS G1).  Env
    VPD_PERC_TEST_ALLOW_LIVE=1 lets the hermetic tests (fake run_vpd) run it from the live tree."""
    from vpd import snapshot

    try:
        st = snapshot.snapshot_state()
    except snapshot.SnapshotError as e:
        st = {"mode": "unreadable", "root": str(paths.WORKTREE), "error": str(e)[:300]}
    keep = {k: st.get(k) for k in ("mode", "root", "commit", "utc", "error") if st.get(k) is not None}
    if st.get("mode") == "snapshot":
        return keep
    if os.environ.get(ALLOW_LIVE_ENV) == "1":
        return dict(keep, override=f"{ALLOW_LIVE_ENV}=1")
    raise SystemExit(f"[perc] eval refused: not inside a job (vpd.snapshot mode {st.get('mode')!r} at "
                     f"{st.get('root')}).  perc eval starts vLLM; it runs only in the sco job vpd-wb-perc "
                     f"(vpd/jobs/wb_perc.sh, code snapshot RUN/code).  {ALLOW_LIVE_ENV}=1 is for the hermetic tests "
                     f"only")


def job_run() -> Path:
    """env RUN (the job's run dir; wb_perc.sh exports it), which names the holder in the run-dir lock."""
    run = os.environ.get("RUN") or ""
    if not os.path.isabs(run):
        raise SystemExit(f"[perc] eval refused: env RUN={run!r} is not an absolute path (the job's run dir, exported by "
                         f"vpd/jobs/wb_perc.sh)")
    return Path(run)


def lock_path(start: str, root=ROOT) -> Path:
    return Path(root) / start / "evals" / LOCK_NAME


def run_finished(run) -> bool:
    """A job's RUN recorded a terminal state: RUN/DONE (every step passed) or RUN/job/exit_code (launch.sh writes it
    after the payload exits, also after the outer timeout)."""
    r = Path(str(run))
    return (r / "DONE").exists() or (r / "job" / "exit_code").exists()


def takeover_reason(holder, run) -> str | None:
    """Why an existing lock may be taken over by RUN `run`, or None (then eval is refused)."""
    if isinstance(holder, dict) and holder.get("run"):
        if str(holder["run"]) == str(run):
            return "same RUN (an earlier process of this job)"
        if run_finished(holder["run"]):
            return "the holder's RUN has finished (DONE or job/exit_code)"
    if os.environ.get(TAKEOVER_ENV) == "1":
        return f"{TAKEOVER_ENV}=1"
    return None


def acquire_lock(path, run, start: str) -> dict:
    """Take the run-dir lock of one start: an O_EXCL file holding {host, pid, utc, run, start, token}.  A lock that
    takeover_reason() allows is moved aside (atomic rename; if the moved file is not the one judged, it is put back
    and eval is refused) and replaced; any other existing lock refuses with SystemExit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    me = OrderedDict([("schema", SCHEMA + ":lock"), ("host", socket.gethostname()), ("pid", os.getpid()),
                      ("utc", utc()), ("run", str(run)), ("start", start), ("token", uuid.uuid4().hex)])
    took_over = None
    for _ in range(5):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = read_json(path)
            why = takeover_reason(holder, run)
            if why is None:
                h = holder if isinstance(holder, dict) else {}
                raise SystemExit(
                    f"[perc] eval refused: {path} is held by RUN {h.get('run')!r} (host {h.get('host')!r}, pid "
                    f"{h.get('pid')!r}, since {h.get('utc')!r}; unreadable lock: {not isinstance(holder, dict)}), which "
                    f"has neither DONE nor job/exit_code.  Resubmit only after that job has ended; if it reached a "
                    f"terminal state without either marker (`vpd.sco status`), resubmit with {TAKEOVER_ENV}=1")
            aside = path.with_name(f"{path.name}.takeover-{os.getpid()}-{uuid.uuid4().hex[:6]}")
            try:
                os.rename(path, aside)
            except FileNotFoundError:
                continue  # released or moved meanwhile: try again
            if read_json(aside) != holder:  # a newer lock written after our read: put it back, refuse
                try:
                    os.link(aside, path)
                    os.unlink(aside)
                except OSError:
                    pass  # the path was taken again meanwhile; the moved file stays next to it
                raise SystemExit(f"[perc] eval refused: {path} changed while being taken over; retry")
            os.unlink(aside)
            took_over = {"holder": holder, "reason": why, "utc": utc()}
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(dict(me, took_over=took_over), indent=1) + "\n")
        return dict(me, path=str(path), took_over=took_over)
    raise SystemExit(f"[perc] eval refused: could not take {path} (it kept changing)")


def release_lock(lock: dict) -> bool:
    """Remove the lock if it is still ours (same token)."""
    p = Path(lock["path"])
    cur = read_json(p)
    if isinstance(cur, dict) and cur.get("token") == lock["token"]:
        p.unlink()
        return True
    return False


def finished_result(run_dir) -> str | None:
    """"results.json" or "results_invalid.json" when run_dir holds that finished run (run_vpd's own refusal rule
    without --overwrite), else None."""
    return next((n for n in ("results.json", "results_invalid.json") if (Path(run_dir) / n).exists()), None)


def cmd_eval(argv: list, run_vpd=None, perc_dir=PERC_DIR, repo=paths.REPO_WB, panos=paths.PANOS,
             expected=EXPECTED) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.walkerbench.perc eval")
    ap.add_argument("--start", required=True, choices=STARTS)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier run of this start (run_vpd)")
    a = ap.parse_args(argv)
    job = job_state()
    run = job_run()
    root = Path(a.root)
    refuse_pilot(root)
    perc_dir = Path(perc_dir)
    probs = set_problems(perc_dir, plan_set(repo, panos, expected)) if perc_dir.exists() else [f"{perc_dir} missing"]
    if probs:
        raise SystemExit("[perc] eval refused, perc294 check failed: " + "; ".join(probs))
    R = register_task_sets(run_vpd, perc_dir / "tasks")
    rv = protocol_argv(a.start, root) + (["--overwrite"] if a.overwrite else [])
    run_dir = (root / a.start / "evals" / run_name(a.start)).resolve()
    lock = acquire_lock(lock_path(a.start, root), run, a.start)
    rc = 1
    try:
        # checked under the lock: a run finished meanwhile (e.g. by another job, after the payload's own skip check)
        # is left exactly as it is, explore.json included, unless --overwrite (run_vpd would refuse it anyway)
        done = finished_result(run_dir)
        if done is not None and not a.overwrite:
            print(f"[perc] {run_dir / done} exists: {a.start} has a finished run, not re-run, explore.json left "
                  f"untouched (--overwrite replaces the run)", file=sys.stderr)
            return 0 if done == "results.json" else 1
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
                    "run_vpd_argv": rv, "rc": rc, "protocol": dict(PROTOCOL), "run": str(run), "code_snapshot": job,
                    "lock": {k: lock[k] for k in ("path", "host", "pid", "utc", "took_over")},
                    "task_sets_added": {TASK_SET: str(perc_dir / "tasks")},
                    "selection_sha256": sha256_file(perc_dir / "selection.json"), "code": code})
    finally:
        release_lock(lock)
    return rc


# ------------------------------------------------------------------------------------------------- analysis
END_KEYS = ("end_stop_number", "end_stop_no_number", "end_cap", "end_max_time", "end_crash", "end_other", "no_log")
STEP_KEYS = ("steps_logged", "parse_valid_steps", "repaired_steps", "unparseable_steps", "invalid_move_crash_steps",
             "action_failed_steps", "api_failed_steps", "context_overflow_steps")
END_MAX_TIME_RULE = (
    "推断 (inference; the log has no done_reason): end_max_time when, after the STOP and crash checks, the last logged "
    "action is a MOVE, fewer actions are logged than the step cap, the task has max_time_seconds (180 for every height "
    "task; angle / dis have none) and the last action's ISO timestamp minus the session_start timestamp "
    f"({START_TS_FORMAT}) is at least max_time_seconds.  The engine's clock is wall time from session start, checked "
    "after a MOVE.  The start stamp is truncated to the second and taken just before the session is created (elapsed "
    "over-stated by up to ~1 s); an action's stamp is taken before that action executes, while the engine checks after "
    "the MOVE (elapsed under-stated), so an episode cut close to the limit can stay in end_other.")
P95_RULE = "nearest rank: the ceil(0.95 n)-th smallest value"


def parse_start_ts(v):
    try:
        return _dt.datetime.strptime(str(v), START_TS_FORMAT)
    except (TypeError, ValueError):
        return None


def parse_iso_ts(v):
    try:
        d = _dt.datetime.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo is None else None  # upstream writes naive local times, as for session_start


def action_elapsed(events: list) -> list | None:
    """Seconds from the session_start stamp to each logged action's stamp, in log order; None when the session_start
    stamp or any action stamp is missing or unreadable."""
    start = next((e for e in events if e.get("event") == "session_start"), None)
    t0 = parse_start_ts((start or {}).get("timestamp"))
    if t0 is None:
        return None
    out = []
    for e in events:
        if e.get("event") != "action":
            continue
        t = parse_iso_ts(e.get("timestamp"))
        if t is None:
            return None
        out.append((t - t0).total_seconds())
    return out


def dist(xs, nd: int = 1) -> dict:
    """n, mean, p95 (P95_RULE) and max of a list of numbers (None when empty)."""
    v = sorted(float(x) for x in xs)
    if not v:
        return {"n": 0, "mean": None, "p95": None, "max": None}
    n = len(v)
    rank = -(-95 * n // 100)  # ceil(0.95 n) in integers (no float rounding at n = 20, 100, ...)
    return {"n": n, "mean": round(sum(v) / n, nd), "p95": round(v[rank - 1], nd), "max": round(v[-1], nd)}


def episode_end(events: list, cap: int, R, max_time=None) -> str:
    """How one logged episode ended (descriptive): STOP with / without a number, crash tag, step cap, the engine's
    time limit (inferred, END_MAX_TIME_RULE; only for a task with max_time_seconds), other."""
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
    if max_time and (last.get("action") or {}).get("type") == "move":
        el = action_elapsed(events)
        if el and el[-1] >= float(max_time):
            return "end_max_time"
    return "end_other"


def protocol_want(start: str) -> dict:
    """results.json values every start must have (all keys are run_vpd.RESULT_META_KEYS; a test checks this)."""
    return {"start": start, "model": start, "task_set": TASK_SET, "tasks_label": f"vpd_{TASK_SET}",
            "render_size": [640, 400], "constrained": PROTOCOL["constrained"], "lenient": True,
            "max_tokens": PROTOCOL["max_tokens"], "history_turns": PROTOCOL["history_turns"],
            "temperature": PROTOCOL["temperature"], "blind": False, "limit": None, "model_dir": None, "adapter": None,
            "run_name": run_name(start)}


def protocol_problems(res: dict, manifest: dict | None, start: str) -> list:
    want = protocol_want(start)
    probs = [f"{k}: {res.get(k)!r} != {v!r}" for k, v in want.items() if res.get(k) != v]
    workers = ((manifest or {}).get("backend") or {}).get("workers")
    if workers != PROTOCOL["workers"]:
        probs.append(f"workers: {workers!r} != {PROTOCOL['workers']}")
    return probs


def task_limits(perc_dir, sel: dict) -> dict:
    """stem -> the task's max_time_seconds (None when it has none), from perc294's own task copies."""
    out = {}
    for e in sel["entries"]:
        t = read_json(Path(perc_dir) / "tasks" / e["file"])
        out[Path(e["file"]).stem] = t.get("max_time_seconds") if isinstance(t, dict) else None
    return out


def analyze_start(run_dir: Path, sel: dict, start: str, R=None, limits: dict | None = None) -> dict:
    """One start's numbers; `limits` = task_limits() (without it no episode is classed end_max_time)."""
    if R is None:
        from vpd.walkerbench import run_vpd as R
    run_dir = Path(run_dir)
    limits = limits or {}
    out = {"run_dir": str(run_dir), "run_name": run_dir.name}
    res_path = next((run_dir / n for n in ("results.json", "results_invalid.json") if (run_dir / n).exists()), None)
    if res_path is None or not (run_dir / "per_task.jsonl").exists():
        out["state"] = "missing"
        return out
    res = read_json(res_path) or {}
    out["state"] = "ok" if res_path.name == "results.json" else "invalid"
    out["results_file"] = res_path.name
    ex = read_json(run_dir / "explore.json")
    ex = ex if isinstance(ex, dict) else {}
    out["produced_by"] = {"run": ex.get("run"), "utc": ex.get("utc"), "rc": ex.get("rc"),
                          "snapshot_commit": (ex.get("code_snapshot") or {}).get("commit"),
                          "perc_py_sha256": (ex.get("code") or {}).get("vpd/walkerbench/perc.py")}
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
    step_s = {t: [] for t in TYPES}
    episode_s = {t: [] for t in TYPES}
    no_ts = {t: 0 for t in TYPES}
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
        ends[t][episode_end(events, int(e["step_cap"]), R, limits.get(s))] += 1
        el = action_elapsed(events)
        if el is None:
            no_ts[t] += 1
        elif el:
            step_s[t] += [b - a for a, b in zip([0.0] + el[:-1], el)]
            episode_s[t].append(el[-1])
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
    tokens = defaultdict(list)
    if (run_dir / "calls.jsonl").exists():
        for c in read_jsonl(run_dir / "calls.jsonl"):
            t = task_type(str(c.get("task") or ""))
            if t not in TYPES:
                continue
            calls[t]["calls"] += 1
            calls[t]["errors"] += c.get("error") is not None
            calls[t]["finish_length"] += c.get("finish_reason") == "length"
            ct = c.get("completion_tokens")
            if c.get("error") is None and isinstance(ct, (int, float)) and not isinstance(ct, bool):
                tokens[t].append(ct)
    out["calls"] = {t: dict(calls[t]) for t in TYPES}
    # decoding length and speed per type: what the wall-clock height limit depends on (descriptive)
    out["timing"] = {t: OrderedDict([
        ("completion_tokens", dist(tokens[t])),  # per answered call (calls.jsonl)
        ("step_seconds", dist(step_s[t], 2)),  # per logged step: the gap to the previous stamp (session_start first)
        ("episode_seconds", dist(episode_s[t])),  # per episode: last action's stamp minus session_start
        ("episodes_without_timestamps", no_ts[t]),
        ("max_time_seconds", sorted({limits.get(s) for s, e in flags.items() if e["type"] == t},
                                    key=lambda v: (v is not None, v or 0)))]) for t in TYPES}
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
    limits = task_limits(perc_dir, sel)
    lim_counts = OrderedDict()
    for t in TYPES:
        c = defaultdict(int)
        for e in sel["entries"]:
            if e["type"] == t:
                c[str(limits.get(Path(e["file"]).stem))] += 1
        lim_counts[t] = dict(sorted(c.items()))
    doc = OrderedDict([
        ("schema", SCHEMA + ":analysis"), ("label", LABEL), ("exploratory", True), ("utc", utc()),
        ("decision", "DECISIONS G5: numbers only, absolute per start; no between-start difference, ranking, gate or "
                     "decision rule is computed (ROUND3 section 3)"),
        ("task_set", {"name": TASK_SET, "dir": str(perc_dir), "selection_sha256": sha256_file(perc_dir / "selection.json"),
                      "counts": sel.get("counts"), "n_tasks": sel.get("n_tasks"),
                      "max_time_seconds": lim_counts}),
        ("protocol", dict(PROTOCOL)), ("paper_reference", PAPER_REFERENCE),
        ("rules", {"end_max_time": END_MAX_TIME_RULE, "p95": P95_RULE,
                   "height_time_limit": "max_time_seconds is wall-clock time from session start, checked only after a "
                                        "MOVE; it depends on decoding speed and on the concurrency on the one vLLM "
                                        "server (8 workers), so per-start height numbers carry that caveat"}),
        ("starts", OrderedDict((s, analyze_start(run_dir_of(s, root), sel, s, R, limits)) for s in starts)),
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
          "| 起点 | 类型 | 步数 | 可解析 % | 修复步 | 无法解析 | 崩溃步 | STOP+数字 | STOP 无数字 | 到步数上限 | "
          "限时结束（推断） | 其他结束 | 无日志 | 截断调用 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, a in starts.items():
        if a.get("state") == "missing":
            continue
        for t in TYPES:
            st, en, ca = a["steps"][t], a["ends"][t], a["calls"][t]
            L.append(f"| {s} | {t} | {st['steps_logged']} | {st['parse_valid_pct']} | {st['repaired_steps']} | "
                     f"{st['unparseable_steps']} | {st['invalid_move_crash_steps'] + st['action_failed_steps']} | "
                     f"{en['end_stop_number']} | {en['end_stop_no_number']} | {en['end_cap']} | "
                     f"{en['end_max_time']} | {en['end_other']} | {en['no_log']} | {ca['finish_length']} |")
    L += ["", "## 解码长度和速度（每类；均值 / p95）", "",
          "| 起点 | 类型 | 回答的调用 | completion_tokens 均值 | completion_tokens p95 | 每步秒 均值 | 每步秒 p95 | "
          "到最后一步秒 均值 | 到最后一步秒 p95 | 到最后一步秒 最大 | 限时（秒） |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, a in starts.items():
        if a.get("state") == "missing":
            continue
        for t in TYPES:
            tm = a["timing"][t]
            ct, ss, es = tm["completion_tokens"], tm["step_seconds"], tm["episode_seconds"]
            lim = "、".join("无" if v is None else str(v) for v in tm["max_time_seconds"]) or "—"
            L.append(f"| {s} | {t} | {ct['n']} | {ct['mean']} | {ct['p95']} | {ss['mean']} | {ss['p95']} | "
                     f"{es['mean']} | {es['p95']} | {es['max']} | {lim} |")
    L += ["", "## 运行", "",
          "| 起点 | 结果文件 | 协议一致 | 复算一致 | 总步数 | 可解析 % | 修复步 | 秒/步（墙钟÷总步数） | 墙钟秒 | "
          "服务器存活 | 产出作业 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, a in starts.items():
        if a.get("state") == "missing":
            L.append(f"| {s} | 缺 | — | — | — | — | — | — | — | — | — |")
            continue
        tt = a["totals"]
        job = Path(str((a.get("produced_by") or {}).get("run") or "")).name or "—"
        L.append(f"| {s} | {a['results_file']} | {'是' if a['protocol_ok'] else '否'} | "
                 f"{'是' if a['consistent'] else '否'} | {tt['steps_logged']} | {tt['parse_valid_pct']} | "
                 f"{tt['repaired_steps']} | {tt['seconds_per_step']} | {tt['wall_seconds']} | "
                 f"{tt['server_alive_at_end']} | {job} |")
    probs = [(s, p) for s, a in starts.items() for p in (a.get("protocol_problems") or []) + (a.get("consistency_problems") or [])]
    if probs:
        L += ["", "## 问题", ""] + [f"- {s}: {p}" for s, p in probs]
    L += ["", "说明：",
          "- height 题的 180 秒限时是从会话开始算的墙钟时间（引擎只在 MOVE 之后检查），它取决于解码速度和同一个 vLLM "
          "服务器上的并发（8 个 worker），所以各起点的 height 数字都带着这个前提。",
          "- 「限时结束（推断）」：最后一步是 MOVE、记录的步数没到上限、这道题有 max_time_seconds（height 都是 180 秒，"
          "angle / dis 没有），并且最后一步的时间戳减去 session_start 的时间戳不少于这个限时。日志里没有 done_reason，"
          "所以这是推断：session_start 的时间戳只精确到秒、记在会话创建之前（多算最多约 1 秒），动作的时间戳记在执行这一步"
          "之前（少算），所以贴着限时被截断的题可能留在「其他结束」里。",
          "- 「其他结束」是其余情况，例如最后一步是转向，或限时推断的条件不满足。",
          "- 每步秒是同一个 episode 里相邻两个时间戳的间隔（第一步从 session_start 算起），包含排队、解码和引擎执行；"
          "「秒/步（墙钟÷总步数）」是整个运行的墙钟时间除以总步数（8 个 worker 并行），两者不是一回事。"
          "p95 按最近秩取（第 ceil(0.95n) 小的值）。completion_tokens 只算有回答的调用。",
          "- 「修复步」是 --lenient 规则从回复里取出最后一个动作的步；它们不算可解析步。",
          "- 论文那一行是全集 100 题/类、1280x800、gpt-4o-mini 修复的数，只作参照，不与本表相减。",
          "- 三个起点都从 runs/explore/perc/<起点>/evals/ 读，不管是哪个作业产出的（见「产出作业」）。",
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
