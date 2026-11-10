"""EXPLORATORY: finish the scoring of a WalkerBench perception run whose episodes all ran but whose scoring crashed
(DECISIONS G5: perception numbers only, absolute per start; nothing here compares starts).

    PY_EVAL -m vpd.walkerbench.perc_finish --start Q3S|E2 [--dry-run] [--overwrite-finish]

Why.  The sco job vpd-wb-perc (RUN = jobs/wb-perc-202611101501, code snapshot of 56142675) ran perc294 for Q3B, Q3S
and E2 through vpd.walkerbench.perc eval -> run_vpd.main.  For Q3S and E2 every episode ran, then run_vpd.finalize ->
score -> the stock WalkerBench scripts/evaluate_all.py (the sandbox copy R/<S>/evals/<run>/WalkerBench) raised in
calculate_trajectory_length -> haversine: "TypeError: must be real number, not NoneType".  Some logged states have lat /
lng null (the agent moved onto a panorama without metadata in the local cache; available_moves []), and the stock
function appends such a point and feeds None to haversine.  So those run dirs hold the episodes (logs, calls.jsonl,
manifest.json, tasks.json, explore.json rc 1) but no results.json / per_task.jsonl.

What this does, without the model, vLLM or a GPU, and without editing any repo file or the sandbox copy:
1. refuses unless (a) the run dir has no results.json / results_invalid.json / per_task.jsonl / finish.json (or
   --overwrite-finish and those are this module's own earlier output), (b) tasks.json lists the 294 perc294 tasks
   (104 / 107 / 83), equal to the copy's task dir, (c) every task has exactly one session log, (d) explore.json is the
   job's record of this start (rc 1, run_vpd argv == perc.protocol_argv, the job's run_vpd.py has the sha256 of the
   run_vpd.py this process imports), (e) the job's worker log RUN/job/worker.log holds the documented traceback for
   this run dir (finalize -> score -> evaluate_session -> calculate_trajectory_length -> haversine -> that TypeError),
   and (f) at least one log has a state with lat or lng null (which logs and how many events are recorded);
2. rebuilds the meta run_vpd.main would have passed to finalize (run_vpd.RESULT_META_KEYS) from manifest.json (models,
   backend, prompt, decoding, subset, extra.vpd), explore.json (run_vpd argv, the task set dir), the copy's task dir
   (max_step_cap) and run_vpd.resolve_model; checks it against the argv and perc.protocol_want; the in-process
   run_vpd.COUNTS are rebuilt from calls.jsonl, lenient_repairs.jsonl, prompt_dump.jsonl and the logs (images_sent and
   the other image counters are not recoverable and are left out, as listed in finished_offline);
3. calls run_vpd.finalize with the stock evaluator, except that run_vpd.load_module is wrapped for the module
   "wb_evaluate_all" so that its calculate_trajectory_length drops the points whose lat or lng is None before the
   haversine sum (PATCH below).  The sandbox file is executed unchanged; only the loaded module object's attribute is
   replaced, and run_vpd.load_module is restored afterwards.  This changes only the per-task 'length' field: in
   evaluate_all.evaluate_session (b197f51) height / dis / angle success is answer-based (the last STOP's answer via
   extract_number against ground_truth height_meters / distance_between_pois_m / bearing_a_to_b_deg); 'length' enters
   only the SPL of nav / vis, which perc294 does not contain;
4. finalize writes results.json (results_invalid.json when not run_vpd.is_valid) and per_task.jsonl as usual; meta
   carries finished_offline {by, utc, why, patch, job run, what the job did not record, ...}.  Then
   R/<S>/evals/<run>/finish.json records {schema, exploratory, utc, why, patch, logs_with_null_coords, n_null_events,
   code sha256 of perc_finish.py and run_vpd.py, the job RUN, ...}.  explore.json (the job's record) is not rewritten.

Not recorded by the job, so not in results.json as measured values: wall_seconds (None; finished_offline holds a
log-based estimate), server.ready_seconds (None), server_alive_at_end (inferred: the last call in calls.jsonl was
answered and vllm.log ends with vLLM's orderly shutdown, which VLLMServer.__exit__ triggers after main's srv.alive()
check; otherwise False and finalize writes results_invalid.json).
The image-token audit (run_vpd.image_token_audit, as in the job: on) runs on CPU: it loads the start's image processor
and, for the prompt-parity check, its processor + tokenizer files (no weights) and runs them on 8 logged JPEGs and the
3 dumped prompts; torch.set_num_threads(OMP_NUM_THREADS = 1), CUDA_VISIBLE_DEVICES="" is set before torch is imported.
The job's run_vpd (and so its audit) ran under PY_Q3 (vpd/jobs/wb_perc.sh, transformers 4.57.6).  Under PY_EVAL
(transformers 4.56.1) AutoProcessor does not build the Qwen3-VL processor, so image_tokens.prompt_parity holds an error
(tokens_per_image is unaffected); run the CLI with PY_Q3 for the job's complete audit.  finish.json records the
interpreter and versions.
Loading the sandbox evaluator imports the copy's engine package, whose CacheManager opens WalkerBench/data/cache.db in
WAL mode at import: cache.db-wal / -shm exist while the process runs (finish.json sandbox_side_effects lists what was
seen right after finalize).  One start per process (refused when another copy's engine is already imported).
--dry-run runs every check and prints the rebuilt meta and the plan; it writes nothing and does not load the sandbox
evaluator.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

from vpd.walkerbench import perc as P

SCHEMA = "vpd.walkerbench.perc_finish/1"
MODULE = "vpd.walkerbench.perc_finish"
LABEL = "EXPLORATORY"
FINISH_STARTS = ("Q3S", "E2")  # the two starts whose scoring crashed in vpd-wb-perc (Q3B has results.json)
EV_MODULE = "wb_evaluate_all"  # the name run_vpd.score loads the sandbox's scripts/evaluate_all.py under
FINISH_JSON = "finish.json"
OUTPUTS = ("results.json", "results_invalid.json", "per_task.jsonl", FINISH_JSON)
CRASH_ERROR = "TypeError: must be real number, not NoneType"
SHUTDOWN_MARKER = "Application shutdown complete"  # vLLM's orderly shutdown (after VLLMServer.__exit__'s terminate)
WHY = ("the job's run_vpd.finalize -> score -> stock evaluate_all.calculate_trajectory_length -> haversine raised "
       f"'{CRASH_ERROR}' on logged states with lat/lng null (panorama without metadata in the local cache, "
       "available_moves []); every episode had run, so the run is scored offline from its logs without the model")
PATCH = ("run_vpd.load_module wrapped for module 'wb_evaluate_all' (the run dir's sandbox scripts/evaluate_all.py, "
         "executed unchanged): its calculate_trajectory_length is replaced on the loaded module object by a wrapper "
         "that, for a log with a session_start initial_state or action state whose lat or lng is None, removes those "
         "states' lat/lng keys and calls the stock function, which then skips them (the points are dropped before the "
         "haversine sum); logs without such a state call the stock function on the unchanged events.  Affects only "
         "the per-task 'length' field; height / dis / angle success is answer-based (evaluate_session: the last STOP's "
         "answer against ground_truth) and length enters only nav / vis SPL")
NOT_RECORDED = ("wall_seconds", "seconds_per_step", "server.ready_seconds", "counts.images_sent",
                "counts.images_missing_url", "counts.images_missing_file", "counts.images_withheld",
                "counts.repair_skipped")


class Refused(SystemExit):
    pass


def refuse(problems) -> None:
    probs = [problems] if isinstance(problems, str) else list(problems)
    raise Refused("[perc_finish] refused: " + "; ".join(probs))


def _R():
    from vpd.walkerbench import run_vpd

    return run_vpd


# ------------------------------------------------------------------------------------------------- patch
def null_coord_state(e: dict):
    """The state the stock calculate_trajectory_length reads from one event, when it holds lat and lng keys and one
    of them is None (such a state becomes a None point there); else None."""
    if e.get("event") == "session_start":
        st = e.get("initial_state", {})
    elif e.get("event") == "action":
        st = e.get("state", {})
    else:
        return None
    if isinstance(st, dict) and "lat" in st and "lng" in st and (st["lat"] is None or st["lng"] is None):
        return st
    return None


def drop_null_coords(events: list) -> list:
    """The events with lat/lng removed from every state null_coord_state() finds (copies; the input is unchanged)."""
    out = []
    for e in events:
        st = null_coord_state(e)
        if st is None:
            out.append(e)
            continue
        key = "initial_state" if e.get("event") == "session_start" else "state"
        out.append({**e, key: {k: v for k, v in st.items() if k not in ("lat", "lng")}})
    return out


def wrap_trajectory_length(orig, stats: dict | None = None):
    """calculate_trajectory_length that drops the None points before the stock haversine sum (PATCH)."""
    stats = stats if stats is not None else {}

    def calculate_trajectory_length(events):
        n = sum(1 for e in events if null_coord_state(e) is not None)
        stats["calls"] = stats.get("calls", 0) + 1
        if not n:
            return orig(events)
        stats["calls_with_null"] = stats.get("calls_with_null", 0) + 1
        stats["null_events"] = stats.get("null_events", 0) + n
        return orig(drop_null_coords(events))

    calculate_trajectory_length.__wrapped__ = orig
    calculate_trajectory_length.vpd_perc_finish = True
    return calculate_trajectory_length


@contextlib.contextmanager
def patched_evaluator(R, stats: dict):
    """While active, run_vpd.load_module(EV_MODULE, path) returns the stock module with the wrapped function."""
    orig_load = R.load_module

    def load_module(name, path):
        mod = orig_load(name, path)
        if name == EV_MODULE:
            fn = getattr(mod, "calculate_trajectory_length", None)
            if fn is None:
                raise RuntimeError(f"{path}: no calculate_trajectory_length (not the stock evaluate_all.py)")
            if not getattr(fn, "vpd_perc_finish", False):
                mod.calculate_trajectory_length = wrap_trajectory_length(fn, stats)
            stats.setdefault("patched", []).append(str(path))
        return mod

    R.load_module = load_module
    try:
        yield
    finally:
        R.load_module = orig_load


# ------------------------------------------------------------------------------------------------- facts
def _read_json(p: Path, what: str, probs: list):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        probs.append(f"{what} {p} unreadable ({type(e).__name__})")
        return None


def _read_jsonl(p: Path) -> list:
    if not Path(p).exists():
        return []
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def own_output(run_dir: Path) -> tuple[list, list]:
    """(existing output files, the ones that are not this module's own earlier output)."""
    have = [n for n in OUTPUTS if (run_dir / n).exists()]
    ours, foreign = [], []
    for n in have:
        doc = P.read_json(run_dir / n) if n != "per_task.jsonl" else None
        if n in ("results.json", "results_invalid.json"):
            mine = isinstance(doc, dict) and (doc.get("finished_offline") or {}).get("by") == MODULE
        elif n == FINISH_JSON:
            mine = isinstance(doc, dict) and doc.get("schema") == SCHEMA
        else:
            continue
        (ours if mine else foreign).append(n)
    if "per_task.jsonl" in have and (foreign or not ours):
        foreign.append("per_task.jsonl")  # written by the same finalize as a result file that is not ours
    return have, foreign


def scan_logs(logs_dir: Path, R) -> dict:
    """Per log: task id (session_start), null-coordinate events, step stats; plus the logs' time span."""
    out = {"logs": OrderedDict(), "by_task": defaultdict(list), "null": OrderedDict(), "n_null_events": 0,
           "first_start": None, "last_action": None, "crash_tags": defaultdict(int)}
    for log in sorted(Path(logs_dir).glob("*.jsonl")):
        events = R.read_events(log)
        start = next((e for e in events if e.get("event") == "session_start"), None)
        tid = (start or {}).get("task_id")
        out["by_task"][tid].append(log.name)
        n = sum(1 for e in events if null_coord_state(e) is not None)
        if n:
            out["null"][log.name] = n
            out["n_null_events"] += n
        st = R.step_stats(events)
        out["crash_tags"]["invalid_move_crash"] += st["invalid_move_crash_steps"]
        out["crash_tags"]["action_failed"] += st["action_failed_steps"]
        out["logs"][log.name] = tid
        t0 = P.parse_start_ts((start or {}).get("timestamp"))
        if t0 is not None and (out["first_start"] is None or t0 < out["first_start"]):
            out["first_start"] = t0
        for e in events:
            if e.get("event") == "action":
                t = P.parse_iso_ts(e.get("timestamp"))
                if t is not None and (out["last_action"] is None or t > out["last_action"]):
                    out["last_action"] = t
    return out


def crash_evidence(worker_log: Path, run_dir: Path, start: str) -> dict | None:
    """The documented traceback for this run dir in the job's worker log, or None.  Needs, within 40 lines: run_vpd.py
    'in finalize' and 'in score', the run dir's WalkerBench/scripts/evaluate_all.py 'in evaluate_session', 'in
    calculate_trajectory_length' and 'in haversine', then CRASH_ERROR; and later '[vpd_step] eval_<start> end ... rc=1'."""
    try:
        lines = Path(worker_log).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    evs = {f'File "{d}/WalkerBench/scripts/evaluate_all.py"' for d in (str(run_dir), os.path.realpath(run_dir))}
    for i, line in enumerate(lines):
        if not (any(ev in line for ev in evs) and "in calculate_trajectory_length" in line):
            continue
        before, after = lines[max(0, i - 40):i], lines[i + 1:i + 12]
        ok_before = (any("run_vpd.py" in ln and "in finalize" in ln for ln in before)
                     and any("run_vpd.py" in ln and "in score" in ln for ln in before)
                     and any(any(ev in ln for ev in evs) and "in evaluate_session" in ln for ln in before))
        hav = next((j for j, ln in enumerate(after) if any(ev in ln for ev in evs) and "in haversine" in ln), None)
        err = next((j for j, ln in enumerate(after) if ln.strip().endswith(CRASH_ERROR)), None)
        if not (ok_before and hav is not None and err is not None and err > hav):
            continue
        step = next((k for k in range(i, len(lines)) if lines[k].startswith(f"[vpd_step] eval_{start} end")), None)
        if step is None or not lines[step].rstrip().endswith("rc=1"):
            continue
        return {"worker_log": str(worker_log), "traceback_line": i + 1, "error_line": i + 2 + err,
                "error": CRASH_ERROR, "step_line": step + 1, "step": lines[step].strip()}
    return None


def _argv_values(argv: list) -> dict:
    vals = {}
    for i, a in enumerate(argv):
        if a.startswith("--"):
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            vals[a] = True if nxt is None or nxt.startswith("--") else nxt
    return vals


def server_alive_evidence(run_dir: Path, calls: list) -> dict:
    last = calls[-1] if calls else None
    try:
        tail = Path(run_dir / "vllm.log").read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
    except OSError:
        tail = []
    answered = last is not None and last.get("error") is None
    shutdown = any(SHUTDOWN_MARKER in ln for ln in tail)
    return {"value": bool(answered and shutdown),
            "rule": ("inferred (the job did not record srv.alive()): True when the last call in calls.jsonl was "
                     f"answered and the last 50 lines of vllm.log contain '{SHUTDOWN_MARKER}' (vLLM's orderly shutdown "
                     "after VLLMServer.__exit__'s terminate, which follows main's srv.alive() check)"),
            "last_call_answered": answered, "vllm_log_orderly_shutdown": shutdown}


def rebuild_counts(run_dir: Path, calls: list, scan: dict, tasks: list) -> dict:
    """run_vpd.COUNTS as the job's process would have held them (keys only when incremented, as defaultdict += does),
    without the image counters (NOT_RECORDED)."""
    c = OrderedDict()

    def put(k, v, always=False):
        if v or always:
            c[k] = v

    put("model_calls", len(calls))
    put("model_call_errors", sum(1 for x in calls if x.get("error") is not None))
    constrained = [x for x in calls if x.get("error") is None and x.get("constrained")]
    put("constrained_calls", len(constrained))
    if constrained:
        put("regex_mismatch", sum(1 for x in constrained if not x.get("regex_ok")), always=True)
    put("finish_length", sum(1 for x in calls if x.get("finish_reason") == "length"))
    put("prompt_dumped", len(_read_jsonl(run_dir / "prompt_dump.jsonl")))
    rep = _read_jsonl(run_dir / "lenient_repairs.jsonl")
    put("lenient_repairs", sum(1 for r in rep if r.get("repaired")))
    put("lenient_no_action", sum(1 for r in rep if not r.get("repaired")))
    put("invalid_move_crash", scan["crash_tags"]["invalid_move_crash"])
    put("action_failed", scan["crash_tags"]["action_failed"])
    put("tasks_without_log", sum(1 for t in tasks if not scan["by_task"].get(t)), always=True)
    return dict(c)


def gather(start: str, *, root=P.ROOT, expected=P.EXPECTED, overwrite_finish: bool = False, R=None) -> dict:
    """Every refusal check (raises Refused listing all problems) and everything finalize needs."""
    R = R or _R()
    root = Path(root)
    P.refuse_pilot(root)
    run_dir = P.run_dir_of(start, root)
    probs = []
    if not run_dir.is_dir():
        refuse(f"{run_dir} is not a directory")
    have, foreign = own_output(run_dir)
    if have and not overwrite_finish:
        probs.append(f"{run_dir} already has {have} (pass --overwrite-finish to replace this module's own output)")
    elif foreign:
        probs.append(f"{run_dir}: {foreign} are not this module's output; not replaced")
    copy = run_dir / "WalkerBench"
    for f in ("manifest.json", "explore.json", "tasks.json", "calls.jsonl", "vllm.log"):
        if not (run_dir / f).exists():
            probs.append(f"{run_dir / f} missing")
    if not (copy / "scripts" / "evaluate_all.py").exists():
        probs.append(f"{copy}/scripts/evaluate_all.py missing")
    if probs:
        refuse(probs)
    manifest = _read_json(run_dir / "manifest.json", "manifest", probs)
    ex = _read_json(run_dir / "explore.json", "explore", probs)
    tasks = _read_json(run_dir / "tasks.json", "tasks", probs)
    if probs:
        refuse(probs)
    # explore.json: the job's record of this start
    want_argv = P.protocol_argv(start, root)
    if ex.get("start") != start:
        probs.append(f"explore.json start {ex.get('start')!r} != {start!r}")
    if ex.get("rc") != 1:
        probs.append(f"explore.json rc {ex.get('rc')!r} != 1 (the documented crash)")
    if ex.get("run_vpd_argv") != want_argv:
        probs.append("explore.json run_vpd_argv differs from perc.protocol_argv")
    rv_sha = P.sha256_file(R.__file__)
    job_rv = (ex.get("code") or {}).get("vpd/walkerbench/run_vpd.py")
    if job_rv != rv_sha:
        probs.append(f"the job's run_vpd.py sha256 {str(job_rv)[:12]} != this process's {rv_sha[:12]} ({R.__file__})")
    run = ex.get("run")
    if not (isinstance(run, str) and os.path.isabs(run)):
        probs.append(f"explore.json run {run!r} is not an absolute path")
        run = None
    # tasks: the 294 perc294 tasks, one log each
    tasks_src = (ex.get("task_sets_added") or {}).get(P.TASK_SET)
    label = R.sandbox.task_label(Path(tasks_src)) if tasks_src else None
    tasks_dir = copy / "tasks" / str(label)
    want_n = sum(expected.values())
    if not isinstance(tasks, list) or len(tasks) != want_n or len(set(tasks)) != len(tasks):
        probs.append(f"tasks.json: {len(tasks) if isinstance(tasks, list) else type(tasks).__name__} entries, "
                     f"expected {want_n} distinct tasks")
        tasks = tasks if isinstance(tasks, list) else []
    else:
        by_type = {t: sum(1 for s in tasks if P.task_type(s) == t) for t in expected}
        if by_type != dict(expected):
            probs.append(f"tasks.json per type {by_type} != {dict(expected)}")
    if label is None or not tasks_dir.is_dir():
        probs.append(f"task dir {tasks_dir} missing (explore.json task_sets_added {tasks_src!r})")
    elif sorted(p.stem for p in tasks_dir.glob("*.json")) != sorted(tasks):
        probs.append(f"tasks.json != the stems of {tasks_dir}")
    elif R.select_tasks([p.stem for p in tasks_dir.glob("*.json")], None) != tasks:
        probs.append("tasks.json is not run_vpd.select_tasks(task dir) (order)")
    scan = scan_logs(copy / "logs", R)
    counts = {t: len(scan["by_task"].get(t, [])) for t in tasks}
    not_one = {t: n for t, n in counts.items() if n != 1}
    extra = sorted(str(t) for t in scan["by_task"] if t not in set(tasks))
    if not_one:
        probs.append(f"{len(not_one)} tasks without exactly one session log (e.g. {sorted(not_one.items())[:3]})")
    if extra:
        probs.append(f"logs of tasks not in tasks.json: {extra[:3]}")
    if not scan["null"]:
        probs.append("no log has a state with lat/lng null: the crash is not the documented one")
    # the worker log's traceback
    crash = None
    if run is not None:
        crash = crash_evidence(Path(run) / "job" / "worker.log", run_dir, start)
        if crash is None:
            probs.append(f"{Path(run) / 'job' / 'worker.log'}: no '{CRASH_ERROR}' traceback through finalize -> score "
                         f"-> {run_dir}/WalkerBench/scripts/evaluate_all.py calculate_trajectory_length -> haversine "
                         f"followed by '[vpd_step] eval_{start} end ... rc=1'")
    if probs:
        refuse(probs)
    calls = R.read_calls(run_dir / "calls.jsonl")
    meta = rebuild_meta(run_dir, start, manifest, ex, tasks_dir, R, probs)
    if probs:
        refuse(probs)
    alive = server_alive_evidence(run_dir, calls)
    wall_est = None
    if scan["first_start"] is not None and scan["last_action"] is not None:
        wall_est = round((scan["last_action"] - scan["first_start"]).total_seconds(), 1)
    meta["counts"] = rebuild_counts(run_dir, calls, scan, tasks)
    return {"start": start, "root": root, "run_dir": run_dir, "copy": copy, "tasks_dir": tasks_dir, "tasks": tasks,
            "explore": ex, "manifest": manifest, "run": run, "crash": crash, "scan": scan, "meta": meta,
            "server_alive": alive, "wall_estimate": wall_est, "existing": have,
            "audit_processor": R.audit_dir_for({"start": meta["start"], "code": meta["code"],
                                                "served": meta["served_model_name"]}, None)}


def rebuild_meta(run_dir: Path, start: str, manifest: dict, ex: dict, tasks_dir: Path, R, probs: list) -> dict:
    """run_vpd.main's finalize meta (RESULT_META_KEYS, main's order) from the job's records; problems appended."""
    try:
        model = (manifest.get("models") or [None])[0] or {}
        backend, prompt = manifest["backend"], manifest["prompt"]
        decoding, subset, vpd = manifest["decoding"], manifest["subset"], manifest["extra"]["vpd"]
        server = dict(backend["server"])
        cons = prompt["constrained"]
    except (KeyError, TypeError, IndexError, AttributeError) as e:
        probs.append(f"manifest.json lacks a field run_vpd writes ({type(e).__name__}: {e})")
        return {}
    argv = ex.get("run_vpd_argv") or []
    a = _argv_values(argv)
    info = R.resolve_model(a.get("--model"))
    if (info["start"], info["code"], info["served"]) != (model.get("start"), model.get("code"),
                                                          model.get("served_model_name")):
        probs.append(f"run_vpd.resolve_model({a.get('--model')!r}) {info} != manifest models[0]")
    server["ready_seconds"] = None  # main sets it after the manifest is written; the job did not record it
    meta = OrderedDict([
        ("model", a.get("--model")), ("start", model.get("start")), ("code", model.get("code")),
        ("served_model_name", model.get("served_model_name")), ("model_dir", vpd.get("model_dir")),
        ("adapter", vpd.get("adapter")), ("run_name", vpd.get("run_name")), ("task_set", vpd.get("task_set")),
        ("tasks_label", tasks_dir.name), ("render_size", list(backend.get("render_size") or [])),
        ("constrained", cons.get("mode")), ("thought_max_chars", cons.get("thought_max_chars")),
        ("history_turns", (backend.get("history") or {}).get("turns")), ("max_tokens", decoding.get("max_tokens")),
        ("max_images", (server.get("cfg") or {}).get("max_images")),
        ("blind", str(prompt.get("images", "")).startswith("withheld")),
        ("lenient", str(prompt.get("json_repair", "")).startswith("lenient")),
        ("temperature", decoding.get("temperature")), ("limit", subset.get("limit")),
        ("max_step_cap", R.max_step_cap(tasks_dir)), ("server", server)])
    if set(meta) != set(R.RESULT_META_KEYS):
        probs.append(f"rebuilt meta keys != run_vpd.RESULT_META_KEYS ({sorted(set(meta) ^ set(R.RESULT_META_KEYS))})")
    # the argv the job passed to run_vpd.main
    try:
        size = list(R.sandbox.parse_size(a.get("--render-size")))
    except Exception:
        size = None
    from_argv = {"model": start, "task_set": a.get("--task-set"), "render_size": size,
                 "constrained": a.get("--constrained"), "lenient": a.get("--lenient") is True,
                 "max_tokens": int(a.get("--max-tokens", -1)), "history_turns": int(a.get("--history-turns", -1)),
                 "temperature": float(a.get("--temperature", "nan")), "run_name": a.get("--run-name"),
                 "blind": "--blind" in a, "limit": None if "--limit" not in a else int(a["--limit"])}
    if "--model-dir" not in a and "--adapter-meta" not in a:  # the perc protocol serves the start's own view
        from_argv.update(model_dir=None, adapter=None)
    else:
        probs.append("run_vpd argv has --model-dir / --adapter-meta (not the perc protocol)")
    for k, v in from_argv.items():
        if meta.get(k) != v:
            probs.append(f"meta {k} {meta.get(k)!r} != run_vpd argv {v!r}")
    if backend.get("workers") != int(a.get("--workers", -1)):
        probs.append(f"manifest workers {backend.get('workers')!r} != argv --workers {a.get('--workers')!r}")
    if meta["constrained"] != "short" and meta["thought_max_chars"] != 0:
        probs.append(f"thought_max_chars {meta['thought_max_chars']!r} != 0 for constrained {meta['constrained']!r}")
    probs.extend(f"protocol: {p}" for p in P.protocol_problems(dict(meta), manifest, start))
    if run_dir.name != meta["run_name"]:
        probs.append(f"run dir name {run_dir.name} != run_name {meta['run_name']!r}")
    return meta


# ------------------------------------------------------------------------------------------------- finish
def tree_state(d: Path) -> dict:
    """relative path -> (size, mtime_ns) of every file under d (side-effect record of the sandbox)."""
    out = {}
    for dirpath, _, files in os.walk(d):
        for f in files:
            p = Path(dirpath) / f
            try:
                st = p.lstat()
            except OSError:
                continue
            out[str(p.relative_to(d))] = (st.st_size, st.st_mtime_ns)
    return out


def tree_diff(before: dict, after: dict) -> dict:
    return {"added": sorted(set(after) - set(before)), "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k])}


def _version(dist: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def _write_json(p: Path, obj) -> None:
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def finished_offline(g: dict, utc: str) -> dict:
    return OrderedDict([
        ("by", MODULE), ("label", LABEL), ("exploratory", True), ("utc", utc), ("finish_json", FINISH_JSON),
        ("why", WHY), ("patch", PATCH), ("job_run", g["run"]), ("explore_rc", g["explore"].get("rc")),
        ("logs_with_null_coords", len(g["scan"]["null"])), ("n_null_events", g["scan"]["n_null_events"]),
        ("not_recorded_by_the_job", list(NOT_RECORDED)),
        ("wall_estimate_seconds_from_logs", g["wall_estimate"]),
        ("wall_estimate_rule", "latest action timestamp minus earliest session_start timestamp over the run's logs "
                               "(descriptive; wall_seconds stays None)"),
        ("server_alive_at_end", g["server_alive"]),
        ("counts_rebuilt_from", "calls.jsonl, lenient_repairs.jsonl, prompt_dump.jsonl and the task logs")])


def plan(g: dict) -> dict:
    return {"start": g["start"], "run_dir": str(g["run_dir"]), "job_run": g["run"], "crash": g["crash"],
            "logs_with_null_coords": list(g["scan"]["null"]), "n_null_events": g["scan"]["n_null_events"],
            "n_tasks": len(g["tasks"]), "n_logs": len(g["scan"]["logs"]),
            "meta": {k: v for k, v in g["meta"].items() if k != "server"},
            "server_alive_at_end": g["server_alive"], "wall_estimate_seconds_from_logs": g["wall_estimate"],
            "audit_processor": str(g["audit_processor"]) if g["audit_processor"] else None,
            "existing_outputs": g["existing"],
            "would_write": [str(g["run_dir"] / n) for n in ("per_task.jsonl", "results.json or results_invalid.json",
                                                             FINISH_JSON)]}


def finish(start: str, *, root=P.ROOT, expected=P.EXPECTED, dry_run: bool = False, overwrite_finish: bool = False,
           R=None, audit: bool = True) -> dict:
    """Check, rebuild the meta, and (unless dry_run) score with the patched stock evaluator via run_vpd.finalize."""
    if start not in P.STARTS:
        raise ValueError(f"start must be one of {P.STARTS}")
    R = R or _R()
    g = gather(start, root=root, expected=expected, overwrite_finish=overwrite_finish, R=R)
    if dry_run:
        return {"dry_run": True, **plan(g)}
    utc = P.utc()
    meta = dict(g["meta"])
    meta["finished_offline"] = finished_offline(g, utc)
    run_dir, copy = g["run_dir"], g["copy"]
    eng = getattr(sys.modules.get("engine"), "__file__", None)
    if eng and not P.is_under(eng, copy):  # evaluate_all imports the copy's engine package; sys.modules would win
        raise RuntimeError(f"module 'engine' is already loaded from {eng}: finish one start per process")
    explore_sha = P.sha256_file(run_dir / "explore.json")
    ev_path = copy / "scripts" / "evaluate_all.py"
    ev_sha = P.sha256_file(ev_path)
    before = tree_state(copy)
    stats: dict = {}
    saved_counts = dict(R.COUNTS)
    saved_bytecode = sys.dont_write_bytecode
    R.COUNTS.clear()  # finalize takes dict(COUNTS) when non-empty; this process ran no episode
    sys.dont_write_bytecode = True
    try:
        with patched_evaluator(R, stats):
            res, valid = R.finalize(run_dir, copy, g["tasks_dir"], g["tasks"], meta=meta,
                                    server_alive=g["server_alive"]["value"], wall=None,
                                    audit_processor=g["audit_processor"] if audit else None)
    finally:
        R.COUNTS.clear()
        R.COUNTS.update(saved_counts)
        sys.dont_write_bytecode = saved_bytecode
    after = tree_state(copy)
    if P.sha256_file(ev_path) != ev_sha or P.sha256_file(run_dir / "explore.json") != explore_sha:
        raise RuntimeError("the sandbox evaluate_all.py or explore.json changed during the finish")
    if len(stats.get("patched", [])) != 1 or stats.get("null_events", 0) != g["scan"]["n_null_events"]:
        raise RuntimeError(f"patch not applied as expected: {stats} vs {g['scan']['n_null_events']} null events")
    rows = _read_jsonl(run_dir / "per_task.jsonl")
    null_tasks = {g["scan"]["logs"][n] for n in g["scan"]["null"]}
    doc = OrderedDict([
        ("schema", SCHEMA), ("label", LABEL), ("exploratory", True), ("utc", utc), ("start", start),
        ("run_dir", str(run_dir)), ("why", WHY), ("patch", PATCH),
        ("success_rule", "evaluate_all.evaluate_session (b197f51): height / dis / angle success = the last STOP's "
                         "answer (extract_number) within tolerance of ground_truth (15 % height, 20 % distance, 30 deg "
                         "angle); independent of 'length', which enters only nav / vis SPL"),
        ("logs_with_null_coords", list(g["scan"]["null"])), ("null_events_per_log", dict(g["scan"]["null"])),
        ("n_null_events", g["scan"]["n_null_events"]),
        ("patch_stats", {"evaluator": stats.get("patched"), "calls": stats.get("calls"),
                         "calls_with_null": stats.get("calls_with_null"), "null_events": stats.get("null_events")}),
        ("lengths_of_null_logs", {r["stem"]: (res.get("per_task") or {}).get(r["stem"], {}).get("length")
                                  for r in rows if r["stem"] in null_tasks}),
        ("results_file", "results.json" if valid else "results_invalid.json"), ("valid", valid),
        ("server_alive_at_end", g["server_alive"]), ("wall_estimate_seconds_from_logs", g["wall_estimate"]),
        ("not_recorded_by_the_job", list(NOT_RECORDED)), ("counts_rebuilt", g["meta"]["counts"]),
        ("image_tokens_processor", str(g["audit_processor"]) if (audit and g["audit_processor"]) else None),
        ("job", {"run": g["run"], "explore_utc": g["explore"].get("utc"), "explore_rc": g["explore"].get("rc"),
                 "snapshot_commit": (g["explore"].get("code_snapshot") or {}).get("commit"),
                 "explore_json_sha256": explore_sha, "crash": g["crash"]}),
        ("evaluator", {"path": str(ev_path), "sha256": ev_sha, "module": EV_MODULE}),
        ("sandbox_side_effects", tree_diff(before, after)),
        ("code_sha256", {"vpd/walkerbench/perc_finish.py": P.sha256_file(__file__),
                         "vpd/walkerbench/run_vpd.py": P.sha256_file(R.__file__)}),
        ("replaced_own_output", g["existing"]),
        ("python", {"executable": sys.executable, "transformers": _version("transformers"),
                    "torch": _version("torch")}),
    ])
    _write_json(run_dir / FINISH_JSON, doc)
    return {"dry_run": False, "results_file": doc["results_file"], "valid": valid, "finish_json": str(run_dir / FINISH_JSON),
            "per_kind": res.get("per_kind"), "steps_logged": res.get("steps_logged"),
            "tasks_without_log": res.get("tasks_without_log"), "n_null_events": doc["n_null_events"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.walkerbench.perc_finish", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, choices=FINISH_STARTS)
    ap.add_argument("--dry-run", action="store_true", help="run every check, print the plan, write nothing")
    ap.add_argument("--overwrite-finish", action="store_true", help="replace this module's own earlier output")
    a = ap.parse_args(argv)
    P.thread_env(1)  # login node: one thread
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the audit's torch stays on CPU
    sys.dont_write_bytecode = True
    try:
        out = finish(a.start, dry_run=a.dry_run, overwrite_finish=a.overwrite_finish)
    except Refused as e:
        print(e.code, file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, default=str))
    return 0 if (out.get("dry_run") or out.get("valid")) else 1


if __name__ == "__main__":
    sys.exit(main())
