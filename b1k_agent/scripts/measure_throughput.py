"""Step-4 throughput from finished eval/launch.py run directories, and run-time estimates computed from it.

    python scripts/measure_throughput.py RUN_DIR [RUN_DIR ...] [--gpu-log FILE] [--local-root ~/b1k_local_logs]
        [--parallel-per-gpu 1] [--gpus 1] [--ladder-configs 6] [--dev-tasks T ...] [--episodes N]
        [--task-catalog configs/tasks/task_catalog_2026.tsv] [--out docs/throughput_<date>.md] [--json FILE]

Inputs per run (read-only). Nothing is invented: a number without a measured input prints "unmeasured".
  status.json (eval/launch.py)     evaluator start / end (ISO, 1 s resolution), executor and agent start -> healthy
                                   (model and server load), instance ids, GPUs
  logs/evaluator.log               the official log lines (eval/launch.py parse_evaluator_log) and their relative
                                   time stamps "[HH:MM:SS.sss]" (OmniGibson/omnigibson/__init__.py LogFormatter: time
                                   since the evaluator process set up logging) of "Instance X (env i) finished at step S"
  eval_output/json/*.json          official steps; on 2026/eval also time.evaluation_time / policy_time / sim_time
  episodes/*/summary.json          agent steps and elapsed_s (budget clock: end of the uncharged first query -> last
                                   observation, agent/state.py), matched to official results by eval.aggregate
  episodes/*/events.jsonl          episode_start wall time (time.time()), executor infer_ms (agent round trip) and
                                   server_timing.infer_ms, provider latency_s
  <local>/<run_id>/episodes/*/steps.jsonl
                                   per-step receive times. Their "t" is time.monotonic() of the agent process, so only
                                   differences inside one file are used (episode span, step interval), never compared
                                   with wall-clock times
  gpu_samples.csv, gpu_apps.csv    scripts/gpu_sampler.sh (in the run dir, or --gpu-log): peak memory per GPU / process
Derived:
  episode_s     steps.jsonl span, else summary elapsed_s, else official time.evaluation_time (the source is recorded)
  steps_per_s   steps / episode_s per episode; the estimates use the pooled rate (all steps / all episode_s), not the
                mean of per-episode rates, which would let short fast episodes understate the time of long ones
  scene_load_s  evaluator start (status.json) -> the agent's first episode_start (wall clock): evaluator start-up,
                scene load and reset before the first query. Needs an agent run (a --direct run has no agent clock).
Estimates, hours = sum over episodes (scene_load_s + steps / steps_per_s) / (parallel_per_gpu x gpus):
  as measured     every episode as long as the measured mean episode_s
  timeout bound   every episode runs to its task's max_steps (--task-catalog; the official int(1.5 x human length)).
                  Failing episodes do, so this is the planning number while the executor is weak.
  sets            dev = --dev-tasks (default: docs/PROTOCOL.md section 13, proposal A) x 10 instances x 3 seeds;
                  dev_ladder = dev x --ladder-configs; report = all catalog tasks (100) x 10 instances x 1;
                  report_ladder = report x --ladder-configs; custom = --episodes N measured-like episodes.
  --parallel-per-gpu and --gpus are ASSUMPTIONS (printed as such). The GPU log's memory ceiling,
  floor(memory.total / peak growth per simulator env), is shown as a hint only (CPU and render contention unmeasured).

Status: written and unit-tested on synthetic run directories (tests/test_ops_measure_throughput.py); not yet run on
real runs from 105.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import math
import os
import re
import statistics
import sys
from typing import Any, Iterable, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.aggregate import InputLog, _official_key, load_run, match_summaries
from eval.launch import parse_evaluator_log

UNMEASURED = "unmeasured"
DEFAULT_CATALOG = os.path.join(REPO_ROOT, "configs", "tasks", "task_catalog_2026.tsv")
DEFAULT_DEV_TASKS = ("turning_out_all_lights_before_sleep", "turning_on_radio", "picking_up_trash", "store_honey",
                     "putting_dishes_away_after_cleaning")  # docs/PROTOCOL.md section 13, proposal A
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_REL_TS = re.compile(r"\[(\d+):(\d{2}):(\d{2}(?:\.\d+)?)\]")
_FINISHED = re.compile(r"Instance (\d+) \(env (\d+)\) finished at step (\d+)")
_EP_ENV = re.compile(r"-env(\d+)$")


# ---------------------------------------------------------------------------------------------- small helpers


def parse_iso(s: Any) -> Optional[float]:
    if not isinstance(s, str) or not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.datetime.strptime(s, fmt).replace(tzinfo=datetime.UTC).timestamp()
        except ValueError:
            continue
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else b - a


def stats(xs: Iterable[Any]) -> Optional[dict]:
    v = sorted(float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(x))
    if not v:
        return None

    def pct(p: float) -> float:
        return v[min(len(v) - 1, round(p * (len(v) - 1)))]

    return {"n": len(v), "mean": statistics.fmean(v), "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": v[-1]}


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return UNMEASURED
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1000 else f"{x:.0f}"
    return str(x)


def fmt_stats(s: Optional[dict], nd: int = 3) -> str:
    if not s:
        return UNMEASURED
    return f"{fmt(s['mean'], nd)} / {fmt(s['p50'], nd)} / {fmt(s['p90'], nd)} / {fmt(s['max'], nd)} (n={s['n']})"


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


# ---------------------------------------------------------------------------------------------- per-file parsers


def evaluator_log_times(lines: Iterable[str]) -> dict:
    """Relative time stamps (seconds since the evaluator set up logging) of the official log lines we time."""
    out: dict = {"first_s": None, "last_s": None, "finished": []}
    for raw in lines:
        line = _ANSI.sub("", raw)
        m = _REL_TS.search(line)
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        out["first_s"] = t if out["first_s"] is None else out["first_s"]
        out["last_s"] = t
        f = _FINISHED.search(line)
        if f:
            out["finished"].append({"instance_id": int(f.group(1)), "env": int(f.group(2)), "step": int(f.group(3)),
                                    "t_s": t})
    return out


def steps_file(path: str) -> Optional[dict]:
    """Span and step intervals of one steps.jsonl (monotonic clock: differences only)."""
    ts, last_step = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec.get("t"), (int, float)):
                ts.append(float(rec["t"]))
                last_step = rec.get("step", last_step)
    if len(ts) < 2:
        return None
    dts = [(b - a) * 1000.0 for a, b in itertools.pairwise(ts)]
    return {"n": len(ts), "span_s": ts[-1] - ts[0], "last_step": last_step, "interval_ms": stats(dts)}


def events_file(path: str) -> dict:
    out: dict = {"start_wall": None, "infer_ms": [], "server_infer_ms": [], "provider_latency_s": []}
    for rec in read_jsonl(path):
        kind = rec.get("kind")
        if kind == "episode_start" and out["start_wall"] is None:
            out["start_wall"] = rec.get("t")
        elif kind == "executor":
            out["infer_ms"].append(rec.get("infer_ms"))
            timing = (rec.get("meta") or {}).get("server_timing") or {}
            out["server_infer_ms"].append(timing.get("infer_ms"))
        elif kind == "provider_call":
            out["provider_latency_s"].append(rec.get("latency_s"))
    return out


def _num(x: str) -> Optional[float]:
    try:
        return float(x.strip())
    except (ValueError, AttributeError):
        return None


def gpu_log(path: str) -> dict:
    """scripts/gpu_sampler.sh gpu_samples.csv -> per GPU index: samples, first / peak / total MiB, utilization."""
    gpus: dict = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6 or not row[1].strip().isdigit():
                continue
            i, used, total, util = int(row[1]), _num(row[3]), _num(row[4]), _num(row[5])
            if used is None:
                continue
            g = gpus.setdefault(i, {"index": i, "uuid": row[2].strip(), "samples": 0, "first_mib": used, "peak_mib": used,
                                    "total_mib": total, "util": []})
            g["samples"] += 1
            g["peak_mib"] = max(g["peak_mib"], used)
            if util is not None:
                g["util"].append(util)
    for g in gpus.values():
        u = g.pop("util")
        g["peak_delta_mib"] = g["peak_mib"] - g["first_mib"]
        g["util_mean"] = statistics.fmean(u) if u else None
        g["util_peak"] = max(u) if u else None
    return gpus


def gpu_apps_log(path: str) -> dict:
    """gpu_apps.csv -> pid: {gpu_uuid, peak_mib}."""
    procs: dict = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4 or not row[1].strip().isdigit():
                continue
            pid, mem = int(row[1]), _num(row[3])
            if mem is None:
                continue
            p = procs.setdefault(pid, {"pid": pid, "gpu_uuid": row[2].strip(), "peak_mib": mem})
            p["peak_mib"] = max(p["peak_mib"], mem)
    return procs


def load_catalog(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["task"]: int(r["max_steps"]) for r in csv.DictReader(f, delimiter="\t") if r.get("max_steps")}


# ---------------------------------------------------------------------------------------------- one run


def local_steps_path(run, eid: str, local_root: Optional[str]) -> Optional[str]:
    cands = []
    if local_root:
        cands.append(os.path.join(os.path.expanduser(local_root), run.run_id))
    ptr = (run.status.get("agent_manifest") or {}).get("local")
    if ptr:
        cands.append(os.path.dirname(os.path.expanduser(ptr)))
    cfg_root = ((run.manifest.get("config") or {}).get("run") or {}).get("local_log_root")
    if cfg_root:
        cands.append(os.path.join(os.path.expanduser(cfg_root), run.run_id))
    cands.append(run.run_dir)
    for c in cands:
        p = os.path.join(c, "episodes", eid, "steps.jsonl")
        if os.path.isfile(p):
            return p
    return None


def measure_run(run_dir: str, local_root: Optional[str] = None) -> dict:
    inputs = InputLog()
    run = load_run(run_dir, inputs)
    st = run.status
    procs = st.get("processes") or {}
    ev_start = parse_iso((procs.get("evaluator") or {}).get("started_at"))
    ev_end = parse_iso((procs.get("evaluator") or {}).get("ended_at"))
    out: dict = {
        "run_id": run.run_id, "run_dir": run.run_dir, "state": st.get("state"), "direct": st.get("direct"),
        "task": st.get("task"), "instance_ids": st.get("instance_ids"), "num_envs": st.get("num_envs"),
        "gpus": st.get("gpus"), "warnings": list(run.warnings),
        "run_wall_s": diff(parse_iso(st.get("started_at")), parse_iso(st.get("ended_at"))),
        "evaluator_wall_s": diff(ev_start, ev_end),
        "executor_load_s": diff(parse_iso((procs.get("executor") or {}).get("started_at")),
                                parse_iso((procs.get("executor") or {}).get("healthy_at"))),
        "agent_load_s": diff(parse_iso((procs.get("agent") or {}).get("started_at")),
                             parse_iso((procs.get("agent") or {}).get("healthy_at"))),
    }
    log_path = os.path.join(run.run_dir, "logs", "evaluator.log")
    log_times, log_info = {"finished": []}, {}
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            # OmniGibson's LogFormatter wraps every line in ANSI colour codes ("...q_score=0.0\x1b[0m")
            lines = [_ANSI.sub("", line) for line in f]
        log_times, log_info = evaluator_log_times(lines), parse_evaluator_log(lines)
    out["evaluator_log"] = {"max_steps": log_info.get("max_steps"), "summary": log_info.get("summary"),
                            "first_s": log_times.get("first_s"), "last_s": log_times.get("last_s")}
    finished = {f["instance_id"]: f for f in log_times["finished"]}

    keys = [(_official_key(p, rec), p, rec) for p, rec in run.official]
    matched, unmatched = match_summaries(run, [k for k, _, _ in keys])
    if unmatched:
        out["warnings"].append(f"{len(unmatched)} agent episode summaries not matched to an official result")
    episodes, starts = [], []
    agg = {"infer_ms": [], "server_infer_ms": [], "provider_latency_s": [], "interval_ms": []}
    for key, path, rec in keys:
        task, iid, rid = key
        t = rec.get("time") or {}
        row = {"task": task, "instance_id": iid, "rollout_id": rid, "steps": rec.get("steps"),
               "q_final": (rec.get("q_score") or {}).get("final"), "success": rec.get("success"),
               "official_evaluation_time_s": t.get("evaluation_time"), "official_policy_time_s": t.get("policy_time"),
               "official_sim_time_s": t.get("sim_time"),
               "evaluator_finish_s": (finished.get(iid) or {}).get("t_s"), "env": (finished.get(iid) or {}).get("env"),
               "agent_steps": None, "agent_elapsed_s": None, "steps_span_s": None, "first_query_wall": None}
        if key in matched:
            spath, summary, _how = matched[key]
            eid = summary.get("episode_id", os.path.basename(os.path.dirname(spath)))
            row.update(agent_steps=summary.get("steps"), agent_elapsed_s=summary.get("elapsed_s"), episode_id=eid)
            ev_path = os.path.join(os.path.dirname(spath), "events.jsonl")
            if os.path.isfile(ev_path):
                ev = events_file(ev_path)
                row["first_query_wall"] = ev["start_wall"]
                if isinstance(ev["start_wall"], (int, float)):
                    starts.append(float(ev["start_wall"]))
                for k in ("infer_ms", "server_infer_ms", "provider_latency_s"):
                    agg[k] += ev[k]
            sp = local_steps_path(run, eid, local_root)
            if sp:
                sf = steps_file(sp)
                if sf:
                    row["steps_span_s"] = sf["span_s"]
                    agg["interval_ms"].append(sf["interval_ms"]["mean"] if sf["interval_ms"] else None)
        if row["env"] is None and isinstance(row.get("episode_id"), str):
            m = _EP_ENV.search(row["episode_id"])
            row["env"] = int(m.group(1)) if m else None
        steps = row["steps"] if isinstance(row["steps"], (int, float)) else row["agent_steps"]
        for src in ("steps_span_s", "agent_elapsed_s", "official_evaluation_time_s"):
            v = row[src]
            if isinstance(v, (int, float)) and v > 0:
                row["episode_s"], row["episode_s_source"] = float(v), src
                break
        else:
            row["episode_s"], row["episode_s_source"] = None, None
        row["steps_per_s"] = steps / row["episode_s"] if row["episode_s"] and isinstance(steps, (int, float)) else None
        episodes.append(row)
    out["episodes"] = episodes
    out["scene_load_s"] = diff(ev_start, min(starts)) if starts and ev_start is not None else None
    out["latency"] = {"executor_infer_ms": stats(agg["infer_ms"]), "server_infer_ms": stats(agg["server_infer_ms"]),
                      "provider_latency_s": stats(agg["provider_latency_s"]),
                      "step_interval_ms_mean_per_episode": stats(agg["interval_ms"])}
    return out


# ---------------------------------------------------------------------------------------------- estimates


def measured_inputs(runs: Sequence[dict]) -> dict:
    eps = [e for r in runs for e in r["episodes"]]
    timed = [e for e in eps if e.get("steps_per_s") and e.get("episode_s")]
    total_s = sum(e["episode_s"] for e in timed)
    pooled = sum(e["steps_per_s"] * e["episode_s"] for e in timed) / total_s if timed else None
    ep_s = [e["episode_s"] for e in eps if e.get("episode_s")]
    loads = [r["scene_load_s"] for r in runs if isinstance(r.get("scene_load_s"), (int, float))]
    return {"steps_per_s": pooled, "episode_s": statistics.fmean(ep_s) if ep_s else None,
            "scene_load_s": statistics.fmean(loads) if loads else None, "episodes_measured": len(ep_s),
            "tasks_measured": sorted({e["task"] for e in eps if e.get("episode_s")})}


def workload_sets(catalog: dict, dev_tasks: Sequence[str], ladder: int, custom: Optional[int]) -> list[dict]:
    report_tasks = sorted(catalog) or None  # no catalog: the task list is unknown, never "0 tasks"
    sets = [
        {"name": "dev", "tasks": list(dev_tasks), "instances": 10, "seeds": 3, "configs": 1},
        {"name": "dev_ladder", "tasks": list(dev_tasks), "instances": 10, "seeds": 3, "configs": ladder},
        {"name": "report", "tasks": report_tasks, "instances": 10, "seeds": 1, "configs": 1},
        {"name": "report_ladder", "tasks": report_tasks, "instances": 10, "seeds": 1, "configs": ladder},
    ]
    if custom:
        sets.append({"name": "custom", "tasks": [], "instances": 0, "seeds": 1, "configs": 1, "episodes": int(custom)})
    return sets


def estimate(sets: Sequence[dict], m: dict, catalog: dict, slots: int) -> list[dict]:
    out = []
    sps, ep_s, load = m["steps_per_s"], m["episode_s"], m["scene_load_s"]
    for s in sets:
        per_task = s["instances"] * s["seeds"] * s["configs"]
        tasks = s["tasks"] or []
        n = s.get("episodes", len(tasks) * per_task if s["tasks"] is not None else None)
        row = {**s, "episodes": n, "slots": slots, "missing_max_steps": [t for t in tasks if t not in catalog]}
        row["as_measured_h"] = (n * (load + ep_s) / slots / 3600
                                if n is not None and load is not None and ep_s is not None else None)
        if sps and tasks and not row["missing_max_steps"]:
            stepping = sum(per_task * catalog[t] / sps for t in tasks)
            row["timeout_bound_stepping_only_h"] = stepping / slots / 3600
            row["timeout_bound_h"] = (stepping + n * load) / slots / 3600 if load is not None else None
        else:
            row["timeout_bound_stepping_only_h"] = row["timeout_bound_h"] = None
        out.append(row)
    return out


def memory_hint(runs: Sequence[dict], gpus: dict) -> list[dict]:
    """floor(total / peak growth per env) on the GPUs a run's evaluator / executor used (a hint, memory only)."""
    hints = []
    for r in runs:
        for role in ("evaluator", "executor"):
            for idx in (r.get("gpus") or {}).get(role) or []:
                g = gpus.get(int(idx)) if str(idx).isdigit() else None
                if not g or not g["peak_delta_mib"] or not g["total_mib"]:
                    continue
                per_env = g["peak_delta_mib"] / max(int(r.get("num_envs") or 1), 1) if role == "evaluator" else g["peak_delta_mib"]
                hints.append({"run_id": r["run_id"], "role": role, "gpu": int(idx), "peak_delta_mib": g["peak_delta_mib"],
                              "per_unit_mib": per_env, "ceiling_per_gpu": int(g["total_mib"] // per_env)})
    return hints


# ---------------------------------------------------------------------------------------------- report


def render(res: dict) -> str:
    a = res["assumptions"]
    m = res["measured"]
    lines = ["# Throughput measurement", "", f"generated {res['generated_at']} from {len(res['runs'])} run dir(s) by "
             "scripts/measure_throughput.py; every number comes from the logs listed in its docstring.", ""]
    lines += ["## Runs", "", "| run | state | task | envs | evaluator wall (s) | executor load (s) | agent load (s) | "
              "scene load (s) | max_steps (log) |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in res["runs"]:
        lines.append(f"| {r['run_id']} | {r['state']} | {r['task']} | {fmt(r['num_envs'])} | {fmt(r['evaluator_wall_s'])} | "
                     f"{fmt(r['executor_load_s'])} | {fmt(r['agent_load_s'])} | {fmt(r['scene_load_s'])} | "
                     f"{fmt(r['evaluator_log']['max_steps'])} |")
    lines += ["", "## Episodes", "", "| run | task | instance | steps | episode s (source) | steps/s | agent elapsed s | "
              "official evaluation_time s | evaluator finish s (rel.) | q |", "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in res["runs"]:
        for e in r["episodes"]:
            lines.append(f"| {r['run_id']} | {e['task']} | {e['instance_id']} | {fmt(e['steps'])} | "
                         f"{fmt(e['episode_s'])} ({e['episode_s_source'] or '-'}) | {fmt(e['steps_per_s'])} | "
                         f"{fmt(e['agent_elapsed_s'])} | {fmt(e['official_evaluation_time_s'])} | "
                         f"{fmt(e['evaluator_finish_s'])} | {fmt(e['q_final'])} |")
    lines += ["", "## Latency (mean / p50 / p90 / max)", "", "| run | executor infer ms (agent round trip) | "
              "executor server infer ms | provider latency s | step interval ms (episode means) |", "|---|---|---|---|---|"]
    for r in res["runs"]:
        lat = r["latency"]
        lines.append(f"| {r['run_id']} | {fmt_stats(lat['executor_infer_ms'])} | {fmt_stats(lat['server_infer_ms'])} | "
                     f"{fmt_stats(lat['provider_latency_s'])} | {fmt_stats(lat['step_interval_ms_mean_per_episode'])} |")
    lines += ["", "## GPU memory (gpu_sampler.sh)", ""]
    if res["gpus"]:
        lines += ["| GPU | samples | first MiB | peak MiB | peak growth MiB | total MiB | util mean / peak % |",
                  "|---:|---:|---:|---:|---:|---:|---|"]
        for g in res["gpus"].values():
            lines.append(f"| {g['index']} | {g['samples']} | {fmt(g['first_mib'])} | {fmt(g['peak_mib'])} | "
                         f"{fmt(g['peak_delta_mib'])} | {fmt(g['total_mib'])} | {fmt(g['util_mean'])} / {fmt(g['util_peak'])} |")
        for h in res["memory_hints"]:
            lines.append(f"- hint ({h['run_id']}, {h['role']} on GPU{h['gpu']}): {fmt(h['per_unit_mib'])} MiB per "
                         f"{'simulator env' if h['role'] == 'evaluator' else 'executor server'} -> memory ceiling "
                         f"{h['ceiling_per_gpu']} per GPU (memory only; contention unmeasured)")
        for p in res["gpu_processes"]:
            lines.append(f"- pid {p['pid']} on {p['gpu_uuid']}: peak {fmt(p['peak_mib'])} MiB")
    else:
        lines.append(f"{UNMEASURED}: no gpu_samples.csv (run scripts/gpu_sampler.sh start/stop around the run)")
    lines += ["", "## Estimates", "",
              f"Measured inputs: steps/s {fmt(m['steps_per_s'])}, episode s {fmt(m['episode_s'])}, scene load s "
              f"{fmt(m['scene_load_s'])} ({m['episodes_measured']} episodes; tasks {', '.join(m['tasks_measured']) or 'none'}).",
              f"Assumptions (command line, NOT measured): {a['parallel_per_gpu']} parallel simulator+policy instance(s) per "
              f"GPU x {a['gpus']} GPU(s) = {a['slots']} slot(s); ladder = {a['ladder_configs']} configs; max_steps from "
              f"{a['task_catalog']}.", ""]
    if m["steps_per_s"] is not None and m["steps_per_s"] < 1.0:
        lines += [f"**Measured {fmt(m['steps_per_s'])} steps/s < 1: a full-length episode would exceed the 2026 budget "
                  "of max_steps x 1 s (over budget scores 0).**", ""]
    lines += ["| set | tasks | instances x seeds x configs | episodes | as measured (h) | timeout bound (h) | "
              "timeout bound, stepping only (h) |", "|---|---:|---|---:|---:|---:|---:|"]
    for e in res["estimates"]:
        missing = f" (no max_steps: {', '.join(e['missing_max_steps'])})" if e["missing_max_steps"] else ""
        n_tasks = UNMEASURED + " (no task catalog)" if e["tasks"] is None else (len(e["tasks"]) or "-")
        lines.append(f"| {e['name']} | {n_tasks} | {e['instances']} x {e['seeds']} x {e['configs']} | "
                     f"{fmt(e['episodes'])} | {fmt(e['as_measured_h'])} | {fmt(e['timeout_bound_h'])}{missing} | "
                     f"{fmt(e['timeout_bound_stepping_only_h'])} |")
    warnings = [f"{r['run_id']}: {w}" for r in res["runs"] for w in r["warnings"]]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"


def measure(run_dirs: Sequence[str], *, gpu_logs: Sequence[str] = (), local_root: Optional[str] = None,
            parallel_per_gpu: int = 1, gpus: int = 1, ladder_configs: int = 6, dev_tasks: Sequence[str] = DEFAULT_DEV_TASKS,
            episodes: Optional[int] = None, task_catalog: str = DEFAULT_CATALOG) -> dict:
    runs = [measure_run(d, local_root) for d in run_dirs]
    gpu_data, procs = {}, {}
    logs = list(gpu_logs) + [os.path.join(d, "gpu_samples.csv") for d in run_dirs]
    for p in dict.fromkeys(logs):
        if os.path.isfile(p):
            for i, g in gpu_log(p).items():
                if i not in gpu_data or g["peak_mib"] > gpu_data[i]["peak_mib"]:
                    gpu_data[i] = g
            apps = os.path.join(os.path.dirname(p), "gpu_apps.csv")
            if os.path.isfile(apps):
                procs.update(gpu_apps_log(apps))
    catalog = load_catalog(task_catalog) if task_catalog and os.path.isfile(task_catalog) else {}
    m = measured_inputs(runs)
    slots = max(int(parallel_per_gpu), 1) * max(int(gpus), 1)
    sets = workload_sets(catalog, dev_tasks, ladder_configs, episodes)
    return {
        "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": runs, "gpus": gpu_data, "gpu_processes": sorted(procs.values(), key=lambda p: -p["peak_mib"]),
        "memory_hints": memory_hint(runs, gpu_data), "measured": m,
        "assumptions": {"parallel_per_gpu": parallel_per_gpu, "gpus": gpus, "slots": slots, "ladder_configs": ladder_configs,
                        "task_catalog": task_catalog if catalog else f"{UNMEASURED} (no catalog)"},
        "estimates": estimate(sets, m, catalog, slots),
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="eval/launch.py run directories (status.json, logs/, eval_output/)")
    ap.add_argument("--gpu-log", action="append", default=[], help="gpu_samples.csv outside the run dirs")
    ap.add_argument("--local-root", default=None, help="agent run.local_log_root (for steps.jsonl)")
    ap.add_argument("--parallel-per-gpu", type=int, default=1, help="ASSUMPTION: simulator+policy instances per GPU")
    ap.add_argument("--gpus", type=int, default=1, help="ASSUMPTION: free GPUs available")
    ap.add_argument("--ladder-configs", type=int, default=6, help="configs per ladder (L0-L4 + budget-matched control)")
    ap.add_argument("--dev-tasks", nargs="+", default=list(DEFAULT_DEV_TASKS))
    ap.add_argument("--episodes", type=int, default=None, help="add a custom set of N measured-like episodes")
    ap.add_argument("--task-catalog", default=DEFAULT_CATALOG, help="TSV with task and max_steps columns")
    ap.add_argument("--out", default=None, help="markdown report")
    ap.add_argument("--json", default=None, help="JSON with every number")
    args = ap.parse_args(argv)
    res = measure(args.run_dirs, gpu_logs=args.gpu_log, local_root=args.local_root, parallel_per_gpu=args.parallel_per_gpu,
                  gpus=args.gpus, ladder_configs=args.ladder_configs, dev_tasks=args.dev_tasks, episodes=args.episodes,
                  task_catalog=args.task_catalog)
    text = render(res)
    print(text, end="")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
