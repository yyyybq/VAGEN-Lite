"""Run a batch of eval.launch jobs from a YAML spec, with a ledger of every attempt.

    python -m eval.batch BATCH.yaml [--dry-run] [--max-parallel N] [--wait-gpu]
    nohup python -m eval.batch BATCH.yaml > batch.out 2>&1 &      # long batches; poll batch_status.json

Spec (configs/batches/*.yaml; relative config paths are relative to the repo root):

    batch: step3_turning_on_radio          # name; run ids are <batch>-<job id>-a<attempt>
    allow_report: true                     # required when a job uses Report instances (public_test 0-9)
    mode: public_test                      # defaults for every job entry ...
    wrapper: rgbd                          #   (or wrappers: [default, rgbd])
    seeds: [0]
    instances: [0]
    group_instances: false                 # false: one evaluator run per instance (num_envs 1); true: one run, all
    retries_on_infra_error: 2
    ports: {executor: 8000, agent: 8100, stride: 10}   # parallel slot k uses base + k * stride
    max_parallel: 1
    jobs_per_gpu: 1                        # >1 stacks this batch's OWN jobs on a GPU it holds (opt-in, Step 4)
    launch: {write_video: true}            # extra eval.launch flags (key -> --key, false -> --no-key), e.g.
                                           #   env_setup / evaluator_cmd_prefix / activation to reuse the helpers
                                           #   in /nas/baiqiao/b1k_runs/ (see eval/launch.py); eval_output_dir
                                           #   gets one subdirectory per attempt (<dir>/<run id>)
    overrides: []                          # -o key=value for every job
    batch_dir: ...                         # ledger + status; default <run.log_root>/_batches/<batch>
    jobs:
      - name: L0                           # job id prefix (default: the config file stem)
        config: configs/L0_exec_only.yaml  # or configs: [...]
        task: turning_on_radio             # or tasks: [...]
        wrapper: default                   # any default above can be set per entry (either form, e.g. wrappers:
                                           #   [...] replaces a default wrapper), plus direct / no_executor / note

Expansion is a cross product per entry (configs x tasks x wrappers x seeds x instance groups) with
deterministic job ids "<name>-<task>-<pt|ht|tr><indices>-<wrapper>-s<seed>". Each job is labelled with its
split: report (public_test 0-9), dev (public_test 10-19, train), hidden, mixed. A spec with report or mixed
jobs needs ``allow_report: true`` (Report instances: frozen configs only, docs/PROTOCOL.md).

Retry policy (CLAUDE.md "Experiment discipline"): a job is attempted until one attempt is "done" (success OR
policy failure, never re-run); an "infra_error" attempt is re-run at most ``retries_on_infra_error`` times,
except when every official result JSON exists and the cause cannot have tainted them (non-zero evaluator
exit or timeout, supervisor error, launcher or batch killed; not a server death or lost connection):
re-running would choose between official outcomes, so batch_status.json says so and a person decides. A
"failed" attempt (launch-level failure, user stop) is not re-run in the same invocation; a later invocation
re-runs it only if its evaluator never started or wrote no official result. Every attempt has its own run id
and run directory and two ledger lines (attempt_start, attempt_end) in <batch_dir>/jobs.jsonl; re-running
the same spec resumes from the ledger (one batch per batch_dir at a time: <batch_dir>/batch.lock; an
attempt left running on another host is resumed there). <batch_dir>/batch_status.json is rewritten after
every event. Jobs run sequentially by default; with --max-parallel N up to N jobs run at once, each on its
own free GPU(s) (eval/gpu.py; checked by the batch, then passed as --gpus) and its own port slot. Without
--wait-gpu the batch stops starting jobs when no GPU is free (one 30 s re-check right after one of its own
attempts ended).

Status: written and unit-tested with a mocked job runner (tests/test_eval_run_batch.py); nothing has been
run on 105 yet.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Optional, Sequence

import yaml

from agent.config import REPO_ROOT, get, load_config
from eval import gpu as gpumod
from eval import launch

MODE_SHORT = {"public_test": "pt", "hidden_test": "ht", "train": "tr"}
DEFAULT_PORTS = {"executor": 8000, "agent": 8100, "stride": 10}
TOP_KEYS = {"batch", "description", "mode", "seed", "seeds", "wrapper", "wrappers", "instances", "group_instances",
            "retries_on_infra_error", "ports", "max_parallel", "jobs_per_gpu", "launch", "overrides", "batch_dir",
            "jobs", "config", "configs", "task", "tasks", "direct", "no_executor", "note", "allow_report"}
ENTRY_KEYS = {"name", "config", "configs", "task", "tasks", "instances", "group_instances", "seed", "seeds",
              "wrapper", "wrappers", "mode", "direct", "no_executor", "overrides", "launch", "note"}
INHERITED = ENTRY_KEYS - {"name", "overrides", "launch", "note"}
ONE_OR_MANY = (("config", "configs"), ("task", "tasks"), ("wrapper", "wrappers"), ("seed", "seeds"))
RESERVED_LAUNCH_KEYS = {"config", "override", "task", "mode", "instances", "wrapper", "seed", "run_id", "run_dir",
                        "executor_port", "agent_port", "gpus", "skip_gpu_check", "direct", "no_executor", "dry_run",
                        "detach", "tag"}


class SpecError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Job:
    job_id: str
    config: str
    task: str
    mode: str
    instances: tuple
    wrapper: str
    seed: int
    split: str
    direct: bool = False
    no_executor: bool = False
    overrides: tuple = ()
    launch: tuple = ()  # ((key, value), ...) extra eval.launch flags
    note: str = ""


def split_of(mode: str, indices: Sequence[int]) -> str:
    """docs/PROTOCOL.md: Report = public_test 0-9, Dev = public_test 10-19 and train instances."""
    if mode == "train":
        return "dev"
    if mode == "hidden_test":
        return "hidden"
    if all(0 <= i <= 9 for i in indices):
        return "report"
    if all(10 <= i <= 19 for i in indices):
        return "dev"
    return "mixed"


def _indices_label(indices: Sequence[int]) -> str:
    idx = list(indices)
    if len(idx) > 3 and idx == list(range(idx[0], idx[0] + len(idx))):
        return f"{idx[0]}to{idx[-1]}"
    return "_".join(map(str, idx))


def _one_or_many(entry: dict, one: str, many: str, default: Optional[list] = None) -> list:
    if one in entry and many in entry:
        raise SpecError(f"give either {one!r} or {many!r}, not both: {entry}")
    if many in entry:
        if not isinstance(entry[many], list) or not entry[many]:
            raise SpecError(f"{many!r} must be a non-empty list")
        return list(entry[many])
    if one in entry:
        return [entry[one]]
    if default is None:
        raise SpecError(f"missing {one!r} / {many!r} in {entry}")
    return list(default)


def resolve_config_path(path: str) -> str:
    path = os.path.expanduser(str(path))
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path))


def launch_flags(options: dict) -> list[str]:
    """{"write_video": True, "max_steps": 500, "strict_exit": False} -> ["--write-video", "--max-steps", "500",
    "--no-strict-exit"]; None, and False for a flag without a --no- form, are dropped."""
    negatable = {s for a in launch.build_parser()._actions for s in a.option_strings if s.startswith("--no-")}
    out = []
    for key, value in options.items():
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            out.append(flag)
        elif value is False:
            if "--no-" + flag[2:] in negatable:
                out.append("--no-" + flag[2:])
        elif value is None:
            continue
        elif isinstance(value, (list, tuple)):
            out += [flag, *map(str, value)]
        else:
            out += [flag, str(value)]
    return out


def load_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    unknown = set(spec) - TOP_KEYS
    if unknown:
        raise SpecError(f"unknown top-level keys {sorted(unknown)}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(spec.get("batch", ""))):
        raise SpecError("'batch' (name: letters, digits, _.-) is required")
    if not spec.get("jobs"):
        raise SpecError("'jobs' must list at least one entry")
    return spec


def expand_jobs(spec: dict) -> list[Job]:
    jobs: list[Job] = []
    for n, raw in enumerate(spec["jobs"]):
        unknown = set(raw) - ENTRY_KEYS
        if unknown:
            raise SpecError(f"jobs[{n}]: unknown keys {sorted(unknown)}")
        inherited = {k: v for k, v in spec.items() if k in INHERITED}
        for one, many in ONE_OR_MANY:  # an entry's wrapper overrides a default wrappers list, and so on
            if one in raw or many in raw:
                inherited.pop(one, None)
                inherited.pop(many, None)
        entry = {**inherited, **raw}
        configs = [resolve_config_path(c) for c in _one_or_many(entry, "config", "configs")]
        tasks = _one_or_many(entry, "task", "tasks")
        wrappers = _one_or_many(entry, "wrapper", "wrappers", ["rgbd"])
        seeds = [int(s) for s in _one_or_many(entry, "seed", "seeds", [0])]
        mode = entry.get("mode", "public_test")
        instances = [int(i) for i in entry.get("instances") or []]
        if mode not in MODE_SHORT or not instances or any(w not in launch.WRAPPERS for w in wrappers):
            raise SpecError(f"jobs[{n}]: need mode in {sorted(MODE_SHORT)}, instances, "
                            f"wrapper in {sorted(launch.WRAPPERS)}")
        try:
            launch.resolve_instance_ids(instances, mode)
        except ValueError as e:
            raise SpecError(f"jobs[{n}]: {e}") from e
        groups = [tuple(instances)] if entry.get("group_instances") else [(i,) for i in instances]
        options = {**(spec.get("launch") or {}), **(raw.get("launch") or {})}
        reserved = set(options) & RESERVED_LAUNCH_KEYS
        if reserved:
            raise SpecError(f"jobs[{n}]: launch keys {sorted(reserved)} are set by eval.batch itself")
        overrides = tuple(spec.get("overrides") or []) + tuple(raw.get("overrides") or [])
        for config in configs:
            if not os.path.isfile(config):
                raise SpecError(f"jobs[{n}]: config {config} not found")
            stem = os.path.splitext(os.path.basename(config))[0]
            prefix = raw.get("name") or stem
            if raw.get("name") and len(configs) > 1:
                prefix = f"{raw['name']}.{stem}"
            for task in tasks:
                for wrapper in wrappers:
                    for seed in seeds:
                        for grp in groups:
                            jobs.append(Job(
                                job_id=f"{prefix}-{task}-{MODE_SHORT[mode]}{_indices_label(grp)}-{wrapper}-s{seed}",
                                config=config, task=str(task), mode=mode, instances=grp, wrapper=wrapper, seed=seed,
                                split=split_of(mode, grp), direct=bool(entry.get("direct")),
                                no_executor=bool(entry.get("no_executor")), overrides=overrides,
                                launch=tuple(options.items()), note=str(raw.get("note") or "")))
    ids = collections.Counter(j.job_id for j in jobs)
    dup = sorted(i for i, c in ids.items() if c > 1)
    if dup:
        raise SpecError(f"duplicate job ids {dup}: give the entries distinct names")
    report = [j.job_id for j in jobs if j.split in ("report", "mixed")]
    if report and spec.get("allow_report") is not True:
        raise SpecError(f"jobs {report[:3]}{'...' if len(report) > 3 else ''} use Report instances (public_test 0-9): "
                        "frozen configs only, set 'allow_report: true' to confirm (docs/PROTOCOL.md)")
    for job in jobs:
        try:
            launch.parse_launch_args(launch_argv(job, run_id="validate", run_dir="/tmp/validate"))
        except (SystemExit, launch.PlanError) as e:
            raise SpecError(f"job {job.job_id}: invalid eval.launch options {dict(job.launch)} ({e})") from e
    return jobs


def attempt_run_id(batch: str, job_id: str, attempt: int) -> str:
    return f"{batch}-{job_id}-a{attempt}"


def slot_ports(ports: dict, slot: int) -> dict:
    stride = int(ports.get("stride", DEFAULT_PORTS["stride"]))
    return {"executor": int(ports.get("executor", DEFAULT_PORTS["executor"])) + slot * stride,
            "agent": int(ports.get("agent", DEFAULT_PORTS["agent"])) + slot * stride}


def launch_argv(job: Job, *, run_id: str, run_dir: str, ports: Optional[dict] = None, gpus: Optional[Sequence] = None,
                tags: Optional[dict] = None) -> list[str]:
    argv = ["--config", job.config]
    for ov in job.overrides:
        argv += ["-o", ov]
    argv += ["--task", job.task, "--mode", job.mode, "--instances", *map(str, job.instances), "--wrapper", job.wrapper,
             "--seed", str(job.seed), "--run-id", run_id, "--run-dir", run_dir]
    if ports:
        argv += ["--executor-port", str(ports["executor"]), "--agent-port", str(ports["agent"])]
    if gpus:
        argv += ["--gpus", ",".join(map(str, gpus)), "--skip-gpu-check"]  # checked and held by the batch
    if job.direct:
        argv.append("--direct")
    if job.no_executor:
        argv.append("--no-executor")
    for k, v in (tags or {}).items():
        argv += ["--tag", f"{k}={v}"]
    options = dict(job.launch)
    if options.get("eval_output_dir"):  # a shared directory would mix jobs' identically named official JSONs
        options["eval_output_dir"] = os.path.join(str(options["eval_output_dir"]), run_id)
    return argv + launch_flags(options)


# Infra causes that do not taint official results written before them (a server death or a lost connection does).
UNTAINTED_KINDS = frozenset({"exit_code", "timeout", "supervisor_error", "launcher_crash", "batch_interrupted"})


def rerun_blocked(end: dict) -> Optional[str]:
    """Why an infra_error attempt must not be re-run: every official result exists and no cause taints them
    (re-running would choose between official outcomes)."""
    kinds = set(end.get("infra_kinds") or ())
    if end.get("state") == "infra_error" and kinds and kinds <= UNTAINTED_KINDS and end.get("results_complete"):
        return f"not re-run: every official result exists, cause {sorted(kinds)} (check by hand)"
    return None


def official_results(st: dict) -> dict:
    """results / results_complete of a run whose launcher never finalized, read from its official JSON files."""
    json_dir = os.path.join(str(st.get("eval_output_dir") or ""), "json")
    results = [launch.read_result(os.path.join(json_dir, f)) for f in st.get("expected_results") or []]
    return {"results": results, "results_complete": bool(results) and all(r["present"] for r in results)}


def next_attempt_allowed(attempts: Sequence[dict], max_retries: int = 2, *, resume: bool = False) -> bool:
    """The retry policy in the module docstring, on this job's attempt_end records (oldest first)."""
    if not attempts:
        return True
    last = attempts[-1]
    if last.get("state") == "infra_error":
        if rerun_blocked(last):
            return False
        return sum(a.get("state") == "infra_error" for a in attempts) <= max_retries
    if last.get("state") == "failed":
        no_result = not any(r.get("present") for r in last.get("results") or [])
        return resume and (not last.get("evaluator_started", True) or no_result)
    return False


class Ledger:
    """Append-only jobs.jsonl: one attempt_start and one attempt_end line per attempt."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def append(self, rec: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def summarize_status(st: dict) -> dict:
    """The parts of a launch status.json the ledger keeps (official results are read after the episodes)."""
    return {
        "state": st.get("state"), "reason": st.get("reason") or [], "warnings": (st.get("warnings") or [])[-5:],
        "evaluator_started": bool(st.get("evaluator_started", "evaluator" in (st.get("processes") or {}))),
        "evaluator_exit_code": st.get("evaluator_exit_code"), "max_steps": st.get("max_steps"),
        "config_hash": st.get("config_hash"), "infra_kinds": list(st.get("infra_kinds") or []),
        "results_complete": bool(st.get("results_complete")),
        "results": [{k: r.get(k) for k in ("file", "present", "success", "q_score", "steps", "failure_reason")}
                    for r in st.get("results") or []],
    }


class SubprocessRunner:
    """Runs one attempt as `python -m eval.launch ...` (foreground supervisor) and returns its status.json."""

    def __init__(self, out_dir: str, python: str = sys.executable):
        self.out_dir = out_dir
        self.python = python
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def __call__(self, job: Job, argv: Sequence[str], run_dir: str, run_id: str) -> dict:
        os.makedirs(self.out_dir, exist_ok=True)
        out = os.path.join(self.out_dir, f"{run_id}.out")
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with open(out, "ab") as f:
            p = subprocess.Popen([self.python, "-m", "eval.launch", *argv], cwd=REPO_ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        with self._lock:
            self._procs[run_id] = p
        rc = p.wait()
        with self._lock:
            self._procs.pop(run_id, None)
        st = launch.read_status(run_dir)
        if st is None:
            with open(out, encoding="utf-8", errors="replace") as f:
                tail = [line.strip() for line in f if line.strip()][-1:]
            return {"state": "failed", "evaluator_started": False,
                    "reason": [f"eval.launch exited {rc} without status.json: {tail[0] if tail else ''} (see {out})"]}
        if st.get("state") not in launch.TERMINAL:
            launch.cmd_stop([run_dir])  # the launcher died: stop the process groups it recorded (ours only)
            return {**st, **official_results(st), "state": "infra_error", "infra_kinds": ["launcher_crash"],
                    "reason": [f"eval.launch exited {rc} while the run was {st.get('state')}"]}
        return st

    def terminate_all(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
        for p in procs:  # our own children; each one stops only the process groups it started
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)


class BatchRunner:
    def __init__(self, spec: dict, jobs: Sequence[Job], *, batch_dir: str, runner: Callable[..., dict],
                 run_dir_for: Callable[[Job, str], str], gpus_needed: Callable[[Job], int],
                 gpu_picker: Callable[..., list] = gpumod.pick_free_gpus, max_parallel: int = 1,
                 wait_gpu: bool = False, poll_s: float = 60.0, settle_s: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, spec_path: Optional[str] = None,
                 log: Callable[[str], None] = lambda m: print(m, flush=True)):
        self.spec, self.jobs, self.batch = spec, list(jobs), str(spec["batch"])
        self.batch_dir = batch_dir
        self.runner, self.run_dir_for, self.gpus_needed, self.gpu_picker = runner, run_dir_for, gpus_needed, gpu_picker
        self.max_parallel = max(1, int(max_parallel))
        self.max_retries = int(spec.get("retries_on_infra_error", 2))
        self.jobs_per_gpu = max(1, int(spec.get("jobs_per_gpu", 1)))
        self.ports = {**DEFAULT_PORTS, **(spec.get("ports") or {})}
        self.wait_gpu, self.poll_s, self.settle_s, self.sleep, self.log = wait_gpu, poll_s, settle_s, sleep, log
        self.spec_path = spec_path
        self.stop_event = threading.Event()
        os.makedirs(batch_dir, exist_ok=True)
        self.ledger = Ledger(os.path.join(batch_dir, "jobs.jsonl"))
        self.history: dict[str, list[dict]] = collections.defaultdict(list)
        self.running: dict[str, dict] = {}
        self.gpu_use: collections.Counter = collections.Counter()
        self.started_at = launch.now_iso()
        self.abort_reason: Optional[str] = None

    # ------------------------------------------------------------------ ledger and status

    def _load_history(self) -> None:
        """Resume: attempt_end records per job; an attempt_start without an end is resolved from its status.json."""
        ends, open_starts = {}, {}
        for rec in self.ledger.read():
            key = (rec.get("job_id"), rec.get("attempt"))
            if rec.get("event") == "attempt_start":
                open_starts[key] = rec
            elif rec.get("event") == "attempt_end":
                ends[key] = rec
                open_starts.pop(key, None)
        for key, start in sorted(open_starts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            st = launch.read_status(start["run_dir"]) or {}
            if st.get("state") in launch.TERMINAL:
                rec = {**self._end_record(start, st), "recovered": True}
            else:
                sup = st.get("supervisor") or {}
                if launch.other_host(st):
                    raise SpecError(f"attempt {start['run_id']} is {st.get('state')} on host {launch.other_host(st)}; "
                                    "resume the batch there")
                if sup and launch.process_alive(sup.get("pid"), sup.get("start_ticks")):
                    raise SpecError(f"attempt {start['run_id']} is still running (supervisor pid {sup['pid']}); "
                                    "wait or stop it")
                if st:
                    launch.cmd_stop([start["run_dir"]])  # orphans of a killed attempt: its own recorded groups only
                lost = {**official_results(st), "state": "infra_error", "infra_kinds": ["batch_interrupted"],
                        "evaluator_started": "evaluator" in (st.get("processes") or {}),
                        "reason": ["batch interrupted during this attempt"]}
                rec = {**self._end_record(start, lost), "recovered": True}
            self.ledger.append(rec)
            ends[key] = rec
        for (job_id, _), rec in sorted(ends.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            self.history[job_id].append(rec)

    def _end_record(self, start: dict, status: dict) -> dict:
        keep = ("batch", "job_id", "attempt", "run_id", "run_dir", "split", "slot", "gpus", "ports")
        return {**{k: start.get(k) for k in keep}, "event": "attempt_end", "t": launch.now_iso(),
                **summarize_status(status)}

    def _job_state(self, job: Job) -> str:
        if job.job_id in self.running:
            return "running"
        hist = self.history.get(job.job_id) or []
        return hist[-1]["state"] if hist else "pending"

    def write_status(self, state: str) -> dict:
        jobs = []
        for job in self.jobs:
            hist = self.history.get(job.job_id) or []
            last = hist[-1] if hist else {}
            rerun = None
            if last.get("state") == "infra_error" and job.job_id not in self.running:
                rerun = rerun_blocked(last) or (None if next_attempt_allowed(hist, self.max_retries) else
                                                f"not re-run: {self.max_retries} infra re-runs used")
            jobs.append({"job_id": job.job_id, "split": job.split, "note": job.note, "state": self._job_state(job),
                         "attempts": len(hist) + (job.job_id in self.running), "run_ids": [h["run_id"] for h in hist],
                         "results": last.get("results") or [], "reason": last.get("reason") or [], "rerun": rerun})
        spec_sha = None
        if self.spec_path and os.path.isfile(self.spec_path):
            with open(self.spec_path, "rb") as f:
                spec_sha = hashlib.sha256(f.read()).hexdigest()
        status = {"batch": self.batch, "state": state, "abort_reason": self.abort_reason, "spec": self.spec_path,
                  "spec_sha256": spec_sha, "batch_dir": self.batch_dir, "started_at": self.started_at,
                  "updated_at": launch.now_iso(), "max_parallel": self.max_parallel, "max_retries": self.max_retries,
                  "counts": dict(collections.Counter(j["state"] for j in jobs)), "jobs": jobs}
        launch.write_json(os.path.join(self.batch_dir, "batch_status.json"), status)
        return status

    # ------------------------------------------------------------------ scheduling

    def _acquire_gpus(self, job: Job) -> list:
        need = self.gpus_needed(job)
        if need == 0:
            return []
        shared = sorted((g for g, n in self.gpu_use.items() if 0 < n < self.jobs_per_gpu),
                        key=lambda g: (self.gpu_use[g], g))
        busy = [g for g, n in self.gpu_use.items() if n > 0]
        fresh = self.gpu_picker(need - min(need, len(shared)), exclude=busy) if need > len(shared) else []
        return (shared[:need] + list(fresh))[:need]

    def _attempt(self, job: Job, attempt: int, slot: int, gpus: list) -> dict:
        run_id = attempt_run_id(self.batch, job.job_id, attempt)
        run_dir = self.run_dir_for(job, run_id)
        ports = slot_ports(self.ports, slot)
        tags = {"batch": self.batch, "job": job.job_id, "split": job.split, "attempt": attempt}
        argv = launch_argv(job, run_id=run_id, run_dir=run_dir, ports=ports, gpus=gpus, tags=tags)
        start = {"event": "attempt_start", "t": launch.now_iso(), "batch": self.batch, "job_id": job.job_id,
                 "attempt": attempt, "run_id": run_id, "run_dir": run_dir, "split": job.split, "note": job.note,
                 "slot": slot, "gpus": gpus, "ports": ports, "command": "python -m eval.launch " + shlex.join(argv)}
        self.ledger.append(start)
        self.log(f"[batch {self.batch}] start {run_id} (slot {slot}, GPUs {gpus})")
        try:
            status = self.runner(job, argv, run_dir, run_id)
        except Exception as e:
            status = {"state": "infra_error", "infra_kinds": ["batch_runner_error"],
                      "reason": [f"batch runner error: {type(e).__name__}: {e}"]}
        end = self._end_record(start, status)
        self.ledger.append(end)
        self.log(f"[batch {self.batch}] {run_id}: {end['state']} {'; '.join(end['reason'])}")
        return end

    def run(self) -> int:
        """Holds <batch_dir>/batch.lock: a second eval.batch on the same ledger would start the same attempts."""
        with open(os.path.join(self.batch_dir, "batch.lock"), "a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise SpecError(f"another eval.batch holds {lock.name}; poll its batch_status.json") from e
            except OSError as e:  # e.g. a NAS mount without lock support
                self.log(f"[batch {self.batch}] warning: cannot lock {lock.name} ({e}); "
                         "run one batch per spec at a time")
            return self._run()

    def _run(self) -> int:
        self._load_history()
        queue = collections.deque(j for j in self.jobs
                                  if next_attempt_allowed(self.history.get(j.job_id, []), self.max_retries, resume=True))
        skipped = len(self.jobs) - len(queue)
        if skipped:
            self.log(f"[batch {self.batch}] {skipped} job(s) already finished in the ledger")
        self.write_status("running")
        free_slots = list(range(self.max_parallel))
        futures: dict = {}
        released, settled = False, False  # GPUs freed by our own finished attempt may need a moment to read as free
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while (queue and self.abort_reason is None and not self.stop_event.is_set()) or futures:
                while queue and free_slots and self.abort_reason is None and not self.stop_event.is_set():
                    job = queue[0]
                    try:
                        gpus = self._acquire_gpus(job)
                    except gpumod.GpuError as e:
                        if futures:
                            break
                        if self.wait_gpu:
                            self.log(f"[batch {self.batch}] waiting for GPUs: {e}")
                            self.sleep(self.poll_s)
                            continue
                        if released and not settled:
                            settled = True
                            self.log(f"[batch {self.batch}] no free GPU right after an attempt ended; "
                                     f"one re-check in {self.settle_s:.0f} s")
                            self.sleep(self.settle_s)
                            continue
                        self.abort_reason = f"no free GPU for {job.job_id}: {e}"
                        self.log(f"[batch {self.batch}] {self.abort_reason}")
                        break
                    queue.popleft()
                    settled = False
                    slot = free_slots.pop(0)
                    self.gpu_use.update(gpus)
                    attempt = len(self.history.get(job.job_id, []))
                    self.running[job.job_id] = {"slot": slot, "gpus": gpus}
                    futures[pool.submit(self._attempt, job, attempt, slot, gpus)] = job
                    self.write_status("running")
                if not futures:
                    continue
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    job = futures.pop(fut)
                    info = self.running.pop(job.job_id)
                    free_slots.append(info["slot"])
                    free_slots.sort()
                    self.gpu_use.subtract(info["gpus"])
                    released = True
                    self.history[job.job_id].append(fut.result())
                    if not self.stop_event.is_set() and next_attempt_allowed(self.history[job.job_id], self.max_retries):
                        queue.appendleft(job)
                    self.write_status("running")
        final = "stopped" if self.stop_event.is_set() else "aborted" if self.abort_reason else "finished"
        status = self.write_status(final)
        states = [j["state"] for j in status["jobs"]]
        self.log(f"[batch {self.batch}] {final}: {dict(collections.Counter(states))}")
        if all(s == "done" for s in states):
            return 0
        return 2 if "infra_error" in states and not ({"failed", "pending"} & set(states)) else 1


# ---------------------------------------------------------------------- CLI


def _job_cfg(job: Job, cache: dict) -> dict:
    key = (job.config, job.overrides)
    if key not in cache:
        cache[key] = load_config(job.config, list(job.overrides))
    return cache[key]


def default_batch_dir(spec: dict, jobs: Sequence[Job], cache: dict) -> str:
    if spec.get("batch_dir"):
        return os.path.abspath(os.path.expanduser(spec["batch_dir"]))
    cfg = _job_cfg(jobs[0], cache)
    root = get(cfg, "run.log_root") or get(cfg, "run.local_log_root", "~/b1k_local_logs")
    return os.path.join(os.path.expanduser(str(root)), "_batches", str(spec["batch"]))


def dry_run(spec: dict, jobs: Sequence[Job], *, max_parallel: int, cache: dict) -> int:
    ports, ok = {**DEFAULT_PORTS, **(spec.get("ports") or {})}, True
    print(f"batch {spec['batch']}: {len(jobs)} job(s), max_parallel {max_parallel}, "
          f"retries_on_infra_error {spec.get('retries_on_infra_error', 2)}")
    for n, job in enumerate(jobs):
        run_id = attempt_run_id(str(spec["batch"]), job.job_id, 0)
        run_dir = launch.default_run_dir(_job_cfg(job, cache), run_id)
        argv = launch_argv(job, run_id=run_id, run_dir=run_dir, ports=slot_ports(ports, n % max_parallel),
                           tags={"batch": spec["batch"], "job": job.job_id, "split": job.split, "attempt": 0})
        print(f"\n## {job.job_id}  [{job.split}]  {job.note}\npython -m eval.launch {shlex.join(argv)}")
        try:
            print(launch.describe_plan(launch.build_plan(launch.parse_launch_args(argv + ["--dry-run"]))))
        except (launch.PlanError, OSError, ValueError) as e:
            ok = False
            print(f"ERROR: {e}")
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.batch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--dry-run", action="store_true", help="expand and print every job's launch plan; start nothing")
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--wait-gpu", action="store_true", help="wait for free GPUs instead of stopping the batch")
    args = ap.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        jobs = expand_jobs(spec)
    except (SpecError, OSError, yaml.YAMLError) as e:
        print(f"eval.batch: error: {e}", file=sys.stderr)
        return 1
    cache: dict = {}
    max_parallel = int(args.max_parallel or spec.get("max_parallel", 1))
    if args.dry_run:
        return dry_run(spec, jobs, max_parallel=max_parallel, cache=cache)

    def gpus_needed(job: Job) -> int:
        cfg = _job_cfg(job, cache)
        share = bool(dict(job.launch).get("share_gpu"))
        return launch.gpus_needed(launch.will_start_executor(cfg, job.direct, job.no_executor), share)

    batch_dir = default_batch_dir(spec, jobs, cache)
    runner = SubprocessRunner(os.path.join(batch_dir, "launch_out"))
    opts = spec.get("launch") or {}
    limits = {"max_mem_mib": opts.get("max_gpu_mem_mib", gpumod.MAX_MEM_MIB),
              "max_util": opts.get("max_gpu_util", gpumod.MAX_UTIL)}
    batch = BatchRunner(
        spec, jobs, batch_dir=batch_dir, runner=runner, spec_path=os.path.abspath(args.spec), max_parallel=max_parallel,
        run_dir_for=lambda job, run_id: launch.default_run_dir(_job_cfg(job, cache), run_id), gpus_needed=gpus_needed,
        gpu_picker=lambda n, exclude: gpumod.pick_free_gpus(n, exclude=exclude, **limits), wait_gpu=args.wait_gpu,
    )

    def on_signal(*_):
        batch.stop_event.set()
        runner.terminate_all()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        if signal.getsignal(sig) is not signal.SIG_IGN:  # under nohup SIGHUP stays ignored
            signal.signal(sig, on_signal)
    print(f"batch {spec['batch']}: {len(jobs)} job(s); ledger {batch.ledger.path}", flush=True)
    try:
        return batch.run()
    except SpecError as e:
        print(f"eval.batch: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
