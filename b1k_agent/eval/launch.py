"""Launch one run of the OFFICIAL evaluator against our servers, supervise it, and record its status.

    python -m eval.launch --config CFG [-o key=val ...] --task T --instances 0 [1 ...]
        [--mode public_test|train|hidden_test] [--wrapper rgbd|default] [--seed S] [--run-id ID]
        [--evaluator-root PATH] [--direct] [--no-executor] [--executor-port P] [--agent-port P]
        [--write-video] [--max-steps M | --agent-max-steps N] [--gpus 0,1] [--share-gpu] [--wait-gpu]
        [--dry-run] [--detach]
    python -m eval.launch status RUN_DIR [--wait] [--interval S] [--json]
    python -m eval.launch stop RUN_DIR

Processes, each in its own process group with output in <run_dir>/logs/<name>.log:
  executor   executor/openpi_server/serve_b1k_prompted.py in the openpi venv (cwd paths.openpi_root):
             --robot b1k/R1Pro --task b1k/<task> --repo-id --policy.config --policy.dir --control-mode
             --action-horizon --port --rename-robot-from robot_r1. Skipped with --no-executor (an executor
             server is already listening on --executor-port) or when the config's executor is not openpi_ws.
  agent      python -m agent.server --config CFG --port --run-id --max-steps --tag task=... (cwd: this
             repo). Skipped with --direct: the evaluator then talks straight to the executor server, the
             "L0-direct" reference for passthrough-equivalence checks.
  evaluator  python -m omnigibson.eval.eval (<run_dir>/run_evaluator.sh), started once every server's
             /healthz answers: --task-name --host --port --instance-indices --mode --num-envs --env-wrapper
             --output-dir <run_dir>/eval_output [--write-video] [--max-steps] [--robot-config]
             [--replay-action-chunk-size]. Environment: OMNIGIBSON_HEADLESS=1, OMNI_KIT_ACCEPT_EULA=YES,
             CUDA_VISIBLE_DEVICES from eval/gpu.py, OMNIGIBSON_DATA_PATH, OMNIGIBSON_APPDATA_PATH; cwd
             --evaluator-root (default paths.b1k_root); PYTHONPATH=<evaluator-root>/OmniGibson when that root
             is not the installed tree (a git worktree of branch 2026/eval).

Reusing the scripts already on 105 (/nas/baiqiao/b1k_runs/; this launcher does not duplicate them):
  * B1K_ENV_SETUP=<file> (or --env-setup) is sourced before the evaluator starts;
  * --activation conda-run (default: `conda run -p /nas/baiqiao/envs/behavior --no-capture-output`, which
    runs the env's own activate hooks, e.g. its libstdc++ preload), activate (conda shell hook + conda
    activate), or none (the caller's setup or --evaluator-cmd-prefix provides python);
  * --evaluator-cmd-prefix "<helper> [args]" is put in front of `python -m omnigibson.eval.eval`;
  * --no-executor --executor-port P uses an executor server started by an existing run helper;
  * <run_dir>/commands.sh and run_evaluator.sh hold the exact commands, to compare with a smoke test or run by hand.

Run directory (default <run.log_root>/<run_id>, the NAS summary directory that the agent's RunRecorder
also uses; per-step agent files stay on local disk): status.json, plan.json, commands.sh,
run_evaluator.sh, logs/, eval_output/{json,videos}/ (official), run_manifest.json + episodes/ (agent).

max_steps (the agent's 1 s/step budget, CLAUDE.md "Budget") is --max-steps (also passed to the evaluator:
not the official default) or --agent-max-steps (agent only), else computed like the evaluator does:
int(1.5 x task.jsonl "length") (OmniGibson/omnigibson/eval/utils/score_utils.py load_human_stats,
evaluator.py:281-286). After the run it is checked against the evaluator's "Setting timeout to be" line.

status.json "state": planned | starting | running | done | failed | infra_error.
  done         every expected official JSON exists and parses (a policy failure is NOT an infrastructure error);
  infra_error  a server we started did not come up or died while the evaluator ran, the evaluator exited
               non-zero or timed out, a result JSON is missing or unreadable, or a result says failure_reason
               policy_connection_lost (2026/eval). With --no-strict-exit a non-zero exit after ALL results is
               done. "infra_kinds" names the causes; eval.batch does not re-run an exit code / timeout after
               complete results (the official records exist; a re-run would choose between them);
  failed       launcher-level failure before the evaluator ran, a GPU placement violation, or a user stop.
Exit codes: done 0, failed 1, infra_error 2, still running (status) 3. On finish the launcher stops ONLY the
process groups it started (pid + start time + owner verified), evaluator first, agent with SIGINT so it
writes its last episode summary. --detach returns at once; a watcher process (`_watch RUN_DIR`) does the
supervision and keeps status.json current. `status` / `stop` on another host than the supervisor's (run dirs
are on the NAS) only read status.json and never signal anything: pids are per host.

GPUs (CLAUDE.md "GPU discipline"): eval/gpu.py picks boards with memory.used < 2048 MiB and utilization < 10 %
(evaluator first, executor second; --share-gpu puts both on one) and every process gets an explicit
CUDA_VISIBLE_DEVICES (the agent gets none). No GPU free: exit 1 without starting anything, unless --wait-gpu.
Every 30 s the supervisor reads nvidia-smi compute apps (read-only); a process of ours on a board it was not
given stops the run (state failed, --gpu-mismatch warn only notes it).

Status: written and unit-tested with mocks and fake processes (tests/test_eval_run_launch.py); nothing
has been run on 105 yet.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from typing import Any, Callable, Optional, Sequence

from agent.config import REPO_ROOT, config_hash, get, git_info, load_config
from eval import gpu as gpumod

EVAL_MODULE = "omnigibson.eval.eval"
WRAPPERS = {
    "default": "omnigibson.eval.wrappers.DefaultWrapper",
    "rgbd": "omnigibson.eval.wrappers.RGBDFullResWrapper",
}
MODES = ("public_test", "train", "hidden_test")
# OmniGibson/omnigibson/eval/utils/eval_utils.py and score_utils.py (v3.9.3-post2; unchanged on 2026/eval)
EVAL_TIMEOUT_MULTIPLIER = 1.5
TEST_INSTANCE_IDS = list(range(301, 341))
NUM_PUBLIC_TEST_INSTANCES = 20
HUMAN_STATS_KEYS = ("length", "distance_traveled", "left_eef_displacement", "right_eef_displacement")
METADATA_SUBDIR = os.path.join("2026-challenge-task-instances", "metadata")

DEFAULT_BEHAVIOR_ENV = "/nas/baiqiao/envs/behavior"
DEFAULT_APPDATA = "~/.omnigibson_appdata"
CONNECTION_FAILURES = ("policy_connection_lost",)  # 2026/eval evaluator.py record_fn failure_reason
TERMINAL = ("done", "failed", "infra_error")
EXIT_CODES = {"done": 0, "failed": 1, "infra_error": 2}
EXIT_RUNNING = 3


class PlanError(RuntimeError):
    pass


class StopRequested(Exception):
    pass


class GpuPlacementError(Exception):
    pass


# ---------------------------------------------------------------------- official mirrors


def resolve_instance_ids(indices: Sequence[int], mode: str) -> list[int]:
    """evaluator.py resolve_instance_ids: train -> raw ids; public_test i -> 301+i; hidden_test i -> 321+i."""
    if mode == "train":
        return [int(i) for i in indices]
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    n_public = NUM_PUBLIC_TEST_INSTANCES
    split = TEST_INSTANCE_IDS[:n_public] if mode == "public_test" else TEST_INSTANCE_IDS[n_public:]
    bad = [i for i in indices if not 0 <= int(i) < len(split)]
    if bad:
        raise ValueError(f"{mode} instance indices must be in 0..{len(split) - 1}, got {bad}")
    return [split[int(i)] for i in indices]


def load_task_ids(csv_path: str) -> dict[str, int]:
    """B100_task_misc.csv "Task" -> "Task ID" (eval_utils._load_challenge_task_misc)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["Task"]: int(row["Task ID"]) for row in csv.DictReader(f)}


def load_human_stats(task_name: str, task_jsonl: str, task_index: Optional[int] = None) -> dict:
    """score_utils.load_human_stats: the first line whose task_name matches or whose task_index equals the
    task's B100_task_misc.csv id; all four HUMAN_STATS_KEYS are required (KeyError otherwise, as there)."""
    with open(os.path.expanduser(task_jsonl), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("task_name") == task_name or (task_index is not None and rec.get("task_index") == task_index):
                return {key: rec[key] for key in HUMAN_STATS_KEYS}
    raise ValueError(f"Could not find human stats for task {task_name} in {task_jsonl}")


def official_max_steps(task_name: str, task_jsonl: str, task_index: Optional[int] = None) -> int:
    """evaluator.py:281-286 default timeout: int(mean human length x EVAL_TIMEOUT_MULTIPLIER)."""
    return int(load_human_stats(task_name, task_jsonl, task_index)["length"] * EVAL_TIMEOUT_MULTIPLIER)


def secret_env_names(cfg: dict) -> list[str]:
    """Provider key variables (providers.*.api_key_env plus the documented names): only the agent may see them."""
    names = {"ASTRA_API_KEY", "ANTHROPIC_API_KEY"}
    for prov in (get(cfg, "providers", {}) or {}).values():
        if isinstance(prov, dict) and prov.get("api_key_env"):
            names.add(str(prov["api_key_env"]))
    return sorted(names)


def detect_repo_id(checkpoint_dir: str) -> Optional[str]:
    """The asset id under <ckpt>/assets/ that holds norm_stats.json (serve_b1k --repo-id); None if absent."""
    assets = os.path.join(checkpoint_dir, "assets")
    found = sorted(
        os.path.relpath(os.path.dirname(p), assets)
        for p in glob.glob(os.path.join(assets, "**", "norm_stats.json"), recursive=True)
    )
    if len(found) > 1:
        raise PlanError(f"several norm_stats.json under {assets}: {found}; pass --executor-repo-id")
    return found[0] if found else None


# ---------------------------------------------------------------------- processes (read /proc; signal only our groups)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def proc_stat(pid: int) -> Optional[dict]:
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            data = f.read()
        uid = os.stat(f"/proc/{int(pid)}").st_uid
    except (OSError, ValueError):
        return None
    rest = data[data.rindex(")") + 2 :].split()
    return {"pid": int(pid), "state": rest[0], "pgrp": int(rest[2]), "start_ticks": int(rest[19]), "uid": uid}


def process_alive(pid: Optional[int], start_ticks: Optional[int] = None) -> bool:
    st = proc_stat(pid) if pid else None
    if st is None or st["state"] == "Z":
        return False
    return start_ticks is None or st["start_ticks"] == start_ticks


def group_members(pgid: int) -> list[dict]:
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            st = proc_stat(int(name))
            if st and st["pgrp"] == pgid and st["state"] != "Z":
                out.append(st)
    return out


def signal_own_group(pgid: int, sig: int, leader_start_ticks: Optional[int]) -> str:
    """Signal process group ``pgid`` only if every live member is ours: same uid and started no earlier than
    the leader we launched (so a recycled pid can never reach someone else's process)."""
    members = group_members(pgid)
    if not members:
        return "gone"
    foreign = [
        m["pid"] for m in members
        if m["uid"] != os.getuid() or (leader_start_ticks is not None and m["start_ticks"] < leader_start_ticks)
    ]
    if foreign:
        return f"refused: pids {foreign} are not ours"
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return "gone"
    return "sent"


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex((host, port)) == 0:
            return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # ignore TIME_WAIT left by a previous run
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False


_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def healthz_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with _DIRECT_OPENER.open(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def connect_host(host: str) -> str:
    return "127.0.0.1" if host in ("", "0.0.0.0", "::") else host


def no_proxy_env(environ: dict) -> dict:
    """When a proxy is configured, local health checks and websockets (requests and websockets>=15 honour
    *_proxy) must bypass it: add 127.0.0.1 and localhost to NO_PROXY unless they are already there."""
    proxies = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    if not any(environ.get(k) for k in proxies):
        return {}
    have = [h.strip() for h in (environ.get("NO_PROXY") or environ.get("no_proxy") or "").split(",") if h.strip()]
    missing = [h for h in ("127.0.0.1", "localhost") if h not in have]
    if not missing:
        return {}
    value = ",".join(have + missing)
    return {"NO_PROXY": value, "no_proxy": value}


def atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path: str, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_status(run_dir: str) -> Optional[dict]:
    return read_json(os.path.join(run_dir, "status.json"))


def other_host(st: dict) -> Optional[str]:
    """The host that supervises the run when it is not this one: its pids and /proc start times mean nothing
    here (run dirs live on the NAS), so nothing is checked or signalled from this host."""
    host = (st.get("supervisor") or {}).get("host")
    return host if host and host != socket.gethostname() else None


# ---------------------------------------------------------------------- plan


@dataclasses.dataclass
class ProcSpec:
    name: str  # executor | agent | evaluator
    argv: list
    cwd: str
    log: str
    env: dict = dataclasses.field(default_factory=dict)  # set on top of the inherited environment (no secrets)
    unset: list = dataclasses.field(default_factory=list)  # removed from the inherited environment
    health_url: Optional[str] = None
    start_timeout_s: float = 0.0
    stop: list = dataclasses.field(default_factory=lambda: [["SIGINT", 60.0], ["SIGTERM", 15.0]])

    def child_env(self, base: dict) -> dict:
        env = {k: v for k, v in base.items() if k not in self.unset}
        env.update(self.env)
        return env

    def shell(self) -> str:
        env = ["env", *(f"-u{k}" for k in self.unset), *(f"{k}={v}" for k, v in self.env.items())]
        return f"(cd {shlex.quote(self.cwd)} && {shlex.join(env + list(self.argv))}) > {shlex.quote(self.log)} 2>&1"


@dataclasses.dataclass
class Plan:
    run_id: str
    run_dir: str
    created_at: str
    launch_argv: list
    task: str
    mode: str
    instance_indices: list
    instance_ids: list
    wrapper: str
    env_wrapper: str
    seed: int
    config: str
    overrides: list
    agent_overrides: list
    config_name: str
    config_hash: str
    direct: bool
    ports: dict
    gpus: dict
    max_steps: Optional[int]
    max_steps_info: dict
    evaluator_root: str
    evaluator_git: dict
    data_root: str
    eval_output_dir: str
    expected_results: list
    agent_manifest: dict
    tags: dict
    external_health: list
    evaluator_script: str
    options: dict
    processes: list
    warnings: list

    def proc(self, name: str) -> Optional[ProcSpec]:
        return next((p for p in self.processes if p.name == name), None)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        d = dict(d)
        d["processes"] = [ProcSpec(**p) for p in d["processes"]]
        return Plan(**d)


def default_run_dir(cfg: dict, run_id: str) -> str:
    root = get(cfg, "run.log_root") or get(cfg, "run.local_log_root", "~/b1k_local_logs")
    return os.path.abspath(os.path.join(os.path.expanduser(str(root)), run_id))


def default_run_id(cfg: dict, task: str, mode: str, indices: Sequence[int], wrapper: str, seed: int, direct: bool) -> str:
    name = "L0_direct" if direct else str(get(cfg, "run.name", "run"))
    short = {"public_test": "pt", "hidden_test": "ht", "train": "tr"}[mode]
    return f"{name}-{task}-{short}{'_'.join(map(str, indices))}-{wrapper}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}"


def will_start_executor(cfg: dict, direct: bool, no_executor: bool) -> bool:
    return (direct or str(get(cfg, "executor.impl", "openpi_ws")) == "openpi_ws") and not no_executor


def gpus_needed(start_executor: bool, share_gpu: bool) -> int:
    return 2 if start_executor and not share_gpu else 1


def _select_gpus(args, n: int, gpu_query: Optional[Callable], warnings: list) -> list:
    explicit = [int(x) for x in str(args.gpus).split(",") if x.strip()] if args.gpus else None
    if explicit is not None and len(explicit) < n:
        raise PlanError(f"--gpus gives {len(explicit)} GPU(s), this run needs {n} (evaluator[,executor])")
    if explicit is not None and args.skip_gpu_check:
        return explicit[:n]
    query = gpu_query or gpumod.query_gpus
    limits = dict(max_mem_mib=args.max_gpu_mem_mib, max_util=args.max_gpu_util)
    try:
        if explicit is not None:
            gpumod.check_gpus_free(explicit[:n], gpus=query(), **limits)
            return explicit[:n]
        if args.wait_gpu and not args.dry_run:
            return gpumod.wait_for_free_gpus(n, query=query, timeout_s=args.wait_gpu_timeout_s, **limits)
        return gpumod.pick_free_gpus(n, gpus=query(), **limits)
    except gpumod.GpuError as e:
        if not args.dry_run:
            raise PlanError(f"{e} (nothing started; never use a busy GPU; --wait-gpu waits)") from e
        warnings.append(f"GPU check (dry run): {e}")
        return explicit[:n] if explicit else [f"<gpu{i}>" for i in range(n)]


def render_evaluator_script(*, run_id: str, cwd: str, env: dict, pythonpath: Optional[str], env_setup: Optional[str],
                            activation: str, behavior_env: str, conda_exe: str, evaluator_python: str,
                            cmd_prefix: Sequence[str], eval_args: Sequence[str]) -> str:
    q = shlex.quote
    lines = ["#!/usr/bin/env bash", f"# Official evaluator of run {run_id}; generated by eval/launch.py, which starts it",
             "# after the servers' /healthz answer. Re-runnable by hand: bash run_evaluator.sh", "set -e"]
    if env_setup:
        lines.append(f"source {q(env_setup)}  # B1K_ENV_SETUP; the exports below win (--data-root / --appdata change them)")
    lines.append(f"cd {q(cwd)}")
    lines += [f"export {k}={q(str(v))}" for k, v in env.items()]
    if pythonpath:
        lines.append(f"export PYTHONPATH={q(pythonpath)}${{PYTHONPATH:+:$PYTHONPATH}}  # evaluator root != installed tree")
    module = ["-m", EVAL_MODULE, *eval_args]
    if activation == "conda-run":
        cmd = [conda_exe, "run", "-p", behavior_env, "--no-capture-output", *cmd_prefix, "python", *module]
    elif activation == "activate":
        lines += [f'eval "$({q(conda_exe)} shell.bash hook)"', f"conda activate {q(behavior_env)}"]
        cmd = [*cmd_prefix, "python", *module]
    else:
        cmd = [*cmd_prefix, evaluator_python, *module]
    lines.append("exec " + shlex.join(cmd))
    return "\n".join(lines) + "\n"


def build_plan(args: argparse.Namespace, *, environ: Optional[dict] = None, gpu_query: Optional[Callable] = None) -> Plan:
    environ = dict(os.environ if environ is None else environ)
    warnings: list[str] = []

    def problem(msg: str) -> None:
        if not args.dry_run:
            raise PlanError(msg)
        warnings.append(msg)

    cfg = load_config(args.config, args.override)
    task, mode, direct = args.task, args.mode, bool(args.direct)
    indices = [int(i) for i in args.instances]
    if len(set(indices)) != len(indices):
        raise PlanError(f"duplicate instance indices {indices}: the evaluator writes one JSON per instance id")
    try:
        instance_ids = resolve_instance_ids(indices, mode)
    except ValueError as e:
        raise PlanError(str(e)) from e
    if len(indices) > 1:
        warnings.append(f"{len(indices)} envs on one policy port: on 2026/eval they share ONE max_steps-second budget "
                        "(evaluator.py time_limit_seconds), the agent budgets each env separately")
    cfg_seed = int(get(cfg, "run.seed", 0) or 0)
    seed = int(args.seed) if args.seed is not None else cfg_seed
    run_id = args.run_id or default_run_id(cfg, task, mode, indices, args.wrapper, seed, direct)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+=-]*", run_id):
        raise PlanError(f"run id {run_id!r}: use letters, digits and _.+=- only")
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir or default_run_dir(cfg, run_id)))
    if not args.dry_run and os.path.exists(os.path.join(run_dir, "status.json")):
        raise PlanError(f"{run_dir} already holds a run (status.json); every run needs a new --run-id")
    logs = os.path.join(run_dir, "logs")

    exec_impl = str(get(cfg, "executor.impl", "openpi_ws"))
    executor_needed = direct or exec_impl == "openpi_ws"
    start_executor = will_start_executor(cfg, direct, args.no_executor)
    start_agent = not direct
    exec_host = str(get(cfg, "executor.params.host", "127.0.0.1"))
    cfg_exec_port = int(get(cfg, "executor.params.port", 8000))
    exec_port = int(args.executor_port or cfg_exec_port)
    agent_host = str(get(cfg, "server.host", "127.0.0.1"))
    agent_port = int(args.agent_port or get(cfg, "server.port", 8100))
    if start_agent and executor_needed and agent_port == exec_port:
        raise PlanError(f"agent and executor would share port {agent_port}")
    if executor_needed and get(cfg, "executor.params.port_per_env") and len(indices) > 1:
        raise PlanError("executor.params.port_per_env with several envs needs one executor server per env "
                        "(not supported here)")
    eval_host = connect_host(exec_host if direct else agent_host)
    eval_port = exec_port if direct else agent_port
    for name, host, port, starts in (("executor", exec_host, exec_port, start_executor),
                                     ("agent", agent_host, agent_port, start_agent)):
        if starts and port_in_use(connect_host(host), port):
            problem(f"{name} port {port} is already in use (not ours to stop); choose another with --{name}-port")

    b1k_root = os.path.expanduser(str(get(cfg, "paths.b1k_root") or ""))
    openpi_root = os.path.expanduser(str(get(cfg, "paths.openpi_root") or ""))
    evaluator_root = os.path.abspath(os.path.expanduser(args.evaluator_root or b1k_root))
    data_root = os.path.expanduser(
        args.data_root or get(cfg, "paths.data_root") or environ.get("OMNIGIBSON_DATA_PATH")
        or os.path.join(b1k_root, "datasets")
    )
    appdata = os.path.expanduser(
        args.appdata or get(cfg, "paths.appdata") or environ.get("OMNIGIBSON_APPDATA_PATH") or DEFAULT_APPDATA
    )
    behavior_env = os.path.expanduser(args.behavior_env or get(cfg, "paths.behavior_env") or DEFAULT_BEHAVIOR_ENV)
    if not os.path.isfile(os.path.join(evaluator_root, "OmniGibson", "omnigibson", "eval", "eval.py")):
        problem(f"no official evaluator at {evaluator_root}/OmniGibson/omnigibson/eval/eval.py (--evaluator-root)")
    if not os.path.isdir(data_root):
        problem(f"data root {data_root} does not exist (OMNIGIBSON_DATA_PATH; --data-root)")
    env_setup = args.env_setup or environ.get("B1K_ENV_SETUP") or None
    if env_setup and not os.path.isfile(os.path.expanduser(env_setup)):
        problem(f"env setup script {env_setup} (B1K_ENV_SETUP / --env-setup) not found")
    conda_exe = args.conda_exe or environ.get("CONDA_EXE") or shutil.which("conda", path=environ.get("PATH")) or "conda"
    if args.activation != "none":
        if not (os.path.isfile(conda_exe) or shutil.which(conda_exe, path=environ.get("PATH"))):
            msg = f"conda executable {conda_exe!r} not found (--conda-exe, $CONDA_EXE, PATH)"
            if env_setup:
                warnings.append(msg + "; assuming the env setup script provides it")
            else:
                problem(msg)
        if not os.path.isdir(behavior_env):
            problem(f"behavior conda env {behavior_env} not found (--behavior-env / paths.behavior_env)")

    metadata = os.path.join(data_root, METADATA_SUBDIR)
    task_misc = os.path.join(metadata, "B100_task_misc.csv")
    if not os.path.isfile(task_misc):
        task_misc = os.path.expanduser(str(get(cfg, "task_metadata.task_misc_csv") or ""))
    task_index = None
    if task_misc and os.path.isfile(task_misc):
        task_ids = load_task_ids(task_misc)
        if task not in task_ids:
            problem(f"task {task!r} is not in {task_misc}; the official evaluator would reject it")
        task_index = task_ids.get(task)
    task_jsonl = os.path.join(metadata, "task.jsonl")
    official, official_err = None, None
    try:
        official = official_max_steps(task, task_jsonl, task_index)
    except (OSError, ValueError, KeyError) as e:
        official_err = f"{type(e).__name__}: {e}"
    if args.max_steps is not None:
        max_steps, source, to_eval = int(args.max_steps), "cli", True
    elif args.agent_max_steps is not None:
        max_steps, source, to_eval = int(args.agent_max_steps), "cli_agent_only", False
    elif official is not None:
        max_steps, source, to_eval = official, "task.jsonl", False
    else:
        max_steps, source, to_eval = None, "unresolved", False
        if start_agent:
            problem(f"cannot compute the official max_steps ({official_err}); pass --max-steps or --agent-max-steps")
        else:
            warnings.append("max_steps unknown before the run; the evaluator uses its own default (read back from its log)")
    if official is not None and source.startswith("cli") and max_steps != official:
        warnings.append(f"max_steps {max_steps} differs from the official default {official}")
    if to_eval:
        warnings.append("--max-steps is passed to the evaluator: not the official default timeout "
                        "(Dev/diagnostic runs only)")
    max_steps_info = {"value": max_steps, "source": source, "to_evaluator": to_eval, "official": official,
                      "official_error": official_err, "task_jsonl": task_jsonl, "task_index": task_index}

    gpus = _select_gpus(args, gpus_needed(start_executor, args.share_gpu), gpu_query, warnings)
    eval_gpus = gpus[:1]
    exec_gpus = (gpus[:1] if args.share_gpu else gpus[1:2]) if start_executor else []
    proxy = no_proxy_env(environ)
    secrets = secret_env_names(cfg)  # unset for the executor and the evaluator (names only; values are never written)
    processes: list[ProcSpec] = []
    srv = dict(get(cfg, "executor.server", {}) or {})
    exec_settings: dict = {}
    if executor_needed:
        ckpt = os.path.expanduser(str(get(cfg, "executor.params.checkpoint_dir") or ""))
        repo_id = args.executor_repo_id or srv.get("repo_id")
        if not repo_id and start_executor:
            repo_id = detect_repo_id(ckpt) if os.path.isdir(ckpt) else None
            if not repo_id:
                problem(f"cannot find <ckpt>/assets/<asset id>/norm_stats.json under {ckpt!r}; pass --executor-repo-id")
                repo_id = "<repo-id>"
        exec_settings = {
            "robot": srv.get("robot", "b1k/R1Pro"), "task": srv.get("task") or f"b1k/{task}", "repo_id": repo_id,
            "train_config": srv.get("train_config", "pi05_b1k"), "checkpoint_dir": ckpt,
            "control_mode": str(srv.get("control_mode", "receding_horizon")),
            "action_horizon": int(srv.get("action_horizon", 16)),
            "rename_robot_from": srv.get("rename_robot_from", "robot_r1"), "multi_task": bool(srv.get("multi_task", False)),
        }
    if start_executor:
        exec_python = args.executor_python or srv.get("python") or os.path.join(openpi_root, ".venv", "bin", "python")
        script = srv.get("script") or os.path.join(REPO_ROOT, "executor", "openpi_server", "serve_b1k_prompted.py")
        if not os.path.isdir(exec_settings["checkpoint_dir"]):
            problem(f"checkpoint dir {exec_settings['checkpoint_dir']!r} not found (executor.params.checkpoint_dir)")
        if not os.path.isfile(exec_python):
            problem(f"executor python {exec_python} not found (--executor-python / executor.server.python)")
        argv = [exec_python, script, "--robot", exec_settings["robot"], "--task", exec_settings["task"],
                "--repo-id", str(exec_settings["repo_id"]), "--policy.config", exec_settings["train_config"],
                "--policy.dir", exec_settings["checkpoint_dir"], "--control-mode", exec_settings["control_mode"],
                "--action-horizon", str(exec_settings["action_horizon"]), "--host", exec_host, "--port", str(exec_port)]
        if exec_settings["rename_robot_from"]:
            argv += ["--rename-robot-from", str(exec_settings["rename_robot_from"])]
        if exec_settings["multi_task"]:
            argv.append("--multi-task")
        mem = args.executor_mem_fraction or srv.get("xla_mem_fraction") or (0.35 if args.share_gpu else 0.85)
        processes.append(ProcSpec(
            "executor", argv, cwd=openpi_root, log=os.path.join(logs, "executor.log"),
            env={**gpumod.cuda_env(exec_gpus), "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem), **proxy},
            unset=["PYTHONPATH", "PYTHONHOME", *secrets],
            health_url=f"http://{connect_host(exec_host)}:{exec_port}/healthz",
            start_timeout_s=float(args.executor_start_timeout_s), stop=[["SIGINT", 30.0], ["SIGTERM", 15.0]],
        ))

    agent_overrides = list(args.override)
    if executor_needed and exec_port != cfg_exec_port:
        agent_overrides.append(f"executor.params.port={exec_port}")
    if seed != cfg_seed:
        agent_overrides.append(f"run.seed={seed}")
    cfg_agent = load_config(args.config, agent_overrides)
    tags = {"task": task, "mode": mode, "instance_indices": ",".join(map(str, indices)),
            "instance_ids": ",".join(map(str, instance_ids)), "wrapper": args.wrapper, "seed": str(seed)}
    tags.update(dict(t.split("=", 1) for t in args.tag))
    if direct:
        config_name = "L0_direct"
        chash = hashlib.sha256(json.dumps({"direct": True, "executor": exec_settings}, sort_keys=True).encode()).hexdigest()
    else:
        config_name, chash = str(get(cfg_agent, "run.name", "run")), config_hash(cfg_agent)
    local_root = os.path.expanduser(str(get(cfg_agent, "run.local_log_root", "~/b1k_local_logs")))
    log_root = get(cfg_agent, "run.log_root")
    agent_manifest = {
        "local": os.path.join(local_root, run_id, "run_manifest.json"),
        "nas": os.path.join(os.path.expanduser(str(log_root)), run_id, "run_manifest.json") if log_root else None,
    } if start_agent else {}
    if start_agent:
        argv = [args.agent_python or sys.executable, "-m", "agent.server", "--config", os.path.abspath(args.config)]
        for ov in agent_overrides:
            argv += ["-o", ov]
        argv += ["--port", str(agent_port), "--run-id", run_id]
        if max_steps is not None:
            argv += ["--max-steps", str(max_steps)]
        for k, v in tags.items():
            argv += ["--tag", f"{k}={v}"]
        processes.append(ProcSpec(
            "agent", argv, cwd=REPO_ROOT, log=os.path.join(logs, "agent.log"),
            env={"PYTHONPATH": REPO_ROOT, **gpumod.cuda_env([]), **proxy}, unset=["PYTHONHOME"],
            health_url=f"http://{connect_host(agent_host)}:{agent_port}/healthz",
            start_timeout_s=float(args.agent_start_timeout_s), stop=[["SIGINT", 60.0], ["SIGTERM", 15.0]],
        ))
    external_health = []
    if executor_needed and not start_executor:
        external_health.append(f"http://{connect_host(exec_host)}:{exec_port}/healthz")

    eval_output_dir = os.path.abspath(os.path.expanduser(args.eval_output_dir or os.path.join(run_dir, "eval_output")))
    expected_results = [f"{task}_{iid}_0.json" for iid in instance_ids]
    stale = [f for f in expected_results if os.path.exists(os.path.join(eval_output_dir, "json", f))]
    if stale:
        problem(f"{eval_output_dir}/json already holds {stale}: they would count as this run's results "
                "(use a fresh --eval-output-dir)")
    eval_args = ["--task-name", task, "--host", eval_host, "--port", str(eval_port), "--mode", mode,
                 "--instance-indices", *map(str, indices), "--num-envs", str(len(indices)),
                 "--env-wrapper", WRAPPERS[args.wrapper], "--output-dir", eval_output_dir]
    if args.write_video:
        eval_args.append("--write-video")
    if to_eval:
        eval_args += ["--max-steps", str(max_steps)]
    if args.robot_config:
        eval_args += ["--robot-config", os.path.abspath(os.path.expanduser(args.robot_config))]
    if args.replay_action_chunk_size:
        eval_args += ["--replay-action-chunk-size", str(int(args.replay_action_chunk_size))]
    same_tree = bool(b1k_root) and os.path.realpath(evaluator_root) == os.path.realpath(b1k_root)
    eval_env = {"OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", **gpumod.cuda_env(eval_gpus),
                "OMNIGIBSON_DATA_PATH": data_root, "OMNIGIBSON_APPDATA_PATH": appdata, **proxy}
    script = render_evaluator_script(
        run_id=run_id, cwd=evaluator_root, env=eval_env,
        pythonpath=None if same_tree else os.path.join(evaluator_root, "OmniGibson"),
        env_setup=os.path.expanduser(env_setup) if env_setup else None, activation=args.activation,
        behavior_env=behavior_env, conda_exe=conda_exe,
        evaluator_python=args.evaluator_python, cmd_prefix=shlex.split(args.evaluator_cmd_prefix or ""),
        eval_args=eval_args,
    )
    processes.append(ProcSpec(
        "evaluator", ["bash", os.path.join(run_dir, "run_evaluator.sh")], cwd=evaluator_root,
        log=os.path.join(logs, "evaluator.log"), unset=["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", *secrets],
        stop=[["SIGINT", 120.0], ["SIGTERM", 30.0]],
    ))

    return Plan(
        run_id=run_id, run_dir=run_dir, created_at=now_iso(), launch_argv=list(getattr(args, "argv", []) or []),
        task=task, mode=mode, instance_indices=indices, instance_ids=instance_ids, wrapper=args.wrapper,
        env_wrapper=WRAPPERS[args.wrapper], seed=seed, config=os.path.abspath(args.config), overrides=list(args.override),
        agent_overrides=agent_overrides if start_agent else [], config_name=config_name, config_hash=chash, direct=direct,
        ports={"executor": exec_port if executor_needed else None, "agent": agent_port if start_agent else None,
               "evaluator_target": eval_port},
        gpus={"evaluator": eval_gpus, "executor": exec_gpus, "agent": []},
        max_steps=max_steps, max_steps_info=max_steps_info, evaluator_root=evaluator_root,
        evaluator_git=git_info(evaluator_root), data_root=data_root, eval_output_dir=eval_output_dir,
        expected_results=expected_results, agent_manifest=agent_manifest, tags=tags,
        external_health=external_health, evaluator_script=script,
        options={"eval_timeout_s": args.eval_timeout_s, "server_death_grace_s": float(args.server_death_grace_s),
                 "strict_exit": bool(args.strict_exit), "sync_videos": bool(args.sync_videos),
                 "poll_s": float(args.poll_s), "heartbeat_s": 30.0, "gpu_mismatch": args.gpu_mismatch,
                 "executor_settings": exec_settings},
        processes=processes, warnings=warnings,
    )


def render_commands(plan: Plan) -> str:
    out = ["#!/usr/bin/env bash", f"# Exact commands of run {plan.run_id} (eval/launch.py, {plan.created_at}).",
           "# The launcher runs each in its own process group, waits for every server's /healthz, then runs the",
           "# evaluator. Variables not shown are inherited from the launcher's environment (provider keys stay there)."]
    for p in plan.processes:
        out += ["", f"# --- {p.name}  (log {p.log}{', health ' + p.health_url if p.health_url else ''})"]
        out.append(p.shell() + (" &" if p.health_url else ""))
    for url in plan.external_health:
        out.insert(4, f"# requires an executor server already answering {url} (--no-executor)")
    out += ["", "# --- run_evaluator.sh", *("# " + line for line in plan.evaluator_script.splitlines())]
    return "\n".join(out) + "\n"


def initial_status(plan: Plan) -> dict:
    return {
        "run_id": plan.run_id, "state": "planned", "reason": [], "warnings": list(plan.warnings),
        "created_at": plan.created_at, "started_at": None, "ended_at": None, "updated_at": now_iso(),
        "task": plan.task, "mode": plan.mode, "instance_indices": plan.instance_indices, "instance_ids": plan.instance_ids,
        "num_envs": len(plan.instance_ids), "num_rollouts": 1, "wrapper": plan.wrapper, "env_wrapper": plan.env_wrapper,
        "seed": plan.seed, "config": plan.config, "overrides": plan.overrides, "config_name": plan.config_name,
        "config_hash": plan.config_hash, "direct": plan.direct, "tags": plan.tags, "ports": plan.ports, "gpus": plan.gpus,
        "max_steps": plan.max_steps, "max_steps_info": plan.max_steps_info, "evaluator_root": plan.evaluator_root,
        "evaluator_git": plan.evaluator_git, "data_root": plan.data_root, "run_dir": plan.run_dir,
        "eval_output_dir": plan.eval_output_dir, "expected_results": plan.expected_results, "results": [],
        "processes": {}, "exit_codes": {}, "evaluator_exit_code": None, "infra_error": False, "infra_errors": [],
        "infra_kinds": [], "results_complete": False,
        "agent_manifest": dict(plan.agent_manifest), "supervisor": None, "progress": {},
    }


def prepare_run_dir(plan: Plan) -> None:
    os.makedirs(os.path.join(plan.run_dir, "logs"), exist_ok=True)
    os.makedirs(plan.eval_output_dir, exist_ok=True)
    write_json(os.path.join(plan.run_dir, "plan.json"), plan.to_dict())
    atomic_write(os.path.join(plan.run_dir, "commands.sh"), render_commands(plan))
    atomic_write(os.path.join(plan.run_dir, "run_evaluator.sh"), plan.evaluator_script)
    write_json(os.path.join(plan.run_dir, "status.json"), initial_status(plan))


def load_plan(run_dir: str) -> Plan:
    data = read_json(os.path.join(run_dir, "plan.json"))
    if data is None:
        raise PlanError(f"no plan.json in {run_dir}")
    return Plan.from_dict(data)


# ---------------------------------------------------------------------- outcome


_LOG_PATTERNS = {
    "timeout_default": re.compile(r"Setting timeout to be ([\d.]+)x the average length of human demos: (\d+)"),
    "timeout_config": re.compile(r"Setting timeout to be (\d+) steps through config"),
    "resolved": re.compile(r"Resolved (\w+) instance ids for (\S+): \[([^\]]*)\]"),
    "instance": re.compile(r"Instance (\d+) \(env (\d+)\) finished at step (\d+): success=(\w+) q_score=(\S+)"),
    "summary": re.compile(r"Eval summary: (\d+)/(\d+) success \| mean q_score=([0-9.]+)"),
    "failure": re.compile(r"Policy evaluation failed at step (\d+) for instance (\d+): (.*)"),
}
# OmniGibson/omnigibson/__init__.py LogFormatter wraps every record in colour codes ("\033[0m...\033[0m")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _float_or_none(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def parse_evaluator_log(lines) -> dict:
    """The official evaluator's own log lines (evaluator.py / eval.py logger.info), read after the run."""
    out: dict = {"max_steps": None, "max_steps_source": None, "resolved_instance_ids": None, "instances": [],
                 "summary": None, "failures": [], "tracebacks": 0}
    for line in lines:
        line = _ANSI.sub("", line)
        if "Traceback (most recent call last)" in line:
            out["tracebacks"] += 1
        if m := _LOG_PATTERNS["timeout_default"].search(line):
            out["max_steps"], out["max_steps_source"] = int(m.group(2)), "default"
        elif m := _LOG_PATTERNS["timeout_config"].search(line):
            out["max_steps"], out["max_steps_source"] = int(m.group(1)), "config"
        elif m := _LOG_PATTERNS["resolved"].search(line):
            out["resolved_instance_ids"] = [int(x) for x in m.group(3).replace(" ", "").split(",") if x]
        elif m := _LOG_PATTERNS["instance"].search(line):
            out["instances"].append({"instance_id": int(m.group(1)), "env": int(m.group(2)), "step": int(m.group(3)),
                                     "success": m.group(4) == "True", "q_score": _float_or_none(m.group(5))})
        elif m := _LOG_PATTERNS["summary"].search(line):
            out["summary"] = {"success": int(m.group(1)), "n": int(m.group(2)), "mean_q_score": float(m.group(3))}
        elif m := _LOG_PATTERNS["failure"].search(line):
            out["failures"].append({"step": int(m.group(1)), "instance_id": int(m.group(2)),
                                    "message": m.group(3).rstrip()[:300]})
    return out


def read_result(path: str) -> dict:
    """Compact view of one official result JSON (read after the episode; never sent to the agent)."""
    rec = read_json(path)
    if rec is None:
        return {"file": os.path.basename(path), "present": False}
    return {"file": os.path.basename(path), "present": True, "instance_id": rec.get("instance_id"),
            "rollout_id": rec.get("rollout_id"), "success": rec.get("success"),
            "q_score": (rec.get("q_score") or {}).get("final"), "steps": rec.get("steps"),
            "failure_reason": rec.get("failure_reason")}


def classify_outcome(*, evaluator_exit: Optional[int], expected: Sequence[str], found: Sequence[str],
                     servers_died: Sequence[str] = (), died_after_exit: Sequence[str] = (), timed_out: bool = False,
                     strict_exit: bool = True, failure_reasons: Sequence[Optional[str]] = ()) -> dict:
    """{"state": "done" | "infra_error", "reasons", "warnings", "kinds"} of one evaluator run.

    infra_error: a server we started died while the evaluator ran, the evaluator timed out, an expected
    official JSON is missing, the evaluator exited non-zero (``strict_exit``; otherwise only a warning once
    every result exists), or a result's failure_reason is policy_connection_lost (2026/eval; eval.aggregate
    counts it as infrastructure too). Otherwise 'done', whatever success / q_score / failure_reason the
    JSONs hold: a policy failure is never an infrastructure error. A server found dead only after the
    evaluator exited is a warning. ``kinds`` names the causes for eval.batch's re-run rule."""
    reasons, warnings, kinds = [], [], []
    missing = [f for f in expected if f not in set(found)]
    if servers_died:
        kinds.append("server_died")
        reasons.append("server died before the evaluator finished: " + ", ".join(servers_died))
    if died_after_exit:
        warnings.append("server(s) found dead after the evaluator exited: " + ", ".join(died_after_exit))
    lost = sorted({r for r in failure_reasons if r in CONNECTION_FAILURES})
    if lost:
        kinds.append("connection_lost")
        reasons.append("official result failure_reason " + ", ".join(lost))
    if timed_out:
        kinds.append("timeout")
        reasons.append("evaluator exceeded --eval-timeout-s")
    if len(missing) == len(expected):
        kinds.append("no_results")
        reasons.append("no official result JSON")
    elif missing:
        kinds.append("missing_results")
        reasons.append("missing official result JSON: " + ", ".join(missing))
    if evaluator_exit not in (0, None):
        if strict_exit or missing:
            kinds.append("exit_code")
            reasons.append(f"evaluator exit code {evaluator_exit}")
            if not missing:
                warnings.append("all official result JSONs were written before the non-zero exit (shutdown crash?)")
        else:
            warnings.append(f"evaluator exit code {evaluator_exit} after all official results were written")
    return {"state": "infra_error" if reasons else "done", "reasons": reasons, "warnings": warnings, "kinds": kinds}


# ---------------------------------------------------------------------- supervisor


class Supervisor:
    """Starts the plan's processes, waits for /healthz, runs the evaluator, classifies, stops its own children."""

    def __init__(self, plan: Plan, *, mode: str = "foreground", environ: Optional[dict] = None,
                 stop_event: Optional[threading.Event] = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, gpu_query: Callable = gpumod.query_gpus,
                 apps_query: Callable = gpumod.query_compute_apps):
        self.plan = plan
        self.mode = mode
        self.environ = dict(os.environ if environ is None else environ)
        self.stop_event = stop_event or threading.Event()
        self.clock = clock
        self.sleep = sleep
        self.gpu_query, self.apps_query = gpu_query, apps_query
        self.procs: dict[str, subprocess.Popen] = {}
        self.status = read_status(plan.run_dir) or initial_status(plan)
        self._last_beat = None

    def request_stop(self, *_args) -> None:
        self.stop_event.set()

    def _write(self) -> None:
        self.status["updated_at"] = now_iso()
        write_json(os.path.join(self.plan.run_dir, "status.json"), self.status)

    def _note(self, msg: str) -> None:
        self.status.setdefault("warnings", []).append(f"{now_iso()} {msg}")
        print(f"[eval.launch {self.plan.run_id}] {msg}", flush=True)

    def run(self) -> dict:
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                if signal.getsignal(sig) is not signal.SIG_IGN:  # keep nohup's ignored SIGHUP ignored
                    previous[sig] = signal.signal(sig, self.request_stop)
        try:
            return self._run()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _run(self) -> dict:
        me = proc_stat(os.getpid()) or {}
        self.status.update(state="starting", started_at=now_iso(), supervisor={
            "pid": os.getpid(), "start_ticks": me.get("start_ticks"), "mode": self.mode, "host": socket.gethostname()})
        self._write()
        outcome: dict = {}
        try:
            servers = [p for p in self.plan.processes if p.name != "evaluator"]
            for spec in servers:
                self._spawn(spec)
            for spec in servers:
                ok, why = self._wait_healthy(spec)
                if not ok:
                    outcome = {"forced": ("infra_error", [why], ["server_start"])}
                    break
            if not outcome:
                for url in self.plan.external_health:
                    if not self._wait_url(url, 60.0):
                        why = f"external executor server not answering {url} (--no-executor)"
                        outcome = {"forced": ("failed", [why], ["launcher"])}
            if not outcome:
                self._record_agent_manifest()
                self._spawn(self.plan.proc("evaluator"))
                self.status["state"] = "running"
                self._write()
                outcome = self._monitor()
        except StopRequested:
            outcome = {"forced": ("failed", ["stopped_by_user"], ["stopped_by_user"])}
        except GpuPlacementError as e:
            outcome = {"forced": ("failed", [str(e)], ["gpu_placement"])}
        except Exception as e:  # the supervisor must always stop its children and finalize
            outcome = {"forced": ("infra_error", [f"supervisor error: {type(e).__name__}: {e}"], ["supervisor_error"])}
            self.status["traceback"] = traceback.format_exc()[-4000:]
        finally:
            self._stop_all()
        try:
            return self._finalize(outcome)
        except Exception as e:  # never leave a non-terminal status.json behind
            json_dir = os.path.join(self.plan.eval_output_dir, "json")
            complete = all(read_result(os.path.join(json_dir, f))["present"] for f in self.plan.expected_results)
            why = f"finalize error: {type(e).__name__}: {e}"
            self.status.update(state="infra_error", reason=list(self.status.get("reason") or []) + [why],
                               infra_kinds=["supervisor_error"], infra_error=why, results_complete=complete,
                               evaluator_started="evaluator" in self.procs, ended_at=now_iso(),
                               traceback=traceback.format_exc()[-4000:])
            self._write()
            return self.status

    def _spawn(self, spec: ProcSpec) -> None:
        os.makedirs(os.path.dirname(spec.log), exist_ok=True)
        with open(spec.log, "ab") as logf:
            p = subprocess.Popen(spec.argv, cwd=spec.cwd, env=spec.child_env(self.environ), stdin=subprocess.DEVNULL,
                                 stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        self.procs[spec.name] = p
        st = proc_stat(p.pid) or {}
        self.status["processes"][spec.name] = {"pid": p.pid, "pgid": p.pid, "start_ticks": st.get("start_ticks"),
                                               "started_at": now_iso(), "started_by_us": True, "log": spec.log}
        self._write()

    def _wait_url(self, url: str, timeout_s: float, proc: Optional[subprocess.Popen] = None) -> bool:
        deadline = self.clock() + timeout_s
        while True:
            if self.stop_event.is_set():
                raise StopRequested()
            if healthz_ok(url):
                return True
            if (proc is not None and proc.poll() is not None) or self.clock() > deadline:
                return False
            self._heartbeat()
            self.sleep(min(2.0, self.plan.options.get("poll_s", 2.0)))

    def _wait_healthy(self, spec: ProcSpec) -> tuple[bool, str]:
        p = self.procs[spec.name]
        if self._wait_url(spec.health_url, spec.start_timeout_s, proc=p):
            self.status["processes"][spec.name]["healthy_at"] = now_iso()
            self._write()
            return True, ""
        if p.poll() is not None:
            return False, f"{spec.name} exited with code {p.returncode} before /healthz answered (see {spec.log})"
        return False, f"{spec.name} /healthz did not answer within {spec.start_timeout_s:.0f} s (see {spec.log})"

    def _monitor(self) -> dict:
        opts = self.plan.options
        ev = self.procs["evaluator"]
        t0, first_death, died, timed_out = self.clock(), None, {}, False
        while ev.poll() is None:
            if self.stop_event.is_set():
                raise StopRequested()
            for name, p in self.procs.items():
                if name != "evaluator" and name not in died and p.poll() is not None:
                    died[name] = p.returncode
                    if first_death is None:
                        first_death = self.clock()
                    self._note(f"{name} exited with code {p.returncode} while the evaluator was running")
            if first_death is not None and self.clock() - first_death > opts["server_death_grace_s"]:
                self._note("stopping the evaluator: a server died and the evaluator did not exit within the grace period")
                self._stop("evaluator")
                break
            if opts.get("eval_timeout_s") and self.clock() - t0 > float(opts["eval_timeout_s"]):
                timed_out = True
                self._note(f"stopping the evaluator after --eval-timeout-s {opts['eval_timeout_s']}")
                self._stop("evaluator")
                break
            self._heartbeat()
            self.sleep(opts.get("poll_s", 2.0))
        rc = ev.wait()
        after = sorted(n for n, p in self.procs.items() if n != "evaluator" and n not in died and p.poll() is not None)
        return {"evaluator_exit": rc, "servers_died": sorted(died), "died_after_exit": after, "timed_out": timed_out}

    def _heartbeat(self) -> None:
        if self._last_beat is not None and self.clock() - self._last_beat < self.plan.options.get("heartbeat_s", 30.0):
            return
        self._last_beat = self.clock()
        json_dir = os.path.join(self.plan.eval_output_dir, "json")
        found = [f for f in self.plan.expected_results if os.path.isfile(os.path.join(json_dir, f))]
        self.status["progress"] = {"results_found": len(found), "results_expected": len(self.plan.expected_results),
                                   "alive": {n: p.poll() is None for n, p in self.procs.items()}}
        self._check_gpu_placement()
        self._write()

    def _check_gpu_placement(self) -> None:
        """Read-only check of which boards our process groups really use (nvidia-smi compute apps). CUDA honours
        CUDA_VISIBLE_DEVICES; whether Isaac Sim's renderer does is unverified, so a process found on a board it
        was not given stops the run (--gpu-mismatch stop, default) or is only noted (warn)."""
        try:
            by_uuid = {g.uuid: g.index for g in self.gpu_query() if g.uuid}
            apps = self.apps_query()
        except gpumod.GpuError:
            return
        for name, proc in self.procs.items():
            pids = {m["pid"] for m in group_members(proc.pid)}
            seen = sorted({by_uuid.get(a.gpu_uuid) for a in apps if a.pid in pids} - {None})
            if not seen:
                continue
            expected = [int(g) for g in self.plan.gpus.get(name) or [] if str(g).isdigit()]
            self.status.setdefault("gpu_placement", {})[name] = {"expected": expected, "seen": seen}
            stray = sorted(set(seen) - set(expected))
            if stray:
                msg = f"{name} uses GPU(s) {stray} it was not assigned (assigned {expected})"
                if self.plan.options.get("gpu_mismatch", "stop") == "stop":
                    raise GpuPlacementError(msg + "; stopped (--gpu-mismatch warn only notes it)")
                self._note(msg)

    def _record_agent_manifest(self) -> None:
        """Point status.json at the agent's run_manifest.json (and copy it into the run dir when it lives elsewhere)."""
        ptr = self.status["agent_manifest"]
        if not ptr:
            return
        for key in ("nas", "local"):
            path = ptr.get(key)
            man = read_json(path) if path else None
            if man is None:
                continue
            ptr.update(found=path, config_hash=man.get("config_hash"), run_id=man.get("run_id"))
            if man.get("config_hash") != self.plan.config_hash:
                self._note(f"agent manifest config_hash {man.get('config_hash')} != planned {self.plan.config_hash}")
            dest = os.path.join(self.plan.run_dir, "run_manifest.json")
            try:
                if not os.path.exists(dest):
                    shutil.copy2(path, dest)
            except OSError as e:
                self._note(f"copying {path} to {dest} failed: {e}")
            break
        else:
            self._note(f"agent run_manifest.json not found at {ptr.get('nas')} or {ptr.get('local')}")
        self._write()

    def _stop(self, name: str) -> None:
        p, rec, spec = self.procs.get(name), self.status["processes"].get(name), self.plan.proc(name)
        if p is None or rec is None or rec.get("ended_at"):
            return
        actions = []
        for signame, wait_s in list(spec.stop) + [["SIGKILL", 5.0]]:
            if p.poll() is not None and not group_members(p.pid):
                break
            result = signal_own_group(p.pid, getattr(signal, signame), rec.get("start_ticks"))
            actions.append(f"{signame}: {result}")
            if result != "sent":
                break
            deadline = self.clock() + float(wait_s)
            while self.clock() < deadline and (p.poll() is None or group_members(p.pid)):
                self.sleep(0.2)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            actions.append("still running after SIGKILL")
        rec.update(ended_at=now_iso(), exit_code=p.returncode, stop_actions=actions)

    def _stop_all(self) -> None:
        for name in ("evaluator", "agent", "executor"):
            p = self.procs.get(name)
            if p is None:
                continue
            if p.poll() is not None and not group_members(p.pid):
                self.status["processes"][name].setdefault("ended_at", now_iso())
                self.status["processes"][name]["exit_code"] = p.returncode
            else:
                self._stop(name)
            self.status["exit_codes"][name] = p.returncode

    def _finalize(self, outcome: dict) -> dict:
        plan, st = self.plan, self.status
        json_dir = os.path.join(plan.eval_output_dir, "json")
        listed = sorted(os.listdir(json_dir)) if os.path.isdir(json_dir) else []
        evaluator_started = "evaluator" in self.procs
        st["results"] = [read_result(os.path.join(json_dir, f)) for f in plan.expected_results]
        found = [r["file"] for r in st["results"] if r["present"]]  # a truncated / unreadable JSON is no result
        if "forced" in outcome:
            state, reasons, kinds = outcome["forced"]
            warnings: list = []
        else:
            verdict = classify_outcome(
                evaluator_exit=outcome["evaluator_exit"], expected=plan.expected_results, found=found,
                servers_died=outcome["servers_died"], died_after_exit=outcome["died_after_exit"],
                timed_out=outcome["timed_out"], strict_exit=plan.options["strict_exit"],
                failure_reasons=[r.get("failure_reason") for r in st["results"]])
            state, reasons, warnings, kinds = verdict["state"], verdict["reasons"], verdict["warnings"], verdict["kinds"]
        log_path = plan.proc("evaluator").log
        if os.path.isfile(log_path):
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    st["evaluator_log"] = parse_evaluator_log(f)
            except Exception as e:  # the log is diagnostic; the official JSONs decide the outcome
                st["evaluator_log"] = parse_evaluator_log([])
                warnings.append(f"could not parse {log_path}: {type(e).__name__}: {e}")
            logged = st["evaluator_log"]["max_steps"]
            if logged is not None:
                st["max_steps_info"]["evaluator_logged"] = logged
                if st["max_steps"] is not None and logged != st["max_steps"]:
                    warnings.append(f"agent budget used max_steps={st['max_steps']} but the evaluator used {logged}")
                st["max_steps"] = logged
        st["unexpected_results"] = [f for f in listed if f.endswith(".json") and f not in plan.expected_results]
        st["progress"] = {"results_found": sum(f in found for f in plan.expected_results),
                          "results_expected": len(plan.expected_results),
                          "alive": {n: p.poll() is None for n, p in self.procs.items()}}
        missing = [r for r, f in zip(plan.instance_ids, plan.expected_results) if f not in found]
        st.update(state=state, reason=reasons, infra_kinds=kinds if state != "done" else [], ended_at=now_iso(),
                  evaluator_started=evaluator_started, evaluator_exit_code=st["exit_codes"].get("evaluator"),
                  results_complete=not missing,
                  infra_error="; ".join(reasons) if state == "infra_error" else False,
                  infra_errors=[{"task": plan.task, "instance_id": iid, "rollout_id": 0,
                                 "reason": "no official result JSON"}
                                for iid in missing] if state == "infra_error" else [])
        st["warnings"] = list(st.get("warnings", [])) + warnings
        self._copy_results(listed)
        self._write()
        print(f"[eval.launch {plan.run_id}] {state}" + (f": {'; '.join(reasons)}" if reasons else ""), flush=True)
        return st

    def _copy_results(self, found: Sequence[str]) -> None:
        """With --eval-output-dir elsewhere (local disk), copy the official JSONs (and videos on request) here."""
        dest = os.path.join(self.plan.run_dir, "eval_output")
        if os.path.realpath(self.plan.eval_output_dir) == os.path.realpath(dest):
            return
        try:
            os.makedirs(os.path.join(dest, "json"), exist_ok=True)
            for f in found:
                shutil.copy2(os.path.join(self.plan.eval_output_dir, "json", f), os.path.join(dest, "json", f))
            videos = os.path.join(self.plan.eval_output_dir, "videos")
            if self.plan.options.get("sync_videos") and os.path.isdir(videos):
                shutil.copytree(videos, os.path.join(dest, "videos"), dirs_exist_ok=True)
        except OSError as e:
            self._note(f"copying results to {dest} failed: {e}")


# ---------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m eval.launch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", "-o", action="append", default=[], help="dotted key=value, also forwarded to the agent")
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", choices=MODES, default="public_test")
    ap.add_argument("--instances", type=int, nargs="+", required=True, help="--instance-indices of the evaluator")
    ap.add_argument("--wrapper", choices=sorted(WRAPPERS), default="rgbd")
    ap.add_argument("--seed", type=int, default=None,
                    help="agent seed (default run.seed); the evaluator's own seed is fixed")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", default=None, help="default <run.log_root>/<run_id>")
    ap.add_argument("--tag", action="append", default=[], help="extra key=value label for the agent's episodes")
    ap.add_argument("--direct", action="store_true", help="L0-direct: evaluator -> executor server, no agent")
    ap.add_argument("--no-executor", action="store_true", help="use an executor server already on --executor-port")
    ap.add_argument("--executor-port", type=int, default=None)
    ap.add_argument("--agent-port", type=int, default=None)
    ap.add_argument("--write-video", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="agent budget AND evaluator --max-steps (non-official)")
    ap.add_argument("--agent-max-steps", type=int, default=None, help="agent budget only; evaluator keeps its default")
    ap.add_argument("--robot-config", default=None)
    ap.add_argument("--replay-action-chunk-size", type=int, default=0)
    ap.add_argument("--evaluator-root", default=None, help="default paths.b1k_root; a 2026/eval worktree also works")
    ap.add_argument("--eval-output-dir", default=None, help="evaluator --output-dir (default <run_dir>/eval_output)")
    ap.add_argument("--data-root", default=None,
                    help="OMNIGIBSON_DATA_PATH (default paths.data_root, env, <b1k_root>/datasets)")
    ap.add_argument("--appdata", default=None, help=f"OMNIGIBSON_APPDATA_PATH (default {DEFAULT_APPDATA})")
    ap.add_argument("--behavior-env", default=None, help=f"conda env prefix (default {DEFAULT_BEHAVIOR_ENV})")
    ap.add_argument("--activation", choices=("conda-run", "activate", "none"), default="conda-run")
    ap.add_argument("--conda-exe", default=None, help="default $CONDA_EXE or conda")
    ap.add_argument("--evaluator-python", default="python", help="python for --activation none")
    ap.add_argument("--evaluator-cmd-prefix", default="", help="words put before `python -m omnigibson.eval.eval`")
    ap.add_argument("--env-setup", default=None, help="file sourced before the evaluator (default $B1K_ENV_SETUP)")
    ap.add_argument("--executor-python", default=None, help="default <paths.openpi_root>/.venv/bin/python")
    ap.add_argument("--executor-repo-id", default=None,
                    help="default executor.server.repo_id or the asset id in <ckpt>/assets")
    ap.add_argument("--executor-mem-fraction", type=float, default=None, help="XLA_PYTHON_CLIENT_MEM_FRACTION")
    ap.add_argument("--agent-python", default=None, help="default: this python")
    ap.add_argument("--gpus", default=None, help="explicit nvidia-smi indices EVAL[,EXEC]; still checked for being free")
    ap.add_argument("--share-gpu", action="store_true", help="executor and evaluator on one GPU")
    ap.add_argument("--skip-gpu-check", action="store_true", help="only for callers (eval.batch) that checked --gpus")
    ap.add_argument("--wait-gpu", action="store_true", help="poll nvidia-smi until enough GPUs are free")
    ap.add_argument("--wait-gpu-timeout-s", type=float, default=None)
    ap.add_argument("--max-gpu-mem-mib", type=float, default=gpumod.MAX_MEM_MIB)
    ap.add_argument("--max-gpu-util", type=float, default=gpumod.MAX_UTIL)
    ap.add_argument("--executor-start-timeout-s", type=float, default=1200.0)
    ap.add_argument("--agent-start-timeout-s", type=float, default=300.0)
    ap.add_argument("--eval-timeout-s", type=float, default=None)
    ap.add_argument("--server-death-grace-s", type=float, default=900.0,
                    help="after a server dies, give the evaluator this long to record the failure and exit")
    ap.add_argument("--poll-s", type=float, default=2.0, help="supervisor poll interval")
    ap.add_argument("--gpu-mismatch", choices=("stop", "warn"), default="stop",
                    help="a started process found on a GPU it was not given (nvidia-smi, every 30 s): stop or note")
    ap.add_argument("--strict-exit", action=argparse.BooleanOptionalAction, default=True,
                    help="non-zero evaluator exit = infra_error even when every result JSON exists (default on)")
    ap.add_argument("--sync-videos", action="store_true", help="with --eval-output-dir: also copy videos to the run dir")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and commands; start and write nothing")
    ap.add_argument("--detach", action="store_true", help="return after starting; a watcher process supervises")
    return ap


def parse_launch_args(argv: Sequence[str]) -> argparse.Namespace:
    args = build_parser().parse_args(list(argv))
    args.argv = list(argv)
    if args.max_steps is not None and args.agent_max_steps is not None:
        raise PlanError("--max-steps and --agent-max-steps are exclusive")
    return args


def describe_plan(plan: Plan) -> str:
    ms = plan.max_steps_info
    lines = [
        f"run {plan.run_id}  ->  {plan.run_dir}",
        f"task {plan.task}  {plan.mode} indices {plan.instance_indices} -> instance ids {plan.instance_ids}  "
        f"wrapper {plan.wrapper}  seed {plan.seed}  config {plan.config_name} ({plan.config_hash[:12]})",
        f"{'direct (evaluator -> executor)' if plan.direct else 'evaluator -> agent -> executor'}  ports {plan.ports}  "
        f"GPUs {plan.gpus}",
        f"max_steps {ms['value']} ({ms['source']}, official default {ms['official']}, to evaluator: {ms['to_evaluator']})",
        f"evaluator {plan.evaluator_root} ({(plan.evaluator_git or {}).get('describe')})",
    ]
    lines += [f"WARNING: {w}" for w in plan.warnings]
    return "\n".join(lines)


def spawn_watcher(plan: Plan, timeout_s: float = 60.0) -> int:
    """Start the detached supervisor (new session: survives the terminal) and return once it took over."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with open(os.path.join(plan.run_dir, "logs", "watcher.log"), "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "eval.launch", "_watch", plan.run_dir], cwd=REPO_ROOT, env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s and (read_status(plan.run_dir) or {}).get("state") == "planned":
        if p.poll() is not None:
            raise PlanError(f"watcher exited with code {p.returncode} before starting "
                            f"(see {plan.run_dir}/logs/watcher.log)")
        time.sleep(0.2)
    return p.pid


def cmd_launch(argv: Sequence[str]) -> int:
    try:
        args = parse_launch_args(argv)
        plan = build_plan(args)
    except (PlanError, OSError, ValueError) as e:
        print(f"eval.launch: error: {e}", file=sys.stderr)
        return EXIT_CODES["failed"]
    if args.dry_run:
        print(describe_plan(plan))
        print()
        print(render_commands(plan), end="")
        return 0
    try:
        prepare_run_dir(plan)
    except OSError as e:
        print(f"eval.launch: error: cannot write the run directory {plan.run_dir}: {e}", file=sys.stderr)
        return EXIT_CODES["failed"]
    print(describe_plan(plan), flush=True)
    if args.detach:
        try:
            pid = spawn_watcher(plan)
        except PlanError as e:
            print(f"eval.launch: error: {e}", file=sys.stderr)
            return EXIT_CODES["failed"]
        print(f"detached: watcher pid {pid}\n  python -m eval.launch status {plan.run_dir} [--wait]", flush=True)
        return 0
    return EXIT_CODES[Supervisor(plan).run()["state"]]


def _status_view(st: dict) -> dict:
    sup = st.get("supervisor") or {}
    view = dict(st)
    view["other_host"] = other_host(st)
    if view["other_host"]:  # cannot be checked from here
        view["supervisor_alive"], view["alive"] = None, {}
        return view
    view["supervisor_alive"] = process_alive(sup.get("pid"), sup.get("start_ticks")) if sup else False
    view["alive"] = {n: process_alive(p.get("pid"), p.get("start_ticks")) for n, p in (st.get("processes") or {}).items()}
    return view


def _print_status(view: dict) -> None:
    print(f"run {view.get('run_id')}: {view.get('state')}  "
          f"(started {view.get('started_at')}, ended {view.get('ended_at')})")
    print(f"  {view.get('task')} {view.get('mode')} {view.get('instance_indices')} -> {view.get('instance_ids')}  "
          f"{view.get('wrapper')}  seed {view.get('seed')}  max_steps {view.get('max_steps')}  "
          f"config {view.get('config_name')}")
    if view.get("other_host"):
        print(f"  supervised on host {view['other_host']}: processes not checked from here")
    else:
        procs = ", ".join(f"{n} pid {p.get('pid')} "
                          f"{'alive' if view['alive'].get(n) else 'exit ' + str(p.get('exit_code'))}"
                          for n, p in (view.get("processes") or {}).items())
        print(f"  processes: {procs or '-'}; supervisor {'alive' if view['supervisor_alive'] else 'not running'}")
    for r in view.get("results") or []:
        print(f"  {r.get('file')}: " + (f"success={r.get('success')} q={r.get('q_score')} steps={r.get('steps')}"
                                         + (f" failure_reason={r.get('failure_reason')}" if r.get("failure_reason") else "")
                                         if r.get("present") else "missing"))
    if view.get("progress"):
        print(f"  progress: {view['progress']}")
    for line in view.get("reason") or []:
        print(f"  reason: {line}")
    for line in (view.get("warnings") or [])[-5:]:
        print(f"  warning: {line}")


def cmd_status(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.launch status")
    ap.add_argument("run_dir")
    ap.add_argument("--wait", action="store_true", help="poll until the run ends")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(list(argv))
    t0 = time.monotonic()
    while True:
        st = read_status(args.run_dir)
        if st is None:
            print(f"no status.json in {args.run_dir}", file=sys.stderr)
            return EXIT_CODES["failed"]
        view = _status_view(st)
        state = st.get("state")
        stale = state not in TERMINAL and not view["other_host"] and not view["supervisor_alive"]
        final = state in TERMINAL or stale or not args.wait or (args.timeout and time.monotonic() - t0 > args.timeout)
        if final:
            if args.json:
                print(json.dumps(view, indent=2, default=str))
            else:
                _print_status(view)
            if stale:
                print("  the supervisor is not running: this status is stale "
                      "(python -m eval.launch stop RUN_DIR cleans up)")
                return EXIT_CODES["infra_error"]
            return EXIT_CODES.get(state, EXIT_RUNNING)
        time.sleep(args.interval)


def _stop_recorded_group(rec: dict) -> list:
    actions = []
    for sig, wait_s in ((signal.SIGINT, 60.0), (signal.SIGTERM, 15.0), (signal.SIGKILL, 5.0)):
        result = signal_own_group(int(rec["pgid"]), sig, rec.get("start_ticks"))
        actions.append(f"{sig.name}: {result}")
        if result != "sent":
            break
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and group_members(int(rec["pgid"])):
            time.sleep(0.5)
    return actions


def cmd_stop(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.launch stop")
    ap.add_argument("run_dir")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args(list(argv))
    st = read_status(args.run_dir)
    if st is None:
        print(f"no status.json in {args.run_dir}", file=sys.stderr)
        return EXIT_CODES["failed"]
    if st.get("state") in TERMINAL:
        print(f"run already {st['state']}")
        return 0
    if other_host(st):
        print(f"run is supervised on host {other_host(st)}: run `stop` there (nothing signalled)", file=sys.stderr)
        return EXIT_CODES["failed"]
    sup = st.get("supervisor") or {}
    sup_stat = proc_stat(sup["pid"]) if sup.get("pid") else None
    if sup_stat and sup_stat["uid"] == os.getuid() and process_alive(sup["pid"], sup.get("start_ticks")):
        os.kill(int(sup["pid"]), signal.SIGTERM)  # our own supervisor; it stops its children and finalizes
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.timeout:
            st = read_status(args.run_dir) or {}
            if st.get("state") in TERMINAL:
                print(f"run {st['state']}: {st.get('reason')}")
                return 0
            time.sleep(1.0)
        print("the supervisor did not finish in time", file=sys.stderr)
        return EXIT_CODES["infra_error"]
    for name in ("evaluator", "agent", "executor"):
        rec = (st.get("processes") or {}).get(name)
        if rec and rec.get("started_by_us") and not rec.get("ended_at"):
            rec["stop_actions"] = _stop_recorded_group(rec)
            rec["ended_at"] = now_iso()
    st.update(state="failed", reason=["stopped_by_user", "supervisor not running"], ended_at=now_iso(),
              updated_at=now_iso())
    write_json(os.path.join(args.run_dir, "status.json"), st)
    print("stopped the recorded process groups; state failed")
    return 0


def cmd_watch(argv: Sequence[str]) -> int:
    plan = load_plan(argv[0])
    return EXIT_CODES[Supervisor(plan, mode="detached").run()["state"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"status": cmd_status, "stop": cmd_stop, "_watch": cmd_watch}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    return cmd_launch(argv)


if __name__ == "__main__":
    sys.exit(main())
