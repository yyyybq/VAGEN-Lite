#!/usr/bin/env python3
"""Seeded, parallel RoboCasa365 closed-loop evaluation of a StarVLA policy served over websocket.

Run with the RoboCasa simulator Python (probe_spatial/robocasa365/policies/diffusion_policy/.venv) and the
PYTHONPATH of examples/evaluate/robocasa/run_starvla_mobile.sh; policy servers run separately
(run_starvla_mobile.sh server, one per GPU).

    python downstream/eval/closed_loop.py --ckpt CKPT --servers 127.0.0.1:5678 --tasks NavigateKitchen \
        --seeds 1000-1007 --workers 12 --out OUT
    python downstream/eval/closed_loop.py --ckpt CKPT --servers ... --eval-plan downstream/eval/eval_d6.json ...

Per episode: gym.make('robocasa/<task>', split=target, seed=s) and env.reset(seed=s) (the seed reaches the env's
RNG, which draws the kitchen layout / style and object placement), the task-specific horizon of RoboCasa 1.0.1
(downstream/demos/target50_tasks.json) as the step limit, one policy call per `execute_steps` actions (= the
model's 16-step chunk), stop on success.  Episodes run longest horizon first.  Each worker process keeps one
websocket client and first makes RoboCasa's counter-region order process-independent (stable_counter_regions; off
with --no-stable-regions).  Writes OUT/episodes.jsonl (one line per episode, including a fingerprint of the initial
state and its joint positions), OUT/summary.json, and per episode an example strip OUT/frames/<task>_s<seed>.jpg
(8 evenly spaced 128 px frames of the left agent-view camera; off with --no-frames).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
TASKS = WORKTREE / "downstream/demos/target50_tasks.json"
_STATE = {}


def parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def horizons(path: Path = TASKS) -> dict:
    return {t["task"]: int(t["horizon"]) for t in json.loads(path.read_text())["tasks"]}


def plan(tasks, seeds, horizon_of=None):
    horizon_of = horizon_of or horizons()
    return [{"task": t, "seed": int(s), "max_steps": horizon_of[t]} for t in tasks for s in seeds]


def eval_plan_episodes(path: Path, tasks_path: Path = TASKS):
    """(episodes, plan dict) of a fixed evaluation plan (downstream/eval/eval_d6.json): every task of the plan's
    groups with the plan's `seeds`; the spatial tasks (plan `spatial_tasks` file, key "spatial") with `seeds_spatial`
    when the plan has it.  A plan with "only_spatial" keeps only the spatial tasks (X3: downstream/x/x3_plan.json);
    one with "tasks_file" takes its tasks (rows with task, group, horizon) from that file instead of target50 (X5)."""
    p = json.loads(Path(path).read_text())
    task_rows = json.loads((WORKTREE / p["tasks_file"] if p.get("tasks_file") else Path(tasks_path)).read_text())["tasks"]
    horizon_of = {t["task"]: int(t["horizon"]) for t in task_rows}
    spatial = set()
    if p.get("seeds_spatial") or p.get("only_spatial"):
        spatial = set(json.loads((WORKTREE / p["spatial_tasks"]).read_text())["spatial"])
    episodes = []
    for t in (row["task"] for row in task_rows if row["group"] in p["groups"]):
        if p.get("only_spatial") and t not in spatial:
            continue
        seeds = p["seeds_spatial"] if t in spatial and p.get("seeds_spatial") else p["seeds"]
        episodes += plan([t], parse_seeds(seeds), horizon_of)
    return episodes, p


EXCLUDED = Path(__file__).resolve().parent / "excluded_episodes.json"


def excluded_episodes(path: Path = EXCLUDED) -> dict:
    """The fixed exclusion list (the user's decision 1 of 2026-10-23): {"keys": {(task, seed)}, "sha256", "path"};
    empty when the file is missing."""
    path = Path(path)
    if not path.is_file():
        return {"keys": set(), "sha256": None, "path": str(path)}
    raw = path.read_bytes()
    keys = {(t, int(s)) for t, s in json.loads(raw)["episodes"]}
    return {"keys": keys, "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path)}


def apply_exclusions(episodes, keys) -> list:
    """The episodes not in the exclusion list, in their order (applied after sharding, so a shard keeps its episodes)."""
    return [e for e in episodes if (e["task"], int(e["seed"])) not in keys]


def state_fingerprint(env, obs) -> dict:
    """What the seed must fix: kitchen layout / style, all joint positions (robot and free objects), the instruction
    and the first camera images."""
    import numpy as np

    inner = getattr(env, "unwrapped", env)
    kitchen = getattr(inner, "env", inner)
    qpos = np.asarray(kitchen.sim.data.qpos, dtype=np.float64)
    images = hashlib.sha256()
    for key in sorted(k for k in obs if k.startswith("video.")):
        images.update(np.ascontiguousarray(obs[key]).tobytes())
    return {"layout_id": _as_int(getattr(kitchen, "layout_id", None)), "style_id": _as_int(getattr(kitchen, "style_id", None)),
            "qpos_sha": hashlib.sha256(np.round(qpos, 6).tobytes()).hexdigest()[:16], "nq": int(qpos.size),
            "images_sha": images.hexdigest()[:16], "instruction": obs.get("annotation.human.task_description")}


def _ordered_set(items=()):
    return dict.fromkeys(items)  # list(...) of it: the distinct items in first-seen order


def with_ordered_set(func):
    """A copy of func in which the name `set` is an insertion-ordered de-duplication, so list(set(xs)) keeps the
    first-seen order of xs.  Nothing else changes (same code, defaults, closure; globals copied)."""
    import types

    new = types.FunctionType(func.__code__, dict(func.__globals__, set=_ordered_set), func.__name__,
                             func.__defaults__, func.__closure__)
    new.__kwdefaults__ = func.__kwdefaults__
    new.__qualname__, new.__doc__, new.__module__ = func.__qualname__, func.__doc__, func.__module__
    new.downstream_ordered_set = True
    return new


def stable_counter_regions() -> bool:
    """Make RoboCasa 1.0.1's counter reset regions come in the same order in every process (this process only;
    RoboCasa's files are not changed).  Counter.get_reset_regions (robocasa/models/fixtures/counter.py:813) de-duplicates the
    candidate counter-top geoms with list(set(geoms)); the geoms are XML elements hashed by memory address, so the
    region order differs between processes, and Fixture.sample_reset_region (fixture.py:383) then picks a region with
    self.rng.choice(regions): the same seed puts a fixture (and the robot in front of it) on a different counter.
    Returns True if the patch was applied now, False if it already was."""
    from robocasa.models.fixtures.counter import Counter

    if getattr(Counter.get_reset_regions, "downstream_ordered_set", False):
        return False
    Counter.get_reset_regions = with_ordered_set(Counter.get_reset_regions)
    return True


def initial_qpos(env) -> list:
    """All joint positions after reset, rounded to 1e-6, so runs can be compared by size of difference, not only by
    hash (fresh processes can differ by ~1e-6 in passive joints: downstream/eval/reset_repeat.py)."""
    inner = getattr(env, "unwrapped", env)
    return [round(float(x), 6) for x in getattr(inner, "env", inner).sim.data.qpos]


def longest_first(episodes):
    """Run order: longest task horizon first (stable), so the long episodes do not form the tail of the run."""
    return sorted(episodes, key=lambda e: -e["max_steps"])


def shard_of(episodes, shard: str):
    """Shard "I/K" (1 <= I <= K) of an episode list already in run order: every K-th episode starting at the I-th,
    so each shard gets its share of the long-horizon tasks.  The K shards partition the list."""
    i, k = (int(x) for x in shard.split("/"))
    if not 1 <= i <= k:
        raise ValueError(f"shard {shard}: need 1 <= I <= K")
    return episodes[i - 1::k]


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None if value is None else str(value)


def connect(ckpt: str, server: str, starvla_root: str):
    """A websocket client to one policy server, checked against the checkpoint's contract."""
    sys.path.insert(0, str(WORKTREE))
    sys.path.insert(0, starvla_root)
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from vagen.envs.robocasa.starvla_contract import PolicyContract, checkpoint_config, verify_server

    config, _ = checkpoint_config(ckpt)
    contract = PolicyContract.from_config(config)
    host, port = server.split(":")
    policy = WebsocketClientPolicy(host, int(port))
    verify_server(policy.get_server_metadata(), ckpt, contract)
    return contract, policy


def _init_worker(ckpt: str, servers, starvla_root: str, stable_regions: bool = True, inflight: str = None):
    """Never raises: a failing initializer makes multiprocessing respawn workers forever, so the error is kept and
    every episode of this worker reports it instead."""
    worker = mp.current_process()._identity[0] if mp.current_process()._identity else 1
    server = servers[(worker - 1) % len(servers)]
    _STATE.update(server=server, worker=worker, inflight=inflight)
    try:
        if stable_regions:
            import robocasa.wrappers.gym_wrapper  # noqa: F401  (imports RoboCasa before patching it)
            stable_counter_regions()
        contract, policy = connect(ckpt, server, starvla_root)
        _STATE.update(contract=contract, policy=policy)
    except Exception as exc:
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


FRAME_CAMERA = "video.robot0_agentview_left"
FRAME_SIZE = 128
FRAMES_PER_EPISODE = 8


def pick_indices(n: int, k: int) -> list:
    """k evenly spaced indices into n items, first and last included (all of them if n <= k)."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


class FrameRecorder:
    """One small frame of FRAME_CAMERA per policy call; at the end FRAMES_PER_EPISODE evenly spaced ones (the first
    and the last included) are written side by side as one JPEG, with the control step under each frame and a
    green (success) or red (failure) bar at the bottom: the example strips of the result report."""

    def __init__(self, path: Path, size: int = FRAME_SIZE, keep: int = FRAMES_PER_EPISODE, camera: str = FRAME_CAMERA):
        self.path, self.size, self.keep, self.camera, self.frames = Path(path), size, keep, camera, []

    def add(self, step: int, obs) -> None:
        import numpy as np
        from PIL import Image

        img = obs.get(self.camera) if hasattr(obs, "get") else None
        if img is None:
            return
        arr = np.asarray(img)
        arr = arr[-1] if arr.ndim == 4 else arr
        small = Image.fromarray(arr.astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        self.frames.append((int(step), np.asarray(small)))

    def save(self, success: bool) -> dict:
        from PIL import Image, ImageDraw

        chosen = [self.frames[i] for i in pick_indices(len(self.frames), self.keep)]
        if not chosen:
            return {"frames_error": f"no {self.camera} in the observations"}
        label, bar = 14, 4
        sheet = Image.new("RGB", (self.size * len(chosen), self.size + label + bar), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, (step, frame) in enumerate(chosen):
            sheet.paste(Image.fromarray(frame), (i * self.size, 0))
            draw.text((i * self.size + 3, self.size + 1), f"t={step}", fill=(40, 40, 40))
        draw.rectangle([0, self.size + label, sheet.width, sheet.height], fill=(34, 139, 34) if success else (200, 40, 40))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(self.path, quality=85)
        return {"frames": f"{self.path.parent.name}/{self.path.name}", "frame_steps": [s for s, _ in chosen]}


def _safe(fn):
    """fn, but a failure only switches it off (recording frames must never stop an episode)."""
    state = {"on": True}

    def call(*args):
        if state["on"]:
            try:
                fn(*args)
            except Exception:
                state["on"] = False
    return call


def _record(fn, *args):
    """fn(*args), or {"error": ...}: recording must never stop or change an episode."""
    try:
        return fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def _load_x3_targets():
    import importlib.util

    spec = importlib.util.spec_from_file_location("x3_targets", WORKTREE / "downstream/x/x3_targets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EndState:
    """X3: the read-only record of a spatial task's candidate targets (downstream/x/x3_targets.py) right after reset
    and after the episode's last step, kept in the episode row as end_state = {"start", "end"}.  Never raises: a
    failure becomes {"error": ...} in the record."""

    def __init__(self, task: str, env, load=_load_x3_targets):
        self.task, self.module = task, _record(load)
        self.start = self._call("start", env)

    def _call(self, name: str, env):
        if isinstance(self.module, dict):  # the recorder did not load: its error is the record
            return self.module
        return _record(getattr(self.module, name), self.task, env)

    def finish(self, env) -> dict:
        return {"start": self.start, "end": self._call("end", env)}


def run_episode(spec: dict) -> dict:
    """One episode; while it runs, INFLIGHT/<pid>.json names it, so the main process can tell which episode a worker
    that died was running (multiprocessing loses that task and would otherwise wait for it forever)."""
    marker = Path(_STATE["inflight"]) / f"{os.getpid()}.json" if _STATE.get("inflight") else None
    if marker is not None:
        marker.write_text(json.dumps({"task": spec["task"], "seed": spec["seed"], "started": time.time(),
                                      "max_steps": spec.get("max_steps")}))
    _STATE["marker"], _STATE["beat"] = marker, 0.0
    try:
        return _run_episode(spec)
    finally:
        _STATE["marker"] = None
        if marker is not None:
            try:
                marker.unlink()
            except OSError:
                pass


def _heartbeat(every: float = 15.0) -> None:
    """Touch the in-flight marker (at most every `every` seconds): the main process reads its age as the time since
    this worker last made progress (a policy call)."""
    marker, now = _STATE.get("marker"), time.time()
    if marker is not None and now - _STATE.get("beat", 0.0) >= every:
        try:
            os.utime(marker)
        except OSError:
            pass
        _STATE["beat"] = now


def _run_episode(spec: dict) -> dict:
    out = dict(spec, worker=_STATE.get("worker"), server=_STATE.get("server"), pid=os.getpid())
    if "init_error" in _STATE:
        return dict(out, status="error", error="worker init: " + _STATE["init_error"], wall_seconds=0.0)
    import gymnasium as gym
    import numpy as np
    import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<task>)
    from vagen.envs.robocasa.starvla_contract import gym_action

    contract, policy = _STATE["contract"], _STATE["policy"]
    env = None
    probe = Probe(PROBE_DIR, spec["task"], spec["seed"]) if PROBE_DIR else None
    started = time.monotonic()
    try:
        if probe:
            probe.arm()
        env = gym.make(f"robocasa/{spec['task']}", enable_render=True, split=spec["split"], seed=spec["seed"])
        obs, _ = env.reset(seed=spec["seed"])
        out["reset_seconds"] = round(time.monotonic() - started, 2)
        if probe:
            probe.log("reset", 0, out["reset_seconds"], env)
        out["init"] = state_fingerprint(env, obs)
        out["init_qpos"] = initial_qpos(env)
        end_state = EndState(spec["task"], env) if spec.get("end_state") else None
        recorder = FrameRecorder(Path(spec["frames_dir"]) / f"{spec['task']}_s{spec['seed']}.jpg") \
            if spec.get("frames_dir") else None
        record = _safe(recorder.add) if recorder else (lambda *a: None)
        record(0, obs)
        steps, calls, success, latencies = 0, 0, False, []
        terminated = truncated = False
        while steps < spec["max_steps"] and not (success or terminated or truncated):
            _heartbeat()
            if probe:
                probe.arm()
            t = time.monotonic()
            response = policy.predict_action({"examples": [contract.example(obs)], "do_sample": False})
            latencies.append(time.monotonic() - t)
            actions = contract.actions(response["data"]["actions"])
            calls += 1
            if probe:
                probe.log("policy", calls, latencies[-1])
            for vector in actions[:min(spec["execute_steps"], spec["max_steps"] - steps)]:
                if probe:
                    probe.arm()
                    t = time.monotonic()
                obs, reward, terminated, truncated, info = env.step(gym_action(vector, env.action_space))
                steps += 1
                if probe:
                    probe.log("step", steps, time.monotonic() - t, env)
                success = success or bool(info.get("success", False))
                if success or terminated or truncated:
                    break
            record(steps, obs)
        out.update(success=bool(success), steps=steps, policy_calls=calls, terminated=bool(terminated),
                   truncated=bool(truncated), step_limit=steps >= spec["max_steps"],
                   inference_mean_seconds=round(float(np.mean(latencies)), 4) if latencies else None,
                   status="complete")
        if end_state is not None:  # after the last step: the episode is over
            out["end_state"] = end_state.finish(env)
        if recorder:
            try:
                out.update(recorder.save(success))
            except Exception as exc:  # an example strip is never worth an episode
                out["frames_error"] = f"{type(exc).__name__}: {exc}"[:300]
    except Exception as exc:
        out.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500], traceback=traceback.format_exc()[-2000:])
    finally:
        if env is not None:
            env.close()
        if probe:
            probe.close()
    out["wall_seconds"] = round(time.monotonic() - started, 2)
    return out


def pid_alive(pid: int) -> bool:
    """False for a process that is gone or a zombie (dead, not yet reaped)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def hang_limit(max_steps) -> float:
    """Seconds after which a running episode counts as hung: 10 minutes of reset plus 3 s per control step, about 4x
    the slowest measured (0.81 s per step with 96 processes on one machine, 2026-10-22).  CLOSED_LOOP_HANG_LIMIT_SECONDS
    replaces it (2026-10-30, only for the diagnostics run of an episode that hangs)."""
    if os.environ.get("CLOSED_LOOP_HANG_LIMIT_SECONDS"):
        return float(os.environ["CLOSED_LOOP_HANG_LIMIT_SECONDS"])
    return 600.0 + 3.0 * float(max_steps or 0)


STALL_SECONDS = float(os.environ.get("CLOSED_LOOP_STALL_SECONDS", "600"))  # no progress (policy call) for this
# long: hung.  A policy call comes every 16 control steps (13 s at 0.81 s per step); the longest gap is the scene build
# and reset at the start, up to about 2 minutes measured.  The variable only changes for the diagnostics run.

PROBE_DIR = os.environ.get("CLOSED_LOOP_PROBE_DIR")  # diagnostics of a hanging episode (2026-10-30), default off
PROBE_DUMP_SECONDS = float(os.environ.get("CLOSED_LOOP_STALL_DUMP", "120"))


class Probe:
    """Per-episode diagnostics, only with CLOSED_LOOP_PROBE_DIR set (the user's request of 2026-10-30: why OpenDrawer
    seed 1012 keeps hanging its simulator process).  Every policy call and every simulator step is timed and written
    with the simulator's contact count, largest joint speed and simulated time to PROBE_DIR/<task>_s<seed>_p<pid>.csv;
    a reset, call or step that takes longer than CLOSED_LOOP_STALL_DUMP seconds (120) dumps every thread's Python stack
    to PROBE_DIR/stacks_<pid>.log (faulthandler, re-armed before each one, so the dump names where it is stuck)."""

    def __init__(self, root, task: str, seed: int):
        import faulthandler

        self.fh = faulthandler
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.rows = open(root / f"{task}_s{seed}_p{os.getpid()}.csv", "w")
        self.rows.write("utc_seconds,phase,index,seconds,ncon,max_abs_qvel,sim_time\n")
        self.stacks = open(root / f"stacks_{os.getpid()}.log", "a")
        self.stacks.write(f"=== {task} seed {seed} pid {os.getpid()} started {time.time():.0f}\n")
        self.stacks.flush()

    def arm(self) -> None:
        self.fh.dump_traceback_later(PROBE_DUMP_SECONDS, repeat=True, file=self.stacks)

    def log(self, phase: str, index: int, seconds: float, env=None) -> None:
        ncon = qvel = sim_time = ""
        if env is not None:
            try:
                import numpy as np

                inner = getattr(env, "unwrapped", env)
                data = getattr(inner, "env", inner).sim.data
                ncon, qvel, sim_time = int(data.ncon), round(float(np.abs(np.asarray(data.qvel)).max()), 4), \
                    round(float(data.time), 4)
            except Exception:  # diagnostics never stop an episode
                pass
        self.rows.write(f"{time.time():.3f},{phase},{index},{seconds:.4f},{ncon},{qvel},{sim_time}\n")
        self.rows.flush()

    def close(self) -> None:
        self.fh.cancel_dump_traceback_later()
        self.rows.close()
        self.stacks.close()


def hung(info: dict, marker_mtime: float, now: float, stall: float = STALL_SECONDS) -> bool:
    """A live worker is hung when its episode made no progress (the marker is touched at every policy call) for
    `stall` seconds, or ran longer than hang_limit() in all."""
    started = float(info.get("started", now))
    return now - max(started, marker_mtime) > stall or now - started > hang_limit(info.get("max_steps"))


def _read_int(path: str):
    try:
        with open(path) as f:
            text = f.read().strip()
        return None if text in ("", "max") else int(text.split()[0])
    except (OSError, ValueError):
        return None


def _status_kb(pid: int, field: str):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class MemoryMonitor:
    """Memory of the evaluation job (the user's decision 3 of 2026-10-22): the container's cgroup limit, current use
    and peak (all its processes: policy servers, simulator workers, this process), its out-of-memory kills, and each
    simulator worker's peak resident memory (VmHWM).  sample() appends a row to OUT/memory.csv; result() goes into
    summary.json["memory"]."""

    V2 = {"limit": "/sys/fs/cgroup/memory.max", "current": "/sys/fs/cgroup/memory.current",
          "peak": "/sys/fs/cgroup/memory.peak", "events": "/sys/fs/cgroup/memory.events"}
    V1 = {"limit": "/sys/fs/cgroup/memory/memory.limit_in_bytes", "current": "/sys/fs/cgroup/memory/memory.usage_in_bytes",
          "peak": "/sys/fs/cgroup/memory/memory.max_usage_in_bytes", "events": "/sys/fs/cgroup/memory/memory.oom_control"}

    def __init__(self, csv_path: Path, pids=lambda: []):
        self.csv, self.pids = Path(csv_path), pids
        self.files = self.V2 if os.path.exists(self.V2["current"]) else self.V1
        self.hwm, self.max_current, self.samples = {}, 0, 0
        if not self.csv.exists():
            self.csv.write_text("utc,cgroup_current,cgroup_peak,workers,workers_rss_sum,worker_hwm_max\n")

    def oom_kills(self):
        try:
            with open(self.files["events"]) as f:
                for line in f:
                    if line.startswith("oom_kill "):
                        return int(line.split()[1])
        except (OSError, ValueError):
            pass
        return None

    def sample(self) -> None:
        current, peak = _read_int(self.files["current"]), _read_int(self.files["peak"])
        self.max_current = max(self.max_current, current or 0)
        rss = 0
        pids = list(self.pids())
        for pid in pids:
            h, r = _status_kb(pid, "VmHWM"), _status_kb(pid, "VmRSS")
            if h is not None:
                self.hwm[pid] = max(self.hwm.get(pid, 0), h)
            rss += r or 0
        self.samples += 1
        with open(self.csv, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},{current or ''},{peak or ''},{len(pids)},"
                    f"{rss},{max(self.hwm.values(), default='')}\n")

    def result(self) -> dict:
        limit = _read_int(self.files["limit"])
        total = None
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
        capacity = limit if limit and (total is None or limit < total) else total
        peak = _read_int(self.files["peak"]) or self.max_current or None
        hwm = list(self.hwm.values())
        return {"capacity_bytes": capacity, "cgroup_limit_bytes": limit, "meminfo_total_bytes": total,
                "peak_bytes": peak, "peak_fraction": round(peak / capacity, 4) if peak and capacity else None,
                "peak_source": "cgroup memory.peak" if _read_int(self.files["peak"]) else "largest sampled current",
                "worker_peak_max_bytes": max(hwm, default=None),
                "worker_peak_mean_bytes": round(sum(hwm) / len(hwm)) if hwm else None, "workers_seen": len(hwm),
                "oom_kills": self.oom_kills(), "samples": self.samples}


def run_all(pool, episodes, write, inflight: Path, retries: int = 2, poll: float = 30.0, alive=pid_alive,
            now=time.time, kill=os.kill, sleep=time.sleep, run=None, monitor=None) -> dict:
    """Run every episode on `pool` (apply_async), passing each finished row to write().  A worker process that dies
    (killed for memory, a simulator crash) loses its episode: multiprocessing replaces the worker but never reports
    the task.  Every `poll` seconds the INFLIGHT markers are read; an episode whose worker is dead is run again (at
    most `retries` times, then recorded as an error), and a worker that hung() (no policy call for STALL_SECONDS, or
    longer than hang_limit() in all) is killed first.  Returns counts of the retries, kills and episodes given up."""
    run = run or run_episode
    key = lambda e: (e["task"], e["seed"])  # noqa: E731
    pending, tries, stats = {}, {}, {"retried": 0, "killed_hung": 0, "given_up": 0}

    def submit(e):
        pending[key(e)] = (e, pool.apply_async(run, (e,)))

    for e in episodes:
        submit(e)
    while pending:
        for k, (e, res) in list(pending.items()):
            if res.ready():
                del pending[k]
                try:
                    row = res.get()
                except Exception as exc:  # run_episode catches its own errors; this is the transport failing
                    row = dict(e, status="error", error=f"{type(exc).__name__}: {exc}"[:500], wall_seconds=0.0)
                write(row)
        for m in sorted(Path(inflight).glob("*.json")):
            try:
                pid, info, mtime = int(m.stem), json.loads(m.read_text()), m.stat().st_mtime
            except (OSError, ValueError):
                continue
            if alive(pid):
                if hung(info, mtime, now()):
                    try:
                        kill(pid, 9)
                        stats["killed_hung"] += 1
                    except OSError:
                        pass
                continue
            try:
                m.unlink()
            except OSError:
                pass
            k = (info["task"], info["seed"])
            if k not in pending or pending[k][1].ready():
                continue
            e = pending.pop(k)[0]
            tries[k] = tries.get(k, 0) + 1
            if tries[k] <= retries:
                stats["retried"] += 1
                submit(dict(e, attempt=tries[k]))
            else:
                stats["given_up"] += 1
                write(dict(e, status="error", error=f"its worker process died {tries[k]} times while running it",
                           wall_seconds=0.0))
        if monitor is not None:
            try:
                monitor.sample()
            except Exception:  # measuring memory must never stop an evaluation
                pass
        if pending:
            sleep(poll)
    return stats


def completed_rows(path: Path) -> list:
    """The complete episodes of an earlier, interrupted run of the same plan (OUT/episodes.jsonl)."""
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "complete":
                out.append(r)
    return out


def summarize(rows, workers: int, wall: float) -> dict:
    done = [r for r in rows if r.get("status") == "complete"]
    per_task = {}
    for r in done:
        t = per_task.setdefault(r["task"], {"n": 0, "successes": 0})
        t["n"] += 1
        t["successes"] += int(r["success"])
    for t in per_task.values():
        t["success_rate"] = t["successes"] / t["n"]
    mean = lambda key: round(sum(r[key] for r in done) / len(done), 3) if done else None  # noqa: E731
    return {"episodes": len(rows), "complete": len(done), "errors": len(rows) - len(done),
            "successes": sum(int(r["success"]) for r in done), "per_task": per_task, "workers": workers,
            "wall_seconds": round(wall, 1), "episode_wall_mean_seconds": mean("wall_seconds"),
            "reset_mean_seconds": mean("reset_seconds"), "steps_mean": mean("steps"),
            "seconds_per_step_mean": round(sum(r["wall_seconds"] - r["reset_seconds"] for r in done)
                                           / max(1, sum(r["steps"] for r in done)), 4) if done else None,
            "episodes_per_hour": round(len(done) / wall * 3600, 1) if wall > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--servers", required=True, help="host:port[,host:port...]; workers are spread over them")
    ap.add_argument("--eval-plan", type=Path, help="a fixed plan file (downstream/eval/eval_d6.json): tasks, seeds, "
                                                   "split, execute steps")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--seeds", help="e.g. 1000-1019 or 1000,1003")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="target")
    ap.add_argument("--execute-steps", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--starvla-root", default=str(WORKTREE / "third_party/starVLA"))
    ap.add_argument("--no-stable-regions", dest="stable_regions", action="store_false",
                    help="leave RoboCasa's counter-region order as it is (differs between processes)")
    ap.add_argument("--no-frames", dest="frames", action="store_false",
                    help="do not write the per-episode example strip OUT/frames/<task>_s<seed>.jpg")
    ap.add_argument("--shard", default=None, help="I/K: run only shard I of K of the plan (one-card evaluation jobs)")
    ap.add_argument("--exclusions", type=Path, default=EXCLUDED,
                    help="the fixed exclusion list applied to an --eval-plan (empty string: none)")
    args = ap.parse_args()
    eval_plan = None
    if args.eval_plan:
        if args.tasks or args.seeds:
            ap.error("--eval-plan replaces --tasks / --seeds")
        episodes, eval_plan = eval_plan_episodes(args.eval_plan)
        args.split, args.execute_steps = eval_plan["split"], int(eval_plan["execute_steps"])
        if not eval_plan.get("stable_counter_regions", True):
            args.stable_regions = False
    elif args.tasks and args.seeds:
        episodes = plan(args.tasks, parse_seeds(args.seeds))
    else:
        ap.error("give --eval-plan, or --tasks and --seeds")
    episodes = longest_first(episodes)
    if args.shard:
        episodes = shard_of(episodes, args.shard)
    excl = excluded_episodes(args.exclusions) if (eval_plan and args.exclusions) else {"keys": set(), "sha256": None}
    episodes = apply_exclusions(episodes, excl["keys"])
    for e in episodes:
        e.update(split=args.split, execute_steps=args.execute_steps)
        if args.frames:
            e["frames_dir"] = str(args.out / "frames")
        if eval_plan and eval_plan.get("end_state"):
            e["end_state"] = True
    record = {"ckpt": args.ckpt, "servers": args.servers, "workers": args.workers,
              "stable_counter_regions": args.stable_regions, "shard": args.shard,
              "eval_plan": eval_plan and {"path": str(args.eval_plan), "name": eval_plan["name"]},
              "excluded": {"sha256": excl["sha256"], "episodes": sorted([t, s] for t, s in excl["keys"])},
              "episodes": episodes}
    resumed = []
    if (args.out / "plan.json").is_file():  # an earlier run of this evaluation stopped: run only what is missing
        old = json.loads((args.out / "plan.json").read_text())
        old_keys = apply_exclusions(old.get("episodes", []), excl["keys"])  # an older, longer plan minus the list
        same = (old.get("ckpt") == args.ckpt and old.get("shard") == args.shard
                and old.get("stable_counter_regions") == args.stable_regions
                and [(e["task"], e["seed"]) for e in old_keys] == [(e["task"], e["seed"]) for e in episodes])
        if not same:
            print(f"[closed_loop] {args.out} holds a different evaluation: not touching it")
            return 2
        resumed = [r for r in completed_rows(args.out / "episodes.jsonl")
                   if (r["task"], int(r["seed"])) not in excl["keys"]]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        (args.out / f"resume_{stamp}.json").write_text(json.dumps(
            {"resumed_complete": len(resumed), "servers": args.servers, "workers": args.workers,
             "excluded": record["excluded"]}, indent=1) + "\n")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "plan.json").write_text(json.dumps(record, indent=1) + "\n")
    done_keys = {(r["task"], r["seed"]) for r in resumed}
    todo = [e for e in episodes if (e["task"], e["seed"]) not in done_keys]
    for server in args.servers.split(","):  # fail fast, before any simulator starts
        try:
            connect(args.ckpt, server, args.starvla_root)
        except Exception as exc:
            (args.out / "summary.json").write_text(json.dumps({"error": f"server {server}: {type(exc).__name__}: {exc}"},
                                                              indent=1) + "\n")
            print(f"[closed_loop] server {server} failed the contract check: {type(exc).__name__}: {exc}")
            return 2
    rows = list(resumed)
    t0 = time.monotonic()
    inflight = args.out / "inflight"
    inflight.mkdir(exist_ok=True)
    for stale in inflight.glob("*.json"):
        stale.unlink()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.ckpt, args.servers.split(","), args.starvla_root, args.stable_regions,
                            str(inflight))) as pool:
        monitor = MemoryMonitor(args.out / "memory.csv",
                                pids=lambda: [p.pid for p in getattr(pool, "_pool", []) if p.pid])
        with open(args.out / "episodes.jsonl", "a") as f:
            def write(row):
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            pool_stats = run_all(pool, todo, write, inflight, monitor=monitor)
        try:
            monitor.sample()
            memory = monitor.result()
        except Exception as exc:
            memory = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    summary = dict(summarize(rows, args.workers, time.monotonic() - t0), stable_counter_regions=args.stable_regions,
                   eval_plan=eval_plan["name"] if eval_plan else None, shard=args.shard, resumed_complete=len(resumed),
                   pool=pool_stats, memory=memory, excluded=record["excluded"])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "complete", "successes", "wall_seconds",
                                              "episode_wall_mean_seconds", "episodes_per_hour")}))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
