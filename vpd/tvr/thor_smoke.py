"""TVR render smoke: AI2-THOR 5.0.0 CloudRendering (Vulkan) inside one sco job (WP-TVR, plan section 3.7).

Ported from the WP-TVR design draft (scratchpad vpd_tvr/design/thor_smoke.py); runs under OUT/envs/tvr python
3.10 with `-I -u`.  No model, no network: the build is pre-staged by stage_tvr_login.sh.

Phase A  raw Controller with TVRBench's eval kwargs (640x360, Ultra, AA 4, fov 60, grid 0.25, 45 deg, snapToGrid
         False), FloorPlan1, N steps cycling RotateRight / LookDown / LookUp / Teleport-to-reachable: start-up
         time, steps/s, frame checks, a PNG every 50 steps; then 50 RotateRight steps at 1280x720 (TVRBench
         ThorEnv's default resolution; reported only, so a failure there is recorded as hd_error and never
         discards the 640x360 counts).
Phase B  (unless --no-tasks) TVRBench's own ThorEnv on the first iTHOR and the first ProcTHOR task of
         data/tasks/eval.json: load scene / house, reset to the target and start poses, 10 grid moves, PNGs.
         Without usable ProcTHOR data (data/procthor-10k/train.jsonl.gz, else train.jsonl, as TVRBench's
         _load_procthor_house) the ProcTHOR row is recorded as skipped='no_procthor_data' (plan 3.7).  With
         --deadline-s, phase B is skipped (one row, skipped='time_budget') when less than PHASE_B_MIN_S is left,
         because it starts a second Controller.
--nproc K runs phase A in K spawned processes on the same card (start-up staggered by 8 s, like eval_va.py's
init_lock) and reports the aggregate steps/s.
--fail-fast (single process only) starts a watchdog thread while a Controller is starting or stepping: it polls
$HOME's Player.log (only what was written after the watch began, so an old crash log never counts) for Unity's
"Caught fatal signal" line, and the Unity player PID (a child of this process whose command line names an
.ai2thor/releases build).  A fatal-signal line, or a player seen and then gone for WATCH_EXIT_GRACE_S, writes
result.json with 'fatal' {reason, lines, pids, t_s} and raw.error, copies Player.log to Player_fatal.log and exits
3 at once, instead of blocking in ai2thor's server_start_timeout (300 s) or the caller's outer timeout.
(smoke_run1, step 6: Unity aborted at SelectPhysicalDevice(-1) with SIGABRT and the harness waited 240 s.)

Output: OUT/result.json, rewritten atomically (tmp + os.replace) after the start-up, every 50 steps, the end of
phase A, the HD block, every phase-B row and every finished --nproc worker, so a kill by the outer `timeout`
keeps everything measured so far.  Also PNGs and copies of Unity's Player.log (Player_raw / _tasks / _parallel).
Exit 0 if this run's own pass rule holds, else 1 (3 = --fail-fast fired).  The gate itself is decided by
verdict.py.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time

import numpy as np
from PIL import Image

W, H = 640, 360
KW = dict(width=W, height=H, gridSize=0.25, rotateStepDegrees=45, snapToGrid=False, fieldOfView=60,
          quality="Ultra", antiAliasing=4, renderDepthImage=False, renderInstanceSegmentation=False,
          makeAgentsVisible=False, server_timeout=300.0)
HD = (1280, 720)
HD_STEPS = 50
STAGGER_S = 8
CHECKPOINT_EVERY = 50
# Phase B starts a second Controller (ai2thor server_start_timeout = 300 s), loads FloorPlan407 and one ProcTHOR
# house (a scan of train.jsonl.gz) and runs 2 x 12 env calls: skip it when less than this is left of --deadline-s.
PHASE_B_MIN_S = 420.0
TASK_ACTIONS = ["MoveAhead", "RotateRight", "MoveAhead", "LookDown", "MoveLeft",
                "RotateLeft", "MoveBack", "LookUp", "MoveRight", "RotateRight"]
PLAYER_LOG = os.path.join(".config", "unity3d", "Allen Institute for Artificial Intelligence", "AI2-THOR",
                          "Player.log")
# --fail-fast watchdog
FATAL_MARK = "Caught fatal signal"
FATAL_CONTEXT_LINES = 12  # the fatal line and the backtrace lines after it, kept in result.json
WATCH_POLL_S = 1.0
WATCH_EXIT_GRACE_S = 5.0  # a vanished player: wait this long for Player.log to say why before firing
FAIL_FAST_RC = 3
UNITY_CMD_MARK = "/.ai2thor/releases/"


def log(*args):
    """Progress line on stdout, flushed at once (the job log must show how far a killed run got)."""
    print("[thor_smoke]", time.strftime("%H:%M:%S", time.gmtime()), *args, flush=True)


def write_json_atomic(path, obj):
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def frame_stats(f):
    if f is None:
        return None
    return {"shape": list(f.shape), "dtype": str(f.dtype), "mean": float(f.mean()), "std": float(f.std())}


def is_bad(f, shape=(H, W, 3)):
    """A bad frame: missing, wrong shape, or near-constant (std < 2, e.g. black or a cleared buffer)."""
    return f is None or tuple(f.shape) != tuple(shape) or float(f.std()) < 2.0


def make_controller(gpu, width=W, height=H):
    """CloudRendering Controller with TVRBench's eval kwargs; returns (controller, start-up seconds)."""
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    kw = dict(KW, scene="FloorPlan1", platform=CloudRendering, width=width, height=height)
    if gpu is not None:
        kw["gpu_device"] = gpu  # ai2thor then runs vulkaninfo + nvidia-smi -L and passes -force-device-index
    t0 = time.time()
    c = Controller(**kw)
    return c, time.time() - t0


def copy_player_log(out, name):
    """Copy Unity's Player.log (and Player-prev.log) from $HOME into out/<name>.log; returns the copies."""
    src = os.path.join(os.path.expanduser("~"), PLAYER_LOG)
    got = []
    for s, suffix in ((src, ""), (src.replace("Player.log", "Player-prev.log"), "_prev")):
        if os.path.isfile(s):
            d = os.path.join(out, f"{name}{suffix}.log")
            shutil.copyfile(s, d)
            got.append(d)
    return got


def player_log_path():
    return os.path.join(os.path.expanduser("~"), PLAYER_LOG)


def unity_children(ppid=None, proc="/proc"):
    """Sorted PIDs of the live (not zombie) children of `ppid` (default: this process) whose command line names an
    AI2-THOR build (UNITY_CMD_MARK).  ai2thor starts the player with subprocess.Popen, so it is a direct child."""
    ppid = os.getpid() if ppid is None else ppid
    try:
        names = os.listdir(proc)
    except OSError:
        return []
    out = []
    for n in names:
        if not n.isdigit():
            continue
        try:
            with open(os.path.join(proc, n, "stat")) as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[1]) != ppid or fields[0] in ("Z", "X"):
                continue
            with open(os.path.join(proc, n, "cmdline"), "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError, IndexError):
            continue
        if UNITY_CMD_MARK in cmd:
            out.append(int(n))
    return sorted(out)


class UnityWatch:
    """The --fail-fast watchdog (see the module doc).  `on_fatal(info)` runs in the watch thread, once.

    Only Player.log text written after the watch was created counts (a new inode = a rotated / new log is read from
    its start; the same inode from the size it had).  Both signals are acted on only while armed (arm() before a
    Controller starts, disarm() before it is stopped on purpose); text that arrives while disarmed is consumed.
    """

    def __init__(self, log_path, on_fatal, children=None, poll_s=None, grace_s=None, clock=time.time):
        self.log_path = log_path
        self.on_fatal = on_fatal
        self.children = children or unity_children
        self.poll_s = WATCH_POLL_S if poll_s is None else poll_s
        self.grace_s = WATCH_EXIT_GRACE_S if grace_s is None else grace_s
        self.clock = clock
        self.t0 = clock()
        st = self._stat()
        self._ino, self._pos = (st[0], st[1]) if st else (None, 0)
        self._text = ""
        self._arm_at = 0
        self._armed = False
        self._seen = set()
        self._gone_since = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.fired = None
        self._thread = threading.Thread(target=self._loop, name="unity-watch", daemon=True)

    def _stat(self):
        try:
            st = os.stat(self.log_path)
        except OSError:
            return None
        return st.st_ino, st.st_size

    def _consume(self):
        st = self._stat()
        if st is None:
            return
        ino, size = st
        if ino != self._ino:
            self._ino, self._pos = ino, 0
            self._text, self._arm_at = "", 0
        elif size < self._pos:  # truncated in place
            self._pos = 0
            self._text, self._arm_at = "", 0
        if size <= self._pos:
            return
        try:
            with open(self.log_path, "rb") as f:
                f.seek(self._pos)
                data = f.read(size - self._pos)
        except OSError:
            return
        self._pos += len(data)
        self._text += data.decode(errors="replace")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def arm(self):
        with self._lock:
            self._consume()
            self._arm_at = len(self._text)
            self._armed = True
            self._seen, self._gone_since = set(), None

    def disarm(self):
        with self._lock:
            self._armed = False

    def poll(self):
        """One check; returns the info dict when the watch fires (also stored in .fired), else None."""
        with self._lock:
            if self.fired is not None:
                return None
            self._consume()
            if not self._armed:
                return None
            new = self._text[self._arm_at:]
            info = None
            k = new.find(FATAL_MARK)
            if k >= 0:
                start = new.rfind("\n", 0, k) + 1
                lines = new[start:].splitlines()[:FATAL_CONTEXT_LINES]
                info = {"reason": "fatal_signal", "lines": [ln[:300] for ln in lines]}
            else:
                kids = set(self.children())
                self._seen |= kids
                if self._seen and not kids:
                    now = self.clock()
                    if self._gone_since is None:
                        self._gone_since = now
                    elif now - self._gone_since >= self.grace_s:
                        tail = [ln[:300] for ln in new.splitlines()[-FATAL_CONTEXT_LINES:]]
                        info = {"reason": "unity_exited", "lines": tail}
                else:
                    self._gone_since = None
            if info is None:
                return None
            info.update(pids=sorted(self._seen), t_s=round(self.clock() - self.t0, 2), log=self.log_path)
            self.fired = info
        self.on_fatal(info)
        return info

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            try:
                if self.poll() is not None:
                    return
            except Exception as e:  # noqa: BLE001  (a broken watch must never kill the run it watches)
                log(f"fail-fast watch error: {e!r}")


def raw_phase(out, steps, gpu, save=True, tag="raw", factory=None, progress=None, watch=None):
    """Phase A on FloorPlan1; `factory(gpu) -> (controller, startup_s)` is make_controller unless injected.

    `progress(snapshot)` is called with a copy of the result so far: phase 'started' (startup_s only),
    'stepping' (counters under 'progress'), 'steps_done' (the final 640x360 counts) and 'done'.  The gated keys
    steps / bad_frames / distinct_frames / action_success exist only from 'steps_done' on.
    `watch` (--fail-fast, a UnityWatch) is armed from the Controller start until just before c.stop().
    """
    def checkpoint():
        if progress is not None:
            progress(dict(res))

    if watch is not None:
        watch.arm()
    try:
        c, t_start = (factory or make_controller)(gpu)
    except BaseException:
        if watch is not None:
            watch.disarm()
        raise
    res = {"phase": "started", "startup_s": round(t_start, 2)}
    log(f"{tag}: Controller up after {t_start:.1f} s")
    checkpoint()
    try:
        ev = c.step(action="GetReachablePositions")
        reach = ev.metadata.get("actionReturn") or []
        rng = np.random.default_rng(0)
        acts = ["RotateRight", "LookDown", "LookUp", "TELEPORT"]
        ok = distinct = bad_frames = 0
        prev = c.last_event.frame.copy() if c.last_event.frame is not None else None
        t0 = time.time()
        for i in range(steps):
            a = acts[i % len(acts)]
            if a == "TELEPORT" and reach:
                p = reach[int(rng.integers(len(reach)))]
                ev = c.step(action="Teleport", position=p,
                            rotation={"x": 0, "y": float(45 * rng.integers(8)), "z": 0},
                            horizon=0.0, standing=True, forceAction=True)
            else:
                ev = c.step(action=a if a != "TELEPORT" else "RotateLeft")
            ok += bool(ev.metadata.get("lastActionSuccess"))
            f = ev.frame
            if is_bad(f):
                bad_frames += 1
            elif prev is None or not np.array_equal(f, prev):
                distinct += 1
            if f is not None:
                prev = f.copy()
            if save and i % 50 == 0 and f is not None:
                Image.fromarray(f).save(os.path.join(out, f"{tag}_{i:04d}.png"))
            if (i + 1) % CHECKPOINT_EVERY == 0 and i + 1 < steps:
                res.update(phase="stepping", progress={"steps_done": i + 1, "action_success": ok,
                                                       "distinct_frames": distinct, "bad_frames": bad_frames,
                                                       "wall_s": round(time.time() - t0, 2)})
                checkpoint()
        dt = max(time.time() - t0, 1e-9)
        res.pop("progress", None)
        res.update(phase="steps_done", steps=steps, wall_s=round(dt, 2), steps_per_s=round(steps / dt, 2),
                   action_success=ok, distinct_frames=distinct, bad_frames=bad_frames, n_reachable=len(reach),
                   last_frame=frame_stats(c.last_event.frame),
                   server_reported={k: c.last_event.metadata.get(k) for k in ("screenWidth", "screenHeight")})
        log(f"{tag}: {steps} steps in {dt:.1f} s, bad={bad_frames} distinct={distinct} success={ok}")
        checkpoint()
        if save:  # TVRBench ThorEnv default resolution: 50 rotation steps, reported only
            try:
                c.step(action="ChangeResolution", x=HD[0], y=HD[1])
                t1 = time.time()
                hd_bad = 0
                for _ in range(HD_STEPS):
                    ev = c.step(action="RotateRight")
                    hd_bad += is_bad(ev.frame, (HD[1], HD[0], 3))
                res["hd_1280x720_steps_per_s"] = round(HD_STEPS / max(time.time() - t1, 1e-9), 2)
                res["hd_bad_frames"] = hd_bad
                res["hd_frame"] = frame_stats(ev.frame)
                if ev.frame is not None:
                    Image.fromarray(ev.frame).save(os.path.join(out, f"{tag}_1280x720.png"))
            except Exception as e:  # noqa: BLE001
                res["hd_error"] = repr(e)
            log(f"{tag}: 1280x720 block {res.get('hd_error') or res.get('hd_1280x720_steps_per_s')}")
        res["phase"] = "done"
        checkpoint()
    finally:
        if watch is not None:
            watch.disarm()
        try:
            c.stop()
        except Exception as e:  # noqa: BLE001
            res["stop_error"] = repr(e)
    return res


def procthor_file(repo):
    """(path, None) for usable ProcTHOR data, else (None, reason).

    Same choice as TVRBench's _load_procthor_house: data/procthor-10k/train.jsonl.gz if it exists, else
    train.jsonl.  A git-lfs pointer or a non-gzip .gz is not data.
    """
    d = os.path.join(repo, "data", "procthor-10k")
    gz, plain = os.path.join(d, "train.jsonl.gz"), os.path.join(d, "train.jsonl")
    if os.path.isfile(gz):
        p = gz
    elif os.path.isfile(plain):
        p = plain
    else:
        return None, f"missing {gz} (and {plain})"
    with open(p, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        return None, f"git-lfs pointer, not data: {p}"
    if p == gz and head[:2] != b"\x1f\x8b":
        return None, f"not gzip: {p}"
    if p == plain and not head.lstrip().startswith(b"{"):
        return None, f"not jsonl: {p}"
    return p, None


def load_procthor_house(path, scene):
    """House dict of 'procthor_NNNNN' = line NNNNN of train.jsonl(.gz) (thor_view_env._load_procthor_house)."""
    idx = int(scene.split("_", 1)[1])
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for k, line in enumerate(f):
            if k == idx:
                return json.loads(line)
    raise ValueError(f"{scene}: line {idx} not in {path}")


def pick_tasks(tasks):
    """The first iTHOR and the first ProcTHOR task of eval.json, in file order."""
    return [next(t for t in tasks if t["dataset"] == "ithor"),
            next(t for t in tasks if t["scene"].startswith("procthor_"))]


def task_phase(out, repo, gpu, env_cls=None, rows=None, progress=None, watch=None):
    """Phase B with TVRBench's ThorEnv (imported from the pinned repo unless `env_cls` is injected).

    Rows are appended to `rows` (a new list unless given) and `progress()` is called after each one.  `watch`
    (--fail-fast) is armed from the ThorEnv start until just before env.close().
    """
    rows = [] if rows is None else rows
    if env_cls is None:
        sys.path.insert(0, repo)
        from tvrbench.env.thor_env import ThorEnv as env_cls  # noqa: N813
    with open(os.path.join(repo, "data", "tasks", "eval.json")) as f:
        tasks = json.load(f)
    picks = pick_tasks(tasks)
    pt_path, pt_why = procthor_file(repo)
    t0 = time.time()
    if watch is not None:
        watch.arm()
    try:
        env = env_cls(scene="FloorPlan1", width=W, height=H, gpu_device=gpu)
    except BaseException:
        if watch is not None:
            watch.disarm()
        raise
    log(f"tasks: ThorEnv up after {time.time() - t0:.1f} s")
    try:
        for t in picks:
            r = {"task_id": t["task_id"], "scene": t["scene"],
                 "dataset": "procthor" if t["scene"].startswith("procthor_") else t.get("dataset")}
            try:
                scene = t["scene"]
                if scene.startswith("procthor_"):
                    if pt_path is None:
                        r.update(ok=False, skipped="no_procthor_data", why=pt_why)
                        continue
                    scene = load_procthor_house(pt_path, scene)
                t0 = time.time()
                env.load_scene(scene)
                r["load_s"] = round(time.time() - t0, 2)
                tgt = env.reset(t["target"]["position"], t["target"]["rotation_y"], t["target"]["horizon"])
                Image.fromarray(tgt).save(os.path.join(out, f"{t['task_id']}_target.png"))
                cur = env.reset(t["start"]["position"], t["start"]["rotation_y"], t["start"]["horizon"])
                Image.fromarray(cur).save(os.path.join(out, f"{t['task_id']}_start.png"))
                succ = []
                t0 = time.time()
                for a in TASK_ACTIONS:
                    _, s = env.step(a)
                    succ.append(bool(s))
                r["step_s"] = round((time.time() - t0) / len(TASK_ACTIONS), 4)
                r["moves_ok"] = sum(succ)
                r["start_frame"], r["target_frame"] = frame_stats(cur), frame_stats(tgt)
                r["distinct_start_target"] = not np.array_equal(cur, tgt)
                r["ok"] = (not is_bad(cur)) and (not is_bad(tgt)) and r["distinct_start_target"]
            except Exception as e:  # noqa: BLE001
                r["ok"], r["error"] = False, repr(e)
            finally:  # also on a kill (BaseException): the row says which task was in progress
                if "ok" not in r:
                    r.update(ok=False, error="interrupted")
                rows.append(r)
                log(f"tasks: {r['task_id']} ok={r.get('ok')} {r.get('skipped') or r.get('error') or ''}")
                if progress is not None:
                    progress()
    finally:
        if watch is not None:
            watch.disarm()
        env.close()
    return rows


def _worker(args):
    """One --nproc process: phase A; an exception keeps the last checkpoint and adds 'error'."""
    i, out, steps, gpu, factory = args
    time.sleep(STAGGER_S * i)  # serialise Unity/Vulkan start-up (eval_va.py holds init_lock for the same reason)
    last = {}

    def keep(snapshot):
        last.clear()
        last.update(snapshot)

    try:
        return raw_phase(out, steps, gpu, save=(i == 0), tag=f"p{i}", factory=factory, progress=keep)
    except Exception as e:  # noqa: BLE001
        return dict(last, error=repr(e))


def _raw_pass(r, steps):
    return ("error" not in r and r.get("steps") == steps and r.get("bad_frames") == 0
            and (r.get("distinct_frames") or 0) >= 0.9 * steps and (r.get("action_success") or 0) >= 0.95 * steps)


def run(a, factory=None, env_cls=None, pool_factory=None, exit_fn=None):
    """All phases for parsed args `a`; returns the result dict (kept current in OUT/result.json).

    Test hooks: `factory` replaces make_controller (it must be picklable for the default spawn pool),
    `env_cls` replaces TVRBench's ThorEnv, `pool_factory(n)` replaces the spawn Pool (e.g. a ThreadPool),
    `exit_fn(rc)` replaces os._exit for --fail-fast (called from the watch thread).
    """
    t_run = time.time()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "result.json")
    gpu = None if a.no_gpu_device else a.gpu
    res = {"argv": sys.argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pass": False,
           "deadline_s": a.deadline_s,
           "env": {k: os.environ.get(k) for k in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES", "LD_LIBRARY_PATH", "HOME",
                                                  "CUDA_VISIBLE_DEVICES", "__EGL_VENDOR_LIBRARY_FILENAMES",
                                                  "LP_NUM_THREADS")}}
    save_lock = threading.RLock()  # the --fail-fast watch thread writes result.json too

    def save():
        with save_lock:
            res["elapsed_s"] = round(time.time() - t_run, 2)
            write_json_atomic(path, res)

    def on_fatal(info):
        """Watch thread: record why, keep the log, end the process now (the main thread is blocked in ai2thor)."""
        with save_lock:  # held until exit: the main thread can no longer overwrite result.json
            res["fatal"] = info
            res["pass"] = False
            msg = f"unity_{info['reason']}: {(info.get('lines') or [''])[0]}"
            raw = res.get("raw")
            if isinstance(raw, dict) and raw.get("phase") in ("steps_done", "done"):
                res.setdefault("tasks", []).append({"task_id": None, "dataset": None, "ok": False, "error": msg})
            else:
                res["raw"] = dict(raw or {}, error=msg)
            for attempt in (0, 1):
                try:
                    save()
                    break
                except Exception as e:  # noqa: BLE001  (res changed size under json.dump: retry once)
                    if attempt:
                        log(f"fail-fast: result.json not written: {e!r}")
                    time.sleep(0.1)
            try:
                copy_player_log(a.out, "Player_fatal")
            except OSError as e:
                log(f"fail-fast: Player.log not copied: {e!r}")
            log(f"FAIL FAST after {info['t_s']} s: {msg} (pids {info['pids']})")
            (exit_fn or os._exit)(FAIL_FAST_RC)

    watch = None
    if getattr(a, "fail_fast", False):
        if a.nproc > 1:
            res["fail_fast"] = "ignored: --nproc > 1 (one Player.log is shared by all players)"
        else:
            watch = UnityWatch(player_log_path(), on_fatal).start()
            res["fail_fast"] = {"log": watch.log_path, "poll_s": watch.poll_s, "grace_s": watch.grace_s}
    save()
    try:
        _run_phases(a, res, save, gpu, factory, env_cls, pool_factory, watch, t_run)
    finally:
        if watch is not None:
            watch.stop()
    save()
    print(json.dumps(res, indent=1, default=str)[:4000], flush=True)
    return res


def _run_phases(a, res, save, gpu, factory, env_cls, pool_factory, watch, t_run):
    if a.nproc > 1:
        rows = []
        res["procs"] = rows
        mk = pool_factory or mp.get_context("spawn").Pool
        with mk(a.nproc) as pool:
            for r in pool.imap(_worker, [(i, a.out, a.steps, gpu, factory) for i in range(a.nproc)]):
                rows.append(r)
                log(f"nproc: worker {len(rows) - 1} done {r.get('error') or r.get('steps_per_s')}")
                save()
        good = [r for r in rows if "error" not in r]
        res["agg_steps_per_s"] = round(sum(r.get("steps_per_s") or 0 for r in good), 2)
        # = verdict T8: every worker finished all steps with 0 bad frames
        res["pass"] = len(good) == a.nproc and all(r.get("bad_frames") == 0 and r.get("steps") == a.steps
                                                   for r in good)
        copy_player_log(a.out, "Player_parallel")
    else:
        def keep(snapshot):
            res["raw"] = snapshot
            save()

        try:
            res["raw"] = raw_phase(a.out, a.steps, gpu, factory=factory, progress=keep, watch=watch)
        except Exception as e:  # noqa: BLE001
            res["raw"] = dict(res.get("raw") or {})
            res["raw"].setdefault("error", repr(e))  # a --fail-fast error recorded first is kept
        save()
        copy_player_log(a.out, "Player_raw")
        r = res["raw"]
        res["raw_pass"] = _raw_pass(r, a.steps)
        phase_b = not a.no_tasks and a.repo and "error" not in r
        res["pass"] = res["raw_pass"] and not phase_b  # stays False (also on disk) while phase B is pending
        save()
        # phase B runs whenever phase A produced a working Controller, so a frame-count fail still gets task data
        if phase_b:
            left = None if a.deadline_s is None else a.deadline_s - (time.time() - t_run)
            if left is not None and left < PHASE_B_MIN_S:
                res["tasks"] = [{"task_id": None, "dataset": None, "ok": False, "skipped": "time_budget",
                                 "left_s": round(left, 1), "need_s": PHASE_B_MIN_S}]
                log(f"tasks: skipped, {left:.0f} s left < {PHASE_B_MIN_S:.0f} s")
            else:
                res["tasks"] = []
                try:
                    task_phase(a.out, a.repo, gpu, env_cls=env_cls, rows=res["tasks"], progress=save, watch=watch)
                except Exception as e:  # noqa: BLE001
                    res["tasks"].append({"task_id": None, "dataset": None, "ok": False, "error": repr(e)})
                copy_player_log(a.out, "Player_tasks")
            tested = [t for t in res["tasks"] if t.get("skipped") is None]
            res["tasks_ok"] = sum(bool(t.get("ok")) for t in tested)
            res["tasks_skipped"] = [t.get("task_id") for t in res["tasks"] if t.get("skipped")]
            res["pass"] = (res["raw_pass"] and bool(tested) and res["tasks_ok"] == len(tested)
                           and all(t.get("skipped") != "time_budget" for t in res["tasks"]))
    if res.get("fatal"):
        res["pass"] = False


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-gpu-device", action="store_true",
                    help="do not pass gpu_device (no -force-device-index; Unity 2020.3 then picks no CPU device, so "
                         "never use this for lavapipe: smoke_run1 aborted at SelectPhysicalDevice(-1))")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--no-tasks", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="wall budget of this run in s; phase B is skipped if less than PHASE_B_MIN_S is left")
    ap.add_argument("--fail-fast", action="store_true",
                    help="single process: exit 3 as soon as Player.log shows a fatal signal or the Unity player "
                         "dies (see the module doc)")
    return ap.parse_args(argv)


def main():
    res = run(parse_args())
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
