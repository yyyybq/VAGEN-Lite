"""V3-V8, V10 thor_diag: AI2-THOR 5.0.0 CloudRendering with per-frame RGB / depth / segmentation statistics.

JOB ONLY (starts Unity on the GPU, or on lavapipe for V10).  thor_smoke.py's Controller kwargs (gridSize 0.25,
rotateStepDegrees 45, snapToGrid False, fieldOfView 60, makeAgentsVisible False) with the quality, antiAliasing
('none' omits the key), resolution, renderDepthImage and renderInstanceSegmentation of the variant, and:
  --unity-arg=ARG        appended to the Unity command line through a patched Controller.unity_command (the
                         technique of TVRBench thor_env.py:36-66), e.g. -force-gfx-direct;
  --child-env KEY=VAL    set for the Unity process only, through a patched Controller._start_unity_thread (the
                         env dict ai2thor hands to Popen), e.g. LD_DEBUG, VK_LOADER_DEBUG; ai2thor's own
                         vulkaninfo / nvidia-smi calls do not see them;
  --gpu-device N         ai2thor's gpu_device (vulkaninfo + nvidia-smi -L -> -force-device-index), or
  --force-device-index K append -force-device-index K without gpu_device.
A watchdog thread (off with --no-fatal-watch) ends the run within ~1 s when Player.log shows "Caught fatal
signal" or Unity exits during start-up: it records the reason, kills Unity, writes result.json, os._exit(3).
Steps cycle ACTIONS; every frame (the start-up frame as index -1) goes to frames.jsonl; result.json is rewritten
after the start-up and every CHECKPOINT_EVERY steps, so an outer timeout keeps what was measured.
PASS = at least one step frame and >= PASS_FRAC of the step frames with RGB std >= STD_MIN.
Also: start-up time, ms per step, the Unity command line, the GPU libraries mapped into Unity (/proc/PID/maps)
after start-up and after the steps, an `nvidia-smi` snapshot after start-up, rgb.png / depth.png / seg.png of the
last frame, and copies of Player.log and ~/.ai2thor/log/unity.log (Unity's stderr: the Vulkan loader log).
Module import is stdlib only (numpy, PIL and ai2thor are imported in run()).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time

from . import common

PLAYER_LOG = os.path.join(".config", "unity3d", "Allen Institute for Artificial Intelligence", "AI2-THOR",
                          "Player.log")
UNITY_LOG = os.path.join(".ai2thor", "log", "unity.log")
MAPPING = os.path.join(".ai2thor", "cuda-vulkan-mapping.json")
ACTIONS = ["RotateRight", "LookDown", "LookUp", "MoveAhead", "RotateLeft"]
PASS_FRAC = 0.9
STD_MIN = 2.0
DEPTH_STD_MIN = 1e-3  # metres
CHECKPOINT_EVERY = 10
FATAL = "Caught fatal signal"
BASE_KW = dict(gridSize=0.25, rotateStepDegrees=45, snapToGrid=False, fieldOfView=60, makeAgentsVisible=False)

# V7: the login node's /etc/vulkan/implicit_layer.d/nvidia_layers.json (driver 595.71.05, read 2026-11-09), only
# its VK_LAYER_NV_optimus entry, with an absolute library_path (the job image's copy of the file is 0 bytes).
OPTIMUS_LAYER = {
    "file_format_version": "1.0.1",
    "layers": [{
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": None,
        "api_version": "1.4.329",
        "implementation_version": "1",
        "description": "NVIDIA Optimus layer",
        "functions": {"vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
                      "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"},
        "enable_environment": {"__NV_PRIME_RENDER_OFFLOAD": "1"},
        "disable_environment": {"DISABLE_LAYER_NV_OPTIMUS_1": ""},
    }],
}


def write_layers_json(path: str, lib: str) -> dict:
    if not os.path.isabs(lib):
        raise ValueError(f"library_path must be absolute: {lib}")
    doc = json.loads(json.dumps(OPTIMUS_LAYER))
    doc["layers"][0]["library_path"] = lib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")
    return doc


def add_arguments(p) -> None:
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="thor_diag")
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--quality", default="Ultra")
    p.add_argument("--aa", default="4", help="antiAliasing initialisation value; 'none' leaves the key out")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--depth", action="store_true", help="renderDepthImage=True")
    p.add_argument("--seg", action="store_true", help="renderInstanceSegmentation=True")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu-device", type=int)
    g.add_argument("--force-device-index", type=int)
    p.add_argument("--unity-arg", action="append", default=[], help="give it as --unity-arg=-flag")
    p.add_argument("--child-env", action="append", default=[], help="KEY=VAL, for the Unity process only")
    p.add_argument("--startup-timeout", type=float, default=180.0, help="ai2thor server_start_timeout")
    p.add_argument("--server-timeout", type=float, default=100.0, help="ai2thor server_timeout (per step)")
    p.add_argument("--deadline-s", type=float, help="stop stepping after this many seconds of the run")
    p.add_argument("--no-fatal-watch", action="store_true")
    p.add_argument("--no-smi", action="store_true", help="no nvidia-smi snapshot after start-up")


def parse_child_env(items) -> dict:
    env = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--child-env wants KEY=VAL, got {it!r}")
        k, v = it.split("=", 1)
        env[k] = v
    return env


def controller_kwargs(a, platform) -> dict:
    kw = dict(BASE_KW, scene=a.scene, width=a.width, height=a.height, quality=a.quality,
              renderDepthImage=bool(a.depth), renderInstanceSegmentation=bool(a.seg),
              server_timeout=a.server_timeout, server_start_timeout=a.startup_timeout)
    if platform is not None:
        kw["platform"] = platform
    if str(a.aa).lower() != "none":
        kw["antiAliasing"] = int(a.aa) if str(a.aa).isdigit() else a.aa
    if a.gpu_device is not None:
        kw["gpu_device"] = a.gpu_device
    return kw


def patch_controller(cls, state: dict, unity_args=(), child_env=None, force_device_index=None):
    """Wrap cls.unity_command (extra Unity args) and cls._start_unity_thread (Unity-only env); returns a restore()."""
    orig_cmd, orig_start = cls.unity_command, cls._start_unity_thread
    extra = list(unity_args or [])
    cenv = dict(child_env or {})

    def unity_command(self, width, height, headless):
        cmd = list(orig_cmd(self, width, height, headless))
        if force_device_index is not None and "-force-device-index" not in cmd:
            cmd += ["-force-device-index", str(force_device_index)]
        cmd += extra
        state["unity_command"] = cmd
        return cmd

    def _start_unity_thread(self, env, *args, **kw):
        env.update(cenv)
        state["controller"] = self
        return orig_start(self, env, *args, **kw)

    cls.unity_command = unity_command
    cls._start_unity_thread = _start_unity_thread

    def restore():
        cls.unity_command, cls._start_unity_thread = orig_cmd, orig_start
    return restore


def find_unity_pids(home: str) -> list:
    """PIDs whose command line runs a player under HOME/.ai2thor/releases/ (this variant's HOME only)."""
    needle = os.path.join(home, ".ai2thor", "releases") + os.sep
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")
        except OSError:
            continue
        if cmd and cmd[0].decode(errors="replace").startswith(needle):
            pids.append(int(d))
    return pids


def unity_proc(state):
    c = state.get("controller")
    srv = getattr(c, "server", None)
    return getattr(srv, "unity_proc", None)


class Watchdog(threading.Thread):
    """Fail fast: 'Caught fatal signal' in Player.log, or Unity gone during start-up -> result.json, os._exit(3)."""

    def __init__(self, home, state, res, save, out, poll=1.0, exit_fn=os._exit):
        super().__init__(daemon=True)
        self.log = os.path.join(home, PLAYER_LOG)
        self.home, self.state, self.res, self.save, self.out = home, state, res, save, out
        self.poll, self.exit_fn = poll, exit_fn
        self.stop_evt = threading.Event()
        self.offset = 0

    def check(self):
        try:
            with open(self.log, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
                self.offset += len(chunk)
        except OSError:
            chunk = b""
        if FATAL.encode() in chunk:
            return f"'{FATAL}' in Player.log"
        p = unity_proc(self.state)
        if self.state.get("phase") == "startup" and p is not None and p.poll() is not None:
            return f"Unity exited with rc={p.returncode} during start-up"
        return None

    def fire(self, reason):
        self.res["fatal"] = reason
        self.res["stage"] = "fatal"
        self.res["notes"].append(f"watchdog: {reason} after {time.time() - self.state.get('t_run', time.time()):.1f} s")
        copy_logs(self.home, self.out)
        p = unity_proc(self.state)
        pids = find_unity_pids(self.home)
        if p is not None and p.poll() is None:
            pids.append(p.pid)
        for pid in sorted(set(pids)):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        self.res["metrics"]["killed_unity_pids"] = sorted(set(pids))
        self.save()
        self.exit_fn(3)

    def run(self):
        while not self.stop_evt.wait(self.poll):
            reason = self.check()
            if reason:
                self.fire(reason)
                return


def copy_logs(home: str, out: str) -> list:
    got = []
    for src, name in ((os.path.join(home, PLAYER_LOG), "Player.log"),
                      (os.path.join(home, PLAYER_LOG).replace("Player.log", "Player-prev.log"), "Player-prev.log"),
                      (os.path.join(home, UNITY_LOG), "unity.log")):
        if os.path.isfile(src):
            try:
                shutil.copyfile(src, os.path.join(out, name))
                got.append(name)
            except OSError:
                pass
    return got


def rgb_stats(f, np):
    if f is None:
        return None
    a = np.asarray(f)
    flat = a.reshape(-1, a.shape[-1])[:, :3].astype(np.uint32) if a.ndim == 3 else a.reshape(-1, 1)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2] if flat.shape[1] >= 3 else flat[:, 0]
    return {"shape": list(a.shape), "min": int(a.min()), "max": int(a.max()), "mean": round(float(a.mean()), 3),
            "std": round(float(a.std()), 3), "n_colors": int(np.unique(packed).size)}


def depth_stats(d, np):
    if d is None:
        return None
    a = np.asarray(d, dtype=np.float64)
    fin = np.isfinite(a)
    v = a[fin]
    if v.size == 0:
        return {"shape": list(a.shape), "n_nonfinite": int(a.size), "std": None}
    return {"shape": list(a.shape), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 5),
            "frac_zero": round(float((v == 0).mean()), 4), "n_nonfinite": int((~fin).sum())}


def frame_record(ev, i, np) -> dict:
    md = getattr(ev, "metadata", {}) or {}
    rec = {"i": i, "action": (md.get("lastAction") if i >= 0 else "init"),
           "success": bool(md.get("lastActionSuccess")), "rgb": rgb_stats(getattr(ev, "frame", None), np),
           "depth": depth_stats(getattr(ev, "depth_frame", None), np),
           "seg": rgb_stats(getattr(ev, "instance_segmentation_frame", None), np)}
    err = md.get("errorMessage")
    if err:
        rec["error"] = str(err)[:200]
    return rec


def summarise_frames(frames, want_depth: bool, want_seg: bool) -> dict:
    steps = [r for r in frames if r["i"] >= 0]
    n = len(steps)

    def frac(k):
        return round(k / n, 4) if n else None

    rgb_ok = sum(1 for r in steps if r["rgb"] and r["rgb"]["std"] >= STD_MIN)
    s = {"n_step_frames": n, "rgb_std_ge2": rgb_ok, "frac_rgb_std_ge2": frac(rgb_ok),
         "rgb_all_zero_frames": sum(1 for r in steps if r["rgb"] and r["rgb"]["max"] == 0),
         "rgb_missing_frames": sum(1 for r in steps if not r["rgb"]),
         "rgb_max_overall": max((r["rgb"]["max"] for r in steps if r["rgb"]), default=None),
         "rgb_distinct_frame_stats": len({(r["rgb"]["mean"], r["rgb"]["std"]) for r in steps if r["rgb"]}),
         "action_success": sum(1 for r in steps if r["success"])}
    init = [r for r in frames if r["i"] < 0]
    if init and init[0]["rgb"]:
        s["init_rgb"] = init[0]["rgb"]
    if want_depth:
        dv = sum(1 for r in steps if r["depth"] and (r["depth"].get("std") or 0) > DEPTH_STD_MIN)
        s.update(depth_varies_frames=dv, frac_depth_varies=frac(dv),
                 depth_missing_frames=sum(1 for r in steps if not r["depth"]),
                 depth_max_overall=max((r["depth"].get("max") or 0 for r in steps if r["depth"]), default=None))
    if want_seg:
        sv = sum(1 for r in steps if r["seg"] and r["seg"]["n_colors"] >= 2)
        s.update(seg_varies_frames=sv, frac_seg_varies=frac(sv),
                 seg_missing_frames=sum(1 for r in steps if not r["seg"]),
                 seg_n_colors_max=max((r["seg"]["n_colors"] for r in steps if r["seg"]), default=None))
    return s


def save_pngs(ev, out, np) -> dict:
    saved = {}
    try:
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    f = getattr(ev, "frame", None)
    if f is not None:
        Image.fromarray(np.asarray(f)[..., :3].astype(np.uint8)).save(os.path.join(out, "rgb.png"))
        saved["rgb"] = "rgb.png"
    d = getattr(ev, "depth_frame", None)
    if d is not None:
        a = np.nan_to_num(np.asarray(d, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(a.max()) if a.size else 0.0
        g = (255.0 * a / mx) if mx > 0 else a * 0
        Image.fromarray(g.clip(0, 255).astype(np.uint8)).save(os.path.join(out, "depth.png"))
        saved["depth"] = "depth.png"
    s = getattr(ev, "instance_segmentation_frame", None)
    if s is not None:
        Image.fromarray(np.asarray(s)[..., :3].astype(np.uint8)).save(os.path.join(out, "seg.png"))
        saved["seg"] = "seg.png"
    return saved


def run(a, controller_cls=None, platform=None, np=None, smi=None) -> dict:
    """All of one variant; returns the result (also in OUT/result.json).  Test hooks: `controller_cls` (+
    `platform`) replace ai2thor's Controller / CloudRendering, `np` numpy, `smi(out)` the nvidia-smi snapshot."""
    t_run = time.time()
    out = a.out
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "result.json")
    home = os.path.expanduser("~")
    res = common.new_result(a.variant)
    m = res["metrics"]
    child_env = parse_child_env(a.child_env)
    res["config"] = {"scene": a.scene, "width": a.width, "height": a.height, "quality": a.quality, "aa": a.aa,
                     "steps": a.steps, "depth": bool(a.depth), "seg": bool(a.seg), "gpu_device": a.gpu_device,
                     "force_device_index": a.force_device_index, "unity_args": list(a.unity_arg),
                     "child_env": child_env, "startup_timeout": a.startup_timeout, "deadline_s": a.deadline_s}
    res["env"] = {k: os.environ.get(k) for k in (
        "HOME", "LD_LIBRARY_PATH", "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "__EGL_VENDOR_LIBRARY_FILENAMES",
        "XDG_DATA_HOME", "__NV_PRIME_RENDER_OFFLOAD", "CUDA_VISIBLE_DEVICES", "LP_NUM_THREADS",
        "NVIDIA_DRIVER_CAPABILITIES", "PATH")}
    lock = threading.Lock()

    def save():
        with lock:
            res["elapsed_s"] = round(time.time() - t_run, 2)
            common.write_json_atomic(path, res)

    state = {"phase": "import", "t_run": t_run}
    save()
    if np is None:
        import numpy as np  # noqa: N813
    if controller_cls is None:
        from ai2thor.controller import Controller as controller_cls  # noqa: N813
        from ai2thor.platform import CloudRendering as platform
    restore = patch_controller(controller_cls, state, a.unity_arg, child_env, a.force_device_index)
    wd = None
    if not a.no_fatal_watch:
        wd = Watchdog(home, state, res, save, out)
        wd.start()
    kw = controller_kwargs(a, platform)
    res["config"]["controller_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None
                                              else getattr(v, "__name__", str(v))) for k, v in kw.items()}
    if a.deadline_s is not None:
        kw["server_start_timeout"] = max(10.0, min(float(kw["server_start_timeout"]), a.deadline_s - 15.0))
    state["phase"] = res["stage"] = "startup"
    save()
    frames, c = [], None
    fpath = os.path.join(out, "frames.jsonl")
    open(fpath, "w").close()

    def add(rec):
        frames.append(rec)
        with open(fpath, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def conclude():
        """Frame summary, pass, the mapping file and log copies; saved before Controller.stop (it can hang)."""
        m["frames"] = summarise_frames(frames, a.depth, a.seg)
        fr = m["frames"]
        res["pass"] = bool(fr["n_step_frames"] >= 1 and (fr["frac_rgb_std_ge2"] or 0) >= PASS_FRAC)
        short = f"only {fr['n_step_frames']} of {a.steps} steps"
        if fr["n_step_frames"] < a.steps and short not in res["notes"]:
            res["notes"].append(short)
        mp = os.path.join(home, MAPPING)
        m["cuda_vulkan_mapping"] = common.read_json(mp) if os.path.isfile(mp) else None
        m["logs_copied"] = copy_logs(home, out)
        save()

    t0 = time.time()
    try:
        c = controller_cls(**kw)
    except Exception as e:  # noqa: BLE001
        state["phase"] = "start_failed"
        m["startup_failed_after_s"] = round(time.time() - t0, 2)
        m["unity_command"] = state.get("unity_command")
        res["error"] = repr(e)
        res["notes"].append(f"Controller start failed after {m['startup_failed_after_s']} s: {e!r}"[:600])
    if c is not None:
        m["startup_s"] = round(time.time() - t0, 2)
        state["phase"] = res["stage"] = "stepping"
        m["unity_command"] = state.get("unity_command")
        pid = getattr(c, "unity_pid", None)
        m["unity_pid"] = pid
        if pid:
            m["unity_maps_after_startup"] = common.mapped_libs(pid)
        if not a.no_smi:
            (smi or nvidia_smi_snapshot)(out)
        try:
            add(frame_record(c.last_event, -1, np))
            save()
            ts = time.time()
            last = c.last_event
            for i in range(a.steps):
                if a.deadline_s is not None and time.time() - t_run > a.deadline_s:
                    res["notes"].append(f"deadline {a.deadline_s} s reached after {i} steps")
                    break
                last = c.step(action=ACTIONS[i % len(ACTIONS)])
                add(frame_record(last, i, np))
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    m["frames"] = summarise_frames(frames, a.depth, a.seg)
                    save()
            n = len(frames) - 1
            dt = max(time.time() - ts, 1e-9)
            if n > 0:
                m["steps_per_s"] = round(n / dt, 3)
                m["ms_per_step"] = round(1000.0 * dt / n, 2)
            m["pngs"] = save_pngs(last, out, np)
        except Exception as e:  # noqa: BLE001
            res["error"] = repr(e)
            res["notes"].append(f"stepping failed after {len(frames) - 1} steps: {e!r}"[:600])
        finally:
            state["phase"] = "stopping"
            if wd is not None:
                wd.stop_evt.set()
            if pid:
                m["unity_maps_after_steps"] = common.mapped_libs(pid)
            res["stage"] = "stopping"
            conclude()
            try:
                c.stop()
            except Exception as e:  # noqa: BLE001
                res["notes"].append(f"stop: {e!r}"[:300])
    if wd is not None:
        wd.stop_evt.set()
    restore()
    res["stage"] = "done"
    conclude()
    return res


def nvidia_smi_snapshot(out: str) -> None:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        text = r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        text = repr(e)
    with open(os.path.join(out, "nvidia_smi_mid.txt"), "w") as f:
        f.write(text)


def main_args(a) -> int:
    res = run(a)
    print(json.dumps({k: res.get(k) for k in ("variant", "pass", "stage", "notes")}, default=str)[:3000],
          flush=True)
    print(json.dumps(res["metrics"].get("frames"), default=str)[:3000], flush=True)
    return 0 if res["pass"] else 1
