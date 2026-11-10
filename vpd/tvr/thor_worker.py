"""TVR THOR worker processes for the demo replay and the eval harness (CONTRACT_wave2 5.2.1; DECISIONS G3, G10).

Worker side (inside a job only: it starts Unity; TVR env python 3.10, ai2thor 5.0.0):
    TVR_PY -B -s -u -m vpd.tvr.thor_worker worker --wid N --repo REPO --houses DIR --procthor-dir REPO/data/procthor-10k
        --cmd-fd A --reply-fd B
One TVRBench ThorViewEnv per process, built by vpd.tvr.lvp (the single render setup: lavapipe profile, pinned
badge font, official eval settings).  JSON line protocol exactly as probe_lvp: stdin /dev/null, stdout/stderr to
the worker log, reply 0 = ready, one reply per command, envelope {id, ok, error, tb}; the parent side reuses
probe_lvp.SubprocessHandle / LocalHandle (imported, not copied).

NAMING (normative): `env` = the ThorViewEnv; `thor = env._env` = the ThorEnv INSTANCE that owns
`_reachable_positions` / `_reachable_set`.  Every op works on `thor`, never on the thor_env module.

Ops (every reply also carries "t_s"):
  load        {"scene", "force"} -> {"scene", "load_s", "loaded"}
  reach_task  {"task"} -> {"scene", "loaded", "load_s", "call_pose", "positions", "n", "digest", "reach_s"}:
              env._sync_reset(task) (official), then the cache cleared and GetReachablePositions called at the
              task's start pose (the frozen per-task start-pose set R(task) of contract 4.3).
  reset       {"task", "reach": [positions] | null, "images": {"inline", "cur", "tgt"}} -> {"scene", "loaded",
              "load_s", "obs_str", "valid", "state", "cur", "tgt", "reach_live", "reach_used", "reach_equal"}:
              _sync_reset, live set at the start pose (recorded), INJECTION of the frozen set into thor's cache
              (verified from thor._get_reachable_set(); "injection_failed" otherwise), then env._build_obs() so
              the valid list and text follow the set in force.
  step        {"action", "images": {"inline", "cur", "skip"}} -> {"obs_str", "valid", "state", "cur", "info",
              "done"}: env._sync_step(f"Action: {action}") (official; "InvalidAction" spends a step, no env step).
  quit        -> {"event": "bye"}
IMG = {"sha256", "bytes", "png_b64" (inline), "path" (when a path was given; written atomically)}: PIL
img.save(buf, format="PNG") with default parameters, as eval_va._img_to_data_url.  No frame is ever cached or
reused: every IMG is the frame rendered for that step (frames are not bit-reproducible, F4).

Parent side (py3.12 stdlib): ThorPool (serialized start-ups, one retry, restart, fast close), reap_unity.
Python 3.10 and 3.12; stdlib only at import time (numpy / PIL are imported by TVRBench inside the worker).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from vpd import paths
from vpd.tvr import kin_env, lvp, pilot_cfg, probe_lvp

SCHEMA = "vpd.tvr.thor_worker/1"
TIMEOUTS = {"load": 240.0, "reach_task": 240.0, "reset": 240.0, "step": 120.0}
QUIT_TIMEOUT_S = 20.0
STARTUP_TIMEOUT_S = 150.0
# the TVRBench code files whose sha256 every ready reply reports (pilot_cfg tvrbench.files_sha256, code files only)
CODE_FILES_TVRBENCH = tuple(k for k in pilot_cfg.FROZEN["tvrbench"]["files_sha256"] if k.endswith(".py"))


class PoolError(RuntimeError):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(p) -> Optional[str]:
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def digest_cells(cells) -> str:
    """Contract 4.3 digest of a set of (x, z) cells (x, z rounded by kin_env.r2)."""
    norm = sorted({(kin_env.r2(x), kin_env.r2(z)) for (x, z) in cells})
    return sha256_bytes("\n".join(f"{x:.2f},{z:.2f}" for (x, z) in norm).encode("utf-8"))


def norm_cells(cells) -> frozenset:
    return frozenset((kin_env.r2(x), kin_env.r2(z)) for (x, z) in cells)


def write_bytes_atomic(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def png_img(pil, *, inline: bool, path=None) -> Dict[str, Any]:
    """IMG of one PIL image: PNG bytes with PIL's default parameters (eval_va._img_to_data_url)."""
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()
    out: Dict[str, Any] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    if inline:
        out["png_b64"] = base64.b64encode(data).decode("ascii")
    if path:
        write_bytes_atomic(path, data)
        out["path"] = str(path)
    return out


def tvrbench_files_sha256(repo) -> Dict[str, Optional[str]]:
    return {rel: sha256_path(Path(repo) / rel) for rel in CODE_FILES_TVRBENCH}


def ai2thor_version() -> Optional[str]:
    try:
        from importlib import metadata

        return metadata.version("ai2thor")
    except Exception:  # noqa: BLE001
        m = sys.modules.get("ai2thor")
        return getattr(m, "__version__", None)


# --- the worker (runs in its own process, inside the job) ---------------------------------------------------------
class Worker:
    """One THOR worker: TVRBench's ThorViewEnv built by lvp.make_view_env (official eval settings).  `tv` is the
    thor_view_env module (tests pass the real TVRBench module on a fake ai2thor).  handle(cmd) runs one op."""

    def __init__(self, tv, *, wid: int = 0, houses_dir=None, procthor_dir=lvp.PROCTHOR_DIR, font=None,
                 repo=lvp.TVR_REPO, max_steps: int = lvp.ENV_MAX_STEPS, environ: Mapping[str, str] = os.environ):
        self.tv, self.wid = tv, wid
        self.houses_dir = None if houses_dir is None else str(houses_dir)
        self.procthor_dir, self.font, self.repo = procthor_dir, font, repo
        self.max_steps, self.environ = max_steps, environ
        self.env = None

    # -- life cycle
    def start(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.env = lvp.make_view_env(self.tv, procthor_dir=self.procthor_dir, max_steps=self.max_steps,
                                     environ=self.environ)
        dt = time.perf_counter() - t0
        thor = self.env._env
        return {"event": "ready", "schema": SCHEMA, "startup_s": round(dt, 3), "pid": os.getpid(), "wid": self.wid,
                "lp": self.environ.get("LP_NUM_THREADS"), "env": lvp.env_summary(self.environ),
                "profile": self.environ.get("VPD_LVP_PROFILE"), "font": self.font,
                "settings": lvp.controller_settings(thor),
                "system_prompt_sha256": sha256_bytes(self.tv.SYSTEM_PROMPT.encode("utf-8")),
                "tvrbench_files_sha256": tvrbench_files_sha256(self.repo), "python": sys.version,
                "ai2thor": ai2thor_version(), "unity": probe_lvp.proc_info(probe_lvp.unity_pid(thor)),
                "scene": self.env._scene}

    def close(self) -> None:
        if self.env is not None:
            self.env._sync_close()
            self.env = None

    def handle(self, cmd: Mapping[str, Any]) -> Dict[str, Any]:
        fn = getattr(self, "op_" + str(cmd.get("op")), None)
        if fn is None:
            raise ValueError(f"unknown op {cmd.get('op')!r}")
        t0 = time.perf_counter()
        out = fn(cmd)
        out["t_s"] = round(time.perf_counter() - t0, 4)
        return out

    @property
    def thor(self):
        return self.env._env

    # -- ops
    def op_load(self, cmd):
        scene = cmd["scene"]
        if not cmd.get("force") and self.env._scene == scene:
            return {"scene": scene, "load_s": 0.0, "loaded": False}
        s = lvp.load_scene(self.env, scene, self.houses_dir)
        return {"scene": self.env._scene, "load_s": round(s, 3), "loaded": True}

    def _official_reset(self, task) -> Dict[str, Any]:
        env = self.env
        lvp.resolve_scene(env, task["scene"], self.houses_dir)  # ProcTHOR cache prime only
        loaded = env._scene != task["scene"]
        t0 = time.perf_counter()
        env._sync_reset(task)  # official: loads if needed, renders TARGET, teleports to the start
        dt = time.perf_counter() - t0
        return {"scene": env._scene, "loaded": loaded, "load_s": round(dt, 3) if loaded else 0.0,
                "reset_s": round(dt, 3)}

    def op_reach_task(self, cmd):
        task = cmd["task"]
        out = self._official_reset(task)
        thor = self.thor
        thor._reachable_positions = None  # the ThorEnv INSTANCE cache: force a fresh GetReachablePositions
        thor._reachable_set = None
        out["call_pose"] = thor.get_state()
        t0 = time.perf_counter()
        positions = list(thor.get_reachable_positions())
        out["reach_s"] = round(time.perf_counter() - t0, 4)
        rs = kin_env.ReachSet.from_positions(positions)
        out.update(positions=positions, n=rs.n, digest=rs.digest)
        return out

    def _img(self, pil, spec: Mapping[str, Any], key: str) -> Dict[str, Any]:
        return png_img(pil, inline=bool(spec.get("inline")), path=spec.get(key))

    def op_reset(self, cmd):
        task = cmd["task"]
        spec = cmd.get("images") or {}
        out = self._official_reset(task)
        env, thor = self.env, self.thor
        # live set at the start pose (recorded, not used when a frozen set is given)
        thor._reachable_positions = None
        thor._reachable_set = None
        live = list(thor.get_reachable_positions())
        live_rs = kin_env.ReachSet.from_positions(live)
        reach = cmd.get("reach")
        if reach is not None:
            want = kin_env.ReachSet.from_positions(reach)
            thor._reachable_positions = list(reach)
            thor._reachable_set = None
            used = norm_cells(thor._get_reachable_set())  # what ThorEnv now holds, never the input list
            if used != frozenset(want.cells):
                raise RuntimeError("injection_failed: ThorEnv's reachable set after injection has "
                                   f"{len(used)} cells (digest {digest_cells(used)[:12]}), the frozen set "
                                   f"{want.n} (digest {want.digest[:12]})")
            source = "frozen"
            reach_equal = frozenset(live_rs.cells) == frozenset(want.cells)
        else:
            used = norm_cells(thor._get_reachable_set())
            source = "live"
            reach_equal = True
        obs = env._build_obs()  # valid list and text from the set in force; burns CURRENT again
        cur = obs["multi_modal_input"][self.tv.IMAGE_PH][0]
        tgt = env._target_img
        out.update(obs_str=obs["obs_str"], valid=list(thor.get_valid_actions(include_stop=True)),
                   state=thor.get_state(), cur=self._img(cur, spec, "cur"), tgt=self._img(tgt, spec, "tgt"),
                   reach_live={"n": live_rs.n, "digest": live_rs.digest},
                   reach_used={"n": len(used), "digest": digest_cells(used), "source": source},
                   reach_equal=reach_equal)
        return out

    def op_step(self, cmd):
        action = str(cmd["action"])
        spec = cmd.get("images") or {}
        env, thor = self.env, self.thor
        obs, _reward, done, info = env._sync_step(f"Action: {action}")
        cur = None
        if not spec.get("skip") and action != "Stop":
            cur = self._img(obs["multi_modal_input"][self.tv.IMAGE_PH][0], spec, "cur")
        return {"obs_str": obs["obs_str"], "valid": list(thor.get_valid_actions(include_stop=True)),
                "state": thor.get_state(), "cur": cur,
                "info": {"action": action, "action_executed": bool(info.get("action_executed")),
                         "action_success": info.get("action_success"), "success": info.get("success"),
                         "stopped": bool(info.get("stopped", False)), "truncated": bool(info.get("truncated", False)),
                         "format_strict": info.get("format_strict")},
                "done": bool(done)}


def worker_main(a) -> int:
    reply = os.fdopen(a.reply_fd, "w", buffering=1)
    cmds = os.fdopen(a.cmd_fd, "r")

    def send(obj):
        reply.write(json.dumps(probe_lvp.jsafe(obj)) + "\n")
        reply.flush()

    try:
        _, tv = lvp.import_tvrbench(a.repo)
        font = lvp.install_font_rule(tv)  # allow_missing False: a missing / other font refuses
        w = Worker(tv, wid=a.wid, houses_dir=a.houses, procthor_dir=a.procthor_dir, font=font, repo=a.repo)
        send(dict(w.start(), id=0, ok=True))
    except BaseException as e:  # noqa: BLE001
        send({"id": 0, "ok": False, "error": repr(e), "tb": traceback.format_exc()[-3000:]})
        return 1
    rc = 0
    for line in cmds:
        try:
            cmd = json.loads(line)
        except ValueError:
            send({"id": None, "ok": False, "error": "bad command line"})
            continue
        if cmd.get("op") == "quit":
            try:
                w.close()
                send({"id": cmd.get("id"), "ok": True, "event": "bye"})
            except Exception as e:  # noqa: BLE001
                send({"id": cmd.get("id"), "ok": False, "error": repr(e)})
                rc = 1
            break
        send(probe_lvp.dispatch(w, cmd))
    else:
        w.close()
    return rc


# --- parent side --------------------------------------------------------------------------------------------------
class ThorPool:
    """n_envs THOR workers at lp_threads lavapipe threads each, lavapipe profile `profile`.

    local_factory(wid) -> a thor_worker.Worker (tests: TVRBench on the fake ai2thor); ThorPool calls its start()
    and wraps it as probe_lvp.LocalHandle(wid, L, worker).  Otherwise subprocess workers with env =
    lvp.worker_env(base_env, lp_threads=L, home=out_dir/homes/w{wid:02d}_L{L}, profile=profile) and PYTHONPATH =
    extra_pythonpath + code_root (lvp.prepare_home first).  tvr_env / releases are passed to lvp (tests point them
    at fakes).  One harness thread uses one worker slot at a time; start-ups and restarts are serialized."""

    def __init__(self, n_envs: int, lp_threads: int, profile: str, *, out_dir, houses_dir, repo=lvp.TVR_REPO,
                 py=lvp.TVR_PY, code_root=None, base_env: Optional[Mapping[str, str]] = None,
                 startup_timeout: float = STARTUP_TIMEOUT_S, local_factory: Optional[Callable[[int], Any]] = None,
                 extra_pythonpath: Sequence = (), tvr_env=lvp.TVR_ENV, releases=lvp.RELEASES, procthor_dir=None):
        if n_envs < 1 or lp_threads < 1:
            raise ValueError("n_envs and lp_threads must be >= 1")
        if profile not in lvp.LVP_PROFILES:
            raise ValueError(f"unknown lavapipe profile {profile!r}")
        self.n, self.L, self.profile = int(n_envs), int(lp_threads), profile
        self.out_dir = Path(out_dir)
        self.houses_dir = houses_dir
        self.repo, self.py = Path(repo), str(py)
        self.code_root = Path(code_root) if code_root is not None else paths.WORKTREE
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.startup_timeout = float(startup_timeout)
        self.local_factory = local_factory
        self.extra_pythonpath = [str(p) for p in extra_pythonpath]
        self.tvr_env, self.releases = tvr_env, releases
        self.procthor_dir = procthor_dir if procthor_dir is not None else self.repo / "data" / "procthor-10k"
        self.handles: List[Any] = [None] * self.n
        self.readys: List[Optional[Dict[str, Any]]] = [None] * self.n
        self.history: List[Dict[str, Any]] = []  # every start-up attempt (manifest)
        self._startup_lock = threading.Lock()
        self._spawning = None
        self.closed = False

    # -- spawning
    def _spawn(self, wid: int):
        if self.local_factory is not None:
            try:
                worker = self.local_factory(wid)
                ready = probe_lvp.jsafe(dict(worker.start(), ok=True, id=0))
                return probe_lvp.LocalHandle(wid, self.L, worker), ready
            except Exception as e:  # noqa: BLE001
                return None, {"id": 0, "ok": False, "error": repr(e), "tb": traceback.format_exc()[-2000:]}
        home = self.out_dir / "homes" / f"w{wid:02d}_L{self.L}"
        lvp.prepare_home(home, releases=self.releases)
        env = lvp.worker_env(self.base_env, lp_threads=self.L, home=home, profile=self.profile,
                             tvr_env=self.tvr_env)
        env["PYTHONPATH"] = os.pathsep.join(self.extra_pythonpath + [str(self.code_root)])
        argv = [self.py, "-B", "-s", "-u", "-m", "vpd.tvr.thor_worker", "worker", "--wid", str(wid),
                "--repo", str(self.repo), "--procthor-dir", str(self.procthor_dir)]
        if self.houses_dir is not None:
            argv += ["--houses", str(self.houses_dir)]
        (self.out_dir / "logs").mkdir(parents=True, exist_ok=True)
        h = probe_lvp.SubprocessHandle(wid, self.L, argv, env, self.code_root,
                                       self.out_dir / "logs" / f"worker_w{wid:02d}_L{self.L}.log")
        self._spawning = h  # a fast close() during the start-up wait kills this one too
        try:
            ready = dict(h.wait(0, self.startup_timeout), spawn_env=lvp.env_summary(env))
        except BaseException:
            # SIGTERM (run_tvr.Terminated is raised in the main thread inside this wait) or any error: this handle
            # is not in self.handles yet, so a later close(fast=True) would not see it -> kill its session now
            try:
                h.kill()
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            self._spawning = None
        if not ready.get("ok") or self.closed:
            h.kill()
        return h, ready

    def _spawn_logged(self, wid: int):
        """_spawn that never raises an Exception (prepare_home OSError, Popen EAGAIN / ENOMEM / EMFILE, ...): the
        failure is a ready reply with ok False (contract 5.2.1: start / restart return the ready reply)."""
        t0 = time.monotonic()
        try:
            h, ready = self._spawn(wid)
        except Exception as e:  # noqa: BLE001
            h, ready = None, {"id": 0, "ok": False, "error": f"spawn failed: {type(e).__name__}: {e}"[:1000],
                              "tb": traceback.format_exc()[-2000:]}
        self.history.append({"wid": wid, "ok": bool(ready.get("ok")), "utc": utc(),
                             "spawn_s": round(time.monotonic() - t0, 3), "error": ready.get("error")})
        return h, ready

    def start(self) -> List[Dict[str, Any]]:
        """Spawn the n_envs workers ONE AT A TIME; a failed start is retried once; PoolError if a worker cannot
        start twice (the workers already started are closed)."""
        with self._startup_lock:
            for wid in range(self.n):
                if self.closed:
                    raise PoolError("pool closed during start-up")
                h, ready = self._spawn_logged(wid)
                if not ready.get("ok") and not self.closed:
                    h, ready = self._spawn_logged(wid)
                if not ready.get("ok"):
                    self.readys[wid] = ready
                    self.close(fast=True)
                    raise PoolError(f"worker {wid} did not start twice: {ready.get('error')}")
                self.handles[wid], self.readys[wid] = h, ready
        return self.ready()

    def restart(self, wid: int) -> Dict[str, Any]:
        """Kill worker `wid` and start a new one in its slot (under the start-up lock); the ready reply (ok False
        when the new start-up failed: the slot is then empty).  Never raises an Exception."""
        with self._startup_lock:
            old = self.handles[wid]
            self.handles[wid] = None
            if old is not None:
                try:
                    old.kill()
                except Exception:  # noqa: BLE001
                    pass
            if self.closed:
                return {"ok": False, "error": "pool closed"}
            h, ready = self._spawn_logged(wid)
            if ready.get("ok") and not self.closed:
                self.handles[wid], self.readys[wid] = h, ready
            elif h is not None:
                try:
                    h.kill()
                except Exception:  # noqa: BLE001
                    pass
            return ready

    def alive(self, wid: int) -> bool:
        h = self.handles[wid]
        return h is not None and bool(getattr(h, "alive", False)) and not self.closed

    def call(self, wid: int, cmd: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
        """One command to worker `wid`; never raises: {"ok": False, "timeout"|"died": True, ...} on failure (the
        worker is then killed; restart() gives the slot a new one)."""
        h = self.handles[wid]
        if self.closed or h is None or not getattr(h, "alive", False):
            return {"ok": False, "error": "worker not alive" if not self.closed else "pool closed", "died": True}
        try:
            r = h.wait(h.send(cmd), max(0.0, float(timeout)))
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": f"call failed: {e!r}", "died": True}
        if r.get("timeout") or r.get("died"):
            try:
                h.kill()
            except Exception:  # noqa: BLE001
                pass
        return r

    def ready(self) -> List[Dict[str, Any]]:
        return [r for r, h in zip(self.readys, self.handles) if r is not None and h is not None]

    def close(self, fast: bool = False) -> None:
        """normal: quit to every live worker at once (one shared QUIT_TIMEOUT_S), then kill every session;
        fast (SIGTERM / deadline): killpg every worker at once, no quit round-trip (returns in < 2 s)."""
        self.closed = True
        hs = [h for h in self.handles if h is not None]
        if self._spawning is not None:
            hs.append(self._spawning)
        if not fast:
            sent = []
            for h in hs:
                if getattr(h, "alive", False):
                    try:
                        sent.append((h, h.send({"op": "quit"})))
                    except Exception:  # noqa: BLE001
                        pass
            end = time.monotonic() + QUIT_TIMEOUT_S
            for h, cid in sent:
                try:
                    h.wait(cid, max(0.0, end - time.monotonic()))
                except Exception:  # noqa: BLE001
                    pass
        else:
            for h in hs:  # SIGKILL every session first, then reap
                proc = getattr(h, "proc", None)
                if proc is not None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
        for h in hs:
            try:
                h.kill()
            except Exception:  # noqa: BLE001
                pass


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def find_unity(pattern: str = lvp.UNITY_PAT) -> List[int]:
    me = os.getpid()
    out = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for name in names:
        if not name.isdigit() or int(name) == me:
            continue
        if pattern in _cmdline(int(name)):
            out.append(int(name))
    return sorted(out)


def reap_unity(log=None, pattern: str = lvp.UNITY_PAT, grace_s: float = 5.0) -> List[int]:
    """Kill lingering Unity players whose command line contains `pattern` (TERM, up to grace_s, then KILL); the
    pids found.  `log` (a file-like object or a callable) gets one line."""
    pids = find_unity(pattern)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    end = time.monotonic() + grace_s
    left = list(pids)
    while left and time.monotonic() < end:
        left = [p for p in left if os.path.exists(f"/proc/{p}") and pattern in _cmdline(p)]
        if left:
            time.sleep(0.2)
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    msg = f"[thor_worker] reap_unity {utc()}: found {pids}, killed after TERM grace {left}"
    if callable(log):
        log(msg)
    elif log is not None:
        log.write(msg + "\n")
    return pids


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker", help="one THOR worker (inside a job only: starts Unity)")
    w.add_argument("--cmd-fd", type=int, required=True)
    w.add_argument("--reply-fd", type=int, required=True)
    w.add_argument("--wid", type=int, required=True)
    w.add_argument("--repo", default=str(lvp.TVR_REPO))
    w.add_argument("--procthor-dir", default=str(lvp.PROCTHOR_DIR))
    w.add_argument("--houses", default=None)
    r = sub.add_parser("reap", help="kill lingering Unity players of the pinned build")
    r.add_argument("--grace-s", type=float, default=5.0)
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        return worker_main(a)
    pids = reap_unity(log=sys.stdout, grace_s=a.grace_s)
    print(json.dumps({"reaped": pids}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
