"""TVR closed-loop eval harness and demo replay (CONTRACT_wave2 4, 5.2.2, 6.4; DECISIONS G3, G10).  PY_Q3.

    PY_Q3 -B -m vpd.tvr.run_tvr eval --start S --tier T --arms c[,u][,blind] --task-set eval500|dev200|smoke8|smoke4
          --out-root DIR [--name-prefix pilot|cal|smoke|calshake] [--model-dir LOCAL --adapter-meta A/meta.json]
          [--envs E] [--lp L] [--profile P] [--blind-concurrency 16] [--deadline-s S] [--resume]
          [--mechanism-only] [--vllm-parity DRY_RUN_JSON] [--keep-images-per-category 2]
          [--limit N --allow-limit] [--base-url URL | --fake-server] [--fake-thor FAKE_ROOT]
          [--gpu-memory-utilization 0.85] [--max-model-len 128000 (the O9 fallback only)]
    PY_Q3 -B -m vpd.tvr.run_tvr replay --task-set sft1600|sft2 --public SFT_TRAIN --run-dir D --images-dir I
          [--envs E] [--lp L] [--profile P] [--deadline-s S] [--resume] [--fake-thor FAKE_ROOT]
    PY_Q3 -B -m vpd.tvr.run_tvr rescore --run-dir D
Data roots (defaults = the frozen layout; tests point them at fakes): --tvr-out (OUT/tvr: split/, reach/v1, bfs/,
houses/), --tasks-dir (REPO/data/tasks), --reach, --bfs, --houses, --repo.

Protocol (normative, contract 4.5 / 4.6, identical to eval_va.eval_one_task): messages = [system(SYSTEM_PROMPT as a
string), user(obs_0)]; per reply: the raw reply is appended as the assistant turn, a = parse(reply) or
"InvalidAction" (c / blind: re.fullmatch(REGEX); u: eval_va.parse_action), the env is stepped through TVRBench's
own ThorViewEnv._sync_step (thor_worker), the episode ends at Stop or the cap (30 iTHOR / 40 ProcTHOR replies),
otherwise the next user turn carries CURRENT + TARGET (re-sent every turn) as PNG data URLs (image arms) or the
text with the placeholders removed (blind arm, kin_env); success = stopped and is_exact_match(final, target).
Request body: {"model", "messages", "temperature": 0.0, "top_p": 1.0, "max_tokens": 16 (c, blind) / 64 (u),
"top_k": -1} + "structured_outputs": {"regex": REGEX} on c / blind; no chat_template_kwargs, seed or uuid fields.
Every reset injects the FROZEN per-task start-pose reach set (reach/v1 index) into ThorEnv (contract 4.3); the live
set is recorded.  Kin shadow: every THOR step is re-computed with kin_env (typed mismatches: valid_list / pose /
success).  No pose -> frame cache: every frame sent or saved is the one rendered for that step (F4).

Run dirs: DIR/<prefix>__<START>__<TIER>__<ARM> per arm (one VLLMServer session for all arms; one ThorPool of E x L
workers kept across the image arms; blind after them with --blind-concurrency threads).  Files (6.4):
protocol.json, task_ids.json, manifest.json, tasks/<id>/{trajectory.json, calls.jsonl, prompt_dump.jsonl},
calls.jsonl, prompt_dump.jsonl, images/<id>/ (kept tasks), per_task.jsonl, perf_parts.jsonl (one throughput
segment per process; results.perf sums them over resumes), results.json | results_invalid.json |
results_partial.json.  Resume: protocol.json must be equal; finished tasks (same protocol_sha256) are skipped; a
complete arm is reported and skipped (rc 0); a valid results.json is never overwritten.  Every pending arm gets a
result file: an arm with missing tasks after the soft deadline passed (anywhere: server / pool start-up, an earlier
arm's finalize, inside the arm) is partial; after the model server died, the remaining arms are not run (invalid).
An image arm / the replay is never valid without a measured renderer (a worker ready reply of this process; the
pool is not started with fewer than NEW_TASK_MIN_S left): partial near the deadline, else invalid.
Exit: 0 every arm valid, 1 some arm invalid, 2 refused, 4 incomplete (deadline), 143 SIGTERM.
Never run eval / replay on the login node with real workers or a real server (Unity, GPU).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vpd import paths
from vpd.tvr import kin_env, lvp, pilot_cfg, probe_lvp, score_tvr, thor_worker, tvr_text
from vpd.tvr import reach as reach_mod
from vpd.tvr import tasks as tasks_mod

TRAJ_SCHEMA = "vpd.tvr.trajectory/1"
MANIFEST_SCHEMA = "vpd.tvr.manifest/1"
PROTOCOL_SCHEMA = "vpd.tvr.protocol/1"
CODE_FILES = ("vpd/tvr/run_tvr.py", "vpd/tvr/thor_worker.py", "vpd/tvr/score_tvr.py", "vpd/tvr/parity_tvr.py",
              "vpd/tvr/tvr_text.py", "vpd/tvr/kin_env.py", "vpd/tvr/bfs.py", "vpd/tvr/reach.py", "vpd/tvr/tasks.py",
              "vpd/tvr/pilot_cfg.py", "vpd/tvr/lvp.py", "vpd/common/vllm_server_vpd.py")
TIER_PHASE = {"0W": "0W", "100": "tier100", "400": "tier400", "ALL": "tierALL", "100s1002": "tier100_s1002"}
TIERS = ("raw",) + tuple(TIER_PHASE)
EVAL_SETS = ("eval500", "dev200", "smoke8", "smoke4")
REPLAY_SETS = ("sft1600", "sft2")
SFT2 = tuple(probe_lvp.SAMPLE_TASKS)  # contract 4.1: the probe's public sample tasks
IMAGE_ARMS = ("c", "u")
ACTIONS = pilot_cfg.ACTIONS
NON_STOP = pilot_cfg.NON_STOP
PROTO = pilot_cfg.FROZEN["protocol"]
MAX_EL = 11  # the probe's `recommended` rule (E*L <= 11, F4)
NEW_TASK_MIN_S = 150.0  # no new task starts with fewer seconds left
HTTP_BACKOFF_S = (2.0, 4.0, 8.0)  # an HTTP error is retried 3 times
HTTP_TIMEOUT_S = 600.0
KEEPALIVE_IDLE_S = 3.0  # vLLM's uvicorn closes idle keep-alive connections after 5 s: reconnect after 3 s idle
STALE_CONN_ERRORS = (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
PERF_PARTS = "perf_parts.jsonl"  # one line per process that ran the arm (perf over resumes)
DUMP_TURNS = (0, 1, 2)  # prompt dump: these calls + the last one of each kept task
LOAD_COST_S = 6.0
CPU_SAMPLE_S = 10.0
AUDIT_MAX_IMAGES = 8
EXIT_OK, EXIT_INVALID, EXIT_REFUSED, EXIT_INCOMPLETE, EXIT_TERM = 0, 1, 2, 4, 143
SERVE_PACKAGES = ("vllm", "xgrammar", "transformers", "torch")


class Refused(Exception):
    pass


class Terminated(BaseException):
    """SIGTERM / SIGINT in the main thread: fast shutdown, exit 143."""


class Abandon(Exception):
    """Stop or deadline reached inside a task: the task is not recorded (re-run on resume)."""


class TaskError(Exception):
    def __init__(self, kind: str, msg: str):
        super().__init__(f"{kind}: {msg}")
        self.kind = kind


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(*a) -> None:
    print("[run_tvr]", time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def jb(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


def write_json_atomic(path, obj) -> None:
    score_tvr.write_json_atomic(path, probe_lvp.jsafe(obj))


def write_text_atomic(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}.{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def read_json(p) -> Optional[dict]:
    return score_tvr.read_json(p)


def code_sha256(root=None) -> Dict[str, Any]:
    """{"code_sha256": sha256 of the "rel sha" lines (sorted), "files": {rel: sha256 | None}} over CODE_FILES."""
    root = Path(root) if root is not None else paths.WORKTREE
    files = {rel: sha256_file(root / rel) for rel in CODE_FILES}
    total = sha256_bytes("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode())
    return {"code_sha256": total, "files": files}


# --- TVRBench metrics (tvrbench/evaluation/metrics.py, identical formulas) ------------------------------------
def position_error(state, target) -> float:
    dx = state["position"]["x"] - target["position"]["x"]
    dz = state["position"]["z"] - target["position"]["z"]
    return math.sqrt(dx * dx + dz * dz)


def rotation_error(state, target) -> float:
    diff = abs(state["rotation_y"] - target["rotation_y"]) % 360
    return min(diff, 360 - diff)


def horizon_error(state, target) -> float:
    return abs(state["horizon"] - target["horizon"])


def is_exact_match(state, target, pos_tol=0.01, rot_tol=1.0, hor_tol=1.0) -> bool:
    return (position_error(state, target) < pos_tol and rotation_error(state, target) < rot_tol
            and horizon_error(state, target) < hor_tol)


# --- control: deadline, stop -----------------------------------------------------------------------------------
class Ctl:
    def __init__(self, deadline_s: Optional[float], t0: Optional[float] = None):
        self.t0 = time.monotonic() if t0 is None else t0
        self.deadline_s = None if deadline_s is None else float(deadline_s)
        self.stop = threading.Event()
        self.deadline_hit = False
        self.new_tasks_stopped = False
        self.terminated = False

    def left(self) -> float:
        if self.deadline_s is None:
            return math.inf
        return self.deadline_s - (time.monotonic() - self.t0)

    def stopped(self) -> bool:
        return self.stop.is_set() or self.left() <= 0

    def wait(self, s: float) -> None:
        self.stop.wait(max(0.0, min(float(s), self.left())))

    def bounded(self, timeout: float) -> float:
        return max(1.0, min(float(timeout), self.left()))


# --- task sets ----------------------------------------------------------------------------------------------------
def resolve_tasks(name: str, *, tvr_out, tasks_dir) -> List[Dict[str, Any]]:
    """The named task set of contract 4.1 (task dicts unmodified, in set order)."""
    if name == "sft1600":
        return list(tasks_mod.load_split("sft", tasks_dir))
    if name == "eval500":
        return list(tasks_mod.load_split("eval", tasks_dir))
    if name == "sft2":
        sft = tasks_mod.by_id(tasks_mod.load_split("sft", tasks_dir))
        return [sft[t] for t in SFT2]
    if name in ("dev200", "smoke8", "smoke4"):
        sel = read_json(Path(tvr_out) / "split" / "dev200" / "selection.json")
        if sel is None:
            raise Refused(f"no dev selection at {Path(tvr_out) / 'split' / 'dev200' / 'selection.json'}")
        rl = tasks_mod.by_id(tasks_mod.load_split("rl", tasks_dir))
        ids = sel["ids"] if name == "dev200" else sel[name]
        return [rl[t] for t in ids]
    raise Refused(f"unknown task set {name!r}")


def bfs_name(task_set: str) -> str:
    return "eval500" if task_set == "eval500" else "dev200"


def keep_ids(tasks: List[Dict[str, Any]], per_cat: int) -> set:
    out = set()
    for c in pilot_cfg.CATEGORIES:
        ids = sorted(t["task_id"] for t in tasks if pilot_cfg.category_of(t) == c)
        out.update(ids[:max(0, int(per_cat))])
    return out


def scene_groups(tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Tasks grouped by scene; groups by estimated cost (sum of caps + 6 s load) descending; task_id order inside."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        by.setdefault(t["scene"], []).append(t)
    groups = [sorted(g, key=lambda t: t["task_id"]) for g in by.values()]
    groups.sort(key=lambda g: (-(sum(pilot_cfg.cap_of(t) for t in g) + LOAD_COST_S), g[0]["scene"]))
    return groups


# --- servers ------------------------------------------------------------------------------------------------------
class ExternalServer:
    """--base-url: a server someone else runs (tests, debugging)."""
    fake = False
    proc = None

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.ready_seconds = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=5) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def manifest(self) -> Dict[str, Any]:
        return {"source": "external --base-url", "base_url": self.base_url}


def serve_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    out: Dict[str, Optional[str]] = {}
    for p in SERVE_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


class ChatClient:
    """One keep-alive http.client connection per harness thread; 3 retries (2, 4, 8 s) on any error.

    A connection idle for more than KEEPALIVE_IDLE_S is replaced before use, and a stale keep-alive socket (the
    server closed it: RemoteDisconnected / BrokenPipe / ConnectionReset) is retried once at once on a fresh
    connection without spending a retry.  `alive` (the server's alive()) is checked after every failed attempt:
    a dead server ends the call at once with TaskError("server_dead")."""

    def __init__(self, base_url: str, ctl: Ctl, alive: Optional[Callable[[], bool]] = None):
        u = urllib.parse.urlsplit(base_url)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.path = (u.path.rstrip("/") or "") + "/chat/completions"
        self.ctl = ctl
        self.alive = alive
        self.conn: Optional[http.client.HTTPConnection] = None
        self.last_used = 0.0
        self.reconnects = 0

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            self.conn = None

    def _post(self, body: bytes, timeout: float) -> Dict[str, Any]:
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        else:
            self.conn.timeout = timeout
            if self.conn.sock is not None:
                self.conn.sock.settimeout(timeout)
        self.conn.request("POST", self.path, body=body, headers={"Content-Type": "application/json"})
        r = self.conn.getresponse()
        data = r.read()
        self.last_used = time.monotonic()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
        return json.loads(data)

    def _once(self, body: bytes, timeout: float) -> Dict[str, Any]:
        if self.conn is not None and time.monotonic() - self.last_used > KEEPALIVE_IDLE_S:
            self.close()  # the server may have closed it already (uvicorn timeout_keep_alive)
        reused = self.conn is not None
        try:
            return self._post(body, timeout)
        except STALE_CONN_ERRORS:
            if not reused:
                raise
            self.close()
            self.reconnects += 1
            return self._post(body, timeout)  # a stale keep-alive socket: once more, fresh connection, no backoff

    def chat(self, body: bytes) -> Tuple[Dict[str, Any], float, int, List[str]]:
        errs: List[str] = []
        for i in range(len(HTTP_BACKOFF_S) + 1):
            if self.ctl.stopped():
                raise Abandon()
            t0 = time.perf_counter()
            try:
                resp = self._once(body, self.ctl.bounded(HTTP_TIMEOUT_S))
                return resp, time.perf_counter() - t0, i, errs
            except Exception as e:  # noqa: BLE001
                self.close()
                errs.append(f"{type(e).__name__}: {e}"[:300])
                if self.ctl.stopped():
                    raise Abandon()
                if self.alive is not None and not self.alive():
                    raise TaskError("server_dead", "the model server is not alive; " + "; ".join(errs[-2:]))
                if i < len(HTTP_BACKOFF_S):
                    self.ctl.wait(HTTP_BACKOFF_S[i])
        raise TaskError("api", "; ".join(errs[-2:]))


# --- conversation (cached per-message JSON; never re-serializes old images) ---------------------------------------
def image_part(png_b64: str) -> bytes:
    return jb({"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64}})


class Convo:
    def __init__(self, served: str, max_tokens: int, regex: Optional[str]):
        sysmsg = {"role": "system", "content": tvr_text.SYSTEM_PROMPT}
        self.msgs: List[bytes] = [jb(sysmsg)]
        self.red: List[Dict[str, Any]] = [sysmsg]
        tail: Dict[str, Any] = {"temperature": float(PROTO["temperature"]), "top_p": float(PROTO["top_p"]),
                                "max_tokens": int(max_tokens), "top_k": int(PROTO["top_k"])}
        if regex is not None:
            tail["structured_outputs"] = {"regex": regex}
        self.tail_obj = tail
        self.head = b'{"model": ' + jb(served) + b', "messages": ['
        self.tail = b"], " + jb(tail)[1:]
        self.n_images = 0

    def add_user_images(self, obs_str: str, cur: Tuple[bytes, dict], tgt: Tuple[bytes, dict]) -> None:
        """eval_va._obs_to_oai_content: obs_str split on <image>; non-empty text pieces; images [CURRENT, TARGET]."""
        pieces = obs_str.split(tvr_text.IMAGE_PH)
        imgs = [cur, tgt]
        if len(pieces) - 1 != len(imgs):
            raise TaskError("protocol", f"obs has {len(pieces) - 1} <image> placeholders but {len(imgs)} images")
        parts: List[bytes] = []
        red: List[Dict[str, Any]] = []
        for i, piece in enumerate(pieces):
            if piece:
                parts.append(jb({"type": "text", "text": piece}))
                red.append({"type": "text", "text": piece})
            if i < len(imgs):
                parts.append(imgs[i][0])
                red.append(imgs[i][1])
        self.msgs.append(b'{"role": "user", "content": [' + b", ".join(parts) + b"]}")
        self.red.append({"role": "user", "content": red})
        self.n_images += len(imgs)

    def add_user_text(self, text: str) -> None:
        m = {"role": "user", "content": text}
        self.msgs.append(jb(m))
        self.red.append(m)

    def add_assistant(self, text: str) -> None:
        m = {"role": "assistant", "content": text}
        self.msgs.append(jb(m))
        self.red.append(m)

    def body(self) -> bytes:
        return self.head + b", ".join(self.msgs) + self.tail


# --- kin shadow ---------------------------------------------------------------------------------------------------
class KinShadow:
    """kin_env re-computation of every THOR step (contract 5.2.2): typed mismatches valid_list / pose / success."""

    def __init__(self, reach):
        self.reach = reach
        self.counts = {"valid_list": 0, "pose": 0, "success": 0}
        self.first: Optional[Dict[str, Any]] = None

    def _mm(self, t: int, typ: str, thor, kin) -> None:
        self.counts[typ] += 1
        if self.first is None:
            self.first = {"t": t, "type": typ, "thor": thor, "kin": kin}

    def check_valid(self, t: int, state, valid) -> None:
        kv = kin_env.valid_actions(kin_env.pose_of(state), self.reach, True)
        if list(valid) != list(kv):
            self._mm(t, "valid_list", list(valid), list(kv))

    def check_step(self, t: int, prev, action: str, after, success) -> None:
        p = kin_env.pose_of(prev)
        if action in NON_STOP:
            kp, ks = kin_env.step(p, action, self.reach)
        else:  # Stop / InvalidAction: no transition
            kp, ks = p, None
        got = kin_env.pose_of(after)
        if tuple(got) != tuple(kp):
            self._mm(t, "pose", list(got), list(kp))
        if action in NON_STOP and bool(success) != bool(ks):
            self._mm(t, "success", success, ks)

    @property
    def equal(self) -> bool:
        return sum(self.counts.values()) == 0


# --- one arm ------------------------------------------------------------------------------------------------------
class ArmRun:
    def __init__(self, *, arm: str, name: str, run_dir: Path, tasks: List[Dict[str, Any]], protocol: Dict[str, Any],
                 mechanism_only: bool = False, images_dir: Optional[Path] = None, public=None, keep: int = 0):
        self.arm, self.name, self.run_dir = arm, name, Path(run_dir)
        self.tasks = tasks
        self.task_ids = [t["task_id"] for t in tasks]
        self.protocol = protocol
        self.protocol_sha = sha256_bytes(canon(protocol).encode())
        self.mech = bool(mechanism_only)
        self.images_dir = None if images_dir is None else Path(images_dir)
        self.public = public
        self.keep_ids = keep_ids(tasks, keep) if arm != "replay" else set()
        self.lock = threading.Lock()
        self.api_errors = 0
        self.failures: List[Dict[str, Any]] = []
        self.requeued: List[Dict[str, Any]] = []  # attempt-1 errors of tasks requeued to another slot (no failure)
        self.steps = 0
        self.max_images = 0
        self.max_prompt = 0
        self.wall = 0.0
        self.complete = False
        self.todo: List[Dict[str, Any]] = []
        self.cpu: Dict[str, Any] = {}
        self.ran = False
        self.n_written = 0  # trajectories written by this process
        self.n_slots = 0

    @property
    def max_tokens(self) -> int:
        return int(PROTO["max_tokens"].get(self.arm, 16))

    @property
    def regex(self) -> Optional[str]:
        return pilot_cfg.REGEX if self.arm in ("c", "blind") else None

    def task_dir(self, tid: str) -> Path:
        return self.run_dir / "tasks" / tid

    def done_ids(self) -> set:
        out = set()
        for tid in self.task_ids:
            tr = read_json(self.task_dir(tid) / "trajectory.json")
            if tr is not None and (tr.get("vpd") or {}).get("protocol_sha256") == self.protocol_sha:
                out.add(tid)
        return out


def results_state(run_dir: Path) -> Optional[str]:
    if (run_dir / "results.json").exists():
        return "valid"
    if (run_dir / "results_partial.json").exists():
        return "partial"
    if (run_dir / "results_invalid.json").exists():
        return "invalid"
    return None


def has_trajectories(run_dir: Path) -> bool:
    return any((run_dir / "tasks").glob("*/trajectory.json")) if (run_dir / "tasks").is_dir() else False


def preflight(ar: ArmRun, resume: bool) -> None:
    """Resume / refusal rules (contract 5.2.2).  Sets ar.complete / ar.todo; Refused on a conflict."""
    rd = ar.run_dir
    st = results_state(rd)
    old = read_json(rd / "protocol.json")
    if st == "valid":
        if not resume:
            raise Refused(f"{rd}/results.json exists (a valid result is never overwritten; --resume reports it)")
        res = read_json(rd / "results.json")
        if res is None or not res.get("valid"):
            raise Refused(f"{rd}/results.json is unreadable or not valid")
        if old != ar.protocol:
            raise Refused(f"{rd}: protocol.json differs from this run's protocol")
        ar.complete = True
        return
    if resume:
        if old is not None and old != ar.protocol:
            raise Refused(f"{rd}: protocol.json differs from this run's protocol (resume refused)")
        if old is None and has_trajectories(rd):
            raise Refused(f"{rd}: trajectories without protocol.json (unknown provenance)")
        done = ar.done_ids()
        ar.todo = [t for t in ar.tasks if t["task_id"] not in done]
    else:
        if has_trajectories(rd) or st is not None:
            raise Refused(f"{rd} already holds a run (pass --resume to continue it)")
        ar.todo = list(ar.tasks)


def concat_jsonl(ar: ArmRun, name: str) -> int:
    n = 0
    lines: List[str] = []
    for tid in ar.task_ids:
        p = ar.task_dir(tid) / name
        if p.is_file() and (ar.task_dir(tid) / "trajectory.json").is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    lines.append(line)
                    n += 1
    write_text_atomic(ar.run_dir / name, "".join(x + "\n" for x in lines))
    return n


# --- CPU monitor --------------------------------------------------------------------------------------------------
class CpuMon:
    def __init__(self, path=lvp.CPU_STAT, every_s: float = CPU_SAMPLE_S):
        self.path, self.every = path, every_s
        self.samples: List[Tuple[float, Optional[int]]] = []
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def _read(self) -> Tuple[float, Optional[int]]:
        return time.monotonic(), lvp.read_cpu_stat(self.path).get("usage_usec")

    def start(self) -> "CpuMon":
        self.samples.append(self._read())

        def loop():
            while not self._stop.wait(self.every):
                self.samples.append(self._read())

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def stop(self) -> Dict[str, Optional[float]]:
        self._stop.set()
        self.samples.append(self._read())
        s = [(t, u) for t, u in self.samples if u is not None]
        if len(s) < 2 or s[-1][0] <= s[0][0]:
            return {"cores_mean": None, "cores_max": None}
        mean = (s[-1][1] - s[0][1]) / 1e6 / (s[-1][0] - s[0][0])
        mx = max(((b[1] - a[1]) / 1e6 / (b[0] - a[0])) for a, b in zip(s, s[1:]) if b[0] > a[0])
        return {"cores_mean": round(mean, 3), "cores_max": round(mx, 3)}


# --- audits (in-run, image arms) ----------------------------------------------------------------------------------
def kept_pngs(run_dir: Path) -> List[Path]:
    pngs = sorted((run_dir / "images").glob("*/*.png")) if (run_dir / "images").is_dir() else []
    return pngs[:: max(1, len(pngs) // AUDIT_MAX_IMAGES)][:AUDIT_MAX_IMAGES] if pngs else []


def qwen_grid(w: int, h: int, patch: int = 16, merge: int = 2, min_pixels: int = 65536,
              max_pixels: int = paths.MAX_PIXELS) -> List[int]:
    """Qwen3-VL smart_resize arithmetic (fake-server audit): [1, H/patch, W/patch] after rounding to patch*merge."""
    f = patch * merge
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb, wb = math.floor(h / beta / f) * f, math.floor(w / beta / f) * f
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return [1, hb // patch, wb // patch]


def fake_image_token_audit(run_dir: Path) -> Dict[str, Any]:
    from PIL import Image

    samples = []
    for p in kept_pngs(run_dir):
        with Image.open(p) as im:
            grid = qwen_grid(*im.size)
        samples.append({"path": str(p.relative_to(run_dir)), "size": [im.size[0], im.size[1]], "grid_thw": grid,
                        "tokens": grid[0] * grid[1] * grid[2] // 4 + 2})
    return {"method": "FAKE (fake server): Qwen3-VL smart_resize arithmetic, prod(grid)/4 + 2",
            "n": len(samples), "grid_set": sorted({tuple(s["grid_thw"]) for s in samples}),
            "tokens_per_image": sorted({s["tokens"] for s in samples}), "samples": samples}


def fake_prompt_parity(run_dir: Path) -> Dict[str, Any]:
    from vpd.tvr import fake_server

    rows = []
    p = run_dir / "prompt_dump.jsonl"
    for line in (p.read_text(encoding="utf-8").splitlines() if p.is_file() else []):
        rec = json.loads(line)
        rows.append({"task": rec.get("task"), "step": rec.get("step"), "n_images": rec.get("n_images"),
                     "hf_prompt_tokens": fake_server.prompt_tokens(rec["messages"]),
                     "server_prompt_tokens": rec.get("prompt_tokens")})
    return {"method": "FAKE (fake server): fake_server.prompt_tokens on the dumped messages", "n": len(rows),
            "equal": bool(rows) and all(r["hf_prompt_tokens"] == r["server_prompt_tokens"] for r in rows),
            "rows": rows}


def image_token_audit(run_dir: Path, processor_dir: Path, max_pixels: int = paths.MAX_PIXELS) -> Dict[str, Any]:
    """Tokens per image from the served processor's image_grid_thw on the kept PNGs (prod(grid)/merge^2 + 2,
    run_vpd.image_token_audit convention, DECISIONS B4).  Processor files only."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    from vpd import starts

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    ip = AutoImageProcessor.from_pretrained(str(processor_dir))
    starts.set_max_pixels(ip, max_pixels)
    merge = int(getattr(ip, "merge_size", 2))
    samples = []
    for p in kept_pngs(run_dir):
        with Image.open(p) as im:
            im = im.convert("RGB")
            grid = [int(g) for g in ip(images=[im], return_tensors="pt")["image_grid_thw"][0].tolist()]
        n = 1
        for g in grid:
            n *= g
        samples.append({"path": str(p.relative_to(run_dir)), "size": list(im.size), "grid_thw": grid,
                        "tokens": n // (merge * merge) + 2})
    return {"method": "processor image_grid_thw on kept PNGs: prod(grid)/merge^2 + 2 (DECISIONS B4)",
            "processor_dir": str(processor_dir), "max_pixels": max_pixels, "n": len(samples),
            "grid_set": sorted({tuple(s["grid_thw"]) for s in samples}),
            "tokens_per_image": sorted({s["tokens"] for s in samples}), "samples": samples}


def hf_prompt_parity(run_dir: Path, processor_dir: Path, max_pixels: int = paths.MAX_PIXELS) -> Dict[str, Any]:
    """run_vpd.hf_prompt_parity on RUN/prompt_dump.jsonl (image parts -> the kept PNG files, RUN-relative)."""
    from vpd.walkerbench import run_vpd

    return run_vpd.hf_prompt_parity(run_dir, Path(processor_dir), run_dir / "prompt_dump.jsonl", max_pixels)


def processor_dir_for(start: str, model_dir: Optional[str]) -> Path:
    if model_dir and os.path.lexists(Path(model_dir) / "preprocessor_config.json"):
        return Path(model_dir)
    from vpd import starts

    return starts.processor_dir(start)


def adapter_info(meta_path: Optional[Path], start: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Provenance of the merged adapter (run_vpd.adapter_info keys)."""
    if meta_path is None:
        return None
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    pt = meta_path.parent / "adapter.pt"
    return {"meta_path": str(meta_path), "meta_sha256": sha256_file(meta_path),
            "adapter_path": str(pt) if pt.exists() else None, "adapter_sha256": sha256_file(pt) if pt.exists() else None,
            "start": meta.get("start"), "tier": meta.get("phase"), "seed": meta.get("seed"), "data": meta.get("data"),
            "data_sha256": meta.get("data_sha256"), "init_adapter": meta.get("init_adapter"),
            "code_sha256": meta.get("code_sha256"), "optimizer_steps": meta.get("optimizer_steps")}


def git_facts() -> Dict[str, Any]:
    out: Dict[str, Any] = {"root": str(paths.WORKTREE)}
    try:
        from vpd import snapshot

        st = snapshot.snapshot_state(paths.WORKTREE)
        out["snapshot"] = {k: st.get(k) for k in ("mode", "commit", "utc") if k in st}
        if st.get("mode") == "snapshot":
            out["commit"] = st.get("commit")
            return out
    except Exception as e:  # noqa: BLE001
        out["snapshot_error"] = repr(e)[:300]
    try:
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(paths.WORKTREE), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        out["commit"] = r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        out["commit"] = None
    return out


# --- the harness ------------------------------------------------------------------------------------------------
class Harness:
    def __init__(self, *, ctl: Ctl, reach_index, served: Optional[str], pool=None, server=None,
                 bfs_table: Optional[dict] = None, bfs_path: Optional[str] = None, start: Optional[str] = None,
                 tier: Optional[str] = None, task_set: str = "", model_dir: Optional[str] = None,
                 adapter: Optional[dict] = None, envs: Optional[int] = None, lp: Optional[int] = None,
                 profile: Optional[str] = None, blind_concurrency: int = 16, max_model_len: int = 32768,
                 deadline_s: Optional[float] = None, argv: Optional[Sequence[str]] = None):
        self.ctl, self.reach_index, self.served = ctl, reach_index, served
        self.pool, self.server = pool, server
        self.bfs_table, self.bfs_path = bfs_table, bfs_path
        self.start, self.tier, self.task_set = start, tier, task_set
        self.model_dir, self.adapter = model_dir, adapter
        self.envs, self.lp, self.profile = envs, lp, profile
        self.blind_concurrency = int(blind_concurrency)
        self.max_model_len = int(max_model_len)
        self.deadline_s = deadline_s
        self.argv = list(argv or [])
        self.code = code_sha256()
        self.server_ready_s: Optional[float] = None
        self.server_manifest: Optional[Dict[str, Any]] = None
        self.versions = serve_versions()
        self.server_dead = False  # a failed model call found the server not alive: no new tasks, no further arms
        self.server_alive_cached = False
        self._reach_cache: Dict[str, Tuple[list, Any]] = {}
        self._rlock = threading.Lock()

    # -- helpers
    def reach_of(self, tid: str):
        with self._rlock:
            if tid not in self._reach_cache:
                pos = self.reach_index.positions(tid)
                self._reach_cache[tid] = (list(pos), kin_env.ReachSet.from_positions(pos))
            return self._reach_cache[tid]

    def _call(self, wid: int, cmd: Dict[str, Any], op: str) -> Dict[str, Any]:
        r = self.pool.call(wid, cmd, self.ctl.bounded(thor_worker.TIMEOUTS[op]))
        if not r.get("ok"):
            if self.ctl.stopped():
                raise Abandon()
            raise TaskError("worker", f"{op}: {r.get('error')}" + (" (timeout)" if r.get("timeout") else ""))
        return r

    def _model(self, ar: ArmRun, client: ChatClient, convo: Convo) -> Dict[str, Any]:
        body = convo.body()
        resp, lat, retries, errs = client.chat(body)
        try:
            choice = resp["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise TaskError("api", f"bad response {e!r}: {str(resp)[:200]}")
        usage = resp.get("usage") or {}
        with ar.lock:
            ar.steps += 1
            ar.max_images = max(ar.max_images, convo.n_images)
            if isinstance(usage.get("prompt_tokens"), int):
                ar.max_prompt = max(ar.max_prompt, usage["prompt_tokens"])
        return {"reply": text, "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"), "finish_reason": choice.get("finish_reason"),
                "latency_vlm_s": round(lat, 4), "retries": retries, "retry_errors": errs}

    def _parse(self, arm: str, reply: str) -> Optional[str]:
        if arm in ("c", "blind"):
            return tvr_text.regex_action(reply)
        if arm == "u":
            return tvr_text.parse_action(reply)
        return reply[len("Action: "):] if reply.startswith("Action: ") and reply[8:] in ACTIONS else None

    def _dump_row(self, ar: ArmRun, tid: str, t: int, convo: Convo, m: Dict[str, Any]) -> Dict[str, Any]:
        return {"task": tid, "task_id": tid, "step": t, "t": t, "arm": ar.arm, "n_images": convo.n_images,
                "prompt_tokens": m.get("prompt_tokens"), "model": self.served, "max_tokens": ar.max_tokens,
                "structured_outputs": convo.tail_obj.get("structured_outputs"), "messages": list(convo.red)}

    # -- one image-arm / replay episode
    def episode_image(self, ar: ArmRun, wid: int, task: Dict[str, Any], attempt: int,
                      client: Optional[ChatClient]):
        tid = task["task_id"]
        replay = ar.arm == "replay"
        cap = pilot_cfg.cap_of(task)
        positions, R = self.reach_of(tid)
        if replay:
            img_dir: Optional[Path] = ar.images_dir / tid
        elif tid in ar.keep_ids:
            img_dir = ar.run_dir / "images" / tid
        else:
            img_dir = None

        def ipath(name: str) -> Optional[str]:
            return str(img_dir / name) if img_dir is not None else None

        def rel(name: str) -> Optional[str]:
            if img_dir is None:
                return None
            return str(img_dir / name) if replay else f"images/{tid}/{name}"

        inline = not replay
        t_task = time.time()
        r = self._call(wid, {"op": "reset", "task": task, "reach": positions,
                             "images": {"inline": inline, "cur": ipath("step_00.png"), "tgt": ipath("target.png")}},
                       "reset")
        used = r.get("reach_used") or {}
        if used.get("source") != "frozen" or used.get("digest") != R.digest:
            raise TaskError("protocol", f"frozen reach set not in force: {used}")
        states = [r["state"]]
        valids = [list(r["valid"])]
        texts = [r["obs_str"]]
        images = [{"cur_sha256": r["cur"]["sha256"], "cur_path": rel("step_00.png")}]
        target_img = {"sha256": r["tgt"]["sha256"], "path": rel("target.png")}
        kin = KinShadow(R)
        kin.check_valid(0, states[0], valids[0])
        convo = None
        tgt_pair = None
        if not replay:
            convo = Convo(self.served, ar.max_tokens, ar.regex)
            tgt_pair = (image_part(r["tgt"]["png_b64"]),
                        {"type": "image_url", "sha256": r["tgt"]["sha256"], "path": rel("target.png")})
            convo.add_user_images(r["obs_str"], (image_part(r["cur"]["png_b64"]),
                                                 {"type": "image_url", "sha256": r["cur"]["sha256"],
                                                  "path": rel("step_00.png")}), tgt_pair)
        public = ar.public[tid]["actions"] if replay else None
        n_turns = min(cap, len(public)) if replay else cap
        actions: List[str] = []
        replies: List[str] = []
        succ: List[Optional[bool]] = []
        calls: List[Dict[str, Any]] = []
        dumps: List[Dict[str, Any]] = []
        last_dump: Optional[Dict[str, Any]] = None
        n_blocked = n_unparse = n_regex_bad = 0
        stopped = False
        keep = tid in ar.keep_ids
        for t in range(n_turns):
            if self.ctl.stopped():
                raise Abandon()
            if replay:
                a0 = public[t]
                m: Dict[str, Any] = {"reply": f"Action: {a0}", "prompt_tokens": None, "completion_tokens": None,
                                     "finish_reason": None, "latency_vlm_s": None, "retries": 0}
                with ar.lock:
                    ar.steps += 1
            else:
                m = self._model(ar, client, convo)
            reply = m["reply"]
            parsed = self._parse(ar.arm, reply)
            a = parsed or "InvalidAction"
            if a == "InvalidAction":
                n_unparse += 1
            if ar.arm == "c" and parsed is None:
                n_regex_bad += 1
            if convo is not None and keep:
                row = self._dump_row(ar, tid, t, convo, m)
                if t in DUMP_TURNS:
                    dumps.append(row)
                else:
                    last_dump = row
            if convo is not None:
                convo.add_assistant(reply)
            last = t == n_turns - 1
            t0 = time.perf_counter()
            rs = self._call(wid, {"op": "step", "action": a,
                                  "images": {"inline": inline, "cur": ipath(f"step_{t + 1:02d}.png"), "skip": last}},
                            "step")
            lat_env = time.perf_counter() - t0
            info = rs.get("info") or {}
            prev = states[-1]
            states.append(rs["state"])
            s = info.get("action_success") if a in NON_STOP else None
            succ.append(s)
            if a in NON_STOP and s is False:
                n_blocked += 1
            kin.check_step(t, prev, a, rs["state"], s)
            actions.append(a)
            replies.append(reply)
            calls.append({"task_id": tid, "t": t, "arm": ar.arm, "n_images": (convo.n_images if convo else None),
                          "prompt_tokens": m.get("prompt_tokens"), "completion_tokens": m.get("completion_tokens"),
                          "finish_reason": m.get("finish_reason"), "reply": reply, "action": a,
                          "regex_ok": (parsed is not None) if ar.arm == "c" else None,
                          "parse_ok": (parsed is not None) if ar.arm == "u" else None,
                          "valid": valids[t], "action_in_valid": a in valids[t], "action_success": s,
                          "pose_before": list(kin_env.pose_of(prev)), "pose_after": list(kin_env.pose_of(rs["state"])),
                          "cur_sha256": images[t]["cur_sha256"], "latency_vlm_s": m.get("latency_vlm_s"),
                          "latency_env_s": round(lat_env, 4), "retries": m.get("retries"), "utc": utc(),
                          "error": None})
            if a == "Stop":
                stopped = True
                break
            if rs.get("done") or last:
                break
            valids.append(list(rs["valid"]))
            texts.append(rs["obs_str"])
            kin.check_valid(t + 1, rs["state"], rs["valid"])
            cur = rs.get("cur") or {}
            name = f"step_{t + 1:02d}.png"
            images.append({"cur_sha256": cur.get("sha256"), "cur_path": rel(name)})
            if convo is not None:
                convo.add_user_images(rs["obs_str"], (image_part(cur["png_b64"]),
                                                      {"type": "image_url", "sha256": cur.get("sha256"),
                                                       "path": rel(name)}), tgt_pair)
        if last_dump is not None:
            dumps.append(last_dump)
        traj = self._trajectory(ar, task, cap=cap, states=states, actions=actions, replies=replies, succ=succ,
                                valids=valids, texts=texts, images=images, target_img=target_img, stopped=stopped,
                                n_blocked=n_blocked, n_unparse=n_unparse, n_regex_bad=n_regex_bad, R=R,
                                reach_live_equal=r.get("reach_equal"), kin=kin, worker=wid, attempt=attempt,
                                wall=time.time() - t_task, reach_used=used)
        return traj, calls, dumps

    # -- one blind episode (kin_env, no images)
    def episode_blind(self, ar: ArmRun, task: Dict[str, Any], attempt: int, client: ChatClient):
        tid = task["task_id"]
        cap = pilot_cfg.cap_of(task)
        positions, R = self.reach_of(tid)
        t_task = time.time()
        ep = kin_env.KinEpisode(task, positions)
        o = ep.reset()
        convo = Convo(self.served, ar.max_tokens, ar.regex)
        convo.add_user_text(o["obs_str"].replace(tvr_text.IMAGE_PH, ""))
        states, valids, texts = [o["state"]], [list(o["valid"])], [o["obs_str"]]
        actions: List[str] = []
        replies: List[str] = []
        succ: List[Optional[bool]] = []
        calls: List[Dict[str, Any]] = []
        dumps: List[Dict[str, Any]] = []
        last_dump = None
        n_unparse = n_regex_bad = n_blocked = 0
        stopped = False
        keep = tid in ar.keep_ids
        for t in range(cap):
            if self.ctl.stopped():
                raise Abandon()
            m = self._model(ar, client, convo)
            reply = m["reply"]
            parsed = self._parse(ar.arm, reply)
            a = parsed or "InvalidAction"
            if a == "InvalidAction":
                n_unparse += 1
            if parsed is None:
                n_regex_bad += 1
            if keep:
                row = self._dump_row(ar, tid, t, convo, m)
                if t in DUMP_TURNS:
                    dumps.append(row)
                else:
                    last_dump = row
            convo.add_assistant(reply)
            t0 = time.perf_counter()
            so = ep.step(a)
            lat_env = time.perf_counter() - t0
            prev = states[-1]
            states.append(so["state"])
            s = so["info"].get("action_success") if a in NON_STOP else None
            succ.append(s)
            if a in NON_STOP and s is False:
                n_blocked += 1
            actions.append(a)
            replies.append(reply)
            calls.append({"task_id": tid, "t": t, "arm": ar.arm, "n_images": 0,
                          "prompt_tokens": m.get("prompt_tokens"), "completion_tokens": m.get("completion_tokens"),
                          "finish_reason": m.get("finish_reason"), "reply": reply, "action": a,
                          "regex_ok": parsed is not None, "parse_ok": None, "valid": valids[t],
                          "action_in_valid": a in valids[t], "action_success": s,
                          "pose_before": list(kin_env.pose_of(prev)), "pose_after": list(kin_env.pose_of(so["state"])),
                          "cur_sha256": None, "latency_vlm_s": m.get("latency_vlm_s"),
                          "latency_env_s": round(lat_env, 6), "retries": m.get("retries"), "utc": utc(), "error": None})
            if a == "Stop" or so.get("done"):
                stopped = a == "Stop"
                break
            if t == cap - 1:
                break
            valids.append(list(so["valid"]))
            texts.append(so["obs_str"])
            convo.add_user_text(so["obs_str"].replace(tvr_text.IMAGE_PH, ""))
        if last_dump is not None:
            dumps.append(last_dump)
        traj = self._trajectory(ar, task, cap=cap, states=states, actions=actions, replies=replies, succ=succ,
                                valids=valids, texts=texts, images=None, target_img=None, stopped=stopped,
                                n_blocked=n_blocked, n_unparse=n_unparse, n_regex_bad=n_regex_bad, R=R,
                                reach_live_equal=None, kin=None, worker=None, attempt=attempt,
                                wall=time.time() - t_task)
        return traj, calls, dumps

    def _trajectory(self, ar: ArmRun, task, *, cap, states, actions, replies, succ, valids, texts, images,
                    target_img, stopped, n_blocked, n_unparse, n_regex_bad, R, reach_live_equal, kin, worker,
                    attempt, wall, reach_used=None) -> Dict[str, Any]:
        final, target = states[-1], task["target"]
        success = bool(stopped and is_exact_match(final, target))
        # reach_digest: the digest the WORKER reported for the set in force after the injection (image arms; it
        # was checked == R.digest at the reset); the blind arm uses R itself (kin_env on the frozen positions)
        used = None
        if reach_used is not None:
            used = {k: reach_used.get(k) for k in ("source", "n", "digest")}
        v = {"schema": TRAJ_SCHEMA, "arm": ar.arm, "category": pilot_cfg.category_of(task), "cap": cap,
             "states": states, "replies": replies, "action_success": succ, "valid": valids, "user_texts": texts,
             "images": images, "target": target_img, "n_blocked": n_blocked, "n_unparseable": n_unparse,
             "regex_mismatch": n_regex_bad if ar.arm in ("c", "blind") else 0,
             "reach_digest": (used or {}).get("digest") or R.digest, "reach_used": used,
             "reach_live_equal": reach_live_equal, "kin_equal": None if kin is None else kin.equal,
             "kin_first_mismatch": None if kin is None else kin.first,
             "kin_mismatch_counts": None if kin is None else dict(kin.counts), "worker": worker, "attempt": attempt,
             "wall_s": round(wall, 3), "protocol_sha256": ar.protocol_sha}
        if ar.mech:
            v["withheld"] = score_tvr.WITHHELD
        return {"task_id": task["task_id"], "dataset": task["dataset"], "scene": task["scene"],
                "difficulty": task["difficulty"],
                "result": {"success": None if ar.mech else success, "stopped": bool(stopped),
                           "num_steps": len(actions), "n_invalid_actions": n_unparse,
                           "final_pos_error": position_error(final, target),
                           "final_rot_error": rotation_error(final, target),
                           "final_hor_error": horizon_error(final, target)},
                "actions": actions, "start": task["start"], "target": task["target"],
                "raw_first_response": (replies[0] if replies else "")[:300],
                "raw_last_response": (replies[-1] if replies else "")[:300], "vpd": v}

    def write_task(self, ar: ArmRun, tid: str, traj, calls, dumps) -> None:
        d = ar.task_dir(tid)
        d.mkdir(parents=True, exist_ok=True)
        write_text_atomic(d / "calls.jsonl", "".join(json.dumps(c) + "\n" for c in calls))
        if dumps:
            write_text_atomic(d / "prompt_dump.jsonl", "".join(json.dumps(x) + "\n" for x in dumps))
        elif (d / "prompt_dump.jsonl").exists():
            (d / "prompt_dump.jsonl").unlink()
        write_json_atomic(d / "trajectory.json", traj)
        with ar.lock:
            ar.n_written += 1

    def _fail(self, ar: ArmRun, tid: str, e: BaseException, attempt: int) -> None:
        with ar.lock:
            ar.failures.append({"task_id": tid, "attempt": attempt, "error": str(e)[:1000], "utc": utc()})
        log(f"{ar.name}: task {tid} FAILED after attempt {attempt}: {str(e)[:300]}")

    def task_with_retry(self, ar: ArmRun, task, *, wid: Optional[int], client, first_attempt: int = 1) -> str:
        """'ok' | 'failed' | 'abandoned' | 'server_dead' | 'slot_dead' (image arms: the worker could not be
        restarted after the last attempt) | 'slot_dead_retry' (... after attempt 1: the task is requeued for
        attempt 2 on another slot; attempt 1's error goes to ar.requeued -> results requeued_tasks, not a task
        failure).  At most 2 attempts in total, also across slots (first_attempt)."""
        tid = task["task_id"]
        for attempt in range(int(first_attempt), 3):
            try:
                if ar.arm == "blind":
                    traj, calls, dumps = self.episode_blind(ar, task, attempt, client)
                else:
                    traj, calls, dumps = self.episode_image(ar, wid, task, attempt, client)
                self.write_task(ar, tid, traj, calls, dumps)
                return "ok"
            except Abandon:
                return "abandoned"
            except Exception as e:  # noqa: BLE001  (TaskError and harness errors alike)
                if self.ctl.stopped():
                    return "abandoned"
                kind = getattr(e, "kind", "harness")
                if kind in ("api", "server_dead"):
                    with ar.lock:
                        ar.api_errors += 1
                log(f"{ar.name}: task {tid} attempt {attempt} error ({kind}): {str(e)[:300]}")
                if kind == "server_dead":
                    if not self.server_dead:
                        log(f"{ar.name}: the model server is dead: no new tasks; the remaining arms are not run")
                    self.server_dead = True
                    self._fail(ar, tid, e, attempt)
                    return "server_dead"
                if kind == "harness":
                    log(traceback.format_exc()[-1500:])
                slot_dead = False
                if wid is not None and kind != "api":
                    rr = self.pool.restart(wid)
                    slot_dead = not rr.get("ok")
                if attempt == 2:
                    self._fail(ar, tid, e, attempt)
                    return "slot_dead" if slot_dead else "failed"
                if slot_dead:
                    with ar.lock:
                        ar.requeued.append({"task_id": tid, "attempt": attempt, "wid": wid, "error": str(e)[:1000],
                                            "utc": utc()})
                    return "slot_dead_retry"
        return "failed"

    def halted(self) -> bool:
        return self.ctl.stopped() or self.server_dead

    # -- arms
    def run_image_arm(self, ar: ArmRun) -> None:
        # queue items: scene groups of (task, first_attempt) pairs
        q: List[List[Tuple[Dict[str, Any], int]]] = [[(t, 1) for t in g] for g in scene_groups(ar.todo)]
        cv = threading.Condition()
        busy = {"n": 0}  # slots holding a group: an idle slot waits for them (a dead slot may requeue its rest)
        replay = ar.arm == "replay"
        base_url = None if replay else self.server.base_url

        def take():
            with cv:
                while not q and busy["n"] > 0 and not self.halted():
                    cv.wait(0.5)
                if not q or self.halted():
                    return None
                busy["n"] += 1
                return q.pop(0)

        def release() -> None:
            with cv:
                busy["n"] -= 1
                cv.notify_all()

        def requeue(rest, wid: int, why: str) -> None:
            if rest:
                with cv:
                    q.insert(0, rest)
                    cv.notify_all()
            what = (f"slot thread {wid} exits ({why})" if why == "harness error"
                    else f"worker slot {wid} is dead ({why})")
            log(f"{ar.name}: {what}; {len(rest)} tasks requeued")

        def slot(wid: int) -> None:
            client = None if replay else ChatClient(base_url, self.ctl, alive=self.server_alive)
            try:
                while not self.halted():
                    g = take()
                    if g is None:
                        return
                    try:
                        for i, (task, first) in enumerate(g):
                            if self.halted():
                                return
                            if self.ctl.left() < NEW_TASK_MIN_S:
                                self.ctl.new_tasks_stopped = True
                                return
                            try:
                                st = self.task_with_retry(ar, task, wid=wid, client=client, first_attempt=first)
                            except Exception as e:  # noqa: BLE001  (a harness bug outside the task: keep the group)
                                log(f"{ar.name}: slot {wid} harness error: {e!r}\n{traceback.format_exc()[-1500:]}")
                                requeue(g[i:], wid, "harness error")
                                return
                            if st == "slot_dead":
                                requeue(g[i + 1:], wid, "restart failed")
                                return
                            if st == "slot_dead_retry":
                                requeue([(task, 2)] + list(g[i + 1:]), wid, "restart failed")
                                return
                    finally:
                        release()
            finally:
                if client is not None:
                    client.close()

        self._run_threads(ar, [threading.Thread(target=slot, args=(w,), daemon=True, name=f"slot{w}")
                               for w in range(self.pool.n) if self.pool.alive(w)])

    def run_blind_arm(self, ar: ArmRun) -> None:
        q = list(ar.todo)
        qlock = threading.Lock()

        def thread() -> None:
            client = ChatClient(self.server.base_url, self.ctl, alive=self.server_alive)
            try:
                while not self.halted():
                    with qlock:
                        task = q.pop(0) if q else None
                    if task is None:
                        return
                    if self.ctl.left() < NEW_TASK_MIN_S:
                        self.ctl.new_tasks_stopped = True
                        return
                    try:
                        self.task_with_retry(ar, task, wid=None, client=client)
                    except Exception as e:  # noqa: BLE001  (a harness bug outside the task)
                        log(f"{ar.name}: blind thread harness error: {e!r}\n{traceback.format_exc()[-1500:]}")
                        self._fail(ar, task["task_id"], e, 0)
            finally:
                client.close()

        self._run_threads(ar, [threading.Thread(target=thread, daemon=True, name=f"blind{i}")
                               for i in range(max(1, self.blind_concurrency))])

    def _run_threads(self, ar: ArmRun, threads: List[threading.Thread]) -> None:
        ar.ran = True
        ar.n_slots = len(threads)
        mon = CpuMon().start()
        t0 = time.monotonic()
        for th in threads:
            th.start()
        try:
            while any(th.is_alive() for th in threads):
                for th in threads:
                    th.join(0.25)
                if self.ctl.left() <= 0 and not self.ctl.stop.is_set():
                    log(f"{ar.name}: soft deadline reached; abandoning running tasks")
                    self.ctl.deadline_hit = True
                    self.ctl.stop.set()
                    self.fast_shutdown()
                if self.ctl.stop.is_set():
                    for th in threads:
                        th.join(0.5)
                    break
        finally:
            ar.wall += time.monotonic() - t0
            ar.cpu = mon.stop()

    def fast_shutdown(self) -> None:
        """pool.close(fast=True), then server.proc.kill() if alive, then server.__exit__ (contract 5.2.2)."""
        if self.pool is not None:
            try:
                self.pool.close(fast=True)
            except Exception:  # noqa: BLE001
                pass
        self.kill_server()

    def kill_server(self) -> None:
        srv = self.server
        if srv is None:
            return
        self.server_alive_cached = self.server_alive()
        proc = getattr(srv, "proc", None)
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            srv.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        self.server = None

    def normal_shutdown(self) -> None:
        """quit the workers (20 s), then VLLMServer.__exit__ (terminate, wait up to 60 s)."""
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception:  # noqa: BLE001
                pass
        if self.server is not None:
            self.server_alive_cached = self.server_alive()
            try:
                self.server.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self.server = None

    def server_alive(self) -> bool:
        srv = self.server
        if srv is None:
            return bool(getattr(self, "server_alive_cached", False))
        try:
            return bool(srv.alive())
        except Exception:  # noqa: BLE001
            return False

    # -- finalize
    def _renderer(self, ar: ArmRun) -> Optional[Dict[str, Any]]:
        if ar.arm == "blind":
            return {"none": "kin_env (no rendering)"}
        ready = self.pool.ready() if self.pool is not None else []
        if self.pool is not None and not ready:
            ready = [r for r in self.pool.readys if r]
        want_sp = sha256_bytes(tvr_text.SYSTEM_PROMPT.encode("utf-8"))
        want_tvb = {k: pilot_cfg.FROZEN["tvrbench"]["files_sha256"][k] for k in thor_worker.CODE_FILES_TVRBENCH}
        workers = [{"wid": r.get("wid"), "profile": r.get("profile"), "lp": r.get("lp"),
                    "font_mode": (r.get("font") or {}).get("mode"), "font_sha256": (r.get("font") or {}).get("sha256"),
                    "pid": r.get("pid"), "startup_s": r.get("startup_s"), "ai2thor": r.get("ai2thor"),
                    "settings": r.get("settings"),
                    # the request's system prompt is tvr_text's copy; the user texts / valid lists come from the
                    # worker's TVRBench: both must be the pinned ones
                    "system_prompt_ok": r.get("system_prompt_sha256") == want_sp,
                    "tvrbench_files_ok": r.get("tvrbench_files_sha256") == want_tvb} for r in ready]
        fonts = sorted({str(w["font_sha256"]) for w in workers})
        want_font = pilot_cfg.FROZEN["render"]["badge_font_sha256"]
        ok = bool(workers) and all(w["profile"] == self.profile and str(w["lp"]) == str(self.lp)
                                   and w["font_mode"] == "pinned" and w["font_sha256"] == want_font
                                   and w["system_prompt_ok"] and w["tvrbench_files_ok"] for w in workers)
        live = sum(1 for w in range(self.pool.n) if self.pool.alive(w)) if self.pool is not None else 0
        return {"profile": self.profile, "lp": self.lp, "envs": self.envs, "n_workers_live": live,
                "font_sha256": fonts[0] if len(fonts) == 1 else fonts, "build": pilot_cfg.FROZEN["render"]["build"],
                "workers_ok": ok, "workers": workers,
                "start_history": (self.pool.history if self.pool is not None else [])}

    def renderer_ready(self, ar: ArmRun) -> bool:
        """The blind arm needs no renderer; an image arm / the replay needs a measured one (score_tvr's rule on the
        renderer block finalize writes)."""
        return ar.arm not in score_tvr.RENDER_ARMS or score_tvr.renderer_measured(self._renderer(ar))

    def _kin_reach(self, ar: ArmRun) -> Tuple[Optional[dict], dict]:
        kin = {"tasks_equal": 0, "tasks_unequal": 0, "valid_list_mismatches": 0, "pose_or_success_mismatches": 0,
               "first": []}
        live_eq, live_uneq, dig_ok = 0, [], True
        for tid in ar.task_ids:
            tr = read_json(ar.task_dir(tid) / "trajectory.json")
            if tr is None:
                continue
            v = tr.get("vpd") or {}
            if v.get("reach_digest") != self.reach_index.digest(tid):
                dig_ok = False
            if v.get("reach_live_equal") is True:
                live_eq += 1
            elif v.get("reach_live_equal") is False:
                live_uneq.append(tid)
            if v.get("kin_equal") is None:
                continue
            c = v.get("kin_mismatch_counts") or {}
            if v["kin_equal"]:
                kin["tasks_equal"] += 1
            else:
                kin["tasks_unequal"] += 1
                if len(kin["first"]) < 20:
                    kin["first"].append(dict(task_id=tid, **(v.get("kin_first_mismatch") or {})))
            kin["valid_list_mismatches"] += int(c.get("valid_list", 0))
            kin["pose_or_success_mismatches"] += int(c.get("pose", 0)) + int(c.get("success", 0))
        reach = {"index_sha256": self.reach_index.sha256, "root": str(getattr(self.reach_index, "root", "")),
                 "frozen_digest_equal_index": dig_ok, "live_equal": live_eq, "live_unequal": live_uneq}
        return (None if ar.arm == "blind" else kin), reach

    def _audit(self, ar: ArmRun, partial: bool) -> Dict[str, Any]:
        if partial:
            return {"image_tokens": "skipped_deadline", "prompt_parity": "skipped_deadline"}
        if ar.arm in ("replay", "blind"):
            why = "replay (no model)" if ar.arm == "replay" else "blind arm (no images)"
            return {"image_tokens": {"skipped": why}, "prompt_parity": {"skipped": why}}
        out: Dict[str, Any] = {}
        fake = bool(getattr(self.server, "fake", False)) or bool(getattr(self, "fake_server", False))
        try:
            pdir = None if fake else processor_dir_for(self.start, self.model_dir)
            out["image_tokens"] = (fake_image_token_audit(ar.run_dir) if fake
                                   else image_token_audit(ar.run_dir, pdir))
        except Exception as e:  # noqa: BLE001
            out["image_tokens"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        try:
            out["prompt_parity"] = (fake_prompt_parity(ar.run_dir) if fake
                                    else hf_prompt_parity(ar.run_dir, processor_dir_for(self.start, self.model_dir)))
        except Exception as e:  # noqa: BLE001
            out["prompt_parity"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        # an audit over nothing never passes (run_vpd.hf_prompt_parity's all([]) would be True)
        pp, it = out.get("prompt_parity"), out.get("image_tokens")
        if isinstance(pp, dict) and "error" not in pp and not pp.get("n"):
            pp["equal"], pp["empty"] = False, True
        if isinstance(it, dict) and "error" not in it and not it.get("n"):
            it["empty"] = True
        return out

    def settings(self, ar: ArmRun) -> Dict[str, Any]:
        s = dict(probe_lvp.jsafe(PROTO))
        s.update(arm=ar.arm, max_tokens_used=ar.max_tokens, regex_used=ar.regex, max_model_len_used=self.max_model_len,
                 envs=self.envs if ar.arm != "blind" else None, lp=self.lp if ar.arm != "blind" else None,
                 blind_concurrency_used=self.blind_concurrency if ar.arm == "blind" else None,
                 request_fields=sorted(["model", "messages"] + list(Convo(self.served or "", ar.max_tokens,
                                                                          ar.regex).tail_obj)))
        return s

    def perf(self, ar: ArmRun) -> Dict[str, Any]:
        """Throughput over EVERY process that ran this arm (RUN/perf_parts.jsonl, one segment per process with the
        same protocol_sha256): steps = sum, wall_s = sum, steps_per_s = steps / wall_s, cgroup_cores_mean =
        wall-weighted mean, cgroup_cores_max = max.  This process's segment is appended first (when it ran)."""
        p = ar.run_dir / PERF_PARTS
        if ar.ran and (ar.steps > 0 or ar.n_written > 0 or ar.wall > 1.0):
            seg = {"protocol_sha256": ar.protocol_sha, "pid": os.getpid(), "utc": utc(), "steps": ar.steps,
                   "wall_s": round(ar.wall, 3), "cores_mean": ar.cpu.get("cores_mean"),
                   "cores_max": ar.cpu.get("cores_max"), "n_slots": ar.n_slots, "tasks_written": ar.n_written}
            with open(p, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(seg) + "\n")
        segs = []
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                if s.get("protocol_sha256") == ar.protocol_sha:
                    segs.append(s)
        steps = sum(int(s.get("steps") or 0) for s in segs)
        wall = sum(float(s.get("wall_s") or 0.0) for s in segs)
        cw = [(float(s["cores_mean"]), float(s.get("wall_s") or 0.0)) for s in segs if s.get("cores_mean") is not None]
        cmax = [float(s["cores_max"]) for s in segs if s.get("cores_max") is not None]
        return {"steps": steps, "wall_s": round(wall, 3), "steps_per_s": (round(steps / wall, 4) if wall > 0 else None),
                "cgroup_cores_mean": (round(sum(c * w for c, w in cw) / sum(w for _, w in cw), 3)
                                      if cw and sum(w for _, w in cw) > 0 else None),
                "cgroup_cores_max": (max(cmax) if cmax else None), "segments": len(segs), "resumed": len(segs) > 1,
                "this_process": {"steps": ar.steps, "wall_s": round(ar.wall, 3),
                                 "steps_per_s": (round(ar.steps / ar.wall, 4) if ar.wall > 0 else None),
                                 "tasks_written": ar.n_written}}

    @staticmethod
    def resumed_from(ar: ArmRun) -> Optional[Dict[str, Any]]:
        """The invalidity record of the result file this finalize replaces (results_invalid / results_partial)."""
        for name in ("results_invalid.json", "results_partial.json"):
            old = read_json(ar.run_dir / name)
            if old is not None:
                return {"file": name, "utc_scored": old.get("utc_scored"), "api_errors": old.get("api_errors"),
                        "task_failures": old.get("task_failures"), "failed_tasks": old.get("failed_tasks"),
                        "server_alive_at_end": old.get("server_alive_at_end"), "server_dead": old.get("server_dead"),
                        "tasks_missing": len(old.get("tasks_missing") or []), "deadline": old.get("deadline"),
                        "previous": old.get("resumed_from")}
        return None

    def finalize(self, ar: ArmRun, partial: bool) -> Tuple[Dict[str, Any], bool]:
        n_calls = concat_jsonl(ar, "calls.jsonl")
        concat_jsonl(ar, "prompt_dump.jsonl")
        audit = self._audit(ar, partial)
        kin, reach = self._kin_reach(ar)
        trajs_steps = []
        for tid in ar.task_ids:
            tr = read_json(ar.task_dir(tid) / "trajectory.json")
            if tr is not None:
                trajs_steps.append(int(tr["result"]["num_steps"]))
        alive = True if ar.arm == "replay" else self.server_alive()
        perf = self.perf(ar)
        perf.update(vllm_ready_s=self.server_ready_s, max_images_per_request=ar.max_images,
                    max_prompt_tokens=ar.max_prompt,
                    mean_steps_per_task=(round(sum(trajs_steps) / len(trajs_steps), 4) if trajs_steps else None))
        ai2 = sorted({str(r.get("ai2thor")) for r in (self.pool.readys if self.pool is not None else []) if r})
        server = None
        if ar.arm != "replay":
            # VLLMServer.manifest() flat (the WB run_vpd convention: server.cfg / cmd / extra_args /
            # mm_processor_kwargs) and also under "manifest"; our keys win (no collision with the manifest's)
            server = dict(self.server_manifest or {})
            server.update(manifest=self.server_manifest,
                          versions=dict(self.versions, ai2thor=(ai2[0] if len(ai2) == 1 else ai2)),
                          max_model_len=self.max_model_len, ready_s=self.server_ready_s,
                          served_model_name=self.served)
        meta = {"run_name": ar.name, "start": self.start, "code": self.served, "served_model_name": self.served, "tier": self.tier, "task_set": self.task_set,
                "mechanism_only": ar.mech, "model_dir": self.model_dir, "adapter": self.adapter, "audit": audit,
                "kin": kin, "reach": reach, "renderer": self._renderer(ar), "settings": self.settings(ar),
                "server": server, "code_fingerprint": self.code, "code_sha256": self.code["code_sha256"],
                "protocol_sha256": ar.protocol_sha, "wall_seconds": round(ar.wall, 3), "calls": n_calls,
                "api_errors": ar.api_errors, "task_failures": len(ar.failures), "failed_tasks": ar.failures,
                "requeued_tasks": ar.requeued, "server_alive_at_end": alive, "perf": perf, "bfs_path": self.bfs_path,
                "deadline_s": self.deadline_s,
                "deadline": {"hit": self.ctl.deadline_hit, "new_tasks_stopped": self.ctl.new_tasks_stopped},
                "server_dead": self.server_dead, "utc_end": utc()}
        prev = self.resumed_from(ar)
        if prev is not None:
            meta["resumed_from"] = prev
        meta["meta_keys"] = sorted(meta) + ["meta_keys"]
        bfs = None if (ar.mech or ar.arm == "replay") else self.bfs_table
        res, valid = score_tvr.score_run(ar.run_dir, task_ids=ar.task_ids, bfs_table=bfs,
                                         reach_index=self.reach_index, arm=ar.arm, mechanism_only=ar.mech,
                                         meta=meta, partial=partial)
        self.write_manifest(ar)
        log(f"{ar.name}: finalized {'PARTIAL' if partial else ('valid' if valid else 'INVALID')}: "
            f"{res.get('n_scored')}/{res.get('n_tasks')} tasks, missing {len(res.get('tasks_missing') or [])}, "
            f"failures {len(ar.failures)}, api errors {ar.api_errors}")
        return res, valid

    def write_manifest(self, ar: ArmRun) -> None:
        write_json_atomic(ar.run_dir / "manifest.json", {
            "schema": MANIFEST_SCHEMA, "utc": utc(), "run_name": ar.name, "arm": ar.arm, "start": self.start,
            "served_model_name": self.served, "tier": self.tier, "task_set": self.task_set, "n_tasks": len(ar.tasks),
            "server": self.server_manifest,
            "pool": None if self.pool is None or ar.arm == "blind" else
            {"E": self.pool.n, "L": self.pool.L, "profile": self.pool.profile, "ready": self.pool.readys,
             "history": self.pool.history},
            "settings": self.settings(ar), "code": self.code, "git": git_facts(), "adapter": self.adapter,
            "protocol_sha256": ar.protocol_sha, "argv": self.argv, "host": socket.gethostname(),
            "uid": os.getuid() if hasattr(os, "getuid") else None})


# --- CLI glue -------------------------------------------------------------------------------------------------------
def render_defaults(a, kind: str) -> Tuple[int, int, str]:
    rd = pilot_cfg.FROZEN["render"]
    envs = a.envs if a.envs is not None else rd["eval_envs" if kind == "eval" else "replay_envs"]
    lp = a.lp if a.lp is not None else rd["lp_threads"]
    prof = a.profile if a.profile is not None else rd["lvp_profile"]
    if prof is None:
        raise Refused("render.lvp_profile is not filled in pilot_cfg and --profile was not given")
    if envs is None or lp is None:
        raise Refused("render.eval_envs / replay_envs / lp_threads not filled in pilot_cfg; give --envs / --lp")
    envs, lp = int(envs), int(lp)
    if envs < 1 or lp < 1 or envs * lp > MAX_EL:
        raise Refused(f"E x L = {envs} x {lp} must be >= 1 and <= {MAX_EL} (the probe's recommended rule)")
    if prof not in lvp.LVP_PROFILES:
        raise Refused(f"unknown lavapipe profile {prof!r}")
    return envs, lp, prof


def make_pool(a, envs: int, lp: int, profile: str, out_dir: Path, houses: Path):
    kw: Dict[str, Any] = {}
    if a.fake_thor:
        fr = Path(a.fake_thor).resolve()
        (fr / "tvr_env" / "lib").mkdir(parents=True, exist_ok=True)
        (fr / "releases").mkdir(parents=True, exist_ok=True)
        kw = {"py": sys.executable, "extra_pythonpath": (str(fr),), "tvr_env": fr / "tvr_env",
              "releases": fr / "releases"}
    return thor_worker.ThorPool(envs, lp, profile, out_dir=out_dir, houses_dir=houses, repo=Path(a.repo),
                                code_root=paths.WORKTREE, **kw)


def load_reach(root) -> Any:
    try:
        return reach_mod.ReachIndex(Path(root))
    except Exception as e:  # noqa: BLE001
        raise Refused(f"cannot read the reach index {root}: {e!r}")


def check_reach(tasks, ridx) -> None:
    bad = []
    for t in tasks:
        try:
            ridx.digest(t["task_id"])
        except Exception:  # noqa: BLE001
            bad.append(t["task_id"])
    if bad:
        raise Refused(f"{len(bad)} tasks are not in the reach index (first {bad[:3]})")


def render_block(envs, lp, profile) -> Dict[str, Any]:
    return {"profile": profile, "lp": lp, "envs": envs,
            "font_sha256": pilot_cfg.FROZEN["render"]["badge_font_sha256"]}


def run_name(prefix: str, start: str, tier: str, arm: str) -> str:
    return f"{prefix}__{start}__{tier}__{arm}"


def served_name(start: str) -> str:
    from vpd import starts

    try:
        return starts.resolve(start)["code"] or start
    except Exception as e:  # noqa: BLE001
        raise Refused(f"unknown start {start!r}: {e!r}")


def start_server(a, served: str, log_path: Path, mml: int):
    if a.fake_server:
        from vpd.tvr import fake_server

        return fake_server.FakeServer(policy=a.fake_policy, stop_after=a.fake_stop_after, model=served)
    if a.base_url:
        return ExternalServer(a.base_url)
    from vpd.common.vllm_server_vpd import VLLMServer

    return VLLMServer(served, log=log_path, model_dir=a.model_dir, max_images=int(PROTO["max_images"]),
                      max_model_len=mml, gpu_memory_utilization=float(a.gpu_memory_utilization))


def exit_code(results: List[Tuple[bool, bool]], ctl: Ctl, n_pending: Optional[int] = None) -> int:
    """results: (valid, partial) per arm finalized in this process; n_pending: the arms this process had to run.
    4 if some arm is partial or was not finalized; else 1 if some arm is invalid; else 0 (every arm valid)."""
    if any(p for _, p in results) or (n_pending is not None and len(results) < n_pending):
        return EXIT_INCOMPLETE
    if any(not v for v, _ in results):
        return EXIT_INVALID
    return EXIT_OK


def deadline_partial(ar: ArmRun, ctl: Ctl, h: "Harness") -> bool:
    """An arm is PARTIAL (results_partial.json, exit 4) when the soft deadline passed or new tasks were stopped
    near it - whether inside the arm's threads or before / between arms (server or pool start-up, an earlier arm's
    finalize) - and some of its tasks have no trajectory, or (image arm / replay) its renderer was not measured (no
    pool start-up near the deadline, also when every trajectory is on disk: a --resume after a kill during
    finalize).  A dead server makes the arm invalid, not partial."""
    if ctl.left() <= 0 and not ctl.terminated:
        ctl.deadline_hit = True
    if h.server_dead or not (ctl.deadline_hit or ctl.new_tasks_stopped):
        return False
    return len(ar.done_ids()) < len(ar.task_ids) or not h.renderer_ready(ar)


def mark_near_deadline(ar: ArmRun, ctl: Ctl) -> None:
    """Before an arm: no task can start with fewer than NEW_TASK_MIN_S left (the slots' own rule, also applied
    when the arm would not start its threads at all)."""
    if ctl.left() <= 0:
        ctl.deadline_hit = True
    if ar.todo and ctl.left() < NEW_TASK_MIN_S:
        ctl.new_tasks_stopped = True


def reset_perf_parts(ar: ArmRun, resume: bool) -> None:
    if not resume and (ar.run_dir / PERF_PARTS).exists():
        (ar.run_dir / PERF_PARTS).unlink()


def cmd_eval(a, *, ctl: Optional[Ctl] = None, pool_factory: Optional[Callable] = None,
             server_factory: Optional[Callable] = None) -> int:
    ctl = ctl or Ctl(a.deadline_s)
    arms = [x for x in a.arms.split(",") if x]
    if not arms or len(set(arms)) != len(arms) or any(x not in pilot_cfg.ARMS for x in arms):
        raise Refused(f"--arms must be distinct items of {pilot_cfg.ARMS}, got {a.arms!r}")
    if "blind" in arms and arms[-1] != "blind":
        raise Refused("the blind arm runs after the image arms: put it last")
    if a.task_set not in EVAL_SETS:
        raise Refused(f"--task-set must be one of {EVAL_SETS}")
    if a.tier not in TIERS:
        raise Refused(f"--tier must be one of {TIERS}")
    if a.limit is not None and not a.allow_limit:
        raise Refused("--limit needs --allow-limit (never in a pilot eval)")
    if a.fake_server and a.base_url:
        raise Refused("--fake-server and --base-url are exclusive")
    mml = int(PROTO["max_model_len"])
    if a.max_model_len is not None:
        if int(a.max_model_len) != int(PROTO["max_model_len_fallback"]):
            raise Refused(f"--max-model-len only takes the registered fallback {PROTO['max_model_len_fallback']}")
        mml = int(a.max_model_len)
    adapter = None
    if a.adapter_meta is not None:
        if a.model_dir is None:
            raise Refused("--adapter-meta describes merged weights; give --model-dir too")
        adapter = adapter_info(Path(a.adapter_meta))
        if adapter["start"] != a.start:
            raise Refused(f"adapter of start {adapter['start']!r}, but --start is {a.start!r}")
        if a.tier == "raw" or adapter["tier"] != TIER_PHASE[a.tier]:
            raise Refused(f"--tier {a.tier} does not match the adapter phase {adapter['tier']!r}")
    elif a.tier != "raw":
        raise Refused(f"--tier {a.tier} needs --adapter-meta (and --model-dir)")
    if a.vllm_parity and not Path(a.vllm_parity).is_file():
        raise Refused(f"--vllm-parity {a.vllm_parity}: no such dry-run json")
    served = served_name(a.start)
    tvr_out = Path(a.tvr_out)
    tasks = resolve_tasks(a.task_set, tvr_out=tvr_out, tasks_dir=Path(a.tasks_dir))
    if a.limit is not None:
        tasks = tasks[:int(a.limit)]
    ridx = load_reach(a.reach or tvr_out / "reach" / "v1")
    check_reach(tasks, ridx)
    bfs_path = None
    bfs_table = None
    if not a.mechanism_only:
        bfs_path = str(a.bfs or tvr_out / "bfs" / f"{bfs_name(a.task_set)}_opt.json")
        bfs_table = read_json(bfs_path)
        if bfs_table is None:
            raise Refused(f"cannot read the BFS table {bfs_path}")
    image_arms = [x for x in arms if x in IMAGE_ARMS]
    envs = lp = prof = None
    if image_arms:
        envs, lp, prof = render_defaults(a, "eval")
    code = code_sha256()
    ids = [t["task_id"] for t in tasks]
    out_root = Path(a.out_root)
    ars: List[ArmRun] = []
    for arm in arms:
        proto = {"schema": PROTOCOL_SCHEMA, "arm": arm, "start": a.start, "code": served, "tier": a.tier,
                 "adapter_sha256": (adapter or {}).get("adapter_sha256"), "model_dir_is_merged": a.model_dir is not None,
                 "task_set": a.task_set, "task_ids_sha256": sha256_bytes("\n".join(ids).encode()),
                 "protocol": probe_lvp.jsafe(PROTO), "max_model_len_used": mml,
                 "render": render_block(envs, lp, prof) if arm in IMAGE_ARMS else None,
                 "blind_concurrency": int(a.blind_concurrency) if arm == "blind" else None,
                 "reach_index_sha256": ridx.sha256, "code_sha256": code["code_sha256"],
                 "mechanism_only": bool(a.mechanism_only)}
        name = run_name(a.name_prefix, a.start, a.tier, arm)
        ar = ArmRun(arm=arm, name=name, run_dir=out_root / name, tasks=tasks, protocol=proto,
                    mechanism_only=a.mechanism_only, keep=a.keep_images_per_category)
        preflight(ar, a.resume)
        ars.append(ar)
    pending = [ar for ar in ars if not ar.complete]
    for ar in ars:
        if ar.complete:
            log(f"{ar.name}: complete (valid results.json, equal protocol): skipped")
    if not pending:
        return EXIT_OK
    for ar in pending:
        ar.run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(ar.run_dir / "protocol.json", ar.protocol)
        write_json_atomic(ar.run_dir / "task_ids.json", ar.task_ids)
        reset_perf_parts(ar, a.resume)
    base = f"{a.name_prefix}__{a.start}__{a.tier}"
    h = Harness(ctl=ctl, reach_index=ridx, served=served, bfs_table=bfs_table, bfs_path=bfs_path, start=a.start,
                tier=a.tier, task_set=a.task_set, model_dir=a.model_dir, adapter=adapter, envs=envs, lp=lp,
                profile=prof, blind_concurrency=a.blind_concurrency, max_model_len=mml, deadline_s=a.deadline_s,
                argv=sys.argv)
    results: List[Tuple[bool, bool]] = []
    srv = (server_factory or (lambda: start_server(a, served, out_root / f"{base}__vllm.log", mml)))()
    h.server = srv  # before __enter__: a SIGTERM during the start-up still kills the server process
    try:
        t0 = time.time()
        try:
            srv.__enter__()
        except Exception as e:  # noqa: BLE001  (the payload may retry with the O9 fallback)
            write_json_atomic(out_root / f"{base}__server_error.json",
                              {"error": f"{type(e).__name__}: {e}"[:2000], "max_model_len": mml, "utc": utc()})
            log(f"server did not start: {e!r}")
            h.server = None
            return EXIT_INVALID
        h.fake_server = bool(getattr(srv, "fake", False))
        h.server_ready_s = getattr(srv, "ready_seconds", round(time.time() - t0, 1))
        h.server_manifest = srv.manifest() if hasattr(srv, "manifest") else None
        if a.vllm_parity:
            from vpd.tvr import parity_tvr

            out = out_root / f"{base}__vllm_parity.json"
            try:
                rep = parity_tvr.vllm_parity(a.start, a.vllm_parity, srv.base_url, out, served=served)
                log(f"vllm parity: equal {rep['equal']} ({rep['n']} cases)")
            except Exception as e:  # noqa: BLE001  (recorded; the smoke verdict / gate read the file)
                write_json_atomic(out, {"equal": False, "error": f"{type(e).__name__}: {e}"[:2000], "utc": utc()})
        img_pending = [ar for ar in pending if ar.arm in IMAGE_ARMS]
        if img_pending and ctl.left() >= NEW_TASK_MIN_S:  # no pool start-up when no task could start anyway
            houses = Path(a.houses or tvr_out / "houses")
            pool = (pool_factory(envs, lp, prof, out_root / f"{base}__pool") if pool_factory
                    else make_pool(a, envs, lp, prof, out_root / f"{base}__pool", houses))
            h.pool = pool
            try:
                pool.start()
            except thor_worker.PoolError as e:
                log(f"THOR pool did not start: {e}")
        elif img_pending:  # the image arms are not run and, without a measured renderer, cannot be valid
            ctl.new_tasks_stopped = True
            log(f"no THOR pool start-up: {ctl.left():.0f} s left (< {NEW_TASK_MIN_S:.0f} s)")
        # every pending arm is finalized (a result file each), also when it cannot run: the deadline passed / new
        # tasks were stopped (-> results_partial.json) or the server died in an earlier arm (-> results_invalid)
        for ar in pending:
            mark_near_deadline(ar, ctl)
            h.write_manifest(ar)
            if ctl.stopped() or ctl.new_tasks_stopped or h.server_dead:
                why = ("server dead" if h.server_dead else "soft deadline" if ctl.left() <= 0 or ctl.deadline_hit
                       else "stopped" if ctl.stop.is_set() else f"< {NEW_TASK_MIN_S:.0f} s left")
                log(f"{ar.name}: not run ({why}); {len(ar.todo)} of {len(ar.tasks)} tasks left")
            else:
                log(f"{ar.name}: {len(ar.todo)} of {len(ar.tasks)} tasks to run")
                if ar.arm in IMAGE_ARMS:
                    if h.pool is not None and any(h.pool.alive(w) for w in range(h.pool.n)):
                        h.run_image_arm(ar)
                else:
                    if h.pool is not None and not ctl.stopped():
                        h.pool.close()
                    h.run_blind_arm(ar)
            partial = deadline_partial(ar, ctl, h)
            _, valid = h.finalize(ar, partial)
            results.append((valid, partial))
        return exit_code(results, ctl, len(pending))
    finally:
        if ctl.terminated or ctl.deadline_hit:
            h.fast_shutdown()
        else:
            h.normal_shutdown()


def cmd_replay(a, *, ctl: Optional[Ctl] = None, pool_factory: Optional[Callable] = None) -> int:
    ctl = ctl or Ctl(a.deadline_s)
    if a.task_set not in REPLAY_SETS:
        raise Refused(f"--task-set must be one of {REPLAY_SETS}")
    tvr_out = Path(a.tvr_out)
    tasks = resolve_tasks(a.task_set, tvr_out=tvr_out, tasks_dir=Path(a.tasks_dir))
    ridx = load_reach(a.reach or tvr_out / "reach" / "v1")
    check_reach(tasks, ridx)
    try:
        public = probe_lvp.load_public(a.public, [t["task_id"] for t in tasks])
    except (OSError, ValueError, KeyError) as e:
        raise Refused(f"cannot read the public sequences {a.public}: {e!r}")
    envs, lp, prof = render_defaults(a, "replay")
    code = code_sha256()
    ids = [t["task_id"] for t in tasks]
    proto = {"schema": PROTOCOL_SCHEMA, "arm": "replay", "task_set": a.task_set,
             "task_ids_sha256": sha256_bytes("\n".join(ids).encode()), "public_sha256": sha256_file(a.public),
             "images_dir": str(Path(a.images_dir).resolve()), "protocol": probe_lvp.jsafe(PROTO),
             "render": render_block(envs, lp, prof), "reach_index_sha256": ridx.sha256,
             "code_sha256": code["code_sha256"], "mechanism_only": False}
    run_dir = Path(a.run_dir)
    ar = ArmRun(arm="replay", name=run_dir.name, run_dir=run_dir, tasks=tasks, protocol=proto,
                images_dir=Path(a.images_dir).resolve(), public=public)
    preflight(ar, a.resume)
    if ar.complete:
        log(f"{ar.name}: complete (valid results.json, equal protocol): skipped")
        return EXIT_OK
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(a.images_dir).mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "protocol.json", proto)
    write_json_atomic(run_dir / "task_ids.json", ids)
    reset_perf_parts(ar, a.resume)
    h = Harness(ctl=ctl, reach_index=ridx, served=None, task_set=a.task_set, envs=envs, lp=lp, profile=prof,
                deadline_s=a.deadline_s, argv=sys.argv)
    houses = Path(a.houses or tvr_out / "houses")
    pool = pool_factory(envs, lp, prof, run_dir / "pool") if pool_factory else make_pool(a, envs, lp, prof,
                                                                                        run_dir / "pool", houses)
    h.pool = pool
    try:
        if ctl.left() >= NEW_TASK_MIN_S:
            try:
                pool.start()
            except thor_worker.PoolError as e:
                log(f"THOR pool did not start: {e}")
        else:  # nothing is replayed and, without a measured renderer, the replay cannot be valid
            ctl.new_tasks_stopped = True
            log(f"no THOR pool start-up: {ctl.left():.0f} s left (< {NEW_TASK_MIN_S:.0f} s)")
        mark_near_deadline(ar, ctl)
        h.write_manifest(ar)
        if ctl.stopped() or ctl.new_tasks_stopped:
            log(f"{ar.name}: not run (soft deadline); {len(ar.todo)} of {len(ar.tasks)} tasks left")
        else:
            log(f"{ar.name}: {len(ar.todo)} of {len(ar.tasks)} tasks to replay")
            if any(pool.alive(w) for w in range(pool.n)):
                h.run_image_arm(ar)
        partial = deadline_partial(ar, ctl, h)
        _, valid = h.finalize(ar, partial)
        return exit_code([(valid, partial)], ctl, 1)
    finally:
        pool.close(fast=ctl.deadline_hit or ctl.terminated)


def cmd_rescore(a) -> int:
    run_dir = Path(a.run_dir)
    proto = read_json(run_dir / "protocol.json")
    ids = read_json(run_dir / "task_ids.json")
    old_name = next((n for n in score_tvr.RESULT_FILES if (run_dir / n).exists()), None)
    if proto is None or ids is None or old_name is None:
        raise Refused(f"{run_dir}: needs protocol.json, task_ids.json and a result file")
    old = read_json(run_dir / old_name) or {}
    meta = {k: old.get(k) for k in (old.get("meta_keys") or []) if k in old}
    meta["rescored"] = {"from": old_name, "utc": utc()}
    meta["meta_keys"] = sorted(set(meta) | {"meta_keys"})
    ridx = load_reach((old.get("reach") or {}).get("root") or Path(a.tvr_out) / "reach" / "v1")
    mech = bool(proto.get("mechanism_only"))
    bfs = None
    if proto.get("arm") != "replay" and not mech and old.get("bfs_path"):
        bfs = read_json(old["bfs_path"])
    _, valid = score_tvr.score_run(run_dir, task_ids=list(ids), bfs_table=bfs, reach_index=ridx,
                                   arm=proto.get("arm"), mechanism_only=mech, meta=meta,
                                   partial=old_name == "results_partial.json", overwrite_valid=True)
    return EXIT_OK if valid else EXIT_INVALID


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p):
        p.add_argument("--tvr-out", default=str(paths.TVR_OUT), help="OUT/tvr (split/, reach/v1, bfs/, houses/)")
        p.add_argument("--tasks-dir", default=str(lvp.TASKS_DIR), help="REPO/data/tasks")
        p.add_argument("--reach", default=None, help="default <tvr-out>/reach/v1")
        p.add_argument("--houses", default=None, help="default <tvr-out>/houses")
        p.add_argument("--repo", default=str(lvp.TVR_REPO))

    def render_args(p):
        p.add_argument("--envs", type=int, default=None)
        p.add_argument("--lp", type=int, default=None)
        p.add_argument("--profile", default=None)
        p.add_argument("--deadline-s", type=float, default=None, help="seconds from process start (soft)")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--fake-thor", default=None, help="tests: a fake ai2thor root (workers on this python)")

    e = sub.add_parser("eval")
    e.add_argument("--start", required=True)
    e.add_argument("--tier", required=True)
    e.add_argument("--arms", required=True)
    e.add_argument("--task-set", required=True)
    e.add_argument("--out-root", required=True)
    e.add_argument("--name-prefix", default="pilot")
    e.add_argument("--model-dir", default=None)
    e.add_argument("--adapter-meta", default=None)
    e.add_argument("--blind-concurrency", type=int, default=int(PROTO["blind_concurrency"]))
    e.add_argument("--mechanism-only", action="store_true")
    e.add_argument("--vllm-parity", default=None, help="the n2_train_vpd dry_run.json of this start")
    e.add_argument("--keep-images-per-category", type=int, default=2)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--allow-limit", action="store_true")
    e.add_argument("--base-url", default=None)
    e.add_argument("--fake-server", action="store_true", help="tests: an in-process fake_server.FakeServer")
    e.add_argument("--fake-policy", default="stop_at", help="tests: FakeServer policy")
    e.add_argument("--fake-stop-after", type=int, default=2, help="tests: FakeServer stop_at")
    e.add_argument("--gpu-memory-utilization", type=float, default=float(PROTO["gpu_memory_utilization"]))
    e.add_argument("--max-model-len", type=int, default=None, help="only the registered fallback (O9)")
    e.add_argument("--bfs", default=None, help="default <tvr-out>/bfs/<eval500|dev200>_opt.json")
    data_args(e)
    render_args(e)
    r = sub.add_parser("replay")
    r.add_argument("--task-set", required=True)
    r.add_argument("--public", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--images-dir", required=True)
    data_args(r)
    render_args(r)
    s = sub.add_parser("rescore")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--tvr-out", default=str(paths.TVR_OUT))
    return ap


_CTL: Optional[Ctl] = None


def _on_signal(signum, frame):  # noqa: ARG001
    """Only the FIRST SIGTERM / SIGINT raises Terminated (GNU timeout may deliver TERM twice: to the child and to
    its group); later ones are ignored so they cannot interrupt the fast shutdown in the finally blocks."""
    if _CTL is not None:
        if _CTL.terminated:
            return
        _CTL.terminated = True
        _CTL.stop.set()
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
    raise Terminated(f"signal {signum}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _CTL
    t0 = time.monotonic()
    a = build_parser().parse_args(argv)
    if a.cmd == "rescore":
        try:
            return cmd_rescore(a)
        except Refused as e:
            print(f"refused: {e}", file=sys.stderr)
            return EXIT_REFUSED
    _CTL = Ctl(a.deadline_s, t0=t0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        if a.cmd == "eval":
            return cmd_eval(a, ctl=_CTL)
        return cmd_replay(a, ctl=_CTL)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Terminated:
        # cmd_eval / cmd_replay already shut down fast in their finally blocks (pool killpg, server killed)
        log("terminated (SIGTERM): fast shutdown done, no results written")
        return EXIT_TERM


if __name__ == "__main__":
    sys.exit(main())
