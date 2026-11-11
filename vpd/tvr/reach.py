"""Frozen per-task start-pose reachable sets R(task): reach dump and index (CONTRACT_wave2 4.3, 5.1.6, 6.2, 6.3).

R(task) = what GetReachablePositions returns with the agent at task.start, right after TVRBench's own
ThorViewEnv._sync_reset(task), with ThorEnv's cache cleared just before the call (the thor_worker op `reach_task`).
It is computed ONCE here, in the data job, on two independent loads: per scene, pass A on one worker (force load,
then the scene's tasks in ascending task_id) and pass B on ANOTHER worker (force load, descending order).  A scene
file is written only when both passes finished; the frozen value is pass A's; equal_AB / stable record whether B
agreed (recorded, not a gate).  Scenes run largest-first (ProcTHOR first, then by task count) from one work queue,
one thread per worker; a complete scene file is skipped on resume.  index.json (and DONE) is written only when
every scene of the requested sets is complete; an identical existing index.json is kept byte-for-byte.

Failures: a failed pass gets a fresh worker (ThorPool.restart) and one retry; a second failure fails the scene.  A
worker whose restart fails (ThorPool.restart returns ok False, or a stand-in raises) is retired, and when that
happened before the pass's retry the pass goes back to the queue for another worker (the A/B rule still holds).
After the deadline (fewer than NO_NEW_PASS_S seconds left) no restart, retry or new pass starts; a restart starts
only with at least NO_NEW_PASS_S + the pool's startup_timeout seconds left (restarts are serialized), else the
worker is retired and the scene fails (deadline).  An unexpected error in the harness while it runs a pass (e.g. a
malformed reply) fails that pass and its scene (recorded in errors); the thread goes on with the next pass.
call_pose (the worker's pose at the GetReachablePositions call) is compared with task.start and recorded
(call_pose_ok; index.json call_pose_mismatch); not a gate.

Files: houses/<scene>.json (+ houses.json), reach/v1/scenes/<scene>.json, reach/v1/index.json, reach/v1/DONE.
Digest (CONTRACT 4.3) = kin_env.digest_cells of the (r2(x), r2(z)) cells.

CLI (PY_Q3 parent; the workers are TVR_PY thor_worker processes, tvreval's ThorPool):
  python -B -m vpd.tvr.reach houses --task-sets sft1600,eval500,dev200 --out OUT/tvr/houses
  python -B -m vpd.tvr.reach dump --task-sets sft1600,eval500,dev200 --out OUT/tvr/reach/v1 --envs E --lp L
         --profile P --houses OUT/tvr/houses --deadline-s S       (exit 0 complete, 4 incomplete (deadline),
                                                                    1 error, 143 SIGTERM)
      (no THOR worker is started when every scene file is already complete: only index.json / DONE are written)
  python -B -m vpd.tvr.reach check --out OUT/tvr/reach/v1          (exit 0 iff complete)

Python 3.10 and 3.12, stdlib only at import time (thor_worker is imported inside the dump CLI only).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from vpd import paths
from vpd.tvr import kin_env, lvp, pilot_cfg, tasks as T
from vpd.tvr.kin_env import ReachSet

SCHEMA = "vpd.tvr.reach/1"
INDEX_SCHEMA = SCHEMA  # index.json carries the module schema too (CONTRACT 6.3)
DEFAULT_ROOT = paths.TVR_OUT / "reach" / "v1"
DEFAULT_TIMEOUTS = {"load": 240.0, "reach_task": 240.0, "reset": 240.0, "step": 120.0}  # = thor_worker.TIMEOUTS
NO_NEW_PASS_S = 60.0  # no pass starts when fewer seconds than this remain before the deadline
STARTUP_S = 150.0  # = thor_worker.STARTUP_TIMEOUT_S: a restart's bound when the pool has no startup_timeout
NO_TIME = "no restart: too close to the deadline"
MAX_EL = 11  # E x L rule of the probe's summary.recommended (CONTRACT F4)


class Terminated(BaseException):
    """SIGTERM / SIGINT in the dump CLI."""


def digest(positions: List[Dict]) -> str:
    """CONTRACT 4.3 digest of GetReachablePositions' positions."""
    return ReachSet.from_positions(positions).digest


# --- index ------------------------------------------------------------------------------------------------------------
class ReachIndex:
    """Read side of reach/v1: index.json plus the scene files (loaded lazily, each checked against the sha256 that
    index.json records).  positions(id) are the frozen (pass A) raw positions of R(task)."""

    def __init__(self, root=None):
        self.root = Path(DEFAULT_ROOT if root is None else root)
        raw = (self.root / "index.json").read_bytes()
        self.sha256 = T.sha256_bytes(raw)
        self.doc = json.loads(raw)
        if self.doc.get("schema") != INDEX_SCHEMA:
            raise ValueError(f"{self.root}/index.json: schema {self.doc.get('schema')!r} != {INDEX_SCHEMA!r}")
        self.tasks: Dict[str, Dict] = self.doc["tasks"]
        self._scenes: Dict[str, Dict] = {}
        self._reach: Dict[str, ReachSet] = {}

    def _scene(self, scene: str) -> Dict:
        if scene not in self._scenes:
            p = self.root / "scenes" / f"{scene}.json"
            raw = p.read_bytes()
            want = self.doc["scene_files"].get(scene)
            if T.sha256_bytes(raw) != want:
                raise ValueError(f"{p}: sha256 differs from index.json ({want})")
            self._scenes[scene] = json.loads(raw)
        return self._scenes[scene]

    def __contains__(self, task_id: str) -> bool:
        return task_id in self.tasks

    def digest(self, task_id: str) -> str:
        return self.tasks[task_id]["digest"]

    def positions(self, task_id: str) -> List[Dict]:
        e = self.tasks[task_id]
        return self._scene(e["scene"])["sets"][e["digest"]]

    def reach(self, task_id: str) -> ReachSet:
        d = self.digest(task_id)
        if d not in self._reach:
            r = ReachSet.from_positions(self.positions(task_id))
            if r.digest != d:
                raise ValueError(f"{task_id}: stored positions have digest {r.digest}, index says {d}")
            self._reach[d] = r
        return self._reach[d]


# --- houses -----------------------------------------------------------------------------------------------------------
def houses(task_sets: Sequence[str], out=None, *, src=None, root=None, tasks_dir=None) -> Dict:
    """lvp.extract_procthor_houses for every ProcTHOR scene of the sets -> out/<scene>.json + out/houses.json
    {"src", "src_sha256", "scenes": {scene: {"line", "sha256", "bytes"}}} (no timestamp: a rerun rewrites the same
    bytes)."""
    out = Path(paths.TVR_OUT / "houses" if out is None else out)
    src = Path(lvp.PROCTHOR_TRAIN if src is None else src)
    scenes = sorted({t["scene"] for n in task_sets for t in T.task_set(n, root, tasks_dir)
                     if lvp.is_procthor(t["scene"])})
    got = lvp.extract_procthor_houses(src, scenes, out)
    doc = {"src": str(src), "src_sha256": T.sha256_file(src),
           "scenes": {s: {"line": got[s]["line"], "sha256": got[s]["sha256"], "bytes": got[s]["bytes"]}
                      for s in sorted(got)}}
    T.write_json(out / "houses.json", doc)
    return doc


# --- dump -------------------------------------------------------------------------------------------------------------
def _scene_order(by_scene: Mapping[str, List[Dict]]) -> List[str]:
    """ProcTHOR first, then by task count descending, then by name."""
    return sorted(by_scene, key=lambda s: (0 if lvp.is_procthor(s) else 1, -len(by_scene[s]), s))


def scene_complete(path: Path, need_ids: Iterable[str]) -> bool:
    """A scene file is complete when it parses, has both passes and every needed task with A and B entries whose
    sets are stored."""
    try:
        doc = T.read_json(path)
    except (OSError, ValueError):
        return False
    if doc.get("schema") != SCHEMA or {p.get("pass") for p in doc.get("passes", [])} != {"A", "B"}:
        return False
    tk, sets = doc.get("tasks", {}), doc.get("sets", {})
    for i in need_ids:
        e = tk.get(i)
        if not e or "A" not in e or "B" not in e or e["A"].get("digest") not in sets:
            return False
    return True


class _Dump:
    def __init__(self, by_scene, out: Path, pool, deadline, timeouts, log):
        self.by_scene, self.out, self.pool, self.deadline = by_scene, out, pool, deadline
        self.timeouts, self.log = timeouts, log
        self.lock = threading.Condition()
        self.items: List[List[Any]] = []  # [scene, pass, state] state: None | "taken" | "done" | "failed"
        self.assigned: Dict[str, Dict[str, int]] = {}
        self.results: Dict[str, Dict[str, Dict]] = {}
        self.errors: List[Dict] = []
        self.deadline_hit = False
        self.live = 0
        self.dead: set = set()
        self.written: List[str] = []
        self.requeued: List[Dict] = []
        self.n_todo = 0
        self.t0 = time.monotonic()
        self.restart_lock = threading.Lock()
        self.restart_s = float(getattr(pool, "startup_timeout", STARTUP_S))

    # time
    def left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()

    def bounded(self, op: str) -> float:
        return max(5.0, min(self.timeouts[op], self.left())) if self.deadline is not None else self.timeouts[op]

    # scheduling
    def take(self, wid: int):
        with self.lock:
            while True:
                if self.left() < NO_NEW_PASS_S:
                    if any(it[2] is None for it in self.items):
                        self.deadline_hit = True
                    return None
                pending = [it for it in self.items if it[2] is None]
                if not pending:
                    if not any(it[2] == "taken" for it in self.items):
                        return None
                    self.lock.wait(1.0)  # a pass in flight may come back to the queue (its worker retired)
                    continue
                for it in pending:
                    scene, p = it[0], it[1]
                    other = self.assigned.get(scene, {}).get("B" if p == "A" else "A")
                    if other != wid:
                        it[2] = "taken"
                        self.assigned.setdefault(scene, {})[p] = wid
                        return it
                if self.live <= 1:
                    return None  # only passes excluded for this worker remain and nobody else can run them
                self.lock.wait(1.0)

    def restart(self, wid: int) -> Optional[str]:
        """A fresh worker in slot `wid`: None on success, else why.  ThorPool.restart returns the ready reply (ok
        False when the new start-up failed, the slot is then empty); a stand-in may raise instead.  Restarts run one
        at a time (as ThorPool's start-up lock does), and the time left is judged when it is this one's turn: with
        fewer than NO_NEW_PASS_S + restart_s seconds left none starts (NO_TIME), so a restart ends before the
        deadline's last NO_NEW_PASS_S seconds."""
        with self.restart_lock:
            if self.left() < NO_NEW_PASS_S + self.restart_s:
                return NO_TIME
            try:
                rr = self.pool.restart(wid)
            except Exception as e:  # noqa: BLE001
                return f"restart: {e!r}"
        if not isinstance(rr, dict) or rr.get("ok") is not True:
            return f"restart: {rr.get('error') if isinstance(rr, dict) else repr(rr)}"
        return None

    def run_pass(self, wid: int, scene: str, p: str):
        """(result or None, requeue).  requeue is True when the pass failed once and its worker could not be
        restarted for the retry: the worker is retired and another worker may run the pass (not when the restart
        was skipped or ended too close to the deadline: then deadline_hit)."""
        ids = sorted(t["task_id"] for t in self.by_scene[scene])
        if p == "B":
            ids = ids[::-1]
        tasks = {t["task_id"]: t for t in self.by_scene[scene]}
        for attempt in (1, 2):
            res, err = self._one_pass(wid, scene, p, ids, tasks)
            if res is not None:
                res["attempt"] = attempt
                return res, False
            self.errors.append({"scene": scene, "pass": p, "wid": wid, "attempt": attempt, "error": err})
            self.log(f"[reach] {scene} pass {p} on w{wid} attempt {attempt} failed: {err}")
            if err == "deadline" or self.left() < NO_NEW_PASS_S:
                self.deadline_hit = True  # no restart / retry after the deadline (a restart can take minutes)
                return None, False
            why = self.restart(wid)  # a fresh worker for the retry (and for the next pass after a second failure)
            if why is not None:
                self.errors.append({"scene": scene, "pass": p, "wid": wid, "error": why})
                self.log(f"[reach] w{wid} retired: {why}")
                self.dead.add(wid)  # this worker takes no further pass
                if attempt == 1 and (why == NO_TIME or self.left() < NO_NEW_PASS_S):
                    self.deadline_hit = True  # the retry is lost to the deadline (no other worker would start it)
                    return None, False
                return None, attempt == 1
            if attempt == 1 and self.left() < NO_NEW_PASS_S:  # the restart ran into the deadline: no retry
                self.deadline_hit = True
                return None, False
        return None, False

    def _one_pass(self, wid, scene, p, ids, tasks):
        r = self.pool.call(wid, {"op": "load", "scene": scene, "force": True}, self.bounded("load"))
        if not r.get("ok"):
            return None, f"load: {r.get('error')}"
        out = {"pass": p, "wid": wid, "load_s": r.get("load_s"), "order": list(ids), "tasks": {}, "sets": {}}
        for i in ids:
            if self.left() <= 0:
                return None, "deadline"
            r = self.pool.call(wid, {"op": "reach_task", "task": tasks[i]}, self.bounded("reach_task"))
            if not r.get("ok"):
                return None, f"reach_task {i}: {r.get('error')}"
            pos = list(r.get("positions") or [])
            if not pos:
                return None, f"reach_task {i}: no positions"
            d = digest(pos)
            if r.get("digest") not in (None, d):
                return None, f"reach_task {i}: worker digest {r.get('digest')} != {d}"
            if r.get("scene") not in (None, scene):
                return None, f"reach_task {i}: worker scene {r.get('scene')} != {scene}"
            out["tasks"][i] = {"digest": d, "n": ReachSet.from_positions(pos).n, "n_positions": len(pos),
                               "call_pose": r.get("call_pose"),
                               "call_pose_ok": _same_pose(r.get("call_pose"), tasks[i]["start"]),
                               "reach_s": r.get("reach_s"), "loaded": r.get("loaded")}
            out["sets"].setdefault(d, pos)
        return out, None

    def finish_scene(self, scene: str) -> None:
        a, b = self.results[scene]["A"], self.results[scene]["B"]
        tasks = {}
        sets: Dict[str, List[Dict]] = {}
        for i in sorted(a["tasks"]):  # every pass-A set first: stored raw positions are pass A's (frozen = A)
            ea, eb = a["tasks"][i], b["tasks"][i]
            tasks[i] = {"A": ea, "B": eb, "equal_AB": ea["digest"] == eb["digest"]}
            sets.setdefault(ea["digest"], a["sets"][ea["digest"]])
        for i in sorted(b["tasks"]):
            eb = b["tasks"][i]
            sets.setdefault(eb["digest"], b["sets"][eb["digest"]])
        doc = {"schema": SCHEMA, "scene": scene, "kind": lvp.scene_kind(scene),
               "passes": [{k: x[k] for k in ("pass", "wid", "load_s", "order", "attempt")} for x in (a, b)],
               "tasks": tasks, "sets": sets, "stable": all(v["equal_AB"] for v in tasks.values()),
               "created_utc": T.utc()}
        T.write_json(self.out / "scenes" / f"{scene}.json", doc)
        with self.lock:
            self.written.append(scene)
            k = len(self.written)
        self.log(f"[reach] {k}/{self.n_todo} {scene}: {len(tasks)} tasks, A w{a['wid']} B w{b['wid']}, "
                 f"stable {doc['stable']}, {time.monotonic() - self.t0:.0f} s")

    def _fail(self, it) -> None:
        """Under the lock: pass `it` failed; the scene cannot complete, so its other pass is not run either."""
        it[2] = "failed"
        for other in self.items:
            if other[0] == it[0] and other[2] is None:
                other[2] = "failed"

    def worker(self, wid: int) -> None:
        try:
            while True:
                it = self.take(wid)
                if it is None:
                    return
                scene, p = it[0], it[1]
                try:
                    res, requeue = self.run_pass(wid, scene, p)
                except Exception as e:  # noqa: BLE001  harness error (e.g. a reply without x / z): the pass fails
                    self.errors.append({"scene": scene, "pass": p, "wid": wid, "error": f"harness: {e!r}"})
                    self.log(f"[reach] {scene} pass {p} on w{wid}: harness error {e!r}")
                    res, requeue = None, False
                with self.lock:
                    both = False
                    if res is not None:
                        it[2] = "done"
                        self.results.setdefault(scene, {})[p] = res
                        both = len(self.results[scene]) == 2
                    elif requeue:  # back to the queue for another worker (this one is retired)
                        it[2] = None
                        self.assigned.get(scene, {}).pop(p, None)
                        self.requeued.append({"scene": scene, "pass": p, "wid": wid})
                    else:
                        self._fail(it)
                    self.lock.notify_all()
                if both:
                    try:
                        self.finish_scene(scene)
                    except Exception as e:  # noqa: BLE001  e.g. the scene file cannot be written
                        self.errors.append({"scene": scene, "wid": wid, "error": f"finish: {e!r}"})
                        self.log(f"[reach] {scene}: cannot finish the scene: {e!r}")
                        with self.lock:
                            for x in self.items:
                                if x[0] == scene:
                                    x[2] = "failed"
                if wid in self.dead:
                    return
        finally:
            with self.lock:
                for x in self.items:  # a pass this thread still holds (it ended on an unexpected error) fails, so
                    if x[2] == "taken" and self.assigned.get(x[0], {}).get(x[1]) == wid:  # take() stops waiting
                        self.errors.append({"scene": x[0], "pass": x[1], "wid": wid,
                                            "error": "harness: the worker thread ended during the pass"})
                        self._fail(x)
                self.live -= 1
                self.lock.notify_all()


def _same_pose(call_pose, start) -> Optional[bool]:
    """kin_env.pose_of(call_pose) == kin_env.pose_of(task.start); None when the worker reported no usable pose."""
    try:
        return kin_env.pose_of(call_pose) == kin_env.pose_of(start)
    except (KeyError, TypeError, ValueError):
        return None


def _worker_ids(pool) -> List[int]:
    return sorted(int(r["wid"]) for r in pool.ready())


def build_index(by_set: Mapping[str, List[Dict]], out: Path) -> Dict:
    tasks_out: Dict[str, Dict] = {}
    scene_files: Dict[str, str] = {}
    docs: Dict[str, Dict] = {}
    for name, ts in by_set.items():
        for t in ts:
            s = t["scene"]
            if s not in docs:
                raw = (out / "scenes" / f"{s}.json").read_bytes()
                scene_files[s] = T.sha256_bytes(raw)
                docs[s] = json.loads(raw)
            e = docs[s]["tasks"][t["task_id"]]
            tasks_out[t["task_id"]] = {"scene": s, "digest": e["A"]["digest"], "n": e["A"]["n"],
                                       "stable": bool(e["equal_AB"]), "call_pose_ok": e["A"].get("call_pose_ok")}
    return {"schema": INDEX_SCHEMA,
            "task_sets": {n: {"n": len(ts), "ids_sha256": T.ids_sha256(t["task_id"] for t in ts)}
                          for n, ts in by_set.items()},
            "tasks": dict(sorted(tasks_out.items())), "scene_files": dict(sorted(scene_files.items())),
            "unstable_tasks": sorted(i for i, e in tasks_out.items() if not e["stable"]),
            "call_pose_mismatch": sorted(i for i, e in tasks_out.items() if e["call_pose_ok"] is False),
            "code_sha256": T.code_sha256()}


def write_index(by_set, out: Path) -> Dict:
    """Write index.json (kept byte-for-byte when an existing one has the same content apart from created_utc /
    code_sha256, tasks.same_content) and DONE."""
    doc = build_index(by_set, out)
    p = out / "index.json"
    keep = False
    if p.is_file():
        try:
            old = T.read_json(p)
            keep = T.same_content(old, doc)
        except (OSError, ValueError):
            keep = False
    if not keep:
        T.write_json(p, dict(doc, created_utc=T.utc()))
    if not (out / "DONE").is_file():
        T.write_bytes_atomic(out / "DONE", (T.utc() + "\n").encode())
    return {"index_sha256": T.sha256_file(p), "kept_existing_index": keep}


def _plan(task_sets: Sequence[str], out: Path, root=None, tasks_dir=None):
    """(by_set, by_scene {scene: {task_id: task}}, order, todo): the scenes of the sets in dump order and those
    without a complete scene file."""
    by_set = {n: T.task_set(n, root, tasks_dir) for n in task_sets}
    by_scene: Dict[str, Dict[str, Dict]] = {}
    for ts in by_set.values():
        for t in ts:
            by_scene.setdefault(t["scene"], {})[t["task_id"]] = t
    order = _scene_order({s: list(v.values()) for s, v in by_scene.items()})
    todo = [s for s in order if not scene_complete(out / "scenes" / f"{s}.json", by_scene[s])]
    return by_set, by_scene, order, todo


def dump(task_sets: Sequence[str], out, pool, deadline: Optional[float] = None, *, root=None, tasks_dir=None,
         timeouts: Optional[Mapping[str, float]] = None, log: Callable[..., None] = print) -> Dict:
    """The dump procedure (module doc).  `pool` is a started thor_worker.ThorPool (or a stand-in with call /
    restart / ready; with nothing left to dump it is not used beyond ready(), so it may be unstarted); `deadline`
    is a time.monotonic() value (None: none).  Returns {"complete", "deadline_hit", "scenes", "skipped",
    "written", "failed", "errors", "retired", "requeued", "index"}."""
    out = Path(out)
    by_set, by_scene, order, todo = _plan(task_sets, out, root, tasks_dir)
    by_scene_l = {s: list(v.values()) for s, v in by_scene.items()}
    skipped = [s for s in order if s not in todo]
    wids = _worker_ids(pool)
    if todo and len(wids) < 2:
        raise ValueError(f"reach dump needs >= 2 workers (pass A and B on different workers), have {wids}")
    d = _Dump(by_scene_l, out, pool, deadline, dict(DEFAULT_TIMEOUTS if timeouts is None else timeouts), log)
    d.items = [[s, p, None] for s in todo for p in ("A", "B")]
    d.n_todo = len(todo)
    log(f"[reach] {len(order)} scenes ({len(skipped)} complete, {len(todo)} to dump) on workers {wids}")
    if todo:
        d.live = len(wids)
        th = [threading.Thread(target=d.worker, args=(w,), daemon=True, name=f"reach-w{w}") for w in wids]
        for x in th:
            x.start()
        for x in th:
            x.join()
    failed = sorted({it[0] for it in d.items if it[2] != "done"})
    complete = not failed and all(scene_complete(out / "scenes" / f"{s}.json", by_scene[s]) for s in order)
    res: Dict[str, Any] = {"complete": complete, "deadline_hit": d.deadline_hit, "scenes": len(order),
                           "skipped": skipped, "written": sorted(d.written), "failed": failed, "errors": d.errors,
                           "retired": sorted(d.dead), "requeued": d.requeued, "index": None}
    if complete:
        res["index"] = write_index(by_set, out)
    return res


# --- check ------------------------------------------------------------------------------------------------------------
def check(out) -> Dict:
    out = Path(out)
    res: Dict[str, Any] = {"complete": False, "why": None}
    if not (out / "DONE").is_file() or not (out / "index.json").is_file():
        res["why"] = "no index.json / DONE"
        return res
    try:
        idx = ReachIndex(out)
        for i in idx.tasks:
            idx.reach(i)
    except (OSError, ValueError, KeyError) as e:
        res["why"] = repr(e)
        return res
    res.update(complete=True, index_sha256=idx.sha256, tasks=len(idx.tasks),
               stable=sum(1 for e in idx.tasks.values() if e["stable"]),
               unstable=len(idx.doc.get("unstable_tasks", [])), scenes=len(idx.doc["scene_files"]),
               call_pose_mismatch=len(idx.doc.get("call_pose_mismatch", [])))
    return res


# --- CLI --------------------------------------------------------------------------------------------------------------
def _on_signal(signum, frame):  # noqa: ARG001
    raise Terminated(signum)


def cmd_dump(a) -> int:
    from vpd.tvr import thor_worker  # tvreval's pool (TVR_PY workers)

    t0 = time.monotonic()
    deadline = None if a.deadline_s is None else t0 + float(a.deadline_s)
    E = a.envs if a.envs is not None else pilot_cfg.get("render.replay_envs")
    L = a.lp if a.lp is not None else pilot_cfg.get("render.lp_threads")
    P = a.profile if a.profile is not None else pilot_cfg.get("render.lvp_profile")
    if E is None or L is None or P is None:
        print("[reach] refusing: --envs / --lp / --profile not given and pilot_cfg render.* still None")
        return 1
    if int(E) * int(L) > MAX_EL:
        print(f"[reach] refusing: E x L = {int(E) * int(L)} > {MAX_EL}")
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw: Dict[str, Any] = {"out_dir": Path(a.pool_dir) if a.pool_dir else out.parent / (out.name + "_pool"),
                          "houses_dir": Path(a.houses), "repo": Path(a.repo)}
    if a.fake_thor:
        kw.update(py=sys.executable, extra_pythonpath=(str(a.fake_thor),))
    timeouts = dict(getattr(thor_worker, "TIMEOUTS", DEFAULT_TIMEOUTS))
    pool = thor_worker.ThorPool(int(E), int(L), str(P), **kw)
    old = {s: signal.signal(s, _on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    fast = False
    try:
        try:
            sets = T.parse_sets(a.task_sets)
            if _plan(sets, out, a.tvr_out, a.tasks_dir)[3]:
                pool.start()
            else:  # a rerun after a complete dump: no THOR worker start (that could only fail the step)
                print("[reach] every scene file is complete: no THOR worker started")
            res = dump(sets, out, pool, deadline, root=a.tvr_out, tasks_dir=a.tasks_dir, timeouts=timeouts)
            fast = res["deadline_hit"]
        except Terminated:
            print("[reach] terminated: fast shutdown")
            pool.close(fast=True)
            return 143
        except Exception as e:  # noqa: BLE001
            print(f"[reach] error: {e!r}")
            pool.close(fast=True)
            return 1
        pool.close(fast=fast)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    T.write_json(out / "dump_last.json", dict(res, utc=T.utc(), envs=int(E), lp=int(L), profile=str(P),
                                              wall_s=round(time.monotonic() - t0, 3)))
    print(f"[reach] complete={res['complete']} written={len(res['written'])} skipped={len(res['skipped'])} "
          f"failed={res['failed']} deadline_hit={res['deadline_hit']} retired={res['retired']}")
    if res["complete"]:
        return 0
    return 4 if res["deadline_hit"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.tvr.reach", description="TVR frozen per-task reach sets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("houses", help="extract the ProcTHOR house lines of the task sets")
    h.add_argument("--task-sets", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--src", default=str(lvp.PROCTHOR_TRAIN))
    d = sub.add_parser("dump", help="two-pass reach dump on a THOR worker pool")
    d.add_argument("--task-sets", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--envs", type=int, default=None)
    d.add_argument("--lp", type=int, default=None)
    d.add_argument("--profile", default=None)
    d.add_argument("--houses", required=True)
    d.add_argument("--deadline-s", type=float, default=None)
    d.add_argument("--repo", default=str(lvp.TVR_REPO))
    d.add_argument("--pool-dir", default=None,
                   help="worker homes / logs (default <OUT>_pool next to OUT, e.g. reach/v1_pool)")
    d.add_argument("--fake-thor", default=None, help="tests: a tvr_fakes root put first on the workers' PYTHONPATH")
    c = sub.add_parser("check", help="exit 0 iff the dump is complete")
    c.add_argument("--out", required=True)
    for p in (h, d):
        p.add_argument("--tasks-dir", default=None)
        p.add_argument("--tvr-out", default=None, help="root holding split/dev200/selection.json (default OUT/tvr)")
    a = ap.parse_args(argv)
    if a.cmd == "houses":
        try:
            doc = houses(T.parse_sets(a.task_sets), a.out, src=a.src, root=a.tvr_out, tasks_dir=a.tasks_dir)
        except (OSError, ValueError, KeyError) as e:
            print(f"[reach] error: {e!r}")
            return 1
        print(f"[reach] houses: {len(doc['scenes'])} ProcTHOR scenes -> {a.out}")
        return 0
    if a.cmd == "dump":
        return cmd_dump(a)
    r = check(a.out)
    print(json.dumps(r))
    return 0 if r["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
