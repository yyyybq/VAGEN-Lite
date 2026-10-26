"""Per-task distance tables: the remaining minimum number of actions from every state near a task's start to success
(round 13, the user's 12th supplement of 2026-10-26, sections 2.2 and 3).

A table holds every state reachable from the task's start within ``depth`` primitive actions under the runtime rule
(``difficulty.py``'s action replica and collision check; states merged at the contract resolution, 1 cm / 1 deg, so
the env can look its pose up), every legal transition between them, and by a breadth-first search backwards from the
successful states (the env's success rule, ``release.env_success_oracle``) each state's remaining minimum number of
actions inside that graph.  The start's distance is the row's ``difficulty.d_star`` (checked when built).

Judging success is the cost (about 2.6 ms a state); ``may_succeed`` skips the states where no task object can be in
view, which no family's success allows (checked against judging every state when the table is built with
``check_prefilter``).

The "optimal" progress of the env (``progress_version: optimal``, off by default) is
``phi = 1 - d / (d_star + slack)`` clipped to [0, 1], 1 on success, 0 for a pose missing from the table.

    python -m data_gen.interiorgs_v10.distance_table build --rows ROWS --gs-root DIR --out DIR [--depth 8]
        [--depth-offset 3] [--shard i/n] [--workers 12] [--reserve-train] [--task-ids FILE] [--limit N]
        [--check-prefilter N]

On the cluster a shard runs as a one-card sco job (examples/train/active_spatial/sco_interiorgs_v10_train.sh,
mode distance-tables) for its 14 vCPUs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import ACTIONS
from .difficulty import IS_TRANSLATION, FastGeometry, _split, apply_action_batch, contract_keys, to_c2w

DEFAULT_DEPTH = 8
DEFAULT_SLACK = 4
VERSION = "interiorgs_v10_distance_table_v1"


def may_succeed(row: Mapping[str, Any], c2w: np.ndarray, margin_deg: float = 5.0) -> np.ndarray:
    """A necessary condition for success in every family, per pose: some task object's bounding sphere meets the
    camera's viewing cone (the image corner's half-angle from the row's intrinsics, plus ``margin_deg``)."""
    K = np.asarray(row["init_camera"]["intrinsics"], float)
    half = math.degrees(math.atan(math.hypot(K[0, 2] / K[0, 0], K[1, 2] / K[1, 1])))
    c2w = np.asarray(c2w, float).reshape(-1, 4, 4)
    pos, fwd = c2w[:, :3, 3], c2w[:, :3, 2]
    fwd = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    ok = np.zeros(len(c2w), dtype=bool)
    for obj in (row.get("target_object") or {}).get("objects") or []:
        lo, hi = np.asarray(obj["bbox_min"], float), np.asarray(obj["bbox_max"], float)
        center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)
        v = center[None, :] - pos
        dist = np.linalg.norm(v, axis=1)
        cosang = np.einsum("ij,ij->i", v, fwd) / np.maximum(dist, 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        spread = np.degrees(np.arcsin(np.clip(radius / np.maximum(dist, 1e-9), 0.0, 1.0)))
        ok |= (dist <= radius) | (ang <= half + spread + margin_deg)
    return ok


def reachable_graph(geom: FastGeometry, init_c2w: Any, depth: int,
                    max_states: int = 2_000_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(keys, P, R, depth_of, edges): the states within ``depth`` actions of ``init_c2w`` (contract keys, the first
    pose reaching each key kept), and every legal transition between two of them as (from, to) rows."""
    P0, R0 = _split(init_c2w)
    keys: List[int] = [int(contract_keys(P0, R0)[0])]
    index = {keys[0]: 0}
    Ps, Rs, depth_of = [P0[0]], [R0[0]], [0]
    src: List[int] = []
    dst: List[int] = []

    def expand(frontier: List[int], d: Optional[int]) -> List[int]:
        fP = np.stack([Ps[i] for i in frontier])
        fR = np.stack([Rs[i] for i in frontier])
        new: List[int] = []
        for ai, a in enumerate(ACTIONS):
            p, r = apply_action_batch(fP, fR, a)
            r = np.broadcast_to(r, (len(frontier), 3, 3))
            legal = np.asarray(geom.transitions_legal(fP, p, r), dtype=bool) if IS_TRANSLATION[ai] else \
                np.ones(len(frontier), dtype=bool)
            ks = contract_keys(p, r)
            for j in np.flatnonzero(legal):
                k = int(ks[j])
                t = index.get(k)
                if t is None:
                    if d is None:  # the last level: transitions back into the graph only
                        continue
                    t = len(keys)
                    index[k] = t
                    keys.append(k)
                    Ps.append(p[j])
                    Rs.append(np.array(r[j]))
                    depth_of.append(d)
                    new.append(t)
                if t != frontier[j]:
                    src.append(frontier[j])
                    dst.append(t)
        return new

    frontier = [0]
    for d in range(1, depth + 1):
        if not frontier or len(keys) > max_states:
            break
        frontier = expand(frontier, d)
    if frontier:
        expand(frontier, None)
    return (np.asarray(keys, dtype=np.int64), np.stack(Ps), np.stack(Rs), np.asarray(depth_of, dtype=np.int16),
            np.stack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)], axis=1))


def backward_distances(n: int, edges: np.ndarray, success: np.ndarray) -> np.ndarray:
    """Each state's minimum number of transitions to a successful state (-1: none in the graph)."""
    order = np.argsort(edges[:, 1], kind="stable")
    by_dst, srcs = edges[order, 1], edges[order, 0]
    starts = np.searchsorted(by_dst, np.arange(n + 1))
    dist = np.full(n, -1, dtype=np.int16)
    q = deque(np.flatnonzero(success).tolist())
    dist[list(q)] = 0
    while q:
        t = q.popleft()
        for s in srcs[starts[t]:starts[t + 1]]:
            if dist[s] < 0:
                dist[s] = dist[t] + 1
                q.append(int(s))
    return dist


def build(scene: Any, row: Mapping[str, Any], depth: int = DEFAULT_DEPTH, *, oracle: Optional[Callable] = None,
          prefilter: bool = True, check_prefilter: bool = False, max_states: int = 2_000_000) -> Dict[str, Any]:
    """The distance table of one task: {"keys" (sorted int64), "dist" (int16, -1 none), "d_init", "states", ...}."""
    t0 = time.perf_counter()
    if oracle is None:
        from .release import env_success_oracle

        oracle = env_success_oracle(row, scene)
    geom = FastGeometry.for_scene(scene)
    init = np.asarray(row["init_camera"]["extrinsics"], float)
    keys, P, R, depth_of, edges = reachable_graph(geom, init, depth, max_states)
    t1 = time.perf_counter()
    c2w = to_c2w(P, R)
    passed = may_succeed(row, c2w) if prefilter else np.ones(len(c2w), dtype=bool)
    success = np.zeros(len(c2w), dtype=bool)
    idx = np.flatnonzero(passed)
    success[idx] = np.fromiter((bool(oracle(c2w[i])) for i in idx), dtype=bool, count=len(idx))
    missed = None
    if check_prefilter and prefilter:
        rest = np.flatnonzero(~passed)
        missed = int(sum(bool(oracle(c2w[i])) for i in rest))
    t2 = time.perf_counter()
    dist = backward_distances(len(keys), edges, success)
    order = np.argsort(keys, kind="stable")
    d_star = (row.get("difficulty") or {}).get("d_star")
    return {"version": VERSION, "task_id": row.get("task_id"), "depth": int(depth), "depth_reached": int(depth_of.max()),
            "keys": keys[order], "dist": dist[order], "depth_of": depth_of[order], "d_init": int(dist[0]),
            "d_star": None if d_star is None else int(d_star),
            "agree": None if d_star is None else int(dist[0]) == int(d_star),
            "states": int(len(keys)), "edges": int(len(edges)), "success_states": int(success.sum()),
            "prefilter_pass": int(passed.sum()), "prefilter_missed_success": missed,
            "unreached_share": round(float((dist < 0).mean()), 4),
            "seconds": {"graph": round(t1 - t0, 2), "success": round(t2 - t1, 2),
                        "total": round(time.perf_counter() - t0, 2)}}


def save(path: Path, table: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in table.items() if k not in ("keys", "dist", "depth_of")}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, keys=table["keys"], dist=table["dist"], depth_of=table["depth_of"],
                        meta=np.array(json.dumps(meta)))
    tmp.replace(path)


def load(path: Path) -> Dict[str, Any]:
    with np.load(path) as z:
        out = json.loads(str(z["meta"]))
        out.update(keys=z["keys"], dist=z["dist"], depth_of=z["depth_of"])
    return out


def table_path(root: Path, task_id: str) -> Path:
    return Path(root) / f"{task_id}.npz"


def lookup(table: Mapping[str, Any], c2w: Any) -> Optional[int]:
    """The remaining minimum number of actions of a pose, or None (not in the table, or no success reachable)."""
    P, R = _split(c2w)
    k = int(contract_keys(P, R)[0])
    keys = table["keys"]
    i = int(np.searchsorted(keys, k))
    if i >= len(keys) or int(keys[i]) != k:
        return None
    d = int(table["dist"][i])
    return d if d >= 0 else None


def phi(distance: Optional[int], d_star: int, slack: int = DEFAULT_SLACK, success: bool = False) -> float:
    """The optimal progress: 1 on success, 1 - d / (d_star + slack) clipped to [0, 1], 0 for an unknown pose."""
    if success:
        return 1.0
    if distance is None:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - distance / float(max(1, int(d_star) + int(slack))))))


class TableCache:
    """The env's per-task table, loaded once per task from ``root`` (``<task_id>.npz``)."""

    def __init__(self, root: Any) -> None:
        self.root = Path(root)
        self._task: Optional[str] = None
        self._table: Optional[Dict[str, Any]] = None

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        if task_id != self._task:
            path = table_path(self.root, task_id)
            self._table = load(path) if path.is_file() else None
            self._task = task_id
        return self._table

    def progress(self, item: Mapping[str, Any], c2w: Any, success: bool, slack: int = DEFAULT_SLACK) -> Dict[str, Any]:
        table = self.get(str(item.get("task_id")))
        d_star = int((item.get("difficulty") or {}).get("d_star") or 0)
        dist = lookup(table, c2w) if table is not None else None
        return {"optimal_distance": dist, "optimal_table": table is not None,
                "optimal_progress": phi(dist, d_star, slack, success)}


def depth_for(row: Mapping[str, Any], depth: int, offset: Optional[int]) -> int:
    """The table depth of a task: ``depth``, or with ``offset`` min(depth, d_star + offset)."""
    if offset is None:
        return int(depth)
    return int(min(depth, int((row.get("difficulty") or {}).get("d_star") or 0) + int(offset)))


def in_shard(task_id: str, shard: Optional[str]) -> bool:
    """``shard`` "i/n": the tasks whose id hashes to i modulo n (stable across runs and machines)."""
    if not shard:
        return True
    import hashlib

    i, n = (int(x) for x in shard.split("/"))
    return int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % n == i


def _work(args: Tuple[List[Dict[str, Any]], str, str, int, int, Optional[int]]) -> List[Dict[str, Any]]:
    rows, gs_root, out, depth, check, offset = args
    import contextlib
    import io

    from .scene import load_scene

    scenes: Dict[str, Any] = {}
    res = []
    for i, row in enumerate(rows):
        path = table_path(Path(out), row["task_id"])
        if path.is_file():
            res.append({"task_id": row["task_id"], "skipped": "exists"})
            continue
        try:
            sid = str(row["scene_id"])
            if sid not in scenes:
                with contextlib.redirect_stdout(io.StringIO()):
                    scenes = {sid: load_scene(Path(gs_root) / sid, sid, wall_model="thick")}
            table = build(scenes[sid], row, depth_for(row, depth, offset), check_prefilter=i < check)
            save(path, table)
            res.append({k: v for k, v in table.items() if k not in ("keys", "dist", "depth_of")})
        except Exception as exc:  # one task failing does not stop the rest
            res.append({"task_id": row.get("task_id"), "error": f"{type(exc).__name__}: {exc}"})
    return res


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the tables of a rows file")
    b.add_argument("--rows", required=True)
    b.add_argument("--gs-root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="the table depth (with --depth-offset: its cap)")
    b.add_argument("--depth-offset", type=int, default=None,
                   help="depth min(--depth, d_star + offset) per task (states multiply by about 3.3 a level)")
    b.add_argument("--shard", default=None, help="i/n: this process builds the tasks hashing to i modulo n")
    b.add_argument("--workers", type=int, default=12)
    b.add_argument("--reserve-train", action="store_true", help="only the reserve training rows (RL's tasks)")
    b.add_argument("--task-ids", default=None, help="a file of task ids (one per line) to build")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--check-prefilter", type=int, default=0,
                   help="judge every state of the first N tasks of each scene batch too (prefilter check)")
    args = ap.parse_args(argv)
    from multiprocessing import Pool

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    if args.reserve_train:
        rows = [r for r in rows if r.get("reserve") and r.get("split") == "train"]
    if args.task_ids:
        want = {l.strip() for l in Path(args.task_ids).read_text().splitlines() if l.strip()}
        rows = [r for r in rows if r["task_id"] in want]
    rows = [r for r in rows if in_shard(r["task_id"], args.shard)]
    rows.sort(key=lambda r: (str(r["scene_id"]), r["task_id"]))
    if args.limit:
        rows = rows[:args.limit]
    batches: List[List[Dict[str, Any]]] = []
    for r in rows:  # one scene per batch, so a worker loads each scene once
        if not batches or batches[-1][0]["scene_id"] != r["scene_id"] or len(batches[-1]) >= 8:
            batches.append([])
        batches[-1].append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done = 0
    log_name = f"summary_{args.shard.replace('/', 'of')}.jsonl" if args.shard else "summary.jsonl"
    with Pool(args.workers) as pool, open(out / log_name, "a") as log:
        for part in pool.imap_unordered(_work, [(bt, args.gs_root, args.out, args.depth, args.check_prefilter,
                                                 args.depth_offset) for bt in batches]):
            for rec in part:
                log.write(json.dumps(rec) + "\n")
                done += 1
            log.flush()
    print(json.dumps({"tasks": len(rows), "written": done, "seconds": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main()
