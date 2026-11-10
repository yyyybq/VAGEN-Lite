"""Exact pose-graph BFS on the frozen per-task reachable set (CONTRACT_wave2 4.7 opt, 5.1.4, 6.7; G10(j)).

opt(task) = the minimum number of non-Stop actions that take the agent from task.start to task.target in the
kin_env transition graph of R(task) (directed; failed transitions are self-loops and are ignored); None when the
target pose is unreachable (the task stays in every denominator, O12).  States are (x, z, rot, hor) with x, z in
integer hundredths, so the 8 transitions are exact integer arithmetic; a test checks them against kin_env.step.

CLI (PY_Q3, 1 core):
  python -B -m vpd.tvr.bfs table --task-sets eval500,dev200,sft1600 --reach OUT/tvr/reach/v1 --out OUT/tvr/bfs
  -> <set>_opt.json {"schema": "vpd.tvr.bfs/1", "reach_index_sha256", "task_set", "n", "tasks": {task_id:
     {"feasible", "opt", "start_in_reach", "target_in_reach"}}, "n_infeasible", "code_sha256"} (no timestamp;
     an existing file with the same content is kept byte-for-byte)
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from vpd.tvr import kin_env, tasks as T
from vpd.tvr.kin_env import Pose, ReachSet

SCHEMA = "vpd.tvr.bfs/1"
_DELTAS = {r: (dx * 25, dz * 25) for r, (dx, dz) in kin_env._MOVE_DELTAS.items()}  # hundredths of a metre


def _h(v: float) -> int:
    return int(round(v * 100))


def _neighbours(s, cells):
    """The successor states of s = (ix, iz, rot, hor) under the 8 non-Stop actions (successful ones only)."""
    ix, iz, rot, hor = s
    d = _DELTAS.get(rot)
    if d is not None:
        dx, dz = d
        for mx, mz in ((dx, dz), (-dx, -dz), (-dz, dx), (dz, -dx)):  # Ahead, Back, Left, Right
            c = (ix + mx, iz + mz)
            if c in cells:
                yield (c[0], c[1], rot, hor)
    yield (ix, iz, (rot + 45) % 360, hor)
    yield (ix, iz, (rot - 45) % 360, hor)
    if hor > kin_env.HOR_MIN:
        yield (ix, iz, rot, hor - 30)
    if hor < kin_env.HOR_MAX:
        yield (ix, iz, rot, hor + 30)


def _state(p: Pose):
    return (_h(p.x), _h(p.z), p.rot, p.hor)


def bfs_opt(start: Pose, target: Pose, reach: ReachSet, max_nodes: int = 2_000_000) -> Optional[int]:
    """Minimum number of non-Stop actions from start to the exact target pose on `reach`; None if unreachable.
    RuntimeError if more than max_nodes states are visited."""
    start, target = kin_env.pose_of(start), kin_env.pose_of(target)
    if kin_env.is_exact(start, target):
        return 0
    cells = {(_h(x), _h(z)) for (x, z) in reach.cells}
    s0, goal = _state(start), _state(target)
    if (goal[0], goal[1]) not in cells and (goal[0], goal[1]) != (s0[0], s0[1]):
        return None  # a move can only enter a cell of the set
    seen = {s0}
    frontier = deque([(s0, 0)])
    while frontier:
        s, d = frontier.popleft()
        for t in _neighbours(s, cells):
            if t in seen:
                continue
            if t == goal:
                return d + 1
            seen.add(t)
            if len(seen) > max_nodes:
                raise RuntimeError(f"bfs: more than {max_nodes} states")
            frontier.append((t, d + 1))
    return None


def opt_table(tasks: Iterable[Mapping], reach_index) -> Dict[str, Dict]:
    """{task_id: {"feasible", "opt", "start_in_reach", "target_in_reach"}} with R(task) = reach_index.reach(id)."""
    out: Dict[str, Dict] = {}
    for t in tasks:
        r = reach_index.reach(t["task_id"])
        s, g = kin_env.pose_of(t["start"]), kin_env.pose_of(t["target"])
        opt = bfs_opt(s, g, r)
        out[t["task_id"]] = {"feasible": opt is not None, "opt": opt, "start_in_reach": (s.x, s.z) in r,
                             "target_in_reach": (g.x, g.z) in r}
    return out


def table_doc(name: str, tasks, reach_index) -> Dict:
    tab = opt_table(tasks, reach_index)
    return {"schema": SCHEMA, "task_set": name, "n": len(tab), "reach_index_sha256": reach_index.sha256,
            "tasks": tab, "n_infeasible": sum(1 for v in tab.values() if not v["feasible"]),
            "code_sha256": T.code_sha256()}


def cmd_table(a) -> int:
    from vpd.tvr import reach as R

    idx = R.ReachIndex(a.reach)
    out = Path(a.out)
    for name in T.parse_sets(a.task_sets):
        doc = table_doc(name, T.task_set(name, a.tvr_out, a.tasks_dir), idx)
        p = out / f"{name}_opt.json"
        sha, kept = T.write_json_stable(p, doc)
        print(f"[bfs] {p}: {doc['n']} tasks, {doc['n_infeasible']} infeasible, sha256 {sha}"
              f"{' (kept)' if kept else ''}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.tvr.bfs", description="TVR BFS optimum tables")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="write <set>_opt.json for each task set")
    t.add_argument("--task-sets", required=True)
    t.add_argument("--reach", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--tasks-dir", default=None)
    t.add_argument("--tvr-out", default=None, help="root holding split/dev200/selection.json (default OUT/tvr)")
    a = ap.parse_args(argv)
    try:
        return cmd_table(a)
    except (OSError, ValueError, KeyError) as e:
        print(f"[bfs] error: {e!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
