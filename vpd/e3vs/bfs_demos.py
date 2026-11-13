"""Shortest action paths on the E3VS action lattice: train demo pool, val/test optimal lengths (SPL), capped list.

Search (ported from the survey prototype plan/login_tmp/vpd_e3vs/scripts/bfs.py, 2026-11-08; that run gave
val 218/231 reached, 13 capped, median 3; test 355/377, 22 capped; train 1,344/1,406, 62 capped):
  - state = (x, y, z, yaw index, pitch index) with yaw = yaw0 + 30*i and pitch = pitch0 + 30*j, where yaw0 and
    pitch0 are the start pose's view direction (every E3VS pose has zero roll, so turn = yaw and look = pitch
    exactly; vpd/tests/test_e3vs_geometry.py checks the lattice against geometry.apply_action);
  - state key = position rounded to 1 cm, yaw index mod 12, pitch index; looks that would reach |pitch| >= 90
    (within 1e-6 deg, PITCH_EPS_DEG) are not expanded;
  - breadth first, actions expanded in the fixed order vpd.e3vs.SEARCH_ORDER; the goal test runs when a state
    is first generated, so the first success found is the nearest answerable viewpoint in steps and ties break
    by that order; "stop" is appended to every path;
  - depth cap 16; state cap 1.5M, checked after each completed depth (as the prototype: the state count can
    overshoot the cap by the last layer);
  - success = geometry.success (0.25 m AND <= 20 deg to any answerable viewpoint).

Collision is pluggable: ``--collides MODULE:FUNC`` names a factory ``FUNC(scene_id) -> collides(p0, p1)``
(the pilot: vpd.e3vs.mesh_collision:load_collider, the official mesh model; the occupancy grid of WP-E3VS-B
until 2026-11-12).  A translation whose segment collides is not an edge (the official
restrict_collision rule skips it, so it never helps a shortest path); rotations never collide.  The default is
no collision; that run must reproduce the survey numbers above.

Every found path is replayed with the official-semantics geometry.rollout from the start c2w (with the same
collides); the replayed final pose must be a success and no replayed move may collide.

Outputs (``--out``, default OUT/e3vs/bfs/<tag>/<split>/):
  records.jsonl   one record per start: episode fields, status (reached | capped | unreachable |
                  start_success), opt_len, actions (incl. "stop"), poses (start + after each action),
                  goal_viewpoint, n_states, extra flag;
  demo_pool.jsonl train only: the reached records (BFS-capped / unreachable episodes are excluded, PREREG);
  opt_len.json    {episode_index: optimal length or null} for the real starts (SPL denominators);
  capped.json     the real starts that were not reached;
  summary.json    counts, quantiles, timings, inputs (md5s, commit, code sha256).

``--extra`` also searches from each episode's other unanswerable viewpoints (records tagged extra=True), the
start pool for tiers above about 900 demos.  CPU: pure Python, one thread per process; ``--procs`` is an
explicit process count (default 1); each pool worker is pinned to one CPU (round robin over the affinity), and
a --procs 1 CLI run with --collides to one CPU, because the mesh collider's Embree / TBB sizes its thread pool
from the affinity.  With --procs > 1 a factory's ``prefork(scene_ids)`` (the mesh factory's hashes the meshes)
runs in the parent before the fork.  The full train split (about 18 min at 1 core) belongs in the e3vs-prep
sco job, not on the login node.

Usage: python -m vpd.e3vs.bfs_demos --split val [--limit N] [--procs 1] [--collides MOD:FUNC --tag T]
       [--extra] [--out DIR]
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import (LOGIN_TMP_REPO, SEARCH_ORDER, STEP_M, SUCCESS_ANG_DEG, SUCCESS_POS_M, check_repo, code_meta,
                      load_episodes, repo_dir)
from vpd.e3vs import geometry as G

MAX_DEPTH = 16
STATE_CAP = 1_500_000
N_TRANSLATIONS = 6  # SEARCH_ORDER[:6] are the move_* actions
assert all(a.startswith("move_") for a in SEARCH_ORDER[:N_TRANSLATIONS])
assert not any(a.startswith("move_") for a in SEARCH_ORDER[N_TRANSLATIONS:])

_OFF = 1 << 20  # cm offset for key packing (+-10 km)
# |pitch| must stay below 90 deg.  The +30 deg template starts are stored as 29.99999999999998 deg, so the survey
# prototype (strict ">= 90") let them look up twice to a vertical view; there the official planar_forward is
# zero and move_left/right are no-ops, which the yaw-based lattice cannot represent.  The tolerance closes that.
PITCH_EPS_DEG = 1e-6
_COS_TOL = math.cos(math.radians(SUCCESS_ANG_DEG))
_POS_TOL2 = SUCCESS_POS_M * SUCCESS_POS_M


# ------------------------------------------------------------------------------------------------------------
# lattice
# ------------------------------------------------------------------------------------------------------------
def _fwd(yaw: float, pit: float) -> tuple[float, float, float]:
    cp = math.cos(pit)
    return (cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pit))


class Lattice:
    """The 30 deg / 0.25 m action lattice around one start pose."""

    def __init__(self, start_c2w):
        c = np.asarray(start_c2w, dtype=np.float64)
        if G.roll_of(c) > 1e-6:
            raise ValueError(f"start pose has roll ({G.roll_of(c):.2e}); the yaw/pitch lattice needs roll 0")
        self.y0, self.p0 = G.yaw_pitch(c)
        self.p0_deg = math.degrees(self.p0)
        self.start = (float(c[0, 3]), float(c[1, 3]), float(c[2, 3]), 0, 0)
        if max(abs(v) for v in self.start[:3]) * 100 >= _OFF / 2:
            raise ValueError("start position out of the key-packing range")

    def yaw(self, st) -> float:
        return self.y0 + math.radians(30 * st[3])

    def pitch(self, st) -> float:
        return self.p0 + math.radians(30 * st[4])

    def forward(self, st) -> tuple[float, float, float]:
        return _fwd(self.yaw(st), self.pitch(st))

    def step(self, st, a: str):
        """Next lattice state, or None for a look that would reach |pitch| >= 90 deg."""
        x, y, z, yi, pj = st
        if a == "move_forward" or a == "move_backward":
            f = _fwd(self.y0 + math.radians(30 * yi), self.p0 + math.radians(30 * pj))
            s = STEP_M if a == "move_forward" else -STEP_M
            return (x + s * f[0], y + s * f[1], z + s * f[2], yi, pj)
        if a == "move_left" or a == "move_right":
            yaw = self.y0 + math.radians(30 * yi)
            r = (math.sin(yaw), -math.cos(yaw))
            s = STEP_M if a == "move_right" else -STEP_M
            return (x + s * r[0], y + s * r[1], z, yi, pj)
        if a == "move_up":
            return (x, y, z + STEP_M, yi, pj)
        if a == "move_down":
            return (x, y, z - STEP_M, yi, pj)
        if a == "turn_left":
            return (x, y, z, yi + 1, pj)
        if a == "turn_right":
            return (x, y, z, yi - 1, pj)
        if a == "look_up":
            return None if self.p0_deg + 30 * (pj + 1) >= 90 - PITCH_EPS_DEG else (x, y, z, yi, pj + 1)
        if a == "look_down":
            return None if self.p0_deg + 30 * (pj - 1) <= -90 + PITCH_EPS_DEG else (x, y, z, yi, pj - 1)
        raise ValueError(f"not a lattice action: {a!r}")


def state_key(st) -> int:
    """Injective int packing of (round(x cm), round(y cm), round(z cm), yaw index mod 12, pitch index)."""
    rx, ry, rz = round(st[0] * 100) + _OFF, round(st[1] * 100) + _OFF, round(st[2] * 100) + _OFF
    return ((((rx << 21) | ry) << 21 | rz) * 12 + st[3] % 12) * 16 + (st[4] + 8)


def goal_table(goals) -> list[tuple]:
    """[(vid, gx, gy, gz, fx, fy, fz)] with unit view directions."""
    out = []
    for vid, g in goals.items():
        g = np.asarray(g, dtype=np.float64)
        f = g[:3, 2] / np.linalg.norm(g[:3, 2])
        out.append((int(vid), float(g[0, 3]), float(g[1, 3]), float(g[2, 3]), float(f[0]), float(f[1]), float(f[2])))
    return out


def _near(lat: Lattice, st, gt) -> bool:
    x, y, z = st[0], st[1], st[2]
    f = lat.forward(st)
    for _, gx, gy, gz, fx, fy, fz in gt:
        if ((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2 <= _POS_TOL2
                and f[0] * fx + f[1] * fy + f[2] * fz >= _COS_TOL):
            return True
    return False


def bfs(start_c2w, goals, collides=None, max_depth: int = MAX_DEPTH, cap: int = STATE_CAP) -> dict:
    """Breadth-first search from start_c2w to any goal (dict vid -> c2w).

    Returns {status, opt_len, actions (without stop) or None, n_states, end_state}.
    """
    lat = Lattice(start_c2w)
    gt = goal_table(goals)
    s0 = lat.start
    if _near(lat, s0, gt):
        return {"status": "start_success", "opt_len": 0, "actions": [], "n_states": 1, "end_state": s0}
    k0 = state_key(s0)
    parent = {k0: -1}  # key -> parent_key * 16 + action index; start -> -1
    front = [(s0, k0)]
    order = SEARCH_ORDER
    for d in range(1, max_depth + 1):
        nf = []
        for s, ks in front:
            for ai in range(len(order)):
                t = lat.step(s, order[ai])
                if t is None:
                    continue
                k = state_key(t)
                if k in parent:
                    continue
                if collides is not None and ai < N_TRANSLATIONS and collides(np.array(s[:3]), np.array(t[:3])):
                    continue  # not an edge; t may still be reached another way
                parent[k] = ks * 16 + ai
                if _near(lat, t, gt):
                    acts = []
                    kk = k
                    while parent[kk] != -1:
                        v = parent[kk]
                        acts.append(order[v & 15])
                        kk = v >> 4
                    acts.reverse()
                    return {"status": "reached", "opt_len": d, "actions": acts, "n_states": len(parent),
                            "end_state": t}
                nf.append((t, k))
        front = nf
        if len(parent) > cap:
            return {"status": "capped", "opt_len": None, "actions": None, "n_states": len(parent), "end_state": None}
    return {"status": "unreachable", "opt_len": None, "actions": None, "n_states": len(parent), "end_state": None}


# ------------------------------------------------------------------------------------------------------------
# episodes
# ------------------------------------------------------------------------------------------------------------
def load_collider_factory(spec: str | None):
    """'module:func' -> func (a factory scene_id -> collides(p0, p1)); None -> None."""
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    if not mod or not fn:
        raise ValueError(f"--collides wants MODULE:FUNC, got {spec!r}")
    return getattr(importlib.import_module(mod), fn)


def solve_start(ep: dict, start_vid: int, start_c2w, collides, extra: bool, max_depth: int, cap: int) -> dict:
    """BFS + replay for one start; returns a JSON-ready record."""
    r = bfs(start_c2w, ep["goals"], collides=collides, max_depth=max_depth, cap=cap)
    rec = {
        "split": ep["split"], "episode_index": ep["episode_index"], "scene_id": ep["scene_id"],
        "question_type": ep["question_type"], "question": ep["question"], "instance_id": ep["instance_id"],
        "start_viewpoint": int(start_vid), "extra": bool(extra), "status": r["status"], "opt_len": r["opt_len"],
        "n_states": r["n_states"], "actions": None, "poses": None, "goal_viewpoint": None, "goal_viewpoints": None,
        "replay_dev": None,
    }
    if r["status"] not in ("reached", "start_success"):
        return rec
    acts = list(r["actions"]) + ["stop"]
    poses, hits = G.rollout(start_c2w, acts, collides=collides)
    vids = sorted(ep["goals"])
    mask = G.match_goals(poses[-1], [ep["goals"][v] for v in vids])
    if not mask.any():
        raise AssertionError(f"episode {ep['episode_index']} start {start_vid}: replayed path ends off-goal")
    if any(hits):
        raise AssertionError(f"episode {ep['episode_index']} start {start_vid}: replayed path collides")
    lat = Lattice(start_c2w)
    st = lat.start
    dev = 0.0
    for a, p in zip(acts[:-1], poses[1:-1]):
        st = lat.step(st, a)
        dev = max(dev, float(np.abs(np.array(st[:3]) - p[:3, 3]).max()),
                  float(np.abs(np.array(lat.forward(st)) - G.view_forward(p)).max()))
    matched = [vids[i] for i in np.flatnonzero(mask)]
    dist = {v: float(np.linalg.norm(ep["goals"][v][:3, 3] - poses[-1][:3, 3])) for v in matched}
    rec.update(actions=acts, poses=[p.tolist() for p in poses], goal_viewpoints=matched,
               goal_viewpoint=min(matched, key=lambda v: (dist[v], v)), replay_dev=dev)
    return rec


def episode_starts(ep: dict, extra: bool):
    yield ep["start_viewpoint"], ep["start_c2w"], False
    if extra:
        for vid in sorted(ep["unans"]):
            if vid != ep["start_viewpoint"]:
                yield vid, ep["unans"][vid], True


# worker state (fork: inherited; set by _init)
_W: dict = {}


def _pin_worker() -> int | None:
    """In a pool worker: pin it to one CPU (round robin over the inherited affinity by worker number) before any
    collider is built: Embree / TBB (the mesh collider) sizes its thread pool from the affinity."""
    import multiprocessing as mp

    ident = getattr(mp.current_process(), "_identity", ())
    if not ident or not hasattr(os, "sched_setaffinity"):
        return None  # the parent (--procs 1 runs in-process; main() pins the CLI process)
    cpus = sorted(os.sched_getaffinity(0))
    cpu = cpus[(ident[0] - 1) % len(cpus)]
    os.sched_setaffinity(0, {cpu})
    return cpu


def _init(episodes, factory_spec, extra, max_depth, cap):
    for k, v in {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "NUMEXPR_NUM_THREADS": "1"}.items():
        os.environ[k] = v
    cpu = _pin_worker()
    _W.update(episodes=episodes, factory=load_collider_factory(factory_spec), colliders={}, extra=extra,
              max_depth=max_depth, cap=cap, cpu=cpu)


def _collider(scene: str):
    f = _W["factory"]
    if f is None:
        return None
    if scene not in _W["colliders"]:
        _W["colliders"] = {scene: f(scene)}  # keep one scene at a time (episodes are mostly scene-sorted)
    return _W["colliders"][scene]


def _solve_episode(i: int) -> list[dict]:
    ep = _W["episodes"][i]
    col = _collider(ep["scene_id"])
    return [solve_start(ep, vid, c2w, col, ex, _W["max_depth"], _W["cap"])
            for vid, c2w, ex in episode_starts(ep, _W["extra"])]


def quantiles(ds: list[int]) -> dict:
    """The prototype's quantile rule ds[min(n-1, int(p*n))] on sorted lengths."""
    if not ds:
        return {}
    ds = sorted(ds)
    q = lambda p: ds[min(len(ds) - 1, int(p * len(ds)))]  # noqa: E731
    return {"p10": q(.1), "p25": q(.25), "p50": q(.5), "p75": q(.75), "p90": q(.9), "max": ds[-1]}


def summarize(records: list[dict], n_eps: int) -> dict:
    main = [r for r in records if not r["extra"]]
    st = collections.Counter(r["status"] for r in main)
    ds = [r["opt_len"] for r in main if r["status"] == "reached"]
    out = {
        "n_episodes": n_eps, "reached": st["reached"], "capped": st["capped"], "unreachable": st["unreachable"],
        "start_success": st["start_success"], "quantiles": quantiles(ds),
        "median": (sorted(ds)[len(ds) // 2] if ds else None),
        "mean": (round(sum(ds) / len(ds), 4) if ds else None),
        "frac_le3_of_all": (round(sum(d <= 3 for d in ds) / len(main), 4) if main else None),
        "dist": {str(k): v for k, v in sorted(collections.Counter(ds).items())},
        "max_states": max((r["n_states"] for r in main), default=0),
        "max_replay_dev": max((r["replay_dev"] or 0.0 for r in main), default=0.0),
        "by_question_type": {},
    }
    for qt in sorted({r["question_type"] for r in main}):
        rs = [r for r in main if r["question_type"] == qt]
        out["by_question_type"][qt] = {"n": len(rs), "reached": sum(r["status"] == "reached" for r in rs)}
    ex = [r for r in records if r["extra"]]
    if ex:
        se = collections.Counter(r["status"] for r in ex)
        out["extra"] = {"n_starts": len(ex), **{k: se[k] for k in ("reached", "capped", "unreachable",
                                                                    "start_success")}}
    return out


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def run(split: str, out: Path | None = None, limit: int | None = None, procs: int = 1, collides: str | None = None,
        tag: str | None = None, extra: bool = False, max_depth: int = MAX_DEPTH, cap: int = STATE_CAP,
        repo: Path | None = None, quiet: bool = False) -> dict:
    repo = Path(repo) if repo else repo_dir()
    tag = tag or ("nocoll" if not collides else collides.replace(":", "_").replace(".", "_"))
    if out is None:
        if repo.resolve() == LOGIN_TMP_REPO.resolve():
            raise SystemExit("the repo is the read-only login_tmp copy (provenance pending the user's OK): "
                             "pass --out under the scratch dir instead of writing OUT/e3vs/bfs")
        out = paths.E3VS_BFS / tag / split
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = check_repo(repo, splits=(split,))
    eps = load_episodes(split, repo, check=False)
    if limit:
        eps = eps[:limit]
    t0 = time.time()
    if procs <= 1:
        _init(eps, collides, extra, max_depth, cap)
        results = []
        for i in range(len(eps)):
            results.append(_solve_episode(i))
            if not quiet and (i + 1) % 25 == 0:
                print(f"  {split} {i + 1}/{len(eps)} {time.time() - t0:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        hook = getattr(load_collider_factory(collides), "prefork", None)
        if hook is not None:  # e.g. the mesh factory hashes the meshes once here; the forked workers inherit it
            hook(sorted({e["scene_id"] for e in eps}))
        ctx = mp.get_context("fork")
        with ctx.Pool(int(procs), initializer=_init, initargs=(eps, collides, extra, max_depth, cap)) as pool:
            results = list(pool.imap(_solve_episode, range(len(eps)), chunksize=1))
    records = [r for rs in results for r in rs]
    elapsed = time.time() - t0
    summ = summarize(records, len(eps))
    summ.update({
        "split": split, "tag": tag, "collides": collides, "limit": limit, "procs": procs, "extra": extra,
        "max_depth": max_depth, "state_cap": cap, "seconds": round(elapsed, 1),
        "search_order": list(SEARCH_ORDER), "tolerance": {"pos_m": SUCCESS_POS_M, "ang_deg": SUCCESS_ANG_DEG},
        "inputs": inputs,
        "code": code_meta(Path(__file__), Path(G.__file__), Path(sys.modules["vpd.e3vs"].__file__)),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    main = [r for r in records if not r["extra"]]
    _write_jsonl(out / "records.jsonl", records)
    if split == "train":
        _write_jsonl(out / "demo_pool.jsonl", [r for r in records if r["status"] == "reached"])
    _write_json(out / "opt_len.json", {str(r["episode_index"]): r["opt_len"] for r in main})
    _write_json(out / "capped.json", [{"episode_index": r["episode_index"], "scene_id": r["scene_id"],
                                       "status": r["status"], "n_states": r["n_states"]}
                                      for r in main if r["status"] != "reached"])
    _write_json(out / "summary.json", summ)
    if not quiet:
        q = summ["quantiles"]
        print(f"{split} n {len(eps)} time {elapsed:.1f}s reached<=16: {summ['reached']} "
              f"unreached(-1): {summ['unreachable']} capped(-2): {summ['capped']} start_success: "
              f"{summ['start_success']}")
        print(f"  optimal steps p10/25/50/75/90/max: {[q.get(k) for k in ('p10', 'p25', 'p50', 'p75', 'p90')]} "
              f"{q.get('max')}  frac<=3 (of all): {summ['frac_le3_of_all']}  mean {summ['mean']}")
        print(f"  max states {summ['max_states']}  dist {summ['dist']}  max replay dev {summ['max_replay_dev']:.2e}")
        print(f"  wrote {out}")
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=1, help="explicit process count (default 1)")
    ap.add_argument("--collides", default=None, help="MODULE:FUNC factory scene_id -> collides(p0, p1)")
    ap.add_argument("--tag", default=None, help="output tag (default nocoll, or derived from --collides)")
    ap.add_argument("--extra", action="store_true", help="also search from the other unanswerable viewpoints")
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    ap.add_argument("--cap", type=int, default=STATE_CAP)
    ap.add_argument("--repo", type=Path, default=None, help="E3VS checkout (default: vpd.e3vs.repo_dir())")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.collides and a.procs <= 1 and hasattr(os, "sched_setaffinity"):  # in-process search: one CPU
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    run(a.split, out=a.out, limit=a.limit, procs=a.procs, collides=a.collides, tag=a.tag, extra=a.extra,
        max_depth=a.max_depth, cap=a.cap, repo=a.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
