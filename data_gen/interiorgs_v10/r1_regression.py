"""R1 regression: replay R1 tasks in a copy of the env and compare turn-by-turn traces.

Used to show that an env change leaves R1 behaviour unchanged (with the R1 legacy switch on):

    # 1. fixed action sequences per R1 task: the shortest path under the R1 success rule
    #    (BFS, difficulty.make_success_fn) plus two seeded random walks
    python -m data_gen.interiorgs_v10.r1_regression plan --rows R1.jsonl --gs-root DIR --out plan.json
    # 2. replay them in an env copy (ROOT/vagen), recording every turn
    python -m data_gen.interiorgs_v10.r1_regression trace --env-root ROOT --rows R1.jsonl --gs-root DIR \
        --plan plan.json --out before.json [--env-config '{"r1_legacy_success": true}']
    # 3. compare two traces (exit 1 on any difference)
    python -m data_gen.interiorgs_v10.r1_regression compare before.json after.json

A trace records, per task and action sequence and turn: done, reward, the observation text, the
camera pose and every JSON-serialisable info field except wall-clock timings, all compared exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# R1-clean-Projective-v0/frozen_v1/train.yaml env settings, without the renderer
R1_ENV_CONFIG = {
    "render_backend": None, "action_space": "strafe", "enable_auto_termination": True,
    "enable_distance_in_obs": False, "enable_explicit_done": False, "enable_potential_field": True,
    "max_actions_per_step": 5, "near_success_bonus": 0.5, "near_success_threshold": 0.55,
    "potential_field_gamma": 0.99, "potential_field_orientation_weight": 0.3,
    "potential_field_position_weight": 0.7, "potential_field_progress_mode": "potential",
    "potential_field_reward_scale": 1.0, "step_rotation_deg": 20.0, "step_translation": 0.3,
    "success_reward": 5.0, "success_score_threshold": 0.65, "max_episode_steps": 12, "turn_budget": 12,
    "image_height": 256, "image_width": 256,
}
RANDOM_WALKS = 2
RANDOM_WALK_LEN = 9


def plan(rows: Sequence[Dict[str, Any]], gs_root: Path, seed: int = 0) -> List[Dict[str, Any]]:
    from .contract import ACTIONS
    from .difficulty import certify_d_star, make_success_fn
    from .initial_states import chunk_turns
    from .scene import load_scene

    scenes: Dict[str, Any] = {}
    out = []
    for i, row in enumerate(rows):
        sid = str(row["scene_id"])
        if sid not in scenes:
            with contextlib.redirect_stdout(io.StringIO()):
                scenes[sid] = load_scene(gs_root / sid, sid)
        init = np.asarray(row["init_camera"]["extrinsics"], float)
        cert = certify_d_star(scenes[sid], make_success_fn(row), init, d_max=8, max_terminal_checks=0)
        seqs = []
        if cert["certificate"]:
            seqs.append({"kind": "r1_shortest_path", "turns": chunk_turns(list(cert["certificate"]))})
        rng = np.random.default_rng([seed, i])
        for k in range(RANDOM_WALKS):
            acts = [ACTIONS[int(a)] for a in rng.integers(0, len(ACTIONS), RANDOM_WALK_LEN)]
            seqs.append({"kind": f"random_walk_{k}", "turns": [acts[j:j + 3] for j in range(0, len(acts), 3)]})
        out.append({"task_id": row["task_id"], "scene_id": sid, "r1_d_star": cert["d_star"], "sequences": seqs})
    return out


_WORKER = r"""
import json, math, sys
root, rows_path, gs_root, plan_path, out_path, env_config = sys.argv[1:7]
sys.path.insert(0, root)
import vagen
assert vagen.__file__.startswith(root), vagen.__file__
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.envs.active_spatial.utils import c2w_extrinsic_to_se3

def clean(v):
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if hasattr(v, "tolist"):
        return clean(v.tolist())
    if v is None or isinstance(v, (bool, int, str)):
        return v
    return repr(type(v))

SKIP = ("time", "latency", "seconds", "duration")
cfg = ActiveSpatialEnvConfig(jsonl_path=rows_path, gs_root=gs_root, **json.loads(env_config))
env = ActiveSpatialEnv(cfg)
plan = json.load(open(plan_path))
out = []
for i, task in enumerate(plan):
    for seq in task["sequences"]:
        obs, info0 = env.reset(seed=i)
        assert env.current_task_id == task["task_id"], (env.current_task_id, task["task_id"])
        rec = {"task_id": task["task_id"], "kind": seq["kind"], "reset_obs": obs.get("obs_str"), "turns": []}
        for turn in seq["turns"]:
            obs, reward, done, info = env.step("<think>replay</think><action>" + "|".join(turn) + "</action>")
            pose = c2w_extrinsic_to_se3(env.view_engine.get_pose())
            rec["turns"].append({
                "actions": turn, "done": bool(done), "reward": repr(float(reward)), "obs": obs.get("obs_str"),
                "pose": [repr(float(x)) for x in pose],
                "info": {k: clean(v) for k, v in sorted(info.items()) if not any(s in k for s in SKIP)},
            })
            if done:
                break
        out.append(rec)
json.dump(out, open(out_path, "w"))
"""


def trace(env_root: Path, rows_path: Path, gs_root: Path, plan_path: Path, out: Path,
          env_config: Optional[Dict[str, Any]] = None) -> None:
    cfg = dict(R1_ENV_CONFIG)
    cfg.update(env_config or {})
    res = subprocess.run([sys.executable, "-c", _WORKER, str(env_root), str(rows_path), str(gs_root),
                          str(plan_path), str(out), json.dumps(cfg)], capture_output=True, text=True)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f"trace worker failed: {res.stderr[-3000:]}")


def compare(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    diffs = []
    for x, y in zip(a, b):
        if x != y:
            where = "reset_obs" if x.get("reset_obs") != y.get("reset_obs") else "turns"
            if where == "turns":
                for t, (u, v) in enumerate(zip(x["turns"], y["turns"])):
                    if u != v:
                        keys = sorted(k for k in set(u) | set(v) if u.get(k) != v.get(k))
                        if "info" in keys:
                            keys += [f"info.{k}" for k in sorted(set(u["info"]) | set(v["info"]))
                                     if u["info"].get(k) != v["info"].get(k)]
                        where = f"turn{t}:{','.join(keys)}"
                        break
                else:
                    where = f"turn_count {len(x['turns'])} vs {len(y['turns'])}"
            diffs.append({"task_id": x["task_id"], "kind": x["kind"], "where": where})
    turns = sum(len(x["turns"]) for x in a)
    successes = sum(1 for x in a for t in x["turns"] if t["info"].get("success") is True)
    return {"episodes": [len(a), len(b)], "turns": turns, "episodes_reaching_success": successes,
            "identical": len(a) == len(b) and not diffs, "differences": len(diffs), "first_differences": diffs[:20]}


def _rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--rows", required=True, type=Path)
    p.add_argument("--gs-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    t = sub.add_parser("trace")
    t.add_argument("--env-root", required=True, type=Path, help="directory that contains the vagen package")
    t.add_argument("--rows", required=True, type=Path)
    t.add_argument("--gs-root", required=True, type=Path)
    t.add_argument("--plan", required=True, type=Path)
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--env-config", default="{}", help="JSON of extra ActiveSpatialEnvConfig fields")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        args.out.write_text(json.dumps(plan(_rows(args.rows), args.gs_root), indent=1))
    elif args.cmd == "trace":
        trace(args.env_root, args.rows, args.gs_root, args.plan, args.out, json.loads(args.env_config))
    else:
        rep = compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
        print(json.dumps(rep, indent=1))
        return 0 if rep["identical"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
