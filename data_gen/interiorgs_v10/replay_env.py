"""Replay generated tasks in a patched copy of the RL env (CPU only, no rendering).

The env decides success with the canonical metric (a geometric projection test)
and collisions with its CollisionDetector, neither of which needs a renderer.
This tool copies ``vagen/`` to a temporary directory, applies the three patches
the v10 rows need (``families/canonical_task_metrics_v2.patch``,
``patches/env_structure_override.patch``, ``patches/env_ignored_objects.patch``),
and replays each sampled task: ``env.reset`` on the row, then one ``env.step``
per assistant turn of the generator's certificate (``turns_audit_only``),
formatted as a model would answer.  It checks that the env declares success
exactly when the d*-th primitive action completes and that no move collides.

    python -m data_gen.interiorgs_v10.replay_env --run OUT \
        --scenes-root /mnt/umm/users/yinbaiqiao/InteriorGS_meta --per-family 40 --json replay.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PATCHES = (HERE / "families" / "canonical_task_metrics_v2.patch",
           HERE / "patches" / "env_structure_override.patch",
           HERE / "patches" / "env_ignored_objects.patch")

ENV_CONFIG = {  # R1-clean-Projective-v0/frozen_v1/train.yaml, minus the renderer
    "render_backend": None, "action_space": "strafe", "enable_auto_termination": True,
    "enable_explicit_done": False, "enable_potential_field": True, "max_actions_per_step": 5,
    "step_translation": 0.3, "step_rotation_deg": 20.0, "max_episode_steps": 12, "turn_budget": 12,
    "image_height": 256, "image_width": 256,
}

_WORKER = r"""
import json, sys
root, tasks_path, scenes_root, out_path, env_config = sys.argv[1:6]
sys.path.insert(0, root)
import vagen
assert vagen.__file__.startswith(root), vagen.__file__
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
cfg = ActiveSpatialEnvConfig(jsonl_path=tasks_path, gs_root=scenes_root, **json.loads(env_config))
env = ActiveSpatialEnv(cfg)
env._render = lambda init_obs=True, task_prompt="", env_feedback="": {"obs_str": "", "multi_modal_data": {}}
rows = [json.loads(l) for l in open(tasks_path) if l.strip()]
out = []
for i, row in enumerate(rows):
    rec = {"task_id": row["task_id"], "scene_id": row["scene_id"], "task_type": row["task_type"],
           "d_star": int(row["difficulty"]["d_star"]),
           "requires_env_override": bool((row.get("collision_convention") or {}).get("requires_env_override")),
           "ignored_object_ids": (row.get("collision_convention") or {}).get("ignored_object_ids") or []}
    try:
        env.reset(seed=i)
        assert env.current_task_id == row["task_id"], (env.current_task_id, row["task_id"])
        turns = row["turns_audit_only"]
        success_step, done_turn, max_coll = None, None, 0
        for t, turn in enumerate(turns):
            _, _, done, info = env.step("<think>replay</think><action>" + "|".join(turn) + "</action>")
            max_coll = max(max_coll, int(info.get("collision_count", 0)))
            if info.get("success") and success_step is None:
                success_step = int(info["env_step"])
            if done:
                done_turn = t
                break
        rec.update(success_step=success_step, done_turn=done_turn, n_turns=len(turns), collisions=max_coll,
                   ok=bool(success_step == rec["d_star"] and done_turn == len(turns) - 1 and max_coll == 0))
    except Exception as exc:
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    out.append(rec)
json.dump(out, open(out_path, "w"))
"""


def patched_env_copy(dst: Path) -> Path:
    shutil.copytree(REPO / "vagen", dst / "vagen", ignore=shutil.ignore_patterns("__pycache__"))
    for patch in PATCHES:
        for extra in (["--dry-run"], []):
            res = subprocess.run(["patch", "-p1", "--batch", "--forward"] + extra, cwd=dst,
                                 input=patch.read_bytes(), capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(f"{patch.name} does not apply: {res.stdout.decode()} {res.stderr.decode()}")
    return dst


def select_rows(rows: Sequence[Dict[str, Any]], per_family: int, per_group: int, seed: int = 0
                ) -> List[Dict[str, Any]]:
    """Up to ``per_family`` rows per task family, plus ``per_group`` rows from scenes that need the
    structure-sign override and every row from scenes with ignored objects."""
    rng = random.Random(seed)
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_family[r["task_type"]].append(r)
    picked: Dict[str, Dict[str, Any]] = {}
    for fam in sorted(by_family):
        for r in rng.sample(by_family[fam], min(per_family, len(by_family[fam]))):
            picked[r["task_id"]] = r
    override = [r for r in rows if (r.get("collision_convention") or {}).get("requires_env_override")]
    for r in rng.sample(override, min(per_group, len(override))):
        picked[r["task_id"]] = r
    for r in rows:
        if (r.get("collision_convention") or {}).get("ignored_object_ids"):
            picked[r["task_id"]] = r
    return sorted(picked.values(), key=lambda r: r["task_id"])


def replay(rows: Sequence[Dict[str, Any]], scenes_root: Path) -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        patched_env_copy(tmp / "env")
        tasks = tmp / "tasks.jsonl"
        tasks.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = tmp / "out.json"
        res = subprocess.run([sys.executable, "-c", _WORKER, str(tmp / "env"), str(tasks), str(scenes_root),
                              str(out), json.dumps(ENV_CONFIG)], capture_output=True, text=True, cwd=str(tmp))
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"replay worker failed: {res.stderr[-3000:]}")
        return json.loads(out.read_text())


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r["task_type"]].append(r)
        if r["requires_env_override"]:
            groups["(scenes needing the structure-sign override)"].append(r)
        if r["ignored_object_ids"]:
            groups["(scenes with ignored objects)"].append(r)
    table = {}
    for g, rs in sorted(groups.items()):
        table[g] = {"tasks": len(rs), "success_exactly_at_d_star": sum(1 for r in rs if r.get("ok")),
                    "success_at_other_step": sum(1 for r in rs if r.get("success_step") not in (None, r["d_star"])),
                    "no_success": sum(1 for r in rs if r.get("success_step") is None and "error" not in r),
                    "with_collision": sum(1 for r in rs if r.get("collisions", 0) > 0),
                    "errors": sum(1 for r in rs if "error" in r)}
    return {"tasks": len(results), "all_ok": all(r.get("ok") for r in results), "groups": table,
            "failures": [r for r in results if not r.get("ok")][:50]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run_all output directory (tasks.jsonl)")
    ap.add_argument("--scenes-root", required=True, type=Path)
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--per-override", type=int, default=40)
    ap.add_argument("--include-reserve", action="store_true", help="also sample tasks_reserve.jsonl")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in (args.run / "tasks.jsonl").read_text().splitlines() if l.strip()]
    if args.include_reserve and (args.run / "tasks_reserve.jsonl").exists():
        rows += [json.loads(l) for l in (args.run / "tasks_reserve.jsonl").read_text().splitlines() if l.strip()]
    results = replay(select_rows(rows, args.per_family, args.per_override), args.scenes_root)
    rep = {"env_config": ENV_CONFIG, "patches": [p.name for p in PATCHES], "summary": summarize(results),
           "results": results}
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))
    return 0 if rep["summary"]["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
