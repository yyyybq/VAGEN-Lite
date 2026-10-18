"""Success states inside the target region, away from its boundary (user pre-authorisation of 2026-10-18).

Shortest-path terminal frames are the first pose that passes every gate, so most of them sit on the success
boundary and the goal question is not asked there (``goal_now`` "yes" below half of the terminals).  For each
training trajectory this module looks for one more success pose near the terminal one that keeps succeeding
after a larger pose change than the label buffer, renders it (at most 10,000 frames, the user's cap) and asks
``goal_now`` on it (answer Yes).

    python -m data_gen.transition_qa.interior plan --sft-dir SFT --scenes-root META --out OUT [--workers 12]
    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot OUT/interior --gs-root PLY   # then submit
    python -m data_gen.transition_qa.interior questions --sft-dir SFT --out OUT
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10.contract import apply_action

from . import generate as GEN
from . import geometry as G
from . import questions as Q

MAX_FRAMES = 10_000                     # the user's cap
STEP_M, STEP_DEG, RINGS = 0.15, 5.0, 3   # candidate offsets: up to 3 steps right / forward / yaw, nearest first
INTERIOR_PERTURB_M = 0.10               # twice the goal-label buffer (0.05 m, 3 degrees)
INTERIOR_PERTURB_DEG = 6.0
_INPUTS: Dict[str, Any] = {}


def offset_pose(c2w: np.ndarray, right_m: float, forward_m: float, yaw_deg: float) -> np.ndarray:
    pose = np.asarray(c2w, float)
    if right_m:
        pose = apply_action(pose, "move_right", step_t=right_m)
    if forward_m:
        pose = apply_action(pose, "move_forward", step_t=forward_m)
    if yaw_deg:
        pose = apply_action(pose, "turn_right", step_r_deg=yaw_deg)
    return pose


def find_interior(geo: G.TaskGeometry, scene: Any, terminal: np.ndarray, key: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    rng = random.Random(Q.stable_int(key, "interior"))
    steps = range(-RINGS, RINGS + 1)
    offsets = []
    for ring in range(1, RINGS + 1):  # ring r: the largest of the three offsets is r steps
        group = [(i * STEP_M, j * STEP_M, k * STEP_DEG) for i in steps for j in steps for k in steps
                 if max(abs(i), abs(j), abs(k)) == ring]
        rng.shuffle(group)
        offsets.extend(group)
    tried = collections.Counter()
    for x, z, d in offsets:
        pose = offset_pose(terminal, x, z, d)
        if not scene.is_pose_legal(pose)[0]:
            tried["illegal"] += 1
            continue
        if not geo.success(pose)["success"]:
            tried["not_success"] += 1
            continue
        ok, robust = geo.success_robust(pose, INTERIOR_PERTURB_M, INTERIOR_PERTURB_DEG)
        if not robust:
            tried["near_boundary"] += 1
            continue
        return pose, {"offset_right_m": x, "offset_forward_m": z, "offset_yaw_deg": d, "tried": dict(tried)}
    return None, {"tried": dict(tried)}


def _plan_scene(scene_id: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    from data_gen.interiorgs_v10.occlusion import face_occluders_for
    from data_gen.interiorgs_v10.scene import load_scene

    cfg = _INPUTS["cfg"]
    scene = load_scene(Path(cfg["scenes_root"]) / scene_id, scene_id, wall_model=cfg["wall_model"])
    occ = face_occluders_for(scene)
    out, funnel = [], collections.Counter()
    for traj in _INPUTS["by_scene"][scene_id]:
        item = _INPUTS["rows"][traj["source_task_id"]]
        geo = G.TaskGeometry(item, scene, occ)
        terminal_id = GEN.frame_id_of(traj["image_paths"][-1])
        frame = _INPUTS["frames"][terminal_id]
        pose, info = find_interior(geo, scene, np.asarray(frame["c2w"], float), item["task_id"])
        if pose is None:
            funnel["not_found"] += 1
            continue
        funnel["found"] += 1
        out.append({"frame_id": f"{item['task_id']}__i0", "kind": "transition_qa_interior", "task_id": item["task_id"],
                    "scene_id": scene_id, "state_id": "i0", "c2w": pose.tolist(), "camera_item": frame["camera_item"],
                    "overlay": {}, "family": item["task_type"], "terminal_frame_id": terminal_id, "search": info})
    return scene_id, out, dict(funnel)


def cmd_plan(args: argparse.Namespace) -> None:
    sft, out = Path(args.sft_dir), Path(args.out) / "interior"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frames.jsonl").exists():
        raise SystemExit(f"{out / 'frames.jsonl'} exists; remove it to plan again")
    _INPUTS["cfg"] = {"scenes_root": str(args.scenes_root), "wall_model": args.wall_model}
    _INPUTS["rows"] = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    _INPUTS["frames"] = {f["frame_id"]: f for f in GEN.read_jsonl(sft / "frames.jsonl")}
    by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for t in GEN.read_jsonl(sft / "sft_trajectories.jsonl"):
        if t["split"] == "train" and not t.get("reserve"):
            by_scene[t["scene_id"]].append(t)
    _INPUTS["by_scene"] = by_scene
    scenes = sorted(by_scene)[: args.limit_scenes or None]
    frames, funnel = [], collections.Counter()
    t0 = time.time()
    import multiprocessing as mp

    with mp.get_context("fork").Pool(max(1, args.workers)) as pool:
        for i, (sid, rows, fn) in enumerate(pool.imap_unordered(_plan_scene, scenes, chunksize=1), 1):
            frames.extend(rows)
            funnel.update(fn)
            if i % 50 == 0 or i == len(scenes):
                print(f"[{i}/{len(scenes)} {time.time() - t0:.0f}s] found {funnel['found']} not found {funnel['not_found']}",
                      flush=True)
    frames.sort(key=lambda f: (f["scene_id"], f["frame_id"]))
    if len(frames) > MAX_FRAMES:
        raise SystemExit(f"{len(frames)} frames, above the user's cap of {MAX_FRAMES}")
    GEN.write_jsonl(out / "frames.jsonl", frames)
    report = {"trajectories": sum(len(v) for v in by_scene.values()), "scenes": len(scenes), **funnel,
              "perturb_m": INTERIOR_PERTURB_M, "perturb_deg": INTERIOR_PERTURB_DEG, "seconds": round(time.time() - t0)}
    (out / "plan_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_questions(args: argparse.Namespace) -> None:
    """goal_now (Yes) on every rendered, non-blank interior frame -> OUT/raw/zz_interior.jsonl (read by build)."""
    from data_gen.interiorgs_v10.pilot import safe_name

    sft, out = Path(args.sft_dir), Path(args.out).resolve()  # absolute image paths: training runs elsewhere
    d = out / "interior"
    rows = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    manifest = json.loads((d / "frames" / "render_manifest.json").read_text())
    ok = {r["frame_id"] for r in manifest["rendered"] if float(r["std"]) >= 2.0}
    funnel = collections.Counter(render_errors=len(manifest["errors"]))
    recs = []
    for f in GEN.read_jsonl(d / "frames.jsonl"):
        image = d / "frames" / "plain" / safe_name(f["frame_id"])
        if f["frame_id"] not in ok or not image.is_file():
            funnel["not_rendered_or_blank"] += 1
            continue
        item = rows[f["task_id"]]
        base = {"pool": "P1", "family": item["task_type"], "task_id": item["task_id"], "scene": f["scene_id"],
                "split": item["split"], "state_id": "i0", "state_pos": -1, "state_role": "interior"}
        text = Q.build_text("goal", item, {}, True, None, f"{item['task_id']}:interior:goal", Q.DEFAULT_CONFIG)
        recs.append(GEN._record(base, "goal_now", "goal", text, [str(image)], {"interior": f["search"]}))
        funnel["goal_now:emitted_interior"] += 1
    GEN.write_jsonl(out / "raw" / "zz_interior.jsonl", recs)
    (out / "raw" / "zz_interior.json").write_text(json.dumps({"funnel": dict(funnel), "checks": {}}, indent=1))
    print(json.dumps(dict(funnel), indent=1))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--sft-dir", required=True)
    a.add_argument("--scenes-root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--wall-model", default="thick")
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--limit-scenes", type=int, default=0)
    a.set_defaults(func=cmd_plan)
    b = sub.add_parser("questions")
    b.add_argument("--sft-dir", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_questions)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
