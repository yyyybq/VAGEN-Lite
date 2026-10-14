"""Rendered-depth check of terminal frames (round-5 report §8 item 4; decision of 2026-10-14, round 6).

The visibility rule (vagen/envs/active_spatial/target_visibility.py) knows only the labelled boxes and the
structure.json walls.  Geometry with no label (a decorative wall, a pillar, a sloped ceiling) can hide a
target that the rule calls clearly visible.  This check renders each task's terminal frame (the start pose
followed by the task's shortest action sequence) with depth and repeats the rule with one more way to be
hidden: a face sample point counts as hidden when the rendered surface at its pixel lies more than
``max(DEPTH_TOL_M, DEPTH_TOL_REL x depth)`` in front of it (pixels the splats barely cover, accumulated opacity
below ``MIN_ALPHA``, never hide).  A task is flagged when a target or reference object the rule needs is
clearly visible under the rule but not once depth is taken into account.  It only marks tasks; training
data is not changed (decision of 2026-10-14).

    python -m data_gen.interiorgs_v10.depth_check prepare --run RUN/output --split-manifest RUN/split_manifest.json \\
        --splits val,test --out DIR
    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot DIR --gs-root PLY --command "depth_check run"
    python data_gen/interiorgs_v10/pilot_sco.py submit --pilot DIR        # one card; status / delete as usual
    python -m data_gen.interiorgs_v10.depth_check report --pilot DIR
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEPTH_TOL_M = 0.15     # a surface this far in front of a sample point hides it (box fit and splat fuzz)
DEPTH_TOL_REL = 0.10   # ... or this share of the point's depth, whichever is larger
MIN_ALPHA = 0.5        # pixels the splats barely cover say nothing about depth
VERSION = "interiorgs_v10_depth_check_v1"


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def terminal_pose(row: Dict[str, Any]) -> np.ndarray:
    """The start pose followed by the task's certificate (its shortest action sequence)."""
    from .contract import apply_action

    c2w = np.asarray(row["init_camera"]["extrinsics"], float)
    for action in row["difficulty"]["certificate"]:
        c2w = apply_action(c2w, action)
    return c2w


def required_objects(item: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """(name, entry) of every object the visibility rule needs in a success frame: targets and references."""
    from vagen.envs.active_spatial import target_visibility as TV

    out = [(f"target_{'abcd'[slot]}", e) for slot, e in TV.required_targets(item)]
    out += [(f"reference_{'abcd'[slot]}", e) for slot, e in TV.required_references(item)]
    return out


def depth_hidden(points: np.ndarray, c2w: np.ndarray, K: np.ndarray, depth: np.ndarray,
                 alpha: np.ndarray) -> np.ndarray:
    """(S,) bool: sample points (world) hidden by the rendered surface (depth along the viewing axis)."""
    from vagen.envs.active_spatial.target_visibility import MIN_DEPTH_M

    R, C = c2w[:3, :3], c2w[:3, 3]
    cam = (np.asarray(points, float) - C) @ R  # camera coordinates (right, down, forward)
    z = cam[:, 2]
    h, w = depth.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    inside = (z > MIN_DEPTH_M) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui = np.clip(np.nan_to_num(u), 0, w - 1).astype(int)
    vi = np.clip(np.nan_to_num(v), 0, h - 1).astype(int)
    d, a = depth[vi, ui], alpha[vi, ui]
    tol = np.maximum(DEPTH_TOL_M, DEPTH_TOL_REL * z)
    return inside & (a >= MIN_ALPHA) & (d < z - tol)


def check_frame(item: Dict[str, Any], c2w: np.ndarray, depth: np.ndarray, alpha: np.ndarray,
                occ: Any) -> Dict[str, Any]:
    """The visibility rule with and without the rendered depth, per required object, at one terminal pose."""
    from vagen.envs.active_spatial import target_visibility as TV

    K, width, height = TV.item_camera(item)
    c2w = np.asarray(c2w, float)
    objects = {}
    flagged = []
    for name, e in required_objects(item):
        geom = TV.FaceGeometry(e["bbox_min"], e["bbox_max"])
        boxes = TV.blocked_mask(occ, c2w[None, :3, 3], geom.points, geom.lo, geom.hi,
                                exclude_ids=(str(e.get("id", "")),))
        hidden = depth_hidden(geom.points, c2w, K, depth, alpha)[None, :]
        rule = TV.best_face(TV.face_visibility(c2w[None, :3, :3], c2w[None, :3, 3], geom, boxes, K, width, height))
        with_depth = TV.best_face(TV.face_visibility(c2w[None, :3, :3], c2w[None, :3, 3], geom, boxes | hidden, K,
                                                     width, height))
        objects[name] = {"id": str(e.get("id", "")), "label": e.get("label"), "rule_visible": rule["visible"],
                         "depth_visible": with_depth["visible"],
                         "rule_face_fraction": round(rule["visible_fraction"], 3),
                         "depth_face_fraction": round(with_depth["visible_fraction"], 3),
                         "hidden_samples": int(hidden.sum())}
        if rule["visible"] and not with_depth["visible"]:
            flagged.append(name)
    return {"objects": objects, "flagged_objects": flagged, "hidden_by_unlabelled_geometry": bool(flagged)}


# ---------------------------------------------------------------------------
# prepare / run / report
# ---------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.split_manifest).read_text())
    split_of = {s: k.replace("_scenes", "") for k in ("train_scenes", "val_scenes", "test_scenes")
                for s in manifest.get(k, [])}
    want = set(args.splits.split(","))
    rows = [r for r in _jsonl(Path(args.run) / "tasks.jsonl") if split_of.get(r["scene_id"]) in want]
    if args.limit:
        rows = rows[: args.limit]
    frames, tasks = [], []
    for r in rows:
        d = int(r["difficulty"]["d_star"])
        r = dict(r, split=split_of[r["scene_id"]])
        tasks.append(r)
        frames.append({"frame_id": f"{r['task_id']}__s{d}", "task_id": r["task_id"], "scene_id": r["scene_id"],
                       "state_id": f"s{d}", "roles": ["terminal"], "c2w": terminal_pose(r).tolist(),
                       "camera_item": {k: r.get(k) for k in ("task_type", "canonical_task_metric_version",
                                                             "camera_model_version", "init_camera")}})
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    print(json.dumps({"tasks": len(tasks), "by_split": dict(Counter(t["split"] for t in tasks))}))


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - needs a GPU and the 3DGS files
    from PIL import Image

    from vagen.envs.active_spatial.env import runtime_render_camera_parameters
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    from .contract import RENDER_SIZE
    from .pilot import safe_name
    from .scene import load_scene

    pilot = Path(args.pilot)
    tasks = {t["task_id"]: t for t in _jsonl(pilot / "tasks.jsonl")}
    frames = sorted(_jsonl(pilot / "frames.jsonl"), key=lambda f: (f["scene_id"], f["frame_id"]))
    done = {r["task_id"] for r in _jsonl(pilot / "results.jsonl")} if (pilot / "results.jsonl").exists() else set()
    (pilot / "examples").mkdir(exist_ok=True)
    meta_root = Path(args.meta_root) if args.meta_root else Path(args.gs_root)
    unified, scene_id, occ = None, None, None
    t0 = time.time()
    with open(pilot / "results.jsonl", "a") as out:
        for f in frames:
            if f["task_id"] in done:
                continue
            try:
                if f["scene_id"] != scene_id:
                    scene_id = f["scene_id"]
                    if unified is None:
                        unified = UnifiedRenderGS(render_backend="local", gs_root=args.gs_root, scene_id=scene_id,
                                                  gpu_device=0)
                    unified.set_scene(scene_id)  # local mode: drops the previous scene's splats
                    from vagen.envs.active_spatial.target_visibility import OccluderSet

                    occ = OccluderSet.from_detector(load_scene(meta_root / scene_id, scene_id).detector)
                renderer = unified._ensure_local()
                c2w = np.asarray(f["c2w"], float)
                K_native = np.asarray(f["camera_item"]["init_camera"]["intrinsics"], float)
                K, w2c = runtime_render_camera_parameters(f["camera_item"], c2w, K_native, RENDER_SIZE)
                rgb, depth, alpha = renderer.render_depth_from_cam_param(K, w2c, *RENDER_SIZE)
                res = check_frame(tasks[f["task_id"]], c2w, depth, alpha, occ)
                if res["hidden_by_unlabelled_geometry"]:
                    Image.fromarray(rgb).save(pilot / "examples" / safe_name(f["frame_id"]))
                    np.save(pilot / "examples" / (safe_name(f["frame_id"]) + ".depth.npy"), depth.astype(np.float16))
                row = {"task_id": f["task_id"], "scene_id": f["scene_id"], "frame_id": f["frame_id"], **res}
            except Exception as exc:  # keep going; the report counts errors
                row = {"task_id": f["task_id"], "scene_id": f["scene_id"], "error": f"{type(exc).__name__}: {exc}"}
            out.write(json.dumps(row) + "\n")
            out.flush()
    (pilot / "run_done.json").write_text(json.dumps({"version": VERSION, "seconds": round(time.time() - t0, 1),
                                                     "frames": len(frames)}))


def summarize(tasks: Sequence[Dict[str, Any]], results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {t["task_id"]: t for t in tasks}
    ok = [r for r in results if "error" not in r]
    flagged = [r for r in ok if r["hidden_by_unlabelled_geometry"]]
    fam = Counter(by_id[r["task_id"]]["task_type"] for r in ok)
    fam_flag = Counter(by_id[r["task_id"]]["task_type"] for r in flagged)
    split = Counter(by_id[r["task_id"]]["split"] for r in ok)
    split_flag = Counter(by_id[r["task_id"]]["split"] for r in flagged)
    roles = Counter(n.split("_")[0] for r in flagged for n in r["flagged_objects"])
    return {
        "version": VERSION, "tolerance": {"metres": DEPTH_TOL_M, "relative": DEPTH_TOL_REL, "min_alpha": MIN_ALPHA},
        "checked": len(ok), "errors": len(results) - len(ok), "flagged": len(flagged),
        "flagged_share": round(len(flagged) / max(1, len(ok)), 4),
        "by_family": {k: {"checked": fam[k], "flagged": fam_flag[k], "share": round(fam_flag[k] / fam[k], 4)}
                      for k in sorted(fam)},
        "by_split": {k: {"checked": split[k], "flagged": split_flag[k]} for k in sorted(split)},
        "flagged_role": dict(roles),
        "flagged_task_ids": sorted(r["task_id"] for r in flagged),
    }


def cmd_report(args: argparse.Namespace) -> None:
    pilot = Path(args.pilot)
    summary = summarize(_jsonl(pilot / "tasks.jsonl"), _jsonl(pilot / "results.jsonl"))
    (pilot / "depth_check_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "flagged_task_ids"}, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True, help="run_all output folder (tasks.jsonl)")
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--splits", default="val,test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", required=True)
    r.add_argument("--gs-root", required=True)
    r.add_argument("--meta-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_meta",
                   help="labels.json / structure.json per scene (the occluders)")
    s = sub.add_parser("report")
    s.add_argument("--pilot", required=True)
    args = ap.parse_args(argv)
    {"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
