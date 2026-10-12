"""Static top-down task browser (one HTML file, no server) for spot-checking tasks.

For each scene: room outlines, walls, object footprints (targets highlighted,
ignored annotation errors in their own colour).  For each task: start pose and
heading, the generator's shortest action path, the success region (0.2 m cells
of legal floor where some heading satisfies the env's success test for the
task) and the task text, family and d*.  Filters by scene, family and d*.

    python -m data_gen.interiorgs_v10.make_viewer --run OUT --scenes-root ROOT \
        --scenes-file sample100.txt --extra-scenes 0068_839928 ... \
        --ignored-objects configs/ignored_objects_20261012.json --workers 10 --out viewer.html
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

MIN_FOOTPRINT_M2 = 0.03  # smaller boxes (cups, books ...) are left out unless they are targets or ignored
REGION_SPACING_M = 0.2
YAW_STEP_DEG = 10.0
FAMILY_YAW_WINDOW_DEG = 70.0  # family judges need every target in the 90 deg view: search near the anchor


def scene_geometry(scene: Any, labels: Sequence[Dict[str, Any]], keep_ids: set) -> Dict[str, Any]:
    ignored = set(getattr(scene, "ignored_object_ids", []) or [])
    objs = []
    for o in scene.objects:
        area = float((o.aabb_max[0] - o.aabb_min[0]) * (o.aabb_max[1] - o.aabb_min[1]))
        if area < MIN_FOOTPRINT_M2 and o.id not in keep_ids:
            continue
        objs.append([o.id, o.label, *(round(float(v), 3) for v in (o.aabb_min[0], o.aabb_min[1], o.aabb_max[0],
                                                                    o.aabb_max[1], o.aabb_min[2], o.aabb_max[2]))])
    ign = []
    for item in labels:
        if str(item.get("ins_id")) in ignored:
            pts = np.array([[p["x"], p["y"], p["z"]] for p in item["bounding_box"]], float)
            lo, hi = pts.min(0), pts.max(0)
            ign.append([str(item["ins_id"]), str(item.get("label", "")),
                        *(round(float(v), 3) for v in (lo[0], lo[1], hi[0], hi[1], lo[2], hi[2]))])
    return {
        "rooms": [[[round(float(x), 3), round(float(y), 3)] for x, y in r.polygon] for r in scene.rooms],
        "walls": [[round(float(a[0]), 3), round(float(a[1]), 3), round(float(b[0]), 3), round(float(b[1]), 3)]
                  for a, b in (scene.detector.wall_segments or [])],
        "objects": objs,
        "ignored": ign,
    }


def rooms_at(scene: Any, xys: Sequence[Sequence[float]]) -> List[int]:
    return sorted({int(r.index) for r in scene.rooms for x, y in xys if r.contains(float(x), float(y))})


def success_region(scene: Any, item: Dict[str, Any], spacing: float = REGION_SPACING_M,
                   yaw_step_deg: float = YAW_STEP_DEG, extra_rooms: Sequence[int] = (),
                   known: Sequence[Sequence[float]] = ()) -> List[List[float]]:
    """Centres of legal 0.2 m cells where at least one heading (every 10 deg) succeeds.

    Cells are searched in the targets' rooms and ``extra_rooms`` (the rooms the tasks' paths cross: the
    camera often looks into the targets' room from the next one).  ``known`` successful camera
    positions (the tasks' final poses) are always included, so a region narrower than the grid or
    the heading step is never drawn empty."""
    from .families.common import cached_floor_grid, item_rooms

    rooms = item_rooms(scene, item)
    if rooms is not None:
        rooms = sorted(set(rooms) | {int(r) for r in extra_rooms})
    known_out = [[round(float(x), 3), round(float(y), 3)] for x, y in known]
    pts = cached_floor_grid(scene, spacing, rooms)
    if len(pts) == 0:
        return known_out
    yaws = np.radians(np.arange(0.0, 360.0, yaw_step_deg))
    task_type = item["task_type"]
    if task_type in ("projective_relations", "fov_inclusion"):
        from .endpoints import batch_evaluate

        xs, ys = np.repeat(pts[:, 0], yaws.size), np.repeat(pts[:, 1], yaws.size)
        ok = np.asarray(batch_evaluate(scene, item, xs, ys, np.tile(yaws, len(pts)))["success"], bool)
        hit = ok.reshape(len(pts), yaws.size).any(axis=1)
    else:
        from .contract import pose_from_xy_yaw
        from .families.canonical_v2 import FAMILIES

        fam = FAMILIES[task_type]
        ax, ay = fam.anchor_xy(item)
        hit = np.zeros(len(pts), bool)
        win = math.radians(FAMILY_YAW_WINDOW_DEG)
        for i, (x, y) in enumerate(pts):
            base = math.atan2(ay - y, ax - x)
            for yaw in yaws:
                if abs((yaw - base + math.pi) % (2 * math.pi) - math.pi) > win:
                    continue
                if fam.judge(item, pose_from_xy_yaw(float(x), float(y), float(yaw)))["success"]:
                    hit[i] = True
                    break
    return [[round(float(x), 3), round(float(y), 3)] for x, y in pts[hit]] + known_out


def task_path(row: Dict[str, Any]) -> List[List[float]]:
    """Camera (x, y, yaw deg) at the start and after each certificate action."""
    from .contract import apply_action, yaw_of

    pose = np.asarray(row["init_camera"]["extrinsics"], float)
    out = [[round(float(pose[0, 3]), 3), round(float(pose[1, 3]), 3), round(math.degrees(yaw_of(pose)), 1)]]
    for a in row["difficulty"]["certificate"]:
        pose = apply_action(pose, a)
        out.append([round(float(pose[0, 3]), 3), round(float(pose[1, 3]), 3), round(math.degrees(yaw_of(pose)), 1)])
    return out


def _item_key(row: Dict[str, Any]) -> str:
    from .contract import stable_hash

    return stable_hash({k: row[k] for k in ("task_type", "target_object", "target_region")})[:16]


def build_scene(args: tuple) -> Dict[str, Any]:
    scene_dir, scene_rows, ignored_ids = args
    from .scene import load_scene

    sid = Path(scene_dir).name
    with contextlib.redirect_stdout(io.StringIO()):
        scene = load_scene(Path(scene_dir), sid, ignored_ids=ignored_ids)
    labels = json.loads((Path(scene_dir) / "labels.json").read_text())
    keep = {str(e["id"]) for r in scene_rows for e in r["target_object"]["objects"]}
    paths = {r["task_id"]: task_path(r) for r in scene_rows}
    by_item: Dict[str, List[Dict[str, Any]]] = {}
    for r in scene_rows:
        by_item.setdefault(_item_key(r), []).append(r)
    regions: Dict[str, List[List[float]]] = {}
    for key, rs in by_item.items():
        known = sorted({(p[-1][0], p[-1][1]) for p in (paths[r["task_id"]] for r in rs)})
        crossed = rooms_at(scene, [xy[:2] for r in rs for xy in paths[r["task_id"]]])
        try:
            regions[key] = success_region(scene, rs[0], extra_rooms=crossed, known=known)
        except Exception as exc:  # keep the task visible even if its region cannot be drawn
            regions[key] = [list(xy) for xy in known]
            print(f"[viewer] {sid} {rs[0]['task_id']}: region failed: {exc!r}")
    tasks = []
    for r in scene_rows:
        tasks.append({
            "id": r["task_id"], "family": r["task_type"], "d": int(r["difficulty"]["d_star"]),
            "text": r.get("task_description", ""), "reserve": bool(r.get("reserve")),
            "targets": [str(e["id"]) for e in r["target_object"]["objects"]],
            "target_labels": [e.get("ref") or e.get("label") for e in r["target_object"]["objects"]],
            "path": paths[r["task_id"]], "actions": list(r["difficulty"]["certificate"]), "region": _item_key(r),
        })
    return {"id": sid, "geometry": scene_geometry(scene, labels, keep), "tasks": tasks, "regions": regions,
            "convention": (scene.convention_v2 or {}).get("status")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--scenes-root", required=True, type=Path)
    ap.add_argument("--scenes-file", type=Path, help="one scene id per line")
    ap.add_argument("--extra-scenes", nargs="*", default=[])
    ap.add_argument("--ignored-objects", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    from .run_scene import ignored_object_ids

    ids = [s for s in (args.scenes_file.read_text().split() if args.scenes_file else [])] + list(args.extra_scenes)
    ids = list(dict.fromkeys(ids))
    rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ids}
    for name in ("tasks.jsonl", "tasks_reserve.jsonl"):
        path = args.run / name
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["scene_id"] in rows:
                        rows[r["scene_id"]].append(r)
    jobs = [(str(args.scenes_root / s), rows[s], ignored_object_ids(args.ignored_objects, s)) for s in ids]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        scenes = list(pool.map(build_scene, jobs))
    data = {"run": str(args.run), "region_spacing_m": REGION_SPACING_M, "scenes": scenes}
    html = (Path(__file__).with_name("viewer_template.html").read_text()
            .replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    args.out.write_text(html, encoding="utf-8")
    n_tasks = sum(len(s["tasks"]) for s in scenes)
    print(json.dumps({"scenes": len(scenes), "tasks": n_tasks, "bytes": len(html)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
