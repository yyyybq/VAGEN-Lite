"""Human-labelling pilot (user decisions of 2026-10-13): a stratified task sample, examples near the visibility
thresholds, the frames to render and the top-down views.  The labelling site is built by ``pilot_site.py``.

    # CPU: choose tasks and threshold examples, write the frame list
    python -m data_gen.interiorgs_v10.pilot select --run OUT --scenes-root META \
        --no-task-objects configs/no_task_objects_20261013.json --out PILOT
    # GPU (one card): render every frame at the training resolution, plain and with boxes drawn
    python -m data_gen.interiorgs_v10.pilot render --pilot PILOT --gs-root SCENES_WITH_PLY
    # CPU: one top-down view per task
    python -m data_gen.interiorgs_v10.pilot topdown --pilot PILOT --scenes-root META \
        --no-task-objects configs/no_task_objects_20261013.json
    # CPU: the 2026-10-14 check page instead (then render, topdown and pilot_site --mode check)
    python -m data_gen.interiorgs_v10.pilot check --run OUT --out CHECK

Selection
---------
* Tasks: ``FAMILY_QUOTA`` tasks per family (formal tasks first, reserve tasks if a stratum is short), spread
  evenly over the d* bands 1 / 2-3 / 4-6 / 7-8 and, inside a band, round-robin over scenes.
* Threshold examples: (target, camera pose) pairs whose best face lies near the initial visibility thresholds,
  on both sides, binned by the face's visible fraction (area already large enough) and by its on-screen area
  (fraction already large enough); objects that are built into furniture or walls (fridges, ovens, ...) form a
  separate group.  Poses are legal camera positions in the target's room facing it within +-35 degrees.
  The measured fractions / areas are stored in ``threshold_examples.jsonl`` and are never shown to the labeller.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FAMILY_QUOTA = {"projective_relations": 40, "absolute_positioning": 32, "screen_occupancy": 32, "equidistance": 32,
                "occlusion_alignment": 32, "fov_inclusion": 32}
BANDS = ((1, 1), (2, 3), (4, 6), (7, 8))
# (visible-fraction bin with area >= the area threshold) and (area bin with fraction >= the fraction threshold)
FRACTION_BINS = ((0.15, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.80))
AREA_BINS = ((0.0008, 0.0015), (0.0015, 0.0030), (0.0030, 0.0060))
PER_FRACTION_BIN = 24
PER_AREA_BIN = 16
PER_EMBEDDED_BIN = 8
EMBEDDED_WORDS = ("refrigerator", "fridge", "freezer", "oven", "dishwasher", "microwave", "washing machine",
                  "washer", "dryer", "range hood", "cooker hood", "built-in", "built in")
RENDER_SIZE = (256, 256)


def is_embedded(label: Any) -> bool:
    text = " ".join(str(label or "").lower().split())
    return any(w in text for w in EMBEDDED_WORDS)


def band_of(d: int) -> Optional[Tuple[int, int]]:
    for lo, hi in BANDS:
        if lo <= d <= hi:
            return (lo, hi)
    return None


def _read_rows(run: Path) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for name, kind in (("tasks.jsonl", "formal"), ("tasks_reserve.jsonl", "reserve")):
        p = run / name
        if p.exists():
            out += [(kind, json.loads(l)) for l in p.read_text().splitlines() if l.strip()]
    return out


def select_tasks(rows: Sequence[Tuple[str, Dict[str, Any]]], quota: Dict[str, int] = FAMILY_QUOTA,
                 seed: int = 0) -> List[Dict[str, Any]]:
    """Stratified sample: per family, an even share per d* band, round-robin over scenes; formal before reserve."""
    rng = np.random.default_rng(seed)
    picked: List[Dict[str, Any]] = []
    for fam, n in quota.items():
        strata: Dict[Tuple[int, int], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for kind, r in rows:
            b = band_of(int(r["difficulty"]["d_star"]))
            if r["task_type"] == fam and b is not None:
                strata[b].append((kind, r))
        want = {b: n // len(BANDS) + (1 if i < n % len(BANDS) else 0) for i, b in enumerate(BANDS)}
        chosen: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for b in BANDS:
            out: List[Dict[str, Any]] = []
            for pool_kind in ("formal", "reserve"):  # formal tasks first, reserve tasks only to fill a band
                by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for kind, r in strata.get(b, []):
                    if kind == pool_kind:
                        by_scene[r["scene_id"]].append(dict(r, pilot_pool=kind))
                scenes = sorted(by_scene)
                rng.shuffle(scenes)
                for sc in scenes:
                    order = rng.permutation(len(by_scene[sc]))
                    by_scene[sc] = [by_scene[sc][i] for i in order]
                while len(out) < want[b] and any(by_scene[sc] for sc in scenes):
                    for sc in scenes:
                        if by_scene[sc] and len(out) < want[b]:
                            out.append(by_scene[sc].pop(0))
            chosen[b] = out
        short = n - sum(len(v) for v in chosen.values())
        if short > 0:  # move the shortfall to bands that still have tasks
            used = {r["task_id"] for v in chosen.values() for r in v}
            rest = [dict(r, pilot_pool=kind) for kind, r in rows if r["task_type"] == fam
                    and band_of(int(r["difficulty"]["d_star"])) is not None and r["task_id"] not in used]
            rest.sort(key=lambda r: (r["pilot_pool"] != "formal", r["task_id"]))
            chosen[("extra", 0)] = rest[:short]
        for v in chosen.values():
            picked.extend(v)
    return picked


def task_frames(row: Dict[str, Any], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Start, final and SFT turn-boundary frames of a task, with the review overlay boxes."""
    from .admission import overlay_boxes, required_review_frames
    from .qa_states import certificate_poses

    poses = certificate_poses(row)
    out = []
    for sid, roles in required_review_frames(row, policy).items():
        k = int(sid[1:])
        out.append({"frame_id": f"{row['task_id']}__{sid}", "kind": "task", "task_id": row["task_id"],
                    "scene_id": row["scene_id"], "state_id": sid, "roles": roles,
                    "c2w": np.asarray(poses[k]).tolist(), "camera_item": camera_item(row),
                    "overlay": overlay_boxes(row, poses[k])})
    return out


def camera_item(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: row.get(k) for k in ("task_type", "canonical_task_metric_version", "camera_model_version",
                                     "init_camera")}


def _example_overlay(entry: Dict[str, Any], c2w: np.ndarray, K: np.ndarray) -> Dict[str, Any]:
    from vagen.envs.active_spatial.visual_bbox_metrics import _project_object

    p = _project_object(entry, np.asarray(c2w, float), np.asarray(K, float), *RENDER_SIZE)
    return {"A": {"bbox": p.get("bbox") if p.get("visible") else None, "color": "red",
                  "ref": entry.get("ref") or f"the {entry.get('label')}", "label": entry.get("label"),
                  "image_size": list(RENDER_SIZE)}}


def candidate_poses(scene: Any, obj: Any, rng: np.random.Generator, n: int = 60) -> List[np.ndarray]:
    from .contract import pose_from_xy_yaw
    from .sampling import floor_grid

    rooms = [obj.room_index] if getattr(obj, "room_index", None) is not None else None
    pts = floor_grid(scene, 0.3, rooms)
    if len(pts) == 0:
        return []
    c = np.asarray(obj.center, float)
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    pts = pts[(d >= 0.6) & (d <= 5.0)]
    if len(pts) == 0:
        return []
    idx = rng.permutation(len(pts))[:n]
    out = []
    for i in idx:
        x, y = float(pts[i][0]), float(pts[i][1])
        yaw = math.atan2(c[1] - y, c[0] - x) + math.radians(float(rng.uniform(-35.0, 35.0)))
        out.append(pose_from_xy_yaw(x, y, yaw))
    return out


def select_threshold_examples(scenes: Dict[str, Any], targets: Dict[str, List[Any]], embedded: Dict[str, List[Any]],
                              seed: int = 0) -> List[Dict[str, Any]]:
    """Binned (target, pose) examples near the initial thresholds (see module docstring)."""
    from vagen.envs.active_spatial import target_visibility as TV

    from .occlusion import canonical_intrinsics, face_occluders_for

    K = canonical_intrinsics()
    rng = np.random.default_rng(seed)
    measured: Dict[str, List[Dict[str, Any]]] = {"general": [], "embedded": []}
    for group, pool in (("general", targets), ("embedded", embedded)):
        for sid in sorted(pool):
            scene = scenes[sid]
            occ = face_occluders_for(scene)
            for obj in pool[sid]:
                entry = obj.target_object_entry()
                for pose in candidate_poses(scene, obj, rng):
                    v = TV.object_visibility(pose, entry, occ, K, *RENDER_SIZE)
                    if v["visible_fraction"] < FRACTION_BINS[0][0]:
                        continue
                    measured[group].append({"scene_id": sid, "target": entry, "c2w": pose.tolist(),
                                            "metrics": {"visible_fraction": v["visible_fraction"],
                                                        "area_fraction": v["area_fraction"], "face": v["face"],
                                                        "visible_under_initial_thresholds": v["visible"]}})
    frac_t, area_t = TV.MIN_FACE_VISIBLE_FRACTION, TV.MIN_FACE_AREA_FRACTION
    bins = [("fraction", lo, hi, PER_FRACTION_BIN, "general") for lo, hi in FRACTION_BINS]
    bins += [("area", lo, hi, PER_AREA_BIN, "general") for lo, hi in AREA_BINS]
    bins += [("fraction", lo, hi, PER_EMBEDDED_BIN, "embedded") for lo, hi in FRACTION_BINS]
    out: List[Dict[str, Any]] = []
    used_ids = set()  # an example sits in at most one bin (fraction and area bins overlap)
    for axis, lo, hi, n, group in bins:
        if axis == "fraction":
            # the area must not decide (embedded: area at least the lower area bin)
            amin = area_t if group == "general" else AREA_BINS[1][0]
            pool = [m for m in measured[group] if lo <= m["metrics"]["visible_fraction"] < hi
                    and m["metrics"]["area_fraction"] >= amin]
        else:
            pool = [m for m in measured[group] if lo <= m["metrics"]["area_fraction"] < hi
                    and m["metrics"]["visible_fraction"] >= frac_t]
        taken: List[Dict[str, Any]] = []
        per_target: Counter = Counter()
        for i in rng.permutation(len(pool)):
            m = pool[i]
            key = (m["scene_id"], m["target"]["id"])
            if per_target[key] >= 2 or id(m) in used_ids:
                continue
            per_target[key] += 1
            used_ids.add(id(m))
            taken.append(m)
            if len(taken) >= n:
                break
        for m in taken:
            out.append(dict(m, group=group, bin={"axis": axis, "lo": lo, "hi": hi}))
    for i, ex in enumerate(out):
        ex["example_id"] = f"vis{i:04d}"
        ex["overlay"] = _example_overlay(ex["target"], np.asarray(ex["c2w"]), K)
    return out


def example_frame(ex: Dict[str, Any]) -> Dict[str, Any]:
    from .contract import CAMERA_MODEL_VERSION, CANONICAL_TASK_METRIC_V2, NATIVE_K

    item = {"task_type": "absolute_positioning", "canonical_task_metric_version": CANONICAL_TASK_METRIC_V2,
            "camera_model_version": CAMERA_MODEL_VERSION,
            "init_camera": {"intrinsics": [list(map(float, r)) for r in NATIVE_K]}}
    return {"frame_id": ex["example_id"], "kind": "threshold", "example_id": ex["example_id"],
            "scene_id": ex["scene_id"], "c2w": ex["c2w"], "camera_item": item, "overlay": ex["overlay"]}


def cmd_select(args: argparse.Namespace) -> None:
    from .admission import load_policy
    from .objects import candidate_objects
    from .run_scene import ignored_object_ids
    from .scene import load_scene

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(Path(args.run))
    tasks = select_tasks(rows, seed=args.seed)
    policy = load_policy()
    frames = [f for r in tasks for f in task_frames(r, policy)]
    # threshold examples: objects that are task targets in the run (and built-in objects) of the pilot scenes
    scene_ids = sorted({r["scene_id"] for r in tasks})
    used = defaultdict(set)
    for _, r in rows:
        for e in r["target_object"]["objects"]:
            used[r["scene_id"]].add(str(e["id"]))
    scenes, targets, embedded = {}, {}, {}
    for sid in scene_ids:
        with contextlib.redirect_stdout(io.StringIO()):
            scene = load_scene(Path(args.scenes_root) / sid, sid,
                               no_task_ids=ignored_object_ids(args.no_task_objects, sid))
        scenes[sid] = scene
        kept, _ = candidate_objects(scene)
        targets[sid] = [o for o in kept if o.id in used[sid]]
        embedded[sid] = [o for o in kept if is_embedded(o.label)]
    examples = select_threshold_examples(scenes, targets, {s: v for s, v in embedded.items() if v}, seed=args.seed)
    frames += [example_frame(ex) for ex in examples]
    (out / "pilot_tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
    (out / "threshold_examples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    (out / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    report = {
        "tasks": len(tasks), "by_family": dict(Counter(r["task_type"] for r in tasks)),
        "by_band": dict(Counter(str(band_of(int(r["difficulty"]["d_star"]))) for r in tasks)),
        "by_pool": dict(Counter(r["pilot_pool"] for r in tasks)), "scenes": len(scene_ids),
        "max_tasks_per_scene": max(Counter(r["scene_id"] for r in tasks).values()) if tasks else 0,
        "task_frames": sum(1 for f in frames if f["kind"] == "task"),
        "threshold_examples": len(examples),
        "threshold_by_bin": dict(Counter(f"{e['group']}:{e['bin']['axis']}:{e['bin']['lo']}-{e['bin']['hi']}"
                                         for e in examples)),
        "embedded_labels": dict(Counter(e["target"]["label"] for e in examples if e["group"] == "embedded")),
    }
    (out / "selection_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------
# rendering (GPU job)
# ---------------------------------------------------------------------------
def safe_name(frame_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in frame_id) + ".png"


def draw_overlay(image: Any, overlay: Dict[str, Dict[str, Any]]) -> Any:
    """Copy of ``image`` with each slot's projected box outlined (A red, B blue, ...)."""
    from PIL import ImageDraw

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for slot, spec in sorted(overlay.items()):
        box = spec.get("bbox")
        if not box:
            continue
        sw, sh = spec.get("image_size") or list(RENDER_SIZE)
        fx, fy = img.size[0] / float(sw), img.size[1] / float(sh)
        draw.rectangle([box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy], outline=spec.get("color", "white"),
                       width=2)
        draw.text((box[0] * fx + 3, box[1] * fy + 2), slot, fill=spec.get("color", "white"))
    return img


async def _render_all(frames: List[Dict[str, Any]], gs_root: str, out: Path, gpu: int) -> Dict[str, Any]:
    from vagen.envs.active_spatial.env import runtime_render_camera_parameters
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    (out / "plain").mkdir(parents=True, exist_ok=True)
    (out / "boxes").mkdir(parents=True, exist_ok=True)
    renderer = None
    done, errors = [], []
    frames = sorted(frames, key=lambda f: (f["scene_id"], f["frame_id"]))
    try:
        for f in frames:
            try:
                scene = str(f["scene_id"])
                if renderer is None or renderer.cfg.scene_id != scene:
                    if renderer is not None:
                        await renderer.close()
                    renderer = UnifiedRenderGS(render_backend="local", gs_root=gs_root, scene_id=scene, gpu_device=gpu)
                pose = np.asarray(f["c2w"], dtype=np.float64)
                K_native = np.asarray(f["camera_item"]["init_camera"]["intrinsics"], dtype=np.float64)
                K, w2c = runtime_render_camera_parameters(f["camera_item"], pose, K_native, RENDER_SIZE)
                image = await renderer.render_image_from_cam_param(K, w2c, *RENDER_SIZE)
                if image.size != RENDER_SIZE:
                    raise ValueError(f"renderer returned {image.size}")
                name = safe_name(f["frame_id"])
                image.save(out / "plain" / name)
                if f.get("overlay"):  # SFT frames have none: no boxed copy
                    draw_overlay(image, f["overlay"]).save(out / "boxes" / name)
                arr = np.asarray(image.convert("RGB"), dtype=float)
                done.append({"frame_id": f["frame_id"], "file": name, "mean": float(arr.mean()),
                             "std": float(arr.std())})
            except Exception as exc:  # keep going; the manifest lists failures
                errors.append({"frame_id": f.get("frame_id"), "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if renderer is not None:
            await renderer.close()
    return {"rendered": done, "errors": errors}


def cmd_render(args: argparse.Namespace) -> None:
    import asyncio

    pilot = Path(args.pilot)
    frames = [json.loads(l) for l in (pilot / "frames.jsonl").read_text().splitlines() if l.strip()]
    res = asyncio.run(_render_all(frames, args.gs_root, pilot / "frames", args.gpu))
    (pilot / "frames" / "render_manifest.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"frames": len(frames), "rendered": len(res["rendered"]), "errors": len(res["errors"]),
                      "blank": sum(1 for r in res["rendered"] if r["std"] < 2.0)}))


# ---------------------------------------------------------------------------
# top-down views
# ---------------------------------------------------------------------------
TARGET_COLORS = {"A": "#d0233b", "B": "#1f6fd1", "C": "#1f9a7f"}


def draw_topdown(scene: Any, labels: List[Dict[str, Any]], row: Dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, Polygon, Rectangle

    from .make_viewer import scene_geometry, task_path

    targets = [str(e["id"]) for e in row["target_object"]["objects"]]
    geo = scene_geometry(scene, labels, set(targets))
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    for poly in geo["rooms"]:
        ax.add_patch(Polygon(poly, closed=True, facecolor="#eef1ea", edgecolor="#b9c0b3", lw=0.8))
    for x0, y0, x1, y1 in geo["walls"]:
        ax.plot([x0, x1], [y0, y1], color="#2f3337", lw=1.6)
    for oid, label, x0, y0, x1, y1, z0, z1 in geo["objects"]:
        if oid in targets:
            continue
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#9aa1a9", lw=0.6))
    for oid, label, x0, y0, x1, y1, z0, z1 in geo["ignored"]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e07b00", lw=1.0, ls="--"))
    by_id = {o[0]: o for o in geo["objects"]}
    for slot, oid in zip("ABC", targets):
        if oid in by_id:
            _, label, x0, y0, x1, y1, _, _ = by_id[oid]
            color = TARGET_COLORS[slot]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, alpha=0.25, edgecolor=color, lw=1.6))
            ax.text(x1, y1, f" {slot}", color=color, fontsize=10, weight="bold")
    for slot, e in zip("ABC", row["target_object"]["objects"]):  # reference objects of the descriptions
        ref = e.get("ref_anchor")
        if ref:
            (x0, y0), (x1, y1) = ref["bbox_min"][:2], ref["bbox_max"][:2]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#2fa84f", alpha=0.2, edgecolor="#2fa84f",
                                   lw=1.4, ls="--"))
            ax.text(x1, y0, f" ref {slot}", color="#2fa84f", fontsize=8, weight="bold")
    pts = np.asarray(task_path(row), float)
    ax.plot(pts[:, 0], pts[:, 1], color="#7a3cc4", lw=1.4, marker="o", ms=2.5)
    for (x, y, yaw), color in ((pts[0], "#111111"), (pts[-1], "#7a3cc4")):
        dx, dy = 0.45 * math.cos(math.radians(yaw)), 0.45 * math.sin(math.radians(yaw))
        ax.add_patch(FancyArrow(x, y, dx, dy, width=0.06, head_width=0.22, head_length=0.18, color=color))
    xs = np.concatenate([np.asarray(p)[:, 0] for p in geo["rooms"]] + [pts[:, 0]]) if geo["rooms"] else pts[:, 0]
    ys = np.concatenate([np.asarray(p)[:, 1] for p in geo["rooms"]] + [pts[:, 1]]) if geo["rooms"] else pts[:, 1]
    ax.set_xlim(xs.min() - 0.5, xs.max() + 0.5)
    ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_topdown(args: argparse.Namespace) -> None:
    from .run_scene import ignored_object_ids
    from .scene import load_scene

    pilot = Path(args.pilot)
    tasks = [json.loads(l) for l in (pilot / "pilot_tasks.jsonl").read_text().splitlines() if l.strip()]
    (pilot / "topdown").mkdir(exist_ok=True)
    cache: Dict[str, Any] = {}
    for r in tasks:
        sid = r["scene_id"]
        if sid not in cache:
            with contextlib.redirect_stdout(io.StringIO()):
                scene = load_scene(Path(args.scenes_root) / sid, sid,
                                   no_task_ids=ignored_object_ids(args.no_task_objects, sid))
            cache[sid] = (scene, json.loads((Path(args.scenes_root) / sid / "labels.json").read_text()))
        draw_topdown(*cache[sid], r, pilot / "topdown" / f"{r['task_id']}.png")
    print(json.dumps({"topdown": len(tasks)}))


# ---------------------------------------------------------------------------
# check page (2026-10-14): did the three fixes work?
# ---------------------------------------------------------------------------
# group -> number of tasks: left/right closest to the 45-degree limit, name only, room, reference object
CHECK_GROUPS = (("lateral_near_45", 10), ("name_only", 10), ("room", 10), ("reference", 10))


def ref_methods(row: Dict[str, Any]) -> List[Optional[str]]:
    return [e.get("ref_method") for e in row["target_object"]["objects"]]


def final_lateral_angle(row: Dict[str, Any]) -> Optional[float]:
    from vagen.envs.active_spatial.target_visibility import lateral_gate

    from .qa_states import certificate_poses

    return lateral_gate(row, certificate_poses(row)[-1]).get("angle_deg")


def select_check_tasks(rows: Sequence[Tuple[str, Dict[str, Any]]],
                       groups: Sequence[Tuple[str, int]] = CHECK_GROUPS) -> List[Dict[str, Any]]:
    """Formal tasks (reserve only if a group is short) for the check page, round-robin over scenes and
    families inside a group, no task twice."""
    from .referring import METHOD_ANCHOR, METHOD_ROOM, METHOD_UNIQUE

    def member(group: str, r: Dict[str, Any]) -> Optional[float]:
        """Sort key (smaller first) if ``r`` belongs to ``group``, else None."""
        methods = ref_methods(r)
        if group == "lateral_near_45":
            if r["task_type"] != "projective_relations":
                return None
            angle = final_lateral_angle(r)
            return None if angle is None else 45.0 - angle  # closest to the limit first
        if group == "name_only":
            return 0.0 if methods and all(m == METHOD_UNIQUE for m in methods) else None
        if group == "room":
            return 0.0 if METHOD_ROOM in methods else None
        if group == "reference":
            return 0.0 if METHOD_ANCHOR in methods else None
        raise ValueError(group)

    picked: List[Dict[str, Any]] = []
    taken = set()
    for group, n in groups:
        chosen: List[Dict[str, Any]] = []
        for pool in ("formal", "reserve"):
            cands = []
            for kind, r in rows:
                if kind != pool or r["task_id"] in taken:
                    continue
                key = member(group, r)
                if key is not None:
                    cands.append((key, r["task_id"], r))
            cands.sort(key=lambda c: (c[0], c[1]))
            scenes_used: Counter = Counter()
            fams_used: Counter = Counter()
            while len(chosen) < n and cands:
                # the best-ranked candidate among those from the least-used scene and family
                best = min(range(len(cands)), key=lambda i: (scenes_used[cands[i][2]["scene_id"]],
                                                             fams_used[cands[i][2]["task_type"]], i))
                _, tid, r = cands.pop(best)
                scenes_used[r["scene_id"]] += 1
                fams_used[r["task_type"]] += 1
                chosen.append(dict(r, check_group=group, pilot_pool=pool,
                                   check_lateral_angle_deg=(final_lateral_angle(r)
                                                            if r["task_type"] == "projective_relations" else None)))
                taken.add(tid)
            if len(chosen) >= n:
                break
        picked.extend(chosen)
    return picked


def reference_overlay(row: Dict[str, Any], c2w: Any) -> Dict[str, Dict[str, Any]]:
    """Green boxes for the reference objects of the targets' descriptions ("RA": A's reference)."""
    from vagen.envs.active_spatial.visual_bbox_metrics import _project_object

    from .occlusion import canonical_intrinsics

    out: Dict[str, Dict[str, Any]] = {}
    for slot, e in zip("ABCD", row["target_object"]["objects"]):
        ref = e.get("ref_anchor")
        if not ref:
            continue
        p = _project_object(ref, np.asarray(c2w, float), np.asarray(canonical_intrinsics(), float), *RENDER_SIZE)
        out["R" + slot] = {"bbox": p.get("bbox") if p.get("visible") else None, "color": "lime",
                           "ref": f"the {ref.get('label')}", "label": ref.get("label"),
                           "image_size": list(RENDER_SIZE)}
    return out


def check_frames(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Start and end frame of a check task, targets and reference objects boxed."""
    from .admission import overlay_boxes
    from .qa_states import certificate_poses

    poses = certificate_poses(row)
    d = len(poses) - 1
    out = []
    for k, role in ((0, "start"), (d, "terminal")):
        overlay = dict(overlay_boxes(row, poses[k]))
        overlay.update(reference_overlay(row, poses[k]))
        out.append({"frame_id": f"{row['task_id']}__s{k}", "kind": "task", "task_id": row["task_id"],
                    "scene_id": row["scene_id"], "state_id": f"s{k}", "roles": [role],
                    "c2w": np.asarray(poses[k]).tolist(), "camera_item": camera_item(row), "overlay": overlay})
    return out


def cmd_check(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = select_check_tasks(_read_rows(Path(args.run)))
    frames = [f for r in tasks for f in check_frames(r)]
    (out / "pilot_tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
    (out / "threshold_examples.jsonl").write_text("")
    (out / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    report = {"tasks": len(tasks), "by_group": dict(Counter(r["check_group"] for r in tasks)),
              "by_family": dict(Counter(r["task_type"] for r in tasks)),
              "by_pool": dict(Counter(r["pilot_pool"] for r in tasks)),
              "scenes": len({r["scene_id"] for r in tasks}), "frames": len(frames),
              "lateral_angles_deg": sorted(round(r["check_lateral_angle_deg"], 1) for r in tasks
                                           if r["check_group"] == "lateral_near_45")}
    (out / "selection_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


AUDIT_GROUP = "audit_r6"


def audit_quota(families: Sequence[str], n: int) -> Dict[str, int]:
    """``n`` tasks over the families as evenly as possible (the first families in order get the remainder)."""
    fams = sorted(families)
    return {f: n // len(fams) + (1 if i < n % len(fams) else 0) for i, f in enumerate(fams)}


def cmd_audit(args: argparse.Namespace) -> None:
    """Round-6 human check (decision of 2026-10-14): formal val and test tasks, stratified by family and d*,
    the check page's questions (pilot_site --mode audit), start and end frame of each task."""
    manifest = json.loads(Path(args.split_manifest).read_text())
    split_of = {s: k.replace("_scenes", "") for k in ("train_scenes", "val_scenes", "test_scenes")
                for s in manifest.get(k, [])}
    rows = [(kind, dict(r, split=split_of.get(r["scene_id"]))) for kind, r in _read_rows(Path(args.run))
            if kind == "formal" and split_of.get(r["scene_id"]) in ("val", "test")]
    quota = audit_quota({r["task_type"] for _, r in rows}, args.n)
    tasks = [dict(r, check_group=AUDIT_GROUP,
                  check_lateral_angle_deg=final_lateral_angle(r) if r["task_type"] == "projective_relations" else None)
             for r in select_tasks(rows, quota, seed=args.seed)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = [f for r in tasks for f in check_frames(r)]
    (out / "pilot_tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
    (out / "threshold_examples.jsonl").write_text("")
    (out / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    report = {"tasks": len(tasks), "seed": args.seed, "quota": quota,
              "by_family": dict(Counter(r["task_type"] for r in tasks)),
              "by_split": dict(Counter(r["split"] for r in tasks)),
              "by_band": dict(Counter(str(band_of(int(r["difficulty"]["d_star"]))) for r in tasks)),
              "by_method": dict(Counter(m for r in tasks for m in ref_methods(r))),
              "scenes": len({r["scene_id"] for r in tasks}), "frames": len(frames)}
    (out / "selection_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--out", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--run", required=True)
    a.add_argument("--split-manifest", required=True)
    a.add_argument("--n", type=int, default=100)
    a.add_argument("--seed", type=int, default=20261014)
    a.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--scenes-root", required=True)
    s.add_argument("--no-task-objects", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("render")
    r.add_argument("--pilot", required=True)
    r.add_argument("--gs-root", required=True)
    r.add_argument("--gpu", type=int, default=0)
    t = sub.add_parser("topdown")
    t.add_argument("--pilot", required=True)
    t.add_argument("--scenes-root", required=True)
    t.add_argument("--no-task-objects", default=None)
    args = ap.parse_args(argv)
    {"select": cmd_select, "render": cmd_render, "topdown": cmd_topdown, "check": cmd_check,
     "audit": cmd_audit}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
