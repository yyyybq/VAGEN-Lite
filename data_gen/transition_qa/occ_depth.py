"""Depth-checked occlusion labels (2026-10-21): is the target A really hidden by the occluder B in the rendered scene?

The occlusion judge (canonical_task_metrics_v2.occlusion_silhouette) counts a sample point of A's camera-facing faces as
hidden when the ray to it crosses B's labelled box.  A box larger than B (a person, a monitor on its stand, a camera
rig), a box with open space in it (an open wardrobe) or a box in the wrong place hides what the image shows, and
nothing else in the scene (a wall between two rooms) is looked at: the user's check of 30 occlusion questions with such
occluders found 8 wrong (docs/transition_qa/PROGRESS.md section 12.11).  Shrinking the boxes does not separate them
(the same check: every shrink that drops the wrong labels drops about half the right ones).

This renders the depth and opacity at each question's pose with the renderer of data_gen/interiorgs_v10/depth_check.py
and takes the judge's own sample points and weights: a point is hidden by B when the rendered surface at its pixel lies
in front of it (depth_check.depth_hidden: by more than max(0.15 m, 10% of its depth), opacity >= 0.5) and that surface
point lies inside B's box widened by MARGIN_M.  ``report`` compares the depth-checked answer with the bank's.

    python -m data_gen.transition_qa.occ_depth prepare --out BANK --sft-dir SFT --extra EXTRA --dir NAME [--ids a,b]
    python -m data_gen.transition_qa.occ_depth job --pilot BANK/NAME --gs-root PLY [--pool zoetrope] [--submit]
    python -m data_gen.transition_qa.occ_depth report --pilot BANK/NAME [--verdicts V.json ...]
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import generate as GEN
from . import geometry as G

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENV = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"
MARGIN_M = 0.10        # a hiding surface counts as B when it lies within B's box widened by this much
VERSION = "transition_qa_occ_depth_v1"


# ---------------------------------------------------------------------------
# geometry (no GPU)
# ---------------------------------------------------------------------------
def judge_samples(ctx: Any, target: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The occlusion judge's sample points on A's camera-facing faces, their area weights and the in-frame mask
    (the first half of canonical_task_metrics_v2.occlusion_silhouette)."""
    from vagen.envs.active_spatial import canonical_task_metrics_v2 as M2

    lo, hi = M2._bounds(target)
    samples, weights = [], []
    for axis, plane in M2._facing_faces(lo, hi, ctx.position):
        pts, quad = M2._face_samples(lo, hi, axis, plane, M2.OCC_FACE_SAMPLES)
        quad_c = ctx.to_camera(quad)
        if float(quad_c[:, 2].min()) <= M2.MIN_CORNER_DEPTH_M:
            continue
        area = M2._polygon_area(ctx.uv(quad_c))
        if area <= 1e-9:
            continue
        samples.append(pts)
        weights.append(np.full(len(pts), area / len(pts)))
    if not weights:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, bool)
    pts, w = np.concatenate(samples), np.concatenate(weights)
    pc = ctx.to_camera(pts)
    uv = ctx.uv(pc)
    framed = ((pc[:, 2] > M2.MIN_CORNER_DEPTH_M) & (uv[:, 0] >= 0.0) & (uv[:, 0] <= ctx.width - 1)
              & (uv[:, 1] >= 0.0) & (uv[:, 1] <= ctx.height - 1))
    return pts, w, framed


def surface_points(points: np.ndarray, c2w: np.ndarray, K: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """(S, 3) world points of the rendered surface at the pixels of ``points`` (depth along the viewing axis, the
    camera convention of depth_check.depth_hidden: right, down, forward); NaN where a point is off the image."""
    R, C = c2w[:3, :3], c2w[:3, 3]
    cam = (np.asarray(points, float) - C) @ R
    h, w = depth.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
        v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    ok = (cam[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui = np.clip(np.nan_to_num(u), 0, w - 1).astype(int)
    vi = np.clip(np.nan_to_num(v), 0, h - 1).astype(int)
    d = depth[vi, ui].astype(float)
    xs = (np.nan_to_num(u) - K[0, 2]) / K[0, 0] * d  # on the ray through the sample point, at the rendered depth
    ys = (np.nan_to_num(v) - K[1, 2]) / K[1, 1] * d
    world = np.stack([xs, ys, d], axis=1) @ R.T + C
    world[~ok] = np.nan
    return world


def inside_box(points: np.ndarray, obj: Dict[str, Any], margin: float = MARGIN_M) -> np.ndarray:
    lo = np.minimum(np.asarray(obj["bbox_min"], float)[:3], np.asarray(obj["bbox_max"], float)[:3]) - margin
    hi = np.maximum(np.asarray(obj["bbox_min"], float)[:3], np.asarray(obj["bbox_max"], float)[:3]) + margin
    with np.errstate(invalid="ignore"):
        return np.all((points >= lo) & (points <= hi), axis=1)


def fractions(w: np.ndarray, framed: np.ndarray, box_hit: np.ndarray, depth_hit: np.ndarray,
              in_b: np.ndarray) -> Dict[str, float]:
    """Hidden shares of the in-frame silhouette (weights as the judge): by B's box (the judge), by any rendered
    surface, by a rendered surface inside B's box (the depth-checked answer)."""
    total = float(w[framed].sum())
    if total <= 0:
        return {"box": 0.0, "depth_any": 0.0, "depth_b": 0.0}
    return {"box": float(w[framed & box_hit].sum()) / total, "depth_any": float(w[framed & depth_hit].sum()) / total,
            "depth_b": float(w[framed & depth_hit & in_b].sum()) / total}


def answer_of(h: float) -> str:
    """The bank's thresholds (questions.DEFAULT_CONFIG): full >= 0.85, none <= 0.05, partial 0.25-0.70, else between."""
    return "full" if h >= 0.85 else "none" if h <= 0.05 else "partial" if 0.25 <= h <= 0.70 else "between"


# ---------------------------------------------------------------------------
# prepare / job / run / report
# ---------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> None:
    """DIR/frames.jsonl: one frame per occlusion candidate (raw files) at the question's pose (after the action for a
    forward question); DIR/tasks.jsonl: their task rows."""
    out, sft = Path(args.out), Path(args.sft_dir)
    d = out / args.dir
    d.mkdir(parents=True, exist_ok=True)
    rows = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    frames = {f["frame_id"]: f for f in GEN.read_jsonl(sft / "frames.jsonl")}
    for extra in args.extra:
        rows.update({r["task_id"]: r for r in GEN.read_jsonl(Path(extra) / "rows.jsonl")})
        frames.update({f["frame_id"]: f for f in GEN.read_jsonl(Path(extra) / "frames.jsonl")})
    want = set(x for x in args.ids.split(",") if x) if args.ids else None
    recs = [r for p in sorted((out / "raw").glob("*.jsonl")) for r in GEN.read_jsonl(p)
            if r["kind"] == "occluded" and (want is None or r["sample_id"] in want)
            and (want is not None or r["answer_value"] in args.answers.split(","))]
    todo, tasks = [], {}
    for r in recs:
        f = frames[GEN.frame_id_of(r["images"][0])]
        c2w = np.asarray(f["c2w"], float)
        if r["qclass"] == "forward":
            c2w = G.repeated_action_pose(c2w, r["action"], int(r["k"]))
        todo.append({"frame_id": r["sample_id"], "sample_id": r["sample_id"], "task_id": r["task_id"],
                     "scene_id": r["scene"], "qtype": r["qtype"], "answer_value": r["answer_value"],
                     "c2w": c2w.tolist(), "camera_item": f["camera_item"]})
        tasks[r["task_id"]] = rows[r["task_id"]]
    GEN.write_jsonl(d / "frames.jsonl", sorted(todo, key=lambda x: (x["scene_id"], x["frame_id"])))
    GEN.write_jsonl(d / "tasks.jsonl", list(tasks.values()))
    print(json.dumps({"frames": len(todo), "scenes": len({x["scene_id"] for x in todo}), "tasks": len(tasks)}))


LAUNCH = """#!/usr/bin/env bash
set -euo pipefail
PILOT={pilot}
OWNER=$(stat -c %u:%g "$PILOT/frames.jsonl")
trap 'chown -R "$OWNER" "$PILOT" 2>/dev/null || true' EXIT
SRC=$PILOT/job/src
mkdir -p "$SRC" "$PILOT/job/logs"
exec > >(tee -a "$PILOT/job/logs/occ-depth-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "$SRC" && tar -xzf {archive}
RUNTIME_ENV={env}
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1 PATH="$RUNTIME_ENV/bin:$PATH" CUDA_VISIBLE_DEVICES=0
export VAGEN_GSPLAT_PREBUILT=1 TORCH_EXTENSIONS_DIR=/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer
export CC=$RUNTIME_ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$RUNTIME_ENV/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$RUNTIME_ENV TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*'
export CPATH="$RUNTIME_ENV/targets/x86_64-linux/include:${{CPATH:-}}"
export LIBRARY_PATH="$RUNTIME_ENV/targets/x86_64-linux/lib:${{LIBRARY_PATH:-}}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
$RUNTIME_ENV/bin/python -m {module} run --pilot "$PILOT" --gs-root {gs_root} {extra}
"""


def write_job(pilot: Path, gs_root: str, pool: str, module: str = "data_gen.transition_qa.occ_depth",
              extra: str = "") -> Path:
    """PILOT/job: the code frozen into source.tar.gz and a launch.sh in the format of pilot_sco (one card) running
    ``python -m MODULE run --pilot PILOT --gs-root GS_ROOT EXTRA``; submit with data_gen.interiorgs_v10.job_sco."""
    from data_gen.interiorgs_v10 import job_sco

    pilot = Path(pilot).resolve()
    job = pilot / "job"
    job.mkdir(parents=True, exist_ok=True)
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for rel in ("data_gen/__init__.py", "data_gen/interiorgs_v10", "data_gen/transition_qa", "vagen", "ply_gaussian_loader.py"):
            if (ROOT / rel).exists():
                tar.add(ROOT / rel, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name or "figures_cabinet" in ti.name else ti)
    (job / "launch.sh").write_text(LAUNCH.format(pilot=shlex.quote(str(pilot)), archive=shlex.quote(str(archive)),
                                                 env=RUNTIME_ENV, gs_root=shlex.quote(gs_root), extra=extra,
                                                 module=module))
    (job / "launch.sh").chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "mode_args": [module.rsplit(".", 1)[-1].replace("_", "-")],
        "spec": job_sco.SPECS[1], "pool": pool}, indent=1))
    return job / "launch.sh"


def cmd_job(args: argparse.Namespace) -> None:
    """DIR/job (write_job) for this module; --submit submits it."""
    from data_gen.interiorgs_v10 import job_sco

    launch = write_job(Path(args.pilot), args.gs_root, args.pool, extra="--save-rgb" if args.save_rgb else "")
    print(json.dumps({"launch": str(launch)}))
    if args.submit:
        print(job_sco.submit(Path(args.pilot).resolve()))


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - needs a GPU and the 3DGS files
    from PIL import Image

    from data_gen.interiorgs_v10.contract import RENDER_SIZE
    from data_gen.interiorgs_v10.depth_check import depth_hidden
    from data_gen.interiorgs_v10.scene import load_scene
    from data_gen.interiorgs_v10.occlusion import face_occluders_for
    from vagen.envs.active_spatial import canonical_task_metrics_v2 as M2
    from vagen.envs.active_spatial.env import runtime_render_camera_parameters
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    pilot = Path(args.pilot)
    tasks = {t["task_id"]: t for t in GEN.read_jsonl(pilot / "tasks.jsonl")}
    frames = GEN.read_jsonl(pilot / "frames.jsonl")
    done = {r["sample_id"] for r in GEN.read_jsonl(pilot / "results.jsonl")} if (pilot / "results.jsonl").exists() else set()
    if args.save_rgb:
        (pilot / "rgb").mkdir(exist_ok=True)
    meta_root = Path(args.meta_root)
    unified, scene_id, scene, occ = None, None, None, None
    t0 = time.time()
    with open(pilot / "results.jsonl", "a") as out:
        for f in frames:
            if f["sample_id"] in done:
                continue
            try:
                if f["scene_id"] != scene_id:
                    scene_id = f["scene_id"]
                    if unified is None:
                        unified = UnifiedRenderGS(render_backend="local", gs_root=args.gs_root, scene_id=scene_id, gpu_device=0)
                    unified.set_scene(scene_id)
                    scene = load_scene(meta_root / scene_id, scene_id, wall_model="thick")
                    occ = face_occluders_for(scene)
                renderer = unified._ensure_local()
                item = tasks[f["task_id"]]
                geo = G.TaskGeometry(item, scene, occ)
                a, b = geo.roles["a"], geo.roles["b"]
                c2w = np.asarray(f["c2w"], float)
                K_native = np.asarray(f["camera_item"]["init_camera"]["intrinsics"], float)
                K, w2c = runtime_render_camera_parameters(f["camera_item"], c2w, K_native, RENDER_SIZE)
                rgb, depth, alpha = renderer.render_depth_from_cam_param(K, w2c, *RENDER_SIZE)
                ctx = M2._Context(item, c2w)
                pts, w, framed = judge_samples(ctx, a)
                olo, ohi = M2._bounds(b)
                box_hit = M2.segments_hit_aabb(ctx.position, pts, olo, ohi) if len(pts) else np.zeros(0, bool)
                depth_hit = depth_hidden(pts, c2w, K, depth, alpha) if len(pts) else np.zeros(0, bool)
                in_b = inside_box(surface_points(pts, c2w, K, depth), b) if len(pts) else np.zeros(0, bool)
                fr = fractions(w, framed, box_hit, depth_hit, in_b)
                judge = M2.occlusion_silhouette(ctx, a, b)["hidden_fraction"]
                row = {"sample_id": f["sample_id"], "scene_id": f["scene_id"], "qtype": f["qtype"],
                       "answer_value": f["answer_value"], "judge": round(judge, 4),
                       **{k: round(v, 4) for k, v in fr.items()}, "samples": int(len(pts)),
                       "b_label": b.get("label"), "version": VERSION}
                if args.save_rgb:
                    Image.fromarray(rgb).save(pilot / "rgb" / f"{f['sample_id']}.png")
            except Exception as exc:  # keep going; the report counts errors
                row = {"sample_id": f["sample_id"], "scene_id": f["scene_id"], "error": f"{type(exc).__name__}: {exc}"}
            out.write(json.dumps(row) + "\n")
            out.flush()
    (pilot / "run_done.json").write_text(json.dumps({"version": VERSION, "seconds": round(time.time() - t0, 1),
                                                     "frames": len(frames)}))


def cmd_report(args: argparse.Namespace) -> None:
    pilot = Path(args.pilot)
    res = [r for r in GEN.read_jsonl(pilot / "results.jsonl")]
    errors = [r for r in res if "error" in r]
    ok = [r for r in res if "error" not in r]
    verdict: Dict[str, Tuple[str, str]] = {}
    for p in args.verdicts:
        data = json.loads(Path(p).read_text())
        for v in data["verdicts"]:
            verdict[v["sample_id"]] = (v.get("verdict") or "", v.get("note") or "")
    agree = sum(answer_of(r["depth_b"]) == r["answer_value"] for r in ok)
    print(json.dumps({"frames": len(res), "errors": len(errors), "depth agrees with the bank": agree,
                      "judge reproduced": sum(abs(r["judge"] - r["box"]) < 1e-6 for r in ok)}))
    for r in sorted(ok, key=lambda r: (verdict.get(r["sample_id"], ("", ""))[0], r["sample_id"])):
        v, note = verdict.get(r["sample_id"], ("", ""))
        print(f"{r['sample_id']} {v:6s} {r['b_label'][:20]:20s} bank {r['answer_value']:7s} box {r['box']:.2f} "
              f"depth-any {r['depth_any']:.2f} depth-B {r['depth_b']:.2f} -> {answer_of(r['depth_b']):8s} {note[:30]}")
    for r in errors[:10]:
        print("error", r["sample_id"], r["error"][:200])


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True, help="the bank (its raw/ files)")
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--dir", required=True)
    p.add_argument("--ids", default="", help="only these sample ids (comma-separated)")
    p.add_argument("--answers", default="full,partial", help="without --ids: the answers to check")
    p.set_defaults(func=cmd_prepare)
    j = sub.add_parser("job")
    j.add_argument("--pilot", required=True)
    j.add_argument("--gs-root", required=True)
    j.add_argument("--pool", default="zoetrope")
    j.add_argument("--save-rgb", action="store_true")
    j.add_argument("--submit", action="store_true")
    j.set_defaults(func=cmd_job)
    r = sub.add_parser("run")
    r.add_argument("--pilot", required=True)
    r.add_argument("--gs-root", required=True)
    r.add_argument("--meta-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_meta")
    r.add_argument("--save-rgb", action="store_true")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("--pilot", required=True)
    s.add_argument("--verdicts", action="append", default=[])
    s.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
