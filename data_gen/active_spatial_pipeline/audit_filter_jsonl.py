#!/usr/bin/env python3
"""Audit and filter Active Spatial JSONL with layout and optional render sanity."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .layout_quality import LayoutGeometry, extract_init_xy, extract_target_xy, image_std
    from .render_utils import look_at_matrix
except ImportError:
    from layout_quality import LayoutGeometry, extract_init_xy, extract_target_xy, image_std
    from render_utils import look_at_matrix


def camera_forward_xy(item: Dict[str, Any]) -> Optional[np.ndarray]:
    extrinsics = item.get("init_camera", {}).get("extrinsics")
    if extrinsics is None:
        return None
    mat = np.asarray(extrinsics, dtype=np.float64)
    if mat.shape[0] < 3 or mat.shape[1] < 3:
        return None
    fwd = mat[:2, 2]
    norm = float(np.linalg.norm(fwd))
    return fwd / norm if norm > 1e-8 else None


def one_step_collision(item: Dict[str, Any], layout: LayoutGeometry, min_clearance: float, step: float) -> Optional[bool]:
    init_xy = extract_init_xy(item)
    fwd = camera_forward_xy(item)
    if init_xy is None or fwd is None:
        return None
    next_xy = init_xy + step * fwd
    return not layout.is_safe_xy(next_xy, min_clearance)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def target_c2w(item: Dict[str, Any]) -> Optional[np.ndarray]:
    target = item.get("sample_target")
    forward = item.get("camera_params", {}).get("forward")
    if target is None or forward is None:
        return None
    pos = np.asarray(target, dtype=np.float64)[:3]
    fwd = np.asarray(forward, dtype=np.float64)[:3]
    norm = float(np.linalg.norm(fwd))
    if norm <= 1e-8:
        return None
    look_at = pos + fwd / norm
    return look_at_matrix(pos, look_at)


class OptionalRenderer:
    def __init__(self, backend: Optional[str], gs_root: str, client_url: str, width: int, height: int):
        self.backend = backend
        self.width = width
        self.height = height
        self.renderer = None
        if backend:
            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            try:
                from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"render sanity requested but renderer dependency is missing: {exc.name}"
                ) from exc
            self.renderer = UnifiedRenderGS(render_backend=backend, gs_root=gs_root, client_url=client_url, scene_id=None)

    async def render_std(
        self,
        scene_id: str,
        intrinsics: Any,
        c2w: np.ndarray,
        retries: int = 2,
        call_timeout: float = 45.0,
    ) -> Tuple[Optional[float], Optional[str], Optional[Any]]:
        if self.renderer is None:
            return None, None, None
        k = np.asarray(intrinsics, dtype=np.float64)
        if k.shape == (4, 4):
            k = k[:3, :3]
        last_error = None
        for attempt in range(max(1, retries + 1)):
            try:
                async def _render_once():
                    self.renderer.set_scene(scene_id)
                    return await self.renderer.render_image_from_cam_param(
                        camera_intrinsics=k,
                        camera_extrinsics=np.linalg.inv(c2w),
                        width=self.width,
                        height=self.height,
                    )

                image = await asyncio.wait_for(_render_once(), timeout=call_timeout)
                if image is None:
                    last_error = "empty_image"
                    continue
                std = image_std(image)
                if std is None:
                    last_error = "image_decode_failed"
                    continue
                return std, None, image
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{str(exc)[:220]}"
                if attempt < retries:
                    await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
        return None, last_error or "unknown_render_error", None


def write_overlay(path: Path, layout: LayoutGeometry, init_xy, target_xy, kept: bool) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    for poly in layout.room_polys:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, color="black", linewidth=1)
    if init_xy is not None:
        ax.scatter([init_xy[0]], [init_xy[1]], color="tab:blue", label="init")
    if target_xy is not None:
        ax.scatter([target_xy[0]], [target_xy[1]], color="tab:green" if kept else "tab:red", label="target")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--gs-root", required=True)
    parser.add_argument("--min-wall-clearance", type=float, default=0.5)
    parser.add_argument("--step-translation", type=float, default=0.3)
    parser.add_argument("--filter-collision-after-one-step", action="store_true")
    parser.add_argument("--render-backend", default="")
    parser.add_argument("--client-url", default="")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--min-render-std", type=float, default=8.0)
    parser.add_argument("--require-render-std", action="store_true")
    parser.add_argument("--render-retries", type=int, default=2)
    parser.add_argument("--render-call-timeout", type=float, default=45.0)
    parser.add_argument("--overlay-dir", default="")
    parser.add_argument("--render-sample-dir", default="")
    parser.add_argument("--save-render-samples", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    renderer = OptionalRenderer(args.render_backend or None, args.gs_root, args.client_url, args.image_width, args.image_height)
    out_path = Path(args.output)
    audit_path = Path(args.audit_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    render_sample_dir = Path(args.render_sample_dir) if args.render_sample_dir else None
    if render_sample_dir:
        render_sample_dir.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    total = 0
    scenes: Dict[str, LayoutGeometry] = {}
    counters = Counter()
    drop_reasons = Counter()
    task_input = Counter()
    task_kept = Counter()
    task_drop_reasons: Dict[str, Counter] = defaultdict(Counter)
    init_stds: List[float] = []
    target_stds: List[float] = []
    output_sha = hashlib.sha256()
    with Path(args.input).open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as out, audit_path.open("w", encoding="utf-8") as audit:
        for idx, line in enumerate(src):
            if args.max_items and idx >= args.max_items:
                break
            item = json.loads(line)
            total += 1
            counters["total"] += 1
            scene_id = item.get("scene_id", "")
            task_type = item.get("task_type", "unknown")
            task_input[task_type] += 1
            if scene_id not in scenes:
                scenes[scene_id] = LayoutGeometry.from_scene_path(Path(args.gs_root) / scene_id)
            layout = scenes[scene_id]
            init_xy = extract_init_xy(item)
            target_xy = extract_target_xy(item)
            init_inside = bool(init_xy is not None and layout.contains_xy(init_xy))
            target_inside = bool(target_xy is not None and layout.contains_xy(target_xy))
            init_wall = float(layout.wall_distance(init_xy)) if init_xy is not None else None
            target_wall = float(layout.wall_distance(target_xy)) if target_xy is not None else None
            collision = one_step_collision(item, layout, args.min_wall_clearance, args.step_translation)

            init_std = None
            target_std = None
            init_render_error = None
            target_render_error = None
            if renderer.renderer is not None:
                intrinsics = item.get("init_camera", {}).get("intrinsics")
                init_c2w = item.get("init_camera", {}).get("extrinsics")
                init_img = None
                target_img = None
                if intrinsics is None or init_c2w is None:
                    init_render_error = "missing_init_camera"
                    target_render_error = "missing_init_camera"
                else:
                    init_std, init_render_error, init_img = await renderer.render_std(
                        scene_id, intrinsics, np.asarray(init_c2w, dtype=np.float64),
                        args.render_retries, args.render_call_timeout
                    )
                    tgt_c2w = target_c2w(item)
                    if tgt_c2w is None:
                        target_render_error = "missing_target_pose"
                    else:
                        target_std, target_render_error, target_img = await renderer.render_std(
                            scene_id, intrinsics, tgt_c2w,
                            args.render_retries, args.render_call_timeout
                        )
                if render_sample_dir and idx < args.save_render_samples:
                    if init_img is not None:
                        init_img.save(render_sample_dir / f"{idx:06d}_init.png")
                    if target_img is not None:
                        target_img.save(render_sample_dir / f"{idx:06d}_target.png")

            reasons = []
            if not init_inside:
                reasons.append("init_outside")
            if not target_inside:
                reasons.append("target_outside")
            if init_wall is None or init_wall < args.min_wall_clearance:
                reasons.append("init_wall_too_close")
            if target_wall is None or target_wall < args.min_wall_clearance:
                reasons.append("target_wall_too_close")
            if collision is True and args.filter_collision_after_one_step:
                reasons.append("collision_after_one_step")
            if args.require_render_std:
                if init_render_error:
                    reasons.append("init_render_failed")
                elif init_std is None or init_std < args.min_render_std:
                    reasons.append("init_render_std_low")
                if target_render_error:
                    reasons.append("target_render_failed")
                elif target_std is None or target_std < args.min_render_std:
                    reasons.append("target_render_std_low")

            if init_std is not None:
                init_stds.append(float(init_std))
                counters["init_render_success"] += 1
                if init_std < args.min_render_std:
                    counters["init_low_info"] += 1
            if target_std is not None:
                target_stds.append(float(target_std))
                counters["target_render_success"] += 1
                if target_std < args.min_render_std:
                    counters["target_low_info"] += 1
            if init_render_error:
                counters["init_render_failed"] += 1
            if target_render_error:
                counters["target_render_failed"] += 1
            if collision is True:
                counters["collision_after_one_step"] += 1
            if not init_inside:
                counters["init_outside"] += 1
            if not target_inside:
                counters["target_outside"] += 1
            if init_wall is None or init_wall < args.min_wall_clearance:
                counters["init_wall_too_close"] += 1
            if target_wall is None or target_wall < args.min_wall_clearance:
                counters["target_wall_too_close"] += 1

            kept = not reasons
            if not kept:
                for reason in reasons:
                    drop_reasons[reason] += 1
                    task_drop_reasons[task_type][reason] += 1
            overlay_path = None
            if args.overlay_dir:
                overlay_path = write_overlay(
                    Path(args.overlay_dir) / f"{idx:06d}_{'keep' if kept else 'drop'}.png",
                    layout,
                    init_xy,
                    target_xy,
                    kept,
                )

            audit_rec = {
                "idx": idx,
                "scene_id": scene_id,
                "task_type": item.get("task_type"),
                "object_label": item.get("object_label"),
                "preset": item.get("preset"),
                "kept": kept,
                "drop_reasons": reasons,
                "init_inside": init_inside,
                "init_wall_distance": init_wall,
                "target_inside": target_inside,
                "target_wall_distance": target_wall,
                "render_image_std_init": init_std,
                "render_image_std_target": target_std,
                "render_error_init": init_render_error,
                "render_error_target": target_render_error,
                "collision_after_one_step": collision,
                "layout_overlay": overlay_path,
            }
            audit.write(json.dumps(audit_rec, ensure_ascii=False) + "\n")
            audit.flush()
            if kept:
                out_line = json.dumps(item, ensure_ascii=False) + "\n"
                out.write(out_line)
                out.flush()
                output_sha.update(out_line.encode("utf-8"))
                kept_count += 1
                counters["kept"] += 1
                task_kept[task_type] += 1
            else:
                counters["dropped"] += 1
            if total % 25 == 0:
                print(f"[audit] processed={total} kept={kept_count}", flush=True)

    def dist(values: List[float]) -> Dict[str, Optional[float]]:
        return {
            "count": len(values),
            "min": percentile(values, 0),
            "p1": percentile(values, 1),
            "p5": percentile(values, 5),
            "median": percentile(values, 50),
            "p95": percentile(values, 95),
            "max": percentile(values, 100),
        }

    summary = {
        "total": total,
        "kept": kept_count,
        "dropped": total - kept_count,
        "sha256_output": output_sha.hexdigest() if kept_count else None,
        "counters": dict(counters),
        "drop_reasons": dict(drop_reasons),
        "task_input": dict(task_input),
        "task_kept": dict(task_kept),
        "task_drop_reasons": {k: dict(v) for k, v in task_drop_reasons.items()},
        "init_image_std": dist(init_stds),
        "target_image_std": dist(target_stds),
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
