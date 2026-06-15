"""
SFT Data Generator.

Orchestrates the full pipeline:
  1. Reads items from the pipeline JSONL (produced by active_spatial_pipeline).
  2. For each item, finds the optimal camera trajectory with path_finder.
  3. Renders images at every step via the Gaussian Splatting renderer.
  4. Formats each trajectory as a conversation with sft_formatter.
  5. Writes the conversation records to an output JSONL file.

Usage
-----
    from data_gen.active_spatial_sft import SFTDataGenerator, SFTGenerationConfig

    cfg = SFTGenerationConfig(
        jsonl_path="/path/to/pipeline_output.jsonl",
        gs_root="/path/to/gaussian_scenes",
        output_dir="/path/to/sft_output",
        render_backend="local",
        gpu_device=0,
    )
    gen = SFTDataGenerator(cfg)
    stats = gen.run()
    print(stats)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import SFTGenerationConfig
from .path_finder import find_trajectory, find_trajectory_guided, simulate_action, Trajectory, score_c2w
from .sft_formatter import format_trajectory

# ── Ensure VAGEN is importable ───────────────────────────────────────────────
_VAGEN_ROOT = str(Path(__file__).parent.parent.parent)
if _VAGEN_ROOT not in sys.path:
    sys.path.insert(0, _VAGEN_ROOT)


def _run_async(coro):
    """Run an async coroutine in a sync context (handles nested loops)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Renderer wrapper
# ---------------------------------------------------------------------------

class _Renderer:
    """Thin wrapper around UnifiedRenderGS that supports scene switching."""

    def __init__(self, cfg: SFTGenerationConfig):
        self.cfg = cfg
        self._renderer = None
        self._current_scene: Optional[str] = None

    def _init(self):
        if self._renderer is not None:
            return
        if self.cfg.render_backend in (None, "none"):
            return

        from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
        self._renderer = UnifiedRenderGS(
            render_backend=self.cfg.render_backend,
            gs_root=self.cfg.gs_root or None,
            client_url=self.cfg.client_url,
            client_origin=self.cfg.client_origin,
            scene_id=None,
            gpu_device=self.cfg.gpu_device,
        )

    def set_scene(self, scene_id: str):
        self._init()
        if self._renderer is None:
            return
        if scene_id != self._current_scene:
            self._renderer.set_scene(scene_id)
            self._current_scene = scene_id

    def render(
        self,
        K: np.ndarray,
        c2w: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[Image.Image]:
        """Render the scene at the given camera pose."""
        if self._renderer is None:
            return None
        try:
            w2c = np.linalg.inv(c2w)
            K3 = K[:3, :3] if K.shape == (4, 4) else K
            img = _run_async(
                self._renderer.render_image_from_cam_param(
                    camera_intrinsics=K3,
                    camera_extrinsics=w2c,
                    width=width,
                    height=height,
                )
            )
            return img
        except Exception as e:
            print(f"[Renderer] Render error: {e}")
            return None


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def _save_image(
    img: Optional[Image.Image],
    path: Path,
    fmt: str = "jpg",
    quality: int = 85,
) -> bool:
    """Save a PIL image to disk. Returns True on success."""
    if img is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt.lower() == "jpg":
            img.convert("RGB").save(str(path), "JPEG", quality=quality)
        else:
            img.save(str(path), fmt.upper())
        return True
    except Exception as e:
        print(f"[ImageSave] Failed to save {path}: {e}")
        return False


def _image_to_base64(img: Optional[Image.Image], fmt: str = "jpg", quality: int = 85) -> str:
    """Encode a PIL image as a base64 string."""
    import base64
    import io

    if img is None:
        return ""
    buf = io.BytesIO()
    if fmt.lower() == "jpg":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, fmt.upper())
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Collision detector helper
# ---------------------------------------------------------------------------

def _make_collision_detector(cfg: SFTGenerationConfig):
    """Create a CollisionDetector if collision detection is enabled."""
    if not cfg.enable_collision_detection:
        return None
    try:
        from vagen.envs.active_spatial.collision_detector import create_collision_detector
        return create_collision_detector({
            "camera_radius": cfg.collision_camera_radius,
            "floor_height": cfg.collision_floor_height,
            "ceiling_height": cfg.collision_ceiling_height,
            "safety_margin": cfg.collision_safety_margin,
            "enable_object_collision": True,
            "enable_boundary_collision": True,
        })
    except Exception as e:
        print(f"[CollisionDetector] Could not create: {e}. Disabling collision detection.")
        return None


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class SFTDataGenerator:
    """
    Generates SFT training data from pipeline JSONL items.

    For each item:
      1. Find optimal trajectory (greedy path finding on potential field).
      2. Render images at each step.
      3. Format as multi-turn conversation.
      4. Write to output JSONL.
    """

    def __init__(self, cfg: SFTGenerationConfig):
        self.cfg = cfg
        self.renderer = _Renderer(cfg)
        self.collision_detector = _make_collision_detector(cfg)
        self._loaded_scene: Optional[str] = None

    # ── Scene data loading ───────────────────────────────────────────────────

    def _load_scene_for_collision(self, scene_id: str):
        """Load scene data into the collision detector (only when scene changes)."""
        if self.collision_detector is None:
            return
        if self._loaded_scene == scene_id:
            return
        if not self.cfg.gs_root:
            return
        try:
            self.collision_detector.load_scene_from_gs_root(self.cfg.gs_root, scene_id)
            self._loaded_scene = scene_id
        except Exception as e:
            print(f"[CollisionDetector] Failed to load scene '{scene_id}': {e}")

    # ── Per-item processing ──────────────────────────────────────────────────

    def _process_goal_reached(
        self,
        item: Dict[str, Any],
        item_idx: int,
        sft_id: str,
        image_dir: Path,
        initial_score: float,
    ) -> Optional[Dict[str, Any]]:
        """Emit a one-turn 'already at goal → done' example.

        This handles the 'stopping-boundary' problem: the model must learn to
        issue 'done' when the current view ALREADY satisfies the task, rather
        than learning to count steps.  Without these examples, 'done' is always
        the last element of a long successful sequence and the model may learn
        to associate it with sequence position rather than visual content.
        """
        cfg = self.cfg
        scene_id = item.get("scene_id", "")
        self.renderer.set_scene(scene_id)

        init_cam = item.get("init_camera", {})
        K = np.array(init_cam.get("intrinsics", _fallback_K()), dtype=np.float64)
        E = np.array(init_cam.get("extrinsics", np.eye(4)), dtype=np.float64)

        rel_path = f"images/{sft_id}_step00.{cfg.image_format}"
        abs_path = Path(cfg.output_dir) / rel_path
        img = self.renderer.render(K, E, cfg.image_width, cfg.image_height)
        if img is not None and cfg.save_images:
            _save_image(img, abs_path, cfg.image_format, cfg.image_quality)
        image_path = rel_path if cfg.save_images else _image_to_base64(img)

        traj = Trajectory(
            steps=[],
            final_c2w=E,
            initial_score=initial_score,
            final_score=initial_score,
            success=True,
            total_actions=0,
            scene_id=scene_id,
            item_idx=item_idx,
        )
        record = format_trajectory(
            item=item,
            trajectory=traj,
            image_paths=[image_path],
            sft_id=sft_id,
            prompt_format=cfg.prompt_format,
            add_think=cfg.add_think,
            include_scores=cfg.include_score_in_think,
        )
        if cfg.verbose:
            print(
                f"[Generator] Item {item_idx} → {sft_id} (goal-reached-at-start): "
                f"initial_score={initial_score:.3f}"
            )
        return record

    # ── Per-item processing ──────────────────────────────────────────────────

    def _process_item(
        self,
        item: Dict[str, Any],
        item_idx: int,
        sft_id: str,
        image_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Process a single pipeline item → SFT record (or None on failure)."""
        cfg = self.cfg
        scene_id = item.get("scene_id", "")

        # ── Load scene ───────────────────────────────────────────────────────
        self.renderer.set_scene(scene_id)
        self._load_scene_for_collision(scene_id)

        # ── Extract task data ────────────────────────────────────────────────
        init_cam = item.get("init_camera", {})
        K = np.array(init_cam.get("intrinsics", _fallback_K()), dtype=np.float64)
        E = np.array(init_cam.get("extrinsics", np.eye(4)), dtype=np.float64)

        task_type = item.get("task_type", "absolute_positioning")
        task_params = item.get("task_params", {})
        target_region = item.get("target_region", {})

        if not target_region:
            if cfg.verbose:
                print(f"[Generator] Item {item_idx}: no target_region, skipping.")
            return None

        # ── Find trajectory ──────────────────────────────────────────────────
        t0 = time.time()
        _finder = find_trajectory_guided if cfg.use_guided_search else find_trajectory
        trajectory = _finder(
            init_c2w=E,
            task_type=task_type,
            task_params=task_params,
            target_region=target_region,
            position_weight=cfg.position_weight,
            orientation_weight=cfg.orientation_weight,
            max_distance=cfg.max_distance,
            step_translation=cfg.step_translation,
            step_rotation_deg=cfg.step_rotation_deg,
            success_threshold=cfg.success_threshold,
            max_total_actions=cfg.max_total_actions,
            max_actions_per_turn=cfg.max_actions_per_turn,
            min_improvement=cfg.min_improvement,
            plateau_tolerance=cfg.plateau_tolerance,
            beam_width=cfg.beam_width,
            adaptive_min_improvement=cfg.adaptive_min_improvement,
            collision_checker=self.collision_detector,
            scene_id=scene_id,
            item_idx=item_idx,
            verbose=cfg.verbose,
        )
        t_find = time.time() - t0

        # ── Validate trajectory ──────────────────────────────────────────────
        n_steps = len(trajectory.steps)

        if cfg.only_successful and not trajectory.success:
            if cfg.verbose:
                print(f"[Generator] Item {item_idx}: trajectory not successful "
                      f"(score={trajectory.final_score:.4f}), skipping.")
            return None

        if not trajectory.success and trajectory.final_score < cfg.partial_success_min_score:
            if cfg.verbose:
                print(f"[Generator] Item {item_idx}: score {trajectory.final_score:.4f} below "
                      f"partial_success_min_score={cfg.partial_success_min_score:.2f}, skipping.")
            return None

        if n_steps < cfg.min_trajectory_steps:
            if cfg.verbose:
                print(f"[Generator] Item {item_idx}: trajectory too short ({n_steps} steps), skipping.")
            return None

        if n_steps > cfg.max_trajectory_steps:
            if cfg.verbose:
                print(f"[Generator] Item {item_idx}: trajectory too long ({n_steps} steps), truncating.")
            trajectory.steps = trajectory.steps[:cfg.max_trajectory_steps]

        # ── Render images ────────────────────────────────────────────────────
        # We need one image per step PLUS the initial image.
        # image[0] = initial view (before any action)
        # image[k] = view after step k-1's actions (k = 1 … len(steps))
        poses_to_render: List[np.ndarray] = [E]  # initial
        for step in trajectory.steps:
            poses_to_render.append(step.c2w_after)

        image_paths: List[str] = []
        rendered_ok = True

        t1 = time.time()
        for img_idx, pose in enumerate(poses_to_render):
            rel_path = f"images/{sft_id}_step{img_idx:02d}.{cfg.image_format}"
            abs_path = Path(cfg.output_dir) / rel_path

            img = self.renderer.render(K, pose, cfg.image_width, cfg.image_height)

            if img is not None and cfg.save_images:
                ok = _save_image(img, abs_path, cfg.image_format, cfg.image_quality)
                if not ok:
                    rendered_ok = False
            elif img is None:
                rendered_ok = False

            if cfg.save_images:
                image_paths.append(rel_path)
            else:
                # Embed as base64
                image_paths.append(_image_to_base64(img, cfg.image_format, cfg.image_quality))

        t_render = time.time() - t1

        if not rendered_ok and cfg.render_backend not in (None, "none"):
            if cfg.verbose:
                print(f"[Generator] Item {item_idx}: some renders failed.")
            # We continue – the image_paths list may contain partial results

        # ── Format conversation ──────────────────────────────────────────────
        record = format_trajectory(
            item=item,
            trajectory=trajectory,
            image_paths=image_paths,
            sft_id=sft_id,
            prompt_format=cfg.prompt_format,
            add_think=cfg.add_think,
            include_scores=cfg.include_score_in_think,
            force_no_done=not trajectory.success,  # partial trajectories end without 'done'
        )

        if cfg.verbose:
            print(
                f"[Generator] Item {item_idx} → {sft_id}: "
                f"steps={n_steps}, actions={trajectory.total_actions}, "
                f"score={trajectory.initial_score:.3f}→{trajectory.final_score:.3f}, "
                f"success={trajectory.success}, "
                f"t_find={t_find:.2f}s, t_render={t_render:.2f}s"
            )

        return record

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run the full SFT data generation pipeline.

        Returns a statistics dict with counts of processed/skipped/failed items.
        """
        cfg = self.cfg

        if not cfg.jsonl_path:
            raise ValueError("SFTGenerationConfig.jsonl_path must be set.")
        if not cfg.output_dir:
            raise ValueError("SFTGenerationConfig.output_dir must be set.")

        jsonl_path = Path(cfg.jsonl_path)
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)

        output_jsonl = output_dir / f"{cfg.output_name}.jsonl"

        # ── Load source items ────────────────────────────────────────────────
        items = _load_jsonl(jsonl_path)
        total_available = len(items)

        start = cfg.start_idx
        end = cfg.end_idx if cfg.end_idx >= 0 else total_available
        items = items[start:end]

        if cfg.max_items > 0:
            items = items[:cfg.max_items]

        print(f"[Generator] Source: {jsonl_path} ({total_available} items)")
        print(f"[Generator] Processing items [{start}, {start + len(items)}) → {output_jsonl}")

        # ── Statistics ───────────────────────────────────────────────────────
        stats = {
            "total_input": len(items),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "successful_trajectories": 0,
            "goal_reached_examples": 0,
            "total_steps": 0,
            "total_actions": 0,
        }

        # ── Process ──────────────────────────────────────────────────────────
        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for local_idx, item in enumerate(items):
                global_idx = start + local_idx
                sft_id = f"sft_{global_idx:06d}"

                # ── Optional: goal-reached-at-start example ──────────────────
                if cfg.include_goal_reached_examples:
                    try:
                        from vagen.envs.active_spatial.spatial_potential_field import create_potential_field
                        pf = create_potential_field({
                            "position_weight": cfg.position_weight,
                            "orientation_weight": cfg.orientation_weight,
                            "max_distance": cfg.max_distance,
                        })
                        init_cam = item.get("init_camera", {})
                        E = np.array(init_cam.get("extrinsics", np.eye(4)), dtype=np.float64)
                        task_type = item.get("task_type", "absolute_positioning")
                        task_params = item.get("task_params", {})
                        target_region = item.get("target_region", {})
                        if target_region:
                            init_score, _, _ = score_c2w(E, pf, task_type, task_params, target_region)
                            if init_score >= cfg.success_threshold:
                                gr_id = f"sft_{global_idx:06d}_gr"
                                gr_record = self._process_goal_reached(
                                    item, global_idx, gr_id,
                                    output_dir / "images", init_score,
                                )
                                if gr_record is not None:
                                    fout.write(json.dumps(gr_record, ensure_ascii=False) + "\n")
                                    stats["processed"] += 1
                                    stats["goal_reached_examples"] += 1
                    except Exception as e:
                        if cfg.verbose:
                            print(f"[Generator] Item {global_idx}: goal-reached check failed: {e}")

                try:
                    record = self._process_item(
                        item=item,
                        item_idx=global_idx,
                        sft_id=sft_id,
                        image_dir=output_dir / "images",
                    )
                except Exception as e:
                    if cfg.skip_failed:
                        print(f"[Generator] Item {global_idx}: error – {e} (skipping)")
                        stats["failed"] += 1
                        continue
                    else:
                        raise

                if record is None:
                    stats["skipped"] += 1
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["processed"] += 1
                if record["success"]:
                    stats["successful_trajectories"] += 1
                stats["total_steps"] += record["trajectory_steps"]
                stats["total_actions"] += record["total_actions"]

                if (local_idx + 1) % 100 == 0:
                    pct = (local_idx + 1) / len(items) * 100
                    print(f"[Generator] Progress: {local_idx + 1}/{len(items)} ({pct:.1f}%) "
                          f"processed={stats['processed']} skipped={stats['skipped']} "
                          f"failed={stats['failed']}")

        # ── Summary ──────────────────────────────────────────────────────────
        print("\n[Generator] ── Summary ──────────────────────────────────────")
        print(f"  Input items:           {stats['total_input']}")
        print(f"  Processed (saved):     {stats['processed']}")
        print(f"  Skipped (no path/…):   {stats['skipped']}")
        print(f"  Failed (errors):       {stats['failed']}")
        print(f"  Successful trajectories: {stats['successful_trajectories']}")
        if stats["processed"] > 0:
            avg_steps = stats["total_steps"] / stats["processed"]
            avg_actions = stats["total_actions"] / stats["processed"]
            print(f"  Avg steps/trajectory:  {avg_steps:.1f}")
            print(f"  Avg actions/trajectory:{avg_actions:.1f}")
        print(f"  Output: {output_jsonl}")
        print("[Generator] ─────────────────────────────────────────────────\n")

        return stats


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _fallback_K() -> np.ndarray:
    """Minimal camera intrinsics (used when item has none)."""
    focal = 300.0
    cx, cy = 256.0, 256.0
    return np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1],
    ], dtype=np.float64)
