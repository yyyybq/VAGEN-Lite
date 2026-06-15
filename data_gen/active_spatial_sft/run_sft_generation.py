#!/usr/bin/env python3
"""
CLI entry point for active_spatial SFT data generation.

Usage
-----
python run_sft_generation.py \\
    --jsonl_path /path/to/pipeline_output.jsonl \\
    --gs_root    /path/to/gaussian_scenes \\
    --output_dir /path/to/sft_output \\
    --render_backend local \\
    --gpu_device 0 \\
    --max_items 500 \\
    --verbose

Run `python run_sft_generation.py --help` for full option list.
"""

import argparse
import json
import sys
from pathlib import Path

# ── Make VAGEN and data_gen importable ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent        # data_gen/active_spatial_sft/
_DATA_GEN = _HERE.parent                       # data_gen/
_VAGEN_ROOT = _DATA_GEN.parent                 # VAGEN/

for p in [str(_VAGEN_ROOT), str(_DATA_GEN)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from active_spatial_sft.config import SFTGenerationConfig
from active_spatial_sft.sft_generator import SFTDataGenerator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SFT training data from active_spatial_pipeline JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ─────────────────────────────────────────────────────────────────
    p.add_argument("--jsonl_path", required=True,
                   help="Path to input pipeline JSONL file.")
    p.add_argument("--gs_root", default="",
                   help="Root directory of Gaussian Splatting scenes (needed for rendering).")
    p.add_argument("--output_dir", required=True,
                   help="Directory where output JSONL and images will be saved.")
    p.add_argument("--output_name", default="sft_data",
                   help="Base name for the output JSONL file (without .jsonl extension).")

    # ── Rendering ────────────────────────────────────────────────────────────
    p.add_argument("--render_backend", default="local",
                   choices=["local", "client", "none"],
                   help="Rendering backend.")
    p.add_argument("--client_url", default="ws://127.0.0.1:8777/render/interiorgs",
                   help="WebSocket URL for client rendering backend.")
    p.add_argument("--gpu_device", type=int, default=None,
                   help="GPU device ID for local rendering (default: auto).")
    p.add_argument("--image_width", type=int, default=512)
    p.add_argument("--image_height", type=int, default=512)
    p.add_argument("--image_format", default="jpg", choices=["jpg", "png"])
    p.add_argument("--image_quality", type=int, default=85,
                   help="JPEG quality (1–95).")
    p.add_argument("--no_save_images", action="store_true",
                   help="Embed images as base64 instead of saving to disk.")

    # ── Camera ───────────────────────────────────────────────────────────────
    p.add_argument("--step_translation", type=float, default=0.3,
                   help="Camera translation step in metres (must match RL env).")
    p.add_argument("--step_rotation_deg", type=float, default=30.0,
                   help="Camera rotation step in degrees (must match RL env).")

    # ── Path finding ─────────────────────────────────────────────────────────
    p.add_argument("--success_threshold", type=float, default=0.95)
    p.add_argument("--max_total_actions", type=int, default=100)
    p.add_argument("--max_actions_per_turn", type=int, default=5)
    p.add_argument("--min_improvement", type=float, default=0.005)
    p.add_argument("--plateau_tolerance", type=int, default=5)

    # ── Search strategy ───────────────────────────────────────────────────────
    p.add_argument("--beam_width", type=int, default=3,
                   help="Beam width within each LLM turn. 1=greedy (fast), 3+=beam search "
                        "(tolerates short-term score drops, higher success rate).")
    p.add_argument("--no_adaptive_min_improvement", action="store_true",
                   help="Disable adaptive min_improvement scaling near success_threshold.")
    p.add_argument("--no_guided_search", action="store_true",
                   help="Disable geometry-guided planning (Phase 1/2). "
                        "Use purely score-based beam/greedy search instead.")

    # ── Potential field ───────────────────────────────────────────────────────
    p.add_argument("--position_weight", type=float, default=0.7)
    p.add_argument("--orientation_weight", type=float, default=0.3)

    # ── Collision detection ───────────────────────────────────────────────────
    p.add_argument("--no_collision_detection", action="store_true")

    # ── Data selection ────────────────────────────────────────────────────────
    p.add_argument("--max_items", type=int, default=-1,
                   help="Max items to process (-1 = all).")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    p.add_argument("--min_trajectory_steps", type=int, default=2)
    p.add_argument("--max_trajectory_steps", type=int, default=50)
    p.add_argument("--allow_failed", action="store_true",
                   help="Also save trajectories that did not reach the success threshold.")
    p.add_argument("--partial_success_min_score", type=float, default=0.0,
                   help="When --allow_failed is set, keep only partial trajectories whose "
                        "final score is at least this value (0.0 = keep all).")
    p.add_argument("--include_goal_reached_examples", action="store_true",
                   help="Emit one-turn 'already at goal → done' examples for items whose "
                        "initial pose already satisfies success_threshold. Teaches the model "
                        'to issue done immediately when the task is already satisfied.')

    # ── Format ────────────────────────────────────────────────────────────────
    p.add_argument("--prompt_format", default="free_think",
                   choices=["free_think", "no_think", "grounding", "worldmodeling",
                            "grounding_worldmodeling"])
    p.add_argument("--no_think", action="store_true",
                   help="Omit <think> blocks in assistant responses.")
    p.add_argument("--no_scores_in_think", action="store_true",
                   help="Do not include numerical scores in <think> blocks.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def main():
    args = _parse_args()

    cfg = SFTGenerationConfig(
        # I/O
        jsonl_path=args.jsonl_path,
        gs_root=args.gs_root,
        output_dir=args.output_dir,
        output_name=args.output_name,
        # Rendering
        render_backend=args.render_backend,
        client_url=args.client_url,
        gpu_device=args.gpu_device,
        image_width=args.image_width,
        image_height=args.image_height,
        image_format=args.image_format,
        image_quality=args.image_quality,
        save_images=not args.no_save_images,
        # Camera
        step_translation=args.step_translation,
        step_rotation_deg=args.step_rotation_deg,
        # Path finding
        success_threshold=args.success_threshold,
        max_total_actions=args.max_total_actions,
        max_actions_per_turn=args.max_actions_per_turn,
        min_improvement=args.min_improvement,
        plateau_tolerance=args.plateau_tolerance,
        beam_width=args.beam_width,
        adaptive_min_improvement=not args.no_adaptive_min_improvement,
        use_guided_search=not args.no_guided_search,
        # Potential field
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        # Collision
        enable_collision_detection=not args.no_collision_detection,
        # Data selection
        max_items=args.max_items,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        min_trajectory_steps=args.min_trajectory_steps,
        max_trajectory_steps=args.max_trajectory_steps,
        only_successful=not args.allow_failed,
        partial_success_min_score=args.partial_success_min_score,
        include_goal_reached_examples=args.include_goal_reached_examples,
        # Format
        prompt_format=args.prompt_format,
        add_think=not args.no_think,
        include_score_in_think=not args.no_scores_in_think,
        # Misc
        verbose=args.verbose,
    )

    print("[SFT Generator] Configuration:")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    print()

    gen = SFTDataGenerator(cfg)
    stats = gen.run()

    # Save stats alongside the output
    stats_path = Path(cfg.output_dir) / f"{cfg.output_name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[SFT Generator] Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
