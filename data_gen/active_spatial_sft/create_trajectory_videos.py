#!/usr/bin/env python3
"""
Create MP4 trajectory videos from SFT output data.

After running run_sft_generation.py, this script reads the output JSONL
and stitches the saved images for each trajectory into an MP4 video,
annotated with the pose, score, and action information.

Usage
-----
    python create_trajectory_videos.py \\
        --sft_jsonl /path/to/sft_output/sft_data.jsonl \\
        --output_dir /path/to/sft_output/videos \\
        --fps 4 \\
        --max_videos 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_VAGEN_ROOT = _HERE.parent.parent
if str(_VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VAGEN_ROOT))


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _get_font(size: int = 16):
    """Try to get a proportional font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _annotate_frame(
    img: Image.Image,
    text_lines: List[str],
    bg_alpha: int = 180,
    font_size: int = 14,
) -> Image.Image:
    """Overlay text lines at the bottom of an image."""
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)

    line_h = font_size + 4
    n = len(text_lines)
    box_h = n * line_h + 8
    w, h = img.size
    y0 = h - box_h

    # Semi-transparent background
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, y0, w, h], fill=(0, 0, 0, bg_alpha))
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    img = img_rgba.convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(text_lines):
        y = y0 + 4 + i * line_h
        draw.text((6, y), line, fill=(255, 255, 255), font=font)

    return img


def _parse_action_for_step(conversations: List[Dict], step_idx: int) -> str:
    """Extract action string for step_idx from conversations."""
    # conversations: sys, user_0, asst_0, user_1, asst_1, ...
    # step_idx 0 corresponds to asst_0 (index 2)
    asst_idx = 2 + step_idx * 2
    if asst_idx < len(conversations):
        content = conversations[asst_idx].get("content", "")
        import re
        m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if m:
            return m.group(1).strip().replace("|", " → ")
    return ""


# ---------------------------------------------------------------------------
# Video writer (imageio-based with ffmpeg fallback)
# ---------------------------------------------------------------------------

def _frames_to_mp4(
    frames: List[Image.Image],
    output_path: Path,
    fps: int = 4,
) -> bool:
    """Write a list of PIL Images to an MP4 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try imageio first (most common)
    try:
        import imageio
        writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                    quality=7, pixelformat="yuv420p")
        for frame in frames:
            writer.append_data(np.array(frame.convert("RGB")))
        writer.close()
        return True
    except Exception as e:
        pass

    # Fallback: save frames to tmp dir and call ffmpeg
    try:
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                frame.save(os.path.join(tmpdir, f"frame_{i:04d}.jpg"), quality=90)
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%04d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            return result.returncode == 0
    except Exception as e:
        print(f"[VideoWriter] ffmpeg fallback failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main video creation
# ---------------------------------------------------------------------------

def create_video_for_record(
    record: Dict[str, Any],
    sft_dir: Path,
    output_dir: Path,
    fps: int = 4,
    font_size: int = 14,
    hold_last_frame: int = 2,
) -> Optional[Path]:
    """Create an annotated MP4 for one SFT trajectory record.

    Args:
        record: One SFT JSONL record.
        sft_dir: Root directory of the SFT output (images are relative to this).
        output_dir: Where to save the MP4.
        fps: Frames per second.
        font_size: Annotation font size.
        hold_last_frame: Number of duplicate frames to hold at the end.

    Returns:
        Path to the created MP4, or None on failure.
    """
    image_paths = record.get("image_paths", [])
    conversations = record.get("conversations", [])
    sft_id = record.get("id", "unknown")
    task_type = record.get("task_type", "")
    task_desc = record.get("task_description", "")[:60]
    init_score = record.get("initial_score", 0)
    final_score = record.get("final_score", 0)
    success = record.get("success", False)

    if not image_paths:
        return None

    frames: List[Image.Image] = []

    for img_idx, rel_path in enumerate(image_paths):
        full_path = sft_dir / rel_path
        if not full_path.exists():
            continue

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception:
            continue

        # Build annotation text
        step_label = f"Step {img_idx}/{len(image_paths) - 1}"
        if img_idx == 0:
            action_str = "Initial view"
        else:
            action_str = _parse_action_for_step(conversations, img_idx - 1)
            if not action_str:
                action_str = ""

        # Score progress annotation
        if img_idx == 0:
            score_str = f"Init score: {init_score:.3f}"
        elif img_idx == len(image_paths) - 1:
            status = "SUCCESS" if success else "PARTIAL"
            score_str = f"Final score: {final_score:.3f} [{status}]"
        else:
            score_str = ""

        lines = [
            f"{sft_id}  {step_label}  |  {task_type}",
            f"Task: {task_desc}",
        ]
        if action_str:
            lines.append(f"Action: {action_str}")
        if score_str:
            lines.append(score_str)

        annotated = _annotate_frame(img, lines, font_size=font_size)
        frames.append(annotated)

    if not frames:
        return None

    # Hold last frame
    for _ in range(hold_last_frame):
        frames.append(frames[-1])

    video_path = output_dir / f"{sft_id}.mp4"
    ok = _frames_to_mp4(frames, video_path, fps=fps)
    if ok:
        return video_path
    return None


def create_all_videos(
    sft_jsonl: Path,
    output_dir: Path,
    fps: int = 4,
    max_videos: int = -1,
    font_size: int = 14,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Create MP4 videos for all (or up to max_videos) SFT records."""
    sft_dir = sft_jsonl.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(sft_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_videos > 0:
        records = records[:max_videos]

    print(f"[VideoGen] Creating videos for {len(records)} records → {output_dir}")

    stats = {"total": len(records), "created": 0, "failed": 0}

    for i, record in enumerate(records):
        sft_id = record.get("id", f"sft_{i:06d}")
        video_path = create_video_for_record(
            record=record,
            sft_dir=sft_dir,
            output_dir=output_dir,
            fps=fps,
            font_size=font_size,
        )
        if video_path:
            stats["created"] += 1
            if verbose:
                print(f"  [{i+1}/{len(records)}] {sft_id} → {video_path.name}")
        else:
            stats["failed"] += 1
            if verbose:
                print(f"  [{i+1}/{len(records)}] {sft_id} → FAILED")

        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(records) * 100
            print(f"[VideoGen] {i+1}/{len(records)} ({pct:.1f}%) "
                  f"created={stats['created']} failed={stats['failed']}")

    print(f"\n[VideoGen] Done: {stats['created']} videos created, {stats['failed']} failed")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Create MP4 trajectory videos from SFT output JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sft_jsonl", required=True,
                   help="Path to the SFT output JSONL file.")
    p.add_argument("--output_dir", default="",
                   help="Directory to save videos. Default: <sft_jsonl parent>/videos/")
    p.add_argument("--fps", type=int, default=4, help="Video frame rate.")
    p.add_argument("--max_videos", type=int, default=-1,
                   help="Max number of videos to create (-1 = all).")
    p.add_argument("--font_size", type=int, default=14)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    sft_jsonl = Path(args.sft_jsonl)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = sft_jsonl.parent / "videos"

    create_all_videos(
        sft_jsonl=sft_jsonl,
        output_dir=output_dir,
        fps=args.fps,
        max_videos=args.max_videos,
        font_size=args.font_size,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
