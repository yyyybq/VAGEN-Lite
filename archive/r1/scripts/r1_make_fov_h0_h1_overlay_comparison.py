#!/usr/bin/env python3
"""Create supplemental FOV H0-vs-H1 overlay comparisons from existing artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_AUDIT_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/"
    "r1_observation_aligned_reward_audit"
)
DEFAULT_FOV_DIR = DEFAULT_AUDIT_ROOT / "r1_fov_h1_audit_20260902"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def draw_box(draw: ImageDraw.ImageDraw, box: Any, color: tuple[int, int, int], label: str) -> None:
    if not box:
        return
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return
    draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
    draw.text((x0, max(0.0, y0 - 12.0)), label, fill=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fov-dir", type=Path, default=DEFAULT_FOV_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "r1_fov_h0_h1_overlay_comparison_20260904",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.fov_dir / "fov_h1_overlays"
    audit_by_index = {
        row["dataset_index"]: row
        for row in read_jsonl(args.fov_dir / "r1_fov_h1_audit_manifest.jsonl")
    }
    overlay_rows = read_jsonl(overlay_dir / "manifest.jsonl")

    manifest: list[dict[str, Any]] = []
    thumbs: list[Image.Image] = []
    for row in overlay_rows:
        rec = audit_by_index[row["dataset_index"]]
        image = Image.open(overlay_dir / row["image"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in rec.get("h0_target", {}).get("objects") or []:
            draw_box(draw, obj.get("bbox"), (0, 220, 255), f"H0 {obj.get('label') or obj.get('id')}")
        for obj in rec.get("h1_target", {}).get("objects") or []:
            draw_box(draw, obj.get("bbox"), (255, 40, 40), f"H1 {obj.get('label') or obj.get('id')}")
        draw.text(
            (3, 242),
            (
                f"H0={rec['h0_target']['success']} "
                f"H1={rec['h1_target']['success']} "
                f"idx={rec['dataset_index']}"
            ),
            fill=(255, 255, 0),
        )
        name = f"{rec['dataset_index']:05d}_{rec['scene_id']}_h0_h1.png"
        image.save(args.output_dir / name)
        thumbs.append(image.resize((128, 128)))
        manifest.append(
            {
                "dataset_index": rec["dataset_index"],
                "scene_id": rec["scene_id"],
                "image": name,
                "base_image": row["image"],
                "note": "cyan=H0 projected bbox, red=H1 projected bbox on saved real H1 RGB overlay image",
                "h0_target": rec["h0_target"],
                "h1_target": rec["h1_target"],
            }
        )

    cols = 10
    rows = max(1, int(math.ceil(len(thumbs) / cols)))
    sheet = Image.new("RGB", (cols * 128, rows * 128), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 128, (idx // cols) * 128))
    sheet.save(args.output_dir / "contact_sheet.png")

    with (args.output_dir / "manifest.jsonl").open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    summary = {
        "rows": len(manifest),
        "contact_sheet": str(args.output_dir / "contact_sheet.png"),
        "manifest": str(args.output_dir / "manifest.jsonl"),
        "note": "supplemental overlay; source RGB comes from existing H1 real-render overlay artifact",
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
