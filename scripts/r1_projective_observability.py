#!/usr/bin/env python3
"""Versioned RGB/geometric observability audit for R1 Projective paths.

This is an auditable heuristic quality gate, not a true occlusion oracle.  It
combines canonical bbox projections, local RGB texture inside each projected
bbox, whole-frame near-wall signals, room-wall clearance, and temporal path
coverage.  Threshold provenance is recorded in the calibration report.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from r1_canonical_tasks import canonical_projective, score_observation


PROJECTIVE_OBSERVABILITY_VERSION = "projective_observability_v1_initial_keyframe"

# Frozen from positive/negative real-render calibration; see the versioned
# calibration report.  These do not alter canonical task success.
OBSERVABILITY_THRESHOLDS: dict[str, float | int] = {
    "minimum_bbox_area_px": 64.0,
    "minimum_inside_frame_fraction": 0.25,
    "minimum_patch_entropy_bits": 2.0,
    "minimum_patch_edge_fraction": 0.03,
    "maximum_patch_clipped_luminance_fraction": 0.65,
    "wall_like_max_entropy_bits": 3.0,
    "wall_like_max_edge_fraction": 0.05,
    "wall_like_min_clipped_luminance_fraction": 0.45,
    "maximum_consecutive_wall_like_frames": 2,
    # A one-action path has only an unsuccessful initial frame and a successful
    # terminal frame.  Requiring two dual-target frames would silently turn
    # observation quality into a Projective minimum-path-length definition.
    "minimum_dual_target_frames": 1,
    # The initial state is a key task observation.  A path may not recover
    # acceptability merely by turning away from an uninformative initial RGB.
    # This is a boolean key-frame requirement, not a new difficulty threshold.
    "require_initial_dual_target_observation": 1,
    "minimum_wall_clearance_m": 0.5,
}


def _area(box: list[float] | None) -> float:
    if not box or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _inside_fraction(obj: dict[str, Any]) -> float:
    raw = _area(obj.get("bbox_raw"))
    return _area(obj.get("bbox")) / raw if raw > 1e-8 else 0.0


def _entropy(gray: np.ndarray) -> float:
    histogram = np.histogram(gray, bins=32, range=(0.0, 256.0))[0].astype(float)
    histogram = histogram[histogram > 0]
    if not histogram.size:
        return 0.0
    probabilities = histogram / histogram.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def rgb_features(image: Image.Image, bbox: list[float] | None = None) -> dict[str, Any]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    if bbox is not None:
        x1 = max(0, min(width - 1, int(math.floor(float(bbox[0])))))
        y1 = max(0, min(height - 1, int(math.floor(float(bbox[1])))))
        x2 = max(x1 + 1, min(width, int(math.ceil(float(bbox[2])))))
        y2 = max(y1 + 1, min(height, int(math.ceil(float(bbox[3])))))
        array = array[y1:y2, x1:x2]
    gray = array.mean(axis=2)
    gradients = np.concatenate(
        (
            np.abs(np.diff(gray, axis=1)).ravel(),
            np.abs(np.diff(gray, axis=0)).ravel(),
        )
    )
    return {
        "pixel_count": int(gray.size),
        "mean_luminance": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "entropy_bits_32bin": _entropy(gray),
        "edge_fraction_gt8": float((gradients > 8.0).mean()) if gradients.size else 0.0,
        "mean_absolute_gradient": float(gradients.mean()) if gradients.size else 0.0,
        "clipped_luminance_fraction": float(((gray < 8.0) | (gray > 247.0)).mean()),
    }


def detector_wall_distance(detector: Any, position: np.ndarray) -> float | None:
    if detector is None or not getattr(detector, "wall_segments", None):
        return None
    point = np.asarray(position, dtype=float)[:2]
    distances = [
        float(detector._point_to_segment_distance_2d(point, start, end))
        for start, end in detector.wall_segments
    ]
    return min(distances) if distances else None


def frame_observability(
    item: dict[str, Any],
    pose: np.ndarray,
    image: Image.Image,
    detector: Any,
) -> dict[str, Any]:
    result = score_observation(item, np.asarray(pose, dtype=float))
    metric = canonical_projective(result)
    objects = result["visual_metrics"].get("objects") or []
    frame_rgb = rgb_features(image)
    threshold = OBSERVABILITY_THRESHOLDS
    wall_like = bool(
        frame_rgb["edge_fraction_gt8"] < threshold["wall_like_max_edge_fraction"]
        and (
            frame_rgb["entropy_bits_32bin"] < threshold["wall_like_max_entropy_bits"]
            or frame_rgb["clipped_luminance_fraction"]
            > threshold["wall_like_min_clipped_luminance_fraction"]
        )
    )
    object_rows = []
    for obj in objects:
        patch = rgb_features(image, obj.get("bbox"))
        area_px = _area(obj.get("bbox"))
        inside = _inside_fraction(obj)
        gates = {
            "projectable": bool(obj.get("available")),
            "in_front": bool(obj.get("center_in_front")),
            "bbox_area": area_px >= threshold["minimum_bbox_area_px"],
            "inside_frame_fraction": inside >= threshold["minimum_inside_frame_fraction"],
            "patch_entropy": patch["entropy_bits_32bin"]
            >= threshold["minimum_patch_entropy_bits"],
            "patch_edges": patch["edge_fraction_gt8"]
            >= threshold["minimum_patch_edge_fraction"],
            "patch_not_luminance_clipped": patch["clipped_luminance_fraction"]
            <= threshold["maximum_patch_clipped_luminance_fraction"],
        }
        object_rows.append(
            {
                "id": obj.get("id"),
                "label": obj.get("label"),
                "bbox": obj.get("bbox"),
                "bbox_raw": obj.get("bbox_raw"),
                "bbox_area_px": area_px,
                "inside_frame_fraction": inside,
                "rgb_patch": patch,
                "gates": gates,
                "discernible_heuristic": all(gates.values()),
            }
        )
    wall_distance = detector_wall_distance(detector, np.asarray(pose)[:3, 3])
    wall_clear = bool(
        wall_distance is not None
        and wall_distance >= threshold["minimum_wall_clearance_m"]
    )
    dual_target = bool(
        len(object_rows) == 2
        and all(row["discernible_heuristic"] for row in object_rows)
        and wall_clear
        and not wall_like
    )
    return {
        "canonical_metric": metric,
        "frame_rgb": frame_rgb,
        "wall_like": wall_like,
        "wall_distance_m": wall_distance,
        "wall_clear": wall_clear,
        "objects": object_rows,
        "dual_target_observation": dual_target,
    }


def evaluate_projective_path(
    item: dict[str, Any],
    path: list[dict[str, Any]],
    images: list[Image.Image],
    detector: Any,
) -> dict[str, Any]:
    if len(images) != len(path):
        return {
            "version": PROJECTIVE_OBSERVABILITY_VERSION,
            "passed": False,
            "reasons": ["renderer_frame_count_mismatch"],
            "frames_expected": len(path),
            "frames_rendered": len(images),
            "thresholds": OBSERVABILITY_THRESHOLDS,
            "frames": [],
        }
    frames = []
    consecutive_wall = 0
    maximum_consecutive_wall = 0
    for step, (entry, image) in enumerate(zip(path, images)):
        row = frame_observability(item, np.asarray(entry["c2w"], dtype=float), image, detector)
        row["step"] = step
        row["action"] = entry.get("action")
        frames.append(row)
        consecutive_wall = consecutive_wall + 1 if row["wall_like"] else 0
        maximum_consecutive_wall = max(maximum_consecutive_wall, consecutive_wall)

    dual_steps = [row["step"] for row in frames if row["dual_target_observation"]]
    final = frames[-1]
    reasons = []
    if not final["canonical_metric"]["success"]:
        reasons.append("terminal_canonical_relation_not_successful")
    if not final["dual_target_observation"]:
        reasons.append("terminal_dual_target_not_discernible")
    if (
        OBSERVABILITY_THRESHOLDS["require_initial_dual_target_observation"]
        and not frames[0]["dual_target_observation"]
    ):
        reasons.append("initial_dual_target_not_discernible")
    if len(dual_steps) < int(OBSERVABILITY_THRESHOLDS["minimum_dual_target_frames"]):
        reasons.append("insufficient_dual_target_path_frames")
    if maximum_consecutive_wall > int(
        OBSERVABILITY_THRESHOLDS["maximum_consecutive_wall_like_frames"]
    ):
        reasons.append("consecutive_near_wall_or_low_structure_frames")
    if not all(row["wall_clear"] for row in frames):
        reasons.append("path_wall_clearance_below_frozen_minimum")
    return {
        "version": PROJECTIVE_OBSERVABILITY_VERSION,
        "oracle_claim": "heuristic_not_true_occlusion_oracle",
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": OBSERVABILITY_THRESHOLDS,
        "frames_expected": len(path),
        "frames_rendered": len(images),
        "dual_target_steps": dual_steps,
        "maximum_consecutive_wall_like_frames": maximum_consecutive_wall,
        "frames": frames,
    }


def save_path_contact_sheet(
    images: list[Image.Image],
    path: list[dict[str, Any]],
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = []
    image_paths = []
    for step, (image, entry, frame) in enumerate(zip(images, path, audit.get("frames", []))):
        action = entry.get("action") or "initial"
        frame_path = output_dir / f"step_{step:02d}_{action}.png"
        image.convert("RGB").save(frame_path)
        image_paths.append(str(frame_path))
        panel = image.convert("RGB").copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width - 1, 20), fill=(0, 0, 0))
        draw.text(
            (3, 3),
            f"{step}:{action} dual={int(frame['dual_target_observation'])} wall={int(frame['wall_like'])}",
            fill=(0, 255, 0) if frame["dual_target_observation"] else (255, 255, 0),
        )
        annotated.append(panel)
    sheet_path = output_dir / "observability_contact_sheet.png"
    if annotated:
        sheet = Image.new(
            "RGB",
            (sum(panel.width for panel in annotated), max(panel.height for panel in annotated)),
            "white",
        )
        cursor = 0
        for panel in annotated:
            sheet.paste(panel, (cursor, 0))
            cursor += panel.width
        sheet.save(sheet_path)
    return {
        "frame_images": image_paths,
        "contact_sheet": str(sheet_path) if annotated else None,
    }


def summarize_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    for row in audits:
        reasons.update(row.get("reasons") or [])
    return {
        "version": PROJECTIVE_OBSERVABILITY_VERSION,
        "rows": len(audits),
        "passed": sum(bool(row.get("passed")) for row in audits),
        "failed": sum(not bool(row.get("passed")) for row in audits),
        "failure_reasons": dict(reasons),
        "thresholds": OBSERVABILITY_THRESHOLDS,
        "oracle_claim": "heuristic_not_true_occlusion_oracle",
    }
