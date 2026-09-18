#!/usr/bin/env python3
"""Official-RGB path audit for frozen canonical FOV repairs.

The image-quality predicates are intentionally identical to the calibrated
``projective_observability_v1_initial_keyframe`` predicates.  This module only
changes the terminal semantic check to the frozen canonical FOV metric.  It is
therefore an RGB/data-quality gate, not a new FOV success definition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from r1_canonical_tasks import canonical_fov, score_observation
from r1_projective_observability import (
    OBSERVABILITY_THRESHOLDS,
    frame_observability,
    save_path_contact_sheet,
    summarize_audits,
)


FOV_OBSERVABILITY_VERSION = "fov_observability_v1_shared_rgb_quality"


def evaluate_fov_path(
    item: dict[str, Any],
    path: list[dict[str, Any]],
    images: list[Image.Image],
    detector: Any,
) -> dict[str, Any]:
    if len(images) != len(path):
        return {
            "version": FOV_OBSERVABILITY_VERSION,
            "passed": False,
            "reasons": ["renderer_frame_count_mismatch"],
            "frames_expected": len(path),
            "frames_rendered": len(images),
            "thresholds": OBSERVABILITY_THRESHOLDS,
            "frames": [],
        }

    frames: list[dict[str, Any]] = []
    consecutive_wall = 0
    maximum_consecutive_wall = 0
    for step, (entry, image) in enumerate(zip(path, images)):
        pose = np.asarray(entry["c2w"], dtype=float)
        row = frame_observability(item, pose, image, detector)
        # frame_observability owns only the frozen RGB-quality predicates here;
        # replace its Projective diagnostic metric with the canonical FOV metric.
        row["canonical_metric"] = canonical_fov(score_observation(item, pose))
        row["step"] = step
        row["action"] = entry.get("action")
        frames.append(row)
        consecutive_wall = consecutive_wall + 1 if row["wall_like"] else 0
        maximum_consecutive_wall = max(maximum_consecutive_wall, consecutive_wall)

    dual_steps = [row["step"] for row in frames if row["dual_target_observation"]]
    initial = frames[0]
    final = frames[-1]
    reasons: list[str] = []
    if initial["canonical_metric"]["success"]:
        reasons.append("initial_canonical_fov_already_successful")
    if not final["canonical_metric"]["success"]:
        reasons.append("terminal_canonical_fov_not_successful")
    if not initial["dual_target_observation"]:
        reasons.append("initial_dual_target_not_discernible")
    if not final["dual_target_observation"]:
        reasons.append("terminal_dual_target_not_discernible")
    if len(dual_steps) < int(OBSERVABILITY_THRESHOLDS["minimum_dual_target_frames"]):
        reasons.append("insufficient_dual_target_path_frames")
    if maximum_consecutive_wall > int(
        OBSERVABILITY_THRESHOLDS["maximum_consecutive_wall_like_frames"]
    ):
        reasons.append("consecutive_near_wall_or_low_structure_frames")
    if not all(row["wall_clear"] for row in frames):
        reasons.append("path_wall_clearance_below_frozen_minimum")
    return {
        "version": FOV_OBSERVABILITY_VERSION,
        "rgb_quality_predicate_source": "projective_observability_v1_initial_keyframe",
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


__all__ = [
    "FOV_OBSERVABILITY_VERSION",
    "evaluate_fov_path",
    "save_path_contact_sheet",
    "summarize_audits",
]
