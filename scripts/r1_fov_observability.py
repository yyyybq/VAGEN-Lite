#!/usr/bin/env python3
"""Official-RGB path audit for frozen canonical FOV repairs.

The image-quality predicates are intentionally identical to the calibrated
``projective_observability_v1_initial_keyframe`` predicates.  This module only
changes the terminal semantic check to the frozen canonical FOV metric.  It is
therefore an RGB/data-quality gate, not a new FOV success definition.
"""

from __future__ import annotations

import hashlib
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

# Do not retroactively reinterpret v1 evidence.  The v1 predicate accidentally
# made a Projective-specific ``inside_frame_fraction`` requirement an FOV input
# quality requirement.  That is not valid: a well posed FOV task is supposed to
# start with an incompletely framed object.  v2 is intentionally a *separate*
# version and never mutates a v1 manifest.
FOV_OBSERVABILITY_V2_VERSION = "fov_observability_v2_keyframe_rgb_integrity"

# These values are frozen from the documented FOV pilot positives by
# r1_fov_observability_calibrate.py.  In particular, there is deliberately no
# minimum inside-frame fraction: several manually accepted FOV pilots have a
# tiny raw-bbox fraction while retaining a large, textured, recognisable image
# region.  The calibration report explicitly records whether a human-labelled
# negative set was available; callers must not describe this heuristic as an
# occlusion oracle or a new canonical FOV success definition.
FOV_V2_THRESHOLDS: dict[str, float | int] = {
    # Rounded only outward from the extrema of every rendered key/path frame
    # in the 15-row manually accepted FOV pilot: area 5140.74px², entropy
    # 3.3521 bits, edge fraction .06584, and clipping .07620.  The outward
    # rounding avoids rejecting an accepted pilot because of image IO noise.
    "minimum_visible_bbox_area_px": 5120.0,
    "minimum_patch_entropy_bits": 3.35,
    "minimum_patch_edge_fraction": 0.065,
    "maximum_patch_clipped_luminance_fraction": 0.08,
    "minimum_frame_mean_luminance": 10.0,
    "minimum_frame_luminance_std": 5.0,
    "minimum_dual_target_keyframes": 2,
}


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


def _fov_v2_object_quality(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate discernibility without importing Projective's crop fraction.

    ``bbox_area_px`` is the on-image clipped area and is therefore the useful
    quantity for an incompletely framed FOV object.  ``inside_frame_fraction``
    remains recorded in every audit for diagnosis only.
    """
    thresholds = FOV_V2_THRESHOLDS
    patch = row["rgb_patch"]
    reasons: list[str] = []
    if not bool(row["gates"].get("projectable")):
        reasons.append("object_not_projectable")
    if float(row["bbox_area_px"]) < float(thresholds["minimum_visible_bbox_area_px"]):
        reasons.append("visible_bbox_area_too_small")
    if float(patch["entropy_bits_32bin"]) < float(thresholds["minimum_patch_entropy_bits"]):
        reasons.append("object_patch_low_texture")
    if float(patch["edge_fraction_gt8"]) < float(thresholds["minimum_patch_edge_fraction"]):
        reasons.append("object_patch_low_structure")
    if float(patch["clipped_luminance_fraction"]) > float(
        thresholds["maximum_patch_clipped_luminance_fraction"]
    ):
        reasons.append("object_patch_luminance_clipped")
    return not reasons, reasons


def _fov_v2_frame_quality(frame: dict[str, Any]) -> tuple[bool, list[str]]:
    thresholds = FOV_V2_THRESHOLDS
    reasons: list[str] = []
    rgb = frame["frame_rgb"]
    if int(rgb["pixel_count"]) <= 0:
        reasons.append("renderer_empty_frame")
    if float(rgb["mean_luminance"]) < float(thresholds["minimum_frame_mean_luminance"]):
        reasons.append("renderer_black_frame")
    if float(rgb["luminance_std"]) < float(thresholds["minimum_frame_luminance_std"]):
        reasons.append("renderer_low_information_frame")
    object_reasons: list[str] = []
    for index, obj in enumerate(frame["objects"]):
        passed, reasons_for_object = _fov_v2_object_quality(obj)
        obj["fov_v2_discernible"] = passed
        obj["fov_v2_reasons"] = reasons_for_object
        object_reasons.extend(f"object_{index}_{reason}" for reason in reasons_for_object)
    if len(frame["objects"]) != 2:
        reasons.append("not_exactly_two_target_objects")
    reasons.extend(object_reasons)
    frame["fov_v2_dual_target_observation"] = not reasons
    return not reasons, reasons


def _pose_key(entry: dict[str, Any]) -> tuple[float, ...] | None:
    pose = np.asarray(entry.get("c2w"), dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return tuple(np.round(pose.ravel(), 8).tolist())


def evaluate_fov_path_v2(
    item: dict[str, Any],
    path: list[dict[str, Any]],
    images: list[Image.Image],
    detector: Any,
) -> dict[str, Any]:
    """Audit FOV keyframes plus renderer frame/pose integrity.

    The canonical FOV scorer remains the sole source of task success.  This
    function assesses whether RGB supplied to the policy can support the task;
    it does not impose an FOV success threshold or path-length rule.
    """
    if len(images) != len(path):
        return {
            "version": FOV_OBSERVABILITY_V2_VERSION,
            "passed": False,
            "reasons": ["renderer_frame_count_mismatch"],
            "frames_expected": len(path),
            "frames_rendered": len(images),
            "thresholds": FOV_V2_THRESHOLDS,
            "frames": [],
        }

    frames: list[dict[str, Any]] = []
    image_hashes: list[str] = []
    pose_keys: list[tuple[float, ...] | None] = []
    all_frame_reasons: dict[int, list[str]] = {}
    for step, (entry, image) in enumerate(zip(path, images)):
        pose = np.asarray(entry.get("c2w"), dtype=float)
        row = frame_observability(item, pose, image, detector)
        # frame_observability's Projective-quality Boolean is not used here.
        row["canonical_metric"] = canonical_fov(score_observation(item, pose))
        row["step"] = step
        row["action"] = entry.get("action")
        _, frame_reasons = _fov_v2_frame_quality(row)
        if frame_reasons:
            all_frame_reasons[step] = frame_reasons
        frames.append(row)
        image_hashes.append(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
        pose_keys.append(_pose_key(entry))

    reasons: list[str] = []
    initial, final = frames[0], frames[-1]
    if initial["canonical_metric"]["success"]:
        reasons.append("initial_canonical_fov_already_successful")
    if not final["canonical_metric"]["success"]:
        reasons.append("terminal_canonical_fov_not_successful")
    for keyframe_name, frame in (("initial", initial), ("terminal", final)):
        if not frame.get("fov_v2_dual_target_observation"):
            reasons.append(f"{keyframe_name}_dual_target_not_discernible")
    for step, frame_reasons in all_frame_reasons.items():
        # Render integrity applies to every path frame; object discernibility
        # only applies to the initial and terminal keyframes above.
        integrity = [reason for reason in frame_reasons if reason.startswith("renderer_")]
        reasons.extend(f"step_{step}_{reason}" for reason in integrity)
    for step in range(1, len(path)):
        if pose_keys[step] is None:
            reasons.append(f"step_{step}_invalid_pose")
        elif pose_keys[step] == pose_keys[step - 1] and path[step].get("action"):
            reasons.append(f"step_{step}_pose_reused_after_action")
        elif image_hashes[step] == image_hashes[step - 1] and pose_keys[step] != pose_keys[step - 1]:
            reasons.append(f"step_{step}_renderer_image_reused_for_distinct_pose")
    if pose_keys and pose_keys[0] is None:
        reasons.append("step_0_invalid_pose")

    keyframe_steps = [0, len(frames) - 1]
    return {
        "version": FOV_OBSERVABILITY_V2_VERSION,
        "oracle_claim": "heuristic_not_true_occlusion_oracle",
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": FOV_V2_THRESHOLDS,
        "frames_expected": len(path),
        "frames_rendered": len(images),
        "keyframe_steps": keyframe_steps,
        "keyframe_dual_target_steps": [
            step for step in keyframe_steps if frames[step].get("fov_v2_dual_target_observation")
        ],
        "all_frame_quality_reasons": all_frame_reasons,
        "frames": frames,
    }


__all__ = [
    "FOV_OBSERVABILITY_VERSION",
    "FOV_OBSERVABILITY_V2_VERSION",
    "FOV_V2_THRESHOLDS",
    "evaluate_fov_path",
    "evaluate_fov_path_v2",
    "save_path_contact_sheet",
    "summarize_audits",
]
