"""Fail-closed admission of versioned, runtime-replayed canonical datasets.

Certificates are local provenance records, not cryptographic attestations.
They bind complete rows (including instructions) and the execution protocol.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .canonical_camera import CANONICAL_CAMERA_H1_RESIZE_V1
from .canonical_task_metrics import CANONICAL_TASK_METRIC_VERSION, uses_canonical_backend

CONTRACT_VERSION = "active_spatial_runtime_verified_v1"
PROTOCOL_KEYS = (
    "image_width", "image_height", "step_translation", "step_rotation_deg",
    "action_space", "enable_explicit_done", "max_actions_per_step",
    "success_score_threshold", "success_require_both", "success_position_threshold",
    "success_orientation_threshold", "enable_auto_termination", "max_episode_steps",
    "enable_collision_detection", "collision_camera_radius", "collision_floor_height",
    "collision_ceiling_height", "collision_safety_margin", "collision_invalidate_action",
    "enable_low_info_frame_check", "low_info_image_std_threshold", "max_consecutive_low_info_frames",
    "max_consecutive_collisions", "use_visual_bbox_scoring",
)


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()


def protocol(config):
    raw = config if isinstance(config, dict) else asdict(config)
    return {key: raw[key] for key in PROTOCOL_KEYS}


def validate_canonical_row(row):
    if not uses_canonical_backend(row):
        raise ValueError("formal data requires a supported canonical task metric; regenerate legacy candidates")
    if row.get("camera_model_version") not in (
        CANONICAL_CAMERA_H1_RESIZE_V1, "canonical_h1_from_frozen_candidate_intrinsics_and_pose",
    ):
        raise ValueError("formal data requires canonical H1 camera metadata")


def validate_contract(rows, config, path):
    if not rows:
        raise ValueError("cannot certify an empty dataset")
    if not path or not Path(path).is_file():
        raise ValueError("missing dataset contract: run scripts/verify_active_spatial_dataset.py first; legacy reproduction must be explicitly enabled")
    document = json.loads(Path(path).read_text())
    if document.get("version") != CONTRACT_VERSION or document.get("status") != "PASS":
        raise ValueError("dataset contract is not a passing supported certificate")
    if document.get("protocol") != protocol(config):
        raise ValueError("dataset contract protocol differs from the runtime configuration")
    certified = document.get("rows", {})
    for row in rows:
        validate_canonical_row(row)
        evidence = certified.get(row_digest(row), {})
        if not (evidence.get("success") is True and evidence.get("rgb_match") is True
                and evidence.get("steps", 0) > 0):
            raise ValueError("uncertified or changed dataset row (instruction, camera, target or metadata)")
    return document
