"""Shared contract for the InteriorGS v10 task generator.

Every module in ``data_gen/interiorgs_v10`` imports constants and simple data
types from here so that independently written modules agree on units, camera
conventions and record schemas.  Keep this file dependency-light (numpy only)
and Python 3.10 compatible.

Coordinate conventions (inherited from R1, see vagen/envs/active_spatial):
  * World: metres, Z up.  labels.json boxes are 8 corner dicts {x, y, z}.
  * structure.json room ``profile`` points are [x, y]; their Y sign relative to
    labels.json is scene dependent (see ``scene.py`` / coordinate calibration).
  * Camera c2w: OpenCV, X right, Y down, Z forward (``camera_pose_from_forward``).
  * Motion model (vagen/envs/active_spatial/utils.py ViewManipulator):
      move_forward/backward translate along camera +Z/-Z,
      move_left/right along camera -X/+X,
      turn_left/right rotate about the camera local Y axis.
    With pitch 0 this keeps the camera on a horizontal plane.  A non-zero
    pitch would make translations change height, so the v1 protocol keeps
    pitch at exactly 0 for every state of an episode.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------
GENERATOR_VERSION = "interiorgs_v10_generator_dev"
# Canonical metric for Projective / FOV stays the frozen R1 backend.
CANONICAL_TASK_METRIC_V1 = "canonical_spatial_task_h1_v1"
# New task families must register under a NEW version string; v1 behaviour of
# projective_relations / fov_inclusion must not change.
CANONICAL_TASK_METRIC_V2 = "canonical_spatial_task_h1_v2"
CAMERA_MODEL_VERSION = "canonical_camera_h1_resize_v1"  # must equal vagen canonical_camera.CANONICAL_CAMERA_H1_RESIZE_V1

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
NATIVE_K = [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
NATIVE_SIZE = (640, 480)
RENDER_SIZE = (256, 256)
CAMERA_HEIGHT_M = 1.5  # all R1 canonical poses use z = 1.5 m, pitch 0

# --------------------------------------------------------------------------
# Action protocol (R1 strafe protocol)
# --------------------------------------------------------------------------
ACTIONS: Tuple[str, ...] = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
)
INVERSE_ACTION = {
    "move_forward": "move_backward",
    "move_backward": "move_forward",
    "move_left": "move_right",
    "move_right": "move_left",
    "turn_left": "turn_right",
    "turn_right": "turn_left",
}
STEP_TRANSLATION_M = 0.3
STEP_ROTATION_DEG = 20.0
MAX_PRIMITIVE_STEPS = 12
MAX_ACTIONS_PER_TURN = 5
ACTION_PROTOCOL_VERSION = "r1_strafe6_t0.3_r20_max12"

# Collision geometry (matches CollisionDetector defaults used by R1)
CAMERA_RADIUS_M = 0.15
SAFETY_MARGIN_M = 0.05
MIN_WALL_DISTANCE_M = 0.5  # R1 generator safe subset; runtime itself is looser

# State quantisation for de-duplication in searches
POS_QUANT_M = 0.01
YAW_QUANT_DEG = 1.0


# --------------------------------------------------------------------------
# Motion model
# --------------------------------------------------------------------------
# "r1" is the protocol above (pitch 0) and the only one the RL env implements.
# "horizon_locked" is EXPERIMENTAL and generator-side only: a constant pitch,
# move_forward/backward along the horizontal heading, move_left/right along the
# (horizontal) camera X axis, turns about world Z.  At pitch 0 it is the r1
# motion up to float rounding, so r1 keeps its own code paths bit-for-bit.
MOTION_MODELS = ("r1", "horizon_locked")
_MOTION: Dict[str, Any] = {"model": "r1", "pitch_deg": 0.0}


def set_motion_model(model: str = "r1", pitch_deg: float = 0.0) -> None:
    if model not in MOTION_MODELS:
        raise ValueError(f"unknown motion model {model!r}")
    if model == "r1" and float(pitch_deg) != 0.0:
        raise ValueError("the r1 motion model keeps pitch at 0")
    _MOTION.update(model=model, pitch_deg=float(pitch_deg))


def motion_model() -> Tuple[str, float]:
    return str(_MOTION["model"]), float(_MOTION["pitch_deg"])


def horizon_locked() -> bool:
    return _MOTION["model"] == "horizon_locked"


def camera_forward_z() -> float:
    """c2w[2, 2] (forward z) of every legal pose under the active motion model."""
    return math.sin(math.radians(float(_MOTION["pitch_deg"]))) if horizon_locked() else 0.0


def yaw_of(c2w: np.ndarray) -> float:
    """Yaw (radians) of the camera forward axis projected on the XY plane."""
    f = np.asarray(c2w, dtype=float)[:3, 2]
    return math.atan2(float(f[1]), float(f[0]))


def pose_from_xy_yaw(x: float, y: float, yaw: float, z: float = CAMERA_HEIGHT_M) -> np.ndarray:
    """c2w at (x, y, z) looking along world yaw (radians); pitch 0 unless a
    horizon-locked motion model with a non-zero pitch is active."""
    if horizon_locked() and float(_MOTION["pitch_deg"]) != 0.0:
        p = math.radians(float(_MOTION["pitch_deg"]))
        c, s, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(p), math.sin(p)
        c2w = np.eye(4)
        c2w[:3, 0] = [s, -c, 0.0]               # right (horizontal)
        c2w[:3, 1] = [sp * c, sp * s, -cp]      # down
        c2w[:3, 2] = [cp * c, cp * s, sp]       # forward
        c2w[:3, 3] = [x, y, z]
        return c2w
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = [x, y, z]
    return c2w


def apply_action(c2w: np.ndarray, action: str,
                 step_t: float = STEP_TRANSLATION_M,
                 step_r_deg: float = STEP_ROTATION_DEG) -> np.ndarray:
    """Pure-numpy replica of ViewManipulator.step for the six strafe actions.

    Must stay bit-compatible (up to float rounding) with
    vagen.envs.active_spatial.utils.ViewManipulator; tests compare both.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    R = c2w[:3, :3]
    C = c2w[:3, 3]
    out = np.eye(4)
    if horizon_locked():
        return _apply_action_horizon_locked(R, C, action, step_t, step_r_deg)
    if action in ("move_forward", "move_backward"):
        sign = 1.0 if action == "move_forward" else -1.0
        out[:3, :3] = R
        out[:3, 3] = C + R @ np.array([0.0, 0.0, 1.0]) * step_t * sign
        return out
    if action in ("move_right", "move_left"):
        sign = 1.0 if action == "move_right" else -1.0
        out[:3, :3] = R
        out[:3, 3] = C + R @ np.array([1.0, 0.0, 0.0]) * step_t * sign
        return out
    if action in ("turn_left", "turn_right"):
        ang = math.radians(step_r_deg) * (-1.0 if action == "turn_left" else 1.0)
        c, s = math.cos(ang), math.sin(ang)
        # rotation about local Y (scipy from_euler('y', ang))
        R_local = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        out[:3, :3] = R @ R_local
        out[:3, 3] = C
        return out
    raise ValueError(f"unknown action {action!r}")


def heading_turn(step_r_deg: float, action: str) -> np.ndarray:
    """World-Z rotation of a horizon-locked turn (turn_left increases yaw, as in r1)."""
    ang = math.radians(step_r_deg) * (1.0 if action == "turn_left" else -1.0)
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _apply_action_horizon_locked(R: np.ndarray, C: np.ndarray, action: str, step_t: float,
                                 step_r_deg: float) -> np.ndarray:
    out = np.eye(4)
    if action in ("move_forward", "move_backward"):
        sign = 1.0 if action == "move_forward" else -1.0
        h = np.array([R[0, 2], R[1, 2], 0.0])
        out[:3, :3] = R
        out[:3, 3] = C + h / np.linalg.norm(h) * step_t * sign
        return out
    if action in ("move_right", "move_left"):
        sign = 1.0 if action == "move_right" else -1.0
        out[:3, :3] = R
        out[:3, 3] = C + R[:, 0] * step_t * sign
        return out
    if action in ("turn_left", "turn_right"):
        out[:3, :3] = heading_turn(step_r_deg, action) @ R
        out[:3, 3] = C
        return out
    raise ValueError(f"unknown action {action!r}")


def state_key(c2w: np.ndarray) -> Tuple[int, int, int]:
    """Quantised (x, y, yaw) key used for de-duplication."""
    c2w = np.asarray(c2w, dtype=float)
    x, y = float(c2w[0, 3]), float(c2w[1, 3])
    yaw_deg = math.degrees(yaw_of(c2w)) % 360.0
    return (
        int(round(x / POS_QUANT_M)),
        int(round(y / POS_QUANT_M)),
        int(round(yaw_deg / YAW_QUANT_DEG)) % int(round(360.0 / YAW_QUANT_DEG)),
    )


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Obj:
    """One labels.json instance in world coordinates (Z up)."""

    id: str
    label: str
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    room_index: Optional[int] = None
    # Filled by referring.py; None means not (yet) uniquely describable.
    ref: Optional[str] = None
    ref_method: Optional[str] = None

    @property
    def center(self) -> np.ndarray:
        return (self.aabb_min + self.aabb_max) / 2.0

    @property
    def dims(self) -> np.ndarray:
        return self.aabb_max - self.aabb_min

    def target_object_entry(self) -> Dict[str, Any]:
        """Schema consumed by visual_bbox_metrics._extract_objects."""
        return {
            "id": self.id,
            "label": self.label,
            "bbox_min": [float(v) for v in self.aabb_min],
            "bbox_max": [float(v) for v in self.aabb_max],
            "center": [float(v) for v in self.center],
            "ref": self.ref,
        }


# --------------------------------------------------------------------------
# Funnel accounting: every rejection must be counted with a reason.
# --------------------------------------------------------------------------
class Funnel:
    """Counter of (stage, reason) -> count plus per-stage pass counts."""

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, int]] = {}

    def add(self, stage: str, reason: str, n: int = 1) -> None:
        self.counts.setdefault(stage, {})
        self.counts[stage][reason] = self.counts[stage].get(reason, 0) + int(n)

    def merge(self, other: "Funnel") -> None:
        for stage, reasons in other.counts.items():
            for reason, n in reasons.items():
                self.add(stage, reason, n)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(sorted(v.items())) for k, v in sorted(self.counts.items())}


class WorkMeter:
    """Deterministic count of the expensive operations of one scene's generation.

    Time budgets used to be wall-clock seconds, so a busy machine changed the
    output.  A budget in work units (operation counts times fixed per-operation
    weights, ``units``) gives the same result for the same input on any machine.
    One meter per process (``WORK``); ``run_scene`` resets it at the start of a scene.
    """

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}

    def add(self, kind: str, n: int = 1) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + int(n)

    def reset(self) -> None:
        self.counts.clear()

    def snapshot(self) -> Dict[str, int]:
        return dict(sorted(self.counts.items()))

    def units(self, weights: Dict[str, float]) -> float:
        return float(sum(float(weights.get(k, 0.0)) * v for k, v in self.counts.items()))


WORK = WorkMeter()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=_json_default).encode()).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


# --------------------------------------------------------------------------
# Task row schema (one row = one episode = one RL task; SFT/QA derive from it)
# --------------------------------------------------------------------------
TASK_ROW_REQUIRED_KEYS = (
    "task_id",            # stable hash of the episode definition
    "scene_id",
    "task_type",          # e.g. projective_relations, fov_inclusion, absolute_positioning ...
    "task_description",   # natural language, uses Obj.ref text for every target
    "target_object",      # {"objects": [Obj.target_object_entry(), ...]}
    "target_region",      # {"type": ..., "params": {...}} family-specific, read by the canonical metric
    "init_camera",        # {"intrinsics": NATIVE_K, "extrinsics": c2w(4x4 list)}
    "canonical_task_metric_version",
    "camera_model_version",
    "action_protocol_version",
    "difficulty",         # {"d_star": int, "lower_bound_complete": bool, "certificate": [actions]}
    "initial_observability",  # per-target geometric pre-check at the initial pose
    "generator_version",
)
