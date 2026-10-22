"""Data-path categories (task spec 7.2) and the policy-observation whitelist.

Every field that enters or leaves the system belongs to exactly one category:

  POLICY_OBSERVATION   visible to the running system: RGB, depth, proprioception, the camera poses
                       relative to the base that the official evaluator sends to every policy, the
                       task id, and the BDDL task definition / task prompt (offline task metadata).
  TRAINING_ANNOTATION  may use simulator ground truth; only for training data and labels.
  EVALUATION_RECORD    official per-episode results (metrics JSON, videos); only read after the
                       episode by eval/ and diagnostics/.

The whitelist below is the single place that decides which evaluator keys modules may see.
Anything not matched is dropped (and its key name logged) before an Observation is built.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Mapping

import numpy as np

from agent.interfaces import Observation

POLICY_OBSERVATION = "policy_observation"
TRAINING_ANNOTATION = "training_annotation"
EVALUATION_RECORD = "evaluation_record"

# Flattened evaluator keys look like "robot_r1::robot_r1:zed_link:Camera:0::rgb"
# (OmniGibson/omnigibson/eval/utils/eval_utils.py flatten_obs_dict + evaluator._preprocess_obs).
_CAMERA_KEY = re.compile(r"^(?P<robot>[^:]+)::(?P=robot):(?P<link>[A-Za-z0-9_]+):Camera:0::(?P<mod>rgb|depth_linear)$")
_PROPRIO_KEY = re.compile(r"^(?P<robot>[^:]+)::proprio$")
_CAM_POSES_KEY = re.compile(r"^(?P<robot>[^:]+)::cam_rel_poses$")
_TASK_ID_KEY = "task_id"

# Camera link -> eval camera role (OmniGibson/omnigibson/eval/r1pro.yaml eval.camera_sensor_names).
DEFAULT_CAMERA_ROLES: Mapping[str, str] = {
    "zed_link": "head",
    "left_realsense_link": "left_wrist",
    "right_realsense_link": "right_wrist",
}

# R1Pro proprioception layout (eval_utils.PROPRIOCEPTION_INDICES["R1Pro"]), 61 dims.
PROPRIO_SLICES: Mapping[str, slice] = OrderedDict(
    base_qvel=slice(0, 3),
    arm_left_qpos=slice(3, 10),
    arm_left_qvel=slice(10, 17),
    eef_left_pos=slice(17, 20),
    eef_left_quat=slice(20, 24),
    gripper_left_qpos=slice(24, 26),
    gripper_left_qvel=slice(26, 28),
    arm_right_qpos=slice(28, 35),
    arm_right_qvel=slice(35, 42),
    eef_right_pos=slice(42, 45),
    eef_right_quat=slice(45, 49),
    gripper_right_qpos=slice(49, 51),
    gripper_right_qvel=slice(51, 53),
    trunk_qpos=slice(53, 57),
    trunk_qvel=slice(57, 61),
)
PROPRIO_DIM = 61

# R1Pro action layout (eval_utils.ACTION_QPOS_INDICES["R1Pro"]) and units from eval/r1pro.yaml:
#   base: HolonomicBaseJointController velocity, input [-1, 1] -> output [-0.75, 0.75] m/s (x, y),
#         [-1, 1] rad/s (yaw); robot-local frame
#   torso / arms: JointController absolute joint positions (use_delta_commands: false), radians
#   grippers: MultiFingerGripperController "smooth", input [-1, 1]; -1 = closed, +1 = open
ACTION_SLICES: Mapping[str, slice] = OrderedDict(
    base=slice(0, 3),
    torso=slice(3, 7),
    left_arm=slice(7, 14),
    left_gripper=slice(14, 15),
    right_arm=slice(15, 22),
    right_gripper=slice(22, 23),
)

# Field-category registry for everything the system logs. diagnostics/ adds TRAINING_ANNOTATION
# fields; tests/test_truth_isolation.py checks that runtime code only consumes POLICY_OBSERVATION.
FIELD_CATEGORIES: Mapping[str, str] = {
    "obs.rgb": POLICY_OBSERVATION,
    "obs.depth": POLICY_OBSERVATION,
    "obs.proprio": POLICY_OBSERVATION,
    "obs.cam_rel_poses": POLICY_OBSERVATION,
    "obs.task_id": POLICY_OBSERVATION,
    "task.bddl_definition": POLICY_OBSERVATION,
    "task.instruction": POLICY_OBSERVATION,
    "sim.object_poses": TRAINING_ANNOTATION,
    "sim.robot_global_pose": TRAINING_ANNOTATION,
    "sim.segmentation": TRAINING_ANNOTATION,
    "sim.object_states": TRAINING_ANNOTATION,
    "sim.predicate_truth": TRAINING_ANNOTATION,
    "sim.scene_map": TRAINING_ANNOTATION,
    "eval.metrics_json": EVALUATION_RECORD,
    "eval.q_score": EVALUATION_RECORD,
    "eval.video": EVALUATION_RECORD,
}


def classify_key(key: str) -> str | None:
    """Return the whitelist kind of an evaluator key ("camera" | "proprio" | "cam_rel_poses" |
    "task_id"), or None when the key is not a legal policy observation."""
    if key == _TASK_ID_KEY:
        return "task_id"
    if _CAMERA_KEY.match(key):
        return "camera"
    if _PROPRIO_KEY.match(key):
        return "proprio"
    if _CAM_POSES_KEY.match(key):
        return "cam_rel_poses"
    return None


def whitelist(obs: Mapping[str, object]) -> tuple[dict, tuple[str, ...]]:
    kept, dropped = {}, []
    for key, value in obs.items():
        if isinstance(key, bytes):
            key = key.decode()
        if classify_key(key) is None:
            dropped.append(key)
        else:
            kept[key] = value
    return kept, tuple(sorted(dropped))


def _readonly(x) -> np.ndarray:
    """A non-writable VIEW: modules cannot change the data, and the caller's own arrays keep their flags."""
    v = np.asarray(x).view()
    v.flags.writeable = False
    return v


def _squeeze_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x[0] if x.ndim >= 1 and x.shape[0] == 1 else x


def build_observation(
    env_obs: Mapping[str, object],
    *,
    step: int,
    received_at: float,
    camera_roles: Mapping[str, str] = DEFAULT_CAMERA_ROLES,
) -> Observation:
    """Build an Observation for ONE environment from its slice of the evaluator dict.

    ``env_obs`` values must keep a leading batch dim of 1 (as the evaluator sends for one env).
    """
    kept, dropped = whitelist(env_obs)
    rgb, depth = {}, {}
    proprio = None
    cam_rel_poses = None
    task_id = -1
    for key, value in kept.items():
        arr = np.asarray(value)
        m = _CAMERA_KEY.match(key)
        if m:
            role = camera_roles.get(m.group("link"), m.group("link"))
            if m.group("mod") == "rgb":
                img = _squeeze_batch(arr)
                rgb[role] = _readonly(img[..., :3] if img.ndim == 3 else img)
            else:
                depth[role] = _readonly(_squeeze_batch(arr).astype(np.float32, copy=False))
            continue
        if _PROPRIO_KEY.match(key):
            proprio = _readonly(_squeeze_batch(arr).astype(np.float32, copy=False).reshape(-1))
        elif _CAM_POSES_KEY.match(key):
            cam_rel_poses = _readonly(_squeeze_batch(arr))
        elif key == _TASK_ID_KEY:
            task_id = int(np.asarray(arr).reshape(-1)[0])
    if proprio is None:
        raise KeyError(f"observation has no '<robot>::proprio' key; got {sorted(kept)}")
    return Observation(
        step=step,
        received_at=received_at,
        task_id=task_id,
        rgb=rgb,
        depth=depth,
        proprio=proprio,
        cam_rel_poses=cam_rel_poses,
        # arrays become read-only views (same bytes, so L0 forwarding stays byte-identical); scalars are immutable
        policy_dict={k: (_readonly(v) if isinstance(v, np.ndarray) else v) for k, v in kept.items()},
        dropped_keys=dropped,
    )


def proprio_part(proprio: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(proprio)[..., PROPRIO_SLICES[name]]
