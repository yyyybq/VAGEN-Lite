# Active Spatial Intelligence Environment Utilities
# Contains camera manipulation, pose scoring, and action parsing utilities.

import json
import re
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from dataclasses import dataclass
from PIL import Image
from scipy.spatial.transform import Rotation as R


# ========== IO Helpers ==========

def count_lines(jsonl_path: Path) -> int:
    """Count the number of lines in a JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_jsonl_line_by_index(jsonl_path: Path, idx: int) -> Dict[str, Any]:
    """Read a specific line from a JSONL file by index."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {jsonl_path}")


def resolve_rel_image(jsonl_path: Path, rel: str, dataset_root: str = None) -> Path:
    """Resolve a relative image path."""
    if dataset_root is not None:
        base = Path(dataset_root)
    else:
        base = jsonl_path.parent
    if rel.startswith("./"):
        rel = rel[2:]
    return base / rel


def safe_open_rgb(path: Path) -> Optional[Image.Image]:
    """Safely open an image and convert to RGB."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ========== Action Parsing ==========

@dataclass
class ParsedAction:
    """Represents a parsed action with name and optional argument."""
    name: str
    arg: Optional[str]


def parse_free_think(response: str) -> Dict[str, Any]:
    """
    Parse the free_think format response.
    Expected format: <think>...</think><action>action1|action2|...|</action>

    Also accepts common model output variants:
      - action>...< /action> (missing leading <)
      - <think>...</think>\naction_name| (bare action without <action> tags)

    Returns:
        Dict with 'ok', 'think', and 'actions_blob' keys
    """
    # Preferred: <think>...</think> ... <action>...</action>
    m = re.search(r"<think>(.*?)</think>\s*<?\s*action\s*>(.*?)<\s*/\s*action\s*>",
                  response, re.DOTALL | re.IGNORECASE)
    if m:
        blob = m.group(2).strip()
        if "|" in blob:
            blob = blob[: blob.rfind("|") + 1]
        return {"ok": True, "think": m.group(1).strip(), "actions_blob": blob}

    # No think block: <action>...</action> or action>...</action>
    m = re.search(r"<?\s*action\s*>(.*?)<\s*/\s*action\s*>", response, re.DOTALL | re.IGNORECASE)
    if m:
        blob = m.group(1).strip()
        if "|" in blob:
            blob = blob[: blob.rfind("|") + 1]
        return {"ok": True, "think": "", "actions_blob": blob}

    # Fallback: bare actions after </think> (model omits <action> wrapper)
    # Matches: <think>...</think>\naction_name|  or  <think>...</think>\nact1|act2|
    think_m = re.search(r"<think>(.*?)</think>\s*\n?\s*(.+)", response, re.DOTALL | re.IGNORECASE)
    if think_m:
        remainder = think_m.group(2).strip()
        # Strip EOS / special tokens like <|im_end|>, <|endoftext|>, etc.
        remainder = re.sub(r"<\|[^>]*\|?>.*$", "", remainder).strip()
        # Also strip any trailing XML-like tags
        remainder = re.sub(r"<[^>]+>.*$", "", remainder).strip()
        # Strip common action label prefixes like "action: " or "动作 "
        remainder = re.sub(r"^(?:action\s*:\s*|动作\s*)", "", remainder, flags=re.IGNORECASE)
        # Truncate to last complete pipe-separated token (handles 512-token response truncation)
        if "|" in remainder:
            remainder = remainder[: remainder.rfind("|") + 1]
        # Accept only if non-empty and looks like action tokens (word chars, |, parens)
        if remainder and re.match(r"^[A-Za-z_][\w|() ]*$", remainder):
            return {"ok": True, "think": think_m.group(1).strip(), "actions_blob": remainder}

    return {"ok": False, "think": "", "actions_blob": ""}


def _normalize_ws(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", s.strip())


def parse_actions(actions_blob: str,
                  sep: str = "|",
                  allow_trailing_empty: bool = True) -> Tuple[bool, List[ParsedAction]]:
    """
    Parse action string like 'action1|action2|...|' into list of ParsedAction.
    
    Returns:
        Tuple of (format_ok, actions_list)
    """
    tokens = [t for t in (x.strip() for x in actions_blob.split(sep)) if t or not allow_trailing_empty]
    actions: List[ParsedAction] = []
    
    for tok in tokens:
        # Match action(arg) format
        m = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", tok)
        if m:
            actions.append(ParsedAction(name=m.group(1).lower(), arg=_normalize_ws(m.group(2))))
            continue
        # Match simple action name
        m = re.fullmatch(r"([A-Za-z_]\w*)", tok)
        if m:
            actions.append(ParsedAction(name=m.group(1).lower(), arg=None))
            continue
        # Invalid token
        return False, []
    
    return True, actions


# Allowed actions for active spatial navigation
ALLOWED_ACTIONS = {
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "done",
}

# Predefined action-space presets selectable via env_config.action_space
ACTION_SPACE_PRESETS = {
    "legacy": {
        "move_forward", "move_backward",
        "turn_left", "turn_right",
        "look_up", "look_down",
        "done",
    },
    "strafe": {  # v17: add lateral motion, drop pitch (look_up/down)
        "move_forward", "move_backward",
        "move_left", "move_right",
        "turn_left", "turn_right",
        "done",
    },
}


def check_actions(actions: List[ParsedAction], allowed: Optional[set] = None) -> bool:
    """
    Validate that all actions are allowed.
    'done' must be the last action if present.

    Args:
        actions: parsed actions
        allowed: optional override of allowed action names. Defaults to
                 ALLOWED_ACTIONS (full legacy set).
    """
    allow = allowed if allowed is not None else ALLOWED_ACTIONS
    for i, a in enumerate(actions):
        if a.name not in allow:
            return False
        if a.name == "done" and i != len(actions) - 1:
            return False
    return True


# ========== Pose Utilities ==========

def deg2rad(x: float) -> float:
    """Convert degrees to radians."""
    return x * math.pi / 180.0


def rad2deg(x: float) -> float:
    """Convert radians to degrees."""
    return x * 180.0 / math.pi


def check_4x4(M: np.ndarray, name: str = "matrix"):
    """Validate a (4,4) homogeneous matrix."""
    if not isinstance(M, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(M)}")
    if M.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {M.shape}")


def extrinsic_c2w_to_w2c(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert camera-to-world matrix to world-to-camera matrix."""
    check_4x4(camera_to_world, name="camera_to_world")
    return np.linalg.inv(camera_to_world)


def extrinsic_w2c_to_c2w(world_to_camera: np.ndarray) -> np.ndarray:
    """Convert world-to-camera matrix to camera-to-world matrix."""
    check_4x4(world_to_camera, name="world_to_camera")
    return np.linalg.inv(world_to_camera)


def c2w_extrinsic_to_se3(E: np.ndarray) -> List[float]:
    """
    Convert 4x4 camera-to-world extrinsic to [tx, ty, tz, rx, ry, rz] with angles in degrees.
    """
    check_4x4(E, name="extrinsic")
    t = E[:3, 3]
    rot_matrix = E[:3, :3]
    euler_rad = R.from_matrix(rot_matrix).as_euler("xyz", degrees=False)
    euler_deg = [rad2deg(e) for e in euler_rad]
    return [float(t[0]), float(t[1]), float(t[2]),
            float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])]


def c2w_se3_to_extrinsic(pose6_deg: Sequence[float]) -> np.ndarray:
    """
    Convert [tx, ty, tz, rx, ry, rz] in degrees to 4x4 camera-to-world extrinsic.
    """
    if len(pose6_deg) != 6:
        raise ValueError("6-DoF requires 6 values.")
    tx, ty, tz, rx_deg, ry_deg, rz_deg = [float(x) for x in pose6_deg]
    E = np.eye(4, dtype=np.float64)
    rx, ry, rz = deg2rad(rx_deg), deg2rad(ry_deg), deg2rad(rz_deg)
    E[:3, :3] = R.from_euler("xyz", [rx, ry, rz], degrees=False).as_matrix()
    E[:3, 3] = [tx, ty, tz]
    return E


def format_pose6_deg(p: List[float]) -> str:
    """Format 6-DoF pose for display."""
    tx, ty, tz, rx, ry, rz = p
    return f"[tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}, rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°]"


# ========== View Manipulator ==========

class ViewManipulator:
    """
    Camera pose controller for 3D environments.
    Maintains camera-to-world (c2w) transformation.
    """
    
    def __init__(
        self,
        step_translation: float = 0.1,
        step_rotation_deg: float = 5.0,
        world_up_axis: str = "Z",
        image_y_down: bool = True,
    ):
        """
        Initialize the view manipulator.
        
        Args:
            step_translation: Translation step size in world units
            step_rotation_deg: Rotation step size in degrees
            world_up_axis: 'Z' or 'Y' depending on coordinate system
            image_y_down: Whether image Y axis points down
        """
        self.step_t = float(step_translation)
        self.step_r_deg = float(step_rotation_deg)
        self.step_r = np.radians(self.step_r_deg)
        self.up_axis = world_up_axis.upper()
        assert self.up_axis in ("Z", "Y"), "world_up_axis must be 'Z' or 'Y'"
        self.image_y_down = bool(image_y_down)
        
        # Camera-to-world transformation
        self.c2w = np.eye(4, dtype=np.float64)
    
    def reset(self, initial_extrinsic_c2w: np.ndarray = None) -> np.ndarray:
        """Reset to identity or provided camera-to-world matrix."""
        if initial_extrinsic_c2w is None:
            self.c2w = np.eye(4, dtype=np.float64)
        else:
            check_4x4(initial_extrinsic_c2w)
            self.c2w = initial_extrinsic_c2w.astype(np.float64)
        return self.get_pose()
    
    def get_pose(self, mode: str = "c2w") -> np.ndarray:
        """Get current camera pose."""
        if mode == "c2w":
            return self.c2w.copy()
        elif mode == "w2c":
            return extrinsic_c2w_to_w2c(self.c2w)
        else:
            raise ValueError("mode must be 'c2w' or 'w2c'")
    
    def _Rc_t_from_c2w(self, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract rotation and translation from c2w matrix."""
        return c2w[:3, :3].copy(), c2w[:3, 3].copy()
    
    def _compose_c2w(self, R_c2w: np.ndarray, C_world: np.ndarray) -> np.ndarray:
        """Compose c2w matrix from rotation and camera center."""
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = C_world
        return c2w
    
    def _translate_camera_center(self, C_world: np.ndarray, R_c2w: np.ndarray, delta: np.ndarray):
        """Translate camera center by delta."""
        self.c2w = self._compose_c2w(R_c2w, C_world + delta)
    
    def step(self, action: str) -> np.ndarray:
        """Execute a discrete movement action."""
        a = action.strip().lower()
        if a == "move_forward" or a == "w":
            self.move_forward(+self.step_t)
        elif a == "move_backward" or a == "s":
            self.move_forward(-self.step_t)
        elif a == "move_right" or a == "d":
            self.move_right(+self.step_t)
        elif a == "move_left" or a == "a":
            self.move_right(-self.step_t)
        elif a == "turn_left" or a == "q":
            self.yaw_camera(-self.step_r)
        elif a == "turn_right" or a == "e":
            self.yaw_camera(+self.step_r)
        elif a == "look_up" or a == "r":
            ang = (+self.step_r) if self.image_y_down else (-self.step_r)
            self.pitch_camera(ang)
        elif a == "look_down" or a == "f":
            ang = (-self.step_r) if self.image_y_down else (+self.step_r)
            self.pitch_camera(ang)
        else:
            pass  # Unknown action, do nothing
        return self.get_pose()
    
    def move_forward(self, distance: float):
        """Move camera forward along its view direction."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        dir_world = R_c2w @ np.array([0.0, 0.0, 1.0])
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)

    def move_right(self, distance: float):
        """Strafe camera along its local +X (right) axis. Negative distance = left."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        dir_world = R_c2w @ np.array([1.0, 0.0, 0.0])
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)
    
    def yaw_camera(self, angle_rad: float):
        """Rotate camera around its local Y axis."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("y", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        self.c2w = self._compose_c2w(R_new, C_world)
    
    def pitch_camera(self, angle_rad: float):
        """Rotate camera around its local X axis."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("x", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        self.c2w = self._compose_c2w(R_new, C_world)



# ========== Pose Distance (aligned with ActiveVLN) ==========

def compute_translation_distance(
    current_pose: Sequence[float],
    target_pose: Sequence[float],
) -> float:
    """
    Compute only translation (Euclidean) distance between two poses.
    Aligned with ActiveVLN's pose_utils.py
    
    Args:
        current_pose: Current camera pose [tx, ty, tz, ...]
        target_pose: Target camera pose [tx, ty, tz, ...]
        
    Returns:
        Translation distance in meters
    """
    dx = current_pose[0] - target_pose[0]
    dy = current_pose[1] - target_pose[1]
    dz = current_pose[2] - target_pose[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def is_goal_reached(
    current_pose: Sequence[float],
    target_pose: Sequence[float],
    success_distance: float = 1.0,
) -> bool:
    """
    Check if agent has reached the goal pose.
    Similar to VLNCE's "reached goal within 1m" reward check.
    Aligned with ActiveVLN's pose_utils.py
    
    Args:
        current_pose: Current camera pose [tx, ty, tz, rx, ry, rz]
        target_pose: Target camera pose [tx, ty, tz, rx, ry, rz]
        success_distance: Distance threshold for success (meters)
        
    Returns:
        True if goal is reached
    """
    trans_dist = compute_translation_distance(current_pose, target_pose)
    return trans_dist <= success_distance


def compute_approach_reward(
    current_pose: Sequence[float],
    target_pose: Sequence[float],
    success_distance: float = 1.0,
    max_distance: float = 5.0,
    reward_base: float = 1.0,
    reward_shaping: str = "weighted",
) -> float:
    """
    Compute approach/navigation reward based on distance to target.
    Aligned with ActiveVLN's pose_utils.py
    
    Reward schemes:
    - "binary": Full reward if within success_distance, 0 otherwise
    - "linear": Linear interpolation from 0 at max_distance to reward_base at 0
    - "weighted": reward_base * (1 - distance / max_distance), similar to VLNCE
    - "exponential": Exponential decay with distance
    
    Args:
        current_pose: Current camera pose
        target_pose: Target camera pose
        success_distance: Distance for full success
        max_distance: Distance at which reward becomes 0
        reward_base: Maximum reward value
        reward_shaping: Reward shaping strategy
        
    Returns:
        Reward value
    """
    distance = compute_translation_distance(current_pose, target_pose)
    
    if reward_shaping == "binary":
        return reward_base if distance <= success_distance else 0.0
    
    elif reward_shaping == "linear":
        if distance <= success_distance:
            return reward_base
        elif distance >= max_distance:
            return 0.0
        else:
            # Linear interpolation
            return reward_base * (1 - (distance - success_distance) / (max_distance - success_distance))
    
    elif reward_shaping == "weighted":
        # Similar to VLNCE's weighted success reward
        if distance >= max_distance:
            return 0.0
        return reward_base * (1 - min(distance / max_distance, 1.0))
    
    elif reward_shaping == "exponential":
        # Exponential decay
        decay_rate = 1.0 / max_distance
        return reward_base * math.exp(-decay_rate * distance)
    
    else:
        raise ValueError(f"Unknown reward shaping: {reward_shaping}")


def compute_progress_reward(
    current_distance: float,
    previous_distance: float,
    success_distance: float = 1.0,
    max_distance: float = 5.0,
    reward_scale: float = 1.0,
    reward_mode: str = "delta",
) -> float:
    """
    Compute per-step progress reward based on distance improvement.
    
    This function rewards the agent for getting closer to the target at each step.
    The closer the agent gets, the higher the cumulative reward.
    
    Reward modes:
    - "delta": Reward = (prev_distance - curr_distance) * scale
      Positive when approaching, negative when moving away
      
    - "delta_normalized": Reward = (prev_distance - curr_distance) / max_distance * scale
      Normalized to [−scale, +scale] range
      
    - "potential": Potential-based reward shaping (Ng et al., 1999)
      Reward = gamma * Φ(s') - Φ(s), where Φ(s) = -distance
      Ensures optimal policy is preserved
      
    - "scaled_delta": Larger reward when closer to goal
      Reward = delta * (1 + (max_distance - curr_distance) / max_distance)
      
    Args:
        current_distance: Current distance to target (meters)
        previous_distance: Previous distance to target (meters)
        success_distance: Distance threshold for success (meters)
        max_distance: Maximum distance for normalization
        reward_scale: Scale factor for reward
        reward_mode: Reward computation mode
        
    Returns:
        Progress reward value (can be positive or negative)
    """
    delta = previous_distance - current_distance  # Positive = getting closer
    
    if reward_mode == "delta":
        # Simple delta reward
        return delta * reward_scale
    
    elif reward_mode == "delta_normalized":
        # Normalized delta reward
        return (delta / max_distance) * reward_scale
    
    elif reward_mode == "potential":
        # Potential-based shaping: Φ(s) = -distance
        # Reward = γ * Φ(s') - Φ(s) = γ * (-d') - (-d) = d - γ*d'
        # With γ = 1: Reward = d - d' = -(d' - d) = -delta... wait, that's wrong
        # Actually: Reward = Φ(s') - Φ(s) = -d' - (-d) = d - d' = delta
        gamma = 0.99
        return (gamma * (-current_distance) - (-previous_distance)) * reward_scale
    
    elif reward_mode == "scaled_delta":
        # Scale delta by proximity: closer = bigger reward for same delta
        proximity_factor = 1.0 + (max_distance - current_distance) / max_distance
        proximity_factor = max(0.5, min(2.0, proximity_factor))  # Clamp to [0.5, 2.0]
        return delta * proximity_factor * reward_scale
    
    else:
        raise ValueError(f"Unknown reward mode: {reward_mode}")


def compute_distance_reward(
    current_distance: float,
    success_distance: float = 1.0,
    max_distance: float = 5.0,
    reward_base: float = 1.0,
) -> float:
    """
    Compute reward based on current distance to target.
    Closer distance = higher reward.
    
    This is a simpler version that just returns a reward proportional to closeness.
    
    Args:
        current_distance: Current distance to target (meters)
        success_distance: Distance threshold for full reward
        max_distance: Distance at which reward becomes 0
        reward_base: Maximum reward value
        
    Returns:
        Reward value in [0, reward_base]
    """
    if current_distance <= success_distance:
        return reward_base
    elif current_distance >= max_distance:
        return 0.0
    else:
        # Linear interpolation based on distance
        return reward_base * (1 - (current_distance - success_distance) / (max_distance - success_distance))


# ========== Pose Scoring ==========

def euler_to_quaternion(euler: Sequence[float]) -> np.ndarray:
    """Convert Euler angles (yaw, pitch, roll) to quaternion."""
    yaw, pitch, roll = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([w, x, y, z])


def calculate_angle_difference(orient1, orient2) -> float:
    """Calculate angle difference between two orientations."""
    try:
        if len(orient1) == 3:
            orient1 = euler_to_quaternion(orient1)
        if len(orient2) == 3:
            orient2 = euler_to_quaternion(orient2)
        
        dot_product = abs(np.dot(orient1, orient2))
        dot_product = min(1.0, dot_product)
        angle_diff = 2 * np.arccos(dot_product)
        
        return min(angle_diff, 2 * np.pi - angle_diff)
    except:
        if len(orient1) == 3 and len(orient2) == 3:
            orient1_norm = orient1 / (np.linalg.norm(orient1) + 1e-8)
            orient2_norm = orient2 / (np.linalg.norm(orient2) + 1e-8)
            dot_product = np.clip(np.dot(orient1_norm, orient2_norm), -1.0, 1.0)
            return np.arccos(abs(dot_product))
        return np.pi


def calculate_pose_score_smooth(
    current_pos: np.ndarray,
    current_orient: np.ndarray,
    target_pos: np.ndarray,
    target_orient: np.ndarray,
    transition_distance: float = 2.0,
    max_distance: float = 5.0
) -> Tuple[float, float, float, float, float]:
    """
    Calculate smooth pose score with adaptive weights.
    
    When far: prioritize position
    When close: prioritize orientation
    
    Returns:
        Tuple of (final_score, pos_score, orient_score, pos_weight, orient_weight)
    """
    # Position score
    pos_distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    pos_score = max(0, 1 - pos_distance / max_distance)
    
    # Orientation score
    angle_diff = calculate_angle_difference(current_orient, target_orient)
    orient_score = max(0, 1 - angle_diff / np.pi)
    
    # Smooth weight transition using sigmoid
    x = (pos_distance - transition_distance) / transition_distance
    sigmoid = 1 / (1 + np.exp(-3 * x))
    
    pos_weight = 0.2 + 0.6 * sigmoid      # Range [0.2, 0.8]
    orient_weight = 0.8 - 0.6 * sigmoid   # Range [0.8, 0.2]
    
    final_score = pos_weight * pos_score + orient_weight * orient_score
    
    return final_score, pos_score, orient_score, pos_weight, orient_weight


# ========== Fallback Camera Parameters ==========

def fallback_K() -> np.ndarray:
    """Generate fallback camera intrinsics."""
    fx = fy = 300.0
    cx = cy = 150.0
    K4 = np.array([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K4
