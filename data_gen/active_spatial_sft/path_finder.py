"""
Path Finder for Active Spatial Navigation SFT Data Generation.

Uses the SpatialPotentialField (same scoring system as the RL environment) to
find optimal camera trajectories from an initial pose to a target region.

Algorithm (beam_width=1 → greedy, beam_width>1 → beam search)
--------------------------------------------------------------
1. Start from init_c2w.
2. At each "LLM turn", find up to max_actions_per_turn individual actions:
   - Greedy mode (beam_width=1): pick the single best action at each step,
     stop when no action improves by at least min_improvement.
   - Beam search mode (beam_width>1): maintain a beam of top-K states; this
     tolerates short-term score drops to find better long-term paths.
3. If no sequence of actions improves the score (plateau), attempt escape
   strategies. If still stuck after plateau_tolerance turns, stop.
4. Terminate early if total_score >= success_threshold.

Adaptive min_improvement
------------------------
When adaptive_min_improvement=True, the improvement threshold is scaled down
as the score approaches success_threshold. This prevents premature termination
in the final approach where remaining headroom is small.

effective_min_improvement = max(MIN_FLOOR, min_improvement * remaining_headroom_ratio)

Key invariant
-------------
All camera manipulations here mirror ViewManipulator.step() *exactly* so that
the images rendered by the environment match the poses computed here.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R_scipy

# ---------------------------------------------------------------------------
# Lazy import guard – allow the module to be imported before VAGEN is on sys.path
# ---------------------------------------------------------------------------
import sys
import os
_VAGEN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _VAGEN_ROOT not in sys.path:
    sys.path.insert(0, _VAGEN_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOVEMENT_ACTIONS: List[str] = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
]

# Actions that only change orientation (useful for "escape from plateau")
ROTATION_ACTIONS: List[str] = ["turn_left", "turn_right", "look_up", "look_down"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrajStep:
    """One LLM turn in a trajectory.

    A turn bundles up to max_actions_per_turn individual actions executed
    sequentially.  The image shown to the model BEFORE the turn corresponds to
    c2w_before; the image shown AFTER (next turn's input) corresponds to
    c2w_after.
    """
    step_idx: int
    c2w_before: np.ndarray          # 4x4 camera-to-world before actions
    actions: List[str]              # Individual actions taken this turn
    c2w_after: np.ndarray           # 4x4 camera-to-world after actions
    score_before: float             # Total potential field score before
    score_after: float              # Total potential field score after
    pos_score_before: float         # Position component score before
    pos_score_after: float          # Position component score after
    ori_score_before: float         # Orientation component score before
    ori_score_after: float          # Orientation component score after


@dataclass
class Trajectory:
    """Complete optimal trajectory from initial to target pose."""
    steps: List[TrajStep]           # All LLM turns (not including the terminal "done" turn)
    final_c2w: np.ndarray           # Camera pose after the last action
    initial_score: float            # Score at the very beginning
    final_score: float              # Score after the last action
    success: bool                   # True if final_score >= success_threshold
    total_actions: int              # Total individual actions taken
    scene_id: str = ""
    item_idx: int = -1


# ---------------------------------------------------------------------------
# Core geometry helpers (mirror ViewManipulator exactly)
# ---------------------------------------------------------------------------

def simulate_action(
    c2w: np.ndarray,
    action: str,
    step_translation: float = 0.3,
    step_rotation_deg: float = 30.0,
    image_y_down: bool = True,
) -> np.ndarray:
    """Apply a navigation action to a c2w matrix, returning a NEW matrix.

    This is a pure function that mirrors ViewManipulator.step() exactly.
    The original c2w is never modified.

    Args:
        c2w: 4x4 camera-to-world matrix.
        action: One of MOVEMENT_ACTIONS.
        step_translation: Translation step in world units (metres).
        step_rotation_deg: Rotation step in degrees.
        image_y_down: Whether image Y axis points down (matches ViewManipulator default).

    Returns:
        New 4x4 c2w matrix after the action.
    """
    c2w = c2w.copy()
    R_c2w = c2w[:3, :3].copy()
    C_world = c2w[:3, 3].copy()
    step_r = np.radians(step_rotation_deg)

    if action == "move_forward":
        # Forward = +Z column of c2w (ViewManipulator convention)
        dir_world = R_c2w @ np.array([0.0, 0.0, 1.0])
        c2w[:3, 3] = C_world + dir_world * step_translation

    elif action == "move_backward":
        dir_world = R_c2w @ np.array([0.0, 0.0, 1.0])
        c2w[:3, 3] = C_world - dir_world * step_translation

    elif action == "turn_left":
        # Yaw around local Y axis (negative = left)
        R_local = R_scipy.from_euler("y", -step_r, degrees=False).as_matrix()
        c2w[:3, :3] = R_c2w @ R_local

    elif action == "turn_right":
        R_local = R_scipy.from_euler("y", +step_r, degrees=False).as_matrix()
        c2w[:3, :3] = R_c2w @ R_local

    elif action == "look_up":
        ang = (+step_r) if image_y_down else (-step_r)
        R_local = R_scipy.from_euler("x", ang, degrees=False).as_matrix()
        c2w[:3, :3] = R_c2w @ R_local

    elif action == "look_down":
        ang = (-step_r) if image_y_down else (+step_r)
        R_local = R_scipy.from_euler("x", ang, degrees=False).as_matrix()
        c2w[:3, :3] = R_c2w @ R_local

    return c2w


def get_camera_pos_and_forward(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract position and forward vector from a c2w matrix.

    Convention (matches env.py):
        camera_forward = c2w[:3, 2]   (local positive Z column)
    """
    cam_pos = c2w[:3, 3].copy()
    cam_forward = c2w[:3, 2].copy()
    return cam_pos, cam_forward


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_LOOKAT_TIEBREAKER_WEIGHT = 0.03  # SFT-only tie-breaker; not part of env reward


def score_c2w(
    c2w: np.ndarray,
    potential_field,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
) -> Tuple[float, float, float]:
    """Evaluate the potential field score for a given camera pose.

    AUDIT FIX (v20-prep): adds a small additive "look-at" tie-breaker
    (+_LOOKAT_TIEBREAKER_WEIGHT if all target objects are in FoV) to the
    returned total_score. This biases the SFT path-finder toward in-FoV
    poses *as a tie-breaker only* — the env reward is unaffected because this
    bonus lives in the SFT-side scoring wrapper, not in
    SpatialPotentialField.compute_score itself.

    The reported pos_score / ori_score are NOT modified, so downstream
    statistics (initial_score / final_score logged in stats.json) stay
    consistent with the env-side definition.

    Returns:
        (total_score_with_lookat_bias, position_score, orientation_score)
    """
    cam_pos, cam_forward = get_camera_pos_and_forward(c2w)
    scoring_params = dict(task_params or {})
    scoring_params["_camera_pose_c2w"] = np.asarray(c2w, dtype=np.float64).tolist()
    result = potential_field.compute_score(
        camera_position=cam_pos,
        camera_forward=cam_forward,
        task_type=task_type,
        task_params=scoring_params,
        target_region=target_region,
    )

    total = float(result.total_score)
    pos = float(result.position_score)
    ori = float(result.orientation_score)

    # Look-at tie-breaker: small bonus when all target objects are in FoV.
    # The flag is set by SpatialPotentialField._combine_scores when target_objects
    # is provided (which is true for every per-task scorer).
    try:
        if result.details.get("all_targets_in_fov", False):
            total = min(1.0, total + _LOOKAT_TIEBREAKER_WEIGHT)
    except AttributeError:
        pass

    return total, pos, ori


# Minimum effective improvement floor (prevents threshold from collapsing to 0)
_MIN_IMPROVEMENT_FLOOR = 0.001


# ---------------------------------------------------------------------------
# Single greedy step
# ---------------------------------------------------------------------------

def _greedy_single_step(
    c2w: np.ndarray,
    potential_field,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
    step_translation: float,
    step_rotation_deg: float,
    collision_checker=None,
) -> Tuple[Optional[str], np.ndarray, float, float, float]:
    """Find the single best action by exhaustive evaluation of all 6 moves.

    Returns:
        (best_action, best_c2w, best_total, best_pos, best_ori)
        best_action is None if no action improves over current pose.
    """
    current_total, current_pos, current_ori = score_c2w(
        c2w, potential_field, task_type, task_params, target_region
    )

    best_action: Optional[str] = None
    best_c2w = c2w
    best_total = current_total
    best_pos = current_pos
    best_ori = current_ori

    for action in MOVEMENT_ACTIONS:
        candidate = simulate_action(c2w, action, step_translation, step_rotation_deg)

        # Collision check
        if collision_checker is not None:
            new_pos = candidate[:3, 3]
            old_pos = c2w[:3, 3]
            try:
                col_result = collision_checker.check_collision(new_pos, old_pos)
                if col_result.has_collision:
                    continue
            except Exception:
                pass  # If collision check fails, allow the move

        cand_total, cand_pos, cand_ori = score_c2w(
            candidate, potential_field, task_type, task_params, target_region
        )

        if cand_total > best_total:
            best_total = cand_total
            best_pos = cand_pos
            best_ori = cand_ori
            best_action = action
            best_c2w = candidate

    return best_action, best_c2w, best_total, best_pos, best_ori


# ---------------------------------------------------------------------------
# Beam search turn
# ---------------------------------------------------------------------------

def _beam_search_turn(
    c2w: np.ndarray,
    potential_field,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
    step_translation: float,
    step_rotation_deg: float,
    max_actions: int,
    beam_width: int,
    collision_checker=None,
) -> Tuple[List[str], np.ndarray, float, float, float]:
    """Find the best action sequence for one LLM turn using beam search.

    Unlike greedy, beam search tolerates short-term score drops (e.g., rotating
    away from a suboptimal position) to discover better long-term paths such as
    "turn 180°, then move forward twice".

    The beam is expanded depth-first up to max_actions steps. At each depth all
    6 actions are tried from every beam state; the top beam_width unique next
    states (by total score) are kept.  The path that reaches the highest total
    score across ALL depths is returned.

    Returns:
        (actions, final_c2w, final_total_score, final_pos_score, final_ori_score)
        actions is empty if no path improved over the initial state.
    """
    init_total, init_pos, init_ori = score_c2w(
        c2w, potential_field, task_type, task_params, target_region
    )

    # Beam state: (total_score, pos_score, ori_score, c2w_4x4, actions_list)
    beam: List[Tuple[float, float, float, np.ndarray, List[str]]] = [
        (init_total, init_pos, init_ori, c2w.copy(), [])
    ]

    # Best state found across all expansion depths
    best_state = beam[0]

    for _depth in range(max_actions):
        candidates: List[Tuple[float, float, float, np.ndarray, List[str]]] = []

        for (_, _, _, state_c2w, actions_so_far) in beam:
            for action in MOVEMENT_ACTIONS:
                candidate_c2w = simulate_action(
                    state_c2w, action, step_translation, step_rotation_deg
                )

                # Collision check
                if collision_checker is not None:
                    try:
                        col = collision_checker.check_collision(
                            candidate_c2w[:3, 3], state_c2w[:3, 3]
                        )
                        if col.has_collision:
                            continue
                    except Exception:
                        pass

                cand_total, cand_pos, cand_ori = score_c2w(
                    candidate_c2w, potential_field, task_type, task_params, target_region
                )
                candidates.append(
                    (cand_total, cand_pos, cand_ori, candidate_c2w, actions_so_far + [action])
                )

        if not candidates:
            break

        # Sort by total score descending; deduplicate by pose fingerprint
        candidates.sort(key=lambda x: -x[0])
        new_beam: List[Tuple[float, float, float, np.ndarray, List[str]]] = []
        seen_poses: set = set()
        for item in candidates:
            fp = (
                tuple(item[3][:3, 3].round(3)) +
                tuple(item[3][:3, 2].round(3))   # forward direction column
            )
            if fp not in seen_poses:
                seen_poses.add(fp)
                new_beam.append(item)
                if len(new_beam) >= beam_width:
                    break

        beam = new_beam

        # Update best: the highest-scoring state ever seen in the beam
        if beam[0][0] > best_state[0]:
            best_state = beam[0]

    best_total, best_pos, best_ori, best_c2w, best_actions = best_state
    return best_actions, best_c2w, best_total, best_pos, best_ori


# ---------------------------------------------------------------------------
# Guided planner helpers
# ---------------------------------------------------------------------------

def _extract_target_pose(
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
    init_c2w: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract the ideal (target_pos, target_forward) from known task geometry.

    target_pos:     world-space 3-D position the camera should reach.
    target_forward: world-space unit forward direction the camera should face.

    Relies on fields that the active_spatial pipeline always populates:
      - target_region["sample_point"]: the exact sampled optimal camera position.
      - target_region["params"]["object_center"] (or "center"): the object to look at.

    Falls back to computing the nearest point on the target circle when
    sample_point is absent (absolute_positioning only).

    Returns (None, None) if geometry cannot be determined.
    """
    # ── Target position ───────────────────────────────────────────────────────
    sample_point = target_region.get("sample_point", None)
    if sample_point is not None:
        target_pos = np.array(sample_point, dtype=float)
    else:
        # Absolute positioning fallback: nearest point on circle
        params = target_region.get("params", {})
        raw_center = params.get("object_center", params.get("center", None))
        if raw_center is None:
            return None, None
        object_center = np.array(raw_center, dtype=float)
        radius = float(params.get("radius", params.get("sample_distance", 2.0)))
        init_pos = init_c2w[:3, 3]
        offset_xy = init_pos[:2] - object_center[:2]
        dist_xy = np.linalg.norm(offset_xy)
        if dist_xy < 1e-6:
            offset_xy = np.array([1.0, 0.0])
        else:
            offset_xy /= dist_xy
        target_xy = object_center[:2] + offset_xy * radius
        target_pos = np.array([target_xy[0], target_xy[1], init_pos[2]])

    # ── Target forward direction ──────────────────────────────────────────────
    params = target_region.get("params", {})
    for key in ("object_center", "center", "pivot_point", "reference_point"):
        raw = params.get(key, None)
        if raw is not None:
            obj = np.array(raw, dtype=float)
            # Pad 2-D centres (some tasks store [x, y] without Z)
            if obj.shape == (2,):
                obj = np.array([obj[0], obj[1], target_pos[2]])
            to_obj = obj - target_pos
            to_obj[2] = 0.0  # horizontal only
            norm = np.linalg.norm(to_obj)
            if norm > 1e-6:
                return target_pos, to_obj / norm

    # Priority 2: sample_forward stored in target_region — exact ground truth
    # direction the camera should face at the sampled target pose.  Present for
    # ALL 9 active_spatial task types (incl. delta_control, equidistance, etc.)
    # that lack an explicit look-at object in params.  Use the full 3-D vector
    # so that _plan_to_orientation can also correct pitch (look_up/look_down).
    sample_fwd = target_region.get("sample_forward", None)
    if sample_fwd is not None:
        sf = np.array(sample_fwd, dtype=float)
        sf_norm = np.linalg.norm(sf)
        if sf_norm > 1e-6:
            return target_pos, sf / sf_norm

    # Last resort: keep current forward direction (horizontal only)
    fwd = (-init_c2w[:3, 2]).copy()
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    if n > 1e-6:
        fwd /= n
    else:
        fwd = np.array([1.0, 0.0, 0.0])
    return target_pos, fwd


def _angular_error(forward_2d: np.ndarray, target_2d: np.ndarray) -> float:
    """Unsigned angle (radians) between two 2-D unit vectors."""
    c = float(np.clip(np.dot(forward_2d, target_2d), -1.0, 1.0))
    return float(np.arccos(c))


def _plan_to_position(
    c2w: np.ndarray,
    target_pos: np.ndarray,
    step_translation: float,
    step_rotation_deg: float,
    position_tolerance: float,
    max_actions: int,
    collision_checker=None,
) -> Tuple[np.ndarray, List[str]]:
    """Geometrically navigate to within *position_tolerance* of target_pos.

    Convention note
    ---------------
    In this codebase the camera-to-world (c2w) matrix uses the convention where
    ``move_forward`` translates along **+c2w[:3, 2]** (the +Z column), which is
    also the visual forward direction used by the env reward/scorer.

    Strategy: align-then-approach.
      1. Align the visual forward (+c2w[:,2]) with the direction to target_pos.
         (try turn_left vs turn_right, pick the one that reduces angular error)
      2. Once roughly aligned, approach via ``move_forward``.
      3. After each approach step, re-check alignment.
    """
    actions: List[str] = []
    step_r = np.radians(step_rotation_deg)

    for _ in range(max_actions):
        pos = c2w[:3, 3]
        to_target = target_pos[:2] - pos[:2]
        dist = float(np.linalg.norm(to_target))

        if dist < position_tolerance:
            break

        to_target_norm = to_target / dist

        # Visual forward in XY: +c2w[:,2] (robust to near-vertical tilt)
        fwd3 = c2w[:3, 2].copy()
        fwd3[2] = 0.0
        f_len = float(np.linalg.norm(fwd3))
        if f_len < 1e-6:
            # Camera tilted straight up/down – apply look_up to recover
            c2w = simulate_action(c2w, "look_up", step_translation, step_rotation_deg)
            actions.append("look_up")
            continue
        fwd2 = fwd3[:2] / f_len

        ang = _angular_error(fwd2, to_target_norm)

        if ang > step_r * 0.45:
            # Rotation needed – pick the direction that reduces angular error on
            # the visual forward (+c2w[:,2]).
            cand_l = simulate_action(c2w, "turn_left", step_translation, step_rotation_deg)
            cand_r = simulate_action(c2w, "turn_right", step_translation, step_rotation_deg)

            def _err_after(cand: np.ndarray) -> float:
                f = cand[:3, 2].copy(); f[2] = 0.0
                n = float(np.linalg.norm(f))
                if n < 1e-6:
                    return float("inf")
                return _angular_error(f[:2] / n, to_target_norm)

            err_l = _err_after(cand_l)
            err_r = _err_after(cand_r)
            action = "turn_left" if err_l <= err_r else "turn_right"
        else:
            # Visual forward is aligned with target, so move_forward approaches it.
            cand_fwd = simulate_action(c2w, "move_forward", step_translation, step_rotation_deg)
            new_dist = float(np.linalg.norm(target_pos[:2] - cand_fwd[:3, 3][:2]))

            if new_dist >= dist:
                break  # Cannot reduce distance further (e.g. target behind a wall)

            # Collision check
            if collision_checker is not None:
                try:
                    col = collision_checker.check_collision(cand_fwd[:3, 3], c2w[:3, 3])
                    if col.has_collision:
                        break
                except Exception:
                    pass
            action = "move_forward"

        c2w = simulate_action(c2w, action, step_translation, step_rotation_deg)
        actions.append(action)

    return c2w, actions


def _plan_to_orientation(
    c2w: np.ndarray,
    target_forward: np.ndarray,
    step_rotation_deg: float,
    max_actions: int,
) -> Tuple[np.ndarray, List[str]]:
    """Rotate camera so its forward vector matches target_forward.

    Handles both yaw (turn_left/right) and pitch (look_up/down) independently:
      - Horizontal phase first (yaw alignment)
      - Then vertical phase (pitch alignment)

    At each step, both candidate directions are tried and the one that
    reduces the error is chosen (avoids sign convention issues).
    """
    actions: List[str] = []
    step_translation = 0.3  # unused for rotations
    step_r = np.radians(step_rotation_deg)

    tf = np.array(target_forward, dtype=float)
    tf_norm = tf / (np.linalg.norm(tf) + 1e-9)

    for _ in range(max_actions):
        fwd = c2w[:3, 2].copy()
        fwd_norm = fwd / (np.linalg.norm(fwd) + 1e-9)

        # Total 3-D angular error
        total_err = float(np.arccos(np.clip(np.dot(fwd_norm, tf_norm), -1.0, 1.0)))
        if total_err < step_r * 0.45:
            break

        # Horizontal error (yaw) – compare XY projections
        fwd_h = fwd_norm[:2]
        tf_h = tf_norm[:2]
        fwd_h_n = fwd_h / (np.linalg.norm(fwd_h) + 1e-9)
        tf_h_n = tf_h / (np.linalg.norm(tf_h) + 1e-9)
        yaw_err = _angular_error(fwd_h_n, tf_h_n)

        # Vertical error (pitch) – elevation angles
        fwd_el = float(np.arctan2(fwd_norm[2], np.linalg.norm(fwd_norm[:2]) + 1e-9))
        tf_el = float(np.arctan2(tf_norm[2], np.linalg.norm(tf_norm[:2]) + 1e-9))
        pitch_err = abs(fwd_el - tf_el)

        # Pick the dominant error to correct first
        if yaw_err >= step_r * 0.45 and yaw_err >= pitch_err:
            cand_l = simulate_action(c2w, "turn_left", step_translation, step_rotation_deg)
            cand_r = simulate_action(c2w, "turn_right", step_translation, step_rotation_deg)

            def _yaw_err(cand: np.ndarray) -> float:
                f = -cand[:3, 2]; h = f[:2]; n = float(np.linalg.norm(h))
                if n < 1e-6: return float("inf")
                return _angular_error(h / n, tf_h_n)

            action = "turn_left" if _yaw_err(cand_l) <= _yaw_err(cand_r) else "turn_right"

        elif pitch_err >= step_r * 0.45:
            cand_u = simulate_action(c2w, "look_up", step_translation, step_rotation_deg)
            cand_d = simulate_action(c2w, "look_down", step_translation, step_rotation_deg)

            def _pitch_err(cand: np.ndarray) -> float:
                f = -cand[:3, 2]; f_n = f / (np.linalg.norm(f) + 1e-9)
                el = float(np.arctan2(f_n[2], np.linalg.norm(f_n[:2]) + 1e-9))
                return abs(el - tf_el)

            action = "look_up" if _pitch_err(cand_u) <= _pitch_err(cand_d) else "look_down"
        else:
            break  # Both errors below threshold

        c2w = simulate_action(c2w, action, step_translation, step_rotation_deg)
        actions.append(action)

    return c2w, actions


# ---------------------------------------------------------------------------
# v20-prep: Always-look-at-target constraint on intermediate (rendered) frames
# ---------------------------------------------------------------------------

def _extract_object_center_xy(
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
    target_pos: np.ndarray,
) -> Optional[np.ndarray]:
    """Best-effort extraction of the world-XY position of the object the
    camera should look at. Falls back to *target_pos* when no explicit centre
    is provided.

    Returns a 2-D float ndarray (x, y) or None when target_pos itself is None.
    """
    if target_pos is None:
        return None
    params = target_region.get("params", {}) if target_region else {}
    for key in ("object_center", "center", "pivot_point", "reference_point"):
        raw = params.get(key, None)
        if raw is None:
            continue
        obj = np.array(raw, dtype=float).flatten()
        if obj.size >= 2:
            return obj[:2]
    # Fallback: use target_pos itself (the camera sample point is close to
    # the look-at object for all 9 active_spatial task types).
    return np.array(target_pos[:2], dtype=float)


def _yaw_alignment_actions(
    c2w: np.ndarray,
    object_xy: np.ndarray,
    step_translation: float,
    step_rotation_deg: float,
    max_steps: int,
    tol_rad: float,
) -> Tuple[np.ndarray, List[str]]:
    """Generate minimal turn_left/turn_right actions so that the camera's
    visual forward (in XY plane) points at *object_xy*. Stops once the
    yaw error is below *tol_rad* or *max_steps* is exhausted.

    Returns (new_c2w, action_list).
    """
    out: List[str] = []
    for _ in range(max_steps):
        pos = c2w[:3, 3]
        to_obj = object_xy - pos[:2]
        d = float(np.linalg.norm(to_obj))
        if d < 1e-6:
            break
        to_obj_n = to_obj / d
        fwd = c2w[:3, 2].copy()
        fwd[2] = 0.0
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-6:
            break
        fwd_n = fwd[:2] / fn
        ang = _angular_error(fwd_n, to_obj_n)
        if ang < tol_rad:
            break
        cand_l = simulate_action(c2w, "turn_left", step_translation, step_rotation_deg)
        cand_r = simulate_action(c2w, "turn_right", step_translation, step_rotation_deg)

        def _err(cand: np.ndarray) -> float:
            f = (-cand[:3, 2]).copy(); f[2] = 0.0
            n = float(np.linalg.norm(f))
            if n < 1e-6:
                return float("inf")
            return _angular_error(f[:2] / n, to_obj_n)

        action = "turn_left" if _err(cand_l) <= _err(cand_r) else "turn_right"
        c2w = simulate_action(c2w, action, step_translation, step_rotation_deg)
        out.append(action)
    return c2w, out


def _enforce_lookat_at_turn_boundaries(
    flat_actions: List[str],
    init_c2w: np.ndarray,
    object_xy: Optional[np.ndarray],
    step_translation: float,
    step_rotation_deg: float,
    max_actions_per_turn: int,
    max_total_actions: int,
    max_corrections_per_boundary: int = 2,
    verbose: bool = False,
) -> List[str]:
    """Post-process a flat action list so that every *rendered* intermediate
    frame (i.e. every turn boundary EXCEPT the last) has the camera yaw
    pointing at *object_xy*.

    Strategy: walk the list turn-by-turn (chunks of ``max_actions_per_turn``),
    simulate the resulting c2w, and at each boundary that is not the final
    turn, prepend up to ``max_corrections_per_boundary`` ``turn_left``/
    ``turn_right`` corrections to the next chunk if the yaw error exceeds the
    step half-angle.

    The total action count is hard-capped at ``max_total_actions``: extra
    corrections that would exceed the cap are dropped.
    """
    if object_xy is None or not flat_actions:
        return flat_actions

    # Tolerance: half of the rotation step. With 20° steps that's 10° — i.e.
    # the camera is "close enough" if a single turn won't reduce error further.
    tol_rad = np.radians(step_rotation_deg) * 0.5

    out: List[str] = []
    pending: List[str] = list(flat_actions)
    c2w = init_c2w.copy()
    boundaries_corrected = 0

    while pending:
        # Take next chunk
        chunk = pending[:max_actions_per_turn]
        pending = pending[max_actions_per_turn:]
        for a in chunk:
            c2w = simulate_action(c2w, a, step_translation, step_rotation_deg)
        out.extend(chunk)

        # If more turns remain, ensure yaw faces object on the rendered frame
        if pending and len(out) < max_total_actions:
            budget = min(
                max_corrections_per_boundary,
                max_total_actions - len(out),
            )
            if budget <= 0:
                continue
            c2w, fixes = _yaw_alignment_actions(
                c2w, object_xy,
                step_translation=step_translation,
                step_rotation_deg=step_rotation_deg,
                max_steps=budget,
                tol_rad=tol_rad,
            )
            if fixes:
                out.extend(fixes)
                boundaries_corrected += 1

    if verbose and boundaries_corrected:
        print(f"  [LookAt] inserted yaw fixes at {boundaries_corrected} turn boundaries "
              f"(total actions: {len(flat_actions)} → {len(out)})")
    return out[:max_total_actions]


def _actions_to_steps(
    flat_actions: List[str],
    init_c2w: np.ndarray,
    potential_field,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
    step_translation: float,
    step_rotation_deg: float,
    max_actions_per_turn: int,
    success_threshold: float,
) -> Tuple[List[TrajStep], np.ndarray, float, float, float, int]:
    """Pack a flat action list into scored LLM turns (TrajStep objects).

    Stops chunking early if success_threshold is reached mid-sequence.

    Returns:
        (steps, final_c2w, final_total, final_pos, final_ori, n_actions_used)
    """
    steps: List[TrajStep] = []
    c2w = init_c2w.copy()
    cur_total, cur_pos, cur_ori = score_c2w(
        c2w, potential_field, task_type, task_params, target_region
    )
    step_idx = 0
    n_used = 0

    i = 0
    while i < len(flat_actions):
        if cur_total >= success_threshold:
            break

        chunk = flat_actions[i: i + max_actions_per_turn]

        c2w_before = c2w.copy()
        score_before = cur_total
        pos_before = cur_pos
        ori_before = cur_ori

        accepted: List[str] = []
        for act in chunk:
            c2w = simulate_action(c2w, act, step_translation, step_rotation_deg)
            accepted.append(act)
            # Stop this turn early if success already reached mid-chunk
            t, p, o = score_c2w(c2w, potential_field, task_type, task_params, target_region)
            if t >= success_threshold:
                cur_total, cur_pos, cur_ori = t, p, o
                break
        else:
            cur_total, cur_pos, cur_ori = score_c2w(
                c2w, potential_field, task_type, task_params, target_region
            )

        if accepted:
            steps.append(TrajStep(
                step_idx=step_idx,
                c2w_before=c2w_before,
                actions=accepted,
                c2w_after=c2w.copy(),
                score_before=score_before,
                score_after=cur_total,
                pos_score_before=pos_before,
                pos_score_after=cur_pos,
                ori_score_before=ori_before,
                ori_score_after=cur_ori,
            ))
            step_idx += 1
            n_used += len(accepted)

        i += max_actions_per_turn

    return steps, c2w, cur_total, cur_pos, cur_ori, n_used


def find_trajectory_guided(
    init_c2w: np.ndarray,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
    position_weight: float = 0.7,
    orientation_weight: float = 0.3,
    max_distance: float = 5.0,
    step_translation: float = 0.3,
    step_rotation_deg: float = 30.0,
    success_threshold: float = 0.95,
    max_total_actions: int = 100,
    max_actions_per_turn: int = 5,
    min_improvement: float = 0.005,
    plateau_tolerance: int = 5,
    beam_width: int = 3,
    adaptive_min_improvement: bool = True,
    collision_checker=None,
    scene_id: str = "",
    item_idx: int = -1,
    verbose: bool = False,
) -> Trajectory:
    """Geometry-guided trajectory finder.

    Instead of blind score optimization, this planner uses the known
    target position and orientation from ``target_region`` to navigate
    in two explicit phases:

      Phase 1 – Position approach
        Navigate directly towards ``target_region["sample_point"]`` by
        minimising XY distance.  No score function is consulted; the
        path is deterministic and free of local optima.

      Phase 2 – Orientation alignment
        Rotate the camera to face the object centre (``params.object_center``).
        Again purely geometric – minimises the 3-D angular error.

      Phase 3 – Score fine-tune  (beam search)
        The discrete step size (0.3 m / 30°) leaves a residual error.
        A short beam search polishes the final pose until
        score >= success_threshold or the budget runs out.

    Falls back to :func:`find_trajectory` when target geometry cannot be
    extracted from ``target_region``.

    Parameters mirror :func:`find_trajectory` exactly so callers can
    swap between the two without changing call sites.
    """
    from vagen.envs.active_spatial.spatial_potential_field import create_potential_field

    # ── Extract target geometry ───────────────────────────────────────────────
    target_pos, target_fwd = _extract_target_pose(
        task_type, task_params, target_region, init_c2w
    )

    if target_pos is None:
        if verbose:
            print(f"[GuidedPathFinder] item={item_idx}: no target geometry, "
                  "falling back to beam search.")
        return find_trajectory(
            init_c2w, task_type, task_params, target_region,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            max_distance=max_distance,
            step_translation=step_translation,
            step_rotation_deg=step_rotation_deg,
            success_threshold=success_threshold,
            max_total_actions=max_total_actions,
            max_actions_per_turn=max_actions_per_turn,
            min_improvement=min_improvement,
            plateau_tolerance=plateau_tolerance,
            beam_width=beam_width,
            adaptive_min_improvement=adaptive_min_improvement,
            collision_checker=collision_checker,
            scene_id=scene_id,
            item_idx=item_idx,
            verbose=verbose,
        )

    potential_field = create_potential_field({
        "position_weight": position_weight,
        "orientation_weight": orientation_weight,
        "max_distance": max_distance,
    })

    init_total, init_pos_s, init_ori_s = score_c2w(
        init_c2w, potential_field, task_type, task_params, target_region
    )

    if verbose:
        dist_to_target = float(np.linalg.norm(init_c2w[:3, 3][:2] - target_pos[:2]))
        print(f"[GuidedPathFinder] item={item_idx} scene={scene_id} task={task_type}")
        print(f"  Initial score: {init_total:.4f}  |  distance to target: {dist_to_target:.2f}m")
        print(f"  Target pos:    {target_pos.round(3)}")
        if target_fwd is not None:
            print(f"  Target fwd:    {target_fwd.round(3)}")

    # Allow slightly more actions for the geometric phases (they are fast)
    phase_budget = max_total_actions * 2  # generous; will be clipped later

    # ── Phase 1: Position approach ────────────────────────────────────────────
    position_tolerance = step_translation * 0.9  # within one step = "arrived"
    c2w_after_pos, pos_actions = _plan_to_position(
        init_c2w, target_pos,
        step_translation=step_translation,
        step_rotation_deg=step_rotation_deg,
        position_tolerance=position_tolerance,
        max_actions=phase_budget,
        collision_checker=collision_checker,
    )

    if verbose:
        dist_now = float(np.linalg.norm(c2w_after_pos[:3, 3][:2] - target_pos[:2]))
        print(f"  Phase 1: {len(pos_actions)} actions → distance to target: {dist_now:.2f}m")

    # ── Phase 2: Orientation alignment ────────────────────────────────────────
    c2w_after_ori, ori_actions = _plan_to_orientation(
        c2w_after_pos, target_fwd,
        step_rotation_deg=step_rotation_deg,
        max_actions=phase_budget,
    ) if target_fwd is not None else (c2w_after_pos, [])

    if verbose:
        t_guided, _, _ = score_c2w(
            c2w_after_ori, potential_field, task_type, task_params, target_region
        )
        print(f"  Phase 2: {len(ori_actions)} actions → score after geometry: {t_guided:.4f}")

    # ── Combine, clip to budget, and pack into turns ──────────────────────────
    all_guided = pos_actions + ori_actions

    # v20-prep: enforce "always-look-at-target" on every rendered intermediate
    # frame. The final frame is allowed to satisfy target_forward exactly; only
    # intermediate turn boundaries are forced to keep the object centred.
    object_xy = _extract_object_center_xy(task_params, target_region, target_pos)
    if object_xy is not None and len(all_guided) > max_actions_per_turn:
        all_guided = _enforce_lookat_at_turn_boundaries(
            all_guided, init_c2w, object_xy,
            step_translation=step_translation,
            step_rotation_deg=step_rotation_deg,
            max_actions_per_turn=max_actions_per_turn,
            max_total_actions=max_total_actions,
            max_corrections_per_boundary=2,
            verbose=verbose,
        )

    all_guided = all_guided[:max_total_actions]  # hard cap

    steps, c2w, cur_total, cur_pos, cur_ori, n_used = _actions_to_steps(
        all_guided,
        init_c2w,
        potential_field,
        task_type, task_params, target_region,
        step_translation, step_rotation_deg,
        max_actions_per_turn,
        success_threshold,
    )
    total_actions = n_used

    if verbose:
        print(f"  After geometry phases: score={cur_total:.4f}, "
              f"actions={total_actions}, success={cur_total >= success_threshold}")

    # ── Phase 3: Beam-search fine-tune (handles discretization residual) ──────
    if cur_total < success_threshold and total_actions < max_total_actions:
        remaining = max_total_actions - total_actions
        if verbose:
            print(f"  Phase 3: beam search fine-tune (budget={remaining}, beam={beam_width})")

        finetune = find_trajectory(
            c2w,
            task_type, task_params, target_region,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            max_distance=max_distance,
            step_translation=step_translation,
            step_rotation_deg=step_rotation_deg,
            success_threshold=success_threshold,
            max_total_actions=remaining,
            max_actions_per_turn=max_actions_per_turn,
            min_improvement=min_improvement,
            plateau_tolerance=plateau_tolerance,
            beam_width=beam_width,
            adaptive_min_improvement=adaptive_min_improvement,
            collision_checker=collision_checker,
            scene_id=scene_id,
            item_idx=item_idx,
            verbose=verbose,
        )

        # Re-index and merge finetune steps
        base_idx = len(steps)
        for ft_step in finetune.steps:
            ft_step.step_idx += base_idx
            steps.append(ft_step)

        c2w = finetune.final_c2w
        cur_total = finetune.final_score
        total_actions += finetune.total_actions

        if verbose:
            print(f"  Phase 3 done: score={cur_total:.4f}, "
                  f"total_actions={total_actions}")

    return Trajectory(
        steps=steps,
        final_c2w=c2w,
        initial_score=init_total,
        final_score=cur_total,
        success=cur_total >= success_threshold,
        total_actions=total_actions,
        scene_id=scene_id,
        item_idx=item_idx,
    )


# ---------------------------------------------------------------------------
# Main trajectory finder
# ---------------------------------------------------------------------------

def find_trajectory(
    init_c2w: np.ndarray,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
    position_weight: float = 0.7,
    orientation_weight: float = 0.3,
    max_distance: float = 5.0,
    step_translation: float = 0.3,
    step_rotation_deg: float = 30.0,
    success_threshold: float = 0.95,
    max_total_actions: int = 100,
    max_actions_per_turn: int = 5,
    min_improvement: float = 0.005,
    plateau_tolerance: int = 5,
    beam_width: int = 1,
    adaptive_min_improvement: bool = True,
    collision_checker=None,
    scene_id: str = "",
    item_idx: int = -1,
    verbose: bool = False,
) -> Trajectory:
    """Find an optimal camera trajectory from init_c2w to the target region.

    The path is broken into "LLM turns", each containing up to
    max_actions_per_turn individual actions.

    Args:
        init_c2w: 4x4 initial camera-to-world matrix.
        task_type: Task type string (e.g. 'absolute_positioning').
        task_params: Task-specific parameters dict.
        target_region: Target region dict from the pipeline JSONL.
        position_weight: Weight for position score component.
        orientation_weight: Weight for orientation score component.
        max_distance: Max distance for score normalisation.
        step_translation: Translation step size (metres).
        step_rotation_deg: Rotation step size (degrees).
        success_threshold: Score threshold for success.
        max_total_actions: Hard cap on total individual actions.
        max_actions_per_turn: Max actions bundled per LLM turn.
        min_improvement: Base minimum score delta to accept a turn.
        plateau_tolerance: Number of turns without improvement before stopping.
        beam_width: Beam width within each LLM turn.
            1 = original greedy behaviour (fast, may get stuck in local optima).
            3+ = beam search (tolerates short-term drops, higher success rate).
        adaptive_min_improvement: If True, scale min_improvement down as the
            score approaches success_threshold.
        collision_checker: Optional CollisionDetector instance.
        scene_id: Scene identifier (metadata only).
        item_idx: Source item index (metadata only).
        verbose: Print progress.

    Returns:
        Trajectory with all steps.
    """
    from vagen.envs.active_spatial.spatial_potential_field import create_potential_field

    potential_field = create_potential_field({
        "position_weight": position_weight,
        "orientation_weight": orientation_weight,
        "max_distance": max_distance,
    })

    c2w = init_c2w.copy()
    steps: List[TrajStep] = []
    total_actions = 0
    plateau_count = 0
    step_idx = 0
    best_total_ever = -1.0
    turns_since_best_improved = 0
    # Beam search rarely oscillates, so give more patience
    MAX_TURNS_WITHOUT_BEST = 20 if beam_width > 1 else 12

    init_total, init_pos, init_ori = score_c2w(
        c2w, potential_field, task_type, task_params, target_region
    )
    current_total = init_total
    current_pos = init_pos
    current_ori = init_ori

    if verbose:
        mode = f"beam_width={beam_width}" if beam_width > 1 else "greedy"
        print(f"[PathFinder] item={item_idx} scene={scene_id} task={task_type} mode={mode}")
        print(f"  Initial score: {current_total:.4f} (pos={current_pos:.4f}, ori={current_ori:.4f})")

    while total_actions < max_total_actions and current_total < success_threshold:
        # ── Adaptive min_improvement ─────────────────────────────────────────
        if adaptive_min_improvement:
            remaining = max(0.0, success_threshold - current_total)
            # Scale proportionally to remaining headroom; floor at _MIN_IMPROVEMENT_FLOOR
            headroom_ratio = min(1.0, remaining / max(0.05, 1.0 - success_threshold + remaining))
            effective_min_improvement = max(_MIN_IMPROVEMENT_FLOOR, min_improvement * headroom_ratio)
        else:
            effective_min_improvement = min_improvement

        # ── Global oscillation guard ─────────────────────────────────────────
        if current_total > best_total_ever + effective_min_improvement:
            best_total_ever = current_total
            turns_since_best_improved = 0
        else:
            turns_since_best_improved += 1
            if turns_since_best_improved >= MAX_TURNS_WITHOUT_BEST:
                if verbose:
                    print(f"  Stopping: best score unchanged for {MAX_TURNS_WITHOUT_BEST} turns "
                          f"(best={best_total_ever:.4f})")
                break

        c2w_turn_start = c2w.copy()
        score_turn_start = current_total
        pos_turn_start = current_pos
        ori_turn_start = current_ori

        turn_actions: List[str] = []

        # ── Find actions for this LLM turn ───────────────────────────────────
        if beam_width > 1:
            # Beam search: tolerates short-term score drops to find better paths
            raw_actions, new_c2w, new_total, new_pos, new_ori = _beam_search_turn(
                c2w, potential_field, task_type, task_params, target_region,
                step_translation, step_rotation_deg,
                max_actions=max_actions_per_turn,
                beam_width=beam_width,
                collision_checker=collision_checker,
            )
            # Respect global action budget
            remaining_budget = max_total_actions - total_actions
            raw_actions = raw_actions[:remaining_budget]
            if raw_actions and new_total >= current_total + effective_min_improvement:
                # Replay prefix to get exact final state (beam returns state for full sequence)
                replay_c2w = c2w.copy()
                for act in raw_actions:
                    replay_c2w = simulate_action(replay_c2w, act, step_translation, step_rotation_deg)
                replay_total, replay_pos, replay_ori = score_c2w(
                    replay_c2w, potential_field, task_type, task_params, target_region
                )
                c2w = replay_c2w
                current_total = replay_total
                current_pos = replay_pos
                current_ori = replay_ori
                turn_actions = raw_actions
                total_actions += len(raw_actions)
        else:
            # Original greedy: pack one action at a time
            for _ in range(max_actions_per_turn):
                if total_actions >= max_total_actions:
                    break
                if current_total >= success_threshold:
                    break

                best_action, best_c2w, best_total, best_pos, best_ori = _greedy_single_step(
                    c2w, potential_field, task_type, task_params, target_region,
                    step_translation, step_rotation_deg, collision_checker,
                )

                if best_action is None or best_total < current_total + effective_min_improvement:
                    break

                c2w = best_c2w
                current_total = best_total
                current_pos = best_pos
                current_ori = best_ori
                turn_actions.append(best_action)
                total_actions += 1

        # ── Handle plateau ───────────────────────────────────────────────────
        if not turn_actions:
            plateau_count += 1
            if verbose:
                print(f"  Turn {step_idx}: plateau ({plateau_count}/{plateau_tolerance}), "
                      f"score={current_total:.4f}")

            if plateau_count >= plateau_tolerance:
                break

            if beam_width > 1:
                # Beam search already explored rotations internally; a plateau here means
                # no sequence of max_actions_per_turn actions helps. Try a longer rotation
                # sweep (up to full 360°) as a last-resort escape before giving up.
                MAX_ESCAPE_STEPS = 12  # 12 × 30° = 360°
                escape_found = False
                for action in ROTATION_ACTIONS:
                    seq_c2w = c2w.copy()
                    for n in range(1, MAX_ESCAPE_STEPS + 1):
                        seq_c2w = simulate_action(seq_c2w, action, step_translation, step_rotation_deg)
                        cand_total, cand_pos, cand_ori = score_c2w(
                            seq_c2w, potential_field, task_type, task_params, target_region
                        )
                        if cand_total > current_total + effective_min_improvement:
                            c2w = seq_c2w
                            current_total = cand_total
                            current_pos = cand_pos
                            current_ori = cand_ori
                            turn_actions = [action] * n
                            total_actions += n
                            escape_found = True
                            break
                    if escape_found:
                        break
                if not escape_found:
                    break  # Truly stuck

            else:
                # Original two-phase escape for greedy mode
                # Phase 1: single-step rotation (allow up to 2% regression)
                escape_action = None
                escape_c2w = c2w
                escape_total = current_total - 1.0
                escape_pos_val = current_pos
                escape_ori_val = current_ori

                for action in ROTATION_ACTIONS:
                    candidate = simulate_action(c2w, action, step_translation, step_rotation_deg)
                    cand_total, cand_pos, cand_ori = score_c2w(
                        candidate, potential_field, task_type, task_params, target_region
                    )
                    if cand_total >= current_total - 0.02 and cand_total > escape_total:
                        escape_total = cand_total
                        escape_c2w = candidate
                        escape_action = action
                        escape_pos_val = cand_pos
                        escape_ori_val = cand_ori

                if escape_action is not None:
                    c2w = escape_c2w
                    current_total = escape_total
                    current_pos = escape_pos_val
                    current_ori = escape_ori_val
                    turn_actions = [escape_action]
                    total_actions += 1
                else:
                    # Phase 2: multi-step rotation (object behind camera)
                    MAX_ESCAPE_STEPS = 6
                    escape_actions_multi: List[str] = []
                    escape_c2w_best = c2w
                    escape_total_multi = current_total - 1.0
                    escape_pos_val = current_pos
                    escape_ori_val = current_ori

                    for action in ROTATION_ACTIONS:
                        seq_c2w = c2w.copy()
                        for n in range(2, MAX_ESCAPE_STEPS + 1):
                            seq_c2w = simulate_action(
                                seq_c2w, action, step_translation, step_rotation_deg
                            )
                            cand_total, cand_pos, cand_ori = score_c2w(
                                seq_c2w, potential_field, task_type, task_params, target_region
                            )
                            if cand_total > escape_total_multi:
                                escape_total_multi = cand_total
                                escape_c2w_best = seq_c2w.copy()
                                escape_actions_multi = [action] * n
                                escape_pos_val = cand_pos
                                escape_ori_val = cand_ori

                    if escape_actions_multi and escape_total_multi > current_total + min_improvement:
                        c2w = escape_c2w_best
                        current_total = escape_total_multi
                        current_pos = escape_pos_val
                        current_ori = escape_ori_val
                        turn_actions = escape_actions_multi
                        total_actions += len(escape_actions_multi)
                    else:
                        break  # Truly stuck
        else:
            plateau_count = 0

        step = TrajStep(
            step_idx=step_idx,
            c2w_before=c2w_turn_start,
            actions=turn_actions,
            c2w_after=c2w.copy(),
            score_before=score_turn_start,
            score_after=current_total,
            pos_score_before=pos_turn_start,
            pos_score_after=current_pos,
            ori_score_before=ori_turn_start,
            ori_score_after=current_ori,
        )
        steps.append(step)
        step_idx += 1

        if verbose:
            print(f"  Turn {step_idx - 1}: actions={turn_actions} "
                  f"score {score_turn_start:.4f} → {current_total:.4f}")

    if verbose:
        print(f"  Final score: {current_total:.4f}, success={current_total >= success_threshold}, "
              f"steps={len(steps)}, total_actions={total_actions}")

    return Trajectory(
        steps=steps,
        final_c2w=c2w,
        initial_score=init_total,
        final_score=current_total,
        success=current_total >= success_threshold,
        total_actions=total_actions,
        scene_id=scene_id,
        item_idx=item_idx,
    )
