"""E3VS camera geometry: pure-numpy copies of the official ``e3vs/environments/camera_geometry.py``.

Source: https://github.com/K0uya/E3VS-BENCH @ bb7e603ccdcf2dd77fbeedd15660a09310a8a881,
``e3vs/environments/camera_geometry.py`` (view_forward, planar_forward, planar_right, movement_directions,
apply_action, viewmat_and_K).  The official module imports torch and scipy; this copy uses numpy only, so the
BFS, the floors and the closed-loop eval need neither.  vpd/tests/test_e3vs_geometry.py checks it against the
official module on 1,000 random action sequences (max abs diff < 1e-9).

Conventions (official docstring): world frame right-handed, Z up.  Camera frame OpenCV/gsplat: c2w[:,0] = image
right, c2w[:,1] = image down, c2w[:,2] = forward (view direction).  Rendering uses viewmat = inv(c2w) and a
pinhole K with focal W/2 and the principal point at the image centre (90 deg FOV at 512x512).

Action semantics (official config: action_coordinate='camera', 0.25 m, 30 deg):
  move_forward/backward : +-step along the full 3D view direction ('world': its horizontal projection);
  move_left/right       : -+step along planar_forward x world_up (horizontal);
  move_up/down          : +-step along world Z;
  turn_left/right       : +-angle about world Z (pre-multiplied);
  look_up/down          : +-angle about the camera's own right axis c2w[:3,0] (pre-multiplied);
  stop                  : unchanged.
There is no pitch limit in the official environment.

Differences from the official module (all intentional):
  - rotations come from a numpy port of scipy's ``Rotation.from_rotvec(..).as_matrix()`` (same quaternion path);
  - ``apply_action`` raises ValueError on an unknown action name (official: silently a no-op); None / "" are
    still a no-op as in the official code;
  - ``viewmat_and_K`` returns float64 numpy arrays (official: float32 torch tensors of the same values);
  - ``success`` / ``match_goals`` are new: the official repo scores with a VQA judge, the pilot scores the
    viewpoint (report v2 section 3.2): position within 0.25 m AND view-direction angle <= 20 deg of any
    answerable viewpoint.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from vpd.e3vs import ACTIONS

WORLD_UP = np.array([0.0, 0.0, 1.0])
_ACTION_SET = frozenset(ACTIONS)


# ------------------------------------------------------------------------------------------------------------
# rotations (numpy port of scipy.spatial.transform.Rotation.from_rotvec(v).as_matrix())
# ------------------------------------------------------------------------------------------------------------
def rotvec_matrix(rotvec) -> np.ndarray:
    """3x3 rotation matrix of a rotation vector (axis * angle in radians), scipy's quaternion path."""
    v = np.asarray(rotvec, dtype=np.float64)
    angle = math.sqrt(float(v @ v))
    if angle <= 1e-3:  # scipy's Taylor branch
        a2 = angle * angle
        scale = 0.5 - a2 / 48.0 + a2 * a2 / 3840.0
    else:
        scale = math.sin(angle / 2.0) / angle
    x, y, z = (scale * v).tolist()
    w = math.cos(angle / 2.0)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.array([
        [x2 - y2 - z2 + w2, 2.0 * (xy - zw), 2.0 * (xz + yw)],
        [2.0 * (xy + zw), -x2 + y2 - z2 + w2, 2.0 * (yz - xw)],
        [2.0 * (xz - yw), 2.0 * (yz + xw), -x2 - y2 + z2 + w2],
    ])


# ------------------------------------------------------------------------------------------------------------
# directions
# ------------------------------------------------------------------------------------------------------------
def view_forward(c2w) -> np.ndarray:
    """Unit view direction (camera +Z) in world coordinates."""
    v = np.asarray(c2w)[:3, 2]
    return v / np.linalg.norm(v)


def planar_forward(c2w) -> np.ndarray:
    """View direction projected on the world XY plane, unit length (zero vector when looking straight up/down)."""
    c2w = np.asarray(c2w)
    pf = np.array([c2w[0, 2], c2w[1, 2], 0.0])
    norm = np.linalg.norm(pf)
    if norm > 1e-6:
        return pf / norm
    return np.zeros(3)


def planar_right(c2w) -> np.ndarray:
    """Horizontal right direction: planar_forward x world up."""
    return np.cross(planar_forward(c2w), WORLD_UP)


def movement_directions(c2w, coord: str = "camera"):
    """(forward, right) used by the translation actions; right is always horizontal."""
    pf = planar_forward(c2w)
    if coord == "world":
        forward = pf.copy()
    elif coord == "camera":
        forward = view_forward(c2w)
    else:
        raise ValueError(f"Unknown action_coordinate: {coord}")
    right = np.cross(pf, WORLD_UP)
    return forward, right


# ------------------------------------------------------------------------------------------------------------
# actions
# ------------------------------------------------------------------------------------------------------------
def apply_action(c2w, a: str | None, step: float = 0.25, ang: float = 30.0, coord: str = "camera") -> np.ndarray:
    """New 4x4 c2w after one discrete action (c2w is not mutated)."""
    c2w = np.array(c2w, dtype=np.float64, copy=True)
    if not a or a == "stop":
        return c2w
    if a not in _ACTION_SET:
        raise ValueError(f"unknown E3VS action {a!r}")
    if a.startswith("move"):
        forward, right = movement_directions(c2w, coord)
        if a == "move_forward":
            delta = step * forward
        elif a == "move_backward":
            delta = -step * forward
        elif a == "move_left":
            delta = -step * right
        elif a == "move_right":
            delta = step * right
        elif a == "move_up":
            delta = np.array([0.0, 0.0, step])
        else:  # move_down
            delta = np.array([0.0, 0.0, -step])
        c2w[:3, 3] += delta
        return c2w
    rad = math.radians(ang)
    if a == "turn_left":
        rot = rotvec_matrix(rad * WORLD_UP)
    elif a == "turn_right":
        rot = rotvec_matrix(-rad * WORLD_UP)
    elif a == "look_up":
        rot = rotvec_matrix(rad * c2w[:3, 0])
    else:  # look_down
        rot = rotvec_matrix(-rad * c2w[:3, 0])
    c2w[:3, :3] = rot @ c2w[:3, :3]
    return c2w


def rollout(c2w, actions: Iterable[str], step: float = 0.25, ang: float = 30.0, coord: str = "camera",
            collides=None) -> tuple[list[np.ndarray], list[bool]]:
    """Poses after each action (index 0 = start) under the official restrict_collision rule.

    ``collides(p0, p1) -> bool`` is asked for translations only (a rotation's segment has zero length and the
    official ray test never fires on it).  A colliding move is skipped: the pose stays, the step is consumed.
    Returns (poses, collided flags per action).
    """
    poses = [np.array(c2w, dtype=np.float64, copy=True)]
    flags = []
    cur = poses[0]
    for a in actions:
        nxt = apply_action(cur, a, step, ang, coord)
        hit = False
        if collides is not None and a and a.startswith("move_"):
            hit = bool(collides(cur[:3, 3].copy(), nxt[:3, 3].copy()))
        if hit:
            nxt = cur.copy()
        flags.append(hit)
        poses.append(nxt)
        cur = nxt
    return poses, flags


# ------------------------------------------------------------------------------------------------------------
# success
# ------------------------------------------------------------------------------------------------------------
def _goal_stack(goals) -> np.ndarray:
    if isinstance(goals, Mapping):
        goals = list(goals.values())
    g = np.asarray(goals, dtype=np.float64)
    if g.ndim == 2:
        g = g[None]
    if g.ndim != 3 or g.shape[1:] != (4, 4):
        raise ValueError(f"goals must be (G,4,4), got {g.shape}")
    return g


def match_goals(c2w, goals, pos_tol: float = 0.25, ang_tol: float = 20.0) -> np.ndarray:
    """Boolean mask over goals: within pos_tol metres AND view directions within ang_tol degrees."""
    g = _goal_stack(goals)
    if len(g) == 0:
        return np.zeros(0, dtype=bool)
    c2w = np.asarray(c2w, dtype=np.float64)
    d = g[:, :3, 3] - c2w[:3, 3]
    d2 = np.einsum("ij,ij->i", d, d)
    gf = g[:, :3, 2] / np.linalg.norm(g[:, :3, 2], axis=1, keepdims=True)
    dot = gf @ view_forward(c2w)
    return (d2 <= pos_tol * pos_tol) & (dot >= math.cos(math.radians(ang_tol)))


def success(c2w, goals, pos_tol: float = 0.25, ang_tol: float = 20.0) -> bool:
    """Viewpoint success: within pos_tol of ANY goal viewpoint with view angle <= ang_tol (both inclusive)."""
    return bool(match_goals(c2w, goals, pos_tol, ang_tol).any())


def view_angle_deg(c2w_a, c2w_b) -> float:
    """Angle between two view directions, degrees."""
    c = float(np.clip(view_forward(c2w_a) @ view_forward(c2w_b), -1.0, 1.0))
    return math.degrees(math.acos(c))


def nearest_goal(c2w, goals) -> tuple[int, float, float]:
    """(index, position distance m, view angle deg) of the goal nearest in position."""
    g = _goal_stack(goals)
    d = np.linalg.norm(g[:, :3, 3] - np.asarray(c2w)[:3, 3], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i]), view_angle_deg(c2w, g[i])


# ------------------------------------------------------------------------------------------------------------
# rendering setup
# ------------------------------------------------------------------------------------------------------------
def viewmat_and_K(c2w, W: int = 512, H: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera matrix inv(c2w) and pinhole K (focal W/2, centre (W/2, H/2)); float64 numpy."""
    viewmat = np.linalg.inv(np.array(c2w, dtype=np.float64, copy=True))
    focal = W / 2.0
    K = np.array([[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]])
    return viewmat, K


def yaw_pitch(c2w) -> tuple[float, float]:
    """(yaw, pitch) of the view direction in radians; pitch > 0 looks up (world Z up)."""
    f = np.asarray(c2w)[:3, 2]
    return math.atan2(float(f[1]), float(f[0])), math.asin(max(-1.0, min(1.0, float(f[2]))))


def roll_of(c2w) -> float:
    """|z component of the camera right axis|: 0 for a level (roll-free) camera."""
    return abs(float(np.asarray(c2w)[2, 0]))
