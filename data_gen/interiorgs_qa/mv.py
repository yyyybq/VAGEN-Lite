"""Helpers for multi-image and video questions: camera motions, directions in a camera or compass frame,
and per-view visibility classes on the rendered depth."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geom import CAMERA_HEIGHT_M, angle_diff_deg, pose_yaw_pitch, yaw_pitch_of

DIR8 = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
COMPASS8 = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
MMSI_DIR8 = {"front": "Directly in front", "front-right": "Front right", "right": "Directly to the right",
             "back-right": "Back right", "back": "Directly behind", "back-left": "Back left",
             "left": "Directly to the left", "front-left": "Front left"}


def frame_h(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera position (2,), horizontal forward (2,), horizontal right (2,)."""
    yaw, _ = yaw_pitch_of(c2w)
    f = np.array([math.cos(yaw), math.sin(yaw)])
    r = np.array([math.sin(yaw), -math.cos(yaw)])
    return np.asarray(c2w, float)[:2, 3], f, r


def bearing(c2w: np.ndarray, p: Sequence[float]) -> Tuple[float, float]:
    """(angle in degrees, 0 = straight ahead, +90 = right; horizontal distance) of world point p."""
    c, f, r = frame_h(c2w)
    v = np.asarray(p, float)[:2] - c
    return (math.degrees(math.atan2(float(v @ r), float(v @ f))) + 360.0) % 360.0, float(np.linalg.norm(v))


def bearing_from(origin: Sequence[float], facing: Sequence[float], p: Sequence[float]) -> float:
    """Angle of p seen from ``origin`` facing towards ``facing`` (0 ahead, +90 right), horizontal."""
    o = np.asarray(origin, float)[:2]
    f = np.asarray(facing, float)[:2] - o
    f /= max(np.linalg.norm(f), 1e-9)
    r = np.array([f[1], -f[0]])
    v = np.asarray(p, float)[:2] - o
    return (math.degrees(math.atan2(float(v @ r), float(v @ f))) + 360.0) % 360.0


def sector8(angle: float, margin: float = 7.5) -> Optional[str]:
    """8 sectors of 45 degrees centred on front (0), front-right (45), ...; None within ``margin`` of a
    boundary (ViewSpatial's get_direction uses the same sectors)."""
    k = int(((angle + 22.5) % 360.0) // 45.0)
    centre = 45.0 * k
    if angle_diff_deg(angle, centre) > 22.5 - margin:
        return None
    return DIR8[k]


def quadrant(angle: float, margin: float = 10.0) -> Optional[str]:
    """front-left / front-right / back-left / back-right; None within ``margin`` of an axis."""
    if min(angle_diff_deg(angle, a) for a in (0, 90, 180, 270)) < margin:
        return None
    if angle < 90:
        return "front-right"
    if angle < 180:
        return "back-right"
    if angle < 270:
        return "back-left"
    return "front-left"


def four_way(angle: float, margin: float = 10.0) -> Optional[str]:
    """front / right / back / left (90-degree sectors centred on the axes)."""
    if min(angle_diff_deg(angle, a) for a in (45, 135, 225, 315)) < margin:
        return None
    return ["front", "right", "back", "left"][int(((angle + 45.0) % 360.0) // 90.0)]


def compass(north: Sequence[float], origin: Sequence[float], p: Sequence[float], margin: float = 7.5) -> Optional[str]:
    """8-way compass direction of p from origin, north = the given horizontal vector (east = north turned
    90 degrees clockwise seen from above)."""
    n = np.asarray(north, float)[:2]
    n /= max(np.linalg.norm(n), 1e-9)
    e = np.array([n[1], -n[0]])
    v = np.asarray(p, float)[:2] - np.asarray(origin, float)[:2]
    ang = (math.degrees(math.atan2(float(v @ e), float(v @ n))) + 360.0) % 360.0
    s = sector8(ang, margin)
    return None if s is None else COMPASS8[DIR8.index(s)]


def rotate2(v: Sequence[float], deg: float) -> np.ndarray:
    """Rotate a 2D vector counter-clockwise (seen from above) by ``deg``."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    v = np.asarray(v, float)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def move(c2w: np.ndarray, right: float = 0.0, forward: float = 0.0, turn_left_deg: float = 0.0,
         pitch_deg: Optional[float] = None) -> np.ndarray:
    """Camera after a horizontal translation in its own frame and a turn (positive = left)."""
    c, f, r = frame_h(c2w)
    yaw, pitch = yaw_pitch_of(c2w)
    p = c + right * r + forward * f
    return pose_yaw_pitch((p[0], p[1], CAMERA_HEIGHT_M), yaw + math.radians(turn_left_deg),
                          pitch if pitch_deg is None else math.radians(pitch_deg))


def movement_label(c2w1: np.ndarray, c2w2: np.ndarray, tol: float = 20.0) -> Optional[str]:
    """MindCube self-movement label from view 1 to view 2 (in view 1's frame): Directly left / right,
    Diagonally forward and left / right; None otherwise."""
    ang, dist = bearing(c2w1, np.asarray(c2w2, float)[:3, 3])
    if dist < 0.3:
        return None
    for centre, lab in ((90, "Directly right"), (270, "Directly left"), (45, "Diagonally forward and right"),
                        (315, "Diagonally forward and left")):
        if angle_diff_deg(ang, centre) <= tol:
            return lab
    return None


def turn_move_distance(c2w: np.ndarray, turn: str, p: Sequence[float], step: float = 1.0) -> Tuple[float, float]:
    """Distance to p before and after turning (left / right / around / none) and moving ``step`` forward."""
    t = {"left": 90.0, "right": -90.0, "around": 180.0, "none": 0.0}[turn]
    c2 = move(c2w, turn_left_deg=t)
    c3 = move(c2, forward=step)
    p = np.asarray(p, float)[:2]
    return float(np.linalg.norm(p - c2[:2, 3])), float(np.linalg.norm(p - c3[:2, 3]))


ABSENT_LOOSE_AREA = 0.002   # "does not appear": at most an unrecognisable sliver (< 0.2% of the image)


def vis_class(va, oid: str, min_in_frame: float = 0.6, loose_absent: bool = False) -> str:
    """'rec' recognisable, 'abs' absent, 'part' in between, on the rendered view.  ``loose_absent``: a
    sliver below ABSENT_LOOSE_AREA of the image (and not recognisable) also counts as absent."""
    ov = va.obj(oid)
    if ov.recognisable(min_in_frame):
        return "rec"
    if ov.absent() or (loose_absent and ov.area < ABSENT_LOOSE_AREA):
        return "abs"
    return "part"
