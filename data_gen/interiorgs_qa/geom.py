"""Geometry for constructive QA: oriented boxes from the raw labels.json corners, the camera, rays.

Conventions (inherited from v10, data_gen/interiorgs_v10/contract.py): world metres, Z up; camera c2w in
OpenCV axes (columns right, down, forward).  The image camera is v10's canonical one -- the native
640 x 480 intrinsics (horizontal field of view 90 degrees) resized to a square image -- so QA images look
like the Act-SFT images, only larger (512 x 512 instead of 256 x 256).

A box is the labels.json box as given: corners p0..p7 with edges e1 = p1 - p0, e2 = p3 - p0, e3 = p4 - p0
(trial F step 1, trial_f_axes.py: the order follows the object's own frame).  v10 turns every box into an
axis-aligned one; rotated boxes (about 9% of all) need the original corners for exact sizes, distances and
silhouettes, so this module keeps them.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10.contract import CAMERA_HEIGHT_M, NATIVE_K, NATIVE_SIZE
from vagen.envs.active_spatial.canonical_camera import camera_pose_from_forward, scale_intrinsics_for_resize

IMAGE_SIZE = 512
MIN_Z = 0.05


# =============================================================================
# Camera
# =============================================================================
def camera_K(size: int = IMAGE_SIZE) -> np.ndarray:
    """v10's canonical camera (native 640 x 480, fx = fy = 320) resized to ``size`` x ``size``."""
    return scale_intrinsics_for_resize(NATIVE_K, NATIVE_SIZE, (size, size))


def look_at(eye: Sequence[float], target: Sequence[float], max_pitch_down_deg: float = 40.0,
            max_pitch_up_deg: float = 8.0) -> np.ndarray:
    """c2w at ``eye`` looking towards ``target``; pitch clamped (no roll)."""
    eye = np.asarray(eye, float)
    d = np.asarray(target, float) - eye
    h = math.hypot(float(d[0]), float(d[1]))
    yaw = math.atan2(float(d[1]), float(d[0]))
    pitch = math.atan2(float(d[2]), max(h, 1e-6))
    pitch = max(-math.radians(max_pitch_down_deg), min(math.radians(max_pitch_up_deg), pitch))
    return pose_yaw_pitch(eye, yaw, pitch)


def pose_yaw_pitch(eye: Sequence[float], yaw: float, pitch: float = 0.0) -> np.ndarray:
    """c2w at ``eye`` with heading ``yaw`` (radians, world XY) and ``pitch`` (radians, positive up)."""
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    return camera_pose_from_forward(np.asarray(eye, float), f)


def yaw_pitch_of(c2w: np.ndarray) -> Tuple[float, float]:
    f = np.asarray(c2w, float)[:3, 2]
    return math.atan2(float(f[1]), float(f[0])), math.asin(max(-1.0, min(1.0, float(f[2]))))


def to_camera(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """(N, 3) world points -> camera coordinates (right, down, forward)."""
    c2w = np.asarray(c2w, float)
    return (np.asarray(points, float).reshape(-1, 3) - c2w[:3, 3]) @ c2w[:3, :3]


def project(points: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (N, 2) and camera depth z (N,) of world points (z <= 0: behind the camera)."""
    cam = to_camera(points, c2w)
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
    return uv, z


def pixel_rays(c2w: np.ndarray, K: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """World directions (..., 3) of the rays through pixel centres (us, vs), scaled so that a ray parameter t
    equals the camera depth z of the point it reaches (camera-space direction has z = 1)."""
    x = (np.asarray(us, float) + 0.5 - K[0, 2]) / K[0, 0]
    y = (np.asarray(vs, float) + 0.5 - K[1, 2]) / K[1, 1]
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    return d_cam @ np.asarray(c2w, float)[:3, :3].T


# =============================================================================
# Oriented boxes
# =============================================================================
@dataclasses.dataclass
class OBB:
    """An oriented box: centre, unit axes (columns), half extents, and the 8 corners in labels.json order."""

    center: np.ndarray
    axes: np.ndarray       # (3, 3), column k = unit direction of edge e_{k+1}
    half: np.ndarray       # (3,)
    corners: np.ndarray    # (8, 3)
    from_corners: bool = True  # False: rebuilt as an axis-aligned box (corner pattern invalid)

    @classmethod
    def from_label_corners(cls, xyz: np.ndarray) -> "OBB":
        xyz = np.asarray(xyz, float).reshape(8, 3)
        e1, e2, e3 = xyz[1] - xyz[0], xyz[3] - xyz[0], xyz[4] - xyz[0]
        pred = np.array([xyz[0], xyz[0] + e1, xyz[0] + e1 + e2, xyz[0] + e2,
                         xyz[0] + e3, xyz[0] + e1 + e3, xyz[0] + e1 + e2 + e3, xyz[0] + e2 + e3])
        lens = np.array([np.linalg.norm(e) for e in (e1, e2, e3)])
        ok = bool(np.max(np.abs(pred - xyz)) <= 1e-4 and lens.min() > 1e-4)
        if ok:
            cos = max(abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
                      for a, b in ((e1, e2), (e1, e3), (e2, e3)))
            ok = cos <= 1e-3
        if not ok:
            return cls.axis_aligned(xyz.min(0), xyz.max(0), from_corners=False)
        axes = np.stack([e1 / lens[0], e2 / lens[1], e3 / lens[2]], axis=1)
        return cls(center=xyz.mean(0), axes=axes, half=lens / 2.0, corners=xyz, from_corners=True)

    @classmethod
    def axis_aligned(cls, lo: Sequence[float], hi: Sequence[float], from_corners: bool = False) -> "OBB":
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = (lo + hi) / 2.0
        h = (hi - lo) / 2.0
        corners = np.array([[lo[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
                            [hi[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]], [lo[0], hi[1], hi[2]],
                            [hi[0], hi[1], hi[2]], [hi[0], lo[1], hi[2]]])
        # edges of this corner order: e1 = +y, e2 = +x, e3 = +z
        return cls(center=c, axes=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T,
                   half=np.array([h[1], h[0], h[2]]), corners=corners, from_corners=from_corners)

    # ------------------------------------------------------------------ basics
    @property
    def lo(self) -> np.ndarray:
        return self.corners.min(0)

    @property
    def hi(self) -> np.ndarray:
        return self.corners.max(0)

    @property
    def dims(self) -> np.ndarray:
        """Edge lengths |e1|, |e2|, |e3|."""
        return 2.0 * self.half

    @property
    def longest_edge(self) -> float:
        return float(2.0 * self.half.max())

    def vertical_axis(self) -> int:
        """Index of the local axis closest to world Z."""
        return int(np.argmax(np.abs(self.axes[2, :])))

    def horizontal_axes(self) -> List[int]:
        v = self.vertical_axis()
        return [k for k in range(3) if k != v]

    def is_upright(self, tol_deg: float = 5.0) -> bool:
        return abs(float(self.axes[2, self.vertical_axis()])) >= math.cos(math.radians(tol_deg))

    def footprint(self) -> np.ndarray:
        """(4, 2) ground-plane corners of an upright box (bottom face), counter-clockwise."""
        pts = self.corners[:, :2]
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        uniq = []
        for i in np.argsort(ang):
            if not any(np.linalg.norm(pts[i] - pts[j]) < 1e-6 for j in uniq):
                uniq.append(i)
        return pts[uniq][:4] if len(uniq) >= 4 else pts[uniq]

    def footprint_area(self) -> float:
        h = sorted(self.horizontal_axes(), key=lambda k: -self.half[k])
        return float(4.0 * self.half[h[0]] * self.half[h[1]])

    def long_axis_yaw(self) -> Tuple[float, float]:
        """(yaw in [0, pi) of the longer horizontal axis, ratio long / short horizontal edge)."""
        h = self.horizontal_axes()
        k = max(h, key=lambda i: self.half[i])
        s = min(h, key=lambda i: self.half[i])
        a = self.axes[:, k]
        return math.atan2(float(a[1]), float(a[0])) % math.pi, float(self.half[k] / max(self.half[s], 1e-6))

    # ------------------------------------------------------------------ queries
    def local(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, float).reshape(-1, 3) - self.center) @ self.axes

    def contains(self, p: np.ndarray, margin: float = 0.0) -> np.ndarray:
        q = np.abs(self.local(p))
        return np.all(q <= self.half + margin, axis=1)

    def point_distance(self, p: np.ndarray) -> np.ndarray:
        q = np.abs(self.local(p))
        return np.linalg.norm(np.maximum(q - self.half, 0.0), axis=1)

    def ray_interval(self, o: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Entry / exit parameters of rays o + t d (broadcast; d need not be unit)."""
        ol = (np.asarray(o, float) - self.center) @ self.axes
        dl = np.asarray(d, float) @ self.axes
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / dl
            t1 = (-self.half - ol) * inv
            t2 = (self.half - ol) * inv
        tnear = np.nanmax(np.minimum(t1, t2), axis=-1)
        tfar = np.nanmin(np.maximum(t1, t2), axis=-1)
        return tnear, tfar

    def sample_points(self, n_per_axis: int = 6) -> np.ndarray:
        f = (np.arange(n_per_axis) + 0.5) / n_per_axis * 2.0 - 1.0
        g = np.stack(np.meshgrid(f, f, f, indexing="ij"), axis=-1).reshape(-1, 3)
        return self.center + (g * self.half) @ self.axes.T

    def edges(self) -> np.ndarray:
        """(12, 2, 3) segment end points."""
        idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        return np.array([[self.corners[a], self.corners[b]] for a, b in idx])

    def to_json(self) -> Dict[str, Any]:
        return {"center": self.center.round(4).tolist(), "axes": self.axes.round(5).tolist(),
                "half": self.half.round(4).tolist(), "from_corners": self.from_corners}


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Distance between segments [p1, q1] and [p2, q2] (broadcast over leading axes)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 1e-12, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0)
        t = (b * s + f) / np.where(e > 1e-12, e, 1.0)
    t_c = np.clip(t, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(t < 0.0, np.clip(-c / np.where(a > 1e-12, a, 1.0), 0.0, 1.0),
                     np.where(t > 1.0, np.clip((b - c) / np.where(a > 1e-12, a, 1.0), 0.0, 1.0), s))
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t_c[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def boxes_overlap(a: OBB, b: OBB) -> bool:
    """Separating-axis test for two oriented boxes."""
    axes = [a.axes[:, i] for i in range(3)] + [b.axes[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(a.axes[:, i], b.axes[:, j])
            if np.linalg.norm(c) > 1e-8:
                axes.append(c / np.linalg.norm(c))
    t = b.center - a.center
    for ax in axes:
        ra = float(np.sum(a.half * np.abs(a.axes.T @ ax)))
        rb = float(np.sum(b.half * np.abs(b.axes.T @ ax)))
        if abs(float(t @ ax)) > ra + rb + 1e-9:
            return False
    return True


def box_distance(a: OBB, b: OBB) -> float:
    """Exact minimum distance between two oriented boxes (0 if they overlap).

    The closest points of two boxes are a corner and a face, or two edges: the minimum over corner-box and
    edge-edge distances.  VSI-Bench builds its distance answer by sampling points inside both boxes and
    taking the closest pair; this is the limit of that procedure."""
    if boxes_overlap(a, b):
        return 0.0
    d = min(float(b.point_distance(a.corners).min()), float(a.point_distance(b.corners).min()))
    ea, eb = a.edges(), b.edges()
    seg = _segment_distance(ea[:, None, 0], ea[:, None, 1], eb[None, :, 0], eb[None, :, 1])
    return float(min(d, float(seg.min())))


def footprint_overlap_fraction(a: OBB, b: OBB, of: str = "smaller") -> float:
    """Shared ground-plane area / the footprint of ``of`` ("a", "b" or the smaller one), by rasterising."""
    lo = np.minimum(a.lo[:2], b.lo[:2])
    hi = np.maximum(a.hi[:2], b.hi[:2])
    n = 60
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    g = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    def inside(box: OBB) -> np.ndarray:
        p = np.concatenate([g, np.full((len(g), 1), box.center[2])], axis=1)
        q = np.abs(box.local(p))
        hz = box.horizontal_axes()
        return np.all(q[:, hz] <= box.half[hz], axis=1)

    ia, ib = inside(a), inside(b)
    both = float((ia & ib).sum())
    na, nb = float(ia.sum()), float(ib.sum())
    if of == "a":
        return both / max(na, 1.0)
    if of == "b":
        return both / max(nb, 1.0)
    return both / max(min(na, nb), 1.0)


def polygon_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, float).reshape(-1, 2)
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of 2D points, counter-clockwise."""
    pts = sorted(set(map(tuple, np.asarray(points, float).round(6))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def offset_convex_polygon(poly: np.ndarray, margin: float, n_arc: int = 6) -> np.ndarray:
    """Outward offset of a convex polygon by ``margin`` (Minkowski sum with a disc, arcs sampled)."""
    p = np.asarray(poly, float)
    out = []
    for i in range(len(p)):
        a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
        n1 = np.array([(b - a)[1], -(b - a)[0]])
        n2 = np.array([(c - b)[1], -(c - b)[0]])
        n1 /= max(np.linalg.norm(n1), 1e-9)
        n2 /= max(np.linalg.norm(n2), 1e-9)
        a1, a2 = math.atan2(n1[1], n1[0]), math.atan2(n2[1], n2[0])
        if a2 < a1:
            a2 += 2 * math.pi
        for t in np.linspace(a1, a2, n_arc):
            out.append(b + margin * np.array([math.cos(t), math.sin(t)]))
    return convex_hull(np.array(out))


def point_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(N,) bool, even-odd rule."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    p = np.asarray(poly, float)
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    xi, yi = p[:, 0][None], p[:, 1][None]
    xj, yj = np.roll(p[:, 0], 1)[None], np.roll(p[:, 1], 1)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
    return (np.sum(cross, axis=1) % 2) == 1


def angle_diff_deg(a: float, b: float, period: float = 360.0) -> float:
    d = (a - b) % period
    return min(d, period - d)
