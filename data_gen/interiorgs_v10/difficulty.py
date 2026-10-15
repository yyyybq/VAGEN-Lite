"""Exact shortest-success-depth (d*) certification for InteriorGS v10 episodes.

Runtime semantics this module certifies against (verified in
``vagen/envs/active_spatial/env.py``, see the report of workstream D):

* Every primitive action is applied with ``ViewManipulator.step``; a
  translation is followed by ``CollisionDetector.check_collision(new, prev)``.
  With ``collision_invalidate_action=True`` (R1 configs) a colliding action
  restores the previous position, i.e. the pose is unchanged (a self-loop
  that still consumes one of the <= 12 primitive steps).  Self-loops can
  never shorten a path, so the BFS ignores them.
* A rotation keeps the position; the runtime still calls
  ``check_collision(p, p)``, which reduces to the point check at ``p``.  Every
  state reached through legal transitions from a point-legal initial pose is
  point-legal, so rotations are always legal (as ``Scene.transition_legal``
  assumes).
* Success (auto-termination) is evaluated on the pose at the END of each
  assistant turn (<= 5 actions), and at the 12th primitive step.  Because a
  turn can always be ended right after any primitive action, the minimum
  number of primitive actions needed to be successful at a turn end equals the
  first-success depth over primitive actions computed here.

``certify_d_star`` runs an exact, level-synchronous, vectorised forward BFS
over the 6 strafe actions with the runtime transition rule.  States are
de-duplicated at float-noise resolution (1e-7 m, 1e-7 deg) by default.  The
coarser ``contract.state_key`` (1 cm / 1 deg) is available as
``key_mode="contract"`` but it merges genuinely different poses that are
1.6-12 mm apart from depth 6 on (measured on synthetic scenes), so it is not
used for certification by default.

Fast paths (all cross-checked against the scalar reference on a random sample
per scene; any disagreement silently falls back to the scalar reference):
  * runtime transition legality: ``scene.transitions_legal`` (workstream G) if
    present, else a vectorised replica of ``CollisionDetector.check_collision``
    built from ``scene.detector``, else per-edge ``scene.transition_legal``;
  * generator legality: ``scene.poses_legal`` / ``scene.is_pose_legal_batch``
    if present, else a vectorised replica of the baseline
    ``Scene.is_pose_legal``, else per-pose ``scene.is_pose_legal``;
  * success: callables with a ``batch(c2ws)`` method are evaluated per BFS
    level; ``make_success_fn(item)`` provides an exact vectorised replica of
    the frozen ``canonical_spatial_task_h1_v1`` Projective / FOV judge that
    defers to ``sampling.score`` whenever any gate quantity is within 1e-6 of
    its threshold.
Every accepted certificate should additionally be replayed with
``replay_certificate`` (pure scalar reference code).
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contract import (
    ACTIONS,
    CAMERA_HEIGHT_M,
    INVERSE_ACTION,
    POS_QUANT_M,
    STEP_ROTATION_DEG,
    STEP_TRANSLATION_M,
    YAW_QUANT_DEG,
    apply_action,
    camera_forward_z,
    heading_turn,
    horizon_locked,
)

N_ACTIONS = len(ACTIONS)
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
IS_TRANSLATION = np.array([a.startswith("move_") for a in ACTIONS], dtype=bool)
DIFFICULTY_VERSION = "interiorgs_v10_dstar_exact_bfs_v1"

EXACT_POS_RES_M = 1e-7
EXACT_YAW_RES_DEG = 1e-7

# Path-preference weights used to choose ONE certificate among all shortest paths.
_UNSAFE_STATE_COST = 1000
_SWITCH_COST = 1


# ---------------------------------------------------------------------------
# Batched pose algebra (bit-compatible with contract.apply_action for moves)
# ---------------------------------------------------------------------------
def _turn_local(action: str) -> np.ndarray:
    ang = math.radians(STEP_ROTATION_DEG) * (-1.0 if action == "turn_left" else 1.0)
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


_TURN_LOCAL = {a: _turn_local(a) for a in ("turn_left", "turn_right")}


def apply_action_batch(P: np.ndarray, R: np.ndarray, action: str) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised ``contract.apply_action`` on positions (N,3) and rotations (N,3,3)."""
    if horizon_locked():
        return _apply_action_batch_horizon_locked(P, R, action)
    if action in ("move_forward", "move_backward"):
        sign = 1.0 if action == "move_forward" else -1.0
        return P + R[:, :, 2] * STEP_TRANSLATION_M * sign, R
    if action in ("move_right", "move_left"):
        sign = 1.0 if action == "move_right" else -1.0
        return P + R[:, :, 0] * STEP_TRANSLATION_M * sign, R
    if action in _TURN_LOCAL:
        return P, np.matmul(R, _TURN_LOCAL[action])
    raise ValueError(f"unknown action {action!r}")


def _apply_action_batch_horizon_locked(P: np.ndarray, R: np.ndarray, action: str) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised ``contract._apply_action_horizon_locked`` (EXPERIMENTAL motion model)."""
    if action in ("move_forward", "move_backward"):
        sign = 1.0 if action == "move_forward" else -1.0
        h = np.zeros((len(R), 3))
        h[:, :2] = R[:, :2, 2]
        return P + h / np.linalg.norm(h, axis=1, keepdims=True) * STEP_TRANSLATION_M * sign, R
    if action in ("move_right", "move_left"):
        sign = 1.0 if action == "move_right" else -1.0
        return P + R[:, :, 0] * STEP_TRANSLATION_M * sign, R
    if action in ("turn_left", "turn_right"):
        return P, np.matmul(heading_turn(STEP_ROTATION_DEG, action)[None], R)
    raise ValueError(f"unknown action {action!r}")


def to_c2w(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = np.zeros((len(P), 4, 4))
    out[:, :3, :3] = R
    out[:, :3, 3] = P
    out[:, 3, 3] = 1.0
    return out


def yaw_deg_batch(R: np.ndarray) -> np.ndarray:
    return np.mod(np.degrees(np.arctan2(R[:, 1, 2], R[:, 0, 2])), 360.0)


def contract_keys(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``contract.state_key`` packed into int64 (validated against it in tests)."""
    kx = np.rint(P[:, 0] / POS_QUANT_M).astype(np.int64)
    ky = np.rint(P[:, 1] / POS_QUANT_M).astype(np.int64)
    n_yaw = int(round(360.0 / YAW_QUANT_DEG))
    kyaw = np.mod(np.rint(yaw_deg_batch(R) / YAW_QUANT_DEG).astype(np.int64), n_yaw)
    off = 1 << 21
    if n_yaw > 512 or len(kx) and (np.abs(kx).max() >= off or np.abs(ky).max() >= off):
        raise ValueError("pose outside the packable contract-key range")
    return ((kx + off) << 31) | ((ky + off) << 9) | kyaw


def exact_keys(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Float-noise resolution keys (1e-7 m, 1e-7 deg) as hashable np.void items."""
    kx = np.rint(P[:, 0] / EXACT_POS_RES_M).astype(np.int64)
    ky = np.rint(P[:, 1] / EXACT_POS_RES_M).astype(np.int64)
    n_yaw = int(round(360.0 / EXACT_YAW_RES_DEG))
    kyaw = np.mod(np.rint(yaw_deg_batch(R) / EXACT_YAW_RES_DEG).astype(np.int64), n_yaw)
    arr = np.ascontiguousarray(np.stack([kx, ky, kyaw], axis=1))
    return arr.view(np.dtype((np.void, 24))).ravel()


def make_keys(P: np.ndarray, R: np.ndarray, mode: str = "exact") -> np.ndarray:
    if mode == "exact":
        return exact_keys(P, R)
    if mode == "contract":
        return contract_keys(P, R)
    raise ValueError(mode)


def _split(c2w: Any) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c2w, dtype=np.float64)
    if c.ndim == 2:
        c = c[None]
    return np.ascontiguousarray(c[:, :3, 3]), np.ascontiguousarray(c[:, :3, :3])


# ---------------------------------------------------------------------------
# Legality: runtime transitions + generator-safe poses
# ---------------------------------------------------------------------------
class _DetectorReplica:
    """Vectorised replica of CollisionDetector.check_collision and the baseline
    Scene.is_pose_legal.  Built from the scene's live detector."""

    def __init__(self, scene: Any) -> None:
        det = scene.detector
        self.boundary = bool(getattr(det, "enable_boundary_collision", True))
        self.floor = float(det.floor_height)
        self.ceiling = float(det.ceiling_height)
        self.objects_on = bool(getattr(det, "enable_object_collision", True))
        loaded = bool(getattr(det, "scene_loaded", True))
        boxes = list(det.object_boxes) if (self.objects_on and loaded) else []
        self.bmin = np.array([b.min_point for b in boxes], dtype=np.float64).reshape(-1, 3)
        self.bmax = np.array([b.max_point for b in boxes], dtype=np.float64).reshape(-1, 3)
        all_segs = list(det.wall_segments or [])
        seg_a = np.array([np.asarray(a, dtype=np.float64)[:2] for a, _ in all_segs]).reshape(-1, 2)
        seg_b = np.array([np.asarray(b, dtype=np.float64)[:2] for _, b in all_segs]).reshape(-1, 2)
        # scene.wall_distance always uses every clearance segment (the room edges; the same list as the
        # collision segments under the "room_edges" wall model) ...
        clear = list(getattr(scene, "clearance_segments", None) or all_segs)
        self.swa = np.array([np.asarray(a, dtype=np.float64)[:2] for a, _ in clear]).reshape(-1, 2)
        self.swb = np.array([np.asarray(b, dtype=np.float64)[:2] for _, b in clear]).reshape(-1, 2)
        # ... the detector only when object collision is enabled
        self.wa = seg_a if self.objects_on else seg_a[:0]
        self.wb = seg_b if self.objects_on else seg_b[:0]
        self.cam_r = float(det.camera_radius)
        self.min_wall = float(getattr(scene, "min_wall_distance", 0.5))
        self.polys = [np.asarray(r.polygon, dtype=np.float64) for r in getattr(scene, "rooms", [])]
        self.chunk = max(64, int(1_000_000 // max(1, len(self.bmin))))

    # -- point checks (CollisionDetector without previous_position) --------
    def _point_collide(self, P: np.ndarray) -> np.ndarray:
        out = np.zeros(len(P), dtype=bool)
        if self.boundary:
            out |= (P[:, 2] < self.floor) | (P[:, 2] > self.ceiling)
        if len(self.bmin):
            for s in range(0, len(P), self.chunk):
                p = P[s:s + self.chunk, None, :]
                inside = np.all(p >= self.bmin[None], axis=2) & np.all(p <= self.bmax[None], axis=2)
                out[s:s + self.chunk] |= inside.any(axis=1)
        if len(self.wa):
            out |= self._det_wall_dist(P[:, :2]) < self.cam_r
        return out

    def _det_wall_dist(self, Q: np.ndarray) -> np.ndarray:
        """min over walls of CollisionDetector._point_to_segment_distance_2d."""
        best = np.full(len(Q), np.inf)
        for a, b in zip(self.wa, self.wb):
            v = b - a
            w = Q - a
            c1 = w[:, 0] * v[0] + w[:, 1] * v[1]
            c2 = v[0] * v[0] + v[1] * v[1]
            d_a = np.hypot(Q[:, 0] - a[0], Q[:, 1] - a[1])
            d_b = np.hypot(Q[:, 0] - b[0], Q[:, 1] - b[1])
            with np.errstate(divide="ignore", invalid="ignore"):
                t = c1 / c2
            px = a[0] + t * v[0]
            py = a[1] + t * v[1]
            d_p = np.hypot(Q[:, 0] - px, Q[:, 1] - py)
            d = np.where(c1 <= 0, d_a, np.where(c2 <= c1, d_b, d_p))
            best = np.minimum(best, d)
        return best

    def transitions_collide(self, Pf: np.ndarray, Pt: np.ndarray) -> np.ndarray:
        """check_collision(Pt, previous_position=Pf).has_collision, vectorised."""
        out = self._point_collide(Pt)
        if len(self.bmin):
            for s in range(0, len(Pt), self.chunk):
                out[s:s + self.chunk] |= self._seg_boxes(Pf[s:s + self.chunk], Pt[s:s + self.chunk])
        if len(self.wa):
            out |= self._seg_walls(Pf[:, :2], Pt[:, :2])
        return out

    def _seg_boxes(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d = B - A
        length = np.linalg.norm(d, axis=1)
        degenerate = length < 1e-10
        safe_len = np.where(degenerate, 1.0, length)
        dirn = d / safe_len[:, None]
        n, m = len(A), len(self.bmin)
        tmin = np.zeros((n, m))
        tmax = np.broadcast_to(length[:, None], (n, m)).copy()
        fail = np.zeros((n, m), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(3):
                par = (np.abs(dirn[:, i]) < 1e-10)[:, None]
                s_i = A[:, i][:, None]
                lo_b, hi_b = self.bmin[None, :, i], self.bmax[None, :, i]
                fail |= par & ((s_i < lo_b) | (s_i > hi_b))
                di = dirn[:, i][:, None]
                t1 = (lo_b - s_i) / di
                t2 = (hi_b - s_i) / di
                tmin = np.where(par, tmin, np.maximum(tmin, np.minimum(t1, t2)))
                tmax = np.where(par, tmax, np.minimum(tmax, np.maximum(t1, t2)))
            hit = (~fail) & (tmin <= tmax) & ((tmin / safe_len[:, None]) < 1.0)
        if degenerate.any():
            a = A[degenerate][:, None, :]
            hit[degenerate] = np.all(a >= self.bmin[None], axis=2) & np.all(a <= self.bmax[None], axis=2)
        return hit.any(axis=1)

    def _seg_walls(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        out = np.zeros(len(A), dtype=bool)
        d1 = B - A
        for a, b in zip(self.wa, self.wb):
            d2 = b - a
            cross = d1[:, 0] * d2[1] - d1[:, 1] * d2[0]
            ok = np.abs(cross) >= 1e-10
            diff = a - A
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (diff[:, 0] * d2[1] - diff[:, 1] * d2[0]) / cross
                u = (diff[:, 0] * d1[:, 1] - diff[:, 1] * d1[:, 0]) / cross
            out |= ok & (t > 0) & (t < 1.0) & (u >= 0) & (u <= 1)
        return out

    # -- baseline Scene.is_pose_legal ----------------------------------------
    def poses_legal(self, P: np.ndarray, R: np.ndarray) -> np.ndarray:
        ok = (np.abs(P[:, 2] - CAMERA_HEIGHT_M) <= 1e-6) & (np.abs(R[:, 2, 2] - camera_forward_z()) <= 1e-6)
        ok &= self._in_any_room(P[:, 0], P[:, 1])
        if len(self.swa):
            ok &= self._scene_wall_dist(P[:, :2]) >= self.min_wall
        ok &= ~self._point_collide(P)
        return ok

    def _in_any_room(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        any_in = np.zeros(len(x), dtype=bool)
        for poly in self.polys:
            inside = np.zeros(len(x), dtype=bool)
            n = len(poly)
            j = n - 1
            for i in range(n):
                xi, yi = float(poly[i][0]), float(poly[i][1])
                xj, yj = float(poly[j][0]), float(poly[j][1])
                cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
                inside ^= cond
                j = i
            any_in |= inside
        return any_in

    def _scene_wall_dist(self, Q: np.ndarray) -> np.ndarray:
        """min over walls of scene.point_segment_distance."""
        best = np.full(len(Q), np.inf)
        for a, b in zip(self.swa, self.swb):
            ab = b - a
            denom = float(ab @ ab)
            if denom < 1e-12:
                t = np.zeros(len(Q))
            else:
                t = np.clip(((Q[:, 0] - a[0]) * ab[0] + (Q[:, 1] - a[1]) * ab[1]) / denom, 0.0, 1.0)
            best = np.minimum(best, np.hypot(Q[:, 0] - (a[0] + t * ab[0]), Q[:, 1] - (a[1] + t * ab[1])))
        return best


def _as_bool_array(res: Any, n: int) -> np.ndarray:
    if isinstance(res, tuple) and len(res) == 2 and np.ndim(res[0]) == 1 and len(res[0]) == n:
        res = res[0]
    if isinstance(res, list) and res and isinstance(res[0], tuple):
        res = [bool(r[0]) for r in res]
    arr = np.asarray(res, dtype=bool).reshape(-1)
    if arr.shape != (n,):
        raise ValueError("batch legality returned an unexpected shape")
    return arr


class FastGeometry:
    """Batched legality with automatic backend selection and self-validation."""

    def __init__(self, scene: Any, validate_samples: int = 192, seed: int = 0) -> None:
        self.scene = scene
        self.info: Dict[str, Any] = {}
        self._scalar_pose_cache: Dict[Tuple[float, float, float], bool] = {}
        self._scalar_trans_cache: Dict[Tuple[float, ...], bool] = {}
        try:
            self._replica: Optional[_DetectorReplica] = _DetectorReplica(scene)
        except Exception as exc:  # detector missing / different structure
            self._replica = None
            self.info["replica_error"] = repr(exc)
        rng = np.random.default_rng(seed)
        poses_P, poses_R, edges_A, edges_B, edges_R = self._validation_sample(rng, validate_samples)
        self._trans_impl, self.info["transition_backend"] = self._pick_transition_backend(edges_A, edges_B, edges_R)
        self._pose_impl, self.info["pose_backend"] = self._pick_pose_backend(poses_P, poses_R)

    # ------------------------------------------------------------- factory
    @classmethod
    def for_scene(cls, scene: Any) -> "FastGeometry":
        geom = getattr(scene, "_v10_fast_geometry", None)
        if isinstance(geom, FastGeometry) and geom.scene is scene:
            return geom
        geom = cls(scene)
        try:
            setattr(scene, "_v10_fast_geometry", geom)
        except Exception:
            pass
        return geom

    # ------------------------------------------------------- public API
    def transitions_legal(self, Pf: np.ndarray, Pt: np.ndarray, R: np.ndarray) -> np.ndarray:
        if len(Pf) == 0:
            return np.zeros(0, dtype=bool)
        return self._trans_impl(np.asarray(Pf, float), np.asarray(Pt, float), np.asarray(R, float))

    def poses_legal(self, P: np.ndarray, R: np.ndarray) -> np.ndarray:
        if len(P) == 0:
            return np.zeros(0, dtype=bool)
        return self._pose_impl(np.asarray(P, float), np.asarray(R, float))

    def runtime_points_legal(self, P: np.ndarray) -> np.ndarray:
        """CollisionDetector.check_collision(p) without previous position."""
        P = np.asarray(P, float)
        if self._replica is not None and self.info.get("replica_transitions_validated"):
            return ~self._replica._point_collide(P)
        det = getattr(self.scene, "detector", None)
        if det is None:  # degenerate: a zero-length transition is the point check
            return self.transitions_legal(P, P, np.repeat(np.eye(3)[None], len(P), axis=0))
        return np.array([not det.check_collision(p).has_collision for p in P], dtype=bool)

    # ---------------------------------------------------------- backends
    def _scalar_trans(self, Pf: np.ndarray, Pt: np.ndarray, R: np.ndarray) -> np.ndarray:
        out = np.empty(len(Pf), dtype=bool)
        for i in range(len(Pf)):
            key = tuple(np.round(np.concatenate([Pf[i], Pt[i]]), 9).tolist())
            v = self._scalar_trans_cache.get(key)
            if v is None:
                a = np.eye(4)
                a[:3, :3] = R[i]
                a[:3, 3] = Pf[i]
                b = a.copy()
                b[:3, 3] = Pt[i]
                v = bool(self.scene.transition_legal(a, b)[0])
                self._scalar_trans_cache[key] = v
            out[i] = v
        return out

    def _scalar_pose(self, P: np.ndarray, R: np.ndarray) -> np.ndarray:
        out = np.empty(len(P), dtype=bool)
        for i in range(len(P)):
            # legality does not depend on yaw at pitch 0 (sampling.floor_grid relies on it)
            key = tuple(np.round(P[i], 9).tolist())
            v = self._scalar_pose_cache.get(key)
            if v is None:
                c = np.eye(4)
                c[:3, :3] = R[i]
                c[:3, 3] = P[i]
                v = bool(self.scene.is_pose_legal(c)[0])
                self._scalar_pose_cache[key] = v
            out[i] = v
        return out

    def _pick_transition_backend(self, A, B, Rs):
        ref = self._scalar_trans(A, B, Rs) if len(A) else None
        r_impl = None
        if self._replica is not None:
            rep = self._replica

            def r_impl(Pf, Pt, R, rep=rep):
                return ~rep.transitions_collide(Pf, Pt)
            try:
                got = r_impl(A, B, Rs)
                if ref is None or np.array_equal(got, ref):
                    self.info["replica_transitions_validated"] = True
                else:
                    self.info["replica_transition_mismatch"] = int((got != ref).sum())
                    r_impl = None
            except Exception as exc:
                self.info["replica_transition_error"] = repr(exc)
                r_impl = None
        fn = getattr(self.scene, "transitions_legal", None)
        if callable(fn):
            # workstream G API: transitions_legal(from_xy (N,2), to_xy (N,2)) at z = 1.5 m
            def g_impl(Pf, Pt, R, fn=fn, fallback=r_impl):
                if not (np.all(Pf[:, 2] == CAMERA_HEIGHT_M) and np.all(Pt[:, 2] == CAMERA_HEIGHT_M)):
                    return fallback(Pf, Pt, R) if fallback is not None else self._scalar_trans(Pf, Pt, R)
                return _as_bool_array(fn(Pf[:, :2], Pt[:, :2]), len(Pf))
            try:
                if ref is None or np.array_equal(g_impl(A, B, Rs), ref):
                    return g_impl, "scene.transitions_legal"
                self.info["scene_batch_transition_mismatch"] = True
            except Exception as exc:
                self.info["scene_batch_transition_error"] = repr(exc)
        if r_impl is not None:
            return r_impl, "detector_replica"
        return self._scalar_trans, "scalar"

    def _pick_pose_backend(self, P, Rs):
        ref = self._scalar_pose(P, Rs) if len(P) else None
        for name in ("legal_mask",):
            fn = getattr(self.scene, name, None)
            if not callable(fn):
                continue

            # workstream G API: legal_mask(xy (N,2)) == is_pose_legal of the pitch-0, z = 1.5 m pose
            def g_impl(Pq, R, fn=fn):
                canon = (np.abs(Pq[:, 2] - CAMERA_HEIGHT_M) <= 1e-6) & (np.abs(R[:, 2, 2] - camera_forward_z()) <= 1e-6)
                out = np.zeros(len(Pq), dtype=bool)
                if canon.any():
                    out[canon] = _as_bool_array(fn(Pq[canon, :2]), int(canon.sum()))
                return out
            try:
                if ref is None or np.array_equal(g_impl(P, Rs), ref):
                    return g_impl, f"scene.{name}"
                self.info[f"scene_{name}_mismatch"] = True
            except Exception as exc:
                self.info[f"scene_{name}_error"] = repr(exc)
        if self._replica is not None:
            rep = self._replica

            def r_impl(Pq, R, rep=rep):
                return rep.poses_legal(Pq, R)
            try:
                if ref is None or np.array_equal(r_impl(P, Rs), ref):
                    return r_impl, "baseline_replica"
                self.info["replica_pose_mismatch"] = int((r_impl(P, Rs) != ref).sum())
            except Exception as exc:
                self.info["replica_pose_error"] = repr(exc)
        return self._scalar_pose, "scalar"

    def _validation_sample(self, rng: np.random.Generator, n: int):
        rooms = list(getattr(self.scene, "rooms", []) or [])
        if not rooms or n <= 0:
            z = np.zeros((0, 3))
            return z, np.zeros((0, 3, 3)), z, z, np.zeros((0, 3, 3))
        lo = np.min([r.polygon.min(axis=0) for r in rooms], axis=0) - 0.3
        hi = np.max([r.polygon.max(axis=0) for r in rooms], axis=0) + 0.3
        xy = rng.uniform(lo, hi, size=(n, 2))
        # add points straddling the generator wall clearance and the camera radius
        if self._replica is not None and len(self._replica.swa):
            k = n // 2
            idx = rng.integers(len(self._replica.swa), size=k)
            t = rng.uniform(0, 1, size=k)
            a, b = self._replica.swa[idx], self._replica.swb[idx]
            base = a + t[:, None] * (b - a)
            nrm = np.stack([-(b - a)[:, 1], (b - a)[:, 0]], axis=1)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            off = rng.choice([0.14, 0.16, 0.49, 0.51, -0.2, 0.3], size=k)[:, None]
            xy = np.concatenate([xy, base + nrm * off])
        yaw = rng.uniform(0, 2 * math.pi, size=len(xy))
        P = np.concatenate([xy, np.full((len(xy), 1), CAMERA_HEIGHT_M)], axis=1)
        from .contract import pose_from_xy_yaw
        R = np.stack([pose_from_xy_yaw(0.0, 0.0, float(t))[:3, :3] for t in yaw])
        step = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1) * STEP_TRANSLATION_M
        return P, R, P.copy(), P + step, R


# ---------------------------------------------------------------------------
# Success oracles
# ---------------------------------------------------------------------------
class SuccessOracle:
    """Wraps a success function: scalar or batched evaluation, cached by exact key."""

    def __init__(self, fn: Callable[[np.ndarray], Any], batch_fn: Optional[Callable] = None,
                 exact_fn: Optional[Callable] = None, cache: Optional[bool] = None) -> None:
        self.fn = fn
        self.batch_fn = batch_fn
        self.exact_fn = exact_fn or fn
        # A vectorised judge is cheaper than a Python dict lookup per state, so by
        # default only scalar judges are cached (by exact state key).
        if cache is None:
            cache = batch_fn is None
        self._cache: Optional[Dict[Any, bool]] = {} if cache else None
        self.evaluations = 0
        self.cache_hits = 0

    def __call__(self, c2w: np.ndarray) -> bool:
        return bool(self.batch(np.asarray(c2w, float)[None])[0])

    def exact(self, c2w: np.ndarray) -> bool:
        return _truthy(self.exact_fn(np.asarray(c2w, float)))

    def batch(self, c2ws: np.ndarray, keys: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(c2ws)
        out = np.zeros(n, dtype=bool)
        todo = np.arange(n)
        if self._cache is not None and keys is not None:
            klist = keys.tolist()
            known = [self._cache.get(k) for k in klist]
            hit = np.array([v is not None for v in known], dtype=bool)
            if hit.any():
                out[hit] = [bool(v) for v, h in zip(known, hit) if h]
                self.cache_hits += int(hit.sum())
            todo = np.flatnonzero(~hit)
        if len(todo):
            if self.batch_fn is not None:
                vals = np.asarray(self.batch_fn(c2ws[todo]), dtype=bool).reshape(-1)
            else:
                vals = np.array([_truthy(self.fn(c2ws[i])) for i in todo], dtype=bool)
            out[todo] = vals
            self.evaluations += len(todo)
            if self._cache is not None and keys is not None:
                for k, v in zip(keys[todo].tolist(), vals.tolist()):
                    self._cache[k] = bool(v)
        return out


def _truthy(v: Any) -> bool:
    if isinstance(v, dict):
        return bool(v.get("success"))
    return bool(v)


def as_oracle(success_fn: Any) -> SuccessOracle:
    if isinstance(success_fn, SuccessOracle):
        return success_fn
    batch = getattr(success_fn, "batch", None)
    exact = getattr(success_fn, "exact", None)
    return SuccessOracle(success_fn, batch_fn=batch if callable(batch) else None,
                         exact_fn=exact if callable(exact) else None)


class CanonicalSuccess:
    """Vectorised replica of canonical_spatial_task_h1_v1 success (Projective, FOV).

    ``batch`` is exact: any pose whose gate quantities lie within ``tol`` of a
    decision threshold is re-scored with the frozen ``sampling.score``.
    ``exact`` always uses ``sampling.score``.
    """

    TOL = 1e-6      # pixel-valued quantities (float error is ~1e-9 px)
    TOL_Z = 1e-10   # depth in metres (float error is ~1e-15 m)

    def __init__(self, item: Dict[str, Any]) -> None:
        from vagen.envs.active_spatial.canonical_camera import (
            CANONICAL_CAMERA_H1_RESIZE_V1, build_canonical_camera)
        from vagen.envs.active_spatial.canonical_task_metrics import (
            CANONICAL_TASK_METRIC_VERSION, FOV_CENTER_MARGIN_FRACTION, FOV_MIN_AREA_RATIO,
            FOV_MIN_INSIDE_FRACTION, PROJECTIVE_MIN_INSIDE_FRACTION, PROJECTIVE_MIN_MARGIN_PX)
        from vagen.envs.active_spatial.visual_bbox_metrics import (
            _as_array, _bbox_corners, _extract_objects, _match_object)

        if item.get("canonical_task_metric_version") != CANONICAL_TASK_METRIC_VERSION:
            raise ValueError("CanonicalSuccess only supports canonical_spatial_task_h1_v1 items")
        self.task_type = str(item.get("task_type"))
        if self.task_type not in ("projective_relations", "fov_inclusion"):
            raise ValueError(f"unsupported task_type {self.task_type!r}")
        self.item = item
        cam = build_canonical_camera(K_native=item["init_camera"]["intrinsics"], render_size=(256, 256),
                                     transform="resize", c2w=np.eye(4), item=item,
                                     camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1)
        self.K = np.asarray(cam.K_effective, dtype=np.float64)
        self.W, self.H = (int(v) for v in cam.render_resolution)
        params = item.get("target_region", {}).get("params", {})
        objects = _extract_objects({"_target_object": item.get("target_object")})
        sel = (_match_object(objects, center=params.get("object_a_center"), fallback_index=0),
               _match_object(objects, center=params.get("object_b_center"), fallback_index=1))
        self.valid = all(o is not None for o in sel)
        self.corners, self.centers = [], []
        for o in sel:
            if o is None:
                continue
            bmin, bmax = _as_array(o.get("bbox_min")), _as_array(o.get("bbox_max"))
            if bmin is None or bmax is None:
                self.valid = False
                continue
            self.corners.append(_bbox_corners(bmin, bmax))
            self.centers.append(_as_array(o.get("center"), (bmin + bmax) / 2.0)[:3])
        self.relation = params.get("relation", "left")
        self.min_area = FOV_MIN_AREA_RATIO
        self.proj_inside = PROJECTIVE_MIN_INSIDE_FRACTION
        self.proj_margin = PROJECTIVE_MIN_MARGIN_PX
        self.fov_inside = FOV_MIN_INSIDE_FRACTION
        self.fov_boundary = min(self.W, self.H) * FOV_CENTER_MARGIN_FRACTION
        self.fallbacks = 0
        self.evaluations = 0

    # -- exact reference ----------------------------------------------------
    def exact(self, c2w: np.ndarray) -> bool:
        from .sampling import score
        return bool(score(self.item, np.asarray(c2w, dtype=float))["success"])

    def __call__(self, c2w: np.ndarray) -> bool:
        return bool(self.batch(np.asarray(c2w, float)[None])[0])

    # -- vectorised ---------------------------------------------------------
    def _project(self, w2c: np.ndarray, corners: np.ndarray, center: np.ndarray):
        tol, K, W, H = self.TOL, self.K, self.W, self.H
        Rw, tw = w2c[:, :3, :3], w2c[:, :3, 3]
        pc = np.matmul(corners[None], Rw.transpose(0, 2, 1)) + tw[:, None, :]  # (N,8,3)
        cc = np.matmul(Rw, center) + tw
        z = pc[:, :, 2]
        valid = z > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, :, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, :, 1] / z + K[1, 2]
        anyv = valid.any(axis=1)
        x1 = np.where(valid, u, np.inf).min(axis=1)
        y1 = np.where(valid, v, np.inf).min(axis=1)
        x2 = np.where(valid, u, -np.inf).max(axis=1)
        y2 = np.where(valid, v, -np.inf).max(axis=1)
        cx1, cy1 = np.maximum(0.0, x1), np.maximum(0.0, y1)
        cx2, cy2 = np.minimum(float(W - 1), x2), np.minimum(float(H - 1), y2)
        cw, ch = np.maximum(0.0, cx2 - cx1), np.maximum(0.0, cy2 - cy1)
        area = np.where(anyv, cw * ch, 0.0)
        cw, ch = np.where(anyv, cw, 0.0), np.where(anyv, ch, 0.0)
        cz = cc[:, 2]
        front = cz > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            cu = np.where(front, K[0, 0] * cc[:, 0] / cz + K[0, 2], np.nan)
            cv = np.where(front, K[1, 1] * cc[:, 1] / cz + K[1, 2], np.nan)
        in_frame = front & (cu >= 0.0) & (cu <= W - 1) & (cv >= 0.0) & (cv <= H - 1)
        visible = anyv & (area > 1.0) & (in_frame | ((cw > 2.0) & (ch > 2.0)))
        ratio = area / max(W * H, 1)
        raw = np.where(anyv, np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inside = np.where(raw > 1e-8, area / raw, 0.0)
        near = (np.abs(z - 1e-6) < self.TOL_Z).any(axis=1) | (np.abs(cz - 1e-6) < self.TOL_Z)
        near |= anyv & ((np.abs(area - 1.0) < tol) | (np.abs(cw - 2.0) < tol) | (np.abs(ch - 2.0) < tol))
        near |= front & ((np.abs(cu) < tol) | (np.abs(cu - (W - 1)) < tol) | (np.abs(cv) < tol)
                         | (np.abs(cv - (H - 1)) < tol))
        near |= anyv & ((np.abs(ratio - self.min_area) < 1e-12) | (np.abs(raw - 1e-8) < 1e-12))
        return dict(front=front, visible=visible, ratio=ratio, inside=inside, cu=cu, cv=cv, near=near)

    def batch(self, c2ws: np.ndarray) -> np.ndarray:
        c2ws = np.asarray(c2ws, dtype=np.float64).reshape(-1, 4, 4)
        n = len(c2ws)
        self.evaluations += n
        if not self.valid or n == 0:
            return np.zeros(n, dtype=bool)
        w2c = np.linalg.inv(c2ws)
        a = self._project(w2c, self.corners[0], self.centers[0])
        b = self._project(w2c, self.corners[1], self.centers[1])
        tol = self.TOL
        base = a["front"] & b["front"] & a["visible"] & b["visible"]
        base &= (a["ratio"] >= self.min_area) & (b["ratio"] >= self.min_area)
        near = a["near"] | b["near"]
        if self.task_type == "projective_relations":
            ok = base & (a["inside"] >= self.proj_inside) & (b["inside"] >= self.proj_inside)
            margin = (b["cu"] - a["cu"]) if self.relation == "left" else (a["cu"] - b["cu"])
            ok &= np.nan_to_num(margin, nan=-np.inf) >= self.proj_margin
            near |= (np.abs(a["inside"] - self.proj_inside) < tol) | (np.abs(b["inside"] - self.proj_inside) < tol)
            near |= np.nan_to_num(np.abs(margin - self.proj_margin) < tol, nan=False)
        else:
            ok = base & (a["inside"] >= self.fov_inside) & (b["inside"] >= self.fov_inside)
            W, H = self.W, self.H

            def cmargin(p):
                return np.minimum(np.minimum(p["cu"], W - p["cu"]), np.minimum(p["cv"], H - p["cv"]))
            m = np.minimum(cmargin(a), cmargin(b))
            ok &= np.nan_to_num(m, nan=-np.inf) >= self.fov_boundary
            near |= (np.abs(a["inside"] - self.fov_inside) < tol) | (np.abs(b["inside"] - self.fov_inside) < tol)
            near |= np.nan_to_num(np.abs(m - self.fov_boundary) < tol, nan=False)
        idx = np.flatnonzero(near)
        if len(idx):
            self.fallbacks += len(idx)
            ok = ok.copy()
            for i in idx:
                ok[i] = self.exact(c2ws[i])
        return ok


def make_success_fn(item: Dict[str, Any]) -> SuccessOracle:
    """Cached success oracle for a canonical v1 Projective/FOV item."""
    fast = CanonicalSuccess(item)
    return SuccessOracle(fast, batch_fn=fast.batch, exact_fn=fast.exact)


# ---------------------------------------------------------------------------
# BFS certification
# ---------------------------------------------------------------------------
class _Level:
    __slots__ = ("P", "R", "keys", "parent", "action", "_index")

    def __init__(self, P, R, keys, parent, action):
        self.P, self.R, self.keys, self.parent, self.action = P, R, keys, parent, action
        self._index: Optional[Dict[Any, int]] = None

    def __len__(self) -> int:
        return len(self.keys)

    def index(self) -> Dict[Any, int]:
        if self._index is None:
            self._index = {k: i for i, k in enumerate(self.keys.tolist())}
        return self._index


def _expand(level: _Level, seen: set, geom: FastGeometry, key_mode: str,
            state_budget: Optional[int]) -> Optional[_Level]:
    """All legal successors of ``level`` not in ``seen``; representative = first
    legal incoming edge in (parent, action) order.  Returns None if the number
    of distinct new keys exceeds ``state_budget``."""
    n = len(level)
    Ps, Rs = [], []
    for a in ACTIONS:
        p, r = apply_action_batch(level.P, level.R, a)
        Ps.append(p)
        Rs.append(np.broadcast_to(r, (n, 3, 3)))
    P = np.stack(Ps, axis=1).reshape(-1, 3)
    R = np.stack(Rs, axis=1).reshape(-1, 3, 3)
    parent = np.repeat(np.arange(n), N_ACTIONS)
    action = np.tile(np.arange(N_ACTIONS), n)
    keys = make_keys(P, R, key_mode)
    klist = keys.tolist()
    fresh = np.fromiter((k not in seen for k in klist), dtype=bool, count=len(klist))
    eidx = np.flatnonzero(fresh)
    if len(eidx) == 0:
        return _Level(np.zeros((0, 3)), np.zeros((0, 3, 3)), keys[:0], np.zeros(0, int), np.zeros(0, int))
    ek = keys[eidx]
    uniq, inv = np.unique(ek, return_inverse=True)
    inv = inv.reshape(-1)
    if state_budget is not None and len(uniq) > state_budget:
        return None
    # rank of each edge within its key group (edges are already in BFS order)
    order = np.argsort(inv, kind="stable")
    grp_sorted = inv[order]
    starts = np.r_[0, np.flatnonzero(np.diff(grp_sorted)) + 1]
    rank_sorted = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank = np.empty_like(rank_sorted)
    rank[order] = rank_sorted
    rep = np.full(len(uniq), -1, dtype=np.int64)
    is_tr = IS_TRANSLATION[action[eidx]]
    for r in range(int(rank.max()) + 1):
        cand = np.flatnonzero((rank == r) & (rep[inv] < 0))
        if len(cand) == 0:
            if (rep >= 0).all():
                break
            continue
        legal = np.ones(len(cand), dtype=bool)
        tr = is_tr[cand]
        if tr.any():
            ce = eidx[cand[tr]]
            legal[tr] = geom.transitions_legal(level.P[parent[ce]], P[ce], R[ce])
        good = cand[legal]
        rep[inv[good]] = good
    resolved = np.flatnonzero(rep >= 0)
    rep_e = rep[resolved]
    rep_e = rep_e[np.argsort(rep_e)]  # BFS order
    ge = eidx[rep_e]
    return _Level(P[ge], R[ge], keys[ge], parent[ge], action[ge])


def certify_d_star(scene: Any, success_fn: Any, init_c2w: Any, d_max: int = 8,
                   max_states: int = 200000, *, upper_bound: Optional[int] = None,
                   hint_actions: Optional[Sequence[str]] = None,
                   terminal_ok: Optional[Callable[[np.ndarray], bool]] = None,
                   prefer_generator_safe: bool = True, max_terminal_checks: int = 8,
                   key_mode: str = "exact", geometry: Optional[FastGeometry] = None) -> Dict[str, Any]:
    """Exact first-success depth from ``init_c2w`` under the runtime rule.

    Returns dict(d_star, lower_bound_complete, certificate, expansions, states, ...):
      * d_star: first depth with a successful state, or None if none within
        ``d_max`` (or the state cap was hit).
      * lower_bound_complete: True iff every state at depth < d_star (or
        <= d_max when d_star is None) was enumerated.
      * lower_bound: largest depth L such that no state at depth <= L succeeds.
      * certificate: one shortest action list (preferring generator-safe
        states, then fewest action switches, then a terminal accepted by
        ``terminal_ok``).
    ``upper_bound``/``hint_actions`` (e.g. a reverse-search path) let the BFS stop
    one level early: if no success exists at depth <= upper_bound - 1 the
    verified hint path is a shortest certificate.
    """
    t0 = time.perf_counter()
    oracle = as_oracle(success_fn)
    geom = geometry or FastGeometry.for_scene(scene)
    P0, R0 = _split(init_c2w)
    k0 = make_keys(P0, R0, key_mode)
    out: Dict[str, Any] = {
        "d_star": None, "lower_bound_complete": False, "certificate": [], "expansions": 0, "states": 1,
        "lower_bound": 0, "status": "", "certificate_source": None, "certificate_generator_safe": None,
        "certificate_unsafe_states": None, "certificate_switches": None, "terminal_ok": None,
        "success_states_at_d_star": 0, "level_sizes": [1], "d_max": int(d_max), "key_mode": key_mode,
        "version": DIFFICULTY_VERSION,
    }

    def finish(status: str) -> Dict[str, Any]:
        out["status"] = status
        out["seconds"] = time.perf_counter() - t0
        return out

    if not bool(geom.runtime_points_legal(P0)[0]):
        return finish("init_runtime_illegal")
    if bool(oracle.batch(to_c2w(P0, R0), k0)[0]):
        out.update(d_star=0, lower_bound_complete=True)
        return finish("init_already_success")

    # ---- optional hint (upper bound + path) -----------------------------
    hint: Optional[List[str]] = None
    if hint_actions is not None and len(hint_actions) > 0:
        hint, why = _check_hint(P0, R0, list(hint_actions), oracle, geom, key_mode)
        out["hint_status"] = why
    elif upper_bound is not None:
        out["hint_status"] = "upper_bound_without_path_ignored"
    limit = int(d_max)
    if hint is not None:
        limit = min(limit, len(hint) - 1)

    levels: List[_Level] = [_Level(P0, R0, k0, np.zeros(1, int) - 1, np.zeros(1, int) - 1)]
    seen = set(k0.tolist())
    found: Optional[np.ndarray] = None
    complete = True
    for depth in range(1, limit + 1):
        frontier = levels[-1]
        if len(frontier) == 0:
            break
        new = _expand(frontier, seen, geom, key_mode, max_states - len(seen))
        if new is None:
            complete = False
            break
        out["expansions"] += len(frontier)
        levels.append(new)
        seen.update(new.keys.tolist())
        out["level_sizes"].append(len(new))
        if len(new):
            succ = oracle.batch(to_c2w(new.P, new.R), new.keys)
            if succ.any():
                found = np.flatnonzero(succ)
                break
        out["lower_bound"] = depth
    out["states"] = len(seen)

    if found is not None:
        d = len(levels) - 1
        out.update(d_star=d, lower_bound_complete=True, success_states_at_d_star=int(len(found)))
        cert = _select_certificate(levels, found, geom, key_mode, terminal_ok, prefer_generator_safe,
                                   max_terminal_checks)
        out.update(cert)
        out["certificate_source"] = "bfs"
        return finish("certified")
    if not complete:
        return finish("state_cap")
    exhausted = len(levels[-1]) == 0
    if hint is not None and len(hint) <= d_max:
        out.update(d_star=len(hint), lower_bound_complete=True, certificate=list(hint),
                   certificate_source="hint")
        out.update(_path_quality(P0, R0, hint, geom))
        if terminal_ok is not None:
            out["terminal_ok"] = bool(terminal_ok(_replay_pose(P0, R0, hint)))
        return finish("certified")
    out["lower_bound_complete"] = True
    if exhausted:  # every reachable state enumerated, none succeeds
        out["lower_bound"] = int(d_max)
        return finish("reachable_set_exhausted")
    return finish("no_success_within_d_max")


def _replay_pose(P0, R0, actions) -> np.ndarray:
    P, R = P0, R0
    for a in actions:
        P, R = apply_action_batch(P, R, a)
    return to_c2w(P, R)[0]


def _check_hint(P0, R0, actions, oracle, geom, key_mode) -> Tuple[Optional[List[str]], str]:
    P, R = P0, R0
    for i, a in enumerate(actions, start=1):
        if a not in ACTION_INDEX:
            return None, "hint_unknown_action"
        Pn, Rn = apply_action_batch(P, R, a)
        if a.startswith("move_") and not bool(geom.transitions_legal(P, Pn, R)[0]):
            return None, "hint_collides"
        P, R = Pn, Rn
        if bool(oracle.batch(to_c2w(P, R), make_keys(P, R, key_mode))[0]):
            return actions[:i], ("hint_ok" if i == len(actions) else "hint_truncated_at_first_success")
    return None, "hint_never_succeeds"


def _path_quality(P0, R0, actions, geom) -> Dict[str, Any]:
    Ps, Rs = [], []
    P, R = P0, R0
    for a in actions:
        P, R = apply_action_batch(P, R, a)
        Ps.append(P[0])
        Rs.append(R[0])
    if not Ps:
        return {"certificate_generator_safe": True, "certificate_unsafe_states": 0, "certificate_switches": 0}
    safe = geom.poses_legal(np.array(Ps), np.array(Rs))
    switches = sum(1 for x, y in zip(actions, actions[1:]) if x != y)
    return {"certificate_generator_safe": bool(safe.all()), "certificate_unsafe_states": int((~safe).sum()),
            "certificate_switches": int(switches)}


def _select_certificate(levels: List[_Level], success_idx: np.ndarray, geom: FastGeometry, key_mode: str,
                        terminal_ok, prefer_safe: bool, max_checks: int) -> Dict[str, Any]:
    """Choose one shortest path through the DAG of ALL shortest legal paths."""
    d = len(levels) - 1
    marked: Dict[int, np.ndarray] = {d: np.unique(success_idx)}
    edges: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lv in range(d, 0, -1):
        child = marked[lv]
        cur, prev = levels[lv], levels[lv - 1]
        lookup = prev.index()
        # always keep the BFS representative edge so every marked node stays reachable
        ep, ea, ec = [cur.parent[child]], [cur.action[child]], [child]
        for ai, a in enumerate(ACTIONS):
            Pp, Rp = apply_action_batch(cur.P[child], cur.R[child], INVERSE_ACTION[a])
            pk = make_keys(Pp, Rp, key_mode).tolist()
            pidx = np.array([lookup.get(k, -1) for k in pk], dtype=np.int64)
            ok = pidx >= 0
            if not ok.any():
                continue
            pi, ci = pidx[ok], child[ok]
            Pf, Rf = apply_action_batch(prev.P[pi], prev.R[pi], a)
            same = make_keys(Pf, Rf, key_mode) == cur.keys[ci]
            if a.startswith("move_") and same.any():
                same[same] = geom.transitions_legal(prev.P[pi[same]], Pf[same], prev.R[pi[same]])
            ep.append(pi[same])
            ea.append(np.full(int(same.sum()), ai))
            ec.append(ci[same])
        P_, A_, C_ = np.concatenate(ep), np.concatenate(ea), np.concatenate(ec)
        trip = np.unique(np.stack([P_, A_, C_], axis=1), axis=0)
        edges[lv] = (trip[:, 0], trip[:, 1], trip[:, 2])
        marked[lv - 1] = np.unique(trip[:, 0])

    unsafe: Dict[int, np.ndarray] = {}
    for lv in range(1, d + 1):
        idx = marked[lv]
        if prefer_safe:
            unsafe[lv] = ~geom.poses_legal(levels[lv].P[idx], levels[lv].R[idx])
        else:
            unsafe[lv] = np.zeros(len(idx), dtype=bool)

    INF = np.iinfo(np.int64).max // 4
    cost = np.full((1, N_ACTIONS + 1), INF, dtype=np.int64)
    cost[0, N_ACTIONS] = 0
    back: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for lv in range(1, d + 1):
        pa, ac, ch = edges[lv]
        p_loc = np.searchsorted(marked[lv - 1], pa)
        c_loc = np.searchsorted(marked[lv], ch)
        prev_cost = cost[p_loc]  # (E, 7)
        sw = np.ones((len(ac), N_ACTIONS + 1), dtype=np.int64) * _SWITCH_COST
        sw[np.arange(len(ac)), ac] = 0
        sw[:, N_ACTIONS] = 0
        cand = np.where(prev_cost >= INF, INF, prev_cost + sw)
        best_la = np.argmin(cand, axis=1)
        best = cand[np.arange(len(ac)), best_la]
        best = np.where(best >= INF, INF, best + unsafe[lv][c_loc].astype(np.int64) * _UNSAFE_STATE_COST)
        new_cost = np.full((len(marked[lv]), N_ACTIONS + 1), INF, dtype=np.int64)
        bp_parent = np.full((len(marked[lv]), N_ACTIONS + 1), -1, dtype=np.int64)
        bp_la = np.full((len(marked[lv]), N_ACTIONS + 1), -1, dtype=np.int64)
        for e in range(len(ac)):  # deterministic: first minimal edge wins
            c, a = c_loc[e], ac[e]
            if best[e] < new_cost[c, a]:
                new_cost[c, a] = best[e]
                bp_parent[c, a] = p_loc[e]
                bp_la[c, a] = best_la[e]
        cost = new_cost
        back[lv] = (bp_parent, bp_la)

    term_cost = cost.min(axis=1)
    term_la = cost.argmin(axis=1)
    order = np.lexsort((marked[d], term_cost))
    chosen = int(order[0])
    term_flag: Optional[bool] = None
    if terminal_ok is not None:
        term_flag = False
        for j in order[:max(1, max_checks)]:
            j = int(j)
            if term_cost[j] >= INF:
                break
            node = marked[d][j]
            if terminal_ok(to_c2w(levels[d].P[node:node + 1], levels[d].R[node:node + 1])[0]):
                chosen, term_flag = j, True
                break
    actions: List[str] = []
    loc, la = chosen, int(term_la[chosen])
    total = int(term_cost[chosen])
    for lv in range(d, 0, -1):
        actions.append(ACTIONS[la])
        bp_parent, bp_la = back[lv]
        loc, la = int(bp_parent[loc, la]), int(bp_la[loc, la])
    actions.reverse()
    n_unsafe = total // _UNSAFE_STATE_COST
    return {"certificate": actions, "certificate_generator_safe": (n_unsafe == 0) if prefer_safe else None,
            "certificate_unsafe_states": int(n_unsafe) if prefer_safe else None,
            "certificate_switches": int(total % _UNSAFE_STATE_COST), "terminal_ok": term_flag,
            "shortest_path_dag_states": int(sum(len(v) for v in marked.values()))}


# ---------------------------------------------------------------------------
# Scalar reference replay (no fast paths)
# ---------------------------------------------------------------------------
def replay_certificate(scene: Any, success_fn: Any, init_c2w: Any, actions: Sequence[str],
                       check_generator_safe: bool = True) -> Dict[str, Any]:
    """Replay with contract.apply_action + scene.transition_legal + exact success.

    ok iff: initial pose not successful, no transition collides, the first
    successful state is exactly the last one.
    """
    exact = getattr(success_fn, "exact", None)
    judge = exact if callable(exact) else success_fn
    pose = np.asarray(init_c2w, dtype=float)
    res: Dict[str, Any] = {"ok": False, "reason": "", "first_success_step": None, "poses": [pose.tolist()],
                           "generator_safe": []}
    if _truthy(judge(pose)):
        res["reason"] = "init_success"
        res["first_success_step"] = 0
        return res
    for i, a in enumerate(actions, start=1):
        nxt = apply_action(pose, a)
        legal, why = scene.transition_legal(pose, nxt)
        if not legal:
            res["reason"] = f"collision_step_{i}:{why}"
            return res
        pose = nxt
        res["poses"].append(pose.tolist())
        if check_generator_safe:
            res["generator_safe"].append(bool(scene.is_pose_legal(pose)[0]))
        if _truthy(judge(pose)):
            res["first_success_step"] = i
            if i < len(actions):
                res["reason"] = "early_success"
                return res
    if res["first_success_step"] != len(actions) or not actions:
        res["reason"] = "no_success_at_end"
        return res
    res["ok"] = True
    res["reason"] = "ok"
    return res


def brute_force_d_star(scene: Any, success_fn: Callable[[np.ndarray], Any], init_c2w: Any,
                       depth: int = 4) -> Optional[int]:
    """Reference: enumerate ALL action sequences up to ``depth`` with exact
    scalar code (no de-duplication).  Colliding actions leave the pose
    unchanged; success is checked after every executed action."""
    judge = getattr(success_fn, "exact", success_fn)
    s_cache: Dict[bytes, bool] = {}
    t_cache: Dict[bytes, bool] = {}

    def succ(p: np.ndarray) -> bool:
        k = p.tobytes()
        if k not in s_cache:
            s_cache[k] = _truthy(judge(p))
        return s_cache[k]

    def legal(p: np.ndarray, q: np.ndarray) -> bool:
        k = p.tobytes() + q.tobytes()
        if k not in t_cache:
            t_cache[k] = bool(scene.transition_legal(p, q)[0])
        return t_cache[k]

    init = np.asarray(init_c2w, dtype=float)
    if succ(init):
        return 0
    best: List[Optional[int]] = [None]

    def rec(p: np.ndarray, k: int) -> None:
        if k >= depth or (best[0] is not None and k + 1 >= best[0]):
            return
        for a in ACTIONS:
            q = apply_action(p, a)
            if not legal(p, q):
                q = p  # runtime: collision invalidates the action
            if succ(q):
                best[0] = k + 1 if best[0] is None else min(best[0], k + 1)
                continue
            rec(q, k + 1)

    rec(init, 0)
    return best[0]
