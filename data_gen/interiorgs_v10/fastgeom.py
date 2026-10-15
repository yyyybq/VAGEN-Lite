"""Vectorised, exact replicas of the v10 pose / transition legality rules.

Why this module exists
----------------------
Endpoint, initial-state and difficulty searches ask "is this pose legal?" and
"is this 0.3 m move legal?" millions of times.  The reference answers are

* ``Scene.is_pose_legal``  (room membership, 0.5 m wall clearance, and
  ``CollisionDetector.check_collision(point)``), and
* ``CollisionDetector.check_collision(to, previous_position=from)`` (the
  runtime rule the RL environment applies after every translation),

both of which are pure-Python loops over every object box and wall segment
(~0.1-1 ms per call).  ``LegalityKernel`` evaluates the same predicates for
whole batches with numpy and returns *bit-identical booleans*.

How exactness is guaranteed
---------------------------
1.  Every comparison is replicated with the same operands and, where the
    reference uses scalar arithmetic (point-in-polygon crossing test,
    2-D segment/segment test, AABB containment), the same operation order, so
    those results are identical.
2.  Where the reference goes through ``np.dot`` / ``np.linalg.norm`` (BLAS,
    possibly fused multiply-add) our elementwise value can differ from the
    reference value by a few ulps.  Every such comparison is therefore
    evaluated with a *guard band*: if the fast value is within ``GUARD``
    (relative 1e-9, i.e. ~6 orders of magnitude above the possible rounding
    difference) of the decision threshold, the query is re-evaluated with the
    reference implementation (``ref_point`` / ``ref_transition`` callbacks).
    In practice this happens only for constructed ties (e.g. a grid point at
    exactly 0.5 m from an axis-aligned wall) and costs nothing on random
    queries.
3.  Spatial culling uses a uniform grid whose per-cell candidate lists are a
    *superset* of the geometry that can influence any query in the cell (cell
    rectangles expanded by the relevant radius plus a 1e-6 m margin), so
    culling can never change an answer.  Queries outside the grid, and moves
    longer than ``reach``, fall back to brute force over all candidates.

The kernel is constructed from plain arrays (see ``Scene._fast_kernel``) and
is independent of the ``vagen`` package.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

GUARD = 1e-9          # relative guard band for comparisons that may differ by ulps
CELL_MARGIN = 1e-6    # metres added to every culling radius
PIP_EPS = 1e-12       # the constant used by the reference point-in-polygon test


# ---------------------------------------------------------------------------
# Point in polygon (exact replica of scene.point_in_polygon / CollisionDetector._point_in_polygon)
# ---------------------------------------------------------------------------
def points_in_polygon(xy: np.ndarray, poly: np.ndarray, return_near: bool = False):
    """Vectorised crossing-number test, operation-for-operation identical to
    ``scene.point_in_polygon`` (``x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi``).

    Returns a bool array (N,), plus a "near" mask (|x - xcross| within the
    guard band for a straddling edge) when ``return_near`` is true.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = len(xy)
    inside = np.zeros(n, dtype=bool)
    near = np.zeros(n, dtype=bool)
    if n == 0 or len(poly) == 0:
        return (inside, near) if return_near else inside
    xi = poly[:, 0]
    yi = poly[:, 1]
    xj = np.roll(xi, 1)   # j = i - 1 (the reference starts with j = n - 1)
    yj = np.roll(yi, 1)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    straddle = (yi[None, :] > y) != (yj[None, :] > y)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xcross = (xj - xi)[None, :] * (y - yi[None, :]) / ((yj - yi)[None, :] + PIP_EPS) + xi[None, :]
        hit = straddle & (x < xcross)
    inside = (np.count_nonzero(hit, axis=1) % 2) == 1
    if return_near:
        with np.errstate(invalid="ignore", over="ignore"):
            close = np.abs(x - xcross) <= GUARD * (1.0 + np.abs(x))
        near = np.any(straddle & ~np.isfinite(xcross), axis=1) | np.any(straddle & close, axis=1)
        return inside, near
    return inside


def points_in_any_polygon(xy: np.ndarray, polygons: Sequence[np.ndarray], return_near: bool = False):
    """OR of ``points_in_polygon`` over polygons with exact bounding-box culling."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(len(xy), dtype=bool)
    near = np.zeros(len(xy), dtype=bool)
    for poly in polygons:
        poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(poly) == 0:
            continue
        lo = poly.min(axis=0) - CELL_MARGIN
        hi = poly.max(axis=0) + CELL_MARGIN
        # Outside the y-range no edge straddles; left of the x-range every
        # straddling edge is crossed (even count); right of it none is.
        sel = np.nonzero((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))[0]
        if len(sel) == 0:
            continue
        res = points_in_polygon(xy[sel], poly, return_near=return_near)
        if return_near:
            inside[sel] |= res[0]
            near[sel] |= res[1]
        else:
            inside[sel] |= res
    return (inside, near) if return_near else inside


def points_room_index(xy: np.ndarray, polygons: Sequence[np.ndarray]) -> np.ndarray:
    """Index of the FIRST polygon containing each point (-1 if none), like ``Scene.room_of``."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out = np.full(len(xy), -1, dtype=np.int64)
    for k, poly in enumerate(polygons):
        todo = np.nonzero(out < 0)[0]
        if len(todo) == 0:
            break
        hit = points_in_any_polygon(xy[todo], [poly])
        out[todo[hit]] = k
    return out


# ---------------------------------------------------------------------------
# Point / segment distances (elementwise; callers guard the thresholds)
# ---------------------------------------------------------------------------
def point_segment_distance_clip(px, py, ax, ay, bx, by):
    """Distance as in ``scene.point_segment_distance`` (clip formulation).

    Returns (distance, degenerate_near) where ``degenerate_near`` flags
    segments whose squared length is within the guard band of the 1e-12
    degeneracy threshold used by the reference.
    """
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((px - ax) * abx + (py - ay) * aby) / denom
    degenerate = denom < 1e-12
    t = np.where(degenerate, 0.0, np.clip(t, 0.0, 1.0))
    rx = px - (ax + t * abx)
    ry = py - (ay + t * aby)
    dist = np.sqrt(rx * rx + ry * ry)
    deg_near = np.abs(denom - 1e-12) <= 1e-12 * 1e-3
    return dist, deg_near


def point_segment_distance_branch(px, py, ax, ay, bx, by):
    """Distance as in ``CollisionDetector._point_to_segment_distance_2d``."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    c1 = wx * vx + wy * vy
    c2 = vx * vx + vy * vy
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where((c1 <= 0) | (c2 <= c1), 0.0, c1 / c2)
    d_start = np.sqrt(wx * wx + wy * wy)
    ex = px - bx
    ey = py - by
    d_end = np.sqrt(ex * ex + ey * ey)
    rx = px - (ax + b * vx)
    ry = py - (ay + b * vy)
    d_proj = np.sqrt(rx * rx + ry * ry)
    return np.where(c1 <= 0, d_start, np.where(c2 <= c1, d_end, d_proj))


def _close(value, threshold):
    return np.abs(value - threshold) <= GUARD * (1.0 + np.abs(threshold) + np.abs(value))


# ---------------------------------------------------------------------------
# Grid with per-cell candidate lists (CSR)
# ---------------------------------------------------------------------------
class _Grid:
    def __init__(self, lo: np.ndarray, hi: np.ndarray, cell: float) -> None:
        self.cell = float(cell)
        self.x0 = float(lo[0])
        self.y0 = float(lo[1])
        self.nx = max(1, int(math.ceil((float(hi[0]) - self.x0) / self.cell)))
        self.ny = max(1, int(math.ceil((float(hi[1]) - self.y0) / self.cell)))

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_of(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore"):
            fx = np.floor((xy[:, 0] - self.x0) / self.cell)
            fy = np.floor((xy[:, 1] - self.y0) / self.cell)
        ok = np.isfinite(fx) & np.isfinite(fy) & (fx >= 0) & (fx < self.nx) & (fy >= 0) & (fy < self.ny)
        ix = np.where(ok, fx, 0).astype(np.int64)
        iy = np.where(ok, fy, 0).astype(np.int64)
        return ix * self.ny + iy, ok

    def cell_range(self, xmin: float, xmax: float, ymin: float, ymax: float):
        """Index ranges of cells whose closed rectangle meets [xmin,xmax]x[ymin,ymax]."""
        i0 = max(0, int(math.floor((xmin - self.x0) / self.cell)) - 1)
        i1 = min(self.nx - 1, int(math.floor((xmax - self.x0) / self.cell)) + 1)
        j0 = max(0, int(math.floor((ymin - self.y0) / self.cell)) - 1)
        j1 = min(self.ny - 1, int(math.floor((ymax - self.y0) / self.cell)) + 1)
        return i0, i1, j0, j1

    def csr(self, rects: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Candidate lists: item k is listed in every cell within ``radius`` of rects[k]
        (rects: (K, 4) = xmin, ymin, xmax, ymax).  A superset (bbox based)."""
        cells: List[np.ndarray] = []
        items: List[np.ndarray] = []
        for k, (xa, ya, xb, yb) in enumerate(np.asarray(rects, dtype=float).reshape(-1, 4)):
            i0, i1, j0, j1 = self.cell_range(xa - radius, xb + radius, ya - radius, yb + radius)
            if i1 < i0 or j1 < j0:
                continue
            ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
            c = (ii * self.ny + jj).ravel()
            cells.append(c)
            items.append(np.full(len(c), k, dtype=np.int64))
        if cells:
            cell_arr = np.concatenate(cells)
            item_arr = np.concatenate(items)
            order = np.lexsort((item_arr, cell_arr))
            cell_arr = cell_arr[order]
            item_arr = item_arr[order]
        else:
            cell_arr = np.zeros(0, dtype=np.int64)
            item_arr = np.zeros(0, dtype=np.int64)
        counts = np.bincount(cell_arr, minlength=self.n_cells)
        offsets = np.zeros(self.n_cells + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return offsets, item_arr


def _expand_pairs(cells: np.ndarray, offsets: np.ndarray, flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(query index, candidate index) pairs for queries in the given cells."""
    starts = offsets[cells]
    counts = offsets[cells + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    qidx = np.repeat(np.arange(len(cells), dtype=np.int64), counts)
    out_start = np.cumsum(counts) - counts
    cand = flat[np.arange(total, dtype=np.int64) + np.repeat(starts - out_start, counts)]
    return qidx, cand


def _all_pairs(n_query: int, n_items: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_query == 0 or n_items == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return (np.repeat(np.arange(n_query, dtype=np.int64), n_items),
            np.tile(np.arange(n_items, dtype=np.int64), n_query))


def _any_by(qidx: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    if len(qidx) == 0:
        return np.zeros(n, dtype=bool)
    return np.bincount(qidx[mask], minlength=n) > 0


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
class LegalityKernel:
    """Batch legality for a fixed camera height ``z``.

    Parameters mirror the reference objects:
      polygons          -- Scene.rooms polygons (label coordinates, applied sign)
      wall_segments     -- CollisionDetector.wall_segments (door edges removed)
      box_min/box_max   -- CollisionDetector.object_boxes (already expanded)
      floor_height/ceiling_height/camera_radius -- detector attributes
      min_wall_distance -- Scene.min_wall_distance (generator-safe clearance)
      object_collision  -- detector.enable_object_collision (objects AND walls)
      boundary_collision-- detector.enable_boundary_collision
      scene_loaded      -- detector.scene_loaded (gates the object checks)
      ref_point(x, y) -> bool        exact reference for legal_mask fallbacks
      ref_transition(fx, fy, tx, ty) -> bool   exact reference (True = legal)
    """

    def __init__(self, polygons: Sequence[np.ndarray], wall_segments: Sequence[Tuple[np.ndarray, np.ndarray]],
                 box_min: np.ndarray, box_max: np.ndarray, *, z: float, floor_height: float,
                 ceiling_height: float, camera_radius: float, min_wall_distance: float,
                 object_collision: bool = True, boundary_collision: bool = True, scene_loaded: bool = True,
                 ref_point: Optional[Callable[[float, float], bool]] = None,
                 ref_transition: Optional[Callable[[float, float, float, float], bool]] = None,
                 cell: float = 0.5, reach: float = 0.45,
                 clearance_segments: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None) -> None:
        """``clearance_segments``: the walls Scene.wall_distance measures ``min_wall_distance`` from (the room
        edges); default the collision ``wall_segments`` (the two coincide under the "room_edges" wall model)."""
        self.z = float(z)
        self.polygons = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons]
        segs = list(wall_segments)
        self.walls = (np.array([[float(a[0]), float(a[1]), float(b[0]), float(b[1])] for a, b in segs],
                               dtype=np.float64).reshape(-1, 4))
        csegs = segs if clearance_segments is None else list(clearance_segments)
        self.cwalls = (np.array([[float(a[0]), float(a[1]), float(b[0]), float(b[1])] for a, b in csegs],
                                dtype=np.float64).reshape(-1, 4))
        box_min = np.asarray(box_min, dtype=np.float64).reshape(-1, 3)
        box_max = np.asarray(box_max, dtype=np.float64).reshape(-1, 3)
        # Only boxes whose z-slab contains z can contain a point at height z, and a
        # horizontal move (dz = 0, so direction z is exactly 0 -> "parallel" slab
        # branch) intersects a box only if start z lies in the same slab.
        zsel = (self.z >= box_min[:, 2]) & (self.z <= box_max[:, 2])
        self.bmin = box_min[zsel, :2].copy()
        self.bmax = box_max[zsel, :2].copy()
        self.n_boxes_total = int(len(box_min))
        self.floor_height = float(floor_height)
        self.ceiling_height = float(ceiling_height)
        self.camera_radius = float(camera_radius)
        self.min_wall_distance = float(min_wall_distance)
        self.object_collision = bool(object_collision)
        self.boundary_collision = bool(boundary_collision)
        self.scene_loaded = bool(scene_loaded)
        self.ref_point = ref_point
        self.ref_transition = ref_transition
        self.cell = float(cell)
        self.reach = float(reach)
        self.stats: Dict[str, int] = {"point_queries": 0, "point_fallbacks": 0,
                                      "transition_queries": 0, "transition_fallbacks": 0,
                                      "transition_brute_force": 0, "point_brute_force": 0}
        # Height test is identical for every query at this z.
        z_collide = False
        if self.boundary_collision:
            z_collide = (self.z < self.floor_height) or (self.z > self.ceiling_height)
        self.z_collide = bool(z_collide)
        self.check_boxes = self.object_collision and self.scene_loaded and len(self.bmin) > 0
        self.check_walls = self.object_collision and len(self.walls) > 0
        self._build_grid()

    # ------------------------------------------------------------------ build
    def _build_grid(self) -> None:
        pts = [np.zeros((0, 2))]
        pts += [p for p in self.polygons if len(p)]
        for w in (self.walls, self.cwalls):
            if len(w):
                pts += [w[:, :2], w[:, 2:]]
        if len(self.bmin):
            pts += [self.bmin, self.bmax]
        allp = np.concatenate(pts, axis=0)
        if len(allp) == 0:
            allp = np.zeros((1, 2))
        pad = max(self.min_wall_distance, self.camera_radius) + self.reach + 1.0
        lo = allp.min(axis=0) - pad
        hi = allp.max(axis=0) + pad
        self.grid = _Grid(lo, hi, self.cell)
        g = self.grid

        def rects(w: np.ndarray) -> np.ndarray:
            return (np.stack([np.minimum(w[:, 0], w[:, 2]), np.minimum(w[:, 1], w[:, 3]),
                              np.maximum(w[:, 0], w[:, 2]), np.maximum(w[:, 1], w[:, 3])], axis=1)
                    if len(w) else np.zeros((0, 4)))

        wall_rects = rects(self.walls)
        box_rects = np.concatenate([self.bmin, self.bmax], axis=1) if len(self.bmin) else np.zeros((0, 4))
        r_tr = self.reach + self.camera_radius + CELL_MARGIN
        self.pt_walls = g.csr(wall_rects, self.camera_radius + CELL_MARGIN)
        self.pt_cwalls = g.csr(rects(self.cwalls), self.min_wall_distance + CELL_MARGIN)
        self.pt_boxes = g.csr(box_rects, CELL_MARGIN)
        self.tr_walls = g.csr(wall_rects, r_tr)
        self.tr_boxes = g.csr(box_rects, self.reach + CELL_MARGIN)
        # Point-in-room state per cell: 0 outside every room, 1 inside some room,
        # 2 mixed (an edge passes within CELL_MARGIN of the cell -> exact test).
        state = np.zeros((g.nx, g.ny), dtype=np.int8)
        for poly in self.polygons:
            n = len(poly)
            for i in range(n):
                a = poly[i]
                b = poly[(i + 1) % n]
                i0, i1, j0, j1 = g.cell_range(min(a[0], b[0]) - CELL_MARGIN, max(a[0], b[0]) + CELL_MARGIN,
                                              min(a[1], b[1]) - CELL_MARGIN, max(a[1], b[1]) + CELL_MARGIN)
                if i1 >= i0 and j1 >= j0:
                    state[i0:i1 + 1, j0:j1 + 1] = 2
        ii, jj = np.nonzero(state != 2)
        if len(ii):
            centres = np.stack([g.x0 + (ii + 0.5) * g.cell, g.y0 + (jj + 0.5) * g.cell], axis=1)
            inside = points_in_any_polygon(centres, self.polygons)
            state[ii, jj] = inside.astype(np.int8)
        self.cell_room_state = state.ravel()

    # ------------------------------------------------------------- utilities
    def _pairs(self, cells: np.ndarray, in_grid: np.ndarray, csr: Tuple[np.ndarray, np.ndarray], n_items: int):
        """Pairs for in-grid queries via the CSR lists, brute force for the rest."""
        n = len(cells)
        q_parts: List[np.ndarray] = []
        c_parts: List[np.ndarray] = []
        gi = np.nonzero(in_grid)[0]
        if len(gi):
            q, c = _expand_pairs(cells[gi], csr[0], csr[1])
            q_parts.append(gi[q])
            c_parts.append(c)
        bi = np.nonzero(~in_grid)[0]
        if len(bi) and n_items:
            q, c = _all_pairs(len(bi), n_items)
            q_parts.append(bi[q])
            c_parts.append(c)
        if not q_parts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate(q_parts), np.concatenate(c_parts)

    # ----------------------------------------------------------- legal_mask
    def legal_mask(self, xy: np.ndarray) -> np.ndarray:
        """Same booleans as ``Scene.is_pose_legal(pose_from_xy_yaw(x, y, yaw))`` at z = self.z."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        n = len(xy)
        self.stats["point_queries"] += n
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self.z_collide:
            return np.zeros(n, dtype=bool)
        x = xy[:, 0]
        y = xy[:, 1]
        cells, in_grid = self.grid.cell_of(xy)
        self.stats["point_brute_force"] += int(np.count_nonzero(~in_grid))
        near = np.zeros(n, dtype=bool)

        # 1. inside some room (Scene.room_of is not None)
        state = np.where(in_grid, self.cell_room_state[cells], 2)
        in_room = state == 1
        exact = np.nonzero(state == 2)[0]
        if len(exact):
            ins, nr = points_in_any_polygon(xy[exact], self.polygons, return_near=True)
            in_room[exact] = ins
            near[exact] |= nr
        legal = in_room.copy()

        # 2. walls: Scene.wall_distance (clearance segments) < min_wall_distance, then the detector's
        #    distance to its collision segments < camera_radius
        if len(self.cwalls):
            q, k = self._pairs(cells, in_grid, self.pt_cwalls, len(self.cwalls))
            if len(q):
                w = self.cwalls[k]
                d_clip, deg_near = point_segment_distance_clip(x[q], y[q], w[:, 0], w[:, 1], w[:, 2], w[:, 3])
                near |= _any_by(q, _close(d_clip, self.min_wall_distance) | deg_near, n)
                legal &= ~_any_by(q, d_clip < self.min_wall_distance, n)
        if self.check_walls:
            q, k = self._pairs(cells, in_grid, self.pt_walls, len(self.walls))
            if len(q):
                w = self.walls[k]
                d_det = point_segment_distance_branch(x[q], y[q], w[:, 0], w[:, 1], w[:, 2], w[:, 3])
                near |= _any_by(q, _close(d_det, self.camera_radius), n)
                legal &= ~_any_by(q, d_det < self.camera_radius, n)

        # 3. object boxes (AABB.contains, inclusive, exact comparisons)
        if self.check_boxes:
            q, k = self._pairs(cells, in_grid, self.pt_boxes, len(self.bmin))
            if len(q):
                px = x[q]
                py = y[q]
                inside = ((px >= self.bmin[k, 0]) & (py >= self.bmin[k, 1])
                          & (px <= self.bmax[k, 0]) & (py <= self.bmax[k, 1]))
                legal &= ~_any_by(q, inside, n)

        fb = np.nonzero(near)[0]
        if len(fb):
            if self.ref_point is None:
                raise RuntimeError("near-tie query without a reference callback")
            self.stats["point_fallbacks"] += int(len(fb))
            for i in fb:
                legal[i] = bool(self.ref_point(float(x[i]), float(y[i])))
        return legal

    # ----------------------------------------------------- transitions_legal
    def transitions_legal(self, from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
        """Same booleans as ``not detector.check_collision([tx, ty, z], previous_position=[fx, fy, z]).has_collision``."""
        a = np.asarray(from_xy, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(to_xy, dtype=np.float64).reshape(-1, 2)
        if len(a) == 1 and len(b) > 1:
            a = np.repeat(a, len(b), axis=0)
        if len(a) != len(b):
            raise ValueError("from_xy and to_xy must have the same length")
        n = len(a)
        self.stats["transition_queries"] += n
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self.z_collide:
            return np.zeros(n, dtype=bool)
        fx, fy = a[:, 0], a[:, 1]
        tx, ty = b[:, 0], b[:, 1]
        dx = tx - fx
        dy = ty - fy
        length = np.sqrt(dx * dx + dy * dy)  # reference: np.linalg.norm (ulps may differ -> guards)
        cells, in_grid = self.grid.cell_of(a)
        in_grid = in_grid & (length <= self.reach)
        self.stats["transition_brute_force"] += int(np.count_nonzero(~in_grid))
        collide = np.zeros(n, dtype=bool)
        near = (length > 0) & (length < 2e-10)

        # 1. object boxes: point containment of `to`, then the slab segment test.
        if self.check_boxes:
            q, k = self._pairs(cells, in_grid, self.tr_boxes, len(self.bmin))
            if len(q):
                mnx, mny = self.bmin[k, 0], self.bmin[k, 1]
                mxx, mxy = self.bmax[k, 0], self.bmax[k, 1]
                sx, sy = fx[q], fy[q]
                ex, ey = tx[q], ty[q]
                contains_to = (ex >= mnx) & (ey >= mny) & (ex <= mxx) & (ey <= mxy)
                contains_from = (sx >= mnx) & (sy >= mny) & (sx <= mxx) & (sy <= mxy)
                L = length[q]
                zero = L < 1e-10
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    Ls = np.where(zero, 1.0, L)
                    dirx = dx[q] / Ls
                    diry = dy[q] / Ls
                    t_min = np.zeros(len(q))
                    t_max = L.copy()
                    ok = np.ones(len(q), dtype=bool)
                    pn = np.zeros(len(q), dtype=bool)
                    for d, s, mn, mx in ((dirx, sx, mnx, mxx), (diry, sy, mny, mxy)):
                        par = np.abs(d) < 1e-10
                        pn |= (np.abs(d) > 0.5e-10) & (np.abs(d) < 2e-10)
                        ok &= ~(par & ((s < mn) | (s > mx)))
                        dd = np.where(par, 1.0, d)
                        t1 = (mn - s) / dd
                        t2 = (mx - s) / dd
                        lo = np.minimum(t1, t2)
                        hi = np.maximum(t1, t2)
                        t_min = np.where(par, t_min, np.maximum(t_min, lo))
                        t_max = np.where(par, t_max, np.minimum(t_max, hi))
                    intersects = ok & (t_min <= t_max)
                    t = t_min / Ls
                    seg_hit = np.where(zero, contains_from, intersects & (t < 1.0))
                    tie = (~zero) & ok & (
                        (np.abs(t_min - t_max) <= GUARD * (1.0 + np.abs(t_min) + np.abs(t_max)))
                        | (intersects & (np.abs(t - 1.0) <= GUARD))
                    )
                hit = contains_to | seg_hit
                collide |= _any_by(q, hit, n)
                near |= _any_by(q, (tie | pn) & ~contains_to, n)

        # 2. walls: distance of `to` < camera_radius, then 2-D segment crossing.
        if self.check_walls:
            q, k = self._pairs(cells, in_grid, self.tr_walls, len(self.walls))
            if len(q):
                w = self.walls[k]
                ex, ey = tx[q], ty[q]
                d_det = point_segment_distance_branch(ex, ey, w[:, 0], w[:, 1], w[:, 2], w[:, 3])
                close_hit = d_det < self.camera_radius
                tie = _close(d_det, self.camera_radius)
                # _segment_intersects_wall_2d(prev_2d, pos_2d, wall_start, wall_end)
                d1x = dx[q]
                d1y = dy[q]
                d2x = w[:, 2] - w[:, 0]
                d2y = w[:, 3] - w[:, 1]
                cross = d1x * d2y - d1y * d2x
                nonpar = ~(np.abs(cross) < 1e-10)
                diffx = w[:, 0] - fx[q]
                diffy = w[:, 1] - fy[q]
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    cs = np.where(nonpar, cross, 1.0)
                    t = (diffx * d2y - diffy * d2x) / cs
                    u = (diffx * d1y - diffy * d1x) / cs
                seg_hit = nonpar & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & (t > 0) & (t < 1.0)
                # The 2-D crossing arithmetic is replicated operation for operation,
                # so these guards are belt and braces; they only fire when the
                # crossing point sits (within GUARD) on an end of either segment.
                tie = tie | (np.abs(np.abs(cross) - 1e-10) <= 1e-10 * 1e-3)
                in_t = (t >= -GUARD) & (t <= 1.0 + GUARD)
                in_u = (u >= -GUARD) & (u <= 1.0 + GUARD)
                t_end = (np.abs(t) <= GUARD) | (np.abs(t - 1.0) <= GUARD)
                u_end = (np.abs(u) <= GUARD) | (np.abs(u - 1.0) <= GUARD)
                tie = tie | (nonpar & ((t_end & in_u) | (u_end & in_t)))
                collide |= _any_by(q, close_hit | seg_hit, n)
                near |= _any_by(q, tie, n)

        legal = ~collide
        fb = np.nonzero(near)[0]
        if len(fb):
            if self.ref_transition is None:
                raise RuntimeError("near-tie query without a reference callback")
            self.stats["transition_fallbacks"] += int(len(fb))
            for i in fb:
                legal[i] = bool(self.ref_transition(float(fx[i]), float(fy[i]), float(tx[i]), float(ty[i])))
        return legal
