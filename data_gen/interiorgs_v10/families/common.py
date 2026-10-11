"""Shared helpers for the v2 task families (workstream F).

Every family module (absolute_positioning.py, equidistance.py, ...) exposes
the same interface, collected in ``canonical_v2.FAMILIES``:

    TASK_TYPE, REGION_TYPE, N_OBJECTS, ROLE_NAMES, TERMINAL_IDENTIFIABLE_ROLES,
    PARAM_MENU, GATES_DOC
    tuple_ok(objs) -> (bool, reason)                 ordered role tuple of Obj
    param_menu(objs, funnel=None) -> [params]        geometrically feasible params
    build_item(scene_id, objs, params, init_c2w=None) -> item (task row core)
    describe(item) -> str                            English instruction
    judge(item, c2w) -> metric dict                  canonical v2 judge
    anchor_xy(item) -> (x, y)                        yaw heuristic target
    necessary_mask(item, xy) -> bool[N]              exact necessary condition
    success_pose_candidates(scene, item, seed, ...) -> [candidate dict]

The judge itself lives in ``judges_v2.py`` (vendored into vagen).
"""

from __future__ import annotations

import math
import weakref
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import (
    ACTION_PROTOCOL_VERSION,
    ACTIONS,
    CAMERA_MODEL_VERSION,
    CANONICAL_TASK_METRIC_V2,
    GENERATOR_VERSION,
    NATIVE_K,
    STEP_ROTATION_DEG,
    STEP_TRANSLATION_M,
    Funnel,
    Obj,
    apply_action,
    pose_from_xy_yaw,
)
from ..sampling import floor_grid
from . import judges_v2 as J

if J.CANONICAL_TASK_METRIC_VERSION_V2 != CANONICAL_TASK_METRIC_V2:  # pragma: no cover
    raise RuntimeError("judges_v2 and contract disagree on the v2 metric version string")

ROLE_LETTERS = ("a", "b", "c")
DEFAULT_YAW_OFFSETS_DEG = (0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0)
STRUCTURAL_LABEL_HEADS = frozenset({"wall", "floor", "ceiling", "room", "niche", "skirting", "baseboard"})

try:  # workstream E; optional
    from .. import occlusion as _occlusion  # type: ignore
except Exception:  # pragma: no cover - module absent or broken
    _occlusion = None


# =============================================================================
# objects / text
# =============================================================================
def ref_of(obj: Any) -> str:
    """Referring expression for an Obj or a target_object entry."""
    if isinstance(obj, Obj):
        ref, label = obj.ref, obj.label
    else:
        ref, label = obj.get("ref"), obj.get("label")
    return str(ref) if ref else f"the {label}"


def item_objects(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((item.get("target_object") or {}).get("objects") or [])


def label_head(label: Any) -> str:
    words = str(label or "").strip().lower().split()
    return words[-1] if words else ""


def is_structural(obj: Obj) -> bool:
    return label_head(obj.label) in STRUCTURAL_LABEL_HEADS


def footprint_gap(a: Obj, b: Obj) -> float:
    """Horizontal gap between two AABB footprints (0 when they overlap)."""
    dx = max(float(a.aabb_min[0]) - float(b.aabb_max[0]), 0.0, float(b.aabb_min[0]) - float(a.aabb_max[0]))
    dy = max(float(a.aabb_min[1]) - float(b.aabb_max[1]), 0.0, float(b.aabb_min[1]) - float(a.aabb_max[1]))
    return float(math.hypot(dx, dy))


def centre_separation_xy(a: Obj, b: Obj) -> float:
    return float(np.linalg.norm(a.center[:2] - b.center[:2]))


def same_room_ok(objs: Sequence[Obj]) -> bool:
    rooms = {o.room_index for o in objs if o.room_index is not None}
    return len(rooms) <= 1


def footprint_distances(entry: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    """Vectorised judges_v2.footprint_distance for (N, 2) positions."""
    lo = np.asarray(entry["bbox_min"], float)
    hi = np.asarray(entry["bbox_max"], float)
    xy = np.asarray(xy, float).reshape(-1, 2)
    dx = np.maximum.reduce([lo[0] - xy[:, 0], np.zeros(len(xy)), xy[:, 0] - hi[0]])
    dy = np.maximum.reduce([lo[1] - xy[:, 1], np.zeros(len(xy)), xy[:, 1] - hi[1]])
    return np.hypot(dx, dy)


def role_entry(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Target entry for role ``index`` exactly as the judge resolves it."""
    params = (item.get("target_region") or {}).get("params") or {}
    letter = ROLE_LETTERS[index]
    objects = item_objects(item)
    from vagen.envs.active_spatial.visual_bbox_metrics import _match_object

    obj = _match_object(objects, object_id=params.get(f"object_{letter}_id"),
                        center=params.get(f"object_{letter}_center"), fallback_index=index)
    if obj is None:
        raise ValueError(f"item has no object for role {letter!r}")
    return obj


def entry_center(entry: Dict[str, Any]) -> np.ndarray:
    return 0.5 * (np.asarray(entry["bbox_min"], float) + np.asarray(entry["bbox_max"], float))


# =============================================================================
# item rows
# =============================================================================
def make_item(scene_id: str, task_type: str, objs: Sequence[Obj], region_type: str,
              params: Dict[str, Any], describe: Callable[[Dict[str, Any]], str],
              init_c2w: Optional[np.ndarray] = None,
              role_aliases: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Canonical v2 item.  Role order == target_object.objects order."""
    region_params: Dict[str, Any] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in params.items()}
    for letter, obj in zip(ROLE_LETTERS, objs):
        region_params[f"object_{letter}_id"] = str(obj.id)
        region_params[f"object_{letter}_center"] = [float(v) for v in obj.center]
    for alias, index in (role_aliases or {}).items():
        region_params[alias] = str(objs[index].id)
    item: Dict[str, Any] = {
        "scene_id": scene_id,
        "task_type": task_type,
        "task_description": "",
        "target_object": {"objects": [o.target_object_entry() for o in objs]},
        "target_region": {"type": region_type, "params": region_params},
        "init_camera": {
            "intrinsics": [list(map(float, row)) for row in NATIVE_K],
            "extrinsics": (np.asarray(init_c2w, float).tolist() if init_c2w is not None else None),
        },
        "canonical_task_metric_version": CANONICAL_TASK_METRIC_V2,
        "camera_model_version": CAMERA_MODEL_VERSION,
        "action_protocol_version": ACTION_PROTOCOL_VERSION,
        "generator_version": GENERATOR_VERSION,
    }
    item["task_description"] = describe(item)
    return item


def format_metres(d: float) -> str:
    return f"{float(d):.1f}"


# =============================================================================
# scene-free feasibility probes
# =============================================================================
def point_at_footprint_distance(entry: Dict[str, Any], azimuth: float, d: float) -> Tuple[float, float]:
    """Point on the ray from the AABB centre along ``azimuth`` whose footprint
    distance equals ``d`` (bisection; footprint distance is monotone on the ray)."""
    c = entry_center(entry)[:2]
    u = np.array([math.cos(azimuth), math.sin(azimuth)])
    lo, hi = 0.0, float(d) + float(np.linalg.norm(np.asarray(entry["bbox_max"], float)[:2]
                                                    - np.asarray(entry["bbox_min"], float)[:2])) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = c + mid * u
        if float(footprint_distances(entry, p[None, :])[0]) < d:
            lo = mid
        else:
            hi = mid
    p = c + hi * u
    return float(p[0]), float(p[1])


def facing_pose(x: float, y: float, target_xy: Sequence[float], offset_deg: float = 0.0) -> np.ndarray:
    yaw = math.atan2(float(target_xy[1]) - y, float(target_xy[0]) - x) + math.radians(offset_deg)
    return pose_from_xy_yaw(x, y, yaw)


def probe_item_feasible(item: Dict[str, Any], judge: Callable, poses: Iterable[np.ndarray]) -> bool:
    for pose in poses:
        if judge(item, pose)["success"]:
            return True
    return False


def generic_probe_feasible(item: Dict[str, Any], family: Any, radius: float = 5.0, spacing: float = 0.4,
                           offsets_deg: Sequence[float] = (0.0,), seed: int = 0) -> bool:
    """Scene-free check: does any pose on a disc around the anchor succeed?

    Ignores walls / legality; it only rejects parameterisations that are
    impossible under pitch 0 so that the scene sampler is not wasted on them.
    """
    ax, ay = family.anchor_xy(item)
    g = np.arange(-radius, radius + 1e-9, spacing)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    xy = np.stack([gx.ravel() + ax, gy.ravel() + ay], axis=1)
    xy = xy[np.hypot(xy[:, 0] - ax, xy[:, 1] - ay) <= radius]
    for entry in item_objects(item):  # never inside a target footprint
        xy = xy[footprint_distances(entry, xy) > 0.3]
    xy = xy[np.asarray(family.necessary_mask(item, xy), bool)]
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(xy)):
        x, y = float(xy[i][0]), float(xy[i][1])
        for off in offsets_deg:
            if family.judge(item, facing_pose(x, y, (ax, ay), off))["success"]:
                return True
    return False


# =============================================================================
# floor grid cache (legality only depends on xy at pitch 0)
# =============================================================================
_GRID_CACHE: Dict[int, Tuple[Any, Dict[Tuple[float, Tuple[int, ...]], np.ndarray]]] = {}


def cached_floor_grid(scene: Any, spacing: float, rooms: Optional[Sequence[int]]) -> np.ndarray:
    key_rooms = tuple(sorted(int(r) for r in rooms)) if rooms is not None else (-1,)
    sid = id(scene)
    entry = _GRID_CACHE.get(sid)
    if entry is None or entry[0]() is not scene:
        try:
            ref = weakref.ref(scene)
        except TypeError:  # pragma: no cover
            ref = (lambda s=scene: s)
        entry = (ref, {})
        _GRID_CACHE[sid] = entry
        if len(_GRID_CACHE) > 64:
            for k in [k for k, v in _GRID_CACHE.items() if v[0]() is None]:
                _GRID_CACHE.pop(k, None)
    grids = entry[1]
    key = (round(float(spacing), 6), key_rooms)
    if key not in grids:
        grids[key] = floor_grid(scene, spacing=spacing, room_indices=(None if rooms is None else list(rooms)))
    return grids[key]


def item_rooms(scene: Any, item: Dict[str, Any]) -> Optional[List[int]]:
    """Rooms that contain the target objects (None -> all rooms)."""
    by_id = {str(o.id): o for o in getattr(scene, "objects", [])}
    rooms = set()
    for entry in item_objects(item):
        obj = by_id.get(str(entry.get("id")))
        if obj is None or obj.room_index is None:
            return None
        rooms.add(int(obj.room_index))
    return sorted(rooms) if rooms else None


def scene_object(scene: Any, entry: Dict[str, Any]) -> Any:
    for obj in getattr(scene, "objects", []):
        if str(obj.id) == str(entry.get("id")):
            return obj
    return entry


# =============================================================================
# success-pose sampler
# =============================================================================
def identifiability_report(scene: Any, item: Dict[str, Any], c2w: np.ndarray,
                           roles: Sequence[int]) -> Optional[Dict[str, Any]]:
    """occlusion.identifiable for the given roles (None if module unavailable)."""
    if _occlusion is None or not hasattr(_occlusion, "identifiable"):
        return None
    out: Dict[str, Any] = {"ok": True, "roles": {}}
    for index in roles:
        entry = role_entry(item, index)
        rep = _occlusion.identifiable(scene, c2w, scene_object(scene, entry))
        out["roles"][ROLE_LETTERS[index]] = {
            "identifiable": bool(rep.get("identifiable")),
            "visible_fraction": float(rep.get("visible_fraction", 0.0)),
        }
        out["ok"] = out["ok"] and bool(rep.get("identifiable"))
    return out


def lattice_support(item: Dict[str, Any], judge: Callable, c2w: np.ndarray,
                    scene: Any = None) -> int:
    """How many of the 6 one-step neighbours (legal transitions) also succeed."""
    n = 0
    for action in ACTIONS:
        nxt = apply_action(c2w, action, STEP_TRANSLATION_M, STEP_ROTATION_DEG)
        if scene is not None and action.startswith("move"):
            ok, _ = scene.transition_legal(c2w, nxt)
            if not ok:
                continue
        if judge(item, nxt)["success"]:
            n += 1
    return n


def sample_success_poses(scene: Any, item: Dict[str, Any], seed: int, family: Any, *,
                         spacing: float = 0.2,
                         yaw_offsets_deg: Sequence[float] = DEFAULT_YAW_OFFSETS_DEG,
                         max_candidates: int = 24,
                         require_identifiable: bool = True,
                         funnel: Optional[Funnel] = None,
                         grid: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Legal success poses for ``item`` (at most one yaw per grid position).

    Positions: ``sampling.floor_grid`` (cached) restricted to the target rooms
    and to the family's exact necessary condition; yaw = face the family anchor
    plus the first offset (random order) that passes the judge.  If
    ``occlusion.py`` is available, the family's terminal roles must also be
    ``occlusion.identifiable``.  Candidates are returned most-robust first
    (``lattice_support`` = number of successful one-step neighbours).
    """
    stage = f"family_{family.TASK_TYPE}_sampler"
    fn = funnel if funnel is not None else Funnel()
    rng = np.random.default_rng(seed)
    pts = grid if grid is not None else cached_floor_grid(scene, spacing, item_rooms(scene, item))
    if len(pts) == 0:
        fn.add(stage, "no_legal_grid_points")
        return []
    mask = np.asarray(family.necessary_mask(item, pts), bool)
    pts = pts[mask]
    if len(pts) == 0:
        fn.add(stage, "necessary_condition_empty")
        return []
    ax, ay = family.anchor_xy(item)
    out: List[Dict[str, Any]] = []
    for i in rng.permutation(len(pts)):
        x, y = float(pts[i][0]), float(pts[i][1])
        base = math.atan2(ay - y, ax - x)
        for off in rng.permutation(np.asarray(yaw_offsets_deg, float)):
            c2w = pose_from_xy_yaw(x, y, base + math.radians(float(off)))
            metric = family.judge(item, c2w)
            if not metric["success"]:
                continue
            ident = None
            if require_identifiable:
                ident = identifiability_report(scene, item, c2w, family.TERMINAL_IDENTIFIABLE_ROLES)
                if ident is None:
                    fn.add(stage, "identifiability_unchecked")
                elif not ident["ok"]:
                    fn.add(stage, "terminal_not_identifiable")
                    continue
            out.append({
                "c2w": c2w,
                "xy": [x, y],
                "yaw_deg": math.degrees(base) + float(off),
                "shaping_score": float(metric["shaping_score"]),
                "identifiability": ident,
            })
            break
        if len(out) >= max_candidates:
            break
    for cand in out:
        cand["lattice_support"] = lattice_support(item, family.judge, cand["c2w"], scene)
    out.sort(key=lambda c: -c["lattice_support"])
    fn.add(stage, "ok" if out else "no_success_pose")
    return out


# =============================================================================
# success-region statistics (tests / audits)
# =============================================================================
def region_stats(scene: Any, item: Dict[str, Any], family: Any, *, spacing: float = 0.15,
                 yaw_step_deg: float = 10.0, yaw_window_deg: float = 90.0,
                 neighbour_samples: int = 40, seed: int = 0,
                 grid: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Size of the success region on a fine (spacing x yaw_step) pose grid.

    Yaws further than ``yaw_window_deg`` from the family anchor are skipped:
    every family needs all its targets in the 90 deg HFOV, so they cannot
    succeed there (asserted in tests on random poses).
    ``expected_lattice_successes`` is the expected number of success poses on
    a randomly offset 0.3 m x 20 deg action lattice = success measure / cell.
    ``sublattice_*`` count successes on the (0.3/spacing)^2 x (20/yaw_step)
    sub-lattices of the fine grid that are exact 0.3 m / 20 deg lattices.
    """
    pts = grid if grid is not None else cached_floor_grid(scene, spacing, item_rooms(scene, item))
    yaws = np.radians(np.arange(0.0, 360.0, yaw_step_deg))
    n_pos = len(pts)
    stats: Dict[str, Any] = {"legal_positions": int(n_pos), "spacing_m": spacing, "yaw_step_deg": yaw_step_deg}
    if n_pos == 0:
        stats.update({"success_poses": 0, "success_pose_fraction": 0.0, "success_area_m2": 0.0,
                      "expected_lattice_successes": 0.0})
        return stats
    nec = np.asarray(family.necessary_mask(item, pts), bool)
    ax, ay = family.anchor_xy(item)
    m_pos = max(1, int(round(STEP_TRANSLATION_M / spacing)))
    m_yaw = max(1, int(round(STEP_ROTATION_DEG / yaw_step_deg)))
    sub_counts: Dict[Tuple[int, int, int], int] = {}
    successes: List[Tuple[float, float, float]] = []
    evaluated = 0
    for (x, y), ok in zip(pts, nec):
        if not ok:
            continue
        base = math.atan2(ay - y, ax - x)
        ix, iy = int(round(x / spacing)), int(round(y / spacing))
        for k, yaw in enumerate(yaws):
            diff = (yaw - base + math.pi) % (2 * math.pi) - math.pi
            if abs(math.degrees(diff)) > yaw_window_deg:
                continue
            evaluated += 1
            if family.judge(item, pose_from_xy_yaw(float(x), float(y), float(yaw)))["success"]:
                successes.append((float(x), float(y), float(yaw)))
                key = (ix % m_pos, iy % m_pos, k % m_yaw)
                sub_counts[key] = sub_counts.get(key, 0) + 1
    n_sub = m_pos * m_pos * m_yaw
    counts = [sub_counts.get((a, b, c), 0) for a in range(m_pos) for b in range(m_pos) for c in range(m_yaw)]
    positions = {(round(s[0], 4), round(s[1], 4)) for s in successes}
    cell = (STEP_TRANSLATION_M ** 2) * STEP_ROTATION_DEG
    fine = spacing ** 2 * yaw_step_deg
    stats.update({
        "necessary_positions": int(nec.sum()),
        "evaluated_poses": int(evaluated),
        "success_poses": len(successes),
        "success_pose_fraction": len(successes) / float(n_pos * len(yaws)),
        "success_positions": len(positions),
        "success_position_fraction": len(positions) / float(n_pos),
        "success_area_m2": len(positions) * spacing ** 2,
        "expected_lattice_successes": len(successes) * fine / cell,
        "sublattices": n_sub,
        "sublattice_successes_min": int(min(counts)),
        "sublattice_hit_fraction": float(np.mean([c > 0 for c in counts])),
    })
    if successes and neighbour_samples > 0:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(successes), size=min(neighbour_samples, len(successes)), replace=False)
        supports = [lattice_support(item, family.judge, pose_from_xy_yaw(*successes[i]), None) for i in picks]
        stats["neighbour_support_mean"] = float(np.mean(supports))
        stats["isolated_success_fraction"] = float(np.mean([s == 0 for s in supports]))
    return stats


# =============================================================================
# candidate tuples
# =============================================================================
def enumerate_tuples(family: Any, objects: Sequence[Obj], seed: int = 0, max_tuples: int = 200,
                     funnel: Optional[Funnel] = None, ordered: bool = True) -> List[Tuple[Obj, ...]]:
    """Role tuples of distinct objects that pass ``family.tuple_ok``.

    ``ordered=False`` keeps one ordering per unordered set for symmetric
    families (equidistance, size_distance_invariance) and, for centering,
    one ordering of the two flanks.
    """
    import itertools

    stage = f"family_{family.TASK_TYPE}_tuples"
    fn = funnel if funnel is not None else Funnel()
    rng = np.random.default_rng(seed)
    objs = list(objects)
    n = family.N_OBJECTS
    total = math.perm(len(objs), n) if len(objs) >= n else 0
    if total <= 50_000:
        all_combos = list(itertools.permutations(range(len(objs)), n))
        order: Iterable[Tuple[int, ...]] = (all_combos[int(i)] for i in rng.permutation(len(all_combos)))
    else:  # large scenes: random draws with a fixed attempt budget
        order = (tuple(int(v) for v in rng.choice(len(objs), size=n, replace=False))
                 for _ in range(max(1000, 50 * max_tuples)))
    out: List[Tuple[Obj, ...]] = []
    seen = set()
    for combo in order:
        if ordered:
            key = tuple(combo)
        else:
            key = tuple(sorted(combo)) if n == 2 else (combo[0],) + tuple(sorted(combo[1:]))
        if key in seen:
            continue
        seen.add(key)
        tup = tuple(objs[i] for i in combo)
        ok, reason = family.tuple_ok(tup)
        fn.add(stage, reason)
        if ok:
            out.append(tup)
            if len(out) >= max_tuples:
                break
    return out
