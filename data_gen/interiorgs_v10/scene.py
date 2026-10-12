"""Scene loading and pose legality for the v10 generator (baseline version).

Baseline behaviour deliberately reuses the runtime CollisionDetector so that
every legality decision made while generating data agrees with what the RL
environment will do at train time.  The coordinate-convention calibration
and fast vectorised legality are owned by the scene-geometry workstream and
must keep this public API:

    scene = load_scene(scene_dir, scene_id)
    scene.objects            -> List[Obj]   (every labels.json instance)
    scene.rooms              -> List[Room]  (polygons in label coordinates)
    scene.convention         -> dict        (sign, status, evidence)
    scene.room_of(x, y)      -> Optional[int]
    scene.is_pose_legal(c2w) -> (bool, reason)          generator-safe subset
    scene.transition_legal(c2w_from, c2w_to) -> (bool, reason)  runtime rule

Workstream G additions (calibration v2 + vectorised legality):

    scene.convention_v2      -> dict   coordinates.calibrate_structure_y_sign record
                                       (+ "applied_sign", "applied_source")
    scene.legal_mask(xy)     -> (N,) bool   == is_pose_legal at pitch 0, z = 1.5
    scene.transitions_legal(from_xy, to_xy) -> (N,) bool
                             == not detector.check_collision(to, previous_position=from).has_collision
    scene.rooms_of(xy)       -> (N,) int    == room_of (first containing room, -1 for None)

``load_scene`` runs calibration v2.  When v2 freezes a sign that the v1 rule
inside CollisionDetector would call ambiguous, the sign is passed to the
embedded detector through ``structure_y_sign_overrides`` -- exactly what the
RL env must do at train time (see coordinates.env_overrides).  Ambiguous
scenes keep v1's fallback sign and ``scene.convention["status"] ==
"ambiguous"``; callers must reject them as before.  The fast paths
(fastgeom.LegalityKernel) are exact: near-ties are re-checked with the slow
reference methods below.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contract import (
    CAMERA_HEIGHT_M,
    CAMERA_RADIUS_M,
    MIN_WALL_DISTANCE_M,
    SAFETY_MARGIN_M,
    Obj,
    camera_forward_z,
)


@dataclasses.dataclass
class Room:
    index: int
    name: str
    polygon: np.ndarray  # (N, 2) in label coordinates

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def parse_labels(labels: Sequence[Dict[str, Any]]) -> List[Obj]:
    out: List[Obj] = []
    for item in labels:
        bbox = item.get("bounding_box") or []
        if len(bbox) < 8:
            continue
        obj_id = str(item.get("ins_id") or item.get("id") or "")
        if not obj_id:
            continue
        pts = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in bbox], dtype=float)
        out.append(Obj(id=obj_id, label=str(item.get("label", "")).strip().lower(),
                       aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0)))
    return out


@dataclasses.dataclass
class Scene:
    scene_id: str
    scene_dir: Path
    objects: List[Obj]
    rooms: List[Room]
    convention: Dict[str, Any]
    detector: Any  # vagen.envs.active_spatial.collision_detector.CollisionDetector
    min_wall_distance: float = MIN_WALL_DISTANCE_M
    convention_v2: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # labels.json instances dropped as annotation errors (configs/ignored_objects_*.json): not in
    # ``objects`` (no task, anchor or occluder) and not in the detector's collision boxes
    ignored_object_ids: List[str] = dataclasses.field(default_factory=list)
    _kernel: Any = dataclasses.field(default=None, repr=False, compare=False)
    _kernel_key: Any = dataclasses.field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ rooms
    def room_of(self, x: float, y: float) -> Optional[int]:
        for room in self.rooms:
            if room.contains(x, y):
                return room.index
        return None

    def wall_distance(self, x: float, y: float) -> float:
        p = np.array([x, y], dtype=float)
        segs = self.detector.wall_segments
        if not segs:
            return math.inf
        return min(point_segment_distance(p, a, b) for a, b in segs)

    # --------------------------------------------------------------- legality
    def is_pose_legal(self, c2w: np.ndarray) -> Tuple[bool, str]:
        """Generator-safe subset: inside a room, clear of walls and objects, z/pitch canonical."""
        c2w = np.asarray(c2w, dtype=float)
        x, y, z = (float(v) for v in c2w[:3, 3])
        if abs(z - CAMERA_HEIGHT_M) > 1e-6:
            return False, "height_not_canonical"
        if abs(float(c2w[2, 2]) - camera_forward_z()) > 1e-6:
            return False, "pitch_not_zero"
        if self.room_of(x, y) is None:
            return False, "outside_rooms"
        if self.wall_distance(x, y) < self.min_wall_distance:
            return False, "too_close_to_wall"
        res = self.detector.check_collision(np.array([x, y, z]))
        if res.has_collision:
            return False, f"collision_{res.collision_type}"
        return True, "ok"

    # ----------------------------------------------------- vectorised (exact)
    def _fast_kernel(self):
        from .fastgeom import LegalityKernel

        det = self.detector
        key = (float(self.min_wall_distance), id(det), len(det.object_boxes), len(det.wall_segments),
               float(det.floor_height), float(det.ceiling_height), float(det.camera_radius),
               bool(det.enable_object_collision), bool(det.enable_boundary_collision), bool(det.scene_loaded),
               len(self.rooms))
        if self._kernel is None or self._kernel_key != key:
            boxes = det.object_boxes
            bmin = np.array([b.min_point for b in boxes], dtype=float).reshape(-1, 3)
            bmax = np.array([b.max_point for b in boxes], dtype=float).reshape(-1, 3)
            self._kernel = LegalityKernel(
                [r.polygon for r in self.rooms], det.wall_segments, bmin, bmax,
                z=CAMERA_HEIGHT_M, floor_height=det.floor_height, ceiling_height=det.ceiling_height,
                camera_radius=det.camera_radius, min_wall_distance=self.min_wall_distance,
                object_collision=det.enable_object_collision, boundary_collision=det.enable_boundary_collision,
                scene_loaded=det.scene_loaded, ref_point=self._ref_point, ref_transition=self._ref_transition)
            self._kernel_key = key
        return self._kernel

    def _ref_point(self, x: float, y: float) -> bool:
        from .contract import pose_from_xy_yaw

        return bool(self.is_pose_legal(pose_from_xy_yaw(x, y, 0.0))[0])

    def _ref_transition(self, fx: float, fy: float, tx: float, ty: float) -> bool:
        res = self.detector.check_collision(np.array([tx, ty, CAMERA_HEIGHT_M], dtype=float),
                                            previous_position=np.array([fx, fy, CAMERA_HEIGHT_M], dtype=float))
        return not res.has_collision

    def legal_mask(self, xy: np.ndarray) -> np.ndarray:
        """(N,) bool, identical to ``is_pose_legal`` of the pitch-0, z = 1.5 m pose at each xy."""
        return self._fast_kernel().legal_mask(np.asarray(xy, dtype=float).reshape(-1, 2))

    def transitions_legal(self, from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
        """(N,) bool, identical to ``not detector.check_collision([to, 1.5], previous_position=[from, 1.5]).has_collision``.

        Unlike ``transition_legal`` there is no rotation shortcut: from == to is
        evaluated by the detector rule (collision iff the point is inside a box
        or within camera_radius of a wall).  ``from_xy`` may be a single (2,) row.
        """
        return self._fast_kernel().transitions_legal(np.asarray(from_xy, dtype=float).reshape(-1, 2),
                                                     np.asarray(to_xy, dtype=float).reshape(-1, 2))

    def rooms_of(self, xy: np.ndarray) -> np.ndarray:
        """(N,) int room index per point (first containing room, -1 for none), == room_of."""
        from .fastgeom import points_room_index

        idx = points_room_index(np.asarray(xy, dtype=float).reshape(-1, 2), [r.polygon for r in self.rooms])
        lookup = np.array([r.index for r in self.rooms] + [-1], dtype=np.int64)
        return lookup[idx]

    def transition_legal(self, c2w_from: np.ndarray, c2w_to: np.ndarray) -> Tuple[bool, str]:
        """Exactly the runtime rule: CollisionDetector.check_collision(new, prev)."""
        a = np.asarray(c2w_from, dtype=float)[:3, 3]
        b = np.asarray(c2w_to, dtype=float)[:3, 3]
        if np.allclose(a, b):
            return True, "rotation"
        res = self.detector.check_collision(b, previous_position=a)
        if res.has_collision:
            return False, f"collision_{res.collision_type}"
        return True, "ok"


def load_scene(scene_dir: Path, scene_id: Optional[str] = None,
               y_sign_override: Optional[float] = None, ignored_ids: Optional[Sequence[str]] = None) -> Scene:
    """Load a scene, calibrate the structure Y sign (v2) and build the detector.

    Sign selection: explicit ``y_sign_override`` > calibration v2 (passed to the
    detector as an override only when v1 would not freeze it) > v1 inside the
    detector.  ``scene.convention`` is the detector's record (what the env will
    report); ``scene.convention_v2`` the full calibration record.

    ``ignored_ids``: labels.json ``ins_id`` values to drop (confirmed annotation
    errors).  They leave ``objects`` and the detector's collision boxes, exactly
    as ``patches/env_ignored_objects.patch`` drops them in the RL env; the
    coordinate calibration still sees every label, as the env's does.
    """
    from vagen.envs.active_spatial.collision_detector import CollisionDetector

    from .coordinates import calibrate_structure_y_sign

    scene_dir = Path(scene_dir)
    scene_id = scene_id or scene_dir.name
    labels = json.loads((scene_dir / "labels.json").read_text(encoding="utf-8"))
    structure_path = scene_dir / "structure.json"
    structure = json.loads(structure_path.read_text(encoding="utf-8")) if structure_path.exists() else {}

    calib = calibrate_structure_y_sign(labels, structure if isinstance(structure, dict) else {})
    calib["scene_id"] = scene_id
    if y_sign_override is not None:
        overrides = {scene_id: float(y_sign_override)}
        source = "explicit_argument"
    elif calib["env_override_required"]:
        overrides = {scene_id: float(calib["sign"])}
        source = "calibration_v2_override"
    else:
        overrides = None
        source = "detector_v1"
    detector = CollisionDetector(camera_radius=CAMERA_RADIUS_M, safety_margin=SAFETY_MARGIN_M,
                                 structure_y_sign_overrides=overrides)
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        detector.load_scene(scene_dir, scene_id=scene_id)
    ignored = {str(i) for i in (ignored_ids or ())}
    if ignored:
        detector.object_boxes = [b for b in detector.object_boxes if str(getattr(b, "ins_id", "")) not in ignored]
    sign = float(detector.structure_y_sign)
    calib["applied_sign"] = sign
    calib["applied_source"] = source
    rooms: List[Room] = []
    for i, room in enumerate(structure.get("rooms", []) if isinstance(structure, dict) else []):
        prof = room.get("profile") or []
        if len(prof) < 3:
            continue
        name = str(room.get("name") or room.get("type") or room.get("room_type") or "").strip().lower()
        rooms.append(Room(index=len(rooms), name=name,
                          polygon=np.array([[float(p[0]), sign * float(p[1])] for p in prof], dtype=float)))
    objects = parse_labels(labels)
    present = {o.id for o in objects}
    if ignored:
        objects = [o for o in objects if o.id not in ignored]
    for obj in objects:
        c = obj.center
        for room in rooms:
            if room.contains(float(c[0]), float(c[1])):
                obj.room_index = room.index
                break
    convention = detector.convention_record()
    return Scene(scene_id=scene_id, scene_dir=scene_dir, objects=objects, rooms=rooms,
                 convention=convention, detector=detector, convention_v2=calib,
                 ignored_object_ids=sorted(ignored & present))
