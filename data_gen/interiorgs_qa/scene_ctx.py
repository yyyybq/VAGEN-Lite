"""One scene as the QA generators see it: v10's scene, oriented boxes, who can be named and how, rooms.

Everything that decides legality, occlusion and naming is imported from main, not copied:

* positions: ``data_gen.interiorgs_v10.scene.Scene.legal_mask`` (v10 collision check, camera at 1.5 m);
* prefilter visibility: ``vagen.envs.active_spatial.target_visibility`` (face_visible_v3);
* scene-unique names: ``data_gen.interiorgs_v10.objects.candidate_objects(scene, "v10_r5")`` (multi-image
  and video questions);
* single-image names: the same single-object filters, but the name only has to be unique in the image
  (``image_unique_name``).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10 import objects as V10O
from data_gen.interiorgs_v10.referring import (category_matches, display_label, effective_room_index,
                                               infer_room_type)
from data_gen.interiorgs_v10.run_scene import ignored_object_ids
from data_gen.interiorgs_v10.scene import load_scene
from vagen.envs.active_spatial import target_visibility as TV

from .geom import OBB, CAMERA_HEIGHT_M, polygon_area

META_ROOT = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_meta")
REPO = Path(__file__).resolve().parents[2]
NO_TASK_LIST = "data_gen/interiorgs_v10/configs/no_task_objects_20261014.json"
NO_TASK_LIST_R5 = "data_gen/interiorgs_v10/configs/no_task_objects_20261013.json"
OBJECT_PRESET = "v10_r5"


class SceneCtx:
    def __init__(self, scene_id: str, meta_root: Path = META_ROOT, no_task_list: str = NO_TASK_LIST,
                 preset: str = OBJECT_PRESET) -> None:
        self.scene_id = scene_id
        self.scene_dir = Path(meta_root) / scene_id
        self.no_task_list = no_task_list
        self.scene = load_scene(self.scene_dir, scene_id, no_task_ids=ignored_object_ids(no_task_list, scene_id))
        self.objects = list(self.scene.objects)
        self.by_id = {o.id: o for o in self.objects}
        raw = json.loads((self.scene_dir / "labels.json").read_text())
        self.obb: Dict[str, OBB] = {}
        for item in raw:
            oid = str(item.get("ins_id") or item.get("id") or "")
            bb = item.get("bounding_box") or []
            if oid in self.by_id and len(bb) == 8:
                xyz = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in bb])
                self.obb[oid] = OBB.from_label_corners(xyz)
        self.disp = {o.id: display_label(o.label) for o in self.objects}
        self.rooms = list(self.scene.rooms)
        self.occ = TV.OccluderSet.from_detector(self.scene.detector)
        self.no_task = set(self.scene.no_task_object_ids)
        # scene-unique names (v10_r5 referring rules)
        kept, funnel = V10O.candidate_objects(self.scene, preset)
        self.funnel = funnel.to_dict()
        self.named: Dict[str, Any] = {o.id: o for o in kept}
        # single-object filters only (no naming rule): the pool of single-image targets
        rooms_present = bool(self.rooms)
        cfg = V10O.get_preset(preset)
        room_map = V10O._room_map(self.scene, self.objects, cfg)
        self.room_of_obj = room_map
        self.eligible = {o.id for o in self.objects
                         if o.id not in self.no_task and o.id in self.obb
                         and V10O.single_object_rejection(o, cfg, room_map.get(o.id), rooms_present) is None}
        self.duplicates = self._duplicate_boxes()
        self.room_types = self._room_types()

    # ------------------------------------------------------------------ objects
    def name(self, oid: str) -> str:
        return self.disp[oid]

    def scene_ref(self, oid: str) -> Optional[str]:
        o = self.named.get(oid)
        return o.ref if o is not None else None

    def scene_ref_anchor(self, oid: str) -> Optional[str]:
        o = self.named.get(oid)
        if o is None or not o.ref_extra:
            return None
        a = o.ref_extra.get("ref_anchor")
        return str(a["id"]) if a else None

    def same_name(self, oid: str) -> List[str]:
        """Every other labels.json object a reader could call by ``oid``'s display name (v10 semantics: "the
        table" also names a dining table), over ALL instances, small or not."""
        d = self.disp[oid]
        return [o.id for o in self.objects if o.id != oid and category_matches(d, self.disp[o.id])]

    def _duplicate_boxes(self) -> Dict[str, str]:
        """id -> id of an identical earlier box of the same display label (labels.json repeats some boxes)."""
        seen: Dict[Tuple, str] = {}
        dup = {}
        for o in self.objects:
            if o.id not in self.obb:
                continue
            key = (self.disp[o.id], tuple(np.round(self.obb[o.id].corners, 3).ravel()))
            if key in seen:
                dup[o.id] = seen[key]
            else:
                seen[key] = o.id
        return dup

    # ------------------------------------------------------------------ rooms
    def _room_types(self) -> Dict[int, Optional[str]]:
        shown: Dict[int, List[str]] = defaultdict(list)
        for o in self.objects:
            r = self.room_of_obj.get(o.id)
            if r is not None:
                shown[int(r)].append(self.disp[o.id])
        return {int(r.index): infer_room_type(shown.get(int(r.index), [])) for r in self.rooms}

    def room_at(self, xy: Sequence[float]) -> Optional[int]:
        return self.scene.room_of(float(xy[0]), float(xy[1]))

    def room_area(self, idx: int) -> float:
        return polygon_area(self.rooms[idx].polygon)

    def objects_in_room(self, idx: int) -> List[str]:
        return [o.id for o in self.objects if self.room_of_obj.get(o.id) == idx]

    # ------------------------------------------------------------------ positions
    def legal(self, xy: np.ndarray) -> np.ndarray:
        """v10 collision rule at camera height 1.5 m (``Scene.legal_mask``)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        return self.scene.legal_mask(xy) if len(xy) else np.zeros(0, bool)

    def legal_transitions(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.scene.transitions_legal(np.asarray(a, float).reshape(-1, 2), np.asarray(b, float).reshape(-1, 2))

    def legal_grid(self, spacing: float = 0.25) -> np.ndarray:
        pts = []
        for r in self.rooms:
            lo, hi = r.polygon.min(0), r.polygon.max(0)
            xs = np.arange(math.ceil(lo[0] / spacing) * spacing, hi[0] + 1e-9, spacing)
            ys = np.arange(math.ceil(lo[1] / spacing) * spacing, hi[1] + 1e-9, spacing)
            g = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
            pts.append(g)
        if not pts:
            return np.zeros((0, 2))
        g = np.unique(np.round(np.concatenate(pts), 4), axis=0)
        return g[self.legal(g)]

    # ------------------------------------------------------------------ prefilter (face_visible_v3)
    def face_visible(self, oid: str, c2ws: np.ndarray, K: np.ndarray, size: int) -> np.ndarray:
        """(N,) the v10 visibility rule (one face clearly visible) at N poses, on the axis-aligned box."""
        o = self.by_id[oid]
        c2ws = np.asarray(c2ws, float).reshape(-1, 4, 4)
        geom = TV.FaceGeometry(o.aabb_min, o.aabb_max)
        blocked = TV.blocked_mask(self.occ, c2ws[:, :3, 3], geom.points, geom.lo, geom.hi, exclude_ids=(oid,))
        stats = TV.face_visibility(c2ws[:, :3, :3], c2ws[:, :3, 3], geom, blocked, K, size, size)
        return stats["visible"]

    def face_seen_any(self, oid: str, c2ws: np.ndarray, K: np.ndarray, size: int) -> np.ndarray:
        """(N,) any sample point of any facing face visible -- used to require that an object is ABSENT."""
        o = self.by_id[oid]
        c2ws = np.asarray(c2ws, float).reshape(-1, 4, 4)
        geom = TV.FaceGeometry(o.aabb_min, o.aabb_max)
        blocked = TV.blocked_mask(self.occ, c2ws[:, :3, 3], geom.points, geom.lo, geom.hi, exclude_ids=(oid,))
        stats = TV.face_visibility(c2ws[:, :3, :3], c2ws[:, :3, 3], geom, blocked, K, size, size)
        return (stats["visible_fraction"] * stats["facing"]).max(axis=1) > 0.0

    # ------------------------------------------------------------------ naming census
    def naming_census(self) -> Dict[str, Any]:
        """How many boxes can be a question's object, and why the rest cannot (v10_r5 funnel), plus how
        many more pass when the name only has to be unique inside one image (single-image questions)."""
        return {"boxes": len(self.objects), "funnel_v10_r5": self.funnel, "scene_named": len(self.named),
                "eligible_single_filters": len(self.eligible),
                "eligible_not_scene_named": len(self.eligible - set(self.named))}


def load_split(path: Path) -> Dict[str, str]:
    m = json.loads(Path(path).read_text())
    return {s: k.replace("_scenes", "") for k in ("train_scenes", "val_scenes", "test_scenes") for s in m.get(k, [])}
