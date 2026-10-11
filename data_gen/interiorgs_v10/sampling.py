"""Shared helpers: canonical task items and candidate pose grids.

Kept intentionally small so that independently developed modules (endpoints,
families, initial states) can depend on it from day one.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contract import (
    ACTION_PROTOCOL_VERSION,
    CAMERA_MODEL_VERSION,
    CANONICAL_TASK_METRIC_V1,
    GENERATOR_VERSION,
    NATIVE_K,
    Obj,
    pose_from_xy_yaw,
)


def projective_item(scene_id: str, obj_a: Obj, obj_b: Obj, relation: str,
                    init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Canonical v1 Projective item: 'A appears to the <relation> of B'."""
    if relation not in ("left", "right"):
        raise ValueError(relation)
    a_ref = obj_a.ref or f"the {obj_a.label}"
    b_ref = obj_b.ref or f"the {obj_b.label}"
    return {
        "scene_id": scene_id,
        "task_type": "projective_relations",
        "task_description": f"Move so that {a_ref} appears to the {relation} of {b_ref} in your view.",
        "target_object": {"objects": [obj_a.target_object_entry(), obj_b.target_object_entry()]},
        "target_region": {
            "type": "projective",
            "params": {
                "relation": relation,
                "object_a_center": [float(v) for v in obj_a.center],
                "object_b_center": [float(v) for v in obj_b.center],
            },
        },
        "init_camera": {
            "intrinsics": NATIVE_K,
            "extrinsics": (np.asarray(init_c2w).tolist() if init_c2w is not None else None),
        },
        "canonical_task_metric_version": CANONICAL_TASK_METRIC_V1,
        "camera_model_version": CAMERA_MODEL_VERSION,
        "action_protocol_version": ACTION_PROTOCOL_VERSION,
        "generator_version": GENERATOR_VERSION,
    }


def fov_item(scene_id: str, obj_a: Obj, obj_b: Obj, init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    a_ref = obj_a.ref or f"the {obj_a.label}"
    b_ref = obj_b.ref or f"the {obj_b.label}"
    item = projective_item(scene_id, obj_a, obj_b, "left", init_c2w)
    item.update({
        "task_type": "fov_inclusion",
        "task_description": f"Move so that both {a_ref} and {b_ref} are fully inside your view.",
    })
    item["target_region"] = {
        "type": "fov",
        "params": {
            "object_a_center": [float(v) for v in obj_a.center],
            "object_b_center": [float(v) for v in obj_b.center],
        },
    }
    return item


def score(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    """Score with the frozen runtime canonical metric (projective / fov)."""
    from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task

    return score_canonical_task(item, np.asarray(c2w, dtype=float))


def floor_grid(scene: Any, spacing: float = 0.3, room_indices: Optional[Iterable[int]] = None,
               jitter: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """(N, 2) candidate camera xy positions inside the given rooms that pass
    ``scene.is_pose_legal`` (legality does not depend on yaw at pitch 0)."""
    rooms = [r for r in scene.rooms if room_indices is None or r.index in set(room_indices)]
    pts: List[Tuple[float, float]] = []
    for room in rooms:
        lo = room.polygon.min(axis=0)
        hi = room.polygon.max(axis=0)
        xs = np.arange(lo[0] + spacing / 2, hi[0], spacing)
        ys = np.arange(lo[1] + spacing / 2, hi[1], spacing)
        for x in xs:
            for y in ys:
                px, py = float(x), float(y)
                if jitter > 0 and rng is not None:
                    px += float(rng.uniform(-jitter, jitter))
                    py += float(rng.uniform(-jitter, jitter))
                if not room.contains(px, py):
                    continue
                ok, _ = scene.is_pose_legal(pose_from_xy_yaw(px, py, 0.0))
                if ok:
                    pts.append((px, py))
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def yaw_set(step_deg: float = 10.0) -> np.ndarray:
    return np.radians(np.arange(0.0, 360.0, step_deg))
