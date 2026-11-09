"""Independent geometry and wording for a versioned two-task QA pilot.

No imports from Active Spatial scorers, renderers, or model packages.  The
mathematics intentionally audits canonical_spatial_task_h1_v1 at 256 x 256.
Success is the task result, not permission to use an observation for training.
Occlusion, ambiguous target names, and malformed RGB require separate review.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

CONTRACT_VERSION = "paired_spatial_visual_h1_v1"
AUDIT_VERSION = "independent_projected_geometry_v1"
TASK_METRIC_VERSION = "canonical_spatial_task_h1_v1"
CAMERA_VERSION = "canonical_camera_h1_resize_v1"


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def _objects(item: dict[str, Any]) -> list[dict[str, Any]]:
    target = item.get("target_object") or {}
    if isinstance(target.get("objects"), list) and target["objects"]:
        return [x for x in target["objects"] if isinstance(x, dict)]
    if isinstance(target.get("primary"), dict):
        return [target["primary"]]
    if "bbox_min" in target and "bbox_max" in target:
        return [target]
    return []


def _center(obj: dict[str, Any]) -> np.ndarray:
    if obj.get("center") is not None:
        return _vector3(obj["center"], "object center")
    return (_vector3(obj["bbox_min"], "bbox_min") +
            _vector3(obj["bbox_max"], "bbox_max")) / 2.0


def _selected(item: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    objects = _objects(item)
    params = (item.get("target_region") or {}).get("params") or {}
    selected, indices, matching = [], [], []
    for fallback, key in enumerate(("object_a_center", "object_b_center")):
        requested = params.get(key)
        if requested is not None and objects:
            target = _vector3(requested, key)
            distances = [float(np.linalg.norm(_center(obj) - target)) for obj in objects]
            index = int(np.argmin(distances))
            ordered = sorted(distances)
            matching.append({"role": "AB"[fallback], "method": "nearest_center",
                             "distance": distances[index],
                             "unique_nearest": len(ordered) < 2 or ordered[1] - ordered[0] > 1e-8})
        elif fallback < len(objects):
            index = fallback
            matching.append({"role": "AB"[fallback], "method": "ordered_fallback",
                             "distance": None, "unique_nearest": True})
        else:
            continue
        selected.append(objects[index])
        indices.append(index)
    return selected, indices, matching


def _native_size(item: dict[str, Any], K: np.ndarray) -> tuple[int, int, str]:
    for key in ("native_resolution", "source_resolution", "camera_native_resolution"):
        size = item.get(key)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            w, h = (int(round(float(x))) for x in size)
            if w <= 0 or h <= 0:
                raise ValueError("native resolution must be positive")
            return w, h, key
    for w_key, h_key in (("native_width", "native_height"),
                         ("source_width", "source_height"),
                         ("camera_native_width", "camera_native_height")):
        if w_key in item and h_key in item:
            w, h = int(round(float(item[w_key]))), int(round(float(item[h_key])))
            if w <= 0 or h <= 0:
                raise ValueError("native resolution must be positive")
            return w, h, f"{w_key}/{h_key}"
    # Match the existing H1 camera's documented, auditable legacy fallback.
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if not math.isfinite(cx + cy) or cx <= 0 or cy <= 0:
        raise ValueError("missing native resolution and invalid principal point")
    w, h = int(round(2 * cx)), int(round(2 * cy))
    if w <= 0 or h <= 0:
        raise ValueError("inferred native resolution must be positive")
    return w, h, "intrinsics_principal_point_center_2x"


def _area(box: list[float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _projection(obj: dict[str, Any], w2c: np.ndarray, K: np.ndarray,
                width: int, height: int, role: str) -> dict[str, Any]:
    low, high = _vector3(obj["bbox_min"], "bbox_min"), _vector3(obj["bbox_max"], "bbox_max")
    if not np.all(high > low):
        raise ValueError("bbox must have positive extent on all three axes")
    corners = np.array(list(itertools.product(*zip(low, high))), dtype=float)
    corners_camera = (w2c @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    center_camera = (w2c @ np.r_[_center(obj), 1.0])[:3]
    forward = corners_camera[:, 2] > 1e-6
    front_center = bool(center_camera[2] > 1e-6)
    center_uv = None
    if front_center:
        center_uv = [float(K[0, 0] * center_camera[0] / center_camera[2] + K[0, 2]),
                     float(K[1, 1] * center_camera[1] / center_camera[2] + K[1, 2])]
    center_inside = bool(center_uv is not None and 0 <= center_uv[0] <= width - 1
                         and 0 <= center_uv[1] <= height - 1)
    result = {"role": role, "object_id": str(obj.get("id", obj.get("ins_id", obj.get("label", "")))),
              "label": obj.get("label"), "available": True,
              "center_in_front": front_center, "center_in_frame": center_inside,
              "center_uv": center_uv, "center_camera": center_camera.tolist(),
              "corners_camera": corners_camera.tolist(),
              "all_corners_in_front": bool(forward.all()),
              "positive_corner_count": int(forward.sum()),
              "visible": False, "bbox_raw": None, "bbox": None, "clipped_bbox": None,
              "area_px": 0.0, "raw_area_px": 0.0, "area_ratio": 0.0,
              "inside_fraction": 0.0, "width_px": 0.0, "height_px": 0.0}
    if not forward.any():
        return result
    points = corners_camera[forward]
    uv = np.column_stack((K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
                          K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]))
    raw = [float(uv[:, 0].min()), float(uv[:, 1].min()),
           float(uv[:, 0].max()), float(uv[:, 1].max())]
    clipped = [max(0.0, raw[0]), max(0.0, raw[1]),
               min(float(width - 1), raw[2]), min(float(height - 1), raw[3])]
    raw_area, clipped_area = _area(raw), _area(clipped)
    clipped_w, clipped_h = max(0.0, clipped[2] - clipped[0]), max(0.0, clipped[3] - clipped[1])
    result.update(bbox_raw=raw, bbox=clipped, clipped_bbox=clipped, area_px=clipped_area, raw_area_px=raw_area,
                  area_ratio=clipped_area / (width * height),
                  inside_fraction=clipped_area / raw_area if raw_area > 1e-8 else 0.0,
                  width_px=clipped_w, height_px=clipped_h,
                  visible=bool(clipped_area > 1 and (center_inside or (clipped_w > 2 and clipped_h > 2))))
    return result


def independent_check(item: dict[str, Any], pose: Any, width: int = 256,
                      height: int = 256) -> dict[str, Any]:
    """Recompute canonical task truth independently; admission is separate.

    The fixed output size prevents silently reinterpreting the 12 px rule.
    No success or projection fields from a production audit are consumed.
    """
    if (width, height) != (256, 256):
        raise ValueError("this frozen contract requires a 256 x 256 observation")
    task = item.get("task_type")
    if task not in {"projective_relations", "fov_inclusion"}:
        raise ValueError(f"unsupported task: {task!r}")
    c2w = np.asarray(pose, dtype=float)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError("pose must be a finite 4 x 4 c2w matrix")
    if not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("pose has an invalid homogeneous row")
    K_native = np.asarray(item["init_camera"]["intrinsics"], dtype=float)
    if K_native.shape == (4, 4):
        K_native = K_native[:3, :3]
    if K_native.shape != (3, 3) or not np.isfinite(K_native).all():
        raise ValueError("native intrinsics must be a finite 3 x 3 or 4 x 4 matrix")
    if K_native[0, 0] <= 0 or K_native[1, 1] <= 0:
        raise ValueError("focal lengths must be positive")
    native_w, native_h, native_source = _native_size(item, K_native)
    K = K_native.copy()
    K[0, 0] *= width / native_w
    K[0, 2] *= width / native_w
    K[1, 1] *= height / native_h
    K[1, 2] *= height / native_h
    w2c = np.linalg.inv(c2w)
    objects, indices, matching = _selected(item)
    projections = [_projection(obj, w2c, K, width, height, "AB"[i])
                   for i, obj in enumerate(objects)]
    two = len(projections) == 2
    fractions = [obj["inside_fraction"] for obj in projections]
    areas = [obj["area_ratio"] for obj in projections]
    gates = {"two_objects": two,
             "in_front": two and all(obj["center_in_front"] for obj in projections),
             "visible": two and all(obj["visible"] for obj in projections),
             "min_area": two and all(value >= 1e-4 for value in areas)}
    centers = [obj["center_uv"] for obj in projections]
    center_margin = None
    if two and all(center is not None for center in centers):
        center_margin = min(min(u, width - u, v, height - v) for u, v in centers)
    relation_margin = None
    if task == "projective_relations":
        relation = (item.get("target_region") or {}).get("params", {}).get("relation", "left")
        if relation not in {"left", "right"}:
            raise ValueError("projective contract supports only left or right")
        if two and all(center is not None for center in centers) and gates["visible"] and all(a > 0 for a in areas):
            relation_margin = float(centers[1][0] - centers[0][0])
            if relation == "right":
                relation_margin = -relation_margin
        gates.update(inside_frame=two and all(value >= 0.5 for value in fractions),
                     relation=relation_margin is not None and relation_margin > 0,
                     margin=relation_margin is not None and relation_margin >= 12.0)
        inside_threshold, margin_threshold = 0.5, 12.0
    else:
        gates.update(full_bbox_in_frame=two and all(value >= 0.95 for value in fractions),
                     center_margin=center_margin is not None and center_margin >= 0.05 * min(width, height))
        inside_threshold, margin_threshold = 0.95, 0.05 * min(width, height)
    all_front = two and all(obj["all_corners_in_front"] for obj in projections)
    distinct = two and len(set(indices)) == 2
    unambiguous = two and all(entry["unique_nearest"] for entry in matching)
    rotation_valid = bool(np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                          and np.isclose(np.linalg.det(c2w[:3, :3]), 1.0, atol=1e-6, rtol=0))
    return {"audit_version": AUDIT_VERSION, "paired_task_contract_version": CONTRACT_VERSION,
            "metric_version": TASK_METRIC_VERSION, "task_type": task,
            "success": bool(all(gates.values())), "gates": gates,
            "relation_margin_px": relation_margin,
            "projections": projections, "target_matching": matching,
            "camera": {"K_native": K_native.tolist(), "K_effective": K.tolist(),
                       "native_resolution": [native_w, native_h], "render_resolution": [width, height],
                       "native_resolution_source": native_source, "camera_model_version": CAMERA_VERSION,
                       "c2w": c2w.tolist(), "w2c": w2c.tolist()},
            "margins": {"relation_margin_px": relation_margin,
                        "relation_threshold_px": 12.0 if task == "projective_relations" else None,
                        "center_margin_px": center_margin,
                        "center_margin_threshold_px": margin_threshold if task == "fov_inclusion" else None,
                        "inside_frame_fractions": fractions, "inside_fraction_threshold": inside_threshold,
                        "inside_fraction_excesses": [v - inside_threshold for v in fractions],
                        "area_ratios": areas, "min_area_ratio": 1e-4,
                        "relation_margin_excess_px": relation_margin - 12 if relation_margin is not None else None},
            "admission": {"distinct_targets": distinct, "unambiguous_center_matching": unambiguous,
                          "all_corners_in_front": all_front, "proper_rotation": rotation_valid,
                          "geometry_admissible": bool(distinct and unambiguous and all_front and rotation_valid),
                          "rgb_review_required": True,
                          "occlusion_verified": False, "unique_target_identity_verified": False}}


def question_variants(item: dict[str, Any]) -> list[str]:
    """Three equivalent, complete questions without private IDs or answers.

    Rectangles refer to projected 3D bounds, never segmentation masks. Manual
    review must reject occlusion and indistinguishable target identities.
    """
    objects, indices, _ = _selected(item)
    if len(objects) != 2 or len(set(indices)) != 2:
        raise ValueError("questions require two distinct target objects")
    names = [str(obj.get("label") or "").replace("_", " ").strip() for obj in objects]
    if not all(names) or names[0].casefold() == names[1].casefold():
        raise ValueError("target labels are missing or ambiguous; require reviewed visual descriptions")
    a, b = names
    # A fixed common definition makes all gates available to the reader.
    common = ("Use the rectangular projections of the objects' full 3D bounds, not only their visible pixels. "
              "Both objects must be in front of the camera and visible in the image, and each projected rectangle "
              "must cover at least 0.01% of the image after clipping at the borders. ")
    if item.get("task_type") == "projective_relations":
        relation = (item.get("target_region") or {}).get("params", {}).get("relation", "left")
        if relation not in {"left", "right"}:
            raise ValueError("relation must be left or right")
        questions = [
            f"In this 256×256 view, is the projected centre of the {a} at least 12 pixels to the {relation} of the {b}'s projected centre, with at least half of each projected rectangle inside the image?",
            f"Does this 256×256 view show the {a}'s projected centre 12 pixels or more to the {relation} of the {b}'s, while at least 50% of each projected rectangle remains in frame?",
            f"Has this goal already been met in the 256×256 image: the {a}'s projected centre is at least 12 pixels to the {relation} of the {b}'s projected centre, and at least half of both projected rectangles is in frame?",
        ]
    elif item.get("task_type") == "fov_inclusion":
        questions = [
            f"Are the {a} and the {b} both framed with at least 95% of each projected rectangle inside the image and both projected centres at least 5% of the image's shorter side away from every border?",
            f"Does the current view meet this framing goal for the {a} and the {b}: at least 95% of both projected rectangles stays in the image, and each projected centre has a border margin of at least 5% of the shorter image side?",
            f"Have the {a} and the {b} both been framed so that no more than 5% of either projected rectangle lies outside the image, and both projected centres are at least 5% of the shorter image side from every edge?",
        ]
    else:
        raise ValueError("question templates support only projective_relations and fov_inclusion")
    return [common + question + " Answer Yes or No." for question in questions]
