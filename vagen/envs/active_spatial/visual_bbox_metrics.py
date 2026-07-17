"""Projected 3D-bbox visual relation metrics for Active Spatial tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _as_array(value: Any, default: Any = None) -> Optional[np.ndarray]:
    if value is None:
        value = default
    if value is None:
        return None
    try:
        return np.array(value, dtype=np.float64)
    except Exception:
        return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-10:
        return vec
    return vec / norm


def _sigmoid(x: float, scale: float = 1.0) -> float:
    scale = max(float(scale), 1e-6)
    return float(1.0 / (1.0 + math.exp(-x / scale)))


def _gaussian_error(error: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    return float(math.exp(-0.5 * (float(error) / sigma) ** 2))


def _bbox_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return np.array(
        [[x, y, z] for x in (bmin[0], bmax[0])
         for y in (bmin[1], bmax[1])
         for z in (bmin[2], bmax[2])],
        dtype=np.float64,
    )


def _fallback_intrinsics(width: int, height: int, hfov_deg: float = 90.0) -> np.ndarray:
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_pose_from_forward(camera_position: np.ndarray, camera_forward: np.ndarray) -> np.ndarray:
    forward = _normalize(np.array(camera_forward[:3], dtype=np.float64))
    if float(np.linalg.norm(forward)) < 1e-8:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = _normalize(right)
    # Match data_gen.active_spatial_pipeline.look_at_matrix:
    # local X is right, local Y is image-down, local Z is visual forward.
    down = _normalize(np.cross(forward, right))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_position[:3]
    return c2w


def _camera_setup(
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    task_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    width = int(task_params.get("_image_width", 640) or 640)
    height = int(task_params.get("_image_height", 480) or 480)
    K = _as_array(task_params.get("_camera_intrinsics"))
    if K is None:
        K = _fallback_intrinsics(width, height, float(task_params.get("_fov_horizontal", 90.0)))
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        K = _fallback_intrinsics(width, height, float(task_params.get("_fov_horizontal", 90.0)))

    pose = _as_array(task_params.get("_camera_pose_c2w"))
    if pose is None or pose.shape != (4, 4):
        pose = _camera_pose_from_forward(camera_position, camera_forward)
    return pose, K, width, height


def _extract_objects(task_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_object = task_params.get("_target_object") or task_params.get("target_object") or {}
    if not isinstance(target_object, dict):
        return []
    objects = target_object.get("objects")
    if isinstance(objects, list) and objects:
        return [obj for obj in objects if isinstance(obj, dict)]
    primary = target_object.get("primary")
    if isinstance(primary, dict):
        return [primary]
    if "bbox_min" in target_object and "bbox_max" in target_object:
        return [target_object]
    return []


def _object_id(obj: Dict[str, Any]) -> str:
    return str(obj.get("id", obj.get("ins_id", obj.get("label", ""))))


def _match_object(
    objects: List[Dict[str, Any]],
    object_id: Any = None,
    center: Any = None,
    fallback_index: int = 0,
) -> Optional[Dict[str, Any]]:
    if object_id is not None:
        oid = str(object_id)
        for obj in objects:
            if oid in {_object_id(obj), str(obj.get("ins_id", "")), str(obj.get("label", ""))}:
                return obj
    if center is not None:
        c = _as_array(center)
        if c is not None:
            best_obj = None
            best_dist = float("inf")
            for obj in objects:
                oc = _as_array(obj.get("center"))
                if oc is None and "bbox_min" in obj and "bbox_max" in obj:
                    bmin = _as_array(obj.get("bbox_min"))
                    bmax = _as_array(obj.get("bbox_max"))
                    if bmin is not None and bmax is not None:
                        oc = (bmin + bmax) / 2.0
                if oc is None:
                    continue
                dist = float(np.linalg.norm(oc[:3] - c[:3]))
                if dist < best_dist:
                    best_dist = dist
                    best_obj = obj
            if best_obj is not None:
                return best_obj
    if 0 <= fallback_index < len(objects):
        return objects[fallback_index]
    return None


def _project_object(obj: Dict[str, Any], c2w: np.ndarray, K: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    bmin = _as_array(obj.get("bbox_min"))
    bmax = _as_array(obj.get("bbox_max"))
    if bmin is None or bmax is None:
        return {"available": False, "label": obj.get("label"), "id": _object_id(obj)}

    corners = _bbox_corners(bmin, bmax)
    center = _as_array(obj.get("center"), (bmin + bmax) / 2.0)
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    pts_c = (w2c @ pts_h.T).T[:, :3]
    center_c = (w2c @ np.array([center[0], center[1], center[2], 1.0]))[:3]

    projected = []
    depths = []
    for pc in pts_c:
        if pc[2] <= 1e-6:
            continue
        u = float(K[0, 0] * pc[0] / pc[2] + K[0, 2])
        v = float(K[1, 1] * pc[1] / pc[2] + K[1, 2])
        projected.append([u, v])
        depths.append(float(pc[2]))

    center_uv = None
    center_in_front = bool(center_c[2] > 1e-6)
    if center_in_front:
        center_uv = [
            float(K[0, 0] * center_c[0] / center_c[2] + K[0, 2]),
            float(K[1, 1] * center_c[1] / center_c[2] + K[1, 2]),
        ]

    if not projected:
        return {
            "available": True,
            "label": obj.get("label"),
            "id": _object_id(obj),
            "visible": False,
            "center_in_front": center_in_front,
            "center_uv": center_uv,
            "depth": float(center_c[2]),
            "area_px": 0.0,
            "area_ratio": 0.0,
            "height_px": 0.0,
            "width_px": 0.0,
        }

    pts = np.array(projected, dtype=np.float64)
    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(width - 1), x2), min(float(height - 1), y2)
    clipped_w = max(0.0, cx2 - cx1)
    clipped_h = max(0.0, cy2 - cy1)
    area_px = clipped_w * clipped_h
    center_in_frame = bool(
        center_uv is not None
        and 0.0 <= center_uv[0] <= width - 1
        and 0.0 <= center_uv[1] <= height - 1
    )
    visible = bool(area_px > 1.0 and (center_in_frame or clipped_w > 2.0 and clipped_h > 2.0))
    return {
        "available": True,
        "label": obj.get("label"),
        "id": _object_id(obj),
        "visible": visible,
        "center_in_front": center_in_front,
        "center_in_frame": center_in_frame,
        "center_uv": center_uv,
        "depth": float(np.mean(depths)) if depths else float(center_c[2]),
        "min_depth": float(np.min(depths)) if depths else float(center_c[2]),
        "bbox_raw": [float(x1), float(y1), float(x2), float(y2)],
        "bbox": [float(cx1), float(cy1), float(cx2), float(cy2)],
        "area_px": float(area_px),
        "area_ratio": float(area_px / max(width * height, 1)),
        "height_px": float(clipped_h),
        "width_px": float(clipped_w),
    }


def _overlap(box_a: Optional[List[float]], box_b: Optional[List[float]]) -> Tuple[float, float, float]:
    if box_a is None or box_b is None:
        return 0.0, 0.0, 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-6)
    return float(inter), float(inter / max(min(area_a, area_b), 1e-6)), float(inter / union)


def _visibility_score(*projs: Dict[str, Any], min_area_ratio: float = 1e-4) -> float:
    scores = []
    for proj in projs:
        if not proj or not proj.get("available") or not proj.get("visible"):
            scores.append(0.0)
            continue
        area_score = min(1.0, float(proj.get("area_ratio", 0.0)) / max(min_area_ratio, 1e-8))
        scores.append(area_score)
    return float(min(scores)) if scores else 0.0


def _finite_center_u(proj: Dict[str, Any]) -> Optional[float]:
    center = proj.get("center_uv") if proj else None
    if center is None:
        return None
    try:
        value = float(center[0])
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _basic_payload(task_type: str, objects: List[Dict[str, Any]], projections: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "available": bool(objects and all(p.get("available") for p in projections)),
        "task_type": task_type,
        "objects": [
            {
                "label": p.get("label"),
                "id": p.get("id"),
                "visible": p.get("visible"),
                "center_uv": p.get("center_uv"),
                "bbox": p.get("bbox"),
                "area_ratio": p.get("area_ratio"),
                "height_px": p.get("height_px"),
                "depth": p.get("depth"),
            }
            for p in projections
        ],
    }


def compute_visual_bbox_metrics(
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
) -> Dict[str, Any]:
    params = target_region.get("params", {}) if isinstance(target_region, dict) else {}
    objects = _extract_objects(task_params)
    if not objects:
        return {"available": False, "reason": "missing_target_object_bboxes"}

    c2w, K, width, height = _camera_setup(camera_position, camera_forward, task_params)

    def pick(idx: int, object_id: Any = None, center_key: str = "") -> Optional[Dict[str, Any]]:
        return _match_object(objects, object_id=object_id, center=params.get(center_key) if center_key else None, fallback_index=idx)

    if task_type == "screen_occupancy":
        selected = pick(0, center_key="object_center")
        projs = [_project_object(selected, c2w, K, width, height)] if selected else []
        payload = _basic_payload(task_type, objects, projs)
        if not projs:
            return payload
        p = projs[0]
        occupancy = float(p.get("height_px", 0.0)) / max(float(height), 1.0)
        target = float(params.get("occupancy_ratio", 0.5))
        occupancy_score = _gaussian_error(abs(occupancy - target), max(target * 0.20, 0.03))
        visibility = _visibility_score(p, min_area_ratio=1e-4)
        payload.update({
            "visual_score": occupancy_score * visibility,
            "visual_position_score": occupancy_score * visibility,
            "visual_orientation_score": visibility,
            "visual_occupancy": occupancy,
            "target_occupancy": target,
            "visibility_score": visibility,
        })
        return payload

    if task_type in {"projective_relations", "fov_inclusion", "size_distance_invariance"}:
        obj_a = pick(0, center_key="object_a_center")
        obj_b = pick(1, center_key="object_b_center")
        projs = [_project_object(o, c2w, K, width, height) for o in (obj_a, obj_b) if o is not None]
        payload = _basic_payload(task_type, objects, projs)
        if len(projs) < 2:
            return payload
        a, b = projs
        visibility = _visibility_score(a, b, min_area_ratio=1e-4)
        if task_type == "projective_relations":
            au = _finite_center_u(a)
            bu = _finite_center_u(b)
            relation = params.get("relation", "left")
            if au is None or bu is None or visibility <= 0.0:
                margin = -float("inf")
                relation_score = 0.0
            else:
                margin = (bu - au) if relation == "left" else (au - bu)
                scale = max((float(a.get("width_px", 0.0)) + float(b.get("width_px", 0.0))) * 0.125, 8.0)
                relation_score = _sigmoid(margin, scale=scale)
            payload.update({
                "visual_score": relation_score * visibility,
                "visual_position_score": relation_score * visibility,
                "visual_orientation_score": visibility,
                "visual_relation_satisfied": bool(math.isfinite(margin) and margin > 0),
                "visual_relation_margin_px": float(margin) if math.isfinite(margin) else None,
                "visibility_score": visibility,
            })
        elif task_type == "fov_inclusion":
            centers = [p.get("center_uv") for p in projs if p.get("center_uv") is not None]
            if centers:
                center_error = max(abs(c[0] - width / 2.0) / max(width / 2.0, 1.0) for c in centers)
                center_score = max(0.0, 1.0 - center_error)
            else:
                center_score = 0.0
            payload.update({
                "visual_score": visibility * (0.7 + 0.3 * center_score),
                "visual_position_score": visibility,
                "visual_orientation_score": center_score * visibility,
                "visibility_score": visibility,
                "center_score": center_score,
            })
        else:
            ha, hb = max(float(a.get("height_px", 0.0)), 1e-6), max(float(b.get("height_px", 0.0)), 1e-6)
            ratio = ha / hb
            ratio_score = math.exp(-abs(math.log(max(ratio, 1e-6))))
            payload.update({
                "visual_score": ratio_score * visibility,
                "visual_position_score": ratio_score * visibility,
                "visual_orientation_score": visibility,
                "visual_size_ratio": ratio,
                "visibility_score": visibility,
            })
        return payload

    if task_type == "apparent_size_ordering":
        large = _match_object(objects, object_id=params.get("larger_object"), center=params.get("larger_object_center"), fallback_index=0)
        small = _match_object(objects, object_id=params.get("smaller_object"), center=params.get("smaller_object_center"), fallback_index=1)
        projs = [_project_object(o, c2w, K, width, height) for o in (large, small) if o is not None]
        payload = _basic_payload(task_type, objects, projs)
        if len(projs) < 2:
            return payload
        large_p, small_p = projs
        visibility = _visibility_score(large_p, small_p, min_area_ratio=1e-4)
        h_large = max(float(large_p.get("height_px", 0.0)), 1e-6)
        h_small = max(float(small_p.get("height_px", 0.0)), 1e-6)
        ratio = h_large / h_small
        target_ratio = max(float(params.get("target_ratio", 1.25)), 1.01)
        violation = max(0.0, math.log(target_ratio) - math.log(max(ratio, 1e-6)))
        ordering_score = 1.0 if violation == 0.0 else math.exp(-violation / 0.35)
        payload.update({
            "visual_score": ordering_score * visibility,
            "visual_position_score": ordering_score * visibility,
            "visual_orientation_score": visibility,
            "visual_apparent_ratio": ratio,
            "target_ratio": target_ratio,
            "visual_ordering_satisfied": bool(ratio >= target_ratio),
            "visibility_score": visibility,
        })
        return payload

    if task_type == "centering":
        obj_a = pick(0, center_key="object_a_center")
        obj_b = pick(1, center_key="object_b_center")
        obj_c = pick(2, center_key="object_c_center")
        projs = [_project_object(o, c2w, K, width, height) for o in (obj_a, obj_b, obj_c) if o is not None]
        payload = _basic_payload(task_type, objects, projs)
        if len(projs) < 3:
            return payload
        a, b, c = projs
        visibility = _visibility_score(a, b, c, min_area_ratio=1e-4)
        au = _finite_center_u(a)
        bu = _finite_center_u(b)
        cu = _finite_center_u(c)
        if au is None or bu is None or cu is None or visibility <= 0.0:
            sep = 0.0
            error = float("inf")
            center_score = 0.0
            between = False
            sep_score = 0.0
        else:
            sep = abs(bu - cu)
            midpoint = (bu + cu) / 2.0
            error = abs(au - midpoint) / max(sep, 8.0)
            center_score = _gaussian_error(error, sigma=0.20)
            between = min(bu, cu) <= au <= max(bu, cu)
            if not between:
                center_score *= 0.5
            sep_score = min(1.0, sep / max(width * 0.10, 12.0))
        payload.update({
            "visual_score": center_score * sep_score * visibility,
            "visual_position_score": center_score * sep_score * visibility,
            "visual_orientation_score": visibility,
            "visual_center_error": float(error) if math.isfinite(error) else None,
            "visual_between": bool(between),
            "visual_bc_separation_px": float(sep),
            "visibility_score": visibility,
        })
        return payload

    if task_type == "occlusion_alignment":
        target = pick(0, object_id=params.get("occluded"), center_key="object_a_center")
        occluder = pick(1, object_id=params.get("occluder"), center_key="object_b_center")
        projs = [_project_object(o, c2w, K, width, height) for o in (target, occluder) if o is not None]
        payload = _basic_payload(task_type, objects, projs)
        if len(projs) < 2:
            return payload
        target_p, occ_p = projs
        # The hidden target must project into the frame; otherwise this is an
        # "outside frame" shortcut, not occlusion.
        target_proj = 1.0 if target_p.get("area_px", 0.0) > 1.0 else 0.0
        occ_vis = _visibility_score(occ_p, min_area_ratio=1e-4)
        inter, overlap_min, iou = _overlap(target_p.get("bbox"), occ_p.get("bbox"))
        depth_ok = bool(float(occ_p.get("depth", float("inf"))) < float(target_p.get("depth", -float("inf"))))
        depth_score = 1.0 if depth_ok else 0.1
        overlap_score = min(1.0, overlap_min / 0.25)
        center_a = target_p.get("center_uv") or [0.0, 0.0]
        center_b = occ_p.get("center_uv") or [float("inf"), float("inf")]
        center_dist = math.sqrt((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2)
        scale = max(float(occ_p.get("width_px", 0.0)), float(occ_p.get("height_px", 0.0)), 8.0)
        alignment_score = math.exp(-center_dist / scale)
        visual = target_proj * occ_vis * depth_score * max(overlap_score, 0.5 * alignment_score * overlap_score)
        payload.update({
            "visual_score": visual,
            "visual_position_score": overlap_score * depth_score * target_proj * occ_vis,
            "visual_orientation_score": occ_vis,
            "occlusion_overlap_min": overlap_min,
            "occlusion_iou": iou,
            "occlusion_intersection_px": inter,
            "occluder_in_front": depth_ok,
            "target_projects_into_frame": bool(target_proj > 0),
            "center_alignment_score": alignment_score,
            "visibility_score": occ_vis,
        })
        return payload

    return {"available": False, "reason": f"task_not_visualized:{task_type}"}
