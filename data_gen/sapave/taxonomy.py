"""SaPaVe task taxonomy for ActiveViewPose / ActiveManip-Bench splits.

Official ActiveViewPose-200K and ActiveManip-Bench assets are not released.
This module maps existing InteriorGS active_spatial JSONL samples onto the
paper's evaluation protocol:

* ActiveViewPose modalities (Appendix B.3): visual centering, spatial
  directive, commonsense, conditional reasoning, container interaction.
* ActiveViewPose splits (Sec. 4.1): Train / Val / Test1 / Test2, where
  Test1 has explicit positional language and Test2 requires inference.
* ActiveManip-Bench visibility (Appendix C.2.2): unoccluded, occluded,
  out-of-view, crossed with pick-and-place vs articulated families.

Original ``task_type`` and ``task_description`` are preserved so the
active_spatial env scorer stays aligned with the potential field.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SAPAVE_MODALITIES = (
    "visual_centering",
    "spatial_directive",
    "commonsense",
    "conditional_reasoning",
    "container_interaction",
)

SAPAVE_VISIBILITY = ("unoccluded", "occluded", "out_of_view")
SAPAVE_MANIP_FAMILIES = ("pick_and_place", "articulated")
SAPAVE_SPLITS = ("train", "val", "test1", "test2")

EXCLUDE_TASK_TYPES = frozenset({"delta_control"})

CONTAINER_KEYWORDS = (
    "cabinet",
    "wardrobe",
    "drawer",
    "cupboard",
    "bookshelf",
    "shelf",
    "fridge",
    "refrigerator",
    "oven",
    "microwave",
    "door",
)

EXPLICIT_SPATIAL_TOKENS = (
    "left",
    "right",
    "behind",
    "hidden",
    "front",
    "back",
    "turn",
    "look up",
    "look down",
    "on the left",
    "on the right",
)

OOD_TEST_SCENES = (
    "0229_840306",
    "0240_840881",
    "0266_840789",
    "0275_840778",
    "0276_840780",
    "0351_840366",
    "0367_840260",
    "0374_840227",
)

# Scene-held-out ID val. Chosen for moderate size and mixed task coverage
# rather than taking the largest training scenes.
DEFAULT_VAL_SCENES = (
    "0012_840878",
    "0238_840875",
    "0253_840830",
    "0322_840394",
    "0342_840385",
)


def _lower(text: Any) -> str:
    return str(text or "").strip().lower()


def _objects(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    target = item.get("target_object")
    if isinstance(target, dict):
        if isinstance(target.get("objects"), list):
            return [obj for obj in target["objects"] if isinstance(obj, dict)]
        if isinstance(target.get("primary"), dict):
            return [target["primary"]]
        return [target]
    if isinstance(target, list):
        return [obj for obj in target if isinstance(obj, dict)]
    return []


def object_label_text(item: Dict[str, Any]) -> str:
    label = _lower(item.get("object_label"))
    if label:
        return label
    parts = []
    for obj in _objects(item):
        parts.append(_lower(obj.get("label")))
    return "+".join(p for p in parts if p)


def is_container_item(item: Dict[str, Any]) -> bool:
    label = object_label_text(item)
    return any(token in label for token in CONTAINER_KEYWORDS)


def has_explicit_spatial_language(item: Dict[str, Any]) -> bool:
    desc = _lower(item.get("task_description"))
    preset = _lower(item.get("preset"))
    if any(token in desc for token in EXPLICIT_SPATIAL_TOKENS):
        return True
    if preset in {"left", "right", "left_of", "right_of", "front", "back", "occluded"}:
        return True
    return item.get("task_type") in {"projective_relations", "occlusion_alignment"}


def sapave_modality(item: Dict[str, Any]) -> str:
    """Map one active_spatial sample onto an ActiveViewPose prompt modality."""
    task = item.get("task_type")
    container = is_container_item(item)

    if task in {"centering", "screen_occupancy"}:
        return "visual_centering"
    if task in {"size_distance_invariance", "apparent_size_ordering"}:
        return "conditional_reasoning"
    if task in {"projective_relations", "occlusion_alignment"}:
        return "spatial_directive"
    if container:
        return "container_interaction"
    if has_explicit_spatial_language(item):
        return "spatial_directive"
    return "commonsense"


def sapave_manip_family(item: Dict[str, Any]) -> str:
    return "articulated" if is_container_item(item) else "pick_and_place"


def _bbox_corners(obj: Dict[str, Any]) -> Optional[List[List[float]]]:
    mins = obj.get("bbox_min")
    maxs = obj.get("bbox_max")
    if not (isinstance(mins, (list, tuple)) and isinstance(maxs, (list, tuple))):
        return None
    if len(mins) < 3 or len(maxs) < 3:
        return None
    xs = [float(mins[0]), float(maxs[0])]
    ys = [float(mins[1]), float(maxs[1])]
    zs = [float(mins[2]), float(maxs[2])]
    return [[x, y, z] for x in xs for y in ys for z in zs]


def _image_size_from_k(k: Sequence[Sequence[float]]) -> Tuple[int, int]:
    width = max(int(round(2.0 * float(k[0][2]))), 1)
    height = max(int(round(2.0 * float(k[1][2]))), 1)
    return width, height


def _matmul4(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(float(a[i][k]) * float(b[k][j]) for k in range(4))
    return out


def _invert4(m: Sequence[Sequence[float]]) -> Optional[List[List[float]]]:
    a = [list(map(float, row)) + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        a[col] = [v / div for v in a[col]]
        for row in range(4):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [a[row][c] - factor * a[col][c] for c in range(8)]
    return [row[4:] for row in a]


def _transform_point(mat: Sequence[Sequence[float]], xyz: Sequence[float]) -> List[float]:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return [
        mat[0][0] * x + mat[0][1] * y + mat[0][2] * z + mat[0][3],
        mat[1][0] * x + mat[1][1] * y + mat[1][2] * z + mat[1][3],
        mat[2][0] * x + mat[2][1] * y + mat[2][2] * z + mat[2][3],
    ]


def project_object_visibility(item: Dict[str, Any]) -> Dict[str, Any]:
    """Project target AABB corners into the init camera."""
    camera = item.get("init_camera") or {}
    try:
        k = camera["intrinsics"]
        c2w = camera["extrinsics"]
        w2c = _invert4(c2w)
        if w2c is None:
            raise ValueError("singular camera matrix")
    except Exception:
        return {"ok": False, "n_objects": 0, "in_frame_corners": [], "in_front_corners": []}

    width, height = _image_size_from_k(k)
    in_frame: List[int] = []
    in_front: List[int] = []
    for obj in _objects(item):
        corners = _bbox_corners(obj)
        if corners is None:
            continue
        n_front = 0
        n_inside = 0
        for corner in corners:
            cam = _transform_point(w2c, corner)
            z = cam[2]
            if z <= 1e-6:
                continue
            n_front += 1
            u = (float(k[0][0]) * cam[0] + float(k[0][1]) * cam[1] + float(k[0][2]) * cam[2]) / z
            v = (float(k[1][0]) * cam[0] + float(k[1][1]) * cam[1] + float(k[1][2]) * cam[2]) / z
            if 0.0 <= u < width and 0.0 <= v < height:
                n_inside += 1
        in_front.append(n_front)
        in_frame.append(n_inside)
    return {
        "ok": True,
        "n_objects": len(in_frame),
        "in_frame_corners": in_frame,
        "in_front_corners": in_front,
        "image_size": [width, height],
    }



def yaw_c2w(c2w: Sequence[Sequence[float]], yaw_deg: float) -> List[List[float]]:
    """Rotate camera orientation about world +Z, keeping position fixed."""
    import math
    rad = math.radians(float(yaw_deg))
    c, s = math.cos(rad), math.sin(rad)
    r00, r01, r02 = float(c2w[0][0]), float(c2w[0][1]), float(c2w[0][2])
    r10, r11, r12 = float(c2w[1][0]), float(c2w[1][1]), float(c2w[1][2])
    r20, r21, r22 = float(c2w[2][0]), float(c2w[2][1]), float(c2w[2][2])
    # new_R = Rz @ R, translation unchanged
    return [
        [c * r00 - s * r10, c * r01 - s * r11, c * r02 - s * r12, float(c2w[0][3])],
        [s * r00 + c * r10, s * r01 + c * r11, s * r02 + c * r12, float(c2w[1][3])],
        [r20, r21, r22, float(c2w[2][3])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def make_out_of_view_item(item: Dict[str, Any], yaw_candidates: Sequence[float] = (90.0, -90.0, 120.0, -120.0, 150.0, 180.0)) -> Optional[Dict[str, Any]]:
    """Synthesize an out-of-view init by yawing the existing camera."""
    camera = item.get("init_camera") or {}
    c2w = camera.get("extrinsics")
    if not c2w:
        return None
    for yaw in yaw_candidates:
        candidate = dict(item)
        new_cam = dict(camera)
        new_cam["extrinsics"] = yaw_c2w(c2w, yaw)
        candidate["init_camera"] = new_cam
        proj = project_object_visibility(candidate)
        frames = proj.get("in_frame_corners") or []
        if proj.get("ok") and frames and max(frames) <= 0:
            candidate["sapave_init_synthesized"] = True
            candidate["sapave_init_yaw_deg"] = float(yaw)
            candidate["sapave_visibility"] = "out_of_view"
            return candidate
    return None


def sapave_visibility(item: Dict[str, Any]) -> str:
    """Label init-view visibility using the paper's three-level protocol."""
    proj = project_object_visibility(item)
    if not proj["ok"] or not proj["in_frame_corners"]:
        return "unknown"

    task = item.get("task_type")
    if task == "occlusion_alignment" or _lower(item.get("preset")) == "occluded":
        return "occluded"

    frames: Sequence[int] = proj["in_frame_corners"]
    max_in = max(frames)
    min_in = min(frames)
    if max_in <= 0:
        return "out_of_view"
    if min_in <= 2 or max_in <= 3:
        return "occluded"
    return "unoccluded"


def avp_eval_bucket(item: Dict[str, Any]) -> str:
    """Test1 = explicit positional language; Test2 = inferred camera motion."""
    if has_explicit_spatial_language(item):
        return "test1"
    return "test2"


def annotate_item(item: Dict[str, Any], split: Optional[str] = None) -> Dict[str, Any]:
    out = dict(item)
    out["sapave_modality"] = sapave_modality(item)
    out["sapave_visibility"] = sapave_visibility(item)
    out["sapave_manip_family"] = sapave_manip_family(item)
    out["sapave_explicit_spatial"] = has_explicit_spatial_language(item)
    if split is not None:
        out["sapave_split"] = split
    return out


def keep_for_sapave(item: Dict[str, Any]) -> bool:
    return item.get("task_type") not in EXCLUDE_TASK_TYPES


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    return {
        "n": len(rows),
        "n_scenes": len({r.get("scene_id") for r in rows}),
        "modality": dict(Counter(r.get("sapave_modality") for r in rows)),
        "visibility": dict(Counter(r.get("sapave_visibility") for r in rows)),
        "manip_family": dict(Counter(r.get("sapave_manip_family") for r in rows)),
        "task_type": dict(Counter(r.get("task_type") for r in rows)),
        "split": dict(Counter(r.get("sapave_split") for r in rows)),
    }
