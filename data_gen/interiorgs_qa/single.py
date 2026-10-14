"""Single-image groups S1-S7 (user prompt, section 6): choose the objects first, then the camera.

Every group follows the same loop:

1. ``combos``: object sets that satisfy the group's 3D conditions, each with legal cameras that frame them
   and pass the v10 visibility prefilter plus a group-specific image check on the projected boxes (CPU);
2. ``realize``: render the best camera(s), recompute everything on the visible parts from the rendered
   depth (visible.py), name the objects (unique in the image), and emit the question when the answer is
   clear by a margin; otherwise record why not.

Names: single-image questions only need the object to be unique in this image (GenCtx.name_single).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .common import (Combo, GenCtx, bare, bbox_gap_x, bbox_gap_y, cap, distance, overlap_1d, pairs_near,
                     single_pool)
from .draw import draw_boxes, draw_points, flat_surface_points
from .geom import OBB, angle_diff_deg, camera_K, footprint_overlap_fraction, project, to_camera
from .qa import Phrasing
from .views import frame_candidates

SIZE = 512
K512 = camera_K(SIZE)

SUBTASKS = {
    "S1": ["ES-01", "ES-02", "VS-01", "BL-03"],
    "S2": ["ES-03", "ES-04", "3D-01", "3D-02", "BL-03"],
    "S3": ["ES-05", "ES-06", "3D-03", "BL-02"],
    "S4": ["3D-04", "3D-08", "BL-03"],
    "S5": ["BL-01"],
    "S6": ["BL-05"],
    "S7": ["3D-11"],
}
BENCH = {"ES": "EmbSpatial-Bench", "VS": "ViewSpatial-Bench", "BL": "BLINK", "3D": "3DSRBench"}


def bench_of(subtask: str) -> str:
    return BENCH[subtask.split("-")[0]]


def proj_rect(box: OBB, c2w: np.ndarray) -> Tuple[float, float, float, float]:
    uv, _ = project(box.corners, c2w, K512)
    return float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())


# =============================================================================
# Phrasings
# =============================================================================
EMB_REL = {"left": "left of", "right": "right of", "above": "above", "below": "beneath"}
EMB_PAIR = {
    "orig": Phrasing("How are {a_bare} and {b_bare} positioned in relation to each other in the image?",
                     "embspatial", {k: "The {a_bare} is " + v + " the {b_bare}." for k, v in EMB_REL.items()}),
    "t1": Phrasing(["What is the spatial relationship between {a} and {b} in the image?",
                    "In the image, how is {a} placed relative to {b}?"], "t_paren",
                   {"left": "{A} is to the left of {b}", "right": "{A} is to the right of {b}",
                    "above": "{A} is above {b}", "below": "{A} is below {b}"}),
    "t2": Phrasing("In this picture, where does {a} appear relative to {b}?", "t_dot",
                   {"left": "to its left", "right": "to its right", "above": "higher up in the picture",
                    "below": "lower down in the picture"}),
    "t3": Phrasing("Judging only by their positions in the image, complete the sentence: {A} is ___ {b}.",
                   "t_colon", {"left": "left of", "right": "right of", "above": "above", "below": "below"}),
    "h1": Phrasing("Consider where the two objects sit inside the frame. Which description of {a} versus {b} is right?",
                   "h_inline", {"left": "{a} is on the left side of {b}", "right": "{a} is on the right side of {b}",
                                "above": "{a} is over {b} in the frame", "below": "{a} is under {b} in the frame"}),
}

EMB_DIST = {
    "orig": Phrasing(["From your perspective, which object in the image is at the {sup} distance?",
                      "Which object in the image is {sup_alt} to you?"], "embspatial", None),
    "t1": Phrasing("Which of these objects is {cmp_word} the camera?", "t_paren", None),
    "t2": Phrasing("Looking at the photo, pick the object that is {cmp_word} the viewer.", "t_dot", None),
    "t3": Phrasing("Among the listed objects, which one has the {depth_word} depth from the camera?", "t_colon", None),
    "h1": Phrasing("Of the objects named below, one stands {cmp_word} the photographer than all the others. Which?",
                   "h_inline", None),
}

VS_DIRS = ["front", "back", "left", "right", "front-left", "front-right", "back-left", "back-right", "above", "down",
           "front-up", "front-down", "back-up", "back-down", "above-left", "above-right", "down-left", "down-right"]
VS_CAM = {
    "orig": Phrasing(["Could you tell me the location of the {x} in comparison to the {y}?",
                      "How is the {x} positioned with respect to the {y}?",
                      "If you're looking at the {y}, where would you find the {x}?"], "viewspatial", None),
    "t1": Phrasing("From the camera's point of view, in which direction is {x_the} from {y_the}? (front means "
                   "farther from the camera, back means closer; up and down follow gravity)", "t_paren", None),
    "t2": Phrasing("Take the camera's viewing direction as front. Where is {x_the} relative to {y_the}?", "t_dot", None),
    "t3": Phrasing("Using the camera's frame (front = away from the camera), give the direction from {y_the} to {x_the}.",
                   "t_colon", None),
    "h1": Phrasing("Treat the lens direction as forward. Starting at {y_the} and heading to {x_the}, which way do you go?",
                   "h_inline", None),
}

BL_REL_ORIG = {"left": ["left of", "at the left side of", "on the left side of"],
               "right": ["right of", "at the right side of", "on the right side of"],
               "above": ["above", "over"], "below": ["below", "under", "beneath"],
               "on_top": ["on top of", "on"], "near": ["near", "next to", "beside", "close to"],
               "far": ["far away from", "far from", "away from"], "touching": ["touching"]}
BL_REL_PARA = {"left": "to the left of", "right": "to the right of", "above": "directly above",
               "below": "directly below", "on_top": "resting on top of", "near": "close to",
               "far": "far away from", "touching": "in contact with"}


def blink_yesno(rel_orig: str, rel_para: str) -> Dict[str, Phrasing]:
    return {
        "orig": Phrasing("Is the {a_bare} " + rel_orig + " the {b_bare}?", "blink", None),
        "t1": Phrasing("Answer yes or no: is {a} " + rel_para + " {b}?", "t_paren", None),
        "t2": Phrasing("True or false (answer yes for true): {A} is " + rel_para + " {b}.", "t_dot", None),
        "t3": Phrasing("Look at the image. Would you say {a} is " + rel_para + " {b}?", "t_colon", None),
        "h1": Phrasing("Someone claims that {a} is " + rel_para + " {b}. Do you agree?", "h_inline", None),
    }


TDSR = {
    "height": {"orig": Phrasing("Consider the real-world 3D locations of the objects. Which object has a {hl} location?", "3dsr"),
               "t1": Phrasing("In the real room (not the picture), which object sits {hl_er} above the floor?", "t_paren"),
               "t2": Phrasing("Which of the two objects is physically {hl_er} up in the room?", "t_dot"),
               "t3": Phrasing("Ignoring the camera, compare the heights of the objects in 3D: which is {hl_er}?", "t_colon"),
               "h1": Phrasing("Measured from the floor, one of these objects is located {hl_er}. Which one?", "h_inline")},
    "above": {"orig": Phrasing(["Consider the real-world 3D locations of the objects. Is the {a_bare} directly above the {b_bare}?"], "3dsr"),
              "t1": Phrasing("In 3D, is {a} directly over {b} (one above the other)?", "t_paren"),
              "t2": Phrasing("Does {a} sit vertically above {b} in the room?", "t_dot"),
              "t3": Phrasing("If you dropped {a} straight down, would it land on {b}?", "t_colon"),
              "h1": Phrasing("Think about the real positions. Is {a} located right on top of the space of {b}, higher up?", "h_inline")},
    "under": {"orig": Phrasing("Consider the real-world 3D locations of the objects. Is the {a_bare} directly underneath the {b_bare}?", "3dsr"),
              "t1": Phrasing("In 3D, is {a} directly under {b}?", "t_paren"),
              "t2": Phrasing("Does {a} sit vertically below {b} in the room?", "t_dot"),
              "t3": Phrasing("If you dropped {b} straight down, would it land on {a}?", "t_colon"),
              "h1": Phrasing("Think about the real positions. Is {a} located right beneath {b}?", "h_inline")},
    "closer_cam": {"orig": Phrasing("Consider the real-world 3D location of the objects. Which object is {cf} the camera?", "3dsr"),
                   "t1": Phrasing("Which object is {cf} the camera in real distance?", "t_paren"),
                   "t2": Phrasing("Which of the two is {cf} the viewer?", "t_dot"),
                   "t3": Phrasing("Compare how far the objects are from the camera. Which is {cf} it?", "t_colon"),
                   "h1": Phrasing("Picture yourself holding the camera. Which object is {cf} you?", "h_inline")},
    "next_to": {"orig": Phrasing("Consider the real-world 3D locations of the objects. Are the {a_bare} and the {b_bare} next to each other or far away from each other?", "3dsr",
                                 {"next": "next to each other", "far": "far away from each other"}),
                "t1": Phrasing("In the room, are {a} and {b} close together or far apart?", "t_paren", {"next": "close together", "far": "far apart"}),
                "t2": Phrasing("Do {a} and {b} stand side by side or at a distance?", "t_dot", {"next": "side by side", "far": "at a distance"}),
                "t3": Phrasing("Describe the 3D distance between {a} and {b}.", "t_colon", {"next": "they are adjacent", "far": "they are far from each other"}),
                "h1": Phrasing("Thinking of the actual room, how would you describe {a} and {b}?", "h_inline", {"next": "neighbours", "far": "well separated"})},
    "closer_to": {"orig": Phrasing("Consider the real-world 3D locations of the objects. Which is closer to the {c_bare}, the {a_bare} or the {b_bare}?", "3dsr"),
                  "t1": Phrasing("In 3D, which object is nearer to {c}?", "t_paren"),
                  "t2": Phrasing("Which of these lies closer to {c} in the room?", "t_dot"),
                  "t3": Phrasing("Measuring between the closest points, which object is the shortest distance from {c}?", "t_colon"),
                  "h1": Phrasing("Imagine walking from {c}. Which of the two would you reach first?", "h_inline")},
    "parallel": {"orig": Phrasing("Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the {a_bare} and the {b_bare}, parallel of perpendicular to each other?", "3dsr",
                                  {"parallel": "parallel", "perpendicular": "perpendicular"}),
                 "t1": Phrasing("Are the long sides of {a} and {b} parallel or perpendicular?", "t_paren", None),
                 "t2": Phrasing("In the room, do {a} and {b} run in the same direction or at a right angle?", "t_dot",
                                {"parallel": "same direction (parallel)", "perpendicular": "at a right angle"}),
                 "t3": Phrasing("Compare the orientation of {a} with that of {b}.", "t_colon",
                                {"parallel": "they are parallel", "perpendicular": "they are perpendicular"}),
                 "h1": Phrasing("Looking at how the two objects are laid out, how are their main axes related?", "h_inline",
                                {"parallel": "aligned with each other", "perpendicular": "crossing at 90 degrees"})},
}

BL_DEPTH = {
    "orig": Phrasing("Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?",
                     "blink", {"A": "A is closer", "B": "B is closer"}),
    "t1": Phrasing("The image marks two points with circles, A and B. Which one is nearer to the camera?", "t_paren",
                   {"A": "point A", "B": "point B"}),
    "t2": Phrasing("Of the circled points A and B, which lies at the smaller depth?", "t_dot", {"A": "A", "B": "B"}),
    "t3": Phrasing("Two circled points are labelled A and B. Which surface point is closer to the viewer?", "t_colon",
                   {"A": "the point labelled A", "B": "the point labelled B"}),
    "h1": Phrasing("Estimate the distance from the camera to each circled point (A, B). Which distance is shorter?",
                   "h_inline", {"A": "the distance to A", "B": "the distance to B"}),
}
BL_COUNT = {
    "orig": Phrasing("How many {plural} are there?", "blink", None),
    "t1": Phrasing("Count the {plural} you can see in the image.", "t_paren", None),
    "t2": Phrasing("What is the number of {plural} visible in this picture?", "t_dot", None),
    "t3": Phrasing("How many {plural} appear in the photo?", "t_colon", None),
    "h1": Phrasing("Tally every {singular} shown in the image. What total do you get?", "h_inline", None),
}
BL_LOC = {
    "orig": Phrasing("A bounding box is an annotated rectangle surrounding an object. The edges of bounding boxes should touch the outermost pixels of the object that is being labeled. Given the two bounding boxes on the image, labeled by A and B, which bounding box more accurately localizes and encloses the {x}?",
                     "blink_options", {"A": "Box A", "B": "Box B"}),
    "t1": Phrasing("Two boxes, A and B, are drawn on the image. Which one fits the {x} tightly?", "t_paren", {"A": "box A", "B": "box B"}),
    "t2": Phrasing("Which labelled rectangle (A or B) correctly encloses the {x}?", "t_dot", {"A": "A", "B": "B"}),
    "t3": Phrasing("The image shows two candidate boxes for the {x}. Which is the accurate one?", "t_colon", {"A": "the box labelled A", "B": "the box labelled B"}),
    "h1": Phrasing("Only one of the two drawn rectangles hugs the outline of the {x}. Which one?", "h_inline", {"A": "rectangle A", "B": "rectangle B"}),
}

YESNO = ["yes", "no"]


# =============================================================================
# GT definitions shown on the review page and stored with each question
# =============================================================================
GT = {
    "ES-lr": ("两个物体可见部分的二维框左右分离（间隔 ≥4 像素）且上下有重叠。", "EmbSpatial 用标注框投影的整框；这里用深度核验后的可见部分二维框。"),
    "ES-ud": ("两个物体可见部分的二维框上下分离（间隔 ≥4 像素）且左右有重叠。", "同上：可见部分二维框代替原版的标注框投影。"),
    "VS-01": ("相机坐标系下 Y 中心指向 X 中心的向量：水平方向按 ViewSpatial 的 8 个 45° 扇区（前=沿视线更远）；仰角 ≥60° 为纯上下，30°–60° 为上下+水平组合；离扇区边界 <7.5° 或离 30°/60° <5° 的不出题。",
              "ViewSpatial 未公开相机视角题的生成代码：水平扇区照搬其场景模拟题代码，'前=更远'按答案分布推断，上下分档为本项目自定。"),
    "BL-03-lr": ("左右按可见部分二维框判断（图像左右）。", "VSR 原题的左右可能指物体自身左右，这里统一为观察者（图像）左右。"),
    "BL-03-ud": ("按三维框：A 底面不低于 B 顶面 −5 cm，且 A 的地面投影至少 50% 落在 B 上方为'在上方'；'on top of' 另要求 A 底面与 B 顶面相距 ≤5 cm。反例：A 与 B 地面投影相隔 ≥30 cm 或高低相反。",
                 "VSR 原题由人标注；这里用三维框几何。"),
    "BL-03-dist": ("按两个三维框的最近距离：near ≤0.3 m、far ≥2.0 m；touching ≤0.02 m（反例 ≥0.5 m）；中间地带不出题。", "VSR 由人标注，无距离阈值。"),
    "3D-01": ("比较三维框中心高度：中心高差 ≥0.25 m，且顶面、底面的高低顺序与中心一致才出题。", "3DSRBench 为人工判断；本项目用框中心并要求三种量一致。"),
    "3D-02": ("正例：A 底面不低于 B 顶面 −5 cm 且 A 的地面投影至少 50% 在 B 上；反例：A 更高但两者地面投影相隔 ≥30 cm，或 A 在 B 下方。", "3DSRBench 为人工判断，无统一阈值。"),
    "3D-03": ("可见部分的平均深度（沿视线），差距 ≥ max(0.4 m, 15%)。", "3DSRBench 为人工判断的到相机距离；这里用可见部分平均深度。"),
    "3D-04": ("两个三维框的最近距离：≤0.3 m 为挨着，≥1.5 m 为远离；中间不出题。", "3DSRBench 为人工判断。"),
    "3D-08": ("两个候选物体到锚点物体的三维框最近距离，差距 ≥ max(0.5 m, 30%)。", "3DSRBench 为人工判断。"),
    "3D-11": ("只用水平长边 ≥ 短边 1.5 倍且 ≥0.5 m 的物体，比较长轴水平方向：≤10° 为平行，≥80° 为垂直。", "3DSRBench 为人工判断；这里只比较框的长轴。"),
    "ES-dist": ("可见部分的平均深度；最近（最远）者与第二名差距 ≥ max(0.4 m, 15%)。", "EmbSpatial 用框内平均深度（来自其 RGB-D）；这里用渲染深度上的可见部分，定义一致。"),
    "BL-02": ("两个点各取在不同物体的可见部分内部（离边缘 ≥8 像素、7×7 邻域深度标准差 ≤3 cm），按该像素渲染深度比较，差距 ≥ max(0.3 m, 15%)。", "BLINK 来自 DIW 人工标注；这里用渲染深度。"),
    "BL-01": ("同类（按显示名，含其下位类）的每个实例要么清楚可见（可见部分 ≥0.4% 画面且露出 ≥50%）、要么完全不在画面（<0.05%），有介于两者之间的实例就不出题。", "BLINK 计数题来自 TallyQA 人工计数。"),
    "BL-05": ("正确框 = 物体可见部分的二维框；干扰框由正确框加高斯扰动得到，与正确框的 IoU 在 0.25–0.6 之间且仍在画面内。", "BLINK 的正确框来自 LVIS 实例标注（完整物体框），干扰框同为高斯扰动。"),
}


def gt_record(key: str, **values) -> Dict[str, Any]:
    d, diff = GT[key]
    return {"definition": d, "differs_from_original": diff, "values": values}


# =============================================================================
# Iteration scaffolding
# =============================================================================
class Runner:
    """Shared loop: walk combos, realise until the quotas are met; then keep verifying up to ``verify_cap``
    combos to measure the pass rate."""

    def __init__(self, g: GenCtx, group: str, quota: Dict[str, int], verify_cap: int, census: bool = False) -> None:
        self.g = g
        self.group = group
        self.need = Counter(quota)
        self.verify_cap = verify_cap
        self.census = census
        self.tried = 0

    def want(self, subtasks: Sequence[str]) -> bool:
        return any(self.need[s] > 0 for s in subtasks)

    def emit_ok(self, subtask: str) -> bool:
        return self.need[subtask] > 0

    def done(self) -> bool:
        return (not self.census) and self.tried >= self.verify_cap and not any(v > 0 for v in self.need.values())

    def run(self, combos: Iterator[Combo], realize: Callable[[Combo, bool], Optional[str]]) -> None:
        for combo in combos:
            self.g.count(self.group, "combos_cpu")
            self.g.count(self.group, "combos_cpu:" + combo.kind)
            if self.census:
                continue
            if self.done():
                break
            if self.tried >= self.verify_cap and not self.want(combo.info.get("subtasks", [])):
                continue
            self.tried += 1
            self.g.count(self.group, "rendered_combos")
            reason = realize(combo, True)
            if reason is None:
                self.g.count(self.group, "verified_combos")
                self.g.count(self.group, "verified:" + combo.kind)
            else:
                self.g.reject(self.group, reason)

    def take(self, subtask: str) -> None:
        self.need[subtask] -= 1


def pick_subtask(r: Runner, options: Sequence[str]) -> Optional[str]:
    live = [s for s in options if r.need[s] > 0]
    if not live:
        return None
    return max(live, key=lambda s: r.need[s])


def entries(g: GenCtx, va, oids: Sequence[str], roles: Sequence[str]) -> Optional[Tuple[List[str], List[Dict]]]:
    names, objs = [], []
    for o, role in zip(oids, roles):
        n = g.name_single(va, o)
        if n is None:
            return None
        names.append(n[0])
        objs.append(g.obj_entry(o, n[0], role, n[1]))
    return names, objs


def fields_ab(na: str, nb: str, **kw) -> Dict[str, Any]:
    f = {"a": na, "b": nb, "A": cap(na), "B": cap(nb), "a_bare": bare(na), "b_bare": bare(nb)}
    f.update(kw)
    return f


def review_boxes(va, oids: Sequence[str], labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Projected 3D boxes (corners in 512 pixels) for the toggleable overlay on the review page."""
    out = []
    for o, lab in zip(oids, labels):
        uv, z = project(va.ctx.obb[o].corners, va.c2w, va.K)
        out.append({"oid": o, "label": lab, "corners": None if np.any(z <= 0.05) else uv.round(1).tolist(),
                    "visible_bbox": va.obj(o).bbox})
    return out


# =============================================================================
# S1: left / right
# =============================================================================
def s1_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = single_pool(ctx)
    for a, b in pairs_near(ctx, pool, 0.6, 4.0, rng=g.rng):
        if not g.distinct_names([a, b]):
            continue

        def lateral(c2w, a=a, b=b):
            ra, rb = proj_rect(ctx.obb[a], c2w), proj_rect(ctx.obb[b], c2w)
            sep = bbox_gap_x(ra, rb) >= 12 or bbox_gap_x(rb, ra) >= 12
            return sep and overlap_1d(ra[1], ra[3], rb[1], rb[3]) > 0

        cams = frame_candidates(ctx, [a, b], extra=lateral, max_out=2)
        if cams:
            yield Combo("S1", "pair_lateral", (a, b), [[c] for c in cams], {"subtasks": SUBTASKS["S1"]})


def vs_direction(ctx, c2w: np.ndarray, x: str, y: str) -> Optional[str]:
    """ViewSpatial camera-perspective label of x relative to y, or None near a boundary."""
    v = ctx.obb[x].center - ctx.obb[y].center
    R = np.asarray(c2w, float)[:3, :3]
    right = R[:, 0]
    fwd = np.array([R[0, 2], R[1, 2], 0.0])
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    rh = np.array([right[0], right[1], 0.0])
    rh /= max(np.linalg.norm(rh), 1e-9)
    lx, lz, up = float(v @ rh), float(v @ fwd), float(v[2])
    h = math.hypot(lx, lz)
    elev = math.degrees(math.atan2(up, h))
    ang = (math.degrees(math.atan2(lx, lz)) + 360.0) % 360.0
    bounds = [22.5 + 45 * k for k in range(8)]
    names = ["front-right", "right", "back-right", "back", "back-left", "left", "front-left"]

    def sector(a: float) -> Optional[str]:
        if min(angle_diff_deg(a, b) for b in bounds) < 7.5:
            return None
        for k in range(7):
            if bounds[k] <= a < bounds[k + 1]:
                return names[k]
        return "front"

    if abs(abs(elev) - 60) < 5 or abs(abs(elev) - 30) < 5:
        return None
    vert = "above" if elev > 0 else "down"
    if abs(elev) >= 60:
        return vert
    if abs(elev) >= 30:
        # compound vertical + the dominant horizontal axis, if clearly one of front/back/left/right
        if min(angle_diff_deg(ang, b) for b in (45, 135, 225, 315)) < 10:
            return None
        if angle_diff_deg(ang, 0) < 45:
            return "front-up" if vert == "above" else "front-down"
        if angle_diff_deg(ang, 180) < 45:
            return "back-up" if vert == "above" else "back-down"
        side = "right" if angle_diff_deg(ang, 90) < 45 else "left"
        return f"{vert}-{side}"
    return sector(ang)


def vs_distractors(answer: str, rng) -> List[str]:
    parts = answer.replace("up", "above").split("-") if "-" in answer else [answer]
    comp = {"front-up": ["front", "above"], "front-down": ["front", "down"], "back-up": ["back", "above"],
            "back-down": ["back", "down"]}.get(answer, parts)
    if "-" in answer:
        pool = [d for d in VS_DIRS if d != answer and d not in comp]
    else:
        pool = [d for d in VS_DIRS if d != answer and answer not in d.replace("up", "above").split("-")]
    rng.shuffle(pool)
    return pool[:3]


def s1_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        a, b = combo.objects
        for cams in combo.shots:
            vid, va = g.bank.view(cams[0], "S1")
            oa, ob = va.obj(a), va.obj(b)
            if not (oa.recognisable() and ob.recognisable()):
                continue
            ra, rb = oa.bbox, ob.bbox
            if bbox_gap_x(ra, rb) >= 4:
                left, right = a, b
            elif bbox_gap_x(rb, ra) >= 4:
                left, right = b, a
            else:
                continue
            ra, rb = va.obj(left).bbox, va.obj(right).bbox
            if overlap_1d(ra[1], ra[3], rb[1], rb[3]) <= 0:
                continue
            ent = entries(g, va, [left, right], ["left_object", "right_object"])
            if ent is None:
                return "name_not_unique_in_image"
            if not emit:
                return None
            vsdir = vs_direction(g.ctx, cams[0], left, right)
            opts = ["ES-01", "ES-02", "BL-03"] + (["VS-01"] if vsdir else [])
            st = pick_subtask(r, opts)
            if st is None:
                return None
            names, objs = ent
            img = g.bank.save(vid)
            rev = {"boxes": review_boxes(va, [left, right], ["L", "R"])}
            gap = bbox_gap_x(ra, rb)
            if st in ("ES-01", "ES-02"):
                # ES-01 asks the pair with A on the left (answer "left"); ES-02 with A on the right
                A, B = (left, right) if st == "ES-01" else (right, left)
                nA, nB = (names[0], names[1]) if st == "ES-01" else (names[1], names[0])
                ans = "left" if st == "ES-01" else "right"
                g.emit("S1", [st, "SITE-C02"], bench_of(st), "image", [vid], [img], EMB_PAIR, fields_ab(nA, nB),
                       ["left", "right", "above", "below"], ans, "mc",
                       gt_record("ES-lr", gap_px=gap, bbox_left=ra, bbox_right=rb),
                       [o for o in objs if o["id"] == A] + [o for o in objs if o["id"] == B], rev)
            elif st == "VS-01":
                x, y = (left, right) if g.rng.random() < 0.5 else (right, left)
                nx, ny = names[[left, right].index(x)], names[[left, right].index(y)]
                lab = vs_direction(g.ctx, cams[0], x, y)
                if lab is None:
                    return "vs01_boundary"
                keys = [lab] + vs_distractors(lab, g.rng)
                g.emit("S1", ["VS-01", "SITE-C02"], bench_of("VS-01"), "image", [vid], [img], VS_CAM,
                       {"x": bare(nx), "y": bare(ny), "x_the": nx, "y_the": ny}, keys, lab, "mc",
                       gt_record("VS-01", label=lab), objs, rev)
            else:
                ask = g.rng.choice(["left", "right"])
                truth = "yes" if ask == "left" else "no"   # asked about A=left object
                fam = blink_yesno(g.rng.choice(BL_REL_ORIG[ask]), BL_REL_PARA[ask])
                g.emit("S1", ["BL-03", "SITE-C02"], bench_of("BL-03"), "image", [vid], [img], fam,
                       fields_ab(names[0], names[1]), YESNO, truth, "yesno",
                       gt_record("BL-03-lr", asked=ask, gap_px=gap), objs, rev, keep_orig_order=YESNO)
            r.take(st)
            return None
        return "not_clear_after_render"

    return realize


# =============================================================================
# S2: above / below, higher / lower
# =============================================================================
def s2_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = single_pool(ctx)
    for a, b in pairs_near(ctx, pool, 0.0, 3.5, rng=g.rng):
        if not g.distinct_names([a, b]):
            continue
        A, B = ctx.obb[a], ctx.obb[b]
        if A.center[2] < B.center[2]:
            a, b, A, B = b, a, B, A          # a is the higher one
        kinds = []
        dz = float(A.center[2] - B.center[2])
        if dz >= 0.25 and A.hi[2] > B.hi[2] and A.lo[2] > B.lo[2]:
            kinds.append("higher")
        over = footprint_overlap_fraction(A, B, of="a")
        if A.lo[2] >= B.hi[2] - 0.05 and over >= 0.5:
            kinds.append("above_pos")
        elif A.lo[2] >= B.hi[2] + 0.1 and _xy_gap(A, B) >= 0.3:
            kinds.append("above_neg")

        def vertical(c2w, a=a, b=b):
            ra, rb = proj_rect(ctx.obb[a], c2w), proj_rect(ctx.obb[b], c2w)
            return bbox_gap_y(ra, rb) >= 12 and overlap_1d(ra[0], ra[2], rb[0], rb[2]) > 0

        if dz >= 0.2:
            kinds.append("image_vertical")
        if not kinds:
            continue
        extra = vertical if kinds == ["image_vertical"] else None
        cams = frame_candidates(ctx, [a, b], extra=extra, max_out=2)
        if cams:
            yield Combo("S2", "+".join(kinds), (a, b), [[c] for c in cams], {"subtasks": SUBTASKS["S2"], "kinds": kinds})


def _xy_gap(A: OBB, B: OBB) -> float:
    dx = max(0.0, float(A.lo[0] - B.hi[0]), float(B.lo[0] - A.hi[0]))
    dy = max(0.0, float(A.lo[1] - B.hi[1]), float(B.lo[1] - A.hi[1]))
    return math.hypot(dx, dy)


def s2_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        a, b = combo.objects       # a: higher in 3D
        kinds = combo.info["kinds"]
        for cams in combo.shots:
            vid, va = g.bank.view(cams[0], "S2")
            oa, ob = va.obj(a), va.obj(b)
            if not (oa.recognisable() and ob.recognisable()):
                continue
            ent = entries(g, va, [a, b], ["higher_object", "lower_object"])
            if ent is None:
                return "name_not_unique_in_image"
            names, objs = ent
            ra, rb = oa.bbox, ob.bbox
            img_rel = None
            if overlap_1d(ra[0], ra[2], rb[0], rb[2]) > 0:
                if bbox_gap_y(ra, rb) >= 4:
                    img_rel = "a_above"
                elif bbox_gap_y(rb, ra) >= 4:
                    img_rel = "b_above"
            opts = []
            if img_rel:
                opts += ["ES-03", "ES-04"]
            if "higher" in kinds:
                opts.append("3D-01")
            if "above_pos" in kinds or "above_neg" in kinds:
                opts += ["3D-02", "BL-03"]
            if not opts:
                continue
            if not emit:
                return None
            st = pick_subtask(r, opts)
            if st is None:
                return None
            img = g.bank.save(vid)
            rev = {"boxes": review_boxes(va, [a, b], ["hi", "lo"])}
            A, B = ctx.obb[a], ctx.obb[b]
            vals = {"center_z": [round(float(A.center[2]), 3), round(float(B.center[2]), 3)],
                    "bottom_top": [round(float(A.lo[2]), 3), round(float(B.hi[2]), 3)],
                    "footprint_overlap_of_upper": round(footprint_overlap_fraction(A, B, of="a"), 3)}
            if st in ("ES-03", "ES-04"):
                up, dn = (a, b) if img_rel == "a_above" else (b, a)
                nu, nd = names[[a, b].index(up)], names[[a, b].index(dn)]
                X, Y, nX, nY, ans = (up, dn, nu, nd, "above") if st == "ES-03" else (dn, up, nd, nu, "below")
                g.emit("S2", [st, "SITE-C02"], bench_of(st), "image", [vid], [img], EMB_PAIR, fields_ab(nX, nY),
                       ["left", "right", "above", "below"], ans, "mc",
                       gt_record("ES-ud", bbox_upper=va.obj(up).bbox, bbox_lower=va.obj(dn).bbox),
                       [o for o in objs if o["id"] == X] + [o for o in objs if o["id"] == Y], rev)
            elif st == "3D-01":
                hl = g.rng.choice(["higher", "lower"])
                ans = names[0] if hl == "higher" else names[1]
                g.emit("S2", [st, "SITE-C04"], bench_of(st), "image", [vid], [img], TDSR["height"],
                       {"hl": hl, "hl_er": hl}, [bare(names[0]), bare(names[1])], bare(ans), "mc", gt_record("3D-01", **vals), objs, rev)
            else:
                pos = "above_pos" in kinds
                form = g.rng.choice(["above", "under"])
                # "Is X directly above Y?": X = a (upper) gives the truth for positives, swapped gives "no"
                if form == "above":
                    X, Y = (a, b) if (pos or g.rng.random() < 0.5) else (b, a)
                    truth = "yes" if (pos and X == a) else "no"
                else:
                    X, Y = (b, a) if (pos or g.rng.random() < 0.5) else (a, b)
                    truth = "yes" if (pos and X == b) else "no"
                nX, nY = names[[a, b].index(X)], names[[a, b].index(Y)]
                if st == "3D-02":
                    g.emit("S2", [st, "SITE-C02"], bench_of(st), "image", [vid], [img], TDSR[form],
                           fields_ab(nX, nY), YESNO, truth, "yesno", gt_record("3D-02", **vals), objs, rev)
                else:
                    on_top = pos and form == "above" and float(A.lo[2] - B.hi[2]) <= 0.05 and g.rng.random() < 0.5
                    rel = "on_top" if on_top else ("above" if form == "above" else "below")
                    fam = blink_yesno(g.rng.choice(BL_REL_ORIG[rel]), BL_REL_PARA[rel])
                    g.emit("S2", ["BL-03", "SITE-C02"], bench_of("BL-03"), "image", [vid], [img], fam,
                           fields_ab(nX, nY), YESNO, truth, "yesno", gt_record("BL-03-ud", relation=rel, **vals),
                           objs, rev, keep_orig_order=YESNO)
            r.take(st)
            return None
        return "not_clear_after_render"

    return realize


# =============================================================================
# S3: distance from the camera
# =============================================================================
def s3_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = single_pool(ctx)
    pairs = pairs_near(ctx, pool, 0.8, 5.0, rng=g.rng)
    for a, b in pairs:
        if not g.distinct_names([a, b]):
            continue

        def depth_gap(c2w, a=a, b=b):
            za, zb = to_camera(np.stack([ctx.obb[a].center, ctx.obb[b].center]), c2w)[:, 2]
            return abs(float(za - zb)) >= 0.8

        cams = frame_candidates(ctx, [a, b], extra=depth_gap, max_out=2)
        if cams:
            yield Combo("S3", "pair_depth", (a, b), [[c] for c in cams], {"subtasks": ["3D-03", "BL-02"]})
    # triples / quadruples for EmbSpatial closest / farthest
    room_objs: Dict[Any, List[str]] = {}
    for o in pool:
        room_objs.setdefault(ctx.room_of_obj.get(o), []).append(o)
    for room, objs in room_objs.items():
        if len(objs) < 3:
            continue
        trip = list(itertools.combinations(sorted(objs), 3))
        g.rng.shuffle(trip)
        for t in trip[:g.cap(40)]:
            if not g.distinct_names(t):
                continue
            cs = [ctx.obb[o].center for o in t]
            if min(np.linalg.norm(cs[i][:2] - cs[j][:2]) for i, j in itertools.combinations(range(3), 2)) < 0.8:
                continue

            def spread(c2w, t=t):
                z = np.sort(to_camera(np.stack([ctx.obb[o].center for o in t]), c2w)[:, 2])
                return (z[1] - z[0] >= 0.6) or (z[2] - z[1] >= 0.6)

            cams = frame_candidates(ctx, list(t), extra=spread, max_out=2)
            if cams:
                yield Combo("S3", "set_depth", tuple(t), [[c] for c in cams], {"subtasks": ["ES-05", "ES-06"]})


def s3_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        oids = combo.objects
        for cams in combo.shots:
            vid, va = g.bank.view(cams[0], "S3")
            ovs = [va.obj(o) for o in oids]
            if not all(ov.recognisable() for ov in ovs):
                continue
            ent = entries(g, va, oids, ["object"] * len(oids))
            if ent is None:
                return "name_not_unique_in_image"
            names, objs = ent
            d = [ov.mean_depth for ov in ovs]
            order = list(np.argsort(d))
            img_needed = True
            if combo.kind == "set_depth":
                near_ok = d[order[1]] - d[order[0]] >= max(0.4, 0.15 * d[order[0]])
                far_ok = d[order[-1]] - d[order[-2]] >= max(0.4, 0.15 * d[order[-2]])
                opts = (["ES-05"] if near_ok else []) + (["ES-06"] if far_ok else [])
                if not opts:
                    continue
                if not emit:
                    return None
                st = pick_subtask(r, opts)
                if st is None:
                    return None
                img = g.bank.save(vid)
                ans_i = order[0] if st == "ES-05" else order[-1]
                fields = {"sup": "shortest" if st == "ES-05" else "farthest",
                          "sup_alt": "closest" if st == "ES-05" else "farthest",
                          "cmp_word": "closest to" if st == "ES-05" else "farthest from",
                          "depth_word": "smallest" if st == "ES-05" else "largest"}
                keys = [bare(n) for n in names]
                g.emit("S3", [st, "SITE-C04"], bench_of(st), "image", [vid], [img], EMB_DIST, fields, keys,
                       keys[ans_i], "mc", gt_record("ES-dist", mean_depths=[round(x, 3) for x in d]), objs,
                       {"boxes": review_boxes(va, oids, [bare(n) for n in names])})
                r.take(st)
                return None
            # pairs: 3D-03 or BL-02
            if d[order[1]] - d[order[0]] < max(0.4, 0.15 * d[order[0]]):
                continue
            if not emit:
                return None
            st = pick_subtask(r, ["3D-03", "BL-02"])
            if st is None:
                return None
            if st == "3D-03":
                img = g.bank.save(vid)
                cf = g.rng.choice(["closer to", "further away from"])
                ans = names[order[0]] if cf == "closer to" else names[order[1]]
                g.emit("S3", [st, "SITE-C04"], bench_of(st), "image", [vid], [img], TDSR["closer_cam"], {"cf": cf},
                       [bare(names[0]), bare(names[1])], bare(ans), "mc", gt_record("3D-03", mean_depths=[round(x, 3) for x in d]), objs,
                       {"boxes": review_boxes(va, oids, ["1", "2"])})
            else:
                pts = []
                for o in oids:
                    p = flat_surface_points(va.mask(o), va.depth, erode_px=8, rng=g.rng, n=1)
                    if not p:
                        return "no_flat_point"
                    pts.append(p[0])
                dp = [float(va.depth[y, x]) for x, y in pts]
                if abs(dp[0] - dp[1]) < max(0.3, 0.15 * min(dp)):
                    return "point_depth_margin"
                labs = ["A", "B"] if g.rng.random() < 0.5 else ["B", "A"]
                tag = f"pts{labs[0]}{labs[1]}"
                img = g.bank.save_marked(vid, tag, lambda im: draw_points(im, pts, labs))
                closer = labs[int(np.argmin(dp))]
                g.emit("S3", [st, "SITE-C04"], bench_of(st), "image", [vid], [img], BL_DEPTH, {}, ["A", "B"], closer,
                       "mc", gt_record("BL-02", points={labs[0]: pts[0], labs[1]: pts[1]}, depths=dict(zip(labs, dp))),
                       objs, {"points": [{"label": l, "xy": p} for l, p in zip(labs, pts)]},
                       keep_orig_order=["A", "B"])
            r.take(st)
            return None
        return "not_clear_after_render"

    return realize


# =============================================================================
# S4: distances between objects
# =============================================================================
def s4_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = single_pool(ctx)
    for a, b in pairs_near(ctx, pool, 0.0, 5.0, rng=g.rng):
        if not g.distinct_names([a, b]):
            continue
        d = distance(ctx, a, b)
        kinds = []
        if d <= 0.3:
            kinds.append("near")
        if d <= 0.02:
            kinds.append("touch")
        if d >= 1.5:
            kinds.append("far")
        if d >= 0.5:
            kinds.append("not_touch")
        if not kinds:
            continue
        cams = frame_candidates(ctx, [a, b], max_out=2)
        if cams:
            yield Combo("S4", "pair_" + kinds[0], (a, b), [[c] for c in cams],
                        {"subtasks": ["3D-04", "BL-03"], "kinds": kinds, "distance": d})
    room_objs: Dict[Any, List[str]] = {}
    for o in pool:
        room_objs.setdefault(ctx.room_of_obj.get(o), []).append(o)
    for room, objs in room_objs.items():
        trip = list(itertools.permutations(sorted(objs), 3))
        g.rng.shuffle(trip)
        n = 0
        for c, a, b in trip:
            if n >= g.cap(25) or a > b:
                continue
            if not g.distinct_names([a, b, c]):
                continue
            da, db = distance(ctx, a, c), distance(ctx, b, c)
            if abs(da - db) < max(0.5, 0.3 * min(da, db)):
                continue
            cams = frame_candidates(ctx, [a, b, c], max_out=2)
            if cams:
                n += 1
                yield Combo("S4", "triple_closer", (c, a, b), [[x] for x in cams],
                            {"subtasks": ["3D-08"], "d": [da, db]})


def s4_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        oids = combo.objects
        for cams in combo.shots:
            vid, va = g.bank.view(cams[0], "S4")
            if not all(va.obj(o).recognisable() for o in oids):
                continue
            roles = ["anchor", "candidate", "candidate"] if combo.kind == "triple_closer" else ["object", "object"]
            ent = entries(g, va, oids, roles)
            if ent is None:
                return "name_not_unique_in_image"
            if not emit:
                return None
            names, objs = ent
            img_needed = None
            if combo.kind == "triple_closer":
                if not r.emit_ok("3D-08"):
                    return None
                img = g.bank.save(vid)
                da, db = combo.info["d"]
                ans = names[1] if da < db else names[2]
                g.emit("S4", ["3D-08", "SITE-C02"], bench_of("3D-08"), "image", [vid], [img], TDSR["closer_to"],
                       {"c": names[0], "c_bare": bare(names[0]), "a_bare": bare(names[1]), "b_bare": bare(names[2])},
                       [bare(names[1]), bare(names[2])], bare(ans), "mc",
                       gt_record("3D-08", distances_m={bare(names[1]): round(da, 3), bare(names[2]): round(db, 3)}),
                       objs, {"boxes": review_boxes(va, oids, ["C", "1", "2"])})
                r.take("3D-08")
                return None
            kinds = combo.info["kinds"]
            dist = combo.info["distance"]
            opts = []
            if "near" in kinds or "far" in kinds:
                opts += ["3D-04", "BL-03"]
            elif "touch" in kinds or "not_touch" in kinds:
                opts += ["BL-03"]
            st = pick_subtask(r, opts)
            if st is None:
                return None
            img = g.bank.save(vid)
            rev = {"boxes": review_boxes(va, oids, ["1", "2"])}
            if st == "3D-04":
                ans = "next" if "near" in kinds else "far"
                g.emit("S4", [st, "SITE-C02"], bench_of(st), "image", [vid], [img], TDSR["next_to"],
                       fields_ab(names[0], names[1]), ["next", "far"], ans, "mc",
                       gt_record("3D-04", box_distance_m=round(dist, 3)), objs, rev)
            else:
                cand = []
                if "near" in kinds:
                    cand += [("near", "yes"), ("far", "no")]
                if "far" in kinds and dist >= 2.0:
                    cand += [("far", "yes"), ("near", "no")]
                if "touch" in kinds:
                    cand += [("touching", "yes")]
                if "not_touch" in kinds:
                    cand += [("touching", "no")]
                if not cand:
                    return "bl03_distance_band"
                rel, truth = g.rng.choice(cand)
                fam = blink_yesno(g.rng.choice(BL_REL_ORIG[rel]), BL_REL_PARA[rel])
                g.emit("S4", ["BL-03", "SITE-C02"], bench_of("BL-03"), "image", [vid], [img], fam,
                       fields_ab(names[0], names[1]), YESNO, truth, "yesno",
                       gt_record("BL-03-dist", relation=rel, box_distance_m=round(dist, 3)), objs, rev,
                       keep_orig_order=YESNO)
            r.take(st)
            return None
        return "not_clear_after_render"

    return realize


# =============================================================================
# S5: counting (BLINK)
# =============================================================================
def plural(word: str) -> str:
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def s5_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    groups: Dict[str, List[str]] = {}
    for o in sorted(ctx.eligible):
        if o in ctx.duplicates:
            continue
        groups.setdefault(ctx.disp[o], []).append(o)
    items = list(groups.items())
    g.rng.shuffle(items)
    for disp, members in items:
        # every object a reader would count as "<disp>" (incl. sub-categories and non-eligible small ones)
        everyone = sorted({members[0]} | set(ctx.same_name(members[0])))
        everyone = [o for o in everyone if o in ctx.obb and o not in ctx.duplicates]
        if len(everyone) < 2 or len(everyone) > 8:
            continue
        rooms = Counter(ctx.room_of_obj.get(o) for o in everyone)
        room, n = rooms.most_common(1)[0]
        cluster = [o for o in everyone if ctx.room_of_obj.get(o) == room]
        if len(cluster) < 2:
            continue
        cams = frame_candidates(ctx, cluster, need_visible=cluster, max_out=3)
        if cams:
            yield Combo("S5", "category_cluster", tuple(cluster), [[c] for c in cams],
                        {"subtasks": ["BL-01"], "category": disp, "everyone": everyone})


def s5_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        everyone = combo.info["everyone"]
        disp = combo.info["category"]
        for cams in combo.shots:
            vid, va = g.bank.view(cams[0], "S5")
            state = []
            for o in everyone:
                ov = va.obj(o)
                if ov.recognisable(0.6):
                    state.append("in")
                elif ov.absent():
                    state.append("out")
                else:
                    state.append("partial")
            if "partial" in state:
                continue
            n = state.count("in")
            if n < 1:
                continue
            if not emit:
                return None
            if not r.emit_ok("BL-01"):
                return None
            img = g.bank.save(vid)
            word = disp.split()[-1] if " " in disp and disp.split()[-1] in ("table", "chair", "cabinet") else disp
            choices = sorted({n} | set(g.rng.sample([x for x in range(0, 8) if x != n], 3)))
            objs = [g.obj_entry(o, disp, "counted" if s == "in" else "outside_image") for o, s in zip(everyone, state)]
            g.emit("S5", ["BL-01", "SITE-C01"], "BLINK", "image", [vid], [img], BL_COUNT,
                   {"plural": plural(disp), "singular": disp}, [str(c) for c in choices], str(n), "mc",
                   gt_record("BL-01", count=n, instances=len(everyone)), objs,
                   {"boxes": review_boxes(va, [o for o, s in zip(everyone, state) if s == "in"], ["#"] * n)})
            r.take("BL-01")
            return None
        return "partially_visible_instance"

    return realize


# =============================================================================
# S6: object localisation (BLINK)
# =============================================================================
def s6_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = single_pool(ctx)
    g.rng.shuffle(pool)
    for o in pool:
        cams = frame_candidates(ctx, [o], max_out=2, radii=(1.5, 2.0, 2.6, 3.2, 3.9))
        if cams:
            yield Combo("S6", "single", (o,), [[c] for c in cams], {"subtasks": ["BL-05"]})


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def perturb_box(box, size: int, rng) -> Optional[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = [float(v) for v in box]
    w, h = x1 - x0, y1 - y0
    for _ in range(200):
        n = [rng.gauss(0, 0.3) for _ in range(4)]
        c = (x0 + w * n[0], y0 + h * n[1], x1 + w * n[2], y1 + h * n[3])
        c = (min(c[0], c[2]), min(c[1], c[3]), max(c[0], c[2]), max(c[1], c[3]))
        if c[2] - c[0] < 8 or c[3] - c[1] < 8 or c[0] < 0 or c[1] < 0 or c[2] >= size or c[3] >= size:
            continue
        if 0.25 <= iou(c, (x0, y0, x1, y1)) <= 0.6:
            return c
    return None


def s6_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        (o,) = combo.objects
        for cams in combo.shots:
            vid, va = g.bank.view(cams[0], "S6")
            ov = va.obj(o)
            if not ov.recognisable(0.9):
                continue
            n = g.name_single(va, o)
            if n is None:
                return "name_not_unique_in_image"
            if not emit:
                return None
            if not r.emit_ok("BL-05"):
                return None
            true = tuple(float(v) for v in ov.bbox)
            fake = perturb_box(true, va.size, g.rng)
            if fake is None:
                return "no_perturbed_box"
            labels = ["A", "B"]
            correct = "A" if g.rng.random() < 0.5 else "B"
            boxes = [true, fake] if correct == "A" else [fake, true]
            img = g.bank.save_marked(vid, f"loc{correct}", lambda im: draw_boxes(im, boxes, labels,
                                                                               [(230, 30, 40), (30, 110, 230)]))
            g.emit("S6", ["BL-05", "SITE-C03"], "BLINK", "image", [vid], [img], BL_LOC, {"x": bare(n[0])},
                   ["A", "B"], correct, "mc", gt_record("BL-05", true_box=true, distractor=fake, iou=round(iou(true, fake), 3)),
                   [g.obj_entry(o, n[0], "target", n[1])], {"drawn_boxes": {"A": boxes[0], "B": boxes[1]}},
                   keep_orig_order=["A", "B"])
            r.take("BL-05")
            return None
        return "not_clear_after_render"

    return realize


# =============================================================================
# S7: parallel (3DSRBench)
# =============================================================================
def s7_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = [o for o in single_pool(ctx, 3.5) if ctx.obb[o].is_upright()]
    long = []
    for o in pool:
        yaw, ratio = ctx.obb[o].long_axis_yaw()
        if ratio >= 1.5 and ctx.obb[o].longest_edge >= 0.5:
            long.append(o)
    for a, b in pairs_near(ctx, long, 0.5, 5.0, rng=g.rng):
        if not g.distinct_names([a, b]):
            continue
        ya, _ = ctx.obb[a].long_axis_yaw()
        yb, _ = ctx.obb[b].long_axis_yaw()
        diff = angle_diff_deg(math.degrees(ya), math.degrees(yb), 180.0)
        if diff <= 10:
            kind = "parallel"
        elif diff >= 80:
            kind = "perpendicular"
        else:
            continue
        cams = frame_candidates(ctx, [a, b], max_out=2)
        if cams:
            yield Combo("S7", kind, (a, b), [[c] for c in cams], {"subtasks": ["3D-11"], "angle_deg": diff})


def s7_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        a, b = combo.objects
        for cams in combo.shots:
            vid, va = g.bank.view(cams[0], "S7")
            if not (va.obj(a).recognisable() and va.obj(b).recognisable()):
                continue
            ent = entries(g, va, [a, b], ["object", "object"])
            if ent is None:
                return "name_not_unique_in_image"
            if not emit:
                return None
            if not r.emit_ok("3D-11"):
                return None
            names, objs = ent
            img = g.bank.save(vid)
            g.emit("S7", ["3D-11", "SITE-C02"], "3DSRBench", "image", [vid], [img], TDSR["parallel"],
                   fields_ab(names[0], names[1]), ["parallel", "perpendicular"], combo.kind, "mc",
                   gt_record("3D-11", long_axis_angle_deg=round(combo.info["angle_deg"], 1)), objs,
                   {"boxes": review_boxes(va, [a, b], ["1", "2"])})
            r.take("3D-11")
            return None
        return "not_clear_after_render"

    return realize


GROUPS = {
    "S1": (s1_combos, s1_realize), "S2": (s2_combos, s2_realize), "S3": (s3_combos, s3_realize),
    "S4": (s4_combos, s4_realize), "S5": (s5_combos, s5_realize), "S6": (s6_combos, s6_realize),
    "S7": (s7_combos, s7_realize),
}
