"""Multi-image groups M1-M11 (user prompt, section 6): choose the objects, then place the cameras.

Names are unique in the whole scene (v10_r5).  Two-image questions are built on a *view pair*: camera 1
frames two landmarks, camera 2 follows from camera 1 by a known translation and turn (M1) or orbit (BL-04),
and both must show at least two recognisable shared objects so the images can be related.  A key object of
M2/M3 shows in only one image, so one image alone cannot answer.  MindCube settings (M7 rotation, M8
around, M9 among, M10 translation) and the panning chains of M6/M11 have their own constructions.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import mv
from .common import Combo, GenCtx, bare, cap, scene_pool, simple_named_pool, single_pool
from .draw import draw_points, flat_surface_points
from .geom import (CAMERA_HEIGHT_M, OBB, angle_diff_deg, camera_K, convex_hull, look_at, offset_convex_polygon,
                   point_in_polygon, polygon_area, pose_yaw_pitch, project, yaw_pitch_of)
from .qa import Phrasing
from .single import Runner, pick_subtask, review_boxes
from .views import box_outside_view, frame_candidates, framed, framed_loose

SIZE = 512
K = camera_K(SIZE)
HALF_FOV = 45.0

SUBTASKS = {
    "M1": ["MMSI-position-camera-camera", "MMSI-motion-camera", "BL-04"],
    "M2": ["MMSI-position-camera-object", "MMSI-position-object-object", "MMSI-attribute-measurement"],
    "M3": ["MMSI-position-camera-region", "MMSI-position-object-region", "MMSI-position-region-region"],
    "M4": ["BL-06"],
    "M5": ["MMSI-attribute-appearance"],
    "M6": ["MMSI-multistep"],
    "M7": ["MC-rotation-current", "MC-rotation-mental-turn"],
    "M8": ["MC-around-q1", "MC-around-q2", "MC-around-q3"],
    "M9": ["MC-among-q1", "MC-among-q2-current", "MC-among-q2-sequence", "MC-among-q5"],
    "M10": ["MC-translation-transitive"],
    "M11": ["VS-05"],
}
TAGS_ROT = ["MC-setting-rotation", "MC-dynamics-rotation", "MC-perspective-self", "MC-relation-agent-object"]
TAGS_AROUND = ["MC-setting-around", "MC-pattern-linear"]
TAGS_AMONG = ["MC-setting-among", "MC-pattern-perpendicular"]


def gt(definition: str, diff: str, **values) -> Dict[str, Any]:
    return {"definition": definition, "differs_from_original": diff, "values": values}


def landmarks(ctx) -> List[str]:
    return [o for o in single_pool(ctx, 3.5) if ctx.obb[o].longest_edge >= 0.4]


def shared_rec(vas, oids: Sequence[str], min_in_frame: float = 0.6) -> List[str]:
    return [o for o in oids if all(mv.vis_class(va, o, min_in_frame) == "rec" for va in vas)]


def in_frustum(vas_or_va, oids: Sequence[str]) -> List[str]:
    va = vas_or_va
    return va.candidates_in_frustum(oids)


def three_f(vas) -> List[str]:
    """Landmarks in the frustum of every view (candidates for 'shared')."""
    s = None
    for va in vas:
        cur = set(va.candidates_in_frustum(landmarks(va.ctx)))
        s = cur if s is None else s & cur
    return sorted(s or [])


# =============================================================================
# Phrasings
# =============================================================================
MMSI_CAMPOS = {
    "orig": Phrasing("Assuming I am taking the first photo, where is the camera positioned relative to me when taking the second photo?", "mmsi"),
    "t1": Phrasing("Image 1 and image 2 were taken in the same room. Seen from where image 1 was taken (facing its view direction), where was the camera of image 2?", "t_paren"),
    "t2": Phrasing("Relative to the first camera's position and heading, in which direction does the second camera stand?", "t_dot"),
    "t3": Phrasing("I took photo 1, then walked to take photo 2. From my original position and facing, where is the second spot?", "t_colon"),
    "h1": Phrasing("Two shots of one room. Using the first shot's viewpoint as reference, locate the spot of the second shot.", "h_inline"),
}
MMSI_XYZ_MOVE = {
    "orig": Phrasing("The camera coordinate system is defined as +Y up, -Z forward, and is a right-handed system. From the perspective of the camera coordinate system, how is the viewpoint of the second image obtained by moving the viewpoint of the first image?", "mmsi"),
    "t1": Phrasing("Use the first camera's axes (+X right, +Y up, -Z forward). Which translation takes the first viewpoint to the second?", "t_paren"),
    "t2": Phrasing("In camera coordinates of image 1 (+Y up, -Z forward, right-handed), describe the move to the viewpoint of image 2.", "t_dot"),
    "t3": Phrasing("With X to the right and -Z pointing forward for camera 1, along which axes did the camera move to take image 2?", "t_colon"),
    "h1": Phrasing("Express the second viewpoint's offset in the first camera's right-handed frame (X right, Y up, Z backward).", "h_inline"),
}
MMSI_XYZ_ROT = {k: Phrasing(p.question.replace("moving the viewpoint", "rotating the viewpoint").replace(
    "Which translation takes", "Which rotation takes").replace("describe the move", "describe the rotation").replace(
    "along which axes did the camera move", "about which axis and in which sense did the camera rotate").replace(
    "Express the second viewpoint's offset", "Express the second viewpoint's rotation"), p.style)
    for k, p in MMSI_XYZ_MOVE.items()}
MMSI_MOTION = {
    "orig": Phrasing("The images are taken continuously from a first-person perspective. In which direction are you moving?", "mmsi"),
    "t1": Phrasing("These two frames come from one walk through the room. How did I move between them?", "t_paren"),
    "t2": Phrasing("From the first frame to the second, which way did the person holding the camera move?", "t_dot"),
    "t3": Phrasing("Consecutive first-person photos: describe my movement.", "t_colon"),
    "h1": Phrasing("Judging from how the scene shifts between the two first-person shots, my motion was:", "h_inline"),
}
MMSI_ROTATE = {
    "orig": Phrasing("The images are taken continuously from a first-person perspective. In which direction is the camera rotating?", "mmsi"),
    "t1": Phrasing("Between these two consecutive frames, which way did the camera turn?", "t_paren"),
    "t2": Phrasing("From image 1 to image 2 the camera rotated in place. In which direction?", "t_dot"),
    "t3": Phrasing("The photographer only turned the camera between the shots. Which way?", "t_colon"),
    "h1": Phrasing("Without walking, I turned my head between the two shots. Toward which side did I look?", "h_inline"),
}
BLINK_MV = {
    "orig": Phrasing("The images are frames from a video. The video is shooting a static scene. The camera is either moving clockwise (left) or counter-clockwise (right) around the object. The first image is from the beginning of the video and the second image is from the end. Is the camera moving left or right when shooting the video?",
                     "blink_options"),
    "t1": Phrasing("Two frames from a video circling an object (first frame, then last). Did the camera go around to the left (clockwise) or to the right (counter-clockwise)?", "t_paren"),
    "t2": Phrasing("A camera orbits the main object between these frames. Which way did it move?", "t_dot"),
    "t3": Phrasing("The camera walked around the object from frame 1 to frame 2. Left (clockwise) or right (counter-clockwise)?", "t_colon"),
    "h1": Phrasing("Watching the object from the first to the second frame, the photographer circled it. In which sense?", "h_inline"),
}
MMSI_CAMOBJ = {
    "orig": Phrasing(["When you took the second photo, where was the {x} in relation to you?",
                      "When I took the second photo, where was the {x} in relation to me?"], "mmsi"),
    "t1": Phrasing("At the moment the second picture was taken, in which direction was {x_the} from the camera?", "t_paren"),
    "t2": Phrasing("Standing where photo 2 was taken and facing its direction, where is {x_the}?", "t_dot"),
    "t3": Phrasing("{X} is not in the second image. Relative to the second camera, where is it?", "t_colon"),
    "h1": Phrasing("Picture yourself as the second camera. Which way would you have to look to see {x_the}?", "h_inline"),
}
MMSI_OBJOBJ = {
    "orig": Phrasing("The {p} is to the {pdir} of the {q}. In which direction is the {r} from the {q}?", "mmsi"),
    "t1": Phrasing("Define the compass so that {p_the} lies {pdir} of {q_the}. What is the compass direction from {q_the} to {r_the}?", "t_paren"),
    "t2": Phrasing("If {p_the} is {pdir} of {q_the}, then {r_the} is in which direction from {q_the}?", "t_dot"),
    "t3": Phrasing("Taking '{p_the} is {pdir} of {q_the}' as the reference, locate {r_the} relative to {q_the}.", "t_colon"),
    "h1": Phrasing("Suppose a map is drawn where {p_the} sits {pdir} of {q_the}. On that map, where does {r_the} lie with respect to {q_the}?", "h_inline"),
}
MMSI_MEAS = {
    "orig": Phrasing("Which is {adj}, the {x} or the {y}?", "mmsi"),
    "t1": Phrasing("Comparing {x_the} (image 1) and {y_the} (image 2), which one is {adj}?", "t_paren"),
    "t2": Phrasing("In real size, is {x_the} or {y_the} {adj}?", "t_dot"),
    "t3": Phrasing("Using both photos, decide which object is {adj}: {x_the} or {y_the}.", "t_colon"),
    "h1": Phrasing("Measure in your head: between {x_the} and {y_the}, which has the bigger {dim}?", "h_inline"),
}
MMSI_CAMREG = {
    "orig": Phrasing("When you took the second picture, where was the {reg} in relation to you?", "mmsi"),
    "t1": Phrasing("At the moment of the second picture, in which direction was {reg_the} from the camera?", "t_paren"),
    "t2": Phrasing("Standing where the second photo was taken, which way is {reg_the}?", "t_dot"),
    "t3": Phrasing("{Reg} is not in view in photo 2. Relative to that camera, where is it?", "t_colon"),
    "h1": Phrasing("Imagine being the second camera. Toward which side lies {reg_the}?", "h_inline"),
}
MMSI_OBJREG = {
    "orig": Phrasing("In relation to the {q}, the {p} is to the {pdir}. In which direction is the {reg} from the {x}?", "mmsi"),
    "t1": Phrasing("Let {p_the} be {pdir} of {q_the}. Then in which compass direction is {reg_the} from {x_the}?", "t_paren"),
    "t2": Phrasing("With {p_the} {pdir} of {q_the}, where does {reg_the} lie as seen from {x_the}?", "t_dot"),
    "t3": Phrasing("Reference: {p_the} is {pdir} of {q_the}. Give the direction from {x_the} to {reg_the}.", "t_colon"),
    "h1": Phrasing("On a map where {p_the} is {pdir} of {q_the}, in which direction must you walk from {x_the} to reach {reg_the}?", "h_inline"),
}
MMSI_REGREG = {
    "orig": Phrasing("Assuming the {p} is to the {pdir} of the {reg_a}, in which direction is the {reg_b} from the {reg_a}?", "mmsi"),
    "t1": Phrasing("Let {p_the} be {pdir} of {reg_a_the}. In which compass direction is {reg_b_the} from {reg_a_the}?", "t_paren"),
    "t2": Phrasing("If {p_the} lies {pdir} of {reg_a_the}, where is {reg_b_the} relative to {reg_a_the}?", "t_dot"),
    "t3": Phrasing("Reference: {p_the} is {pdir} of {reg_a_the}. Locate {reg_b_the} from {reg_a_the}.", "t_colon"),
    "h1": Phrasing("Draw a floor plan with {p_the} {pdir} of {reg_a_the}. Where on it is {reg_b_the}, seen from {reg_a_the}?", "h_inline"),
}
BL_CORR = {
    "orig": Phrasing("A point is circled on the first image, labeled with REF. We change the camera position or lighting and shoot the second image. You are given multiple red-circled points on the second image, choices of \"A, B, C, D\" are drawn beside each circle. Which point on the second image corresponds to the point in the first image?",
                     "blink_options", {k: "Point " + k for k in "ABCD"}),
    "t1": Phrasing("The point marked REF in image 1 is also visible in image 2, where four candidates A-D are circled. Which candidate is the same 3D point?", "t_paren", {k: "point " + k for k in "ABCD"}),
    "t2": Phrasing("Find the match of the REF point (image 1) among the labelled circles in image 2.", "t_dot", {k: k for k in "ABCD"}),
    "t3": Phrasing("Which circled point in the second image shows the same spot as REF in the first image?", "t_colon", {k: "the circle labelled " + k for k in "ABCD"}),
    "h1": Phrasing("Track the REF location from the first photo into the second photo. Where did it end up?", "h_inline", {k: "at circle " + k for k in "ABCD"}),
}
MMSI_APPR = {
    "orig": Phrasing("Figure 1 shows the front of the {x}, with the {x} at the center. Which side of the {x} is shown in Figure 2?", "mmsi",
                     {"front": "Front", "back": "Back", "left": "Left", "right": "Right"}),
    "t1": Phrasing("Call the side of {x_the} facing the camera in image 1 its front. Which of its sides faces the camera in image 2?", "t_paren",
                   {"front": "the front", "back": "the back", "left": "its left side", "right": "its right side"}),
    "t2": Phrasing("Image 1 views {x_the} from its front. From which side is it viewed in image 2?", "t_dot",
                   {"front": "front", "back": "back", "left": "left", "right": "right"}),
    "t3": Phrasing("Treat image 1 as the front view of {x_the}. What view of it does image 2 give?", "t_colon",
                   {"front": "front view", "back": "back view", "left": "left-side view", "right": "right-side view"}),
    "h1": Phrasing("If {x_the} faced the first camera, which of its faces would the second camera be looking at?", "h_inline",
                   {"front": "its front face", "back": "its back face", "left": "its left face", "right": "its right face"}),
}
MMSI_MSR = {
    "orig": Phrasing("Suppose I am standing by the {x} in Figure 1, facing the {y} in Figure 2. If I want to photograph the {z} shown in Figure 3, in which direction should I take the photo?", "mmsi"),
    "t1": Phrasing("I stand at {x_the} (image 1) and face {y_the} (image 2). Toward which direction is {z_the} (image 3)?", "t_paren"),
    "t2": Phrasing("Using all three images: from {x_the}, looking at {y_the}, where is {z_the}?", "t_dot"),
    "t3": Phrasing("Combine the photos. Standing next to {x_the} and facing {y_the}, which way must I point the camera to capture {z_the}?", "t_colon"),
    "h1": Phrasing("Chain the three views together. If my position is {x_the} and my heading is toward {y_the}, in what direction does {z_the} lie?", "h_inline"),
}
MSR_DIR = {"front": "Directly in front of me", "front-right": "To my right front", "right": "To my immediate right",
           "back-right": "To my right rear", "back": "Directly behind me", "back-left": "To my left rear",
           "left": "To my immediate left", "front-left": "To my left front"}
VS_SIM = {
    "orig": Phrasing("If you stand at {a} facing {b}, where is {c}?", "viewspatial"),
    "t1": Phrasing("Imagine standing by {a_the} and looking toward {b_the}. In which direction is {c_the}?", "t_paren"),
    "t2": Phrasing("From the position of {a_the}, facing {b_the}, where would {c_the} be?", "t_dot"),
    "t3": Phrasing("Using these views of the room: place yourself at {a_the}, turn to face {b_the}. Where is {c_the}?", "t_colon"),
    "h1": Phrasing("Suppose your body is at {a_the} and your nose points to {b_the}. Which way is {c_the}?", "h_inline"),
}
MC_MOVE = {
    "orig": Phrasing("Based on these two views showing the same scene: in which direction did I move from the first view to the second view?", "mindcube"),
    "t1": Phrasing("Two photos of one scene. How did I walk from where the first was taken to where the second was taken?", "t_paren"),
    "t2": Phrasing("From viewpoint 1 to viewpoint 2, describe my movement.", "t_dot"),
    "t3": Phrasing("I took the first view, moved, and took the second. Which way did I move?", "t_colon"),
    "h1": Phrasing("Compare the two viewpoints of the same objects. My displacement between them was:", "h_inline"),
}
YESNO_MC = ["Yes", "No"]


def around_header(view_names: str) -> str:
    n = view_names.count(",") + 1
    imgs = "image 1, 2, and 3" if n == 3 else "image 1, 2, 3, and 4"
    word = {3: "three", 4: "four"}[n]
    return f"Based on these {word} images ({imgs}) showing the same scene from different viewpoints ({view_names}): "


# =============================================================================
# View pairs (M1, M2, M3, M4)
# =============================================================================
TRANSLATIONS = [(dx, dz) for dx in (-1.6, -1.0, -0.6, 0.0, 0.6, 1.0, 1.6) for dz in (-1.2, -0.6, 0.0, 0.6, 1.2)
                if (dx, dz) != (0.0, 0.0)]
TURNS = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)


def pair_motions(kind: str, rng) -> List[Tuple[float, float, float, Optional[float]]]:
    """(right, forward, turn_left_deg, pitch_deg) candidates for camera 2 relative to camera 1."""
    out = []
    if kind == "translate":
        for dx, dz in TRANSLATIONS:
            for t in TURNS:
                out.append((dx, dz, t, None))
    elif kind == "rotate":
        for t in (-45.0, -35.0, -25.0, 25.0, 35.0, 45.0):
            out.append((0.0, 0.0, t, None))
    elif kind == "pitch":
        for p in (-35.0, 15.0):
            out.append((0.0, 0.0, 0.0, p))
    elif kind == "far":
        for dx, dz in TRANSLATIONS:
            for t in (-120.0, -90.0, -60.0, 60.0, 90.0, 120.0, 180.0):
                out.append((dx, dz, t, None))
    rng.shuffle(out)
    return out


def view_pairs(g: GenCtx, kind: str, max_pairs: int = 40, min_shared: int = 2,
               group: str = "") -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Camera pairs where >= ``min_shared`` landmarks pass the v10 visibility prefilter in both views."""
    ctx = g.ctx
    lms = landmarks(ctx)
    seeds = list(itertools.combinations(lms, 2))
    g.rng.shuffle(seeds)
    n = 0
    lim = g.cap(60, census=150)
    if group:
        g.count(group, "seeds_total", len(seeds))
        g.count(group, "seeds_examined", min(len(seeds), lim))
    for a, b in seeds[:lim]:
        if float(np.linalg.norm(ctx.obb[a].center[:2] - ctx.obb[b].center[:2])) > 3.5:
            continue
        cams1 = frame_candidates(ctx, [a, b], max_out=2)
        for c1 in cams1:
            cands = [mv.move(c1, dx, dz, t, p) for dx, dz, t, p in pair_motions(kind, g.rng)]
            cands = [c for c in cands if ctx.legal(c[:2, 3][None])[0]]
            if not cands:
                continue
            arr = np.asarray(cands)
            near = [o for o in lms if float(np.linalg.norm(ctx.obb[o].center[:2] - c1[:2, 3])) < 6.0]
            v1 = {o: bool(ctx.face_visible(o, c1[None], K, SIZE)[0]) and framed_loose(ctx.obb[o], c1, K, SIZE) for o in near}
            shared = np.zeros(len(arr), int)
            for o in [o for o in near if v1[o]]:
                fr = np.array([framed_loose(ctx.obb[o], c, K, SIZE) for c in arr])
                shared += (ctx.face_visible(o, arr, K, SIZE) & fr).astype(int)
            ok = np.flatnonzero(shared >= min_shared)
            if not len(ok):
                continue
            yield c1, arr[g.rng.choice(list(ok))]
            n += 1
            if n >= g.cap(max_pairs):
                return
            break


def render_pair(g: GenCtx, c1: np.ndarray, c2: np.ndarray, purpose: str, min_in_frame: float = 0.6):
    v1, a1 = g.bank.view(c1, purpose)
    v2, a2 = g.bank.view(c2, purpose)
    cand = sorted(set(a1.candidates_in_frustum(landmarks(g.ctx))) & set(a2.candidates_in_frustum(landmarks(g.ctx))))
    shared = shared_rec([a1, a2], cand, min_in_frame)
    return (v1, a1), (v2, a2), shared


# =============================================================================
# M1: camera-camera relations, camera motion, BLINK multi-view
# =============================================================================
def m1_combos(g: GenCtx) -> Iterator[Combo]:
    for kind in ("translate", "rotate", "pitch"):
        for c1, c2 in view_pairs(g, kind, max_pairs=12, group="M1"):
            yield Combo("M1", kind, (), [[c1, c2]], {"subtasks": SUBTASKS["M1"]})
    # BLINK multi-view: orbit around one object
    ctx = g.ctx
    pool = [o for o in scene_pool(ctx) if ctx.obb[o].longest_edge >= 0.4] or landmarks(ctx)
    g.rng.shuffle(pool)
    for o in pool[:g.cap(20)]:
        c = ctx.obb[o].center
        for r in (2.0, 2.6, 3.2):
            for a0 in np.linspace(0, 360, 12, endpoint=False):
                # random order: the first legal orbit is taken, a fixed order made almost every answer "right"
                for d in sorted((30.0, -30.0, 40.0, -40.0), key=lambda _: g.rng.random()):
                    e1 = c[:2] + r * np.array([math.cos(math.radians(a0)), math.sin(math.radians(a0))])
                    e2 = c[:2] + r * np.array([math.cos(math.radians(a0 + d)), math.sin(math.radians(a0 + d))])
                    if not ctx.legal(np.stack([e1, e2])).all():
                        continue
                    c1 = look_at((e1[0], e1[1], CAMERA_HEIGHT_M), c)
                    c2 = look_at((e2[0], e2[1], CAMERA_HEIGHT_M), c)
                    if not (framed(ctx.obb[o], c1, K, SIZE) and framed(ctx.obb[o], c2, K, SIZE)):
                        continue
                    if not ctx.face_visible(o, np.stack([c1, c2]), K, SIZE).all():
                        continue
                    yield Combo("M1", "orbit", (o,), [[c1, c2]], {"subtasks": ["BL-04"], "orbit_deg": d})
                    break
                else:
                    continue
                break
            else:
                continue
            break


def m1_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        c1, c2 = combo.shots[0]
        (v1, a1), (v2, a2), shared = render_pair(g, c1, c2, "M1")
        if len(shared) < 2:
            return "fewer_than_2_shared_objects"
        if not emit:
            return None
        imgs = [g.bank.save(v1), g.bank.save(v2)]
        rev = {"shared": shared, "boxes_per_image": [review_boxes(a1, shared, ["s"] * len(shared)),
                                                     review_boxes(a2, shared, ["s"] * len(shared))]}
        objs = [g.obj_entry(o, g.ctx.disp[o], "shared_landmark") for o in shared]
        ang, dist = mv.bearing(c1, c2[:3, 3])
        y1, p1 = yaw_pitch_of(c1)
        y2, p2 = yaw_pitch_of(c2)
        dyaw = (math.degrees(y2 - y1) + 540) % 360 - 180   # positive = turned left
        dpitch = math.degrees(p2 - p1)
        _, f, rr = mv.frame_h(c1)
        dv = c2[:2, 3] - c1[:2, 3]
        dx, dz = float(dv @ rr), float(dv @ f)
        vals = {"translation_right_m": round(dx, 3), "translation_forward_m": round(dz, 3),
                "turn_left_deg": round(dyaw, 1), "pitch_change_deg": round(dpitch, 1)}
        if combo.kind == "orbit":
            if not r.emit_ok("BL-04"):
                return None
            ans = "left" if combo.info["orbit_deg"] < 0 else "right"
            # orbit angle measured counter-clockwise from above: negative = clockwise = BLINK "left"
            g.emit("M1", ["BL-04", "SITE-C05"], "BLINK", "multi_image", [v1, v2], imgs, BLINK_MV, {}, ["left", "right"],
                   ans, "mc", gt("相机绕物体转：从上往下看顺时针 = 向左（BLINK 的定义）；转角 ±30° 或 ±40°，两张图都对着这个物体。",
                                 "BLINK 来自 Wild6D 视频真实相机运动；这里是已知的绕物体转动。", orbit_deg=combo.info["orbit_deg"], **vals),
                   objs, rev, keep_orig_order=["left", "right"])
            r.take("BL-04")
            return None
        opts = []
        if combo.kind == "translate":
            if dist >= 0.8 and mv.sector8(ang) is not None:
                opts.append(("MMSI-position-camera-camera", "dir8"))
            if abs(dx) >= 0.4 and abs(dz) >= 0.4:
                opts.append(("MMSI-position-camera-camera", "xyz_move"))
                if abs(dyaw) <= 15:
                    opts.append(("MMSI-motion-camera", "move"))
        elif combo.kind == "rotate":
            opts += [("MMSI-position-camera-camera", "xyz_rot"), ("MMSI-motion-camera", "rotate")]
        else:
            opts.append(("MMSI-motion-camera", "rotate"))
        live = [o for o in opts if r.need[o[0]] > 0]
        if not live:
            return None
        st, form = max(live, key=lambda o: r.need[o[0]])
        tags = [st, "SITE-C05"]
        if form == "dir8":
            lab = mv.sector8(ang)
            others = [k for k in mv.DIR8 if k != lab]
            g.rng.shuffle(others)
            keys = [lab] + others[:3]
            fam = {k: Phrasing(p.question, p.style, mv.MMSI_DIR8) for k, p in MMSI_CAMPOS.items()}
            g.emit("M1", tags, "MMSI-Bench", "multi_image", [v1, v2], imgs, fam, {}, keys, lab, "mc",
                   gt("相机 2 的位置在相机 1 水平坐标系里的方向，8 个 45° 扇区，离边界 <7.5° 不出题；平移 ≥0.8 m。",
                      "MMSI 原题由人工设计，没有统一公式；这里的答案由两台相机的已知位姿算出。", bearing_deg=round(ang, 1), **vals),
                   objs, rev)
        elif form == "xyz_move":
            xs = "positive" if dx > 0 else "negative"
            zs = "negative" if dz > 0 else "positive"
            lab = f"Move in the {xs} X direction and {zs} Z direction"
            keys = [f"Move in the {a} X direction and {b} Z direction" for a in ("positive", "negative") for b in ("positive", "negative")]
            g.emit("M1", tags, "MMSI-Bench", "multi_image", [v1, v2], imgs, MMSI_XYZ_MOVE, {}, keys, lab, "mc",
                   gt("相机 1 坐标系（+X 右、+Y 上、−Z 前）里相机 2 的平移方向；左右、前后分量都 ≥0.4 m。", "同上，答案由已知位姿算出。", **vals),
                   objs, rev)
        elif form == "xyz_rot":
            lab = "Rotate by a positive angle around the Y axis" if dyaw > 0 else "Rotate by a negative angle around the Y axis"
            keys = [f"Rotate by a {s} angle around the {a} axis" for s in ("positive", "negative") for a in ("Y", "Z")]
            g.emit("M1", tags, "MMSI-Bench", "multi_image", [v1, v2], imgs, MMSI_XYZ_ROT, {}, keys, lab, "mc",
                   gt("原地左转 = 绕 +Y（向上）轴正角度（右手系）；转角 25°–45°。", "同上。", **vals), objs, rev)
        elif form == "move":
            lab = ("Forward to the right" if dx > 0 else "Forward to the left") if dz > 0 else \
                  ("Right while moving backward" if dx > 0 else "Left while moving backward")
            keys = ["Forward to the left", "Forward to the right", "Left while moving backward", "Right while moving backward"]
            g.emit("M1", tags, "MMSI-Bench", "multi_image", [v1, v2], imgs, MMSI_MOTION, {}, keys, lab, "mc",
                   gt("相机 1 水平坐标系里的平移：前后、左右分量都 ≥0.4 m，转向 ≤15°。", "同上。", **vals), objs, rev)
        else:
            if combo.kind == "pitch":
                lab = "Up" if dpitch > 0 else "Down"
            else:
                lab = "Left" if dyaw > 0 else "Right"
            keys = ["Left", "Right", "Up", "Down"]
            g.emit("M1", tags, "MMSI-Bench", "multi_image", [v1, v2], imgs, MMSI_ROTATE, {}, keys, lab, "mc",
                   gt("原地转动：水平转 25°–45° 或俯仰变化 15°–35°。", "同上。", **vals), objs, rev)
        r.take(st)
        return None

    return realize


# =============================================================================
# M2: camera-object, object-object, measurement
# =============================================================================
def m2_combos(g: GenCtx) -> Iterator[Combo]:
    for kind in ("translate", "far"):
        for c1, c2 in view_pairs(g, kind, max_pairs=20, group="M2"):
            yield Combo("M2", kind, (), [[c1, c2]], {"subtasks": SUBTASKS["M2"]})


def only_in(vas, oids: Sequence[str], i: int) -> List[str]:
    """Objects recognisable in view i and absent from every other view."""
    out = []
    for o in oids:
        cls = [mv.vis_class(va, o, loose_absent=True) for va in vas]
        if cls[i] == "rec" and all(c == "abs" for j, c in enumerate(cls) if j != i):
            out.append(o)
    return out


def named_the(g: GenCtx, o: str) -> Tuple[str, str]:
    n = g.name_scene(o)
    return n, bare(n)


def m2_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        c1, c2 = combo.shots[0]
        (v1, a1), (v2, a2), shared = render_pair(g, c1, c2, "M2")
        if len(shared) < 2:
            return "fewer_than_2_shared_objects"
        named = [o for o in scene_pool(ctx)]
        only1 = only_in([a1, a2], named, 0)
        only2 = only_in([a1, a2], named, 1)
        rec1 = [o for o in named if mv.vis_class(a1, o) == "rec"]
        opts = []
        if only1:
            opts.append("MMSI-position-camera-object")
        if only2 and len(rec1) >= 2:
            opts.append("MMSI-position-object-object")
        if only1 and only2:
            opts.append("MMSI-attribute-measurement")
        if not opts:
            return "no_object_in_one_image_only"
        if not emit:
            return None
        def emit_st(st: str) -> Optional[str]:
            imgs = [g.bank.save(v1), g.bank.save(v2)]
            rev = {"shared": shared}
            land = [g.obj_entry(o, ctx.disp[o], "shared_landmark") for o in shared]
            if st == "MMSI-position-camera-object":
                for x in g.rng.sample(only1, len(only1)):
                    ang, dist = mv.bearing(c2, ctx.obb[x].center)
                    form = g.rng.choice(["quad", "four"])
                    lab = mv.quadrant(ang) if form == "quad" else mv.four_way(ang)
                    if lab is None:
                        continue
                    n, nb = named_the(g, x)
                    if form == "quad":
                        keys = ["front-left", "front-right", "back-left", "back-right"]
                        omap = {k: k.replace("-", " ") for k in keys}
                    else:
                        keys = ["front", "back", "left", "right"]
                        omap = {"front": "In front of me", "back": "Behind me", "left": "On my left", "right": "On my right"}
                    fam = {k: Phrasing(p.question, p.style, omap) for k, p in MMSI_CAMOBJ.items()}
                    g.emit("M2", [st, "SITE-C05"], "MMSI-Bench", "multi_image", [v1, v2], imgs, fam,
                           {"x": nb, "x_the": n, "X": cap(n)}, keys, lab, "mc",
                           gt("物体中心在相机 2 水平坐标系里的方向（四象限或前后左右四向），离分界 <10° 不出题；物体只在图 1 里能认出、在图 2 里完全看不到；两图至少共同拍到 2 个能认出的物体。",
                              "MMSI 原题由人工设计；这里由位姿和三维框中心算出。", bearing_deg=round(ang, 1), distance_m=round(dist, 2)),
                           [g.obj_entry(x, n, "key_only_in_image_1")] + land, rev)
                    r.take(st)
                    return None
                return "camobj_direction_ambiguous"
            if st == "MMSI-position-object-object":
                for rr_ in g.rng.sample(only2, len(only2)):
                    pairs = [(p, q) for p, q in itertools.permutations(rec1, 2) if rr_ not in (p, q)]
                    g.rng.shuffle(pairs)
                    for p, q in pairs:
                        pq = ctx.obb[p].center[:2] - ctx.obb[q].center[:2]
                        if np.linalg.norm(pq) < 0.8:
                            continue
                        k = g.rng.randrange(8)
                        pdir = mv.COMPASS8[k]
                        north = mv.rotate2(pq, 45.0 * k)  # p lies at compass angle 45k (clockwise) from north
                        lab = mv.compass(north, ctx.obb[q].center, ctx.obb[rr_].center)
                        if lab is None or float(np.linalg.norm(ctx.obb[rr_].center[:2] - ctx.obb[q].center[:2])) < 0.8:
                            continue
                        others = [c for c in mv.COMPASS8 if c != lab]
                        g.rng.shuffle(others)
                        keys = [lab] + others[:3]
                        omap = {c: cap(c) for c in mv.COMPASS8}
                        fam = {kk: Phrasing(pp.question, pp.style, omap) for kk, pp in MMSI_OBJOBJ.items()}
                        names = {}
                        for key, o in (("p", p), ("q", q), ("r", rr_)):
                            n, nb = named_the(g, o)
                            names[key], names[key + "_the"] = nb, n
                        g.emit("M2", [st, "SITE-C05"], "MMSI-Bench", "multi_image", [v1, v2], imgs, fam,
                               dict(names, pdir=pdir), keys, lab, "mc",
                               gt("指南针方向由题目给出的'P 在 Q 的某方向'定义（以 Q→P 的方向为准）；答 R 中心相对 Q 中心的 8 向，离扇区边界 <7.5° 不出题；R 只在图 2、P 和 Q 在图 1。",
                                  "MMSI 原题的参照方向由人从画面推断；这里精确按框中心计算。", p_dir=pdir),
                               [g.obj_entry(p, names["p_the"], "reference_p"), g.obj_entry(q, names["q_the"], "origin_q"),
                                g.obj_entry(rr_, names["r_the"], "query_only_in_image_2")] + land, rev)
                        r.take(st)
                        return None
                return "objobj_no_clear_direction"
            # measurement
            for x, y in itertools.product(g.rng.sample(only1, len(only1)), g.rng.sample(only2, len(only2))):
                X, Y = ctx.obb[x], ctx.obb[y]
                for dim in ("height", "length"):
                    vx = float(X.hi[2] - X.lo[2]) if dim == "height" else float(max(X.dims[k] for k in X.horizontal_axes()))
                    vy = float(Y.hi[2] - Y.lo[2]) if dim == "height" else float(max(Y.dims[k] for k in Y.horizontal_axes()))
                    if max(vx, vy) < 1.3 * min(vx, vy):
                        continue
                    adj = "taller" if dim == "height" else "longer"
                    nx, nxb = named_the(g, x)
                    ny, nyb = named_the(g, y)
                    keys = ["x", "y", "same", "sometimes"]
                    omap = {"x": "The " + nxb, "y": "The " + nyb, "same": "The same " + dim,
                            "sometimes": f"Sometimes the former is {adj}, sometimes the latter is {adj}"}
                    fam = {kk: Phrasing(pp.question, pp.style, omap) for kk, pp in MMSI_MEAS.items()}
                    g.emit("M2", [st, "SITE-C05"], "MMSI-Bench", "multi_image", [v1, v2], imgs, fam,
                           {"x": nxb, "y": nyb, "x_the": nx, "y_the": ny, "adj": adj, "dim": dim}, keys,
                           "x" if vx > vy else "y", "mc",
                           gt("高度 = 三维框竖直方向的长度；长度 = 框水平最长边；两者相差 ≥30% 才出题；X 只在图 1、Y 只在图 2。",
                              "MMSI 原题由人估计；这里用三维框尺寸（框比实物略大）。", **{x: round(vx, 3), y: round(vy, 3)}),
                           [g.obj_entry(x, nx, "only_in_image_1"), g.obj_entry(y, ny, "only_in_image_2")] + land, rev)
                    r.take(st)
                    return None
            return "measurement_no_clear_pair"

        # try the subtasks this pair supports, most needed first, until one yields a question
        reason = None
        for st in sorted([o for o in opts if r.need[o] > 0], key=lambda o: -r.need[o]):
            reason = emit_st(st)
            if reason is None:
                return None
        return reason

    return realize


# =============================================================================
# M3: regions (rooms named by a unique object; functional areas from object groups)
# =============================================================================
AREA_GROUPS = [("dining area", "dining table", ("chair",), "the dining table and its chairs"),
               ("sitting area", "sofa", ("coffee table", "armchair"), "the sofa and the coffee table"),
               ("sleeping area", "bed", ("bedside table", "nightstand"), "the bed and the bedside tables"),
               ("work area", "desk", ("chair",), "the desk and its chair")]
AREA_MARGIN_M = 0.4


def regions(g: GenCtx) -> List[Dict[str, Any]]:
    """Rooms named by an object that is unique in the scene and lies in that room, and functional areas:
    the convex hull of a group's footprints expanded by AREA_MARGIN_M."""
    ctx = g.ctx
    out = []
    for o in scene_pool(ctx):
        if ctx.scene_ref(o) != "the " + ctx.disp[o]:
            continue
        room = ctx.room_of_obj.get(o)
        if room is None or ctx.room_area(room) < 4.0:
            continue
        name = f"room with the {ctx.disp[o]}"
        if any(rg["kind"] == "room" and rg["room"] == room for rg in out):
            continue
        out.append({"kind": "room", "room": room, "name": name, "anchor": o,
                    "poly": ctx.rooms[room].polygon, "members": [o]})
    for area, head, partners, desc in AREA_GROUPS:
        heads = [o for o in ctx.objects if ctx.disp[o.id] == head and o.id in ctx.obb]
        if len(heads) != 1:
            continue
        h = heads[0].id
        mem = [h] + [o.id for o in ctx.objects if o.id in ctx.obb and any(ctx.disp[o.id].endswith(p) for p in partners)
                     and float(np.linalg.norm(ctx.obb[o.id].center[:2] - ctx.obb[h].center[:2])) < 1.6]
        if len(mem) < 2:
            continue
        pts = np.concatenate([ctx.obb[m].corners[:, :2] for m in mem])
        poly = offset_convex_polygon(convex_hull(pts), AREA_MARGIN_M)
        out.append({"kind": "area", "name": area, "anchor": h, "poly": poly, "members": mem, "desc": desc})
    return out


REGION_SECTOR_SHARE = 0.85   # a region "lies in" a direction when at least this share of its area does


def region_direction(c2w_or_origin, facing, poly: np.ndarray, mode: str) -> Optional[str]:
    """Direction label of a region: the sector that holds at least REGION_SECTOR_SHARE of the region's area
    (grid samples inside the polygon), seen from a camera (``facing`` None) or from an origin facing a point;
    None if no sector holds that much or the origin lies inside the region."""
    poly = np.asarray(poly, float)
    lo, hi = poly.min(0), poly.max(0)
    xs = np.linspace(lo[0], hi[0], 24)
    ys = np.linspace(lo[1], hi[1], 24)
    pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    pts = pts[point_in_polygon(pts, poly)]
    if not len(pts):
        return None
    origin = np.asarray(c2w_or_origin, float)
    o_xy = origin[:2, 3] if origin.ndim == 2 else origin[:2]
    if point_in_polygon(o_xy[None], poly)[0]:
        return None
    labs = []
    for v in pts:
        ang = mv.bearing(c2w_or_origin, v)[0] if facing is None else mv.bearing_from(c2w_or_origin, facing, v)
        labs.append(mv.quadrant(ang, 0.0) if mode == "quad" else (mv.four_way(ang, 0.0) if mode == "four" else mv.sector8(ang, 0.0)))
    lab, n = Counter(labs).most_common(1)[0]
    return lab if lab is not None and n >= REGION_SECTOR_SHARE * len(labs) else None


def m3_combos(g: GenCtx) -> Iterator[Combo]:
    """Constructive: camera 1 frames region A's anchor; camera 2 is turned / moved so that no member of A can
    show (whole boxes outside its view) while it shares >= 2 landmarks with camera 1; among those, prefer a
    camera 2 that sees another region B none of whose members camera 1 can show (region-region questions)."""
    regs = regions(g)
    if not regs:
        return
    ctx = g.ctx
    lms = landmarks(ctx)
    order = list(range(len(regs)))
    g.rng.shuffle(order)
    n = 0
    for ia in order:
        A = regs[ia]
        for c1 in frame_candidates(ctx, [A["anchor"]], max_out=3):
            near = [o for o in lms if o not in A["members"]
                    and float(np.linalg.norm(ctx.obb[o].center[:2] - c1[:2, 3])) < 6.0]
            v1 = [o for o in near if ctx.face_visible(o, c1[None], K, SIZE)[0] and framed_loose(ctx.obb[o], c1, K, SIZE)]
            if len(v1) < 1:
                continue
            cands = [mv.move(c1, dx, dz, t, p) for dx, dz, t, p in pair_motions("far", g.rng)]
            cands = [c for c in cands if ctx.legal(c[:2, 3][None])[0]
                     and not any(ctx.face_seen_any(m, c[None], K, SIZE)[0] for m in A["members"])]
            if not cands:
                continue
            arr = np.asarray(cands)
            shared = np.zeros(len(arr), int)
            for o in v1:
                fr = np.array([framed_loose(ctx.obb[o], c, K, SIZE) for c in arr])
                shared += (ctx.face_visible(o, arr, K, SIZE) & fr).astype(int)
            ok = shared >= 1
            if not ok.any():
                continue
            bonus = np.zeros(len(arr), bool)
            for B in regs:
                if B is A or any(ctx.face_seen_any(m, c1[None], K, SIZE)[0] for m in B["members"]):
                    continue
                bonus |= ctx.face_visible(B["anchor"], arr, K, SIZE)
            pick = np.flatnonzero(ok & bonus)
            pick = pick if len(pick) else np.flatnonzero(ok)
            c2 = arr[g.rng.choice(list(pick))]
            yield Combo("M3", "region_" + A["kind"], (A["anchor"],), [[c1, c2]],
                        {"subtasks": SUBTASKS["M3"], "regions": regs})
            n += 1
            break
        if n >= g.cap(40):
            return


def reg_name(rg) -> Tuple[str, str]:
    if rg["kind"] == "room":
        return rg["name"], "the " + rg["name"]
    return rg["name"], "the " + rg["name"] + f" ({rg['desc']})"


def m3_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        c1, c2 = combo.shots[0]
        (v1, a1), (v2, a2), shared = render_pair(g, c1, c2, "M3", min_in_frame=0.4)
        if len(shared) < 1:   # M3: one clearly shared object links the two images
            return "no_shared_object"
        regs = combo.info["regions"]

        def reg_vis(va, rg):
            return [mv.vis_class(va, m, loose_absent=True) for m in rg["members"]]

        in1 = [rg for rg in regs if any(c == "rec" for c in reg_vis(a1, rg)) and all(c == "abs" for c in reg_vis(a2, rg))]
        in2 = [rg for rg in regs if any(c == "rec" for c in reg_vis(a2, rg)) and all(c == "abs" for c in reg_vis(a1, rg))]
        named = scene_pool(ctx)
        rec1 = [o for o in named if mv.vis_class(a1, o) == "rec"]
        opts = []
        if in1:
            opts.append("MMSI-position-camera-region")
        if in2 and len(rec1) >= 2:
            opts.append("MMSI-position-object-region")
        if in1 and in2:
            opts.append("MMSI-position-region-region")
        if not opts:
            return "no_region_in_one_image_only"
        if not emit:
            return None
        def emit_st(st: str) -> Optional[str]:
            imgs = [g.bank.save(v1), g.bank.save(v2)]
            land = [g.obj_entry(o, ctx.disp[o], "shared_landmark") for o in shared]
            rdef = "两张图至少共同拍到 1 个能认出的物体。区域：'有某物体的房间' = 该物体所在房间的轮廓；功能区 = 一组物体（如餐桌和餐椅）在地面投影的最小外包凸多边形再向外扩 0.4 m。"
            rdiff = "MMSI 原题的区域由人工划定（如'刷牙区'），这里的区域按上面的几何规则定义，每题都标明。"
            if st == "MMSI-position-camera-region":
                for rg in in1:
                    if point_in_polygon(c2[:2, 3][None], rg["poly"])[0]:
                        continue
                    form = g.rng.choice(["quad", "four"])
                    lab = region_direction(c2, None, rg["poly"], form)
                    if lab is None:
                        continue
                    keys = ["front-left", "front-right", "back-left", "back-right"] if form == "quad" else ["front", "back", "left", "right"]
                    omap = ({k: k.replace("-", " ").capitalize() for k in keys} if form == "quad"
                            else {"front": "Front", "back": "Back", "left": "Left", "right": "Right"})
                    fam = {k: Phrasing(p.question, p.style, omap) for k, p in MMSI_CAMREG.items()}
                    nm, nm_the = reg_name(rg)
                    g.emit("M3", [st, "SITE-C05"], "MMSI-Bench", "multi_image", [v1, v2], imgs, fam,
                           {"reg": nm, "reg_the": nm_the, "Reg": cap(nm_the)}, keys, lab, "mc",
                           gt("区域面积至少 85% 落在相机 2 的同一个方向扇区里才出题。" + rdef, rdiff,
                              region_kind=rg["kind"], region_polygon=np.round(rg["poly"], 3).tolist()),
                           [g.obj_entry(m, ctx.disp[m], "region_member") for m in rg["members"]] + land,
                           {"region_polygon": np.round(rg["poly"], 3).tolist(), "shared": shared})
                    r.take(st)
                    return None
                return "camreg_ambiguous"
            if st == "MMSI-position-object-region":
                for rg in in2:
                    for x in g.rng.sample(rec1, len(rec1)):
                        if x in rg["members"] or point_in_polygon(ctx.obb[x].center[:2][None], rg["poly"])[0]:
                            continue
                        others = [o for o in rec1 if o != x]
                        p = g.rng.choice(others)
                        q = x
                        k = g.rng.randrange(8)
                        north = mv.rotate2(ctx.obb[p].center[:2] - ctx.obb[q].center[:2], 45.0 * k)
                        facing = ctx.obb[x].center[:2] + north
                        lab = region_direction(ctx.obb[x].center, facing, rg["poly"], "eight")
                        if lab is None:
                            continue
                        lab = mv.COMPASS8[mv.DIR8.index(lab)]
                        keys = [lab] + g.rng.sample([c for c in mv.COMPASS8 if c != lab], 3)
                        omap = {c: cap(c) for c in mv.COMPASS8}
                        fam = {kk: Phrasing(pp.question, pp.style, omap) for kk, pp in MMSI_OBJREG.items()}
                        nm, nm_the = reg_name(rg)
                        np_, npb = named_the(g, p)
                        nq, nqb = named_the(g, q)
                        g.emit("M3", [st, "SITE-C05"], "MMSI-Bench", "multi_image", [v1, v2], imgs, fam,
                               {"p": npb, "p_the": np_, "q": nqb, "q_the": nq, "x": nqb, "x_the": nq, "pdir": mv.COMPASS8[k],
                                "reg": nm, "reg_the": nm_the}, keys, lab, "mc",
                               gt("指南针方向由'P 在 Q 的某方向'定义；区域面积至少 85% 相对物体 X 落在同一个 8 向扇区内才出题。" + rdef, rdiff,
                                  region_kind=rg["kind"]),
                               [g.obj_entry(p, np_, "reference_p"), g.obj_entry(q, nq, "origin")] +
                               [g.obj_entry(m, ctx.disp[m], "region_member") for m in rg["members"]] + land,
                               {"region_polygon": np.round(rg["poly"], 3).tolist(), "shared": shared})
                        r.take(st)
                        return None
                return "objreg_ambiguous"
            for ra, rb in itertools.product(in1, in2):
                ca = np.mean(ra["poly"], axis=0)
                rec_any = [o for o in rec1 if o not in ra["members"]]
                if not rec_any:
                    break
                p = g.rng.choice(rec_any)
                k = g.rng.randrange(8)
                north = mv.rotate2(ctx.obb[p].center[:2] - ca, 45.0 * k)
                lab = region_direction(ca, ca + north, rb["poly"], "eight")
                if lab is None:
                    continue
                lab = mv.COMPASS8[mv.DIR8.index(lab)]
                keys = [lab] + g.rng.sample([c for c in mv.COMPASS8 if c != lab], 3)
                omap = {c: cap(c) for c in mv.COMPASS8}
                fam = {kk: Phrasing(pp.question, pp.style, omap) for kk, pp in MMSI_REGREG.items()}
                na, na_the = reg_name(ra)
                nb_, nb_the = reg_name(rb)
                np_, npb = named_the(g, p)
                g.emit("M3", [st, "SITE-C05"], "MMSI-Bench", "multi_image", [v1, v2], imgs, fam,
                       {"p": npb, "p_the": np_, "pdir": mv.COMPASS8[k], "reg_a": na, "reg_a_the": na_the, "reg_b": nb_,
                        "reg_b_the": nb_the}, keys, lab, "mc",
                       gt("从区域 A 的中心看区域 B，B 的面积至少 85% 落在同一个 8 向扇区内才出题；北向由'P 在 A 的某方向'定义。" + rdef, rdiff),
                       [g.obj_entry(p, np_, "reference_p")] + [g.obj_entry(m, ctx.disp[m], "region_a_member") for m in ra["members"]]
                       + [g.obj_entry(m, ctx.disp[m], "region_b_member") for m in rb["members"]] + land,
                       {"region_polygons": [np.round(ra["poly"], 3).tolist(), np.round(rb["poly"], 3).tolist()], "shared": shared})
                r.take(st)
                return None
            return "regreg_ambiguous"

        # try the subtasks this pair supports, most needed first, until one yields a question
        reason = None
        for st in sorted([o for o in opts if r.need[o] > 0], key=lambda o: -r.need[o]):
            reason = emit_st(st)
            if reason is None:
                return None
        return reason

    return realize


# =============================================================================
# M4: cross-image point correspondence (BLINK)
# =============================================================================
def m4_combos(g: GenCtx) -> Iterator[Combo]:
    for c1, c2 in view_pairs(g, "translate", max_pairs=20, min_shared=3, group="M4"):
        yield Combo("M4", "pair", (), [[c1, c2]], {"subtasks": ["BL-06"]})


def unproject(va, x: int, y: int) -> np.ndarray:
    z = float(va.depth[y, x])
    d = np.array([(x + 0.5 - va.K[0, 2]) / va.K[0, 0], (y + 0.5 - va.K[1, 2]) / va.K[1, 1], 1.0])
    return va.c2w[:3, 3] + (va.c2w[:3, :3] @ d) * z


def m4_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        c1, c2 = combo.shots[0]
        (v1, a1), (v2, a2), shared = render_pair(g, c1, c2, "M4")
        if len(shared) < 2:
            return "fewer_than_2_shared_objects"
        surf = np.zeros(a1.depth.shape, bool)
        for o in shared:
            surf |= a1.mask(o)
        for x, y in flat_surface_points(surf, a1.depth, erode_px=6, rng=g.rng, n=30):
            P = unproject(a1, x, y)
            uv, z = project(P[None], c2, a2.K)
            u2, v2_ = uv[0]
            if not (24 <= u2 < SIZE - 24 and 24 <= v2_ < SIZE - 24) or z[0] <= 0.2:
                continue
            d2 = float(a2.depth[int(v2_), int(u2)])
            if abs(d2 - z[0]) > max(0.06, 0.03 * z[0]):
                continue  # hidden in image 2 (or off the surface)
            win = a2.depth[int(v2_) - 3:int(v2_) + 4, int(u2) - 3:int(u2) + 4]
            if float(win.std()) > 0.03:
                continue
            # distractors: smooth surface points in image 2, >= 60 px from the answer and from each other
            pool = flat_surface_points(a2.alpha > 0.5, a2.depth, erode_px=4, rng=g.rng, n=200)
            ds = []
            for q in pool:
                if math.hypot(q[0] - u2, q[1] - v2_) >= 60 and all(math.hypot(q[0] - e[0], q[1] - e[1]) >= 60 for e in ds):
                    ds.append(q)
                if len(ds) == 3:
                    break
            if len(ds) < 3:
                continue
            if not emit:
                return None
            if not r.emit_ok("BL-06"):
                return None
            pts = [(float(u2), float(v2_))] + [(float(a), float(b)) for a, b in ds]
            order = list(range(4))
            g.rng.shuffle(order)
            labels = "ABCD"
            placed = [pts[i] for i in order]
            ans = labels[order.index(0)]
            img1 = g.bank.save_marked(v1, f"ref{x}_{y}", lambda im: draw_points(im, [(x, y)], ["REF"]))
            img2 = g.bank.save_marked(v2, f"cand{ans}{x}_{y}", lambda im: draw_points(im, placed, list(labels)))
            g.emit("M4", ["BL-06", "SITE-C05"], "BLINK", "multi_image", [v1, v2], [img1, img2], BL_CORR, {}, list("ABCD"),
                   ans, "mc", gt("图 1 可见表面上的点按深度和两台相机位姿投到图 2，再用图 2 的深度确认没被挡住（差 ≤ max(6 cm, 3%)）；干扰点是图 2 里离答案 ≥60 像素的其他平滑表面点。",
                                 "BLINK 原题来自 HPatches 的单应变换（平面场景）；这里是一般三维场景的深度重投影。",
                                 ref_xy=[x, y], answer_xy=[round(u2, 1), round(v2_, 1)], point_world=P.round(3).tolist()),
                   [g.obj_entry(o, g.ctx.disp[o], "shared_landmark") for o in shared],
                   {"ref": [x, y], "candidates": {labels[i]: placed[i] for i in range(4)}}, keep_orig_order=list("ABCD"))
            r.take("BL-06")
            return None
        return "no_valid_correspondence_point"

    return realize


# =============================================================================
# M5: which side of the object is shown (MMSI appearance)
# =============================================================================
SYMMETRIC = ("plant", "vase", "flower", "lamp", "stool", "ball", "pot", "round", "basket", "bin", "bucket")
SIDE_OFFSETS = {"front": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}


def m5_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = [o for o in scene_pool(ctx) if not any(s in ctx.disp[o] for s in SYMMETRIC) and ctx.obb[o].longest_edge >= 0.4]
    g.rng.shuffle(pool)
    for o in pool:
        c = ctx.obb[o].center
        found = 0
        for side, off in sorted(SIDE_OFFSETS.items(), key=lambda kv: g.rng.random()):
            for a0 in g.rng.sample(range(0, 360, 20), 18):
                hit = False
                for r1, r2 in ((2.4, 2.4), (2.0, 3.0), (3.0, 2.2)):
                    if side == "front" and abs(r1 - r2) < 0.5:
                        continue
                    e1 = c[:2] + r1 * np.array([math.cos(math.radians(a0)), math.sin(math.radians(a0))])
                    a2 = a0 + off + g.rng.uniform(-10, 10)
                    e2 = c[:2] + r2 * np.array([math.cos(math.radians(a2)), math.sin(math.radians(a2))])
                    if not ctx.legal(np.stack([e1, e2])).all():
                        continue
                    c1 = look_at((e1[0], e1[1], CAMERA_HEIGHT_M), c)
                    c2 = look_at((e2[0], e2[1], CAMERA_HEIGHT_M), c)
                    if not (framed(ctx.obb[o], c1, K, SIZE) and framed(ctx.obb[o], c2, K, SIZE)):
                        continue
                    if not ctx.face_visible(o, np.stack([c1, c2]), K, SIZE).all():
                        continue
                    yield Combo("M5", side, (o,), [[c1, c2]], {"subtasks": ["MMSI-attribute-appearance"],
                                                              "azimuths": [a0, a2]})
                    found += 1
                    hit = True
                    break
                if hit:
                    break
            if found >= 2:
                break


def m5_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        (o,) = combo.objects
        c1, c2 = combo.shots[0]
        v1, a1 = g.bank.view(c1, "M5")
        v2, a2 = g.bank.view(c2, "M5")
        if not (a1.obj(o).recognisable(0.8) and a2.obj(o).recognisable(0.8)):
            return "object_not_recognisable_in_both"
        if not emit:
            return None
        if not r.emit_ok("MMSI-attribute-appearance"):
            return None
        n, nb = named_the(g, o)
        imgs = [g.bank.save(v1), g.bank.save(v2)]
        az = combo.info["azimuths"]
        g.emit("M5", ["MMSI-attribute-appearance", "SITE-C05"], "MMSI-Bench", "multi_image", [v1, v2], imgs, MMSI_APPR,
               {"x": nb, "x_the": n}, ["front", "back", "left", "right"], combo.kind, "mc",
               gt("以图 1 拍到的那一面为'正面'（把物体看成面向相机 1）；相机 2 相对物体的方位角与相机 1 相差约 0°（另换距离）、180°、+90°、-90° 分别为正面、背面、物体自身的左侧、右侧（物体面向相机 1 时，相机 1 右手边是物体的左侧）；误差 ≤10°。",
                  "MMSI 原题由人判断；这里由两台相机的方位角算出，'左侧'按物体自身左右定义（题目未写明，可能有歧义）。",
                  azimuth_deg=[round(a, 1) for a in az]),
               [g.obj_entry(o, n, "object")], {"boxes_per_image": [review_boxes(a1, [o], ["X"]), review_boxes(a2, [o], ["X"])]})
        r.take("MMSI-attribute-appearance")
        return None

    return realize


# =============================================================================
# M6 / M11: panning chains -- three images from one spot, turning step by step
# =============================================================================
def pan_chains(g: GenCtx, n_img: int, max_out: int = 30, group: str = "") -> Iterator[Tuple[List[np.ndarray], List[str]]]:
    """From a legal spot, yaws psi, psi -/+ step, ... (turning right or left, random per spot so the answers of
    M6/M11 do not all fall on one side); key object i must pass the visibility prefilter in image i (within 20 deg
    of its axis) and lie outside the other images' field of view."""
    ctx = g.ctx
    named = [o for o in simple_named_pool(ctx) if ctx.obb[o].longest_edge >= 0.3]
    if len(named) < n_img:
        return
    grid = ctx.legal_grid(0.5)
    if not len(grid):
        return
    idx = list(range(len(grid)))
    g.rng.shuffle(idx)
    n = 0
    if group:
        g.count(group, "grid_points_total", len(idx))
        g.count(group, "grid_points_examined", min(len(idx), g.cap(150, census=400)))
    lms_all = landmarks(ctx)
    for i in idx[:g.cap(150, census=400)]:
        p = grid[i]
        lm_near = [o for o in lms_all if float(np.linalg.norm(ctx.obb[o].center[:2] - p)) < 5.0]
        done = False
        turn = g.rng.choice((-1.0, 1.0))
        for step in (65.0, 75.0):
            for psi0 in range(0, 360, 30):
                yaws = [math.radians(psi0 + turn * k * step) for k in range(n_img)]
                cams = [pose_yaw_pitch((p[0], p[1], CAMERA_HEIGHT_M), y, math.radians(-12.0)) for y in yaws]
                keys = []
                for k, c in enumerate(cams):
                    cand = []
                    for o in named:
                        ang, dist = mv.bearing(c, ctx.obb[o].center)
                        a = ang if ang <= 180 else ang - 360
                        if abs(a) > 20 or not (1.0 <= dist <= 5.0):
                            continue
                        others = [cams[j] for j in range(n_img) if j != k]
                        if any(abs(((mv.bearing(c_o, ctx.obb[o].center)[0] + 180) % 360) - 180) < 55 for c_o in others):
                            continue
                        cand.append(o)
                    others = [cams[j] for j in range(n_img) if j != k]
                    cand = [o for o in cand if ctx.face_visible(o, c[None], K, SIZE)[0]
                            and not ctx.face_seen_any(o, np.stack(others), K, SIZE).any()]
                    if not cand:
                        break
                    keys.append(g.rng.choice(cand))
                if (len(keys) == n_img and len(set(keys)) == n_img and g.distinct_names(keys)
                        and all(any(framed_loose(ctx.obb[o], cams[j], K, SIZE) and framed_loose(ctx.obb[o], cams[j + 1], K, SIZE)
                                    and ctx.face_visible(o, np.stack([cams[j], cams[j + 1]]), K, SIZE).all()
                                    for o in lm_near) for j in range(n_img - 1))):
                    yield cams, keys
                    n += 1
                    done = True
                    if n >= g.cap(max_out):
                        return
                    break
            if done:
                break


def chain_ok(g: GenCtx, vas, keys: Sequence[str]) -> Optional[str]:
    for i, k in enumerate(keys):
        cls = [mv.vis_class(va, k, loose_absent=True) for va in vas]
        if cls[i] != "rec" or any(c != "abs" for j, c in enumerate(cls) if j != i):
            return "key_object_not_only_in_its_image"
    lms = landmarks(g.ctx)
    for i in range(len(vas) - 1):
        cand = sorted(set(vas[i].candidates_in_frustum(lms)) & set(vas[i + 1].candidates_in_frustum(lms)))
        if not shared_rec([vas[i], vas[i + 1]], cand):
            return "consecutive_images_share_no_object"
    return None


def m6_combos(g: GenCtx) -> Iterator[Combo]:
    for cams, keys in pan_chains(g, 3, group="M6"):
        yield Combo("M6", "pan3", tuple(keys), [cams], {"subtasks": ["MMSI-multistep"]})


def m6_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        cams = combo.shots[0]
        vv = [g.bank.view(c, "M6") for c in cams]
        vas = [va for _, va in vv]
        why = chain_ok(g, vas, combo.objects)
        if why:
            return why
        # roles are shuffled (otherwise "stand at the first, face the second" puts the third almost always on the
        # turning side); the images are then reordered so that figure i shows the object of role i
        order = list(range(3))
        g.rng.shuffle(order)
        x, y, z = (combo.objects[i] for i in order)
        ang = mv.bearing_from(ctx.obb[x].center, ctx.obb[y].center, ctx.obb[z].center)
        lab = mv.sector8(ang)
        if lab is None:
            return "msr_direction_boundary"
        if not emit:
            return None
        if not r.emit_ok("MMSI-multistep"):
            return None
        names = {}
        for k, o in zip("xyz", (x, y, z)):
            n, nb = named_the(g, o)
            names[k], names[k + "_the"] = nb, n
        keys = [lab] + g.rng.sample([d for d in mv.DIR8 if d != lab], 3)
        fam = {kk: Phrasing(pp.question, pp.style, MSR_DIR) for kk, pp in MMSI_MSR.items()}
        vv = [vv[i] for i in order]
        imgs = [g.bank.save(v) for v, _ in vv]
        g.emit("M6", ["MMSI-multistep", "SITE-C05"], "MMSI-Bench", "multi_image", [v for v, _ in vv], imgs, fam, names,
               keys, lab, "mc",
               gt("三张图从同一位置朝同一方向（随机向左或向右）依次转身拍摄，拍摄顺序上相邻的两张至少共同拍到一个能认出的物体；X、Y、Z 各只在一张图里能认出、在其他图里完全看不到，三个物体的角色随机分配，图按角色重排成图 1、2、3；答案是站在 X 中心面向 Y 中心时 Z 的 8 向方位（离扇区边界 <7.5° 不出题）。少任何一张图都无法定位对应物体。",
                  "MMSI 多步题由人工设计，可组合任意能力；这里只组合'跨图定位 + 想象站位'两步。",
                  bearing_deg=round(ang, 1), capture_order_of_figures=[i + 1 for i in order]),
               [g.obj_entry(x, names["x_the"], "stand_at_only_in_image_1"), g.obj_entry(y, names["y_the"], "face_only_in_image_2"),
                g.obj_entry(z, names["z_the"], "target_only_in_image_3")], {})
        r.take("MMSI-multistep")
        return None

    return realize


def vs_distractors8(answer: str, rng) -> List[str]:
    """ViewSpatial Sce_Sim_make.generate_qa_pairs: single directions exclude compounds containing them,
    compounds exclude their parts."""
    if "-" in answer:
        parts = answer.split("-")
        pool = [d for d in mv.DIR8 if d != answer and d not in parts]
    else:
        pool = [d for d in mv.DIR8 if d != answer and answer not in d.split("-")]
    rng.shuffle(pool)
    return pool[:3]


def m11_combos(g: GenCtx) -> Iterator[Combo]:
    for cams, keys in pan_chains(g, 3, group="M11"):
        yield Combo("M11", "pan3", tuple(keys), [cams], {"subtasks": ["VS-05"]})


def m11_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        cams = combo.shots[0]
        vv = [g.bank.view(c, "M11") for c in cams]
        vas = [va for _, va in vv]
        why = chain_ok(g, vas, combo.objects)
        if why:
            return why
        order = list(combo.objects)
        g.rng.shuffle(order)
        a, b, c = order
        ang = mv.bearing_from(ctx.obb[a].center, ctx.obb[b].center, ctx.obb[c].center)
        lab = mv.sector8(ang)
        if lab is None:
            return "vs05_direction_boundary"
        if not emit:
            return None
        if not r.emit_ok("VS-05"):
            return None
        names = {}
        for k, o in zip("abc", (a, b, c)):
            n, nb = named_the(g, o)
            names[k], names[k + "_the"] = nb, n
        keys = [lab] + vs_distractors8(lab, g.rng)
        imgs = [g.bank.save(v) for v, _ in vv]
        g.emit("M11", ["VS-05", "SITE-C05"], "ViewSpatial-Bench", "multi_image", [v for v, _ in vv], imgs, VS_SIM, names,
               keys, lab, "mc",
               gt("ViewSpatial 场景模拟题的算法：以 A 指向 B 为前方（水平），C 的方位角 atan2(右, 前) 落在 8 个 45° 扇区（离边界 <7.5° 不出题）；A、B、C 分别只在不同的一张图里，相邻图共同拍到至少一个物体。",
                  "ViewSpatial 原题用约 9 张 ScanNet 帧且每帧可含多个题目物体；这里用 3 张转身连拍，每个物体只在一张图里。", bearing_deg=round(ang, 1)),
               [g.obj_entry(a, names["a_the"], "stand_at"), g.obj_entry(b, names["b_the"], "face"),
                g.obj_entry(c, names["c_the"], "query")], {})
        r.take("VS-05")
        return None

    return realize


# =============================================================================
# M7: rotation in place (MindCube Rotation)
# =============================================================================
ROT_HEAD = {
    2: ("These two images (image 1 and 2) show the same scene from two different viewpoints. {turn_desc} "
        "Based on these two images: "),
    3: ("These three images (image 1, 2, and 3) show the same scene from three different viewpoints. The image 2 was taken "
        "after turning the camera 90 degrees to the right (clockwise) from the position of image 1. For image 3, the camera "
        "was turned another 90 degrees right, so it's basically facing the opposite direction of image 1. Based on these three images: "),
    4: ("These four images (image 1, 2, 3, and 4) show the same scene from different viewpoints. The camera basically "
        "stood in one spot and turned 90 degrees each time to get all sides of the area. Based on these four images: "),
}
TURN_TXT = {0: "", -1: "then I turn 90 degrees to the left, ", 1: "then I turn 90 degrees to the right, ", 2: "then I turn 180 degrees around, "}


def m7_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    named = [o for o in simple_named_pool(ctx) if ctx.obb[o].longest_edge >= 0.3]
    grid = ctx.legal_grid(0.5)
    idx = list(range(len(grid)))
    g.rng.shuffle(idx)
    n = 0
    g.count("M7", "grid_points_total", len(idx))
    g.count("M7", "grid_points_examined", min(len(idx), g.cap(200, census=400)))
    for i in idx[:g.cap(200, census=400)]:
        p = grid[i]
        for psi0 in ((0.0, 90.0, 180.0, 270.0) if g.rng.random() < 0.5 else (45.0, 135.0, 225.0, 315.0)):
            cams = [pose_yaw_pitch((p[0], p[1], CAMERA_HEIGHT_M), math.radians(psi0 - 90.0 * k), math.radians(-12.0))
                    for k in range(4)]   # view k: turned k times 90 degrees to the right
            mains = []
            for c in cams:
                cand = []
                for o in named:
                    ang, dist = mv.bearing(c, ctx.obb[o].center)
                    a = ang if ang <= 180 else ang - 360
                    if abs(a) <= 20 and 1.0 <= dist <= 5.0:
                        cand.append((dist, o))
                cand = [o for _, o in sorted(cand) if ctx.face_visible(o, c[None], K, SIZE)[0]]
                mains.append(cand[0] if cand else None)
            have = [k for k in range(4) if mains[k] is not None]
            if len(have) < 2:
                continue
            if len({mains[k] for k in have}) < len(have) or not g.distinct_names([mains[k] for k in have]):
                continue
            yield Combo("M7", "rotation", tuple(m or "" for m in mains), [cams], {"subtasks": SUBTASKS["M7"], "pos": p.tolist()})
            n += 1
            if n >= g.cap(40):
                return
            break


def m7_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        mains = list(combo.objects)
        cams = combo.shots[0]
        ok = []
        for k in range(4):
            if not mains[k]:
                continue
            _, va = g.bank.view(cams[k], "M7")
            if mv.vis_class(va, mains[k], 0.5) == "rec":
                ok.append(k)
        sets = []
        if all(k in ok for k in range(4)):
            sets.append(("four", [0, 1, 2, 3]))
        if all(k in ok for k in (0, 1, 2)):
            sets.append(("three", [0, 1, 2]))
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            if a in ok and b in ok:
                sets.append(("two_cw", [a, b]))
                sets.append(("two_ccw", [b, a]))
        if not sets:
            return "fewer_than_2_views_with_a_clear_object"
        if not emit:
            return None
        st = pick_subtask(r, SUBTASKS["M7"])
        if st is None:
            return None
        kind, views = sets[0] if g.rng.random() < 0.6 else g.rng.choice(sets)
        n_img = len(views)
        facing = [v % 4 for v in views]          # right turns (x 90 deg) from view 0
        obj_at = {f: mains[v] for f, v in zip(facing, views)}
        names = {f: named_the(g, obj_at[f]) for f in obj_at}
        choice = None
        for _ in range(60):
            img_i = g.rng.randrange(n_img)
            turn = 0 if st == "MC-rotation-current" else g.rng.choice([-1, 1, 2])
            ask = g.rng.choice(["right", "left", "behind"])
            h = (facing[img_i] + turn) % 4
            target = (h + {"right": 1, "left": -1, "behind": 2}[ask]) % 4
            if target in obj_at:
                choice = (img_i, turn, ask, target)
                break
        if choice is None:
            return "no_answerable_rotation_question"
        img_i, turn, ask, target = choice
        if kind == "two_cw":
            head = ROT_HEAD[2].format(turn_desc="Image 2 was taken after turning the camera 90 degrees to the right (clockwise) from the position of image 1.")
        elif kind == "two_ccw":
            head = ROT_HEAD[2].format(turn_desc="The image 2 was taken after turning the camera 90 degrees to the left (counterclockwise) from the position of image 1.")
        else:
            head = ROT_HEAD[n_img]
        what = {"right": "what is to my right?", "left": "what is to my left?", "behind": "what is to my behind?"}[ask]
        q_orig = head + f"If I am standing at the same spot and facing the same direction as shown in image {img_i + 1}, " + TURN_TXT[turn] + what
        para_turn = {0: "", -1: " and then turn left by 90 degrees", 1: " and then turn right by 90 degrees", 2: " and then turn around"}[turn]
        para_ask = {"right": "on my right", "left": "on my left", "behind": "behind me"}[ask]
        sense = "counterclockwise" if kind == "two_ccw" else "clockwise"
        fam = {
            "orig": Phrasing(q_orig, "mindcube"),
            "t1": Phrasing(f"All {n_img} photos were taken from one spot by turning in place. If I face the way image {img_i + 1} faces{para_turn}, what is {para_ask}?", "t_paren"),
            "t2": Phrasing(f"I stand where these photos were taken, looking as in image {img_i + 1}{para_turn}. Which object is {para_ask}?", "t_dot"),
            "t3": Phrasing(f"The camera rotated in place ({sense}) between the shots. Facing as in image {img_i + 1}{para_turn}, name the object {para_ask}.", "t_colon"),
            "h1": Phrasing(f"Imagine yourself at the camera's spot, heading as in image {img_i + 1}{para_turn}. What do you find {para_ask}?", "h_inline"),
        }
        keys = sorted({cap(names[f][1]) for f in obj_at})
        ans = cap(names[target][1])
        if len(keys) < 2:
            return "single_option"
        vids = [g.bank.view(cams[v], "M7")[0] for v in views]
        imgs = [g.bank.save(v) for v in vids]
        g.emit("M7", [st] + TAGS_ROT, "MindCube", "multi_image", vids, imgs, fam, {}, keys, ans, "mc",
               gt("同一位置原地每次右转 90° 拍摄，每张图正中（±20°）有一个能认出的物体；按题目给的朝向和转身算出该方向是哪张图的物体。",
                  "MindCube 原题由人选景、人标物体；这里按位姿计算，选项只含这几张图的主物体。",
                  views=views, facing=facing[img_i], turn=turn, ask=ask),
               [g.obj_entry(obj_at[f], names[f][0], f"main_of_facing_{f}") for f in obj_at], {})
        r.take(st)
        return None

    return realize


def signed(ang: float) -> float:
    return ((ang + 180.0) % 360.0) - 180.0


# =============================================================================
# M8: around two objects (MindCube Around)
# =============================================================================
AROUND_HIDDEN_MAX_FRACTION = 0.35   # far object's visible share of its projection must be below this ...
AROUND_MIN_OCCLUDED = 0.40          # ... and at least this share of it must be covered by something in front
                                    # (first setting; the user calibrates it on the review page)


def m8_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = [o for o in simple_named_pool(ctx) if ctx.obb[o].longest_edge >= 0.3]
    pairs = [(a, b) for a, b in itertools.combinations(pool, 2)
             if 0.6 <= float(np.linalg.norm(ctx.obb[a].center[:2] - ctx.obb[b].center[:2])) <= 2.5
             and g.distinct_names([a, b])]
    g.rng.shuffle(pairs)
    for a, b in pairs:
        A, B = ctx.obb[a].center, ctx.obb[b].center
        M = (A + B) / 2.0
        u = (A[:2] - B[:2]) / np.linalg.norm(A[:2] - B[:2])     # from B to A
        for sgn in (1.0, -1.0):
            n = sgn * np.array([-u[1], u[0]])
            front = back = None
            for D in (2.2, 2.8, 3.4):
                e = M[:2] + D * n
                if ctx.legal(e[None])[0]:
                    c = look_at((e[0], e[1], CAMERA_HEIGHT_M), M)
                    if signed(mv.bearing(c, A)[0]) < signed(mv.bearing(c, B)[0]) - 5.0:
                        if framed(ctx.obb[a], c, K, SIZE) and framed(ctx.obb[b], c, K, SIZE) and \
                           ctx.face_visible(a, c[None], K, SIZE)[0] and ctx.face_visible(b, c[None], K, SIZE)[0]:
                            front = c
                            break
            if front is None:
                continue
            for D in (2.2, 2.8, 3.4):
                e = M[:2] - D * n
                if ctx.legal(e[None])[0]:
                    back = look_at((e[0], e[1], CAMERA_HEIGHT_M), M)
                    break

            def side(near_xy, away, near_id):
                for off in (0.0, 8.0, -8.0, 15.0, -15.0, 25.0, -25.0, 35.0, -35.0):
                    for dist in (1.2, 1.6, 2.0, 2.5):
                        e = near_xy + dist * mv.rotate2(away, off)
                        if ctx.legal(e[None])[0]:
                            c = look_at((e[0], e[1], CAMERA_HEIGHT_M), M)
                            if ctx.face_visible(near_id, c[None], K, SIZE)[0]:
                                return c
                return None

            left = side(A[:2], u, a)       # beyond A: A in front of B
            right = side(B[:2], -u, b)     # beyond B: B in front of A
            if left is None and right is None:
                continue
            yield Combo("M8", "around", (a, b), [[front, left, right, back]], {"subtasks": SUBTASKS["M8"]})
            break


def m8_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        a, b = combo.objects
        front, left, right, back = combo.shots[0]
        vf, af = g.bank.view(front, "M8")
        if not (af.obj(a).recognisable() and af.obj(b).recognisable()):
            return "front_view_not_clear"
        ba, bb = af.obj(a).bbox, af.obj(b).bbox
        if not (af.obj(a).centroid[0] + 10 < af.obj(b).centroid[0]):
            return "front_view_a_not_left_of_b"
        sides = {}
        for nm, cam, near, far in (("left", left, a, b), ("right", right, b, a)):
            if cam is None:
                continue
            vid, va = g.bank.view(cam, "M8")
            on, of = va.obj(near), va.obj(far)
            hidden = (of.fraction < AROUND_HIDDEN_MAX_FRACTION and of.occ_px / max(of.sil_px, 1) >= AROUND_MIN_OCCLUDED
                      and of.sil_px > 0)
            sides[nm] = {"vid": vid, "va": va, "cam": cam, "valid": bool(on.recognisable(0.6) and hidden),
                         "far_fraction": round(of.fraction, 3), "far_occluded": round(of.occ_px / max(of.sil_px, 1), 3)}
        valid = [s for s in ("left", "right") if s in sides and sides[s]["valid"]]
        if not valid:
            return "no_side_view_with_occlusion"
        if not emit:
            return None
        st = pick_subtask(r, SUBTASKS["M8"])
        if st is None:
            return None
        na, nab = named_the(g, a)
        nb, nbb = named_the(g, b)
        occ_vals = {s: {k: sides[s][k] for k in ("far_fraction", "far_occluded", "valid")} for s in sides}
        objs = [g.obj_entry(a, na, "A_left_in_front_view"), g.obj_entry(b, nb, "B_right_in_front_view")]
        defin = (f"正视图：相机在 A、B 连线的垂直方向上，A 在 B 左边；左视图：相机在 B→A 延长线 A 的外侧（可偏 ≤15°），A 挡住 B；右视图对称。"
                 f"'挡住'用深度核验：远处物体可见部分 < 投影的 {AROUND_HIDDEN_MAX_FRACTION}，且被前景盖住的部分 ≥ {AROUND_MIN_OCCLUDED}（待你审图校准）。")
        diff = "MindCube 原题由人选景、人工标注遮挡；这里按几何构造并用渲染深度核验。"
        if st == "MC-around-q1":
            s = g.rng.choice(valid)
            lab = mv.movement_label(front, sides[s]["cam"])
            if lab is None:
                return "q1_movement_ambiguous"
            imgs = [g.bank.save(vf), g.bank.save(sides[s]["vid"])]
            keys = ["Directly left", "Directly right", "Diagonally forward and left", "Diagonally forward and right"]
            g.emit("M8", [st, "MC-relation-agent-agent", "MC-dynamics-simultaneous"] + TAGS_AROUND, "MindCube", "multi_image",
                   [vf, sides[s]["vid"]], imgs, MC_MOVE, {}, keys, lab, "mc",
                   gt(defin + " q1 答案：相机 2 位置在相机 1 水平坐标系里的方位（正右 90°±20°、右前 45°±20° 等）。", diff,
                      side=s, occlusion=occ_vals), objs, {})
            r.take(st)
            return None
        # three or four images: front, left, right (180 deg) or front, left, back, right (360 deg)
        views = [("front", vf, front), ("left", sides["left"]["vid"] if "left" in sides else None, left),
                 ("right", sides["right"]["vid"] if "right" in sides else None, right)]
        if any(v[1] is None for v in views):
            return "missing_side_view_for_three_view_set"
        use_back = back is not None and g.rng.random() < 0.4
        if use_back:
            vb, ab = g.bank.view(back, "M8")
            if not (ab.obj(a).recognisable(0.6) and ab.obj(b).recognisable(0.6)):
                use_back = False
        if use_back:
            order = [("front", vf), ("left", sides["left"]["vid"]), ("back", vb), ("right", sides["right"]["vid"])]
            header = around_header("front, left, back, and right")
        else:
            order = [("front", vf), ("left", sides["left"]["vid"]), ("right", sides["right"]["vid"])]
            header = around_header("front, left, and right")
        idx = {nm: i + 1 for i, (nm, _) in enumerate(order)}
        s = g.rng.choice(valid)
        img_k = idx[s]
        near, far = (na, nb) if s == "left" else (nb, na)
        near_b, far_b = (nab, nbb) if s == "left" else (nbb, nab)
        if st == "MC-around-q2":
            truth = g.rng.random() < 0.5
            X, Y = (far_b, near_b) if truth else (near_b, far_b)   # "is there a X behind the Y?"
            q = header + f"From the viewpoint presented in image {img_k}, is there a {X} behind the {Y}?"
            fam = {"orig": Phrasing(q, "mindcube", {"Yes": "Yes", "No": "No."}),
                   "t1": Phrasing(f"Looking from where image {img_k} was taken, is {cap('the ' + X)} hidden behind the {Y}?", "t_paren"),
                   "t2": Phrasing(f"In the viewpoint of image {img_k}, does the {X} lie behind the {Y}?", "t_dot"),
                   "t3": Phrasing(f"From image {img_k}'s camera, would you find the {X} behind the {Y}?", "t_colon"),
                   "h1": Phrasing(f"Stand where image {img_k} was taken. Is the {Y} between you and the {X}?", "h_inline")}
            imgs = [g.bank.save(v) for _, v in order]
            g.emit("M8", [st, "MC-relation-object-object", "MC-perspective-self"] + TAGS_AROUND, "MindCube", "multi_image",
                   [v for _, v in order], imgs, fam, {}, YESNO_MC, "Yes" if truth else "No", "yesno",
                   gt(defin + " q2 答案：该侧视图里近处物体在前、远处物体在后。", diff, side=s, occlusion=occ_vals), objs, {},
                   keep_orig_order=YESNO_MC)
            r.take(st)
            return None
        # q3: turn and go straight
        turn = g.rng.choice(["left", "right"])
        tgt_id = g.rng.choice([a, b])
        tgt_b = nab if tgt_id == a else nbb
        cam = sides[s]["cam"]
        d0, d1 = mv.turn_move_distance(cam, turn, ctx.obb[tgt_id].center, step=1.0)
        rel = (d1 - d0) / max(d0, 1e-6)
        c_after = mv.move(mv.move(cam, turn_left_deg=90.0 if turn == "left" else -90.0), forward=1.0)
        ang_after, _ = mv.bearing(c_after, ctx.obb[tgt_id].center)
        if rel < -0.1:
            ans = "closer"
        elif rel > 0.08:
            ans = "farther"
        else:
            qd = mv.quadrant(ang_after)
            if qd is None:
                return "q3_ambiguous"
            ans = "same_" + qd
        fb = "front"
        keys = ["farther", "closer", f"same_{fb}-right", f"same_{fb}-left"]
        if ans not in keys:
            keys[2] = ans
        omap = {"farther": "It would become farther from me.", "closer": "It would become closer to me."}
        for k in keys:
            if k.startswith("same_"):
                omap[k] = f"The distance wouldn't change much, but it would be on my {k[5:]}."
        form = g.rng.choice(["will", "what"])
        tail = (f"will I get closer to the {tgt_b}?" if form == "will" else f"what would happen to my relative position to {tgt_b}?")
        q = header + f"If I am standing at the same spot and facing the same direction as shown in image {img_k}, then I turn {turn} and go straight, " + tail
        fam = {"orig": Phrasing(q, "mindcube", omap),
               "t1": Phrasing(f"From the viewpoint of image {img_k}, I turn {turn} by 90 degrees and walk straight ahead for about a metre. How does my distance to the {tgt_b} change?", "t_paren", omap),
               "t2": Phrasing(f"Start at image {img_k}'s camera, turn {turn}, walk forward. What happens to the {tgt_b} relative to me?", "t_dot", omap),
               "t3": Phrasing(f"Imagine standing as in image {img_k}, turning {turn} and moving ahead one step. Where does the {tgt_b} end up?", "t_colon", omap),
               "h1": Phrasing(f"After a 90-degree {turn} turn from image {img_k}'s viewpoint and a short walk forward, describe the {tgt_b}.", "h_inline", omap)}
        imgs = [g.bank.save(v) for _, v in order]
        g.emit("M8", [st, "MC-dynamics-sequential", "MC-relation-agent-object"] + TAGS_AROUND, "MindCube", "multi_image",
               [v for _, v in order], imgs, fam, {}, keys, ans, "mc",
               gt(defin + " q3 答案：从该视图的相机位置原地转 90° 再直走 1 m，距离减少 >10% 为更近，增加 >8% 为更远，否则按转后方位答'距离差不多、在某侧'。",
                  diff + " 原题未规定'直走'的距离，这里取 1 m。", d0=round(d0, 3), d1=round(d1, 3), side=s, occlusion=occ_vals),
               objs, {})
        r.take(st)
        return None

    return realize


# =============================================================================
# M9: among -- four views around a central object (MindCube Among)
# =============================================================================
AMONG_SEVERE_OCCLUSION = 0.6


def m9_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = [o for o in scene_pool(ctx) if ctx.obb[o].longest_edge >= 0.3]
    centers = [o for o in simple_named_pool(ctx) if ctx.obb[o].longest_edge >= 0.3]
    g.rng.shuffle(centers)
    n = 0
    for c in centers:
        C = ctx.obb[c].center
        around = []
        for o in pool:
            if o == c:
                continue
            v = ctx.obb[o].center[:2] - C[:2]
            d = float(np.linalg.norm(v))
            if 1.0 <= d <= 5.0:
                around.append((math.degrees(math.atan2(v[1], v[0])) % 360.0, d, o))
        if len(around) < 4:
            continue
        found = None
        for th0 in range(0, 90, 10):
            pick = []
            for k in range(4):
                tgt = (th0 + 90 * k) % 360
                cand = [(angle_diff_deg(a, tgt), d, o) for a, d, o in around if angle_diff_deg(a, tgt) <= 35]
                if not cand:
                    break
                pick.append(min(cand)[2])
            if len(pick) == 4 and len(set(pick)) == 4 and g.distinct_names(pick + [c]):
                found = (th0, pick)
                break
        if found is None:
            continue
        th0, S = found
        # camera k between C and S_k, looking at C; image order front, left, back, right = S angles th0, th0-90, th0+180, th0+90
        order = [0, 3, 2, 1]   # indices into S (angles th0 + 90k): front=0, left=th0-90 (k=3), back=2, right=1
        cams = []
        for k in order:
            v = ctx.obb[S[k]].center[:2] - C[:2]
            u = v / np.linalg.norm(v)
            dmax = float(np.linalg.norm(v)) - 0.5
            cam = None
            bg = S[(k + 2) % 4]
            # around the centre in direction k (offset up to 24 deg so the background is not hidden right
            # behind the centre), nearer to the centre than S_k so S_k stays behind the camera
            for off in (0.0, 12.0, -12.0, 24.0, -24.0, 35.0, -35.0):
                for dist in (2.4, 2.0, 1.7, 1.4, 1.1, 2.8, 3.2):
                    e = C[:2] + dist * mv.rotate2(u, off)
                    if not ctx.legal(e[None])[0]:
                        continue
                    cam_t = look_at((e[0], e[1], CAMERA_HEIGHT_M), C)
                    b_ang = signed(mv.bearing(cam_t, ctx.obb[S[k]].center)[0])
                    if abs(b_ang) < 120.0:
                        continue   # S_k must stay behind the camera (the "what is behind me" answer)
                    if (framed_loose(ctx.obb[c], cam_t, K, SIZE) and ctx.face_visible(c, cam_t[None], K, SIZE)[0]
                            and ctx.face_seen_any(bg, cam_t[None], K, SIZE)[0]
                            and not ctx.face_seen_any(S[k], cam_t[None], K, SIZE)[0]):
                        cam = cam_t
                        break
                if cam is not None:
                    break
            if cam is None:
                break
            cams.append(cam)
        if len(cams) < 4:
            continue
        surr = [S[k] for k in order]          # surrounding object behind each camera, in image order
        yield Combo("M9", "among", (c,) + tuple(surr), [cams], {"subtasks": SUBTASKS["M9"]})
        n += 1
        if n >= g.cap(30):
            return


def m9_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        c = combo.objects[0]
        surr = list(combo.objects[1:])     # surr[i] stands behind camera i
        cams = combo.shots[0]
        vv = [g.bank.view(cm, "M9") for cm in cams]
        severe = 0
        for i, (vid, va) in enumerate(vv):
            if mv.vis_class(va, c) != "rec":
                return "center_not_recognisable"
            if mv.vis_class(va, surr[i], loose_absent=True) != "abs":
                return "object_behind_camera_visible"
            bg = va.obj(surr[(i + 2) % 4])
            if bg.absent():
                return "background_object_not_visible"
            if bg.occ_px / max(bg.sil_px, 1) >= AMONG_SEVERE_OCCLUSION:
                severe += 1
        if severe >= 3:
            return "three_or_more_backgrounds_severely_occluded"
        if not emit:
            return None
        def emit_st(st: str) -> Optional[str]:
            nc, ncb = named_the(g, c)
            sn = [named_the(g, s) for s in surr]
            header = (f"Based on these four images (image 1, 2, 3, and 4) showing the {ncb} from different viewpoints (front, left, back, and right), "
                      "with each camera aligned with room walls and partially capturing the surroundings: ")
            objs = [g.obj_entry(c, nc, "center")] + [g.obj_entry(s, sn[i][0], f"behind_camera_{i + 1}") for i, s in enumerate(surr)]
            defin = ("中心物体四周大致成直角分布 4 个物体（与直角的偏差 ≤35°）；第 i 台相机在第 i 个周围物体那一侧（偏开 ≤35°）、朝向中心物体，该周围物体在相机身后（方位角 ≥120° 且图中完全看不到），对面的周围物体在中心物体后方露出；"
                     "按 MindCube 附录 B，四张图里背景物体被严重遮挡（被盖住 ≥60%）的不超过两张。")
            diff = "MindCube 原题由人选景标注；这里按几何构造、渲染深度核验。"
            keys_all = [cap(x[1]) for x in sn]
            imgs4 = [g.bank.save(v) for v, _ in vv]
            if st == "MC-among-q1":
                i = g.rng.randrange(4)
                j = g.rng.choice([(i + 1) % 4, (i + 3) % 4])
                lab = mv.movement_label(cams[i], cams[j])
                if lab is None:
                    return "q1_movement_ambiguous"
                keys = ["Directly left", "Directly right", "Diagonally forward and left", "Diagonally forward and right"]
                g.emit("M9", [st, "MC-relation-agent-agent", "MC-dynamics-simultaneous"] + TAGS_AMONG, "MindCube", "multi_image",
                       [vv[i][0], vv[j][0]], [imgs4[i], imgs4[j]], MC_MOVE, {}, keys, lab, "mc",
                       gt(defin + " q1：相机 2 位置在相机 1 水平坐标系里的方位。", diff), objs, {})
                r.take(st)
                return None
            k = g.rng.randrange(4)
            if st == "MC-among-q2-current":
                q = header + f"If I am standing at the same spot and facing the same direction as shown in image {k + 1}, what is behind me?"
                fam = {"orig": Phrasing(q, "mindcube"),
                       "t1": Phrasing(f"The four photos circle the {ncb}. Standing where image {k + 1} was taken and facing the same way, what is directly behind me?", "t_paren"),
                       "t2": Phrasing(f"From image {k + 1}'s camera position, which object is at my back?", "t_dot"),
                       "t3": Phrasing(f"Which object would I see if I turned around at the spot of image {k + 1}?", "t_colon"),
                       "h1": Phrasing(f"Behind the photographer of image {k + 1}, which of these objects stands?", "h_inline")}
                g.emit("M9", [st, "MC-perspective-self", "MC-relation-agent-object"] + TAGS_AMONG, "MindCube", "multi_image",
                       [v for v, _ in vv], imgs4, fam, {}, keys_all, cap(sn[k][1]), "mc",
                       gt(defin + " q2-current：第 k 台相机身后的周围物体。", diff), objs, {})
                r.take(st)
                return None
            if st == "MC-among-q2-sequence":
                turn = g.rng.choice(["left", "right"])
                cands = [i for i in range(4) if i != k]
                g.rng.shuffle(cands)
                for i in cands:
                    d0, d1 = mv.turn_move_distance(cams[k], turn, ctx.obb[surr[i]].center, step=1.0)
                    if d1 < d0 - 0.1:
                        truth = "Yes"
                    elif d1 > d0 + 0.02:
                        truth = "No"
                    else:
                        continue
                    q = header + f"If I am standing at the same spot and facing the same direction as shown in image {k + 1}, then I turn {turn} and move forward, will I get closer to the {sn[i][1]}?"
                    fam = {"orig": Phrasing(q, "mindcube"),
                           "t1": Phrasing(f"From image {k + 1}'s viewpoint I turn {turn} and walk forward about a metre. Do I get closer to the {sn[i][1]}?", "t_paren"),
                           "t2": Phrasing(f"Turning {turn} at image {k + 1}'s spot and stepping ahead: am I approaching the {sn[i][1]}?", "t_dot"),
                           "t3": Phrasing(f"Standing as in image {k + 1}, I make a {turn} turn and go straight. Will the {sn[i][1]} be nearer?", "t_colon"),
                           "h1": Phrasing(f"A {turn} turn and a short walk from image {k + 1}'s camera: does the distance to the {sn[i][1]} shrink?", "h_inline")}
                    g.emit("M9", [st, "MC-dynamics-sequential", "MC-relation-agent-object"] + TAGS_AMONG, "MindCube", "multi_image",
                           [v for v, _ in vv], imgs4, fam, {}, YESNO_MC, truth, "yesno",
                           gt(defin + " q2-sequence：原地转 90° 后直走 1 m，距离减少 >0.1 m 为'是'，增加为'否'，其余不出题。",
                              diff + " 原题未规定前进距离，这里取 1 m。", d0=round(d0, 3), d1=round(d1, 3)), objs, {},
                           keep_orig_order=YESNO_MC)
                    r.take(st)
                    return None
                return "q2_sequence_ambiguous"
            # q5: what is to the left / right of the central object in image k
            side = g.rng.choice(["left", "right"])
            cand = []
            for i in range(4):
                if i in (k, (k + 2) % 4):
                    continue
                ang, _ = mv.bearing(cams[k], ctx.obb[surr[i]].center)
                a = ang if ang <= 180 else ang - 360
                if side == "left" and a < -15 or side == "right" and a > 15:
                    cand.append(i)
            if len(cand) != 1:
                return "q5_ambiguous"
            i = cand[0]
            q = header + f"From the viewpoint presented in image {k + 1}, what is to the {side} of the {ncb}?"
            fam = {"orig": Phrasing(q, "mindcube"),
                   "t1": Phrasing(f"Seen from where image {k + 1} was taken, which object is on the {side} side of the {ncb}?", "t_paren"),
                   "t2": Phrasing(f"In image {k + 1}'s viewpoint, what stands to the {side} of the {ncb} (possibly outside the frame)?", "t_dot"),
                   "t3": Phrasing(f"Facing the {ncb} as in image {k + 1}, name the object on its {side}.", "t_colon"),
                   "h1": Phrasing(f"Looking at the {ncb} from image {k + 1}'s camera, which of these lies off to the {side}?", "h_inline")}
            g.emit("M9", [st, "MC-relation-object-object", "MC-perspective-self"] + TAGS_AMONG, "MindCube", "multi_image",
                   [v for v, _ in vv], imgs4, fam, {}, keys_all, cap(sn[i][1]), "mc",
                   gt(defin + " q5：从第 k 台相机看，中心物体左（右）侧的周围物体（方位偏离中心 >15°）。", diff), objs, {})
            r.take(st)
            return None

        # one verified set of four views carries one question per subtask still needed (MindCube asks several
        # questions about the same four images)
        todo = [s_ for s_ in SUBTASKS["M9"] if r.need[s_] > 0]
        if not todo:
            return None
        outcomes = [emit_st(s_) for s_ in todo]
        return None if any(o is None for o in outcomes) else outcomes[0]

    return realize


# =============================================================================
# M10: translation (MindCube translation setting)
# =============================================================================
TR_REL = {"left": "left", "right": "right", "front": "in front", "behind": "behind", "above": "above", "below": "below"}


def m10_combos(g: GenCtx) -> Iterator[Combo]:
    ctx = g.ctx
    pool = [o for o in simple_named_pool(ctx) if ctx.obb[o].longest_edge >= 0.3]
    pairs = [(a, b) for a, b in itertools.permutations(pool, 2)
             if 0.6 <= float(np.linalg.norm(ctx.obb[a].center[:2] - ctx.obb[b].center[:2])) <= 3.0]
    g.rng.shuffle(pairs)
    n = 0
    lim = g.cap(150, census=300)
    g.count("M10", "pairs_total", len(pairs))
    g.count("M10", "pairs_examined", min(len(pairs), lim))
    for a, b in pairs[:lim]:
        cams1 = frame_candidates(ctx, [a, b], max_out=2)
        for c1 in cams1:
            for s in (1.0, 1.4, 1.8, 2.3, 2.9, 3.5, -1.0, -1.4, -1.8, -2.3, -2.9, -3.5):
                c2 = mv.move(c1, right=s)
                if not ctx.legal(c2[:2, 3][None])[0]:
                    continue
                if not ctx.face_visible(b, c2[None], K, SIZE)[0] or ctx.face_seen_any(a, c2[None], K, SIZE)[0]:
                    continue
                cs = [o for o in pool if o not in (a, b) and not ctx.face_seen_any(o, c1[None], K, SIZE)[0]
                      and framed_loose(ctx.obb[o], c2, K, SIZE) and ctx.face_visible(o, c2[None], K, SIZE)[0]]
                if not cs:
                    continue
                yield Combo("M10", "translation", (a, b, g.rng.choice(cs)), [[c1, c2]],
                            {"subtasks": ["MC-translation-transitive"], "shift_m": s})
                n += 1
                break
            else:
                continue
            break
        if n >= g.cap(30):
            return


def tr_relation(c2w: np.ndarray, p: np.ndarray, q: np.ndarray) -> Optional[str]:
    """Relation of p to q in the (shared) camera orientation: dominant axis by a factor 2."""
    _, f, r = mv.frame_h(c2w)
    v = np.asarray(p, float) - np.asarray(q, float)
    x, z, y = float(v[:2] @ r), float(v[:2] @ f), float(v[2])
    comps = sorted([(abs(x), "x"), (abs(z), "z"), (abs(y), "y")], reverse=True)
    if comps[0][0] < 0.3 or comps[0][0] < 2.0 * comps[1][0]:
        return None
    ax = comps[0][1]
    if ax == "x":
        return "right" if x > 0 else "left"
    if ax == "z":
        return "behind" if z > 0 else "front"
    return "above" if y > 0 else "below"


def m10_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        a, b, c = combo.objects
        c1, c2 = combo.shots[0]
        v1, a1 = g.bank.view(c1, "M10")
        v2, a2 = g.bank.view(c2, "M10")
        cls = {o: (mv.vis_class(a1, o, loose_absent=True), mv.vis_class(a2, o, loose_absent=True)) for o in (a, b, c)}
        if cls[a] != ("rec", "abs") or cls[b] != ("rec", "rec") or cls[c] != ("abs", "rec"):
            return "translation_visibility_pattern"
        r1 = tr_relation(c1, ctx.obb[a].center, ctx.obb[b].center)
        r2 = tr_relation(c1, ctx.obb[b].center, ctx.obb[c].center)
        rel = tr_relation(c1, ctx.obb[a].center, ctx.obb[c].center)
        if rel is None or r1 is None or r2 is None:
            return "relation_not_dominant"
        if not emit:
            return None
        if not r.emit_ok("MC-translation-transitive"):
            return None
        inverse = g.rng.random() < 0.5
        ask_rev = g.rng.random() < 0.5
        na, nab = named_the(g, a)
        nc, ncb = named_the(g, c)
        X, Y = (ncb, nab) if ask_rev else (nab, ncb)
        inv = {"left": "right", "right": "left", "front": "behind", "behind": "front", "above": "below", "below": "above"}
        ans = inv[rel] if ask_rev else rel
        keys = [ans] + g.rng.sample([k for k in TR_REL if k != ans], 3)
        omap = {k: v for k, v in TR_REL.items()}
        q = f"Which direction is the {X} relative to the {Y}?"
        fam = {"orig": Phrasing("Based on these two views of the same scene: " + q, "mindcube", omap),
               "t1": Phrasing(f"The camera slid sideways between the two photos without turning. Using the camera's view, where is the {X} relative to the {Y}?", "t_paren", omap),
               "t2": Phrasing(f"Both images face the same way. Combine them: the {X} is ___ the {Y}.", "t_dot", omap),
               "t3": Phrasing(f"From the shared viewing direction of both photos, how is the {X} placed with respect to the {Y}?", "t_colon", omap),
               "h1": Phrasing(f"Link the two shots through the object they share. Relative to the {Y}, the {X} lies:", "h_inline", omap)}
        vids = [v2, v1] if inverse else [v1, v2]
        imgs = [g.bank.save(v) for v in vids]
        g.emit("M10", ["MC-translation-transitive", "MC-dynamics-translation"], "MindCube", "multi_image", vids, imgs, fam,
               {}, keys, ans, "mc",
               gt("两台相机朝向相同、只平移；A 只在一张图、C 只在另一张图、B 两张都有。答 A（或 C）相对 C（或 A）在共同视角下的关系：左右、前后（前 = 离相机更近）、上下中最大的分量，且至少是第二大的 2 倍。",
                  "MindCube 的平移设定未随数据发布（官方汇总也不计入总分）；按其辅助代码的题型构造。",
                  rel_a_b=r1, rel_b_c=r2, rel_a_c=rel, inverse_order=inverse, shift_m=combo.info["shift_m"]),
               [g.obj_entry(a, na, "A_only_in_one_image"), g.obj_entry(b, named_the(g, b)[0], "anchor_in_both"),
                g.obj_entry(c, nc, "C_only_in_other_image")], {})
        r.take("MC-translation-transitive")
        return None

    return realize


GROUPS = {
    "M1": (m1_combos, m1_realize), "M2": (m2_combos, m2_realize), "M3": (m3_combos, m3_realize),
    "M4": (m4_combos, m4_realize), "M5": (m5_combos, m5_realize), "M6": (m6_combos, m6_realize),
    "M7": (m7_combos, m7_realize), "M8": (m8_combos, m8_realize), "M9": (m9_combos, m9_realize),
    "M10": (m10_combos, m10_realize), "M11": (m11_combos, m11_realize),
}
