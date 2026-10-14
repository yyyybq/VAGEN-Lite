"""Video questions V1-V6 (VSI-Bench's eight tasks) and SITE-C06's static navigation part.

A video is a room tour like a person scanning the room with a phone: the camera walks along the free space
between a few viewpoints chosen to cover the room's main objects (greedy coverage with v10's visibility
prefilter), pans left and right while walking and sweeps around at each viewpoint, at eye height with a
slight downward pitch.  The model sees N_FRAMES frames sampled uniformly in time; 32 is the frame count of
VSI-Bench's official evaluation script for open models (thinking-in-space evaluate_all_in_one.sh,
``num_frames=32``).  Every frame stores its pose and the visible pixels of every object of the room.
Several questions share one video.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import mv
from .common import Combo, GenCtx, bare, cap, single_pool
from .geom import CAMERA_HEIGHT_M, angle_diff_deg, box_distance, camera_K, point_in_polygon, polygon_area, pose_yaw_pitch
from .qa import Phrasing
from .single import Runner, pick_subtask
from .visible import RECOGNISABLE_AREA

SIZE = 512
K = camera_K(SIZE)
N_FRAMES = 32
PITCH_DEG = -15.0
WALK_MPS = 0.5
PAN_DPS = 45.0

SUBTASKS = {
    "V1": ["VSI-count"],
    "V2": ["VSI-distance-absolute", "VSI-distance-relative"],
    "V3": ["VSI-size-object", "VSI-size-room"],
    "V4": ["VSI-direction-easy", "VSI-direction-medium", "VSI-direction-hard"],
    "V5": ["VSI-route"],
    "V6": ["VSI-order"],
}


def gt(definition: str, diff: str, **values) -> Dict[str, Any]:
    return {"definition": definition, "differs_from_original": diff, "values": values}


# =============================================================================
# Trajectory
# =============================================================================
def grid_path(ctx, a: np.ndarray, b: np.ndarray, pts: np.ndarray, spacing: float) -> Optional[List[np.ndarray]]:
    """Shortest 8-connected path on the legal grid ``pts`` from the node nearest a to the node nearest b,
    every step a legal transition (v10 collision rule)."""
    key = {tuple(np.round(p / spacing).astype(int)): i for i, p in enumerate(pts)}
    ia = int(np.argmin(np.linalg.norm(pts - a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(pts - b, axis=1)))
    dist = {ia: 0.0}
    prev = {}
    pq = [(0.0, ia)]
    steps = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    while pq:
        d, i = heapq.heappop(pq)
        if i == ib:
            break
        if d > dist.get(i, 1e18):
            continue
        k = tuple(np.round(pts[i] / spacing).astype(int))
        nbrs = [key.get((k[0] + dx, k[1] + dy)) for dx, dy in steps]
        nbrs = [j for j in nbrs if j is not None]
        if not nbrs:
            continue
        ok = ctx.legal_transitions(np.repeat(pts[i][None], len(nbrs), 0), pts[nbrs])
        for j, good in zip(nbrs, ok):
            if not good:
                continue
            nd = d + float(np.linalg.norm(pts[j] - pts[i]))
            if nd < dist.get(j, 1e18):
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd, j))
    if ib not in dist:
        return None
    path = [ib]
    while path[-1] != ia:
        path.append(prev[path[-1]])
    return [pts[i] for i in reversed(path)]


def room_tour(g: GenCtx, room: int, main: Sequence[str]) -> Optional[Dict[str, Any]]:
    ctx = g.ctx
    poly = ctx.rooms[room].polygon
    pts = ctx.legal_grid(0.25)
    pts = pts[point_in_polygon(pts, poly)] if len(pts) else pts
    if len(pts) < 8:
        return None
    # candidate viewpoints: every other grid node x 8 headings
    cand = pts[::3]
    yaws = np.radians(np.arange(0, 360, 45))
    poses = np.array([pose_yaw_pitch((p[0], p[1], CAMERA_HEIGHT_M), y, math.radians(PITCH_DEG)) for p in cand for y in yaws])
    if not len(poses):
        return None
    cover = np.zeros((len(poses), len(main)), bool)
    for j, o in enumerate(main):
        cover[:, j] = ctx.face_visible(o, poses, K, SIZE)
    chosen, covered = [], np.zeros(len(main), bool)
    while not covered.all() and len(chosen) < 5:
        gain = (cover & ~covered).sum(1)
        i = int(np.argmax(gain))
        if gain[i] == 0:
            break
        chosen.append(i)
        covered |= cover[i]
    if len(chosen) < 2 or covered.sum() < max(3, int(0.6 * len(main))):
        return None
    # order viewpoints greedily by distance, starting from the one nearest the room's first corner
    vps = [(poses[i][:2, 3], math.atan2(poses[i][1, 2], poses[i][0, 2])) for i in chosen]
    start = int(np.argmin([np.linalg.norm(v[0] - poly[0]) for v in vps]))
    order = [start]
    while len(order) < len(vps):
        last = vps[order[-1]][0]
        rest = [k for k in range(len(vps)) if k not in order]
        order.append(min(rest, key=lambda k: float(np.linalg.norm(vps[k][0] - last))))
    vps = [vps[k] for k in order]
    # timed samples: sweep +-45 deg at each viewpoint, walk with a gentle pan in between
    samples: List[Tuple[float, np.ndarray, float]] = []   # (time, position, yaw)
    t = 0.0

    def add_sweep(p, yaw0):
        nonlocal t
        for dy in (-45.0, -22.5, 0.0, 22.5, 45.0, 22.5, 0.0):
            samples.append((t, p.copy(), yaw0 + math.radians(dy)))
            t += 22.5 / PAN_DPS
    for k, (p, yaw) in enumerate(vps):
        add_sweep(p, yaw)
        if k + 1 == len(vps):
            break
        path = grid_path(ctx, p, vps[k + 1][0], pts, 0.25)
        if path is None:
            return None
        nxt_yaw = vps[k + 1][1]
        L = sum(float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1))
        s = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            seg = float(np.linalg.norm(b - a))
            travel = math.atan2(b[1] - a[1], b[0] - a[0])
            for f in np.linspace(0, 1, 3, endpoint=False):
                q = a + f * (b - a)
                frac = (s + f * seg) / max(L, 1e-6)
                base = travel if 0.15 < frac < 0.85 else (yaw if frac <= 0.15 else nxt_yaw)
                pan = math.radians(30.0) * math.sin(2 * math.pi * (s + f * seg) / 2.5)
                samples.append((t, q, base + pan))
                t += seg / 3 / WALK_MPS
            s += seg
    if t <= 0 or len(samples) < N_FRAMES:
        return None
    times = np.linspace(0, samples[-1][0], N_FRAMES)
    frames = []
    j = 0
    for tt in times:
        while j + 1 < len(samples) and samples[j + 1][0] <= tt:
            j += 1
        _, p, y = samples[j]
        frames.append(pose_yaw_pitch((p[0], p[1], CAMERA_HEIGHT_M), y, math.radians(PITCH_DEG)))
    return {"room": room, "frames": frames, "viewpoints": [(v[0].tolist(), v[1]) for v in vps],
            "duration_s": round(samples[-1][0], 1), "main": list(main), "covered_prefilter": int(covered.sum())}


def tour_rooms(g: GenCtx) -> List[int]:
    ctx = g.ctx
    out = []
    for r in ctx.rooms:
        if ctx.room_area(r.index) < 8.0:
            continue
        main = [o for o in single_pool(ctx, 3.5) if ctx.room_of_obj.get(o) == r.index]
        if len(main) >= 4:
            out.append(r.index)
    g.rng.shuffle(out)
    return out


# =============================================================================
# One rendered video, shared by every group (cached per scene)
# =============================================================================
class Video:
    def __init__(self, g: GenCtx, tour: Dict[str, Any]) -> None:
        self.g = g
        self.tour = tour
        ctx = g.ctx
        self.room = tour["room"]
        self.vids, self.vas = [], []
        for c in tour["frames"]:
            vid, va = g.bank.view(c, "video")
            self.vids.append(vid)
            self.vas.append(va)
        # every object in or seen from the room: per-frame visible pixels
        cands = set()
        for va in self.vas:
            cands |= set(va.candidates_in_frustum([o for o in ctx.obb if ctx.by_id[o].label]))
        self.objs = sorted(cands)
        self.area = {o: np.array([va.obj(o).area for va in self.vas]) for o in self.objs}
        self.rec = {o: np.array([va.obj(o).recognisable(0.6) for va in self.vas]) for o in self.objs}
        self.saved = False

    def images(self) -> List[str]:
        return [self.g.bank.save(v) for v in self.vids]

    def seen(self, o: str) -> bool:
        return bool(self.rec.get(o, np.zeros(1, bool)).any())

    def visible_any(self, o: str) -> bool:
        return bool((self.area.get(o, np.zeros(1)) >= 0.0005).any())

    def unique_names(self) -> Dict[str, str]:
        """Objects of the room whose bare display name names no other object of the room nor any other object
        that shows in the video: VSI asks about categories that occur once."""
        ctx = self.g.ctx
        room_objs = [o.id for o in ctx.objects if ctx.room_of_obj.get(o.id) == self.room]
        out = {}
        for o in room_objs:
            if o not in ctx.eligible or o in ctx.duplicates or not self.seen(o):
                continue
            others = [x for x in ctx.same_name(o) if ctx.duplicates.get(x) != o]
            if any(ctx.room_of_obj.get(x) == self.room or self.visible_any(x) for x in others):
                continue
            out[o] = ctx.disp[o]
        return out

    def first_frame(self, o: str) -> Optional[int]:
        idx = np.flatnonzero(self.area.get(o, np.zeros(1)) >= RECOGNISABLE_AREA)
        return int(idx[0]) if len(idx) else None


def get_videos(g: GenCtx) -> List[Video]:
    cache = getattr(g, "_videos", None)
    if cache is None:
        cache = []
        for room in tour_rooms(g)[:g.cap(2)]:
            main = [o for o in single_pool(g.ctx, 3.5) if g.ctx.room_of_obj.get(o) == room]
            tour = room_tour(g, room, main)
            if tour is not None:
                cache.append(tour)
        g._videos = cache
        g._rendered = {}
    return cache


def rendered(g: GenCtx, k: int) -> Video:
    if k not in g._rendered:
        g._rendered[k] = Video(g, g._videos[k])
    return g._rendered[k]


def v_combos(group: str):
    def combos(g: GenCtx) -> Iterator[Combo]:
        for k, tour in enumerate(get_videos(g)):
            yield Combo(group, "room_tour", (str(k),), [tour["frames"]], {"subtasks": SUBTASKS[group], "video": k})
    return combos


VSI_STYLE = {"mc": "vsi_mc", "na": "vsi_na"}


def vsi_fam(q_orig: str, t: Sequence[str], h: str, numeric: bool, omap: Optional[Dict[str, str]] = None) -> Dict[str, Phrasing]:
    if numeric:
        return {"orig": Phrasing(q_orig, "vsi_na"), "t1": Phrasing(t[0], "num"), "t2": Phrasing(t[1], "num"),
                "t3": Phrasing(t[2], "num"), "h1": Phrasing(h, "num_h")}
    return {"orig": Phrasing(q_orig, "vsi_mc", omap), "t1": Phrasing(t[0], "t_paren", omap), "t2": Phrasing(t[1], "t_dot", omap),
            "t3": Phrasing(t[2], "t_colon", omap), "h1": Phrasing(h, "h_inline", omap)}


def emit_video(g: GenCtx, v: Video, group: str, tags: Sequence[str], fam, keys, ans, atype: str, gtrec, objs,
               extra=None, keep=None):
    imgs = v.images()
    return g.emit(group, list(tags), "VSI-Bench", "video", v.vids, imgs, fam, {}, keys, ans, atype, gtrec, objs,
                  {"frames": len(imgs), "room": v.room}, extra=dict(extra or {}, video_room=v.room,
                                                                    duration_s=v.tour["duration_s"]),
                  keep_orig_order=keep)


def num_answer(x: float, nd: int) -> str:
    return f"{x:.{nd}f}" if nd else str(int(round(x)))


# =============================================================================
# V1 count
# =============================================================================
def v1_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        v = rendered(g, combo.info["video"])
        groups: Dict[str, List[str]] = {}
        for o in ctx.objects:
            if ctx.room_of_obj.get(o.id) == v.room and o.id in ctx.eligible and o.id not in ctx.duplicates:
                groups.setdefault(ctx.disp[o.id], []).append(o.id)
        for disp, members in sorted(groups.items(), key=lambda kv: g.rng.random()):
            everyone = sorted({members[0]} | set(ctx.same_name(members[0])))
            everyone = [o for o in everyone if o not in ctx.duplicates]
            inside = [o for o in everyone if ctx.room_of_obj.get(o) == v.room]
            outside = [o for o in everyone if ctx.room_of_obj.get(o) != v.room]
            if len(inside) < 2:
                continue                       # VSI excludes counts of 1
            if not all(v.seen(o) for o in inside):
                continue                       # every instance clearly visible in some frame
            if any(v.visible_any(o) for o in outside):
                continue                       # no instance outside the room shows in the video
            if not emit:
                return None
            if not r.emit_ok("VSI-count"):
                return None
            q = f"How many {disp}(s) are in this room?"
            fam = vsi_fam(q, [f"Count the {disp}s in this room.", f"How many {disp}s does the room in the video contain?",
                              f"What is the number of {disp}s in this room?"],
                          f"Walking through the video, tally every {disp} in the room. Total?", True)
            emit_video(g, v, "V1", ["VSI-count", "SITE-C01"], fam, None, str(len(inside)), "numeric",
                       gt("房间内该类（按显示名，含其下位类）的实例数；每个实例至少在一帧里清楚可见（可见部分 ≥0.4% 画面且露出 ≥50%）；视频里看到的房间外同类实例为 0；按官方规则不出答案为 1 的题。",
                          "VSI 由人工核对的实例标注计数；这里按 InteriorGS 标注框计数。", count=len(inside), instances=inside),
                       [g.obj_entry(o, disp, "counted") for o in inside])
            r.take("VSI-count")
            return None
        return "no_countable_category"

    return realize


# =============================================================================
# V2 distances
# =============================================================================
def v2_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        v = rendered(g, combo.info["video"])
        names = v.unique_names()
        ids = list(names)
        if len(ids) < 2:
            return "fewer_than_2_unique_objects"
        if not emit:
            return None
        st = pick_subtask(r, SUBTASKS["V2"])
        if st is None:
            return None
        if st == "VSI-distance-absolute":
            pairs = list(itertools.combinations(ids, 2))
            g.rng.shuffle(pairs)
            for a, b in pairs:
                d = box_distance(ctx.obb[a], ctx.obb[b])
                if d < 0.3:
                    continue
                q = f"Measuring from the closest point of each object, what is the distance between the {names[a]} and the {names[b]} (in meters)?"
                fam = vsi_fam(q, [f"How far apart are the {names[a]} and the {names[b]} at their closest points, in metres?",
                                  f"What is the gap in metres between the {names[a]} and the {names[b]}?",
                                  f"Give the shortest distance between the {names[a]} and the {names[b]} in meters."],
                              f"Estimate, in metres, the space separating the {names[a]} from the {names[b]} where they come nearest.", True)
                emit_video(g, v, "V2", [st, "SITE-C04"], fam, None, num_answer(d, 1), "numeric",
                           gt("两个三维框之间的精确最近距离（框内采样取最小点对距离的极限），保留 1 位小数；只用房间里、且视频里只有一个的物体类别。",
                              "VSI 在框内均匀采样取最小点对距离，这里取精确值（差异只在采样误差）；框来自 InteriorGS 标注。", distance_m=round(d, 3)),
                           [g.obj_entry(a, "the " + names[a], "object"), g.obj_entry(b, "the " + names[b], "object")])
                r.take(st)
                return None
            return "no_distance_pair"
        thr = 0.30 if ctx.room_area(v.room) > 40 else 0.15
        g.rng.shuffle(ids)
        for x in ids:
            others = [o for o in ids if o != x]
            for cand in itertools.combinations(others, 4):
                ds = [box_distance(ctx.obb[x], ctx.obb[c]) for c in cand]
                if min(abs(p - q) for p, q in itertools.combinations(ds, 2)) < thr:
                    continue
                keys = [names[c] for c in cand]
                ans = names[cand[int(np.argmin(ds))]]
                q = f"Measuring from the closest point of each object, which of these objects ({', '.join(keys)}) is the closest to the {names[x]}?"
                fam = vsi_fam(q, [f"Which object is nearest to the {names[x]}?", f"Of the listed objects, which one is closest to the {names[x]} (closest points)?",
                                  f"Pick the object with the smallest distance to the {names[x]}."],
                              f"Find the object that sits nearest the {names[x]}.", False)
                emit_video(g, v, "V2", [st, "SITE-C04"], fam, keys, ans, "mc",
                           gt(f"候选物体到参照物的三维框最近距离；按官方规则候选距离两两相差 ≥{int(thr * 100)} cm（房间 >40 m² 用 30 cm，否则 15 cm）。",
                              "定义与 VSI 一致（框级距离）；框来自 InteriorGS 标注。", distances_m=dict(zip(keys, [round(d, 3) for d in ds]))),
                           [g.obj_entry(x, "the " + names[x], "reference")] + [g.obj_entry(c, "the " + names[c], "candidate") for c in cand])
                r.take(st)
                return None
        return "no_relative_distance_set"

    return realize


# =============================================================================
# V3 sizes
# =============================================================================
def v3_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        v = rendered(g, combo.info["video"])
        names = v.unique_names()
        if not emit:
            return None
        st = pick_subtask(r, SUBTASKS["V3"])
        if st is None:
            return None
        if st == "VSI-size-object":
            if not names:
                return "no_unique_object"
            o = g.rng.choice(list(names))
            cm = ctx.obb[o].longest_edge * 100.0
            q = f"What is the length of the longest dimension (length, width, or height) of the {names[o]}, measured in centimeters?"
            fam = vsi_fam(q, [f"How long is the longest side of the {names[o]}, in centimetres?",
                              f"Give the largest dimension of the {names[o]} in cm.",
                              f"In centimeters, what is the {names[o]}'s biggest extent (length, width or height)?"],
                          f"Estimate the {names[o]}'s greatest dimension in centimetres.", True)
            emit_video(g, v, "V3", [st, "SITE-C04"], fam, None, num_answer(cm, 0), "numeric",
                       gt("物体三维框（按原始 8 角点恢复的带旋转框）的最长边，单位厘米。", "与 VSI 定义一致（最长边）；VSI 的框来自其点云标注。", longest_edge_cm=round(cm, 1)),
                       [g.obj_entry(o, "the " + names[o], "object")])
            r.take(st)
            return None
        area = ctx.room_area(v.room)
        q = "What is the size of this room (in square meters)? \nIf multiple rooms are shown, estimate the size of the combined space."
        fam = vsi_fam(q, ["How many square metres is this room?", "Estimate the floor area of the room in the video (m^2).",
                          "What is the area of this room in square meters?"],
                      "Judging from the walk-through, what floor area does the room cover, in square metres?", True)
        emit_video(g, v, "V3", [st, "SITE-C04"], fam, None, num_answer(area, 1), "numeric",
                   gt("房间轮廓（structure.json 的 profile）的多边形面积，单位平方米；视频只在这一个房间里走。",
                      "VSI 用场景点云的 alpha-shape 估计面积；这里用房间轮廓，定义不同（通常更接近建筑面积）。", area_m2=round(area, 2)),
                   [])
        r.take(st)
        return None

    return realize


# =============================================================================
# V4 relative direction (easy / medium / hard)
# =============================================================================
def v4_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        v = rendered(g, combo.info["video"])
        names = v.unique_names()
        ids = list(names)
        if len(ids) < 3:
            return "fewer_than_3_unique_objects"
        if not emit:
            return None
        st = pick_subtask(r, SUBTASKS["V4"])
        if st is None:
            return None
        trips = list(itertools.permutations(ids, 3))
        g.rng.shuffle(trips)
        for p, f, z in trips:
            P, F, Z = (ctx.obb[o].center for o in (p, f, z))
            if min(np.linalg.norm(P[:2] - F[:2]), np.linalg.norm(P[:2] - Z[:2])) < 0.8:
                continue
            ang = mv.bearing_from(P, F, Z)
            np_, nf, nz = names[p], names[f], names[z]
            if st == "VSI-direction-easy":
                if not (30 <= ang <= 150 or 210 <= ang <= 330):
                    continue
                ans = "right" if ang < 180 else "left"
                q = f"If I am standing by the {np_} and facing the {nf}, is the {nz} to the left or the right of the {nf}?"
                keys = ["left", "right"]
                desc = "以'站在 P、面向 F'为前方，Z 的方位角在 30°–150° 为右、210°–330° 为左，其余（接近前后）不出题。"
            elif st == "VSI-direction-medium":
                if angle_diff_deg(ang, 0) < 20 or min(angle_diff_deg(ang, 135), angle_diff_deg(ang, 225)) < 10:
                    continue
                ans = "back" if 135 <= ang <= 225 else ("right" if ang < 180 else "left")
                q = (f"If I am standing by the {np_} and facing the {nf}, is the {nz} to my left, right, or back?\n"
                     "An object is to my back if I would have to turn at least 135 degrees in order to face it.")
                keys = ["left", "right", "back"]
                desc = "需要转身至少 135° 才能面向的为'后'（官方定义）；离 135°/225° 分界 <10° 或离正前方 <20° 不出题。"
            else:
                ans = mv.quadrant(ang, 10.0)
                if ans is None:
                    continue
                q = (f"If I am standing by the {np_} and facing the {nf}, is the {nz} to my front-left, front-right, back-left, or back-right?\n"
                     "The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).")
                keys = ["front-left", "front-right", "back-left", "back-right"]
                desc = "以 P 为原点、P→F 为 +y 轴的四个象限；离坐标轴 <10° 不出题。"
            t = [f"Imagine standing at the {np_}, looking toward the {nf}. Where is the {nz}?",
                 f"Facing the {nf} from the {np_}, in which direction lies the {nz}?",
                 f"From the {np_}, with the {nf} straight ahead, locate the {nz}."]
            h = f"Put yourself next to the {np_} with your eyes on the {nf}. The {nz} is"
            fam = vsi_fam(q, t, h, False)
            emit_video(g, v, "V4", [st, "SITE-C02"], fam, keys, ans, "mc",
                       gt(desc + " 位置用三维框中心。", "VSI 原题对三档也做角度边界过滤，具体阈值未公开；这里的边界阈值为本项目自定。",
                          bearing_deg=round(ang, 1)),
                       [g.obj_entry(p, "the " + np_, "stand_by"), g.obj_entry(f, "the " + nf, "facing"),
                        g.obj_entry(z, "the " + nz, "query")])
            r.take(st)
            return None
        return "no_direction_triple"

    return realize


# =============================================================================
# V5 route (VSI route planning; also SITE-C06 static navigation)
# =============================================================================
def walkable(ctx, a: np.ndarray, b: np.ndarray) -> bool:
    n = max(2, int(np.linalg.norm(b - a) / 0.15))
    pts = np.stack([a + (b - a) * t for t in np.linspace(0, 1, n)])
    return bool(ctx.legal(pts).all())


def near_spot(ctx, o: str, pts: np.ndarray) -> Optional[np.ndarray]:
    c = ctx.obb[o].center[:2]
    d = np.linalg.norm(pts - c, axis=1)
    ok = np.flatnonzero((d >= 0.5) & (d <= 1.4))
    if not len(ok):
        return None
    return pts[ok[int(np.argmin(d[ok]))]]


def turn_name(h1: float, h2: float) -> Optional[str]:
    d = (math.degrees(h2 - h1) + 540) % 360 - 180   # positive = left
    if abs(d) >= 150:
        return "turn back"
    if 45 <= d <= 135:
        return "turn left"
    if -135 <= d <= -45:
        return "turn right"
    return None


def v5_realize(r: Runner):
    g = r.g
    ctx = g.ctx

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        v = rendered(g, combo.info["video"])
        names = v.unique_names()
        ids = list(names)
        if len(ids) < 4:
            return "fewer_than_4_unique_objects"
        pts = ctx.legal_grid(0.25)
        spots = {o: near_spot(ctx, o, pts) for o in ids}
        ids = [o for o in ids if spots[o] is not None]
        perms = list(itertools.permutations(ids, 4))
        g.rng.shuffle(perms)
        for start, w1, w2, dest in perms[:4000]:
            path = [spots[start], spots[w1], spots[w2], spots[dest]]
            if any(np.linalg.norm(path[i + 1] - path[i]) < 0.8 for i in range(3)):
                continue
            if not all(walkable(ctx, path[i], path[i + 1]) for i in range(3)):
                continue
            hd = [math.atan2(*(path[i + 1] - path[i])[::-1]) for i in range(3)]
            t1, t2 = turn_name(hd[0], hd[1]), turn_name(hd[1], hd[2])
            if t1 is None or t2 is None:
                continue
            # facing object: one whose direction from the start matches the first leg within 25 degrees
            facing = [f for f in ids if f not in (start, w1, w2, dest) and
                      angle_diff_deg(math.degrees(math.atan2(*(ctx.obb[f].center[:2] - path[0])[::-1])), math.degrees(hd[0])) <= 25]
            if not facing:
                continue
            f = facing[0]
            if not emit:
                return None
            if not r.emit_ok("VSI-route"):
                return None
            ans = f"{t1.title()}, {t2.title()}"
            allc = [f"{a.title()}, {b.title()}" for a in ("turn back", "turn left", "turn right") for b in ("turn back", "turn left", "turn right")]
            keys = [ans] + g.rng.sample([c for c in allc if c != ans], 3)
            q = (f"You are a robot beginning at the {names[start]} facing the {names[f]}. You want to navigate to the {names[dest]}. "
                 "You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): "
                 f"1. Go forward until the {names[w1]} 2. [please fill in] 3. Go forward until the {names[w2]} 4. [please fill in] "
                 f"5. Go forward until the {names[dest]}. You have reached the final destination.")
            t = [f"Starting at the {names[start]} and facing the {names[f]}, I walk to the {names[w1]}, then to the {names[w2]}, then to the {names[dest]}. Which two turns do I make?",
                 f"Route from the {names[start]} (facing the {names[f]}): forward to the {names[w1]}, turn, forward to the {names[w2]}, turn, forward to the {names[dest]}. Fill in the turns.",
                 f"A robot at the {names[start]} faces the {names[f]}. It goes straight to the {names[w1]}, turns, goes to the {names[w2]}, turns, and reaches the {names[dest]}. Which turns?"]
            h = f"Plan the walk {names[start]} -> {names[w1]} -> {names[w2]} -> {names[dest]} (initially facing the {names[f]}). The two turns are"
            fam = vsi_fam(q, t, h, False)
            emit_video(g, v, "V5", ["VSI-route", "SITE-C06"], fam, keys, ans, "mc",
                       gt("在可通行区域里取各物体旁的落脚点，相邻落脚点之间直线可走（每 0.15 m 检查一次碰撞）；转向按相邻两段行进方向的夹角：左转 45°–135°、右转、≥150° 为掉头；起始朝向物体与第一段方向相差 ≤25°。所有地标都是视频里清楚出现过、且在房间里名字唯一的物体。",
                          "VSI 的路线由人设计和核验；这里由可通行区域几何生成。", turns=[t1, t2], heading_deg=[round(math.degrees(h), 1) for h in hd]),
                       [g.obj_entry(o, "the " + names[o], role) for o, role in
                        ((start, "start"), (f, "facing"), (w1, "waypoint_1"), (w2, "waypoint_2"), (dest, "destination"))])
            r.take("VSI-route")
            return None
        return "no_route"

    return realize


# =============================================================================
# V6 appearance order
# =============================================================================
MIN_GAP_FRAMES = 3


def v6_realize(r: Runner):
    g = r.g

    def realize(combo: Combo, emit: bool) -> Optional[str]:
        v = rendered(g, combo.info["video"])
        names = v.unique_names()
        first = {o: v.first_frame(o) for o in names}
        first = {o: f for o, f in first.items() if f is not None}
        ids = sorted(first, key=lambda o: first[o])
        for combo4 in itertools.combinations(ids, 4):
            fr = [first[o] for o in combo4]
            if min(b - a for a, b in zip(fr, fr[1:])) < MIN_GAP_FRAMES:
                continue
            if not emit:
                return None
            if not r.emit_ok("VSI-order"):
                return None
            truth = ", ".join(names[o] for o in combo4)
            shown = list(combo4)
            g.rng.shuffle(shown)
            perms = [", ".join(names[o] for o in p) for p in itertools.permutations(combo4)]
            keys = [truth] + g.rng.sample([p for p in perms if p != truth], 3)
            q = f"What will be the first-time appearance order of the following categories in the video: {', '.join(names[o] for o in shown)}?"
            fam = vsi_fam(q, [f"In which order do these objects first show up in the video: {', '.join(names[o] for o in shown)}?",
                              f"Sort by first appearance in the video: {', '.join(names[o] for o in shown)}.",
                              f"Which sequence lists {', '.join(names[o] for o in shown)} in the order they first appear?"],
                          f"Watching from the start, the objects {', '.join(names[o] for o in shown)} appear for the first time in the order", False)
            emit_video(g, v, "V6", ["VSI-order"], fam, keys, truth, "mc",
                       gt(f"首次出现 = 物体可见部分第一次 ≥0.4% 画面的那一帧（共 {N_FRAMES} 帧）；相邻两个物体的首次出现至少隔 {MIN_GAP_FRAMES} 帧；只用视频里唯一的类别。",
                          "VSI 用对象像素数阈值在原始视频上确定首次出现；这里在模型看到的 32 帧上按深度核验的可见部分确定。",
                          first_frames={names[o]: first[o] for o in combo4}),
                       [g.obj_entry(o, "the " + names[o], "listed") for o in combo4])
            r.take("VSI-order")
            return None
        return "no_well_separated_order"

    return realize


GROUPS = {
    "V1": (v_combos("V1"), v1_realize), "V2": (v_combos("V2"), v2_realize), "V3": (v_combos("V3"), v3_realize),
    "V4": (v_combos("V4"), v4_realize), "V5": (v_combos("V5"), v5_realize), "V6": (v_combos("V6"), v6_realize),
}
