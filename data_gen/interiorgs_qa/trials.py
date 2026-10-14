"""Trials F (object front), G (object motion) and the colour try-out (user prompt, section 7).

    python -m data_gen.interiorgs_qa.trials --out OUT --scenes-file F [--extra-scenes-file F2] --gs-root PLY

F  For the ten categories, five instances each: an arrow along the fixed local axis -e1 = p0 - p1 of the
   labels.json box (step 1 showed the corner order follows the object's frame; trial_f_walls.py showed that
   for objects backed against a wall the wall lies on the +e1 side in 69-99% of cases for most categories),
   drawn on the floor from the box centre out through the face.  Where the back-against-the-wall rule (or,
   for chairs, the nearest table) gives a front, a second image shows that arrow (step 3).
G  Gaussians inside an object's box (bottom trimmed so the floor / table top stays) are shifted to free space
   nearby; the same camera renders before and after.  Ten small items standing on furniture, ten floor
   pieces.
C  Colour: the dominant colour of the visible part (k-means in Lab, named by the nearest of 11 basic colours);
   only clear cases, with distractor colours far away.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import GenCtx, scene_pool, single_pool
from .geom import CAMERA_HEIGHT_M, OBB, camera_K, look_at, project
from .render import CachedRenderer, GSRenderer
from .scene_ctx import SceneCtx, load_split
from .trial_f_axes import TRIAL_CATEGORIES
from .trial_f_walls import category_of, local_dirs, scene_walls, wall_behind
from .trial_f_axes import edge_frame
from .views import ViewBank, frame_candidates, framed
from .visible import object_view

SIZE = 512
K = camera_K(SIZE)


def ensure_scene(R, sid: str) -> None:
    if getattr(R, "scene_id", None) != sid:
        R.set_scene(sid)


def draw_arrow(img, p0: np.ndarray, p1: np.ndarray, c2w: np.ndarray, color, width: int = 5) -> bool:
    from PIL import ImageDraw

    uv, z = project(np.stack([p0, p1]), c2w, K)
    if np.any(z <= 0.1):
        return False
    d = ImageDraw.Draw(img)
    (x0, y0), (x1, y1) = uv
    d.line([x0, y0, x1, y1], fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (+1, -1):
        a = ang + math.pi + s * math.radians(25)
        d.line([x1, y1, x1 + 18 * math.cos(a), y1 + 18 * math.sin(a)], fill=color, width=width)
    return True


def draw_box(img, box: OBB, c2w: np.ndarray, color=(255, 255, 255), width: int = 2) -> None:
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    for a, b in box.edges():
        uv, z = project(np.stack([a, b]), c2w, K)
        if np.all(z > 0.1):
            d.line([*uv[0], *uv[1]], fill=color, width=width)


# =============================================================================
# F
# =============================================================================
def wall_front(ctx, oid: str) -> Optional[str]:
    """Step 3: the local direction whose opposite face has a wall right behind it (exactly one)."""
    box = ctx.obb[oid]
    f = edge_frame(box.corners)
    if not f["ok"]:
        return None
    walls = scene_walls(ctx.scene_dir, float(ctx.scene.convention_v2.get("applied_sign", 1.0)))
    dirs = local_dirs(f)
    hits = [n for n, (u, hd, hw) in dirs.items() if wall_behind(box.center[:2], u, hd, hw, walls) is not None]
    return hits[0] if len(hits) == 1 else None


def chair_front(ctx, oid: str) -> Optional[str]:
    """Chairs: the local horizontal direction closest to the direction towards the nearest table / desk."""
    box = ctx.obb[oid]
    f = edge_frame(box.corners)
    if not f["ok"]:
        return None
    tables = [o for o in ctx.obb if ("table" in ctx.disp[o] or "desk" in ctx.disp[o]) and o != oid]
    if not tables:
        return None
    t = min(tables, key=lambda o: float(np.linalg.norm(ctx.obb[o].center[:2] - box.center[:2])))
    v = ctx.obb[t].center[:2] - box.center[:2]
    if np.linalg.norm(v) > 1.5:
        return None
    v /= np.linalg.norm(v)
    dirs = local_dirs(f)
    best = max(dirs, key=lambda n: float(dirs[n][0] @ v))
    return best if float(dirs[best][0] @ v) >= math.cos(math.radians(35)) else None


def dir_vector(ctx, oid: str, name: str) -> np.ndarray:
    f = edge_frame(ctx.obb[oid].corners)
    u = local_dirs(f)[name][0]
    return np.array([u[0], u[1], 0.0])


def trial_f(ctxs: Dict[str, SceneCtx], R, out: Path, rng: random.Random) -> List[Dict[str, Any]]:
    items = []
    (out / "trial_f").mkdir(parents=True, exist_ok=True)
    for cat in TRIAL_CATEGORIES:
        cands = []
        for sid, ctx in ctxs.items():
            for o in ctx.obb:
                if category_of(ctx.by_id[o].label) == cat and ctx.obb[o].from_corners and o not in ctx.duplicates:
                    cands.append((sid, o))
        rng.shuffle(cands)
        # prefer different world directions of the arrow and different scenes
        taken, seen_dir, seen_scene = [], Counter(), Counter()
        for sid, o in sorted(cands, key=lambda so: rng.random()):
            if len(taken) >= 5:
                break
            ctx = ctxs[sid]
            v = dir_vector(ctx, o, "-e1")
            dkey = int(round(math.degrees(math.atan2(v[1], v[0])) / 90.0)) % 4
            if seen_dir[dkey] >= 2 or seen_scene[sid] >= 2:
                continue
            res = render_arrow_item(ctx, R, o, cat, out, rng)
            if res is None:
                continue
            taken.append(res)
            seen_dir[dkey] += 1
            seen_scene[sid] += 1
        items += taken
        print("trial F", cat, len(taken), "of", len(cands), flush=True)
    return items


def render_arrow_item(ctx, R, o: str, cat: str, out: Path, rng) -> Optional[Dict[str, Any]]:
    from PIL import Image

    box = ctx.obb[o]
    ensure_scene(R, ctx.scene_id)
    v = dir_vector(ctx, o, "-e1")
    a0 = math.degrees(math.atan2(v[1], v[0]))
    poses = []
    for off in (35.0, -35.0, 55.0, -55.0, 20.0, -20.0):
        for r in (1.8, 2.3, 2.9, 3.6):
            a = math.radians(a0 + off)
            e = box.center[:2] + r * np.array([math.cos(a), math.sin(a)])
            if ctx.legal(e[None])[0]:
                poses.append(look_at((e[0], e[1], CAMERA_HEIGHT_M), box.center))
    cams = frame_candidates(ctx, [o], poses=poses, max_out=1)
    if not cams:
        return None
    c2w = cams[0]
    rgb, depth, alpha = R.render(c2w, SIZE)
    ov, _ = object_view(o, box, c2w, depth, alpha)
    if not ov.recognisable(0.7):
        return None
    tag = f"{cat.replace(' ', '_')}_{ctx.scene_id}_{o}"
    base = Image.fromarray(rgb)
    base.save(out / "trial_f" / f"{tag}_plain.png")
    floor = np.array([box.center[0], box.center[1], box.lo[2] + 0.03])
    half = box.half[0] if True else 0.0
    out_imgs = {}
    for name, color, key in (("-e1", (230, 30, 40), "axis"),):
        u = dir_vector(ctx, o, name)
        img = base.copy()
        draw_box(img, box, c2w)
        draw_arrow(img, floor, floor + u * (box.half[0] + 0.6), c2w, color)
        img.save(out / "trial_f" / f"{tag}_{key}.png")
        out_imgs[key] = f"trial_f/{tag}_{key}.png"
    inferred = chair_front(ctx, o) if cat == "chair" else wall_front(ctx, o)
    if inferred is not None:
        u = dir_vector(ctx, o, inferred)
        img = base.copy()
        draw_box(img, box, c2w)
        f = edge_frame(box.corners)
        hd = local_dirs(f)[inferred][1]
        draw_arrow(img, floor, floor + u * (hd + 0.6), c2w, (20, 170, 60))
        img.save(out / "trial_f" / f"{tag}_inferred.png")
        out_imgs["inferred"] = f"trial_f/{tag}_inferred.png"
    Image.fromarray(rgb).resize((256, 256)).save(out / "trial_f" / f"{tag}_plain_256.png")
    return {"category": cat, "scene_id": ctx.scene_id, "oid": o, "label": ctx.by_id[o].label,
            "images": out_imgs, "plain": f"trial_f/{tag}_plain.png", "c2w": c2w.round(5).tolist(),
            "axis": "-e1 (p0 - p1)", "axis_world_deg": round(a0, 1), "inferred_dir": inferred,
            "inferred_rule": "nearest table" if cat == "chair" else "back against wall",
            "handedness": edge_frame(box.corners)["handedness"]}


# =============================================================================
# G
# =============================================================================
def support_of(ctx, o: str) -> Optional[str]:
    """The box an object stands on: its bottom within 6 cm of the other's top, footprint mostly over it."""
    from .geom import footprint_overlap_fraction

    b = ctx.obb[o]
    best = None
    for s, sb in ctx.obb.items():
        if s == o or sb.hi[2] > b.lo[2] + 0.06 or sb.hi[2] < b.lo[2] - 0.06:
            continue
        if footprint_overlap_fraction(b, sb, of="a") >= 0.7:
            best = s if best is None or sb.footprint_area() < ctx.obb[best].footprint_area() else best
    return best


def free_shift(ctx, o: str, support: Optional[str], rng) -> Optional[np.ndarray]:
    """A horizontal shift moving the object to free space: on its support's top (tabletop items) or on the
    floor (floor pieces); the moved box overlaps no other box and stays in the room."""
    from .geom import boxes_overlap

    b = ctx.obb[o]
    others = [x for x in ctx.obb if x != o and x != support and not ctx.by_id[x].label.lower().endswith(("rug", "carpet", "mat"))]
    dists = (0.25, 0.35, 0.5) if support else (0.8, 1.1, 1.4, 1.8)
    for dist in dists:
        for k in rng.sample(range(16), 16):
            a = 2 * math.pi * k / 16
            t = np.array([dist * math.cos(a), dist * math.sin(a), 0.0])
            nb = OBB(b.center + t, b.axes, b.half, b.corners + t, b.from_corners)
            if support:
                s = ctx.obb[support]
                if not s.contains(nb.corners[:4] - np.array([0, 0, 0.03]), margin=0.0).all():
                    continue
            else:
                if ctx.room_at(nb.center[:2]) != ctx.room_at(b.center[:2]):
                    continue
            shrink = OBB(nb.center, nb.axes, nb.half * 0.98, nb.corners, nb.from_corners)
            if any(boxes_overlap(shrink, ctx.obb[x]) for x in others if np.linalg.norm(ctx.obb[x].center[:2] - nb.center[:2]) < 3):
                continue
            return t
    return None


def trial_g(ctxs: Dict[str, SceneCtx], R, out: Path, rng: random.Random) -> List[Dict[str, Any]]:
    import torch
    from PIL import Image

    (out / "trial_g").mkdir(parents=True, exist_ok=True)
    pools = {"tabletop": [], "floor": []}
    for sid, ctx in ctxs.items():
        for o in single_pool(ctx, 2.5):
            b = ctx.obb[o]
            if b.lo[2] <= 0.05:
                pools["floor"].append((sid, o, None))
            else:
                s = support_of(ctx, o)
                if s is not None and b.longest_edge <= 0.8:
                    pools["tabletop"].append((sid, o, s))
    items = []
    for kind in ("tabletop", "floor"):
        cand = pools[kind]
        rng.shuffle(cand)
        n, per_scene = 0, Counter()
        for sid, o, s in cand:
            if n >= 10 or per_scene[sid] >= 2:
                continue
            ctx = ctxs[sid]
            t = free_shift(ctx, o, s, rng)
            if t is None:
                continue
            b = ctx.obb[o]
            moved = OBB(b.center + t, b.axes, b.half, b.corners + t, b.from_corners)
            # camera framing both the old and the new place
            mid = (b.center + moved.center) / 2
            poses = []
            for r_ in (1.8, 2.4, 3.0, 3.6):
                for k in range(24):
                    a = 2 * math.pi * k / 24
                    e = mid[:2] + r_ * np.array([math.cos(a), math.sin(a)])
                    if ctx.legal(e[None])[0]:
                        poses.append(look_at((e[0], e[1], CAMERA_HEIGHT_M), mid))
            good = [p for p in poses if framed(b, p, K, SIZE) and framed(moved, p, K, SIZE)]
            good = [p for p in good if ctx.face_visible(o, p[None], K, SIZE)[0]]
            if not good:
                continue
            c2w = good[len(good) // 2]
            ensure_scene(R, sid)
            G = R.gaussians
            means = G.means
            lo_trim = 0.02 if kind == "floor" else 0.01
            with torch.no_grad():
                bt = torch.tensor(b.center, device=means.device, dtype=means.dtype)
                ax = torch.tensor(b.axes, device=means.device, dtype=means.dtype)
                hf = torch.tensor(b.half + np.array([0.03, 0.03, 0.03]), device=means.device, dtype=means.dtype)
                loc = (means - bt) @ ax
                inside = (loc.abs() <= hf).all(dim=1) & (means[:, 2] >= float(b.lo[2]) + lo_trim)
                idx = torch.nonzero(inside).squeeze(1)
                rgb0, d0, a0 = R.render(c2w, SIZE)
                shift = torch.tensor(t, device=means.device, dtype=means.dtype)
                means[idx] += shift
                rgb1, d1, a1 = R.render(c2w, SIZE)
                means[idx] -= shift
            tag = f"{kind}_{sid}_{o}"
            Image.fromarray(rgb0).save(out / "trial_g" / f"{tag}_before.png")
            Image.fromarray(rgb1).save(out / "trial_g" / f"{tag}_after.png")
            img = Image.fromarray(rgb1)
            draw_box(img, b, c2w, (230, 30, 40))
            draw_box(img, moved, c2w, (30, 160, 70))
            img.save(out / "trial_g" / f"{tag}_after_boxes.png")
            items.append({"kind": kind, "scene_id": sid, "oid": o, "label": ctx.by_id[o].label, "display": ctx.disp[o],
                          "support": s, "shift_m": np.round(t, 3).tolist(), "gaussians_moved": int(idx.numel()),
                          "bottom_trim_m": lo_trim, "images": {"before": f"trial_g/{tag}_before.png",
                                                               "after": f"trial_g/{tag}_after.png",
                                                               "after_boxes": f"trial_g/{tag}_after_boxes.png"},
                          "c2w": c2w.round(5).tolist()})
            n += 1
            per_scene[sid] += 1
        print("trial G", kind, n, "of", len(cand), flush=True)
    return items


# =============================================================================
# Colour try-out
# =============================================================================
BASIC = {"black": (20, 20, 20), "white": (240, 240, 240), "gray": (128, 128, 128), "red": (200, 30, 30),
         "orange": (230, 130, 30), "yellow": (230, 210, 40), "green": (40, 150, 60), "blue": (40, 80, 200),
         "purple": (120, 50, 160), "pink": (235, 150, 180), "brown": (120, 75, 40)}


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


BASIC_LAB = {k: rgb2lab(np.array(v)) for k, v in BASIC.items()}


def dominant_colour(pixels: np.ndarray, rng) -> Tuple[str, float, float, np.ndarray]:
    """(colour name, share of the dominant cluster, naming margin, cluster centre RGB) by 3-means in Lab."""
    lab = rgb2lab(pixels)
    k = 3
    cent = lab[rng.sample(range(len(lab)), k)]
    for _ in range(15):
        d = np.linalg.norm(lab[:, None] - cent[None], axis=2)
        lab_i = d.argmin(1)
        cent = np.array([lab[lab_i == j].mean(0) if (lab_i == j).any() else cent[j] for j in range(k)])
    share = np.bincount(lab_i, minlength=k) / len(lab)
    j = int(share.argmax())
    names = sorted(BASIC_LAB, key=lambda n: float(np.linalg.norm(cent[j] - BASIC_LAB[n])))
    d1 = float(np.linalg.norm(cent[j] - BASIC_LAB[names[0]]))
    d2 = float(np.linalg.norm(cent[j] - BASIC_LAB[names[1]]))
    rgb_c = pixels[lab_i == j].mean(0)
    return names[0], float(share[j]), d2 - d1, rgb_c


def trial_colour(ctxs: Dict[str, SceneCtx], R, out: Path, rng) -> List[Dict[str, Any]]:
    from PIL import Image

    from .draw import erode

    (out / "trial_c").mkdir(parents=True, exist_ok=True)
    items = []
    cands = [(sid, o) for sid, ctx in ctxs.items() for o in scene_pool(ctx)]
    rng.shuffle(cands)
    for sid, o in cands:
        if len(items) >= 10:
            break
        ctx = ctxs[sid]
        cams = frame_candidates(ctx, [o], max_out=1, radii=(1.5, 2.0, 2.6, 3.2))
        if not cams:
            continue
        ensure_scene(R, sid)
        rgb, depth, alpha = R.render(cams[0], SIZE)
        ov, mask = object_view(o, ctx.obb[o], cams[0], depth, alpha)
        if not ov.recognisable(0.8):
            continue
        m = erode(mask, 4)
        px = rgb[m]
        if len(px) < 400:
            continue
        name, share, margin, centre = dominant_colour(px.astype(float), rng)
        if share < 0.55 or margin < 12:
            continue
        lab0 = BASIC_LAB[name]
        far = [n for n in BASIC_LAB if n != name and float(np.linalg.norm(BASIC_LAB[n] - lab0)) >= 45]
        if len(far) < 3:
            continue
        opts = [name] + rng.sample(far, 3)
        rng.shuffle(opts)
        tag = f"{sid}_{o}"
        Image.fromarray(rgb).save(out / "trial_c" / f"{tag}.png")
        Image.fromarray(rgb).resize((256, 256)).save(out / "trial_c" / f"{tag}_256.png")
        ref = ctx.scene_ref(o)
        items.append({"scene_id": sid, "oid": o, "name": ref, "question": f"What is the main color of {ref}?",
                      "options": opts, "answer": name, "dominant_share": round(share, 3), "naming_margin": round(margin, 1),
                      "cluster_rgb": np.round(centre).astype(int).tolist(), "image": f"trial_c/{tag}.png",
                      "bbox": ov.bbox})
    print("colour", len(items), flush=True)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes-file", required=True)
    ap.add_argument("--extra-scenes-file", default=None)
    ap.add_argument("--which", default="f,g,c")
    ap.add_argument("--gs-root", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = [l.strip() for l in Path(a.scenes_file).read_text().splitlines() if l.strip()]
    extra = [l.strip() for l in Path(a.extra_scenes_file).read_text().splitlines() if l.strip()] if a.extra_scenes_file else []
    rng = random.Random(20261014)
    t0 = time.time()
    ctxs = {s: SceneCtx(s) for s in scenes}
    R = GSRenderer(a.gs_root)
    res: Dict[str, Any] = {"scenes": scenes}
    which = a.which.split(",")
    if "f" in which:
        items = trial_f(ctxs, R, out, rng)
        short = [c for c in TRIAL_CATEGORIES if sum(i["category"] == c for i in items) < 5]
        if short and extra:
            ext = {s: SceneCtx(s) for s in extra}
            more = trial_f({**ext}, R, out, rng)
            for c in short:
                need = 5 - sum(i["category"] == c for i in items)
                items += [i for i in more if i["category"] == c][:need]
            res["extra_scenes_used_for_f"] = sorted({i["scene_id"] for i in items} - set(scenes))
        res["trial_f"] = items
    if "g" in which:
        res["trial_g"] = trial_g(ctxs, R, out, rng)
    if "c" in which:
        res["trial_colour"] = trial_colour(ctxs, R, out, rng)
    res["seconds"] = round(time.time() - t0, 1)
    (out / "trials.json").write_text(json.dumps(res, indent=1))
    print("done", res["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
