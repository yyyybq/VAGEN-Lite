"""Camera placement for chosen objects, and the store of rendered views.

Constructive placement (user prompt, section 3): the generators first pick the objects a question needs,
then ``frame_candidates`` proposes legal camera positions around them (rings at several distances, the
camera at eye height 1.5 m looking at the objects), keeps those where every object passes v10's visibility
rule (face_visible_v3, used as a cheap prefilter) and is framed, and ranks them.  Only the best few are
rendered; the rendered depth then decides (visible.py).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geom import CAMERA_HEIGHT_M, IMAGE_SIZE, OBB, camera_K, look_at, project
from .visible import ViewAnalysis

RADII = (1.3, 1.7, 2.1, 2.6, 3.1, 3.7, 4.4)
N_AZIMUTH = 36


def framed(box: OBB, c2w: np.ndarray, K: np.ndarray, size: int, margin: float = 0.02,
           allow_partial_large: bool = True) -> bool:
    """All 8 corners in front of the camera and inside the image (margin as a share of the size); a large
    object (longest edge > 2.2 m) may instead have its centre and 60% of its corners inside."""
    uv, z = project(box.corners, c2w, K)
    m = margin * size
    inside = (z > 0.1) & (uv[:, 0] >= m) & (uv[:, 0] <= size - m) & (uv[:, 1] >= m) & (uv[:, 1] <= size - m)
    if inside.all():
        return True
    if allow_partial_large and box.longest_edge > 2.2:
        cuv, cz = project(box.center[None], c2w, K)
        c_in = cz[0] > 0.1 and 0 <= cuv[0, 0] <= size and 0 <= cuv[0, 1] <= size
        return bool(c_in and inside.mean() >= 0.6)
    return False


def ring_poses(ctx, target: np.ndarray, radii: Sequence[float] = RADII, n_az: int = N_AZIMUTH,
               az_range: Optional[Tuple[float, float]] = None) -> List[np.ndarray]:
    """Legal cameras on rings around ``target`` (world point), looking at it."""
    target = np.asarray(target, float)
    eyes = []
    for r in radii:
        for k in range(n_az):
            a = 2 * math.pi * k / n_az
            if az_range is not None:
                lo, hi = az_range
                a = lo + (hi - lo) * k / max(n_az - 1, 1)
            eyes.append((target[0] + r * math.cos(a), target[1] + r * math.sin(a)))
    eyes = np.asarray(eyes)
    ok = ctx.legal(eyes)
    return [look_at((x, y, CAMERA_HEIGHT_M), target) for (x, y) in eyes[ok]]


SAME_NAME_PREFILTER = True   # module switch: the CPU same-name check below (off only to measure its effect)


def same_name_visible(ctx, oids: Sequence[str], c2ws: np.ndarray, K: np.ndarray, size: int) -> np.ndarray:
    """(N,) True where some other object a reader would call by the same name as one of ``oids`` may show in
    the picture (any sample point of a facing side not blocked by boxes or walls). Objects with a scene-unique
    description are skipped: a single-image question can still name them by that description. The render check
    (ViewAnalysis.name_unique_in_image) stays the authority; this only saves renders that would fail it."""
    hit = np.zeros(len(c2ws), bool)
    for o in oids:
        if ctx.scene_ref(o) is not None:
            continue
        for x in ctx.same_name(o):
            if ctx.duplicates.get(x) == o or ctx.duplicates.get(o) == x:
                continue
            todo = np.flatnonzero(~hit)
            if not len(todo):
                return hit
            box = ctx.obb.get(x)
            if box is not None:   # cheap projection test first: most instances are behind or beside the camera
                todo = todo[[not box_outside_view(box, c2ws[i], K, size) for i in todo]]
                if not len(todo):
                    continue
            try:
                seen = ctx.face_seen_any(x, c2ws[todo], K, size)
            except (AttributeError, KeyError, ValueError):
                continue
            hit[todo[seen]] = True
    return hit


def frame_candidates(ctx, oids: Sequence[str], size: int = IMAGE_SIZE, target: Optional[np.ndarray] = None,
                     poses: Optional[List[np.ndarray]] = None, extra: Optional[Callable[[np.ndarray], bool]] = None,
                     need_visible: Optional[Sequence[str]] = None, max_out: int = 4,
                     radii: Sequence[float] = RADII, unique_in_image: Sequence[str] = ()) -> List[np.ndarray]:
    """Cameras that see every object of ``need_visible`` (default ``oids``) clearly by the v10 rule and
    frame every object of ``oids``; ``extra`` adds a question-specific check.  Ranked by the smallest
    projected size of the objects (larger is better) with diversity in viewing direction.
    ``unique_in_image``: objects a single-image question names as "the X" -- cameras where another X may show
    are dropped before rendering (same_name_visible)."""
    K = camera_K(size)
    if target is None:
        target = np.mean([ctx.obb[o].center for o in oids], axis=0)
    cands = ring_poses(ctx, target, radii) if poses is None else list(poses)
    if not cands:
        return []
    c2ws = np.asarray(cands)
    ok = np.ones(len(cands), bool)
    for o in oids:
        ok &= np.array([framed(ctx.obb[o], c, K, size) for c in c2ws])
    if not ok.any():
        return []
    idx = np.flatnonzero(ok)
    for o in (need_visible if need_visible is not None else oids):
        vis = ctx.face_visible(o, c2ws[idx], K, size)
        idx = idx[vis]
        if not len(idx):
            return []
    if extra is not None:
        idx = np.array([i for i in idx if extra(c2ws[i])], dtype=int)
        if not len(idx):
            return []
    if unique_in_image and SAME_NAME_PREFILTER:
        idx = idx[~same_name_visible(ctx, unique_in_image, c2ws[idx], K, size)]
        if not len(idx):
            return []
    # score: the smallest object's projected height share (bigger is better), mild preference for ~2.5 m
    scores = []
    for i in idx:
        c = c2ws[i]
        s = []
        for o in oids:
            uv, z = project(ctx.obb[o].corners, c, K)
            s.append(min(1.0, (uv[:, 1].max() - uv[:, 1].min()) / size + (uv[:, 0].max() - uv[:, 0].min()) / size))
        dist = float(np.linalg.norm(c[:2, 3] - target[:2]))
        scores.append(min(s) - 0.03 * abs(dist - 2.5))
    order = [idx[j] for j in np.argsort(scores)[::-1]]
    chosen: List[np.ndarray] = []
    for i in order:
        c = c2ws[i]
        f = c[:3, 2]
        if all(float(f @ d[:3, 2]) < math.cos(math.radians(25)) or
               np.linalg.norm(c[:3, 3] - d[:3, 3]) > 1.0 for d in chosen):
            chosen.append(c)
        if len(chosen) >= max_out:
            break
    return chosen


def view_key(scene_id: str, c2w: np.ndarray, size: int) -> str:
    h = hashlib.sha256((scene_id + "|" + str(size) + "|" +
                        ",".join(f"{v:.5f}" for v in np.asarray(c2w, float)[:3, :4].ravel())).encode())
    return h.hexdigest()[:14]


class ViewBank:
    """Renders views on demand, saves them once (512 and 256 PNG, depth), and analyses them."""

    def __init__(self, ctx, renderer, out_dir: Path, size: int = IMAGE_SIZE, save_depth: bool = True) -> None:
        self.ctx = ctx
        self.R = renderer
        self.out = Path(out_dir)
        self.size = size
        self.save_depth = save_depth
        self.records: Dict[str, Dict[str, Any]] = {}
        self.analysis: Dict[str, ViewAnalysis] = {}

    def view(self, c2w: np.ndarray, purpose: str = "") -> Tuple[str, ViewAnalysis]:
        c2w = np.asarray(c2w, float)
        vid = view_key(self.ctx.scene_id, c2w, self.size)
        if vid not in self.analysis:
            rgb, depth, alpha = self.R.render(c2w, self.size)
            self.analysis[vid] = ViewAnalysis(self.ctx, c2w, rgb, depth, alpha)
            self.records[vid] = {"view_id": vid, "scene_id": self.ctx.scene_id, "c2w": c2w.round(6).tolist(),
                                 "size": self.size, "renderer": getattr(self.R, "kind", "?"), "purposes": [],
                                 "saved": False}
        if purpose and purpose not in self.records[vid]["purposes"]:
            self.records[vid]["purposes"].append(purpose)
        return vid, self.analysis[vid]

    def rel_path(self, vid: str, suffix: str = ".png") -> str:
        return f"views/{self.ctx.scene_id}/{vid}{suffix}"

    def save(self, vid: str) -> str:
        """Write the view's images (and depth) the first time a question uses it; return the 512 path."""
        rec = self.records[vid]
        if not rec["saved"]:
            from PIL import Image

            va = self.analysis[vid]
            d = self.out / "views" / self.ctx.scene_id
            d.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(va.rgb)
            img.save(d / f"{vid}.png")
            img.resize((256, 256), Image.BILINEAR).save(d / f"{vid}_256.png")
            if self.save_depth:
                np.savez_compressed(d / f"{vid}.depth.npz", depth=va.depth.astype(np.float16),
                                    alpha=(va.alpha * 255).astype(np.uint8))
            rec["saved"] = True
        return self.rel_path(vid)

    def save_marked(self, vid: str, tag: str, draw: Callable[[Any], None]) -> str:
        """A copy of the view with marks drawn into it (BLINK points or boxes): it is the model's input."""
        from PIL import Image

        va = self.analysis[vid]
        d = self.out / "views" / self.ctx.scene_id
        d.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(va.rgb).convert("RGB")
        draw(img)
        name = f"{vid}__{tag}"
        img.save(d / f"{name}.png")
        img.resize((256, 256), Image.BILINEAR).save(d / f"{name}_256.png")
        self.save(vid)
        return self.rel_path(name)

    def dump(self, path: Path) -> None:
        with open(path, "a") as f:
            for vid, rec in self.records.items():
                if rec["saved"]:
                    f.write(json.dumps(rec) + "\n")


def box_outside_view(box: OBB, c2w: np.ndarray, K: np.ndarray, size: int, margin: float = 0.03) -> bool:
    """True when the box's whole projection lies outside the image (with a margin): it cannot show at all."""
    uv, z = project(box.corners, c2w, K)
    if np.all(z <= 0.05):
        return True
    if np.any(z <= 0.05):
        return False   # straddles the camera plane: may reach into the image
    m = margin * size
    return bool(uv[:, 0].max() < -m or uv[:, 0].min() > size + m or uv[:, 1].max() < -m or uv[:, 1].min() > size + m)


def framed_loose(box: OBB, c2w: np.ndarray, K: np.ndarray, size: int) -> bool:
    """The box's centre projects inside the image and at least half of its corners do (a landmark that is
    mostly in the picture)."""
    uv, z = project(np.vstack([box.corners, box.center[None]]), c2w, K)
    inside = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] <= size) & (uv[:, 1] >= 0) & (uv[:, 1] <= size)
    return bool(inside[-1] and inside[:8].mean() >= 0.5)
