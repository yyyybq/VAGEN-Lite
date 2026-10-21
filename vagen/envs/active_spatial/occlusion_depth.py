"""Occlusion judged on a rendered depth map (branch transition-qa, the user's decision of 2026-10-21).

The v2 occlusion judge (``canonical_task_metrics_v2.occlusion_silhouette``) calls a point of the target A hidden when
the ray to it crosses the occluder B's labelled box.  A box larger than B (a person, a monitor on its stand), a box
with open space in it (an open wardrobe), a box in the wrong place, and everything that is not B (a wall between two
rooms) all make it wrong; docs/transition_qa/OCCLUSION_JUDGE.md has the cases.  This judge looks at what is rendered.

One depth map at the scoring camera (the canonical H1 camera, 256 x 256, as ``canonical_task_metrics_v2._Context``
builds it) and every pixel whose ray crosses A's box, the rendered surface at that pixel is

* ``A``           between A's box's near and far faces along the ray (A itself, or whatever else is inside its box);
* ``B``           in front of the near face, and its 3-D point is inside B's box widened by ``B_MARGIN_M`` (not
                  widened for the thin or irregular labels, ``THIN_LABELS``);
* ``other``       in front of the near face and not inside B's box (walls, every other object);
* background      behind the far face, or no surface at all (accumulated opacity below ``MIN_ALPHA``): not counted.

``measure`` gives the four quantities: the shares hidden by B, hidden by something else and visible, all of the
counted (non-background) pixels, and how many pixels were counted.  ``label`` turns them into the transition-QA
answer (full / partial / none) or no question; ``hidden`` is the occlusion task's gate (hidden by B >= 0.85, the gate
of canonical_task_metrics_v2.OCC_HIDDEN_MIN).  Pure numpy: the depth map comes from the renderer's
``render_depth_from_cam_param`` (gsplat "RGB+ED": expected depth along the viewing axis, and the opacity).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

OCC_DEPTH_VERSION = "occlusion_depth_v1"

B_MARGIN_M = 0.05        # B's box is widened by this much (box fit), except for THIN_LABELS
NEAR_TOL_M = 0.0         # a surface less than this in front of A's near face still counts as A (splat fuzz)
MIN_ALPHA = 0.5          # = depth_check.MIN_ALPHA: pixels the splats barely cover have no surface
FULL_MIN = 0.85          # = canonical_task_metrics_v2.OCC_HIDDEN_MIN: "fully hidden", and the task's gate
NONE_MAX = 0.05          # "not hidden": hidden by B at most this ...
NONE_VISIBLE_MIN = 0.50  # ... and A visible at least this
PARTIAL_MIN, PARTIAL_MAX = 0.25, 0.70   # "partly hidden"; between the bands: no question
OTHER_MAX = 0.30         # hidden by something other than B at least this: no question (a wall in between)
MIN_PIXELS = 100         # fewer counted pixels (of 65,536): no question

# Occluder labels whose box is not widened: thin or irregular things (the user's occluder check, 2026-10-21;
# data_gen/transition_qa/review.py OCCLUDER_GROUPS["thin"]).
THIN_LABELS = frozenset((
    "live streaming equipment", "computer", "monitor", "easel", "telescope", "musical instrument", "shower head",
    "dry branches", "tv", "microphone", "faucet combination", "clock", "sculpture",
))

# pixel classes of ``classify``
OUTSIDE, CLASS_A, CLASS_B, CLASS_OTHER, BACKGROUND = 0, 1, 2, 3, 4

DEFAULTS: Dict[str, float] = {
    "b_margin": B_MARGIN_M, "near_tol": NEAR_TOL_M, "min_alpha": MIN_ALPHA, "full_min": FULL_MIN,
    "none_max": NONE_MAX, "none_visible_min": NONE_VISIBLE_MIN, "partial_min": PARTIAL_MIN,
    "partial_max": PARTIAL_MAX, "other_max": OTHER_MAX, "min_pixels": MIN_PIXELS,
}


def bounds(obj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(obj["bbox_min"], dtype=np.float64)[:3]
    hi = np.asarray(obj["bbox_max"], dtype=np.float64)[:3]
    return np.minimum(lo, hi), np.maximum(lo, hi)


def b_margin_for(obj: Dict[str, Any], margin: float = B_MARGIN_M) -> float:
    label = str(obj.get("label") or "").strip().lower()
    return 0.0 if label in THIN_LABELS else float(margin)


def pixel_rays(K: np.ndarray, c2w: np.ndarray, width: int, height: int) -> np.ndarray:
    """(H*W, 3) world directions through the pixel centres, scaled so that the camera-z component is 1: a ray
    parameter is then the depth along the viewing axis, the renderer's depth (camera axes right, down, forward)."""
    K = np.asarray(K, dtype=np.float64)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64) + 0.5, np.arange(height, dtype=np.float64) + 0.5)
    cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1).reshape(-1, 3)
    return cam @ np.asarray(c2w, dtype=np.float64)[:3, :3].T


def ray_box(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Entry and exit parameters of each ray through the box [lo, hi] (slab test); NaN where it misses the box or
    the box is behind the camera.  The entry is clipped at 0 (camera inside the box)."""
    origin = np.asarray(origin, dtype=np.float64)
    zero = np.abs(dirs) < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(zero, 1.0, dirs)
        t1 = (lo[None, :] - origin[None, :]) * inv
        t2 = (hi[None, :] - origin[None, :]) * inv
    t_lo, t_hi = np.minimum(t1, t2), np.maximum(t1, t2)
    in_slab = (origin >= lo) & (origin <= hi)
    t_lo = np.where(zero, np.where(in_slab[None, :], -np.inf, np.inf), t_lo)
    t_hi = np.where(zero, np.where(in_slab[None, :], np.inf, -np.inf), t_hi)
    near, far = t_lo.max(axis=1), t_hi.min(axis=1)
    hit = (far >= near) & (far > 0)
    return np.where(hit, np.maximum(near, 0.0), np.nan), np.where(hit, far, np.nan)


def classify(depth: np.ndarray, alpha: np.ndarray, K: np.ndarray, c2w: np.ndarray, target: Dict[str, Any],
             occluder: Dict[str, Any], b_margin: Optional[float] = None, near_tol: float = NEAR_TOL_M,
             min_alpha: float = MIN_ALPHA) -> np.ndarray:
    """(H, W) uint8 pixel classes: OUTSIDE (the ray misses A's box), CLASS_A, CLASS_B, CLASS_OTHER, BACKGROUND.
    ``b_margin`` None: B_MARGIN_M, or 0 for THIN_LABELS."""
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    c2w = np.asarray(c2w, dtype=np.float64)
    origin = c2w[:3, 3]
    dirs = pixel_rays(K, c2w, w, h)
    near, far = ray_box(origin, dirs, *bounds(target))
    d = depth.reshape(-1)
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    region = np.isfinite(near)
    surface = (a >= min_alpha) & np.isfinite(d) & (d > 0)
    with np.errstate(invalid="ignore"):
        front = region & surface & (d < near - near_tol)
        on_a = region & surface & ~front & (d <= far)
    cls = np.full(h * w, OUTSIDE, dtype=np.uint8)
    cls[region] = BACKGROUND
    cls[on_a] = CLASS_A
    idx = np.flatnonzero(front)
    if len(idx):
        margin = b_margin_for(occluder) if b_margin is None else float(b_margin)
        lo, hi = bounds(occluder)
        pts = origin[None, :] + dirs[idx] * d[idx, None]
        in_b = np.all((pts >= lo - margin) & (pts <= hi + margin), axis=1)
        cls[idx[in_b]] = CLASS_B
        cls[idx[~in_b]] = CLASS_OTHER
    return cls.reshape(h, w)


def measure(cls: np.ndarray) -> Dict[str, Any]:
    """The four quantities: shares hidden by B, hidden by something else, visible (of the counted pixels), and the
    number of counted (non-background) pixels; also the pixels of A's box region and the background among them."""
    counts = np.bincount(np.asarray(cls, dtype=np.uint8).reshape(-1), minlength=5)
    n_a, n_b, n_o, n_bg = (int(counts[k]) for k in (CLASS_A, CLASS_B, CLASS_OTHER, BACKGROUND))
    n = n_a + n_b + n_o
    return {
        "hidden_by_b": n_b / n if n else 0.0,
        "hidden_by_other": n_o / n if n else 0.0,
        "visible": n_a / n if n else 0.0,
        "pixels": n,
        "box_pixels": n + n_bg,
        "background_pixels": n_bg,
    }


def label(m: Dict[str, Any], params: Optional[Dict[str, float]] = None) -> Tuple[Optional[str], str]:
    """The transition-QA answer for the quantities of ``measure``: ("full" | "partial" | "none", "") or
    (None, why there is no question)."""
    p = {**DEFAULTS, **(params or {})}
    if m["pixels"] < p["min_pixels"]:
        return None, "too_few_pixels"
    if m["hidden_by_other"] >= p["other_max"]:
        return None, "hidden_by_other"
    h = m["hidden_by_b"]
    if h >= p["full_min"]:
        return "full", ""
    if h <= p["none_max"]:
        return ("none", "") if m["visible"] >= p["none_visible_min"] else (None, "none_not_visible")
    if p["partial_min"] <= h <= p["partial_max"]:
        return "partial", ""
    return None, "between_bands"


def hidden(m: Dict[str, Any], full_min: float = FULL_MIN) -> bool:
    """The occlusion task's gate: A hidden by B (>= 85% of the counted pixels)."""
    return bool(m["hidden_by_b"] >= full_min)


def judge(depth: np.ndarray, alpha: np.ndarray, K: np.ndarray, c2w: np.ndarray, target: Dict[str, Any],
          occluder: Dict[str, Any], params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """``measure`` of ``classify`` plus the label: the whole judge for one depth map."""
    p = {**DEFAULTS, **(params or {})}
    margin = b_margin_for(occluder, p["b_margin"])
    cls = classify(depth, alpha, K, c2w, target, occluder, b_margin=margin, near_tol=p["near_tol"],
                   min_alpha=p["min_alpha"])
    m = measure(cls)
    answer, why = label(m, p)
    return {**m, "label": answer, "no_question": why, "b_margin": margin, "version": OCC_DEPTH_VERSION}


def scoring_camera(item: Dict[str, Any], c2w: Any) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """K, c2w and (width, height) of the camera the occlusion task is scored with (canonical H1, 256 x 256): render
    the depth map with this K and inv(c2w)."""
    from vagen.envs.active_spatial import canonical_task_metrics_v2 as M2

    ctx = M2._Context(item, c2w)
    return ctx.K, ctx.c2w, (ctx.width, ctx.height)


def task_roles(item: Dict[str, Any], c2w: Any) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(target A, occluder B) of an occlusion_alignment row, matched as the v2 judge matches them."""
    from vagen.envs.active_spatial import canonical_task_metrics_v2 as M2

    objs = M2._Context(item, c2w).roles(M2.ROLES["occlusion_alignment"])
    return None if objs is None else (objs[0], objs[1])


def judge_item(item: Dict[str, Any], c2w: Any, depth: np.ndarray, alpha: np.ndarray,
               params: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """``judge`` for an occlusion_alignment row at pose ``c2w``, with the depth map rendered at ``scoring_camera``."""
    roles = task_roles(item, c2w)
    if roles is None:
        return None
    K, cam, _ = scoring_camera(item, c2w)
    return judge(depth, alpha, K, cam, roles[0], roles[1], params)


def overlay(rgb: np.ndarray, cls: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """RGB with the classes tinted (A green, hidden by B red, hidden by something else blue, background grey) and
    the outline of A's box region, for the review pages."""
    out = np.asarray(rgb, dtype=np.float64).copy()
    tints: Sequence[Tuple[int, Tuple[float, float, float]]] = (
        (CLASS_A, (40, 200, 70)), (CLASS_B, (230, 50, 50)), (CLASS_OTHER, (60, 110, 240)), (BACKGROUND, (150, 150, 150)))
    for k, colour in tints:
        mask = cls == k
        out[mask] = (1 - strength) * out[mask] + strength * np.asarray(colour, dtype=np.float64)
    region = cls != OUTSIDE
    edge = region & ~(np.roll(region, 1, 0) & np.roll(region, -1, 0) & np.roll(region, 1, 1) & np.roll(region, -1, 1))
    out[edge] = (255, 255, 0)
    return np.clip(out, 0, 255).astype(np.uint8)
