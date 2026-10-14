"""The visible part of an object in a rendered view (user prompt, section 5.2).

A pixel belongs to the visible part of object X when

1. its ray passes through X's 3D box (the box's silhouette), and
2. the rendered depth at the pixel lies within the stretch of the ray inside the box (with a tolerance for the
   box fit and the splats' fuzz), and the splats cover the pixel (accumulated opacity >= MIN_ALPHA).

Everything else derives from this mask: how much of X shows (visible pixels / silhouette pixels inside the
image), who hides whom, the 2D box of the visible part, its mean depth, cross-image point correspondence and
the first frame an object appears in a video.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .geom import OBB, camera_K, pixel_rays, project

MIN_ALPHA = 0.5
TOL_FRONT_M = 0.08      # a surface up to this far in front of the box still counts as the object ...
TOL_FRONT_REL = 0.04    # ... or this share of the depth, whichever is larger (box fit, splat fuzz)
TOL_BACK_M = 0.08
EXTENDED = 0.5          # silhouette also measured on a grid this much (share of the size) beyond each border

# Visibility classes at any image size (shares of the image area)
RECOGNISABLE_AREA = 0.004   # ~1,050 px at 512 x 512, ~260 px at 256 x 256 (v10 needs 0.3% for one face)
RECOGNISABLE_FRACTION = 0.5  # at least half of the in-image silhouette not hidden by something in front
MIN_SOLID_FRACTION = 0.2     # and at least a fifth of it actually showing the object (not see-through)
ABSENT_AREA = 0.0005         # below ~130 px at 512 x 512 the object counts as not in the image


@dataclasses.dataclass
class ObjView:
    oid: str
    vis_px: int                    # visible pixels
    sil_px: int                    # silhouette pixels inside the image
    occ_px: int                    # silhouette pixels where something lies in front of the box (hidden)
    sil_total_est: float           # silhouette pixels including beyond the borders (estimated)
    bbox: Optional[Tuple[int, int, int, int]]  # x0, y0, x1, y1 of the visible part (inclusive)
    mean_depth: Optional[float]
    median_depth: Optional[float]
    centroid: Optional[Tuple[float, float]]
    size: int

    @property
    def area(self) -> float:
        return self.vis_px / float(self.size * self.size)

    @property
    def fraction(self) -> float:
        """Visible share of the in-image silhouette (1 = nothing in front of it)."""
        return self.vis_px / max(self.sil_px, 1)

    @property
    def unoccluded(self) -> float:
        """Share of the in-image silhouette NOT hidden by something in front of the box.  Unlike
        ``fraction`` it does not count the background seen through an open object (chair legs, a table
        top on legs) as hidden."""
        return 1.0 - self.occ_px / max(self.sil_px, 1)

    @property
    def in_frame(self) -> float:
        """Share of the whole silhouette that falls inside the image."""
        return self.sil_px / max(self.sil_total_est, 1.0)

    def recognisable(self, min_in_frame: float = 0.7) -> bool:
        return (self.area >= RECOGNISABLE_AREA and self.unoccluded >= RECOGNISABLE_FRACTION
                and self.fraction >= MIN_SOLID_FRACTION and self.in_frame >= min_in_frame)

    def absent(self) -> bool:
        return self.area < ABSENT_AREA

    def to_json(self) -> Dict:
        return {"oid": self.oid, "vis_px": self.vis_px, "sil_px": self.sil_px, "occ_px": self.occ_px,
                "unoccluded": round(self.unoccluded, 3),
                "sil_total_est": round(self.sil_total_est, 1), "bbox": self.bbox,
                "mean_depth": None if self.mean_depth is None else round(self.mean_depth, 3),
                "median_depth": None if self.median_depth is None else round(self.median_depth, 3),
                "centroid": None if self.centroid is None else [round(c, 1) for c in self.centroid],
                "fraction": round(self.fraction, 3), "in_frame": round(self.in_frame, 3),
                "area": round(self.area, 5)}


def _box_rect(box: OBB, c2w: np.ndarray, K: np.ndarray, size: int, pad: float) -> Optional[Tuple[int, int, int, int]]:
    """Pixel rectangle (u0, v0, u1, v1) on a grid extended by ``pad`` * size around the image that holds
    the box's silhouette; None if the box is entirely behind the camera."""
    uv, z = project(box.corners, c2w, K)
    lo_b, hi_b = -pad * size, (1 + pad) * size
    if np.all(z <= 0.05):
        return None
    if np.any(z <= 0.05):  # straddles the camera plane: the silhouette may reach any border
        return int(lo_b), int(lo_b), int(hi_b), int(hi_b)
    u0, v0 = np.floor(uv.min(0)).astype(int)
    u1, v1 = np.ceil(uv.max(0)).astype(int)
    u0, v0 = max(u0, int(lo_b)), max(v0, int(lo_b))
    u1, v1 = min(u1, int(hi_b)), min(v1, int(hi_b))
    if u0 > u1 or v0 > v1:
        return None
    return u0, v0, u1, v1


def silhouette(box: OBB, c2w: np.ndarray, K: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-image silhouette mask (size, size) and the ray entry / exit depths (z) of the box per pixel."""
    sil = np.zeros((size, size), bool)
    tn_img = np.full((size, size), np.nan, np.float32)
    tf_img = np.full((size, size), np.nan, np.float32)
    rect = _box_rect(box, c2w, K, size, 0.0)
    if rect is None:
        return sil, tn_img, tf_img
    u0, v0, u1, v1 = rect
    u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, size - 1), min(v1, size - 1)
    if u0 > u1 or v0 > v1:
        return sil, tn_img, tf_img
    vs, us = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    d = pixel_rays(c2w, K, us, vs)
    tn, tf = box.ray_interval(np.asarray(c2w, float)[:3, 3], d)
    hit = tf >= np.maximum(tn, 0.05)
    sil[v0:v1 + 1, u0:u1 + 1] = hit
    tn_img[v0:v1 + 1, u0:u1 + 1] = np.where(hit, np.maximum(tn, 0.0), np.nan)
    tf_img[v0:v1 + 1, u0:u1 + 1] = np.where(hit, tf, np.nan)
    return sil, tn_img, tf_img


def silhouette_total(box: OBB, c2w: np.ndarray, K: np.ndarray, size: int, stride: int = 4) -> float:
    """Estimated silhouette pixels on the image plane extended by EXTENDED * size beyond every border."""
    rect = _box_rect(box, c2w, K, size, EXTENDED)
    if rect is None:
        return 0.0
    u0, v0, u1, v1 = rect
    vs, us = np.mgrid[v0:v1 + 1:stride, u0:u1 + 1:stride]
    d = pixel_rays(c2w, K, us, vs)
    tn, tf = box.ray_interval(np.asarray(c2w, float)[:3, 3], d)
    return float((tf >= np.maximum(tn, 0.05)).sum()) * stride * stride


def visible_mask(box: OBB, c2w: np.ndarray, depth: np.ndarray, alpha: np.ndarray,
                 K: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(visible, silhouette, occluded): occluded = silhouette pixels whose rendered surface lies in front of
    the box (something hides the object there)."""
    size = depth.shape[0]
    K = camera_K(size) if K is None else K
    sil, tn, tf = silhouette(box, c2w, K, size)
    with np.errstate(invalid="ignore"):
        tol_f = np.maximum(TOL_FRONT_M, TOL_FRONT_REL * tn)
        covered = sil & (alpha >= MIN_ALPHA)
        vis = covered & (depth >= tn - tol_f) & (depth <= tf + TOL_BACK_M)
        occ = covered & (depth < tn - tol_f)
    return vis, sil, occ


def object_view(oid: str, box: OBB, c2w: np.ndarray, depth: np.ndarray, alpha: np.ndarray,
                K: Optional[np.ndarray] = None) -> Tuple[ObjView, np.ndarray]:
    size = depth.shape[0]
    K = camera_K(size) if K is None else K
    vis, sil, occ = visible_mask(box, c2w, depth, alpha, K)
    n = int(vis.sum())
    n_occ = int(occ.sum())
    total = max(silhouette_total(box, c2w, K, size), float(sil.sum()))
    if n:
        ys, xs = np.nonzero(vis)
        dv = depth[vis]
        ov = ObjView(oid, n, int(sil.sum()), n_occ, total, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                     float(dv.mean()), float(np.median(dv)), (float(xs.mean()), float(ys.mean())), size)
    else:
        ov = ObjView(oid, 0, int(sil.sum()), n_occ, total, None, None, None, None, size)
    return ov, vis


class ViewAnalysis:
    """Per-object visible parts in one rendered view (computed lazily, cached)."""

    def __init__(self, ctx, c2w: np.ndarray, rgb: np.ndarray, depth: np.ndarray, alpha: np.ndarray) -> None:
        self.ctx = ctx
        self.c2w = np.asarray(c2w, float)
        self.rgb, self.depth, self.alpha = rgb, depth, alpha
        self.size = depth.shape[0]
        self.K = camera_K(self.size)
        self._ov: Dict[str, ObjView] = {}
        self._mask: Dict[str, np.ndarray] = {}

    def obj(self, oid: str) -> ObjView:
        if oid not in self._ov:
            box = self.ctx.obb.get(oid)
            if box is None:
                self._ov[oid] = ObjView(oid, 0, 0, 0, 0.0, None, None, None, None, self.size)
                self._mask[oid] = np.zeros(self.depth.shape, bool)
            else:
                self._ov[oid], self._mask[oid] = object_view(oid, box, self.c2w, self.depth, self.alpha, self.K)
        return self._ov[oid]

    def mask(self, oid: str) -> np.ndarray:
        self.obj(oid)
        return self._mask[oid]

    def candidates_in_frustum(self, oids: Iterable[str]) -> List[str]:
        out = []
        for oid in oids:
            box = self.ctx.obb.get(oid)
            if box is not None and _box_rect(box, self.c2w, self.K, self.size, 0.0) is not None:
                out.append(oid)
        return out

    def name_unique_in_image(self, oid: str) -> bool:
        """No other object a reader could call by the same name shows in the image (absent threshold)."""
        for other in self.ctx.same_name(oid):
            if other in self.ctx.duplicates and self.ctx.duplicates[other] == oid:
                continue
            if oid in self.ctx.duplicates and self.ctx.duplicates[oid] == other:
                continue
            if not self.obj(other).absent():
                return False
        return True

    def summary(self, oids: Iterable[str]) -> Dict[str, Dict]:
        return {oid: self.obj(oid).to_json() for oid in oids}
