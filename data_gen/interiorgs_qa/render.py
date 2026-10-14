"""Renderers: the 3DGS scene on a GPU (RGB, depth, opacity) and a box proxy for CPU tests.

``GSRenderer`` wraps the env's local renderer (``render_depth_from_cam_param``: gsplat "RGB+ED", the
expected depth along the camera's viewing axis in metres, and the accumulated opacity).  ``ProxyRenderer``
ray-casts the labelled boxes, the structure walls and the floor; it only exists so the generators can be
tested without a GPU and is never used for data.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np

from vagen.envs.active_spatial import target_visibility as TV

from .geom import OBB, camera_K, pixel_rays

View = Tuple[np.ndarray, np.ndarray, np.ndarray]  # rgb (H, W, 3) uint8, depth (H, W) float32, alpha (H, W)


class GSRenderer:
    def __init__(self, gs_root: str, gpu_device: int = 0) -> None:
        from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

        self.gs_root = gs_root
        self.gpu_device = gpu_device
        self._unified: Optional[UnifiedRenderGS] = None
        self.scene_id: Optional[str] = None
        self.kind = "gs"

    def set_scene(self, scene_id: str) -> None:
        from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

        if self._unified is None:
            self._unified = UnifiedRenderGS(render_backend="local", gs_root=self.gs_root, scene_id=scene_id,
                                            gpu_device=self.gpu_device)
        self._unified.set_scene(scene_id)
        self.scene_id = scene_id

    @property
    def gaussians(self):
        """The loaded GaussianRenderer (means, quats, scales, opacities, colors tensors)."""
        return self._unified._ensure_local()

    def render(self, c2w: np.ndarray, size: int, K: Optional[np.ndarray] = None) -> View:
        K = camera_K(size) if K is None else K
        w2c = np.linalg.inv(np.asarray(c2w, float))
        rgb, depth, alpha = self.gaussians.render_depth_from_cam_param(K, w2c, size, size)
        return rgb, depth, alpha


class ProxyRenderer:
    """CPU stand-in: depth of the nearest labelled box, structure wall or floor along each pixel ray."""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.kind = "proxy"
        self.boxes = [(oid, b) for oid, b in ctx.obb.items() if not TV.is_non_occluding(ctx.by_id[oid].label)]
        occ = ctx.occ
        self.walls = [(occ.wall_a[k], occ.wall_b[k], occ.wall_z0[k], occ.wall_z1[k]) for k in range(len(occ.wall_a))]

    def set_scene(self, scene_id: str) -> None:
        pass

    def render(self, c2w: np.ndarray, size: int, K: Optional[np.ndarray] = None) -> View:
        K = camera_K(size) if K is None else K
        vs, us = np.mgrid[0:size, 0:size]
        d = pixel_rays(c2w, K, us, vs).reshape(-1, 3)
        o = np.asarray(c2w, float)[:3, 3]
        depth = np.full(len(d), np.inf)
        ident = np.full(len(d), -1, dtype=np.int64)
        for k, (oid, b) in enumerate(self.boxes):
            tn, tf = b.ray_interval(o, d)
            hit = (tf >= np.maximum(tn, 0.05)) & (tn < depth)
            t = np.where(tn > 0.05, tn, tf)
            upd = hit & (t < depth)
            depth[upd] = t[upd]
            ident[upd] = k
        for k, (a, bpt, z0, z1) in enumerate(self.walls):
            e = bpt - a
            denom = d[:, 0] * e[1] - d[:, 1] * e[0]
            diff = a - o[:2]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (diff[0] * e[1] - diff[1] * e[0]) / denom
                u = (diff[0] * d[:, 1] - diff[1] * d[:, 0]) / denom
            z = o[2] + t * d[:, 2]
            hit = (np.abs(denom) > 1e-12) & (t > 0.05) & (u >= 0) & (u <= 1) & (z >= z0) & (z <= z1) & (t < depth)
            depth[hit] = t[hit]
            ident[hit] = 100000 + k
        with np.errstate(divide="ignore", invalid="ignore"):
            tfl = -o[2] / d[:, 2]
        hit = (d[:, 2] < 0) & (tfl > 0.05) & (tfl < depth)
        depth[hit] = tfl[hit]
        ident[hit] = 200000
        alpha = np.isfinite(depth).astype(np.float32)
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        rgb = np.zeros((len(d), 3), np.uint8)
        for v in np.unique(ident):
            h = hashlib.md5(str(v).encode()).digest()
            rgb[ident == v] = (np.frombuffer(h[:3], np.uint8) // 2 + 64) if v >= 0 else 0
        shade = np.clip(1.2 - depth / 10.0, 0.3, 1.0)[:, None]
        rgb = (rgb * shade).astype(np.uint8)
        return rgb.reshape(size, size, 3), depth.reshape(size, size), alpha.reshape(size, size)


class CachedRenderer:
    """Memoise renders per (scene, pose, size) so a view shared by several questions renders once."""

    def __init__(self, base) -> None:
        self.base = base
        self.cache: Dict[Tuple, View] = {}
        self.renders = 0
        self.kind = getattr(base, "kind", "?")

    def set_scene(self, scene_id: str) -> None:
        self.cache.clear()
        self.base.set_scene(scene_id)

    def render(self, c2w: np.ndarray, size: int) -> View:
        key = (size,) + tuple(np.round(np.asarray(c2w, float)[:3, :4], 5).ravel())
        if key not in self.cache:
            self.cache[key] = self.base.render(c2w, size)
            self.renders += 1
        return self.cache[key]
