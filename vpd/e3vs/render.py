"""E3VS rendering: SceneSplat++ 3DGS scenes with gsplat 1.5.3, matched to the official E3VS renderer; plus a
CPU fake renderer for tests.

Official renderer (E3VS-BENCH @ bb7e603c, e3vs/environments/base.py:25-81 load_ply_model and :263-285
_render_frame): the PLY is read with plyfile; rot_0..rot_3 go to gsplat as stored (wxyz, normalised); opacity
sigmoid, scale exp, SH degree from the f_rest count, SH layout (N, K, 3); rasterization(..., packed=False,
render_mode "RGB") with gsplat's default near 0.01 / far 1e10, no background; colours clamped to [0, 1] and cast
with (x * 255).astype(uint8).  viewmat = inv(c2w), K = focal W/2, principal point at the centre.

SplatRenderer does the same from ply_gaussian_loader.PLYGaussianLoader().load_ply (the worktree-root module, which
every code snapshot carries, vpd.snapshot.EXTRA_FILES; the eval-external env has another copy; the one imported is
recorded as meta / parity code "ply_gaussian_loader": path, sha256): it passes
gs_data.rotations (WXYZ) and NEVER get_rotations_xyzw() (which reorders to xyzw; that is the possible quaternion
bug in vagen GaussianRenderer, gs_render_local.py:186, which is not used here).  render_depth uses render_mode
"RGB+ED" (expected depth along the camera axis, alpha-normalised by gsplat) and returns (depth, alpha).

FakeRenderer (numpy only, no GPU): ray-casts an axis-aligned room with optional solid boxes; RGB is a
deterministic texture of the hit point, depth/alpha are exact.  Used by the tests and by --renderer fake.

CLI (sco jobs: needs a GPU, the splats and the pinned repo):
    python -m vpd.e3vs.render --check-inputs                # pinned repo + all 99 PLYs present (no GPU)
    python -m vpd.e3vs.render --probe [--out FILE]          # B18: does gsplat JIT-compile here, and how long
    python -m vpd.e3vs.render --parity SCENE [--out DIR]    # official-loader parity (PSNR) + iPhone frames
Parity: our renderer vs the official load_ply_model + _render_frame (both executed from the pinned repo files,
read-only: load_ply_model is taken from base.py by ast because base.py imports open3d, which no env here has)
on the scene's E3VS viewpoints, PSNR per image (100.0 = identical).  Pass = min PSNR >= 40 dB (PREREG section 5,
E0).  Then, descriptively, renders at every k-th iPhone pose of InsScene's scene_iphone_metadata.npz with the
iPhone intrinsics and compares with the iPhone RGB (PSNR, SSIM) under both camera conventions: "opencv" (c2w as
stored: +x right, +y down, +z forward, as E3VS) and "opengl" (c2w @ diag(1, -1, -1, 1)); the better one is
recorded as iphone.best and used by occupancy.py (--convention auto).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import IMAGE_H, IMAGE_W, code_meta, repo_dir, sha256_file
from vpd.e3vs import geometry as G

PLY_REL = "ckpts/point_cloud_30000.ply"
NEAR, FAR = 0.01, 1e10  # gsplat 1.5.3 rasterization defaults, used by the official renderer
PARITY_MIN_PSNR = 40.0
PSNR_IDENTICAL = 100.0
INSSCENE_ROOT = Path(os.environ.get(
    "VPD_INSSCENE_ROOT",
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"))
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
CONVENTIONS = ("opencv", "opengl")


# ------------------------------------------------------------------------------------------------------------
# helpers (numpy only)
# ------------------------------------------------------------------------------------------------------------
def ply_path(scene_id: str) -> Path:
    return paths.E3VS_DATA / scene_id / PLY_REL


def intrinsics(W: int = IMAGE_W, H: int = IMAGE_H) -> np.ndarray:
    """The official pinhole K: focal W/2 (90 deg horizontal FOV), principal point at the image centre."""
    return G.viewmat_and_K(np.eye(4), W, H)[1]


def to_png(rgb: np.ndarray) -> bytes:
    """Lossless PNG bytes of an HxWx3 uint8 image (what the eval sends and the SFT data stores)."""
    from PIL import Image

    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {rgb.dtype} {rgb.shape}")
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def write_png(rgb: np.ndarray, path: Path) -> str:
    """Write the PNG atomically; returns its sha256."""
    data = to_png(rgb)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def convention_c2w(c2w, convention: str) -> np.ndarray:
    """The OpenCV-convention c2w (what gsplat and E3VS use) for a pose stored in `convention`."""
    c2w = np.asarray(c2w, dtype=np.float64)
    if convention == "opencv":
        return c2w.copy()
    if convention == "opengl":
        return c2w @ GL_TO_CV
    raise ValueError(f"unknown camera convention {convention!r}")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB of two uint8 images; PSNR_IDENTICAL (100) when they are equal."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mse = float(np.mean(d * d))
    if mse == 0.0:
        return PSNR_IDENTICAL
    return min(PSNR_IDENTICAL, 10.0 * math.log10(255.0 * 255.0 / mse))


def _box_mean(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window (valid region only) via integral images."""
    k = 2 * r + 1
    c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), axis=0), axis=1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / float(k * k)


def ssim(a: np.ndarray, b: np.ndarray, r: int = 3) -> float:
    """Mean SSIM of two uint8 RGB images on the luma channel, 7x7 uniform window (skimage's default)."""
    w = np.array([0.299, 0.587, 0.114])
    x = np.asarray(a, dtype=np.float64) @ w
    y = np.asarray(b, dtype=np.float64) @ w
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _box_mean(x, r), _box_mean(y, r)
    vx = _box_mean(x * x, r) - mx * mx
    vy = _box_mean(y * y, r) - my * my
    cxy = _box_mean(x * y, r) - mx * my
    n = (2 * r + 1) ** 2
    vx, vy, cxy = vx * n / (n - 1), vy * n / (n - 1), cxy * n / (n - 1)  # sample covariance, as skimage
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


def pixel_rays(W: int, H: int, K) -> np.ndarray:
    """(H, W, 3) camera-frame ray directions with z = 1 through the pixel centres (u + 0.5, v + 0.5)."""
    K = np.asarray(K, dtype=np.float64)
    u = (np.arange(W) + 0.5 - K[0, 2]) / K[0, 0]
    v = (np.arange(H) + 0.5 - K[1, 2]) / K[1, 1]
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv, np.ones_like(uu)], axis=-1)


# ------------------------------------------------------------------------------------------------------------
# fake renderer (tests, --renderer fake)
# ------------------------------------------------------------------------------------------------------------
class FakeRenderer:
    """Analytic scene: the inside of an axis-aligned `room` plus solid axis-aligned `boxes`.

    render_depth gives the exact z-depth (along the camera axis) of the first surface hit and alpha 1 there (0
    where nothing is hit, which only happens when room is None).  render_rgb gives a deterministic texture of
    the hit point (0.5 m checker x surface position), so images change with the pose.  numpy only.
    """

    name = "fake"

    def __init__(self, scene_id: str = "fake", room=((-20.0, -20.0, -5.0), (30.0, 30.0, 10.0)), boxes=()):
        self.scene_id = scene_id
        self.room = None if room is None else (np.asarray(room[0], float), np.asarray(room[1], float))
        self.boxes = [(np.asarray(b[0], float), np.asarray(b[1], float)) for b in boxes]
        self.meta = {"renderer": "fake", "scene_id": scene_id, "room": None if room is None else
                     [list(map(float, room[0])), list(map(float, room[1]))],
                     "boxes": [[list(map(float, b[0])), list(map(float, b[1]))] for b in boxes]}

    @staticmethod
    def _slab(o, d, lo, hi):
        """(t_near, t_far) of rays o + t d against the box [lo, hi] (inf/-inf safe)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1 = (lo - o) * inv
            t2 = (hi - o) * inv
        t1 = np.where(np.isnan(t1), -np.inf, t1)
        t2 = np.where(np.isnan(t2), np.inf, t2)
        tmin = np.minimum(t1, t2).max(axis=-1)
        tmax = np.maximum(t1, t2).min(axis=-1)
        return tmin, tmax

    def raycast(self, origin, dirs: np.ndarray) -> np.ndarray:
        """t of the first hit along o + t d (d not normalised) for each ray; inf where nothing is hit."""
        o = np.asarray(origin, dtype=np.float64)
        t = np.full(dirs.shape[:-1], np.inf)
        if self.room is not None:
            _, tmax = self._slab(o, dirs, *self.room)
            t = np.where(tmax > 0, tmax, t)
        for lo, hi in self.boxes:
            tmin, tmax = self._slab(o, dirs, lo, hi)
            hit = (tmax >= tmin) & (tmax > 0)
            enter = np.where(tmin > 0, tmin, 0.0)  # an origin inside a box hits at once
            t = np.where(hit & (enter < t), enter, t)
        return t

    def _hits(self, c2w, W, H, K):
        c2w = np.asarray(c2w, dtype=np.float64)
        K = intrinsics(W, H) if K is None else np.asarray(K, dtype=np.float64)
        rc = pixel_rays(W, H, K)
        dw = rc @ c2w[:3, :3].T
        t = self.raycast(c2w[:3, 3], dw)
        return c2w, dw, t

    def render_depth(self, c2w, W: int = IMAGE_W, H: int = IMAGE_H, K=None) -> tuple[np.ndarray, np.ndarray]:
        _, _, t = self._hits(c2w, W, H, K)
        ok = np.isfinite(t)
        return np.where(ok, t, 0.0).astype(np.float32), ok.astype(np.float32)

    def render_rgb(self, c2w, W: int = IMAGE_W, H: int = IMAGE_H, K=None) -> np.ndarray:
        c2w, dw, t = self._hits(c2w, W, H, K)
        ok = np.isfinite(t)
        p = c2w[:3, 3] + dw * np.where(ok, t, 0.0)[..., None]
        checker = (np.floor(p / 0.5).sum(axis=-1) % 2).astype(np.float64)
        frac = p - np.floor(p)
        rgb = 0.25 + 0.5 * frac * (0.6 + 0.4 * checker[..., None])
        rgb = np.where(ok[..., None], rgb, 0.0)
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------------
# gsplat renderer
# ------------------------------------------------------------------------------------------------------------
def _torch_threads():
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    return torch


_LOADER_FACTS: dict = {}


def loader_facts() -> dict:
    """Which ply_gaussian_loader this process imported: {"module", "path", "sha256", "in_worktree"}, from
    Path(module.__file__) (hashed once per path).  Since the worktree-root copy is in every code snapshot
    (vpd.snapshot.EXTRA_FILES), a job imports RUN/code/ply_gaussian_loader.py (in_worktree True) as live jobs imported
    the worktree's; without it the eval-external site-packages copy would load (in_worktree False)."""
    import ply_gaussian_loader as mod

    p = Path(mod.__file__).resolve()
    if str(p) not in _LOADER_FACTS:
        _LOADER_FACTS[str(p)] = {"module": "ply_gaussian_loader", "path": str(p), "sha256": sha256_file(p),
                                 "in_worktree": p.is_relative_to(paths.WORKTREE.resolve())}
    return dict(_LOADER_FACTS[str(p)])


def load_splat(ply, device: str = "cuda") -> dict:
    """Tensors of one 3DGS PLY as the official renderer builds them, read with ply_gaussian_loader.

    Returns {means, quats (wxyz, normalised), scales (exp), opacities (sigmoid, (N,)), colors ((N, K, 3) SH),
    sh_degree, n}.  The loader stores rotations as WXYZ (rot_0..rot_3); they are passed as stored.
    """
    torch = _torch_threads()
    import torch.nn.functional as F
    from ply_gaussian_loader import PLYGaussianLoader

    gs = PLYGaussianLoader().load_ply(str(ply))
    if gs.is_compressed:
        raise ValueError(f"{ply}: SuperSplat-compressed PLY; E3VS uses standard 3DGS PLYs")
    n = int(gs.num_splats)
    sh = gs.get_sh_coefficients()
    if sh is None:
        colors = gs.colors_dc.reshape(n, 1, 3).astype(np.float32)
        degree = 0
    else:
        colors = sh
        k = colors.shape[1]
        degree = int(round(math.sqrt(k))) - 1
        if (degree + 1) ** 2 != k or degree != int(gs.sh_bands):
            raise ValueError(f"{ply}: {k} SH coefficients per channel is not a full band set (loader {gs.sh_bands})")
    dev = torch.device(device)
    quats = torch.from_numpy(np.ascontiguousarray(gs.rotations)).float()  # WXYZ, never get_rotations_xyzw()
    return {
        "means": torch.from_numpy(np.ascontiguousarray(gs.positions)).float().to(dev),
        "quats": F.normalize(quats, p=2, dim=-1).to(dev),
        "scales": torch.exp(torch.from_numpy(np.ascontiguousarray(gs.scales)).float()).to(dev),
        "opacities": torch.sigmoid(torch.from_numpy(np.ascontiguousarray(gs.opacities)).float()).squeeze(-1).to(dev),
        "colors": torch.from_numpy(np.ascontiguousarray(colors)).float().to(dev),
        "sh_degree": degree, "n": n,
    }


class SplatRenderer:
    """gsplat 1.5.3 renderer of one scene (see module docstring)."""

    name = "gsplat"

    def __init__(self, scene_id: str | None = None, ply: Path | str | None = None, device: str = "cuda"):
        if ply is None:
            if scene_id is None:
                raise ValueError("SplatRenderer needs scene_id or ply")
            ply = ply_path(scene_id)
        self.scene_id = scene_id
        self.ply = Path(ply)
        if not self.ply.is_file():
            raise FileNotFoundError(f"{self.ply} missing (splats come from vpd/e3vs/download_splats.sh, user OK)")
        self.device = device
        t0 = time.time()
        self.g = load_splat(self.ply, device)
        self.meta = {"renderer": "gsplat", "scene_id": scene_id, "ply": str(self.ply), "n_gaussians": self.g["n"],
                     "sh_degree": self.g["sh_degree"], "load_seconds": round(time.time() - t0, 2),
                     "near": NEAR, "far": FAR, "packed": False, "quats": "wxyz (ply_gaussian_loader rotations)",
                     "ply_gaussian_loader": loader_facts()}

    def _raster(self, c2w, W: int, H: int, K, mode: str):
        import torch
        from gsplat.rendering import rasterization

        viewmat = torch.from_numpy(np.linalg.inv(np.asarray(c2w, dtype=np.float64))).float().to(self.device)
        K = intrinsics(W, H) if K is None else np.asarray(K, dtype=np.float64)
        Kt = torch.from_numpy(K).float().to(self.device)
        g = self.g
        with torch.no_grad():
            colors, alphas, _ = rasterization(
                g["means"], g["quats"], g["scales"], g["opacities"], g["colors"], viewmat[None], Kt[None],
                int(W), int(H), near_plane=NEAR, far_plane=FAR, sh_degree=g["sh_degree"], render_mode=mode,
                packed=False)
        return colors[0], alphas[0]

    def render_rgb(self, c2w, W: int = IMAGE_W, H: int = IMAGE_H, K=None) -> np.ndarray:
        colors, _ = self._raster(c2w, W, H, K, "RGB")
        return (colors[..., 0:3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def render_depth(self, c2w, W: int = IMAGE_W, H: int = IMAGE_H, K=None) -> tuple[np.ndarray, np.ndarray]:
        colors, alphas = self._raster(c2w, W, H, K, "RGB+ED")
        return (colors[..., 3].float().cpu().numpy().astype(np.float32),
                alphas[..., 0].float().cpu().numpy().astype(np.float32))


def make_renderer(kind: str, scene_id: str, device: str = "cuda"):
    if kind == "fake":
        return FakeRenderer(scene_id)
    if kind == "gsplat":
        return SplatRenderer(scene_id, device=device)
    raise ValueError(f"unknown renderer {kind!r} (gsplat | fake)")


def ply_sha256(scene_id: str) -> str | None:
    """sha256 of the scene's PLY: from DATA/SHA256SUMS (written by download_splats.sh) when listed, else hashed."""
    sums = paths.E3VS_DATA / "SHA256SUMS"
    rel = f"{scene_id}/{PLY_REL}"
    if sums.is_file():
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == rel:
                return parts[0]
    p = ply_path(scene_id)
    return sha256_file(p) if p.is_file() else None


# ------------------------------------------------------------------------------------------------------------
# official renderer (parity only; executed from the pinned repo files, read-only)
# ------------------------------------------------------------------------------------------------------------
def _load_by_path(name: str, path: Path):
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def official_loader(repo: Path | None = None):
    """The official load_ply_model function, taken from e3vs/environments/base.py by ast (base.py imports
    open3d at module level, which none of our envs has) and executed with its own imports."""
    torch = _torch_threads()
    import torch.nn.functional as F
    from plyfile import PlyData

    repo = Path(repo) if repo else repo_dir()
    src_path = repo / "e3vs" / "environments" / "base.py"
    src = src_path.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "load_ply_model")
    ns = {"np": np, "torch": torch, "F": F, "math": math, "Path": Path, "PlyData": PlyData,
          "logger": logging.getLogger("e3vs_official")}
    exec(compile(ast.get_source_segment(src, fn), str(src_path), "exec"), ns)  # noqa: S102 (pinned repo file)
    return ns["load_ply_model"]


def official_camera_geometry(repo: Path | None = None):
    repo = Path(repo) if repo else repo_dir()
    return _load_by_path("_e3vs_official_camera_geometry", repo / "e3vs" / "environments" / "camera_geometry.py")


class OfficialRenderer:
    """base.Environment3D._render_frame (lines 263-285) on load_ply_model's tensors."""

    def __init__(self, ply, device: str = "cuda", repo: Path | None = None):
        import torch

        self.device = torch.device(device)
        self.cg = official_camera_geometry(repo)
        (self.means, self.quats, self.scales, self.opacities, self.colors,
         self.sh_degree) = official_loader(repo)(Path(ply), self.device)

    def render_rgb(self, c2w, W: int = IMAGE_W, H: int = IMAGE_H) -> np.ndarray:
        import torch
        from gsplat.rendering import rasterization

        viewmat, K = self.cg.viewmat_and_K(np.asarray(c2w, dtype=np.float64), W, H, self.device)
        with torch.no_grad():
            render_colors, _, _ = rasterization(self.means, self.quats, self.scales, self.opacities, self.colors,
                                                viewmat[None], K[None], W, H, sh_degree=self.sh_degree,
                                                render_mode="RGB", packed=False)
        render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
        return (render_colors.cpu().numpy() * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------------
# B18 probe, parity
# ------------------------------------------------------------------------------------------------------------
def gsplat_probe(device: str = "cuda") -> dict:
    """Whether gsplat's CUDA extension is prebuilt, already JIT-built, or compiled now; and how long it took."""
    out = {"uid": os.getuid(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR"), "CUDA_HOME": os.environ.get("CUDA_HOME"),
           "MAX_JOBS": os.environ.get("MAX_JOBS")}
    t0 = time.time()
    torch = _torch_threads()
    import gsplat
    from torch.utils.cpp_extension import _get_build_directory

    out.update(torch=torch.__version__, gsplat=getattr(gsplat, "__version__", None),
               prebuilt_csrc=importlib.util.find_spec("gsplat.csrc") is not None)
    try:
        bdir = Path(_get_build_directory("gsplat_cuda", verbose=False))
    except Exception as e:  # noqa: BLE001
        bdir = None
        out["build_dir_error"] = repr(e)[:300]
    out["build_dir"] = None if bdir is None else str(bdir)
    out["so_before"] = bool(bdir and (bdir / "gsplat_cuda.so").exists())
    t1 = time.time()
    from gsplat.cuda import _backend

    out["extension_seconds"] = round(time.time() - t1, 1)
    out["extension_loaded"] = _backend._C is not None
    out["compiled_now"] = bool(out["extension_loaded"] and not out["prebuilt_csrc"] and not out["so_before"])
    if out["extension_loaded"] and device.startswith("cuda") and torch.cuda.is_available():
        from gsplat.rendering import rasterization

        g = torch.Generator().manual_seed(0)
        n = 64
        means = (torch.rand(n, 3, generator=g) - 0.5).cuda()
        means[:, 2] += 3.0
        quats = torch.nn.functional.normalize(torch.rand(n, 4, generator=g), dim=-1).cuda()
        scales = torch.full((n, 3), 0.05).cuda()
        opac = torch.full((n,), 0.8).cuda()
        cols = torch.rand(n, 3, generator=g).cuda()
        Kt = torch.from_numpy(intrinsics(64, 64)).float().cuda()
        t2 = time.time()
        img, _, _ = rasterization(means, quats, scales, opac, cols, torch.eye(4).cuda()[None], Kt[None], 64, 64,
                                  packed=False)
        torch.cuda.synchronize()
        out["first_render_seconds"] = round(time.time() - t2, 2)
        out["first_render_mean"] = float(img.mean())
    out["total_seconds"] = round(time.time() - t0, 1)
    return out


def iphone_metadata(scene_id: str, root: Path = INSSCENE_ROOT) -> dict:
    """InsScene's scene_iphone_metadata.npz: trajectories (N,4,4), intrinsics (N,3,3), image names (N,)."""
    d = Path(root) / scene_id
    z = np.load(d / "scene_iphone_metadata.npz", allow_pickle=False)
    names = [str(x) for x in z["images"]]
    return {"dir": d, "c2w": np.asarray(z["trajectories"], dtype=np.float64),
            "K": np.asarray(z["intrinsics"], dtype=np.float64), "images": names}


def scene_viewpoints(episodes: list[dict], scene_id: str) -> list[tuple[str, np.ndarray]]:
    """Distinct E3VS viewpoints (start, answerable, unanswerable) of the scene's episodes, deterministic order."""
    seen, out = set(), []
    for ep in sorted((e for e in episodes if e["scene_id"] == scene_id), key=lambda e: e["episode_index"]):
        cands = [("start", ep["start_c2w"])] + [(f"goal{v}", ep["goals"][v]) for v in sorted(ep["goals"])] + \
                [(f"unans{v}", ep["unans"][v]) for v in sorted(ep["unans"])]
        for tag, c in cands:
            key = tuple(np.round(np.asarray(c)[:3, :4], 4).reshape(-1).tolist())
            if key in seen:
                continue
            seen.add(key)
            out.append((f"ep{ep['episode_index']}_{tag}", np.asarray(c, dtype=np.float64)))
    return out


def parity(scene_id: str, out_dir: Path | None = None, n_poses: int = 27, iphone_every: int = 50,
           device: str = "cuda", repo: Path | None = None, save_samples: int = 3) -> dict:
    """Official-vs-ours PSNR on the scene's E3VS viewpoints, then the descriptive iPhone comparison."""
    from PIL import Image

    from vpd.e3vs import SPLITS, load_episodes

    repo = Path(repo) if repo else repo_dir()
    out_dir = Path(out_dir) if out_dir else paths.E3VS_PARITY
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = [e for sp in SPLITS for e in load_episodes(sp, repo) if e["scene_id"] == scene_id]
    if not eps:
        raise SystemExit(f"scene {scene_id} has no E3VS episode")
    vps = scene_viewpoints(eps, scene_id)[:n_poses]
    ours = SplatRenderer(scene_id, device=device)
    off = OfficialRenderer(ours.ply, device=device, repo=repo)
    rows = []
    for i, (tag, c2w) in enumerate(vps):
        a, b = ours.render_rgb(c2w), off.render_rgb(c2w)
        rows.append({"pose": tag, "psnr": round(psnr(a, b), 3), "max_abs": int(np.abs(a.astype(int) - b).max())})
        if i < save_samples:
            write_png(a, out_dir / f"{scene_id}_{tag}_ours.png")
            write_png(b, out_dir / f"{scene_id}_{tag}_official.png")
    del off
    ps = [r["psnr"] for r in rows]
    res = {"scene_id": scene_id, "ply": str(ours.ply), "ply_sha256": ply_sha256(scene_id), "renderer": ours.meta,
           "official": {"n": len(rows), "min_psnr": min(ps), "mean_psnr": round(float(np.mean(ps)), 3),
                        "threshold": PARITY_MIN_PSNR, "rows": rows},
           "pass": min(ps) >= PARITY_MIN_PSNR}
    try:
        md = iphone_metadata(scene_id)
        idx = list(range(0, len(md["c2w"]), max(1, iphone_every)))
        conv = {c: {"psnr": [], "ssim": []} for c in CONVENTIONS}
        for i in idx:
            with Image.open(md["dir"] / "images" / md["images"][i]) as im:
                ref = np.asarray(im.convert("RGB"))
            H, W = ref.shape[:2]
            for c in CONVENTIONS:
                img = ours.render_rgb(convention_c2w(md["c2w"][i], c), W, H, md["K"][i])
                conv[c]["psnr"].append(psnr(img, ref))
                conv[c]["ssim"].append(ssim(img, ref))
                if i == idx[0]:
                    write_png(img, out_dir / f"{scene_id}_iphone{i}_{c}.png")
        summ = {c: {"n": len(v["psnr"]), "psnr_mean": round(float(np.mean(v["psnr"])), 3),
                    "ssim_mean": round(float(np.mean(v["ssim"])), 4)} for c, v in conv.items()}
        best = max(CONVENTIONS, key=lambda c: (summ[c]["ssim_mean"], summ[c]["psnr_mean"]))
        res["iphone"] = {"metadata": str(md["dir"] / "scene_iphone_metadata.npz"), "every": iphone_every,
                         "frames": idx, **summ, "best": best, "note": "descriptive only (PREREG section 5)"}
    except Exception as e:  # noqa: BLE001  (descriptive part; recorded, never fatal)
        res["iphone"] = {"error": f"{type(e).__name__}: {e}"[:500]}
    res["code"] = code_meta(Path(__file__), Path(G.__file__))
    res["code"]["ply_gaussian_loader"] = loader_facts()  # the loader module that actually ran (path, sha256)
    res["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = out_dir / f"parity_{scene_id}.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("scene_id", "pass")} | {"min_psnr": res["official"]["min_psnr"],
                                                                 "iphone_best": res["iphone"].get("best")}))
    return res


def best_iphone_convention(parity_dir: Path | None = None) -> str | None:
    """The convention most parity files chose (None if there is no parity file with an iPhone result)."""
    d = Path(parity_dir) if parity_dir else paths.E3VS_PARITY
    votes = {}
    for p in sorted(d.glob("parity_*.json")) if d.is_dir() else []:
        b = (json.loads(p.read_text()).get("iphone") or {}).get("best")
        if b in CONVENTIONS:
            votes[b] = votes.get(b, 0) + 1
    return max(sorted(votes), key=lambda c: votes[c]) if votes else None


def check_inputs(repo: Path | None = None, data: Path | None = None) -> dict:
    """The pinned repo (commit, episode md5s) and a non-empty PLY for every E3VS scene; raises on a problem."""
    from vpd.e3vs import SPLITS, check_repo, load_episodes

    repo = Path(repo) if repo else repo_dir()
    inputs = check_repo(repo)
    if inputs["commit"] is None:
        raise AssertionError(f"{repo} is not a git checkout; the pinned commit cannot be verified")
    data = Path(data) if data else paths.E3VS_DATA
    scenes = sorted({e["scene_id"] for sp in SPLITS for e in load_episodes(sp, repo, check=False)})
    missing = [s for s in scenes if not (data / s / PLY_REL).is_file() or (data / s / PLY_REL).stat().st_size == 0]
    if missing:
        raise AssertionError(f"{len(missing)} of {len(scenes)} scenes lack {PLY_REL} under {data}: {missing[:5]}")
    return {**inputs, "n_scenes": len(scenes), "data": str(data),
            "sha256sums": (data / "SHA256SUMS").is_file()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", action="store_true", help="B18: gsplat JIT status and timing")
    g.add_argument("--parity", metavar="SCENE", help="parity of one scene (pass = min PSNR >= 40 dB)")
    g.add_argument("--check-inputs", action="store_true", help="pinned repo + every scene's PLY present")
    ap.add_argument("--out", type=Path, default=None, help="probe: JSON file; parity: directory")
    ap.add_argument("--n-poses", type=int, default=27)
    ap.add_argument("--iphone-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.check_inputs:
        print(json.dumps(check_inputs()))
        return 0
    if a.probe:
        res = gsplat_probe(a.device)
        print(json.dumps(res))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1) + "\n")
        return 0 if res.get("extension_loaded") else 1
    res = parity(a.parity, a.out, n_poses=a.n_poses, iphone_every=a.iphone_every, device=a.device)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
