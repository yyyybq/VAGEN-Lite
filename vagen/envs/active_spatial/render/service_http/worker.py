from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ACTIVE_SCENE_ID: Optional[str] = None
_ACTIVE_RENDERER: Any = None
_ACTIVE_PLY: Optional[str] = None
_BOUND_GPU_ID: Optional[int] = None
_FORCED_RENDER_SIZE: Optional[Tuple[int, int]] = None


def _bind_process_to_gpu(gpu_id: Optional[int]) -> None:
    global _BOUND_GPU_ID
    if gpu_id is None or _BOUND_GPU_ID == gpu_id:
        return
    # The launcher has already narrowed CUDA_VISIBLE_DEVICES. Select the
    # worker's logical device explicitly; changing CUDA_VISIBLE_DEVICES after
    # CUDA initialization leaves every spawned worker on logical device 0.
    import torch

    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"renderer gpu_id={gpu_id} is outside the visible logical "
                f"device range [0, {torch.cuda.device_count()})"
            )
        torch.cuda.set_device(gpu_id)
    _BOUND_GPU_ID = gpu_id


def _release_renderer() -> None:
    global _ACTIVE_RENDERER, _ACTIVE_SCENE_ID, _ACTIVE_PLY
    renderer = _ACTIVE_RENDERER
    if renderer is not None:
        release = getattr(renderer, "release", None) or getattr(renderer, "close", None)
        if callable(release):
            release()
    _ACTIVE_RENDERER = None
    _ACTIVE_SCENE_ID = None
    _ACTIVE_PLY = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ensure_scene_loaded(scene_id: str, gs_root: str, gpu_id: Optional[int]):
    global _ACTIVE_SCENE_ID, _ACTIVE_RENDERER, _ACTIVE_PLY
    if _ACTIVE_SCENE_ID == scene_id and _ACTIVE_RENDERER is not None:
        return _ACTIVE_RENDERER

    _release_renderer()
    _bind_process_to_gpu(gpu_id)

    from vagen.envs.active_spatial import gs_io
    from vagen.envs.active_spatial.render.gs_render_local import GaussianRenderer

    ply_path = gs_io.resolve_ply(gs_root, scene_id)
    renderer = GaussianRenderer(ply_path, gpu_device=gpu_id)
    _ACTIVE_SCENE_ID = scene_id
    _ACTIVE_PLY = ply_path
    _ACTIVE_RENDERER = renderer
    return renderer


def _to_image_bytes(image: Any, image_format: str = "PNG", image_quality: Optional[int] = None) -> bytes:
    import io
    from PIL import Image

    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    save_kwargs = {}
    fmt = image_format.upper()
    if image_quality is not None and fmt in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = int(image_quality)
    if fmt == "JPEG":
        save_kwargs.setdefault("subsampling", 0)
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _render_images_worker(
    scene_id: str,
    tasks: List[Dict[str, Any]],
    gs_root: str,
    gpu_id: Optional[int],
    forced_render_size: Optional[Tuple[int, int]] = None,
    image_format: str = "PNG",
    image_quality: Optional[int] = None,
) -> List[bytes]:
    renderer = _ensure_scene_loaded(scene_id, gs_root, gpu_id)
    out: List[bytes] = []
    for task in tasks:
        if task.get("mode", "cam_param") != "cam_param":
            raise ValueError(f"Unsupported render mode: {task.get('mode')}")
        width, height = task.get("size", [256, 256])
        if forced_render_size is not None:
            width, height = forced_render_size
        image = renderer.render_image_from_cam_param(
            np.asarray(task["intrinsics"], dtype=np.float32),
            np.asarray(task["extrinsics"], dtype=np.float32),
            int(width),
            int(height),
        )
        out.append(_to_image_bytes(image, image_format=image_format, image_quality=image_quality))
    return out
