from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .pool import InteriorGSWorkerPool


def _parse_gpu_ids(raw: Sequence[int] | str | None) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    return [int(x) for x in raw]


def _parse_forced_size(raw: Sequence[int] | int | str | None) -> Optional[Tuple[int, int]]:
    if raw is None or raw == "" or raw == "None":
        return None
    if isinstance(raw, int):
        return (raw, raw)
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.replace("x", ",").split(",") if x.strip()]
        if len(parts) == 1:
            v = int(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        raise ValueError(f"Invalid forced_render_size: {raw!r}")
    vals = list(raw)
    if len(vals) != 2:
        raise ValueError("forced_render_size must contain exactly two integers")
    return (int(vals[0]), int(vals[1]))


class InteriorGSRenderHandler:
    """HTTP handler for InteriorGS camera renders."""

    def __init__(
        self,
        *,
        gs_root: str,
        max_workers: int = 4,
        gpu_ids: Sequence[int] | str | None = None,
        forced_render_size: Sequence[int] | int | str | None = None,
        image_format: str = "PNG",
        image_quality: int | None = None,
    ):
        if not gs_root:
            raise ValueError("gs_root is required")
        self.pool = InteriorGSWorkerPool(
            max_workers=max_workers,
            gs_root=gs_root,
            gpu_ids=_parse_gpu_ids(gpu_ids),
            forced_render_size=_parse_forced_size(forced_render_size),
            image_format=image_format,
            image_quality=image_quality,
        )
        self.image_format = image_format.upper()

    async def handle(self, meta: Dict[str, Any], images: list[Any] | None = None) -> Dict[str, Any]:
        scene_id = meta.get("scene_id")
        if not scene_id:
            raise ValueError("meta.scene_id is required")
        tasks = meta.get("tasks") or []
        encoded_images = await self.pool.render(scene_id, tasks)
        return {
            "meta": {"ok": True, "scene_id": scene_id, "count": len(encoded_images), "image_format": self.image_format, "metrics": self.pool.metrics()},
            "encoded_images": encoded_images,
        }

    async def aclose(self) -> None:
        await self.pool.aclose()
