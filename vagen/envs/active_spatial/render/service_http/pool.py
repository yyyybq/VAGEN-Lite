from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .worker import _render_images_worker


@dataclass
class _WorkerSlot:
    executor: ProcessPoolExecutor
    gpu_id: Optional[int]
    current_scene: Optional[str] = None
    last_used: float = field(default_factory=time.monotonic)


class InteriorGSWorkerPool:
    """Fixed-size process pool with sticky scene routing and LRU eviction."""

    def __init__(
        self,
        *,
        max_workers: int,
        gs_root: str,
        gpu_ids: Sequence[int] | None = None,
        forced_render_size: Optional[Tuple[int, int]] = None,
        image_format: str = "PNG",
        image_quality: Optional[int] = None,
        crash_cooldown_s: float = 0.5,
    ):
        self.max_workers = max(1, int(max_workers))
        self.gs_root = gs_root
        self.gpu_ids: List[Optional[int]] = [int(g) for g in (gpu_ids or [])] or [None]
        self.forced_render_size = forced_render_size
        self.image_format = image_format.upper()
        self.image_quality = image_quality
        self.crash_cooldown_s = float(crash_cooldown_s)
        self._ctx = mp.get_context("spawn")
        self.worker_slots: List[_WorkerSlot] = [
            _WorkerSlot(self._create_executor(), self.gpu_ids[i % len(self.gpu_ids)])
            for i in range(self.max_workers)
        ]
        self.scene_to_worker: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._metrics: Counter[str] = Counter()

    def _create_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=1, mp_context=self._ctx)

    def _assign_worker_locked(self, scene_id: str) -> int:
        mapped = self.scene_to_worker.get(scene_id)
        if mapped is not None:
            return mapped
        for idx, slot in enumerate(self.worker_slots):
            if slot.current_scene is None:
                slot.current_scene = scene_id
                self.scene_to_worker[scene_id] = idx
                return idx
        idx = min(range(len(self.worker_slots)), key=lambda i: self.worker_slots[i].last_used)
        prev = self.worker_slots[idx].current_scene
        if prev:
            self.scene_to_worker.pop(prev, None)
        self.worker_slots[idx].current_scene = scene_id
        self.scene_to_worker[scene_id] = idx
        return idx

    async def render(self, scene_id: str, tasks: List[dict]) -> List[bytes]:
        async with self._lock:
            idx = self._assign_worker_locked(scene_id)
            slot = self.worker_slots[idx]
            slot.last_used = time.monotonic()
            executor = slot.executor
            gpu_id = slot.gpu_id
        loop = asyncio.get_running_loop()
        try:
            self._metrics["submit"] += 1
            return await loop.run_in_executor(
                executor,
                _render_images_worker,
                scene_id,
                tasks,
                self.gs_root,
                gpu_id,
                self.forced_render_size,
                self.image_format,
                self.image_quality,
            )
        except BrokenProcessPool:
            self._metrics["broken_pool"] += 1
            async with self._lock:
                slot = self.worker_slots[idx]
                if slot.current_scene:
                    self.scene_to_worker.pop(slot.current_scene, None)
                with contextlib.suppress(Exception):
                    slot.executor.shutdown(wait=True, cancel_futures=True)
            await asyncio.sleep(self.crash_cooldown_s)
            async with self._lock:
                slot = self.worker_slots[idx]
                slot.executor = self._create_executor()
                slot.current_scene = scene_id
                slot.last_used = time.monotonic()
                self.scene_to_worker[scene_id] = idx
                executor = slot.executor
                gpu_id = slot.gpu_id
            self._metrics["broken_pool_retry"] += 1
            return await loop.run_in_executor(
                executor,
                _render_images_worker,
                scene_id,
                tasks,
                self.gs_root,
                gpu_id,
                self.forced_render_size,
                self.image_format,
                self.image_quality,
            )

    async def warm(self, scene_ids: Sequence[str]) -> None:
        for scene_id in scene_ids:
            if scene_id:
                await self.render(scene_id, [])

    async def aclose(self) -> None:
        for slot in self.worker_slots:
            slot.current_scene = None
            with contextlib.suppress(Exception):
                slot.executor.shutdown(wait=True, cancel_futures=True)

    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)
