# Unified Gaussian Splatting Renderer
# Adapted from ViewSuite's interiorGS/unified_renderer.py
#
# This provides a unified interface for rendering via:
# - "local": Direct GPU rendering using GaussianRenderer (requires gsplat)
# - "client": Remote rendering via WebSocket to a GPU server

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np

from .gs_render_client import GSRenderClient


@dataclass
class RenderConfig:
    """Configuration for the unified renderer."""
    render_backend: str                  # "local" | "client"
    gs_root: Optional[str] = None        # Required for local rendering
    client_url: Optional[str] = None     # Required for client rendering
    client_origin: Optional[str] = None  # Optional origin header
    scene_id: Optional[str] = None       # Current scene
    gpu_device: Optional[int] = None     # GPU device ID for local rendering


def _to_jsonable(x: Any) -> Any:
    """Convert numpy arrays and other types to JSON-serializable format."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _ensure_K3x3(K: np.ndarray) -> np.ndarray:
    """Ensure camera intrinsics is 3x3."""
    K = np.asarray(K)
    if K.shape == (4, 4):
        return K[:3, :3]
    if K.shape == (3, 3):
        return K
    raise ValueError(f"Intrinsics must be 3x3 or 4x4; got {K.shape}")


class UnifiedRenderGS:
    """
    Unified renderer for Gaussian Splatting scenes.
    
    Supports two backends:
    - "local": Direct GPU rendering (requires gsplat library)
    - "client": Remote rendering via WebSocket server
    
    Usage:
        renderer = UnifiedRenderGS(
            render_backend="client",
            client_url="ws://gpu-server:8777/render/interiorgs"
        )
        renderer.set_scene("scene_001")
        image = await renderer.render_image_from_cam_param(K, E, 512, 512)
    """

    def __init__(
        self,
        render_backend: str,
        gs_root: Optional[str] = None,
        client_url: Optional[str] = None,
        client_origin: Optional[str] = None,
        scene_id: Optional[str] = None,
        gpu_device: Optional[int] = None,
    ):
        """
        Initialize the unified renderer.
        
        Args:
            render_backend: "local" or "client"
            gs_root: Root directory containing {scene_id}.ply files (for local)
            client_url: WebSocket URL of render server (for client)
            client_origin: Optional origin header for WebSocket
            scene_id: Initial scene ID
            gpu_device: GPU device ID for local rendering (None = auto-detect)
        """
        self.cfg = RenderConfig(
            render_backend, gs_root, client_url, client_origin, scene_id, gpu_device
        )
        self._gs_renderer = None  # GaussianRenderer for local mode
        self._client: Optional[GSRenderClient] = None
        self._ply: Optional[str] = None

    async def close(self) -> None:
        """Close connections and release resources."""
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None

    def set_scene(self, scene_id: str) -> None:
        """
        Switch to a different scene.
        
        Args:
            scene_id: New scene identifier
        """
        if scene_id != self.cfg.scene_id:
            self.cfg.scene_id = scene_id
            self._ply = None
            self._gs_renderer = None

    def _ensure_ply(self) -> str:
        """Get the .ply path for current scene."""
        if self._ply is None:
            if self.cfg.gs_root is None or self.cfg.scene_id is None:
                raise ValueError("gs_root and scene_id are required for local rendering")
            
            # Try multiple possible PLY file locations
            import os
            
            # Check for verbose logging
            verbose = os.environ.get('GS_RENDERER_VERBOSE', '0') == '1'
            
            candidates = [
                f"{self.cfg.gs_root}/{self.cfg.scene_id}/3dgs_compressed.ply",  # InteriorGS format
                f"{self.cfg.gs_root}/{self.cfg.scene_id}.ply",                  # Flat format
                f"{self.cfg.gs_root}/{self.cfg.scene_id}/gaussian.ply",         # Alternative name
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate):
                    self._ply = candidate
                    if verbose:
                        print(f"[UnifiedRenderGS] Found PLY file: {self._ply}")
                    break
            
            if self._ply is None:
                raise FileNotFoundError(
                    f"Could not find PLY file for scene {self.cfg.scene_id}. "
                    f"Tried: {candidates}"
                )
        
        return self._ply

    def _ensure_local(self):
        """Ensure local GaussianRenderer is initialized."""
        if self._gs_renderer is None:
            try:
                # Import gsplat-based renderer (only when needed)
                from .gs_render_local import GaussianRenderer
                self._gs_renderer = GaussianRenderer(
                    self._ensure_ply(),
                    gpu_device=self.cfg.gpu_device
                )
            except ImportError as e:
                raise ImportError(
                    f"Local rendering requires gsplat library. "
                    f"Install it or use render_backend='client'. Error: {e}"
                )
        return self._gs_renderer

    async def _ensure_client(self) -> GSRenderClient:
        """Ensure WebSocket client is connected."""
        if self._client is None:
            if not self.cfg.client_url:
                raise ValueError("client_url is required for client rendering")
            self._client = GSRenderClient(
                url=self.cfg.client_url, 
                origin=self.cfg.client_origin, 
                session_id="unified-gs"
            )
            await self._client.connect()
        return self._client

    @staticmethod
    def _to_pil(img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Convert to PIL Image if needed."""
        return img if isinstance(img, Image.Image) else Image.fromarray(img)

    async def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 300,
        height: int = 300
    ) -> Image.Image:
        """
        Render an image from camera parameters.
        
        Args:
            camera_intrinsics: 3x3 or 4x4 camera intrinsic matrix
            camera_extrinsics: 4x4 world-to-camera extrinsic matrix
            width: Output image width
            height: Output image height
            
        Returns:
            Rendered PIL Image
        """
        if self.cfg.render_backend == "local":
            renderer = self._ensure_local()
            img = renderer.render_image_from_cam_param(
                camera_intrinsics, camera_extrinsics, width, height
            )
            return self._to_pil(img)

        elif self.cfg.render_backend == "client":
            client = await self._ensure_client()
            K = _ensure_K3x3(np.asarray(camera_intrinsics, dtype=np.float32))
            E = np.asarray(camera_extrinsics, dtype=np.float32)

            tasks = [{
                "mode": "cam_param",
                "intrinsics": _to_jsonable(K),
                "extrinsics": _to_jsonable(E),
                "size": [int(width), int(height)]
            }]
            imgs = await client.render(self.cfg.scene_id, tasks)
            
            if not imgs:
                raise RuntimeError(f"Render server returned no images for scene {self.cfg.scene_id}")
            return imgs[0]

        else:
            raise ValueError(f"Unknown render_backend: {self.cfg.render_backend}")

    async def render_tasks(self, tasks: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        Render multiple images from a list of tasks.
        
        Args:
            tasks: List of render tasks, each containing:
                - intrinsics: Camera intrinsic matrix
                - extrinsics: Camera extrinsic matrix
                - size: [width, height] (optional, default [300, 300])
                
        Returns:
            List of rendered PIL Images
        """
        if self.cfg.render_backend == "local":
            out: List[Image.Image] = []
            renderer = self._ensure_local()
            for t in tasks:
                w, h = t.get("size", [300, 300])
                out.append(
                    self._to_pil(renderer.render_image_from_cam_param(
                        t["intrinsics"], t["extrinsics"], w, h
                    ))
                )
            return out
        else:
            client = await self._ensure_client()
            return await client.render(self.cfg.scene_id, tasks)
