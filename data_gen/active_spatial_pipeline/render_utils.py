"""
Shared Rendering Utilities for Active Spatial Pipeline
=======================================================

This module provides a unified interface for rendering using the UnifiedRenderGS
renderer from the VAGEN environment. It is designed to be reused across different
rendering scripts (render_path_images.py, render_init_target_views.py, etc.)

Usage:
    from render_utils import RenderConfig, SceneRenderer
    
    config = RenderConfig(
        gs_root="/path/to/InteriorGS",
        render_backend="local",
    )
    
    async with SceneRenderer(config) as renderer:
        await renderer.set_scene("scene_id")
        image = await renderer.render_image(intrinsics, extrinsics_c2w)
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class RenderConfig:
    """Configuration for scene rendering."""
    gs_root: str = None                # Root directory for Gaussian Splatting scenes
    render_backend: str = "local"      # "local" or "client"
    client_url: str = None             # WebSocket URL for client mode
    image_width: int = 640
    image_height: int = 480
    gpu_device: int = None             # GPU device ID (None = auto)


def _run_async(coro):
    """Helper to run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class SceneRenderer:
    """
    Unified scene renderer using UnifiedRenderGS.
    
    This class wraps the UnifiedRenderGS from VAGEN environment and provides
    a simplified interface for rendering images from camera parameters.
    
    The renderer handles:
    - PLY file discovery within gs_root/scene_id/
    - Scene switching
    - c2w to w2c conversion
    - Async rendering
    """
    
    def __init__(self, config: RenderConfig):
        self.config = config
        self.renderer = None
        self._initialized = False
        self._current_scene = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _ensure_initialized(self, scene_id: str):
        """Initialize or reinitialize renderer for a scene."""
        if self._initialized and self._current_scene == scene_id:
            return
        
        # Add VAGEN env path for imports
        vagen_env_path = Path(__file__).parent.parent.parent / "VAGEN" / "vagen" / "env" / "active_spatial"
        if str(vagen_env_path) not in sys.path:
            sys.path.insert(0, str(vagen_env_path.parent.parent.parent))
        
        try:
            from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
            
            self.renderer = UnifiedRenderGS(
                render_backend=self.config.render_backend,
                gs_root=self.config.gs_root,
                client_url=self.config.client_url,
                scene_id=scene_id,
                gpu_device=self.config.gpu_device,
            )
            self._current_scene = scene_id
            self._initialized = True
            print(f"[SceneRenderer] Initialized renderer for scene {scene_id}")
            
        except ImportError as e:
            print(f"[SceneRenderer] Error importing UnifiedRenderGS: {e}")
            print("[SceneRenderer] Make sure gsplat is installed for local rendering")
            raise
    
    async def set_scene(self, scene_id: str):
        """Switch to a different scene."""
        if not self._initialized:
            await self._ensure_initialized(scene_id)
        elif self._current_scene != scene_id:
            if self.renderer is not None:
                self.renderer.set_scene(scene_id)
            self._current_scene = scene_id
            print(f"[SceneRenderer] Switched to scene {scene_id}")
    
    async def render_image(self, intrinsics: np.ndarray, extrinsics_c2w: np.ndarray) -> Optional[Image.Image]:
        """
        Render an image from camera parameters.
        
        Args:
            intrinsics: 3x3 or 4x4 camera intrinsic matrix
            extrinsics_c2w: 4x4 camera-to-world extrinsic matrix
            
        Returns:
            Rendered PIL Image, or None on failure
        """
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized. Call set_scene first.")
        
        # Ensure 3x3 intrinsics
        if intrinsics.shape == (4, 4):
            intrinsics = intrinsics[:3, :3]
        
        # Convert c2w to w2c for renderer
        w2c = np.linalg.inv(extrinsics_c2w)
        
        try:
            return await self.renderer.render_image_from_cam_param(
                camera_intrinsics=intrinsics,
                camera_extrinsics=w2c,
                width=self.config.image_width,
                height=self.config.image_height,
            )
        except Exception as e:
            print(f"[SceneRenderer] Render error: {e}")
            return None
    
    async def close(self):
        """Close the renderer and release resources."""
        if self.renderer is not None:
            try:
                await self.renderer.close()
            except Exception:
                pass
            self.renderer = None
            self._initialized = False
            self._current_scene = None


def look_at_matrix(camera_pos: np.ndarray, target_pos: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    Create camera extrinsic matrix (camera-to-world) with camera looking at target.
    Uses OpenCV/COLMAP convention: X=right, Y=down, Z=forward
    
    Args:
        camera_pos: [x, y, z] camera position in world coordinates
        target_pos: [x, y, z] target position to look at
        up: [x, y, z] up vector (default: [0, 0, 1])
        
    Returns:
        4x4 camera-to-world transformation matrix
    """
    if up is None:
        up = np.array([0, 0, 1])
    
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    
    forward = target_pos - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0, 0, 1])
    else:
        forward = forward / forward_norm
    
    # OpenCV convention
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    
    return c2w


def compute_intrinsics(width: int, height: int, fov_deg: float = 90.0) -> np.ndarray:
    """
    Compute camera intrinsic matrix from image size and field of view.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_deg: Horizontal field of view in degrees
        
    Returns:
        3x3 intrinsic matrix
    """
    fov_rad = np.radians(fov_deg)
    focal_length = width / (2 * np.tan(fov_rad / 2))
    
    cx = width / 2
    cy = height / 2
    
    return np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
