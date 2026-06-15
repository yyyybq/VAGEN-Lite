"""
Path Renderer for Active Spatial Navigation
============================================

This module renders images along generated navigation paths using the 
UnifiedRenderGS renderer from the VAGEN environment.

For each generated path, it renders:
1. The initial camera view
2. All intermediate steps
3. The final target view

Output Structure:
    output_dir/
        scene_id/
            path_{idx}/
                step_000.png  # Initial view
                step_001.png  # After first action
                ...
                step_N.png    # Final view (done)
                path_info.json  # Path metadata

Requirements:
- gsplat library for local rendering, OR
- WebSocket connection to render server for remote rendering
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
from dataclasses import dataclass
from tqdm import tqdm


def check_image_quality(img: Image.Image, 
                       min_color_variance: float = 100.0,
                       min_edge_density: float = 0.02,
                       uniform_color_threshold: float = 0.85) -> Tuple[bool, str]:
    """
    Check if an image has sufficient visual quality for VLM training.
    
    This detects images that are too close to walls (uniform color fills),
    too dark, or otherwise visually uninformative.
    
    Args:
        img: PIL Image to check
        min_color_variance: Minimum variance in color (0-255 scale)
        min_edge_density: Minimum ratio of edge pixels
        uniform_color_threshold: If this fraction of pixels have same color, reject
        
    Returns:
        (is_valid, rejection_reason)
    """
    # Convert to numpy array
    img_np = np.array(img)
    
    # Check 1: Color variance - detect uniform colored images (too close to wall)
    if len(img_np.shape) == 3:
        # Color image
        color_std = np.mean([np.std(img_np[:, :, c]) for c in range(3)])
    else:
        # Grayscale
        color_std = np.std(img_np)
    
    if color_std < min_color_variance:
        return False, f'low_color_variance_{color_std:.1f}'
    
    # Check 2: Detect if most pixels are the same color (wall fill)
    if len(img_np.shape) == 3:
        # Quantize colors to detect dominant color
        quantized = (img_np // 32) * 32  # Reduce to 8 levels per channel
        # Convert to single value per pixel
        pixel_codes = quantized[:, :, 0].astype(np.int32) * 65536 + \
                     quantized[:, :, 1].astype(np.int32) * 256 + \
                     quantized[:, :, 2].astype(np.int32)
        unique, counts = np.unique(pixel_codes, return_counts=True)
        max_ratio = counts.max() / counts.sum()
        
        if max_ratio > uniform_color_threshold:
            return False, f'uniform_color_{max_ratio:.2f}'
    
    # Check 3: Edge density - images with actual content have more edges
    try:
        from scipy import ndimage
        
        # Convert to grayscale for edge detection
        if len(img_np.shape) == 3:
            gray = np.mean(img_np, axis=2)
        else:
            gray = img_np.astype(float)
        
        # Sobel edge detection
        sx = ndimage.sobel(gray, axis=0, mode='constant')
        sy = ndimage.sobel(gray, axis=1, mode='constant')
        edges = np.hypot(sx, sy)
        
        # Count edge pixels (above threshold)
        edge_threshold = np.percentile(edges, 90)
        edge_pixels = np.sum(edges > edge_threshold)
        edge_density = edge_pixels / edges.size
        
        if edge_density < min_edge_density:
            return False, f'low_edge_density_{edge_density:.3f}'
            
    except ImportError:
        # scipy not available, skip edge check
        pass
    
    return True, 'ok'


@dataclass
class RenderConfig:
    """Configuration for path rendering."""
    render_backend: str = "local"      # "local" or "client"
    gs_root: str = None                # Root directory for Gaussian Splatting PLY files
    client_url: str = None             # WebSocket URL for client mode
    image_width: int = 640
    image_height: int = 480
    gpu_device: int = None             # GPU device ID (None = auto)
    max_paths_per_scene: int = None    # Limit number of paths to render
    # Quality filtering options
    check_init_quality: bool = True    # Check initial frame quality
    min_color_variance: float = 100.0  # Minimum color variance for quality check
    min_edge_density: float = 0.02     # Minimum edge density for quality check


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


class PathRenderer:
    """Render images along navigation paths."""
    
    def __init__(self, config: RenderConfig):
        self.config = config
        self.renderer = None
        self._initialized = False
        self._current_scene = None
    
    async def _init_renderer(self, scene_id: str = None):
        """Initialize the Gaussian Splatting renderer."""
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
            print(f"[PathRenderer] Initialized renderer for scene {scene_id}")
            
        except ImportError as e:
            print(f"[PathRenderer] Error importing renderer: {e}")
            print("[PathRenderer] Make sure gsplat is installed for local rendering")
            raise
    
    def set_scene(self, scene_id: str):
        """Switch to a different scene."""
        if self.renderer is not None:
            self.renderer.set_scene(scene_id)
            self._current_scene = scene_id
    
    async def render_image(self, intrinsics: np.ndarray, extrinsics: np.ndarray) -> Image.Image:
        """
        Render a single image from camera parameters.
        
        Args:
            intrinsics: 3x3 camera intrinsic matrix
            extrinsics: 4x4 camera-to-world extrinsic matrix
            
        Returns:
            Rendered PIL Image
        """
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized. Call _init_renderer first.")
        
        # Convert c2w to w2c for renderer
        w2c = np.linalg.inv(extrinsics)
        
        return await self.renderer.render_image_from_cam_param(
            camera_intrinsics=intrinsics,
            camera_extrinsics=w2c,
            width=self.config.image_width,
            height=self.config.image_height,
        )
    
    async def render_path(self, 
                          path_data: Dict[str, Any],
                          output_dir: Path,
                          render_every_n: int = 1) -> Dict[str, Any]:
        """
        Render all images along a navigation path.
        
        Args:
            path_data: Path data dictionary (from path_generator)
            output_dir: Directory to save rendered images
            render_every_n: Render every N steps (1 = all steps)
            
        Returns:
            Dictionary with render results (includes 'skipped' flag if quality check fails)
        """
        scene_id = path_data['scene_id']
        path_steps = path_data['path']
        intrinsics = np.array(path_data['init_camera']['intrinsics'])
        
        # Ensure 3x3 intrinsics
        if intrinsics.shape == (4, 4):
            intrinsics = intrinsics[:3, :3]
        
        # Initialize renderer for this scene
        await self._init_renderer(scene_id)
        self.set_scene(scene_id)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rendered_images = []
        init_quality_ok = True
        init_quality_reason = 'ok'
        
        # Render each step
        for i, step in enumerate(path_steps):
            if i % render_every_n != 0 and step['action'] != 'done':
                continue
            
            extrinsics = np.array(step['extrinsics'])
            
            try:
                img = await self.render_image(intrinsics, extrinsics)
                
                # Check quality of initial frame
                if step['step_idx'] == 0 and self.config.check_init_quality:
                    init_quality_ok, init_quality_reason = check_image_quality(
                        img,
                        min_color_variance=self.config.min_color_variance,
                        min_edge_density=self.config.min_edge_density,
                    )
                    if not init_quality_ok:
                        print(f"  [Quality Check] Initial frame failed: {init_quality_reason}")
                
                # Save image
                img_path = output_dir / f"step_{step['step_idx']:03d}.png"
                img.save(img_path)
                
                rendered_images.append({
                    'step_idx': step['step_idx'],
                    'action': step['action'],
                    'reward': step['reward'],
                    'image_path': str(img_path.relative_to(output_dir.parent.parent)),
                })
                
            except Exception as e:
                print(f"Error rendering step {step['step_idx']}: {e}")
                continue
        
        # Save path info
        info = {
            'data_idx': path_data['data_idx'],
            'scene_id': scene_id,
            'task_type': path_data['task_type'],
            'task_description': path_data['task_description'],
            'object_label': path_data['object_label'],
            'sample_target': path_data['sample_target'],
            'final_reward': path_data['final_reward'],
            'success': path_data['success'],
            'total_steps': path_data['total_steps'],
            'rendered_images': rendered_images,
            # Quality check results
            'init_quality_ok': init_quality_ok,
            'init_quality_reason': init_quality_reason,
        }
        
        with open(output_dir / 'path_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        return info
    
    async def close(self):
        """Close the renderer and release resources."""
        if self.renderer is not None:
            await self.renderer.close()
            self.renderer = None
            self._initialized = False


async def render_paths_for_scene(
    paths_jsonl: Path,
    output_dir: Path,
    config: RenderConfig,
    max_paths: int = None,
    render_every_n: int = 1,
) -> List[Dict[str, Any]]:
    """
    Render all paths from a generated paths file.
    
    Args:
        paths_jsonl: Path to generated paths JSONL file
        output_dir: Root output directory for rendered images
        config: Render configuration
        max_paths: Maximum number of paths to render
        render_every_n: Render every N steps
        
    Returns:
        List of render results
    """
    # Load paths
    paths = []
    with open(paths_jsonl, 'r') as f:
        for line in f:
            paths.append(json.loads(line.strip()))
    
    print(f"Loaded {len(paths)} paths")
    
    if max_paths:
        paths = paths[:max_paths]
    
    # Create renderer
    renderer = PathRenderer(config)
    results = []
    
    try:
        for i, path_data in enumerate(tqdm(paths, desc="Rendering paths")):
            scene_id = path_data['scene_id']
            path_idx = path_data['data_idx']
            
            path_output_dir = output_dir / scene_id / f"path_{path_idx:06d}"
            
            result = await renderer.render_path(
                path_data=path_data,
                output_dir=path_output_dir,
                render_every_n=render_every_n,
            )
            results.append(result)
            
    finally:
        await renderer.close()
    
    # Save summary
    quality_ok_count = sum(1 for r in results if r.get('init_quality_ok', True))
    quality_failed = [r for r in results if not r.get('init_quality_ok', True)]
    
    # Count rejection reasons
    rejection_reasons = {}
    for r in quality_failed:
        reason = r.get('init_quality_reason', 'unknown')
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    summary = {
        'total_paths': len(results),
        'successful_paths': sum(1 for r in results if r['success']),
        'avg_steps': float(np.mean([r['total_steps'] for r in results])),
        'avg_final_reward': float(np.mean([r['final_reward'] for r in results])),
        # Quality check stats
        'quality_check_passed': quality_ok_count,
        'quality_check_failed': len(quality_failed),
        'quality_rejection_reasons': rejection_reasons,
    }
    
    with open(output_dir / 'render_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nRender Summary:")
    print(f"  Total paths rendered: {summary['total_paths']}")
    print(f"  Successful paths: {summary['successful_paths']}")
    print(f"  Average steps: {summary['avg_steps']:.1f}")
    print(f"  Average final reward: {summary['avg_final_reward']:.4f}")
    print(f"  Quality check passed: {summary['quality_check_passed']}")
    print(f"  Quality check failed: {summary['quality_check_failed']}")
    if rejection_reasons:
        print(f"  Rejection reasons: {rejection_reasons}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Render images along navigation paths")
    parser.add_argument("--paths_jsonl", type=str, required=True,
                        help="Path to generated paths JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered images")
    parser.add_argument("--gs_root", type=str, required=True,
                        help="Root directory containing scene PLY files")
    parser.add_argument("--render_backend", type=str, default="local",
                        choices=["local", "client"],
                        help="Rendering backend")
    parser.add_argument("--client_url", type=str, default=None,
                        help="WebSocket URL for client rendering")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Output image width")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Output image height")
    parser.add_argument("--gpu_device", type=int, default=None,
                        help="GPU device ID for local rendering")
    parser.add_argument("--max_paths", type=int, default=None,
                        help="Maximum number of paths to render")
    parser.add_argument("--render_every_n", type=int, default=1,
                        help="Render every N steps (1 = all steps)")
    
    args = parser.parse_args()
    
    config = RenderConfig(
        render_backend=args.render_backend,
        gs_root=args.gs_root,
        client_url=args.client_url,
        image_width=args.image_width,
        image_height=args.image_height,
        gpu_device=args.gpu_device,
    )
    
    results = _run_async(render_paths_for_scene(
        paths_jsonl=Path(args.paths_jsonl),
        output_dir=Path(args.output_dir),
        config=config,
        max_paths=args.max_paths,
        render_every_n=args.render_every_n,
    ))
    
    print(f"\nRendered {len(results)} paths to {args.output_dir}")


if __name__ == "__main__":
    main()
