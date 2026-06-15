# Local Gaussian Splatting Renderer
# Adapted from ViewSuite's interiorGS/render/gs_render.py

from typing import Optional
import numpy as np
import torch
from PIL import Image
import os
import threading

# Module-level lock to serialize the first CUDA call (torch.inverse lazy init)
# to avoid "lazy wrapper should be called at most once" race in PyTorch 2.8.0
_CUDA_PREWARM_LOCK = threading.Lock()
_CUDA_PREWARM_DONE: set = set()  # tracks devices already pre-warmed


def _prewarm_cuda_inverse(device: str) -> None:
    """Thread-safe one-time pre-warm of torch.inverse and cuSolver CUDA lazy wrappers.

    gsplat's rasterization C extension calls cusolverDnCreate on first use. Like
    torch.inverse, cuSolver has a lazy initialization wrapper that must be called
    from a single thread to avoid "CUSOLVER_STATUS_INTERNAL_ERROR". We trigger
    torch.linalg.svd here to force cusolverDnCreate to complete before any
    concurrent gsplat rasterization calls.
    """
    if device in _CUDA_PREWARM_DONE:
        return
    with _CUDA_PREWARM_LOCK:
        if device in _CUDA_PREWARM_DONE:
            return
        _dummy = torch.eye(4, device=device).unsqueeze(0)
        torch.inverse(_dummy)
        # Pre-warm cuSolver (used internally by gsplat rasterization C extension)
        torch.linalg.svd(torch.eye(4, device=device))
        torch.cuda.synchronize(device)
        _CUDA_PREWARM_DONE.add(device)


class GaussianRenderer:
    """
    Local Gaussian Splatting renderer using gsplat.
    
    This class wraps gsplat for local GPU rendering.
    For distributed training with multiple GPUs, it's recommended to use 
    render_backend="client" instead, which offloads rendering to a dedicated GPU server.
    
    Note: This requires gsplat and ply_gaussian_loader libraries to be installed.
    """
    
    # Class variable to control logging
    VERBOSE = os.environ.get('GS_RENDERER_VERBOSE', '0') == '1'
    
    def __init__(self, ply_path: str, use_sh: bool = True, gpu_device: Optional[int] = None):
        """
        Initialize the renderer with a PLY file.
        
        Args:
            ply_path: Path to the Gaussian Splatting PLY file
            use_sh: Whether to use spherical harmonics for colors
            gpu_device: GPU device ID (None = auto-detect from environment or use cuda:0)
        """
        self.ply_path = ply_path
        self.use_sh = use_sh
        
        # Determine device - MUST detect at runtime, not from config
        # (config may be created offline where CUDA is not available)
        
        # Debug: Print CUDA availability (only if verbose)
        cuda_available = torch.cuda.is_available()
        if self.VERBOSE:
            print(f"[GaussianRenderer] DEBUG: torch.cuda.is_available() = {cuda_available}")
            print(f"[GaussianRenderer] DEBUG: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            print(f"[GaussianRenderer] DEBUG: gpu_device parameter = {gpu_device}")
            print(f"[GaussianRenderer] DEBUG: LOCAL_RANK = {os.environ.get('LOCAL_RANK', 'not set')}")
        
        if cuda_available:
            if gpu_device is not None:
                # Explicitly specified GPU device
                self.device = f'cuda:{gpu_device}'
            else:
                # Auto-detect: try LOCAL_RANK env var first (for distributed training)
                local_rank = os.environ.get('LOCAL_RANK', None)
                if local_rank is not None:
                    self.device = f'cuda:{local_rank}'
                else:
                    # Use current CUDA device
                    current_device = torch.cuda.current_device()
                    self.device = f'cuda:{current_device}'
                    if self.VERBOSE:
                        print(f"[GaussianRenderer] Auto-detected CUDA device: {current_device}")
        else:
            # No CUDA available - will use CPU (slow!)
            self.device = 'cpu'
            print("[GaussianRenderer] WARNING: CUDA not available, using CPU (very slow!)")
            if self.VERBOSE:
                print(f"[GaussianRenderer] DEBUG: PyTorch CUDA build version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'unknown'}")
            
        if self.VERBOSE:
            print(f"[GaussianRenderer] Using device: {self.device}")
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Gaussian Splatting model from PLY file."""
        try:
            # Import required libraries
            from ply_gaussian_loader import PLYGaussianLoader
            from gsplat.rendering import rasterization
            
            if self.VERBOSE:
                print(f"[GaussianRenderer] Loading model from {self.ply_path}")
            
            # Load PLY data
            loader = PLYGaussianLoader()
            self.gs_data = loader.load_ply(self.ply_path)
            
            # Prepare data for rendering
            (
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                self.sh_degree
            ) = self._prepare_data(self.gs_data, self.device, self.use_sh)
            
            if self.VERBOSE:
                print(f"[GaussianRenderer] Loaded {len(self.means)} Gaussians on {self.device}")
            
            # Pre-warm PyTorch CUDA lazy wrappers (thread-safe, once per device).
            # Multiple threads may call _load_model() concurrently; the lock ensures
            # only one thread triggers the torch.inverse lazy init at a time,
            # avoiding the "lazy wrapper should be called at most once" race.
            if 'cuda' in self.device:
                _prewarm_cuda_inverse(self.device)
            
        except ImportError as e:
            raise ImportError(
                f"Local Gaussian Splatting rendering requires gsplat and ply_gaussian_loader libraries.\n"
                f"Install with:\n"
                f"  pip install gsplat\n"
                f"  pip install ply_gaussian_loader\n"
                f"Or use render_backend='client' to render on a remote GPU server.\n"
                f"Error: {e}"
            )
    
    def _prepare_data(self, gs_data, device, use_sh=True):
        """
        Convert Gaussian splat data to PyTorch tensors for rendering.
        
        Args:
            gs_data: Loaded Gaussian data from PLY file
            device: PyTorch device (cuda/cpu)
            use_sh: Whether to use spherical harmonics
            
        Returns:
            Tuple of prepared tensors (means, quats, scales, opacities, colors, sh_degree)
        """
        # Convert positions
        means = torch.from_numpy(gs_data.positions).float().to(device)

        # Get rotations in XYZW format (normalized)
        quats = torch.from_numpy(gs_data.get_rotations_xyzw()).float().to(device)

        # Convert scales (from log space)
        scales_log = torch.from_numpy(gs_data.scales).float().to(device)
        scales = torch.exp(scales_log)

        # Convert opacities from logit to probability space
        opacities_logit = torch.from_numpy(gs_data.opacities.squeeze()).float().to(device)
        opacities = torch.sigmoid(opacities_logit)

        # Handle colors/SH coefficients
        sh_degree = None
        if use_sh and hasattr(gs_data, 'sh_rest') and gs_data.sh_rest is not None:
            # Use spherical harmonics
            sh_coeffs = gs_data.get_sh_coefficients()
            colors = torch.from_numpy(sh_coeffs).float().to(device)
            sh_degree = gs_data.sh_bands if hasattr(gs_data, 'sh_bands') else 3
        else:
            # Use RGB colors (fallback)
            colors_rgb = gs_data.get_linear_colors()
            colors = torch.from_numpy(colors_rgb).float().to(device)

        return means, quats, scales, opacities, colors, sh_degree
    
    def _prepare_camera_data(self, camera_intrinsics, camera_extrinsics):
        """
        Prepare camera parameters for rendering.
        
        Args:
            camera_intrinsics: 3x3 or 4x4 intrinsic matrix
            camera_extrinsics: 4x4 world-to-camera transformation matrix (w2c)
            
        Returns:
            Tuple of (K_tensor, viewmat) - intrinsics and view matrix as tensors
        """
        # Convert to numpy arrays
        K_np = np.asarray(camera_intrinsics, dtype=np.float32)
        E_np = np.asarray(camera_extrinsics, dtype=np.float32)
        
        # Ensure K is 3x3
        if K_np.shape == (4, 4):
            K_np = K_np[:3, :3]
        elif K_np.shape != (3, 3):
            raise ValueError(f"Intrinsics must be 3x3 or 4x4; got {K_np.shape}")
            
        # View Matrix is the world-to-camera matrix (extrinsics is already w2c)
        viewmat_np = E_np

        # Convert to PyTorch tensors
        K_tensor = torch.from_numpy(K_np).to(self.device).unsqueeze(0)  # (1, 3, 3)
        viewmat = torch.from_numpy(viewmat_np).to(self.device).unsqueeze(0)  # (1, 4, 4)
        
        return K_tensor, viewmat
    
    def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 512,
        height: int = 512
    ) -> np.ndarray:
        """
        Render an image from camera parameters.
        
        Args:
            camera_intrinsics: 3x3 or 4x4 camera intrinsic matrix
            camera_extrinsics: 4x4 world-to-camera extrinsic matrix (w2c)
            width: Output image width
            height: Output image height
            
        Returns:
            Rendered image as numpy array (H, W, 3) uint8 [0-255]
        """
        from gsplat.rendering import rasterization
        
        # Prepare camera parameters
        K_tensor, viewmat = self._prepare_camera_data(camera_intrinsics, camera_extrinsics)
        
        # Render using gsplat
        render_colors, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmat,
            Ks=K_tensor,
            width=width,
            height=height,
            sh_degree=self.sh_degree,
            packed=False
        )

        # Post-process: convert to numpy and uint8
        rendered_image = render_colors[0].cpu().numpy()
        rendered_image = np.clip(rendered_image, 0, 1)
        rendered_image_uint8 = (rendered_image * 255).astype(np.uint8)
        
        return rendered_image_uint8

