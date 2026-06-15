# Gaussian Splatting Render Client
# Adapted from ViewSuite's interiorGS/service/gs_render_client.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import uuid
from PIL import Image

from .ws_client import WSClient
from .binary_utils import decode_image_list


class GSRenderClient:
    """
    WebSocket client for Gaussian Splatting render server.
    
    Sends render requests with camera parameters and receives rendered images.
    Protocol:
      - Sends JSON: {"op": "render", "payload": {"scene_id": ..., "tasks": [...]}}
      - Receives JSON + binary frame (length-prefixed PNG list)
    """
    
    def __init__(
        self, 
        url: str, 
        session_id: Optional[str] = None, 
        origin: Optional[str] = None, 
        **ws_kwargs
    ):
        """
        Initialize the render client.
        
        Args:
            url: WebSocket URL of the render server
            session_id: Optional session identifier
            origin: Optional origin header for WebSocket
            **ws_kwargs: Additional WebSocket arguments
        """
        self.url = url
        self.session_id = session_id or f"render-{uuid.uuid4()}"
        self.ws = WSClient(url, origin=origin, **ws_kwargs)
        self.connected = False

    async def connect(self):
        """Connect to the render server."""
        if not self.connected:
            await self.ws.connect()
            self.connected = True

    async def close(self):
        """Close the connection."""
        if self.connected:
            await self.ws.close()
            self.connected = False

    async def render(
        self, 
        scene_id: str, 
        tasks: List[Dict[str, Any]]
    ) -> List[Image.Image]:
        """
        Render images for the given scene and camera tasks.
        
        Args:
            scene_id: Scene identifier (e.g., "scene0011_00")
            tasks: List of render tasks, each containing:
                - mode: "cam_param"
                - intrinsics: 3x3 camera intrinsic matrix
                - extrinsics: 4x4 camera extrinsic matrix (world-to-camera)
                - size: [width, height]
                
        Returns:
            List of rendered PIL Images
        """
        if not self.connected:
            await self.connect()
            
        resp_json, resp_bin = await self.ws.request(
            op="render",
            payload={"scene_id": scene_id, "tasks": tasks},
            binary=None,
            extra={"session_id": self.session_id},
        )
        
        # Decode PNG images from binary response
        return decode_image_list(resp_bin or b"")
