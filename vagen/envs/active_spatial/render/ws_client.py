# WebSocket client for render server communication
# Adapted from ViewSuite's service/ws_client.py

import json
import uuid
import websockets
from typing import Optional, Dict, Any, Tuple, List, Union


class WSClient:
    """
    Simple WebSocket client for communicating with render servers.
    Sends JSON requests and receives JSON + binary responses.
    """
    
    def __init__(
        self, 
        url: str = "ws://127.0.0.1:8765/ws",
        origin: Optional[str] = None, 
        **ws_kwargs
    ):
        self.url = url
        self.origin = origin
        self.ws = None
        self.ws_kwargs = ws_kwargs

    async def connect(self):
        """Establish WebSocket connection."""
        addl = self.ws_kwargs.pop("additional_headers", None)
        legacy = self.ws_kwargs.pop("extra_headers", None)
        headers = []
        if addl:
            headers.extend(addl if isinstance(addl, (list, tuple)) else [addl])
        if legacy:
            headers.extend(legacy if isinstance(legacy, (list, tuple)) else [legacy])

        self.ws = await websockets.connect(
            self.url,
            origin=self.origin,
            additional_headers=headers or None,
            **self.ws_kwargs,
        )

    async def close(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def request(
        self, 
        op: str, 
        payload: Optional[Dict[str, Any]] = None, 
        binary: Optional[bytes] = None,
        extra: Optional[Dict[str, Any]] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Optional[bytes]]:
        """
        Send a request and receive response.
        
        Args:
            op: Operation name
            payload: JSON payload
            binary: Optional binary data
            extra: Extra fields to include in request
            metadata: Additional metadata
            
        Returns:
            Tuple of (response_json, response_binary)
        """
        meta = {
            "req_id": str(uuid.uuid4()), 
            "op": op, 
            "payload": payload or {}
        }
        if extra:
            meta.update(extra)
        if metadata:
            meta.update(metadata)
            
        await self.ws.send(json.dumps(meta))
        await self.ws.send(binary if binary is not None else b"")
        
        resp_json = json.loads(await self.ws.recv())
        msg2 = await self.ws.recv()
        resp_bin = bytes(msg2) if isinstance(msg2, (bytes, bytearray)) else None
        
        return resp_json, (resp_bin or None)
