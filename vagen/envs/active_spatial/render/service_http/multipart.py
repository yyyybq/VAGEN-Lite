from __future__ import annotations

import io
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs_remote.multipart_codec import decode_multipart as _decode_multipart


def decode_multipart(content_type: str, body: bytes) -> Tuple[Dict[str, Any], List[Image.Image]]:
    """Decode a render request/response body.

    The underlying decoder ignores form field names and accepts either the
    VAGEN GymService "data" field or ViewAgent's "meta" convention as long as
    the JSON part has content type application/json.
    """
    return _decode_multipart(content_type, body)


def encode_multipart(
    meta: Dict[str, Any],
    images: Optional[List[Image.Image]] = None,
    *,
    encoded_images: Optional[List[bytes]] = None,
    image_format: str = "PNG",
    image_mime: str = "image/png",
    boundary_prefix: str = "interiorgs_",
) -> Tuple[str, bytes]:
    """Encode JSON meta plus image parts as multipart/mixed bytes."""
    boundary = f"{boundary_prefix}{uuid.uuid4().hex}"
    crlf = b"\r\n"
    bnd = boundary.encode("utf-8")
    body = bytearray()

    meta_bytes = json.dumps(meta or {}, ensure_ascii=False).encode("utf-8")
    body += b"--" + bnd + crlf
    body += b'Content-Disposition: form-data; name="meta"' + crlf
    body += b"Content-Type: application/json; charset=utf-8" + crlf + crlf
    body += meta_bytes + crlf

    image_bytes_list: List[bytes] = []
    if encoded_images is not None:
        image_bytes_list = list(encoded_images)
    else:
        for img in images or []:
            buf = io.BytesIO()
            img.save(buf, format=image_format)
            image_bytes_list.append(buf.getvalue())

    for i, img_bytes in enumerate(image_bytes_list):
        body += b"--" + bnd + crlf
        body += f'Content-Disposition: form-data; name="images"; filename="{i}.png"'.encode("utf-8") + crlf
        body += f"Content-Type: {image_mime}".encode("utf-8") + crlf + crlf
        body += img_bytes + crlf

    body += b"--" + bnd + b"--" + crlf
    return boundary, bytes(body)
