# Binary encoding/decoding utilities for render service
# Adapted from ViewSuite's service/utils/service_binary_utils.py

import io
from typing import List
from PIL import Image
import numpy as np


def pack_length_prefixed(chunks: List[bytes]) -> bytes:
    """
    Pack multiple byte chunks with length prefixes.
    Format: [N][len1][data1]...[lenN][dataN] (uint32 big-endian)
    """
    out: List[bytes] = []
    out.append(len(chunks).to_bytes(4, "big"))
    for b in chunks:
        out.append(len(b).to_bytes(4, "big"))
        out.append(b)
    return b"".join(out)


def decode_image_list(blob: bytes) -> List[Image.Image]:
    """
    Decode length-prefixed PNG blob into list of PIL Images.
    Format: [N][len1][png1]...[lenN][pngN] (uint32 big-endian)
    """
    out: List[Image.Image] = []
    if not blob:
        return out
        
    mv, p = memoryview(blob), 0
    if len(mv) < 4:
        return out
        
    n = int.from_bytes(mv[p:p+4], "big")
    p += 4
    
    for _ in range(n):
        if len(mv) - p < 4:
            break
        ln = int.from_bytes(mv[p:p+4], "big")
        p += 4
        if ln < 0 or len(mv) - p < ln:
            break
        img = Image.open(io.BytesIO(mv[p:p+ln].tobytes()))
        img.load()
        out.append(img)
        p += ln
        
    return out


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array to PNG bytes."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
