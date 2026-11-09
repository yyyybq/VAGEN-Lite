"""Shared helpers of the vpd-tvr-debug probes (stdlib only)."""
from __future__ import annotations

import json
import os
import re
import time

# V1 / V2 pixel rule (spec): 256x256 target cleared to (0.2, 0.4, 0.6), one opaque red triangle over the centre.
SIZE = 256
CLEAR = (0.2, 0.4, 0.6, 1.0)
CLEAR_RGB = (51, 102, 153)  # round(255 * CLEAR) for UNORM8
RED_RGB = (255, 0, 0)
CENTRE_TOL = 3
CORNER_TOL = 2
GPU_LIB_RE = re.compile(r"(nvidia|vulkan|lvp|LLVM|libEGL|libGLX|libGLdispatch|libOpenGL|libGL\.so|libcuda)")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path: str, obj) -> None:
    tmp = f"{path}.part{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def new_result(variant: str, **kw) -> dict:
    """The common result.json shape: {variant, pass, stage, metrics, notes, utc} (+ rc / secs from finalize)."""
    res = {"variant": variant, "pass": False, "stage": "start", "metrics": {}, "notes": [], "utc": utc()}
    res.update(kw)
    return res


def close(px, expect, tol) -> bool:
    """Every channel of px within +-tol of expect (expect may be shorter than px: RGB of an RGBA pixel)."""
    if px is None or len(px) < len(expect):
        return False
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(px, expect))


def pixel_at(buf, width: int, x: int, y: int, bpp: int = 4):
    i = (y * width + x) * bpp
    return [int(v) for v in bytes(buf[i:i + bpp])]


def pixel_verdict(buf, width: int = SIZE, height: int = SIZE) -> dict:
    """Centre and the four corners of a tightly packed RGBA8 image (row order does not matter for these)."""
    cx, cy = width // 2, height // 2
    centre = pixel_at(buf, width, cx, cy)
    corners = {name: pixel_at(buf, width, x, y) for name, (x, y) in
               {"x0y0": (0, 0), "x1y0": (width - 1, 0), "x0y1": (0, height - 1),
                "x1y1": (width - 1, height - 1)}.items()}
    out = {"centre": centre, "corners": corners,
           "centre_ok": close(centre, RED_RGB, CENTRE_TOL),
           "corners_ok": all(close(p, CLEAR_RGB, CORNER_TOL) for p in corners.values())}
    data = bytes(buf[:width * height * 4])
    out["all_zero"] = not any(data)
    # how many pixels are red (the triangle, base 1.2 x height 1.2 in NDC, covers 0.72 / 4 = 18 % of the target)
    red = 0
    for i in range(0, len(data), 4):
        if data[i] >= 250 and data[i + 1] <= 5 and data[i + 2] <= 5:
            red += 1
    out["red_fraction"] = round(red / float(width * height), 4)
    out["ok"] = out["centre_ok"] and out["corners_ok"]
    return out


def mapped_libs(pid="self", pattern=GPU_LIB_RE) -> list:
    """Sorted unique file paths of shared objects mapped into a process that match `pattern`."""
    seen = set()
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6:
                    p = parts[5].strip()
                    if ".so" in p and pattern.search(os.path.basename(p)):
                        seen.add(p)
    except OSError:
        return []
    return sorted(seen)


def save_rgba_png(path: str, buf, width: int, height: int) -> str | None:
    """PNG of an RGBA8 buffer; returns an error string when PIL is missing or fails, else None."""
    try:
        from PIL import Image
        Image.frombuffer("RGBA", (width, height), bytes(buf[:width * height * 4]), "raw", "RGBA", 0, 1).save(path)
    except Exception as e:  # noqa: BLE001
        return repr(e)
    return None
