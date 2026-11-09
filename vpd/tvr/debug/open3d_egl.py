"""V9 open3d_egl (optional): a full engine on EGL as a positive control.  JOB ONLY.

probe-spatial env (open3d 0.19, Filament, EGL headless through glvnd libEGL.so.1 + the staged NVIDIA vendor
library): OffscreenRenderer 320x240, a 1 m box with computed normals, 'defaultLit', background (0.2, 0.4, 0.6),
a sun light, camera looking at the box.  PASS = RGB std >= 2 (an empty or failed render is a uniform colour).
Also: the centre pixel, mean colour, the fraction of non-background pixels, rgb.png.
"""
from __future__ import annotations

import os
import time

from . import common

BACKGROUND = (0.2, 0.4, 0.6, 1.0)


def run(out: str, width: int = 320, height: int = 240) -> dict:
    res = common.new_result("open3d_egl")
    m = res["metrics"]
    res["env"] = {k: os.environ.get(k) for k in ("LD_LIBRARY_PATH", "__EGL_VENDOR_LIBRARY_FILENAMES", "EGL_PLATFORM")}
    t0 = time.time()
    try:
        res["stage"] = "import"
        import numpy as np
        import open3d as o3d
        m["open3d_version"] = o3d.__version__
        res["stage"] = "renderer"
        r = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        box.compute_vertex_normals()
        box.paint_uniform_color([0.8, 0.2, 0.2])
        r.scene.add_geometry("box", box, mat)
        r.scene.set_background(list(BACKGROUND))
        try:
            r.scene.scene.set_sun_light([-0.5, -1.0, -0.3], [1.0, 1.0, 1.0], 75000)
            r.scene.scene.enable_sun_light(True)
        except Exception as e:  # noqa: BLE001
            res["notes"].append(f"sun light: {e!r}")
        r.setup_camera(60.0, [0.5, 0.5, 0.5], [2.2, 1.8, 2.6], [0.0, 1.0, 0.0])
        res["stage"] = "render"
        t = time.time()
        img = np.asarray(r.render_to_image())
        m["render_ms"] = round(1000 * (time.time() - t), 2)
        rgb = img[..., :3].astype(np.float64)
        m["shape"] = list(img.shape)
        m["std"] = round(float(rgb.std()), 3)
        m["mean_rgb"] = [round(float(x), 2) for x in rgb.reshape(-1, 3).mean(axis=0)]
        m["centre"] = [int(x) for x in img[height // 2, width // 2, :3]]
        bg = np.array([round(255 * c) for c in BACKGROUND[:3]], dtype=np.float64)
        m["frac_not_background"] = round(float((np.abs(rgb - bg).max(axis=-1) > 6).mean()), 4)
        try:
            from PIL import Image
            Image.fromarray(img[..., :3].astype(np.uint8)).save(os.path.join(out, "rgb.png"))
        except Exception as e:  # noqa: BLE001
            res["notes"].append(f"png: {e!r}")
        res["pass"] = m["std"] >= 2.0
        res["stage"] = "done"
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
        res["notes"].append(f"exception at stage {res['stage']}: {e!r}"[:600])
    m["probe_s"] = round(time.time() - t0, 3)
    m["mapped_gpu_libs"] = common.mapped_libs()
    return res
