"""EGL probe (eglinfo-like, ctypes only) for the TVR render smoke job (WP-TVR, plan section 3.7).

Ported from the WP-TVR design draft (scratchpad vpd_tvr/design/egl_probe.py).  JOB ONLY: never run it on the
login node.  glvnd libEGL.so.1 (OUT/tvr/nvgl_595.71.05/glvnd) dispatches to the NVIDIA vendor library named by
__EGL_VENDOR_LIBRARY_FILENAMES.

1. eglQueryDevicesEXT: number of devices; per device EGL_DRM_DEVICE_FILE_EXT and EGL_CUDA_DEVICE_NV.
2. Device 0: eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT), eglInitialize, vendor / version / client APIs.
3. 256x256 RGBA8 + depth-24 pbuffer, desktop OpenGL context, make current, GL_VENDOR / GL_RENDERER / GL_VERSION,
   glClearColor(0.2, 0.4, 0.6, 1) + glClear + glReadPixels -> expect about (51, 102, 153, 255); saves clear.png.
Own pass rule = >=1 device, eglInitialize ok, EGL vendor NVIDIA, GL_RENDERER contains "H800", readback within
+-2; the gate itself is decided by verdict.py.  Output OUT/result.json with the stage reached ('stage').
This is the path MuJoCo (MUJOCO_GL=egl), Habitat-sim and Open3D's headless renderer need.
"""
import argparse
import ctypes as C
import json
import os
import sys

EGL_NONE, EGL_VENDOR, EGL_VERSION, EGL_EXTENSIONS, EGL_CLIENT_APIS = 0x3038, 0x3053, 0x3054, 0x3055, 0x308D
EGL_PLATFORM_DEVICE_EXT, EGL_DRM_DEVICE_FILE_EXT, EGL_CUDA_DEVICE_NV = 0x313F, 0x3233, 0x323A
EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT = 0x3033, 0x1, 0x3040, 0x8
EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE, EGL_DEPTH_SIZE = 0x3024, 0x3023, 0x3022, 0x3021, 0x3025
EGL_WIDTH, EGL_HEIGHT, EGL_OPENGL_API = 0x3057, 0x3056, 0x30A2
GL_VENDOR, GL_RENDERER, GL_VERSION, GL_COLOR_BUFFER_BIT, GL_RGBA, GL_UNSIGNED_BYTE = (
    0x1F00, 0x1F01, 0x1F02, 0x4000, 0x1908, 0x1401)
EXPECT = (51, 102, 153, 255)
SIZE = 256


class _Stop(Exception):
    """Ends the probe at a stage; the result so far is written."""


def attribs(*xs):
    return (C.c_int * (len(xs) + 1))(*xs, EGL_NONE)


def readback_ok(px, expect=EXPECT, tol=2):
    return isinstance(px, list) and len(px) == 4 and all(abs(x - y) <= tol for x, y in zip(px, expect))


def probe(out, cdll=None):
    """Run the probe; returns the result dict (stage = last stage reached).  `cdll` replaces ctypes.CDLL."""
    cdll = cdll or C.CDLL
    res = {"env": {k: os.environ.get(k) for k in ("__EGL_VENDOR_LIBRARY_FILENAMES", "LD_LIBRARY_PATH")},
           "pass": False, "stage": "start"}
    try:
        _probe(out, cdll, res)
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    return res


def _probe(out, cdll, res):
    def stop(stage, **kw):
        res.update(kw)
        res["stage"] = stage
        raise _Stop()

    try:
        egl = cdll("libEGL.so.1")
    except OSError as e:
        stop("load_libEGL", error=repr(e))
    egl.eglGetProcAddress.restype = C.c_void_p
    egl.eglGetProcAddress.argtypes = [C.c_char_p]
    egl.eglQueryString.restype = C.c_char_p
    egl.eglQueryString.argtypes = [C.c_void_p, C.c_int]
    egl.eglGetError.restype = C.c_int

    def proc(name, restype, *argtypes):
        p = egl.eglGetProcAddress(name.encode())
        return C.CFUNCTYPE(restype, *argtypes)(p) if p else None

    res["client_extensions"] = (egl.eglQueryString(None, EGL_EXTENSIONS) or b"").decode()
    qdev = proc("eglQueryDevicesEXT", C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))
    qdstr = proc("eglQueryDeviceStringEXT", C.c_char_p, C.c_void_p, C.c_int)
    qdatt = proc("eglQueryDeviceAttribEXT", C.c_uint, C.c_void_p, C.c_int, C.POINTER(C.c_ssize_t))
    getpd = proc("eglGetPlatformDisplayEXT", C.c_void_p, C.c_int, C.c_void_p, C.POINTER(C.c_int))
    if not (qdev and getpd):
        stop("procs", error="EGL_EXT_device_enumeration / platform_base missing (vendor library not loaded?)")
    devs = (C.c_void_p * 16)()
    n = C.c_int(0)
    qdev(16, devs, C.byref(n))
    res["num_devices"] = n.value
    res["devices"] = []
    for i in range(n.value):
        d = {"index": i}
        if qdstr:
            s = qdstr(devs[i], EGL_DRM_DEVICE_FILE_EXT)
            d["drm_device_file"] = s.decode() if s else None
            s = qdstr(devs[i], EGL_EXTENSIONS)
            d["extensions"] = s.decode() if s else None
        if qdatt:
            v = C.c_ssize_t(-1)
            d["cuda_device"] = v.value if qdatt(devs[i], EGL_CUDA_DEVICE_NV, C.byref(v)) else None
        res["devices"].append(d)
    if n.value == 0:
        stop("query_devices", egl_error=hex(egl.eglGetError()))
    dpy = getpd(EGL_PLATFORM_DEVICE_EXT, devs[0], None)
    egl.eglInitialize.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int)]
    maj, mnr = C.c_int(), C.c_int()
    if not dpy or not egl.eglInitialize(dpy, C.byref(maj), C.byref(mnr)):
        stop("initialize", egl_error=hex(egl.eglGetError()))
    res["egl_version"] = f"{maj.value}.{mnr.value}"
    for k, v in (("EGL_VENDOR", EGL_VENDOR), ("EGL_VERSION", EGL_VERSION), ("EGL_CLIENT_APIS", EGL_CLIENT_APIS)):
        res[k] = (egl.eglQueryString(dpy, v) or b"").decode()
    cfg, ncfg = C.c_void_p(), C.c_int()
    egl.eglChooseConfig.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_void_p), C.c_int,
                                    C.POINTER(C.c_int)]
    egl.eglChooseConfig(dpy, attribs(EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                     EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                                     EGL_DEPTH_SIZE, 24), C.byref(cfg), 1, C.byref(ncfg))
    res["num_configs"] = ncfg.value
    if ncfg.value < 1:
        stop("choose_config", egl_error=hex(egl.eglGetError()))
    egl.eglCreatePbufferSurface.restype = C.c_void_p
    egl.eglCreatePbufferSurface.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]
    surf = egl.eglCreatePbufferSurface(dpy, cfg, attribs(EGL_WIDTH, SIZE, EGL_HEIGHT, SIZE))
    egl.eglBindAPI.argtypes = [C.c_uint]
    egl.eglBindAPI(EGL_OPENGL_API)
    egl.eglCreateContext.restype = C.c_void_p
    egl.eglCreateContext.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]
    ctx = egl.eglCreateContext(dpy, cfg, None, attribs())
    egl.eglMakeCurrent.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
    if not (surf and ctx and egl.eglMakeCurrent(dpy, surf, surf, ctx)):
        stop("context", egl_error=hex(egl.eglGetError()))
    try:
        gl = cdll("libOpenGL.so.0")  # glvnd GL dispatch; resolves into libnvidia-eglcore through the context
    except OSError as e:
        stop("load_libOpenGL", error=repr(e))
    gl.glGetString.restype = C.c_char_p
    gl.glGetString.argtypes = [C.c_uint]
    for k, v in (("GL_VENDOR", GL_VENDOR), ("GL_RENDERER", GL_RENDERER), ("GL_VERSION", GL_VERSION)):
        res[k] = (gl.glGetString(v) or b"").decode()
    gl.glClearColor.argtypes = [C.c_float] * 4
    gl.glClearColor(0.2, 0.4, 0.6, 1.0)
    gl.glClear(GL_COLOR_BUFFER_BIT)
    gl.glFinish()
    buf = (C.c_ubyte * (SIZE * SIZE * 4))()
    gl.glReadPixels(0, 0, SIZE, SIZE, GL_RGBA, GL_UNSIGNED_BYTE, buf)
    px = list(buf[0:4])
    res["readback_px0"] = px
    try:
        from PIL import Image
        Image.frombuffer("RGBA", (SIZE, SIZE), bytes(buf), "raw", "RGBA", 0, 1).save(
            os.path.join(out, "clear.png"))
    except Exception as e:  # noqa: BLE001
        res["png_error"] = repr(e)
    res["pass"] = ("NVIDIA" in res.get("EGL_VENDOR", "") and "H800" in res.get("GL_RENDERER", "")
                   and readback_ok(px))
    egl.eglTerminate.argtypes = [C.c_void_p]
    egl.eglTerminate(dpy)
    stop("readback")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = probe(a.out)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
