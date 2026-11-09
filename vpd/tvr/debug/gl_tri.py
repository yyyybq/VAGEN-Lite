"""V1 gl_tri: EGL device pbuffer + desktop GL, GLSL 330 program, one red triangle with the depth test on.

JOB ONLY (it creates an NVIDIA GL context; never run it on the login node).  Extends vpd/tvr/egl_probe.py, which
only cleared and read back (no shader, no draw), so its EGL_GATE PASS proved no rasterisation.
  1. CDLL(--egl-lib) (glvnd libEGL.so.1; an absolute path, so a conda RPATH cannot pick another copy),
     eglQueryDevicesEXT, eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device --device), eglInitialize.
  2. Config RGBA8 + depth 24, PBUFFER, OPENGL; 256x256 pbuffer; eglBindAPI(OpenGL); context; make current.
  3. CDLL(--gl-lib) (glvnd libOpenGL.so.0); entry points by symbol, else eglGetProcAddress.  Vertex + fragment
     shaders (#version 330, positions from gl_VertexID: the same triangle as vk_tri), compile / link status and
     info logs saved (shader_logs.txt), a VAO bound, glEnable(GL_DEPTH_TEST) + GL_LESS, clear colour
     (0.2, 0.4, 0.6, 1) and depth 1, glDrawArrays(GL_TRIANGLES, 0, 3) at z = 0.5, glFinish, glReadPixels RGBA8.
PASS = compile and link OK, glGetError 0 after the draw and the readback, and the pixel rule of
common.pixel_verdict (centre within +-3 of (255, 0, 0), all four corners within +-2 of (51, 102, 153)).
Writes OUT/result.json at every stage (+ tri.png, shader_logs.txt).
"""
from __future__ import annotations

import ctypes as C
import os
import time

from . import common

EGL_NONE, EGL_VENDOR, EGL_VERSION, EGL_EXTENSIONS, EGL_CLIENT_APIS = 0x3038, 0x3053, 0x3054, 0x3055, 0x308D
EGL_PLATFORM_DEVICE_EXT, EGL_DRM_DEVICE_FILE_EXT, EGL_CUDA_DEVICE_NV = 0x313F, 0x3233, 0x323A
EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT = 0x3033, 0x1, 0x3040, 0x8
EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE, EGL_DEPTH_SIZE = 0x3024, 0x3023, 0x3022, 0x3021, 0x3025
EGL_WIDTH, EGL_HEIGHT, EGL_OPENGL_API = 0x3057, 0x3056, 0x30A2
GL_VENDOR, GL_RENDERER, GL_VERSION, GL_SHADING_LANGUAGE_VERSION = 0x1F00, 0x1F01, 0x1F02, 0x8B8C
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_LESS = 0x4000, 0x0100, 0x0B71, 0x0201
GL_RGBA, GL_UNSIGNED_BYTE, GL_TRIANGLES = 0x1908, 0x1401, 0x0004
GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS = 0x8B31, 0x8B30, 0x8B81, 0x8B82
GL_DEPTH_BITS = 0x0D56
GL_ERRORS = {0x0500: "GL_INVALID_ENUM", 0x0501: "GL_INVALID_VALUE", 0x0502: "GL_INVALID_OPERATION",
             0x0505: "GL_OUT_OF_MEMORY", 0x0506: "GL_INVALID_FRAMEBUFFER_OPERATION"}

VERT = b"""#version 330
void main() {
    vec2 p = vec2(0.0, -0.6);
    if (gl_VertexID == 1) p = vec2(0.6, 0.6);
    else if (gl_VertexID == 2) p = vec2(-0.6, 0.6);
    gl_Position = vec4(p, 0.5, 1.0);
}
"""
FRAG = b"""#version 330
out vec4 color;
void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
"""

c_int, c_uint, c_float, c_double, vp, cp = C.c_int, C.c_uint, C.c_float, C.c_double, C.c_void_p, C.c_char_p
# GL entry points: name -> (restype, argtypes)
GL_PROTOS = {
    "glGetString": (cp, [c_uint]),
    "glGetError": (c_uint, []),
    "glGetIntegerv": (None, [c_uint, C.POINTER(c_int)]),
    "glViewport": (None, [c_int, c_int, c_int, c_int]),
    "glCreateShader": (c_uint, [c_uint]),
    "glShaderSource": (None, [c_uint, c_int, C.POINTER(cp), C.POINTER(c_int)]),
    "glCompileShader": (None, [c_uint]),
    "glGetShaderiv": (None, [c_uint, c_uint, C.POINTER(c_int)]),
    "glGetShaderInfoLog": (None, [c_uint, c_int, C.POINTER(c_int), vp]),  # vp: takes the char buffer as is
    "glCreateProgram": (c_uint, []),
    "glAttachShader": (None, [c_uint, c_uint]),
    "glLinkProgram": (None, [c_uint]),
    "glGetProgramiv": (None, [c_uint, c_uint, C.POINTER(c_int)]),
    "glGetProgramInfoLog": (None, [c_uint, c_int, C.POINTER(c_int), vp]),
    "glUseProgram": (None, [c_uint]),
    "glGenVertexArrays": (None, [c_int, C.POINTER(c_uint)]),
    "glBindVertexArray": (None, [c_uint]),
    "glEnable": (None, [c_uint]),
    "glDepthFunc": (None, [c_uint]),
    "glClearDepth": (None, [c_double]),
    "glClearColor": (None, [c_float, c_float, c_float, c_float]),
    "glClear": (None, [c_uint]),
    "glDrawArrays": (None, [c_uint, c_int, c_int]),
    "glFinish": (None, []),
    "glReadPixels": (None, [c_int, c_int, c_int, c_int, c_uint, c_uint, vp]),
}


class _Stop(Exception):
    pass


def attribs(*xs):
    return (c_int * (len(xs) + 1))(*xs, EGL_NONE)


def run(out: str, egl_lib: str, gl_lib: str, device: int = 0, size: int = common.SIZE, cdll=None,
        on_stage=None) -> dict:
    res = common.new_result("gl_tri", egl_lib=egl_lib, gl_lib=gl_lib, device_arg=device)
    res["env"] = {k: os.environ.get(k) for k in ("__EGL_VENDOR_LIBRARY_FILENAMES", "__EGL_VENDOR_LIBRARY_DIRS",
                                                 "LD_LIBRARY_PATH")}
    t0 = time.time()
    try:
        _run(out, egl_lib, gl_lib, device, size, cdll or C.CDLL, res, on_stage)
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
        res["notes"].append(f"exception at stage {res['stage']}: {e!r}")
    res["metrics"]["probe_s"] = round(time.time() - t0, 3)
    res["metrics"]["mapped_gpu_libs"] = common.mapped_libs()
    return res


def gl_errors(fn, limit=8):
    errs = []
    for _ in range(limit):
        e = int(fn["glGetError"]())
        if e == 0:
            break
        errs.append(GL_ERRORS.get(e, hex(e)))
    return errs


def _run(out, egl_lib, gl_lib, device, size, cdll, res, on_stage):
    m = res["metrics"]

    def stage(name, **kw):
        res["stage"] = name
        res.update(kw)
        if on_stage is not None:
            on_stage(res)

    def stop(note):
        res["notes"].append(note)
        raise _Stop()

    stage("load_libEGL")
    try:
        egl = cdll(egl_lib)
    except OSError as e:
        stop(f"cannot load {egl_lib}: {e!r}")
    egl.eglGetProcAddress.restype = vp
    egl.eglGetProcAddress.argtypes = [cp]
    egl.eglQueryString.restype = cp
    egl.eglQueryString.argtypes = [vp, c_int]
    egl.eglGetError.restype = c_int

    def proc(name, restype, *argtypes):
        p = egl.eglGetProcAddress(name.encode())
        return C.CFUNCTYPE(restype, *argtypes)(p) if p else None

    def egl_err():
        return hex(egl.eglGetError())

    stage("query_devices")
    qdev = proc("eglQueryDevicesEXT", c_uint, c_int, C.POINTER(vp), C.POINTER(c_int))
    qdstr = proc("eglQueryDeviceStringEXT", cp, vp, c_int)
    getpd = proc("eglGetPlatformDisplayEXT", vp, c_int, vp, C.POINTER(c_int))
    if not (qdev and getpd):
        stop("EGL_EXT_device_enumeration / EGL_EXT_platform_base missing (vendor library not loaded?)")
    devs = (vp * 16)()
    n = c_int(0)
    qdev(16, devs, C.byref(n))
    m["num_devices"] = n.value
    m["devices"] = []
    for i in range(n.value):
        d = {"index": i}
        if qdstr:
            s = qdstr(devs[i], EGL_DRM_DEVICE_FILE_EXT)
            d["drm_device_file"] = s.decode() if s else None
            s = qdstr(devs[i], EGL_EXTENSIONS)
            d["extensions"] = s.decode() if s else None
        m["devices"].append(d)
    if n.value <= device:
        stop(f"device {device} not among {n.value} EGL devices (egl error {egl_err()})")

    stage("initialize")
    dpy = getpd(EGL_PLATFORM_DEVICE_EXT, devs[device], None)
    egl.eglInitialize.argtypes = [vp, C.POINTER(c_int), C.POINTER(c_int)]
    maj, mnr = c_int(), c_int()
    if not dpy or not egl.eglInitialize(dpy, C.byref(maj), C.byref(mnr)):
        stop(f"eglInitialize failed ({egl_err()})")
    m["egl_version"] = f"{maj.value}.{mnr.value}"
    for k, v in (("EGL_VENDOR", EGL_VENDOR), ("EGL_VERSION", EGL_VERSION), ("EGL_CLIENT_APIS", EGL_CLIENT_APIS)):
        m[k] = (egl.eglQueryString(dpy, v) or b"").decode()

    stage("context")
    cfg, ncfg = vp(), c_int()
    egl.eglChooseConfig.argtypes = [vp, C.POINTER(c_int), C.POINTER(vp), c_int, C.POINTER(c_int)]
    egl.eglChooseConfig(dpy, attribs(EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                     EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                                     EGL_DEPTH_SIZE, 24), C.byref(cfg), 1, C.byref(ncfg))
    m["num_configs"] = ncfg.value
    if ncfg.value < 1:
        stop(f"eglChooseConfig: no RGBA8 + depth-24 pbuffer config ({egl_err()})")
    egl.eglCreatePbufferSurface.restype = vp
    egl.eglCreatePbufferSurface.argtypes = [vp, vp, C.POINTER(c_int)]
    surf = egl.eglCreatePbufferSurface(dpy, cfg, attribs(EGL_WIDTH, size, EGL_HEIGHT, size))
    egl.eglBindAPI.argtypes = [c_uint]
    egl.eglBindAPI(EGL_OPENGL_API)
    egl.eglCreateContext.restype = vp
    egl.eglCreateContext.argtypes = [vp, vp, vp, C.POINTER(c_int)]
    ctx = egl.eglCreateContext(dpy, cfg, None, attribs())
    egl.eglMakeCurrent.argtypes = [vp, vp, vp, vp]
    if not (surf and ctx and egl.eglMakeCurrent(dpy, surf, surf, ctx)):
        stop(f"pbuffer / context / make-current failed ({egl_err()})")

    stage("load_libOpenGL")
    try:
        gl = cdll(gl_lib)
    except OSError as e:
        stop(f"cannot load {gl_lib}: {e!r}")
    fn = {}
    for name, (restype, argtypes) in GL_PROTOS.items():
        f = getattr(gl, name, None)
        if f is not None:
            f.restype, f.argtypes = restype, argtypes
        else:
            f = proc(name, restype, *argtypes)
        if f is None:
            stop(f"GL entry point {name} not found")
        fn[name] = f
    for k, v in (("GL_VENDOR", GL_VENDOR), ("GL_RENDERER", GL_RENDERER), ("GL_VERSION", GL_VERSION),
                 ("GL_SHADING_LANGUAGE_VERSION", GL_SHADING_LANGUAGE_VERSION)):
        m[k] = (fn["glGetString"](v) or b"").decode(errors="replace")
    bits = c_int(-1)
    fn["glGetIntegerv"](GL_DEPTH_BITS, C.byref(bits))
    m["depth_bits"] = bits.value
    m["gl_errors_after_context"] = gl_errors(fn)

    stage("compile")
    logs = []

    def shader(kind, src, label):
        sh = fn["glCreateShader"](kind)
        srcs = (cp * 1)(src)
        fn["glShaderSource"](sh, 1, srcs, None)
        fn["glCompileShader"](sh)
        ok = c_int(0)
        fn["glGetShaderiv"](sh, GL_COMPILE_STATUS, C.byref(ok))
        buf = C.create_string_buffer(8192)
        ln = c_int(0)
        fn["glGetShaderInfoLog"](sh, 8192, C.byref(ln), buf)
        logs.append(f"## {label} compile status {ok.value}\n{buf.value.decode(errors='replace')}")
        return sh, bool(ok.value)

    vs, vok = shader(GL_VERTEX_SHADER, VERT, "vertex")
    fs, fok = shader(GL_FRAGMENT_SHADER, FRAG, "fragment")
    m["compile_ok"] = vok and fok
    prog = fn["glCreateProgram"]()
    fn["glAttachShader"](prog, vs)
    fn["glAttachShader"](prog, fs)
    fn["glLinkProgram"](prog)
    lok = c_int(0)
    fn["glGetProgramiv"](prog, GL_LINK_STATUS, C.byref(lok))
    buf = C.create_string_buffer(8192)
    ln = c_int(0)
    fn["glGetProgramInfoLog"](prog, 8192, C.byref(ln), buf)
    logs.append(f"## program link status {lok.value}\n{buf.value.decode(errors='replace')}")
    m["link_ok"] = bool(lok.value)
    with open(os.path.join(out, "shader_logs.txt"), "w") as f:
        f.write("\n".join(logs) + "\n")
    m["gl_errors_after_link"] = gl_errors(fn)
    if not (m["compile_ok"] and m["link_ok"]):
        stop("GLSL 330 compile or link failed (shader_logs.txt)")

    stage("draw")
    vao = c_uint(0)
    fn["glGenVertexArrays"](1, C.byref(vao))
    fn["glBindVertexArray"](vao.value)
    fn["glUseProgram"](prog)
    fn["glViewport"](0, 0, size, size)
    fn["glEnable"](GL_DEPTH_TEST)
    fn["glDepthFunc"](GL_LESS)
    fn["glClearDepth"](1.0)
    fn["glClearColor"](*common.CLEAR)
    fn["glClear"](GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    t = time.time()
    fn["glDrawArrays"](GL_TRIANGLES, 0, 3)
    fn["glFinish"]()
    m["draw_finish_ms"] = round(1000 * (time.time() - t), 3)
    m["gl_errors_after_draw"] = gl_errors(fn)

    stage("readback")
    px = (C.c_ubyte * (size * size * 4))()
    fn["glReadPixels"](0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, px)
    m["gl_errors_after_readback"] = gl_errors(fn)
    data = bytes(px)
    pv = common.pixel_verdict(data, size, size)
    m["pixels"] = pv
    err = common.save_rgba_png(os.path.join(out, "tri.png"), data, size, size)
    if err:
        res["notes"].append(f"png: {err}")
    errs = m["gl_errors_after_draw"] + m["gl_errors_after_readback"]
    res["pass"] = bool(pv["ok"] and not errs)
    if errs:
        res["notes"].append(f"glGetError after draw / readback: {errs}")
    if not pv["ok"]:
        res["notes"].append(f"pixel rule failed: centre {pv['centre']}, corners {pv['corners']}"
                            f"{' (all zero)' if pv['all_zero'] else ''}")
    egl.eglTerminate.argtypes = [vp]
    stage("done")
    egl.eglTerminate(dpy)
