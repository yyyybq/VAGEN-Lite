"""EGL probe (eglinfo-like, ctypes only) for the TVR render smoke job (WP-TVR, plan section 3.7).

Ported from the WP-TVR design draft (scratchpad vpd_tvr/design/egl_probe.py).  JOB ONLY: never run it on the
login node.  glvnd libEGL.so.1 (OUT/tvr/nvgl_595.71.05/glvnd) dispatches to the NVIDIA vendor library named by
__EGL_VENDOR_LIBRARY_FILENAMES.

1. eglQueryDevicesEXT: number of devices; per device EGL_DRM_DEVICE_FILE_EXT and EGL_CUDA_DEVICE_NV.
2. Device 0: eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT), eglInitialize, vendor / version / client APIs.
3. 256x256 RGBA8 + depth-24 pbuffer, desktop OpenGL context, make current, GL_VENDOR / GL_RENDERER / GL_VERSION,
   glClearColor(0.2, 0.4, 0.6, 1) + glClear + glReadPixels -> expect about (51, 102, 153, 255); saves clear.png.
   (= E5.  A clear is a fill, not rasterisation: it shows neither shaders nor triangles.)
4. E6 (added 2026-11-09), same context: compile + link a GLSL 330 core vertex / fragment shader (info logs kept in
   result.json 'draw' and OUT/shader_info.log), depth test on (GL_LESS), clear colour + depth, one red triangle
   from a VBO / VAO, glFinish, glReadPixels -> centre pixel about (255, 0, 0) (+-3), the four corner pixels still the
   clear colour (51, 102, 153) (+-2), glGetError 0 after the draw and after the readback; saves draw.png.
Own pass rule ('pass', unchanged) = >=1 device, eglInitialize ok, EGL vendor NVIDIA, GL_RENDERER contains "H800",
readback within +-2; 'draw_pass' = E6.  The gates themselves are decided by verdict.py (EGL_GATE = E0-E5,
EGL_DRAW = E6).  Output OUT/result.json with the stage reached ('stage').
MuJoCo (MUJOCO_GL=egl), Habitat-sim and Open3D's headless renderer need this EGL path AND shader rasterisation,
so only E6 passing (not E5) supports reconsidering them.
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
# E6 (draw)
GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_LESS, GL_TRIANGLES, GL_FLOAT, GL_FALSE = 0x0100, 0x0B71, 0x0201, 0x0004, \
    0x1406, 0
GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER = 0x8892, 0x88E4, 0x8B31, 0x8B30
GL_COMPILE_STATUS, GL_LINK_STATUS, GL_INFO_LOG_LENGTH, GL_NO_ERROR = 0x8B81, 0x8B82, 0x8B84, 0
DRAW_RGB = (255, 0, 0)  # the triangle (fragment shader output)
DRAW_TOL = 3
CLEAR_RGB = EXPECT[:3]  # a corner pixel outside the triangle keeps the clear colour
CLEAR_TOL = 2
CENTRE = (SIZE // 2, SIZE // 2)  # (x, y) in GL window coordinates (origin bottom-left)
CORNERS = ((0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1))
# NDC (x, y, z): the centre (0, 0) is inside, the corners are outside; z = 0 -> depth 0.5 < the cleared 1.0
TRIANGLE = (-0.6, -0.6, 0.0, 0.6, -0.6, 0.0, 0.0, 0.6, 0.0)
VS_330 = b"""#version 330 core
layout(location = 0) in vec3 pos;
void main() { gl_Position = vec4(pos, 1.0); }
"""
FS_330 = b"""#version 330 core
out vec4 colour;
void main() { colour = vec4(1.0, 0.0, 0.0, 1.0); }
"""


class _Stop(Exception):
    """Ends the probe at a stage; the result so far is written."""


def attribs(*xs):
    return (C.c_int * (len(xs) + 1))(*xs, EGL_NONE)


def readback_ok(px, expect=EXPECT, tol=2):
    return isinstance(px, list) and len(px) == 4 and all(abs(x - y) <= tol for x, y in zip(px, expect))


def rgb_close(px, expect, tol):
    """px = [r, g, b, ...] within +-tol of expect (r, g, b)."""
    return (isinstance(px, list) and len(px) >= 3 and all(isinstance(x, int) for x in px[:3])
            and all(abs(x - y) <= tol for x, y in zip(px[:3], expect)))


def draw_ok(d):
    """The E6 rule on a result['draw'] dict (the same rule verdict.py applies): compiled + linked, no GL error,
    centre within +-DRAW_TOL of DRAW_RGB, all four corners within +-CLEAR_TOL of the clear colour."""
    corners = d.get("corners_px") if isinstance(d, dict) else None
    return (isinstance(d, dict) and d.get("vs_compiled") is True and d.get("fs_compiled") is True
            and d.get("linked") is True and d.get("gl_errors") == [] and rgb_close(d.get("centre_px"), DRAW_RGB, DRAW_TOL)
            and isinstance(corners, list) and len(corners) == 4
            and all(rgb_close(c, CLEAR_RGB, CLEAR_TOL) for c in corners))


def pixel(buf, x, y, size=SIZE):
    """RGBA of GL window pixel (x, y) in a glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE) buffer."""
    i = (y * size + x) * 4
    return list(buf[i:i + 4])


class _GL:
    """GL entry points: the glvnd libOpenGL.so.0 export if there is one, else eglGetProcAddress."""

    def __init__(self, gl, egl):
        self.gl, self.egl = gl, egl

    def fn(self, name, restype, *argtypes):
        try:
            f = getattr(self.gl, name)
        except AttributeError:
            p = self.egl.eglGetProcAddress(name.encode())
            if not p:
                raise RuntimeError(f"no GL entry point {name}") from None
            return C.CFUNCTYPE(restype, *argtypes)(p)
        try:
            f.restype, f.argtypes = restype, list(argtypes)
        except AttributeError:  # not a ctypes function (the tests' Python stand-in)
            pass
        return f


def draw_probe(gl, egl, out):
    """E6 in the current context; returns the 'draw' dict (never raises: an exception is recorded in 'error')."""
    d = {"vs_compiled": None, "fs_compiled": None, "linked": None, "gl_errors": None}
    try:
        _draw(_GL(gl, egl), out, d)
    except Exception as e:  # noqa: BLE001
        d["error"] = repr(e)
    d["pass"] = draw_ok(d)
    return d


def _draw(G, out, d):
    u, i, vp = C.c_uint, C.c_int, C.c_void_p
    get_error = G.fn("glGetError", u)
    errors = []

    def check(where):
        for _ in range(8):  # glGetError returns one flag per call
            e = get_error()
            if e == GL_NO_ERROR:
                return
            errors.append({"after": where, "error": hex(e)})

    check("context")  # flags left by E5, if any, are recorded too
    create_shader = G.fn("glCreateShader", u, u)
    shader_source = G.fn("glShaderSource", None, u, i, C.POINTER(C.c_char_p), C.POINTER(i))
    compile_shader = G.fn("glCompileShader", None, u)
    get_shader_iv = G.fn("glGetShaderiv", None, u, u, C.POINTER(i))
    get_shader_log = G.fn("glGetShaderInfoLog", None, u, i, C.POINTER(i), C.c_char_p)
    create_program = G.fn("glCreateProgram", u)
    attach = G.fn("glAttachShader", None, u, u)
    link = G.fn("glLinkProgram", None, u)
    get_program_iv = G.fn("glGetProgramiv", None, u, u, C.POINTER(i))
    get_program_log = G.fn("glGetProgramInfoLog", None, u, i, C.POINTER(i), C.c_char_p)

    def info_log(getter, logger, obj):
        n = i(0)
        getter(obj, GL_INFO_LOG_LENGTH, C.byref(n))
        if n.value <= 1:
            return ""
        buf = C.create_string_buffer(n.value)
        logger(obj, n.value, None, buf)
        return buf.value.decode(errors="replace")

    def shader(kind, src, key):
        s = create_shader(kind)
        arr = (C.c_char_p * 1)(src)
        shader_source(s, 1, arr, None)
        compile_shader(s)
        ok = i(0)
        get_shader_iv(s, GL_COMPILE_STATUS, C.byref(ok))
        d[f"{key}_compiled"] = bool(ok.value)
        d[f"{key}_log"] = info_log(get_shader_iv, get_shader_log, s)
        return s

    vs = shader(GL_VERTEX_SHADER, VS_330, "vs")
    fs = shader(GL_FRAGMENT_SHADER, FS_330, "fs")
    prog = create_program()
    attach(prog, vs)
    attach(prog, fs)
    link(prog)
    ok = i(0)
    get_program_iv(prog, GL_LINK_STATUS, C.byref(ok))
    d["linked"] = bool(ok.value)
    d["link_log"] = info_log(get_program_iv, get_program_log, prog)
    try:
        with open(os.path.join(out, "shader_info.log"), "w") as f:
            for k, status in (("vs_log", d["vs_compiled"]), ("fs_log", d["fs_compiled"]), ("link_log", d["linked"])):
                f.write(f"## {k} (ok={status})\n{d.get(k) or ''}\n")
    except OSError as e:
        d["log_file_error"] = repr(e)
    check("compile_link")
    if not (d["vs_compiled"] and d["fs_compiled"] and d["linked"]):
        d["gl_errors"] = errors
        return
    gen_vao = G.fn("glGenVertexArrays", None, i, C.POINTER(u))
    bind_vao = G.fn("glBindVertexArray", None, u)
    gen_buf = G.fn("glGenBuffers", None, i, C.POINTER(u))
    bind_buf = G.fn("glBindBuffer", None, u, u)
    buf_data = G.fn("glBufferData", None, u, C.c_ssize_t, vp, u)
    attrib_ptr = G.fn("glVertexAttribPointer", None, u, i, u, C.c_ubyte, i, vp)
    enable_attrib = G.fn("glEnableVertexAttribArray", None, u)
    use_program = G.fn("glUseProgram", None, u)
    viewport = G.fn("glViewport", None, i, i, i, i)
    enable = G.fn("glEnable", None, u)
    depth_func = G.fn("glDepthFunc", None, u)
    clear_color = G.fn("glClearColor", None, C.c_float, C.c_float, C.c_float, C.c_float)
    clear_depth = G.fn("glClearDepth", None, C.c_double)
    clear = G.fn("glClear", None, u)
    draw_arrays = G.fn("glDrawArrays", None, u, i, i)
    finish = G.fn("glFinish", None)
    read_pixels = G.fn("glReadPixels", None, i, i, i, i, u, u, vp)

    vao, vbo = u(0), u(0)
    gen_vao(1, C.byref(vao))
    bind_vao(vao)
    gen_buf(1, C.byref(vbo))
    bind_buf(GL_ARRAY_BUFFER, vbo)
    verts = (C.c_float * len(TRIANGLE))(*TRIANGLE)
    buf_data(GL_ARRAY_BUFFER, C.sizeof(verts), C.cast(verts, vp), GL_STATIC_DRAW)
    attrib_ptr(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    enable_attrib(0)
    viewport(0, 0, SIZE, SIZE)
    enable(GL_DEPTH_TEST)
    depth_func(GL_LESS)
    clear_color(0.2, 0.4, 0.6, 1.0)
    clear_depth(1.0)
    clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    use_program(prog)
    draw_arrays(GL_TRIANGLES, 0, 3)
    finish()
    check("draw")
    buf = (C.c_ubyte * (SIZE * SIZE * 4))()
    read_pixels(0, 0, SIZE, SIZE, GL_RGBA, GL_UNSIGNED_BYTE, C.cast(buf, vp))
    check("readback")
    d["centre_px"] = pixel(buf, *CENTRE)
    d["corners_px"] = [pixel(buf, x, y) for x, y in CORNERS]
    d["red_pixels"] = sum(1 for k in range(0, SIZE * SIZE * 4, 4)
                          if buf[k] >= 250 and buf[k + 1] <= 5 and buf[k + 2] <= 5)
    d["gl_errors"] = errors
    try:
        from PIL import Image
        Image.frombuffer("RGBA", (SIZE, SIZE), bytes(buf), "raw", "RGBA", 0, -1).save(os.path.join(out, "draw.png"))
    except Exception as e:  # noqa: BLE001
        d["png_error"] = repr(e)


def probe(out, cdll=None, draw=True, checkpoint=None):
    """Run the probe; returns the result dict (stage = last stage reached).  `cdll` replaces ctypes.CDLL;
    draw=False skips E6; `checkpoint(res)` is called after E5, before E6."""
    cdll = cdll or C.CDLL
    res = {"env": {k: os.environ.get(k) for k in ("__EGL_VENDOR_LIBRARY_FILENAMES", "LD_LIBRARY_PATH")},
           "pass": False, "draw_pass": False, "stage": "start"}
    try:
        _probe(out, cdll, res, draw, checkpoint)
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    return res


def _probe(out, cdll, res, draw=True, checkpoint=None):
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
    res["stage"] = "readback"
    if draw:
        if checkpoint is not None:  # E1-E5 are on disk before E6 touches the driver's shader compiler
            checkpoint(res)
        res["draw"] = draw_probe(gl, egl, out)  # E6; never changes the E1-E5 fields above
        res["draw_pass"] = res["draw"]["pass"]
    egl.eglTerminate.argtypes = [C.c_void_p]
    egl.eglTerminate(dpy)
    stop("draw" if draw else "readback")


def write_result(out, res):
    tmp = os.path.join(out, "result.json.part")
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, os.path.join(out, "result.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-draw", action="store_true", help="skip E6 (the GLSL triangle)")
    ap.add_argument("--require-draw", action="store_true", help="exit 0 only if E5 and E6 both pass")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = probe(a.out, draw=not a.no_draw, checkpoint=lambda r: write_result(a.out, r))
    write_result(a.out, res)
    print(json.dumps(res, indent=1))
    ok = res["pass"] and (res.get("draw_pass") is True or not a.require_draw)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
