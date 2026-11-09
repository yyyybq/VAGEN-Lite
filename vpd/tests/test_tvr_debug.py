"""Tests of vpd-tvr-debug: vpd/tvr/{debug_job.sh, debug_probe.py, debug/*} (EXPLORATORY TVR render debug job).

Login node, 1 core, no GPU use: no NVIDIA EGL / Vulkan call, no Controller, no real nvidia-smi in the shell runs
(a fake one is first on PATH).  The Vulkan triangle runs for real on lavapipe only (LP_NUM_THREADS=1, the lvp ICD
as the only driver, the NVIDIA implicit layers disabled; the test asserts no NVIDIA library was mapped), with both
the image loader 1.2.131 and the env loader 1.4.357.  gl_tri runs against fake EGL / GL libraries.  The job
script runs on a fake staged tree with a fake probe interpreter; its bookkeeping subcommands (facts, finalize,
post-thor, summarize, get, layers-json) run for real under this interpreter.  Scratch lives under
eval_external_assets/vpd/scratch/impl-tvrdebug/pytest_<rand>/ (env VPD_TVR_TEST_DIR overrides), removed after.
    cd $WT && PYTHONPATH=$WT OMP_NUM_THREADS=1 nice -n 19 $PY_EVAL -m pytest -q -p no:cacheprovider \
        --basetemp $VPD_OUT/scratch/impl-tvrdebug/pt vpd/tests/test_tvr_debug.py
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from vpd import paths
from vpd.tvr.debug import common, facts, gl_tri, post, summarize, thor_diag, vk_tri

TVR = paths.VPD_PKG / "tvr"
JOB = TVR / "debug_job.sh"
PROBE = TVR / "debug_probe.py"
DRV = "595.71.05"
BUILD_NAME = "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917"
TVR_PY = paths.TVR_ENV / "bin" / "python"
LVP_ICD = paths.TVR_ENV / "share" / "vulkan" / "icd.d" / "lvp_icd.x86_64.json"


@pytest.fixture
def base():
    root = Path(os.environ.get("VPD_TVR_TEST_DIR") or paths.scratch("impl-tvrdebug"))
    d = root / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def login_env(**extra):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", VPD_THREADS="1", PYTHONDONTWRITEBYTECODE="1", LP_NUM_THREADS="1")
    for k in ("RUN", "CUDA_VISIBLE_DEVICES", "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "LD_LIBRARY_PATH",
              "NVIDIA_DRIVER_CAPABILITIES", "VPD_JOB_DEADLINE_S"):
        env.pop(k, None)
    env.update(extra)
    return env


def wjson(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


# --- static ---------------------------------------------------------------------------------------------------

def test_bash_syntax():
    r = subprocess.run(["bash", "-n", str(JOB)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_script_variant_order_matches_summarize():
    sh = JOB.read_text()
    code = "\n".join(ln for ln in sh.splitlines() if not ln.lstrip().startswith("#"))
    names = re.findall(r"^(?:variant|maybe \"[^\n]*?\") (v\d\d_[a-z0-9_SR]+) \d+ ", code, re.M)
    assert names == summarize.VARIANTS


def test_time_budget_fits_the_soft_deadline():
    sh = JOB.read_text()
    code = "\n".join(ln for ln in sh.splitlines() if not ln.lstrip().startswith("#"))
    caps = [int(x) for x in re.findall(r"^(?:variant|maybe \"[^\n]*?\") v\d\d_\w+ (\d+) ", code, re.M)]
    assert caps == [30, 60, 60, 60, 60, 60, 60, 240, 240, 240, 240, 240, 240, 120, 900]
    assert sum(caps) == 2850
    assert "DEADLINE_S=${VPD_JOB_DEADLINE_S:-3120}" in sh and "RESERVE_S=${VPD_TVR_RESERVE_S:-60}" in sh
    assert sum(caps) + 15 * 6 <= 3120 - 60  # every cap hit + ~6 s of reaping / bookkeeping each
    assert "VPD_JOB_DEADLINE_S=3120 timeout -k 60 3300" in sh  # the outer cap of the submit line
    assert 'UNITY_PAT="/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME"' in sh
    assert f"DRV={DRV}" in sh and f"BUILD_NAME={BUILD_NAME}" in sh


def test_struct_sizes_match_the_c_layout():
    for name, size in vk_tri.EXPECTED_SIZES.items():
        assert C.sizeof(getattr(vk_tri, name)) == size, name


def test_spirv_sources_match_the_binaries(base):
    spv_as, spv_val = paths.TVR_ENV / "bin" / "spirv-as", paths.TVR_ENV / "bin" / "spirv-val"
    if not (spv_as.exists() and spv_val.exists()):
        pytest.skip("envs/tvr spirv-tools missing")
    for s in ("vert", "frag"):
        src = Path(vk_tri.SHADERS) / f"tri.{s}.spvasm"
        out = base / f"tri.{s}.spv"
        r = subprocess.run(["nice", "-n", "19", str(spv_as), "--target-env", "vulkan1.0", str(src), "-o", str(out)],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert out.read_bytes() == (Path(vk_tri.SHADERS) / f"tri.{s}.spv").read_bytes()
        for env in ("vulkan1.0", "vulkan1.1"):
            r = subprocess.run([str(spv_val), "--target-env", env, str(out)], capture_output=True, text=True,
                               timeout=60)
            assert r.returncode == 0, (s, env, r.stderr)


def test_load_spv(base):
    arr, n, sha = vk_tri.load_spv(os.path.join(vk_tri.SHADERS, "tri.vert.spv"))
    assert n % 4 == 0 and arr[0] == vk_tri.SPV_MAGIC and len(sha) == 64
    bad = base / "bad.spv"
    bad.write_bytes(b"\0" * 24)
    with pytest.raises(ValueError):
        vk_tri.load_spv(str(bad))
    bad.write_bytes(b"\x03\x02\x23\x07" * 5 + b"x")
    with pytest.raises(ValueError):
        vk_tri.load_spv(str(bad))


def image(size=common.SIZE, clear=common.CLEAR_RGB, centre=common.RED_RGB, box=20):
    buf = bytearray(bytes([*clear, 255]) * (size * size))
    c = size // 2
    for y in range(c - box, c + box):
        for x in range(c - box, c + box):
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes([*centre, 255])
    return bytes(buf)


def test_pixel_verdict():
    v = common.pixel_verdict(image())
    assert v["ok"] and v["centre_ok"] and v["corners_ok"] and not v["all_zero"]
    assert v["centre"] == [255, 0, 0, 255] and v["red_fraction"] == round(1600 / 65536, 4)
    z = common.pixel_verdict(bytes(common.SIZE * common.SIZE * 4))
    assert not z["ok"] and z["all_zero"]
    off = common.pixel_verdict(image(centre=(251, 0, 0)))
    assert not off["centre_ok"] and off["corners_ok"]   # 4 off > tol 3
    near = common.pixel_verdict(image(centre=(252, 3, 0), clear=(53, 100, 153)))
    assert near["ok"]                                    # within 3 / 2
    far = common.pixel_verdict(image(clear=(54, 102, 153)))
    assert not far["corners_ok"]


# --- vk_tri on lavapipe (real Vulkan, CPU only) ---------------------------------------------------------------

@pytest.mark.parametrize("loader", ["env", "image"])
def test_vk_tri_on_lavapipe(base, loader):
    if not (TVR_PY.exists() and LVP_ICD.exists()):
        pytest.skip("envs/tvr python or the lavapipe ICD missing")
    ldr = paths.TVR_ENV / "lib" / "libvulkan.so.1" if loader == "env" else Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1")
    if not ldr.exists():
        pytest.skip(f"{ldr} missing")
    out = base / f"vk_{loader}"
    env = login_env(VK_ICD_FILENAMES=str(LVP_ICD), VK_DRIVER_FILES=str(LVP_ICD), DISABLE_LAYER_NV_OPTIMUS_1="1",
                    DISABLE_LAYER_NV_PRESENT_1="1", HOME=str(base / "home"))
    if loader == "env":
        env["LD_LIBRARY_PATH"] = str(paths.TVR_ENV / "lib")
    r = subprocess.run(["nice", "-n", "19", "timeout", "120", str(TVR_PY), "-I", "-u", str(PROBE), "vk-tri",
                        "--out", str(out), "--loader", str(ldr), "--want", "cpu"],
                       env=env, capture_output=True, text=True, timeout=150)
    res = json.loads((out / "result.json").read_text())
    assert r.returncode == 0 and res["pass"] is True, (r.stdout[-3000:], r.stderr[-2000:])
    m = res["metrics"]
    assert res["stage"] == "done" and m["device"]["type"] == "CPU"
    assert m["loader_instance_version"].startswith("1.4." if loader == "env" else "1.2.131")
    for k in ("vkCreateGraphicsPipelines", "vkQueueSubmit", "vkWaitForFences"):
        assert m["vk"][k] == "VK_SUCCESS(0)"
    assert m["pixels"]["centre"] == [255, 0, 0, 255]
    assert all(p == [51, 102, 153, 255] for p in m["pixels"]["corners"].values())
    assert 0.17 <= m["pixels"]["red_fraction"] <= 0.19  # the triangle covers 18 % of the target
    assert not [p for p in m["mapped_gpu_libs"] if "nvidia" in p or "GLX" in p]   # never touched the GPU
    assert (out / "tri.png").exists()


def test_vk_tri_missing_loader(base):
    res = vk_tri.run(str(base), str(base / "no_libvulkan.so.1"))
    assert res["pass"] is False and res["stage"] == "load_loader"
    assert any("cannot load the Vulkan loader" in n for n in res["notes"])


# --- gl_tri against fake EGL / GL -------------------------------------------------------------------------------

class FakeFn:
    def __init__(self, impl):
        self.impl, self.restype, self.argtypes = impl, None, None

    def __call__(self, *a):
        return self.impl(*a)


class FakeLib:
    def __init__(self, impls):
        self._impls = impls

    def __getattr__(self, name):
        if name.startswith("_") or name not in self._impls:
            raise AttributeError(name)
        f = FakeFn(self._impls[name])
        setattr(self, name, f)
        return f


def _set(ref, value):
    ref._obj.value = value  # a ctypes byref() argument


def fake_gl_libs(compile_ok=True, pixels=None, gl_error=0):
    keep = []

    def cb(proto, fn):
        f = proto(fn)
        keep.append(f)
        return C.cast(f, C.c_void_p).value

    def qdev(maxn, devs, n):
        devs[0] = 0x1234
        n[0] = 1
        return 1

    procs = {b"eglQueryDevicesEXT": cb(C.CFUNCTYPE(C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int)),
                                       qdev),
             b"eglGetPlatformDisplayEXT": cb(C.CFUNCTYPE(C.c_void_p, C.c_int, C.c_void_p, C.POINTER(C.c_int)),
                                             lambda plat, dev, att: 0x5678)}

    def choose(dpy, att, cfg, n, ncfg):
        _set(cfg, 0x99)
        _set(ncfg, 1)
        return 1

    calls, rec = [], {}
    err = {"pending": 0}

    def call(name, ret=None):
        def f(*a):
            calls.append(name)
            return ret
        return f

    def get_error():
        e, err["pending"] = err["pending"], 0
        return e

    def shader_source(sh, n, srcs, lens):
        rec.setdefault("sources", []).append(srcs[0])

    def get_iv(value):
        def f(obj, pname, p):
            p[0] = value
        return f

    def info_log(obj, n, ln, addr):
        text = b"fake info log"
        C.memmove(addr, text + b"\0", len(text) + 1)
        ln[0] = len(text)

    def gen_vao(n, p):
        p[0] = 7

    def clear_color(r, g, b, a):
        rec["clear_color"] = (r, g, b, a)

    def clear_depth(d):
        rec["clear_depth"] = d

    def draw(mode, first, count):
        calls.append("glDrawArrays")
        rec["draw"] = (mode, first, count)
        err["pending"] = gl_error  # the draw raises the GL error, if any

    def read(x, y, w, h, fmt, typ, addr):
        rec["read"] = (x, y, w, h, fmt, typ)
        C.memmove(addr, pixels, len(pixels))

    def integerv(pname, p):
        p[0] = 24

    # every GL entry point except glGetString is a real C callback with gl_tri's own prototype, reached through
    # eglGetProcAddress, so every argument goes through ctypes' conversion exactly as against libOpenGL
    impls = {"glGetError": get_error, "glGetIntegerv": integerv, "glViewport": call("glViewport"),
             "glCreateShader": lambda k: 1 if k == gl_tri.GL_VERTEX_SHADER else 2, "glShaderSource": shader_source,
             "glCompileShader": call("compile"), "glGetShaderiv": get_iv(int(compile_ok)),
             "glGetShaderInfoLog": info_log, "glCreateProgram": lambda: 3, "glAttachShader": call("attach"),
             "glLinkProgram": call("link"), "glGetProgramiv": get_iv(1), "glGetProgramInfoLog": info_log,
             "glUseProgram": call("use"), "glGenVertexArrays": gen_vao, "glBindVertexArray": call("vao"),
             "glEnable": call("glEnable"), "glDepthFunc": call("depthfunc"), "glClearDepth": clear_depth,
             "glClearColor": clear_color, "glClear": call("clear"), "glDrawArrays": draw,
             "glFinish": call("finish"), "glReadPixels": read}
    assert set(impls) | {"glGetString"} == set(gl_tri.GL_PROTOS)
    for name, f in impls.items():
        restype, argtypes = gl_tri.GL_PROTOS[name]
        procs[name.encode()] = cb(C.CFUNCTYPE(restype, *argtypes), f)
    egl = FakeLib({"eglGetProcAddress": lambda name: procs.get(name), "eglQueryString": lambda d, k: b"NVIDIA",
                   "eglGetError": lambda: 0x3000, "eglInitialize": lambda d, a, b: (_set(a, 1), _set(b, 5), 1)[-1],
                   "eglChooseConfig": choose, "eglCreatePbufferSurface": lambda *a: 0x77,
                   "eglBindAPI": lambda a: 1, "eglCreateContext": lambda *a: 0x88, "eglMakeCurrent": lambda *a: 1,
                   "eglTerminate": lambda d: 1})
    gl = FakeLib({"glGetString": lambda v: b"FAKE H800"})
    libs = {"egl": egl, "gl": gl}
    return (lambda path: libs["egl" if "EGL" in path else "gl"]), calls, keep, rec


def test_gl_tri_pass_with_fakes(base):
    cdll, calls, _keep, rec = fake_gl_libs(pixels=image())
    stages = []
    res = gl_tri.run(str(base), "/x/libEGL.so.1", "/x/libOpenGL.so.0", cdll=cdll,
                     on_stage=lambda r: stages.append(r["stage"]))
    assert res["pass"] is True, res["notes"]
    m = res["metrics"]
    assert m["compile_ok"] and m["link_ok"] and m["num_devices"] == 1 and m["depth_bits"] == 24
    assert m["EGL_VENDOR"] == "NVIDIA" and m["GL_RENDERER"] == "FAKE H800" and m["egl_version"] == "1.5"
    assert stages == ["load_libEGL", "query_devices", "initialize", "context", "load_libOpenGL", "compile",
                      "draw", "readback", "done"]
    assert calls.index("glEnable") < calls.index("clear") < calls.index("glDrawArrays") < calls.index("finish")
    assert rec["sources"] == [gl_tri.VERT, gl_tri.FRAG] and gl_tri.VERT.startswith(b"#version 330")
    assert rec["draw"] == (gl_tri.GL_TRIANGLES, 0, 3) and rec["clear_depth"] == 1.0
    assert rec["read"] == (0, 0, 256, 256, gl_tri.GL_RGBA, gl_tri.GL_UNSIGNED_BYTE)
    assert [round(x, 4) for x in rec["clear_color"]] == [0.2, 0.4, 0.6, 1.0]
    assert "fake info log" in (base / "shader_logs.txt").read_text() and (base / "tri.png").exists()


def test_gl_tri_compile_failure_and_gl_error(base):
    cdll, calls, _keep, _rec = fake_gl_libs(compile_ok=False, pixels=image())
    res = gl_tri.run(str(base), "/x/libEGL.so.1", "/x/libOpenGL.so.0", cdll=cdll)
    assert res["pass"] is False and res["stage"] == "compile" and res["metrics"]["compile_ok"] is False
    assert "glDrawArrays" not in calls
    # a clean compile with a black readback and one GL error from the draw: FAIL, both recorded
    cdll, calls, _keep, _rec = fake_gl_libs(pixels=bytes(common.SIZE * common.SIZE * 4), gl_error=0x0502)
    res = gl_tri.run(str(base), "/x/libEGL.so.1", "/x/libOpenGL.so.0", cdll=cdll)
    assert res["pass"] is False and res["stage"] == "done" and res["metrics"]["compile_ok"] is True
    assert res["metrics"]["pixels"]["all_zero"]
    assert res["metrics"]["gl_errors_after_draw"] == ["GL_INVALID_OPERATION"]
    assert any("glGetError" in n for n in res["notes"]) and any("all zero" in n for n in res["notes"])
    # good pixels but a GL error: still FAIL
    cdll, calls, _keep, _rec = fake_gl_libs(pixels=image(), gl_error=0x0505)
    res = gl_tri.run(str(base), "/x/libEGL.so.1", "/x/libOpenGL.so.0", cdll=cdll)
    assert res["pass"] is False and res["metrics"]["pixels"]["ok"] is True


def test_gl_tri_missing_library(base):
    res = gl_tri.run(str(base), str(base / "nope" / "libEGL.so.1"), "x")
    assert res["pass"] is False and res["stage"] == "load_libEGL"


# --- thor_diag with a fake Controller ---------------------------------------------------------------------------

class FakeProc:
    def __init__(self, rc=None):
        self.rc, self.pid = rc, 999999

    def poll(self):
        return self.rc

    @property
    def returncode(self):
        return self.rc


class FakeEvent:
    def __init__(self, frame, depth=None, seg=None, action="X"):
        self.frame, self.depth_frame, self.instance_segmentation_frame = frame, depth, seg
        self.metadata = {"lastAction": action, "lastActionSuccess": True}


def make_controller(kind="good", fail_start=False, fail_at=None):
    rng = np.random.default_rng(0)

    class Ctl:
        seen = {}

        def __init__(self, **kw):
            Ctl.seen["kw"] = kw
            self.server = type("Srv", (), {})()
            cmd = self.unity_command(kw["width"], kw["height"], False)
            env = {}
            self._start_unity_thread(env, kw["width"], kw["height"], {}, None)
            Ctl.seen.update(cmd=cmd, env=env)
            if fail_start:
                raise RuntimeError("fake start-up failure")
            self.unity_pid = None
            self.n = 0
            self.last_event = self.event("init")

        def unity_command(self, width, height, headless):
            return ["/fake/thor", "-screen-width", str(width)]

        def _start_unity_thread(self, env, width, height, params, image_name):
            self.server.unity_proc = FakeProc()

        def event(self, action):
            h, w = Ctl.seen["kw"]["height"], Ctl.seen["kw"]["width"]
            if kind == "good":
                f = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            else:
                f = np.zeros((h, w, 3), dtype=np.uint8)
            d = rng.random((h, w)).astype(np.float32) if Ctl.seen["kw"].get("renderDepthImage") else None
            s = rng.integers(0, 4, (h, w, 3), dtype=np.uint8) if Ctl.seen["kw"].get("renderInstanceSegmentation") \
                else None
            return FakeEvent(f, d, s, action)

        def step(self, action=None, **kw):
            self.n += 1
            if fail_at is not None and self.n > fail_at:
                raise TimeoutError("fake step timeout")
            self.last_event = self.event(action)
            return self.last_event

        def stop(self):
            Ctl.seen["stopped"] = True

    return Ctl


def thor_args(out, *extra):
    p = argparse.ArgumentParser()
    thor_diag.add_arguments(p)
    return p.parse_args(["--out", str(out), "--width", "32", "--height", "24", "--steps", "12", "--no-smi",
                         "--no-fatal-watch", *extra])


def test_thor_diag_good_frames(base, monkeypatch):
    monkeypatch.setenv("HOME", str(base / "home"))
    ctl = make_controller("good")
    orig = (ctl.unity_command, ctl._start_unity_thread)
    a = thor_args(base / "v", "--gpu-device", "0", "--depth", "--seg", "--aa", "none", "--quality", "Very Low",
                  "--unity-arg=-force-gfx-direct", "--child-env", "LD_DEBUG=libs,files",
                  "--child-env", "VK_LOADER_DEBUG=all")
    res = thor_diag.run(a, controller_cls=ctl, platform="CR", np=np)
    assert res["pass"] is True and res["stage"] == "done"
    fr = res["metrics"]["frames"]
    assert fr["n_step_frames"] == 12 and fr["frac_rgb_std_ge2"] == 1.0
    assert fr["frac_depth_varies"] == 1.0 and fr["frac_seg_varies"] == 1.0
    kw = ctl.seen["kw"]
    assert kw["gpu_device"] == 0 and kw["quality"] == "Very Low" and "antiAliasing" not in kw
    assert kw["renderDepthImage"] and kw["renderInstanceSegmentation"] and kw["platform"] == "CR"
    assert ctl.seen["cmd"][-1] == "-force-gfx-direct"
    assert ctl.seen["env"] == {"LD_DEBUG": "libs,files", "VK_LOADER_DEBUG": "all"}   # Unity only
    assert "LD_DEBUG" not in os.environ
    assert (ctl.unity_command, ctl._start_unity_thread) == orig   # patches undone
    assert ctl.seen["stopped"]
    lines = (base / "v" / "frames.jsonl").read_text().splitlines()
    assert len(lines) == 13 and json.loads(lines[0])["i"] == -1
    for f in ("rgb.png", "depth.png", "seg.png", "result.json"):
        assert (base / "v" / f).exists(), f
    assert res["metrics"]["ms_per_step"] is not None and res["metrics"]["steps_per_s"] > 0


def test_thor_diag_black_rgb_with_depth_and_force_index(base, monkeypatch):
    monkeypatch.setenv("HOME", str(base / "home"))
    ctl = make_controller("black")
    a = thor_args(base / "v", "--force-device-index", "0", "--depth", "--seg")
    res = thor_diag.run(a, controller_cls=ctl, platform="CR", np=np)
    fr = res["metrics"]["frames"]
    assert res["pass"] is False and fr["rgb_all_zero_frames"] == 12 and fr["frac_rgb_std_ge2"] == 0.0
    assert fr["frac_depth_varies"] == 1.0 and fr["frac_seg_varies"] == 1.0
    assert ctl.seen["cmd"][-2:] == ["-force-device-index", "0"] and "gpu_device" not in ctl.seen["kw"]
    assert ctl.seen["kw"]["antiAliasing"] == 4 and ctl.seen["kw"]["quality"] == "Ultra"


def test_thor_diag_start_failure_and_step_failure(base, monkeypatch):
    monkeypatch.setenv("HOME", str(base / "home"))
    res = thor_diag.run(thor_args(base / "a"), controller_cls=make_controller(fail_start=True), platform="CR", np=np)
    assert res["pass"] is False and "fake start-up failure" in res["error"]
    assert res["metrics"]["frames"]["n_step_frames"] == 0 and res["stage"] == "done"
    res = thor_diag.run(thor_args(base / "b"), controller_cls=make_controller(fail_at=5), platform="CR", np=np)
    assert res["metrics"]["frames"]["n_step_frames"] == 5 and res["pass"] is True   # 5 good frames of 12
    assert any("stepping failed after 5 steps" in n for n in res["notes"])
    assert any("only 5 of 12 steps" in n for n in res["notes"])


def test_thor_diag_deadline_caps_startup_timeout(base, monkeypatch):
    monkeypatch.setenv("HOME", str(base / "home"))
    ctl = make_controller("good")
    thor_diag.run(thor_args(base / "v", "--deadline-s", "100", "--startup-timeout", "500"), controller_cls=ctl,
                  platform="CR", np=np)
    assert ctl.seen["kw"]["server_start_timeout"] == 85.0


def test_watchdog_fatal_signal_and_early_exit(base):
    home = base / "home"
    log = home / thor_diag.PLAYER_LOG
    log.parent.mkdir(parents=True)
    log.write_text("Initialize engine\n")
    saved, exits = [], []
    res = common.new_result("x")
    state = {"phase": "startup", "t_run": 0}
    wd = thor_diag.Watchdog(str(home), state, res, lambda: saved.append(dict(res)), str(base), exit_fn=exits.append)
    assert wd.check() is None
    with open(log, "a") as f:
        f.write("Caught fatal signal - signo:11 code:1\n")
    reason = wd.check()
    assert reason and "Caught fatal signal" in reason
    wd.fire(reason)
    assert exits == [3] and saved and res["fatal"] == reason and res["stage"] == "fatal"
    assert (base / "Player.log").exists()
    # Unity gone during start-up
    wd2 = thor_diag.Watchdog(str(home), {"phase": "startup", "controller": type(
        "C", (), {"server": type("S", (), {"unity_proc": FakeProc(rc=1)})()})()}, common.new_result("y"),
        lambda: None, str(base), exit_fn=exits.append)
    wd2.offset = log.stat().st_size
    assert "exited with rc=1" in wd2.check()


def test_layers_json(base):
    p = base / "xdg" / "vulkan" / "implicit_layer.d" / "nvidia_layers.json"
    doc = thor_diag.write_layers_json(str(p), "/abs/libGLX_nvidia.so.0")
    got = json.loads(p.read_text())
    assert got == doc and len(got["layers"]) == 1
    ly = got["layers"][0]
    assert ly["name"] == "VK_LAYER_NV_optimus" and ly["library_path"] == "/abs/libGLX_nvidia.so.0"
    assert ly["enable_environment"] == {"__NV_PRIME_RENDER_OFFLOAD": "1"}
    with pytest.raises(ValueError):
        thor_diag.write_layers_json(str(p), "libGLX_nvidia.so.0")


def test_parse_child_env():
    assert thor_diag.parse_child_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(ValueError):
        thor_diag.parse_child_env(["nope"])


# --- post parsers -----------------------------------------------------------------------------------------------

LDDEBUG = """\
     4711:	find library=libnvidia-glvkspirv.so.595.71.05 [0]; searching
     4711:	 search path=/nvgl/lib:/nvgl/glvnd		(LD_LIBRARY_PATH)
     4711:	  trying file=/nvgl/lib/libnvidia-glvkspirv.so.595.71.05
     4711:
     4711:	file=libnvidia-glvkspirv.so.595.71.05 [0];  dynamically loaded by /nvgl/lib/libGLX_nvidia.so.0 [0]
     4711:	file=libnvidia-glvkspirv.so.595.71.05 [0];  generating link map
     4711:	calling init: /nvgl/lib/libnvidia-glvkspirv.so.595.71.05
     4711:	file=libnvidia-present.so.595.71.05 [0];  needed by /x/liby.so [0]
     4711:	find library=libnvidia-present.so.595.71.05 [0]; searching
     4711:	 search path=/nvgl/lib		(LD_LIBRARY_PATH)
     4711:	  trying file=/nvgl/lib/libnvidia-present.so.595.71.05
     4711:	find library=libfoo.so.1 [0]; searching
     4711:	calling init: /lib/x86_64-linux-gnu/libc.so.6
"""


def test_post_parsers(base):
    (base / "lddebug.4711").write_text(LDDEBUG)
    ld = post.parse_lddebug([str(base / "lddebug.4711")])
    assert ld["never_loaded"] == ["libfoo.so.1", "libnvidia-present.so.595.71.05"]
    assert ld["never_loaded_gpu"] == ["libnvidia-present.so.595.71.05"]
    assert ld["never_loaded_gpu_requested_by"]["libnvidia-present.so.595.71.05"].startswith("needed by /x/liby.so")
    assert ld["initialised_gpu_libs"] == ["/nvgl/lib/libnvidia-glvkspirv.so.595.71.05"]
    (base / "gpu.csv").write_text("2026/11/09 10:00:00.000, 0, 345, 600\n2026/11/09 10:00:00.500, 37, 1980, 2100\n"
                                  "2026/11/09 10:00:01.000, [N/A], 1980, 2100\n")
    g = post.parse_gpu_csv(str(base / "gpu.csv"))
    assert g == {"samples": 2, "util_max": 37.0, "util_mean": 18.5, "util_nonzero_samples": 1,
                 "clock_gr_max_mhz": 1980.0, "clock_gr_min_mhz": 345.0, "mem_first_mib": 600.0,
                 "mem_max_mib": 2100.0, "mem_rise_mib": 1500.0}
    assert post.parse_gpu_csv(str(base / "none.csv"))["samples"] == 0
    (base / "nvcache" / "GLCache" / "x").mkdir(parents=True)
    (base / "nvcache" / "GLCache" / "x" / "a.bin").write_bytes(b"12345")
    assert post.dir_stats(str(base / "nvcache")) == {"exists": True, "files": 1, "bytes": 5}
    (base / "unity.log").write_text(
        "INFO:             Found ICD manifest file /nvgl/vulkan/nvidia_icd.json, version \"1.0.1\"\n"
        "DEBUG:            Searching for ICD drivers named /nvgl/lib/libGLX_nvidia.so.0\n"
        "[Vulkan Loader] LAYER:      Insert instance layer \"VK_LAYER_NV_optimus\" (/nvgl/lib/libGLX_nvidia.so.0)\n"
        "ERROR: [Loader Message] Code 0 : loader_get_json: Invalid JSON file /etc/vulkan/implicit_layer.d/x.json\n"
        "unrelated line\n")
    lo = post.parse_loader_log(str(base / "unity.log"), str(base / "excerpt.txt"))
    assert lo["manifests"][0] == "/nvgl/vulkan/nvidia_icd.json"
    assert "/nvgl/lib/libGLX_nvidia.so.0" in lo["driver_libs"] and lo["layers"] == ["VK_LAYER_NV_optimus"]
    assert len(lo["errors"]) == 1 and lo["loader_lines"] == 4
    assert "unrelated" not in (base / "excerpt.txt").read_text()
    (base / "nvidia_smi_mid.txt").write_text("| Processes: |\n|  0  N/A N/A  123  G  ...thor  512MiB |\n")
    assert post.parse_smi_processes(str(base / "nvidia_smi_mid.txt")) == ["0  N/A N/A  123  G  ...thor  512MiB"]
    res = post.merge_into_result(str(base))   # no result.json: a stub carries the parse
    assert res["pass"] is False and res["metrics"]["post"]["gpu"]["samples"] == 2
    assert json.loads((base / "result.json").read_text())["metrics"]["post"]["nvcache"]["files"] == 1


# --- facts ------------------------------------------------------------------------------------------------------

SMI_Q = """
==============NVSMI LOG==============

Timestamp                                 : Mon Nov  9 17:09:52 2026
Driver Version                            : 595.71.05
CUDA Version                              : 13.2

Attached GPUs                             : 1
GPU 00000000:18:00.0
    Product Name                          : NVIDIA H800
    Product Architecture                  : Hopper
    Display Mode                          : Disabled
    Display Active                        : Disabled
    Persistence Mode                      : Enabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    GPU Virtualization Mode
        Virtualization Mode               : None
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    Compute Mode                          : Default
"""


def fake_system(root: Path, honoured: bool):
    sl = root / "syslib"
    sl.mkdir(parents=True)
    vk = root / "etc_vulkan"
    (vk / "icd.d").mkdir(parents=True)
    (vk / "implicit_layer.d").mkdir(parents=True)
    if honoured:
        (sl / f"libGLX_nvidia.so.{DRV}").write_bytes(b"x" * 100)
        (sl / f"libnvidia-glvkspirv.so.{DRV}").write_bytes(b"x" * 10)
        wjson(vk / "icd.d" / "nvidia_icd.json", {"file_format_version": "1.0.1",
                                                 "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.4"}})
    else:
        (sl / "libGLX_nvidia.so.550.90.07").write_bytes(b"")
        (sl / "libnvidia-glvkspirv.so.550.90.07").write_bytes(b"")
        (vk / "icd.d" / "nvidia_icd.json").write_text("")
        (vk / "implicit_layer.d" / "nvidia_layers.json").write_text("")
    (sl / "libGLX_nvidia.so.0").symlink_to(f"libGLX_nvidia.so.{DRV if honoured else '550.90.07'}")
    (sl / "libvulkan.so.1").write_bytes(b"x")
    (root / "egl_vendor").mkdir()
    dev = root / "dev"
    dev.mkdir()
    (dev / "nvidia-modeset").write_text("")
    (dev / "nvidia4").write_text("")
    drvf = root / "driver_version.txt"
    drvf.write_text(f"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  {DRV}  Release Build\n")
    return sl, vk, dev, drvf


@pytest.mark.parametrize("honoured", [False, True])
def test_facts(base, honoured):
    sl, vk, dev, drvf = fake_system(base, honoured)
    out = base / "v00"
    out.mkdir()
    (out / "nvidia_smi_q.txt").write_text(SMI_Q)
    (out / "ldconfig_nvidia.txt").write_text("\tlibGLX_nvidia.so.0 (libc6,x86-64) => /usr/lib/libGLX_nvidia.so.0\n")
    environ = {"NVIDIA_VISIBLE_DEVICES": "GPU-1", **({"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"}
                                                      if honoured else {})}
    res = facts.collect(str(out), sys_lib=str(sl), vk_dirs=[str(vk)], egl_vendor_dirs=[str(base / "egl_vendor")],
                        dev_dir=str(dev), driver=DRV, driver_file=str(drvf), environ=environ)
    m = res["metrics"]
    assert res["pass"] is True and m["caps_honored"] is honoured and m["graphics_requested"] is honoured
    assert m["caps_detail"] == {"libGLX_nvidia_ok": honoured, "nvidia_icd_json_ok": honoured}
    assert m["driver_match"] is True and m["dev"]["dri"] is None and m["dev"]["nvidia_modeset"] is True
    assert m["smi_q"]["Product Name"] == "NVIDIA H800" and m["smi_q"]["MIG Mode/Current"] == "Disabled"
    assert m["smi_q"]["GPU Operation Mode/Current"] == "N/A" and m["smi_q"]["Compute Mode"] == "Default"
    assert m["smi_q"]["GPU Virtualization Mode/Virtualization Mode"] == "None"
    assert m["ldconfig_has_libGLX_nvidia"] is True
    glx = m["system_libs"]["libGLX_nvidia.so.0"]
    assert glx["realpath"].endswith(DRV if honoured else "550.90.07")


def test_parse_smi_q_does_not_mix_sections():
    q = facts.parse_smi_q(SMI_Q)
    assert q["Driver Version"] == DRV and "Current" not in q and q["MIG Mode/Pending"] == "Disabled"


# --- finalize / summarize ---------------------------------------------------------------------------------------

def test_finalize(base):
    d = base / "v03_thor_diag_S"
    r = summarize.finalize(str(d), "v03_thor_diag_S", rc=124, secs=240, cap=240)
    assert r["pass"] is False and r["timed_out"] and any("no result.json" in n for n in r["notes"])
    wjson(d / "result.json", {"pass": True, "stage": "done", "metrics": {}, "notes": []})
    r = summarize.finalize(str(d), "v03_thor_diag_S", rc=0, secs=50, cap=240)
    assert r["pass"] is True and r["timed_out"] is False and r["secs"] == 50 and r["cap_s"] == 240
    r = summarize.finalize(str(base / "v04_thor_R"), "v04_thor_R", skipped="caps_not_honored")
    assert r["pass"] is None and r["skipped"] == "caps_not_honored" and summarize.state(r) is None


def results(**states):
    """Synthetic results: name -> True / False / None (skipped), plus metric overrides via name__metrics."""
    out = {}
    for n in summarize.VARIANTS:
        s = states.get(n, None)
        if s is None:
            out[n] = {"pass": None, "skipped": "x"}
        else:
            out[n] = {"pass": s, "metrics": {}}
    for k, v in states.items():
        if k.endswith("__metrics"):
            out[k[:-9]].setdefault("metrics", {}).update(v)
    return out


BLACK_V3 = {"frames": {"n_step_frames": 40, "frac_rgb_std_ge2": 0.0, "frac_depth_varies": 1.0,
                       "frac_seg_varies": 1.0}}


def rows(r):
    return [x["row"] for x in summarize.decide(r)["matched"]]


def test_decision_rows():
    # (1) + (4): GL and Vulkan fine, V3 black with depth / seg varying, V5 is the fix; (5) lavapipe works
    r = results(v00_facts=True, v01_gl_tri_S=True, v02_vk_nvS_img=True, v02_vk_nvS_env=True, v02_vk_lvp=True,
                v03_thor_diag_S=False, v03_thor_diag_S__metrics=BLACK_V3, v05_thor_lowq=True, v06_thor_newloader=False,
                v07_thor_layers=False, v08_thor_direct=False, v10_thor_lvp_forced=True,
                v10_thor_lvp_forced__metrics={"startup_s": 300.0, "ms_per_step": 120.0})
    d = summarize.decide(r)
    assert rows(r) == [1, 4, 5]
    assert d["flags"]["fix_variants"] == ["v05_thor_lowq"] and "v05_thor_lowq" in d["matched"][0]["next"]
    assert "V5 PASS" in d["matched"][1]["next"] and "120.0 ms/step" in d["matched"][2]["next"]
    # (2): NVIDIA Vulkan broken, lavapipe fine; V4 not run -> admin question
    r = results(v01_gl_tri_S=True, v02_vk_nvS_img=False, v02_vk_nvS_env=False, v02_vk_lvp=True,
                v03_thor_diag_S=False, v03_thor_diag_S__metrics={"frames": {"n_step_frames": 40,
                                                                             "frac_rgb_std_ge2": 0.0}})
    d = summarize.decide(r)
    assert rows(r) == [2] and "admin question (a)" in d["matched"][0]["next"]
    r["v04_thor_R"] = {"pass": True, "metrics": {}}
    assert "request graphics capability" in summarize.decide(r)["matched"][0]["next"]
    # (3s): GL compiles but draws nothing, NVIDIA Vulkan fails too, staged overlay only: not distinguishable
    r = results(v01_gl_tri_S=False, v01_gl_tri_S__metrics={"compile_ok": True, "link_ok": True},
                v02_vk_nvS_img=False, v02_vk_nvS_env=False, v02_vk_lvp=False)
    assert rows(r) == ["3s"] and "not distinguishable" in summarize.decide(r)["matched"][0]["reading"]
    # (3): the same with the system stack (R variants) tested and failing too
    r3 = results(v01_gl_tri_S=False, v01_gl_tri_S__metrics={"compile_ok": True, "link_ok": True},
                 v01_gl_tri_R=False, v01_gl_tri_R__metrics={"compile_ok": True, "link_ok": True},
                 v02_vk_nvS_img=False, v02_vk_nvS_env=False, v02_vk_nvR=False, v02_vk_lvp=False)
    assert rows(r3) == [3] and summarize.decide(r3)["flags"]["r_tested"] == ["v01_gl_tri_R", "v02_vk_nvR"]
    # a GL compile failure is not row 3
    r["v01_gl_tri_S"]["metrics"]["compile_ok"] = False
    assert rows(r) == [None]
    # (0): V3 renders
    r = results(v01_gl_tri_S=True, v02_vk_nvS_img=True, v03_thor_diag_S=True,
                v03_thor_diag_S__metrics={"frames": {"n_step_frames": 40, "frac_rgb_std_ge2": 1.0}})
    assert rows(r) == [0]
    # V3 never started: not "black", so no row 1
    r = results(v01_gl_tri_S=True, v02_vk_nvS_img=True, v03_thor_diag_S=False,
                v03_thor_diag_S__metrics={"frames": {"n_step_frames": 0, "frac_rgb_std_ge2": None}})
    assert rows(r) == [None]


def test_summarize_writes_summary_and_views(base):
    r = results(v01_gl_tri_S=True, v02_vk_nvS_img=True, v03_thor_diag_S=False, v03_thor_diag_S__metrics=BLACK_V3)
    for n, v in r.items():
        wjson(base / n / "result.json", v)
    (base / "v03_thor_diag_S" / "rgb.png").write_bytes(b"png")
    (base / "v01_gl_tri_S" / "tri.png").write_bytes(b"png")
    s = summarize.summarize(str(base))
    got = json.loads((base / "summary.json").read_text())
    assert got["matched"] == s["matched"] and [x["row"] for x in got["matched"]] == [1, 4]
    assert got["variants_pass"]["v04_thor_R"] is None and got["variants_pass"]["v03_thor_diag_S"] is False
    assert got["view"] == {"v01_gl_tri_S": ["view/v01_gl_tri_S__tri.png"],
                           "v03_thor_diag_S": ["view/v03_thor_diag_S__rgb.png"]}
    assert (base / "view" / "v03_thor_diag_S__rgb.png").exists()
    assert got["variants"]["v03_thor_diag_S"]["key"]["frac_depth_varies"] == 1.0


# --- debug_job.sh on a fake staged tree --------------------------------------------------------------------------

FAKE_PY = r"""#!/usr/bin/env bash
# fake probe interpreter: interpreter flags, then debug_probe.py, the subcommand and its args
flags=""; while [[ $1 == -* ]]; do flags="$flags $1"; shift; done
script=$1; shift
argv=("$@")
sub=$1; shift
all="$*"
out=""
while [[ $# -gt 0 ]]; do case $1 in --out) out=$2; shift 2;; *) shift;; esac; done
name=$(basename "$out")
mkdir -p "$out"
echo "$flags $(basename "$script") $sub" > "$out/fake_flags.txt"
printf '%s\n' "$all" > "$out/fake_args.txt"
printf '%s\n' "${argv[@]}" > "$out/fake_argv.txt"
env | grep -E '^(LD_LIBRARY_PATH|VK_ICD_FILENAMES|VK_DRIVER_FILES|__EGL_VENDOR_LIBRARY_FILENAMES|__EGL_VENDOR_LIBRARY_DIRS|HOME|XDG_DATA_HOME|XDG_CACHE_HOME|__NV_PRIME_RENDER_OFFLOAD|CUDA_VISIBLE_DEVICES|LP_NUM_THREADS)=' | sort > "$out/fake_env.txt"
if [[ -f "@C@/$name.json" ]]; then cp "@C@/$name.json" "$out/result.json"
else echo '{"pass": true, "stage": "done", "metrics": {}, "notes": []}' > "$out/result.json"; fi
if [[ $sub == thor && $name == v03_thor_diag_S ]]; then
  "$HOME/.ai2thor/releases/@B@/@B@" < /dev/null > /dev/null 2>&1 &
  echo $! > "$out/fake_unity.pid"
  mkdir -p "$HOME/.ai2thor/log"; echo "INFO: Found ICD manifest file $VK_ICD_FILENAMES" > "$HOME/.ai2thor/log/unity.log"
fi
grep -q '"pass": true' "$out/result.json"
"""
FAKE_PLAYER = "#!/bin/sh\ntrap '' TERM\nwhile :; do sleep 1; done\n"
FAKE_SMI = r"""#!/usr/bin/env bash
case "$*" in
  *-q*) printf 'Driver Version : 595.71.05\nGPU 00000000:18:00.0\n    Product Name                          : NVIDIA H800\n    MIG Mode\n        Current                           : Disabled\n' ;;
  *-L*) echo "GPU 0: NVIDIA H800 (UUID: GPU-9a55b2fc-6fcb-c7f7-eb64-b35ae469e3ff)" ;;
  *--query-gpu*) while :; do echo "2026/11/09 10:00:00.000, 3, 1755, 1200"; sleep 0.2; done ;;
  *) echo "fake nvidia-smi" ;;
esac
"""


def canned_results(d: Path):
    frames_black = {"frames": {"n_step_frames": 40, "frac_rgb_std_ge2": 0.0, "frac_depth_varies": 1.0,
                               "frac_seg_varies": 1.0}}
    good = {"frames": {"n_step_frames": 40, "frac_rgb_std_ge2": 1.0}}
    can = {"v01_gl_tri_S": (True, {"compile_ok": True, "link_ok": True}),
           "v02_vk_nvS_img": (True, {}), "v02_vk_nvS_env": (True, {}), "v02_vk_lvp": (True, {}),
           "v03_thor_diag_S": (False, frames_black), "v05_thor_lowq": (True, good),
           "v06_thor_newloader": (False, frames_black), "v07_thor_layers": (False, frames_black),
           "v08_thor_direct": (False, frames_black), "v09_open3d_egl": (True, {"std": 30.0}),
           "v10_thor_lvp_forced": (True, dict(good, startup_s=250.0, ms_per_step=90.0))}
    for n, (p, m) in can.items():
        wjson(d / f"{n}.json", {"pass": p, "stage": "done", "metrics": m, "notes": []})


def fake_tree(base: Path, honoured=False, **extra):
    out = base / "vout"
    t = out / "tvr"
    nvgl = t / f"nvgl_{DRV}"
    for rel in ("lib/libGLX_nvidia.so.0", "lib/libEGL_nvidia.so.0", "glvnd/libEGL.so.1", "glvnd/libOpenGL.so.0",
                "vulkan/nvidia_icd.json", "egl/10_nvidia.json"):
        (nvgl / rel).parent.mkdir(parents=True, exist_ok=True)
        (nvgl / rel).write_text("fake\n")
    env_dir = out / "envs" / "tvr"
    (env_dir / "lib").mkdir(parents=True)
    (env_dir / "lib" / "libvulkan.so.1").write_text("fake")
    wjson(env_dir / "share" / "vulkan" / "icd.d" / "lvp_icd.x86_64.json", {"ICD": {"library_path": "/x/lvp.so"}})
    b = t / "ai2thor_home" / ".ai2thor" / "releases" / BUILD_NAME / BUILD_NAME
    b.parent.mkdir(parents=True)
    b.write_text(FAKE_PLAYER)
    b.chmod(0o755)
    canned = base / "canned"
    canned_results(canned)
    bindir = base / "bin"
    bindir.mkdir()
    py = base / "fakepy"
    py.write_text(FAKE_PY.replace("@C@", str(canned)).replace("@B@", BUILD_NAME))
    (bindir / "nvidia-smi").write_text(FAKE_SMI)
    for f in (py, bindir / "nvidia-smi"):
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    sl, vk, dev, drvf = fake_system(base / "sys", honoured)
    env = login_env(VPD_TVR_OUT=str(out), VPD_TVR_PY=str(py), VPD_TVR_TOOL_PY=sys.executable,
                    VPD_TVR_O3D_PY=str(py), VPD_TVR_SYS_LIB=str(sl), VPD_TVR_VK_SYS_DIRS=str(vk),
                    VPD_TVR_EGL_SYS_DIRS=str(base / "sys" / "egl_vendor"), VPD_TVR_DEV_DIR=str(dev),
                    VPD_TVR_DRIVER_FILE=str(drvf), VPD_TVR_REAP_WAIT="10", PATH=f"{bindir}:{os.environ['PATH']}",
                    **extra)
    return env, nvgl, env_dir


def alive(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return st.rsplit(")", 1)[1].split()[0] != "Z"


def run_job(env, outdir):
    r = None
    try:
        r = subprocess.run(["nice", "-n", "19", "bash", str(JOB), str(outdir)], env=env, capture_output=True,
                           text=True, timeout=300)
        r.left_alive = False
        return r
    finally:
        pidf = Path(outdir) / "v03_thor_diag_S" / "fake_unity.pid"
        if pidf.exists():
            pid = int(pidf.read_text())
            if alive(pid):
                os.kill(pid, 9)
                if r is not None:
                    r.left_alive = True


def fenv(d: Path) -> dict:
    return dict(ln.split("=", 1) for ln in (d / "fake_env.txt").read_text().splitlines() if "=" in ln)


def test_job_full_flow_with_fakes(base):
    env, nvgl, env_dir = fake_tree(base)
    od = base / "job" / "out"
    r = run_job(env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    steps = [json.loads(x)["name"] for x in (od / "steps.jsonl").read_text().splitlines()]
    order = [s.split(":")[0] for s in steps if s != "summary"]
    assert order == summarize.VARIANTS and steps[-1] == "summary"
    for n in ("v01_gl_tri_R", "v02_vk_nvR", "v04_thor_R"):
        assert f"{n}:skipped" in steps
        res = json.loads((od / n / "result.json").read_text())
        assert res["pass"] is None and res["skipped"].startswith("caps_not_honored")
    for n in summarize.VARIANTS:
        res = json.loads((od / n / "result.json").read_text())
        assert res["name"] == n
        if not res.get("skipped"):
            assert isinstance(res["rc"], int) and isinstance(res["secs"], int) and res["cap_s"] > 0, n
    s = json.loads((od / "summary.json").read_text())
    assert s["caps_honored"] is False and [x["row"] for x in s["matched"]] == [1, 4, 5]
    assert s["flags"]["fix_variants"] == ["v05_thor_lowq"]
    # environments the probes saw
    e3 = fenv(od / "v03_thor_diag_S")
    assert e3["LD_LIBRARY_PATH"].startswith(f"{nvgl}/lib:{nvgl}/glvnd")
    assert e3["VK_ICD_FILENAMES"] == e3["VK_DRIVER_FILES"] == f"{nvgl}/vulkan/nvidia_icd.json"
    assert e3["HOME"] == str(od / "v03_thor_diag_S" / "home") and "CUDA_VISIBLE_DEVICES" not in e3
    a3 = (od / "v03_thor_diag_S" / "fake_args.txt").read_text()
    for x in ("--gpu-device 0", "--width 640 --height 360", "--quality Ultra --aa 4", "--steps 40 --depth --seg",
              f"--child-env LD_DEBUG_OUTPUT={od}/v03_thor_diag_S/lddebug", "--child-env VK_LOADER_DEBUG=all",
              f"--child-env __GL_SHADER_DISK_CACHE_PATH={od}/v03_thor_diag_S/nvcache"):
        assert x in a3, x
    assert (od / "v03_thor_diag_S" / "fake_flags.txt").read_text().split() == ["-I", "-u", "debug_probe.py",
                                                                                "thor"]
    assert "--quality Very Low --aa none --width 300 --height 300" in (od / "v05_thor_lowq" / "fake_args.txt").read_text()
    assert fenv(od / "v06_thor_newloader")["LD_LIBRARY_PATH"].startswith(f"{env_dir}/lib:{nvgl}/lib")
    e7 = fenv(od / "v07_thor_layers")
    assert e7["__NV_PRIME_RENDER_OFFLOAD"] == "1"
    lj = json.loads((Path(e7["XDG_DATA_HOME"]) / "vulkan" / "implicit_layer.d" / "nvidia_layers.json").read_text())
    assert lj["layers"][0]["library_path"] == f"{nvgl}/lib/libGLX_nvidia.so.0"
    assert "--unity-arg=-force-gfx-direct" in (od / "v08_thor_direct" / "fake_args.txt").read_text()
    e10 = fenv(od / "v10_thor_lvp_forced")
    lvp = f"{env_dir}/share/vulkan/icd.d/lvp_icd.x86_64.json"
    assert e10["VK_ICD_FILENAMES"] == e10["VK_DRIVER_FILES"] == lvp and e10["LD_LIBRARY_PATH"].startswith(f"{env_dir}/lib")
    assert e10["CUDA_VISIBLE_DEVICES"] == "0" and e10["LP_NUM_THREADS"] == "12"
    assert json.loads((od / "v10_thor_lvp_forced" / "home" / ".ai2thor" / "cuda-vulkan-mapping.json").read_text()) \
        == {"0": 0}
    a10 = (od / "v10_thor_lvp_forced" / "fake_args.txt").read_text()
    assert "--gpu-device 0" in a10 and "--quality Very Low --aa none --steps 20 --depth" in a10
    assert re.search(r"--startup-timeout \d+$", a10.strip())
    e1 = fenv(od / "v01_gl_tri_S")
    assert e1["__EGL_VENDOR_LIBRARY_FILENAMES"] == f"{nvgl}/egl/10_nvidia.json"
    assert f"--egl-lib {nvgl}/glvnd/libEGL.so.1 --gl-lib {nvgl}/glvnd/libOpenGL.so.0" in \
        (od / "v01_gl_tri_S" / "fake_args.txt").read_text()
    a2 = (od / "v02_vk_nvS_img" / "fake_args.txt").read_text()
    assert f"--loader {base}/sys/syslib/libvulkan.so.1 --want nvidia" in a2
    assert f"--loader {env_dir}/lib/libvulkan.so.1 --want cpu" in (od / "v02_vk_lvp" / "fake_args.txt").read_text()
    assert fenv(od / "v02_vk_lvp")["VK_ICD_FILENAMES"] == lvp
    assert (od / "v09_open3d_egl" / "fake_flags.txt").read_text().split()[-1] == "open3d"
    # the thor side outputs were parsed (post-thor) and the loader log copied out of the variant HOME
    r3 = json.loads((od / "v03_thor_diag_S" / "result.json").read_text())
    assert r3["metrics"]["post"]["loader"]["manifests"] == [f"{nvgl}/vulkan/nvidia_icd.json"]
    assert r3["metrics"]["post"]["nvcache"]["exists"] is True
    # the fake player (TERM ignored) left by V3 was reaped with KILL
    pid = int((od / "v03_thor_diag_S" / "fake_unity.pid").read_text())
    assert r.left_alive is False and not alive(pid)
    reap = (od / "logs" / "reap.log").read_text()
    assert f"TERM lingering Unity player(s): {pid}" in reap and "KILL" in reap
    facts_res = json.loads((od / "v00_facts" / "result.json").read_text())
    assert facts_res["pass"] is True and facts_res["metrics"]["caps_honored"] is False
    assert facts_res["metrics"]["staging"]["build"] is True and facts_res["metrics"]["staging"]["lvp_icd"] is True
    # every probe argv the job built parses with the real debug_probe.py parser, to the intended values
    parsed = {}
    for n in summarize.VARIANTS[1:]:
        f = od / n / "fake_argv.txt"
        if not f.exists():
            continue
        argv = f.read_text().splitlines()
        p = subprocess.run([sys.executable, "-I", str(PROBE), *argv], env=login_env(VPD_TVR_PARSE_ONLY="1"),
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, (n, argv, p.stderr[-1500:])
        parsed[n] = json.loads(p.stdout)
    assert len(parsed) == 11   # 15 variants - facts - the three skipped R ones
    assert parsed["v08_thor_direct"]["unity_arg"] == ["-force-gfx-direct"]
    assert parsed["v05_thor_lowq"]["quality"] == "Very Low" and parsed["v05_thor_lowq"]["aa"] == "none"
    assert parsed["v05_thor_lowq"]["width"] == 300 and parsed["v05_thor_lowq"]["seg"] is True
    p3 = parsed["v03_thor_diag_S"]
    assert (p3["gpu_device"], p3["quality"], p3["aa"], p3["steps"], p3["depth"], p3["seg"]) == (0, "Ultra", "4", 40,
                                                                                                True, True)
    assert p3["startup_timeout"] == 200.0 and p3["deadline_s"] == 215.0
    assert "LD_DEBUG=libs,files" in p3["child_env"] and "VK_LOADER_DEBUG=all" in p3["child_env"]
    p10 = parsed["v10_thor_lvp_forced"]
    assert (p10["steps"], p10["depth"], p10["seg"], p10["width"]) == (20, True, False, 300)
    assert p10["startup_timeout"] == 660.0 and p10["deadline_s"] == 875.0
    assert parsed["v02_vk_lvp"]["want"] == "cpu" and parsed["v01_gl_tri_S"]["cmd"] == "gl-tri"
    # a second run into the same OUTDIR is refused
    r2 = subprocess.run(["bash", str(JOB), str(od)], env=env, capture_output=True, text=True, timeout=60)
    assert r2.returncode == 2 and "refusing" in r2.stderr


def test_job_runs_r_variants_when_caps_are_honoured(base):
    env, nvgl, env_dir = fake_tree(base, honoured=True, NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics")
    od = base / "job" / "out"
    r = run_job(env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    s = json.loads((od / "summary.json").read_text())
    assert s["caps_honored"] is True
    for n in ("v01_gl_tri_R", "v02_vk_nvR", "v04_thor_R"):
        res = json.loads((od / n / "result.json").read_text())
        assert not res.get("skipped") and res["pass"] is True, n
        e = fenv(od / n)
        assert "VK_ICD_FILENAMES" not in e and "__EGL_VENDOR_LIBRARY_FILENAMES" not in e
        assert "LD_LIBRARY_PATH" not in e or str(nvgl) not in e["LD_LIBRARY_PATH"]
    assert fenv(od / "v01_gl_tri_R")["__EGL_VENDOR_LIBRARY_DIRS"] == str(base / "sys" / "egl_vendor")
    # the image has no glvnd libEGL / libOpenGL here, so R dispatches through the staged copies
    assert f"--egl-lib {nvgl}/glvnd/libEGL.so.1" in (od / "v01_gl_tri_R" / "fake_args.txt").read_text()


def test_job_soft_deadline_skips_but_still_summarises(base):
    env, _, _ = fake_tree(base, VPD_JOB_DEADLINE_S="1", VPD_TVR_RESERVE_S="0", VPD_TVR_MIN_VARIANT_S="30")
    od = base / "job" / "out"
    r = run_job(env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    for n in summarize.VARIANTS:
        res = json.loads((od / n / "result.json").read_text())
        assert res["pass"] is None and res["skipped"], n
    assert json.loads((od / "v00_facts" / "result.json").read_text())["skipped"].startswith("no_time")
    s = json.loads((od / "summary.json").read_text())
    assert [x["row"] for x in s["matched"]] == [None]


def test_job_staging_missing_skips_s_variants(base):
    env, nvgl, _ = fake_tree(base)
    (nvgl / "vulkan" / "nvidia_icd.json").unlink()
    od = base / "job" / "out"
    r = run_job(env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    for n in ("v01_gl_tri_S", "v02_vk_nvS_img", "v03_thor_diag_S", "v05_thor_lowq", "v07_thor_layers"):
        assert json.loads((od / n / "result.json").read_text())["skipped"].startswith("staged overlay incomplete"), n
    assert json.loads((od / "v02_vk_lvp" / "result.json").read_text())["pass"] is True   # lavapipe needs no overlay
    assert json.loads((od / "v10_thor_lvp_forced" / "result.json").read_text())["pass"] is True
