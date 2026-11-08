"""Tests of WP-TVR: vpd/tvr/{verdict.py, thor_smoke.py, egl_probe.py, stage_tvr_login.sh, tvr_smoke.sh} and the
frozen thresholds of vpd/tvr/SMOKE_CRITERIA.md.

Login node, 1 core, no GPU, no network: no Controller, no vulkaninfo, no EGL call, no real nvidia-smi (a fake one
is put first on PATH for the shell runs).  The staging script runs on fake source libs; the smoke script runs on a
fake staged tree with a fake interpreter.  Scratch lives under eval_external_assets/vpd/scratch/tvr/pytest_<rand>/
(env VPD_TVR_TEST_DIR overrides) and is removed at teardown.
    PYTHONPATH=$WT OMP_NUM_THREADS=1 PY_EVAL -m pytest -q -p no:cacheprovider vpd/tests/test_tvr.py
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
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

TVR = paths.VPD_PKG / "tvr"
STAGE = TVR / "stage_tvr_login.sh"
SMOKE = TVR / "tvr_smoke.sh"
DRV = "595.71.05"
UUID = "b67ed4f9-1111-2222-3333-444455556666"
LIBS = ["libGLX_nvidia", "libEGL_nvidia", "libnvidia-glcore", "libnvidia-eglcore", "libnvidia-glsi",
        "libnvidia-tls", "libnvidia-glvkspirv", "libnvidia-gpucomp", "libnvidia-rtcore", "libnvidia-allocator",
        "libGLESv2_nvidia", "libGLESv1_CM_nvidia"]
BUILD_NAME = "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"vpd_tvr_{name}", TVR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verdict = _load("verdict")
thor = _load("thor_smoke")
eglp = _load("egl_probe")


@pytest.fixture
def base():
    root = Path(os.environ.get("VPD_TVR_TEST_DIR") or paths.scratch("tvr"))
    d = root / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def login_env(**extra):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", VPD_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for k in ("VPD_TVR_NET_OK", "RUN", "CUDA_VISIBLE_DEVICES"):
        env.pop(k, None)
    env.update(extra)
    return env


# --- static checks --------------------------------------------------------------------------------------------

def test_bash_syntax():
    for f in (STAGE, SMOKE):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)


def test_shell_paths_mirror_paths_py():
    assert paths.TVR_OUT == paths.OUT / "tvr"
    assert paths.NVGL == paths.TVR_OUT / f"nvgl_{DRV}"
    assert paths.TVR_ENV == paths.OUT / "envs" / "tvr"
    assert paths.AI2THOR_HOME == paths.TVR_OUT / "ai2thor_home"
    assert paths.TVR_REPO == paths.TVR_OUT / "repos" / "TVRBench"
    st = STAGE.read_text()
    for line in (f"OUT=${{VPD_TVR_OUT:-{paths.OUT}}}", f"DRV={DRV}", "T=$OUT/tvr", "NVGL=$T/nvgl_$DRV",
                 "ENV=$OUT/envs/tvr", "AI2THOR_HOME=$T/ai2thor_home", "REPO=$T/repos/TVRBench"):
        assert re.search(rf"(?m)^{re.escape(line)}\s*(#.*)?$", st), line
    sm = SMOKE.read_text()
    for line in ("VOUT=${VPD_TVR_OUT:-$VPD_OUT}", f"DRV={DRV}", "T=$VOUT/tvr", "NVGL=$T/nvgl_$DRV",
                 "ENV=$VOUT/envs/tvr", "AI2THOR_HOME=$T/ai2thor_home", "REPO=$T/repos/TVRBench",
                 f"BUILD_NAME={BUILD_NAME}"):
        assert re.search(rf"(?m)^{re.escape(line)}\s*(#.*)?$", sm), line
    assert f"export VPD_OUT={paths.OUT}" in (paths.VPD_PKG / "jobs" / "common_env.sh").read_text()
    commit = re.search(r"(?m)^COMMIT=([0-9a-f]{40})", st).group(1)
    assert BUILD_NAME.endswith(commit)
    # the 12 libs of the WP spec, in the staging script
    arr = re.search(r"LIBS=\(([^)]*)\)", st).group(1).split()
    assert arr == LIBS


def test_criteria_doc_matches_verdict_constants():
    doc = (TVR / "SMOKE_CRITERIA.md").read_text()
    block = re.search(r"```thresholds\n(.*?)```", doc, re.S).group(1)
    kv = dict(line.split(" = ", 1) for line in block.strip().splitlines())
    assert kv.pop("DRIVER") == verdict.DRIVER == DRV
    assert kv.pop("GPU_NAME") == verdict.GPU_NAME
    assert int(kv.pop("STEPS")) == verdict.STEPS == 300
    assert float(kv.pop("STARTUP_MAX_S")) == verdict.STARTUP_MAX_S == 300
    assert float(kv.pop("DISTINCT_MIN")) == verdict.DISTINCT_MIN == 0.90
    assert float(kv.pop("SUCCESS_MIN")) == verdict.SUCCESS_MIN == 0.95
    assert int(kv.pop("NPROC")) == verdict.NPROC == 4
    assert int(kv.pop("NPROC_STEPS")) == verdict.NPROC_STEPS == 150
    assert tuple(int(x) for x in kv.pop("EGL_RGBA").split(",")) == verdict.EGL_RGBA == (51, 102, 153, 255)
    assert int(kv.pop("EGL_TOL")) == verdict.EGL_TOL == 2
    assert not kv, kv
    # the smoke script asks thor_smoke for exactly the frozen step counts
    sm = SMOKE.read_text()
    assert f"--steps {verdict.STEPS} --gpu 0" in sm
    assert f"--steps {verdict.NPROC_STEPS} --gpu 0 --nproc {verdict.NPROC}" in sm
    assert "timeout 1800" in sm
    # the criterion names of the doc are the verdict's
    for n in ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "E0", "E1", "E2", "E3", "E4", "E5"):
        assert f"| {n} |" in doc, n


# --- verdict parsers ------------------------------------------------------------------------------------------

SUMMARY = f"""==========
VULKANINFO
==========

Vulkan Instance Version: 1.4.357


Instance Extensions: count = 24
-------------------------------
VK_EXT_debug_report : extension revision 10

Devices:
========
GPU0:
\tapiVersion         = 1.4.329
\tdriverVersion      = 595.71.5.0
\tvendorID           = 0x10de
\tdeviceID           = 0x2324
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = NVIDIA H800
\tdriverID           = DRIVER_ID_NVIDIA_PROPRIETARY
\tdriverName         = NVIDIA
\tdriverInfo         = {DRV}
\tconformanceVersion = 1.4.1.0
\tdeviceUUID         = {UUID}
\tdriverUUID         = 0f0e0d0c-0b0a-0908-0706-050403020100
"""
SMI_L = f"GPU 0: NVIDIA H800 (UUID: GPU-{UUID})\n"
DRIVER_TXT = (f"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  {DRV}  Release Build  "
              "(dvs-builder@U22-I3-G08-03-1)  Fri Apr 24 06:42:30 UTC 2026\n")


def test_parsers():
    devs = verdict.parse_vulkaninfo_summary(SUMMARY)
    assert len(devs) == 1 and devs[0]["deviceName"] == "NVIDIA H800" and devs[0]["deviceUUID"] == UUID
    assert devs[0]["deviceType"] == verdict.DISCRETE and devs[0]["driverInfo"] == DRV
    assert verdict.parse_vulkaninfo_ai2thor(SUMMARY) == ({UUID: 0}, True)
    assert verdict.parse_smi_L(SMI_L) == {0: UUID}
    assert verdict.driver_ok(DRIVER_TXT)
    assert not verdict.driver_ok(DRIVER_TXT.replace(DRV, "595.71.051"))
    assert not verdict.driver_ok("NVRM version: ... 550.90.07 ...")
    assert not verdict.driver_ok(None)
    # ai2thor's parser fails when deviceUUID appears before any GPU<n>: header
    assert verdict.parse_vulkaninfo_ai2thor(f"\tdeviceUUID = {UUID}\n")[1] is False


def test_mapping_modes():
    m = verdict.cuda_vulkan_mapping(SUMMARY, SMI_L, SUMMARY)
    assert m["mode"] == "ai2thor_native" and m["mapping"] == {"0": 0}
    # full output without UUID lines but one Vulkan device in the summary -> 1-card fallback
    no_uuid = "\n".join(ln for ln in SUMMARY.splitlines() if "deviceUUID" not in ln)
    m = verdict.cuda_vulkan_mapping(no_uuid, SMI_L, no_uuid)
    assert m["mode"] == "fallback" and m["mapping"] == {"0": 0}
    two = no_uuid + "\nGPU1:\n\tdeviceName = llvmpipe\n"
    assert verdict.cuda_vulkan_mapping(two, SMI_L, two)["mode"] == "impossible"
    assert verdict.cuda_vulkan_mapping(SUMMARY, "", SUMMARY)["mode"] == "impossible"


def test_player_log_check():
    ok = verdict.player_log_check(["[Vulkan init] SelectPhysicalDevice requestedDeviceIndex=0\n"
                                   "Vulkan renderer=[NVIDIA H800] id=[10de:2324]\n"])
    assert ok["ok"]
    sw = verdict.player_log_check(["Vulkan renderer=[NVIDIA H800]\nRenderer: llvmpipe (LLVM 12.0.0)\n"])
    assert not sw["ok"] and sw["software_lines"]
    assert not verdict.player_log_check(["nothing here\n"])["ok"]
    # plan 3.7 names the H800 *Vulkan* renderer: a card name without any Vulkan line is not enough
    gl = verdict.player_log_check(["Renderer: NVIDIA H800/PCIe/SSE2\n"])
    assert not gl["ok"] and gl["vulkan_lines"] == []
    split = verdict.player_log_check(["Vulkan API version in use: 1.3.0\n", "Renderer: NVIDIA H800\n"])
    assert split["ok"] and split["vulkan_lines"] == ["Vulkan API version in use: 1.3.0"]


# --- verdict end to end on a synthetic OUTDIR -----------------------------------------------------------------

def wjson(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def make_pass(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "driver_version.txt").write_text(DRIVER_TXT)
    wjson(out / "staging.json", {"nvgl_lib": True, "glvnd": True, "vulkan_icd": True, "egl_vendor": True,
                                 "env_python": True, "vulkaninfo": True, "build": True, "tvrbench": True,
                                 "procthor": "ok"})
    wjson(out / "facts.json", {"nvidia_modeset": True, "dev_dri": True, "tmp_free_bytes": 10 ** 11})
    (out / "nvidia_smi_L.txt").write_text(SMI_L)
    (out / "vulkaninfo_summary.txt").write_text(SUMMARY)
    (out / "vulkaninfo_full.txt").write_text(SUMMARY)
    wjson(out / "thor1" / "result.json", {
        "raw": {"startup_s": 21.5, "steps": 300, "bad_frames": 0, "distinct_frames": 299, "action_success": 300,
                "steps_per_s": 41.2, "hd_1280x720_steps_per_s": 22.0},
        "tasks": [{"task_id": "ithor_FloorPlan407_easy_t07_s00", "dataset": "ithor", "ok": True,
                   "distinct_start_target": True},
                  {"task_id": "procthor_00012_easy", "dataset": "procthor", "ok": True,
                   "distinct_start_target": True}],
        "pass": True})
    (out / "thor1" / "Player_raw.log").write_text("Vulkan renderer=[NVIDIA H800] id=[10de:2324]\n")
    wjson(out / "thor4" / "result.json", {"procs": [{"steps": 150, "bad_frames": 0, "steps_per_s": 30.0}] * 4,
                                          "agg_steps_per_s": 120.0, "pass": True})
    wjson(out / "egl" / "result.json", {"num_devices": 1, "egl_version": "1.5", "EGL_VENDOR": "NVIDIA",
                                        "GL_RENDERER": "NVIDIA H800/PCIe/SSE2", "readback_px0": [51, 102, 153, 255],
                                        "stage": "readback", "pass": True})


def edit(out: Path, rel: str, fn):
    p = out / rel
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


def test_verdict_pass(base):
    make_pass(base)
    v = verdict.verdict(str(base))
    assert (v["TVR_GATE"], v["EGL_GATE"]) == ("PASS", "PASS"), v["reasons"]
    assert v["reasons"] == [] and v["caveats"] == []
    assert v["steps_per_s"] == {"single_640x360": 41.2, "single_1280x720": 22.0, "nproc4_aggregate_640x360": 120.0,
                                "lavapipe_640x360": None}
    assert json.loads((base / "verdict.json").read_text())["TVR_GATE"] == "PASS"
    assert "TVR pilot may be planned" in v["next"]


def _set(rel, path, value):
    def fn(d):
        x = d
        for k in path[:-1]:
            x = x[k]
        x[path[-1]] = value
    return rel, fn


MUTATIONS = {
    "staging_env": (_set("staging.json", ["env_python"], False), {"T0_staging"}),
    "vk_integrated": (None, {"T2_vulkan_device"}),
    "vk_uuid": (None, {"T2_vulkan_device"}),
    "startup_300": (_set("thor1/result.json", ["raw", "startup_s"], 300.0), {"T3_startup"}),
    "bad_frame": (_set("thor1/result.json", ["raw", "bad_frames"], 1), {"T4_bad_frames"}),
    "distinct_269": (_set("thor1/result.json", ["raw", "distinct_frames"], 269), {"T5_frames_actions"}),
    "success_284": (_set("thor1/result.json", ["raw", "action_success"], 284), {"T5_frames_actions"}),
    "steps_299": (_set("thor1/result.json", ["raw", "steps"], 299), {"T5_frames_actions"}),
    "raw_error": (_set("thor1/result.json", ["raw"], {"error": "RuntimeError('vulkaninfo failed')"}),
                  {"T3_startup", "T4_bad_frames", "T5_frames_actions"}),
    # start-up measured, then a step raised: T3 holds, T4/T5 fail
    "raw_error_after_start": (_set("thor1/result.json", ["raw"], {
        "startup_s": 20.0, "phase": "stepping", "progress": {"steps_done": 100, "bad_frames": 0},
        "error": "TimeoutError()"}), {"T4_bad_frames", "T5_frames_actions"}),
    # killed mid-way (last checkpoint, no error): the partial counters are never the gated ones
    "raw_killed_stepping": (_set("thor1/result.json", ["raw"], {
        "startup_s": 20.0, "phase": "stepping", "progress": {"steps_done": 250, "bad_frames": 0,
                                                             "distinct_frames": 250, "action_success": 250}}),
        {"T4_bad_frames", "T5_frames_actions"}),
    "tasks_time_budget": (_set("thor1/result.json", ["tasks"], [
        {"task_id": None, "dataset": None, "ok": False, "skipped": "time_budget", "left_s": 100.0}]),
        {"T7_eval_tasks"}),
    "player_no_vulkan": (None, {"T6_player_log"}),
    "ithor_bad": (_set("thor1/result.json", ["tasks", 0, "ok"], False), {"T7_eval_tasks"}),
    "procthor_err": (_set("thor1/result.json", ["tasks", 1], {"task_id": "p", "dataset": "procthor", "ok": False,
                                                              "error": "x"}), {"T7_eval_tasks"}),
    "par_3": (_set("thor4/result.json", ["procs"], [{"steps": 150, "bad_frames": 0}] * 3), {"T8_parallel"}),
    "par_bad": (_set("thor4/result.json", ["procs", 2], {"steps": 150, "bad_frames": 2}), {"T8_parallel"}),
    "par_err": (_set("thor4/result.json", ["procs", 1], {"error": "Timeout"}), {"T8_parallel"}),
    "egl_0dev": (_set("egl/result.json", ["num_devices"], 0), {"E1_devices"}),
    "egl_noinit": (_set("egl/result.json", ["egl_version"], None), {"E2_initialize"}),
    "egl_mesa": (_set("egl/result.json", ["EGL_VENDOR"], "Mesa Project"), {"E3_vendor"}),
    "egl_llvmpipe": (_set("egl/result.json", ["GL_RENDERER"], "llvmpipe (LLVM 15)"), {"E4_renderer"}),
    "egl_px": (_set("egl/result.json", ["readback_px0"], [48, 102, 153, 255]), {"E5_readback"}),
}


@pytest.mark.parametrize("case", sorted(MUTATIONS))
def test_verdict_each_criterion_fails(base, case):
    make_pass(base)
    mut, want = MUTATIONS[case]
    if case == "vk_integrated":
        (base / "vulkaninfo_summary.txt").write_text(SUMMARY.replace("DISCRETE_GPU", "INTEGRATED_GPU"))
    elif case == "vk_uuid":
        (base / "nvidia_smi_L.txt").write_text(SMI_L.replace("b67ed4f9", "deadbeef"))
    elif case == "player_no_vulkan":  # an OpenGL-style renderer line names the card but no Vulkan line
        (base / "thor1" / "Player_raw.log").write_text("Renderer: NVIDIA H800/PCIe/SSE2\n")
    else:
        edit(base, *mut)
    v = verdict.verdict(str(base))
    failed = {r["name"] for r in v["criteria"]["tvr"] + v["criteria"]["egl"] if not r["ok"]}
    assert failed == want, (case, failed)
    tvr_fail = any(n.startswith("T") for n in want)
    assert v["TVR_GATE"] == ("FAIL" if tvr_fail else "PASS")
    assert v["EGL_GATE"] == ("PASS" if tvr_fail else "FAIL")  # the gates are independent


def test_verdict_boundaries_pass(base):
    make_pass(base)
    edit(base, "thor1/result.json", lambda d: d["raw"].update(startup_s=299.9, distinct_frames=270,
                                                               action_success=285))
    edit(base, "egl/result.json", lambda d: d.update(readback_px0=[53, 100, 155, 253]))
    v = verdict.verdict(str(base))
    assert (v["TVR_GATE"], v["EGL_GATE"]) == ("PASS", "PASS"), v["reasons"]


def test_verdict_player_log(base):
    make_pass(base)
    (base / "thor1" / "Player_tasks.log").write_text("GfxDevice: creating device client\nRenderer: llvmpipe\n")
    v = verdict.verdict(str(base))
    assert v["TVR_GATE"] == "FAIL" and any(r.startswith("T6_player_log") for r in v["reasons"])
    for f in (base / "thor1").glob("Player*.log"):
        f.unlink()
    v = verdict.verdict(str(base))
    assert any(r.startswith("T6_player_log: no thor1") for r in v["reasons"])


def test_verdict_procthor_skipped_is_a_caveat(base):
    make_pass(base)
    edit(base, "thor1/result.json", lambda d: d["tasks"].__setitem__(
        1, {"task_id": "procthor_00012_easy", "dataset": "procthor", "ok": False, "skipped": "no_procthor_data"}))
    v = verdict.verdict(str(base))
    assert v["TVR_GATE"] == "PASS" and v["caveats"] and v["caveats"][0].startswith("procthor_not_tested")


def test_verdict_driver_and_missing_files(base):
    make_pass(base)
    (base / "driver_version.txt").write_text(DRIVER_TXT.replace(DRV, "550.90.07"))
    v = verdict.verdict(str(base))
    assert (v["TVR_GATE"], v["EGL_GATE"]) == ("FAIL", "FAIL")
    assert "driver mismatch" in v["next"]
    empty = base / "empty"
    empty.mkdir()
    v = verdict.verdict(str(empty))
    assert (v["TVR_GATE"], v["EGL_GATE"]) == ("FAIL", "FAIL")
    assert len(v["reasons"]) >= 10


def test_next_step_modeset_branch(base):
    make_pass(base)
    (base / "vulkaninfo_summary.txt").write_text("ERROR: [Loader Message] vkCreateInstance failed\n")
    (base / "vulkaninfo_full.txt").write_text("")
    wjson(base / "facts.json", {"nvidia_modeset": False, "dev_dri": False})
    v = verdict.verdict(str(base))
    assert v["TVR_GATE"] == "FAIL" and "NVIDIA_DRIVER_CAPABILITIES=graphics" in v["next"]
    wjson(base / "facts.json", {"nvidia_modeset": True})
    assert "overlay problem" in verdict.verdict(str(base))["next"]


def test_write_mapping(base):
    make_pass(base)
    home = base / "home"
    no_uuid = "\n".join(ln for ln in SUMMARY.splitlines() if "deviceUUID" not in ln)
    (base / "vulkaninfo_full.txt").write_text(no_uuid)
    (base / "vulkaninfo_summary.txt").write_text(no_uuid)
    m = verdict.write_mapping(str(base), str(home))
    assert m["mode"] == "fallback"
    assert json.loads((home / ".ai2thor" / "cuda-vulkan-mapping.json").read_text()) == {"0": 0}
    (base / "vulkaninfo_full.txt").write_text(SUMMARY)
    shutil.rmtree(home)
    assert verdict.write_mapping(str(base), str(home))["mode"] == "ai2thor_native"
    assert not (home / ".ai2thor" / "cuda-vulkan-mapping.json").exists()  # ai2thor writes its own


# --- thor_smoke with a fake Controller / ThorEnv ----------------------------------------------------------------

class FakeEvent:
    def __init__(self, frame, ok=True, ret=None):
        self.frame = frame
        self.metadata = {"lastActionSuccess": ok, "actionReturn": ret, "screenWidth": 640, "screenHeight": 360}


class FakeController:
    """Fake ai2thor Controller.  hd_fail: ChangeResolution raises.  raise_at=k: the k-th move raises `exc`
    (KeyboardInterrupt stands in for the outer `timeout` killing the process: `except Exception` does not catch
    it, so only what thor_smoke already wrote survives)."""

    def __init__(self, bad_at=(), fail_at=(), const=False, hd_fail=False, raise_at=None, exc=RuntimeError):
        self.i, self.bad_at, self.fail_at, self.const = 0, set(bad_at), set(fail_at), const
        self.hd_fail, self.raise_at, self.exc = hd_fail, raise_at, exc
        self.rng = np.random.default_rng(1)
        self.res = (360, 640)
        self.stopped = False
        self.last_event = FakeEvent(self._frame())

    def _frame(self):
        if self.const:
            return np.zeros(self.res + (3,), np.uint8)
        return self.rng.integers(0, 255, self.res + (3,), dtype=np.uint8)

    def step(self, action=None, **kw):
        if action == "GetReachablePositions":
            ev = FakeEvent(self._frame(), True, [{"x": 0.25 * k, "y": 0.9, "z": 0.0} for k in range(5)])
        elif action == "ChangeResolution":
            if self.hd_fail:
                raise RuntimeError("ChangeResolution failed")
            self.res = (kw["y"], kw["x"])
            ev = FakeEvent(self._frame())
        else:
            self.i += 1
            if self.raise_at is not None and self.i == self.raise_at:
                raise self.exc(f"fake failure at move {self.i}")
            ev = FakeEvent(None if self.i in self.bad_at else self._frame(), self.i not in self.fail_at)
        self.last_event = ev
        return ev

    def stop(self):
        self.stopped = True


def test_is_bad():
    assert thor.is_bad(None)
    assert thor.is_bad(np.zeros((360, 640, 3), np.uint8))
    assert thor.is_bad(np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8))
    assert not thor.is_bad(np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8))


def test_raw_phase_counts(base):
    made = []

    def factory(gpu):
        made.append(FakeController(bad_at={3}, fail_at={5, 6}))
        return made[-1], 1.5

    r = thor.raw_phase(str(base), 20, 0, factory=factory)
    assert made[0].stopped
    assert (r["steps"], r["bad_frames"], r["distinct_frames"], r["action_success"]) == (20, 1, 19, 18)
    assert r["startup_s"] == 1.5 and r["hd_bad_frames"] == 0 and r["hd_frame"]["shape"] == [720, 1280, 3]
    assert sorted(p.name for p in base.glob("raw_*.png")) == ["raw_0000.png", "raw_1280x720.png"]
    r = thor.raw_phase(str(base), 8, None, save=False, factory=lambda g: (FakeController(const=True), 0.1))
    assert r["bad_frames"] == 8 and r["distinct_frames"] == 0 and "hd_1280x720_steps_per_s" not in r


def test_raw_phase_hd_failure_keeps_counts(base):
    snaps = []
    c = FakeController(hd_fail=True)
    r = thor.raw_phase(str(base), 120, 0, factory=lambda g: (c, 3.0), progress=snaps.append)
    assert c.stopped and r["phase"] == "done" and "ChangeResolution failed" in r["hd_error"]
    assert (r["steps"], r["bad_frames"], r["distinct_frames"], r["action_success"]) == (120, 0, 120, 120)
    assert "hd_1280x720_steps_per_s" not in r and "progress" not in r
    # checkpoints: start-up, every 50 moves, end of the 640x360 steps, after the HD block
    assert [s["phase"] for s in snaps] == ["started", "stepping", "stepping", "steps_done", "done"]
    assert snaps[0] == {"phase": "started", "startup_s": 3.0}
    assert snaps[2]["progress"]["steps_done"] == 100 and "bad_frames" not in snaps[2]
    # the same through run(): the HD error is reported, the 640x360 pass and phase B are kept
    a = thor.parse_args(["--out", str(base / "t"), "--steps", "40", "--no-tasks"])
    res = thor.run(a, factory=lambda g: (FakeController(hd_fail=True), 1.0))
    assert res["pass"] and res["raw"]["steps"] == 40 and "hd_error" in res["raw"]
    disk = json.loads((base / "t" / "result.json").read_text())
    assert disk["raw"]["distinct_frames"] == 40 and disk["raw"]["hd_error"]


def test_run_partial_result_survives_a_kill(base):
    """A kill (KeyboardInterrupt here) mid-run leaves the last checkpoint in result.json, and verdict reads it."""
    out = base / "k1"
    a = thor.parse_args(["--out", str(out), "--steps", "300", "--no-tasks"])
    with pytest.raises(KeyboardInterrupt):
        thor.run(a, factory=lambda g: (FakeController(raise_at=120, exc=KeyboardInterrupt), 7.0))
    raw = json.loads((out / "result.json").read_text())["raw"]
    assert raw["phase"] == "stepping" and raw["progress"]["steps_done"] == 100 and "steps" not in raw
    assert not list(out.glob("*.part"))  # atomic writes leave no temp file behind
    make_pass(base / "v")
    shutil.copyfile(out / "result.json", base / "v" / "thor1" / "result.json")
    v = verdict.verdict(str(base / "v"))
    failed = {r["name"] for r in v["criteria"]["tvr"] if not r["ok"]}
    assert failed == {"T4_bad_frames", "T5_frames_actions", "T7_eval_tasks"}, failed  # T3 measured: 7 s
    assert any("phase=stepping" in r for r in v["reasons"])
    # a kill during phase B keeps phase A and the task rows finished so far
    repo = make_repo(base / "repo")

    class KilledAtHouse(FakeEnv):
        def load_scene(self, scene):
            if isinstance(scene, dict):
                raise KeyboardInterrupt
            super().load_scene(scene)

    out2 = base / "k2"
    a = thor.parse_args(["--out", str(out2), "--repo", str(repo), "--steps", "12"])
    with pytest.raises(KeyboardInterrupt):
        thor.run(a, factory=lambda g: (FakeController(), 2.0), env_cls=KilledAtHouse)
    d = json.loads((out2 / "result.json").read_text())
    assert d["raw"]["phase"] == "done" and d["raw"]["steps"] == 12 and d["pass"] is False
    assert [t["dataset"] for t in d["tasks"]] == ["ithor", "procthor"]
    assert d["tasks"][0]["ok"] is True and d["raw_pass"] is True
    assert d["tasks"][1]["ok"] is False and d["tasks"][1]["error"] == "interrupted"


def test_run_deadline_skips_phase_b(base):
    repo = make_repo(base / "repo")
    a = thor.parse_args(["--out", str(base / "d"), "--repo", str(repo), "--steps", "8", "--deadline-s", "60"])
    FakeEnv.instances.clear()
    res = thor.run(a, factory=lambda g: (FakeController(), 1.0), env_cls=FakeEnv)
    assert not FakeEnv.instances  # no second Controller was started
    assert res["tasks"][0]["skipped"] == "time_budget" and res["tasks"][0]["need_s"] == thor.PHASE_B_MIN_S
    assert res["pass"] is False and res["raw"]["steps"] == 8
    a = thor.parse_args(["--out", str(base / "e"), "--repo", str(repo), "--steps", "8", "--deadline-s", "990"])
    assert thor.run(a, factory=lambda g: (FakeController(), 1.0), env_cls=FakeEnv)["tasks_ok"] == 2


def test_run_nproc_pool_branch(base, monkeypatch):
    from multiprocessing.pool import ThreadPool
    import threading
    monkeypatch.setattr(thor, "STAGGER_S", 0)
    home = base / "home"
    log = home / thor.PLAYER_LOG
    log.parent.mkdir(parents=True)
    log.write_text("Vulkan renderer=[NVIDIA H800]\n")
    monkeypatch.setenv("HOME", str(home))
    a = thor.parse_args(["--out", str(base / "p"), "--steps", "60", "--nproc", "3", "--no-tasks"])
    res = thor.run(a, factory=lambda g: (FakeController(), 0.5), pool_factory=ThreadPool)
    assert res["pass"] and len(res["procs"]) == 3 and all(p["steps"] == 60 for p in res["procs"])
    assert res["agg_steps_per_s"] == round(sum(p["steps_per_s"] for p in res["procs"]), 2)
    assert "hd_bad_frames" in res["procs"][0] and "hd_bad_frames" not in res["procs"][1]  # only p0 saves
    assert (base / "p" / "p0_0000.png").exists() and (base / "p" / "p0_1280x720.png").exists()
    assert not list((base / "p").glob("p1_*.png"))
    assert (base / "p" / "Player_parallel.log").exists()
    assert len(json.loads((base / "p" / "result.json").read_text())["procs"]) == 3
    # one worker fails mid-way: its row keeps the last checkpoint plus the error, and the run fails
    lock, calls = threading.Lock(), []

    def factory(gpu):
        with lock:
            calls.append(1)
            k = len(calls)
        return (FakeController(raise_at=55) if k == 2 else FakeController()), 0.5

    a = thor.parse_args(["--out", str(base / "q"), "--steps", "60", "--nproc", "3", "--no-tasks"])
    res = thor.run(a, factory=factory, pool_factory=ThreadPool)
    bad = [p for p in res["procs"] if "error" in p]
    assert not res["pass"] and len(bad) == 1 and bad[0]["progress"]["steps_done"] == 50
    assert "fake failure at move 55" in bad[0]["error"] and bad[0]["startup_s"] == 0.5


def make_repo(root: Path, procthor="gz"):
    tasks = [
        {"task_id": "ithor_FloorPlan407_easy_t07_s00", "dataset": "ithor", "scene": "FloorPlan407",
         "target": {"position": {"x": -1.75, "y": 0.9, "z": -0.5}, "rotation_y": 90.0, "horizon": 30.0},
         "start": {"position": {"x": -1.0, "y": 0.9, "z": 0.25}, "rotation_y": 225.0, "horizon": 0.0}},
        {"task_id": "ithor_FloorPlan10_easy_t07_s00", "dataset": "ithor", "scene": "FloorPlan10",
         "target": {}, "start": {}},
        {"task_id": "procthor_00002_easy_t01_s00", "dataset": "procthor", "scene": "procthor_00002",
         "target": {"position": {"x": 1.0, "y": 0.9, "z": 1.0}, "rotation_y": 0.0, "horizon": 0.0},
         "start": {"position": {"x": 2.0, "y": 0.9, "z": 1.0}, "rotation_y": 180.0, "horizon": 0.0}},
    ]
    wjson(root / "data" / "tasks" / "eval.json", tasks)
    pt = root / "data" / "procthor-10k" / "train.jsonl.gz"
    pt.parent.mkdir(parents=True, exist_ok=True)
    if procthor == "gz":
        with gzip.open(pt, "wt") as f:
            for k in range(4):
                f.write(json.dumps({"id": f"house{k}"}) + "\n")
    elif procthor in ("lfs", "lfs_and_plain"):
        pt.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 67000000\n")
    if procthor in ("plain", "lfs_and_plain"):
        with open(pt.with_suffix(""), "w") as f:  # data/procthor-10k/train.jsonl
            for k in range(4):
                f.write(json.dumps({"id": f"plain{k}"}) + "\n")
    return root


class FakeEnv:
    instances = []

    def __init__(self, scene, width, height, gpu_device):
        self.loaded, self.closed, self.k = [], False, 0
        self.rng = np.random.default_rng(2)
        FakeEnv.instances.append(self)

    def load_scene(self, scene):
        self.loaded.append(scene)

    def _frame(self):
        return self.rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)

    def reset(self, position, rotation_y, horizon):
        return self._frame()

    def step(self, action):
        return self._frame(), action != "MoveBack"

    def close(self):
        self.closed = True


def test_procthor_file_and_house(base):
    repo = make_repo(base / "repo")
    p, why = thor.procthor_file(str(repo))
    assert why is None and thor.load_procthor_house(p, "procthor_00002") == {"id": "house2"}
    with pytest.raises(ValueError):
        thor.load_procthor_house(p, "procthor_00009")
    lfs = make_repo(base / "lfs", procthor="lfs")
    assert thor.procthor_file(str(lfs))[1].startswith("git-lfs pointer")
    none = make_repo(base / "none", procthor=None)
    (none / "data" / "procthor-10k").rmdir()
    assert thor.procthor_file(str(none))[1].startswith("missing")
    # plain train.jsonl, as TVRBench's _load_procthor_house accepts when there is no .gz
    plain = make_repo(base / "plain", procthor="plain")
    p, why = thor.procthor_file(str(plain))
    assert why is None and p.endswith("train.jsonl")
    assert thor.load_procthor_house(p, "procthor_00003") == {"id": "plain3"}
    # TVRBench prefers the .gz whenever it exists, so a pointer .gz next to a plain file is still not data
    assert thor.procthor_file(str(make_repo(base / "both", procthor="lfs_and_plain")))[1].startswith("git-lfs")
    (plain / "data" / "procthor-10k" / "train.jsonl").write_text("version 1\n")
    assert thor.procthor_file(str(plain))[1].startswith("not jsonl")


def test_task_phase(base):
    repo = make_repo(base / "repo")
    FakeEnv.instances.clear()
    rows = thor.task_phase(str(base), str(repo), 0, env_cls=FakeEnv)
    env = FakeEnv.instances[-1]
    assert env.closed and env.loaded == ["FloorPlan407", {"id": "house2"}]
    assert [r["dataset"] for r in rows] == ["ithor", "procthor"]
    assert all(r["ok"] and r["distinct_start_target"] and r["moves_ok"] == 9 for r in rows)
    assert (base / "ithor_FloorPlan407_easy_t07_s00_target.png").exists()
    lfs = make_repo(base / "lfs", procthor="lfs")
    rows = thor.task_phase(str(base), str(lfs), 0, env_cls=FakeEnv)
    assert rows[0]["ok"] and rows[1]["skipped"] == "no_procthor_data" and not rows[1]["ok"]


def test_run_single_process(base, monkeypatch):
    home = base / "home"
    log = home / thor.PLAYER_LOG
    log.parent.mkdir(parents=True)
    log.write_text("Vulkan renderer=[NVIDIA H800]\n")
    monkeypatch.setenv("HOME", str(home))
    repo = make_repo(base / "repo", procthor="lfs")
    a = thor.parse_args(["--out", str(base / "t1"), "--repo", str(repo), "--steps", "12"])
    res = thor.run(a, factory=lambda g: (FakeController(), 2.0), env_cls=FakeEnv)
    assert res["pass"] and res["tasks_ok"] == 1 and res["tasks_skipped"] == ["procthor_00002_easy_t01_s00"]
    assert json.loads((base / "t1" / "result.json").read_text())["pass"] is True
    assert (base / "t1" / "Player_raw.log").read_text().startswith("Vulkan renderer")
    assert (base / "t1" / "Player_tasks.log").exists()
    # an all-bad Controller fails the run but phase B still records the tasks
    a = thor.parse_args(["--out", str(base / "t2"), "--repo", str(repo), "--steps", "8"])
    res = thor.run(a, factory=lambda g: (FakeController(const=True), 2.0), env_cls=FakeEnv)
    assert not res["pass"] and len(res["tasks"]) == 2


# --- egl_probe without EGL ------------------------------------------------------------------------------------

def test_egl_probe_without_library(base):
    def no_lib(name):
        raise OSError(f"{name}: cannot open shared object file")

    res = eglp.probe(str(base), cdll=no_lib)
    assert res["stage"] == "load_libEGL" and res["pass"] is False and "cannot open" in res["error"]
    assert eglp.readback_ok([51, 102, 153, 255]) and eglp.readback_ok([53, 100, 151, 253])
    assert not eglp.readback_ok([48, 102, 153, 255]) and not eglp.readback_ok(None)
    assert eglp.EXPECT == verdict.EGL_RGBA


# --- stage_tvr_login.sh on fake sources -----------------------------------------------------------------------

def fake_sources(base: Path, driver=DRV):
    sysd = base / "sys"
    sysd.mkdir()
    for i, f in enumerate(LIBS):
        (sysd / f"{f}.so.{DRV}").write_bytes(f"fake {f} {i}\n".encode() * (i + 1))
        (sysd / f"{f}.so.550.90.07").write_bytes(b"")  # the image's 0-byte stubs are never copied
    pkgs = base / "pkgs"
    for pkg, lib, real in (("libegl-1.7.0-ha4b6fd6_5", "libEGL.so.1", "libEGL.so.1.1.0"),
                           ("libopengl-1.7.0-ha4b6fd6_5", "libOpenGL.so.0", "libOpenGL.so.0.0.0"),
                           ("libglvnd-1.7.0-ha4b6fd6_5", "libGLdispatch.so.0", "libGLdispatch.so.0.0.0")):
        d = pkgs / pkg / "lib"
        d.mkdir(parents=True)
        (d / real).write_bytes(f"glvnd {real}\n".encode())
        os.symlink(real, d / lib)
    drvf = base / "driver_version.txt"
    drvf.write_text(DRIVER_TXT.replace(DRV, driver))
    return login_env(VPD_TVR_OUT=str(base / "out"), VPD_TVR_SYS=str(sysd), VPD_TVR_PKGS=str(pkgs),
                     VPD_TVR_DRIVER_FILE=str(drvf), VPD_TVR_CHECK_PY=sys.executable)


def run_stage(env, *args):
    return subprocess.run(["bash", str(STAGE), *args], env=env, capture_output=True, text=True, timeout=120)


def test_stage_local(base):
    env = fake_sources(base)
    r = run_stage(env, "--local")
    assert r.returncode == 0, r.stdout + r.stderr
    nvgl = base / "out" / "tvr" / f"nvgl_{DRV}"
    for f in LIBS:
        assert (nvgl / "lib" / f"{f}.so.{DRV}").read_bytes() == (base / "sys" / f"{f}.so.{DRV}").read_bytes()
    assert not list((nvgl / "lib").glob("*550.90.07"))
    assert os.readlink(nvgl / "lib" / "libGLX_nvidia.so.0") == f"libGLX_nvidia.so.{DRV}"
    assert os.readlink(nvgl / "lib" / "libnvidia-allocator.so.1") == f"libnvidia-allocator.so.{DRV}"
    sums = (nvgl / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 12
    for line in sums:
        h, name = line.split(None, 1)
        assert h == hashlib.sha256((base / "sys" / name[2:]).read_bytes()).hexdigest()
    assert os.readlink(nvgl / "glvnd" / "libEGL.so.1") == "libEGL.so.1.1.0"
    assert (nvgl / "glvnd" / "libGLdispatch.so.0").read_bytes() == b"glvnd libGLdispatch.so.0.0.0\n"
    icd = json.loads((nvgl / "vulkan" / "nvidia_icd.json").read_text())
    assert icd["ICD"]["library_path"] == str(nvgl / "lib" / "libGLX_nvidia.so.0")
    assert icd["ICD"]["api_version"] == "1.4.329"
    assert json.loads((nvgl / "egl" / "10_nvidia.json").read_text())["ICD"]["library_path"] == \
        str(nvgl / "lib" / "libEGL_nvidia.so.0")
    rep = (base / "out" / "tvr" / "staging_report.txt").read_text()
    assert "## NOT_FOUND_COUNT=0" in rep and "Unity player not staged yet" in rep
    assert f"ABSENT  {base / 'out' / 'envs' / 'tvr' / 'bin' / 'python'}" in rep
    assert not (base / "out" / "envs").exists()  # local mode never creates the env
    # idempotent: nothing copied the second time
    r2 = run_stage(env)
    assert r2.returncode == 0 and "  + " not in r2.stdout, r2.stdout
    # a changed source lib is re-copied
    (base / "sys" / f"libnvidia-tls.so.{DRV}").write_bytes(b"new tls\n")
    r3 = run_stage(env)
    assert r3.returncode == 0 and f"  + libnvidia-tls.so.{DRV}" in r3.stdout
    assert (nvgl / "lib" / f"libnvidia-tls.so.{DRV}").read_bytes() == b"new tls\n"


def test_stage_counts_unresolved_libs(base):
    env = fake_sources(base)
    bindir = base / "bin"
    bindir.mkdir()
    ldd = bindir / "ldd"
    ldd.write_text('#!/usr/bin/env bash\necho "\tlibxcb-randr.so.0 => not found"\necho "\tlibc.so.6 => /lib/libc.so.6"\n')
    ldd.chmod(0o755)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    r = run_stage(env)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = (base / "out" / "tvr" / "staging_report.txt").read_text()
    assert "## NOT_FOUND_COUNT=15" in rep  # 12 libs + 3 glvnd, one unresolved line each
    assert "libxcb-randr.so.0 => not found" in rep and "ACTION NEEDED: 15" in r.stdout


def test_stage_refusals(base):
    env = fake_sources(base)
    for mode in ("--net", "--all"):
        r = run_stage(env, mode)
        assert r.returncode == 2 and "REFUSED" in r.stderr
    assert run_stage(env, "--bogus").returncode == 2
    assert not (base / "out").exists()
    (base / "b").mkdir()
    r = run_stage(fake_sources(base / "b", driver="550.90.07"))
    assert r.returncode == 1 and f"is not {DRV}" in r.stderr
    (base / "sys" / f"libnvidia-rtcore.so.{DRV}").unlink()
    r = run_stage(env)
    assert r.returncode == 1 and "missing or empty" in r.stderr


# --- tvr_smoke.sh on a fake staged tree ------------------------------------------------------------------------

FAKE_PY = """#!/usr/bin/env bash
# fake tvr-env interpreter: interpreter flags (-I -u), then the script and its args
flags=""; while [[ $1 == -* ]]; do flags="$flags $1"; shift; done
script=$(basename "$1"); shift
out=""; nproc=1
while [[ $# -gt 0 ]]; do case $1 in --out) out=$2; shift 2;; --nproc) nproc=$2; shift 2;; *) shift;; esac; done
mkdir -p "$out"
echo "$flags" > "$out/flags.txt"
case $script in
  thor_smoke.py)
    if [[ $nproc == 4 ]]; then cp "@C@/thor4.json" "$out/result.json"
    else
      cp "@C@/thor1.json" "$out/result.json"; cp "@C@/Player.log" "$out/Player_raw.log"
      echo "${CUDA_VISIBLE_DEVICES-unset}" > "$out/cvd_seen.txt"
      # leave a Unity player behind, started like ai2thor does: through $HOME/.ai2thor/releases (a symlink)
      "$HOME/.ai2thor/releases/@B@/@B@" < /dev/null > /dev/null 2>&1 &
      echo $! > "$out/fake_unity.pid"
    fi ;;
  egl_probe.py) cp "@C@/egl.json" "$out/result.json" ;;
  *) echo "unexpected $script" >&2; exit 9 ;;
esac
echo "fake $script -> $out"
"""
# a Unity player that ignores TERM, so reap_unity has to fall back to KILL
FAKE_PLAYER = "#!/bin/sh\ntrap '' TERM\nwhile :; do sleep 1; done\n"


def alive(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return st.rsplit(")", 1)[1].split()[0] != "Z"


def fake_tree(base: Path, driver=DRV, staged=True, **cvd):
    out = base / "vout"
    t = out / "tvr"
    nvgl = t / f"nvgl_{DRV}"
    canned = base / "canned"
    make_pass(canned)
    for name, rel in (("thor1.json", "thor1/result.json"), ("thor4.json", "thor4/result.json"),
                      ("egl.json", "egl/result.json"), ("Player.log", "thor1/Player_raw.log")):
        shutil.copyfile(canned / rel, canned / name)
    bindir = base / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text(f'#!/usr/bin/env bash\nif [[ ${{1:-}} == -L ]]; then printf "%s" "{SMI_L}"; '
                   f'else echo "NVIDIA H800, {DRV}, 81559 MiB"; fi\n')
    py = base / "fakepy"
    py.write_text(FAKE_PY.replace("@C@", str(canned)).replace("@B@", BUILD_NAME))
    for f in (smi, py):
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    if staged:
        for rel in ("lib/libGLX_nvidia.so.0", "lib/libEGL_nvidia.so.0", "glvnd/libEGL.so.1", "glvnd/libOpenGL.so.0",
                    "glvnd/libGLdispatch.so.0", "vulkan/nvidia_icd.json", "egl/10_nvidia.json"):
            (nvgl / rel).parent.mkdir(parents=True, exist_ok=True)
            (nvgl / rel).write_text("fake\n")
        vk = out / "envs" / "tvr" / "bin" / "vulkaninfo"
        vk.parent.mkdir(parents=True)
        (base / "summary.txt").write_text(SUMMARY)
        vk.write_text(f'#!/usr/bin/env bash\ncat "{base / "summary.txt"}"\n')
        vk.chmod(0o755)
        b = t / "ai2thor_home" / ".ai2thor" / "releases" / BUILD_NAME / BUILD_NAME
        b.parent.mkdir(parents=True)
        b.write_text(FAKE_PLAYER)
        b.chmod(0o755)
        wjson(t / "repos" / "TVRBench" / "data" / "tasks" / "eval.json", [])
    drvf = base / "driver_version.txt"
    drvf.write_text(DRIVER_TXT.replace(DRV, driver))
    env = login_env(VPD_TVR_OUT=str(out), VPD_TVR_PY=str(py), VPD_TVR_VERDICT_PY=sys.executable,
                    VPD_TVR_DRIVER_FILE=str(drvf), PATH=f"{bindir}:{os.environ['PATH']}", VPD_TVR_REAP_WAIT="10",
                    **cvd)
    return env


def run_smoke(env, outdir):
    """Run tvr_smoke.sh; afterwards kill any fake player it left (whatever the test outcome) and record that in
    r.left_alive, so a test can tell whether the script's own reaper did the job."""
    r = None
    try:
        r = subprocess.run(["bash", str(SMOKE), str(outdir)], env=env, capture_output=True, text=True,
                           timeout=120)
        r.left_alive = False
        return r
    finally:
        pidf = Path(outdir) / "thor1" / "fake_unity.pid"
        if pidf.exists():
            pid = int(pidf.read_text())
            if alive(pid):
                os.kill(pid, 9)
                if r is not None:
                    r.left_alive = True


def steps_of(outdir):
    return [json.loads(ln) for ln in (outdir / "steps.jsonl").read_text().splitlines()]


def test_smoke_full_flow_with_fakes(base):
    env = fake_tree(base)
    od = base / "smoke" / "out"
    r = run_smoke(env, od)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    v = json.loads((od / "verdict.json").read_text())
    assert (v["TVR_GATE"], v["EGL_GATE"]) == ("PASS", "PASS"), v["reasons"]
    st = steps_of(od)
    assert [s["name"] for s in st] == ["0-node-facts", "1-ldd", "2-vulkaninfo", "3-thor1", "4-thor4", "5-egl",
                                       "6-lavapipe", "7-verdict"]
    assert all(s["rc"] == 0 for s in st), st
    facts = json.loads((od / "facts.json").read_text())
    assert set(facts) >= {"nvidia_modeset", "dev_dri", "tmp_free_bytes", "shm_free_bytes", "cpu_max"}
    staging = json.loads((od / "staging.json").read_text())
    assert all(v2 is True for k, v2 in staging.items() if k != "procthor") and staging["procthor"] == "missing"
    assert json.loads((od / "mapping.json").read_text())["mode"] == "ai2thor_native"
    assert (od / "home" / ".ai2thor" / "releases").is_symlink()
    assert "skipped: no lavapipe ICD" in r.stdout
    assert "uuid" in (od / "node.txt").read_text().lower()
    # python runs as -I -u; CVD was unset, so nothing was remapped
    assert (od / "thor1" / "flags.txt").read_text().split() == ["-I", "-u"]
    assert (od / "thor1" / "cvd_seen.txt").read_text().strip() == "unset"
    assert facts["cvd_remap"] is None and facts["n_smi_gpus"] == 1 and facts["cuda_visible_devices"] is None
    # the player left behind (started through the HOME symlink, ignoring TERM) was reaped with KILL
    pid = int((od / "thor1" / "fake_unity.pid").read_text())
    assert r.left_alive is False and not alive(pid)
    reap = (od / "logs" / "reap.log").read_text()
    assert f"thor1: TERM lingering Unity player(s): {pid}" in reap and f"thor1: KILL" in reap, reap
    # a second run into the same OUTDIR is refused
    r2 = run_smoke(env, od)
    assert r2.returncode == 2 and "refusing" in r2.stderr


def test_smoke_cvd_remap(base):
    env = fake_tree(base, CUDA_VISIBLE_DEVICES="5")
    od = base / "smoke" / "out"
    r = run_smoke(env, od)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    facts = json.loads((od / "facts.json").read_text())
    assert facts["cuda_visible_devices"] == "5"
    assert facts["cvd_remap"] == {"from": "5", "to": "0", "n_smi_gpus": 1}
    assert (od / "thor1" / "cvd_seen.txt").read_text().strip() == "0"  # what ai2thor would see
    v = json.loads((od / "verdict.json").read_text())
    assert v["TVR_GATE"] == "PASS" and any("CUDA_VISIBLE_DEVICES remapped" in c for c in v["caveats"])
    assert v["facts"]["cvd_remap"]["from"] == "5"
    # CVD=0 is left alone
    env0 = fake_tree(base / "b0", CUDA_VISIBLE_DEVICES="0")
    od0 = base / "b0" / "smoke" / "out"
    assert run_smoke(env0, od0).returncode == 0
    assert json.loads((od0 / "facts.json").read_text())["cvd_remap"] is None
    assert (od0 / "thor1" / "cvd_seen.txt").read_text().strip() == "0"


def test_smoke_time_budget_fits():
    """Every inner cap together stays under timeout 1800, and thor1 holds two full Controller start-ups."""
    sm = SMOKE.read_text()
    cap = int(re.search(r"(?m)^THOR1_CAP=(\d+)$", sm).group(1))
    dl = int(re.search(r"(?m)^THOR1_DEADLINE=(\d+)$", sm).group(1))
    assert "timeout $THOR1_CAP" in sm and "--deadline-s $THOR1_DEADLINE" in sm
    code = "\n".join(ln for ln in sm.splitlines() if not ln.lstrip().startswith("#"))
    caps = [int(x) for x in re.findall(r"timeout (\d+) ", code)]
    assert sorted(caps) == [60, 60, 90, 240, 360], caps  # vulkaninfo x2, egl, lavapipe, thor4
    lav_end = int(re.search(r"left=\$\(\((\d+) - ", sm).group(1))
    assert int(re.search(r"\(\( left < (\d+) \)\)", sm).group(1)) == 240
    worst_no_lvp = 10 + 60 + 60 + cap + 360 + 90 + 5
    assert worst_no_lvp <= 1700 and lav_end + 5 <= 1700
    assert cap >= 2 * verdict.STARTUP_MAX_S + 300 and dl < cap
    assert thor.PHASE_B_MIN_S > verdict.STARTUP_MAX_S
    # a start-up just under the T3 limit still leaves room for phase B
    assert dl - (verdict.STARTUP_MAX_S + 120) >= thor.PHASE_B_MIN_S
    # the reaper matches the HOME-independent suffix, not the AI2THOR_HOME path
    assert 'UNITY_PAT="/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME"' in sm
    assert 'pkill -f -- "$BUILD/$BUILD_NAME"' not in sm


def test_smoke_driver_mismatch_exits_3(base):
    env = fake_tree(base, driver="550.90.07")
    od = base / "smoke" / "out"
    r = run_smoke(env, od)
    assert r.returncode == 3, r.stdout[-2000:] + r.stderr[-2000:]
    v = json.loads((od / "verdict.json").read_text())
    assert (v["TVR_GATE"], v["EGL_GATE"]) == ("FAIL", "FAIL")
    assert any(x.startswith("T1_driver") for x in v["reasons"])
    assert [s["name"] for s in steps_of(od)] == ["0-node-facts", "7-verdict"]
    assert not (od / "thor1").exists()


def test_smoke_missing_staging_exits_4(base):
    env = fake_tree(base, staged=False)
    od = base / "smoke" / "out"
    r = run_smoke(env, od)
    assert r.returncode == 4, r.stdout[-2000:] + r.stderr[-2000:]
    v = json.loads((od / "verdict.json").read_text())
    assert v["TVR_GATE"] == "FAIL" and any(x.startswith("T0_staging") for x in v["reasons"])
    assert "staging incomplete" in v["next"]
