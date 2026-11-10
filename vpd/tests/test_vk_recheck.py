"""Tests of vpd-tvr-vkre: vpd/tvr/{vk_recheck.sh, vkre.py} (EXPLORATORY NVIDIA Vulkan recheck, DECISIONS G2).

Login node, 1 core, hermetic: no GPU call, no Controller, no rendering, nothing written outside pytest's tmp_path.
The job script runs on a fake staged tree, a fake system graphics stack, a fake probe interpreter (a bash script
that records its argv / environment and writes canned results), a fake nvidia-smi and a fake ldconfig (first on
PATH).  Its bookkeeping (vkre.py facts / icd-json / egl-json / thor-check / summarize and debug_probe.py finalize /
post-thor / get) runs for real under this interpreter.  Every probe argv the job builds is parsed with the real
debug_probe.py parser.
    cd $WT && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 nice -n 19 \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=$VPD_OUT/scratch/<label>/pt vpd/tests/test_vk_recheck.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from vpd.tvr import vkre
from vpd.tvr.debug import common

TVR = Path(__file__).resolve().parents[1] / "tvr"
JOB = TVR / "vk_recheck.sh"
VKRE = TVR / "vkre.py"
PROBE = TVR / "debug_probe.py"
DRV = "595.71.05"
OTHER_DRV = "570.172.08"
UUID_ = "GPU-42b6cb64-7b78-6b3a-1fc7-a674212f0333"
STAGED_GLX = b"glx-595.71.05-bytes\n" * 50
STAGED_EGL = b"egl-595.71.05-bytes\n" * 50
STAGED_GLCORE = b"glcore-595.71.05-bytes\n" * 50


def login_env(**extra):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", VPD_THREADS="1", PYTHONDONTWRITEBYTECODE="1", LP_NUM_THREADS="1")
    for k in list(env):
        if k.startswith("VPD_VKRE_") or k in (
                "RUN", "CUDA_VISIBLE_DEVICES", "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "LD_LIBRARY_PATH", "LD_DEBUG",
                "VK_LOADER_DEBUG", "NVIDIA_DRIVER_CAPABILITIES", "VPD_JOB_DEADLINE_S", "__EGL_VENDOR_LIBRARY_FILENAMES"):
            env.pop(k, None)
    env.update(extra)
    return env


def wjson(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def rjson(p: Path):
    return json.loads(Path(p).read_text())


def code_of(sh: str) -> str:
    return "\n".join(ln for ln in sh.splitlines() if not ln.lstrip().startswith("#"))


# --- static ---------------------------------------------------------------------------------------------------

def test_bash_syntax():
    r = subprocess.run(["bash", "-n", str(JOB)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_script_follows_the_payload_conventions():
    sh = JOB.read_text()
    code = [ln for ln in code_of(sh).splitlines() if ln.strip()]
    assert code[0] == 'source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../jobs" && pwd)/common_env.sh"'
    assert code[1].startswith('cd "$VPD_WORKTREE"')
    assert not [ln for ln in code if "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd" in ln]   # only $VPD_WORKTREE
    assert "VPD_COMMON_ENV" not in sh and "S=$VPD_WORKTREE/vpd/tvr" in sh
    # the graphics capability comes from `vpd.sco prepare --env`, never from the payload itself
    assert not re.search(r"^\s*(export\s+)?NVIDIA_DRIVER_CAPABILITIES=|\bexport\b[^\n]*\bNVIDIA_DRIVER_CAPABILITIES=",
                         code_of(sh), re.M)
    assert "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES-unset}" in code_of(sh)   # only recorded
    for x in ("--gate tvr", "--est-gph 0.25", "--env NVIDIA_DRIVER_CAPABILITIES:all",
              "-- env VPD_JOB_DEADLINE_S=660 timeout -k 30 720", "/vpd/tvr/vk_recheck.sh $RUN/out",
              "vkre.py expect --out $RUN/out"):
        assert x in sh, x
    assert "--as-root" not in sh   # no Qwen3 checkpoint: the recheck runs as the normal user
    assert "cpu_count" not in VKRE.read_text()


def test_script_variant_order_matches_vkre():
    code = code_of(JOB.read_text())
    names = re.findall(r"^(?:variant|maybe \"[^\n]*?\") (v\d\d_[a-zA-Z0-9_]+) \d+ ", code, re.M)
    assert names == vkre.VARIANTS
    assert set(vkre.KIND) == set(vkre.VARIANTS) and vkre.VERDICT_VARIANT == "v13_thor_sysnew"


def test_time_budget_fits_the_soft_deadline():
    sh = JOB.read_text()
    code = code_of(sh)
    caps = [int(x) for x in re.findall(r"^(?:variant|maybe \"[^\n]*?\") v\d\d_\w+ (\d+) ", code, re.M)]
    assert caps == [30, 45, 45, 45, 45, 45, 240]
    assert "DEADLINE_S=${VPD_JOB_DEADLINE_S:-660}" in sh and "RESERVE_S=${VPD_VKRE_RESERVE_S:-60}" in sh
    assert sum(caps) + len(caps) * 6 <= 660 - 60   # every cap hit + ~6 s of reaping / bookkeeping each
    assert 660 + 30 <= 720                          # the outer `timeout -k 30 720` leaves the summary its time
    assert 'UNITY_PAT="/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME"' in sh and f"DRV={DRV}" in sh
    assert "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917" in sh
    assert 'export VK_LOADER_DEBUG=all' in sh and '--child-env "VK_LOADER_DEBUG=all"' in sh


# --- vkre units -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.71.05  Release Build  (dvs-builder@U22-I3-G08-03-1)"
     "  Fri Apr 24 06:42:30 UTC 2026\nGCC version:  gcc version 11.4.1 20231218 (Red Hat 11.4.1-3) (GCC)\n", DRV),
    ("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.90.07  Fri May 31 09:35:42 UTC 2024\n", "550.90.07"),
    ("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  570.172.08  Release Build\n", OTHER_DRV),
    ("NVRM version: an odd build 535.104.05 here\n", "535.104.05"),
    ("GCC version:  gcc version 11.4.1 20231218\n", None), ("", None), (None, None)])
def test_parse_kernel_driver(text, want):
    assert vkre.parse_kernel_driver(text) == want


def test_icd_and_egl_json(tmp_path):
    src = tmp_path / "sys" / "nvidia_icd.json"
    wjson(src, {"file_format_version": "1.0.1", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.4.329"}})
    p = tmp_path / "v11" / "sys_icd.json"
    doc = vkre.write_icd_json(str(p), "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0", str(src))
    assert rjson(p) == doc == {"file_format_version": "1.0.1",
                               "ICD": {"library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
                                       "api_version": "1.4.329"}}
    wjson(src, {"file_format_version": "1.0.0", "ICD": {"library_path": "x.so", "api_version": "1.3.277"}})
    assert vkre.write_icd_json(str(p), "/a/libGLX_nvidia.so.0", str(src))["ICD"]["api_version"] == "1.3.277"
    assert rjson(p)["file_format_version"] == "1.0.0"
    for bad_src in (str(tmp_path / "missing.json"), None):
        d = vkre.write_icd_json(str(p), "/a/libGLX_nvidia.so.0", bad_src)
        assert (d["file_format_version"], d["ICD"]["api_version"]) == ("1.0.1", "1.4.329")   # fallback
    (tmp_path / "empty.json").write_text("")
    assert vkre.write_icd_json(str(p), "/a/x.so", str(tmp_path / "empty.json"))["ICD"]["api_version"] == "1.4.329"
    with pytest.raises(ValueError):
        vkre.write_icd_json(str(p), "libGLX_nvidia.so.0", str(src))
    e = tmp_path / "v11" / "sys_egl.json"
    assert vkre.write_egl_json(str(e), "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0") == rjson(e) == {
        "file_format_version": "1.0.0", "ICD": {"library_path": "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"}}
    with pytest.raises(ValueError):
        vkre.write_egl_json(str(e), "libEGL_nvidia.so.0")
    # the CLI the job calls (a relative library path is refused with a non-zero exit)
    r = subprocess.run([sys.executable, "-I", str(VKRE), "icd-json", "--out", str(p), "--lib", "/b/libGLX_nvidia.so.0",
                        "--src", str(src)], capture_output=True, text=True, timeout=60, env=login_env())
    assert r.returncode == 0 and rjson(p)["ICD"]["library_path"] == "/b/libGLX_nvidia.so.0", r.stderr
    r = subprocess.run([sys.executable, "-I", str(VKRE), "egl-json", "--out", str(e), "--lib", "rel.so"],
                       capture_output=True, text=True, timeout=60, env=login_env())
    assert r.returncode != 0 and "absolute" in r.stderr


def test_read_sums_and_system_vs_staged(tmp_path):
    sl = tmp_path / "syslib"
    sl.mkdir()
    (sl / f"libGLX_nvidia.so.{DRV}").write_bytes(STAGED_GLX)
    (sl / f"libEGL_nvidia.so.{DRV}").write_bytes(b"different")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{hashlib.sha256(STAGED_GLX).hexdigest()}  ./libGLX_nvidia.so.{DRV}\n"
                    f"{hashlib.sha256(STAGED_EGL).hexdigest()} *libEGL_nvidia.so.{DRV}\n"
                    f"{hashlib.sha256(STAGED_GLCORE).hexdigest()}  ./libnvidia-glcore.so.{DRV}\n"
                    "not a checksum line\n\n")
    got = vkre.read_sums(str(sums))
    assert sorted(got) == [f"libEGL_nvidia.so.{DRV}", f"libGLX_nvidia.so.{DRV}", f"libnvidia-glcore.so.{DRV}"]
    cmp = vkre.system_vs_staged(str(sl), got)
    assert {k: v["equal"] for k, v in cmp.items()} == {f"libEGL_nvidia.so.{DRV}": False,
                                                        f"libGLX_nvidia.so.{DRV}": True,
                                                        f"libnvidia-glcore.so.{DRV}": None}
    assert vkre.all_equal(cmp) is False
    assert vkre.all_equal({k: v for k, v in cmp.items() if v["equal"]}) is True
    assert vkre.all_equal({k: v for k, v in cmp.items() if v["equal"] is None}) is None
    assert vkre.read_sums(str(tmp_path / "none")) == {}


SMI_Q = f"""
==============NVSMI LOG==============

Timestamp                                 : Tue Nov 10 07:00:00 2026
Driver Version                            : @D@
CUDA Version                              : 13.2

Attached GPUs                             : 1
GPU 00000000:C1:00.0
    Product Name                          : NVIDIA H800
    Product Architecture                  : Hopper
    Display Mode                          : Disabled
    Serial Number                         : 1650123456789
    GPU UUID                              : {UUID_}
    VBIOS Version                         : 96.00.89.00.06
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    GSP Firmware Version                  : @D@
    PCI
        Bus                               : 0xC1
        Bus Id                            : 00000000:C1:00.0
    Compute Mode                          : Default
"""


def fake_system(root: Path, honoured: bool, drv: str = DRV, kernel: str | None = None):
    """A fake /usr/lib/x86_64-linux-gnu, /etc/vulkan, egl vendor dir, /dev and /proc/driver/nvidia/version."""
    kernel = kernel or drv   # the compute-only image keeps 0-byte 550 stubs, but the node's kernel driver is drv
    sl = root / "syslib"
    sl.mkdir(parents=True)
    vk = root / "etc_vulkan"
    (vk / "icd.d").mkdir(parents=True)
    (vk / "implicit_layer.d").mkdir(parents=True)
    if honoured:
        (sl / f"libGLX_nvidia.so.{drv}").write_bytes(STAGED_GLX if drv == DRV else b"other glx")
        (sl / f"libEGL_nvidia.so.{drv}").write_bytes(b"system egl, not the staged bytes")
        (sl / f"libnvidia-glvkspirv.so.{drv}").write_bytes(b"x" * 10)
        wjson(vk / "icd.d" / "nvidia_icd.json", {"file_format_version": "1.0.1",
                                                 "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.4.329"}})
    else:   # the compute-only image: 0-byte 550 stubs, empty ICD / layer JSON
        drv = "550.90.07"
        (sl / f"libGLX_nvidia.so.{drv}").write_bytes(b"")
        (sl / f"libEGL_nvidia.so.{drv}").write_bytes(b"")
        (vk / "icd.d" / "nvidia_icd.json").write_text("")
        (vk / "implicit_layer.d" / "nvidia_layers.json").write_text("")
    (sl / "libGLX_nvidia.so.0").symlink_to(f"libGLX_nvidia.so.{drv}")
    (sl / "libEGL_nvidia.so.0").symlink_to(f"libEGL_nvidia.so.{drv}")
    (sl / "libvulkan.so.1").write_bytes(b"image loader 1.2.131")
    (root / "egl_vendor").mkdir()
    dev = root / "dev"
    dev.mkdir()
    (dev / "nvidia-modeset").write_text("")
    (dev / "nvidia4").write_text("")
    drvf = root / "driver_version.txt"
    drvf.write_text(f"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  {kernel}  Release Build  "
                    "(dvs-builder@U22-I3-G08-03-1)  Fri Apr 24 06:42:30 UTC 2026\n")
    return sl, vk, dev, drvf


def staged_tree(out: Path) -> Path:
    nvgl = out / "tvr" / f"nvgl_{DRV}"
    lib = nvgl / "lib"
    lib.mkdir(parents=True)
    (lib / f"libGLX_nvidia.so.{DRV}").write_bytes(STAGED_GLX)
    (lib / f"libEGL_nvidia.so.{DRV}").write_bytes(STAGED_EGL)
    (lib / f"libnvidia-glcore.so.{DRV}").write_bytes(STAGED_GLCORE)
    (lib / "libGLX_nvidia.so.0").symlink_to(f"libGLX_nvidia.so.{DRV}")
    (lib / "libEGL_nvidia.so.0").symlink_to(f"libEGL_nvidia.so.{DRV}")
    for rel in ("glvnd/libEGL.so.1", "glvnd/libOpenGL.so.0", "glvnd/libGLdispatch.so.0"):
        (nvgl / rel).parent.mkdir(parents=True, exist_ok=True)
        (nvgl / rel).write_text("fake glvnd\n")
    wjson(nvgl / "vulkan" / "nvidia_icd.json", {"file_format_version": "1.0.1", "ICD": {
        "library_path": str(lib / "libGLX_nvidia.so.0"), "api_version": "1.4.329"}})
    wjson(nvgl / "egl" / "10_nvidia.json", {"file_format_version": "1.0.0", "ICD": {
        "library_path": str(lib / "libEGL_nvidia.so.0")}})
    (nvgl / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(b).hexdigest()}  ./{n}\n" for n, b in (
        (f"libGLX_nvidia.so.{DRV}", STAGED_GLX), (f"libEGL_nvidia.so.{DRV}", STAGED_EGL),
        (f"libnvidia-glcore.so.{DRV}", STAGED_GLCORE))))
    return nvgl


def test_collect_facts(tmp_path):
    sl, vk, dev, drvf = fake_system(tmp_path / "sys", honoured=True)
    nvgl = staged_tree(tmp_path / "vout")
    out = tmp_path / "v00"
    out.mkdir()
    (out / "nvidia_smi_q.txt").write_text(SMI_Q.replace("@D@", DRV))
    (out / "ldconfig_nvidia.txt").write_text("\tlibGLX_nvidia.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libGLX_nvidia.so.0\n")
    (out / "ldconfig_gfx.txt").write_text("\tlibvulkan.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libvulkan.so.1\n"
                                          "\tlibGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGL.so.1\n")
    res = vkre.collect_facts(str(out), sys_lib=str(sl), vk_dirs=[str(vk)], egl_vendor_dirs=[str(tmp_path / "sys" / "egl_vendor")],
                             dev_dir=str(dev), driver_file=str(drvf), staged_sums=str(nvgl / "SHA256SUMS"), nvgl=str(nvgl),
                             environ={"NVIDIA_DRIVER_CAPABILITIES": "all", "NVIDIA_VISIBLE_DEVICES": UUID_})
    m = res["metrics"]
    assert res["pass"] is True and m["kernel_driver"] == DRV and m["staged_driver_match"] is True
    assert m["driver_match"] is True and m["caps_honored"] is True and m["graphics_requested"] is True
    assert m["gpu"] == {"product": "NVIDIA H800", "uuid": UUID_, "pci_bus_id": "00000000:C1:00.0",
                        "vbios": "96.00.89.00.06", "gsp_firmware": DRV, "serial": "1650123456789",
                        "driver_version": DRV, "cuda_version": "13.2"}
    assert m["system_equals_staged"] == {f"libEGL_nvidia.so.{DRV}": False, f"libGLX_nvidia.so.{DRV}": True,
                                         f"libnvidia-glcore.so.{DRV}": None}
    assert m["system_equals_staged_all"] is False and m["staged_sums"]["entries"] == 3
    assert m["ldconfig_gfx"] == {"libvulkan.so.1": ["/usr/lib/x86_64-linux-gnu/libvulkan.so.1"],
                                 "libGL.so.1": ["/usr/lib/x86_64-linux-gnu/libGL.so.1"]}
    assert m["system_icd_json"]["path"] == str(vk / "icd.d" / "nvidia_icd.json")
    assert m["system_icd_json"]["api_version"] == "1.4.329" and m["hostname"]
    assert m["system_glvnd"] == {"libEGL.so.1": False, "libOpenGL.so.0": False}


def test_collect_facts_keys_caps_on_the_kernel_driver(tmp_path):
    """On a node with another driver the system stack still counts as mounted (the 11-09 facts.py would not)."""
    sl, vk, dev, drvf = fake_system(tmp_path / "sys", honoured=True, drv=OTHER_DRV)
    nvgl = staged_tree(tmp_path / "vout")
    out = tmp_path / "v00"
    out.mkdir()
    (out / "nvidia_smi_q.txt").write_text(SMI_Q.replace("@D@", OTHER_DRV))
    res = vkre.collect_facts(str(out), sys_lib=str(sl), vk_dirs=[str(vk)], egl_vendor_dirs=[], dev_dir=str(dev),
                             driver_file=str(drvf), staged_sums=str(nvgl / "SHA256SUMS"), environ={})
    m = res["metrics"]
    assert m["kernel_driver"] == OTHER_DRV and m["staged_driver_match"] is False and m["caps_honored"] is True
    assert m["lib_driver"] == OTHER_DRV and m["system_equals_staged_all"] is None   # no staged lib name exists
    assert m["system_libs"][f"libnvidia-glvkspirv.so.{OTHER_DRV}"]["size"] == 10
    # an unreadable driver file: library names fall back to the staged driver, and the run says so
    res = vkre.collect_facts(str(out), sys_lib=str(sl), vk_dirs=[str(vk)], egl_vendor_dirs=[], dev_dir=str(dev),
                             driver_file=str(tmp_path / "nope"), environ={})
    assert res["metrics"]["kernel_driver"] is None and res["metrics"]["caps_honored"] is False
    assert any("no kernel driver version" in n for n in res["notes"])


def thor_dir(d: Path, frames_pass=True, renderer="NVIDIA H800", maps=None, log=True, result=True):
    d.mkdir(parents=True, exist_ok=True)
    if result:
        wjson(d / "result.json", {"variant": d.name, "pass": frames_pass, "stage": "done", "notes": [], "metrics": {
            "frames": {"n_step_frames": 40, "frac_rgb_std_ge2": 1.0 if frames_pass else 0.0},
            "unity_maps_after_steps": maps or []}})
    if log:
        (d / "Player_after_exit.log").write_text(
            "[Vulkan init] SelectPhysicalDevice requestedDeviceIndex=0 xrDevice=(nil)\n"
            "INFO:             Found ICD manifest file /job/v13_thor_sysnew/sys_icd.json, version \"1.0.1\"\n"
            "ERROR: [Loader Message] Code 0 : something the loader disliked\n"
            "Vulkan API version 1.1.0 (1.4.329 supported by driver, 1.1.0 requested)\n"
            f"Vulkan renderer=[{renderer}] id=[2324]\n")


def test_thor_check(tmp_path):
    sysglx = "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.595.71.05"
    d = tmp_path / "ok"
    thor_dir(d, maps=[sysglx, "/x/libvulkan.so.1.4.357"])
    res = vkre.thor_check(str(d), nvgl="/stage/nvgl_595.71.05")
    v = res["metrics"]["vkre"]
    assert res["pass"] is True and res["pass_frames"] is True and rjson(d / "result.json")["pass"] is True
    assert v["renderers"] == ["NVIDIA H800"] and v["renderer_ok"] and v["player_log"] == "Player_after_exit.log"
    assert v["unity_glx"] == [sysglx] and v["unity_glx_staged"] is False
    assert v["vulkan_api"] == ["Vulkan API version 1.1.0 (1.4.329 supported by driver, 1.1.0 requested)"]
    assert v["loader"]["manifests"] == ["/job/v13_thor_sysnew/sys_icd.json"] and len(v["loader"]["errors"]) == 1
    # frames fine but rendered on lavapipe: FAIL
    d = tmp_path / "lvp"
    thor_dir(d, renderer="llvmpipe (LLVM 23.1.4, 256 bits)")
    res = vkre.thor_check(str(d))
    assert res["pass"] is False and res["pass_frames"] is True and res["metrics"]["vkre"]["renderer_ok"] is False
    assert any("names no 'NVIDIA H800'" in n for n in res["notes"])
    # the H800 renders black: FAIL
    d = tmp_path / "black"
    thor_dir(d, frames_pass=False)
    assert vkre.thor_check(str(d))["pass"] is False
    # libGLX_nvidia mapped from the staged overlay is flagged
    d = tmp_path / "staged"
    thor_dir(d, maps=["/stage/nvgl_595.71.05/lib/libGLX_nvidia.so.595.71.05"])
    assert vkre.thor_check(str(d), nvgl="/stage/nvgl_595.71.05")["metrics"]["vkre"]["unity_glx_staged"] is True
    # nothing at all (the probe was killed before it wrote anything)
    d = tmp_path / "none"
    thor_dir(d, log=False, result=False)
    res = vkre.thor_check(str(d))
    assert res["pass"] is False and res["metrics"]["vkre"]["player_log"] is None
    assert any("no result.json" in n for n in res["notes"])


def write_run(out: Path, st: dict, vk: dict | None = None, v13=None, pool_json=None):
    """Synthetic variant results: name -> True / False / None (skipped) / 'error' (profile failed)."""
    steps = []
    for n in vkre.VARIANTS:
        s = st.get(n)
        if s is None:
            r = {"name": n, "pass": None, "skipped": "caps_not_honored (x)", "stage": "skipped", "notes": []}
            steps.append(f"{n}:skipped")
        else:
            r = {"name": n, "pass": s is True, "stage": "done", "rc": 0 if s is True else 1, "metrics": {}, "notes": []}
            steps.append(n)
            if s == "error":
                (out / n).mkdir(parents=True, exist_ok=True)
                (out / n / "profile_error.txt").write_text("profile sysnew could not be set up\n")
        if vk and n in vk:
            r["metrics"]["vk"] = vk[n]
        if n == "v13_thor_sysnew" and v13:
            r.update(v13)
        wjson(out / n / "result.json", r)
    (out / "steps.jsonl").write_text("".join(json.dumps({"name": s, "rc": 0}) + "\n" for s in steps + ["summary"]))
    if pool_json:
        wjson(out.parent / "job" / "prepared.json", pool_json)


def test_summarize_verdicts_and_flags(tmp_path):
    ok_vk = {"vkCreateInstance": "VK_SUCCESS(0)", "vkWaitForFences": "VK_ERROR_DEVICE_LOST(-4)"}
    bad_icd = {"vkCreateInstance": "VK_ERROR_INCOMPATIBLE_DRIVER(-9)"}
    base = {"v00_facts": True, "v01_gl_tri_R": True, "v02_vk_nvS_env": False, "v02_vk_lvp": True,
            "v11_vk_sysnew": False, "v12_vk_sysnew_noglvnd": False}
    # PASS: v13 frames + renderer; v11 instance OK, v12 INCOMPATIBLE_DRIVER -> glvnd explains R
    out = tmp_path / "a" / "out"
    write_run(out, dict(base, v13_thor_sysnew=True), vk={"v11_vk_sysnew": ok_vk, "v12_vk_sysnew_noglvnd": bad_icd},
              v13={"pass_frames": True, "metrics": {"vkre": {"renderer_ok": True, "renderers": ["NVIDIA H800"]}}},
              pool_json={"pool": "zoe-cf-render"})
    s = vkre.summarize(str(out), environ={"VPD_VKRE_POOL": "ignored", "VPD_WORKTREE": str(tmp_path)})
    assert s["verdict"] == "PASS" and s["pool"] == "zoe-cf-render" and s["pool_source"] == "prepared.json"
    f = s["flags"]
    assert f["glvnd_explains_R"] is True and f["nv_vk_pass"] is False and f["nv_all_fail"] is False
    assert f["thor_sys_pass"] is True and f["lvp_ok"] is True and f["gl_ok"] is True
    assert f["nv_tested"] == ["v02_vk_nvS_env", "v11_vk_sysnew", "v12_vk_sysnew_noglvnd", "v13_thor_sysnew"]
    assert any("frozen TVR smoke" in r["next"] for r in s["readings"])
    assert any("glvnd libEGL.so.1" in r["reading"] for r in s["readings"])
    assert rjson(out / "summary.json")["verdict"] == "PASS" and s["overrides"] == {"VPD_VKRE_POOL": "ignored"}
    assert rjson(out / "expect.json")["files"] == vkre.EXPECT_FILES
    assert s["variants"]["v11_vk_sysnew"]["key"]["vkCreateInstance"] == "VK_SUCCESS(0)"
    assert vkre.expect(str(out))["ok"] is True
    # FAIL: every NVIDIA variant fails, also v13; a triangle that works but THOR black is the Unity row
    out = tmp_path / "b" / "out"
    write_run(out, dict(base, v13_thor_sysnew=False), vk={"v11_vk_sysnew": ok_vk, "v12_vk_sysnew_noglvnd": ok_vk})
    s = vkre.summarize(str(out), environ={"VPD_VKRE_POOL": "zoetrope"})
    assert s["verdict"] == "FAIL" and s["pool"] == "zoetrope" and s["pool_source"] == "env VPD_VKRE_POOL"
    assert s["flags"]["nv_all_fail"] is True and s["flags"]["glvnd_explains_R"] is False
    assert any("closed for round 3" in r["next"] for r in s["readings"])
    out = tmp_path / "c" / "out"
    write_run(out, dict(base, v11_vk_sysnew=True, v13_thor_sysnew=False))
    s = vkre.summarize(str(out), environ={})
    assert s["verdict"] == "FAIL" and s["flags"]["nv_vk_pass"] is True and s["flags"]["nv_all_fail"] is False
    assert any("Unity-specific" in r["reading"] for r in s["readings"]) and s["pool"] is None
    # NOT_TESTED: v13 skipped (caps not honoured), or its profile could not be set up (ERROR, not a GPU FAIL)
    out = tmp_path / "d" / "out"
    write_run(out, {"v00_facts": True, "v02_vk_nvS_env": False, "v02_vk_lvp": True})
    s = vkre.summarize(str(out), environ={})
    assert s["verdict"] == "NOT_TESTED" and "caps_not_honored" in s["verdict_reason"]
    assert s["flags"]["glvnd_explains_R"] is None and s["flags"]["thor_sys_pass"] is None
    assert s["variants_status"]["v11_vk_sysnew"] == "SKIP" and s["flags"]["nv_all_fail"] is True
    out = tmp_path / "e" / "out"
    write_run(out, dict(base, v13_thor_sysnew="error", v02_vk_lvp=False))
    s = vkre.summarize(str(out), environ={})
    assert s["variants_status"]["v13_thor_sysnew"] == "ERROR" and s["verdict"] == "NOT_TESTED"
    assert s["variants"]["v13_thor_sysnew"]["profile_error"].startswith("profile sysnew")
    assert any("lavapipe control failed" in r["reading"] for r in s["readings"])


def test_expect_reports_what_is_missing(tmp_path):
    out = tmp_path / "out"
    write_run(out, {n: True for n in vkre.VARIANTS})
    vkre.summarize(str(out), environ={})
    (out / "v12_vk_sysnew_noglvnd" / "result.json").unlink()
    lines = (out / "steps.jsonl").read_text().splitlines()
    (out / "steps.jsonl").write_text("\n".join(x for x in lines if "v12_" not in x) + "\n")
    e = vkre.expect(str(out))
    assert e["ok"] is False and e["verdict"] == "PASS"
    assert e["problems"] == ["missing v12_vk_sysnew_noglvnd/result.json", "no step v12_vk_sysnew_noglvnd"]
    r = subprocess.run([sys.executable, "-I", str(VKRE), "expect", "--out", str(out)], capture_output=True, text=True,
                       timeout=60, env=login_env())
    assert r.returncode == 1 and json.loads(r.stdout)["ok"] is False
    e = vkre.expect(str(tmp_path / "nothing"))
    assert not e["ok"] and "missing summary.json" in e["problems"] and "no step summary" in e["problems"]


# --- vk_recheck.sh on a fake tree ---------------------------------------------------------------------------------

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
env | grep -E '^(LD_LIBRARY_PATH|VK_[A-Z_]+|__EGL_VENDOR_LIBRARY_[A-Z]+|HOME|XDG_[A-Z_]+|LP_NUM_THREADS|CUDA_VISIBLE_DEVICES)=' | sort > "$out/fake_env.txt"
if [[ -f "@C@/$name.json" ]]; then cp "@C@/$name.json" "$out/result.json"
else echo '{"pass": true, "stage": "done", "metrics": {}, "notes": []}' > "$out/result.json"; fi
if [[ $sub == vk-tri && ${VK_LOADER_DEBUG:-} == all ]]; then
  echo "INFO:             Found ICD manifest file ${VK_ICD_FILENAMES:-none}, version \"1.0.1\"" >&2
fi
if [[ $sub == thor ]]; then
  "$HOME/.ai2thor/releases/@B@/@B@" < /dev/null > /dev/null 2>&1 &
  echo $! > "$out/fake_unity.pid"
  pl="$HOME/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR"
  mkdir -p "$pl"
  rend="NVIDIA H800"; [[ -f "@C@/$name.renderer" ]] && rend=$(cat "@C@/$name.renderer")
  { echo "INFO:             Found ICD manifest file ${VK_ICD_FILENAMES:-none}, version \"1.0.1\""
    echo "Vulkan renderer=[$rend] id=[2324]"; } > "$pl/Player.log"
fi
grep -q '"pass": true' "$out/result.json"
"""
FAKE_PLAYER = "#!/bin/sh\ntrap '' TERM\nwhile :; do sleep 1; done\n"
FAKE_SMI = r"""#!/usr/bin/env bash
case "$*" in
  *-lms*) while :; do echo "2026/11/10 10:00:00.000, 3, 1755, 1200"; sleep 0.2; done ;;
  *--query-gpu*) echo "NVIDIA H800, @D@, 81559 MiB, 00000000:C1:00.0, @U@" ;;
  *-q*) cat "@Q@" ;;
  *-L*) echo "GPU 0: NVIDIA H800 (UUID: @U@)" ;;
  *) echo "fake nvidia-smi" ;;
esac
"""
FAKE_LDCONFIG = """#!/bin/sh
echo "3 libs found in cache /etc/ld.so.cache"
printf '\\tlibvulkan.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libvulkan.so.1\\n'
printf '\\tlibGLX_nvidia.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libGLX_nvidia.so.0\\n'
printf '\\tlibEGL_nvidia.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libEGL_nvidia.so.0\\n'
"""


class Tree:
    pass


def canned_results(c: Path, sl: Path, black13=False, renderer13=None):
    dl = {"vkCreateInstance": "VK_SUCCESS(0)", "vkCreateDevice": "VK_SUCCESS(0)", "vkQueueSubmit": "VK_SUCCESS(0)",
          "vkWaitForFences": "VK_ERROR_DEVICE_LOST(-4)"}
    can = {"v01_gl_tri_R": (True, "done", {"compile_ok": True, "link_ok": True}),
           "v02_vk_nvS_env": (False, "submit", {"vk": dl}),
           "v02_vk_lvp": (True, "done", {"vk": {"vkCreateInstance": "VK_SUCCESS(0)"}}),
           "v11_vk_sysnew": (False, "submit", {"vk": dl}),
           "v12_vk_sysnew_noglvnd": (False, "create_instance",
                                     {"vk": {"vkCreateInstance": "VK_ERROR_INCOMPATIBLE_DRIVER(-9)"}}),
           "v13_thor_sysnew": (not black13, "done", {
               "frames": {"n_step_frames": 40, "frac_rgb_std_ge2": 0.0 if black13 else 1.0},
               "unity_maps_after_steps": [f"{sl}/libGLX_nvidia.so.{DRV}", "/env/lib/libvulkan.so.1.4.357"]})}
    for n, (p, stg, m) in can.items():
        wjson(c / f"{n}.json", {"pass": p, "stage": stg, "metrics": m, "notes": []})
    if renderer13:
        (c / "v13_thor_sysnew.renderer").write_text(renderer13)


def fake_tree(tmp: Path, honoured=True, sys_drv=DRV, black13=False, renderer13=None, **extra) -> Tree:
    t = Tree()
    t.out = tmp / "vout"
    t.nvgl = staged_tree(t.out)
    t.env_dir = t.out / "envs" / "tvr"
    (t.env_dir / "lib").mkdir(parents=True)
    (t.env_dir / "lib" / "libvulkan.so.1").write_text("fake env loader 1.4.357")
    t.lvp = t.env_dir / "share" / "vulkan" / "icd.d" / "lvp_icd.x86_64.json"
    wjson(t.lvp, {"ICD": {"library_path": "/x/libvulkan_lvp.so"}})
    t.build_name = f"thor-CloudRendering-vkretest{uuid.uuid4().hex[:12]}"   # unique: only our fake gets reaped
    b = t.out / "tvr" / "ai2thor_home" / ".ai2thor" / "releases" / t.build_name / t.build_name
    b.parent.mkdir(parents=True)
    b.write_text(FAKE_PLAYER)
    b.chmod(0o755)
    t.sl, t.vk, t.dev, t.drvf = fake_system(tmp / "sys", honoured, drv=sys_drv)
    canned = tmp / "canned"
    canned_results(canned, t.sl, black13=black13, renderer13=renderer13)
    bindir = tmp / "bin"
    bindir.mkdir()
    smi_q = tmp / "smi_q.txt"
    smi_q.write_text(SMI_Q.replace("@D@", sys_drv))
    py = tmp / "fakepy"
    py.write_text(FAKE_PY.replace("@C@", str(canned)).replace("@B@", t.build_name))
    (bindir / "nvidia-smi").write_text(FAKE_SMI.replace("@D@", sys_drv).replace("@U@", UUID_).replace("@Q@", str(smi_q)))
    (bindir / "ldconfig").write_text(FAKE_LDCONFIG)
    for f in (py, bindir / "nvidia-smi", bindir / "ldconfig"):
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    caps = {"NVIDIA_DRIVER_CAPABILITIES": "all"} if honoured else {}
    t.env = login_env(VPD_VKRE_OUT=str(t.out), VPD_VKRE_PY=str(py), VPD_VKRE_TOOL_PY=sys.executable,
                      VPD_VKRE_SYS_LIB=str(t.sl), VPD_VKRE_VK_SYS_DIRS=str(t.vk),
                      VPD_VKRE_EGL_SYS_DIRS=str(tmp / "sys" / "egl_vendor"), VPD_VKRE_DEV_DIR=str(t.dev),
                      VPD_VKRE_DRIVER_FILE=str(t.drvf), VPD_VKRE_REAP_WAIT="10", VPD_VKRE_BUILD_NAME=t.build_name,
                      PATH=f"{bindir}:{os.environ['PATH']}", **caps, **extra)
    return t


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
        for pidf in Path(outdir).glob("*/fake_unity.pid"):
            pid = int(pidf.read_text())
            if alive(pid):
                os.kill(pid, 9)
                if r is not None:
                    r.left_alive = True


def penv(d: Path, name="env_profile.txt") -> dict:
    return dict(ln.split("=", 1) for ln in (d / name).read_text().splitlines() if "=" in ln)


def steps_of(od: Path) -> list:
    return [json.loads(x)["name"] for x in (od / "steps.jsonl").read_text().splitlines()]


def test_job_full_flow_with_fakes(tmp_path):
    t = fake_tree(tmp_path)
    od = tmp_path / "job" / "out"
    r = run_job(t.env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    steps = steps_of(od)
    assert [s.split(":")[0] for s in steps if s != "summary"] == vkre.VARIANTS and steps[-1] == "summary"
    assert not [s for s in steps if s.endswith(":skipped")]
    for n in vkre.VARIANTS:
        res = rjson(od / n / "result.json")
        assert res["name"] == n and isinstance(res["rc"], int) and res["cap_s"] > 0, n
    # v00: the facts the recheck adds
    m = rjson(od / "v00_facts" / "result.json")["metrics"]
    assert m["kernel_driver"] == DRV and m["staged_driver_match"] is True and m["caps_honored"] is True
    assert m["gpu"]["uuid"] == UUID_ and m["gpu"]["pci_bus_id"] == "00000000:C1:00.0" and m["gpu"]["vbios"]
    assert m["system_equals_staged"] == {f"libEGL_nvidia.so.{DRV}": False, f"libGLX_nvidia.so.{DRV}": True,
                                         f"libnvidia-glcore.so.{DRV}": None}
    assert m["ldconfig_gfx"] == {"libvulkan.so.1": ["/usr/lib/x86_64-linux-gnu/libvulkan.so.1"]}
    assert "NVIDIA UNIX Open Kernel Module" in (od / "v00_facts" / "driver_version.txt").read_text()
    assert UUID_ in (od / "v00_facts" / "node.txt").read_text()
    # v11 / v12 / v13: the system stack with absolute JSON written at run time, env loader first
    glvnd = f"{t.env_dir}/lib:{t.nvgl}/glvnd"
    for n, ld in (("v11_vk_sysnew", glvnd), ("v12_vk_sysnew_noglvnd", f"{t.env_dir}/lib"), ("v13_thor_sysnew", glvnd)):
        d = od / n
        e, fe = penv(d), penv(d, "fake_env.txt")
        assert e["LD_LIBRARY_PATH"] == fe["LD_LIBRARY_PATH"] == ld and f"{t.nvgl}/lib" not in ld, n
        assert e["VK_ICD_FILENAMES"] == e["VK_DRIVER_FILES"] == fe["VK_ICD_FILENAMES"] == str(d / "sys_icd.json"), n
        assert e["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(d / "sys_egl.json") and e["HOME"] == str(d / "home"), n
        assert rjson(d / "sys_icd.json") == {"file_format_version": "1.0.1", "ICD": {
            "library_path": f"{t.sl}/libGLX_nvidia.so.0", "api_version": "1.4.329"}}, n
        assert rjson(d / "sys_egl.json")["ICD"]["library_path"] == f"{t.sl}/libEGL_nvidia.so.0", n
    for n in ("v02_vk_nvS_env", "v02_vk_lvp", "v11_vk_sysnew", "v12_vk_sysnew_noglvnd"):
        assert penv(od / n)["VK_LOADER_DEBUG"] == penv(od / n, "fake_env.txt")["VK_LOADER_DEBUG"] == "all", n
        assert f"--loader {t.env_dir}/lib/libvulkan.so.1 --want" in (od / n / "fake_args.txt").read_text(), n
    assert "VK_LOADER_DEBUG" not in penv(od / "v13_thor_sysnew")   # Unity gets it as --child-env only
    assert "VK_LOADER_DEBUG" not in penv(od / "v01_gl_tri_R")
    # the controls are the 11-09 profiles
    e = penv(od / "v02_vk_nvS_env")
    assert e["LD_LIBRARY_PATH"] == f"{t.env_dir}/lib:{t.nvgl}/lib:{t.nvgl}/glvnd"
    assert e["VK_ICD_FILENAMES"] == f"{t.nvgl}/vulkan/nvidia_icd.json"
    assert e["__EGL_VENDOR_LIBRARY_FILENAMES"] == f"{t.nvgl}/egl/10_nvidia.json"
    e = penv(od / "v02_vk_lvp")
    assert e["VK_ICD_FILENAMES"] == str(t.lvp) and e["LP_NUM_THREADS"] == "4"
    e = penv(od / "v01_gl_tri_R")
    assert "VK_ICD_FILENAMES" not in e and "LD_LIBRARY_PATH" not in e
    assert e["__EGL_VENDOR_LIBRARY_DIRS"] == str(tmp_path / "sys" / "egl_vendor")
    assert f"--egl-lib {t.nvgl}/glvnd/libEGL.so.1 --gl-lib {t.nvgl}/glvnd/libOpenGL.so.0" in \
        (od / "v01_gl_tri_R" / "fake_args.txt").read_text()   # the image has no glvnd: staged dispatch libs
    a13 = (od / "v13_thor_sysnew" / "fake_args.txt").read_text()
    for x in ("--gpu-device 0 --scene FloorPlan1 --width 640 --height 360 --quality Ultra --aa 4 --steps 40 --depth "
              "--seg", "--child-env VK_LOADER_DEBUG=all", f"--child-env LD_DEBUG_OUTPUT={od}/v13_thor_sysnew/lddebug"):
        assert x in a13, x
    # every probe argv parses with the real debug_probe.py parser, to the intended values
    parsed = {}
    for n in vkre.VARIANTS[1:]:
        argv = (od / n / "fake_argv.txt").read_text().splitlines()
        p = subprocess.run([sys.executable, "-I", str(PROBE), *argv], env=login_env(VPD_TVR_PARSE_ONLY="1"),
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, (n, argv, p.stderr[-1500:])
        parsed[n] = json.loads(p.stdout)
    assert [parsed[n]["cmd"] for n in vkre.VARIANTS[1:]] == ["gl-tri", "vk-tri", "vk-tri", "vk-tri", "vk-tri", "thor"]
    assert parsed["v11_vk_sysnew"]["want"] == "nvidia" and parsed["v02_vk_lvp"]["want"] == "cpu"
    p13 = parsed["v13_thor_sysnew"]
    assert (p13["gpu_device"], p13["scene"], p13["width"], p13["height"], p13["quality"], p13["aa"], p13["steps"],
            p13["depth"], p13["seg"]) == (0, "FloorPlan1", 640, 360, "Ultra", "4", 40, True, True)
    assert p13["startup_timeout"] == 200.0 and p13["deadline_s"] == 215.0
    # v13: thor-check applied the renderer rule; the loader log came out of Player.log
    r13 = rjson(od / "v13_thor_sysnew" / "result.json")
    v = r13["metrics"]["vkre"]
    assert r13["pass"] is True and v["renderers"] == ["NVIDIA H800"] and v["unity_glx_staged"] is False
    assert v["loader"]["manifests"] == [str(od / "v13_thor_sysnew" / "sys_icd.json")]
    assert "post" in r13["metrics"]   # post-thor ran too
    # the fake player (TERM ignored) left by v13 was reaped with KILL
    pid = int((od / "v13_thor_sysnew" / "fake_unity.pid").read_text())
    assert r.left_alive is False and not alive(pid)
    reap = (od / "logs" / "reap.log").read_text()
    assert f"TERM lingering Unity player(s): {pid}" in reap and "KILL" in reap
    # summary: verdict and flags
    s = rjson(od / "summary.json")
    assert s["verdict"] == "PASS" and s["variants_status"] == {
        "v00_facts": "PASS", "v01_gl_tri_R": "PASS", "v02_vk_nvS_env": "FAIL", "v02_vk_lvp": "PASS",
        "v11_vk_sysnew": "FAIL", "v12_vk_sysnew_noglvnd": "FAIL", "v13_thor_sysnew": "PASS"}
    f = s["flags"]
    assert f["glvnd_explains_R"] is True and f["nv_vk_pass"] is False and f["nv_all_fail"] is False
    assert f["kernel_driver"] == DRV and f["gpu_uuid"] == UUID_ and f["system_equals_staged_all"] is False
    assert s["variants"]["v11_vk_sysnew"]["key"]["loader_log"]["manifests"] == [str(od / "v11_vk_sysnew" / "sys_icd.json")]
    assert s["overrides"]["VPD_VKRE_BUILD_NAME"] == t.build_name   # a real run lists none
    assert "verdict \"PASS\"" in r.stdout
    # the orchestrator's check
    e = subprocess.run([sys.executable, "-I", str(VKRE), "expect", "--out", str(od)], capture_output=True, text=True,
                       timeout=60, env=login_env())
    assert e.returncode == 0, e.stdout
    got = json.loads(e.stdout)
    assert got["ok"] is True and got["verdict"] == "PASS" and got["flags"]["glvnd_explains_R"] is True
    assert "vkre.py expect --out" in rjson(od / "expect.json")["check"]
    # a second run into the same OUTDIR is refused; so is a missing OUTDIR argument
    r2 = subprocess.run(["bash", str(JOB), str(od)], env=t.env, capture_output=True, text=True, timeout=60)
    assert r2.returncode == 2 and "refusing" in r2.stderr
    r3 = subprocess.run(["bash", str(JOB)], env=t.env, capture_output=True, text=True, timeout=60)
    assert r3.returncode == 2 and "usage" in r3.stderr


def test_job_without_graphics_caps_skips_the_system_variants(tmp_path):
    t = fake_tree(tmp_path, honoured=False)
    od = tmp_path / "job" / "out"
    r = run_job(t.env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    for n in ("v01_gl_tri_R", "v11_vk_sysnew", "v12_vk_sysnew_noglvnd", "v13_thor_sysnew"):
        res = rjson(od / n / "result.json")
        assert res["pass"] is None and res["skipped"].startswith("caps_not_honored"), n
        assert f"{n}:skipped" in steps_of(od)
    assert rjson(od / "v02_vk_nvS_env" / "result.json")["pass"] is False   # the staged control still runs
    assert rjson(od / "v02_vk_lvp" / "result.json")["pass"] is True
    s = rjson(od / "summary.json")
    assert s["verdict"] == "NOT_TESTED" and s["flags"]["caps_honored"] is False
    assert vkre.expect(str(od))["ok"] is True


def test_job_on_another_kernel_driver(tmp_path):
    """J-B may land on a node with another driver: the staged control is skipped, the system variants run."""
    t = fake_tree(tmp_path, sys_drv=OTHER_DRV, black13=True)
    od = tmp_path / "job" / "out"
    r = run_job(t.env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    res = rjson(od / "v02_vk_nvS_env" / "result.json")
    assert res["pass"] is None and f'kernel driver "{OTHER_DRV}" is not {DRV}' in res["skipped"]
    m = rjson(od / "v00_facts" / "result.json")["metrics"]
    assert m["kernel_driver"] == OTHER_DRV and m["caps_honored"] is True and m["system_equals_staged_all"] is None
    for n in ("v11_vk_sysnew", "v12_vk_sysnew_noglvnd", "v13_thor_sysnew"):
        assert not rjson(od / n / "result.json").get("skipped"), n
    s = rjson(od / "summary.json")
    assert s["verdict"] == "FAIL" and s["flags"]["nv_all_fail"] is True
    assert s["flags"]["nv_tested"] == ["v11_vk_sysnew", "v12_vk_sysnew_noglvnd", "v13_thor_sysnew"]


def test_job_v13_on_the_wrong_renderer_fails(tmp_path):
    t = fake_tree(tmp_path, renderer13="llvmpipe (LLVM 23.1.4, 256 bits)")
    od = tmp_path / "job" / "out"
    r = run_job(t.env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    r13 = rjson(od / "v13_thor_sysnew" / "result.json")
    assert r13["pass"] is False and r13["pass_frames"] is True
    assert rjson(od / "summary.json")["verdict"] == "FAIL"


def test_job_soft_deadline_skips_but_still_summarises(tmp_path):
    t = fake_tree(tmp_path, VPD_JOB_DEADLINE_S="1", VPD_VKRE_RESERVE_S="0", VPD_VKRE_MIN_VARIANT_S="30")
    od = tmp_path / "job" / "out"
    r = run_job(t.env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    for n in vkre.VARIANTS:
        res = rjson(od / n / "result.json")
        assert res["pass"] is None and res["skipped"], n
    assert rjson(od / "v00_facts" / "result.json")["skipped"].startswith("no_time")
    s = rjson(od / "summary.json")
    assert s["verdict"] == "NOT_TESTED" and set(s["variants_status"].values()) == {"SKIP"}
    assert vkre.expect(str(od))["ok"] is True


def test_job_profile_failure_is_a_harness_error(tmp_path):
    """A sysnew profile that cannot write its JSON (no system ICD manifest dir writable) is ERROR, not FAIL."""
    t = fake_tree(tmp_path)
    od = tmp_path / "job" / "out"
    # pre-create v11's ICD path as a directory: icd-json cannot write it, so the profile fails
    (od / "v11_vk_sysnew" / "sys_icd.json").mkdir(parents=True)
    r = run_job(t.env, od)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert (od / "v11_vk_sysnew" / "profile_error.txt").read_text().startswith("profile sysnew")
    assert not (od / "v11_vk_sysnew" / "fake_argv.txt").exists()   # the probe never started
    s = rjson(od / "summary.json")
    assert s["variants_status"]["v11_vk_sysnew"] == "ERROR" and s["verdict"] == "PASS"
    assert s["flags"]["glvnd_explains_R"] is None
