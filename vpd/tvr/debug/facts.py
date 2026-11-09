"""V0 facts (stdlib only): what graphics stack the job container really has.

debug_job.sh first saves `nvidia-smi -q`, `ldconfig -p | grep -i nvidia`, the NVIDIA_* / VK_* environment and
`ls -la /dev/dri /dev/nvidia*` into OUT/v00_facts/; collect() parses those and inspects the files itself:
  - readlink -f and size of the system libGLX_nvidia.so.0, libEGL_nvidia.so.0, libnvidia-glvkspirv*, the other
    graphics libs, and whether glvnd's libEGL.so.1 / libOpenGL.so.0 exist in the image;
  - every {/etc,/usr/share}/vulkan/{icd.d,implicit_layer.d,explicit_layer.d}/*.json and glvnd EGL vendor
    JSON: size and library_path;
  - /dev/dri, /dev/nvidia-modeset; MIG mode, GPU operation mode (GOM), compute mode, display mode / active.
metrics.caps_honored = the system libGLX_nvidia.so.0 resolves to a non-empty file named *.<driver> AND some
system nvidia_icd.json is non-empty (= the container runtime mounted the graphics stack; then the [R] variants
run).  metrics.staging lists the inputs the other variants need.  pass = nvidia-smi -q was readable.
"""
from __future__ import annotations

import glob
import json
import os
import re

from . import common

SMI_KEYS = ["Product Name", "Product Architecture", "Driver Version", "CUDA Version", "Display Mode",
            "Display Attached", "Display Active", "Persistence Mode", "Addressing Mode", "MIG Mode/Current",
            "GPU Operation Mode/Current", "Compute Mode", "GPU Virtualization Mode/Virtualization Mode"]
LEAF = re.compile(r"^(\s*)(\S.*?)\s+:\s?(.*)$")
GRAPHICS_LIBS = ["libGLX_nvidia.so.0", "libEGL_nvidia.so.0", "libGLESv2_nvidia.so.2", "libnvidia-glcore.so.{d}",
                 "libnvidia-eglcore.so.{d}", "libnvidia-glsi.so.{d}", "libnvidia-tls.so.{d}",
                 "libnvidia-glvkspirv.so.{d}", "libnvidia-gpucomp.so.{d}", "libnvidia-rtcore.so.{d}",
                 "libnvidia-allocator.so.1", "libnvidia-present.so.{d}", "libEGL.so.1", "libOpenGL.so.0",
                 "libGL.so.1", "libGLX.so.0", "libGLdispatch.so.0", "libvulkan.so.1"]


def parse_smi_q(text: str) -> dict:
    """Flat {"Header/Key": value} of `nvidia-smi -q` (first occurrence; the per-GPU header is dropped)."""
    out, stack = {}, []  # stack of (indent, name)
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("="):
            continue
        m = LEAF.match(line)
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if m and m.group(3).strip():
            key = "/".join([n for _, n in stack] + [m.group(2).strip()])
            out.setdefault(key, m.group(3).strip())
        else:
            name = (m.group(2) if m else line).strip()
            if re.match(r"^GPU [0-9A-Fa-f]{4,}", name):
                continue  # 'GPU 00000000:18:00.0' opens the per-GPU block: not part of the keys
            stack.append((indent, name))
    return out


def resolve(path: str) -> dict:
    real = os.path.realpath(path)
    d = {"path": path, "exists": os.path.lexists(path), "realpath": real if os.path.lexists(path) else None}
    if os.path.isfile(real):
        d["size"] = os.path.getsize(real)
    return d


def json_info(path: str) -> dict:
    d = {"path": path, "size": os.path.getsize(path) if os.path.isfile(path) else None}
    try:
        with open(path) as f:
            doc = json.load(f)
        if "ICD" in doc:
            d["library_path"] = doc["ICD"].get("library_path")
            d["api_version"] = doc["ICD"].get("api_version")
        if "layers" in doc or "layer" in doc:
            d["layers"] = [(ly.get("name"), ly.get("library_path")) for ly in (doc.get("layers") or [doc["layer"]])]
    except (OSError, ValueError, AttributeError, TypeError) as e:
        d["parse_error"] = type(e).__name__
    return d


def _read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def collect(out: str, sys_lib="/usr/lib/x86_64-linux-gnu", vk_dirs=("/etc/vulkan", "/usr/share/vulkan"),
            egl_vendor_dirs=("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"), dev_dir="/dev",
            driver="595.71.05", driver_file="/proc/driver/nvidia/version", nvgl=None, env_dir=None, build=None,
            lvp_icd=None, o3d_py=None, environ=None) -> dict:
    environ = os.environ if environ is None else environ
    res = common.new_result("facts")
    m = res["metrics"]
    smi_text = _read(os.path.join(out, "nvidia_smi_q.txt")) or ""
    m["smi_q"] = {k: v for k, v in parse_smi_q(smi_text).items() if k in SMI_KEYS}
    res["pass"] = bool(m["smi_q"].get("Product Name"))
    if not res["pass"]:
        res["notes"].append("nvidia-smi -q gave no Product Name")
    drv_text = _read(driver_file) or ""
    m["driver_file"] = drv_text.strip().splitlines()[0] if drv_text.strip() else None
    m["driver_match"] = f" {driver} " in drv_text
    nv_env = {k: v for k, v in environ.items() if k.startswith("NVIDIA_")}
    m["nvidia_env"] = nv_env
    caps = (nv_env.get("NVIDIA_DRIVER_CAPABILITIES") or "").lower()
    m["graphics_requested"] = "graphics" in caps.split(",") or caps.strip() == "all"
    libs = {}
    for name in GRAPHICS_LIBS:
        n = name.format(d=driver)
        libs[n] = resolve(os.path.join(sys_lib, n))
    for p in sorted(glob.glob(os.path.join(sys_lib, "libnvidia-glvkspirv*"))):
        libs.setdefault(os.path.basename(p), resolve(p))
    m["system_libs"] = libs
    glx = libs["libGLX_nvidia.so.0"]
    glx_ok = bool(glx.get("size")) and glx.get("realpath") and os.path.basename(glx["realpath"]).endswith(driver)
    jsons = {}
    for d in vk_dirs:
        for sub in ("icd.d", "implicit_layer.d", "explicit_layer.d"):
            for p in sorted(glob.glob(os.path.join(d, sub, "*.json"))):
                jsons[p] = json_info(p)
    m["vulkan_json"] = jsons
    icd_ok = any(os.path.basename(p) == "nvidia_icd.json" and (j.get("size") or 0) > 0 for p, j in jsons.items())
    m["egl_vendor_json"] = {p: json_info(p) for d in egl_vendor_dirs
                            for p in sorted(glob.glob(os.path.join(d, "*.json")))}
    m["caps_honored"] = bool(glx_ok and icd_ok)
    m["caps_detail"] = {"libGLX_nvidia_ok": bool(glx_ok), "nvidia_icd_json_ok": bool(icd_ok)}
    m["system_glvnd"] = {"libEGL.so.1": bool(libs["libEGL.so.1"].get("size")),
                         "libOpenGL.so.0": bool(libs["libOpenGL.so.0"].get("size"))}
    m["dev"] = {"dri": sorted(os.listdir(os.path.join(dev_dir, "dri")))
                if os.path.isdir(os.path.join(dev_dir, "dri")) else None,
                "nvidia_modeset": os.path.exists(os.path.join(dev_dir, "nvidia-modeset")),
                "nvidia_nodes": sorted(n for n in (os.listdir(dev_dir) if os.path.isdir(dev_dir) else [])
                                       if n.startswith("nvidia"))}
    ldc = _read(os.path.join(out, "ldconfig_nvidia.txt")) or ""
    m["ldconfig_nvidia_lines"] = len([ln for ln in ldc.splitlines() if ln.strip()])
    m["ldconfig_has_libGLX_nvidia"] = "libGLX_nvidia.so.0" in ldc
    st = {}
    if nvgl:
        for rel in ("lib/libGLX_nvidia.so.0", "lib/libEGL_nvidia.so.0", "glvnd/libEGL.so.1", "glvnd/libOpenGL.so.0",
                    "vulkan/nvidia_icd.json", "egl/10_nvidia.json"):
            p = os.path.join(nvgl, rel)
            st[f"nvgl/{rel}"] = os.path.isfile(p) and os.path.getsize(os.path.realpath(p)) > 0
    if env_dir:
        st["env/bin/python"] = os.access(os.path.join(env_dir, "bin", "python"), os.X_OK)
        st["env/lib/libvulkan.so.1"] = os.path.isfile(os.path.join(env_dir, "lib", "libvulkan.so.1"))
    if lvp_icd is not None:
        st["lvp_icd"] = bool(lvp_icd) and os.path.isfile(lvp_icd)
    if build:
        st["build"] = os.access(build, os.X_OK)
    if o3d_py is not None:
        st["o3d_python"] = bool(o3d_py) and os.access(o3d_py, os.X_OK)
    m["staging"] = st
    res["stage"] = "done"
    return res
