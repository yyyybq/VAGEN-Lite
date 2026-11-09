"""Parsers of a thor variant's side outputs (stdlib only; runs after the variant, in the job or in tests).

  lddebug.<pid>         glibc LD_DEBUG=libs,files of the Unity process: libraries searched for but never mapped
                        (a failed dlopen leaves exactly that), and the GPU libraries that were initialised
  gpu.csv               background `nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.gr,memory.used
                        --format=csv,noheader,nounits -lms 500`: peak utilisation / graphics clock, memory rise
  nvcache/              __GL_SHADER_DISK_CACHE_PATH: files and bytes the NVIDIA shader cache wrote
  unity.log             Unity's stderr = the Vulkan loader log (VK_LOADER_DEBUG=all): ICD manifests, driver
                        libraries and layers the loader found or inserted, ERROR lines
  nvidia_smi_mid.txt    `nvidia-smi` after start-up: the process table rows
merge_into_result(DIR) adds these as metrics.post to DIR/result.json (a stub if the probe wrote none).
"""
from __future__ import annotations

import glob
import os
import re

from . import common

LD_PREFIX = re.compile(r"^\s*(\d+):\s?(.*)$")
FIND = re.compile(r"^find library=(\S+) \[\d+\]; searching")
LINKMAP = re.compile(r"^file=(\S+) \[\d+\];\s+generating link map")
LOADED_BY = re.compile(r"^file=(\S+) \[\d+\];\s+(needed by|dynamically loaded by) (\S+)")
INIT = re.compile(r"^calling init: (\S+)")
GPU_NAME = re.compile(r"(nvidia|GLX|EGL|vulkan|lvp|cuda|libGL\b|OpenGL|GLdispatch)", re.I)
LOADER_KEYS = re.compile(r"(manifest|ICD|[Dd]river|[Ll]ayer|ERROR|WARNING|Vulkan Loader)")
SO_TOKEN = re.compile(r"(/[^\s,\"']+\.so(?:\.[0-9.]+)?|lib[A-Za-z0-9_+-]+\.so(?:\.[0-9.]+)?)")
JSON_TOKEN = re.compile(r"(/[^\s,\"']+\.json)")
LAYER_TOKEN = re.compile(r"\b(VK_LAYER_[A-Za-z0-9_]+)")


def parse_lddebug(paths) -> dict:
    paths = list(paths)
    searched, mapped, inits, requested_by, n_lines = {}, set(), [], {}, 0
    for p in paths:
        try:
            f = open(p, errors="replace")
        except OSError:
            continue
        with f:
            for line in f:
                n_lines += 1
                mm = LD_PREFIX.match(line)
                body = (mm.group(2) if mm else line).strip()
                x = FIND.match(body)
                if x:
                    searched[x.group(1)] = searched.get(x.group(1), 0) + 1
                    continue
                x = LINKMAP.match(body)
                if x:
                    mapped.add(os.path.basename(x.group(1)))
                    continue
                x = LOADED_BY.match(body)
                if x:
                    requested_by.setdefault(os.path.basename(x.group(1)), f"{x.group(2)} {x.group(3)}")
                    continue
                x = INIT.match(body)
                if x:
                    inits.append(x.group(1))
    never = sorted(n for n in searched if os.path.basename(n) not in mapped)
    return {"files": len(paths), "lines": n_lines, "searched": len(searched), "mapped": len(mapped),
            "never_loaded": never[:200], "never_loaded_gpu": [n for n in never if GPU_NAME.search(n)],
            "never_loaded_gpu_requested_by": {n: requested_by.get(os.path.basename(n), "dlopen")
                                              for n in never if GPU_NAME.search(n)},
            "initialised_gpu_libs": sorted({p for p in inits if GPU_NAME.search(os.path.basename(p))})[:200]}


def _num(s):
    s = s.strip().split(" ")[0]
    try:
        return float(s)
    except ValueError:
        return None


def parse_gpu_csv(path) -> dict:
    rows = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if len(parts) < 3:
                    continue
                util, clk, mem = (_num(x) for x in parts[-3:])
                if None in (util, clk, mem):
                    continue  # header or "[N/A]"
                rows.append((util, clk, mem))
    except OSError:
        return {"samples": 0, "error": f"missing {path}"}
    if not rows:
        return {"samples": 0}
    util = [r[0] for r in rows]
    clk = [r[1] for r in rows]
    mem = [r[2] for r in rows]
    return {"samples": len(rows), "util_max": max(util), "util_mean": round(sum(util) / len(util), 2),
            "util_nonzero_samples": sum(1 for u in util if u > 0), "clock_gr_max_mhz": max(clk),
            "clock_gr_min_mhz": min(clk), "mem_first_mib": mem[0], "mem_max_mib": max(mem),
            "mem_rise_mib": max(mem) - mem[0]}


def dir_stats(path) -> dict:
    n = size = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                size += os.path.getsize(os.path.join(root, fn))
                n += 1
            except OSError:
                pass
    return {"exists": os.path.isdir(path), "files": n, "bytes": size}


def parse_loader_log(path, excerpt_path=None, keep=400) -> dict:
    lines, icds, libs, layers, errors = [], [], [], [], []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if not LOADER_KEYS.search(line):
                    continue
                s = line.rstrip("\n")
                if len(lines) < keep:
                    lines.append(s)
                if "manifest" in s.lower() or "icd" in s.lower() or "driver" in s.lower():
                    icds += JSON_TOKEN.findall(s)
                    libs += [t for t in SO_TOKEN.findall(s) if GPU_NAME.search(t)]
                layers += LAYER_TOKEN.findall(s)
                if "ERROR" in s and len(errors) < 30:
                    errors.append(s.strip()[:300])
    except OSError:
        return {"present": False}
    if excerpt_path:
        with open(excerpt_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    def uniq(xs):
        return list(dict.fromkeys(xs))[:50]
    return {"present": True, "loader_lines": len(lines), "manifests": uniq(icds), "driver_libs": uniq(libs),
            "layers": uniq(layers), "errors": errors}


def parse_smi_processes(path) -> list:
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return []
    if "Processes:" not in text:
        return []
    tail = text.split("Processes:", 1)[1]
    return [ln.strip(" |") for ln in tail.splitlines() if "MiB" in ln or "No running processes" in ln][:20]


def collect(d: str) -> dict:
    return {"lddebug": parse_lddebug(sorted(glob.glob(os.path.join(d, "lddebug.*")))),
            "gpu": parse_gpu_csv(os.path.join(d, "gpu.csv")),
            "nvcache": dir_stats(os.path.join(d, "nvcache")),
            "loader": parse_loader_log(os.path.join(d, "unity.log"), os.path.join(d, "loader_excerpt.txt")),
            "smi_processes": parse_smi_processes(os.path.join(d, "nvidia_smi_mid.txt"))}


def merge_into_result(d: str) -> dict:
    p = os.path.join(d, "result.json")
    res = common.read_json(p)
    if not isinstance(res, dict):
        res = common.new_result(os.path.basename(d.rstrip("/")))
        res["notes"].append("no result.json from the probe (killed before it wrote one?)")
    res.setdefault("metrics", {})["post"] = collect(d)
    common.write_json_atomic(p, res)
    return res
