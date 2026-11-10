"""vpd-tvr-vkre: bookkeeping of the NVIDIA Vulkan recheck job (EXPLORATORY; gate tvr; DECISIONS G2).

Run as `PY -I -u vpd/tvr/vkre.py SUBCOMMAND ...` by vpd/tvr/vk_recheck.sh (the job payload).  The GPU probes
themselves are debug_probe.py's gl-tri / vk-tri / thor (debug/gl_tri.py, vk_tri.py, thor_diag.py), unchanged.
Everything here is stdlib only, runs under any interpreter (the TVR env's py3.10 in the job, eval-external or
eval-external-q3 on the login node) and never touches the GPU:
  facts       v00: debug.facts.collect keyed on the PARSED kernel driver version (a node with another driver
              is then not mistaken for "graphics capability not honoured"), plus kernel_driver,
              staged_driver_match, the card's UUID / PCI bus id / VBIOS / GSP firmware (nvidia-smi -q), whether
              the system NVIDIA libraries equal the staged copies (sha256 against nvgl_595.71.05/SHA256SUMS),
              the glvnd / Vulkan libraries ldconfig knows, and the system nvidia_icd.json
  icd-json    write a Vulkan ICD manifest with an ABSOLUTE library_path (version fields copied from --src)
  egl-json    write a glvnd EGL vendor manifest with an ABSOLUTE library_path
  thor-check  v13: pass = thor_diag's frame rule AND Player.log names the wanted Vulkan renderer; also parses
              the Vulkan loader log out of Player.log (Unity writes the loader's stderr there, not to unity.log)
              and records where Unity's libGLX_nvidia was mapped from
  summarize   OUT/summary.json (status per variant, the recheck verdict, flags, readings) and OUT/expect.json
  expect      the orchestrator's check before it deletes the job record: every expected output is present and
              parseable; prints the verdict as one JSON object; exit 0 iff complete
Verdict (DECISIONS G2): PASS on this pool iff v13_thor_sysnew PASS; FAIL if v13 ran and failed; NOT_TESTED if
v13 was skipped or its profile could not be set up.  Never feeds the frozen TVR smoke verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys

if __package__:
    from .debug import common, post, thor_diag
    from .debug import facts as dfacts
    from .debug import summarize as dsum
else:  # run as a script with -I: put vpd/tvr (our code only) back in front, as debug_probe.py does
    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE
    from debug import common, post, thor_diag  # noqa: E402
    from debug import facts as dfacts  # noqa: E402
    from debug import summarize as dsum  # noqa: E402

# run order of vk_recheck.sh (the tests check that the script runs exactly these, in this order)
VARIANTS = ["v00_facts", "v01_gl_tri_R", "v02_vk_nvS_env", "v02_vk_lvp", "v11_vk_sysnew",
            "v12_vk_sysnew_noglvnd", "v13_thor_sysnew"]
KIND = {"v00_facts": "facts", "v01_gl_tri_R": "gl", "v02_vk_nvS_env": "vk", "v02_vk_lvp": "vk",
        "v11_vk_sysnew": "vk", "v12_vk_sysnew_noglvnd": "vk", "v13_thor_sysnew": "thor"}
NV_VK = ["v02_vk_nvS_env", "v11_vk_sysnew", "v12_vk_sysnew_noglvnd"]  # NVIDIA Vulkan triangles
VERDICT_VARIANT = "v13_thor_sysnew"
VERDICTS = ("PASS", "FAIL", "NOT_TESTED")
STAGED_DRIVER = "595.71.05"  # the driver of the staged overlay OUT/tvr/nvgl_595.71.05
WANT_RENDERER = "NVIDIA H800"
# the login node's /etc/vulkan/icd.d/nvidia_icd.json (driver 595.71.05, read 2026-11-09): used when the job
# has no readable system ICD manifest to copy the version fields from
ICD_FALLBACK = {"file_format_version": "1.0.1", "api_version": "1.4.329"}
EGL_VENDOR_FORMAT = "1.0.0"
PASS_RULE = (f"recheck PASS on this pool iff {VERDICT_VARIANT} PASS = at least one step frame, at least "
             f"{thor_diag.PASS_FRAC:.0%} of the step frames with RGB std >= {thor_diag.STD_MIN:g} (debug/thor_diag.py)"
             f" AND Player.log 'Vulkan renderer=[...{WANT_RENDERER}...]' (DECISIONS G2)")
EXPECT_FILES = ["steps.jsonl", "summary.json"] + [f"{n}/result.json" for n in VARIANTS]

KERNEL_RE = re.compile(r"Kernel Module(?: for \S+)?\s+(\d+\.\d+(?:\.\d+)?)")
NVRM_NUM_RE = re.compile(r"\b(\d{3,}\.\d+(?:\.\d+)?)\b")
RENDERER_RE = re.compile(r"Vulkan renderer=\[([^\]]*)\]")
VK_API_RE = re.compile(r"Vulkan API version[^\n]*")
LDCONFIG_RE = re.compile(r"^\s*(\S+\.so[.0-9]*)\s+\(([^)]*)\)\s+=>\s+(\S+)")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SMI_GPU = {"product": "Product Name", "uuid": "GPU UUID", "pci_bus_id": "PCI/Bus Id", "vbios": "VBIOS Version",
           "gsp_firmware": "GSP Firmware Version", "serial": "Serial Number", "driver_version": "Driver Version",
           "cuda_version": "CUDA Version"}
FACT_KEYS = ("kernel_driver", "staged_driver_match", "caps_honored", "caps_detail", "graphics_requested", "gpu",
             "hostname", "system_equals_staged_all", "system_equals_staged", "system_glvnd", "system_icd_json",
             "ldconfig_gfx", "dev", "staging")


def _read(path):
    if not path:
        return None
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


# --- facts --------------------------------------------------------------------------------------------------

def parse_kernel_driver(text) -> str | None:
    """The kernel module version of /proc/driver/nvidia/version ('... Kernel Module [for x86_64]  595.71.05 ...')."""
    if not text:
        return None
    m = KERNEL_RE.search(text)
    if m:
        return m.group(1)
    for line in text.splitlines():
        if "NVRM" in line:
            m = NVRM_NUM_RE.search(line)
            if m:
                return m.group(1)
    return None


def read_sums(path) -> dict:
    """{basename: sha256} of a sha256sum file ('<hash>  ./name' or '<hash> *name'); {} if unreadable."""
    out = {}
    for line in (_read(path) or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and SHA_RE.match(parts[0]):
            out[os.path.basename(parts[1].strip().lstrip("*"))] = parts[0]
    return out


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def system_vs_staged(sys_lib: str, sums: dict) -> dict:
    """Per staged library: does SYS_LIB/<name> exist and have the staged sha256?  equal None = not on the system."""
    cmp = {}
    for name, want in sorted(sums.items()):
        p = os.path.join(sys_lib, name)
        if not os.path.isfile(p):
            cmp[name] = {"equal": None, "present": False}
            continue
        try:
            got = sha256_file(p)
        except OSError as e:
            cmp[name] = {"equal": None, "present": True, "error": repr(e)}
            continue
        cmp[name] = {"equal": got == want, "present": True, "size": os.path.getsize(p), "sha256": got}
    return cmp


def all_equal(cmp: dict):
    """True iff every staged library is on the system with the staged hash; None if none of them is there."""
    vals = [v.get("equal") for v in cmp.values()]
    if not any(v is not None for v in vals):
        return None
    return all(v is True for v in vals)


def parse_ldconfig(text) -> dict:
    """{soname: [paths]} of `ldconfig -p` lines."""
    out = {}
    for line in (text or "").splitlines():
        m = LDCONFIG_RE.match(line)
        if m:
            out.setdefault(m.group(1), []).append(m.group(3))
    return out


def system_icd(vk_dirs) -> dict | None:
    """The first non-empty icd.d/nvidia_icd.json under vk_dirs, with its version fields and library_path."""
    for d in vk_dirs:
        p = os.path.join(d, "icd.d", "nvidia_icd.json")
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            doc = common.read_json(p)
            icd = doc.get("ICD") if isinstance(doc, dict) else None
            return {"path": p, "parsed": isinstance(icd, dict),
                    "file_format_version": doc.get("file_format_version") if isinstance(doc, dict) else None,
                    "library_path": icd.get("library_path") if isinstance(icd, dict) else None,
                    "api_version": icd.get("api_version") if isinstance(icd, dict) else None}
    return None


def collect_facts(out: str, sys_lib="/usr/lib/x86_64-linux-gnu", vk_dirs=("/etc/vulkan", "/usr/share/vulkan"),
                  egl_vendor_dirs=("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"), dev_dir="/dev",
                  driver_file="/proc/driver/nvidia/version", staged_driver=STAGED_DRIVER, staged_sums=None,
                  nvgl=None, env_dir=None, build=None, lvp_icd=None, environ=None) -> dict:
    """v00 result: debug.facts.collect against the kernel driver found on this node, plus the recheck's facts."""
    kd = parse_kernel_driver(_read(driver_file))
    res = dfacts.collect(out, sys_lib=sys_lib, vk_dirs=list(vk_dirs), egl_vendor_dirs=list(egl_vendor_dirs),
                         dev_dir=dev_dir, driver=kd or staged_driver, driver_file=driver_file, nvgl=nvgl,
                         env_dir=env_dir, build=build, lvp_icd=lvp_icd, o3d_py=None, environ=environ)
    m = res["metrics"]
    m["kernel_driver"] = kd
    m["staged_driver"] = staged_driver
    m["staged_driver_match"] = kd == staged_driver
    m["driver_match"] = m["staged_driver_match"]  # debug.facts' meaning: the kernel matches the staged libs
    m["lib_driver"] = kd or staged_driver  # the version the system library names and caps_honored were keyed on
    if kd is None:
        res["notes"].append(f"no kernel driver version in {driver_file}; library names keyed on {staged_driver}")
    smi = dfacts.parse_smi_q(_read(os.path.join(out, "nvidia_smi_q.txt")) or "")
    m["gpu"] = {k: smi.get(v) for k, v in SMI_GPU.items()}
    m["hostname"] = socket.gethostname()
    sums = read_sums(staged_sums) if staged_sums else {}
    m["staged_sums"] = {"path": staged_sums, "entries": len(sums)}
    cmp = system_vs_staged(sys_lib, sums)
    m["system_vs_staged"] = cmp
    m["system_equals_staged"] = {k: v.get("equal") for k, v in cmp.items()}
    m["system_equals_staged_all"] = all_equal(cmp)
    m["ldconfig_gfx"] = parse_ldconfig(_read(os.path.join(out, "ldconfig_gfx.txt")))
    m["system_icd_json"] = system_icd(vk_dirs)
    res["stage"] = "done"
    return res


# --- manifests ----------------------------------------------------------------------------------------------

def _write_manifest(path: str, doc: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")


def write_icd_json(path: str, lib: str, src: str | None = None) -> dict:
    """A Vulkan ICD manifest for `lib` (absolute), file_format_version / api_version from the manifest `src`."""
    if not os.path.isabs(lib):
        raise ValueError(f"library_path must be absolute: {lib}")
    ffv, api = ICD_FALLBACK["file_format_version"], ICD_FALLBACK["api_version"]
    doc = common.read_json(src) if src else None
    icd = doc.get("ICD") if isinstance(doc, dict) else None
    if isinstance(icd, dict):
        ffv = str(doc.get("file_format_version") or ffv)
        api = str(icd.get("api_version") or api)
    out = {"file_format_version": ffv, "ICD": {"library_path": lib, "api_version": api}}
    _write_manifest(path, out)
    return out


def write_egl_json(path: str, lib: str) -> dict:
    """A glvnd EGL vendor manifest for `lib` (absolute)."""
    if not os.path.isabs(lib):
        raise ValueError(f"library_path must be absolute: {lib}")
    out = {"file_format_version": EGL_VENDOR_FORMAT, "ICD": {"library_path": lib}}
    _write_manifest(path, out)
    return out


# --- v13 ----------------------------------------------------------------------------------------------------

def thor_check(d: str, want: str = WANT_RENDERER, nvgl: str | None = None) -> dict:
    """Apply the renderer half of the v13 rule to d/result.json (pass = frames pass AND renderer named)."""
    p = os.path.join(d, "result.json")
    res = common.read_json(p)
    if not isinstance(res, dict):
        res = common.new_result(os.path.basename(d.rstrip("/")))
        res["notes"].append("no result.json from the probe (killed before it wrote one?)")
    m = res.setdefault("metrics", {})
    logs = [f for f in ("Player_after_exit.log", "Player.log") if os.path.isfile(os.path.join(d, f))]
    plog = os.path.join(d, logs[0]) if logs else None
    text = _read(plog) or ""
    renderers = RENDERER_RE.findall(text)
    renderer_ok = any(want in r for r in renderers)
    frames_pass = bool(res.get("pass"))
    maps = m.get("unity_maps_after_steps") or m.get("unity_maps_after_startup") or []
    glx = [x for x in maps if os.path.basename(x).startswith("libGLX_nvidia")]
    root = (nvgl.rstrip("/") + "/") if nvgl else None
    loader = (post.parse_loader_log(plog, os.path.join(d, "loader_excerpt_player.txt")) if plog
              else {"present": False})
    m["vkre"] = {"player_log": logs[0] if logs else None, "want_renderer": want, "renderers": renderers[:5],
                 "renderer_ok": renderer_ok, "vulkan_api": VK_API_RE.findall(text)[:3], "frames_pass": frames_pass,
                 "unity_glx": glx, "unity_glx_staged": bool(root) and any(x.startswith(root) for x in glx),
                 "loader": {k: loader.get(k) for k in ("present", "loader_lines", "manifests", "driver_libs",
                                                       "layers", "errors")}}
    res["pass_frames"] = frames_pass
    res["pass"] = bool(frames_pass and renderer_ok)
    if not renderer_ok:
        res["notes"].append(f"Player.log names no '{want}' Vulkan renderer (found {renderers[:3] or 'none'})")
    common.write_json_atomic(p, res)
    return res


# --- summary ------------------------------------------------------------------------------------------------

def variant_status(out: str, name: str, r) -> str:
    if os.path.isfile(os.path.join(out, name, "profile_error.txt")):
        return "ERROR"  # the environment profile could not be set up: a harness problem, not a GPU result
    if not isinstance(r, dict):
        return "MISSING"
    if r.get("skipped") or r.get("pass") is None:
        return "SKIP"
    return "PASS" if r.get("pass") else "FAIL"


def _vk(r) -> dict:
    return ((r or {}).get("metrics") or {}).get("vk") or {} if isinstance(r, dict) else {}


def key_metrics(out: str, name: str, r) -> dict:
    if not isinstance(r, dict):
        return {}
    kind, m = KIND[name], r.get("metrics") or {}
    if kind == "facts":
        return {k: m.get(k) for k in FACT_KEYS}
    if kind == "gl":
        return dsum.key_metrics("v01_gl_tri_R", r)
    if kind == "vk":
        k = dsum.key_metrics("v02_vk_lvp", r)
        vk = m.get("vk") or {}
        k.update(vkCreateInstance=vk.get("vkCreateInstance"), vkCreateDevice=vk.get("vkCreateDevice"),
                 submit_to_fence_ms=m.get("submit_to_fence_ms"), loader_mapped=m.get("loader_mapped"),
                 mapped_gpu_libs=[os.path.basename(x) for x in (m.get("mapped_gpu_libs") or [])], env=r.get("env"),
                 icd_json=common.read_json(os.path.join(out, name, "sys_icd.json")))
        lo = post.parse_loader_log(os.path.join(out, name, "log.txt"), os.path.join(out, name, "loader_excerpt.txt"))
        k["loader_log"] = {"present": lo.get("present"), "loader_lines": lo.get("loader_lines"),
                           "manifests": lo.get("manifests"), "driver_libs": lo.get("driver_libs"),
                           "errors": (lo.get("errors") or [])[:5]}
        return k
    k = dsum.key_metrics("v03_thor_diag_S", r)
    k.update(pass_frames=r.get("pass_frames"), vkre=m.get("vkre"))
    return k


def job_pool(out: str, environ) -> tuple:
    """The pool this job was prepared for: RUN/job/prepared.json next to OUT (= RUN/out), else VPD_VKRE_POOL."""
    pj = common.read_json(os.path.join(os.path.dirname(os.path.abspath(out)), "job", "prepared.json"))
    if isinstance(pj, dict) and pj.get("pool"):
        return pj["pool"], "prepared.json"
    if environ.get("VPD_VKRE_POOL"):
        return environ["VPD_VKRE_POOL"], "env VPD_VKRE_POOL"
    return None, None


def decide(out: str, results: dict) -> dict:
    st = {n: variant_status(out, n, results.get(n)) for n in VARIANTS}
    facts_m = ((results.get("v00_facts") or {}).get("metrics") or {}) if isinstance(results.get("v00_facts"),
                                                                                    dict) else {}
    tested = [n for n in NV_VK + [VERDICT_VARIANT] if st[n] in ("PASS", "FAIL")]
    ci11, ci12 = _vk(results.get("v11_vk_sysnew")).get("vkCreateInstance"), \
        _vk(results.get("v12_vk_sysnew_noglvnd")).get("vkCreateInstance")
    glvnd = None
    if st["v11_vk_sysnew"] in ("PASS", "FAIL") and st["v12_vk_sysnew_noglvnd"] in ("PASS", "FAIL"):
        glvnd = bool(str(ci11 or "").startswith("VK_SUCCESS")
                     and str(ci12 or "").startswith("VK_ERROR_INCOMPATIBLE_DRIVER"))
    gpu = facts_m.get("gpu") or {}
    v13 = results.get(VERDICT_VARIANT) if isinstance(results.get(VERDICT_VARIANT), dict) else {}
    vk13 = (v13.get("metrics") or {}).get("vkre") or {}
    flags = {"nv_vk_pass": any(st[n] == "PASS" for n in NV_VK),
             "thor_sys_pass": {"PASS": True, "FAIL": False}.get(st[VERDICT_VARIANT]),
             "thor_sys_frames_pass": v13.get("pass_frames"), "thor_sys_renderer_ok": vk13.get("renderer_ok"),
             "thor_sys_renderers": vk13.get("renderers"), "thor_sys_glx_staged": vk13.get("unity_glx_staged"),
             "glvnd_explains_R": glvnd, "v11_vkCreateInstance": ci11, "v12_vkCreateInstance": ci12,
             "nv_tested": tested, "nv_all_fail": bool(tested) and all(st[n] == "FAIL" for n in tested),
             "lvp_ok": st["v02_vk_lvp"] == "PASS", "gl_ok": {"PASS": True, "FAIL": False}.get(st["v01_gl_tri_R"]),
             "caps_honored": facts_m.get("caps_honored"), "kernel_driver": facts_m.get("kernel_driver"),
             "staged_driver_match": facts_m.get("staged_driver_match"),
             "system_equals_staged_all": facts_m.get("system_equals_staged_all"), "gpu_uuid": gpu.get("uuid"),
             "pci_bus_id": gpu.get("pci_bus_id"), "vbios": gpu.get("vbios"), "hostname": facts_m.get("hostname")}
    s13 = st[VERDICT_VARIANT]
    if s13 == "PASS":
        verdict, why = "PASS", f"{VERDICT_VARIANT} PASS: frames render and Player.log names {vk13.get('renderers')}"
    elif s13 == "FAIL":
        verdict = "FAIL"
        why = (f"{VERDICT_VARIANT} FAIL: frames_pass={v13.get('pass_frames')}, "
               f"renderer_ok={vk13.get('renderer_ok')} (renderers {vk13.get('renderers')})")
    else:
        verdict = "NOT_TESTED"
        why = f"{VERDICT_VARIANT} {s13}: " + str((v13 or {}).get("skipped") or (v13 or {}).get("notes") or "")[:300]
    rows = []
    if verdict == "PASS":
        rows.append({"reading": "NVIDIA Vulkan renders AI2-THOR frames through the system driver stack (env loader "
                                "1.4.357, absolute ICD / EGL JSON) on this card",
                     "next": "DECISIONS G2: run the frozen TVR smoke (SMOKE_CRITERIA TVR_GATE) on this pool with the "
                             "sysnew profile; the pilot goes back to GPU rendering only if it passes before the TVR "
                             "demo replay starts, otherwise lavapipe stays frozen"})
    if flags["nv_vk_pass"] and verdict == "FAIL":
        rows.append({"reading": "an NVIDIA Vulkan triangle works but AI2-THOR does not: a Unity-specific problem",
                     "next": "report only; do not pursue (DECISIONS G2)"})
    if flags["nv_all_fail"]:
        rows.append({"reading": "every NVIDIA Vulkan variant tested on this card failed",
                     "next": "if the other pool's job fails too, NVIDIA Vulkan is closed for round 3 and lavapipe "
                             "stays (DECISIONS G2)"})
    if glvnd is True:
        rows.append({"reading": "v11 creates an instance and v12 (no glvnd) gets VK_ERROR_INCOMPATIBLE_DRIVER: the "
                                "image's missing glvnd libEGL.so.1, not loader 1.2.131, explains v02_vk_nvR of the "
                                "11-09 debug job",
                     "next": "report only"})
    if st["v02_vk_lvp"] in ("FAIL", "ERROR"):
        rows.append({"reading": "the lavapipe control failed: the probe or the TVR env is broken on this node, so "
                                "the NVIDIA results say little", "next": "read v02_vk_lvp/log.txt"})
    if verdict == "NOT_TESTED":
        rows.append({"reading": f"the AI2-THOR check did not run: {why}",
                     "next": "no recheck verdict for this pool; read v00_facts (caps_honored, kernel_driver)"})
    return {"status": st, "flags": flags, "verdict": verdict, "verdict_reason": why, "readings": rows}


def summarize(out: str, environ=None) -> dict:
    environ = os.environ if environ is None else environ
    results = {n: common.read_json(os.path.join(out, n, "result.json")) for n in VARIANTS}
    d = decide(out, results)
    pool, pool_src = job_pool(out, environ)
    wt = environ.get("VPD_WORKTREE")
    s = {"utc": common.utc(), "out": out,
         "status": "EXPLORATORY (gate tvr; DECISIONS G2 NVIDIA Vulkan recheck); never feeds the frozen TVR smoke "
                   "verdict",
         "pool": pool, "pool_source": pool_src, "verdict": d["verdict"], "verdict_reason": d["verdict_reason"],
         "pass_rule": PASS_RULE, "variants_status": d["status"],
         "variants": {n: ({"status": d["status"][n], "skipped": r.get("skipped"), "rc": r.get("rc"),
                           "secs": r.get("secs"), "cap_s": r.get("cap_s"), "timed_out": r.get("timed_out"),
                           "stage": r.get("stage"), "notes": (r.get("notes") or [])[:6],
                           "profile_error": _read(os.path.join(out, n, "profile_error.txt")),
                           "key": key_metrics(out, n, r)}
                          if isinstance(r, dict) else {"status": d["status"][n], "missing": True})
                      for n, r in results.items()},
         "flags": d["flags"], "readings": d["readings"],
         "code_snapshot": common.read_json(os.path.join(wt, "SNAPSHOT.json")) if wt else None,
         "overrides": {k: v for k, v in sorted(environ.items()) if k.startswith("VPD_VKRE_")}}
    common.write_json_atomic(os.path.join(out, "summary.json"), s)
    here = os.path.abspath(__file__)
    common.write_json_atomic(os.path.join(out, "expect.json"), {
        "files": EXPECT_FILES, "steps": VARIANTS + ["summary"],
        "check": f"python -I {here} expect --out {out}  (stdlib only; exit 0 iff every output is present)"})
    return s


def expect(out: str) -> dict:
    """Every expected output present and parseable, every variant stepped (run or skipped), summary written."""
    problems = []
    for f in EXPECT_FILES:
        p = os.path.join(out, f)
        if not os.path.isfile(p):
            problems.append(f"missing {f}")
        elif f.endswith(".json") and not isinstance(common.read_json(p), dict):
            problems.append(f"unparseable {f}")
    steps = []
    if os.path.isfile(os.path.join(out, "steps.jsonl")):
        try:
            with open(os.path.join(out, "steps.jsonl")) as fh:
                steps = [json.loads(x)["name"] for x in fh if x.strip()]
        except (OSError, ValueError, KeyError, TypeError):
            problems.append("unparseable steps.jsonl")
    for n in VARIANTS:
        if n not in steps and f"{n}:skipped" not in steps:
            problems.append(f"no step {n}")
    if "summary" not in steps:
        problems.append("no step summary")
    s = common.read_json(os.path.join(out, "summary.json"))
    s = s if isinstance(s, dict) else {}
    if s and s.get("verdict") not in VERDICTS:
        problems.append("summary.json has no verdict")
    fl = s.get("flags") or {}
    return {"ok": not problems, "problems": problems, "out": out, "pool": s.get("pool"), "verdict": s.get("verdict"),
            "verdict_reason": s.get("verdict_reason"), "variants_status": s.get("variants_status"),
            "flags": {k: fl.get(k) for k in ("nv_vk_pass", "thor_sys_pass", "glvnd_explains_R", "nv_all_fail",
                                             "lvp_ok", "kernel_driver", "gpu_uuid", "hostname")}}


# --- command line -------------------------------------------------------------------------------------------

def cmd_facts(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    res = collect_facts(a.out, sys_lib=a.sys_lib, vk_dirs=a.vk_dirs.split(":"),
                        egl_vendor_dirs=a.egl_vendor_dirs.split(":"), dev_dir=a.dev_dir, driver_file=a.driver_file,
                        staged_driver=a.staged_driver, staged_sums=a.staged_sums, nvgl=a.nvgl, env_dir=a.env_dir,
                        build=a.build, lvp_icd=a.lvp_icd)
    common.write_json_atomic(os.path.join(a.out, "result.json"), res)
    print(json.dumps(res, indent=1, default=str)[:6000], flush=True)
    return 0


def cmd_icd_json(a) -> int:
    print(json.dumps(write_icd_json(a.out, a.lib, a.src)), flush=True)
    return 0


def cmd_egl_json(a) -> int:
    print(json.dumps(write_egl_json(a.out, a.lib)), flush=True)
    return 0


def cmd_thor_check(a) -> int:
    res = thor_check(a.dir, a.want_renderer, a.nvgl)
    print(json.dumps({"pass": res.get("pass"), "pass_frames": res.get("pass_frames"),
                      "vkre": res["metrics"].get("vkre")}, default=str)[:3000], flush=True)
    return 0


def cmd_summarize(a) -> int:
    s = summarize(a.out)
    print(json.dumps({k: s[k] for k in ("pool", "verdict", "verdict_reason", "variants_status", "flags")},
                     indent=1, default=str), flush=True)
    return 0


def cmd_expect(a) -> int:
    e = expect(a.out)
    print(json.dumps(e, default=str), flush=True)
    return 0 if e["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("facts")
    p.add_argument("--out", required=True)
    p.add_argument("--sys-lib", default="/usr/lib/x86_64-linux-gnu")
    p.add_argument("--vk-dirs", default="/etc/vulkan:/usr/share/vulkan")
    p.add_argument("--egl-vendor-dirs", default="/usr/share/glvnd/egl_vendor.d:/etc/glvnd/egl_vendor.d")
    p.add_argument("--dev-dir", default="/dev")
    p.add_argument("--driver-file", default="/proc/driver/nvidia/version")
    p.add_argument("--staged-driver", default=STAGED_DRIVER)
    p.add_argument("--staged-sums", help="sha256sum file of the staged NVIDIA libraries (nvgl_<drv>/SHA256SUMS)")
    p.add_argument("--nvgl")
    p.add_argument("--env-dir")
    p.add_argument("--build")
    p.add_argument("--lvp-icd")
    p.set_defaults(fn=cmd_facts)

    p = sub.add_parser("icd-json")
    p.add_argument("--out", required=True)
    p.add_argument("--lib", required=True, help="absolute path of the ICD library (libGLX_nvidia.so.0)")
    p.add_argument("--src", help="manifest to copy file_format_version / api_version from (the system one)")
    p.set_defaults(fn=cmd_icd_json)

    p = sub.add_parser("egl-json")
    p.add_argument("--out", required=True)
    p.add_argument("--lib", required=True, help="absolute path of the vendor library (libEGL_nvidia.so.0)")
    p.set_defaults(fn=cmd_egl_json)

    p = sub.add_parser("thor-check")
    p.add_argument("--dir", required=True)
    p.add_argument("--want-renderer", default=WANT_RENDERER)
    p.add_argument("--nvgl", help="the staged overlay root: a libGLX_nvidia mapped from under it is flagged")
    p.set_defaults(fn=cmd_thor_check)

    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("expect")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_expect)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
