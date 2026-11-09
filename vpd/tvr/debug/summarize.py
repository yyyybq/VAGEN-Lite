"""finalize (per variant) and summarize (OUT/summary.json) of the vpd-tvr-debug job (stdlib only).

VARIANTS is the run order of debug_job.sh (the tests check that the script runs exactly these, in this order).
A variant's state: True (PASS), False (FAIL, including a probe killed by its inner timeout), None (skipped / not
run).  The decision table of the job spec, evaluated on those states:
  (0) V3 PASS                                  the failing smoke config renders in this job: not reproduced
  (1) V1 PASS and V2 NV PASS, V3 black         GPU + server rasterise; Unity/ai2thor x our container/driver
  (2) V2 NV FAIL, V2 lavapipe PASS             the NVIDIA Vulkan path is broken in the container
  (3) V1 FAIL with clean compile, V2 NV FAIL   server level: H800 3D pipeline unusable from job containers
      (only when a system-stack R variant was tested; with the staged overlay alone it is row "3s": overlay or
      server, not distinguishable -- 2026-11-09 review)
  (4) V3 depth/seg vary, RGB black             the lit / post-processing path; V5 decides
  (5) V10 PASS                                 a CPU fallback exists (its speed is reported)
"V1 PASS" = any tested V1 sub-variant passed; "V2 NV PASS" = any tested NVIDIA V2 sub-variant passed;
"V2 NV FAIL" / "V1 FAIL" = every tested sub-variant failed (at least one tested).  "V3 black" = V3 produced
step frames and failed the RGB rule.  Rows are reported, never acted on.
"""
from __future__ import annotations

import os
import shutil

from . import common

VARIANTS = ["v00_facts", "v01_gl_tri_S", "v01_gl_tri_R", "v02_vk_nvS_img", "v02_vk_nvS_env", "v02_vk_nvR",
            "v02_vk_lvp", "v03_thor_diag_S", "v04_thor_R", "v05_thor_lowq", "v06_thor_newloader",
            "v07_thor_layers", "v08_thor_direct", "v09_open3d_egl", "v10_thor_lvp_forced"]
V1 = ["v01_gl_tri_S", "v01_gl_tri_R"]
V2_NV = ["v02_vk_nvS_img", "v02_vk_nvS_env", "v02_vk_nvR"]
FIXES = ["v04_thor_R", "v05_thor_lowq", "v06_thor_newloader", "v07_thor_layers", "v08_thor_direct"]
THOR = ["v03_thor_diag_S"] + FIXES + ["v10_thor_lvp_forced"]
VARY_FRAC = 0.9
TVRBENCH_MS = {"cpu_glx_xvfb": 49.6, "vulkan_h200d": 14.4}  # TVRBench thor_env.py:43-46 (H200D, 2026-05-18)
VIEW_FILES = ("rgb.png", "depth.png", "seg.png", "tri.png")
FIX_TEXT = {"v04_thor_R": "the system graphics stack (graphics capability) instead of our staged overlay",
            "v05_thor_lowq": "quality 'Very Low', no antiAliasing, 300x300",
            "v06_thor_newloader": "Vulkan loader 1.4.357 (envs/tvr/lib first) instead of the image's 1.2.131",
            "v07_thor_layers": "a valid VK_LAYER_NV_optimus implicit layer + __NV_PRIME_RENDER_OFFLOAD=1",
            "v08_thor_direct": "-force-gfx-direct (no render thread)"}


def finalize(d: str, name: str, rc=None, secs=None, cap=None, skipped=None) -> dict:
    """Add rc / secs / cap_s / timed_out to d/result.json; a skipped variant gets pass None + the reason."""
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "result.json")
    if skipped:
        res = common.new_result(name, stage="skipped")
        res.update(skipped=skipped)
        res["pass"] = None
        res["notes"].append(f"skipped: {skipped}")
    else:
        res = common.read_json(p)
        if not isinstance(res, dict):
            res = common.new_result(name)
            res["notes"].append("no result.json from the probe")
    res["name"] = name
    if rc is not None:
        res["rc"] = rc
        res["timed_out"] = rc in (124, 137)
        if res["timed_out"]:
            res["notes"].append(f"killed by the inner timeout ({cap} s), rc {rc}, at stage {res.get('stage')}")
    if secs is not None:
        res["secs"] = secs
    if cap is not None:
        res["cap_s"] = cap
    common.write_json_atomic(p, res)
    return res


def state(r):
    if not isinstance(r, dict) or r.get("skipped") or r.get("pass") is None:
        return None
    return bool(r.get("pass"))


def _g(r, *path):
    for k in path:
        r = r.get(k) if isinstance(r, dict) else None
    return r


def key_metrics(name: str, r) -> dict:
    if not isinstance(r, dict):
        return {}
    m = r.get("metrics") or {}
    if name == "v00_facts":
        return {k: m.get(k) for k in ("caps_honored", "caps_detail", "graphics_requested", "driver_match",
                                      "smi_q", "dev", "system_glvnd", "staging")}
    if name.startswith("v01"):
        return {"GL_RENDERER": m.get("GL_RENDERER"), "compile_ok": m.get("compile_ok"), "link_ok": m.get("link_ok"),
                "gl_errors": (m.get("gl_errors_after_draw") or []) + (m.get("gl_errors_after_readback") or []),
                "centre": _g(m, "pixels", "centre"), "corners_ok": _g(m, "pixels", "corners_ok"),
                "all_zero": _g(m, "pixels", "all_zero"), "stage": r.get("stage")}
    if name.startswith("v02"):
        vk = m.get("vk") or {}
        return {"device": _g(m, "device", "name"), "loader": m.get("loader_instance_version"),
                "vkCreateGraphicsPipelines": vk.get("vkCreateGraphicsPipelines"),
                "vkQueueSubmit": vk.get("vkQueueSubmit"), "vkWaitForFences": vk.get("vkWaitForFences"),
                "centre": _g(m, "pixels", "centre"), "corners_ok": _g(m, "pixels", "corners_ok"),
                "all_zero": _g(m, "pixels", "all_zero"), "stage": r.get("stage")}
    if name in THOR:
        fr = m.get("frames") or {}
        post = m.get("post") or {}
        return {"startup_s": m.get("startup_s"), "n_step_frames": fr.get("n_step_frames"),
                "frac_rgb_std_ge2": fr.get("frac_rgb_std_ge2"), "rgb_max_overall": fr.get("rgb_max_overall"),
                "rgb_all_zero_frames": fr.get("rgb_all_zero_frames"), "frac_depth_varies": fr.get("frac_depth_varies"),
                "frac_seg_varies": fr.get("frac_seg_varies"), "ms_per_step": m.get("ms_per_step"),
                "steps_per_s": m.get("steps_per_s"), "fatal": r.get("fatal"), "error": r.get("error"),
                "never_loaded_gpu": _g(post, "lddebug", "never_loaded_gpu"),
                "nvcache_files": _g(post, "nvcache", "files"), "gpu_util_max": _g(post, "gpu", "util_max"),
                "clock_gr_max_mhz": _g(post, "gpu", "clock_gr_max_mhz"), "mem_rise_mib": _g(post, "gpu", "mem_rise_mib"),
                "loader_manifests": _g(post, "loader", "manifests"), "loader_layers": _g(post, "loader", "layers"),
                "loader_errors": (_g(post, "loader", "errors") or [])[:5],
                "unity_gpu_libs": [os.path.basename(p) for p in (m.get("unity_maps_after_steps")
                                                                  or m.get("unity_maps_after_startup") or [])]}
    if name == "v09_open3d_egl":
        return {k: m.get(k) for k in ("open3d_version", "std", "centre", "frac_not_background", "render_ms")}
    return {}


def decide(results: dict) -> dict:
    st = {n: state(results.get(n)) for n in VARIANTS}
    caps = bool(_g(results.get("v00_facts"), "metrics", "caps_honored"))
    v1_tested = [n for n in V1 if st[n] is not None]
    nv_tested = [n for n in V2_NV if st[n] is not None]
    v1_pass = any(st[n] for n in v1_tested)
    v1_clean_fail = bool(v1_tested) and all(
        st[n] is False and _g(results[n], "metrics", "compile_ok") and _g(results[n], "metrics", "link_ok")
        for n in v1_tested)
    nv_pass = any(st[n] for n in nv_tested)
    r_tested = [n for n in ("v01_gl_tri_R", "v02_vk_nvR") if st[n] is not None]
    nv_fail = bool(nv_tested) and not nv_pass
    lvp_pass = st["v02_vk_lvp"] is True
    fr3 = _g(results.get("v03_thor_diag_S"), "metrics", "frames") or {}
    v3_ran = (fr3.get("n_step_frames") or 0) >= 1
    v3_black = v3_ran and st["v03_thor_diag_S"] is False
    v3_aux = ((fr3.get("frac_depth_varies") or 0) >= VARY_FRAC) or ((fr3.get("frac_seg_varies") or 0) >= VARY_FRAC)
    fixes = [n for n in FIXES if st[n] is True]
    flags = {"caps_honored": caps, "v1_tested": v1_tested, "v1_pass": v1_pass, "v1_fail_clean_compile": v1_clean_fail,
             "r_tested": r_tested, "v2_nv_tested": nv_tested, "v2_nv_pass": nv_pass, "v2_nv_fail": nv_fail, "v2_lvp_pass": lvp_pass,
             "v3_ran": v3_ran, "v3_black": v3_black, "v3_depth_or_seg_vary": v3_aux, "fix_variants": fixes,
             "v4": st["v04_thor_R"], "v5": st["v05_thor_lowq"], "v10": st["v10_thor_lvp_forced"]}
    rows = []
    if st["v03_thor_diag_S"] is True:
        rows.append({"row": 0, "reading": "V3 (the failing smoke config, FloorPlan1 640x360 Ultra AA4, staged "
                     "overlay) rendered in this job: the black frames did not reproduce",
                     "next": "compare node / driver / env with smoke_run1, then rerun the frozen TVR smoke"})
    if v1_pass and nv_pass and v3_black:
        if fixes:
            nxt = ("the passing variant is the fix; rerun the frozen TVR smoke with it: "
                   + "; ".join(f"{n} = {FIX_TEXT[n]}" for n in fixes))
        else:
            nxt = ("no V4-V8 variant fixed it: Unity/ai2thor x our container/driver remains; the follow-ups "
                   "(Linux64 build + Xvfb + Mesa GLX, or VirtualGL EGL) are downloads: stop-and-ask")
        rows.append({"row": 1, "reading": "GPU and server rasterise fine; the problem is Unity/ai2thor x our "
                     "container/driver", "next": nxt})
    if nv_fail and lvp_pass:
        if st["v04_thor_R"] is True:
            nxt = "request graphics capability per job (NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)"
        else:
            nxt = ("admin question (a): can job containers get the NVIDIA graphics capability / a working Vulkan "
                   "ICD?" + ("" if caps else " (this job's request for it was not honoured or not made)"))
        rows.append({"row": 2, "reading": "the NVIDIA Vulkan path is broken in the container", "next": nxt})
    if v1_clean_fail and nv_fail:
        if r_tested:
            rows.append({"row": 3, "reading": "server level: the H800 3D pipeline is not usable from job "
                         "containers (system graphics stack tested too); EGL tasks (Habitat) are blocked too",
                         "next": "only CPU rendering (V10) remains, or another pool if the admin says one differs"})
        else:
            rows.append({"row": "3s", "reading": "every GL / NVIDIA-Vulkan triangle failed, but only on our staged "
                         "595 overlay (no system-stack R variant ran): staged overlay broken OR server level, "
                         "not distinguishable",
                         "next": "rerun with the graphics capability honoured (sco --env "
                                 "NVIDIA_DRIVER_CAPABILITIES:all) or ask the admin question (a)"})
    if v3_black and v3_aux:
        v5 = st["v05_thor_lowq"]
        rows.append({"row": 4, "reading": "V3 depth / segmentation vary but RGB stays black: the lit / "
                     "post-processing path is at fault",
                     "next": "V5 decides: " + {True: "V5 PASS, so Ultra quality / MSAA / shadows / post-FX break it",
                                               False: "V5 FAIL, so the lit path breaks even at 'Very Low'",
                                               None: "V5 did not run"}[v5]})
    if st["v10_thor_lvp_forced"] is True:
        m10 = _g(results.get("v10_thor_lvp_forced"), "metrics") or {}
        rows.append({"row": 5, "reading": "a CPU fallback exists (lavapipe, forced device index 0)",
                     "next": f"speed: startup {m10.get('startup_s')} s, {m10.get('ms_per_step')} ms/step at 300x300 "
                             f"'Very Low' (TVRBench: {TVRBENCH_MS['cpu_glx_xvfb']} ms/step CPU GLX, "
                             f"{TVRBENCH_MS['vulkan_h200d']} ms/step Vulkan on H200D)"})
    if not rows:
        rows.append({"row": None, "reading": "no decision-table row matched", "next": "read the variant table"})
    return {"flags": flags, "matched": rows}


def collect_views(out: str) -> dict:
    view = os.path.join(out, "view")
    got = {}
    for n in VARIANTS:
        for f in VIEW_FILES:
            src = os.path.join(out, n, f)
            if os.path.isfile(src):
                os.makedirs(view, exist_ok=True)
                dst = os.path.join(view, f"{n}__{f}")
                shutil.copyfile(src, dst)
                got.setdefault(n, []).append(os.path.relpath(dst, out))
    return got


def summarize(out: str) -> dict:
    results = {n: common.read_json(os.path.join(out, n, "result.json")) for n in VARIANTS}
    d = decide(results)
    s = {"utc": common.utc(), "out": out, "status": "EXPLORATORY (gate tvr); never feeds the frozen TVR smoke verdict",
         "variants_pass": {n: state(results[n]) for n in VARIANTS},
         "variants": {n: {"pass": state(r), "skipped": (r or {}).get("skipped"), "rc": (r or {}).get("rc"),
                          "secs": (r or {}).get("secs"), "stage": (r or {}).get("stage"),
                          "notes": ((r or {}).get("notes") or [])[:6], "key": key_metrics(n, r)}
                      if isinstance(r, dict) else {"pass": None, "missing": True}
                      for n, r in results.items()},
         "caps_honored": d["flags"]["caps_honored"], "flags": d["flags"], "matched": d["matched"],
         "view": collect_views(out)}
    common.write_json_atomic(os.path.join(out, "summary.json"), s)
    return s
