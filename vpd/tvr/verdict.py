"""TVR render smoke: apply vpd/tvr/SMOKE_CRITERIA.md to one smoke OUTDIR and write OUTDIR/verdict.json.

Stdlib only (runs under the tvr env's python 3.10 inside the job, and under PY_EVAL in the tests).

    python -I verdict.py verdict OUTDIR            # -> OUTDIR/verdict.json {TVR_GATE, EGL_GATE, reasons, ...}
    python -I verdict.py mapping OUTDIR HOME       # cuda-vulkan-mapping decision (step 2 of tvr_smoke.sh)

Inputs read from OUTDIR (all written by tvr_smoke.sh; any may be missing, which counts as a failed criterion):
    driver_version.txt     copy of /proc/driver/nvidia/version
    facts.json             /dev/nvidia-modeset, /dev/dri, env facts (step 0)
    staging.json           which staged inputs were present (step 0)
    nvidia_smi_L.txt       `nvidia-smi -L`
    vulkaninfo_summary.txt `vulkaninfo --summary`;  vulkaninfo_full.txt  `vulkaninfo`
    thor1/result.json, thor1/Player*.log, thor4/result.json, egl/result.json, thor_lvp/result.json (optional)

The thresholds below are the frozen ones of SMOKE_CRITERIA.md (plan section 3.7); the test suite checks that the
two files agree.
"""

from __future__ import annotations

import json
import os
import re
import sys

# --- frozen criteria (SMOKE_CRITERIA.md) ----------------------------------------------------------------------
DRIVER = "595.71.05"
GPU_NAME = "H800"
DISCRETE = "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
STEPS = 300  # phase A steps of the single-process run
STARTUP_MAX_S = 300.0  # Controller start-up, strictly below
DISTINCT_MIN = 0.90  # distinct frames / steps, at least
SUCCESS_MIN = 0.95  # lastActionSuccess / steps, at least
NPROC = 4  # parallel processes on the one card
NPROC_STEPS = 150  # steps per parallel process
EGL_RGBA = (51, 102, 153, 255)  # glClearColor(0.2, 0.4, 0.6, 1.0) read back as RGBA8
EGL_TOL = 2
SOFTWARE_RENDERERS = ("llvmpipe", "lavapipe", "swiftshader", "softpipe")


# --- small readers --------------------------------------------------------------------------------------------

def _read(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _load(path):
    """JSON file -> dict; a missing or broken file -> {'_missing': reason}."""
    text = _read(path)
    if text is None:
        return {"_missing": f"no file {path}"}
    try:
        v = json.loads(text)
    except ValueError as e:
        return {"_missing": f"bad json {path}: {e}"}
    return v if isinstance(v, dict) else {"_missing": f"not an object: {path}"}


# --- parsers (pure) -------------------------------------------------------------------------------------------

def driver_ok(version_text):
    """True if /proc/driver/nvidia/version names exactly DRIVER (as a whole token)."""
    if not version_text:
        return False
    return re.search(r"(?<![0-9.])" + re.escape(DRIVER) + r"(?![0-9.])", version_text) is not None


def parse_vulkaninfo_summary(text):
    """`vulkaninfo --summary` -> [{'index': n, 'deviceName': ..., 'deviceType': ..., 'deviceUUID': ...}, ...].

    The 1.3/1.4 summary prints a `Devices:` section with `GPU<n>:` headers followed by `key = value` lines.
    """
    devs, cur = [], None
    for line in (text or "").splitlines():
        m = re.match(r"\s*GPU([0-9]+):\s*$", line)
        if m:
            cur = {"index": int(m.group(1))}
            devs.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m:
            cur.setdefault(m.group(1), m.group(2))
        elif line.strip() and not line.startswith((" ", "\t")):
            cur = None  # a new top-level section ends the device block
    return devs


def parse_vulkaninfo_ai2thor(text):
    """deviceUUID -> Vulkan GPU index, exactly as ai2thor 5.0.0 Controller.unity_command parses full vulkaninfo.

    Returns (mapping, ok): ok is False when ai2thor's parser would attach a UUID to no GPU index (None) or
    trip its consistency assert.
    """
    current, out, ok = None, {}, True
    for line in (text or "").splitlines():
        m = re.match("GPU([0-9]+):", line)
        if m is not None:
            current = int(m.group(1))
        elif "deviceUUID" in line:
            parts = line.split("=")
            if len(parts) < 2:
                ok = False
                continue
            uuid = parts[1].strip()
            if uuid in out:
                if out[uuid] != current:
                    ok = False
            else:
                out[uuid] = current
    if any(v is None for v in out.values()):
        ok = False
    return out, ok


def parse_smi_L(text):
    """`nvidia-smi -L` -> {cuda index: uuid without the 'GPU-' prefix} (ai2thor's regexes)."""
    out = {}
    for line in (text or "").splitlines():
        m = re.match("GPU ([0-9]+):", line)
        if m is None:
            continue
        u = re.match(".*\\(UUID: GPU-([^)]+)\\)", line)
        if u:
            out[int(m.group(1))] = u.group(1)
    return out


def cuda_vulkan_mapping(full_text, smi_text, summary_text=""):
    """Decide how ai2thor gets its cuda->vulkan device map.

    Returns {'mode': 'ai2thor_native'|'fallback'|'impossible', 'mapping': {...} or None, 'notes': [...]}.
    - ai2thor_native: ai2thor's own parse of full vulkaninfo maps every nvidia-smi UUID; nothing is written.
    - fallback: one Vulkan device and every nvidia-smi index maps to Vulkan device 0 (a 1-card job).
    - impossible: no nvidia-smi GPU, or no / several Vulkan devices without a usable UUID match.
    """
    notes = []
    smi = parse_smi_L(smi_text)
    if not smi:
        return {"mode": "impossible", "mapping": None, "notes": ["nvidia-smi -L lists no GPU with a UUID"]}
    by_uuid, ok = parse_vulkaninfo_ai2thor(full_text)
    if ok and by_uuid and all(u in by_uuid for u in smi.values()):
        return {"mode": "ai2thor_native", "mapping": {str(k): by_uuid[u] for k, u in sorted(smi.items())},
                "notes": notes}
    notes.append(f"ai2thor parse ok={ok} uuids={sorted(by_uuid)} smi={smi}")
    n_vk = len(parse_vulkaninfo_summary(summary_text)) or len({v for v in by_uuid.values() if v is not None})
    if n_vk == 1:
        return {"mode": "fallback", "mapping": {str(k): 0 for k in sorted(smi)}, "notes": notes}
    notes.append(f"{n_vk} Vulkan devices; no safe fallback")
    return {"mode": "impossible", "mapping": None, "notes": notes}


def player_log_check(texts):
    """Unity Player.log(s) -> {'ok', 'renderer_lines', 'vulkan_lines', 'software_lines'}.

    ok (plan 3.7 "names the H800 Vulkan renderer"): some renderer/device line names GPU_NAME, some
    renderer/device line names Vulkan (not necessarily the same line: Unity's exact wording is unverified), and
    no renderer/device line names a software rasteriser.
    """
    pat = re.compile(r"(?i)(renderer|physical device|selectphysicaldevice|gfxdevice|vulkan api|vulkan init|"
                     r"graphics device)")
    renderer, software = [], []
    for t in texts:
        for line in (t or "").splitlines():
            if not pat.search(line):
                continue
            s = line.strip()[:300]
            renderer.append(s)
            if any(w in s.lower() for w in SOFTWARE_RENDERERS):
                software.append(s)
    vulkan = [s for s in renderer if "vulkan" in s.lower()]
    ok = any(GPU_NAME in s for s in renderer) and bool(vulkan) and not software
    return {"ok": ok, "renderer_lines": renderer[:20], "vulkan_lines": vulkan[:5], "software_lines": software[:20]}


# --- gates ----------------------------------------------------------------------------------------------------

def _crit(rows, name, ok, detail):
    rows.append({"name": name, "ok": bool(ok), "detail": detail})


def tvr_gate(out):
    """Criteria rows of TVR_GATE plus caveats and speed numbers."""
    rows, caveats = [], []
    drv = _read(os.path.join(out, "driver_version.txt"))
    _crit(rows, "T1_driver", driver_ok(drv), (drv or "missing").strip()[:200])

    staging = _load(os.path.join(out, "staging.json"))
    missing = [k for k, v in staging.items() if k != "_missing" and v is False]
    _crit(rows, "T0_staging", "_missing" not in staging and not missing,
          staging.get("_missing") or (f"missing: {missing}" if missing else "all staged inputs present"))

    smi = parse_smi_L(_read(os.path.join(out, "nvidia_smi_L.txt")))
    summ = _read(os.path.join(out, "vulkaninfo_summary.txt")) or ""
    full = _read(os.path.join(out, "vulkaninfo_full.txt")) or ""
    devs = parse_vulkaninfo_summary(summ)
    by_uuid, _ = parse_vulkaninfo_ai2thor(full)
    smi_uuids = {u.lower() for u in smi.values()}
    good = []
    for d in devs:
        uuid = (d.get("deviceUUID") or "").lower()
        if not uuid:  # older summaries: take the UUID ai2thor's parser saw for this GPU index
            uuid = next((u.lower() for u, i in by_uuid.items() if i == d["index"]), "")
        if GPU_NAME in d.get("deviceName", "") and d.get("deviceType") == DISCRETE and uuid in smi_uuids:
            good.append({"index": d["index"], "deviceName": d.get("deviceName"), "deviceUUID": uuid,
                         "driverInfo": d.get("driverInfo")})
    _crit(rows, "T2_vulkan_device", bool(good),
          {"matching": good, "devices": [{k: d.get(k) for k in ("index", "deviceName", "deviceType",
                                                                  "deviceUUID", "driverInfo")} for d in devs],
           "smi": smi})
    if good and not any(g.get("driverInfo") == DRIVER for g in good):
        caveats.append(f"vulkan driverInfo is not {DRIVER}: {[g.get('driverInfo') for g in good]}")

    t1 = _load(os.path.join(out, "thor1", "result.json"))
    raw = t1.get("raw") if isinstance(t1.get("raw"), dict) else {"error": t1.get("_missing", "no raw phase")}
    raw_err = raw.get("error")
    # thor_smoke checkpoints phase A: 'started' -> 'stepping' -> 'steps_done' -> 'done'; the gated counts exist
    # only from 'steps_done' on, so a run killed or failed mid-way fails T4/T5 and says how far it got
    phase = raw.get("phase")
    where = (f" (phase={phase}, progress={raw.get('progress')})" if phase in ("started", "stepping") else "")
    st = raw.get("startup_s")
    # T3 is the start-up alone: a start-up that was measured counts even if a later step raised
    _crit(rows, "T3_startup", isinstance(st, (int, float)) and st < STARTUP_MAX_S,
          f"startup_s={st}" + (f" error={raw_err}" if raw_err else ""))
    _crit(rows, "T4_bad_frames", raw_err is None and raw.get("bad_frames") == 0,
          raw_err or f"bad_frames={raw.get('bad_frames')}{where}")
    n = raw.get("steps")
    dist, succ = raw.get("distinct_frames"), raw.get("action_success")
    ok5 = (raw_err is None and n == STEPS and isinstance(dist, int) and isinstance(succ, int)
           and dist >= DISTINCT_MIN * STEPS and succ >= SUCCESS_MIN * STEPS)
    _crit(rows, "T5_frames_actions", ok5, raw_err or f"steps={n} distinct={dist} success={succ}{where}")
    for k in ("hd_error", "stop_error"):
        if raw.get(k):
            caveats.append(f"thor1 {k} (reported only): {raw[k]}")

    logs = []
    tdir = os.path.join(out, "thor1")
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if f.startswith("Player") and f.endswith(".log"):
                logs.append(_read(os.path.join(tdir, f)))
    pl = player_log_check(logs)
    _crit(rows, "T6_player_log", bool(logs) and pl["ok"], pl if logs else "no thor1/Player*.log")

    tasks = t1.get("tasks") if isinstance(t1.get("tasks"), list) else []
    ithor = [t for t in tasks if t.get("dataset") == "ithor"]
    proc = [t for t in tasks if t.get("dataset") == "procthor"]
    ithor_ok = len(ithor) == 1 and bool(ithor[0].get("ok"))
    if len(proc) == 1 and proc[0].get("ok"):
        proc_ok = True
    elif len(proc) == 1 and proc[0].get("skipped") == "no_procthor_data":
        proc_ok = True  # plan 3.7 / open_q: phase B then runs the iTHOR task only, and that is recorded
        caveats.append("procthor_not_tested: data/procthor-10k/train.jsonl.gz not staged (deferred to the TVR "
                       "pilot)")
    else:
        proc_ok = False
    # a phase B skipped for time_budget has no ithor row, so T7 fails (the tasks were not tested)
    _crit(rows, "T7_eval_tasks", ithor_ok and proc_ok,
          [{k: t.get(k) for k in ("task_id", "dataset", "ok", "skipped", "error", "distinct_start_target", "left_s")
            if k in t} for t in tasks] or "no task rows")

    t4 = _load(os.path.join(out, "thor4", "result.json"))
    procs = t4.get("procs") if isinstance(t4.get("procs"), list) else []
    ok8 = (len(procs) == NPROC and all("error" not in p for p in procs)
           and all(p.get("bad_frames") == 0 and p.get("steps") == NPROC_STEPS for p in procs))
    _crit(rows, "T8_parallel", ok8, t4.get("_missing") or [
        {k: p.get(k) for k in ("error", "steps", "bad_frames", "steps_per_s", "startup_s")} for p in procs])

    lvp = _load(os.path.join(out, "thor_lvp", "result.json"))
    speed = {"single_640x360": raw.get("steps_per_s"), "single_1280x720": raw.get("hd_1280x720_steps_per_s"),
             "nproc4_aggregate_640x360": t4.get("agg_steps_per_s"),
             "lavapipe_640x360": (lvp.get("raw") or {}).get("steps_per_s") if isinstance(lvp.get("raw"), dict)
             else None}
    return rows, caveats, speed


def egl_gate(out):
    rows = []
    drv = _read(os.path.join(out, "driver_version.txt"))
    _crit(rows, "E0_driver", driver_ok(drv), (drv or "missing").strip()[:200])
    e = _load(os.path.join(out, "egl", "result.json"))
    miss = e.get("_missing")
    n = e.get("num_devices")
    _crit(rows, "E1_devices", isinstance(n, int) and n >= 1, miss or f"num_devices={n} stage={e.get('stage')}")
    _crit(rows, "E2_initialize", bool(e.get("egl_version")), miss or f"egl_version={e.get('egl_version')} "
                                                                     f"egl_error={e.get('egl_error')}")
    _crit(rows, "E3_vendor", "NVIDIA" in str(e.get("EGL_VENDOR", "")), f"EGL_VENDOR={e.get('EGL_VENDOR')}")
    _crit(rows, "E4_renderer", GPU_NAME in str(e.get("GL_RENDERER", "")), f"GL_RENDERER={e.get('GL_RENDERER')}")
    px = e.get("readback_px0")
    ok5 = (isinstance(px, list) and len(px) == 4
           and all(isinstance(x, int) and abs(x - y) <= EGL_TOL for x, y in zip(px, EGL_RGBA)))
    _crit(rows, "E5_readback", ok5, f"px0={px} want {list(EGL_RGBA)} +-{EGL_TOL}")
    return rows


def next_step(tvr_pass, egl_pass, rows, facts):
    """The SMOKE_CRITERIA.md decision tree, as one line for the report (never acts by itself)."""
    if tvr_pass and egl_pass:
        return "TVR and EGL pass: the TVR pilot may be planned (new approval and budget); EGL renderers reopen"
    if tvr_pass:
        return "TVR pass, EGL fail: TVR goes ahead; Habitat / EGL-based benchmarks stay closed"
    by = {r["name"]: r for r in rows}
    if not by.get("T1_driver", {}).get("ok", False):
        return "driver mismatch on the job node: the copied 595.71.05 overlay cannot work there; ASK USER"
    if not by.get("T0_staging", {}).get("ok", False):
        return "staging incomplete: finish stage_tvr_login.sh (network steps need user OK); ASK USER to resubmit"
    if not by.get("T2_vulkan_device", {}).get("ok", False):
        if facts.get("nvidia_modeset") is False:
            return ("no NVIDIA Vulkan device and /dev/nvidia-modeset absent: ASK USER - request "
                    "NVIDIA_DRIVER_CAPABILITIES=graphics (or all) for aigc jobs from the admins, or fall back "
                    "(lavapipe / Xvfb+llvmpipe / gsplat-on-ScanNet++ pose reproduction)")
        return ("no NVIDIA Vulkan device although /dev/nvidia-modeset is present: overlay problem; read ldd.txt "
                "and vulkaninfo stderr, fix the overlay on the login node, ASK USER before a resubmit")
    return ("Vulkan device present but the AI2-THOR run failed (start-up / frames / renderer / tasks / parallel): "
            "read thor1/Player*.log and logs/; ASK USER before a resubmit (the TVR reservation is 0.5 GPU-h)")


def verdict(out):
    rows, caveats, speed = tvr_gate(out)
    erows = egl_gate(out)
    facts = _load(os.path.join(out, "facts.json"))
    if isinstance(facts.get("cvd_remap"), dict):
        caveats.append(f"CUDA_VISIBLE_DEVICES remapped for ai2thor: {facts['cvd_remap']}")
    tvr_pass = all(r["ok"] for r in rows)
    egl_pass = all(r["ok"] for r in erows)
    v = {
        "TVR_GATE": "PASS" if tvr_pass else "FAIL",
        "EGL_GATE": "PASS" if egl_pass else "FAIL",
        "reasons": [f"{r['name']}: {r['detail']}" for r in rows + erows if not r["ok"]],
        "caveats": caveats,
        "steps_per_s": speed,
        "facts": {k: facts.get(k) for k in ("nvidia_modeset", "dev_dri", "render_node", "cuda_visible_devices",
                                            "cvd_remap", "n_smi_gpus", "nvidia_driver_capabilities",
                                            "tmp_free_bytes", "shm_free_bytes", "cpu_max")},
        "next": next_step(tvr_pass, egl_pass, rows, facts),
        "criteria": {"tvr": rows, "egl": erows},
        "thresholds": {"driver": DRIVER, "gpu": GPU_NAME, "steps": STEPS, "startup_max_s": STARTUP_MAX_S,
                       "distinct_min": DISTINCT_MIN, "success_min": SUCCESS_MIN, "nproc": NPROC,
                       "nproc_steps": NPROC_STEPS, "egl_rgba": list(EGL_RGBA), "egl_tol": EGL_TOL},
    }
    tmp = os.path.join(out, "verdict.json.part")
    with open(tmp, "w") as f:
        json.dump(v, f, indent=1, default=str)
    os.replace(tmp, os.path.join(out, "verdict.json"))
    return v


def write_mapping(out, home):
    """Step 2 helper: keep ai2thor's own mapping when it would work, else write the 1-card fallback."""
    m = cuda_vulkan_mapping(_read(os.path.join(out, "vulkaninfo_full.txt")),
                            _read(os.path.join(out, "nvidia_smi_L.txt")),
                            _read(os.path.join(out, "vulkaninfo_summary.txt")))
    path = os.path.join(home, ".ai2thor", "cuda-vulkan-mapping.json")
    if m["mode"] == "fallback":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(m["mapping"], f)
        m["written"] = path
    with open(os.path.join(out, "mapping.json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "verdict":
        v = verdict(argv[1])
        print(json.dumps({k: v[k] for k in ("TVR_GATE", "EGL_GATE", "reasons", "caveats", "steps_per_s", "next")},
                         indent=1, default=str))
        return 0
    if len(argv) == 3 and argv[0] == "mapping":
        print(json.dumps(write_mapping(argv[1], argv[2]), indent=1))
        return 0
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
