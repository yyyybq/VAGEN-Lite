"""Lavapipe (Mesa CPU Vulkan) worker setup for AI2-THOR / TVRBench: THE single source of the TVR render setup.

The TVR probe (vpd/tvr/probe_lvp.py), the demo replay and the eval harness all build their AI2-THOR workers from
this module, so demos and eval share one renderer, one badge font rule and one ProcTHOR loader (DECISIONS G3(a):
lavapipe, 640x360, quality Ultra, antiAliasing 4, fov 60, as in the official eval; approved by the user
2026-11-10).  Never run any of the env/controller factories on the login node: they start Unity.

Pieces (pure functions unless noted, so they are unit-testable without ai2thor):
  worker_env(base, lp_threads=L, home=H, profile=P)   the environment dict of ONE THOR worker process:
        - unsets every VK_* / __EGL_VENDOR_LIBRARY_* / __NV_* / __GL_* / __GLX_* / MESA_* / GALLIUM_* / LP_* /
          ACTIVESPATIAL_* variable, LD_PRELOAD, DISPLAY, XDG_DATA_HOME, XDG_CONFIG_HOME;
        - VK_ICD_FILENAMES = VK_DRIVER_FILES = <TVR env>/share/vulkan/icd.d/lvp_icd.x86_64.json (image loader
          1.2.131 reads the first name, the env loader 1.4.357 the second);
        - LD_LIBRARY_PATH by the profile P (LVP_PROFILES, recorded as VPD_LVP_PROFILE):
            "envlib"       <TVR env>/lib exactly (the brief's profile; drops the NVGL overlay and every other entry;
                           first run by the TVR probe);
            "envlib+orig"  <TVR env>/lib + the container's original LD_LIBRARY_PATH (the profile of every measured
                           lavapipe run: debug_job.sh prof_lvpforced (v10) and tvr_smoke.sh's lavapipe step).
          The TVR probe tries "envlib" and, only if its first worker cannot start, "envlib+orig" once; its
          summary.lvp_profile names the profile that worked, and the demo replay and eval freeze that one;
        - CUDA_VISIBLE_DEVICES = 0 for THOR workers only (thor=True).  With gpu_device 0 and the pre-written
          cuda-vulkan-mapping {"0": 0}, ai2thor passes -force-device-index 0, and Vulkan device 0 is llvmpipe
          (the only ICD); Unity 2020.3 aborts without a device index (smoke_run1);
        - LP_NUM_THREADS = L, explicit and required (llvmpipe would otherwise size its pool from the host);
        - OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = 1, TOKENIZERS_PARALLELISM=false, VPD_THREADS=1;
        - HOME = H (per worker: own ~/.ai2thor and Unity Player.log), XDG_CACHE_HOME = H/.cache,
          PATH = <TVR env>/bin first, PYTHONNOUSERSITE=1, PYTHONUNBUFFERED=1.
  prepare_home(H)          (writes) H/.ai2thor/releases -> OUT/tvr/ai2thor_home/.ai2thor/releases and
                           H/.ai2thor/cuda-vulkan-mapping.json = {"0": 0}.
  check_icd()              the lavapipe ICD JSON exists and names an existing absolute library_path.
  font_status() / install_font_rule(tv, font=BADGE_FONT, expected_sha256=BADGE_FONT_SHA256, allow_missing=False)
        TVRBench's burn_label hard-codes /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf and silently falls
        back to ImageFont.load_default() (a different badge); the system copy also differs between images (the
        login node's is not the matplotlib copy's bytes).  The rule, identical for demos and eval: the badge font
        is ALWAYS the TVR env's matplotlib DejaVuSans-Bold.ttf (BADGE_FONT, the same bytes on every node through
        the shared FS), pinned by sha256 (BADGE_FONT_SHA256).  install_font_rule replaces thor_view_env's ImageFont
        by a shim that redirects exactly TVRBench's hard-coded path to BADGE_FONT, whether or not the system font
        exists ("pinned").  A missing BADGE_FONT or one whose sha256 differs is refused, unless allow_missing (the
        probe only: badges are masked when comparing; it then reports "missing" / "mismatch").
  import_tvrbench(repo)    (imports) tvrbench.env.thor_env and tvrbench.online_rl.thor_view_env of the pinned repo.
  view_env_config() / make_view_env(tv) / make_thor_env(te)
        TVRBench's ThorViewEnv / ThorEnv with the official eval settings: 640x360, fov 60, quality Ultra,
        antiAliasing 4 (ThorEnv's default), platform 'auto' = CloudRendering (ACTIVESPATIAL_USE_X11 must not be 1),
        gpu_device 0, gridSize 0.25, rotateStepDegrees 45.  eval_va.py builds its env the same way.
  procthor_line_index() / extract_procthor_houses() / resolve_scene() / load_scene()
        ProcTHOR as TVRBench does: scene id procthor_NNNNN = 0-based line NNNNN of data/procthor-10k/train.jsonl.gz,
        parsed with json.loads.  extract_procthor_houses copies the needed lines once (one pass over the gz);
        resolve_scene primes ThorViewEnv's class cache with that dict, then calls TVRBench's own _resolve_scene.
  read_cpu_stat() / cpu_delta()   cgroup v2 cpu.stat deltas (cores used, throttling).

Python 3.10 (TVR env) and 3.12 (tests).  Stdlib only at import time; numpy / PIL are imported where used.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from vpd import paths

# --- fixed locations ------------------------------------------------------------------------------------------
TVR_ENV = paths.TVR_ENV
TVR_PY = TVR_ENV / "bin" / "python"
LVP_ICD_NAME = "lvp_icd.x86_64.json"
LVP_ICD = TVR_ENV / "share" / "vulkan" / "icd.d" / LVP_ICD_NAME
RELEASES = paths.AI2THOR_HOME / ".ai2thor" / "releases"
BUILD_NAME = "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917"
# ai2thor starts the player as $HOME/.ai2thor/releases/<name>/<name>; match lingering players on this suffix
UNITY_PAT = f"/.ai2thor/releases/{BUILD_NAME}/{BUILD_NAME}"
TVR_REPO = paths.TVR_REPO
TASKS_DIR = TVR_REPO / "data" / "tasks"
PROCTHOR_DIR = TVR_REPO / "data" / "procthor-10k"
PROCTHOR_TRAIN = PROCTHOR_DIR / "train.jsonl.gz"
SYSTEM_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # hard-coded in TVRBench burn_label
# THE badge font of demos and eval: the TVR env's matplotlib 3.10.9 copy (704128 B; sha256 checked 2026-11-10 on
# the login node; the system copy there is another file, aa2ebed7..., 705684 B)
BADGE_FONT = TVR_ENV / "lib" / "python3.10" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf" / \
    "DejaVuSans-Bold.ttf"
BADGE_FONT_SHA256 = "b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167"
CUDA_VULKAN_MAPPING = {"0": 0}
PLAYER_LOG_REL = Path(".config") / "unity3d" / "Allen Institute for Artificial Intelligence" / "AI2-THOR" / \
    "Player.log"

# --- the official eval settings (scripts/eval_va.py + ThorViewEnv + ThorEnv defaults) ---------------------------
W, H = 640, 360
EVAL_SETTINGS = {"width": W, "height": H, "fov": 60, "quality": "Ultra", "anti_aliasing": 4,
                 "platform": "CloudRendering", "gpu_device": 0, "grid_size": 0.25, "rotate_step": 45}
MAX_STEPS = {"ithor": 30, "procthor": 40}  # eval_va.py --max_steps_ithor / --max_steps_procthor
ENV_MAX_STEPS = max(MAX_STEPS.values())  # eval_va sets the env cap to the larger one
# Badge region masked in every image comparison (640x360): x >= 536 and y <= 31 (TVRBench badge x 541-636, y 4-26)
BADGE_X0, BADGE_Y1 = 536, 31

# --- worker environment -----------------------------------------------------------------------------------------
STRIP_PREFIXES = ("VK_", "__EGL_VENDOR_LIBRARY_", "__NV_", "__GL_", "__GLX_", "MESA_", "GALLIUM_", "LP_",
                  "ACTIVESPATIAL_")
STRIP_EXACT = frozenset({"LD_PRELOAD", "LD_LIBRARY_PATH", "DISPLAY", "XDG_DATA_HOME", "XDG_CONFIG_HOME",
                         "XDG_CACHE_HOME", "HOME"})
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# LD_LIBRARY_PATH profiles (module doc), in the order the probe tries them
LVP_PROFILES = ("envlib", "envlib+orig")
LVP_PROFILE_DOC = {"envlib": "<TVR env>/lib only",
                   "envlib+orig": "<TVR env>/lib + the container's original LD_LIBRARY_PATH (measured profile: "
                                  "debug_job.sh prof_lvpforced v10, tvr_smoke.sh lavapipe step)"}
# the payload exports the container's LD_LIBRARY_PATH under this name before it changes anything (set, maybe empty);
# without it, base LD_LIBRARY_PATH is taken
ORIG_LD_VAR = "VPD_TVR_ORIG_LD_LIBRARY_PATH"


def _positive_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 1


def stripped(key: str, thor: bool = True) -> bool:
    """True if worker_env drops `key` from the base environment (it may then set it again)."""
    if key in STRIP_EXACT or key.startswith(STRIP_PREFIXES):
        return True
    return thor and key == "CUDA_VISIBLE_DEVICES"


def original_ld_library_path(base: Mapping[str, str]) -> str:
    """The container's LD_LIBRARY_PATH as the payload saw it at its start (ORIG_LD_VAR when set, even if empty)."""
    if ORIG_LD_VAR in base:
        return base[ORIG_LD_VAR]
    return base.get("LD_LIBRARY_PATH", "")


def ld_library_path(profile: str, base: Mapping[str, str], tvr_env=TVR_ENV) -> str:
    """LD_LIBRARY_PATH of a worker under `profile` (LVP_PROFILES); empty entries (= the cwd) and repeats of
    <TVR env>/lib are dropped from the original value."""
    lib = str(Path(tvr_env) / "lib")
    if profile == "envlib":
        return lib
    if profile == "envlib+orig":
        rest = [p for p in original_ld_library_path(base).split(":") if p and p != lib]
        return ":".join([lib] + rest)
    raise ValueError(f"unknown lavapipe profile {profile!r} (one of {LVP_PROFILES})")


def worker_env(base: Mapping[str, str], *, lp_threads: int, home, thor: bool = True, tvr_env=TVR_ENV,
               icd=None, profile: str = LVP_PROFILES[0]) -> Dict[str, str]:
    """The environment of one lavapipe worker process (see the module doc).  `base` is not modified."""
    if not _positive_int(lp_threads):
        raise ValueError(f"lp_threads must be a positive int (LP_NUM_THREADS is required), got {lp_threads!r}")
    if home is None or str(home) == "":
        raise ValueError("home is required (one HOME per worker)")
    tvr_env = Path(tvr_env)
    ld = ld_library_path(profile, base, tvr_env)
    icd = Path(icd) if icd is not None else tvr_env / "share" / "vulkan" / "icd.d" / LVP_ICD_NAME
    env = {k: v for k, v in base.items() if not stripped(k, thor)}
    bin_dir = str(tvr_env / "bin")
    rest = [p for p in base.get("PATH", "").split(":") if p and p != bin_dir]
    env.update({
        "VK_ICD_FILENAMES": str(icd),
        "VK_DRIVER_FILES": str(icd),
        "LD_LIBRARY_PATH": ld,
        "VPD_LVP_PROFILE": profile,
        "LP_NUM_THREADS": str(lp_threads),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(Path(home) / ".cache"),
        "PATH": ":".join([bin_dir] + rest),
        "TOKENIZERS_PARALLELISM": "false",
        "VPD_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    for k in THREAD_VARS:
        env[k] = "1"
    if thor:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def env_summary(env: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """The render-relevant part of a worker environment (for result files)."""
    keys = ("VK_ICD_FILENAMES", "VK_DRIVER_FILES", "LD_LIBRARY_PATH", "VPD_LVP_PROFILE", "LP_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES", "HOME", "OMP_NUM_THREADS", "ACTIVESPATIAL_USE_X11",
            "__EGL_VENDOR_LIBRARY_FILENAMES", "DISPLAY")
    return {k: env.get(k) for k in keys}


def check_icd(icd=LVP_ICD) -> Dict[str, Any]:
    """{ok, icd, library_path, why}: the ICD JSON parses and its library_path is absolute and exists."""
    icd = Path(icd)
    out: Dict[str, Any] = {"icd": str(icd), "ok": False, "library_path": None, "why": ""}
    try:
        doc = json.loads(icd.read_text())
        lib = doc["ICD"]["library_path"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        out["why"] = f"unreadable ICD {icd}: {e!r}"
        return out
    out["library_path"] = lib
    if not os.path.isabs(lib):
        out["why"] = f"library_path is not absolute: {lib}"
    elif not os.path.isfile(lib):
        out["why"] = f"library_path does not exist: {lib}"
    else:
        out["ok"] = True
    return out


def prepare_home(home, *, releases=RELEASES, mapping: Mapping[str, int] = CUDA_VULKAN_MAPPING) -> Dict[str, Any]:
    """Create one worker HOME: .ai2thor/releases symlink (replaced if it points elsewhere; a real directory or
    file there is refused) and .ai2thor/cuda-vulkan-mapping.json = {"0": 0}.  Returns what was written."""
    home = Path(home)
    a = home / ".ai2thor"
    a.mkdir(parents=True, exist_ok=True)
    link = a / "releases"
    if link.is_symlink():
        if os.readlink(link) != str(releases):
            link.unlink()
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink; refusing to replace it")
    if not link.is_symlink():
        link.symlink_to(str(releases), target_is_directory=True)
    mp = a / "cuda-vulkan-mapping.json"
    mp.write_text(json.dumps(dict(mapping)) + "\n")
    return {"home": str(home), "releases": str(releases), "mapping": dict(mapping), "mapping_file": str(mp)}


def player_log(home) -> Path:
    """Unity's Player.log of the worker whose HOME is `home`."""
    return Path(home) / PLAYER_LOG_REL


# --- badge font -------------------------------------------------------------------------------------------------
def _sha256_file(p) -> Optional[str]:
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def font_status(font=BADGE_FONT, expected_sha256: Optional[str] = BADGE_FONT_SHA256,
                system=SYSTEM_FONT) -> Dict[str, Any]:
    """The pinned badge font: {path, present, bytes, sha256, expected_sha256, sha256_ok, mode} with mode
    "pinned" (present, sha256 == expected), "mismatch" (present, other bytes) or "missing"; plus the system copy
    (system_present / system_sha256), recorded only, never used."""
    present = os.path.isfile(font)
    sha = _sha256_file(font) if present else None
    size = None
    if present:
        try:
            size = os.path.getsize(font)
        except OSError:
            size = None
    ok = bool(sha is not None and expected_sha256 and sha == expected_sha256)
    sp = os.path.isfile(system)
    return {"path": str(font), "present": present, "bytes": size, "sha256": sha,
            "expected_sha256": expected_sha256, "sha256_ok": ok,
            "mode": "pinned" if ok else ("mismatch" if sha is not None else "missing"),
            "system": str(system), "system_present": sp, "system_sha256": _sha256_file(system) if sp else None}


class FontRedirect:
    """Stands in for PIL.ImageFont inside thor_view_env: truetype(<SYSTEM_FONT>, ...) loads `dst` instead; every
    other attribute is the real module's."""

    def __init__(self, real, src: str, dst: str):
        self._real, self._src, self._dst = real, str(src), str(dst)

    def truetype(self, font=None, *args, **kwargs):
        if font is not None and str(font) == self._src:
            font = self._dst
        return self._real.truetype(font, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_font_rule(tv_module, *, font=BADGE_FONT, expected_sha256: str = BADGE_FONT_SHA256,
                      allow_missing: bool = False) -> Dict[str, Any]:
    """Apply the badge-font rule (module doc) to an imported thor_view_env module; returns font_status() plus
    `installed` (whether burn_label now loads `font`).

    The redirect of TVRBench's hard-coded SYSTEM_FONT path to `font` is installed whenever `font` exists, also
    when the system font is present.  mode "pinned" = installed and sha256 == expected_sha256.  Any other mode
    raises RuntimeError unless allow_missing (the probe only, which masks badges): "mismatch" keeps the redirect
    (the file at `font` is used and its sha256 is reported), "missing" installs nothing.
    """
    if not expected_sha256:
        raise ValueError("expected_sha256 is required (the badge font is pinned by sha256)")
    st = font_status(font, expected_sha256)
    mode = st["mode"]
    if mode != "pinned" and not allow_missing:
        what = (f"has sha256 {st['sha256']}, expected {expected_sha256}" if mode == "mismatch" else "is missing")
        raise RuntimeError(f"badge font {font} {what}: demos and eval must burn the badge with the pinned "
                           f"DejaVuSans-Bold.ttf (TVRBench would otherwise use whatever font the image has); "
                           f"refusing")
    st["installed"] = False
    if mode in ("pinned", "mismatch"):
        real = tv_module.ImageFont
        if isinstance(real, FontRedirect):
            real = real._real
        tv_module.ImageFont = FontRedirect(real, SYSTEM_FONT, str(font))
        st["installed"] = True
    return st


# --- TVRBench imports and factories -------------------------------------------------------------------------------
def import_tvrbench(repo=TVR_REPO) -> Tuple[Any, Any]:
    """(thor_env, thor_view_env) modules of the pinned TVRBench checkout `repo` (imports ai2thor)."""
    r = str(Path(repo).resolve())
    if r not in sys.path:
        sys.path.insert(0, r)
    te = importlib.import_module("tvrbench.env.thor_env")
    tv = importlib.import_module("tvrbench.online_rl.thor_view_env")
    for m in (te, tv):
        f = Path(m.__file__).resolve()
        if Path(r) not in f.parents:
            raise ImportError(f"{m.__name__} came from {f}, not from {r}")
    return te, tv


def check_platform_env(environ: Mapping[str, str] = os.environ) -> None:
    """platform 'auto' must resolve to CloudRendering (Vulkan): refuse the X11 / Linux64 switch."""
    if environ.get("ACTIVESPATIAL_USE_X11", "0") == "1":
        raise RuntimeError("ACTIVESPATIAL_USE_X11=1 selects Linux64/X11; the lavapipe setup needs CloudRendering")


def view_env_config(*, procthor_dir=PROCTHOR_DIR, max_steps: int = ENV_MAX_STEPS,
                    gpu_device: int = EVAL_SETTINGS["gpu_device"]) -> Dict[str, Any]:
    """ThorViewEnv env_config as eval_va.py builds it (plus the explicit fov / quality defaults)."""
    return {"gpu_device": gpu_device, "width": EVAL_SETTINGS["width"], "height": EVAL_SETTINGS["height"],
            "fov": EVAL_SETTINGS["fov"], "quality": EVAL_SETTINGS["quality"], "procthor_dir": str(procthor_dir),
            "max_steps": int(max_steps)}


def make_view_env(tv_module, *, procthor_dir=PROCTHOR_DIR, max_steps: int = ENV_MAX_STEPS, warm: bool = True,
                  environ: Mapping[str, str] = os.environ):
    """TVRBench ThorViewEnv with the official eval settings; warm=True starts the Controller now (like eval_va)."""
    check_platform_env(environ)
    env = tv_module.ThorViewEnv(env_config=view_env_config(procthor_dir=procthor_dir, max_steps=max_steps))
    if warm:
        env._ensure_env()
    return env


def make_thor_env(te_module, *, scene: str = "FloorPlan1", environ: Mapping[str, str] = os.environ):
    """TVRBench ThorEnv with the official eval settings (for raw-Controller tools; demos/eval use ThorViewEnv)."""
    check_platform_env(environ)
    s = EVAL_SETTINGS
    return te_module.ThorEnv(scene=scene, width=s["width"], height=s["height"], grid_size=s["grid_size"],
                             rotate_step=s["rotate_step"], fov=s["fov"], quality=s["quality"],
                             anti_aliasing=s["anti_aliasing"], platform="auto", gpu_device=s["gpu_device"])


def controller_settings(thor_env) -> Dict[str, Any]:
    """What the running Controller reports (initialization parameters, server-side screen size, fov, platform)."""
    c = getattr(thor_env, "controller", None)
    out: Dict[str, Any] = {}
    ip = getattr(c, "initialization_parameters", None) or {}
    for k in ("fieldOfView", "antiAliasing", "gridSize", "rotateStepDegrees", "snapToGrid", "renderDepthImage",
              "renderInstanceSegmentation", "makeAgentsVisible"):
        if k in ip:
            out[k] = ip[k]
    for k in ("width", "height", "quality", "gpu_device"):
        if hasattr(c, k):
            out[k] = getattr(c, k)
    build = getattr(c, "_build", None)
    plat = getattr(build, "platform", None)
    if plat is not None:
        out["platform"] = getattr(plat, "__name__", str(plat))
    ev = getattr(c, "last_event", None)
    meta = getattr(ev, "metadata", None) or {}
    for k in ("screenWidth", "screenHeight", "fov"):
        if k in meta:
            out["server_" + k] = meta[k]
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in out.items()}


# --- tasks and ProcTHOR -------------------------------------------------------------------------------------------
def load_tasks(split: str, tasks_dir=TASKS_DIR) -> List[Dict[str, Any]]:
    """data/tasks/<split>.json as a list (a {"tasks": [...]} file is unwrapped)."""
    with open(Path(tasks_dir) / f"{split}.json") as f:
        doc = json.load(f)
    return doc if isinstance(doc, list) else doc["tasks"]


def scenes_of(tasks: Iterable[Mapping[str, Any]]) -> List[str]:
    """Unique scene ids in first-seen order."""
    seen: Dict[str, None] = {}
    for t in tasks:
        seen.setdefault(t["scene"], None)
    return list(seen)


def is_procthor(scene: str) -> bool:
    return str(scene).startswith("procthor_")


def scene_kind(scene: str) -> str:
    return "procthor" if is_procthor(scene) else "ithor"


_PROCTHOR_RE = re.compile(r"procthor_(\d+)")


def procthor_line_index(scene: str) -> int:
    """procthor_NNNNN -> NNNNN (0-based line of train.jsonl.gz), as TVRBench's _load_procthor_house."""
    m = _PROCTHOR_RE.fullmatch(str(scene))
    if not m:
        raise ValueError(f"not a ProcTHOR scene id: {scene!r} (expected procthor_NNNNN)")
    return int(m.group(1))


def extract_procthor_houses(src, scenes: Iterable[str], dest) -> Dict[str, Dict[str, Any]]:
    """Copy the train.jsonl(.gz) lines of the ProcTHOR `scenes` to dest/<scene>.json in ONE pass.

    Each file holds the line exactly (without its newline), so json.loads gives the dict TVRBench would load.
    Returns {scene: {line, sha256 (of the line text), bytes, path}}; a missing line raises ValueError.
    """
    want: Dict[int, str] = {}
    for s in scenes:
        if is_procthor(s):
            want[procthor_line_index(s)] = s
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Any]] = {}
    if not want:
        return out
    src = str(src)
    opener = gzip.open if src.endswith(".gz") else open
    last = max(want)
    with opener(src, "rt") as f:
        for k, line in enumerate(f):
            if k in want:
                text = line.rstrip("\n")
                json.loads(text)  # must parse
                s = want[k]
                p = dest / f"{s}.json"
                tmp = p.with_suffix(".json.part")
                tmp.write_text(text)
                os.replace(tmp, p)
                out[s] = {"line": k, "sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text),
                          "path": str(p)}
            if k >= last:
                break
    missing = sorted(s for s in want.values() if s not in out)
    if missing:
        raise ValueError(f"ProcTHOR lines not found in {src}: {missing}")
    return out


def load_house(houses_dir, scene: str) -> Dict[str, Any]:
    with open(Path(houses_dir) / f"{scene}.json") as f:
        return json.load(f)


def prime_procthor_cache(view_env_cls, scene: str, house: Mapping[str, Any]) -> None:
    """Put `house` into ThorViewEnv's class-level ProcTHOR cache (first value wins, as in TVRBench)."""
    with view_env_cls._CACHE_LOCK:
        view_env_cls._PROCTHOR_CACHE.setdefault(scene, house)


def resolve_scene(view_env, scene: str, houses_dir=None):
    """iTHOR name -> the name; procthor_NNNNN -> the house dict, through TVRBench's own _resolve_scene (with the
    class cache primed from houses_dir/<scene>.json when that file exists)."""
    if is_procthor(scene) and houses_dir is not None and (Path(houses_dir) / f"{scene}.json").is_file():
        prime_procthor_cache(type(view_env), scene, load_house(houses_dir, scene))
    return view_env._resolve_scene(scene)


def load_scene(view_env, scene: str, houses_dir=None) -> float:
    """Load `scene` into a started ThorViewEnv (ThorEnv.load_scene = controller.reset) and keep ThorViewEnv's own
    bookkeeping (_scene) in step, so a later _sync_reset of a task in this scene does not reload.  Returns the
    seconds of the reset itself (the house dict is resolved first, untimed)."""
    data = resolve_scene(view_env, scene, houses_dir)
    t0 = time.perf_counter()
    view_env._env.load_scene(data)
    dt = time.perf_counter() - t0
    view_env._scene = scene
    return dt


# --- cgroup CPU accounting ----------------------------------------------------------------------------------------
CPU_STAT = "/sys/fs/cgroup/cpu.stat"
CPU_MAX = "/sys/fs/cgroup/cpu.max"


def read_cpu_stat(path=CPU_STAT) -> Dict[str, int]:
    """cgroup v2 cpu.stat as {key: int}; {} when unreadable."""
    out: Dict[str, int] = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                    out[parts[0]] = int(parts[1])
    except OSError:
        return {}
    return out


def read_cpu_max(path=CPU_MAX) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cpu_delta(a: Mapping[str, int], b: Mapping[str, int], wall_s: float) -> Dict[str, Optional[float]]:
    """Cores used over wall_s and the CFS throttling between two cpu.stat reads (None when unavailable)."""
    def d(k):
        return (b[k] - a[k]) if (k in a and k in b) else None

    use, per, thr, thr_us = d("usage_usec"), d("nr_periods"), d("nr_throttled"), d("throttled_usec")
    w = max(float(wall_s), 1e-9)
    return {"wall_s": round(w, 3),
            "cores": None if use is None else round(use / 1e6 / w, 3),
            "cpu_s": None if use is None else round(use / 1e6, 3),
            "nr_periods": per, "nr_throttled": thr,
            "throttled_frac": (round(thr / per, 4) if (per and thr is not None) else (0.0 if per == 0 else None)),
            "throttled_s": None if thr_us is None else round(thr_us / 1e6, 3)}
