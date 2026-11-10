"""Tests of package tvrprobe: vpd/tvr/lvp.py, vpd/tvr/probe_lvp.py and vpd/jobs/tvr_probe.sh (DECISIONS G3).

Hermetic: no ai2thor, no Unity, no GPU, no network, no rendering.  TVRBench's real thor_env / thor_view_env code
(read-only checkout OUT/tvr/repos/TVRBench; those tests skip without it) runs on a FAKE ai2thor package written
into tmp_path: a Controller whose frame is a pure function of (scene, pose) and whose reachable set is a fixed
grid.  The shell payload runs in a fake worktree with a fake common_env.sh, a fake interpreter and fake
pgrep/pkill.  Login node, 1 core:
    cd WT && OMP_NUM_THREADS=1 PY_Q3 -B -m pytest -q -p no:cacheprovider vpd/tests/test_tvr_probe.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vpd import paths
from vpd.tvr import lvp, probe_lvp

REPO = lvp.TVR_REPO
PAYLOAD = paths.VPD_PKG / "jobs" / "tvr_probe.sh"
HAVE_REPO = (REPO / "tvrbench" / "online_rl" / "thor_view_env.py").is_file()
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="pinned TVRBench checkout not present")


def _real_ttf():
    """A real DejaVuSans-Bold.ttf to copy for tests (read only), or None."""
    cands = [lvp.BADGE_FONT, Path(lvp.SYSTEM_FONT)]
    try:
        import matplotlib
        cands.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
    except ImportError:
        pass
    return next((Path(c) for c in cands if Path(c).is_file()), None)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# --- the fake ai2thor ---------------------------------------------------------------------------------------------
FAKE_CONTROLLER = '''
"""Fake ai2thor.controller (vpd tests): no Unity.  A frame is a pure function of (scene, pose[, history])."""
import hashlib

import numpy as np

GRID = [round(-1.0 + 0.25 * i, 2) for i in range(11)]
BLOCKED = {(1.0, 1.0)}


class Event:
    def __init__(self, metadata, frame):
        self.metadata = metadata
        self.frame = frame


class Controller:
    HISTORY = False  # True: frames also depend on the steps since the last scene reset (a temporal-AA stand-in)
    created = []

    def __init__(self, **kw):
        self.kw = dict(kw)
        keys = ("fieldOfView", "antiAliasing", "gridSize", "rotateStepDegrees", "snapToGrid", "renderDepthImage",
                "renderInstanceSegmentation", "makeAgentsVisible")
        self.initialization_parameters = {k: kw[k] for k in keys if k in kw}
        self.width, self.height = kw["width"], kw["height"]
        self.quality, self.gpu_device = kw.get("quality"), kw.get("gpu_device")
        self.rot_step = float(kw.get("rotateStepDegrees", 90))
        self.stopped = False
        self._set_scene(kw.get("scene", "FloorPlan1"))
        Controller.created.append(self)

    def _set_scene(self, scene):
        self.scene = scene if isinstance(scene, str) else "house:" + str(scene["id"])
        self.pos = {"x": 0.0, "y": 0.9, "z": 0.0}
        self.rot, self.hor, self.hist = 0.0, 0.0, 0
        self.last_event = self._event(True, None)

    def _frame(self):
        key = "%s|%.2f|%.2f|%d|%d" % (self.scene, self.pos["x"], self.pos["z"], round(self.rot) % 360,
                                      round(self.hor))
        if Controller.HISTORY:
            key += "|%d" % self.hist
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        small = np.random.default_rng(seed).integers(0, 256, (9, 16, 3), dtype=np.uint8)
        return np.repeat(np.repeat(small, self.height // 9, axis=0), self.width // 16, axis=1)

    def _event(self, ok, ret, err=""):
        md = {"lastActionSuccess": ok, "errorMessage": err, "actionReturn": ret, "screenWidth": self.width,
              "screenHeight": self.height, "fov": self.kw.get("fieldOfView"),
              "agent": {"position": dict(self.pos), "rotation": {"x": 0.0, "y": float(self.rot % 360), "z": 0.0},
                        "cameraHorizon": float(self.hor)}}
        return Event(md, self._frame())

    def step(self, action=None, **kw):
        self.hist += 1
        if action == "GetReachablePositions":
            ev = self._event(True, [{"x": x, "y": 0.9, "z": z} for x in GRID for z in GRID
                                    if (x, z) not in BLOCKED])
        elif action == "Teleport":
            p = kw["position"]
            self.pos = {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}
            self.rot = float(kw["rotation"]["y"]) % 360
            self.hor = float(kw.get("horizon", self.hor))
            ev = self._event(True, None)
        elif action in ("RotateRight", "RotateLeft"):
            self.rot = (self.rot + (self.rot_step if action == "RotateRight" else -self.rot_step)) % 360
            ev = self._event(True, None)
        elif action in ("LookUp", "LookDown"):
            self.hor += -30.0 if action == "LookUp" else 30.0
            ev = self._event(True, None)
        else:
            ev = self._event(False, None, "unknown action %r" % (action,))
        self.last_event = ev
        return ev

    def reset(self, scene=None, **kw):
        self._set_scene(scene)
        return self.last_event

    def stop(self):
        self.stopped = True
'''


def write_fake_ai2thor(site: Path) -> Path:
    pkg = site / "ai2thor"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Fake ai2thor (vpd tests)."""\n')
    (pkg / "controller.py").write_text(FAKE_CONTROLLER)
    (pkg / "platform.py").write_text("class CloudRendering:\n    pass\n")
    return site


def _purge():
    for k in list(sys.modules):
        if k in ("tvrbench", "ai2thor") or k.startswith(("tvrbench.", "ai2thor.")):
            del sys.modules[k]


@pytest.fixture
def fake_site(tmp_path):
    return write_fake_ai2thor(tmp_path / "fake_site")


@pytest.fixture
def tvb(fake_site):
    """(thor_env, thor_view_env) of the real TVRBench checkout on the fake ai2thor."""
    if not HAVE_REPO:
        pytest.skip("pinned TVRBench checkout not present")
    saved = list(sys.path)
    _purge()
    sys.path.insert(0, str(fake_site))
    try:
        te, tv = lvp.import_tvrbench(REPO)
        assert sys.modules["ai2thor"].__file__.startswith(str(fake_site))
        yield te, tv
    finally:
        _purge()
        sys.path[:] = saved


# --- a small synthetic world ------------------------------------------------------------------------------------
Y = 0.9


def pose(x, z, rot, hor):
    return {"position": {"x": x, "y": Y, "z": z}, "rotation_y": rot, "horizon": hor}


FP_TID = "ithor_FloorPlan11_easy_t00_s00"
PT_TID = "procthor_procthor_00002_hard_t00_s00"
# start, actions (ending with Stop), target reached by the fake / TVRBench semantics
FP_TASK = {"task_id": FP_TID, "dataset": "ithor", "scene": "FloorPlan11", "difficulty": "easy",
           "start": pose(0.0, 0.0, 0.0, 0.0), "target": pose(0.25, 0.25, 45.0, 30.0)}
FP_ACTS = ["RotateRight", "MoveAhead", "LookDown", "Stop"]
PT_TASK = {"task_id": PT_TID, "dataset": "procthor", "scene": "procthor_00002", "difficulty": "hard",
           "start": pose(0.5, 0.5, 90.0, 0.0), "target": pose(0.75, 0.75, 45.0, 0.0)}
PT_ACTS = ["MoveAhead", "MoveLeft", "RotateLeft", "Stop"]
SFT_EXTRA = {"task_id": "ithor_FloorPlan12_hard_t00_s00", "dataset": "ithor", "scene": "FloorPlan12",
             "difficulty": "hard", "start": pose(0.0, 0.0, 0.0, 0.0), "target": pose(0.25, 0.0, 0.0, 0.0)}
EVAL_TASKS = [
    {"task_id": "ithor_FloorPlan13_easy_t00_s00", "dataset": "ithor", "scene": "FloorPlan13", "difficulty": "easy",
     "start": pose(0.0, 0.0, 0.0, 0.0), "target": pose(0.0, 0.25, 0.0, 0.0)},
    {"task_id": "procthor_procthor_00003_easy_t00_s00", "dataset": "procthor", "scene": "procthor_00003",
     "difficulty": "easy", "start": pose(-0.5, 0.0, 180.0, 0.0), "target": pose(-0.25, 0.25, 0.0, -30.0)},
]


def write_gz(path: Path, n: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"h{i:05d}", "rooms": [{"i": i}]}) + "\n")
    return path


def make_world(tmp: Path, tv, *, perturb=None) -> types.SimpleNamespace:
    """Task files, ProcTHOR gz, ICD, cgroup files and a 'public' sample rendered by a reference worker."""
    w = types.SimpleNamespace()
    w.tasks = tmp / "tasks"
    w.tasks.mkdir()
    (w.tasks / "sft.json").write_text(json.dumps([FP_TASK, PT_TASK, SFT_EXTRA]))
    (w.tasks / "eval.json").write_text(json.dumps(EVAL_TASKS))
    w.pt_dir = tmp / "pt"
    w.gz = write_gz(w.pt_dir / "train.jsonl.gz")
    lib = tmp / "libvulkan_lvp.so"
    lib.write_bytes(b"\x7fELF")
    w.icd = tmp / "lvp_icd.x86_64.json"
    w.icd.write_text(json.dumps({"ICD": {"library_path": str(lib)}, "file_format_version": "1.0.1"}))
    w.cpu_stat = tmp / "cpu.stat"
    w.cpu_stat.write_text("usage_usec 1000\nuser_usec 600\nsystem_usec 400\nnr_periods 10\nnr_throttled 0\n"
                          "throttled_usec 0\n")
    w.cpu_max = tmp / "cpu.max"
    w.cpu_max.write_text("1400000 100000\n")
    # the pinned badge font of this world (a copy of a real TTF when one exists)
    w.font = tmp / "badge_font" / "DejaVuSans-Bold.ttf"
    w.font.parent.mkdir()
    src = _real_ttf()
    if src is not None:
        shutil.copyfile(src, w.font)
    else:
        w.font.write_bytes(b"not a font: TVRBench's burn_label then falls back to load_default")
    w.font_sha = _sha(w.font)
    # the 'public' sample: our reference renders (optionally perturbed), the texts of the reference replay
    w.sample = tmp / "sample"
    ref_houses = tmp / "ref_houses"
    lvp.extract_procthor_houses(w.gz, ["procthor_00002"], ref_houses)
    ref = probe_lvp.Worker(tv, wid=99, houses_dir=ref_houses, procthor_dir=w.pt_dir, environ={})
    ref.start()
    data = []
    for task, acts in ((FP_TASK, FP_ACTS), (PT_TASK, PT_ACTS)):
        d = w.sample / "images" / task["task_id"]
        rep = ref.op_replay({"task": task, "actions": acts, "save_dir": str(d)})
        assert rep["final_exact"], rep
        for f in d.glob("rep_*.png"):
            f.unlink()
        if perturb is not None:
            for f in sorted(d.glob("*.png")):
                perturb(f)
        conv, imgs = [], []
        for k, t in enumerate(rep["turns"]):
            conv += [{"from": "human", "value": t["user_text"]}, {"from": "gpt", "value": f"Action: {acts[k]}"}]
            imgs += [f"images/{task['task_id']}/step_{k:02d}.png", f"images/{task['task_id']}/target.png"]
        data.append({"conversations": conv, "images": imgs, "system": tv.SYSTEM_PROMPT})
    ref.close()
    w.sft_train = tmp / "sft_train.json"
    w.sft_train.write_text(json.dumps(data))
    lines = []
    for f in sorted((w.sample / "images").rglob("*.png")):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(w.sample).as_posix()}")
    (w.sample / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    w.sample_tasks = (FP_TID, PT_TID)
    return w


FAKE_LIB = "/fake/tvr/lib"


def local_spawner(tv, houses_dir, procthor_dir, handle_cls=None, *, font, font_sha, fail=None, log=None):
    """spawn(wid, L, profile) with in-process workers; fail(wid, L, profile) -> True makes that start-up fail."""
    def spawn(wid, L, profile=lvp.LVP_PROFILES[0]):
        if log is not None:
            log.append((wid, L, profile))
        if fail is not None and fail(wid, L, profile):
            return None, {"id": 0, "ok": False, "error": f"RuntimeError('fake start-up failure ({profile})')",
                          "env_profile": profile}
        st = lvp.install_font_rule(tv, font=font, expected_sha256=font_sha, allow_missing=True)
        env = {"LP_NUM_THREADS": str(L), "CUDA_VISIBLE_DEVICES": "0", "VPD_LVP_PROFILE": profile,
               "LD_LIBRARY_PATH": FAKE_LIB + (":/orig/lib" if profile == "envlib+orig" else "")}
        wk = probe_lvp.Worker(tv, wid=wid, houses_dir=houses_dir, procthor_dir=procthor_dir, font=st,
                              font_mode=st["mode"], environ=env)
        ready = probe_lvp.jsafe(dict(wk.start(), ok=True, id=0, env_profile=profile))
        return (handle_cls or probe_lvp.LocalHandle)(wid, L, wk), ready
    return spawn


def make_probe(tmp: Path, w, tv, *, deadline_s=900.0, grid=None, handle_cls=None, cls=None, fail=None,
               spawn_log=None, **kw):
    out = tmp / "out"
    p = (cls or probe_lvp.Probe)(out, deadline_s=deadline_s,
                                 spawner=local_spawner(tv, out / "houses", w.pt_dir, handle_cls, font=w.font,
                                                       font_sha=w.font_sha, fail=fail, log=spawn_log),
                                 grid=grid if grid is not None else {2: (1, 2)}, steps=3, p1_l=4, repo=REPO,
                                 tasks_dir=w.tasks, sample_dir=w.sample, sft_train=w.sft_train, procthor_src=w.gz,
                                 icd=w.icd, sample_tasks=w.sample_tasks, cpu_stat=w.cpu_stat, cpu_max=w.cpu_max,
                                 font=w.font, font_sha256=w.font_sha, **kw)
    p.go_delay_s = 0.0
    return p, out


# ================================================================================================================
# lvp: worker environment, HOME, ICD, font, factories, ProcTHOR, cgroup
# ================================================================================================================
def test_worker_env_strips_and_sets(tmp_path):
    base = {"VK_ICD_FILENAMES": "/x/nvidia_icd.json", "VK_DRIVER_FILES": "/x/nvidia_icd.json", "VK_LAYER_PATH": "/x",
            "__EGL_VENDOR_LIBRARY_FILENAMES": "/x/10_nvidia.json", "__EGL_VENDOR_LIBRARY_DIRS": "/x",
            "LD_LIBRARY_PATH": "/o/tvr/nvgl_595.71.05/lib:/o/tvr/nvgl_595.71.05/glvnd:/usr/lib",
            "LP_NUM_THREADS": "64", "CUDA_VISIBLE_DEVICES": "5", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "ACTIVESPATIAL_USE_X11": "1", "ACTIVESPATIAL_X11_FORCE_VULKAN": "1", "DISPLAY": ":99",
            "__NV_PRIME_RENDER_OFFLOAD": "1", "__GL_SHADER_DISK_CACHE": "1", "MESA_VK_DEVICE_SELECT": "x",
            "GALLIUM_DRIVER": "llvmpipe", "LD_PRELOAD": "/x.so", "XDG_CONFIG_HOME": "/c", "XDG_DATA_HOME": "/d",
            "PATH": "/usr/bin:/tvr/bin", "PYTHONPATH": "/wt", "KEEP": "me", "HOME": "/root"}
    before = dict(base)
    home = tmp_path / "h"
    env = lvp.worker_env(base, lp_threads=3, home=home, tvr_env="/tvr")
    assert base == before  # not modified
    icd = "/tvr/share/vulkan/icd.d/lvp_icd.x86_64.json"
    assert env["VK_ICD_FILENAMES"] == env["VK_DRIVER_FILES"] == icd
    assert env["LD_LIBRARY_PATH"] == "/tvr/lib"  # NVGL overlay and everything else dropped
    assert env["LP_NUM_THREADS"] == "3" and env["CUDA_VISIBLE_DEVICES"] == "0"
    for k in lvp.THREAD_VARS:
        assert env[k] == "1"
    assert env["TOKENIZERS_PARALLELISM"] == "false" and env["VPD_THREADS"] == "1"
    assert env["HOME"] == str(home) and env["XDG_CACHE_HOME"] == str(home / ".cache")
    assert env["PATH"] == "/tvr/bin:/usr/bin"
    assert env["KEEP"] == "me" and env["PYTHONPATH"] == "/wt" and env["PYTHONNOUSERSITE"] == "1"
    gone = ("VK_LAYER_PATH", "__EGL_VENDOR_LIBRARY_FILENAMES", "__EGL_VENDOR_LIBRARY_DIRS", "ACTIVESPATIAL_USE_X11",
            "ACTIVESPATIAL_X11_FORCE_VULKAN", "DISPLAY", "__NV_PRIME_RENDER_OFFLOAD", "__GL_SHADER_DISK_CACHE",
            "MESA_VK_DEVICE_SELECT", "GALLIUM_DRIVER", "LD_PRELOAD", "XDG_CONFIG_HOME", "XDG_DATA_HOME")
    assert not [k for k in gone if k in env]
    assert not [k for k in env if k.startswith("VK_") and k not in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES")]
    assert lvp.env_summary(env)["LP_NUM_THREADS"] == "3"
    assert env["VPD_LVP_PROFILE"] == "envlib" and lvp.env_summary(env)["VPD_LVP_PROFILE"] == "envlib"


def test_worker_env_profiles(tmp_path):
    """envlib = <env>/lib only; envlib+orig appends the container's original LD_LIBRARY_PATH (the payload's
    VPD_TVR_ORIG_LD_LIBRARY_PATH when set, even empty; else base LD_LIBRARY_PATH), without empties or repeats."""
    assert lvp.LVP_PROFILES == ("envlib", "envlib+orig") and set(lvp.LVP_PROFILE_DOC) == set(lvp.LVP_PROFILES)
    base = {"LD_LIBRARY_PATH": "/now/a:/tvr/lib", lvp.ORIG_LD_VAR: "/orig/a::/tvr/lib:/orig/b"}
    e1 = lvp.worker_env(base, lp_threads=2, home=tmp_path, tvr_env="/tvr")
    assert e1["LD_LIBRARY_PATH"] == "/tvr/lib" and e1["VPD_LVP_PROFILE"] == "envlib"
    e2 = lvp.worker_env(base, lp_threads=2, home=tmp_path, tvr_env="/tvr", profile="envlib+orig")
    assert e2["LD_LIBRARY_PATH"] == "/tvr/lib:/orig/a:/orig/b" and e2["VPD_LVP_PROFILE"] == "envlib+orig"
    assert lvp.ld_library_path("envlib+orig", {"LD_LIBRARY_PATH": "/x:/y"}, "/tvr") == "/tvr/lib:/x:/y"
    assert lvp.ld_library_path("envlib+orig", {"LD_LIBRARY_PATH": "/x", lvp.ORIG_LD_VAR: ""}, "/tvr") == "/tvr/lib"
    assert lvp.ld_library_path("envlib+orig", {}, "/tvr") == "/tvr/lib"
    with pytest.raises(ValueError):
        lvp.worker_env(base, lp_threads=2, home=tmp_path, profile="nvgl")


@pytest.mark.parametrize("bad", [None, 0, -1, 2.0, True, "2"])
def test_worker_env_requires_lp(tmp_path, bad):
    with pytest.raises(ValueError):
        lvp.worker_env({}, lp_threads=bad, home=tmp_path)


def test_worker_env_non_thor_keeps_cvd_and_needs_home(tmp_path):
    env = lvp.worker_env({"CUDA_VISIBLE_DEVICES": "5"}, lp_threads=1, home=tmp_path, thor=False)
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert "CUDA_VISIBLE_DEVICES" not in lvp.worker_env({}, lp_threads=1, home=tmp_path, thor=False)
    with pytest.raises(ValueError):
        lvp.worker_env({}, lp_threads=1, home=None)


def test_default_paths():
    assert lvp.LVP_ICD == paths.TVR_ENV / "share" / "vulkan" / "icd.d" / "lvp_icd.x86_64.json"
    assert lvp.RELEASES == paths.AI2THOR_HOME / ".ai2thor" / "releases"
    assert lvp.SYSTEM_FONT == "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    assert lvp.EVAL_SETTINGS == {"width": 640, "height": 360, "fov": 60, "quality": "Ultra", "anti_aliasing": 4,
                                 "platform": "CloudRendering", "gpu_device": 0, "grid_size": 0.25, "rotate_step": 45}
    assert lvp.MAX_STEPS == {"ithor": 30, "procthor": 40} and lvp.ENV_MAX_STEPS == 40
    assert lvp.BADGE_FONT == (paths.TVR_ENV / "lib" / "python3.10" / "site-packages" / "matplotlib" / "mpl-data" /
                              "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
    assert len(lvp.BADGE_FONT_SHA256) == 64 and int(lvp.BADGE_FONT_SHA256, 16) >= 0


@pytest.mark.skipif(not lvp.LVP_ICD.is_file(), reason="TVR env not present")
def test_real_env_icd_and_pinned_font():
    """Read-only: the staged TVR env has the lavapipe ICD (absolute library path) and the pinned badge font."""
    r = lvp.check_icd()
    assert r["ok"], r
    assert r["library_path"] == str(paths.TVR_ENV / "lib" / "libvulkan_lvp.so")
    st = lvp.font_status()
    assert st["mode"] == "pinned" and st["sha256"] == lvp.BADGE_FONT_SHA256, st


def test_check_icd(tmp_path):
    lib = tmp_path / "libvulkan_lvp.so"
    lib.write_bytes(b"x")
    icd = tmp_path / "icd.json"
    icd.write_text(json.dumps({"ICD": {"library_path": str(lib)}}))
    assert lvp.check_icd(icd)["ok"]
    icd.write_text(json.dumps({"ICD": {"library_path": "libvulkan_lvp.so"}}))
    r = lvp.check_icd(icd)
    assert not r["ok"] and "not absolute" in r["why"]
    icd.write_text(json.dumps({"ICD": {"library_path": str(tmp_path / "nope.so")}}))
    assert "does not exist" in lvp.check_icd(icd)["why"]
    icd.write_text("{")
    assert not lvp.check_icd(icd)["ok"]
    assert not lvp.check_icd(tmp_path / "missing.json")["ok"]


def test_prepare_home(tmp_path):
    rel = tmp_path / "releases"
    rel.mkdir()
    home = tmp_path / "home"
    r = lvp.prepare_home(home, releases=rel)
    link = home / ".ai2thor" / "releases"
    assert link.is_symlink() and os.readlink(link) == str(rel)
    assert json.loads((home / ".ai2thor" / "cuda-vulkan-mapping.json").read_text()) == {"0": 0}
    assert r["mapping"] == {"0": 0}
    lvp.prepare_home(home, releases=rel)  # idempotent
    other = tmp_path / "other"
    other.mkdir()
    lvp.prepare_home(home, releases=other)  # a link elsewhere is replaced
    assert os.readlink(link) == str(other)
    link.unlink()
    link.mkdir()
    with pytest.raises(RuntimeError):
        lvp.prepare_home(home, releases=rel)
    assert lvp.player_log(home).name == "Player.log"


class RecordingFont:
    calls = []

    @staticmethod
    def truetype(font=None, size=10, *a, **k):
        RecordingFont.calls.append(str(font))
        return ("ttf", str(font), size)

    @staticmethod
    def load_default():
        return "default"


def test_font_rule(tmp_path):
    """The pinned font is ALWAYS used (also with a system font present); other bytes / no file are refused unless
    allow_missing (then reported: "mismatch" keeps the redirect, "missing" installs nothing)."""
    pin = tmp_path / "pin.ttf"
    pin.write_bytes(b"font")
    sha = hashlib.sha256(b"font").hexdigest()
    sysf = tmp_path / "sys.ttf"
    sysf.write_bytes(b"another font")  # a system copy exists and is NOT used
    st = lvp.font_status(pin, sha, system=sysf)
    assert st["mode"] == "pinned" and st["sha256_ok"] and st["bytes"] == 4 and st["path"] == str(pin)
    assert st["system_present"] and st["system_sha256"] == _sha(sysf)
    mod = types.SimpleNamespace(ImageFont=RecordingFont)
    got = lvp.install_font_rule(mod, font=pin, expected_sha256=sha)
    assert got["mode"] == "pinned" and got["installed"] and got["sha256"] == sha
    assert mod.ImageFont.truetype(lvp.SYSTEM_FONT, 16) == ("ttf", str(pin), 16)  # TVRBench's path -> the pin
    assert mod.ImageFont.truetype("/other.ttf", 12) == ("ttf", "/other.ttf", 12)
    assert mod.ImageFont.load_default() == "default"
    lvp.install_font_rule(mod, font=pin, expected_sha256=sha)  # again: no nesting
    assert mod.ImageFont._real is RecordingFont
    mod2 = types.SimpleNamespace(ImageFont=RecordingFont)
    with pytest.raises(RuntimeError, match="sha256"):
        lvp.install_font_rule(mod2, font=pin, expected_sha256="0" * 64)
    assert mod2.ImageFont is RecordingFont  # nothing installed on a refusal
    st2 = lvp.install_font_rule(mod2, font=pin, expected_sha256="0" * 64, allow_missing=True)
    assert st2["mode"] == "mismatch" and st2["installed"] and isinstance(mod2.ImageFont, lvp.FontRedirect)
    mod3 = types.SimpleNamespace(ImageFont=RecordingFont)
    with pytest.raises(RuntimeError, match="missing"):
        lvp.install_font_rule(mod3, font=tmp_path / "none.ttf", expected_sha256=sha)
    st3 = lvp.install_font_rule(mod3, font=tmp_path / "none.ttf", expected_sha256=sha, allow_missing=True)
    assert st3["mode"] == "missing" and not st3["installed"] and mod3.ImageFont is RecordingFont
    with pytest.raises(ValueError):
        lvp.install_font_rule(mod3, font=pin, expected_sha256="")


@needs_repo
def test_font_pin_with_tvrbench_burn_label(tvb, tmp_path):
    """The redirect works inside TVRBench's own burn_label (real PIL): its hard-coded path loads the pinned file."""
    _, tv = tvb
    src = _real_ttf()
    if src is None:
        pytest.skip("no DejaVuSans-Bold.ttf to test with")
    pin = tmp_path / "pin.ttf"
    shutil.copyfile(src, pin)
    z = np.zeros((360, 640, 3), np.uint8)
    seen = []
    real = tv.ImageFont

    class Spy:
        @staticmethod
        def truetype(font=None, *a, **k):
            seen.append(str(font))
            return real.truetype(font, *a, **k)

        def __getattr__(self, n):
            return getattr(real, n)

    tv.ImageFont = Spy()
    try:
        assert lvp.install_font_rule(tv, font=pin, expected_sha256=_sha(pin))["mode"] == "pinned"
        badge = probe_lvp.badge_bbox(np.asarray(tv.burn_label(z, "TARGET")))
    finally:
        tv.ImageFont = real
    assert seen == [str(pin)]
    assert badge is not None and badge[2] == 636  # right edge = W - pad


class FakeViewEnvCls:
    _PROCTHOR_CACHE: dict = {}

    def __init__(self, env_config=None):
        import threading
        type(self)._CACHE_LOCK = threading.Lock()
        self.env_config = env_config
        self.ensured = 0
        self._scene = None
        self.loaded = []
        self._env = types.SimpleNamespace(load_scene=self.loaded.append)

    def _ensure_env(self):
        self.ensured += 1
        self._scene = "FloorPlan1"

    def _resolve_scene(self, scene):
        return type(self)._PROCTHOR_CACHE.get(scene, scene) if scene.startswith("procthor_") else scene


def test_make_view_env_and_thor_env(tmp_path):
    tv = types.SimpleNamespace(ThorViewEnv=FakeViewEnvCls)
    env = lvp.make_view_env(tv, procthor_dir=tmp_path, environ={})
    assert env.env_config == {"gpu_device": 0, "width": 640, "height": 360, "fov": 60, "quality": "Ultra",
                              "procthor_dir": str(tmp_path), "max_steps": 40}
    assert env.ensured == 1
    assert lvp.make_view_env(tv, procthor_dir=tmp_path, warm=False, environ={}).ensured == 0
    with pytest.raises(RuntimeError):
        lvp.make_view_env(tv, environ={"ACTIVESPATIAL_USE_X11": "1"})
    got = {}

    class TE:
        def __init__(self, **kw):
            got.update(kw)

    lvp.make_thor_env(types.SimpleNamespace(ThorEnv=TE), scene="FloorPlan2", environ={})
    assert got == {"scene": "FloorPlan2", "width": 640, "height": 360, "grid_size": 0.25, "rotate_step": 45,
                   "fov": 60, "quality": "Ultra", "anti_aliasing": 4, "platform": "auto", "gpu_device": 0}


def test_procthor_houses(tmp_path):
    assert lvp.procthor_line_index("procthor_00007") == 7
    for bad in ("FloorPlan1", "procthor_x", "procthor_"):
        with pytest.raises(ValueError):
            lvp.procthor_line_index(bad)
    gz = write_gz(tmp_path / "train.jsonl.gz", 5)
    got = lvp.extract_procthor_houses(gz, ["FloorPlan1", "procthor_00003", "procthor_00001"], tmp_path / "h")
    assert sorted(got) == ["procthor_00001", "procthor_00003"] and got["procthor_00003"]["line"] == 3
    with gzip.open(gz, "rt") as f:
        line3 = f.read().splitlines()[3]
    assert (tmp_path / "h" / "procthor_00003.json").read_text() == line3
    assert got["procthor_00003"]["sha256"] == hashlib.sha256(line3.encode()).hexdigest()
    assert lvp.load_house(tmp_path / "h", "procthor_00003") == {"id": "h00003", "rooms": [{"i": 3}]}
    plain = tmp_path / "train.jsonl"
    with gzip.open(gz, "rt") as f:
        plain.write_text(f.read())
    assert lvp.extract_procthor_houses(plain, ["procthor_00004"], tmp_path / "p")["procthor_00004"]["line"] == 4
    with pytest.raises(ValueError):
        lvp.extract_procthor_houses(gz, ["procthor_00009"], tmp_path / "m")
    # resolve / load through the (fake) TVRBench class cache
    FakeViewEnvCls._PROCTHOR_CACHE = {}
    env = lvp.make_view_env(types.SimpleNamespace(ThorViewEnv=FakeViewEnvCls), environ={})
    assert lvp.resolve_scene(env, "procthor_00003", tmp_path / "h") == {"id": "h00003", "rooms": [{"i": 3}]}
    assert "procthor_00003" in FakeViewEnvCls._PROCTHOR_CACHE
    assert lvp.resolve_scene(env, "FloorPlan5", tmp_path / "h") == "FloorPlan5"
    s = lvp.load_scene(env, "procthor_00001", tmp_path / "h")
    assert s >= 0 and env._scene == "procthor_00001" and env.loaded[-1] == {"id": "h00001", "rooms": [{"i": 1}]}
    assert [lvp.scene_kind(x) for x in ("FloorPlan1", "procthor_1")] == ["ithor", "procthor"]


def test_tasks_and_scenes(tmp_path):
    (tmp_path / "sft.json").write_text(json.dumps([{"scene": "A"}, {"scene": "B"}, {"scene": "A"}]))
    (tmp_path / "eval.json").write_text(json.dumps({"tasks": [{"scene": "C"}]}))
    assert lvp.scenes_of(lvp.load_tasks("sft", tmp_path)) == ["A", "B"]
    assert lvp.scenes_of(lvp.load_tasks("eval", tmp_path)) == ["C"]


def test_cpu_stat(tmp_path):
    f = tmp_path / "cpu.stat"
    f.write_text("usage_usec 1000000\nuser_usec 1\nnr_periods 100\nnr_throttled 5\nthrottled_usec 250000\n")
    a = lvp.read_cpu_stat(f)
    assert a["usage_usec"] == 1000000 and a["nr_throttled"] == 5
    b = dict(a, usage_usec=9000000, nr_periods=200, nr_throttled=25, throttled_usec=1250000)
    d = lvp.cpu_delta(a, b, 2.0)
    assert d["cores"] == 4.0 and d["cpu_s"] == 8.0 and d["throttled_frac"] == 0.2 and d["throttled_s"] == 1.0
    assert lvp.read_cpu_stat(tmp_path / "none") == {}
    assert lvp.cpu_delta({}, {}, 1.0)["cores"] is None
    (tmp_path / "cpu.max").write_text("1400000 100000\n")
    assert lvp.read_cpu_max(tmp_path / "cpu.max") == "1400000 100000"
    assert lvp.read_cpu_max(tmp_path / "x") is None


@needs_repo
def test_tvrbench_on_fake_ai2thor_gets_eval_settings(tvb, tmp_path):
    """The factory drives TVRBench's own ThorViewEnv -> ThorEnv -> Controller with the official settings."""
    te, tv = tvb
    env = lvp.make_view_env(tv, procthor_dir=tmp_path, environ={})
    c = env._env.controller
    kw = c.kw
    assert (kw["width"], kw["height"], kw["fieldOfView"], kw["quality"], kw["antiAliasing"]) == (640, 360, 60,
                                                                                               "Ultra", 4)
    assert (kw["gridSize"], kw["rotateStepDegrees"], kw["snapToGrid"], kw["gpu_device"]) == (0.25, 45, False, 0)
    assert kw["platform"].__name__ == "CloudRendering" and kw["scene"] == "FloorPlan1"
    s = lvp.controller_settings(env._env)
    assert s["fieldOfView"] == 60 and s["antiAliasing"] == 4 and s["server_screenWidth"] == 640
    assert s["width"] == 640 and s["quality"] == "Ultra"
    t = lvp.make_thor_env(te, environ={})
    assert t.controller.kw["antiAliasing"] == 4 and t.controller.kw["fieldOfView"] == 60
    env._sync_close()
    assert c.stopped


# ================================================================================================================
# probe_lvp: pure functions
# ================================================================================================================
def test_percentile_and_dist():
    xs = [5.0, 1.0, 3.0, 2.0, 4.0, 10.0]
    for q in (0, 25, 50, 95, 100):
        assert probe_lvp.percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))
    assert probe_lvp.percentile([], 50) is None
    d = probe_lvp.dist([1, 2, 3])
    assert d["n"] == 3 and d["p50"] == 2.0 and d["max"] == 3.0
    assert probe_lvp.dist([])["p50"] is None


def test_jsafe_and_fnum():
    obj = {"a": float("inf"), "b": float("nan"), "c": (1, 2), "d": Path("/x"), "e": np.float32(1.5), "f": -np.inf}
    j = probe_lvp.jsafe(obj)
    assert j == {"a": "inf", "b": None, "c": [1, 2], "d": "/x", "e": 1.5, "f": "-inf"}
    json.dumps(j)
    assert probe_lvp.fnum("inf") == float("inf") and probe_lvp.fnum(3) == 3.0


def blocky(seed, h=360, w=640):
    small = np.random.default_rng(seed).integers(0, 256, (9, 16, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, h // 9, axis=0), w // 16, axis=1)


def test_frame_metrics_and_rule():
    a = blocky(1)
    m = probe_lvp.frame_metrics(a, a.copy(), probe_lvp.badge_mask())
    assert m["identical"] and m["psnr_db"] == float("inf") and m["ssim"] == 1.0 and probe_lvp.passes_rule(m)
    b = a.copy()
    b[:31, 540:] = 0  # badge region only
    assert probe_lvp.frame_metrics(a, b, probe_lvp.badge_mask())["identical"]
    assert not probe_lvp.frame_metrics(a, b)["identical"]  # unmasked it differs
    c = a.astype(np.int16)
    c[100:200, 100:300] += 40
    c = np.clip(c, 0, 255).astype(np.uint8)
    mc = probe_lvp.frame_metrics(a, c, probe_lvp.badge_mask())
    assert mc["max_abs"] == 40 and 0 < mc["frac_gt8"] < 1 and mc["psnr_db"] < 40 and not probe_lvp.passes_rule(mc)
    # "ssim" is skimage's RETURNED mssim on (public, ours with the badge pixels copied from public); the mean of
    # the full map over the unmasked pixels is reported next to it
    from skimage.metrics import structural_similarity
    cm = c.copy()
    cm[probe_lvp.badge_mask()] = a[probe_lvp.badge_mask()]
    assert mc["ssim"] == pytest.approx(structural_similarity(a, cm, channel_axis=2, data_range=255), abs=1e-6)
    assert "ssim_map_mean_unmasked" in mc and mc["ssim"] != mc["ssim_map_mean_unmasked"]

    def fake_ssim(mssim, map_value):
        return lambda x, y, **kw: (mssim, np.full(x.shape, map_value))

    m1 = probe_lvp.frame_metrics(a, a.copy(), probe_lvp.badge_mask(), ssim_fn=fake_ssim(0.995, 0.98))
    assert (m1["ssim"], m1["ssim_map_mean_unmasked"]) == (0.995, 0.98) and probe_lvp.passes_rule(m1)
    m2 = probe_lvp.frame_metrics(a, a.copy(), probe_lvp.badge_mask(), ssim_fn=fake_ssim(0.985, 0.999))
    assert not probe_lvp.passes_rule(m2)  # the rule reads the returned mssim, not the map mean
    assert "error" in probe_lvp.frame_metrics(a, a[:10])
    assert probe_lvp.passes_rule({"psnr_db": 40.0, "ssim": 0.99})
    assert probe_lvp.passes_rule({"psnr_db": "inf", "ssim": 1.0})
    assert not probe_lvp.passes_rule({"psnr_db": 39.999, "ssim": 1.0})
    assert not probe_lvp.passes_rule({"psnr_db": 50.0, "ssim": 0.98999})
    assert not probe_lvp.passes_rule({"error": "missing"})
    mask = probe_lvp.badge_mask()
    assert mask.shape == (360, 640) and mask[31, 536] and not mask[32, 536] and not mask[31, 535]


def write_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def _mini_compare(tmp_path, pub_change=None, succ=True):
    pub, ours = tmp_path / "pub", tmp_path / "ours"
    frames = {"step_00.png": blocky(1), "step_01.png": blocky(2), "target.png": blocky(3)}
    for n, arr in frames.items():
        write_png(ours / n, arr)
        write_png(pub / n, pub_change(n, arr) if pub_change else arr)
    write_png(ours / "rep_step_00.png", frames["step_00.png"])
    write_png(ours / "rep_target.png", frames["target.png"])
    traj = {"task_id": "t", "turns": [{"user_text": "u0"}, {"user_text": "u1"}], "actions": ["MoveAhead", "Stop"],
            "system": "S"}
    rep = {"turns": [{"user_text": "u0", "state": {"p": 0}, "action_success": succ},
                     {"user_text": "u1", "state": {"p": 1}}], "final_exact": True, "stop": {"success": True}}
    t = probe_lvp.compare_task(pub, ours, traj, rep, system_prompt="S")
    return t, probe_lvp.compare_verdict({"t": t}, 3)


def test_compare_same_different_incomplete(tmp_path):
    t, v = _mini_compare(tmp_path / "a")
    assert v["verdict"] == "same_rendering" and v["n_compared"] == 3 and v["controls"]["repeat"]["identical"]
    assert "returned mssim" in v["rule"] and "RETURNED by skimage" in v["ssim_definition"]
    assert v["min_ssim"] == 1.0 and v["min_ssim_map_mean_unmasked"] == 1.0
    assert v["controls"]["metric_separates_poses"] is True and t["system_prompt_equal"]
    assert t["valid_text_equal_all"] and t["moves_ok_all"] and v["final_exact_all"]

    def badge_only(n, arr):
        b = arr.copy()
        b[2:28, 541:637] = 60
        return b

    assert _mini_compare(tmp_path / "b", badge_only)[1]["verdict"] == "same_rendering"

    def body(n, arr):
        if n != "step_01.png":
            return arr
        b = arr.astype(np.int16)
        b[150:250, 100:400] += 25
        return np.clip(b, 0, 255).astype(np.uint8)

    t3, v3 = _mini_compare(tmp_path / "c", body)
    assert v3["verdict"] == "different_rendering" and v3["n_pass"] == 2
    # a failed move: frames after it are not comparable -> incomplete, and the move is flagged
    t4, v4 = _mini_compare(tmp_path / "d", succ=False)
    assert t4["diverged_from_step"] == 1 and v4["verdict"] == "incomplete" and not t4["moves_ok_all"]
    # a missing public frame is not comparable either
    (tmp_path / "a" / "pub" / "target.png").unlink()
    t6 = probe_lvp.compare_task(tmp_path / "a" / "pub", tmp_path / "a" / "ours",
                                {"task_id": "t", "turns": [{"user_text": "u0"}, {"user_text": "u1"}],
                                 "actions": ["MoveAhead", "Stop"]},
                                {"turns": [{"user_text": "u0", "state": 0, "action_success": True},
                                           {"user_text": "u1", "state": 1}]})
    assert probe_lvp.compare_verdict({"t": t6}, 3)["verdict"] == "incomplete"


def test_public_trajectory_and_sample(tmp_path):
    s = {"system": "S", "images": ["images/T1/step_00.png", "images/T1/target.png", "images/T1/step_01.png",
                                   "images/T1/target.png"],
         "conversations": [{"from": "human", "value": "h0"}, {"from": "gpt", "value": "Action: MoveAhead"},
                           {"from": "human", "value": "h1"}, {"from": "gpt", "value": "Action: Stop"}]}
    t = probe_lvp.public_trajectory(s)
    assert t["task_id"] == "T1" and t["actions"] == ["MoveAhead", "Stop"] and t["turns"][1]["user_text"] == "h1"
    with pytest.raises(ValueError):
        probe_lvp.public_trajectory(dict(s, images=s["images"][:3]))
    with pytest.raises(ValueError):
        probe_lvp.parse_action("I go MoveAhead")
    f = tmp_path / "sft_train.json"
    f.write_text(json.dumps([s]))
    assert list(probe_lvp.load_public(f, ["T1"])) == ["T1"]
    with pytest.raises(ValueError):
        probe_lvp.load_public(f, ["T1", "T2"])
    d = tmp_path / "sample"
    write_png(d / "images" / "T1" / "step_00.png", blocky(1))
    sha = hashlib.sha256((d / "images" / "T1" / "step_00.png").read_bytes()).hexdigest()
    (d / "SHA256SUMS").write_text(f"{sha}  images/T1/step_00.png\n")
    assert probe_lvp.verify_sample(d)["ok"]
    write_png(d / "images" / "T1" / "step_00.png", blocky(2))
    r = probe_lvp.verify_sample(d)
    assert not r["ok"] and r["bad"] == ["images/T1/step_00.png"]
    assert not probe_lvp.verify_sample(tmp_path / "none")["ok"]


def test_parse_grid_and_defaults():
    assert probe_lvp.parse_grid("2:1,2,3; 4:1,2") == {2: (1, 2, 3), 4: (1, 2)}
    for bad in ("3:1,2,3,4,5", "0:1", "2:", "1:0"):
        with pytest.raises(ValueError):
            probe_lvp.parse_grid(bad)
    g = probe_lvp.GRID_DEFAULT
    assert list(g) == [probe_lvp.REACH_L, 2, 3, 4] and max(g[probe_lvp.REACH_L]) == probe_lvp.REACH_E
    assert all(max(E) * L <= 12 for L, E in g.items())
    cells = {(E, L) for L, Es in g.items() for E in Es} | {(1, probe_lvp.P1_L)}
    spec = {(E, L) for E in (1, 2, 3, 4, 6) for L in (1, 2, 3, 4) if E * L <= 12} | {(1, 12)}
    assert spec <= cells  # the brief's grid, plus (5, 2)
    assert probe_lvp.STEPS == 100 and probe_lvp.MAX_EL_RECOMMEND == 11 and probe_lvp.MAX_CORES_RECOMMEND == 11


def test_cell_order(tmp_path):
    """Likely-best first: eligible E*L >= 6, then the other eligible cells, then E*L = 12; E descending per L
    group; a tier starts on the L the previous one ended on."""
    g = probe_lvp.GRID_DEFAULT
    p4 = probe_lvp.cell_order({L: E for L, E in g.items() if L != probe_lvp.REACH_L})
    assert [(E, L) for L, E in p4] == [(5, 2), (4, 2), (3, 2), (3, 3), (2, 3), (2, 4),
                                       (1, 4), (2, 2), (1, 2), (1, 3),
                                       (4, 3), (6, 2), (3, 4)]
    assert sorted(p4) == sorted((L, E) for L, Es in g.items() if L != probe_lvp.REACH_L for E in Es)
    assert [L * E for L, E in p4][-3:] == [12, 12, 12] and all(L * E <= 11 for L, E in p4[:-3])
    p2 = probe_lvp.cell_order({1: g[1]})
    assert [(E, L) for L, E in p2] == [(6, 1), (4, 1), (3, 1), (2, 1), (1, 1)]
    assert probe_lvp.cell_order({2: (1, 1, 2)}) == [(2, 2), (2, 1)] and probe_lvp.cell_order({}) == []
    p = probe_lvp.Probe(tmp_path / "o", deadline_s=10, spawner=None)
    assert p.p4_cells == p4 and p.p2_cells == p2
    assert p.res["plan"]["order_EL"]["p2"][0] == [6, 1] and p.res["plan"]["order_EL"]["p4"][-1] == [3, 4]


def _cell(E, L, scene, agg, ok=True, cores=1.0, **kw):
    return dict({"E": E, "L": L, "EL": E * L, "scene": scene, "agg_steps_per_s": agg, "ok": ok, "p50_ms": 1.0,
                 "p95_ms": 2.0, "cpu": {"cores": cores}}, **kw)


def test_grid_table_and_recommend():
    sc = ["A", "B"]
    env = [{"unity_cpu_s": 20.0, "worker_cpu_s": 2.0, "wall_s": 10.0}]
    cells = [_cell(1, 12, "A", 7.0, cores=13.0), _cell(1, 12, "B", 7.0, cores=12.0),   # E*L 12: not eligible
             _cell(2, 4, "A", 10.0, cores=10.0), _cell(2, 4, "B", 5.0, cores=9.0),     # hmean 6.667
             _cell(3, 3, "A", 8.0, cores=9.0, per_env=env), _cell(3, 3, "B", 8.0, cores=8.5),  # hmean 8.0
             _cell(5, 2, "A", 20.0), _cell(5, 2, "B", 1.0, ok=False),                    # incomplete
             _cell(4, 2, "A", 8.0, cores=10.0), _cell(4, 2, "B", 8.0, cores=12.5),     # ties 3x3, fewer threads
             _cell(1, 1, "A", 2.0, cores=None), _cell(1, 1, "B", 2.0, cores=1.5),     # cores not measured on A
             {"E": 6, "L": 1, "scene": "A", "skipped": "no_time", "ok": False}]
    rows = probe_lvp.grid_table(cells, sc)
    by = {(r["E"], r["L"]): r for r in rows}
    assert by[(2, 4)]["hmean_steps_per_s"] == pytest.approx(6.667, abs=1e-3)
    assert not by[(5, 2)]["ok"] and by[(5, 2)]["hmean_steps_per_s"] is None and (6, 1) not in by
    assert by[(3, 3)]["cores_max"] == 9.0 and by[(3, 3)]["cores_per_env"] == 3.0
    assert by[(3, 3)]["proc_cores_per_env"] == 2.2 and by[(4, 2)]["proc_cores_per_env"] is None
    assert by[(1, 1)]["cores_max"] is None and by[(1, 1)]["cores_per_env"] is None
    rec = probe_lvp.recommend(rows)
    assert (rec["E"], rec["L"]) == (4, 2) and rec["n_eligible"] == 4 and rec["cores_max"] == 12.5
    # measured cores: (4, 2) uses 12.5 > 11 cores and (1, 12) 13; (3, 3) at 9 cores wins
    rm = probe_lvp.recommend_measured(rows)
    assert (rm["E"], rm["L"], rm["cores_max"], rm["cores_per_env"]) == (3, 3, 9.0, 3.0)
    assert rm["n_eligible"] == 2 and rm["n_complete_measured"] == 4 and rm["n_complete_unmeasured"] == 1
    # with 13 cores allowed (4, 2) ties (3, 3) at 8.0 steps/s: fewer cores wins
    assert (probe_lvp.recommend_measured(rows, 13.0)["E"], probe_lvp.recommend_measured(rows, 13.0)["L"]) == (3, 3)
    assert probe_lvp.recommend_measured(rows, 1.0)["E"] is None
    assert probe_lvp.recommend(probe_lvp.grid_table(cells[:2], sc))["E"] is None
    # the pure function of a summary.json (limits from its plan; scenes from summary.scenes or the cells)
    s = {"plan": {"max_el_recommend": 11, "max_cores_recommend": 11.0}, "scenes": {"grid": sc}, "grid_cells": cells}
    rr = probe_lvp.recommendations(s)
    assert rr["recommended"] == rec and rr["recommended_measured"] == rm and rr["grid_table"] == rows
    assert rr["cores_per_env"]["E3xL3"] == {"cgroup": 3.0, "proc": 2.2} and "E5xL2" not in rr["cores_per_env"]
    assert probe_lvp.recommendations({"grid_cells": cells})["recommended"] == rec
    assert probe_lvp.recommendations(s, max_cores=13.0)["limits"] == {"max_el": 11, "max_cores": 13.0}
    assert probe_lvp.recommendations({})["recommended"]["E"] is None


def test_probe_poses_and_reach_digest():
    reach = [{"x": x / 4, "y": Y, "z": z / 4} for x in range(-4, 5) for z in range(-4, 5)]
    p = probe_lvp.pick_probe_poses(list(reversed(reach)), 4, extra=[pose(9, 9, 0, 0)])
    assert len(p) == 5 and p[0]["position"]["x"] == 9
    assert p == probe_lvp.pick_probe_poses(reach, 4, extra=[pose(9, 9, 0, 0)])  # order-independent
    assert [q["rotation_y"] for q in p[1:]] == [0.0, 90.0, 225.0, 315.0]
    assert [q["horizon"] for q in p[1:]] == [0.0, 30.0, -30.0, 0.0]
    assert len({(q["position"]["x"], q["position"]["z"]) for q in p[1:]}) == 4
    assert probe_lvp.pick_probe_poses([], 4) == []
    d1, d2 = probe_lvp.reach_digest(reach), probe_lvp.reach_digest(list(reversed(reach)))
    assert d1["n"] == 81 and d1["sha256_xz"] == d2["sha256_xz"] and d1["sha256_raw"] != d2["sha256_raw"]


def test_summarize_det():
    rows = [{"scene": "A", "pose": 0, "L": L, "wid": w, "sha256": "s0"} for L, w in ((12, 0), (2, 1), (2, 2))]
    rows += [{"scene": "A", "pose": 1, "L": L, "wid": w, "sha256": "s1"} for L, w in ((12, 0), (2, 1))]
    paths_ok, reload_ok = {"A": {"equal": True}}, {"A": {"equal": True}}
    d = probe_lvp.summarize_det(rows, paths_ok, reload_ok, {"A": True}, True)
    assert d["deterministic"] is True and d["equal_across_L"] and d["equal_across_processes"]
    assert d["n_flags_measured"] == 6
    bad = rows + [{"scene": "A", "pose": 1, "L": 3, "wid": 3, "sha256": "zz"}]
    d2 = probe_lvp.summarize_det(bad, paths_ok, reload_ok, {"A": True}, True)
    assert d2["equal_across_L"] is False and d2["deterministic"] is False
    d3 = probe_lvp.summarize_det([], {}, {}, {}, None)
    assert d3["deterministic"] is None and d3["n_flags_measured"] == 0
    d4 = probe_lvp.summarize_det(rows, {"A": {"equal": False}}, reload_ok, {"A": True}, True)
    assert d4["equal_across_paths"] is False and d4["deterministic"] is False
    # a path test with no valid move (or a worker error) is unmeasured, not False
    none = {"equal": None, "error": "no probe pose with a valid move"}
    d5 = probe_lvp.summarize_det(rows, {"A": none, "B": {"equal": True}}, {"A": {"equal": None}}, {"A": True}, True)
    assert d5["equal_across_paths"] is True and d5["equal_after_reload"] is None and d5["n_flags_measured"] == 5
    d6 = probe_lvp.summarize_det(rows, {"A": none}, reload_ok, {"A": True}, True)
    assert d6["equal_across_paths"] is None and d6["deterministic"] is True


# ================================================================================================================
# probe_lvp: the worker on TVRBench's code (fake ai2thor)
# ================================================================================================================
@needs_repo
def test_worker_ops(tvb, tmp_path):
    _, tv = tvb
    gz = write_gz(tmp_path / "pt" / "train.jsonl.gz")
    lvp.extract_procthor_houses(gz, ["procthor_00003"], tmp_path / "houses")
    w = probe_lvp.Worker(tv, wid=3, houses_dir=tmp_path / "houses", procthor_dir=tmp_path / "pt",
                         environ={"LP_NUM_THREADS": "2"})
    ready = w.start()
    assert ready["event"] == "ready" and ready["lp"] == "2" and ready["scene"] == "FloorPlan1"
    assert ready["settings"]["fieldOfView"] == 60
    r = probe_lvp.dispatch(w, {"id": 7, "op": "reach", "scene": "procthor_00003", "load": True})
    assert r["ok"] and r["id"] == 7 and r["n"] == 120 and r["load_s"] >= 0 and w.env._scene == "procthor_00003"
    assert w.env._env.controller.scene == "house:h00003"
    assert r["call_pose"]["position"] == {"x": 0.0, "y": 0.9, "z": 0.0}  # the agent pose at the call (spawn)
    w.op_render({"poses": [pose(0.5, 0.25, 90.0, 0.0)]})
    r2 = probe_lvp.dispatch(w, {"id": 9, "op": "reach", "scene": "procthor_00003", "load": False})
    assert r2["call_pose"]["position"]["x"] == 0.5 and r2["call_pose"]["rotation_y"] == 90.0 and "load_s" not in r2
    S = pose(0.0, 0.0, 0.0, 0.0)
    a = w.op_seq({"start": S, "actions": ["RotateRight", "RotateLeft", "MoveAhead"]})
    b = w.op_seq({"start": S, "actions": ["MoveAhead", "LookDown", "LookUp"]})
    c = w.op_render({"poses": [probe_lvp.pose_of(a["state"])]})["frames"][0]
    assert a["succ"] == [True] * 3 and a["state"] == b["state"] and a["sha256"] == b["sha256"] == c["sha256"]
    assert "MoveAhead" in a["valid_start"] and "Stop" not in a["valid_start"]
    st = w.op_steps({"n": 4, "go_at": None, "seed": 5, "start": S})
    assert st["n"] == 4 and len(st["lat_ms"]) == 4 and st["bad_frames"] == 0 and st["n_ok"] == 4
    assert sum(st["actions"].values()) == 4 and st["t_end"] >= st["t_go"]
    st2 = w.op_steps({"n": 4, "go_at": None, "seed": 5, "start": S})
    assert st2["actions"] == st["actions"]  # seeded
    bd = w.op_badge({})
    assert bd["TARGET"]["bbox_bg"] is not None and bd["TARGET"]["bbox_bg"][2] == 636
    r = probe_lvp.dispatch(w, {"id": 8, "op": "nope"})
    assert not r["ok"] and "unknown op" in r["error"] and r["id"] == 8
    w.close()


@needs_repo
def test_path_test_detects_history_dependence(tvb, tmp_path):
    """A renderer whose frame depends on the step history (a temporal-AA stand-in) fails the path test."""
    _, tv = tvb
    ctrl = sys.modules["ai2thor.controller"].Controller
    w = probe_lvp.Worker(tv, environ={})
    w.start()
    h = probe_lvp.LocalHandle(0, 2, w)
    p = probe_lvp.Probe(tmp_path / "o", deadline_s=100, spawner=None)
    p.probe_poses["FloorPlan1"] = [pose(0.0, 0.0, 0.0, 0.0)]
    good = p.path_test(h, "FloorPlan1")
    assert good["equal"] is True and good["all_succ"] and good["states_equal"]
    ctrl.HISTORY = True
    try:
        bad = p.path_test(h, "FloorPlan1")
    finally:
        ctrl.HISTORY = False
    assert bad["equal"] is False and bad["states_equal"]
    p.probe_poses["X"] = []
    assert p.path_test(h, "X")["equal"] is None


# ================================================================================================================
# probe_lvp: the whole probe in-process (LocalHandle workers on TVRBench + fake ai2thor)
# ================================================================================================================
@needs_repo
def test_probe_end_to_end(tvb, tmp_path):
    _, tv = tvb
    w = make_world(tmp_path, tv)
    p, out = make_probe(tmp_path, w, tv)
    res = p.run()
    s = json.loads((out / "summary.json").read_text())
    assert s["schema"] == probe_lvp.SCHEMA and s["label"] == "EXPLORATORY"
    assert {k: v["status"] for k, v in s["phases"].items()} == {p: "ok" for p in probe_lvp.PHASES}, s["errors"]
    assert s["errors"] == []
    # (f) comparison
    c = s["compare"]
    assert c["verdict"] == "same_rendering" and c["n_compared"] == c["n_expected"] == 10
    assert c["valid_text_equal_all"] and c["moves_ok_all"] and c["final_exact_all"]
    assert c["controls"]["repeat"]["identical"] and c["controls"]["metric_separates_poses"]
    assert c["sample"]["ok"] and all(t["system_prompt_equal"] for t in c["tasks"].values())
    assert sorted(f.name for f in (out / "compare" / FP_TID).glob("*.png")) == [
        "rep_step_00.png", "rep_target.png", "step_00.png", "step_01.png", "step_02.png", "step_03.png", "target.png"]
    assert c["ssim_definition"] == probe_lvp.SSIM_DEFINITION and c["min_ssim"] == 1.0
    # (a) grid: P1 cell (1, 4) on both scenes, then L=2 in cell_order: E=2 (new pool: first grid scene first),
    # then E=1 starting on the scene E=2 ended on
    cells = s["grid_cells"]
    assert [(x["E"], x["L"], x["scene"]) for x in cells] == [
        (1, 4, "FloorPlan1"), (1, 4, "procthor_00003"), (2, 2, "FloorPlan1"), (2, 2, "procthor_00003"),
        (1, 2, "procthor_00003"), (1, 2, "FloorPlan1")]
    assert s["plan"]["order_EL"] == {"p2": [], "p4": [[2, 2], [1, 2]]}
    assert all(x["ok"] for x in cells) and all(len(x["per_env"]) == x["E"] for x in cells)
    assert all(x["cpu"]["cores"] == 0.0 for x in cells)  # the static fake cpu.stat
    assert [x["idle_workers"] for x in cells] == [0, 0, 0, 0, 1, 1]  # E=1 on the 2-worker L=2 pool
    assert s["recommended"]["E"] is not None and s["recommended"]["EL"] <= 11
    rm = s["recommended_measured"]
    assert rm["E"] is not None and rm["cores_max"] == 0.0 and rm["n_eligible"] == 3
    assert s["cores_per_env"]["E2xL2"]["cgroup"] == 0.0
    # (lvp profile) the first worker started under "envlib", which every worker then used
    lp = s["lvp_profile"]
    assert lp["profile"] == "envlib" and not lp["retried"] and lp["LD_LIBRARY_PATH"] == FAKE_LIB
    assert len(lp["attempts"]) == 1 and lp["attempts"][0]["ok"]
    assert {wk["env_profile"] for wk in s["workers"]} == {"envlib"}
    # (b) determinism
    d = s["determinism"]
    assert d["deterministic"] is True and d["n_flags_measured"] == 6, d
    assert d["equal_across_L"] and d["equal_across_processes"] and d["reach_equal_across_two_loads"]
    assert set(d["paths"]) == {"FloorPlan1", "procthor_00003"} and all(v["equal"] for v in d["paths"].values())
    assert d["probe_poses"]["procthor_00003"][0] == EVAL_TASKS[1]["start"]
    # (c) load times, (d) reach, (e) font
    assert set(s["load_times"]["by_kind"]) == {"ithor", "procthor"}
    r = s["reach"]
    assert r["planned"] == r["both_loads"] == r["equal_raw"] == 5 and r["unequal"] == [] and r["L"] == [1]
    files = sorted(f.stem for f in (out / "reach").glob("*.json"))
    assert files == sorted(["FloorPlan11", "procthor_00002", "FloorPlan12", "FloorPlan13", "procthor_00003"])
    doc = json.loads((out / "reach" / "procthor_00002.json").read_text())
    assert doc["equal_raw"] and not doc["same_worker"] and len(doc["positions"]) == 120 and doc["split"] == ["sft"]
    assert [ld["call_pose"]["position"] for ld in doc["loads"]] == [{"x": 0.0, "y": 0.9, "z": 0.0}] * 2
    f = s["font"]
    assert f["ok"] and f["worker_modes"] == ["pinned"] and f["worker_sha256"] == [w.font_sha]
    assert f["path"] == str(w.font) and f["sha256"] == f["expected_sha256"] == w.font_sha
    assert s["houses"]["n"] == 2 and s["homes"]["bytes"] == 0  # in-process workers have no HOME
    assert len(s["workers"]) == 1 + 2 + 6 and all(wk["ok"] for wk in s["workers"])
    e = probe_lvp.expect(out)
    assert e["ok"], e["missing_required"]
    assert json.loads((out / "expect.json").read_text())["ok"]
    assert res["recommended"] == s["recommended"]
    # recommendations are a pure function of summary.json; the CLI recomputes them
    assert probe_lvp.recommendations(s)["recommended_measured"] == s["recommended_measured"]
    assert probe_lvp.main(["recommend", "--out", str(out), "--write", str(tmp_path / "rec.json")]) == 0
    rec = json.loads((tmp_path / "rec.json").read_text())
    assert rec["recommended"] == s["recommended"] and rec["recommended_measured"] == rm
    assert probe_lvp.main(["recommend", "--out", str(out), "--max-cores", "-1"]) == 0
    assert probe_lvp.main(["recommend", "--out", str(out), "--write", str(out / "summary.json")]) == 2
    assert probe_lvp.main(["recommend", "--out", str(tmp_path / "nothing")]) == 1
    # expect: only an eligible skipped cell is fatal; a skipped E*L = 12 cell is informational
    s12 = dict(s, grid_cells=s["grid_cells"] + [{"E": 3, "L": 4, "EL": 12, "scene": "FloorPlan1",
                                                 "skipped": "no_time", "ok": False}])
    (out / "summary.json").write_text(json.dumps(s12))
    e12 = probe_lvp.expect(out)
    assert e12["ok"] and [i["ok"] for i in e12["items"] if i["name"] == "grid.no_skipped"] == [False]
    s8 = dict(s12, grid_cells=s12["grid_cells"] + [{"E": 4, "L": 2, "EL": 8, "scene": "FloorPlan1",
                                                    "skipped": "no_time", "ok": False}])
    (out / "summary.json").write_text(json.dumps(s8))
    assert probe_lvp.expect(out)["missing_required"] == ["grid.no_eligible_skipped"]


@needs_repo
def test_probe_detects_different_public_rendering(tvb, tmp_path):
    _, tv = tvb

    def brighten(f):
        a = np.asarray(Image.open(f).convert("RGB")).astype(np.int16)
        a[60:300, 0:500] += 12
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(f, format="PNG")

    w = make_world(tmp_path, tv, perturb=brighten)
    p, out = make_probe(tmp_path, w, tv, grid={})
    p.reach_override = ["FloorPlan11"]
    p.run()
    s = json.loads((out / "summary.json").read_text())
    c = s["compare"]
    assert c["verdict"] == "different_rendering" and c["n_pass"] == 0 and c["min_psnr_db"] < 40
    assert c["valid_text_equal_all"] and c["moves_ok_all"]
    assert c["controls"]["repeat"]["identical"]
    assert s["reach"]["planned"] == 1


class LateProbe(probe_lvp.Probe):
    """Fake clock: frozen, then 60 s before the deadline once the reach dumps are done."""
    T = [0.0]

    def p3_reach(self):
        super().p3_reach()
        LateProbe.T[0] = self.deadline_s - 60.0


@needs_repo
def test_probe_deadline_skips_cells_and_keeps_reach(tvb, tmp_path):
    _, tv = tvb
    w = make_world(tmp_path, tv)
    LateProbe.T[0] = 0.0
    p, out = make_probe(tmp_path, w, tv, cls=LateProbe, clock=lambda: LateProbe.T[0],
                        grid={1: (1, 2), 2: (1, 2)})
    p.run()
    s = json.loads((out / "summary.json").read_text())
    assert [(x["E"], x["L"], x["scene"][:5], bool(x.get("ok")), x.get("skipped")) for x in s["grid_cells"]] == [
        (1, 4, "Floor", True, None), (1, 4, "proct", True, None),                          # P1
        (2, 1, "Floor", True, None), (2, 1, "proct", True, None),                          # P2, E descending
        (1, 1, "proct", True, None), (1, 1, "Floor", True, None),
        (2, 2, "Floor", False, "no_time"), (2, 2, "proct", False, "no_time"),             # P4, cut
        (1, 2, "proct", False, "no_time"), (1, 2, "Floor", False, "no_time")]
    r = s["reach"]
    assert r["both_loads"] == 5 and r["L"] == [1] and r["workers"] == [1, 2, 3, 4, 5, 6]  # reused + 4 new
    assert s["phases"]["p4_grid"]["status"] == "ok" and s["recommended"]["L"] in (1, 4)
    e = probe_lvp.expect(out)
    assert not e["ok"] and e["missing_required"] == ["grid.no_eligible_skipped"]  # E*L 2 and 4 are eligible


@needs_repo
def test_probe_lvp_profile_retry(tvb, tmp_path):
    """The first worker fails under "envlib": one retry under "envlib+orig", which every later worker uses."""
    _, tv = tvb
    w = make_world(tmp_path, tv)
    calls = []
    p, out = make_probe(tmp_path, w, tv, grid={}, fail=lambda wid, L, prof: prof == "envlib", spawn_log=calls)
    p.reach_override = ["FloorPlan11"]
    p.run()
    s = json.loads((out / "summary.json").read_text())
    assert calls[0] == (0, 4, "envlib") and calls[1] == (1, 4, "envlib+orig")
    assert all(c[2] == "envlib+orig" for c in calls[1:]) and len(calls) == 1 + 1 + 6
    lp = s["lvp_profile"]
    assert lp["profile"] == "envlib+orig" and lp["retried"] and lp["LD_LIBRARY_PATH"] == FAKE_LIB + ":/orig/lib"
    assert [(a["env_profile"], a["ok"]) for a in lp["attempts"]] == [("envlib", False), ("envlib+orig", True)]
    assert [(wk["env_profile"], wk["ok"]) for wk in s["workers"][:2]] == [("envlib", False), ("envlib+orig", True)]
    assert all(wk["ok"] and wk["env_profile"] == "envlib+orig" for wk in s["workers"][1:])
    assert [e["where"] for e in s["errors"]] == ["spawn.profile"]
    assert s["phases"]["p1_compare_det"]["status"] == "ok" and s["compare"]["verdict"] == "same_rendering"
    assert [i["ok"] for i in probe_lvp.expect(out)["items"] if i["name"] == "lvp_profile"] == [True]


@needs_repo
def test_probe_lvp_profile_no_retry_after_first_start(tvb, tmp_path):
    """Only the FIRST worker's start-up may switch the profile; both failing leaves the profile undecided."""
    _, tv = tvb
    w = make_world(tmp_path, tv)
    calls = []
    p, out = make_probe(tmp_path / "a", w, tv, grid={}, fail=lambda wid, L, prof: wid == 1, spawn_log=calls)
    p.reach_override = ["FloorPlan11"]
    p.run()
    s = json.loads((out / "summary.json").read_text())
    assert [c[2] for c in calls] == ["envlib", "envlib"]  # wid 1 (the reach pool) fails: no retry, no switch
    assert s["lvp_profile"]["profile"] == "envlib" and not s["lvp_profile"]["retried"]
    assert s["phases"]["p3_reach"]["status"] == "error"
    calls2 = []
    p2, out2 = make_probe(tmp_path / "b", w, tv, grid={}, fail=lambda *a: True, spawn_log=calls2)
    p2.reach_override = ["FloorPlan11"]
    p2.run()
    s2 = json.loads((out2 / "summary.json").read_text())
    assert [c[2] for c in calls2] == ["envlib", "envlib+orig", "envlib+orig"]  # P1 twice, then P3 once
    assert s2["lvp_profile"]["profile"] is None and len(s2["lvp_profile"]["attempts"]) == 2
    assert s2["phases"]["p1_compare_det"]["status"] == "error"
    e2 = probe_lvp.expect(out2)
    assert "lvp_profile" in e2["missing_required"] and "font.ok" in e2["missing_required"]


class SeqFailHandle(probe_lvp.LocalHandle):
    """Records every op; 'seq' (the path tests' ThorEnv.reset + steps) fails like a refused Teleport."""
    ops: list = []

    def send(self, cmd):
        SeqFailHandle.ops.append(cmd.get("op"))
        if cmd.get("op") == "seq":
            self._next += 1
            self._replies[self._next] = {"id": self._next, "ok": False,
                                         "error": "RuntimeError('fake Teleport failure')"}
            return self._next
        return super().send(cmd)


@needs_repo
def test_probe_p1_compare_first_and_guarded(tvb, tmp_path):
    """The G3(d) comparison runs first; a failing path test is recorded as unmeasured and P1 goes on."""
    _, tv = tvb
    w = make_world(tmp_path, tv)
    SeqFailHandle.ops = []
    p, out = make_probe(tmp_path, w, tv, grid={}, handle_cls=SeqFailHandle)
    p.reach_override = ["FloorPlan11"]
    p.run()
    s = json.loads((out / "summary.json").read_text())
    assert SeqFailHandle.ops[:3] == ["replay", "replay", "badge"]  # right after the start-up
    assert s["phases"]["p1_compare_det"]["status"] == "ok" and s["compare"]["verdict"] == "same_rendering"
    d = s["determinism"]
    assert all(v["equal"] is None and "fake Teleport" in v["error"] for v in d["paths"].values())
    assert d["equal_across_paths"] is None and d["equal_after_reload"] is True  # unmeasured, not False
    assert [e["where"] for e in s["errors"]] == ["p1.path.FloorPlan1", "p1.path.procthor_00003"]
    assert [(x["E"], x["L"], x["ok"]) for x in s["grid_cells"]] == [(1, 4, True), (1, 4, True)]  # P1 cell ran


class TermHandle(probe_lvp.LocalHandle):
    def send(self, cmd):
        if cmd.get("op") == "steps" and self.L == 2:
            raise probe_lvp.Terminated("signal 15")
        return super().send(cmd)


@needs_repo
def test_probe_terminated_writes_summary(tvb, tmp_path):
    _, tv = tvb
    w = make_world(tmp_path, tv)
    p, out = make_probe(tmp_path, w, tv, handle_cls=TermHandle)
    res = p.run()
    s = json.loads((out / "summary.json").read_text())
    assert s["terminated"] and s["phases"]["p4_grid"]["status"] == "terminated"
    assert s["phases"]["p3_reach"]["status"] == "ok" and s["reach"]["both_loads"] == 5
    assert res["compare"]["verdict"] == "same_rendering" and p.pool == []
    e = probe_lvp.expect(out)
    assert not e["ok"] and "no_terminated" in e["missing_required"]


@needs_repo
def test_probe_without_icd_renders_nothing(tvb, tmp_path):
    _, tv = tvb
    w = make_world(tmp_path, tv)
    calls = []
    p, out = make_probe(tmp_path, w, tv)
    p.icd = tmp_path / "no_icd.json"
    p.spawner = lambda wid, L, profile=None: calls.append((wid, L, profile))
    p.run()
    s = json.loads((out / "summary.json").read_text())
    assert calls == [] and list(s["phases"]) == ["p0_preflight"] and s["icd"]["ok"] is False
    assert any(e["where"] == "p0.icd" for e in s["errors"])
    assert not probe_lvp.expect(out)["ok"]


def test_expect_and_main_refusals(tmp_path):
    e = probe_lvp.expect(tmp_path / "nothing")
    assert not e["ok"] and e["missing_required"] == ["summary.json"]
    d = tmp_path / "o"
    d.mkdir()
    (d / "summary.json").write_text("{}")
    assert probe_lvp.main(["run", "--out", str(d), "--deadline-s", "10"]) == 2
    assert probe_lvp.main(["expect", "--out", str(d)]) == 1


# ================================================================================================================
# the worker subprocess protocol (real pipes, real child process; TVRBench on the fake ai2thor)
# ================================================================================================================
@needs_repo
def test_subprocess_worker_protocol(fake_site, tmp_path):
    fake_env = tmp_path / "fake_tvr_env"
    (fake_env / "lib").mkdir(parents=True)
    rel = tmp_path / "releases"
    rel.mkdir()
    gz = write_gz(tmp_path / "pt" / "train.jsonl.gz")
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(fake_site),
            "PYTHONDONTWRITEBYTECODE": "1", "LP_NUM_THREADS": "99", "VK_ICD_FILENAMES": "/x/nvidia_icd.json",
            lvp.ORIG_LD_VAR: str(tmp_path / "orig_a") + "::" + str(tmp_path / "orig_b")}
    out = tmp_path / "out"
    sp = probe_lvp.SubprocessSpawner(out, py=sys.executable, repo=REPO, houses_dir=None,
                                     procthor_dir=gz.parent, base_env=base, startup_timeout=90,
                                     tvr_env=fake_env, releases=rel)
    h, ready = sp(0, 2)
    try:
        assert ready["ok"], ready
        assert ready["lp"] == "2" and ready["env"]["LP_NUM_THREADS"] == "2"
        assert ready["env"]["CUDA_VISIBLE_DEVICES"] == "0" and ready["env"]["LD_LIBRARY_PATH"] == str(fake_env / "lib")
        assert ready["env"]["VK_ICD_FILENAMES"] == str(fake_env / "share/vulkan/icd.d/lvp_icd.x86_64.json")
        assert ready["env"]["HOME"] == str(out / "homes" / "w00_L2") and ready["env"]["OMP_NUM_THREADS"] == "1"
        assert ready["env"]["VPD_LVP_PROFILE"] == "envlib" and ready["env_profile"] == "envlib"
        assert ready["spawn_env"]["LD_LIBRARY_PATH"] == str(fake_env / "lib")
        assert ready["settings"]["fieldOfView"] == 60 and ready["font_mode"] in ("pinned", "mismatch", "missing")
        assert ready["font"]["path"] == str(lvp.BADGE_FONT) and ready["font"]["expected_sha256"] == \
            lvp.BADGE_FONT_SHA256
        home = out / "homes" / "w00_L2" / ".ai2thor"
        assert os.readlink(home / "releases") == str(rel)
        assert json.loads((home / "cuda-vulkan-mapping.json").read_text()) == {"0": 0}
        r = h.wait(h.send({"op": "load", "scene": "procthor_00002"}), 60)  # TVRBench's own gz loader
        assert r["ok"] and r["scene"] == "procthor_00002", r
        r = h.wait(h.send({"op": "steps", "n": 2, "go_at": None, "seed": 1, "start": pose(0.0, 0.0, 0.0, 0.0)}), 60)
        assert r["ok"] and len(r["lat_ms"]) == 2, r
        r = h.wait(h.send({"op": "nope"}), 30)
        assert not r["ok"] and "unknown op" in r["error"]
        q = h.quit(20)
        assert q.get("event") == "bye", q
        assert h.proc.poll() is not None
    finally:
        h.kill()
    assert (out / "logs" / "worker_w00_L2.log").is_file()
    h2, ready2 = sp(1, 1, "envlib+orig")  # the measured profile appends the container's LD_LIBRARY_PATH
    try:
        assert ready2["ok"], ready2
        assert ready2["env"]["LD_LIBRARY_PATH"] == ":".join([str(fake_env / "lib"), str(tmp_path / "orig_a"),
                                                             str(tmp_path / "orig_b")])
        assert ready2["env"]["VPD_LVP_PROFILE"] == "envlib+orig" == ready2["env_profile"]
        os.killpg(h2.pid, signal.SIGKILL)
        r = h2.wait(h2.send({"op": "load", "scene": "FloorPlan2"}), 20)
        assert not r["ok"] and (r.get("died") or "send failed" in str(r.get("error"))), r
        assert not h2.alive
    finally:
        h2.kill()


@needs_repo
def test_subprocess_worker_startup_failure(fake_site, tmp_path):
    """A worker that cannot import TVRBench answers reply 0 with ok false and is killed."""
    sp = probe_lvp.SubprocessSpawner(tmp_path / "out", py=sys.executable, repo=tmp_path / "no_repo",
                                     base_env={"PATH": os.environ.get("PATH", "/usr/bin"),
                                               "PYTHONDONTWRITEBYTECODE": "1"},
                                     startup_timeout=60, tvr_env=tmp_path / "e", releases=tmp_path / "r")
    h, ready = sp(0, 1)
    assert not ready["ok"] and "Error" in ready["error"]
    assert not h.alive
    assert ready["env_profile"] == "envlib" and ready["spawn_env"]["LD_LIBRARY_PATH"] == str(tmp_path / "e" / "lib")


# ================================================================================================================
# the sco payload
# ================================================================================================================
FAKE_COMMON = """# fake common_env.sh (vpd tests)
set -euo pipefail
export VPD_WORKTREE={wt}
export VPD_OUT={vout}
export PYTHONPATH="$VPD_WORKTREE"
vpd_chown_trap() {{ return 0; }}
vpd_node_facts() {{ echo facts > "$1"; }}
vpd_step() {{
  local name=$1 rc had_e=0
  shift
  [[ $- == *e* ]] && had_e=1
  set +e
  "$@"
  rc=$?
  ((had_e)) && set -e
  printf '{{"name":"%s","rc":%d}}\\n' "$name" "$rc" >> "$RUN/steps.jsonl"
  return $rc
}}
"""

FAKE_PY = """#!/usr/bin/env bash
# fake TVR interpreter: '-s [-u] -m vpd.tvr.probe_lvp run|expect --out DIR [...]'
echo "$PWD|$*" >> "$FAKE_LOG"
echo "${VPD_TVR_ORIG_LD_LIBRARY_PATH-<unset>}|${LD_LIBRARY_PATH-<unset>}" >> "$FAKE_LOG.ld"
args=("$@")
cmd=""
out=""
for ((i = 0; i < ${#args[@]}; i++)); do
  case ${args[i]} in
    run|expect) cmd=${args[i]} ;;
    --out) out=${args[i+1]} ;;
  esac
done
if [[ $cmd == run ]]; then
  echo '{"schema": "fake"}' > "$out/summary.json"
  exit "${FAKE_RUN_RC:-0}"
fi
if [[ $cmd == expect ]]; then
  echo '{"ok": true}' > "$out/expect.json"
  exit "${FAKE_EXPECT_RC:-0}"
fi
exit 9
"""


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_tree(tmp_path):
    wt = tmp_path / "wt"
    jobs = wt / "vpd" / "jobs"
    jobs.mkdir(parents=True)
    shutil.copy(PAYLOAD, jobs / "tvr_probe.sh")
    (jobs / "common_env.sh").write_text(FAKE_COMMON.format(wt=wt, vout=tmp_path / "VOUT"))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "pgrep", "#!/usr/bin/env bash\nexit 1\n")  # never touch real processes
    _exe(bin_ / "pkill", "#!/usr/bin/env bash\nexit 1\n")
    py = _exe(tmp_path / "fakepy", FAKE_PY)
    env = {"PATH": f"{bin_}:{os.environ.get('PATH', '/usr/bin:/bin')}", "VPD_TVR_PY": str(py),
           "FAKE_LOG": str(tmp_path / "fake.log"), "HOME": str(tmp_path), "LP_NUM_THREADS": "64"}
    return types.SimpleNamespace(wt=wt, script=jobs / "tvr_probe.sh", env=env, log=tmp_path / "fake.log",
                                 run=tmp_path / "RUN")


def _run_payload(t, *args, **extra):
    env = dict(t.env, **extra)
    return subprocess.run(["bash", str(t.script), *args], env=env, capture_output=True, text=True, timeout=120)


def _steps(run):
    return [(json.loads(x)["name"], json.loads(x)["rc"]) for x in (run / "steps.jsonl").read_text().splitlines()]


def test_payload_static():
    r = subprocess.run(["bash", "-n", str(PAYLOAD)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = PAYLOAD.read_text()
    assert 'source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"' in text
    assert '\ncd "$VPD_WORKTREE"' in text
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "VAGEN-Lite-vpd" in ln]  # only the prepare command in the header names it
    assert not [ln for ln in code if "vllm" in ln.lower()]
    for f in (paths.VPD_PKG / "tvr" / "lvp.py", paths.VPD_PKG / "tvr" / "probe_lvp.py"):
        src = f.read_text()
        assert "VAGEN-Lite-vpd" not in src and "cpu_count" not in src and "import vllm" not in src


def test_payload_runs_steps_and_done(fake_tree):
    t = fake_tree
    r = _run_payload(t, str(t.run), LD_LIBRARY_PATH="/c/lib:/d")
    assert r.returncode == 0, r.stdout + r.stderr
    # the container's LD_LIBRARY_PATH reaches the probe as VPD_TVR_ORIG_LD_LIBRARY_PATH (lvp's "envlib+orig")
    assert Path(str(t.log) + ".ld").read_text().splitlines()[0] == "/c/lib:/d|/c/lib:/d"
    assert _steps(t.run) == [("node", 0), ("probe", 0), ("reap", 0), ("expect", 0)]
    assert (t.run / "DONE").is_file() and (t.run / "out" / "summary.json").is_file()
    assert (t.run / "node.txt").read_text().strip() == "facts"
    calls = t.log.read_text().splitlines()
    assert len(calls) == 2
    cwd, argv = calls[0].split("|", 1)
    assert cwd == str(t.wt)
    parts = argv.split()
    assert parts[:5] == ["-s", "-u", "-m", "vpd.tvr.probe_lvp", "run"]
    assert parts[parts.index("--out") + 1] == str(t.run / "out")
    left = int(parts[parts.index("--deadline-s") + 1])
    assert 3360 - 90 - 5 <= left <= 3360 - 90  # default soft deadline 3360 s (est-gph 1.0), reserve 90 s
    assert calls[1].split("|", 1)[1].split()[:5] == ["-s", "-m", "vpd.tvr.probe_lvp", "expect", "--out"]
    # a second run into the same RUN is refused
    r2 = _run_payload(t, str(t.run))
    assert r2.returncode == 2 and "refusing" in r2.stderr


def test_payload_grid_and_steps_passthrough(fake_tree):
    t = fake_tree
    r = _run_payload(t, str(t.run), VPD_TVR_GRID="2:1,2;3:1", VPD_TVR_STEPS="60")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = t.log.read_text().splitlines()[0].split("|", 1)[1].split()
    assert argv[argv.index("--grid") + 1] == "2:1,2;3:1" and argv[argv.index("--steps") + 1] == "60"
    r2 = _run_payload(t, str(t.run) + "z", VPD_TVR_STEPS="x")
    assert r2.returncode == 5 and ("probe", 2) in _steps(Path(str(t.run) + "z"))
    # without the variables the probe gets neither flag (fresh tree log)
    t.log.unlink()
    Path(str(t.log) + ".ld").unlink()
    _run_payload(t, str(t.run) + "w")
    first = t.log.read_text().splitlines()[0]
    assert "--grid" not in first and "--steps" not in first
    assert Path(str(t.log) + ".ld").read_text().splitlines()[0] == "|<unset>"  # no container value: set, empty


def test_payload_expect_failure_no_done(fake_tree):
    t = fake_tree
    r = _run_payload(t, str(t.run), FAKE_EXPECT_RC="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _steps(t.run)[-1] == ("expect", 1) and not (t.run / "DONE").exists()


def test_payload_short_deadline_and_usage(fake_tree):
    t = fake_tree
    r = _run_payload(t, str(t.run), VPD_JOB_DEADLINE_S="200", FAKE_EXPECT_RC="1")
    assert r.returncode == 5, r.stdout + r.stderr
    assert ("probe", 3) in _steps(t.run) and not (t.run / "DONE").exists()
    assert "probe_lvp run" not in t.log.read_text()  # the probe never started
    assert _run_payload(t).returncode == 2
    assert _run_payload(t, str(t.run) + "x", VPD_JOB_DEADLINE_S="abc").returncode == 2
    r3 = _run_payload(t, str(t.run) + "y", VPD_TVR_PY=str(t.wt / "missing_py"))
    assert r3.returncode == 5 and ("probe", 4) in _steps(Path(str(t.run) + "y"))
