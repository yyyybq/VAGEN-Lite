"""Shared TVR test helpers (CONTRACT_wave2 5.1.8): a fake ai2thor package and a fake TVR data tree.

Used by test_tvr_data.py, test_tvr_eval.py, test_tvr_jobs.py and test_tvr_integration.py.  No Unity, no GPU, no
network: the REAL TVRBench code (thor_env / thor_view_env of the pinned checkout, lvp.TVR_REPO) runs on a fake
ai2thor whose Controller keeps a pose, answers GetReachablePositions from a JSON scene table and renders a frame
that is a pure function of (scene, pose).

write_fake_ai2thor(root, scenes) -> root
    writes root/ai2thor/{__init__.py, controller.py, platform.py} and root/fake_scenes.json; prepend root to
    sys.path (import_real_tvrbench does) or to the PYTHONPATH of a worker subprocess.  scenes:
      {scene_id: {"positions": [{x, y, z}, ...],          # GetReachablePositions (default: DEFAULT_GRID)
                  "by_start": {"x,z": [positions]},       # optional pose dependence: the answer when the agent
                                                          # stands at cell "x,z" (key = start_key(x, z), "%.2f,%.2f")
                  "spawn": pose_dict,                     # the pose after reset(scene) (default: the first
                                                          # position, rotation 0, horizon 0)
                  "fail_teleport": ["x,z", ...]}}         # optional: Teleport to these cells fails (pose kept)
    A scene id not in the table gets DEFAULT_GRID (so ThorEnv's placeholder FloorPlan1 always works).  A ProcTHOR
    house dict maps to its scene id through house["id"].  The table is re-read at every Controller() and reset().
Fake Controller semantics (modeled on test_tvr_probe.py's FAKE_CONTROLLER):
    Controller(**kw) stores kw (width, height, fieldOfView, quality, antiAliasing, gridSize, rotateStepDegrees,
    platform, gpu_device ...) and initialization_parameters; step(action=..., **kw) / step("GetReachablePositions"):
      Teleport      sets the pose (position, rotation.y, horizon); without forceAction=True a cell outside the
                    positions fails; a fail_teleport cell always fails (lastActionSuccess False, pose kept);
      RotateRight / RotateLeft   +/- rotateStepDegrees (mod 360);  LookUp horizon -30;  LookDown +30 (no clamp:
                    ThorEnv clamps);
      GetReachablePositions      actionReturn = the scene positions, or by_start[start_key(current x, z)];
      anything else fails ("unknown action").
    reset(scene=str|dict) sets the scene and the spawn pose; stop() sets .stopped.
    last_event.metadata = {"agent": {"position", "rotation": {"x": 0, "y", "z": 0}, "cameraHorizon"},
    "lastActionSuccess", "errorMessage", "actionReturn", "screenWidth", "screenHeight", "fov"};
    last_event.frame = frame_of(scene, pose) (read-only array; ThorEnv copies it).  Controller.created lists every
    instance; each instance counts .n_steps and .n_reach (GetReachablePositions calls).  ai2thor.platform has
    CloudRendering.

import_real_tvrbench(fake_root) -> (thor_env, thor_view_env) of the pinned checkout on the fake (purges earlier
    ai2thor / tvrbench modules first; purge_tvrbench() undoes it).
fake_task(task_id, scene, dataset, difficulty, start, target) -> a TVRBench task dict (start / target = (x, z,
    rot, hor) tuples).  grid_positions(xs, zs, blocked=(), y=0.9) -> positions.  frame_of(scene, pose) -> the
    (360, 640, 3) uint8 frame (a writable copy).  public_record(task, actions, positions=None) -> one sft_train.json
    record whose valid lines are kin_env's on `positions` (None: free space).
write_fake_tvr_tree(root, tasks, public) -> {"root", "repo_data", "tasks_dir", "procthor_dir", "procthor_gz",
    "sft_train", "houses": {scene: house}, "houses_dir", "scenes": {split: [...]}}: data/tasks/{sft,eval,rl}.json,
    data/scene_splits.json, data/procthor-10k/train.jsonl.gz (house line NNNNN = {"id": "procthor_NNNNN", ...}),
    houses/<scene>.json (as `reach houses`), hf_meta/tvr-sft-va/sft_train.json.
    tasks = {"sft": [...], "eval": [...], "rl": [...]}; public = a list of sft_train records or of
    {"task_id", "actions"[, "positions"]} specs.  Name two sft tasks like tasks.SFT2_IDS to use the sft2 set on a
    fake tree.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from vpd.tvr import lvp

REPO = lvp.TVR_REPO
HAVE_REPO = (REPO / "tvrbench" / "online_rl" / "thor_view_env.py").is_file()
Y = 0.9

# The frame code is shared verbatim by the fake controller module and frame_of() below.
FRAME_SRC = r'''
import hashlib as _hashlib

import numpy as _np

_FRAME_CACHE = {}


def _r2(v):
    return round(float(v), 2) + 0.0


def start_key(x, z):
    return "%.2f,%.2f" % (_r2(x), _r2(z))


def frame_key(scene, x, z, rot, hor):
    return "%s|%.2f|%.2f|%d|%d" % (scene, _r2(x), _r2(z), int(round(float(rot))) % 360, int(round(float(hor))))


def _frame(scene, x, z, rot, hor, h=360, w=640):
    """A blocky frame, a pure function of (scene, pose); read-only and cached."""
    key = frame_key(scene, x, z, rot, hor)
    f = _FRAME_CACHE.get((key, h, w))
    if f is None:
        seed = int(_hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        small = _np.random.default_rng(seed).integers(0, 256, (9, 16, 3), dtype=_np.uint8)
        f = _np.repeat(_np.repeat(small, -(-h // 9), axis=0), -(-w // 16), axis=1)[:h, :w]
        f = _np.ascontiguousarray(f)
        f.setflags(write=False)
        if len(_FRAME_CACHE) > 20000:
            _FRAME_CACHE.clear()
        _FRAME_CACHE[(key, h, w)] = f
    return f
'''

CONTROLLER_SRC = r'''"""Fake ai2thor.controller (vpd tests, vpd/tests/tvr_fakes.py): no Unity."""
import json
import os
''' + FRAME_SRC + r'''

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES_FILE = os.path.join(_ROOT, "fake_scenes.json")
DEFAULT_GRID = [{"x": round(-1.0 + 0.25 * i, 2), "y": 0.9, "z": round(-1.0 + 0.25 * j, 2)}
                for i in range(11) for j in range(11)]


def load_scenes():
    try:
        with open(SCENES_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class Event:
    def __init__(self, metadata, frame):
        self.metadata = metadata
        self.frame = frame


class Controller:
    created = []

    def __init__(self, **kw):
        self.kw = dict(kw)
        keys = ("fieldOfView", "antiAliasing", "gridSize", "rotateStepDegrees", "snapToGrid", "renderDepthImage",
                "renderInstanceSegmentation", "makeAgentsVisible")
        self.initialization_parameters = {k: kw[k] for k in keys if k in kw}
        self.width, self.height = kw.get("width", 640), kw.get("height", 360)
        self.quality, self.gpu_device = kw.get("quality"), kw.get("gpu_device")
        self.rot_step = float(kw.get("rotateStepDegrees", 90))
        self.stopped = False
        self.n_steps = 0
        self.n_reach = 0
        self._set_scene(kw.get("scene", "FloorPlan1"))
        Controller.created.append(self)

    # -- scene table
    def _spec(self):
        return self.scenes.get(self.scene) or {}

    def _positions(self):
        spec = self._spec()
        by = spec.get("by_start") or {}
        key = start_key(self.pos["x"], self.pos["z"])
        if key in by:
            return by[key]
        return spec.get("positions") or DEFAULT_GRID

    def _set_scene(self, scene):
        self.scenes = load_scenes()
        self.scene = scene if isinstance(scene, str) else str(scene["id"])
        spec = self._spec()
        sp = spec.get("spawn")
        if sp is None:
            p0 = (spec.get("positions") or DEFAULT_GRID)[0]
            sp = {"position": {"x": p0["x"], "y": p0.get("y", 0.9), "z": p0["z"]}, "rotation_y": 0.0,
                  "horizon": 0.0}
        self.pos = {"x": float(sp["position"]["x"]), "y": float(sp["position"].get("y", 0.9)),
                    "z": float(sp["position"]["z"])}
        self.rot, self.hor = float(sp["rotation_y"]) % 360, float(sp["horizon"])
        self.last_event = self._event(True, None)

    def _event(self, ok, ret, err=""):
        md = {"lastActionSuccess": ok, "errorMessage": err, "actionReturn": ret, "screenWidth": self.width,
              "screenHeight": self.height, "fov": self.kw.get("fieldOfView"),
              "agent": {"position": dict(self.pos), "rotation": {"x": 0.0, "y": float(self.rot % 360), "z": 0.0},
                        "cameraHorizon": float(self.hor)}}
        return Event(md, _frame(self.scene, self.pos["x"], self.pos["z"], self.rot, self.hor, self.height,
                                self.width))

    def step(self, action=None, **kw):
        if action is None:
            action = kw.pop("action", None)
        self.n_steps += 1
        if action == "GetReachablePositions":
            self.n_reach += 1
            ev = self._event(True, [dict(p) for p in self._positions()])
        elif action == "Teleport":
            p = kw["position"]
            key = start_key(p["x"], p["z"])
            cells = {start_key(q["x"], q["z"]) for q in self._positions()}
            if key in set(self._spec().get("fail_teleport") or ()) or (not kw.get("forceAction") and key not in cells):
                ev = self._event(False, None, "fake teleport failure at %s" % key)
            else:
                self.pos = {"x": float(p["x"]), "y": float(p.get("y", self.pos["y"])), "z": float(p["z"])}
                if "rotation" in kw:
                    self.rot = float(kw["rotation"]["y"]) % 360
                self.hor = float(kw.get("horizon", self.hor))
                ev = self._event(True, None)
        elif action in ("RotateRight", "RotateLeft"):
            d = float(kw.get("degrees", self.rot_step))
            self.rot = (self.rot + (d if action == "RotateRight" else -d)) % 360
            ev = self._event(True, None)
        elif action in ("LookUp", "LookDown"):
            d = float(kw.get("degrees", 30.0))
            self.hor += -d if action == "LookUp" else d
            ev = self._event(True, None)
        else:
            ev = self._event(False, None, "unknown action %r" % (action,))
        self.last_event = ev
        return ev

    def reset(self, scene=None, **kw):
        self._set_scene(scene if scene is not None else self.scene)
        return self.last_event

    def stop(self):
        self.stopped = True
'''

_ns: Dict = {}
exec(compile(FRAME_SRC, "<tvr_fakes.FRAME_SRC>", "exec"), _ns)  # noqa: S102 (our own constant)
start_key = _ns["start_key"]
frame_key = _ns["frame_key"]


def write_fake_ai2thor(root: Path, scenes: Optional[Dict] = None) -> Path:
    """The fake ai2thor package + root/fake_scenes.json (rewritten on every call); returns root."""
    root = Path(root)
    pkg = root / "ai2thor"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text('"""Fake ai2thor (vpd tests, vpd/tests/tvr_fakes.py)."""\n')
    (pkg / "controller.py").write_text(CONTROLLER_SRC)
    (pkg / "platform.py").write_text("class CloudRendering:\n    pass\n")
    (root / "fake_scenes.json").write_text(json.dumps(scenes or {}))
    return root


def purge_tvrbench(fake_root: Optional[Path] = None) -> None:
    """Forget imported ai2thor / tvrbench modules (and drop fake_root from sys.path)."""
    for k in list(sys.modules):
        if k in ("tvrbench", "ai2thor") or k.startswith(("tvrbench.", "ai2thor.")):
            del sys.modules[k]
    if fake_root is not None:
        while str(fake_root) in sys.path:
            sys.path.remove(str(fake_root))


def import_real_tvrbench(fake_root: Path):
    """(thor_env, thor_view_env) of the pinned TVRBench checkout running on the fake ai2thor at fake_root."""
    purge_tvrbench(fake_root)
    sys.path.insert(0, str(fake_root))
    te, tv = lvp.import_tvrbench(REPO)
    got = Path(sys.modules["ai2thor"].__file__).resolve()
    if Path(fake_root).resolve() not in got.parents:
        raise ImportError(f"ai2thor came from {got}, not from the fake at {fake_root}")
    return te, tv


def _pose(t: Sequence[float], y: float = Y) -> Dict:
    x, z, rot, hor = t
    return {"position": {"x": float(x), "y": float(y), "z": float(z)}, "rotation_y": float(rot),
            "horizon": float(hor)}


def fake_task(task_id: str, scene: str, dataset: str, difficulty: str, start: tuple, target: tuple) -> Dict:
    return {"task_id": task_id, "dataset": dataset, "scene": scene, "difficulty": difficulty, "seg_count": 10,
            "bfs_distance": 1, "target": _pose(target), "start": _pose(start)}


def grid_positions(xs: Iterable[float], zs: Iterable[float], blocked: Iterable[Tuple[float, float]] = (),
                   y: float = Y) -> List[Dict]:
    bl = {(round(float(a), 2) + 0.0, round(float(b), 2) + 0.0) for a, b in blocked}
    zs = list(zs)
    out = []
    for x in xs:
        for z in zs:
            c = (round(float(x), 2) + 0.0, round(float(z), 2) + 0.0)
            if c not in bl:
                out.append({"x": c[0], "y": float(y), "z": c[1]})
    return out


def frame_of(scene: str, pose):
    """The fake controller's frame for (scene, pose); pose = (x, z, rot, hor) or a TVRBench pose dict."""
    if isinstance(pose, dict):
        x, z, rot, hor = pose["position"]["x"], pose["position"]["z"], pose["rotation_y"], pose["horizon"]
    else:
        x, z, rot, hor = pose
    return _ns["_frame"](scene, x, z, rot, hor).copy()


def public_record(task: Dict, actions: Sequence[str], positions: Optional[List[Dict]] = None) -> Dict:
    """An sft_train.json record of `task` with these actions (the last must be Stop for a well-formed record);
    the valid lines are kin_env's along the path on `positions` (None: free space)."""
    from vpd.tvr import kin_env, tvr_text

    reach = None if positions is None else kin_env.ReachSet.from_positions(positions)
    p = kin_env.pose_of(task["start"])
    conv, imgs = [], []
    tid = task["task_id"]
    for k, a in enumerate(actions):
        conv.append({"from": "human", "value": tvr_text.build_concat_user_text(kin_env.valid_actions(p, reach))})
        conv.append({"from": "gpt", "value": f"Action: {a}"})
        imgs += [f"images/{tid}/step_{k:02d}.png", f"images/{tid}/target.png"]
        if a != "Stop":
            p = kin_env.step(p, a, reach)[0]
    return {"system": tvr_text.SYSTEM_PROMPT, "conversations": conv, "images": imgs}


def write_fake_tvr_tree(root: Path, tasks: Dict, public: List) -> Dict:
    """A fake TVRBench data tree + public file (module doc)."""
    root = Path(root)
    data = root / "repo" / "data"
    td = data / "tasks"
    td.mkdir(parents=True, exist_ok=True)
    scenes: Dict[str, List[str]] = {}
    for split in ("sft", "eval", "rl"):
        ts = list(tasks.get(split, []))
        (td / f"{split}.json").write_text(json.dumps(ts))
        scenes[split] = list(dict.fromkeys(t["scene"] for t in ts))
    splits = {"generated_at": "fake", "seed": 0, "note": "vpd tests",
              "ithor": {s: [x for x in v if not lvp.is_procthor(x)] for s, v in scenes.items()},
              "procthor": {s: [x for x in v if lvp.is_procthor(x)] for s, v in scenes.items()}}
    (data / "scene_splits.json").write_text(json.dumps(splits))
    pt = sorted({s for v in scenes.values() for s in v if lvp.is_procthor(s)})
    idx = {lvp.procthor_line_index(s): s for s in pt}
    houses = {}
    pdir = data / "procthor-10k"
    pdir.mkdir(parents=True, exist_ok=True)
    gz = pdir / "train.jsonl.gz"
    with gzip.open(gz, "wt") as f:
        for i in range((max(idx) + 1) if idx else 1):
            h = {"id": idx[i], "rooms": [{"i": i}], "fake": True} if i in idx else {"id": f"filler_{i:05d}"}
            if i in idx:
                houses[idx[i]] = h
            f.write(json.dumps(h) + "\n")
    houses_dir = root / "houses"  # what `reach houses` would extract: houses_dir/<scene>.json
    lvp.extract_procthor_houses(gz, pt, houses_dir)
    by = {t["task_id"]: t for split in ("sft", "eval", "rl") for t in tasks.get(split, [])}
    recs = []
    for p in public:
        if "conversations" in p:
            recs.append(p)
        else:
            recs.append(public_record(by[p["task_id"]], p["actions"], p.get("positions")))
    sft_train = root / "hf_meta" / "tvr-sft-va" / "sft_train.json"
    sft_train.parent.mkdir(parents=True, exist_ok=True)
    sft_train.write_text(json.dumps(recs))
    return {"root": root, "repo_data": data, "tasks_dir": td, "procthor_dir": pdir, "procthor_gz": gz,
            "sft_train": sft_train, "houses": houses, "houses_dir": houses_dir, "scenes": scenes}
