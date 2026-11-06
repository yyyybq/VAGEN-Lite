"""Experiment 3A (the user's decision 甲 of 2026-11-06, turn-mismatch RL): in-place turns blocked by the robot base's
footprint, env setting turn_footprint_radius (default None: off, today's rule).

SwitchOffGoldenTest (made on the unchanged code, qwen3-main-dev 26338088): scripted episodes in the RL env on the
synthetic two-room scene with thick walls and a stub renderer, under the live template (rl_env_v10_S0.yaml with real
action names: S0, and V4 = D_cond with the progress scale 10/3).  Every step's observation text, reward, end,
feedback, info keys, turn and trajectory metrics, reward components and pose must equal golden/env_trace_26338088.json.
The golden holds the scene files, the two task rows (both start inside a low object's footprint, where the 3A rule
would block every turn) and the env settings, so the test does not depend on the task generator.  With the switch
off every later commit must give the same trace.  TURN_FOOTPRINT_GOLDEN_REGEN=1 rewrites the golden (only on the
code it is meant to record)."""

import importlib.util
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "configs" / "rl_env_v10_S0.yaml"
GOLDEN = HERE / "golden" / "env_trace_26338088.json"
REGEN = os.environ.get("TURN_FOOTPRINT_GOLDEN_REGEN") == "1"
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
BAND_M = (0.10, 1.00)  # the spec's height band (an object box reaching into it blocks a turn)

# (a) from the task start: two turns; five move_backward (from the sofa start the second one meets the west wall's
# slab); an unreadable answer; five move_forward up to the 12-action limit
SCRIPT_START = ["<action>turn_left|turn_left|</action>",
                "<action>move_backward|move_backward|move_backward|move_backward|move_backward|</action>",
                "<action>jump|</action>",
                "<action>move_forward|move_forward|move_forward|move_forward|move_forward|</action>"]
# (b) the camera above the tea table: inside its footprint (box z 0-0.45), the camera point legal under today's rule
TEA_TABLE = (2.9, 2.5, 0.0)
SCRIPT_TEA_TABLE = ["<action>turn_left|move_forward|turn_right|</action>"]
# (c) 0.2 m from the inner face of the west wall's slab (x = 0.1), facing it: a turn, then a move into the slab
WEST_WALL = (0.30, 1.0, math.pi)
SCRIPT_WEST_WALL = ["<action>turn_right|move_forward|</action>", "<action>move_backward|turn_left|</action>"]
EPISODES = {"start": (None, SCRIPT_START), "tea_table": (TEA_TABLE, SCRIPT_TEA_TABLE),
            "west_wall": (WEST_WALL, SCRIPT_WEST_WALL)}


def stub_frame():
    """The stub renderer's frame (the one test_env_switches.py / test_blocked_turns.py use)."""
    from PIL import Image

    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))


def plain(x):
    """A JSON-plain copy: numpy values as Python ones, floats rounded to 1e-9 (non-finite ones as strings)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return round(x, 9) + 0.0 if math.isfinite(x) else repr(x)  # + 0.0: -0.0 reads as 0.0
    return x


def env_configs(gs_root):
    """{"S0": the live template's env settings with real action names, "V4": the same with D_cond's reward and the
    progress scale 10/3 (configs/matrix_r12.json rl.dense)}, the scenes read from ``gs_root``."""
    from data_gen.interiorgs_v10 import rl_checks as RC
    from data_gen.interiorgs_v10 import training_prep as TP

    s0 = dict(RC.env_config(TEMPLATE, "http://127.0.0.1:1/render"), prompt_format_example="action_names",
              gs_root=str(gs_root))
    return {"S0": s0, "V4": {**s0, **TP.reward_overrides("D_cond"), "potential_field_reward_scale": 10.0 / 3.0}}


def settings(cfg):
    """The env settings as the golden records them (without the scene folder, a temporary path)."""
    return plain({k: v for k, v in cfg.items() if k != "gs_root"})


def low_box_at(labels, xy, band=BAND_M):
    """The label of the first labels.json box (CollisionDetector.IGNORED_LABELS left out, by substring) whose xy
    rectangle holds ``xy`` and that reaches into the height band; None when there is none."""
    from vagen.envs.active_spatial.collision_detector import CollisionDetector

    for obj in labels:
        label = str(obj.get("label", "")).lower()
        if any(ignored in label for ignored in CollisionDetector.IGNORED_LABELS):
            continue
        pts = np.array([[c["x"], c["y"], c["z"]] for c in obj["bounding_box"]], float)
        lo, hi = pts.min(0), pts.max(0)
        if lo[2] < band[1] and hi[2] > band[0] and lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]:
            return label
    return None


def write_scene(golden, root):
    """The golden's scene files written to root/<scene name>; returns the folder."""
    scene = Path(root) / golden["scene"]["name"]
    scene.mkdir(parents=True, exist_ok=True)
    for name in ("labels.json", "structure.json"):
        (scene / name).write_text(json.dumps(golden["scene"][name], indent=1))
    return scene


def step_record(env, obs, reward, done, info):
    """What the golden compares of one env.step (episode_elapsed_seconds is left out; image_std, a float32 standard
    deviation, rounded to 1e-4)."""
    metrics = info.get("metrics") or {}
    turn = dict(metrics.get("turn_metrics") or {})
    if turn.get("image_std") is not None:
        turn["image_std"] = round(float(turn["image_std"]), 4)
    components = (info.get("reward_trace") or {}).get("components") or {}
    return plain({
        "obs_str": obs.get("obs_str"),
        "images": sum(len(v or []) for v in (obs.get("multi_modal_data") or {}).values()),
        "reward": reward, "done": done, "success": info.get("success"),
        "env_step": info.get("env_step"), "termination_reason": info.get("termination_reason"),
        "env_feedback": info.get("env_feedback"), "actions": info.get("actions"),
        "info_keys": sorted(info),
        "turn_metrics": turn, "traj_metrics": metrics.get("traj_metrics"),
        "components": {k: {"applied": v.get("applied"), "scale": v.get("scale"), "events": v.get("events")}
                       for k, v in sorted(components.items())},
        "current_pose": info.get("current_pose"),
        "c2w": env.view_engine.get_pose(),
    })


def make_env(cfg, row, path):
    from data_gen.interiorgs_v10 import rl_checks as RC

    env, _ = RC.make_env([row], Path(path), cfg)
    return env


def place(env, xy_yaw):
    """Moves the camera (after env.reset) to world (x, y) looking along yaw (radians), at the camera height."""
    from data_gen.interiorgs_v10.contract import pose_from_xy_yaw

    env.view_engine.reset(pose_from_xy_yaw(*xy_yaw))


def met(env):
    metric = env._calculate_canonical_metric()
    return bool(metric and metric.get("success"))


def run_episode(cfg, row, path, xy_yaw, answers):
    """One scripted episode in a fresh env: reset, the camera moved to ``xy_yaw`` (None: the task's start), then the
    answers until the episode ends."""
    env = make_env(cfg, row, path)
    obs, _ = env.reset(seed=0)
    out = {"system_prompt": env.system_prompt(), "reset_obs": obs.get("obs_str")}
    if xy_yaw is not None:
        place(env, xy_yaw)
    out["met_at_start"] = met(env)
    out["steps"] = []
    for text in answers:
        obs, reward, done, info = env.step(text)
        out["steps"].append(step_record(env, obs, reward, done, info))
        if done:
            break
    return out


def run_all(golden, workdir, extra=None):
    """Every scripted episode of the golden's rows under both settings (plus ``extra`` env settings)."""
    out = {}
    for name, cfg in env_configs(Path(workdir) / "scenes").items():
        for i, row in enumerate(golden["rows"]):
            for episode, (xy_yaw, answers) in EPISODES.items():
                key = f"{name}/{row['task_id']}/{episode}"
                out[key] = run_episode(dict(cfg, **(extra or {})), row, Path(workdir) / "envs" / f"{name}_{i}_{episode}.jsonl",
                                       xy_yaw, answers)
    return out


def git_head():
    """The commit a regenerated golden records, and whether tracked files outside the tests differ from it."""
    root = HERE.parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", ".", ":(exclude)data_gen/interiorgs_v10/tests"],
                              cwd=root, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": head, "tracked_changes_outside_tests": {0: False, 1: True}.get(diff.returncode)}


def build_golden(workdir):
    """(TURN_FOOTPRINT_GOLDEN_REGEN=1) the scene, two task rows, the env settings and every scripted episode's trace
    on the code at hand.  Rows: the first two by task_id whose start lies inside a low object's footprint."""
    from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
    from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

    scene_dir = write_synthetic_scenes(Path(workdir) / "generated", with_walls=True)["two_rooms_walls"]
    run = RunConfig(target_tasks_per_scene=8, wall_model="thick", visibility_rule="env", require_structure_walls=True)
    rows = sorted(_jsonable(generate_scene(scene_dir, scene_dir.name, run))["rows"], key=lambda r: r["task_id"])
    labels = json.loads((scene_dir / "labels.json").read_text())
    picked = [r for r in rows if low_box_at(labels, np.array(r["init_camera"]["extrinsics"], float)[:2, 3])][:2]
    golden = {"_doc": "Experiment 3A switch-off golden (tests/test_turn_footprint.py SwitchOffGoldenTest): the RL env's "
                      "outputs on scripted episodes with turn_footprint_radius absent, recorded on the unchanged code; "
                      "every later commit must reproduce them. TURN_FOOTPRINT_GOLDEN_REGEN=1 rewrites this file.",
              "generated_at": git_head(),
              "scene": {"name": scene_dir.name, "labels.json": labels,
                        "structure.json": json.loads((scene_dir / "structure.json").read_text())},
              "rows": picked,
              "scripts": {k: {"camera_xy_yaw": None if v[0] is None else list(v[0]), "answers": v[1]}
                          for k, v in EPISODES.items()}}
    write_scene(golden, Path(workdir) / "scenes")
    golden["settings"] = {k: settings(v) for k, v in env_configs(Path(workdir) / "scenes").items()}
    golden["episodes"] = run_all(golden, workdir)
    return golden


@unittest.skipUnless(HAS_SCIPY, "the env needs scipy (the cluster env)")
class SwitchOffGoldenTest(unittest.TestCase):
    """With the switch off (absent), the env's every output on the scripted episodes is the golden's."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.patch = mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image",
                               lambda self, frame=stub_frame(): frame.copy())
        cls.patch.start()
        if REGEN:
            golden = build_golden(cls.root / "regen")
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        write_scene(cls.golden, cls.root / "scenes")
        cls.got = run_all(cls.golden, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls._tmp.cleanup()

    def test_settings_are_the_live_templates(self):
        """The template and the reward preset still give the settings the golden was made with (otherwise the
        golden is out of date for another reason than this switch)."""
        self.assertEqual({k: settings(v) for k, v in env_configs(self.root / "scenes").items()}, self.golden["settings"])

    def test_trace_equals_the_golden(self):
        self.assertEqual(sorted(self.got), sorted(self.golden["episodes"]))
        for key, want in self.golden["episodes"].items():
            got = self.got[key]
            with self.subTest(episode=key):
                self.assertEqual({k: v for k, v in got.items() if k != "steps"},
                                 {k: v for k, v in want.items() if k != "steps"})
                self.assertEqual(len(got["steps"]), len(want["steps"]))
                for i, (g, w) in enumerate(zip(got["steps"], want["steps"])):
                    self.assertEqual(g, w, f"{key}, step {i + 1}")

    def test_the_episodes_cover_what_the_switch_changes(self):
        """Both rows start inside a low object's footprint (the 3A rule would block every turn there); no episode
        meets its task; turns are never blocked under today's rule, translations are (from the sofa start and at
        the west wall); the start episodes end at the 12-action limit."""
        from data_gen.interiorgs_v10.fail_analysis import BLOCKED_RE

        labels = self.golden["scene"]["labels.json"]
        starts = [low_box_at(labels, np.array(r["init_camera"]["extrinsics"], float)[:2, 3]) for r in self.golden["rows"]]
        self.assertEqual(len(starts), 2)
        self.assertTrue(all(starts), starts)
        blocked = []
        for key, ep in self.got.items():
            self.assertFalse(ep["met_at_start"], key)
            self.assertFalse(any(s["success"] for s in ep["steps"]), key)
            for s in ep["steps"]:
                names = BLOCKED_RE.findall(s["env_feedback"].split("blocked: ", 1)[1]) if "blocked: " in s["env_feedback"] else []
                self.assertEqual(len(names), s["turn_metrics"]["collision_count"], key)
                blocked += [(key.split("/")[-1], n) for n in names]
            if key.endswith("/start"):
                self.assertEqual((ep["steps"][-1]["env_step"], ep["steps"][-1]["termination_reason"]),
                                 (12, "max_primitive_steps"), key)
        self.assertFalse([b for b in blocked if b[1].startswith("turn_")])
        self.assertIn(("start", "move_backward"), blocked)
        self.assertIn(("west_wall", "move_forward"), blocked)


# ---------------------------------------------------------------------------
# commit C2: CollisionDetector.check_turn_footprint
# ---------------------------------------------------------------------------
def footprint_room(root, name="fp_room", boxes=None):
    """A 6 m x 4 m room (structure.json walls 0.2 m thick, a 1 m door in the east wall at y 1.5-2.5) with boxes for
    the footprint rule's cases; labels.json order matters for ties."""
    from data_gen.interiorgs_v10.testing.synthetic_scene import _box, _write

    if boxes is None:
        boxes = [("cabinet", 1.0, 1.0, 0.0, 1.6, 1.5, 0.8),           # low: blocks
                 ("range hood", 3.0, 3.0, 2.2, 3.6, 3.5, 2.6),        # above the band
                 ("shelf", 4.0, 0.6, 1.0, 4.4, 1.0, 1.8),             # min_z exactly 1.0: out of the band
                 ("tray", 4.0, 2.0, 0.0, 4.4, 2.4, 0.1),              # max_z exactly 0.1: out of the band
                 ("tv", 2.4, 0.6, 0.95, 3.0, 0.9, 1.5),               # reaches 0.05 m into the band: in
                 ("rug", 4.6, 2.9, 0.0, 5.2, 3.4, 0.3),               # in the band, but an ignored label
                 ("bathroom cabinet", 0.6, 3.0, 0.0, 1.1, 3.5, 0.9),  # ignored by substring ('room'): kept as is
                 ("side table", 2.0, 2.0, 0.0, 2.2, 2.2, 0.5),
                 ("stool", 2.5, 2.0, 0.0, 2.7, 2.2, 0.45),
                 ("radiator", 0.0, 2.6, 0.2, 0.3, 3.0, 0.8)]          # set into the west wall's slab
    labels = [_box("1", "floor", 0.0, 0.0, -0.02, 6.0, 4.0, 0.0)]
    labels += [_box(str(i + 2), label, *xyz) for i, (label, *xyz) in enumerate(boxes)]
    walls = [((0, 0), (6, 0), 0.2), ((6, 0), (6, 4), 0.2), ((6, 4), (0, 4), 0.2), ((0, 4), (0, 0), 0.2)]
    holes = [[(5.95, 1.5), (6.05, 1.5), (6.05, 2.5), (5.95, 2.5)]]
    return _write(Path(root) / name, labels, [("living room", [(0, 0), (6, 0), (6, 4), (0, 4)])], holes, y_sign=1.0,
                  truth={}, walls=walls)


def env_detector():
    """A detector as the RL env makes it (env.py: camera radius 0.15, floor 0.3, ceiling 2.5, margin 0.05, thick)."""
    from vagen.envs.active_spatial.collision_detector import create_collision_detector

    return create_collision_detector({"camera_radius": 0.15, "floor_height": 0.3, "ceiling_height": 2.5,
                                      "safety_margin": 0.05, "enable_object_collision": True,
                                      "enable_boundary_collision": True, "wall_model": "thick"})


class ReplayGeo:
    """An inline copy of the offline replay's rule (instr_20261106 turnblock_3a/turnblock_exposure.py, Geo lines
    35-64: the band's object rectangles and the wall segments of a loaded detector, and its clearance)."""

    def __init__(self, det, zlo=0.10, zhi=1.00):
        from vagen.envs.active_spatial.collision_detector import CollisionDetector

        lo, hi = [], []
        for ins, label, mn, mx in det.label_boxes:
            if any(ig in label for ig in CollisionDetector.IGNORED_LABELS):
                continue
            if mn[2] >= zhi or mx[2] <= zlo:
                continue
            lo.append(mn[:2])
            hi.append(mx[:2])
        self.lo = np.array(lo, float).reshape(-1, 2)
        self.hi = np.array(hi, float).reshape(-1, 2)
        segs = det.wall_segments
        self.a = np.array([s[0] for s in segs], float).reshape(-1, 2)
        self.b = np.array([s[1] for s in segs], float).reshape(-1, 2)

    def clearance(self, p):
        do, dw = math.inf, math.inf
        if len(self.lo):
            dx = np.maximum(np.maximum(self.lo[:, 0] - p[0], 0.0), p[0] - self.hi[:, 0])
            dy = np.maximum(np.maximum(self.lo[:, 1] - p[1], 0.0), p[1] - self.hi[:, 1])
            do = float(np.sqrt(dx * dx + dy * dy).min())
        if len(self.a):
            v = self.b - self.a
            w = p[None, :] - self.a
            t = np.clip((w * v).sum(1) / np.maximum((v * v).sum(1), 1e-12), 0.0, 1.0)
            proj = self.a + t[:, None] * v
            dw = float(np.sqrt(((p[None, :] - proj) ** 2).sum(1)).min())
        return do, dw


class FootprintGeometryTest(unittest.TestCase):
    """CollisionDetector.check_turn_footprint on a small scene (no env)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.scene = footprint_room(cls.root)
        cls.det = env_detector()
        assert cls.det.load_scene(cls.scene, scene_id=cls.scene.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def check(self, x, y, z=1.5, radius=0.25, **kw):
        return self.det.check_turn_footprint(np.array([x, y, z]), radius, **kw)

    def assert_blocked(self, res, kind, name):
        self.assertTrue(res.has_collision)
        self.assertEqual((res.collision_type, res.collision_object), (kind, name))

    def test_the_band_and_the_label_rule(self):
        from vagen.envs.active_spatial.collision_detector import TURN_FOOTPRINT_Z_BAND_M

        self.assertEqual(TURN_FOOTPRINT_Z_BAND_M, (0.10, 1.00))
        inside = self.check(1.3, 1.25)  # inside the cabinet's rectangle, the camera 0.7 m above its top
        self.assert_blocked(inside, "object", "cabinet")
        self.assertEqual((inside.distance_to_collision, inside.details["inside_object"]), (0.0, True))
        self.assertEqual(inside.details["object_id"], "2")
        self.assertFalse(self.det.check_collision(np.array([1.3, 1.25, 1.5])).has_collision)  # legal today
        # 1 mm either side of 0.25 m from the cabinet's +x face (never exactly 0.25: 0.35 - 0.1 < 0.25 in float64)
        self.assert_blocked(self.check(1.6 + 0.249, 1.25), "object", "cabinet")
        clear = self.check(1.6 + 0.251, 1.25)
        self.assertFalse(clear.has_collision)
        self.assertEqual(clear.collision_type, "none")
        self.assertAlmostEqual(clear.distance_to_collision, 0.251, places=9)
        self.assertFalse(self.check(3.3, 3.25).has_collision)   # under the range hood
        self.assertFalse(self.check(4.2, 0.8).has_collision)    # min_z exactly 1.0: out
        self.assertFalse(self.check(4.2, 2.2).has_collision)    # max_z exactly 0.1: out
        self.assert_blocked(self.check(2.7, 0.75), "object", "tv")  # 0.95-1.5 m overlaps the band
        self.assertFalse(self.check(4.9, 3.15).has_collision)   # a rug, though 0.3 m high
        self.assertFalse(self.check(0.85, 3.25).has_collision)  # 'bathroom cabinet' contains 'room' (kept quirk)
        # another band (the parameter, cached apart): the hood and the shelf count
        self.assert_blocked(self.check(3.3, 3.25, z_band=(0.10, 2.50)), "object", "range hood")
        self.assert_blocked(self.check(4.2, 0.8, z_band=(0.10, 2.50)), "object", "shelf")
        self.assertFalse(self.check(3.3, 3.25).has_collision)

    def test_walls_doors_and_names(self):
        wall = self.check(0.3, 1.5)  # 0.2 m from the west slab's inner face (x = 0.1)
        self.assert_blocked(wall, "wall", "wall")
        self.assertAlmostEqual(wall.distance_to_collision, 0.2, places=12)
        self.assertEqual(sorted(wall.details), ["clearance", "inside_object", "object_clearance", "radius", "rule",
                                                "wall_clearance", "wall_end", "wall_start", "z_band"])
        self.assertEqual({tuple(wall.details["wall_start"]), tuple(wall.details["wall_end"])}, {(0.1, 0.0), (0.1, 4.0)})
        self.assertFalse(self.det.check_collision(np.array([0.3, 1.5, 1.5])).has_collision)  # 0.2 >= 0.15: legal
        self.assertFalse(self.check(6.0, 2.0).has_collision)  # the doorway's centre: 0.5 m from both jambs
        self.assertTrue(self.check(6.0, 1.7).has_collision)   # 0.2 m from a jamb
        # two objects within r: the nearer names the block
        self.assert_blocked(self.check(2.3, 2.1), "object", "side table")
        self.assert_blocked(self.check(2.45, 2.1), "object", "stool")
        # an object and a wall at the same distance (0: on the slab's face, inside the radiator's box): the object
        tie = self.check(0.1, 2.75)
        self.assertEqual((tie.details["object_clearance"], tie.details["wall_clearance"]), (0.0, 0.0))
        self.assert_blocked(tie, "object", "radiator")
        # the verdict reads the ground point only
        for z in (0.0, 1.5, 9.0):
            self.assertEqual((self.check(1.3, 1.25, z=z).collision_object, self.check(0.3, 1.5, z=z).collision_type,
                              self.check(3.0, 1.9, z=z).has_collision), ("cabinet", "wall", False))
        self.assertTrue(self.check(3.0, 1.9, radius=0.35).has_collision)  # the radius is the setting's

    def test_same_numbers_as_the_offline_replay(self):
        geo = ReplayGeo(self.det)
        rng = np.random.default_rng(20261106)
        pts = np.column_stack([rng.uniform(-0.3, 6.3, 2000), rng.uniform(-0.3, 4.3, 2000)])
        blocked = 0
        for p in pts:
            do, dw = geo.clearance(p)
            res = self.det.check_turn_footprint(np.array([p[0], p[1], 1.5]), 0.25)
            self.assertEqual((res.details["object_clearance"], res.details["wall_clearance"]), (do, dw))
            self.assertEqual(res.has_collision, min(do, dw) < 0.25)
            blocked += res.has_collision
        self.assertTrue(200 < blocked < 1800, blocked)  # both verdicts well represented

    def test_no_side_effects_and_errors(self):
        probes = [(1.3, 1.25), (0.3, 1.5), (0.2, 1.5), (3.0, 1.9), (5.8, 2.0), (2.7, 0.75)]

        def snapshot():
            return ([id(self.det.object_boxes), id(self.det.label_boxes), id(self.det.wall_segments)],
                    [(b.label, b.min_point.tolist(), b.max_point.tolist()) for b in self.det.object_boxes],
                    [(i, l, mn.tolist(), mx.tolist()) for i, l, mn, mx in self.det.label_boxes],
                    [(a.tolist(), b.tolist()) for a, b in self.det.wall_segments],
                    [(r.has_collision, r.collision_type, r.collision_object, r.distance_to_collision)
                     for r in (self.det.check_collision(np.array([x, y, 1.5]), previous_position=np.array([x - 0.3, y, 1.5]))
                               for x, y in probes)])

        before = snapshot()
        for x, y in probes * 3:
            self.check(x, y)
        self.assertEqual(snapshot(), before)
        for bad in (0, 0.0, -0.25, float("nan")):
            with self.assertRaises(ValueError):
                self.check(1.3, 1.25, radius=bad)
        from vagen.envs.active_spatial.collision_detector import CollisionDetector

        with self.assertRaises(RuntimeError):
            CollisionDetector(wall_model="thick").check_turn_footprint(np.array([1.3, 1.25, 1.5]), 0.25)

    def test_a_reload_rebuilds_the_obstacles(self):
        det = env_detector()
        self.assertTrue(det.load_scene(self.scene, scene_id=self.scene.name))
        self.assertTrue(det.check_turn_footprint(np.array([1.3, 1.25, 1.5]), 0.25).has_collision)
        other = footprint_room(self.root, "fp_room_b", boxes=[("sofa", 3.0, 1.0, 0.0, 4.0, 2.0, 0.9)])
        self.assertTrue(det.load_scene(other, scene_id=other.name))
        self.assertFalse(det.check_turn_footprint(np.array([1.3, 1.25, 1.5]), 0.25).has_collision)  # cabinet gone
        res = det.check_turn_footprint(np.array([3.5, 1.5, 1.5]), 0.25)
        self.assertEqual((res.has_collision, res.collision_object), (True, "sofa"))
        self.assertTrue(det.load_scene(self.scene, scene_id=self.scene.name))  # and back
        self.assertEqual(det.check_turn_footprint(np.array([1.3, 1.25, 1.5]), 0.25).collision_object, "cabinet")


if __name__ == "__main__":
    unittest.main()
