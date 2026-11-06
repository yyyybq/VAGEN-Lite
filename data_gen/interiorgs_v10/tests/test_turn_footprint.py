"""Experiment 3A (the user's decision 甲 of 2026-11-06, turn-mismatch RL): in-place turns blocked by the robot base's
footprint, env setting turn_footprint_radius (default None: off, today's rule).

SwitchOffGoldenTest (made on the unchanged code, qwen3-main-dev 26338088): scripted episodes in the RL env on the
synthetic two-room scene with thick walls and a stub renderer, under the live template (rl_env_v10_S0.yaml with real
action names: S0, and V4 = D_cond with the progress scale 10/3).  The whole of every env.reset and env.step must
equal golden/env_trace_26338088.json: the observation (its text and every other key, the frame as mode, size and
sha256 of the pixels), the reward, the end, every info key (feedback, parsed actions, turn and trajectory metrics,
the whole reward trace, scores, region metrics, pose ...) and the camera's c2w; floats rounded to 1e-9, the elapsed
time and the stub frame's image_std (a float32 statistic of the test's own frame) left out.  The stub frame is a fixed
pattern made without a random generator, so neither its pixels nor its hash can change with NumPy.  The golden holds
the scene files, the two task rows (both start inside a low object's footprint, where the 3A rule would block every
turn) and the env settings, so the test does not depend on the task generator, and the commit it was recorded on
(generated_at, checked).  With the switch off every later commit must give the same trace.
TURN_FOOTPRINT_GOLDEN_REGEN=1 rewrites the golden (only on the code it is meant to record)."""

import hashlib
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
GOLDEN_MADE_ON = {"commit": "26338088", "tracked_changes_outside_tests": False}  # what git_head() gave at the regen
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
# (not in the golden) the first row's camera inside the double bed's footprint (box 7.8-9.8 x 1.2-3.4, z 0-0.6), 0.3 m
# from the inner face of the east wall's slab (x = 9.9), facing 120 degrees: move_right runs into the slab, the task is
# not met, and D_cond's potential is positive there
BED_EAST = (9.6, 1.5, math.radians(120.0))
SOFA = (0.75, 2.6)  # inside the multi person sofa's footprint (box 0.3-1.2 x 1.5-3.7, z 0-0.9), 0.65 m from the slab


def stub_frame():
    """The stub renderer's frame: a fixed 256 x 256 pattern made without a random generator (so its pixels, and the
    frame hash the golden records, cannot change with NumPy), each channel spread evenly over 0-255 like a real view
    (its standard deviation is about 74: never a low-information frame)."""
    from PIL import Image

    y, x = np.mgrid[0:256, 0:256]
    rgb = np.stack([(37 * x + 101 * y) % 256, (53 * x + 17 * y + 85) % 256, (11 * x + 73 * y + 170) % 256], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


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


def obs_record(obs):
    """An observation as the golden records it: every key, the images (multi_modal_data) as [mode, size, sha256 of the
    pixels]."""
    rec = {k: v for k, v in obs.items() if k != "multi_modal_data"}
    rec["frames"] = [[im.mode, list(im.size), hashlib.sha256(im.tobytes()).hexdigest()]
                     for _, images in sorted((obs.get("multi_modal_data") or {}).items()) for im in (images or [])]
    return rec


def info_record(info):
    """An info dict as the golden records it: every key but episode_elapsed_seconds (the clock) and image_std (info's
    and turn_metrics': the float32 standard deviation of the test's stub frame, not a rule of the env)."""
    rec = {k: v for k, v in info.items() if k not in ("episode_elapsed_seconds", "image_std")}
    if isinstance(rec.get("metrics"), dict) and isinstance(rec["metrics"].get("turn_metrics"), dict):
        rec["metrics"] = dict(rec["metrics"], turn_metrics={k: v for k, v in rec["metrics"]["turn_metrics"].items()
                                                            if k != "image_std"})
    return rec


def turn_metrics_of(rec):
    """The turn metrics of a step_record."""
    return rec["info"]["metrics"]["turn_metrics"]


def step_record(env, obs, reward, done, info):
    """What the golden compares of one env.step: the whole step -- the observation (obs_record), reward, end, every
    info key (info_record) and the camera's c2w -- JSON-plain, floats rounded to 1e-9."""
    return plain({"obs": obs_record(obs), "reward": reward, "done": done, "info": info_record(info),
                  "c2w": env.view_engine.get_pose()})


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
    obs, info = env.reset(seed=0)
    out = {"system_prompt": env.system_prompt(), "reset": plain({"obs": obs_record(obs), "info": info_record(info)})}
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
                      "whole outputs on scripted episodes with turn_footprint_radius absent (every reset and step: the "
                      "observation text and frame hash, reward, end, every info key but the clock and the stub frame's "
                      "image_std, the camera's c2w), recorded on the unchanged code (generated_at); every later commit "
                      "must reproduce them. TURN_FOOTPRINT_GOLDEN_REGEN=1 rewrites this file.",
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

    def test_the_golden_was_recorded_on_the_unchanged_code(self):
        """The golden was made at 26338088 with no tracked change outside the tests (git_head() at the regeneration):
        a regeneration on any later commit, or on a changed tree, fails here."""
        self.assertEqual(self.golden["generated_at"], GOLDEN_MADE_ON)

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
            self.assertEqual(len(ep["reset"]["obs"]["frames"]), 1, key)
            for s in ep["steps"]:
                info = s["info"]
                self.assertFalse(info["success"], key)
                # the whole info is recorded: the reward trace, the metrics, the pose ... but not the clock nor the
                # stub frame's statistic; every step shows one frame
                self.assertLessEqual({"reward_trace", "reward_trace_flat", "metrics", "current_pose", "env_feedback",
                                      "actions", "termination_reason"}, set(info), key)
                self.assertFalse({"episode_elapsed_seconds", "image_std"} & set(info), key)
                self.assertNotIn("image_std", turn_metrics_of(s), key)
                self.assertEqual(len(s["obs"]["frames"]), 1, key)
                feedback = info["env_feedback"]
                names = BLOCKED_RE.findall(feedback.split("blocked: ", 1)[1]) if "blocked: " in feedback else []
                self.assertEqual(len(names), turn_metrics_of(s)["collision_count"], key)
                blocked += [(key.split("/")[-1], n) for n in names]
            if key.endswith("/start"):
                self.assertEqual((ep["steps"][-1]["info"]["env_step"], ep["steps"][-1]["info"]["termination_reason"]),
                                 (12, "max_primitive_steps"), key)
        self.assertFalse([b for b in blocked if b[1].startswith("turn_")])
        self.assertIn(("start", "move_backward"), blocked)
        self.assertIn(("west_wall", "move_forward"), blocked)
        # the stub frame (no random generator) is the one every observation shows
        frame = stub_frame()
        self.assertEqual(self.got["S0/" + self.golden["rows"][0]["task_id"] + "/start"]["reset"]["obs"]["frames"][0],
                         ["RGB", [256, 256], hashlib.sha256(frame.tobytes()).hexdigest()])
        self.assertGreater(float(np.asarray(frame, dtype=np.float64).std()), 50.0)


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


# ---------------------------------------------------------------------------
# commit C3: the env setting turn_footprint_radius
# ---------------------------------------------------------------------------
def heading_deg(c2w):
    """The camera's forward direction as a world yaw in degrees (pitch 0 here)."""
    return math.degrees(math.atan2(c2w[1, 2], c2w[0, 2]))


@unittest.skipUnless(HAS_SCIPY, "the env needs scipy (the cluster env)")
class EnvTurnFootprintTest(unittest.TestCase):
    """The env with turn_footprint_radius 0.25 (and off) on the golden's scene and rows, live template, stub
    renderer; the camera placed with env.view_engine.reset at poses where the task is not met."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        write_scene(cls.golden, cls.root / "scenes")
        cls.cfgs = env_configs(cls.root / "scenes")
        labels = cls.golden["scene"]["labels.json"]
        cls.rows = {low_box_at(labels, np.array(r["init_camera"]["extrinsics"], float)[:2, 3]): r
                    for r in cls.golden["rows"]}
        cls.patch = mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image",
                               lambda self, frame=stub_frame(): frame.copy())
        cls.patch.start()
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls._tmp.cleanup()

    def env(self, row=None, config="S0", **switches):
        type(self).count += 1
        env = make_env(dict(self.cfgs[config], **switches), row or self.golden["rows"][0],
                       self.root / "envs" / f"e{self.count}.jsonl")
        env.reset(seed=0)
        return env

    def env_over(self, rows, config="S0", **switches):
        """A fresh env over ``rows`` in this order (episode i is rows[i]), not reset."""
        from vagen.envs.active_spatial.env import ActiveSpatialEnv
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        type(self).count += 1
        path = self.root / "envs" / f"e{self.count}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg = dict(self.cfgs[config], **switches)
        return ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                                          if k in ActiveSpatialEnvConfig.__dataclass_fields__},
                                                       jsonl_path=str(path)))

    def at(self, xy_yaw, radius=0.25, row=None, config="S0"):
        """A fresh env with the switch at ``radius`` (None: off) and the camera at ``xy_yaw``, where the task is not
        met (else the probe would end the episode)."""
        env = self.env(row, config, turn_footprint_radius=radius)
        place(env, xy_yaw)
        self.assertFalse(met(env), xy_yaw)
        return env

    def test_off_by_default_and_outside_the_prompt(self):
        from vagen.envs.active_spatial.dataset_contract import PROTOCOL_KEYS
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        off = ActiveSpatialEnvConfig()
        self.assertIsNone(off.turn_footprint_radius)
        self.assertNotIn("turn_footprint_radius", PROTOCOL_KEYS)
        self.assertNotIn("turn_footprint_radius", off.system_prompt_arguments("fov_inclusion"))
        on = ActiveSpatialEnvConfig(turn_footprint_radius=0.25)
        self.assertEqual((on.cache_key, on.config_id(), on.system_prompt_arguments("equidistance")),
                         (off.cache_key, off.config_id(), off.system_prompt_arguments("equidistance")))
        for row in self.golden["rows"]:
            self.assertEqual(self.env(row, turn_footprint_radius=0.25).system_prompt(), self.env(row).system_prompt())

    def test_the_golden_with_the_switch_given_as_none(self):
        """The C1 golden with turn_footprint_radius=None given explicitly (SwitchOffGoldenTest: absent)."""
        write_scene(self.golden, self.root / "none" / "scenes")
        got = run_all(self.golden, self.root / "none", extra={"turn_footprint_radius": None})
        self.assertEqual(got, self.golden["episodes"])

    def test_a_turn_above_the_tea_table(self):
        """Switch on: the turn is blocked (rotation bit-identical, -0.15, the detailed feedback), the move after it
        runs along the old heading; switch off: the same turn rotates."""
        from data_gen.interiorgs_v10 import rl_checks as RC

        env = self.at(TEA_TABLE)
        before = env.view_engine.get_pose()
        rec = RC.step(env, "<action>turn_left|move_forward|</action>")
        after = env.view_engine.get_pose()
        self.assertTrue(np.array_equal(after[:3, :3], before[:3, :3]))
        self.assertTrue(np.array_equal(after[:3, 3], before[:3, 3] + before[:3, :3] @ np.array([0.0, 0.0, 1.0]) * 0.3))
        self.assertEqual((rec["env_step"], rec["collisions"], rec["done"]), (2, 1, False))
        self.assertEqual(rec["components"]["collision"], {"applied": -0.15, "scale": -0.15, "events": 1})
        self.assertAlmostEqual(rec["reward"], -0.16, places=12)  # S0: the blocked move and the step penalty
        self.assertAlmostEqual(rec["reconstruction_error"], 0.0, places=12)
        self.assertEqual((env.collision_count, env.last_collision_result.collision_type), (1, "object"))
        self.assertIn("Environment Feedback: 1 of 2 actions executed; blocked: turn_left (blocked by tea table).\n",
                      rec["obs"]["obs_str"])
        # two more turns, still inside the footprint: both blocked, counted per turn
        obs, _, _, info = env.step("<action>turn_right|turn_right|</action>")
        tm = info["metrics"]["turn_metrics"]
        self.assertEqual((tm["collision_count"], tm["turn_footprint_blocks"], info["env_step"]), (2, 2, 4))
        self.assertEqual(info["metrics"]["traj_metrics"]["total_collisions"], 3)
        self.assertTrue(np.array_equal(env.view_engine.get_pose(), after))
        # switch off: it rotates 20 degrees on the spot, nothing is blocked, no new turn metric
        off = self.at(TEA_TABLE, radius=None)
        obs, reward, _, info = off.step("<action>turn_left|</action>")
        pose = off.view_engine.get_pose()
        self.assertAlmostEqual(heading_deg(pose), 20.0, places=9)
        self.assertTrue(np.array_equal(pose[:3, 3], before[:3, 3]))
        self.assertEqual((info["env_feedback"], info["metrics"]["turn_metrics"]["collision_count"]), ("Action executed.", 0))
        self.assertNotIn("turn_footprint_blocks", info["metrics"]["turn_metrics"])

    def test_the_wall_and_the_open_floor(self):
        on = self.at(WEST_WALL)
        _, _, _, info = on.step("<action>turn_left|</action>")
        self.assertEqual(info["env_feedback"], "0 of 1 actions executed; blocked: turn_left (blocked by wall).")
        self.assertEqual(info["metrics"]["turn_metrics"]["turn_footprint_blocks"], 1)
        self.assertAlmostEqual(heading_deg(on.view_engine.get_pose()), 180.0, places=9)
        off = self.at(WEST_WALL, radius=None)
        _, _, _, info = off.step("<action>turn_left|</action>")
        self.assertEqual(info["env_feedback"], "Action executed.")
        self.assertAlmostEqual(heading_deg(off.view_engine.get_pose()), -160.0, places=9)
        floor = self.at((3.0, 1.0, 0.0))  # 0.9 m from the south wall, 1.1 m from the tea table
        _, _, _, info = floor.step("<action>turn_left|</action>")
        self.assertEqual((info["env_feedback"], info["metrics"]["turn_metrics"]["turn_footprint_blocks"]),
                         ("Action executed.", 0))
        self.assertAlmostEqual(heading_deg(floor.view_engine.get_pose()), 20.0, places=9)
        # the rule reads the position of each turn: blocked inside the tea table, free once two moves left it
        env = self.at((3.3, 2.5, 0.0))
        _, _, _, info = env.step("<action>turn_left|move_forward|move_forward|turn_left|</action>")
        self.assertEqual(info["env_feedback"], "3 of 4 actions executed; blocked: turn_left (blocked by tea table).")
        self.assertAlmostEqual(heading_deg(env.view_engine.get_pose()), 20.0, places=9)
        self.assertAlmostEqual(env.view_engine.get_pose()[0, 3], 3.9, places=9)

    def test_translations_keep_their_rule(self):
        """From the sofa start (inside the sofa's footprint: every turn would be blocked there), answers without a
        turn give the same outputs with the switch on and off (one move meets the west wall); only the new turn
        metric (0) is added with the switch on."""
        row = self.rows["multi person sofa"]
        answers = ["<action>move_backward|move_backward|</action>", "<action>move_left|move_left|move_forward|</action>",
                   "<action>move_right|move_right|move_right|move_right|move_right|</action>"]
        runs = {}
        for radius in (None, 0.25):
            env = self.env(row, turn_footprint_radius=radius)
            self.assertTrue(env.collision_detector.check_turn_footprint(env.view_engine.get_pose()[:3, 3], 0.25).has_collision)
            runs[radius] = []
            for text in answers:
                runs[radius].append(step_record(env, *env.step(text)))
                if runs[radius][-1]["done"]:
                    break
        self.assertEqual(len(runs[None]), len(runs[0.25]))
        self.assertGreater(sum(turn_metrics_of(s)["collision_count"] for s in runs[None]), 0)
        self.assertFalse(any("turn_footprint_blocks" in turn_metrics_of(s) for s in runs[None]))
        for off, on in zip(runs[None], runs[0.25]):
            self.assertEqual(turn_metrics_of(on).pop("turn_footprint_blocks"), 0)
            self.assertEqual(on, off)

    def test_turning_in_place_until_the_action_limit(self):
        """Three answers of five turn_left inside the footprint: 5 / 10 / 12 actions, the limit ends the episode,
        every turn blocked, the pose never changes."""
        env = self.at(TEA_TABLE)
        start = env.view_engine.get_pose()
        got = []
        for _ in range(3):
            obs, reward, done, info = env.step("<action>" + "turn_left|" * 5 + "</action>")
            tm = info["metrics"]["turn_metrics"]
            got.append((info["env_step"], done, tm["collision_count"], tm["turn_footprint_blocks"]))
            self.assertTrue(np.array_equal(env.view_engine.get_pose(), start))
        self.assertEqual(got, [(5, False, 5, 5), (10, False, 5, 5), (12, True, 2, 2)])
        self.assertTrue(info.get("truncated_max_primitive_steps"))
        self.assertEqual(info["termination_reason"], "max_primitive_steps")

    def test_a_blocked_turn_earns_what_a_blocked_move_does(self):
        """At the west wall (the turn blocked by the footprint, move_forward by today's rule): the same reward, its
        parts and the same pose, under S0 and under V4 (whose progress term sees an unchanged pose either way)."""
        for config in ("S0", "V4"):
            got = {}
            for text in ("<action>turn_left|</action>", "<action>move_forward|</action>"):
                env = self.at(WEST_WALL, config=config)
                before = env.view_engine.get_pose()
                obs, reward, done, info = env.step(text)
                self.assertTrue(np.array_equal(env.view_engine.get_pose(), before))
                self.assertEqual((info["metrics"]["turn_metrics"]["collision_count"], done), (1, False))
                comps = info["reward_trace"]["components"]
                got[text] = (reward, {k: (v["applied"], v["events"]) for k, v in comps.items()},
                             info["current_potential_score"])
            self.assertEqual(got["<action>turn_left|</action>"], got["<action>move_forward|</action>"], config)
            if config == "S0":
                self.assertAlmostEqual(got["<action>turn_left|</action>"][0], -0.16, places=12)

    def test_footprint_blocks_count_as_consecutive_collisions(self):
        """(review fix 6a) Turns blocked by the footprint add up in the consecutive collision count across answers, as
        blocked moves do, and an executed move sets it back to 0 (the template's max_consecutive_collisions 0: nothing
        ends the episode)."""
        env = self.at(TEA_TABLE)
        got = []
        for text in ("<action>turn_left|turn_left|turn_left|</action>", "<action>turn_right|turn_right|</action>",
                     "<action>move_forward|</action>", "<action>turn_left|</action>"):  # the move stays above the table
            _, _, done, info = env.step(text)
            tm, traj = info["metrics"]["turn_metrics"], info["metrics"]["traj_metrics"]
            got.append((tm["turn_footprint_blocks"], tm["collision_count"], tm["consecutive_collision_count"],
                        info["consecutive_collision_count"], env.consecutive_collision_count,
                        traj["max_consecutive_collisions"], traj["total_collisions"], done))
        self.assertEqual(got, [(3, 3, 3, 3, 3, 3, 3, False), (2, 2, 5, 5, 5, 5, 5, False),
                               (0, 0, 0, 0, 0, 5, 5, False), (1, 1, 1, 1, 1, 5, 6, False)])

    def test_one_answer_with_a_footprint_blocked_turn_and_a_wall_blocked_move(self):
        """(review fix 6b) At the west wall a turn is blocked by the footprint (the slab 0.2 m away) and move_forward by
        today's rule (into the slab): one answer with both, in either order, gets one feedback naming both, 2
        collisions of which 1 by the footprint, S0's two collision penalties and step penalty, and keeps the pose."""
        for first, second in (("turn_left", "move_forward"), ("move_forward", "turn_left")):
            env = self.at(WEST_WALL)
            before = env.view_engine.get_pose()
            obs, reward, done, info = env.step(f"<action>{first}|{second}|</action>")
            tm = info["metrics"]["turn_metrics"]
            self.assertEqual(info["env_feedback"], f"0 of 2 actions executed; blocked: {first} (blocked by wall); "
                                                   f"{second} (blocked by wall).")
            self.assertIn(f"Environment Feedback: {info['env_feedback']}\n", obs["obs_str"])
            self.assertEqual((tm["collision_count"], tm["turn_footprint_blocks"], info["env_step"], done), (2, 1, 2, False))
            collision = info["reward_trace"]["components"]["collision"]
            self.assertEqual(collision["events"], 2)
            self.assertAlmostEqual(collision["applied"], -0.30, places=12)
            self.assertAlmostEqual(reward, -0.31, places=12)  # S0: two blocked actions and the step penalty
            self.assertTrue(np.array_equal(env.view_engine.get_pose(), before))

    def test_three_blocked_turns_end_the_episode_as_three_blocked_moves(self):
        """(review fix 6c) With max_consecutive_collisions 3 (the real runs have 0), three turns blocked by the double
        bed's footprint end the episode exactly as three moves blocked there by today's rule: the same reward and its
        parts (consecutive_collision's -0.5 included), the termination reason, three actions taken (the fourth is not),
        the pose unchanged; under S0 and V4."""
        row = self.golden["rows"][0]
        for config in ("S0", "V4"):
            got = {}
            for action in ("turn_left", "move_right"):
                env = self.env(row, config, turn_footprint_radius=0.25, max_consecutive_collisions=3)
                place(env, BED_EAST)
                self.assertFalse(met(env))
                before = env.view_engine.get_pose()
                _, reward, done, info = env.step("<action>" + f"{action}|" * 4 + "</action>")
                self.assertTrue(np.array_equal(env.view_engine.get_pose(), before), (config, action))
                comps = info["reward_trace"]["components"]
                got[action] = (reward, done, info["env_step"], info["termination_reason"],
                               bool(info.get("early_terminated_collision")), info["consecutive_collision_count"],
                               info["metrics"]["turn_metrics"]["collision_count"],
                               {k: (v["applied"], v["events"]) for k, v in sorted(comps.items())})
                self.assertEqual(info["metrics"]["turn_metrics"]["turn_footprint_blocks"], 3 if action == "turn_left" else 0)
            self.assertEqual(got["turn_left"], got["move_right"], config)
            self.assertEqual(got["turn_left"][1:7], (True, 3, "consecutive_collision", True, 3, 3), config)
            self.assertEqual(got["turn_left"][7]["consecutive_collision"], (-0.5, 1), config)
            self.assertAlmostEqual(got["turn_left"][0], 3 * -0.15 - 0.5, places=12)

    def test_a_blocked_turn_gets_the_potential_term_of_a_blocked_move(self):
        """(review fix 6d) V4 inside the double bed's footprint where the potential phi is positive (BED_EAST): after a
        blocked move there (so the env's last potential is that pose's), a blocked turn and a blocked move both keep
        the pose bit-identical and get the same, non-zero potential term scale * (gamma * phi - phi) and the same
        reward and parts."""
        row = self.golden["rows"][0]
        got = {}
        for action in ("turn_left", "move_right"):
            env = self.at(BED_EAST, row=row, config="V4")
            hit = env.collision_detector.check_turn_footprint(env.view_engine.get_pose()[:3, 3], 0.25)
            self.assertEqual((hit.collision_type, hit.collision_object, hit.details["inside_object"]),
                             ("object", "double bed", True))
            _, _, _, info = env.step("<action>move_right|</action>")  # blocked by the slab: the pose's potential
            self.assertEqual(info["metrics"]["turn_metrics"]["collision_count"], 1)
            phi = env.prev_potential_score
            before = env.view_engine.get_pose()
            _, reward, done, info = env.step(f"<action>{action}|</action>")
            tm = info["metrics"]["turn_metrics"]
            self.assertTrue(np.array_equal(env.view_engine.get_pose(), before), action)
            self.assertEqual((tm["collision_count"], tm["turn_footprint_blocks"], done),
                             (1, int(action == "turn_left"), False))
            comps = info["reward_trace"]["components"]
            got[action] = (reward, {k: (v["applied"], v["events"]) for k, v in sorted(comps.items())},
                           info["current_potential_score"], phi)
        self.assertEqual(got["turn_left"], got["move_right"])
        reward, comps, phi_after, phi = got["turn_left"]
        self.assertGreater(phi, 0.3)  # 0.408 here
        self.assertEqual(phi_after, phi)
        cfg = self.cfgs["V4"]
        term = cfg["potential_field_reward_scale"] * (cfg["potential_field_gamma"] * phi - phi)
        self.assertNotEqual(term, 0.0)
        self.assertEqual(comps["potential"][1], 1)
        self.assertAlmostEqual(comps["potential"][0], term, places=12)
        self.assertAlmostEqual(reward, -0.15 - 0.01 + term, places=12)

    def test_one_env_follows_the_scene_of_each_reset(self):
        """(review fix 6e) One env reset alternately on a row of the golden scene and one of a second scene (the tea
        table renamed coffee table, the sofa moved 4 m east): at the same two points the footprint verdict and the
        blocking object's name are always those of the scene just reset."""
        row_a = self.golden["rows"][0]
        name_b = self.golden["scene"]["name"] + "_b"
        labels = json.loads(json.dumps(self.golden["scene"]["labels.json"]))
        for obj in labels:
            if obj["label"] == "tea table":
                obj["label"] = "coffee table"
            elif obj["label"] == "multi person sofa":
                for corner in obj["bounding_box"]:
                    corner["x"] += 4.0
        write_scene({"scene": {"name": name_b, "labels.json": labels,
                               "structure.json": self.golden["scene"]["structure.json"]}}, self.root / "scenes")
        row_b = dict(row_a, scene_id=name_b, task_id=row_a["task_id"] + "_b")
        env = self.env_over([row_a, row_b], turn_footprint_radius=0.25)
        tea, sofa = TEA_TABLE[:2], SOFA
        want = {0: {tea: ("blocked: turn_left (blocked by tea table).", 1),
                    sofa: ("blocked: turn_left (blocked by multi person sofa).", 1)},
                1: {tea: ("blocked: turn_left (blocked by coffee table).", 1), sofa: ("Action executed.", 0)}}
        for seed in (0, 1, 0, 1, 1, 0):
            env.reset(seed=seed)
            self.assertEqual(env.collision_detector._current_scene_id, (row_a, row_b)[seed]["scene_id"])
            for xy in (tea, sofa):
                place(env, (xy[0], xy[1], 0.0))
                self.assertFalse(met(env))
                _, _, done, info = env.step("<action>turn_left|</action>")
                feedback, blocks = want[seed][xy]
                self.assertTrue(info["env_feedback"].endswith(feedback), (seed, xy, info["env_feedback"]))
                self.assertEqual((info["metrics"]["turn_metrics"]["turn_footprint_blocks"], done), (blocks, False))

    def test_settings_that_cannot_apply_the_rule_are_refused(self):
        for bad in ({"turn_footprint_radius": 0.0}, {"turn_footprint_radius": -0.25},
                    {"turn_footprint_radius": 0.25, "enable_collision_detection": False},
                    {"turn_footprint_radius": 0.25, "wall_model": "room_edges"},
                    {"turn_footprint_radius": 0.25, "gs_root": ""}, {"turn_footprint_radius": 0.25, "gs_root": None}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.env(**bad)
        with self.assertRaisesRegex(ValueError, "needs gs_root"):
            self.env_over(self.golden["rows"][:1], turn_footprint_radius=0.25, gs_root="")
        # switch off, an env without scenes still builds (nothing new is refused)
        self.assertIsNone(self.env_over(self.golden["rows"][:1], gs_root="").config.turn_footprint_radius)

    def test_a_row_whose_scene_is_not_loaded_stops_the_reset(self):
        """Switch on: env.reset refuses a row whose scene the detector has not loaded -- no such scene folder under
        gs_root, or no scene id (the last row's scene would still be loaded) -- when the episode starts, not at its
        first turn; the check at each turn stays as a backstop.  Switch off, nothing new is raised (the missing scene
        fails the canonical rows' own check, as before 3A)."""
        good = self.golden["rows"][0]
        missing = dict(good, scene_id="synth_no_such_scene", task_id=good["task_id"] + "_missing")
        nameless = {k: v for k, v in dict(good, task_id=good["task_id"] + "_nameless").items() if k != "scene_id"}
        env = self.env_over([good, missing, nameless], turn_footprint_radius=0.25)
        for seed, scene in ((1, "synth_no_such_scene"), (2, None)):
            env.reset(seed=0)  # the good row's scene is loaded first (the nameless row would find it still there)
            self.assertEqual((env.collision_detector.scene_loaded, env.collision_detector._current_scene_id),
                             (True, good["scene_id"]))
            with self.assertRaisesRegex(RuntimeError, "turn_footprint_radius needs the scene's collision data") as caught:
                env.reset(seed=seed)
            self.assertIn(f"scene {scene!r} not loaded", str(caught.exception))
        # the good row loads again and its turns are judged
        env.reset(seed=0)
        place(env, TEA_TABLE)
        _, _, _, info = env.step("<action>turn_left|</action>")
        self.assertEqual(info["metrics"]["turn_metrics"]["turn_footprint_blocks"], 1)
        # the backstop: a turn judged without the scene's collision data is refused at step time
        env.collision_detector.scene_loaded = False
        with self.assertRaisesRegex(RuntimeError, "turn_footprint_radius needs the scene's collision data"):
            env.step("<action>turn_left|</action>")
        off = self.env_over([good, missing])
        off.reset(seed=0)
        with self.assertRaisesRegex(RuntimeError, "R1 canonical rows require loaded collision scene data"):
            off.reset(seed=1)

    def test_the_trainers_env_class_passes_the_setting(self):
        try:
            from vagen.envs.active_spatial.active_spatial_env import ActiveSpatialGymEnv
        except ImportError as exc:  # pragma: no cover - the trainer's package is missing here
            self.skipTest(f"vagen.envs.gym_image_env: {exc}")
        path = self.root / "envs" / "gym.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.golden["rows"][0]) + "\n")
        cfg = dict(self.cfgs["S0"], turn_footprint_radius=0.25, jsonl_path=str(path))
        self.assertEqual(ActiveSpatialGymEnv(env_config=cfg)._inner.config.turn_footprint_radius, 0.25)
        self.assertIsNone(ActiveSpatialGymEnv(env_config=dict(cfg, turn_footprint_radius=None))._inner.config.turn_footprint_radius)


if __name__ == "__main__":
    unittest.main()
