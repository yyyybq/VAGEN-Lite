#!/usr/bin/env python3
"""N series: the Active Spatial tasks in RoboCasa (the user's confirmation of 2026-10-30, design
reports/N_DESIGN_20261028.md).  The RoboCasa side: the robot's mobile base moved in the upstream action vocabulary,
camera A, collisions, the scene's entities and their boxes, and the upstream success rule, all on a live RoboCasa
environment (gym.make('robocasa/<task>', split='target', seed=s) reset as the closed-loop evaluation does).

* Base: PandaOmron's three base joints (mobilebase0_joint_mobile_forward / _side: slides along robot0_base's x / y,
  which do not turn with the base; mobilebase0_joint_mobile_yaw: the hinge, last in the chain) are set directly and
  mj_forward is run (a pose, not a motion).  move_forward / backward / left / right shift the rotation axis by 0.3 m
  along / across the base's current heading, turn_left / right turn it by 20 degrees; the arm keeps its reset pose.
* Camera A (the user's choice A): robot0_agentview_center (on mobilebase0_support, which turns with the base; the
  policy never uses it) is re-placed at run time (model.cam_pos / cam_quat / cam_fovy, in memory only): 1.5 m above
  the floor straight over the rotation axis, level, looking along the heading, rendered at 320 x 240 with a vertical
  field of view of 73.74 degrees (90 degrees horizontally) and resized to 256 x 256: the geometry of the upstream
  canonical H1 camera (native K fx = fy = 320, 640 x 480, resized to 256 x 256).
* Collision: after a move, any contact deeper than 1 mm between a robot geom (robot0_*, mobilebase0_*, gripper0_*)
  and anything but the floor blocks it (the pose goes back); the feedback names the other geom's entity.
* Entities: X0's (objects by category, compact fixtures); a box is the union of its visible geoms' world boxes
  (MuJoCo geom_aabb).  Only names unique in the scene are used.
* Success (left / right, the upstream rule with the visibility rule replaced as the user decided): the upstream
  canonical projective judge (vagen/envs/active_spatial/canonical_task_metrics.py: both in front, visible, area >=
  0.01% of the image, at least half the box inside the frame, A left / right of B by >= 12 px), the upstream 45-degree
  rule (target_visibility.lateral_gate), and A and B each with at least 0.3% of the image in the segmentation image
  (the upstream MIN_FACE_AREA_FRACTION) instead of the InteriorGS face rule.  The upstream modules are loaded from the
  main checkout by path (their package __init__ is not run).
"""
from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

MAIN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "x"))

STEP_M, TURN_DEG = 0.3, 20.0
CAM_NAME = "robot0_agentview_center"
CAM_HEIGHT_M = 1.5
RENDER_W, RENDER_H = 320, 240
FOVY_DEG = math.degrees(2 * math.atan(RENDER_H / RENDER_W))  # 73.74: 90 degrees across the 4:3 width
OUT_SIZE = 256
MIN_VISIBLE_FRACTION = 0.003  # the upstream MIN_FACE_AREA_FRACTION, on the segmentation image
ROBOT_PREFIXES = ("robot0_", "mobilebase0_", "gripper0_")
FLOOR_PREFIXES = ("floor_",)
PENETRATION_M = 1e-3
ACTIONS = ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")
NATIVE_K = [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
METRIC_V1 = "canonical_spatial_task_h1_v1"


def upstream(module: str, package_dir: Path, name: str):
    """A module of the main checkout loaded by path inside a stand-in package (relative imports work, the
    package's __init__ is not run).  E.g. upstream("canonical_task_metrics", MAIN/'vagen/envs/active_spatial',
    "as_up")."""
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(package_dir)]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.{module}")


def active_spatial(module: str):
    return upstream(module, MAIN / "vagen/envs/active_spatial", "n_up_active_spatial")


# ------------------------------------------------------------------------------------------ poses (pure)
def apply_action(pose, action: str):
    """(x, y, yaw) after one upstream action: moves along / across the heading, turns by TURN_DEG."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    if action == "move_forward":
        return x + STEP_M * c, y + STEP_M * s, yaw
    if action == "move_backward":
        return x - STEP_M * c, y - STEP_M * s, yaw
    if action == "move_left":
        return x - STEP_M * s, y + STEP_M * c, yaw
    if action == "move_right":
        return x + STEP_M * s, y - STEP_M * c, yaw
    if action == "turn_left":
        return x, y, yaw + math.radians(TURN_DEG)
    if action == "turn_right":
        return x, y, yaw - math.radians(TURN_DEG)
    raise ValueError(action)


def pose_key(pose, grid: float = 0.05):
    x, y, yaw = pose
    return round(x / grid), round(y / grid), round(math.degrees(yaw) / TURN_DEG) % int(360 / TURN_DEG)


def c2w_of(pose):
    """The upstream camera-to-world matrix (OpenCV axes: right, down, forward) of camera A at a base pose."""
    import numpy as np

    x, y, yaw = pose
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, [x, y, CAM_HEIGHT_M]
    return c2w


def projective_item(scene: str, a: dict, b: dict, relation: str, start_pose) -> dict:
    """The upstream canonical v1 projective item (data_gen/interiorgs_v10/sampling.projective_item's fields)."""
    return {
        "scene_id": scene, "task_type": "projective_relations",
        "task_description": f"Move so that the {a['name']} appears to the {relation} of the {b['name']} in your view.",
        "target_object": {"objects": [entry(a), entry(b)]},
        "target_region": {"type": "projective", "params": {"relation": relation, "object_a_center": a["center"],
                                                            "object_b_center": b["center"]}},
        "init_camera": {"intrinsics": NATIVE_K, "extrinsics": c2w_of(start_pose).tolist()},
        "canonical_task_metric_version": METRIC_V1, "camera_model_version": "canonical_camera_h1_resize_v1",
        "action_protocol_version": "r1_strafe6_t0.3_r20_max12",
    }


def entry(e: dict) -> dict:
    return {"id": e["key"], "label": e["name"], "bbox_min": list(e["bbox_min"]), "bbox_max": list(e["bbox_max"]),
            "center": list(e["center"])}


def box_judge(item: dict, pose) -> dict:
    """The upstream canonical metric and 45-degree rule at a pose (no rendering)."""
    ctm = active_spatial("canonical_task_metrics")
    tv = active_spatial("target_visibility")
    c2w = c2w_of(pose)
    metric = ctm.score_canonical_task(item, c2w)
    lat = tv.lateral_gate(item, c2w)
    return {"success": bool(metric["success"] and lat["ok"]), "metric": metric, "lateral": lat}


def visible_fractions(seg_ids, entities_geoms: dict) -> dict:
    """{key: share of the image} of each entity's geoms in a segmentation image of geom ids."""
    import numpy as np

    seg = np.asarray(seg_ids)
    total = float(seg.size)
    return {k: float(np.isin(seg, list(g)).sum()) / total for k, g in entities_geoms.items()}


# ------------------------------------------------------------------------------------------- simulator side
class Scene:
    """One reset RoboCasa scene with the base, camera A and the entities (see the module text)."""

    def __init__(self, task: str, seed: int, split: str = "target"):
        import gymnasium as gym
        import mujoco
        import numpy as np
        import robocasa.wrappers.gym_wrapper  # noqa: F401

        import x0_qa

        self.task, self.seed = task, seed
        self.env = gym.make(f"robocasa/{task}", enable_render=True, split=split, seed=seed)
        self.env.reset(seed=seed)
        self.k = self.env.unwrapped.env
        self.m, self.d = self.k.sim.model._model, self.k.sim.data._data
        m, d = self.m, self.d
        self.mj, self.np = mujoco, np
        jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side",
                         "mobilebase0_joint_mobile_yaw")}
        self.jf, self.js, self.jy = (jid[n] for n in jid)
        self.adr = [int(m.jnt_qposadr[j]) for j in (self.jf, self.js, self.jy)]
        self.vadr = [int(m.jnt_dofadr[j]) for j in (self.jf, self.js, self.jy)]
        parent = m.body_parentid[m.jnt_bodyid[self.jy]]
        self.Rp = np.asarray(d.xmat[parent]).reshape(3, 3).copy()
        self.yaw0 = math.atan2(self.Rp[1, 0], self.Rp[0, 0])  # robot0_base's heading
        anchor = np.asarray(d.xanchor[self.jy], float)
        qf, qs = d.qpos[self.adr[0]], d.qpos[self.adr[1]]
        self.A0 = anchor - self.Rp[:, 0] * qf - self.Rp[:, 1] * qs  # the axis when both slides are zero
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.geom_names = names
        self.robot = np.array([n.startswith(ROBOT_PREFIXES) for n in names])
        self.floor = np.array([n.startswith(FLOOR_PREFIXES) for n in names])
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        self.start = self.pose()
        self._place_camera()
        self.entities = self._entities(x0_qa.scene_entities(self.k))
        self.geom_entity = {g: e["key"] for e in self.entities for g in e["geoms"]}
        cats = {c.get("name"): str((c.get("info") or {}).get("cat") or c.get("name"))
                for c in getattr(self.k, "object_cfgs", [])}
        self.prefix_names = {}  # geom-name prefix of every object and fixture -> what the feedback calls it
        for key, obj in self.k.objects.items():
            self.prefix_names[getattr(obj, "naming_prefix", f"{key}_")] = x0_qa.display_name(cats.get(key, key))
        for key, fx in self.k.fixtures.items():
            nat = getattr(fx, "nat_lang", None)
            self.prefix_names[getattr(fx, "naming_prefix", f"{key}_")] = x0_qa.display_name(
                key if not nat or nat == key else nat)

    def close(self):
        self.env.close()

    # pose
    def pose(self):
        q = self.d.qpos
        anchor = self.np.asarray(self.d.xanchor[self.jy], float)
        return float(anchor[0]), float(anchor[1]), self.yaw0 + float(q[self.adr[2]])

    def set_pose(self, pose) -> None:
        np = self.np
        x, y, yaw = pose
        rhs = np.array([x, y]) - self.A0[:2]
        qf, qs = np.linalg.solve(self.Rp[:2, :2], rhs)
        for a, v in zip(self.adr, (qf, qs, yaw - self.yaw0)):
            self.d.qpos[a] = v
        for a in self.vadr:
            self.d.qvel[a] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def collision(self):
        """None, or the name of what a robot geom penetrates (the floor does not count)."""
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -PENETRATION_M:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.robot[g1] == self.robot[g2]:
                continue
            other = g2 if self.robot[g1] else g1
            if self.floor[other]:
                continue
            key = self.geom_entity.get(other)
            return key or self.geom_names[other] or "object"
        return None

    def owner(self, hit: str) -> str:
        """What a blocking geom belongs to, as the feedback names it ("counter", "fridge", ...)."""
        for e in self.entities:
            if e["key"] == hit:
                return e["name"]
        best = max((p for p in self.prefix_names if hit.startswith(p)), key=len, default=None)
        if best is not None:
            return self.prefix_names[best] or "object"
        return "wall" if hit.startswith("wall") else "object"

    def try_action(self, pose, action: str):
        """(new pose, None) or (the old pose, what blocked it); the simulator is left at the returned pose."""
        new = apply_action(pose, action)
        self.set_pose(new)
        hit = self.collision()
        if hit is not None:
            self.set_pose(pose)
            return pose, hit
        return new, None

    # camera A
    def _place_camera(self) -> None:
        np, m, d = self.np, self.m, self.d
        self.set_pose(self.start)
        body = int(m.cam_bodyid[self.cam])
        Rb = np.asarray(d.xmat[body]).reshape(3, 3)
        pb = np.asarray(d.xpos[body])
        c2w = c2w_of(self.start)
        R_mj = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # MuJoCo cameras: x right, y up, looking along -z
        local = Rb.T @ R_mj
        quat = np.zeros(4)
        self.mj.mju_mat2Quat(quat, local.reshape(-1))
        m.cam_pos[self.cam] = Rb.T @ (c2w[:3, 3] - pb)
        m.cam_quat[self.cam] = quat
        m.cam_fovy[self.cam] = FOVY_DEG
        self.mj.mj_forward(m, d)

    def render(self, pose=None):
        """(256 x 256 RGB upright, geom-id segmentation at 320 x 240 upright) of camera A at a pose."""
        from PIL import Image

        np = self.np
        if pose is not None:
            self.set_pose(pose)
        rgb = self.k.sim.render(camera_name=CAM_NAME, width=RENDER_W, height=RENDER_H)
        seg = np.asarray(self.k.sim.render(camera_name=CAM_NAME, width=RENDER_W, height=RENDER_H, segmentation=True))
        geom = np.where(seg[..., 0] == 5, seg[..., 1], -1) if seg.ndim == 3 else seg
        img = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[::-1])).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
        return np.asarray(img), np.ascontiguousarray(geom[::-1])

    def camera_matches(self, pose) -> float:
        """Distance (m) plus angle (rad) between camera A's simulated pose and c2w_of(pose): a self-check."""
        np = self.np
        self.set_pose(pose)
        p = np.asarray(self.d.cam_xpos[self.cam])
        R = np.asarray(self.d.cam_xmat[self.cam]).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        c2w = c2w_of(pose)
        cos = (np.trace(R.T @ c2w[:3, :3]) - 1) / 2
        return float(np.linalg.norm(p - c2w[:3, 3]) + math.acos(max(-1.0, min(1.0, cos))))

    # entities
    def _entities(self, ents) -> list:
        np, m, d = self.np, self.m, self.d
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        out = []
        for e in ents:
            lo, hi, used = np.full(3, np.inf), np.full(3, -np.inf), []
            for g in sorted(e["geoms"]):
                if m.geom_rgba[g][3] < 0.01:  # removed elements are made transparent and moved away
                    continue
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                pts = np.asarray(d.geom_xpos[g]) + (c + h * signs) @ np.asarray(d.geom_xmat[g]).reshape(3, 3).T
                lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
                used.append(g)
            if used and np.all(np.isfinite(lo)):
                out.append(dict(key=e["key"], name=e["name"], kind=e["kind"], geoms=set(used),
                                bbox_min=[float(v) for v in lo], bbox_max=[float(v) for v in hi],
                                center=[float(v) for v in (lo + hi) / 2]))
        counts = {}
        for e in out:
            counts[e["name"]] = counts.get(e["name"], 0) + 1
        return [e for e in out if counts[e["name"]] == 1]

    def judge(self, item: dict, pose, a_key: str, b_key: str, render: bool = True) -> dict:
        """The success rule at a pose: the upstream box judge and 45-degree rule, and (render) the
        segmentation visibility of A and B."""
        out = box_judge(item, pose)
        out["visible"] = None
        if render:
            _, seg = self.render(pose)
            geoms = {e["key"]: e["geoms"] for e in self.entities if e["key"] in (a_key, b_key)}
            frac = visible_fractions(seg, geoms)
            out["visible"] = {"a": frac[a_key], "b": frac[b_key]}
            out["success"] = bool(out["success"] and frac[a_key] >= MIN_VISIBLE_FRACTION
                                  and frac[b_key] >= MIN_VISIBLE_FRACTION)
        return out
