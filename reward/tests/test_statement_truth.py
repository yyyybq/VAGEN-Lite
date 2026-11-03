"""Tests of vagen/statement_truth.py on a synthetic room (no scene files, no rendering)."""

import numpy as np
import pytest

from vagen.envs.active_spatial.target_visibility import OccluderSet, score_with_visibility
from vagen.statement_truth import (
    FIELDS,
    format_statement,
    gate_group,
    missing_groups,
    parse_statement,
    score_statement,
    truth,
)

K = [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
EDGES = (1.0, 2.0)


def pose(x, y, yaw_deg):
    """c2w of a level camera at (x, y, 1.5) looking along yaw (0 = +x, 90 = +y); camera x right, y down, z forward."""
    a = np.radians(yaw_deg)
    z = np.array([np.cos(a), np.sin(a), 0.0])
    yd = np.array([0.0, 0.0, -1.0])
    xr = np.cross(yd, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = xr, yd, z, [x, y, 1.5]
    return m


def box(oid, cx, cy, size=0.4, z0=0.3):
    h = size / 2
    return {"id": oid, "label": f"box {oid}", "bbox_min": [cx - h, cy - h, z0], "bbox_max": [cx + h, cy + h, z0 + size],
            "center": [cx, cy, z0 + h], "ref": f"the box {oid}"}


def item(task_type, objs, params):
    return {"task_id": "t", "task_type": task_type, "canonical_task_metric_version": "canonical_spatial_task_h1_v2",
            "camera_model_version": "canonical_camera_h1_resize_v1",
            "init_camera": {"intrinsics": K, "extrinsics": pose(0, 0, 0).tolist()},
            "target_object": {"objects": objs}, "target_region": {"type": "x", "params": params}}


@pytest.fixture
def occ():
    return OccluderSet.from_boxes([])


def absolute(distance=2.5):
    o = box("1", 3.0, 0.0)
    return item("absolute_positioning", [o], {"distance_m": distance, "object_a_id": "1", "object_a_center": o["center"]})


def test_where_in_view_left_right_behind(occ):
    it = absolute()
    assert truth(it, pose(0, 0, 0), occ, EDGES)["where"] == {"A": "in_view"}
    assert truth(it, pose(0, 0, -90), occ, EDGES)["where"] == {"A": "left"}  # looking along -y: +x is on the left
    assert truth(it, pose(0, 0, 90), occ, EDGES)["where"] == {"A": "right"}
    assert truth(it, pose(0, 0, 180), occ, EDGES)["where"] == {"A": "behind"}


def test_dist_absolute_band(occ):
    it = absolute(2.5)  # tolerance max(0.25, 0.25) = 0.25 around 2.5 m to the footprint (near side at x = 2.8)
    assert truth(it, pose(0.4, 0, 0), occ, EDGES)["dist"] == "mid"  # 2.4 m
    assert truth(it, pose(0.0, 0, 0), occ, EDGES)["dist"] == "far"  # 2.8 m
    assert truth(it, pose(1.5, 0, 0), occ, EDGES)["dist"] == "near"  # 1.3 m


def test_dist_edges_other_family(occ):
    a, b = box("1", 3.0, 0.5), box("2", 3.0, -0.5)
    it = item("equidistance", [a, b], {"object_a_id": "1", "object_a_center": a["center"],
                                       "object_b_id": "2", "object_b_center": b["center"]})
    assert truth(it, pose(2.0, 0.5, 0), occ, EDGES)["dist"] == "near"  # 0.8 m
    assert truth(it, pose(1.3, 0.5, 0), occ, EDGES)["dist"] == "mid"  # 1.5 m
    assert truth(it, pose(0.0, 0.5, 0), occ, EDGES)["dist"] == "far"  # 2.8 m
    t = truth(it, pose(0.0, 0.0, 0), occ, EDGES)
    assert set(t["where"]) == {"A", "B"} and set(t["face"]) == {"A", "B"}


def test_occlusion_states_target_where_and_occluder_face(occ):
    a, b = box("1", 4.0, 0.0), box("2", 2.0, 0.0, size=0.8)
    it = item("occlusion_alignment", [a, b], {"object_a_id": "1", "object_a_center": a["center"],
                                              "object_b_id": "2", "object_b_center": b["center"]})
    t = truth(it, pose(0, 0, 0), occ, EDGES)
    assert list(t["where"]) == ["A"] and list(t["face"]) == ["B"]


def test_missing_agrees_with_metric(occ):
    it = absolute()
    for p in (pose(0.4, 0, 0), pose(0, 0, 0), pose(0, 0, 180), pose(1.5, 0, 0), pose(0.4, 0, 30)):
        m = score_with_visibility(it, p, occ)
        t = truth(it, p, occ, EDGES)
        failed = [k for k, v in m["gates"].items() if not v]
        assert t["success"] == bool(m["success"])
        assert t["missing"] == missing_groups(failed)
        assert (t["missing"] == []) == t["success"]
        assert t["face"]["A"] == bool(m["visibility"]["targets"]["a"]["visible"])
    assert "relation" in truth(it, pose(0, 0, 0), occ, EDGES)["missing"]  # 2.8 m: distance not met
    assert "frame" in truth(it, pose(0, 0, 180), occ, EDGES)["missing"]  # behind the camera


def test_gate_groups():
    assert gate_group("in_front") == "frame" and gate_group("occluder_visible") == "frame"
    assert gate_group("target_a_clearly_visible") == "clear" and gate_group("required_targets_found") == "clear"
    assert gate_group("reference_b_clearly_visible") == "clear"
    for g in ("distance", "equidistant", "relation", "margin", "occupancy", "lateral_within_45deg", "target_hidden"):
        assert gate_group(g) == "relation"
    assert missing_groups(["distance", "target_a_clearly_visible", "visible"]) == ["frame", "clear", "relation"]
    assert missing_groups([]) == []


def test_statement_round_trip_and_score(occ):
    it = absolute()
    t = truth(it, pose(0, 0, 0), occ, EDGES)
    text = format_statement(t)
    assert text.startswith("<observe>where=A:in_view; face=A:") and text.endswith("</observe>")
    p = parse_statement(text + "<action>move_forward|</action>")
    assert score_statement(p, t) == {"readable": True, "accuracy": 1.0, **{k: True for k in FIELDS}}
    wrong = text.replace("dist=far", "dist=near")
    s = score_statement(parse_statement(wrong), t)
    assert s["accuracy"] == 0.75 and not s["dist"] and s["where"]
    s = score_statement(parse_statement("<observe>where=A:up; face=A:maybe; dist=far</observe>"), t)
    assert s["readable"] and not s["where"] and not s["face"] and s["dist"] and not s["missing"]
    assert score_statement(parse_statement("<action>move_forward|</action>"), t)["readable"] is False
    none = parse_statement("<observe>where=A:in_view; face=A:yes; dist=mid; missing=none</observe>")
    assert none["missing"] == [] and none["face"] == {"A": True}
    order = parse_statement("<observe>missing=relation,frame</observe>")
    assert order["missing"] == ["frame", "relation"]


def test_observe_prompt_states_the_edges_and_leaves_no_think_alone():
    from vagen.envs.active_spatial import prompt as P
    from vagen.statement_truth import DIST_EDGES_M

    text = P.system_prompt(format="observe", max_actions_per_step=5, action_space="strafe", enable_explicit_done=False,
                           format_example="action_names")
    assert f"under {DIST_EDGES_M[0]:g} m, {DIST_EDGES_M[0]:g} to {DIST_EDGES_M[1]:g} m, over {DIST_EDGES_M[1]:g} m" in text
    shown = "<observe>where=A:in_view; face=A:yes; dist=mid; missing=relation</observe>"
    assert f"Response format: {shown}<action>move_forward|turn_left|</action>" in text
    for example in (shown, "<observe>where=A:in_view,B:left; face=A:yes,B:no; dist=mid; missing=clear,relation</observe>",
                    P.FORMAT_CONFIGS["observe"]["example"]):
        assert parse_statement(example) is not None and None not in parse_statement(example).values()
    block = P.format_prompt["observe"](max_actions_per_step=5, format_example="action_names")
    assert block.startswith(f"Respond with the statement, then the action:\n{shown}\n"
                            "<action>move_forward|turn_left|</action>\n")
    assert "..." not in block
    assert P.format_prompt["no_think"](max_actions_per_step=5, format_example="action_names") == (
        "Respond with only the action:\n<action>move_forward|turn_left|</action>\n\nYou can take up to 5 actions per "
        "step.\n\nExample: <action>move_forward|</action>")
    assert "<observe>" not in P.system_prompt(format="no_think", max_actions_per_step=5)


def test_env_reads_the_action_after_a_statement():
    from types import SimpleNamespace

    from vagen.envs.active_spatial.env import ActiveSpatialEnv

    text = "<observe>where=A:left; face=A:no; dist=far; missing=frame</observe><action>turn_left|move_forward|</action>"
    rst = ActiveSpatialEnv._strict_parse_func(SimpleNamespace(_allowed_actions=None), text, action_sep="|",
                                              max_actions=5)
    assert rst["format_correct"] and rst["actions"] == ["turn_left", "move_forward"]
    assert rst["strict_parse_success"]
