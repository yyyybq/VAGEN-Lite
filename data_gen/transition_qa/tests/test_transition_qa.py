"""Tests of the transition-question generator (python -m pytest data_gen/transition_qa/tests -q).

The integration tests read the round-11 SFT data and the InteriorGS metadata on the cluster and are skipped
elsewhere."""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import numpy as np
import pytest

from data_gen.interiorgs_v10.contract import ACTIONS
from data_gen.transition_qa import generate as GEN
from data_gen.transition_qa import questions as Q

CFG = dict(Q.DEFAULT_CONFIG)
SFT_DIR = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft")
META = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_meta")


def obj(**kw):
    base = {"center_in_front": True, "u": 128.0, "all_corners_in_front": True, "projected_visible": True,
            "inside_fraction": 1.0, "vertical_inside_fraction": 1.0, "occupancy": 0.3, "distance_m": 2.0,
            "clearly_visible": True, "face_visible_fraction": 0.8, "face_area_fraction": 0.05}
    base.update(kw)
    return base


# ----------------------------------------------------------------------------- texts and options
def test_action_text_uses_the_action_space():
    assert Q.action_text("turn_left", 2) == "turns left by 40 degrees"
    assert Q.action_text("move_forward", 3, past=True) == "moved forward by 0.9 meters"
    assert Q.action_text("move_left", 1) == "moves sideways to the left by 0.3 meters"


def test_inverse_options_distinct_correct_and_uniform():
    positions = collections.Counter()
    for action in ACTIONS:
        for k in range(1, 6):
            for i in range(40):
                opts, pos = Q.inverse_options(action, k, f"t{i}:{action}:{k}")
                assert len(set(opts)) == 4 and opts[pos] == (action, k)
                assert all(a in ACTIONS and 1 <= j <= 5 for a, j in opts)
                positions[pos] += 1
    shares = [positions[p] / sum(positions.values()) for p in range(4)]
    assert max(abs(s - 0.25) for s in shares) < 0.03


def test_grid_options_contain_the_value_within_bounds():
    for v in (0.5, 1.0, 2.5, 4.0):
        vals, pos = Q.grid_options(v, 0.5, 0.5, float("inf"), f"k{v}")
        assert vals[pos] == v and min(vals) >= 0.5 and np.allclose(np.diff(vals), 0.5)
    vals, pos = Q.grid_options(0.9, 0.1, 0.1, 0.9, "x")
    assert vals == [0.6, 0.7, 0.8, 0.9] and pos == 3


# ----------------------------------------------------------------------------- readers
def test_left_right_reader():
    m = {"objects": {"a": obj(u=60.0), "b": obj(u=150.0)}, "lateral_ok": True}
    r = Q.read_left_right(m, CFG, True)
    assert r.value == "left" and r.quantity == -90.0
    m["objects"]["a"]["u"] = 140.0          # 10 px apart: inside the 20 px buffer
    assert Q.read_left_right(m, CFG, True).reason == "near_boundary"
    m["objects"]["a"]["u"] = 200.0
    m["lateral_ok"] = False                 # A behind B: front / back, not left / right
    assert Q.read_left_right(m, CFG, True).reason == "beyond_45deg"
    m["lateral_ok"] = True
    m["objects"]["b"]["clearly_visible"] = False
    assert Q.read_left_right(m, CFG, True).reason == "not_visible"
    assert Q.read_left_right(m, CFG, False).value == "right"   # after an action the model does not see it


def test_closer_and_distance_readers():
    m = {"objects": {"a": obj(distance_m=2.0), "b": obj(distance_m=2.06)}}
    assert Q.read_closer(m, CFG, True).reason == "near_boundary"   # 3% apart
    m["objects"]["b"]["distance_m"] = 2.5
    assert Q.read_closer(m, CFG, True).value == "a"
    assert Q.read_distance_bin({"objects": {"a": obj(distance_m=2.12)}}, CFG, True).value == 2.0
    assert Q.read_distance_bin({"objects": {"a": obj(distance_m=2.2)}}, CFG, True).reason == "near_boundary"
    assert Q.read_distance_bin({"objects": {"a": obj(distance_m=0.2)}}, CFG, True).reason == "below_options"


def test_occluded_reader_thresholds():
    def m(h, a_visible=True):
        return {"objects": {"a": obj(clearly_visible=a_visible), "b": obj()}, "hidden_fraction": h,
                "in_frame_fraction": 1.0}
    assert Q.read_occluded(m(0.9, False), CFG, True).value == "full"
    assert Q.read_occluded(m(0.5), CFG, True).value == "partial"
    assert Q.read_occluded(m(0.0), CFG, True).value == "none"
    assert Q.read_occluded(m(0.0, False), CFG, True).reason == "target_not_visible"
    assert Q.read_occluded(m(0.8), CFG, True).reason == "near_boundary"
    out = m(0.9)
    out["in_frame_fraction"] = 0.5          # out of frame is not hidden
    assert Q.read_occluded(out, CFG, True).reason == "target_out_of_frame"


def test_both_in_view_reader():
    m = {"objects": {"a": obj(), "b": obj(inside_fraction=0.1)}}
    assert Q.read_both_in_view(m, CFG, True).value == "only_a"
    m["objects"]["b"]["inside_fraction"] = 0.7   # half in the frame: no question
    assert Q.read_both_in_view(m, CFG, True).reason == "near_boundary"
    m["objects"]["b"] = obj(center_in_front=False, u=None, all_corners_in_front=False)
    m["objects"]["a"] = obj(center_in_front=False, u=None, all_corners_in_front=False)
    assert Q.read_both_in_view(m, CFG, True).value == "neither"


def test_build_text_answers_point_at_the_value():
    item = {"task_description": "Do X."}
    roles = {"a": {"ref": "the sofa"}, "b": {"ref": "the lamp"}}
    for value in ("a", "b"):
        t = Q.build_text("closer", item, roles, value, None, f"key{value}", CFG)
        assert t["options"][Q.LETTERS.index(t["answer"])] == roles[value]["ref"]
    t = Q.build_text("distance_bin", item, roles, 2.0, ("move_forward", 2), "k", CFG)
    assert t["options"][Q.LETTERS.index(t["answer"])] == "about 2.0 meters"
    assert t["question"].startswith("If the camera moves forward by 0.6 meters")
    t = Q.build_text("goal", item, roles, True, None, "k", CFG)
    assert t["answer"] == "Yes" and "Do X." in t["question"]


def test_interior_offsets_move_in_the_camera_frame():
    from data_gen.interiorgs_v10.contract import pose_from_xy_yaw
    from data_gen.transition_qa.interior import offset_pose

    c2w = pose_from_xy_yaw(1.0, 2.0, 0.3)
    right, forward = c2w[:3, 0], c2w[:3, 2]
    p = offset_pose(c2w, -0.15, 0.3, 0.0)
    assert np.allclose(p[:3, 3], c2w[:3, 3] - 0.15 * right + 0.3 * forward)
    q = offset_pose(c2w, 0.0, 0.0, -10.0)
    assert np.allclose(q[:3, 3], c2w[:3, 3]) and not np.allclose(q[:3, :3], c2w[:3, :3])
    assert abs(q[2, 2] - c2w[2, 2]) < 1e-12   # still level


# ----------------------------------------------------------------------------- build
def _rec(i, qtype, answer, value, scene="s1", split="train", qclass=None, changed=True, flip=True):
    kind = qtype.rsplit("_", 1)[0]
    rec = {"sample_id": f"id{i:06d}", "qtype": qtype, "kind": "action" if kind == "action" else kind,
           "qclass": qclass or Q.QCLASS[qtype.rsplit("_", 1)[-1]], "answer": answer, "answer_value": value,
           "scene": scene, "split": split, "changed": changed, "flip": flip, "images": ["x.png"],
           "options": ["Yes", "No"], "question": "q", "state_pos": 0, "task_id": f"t{i}", "pool": "P1",
           "family": "projective_relations"}
    return rec


def test_balance_brings_answers_within_five_points():
    rng = random.Random(0)
    rs = [_rec(i, "goal_now", "Yes" if i % 5 == 0 else "No", "") for i in range(1000)]
    kept = GEN.balance_split(rs, GEN.BUILD_CONFIG, rng, {})
    share = sum(r["answer"] == "Yes" for r in kept) / len(kept)
    assert abs(share - 0.5) <= 0.05


def test_forward_caps_unchanged_and_flip_share():
    rng = random.Random(0)
    rs = [_rec(i, "goal_after", "Yes" if i % 2 else "No", "", changed=(i % 3 != 0), flip=(i % 10 < 9))
          for i in range(3000)]
    kept = GEN.balance_split(rs, GEN.BUILD_CONFIG, rng, {})
    unchanged = sum(not r["changed"] for r in kept) / len(kept)
    flips = sum(bool(r["flip"]) for r in kept) / len(kept)
    yes = sum(r["answer"] == "Yes" for r in kept) / len(kept)
    assert unchanged <= 0.10 + 1e-9
    assert 0.35 - 0.02 <= flips <= 0.65 + 0.02
    assert abs(yes - 0.5) <= 0.05


def test_flip_band_left_alone_when_flips_are_physically_rare():
    rng = random.Random(0)
    rs = [_rec(i, "left_right_after", "A" if i % 2 else "B", "", flip=(i < 20)) for i in range(1000)]
    kept, dropped = GEN._cap_flips(rs, (0.35, 0.65), rng)
    assert dropped == 0 and len(kept) == 1000


def test_even_letters_moves_the_correct_option_in_turn():
    rng = random.Random(0)
    rs = []
    for i in range(103):
        r = _rec(i, "action_between", "A", "")
        r["options"] = ["right one", "x", "y", "z"]
        r["question"] = "What did the camera do?\n" + Q.options_block(r["options"]) + "\n" + Q.ANSWER_LETTER
        rs.append(r)
    out = GEN.even_letters(rs, GEN.BUILD_CONFIG, rng)
    letters = collections.Counter(r["answer"] for r in out)
    assert max(letters.values()) - min(letters.values()) <= 1
    for r in out:
        assert r["options"][Q.LETTERS.index(r["answer"])] == "right one"
        assert f"({r['answer']}) right one" in r["question"]


def test_class_caps_keep_state_and_cap_forward_inverse():
    rng = random.Random(0)
    rs = ([_rec(i, "goal_now", "Yes" if i % 2 else "No", "") for i in range(800)]
          + [_rec(1000 + i, "closer_now", "A" if i % 2 else "B", "a") for i in range(200)]
          + [_rec(2000 + i, "goal_after", "Yes" if i % 2 else "No", "") for i in range(3000)]
          + [_rec(6000 + i, "closer_after", "A" if i % 2 else "B", "a") for i in range(1000)]
          + [_rec(8000 + i, "action_between", "A", "") for i in range(1000)])
    rep = {}
    out = GEN.class_caps(rs, GEN.BUILD_CONFIG, rng, rep)
    c = collections.Counter(r["qclass"] for r in out)
    assert c["state"] == 1000                                     # all kept, goal_now 80% of them included
    assert c["forward"] == 1200                                   # 1.2 x state
    assert c["inverse"] == int(0.15 / 0.85 * 2200)                # 15% of the bank
    assert c["inverse"] / len(out) <= 0.15
    fwd = [r for r in out if r["qclass"] == "forward"]
    goal = [r for r in fwd if r["kind"] == "goal"]
    assert rep["goal_cap_dropped"] == {"forward": 2000}           # goal_after capped at the other forward ones
    assert abs(len(goal) - 600) <= 1
    assert abs(sum(r["answer"] == "Yes" for r in goal) - len(goal) / 2) <= 1


def test_probe_select_per_qtype_balanced_and_spread():
    rng = random.Random(0)
    fams = ["projective_relations", "equidistance", "fov_inclusion"]
    rs = []
    for i in range(300):
        r = _rec(i, "goal_now", "Yes" if i % 4 == 0 else "No", "", split="val")
        r.update(family=fams[i % 3], task_id=f"t{i % 50}")
        rs.append(r)
    for i in range(300):
        r = _rec(1000 + i, "goal_after", "Yes" if i % 3 == 0 else "No", "", split="val",
                 changed=(i % 2 == 0), flip=(i % 5 == 0))
        r.update(family=fams[i % 3], task_id=f"u{i % 60}")
        rs.append(r)
    for i in range(10):
        rs.append(_rec(2000 + i, "occluded_now", "A", "full", split="val"))
    rep = {}
    picks = GEN.probe_select(rs, GEN.PROBE_CONFIG, rng, rep)
    by_q = collections.Counter(r["qtype"] for r in picks)
    assert by_q["goal_now"] == 40 and by_q["goal_after"] == 40
    assert by_q["occluded_now"] == 10 and rep["occluded_now"]["short"] == 30   # reported, not padded
    now = [r for r in picks if r["qtype"] == "goal_now"]
    assert sum(r["answer"] == "Yes" for r in now) == 20
    assert max(collections.Counter(r["family"] for r in now).values()) <= 16   # 40 / 3 families, task turns first
    assert len({r["task_id"] for r in now}) >= 36   # tasks take turns (greedy; 50 tasks, Yes only on 25)
    after = [r for r in picks if r["qtype"] == "goal_after"]
    assert sum(not r["changed"] for r in after) <= 4
    assert len({r["sample_id"] for r in picks}) == len(picks)


def test_all_anchor_states():
    images = [{"frame_id": f"t__s{k}"} for k in (0, 2, 2, 4, 6)]   # s2 shown twice (a blocked turn)
    picks = GEN.pick_anchor_positions(images, "t", 0)
    assert [p for p, _ in picks] == [0, 2, 3, 4]
    assert dict((p, role) for p, role in picks)[0] == "start" and picks[-1] == (4, "terminal")
    assert len(GEN.pick_anchor_positions(images, "t", 3)) == 3


def test_build_refuses_a_val_scene_in_train(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "s9.json").write_text(json.dumps({"funnel": {}, "checks": {}}))
    GEN.write_jsonl(raw / "s9.jsonl", [_rec(0, "goal_now", "Yes", "", scene="s9", split="train")])
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"train_scenes": ["s1"], "val_scenes": ["s9"], "test_scenes": []}))
    with pytest.raises(SystemExit):
        GEN.main(["build", "--out", str(tmp_path), "--split-manifest", str(manifest), "--no-probe"])


# ----------------------------------------------------------------------------- integration (cluster data)
needs_data = pytest.mark.skipif(not (SFT_DIR.is_dir() and META.is_dir()), reason="cluster data not available")


@needs_data
def test_success_and_left_right_match_the_env():
    from vagen.envs.active_spatial import target_visibility as TV
    from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task

    from data_gen.interiorgs_v10.occlusion import face_occluders_for
    from data_gen.interiorgs_v10.scene import load_scene
    from data_gen.transition_qa import geometry as G

    rows = {r["task_id"]: r for r in GEN.read_jsonl(SFT_DIR / "rows.jsonl")}
    poses = {f["frame_id"]: f["c2w"] for f in GEN.read_jsonl(SFT_DIR / "frames.jsonl")}
    trajs = GEN.read_jsonl(SFT_DIR / "sft_trajectories.jsonl")
    rng = random.Random(1)
    by_family = collections.defaultdict(list)
    for t in trajs:
        by_family[t["task_type"]].append(t)
    picked = [t for fam in sorted(by_family) for t in rng.sample(by_family[fam], 2)]
    n_states = n_lr = 0
    for t in picked:
        scene = load_scene(META / t["scene_id"], t["scene_id"], wall_model="thick")
        occ = face_occluders_for(scene)
        item = rows[t["source_task_id"]]
        geo = G.TaskGeometry(item, scene, occ)
        images, _ = GEN.trajectory_steps(t, poses)
        for im in images:
            ref = TV.score_with_visibility(item, im["c2w"], occ)
            mine = geo.success(im["c2w"])
            assert mine["success"] == ref["success"] and mine["shaping_score"] == ref["shaping_score"]
            n_states += 1
            if item["task_type"] == "projective_relations":
                m = geo.measure(im["c2w"])
                margin = score_canonical_task(item, im["c2w"])["relation_margin_px"]
                a, b = m["objects"]["a"], m["objects"]["b"]
                if margin is not None and a["u"] is not None and b["u"] is not None:
                    du = a["u"] - b["u"]
                    rel = item["target_region"]["params"].get("relation", "left")
                    assert abs((-du if rel == "left" else du) - margin) < 1e-6
                    n_lr += 1
        assert images[-1]["c2w"] is not None
    print("integration states", n_states, "left-right checks", n_lr)
    assert n_states > 20 and n_lr > 0
