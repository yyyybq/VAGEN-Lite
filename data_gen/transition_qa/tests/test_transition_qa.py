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


def test_inverse_options_are_a_symmetric_2x2():
    """2026-10-19 fix (DIAG3_PROBE.md): the options are {a, b} x {k, k2}, so no option stands out by its text (the
    earlier distractors made the answer the only option whose action repeats with the commonest count), and the
    action pairs are symmetric relations, so a pair never occurs only with one of its actions as the answer."""
    from data_gen.transition_qa import shortcuts as SC

    pairs = set()
    for action in ACTIONS:
        for k in range(1, 6):
            for i in range(30):
                opts, pos = Q.inverse_options(action, k, f"t{i}:{action}:{k}")
                texts = [f"the camera {Q.action_text(a, j, past=True)}" for a, j in opts]
                assert [SC.parse_action_option(t) for t in texts] == [(a, j) for a, j in opts]
                assert len(SC.inverse_frequency({"options": texts})) == 4      # every option qualifies: a guess
                acts = collections.Counter(a for a, _ in opts)
                counts = collections.Counter(j for _, j in opts)
                assert sorted(acts.values()) == [2, 2] and sorted(counts.values()) == [2, 2]
                assert k in counts and set(counts) - {k} <= {k - 1, k + 1}             # a neighbouring count
                pairs.add(frozenset(acts))
    for pair in pairs:  # every pair can have either action as the answer
        a, b = sorted(pair)
        assert any(frozenset(c for c, _ in Q.inverse_options(b, 2, f"s{i}")[0]) == pair for i in range(40)), pair
    assert len(pairs) == 5  # fwd-bwd, turn l-r, move l-r, turn_left-move_left, turn_right-move_right


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
    picks = GEN.probe_select(rs, dict(GEN.PROBE_CONFIG, per_qtype_overrides={}), rng, rep)
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


def _inverse_rec(i, action, k, split="train"):
    opts, pos = Q.inverse_options(action, k, f"k{i}")
    rec = _rec(i, "action_between", Q.LETTERS[pos], f"{action}x{k}", split=split)
    rec["options"] = [f"the camera {Q.action_text(a, j, past=True)}" for a, j in opts]
    return rec


def test_inverse_balance_within_cells():
    rng = random.Random(0)
    rs = [_inverse_rec(i, "move_backward" if i % 5 else "move_forward", 1 + i % 3) for i in range(3000)]
    kept, dropped = GEN._balance_inverse(rs, 1.15, rng)
    cells = collections.defaultdict(collections.Counter)
    for r in kept:
        acts, action, counts, low = Q.inverse_balance_keys(r)
        cells[(acts, counts)][(action, low)] += 1
    for groups in cells.values():  # four answers per cell, each at most 1.15 x the rarest
        assert len(groups) == 4 and max(groups.values()) <= 1.15 * min(groups.values()) + 1
    assert dropped == len(rs) - len(kept) and len(kept) > 0


def test_window_balance_hides_the_letter():
    rng = random.Random(0)
    rs = []
    for i in range(2000):
        r = _rec(i, "distance_bin_now", "ABCD"[i % 4 if i % 3 else 0], "")
        r["options"] = ["about 1.0 meters" if i % 2 else "about 1.5 meters", "b", "c", "d"]
        rs.append(r)
    rs.append(dict(_rec(9999, "distance_bin_now", "A", ""), options=["about 9.0 meters", "b", "c", "d"]))
    kept, dropped = GEN._balance_windows(rs, 1.15, rng)
    by = collections.defaultdict(collections.Counter)
    for r in kept:
        by[r["options"][0]][r["answer"]] += 1
    assert "about 9.0 meters" not in by                       # a window missing letters is left out
    for c in by.values():
        assert len(c) == 4 and max(c.values()) <= 1.15 * min(c.values()) + 1


def test_answers_balanced_within_stem_cells():
    rng = random.Random(0)
    rs = []
    for i in range(1200):
        action = ["move_left", "move_right", "turn_left"][i % 3]
        yes = (i % 10 < 8) if action == "move_left" else (i % 10 < 2) if action == "move_right" else (i % 2 == 0)
        r = _rec(i, "goal_after", "Yes" if yes else "No", "yes" if yes else "no", changed=True, flip=None)
        r.update(action=action, k=1)
        rs.append(r)
    rs.append(dict(_rec(5000, "goal_after", "Yes", "yes", flip=None), action="move_forward", k=3))  # a cell with one answer
    kept = GEN.balance_split(rs, GEN.BUILD_CONFIG, rng, {})
    by = collections.defaultdict(collections.Counter)
    for r in kept:
        by[GEN.stem_cell(r)][r["answer"]] += 1
    assert ("projective_relations", "move_forward", 3) not in by          # it would give its answer away
    for c in by.values():
        assert len(c) == 2 and max(c.values()) <= 1.15 * min(c.values()) + 1


def test_occluder_check_spreads_over_types_answers_and_tasks():
    from data_gen.transition_qa import review as RV

    assert RV.occluder_group("wardrobe") == "wardrobe" and RV.occluder_group("easel") == "thin"
    assert RV.occluder_group("sofa") == ""
    rs = []
    for i in range(60):
        r = _rec(i, "occluded_now" if i % 3 else "occluded_after", "A", ["full", "partial"][i % 2])
        r["task_id"] = f"t{i % 30}"
        r["label"] = ["easel", "monitor", "telescope"][i % 3]
        rs.append(r)
    picks = RV.spread_pick(rs, 10, random.Random(0), lambda r: r["label"])
    assert len(picks) == 10 and len({r["sample_id"] for r in picks}) == 10
    cells = collections.Counter((r["qtype"], r["answer_value"]) for r in picks)
    assert len(cells) == 4 and max(cells.values()) - min(cells.values()) <= 1     # both types, both answers, in turn
    assert len({r["task_id"] for r in picks}) == 10                                # new tasks while there are some


def test_review_page_as_an_artifact():
    from data_gen.transition_qa import review as RV

    page = RV.artifact_page(RV.TEMPLATE.read_text(encoding="utf-8"))
    assert not page.lower().startswith("<!doctype") and "<body>" not in page and "</html>" not in page
    assert page.startswith("<title>") and '<meta name="claude:viewport-fit-sides" content="cover">' in page[:200]
    assert 'META.store !== "db"' in page and ":root[data-theme=\"dark\"]" in page
    # the probe check: extra pictures, the right answer of a wrong verdict, its own instructions
    assert "it.extra_images" in page and "it.fix_choices" in page and "fix: v.fix || null" in page
    assert "META.intro" in page and page.count('class="default-intro"') == 2


def test_probe_label_overrides_relabel_and_drop():
    def occ(i, opts, answer, value):
        r = _rec(i, "occluded_now", answer, value)
        r.update(kind="occluded", options=[f"{o} behind the bed" for o in opts])
        return r

    recs = [occ(1, ["partly hidden", "not hidden", "fully hidden"], "A", "partial"),
            occ(2, ["fully hidden", "partly hidden", "not hidden"], "A", "full"),
            occ(3, ["not hidden", "fully hidden", "partly hidden"], "B", "full"),
            _rec(4, "goal_now", "Yes", "")]
    ov = {"relabel": {"id000001": "none", "id000003": "partial"}, "drop": ["id000002"], "_path": "x.json"}
    rep = {}
    out = GEN.apply_label_overrides(recs, ov, rep)
    assert [r["sample_id"] for r in out] == ["id000001", "id000003", "id000004"]
    assert (out[0]["answer"], out[0]["answer_value"]) == ("B", "none")       # the option "not hidden ..." is B
    assert (out[1]["answer"], out[1]["answer_value"]) == ("C", "partial")
    assert out[1]["options"] == recs[2]["options"] and recs[0]["answer"] == "A"  # options stay; the input is untouched
    assert rep["relabelled"] == 2 and rep["dropped"] == 1 and rep["moves"] == {"partial->none": 1, "full->partial": 1}
    assert GEN.apply_label_overrides(recs, None, {}) is recs
    with pytest.raises(SystemExit):
        GEN.apply_label_overrides(recs, {"drop": ["id999999"]}, {})            # an id not in the probe
    with pytest.raises(SystemExit):
        GEN.apply_label_overrides(recs, {"relabel": {"id000004": "none"}}, {})  # a goal question takes yes / no
    recs[3]["kind"], recs[3]["answer_value"] = "goal", "yes"
    rep = {}
    out = GEN.apply_label_overrides(recs, {"relabel": {"id000004": "no"}, "checks": [{"date": "2026-10-28"}]}, rep)
    assert (out[3]["answer"], out[3]["answer_value"]) == ("No", "no")
    assert rep["checks"] == [{"date": "2026-10-28"}] and rep["moves"] == {"yes->no": 1}  # each check goes to the report


def test_scene_exclusion_drops_whole_scenes_and_keeps_the_rest(tmp_path):
    rows = [_rec(i, "closer_now", "A", "", scene=s) for i, s in enumerate(["s1", "s2", "s1", "s3"])]
    kept, gone = GEN.drop_scenes(rows, {"s1"})
    assert [r["sample_id"] for r in kept] == ["id000001", "id000003"] and len(gone) == 2  # order kept
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"excluded": ["s1", "s3"]}))
    assert GEN.read_scene_exclusions({"exclude_scenes": str(p)}) == (str(p), {"s1", "s3"})
    assert GEN.read_scene_exclusions({}) == (None, set())
    name = GEN.BUILD_CONFIG["exclude_scenes"]
    assert name == GEN.PROBE_CONFIG["exclude_scenes"] and (GEN.REPO / name).exists()  # the tracked list
    path, scenes = GEN.read_scene_exclusions(GEN.BUILD_CONFIG)
    assert "0204_840158" in scenes  # the science museum (user, 2026-10-28)


def test_hold_out_sets_question_types_and_family_goals_aside():
    rows = [_rec(i, q, "A", "") for i, q in enumerate(["occluded_now", "goal_now", "occluded_after", "closer_now", "goal_after"])]
    rows[1]["family"] = rows[2]["family"] = "occlusion_alignment"             # an occlusion task's goal question
    rules = GEN.BUILD_CONFIG["hold"]["train"]
    kept, held = GEN.hold_out(rows, rules)
    assert [r["qtype"] for r in kept] == ["closer_now", "goal_after"]       # goal_after of another family stays
    assert [r["qtype"] for r in held] == ["occluded_now", "goal_now", "occluded_after"]
    kept, held = GEN.hold_out(rows, GEN.BUILD_CONFIG["hold"]["test"])
    assert [r["qtype"] for r in held] == ["goal_now"] and len(kept) == 4
    assert GEN.PROBE_CONFIG["label_overrides"] == "probe_label_overrides.json"


def test_probe_check_puts_mismatched_occluders_first():
    from data_gen.transition_qa import review as RV

    qs = [{"sample_id": f"s{i:02d}"} for i in range(10)]
    status = {f"s{i:02d}": ("ok" if i % 3 else "see_through") for i in range(10)}  # s00, s03, s06, s09 flagged
    order = RV.order_for_check(qs, status, seed=1)
    assert [q["sample_id"] for q in order[:4]] != [] and {q["sample_id"] for q in order[:4]} == {"s00", "s03", "s06", "s09"}
    assert sorted(q["sample_id"] for q in order) == [q["sample_id"] for q in qs]
    assert order == RV.order_for_check(list(reversed(qs)), status, seed=1)         # the order does not depend on the input's
    assert RV.FIX_CHOICES[-1] == "不该出这道题" and RV.STATUS_ZH["see_through"] == "框里大半是空的"


def test_depth_checked_occlusion_tells_the_occluder_from_a_wall():
    from data_gen.interiorgs_v10.depth_check import depth_hidden
    from data_gen.transition_qa import occ_depth as OD

    c2w = np.eye(4)                                   # camera at the origin: right x, down y, forward z
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    depth = np.full((100, 100), 2.0)                  # a surface 2 m ahead everywhere
    alpha = np.ones((100, 100))
    pts = np.array([[0.0, 0.0, 3.0], [0.3, 0.0, 3.0], [0.0, 0.0, 1.0]])   # two behind the surface, one in front
    hit = depth_hidden(pts, c2w, K, depth, alpha)
    assert hit.tolist() == [True, True, False]
    surf = OD.surface_points(pts, c2w, K, depth)
    assert np.allclose(surf[1], [0.2, 0.0, 2.0])      # on the ray to the point, at the rendered depth
    occluder = {"bbox_min": [-0.5, -0.5, 1.9], "bbox_max": [0.5, 0.5, 2.1]}      # the surface belongs to B
    elsewhere = {"bbox_min": [3.0, 3.0, 1.9], "bbox_max": [4.0, 4.0, 2.1]}       # B is somewhere else: a wall hides
    w, framed = np.ones(3), np.ones(3, bool)
    box_hit = np.array([True, True, True])
    by_b = OD.fractions(w, framed, box_hit, hit, OD.inside_box(surf, occluder))
    by_wall = OD.fractions(w, framed, box_hit, hit, OD.inside_box(surf, elsewhere))
    assert by_b["depth_b"] == pytest.approx(2 / 3) and by_b["box"] == 1.0
    assert by_wall["depth_any"] == pytest.approx(2 / 3) and by_wall["depth_b"] == 0.0
    assert OD.answer_of(0.9) == "full" and OD.answer_of(0.0) == "none" and OD.answer_of(0.8) == "between"


def test_occlusion_labels_checked_against_the_rendered_depth(tmp_path):
    def occ(i, value, qtype="occluded_now"):
        r = _rec(i, qtype, "A", value)
        r["kind"] = "occluded"
        return r

    recs = [occ(1, "full"), occ(2, "full"), occ(3, "partial"), occ(4, "partial"), occ(5, "none"), occ(6, "full"),
            _rec(7, "goal_now", "Yes", "yes")]
    (tmp_path / "d").mkdir()
    GEN.write_jsonl(tmp_path / "d" / "r.jsonl", [
        {"sample_id": "id000001", "depth_any": 0.95}, {"sample_id": "id000002", "depth_any": 0.40},   # full: kept / dropped
        {"sample_id": "id000003", "depth_any": 0.12}, {"sample_id": "id000004", "depth_any": 0.02},   # partial: kept / dropped
        {"sample_id": "id000006", "error": "RuntimeError: x"}])                                        # no depth: dropped
    cfg = dict(GEN.BUILD_CONFIG, occlusion_depth={"results": "d/r.jsonl", "full_min": 0.70, "partial_min": 0.10})
    rep = {}
    kept = GEN.depth_check_occlusion(recs, tmp_path, cfg, rep)
    assert sorted(r["sample_id"] for r in kept) == ["id000001", "id000003", "id000005", "id000007"]
    assert rep["by_label"]["occluded_now/full"] == {"checked": 3, "dropped": 1, "no_depth": 1}
    assert GEN.depth_check_occlusion([recs[-1]], tmp_path / "nowhere", cfg, {}) == [recs[-1]]   # nothing to check
    with pytest.raises(SystemExit):
        GEN.depth_check_occlusion(recs, tmp_path / "nowhere", cfg, {})


def test_window_quota_is_the_optimum():
    # bins 1..6 (0.5 m .. 3.0 m): windows 1, 2, 3; bin 1 only fits window 1, bin 6 only window 3
    n = {1: 10, 2: 30, 3: 40, 4: 40, 5: 30, 6: 10}
    q = GEN.window_quota(n, 1, 6)
    for b, cnt in n.items():
        assert sum(t for s, t in q.items() if s <= b <= s + 3) <= cnt
    assert sum(q.values()) == 40                              # 10 + 20 + 10: bins 3 and 4 are in every window


def test_assign_windows_balances_and_keeps_more_than_trimming():
    rng = random.Random(0)
    rs = []
    counts = {1: 60, 2: 200, 3: 300, 4: 320, 5: 260, 6: 160, 7: 80, 8: 40}   # bins of 0.5 m, many near the smallest option
    i = 0
    for b, c in counts.items():
        for _ in range(c):
            value = b * 0.5
            vals, pos = Q.grid_options(value, 0.5, 0.5, float("inf"), f"k{i}")
            r = _rec(i, "distance_bin_now", Q.LETTERS[pos], f"{value:.1f}")
            r.update(kind="distance_bin", options=[Q.grid_text("distance_bin", v) for v in vals],
                     question="How far?\n" + Q.options_block([Q.grid_text("distance_bin", v) for v in vals]) + "\n" + Q.ANSWER_LETTER)
            rs.append(r)
            i += 1
    trimmed, _ = GEN._balance_windows([dict(r) for r in rs], 1.15, random.Random(0))
    kept = GEN.balance_split(rs, GEN.BUILD_CONFIG, rng, {})
    assert len(kept) > 1.3 * len(trimmed)
    by = collections.defaultdict(collections.Counter)
    for r in kept:
        assert r["options"][Q.LETTERS.index(r["answer"])] == Q.grid_text("distance_bin", float(r["answer_value"]))
        assert r["question"].endswith(Q.options_block(r["options"]) + "\n" + Q.ANSWER_LETTER)
        by[r["options"][0]][r["answer"]] += 1
    for c in by.values():
        assert len(c) == 4 and max(c.values()) <= 1.15 * min(c.values()) + 1


def test_probe_takes_whole_types_and_shuffles_categorical_options():
    rng = random.Random(0)
    rs = []
    for i in range(90):
        r = _rec(i, "occluded_now", "A", ["full", "partial", "none"][i % 3], split="val")
        r["options"] = ["fully hidden behind the bed", "partly hidden behind the bed", "not hidden behind the bed"]
        r["answer"] = "ABC"[i % 3]
        r["kind"] = "occluded"
        rs.append(r)
    rep = {}
    picks = GEN.probe_select(rs, dict(GEN.PROBE_CONFIG, per_qtype_overrides={"occluded_now": "all"}), rng, rep)
    assert len(picks) == 90 and rep["occluded_now"]["short"] == 0
    shuffled = [GEN.shuffle_options(r) for r in rs]
    assert len({tuple(r["options"]) for r in shuffled}) > 1   # the option order is drawn per question


def test_text_only_rules_and_section(tmp_path):
    from data_gen.transition_qa import report as R
    from data_gen.transition_qa import shortcuts as SC

    def goal(i, split, answer, action):
        r = _rec(i, "goal_after", answer, answer.lower(), split=split)
        r.update(options=[], action=action, k=1)
        return r

    # in training, turning left always gives Yes and turning right No: the rule reads that off the stem
    train = [goal(i, "train", "Yes" if i % 2 else "No", "turn_left" if i % 2 else "turn_right") for i in range(40)]
    test = [goal(100 + i, "val", "Yes", "turn_left") for i in range(10)] + \
           [goal(200 + i, "val", "Yes", "move_forward") for i in range(10)]          # never seen: a guess
    for name, rows in (("p1_train.jsonl", train), ("p1_test.jsonl", test)):
        GEN.write_jsonl(tmp_path / name, rows)
    got = {(r["file"], r["rule"]): r["accuracy"] for r in SC.report(tmp_path)}
    assert got[("p1_test.jsonl", "learned_action")] == pytest.approx((10 * 1.0 + 10 * 0.5) / 20)
    (tmp_path / "blind" / "test").mkdir(parents=True)
    (tmp_path / "blind" / "test" / "base_7b_blind.json").write_text(json.dumps(
        {"all": {"accuracy": 0.55, "unreadable": 0.0}, "qtype": {"goal_after": {"accuracy": 0.55, "unreadable": 0.0}}}))
    text = "\n".join(R.text_only_section(tmp_path, {"p1_test.jsonl": test}))
    assert "base_7b_blind 不看图" in text and "55.0%" in text and "75.0%（learned_action）" in text


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
