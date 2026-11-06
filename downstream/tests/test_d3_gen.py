"""D3 (a) axis (downstream/n/d3_gen.py; the user's task book of 2026-11-06): the three swap rules, the scene check,
dropping candidates the original's success poses already pass, the variant from n_gen.find with an explicit candidate
list, and collect's counts."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import d3_gen as D  # noqa: E402
import n_env as N  # noqa: E402
import n_gen as G  # noqa: E402
import n_smoke as S  # noqa: E402


def box(key, name, c):
    lo = [c[0] - 0.1, c[1] - 0.1, c[2] - 0.1]
    hi = [c[0] + 0.1, c[1] + 0.1, c[2] + 0.1]
    return {"key": key, "name": name, "kind": "object", "bbox_min": lo, "bbox_max": hi, "center": list(c)}


class FakeScene:
    """Free floor for x < 0.95; a wall beyond."""

    def __init__(self):
        self.start, self.pose, self.task, self.seed, self.hide_robot = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "T", 1, True
        self.entities = [box("a", "mug", (2.0, 1.0, 1.0)), box("b", "kettle", (2.0, -1.0, 1.0)),
                         box("c", "cup", (2.0, 0.0, 1.0))]

    def set_pose(self, pose):
        self.pose = pose

    def collision(self):
        return "wall" if self.pose[0] >= 0.95 else None

    def judge(self, item, pose, render=True):
        j = dict(N.box_judge(item, pose))
        j["visible"] = {}
        return j


def target(item):
    return item["target_object"]["objects"][0]["id"]


def fake_box(item, pose):
    """a and c: success forward of 0.5 m; b: success behind -0.5 m; left / right: 'left' forward, 'right' behind."""
    t = item["task_type"]
    if t == "projective_relations":
        ok = pose[0] > 0.5 if item["relation"] == "left" else pose[0] < -0.5
    else:
        ok = pose[0] < -0.5 if target(item) == "b" else pose[0] > 0.5
    ok = ok and abs(pose[2]) < 1e-6 and abs(pose[1]) < 1e-6
    return {"success": ok, "metric": {"gates": {"g": ok}}, "lateral": {"applies": False, "ok": True}}


@pytest.fixture
def fakes(monkeypatch):
    def make_item(t, sid, ents, params, start):
        if params.get("distance_m") == 9:
            raise ValueError("tuple rule: too far")
        return {"task_type": t, "relation": params.get("relation"), "task_description": "",
                "target_object": {"objects": [{"id": e["key"]} for e in ents]}}
    monkeypatch.setattr(N, "make_item", make_item)
    monkeypatch.setattr(N, "box_judge", fake_box)
    monkeypatch.setattr(N, "required_keys", lambda item: [])
    monkeypatch.setattr(N, "margins", lambda item, j: {})
    monkeypatch.setattr(N, "near_threshold", lambda item, j: [])
    monkeypatch.setattr(N, "measurements", lambda item, j: {})


def original(sc, t, objs, params, shortest=(0.6, 0.0, 0.0), best=(0.9, 0.0, 0.0)):
    ents = [next(e for e in sc.entities if e["key"] == k) for k in objs]
    item = N.make_item(t, "T_s1", ents, params, sc.start)
    return {"instance_id": f"T_s1_{t}", "scene_id": "T_s1", "task": "T", "seed": 1, "task_type": t,
            "objects": [S.strip(e) for e in ents], "params": params, "item": item, "start_pose": list(sc.start),
            "shortest": {"pose": list(shortest)}, "best": {"pose": list(best)}}


def test_rules_reverse_left_right_and_swap_the_single_target():
    sc = FakeScene()
    lr = {"task_type": "projective_relations", "objects": [S.strip(sc.entities[0]), S.strip(sc.entities[1])],
          "params": {"relation": "left"}}
    (ents, params), = D.candidates(lr, sc.entities, random.Random(0))
    assert [e["key"] for e in ents] == ["a", "b"] and params == {"relation": "right"}
    ab = {"task_type": "absolute_positioning", "objects": [S.strip(sc.entities[0])], "params": {"distance_m": 1.0}}
    c1 = D.candidates(ab, sc.entities, random.Random(5))
    assert sorted(e[0][0]["key"] for e in c1) == ["b", "c"] and all(p == {"distance_m": 1.0} for _, p in c1)
    assert [e[0][0]["key"] for e in c1] == [e[0][0]["key"] for e in D.candidates(ab, sc.entities, random.Random(5))]
    assert set(D.RULES) == {"projective_relations", "absolute_positioning", "screen_occupancy"}
    assert set(D.NOT_MADE) == {"equidistance", "fov_inclusion", "occlusion_alignment"}


def test_scene_check_catches_a_moved_start_or_object(fakes):
    sc = FakeScene()
    inst = original(sc, "absolute_positioning", ["a"], {"distance_m": 1.0})
    assert D.scene_check(sc, inst) is None
    moved = json.loads(json.dumps(inst))
    moved["start_pose"][0] = 0.01
    assert D.scene_check(sc, moved) == "scene_mismatch:start"
    moved = json.loads(json.dumps(inst))
    moved["objects"][0]["bbox_min"][1] += 0.01
    assert D.scene_check(sc, moved) == "scene_mismatch:box"
    moved["objects"][0]["key"] = "zz"
    assert D.scene_check(sc, moved) == "scene_mismatch:object"


def test_variant_skips_targets_the_original_success_already_satisfies(fakes):
    sc = FakeScene()
    states = S.reachable(sc, max_depth=4)["states"]
    order = sorted(states, key=lambda k: states[k][1])
    inst = original(sc, "absolute_positioning", ["a"], {"distance_m": 1.0})
    var, rec = D.variant_for(sc, states, order, inst, deadline=1e18)
    assert rec["tally"]["original_pose_passes"] == 1  # c is satisfied where a is
    assert var["instance_id"] == "T_s1_absolute_positioning_d3a" and [o["key"] for o in var["objects"]] == ["b"]
    assert var["params"] == {"distance_m": 1.0} and var["shortest"]["pose"][0] < -0.5
    assert var["d3"]["original"] == inst["instance_id"] and var["d3"]["original_item"] == inst["item"]
    assert rec["variant"] == var["instance_id"] and rec["objects"] == ["kettle"]


def test_left_right_variant_and_reasons_for_none(fakes):
    sc = FakeScene()
    states = S.reachable(sc, max_depth=4)["states"]
    order = sorted(states, key=lambda k: states[k][1])
    lr = original(sc, "projective_relations", ["a", "b"], {"relation": "left"})
    var, rec = D.variant_for(sc, states, order, lr, deadline=1e18)
    assert var["params"] == {"relation": "right"} and var["item"]["relation"] == "right"
    bad = original(sc, "absolute_positioning", ["a"], {"distance_m": 1.0}, shortest=(-0.3, 0.0, 0.0))
    assert D.variant_for(sc, states, order, bad, 1e18)[1]["reason"] == "original_fails"
    far = original(sc, "absolute_positioning", ["a"], {"distance_m": 1.0})
    far["params"] = {"distance_m": 9}
    var, rec = D.variant_for(sc, states, order, far, 1e18)
    assert var is None and rec["reason"] == "no_candidate" and rec["tally"]["refused:too far"] == 2


def test_find_tries_only_the_given_candidates(fakes):
    sc = FakeScene()
    states = S.reachable(sc, max_depth=4)["states"]
    order = sorted(states, key=lambda k: states[k][1])
    inst, tally = G.find(sc, states, order, "absolute_positioning", random.Random(0), 1e18,
                         cands=[([sc.entities[1]], {"distance_m": 1.0})])
    assert inst["objects"][0]["key"] == "b" and tally["instance"] == 1
    inst, tally = G.find(sc, states, order, "absolute_positioning", random.Random(0), 1e18, cands=[])
    assert inst is None and not tally


def test_collect_counts_variants_and_reasons(tmp_path):
    rows = [{"instance_id": f"T_s1_{t}", "scene_id": "T_s1", "task": "T", "seed": 1, "task_type": t}
            for t in ("projective_relations", "absolute_positioning", "screen_occupancy", "equidistance")]
    rows.append({"instance_id": "U_s2_screen_occupancy", "scene_id": "U_s2", "task": "U", "seed": 2,
                 "task_type": "screen_occupancy"})
    src = tmp_path / "inst.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    (out / "scenes").mkdir(parents=True)
    var = {"instance_id": "T_s1_projective_relations_d3a", "task_type": "projective_relations"}
    (out / "scenes" / "T_s1.json").write_text(json.dumps({
        "scene_id": "T_s1", "complete": True, "seconds": 50, "variants": {"T_s1_projective_relations": var},
        "records": [{"original": "T_s1_projective_relations", "variant": var["instance_id"]},
                    {"original": "T_s1_absolute_positioning", "reason": "no_solvable_candidate"},
                    {"original": "T_s1_screen_occupancy", "reason": "scene_mismatch:box"}]}))
    rep = D.collect([src], out)
    assert rep["originals"] == 4 and rep["variants"] == 1 and rep["scenes"] == 2 and rep["scenes_done"] == 1
    assert rep["by_type"]["screen_occupancy"] == {"originals": 2, "scene_mismatch:box": 1, "scene_not_run": 1}
    assert [json.loads(x)["instance_id"] for x in (out / "variants.jsonl").read_text().splitlines()] == \
        ["T_s1_projective_relations_d3a"]
