"""D3 results (downstream/n/d3_report.py): per-model rows (strict, relaxed, wrong target, original), any-turn wrong
target with a box judge, both-solvable variants, the versions, and a whole run on synthetic episodes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
sys.path.insert(0, str(ROOT))
import d3_report as D  # noqa: E402


def judge_x_ge_1(item, pose):  # a fake box judge: success when x >= item["x"]
    return {"success": pose[0] >= item["x"]}


def variant(oid, tt="absolute_positioning", near=(), scene="S_s1", vlen=5):
    return {"instance_id": oid + "_d3a", "task_type": tt, "scene_id": scene, "near_threshold": list(near),
            "shortest": {"pose": [0.5, 0, 0], "path": ["a"] * vlen}, "best": {"pose": [0.6, 0, 0]},
            "d3": {"original": oid, "original_item": {"x": 2.0}}}


def ep(iid, success, shaping=0.0, orig_end=False, poses=((0, 0, 0), (1, 0, 0))):
    log = [{"pose_before": list(poses[0]), "pose_after": list(p)} for p in poses[1:]]
    return {"instance_id": iid, "success": success, "max_shaping_score": shaping, "turn_log": log,
            "final_judge_original": {"success": orig_end}}


def test_per_model_rows_and_any_turn():
    var = {"a_d3a": ep("a_d3a", True, 0.9), "b_d3a": ep("b_d3a", False, 0.6, orig_end=True),
           "c_d3a": ep("c_d3a", False, 0.1, poses=((0, 0, 0), (2.5, 0, 0), (0, 0, 0)))}
    orig = {"a": {"success": True}, "b": {"success": False}, "c": {"success": True}}
    items = {k: {"x": 2.0} for k in "abc"}
    rows = D.per_model(var, orig, items, judge_x_ge_1)
    assert rows["a"] == {"strict": True, "relaxed": True, "wrong_target": False, "original": True,
                         "any_turn_original": False}
    assert rows["b"]["wrong_target"] and rows["b"]["relaxed"] and not rows["b"]["any_turn_original"]
    assert rows["c"]["any_turn_original"] and not rows["c"]["wrong_target"]  # passed the original mid-way, left it
    assert D.per_model(var, {}, items) == {}  # no original episode, no row


def test_both_solvable_and_versions():
    vs = [variant("a"), variant("b", near=["visible"]), variant("c", tt="projective_relations", scene="WashLettuce_s1000")]
    vs[1]["best"]["pose"] = [2.2, 0, 0]  # b's own success pose passes the original's judge
    both = D.both_solvable(vs, judge_x_ge_1)
    assert both == {"b"}
    originals = {"a": {"shortest": {"path": ["x"] * 4}}, "b": {"shortest": {"path": ["x"] * 9}},
                 "c": {"shortest": {"path": ["x"] * 5}}}
    v = D.versions(vs, originals, {"pertable": {"a"}}, both)
    assert v["all"] == ["a", "b", "c"] and v["no_left_right"] == ["a", "b"]
    assert v["minus_pertable"] == ["b", "c"] and v["no_near_threshold"] == ["a", "c"]
    assert v["no_both_solvable"] == ["a", "c"] and v["no_suspect_scenes"] == ["a", "b"]
    assert v["not_longer"] == ["b", "c"]  # a's variant (5) is longer than its original (4)
    assert v["longer"] == ["a"]


def test_run_end_to_end(tmp_path):
    vs = [variant(f"i{k}", scene=f"S_s{k % 5}") for k in range(20)]
    (tmp_path / "v.jsonl").write_text("".join(json.dumps(v) + "\n" for v in vs))
    (tmp_path / "o.jsonl").write_text("".join(json.dumps({"instance_id": f"i{k}", "shortest": {"path": ["x"] * 6}}) + "\n"
                                              for k in range(20)))
    for m, rate in (("B", 0), ("S", 6), ("RS", 8), ("RD", 10)):
        for folder, shift in (("eval", 0), ("orig", 3)):
            d = tmp_path / folder / m
            d.mkdir(parents=True)
            rows = [ep(f"i{k}_d3a" if folder == "eval" else f"i{k}", k < rate + (shift if m != "B" else 0))
                    for k in range(20)]
            if folder == "orig":
                for r in rows:
                    r["instance_id"] = r["instance_id"]
            (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "ex.txt").write_text("i0\ni1\n")
    res = D.run(D.argparse.Namespace(variants=tmp_path / "v.jsonl", eval=tmp_path / "eval", orig=tmp_path / "orig",
                                     originals=[tmp_path / "o.jsonl"], models=["B", "S", "RS", "RD"],
                                     exclude=[str(tmp_path / "ex.txt")], partial=None, no_judge=True,
                                     out=tmp_path / "out"))
    a = res["versions"]["all"]["per_model"]
    assert (a["S"]["n"], a["S"]["strict"], a["S"]["original"]) == (20, 6, 9)
    assert a["S"]["variant_minus_original"]["only_b"] == 3
    assert res["versions"]["minus_ex"]["per_model"]["S"]["n"] == 18
    between = {(p["a"], p["b"]): p for p in res["versions"]["all"]["between"]}
    assert set(between) == {("S", "B"), ("RS", "B"), ("RD", "B"), ("RS", "S"), ("RD", "S")}
    assert between[("RD", "S")]["only_a"] == 4 and between[("RD", "S")]["holm"] >= between[("RD", "S")]["p"]
    assert (tmp_path / "out" / "report.md").read_text().startswith("## all")
