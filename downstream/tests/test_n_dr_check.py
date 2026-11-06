"""Tier g's check (n_dr_check.py): the subset and the comparison with recorded episodes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n_dr_check as D  # noqa: E402


def ep(iid, acts, success=False, blocked=None, poses=None, reply=None):
    turns = []
    for k, a in enumerate(acts):
        turns.append({"executed": a, "actions": a, "reply": reply or "|".join(a), "blocked": (blocked or {}).get(k, []),
                      "pose_before": [0.0, 0.0, 0.0] if k == 0 else [float(k), 0.0, 0.0],
                      "pose_after": (poses or {}).get(k, [float(k + 1), 0.0, 0.0])})
    return {"instance_id": iid, "success": success, "turn_log": turns,
            "final_pose": turns[-1]["pose_after"] if turns else [0.0, 0.0, 0.0]}


def test_pick_takes_each_set_and_type_once_per_scene():
    insts = [{"instance_id": f"{s}{t}{k}", "task_type": t, "scene_id": f"{s}scene{k}"}
             for s in "xy" for t in "pq" for k in range(3)]
    set1 = {i["instance_id"] for i in insts if i["instance_id"].startswith("x")}
    sub = D.pick(insts, set1, 1, seed=0)
    keys = [(i["instance_id"][0], i["task_type"]) for i in sub]
    assert sorted(keys) == [("x", "p"), ("x", "q"), ("y", "p"), ("y", "q")]
    assert len({i["scene_id"] for i in sub}) == 4 and sub == [i for i in insts if i in sub]  # file order
    assert D.pick(insts, set1, 1, seed=0) == sub  # fixed seed
    other = D.pick(insts, set1, 1, seed=0, avoid_scenes={i["scene_id"] for i in sub})
    assert not {i["scene_id"] for i in other} & {i["scene_id"] for i in sub}


def test_compare_looks_at_blocking_and_poses_and_needs_every_instance():
    rec = {"a": ep("a", [["move_left"], ["turn_left"]], True), "b": ep("b", [["move_left"]]),
           "c": ep("c", [["move_left"], ["move_left"]]), "d": ep("d", [["turn_right"]]),
           "e": ep("e", [["move_forward"]]), "f": ep("f", [["move_left"]])}
    rerun = {"a": ep("a", [["move_left"], ["turn_left"]], True),         # the same
             "b": ep("b", [["move_left"], ["move_backward"]]),           # one turn longer
             "c": ep("c", [["move_left"], ["turn_left"]], True),         # turn 2 and success differ
             "d": ep("d", [["move_left"]], True),                         # different, but its frames are broken
             "e": ep("e", [["move_forward"]], blocked={0: ["move_forward"]}, poses={0: [0.0, 0.0, 0.0]}),  # blocked now
             "f": ep("f", [["move_left"]], reply="<action>move_left|</action>")}  # same moves, other wording
    sub = ["a", "b", "c", "d", "e", "f"]
    r = D.compare_model(sub, rerun, rec, bad={"d"}, hinted={"f"})
    assert (r["expected"], r["compared"], r["clean"], r["frames_bad"], r["frames_hinted"]) == (6, 6, 5, 1, 1)
    assert r["success_flips"] == 1 and r["actions_diverged"] == 3 and not r["pass"]  # b, c, e: 3 > 2
    firsts = {x["instance_id"]: x["first_different_turn"] for x in r["different"]}
    assert firsts == {"b": 2, "c": 2, "d": 1, "e": 1}
    assert r["replies_identical"] == 2  # a and e word every reply alike
    sub5 = ["a", "b", "c", "d", "f"]
    r = D.compare_model(sub5, rerun, rec, bad={"d"})
    assert r["pass"] and r["actions_diverged"] == 2
    # a rerun that is missing an instance, or whose episodes are errors (a dead server), never passes
    del rerun["e"]
    r = D.compare_model(sub, rerun, rec)
    assert not r["pass"] and r["absent"] == ["e"]
    dead = {i: {"instance_id": i, "error": "no record: its process ended with code 1"} for i in sub}
    r = D.compare_model(sub, dead, rec)
    assert not r["pass"] and len(r["errors"]) == 6 and r["compared"] == 0


def test_frame_flags_and_the_command(tmp_path):
    scan = {"evals": {"x": {"instances": {"a": 2}, "broken": {"b": ["near_black"]}, "hints": {"c": ["flip"], "a": ["x"]}}}}
    assert D.frame_flags(scan) == ({"a", "b"}, {"c"})
    rows = [ep("a", [["move_left"]]), ep("b", [["turn_left"]])]
    for m in ("S", "RS"):
        (tmp_path / "rec" / m).mkdir(parents=True)
        (tmp_path / "rec" / m / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "dr" / "S").mkdir(parents=True)
    (tmp_path / "dr" / "S" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "sub.jsonl").write_text('{"instance_id": "a"}\n{"instance_id": "b"}\n')
    args = ["compare", "--instances", str(tmp_path / "sub.jsonl"), "--rerun", str(tmp_path / "dr"),
            "--recorded", str(tmp_path / "rec"), "--out", str(tmp_path / "out"), "--models"]
    assert D.main(args + ["S"]) == 0
    assert D.main(args + ["S", "RS"]) == 1  # RS has no rerun at all
    rep = json.loads((tmp_path / "out/dr_check.json").read_text())
    assert rep["models"]["S"]["pass"] and rep["models"]["RS"]["absent"] == ["a", "b"] and not rep["pass"]
    md = (tmp_path / "out/dr_check.md").read_text(encoding="utf-8")
    assert "| S | 2 | 2 | 2 | 0 | 0 | 0 / 0 | 0 | 0 | 2 | 是 |" in md and "检查不通过" in md


def test_the_check_fails_on_too_many_flips_over_all_models(tmp_path):
    rec = [ep(i, [["move_left"]]) for i in "abc"]
    flipped_a = ep("a", [["move_left"]], True)
    for m in ("S", "RS", "RD"):
        rerun = ([flipped_a] + rec[1:]) if m != "RD" else rec
        for root, rows in (("rec", rec), ("dr", rerun)):
            (tmp_path / root / m).mkdir(parents=True, exist_ok=True)
            (tmp_path / root / m / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "sub.jsonl").write_text("".join(json.dumps({"instance_id": i}) + "\n" for i in "abc"))
    args = ["compare", "--instances", str(tmp_path / "sub.jsonl"), "--rerun", str(tmp_path / "dr"),
            "--recorded", str(tmp_path / "rec"), "--out", str(tmp_path / "out"), "--models", "S", "RS", "RD"]
    assert D.main(args) == 0  # one flip each in S and RS: 2 in all
    (tmp_path / "dr/RD/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [flipped_a] + rec[1:]))
    assert D.main(args) == 1  # 3 in all, though each model has only one
    assert json.loads((tmp_path / "out/dr_check.json").read_text())["total_success_flips"] == 3
