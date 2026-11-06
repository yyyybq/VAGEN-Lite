"""Tier g's check (n_dr_check.py): the subset and the comparison with recorded episodes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n_dr_check as D  # noqa: E402


def ep(iid, acts, success=False, replies=None):
    return {"instance_id": iid, "success": success,
            "turn_log": [{"executed": a, "actions": a, "reply": (replies or {}).get(k, "|".join(a))}
                         for k, a in enumerate(acts)]}


def test_pick_takes_each_set_and_type_in_file_order():
    insts = [{"instance_id": f"{s}{t}{k}", "task_type": t} for s in "xy" for t in "pq" for k in range(3)]
    sub = D.pick(insts, {i["instance_id"] for i in insts if i["instance_id"].startswith("x")}, 2)
    assert [i["instance_id"] for i in sub] == ["xp0", "xp1", "xq0", "xq1", "yp0", "yp1", "yq0", "yq1"]


def test_compare_counts_flips_and_diverged_actions_on_clean_episodes_only():
    rec = {"a": ep("a", [["move_left"], ["turn_left"]], True), "b": ep("b", [["move_left"]]),
           "c": ep("c", [["move_left"], ["move_left"]]), "d": ep("d", [["turn_right"]]), "e": ep("e", [["move_left"]])}
    rerun = [ep("a", [["move_left"], ["turn_left"]], True),          # the same
             ep("b", [["move_left"], ["move_back"]]),                  # one turn longer: diverged at turn 2
             ep("c", [["move_left"], ["turn_left"]], True),           # diverged at turn 2 and succeeded
             ep("d", [["move_left"]], True),                           # different, but its frames were flagged
             ep("e", [["move_left"]], replies={0: "<action>move_left|</action>"}),  # same actions, other text
             ep("z", [["move_left"]])]                                 # no recorded episode
    r = D.compare_model(rerun, rec, flagged={"d"})
    assert (r["episodes"], r["clean"], r["frame_flagged"], r["missing"]) == (6, 4, 1, 1)
    assert r["success_flips"] == 1 and r["actions_diverged"] == 2 and r["pass"]
    assert r["replies_identical"] == 1  # only a: b, c, d differ in actions, e in wording
    firsts = {x["instance_id"]: (x["first_different_turn"], x["frame_flag"]) for x in r["different"]}
    assert firsts == {"b": (2, False), "c": (2, False), "d": (1, True)}
    rerun.append(ep("f", [["turn_left"]], True))
    rec["f"] = ep("f", [["move_left"]])
    assert not D.compare_model(rerun, rec, flagged={"d"})["pass"]  # 2 flips, 3 diverged


def test_frame_flags_and_the_command(tmp_path):
    scan = {"evals": {"x": {"instances": {"a": 2}, "broken": {"b": ["near_black"]}, "hints": {"c": ["flip"]}}}}
    assert D.frame_flagged(scan) == {"a", "b", "c"}
    for m, rows in (("S", [ep("a", [["move_left"]])]),):
        (tmp_path / "rec" / m).mkdir(parents=True)
        (tmp_path / "rec" / m / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (tmp_path / "dr" / m).mkdir(parents=True)
        (tmp_path / "dr" / m / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc = D.main(["compare", "--rerun", str(tmp_path / "dr"), "--recorded", str(tmp_path / "rec"),
                 "--models", "S", "RS", "--out", str(tmp_path / "out")])
    rep = json.loads((tmp_path / "out/dr_check.json").read_text())
    assert rc == 1 and rep["models"]["S"]["pass"] and not rep["models"]["RS"]["pass"] and not rep["pass"]
    assert "| S | 1 | 1 | 0 | 0 | 0 | 0 | 是 |" in (tmp_path / "out/dr_check.md").read_text(encoding="utf-8")
