"""N series terminal metrics and the move-backward split (the user's task book of 2026-11-05, items 6 and 8;
downstream/n/n_terminal.py, downstream/n/n2_backward.py): rebuilding each step from the turn records, the
distributions and the pairing, and N2 split by the shortest path's first action (CPU, no simulator)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n2_backward as B  # noqa: E402
import n_env as N  # noqa: E402
import n_terminal as T  # noqa: E402


def turn(k, before, executed, blocked, after):
    return {"turn": k, "pose_before": list(before), "actions": executed, "executed": executed,
            "blocked": [[a, "blocked by wall"] for a in blocked], "pose_after": list(after)}


def score_x(pose):
    return round(pose[0], 9)


def test_each_step_is_rebuilt_and_the_score_before_the_first_block_kept():
    row = {"instance_id": "I", "success": False, "max_shaping_score": 0.6, "blocked": 1, "final_pose": [0.3, 0, 0],
           "turn_log": [turn(1, (0, 0, 0), ["move_left", "move_forward", "move_forward"], ["move_left"], (0.6, 0, 0)),
                        {"turn": 2, "pose_before": [0.6, 0, 0], "actions": [], "parse_error": "x"},
                        turn(3, (0.6, 0, 0), ["move_backward"], [], (0.3, 0, 0))]}
    r = T.rebuild_episode(row, score_x)
    assert r["ok"] and r["why"] is None
    assert r["rebuilt_max"] == pytest.approx(0.6) and r["before_first_block"] == 0.0 and r["first_block_step"] == 1
    assert r["steps"] == 4
    never = dict(row, turn_log=[turn(1, (0, 0, 0), ["move_forward", "move_forward"], [], (0.6, 0, 0))], blocked=0)
    r = T.rebuild_episode(never, score_x)
    assert r["ok"] and r["first_block_step"] is None and r["before_first_block"] == pytest.approx(0.6)


def test_the_same_action_from_the_same_pose_is_blocked_alike_and_ambiguity_is_refused():
    # fwd blocked at the start, back, then fwd passes from elsewhere: the only reading that keeps one blocked entry
    opts = T.turn_options((0, 0, 0), ["move_forward", "move_backward", "move_forward"], ["move_forward"])
    assert [o[0] for o in opts] == [[True, False, False]]
    row = {"instance_id": "I", "success": False, "max_shaping_score": 0.0,
           "turn_log": [turn(1, (0, 0, 0), ["move_left", "move_forward", "move_left"], ["move_left"],
                             N.apply_action(N.apply_action((0, 0, 0), "move_left"), "move_forward"))]}
    r = T.rebuild_episode(row, lambda p: 0.0)  # left then forward, or forward then left: the same end
    assert not r["ok"] and "ambiguous" in r["why"]
    bad = dict(row, max_shaping_score=0.5, turn_log=[turn(1, (0, 0, 0), ["move_forward"], [], (0.3, 0, 0))])
    r = T.rebuild_episode(bad, score_x)
    assert not r["ok"] and r["why"].startswith("rebuilt maximum")
    lost = dict(bad, turn_log=[turn(1, (0, 0, 0), ["move_forward"], [], (9, 9, 0))])
    assert "no assignment" in T.rebuild_episode(lost, score_x)["why"]


def test_rebuild_reads_an_evaluation_folder_with_the_box_judge(monkeypatch, tmp_path):
    calls = []

    def judge(item, pose):
        calls.append(tuple(pose))
        return {"metric": {"shaping_score": item["k"] * pose[0]}}

    monkeypatch.setattr(N, "box_judge", judge)
    insts = [{"instance_id": "I1", "scene_id": "T_s1", "item": {"k": 1.0}},
             {"instance_id": "I2", "scene_id": "T_s1", "item": {"k": 2.0}}]
    (tmp_path / "i.jsonl").write_text("".join(json.dumps(i) + "\n" for i in insts))
    (tmp_path / "ev/S").mkdir(parents=True)
    eps = [{"instance_id": "I1", "success": True, "max_shaping_score": 0.3,
            "turn_log": [turn(1, (0, 0, 0), ["move_forward"], [], (0.3, 0, 0))]},
           {"instance_id": "I2", "success": False, "max_shaping_score": 0.6,
            "turn_log": [turn(1, (0, 0, 0), ["move_forward", "move_backward", "move_forward"], [], (0.3, 0, 0))]},
           {"instance_id": "I9", "success": False, "max_shaping_score": 0.0, "turn_log": []},
           {"instance_id": "I1", "error": "x"}]
    (tmp_path / "ev/S/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    assert T.main(["rebuild", "--instances", str(tmp_path / "i.jsonl"), "--eval", str(tmp_path / "ev"),
                   "--models", "S", "RS", "--out", str(tmp_path / "rows.jsonl")]) == 0
    rows = [json.loads(x) for x in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert [(r["instance_id"], r["model"], r["ok"]) for r in rows] == [("I1", "S", True), ("I2", "S", True)]
    assert calls.count((0.3, 0.0, 0.0)) == 2  # once per instance: the poses are cached
    rep = T.build({"c": rows, "d": [dict(r, max_shaping=r["max_shaping"] - 0.1) for r in rows]}, [("c", "d")],
                  scenes={"I1": "T_s1", "I2": "T_s1"})
    s = rep["sets"]["c"]["S"]
    assert s["rebuilt"] == 2 and s["success"]["max_shaping"]["n"] == 1
    assert s["failure"]["max_shaping"]["at_least_half"] == 1 and s["failure"]["ever_blocked"] == 0
    p = rep["pairs"]["c-d"]["S"]
    assert p["n"] == 2 and p["diff"] == pytest.approx(0.1) and p["a_higher"] == 2 and p["p"] == pytest.approx(0.5)
    assert "## 配对：c-d" in T.markdown(rep)


def test_distribution_and_sign_test():
    d = T.dist([0.1, 0.2, 0.5, 0.9])
    assert d["median"] == pytest.approx(0.35) and d["q25"] == pytest.approx(0.175) and d["at_least_half"] == 2
    assert T.binom_two_sided(0, 5) == pytest.approx(2 / 32) and T.dist([])["n"] == 0


def ep(iid, success, first):
    return {"instance_id": iid, "success": success,
            "turn_log": [{"turn": 1, "actions": []}, {"turn": 2, "executed": [first, "turn_left"]}] if first else []}


def test_n2_split_by_the_shortest_paths_first_action(tmp_path):
    insts = {f"I{k}": {"shortest": {"path": ["move_backward" if k < 3 else "turn_left", "move_forward"]}}
             for k in range(6)}
    runs, n2 = tmp_path / "runs", tmp_path / "n2"
    rows = {"B": [ep(f"I{k}", k == 0, "move_forward") for k in range(6)],
            "S": [ep(f"I{k}", k in (0, 1, 3), "move_backward") for k in range(6)],
            "RS": [ep(f"I{k}", k in (0, 3, 4), "turn_left" if k else None) for k in range(6)]}
    for m, rs in rows.items():
        for d in (runs / f"n/n1_eval_20261031/{m}", n2 / f"{m}/eval/100"):
            d.mkdir(parents=True)
            (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    data = B.load(n2, runs, models=("B", "S", "RS"))
    assert sorted(data["S"]) == [0, 100]
    rep = B.build(data, insts, exclude={"I5"})
    assert rep["parts"] == {"backward": 3, "other": 2, "all": 5}
    assert rep["rates"]["backward"]["S"]["0"]["k"] == 2 and rep["rates"]["other"]["S"]["0"]["k"] == 1
    assert rep["vs_base"]["backward"]["S"]["100"]["only_a"] == 1  # S succeeds on I1 where B fails
    assert set(rep["vs_sft"]["all"]) == {"RS"} and "S" in rep["gap"]["vs_base"]
    fs = rep["first_step"]
    assert fs["S"]["0"]["backward"] == {"n": 3, "optimal": 3, "backward": 3, "optimal_rate": 1.0, "backward_rate": 1.0}
    assert fs["S"]["0"]["other"]["optimal"] == 0 and fs["RS"]["100"]["other"]["optimal"] == 2
    assert fs["RS"]["100"]["backward"]["optimal"] == 0  # I0 never acted
    md = B.markdown(rep)
    assert "## RL 减 SFT" in md and "第一步后退 3 个实例" in md


def test_n2_curve_label_names_the_instances_drawn():
    rep = {"rates": {"S": {"0": {"n": 593}, "100": {"n": 590}}, "B": {"0": {"n": 0}}}}
    assert B.N2R.ylabel(rep) == "Success on 593 instances (%)"
