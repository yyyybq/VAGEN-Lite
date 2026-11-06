"""reward.compare_perturbed on made-up results in a temporary folder: the pool leaves margin-0 tasks out, the main
verdict follows PREREG_PERTURB section 4, incomplete results give no verdict, the sign test, the set checks, the
split-matched original results, the replay check and the fail-end coverage."""

import json

import pytest

from reward import compare_perturbed as CP


def write_set(root, name, tasks, split=None):
    """tasks: [(task_id, d_orig, k)]"""
    folder = root / name
    folder.mkdir()
    rows = [{"task_id": t, "task_type": "f", "difficulty": {"d_star": d + k}, "split": split or name.split("_")[0],
             "perturbation": {"k": k, "d_star_original": d}} for t, d, k in tasks]
    (folder / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


def write_result(folder, model, solved, tasks, done=True):
    res = folder / model
    res.mkdir()
    rows = [{"task_id": t, "success": t in solved, "d_star": d + k, "invalid_turns": 0, "turns": 5}
            for t, d, k in tasks]
    (res / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if done:
        (res / "done.json").write_text("{}")


def make(tmp_path, solved_by, k4_tasks, done_for=None):
    k2 = [("a", 3, 2), ("b", 5, 2)]
    sets = {}
    for which in ("val", "test"):
        for k, tasks in ((2, k2), (4, k4_tasks)):
            ts = [(f"{which}{t}", d, kk) for t, d, kk in tasks]
            folder = write_set(tmp_path, f"{which}_k{k}", ts)
            sets[f"{which}_k{k}"] = folder
            for m in CP.MODELS:
                solved = {f"{which}{t}" for t in solved_by.get(m, set())}
                write_result(folder, m, solved, ts, done=(done_for is None or m in done_for))
    return sets


def test_sign_test_and_formats():
    assert CP.sign_test(0, 0) is None
    assert CP.sign_test(5, 0) == 0.0625
    assert CP.sign_test(3, 3) == 1.0
    assert CP.fmt_p(None) == "—" and CP.fmt_p(CP.sign_test(30, 0)) == "< 0.0001" and CP.fmt_p(0.0625) == "0.0625"
    # 4900 / 649 = 7.55008: shown from the unrounded value (+7.6), not from round(., 2) = 7.55 (+7.5)
    c = {"diff_raw": 4900 / 649, "diff_points": 7.55, "wins": 92, "losses": 43, "p_sign": 2.99e-5}
    assert CP.fmt_diff(c) == "+7.6（92/43，p < 0.0001）"
    assert CP.fmt_diff({"missing": ["x"]}) == "缺" and CP.fmt_diff(None) == "—"


def test_pool_and_verdict(tmp_path):
    # k = 4: tasks t1..t9 with d* 2 (margin 6), z with d* 8 (margin 0)
    k4 = [(f"t{i}", 2, 4) for i in range(1, 10)] + [("z", 8, 4)]
    solved = {"rw_ok_250": {"t1", "t2", "t3"}, "rw_s0_150": {"t1", "z"},
              "rw_s2ok_250": {"t1", "t2"}, "rw_s2s0_300": {"t1"}}
    res = CP.run(make(tmp_path, solved, k4), {}, None, None)
    pairs = res["main"]["pairs"]
    # pool = 9 tasks a set x 2 sets; z (margin 0) is out, so the S0 success on z does not count
    assert pairs["seed 20261019"]["n"] == 18
    assert pairs["seed 20261019"]["wins"] == 4 and pairs["seed 20261019"]["losses"] == 0
    assert abs(pairs["seed 20261019"]["diff_raw"] - 100 * 4 / 18) < 1e-9
    assert abs(pairs["seed 20261021"]["diff_raw"] - 100 * 2 / 18) < 1e-9
    assert res["main"]["verdict"] == "pass"  # mean (22.2 + 11.1) / 2 >= 5, both above 0
    band0 = res["main"]["reported_besides"]["seed 20261019"]["k4_by_band"]["0"]
    assert band0["n"] == 2 and band0["losses"] == 2
    assert res["main"]["reported_besides"]["seed 20261019"]["k4_test_pool"]["n"] == 9
    s = res["success"]["rw_s0_150"]["k4"]
    assert s["by_band"]["0"]["rate"] == 1.0 and s["by_band_val"]["0"]["n"] == 1 and s["by_band_test"]["5-"]["n"] == 9
    sec = {c["rl"]: c for c in res["secondary"]}
    assert sec["rw_ok_250"]["k4_test"]["n"] == 9 and sec["rw_ok_250"]["k2_test"]["n"] == 2


def test_rates_round_once(tmp_path):
    # 98 / 568 = 17.254%: stored unrounded, printed 17.3 (rounding to 4 places first gave 0.1725 -> "17.2")
    k4 = [(f"t{i}", 2, 4) for i in range(284)]
    sets = make(tmp_path, {"rw_s0_150": {f"t{i}" for i in range(49)}}, k4)
    res = CP.run(sets, {}, None, None)
    assert res["success"]["rw_s0_150"]["k4"]["pool"]["rate"] == 98 / 568
    assert "| rw_s0_150 | k4 | 17.3 | 17.3 | 17.3 | 17.3 |" in CP.markdown(res)


def test_one_seed_against(tmp_path):
    k4 = [(f"t{i}", 2, 4) for i in range(1, 11)]
    solved = {"rw_ok_250": {f"t{i}" for i in range(1, 6)}, "rw_s0_150": set(),
              "rw_s2ok_250": set(), "rw_s2s0_300": {"t1"}}
    res = CP.run(make(tmp_path, solved, k4), {}, None, None)
    assert res["main"]["mean_diff_points"] > 5 and res["main"]["verdict"] == "no pass"


def test_incomplete_gives_no_verdict(tmp_path):
    k4 = [("t1", 2, 4), ("t2", 3, 4)]
    done = set(CP.MODELS) - {"rw_s2s0_300"}
    res = CP.run(make(tmp_path, {"rw_ok_250": {"t1"}}, k4, done_for=done), {}, None, None)
    assert res["main"]["verdict"].startswith("no verdict")
    assert res["main"]["pairs"]["seed 20261021"]["missing"] == ["rw_s2s0_300"]
    sec = {c["rl"]: c for c in res["secondary"]}
    assert "missing" in sec["rw_s2s0_300"]["k4"] and "missing" not in sec["rw_s0_150"]["k4"]
    assert "缺" in CP.markdown(res)


def test_sets_must_hold_what_their_names_say(tmp_path):
    sets = make(tmp_path, {}, [("t1", 2, 4)])
    with pytest.raises(ValueError):  # a k = 4 folder under a k = 2 name
        CP.run({**sets, "val_k2": sets["val_k4"]}, {}, None, None)
    with pytest.raises(ValueError):  # a test folder under a val name
        CP.run({**sets, "val_k4": sets["test_k4"]}, {}, None, None)
    with pytest.raises(SystemExit):  # k = 4 without its test set
        CP.run({k: v for k, v in sets.items() if k != "test_k4"}, {}, None, None)
    dup = write_set(tmp_path, "dup", [("x", 2, 4), ("x", 2, 4)], split="val")
    with pytest.raises(ValueError):  # a task twice
        CP.run({**sets, "val_k4": dup}, {}, None, None)


def test_original_vs_perturbed_by_split(tmp_path):
    k4 = [("t1", 2, 4), ("t2", 3, 4)]
    sets = make(tmp_path, {"rw_g_250": {"t1"}}, k4)
    val, test = tmp_path / "orig_val", tmp_path / "orig_test"
    for folder, ids in ((val, ("valt1", "valt2", "vala", "testt1")), (test, ("testt1", "testt2"))):
        folder.mkdir()
        (folder / "results.jsonl").write_text("".join(json.dumps({"task_id": t, "success": True}) + "\n" for t in ids))
    (val / "done.json").write_text("{}")
    res = CP.run(sets, {"rw_g_250": [val, test]}, None, None)
    o = res["original"]["rw_g_250"]
    assert o["k4"]["n"] == 4 and o["k4"]["original"] == 1.0 and o["k4"]["perturbed"] == 0.5
    assert o["k4"]["drop_points"] == 50.0 and o["k4"]["by_band"]["5-"]["n"] == 4
    # k = 2: vala in the val folder; testa / testb nowhere (a test id in the val folder does not count for test)
    assert o["k2"]["n"] == 1 and o["k2"]["tasks_without_original"] == 3
    assert o["original_complete"] == {"val": True, "test": False}
    assert "原起点对退步后" in CP.markdown(res)


def ep(d, success, d_star, task="t"):
    return {"task_id": task, "task_type": "f", "d_star": d_star, "success": success, "replay_same": True, "d": d,
            "actions": len(d) - 1, "turn_actions": 0, "move_actions": len(d) - 1, "collisions": 0,
            "actions_to_success": None}


def test_turn_and_fail_end_sides(tmp_path):
    sets = make(tmp_path, {}, [("t1", 2, 4), ("z", 8, 4)])
    turns = {"episodes": {
        "rw_g_250@k4|val_k4": [ep([6, 5, 4, 3, 2, 1, 0], True, 6, "valt1"), ep([12, 13, 12, 12], False, 12, "valz")],
        "rw_g_250@k4|test_k4": [ep([6, 7, 8], False, 6, "testt1")],
        "nobody@k4|test_k4": [ep([6, 7], False, 6, "testt1")]}}
    fail_end = {"episodes": {
        str(sets["val_k4"] / "rw_g_250"): [{"task_id": "valz", "d_star": 12, "kind": "d", "d_end": 3,
                                            "replay_same": True}],
        str(sets["test_k4"] / "rw_g_250"): [
            {"task_id": "testt1", "d_star": 6, "kind": "near", "d_end": 1, "replay_same": True},
            {"task_id": "testz", "d_star": 12, "kind": "missing", "d_end": None, "replay_same": True}],
        "/elsewhere/val_k4x/rw_g_250": []}}
    res = CP.run(sets, {}, turns, fail_end)
    t = res["turns"]["rw_g_250"]["k4"]
    assert t["5-"]["replayed"] == 2 and t["5-"]["back_to_original_level"] == 0.5
    assert t["0"]["back_to_original_level"] == 0.0 and t["pool"]["replayed"] == 2
    assert t["all"]["back_to_original_level"] == pytest.approx(1 / 3)
    # every row records 5 turns: none of the three replays has 5, so none counts as matching
    assert t["replay_check"]["replayed"] == 3 and t["replay_check"]["turns_match"] == 0
    assert len(t["replay_check"]["mismatches"]) == 3
    assert res["turns_coverage"]["unknown_labels"] == ["nobody@k4|test_k4"]
    assert "rw_s0_150@k4" in res["turns_coverage"]["missing"]
    f = res["fail_end"]["rw_g_250"]["k4"]
    assert f["0"]["failed"] == 2 and f["0"]["known"] == 1 and f["0"]["d_end_mean"] == 3
    assert f["5-"]["d_end_le2_share"] == 1.0 and f["pool"]["failed"] == 1
    assert f["all"]["failed"] == 3 and f["all"]["d_end_mean"] == 2.0
    cov = res["fail_end_coverage"]
    assert cov["unknown_paths"] == ["/elsewhere/val_k4x/rw_g_250"]
    assert "val_k4/rw_g_250: 1 failed episodes replayed, 2 in the result" in cov["count_mismatch"]
    assert not any(x.startswith("test_k4/rw_g_250") for x in cov["count_mismatch"])
    md = CP.markdown(res)
    assert "失败终点" in md and "重放核对" in md and "覆盖核对" in md


def test_replay_check_counts_matching_turns(tmp_path):
    sets = make(tmp_path, {}, [("t1", 2, 4)])
    turns = {"episodes": {"rw_g_250@k4|val_k4": [ep([6, 5, 4, 3, 2, 1], False, 6, "valt1")],
                          "rw_g_250@k4|test_k4": [dict(ep([6, 5, 4, 3, 2, 3], False, 6, "testt1"), replay_same=False)]}}
    res = CP.run(sets, {}, turns, None)
    c = res["turns"]["rw_g_250"]["k4"]["replay_check"]
    assert c["replayed"] == 2 and c["turns_match"] == 2 and c["replay_same"] == 1
    assert c["mismatches"] == ["test_k4/testt1"]
