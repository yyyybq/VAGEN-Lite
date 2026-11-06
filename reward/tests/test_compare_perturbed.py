"""reward.compare_perturbed on made-up results in a temporary folder: the pool leaves margin-0 tasks out, the main
verdict follows PREREG_PERTURB section 4, incomplete results give no verdict, the sign test."""

import json

from reward import compare_perturbed as CP


def write_set(root, name, tasks):
    """tasks: [(task_id, d_orig, k)]"""
    folder = root / name
    folder.mkdir()
    rows = [{"task_id": t, "task_type": "f", "difficulty": {"d_star": d + k},
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


def test_sign_test():
    assert CP.sign_test(0, 0) is None
    assert CP.sign_test(5, 0) == 0.0625
    assert CP.sign_test(3, 3) == 1.0


def test_pool_and_verdict(tmp_path):
    # k = 4: tasks t1..t9 with d* 2 (margin 6), z with d* 8 (margin 0)
    k4 = [(f"t{i}", 2, 4) for i in range(1, 10)] + [("z", 8, 4)]
    solved = {"rw_ok_250": {"t1", "t2", "t3"}, "rw_s0_150": {"t1", "z"},
              "rw_s2ok_250": {"t1", "t2"}, "rw_s2s0_300": {"t1"}}
    res = CP.run(make(tmp_path, solved, k4), {}, None, None)
    pairs = res["main"]["pairs"]
    # pool = 9 tasks a set x 2 sets; z (margin 0) is out, so S0's z does not count
    assert pairs["seed 20261019"]["n"] == 18
    assert pairs["seed 20261019"]["wins"] == 4 and pairs["seed 20261019"]["losses"] == 0
    assert abs(pairs["seed 20261019"]["diff_raw"] - 100 * 4 / 18) < 1e-9
    assert abs(pairs["seed 20261021"]["diff_raw"] - 100 * 2 / 18) < 1e-9
    assert res["main"]["verdict"] == "pass"  # mean (22.2 + 11.1) / 2 >= 5, both above 0
    band0 = res["main"]["reported_besides"]["seed 20261019"]["k4_by_band"]["0"]
    assert band0["n"] == 2 and band0["losses"] == 2
    assert res["success"]["rw_s0_150"]["k4"]["by_band"]["0"]["rate"] == 1.0


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


def test_original_vs_perturbed(tmp_path):
    k4 = [("t1", 2, 4), ("t2", 3, 4)]
    sets = make(tmp_path, {"rw_g_250": {"t1"}}, k4)
    orig = tmp_path / "orig"
    orig.mkdir()
    rows = [{"task_id": t, "success": True} for t in ("valt1", "valt2", "testt1", "testt2", "vala")]
    (orig / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = CP.run(sets, {"rw_g_250": [orig]}, None, None)
    o = res["original"]["rw_g_250"]["k4"]
    assert o["n"] == 4 and o["original"] == 1.0 and o["perturbed"] == 0.5 and o["drop_points"] == 50.0
    assert res["original"]["rw_g_250"]["k2"]["n"] == 1 and res["original"]["rw_g_250"]["k2"]["tasks_without_original"] == 3


def ep(d, success, d_star):
    return {"task_id": "t", "task_type": "f", "d_star": d_star, "success": success, "replay_same": True, "d": d,
            "actions": len(d) - 1, "turn_actions": 0, "move_actions": len(d) - 1, "collisions": 0,
            "actions_to_success": None}


def test_turn_and_fail_end_sides(tmp_path):
    sets = make(tmp_path, {}, [("t1", 2, 4), ("z", 8, 4)])
    turns = {"episodes": {
        "rw_g_250@k4|val_k4": [ep([6, 5, 4, 3, 2, 1, 0], True, 6), ep([12, 13, 12, 12], False, 12)],
        "rw_g_250@k4|test_k4": [ep([6, 7, 8], False, 6)]}}
    fail_end = {"episodes": {
        str(sets["val_k4"] / "rw_g_250"): [{"task_id": "valz", "d_star": 12, "kind": "d", "d_end": 3,
                                            "replay_same": True}],
        str(sets["test_k4"] / "rw_g_250"): [
            {"task_id": "testt1", "d_star": 6, "kind": "near", "d_end": 1, "replay_same": True},
            {"task_id": "testz", "d_star": 12, "kind": "missing", "d_end": None, "replay_same": True}]}}
    res = CP.run(sets, {}, turns, fail_end)
    t = res["turns"]["rw_g_250"]["k4"]
    assert t["5-"]["replayed"] == 2 and t["5-"]["back_to_original_level"] == 0.5
    assert t["0"]["back_to_original_level"] == 0.0 and t["pool"]["replayed"] == 2
    assert t["all"]["back_to_original_level"] == round(1 / 3, 4)
    f = res["fail_end"]["rw_g_250"]["k4"]
    assert f["0"]["failed"] == 2 and f["0"]["known"] == 1 and f["0"]["d_end_mean"] == 3
    assert f["5-"]["d_end_le2_share"] == 1.0 and f["pool"]["failed"] == 1
    assert f["all"]["failed"] == 3 and f["all"]["d_end_mean"] == 2.0
    assert "失败终点" in CP.markdown(res)
