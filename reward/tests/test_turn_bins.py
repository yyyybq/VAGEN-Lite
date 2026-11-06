"""reward.turn_bins: bands, the failed-episode split and the per-label merge."""

import pytest

from reward import turn_bins as TB


def ep(d, success, d_star, task="t"):
    return {"task_id": task, "task_type": "x", "d_star": d_star, "success": success, "replay_same": True, "d": d,
            "actions": len(d) - 1, "turn_actions": 0, "move_actions": len(d) - 1, "collisions": 0,
            "actions_to_success": None}


def test_parse_bands():
    assert TB.parse_bands("1-3,4-5,6-8") == [("1-3", 1, 3), ("4-5", 4, 5), ("6-8", 6, 8)]
    assert TB.parse_bands("0,1-2,5-") == [("0", 0, 0), ("1-2", 1, 2), ("5-", 5, TB.INF)]
    with pytest.raises(ValueError):
        TB.parse_bands("1-3,1-3")
    with pytest.raises(ValueError):
        TB.parse_bands("4-2")
    assert TB.band_of(7, TB.parse_bands("0,1-2,5-")) == "5-"
    assert TB.band_of(3, TB.parse_bands("0,1-2,5-")) is None


@pytest.mark.parametrize("d,success,kind", [
    ([5, 4, 3, 2, 1, 0], True, None),
    ([5, 4, 4, 3], False, "no_error"),            # never grew: stalled
    ([5, 6, 4, 3], False, "recovered_ran_out"),   # erred, ended at its closest
    ([5, 6, 5, 6, 5], False, "recovered_ran_out"),  # back to the start each time: still its closest
    ([5, 3, 4, 4, 4], False, "not_recovered"),    # lost ground at turn 2, three turns left
    ([5, 4, 3, 4], False, "not_recovered_late"),  # its only error in the last turn
    ([5, 4, 3, 4, 4], False, "not_recovered_late"),  # last error in the second-to-last turn
    ([5, None], False, "unknown"),
    ([None, None], False, "unknown"),
])
def test_failure_kind(d, success, kind):
    assert TB.failure_kind(ep(d, success, 5)) == kind


def test_run_merges_sets_and_bands_by_margin():
    metrics = {"episodes": {
        "a@50|val": [ep([4, 3, 2, 1, 0], True, 4), ep([11, 12, 11, 10], False, 11)],
        "a@50|test": [ep([12, 11, 12, 13], False, 12), {"task_id": "z", "skipped": "no table"}],
        "b@0|val": [ep([7, 6, 5], False, 7)],
    }}
    res = TB.run(metrics, "margin", TB.parse_bands("0,1-2,3-"), 12, {"all": ["*@*"]}, ["a@*"])
    a = res["labels"]["a@50"]
    assert a["0"]["replayed"] == 1 and a["1-2"]["replayed"] == 1 and a["3-"]["replayed"] == 1
    assert a["all"]["replayed"] == 3 and a["replay_errors"] == 1
    assert a["0"]["failed_split"]["not_recovered_late"] == 1
    assert a["1-2"]["failed_split"]["recovered_ran_out"] == 1
    assert a["all"]["episode_error_share"] == pytest.approx(2 / 3)
    g = res["groups"]["all"]
    assert g["labels"] == ["a@50", "b@0"] and g["all"]["replayed"] == 4 and g["3-"]["failed_split"]["no_error"] == 1
    assert g["all"]["fail_end_mean"] == pytest.approx((10 + 13 + 5) / 3)
    assert g["all"]["failed_beyond_start"] == 1 and g["all"]["failed_beyond_start_share"] == pytest.approx(1 / 3)
    assert TB.table(res).count("|") > 0 and "9.33" in TB.table(res)
    with pytest.raises(SystemExit):
        TB.run(metrics, "dstar", TB.parse_bands("1-3"), 12, {"x": ["nothing@*"]}, [])


def test_summarize_keeps_efficiency_unrounded():
    # 2026-11-06: the efficiency means were rounded to 3 places here and again for display
    from reward.turn_metrics import summarize

    eps = [ep([3, 2, 1, 0], True, 3), ep([3, 2, 1, 1, 0], True, 3), ep([3, 2, 2, 1, 0], True, 3, task="u")]
    for e in eps:
        e["actions_to_success"] = 3
    s = summarize(eps)
    assert s["success_actions_over_dstar_mean"] == pytest.approx(11 / 9, rel=1e-12)  # (3 + 4 + 4) / 3 / 3
    assert s["success_actions_over_dstar_median"] == pytest.approx(4 / 3, rel=1e-12)
    assert s["success_to_success_over_dstar_mean"] == pytest.approx(1.0, rel=1e-12)
