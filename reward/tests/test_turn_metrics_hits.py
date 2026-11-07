"""reward.turn_metrics' optimal-action hit rate (phase 3, 2026-11-07): the summary on synthetic episodes."""

from reward import turn_metrics as TM


def _ep(d, hits, turns, mismatch=0):
    return {"d": d, "success": False, "replay_same": True, "actions": 4, "turn_actions": 1, "move_actions": 3,
            "collisions": 0, "d_star": 3, "actions_to_success": None, "act_hits": hits, "turn_hits": turns,
            "hit_mismatch": mismatch}


def test_hit_rates_in_the_summary():
    eps = [_ep([3, 2, 2], [True, False, True], [True, None]), _ep([3, 4], [False], [False], mismatch=1)]
    s = TM.summarize(eps)
    assert s["action_hit_rate"] == 2 / 4 and s["actions_judged"] == 4
    assert s["turn_hit_rate"] == 1 / 2 and s["turns_judged_for_hits"] == 2 and s["hit_pose_mismatch"] == 1
    old = [{k: v for k, v in e.items() if k not in ("act_hits", "turn_hits", "hit_mismatch")} for e in eps]
    s_old = TM.summarize(old)  # a result replayed before the hit rate: the same keys as before, nothing added
    assert "action_hit_rate" not in s_old and {k: s[k] for k in s_old} == s_old
