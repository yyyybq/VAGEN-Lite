"""reward.drift_p3: the brief's drift definition on synthetic training logs (temp folders only)."""

import json

from reward import drift_p3


def _log(values):
    return "\n".join(f"(TaskRunner pid=1) step:{k} - actor/entropy:0.02 - {drift_p3.KEY}:{v} - x/y:1.0"
                     for k, v in values.items())


def test_rolling_needs_a_full_window():
    vals = {k: 0.0 for k in range(1, 10)}
    assert drift_p3.rolling(vals) == {}
    vals[10] = 0.1
    assert drift_p3.rolling(vals) == {10: 0.01}


def test_recovered_run_is_not_drift():
    vals = {k: 0.0 for k in range(1, 301)}
    for k in range(20, 60):
        vals[k] = 0.03  # above 1% for a while
    r = drift_p3.classify(vals)
    assert r["drift"] is False and r["warnings"] == 1
    assert r["warning_list"][0]["back_below_recover"] is not None
    assert r["reason"].startswith("back below")


def test_never_back_is_drift_and_counts_crossings():
    vals = {k: 0.0 for k in range(1, 301)}
    for k in range(100, 110):
        vals[k] = 0.02  # first warning, back below 0.5% afterwards
    for k in range(200, 301):
        vals[k] = 0.008 if k % 2 else 0.02  # 10-update mean 1.4%, never below 0.5%
    r = drift_p3.classify(vals)
    assert r["drift"] is True and r["stopped"] is False
    assert r["warnings"] == 2 and r["first_over_warn"] >= 100
    assert r["never_back_below_recover"] is True


def test_guard_stop_reason_and_late_flag():
    vals = {k: 0.0 for k in range(1, 245)}
    for k in range(235, 245):
        vals[k] = 0.11
    stopped = drift_p3.classify(vals, guard_stop="unparseable answers 0.109 > 0.1 (mean of updates 235-244)")
    assert stopped["drift"] is True and stopped["stopped"] is True and stopped["late_last_warning"] is False
    late = {k: 0.0 for k in range(1, 301)}
    for k in range(280, 301):
        late[k] = 0.03
    r = drift_p3.classify(late, guard_stop=None)
    assert r["drift"] is True and r["late_last_warning"] is True and r["stopped"] is False


def test_two_crossings_before_one_recovery_both_recover():
    vals = {k: 0.0 for k in range(1, 151)}
    for k in list(range(20, 30)) + list(range(35, 45)):
        vals[k] = 0.019  # 10-update mean: above 1% from 25, 0.95% at 34 (not below 0.5%), above 1% again at 40
    r = drift_p3.classify(vals)
    assert r["warnings"] == 2 and r["drift"] is False
    assert [w["update"] for w in r["warning_list"]] == [25, 40]
    assert [w["back_below_recover"] for w in r["warning_list"]] == [52, 52]


def test_other_stops_are_judged_by_the_second_clause():
    vals = {k: 0.0 for k in range(1, 121)}
    for k in range(100, 121):
        vals[k] = 0.03
    r = drift_p3.classify(vals, guard_stop="greedy evaluation of update 100 on val: unreadable turns 0.12 > 0.1",
                          have_progress=True)
    assert r["stopped"] is False and r["other_stop"].startswith("greedy") and r["drift"] is True
    quiet = drift_p3.classify({k: 0.0 for k in range(1, 61)}, guard_stop="KL to the start model 0.3 > 0.2 at update 60",
                              have_progress=True)
    assert quiet["drift"] is False and quiet["other_stop"].startswith("KL")
    high = {k: 0.2 for k in range(1, 30)}
    assert drift_p3.classify(high, None, have_progress=True)["stopped"] is False
    assert drift_p3.classify(high, None, have_progress=False)["stopped"] is True


def test_retry_drops_the_abandoned_tail(tmp_path):
    run = tmp_path / "rl_run_y" / "logs"
    run.mkdir(parents=True)
    (run / "rl-20261101T000000Z-pt-a-worker-0.log").write_text(_log({k: 0.0 for k in range(1, 181)}))
    (run / "rl-20261101T060000Z-pt-b-worker-0.log").write_text("no step lines here\n")
    (run / "rl-20261102T000000Z-pt-c-worker-0.log").write_text(_log({k: 0.001 for k in range(151, 166)}))
    vals = drift_p3.series(drift_p3.run_dir_logs(run.parent))
    assert max(vals) == 165 and vals[150] == 0.0 and vals[151] == 0.001 and len(vals) == 165


def test_logs_later_file_wins_and_cli(tmp_path):
    run = tmp_path / "rl_run_x" / "logs"
    run.mkdir(parents=True)
    (run / "rl-20261101T000000Z-pt-a-worker-0.log").write_text(_log({k: 0.05 for k in range(1, 21)}))
    (run / "rl-20261102T000000Z-pt-b-worker-0.log").write_text(_log({k: 0.0 for k in range(11, 31)}))
    vals = drift_p3.series(drift_p3.run_dir_logs(run.parent))
    assert vals[5] == 0.05 and vals[15] == 0.0 and max(vals) == 30
    prog = tmp_path / "progress.json"
    prog.write_text(json.dumps({"guard_stops": {}}))
    out = tmp_path / "d.json"
    drift_p3.main(["--run", f"x={run.parent}@{prog}", "--out", str(out)])
    res = json.loads(out.read_text())["runs"]["x"]
    assert res["warnings"] == 1 and res["drift"] is False
    assert out.with_suffix(".md").exists()
