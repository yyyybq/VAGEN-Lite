"""scripts/measure_throughput.py on SYNTHETIC run directories in the eval/launch.py layout (status.json, logs/,
eval_output/, episodes/) plus a local steps.jsonl and a gpu_sampler.sh log. Numbers are made up for the test."""

import calendar
import json
import os
import time

import pytest

from scripts import measure_throughput as mt

RUN = "s4_L0_radio_311"
EID = f"{RUN}-ep001-env0"


def wall(iso):
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))


def write(path, obj=None, text=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text if text is not None else json.dumps(obj))


def make_agent_run(tmp, steps=3225, dt=0.3):
    run = os.path.join(tmp, "runs", RUN)
    local = os.path.join(tmp, "local")
    write(os.path.join(run, "status.json"), {
        "run_id": RUN, "state": "done", "task": "turning_on_radio", "mode": "public_test", "instance_indices": [10],
        "instance_ids": [311], "num_envs": 1, "direct": False, "seed": 0, "config_name": "L0_exec_only",
        "started_at": "2026-10-22T10:00:00Z", "ended_at": "2026-10-22T10:20:00Z",
        "gpus": {"evaluator": [0], "executor": [1], "agent": []},
        "processes": {"executor": {"started_at": "2026-10-22T10:00:00Z", "healthy_at": "2026-10-22T10:01:30Z"},
                      "agent": {"started_at": "2026-10-22T10:00:00Z", "healthy_at": "2026-10-22T10:00:05Z"},
                      "evaluator": {"started_at": "2026-10-22T10:01:31Z", "ended_at": "2026-10-22T10:20:00Z"}},
        "agent_manifest": {"local": os.path.join(local, RUN, "run_manifest.json")},
    })
    write(os.path.join(run, "logs", "evaluator.log"), text="\n".join([
        "\x1b[0m[00:00:05.000] [INFO] [omnigibson.eval.evaluator] Setting timeout to be 1.5x the average length of "
        "human demos: 3224\x1b[0m",
        "\x1b[0m[00:16:40.000] [INFO] [omnigibson.eval.evaluator] Instance 311 (env 0) finished at step 3225: "
        "success=False q_score=0.0\x1b[0m",
        "[00:16:41.000] [INFO] [omnigibson.eval.eval] Eval summary: 0/1 success | mean q_score=0.000 | task=turning_on_radio",
    ]) + "\n")
    write(os.path.join(run, "eval_output", "json", "turning_on_radio_311_0.json"), {
        "task": "turning_on_radio", "instance_id": 311, "rollout_id": 0, "steps": steps, "success": False,
        "q_score": {"final": 0.0}, "time": {"simulator_steps": steps, "simulator_time": steps / 30, "normalized_time": 0.7}})
    ep = os.path.join(run, "episodes", EID)
    write(os.path.join(ep, "summary.json"), {"episode_id": EID, "run_id": RUN, "task_name": "turning_on_radio",
                                             "tags": {"task": "turning_on_radio", "instance_ids": "311"},
                                             "steps": steps, "elapsed_s": 950.0})
    events = [{"kind": "episode_start", "t": wall("2026-10-22T10:04:00Z")},
              {"kind": "executor", "infer_ms": 50.0, "meta": {"server_timing": {"infer_ms": 40.0}}},
              {"kind": "executor", "infer_ms": 70.0, "meta": {"server_timing": {"infer_ms": 60.0}}},
              {"kind": "provider_call", "latency_s": 1.5}]
    write(os.path.join(ep, "events.jsonl"), text="\n".join(json.dumps(e) for e in events) + "\n")
    write(os.path.join(local, RUN, "episodes", EID, "steps.jsonl"),
          text="".join(json.dumps({"step": i, "t": 1000.0 + i * dt, "src": "exec"}) + "\n" for i in range(steps)))
    rows = ["timestamp,index,uuid,memory_used_mib,memory_total_mib,utilization_pct"]
    for k, (m0, m1) in enumerate([(500, 8500), (8500, 12500)]):
        rows.append(f"2026/10/22 10:0{k}:00.000, 0, GPU-aaa, {m0 if k == 0 else m1}, 81920, {10 * k}")
        rows.append(f"2026/10/22 10:0{k}:00.000, 1, GPU-bbb, {300 + 4500 * k}, 81920, 40")
    write(os.path.join(run, "gpu_samples.csv"), text="\n".join(rows) + "\n")
    write(os.path.join(run, "gpu_apps.csv"), text="timestamp,pid,gpu_uuid,used_memory_mib\n"
          "2026/10/22 10:05:00.000, 12345, GPU-aaa, 11800\n2026/10/22 10:05:02.000, 12345, GPU-aaa, 11900\n")
    return run


def make_direct_run(tmp, with_eval_time=False):
    run = os.path.join(tmp, "runs", "direct")
    write(os.path.join(run, "status.json"), {
        "run_id": "direct", "state": "done", "task": "turning_on_radio", "instance_ids": [311], "num_envs": 1,
        "direct": True, "processes": {"evaluator": {"started_at": "2026-10-22T11:00:00Z", "ended_at": "2026-10-22T11:30:00Z"}}})
    rec = {"task": "turning_on_radio", "instance_id": 311, "rollout_id": 0, "steps": 3225, "success": False,
           "q_score": {"final": 0.0}}
    if with_eval_time:
        rec["time"] = {"evaluation_time": 1075.0, "policy_time": 300.0, "sim_time": 775.0}
    write(os.path.join(run, "eval_output", "json", "turning_on_radio_311_0.json"), rec)
    return run


def test_agent_run_measurements_and_estimates(tmp_path):
    run = make_agent_run(str(tmp_path))
    res = mt.measure([run], parallel_per_gpu=2, gpus=4, ladder_configs=6)
    r = res["runs"][0]
    assert r["scene_load_s"] == pytest.approx(149.0)  # 10:01:31 evaluator start -> 10:04:00 first query
    assert r["executor_load_s"] == pytest.approx(90.0) and r["agent_load_s"] == pytest.approx(5.0)
    assert r["evaluator_log"]["max_steps"] == 3224 and r["evaluator_log"]["last_s"] == pytest.approx(1001.0)
    ep = r["episodes"][0]
    assert ep["episode_s_source"] == "steps_span_s" and ep["episode_s"] == pytest.approx(3224 * 0.3)
    assert ep["steps_per_s"] == pytest.approx(3225 / (3224 * 0.3))
    assert ep["agent_elapsed_s"] == 950.0 and ep["evaluator_finish_s"] == pytest.approx(1000.0) and ep["env"] == 0
    lat = r["latency"]
    assert lat["executor_infer_ms"]["mean"] == pytest.approx(60.0) and lat["server_infer_ms"]["max"] == 60.0
    assert lat["provider_latency_s"]["n"] == 1
    assert lat["step_interval_ms_mean_per_episode"]["mean"] == pytest.approx(300.0)
    g0 = res["gpus"][0]
    assert (g0["first_mib"], g0["peak_mib"], g0["peak_delta_mib"]) == (500, 12500, 12000)
    hint = next(h for h in res["memory_hints"] if h["role"] == "evaluator")
    assert hint["ceiling_per_gpu"] == 81920 // 12000
    assert res["gpu_processes"][0]["peak_mib"] == 11900

    m = res["measured"]
    est = {e["name"]: e for e in res["estimates"]}
    slots = 8
    dev = est["dev"]
    assert dev["episodes"] == 5 * 10 * 3 and est["dev_ladder"]["episodes"] == 6 * 150
    assert dev["as_measured_h"] == pytest.approx(150 * (149.0 + 3224 * 0.3) / slots / 3600)
    catalog = mt.load_catalog(mt.DEFAULT_CATALOG)
    stepping = sum(30 * catalog[t] / m["steps_per_s"] for t in mt.DEFAULT_DEV_TASKS)
    assert dev["timeout_bound_stepping_only_h"] == pytest.approx(stepping / slots / 3600)
    assert dev["timeout_bound_h"] == pytest.approx((stepping + 150 * 149.0) / slots / 3600)
    assert est["report"]["episodes"] == 1000 and len(est["report"]["tasks"]) == 100

    text = mt.render(res)
    assert "Assumptions (command line, NOT measured): 2 parallel" in text and "| dev |" in text


def test_missing_inputs_print_unmeasured(tmp_path, capsys):
    run = make_direct_run(str(tmp_path))
    assert mt.main([run, "--episodes", "40"]) == 0
    out = capsys.readouterr().out
    res = mt.measure([run], episodes=40)
    assert res["runs"][0]["scene_load_s"] is None and res["measured"]["steps_per_s"] is None
    assert all(e["as_measured_h"] is None and e["timeout_bound_h"] is None for e in res["estimates"])
    assert "unmeasured: no gpu_samples.csv" in out
    custom = next(line for line in out.splitlines() if line.startswith("| custom |"))
    assert custom.count("unmeasured") == 3
    assert "| turning_on_radio | 311 | 3225 | unmeasured (-) | unmeasured |" in out


def test_2026_eval_json_gives_episode_time_without_agent(tmp_path):
    run = make_direct_run(str(tmp_path), with_eval_time=True)
    res = mt.measure([run])
    ep = res["runs"][0]["episodes"][0]
    assert ep["episode_s_source"] == "official_evaluation_time_s" and ep["steps_per_s"] == pytest.approx(3225 / 1075.0)
    dev = {e["name"]: e for e in res["estimates"]}["dev"]
    assert dev["timeout_bound_stepping_only_h"] is not None and dev["timeout_bound_h"] is None  # no scene-load measurement


def test_slow_steps_warn_about_the_budget(tmp_path):
    run = make_agent_run(str(tmp_path), steps=100, dt=2.0)
    text = mt.render(mt.measure([run]))
    assert "steps/s < 1" in text


def test_estimates_use_the_pooled_rate_not_the_mean_of_rates(tmp_path):
    """Regression: two episodes at 1 and 10 steps/s averaged to 5.5 steps/s and understated the timeout bound 3x;
    the time per step is total time / total steps."""
    runs = [make_agent_run(str(tmp_path / "a"), steps=100, dt=1.0),
            make_agent_run(str(tmp_path / "b"), steps=100, dt=0.1)]
    res = mt.measure(runs)
    eps = [e for r in res["runs"] for e in r["episodes"]]
    assert sorted(round(e["steps_per_s"], 3) for e in eps) == [round(100 / 99, 3), round(100 / 9.9, 3)]
    assert res["measured"]["steps_per_s"] == pytest.approx(200 / (99 + 9.9))


def test_missing_task_catalog_is_unmeasured_not_zero_hours(tmp_path):
    """Regression: an unreadable --task-catalog gave the 100-task Report set 0 episodes and 0.0 hours."""
    run = make_agent_run(str(tmp_path))
    res = mt.measure([run], task_catalog=str(tmp_path / "no_catalog.tsv"))
    est = {e["name"]: e for e in res["estimates"]}
    for name in ("report", "report_ladder"):
        assert est[name]["episodes"] is None and est[name]["as_measured_h"] is None
        assert est[name]["timeout_bound_h"] is None
    assert est["dev"]["as_measured_h"] is not None  # dev's task list does not come from the catalog
    row = next(line for line in mt.render(res).splitlines() if line.startswith("| report |"))
    assert "unmeasured (no task catalog)" in row and " 0 |" not in row
