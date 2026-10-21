"""eval/aggregate.py on SYNTHETIC official result files (2 tasks x 3 instances x 2 configs).

Every JSON written here carries "_synthetic": true; none of these numbers is a result.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from eval import aggregate as agg

TASKS = ("turning_on_radio", "picking_up_trash")
IDS = (301, 302, 303)

# config -> task -> instance -> (q, extra official keys, agent summary overrides)
Q = {
    "L0_exec_only": {
        "turning_on_radio": {301: 1.0, 302: 0.5, 303: 0.0},
        "picking_up_trash": {301: 0.2, 302: 0.4, 303: 0.6},
    },
    "L1_plan": {
        "turning_on_radio": {301: 1.0, 302: 1.0, 303: 0.5},
        "picking_up_trash": {301: 0.4, 302: 0.6, 303: 0.0},
    },
}
OFFICIAL_EXTRA = {
    # 2026/eval-style budget failure: official q already 0, not an infra error
    ("L0_exec_only", "turning_on_radio", 303): {"failure_reason": "group_time_budget_exceeded"},
    # connection lost -> infra error, excluded from Q means
    ("L1_plan", "picking_up_trash", 303): {"failure_reason": "policy_connection_lost"},
}
SUMMARY_EXTRA = {
    # over the 2026 budget per the agent's own clock (no failure_reason in the JSON)
    ("L1_plan", "turning_on_radio", 302): {"elapsed_s": 150.0},
}


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def official_record(task, iid, q, steps=90, rollout=0, **extra):
    rec = {"_synthetic": True, "task": task, "instance_id": iid, "rollout_id": rollout, "steps": steps,
           "success": q == 1.0, "agent_distance": {"base": 1.0, "left": 2.0, "right": 2.0},
           "normalized_agent_distance": {"base": 0.8, "left": 0.9, "right": 1.1},
           "q_score": {"final": q},
           "time": {"simulator_steps": steps, "simulator_time": steps / 30.0, "normalized_time": 1.0}}
    rec.update(extra)
    return rec


def make_run(root, config, task, values, seed=0, run_suffix="", official_extra=None, summary_extra=None,
             status_extra=None, skip=()):
    run_id = f"{config}-{task}-s{seed}{run_suffix}"
    run = os.path.join(root, run_id)
    cfg = {"run": {"name": config, "seed": seed, "log_root": f"/nas/{run_id}"}, "server": {"port": 8100 + seed},
           "modules": {"plan": {"enabled": config != "L0_exec_only"}}}
    _write_json(os.path.join(run, "run_manifest.json"),
                {"_synthetic": True, "run_id": run_id, "config_hash": f"hash-{run_id}", "config": cfg, "seed": seed})
    status = {"_synthetic": True, "run_id": run_id, "task": task, "instance_ids": list(values), "mode": "public_test",
              "eval_output_dir": "eval_output", "max_steps": 100}
    status.update(status_extra or {})
    _write_json(os.path.join(run, "status.json"), status)
    for iid, q in values.items():
        if iid in skip:
            continue
        extra = (official_extra or {}).get(iid, {})
        _write_json(os.path.join(run, "eval_output", "json", f"{task}_{iid}_0.json"), official_record(task, iid, q, **extra))
        eid = f"{run_id}-ep001-env{list(values).index(iid)}"
        summary = {"_synthetic": True, "episode_id": eid, "run_id": run_id, "task_name": task,
                   "tags": {"task": task, "instance": str(iid)}, "steps": 90, "elapsed_s": 60.0,
                   "wallclock_budget_s": 100.0, "provider_calls": 2 if config != "L0_exec_only" else 0,
                   "provider_s": 3.0, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.01, "provider_errors": 0,
                   "recoveries": 0, "replans": 0, "fallbacks": 0, "subtasks": []}
        summary.update((summary_extra or {}).get(iid, {}))
        _write_json(os.path.join(run, "episodes", eid, "summary.json"), summary)
    return run


@pytest.fixture()
def runs(tmp_path):
    out = []
    for config in Q:
        for task in TASKS:
            out.append(make_run(
                str(tmp_path / "runs"), config, task, Q[config][task],
                official_extra={iid: OFFICIAL_EXTRA[(config, task, iid)] for iid in IDS if (config, task, iid) in OFFICIAL_EXTRA},
                summary_extra={iid: SUMMARY_EXTRA[(config, task, iid)] for iid in IDS if (config, task, iid) in SUMMARY_EXTRA}))
    return out


def by_config(result):
    return {c["config"]: c for c in result["configs"]}


def tree_hashes(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def test_per_task_and_overall_means(runs, tmp_path):
    res = agg.aggregate(runs, str(tmp_path / "out"), n_boot=200)
    cfg = by_config(res)
    assert set(cfg) == {"L0_exec_only", "L1_plan"}
    a, b = cfg["L0_exec_only"], cfg["L1_plan"]
    per_task = {(t["config"], t["task"]): t for t in res["per_task"]}
    assert per_task[("L0_exec_only", "turning_on_radio")]["mean_q"] == pytest.approx(0.5)
    assert per_task[("L0_exec_only", "picking_up_trash")]["mean_q"] == pytest.approx(0.4)
    assert a["mean_q"] == pytest.approx(0.45)  # mean over tasks of per-task means
    assert per_task[("L1_plan", "turning_on_radio")]["mean_q"] == pytest.approx(2.5 / 3)
    assert per_task[("L1_plan", "picking_up_trash")]["mean_q"] == pytest.approx(0.5)  # infra 303 excluded
    assert b["mean_q"] == pytest.approx((2.5 / 3 + 0.5) / 2)
    assert a["success_rate"] == pytest.approx((1 / 3 + 0) / 2)
    assert a["mean_time_score"] == pytest.approx(1.0)  # normalized_time 1 -> 3 - 2/1
    assert "SUBSET" in a["scope"]  # no declared list -> labelled as a subset mean
    # seeds pool into one config even though ports and log roots differ per run
    assert len(res["configs"]) == 2 and len(a["runs"]) == 2


def test_infra_errors_excluded_but_listed(runs, tmp_path):
    res = agg.aggregate(runs, str(tmp_path / "out"), n_boot=0)
    b = by_config(res)["L1_plan"]
    assert b["n_infra"] == 1 and b["n_included"] == 5
    assert [(r["task"], r["instance_id"]) for r in res["infra_errors"]] == [("picking_up_trash", 303)]
    assert b["mean_q_infra_as_zero"] == pytest.approx((2.5 / 3 + 1.0 / 3) / 2)
    md = open(tmp_path / "out" / "summary.md", encoding="utf-8").read()
    assert "Infra errors (excluded from Q means)" in md and "failure_reason=policy_connection_lost" in md
    # infra can also be declared as an empty set of failure reasons -> the row counts as an ordinary 0
    res2 = agg.aggregate(runs, None, n_boot=0, infra_reasons=())
    assert by_config(res2)["L1_plan"]["n_infra"] == 0


def test_budget_rule_column_is_separate(runs, tmp_path):
    res = agg.aggregate(runs, str(tmp_path / "out"), n_boot=0)
    rows = {(r["config"], r["task"], r["instance_id"]): r for r in res["rows"]}
    over = rows[("L1_plan", "turning_on_radio", 302)]
    assert over["q_final"] == 1.0 and over["budget_ok"] is False and over["budget_source"] == "agent_elapsed"
    assert over["q_budget2026"] == 0.0
    timed_out = rows[("L0_exec_only", "turning_on_radio", 303)]
    assert timed_out["budget_ok"] is False and timed_out["budget_source"] == "failure_reason"
    assert rows[("L0_exec_only", "turning_on_radio", 301)]["budget_ok"] is True
    b = by_config(res)["L1_plan"]
    assert b["mean_q"] == pytest.approx((2.5 / 3 + 0.5) / 2)  # raw official Q untouched
    assert b["mean_q_budget2026"] == pytest.approx((1.5 / 3 + 0.5) / 2)
    with open(tmp_path / "out" / "episodes.csv", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "q_final" in header and "q_budget2026" in header and "budget_ok" in header


def test_bootstrap_is_deterministic_with_seed(runs, tmp_path):
    r1 = agg.aggregate(runs, None, n_boot=500, seed=0)
    r2 = agg.aggregate(runs, None, n_boot=500, seed=0)
    r3 = agg.aggregate(runs, None, n_boot=500, seed=1)
    ci = lambda r: [(c["mean_q_ci"], c["mean_q_budget2026_ci"]) for c in r["configs"]]
    assert ci(r1) == ci(r2)
    assert ci(r1) != ci(r3)
    for c in r1["configs"]:
        lo, hi = c["mean_q_ci"]
        assert lo <= c["mean_q"] <= hi
    p1 = [p["diff_ci"] for p in r1["paired"]]
    assert p1 == [p["diff_ci"] for p in r2["paired"]]


def test_stratified_bootstrap_resamples_within_tasks():
    import numpy as np

    by_task = {"a": [1.0, 1.0], "b": [0.0, 0.0, 0.0]}  # no within-task variance -> no CI width
    s = agg.stratified_bootstrap(by_task, 100, np.random.default_rng(0))
    assert np.allclose(s, 0.5)


def test_paired_comparison_on_shared_units(runs, tmp_path):
    res = agg.aggregate(runs, None, n_boot=300, compare=[("L0_exec_only", "L1_plan")])
    p = next(p for p in res["paired"] if p["metric"] == "q_final")
    assert p["n_pairs"] == 5 and p["n_tasks"] == 2 and p["only_a"] == 1 and p["only_b"] == 0
    assert p["diff_b_minus_a"] == pytest.approx(((0 + 0.5 + 0.5) / 3 + (0.2 + 0.2) / 2) / 2)
    assert p["mean_a"] == pytest.approx((0.5 + 0.3) / 2)
    lo, hi = p["diff_ci"]
    assert lo <= p["diff_b_minus_a"] <= hi


def test_declared_instances_zero_fill_missing(runs, tmp_path):
    declared = tmp_path / "report.yaml"
    declared.write_text("mode: public_test\nindices: 0-3\ntasks: [turning_on_radio, picking_up_trash]\n")
    res = agg.aggregate(runs, str(tmp_path / "out"), declared_path=str(declared), n_boot=0)
    a, b = by_config(res)["L0_exec_only"], by_config(res)["L1_plan"]
    assert a["n_missing"] == 2 and b["n_missing"] == 2  # instance 304 of both tasks
    assert a["mean_q"] == pytest.approx(((1 + 0.5 + 0 + 0) / 4 + (0.2 + 0.4 + 0.6 + 0) / 4) / 2)
    assert b["mean_q"] == pytest.approx(((1 + 1 + 0.5 + 0) / 4 + (0.4 + 0.6 + 0) / 3) / 2)
    missing = [(r["config"], r["task"], r["instance_id"]) for r in res["missing"]]
    assert ("L0_exec_only", "turning_on_radio", 304) in missing
    assert all(r["q_final"] == 0.0 and r["missing"] for r in res["missing"])
    assert "declared list" in a["scope"]
    assert next(p for p in res["paired"] if p["metric"] == "q_final")["n_pairs"] == 5  # zero-filled units not paired
    md = open(tmp_path / "out" / "summary.md", encoding="utf-8").read()
    assert "Missing results (count as Q = 0)" in md and "304" in md


def test_declared_list_excludes_other_instances(runs, tmp_path):
    declared = tmp_path / "subset.json"
    declared.write_text(json.dumps({"tasks": {"turning_on_radio": [301, 302], "picking_up_trash": [301]}}))
    res = agg.aggregate(runs, None, declared_path=str(declared), n_boot=0)
    a = by_config(res)["L0_exec_only"]
    assert a["n_outside_declared"] == 3 and a["n_missing"] == 0
    assert a["mean_q"] == pytest.approx(((1 + 0.5) / 2 + 0.2) / 2)


def test_infra_rerun_rule(tmp_path):
    root = str(tmp_path / "runs")
    first = make_run(root, "L0_exec_only", "turning_on_radio", {301: 0.0}, run_suffix="-a",
                     official_extra={301: {"failure_reason": "policy_connection_lost"}})
    second = make_run(root, "L0_exec_only", "turning_on_radio", {301: 0.3}, run_suffix="-b")
    third = make_run(root, "L0_exec_only", "turning_on_radio", {301: 0.9}, run_suffix="-c")
    res = agg.aggregate([first, second, third], None, n_boot=0)
    rows = sorted(res["rows"], key=lambda r: r["run_id"])
    assert [r["superseded"] for r in rows] == [True, False, True]
    assert rows[0]["superseded_reason"] == "infra attempt, re-run"
    assert res["configs"][0]["mean_q"] == pytest.approx(0.3)  # never the better later re-run
    assert any("must not be re-run" in w for w in res["warnings"])


def test_status_infra_and_missing_results(tmp_path):
    root = str(tmp_path / "runs")
    crashed = make_run(root, "L0_exec_only", "turning_on_radio", {301: 0.5, 302: 0.5}, skip=(302,),
                       status_extra={"evaluator_exit_code": 1})
    res = agg.aggregate([crashed], None, n_boot=0)
    rows = {r["instance_id"]: r for r in res["rows"]}
    assert rows[302]["infra_error"] and rows[302]["infra_reason"] == "evaluator exit code 1"
    quiet = make_run(root, "L1_plan", "turning_on_radio", {301: 0.5, 302: 0.5}, skip=(302,))
    res = agg.aggregate([quiet], None, n_boot=0)
    rows = {r["instance_id"]: r for r in res["rows"]}
    assert rows[302]["missing"] and rows[302]["q_final"] == 0.0
    assert res["configs"][0]["mean_q"] == pytest.approx(0.25)


def test_summary_join_by_env_index(tmp_path):
    """num_envs > 1: one server episode per env; the summary carries no instance tag."""
    root = str(tmp_path / "runs")
    run = make_run(root, "L1_plan", "turning_on_radio", {301: 1.0, 302: 0.5})
    for d in os.listdir(os.path.join(run, "episodes")):
        p = os.path.join(run, "episodes", d, "summary.json")
        s = json.load(open(p))
        s["tags"] = {"task": "turning_on_radio"}
        s["provider_calls"] = 7 if s["episode_id"].endswith("env1") else 3
        _write_json(p, s)
    res = agg.aggregate([run], None, n_boot=0)
    rows = {r["instance_id"]: r for r in res["rows"]}
    assert rows[301]["join"] == rows[302]["join"] == "status_env_index"
    assert (rows[301]["provider_calls"], rows[302]["provider_calls"]) == (3, 7)


def test_inputs_untouched_and_provenance(runs, tmp_path):
    before = tree_hashes(str(tmp_path / "runs"))
    res = agg.aggregate(runs, str(tmp_path / "out"), n_boot=50, argv=["test"])
    assert tree_hashes(str(tmp_path / "runs")) == before
    summary = json.load(open(tmp_path / "out" / "summary.json"))
    files = {i["path"]: i["sha256"] for i in summary["provenance"]["inputs"]}
    assert len(files) == 4 * (2 + 3 + 3)  # per run: status + manifest, 3 official JSONs, 3 summaries
    assert all(before[p] == h for p, h in files.items())
    assert summary["provenance"]["generated_at"] and "commit" in summary["provenance"]["code"]
    assert summary["synthetic_inputs"] is True
    assert "SYNTHETIC" in open(tmp_path / "out" / "summary.md", encoding="utf-8").read()
    for name in agg.OUTPUT_FILES:
        assert os.path.isfile(tmp_path / "out" / name)
    with pytest.raises(SystemExit):
        agg.aggregate(runs, os.path.join(runs[0], "eval_output", "tables"), n_boot=0)


def test_cli_entry_point(runs, tmp_path):
    out = tmp_path / "cli_out"
    proc = subprocess.run([sys.executable, "-m", "eval.aggregate", *runs, "--out", str(out), "--bootstrap", "100",
                           "--seed", "0"], cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT},
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "L0_exec_only: mean Q 0.450" in proc.stdout
    assert (out / "per_config.csv").is_file()


def test_resolve_instance_ids_matches_official():
    assert agg.resolve_instance_ids("public_test", agg.parse_indices("0-9")) == list(range(301, 311))
    assert agg.resolve_instance_ids("public_test", [10, 19]) == [311, 320]
    assert agg.resolve_instance_ids("hidden_test", [0]) == [321]
    assert agg.resolve_instance_ids("train", [7]) == [7]
    with pytest.raises(ValueError):
        agg.resolve_instance_ids("public_test", [20])
    assert agg.time_score(1.0) == pytest.approx(1.0)
    assert agg.time_score(1 / 1.5) == pytest.approx(0.0)  # official failure normalized_time
