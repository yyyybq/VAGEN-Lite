"""eval/batch.py: spec expansion (incl. the two example specs), deterministic run ids, launch argv, and the
retry policy with a MOCKED job runner (no eval.launch processes, no GPU)."""

import json
import os
import sys
import threading
import time

import pytest
import yaml

from conftest import ROOT
from eval import batch, gpu, launch

L0 = os.path.join(ROOT, "configs", "L0_exec_only.yaml")
L1 = os.path.join(ROOT, "configs", "L1_plan.yaml")
FOUR_FREE = gpu.parse_gpu_csv("".join(f"{i}, 5, 81920, 0, A100, GPU-{i}\n" for i in range(4)))


def spec_of(**kw):
    spec = {"batch": "b", "mode": "public_test", "instances": [10], "jobs": [{"name": "L0", "config": L0, "task": "t"}]}
    spec.update(kw)
    return spec


def result(present=True, success=False, q=0.0, failure_reason=None):
    return {"file": "t_311_0.json", "present": present, "success": success, "q_score": q, "steps": 100,
            "failure_reason": failure_reason}


DONE = {"state": "done", "evaluator_started": True, "results": [result()], "results_complete": True, "reason": []}
INFRA = {"state": "infra_error", "evaluator_started": True, "results": [result(present=False)],
         "infra_kinds": ["no_results", "exit_code"], "results_complete": False, "reason": ["evaluator exit code 1"]}
EXIT_ONLY = {"state": "infra_error", "evaluator_started": True, "results": [result()], "infra_kinds": ["exit_code"],
             "results_complete": True, "reason": ["evaluator exit code -11"]}
FAILED = {"state": "failed", "evaluator_started": False, "reason": ["eval.launch exited 1 without status.json"]}


class ScriptedRunner:
    """Mock job runner: returns the next scripted launch status for the job; records every call."""

    def __init__(self, script, delay=0.0):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []
        self.delay = delay
        self.lock = threading.Lock()
        self.active, self.max_active, self.held = 0, 0, []

    def __call__(self, job, argv, run_dir, run_id):
        gpus = argv[argv.index("--gpus") + 1].split(",") if "--gpus" in argv else []
        ports = (argv[argv.index("--executor-port") + 1], argv[argv.index("--agent-port") + 1])
        with self.lock:
            assert not set(gpus) & {g for h in self.held for g in h[0]}, "GPU handed to two running jobs"
            assert ports not in [h[1] for h in self.held], "port slot handed to two running jobs"
            entry = (gpus, ports)
            self.held.append(entry)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append({"job": job.job_id, "run_id": run_id, "run_dir": run_dir, "argv": list(argv)})
        time.sleep(self.delay)
        with self.lock:
            self.held.remove(entry)
            self.active -= 1
            return dict(self.script[job.job_id].pop(0))


def make_runner(spec, tmp_path, runner, **kw):
    jobs = batch.expand_jobs(spec)
    kw.setdefault("gpu_picker", lambda n, exclude: gpu.pick_free_gpus(n, gpus=FOUR_FREE, exclude=exclude))
    return batch.BatchRunner(spec, jobs, batch_dir=str(tmp_path / "batch"), runner=runner,
                             run_dir_for=lambda job, run_id: str(tmp_path / "runs" / run_id),
                             gpus_needed=lambda job: 1 if job.no_executor else 2, sleep=lambda s: None,
                             log=lambda m: None, **kw), jobs


def ledger(tmp_path):
    return [json.loads(x) for x in open(tmp_path / "batch" / "jobs.jsonl")]


def bstatus(tmp_path):
    return json.load(open(tmp_path / "batch" / "batch_status.json"))


# ---------------------------------------------------------------------- expansion


def test_step3_example_spec():
    path = os.path.join(ROOT, "configs", "batches", "step3_turning_on_radio.yaml")
    jobs = batch.expand_jobs(batch.load_spec(path))
    assert [j.job_id for j in jobs] == ["L0-turning_on_radio-pt0-default-s0", "L0-turning_on_radio-pt0-rgbd-s0"]
    for j in jobs:
        assert j.config == L0 and j.task == "turning_on_radio" and j.mode == "public_test" and j.instances == (0,)
        assert j.split == "report" and not j.direct and "Step-3 baseline run" in j.note
        assert dict(j.launch) == {"write_video": True}
    assert batch.attempt_run_id("step3_turning_on_radio", jobs[1].job_id, 0) == \
        "step3_turning_on_radio-L0-turning_on_radio-pt0-rgbd-s0-a0"


def test_dev_ladder_example_spec():
    jobs = batch.expand_jobs(batch.load_spec(os.path.join(ROOT, "configs", "batches", "dev_ladder_example.yaml")))
    assert len(jobs) == 6 * 10 and len({j.job_id for j in jobs}) == 60
    assert all(j.split == "dev" and j.wrapper == "rgbd" and len(j.instances) == 1 for j in jobs)
    assert jobs[0].job_id == "L0_direct-turning_on_radio-pt10-rgbd-s0" and jobs[0].direct
    assert sum(j.direct for j in jobs) == 10
    assert {os.path.basename(j.config) for j in jobs} == {"L0_exec_only.yaml", "L1_plan.yaml", "L2_plan_check.yaml",
                                                         "L3_goal.yaml", "L4_recover.yaml"}


def test_expansion_cross_product_and_ids():
    spec = spec_of(seeds=[0, 1], instances=[10, 11], jobs=[
        {"name": "lad", "configs": ["configs/L0_exec_only.yaml", "configs/L1_plan.yaml"], "tasks": ["t1", "t2"],
         "wrappers": ["default", "rgbd"]},
        {"config": L0, "task": "t1", "instances": list(range(10, 20)), "group_instances": True, "mode": "public_test"},
        {"name": "tr", "config": L0, "task": "t1", "mode": "train", "instances": [3, 1234], "group_instances": True},
    ])
    jobs = batch.expand_jobs(spec)
    assert len(jobs) == 2 * 2 * 2 * 2 * 2 + 2 + 2
    ids = [j.job_id for j in jobs]
    assert ids[0] == "lad.L0_exec_only-t1-pt10-default-s0" and "lad.L1_plan-t2-pt11-rgbd-s1" in ids
    assert "L0_exec_only-t1-pt10to19-rgbd-s0" in ids  # default wrapper rgbd, consecutive group label
    assert "tr-t1-tr3_1234-rgbd-s1" in ids
    grouped = next(j for j in jobs if j.job_id.startswith("L0_exec_only-t1-pt10to19"))
    assert grouped.instances == tuple(range(10, 20)) and grouped.split == "dev"
    assert {j.split for j in jobs if j.job_id.startswith("tr-")} == {"dev"}
    assert batch.split_of("public_test", [0, 9]) == "report" and batch.split_of("public_test", [9, 10]) == "mixed"
    assert batch.split_of("hidden_test", [0]) == "hidden"
    assert batch.expand_jobs(spec) == jobs  # deterministic


@pytest.mark.parametrize("change, match", [
    (dict(jobs=[{"config": L0, "task": "t", "colour": "red"}]), "unknown keys"),
    (dict(jobs=[{"config": L0, "task": "t", "launch": {"gpus": "0,1"}}]), "set by eval.batch"),
    (dict(jobs=[{"config": L0, "task": "t", "launch": {"no_such_option": 1}}]), "invalid eval.launch options"),
    (dict(jobs=[{"config": L0, "task": "t"}, {"config": L0, "task": "t"}]), "duplicate job ids"),
    (dict(instances=[25]), "0..19"),
    (dict(jobs=[{"config": "configs/missing.yaml", "task": "t"}]), "not found"),
    (dict(jobs=[{"config": L0, "task": "t", "wrapper": "fancy"}]), "wrapper"),
    (dict(jobs=[{"config": L0, "task": "t", "seed": 0, "seeds": [1]}]), "either"),
])
def test_expansion_rejects(change, match):
    with pytest.raises(batch.SpecError, match=match):
        batch.expand_jobs(spec_of(**change))


def test_report_instances_need_explicit_opt_in():
    with pytest.raises(batch.SpecError, match="allow_report"):
        batch.expand_jobs(spec_of(instances=[0, 1]))
    with pytest.raises(batch.SpecError, match="allow_report"):
        batch.expand_jobs(spec_of(instances=[9, 10], group_instances=True))  # mixed Report + Dev
    jobs = batch.expand_jobs(spec_of(instances=[0], allow_report=True))
    assert [j.split for j in jobs] == ["report"]


def test_load_spec_validation(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"batch": "x", "jobs": [{"config": L0, "task": "t"}], "typo_key": 1}))
    with pytest.raises(batch.SpecError, match="unknown top-level"):
        batch.load_spec(str(p))
    p.write_text(yaml.safe_dump({"batch": "bad name", "jobs": [{"config": L0, "task": "t"}]}))
    with pytest.raises(batch.SpecError, match="batch"):
        batch.load_spec(str(p))


def test_launch_argv_and_flags():
    spec = spec_of(launch={"write_video": True, "max_steps": 500, "strict_exit": False, "share_gpu": True,
                           "evaluator_cmd_prefix": "bash /nas/baiqiao/b1k_runs/helper.sh", "robot_config": None},
                   jobs=[{"name": "d", "config": L0, "task": "t", "direct": True, "overrides": ["loop.exec_horizon=1"]}])
    job = batch.expand_jobs(spec)[0]
    argv = batch.launch_argv(job, run_id="b-d-a0", run_dir="/r/b-d-a0", ports=batch.slot_ports({"executor": 8000,
                             "agent": 8100, "stride": 10}, 2), gpus=[3, 1], tags={"batch": "b", "attempt": 0})
    args = launch.parse_launch_args(argv)
    assert args.run_id == "b-d-a0" and args.run_dir == "/r/b-d-a0" and args.instances == [10] and args.seed == 0
    assert (args.executor_port, args.agent_port) == (8020, 8120) and args.gpus == "3,1" and args.skip_gpu_check
    assert args.direct and args.write_video and args.max_steps == 500 and args.strict_exit is False and args.share_gpu
    assert args.evaluator_cmd_prefix == "bash /nas/baiqiao/b1k_runs/helper.sh" and args.robot_config is None
    assert args.override == ["loop.exec_horizon=1"] and args.tag == ["batch=b", "attempt=0"]
    assert batch.launch_flags({"write_video": False, "strict_exit": True}) == ["--strict-exit"]


# ---------------------------------------------------------------------- retry policy


@pytest.mark.parametrize("attempts, resume, allowed", [
    ([], False, True),
    ([DONE], True, False),                                   # success or policy failure: never re-run
    ([dict(DONE, results=[result(q=0.0, failure_reason="group_time_budget_exceeded")])], True, False),
    ([INFRA], False, True),
    ([INFRA, INFRA], False, True),
    ([INFRA, INFRA, INFRA], True, False),                    # at most 2 re-runs
    ([INFRA, DONE], True, False),
    ([EXIT_ONLY], True, False),                              # official results exist: a person decides
    ([dict(EXIT_ONLY, infra_kinds=["launcher_crash"])], True, False),
    ([dict(EXIT_ONLY, infra_kinds=["exit_code", "server_died"])], True, True),  # tainted results: re-run
    ([dict(EXIT_ONLY, results_complete=False)], False, True),
    ([FAILED], False, False),                                # launch-level failure: not in the same invocation
    ([FAILED], True, True),                                  # ... but on resume when the evaluator never started
    ([dict(FAILED, evaluator_started=True, results=[result()])], True, False),
    ([dict(FAILED, evaluator_started=True, results=[result(present=False)])], True, True),
])
def test_next_attempt_allowed(attempts, resume, allowed):
    assert batch.next_attempt_allowed(attempts, 2, resume=resume) is allowed


def test_batch_runner_retry_policy(tmp_path):
    spec = spec_of(instances=[10, 11, 12, 13, 14])
    names = [f"L0-t-pt{i}-rgbd-s0" for i in (10, 11, 12, 13, 14)]
    runner = ScriptedRunner({names[0]: [INFRA, INFRA, INFRA], names[1]: [INFRA, DONE], names[2]: [DONE],
                             names[3]: [FAILED], names[4]: [EXIT_ONLY]})
    br, jobs = make_runner(spec, tmp_path, runner)
    assert br.run() == 1  # a failed job remains
    runs = [c["run_id"] for c in runner.calls]
    assert runs == ["b-L0-t-pt10-rgbd-s0-a0", "b-L0-t-pt10-rgbd-s0-a1", "b-L0-t-pt10-rgbd-s0-a2",
                    "b-L0-t-pt11-rgbd-s0-a0", "b-L0-t-pt11-rgbd-s0-a1", "b-L0-t-pt12-rgbd-s0-a0",
                    "b-L0-t-pt13-rgbd-s0-a0", "b-L0-t-pt14-rgbd-s0-a0"]
    assert all(c["run_dir"] == str(tmp_path / "runs" / c["run_id"]) for c in runner.calls)
    led = ledger(tmp_path)
    assert [r["event"] for r in led] == ["attempt_start", "attempt_end"] * 8
    assert all(r["command"].startswith("python -m eval.launch --config") for r in led if r["event"] == "attempt_start")
    ends = [r for r in led if r["event"] == "attempt_end"]
    assert [r["state"] for r in ends] == ["infra_error"] * 3 + ["infra_error", "done", "done", "failed", "infra_error"]
    st = bstatus(tmp_path)
    by = {j["job_id"]: j for j in st["jobs"]}
    assert st["state"] == "finished" and st["counts"] == {"infra_error": 2, "done": 2, "failed": 1}
    assert by[names[0]]["attempts"] == 3 and "2 infra re-runs used" in by[names[0]]["rerun"]
    assert by[names[1]]["attempts"] == 2 and by[names[1]]["state"] == "done" and by[names[1]]["rerun"] is None
    assert by[names[4]]["attempts"] == 1 and "every official result exists" in by[names[4]]["rerun"]
    assert by[names[2]]["results"][0]["q_score"] == 0.0  # policy failure kept, not re-run


def test_batch_resume_from_ledger(tmp_path):
    spec = spec_of(instances=[10, 11, 12])
    names = [f"L0-t-pt{i}-rgbd-s0" for i in (10, 11, 12)]
    first = ScriptedRunner({names[0]: [DONE], names[1]: [FAILED], names[2]: [INFRA]})
    br, _ = make_runner(spec, tmp_path, first)
    br.stop_event.set()  # nothing runs; then simulate a hard-killed batch by hand
    br.run()
    lg = batch.Ledger(str(tmp_path / "batch" / "jobs.jsonl"))
    lg.append({"event": "attempt_start", "job_id": names[0], "attempt": 0, "run_id": "b-x-a0",
               "run_dir": str(tmp_path / "runs" / "done-run")})
    lg.append({"event": "attempt_end", "job_id": names[0], "attempt": 0, "run_id": "b-x-a0",
               **batch.summarize_status(DONE)})
    lg.append({"event": "attempt_start", "job_id": names[1], "attempt": 0, "run_id": "b-y-a0",
               "run_dir": str(tmp_path / "runs" / "y")})
    lg.append({"event": "attempt_end", "job_id": names[1], "attempt": 0, "run_id": "b-y-a0",
               **batch.summarize_status(FAILED)})
    killed = tmp_path / "runs" / "z"
    killed.mkdir(parents=True)
    (killed / "status.json").write_text(json.dumps({"state": "running", "processes": {"evaluator": {}},
                                                    "supervisor": {"pid": 2 ** 22 + 7, "start_ticks": 1}}))
    lg.append({"event": "attempt_start", "job_id": names[2], "attempt": 0, "run_id": "b-z-a0", "run_dir": str(killed)})

    second = ScriptedRunner({names[1]: [DONE], names[2]: [DONE]})
    br2, _ = make_runner(spec, tmp_path, second)
    assert br2.run() == 0
    assert [c["run_id"] for c in second.calls] == ["b-L0-t-pt11-rgbd-s0-a1", "b-L0-t-pt12-rgbd-s0-a1"]
    recovered = [r for r in ledger(tmp_path) if r.get("recovered")]
    assert len(recovered) == 1 and recovered[0]["state"] == "infra_error"
    assert recovered[0]["infra_kinds"] == ["batch_interrupted"] and recovered[0]["evaluator_started"] is True
    assert bstatus(tmp_path)["counts"] == {"done": 3}


def test_batch_parallel_distinct_gpus_and_ports(tmp_path):
    spec = spec_of(instances=[10, 11, 12, 13])
    runner = ScriptedRunner({f"L0-t-pt{i}-rgbd-s0": [DONE] for i in (10, 11, 12, 13)}, delay=0.2)
    br, _ = make_runner(spec, tmp_path, runner, max_parallel=2)
    assert br.run() == 0
    assert runner.max_active == 2
    gpus_used = {tuple(c["argv"][c["argv"].index("--gpus") + 1].split(",")) for c in runner.calls}
    assert gpus_used <= {("0", "1"), ("2", "3")}
    ports = {(c["argv"][c["argv"].index("--executor-port") + 1], c["argv"][c["argv"].index("--agent-port") + 1])
             for c in runner.calls}
    assert ports == {("8000", "8100"), ("8010", "8110")}


def test_batch_no_free_gpu_aborts_without_waiting(tmp_path):
    sleeps = []

    def none_free(n, exclude):
        raise gpu.NoFreeGpuError("need 2 free GPU(s)")

    runner = ScriptedRunner({})
    br, _ = make_runner(spec_of(), tmp_path, runner, gpu_picker=none_free)
    br.sleep = sleeps.append
    assert br.run() == 1
    assert runner.calls == [] and sleeps == []
    st = bstatus(tmp_path)
    assert st["state"] == "aborted" and "no free GPU" in st["abort_reason"] and st["counts"] == {"pending": 1}


def test_batch_gpu_recheck_after_own_attempt_and_wait_mode(tmp_path):
    picks = iter([[0, 1], gpu.NoFreeGpuError("busy"), gpu.NoFreeGpuError("busy")])

    def picker(n, exclude):
        p = next(picks)
        if isinstance(p, Exception):
            raise p
        return p

    spec = spec_of(instances=[10, 11])
    runner = ScriptedRunner({"L0-t-pt10-rgbd-s0": [DONE]})
    br, _ = make_runner(spec, tmp_path, runner, gpu_picker=picker, settle_s=7)
    sleeps = []
    br.sleep = sleeps.append
    assert br.run() == 1 and sleeps == [7]  # one re-check after our own attempt freed its GPUs, then stop
    assert bstatus(tmp_path)["state"] == "aborted"

    picks2 = iter([gpu.NoFreeGpuError("busy"), gpu.NoFreeGpuError("busy"), [2, 3]])

    def picker2(n, exclude):
        p = next(picks2)
        if isinstance(p, Exception):
            raise p
        return p

    runner2 = ScriptedRunner({"L0-t-pt10-rgbd-s0": [DONE]})
    br2, _ = make_runner(spec_of(), tmp_path / "w", runner2, gpu_picker=picker2, wait_gpu=True, poll_s=60)
    sleeps2 = []
    br2.sleep = sleeps2.append
    assert br2.run() == 0 and sleeps2 == [60, 60]  # --wait-gpu: poll until free


def test_subprocess_runner_reports_launch_failure_without_status(tmp_path):
    job = batch.expand_jobs(spec_of())[0]
    runner = batch.SubprocessRunner(str(tmp_path / "out"), python=sys.executable)
    st = runner(job, ["--config", L0], str(tmp_path / "never"), "b-L0-a0")  # eval.launch rejects: no --task
    assert st["state"] == "failed" and st["evaluator_started"] is False and "without status.json" in st["reason"][0]
    assert batch.next_attempt_allowed([batch.summarize_status(st)], 2, resume=True)


def test_summarize_status_keeps_official_view():
    st = {"state": "done", "reason": [], "warnings": [str(i) for i in range(9)], "evaluator_started": True,
          "evaluator_exit_code": 0, "max_steps": 1500, "config_hash": "abc", "infra_kinds": [], "results_complete": True,
          "results": [{"file": "f.json", "present": True, "success": True, "q_score": 1.0, "steps": 40,
                       "failure_reason": None, "instance_id": 301}], "processes": {}}
    out = batch.summarize_status(st)
    assert out["warnings"] == ["4", "5", "6", "7", "8"] and out["results"] == [
        {"file": "f.json", "present": True, "success": True, "q_score": 1.0, "steps": 40, "failure_reason": None}]


def test_dry_run_cli(capsys):
    rc = batch.main([os.path.join(ROOT, "configs", "batches", "step3_turning_on_radio.yaml"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "## L0-turning_on_radio-pt0-default-s0  [report]" in out
    assert "## L0-turning_on_radio-pt0-rgbd-s0  [report]" in out
    assert out.count("python -m eval.launch --config") == 2 and "--write-video" in out


FAKE_DYING_LAUNCH = r'''
import json, os, subprocess, sys, time
run_dir = sys.argv[sys.argv.index("--run-dir") + 1]
os.makedirs(run_dir, exist_ok=True)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
time.sleep(0.2)
start = int(open("/proc/%d/stat" % child.pid).read().rsplit(")", 1)[1].split()[19])
with open(os.path.join(run_dir, "child.pid"), "w") as f:
    f.write(str(child.pid))
with open(os.path.join(run_dir, "status.json"), "w") as f:
    json.dump({"state": "running", "supervisor": {"pid": os.getpid(), "start_ticks": 0},
               "processes": {"executor": {"pid": child.pid, "pgid": child.pid, "start_ticks": start,
                                          "started_by_us": True}}}, f)
os._exit(1)  # the launcher dies, its executor group is left behind
'''


def test_subprocess_runner_cleans_up_after_a_dead_launcher(tmp_path):
    script = tmp_path / "fake_launch.py"
    script.write_text(FAKE_DYING_LAUNCH)
    python = tmp_path / "fake_python"
    python.write_text(f'#!/bin/sh\nshift 2\nexec "{sys.executable}" "{script}" "$@"\n')  # drop "-m eval.launch"
    python.chmod(0o755)
    job = batch.expand_jobs(spec_of())[0]
    run_dir = tmp_path / "run"
    st = batch.SubprocessRunner(str(tmp_path / "out"), python=str(python))(
        job, ["--run-dir", str(run_dir)], str(run_dir), "b-L0-a0")
    assert st["state"] == "infra_error" and st["infra_kinds"] == ["launcher_crash"]
    pid = int((run_dir / "child.pid").read_text())
    stat = launch.proc_stat(pid)
    assert stat is None or stat["state"] == "Z"  # the orphaned group was stopped (it was ours)
    assert launch.read_status(str(run_dir))["processes"]["executor"]["stop_actions"][0] == "SIGINT: sent"


def test_interrupted_attempt_with_complete_official_results_is_not_rerun(tmp_path):
    """The batch was killed after the evaluator wrote its result: that official record stands."""
    run_dir = tmp_path / "runs" / "b-L0-t-pt10-rgbd-s0-a0"
    (run_dir / "eval_output" / "json").mkdir(parents=True)
    (run_dir / "eval_output" / "json" / "t_311_0.json").write_text(json.dumps(
        {"task": "t", "instance_id": 311, "rollout_id": 0, "steps": 90, "success": False, "q_score": {"final": 0.5}}))
    (run_dir / "status.json").write_text(json.dumps({
        "state": "running", "processes": {"evaluator": {}}, "supervisor": {"pid": 2 ** 22 + 7, "start_ticks": 1},
        "expected_results": ["t_311_0.json"], "eval_output_dir": str(run_dir / "eval_output")}))
    os.makedirs(tmp_path / "batch")
    batch.Ledger(str(tmp_path / "batch" / "jobs.jsonl")).append(
        {"event": "attempt_start", "job_id": "L0-t-pt10-rgbd-s0", "attempt": 0, "run_id": run_dir.name,
         "run_dir": str(run_dir)})
    runner = ScriptedRunner({})
    br, _ = make_runner(spec_of(), tmp_path, runner)
    assert br.run() == 2 and runner.calls == []
    job = bstatus(tmp_path)["jobs"][0]
    assert job["state"] == "infra_error" and job["results"][0]["q_score"] == 0.5
    assert "every official result exists" in job["rerun"]
