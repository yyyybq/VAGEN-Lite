"""CPU tests of the watcher rules of the user's decisions of 2026-10-23 evening (downstream/eval/watch.py): E1's
150k / 175k weights copied out of full states and evaluated, every new 8-card job only on idle machines and never
while the main line waits, released runs resumed, the two-table matrix."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from test_watch_and_tasks import OLD, FakeJobs, FakeRuns, eight_card, watch, watcher


def e1_exp(root: Path, start="B", states=(150000,), step=152000, ratio="100"):
    """A segmented run with complete full states RUN/full_state/steps_<N> (downstream_state.json) and metrics."""
    exp = root / "experiments" / f"DS-{start}-{ratio}-s1"
    (exp / "run").mkdir(parents=True)
    (exp / "experiment.json").write_text(json.dumps({"steps": 200000, "segmented": True, "eval_points": [200000],
                                                     "repeat_final": start == "B"}))
    for s in states:
        (exp / f"run/full_state/steps_{s}").mkdir(parents=True)
        (exp / f"run/full_state/steps_{s}/downstream_state.json").write_text(json.dumps({"completed_steps": s}))
    (exp / "run/metrics.jsonl").write_text(json.dumps({"step": step, "action_dit_loss": 0.05}) + "\n")
    return exp


def fake_extract(calls):
    """Stands in for downstream/train/extract_weights.py: writes the checkpoint (settled) and source.json."""
    def run(cmd, **kw):
        calls.append(cmd)
        dest, step = Path(cmd[-1]), int(Path(cmd[-3]).name.split("_")[1])
        ckpt = dest / "checkpoints" / f"steps_{step}_pytorch_model.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"w")
        os.utime(ckpt, (OLD, OLD))
        (dest / "source.json").write_text(json.dumps({"step": step}))
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")
    return run


def test_e1_extra_points_are_copied_out_of_full_states_then_evaluated_on_an_idle_machine(tmp_path):
    exp = e1_exp(tmp_path)
    calls, jobs = [], FakeJobs()
    w = watcher(tmp_path, "--exps", "DS-B-100-s1", "--pool-nodes", "h800:1", jobs=jobs)
    w.extract_run = fake_extract(calls)
    rows, *_ = w.tick()
    assert len(calls) == 1 and calls[0][0] == w.a.extract_python and calls[0][1].endswith("train/extract_weights.py")
    assert calls[0][2:] == [str(exp / "run/full_state/steps_150000"), str(exp / "run"),
                            str(exp / "evals_src/steps_150000")]
    states = {e["tag"]: e["state"] for e in rows[0]["evals"]}
    assert states["steps_175000"] == "等第 175k 步的全状态，训练到第 152000 步"
    assert [(p[1], p[2], p[3][-3:]) for p in jobs.prepared] == [
        (8, "ev8-ds-b-100-s1-steps150000", [str(exp / "evals/steps_150000/out"), "8", "96"])]
    assert jobs.prepared[0][3][2] == str(exp / "evals_src/steps_150000/checkpoints/steps_150000_pytorch_model.pt")
    assert any("权重已从全状态复制到 evals_src/steps_150000/" in n for n in w.notes)
    w2 = watcher(tmp_path, "--exps", "DS-B-100-s1", "--pool-nodes", "h800:1", jobs=jobs)
    w2.extract_run = fake_extract(calls)
    w2.tick()
    assert len(calls) == 1 and len(jobs.prepared) == 1  # copied once, submitted once


def test_a_full_state_replaced_before_its_copy_is_reported(tmp_path):
    e1_exp(tmp_path, states=(175000,), step=176000)
    calls = []
    w = watcher(tmp_path, "--exps", "DS-B-100-s1", "--pool-nodes", "h800:0")
    w.extract_run = fake_extract(calls)
    rows, *_ = w.tick()
    states = {e["tag"]: e["state"] for e in rows[0]["evals"]}
    assert states["steps_150000"] == "全状态已被替换，没复制到" and len(calls) == 1  # 175k itself is copied
    assert any("第 150k 步的全状态已被第 175k 步的替换" in a for a in w.alerts)


def test_failed_copies_are_retried_and_reported(tmp_path):
    e1_exp(tmp_path)
    w = watcher(tmp_path, "--exps", "DS-B-100-s1", "--pool-nodes", "h800:0")
    w.extract_run = lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="changed during the copy")
    rows, *_ = w.tick()
    assert {e["tag"]: e["state"] for e in rows[0]["evals"]}["steps_150000"] == "权重复制失败，下一轮重试"
    assert any("权重复制失败（退出码 1）" in a and "changed during the copy" in a for a in w.alerts)


def test_main_line_queue_or_no_idle_machine_holds_every_new_eight_card_job(tmp_path):
    e1_exp(tmp_path)
    e2 = e1_exp(tmp_path, ratio="010", states=(75000,), step=77100)
    (e2 / "job_released_20261023T191729Z").mkdir()
    qrun = tmp_path / "RL-SPEED-64"
    jobs = FakeJobs()
    jobs.prepare(qrun, 8, "h800", "downstream-rl-speed64", ["bash", "speed_test.sh"])
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue/a_rl_speed64.json").write_text(json.dumps({"run": str(qrun), "cards": 8,
                                                                  "what": "RL 64 env speed test"}))
    run_tools = FakeRuns()
    args = ("--exps", "DS-B-100-s1", "DS-B-010-s1", "--queue-dir", str(tmp_path / "queue"),
            "--max-whole-per-pass", "3")
    jobs.listing = [eight_card("interiorgs-v10-eval-val--dcks2-100", "STARTING", cards=1)]  # the main line waits
    w = watcher(tmp_path, *args, "--pool-nodes", "h800:5", jobs=jobs, run_tools=run_tools)
    w.extract_run = fake_extract([])
    w.tick()
    assert len(jobs.prepared) == 1 and not run_tools.submitted and not (qrun / "job/submission.json").exists()
    assert any("给排在前面的任务让路" in n for n in w.notes)
    jobs.listing = []  # the main line is on machines, but no machine is idle
    w = watcher(tmp_path, *args, "--pool-nodes", "h800:0", jobs=jobs, run_tools=run_tools)
    w.tick()
    assert len(jobs.prepared) == 1 and not run_tools.submitted and any("没有空闲机器" in n for n in w.notes)
    w = watcher(tmp_path, *args, "--pool-nodes", "h800:2", jobs=jobs, run_tools=run_tools)
    w.tick()  # two idle machines: the E1 evaluation, then the queued test; the E2 run waits for the next pass
    assert [p[2] for p in jobs.prepared[1:]] == ["ev8-ds-b-100-s1-steps150000"]
    assert (qrun / "job/submission.json").exists() and not run_tools.submitted
    jobs.listing = [eight_card("downstream-ev8-ds-b-100-s1-steps150000", "STARTING")]
    w = watcher(tmp_path, *args, "--pool-nodes", "h800:5", jobs=jobs, run_tools=run_tools)
    w.tick()  # one of ours waits in h800: the idle estimate was wrong, nothing more
    assert not run_tools.submitted and any("我们的 8 卡任务在排队" in n for n in w.notes)
    jobs.listing = []
    w = watcher(tmp_path, *args, "--pool-nodes", "h800:5", jobs=jobs, run_tools=run_tools)
    w.tick()
    assert run_tools.submitted == [("B", "010", 1)] and run_tools.pools == ["h800"]


def test_released_runs_resume_only_until_they_are_finished(tmp_path):
    e2 = e1_exp(tmp_path, ratio="010", states=(200000,), step=200000)
    (e2 / "job_released_20261023T191729Z").mkdir()
    (e2 / "run/checkpoints").mkdir()
    (e2 / "run/checkpoints/steps_200000_pytorch_model.pt").write_bytes(b"x")  # finished meanwhile
    run_tools = FakeRuns()
    w = watcher(tmp_path, "--exps", "DS-B-010-s1", "--pool-nodes", "h800:3", run_tools=run_tools)
    assert w.run_finished(e2) and w.resume_released() == [] and not run_tools.submitted
    assert "**训完**" in watch.matrix(w).split("## 评测")[0]


def test_a_cached_finished_job_is_asked_again_once_its_run_has_a_new_job(tmp_path):
    exp = e1_exp(tmp_path, ratio="010", states=(75000,))
    (exp / "job").mkdir()
    (exp / "job/submission.json").write_text(json.dumps({"job_id": "pt-new"}))
    w = watcher(tmp_path, jobs=FakeJobs(states={exp.name: "RUNNING"}))
    w.state["finished"][str(exp)] = {"job_id": "pt-old", "state": "CANCELLED", "finished": True}
    assert w.job(exp)["state"] == "RUNNING"
    w.state["finished"][str(exp)] = {"job_id": "pt-new", "state": "SUCCEEDED", "finished": True}
    assert w.job(exp)["state"] == "SUCCEEDED"


def test_the_gate_never_sends_back_a_run_that_had_a_job(tmp_path):
    """2026-10-23 19:30: the stage-2 gate took the four E2 runs stopped to free machines (job_released_*) for runs
    never submitted and sent them back in; a run with any job history now resumes only through resume_released."""
    from test_watch_and_tasks import stage2_dirs, stopped_100k

    stopped_100k(tmp_path, "B", 3)
    stage2_dirs(tmp_path)
    for s in ("B", "S"):
        (tmp_path / f"experiments/DS-{s}-010-s1/job_released_20261023T191729Z").mkdir()
    run_tools = FakeRuns()
    w = watcher(tmp_path, "--pool-nodes", "h800:0", run_tools=run_tools)
    gate = w.gate()
    assert gate["passed"] and run_tools.submitted == [("RS", "010", 1), ("RD", "010", 1)]


def test_suspended_main_line_jobs_count_as_waiting(tmp_path):
    """The main line's queued one-card evaluations show as SUSPENDED in sco: we yield to them as to waiting jobs."""
    jobs = FakeJobs()
    jobs.listing = [eight_card("interiorgs-v10-eval-val--dcks2-100", "SUSPENDED", cards=1)]
    w = watcher(tmp_path, jobs=jobs)
    assert "给排在前面的任务让路" in w.blocked(8, "h800") and w.yield_waiting()
    jobs.listing = [eight_card("interiorgs-v10-eval-val--dcks2-100", "RUNNING", cards=1)]
    assert not watcher(tmp_path, jobs=jobs).blocked(8, "h800")


def test_e1_matrix_cells_show_each_point_and_the_mean_of_three(tmp_path):
    from test_watch_and_tasks import summary

    exp = e1_exp(tmp_path, states=(200000,), step=200000)
    (exp / "evals_src/steps_175000").mkdir(parents=True)
    (exp / "evals_src/steps_175000/source.json").write_text("{}")
    summary(exp / "evals/steps_150000/out/episodes", {"OpenDrawer": {"n": 50, "successes": 10},
                                                       "CloseFridge": {"n": 20, "successes": 4}})
    w = watcher(tmp_path, "--exps", "DS-B-100-s1")
    cell = watch.matrix_eval(w, "B", "100", w.a.extra_points)
    assert "**1/3 个检查点评完**" in cell and "150k：总体 20.0%（14/70），空间组 20.0%（10/50）" in cell
    assert "175k：权重已复制，等空闲机器评测" in cell and "200k：训完后由训练任务自己评" in cell
    for step, k in ((175000, 12), (200000, 16)):
        summary(exp / f"evals/steps_{step}/out/episodes", {"OpenDrawer": {"n": 50, "successes": k},
                                                            "CloseFridge": {"n": 20, "successes": 6}})
    cell = watch.matrix_eval(w, "B", "100", w.a.extra_points)
    # (14/70 + 18/70 + 22/70) / 3 = 25.7%; spatial (10/50 + 12/50 + 16/50) / 3 = 25.3%
    assert "**3/3 个检查点评完**" in cell and "**3 点平均**：总体 25.7%，空间组 25.3%" in cell
