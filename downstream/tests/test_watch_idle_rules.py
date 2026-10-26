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


def test_only_queued_main_line_jobs_hold_us_back(tmp_path):
    """SUSPENDED in the job list marks stopped jobs (the main line has some from 10-16 that never run again): they
    must not hold us back for ever; queued ones (STARTING, QUEUEING, ...) do."""
    jobs = FakeJobs()
    jobs.listing = [eight_card("interiorgs-v10-rl-rl-run-rd", "SUSPENDED")]
    assert not watcher(tmp_path, jobs=jobs).blocked(8, "h800")
    jobs.listing.append(eight_card("interiorgs-v10-eval-val--dcks2-100", "QUEUEING", cards=1))
    w = watcher(tmp_path, jobs=jobs)
    assert "给排在前面的任务让路" in w.blocked(8, "h800") and w.yield_waiting() == ["interiorgs-v10-eval-val--dcks2-100"]


class SubmittingRuns(FakeRuns):
    """FakeRuns whose submit leaves EXP/job/submission.json, as runs.submit does."""
    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs

    def submit(self, start, ratio, seed, root, pool=None):
        out = super().submit(start, ratio, seed, root, pool)
        exp = Path(root) / "experiments" / out["exp"]
        self.jobs.prepare(exp, 8, pool, exp.name.lower(), ["bash", "run_segmented.sh", str(exp)])
        out["job"] = self.jobs.submit(exp)
        return out


def test_jobs_sent_on_idle_machines_are_withdrawn_while_they_wait_and_the_main_line_queues(tmp_path):
    e2 = e1_exp(tmp_path, ratio="010", states=(75000,), step=77100)
    (e2 / "job_released_20261023T191729Z").mkdir()
    qrun = tmp_path / "RL-SPEED-64"
    jobs = FakeJobs(states={"DS-B-010-s1": "QUEUEING", "RL-SPEED-64": "STARTING"})
    jobs.prepare(qrun, 8, "h800", "downstream-rl-speed64", ["bash", "speed_test.sh"])
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue/a.json").write_text(json.dumps({"run": str(qrun), "cards": 8, "what": "RL test"}))
    run_tools = SubmittingRuns(jobs)
    args = ("--exps", "DS-B-010-s1", "--queue-dir", str(tmp_path / "queue"), "--max-whole-per-pass", "2",
            "--pool-nodes", "h800:3")
    w = watcher(tmp_path, *args, jobs=jobs, run_tools=run_tools)
    w.tick()  # idle machines, nothing of the main line queued: both go
    w.save()
    assert (qrun / "job/submission.json").exists() and (e2 / "job/submission.json").exists()
    assert set(w.state["idle_sent"]) == {"a", "DS-B-010-s1"}
    jobs.listing = [eight_card("interiorgs-v10-eval-x", "QUEUEING", cards=1)]  # the main line queues; ours still wait
    w = watcher(tmp_path, *args, jobs=jobs, run_tools=run_tools)
    w.tick()
    w.save()
    assert jobs.deleted == ["DS-B-010-s1", "RL-SPEED-64"]
    assert not (e2 / "job").exists() and len(list(e2.glob("job_released_*"))) == 2
    assert (qrun / "job/prepared.json").exists() and list((qrun / "job").glob("submission_withdrawn_*.json"))
    assert len(run_tools.submitted) == 1 and any("撤下我们按空闲机器提交、还在排队的 2 个" in n for n in w.notes)
    jobs.listing = []  # the main line is on machines again: both go back in
    w = watcher(tmp_path, *args, jobs=jobs, run_tools=run_tools)
    w.tick()
    assert (qrun / "job/submission.json").exists() and len(run_tools.submitted) == 2
    jobs.states["DS-B-010-s1"] = "RUNNING"  # a started job stays, whatever queues behind it
    jobs.listing = [eight_card("interiorgs-v10-eval-x", "QUEUEING", cards=1)]
    w = watcher(tmp_path, *args, jobs=jobs, run_tools=run_tools)
    w.tick()
    assert (e2 / "job/submission.json").exists() and jobs.deleted.count("DS-B-010-s1") == 1


def test_runs_listed_for_release_stop_at_their_next_full_state_and_wait_before_resuming(tmp_path):
    """The user's choice of 2026-10-23 21:3x: three E2 30% runs give their machines back to the main line once they
    have saved their next full state; afterwards they resume like any released run, but not within the hold."""
    import time

    exp = e1_exp(tmp_path, ratio="030", states=(), step=3000)
    (exp / "job").mkdir()
    (exp / "job/submission.json").write_text(json.dumps({"job_id": "pt-run"}))
    (exp / "job/prepared.json").write_text(json.dumps({"pool": "h800"}))
    jobs = FakeJobs(states={exp.name: "RUNNING"})
    run_tools = SubmittingRuns(jobs)
    args = ("--exps", "DS-B-030-s1", "--release-at-full-state", "DS-B-030-s1", "--resume-ratios", "030",
            "--pool-nodes", "h800:5")

    def one_pass(at=None):
        w = watcher(tmp_path, *args, jobs=jobs, run_tools=run_tools)
        if at:
            w.now = lambda: at
        w.tick()
        w.save()
        return w

    w = one_pass()
    assert not jobs.deleted and w.state["release"]["DS-B-030-s1"]["after"] == 0
    assert "第 25,000 步的全状态存好后停下（机器让给主线）" in watch.matrix(w)
    (exp / "run/full_state/steps_25000.partial").mkdir(parents=True)  # still being written
    one_pass()
    assert not jobs.deleted
    (exp / "run/full_state/steps_25000.partial").rename(exp / "run/full_state/steps_25000")
    w = one_pass()
    assert jobs.deleted == ["DS-B-030-s1"] and list(exp.glob("job_released_*")) and not (exp / "job").exists()
    assert w.state["release"]["DS-B-030-s1"]["full_state"] == 25000 and not run_tools.submitted  # held
    assert any("按要求在全状态检查点停下" in n for n in w.notes)
    one_pass(at=time.time() + 31 * 60)  # after the hold: back in, from its full state, on an idle machine
    assert run_tools.submitted == [("B", "030", 1)] and jobs.deleted == ["DS-B-030-s1"]


def test_a_final_checkpoint_its_ended_job_did_not_evaluate_goes_to_an_idle_machine(tmp_path):
    """2026-10-25 03:47: E1 B saved its final checkpoint, then NCCL failed while shutting down, training exited 1 and
    run_segmented.sh stopped before evaluating (sco still said SUCCEEDED).  While the job lives the board waits for
    the job's own evaluation; once it has ended, the checkpoint goes to one 8-card job on an idle machine."""
    from test_watch_and_tasks import summary

    exp = e1_exp(tmp_path, ratio="010", states=(175000,), step=200000)
    final = exp / "run/final_model/pytorch_model.pt"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"w")
    os.utime(final, (OLD, OLD))
    (exp / "job").mkdir()
    (exp / "job/submission.json").write_text(json.dumps({"job_id": "pt-train"}))
    jobs = FakeJobs(states={exp.name: "RUNNING"})  # evaluating inside the job
    args = ("--exps", "DS-B-010-s1", "--pool-nodes", "h800:1")
    w = watcher(tmp_path, *args, jobs=jobs)
    rows, *_ = w.tick()
    assert [e["state"] for e in rows[0]["evals"]] == ["由训练任务自己评（8 个策略服务 × 12 个仿真进程）"]
    assert not jobs.prepared and "200k：训完后由训练任务自己评" in watch.matrix_eval(w, "B", "010")
    jobs.states[exp.name] = "SUCCEEDED"  # ended without the evaluation
    w = watcher(tmp_path, *args, jobs=jobs)
    assert "200k：训练任务没评就结束了，等空闲机器评测（8 卡）" in watch.matrix_eval(w, "B", "010")
    rows, *_ = w.tick()
    assert rows[0]["evals"][0]["state"] == "训练任务没评就结束了；待提交（8 卡，等空闲机器）"
    assert [(p[2], p[3][-4:]) for p in jobs.prepared] == [
        ("ev8-ds-b-010-s1-steps200000", [str(final), str(exp / "evals/steps_200000/out"), "8", "96"])]
    w = watcher(tmp_path, *args, jobs=jobs)
    assert "200k：评测 `pt-steps_200000` RUNNING" in watch.matrix_eval(w, "B", "010")
    summary(exp / "evals/steps_200000/out/episodes", {"OpenDrawer": {"n": 50, "successes": 9}})
    rows, *_ = w.tick()
    assert rows[0]["evals"][0]["state"] == "8 卡评测完成（训练任务没评）" and len(jobs.prepared) == 1


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


def test_a_final_checkpoint_left_unevaluated_goes_before_the_extra_points(tmp_path):
    exp = e1_exp(tmp_path, states=(175000,), step=200000)
    for s in (150000, 175000):  # both copies made, both waiting for an idle machine
        ck = exp / f"evals_src/steps_{s}/checkpoints/steps_{s}_pytorch_model.pt"
        ck.parent.mkdir(parents=True)
        ck.write_bytes(b"w")
        os.utime(ck, (OLD, OLD))
        (exp / f"evals_src/steps_{s}/source.json").write_text("{}")
    final = exp / "run/final_model/pytorch_model.pt"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"w")
    os.utime(final, (OLD, OLD))
    (exp / "job").mkdir()
    (exp / "job/submission.json").write_text(json.dumps({"job_id": "pt-train"}))
    jobs = FakeJobs(states={exp.name: "SUCCEEDED"})
    watcher(tmp_path, "--exps", "DS-B-100-s1", "--pool-nodes", "h800:1", jobs=jobs).tick()
    assert [p[2] for p in jobs.prepared] == ["ev8-ds-b-100-s1-steps200000"]


def test_with_yield_pool_only_the_main_line_holds_back_only_its_own_pool(tmp_path):
    """The user's decision of 2026-10-25: a main-line job waiting in zoetrope no longer holds back an evaluation on
    an idle machine of h800; one waiting in h800 still does, and withdraws ours still waiting there."""
    e1_exp(tmp_path)
    jobs = FakeJobs()
    jobs.listing = [eight_card("interiorgs-v10-rl-rl-run-cb-v1-v8", "STARTING", pool="zoetrope")]
    args = ("--exps", "DS-B-100-s1", "--whole-pools", "h800", "zoetrope", "--pool-nodes", "h800:1", "zoetrope:1")
    w = watcher(tmp_path, *args, jobs=jobs)  # without the switch: a main-line job waiting anywhere holds all back
    w.extract_run = fake_extract([])
    w.tick()
    assert not jobs.prepared and any("zoetrope：8 卡评测本轮不提交" in n for n in w.notes)
    w = watcher(tmp_path, *args, "--yield-pool-only", jobs=jobs)
    w.tick()
    assert [(p[2], p[4]) for p in jobs.prepared] == [("ev8-ds-b-100-s1-steps150000", "h800")]
    assert any("zoetrope：8 卡评测本轮不提交：给排在前面的任务让路" in n for n in w.notes)
    assert not any(n.startswith("h800：8 卡评测本轮不提交") for n in w.notes)
    jobs.states["steps_150000"] = "STARTING"  # ours waits in h800 (the idle estimate was wrong)
    w = watcher(tmp_path, *args, "--yield-pool-only", jobs=jobs)
    w.tick()
    assert not jobs.deleted  # the main line waits only in zoetrope
    jobs.listing.append(eight_card("interiorgs-v10-eval-val--x", "QUEUEING", pool="h800", cards=1))
    w = watcher(tmp_path, *args, "--yield-pool-only", jobs=jobs)
    w.tick()
    assert jobs.deleted == ["steps_150000"] and len(jobs.prepared) == 1  # withdrawn, not sent again while it waits
    assert list((tmp_path / "experiments/DS-B-100-s1/evals/steps_150000").glob("job_withdrawn_*"))


def test_with_yield_pool_only_runs_sent_on_idle_machines_are_withdrawn_only_for_their_own_pool(tmp_path):
    e2 = e1_exp(tmp_path, ratio="010", states=(75000,), step=77100)
    (e2 / "job_released_20261023T191729Z").mkdir()
    jobs = FakeJobs(states={"DS-B-010-s1": "QUEUEING"})
    run_tools = SubmittingRuns(jobs)
    args = ("--exps", "DS-B-010-s1", "--pool-nodes", "h800:3", "zoetrope:0", "--yield-pool-only")
    jobs.listing = [eight_card("interiorgs-v10-rl-x", "STARTING", pool="zoetrope")]

    def one_pass():
        w = watcher(tmp_path, *args, jobs=jobs, run_tools=run_tools)
        w.tick()
        w.save()

    one_pass()
    assert run_tools.submitted == [("B", "010", 1)] and run_tools.pools == ["h800"]
    one_pass()
    assert not jobs.deleted  # it waits in h800, the main line only in zoetrope
    jobs.listing.append(eight_card("interiorgs-v10-eval-y", "QUEUEING", pool="h800", cards=1))
    one_pass()
    assert jobs.deleted == ["DS-B-010-s1"] and len(list(e2.glob("job_released_*"))) == 2


def test_the_queue_folder_follows_the_runs_root(tmp_path):
    """2026-10-26: the queue folder's fixed default made the tests of this file read the real queue and leave a fake
    submission in a real E3 run; it now lies under --runs-root unless given."""
    assert watcher(tmp_path).a.queue_dir == str(tmp_path / "queue")
    assert watcher(tmp_path, "--queue-dir", str(tmp_path / "q2")).a.queue_dir == str(tmp_path / "q2")
    assert watch.Watcher(watch.parser().parse_args(["--state", str(tmp_path / "s.json")]), jobs=FakeJobs()).a.queue_dir \
        == str(watch.RUNS_ROOT / "queue")


def e3_queue(tmp_path, jobs, starts=("B", "S", "RS", "RD")):
    """E3 runs prepared for h800 and queued as one group of at most 2, either pool (as for the real runs)."""
    (tmp_path / "queue").mkdir(exist_ok=True)
    runs_ = {}
    for i, s in enumerate(starts):
        run = tmp_path / f"experiments/E3-{s}"
        (run / "cfg").mkdir(parents=True)
        (run / "cfg/train.yaml").write_text("{}")
        jobs.prepare(run, 8, "h800", f"downstream-e3-{s.lower()}", ["bash", "run_e3.sh"])
        (tmp_path / f"queue/e3_{i}_{s}.json").write_text(json.dumps(
            {"run": str(run), "cards": 8, "what": f"E3 {s}", "group": "e3", "max_live": 2, "pools": ["h800", "zoetrope"]}))
        runs_[s] = run
    return runs_


def test_e3_runs_go_to_any_pool_with_an_idle_machine_at_most_two_at_once(tmp_path):
    jobs = FakeJobs()
    runs_ = e3_queue(tmp_path, jobs)
    args = ("--exps", "DS-B-100-s1", "--queue-dir", str(tmp_path / "queue"), "--max-whole-per-pass", "4",
            "--whole-pools", "h800", "zoetrope", "--pool-nodes", "h800:0", "zoetrope:3", "--yield-pool-only")
    w = watcher(tmp_path, *args, jobs=jobs)
    w.tick()
    sent = [s for s, run in runs_.items() if (run / "job/submission.json").exists()]
    assert sent == ["B", "S"]  # zoetrope has the idle machines; the group allows two
    assert all(json.loads((runs_[s] / "job/prepared.json").read_text())["pool"] == "zoetrope" for s in sent)
    assert json.loads((runs_["RS"] / "job/prepared.json").read_text())["pool"] == "h800"
    assert any("e3 已有 2 个在跑或排队（同时最多 2 个）" in n for n in w.notes)
    jobs.states["E3-B"] = "SUCCEEDED"  # one ends: the next one goes
    w = watcher(tmp_path, *args, jobs=jobs)
    w.tick()
    assert (runs_["RS"] / "job/submission.json").exists() and not (runs_["RD"] / "job/submission.json").exists()


def test_e3_and_x_rows_of_the_matrix(tmp_path):
    from test_watch_and_tasks import summary

    jobs = FakeJobs()
    runs_ = e3_queue(tmp_path, jobs, starts=("B",))
    w = watcher(tmp_path, jobs=jobs)
    assert "排队等空闲机器（同时最多 2 个）" in watch.matrix_e3_train(w, "B") and watch.matrix_e3_train(w, "S") == "未开始"
    jobs.submit(runs_["B"])
    (runs_["B"] / "rlinf_train_20261026T060000Z.log").write_text("│ Global Step:    36/100 │\n│ Global Step:    37/100 │\n")
    assert "训练到第 37 / 100 次更新" in watch.matrix_e3_train(w, "B") and "**RUNNING**" in watch.matrix_e3_train(w, "B")
    e1 = tmp_path / "experiments/DS-B-100-s1"
    summary(e1 / "evals/steps_200000/out/episodes", {"OpenDrawer": {"n": 50, "successes": 20},
                                                      "CloseFridge": {"n": 20, "successes": 4}})
    summary(runs_["B"] / "evals/final/out/episodes", {"OpenDrawer": {"n": 50, "successes": 30},
                                                       "CloseFridge": {"n": 20, "successes": 4}})
    cell = watch.matrix_e3_eval(w, "B")
    assert "RL 后：总体 48.6%（34/70），空间组 60.0%（30/50），RL 的 5 个任务 60.0%（30/50）" in cell
    assert "RL 前（E1 第 20 万步）：总体 34.3%（24/70）" in cell and "训完、导出、评完" in watch.matrix_e3_train(w, "B")
    text = watch.matrix(w)
    assert "## 新增实验（第一波，你 10-26 的决定）" in text and text.count("设计待你确认（10-26）") == 6
    (tmp_path / "x/X3").mkdir(parents=True)
    (tmp_path / "x/X3/matrix.json").write_text(json.dumps({"state": "设计已确认，重放中", "B": "选对 80%"}))
    row = next(line for line in watch.matrix(w).splitlines() if line.startswith("| X3 |"))
    assert "| 设计已确认，重放中 | 选对 80% | — | — | — |" in row


def test_one_card_queue_jobs_wait_for_what_they_need_and_take_only_free_cards(tmp_path):
    """X4 (2026-10-26): its evaluations are one-card jobs; each E1 model's waits for its export; free cards are those
    on machines already running something plus whole idle machines; the main line in that pool holds them back."""
    from test_watch_and_tasks import busy_view

    jobs = FakeJobs(views={"h800": busy_view(("10.0.0.1", [("other", 6)]))})  # 2 free cards, no idle machine
    (tmp_path / "queue").mkdir()
    done = tmp_path / "x/X4/hf/E1-B-200k/starvla_export.json"
    runs_ = []
    for i, req in enumerate([[str(done)], [], [], []]):
        run = tmp_path / f"x/X4/ev{i}"
        run.mkdir(parents=True)
        jobs.prepare(run, 1, "h800", f"x4-ev{i}", ["bash", "eval.sh"])
        (tmp_path / f"queue/q4_x4_{i}.json").write_text(json.dumps(
            {"run": str(run), "cards": 1, "what": f"X4 {i}", "requires": req, "pools": ["h800"]}))
        runs_.append(run)
    args = ("--exps", "DS-B-100-s1", "--pool-nodes", "h800:0", "--yield-pool-only")
    w = watcher(tmp_path, *args, jobs=jobs)
    w.tick()
    sent = [r.name for r in runs_ if (r / "job/submission.json").exists()]
    assert sent == ["ev1", "ev2"]  # ev0 waits for its export, ev3 finds no free card left
    assert any("还缺 starvla_export.json" in n for n in w.notes) and any("没有空卡" in n for n in w.notes)
    jobs.listing = [eight_card("interiorgs-v10-eval-x", "QUEUEING", pool="h800", cards=1)]  # the main line queues
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    jobs.views["h800"] = busy_view(("10.0.0.1", [("other", 4)]))
    w = watcher(tmp_path, *args, jobs=jobs)
    w.tick()
    assert not (runs_[0] / "job/submission.json").exists() and any("给排在前面的任务让路" in n for n in w.notes)
