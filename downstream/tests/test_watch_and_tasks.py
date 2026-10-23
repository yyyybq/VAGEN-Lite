"""CPU tests of downstream/eval/watch.py (fake sco, quota and run tools), the evaluation plan, the quota parser, the
pool picker and the spatial-task list."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import closed_loop  # noqa: E402
import quota  # noqa: E402
import sco_job  # noqa: E402
import watch  # noqa: E402

OLD = time.time() - 3600
FREE = {"quota_tb": 22.0, "used_tb": 12.0, "free_tb_min": 9.5}


def fake_exp(root: Path, name="DS-B-100-s1", steps=(25000, 50000), final=True, settled=(25000,)):
    exp = root / "experiments" / name
    (exp / "run/checkpoints").mkdir(parents=True)
    (exp / "experiment.json").write_text(json.dumps({"steps": 200000}))
    for s in steps:
        p = exp / f"run/checkpoints/steps_{s}_pytorch_model.pt"
        p.write_bytes(b"x")
        if s in settled:
            os.utime(p, (OLD, OLD))
    if final:
        (exp / "run/final_model").mkdir()
        (exp / "run/final_model/pytorch_model.pt").write_bytes(b"x")
    (exp / "run/metrics.jsonl").write_text(json.dumps({"step": 50100, "action_dit_loss": 0.0123}) + "\n")
    return exp


class FakeJobs:
    def __init__(self, states=None, views=None, job_states=None):
        self.prepared, self.states, self.deleted = [], states or {}, []
        self.views, self.job_states = views or {}, job_states or {}

    def pick_pool(self):
        return "h800"

    def prepare(self, run, gpus, pool, name, payload):
        (run / "job").mkdir(parents=True, exist_ok=True)
        (run / "job/prepared.json").write_text(json.dumps({"pool": pool}))
        self.prepared.append((run.name, gpus, name, payload, pool))

    def submit(self, run):
        jid = f"pt-{run.parent.parent.name if run.parent.name == 'shards' else ''}{run.name}"
        (run / "job/submission.json").write_text(json.dumps({"job_id": jid}))
        return jid

    def status(self, run):
        st = self.states.get(run.name, "RUNNING")
        return {"state": st, "finished": st in ("SUCCEEDED", "FAILED"), "exit_code": "0"}

    def job_state(self, job_id):
        return self.job_states.get(job_id)

    listing = []

    def list_jobs(self):
        return list(self.listing)

    def pool_view(self, pool, jobs=None):
        return self.views.get(pool, {"hosts": {}})

    def delete(self, run):
        self.deleted.append(run.name)
        return "deleted"


class FakeRuns:
    def __init__(self):
        self.made, self.submitted = [], []

    grid = staticmethod(watch.runs.grid)

    def make(self, start, ratio, seed, root):
        self.made.append((start, ratio, seed))
        return {"exp": f"DS-{start}-{ratio}-s{seed}", "made": True}

    def submit(self, start, ratio, seed, root, pool=None):
        self.submitted.append((start, ratio, seed))
        self.pools = getattr(self, "pools", []) + [pool]
        return {"exp": f"DS-{start}-{ratio}-s{seed}", "job": "pt-x"}


def watcher(tmp_path, *extra, jobs=None, free=FREE, run_tools=None, shards=0):
    args = watch.parser().parse_args(["--runs-root", str(tmp_path), "--state", str(tmp_path / "state.json"),
                                      "--report-out", "", "--shards", str(shards),
                                      "--repeat", "DS-B-100-s1:steps_25000",
                                      # never the real pause file (the main window may have created it)
                                      "--pause-file", str(tmp_path / "NO_PAUSE"),
                                      "--exps", "DS-B-100-s1", "DS-S-100-s1", *extra])
    return watch.Watcher(args, jobs=jobs or FakeJobs(), free_space=lambda: free, run_tools=run_tools or FakeRuns())


def summary(path: Path, per_task, mean=100.0, workers=24, outcomes=()):
    path.mkdir(parents=True, exist_ok=True)
    succ = sum(s["successes"] for s in per_task.values())
    n = sum(s["n"] for s in per_task.values())
    (path / "summary.json").write_text(json.dumps({"per_task": per_task, "successes": succ, "complete": n,
                                                   "episode_wall_mean_seconds": mean, "wall_seconds": 3600,
                                                   "workers": workers}))
    (path / "episodes.jsonl").write_text("".join(json.dumps({"task": t, "seed": s, "success": ok, "status": "complete"})
                                                 + "\n" for t, s, ok in outcomes))


def test_checkpoints_in_step_order_with_final_tagged_by_max_steps(tmp_path):
    exp = fake_exp(tmp_path, steps=(50000, 25000))
    assert [t for t, _ in watch.checkpoints(exp)] == ["steps_25000", "steps_50000", "steps_200000"]


def test_dry_pass_waits_for_settled_files_and_never_submits(tmp_path):
    exp = fake_exp(tmp_path)
    w = watcher(tmp_path, "--dry")
    rows, missing, rep, gate = w.tick()
    states = {e["tag"]: e["state"] for e in rows[0]["evals"]}
    assert states["steps_25000"] == "would submit (workers 24)"
    assert states["steps_25000_repeat"] == "would submit (workers 48,96)"
    assert states["steps_50000"].startswith("waiting") and states["steps_200000"].startswith("waiting")
    assert rows[0]["step"] == 50100 and rows[0]["train"] == "未提交" and missing == ["DS-S-100-s1"]
    assert not (exp / "evals").exists() and not w.alerts
    text = watch.board(w, rows, missing, rep, gate)
    assert "DS-B-100-s1" in text and "0.0123" in text and "剩余至少 9.5 TB" in text


def test_submissions_carry_the_plan_parameters(tmp_path):
    exp = fake_exp(tmp_path)
    jobs = FakeJobs()
    watcher(tmp_path, jobs=jobs).tick()
    by_run = {name: payload for name, _, _, payload, _ in jobs.prepared}
    assert by_run["steps_25000"][1:] == [str(watch.HERE / "run_eval_ckpt.sh"),
                                         str(exp / "run/checkpoints/steps_25000_pytorch_model.pt"),
                                         str(exp / "evals/steps_25000/out"), "1", "24"]
    assert by_run["steps_25000_repeat"][-1] == "48,96" and {g for _, g, _, _, _ in jobs.prepared} == {8}


def busy_view(*hosts):
    """{"hosts": {ip: [jobs]}} from (ip, [(name, cards), ...]) pairs."""
    return {"hosts": {ip: [{"job_id": f"pt-{n}", "name": n, "cards": c} for n, c in jobs] for ip, jobs in hosts}}


def test_busy_room_and_alone_on_machine():
    from sco_job import alone_on_machine, busy_room

    view = busy_view(("10.0.0.1", [("interiorgs-v10-rl-x", 4)]), ("10.0.0.2", [("downstream-ev-a-s1of8", 1)]),
                     ("10.0.0.3", [("other-8card", 8)]),
                     ("10.0.0.4", [("downstream-ev-b-s2of8", 1), ("interiorgs-renderer", 1)]))
    assert busy_room(view) == {"10.0.0.1": 4, "10.0.0.4": 6}  # not .2 (only our shard), not .3 (full)
    assert alone_on_machine(view, "pt-downstream-ev-a-s1of8") is True
    assert alone_on_machine(view, "pt-downstream-ev-b-s2of8") is False
    assert alone_on_machine(view, "pt-missing") is None


def test_shards_go_only_into_room_on_busy_machines(tmp_path):
    exp = fake_exp(tmp_path)
    jobs = FakeJobs(views={"h800": busy_view(("10.0.0.1", [("interiorgs-v10-rl-x", 5)]))})
    w = watcher(tmp_path, jobs=jobs, shards=8)
    w.tick()
    shard_jobs = [p for p in jobs.prepared if p[0].startswith("s")]
    assert len(shard_jobs) == 3  # the 3 free cards of the busy machine; zoetrope has no busy machine with room
    name, gpus, job_name, payload, pool = shard_jobs[0]
    assert gpus == 1 and pool == "h800" and job_name == "ev-ds-b-100-s1-steps25000-s1of8"
    assert payload[-3:] == ["1", "12", "1/8"] and payload[2].endswith("steps_25000_pytorch_model.pt")
    assert [p[3][3].split("evals/")[1] for p in shard_jobs] == [  # the first evaluation before its repeat
        "steps_25000/shards/s1of8/out", "steps_25000/shards/s2of8/out", "steps_25000/shards/s3of8/out"]
    assert any("两个池里都没有已有别的任务、还有空卡的机器" in n for n in w.notes)
    assert (exp / "evals/steps_25000/shards/s1of8/job/submission.json").exists()


def test_yield_pause_and_live_cap_block_shards(tmp_path):
    fake_exp(tmp_path)
    room = {"h800": busy_view(("10.0.0.1", [("other", 1)]))}
    jobs = FakeJobs(views=room)
    jobs.listing = [{"display_name": "interiorgs-v10-eval-x", "state": "QUEUEING"}]
    w = watcher(tmp_path, jobs=jobs, shards=8)
    w.tick()
    assert not jobs.prepared and any("给排在前面的任务让路" in n for n in w.notes)
    jobs.listing = [{"display_name": f"downstream-ev-x-s{i}of8", "state": "RUNNING"} for i in range(8)]
    w = watcher(tmp_path, jobs=jobs, shards=8)
    w.tick()
    assert not jobs.prepared and any("已到上限" in n for n in w.notes)
    jobs.listing = []
    (tmp_path / "PAUSE").write_text("main window asked")
    w = watcher(tmp_path, "--pause-file", str(tmp_path / "PAUSE"), jobs=jobs, shards=8)
    w.tick()
    assert not jobs.prepared and any("暂停提交" in n for n in w.notes)
    (tmp_path / "PAUSE").unlink()
    watcher(tmp_path, jobs=jobs, shards=8).tick()
    assert len(jobs.prepared) == 7


def test_bq_tp_jobs_hold_back_only_submissions_of_their_own_size(tmp_path):
    spec = lambda c: [{"resource_spec": [{"name": f"N4lS.Iq.I80.{c}"}]}]  # noqa: E731
    jobs = FakeJobs()
    w = watcher(tmp_path, jobs=jobs)
    jobs.listing = [{"display_name": "bq_tp_mmxbf16_t1000", "state": "STARTING", "roles": spec(8)},
                    {"display_name": "bq_tp_mmxbf16_x", "state": "RUNNING", "roles": spec(1)}]
    assert w.priority_waiting(1) == [] and w.priority_waiting(8) == ["bq_tp_mmxbf16_t1000"]
    w = watcher(tmp_path, jobs=jobs)
    jobs.listing = [{"display_name": "bq_tp_mmxbf16_eval", "state": "QUEUEING", "roles": spec(1)},
                    {"display_name": "bq_tp_mmxbf16_nospec", "state": "QUEUEING"}]
    assert w.priority_waiting(1) == ["bq_tp_mmxbf16_eval", "bq_tp_mmxbf16_nospec"]  # unknown size: wait too
    assert w.priority_waiting(8) == ["bq_tp_mmxbf16_nospec"]
    assert watch.job_cards({"roles": spec(8)}) == 8 and watch.job_cards({}) is None


def test_stage0_goes_100k_then_50k_then_25k_and_75k_start_by_start(tmp_path):
    for start in ("S", "B"):
        exp = tmp_path / f"experiments/DS-{start}-100-s1"
        (exp / "run_stopped_20261022/checkpoints").mkdir(parents=True)
        (exp / "experiment.json").write_text(json.dumps({"steps": 200000}))
        for s in (25000, 50000, 75000, 100000):
            p = exp / f"run_stopped_20261022/checkpoints/steps_{s}_pytorch_model.pt"
            p.write_bytes(b"x")
            os.utime(p, (OLD, OLD))
    jobs = FakeJobs(views={"h800": busy_view(*[(f"10.0.0.{i}", [("other", 1)]) for i in range(6)])})
    w = watcher(tmp_path, "--stage0", "--max-live-shards", "40", jobs=jobs, shards=8)
    rows, *_ = w.tick()
    order = [p[3][3].split("experiments/")[1].replace("/evals_run_stopped_20261022", "").replace("/out", "")
             for p in jobs.prepared if p[1] == 1]  # one-card shards (B 100k's second, 8-card evaluation aside)
    assert len(order) == 40  # 6 busy machines x 7 free cards = 42, capped at 40
    assert [p[2] for p in jobs.prepared if p[1] == 8] == ["ev8-ds-b-100-s1-stopped-steps100000-rep"]
    assert order[0] == "DS-B-100-s1/steps_100000/shards/s1of8" and order[7] == "DS-B-100-s1/steps_100000/shards/s8of8"
    assert order[8] == "DS-S-100-s1/steps_100000/shards/s1of8"
    assert order[16].startswith("DS-B-100-s1/steps_50000") and order[24].startswith("DS-S-100-s1/steps_50000")
    assert {o.split("/")[1] for o in order[32:]} == {"steps_25000"}  # 25k and 75k share a rank: one checkpoint at a time
    assert jobs.prepared[0][2] == "ev-ds-b-100-s1-stopped-steps100000-s1of8"
    assert rows[0]["exp"].startswith("DS-B-100-s1（10-22 停下的那次）")


def test_hold_until_the_priority_jobs_run(tmp_path):
    fake_exp(tmp_path)
    room = {"h800": busy_view(("10.0.0.1", [("other", 1)]))}
    jobs = FakeJobs(views=room, job_states={"pt-dense1": "RUNNING", "pt-dense2": "STARTING"})
    w = watcher(tmp_path, "--hold-until-running", "pt-dense1", "pt-dense2", jobs=jobs, shards=8)
    w.tick()
    assert not jobs.prepared and any("等密集奖励第一批上机器" in n for n in w.notes)
    assert "hold_released" not in w.state
    jobs.job_states["pt-dense2"] = "RUNNING"
    w2 = watcher(tmp_path, "--hold-until-running", "pt-dense1", "pt-dense2", jobs=jobs, shards=8)
    w2.tick()
    assert w2.state["hold_released"] and len(jobs.prepared) == 7  # 7 free cards on the busy machine


def test_guard_deletes_a_shard_that_started_alone_and_waits_before_resubmitting(tmp_path):
    exp = fake_exp(tmp_path)
    jobs = FakeJobs(views={"h800": busy_view(("10.0.0.1", [("other", 6)]))})
    watcher(tmp_path, jobs=jobs, shards=8).tick()  # 2 shards submitted: s1of8, s2of8
    jobs.views["h800"] = busy_view(("10.0.0.9", [("downstream-ev-x", 1)]))
    jobs.views["h800"]["hosts"]["10.0.0.9"] = [{"job_id": "pt-steps_25000s1of8", "name": "downstream-ev-x", "cards": 1}]
    jobs.views["h800"]["hosts"]["10.0.0.1"] = [{"job_id": "pt-steps_25000s2of8", "name": "downstream-ev-y", "cards": 1},
                                              {"job_id": "pt-o", "name": "other", "cards": 6}]
    w = watcher(tmp_path, jobs=jobs, shards=8)
    out = w.guard()
    assert len(out) == 1 and jobs.deleted == ["s1of8"]
    assert list((exp / "evals/steps_25000/shards/s1of8").glob("job_bounced_*"))
    assert "pt-steps_25000s2of8" in w.state["verified"]
    w.save()
    jobs.views["h800"] = busy_view(("10.0.0.1", [("other", 1)]))
    before = len(jobs.prepared)
    w3 = watcher(tmp_path, jobs=jobs, shards=8)
    w3.tick()
    resubmitted = [p for p in jobs.prepared[before:] if "/evals/steps_25000/shards/s1of8/" in p[3][3]]
    assert not resubmitted  # the bounced shard waits --bounce-wait-min
    assert any("/evals/steps_25000/shards/s3of8/" in p[3][3] for p in jobs.prepared[before:])  # others go on


def test_merged_combines_shards(tmp_path):
    exp = fake_exp(tmp_path, steps=(25000,))
    for i, ok in ((1, 0), (2, 2)):
        summary(exp / f"evals/steps_25000/shards/s{i}of2/out/episodes", {"OpenDrawer": {"n": 2, "successes": ok}},
                outcomes=[("OpenDrawer", 1000 + 2 * i, ok > 0), ("OpenDrawer", 1001 + 2 * i, ok > 1)])
    m = watch.merged(exp / "evals/steps_25000")
    assert m[0]["per_task"]["OpenDrawer"] == {"n": 4, "successes": 2} and m[0]["shards"] == 2 and len(m[1]) == 4
    (exp / "evals/steps_25000/shards/s2of2/out/episodes/summary.json").unlink()
    assert watch.merged(exp / "evals/steps_25000") is None


def test_low_or_unknown_space_blocks_every_submission(tmp_path):
    fake_exp(tmp_path)
    for free in ({"quota_tb": 22.0, "used_tb": 21.0, "free_tb_min": 0.5}, None):
        jobs = FakeJobs()
        w = watcher(tmp_path, jobs=jobs, free=free)
        rows, *_ = w.tick()
        assert not jobs.prepared and w.alerts and rows[0]["evals"][0]["state"].startswith("未提交")


def test_repeat_compares_and_picks_the_largest_fast_enough_parallelism(tmp_path):
    exp = fake_exp(tmp_path, steps=(25000,))
    per = {"OpenDrawer": {"n": 2, "successes": 1}}
    summary(exp / "evals/steps_25000/out/episodes", per, mean=100.0,
            outcomes=[("OpenDrawer", 1000, True), ("OpenDrawer", 1001, False)])
    summary(exp / "evals/steps_25000_repeat/out/episodes_P48", per, mean=140.0, workers=48,
            outcomes=[("OpenDrawer", 1000, True), ("OpenDrawer", 1001, False)])
    summary(exp / "evals/steps_25000_repeat/out/episodes_P96", {"OpenDrawer": {"n": 2, "successes": 2}}, mean=160.0,
            workers=96, outcomes=[("OpenDrawer", 1000, True), ("OpenDrawer", 1001, True)])
    w = watcher(tmp_path)
    rep = w.repeat_results()
    by_p = {r["workers_tested"]: r for r in rep}
    assert by_p[48]["slowdown"] == 1.4 and by_p[48]["episodes_with_other_outcome"] == 0
    assert by_p[96]["slowdown"] == 1.6 and by_p[96]["rate_diff_points"] == 50.0
    assert by_p[96]["episodes_with_other_outcome"] == 1
    assert w.workers() == 48 and w.state["workers"]["limit"] == 1.5


def stopped_100k(root: Path, start: str, spatial_successes: int):
    """Stage 0's evaluation of a start's stopped 100k checkpoint (OpenDrawer is a spatial task, CloseFridge is not)."""
    exp = root / "experiments" / f"DS-{start}-100-s1"
    exp.mkdir(parents=True, exist_ok=True)
    summary(exp / f"evals_{watch.STOPPED_RUN}/steps_100000/out/episodes",
            {"OpenDrawer": {"n": 50, "successes": spatial_successes}, "CloseFridge": {"n": 20, "successes": 5}})


def stage2_dirs(root: Path, ready=("B", "S", "RS", "RD")):
    for s in ("B", "S", "RS", "RD"):
        exp = root / "experiments" / f"DS-{s}-010-s1"
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "experiment.json").write_text(json.dumps({"steps": 100000, "segmented": s in ready}))


def eight_card(name, state="RUNNING", pool="h800", cards=8):
    return {"display_name": name, "state": state, "resource_pool": {"name": pool},
            "roles": [{"resource_spec": [{"name": f"N4lS.Iq.I80.{cards}"}]}]}


def test_gate_passes_on_spatial_successes_at_100k_and_fills_free_slots(tmp_path):
    stopped_100k(tmp_path, "B", 0)
    stopped_100k(tmp_path, "S", 2)
    stage2_dirs(tmp_path, ready=("B", "S", "RS"))
    jobs, run_tools = FakeJobs(), FakeRuns()
    jobs.listing = [eight_card("downstream-ds-b-100-s1"), eight_card("downstream-ds-s-100-s1", "QUEUEING"),
                    eight_card("downstream-ev-x-s1of8", cards=1), eight_card("someone-else", cards=8)]
    w = watcher(tmp_path, jobs=jobs, run_tools=run_tools)
    gate = w.gate()
    assert gate["passed"] and "DS-S-100-s1（停下的那次） 空间组 2/50" in gate["note"]
    assert run_tools.submitted == [("B", "010", 1), ("S", "010", 1)]  # 4 places - 2 alive; RD's directory not ready
    assert "1 个实验目录还没按阶段 1 的设置重建" in gate["note"]
    w.save()
    jobs.listing += [eight_card("downstream-ds-b-010-s1"), eight_card("downstream-ds-s-010-s1")]
    again = watcher(tmp_path, jobs=jobs, run_tools=run_tools).gate()  # passed once, kept; no place now
    assert again["passed"] and "4 个 8 卡名额已满" in again["note"] and len(run_tools.submitted) == 2


def test_gate_waits_while_spatial_successes_are_zero(tmp_path):
    for s in ("B", "S", "RS", "RD"):
        stopped_100k(tmp_path, s, 0)
    stage2_dirs(tmp_path)
    run_tools = FakeRuns()
    w = watcher(tmp_path, run_tools=run_tools)
    gate = w.gate()
    assert not gate["passed"] and not run_tools.submitted and any("闸门没过" in a for a in w.alerts)


def test_same_size_yield_is_per_pool_for_eight_card_jobs(tmp_path):
    stopped_100k(tmp_path, "B", 1)
    stage2_dirs(tmp_path)
    jobs, run_tools = FakeJobs(), FakeRuns()
    jobs.listing = [eight_card("bq_tp_mmxbf16_a", "STARTING", pool="zoetrope")]
    w = watcher(tmp_path, jobs=jobs, run_tools=run_tools)
    gate = w.gate()
    assert len(run_tools.submitted) == 4 and set(run_tools.pools) == {"h800"}  # zoetrope's bq_tp job keeps its pool
    assert not w.blocked(1) and w.blocked(8, "zoetrope") and not w.blocked(8, "h800") and w.blocked(8)
    jobs.listing.append(eight_card("bq_tp_mmxbf16_b", "STARTING", pool="h800"))
    w2 = watcher(tmp_path, jobs=jobs, run_tools=FakeRuns())
    assert w2.open_pool(8)[0] is None


def test_segmented_runs_evaluate_the_final_in_the_job_and_only_b_twice(tmp_path):
    b = fake_exp(tmp_path, steps=(), settled=())
    (b / "experiment.json").write_text(json.dumps({"steps": 200000, "segmented": True, "eval_points": [200000],
                                                   "repeat_final": True}))
    os.utime(b / "run/final_model/pytorch_model.pt", (OLD, OLD))
    s = fake_exp(tmp_path, name="DS-S-100-s1", steps=(), settled=())
    (s / "experiment.json").write_text(json.dumps({"steps": 200000, "segmented": True, "eval_points": [200000]}))
    os.utime(s / "run/final_model/pytorch_model.pt", (OLD, OLD))
    jobs = FakeJobs(views={"h800": busy_view(("10.0.0.1", [("other", 1)]))})
    off = watcher(tmp_path, "--repeat", jobs=jobs, shards=8)  # the user cancelled B's second evaluation (10-23)
    assert off.repeat_targets() == [] and {e["tag"] for e in off.experiment_row(b)["evals"]} == {
        "steps_150000", "steps_175000", "steps_200000"}
    w = watcher(tmp_path, "--repeat", "--repeat-final", "--extra-points", jobs=jobs, shards=8)
    assert w.repeat_targets() == ["DS-B-100-s1:steps_200000"]
    rows, *_ = w.tick()
    b_states = {e["tag"]: e["state"] for e in rows[0]["evals"]}
    s_states = {e["tag"]: e["state"] for e in rows[1]["evals"]}
    assert b_states["steps_200000"].startswith("由训练任务自己评")
    assert b_states["steps_200000_repeat"].startswith("待提交（8 卡") and set(s_states) == {"steps_200000"}
    assert [(p[1], p[3][-2:]) for p in jobs.prepared] == [(8, ["8", "96"])]  # B's second evaluation, whole machine
    assert jobs.prepared[0][3][3].endswith("DS-B-100-s1/evals/steps_200000_repeat/out")


def test_report_is_rebuilt_once_per_new_set_of_finished_evaluations(tmp_path):
    from types import SimpleNamespace

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0 if len(calls) <= 2 else 1, stdout="", stderr="boom")

    w = watcher(tmp_path)
    w.a.report_out = str(tmp_path / "report")
    w.report(run=fake_run)  # first pass: a report of the training losses
    assert len(calls) == 1 and calls[0][1:4] == ["-m", "downstream.report.build", "--runs-root"]
    w.report(run=fake_run)
    assert len(calls) == 1  # nothing new finished: no rebuild
    w.state["finished"]["/x/experiments/DS-B-100-s1/evals/steps_25000"] = {"state": "SUCCEEDED"}
    w.report(run=fake_run)
    assert len(calls) == 2 and w.state["report"]["rc"] == 0 and not w.alerts
    w.state["finished"]["/x/experiments/DS-S-100-s1/evals/steps_25000"] = {"state": "SUCCEEDED"}
    w.report(run=fake_run)
    assert len(calls) == 3 and w.alerts and "退出码 1" in w.alerts[0]
    assert len(w.state["report"]["evaluations"]) == 1  # a failed build does not count as built
    w.a.report_out = ""
    w.state["finished"]["/x/experiments/DS-RS-100-s1/evals/steps_25000"] = {"state": "SUCCEEDED"}
    w.report(run=fake_run)
    assert len(calls) == 3  # switched off


def test_score_groups_spatial_and_borderline(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"per_task": {
        "OpenDrawer": {"n": 50, "successes": 10}, "CloseFridge": {"n": 20, "successes": 5},
        "SearingMeat": {"n": 50, "successes": 1}, "KettleBoiling": {"n": 20, "successes": 2}}}))
    s = watch.score(p)
    assert s["all"] == [18, 140] and s["atomic_seen"] == [15, 70] and s["composite_seen"] == [3, 70]
    assert s["spatial"] == [11, 100] and s["borderline"] == [2, 20]
    assert watch.fmt(s) == "18/140 / 15/70 / 3/70 / 11/100 / 2/20"


def test_eval_plan_has_950_episodes_with_50_per_spatial_task():
    episodes, plan = closed_loop.eval_plan_episodes(ROOT / "eval/eval_d6.json")
    spatial = set(json.loads((ROOT / "demos/spatial_tasks.json").read_text())["spatial"])
    per_task = {}
    for e in episodes:
        per_task.setdefault(e["task"], []).append(e["seed"])
    assert len(episodes) == 950 and len(per_task) == 34 and plan["stable_counter_regions"] is True
    assert all(len(v) == (50 if t in spatial else 20) for t, v in per_task.items())
    assert per_task["NavigateKitchen"][:2] == [1000, 1001] and max(per_task["CloseFridge"]) == 1019


def test_quota_parse_gives_a_lower_bound():
    text = ("  ID  DIR PATH  FILE QUANTITY QUOTA  USED FILE QUANTITY  CAPACITY QUOTA  USED CAPACITY  RUNNING STATE\n"
            "  019e  /users/xiaohongcan  0  70038223  22 TB  8.6 TB  success\n"
            "  019e  /users/yinbaiqiao   0  12234998  22 TB  12 TB  success\n")
    assert quota.parse(text) == {"quota_tb": 22.0, "used_tb": 12.0, "free_tb_min": 9.5,
                                 "row": "019e  /users/yinbaiqiao   0  12234998  22 TB  12 TB  success"}
    assert quota.parse(text, "/users/xiaohongcan")["free_tb_min"] == 13.35
    assert quota.parse("nothing here") is None
    setting = "  019e  /users/yinbaiqiao  0  0  11 TB  0 B  setting\n"  # what `-d /users/yinbaiqiao` returned
    assert quota.parse(setting) is None


def test_quota_free_space_retries():
    from types import SimpleNamespace

    good = "  019e  /users/yinbaiqiao  0  1  22 TB  12 TB  success\n"
    replies = [SimpleNamespace(returncode=1, stdout=""), SimpleNamespace(returncode=0, stdout=good)]
    assert quota.free_space(run=lambda *a, **k: replies.pop(0), sleep=lambda s: None)["free_tb_min"] == 9.5
    assert quota.free_space(run=lambda *a, **k: SimpleNamespace(returncode=1, stdout=""), sleep=lambda s: None) is None


def test_waiting_by_pool():
    jobs = [{"resource_pool": {"name": "h800"}, "state": "STARTING"},
            {"resource_pool": "zoetrope", "state": "RUNNING"},
            {"resource_pool": {"name": "zoetrope"}, "state": "QUEUEING"},
            {"resource_pool": {"name": "other"}, "state": "STARTING"}]
    assert sco_job.waiting_by_pool(jobs) == {"h800": 1, "zoetrope": 1}


def test_spatial_list_partitions_the_34_seen_tasks():
    tasks = json.loads((ROOT / "demos/target50_tasks.json").read_text())["tasks"]
    seen = {t["task"] for t in tasks if t["group"] != "composite_unseen"}
    lists = json.loads((ROOT / "demos/spatial_tasks.json").read_text())
    parts = [set(lists["spatial"]), set(lists["borderline_not_in_group"]), set(lists["not_spatial"])]
    assert len(seen) == 34 and set().union(*parts) == seen and sum(len(p) for p in parts) == 34


def test_stage0_report_is_built_into_its_own_folder(tmp_path):
    from types import SimpleNamespace

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    w = watcher(tmp_path, "--stage0")
    w.a.report_out = str(tmp_path / "report")
    w.report(run=fake_run)
    assert len(calls) == 2 and calls[0][-2:] == ["--out", str(tmp_path / "report")]
    assert calls[1][-3:] == ["--out", str(tmp_path / "report_stage0"), "--stage0"]
    w.state["finished"][f"/x/experiments/DS-B-100-s1/evals_{watch.STOPPED_RUN}/steps_100000/shards/s1of8"] = {}
    w.report(run=fake_run)
    assert len(calls) == 3 and calls[2][-1] == "--stage0"  # only the stage-0 report had a new evaluation
    w.state["finished"]["/x/experiments/DS-B-100-s1/evals/steps_25000/shards/s1of8"] = {}
    w.report(run=fake_run)
    assert len(calls) == 4 and calls[3][-1] != "--stage0"


def stopped_exp(root: Path, start: str, steps=(25000, 50000, 75000, 100000)):
    exp = root / "experiments" / f"DS-{start}-100-s1"
    (exp / f"{watch.STOPPED_RUN}/checkpoints").mkdir(parents=True)
    (exp / "experiment.json").write_text(json.dumps({"steps": 200000}))
    for s in steps:
        p = exp / f"{watch.STOPPED_RUN}/checkpoints/steps_{s}_pytorch_model.pt"
        p.write_bytes(b"x")
        os.utime(p, (OLD, OLD))
    return exp


def test_stage0_eight_card_jobs_fill_idle_machines_in_order(tmp_path):
    b = stopped_exp(tmp_path, "B")
    stopped_exp(tmp_path, "S")
    (b / f"evals_{watch.STOPPED_RUN}/steps_100000/shards/s1of8").mkdir(parents=True)  # B 100k started as shards
    jobs = FakeJobs(views={"h800": busy_view(("10.0.0.1", [("other", 8)]), ("10.0.0.2", [("other", 8)]))})
    w = watcher(tmp_path, "--stage0", "--stage0-cards", "8", "--pool-nodes", "h800:4", "--max-whole-per-pass", "4",
                jobs=jobs, shards=8)
    rows, *_ = w.tick()
    eight = [p for p in jobs.prepared if p[1] == 8]
    assert [p[2] for p in eight] == ["ev8-ds-s-100-s1-stopped-steps100000", "ev8-ds-b-100-s1-stopped-steps50000"]
    assert eight[0][3][-2:] == ["8", "96"] and eight[0][4] == "h800"  # 4 machines - 2 busy = 2 idle
    b_states = {e["tag"]: e["state"] for e in rows[0]["evals"]}
    assert b_states["steps_100000"].startswith("8 个 1 卡分片")  # shards already begun: kept
    jobs.listing = [eight_card("downstream-ev8-ds-s-100-s1-stopped-steps100000", "STARTING")]
    before = len(jobs.prepared)
    w2 = watcher(tmp_path, "--stage0", "--stage0-cards", "8", "--pool-nodes", "h800:9", jobs=jobs, shards=8)
    w2.tick()
    assert not [p for p in jobs.prepared[before:] if p[1] == 8]  # one of ours waits: no more
    assert any("先不提交更多" in n for n in w2.notes) and w2.live8() == [] and w2.live_shards() == 0


def test_waiting_eight_card_evaluations_yield_to_the_main_line(tmp_path):
    s = stopped_exp(tmp_path, "S", steps=(50000, 100000))
    jobs = FakeJobs(states={"steps_50000": "STARTING"},
                    views={"h800": busy_view(("10.0.0.1", [("other", 8)]))})
    w = watcher(tmp_path, "--stage0", "--stage0-cards", "8", "--pool-nodes", "h800:3", jobs=jobs, shards=8)
    w.tick()  # nothing waits in the main line: one submission per pass (S 100k first)
    assert [p[2] for p in jobs.prepared] == ["ev8-ds-s-100-s1-stopped-steps100000"]
    (s / f"evals_{watch.STOPPED_RUN}/steps_50000/job").mkdir(parents=True)
    (s / f"evals_{watch.STOPPED_RUN}/steps_50000/job/submission.json").write_text(json.dumps({"job_id": "pt-w"}))
    (s / f"evals_{watch.STOPPED_RUN}/steps_50000/job/prepared.json").write_text(json.dumps({"pool": "h800"}))
    jobs.listing = [eight_card("interiorgs-v10-eval-test-x", "QUEUEING", cards=1)]
    w2 = watcher(tmp_path, "--stage0", "--stage0-cards", "8", "--pool-nodes", "h800:3", jobs=jobs, shards=8)
    w2.tick()
    evaldir = s / f"evals_{watch.STOPPED_RUN}/steps_50000"
    assert jobs.deleted == ["steps_50000"] and not (evaldir / "job").exists()
    assert list(evaldir.glob("job_withdrawn_*")) and any("撤下我们还没开始的 8 卡评测 1 个" in n for n in w2.notes)
    assert len(jobs.prepared) == 1  # not resubmitted while the main line waits


def test_stalled_evaluations_are_released_and_resubmitted(tmp_path):
    b = stopped_exp(tmp_path, "B", steps=(50000, 100000))
    root = b / f"evals_{watch.STOPPED_RUN}"
    for tag in ("steps_50000",):
        (root / tag / "job").mkdir(parents=True)
        (root / tag / "job/submission.json").write_text(json.dumps({"job_id": "pt-old"}))
        (root / tag / "job/prepared.json").write_text(json.dumps({"pool": "h800"}))
        eps = root / tag / "out/episodes"
        eps.mkdir(parents=True)
        (eps / "episodes.jsonl").write_text('{"status": "complete"}\n')
        for p in (root / tag / "job/submission.json", eps / "episodes.jsonl"):
            os.utime(p, (OLD - 7200, OLD - 7200))  # started and last wrote 3 h ago
    shard = root / "steps_100000/shards/s2of8"
    (shard / "job").mkdir(parents=True)
    (shard / "job/submission.json").write_text(json.dumps({"job_id": "pt-shard"}))
    (shard / "job/prepared.json").write_text(json.dumps({"pool": "h800"}))
    (shard / "out/episodes").mkdir(parents=True)
    (shard / "out/episodes/episodes.jsonl").write_text('{"status": "complete"}\n')
    for p in (shard / "job/submission.json", shard / "out/episodes/episodes.jsonl"):
        os.utime(p, (OLD - 7200, OLD - 7200))
    jobs = FakeJobs(states={"steps_50000": "RUNNING", "s2of8": "RUNNING"})
    w = watcher(tmp_path, "--stage0", "--stage0-cards", "8", "--stage0-repeat", jobs=jobs, shards=8)
    w.tick()
    assert sorted(jobs.deleted) == ["s2of8", "steps_50000"]
    assert list((root / "steps_50000").glob("job_hung_*")) and list(shard.glob("job_hung_*"))
    assert sum("分钟没有完成一个回合" in a for a in w.alerts) == 2


def test_matrix_lists_the_starts_and_the_pending_rows(tmp_path):
    b = stopped_exp(tmp_path, "B", steps=(25000, 100000))
    summary(b / f"evals_{watch.STOPPED_RUN}/steps_100000/out/episodes",
            {"OpenDrawer": {"n": 50, "successes": 10}, "CloseFridge": {"n": 20, "successes": 5}})
    (b / f"evals_{watch.STOPPED_RUN}/steps_100000/job").mkdir(parents=True)
    (b / f"evals_{watch.STOPPED_RUN}/steps_100000/job/submission.json").write_text(json.dumps({"job_id": "pt-e0"}))
    s1 = tmp_path / "experiments/DS-S-100-s1"
    (s1 / "run").mkdir(parents=True)
    (s1 / "experiment.json").write_text(json.dumps({"steps": 100000, "segmented": True}))
    (s1 / "run/metrics.jsonl").write_text(json.dumps({"step": 4200, "action_dit_loss": 0.1}) + "\n")
    w = watcher(tmp_path, "--stage0", jobs=FakeJobs(), shards=8)
    text = watch.matrix(w)
    # two tables of the same rows and columns (training, evaluation): header, rule line, four starts, rows to be added
    assert sum(line.startswith("| ") for line in text.splitlines()) == 2 * (2 + 4 + len(watch.MATRIX_PENDING))
    train, evals = text.split("## 评测")
    assert "## 训练" in train and train.count(watch.MATRIX_COLUMNS) == 1 and evals.count(watch.MATRIX_COLUMNS) == 1
    row_b = next(line for line in evals.splitlines() if line.startswith("| B 原始 7B"))
    assert "**1/4 个检查点评完**" in row_b and "`pt-e0`" in row_b and "100k：总体 21.4%（15/70），空间组 20.0%（10/50）" in row_b
    row_s = next(line for line in train.splitlines() if line.startswith("| S 动作 SFT"))
    assert "**未提交**" in row_s and "训练到第 4,200 步（共 100,000）" in row_s
    row_s_eval = next(line for line in evals.splitlines() if line.startswith("| S 动作 SFT"))
    assert "100k：训完后由训练任务自己评" in row_s_eval and "0/1 个检查点评完" in row_s_eval
    row_b_train = next(line for line in train.splitlines() if line.startswith("| B 原始 7B"))
    assert "r12k S0-keep（待接入）" in evals and "暂不跑" in row_b_train and "不再训练" in row_b_train


def test_merged_leaves_out_the_excluded_episodes(tmp_path):
    d = tmp_path / "eval"
    summary(d / "out/episodes", {"SearingMeat": {"n": 2, "successes": 1}, "OpenDrawer": {"n": 1, "successes": 1}},
            outcomes=[("SearingMeat", 1044, True), ("SearingMeat", 1045, False), ("OpenDrawer", 1000, True)])
    s, outs = watch.merged(d)
    assert ("SearingMeat", 1044) not in outs and s["per_task"]["SearingMeat"] == {"n": 1, "successes": 0}
    assert s["complete"] == 2 and s["successes"] == 1


def test_queue_all_submits_every_pending_evaluation_at_once(tmp_path):
    for s in ("B", "S"):
        stopped_exp(tmp_path, s, steps=(25000, 75000))
    jobs = FakeJobs()  # no machine looks idle, and one of ours is already waiting
    jobs.listing = [eight_card("downstream-ev8-x", "QUEUEING")]
    w = watcher(tmp_path, "--stage0", "--stage0-cards", "8", "--stage0-repeat", "--queue-all", "--yield-to",
                "--yield-to-same-size", "--pool-nodes", "h800:0", jobs=jobs, shards=8)
    w.tick()
    assert len([p for p in jobs.prepared if p[1] == 8]) == 4
    done = tmp_path / f"experiments/DS-B-100-s1/evals_{watch.STOPPED_RUN}/steps_25000"
    summary(done / "out/episodes", {"OpenDrawer": {"n": 1, "successes": 1}}, outcomes=[("OpenDrawer", 1000, True)])
    import shutil

    shutil.rmtree(done / "job")  # finished, record gone: never submitted again
    before = len(jobs.prepared)
    watcher(tmp_path, "--stage0", "--stage0-cards", "8", "--stage0-repeat", "--queue-all", "--yield-to",
            "--yield-to-same-size", "--pool-nodes", "h800:0", jobs=jobs, shards=8).tick()
    assert not [p for p in jobs.prepared[before:] if "steps_25000" in p[3][3] and "DS-B" in p[3][3]]
