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

    def submit(self, start, ratio, seed, root):
        self.submitted.append((start, ratio, seed))
        return {"exp": f"DS-{start}-{ratio}-s{seed}", "job": "pt-x"}


def watcher(tmp_path, *extra, jobs=None, free=FREE, run_tools=None, shards=0):
    args = watch.parser().parse_args(["--runs-root", str(tmp_path), "--state", str(tmp_path / "state.json"),
                                      "--report-out", "", "--shards", str(shards),
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
             for p in jobs.prepared]
    assert len(order) == 40  # 6 busy machines x 7 free cards = 42, capped at 40
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


def test_merged_shards_feed_the_gate(tmp_path):
    jobs, run_tools = FakeJobs(), FakeRuns()
    exp = fake_exp(tmp_path, steps=(25000,))
    for i, ok in ((1, 0), (2, 2)):
        summary(exp / f"evals/steps_25000/shards/s{i}of2/out/episodes", {"OpenDrawer": {"n": 2, "successes": ok}},
                outcomes=[("OpenDrawer", 1000 + 2 * i, ok > 0), ("OpenDrawer", 1001 + 2 * i, ok > 1)])
    m = watch.merged(exp / "evals/steps_25000")
    assert m[0]["per_task"]["OpenDrawer"] == {"n": 4, "successes": 2} and m[0]["shards"] == 2 and len(m[1]) == 4
    gate = watcher(tmp_path, jobs=jobs, run_tools=run_tools, shards=2).gate()
    assert gate["decided"] and gate["successes"] == 2 and len(run_tools.submitted) == 4
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
    w = watcher(tmp_path, "--gate", "")
    rep = w.repeat_results()
    by_p = {r["workers_tested"]: r for r in rep}
    assert by_p[48]["slowdown"] == 1.4 and by_p[48]["episodes_with_other_outcome"] == 0
    assert by_p[96]["slowdown"] == 1.6 and by_p[96]["rate_diff_points"] == 50.0
    assert by_p[96]["episodes_with_other_outcome"] == 1
    assert w.workers() == 48 and w.state["workers"]["limit"] == 1.5


def gate_setup(tmp_path, successes):
    exp = fake_exp(tmp_path, steps=(25000,))
    (exp / "evals/steps_25000/job").mkdir(parents=True)
    (exp / "evals/steps_25000/job/submission.json").write_text(json.dumps({"job_id": "pt-g"}))
    summary(exp / "evals/steps_25000/out/episodes", {"OpenDrawer": {"n": 50, "successes": successes},
                                                     "CloseFridge": {"n": 20, "successes": 0}})
    return FakeJobs({"steps_25000": "SUCCEEDED"}), FakeRuns()


def test_gate_submits_the_four_10_percent_seed1_runs_after_a_nonzero_first_checkpoint(tmp_path):
    jobs, run_tools = gate_setup(tmp_path, successes=3)
    w = watcher(tmp_path, jobs=jobs, run_tools=run_tools)
    gate = w.gate()
    assert gate["decided"] and gate["successes"] == 3 and len(run_tools.submitted) == 4
    assert {r for _, r, _ in run_tools.submitted} == {"010"} and {s for *_, s in run_tools.submitted} == {1}
    w.save()
    again = watcher(tmp_path, jobs=jobs, run_tools=run_tools).gate()  # decided once, kept in the state file
    assert again["decided"] and len(run_tools.submitted) == 4


def test_gate_stops_on_zero_successes(tmp_path):
    jobs, run_tools = gate_setup(tmp_path, successes=0)
    w = watcher(tmp_path, jobs=jobs, run_tools=run_tools)
    gate = w.gate()
    assert gate["decided"] and gate["successes"] == 0 and not run_tools.made and not run_tools.submitted
    assert any("成功 0 个回合" in a for a in w.alerts)


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
