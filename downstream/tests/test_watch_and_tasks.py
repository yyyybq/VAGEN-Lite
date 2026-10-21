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
    def __init__(self, states=None):
        self.prepared, self.states = [], states or {}

    def pick_pool(self):
        return "h800"

    def prepare(self, run, gpus, pool, name, payload):
        (run / "job").mkdir(parents=True, exist_ok=True)
        self.prepared.append((run.name, gpus, name, payload))

    def submit(self, run):
        (run / "job/submission.json").write_text(json.dumps({"job_id": f"pt-{run.name}"}))
        return f"pt-{run.name}"

    def status(self, run):
        st = self.states.get(run.name, "RUNNING")
        return {"state": st, "finished": st in ("SUCCEEDED", "FAILED"), "exit_code": "0"}


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


def watcher(tmp_path, *extra, jobs=None, free=FREE, run_tools=None):
    args = watch.parser().parse_args(["--runs-root", str(tmp_path), "--state", str(tmp_path / "state.json"),
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
    by_run = {name: payload for name, _, _, payload in jobs.prepared}
    assert by_run["steps_25000"][1:] == [str(watch.HERE / "run_eval_ckpt.sh"),
                                         str(exp / "run/checkpoints/steps_25000_pytorch_model.pt"),
                                         str(exp / "evals/steps_25000/out"), "1", "24"]
    assert by_run["steps_25000_repeat"][-1] == "48,96" and {g for _, g, _, _ in jobs.prepared} == {8}


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


def test_gate_submits_the_other_24_runs_after_a_nonzero_first_checkpoint(tmp_path):
    jobs, run_tools = gate_setup(tmp_path, successes=3)
    w = watcher(tmp_path, jobs=jobs, run_tools=run_tools)
    gate = w.gate()
    assert gate["decided"] and gate["successes"] == 3 and len(run_tools.submitted) == 24
    assert {r for _, r, _ in run_tools.submitted} == {"030", "010"} and {s for *_, s in run_tools.submitted} == {1, 2, 3}
    w.save()
    again = watcher(tmp_path, jobs=jobs, run_tools=run_tools).gate()  # decided once, kept in the state file
    assert again["decided"] and len(run_tools.submitted) == 24


def test_gate_stops_on_zero_successes(tmp_path):
    jobs, run_tools = gate_setup(tmp_path, successes=0)
    w = watcher(tmp_path, jobs=jobs, run_tools=run_tools)
    gate = w.gate()
    assert gate["decided"] and gate["successes"] == 0 and not run_tools.made and not run_tools.submitted
    assert any("成功 0 个回合" in a for a in w.alerts)


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
