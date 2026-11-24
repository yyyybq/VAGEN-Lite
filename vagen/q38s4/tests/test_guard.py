"""tools.guard: the stage-4 guards on synthetic run folders (no sco, no cluster)."""
import json
from pathlib import Path

import pytest

from vagen.q38s4.tools import guard as G

T0 = 1_800_000_000.0  # a fixed "now"


def iso(t):
    return G.utc(t)


def make_run(tmp_path: Path, arm: str = "nfp", save_steps: int = 500, total: int = 3441) -> Path:
    run = tmp_path / ("c_" + arm)
    (run / "logs").mkdir(parents=True)
    (run / "sft_metrics").mkdir()
    (run / "ckpt").mkdir()
    w = 0.1 if arm == "nfp" else 0.0
    (run / "q38s4.json").write_text(json.dumps({"nfp_mse": w, "nfp_cos": w, "logits": "targets"}))
    (run / "train.yaml").write_text(f"save_strategy: steps\nsave_steps: {save_steps}\nnum_train_epochs: 1\n")
    (run / "plan.json").write_text(json.dumps({"nodes": 8, "gpus_per_node": 8, "estimated_steps": total}))
    (run / "q38_sft.json").write_text(json.dumps({"bf16_copy": "sibling", "host_mem_s": 10}))
    return run


def log_line(step, total, loss="1.2", nfp=(0.15, 0.2), pairs=1000, **extra):
    d = {"loss": loss, "grad_norm": "3.1", "learning_rate": "1e-05", "text_loss": "0.9"}
    if nfp is not None:
        d.update(nfp_pairs=pairs, nfp_videos=8, nfp_mse="2.0", nfp_cos=str(nfp[0]), copy_mse="2.5",
                 copy_cos=str(nfp[1]))
    else:
        d.update(nfp_pairs=0, nfp_videos=0)
    d.update(extra)
    d["epoch"] = "0.01"
    return f" 1%|▏| {step}/{total} [01:00<20:00:00, 25.00s/it]\r   \r{d!r}\n"


def add_steps(run, steps, total=3441, t0=T0 - 3600, step_s=25.0, **kw):
    with open(run / "logs" / "node0.log", "a") as f:
        for s in steps:
            f.write(log_line(s, total, **kw))
    with open(run / "sft_metrics" / "rank0.jsonl", "a") as f:
        for s in steps:
            f.write(json.dumps({"event": "step", "step": s, "utc": iso(t0 + s * step_s), "step_s": step_s,
                                "used_gb": 110.0, "cgroup_current_gb": 300.0}) + "\n")


def guard(run, state=None, start=None, **kw):
    desc = (lambda job: {"state": state, "start_time": start, "display_name": "q38s4-train-nfp-c_nfp"})
    return G.Guard(run, job="pt-x" if state else None, describer=desc, **kw)


# ------------------------------------------------------------------------------------------------ parsing
def test_log_steps_step_total_and_dict():
    lines = [log_line(7, 3441).rstrip("\n"), "no dict here 3/10 [", "{'loss': '1.0'} without a bar",
             log_line(8, 3441).rstrip("\n") + log_line(9, 3441).rstrip("\n")]
    got = G.log_steps(lines)
    assert [(s, t) for s, t, _ in got] == [(7, 3441), (8, 3441), (9, 3441)]
    assert got[0][2]["nfp_cos"] == "0.15" and got[0][2]["nfp_pairs"] == 1000


def test_tail_waits_for_newline_and_restarts_on_truncation(tmp_path):
    p = tmp_path / "f.log"
    t = G.Tail(p)
    assert t.lines() == []
    p.write_text("a\nb")
    assert t.lines() == ["a"]
    with open(p, "a") as f:
        f.write("c\n")
    assert t.lines() == ["bc"]
    assert t.lines() == []
    p.write_text("x\n")  # shorter than before: read from the start
    assert t.lines() == ["x"]


def test_parse_utc_zones():
    assert G.parse_utc("2026-11-24T06:39:52Z") == G.parse_utc("2026-11-24T14:39:52+08:00")
    assert G.parse_utc("2026-11-24T06:39:52") == G.parse_utc("2026-11-24T06:39:52Z")


# ------------------------------------------------------------------------------------------------ rules
def test_nan_fires_on_any_finite_key(tmp_path):
    run = make_run(tmp_path)
    add_steps(run, range(1, 5))
    add_steps(run, [5], nfp=(float("nan"), 0.2))
    st = guard(run).check(T0)
    assert "nan" in st["alerts"] and "step 5: nfp_cos" in st["alerts"]["nan"]
    run2 = make_run(tmp_path / "b")
    add_steps(run2, [1], loss="inf")
    assert "nan" in guard(run2).check(T0)["alerts"]


def test_nonfp_dicts_without_cos_keys_pass(tmp_path):
    run = make_run(tmp_path, arm="nonfp")
    add_steps(run, range(1, 301), nfp=None)
    st = guard(run).check(T0)
    assert st["alerts"] == {} and st["head_on"] is False and st["nfp_cos_below_copy_last200"] is None


@pytest.mark.parametrize("n, below_at, fires", [(200, None, True), (199, None, False), (200, 150, False)])
def test_head_window(tmp_path, n, below_at, fires):
    run = make_run(tmp_path)
    for s in range(1, n + 1):
        add_steps(run, [s], nfp=(0.15, 0.2) if s == below_at else (0.5, 0.2))
    st = guard(run).check(T0)
    assert ("head" in st["alerts"]) is fires


def test_head_counts_only_steps_with_pairs_and_reads_incrementally(tmp_path):
    run = make_run(tmp_path)
    g = guard(run)
    add_steps(run, range(1, 101), nfp=(0.5, 0.2))
    add_steps(run, range(101, 151), nfp=(0.1, 0.2), pairs=0)  # no pairs: not counted, even if below
    assert g.check(T0)["alerts"] == {}
    add_steps(run, range(151, 251), nfp=(0.5, 0.2))
    st = g.check(T0)
    assert "head" in st["alerts"] and st["nfp_cos_below_copy_last200"] == [0, 200]


def test_head_fires_later_in_the_run_too(tmp_path):
    run = make_run(tmp_path)
    add_steps(run, range(1, 101), nfp=(0.1, 0.2))
    add_steps(run, range(101, 301), nfp=(0.3, 0.2))
    assert "head" in guard(run).check(T0)["alerts"]


def test_throughput_median(tmp_path):
    run = make_run(tmp_path)
    add_steps(run, range(1, 21), step_s=200.0)  # slow start before step 21: ignored
    assert "throughput" not in guard(run).check(T0)["alerts"]
    add_steps(run, range(21, 41), step_s=52.0)
    st = guard(run).check(T0)
    assert "throughput" in st["alerts"] and st["median_step_s_20"] == 52.0
    run2 = make_run(tmp_path / "b")
    add_steps(run2, range(1, 41), step_s=25.0)
    add_steps(run2, range(41, 50), step_s=500.0)  # 9 slow steps of 20: the median holds
    assert "throughput" not in guard(run2).check(T0)["alerts"]


def test_stall_limits(tmp_path):
    run = make_run(tmp_path)
    add_steps(run, range(1, 31), t0=T0 - 31 * 60 - 30 * 25.0)  # step 30 ended 31 min ago
    assert "no new step" in guard(run, "RUNNING").check(T0)["alerts"]["throughput"]
    assert "throughput" not in guard(run, "QUEUEING").check(T0)["alerts"]  # not running: no stall rule
    run2 = make_run(tmp_path / "b", save_steps=30)  # step 30 is a save step: 60 min
    add_steps(run2, range(1, 31), t0=T0 - 31 * 60 - 30 * 25.0)
    assert "throughput" not in guard(run2, "RUNNING").check(T0)["alerts"]
    run3 = make_run(tmp_path / "c")
    assert "throughput" not in guard(run3, "RUNNING", iso(T0 - 70 * 60)).check(T0)["alerts"]
    assert "no step" in guard(run3, "RUNNING", iso(T0 - 80 * 60)).check(T0)["alerts"]["throughput"]


def test_host_mem_used_not_cache(tmp_path):
    run = make_run(tmp_path)
    add_steps(run, range(1, 3))
    hm = run / "sft_metrics" / "host_mem_sft-20261124T000000Z.jsonl"
    rec = {"utc": iso(T0), "mem": {"MemTotal_gb": 2015.0, "used_gb": 300.0}, "cgroup": {"current_gb": 1900.0},
           "peak": {"used_gb": 300.0}}
    hm.write_text(json.dumps(rec) + "\n" + '{"utc": "2026-11-24T00:00:10Z", "mem": {"Mem' + "\n")  # a torn line
    g = guard(run)
    st = g.check(T0)
    assert st["alerts"] == {} and st["max_host_used_gb"] == 300.0 and st["bad_json_lines"] == 1
    with open(hm, "a") as f:
        f.write(json.dumps({**rec, "mem": {"used_gb": 1601.0}}) + "\n")
    assert "host_mem" in g.check(T0)["alerts"]


def write_copies(run, copies):
    out = []
    for name, status, ok, started in copies:
        log = run / "sft_metrics" / f"bf16_{name}.log"
        if ok is not None:
            log.write_text(json.dumps({"ok": ok, "write_seconds": 1480.0}, indent=1))
        out.append({"checkpoint": str(run / "ckpt" / name), "log": str(log), "started": iso(started),
                    "exit_status": status, "ended": None if status is None else iso(started + 1800)})
    (run / "sft_metrics" / "bf16_copies.json").write_text(json.dumps(out))


def save_ckpt(run, n):
    (run / "ckpt" / f"checkpoint-{n}").mkdir(parents=True)
    (run / "ckpt" / f"checkpoint-{n}" / "trainer_state.json").write_text("{}")


def test_save_rules(tmp_path):
    run = make_run(tmp_path, save_steps=10)
    add_steps(run, range(1, 11))
    assert guard(run).check(T0)["alerts"] == {}  # step 10's save is still being written
    add_steps(run, [11])
    assert "checkpoint-10 has no trainer_state" in guard(run).check(T0)["alerts"]["save"]
    save_ckpt(run, 10)
    write_copies(run, [("checkpoint-10", None, None, T0 - 50 * 60)])
    st = guard(run).check(T0)
    assert st["alerts"] == {} and st["warnings"] == ["bf16 copy of checkpoint-10 running 50 min"]
    write_copies(run, [("checkpoint-10", 1, None, T0 - 50 * 60)])
    assert "exit_status 1" in guard(run).check(T0)["alerts"]["save"]
    write_copies(run, [("checkpoint-10", 0, False, T0 - 50 * 60)])
    assert "ok false" in guard(run).check(T0)["alerts"]["save"]
    write_copies(run, [("checkpoint-10", 0, True, T0 - 50 * 60)])
    assert guard(run).check(T0)["alerts"] == {}
    # save_total_limit 1: checkpoint-10 is gone once checkpoint-20 is there
    add_steps(run, range(12, 22))
    (run / "ckpt" / "checkpoint-10" / "trainer_state.json").unlink()
    (run / "ckpt" / "checkpoint-10").rmdir()
    save_ckpt(run, 20)
    assert guard(run).check(T0)["alerts"] == {}


def test_end_states(tmp_path):
    run = make_run(tmp_path, save_steps=10, total=20)
    add_steps(run, range(1, 21), total=20)
    save_ckpt(run, 20)
    st = guard(run, "FAILED").check(T0)
    assert st["ended"] and "FAILED at step 20 of 20" in st["alerts"]["save"]
    assert "trainer_state.json is missing" in guard(run, "SUCCEEDED").check(T0)["alerts"]["save"]
    (run / "ckpt" / "trainer_state.json").write_text("{}")
    assert "bf16 copy of checkpoint-20" in guard(run, "SUCCEEDED").check(T0)["alerts"]["save"]
    write_copies(run, [("checkpoint-10", 0, True, T0 - 7200), ("checkpoint-20", 0, True, T0 - 3600)])
    st = guard(run, "SUCCEEDED").check(T0)
    assert st["ended"] and st["alerts"] == {}
    run2 = make_run(tmp_path / "b", save_steps=10, total=20)
    add_steps(run2, range(1, 15), total=20)
    save_ckpt(run2, 10)
    assert "step 14 of 20" in guard(run2, "SUCCEEDED").check(T0)["alerts"]["save"]
    # no bf16 copies asked for (or no saves): the end check does not want one
    run3 = make_run(tmp_path / "c", save_steps=10, total=20)
    add_steps(run3, range(1, 21), total=20)
    save_ckpt(run3, 20)
    (run3 / "ckpt" / "trainer_state.json").write_text("{}")
    (run3 / "q38_sft.json").write_text(json.dumps({"host_mem_s": 10}))
    assert guard(run3, "SUCCEEDED").check(T0)["alerts"] == {}


def test_progress_bar_total_wins(tmp_path):
    run = make_run(tmp_path, total=3441)
    add_steps(run, [1], total=3440)
    assert guard(run).check(T0)["total_steps"] == 3440


def test_final_step_waits_for_the_last_save(tmp_path):
    # the last step (3441) is no save step; the final full save, export, bf16 copy and chmod come after it
    run = make_run(tmp_path, save_steps=500, total=3441)
    add_steps(run, range(3420, 3442), t0=T0 - 3441 * 25.0 - 100 * 60)  # step 3441 ended 100 min ago
    st = guard(run, "RUNNING").check(T0)
    assert "throughput" not in st["alerts"], st["alerts"]
    run2 = make_run(tmp_path / "b", save_steps=500, total=3441)
    add_steps(run2, range(3420, 3442), t0=T0 - 3441 * 25.0 - 125 * 60)
    assert "limit 120" in guard(run2, "RUNNING").check(T0)["alerts"]["throughput"]
    run3 = make_run(tmp_path / "c", save_steps=500, total=3441)  # step 3440 is no final step: 30 min
    add_steps(run3, range(3420, 3441), t0=T0 - 3440 * 25.0 - 40 * 60)
    assert "limit 30" in guard(run3, "RUNNING").check(T0)["alerts"]["throughput"]
    assert G.stall_limit(3441, 500, 3441) == G.FINAL_STALL_MIN and G.stall_limit(3000, 500, 3441) == 60.0


def header(run, t):
    with open(run / "logs" / "node0.log", "a") as f:
        f.write(f"[q38s4 train] {iso(t)} host pt-x-master-0 node_rank 0 nnodes 8 commit abc\n")
        f.write(f"[q38s4 train] {iso(t)} host pt-x-worker-0 node_rank 1 nnodes 8 commit abc\n")


def train_begin(run, t):
    with open(run / "sft_metrics" / "rank0.jsonl", "a") as f:
        f.write(json.dumps({"event": "train_begin", "utc": iso(t), "rank": 0}) + "\n")


def test_resume_is_a_new_attempt(tmp_path):
    run = make_run(tmp_path, save_steps=500, total=3441)
    t_old = T0 - 12 * 3600  # the old attempt stepped until about 3.7 h ago
    header(run, t_old)
    train_begin(run, t_old)
    add_steps(run, range(1, 1201), t0=t_old, step_s=25.0, nfp=(0.5, 0.2))  # a bad head, then a crash at 1200
    add_steps(run, [1200], loss="nan", t0=t_old, nfp=(0.5, 0.2))
    save_ckpt(run, 1000)
    hm = run / "sft_metrics" / "host_mem_sft-old.jsonl"
    hm.write_text(json.dumps({"utc": iso(t_old), "mem": {"used_gb": 1700.0}}) + "\n")
    start = T0 - 20 * 60  # the resubmitted job started 20 minutes ago and is still loading
    g = guard(run, "RUNNING", iso(start))
    st = g.check(T0)
    assert set(st["alerts"]) >= {"nan", "head"}  # the old attempt's findings while only its log is there
    header(run, start + 5)
    st = g.check(T0)
    assert "nan" not in st["alerts"] and "head" not in st["alerts"] and "throughput" not in st["alerts"]
    assert "host_mem" not in st["alerts"] and st["attempts"] == 2, st["alerts"]
    # no step yet: the first-step limit of a resume is 120 min, not 75
    assert "throughput" not in guard(run, "RUNNING", iso(T0 - 100 * 60)).check(T0)["alerts"]
    assert "limit 120" in guard(run, "RUNNING", iso(T0 - 125 * 60)).check(T0)["alerts"]["throughput"]
    # the new attempt steps: train_begin clears the old step times; a slow first step alone does not fire
    train_begin(run, T0 - 10 * 60)
    add_steps(run, [1001], t0=T0 - 5 * 60 - 1001 * 400.0, step_s=400.0)
    st = g.check(T0)
    assert st["alerts"] == {} and st["last_step"] == 1001 and st["median_step_s_20"] == 400.0, st["alerts"]
    add_steps(run, range(1002, 1022), t0=T0 - 5 * 60 - 1001 * 25.0)
    st = g.check(T0)
    assert st["alerts"] == {} and st["median_step_s_20"] == 25.0


def test_old_step_events_do_not_stall_a_new_job(tmp_path):
    run = make_run(tmp_path)
    add_steps(run, range(1, 51), t0=T0 - 5 * 3600)  # an earlier job's steps, no header or train_begin
    st = guard(run, "RUNNING", iso(T0 - 10 * 60)).check(T0)
    assert "throughput" not in st["alerts"]


def test_warnings_suspended_describe_and_log_lag(tmp_path):
    run = make_run(tmp_path)
    add_steps(run, range(1, 5))
    st = guard(run, "SUSPENDED").check(T0)
    assert not st["ended"] and st["alerts"] == {} and "SUSPENDED" in st["warnings"][0]
    g = G.Guard(run, job="pt-x", describer=lambda j: {"state": None, "error": "timed out"})
    for _ in range(G.DESCRIBE_WARN - 1):
        assert g.check(T0)["warnings"] == []
    assert "describe failed 5 passes" in g.check(T0)["warnings"][0]
    with open(run / "sft_metrics" / "rank0.jsonl", "a") as f:  # rank 0 runs on, the log dicts do not come
        for s in range(5, 30):
            f.write(json.dumps({"event": "step", "step": s, "utc": iso(T0 - 60), "step_s": 25.0}) + "\n")
    with open(run / "logs" / "node0.log", "a") as f:
        f.write("{'loss': '1.0', 'grad_norm': '2.0'} no progress bar\n")
    st = guard(run).check(T0)
    assert any("rank 0 is at step 29" in w for w in st["warnings"]) and st["log_dicts_dropped"] == 1


def test_bad_bf16_started_time_is_no_crash(tmp_path):
    run = make_run(tmp_path, save_steps=10)
    add_steps(run, range(1, 11))
    save_ckpt(run, 10)
    (run / "sft_metrics" / "bf16_copies.json").write_text(json.dumps(
        [{"checkpoint": str(run / "ckpt" / "checkpoint-10"), "log": "x", "started": "soon", "exit_status": None}]))
    assert guard(run).check(T0)["alerts"] == {}


# ------------------------------------------------------------------------------------------------ stop + watch
class Runner:
    def __init__(self, rcs=(0,)):
        self.calls, self.rcs = [], list(rcs)

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        rc = self.rcs.pop(0) if self.rcs else 0

        class R:
            returncode, stdout, stderr = rc, "stopped" if rc == 0 else "", "" if rc == 0 else "error"
        return R()


class Desc:
    """Answers describe with the given states in turn (the last one repeats)."""

    def __init__(self, *states, name="q38s4-train-nfp-c_nfp"):
        self.states, self.name, self.n = list(states), name, 0

    def __call__(self, job):
        s = self.states[min(self.n, len(self.states) - 1)]
        self.n += 1
        return {"state": s, "display_name": self.name}


STOP = [G.SCO, "acp", "jobs", "stop", "--workspace-name=aigc", "pt-x"]


@pytest.mark.parametrize("state, name, stops", [("RUNNING", "q38s4-train-nfp-c_nfp", True),
                                                ("QUEUEING", "q38s4-train-nfp-c_nfp", False),
                                                ("RUNNING", "someone-else", False)])
def test_stop_job_describes_again(state, name, stops):
    r = Runner()
    desc = Desc(state, "STOPPED", name=name)
    rec = G.stop_job("pt-x", "q38s4-train-nfp-c_nfp", desc, r, sleep=lambda s: None)
    assert rec["stopped"] is stops
    assert (r.calls == [STOP]) is stops  # `sco acp jobs stop --workspace-name=aigc <jobid>` (its --help)
    if stops:
        assert rec["state_after"] == "STOPPED" and rec["tries"][0]["rc"] == 0


def test_stop_job_needs_a_name():
    r = Runner()
    rec = G.stop_job("pt-x", None, Desc("RUNNING", name="x"), r, sleep=lambda s: None)
    assert rec["stopped"] is False and r.calls == []


def test_stop_job_retries_and_verifies():
    slept = []
    r = Runner(rcs=(1, 1, 0))
    rec = G.stop_job("pt-x", "q38s4-train-nfp-c_nfp", Desc("RUNNING", "RUNNING", "RUNNING", "RUNNING", "STOPPED"), r,
                     sleep=slept.append, wait_s=30.0)
    assert rec["stopped"] is True and r.calls == [STOP] * 3 and [t["rc"] for t in rec["tries"]] == [1, 1, 0]
    assert slept == [30.0, 60.0, 30.0, 30.0]
    # sco says 0 but the job stays RUNNING: not stopped, and said so
    rec = G.stop_job("pt-x", "q38s4-train-nfp-c_nfp", Desc("RUNNING"), Runner(), sleep=lambda s: None, verify=3)
    assert rec["stopped"] is False and "has not left RUNNING" in rec["why"]
    # every try fails
    rec = G.stop_job("pt-x", "q38s4-train-nfp-c_nfp", Desc("RUNNING"), Runner(rcs=(1, 1, 1)), sleep=lambda s: None)
    assert rec["stopped"] is False and len(rec["tries"]) == 3 and "did not succeed" in rec["why"]
    # the job ends by itself between two tries: no further stop
    r = Runner(rcs=(1,))
    rec = G.stop_job("pt-x", "q38s4-train-nfp-c_nfp", Desc("RUNNING", "FAILED"), r, sleep=lambda s: None)
    assert r.calls == [STOP] and rec["state_after"] == "FAILED"

    def slow(cmd, **kw):
        raise G.subprocess.TimeoutExpired(cmd, 180)
    rec = G.stop_job("pt-x", "q38s4-train-nfp-c_nfp", Desc("RUNNING"), slow, sleep=lambda s: None)
    assert rec["stopped"] is False and rec["tries"][0]["out"] == "sco stop timed out"


def test_watch_alert_stops(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    add_steps(run, [1], loss="nan")
    calls = []
    monkeypatch.setattr(G, "stop_job", lambda job, name, desc: calls.append((job, name)) or {"stopped": True})
    rc, st = G.watch(guard(run, "RUNNING", iso(T0)), tmp_path / "out", 1, True, "q38s4-train-nfp-c_nfp",
                     sleep=lambda s: None)
    assert rc == 1 and calls == [("pt-x", "q38s4-train-nfp-c_nfp")]
    alert = json.loads((tmp_path / "out" / "ALERT.json").read_text())
    assert "nan" in alert["alerts"] and alert["stop"] == {"stopped": True}
    assert (tmp_path / "out" / "status.json").is_file() and (tmp_path / "out" / "guard.jsonl").is_file()


def test_watch_writes_the_alert_before_stopping(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    add_steps(run, [1], loss="nan")
    seen = []

    def stop(job, name, desc):
        seen.append(json.loads((tmp_path / "out" / "ALERT.json").read_text())["stop"])
        return {"stopped": False}
    monkeypatch.setattr(G, "stop_job", stop)
    rc, st = G.watch(guard(run, "RUNNING", iso(T0)), tmp_path / "out", 1, True, "n", sleep=lambda s: None)
    assert rc == 1 and seen == ["pending"]


def test_watch_without_stop_only_reports(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    add_steps(run, [1], loss="nan")
    monkeypatch.setattr(G, "stop_job", lambda *a: pytest.fail("must not stop"))
    rc, st = G.watch(guard(run, "RUNNING", iso(T0)), tmp_path / "out", 1, False, None, sleep=lambda s: None)
    assert rc == 1 and st["stop"] is None


def test_watch_end_and_passes(tmp_path):
    run = make_run(tmp_path, save_steps=10, total=10)
    add_steps(run, range(1, 11), total=10)
    (run / "ckpt" / "trainer_state.json").write_text("{}")
    save_ckpt(run, 10)
    write_copies(run, [("checkpoint-10", 0, True, T0)])
    rc, st = G.watch(guard(run, "SUCCEEDED"), tmp_path / "out", 1, True, "n", sleep=lambda s: None)
    assert rc == 0 and (tmp_path / "out" / "END.json").is_file()
    slept = []
    rc, st = G.watch(guard(run, "QUEUEING"), tmp_path / "o2", 7, True, "n", sleep=slept.append, max_passes=3)
    assert rc == 0 and slept == [7, 7] and len((tmp_path / "o2" / "guard.jsonl").read_text().splitlines()) == 3


class Bad:
    job = "pt-x"
    describer = None

    def __init__(self, fails):
        self.n, self.fails = 0, fails

    def check(self):
        self.n += 1
        if self.n <= self.fails:
            raise OSError("stale file handle")
        return {"alerts": {}, "ended": True}


def test_watch_keeps_going_through_bad_passes(tmp_path):
    rc, st = G.watch(Bad(2), tmp_path / "a", 1, True, "n", sleep=lambda s: None)
    assert rc == 0 and not (tmp_path / "a" / "GUARD_ERROR.json").exists()
    slept = []
    rc, st = G.watch(Bad(99), tmp_path / "b", 100, True, "n", sleep=slept.append, max_passes=8)
    assert rc == 0 and st["fails_in_a_row"] == 8 and (tmp_path / "b" / "GUARD_ERROR.json").is_file()
    assert slept == [100, 200, 400, 800, 800, 800, 800]  # growing pause, capped at 8 x interval (<= 900 s)
    rc, st = G.watch(Bad(6), tmp_path / "c", 1, True, "n", sleep=lambda s: None)
    assert rc == 0 and st == {"alerts": {}, "ended": True}
    assert not (tmp_path / "c" / "GUARD_ERROR.json").exists() and list((tmp_path / "c").glob("GUARD_ERROR_*.json"))


def test_describe_tolerates_bad_output():
    class R:
        def __init__(self, out):
            self.out = out

        def __call__(self, cmd, **kw):
            class X:
                returncode, stdout, stderr = 1, self.out, "err"
            return X()

    assert G.describe("pt-x", R("not json"))["state"] is None
    assert G.describe("pt-x", R(json.dumps({"state": "RUNNING", "display_name": "a"})))["state"] == "RUNNING"

    def slow(cmd, **kw):
        raise G.subprocess.TimeoutExpired(cmd, 120)
    assert G.describe("pt-x", slow)["state"] is None

    def gone(cmd, **kw):
        raise FileNotFoundError("sco")
    assert "sco describe" in G.describe("pt-x", gone)["error"]


def test_run_sco_kills_the_whole_group_on_timeout(tmp_path):
    pidfile = tmp_path / "child.pid"
    cmd = ["bash", "-c", f"sleep 60 & echo $! > {pidfile}; wait"]
    with pytest.raises(G.subprocess.TimeoutExpired):
        G.run_sco(cmd, timeout=2)
    pid = int(pidfile.read_text())
    import time

    def alive():
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return False
        return stat.rsplit(")", 1)[1].split()[0] != "Z"  # a killed child not yet reaped is a zombie
    for _ in range(50):
        if not alive():
            break
        time.sleep(0.1)
    else:
        pytest.fail("the background child of the timed-out command is still alive")
    assert G.run_sco(["bash", "-c", "echo hi"], timeout=10).stdout == "hi\n"


# ------------------------------------------------------------------------------------------------ run / job tie
def write_submits(run, recs):
    with open(run / "submits.jsonl", "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_submitted_jobs_and_the_watch_tie(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    write_submits(run, [
        {"utc": "x", "job_name": "q38s4-train-nfp-c_nfp", "rc": 0,
         "stdout": "job pt-aaa submitted successfully, please wait for scheduling!\njob id : pt-aaa \n"},
        {"utc": "y", "job_name": "q38s4-train-nfp-c_nfp", "rc": 1, "stdout": "job id : pt-bad \n"},
    ])
    assert G.submitted_jobs(run) == {"pt-aaa": "q38s4-train-nfp-c_nfp"}
    got = {}
    monkeypatch.setattr(G, "watch", lambda g, out, iv, stop, name: got.update(job=g.job, name=name) or (0, {}))
    with pytest.raises(SystemExit) as e:
        G.main(["watch", "--run", str(run), "--job", "pt-aaa", "--out", str(tmp_path / "o"), "--stop"])
    assert e.value.code == 0 and got == {"job": "pt-aaa", "name": "q38s4-train-nfp-c_nfp"}
    for argv in (["--job", "pt-other"], ["--job", "pt-bad"], ["--job", "pt-aaa", "--job-name", "q38s4-train-nonfp-c_nfp"]):
        with pytest.raises(SystemExit) as e:
            G.main(["watch", "--run", str(run), "--out", str(tmp_path / "o"), "--stop", *argv])
        assert e.value.code == 2  # argparse's error
