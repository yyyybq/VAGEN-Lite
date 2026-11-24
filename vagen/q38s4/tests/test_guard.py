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


# ------------------------------------------------------------------------------------------------ stop + watch
class Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)

        class R:
            returncode, stdout, stderr = 0, "stopped", ""
        return R()


@pytest.mark.parametrize("state, name, stops", [("RUNNING", "q38s4-train-nfp-c_nfp", True),
                                                ("QUEUEING", "q38s4-train-nfp-c_nfp", False),
                                                ("RUNNING", "someone-else", False)])
def test_stop_job_describes_again(state, name, stops):
    r = Runner()
    rec = G.stop_job("pt-x", "q38s4-train-nfp-c_nfp", lambda j: {"state": state, "display_name": name}, r)
    assert rec["stopped"] is stops
    assert (r.calls == [[G.SCO, "acp", "jobs", "stop", "--workspace-name=aigc", "pt-x"]]) is stops


def test_stop_job_needs_a_name():
    r = Runner()
    rec = G.stop_job("pt-x", None, lambda j: {"state": "RUNNING", "display_name": "x"}, r)
    assert rec["stopped"] is False and r.calls == []


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


def test_watch_survives_bad_passes_then_gives_up(tmp_path):
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

    rc, st = G.watch(Bad(2), tmp_path / "a", 1, True, "n", sleep=lambda s: None)
    assert rc == 0 and not (tmp_path / "a" / "GUARD_ERROR.json").exists()
    rc, st = G.watch(Bad(99), tmp_path / "b", 1, True, "n", sleep=lambda s: None)
    assert rc == 2 and st["fails_in_a_row"] == G.GUARD_FAILS and (tmp_path / "b" / "GUARD_ERROR.json").is_file()


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
