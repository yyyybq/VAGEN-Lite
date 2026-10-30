"""CPU tests of the evaluation's worker pool loop (downstream/eval/closed_loop.run_all): an episode whose worker
process dies is run again, a hung worker is killed, an episode that keeps killing its worker is given up; and of the
resume helpers (completed_rows)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import closed_loop  # noqa: E402


class Result:
    def __init__(self, row=None):
        self.row = row

    def ready(self):
        return self.row is not None

    def get(self):
        return self.row


class FakePool:
    """apply_async follows a script per episode: "ok" returns a complete row at once; "die" leaves an in-flight marker
    of a process that is dead; "hang" leaves one of a live process started long ago."""

    def __init__(self, inflight: Path, script: dict):
        self.inflight, self.script, self.pid, self.dead, self.calls = inflight, script, 100, set(), []

    def apply_async(self, fn, args):
        e = args[0]
        self.calls.append((e["task"], e["seed"], e.get("attempt", 0)))
        what = self.script[(e["task"], e["seed"])].pop(0)
        if what == "ok":
            return Result(dict(e, status="complete", success=False, steps=10))
        self.pid += 1
        started = 0.0 if what == "hang" else 1e9
        (self.inflight / f"{self.pid}.json").write_text(json.dumps({"task": e["task"], "seed": e["seed"],
                                                                     "started": started, "max_steps": 100}))
        if what == "die":
            self.dead.add(self.pid)
        return Result()


def run(tmp_path, script, retries=2):
    inflight = tmp_path / "inflight"
    inflight.mkdir()
    pool = FakePool(inflight, script)
    rows = []

    def kill(pid, sig):
        pool.dead.add(pid)

    episodes = [{"task": t, "seed": s} for t, s in script]
    stats = closed_loop.run_all(pool, episodes, rows.append, inflight, retries=retries, poll=0,
                                alive=lambda pid: pid not in pool.dead, now=lambda: 1e9, kill=kill,
                                sleep=lambda s: None, run=lambda e: None)
    return pool, rows, stats


def test_a_dead_worker_episode_runs_again(tmp_path):
    pool, rows, stats = run(tmp_path, {("A", 1): ["ok"], ("B", 2): ["die", "ok"]})
    assert sorted((r["task"], r["status"]) for r in rows) == [("A", "complete"), ("B", "complete")]
    assert stats == {"retried": 1, "killed_hung": 0, "given_up": 0}
    assert ("B", 2, 1) in pool.calls and not list((tmp_path / "inflight").glob("*.json"))


def test_an_episode_that_keeps_killing_its_worker_is_given_up(tmp_path):
    pool, rows, stats = run(tmp_path, {("A", 1): ["ok"], ("C", 3): ["die", "die", "die"]})
    by = {r["task"]: r for r in rows}
    assert by["A"]["status"] == "complete" and by["C"]["status"] == "error" and "died 3 times" in by["C"]["error"]
    assert stats == {"retried": 2, "killed_hung": 0, "given_up": 1}


def test_a_hung_worker_is_killed_and_its_episode_run_again(tmp_path):
    pool, rows, stats = run(tmp_path, {("H", 4): ["hang", "ok"]})
    assert [r["status"] for r in rows] == ["complete"] and stats["killed_hung"] == 1 and stats["retried"] == 1


def test_hang_limit_and_completed_rows(tmp_path):
    assert closed_loop.hang_limit(4350) == 600 + 3 * 4350 and closed_loop.hang_limit(None) == 600
    p = tmp_path / "episodes.jsonl"
    p.write_bytes(b'{"task": "A", "seed": 1, "status": "complete"}\n{"task": "B", "seed": 2, "status": "error"}\n'
                  b'\xa8garbage\n{"task": "C", "seed": 3, "status": "complete"}\n')
    assert [r["task"] for r in closed_loop.completed_rows(p)] == ["A", "C"]
    assert closed_loop.completed_rows(tmp_path / "missing.jsonl") == []
    assert closed_loop.pid_alive(999999999) is False


def _episode(e):
    """A stand-in for run_episode in a real pool: marks itself in flight; the "die" episode kills its own worker the
    first time (as the memory killer would)."""
    import os
    import time

    marker = Path(e["inflight"]) / f"{os.getpid()}.json"
    marker.write_text(json.dumps({"task": e["task"], "seed": e["seed"], "started": time.time(), "max_steps": 10}))
    if e["task"] == "die" and not e.get("attempt"):
        os.kill(os.getpid(), 9)
    marker.unlink()
    return dict(e, status="complete")


def test_a_real_pool_reruns_the_episode_of_a_killed_worker(tmp_path):
    import multiprocessing as mp
    import os

    import pytest

    if not hasattr(os, "fork"):
        pytest.skip("needs fork")
    inflight = tmp_path / "inflight"
    inflight.mkdir()
    eps = [{"task": "ok", "seed": i, "inflight": str(inflight)} for i in range(3)] + \
        [{"task": "die", "seed": 9, "inflight": str(inflight)}]
    rows = []
    with mp.get_context("fork").Pool(2) as pool:
        stats = closed_loop.run_all(pool, eps, rows.append, inflight, poll=0.2, run=_episode)
    assert sorted(r["task"] for r in rows) == ["die", "ok", "ok", "ok"] and all(r["status"] == "complete" for r in rows)
    assert stats["retried"] == 1 and stats["given_up"] == 0


def test_memory_monitor_reads_the_cgroup_and_worker_peaks(tmp_path, monkeypatch):
    import os

    import pytest

    if not os.path.exists("/proc/self/status"):
        pytest.skip("needs /proc")
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text(str(1920 * 2**30))
    (cg / "memory.current").write_text(str(400 * 2**30))
    (cg / "memory.peak").write_text(str(500 * 2**30))
    (cg / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 1\noom_kill 1\n")
    files = {k: str(cg / f"memory.{k if k != 'limit' else 'max'}") for k in ("limit", "current", "peak", "events")}
    monkeypatch.setattr(closed_loop.MemoryMonitor, "V2", files)
    mon = closed_loop.MemoryMonitor(tmp_path / "memory.csv", pids=lambda: [os.getpid()])
    rows = []
    closed_loop.run_all(FakePool(tmp_path, {("A", 1): ["ok"]}), [{"task": "A", "seed": 1}], rows.append, tmp_path,
                        poll=0, sleep=lambda s: None, run=lambda e: None, monitor=mon)
    res = mon.result()
    assert res["cgroup_limit_bytes"] == 1920 * 2**30 and res["peak_bytes"] == 500 * 2**30
    assert res["peak_fraction"] == pytest.approx(500 / 1920, abs=1e-4) and res["oom_kills"] == 1
    assert res["workers_seen"] == 1 and res["worker_peak_max_bytes"] > 0 and res["samples"] == 1
    assert (tmp_path / "memory.csv").read_text().count("\n") == 2  # header + one sample


def test_hung_means_no_progress_for_ten_minutes_or_too_long_in_all():
    now = 10_000.0
    info = {"started": now - 700, "max_steps": 100}
    assert closed_loop.hung(info, now - 700, now)  # no policy call for 700 s > 600 (SearingMeat 1044, 2026-10-22)
    assert not closed_loop.hung(info, now - 30, now)  # a policy call 30 s ago: still working
    assert closed_loop.hung({"started": now - 1000, "max_steps": 100}, now - 5, now)  # 1000 s > 600 + 3 x 100
    assert not closed_loop.hung({"started": now - 100, "max_steps": 100}, now - 100, now)  # scene build, reset


def test_exclusion_list_is_applied_after_sharding(tmp_path):
    path = tmp_path / "excluded.json"
    path.write_text(json.dumps({"episodes": [["SearingMeat", 1044], ["DeliverStraw", 1006]]}))
    ex = closed_loop.excluded_episodes(path)
    assert ex["keys"] == {("SearingMeat", 1044), ("DeliverStraw", 1006)} and len(ex["sha256"]) == 64
    eps = [{"task": "SearingMeat", "seed": 1044}, {"task": "SearingMeat", "seed": 1045}, {"task": "DeliverStraw", "seed": 1006}]
    assert closed_loop.apply_exclusions(eps, ex["keys"]) == [{"task": "SearingMeat", "seed": 1045}]
    assert closed_loop.excluded_episodes(tmp_path / "missing.json")["keys"] == set()
    real = closed_loop.excluded_episodes()  # the list the user fixed on 2026-10-23 (5), LoadDishwasher 1002 added that
    # evening, OpenDrawer 1012 on 2026-10-30 (the user's decision: kick it out, find out separately why it hangs)
    assert len(real["keys"]) == 7 and {("StoreLeftoversInBowl", 1003), ("LoadDishwasher", 1002),
                                       ("OpenDrawer", 1012)} <= real["keys"]
    full, _ = closed_loop.eval_plan_episodes(ROOT / "eval/eval_d6.json")
    assert len(closed_loop.apply_exclusions(full, real["keys"])) == 943


def test_policy_servers_never_close_while_an_evaluation_runs():
    """The user's decision 4 of 2026-10-23 (evening): StarVLA's websocket server closes itself after 30 idle minutes
    by default, which broke late reruns (ConnectionClosedOK); run_eval_ckpt.sh starts every server with no idle
    watchdog and stops them itself when the evaluation ends."""
    script = (ROOT / "eval/run_eval_ckpt.sh").read_text()
    launch = script[script.index('run_starvla_mobile.sh" server'):].split("&", 1)[0]
    assert "--idle_timeout -1" in launch
    launcher = (ROOT.parent / "examples/evaluate/robocasa/run_starvla_mobile.sh").read_text()
    assert '--use_bf16 "$@"' in launcher  # the extra argument reaches server_policy.py
    assert "trap cleanup EXIT" in script
