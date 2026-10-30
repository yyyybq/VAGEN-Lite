"""closed_loop.Probe and the hang-limit overrides (2026-10-30, the user's request to find out why OpenDrawer seed 1012
keeps hanging its simulator process): off unless CLOSED_LOOP_PROBE_DIR is set; with it every reset, policy call and
simulator step is timed and logged with the contact count, and a stall dumps the Python stacks."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import closed_loop  # noqa: E402


def fake_env(ncon=7, qvel=(0.1, -2.5, 0.3), t=1.25):
    data = SimpleNamespace(ncon=ncon, qvel=np.array(qvel), time=t)
    return SimpleNamespace(unwrapped=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data))))


def test_probe_logs_every_phase_and_dumps_the_stack_of_a_stall(tmp_path, monkeypatch):
    monkeypatch.setattr(closed_loop, "PROBE_DUMP_SECONDS", 0.2)
    p = closed_loop.Probe(tmp_path / "probe", "OpenDrawer", 1012)
    p.arm()
    p.log("reset", 0, 81.5, fake_env())
    p.log("policy", 1, 0.31)
    p.arm()
    time.sleep(0.7)  # a "step" that stalls longer than the dump time
    p.log("step", 1, 0.7, fake_env(ncon=4000, qvel=(0.0, 950.0), t=1.27))
    p.close()
    csv = next((tmp_path / "probe").glob("OpenDrawer_s1012_p*.csv")).read_text().splitlines()
    assert csv[0] == "utc_seconds,phase,index,seconds,ncon,max_abs_qvel,sim_time" and len(csv) == 4
    reset, policy, step = (line.split(",") for line in csv[1:])
    assert reset[1:] == ["reset", "0", "81.5000", "7", "2.5", "1.25"]
    assert policy[1:] == ["policy", "1", "0.3100", "", "", ""]
    assert step[1:] == ["step", "1", "0.7000", "4000", "950.0", "1.27"]
    stacks = next((tmp_path / "probe").glob("stacks_*.log")).read_text()
    assert "=== OpenDrawer seed 1012" in stacks and "test_closed_loop_probe.py" in stacks  # the stalled frame


def test_the_probe_is_off_and_the_limits_unchanged_without_the_variables(monkeypatch):
    for v in ("CLOSED_LOOP_PROBE_DIR", "CLOSED_LOOP_HANG_LIMIT_SECONDS"):
        monkeypatch.delenv(v, raising=False)
    assert closed_loop.PROBE_DIR is None or closed_loop.PROBE_DIR == ""  # the test runs never set it
    assert closed_loop.hang_limit(500) == 600 + 3 * 500 and closed_loop.STALL_SECONDS == 600.0
    monkeypatch.setenv("CLOSED_LOOP_HANG_LIMIT_SECONDS", "7200")
    assert closed_loop.hang_limit(500) == 7200.0
