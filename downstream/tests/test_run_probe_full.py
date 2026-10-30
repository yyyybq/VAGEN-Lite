"""downstream/eval/run_probe_full.sh (2026-10-30): SERVERS policy servers, then one closed_loop.py run of a plan with
the probe on, the hang limits raised and the exclusion list switched off, the machine's load sampled; with stand-ins
for the servers, the simulator environment and closed_loop.py."""
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

from test_run_probe_episode import CLOSED_LOOP, SERVER, SIM_ENV

ROOT = Path(__file__).resolve().parents[1]


def free_ports(n: int) -> int:
    """The first of n consecutive free ports."""
    for _ in range(50):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        base = s.getsockname()[1]
        s.close()
        ok = True
        for p in range(base, base + n):
            t = socket.socket()
            try:
                t.bind(("127.0.0.1", p))
            except OSError:
                ok = False
            finally:
                t.close()
        if ok:
            return base
    raise RuntimeError("no free ports")


def tree(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/eval").mkdir(parents=True)
    shutil.copy(ROOT / "eval/run_probe_full.sh", wt / "downstream/eval/run_probe_full.sh")
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    (wt / "downstream/eval/closed_loop.py").write_text(CLOSED_LOOP)
    (tmp_path / "server.sh").write_text(SERVER)
    env = dict(os.environ, PROBE_PORT=str(free_ports(2)), SERVER_SCRIPT=str(tmp_path / "server.sh"), LOAD_EVERY="1")
    return wt, env


def test_a_plan_under_load_with_the_probe_on_and_no_exclusions(tmp_path):
    wt, env = tree(tmp_path)
    out = tmp_path / "out"
    plan = tmp_path / "x3_plan.json"
    plan.write_text("{}")
    res = subprocess.run(["bash", str(wt / "downstream/eval/run_probe_full.sh"), "/ckpt.pt", str(out), str(plan), "2",
                          "3"], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    rec = json.loads((out / "episodes/args.json").read_text())
    argv = rec["argv"]
    assert argv[argv.index("--eval-plan") + 1] == str(plan)
    assert argv[argv.index("--exclusions") + 1] == ""  # closed_loop reads "" as no list
    assert argv[argv.index("--workers") + 1] == "3"
    port = int(env["PROBE_PORT"])
    assert argv[argv.index("--servers") + 1] == f"127.0.0.1:{port},127.0.0.1:{port + 1}"
    assert "--tasks" not in argv and "--seeds" not in argv
    assert rec["env"] == {"CLOSED_LOOP_PROBE_DIR": str(out / "probe"), "CLOSED_LOOP_STALL_DUMP": "120",
                          "CLOSED_LOOP_STALL_SECONDS": "1800", "CLOSED_LOOP_HANG_LIMIT_SECONDS": "10800"}
    load = (out / "load.csv").read_text().splitlines()
    assert load[0].startswith("utc_seconds,load1,") and len(load) >= 2 and len(load[1].split(",")) == 7
    assert "no exclusion list" in res.stdout
