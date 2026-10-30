"""downstream/eval/run_probe_episode.sh (2026-10-30): one policy server, then one closed_loop.py run per phase with
the probe on, the hang limits raised and no plan file (so no exclusion list), with stand-ins for the server, the
simulator environment and closed_loop.py."""
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_ENV = """SIM_PY=%s
STARVLA_ROOT=/nonexistent/starvla
sim() { shift; "$@"; }
""" % sys.executable
SERVER = """#!/usr/bin/env bash
python3 -c '
import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen(4); s.settimeout(1); end = time.time() + 30
while time.time() < end:
    try:
        s.accept()[0].close()
    except OSError:
        pass
' "$PORT"
"""
CLOSED_LOOP = """import json, os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True)
(out / "args.json").write_text(json.dumps({"argv": sys.argv[1:], "env": {k: v for k, v in os.environ.items()
                                            if k.startswith("CLOSED_LOOP_")}}))
"""


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def tree(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/eval").mkdir(parents=True)
    shutil.copy(ROOT / "eval/run_probe_episode.sh", wt / "downstream/eval/run_probe_episode.sh")
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    (wt / "downstream/eval/closed_loop.py").write_text(CLOSED_LOOP)
    (tmp_path / "server.sh").write_text(SERVER)
    env = dict(os.environ, PROBE_PORT=str(free_port()), SERVER_SCRIPT=str(tmp_path / "server.sh"))
    return wt, env


def test_two_phases_with_the_probe_on_and_no_plan(tmp_path):
    wt, env = tree(tmp_path)
    out = tmp_path / "out"
    res = subprocess.run(["bash", str(wt / "downstream/eval/run_probe_episode.sh"), "/ckpt.pt", str(out), "OpenDrawer",
                          "1012", "1", "1004-1016", "12"], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    for i, (seeds, workers) in enumerate((("1012", "1"), ("1004-1016", "12")), start=1):
        rec = json.loads((out / f"phase{i}/episodes/args.json").read_text())
        argv = rec["argv"]
        assert argv[argv.index("--seeds") + 1] == seeds and argv[argv.index("--workers") + 1] == workers
        assert argv[argv.index("--tasks") + 1] == "OpenDrawer" and "--eval-plan" not in argv
        assert rec["env"] == {"CLOSED_LOOP_PROBE_DIR": str(out / f"phase{i}/probe"), "CLOSED_LOOP_STALL_DUMP": "120",
                              "CLOSED_LOOP_STALL_SECONDS": "5400", "CLOSED_LOOP_HANG_LIMIT_SECONDS": "7200"}
    assert "phase 2: OpenDrawer seeds 1004-1016, 12 processes" in res.stdout


def test_phases_must_come_in_pairs(tmp_path):
    wt, env = tree(tmp_path)
    res = subprocess.run(["bash", str(wt / "downstream/eval/run_probe_episode.sh"), "/ckpt.pt", str(tmp_path / "out"),
                          "OpenDrawer", "1012"], capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 2 and "pairs" in res.stdout
