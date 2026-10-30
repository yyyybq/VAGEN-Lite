"""The E3 check (downstream/rlinf/e3_check.py, run_e3_check.sh; 2026-10-30): the two action paths, how often
they read the raw numbers differently, the episode list, the summary and report, and the job script's order of
steps (stand-ins for the server, the simulator environment and e3_check.py)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlinf"))
import e3_check as E  # noqa: E402
from test_run_probe_episode import SERVER, SIM_ENV, free_port  # noqa: E402

CANONICAL = {"robot0_right": (0, 6), "robot0_right_gripper": (6, 7), "robot0_base": (7, 10), "robot0_torso": (10, 11)}


def test_gym_path_thresholds_gripper_and_mode_and_places_parts_by_index():
    v = np.array([1, 2, 3, 4, 5, 6, 0.3, 7, 8, 9, 10, 0.2])
    flat = E.gym_flat(v, CANONICAL)
    assert flat.tolist() == [1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, -1]
    assert E.rlinf_flat(v).tolist() == v.tolist()  # RLinf: the numbers as they are
    other = {"robot0_base": (0, 3), "robot0_torso": (3, 4), "robot0_right": (4, 10), "robot0_right_gripper": (10, 11)}
    flat2 = E.gym_flat(np.array([1, 2, 3, 4, 5, 6, 0.9, 7, 8, 9, 10, 0.7]), other)
    assert flat2.tolist() == [7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 1, 1]


def test_raw_numbers_read_differently():
    assert E.read_differently([0] * 6 + [0.3] + [0] * 4 + [0.2]) == {"gripper": True, "mode": True}  # >0 but <0.5
    assert E.read_differently([0] * 6 + [0.7] + [0] * 4 + [0.9]) == {"gripper": False, "mode": False}
    assert E.read_differently([0] * 6 + [-0.2] + [0] * 4 + [-1.0]) == {"gripper": False, "mode": False}


def test_episode_list_covers_the_four_cells():
    eps = E.episodes_for([1000, 1001], states_dir="/s")
    assert len(eps) == 4 * 3 * 2
    assert {e["cell"] for e in eps} == {"gym/target", "gym/pretrain", "rlinf/target", "rlinf/pretrain"}
    assert all(e["max_steps"] == 512 and e["execute_steps"] == 16 for e in eps)
    assert len({(e["cell"], e["task"], e["seed"]) for e in eps}) == len(eps)


def row(cell, task, seed, success, steps=100, g=10, m=50, status="complete"):
    p, s = cell.split("/")
    return {"cell": cell, "path": p, "split": s, "task": task, "seed": seed, "status": status, "success": success,
            "steps": steps, "read_differently": {"gripper": g, "mode": m}}


def test_summary_and_report(tmp_path):
    rows = [row("gym/target", "CoffeeSetupMug", 1000, True), row("gym/target", "TurnOffStove", 1000, False),
            row("rlinf/pretrain", "CoffeeSetupMug", 1000, False), row("rlinf/pretrain", "CoffeeSetupMug", 1001, True),
            dict(row("rlinf/pretrain", "TurnOffStove", 1000, False), status="error")]
    s = E.summarize(rows)
    assert s["gym/target"]["all"]["successes"] == 1 and s["gym/target"]["all"]["n"] == 2
    assert s["gym/target"]["read_differently"] == {"gripper": 0.1, "mode": 0.5}
    assert s["rlinf/pretrain"]["errors"] == 1 and s["rlinf/pretrain"]["CoffeeSetupMug"]["rate"] == 0.5
    out = tmp_path / "out"
    (out / "run").mkdir(parents=True)
    retried = [row("gym/target", "CoffeeSetupMug", 1000, False, status="error"), row("gym/target", "CoffeeSetupMug", 1000, True)]
    (out / "run/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in retried))
    (out / "rejudge.json").write_text(json.dumps([{"task": "T", "seed": 1, "our_success": False},
                                                   {"task": "T", "seed": 2, "our_success": True}]))
    assert E.report_cmd(type("A", (), {"out": str(out)})) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["cells"]["gym/target"]["all"]["successes"] == 1  # the later row of a retried episode counts
    assert rep["rejudge"] == {"n": 2, "our_success": 1, "errors": 0}
    assert "| gym/target | 1/1 | 0/0 | 0/0 | 1/1（100%）" in (out / "report.md").read_text()


STUB = """import json, sys
from pathlib import Path
cmd = sys.argv[1]
out = Path(sys.argv[sys.argv.index("--out") + 1])
with open(out / "calls.jsonl", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if cmd == "obs":
    (out / "obs").mkdir(exist_ok=True)
    (out / "obs/frames.json").write_text("{}")
if cmd == "rejudge":
    (out / "rejudge.json").write_text("[]")
"""


def test_job_script_runs_the_steps_in_order(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/eval").mkdir(parents=True)
    (wt / "downstream/rlinf").mkdir(parents=True)
    shutil.copy(ROOT / "rlinf/run_e3_check.sh", wt / "downstream/rlinf/run_e3_check.sh")
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    (wt / "downstream/rlinf/e3_check.py").write_text(STUB)
    (tmp_path / "server.sh").write_text(SERVER)
    port = free_port()
    env = dict(os.environ, PROBE_PORT=str(port), SERVER_SCRIPT=str(tmp_path / "server.sh"))
    out = tmp_path / "out"
    res = subprocess.run(["bash", str(wt / "downstream/rlinf/run_e3_check.sh"), str(out), "/ckpt.pt", "1000-1001", "3"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    names = [c[0] for c in calls]
    assert sorted(names[:2]) == ["obs", "run"] and names[2:] == ["rejudge", "report"]
    run = next(c for c in calls if c[0] == "run")
    assert run[run.index("--servers") + 1] == f"127.0.0.1:{port}" and run[run.index("--seeds") + 1] == "1000-1001"
    assert run[run.index("--workers") + 1] == "3" and run[run.index("--ckpt") + 1] == "/ckpt.pt"
