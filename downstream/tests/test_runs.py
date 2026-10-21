"""CPU tests of downstream/train/runs.py (step-3 run names, make / submit bookkeeping)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import runs  # noqa: E402


def test_names_and_grid():
    assert runs.exp_id("RS", "030", 2) == "DS-RS-030-s2"
    with pytest.raises(ValueError):
        runs.exp_id("B", "30", 1)
    g = runs.grid(["B", "S"], ["030", "010"], [1, 2])
    assert len(g) == 8 and g[0] == ("B", "030", 1) and g[-1] == ("S", "010", 2)


def test_make_command_keeps_only_the_final_checkpoint(tmp_path):
    cmd = runs.make_command("B", "010", 3, tmp_path)
    assert cmd[cmd.index("--steps") + 1] == "200000" and cmd[cmd.index("--save-interval") + 1] == "200000"
    assert cmd[cmd.index("--data") + 1] == str(tmp_path / "data/target50_010")
    assert cmd[cmd.index("--exp-id") + 1] == "DS-B-010-s3"


def test_make_skips_existing_and_reports_errors(tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stderr="start B changed", stdout="")

    exp = tmp_path / "experiments/DS-B-100-s2"
    exp.mkdir(parents=True)
    (exp / "experiment.json").write_text("{}")
    assert runs.make("B", "100", 2, tmp_path, run=fake_run)["note"] == "exists" and not calls
    out = runs.make("S", "100", 2, tmp_path, run=fake_run)
    assert out["made"] is False and "start B changed" in out["error"] and len(calls) == 1


class FakeJobs:
    def __init__(self):
        self.prepared = []

    def pick_pool(self):
        return "zoetrope"

    def prepare(self, run, gpus, pool, name, payload):
        self.prepared.append((run.name, gpus, pool, name, payload[-1]))

    def submit(self, run):
        (run / "job").mkdir(exist_ok=True)
        (run / "job/submission.json").write_text(json.dumps({"job_id": "pt-new"}))
        return "pt-new"


def test_submit_once_per_run(tmp_path):
    jobs = FakeJobs()
    assert "error" in runs.submit("B", "030", 1, tmp_path, jobs=jobs)
    exp = tmp_path / "experiments/DS-B-030-s1"
    exp.mkdir(parents=True)
    (exp / "experiment.json").write_text("{}")
    assert runs.submit("B", "030", 1, tmp_path, jobs=jobs) == {"exp": "DS-B-030-s1", "job": "pt-new", "pool": "zoetrope"}
    assert jobs.prepared == [("DS-B-030-s1", 8, "zoetrope", "ds-b-030-s1", str(exp))]
    assert runs.submit("B", "030", 1, tmp_path, jobs=jobs)["note"] == "already submitted" and len(jobs.prepared) == 1
