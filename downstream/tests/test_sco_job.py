"""Tests of downstream/sco_job.py that never call sco."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sco_job  # noqa: E402


def test_prepare_writes_launch_and_receipt_fields(tmp_path):
    launch = sco_job.prepare(tmp_path / "run", 1, "h800", "dsload", ["bash", "payload.sh", "/out dir"])
    text = launch.read_text()
    assert "sco_run_as_artifact_owner.sh" in text
    assert "payload.sh '/out dir'" in text  # arguments are shell-quoted
    assert "exit_code" in text and "worker.log" in text
    prepared = json.loads((tmp_path / "run/job/prepared.json").read_text())
    assert prepared["spec"] == "N4lS.Iq.I80.1" and prepared["pool"] == "h800"
    assert prepared["payload"] == ["bash", "payload.sh", "/out dir"]
    assert "commit" in prepared["code"]


def test_as_root_runs_the_payload_without_the_owner_switch(tmp_path):
    launch = sco_job.prepare(tmp_path / "run", 1, "h800", "qa-qwen3", ["bash", "run.sh", "/out"], as_root=True)
    text = launch.read_text()
    assert "sco_run_as_artifact_owner.sh" not in text and "\nbash run.sh /out\n" in text
    assert json.loads((tmp_path / "run/job/prepared.json").read_text())["as_root"] is True
    sco_job.prepare(tmp_path / "run2", 1, "h800", "qa-qwen25", ["true"])
    assert "as_root" not in json.loads((tmp_path / "run2/job/prepared.json").read_text())


def test_main_prepare_takes_the_payload_after_double_dash(tmp_path, capsys):
    run = tmp_path / "run"
    assert sco_job.main(["prepare", "--run", str(run), "--gpus", "1", "--pool", "h800", "--name", "step1-load",
                         "--", "bash", "x.sh", "--flag", str(run / "out")]) == 0
    prepared = json.loads((run / "job/prepared.json").read_text())
    assert prepared["payload"] == ["bash", "x.sh", "--flag", str(run / "out")]
    assert prepared["name"] == "step1-load"


def test_create_command(tmp_path):
    sco_job.prepare(tmp_path / "run", 8, "zoetrope", "ds-rs-030-s1", ["true"])
    prepared = json.loads((tmp_path / "run/job/prepared.json").read_text())
    cmd = sco_job.create_command(tmp_path / "run/job", prepared)
    assert "--aec2-name=zoetrope" in cmd and "--worker-spec=N4lS.Iq.I80.8" in cmd
    assert "--job-name=downstream-ds-rs-030-s1" in cmd
    assert cmd[-1] == f"--command=bash {tmp_path / 'run/job'}/launch.sh"


@pytest.mark.parametrize("gpus,pool,name", [(2, "h800", "abc"), (1, "other", "abc"), (1, "h800", "Bad_Name")])
def test_prepare_rejects_bad_arguments(tmp_path, gpus, pool, name):
    with pytest.raises(ValueError):
        sco_job.prepare(tmp_path / "run", gpus, pool, name, ["true"])


def test_prepare_refuses_a_submitted_run(tmp_path):
    sco_job.prepare(tmp_path / "run", 1, "h800", "abc", ["true"])
    (tmp_path / "run/job/submission.json").write_text(json.dumps({"job_id": "pt-x"}))
    with pytest.raises(FileExistsError):
        sco_job.prepare(tmp_path / "run", 1, "h800", "abc", ["true"])
    assert sco_job.submit(tmp_path / "run") == "pt-x"  # an existing receipt is returned, never resubmitted
