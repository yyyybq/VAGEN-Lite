from pathlib import Path

from eval_external.sco import job


def test_split_argv_keeps_job_options_out_of_the_command():
    opts, cmd = job.split_argv(["submit", "--run", "R", "--name", "n", "--", "python", "-m", "x", "--run-dir", "R"])
    assert opts == ["submit", "--run", "R", "--name", "n"]
    assert cmd == ["python", "-m", "x", "--run-dir", "R"]
    assert job.split_argv(["active"]) == (["active"], [])


def test_launch_script_runs_offline_in_the_worktree_and_hands_files_back():
    text = job.launch_script(Path("/tmp/run"), ["python", "-m", "m", "--bench", "VSI-Bench@lmms"], {"X": "a b"})
    assert f"cd {job.WORKTREE}" in text
    assert "export HF_HUB_OFFLINE=1" in text and "export HF_DATASETS_OFFLINE=1" in text
    assert f"export HF_HOME={job.HF_HOME}" in text
    assert "export X='a b'" in text
    assert "python -m m --bench VSI-Bench@lmms > /tmp/run/job/run.log 2>&1" in text
    assert f"chown -R {job.OWNER} /tmp/run" in text and "! -name hub" in text
    assert text.rstrip().endswith("exit $rc")
