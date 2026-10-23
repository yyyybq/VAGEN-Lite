"""downstream/rlinf/speed_test2.sh (the user's decision 5 of 2026-10-23 evening) with a stand-in run_rlinf.sh that
records each launch: three launches, the new configuration (2 rollout epochs, 2 update epochs) at 64 and 32
environments."""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_speed_test2_runs_the_three_launches_of_the_new_configuration(tmp_path):
    rl = tmp_path / "wt/downstream/rlinf"
    rl.mkdir(parents=True)
    shutil.copy(ROOT / "rlinf/speed_test2.sh", rl)
    (rl / "run_rlinf.sh").write_text('#!/usr/bin/env bash\necho "$RLINF_TAG $*" >> "$2/calls.txt"\n')
    run = tmp_path / "RUN"
    run.mkdir()
    subprocess.run(["bash", str(rl / "speed_test2.sh"), "ENVDIR", str(run)], check=True, capture_output=True)
    calls = (run / "calls.txt").read_text().splitlines()
    assert [c.split()[0] for c in calls] == ["envs64", "full64", "full32"]
    assert all(c.split()[1:3] == ["ENVDIR", str(run)] and "runner.save_interval=100000" in c for c in calls)
    assert "env.train.total_num_envs=64" in calls[0] and "runner.max_steps=2" in calls[0]
    assert "rollout_epoch" not in calls[0]
    for c, n in ((calls[1], 64), (calls[2], 32)):
        assert f"env.train.total_num_envs={n}" in c and "env.train.rollout_epoch=2" in c
        assert "algorithm.update_epoch=2" in c and "actor.optim.lr=5e-6" in c and "actor.optim.clip_grad=1.0" in c
        assert "runner.max_epochs=1" in c and "runner.max_steps=1" in c and f"logs_full{n}" in c
    log = (run / "speed_test.log").read_text()
    assert log.count(" rc=0 ") == 3 and "[speed-test-2] done" in log
