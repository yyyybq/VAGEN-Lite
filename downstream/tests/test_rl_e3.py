"""downstream/rlinf/run_e3.sh (E3, the user's decisions of 2026-10-23 evening and 2026-10-26) with stand-ins for
RLinf (run_rlinf.sh), the exporter and the evaluation: train with the planned settings, keep only the last
checkpoint, export it, evaluate the export with 944 episodes; a relaunch resumes from the newest complete checkpoint
and skips the stages already done."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RLINF_STUB = """#!/usr/bin/env bash
echo "$RLINF_TAG $*" >> "$2/calls.txt"
for s in 25 50 75 100; do
  d="$2/logs/robocasa365_starvla_ppo/checkpoints/global_step_$s/actor"
  mkdir -p "$d/dcp_checkpoint" "$d/model_state_dict"
  touch "$d/dcp_checkpoint/.metadata" "$d/model_state_dict/full_weights.pt"
done
"""
EVAL_STUB = """#!/usr/bin/env bash
echo "eval $*" >> "$(dirname "$(dirname "$(dirname "$2")")")/calls.txt"
mkdir -p "$2/episodes"
echo '{"episodes": 944, "complete": 944, "successes": 300, "per_task": {}}' > "$2/episodes/summary.json"
"""
EXPORT_STUB = """import sys
from pathlib import Path
a = sys.argv[1:]
out = Path(a[a.index("--out") + 1])
(out / "checkpoints").mkdir(parents=True)
(out / "checkpoints/policy.pt").write_bytes(b"p")
(out / "rlinf_export.json").write_text("{}")
run = out.parent
with open(run / "calls.txt", "a") as f:
    f.write("export " + " ".join(a) + "\\n")
"""


def tree(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/rlinf").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    shutil.copy(ROOT / "rlinf/run_e3.sh", wt / "downstream/rlinf")
    (wt / "downstream/rlinf/run_rlinf.sh").write_text(RLINF_STUB)
    (wt / "downstream/eval/run_eval_ckpt.sh").write_text(EVAL_STUB)
    main = tmp_path / "main"
    (main / "scripts").mkdir(parents=True)
    (main / "scripts/export_rlinf_starvla.py").write_text(EXPORT_STUB)
    (tmp_path / "setup.sh").write_text("")
    run = tmp_path / "RUN"
    run.mkdir()
    return wt, run, {"MAIN_CHECKOUT": str(main), "STARVLA_SETUP": str(tmp_path / "setup.sh"), "EXPORT_PY": sys.executable,
                     "PATH": "/usr/bin:/bin"}


def launch(wt, run, env):
    return subprocess.run(["bash", str(wt / "downstream/rlinf/run_e3.sh"), "ENVDIR", str(run), "/src/policy.pt"],
                          capture_output=True, text=True, env=env)


def calls(run):
    return (run / "calls.txt").read_text().splitlines() if (run / "calls.txt").exists() else []


def test_a_fresh_run_trains_keeps_the_last_checkpoint_exports_and_evaluates(tmp_path):
    wt, run, env = tree(tmp_path)
    res = launch(wt, run, env)
    assert res.returncode == 0, res.stdout + res.stderr
    train, export, ev = calls(run)
    assert train.startswith("train_") and train.split()[1:3] == ["ENVDIR", str(run)]
    for o in ("runner.max_epochs=100", "runner.max_steps=100", "runner.save_interval=20", "runner.val_check_interval=-1",
              "env.train.total_num_envs=32", "env.train.rollout_epoch=2", "algorithm.update_epoch=2",
              "actor.optim.lr=5e-6", "actor.optim.clip_grad=1.0", "env.eval.total_num_envs=8",
              f"runner.logger.log_path={run}/logs"):
        assert o in train.split()
    assert "resume_dir" not in train
    ckpts = run / "logs/robocasa365_starvla_ppo/checkpoints"
    assert sorted(p.name for p in ckpts.iterdir()) == ["global_step_100"]  # only the last is kept
    final = ckpts / "global_step_100/actor/model_state_dict/full_weights.pt"
    assert export == f"export --weights {final} --source-ckpt /src/policy.pt --out {run}/export"
    assert ev == f"eval {run}/export/checkpoints/policy.pt {run}/evals/final/out 8 96"
    summary = json.loads((run / "e3_summary.json").read_text())
    assert summary["train_rc"] == summary["export_rc"] == summary["eval_rc"] == 0 and summary["eval"]["complete"] == 944
    assert launch(wt, run, env).returncode == 0 and len(calls(run)) == 3  # all done: nothing runs again


def test_a_relaunch_resumes_from_the_newest_complete_checkpoint_and_redoes_a_half_export(tmp_path):
    wt, run, env = tree(tmp_path)
    ckpts = run / "logs/robocasa365_starvla_ppo/checkpoints"
    for s in (25, 50):
        d = ckpts / f"global_step_{s}/actor"
        (d / "dcp_checkpoint").mkdir(parents=True)
        (d / "model_state_dict").mkdir(parents=True)
        (d / "dcp_checkpoint/.metadata").write_text("")
        (d / "model_state_dict/full_weights.pt").write_bytes(b"w")
    (ckpts / "global_step_75/actor/dcp_checkpoint").mkdir(parents=True)  # killed while saving: not complete
    (run / "export/checkpoints").mkdir(parents=True)  # an export killed half way (no rlinf_export.json)
    res = launch(wt, run, env)
    assert res.returncode == 0, res.stdout + res.stderr
    train = calls(run)[0]
    assert f"runner.resume_dir={ckpts}/global_step_50" in train.split()
    assert (run / "export/rlinf_export.json").exists() and "removed a half-written export" in (run / "e3.log").read_text()


def test_intervals_rlinf_would_reject_stop_before_any_card_is_used(tmp_path):
    """2026-10-26: checkpoints every 25 updates and evaluation every 10 failed all four E3 runs after their first
    update (RLinf asserts save_interval % val_check_interval == 0 when its evaluation is on); such intervals now stop
    at once.  With the evaluation off (-1, the default since 10-27) only the checkpoint interval must divide 100."""
    wt, run, env = tree(tmp_path)
    for bad in ({"E3_SAVE_EVERY": "25", "E3_EVAL_EVERY": "10"}, {"E3_SAVE_EVERY": "30"}):  # 25 % 10; 100 % 30
        res = subprocess.run(["bash", str(wt / "downstream/rlinf/run_e3.sh"), "ENVDIR", str(run), "/src/policy.pt"],
                             capture_output=True, text=True, env=dict(env, **bad))
        assert res.returncode == 6 and not calls(run)
    assert "must be a multiple of the evaluation interval" in (run / "e3.log").read_text()
    res = subprocess.run(["bash", str(wt / "downstream/rlinf/run_e3.sh"), "ENVDIR", str(run), "/src/policy.pt"],
                         capture_output=True, text=True, env=dict(env, E3_SAVE_EVERY="25"))
    assert res.returncode == 0 and "runner.save_interval=25" in calls(run)[0].split()


STALLED_STUB = """#!/usr/bin/env bash
echo "$RLINF_TAG $*" >> "$2/calls.txt"
printf 'Global Step:    1/100 | Progress\\r' > "$2/rlinf_$RLINF_TAG.log"
python3 -c 'import time; time.sleep(120)' e3_stub_hung_rlinf
"""


def test_the_watchdog_ends_a_training_without_new_updates(tmp_path):
    """2026-10-27: E3-B and E3-RS hung for two hours with the cards held; a training whose log shows no new
    "Global Step" for the stall time is stopped and the job ends with exit code 7 before export and evaluation."""
    wt, run, env = tree(tmp_path)
    (wt / "downstream/rlinf/run_rlinf.sh").write_text(STALLED_STUB)
    t = __import__("time").time()
    res = launch(wt, run, dict(env, E3_STALL_SEC="3", E3_WATCH_EVERY="1", E3_KILL_PATTERN="e3_stub_hung_rlinf"))
    assert res.returncode == 7, res.stdout + res.stderr
    assert __import__("time").time() - t < 60  # not the stub's 120 s
    stalled = list(run.glob("stalled_train_*.json"))
    assert len(stalled) == 1 and json.loads(stalled[0].read_text())["last_update"] == "1"
    log = (run / "e3.log").read_text()
    assert "watchdog: no new PPO update" in log and "ended by the watchdog" in log
    assert len(calls(run)) == 1 and not (run / "export").exists()


def test_a_training_that_ends_without_the_last_checkpoint_stops_before_the_export(tmp_path):
    wt, run, env = tree(tmp_path)
    (wt / "downstream/rlinf/run_rlinf.sh").write_text('#!/usr/bin/env bash\necho "$RLINF_TAG $*" >> "$2/calls.txt"\nexit 1\n')
    res = launch(wt, run, env)
    assert res.returncode == 4 and len(calls(run)) == 1 and not (run / "export").exists()
    assert "is missing: stop" in (run / "e3.log").read_text()
