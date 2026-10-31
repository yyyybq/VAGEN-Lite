"""RLinf's own evaluation of E3's start (downstream/rlinf/rlinf_eval.py, run_rlinf_eval.sh; the user's decision 2 of
2026-10-31): the eval config from E3's train config, the metrics read from a log, and the job script's two runs
(a stand-in Python for RLinf)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlinf"))
import rlinf_eval as RE  # noqa: E402

TRAIN = {
    "hydra": {"run": {"dir": "."}, "output_subdir": None, "searchpath": ["file:///rlinf/examples/embodiment/config"]},
    "env": {"eval": {"rollout_epoch": 1, "total_num_envs": 32, "auto_reset": True, "max_episode_steps": 512,
                     "split": "target", "seed": 1042, "task_names": ["CoffeeSetupMug", "StackBowlsCabinet",
                                                                      "TurnOffStove"],
                     "task_sampling_strategy": "ordered",
                     "video_cfg": {"save_video": True, "video_base_dir": "${runner.logger.log_path}/video/eval"}}},
    "rollout": {"sampling_params": {"do_sample": False, "temperature_eval": 0.3}},
    "actor": {"model": {"model_path": "/ckpt.pt", "action_dim": 12, "num_action_chunks": 16,
                        "unnorm_key": "new_embodiment", "add_value_head": True, "action_stats_source": "minmax",
                        "starvla": {"framework_name": "QwenOFT", "expected_action_dim": "${actor.model.action_dim}",
                                    "expected_num_action_chunks": "${actor.model.num_action_chunks}",
                                    "enable_state_input": False},
                        "policy_setup": "robocasa365", "use_proprio": False}},
}


def test_eval_config_keeps_e3s_eval_block_and_runs_one_episode_per_env(tmp_path):
    cfg = RE.eval_config(TRAIN, tmp_path, envs=30)
    ev = cfg["env"]["eval"]
    assert ev["total_num_envs"] == 30 and ev["auto_reset"] is False and ev["rollout_epoch"] == 1
    assert ev["task_names"] == TRAIN["env"]["eval"]["task_names"] and ev["seed"] == 1042 and ev["split"] == "target"
    assert ev["video_cfg"]["video_base_dir"] == str(tmp_path / "video")
    model = cfg["rollout"]["model"]
    assert model["model_path"] == "/ckpt.pt" and model["add_value_head"] is False and model["policy_setup"] == "robocasa365"
    assert model["starvla"]["expected_action_dim"] == 12 and model["starvla"]["expected_num_action_chunks"] == 16
    assert cfg["runner"]["task_type"] == "embodied_eval" and cfg["runner"]["only_eval"] is True
    assert cfg["defaults"][:2] == ["env/robocasa365@env.eval", "model/starvla@rollout.model"]
    assert cfg["cluster"]["component_placement"] == {"env,rollout": "all"}
    assert TRAIN["env"]["eval"]["total_num_envs"] == 32  # E3's own config untouched
    text = yaml.safe_dump(cfg, sort_keys=False)
    assert "${actor" not in text  # no link to the actor block, which the eval config does not have


def test_metrics_read_from_the_runners_log_line():
    log = "noise\n[INFO] {'eval/success_once': 0.1, 'eval/episode_len': 400.0}\nmore\n" \
          "[INFO] {'eval/success_once': 0.2333, 'eval/num_episodes': 30}\n"
    assert RE.read_metrics(log) == {"eval/success_once": 0.2333, "eval/num_episodes": 30}
    assert RE.read_metrics("nothing here") is None


STUB = """import json, os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--config-path") + 1]).parent
fix = os.environ.get("DOWNSTREAM_RL_IMAGE_SIZE_FIX", "1")
(out / "seen.jsonl").open("a").write(json.dumps({"argv": sys.argv[1:], "fix": fix,
                                                 "hooks": os.environ.get("DOWNSTREAM_RL_HOOKS"),
                                                 "unbuffered": os.environ.get("PYTHONUNBUFFERED")}) + "\\n")
rate = 0.2333 if fix != "0" else 0.0333
print("[INFO RLinf] RLinf is running on a cluster with 1 node and 1 accelerator.")
print({"eval/success_once": rate, "eval/num_episodes": 30})
"""


def test_job_script_runs_fixed_then_control_and_reports(tmp_path):
    out = tmp_path / "out"
    (out / "cfg").mkdir(parents=True)
    (out / "cfg/eval.yaml").write_text("runner: {}\n")
    stub = tmp_path / "rlinf_stub.py"
    stub.write_text(STUB)
    wrapper = tmp_path / "py"
    wrapper.write_text(f"#!/usr/bin/env bash\nif [[ $1 == *eval_embodied_agent.py ]]; then shift; "
                       f"exec {sys.executable} {stub} \"$@\"; fi\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    env = dict(os.environ, ENV=str(tmp_path / "noenv"), RLINF_ROOT=str(tmp_path / "rlinf"), RLINF_PY=str(wrapper))
    res = subprocess.run(["bash", str(ROOT / "rlinf/run_rlinf_eval.sh"), str(out), "60"], capture_output=True,
                         text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    seen = [json.loads(x) for x in (out / "seen.jsonl").read_text().splitlines()]
    assert [s["fix"] for s in seen] == ["1", "0"] and all(s["hooks"] == "1" for s in seen)
    rep = json.loads((out / "report.json").read_text())
    assert rep["fixed"]["metrics"]["eval/success_once"] == 0.2333 and rep["control"]["rc"] == "0"
    again = subprocess.run(["bash", str(ROOT / "rlinf/run_rlinf_eval.sh"), str(out), "60"], capture_output=True,
                           text=True, env=env, timeout=120)
    assert again.returncode == 0 and len((out / "seen.jsonl").read_text().splitlines()) == 2  # done runs are kept
    assert seen[0]["unbuffered"] == "1"


STALLING = """import json, os, sys, time
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--config-path") + 1]).parent
fix = os.environ.get("DOWNSTREAM_RL_IMAGE_SIZE_FIX", "1")
(out / f"pid_{fix}.txt").write_text(str(os.getpid()))
if fix != "0":
    print("Started a local Ray instance.", flush=True)
    time.sleep(60)  # never gets to RLinf's cluster line
print("[INFO RLinf] RLinf is running on a cluster with 1 node and 1 accelerator.", flush=True)
print({"eval/success_once": 0.0333, "eval/num_episodes": 30})
"""


def test_a_run_without_rlinfs_cluster_line_is_diagnosed_and_stopped(tmp_path):
    out = tmp_path / "out"
    (out / "cfg").mkdir(parents=True)
    (out / "cfg/eval.yaml").write_text("runner: {}\n")
    stub = tmp_path / "rlinf_stub.py"
    stub.write_text(STALLING)
    wrapper = tmp_path / "py"
    wrapper.write_text(f"#!/usr/bin/env bash\nif [[ $1 == *eval_embodied_agent.py ]]; then shift; "
                       f"exec {sys.executable} {stub} \"$@\"; fi\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    ray_tmp = tmp_path / "ray"
    (ray_tmp / "session_latest/logs").mkdir(parents=True)
    (ray_tmp / "session_latest/logs/raylet.out").write_text("waiting\n")
    env = dict(os.environ, ENV=str(tmp_path / "noenv"), RLINF_ROOT=str(tmp_path / "rlinf"), RLINF_PY=str(wrapper),
               STALL="4", RAY_TMP=str(ray_tmp))
    res = subprocess.run(["bash", str(ROOT / "rlinf/run_rlinf_eval.sh"), str(out), "60"], capture_output=True,
                         text=True, env=env, timeout=50)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (out / "rc_fixed.txt").read_text().strip() == "stalled" and (out / "rc_control.txt").read_text().strip() == "0"
    assert (out / "diag_fixed/ps.txt").is_file() and (out / "diag_fixed/ray_logs/raylet.out").is_file()
    pid = int((out / "pid_1.txt").read_text())
    with pytest.raises(OSError):  # the stalled run's process is gone
        os.kill(pid, 0)
    assert not (out / "diag_control").exists()
