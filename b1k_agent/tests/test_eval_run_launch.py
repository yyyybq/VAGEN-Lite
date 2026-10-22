"""eval/launch.py on a FAKE server-105 layout: command construction (agent and direct mode), max_steps from a
synthetic task.jsonl, infra classification, and the supervisor with fake executor / agent / evaluator processes
(small python scripts that answer /healthz and write official-style result JSONs). No simulator, no GPU."""

import ast
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from conftest import ROOT
from eval import gpu, launch

L0 = os.path.join(ROOT, "configs", "L0_exec_only.yaml")
# nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,name,uuid (GPU 1 busy, 0 and 2 free)
GPUS = gpu.parse_gpu_csv("0, 3, 81920, 0, A100, GPU-a\n1, 40000, 81920, 95, A100, GPU-b\n2, 10, 81920, 1, A100, GPU-c\n")
ASSET_ID = "behavior-1k/2026-challenge-demos"


def human_stats(task_index, task_name, length):
    return {"task_index": task_index, "task_name": task_name, "num_episodes": 200, "length": length,
            "distance_traveled": 12.5, "left_eef_displacement": 3.25, "right_eef_displacement": 4.0}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def site(tmp_path):
    """Directory layout of server 105 with fake contents (evaluator tree, metadata, checkpoint, envs)."""
    b1k = tmp_path / "BEHAVIOR-1K"
    (b1k / "OmniGibson" / "omnigibson" / "eval").mkdir(parents=True)
    (b1k / "OmniGibson" / "omnigibson" / "eval" / "eval.py").write_text("# stand-in for the official entry point\n")
    meta = b1k / "datasets" / "2026-challenge-task-instances" / "metadata"
    meta.mkdir(parents=True)
    (meta / "B100_task_misc.csv").write_text(
        'Task,Task ID,Rooms to inlcude\nturning_on_radio,0,living_room\npicking_up_trash,1,"kitchen\nliving_room"\n')
    (meta / "task.jsonl").write_text(json.dumps(human_stats(0, "turning_on_radio", 1000.5)) + "\n")
    ckpt = tmp_path / "ckpt"
    (ckpt / "assets" / ASSET_ID).mkdir(parents=True)
    (ckpt / "assets" / ASSET_ID / "norm_stats.json").write_text("{}")
    openpi = tmp_path / "openpi"
    (openpi / ".venv" / "bin").mkdir(parents=True)
    executable(openpi / ".venv" / "bin" / "python", "#!/bin/sh\n")
    env = tmp_path / "envs" / "behavior"
    env.mkdir(parents=True)
    conda = executable(tmp_path / "conda", "#!/bin/sh\n")
    overrides = [f"paths.b1k_root={b1k}", f"paths.openpi_root={openpi}", f"paths.behavior_env={env}",
                 f"executor.params.checkpoint_dir={ckpt}", f"run.log_root={tmp_path / 'nas_logs'}",
                 f"run.local_log_root={tmp_path / 'local_logs'}",
                 f"task_metadata.task_misc_csv={meta / 'B100_task_misc.csv'}"]

    class Site:
        pass

    s = Site()
    s.tmp, s.b1k, s.meta, s.ckpt, s.openpi, s.env, s.conda = tmp_path, b1k, meta, ckpt, openpi, env, conda
    s.overrides = overrides
    s.environ = {"PATH": os.environ.get("PATH", ""), "CONDA_EXE": conda, "HOME": str(tmp_path)}
    s.ports = (free_port(), free_port())
    return s


def launch_argv(site, *extra, instances=("10",), run_id="t-run", task="turning_on_radio"):
    argv = ["--config", L0]
    for ov in site.overrides:
        argv += ["-o", ov]
    argv += ["--task", task, "--instances", *instances, "--wrapper", "default", "--seed", "3",
             "--run-id", run_id, "--executor-port", str(site.ports[0]), "--agent-port", str(site.ports[1])]
    return argv + list(extra)


def plan_for(site, *extra, environ=None, gpus=GPUS, **kw):
    args = launch.parse_launch_args(launch_argv(site, *extra, **kw))
    return launch.build_plan(args, environ=dict(site.environ if environ is None else environ), gpu_query=lambda: gpus)


def exec_line(plan):
    line = next(x for x in plan.evaluator_script.splitlines() if x.startswith("exec "))
    return shlex.split(line)[1:]


def flag(argv, name):
    return argv[argv.index(name) + 1]


# ---------------------------------------------------------------------- official mirrors


def test_official_max_steps_matches_evaluator_formula(tmp_path):
    p = tmp_path / "task.jsonl"
    p.write_text("\n".join([
        json.dumps(human_stats(5, "picking_up_trash", 2000.0)),
        "",
        json.dumps(human_stats(0, "turning_on_radio", 1000.5)),
        json.dumps(human_stats(0, "turning_on_radio", 9999.0)),  # first match wins, as in score_utils
        json.dumps(human_stats(7, "renamed_task", 333.3)),
    ]) + "\n")
    assert launch.official_max_steps("turning_on_radio", str(p)) == 1500  # int(1000.5 * 1.5) = int(1500.75)
    assert launch.official_max_steps("picking_up_trash", str(p)) == 3000
    assert launch.official_max_steps("whatever", str(p), task_index=7) == int(333.3 * 1.5) == 499  # matched by task_index
    assert launch.load_human_stats("turning_on_radio", str(p)) == {
        "length": 1000.5, "distance_traveled": 12.5, "left_eef_displacement": 3.25, "right_eef_displacement": 4.0}
    with pytest.raises(ValueError, match="Could not find human stats"):
        launch.official_max_steps("missing_task", str(p))
    (tmp_path / "bad.jsonl").write_text(json.dumps({"task_name": "t", "task_index": 1, "length": 10}) + "\n")
    with pytest.raises(KeyError):  # score_utils requires all four HUMAN_STATS_KEYS
        launch.official_max_steps("t", str(tmp_path / "bad.jsonl"))


def test_resolve_instance_ids_mirrors_evaluator():
    assert launch.resolve_instance_ids([0, 9, 10, 19], "public_test") == [301, 310, 311, 320]
    assert launch.resolve_instance_ids([0, 19], "hidden_test") == [321, 340]
    assert launch.resolve_instance_ids([7, 1234], "train") == [7, 1234]
    with pytest.raises(ValueError):
        launch.resolve_instance_ids([20], "public_test")


def test_detect_repo_id(tmp_path):
    (tmp_path / "assets" / "a" / "b").mkdir(parents=True)
    assert launch.detect_repo_id(str(tmp_path)) is None
    (tmp_path / "assets" / "a" / "b" / "norm_stats.json").write_text("{}")
    assert launch.detect_repo_id(str(tmp_path)) == "a/b"
    (tmp_path / "assets" / "c").mkdir()
    (tmp_path / "assets" / "c" / "norm_stats.json").write_text("{}")
    with pytest.raises(launch.PlanError, match="several"):
        launch.detect_repo_id(str(tmp_path))


# ---------------------------------------------------------------------- command construction


def test_plan_agent_mode_commands(site):
    plan = plan_for(site, "--write-video")
    exec_port, agent_port = site.ports
    assert plan.instance_ids == [311] and plan.expected_results == ["turning_on_radio_311_0.json"]
    assert plan.max_steps == 1500 and plan.max_steps_info["source"] == "task.jsonl"
    assert plan.max_steps_info["to_evaluator"] is False
    assert plan.gpus == {"evaluator": [0], "executor": [2], "agent": []}
    assert plan.run_dir == str(site.tmp / "nas_logs" / "t-run")
    assert [p.name for p in plan.processes] == ["executor", "agent", "evaluator"]

    ex = plan.proc("executor")
    assert ex.argv[:2] == [str(site.openpi / ".venv" / "bin" / "python"),
                           os.path.join(ROOT, "executor", "openpi_server", "serve_b1k_prompted.py")]
    assert ex.cwd == str(site.openpi)
    for name, value in (("--robot", "b1k/R1Pro"), ("--task", "b1k/turning_on_radio"), ("--repo-id", ASSET_ID),
                        ("--policy.config", "pi05_b1k"), ("--policy.dir", str(site.ckpt)),
                        ("--control-mode", "receding_horizon"), ("--action-horizon", "16"), ("--port", str(exec_port)),
                        ("--rename-robot-from", "robot_r1")):
        assert flag(ex.argv, name) == value, name
    assert ex.env["CUDA_VISIBLE_DEVICES"] == "2" and ex.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert ex.health_url == f"http://127.0.0.1:{exec_port}/healthz"

    ag = plan.proc("agent")
    assert ag.argv[1:3] == ["-m", "agent.server"] and ag.cwd == ROOT
    assert flag(ag.argv, "--config") == L0 and flag(ag.argv, "--port") == str(agent_port)
    assert flag(ag.argv, "--run-id") == "t-run" and flag(ag.argv, "--max-steps") == "1500"
    tags = [ag.argv[i + 1] for i, a in enumerate(ag.argv) if a == "--tag"]
    for t in ("task=turning_on_radio", "instance_ids=311", "instance_indices=10", "seed=3", "wrapper=default"):
        assert t in tags
    overrides = [ag.argv[i + 1] for i, a in enumerate(ag.argv) if a == "-o"]
    assert f"executor.params.port={exec_port}" in overrides and "run.seed=3" in overrides
    assert ag.env["CUDA_VISIBLE_DEVICES"] == ""  # the agent never gets a GPU
    assert ag.health_url == f"http://127.0.0.1:{agent_port}/healthz"

    ev = plan.proc("evaluator")
    assert ev.argv == ["bash", os.path.join(plan.run_dir, "run_evaluator.sh")] and ev.health_url is None
    cmd = exec_line(plan)
    assert cmd[:6] == [site.conda, "run", "-p", str(site.env), "--no-capture-output", "python"]
    assert cmd[6:8] == ["-m", "omnigibson.eval.eval"]
    eargs = cmd[8:]
    assert flag(eargs, "--task-name") == "turning_on_radio" and flag(eargs, "--port") == str(agent_port)
    assert flag(eargs, "--host") == "127.0.0.1" and flag(eargs, "--mode") == "public_test"
    assert flag(eargs, "--instance-indices") == "10" and flag(eargs, "--num-envs") == "1"
    assert flag(eargs, "--env-wrapper") == "omnigibson.eval.wrappers.DefaultWrapper"
    assert flag(eargs, "--output-dir") == os.path.join(plan.run_dir, "eval_output")
    assert "--write-video" in eargs and "--max-steps" not in eargs  # official default timeout
    script = plan.evaluator_script
    for line in ("export OMNIGIBSON_HEADLESS=1", "export OMNI_KIT_ACCEPT_EULA=YES", "export CUDA_VISIBLE_DEVICES=0",
                 f"export OMNIGIBSON_DATA_PATH={site.b1k / 'datasets'}", f"cd {site.b1k}"):
        assert line in script.splitlines(), line
    assert "PYTHONPATH" not in script and "source" not in script  # installed tree, no B1K_ENV_SETUP


def test_plan_direct_mode_is_evaluator_to_executor(site):
    plan = plan_for(site, "--direct", "--wrapper", "rgbd")
    assert [p.name for p in plan.processes] == ["executor", "evaluator"]
    assert plan.direct and plan.config_name == "L0_direct" and plan.ports["agent"] is None
    eargs = exec_line(plan)
    assert flag(eargs, "--port") == str(site.ports[0])  # straight to the executor server
    assert flag(eargs, "--env-wrapper") == "omnigibson.eval.wrappers.RGBDFullResWrapper"
    assert flag(plan.proc("executor").argv, "--rename-robot-from") == "robot_r1"
    assert plan_for(site, "--direct", "--wrapper", "rgbd", run_id="t2").config_hash == plan.config_hash
    assert plan.agent_manifest == {} and "--max-steps" not in eargs


def test_plan_without_task_jsonl_needs_max_steps_unless_direct(site):
    (site.meta / "task.jsonl").unlink()
    with pytest.raises(launch.PlanError, match="--max-steps"):
        plan_for(site)
    direct = plan_for(site, "--direct")
    assert direct.max_steps is None and any("max_steps unknown" in w for w in direct.warnings)
    dry = plan_for(site, "--dry-run")
    assert any("--max-steps" in w for w in dry.warnings) and "--max-steps" not in dry.proc("agent").argv


def test_plan_max_steps_flags(site):
    cli = plan_for(site, "--max-steps", "300")
    assert flag(cli.proc("agent").argv, "--max-steps") == "300" and flag(exec_line(cli), "--max-steps") == "300"
    assert cli.max_steps_info["to_evaluator"] and any("not the official default" in w for w in cli.warnings)
    agent_only = plan_for(site, "--agent-max-steps", "200")
    assert flag(agent_only.proc("agent").argv, "--max-steps") == "200" and "--max-steps" not in exec_line(agent_only)
    with pytest.raises(launch.PlanError, match="exclusive"):
        launch.parse_launch_args(launch_argv(site, "--max-steps", "1", "--agent-max-steps", "2"))


def test_plan_no_executor_uses_external_server(site):
    plan = plan_for(site, "--no-executor")
    assert [p.name for p in plan.processes] == ["agent", "evaluator"]
    assert plan.external_health == [f"http://127.0.0.1:{site.ports[0]}/healthz"]
    assert plan.gpus == {"evaluator": [0], "executor": [], "agent": []}


def test_plan_evaluator_worktree_env_setup_and_prefix(site):
    wt = site.tmp / "b1k-2026-eval"
    (wt / "OmniGibson" / "omnigibson" / "eval").mkdir(parents=True)
    (wt / "OmniGibson" / "omnigibson" / "eval" / "eval.py").write_text("")
    setup = site.tmp / "b1k_runs_env.sh"
    setup.write_text("export LD_PRELOAD=/opt/libstdc++.so.6\n")
    env = {**site.environ, "B1K_ENV_SETUP": str(setup)}
    plan = plan_for(site, "--evaluator-root", str(wt), "--evaluator-cmd-prefix", "bash /nas/b1k_runs/run_helper.sh --x",
                    environ=env)
    lines = plan.evaluator_script.splitlines()
    src = next(i for i, x in enumerate(lines) if x.startswith(f"source {setup}"))
    assert src < lines.index(f"cd {wt}") < next(i for i, x in enumerate(lines) if x.startswith("exec "))
    assert f"export PYTHONPATH={wt / 'OmniGibson'}${{PYTHONPATH:+:$PYTHONPATH}}" in plan.evaluator_script
    assert plan.proc("evaluator").cwd == str(wt) and plan.evaluator_root == str(wt)
    assert exec_line(plan)[5:10] == ["bash", "/nas/b1k_runs/run_helper.sh", "--x", "python", "-m"]
    with pytest.raises(launch.PlanError, match="B1K_ENV_SETUP"):
        plan_for(site, environ={**site.environ, "B1K_ENV_SETUP": str(site.tmp / "missing.sh")})


def test_plan_activation_modes(site):
    plain = plan_for(site, "--activation", "none", "--evaluator-python", "/opt/behavior/bin/python")
    assert exec_line(plain)[:3] == ["/opt/behavior/bin/python", "-m", "omnigibson.eval.eval"]
    act = plan_for(site, "--activation", "activate")
    assert f"conda activate {site.env}" in act.evaluator_script
    assert exec_line(act)[:3] == ["python", "-m", "omnigibson.eval.eval"]
    with pytest.raises(launch.PlanError, match="conda executable"):
        plan_for(site, environ={"PATH": str(site.tmp / "empty"), "HOME": str(site.tmp)})


def test_plan_gpu_discipline(site):
    busy = gpu.parse_gpu_csv("0, 30000, 81920, 99, A100, GPU-a\n1, 10, 81920, 50, A100, GPU-b\n")
    with pytest.raises(launch.PlanError, match="nothing started"):
        plan_for(site, gpus=busy)
    dry = plan_for(site, "--dry-run", gpus=busy)
    assert dry.gpus["evaluator"] == ["<gpu0>"] and any("GPU check" in w for w in dry.warnings)
    shared = plan_for(site, "--share-gpu")
    assert shared.gpus == {"evaluator": [0], "executor": [0], "agent": []}
    assert plan_for(site, "--gpus", "2,0").gpus == {"evaluator": [2], "executor": [0], "agent": []}
    with pytest.raises(launch.PlanError, match="not free"):
        plan_for(site, "--gpus", "0,1")
    with pytest.raises(launch.PlanError, match="needs 2"):
        plan_for(site, "--gpus", "0")
    assert plan_for(site, "--gpus", "0,1", "--skip-gpu-check").gpus["executor"] == [1]  # eval.batch checked them


def test_plan_rejections(site):
    with pytest.raises(launch.PlanError, match="run id"):
        plan_for(site, run_id="bad id/..")
    with pytest.raises(launch.PlanError, match="0..19"):
        plan_for(site, instances=("20",))
    with pytest.raises(launch.PlanError, match="not in"):
        plan_for(site, task="not_a_task")
    with socket.socket() as s:
        s.bind(("127.0.0.1", site.ports[0]))
        s.listen()
        with pytest.raises(launch.PlanError, match="executor port"):
            plan_for(site)
    run_dir = site.tmp / "nas_logs" / "t-run"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text("{}")
    with pytest.raises(launch.PlanError, match="already holds a run"):
        plan_for(site)


def test_dry_run_cli_prints_commands_and_writes_nothing(site, capsys):
    rc = launch.main(launch_argv(site, "--dry-run", "--direct"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "serve_b1k_prompted.py" in out and "omnigibson.eval.eval" in out and "agent.server" not in out
    assert not (site.tmp / "nas_logs").exists()


def test_prepare_run_dir_writes_status_commands_and_plan(site):
    plan = plan_for(site)
    launch.prepare_run_dir(plan)
    st = launch.read_status(plan.run_dir)
    assert st["state"] == "planned" and st["eval_output_dir"] == plan.eval_output_dir and st["gpus"] == plan.gpus
    assert st["instance_ids"] == [311] and st["max_steps"] == 1500 and st["agent_manifest"]["nas"].endswith(
        "t-run/run_manifest.json")
    commands = open(os.path.join(plan.run_dir, "commands.sh")).read()
    assert "agent.server" in commands and "serve_b1k_prompted.py" in commands and "run_evaluator.sh" in commands
    assert launch.load_plan(plan.run_dir) == plan


# ---------------------------------------------------------------------- outcome


EXPECTED = ["t_301_0.json", "t_302_0.json"]


@pytest.mark.parametrize("kw, state, kinds", [
    (dict(evaluator_exit=0, found=EXPECTED), "done", []),
    (dict(evaluator_exit=0, found=EXPECTED, failure_reasons=["group_time_budget_exceeded", None]), "done", []),
    (dict(evaluator_exit=0, found=EXPECTED, failure_reasons=["policy_connection_lost", None]), "infra_error",
     ["connection_lost"]),
    (dict(evaluator_exit=1, found=[]), "infra_error", ["no_results", "exit_code"]),
    (dict(evaluator_exit=0, found=[]), "infra_error", ["no_results"]),
    (dict(evaluator_exit=0, found=EXPECTED[:1]), "infra_error", ["missing_results"]),
    (dict(evaluator_exit=-11, found=EXPECTED), "infra_error", ["exit_code"]),
    (dict(evaluator_exit=-11, found=EXPECTED, strict_exit=False), "done", []),
    (dict(evaluator_exit=1, found=EXPECTED[:1], strict_exit=False), "infra_error", ["missing_results", "exit_code"]),
    (dict(evaluator_exit=0, found=EXPECTED, servers_died=["agent"]), "infra_error", ["server_died"]),
    (dict(evaluator_exit=0, found=EXPECTED, died_after_exit=["agent"]), "done", []),
    (dict(evaluator_exit=None, found=EXPECTED, timed_out=True), "infra_error", ["timeout"]),
])
def test_classify_outcome(kw, state, kinds):
    out = launch.classify_outcome(expected=EXPECTED, **kw)
    assert out["state"] == state and out["kinds"] == kinds
    assert bool(out["reasons"]) == (state == "infra_error")


def test_policy_failure_is_never_infra():
    """success False / q 0 / a 2026 budget failure are policy outcomes: complete JSONs + clean exit = done."""
    out = launch.classify_outcome(evaluator_exit=0, expected=EXPECTED, found=EXPECTED,
                                  failure_reasons=["policy_query_timeout", "policy_error"])
    assert out["state"] == "done"


def test_parse_evaluator_log_official_lines():
    lines = [  # format strings of OmniGibson/omnigibson/eval/{eval,evaluator}.py (v3.9.3-post2 and 2026/eval)
        "[INFO] Resolved public_test instance ids for turning_on_radio: [311, 312]",
        "[INFO] Setting timeout to be 1.5x the average length of human demos: 1500",
        "[INFO] Instance 311 (env 0) finished at step 1501: success=False q_score=0.25",
        "[WARNING] Policy evaluation failed at step 40 for instance 312: Websocket connection lost: boom",
        "[INFO] Instance 312 (env 1) finished at step 40: success=False q_score=None",
        "Traceback (most recent call last):",
        "[INFO] Eval summary: 0/2 success | mean q_score=0.125 | task=turning_on_radio",
    ]
    out = launch.parse_evaluator_log(lines)
    assert out["max_steps"] == 1500 and out["max_steps_source"] == "default"
    assert out["resolved_instance_ids"] == [311, 312] and out["tracebacks"] == 1
    assert out["instances"][0] == {"instance_id": 311, "env": 0, "step": 1501, "success": False, "q_score": 0.25}
    assert out["instances"][1]["q_score"] is None
    assert out["failures"] == [{"step": 40, "instance_id": 312, "message": "Websocket connection lost: boom"}]
    assert out["summary"] == {"success": 0, "n": 2, "mean_q_score": 0.125}
    assert launch.parse_evaluator_log(["Setting timeout to be 300 steps through config."])["max_steps"] == 300


# ---------------------------------------------------------------------- supervisor with fake processes

FAKE_SERVER = r'''
import http.server, json, os, sys, threading

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

role = "agent" if "agent.server" in sys.argv else "executor"
mode = os.environ.get("FAKE_" + role.upper(), "ok")
if mode == "fail_start":
    print("fake " + role + " failing at start", flush=True)
    sys.exit(7)
if role == "agent":  # like agent/server.py: the run manifest is written before the port opens
    from agent.config import config_hash, get, load_config
    cfg = load_config(arg("--config"), [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "-o"])
    d = os.path.join(os.path.expanduser(get(cfg, "run.local_log_root")), arg("--run-id"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run_manifest.json"), "w") as f:
        json.dump({"config_hash": config_hash(cfg), "run_id": arg("--run-id"), "max_steps": arg("--max-steps")}, f)

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404)
        self.end_headers()
        self.wfile.write(b"OK\n")

    def log_message(self, *a):
        pass

srv = http.server.HTTPServer(("127.0.0.1", int(arg("--port"))), Health)
if mode.startswith("die_after_"):
    threading.Timer(float(mode[len("die_after_"):]), lambda: os._exit(9)).start()
print("fake " + role + " up", flush=True)
srv.serve_forever()
'''

FAKE_EVALUATOR = r'''
import json, os, sys, time, urllib.request

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

task, out, split = arg("--task-name"), arg("--output-dir"), arg("--mode")
i = sys.argv.index("--instance-indices") + 1
indices = []
while i < len(sys.argv) and not sys.argv[i].startswith("--"):
    indices.append(int(sys.argv[i]))
    i += 1
ids = [{"public_test": 301, "hidden_test": 321}.get(split, 0) + k for k in indices]
mode = os.environ.get("FAKE_EVAL", "ok")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://127.0.0.1:%s/healthz" % arg("--port"), timeout=5) as r:  # servers must be up already
    assert r.status == 200
print("INFO Resolved %s instance ids for %s: %s" % (split, task, ids), flush=True)
print("INFO Setting timeout to be 1.5x the average length of human demos: 1500", flush=True)
if mode == "hang":
    time.sleep(600)
if mode == "crash":
    print("Traceback (most recent call last):\nRuntimeError: simulated crash", flush=True)
    sys.exit(1)
os.makedirs(os.path.join(out, "json"), exist_ok=True)
for env_idx, iid in enumerate(ids):
    rec = {"task": task, "instance_id": iid, "rollout_id": 0, "steps": 1501, "success": False,
           "agent_distance": {"base": 1.0, "left": 0.5, "right": 0.5},
           "normalized_agent_distance": {"base": 2.0, "left": 1.0, "right": 1.0}, "q_score": {"final": 0.0},
           "time": {"simulator_steps": 1501, "simulator_time": 50.0, "normalized_time": 0.66}}
    if mode == "connection_lost":
        rec["failure_reason"] = "policy_connection_lost"
    with open(os.path.join(out, "json", "%s_%d_0.json" % (task, iid)), "w") as f:
        json.dump(rec, f)
    print("INFO Instance %d (env %d) finished at step 1501: success=False q_score=0.0" % (iid, env_idx), flush=True)
print("INFO Eval summary: 0/%d success | mean q_score=0.000 | task=%s" % (len(ids), task), flush=True)
sys.exit(3 if mode == "exit_after_json" else 0)
'''


@pytest.fixture
def fakes(site):
    """Point the plan at fake processes: executor script, agent python and evaluator python."""
    server = site.tmp / "fake_server.py"
    server.write_text(FAKE_SERVER)
    evaluator = site.tmp / "fake_evaluator.py"
    evaluator.write_text(FAKE_EVALUATOR)
    agent_python = executable(site.tmp / "fake_agent_python", f'#!/bin/sh\nexec "{sys.executable}" "{server}" "$@"\n')
    eval_python = executable(site.tmp / "fake_eval_python", f'#!/bin/sh\nexec "{sys.executable}" "{evaluator}" "$@"\n')
    site.overrides += [f"executor.server.python={sys.executable}", f"executor.server.script={server}"]
    return ["--agent-python", agent_python, "--activation", "none", "--evaluator-python", eval_python,
            "--poll-s", "0.1", "--agent-start-timeout-s", "30", "--executor-start-timeout-s", "30"]


def run_supervised(site, fakes, *extra, env=None, **kw):
    plan = plan_for(site, *fakes, *extra, **kw)
    launch.prepare_run_dir(plan)
    st = launch.Supervisor(plan, environ={**os.environ, **(env or {})}).run()
    assert st["state"] == launch.read_status(plan.run_dir)["state"]
    return plan, launch.read_status(plan.run_dir)


def assert_all_stopped(st):
    for name, rec in st["processes"].items():
        assert rec["started_by_us"] and rec.get("ended_at"), name
        assert launch.group_members(rec["pgid"]) == [], f"{name} process group still alive"


def test_supervisor_done_with_policy_failure(site, fakes):
    plan, st = run_supervised(site, fakes, instances=("10", "11"))
    assert st["state"] == "done" and st["reason"] == [] and st["infra_error"] is False and st["results_complete"]
    assert [r["file"] for r in st["results"]] == ["turning_on_radio_311_0.json", "turning_on_radio_312_0.json"]
    assert all(r["present"] and r["success"] is False and r["q_score"] == 0.0 for r in st["results"])
    assert st["evaluator_exit_code"] == 0 and set(st["processes"]) == {"executor", "agent", "evaluator"}
    assert st["processes"]["agent"]["healthy_at"] and st["processes"]["executor"]["healthy_at"]
    assert st["evaluator_log"]["summary"] == {"success": 0, "n": 2, "mean_q_score": 0.0}
    assert st["max_steps"] == 1500 and st["max_steps_info"]["evaluator_logged"] == 1500
    manifest = json.load(open(os.path.join(plan.run_dir, "run_manifest.json")))
    assert manifest["config_hash"] == plan.config_hash == st["agent_manifest"]["config_hash"]
    assert manifest["max_steps"] == "1500"
    assert_all_stopped(st)


def test_supervisor_direct_mode(site, fakes):
    plan, st = run_supervised(site, fakes, "--direct")
    assert st["state"] == "done" and set(st["processes"]) == {"executor", "evaluator"}
    assert st["ports"]["evaluator_target"] == site.ports[0] and not os.path.exists(
        os.path.join(plan.run_dir, "run_manifest.json"))
    assert_all_stopped(st)


def test_supervisor_evaluator_crash_is_infra_error(site, fakes):
    _, st = run_supervised(site, fakes, env={"FAKE_EVAL": "crash"})
    assert st["state"] == "infra_error" and st["infra_kinds"] == ["no_results", "exit_code"]
    assert st["infra_errors"] == [{"task": "turning_on_radio", "instance_id": 311, "rollout_id": 0,
                                   "reason": "no official result JSON"}]
    assert st["evaluator_log"]["tracebacks"] == 1
    assert_all_stopped(st)


def test_supervisor_server_start_failure(site, fakes):
    _, st = run_supervised(site, fakes, env={"FAKE_EXECUTOR": "fail_start"})
    assert st["state"] == "infra_error" and st["infra_kinds"] == ["server_start"]
    assert "executor exited with code 7" in st["reason"][0] and st["evaluator_started"] is False
    assert "evaluator" not in st["processes"]
    assert_all_stopped(st)


def test_supervisor_server_death_during_run(site, fakes):
    _, st = run_supervised(site, fakes, "--server-death-grace-s", "0.5",
                           env={"FAKE_EVAL": "hang", "FAKE_AGENT": "die_after_1"})
    assert st["state"] == "infra_error" and "server_died" in st["infra_kinds"]
    assert st["processes"]["agent"]["exit_code"] == 9
    assert any("agent exited with code 9" in w for w in st["warnings"])
    assert_all_stopped(st)


def test_supervisor_exit_code_after_complete_results(site, fakes):
    _, strict = run_supervised(site, fakes, env={"FAKE_EVAL": "exit_after_json"})
    assert strict["state"] == "infra_error" and strict["infra_kinds"] == ["exit_code"] and strict["results_complete"]
    _, lenient = run_supervised(site, fakes, "--no-strict-exit", env={"FAKE_EVAL": "exit_after_json"}, run_id="t-2")
    assert lenient["state"] == "done" and any("exit code 3" in w for w in lenient["warnings"])


def test_supervisor_stop_request_stops_only_its_children(site, fakes):
    plan = plan_for(site, *fakes)
    launch.prepare_run_dir(plan)
    sup = launch.Supervisor(plan, environ={**os.environ, "FAKE_EVAL": "hang"})
    t = threading.Thread(target=sup.run)
    t.start()
    deadline = time.monotonic() + 30
    while (launch.read_status(plan.run_dir) or {}).get("state") != "running" and time.monotonic() < deadline:
        time.sleep(0.1)
    sup.request_stop()
    t.join(60)
    st = launch.read_status(plan.run_dir)
    assert st["state"] == "failed" and st["reason"] == ["stopped_by_user"] and st["infra_kinds"] == ["stopped_by_user"]
    assert all(a.startswith("SIGINT: sent") for a in st["processes"]["evaluator"]["stop_actions"][:1])
    assert_all_stopped(st)


def test_detach_status_and_stop_cli(site, fakes, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_EVAL", "ok")
    argv = launch_argv(site, *fakes, "--detach", "--gpus", "0,1", "--skip-gpu-check")
    assert launch.main(argv) == 0
    run_dir = str(site.tmp / "nas_logs" / "t-run")
    assert launch.main(["status", run_dir, "--wait", "--interval", "0.2", "--timeout", "60"]) == 0
    st = launch.read_status(run_dir)
    assert st["state"] == "done" and st["supervisor"]["mode"] == "detached"
    assert_all_stopped(st)

    monkeypatch.setenv("FAKE_EVAL", "hang")
    argv = launch_argv(site, *fakes, "--detach", "--gpus", "0,1", "--skip-gpu-check", run_id="t-hang")
    assert launch.main(argv) == 0
    hang_dir = str(site.tmp / "nas_logs" / "t-hang")
    deadline = time.monotonic() + 30
    while launch.read_status(hang_dir)["state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.1)
    assert launch.main(["status", hang_dir]) == launch.EXIT_RUNNING
    assert launch.main(["stop", hang_dir]) == 0
    st = launch.read_status(hang_dir)
    assert st["state"] == "failed" and st["reason"] == ["stopped_by_user"]
    assert_all_stopped(st)
    capsys.readouterr()


# ---------------------------------------------------------------------- never signal anyone else


def test_signal_own_group_refuses_older_members():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        start = launch.proc_stat(p.pid)["start_ticks"]
        # a recycled pgid whose members started before "our" leader is never signalled
        assert launch.signal_own_group(p.pid, signal.SIGTERM, start + 1000).startswith("refused")
        assert p.poll() is None
        assert launch.signal_own_group(p.pid, signal.SIGTERM, start) == "sent"
        assert p.wait(10) == -signal.SIGTERM
        assert launch.signal_own_group(p.pid, signal.SIGTERM, start) == "gone"
    finally:
        if p.poll() is None:
            p.kill()


def test_signals_only_in_guarded_places():
    """os.kill / os.killpg / send_signal appear only in the guarded helpers (own process groups, own supervisor)."""
    allowed = {"launch.py": {"signal_own_group", "cmd_stop"}, "batch.py": {"terminate_all"}, "gpu.py": set()}
    for fname, funcs in allowed.items():
        tree = ast.parse(open(os.path.join(ROOT, "eval", fname), encoding="utf-8").read())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr in {"kill", "killpg", "send_signal", "terminate"}]
            if calls:
                assert fn.name in funcs, f"{fname}:{fn.name} calls {calls}"


def test_supervisor_stops_run_on_gpu_placement_violation(site, fakes):
    plan = plan_for(site, *fakes)
    plan.options["heartbeat_s"] = 0.0
    launch.prepare_run_dir(plan)
    holder = {}

    def apps():  # nvidia-smi --query-compute-apps: our evaluator shows up on GPU 1 (uuid GPU-b), assigned GPU 0
        ev = holder["sup"].procs.get("evaluator")
        return [gpu.GpuProcess(ev.pid, "GPU-b", 900.0, "python")] if ev else []

    sup = launch.Supervisor(plan, environ={**os.environ, "FAKE_EVAL": "hang"}, gpu_query=lambda: GPUS, apps_query=apps)
    holder["sup"] = sup
    st = sup.run()
    assert st["state"] == "failed" and st["infra_kinds"] == ["gpu_placement"]
    assert "evaluator uses GPU(s) [1] it was not assigned (assigned [0])" in st["reason"][0]
    assert st["gpu_placement"]["evaluator"] == {"expected": [0], "seen": [1]}
    assert_all_stopped(st)

    plan2 = plan_for(site, *fakes, "--gpu-mismatch", "warn", run_id="t-warn")
    plan2.options["heartbeat_s"] = 0.0
    launch.prepare_run_dir(plan2)
    sup2 = launch.Supervisor(plan2, environ=dict(os.environ), gpu_query=lambda: GPUS, apps_query=lambda: [
        gpu.GpuProcess(p.pid, "GPU-b", 900.0, "python") for n, p in sup2.procs.items() if n == "agent"])
    st2 = sup2.run()
    assert st2["state"] == "done" and any("agent uses GPU(s) [1]" in w for w in st2["warnings"])


def test_provider_keys_only_reach_the_agent(site):
    plan = plan_for(site, environ={**site.environ, "ASTRA_API_KEY": "sk-secret", "ANTHROPIC_API_KEY": "sk-secret"})
    for name in ("executor", "evaluator"):
        assert {"ASTRA_API_KEY", "ANTHROPIC_API_KEY"} <= set(plan.proc(name).unset)
    assert "ASTRA_API_KEY" not in plan.proc("agent").unset
    child = plan.proc("evaluator").child_env({"ASTRA_API_KEY": "sk-secret", "PATH": "/bin"})
    assert "ASTRA_API_KEY" not in child and child["PATH"] == "/bin"
    launch.prepare_run_dir(plan)
    for f in ("commands.sh", "plan.json", "status.json", "run_evaluator.sh"):
        assert "sk-secret" not in open(os.path.join(plan.run_dir, f)).read()
