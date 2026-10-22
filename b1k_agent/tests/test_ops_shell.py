"""The 105 shell scripts: bash -n syntax checks, plus runs against FAKE environments (temporary directories, a
fake nvidia-smi, a dummy "watcher" process this test starts and stops itself). Nothing here touches a real GPU,
a real env, or any process the test did not start."""

import glob
import os
import signal
import stat
import subprocess
import time

import pytest
from conftest import ROOT

SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "scripts", "*.sh")))
FAKE_SMI = """#!/usr/bin/env bash
while true; do
  case "$*" in
    *query-gpu*) echo "2026/10/22 10:00:00.000, 0, GPU-aaa, 1000, 81920, 5" ;;
    *) echo "2026/10/22 10:00:00.000, 4242, GPU-aaa, 900" ;;
  esac
  sleep 0.1
done
"""


def test_expected_scripts_exist():
    names = {os.path.basename(p) for p in SCRIPTS}
    assert {"takeover_check.sh", "gpu_sampler.sh", "setup_agent_env.sh"} <= names


@pytest.mark.parametrize("path", SCRIPTS, ids=os.path.basename)
def test_bash_syntax(path):
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_setup_agent_env_dry_run_and_refusals(tmp_path):
    script = os.path.join(ROOT, "scripts", "setup_agent_env.sh")
    prefix = str(tmp_path / "newenv")
    r = subprocess.run(["bash", script, "--prefix", prefix, "--dry-run", "--with-anthropic"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0 and "neither uv nor python3.11" in r.stderr:
        pytest.skip("no uv / python3.11 on this machine")
    assert r.returncode == 0, r.stderr
    for pin in ("numpy==1.26.4", "websockets==15.0.1", "msgpack==1.1.0", "pyyaml==6.0.2", "pillow==11.2.1",
                "httpx==0.28.1", "pytest==8.3.4", "anthropic"):
        assert pin in r.stdout
    assert not os.path.exists(prefix), "--dry-run must not create anything"
    for protected in ("/nas/baiqiao/envs/behavior", "/nas/baiqiao/openpi/.venv"):
        r = subprocess.run(["bash", script, "--prefix", protected, "--dry-run"], capture_output=True, text=True)
        assert r.returncode == 1 and "refusing" in r.stderr
    r = subprocess.run(["bash", script, "--prefix", str(tmp_path), "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 1 and "already exists" in r.stderr


def test_gpu_sampler_with_fake_nvidia_smi(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    script = os.path.join(ROOT, "scripts", "gpu_sampler.sh")
    run_dir = str(tmp_path / "run")
    assert subprocess.run(["bash", script], capture_output=True, text=True).returncode == 2
    r = subprocess.run(["bash", script, "start", run_dir, "100"], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    pids = [int(line.split()[0]) for line in open(os.path.join(run_dir, "gpu_sampler.pid"))]
    assert len(pids) == 2
    again = subprocess.run(["bash", script, "start", run_dir], capture_output=True, text=True, env=env, timeout=30)
    assert again.returncode == 1 and "already running" in again.stderr
    time.sleep(0.5)
    r = subprocess.run(["bash", script, "stop", run_dir], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0 and r.stdout.count("stopped") == 2, r.stdout + r.stderr
    assert "GPU0 samples" in r.stdout and "peak 1000 MiB" in r.stdout
    time.sleep(0.3)
    for pid in pids:
        assert not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().split(") ")[1][0] == "Z"
    lines = open(os.path.join(run_dir, "gpu_samples.csv")).read().splitlines()
    assert lines[0].startswith("timestamp,index") and len(lines) > 1


FAKE_SMI_NO_APPS = """#!/usr/bin/env bash
case "$*" in
  *query-compute-apps*) echo "Field \\"timestamp\\" is not a valid field to query." >&2; exit 2 ;;
  *query-gpu*) while true; do echo "2026/10/22 10:00:00.000, 0, GPU-aaa, 1000, 81920, 5"; sleep 0.1; done ;;
esac
exit 2
"""


def _fake_smi(tmp_path, text):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    smi = fake_bin / "nvidia-smi"
    smi.write_text(text)
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    return {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}


def test_gpu_sampler_reports_a_sampler_that_exits_at_once(tmp_path):
    """A query field this driver rejects must not be reported as a running sampler (regression)."""
    script = os.path.join(ROOT, "scripts", "gpu_sampler.sh")
    run_dir = str(tmp_path / "run")
    env = _fake_smi(tmp_path, FAKE_SMI_NO_APPS)
    r = subprocess.run(["bash", script, "start", run_dir, "100"], capture_output=True, text=True, env=env, timeout=30)
    try:
        assert r.returncode == 0 and "exited at once" in r.stderr and "started 1 sampler(s)" in r.stdout
        assert len(open(os.path.join(run_dir, "gpu_sampler.pid")).read().splitlines()) == 1
        assert "not a valid field" in open(os.path.join(run_dir, "gpu_sampler.err")).read()
    finally:
        subprocess.run(["bash", script, "stop", run_dir], capture_output=True, text=True, env=env, timeout=30)
    env = _fake_smi(tmp_path, "#!/usr/bin/env bash\nexit 9\n")
    r = subprocess.run(["bash", script, "start", str(tmp_path / "run2"), "100"], capture_output=True, text=True,
                       env=env, timeout=30)
    assert r.returncode == 1 and r.stderr.count("exited at once") == 2


def test_setup_agent_env_argument_errors_and_help():
    script = os.path.join(ROOT, "scripts", "setup_agent_env.sh")
    r = subprocess.run(["bash", script, "--prefix"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "--prefix needs a path" in r.stderr
    r = subprocess.run(["bash", script, "--help"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "exact pins" in r.stdout
    assert "set -euo" not in r.stdout and "PREFIX=" not in r.stdout


def test_gpu_sampler_stop_leaves_foreign_pids_alone(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    victim = subprocess.Popen(["sleep", "30"])  # ours, but not an nvidia-smi sampler: must not be signalled
    try:
        (run_dir / "gpu_sampler.pid").write_text(f"{victim.pid} 1\n")
        r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "gpu_sampler.sh"), "stop", str(run_dir)],
                           capture_output=True, text=True, timeout=30)
        assert "left alone" in r.stdout and victim.poll() is None
    finally:
        victim.kill()
        victim.wait()


def _takeover(tmp_path, runs_dir, pattern):
    env = {**os.environ, "OUT_DIR": str(tmp_path / "out"), "RUNS_DIR": str(runs_dir), "B1K_ROOT": str(tmp_path / "b1k"),
           "OPENPI_ROOT": str(tmp_path / "openpi"), "BEHAVIOR_ENV": str(tmp_path / "env"),
           "CKPT_DIR": str(tmp_path / "ckpt"), "WATCH_PATTERN": pattern, "DU_TIMEOUT": "5",
           "TAKEOVER_SKIP_IMPORTS": "1", "PY": "python3"}
    (tmp_path / "b1k" / "datasets").mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "takeover_check.sh")], capture_output=True, text=True,
                       env=env, timeout=240)
    assert r.returncode == 0, r.stderr
    report = r.stdout.strip().splitlines()[-1]
    assert report.startswith(str(tmp_path / "out")) and os.path.isfile(report)
    return open(report).read()


def test_takeover_check_classifies_watcher_state(tmp_path):
    runs = tmp_path / "b1k_runs"
    runs.mkdir()
    (runs / "smoke_watch.sh").write_text("#!/bin/bash\n# user's watcher script\n")
    log = runs / "smoke_watch.log"
    log.write_text("checking GPUs\nwaiting for free GPU (memory < 2GB)\n")
    pattern = "zz_fake_smoke_watcher_q7"
    dummy = subprocess.Popen(["bash", "-c", f"exec -a {pattern} sleep 60"], start_new_session=True)
    try:
        text = _takeover(tmp_path, runs, pattern)
    finally:
        os.killpg(dummy.pid, signal.SIGTERM)
        dummy.wait()
    assert "**Watcher state: not_started**" in text and f"pid {dummy.pid}" in text
    assert "# user's watcher script" in text and "waiting for free GPU" in text
    for section in ("## 2. GPUs", "## 3. Versions", "## 4. Disk", "## 5. Checkpoint", "## Summary"):
        assert section in text

    log.write_text("launching smoke test\nTraceback (most recent call last):\nRuntimeError: Failed to create Vulkan device\n")
    text = _takeover(tmp_path, runs, pattern)
    assert "**Watcher state: failed**" in text and "Vulkan device" in text

    log.write_text("launching smoke test\nreset ok\nSMOKE OK: rgb and depth non-empty, 50 steps\n")
    assert "**Watcher state: finished**" in _takeover(tmp_path, runs, pattern)


def test_takeover_report_redacts_secrets(tmp_path):
    """The report quotes the user's scripts, logs and command lines and lands in the repo's docs/: no key may reach
    it (regression: the first version copied them verbatim)."""
    runs = tmp_path / "b1k_runs"
    runs.mkdir()
    secrets = ["sk-ant-api03-FAKEFAKEFAKE", "hf_FAKEFAKEFAKEFAKE", "plainvalue123", "flagvalue456", "eyJFAKEJWT"]
    (runs / "run_smoke.sh").write_text(
        "#!/bin/bash\n"
        f"export ANTHROPIC_API_KEY={secrets[0]}\n"
        f'export HF_TOKEN="{secrets[1]}"\n'
        f"ASTRA_API_KEY='{secrets[2]}'\n"
        f"python -m agent.server --api-key {secrets[3]} --port 8100\n"
        f'curl -H "Authorization: Bearer {secrets[4]}" http://x\n')
    (runs / "smoke.log").write_text(
        f"starting with key {secrets[0]}\n"
        "[00:16:40.000] [INFO] [omnigibson.eval.evaluator] Instance 311 (env 0) finished at step 3225: "
        "success=False q_score=0.0\n")
    text = _takeover(tmp_path, runs, "zz_no_such_process_q8")
    for s in secrets:
        assert s not in text, s
    assert "export ANTHROPIC_API_KEY=<redacted>" in text and "--api-key <redacted> --port 8100" in text
    # evidence stays readable, and the state survives the output filter
    assert "finished at step 3225: success=False q_score=0.0" in text and "**Watcher state: finished**" in text
