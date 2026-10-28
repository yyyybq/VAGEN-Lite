"""downstream/eval/watch.sh (the user's login-node rule of 2026-10-28): the watcher runs at nice 19 with one thread per
process; `start` without options is refused (on 10-28 it emptied watch.args); `restart` starts again with the saved
options.  A stand-in watch.py records its environment, niceness and arguments; WATCH_RUNS_ROOT keeps every file in
the test's own folder."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = """import json, os, sys, time
from pathlib import Path
keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM")
board = sys.argv[sys.argv.index("--board") + 1]
rec = {"pid": os.getpid(), "nice": os.nice(0), "argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}}
(Path(board).parent / f"stub_{os.getpid()}.json").write_text(json.dumps(rec))
time.sleep(60)
"""


def tree(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/eval").mkdir(parents=True)
    shutil.copy(ROOT / "eval/watch.sh", wt / "downstream/eval/watch.sh")
    (wt / "downstream/eval/watch.py").write_text(STUB)
    runs = tmp_path / "runs"
    runs.mkdir()
    return wt, runs, dict(os.environ, WATCH_RUNS_ROOT=str(runs))


def sh(wt, env, *args):
    return subprocess.run(["bash", str(wt / "downstream/eval/watch.sh"), *args], capture_output=True, text=True, env=env)


def records(runs, wait=10.0):
    end = time.time() + wait
    while time.time() < end:
        found = sorted(runs.glob("stub_*.json"), key=lambda p: p.stat().st_mtime)
        if found:
            return [json.loads(p.read_text()) for p in found]
        time.sleep(0.2)
    return []


def test_start_needs_options_and_runs_the_watcher_niced_with_one_thread(tmp_path):
    wt, runs, env = tree(tmp_path)
    (runs / "watch.args").write_text("--every\n300\n")
    res = sh(wt, env, "start")
    assert res.returncode == 2 and "not started" in res.stdout
    assert (runs / "watch.args").read_text() == "--every\n300\n" and not (runs / "watch.pid").exists()
    try:
        res = sh(wt, env, "start", "--every", "300", "--whole-pools", "h800", "zoetrope")
        assert res.returncode == 0, res.stdout + res.stderr
        (rec,) = records(runs)
        assert rec["nice"] == 19
        assert rec["env"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                              "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
        assert rec["argv"] == ["--every", "900", "--board", str(runs / "BOARD.md"), "--every", "300", "--whole-pools",
                               "h800", "zoetrope"]
        assert int((runs / "watch.pid").read_text()) == rec["pid"]  # the PID file names the watcher itself
        assert (runs / "watch.args").read_text().split() == ["--every", "300", "--whole-pools", "h800", "zoetrope"]
        assert sh(wt, env, "start", "--every", "300").returncode == 1  # one is already running
    finally:
        sh(wt, env, "stop")


def test_restart_reuses_the_saved_options(tmp_path):
    wt, runs, env = tree(tmp_path)
    assert sh(wt, env, "restart").returncode == 2  # nothing saved: refused
    try:
        assert sh(wt, env, "start", "--every", "300", "--yield-pool-only").returncode == 0
        first = records(runs)[0]
        for p in runs.glob("stub_*.json"):
            p.unlink()
        res = sh(wt, env, "restart")
        assert res.returncode == 0, res.stdout + res.stderr
        second = records(runs)[0]
        assert second["pid"] != first["pid"] and second["argv"] == first["argv"]
        assert subprocess.run(["kill", "-0", str(first["pid"])], capture_output=True).returncode != 0  # stopped
    finally:
        sh(wt, env, "stop")
    assert not (runs / "watch.pid").exists()
