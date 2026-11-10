"""Tests of the round-3 wave-1c fixes (CONTRACT_wave2 section 5.5):
1. vpd/walkerbench/perc.py cmd_eval leaves a finished start's run dir (results.json / results_invalid.json) and its
   explore.json untouched unless --overwrite (checked right after the run-dir lock is taken);
2. vpd/jobs/wb_learn483.sh runs every block of its step list through `part` (wb_variants.sh's helper), so a failed
   init-check / training no longer ends the payload under `set -e`;
3. vpd/e3vs/__init__.py code_meta: worktree_head falls back to the code snapshot's commit (worktree_head_source
   "snapshot") when the tree has no .git;
4. vpd/sco.py delete(): on the never_started path a terminal re-check that carries a start_time is not "never started"
   (the estimate stays charged), and the forced delete event records the parsed delete rc;
5. vpd/README.md has rows for the wave-1b / learn483 items and for every file of the wave-2 contract section 2.

Hermetic: fake run_vpd, fake sco, fake interpreters (bash scripts) in a temp code tree, temp snapshot dirs; no GPU,
no network, no rendering, no model.  Reuses the fixtures of test_perc.py and test_ops.py.
    cd $TREE && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 nice -n 19 \
      env VPD_WORKTREE=$TREE PYTHONPATH=$TREE $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=<dir> \
      vpd/tests/test_wave1c.py
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

import pytest  # noqa: E402

from vpd import ledger, paths, sco  # noqa: E402
from vpd.tests import test_ops as OPS  # noqa: E402
from vpd.tests import test_perc as TP  # noqa: E402
from vpd.walkerbench import perc as P  # noqa: E402

# fixtures reused from the two test modules (pytest finds fixtures among the module's globals)
base = OPS.base
fake = OPS.fake
in_job = TP.in_job

JOBS_DIR = paths.VPD_PKG / "jobs"
L483 = JOBS_DIR / "wb_learn483.sh"
VARIANTS = JOBS_DIR / "wb_variants.sh"
README = paths.VPD_PKG / "README.md"


# ------------------------------------------------------------------------------- 1. perc: a finished start is kept
def _state(p: Path) -> tuple:
    return p.read_bytes(), p.stat().st_mtime_ns


def test_perc_eval_keeps_a_finished_start(tmp_path, in_job, monkeypatch):
    """The wave-1b reviewer's repro: a second cmd_eval with a different env RUN on a finished start leaves
    explore.json byte-identical (and its mtime unchanged); run_vpd is not even called; rc 0 for results.json."""
    repo, pdir, dest = TP.built_set(tmp_path)
    root, seen = tmp_path / "R", []
    R = TP._fake_run_vpd(seen, dest)
    assert P.cmd_eval(["--start", "Q3S", "--root", str(root)], R, dest, repo, pdir, TP.SMALL) == 0
    ex = P.run_dir_of("Q3S", root) / "explore.json"
    assert json.loads(ex.read_text())["run"] == str(in_job) and len(seen) == 1
    before = _state(ex)
    run2 = tmp_path / "jobs" / "wb-perc-second"
    run2.mkdir(parents=True)
    monkeypatch.setenv("RUN", str(run2))
    assert P.cmd_eval(["--start", "Q3S", "--root", str(root)], R, dest, repo, pdir, TP.SMALL) == 0
    assert _state(ex) == before and len(seen) == 1  # nothing ran, nothing rewritten
    assert not P.lock_path("Q3S", root).exists()  # the lock was released on the early return
    # --overwrite still replaces the run (the fake raises "invalid run" then: rc 1) and rewrites explore.json
    assert P.cmd_eval(["--start", "Q3S", "--root", str(root), "--overwrite"], R, dest, repo, pdir, TP.SMALL) == 1
    doc = json.loads(ex.read_text())
    assert len(seen) == 2 and doc["run"] == str(run2) and doc["rc"] == 1
    assert not P.lock_path("Q3S", root).exists()


def test_perc_eval_keeps_an_invalid_start_and_a_refusing_run_vpd(tmp_path, in_job):
    """results_invalid.json without --overwrite: rc 1, explore.json untouched.  The contract's variant: a run_vpd that
    would refuse ("refusing: exists") on a finished run dir changes nothing either (it is not reached)."""
    repo, pdir, dest = TP.built_set(tmp_path)
    root, calls = tmp_path / "R", []

    def refusing_main(argv):
        calls.append(list(argv))
        raise SystemExit("refusing: exists")

    import types
    R = types.SimpleNamespace(TASK_SETS={}, main=refusing_main, code_sha256=lambda: {"run_vpd": "ab"})
    for start, name, rc_want in (("E2", "results_invalid.json", 1), ("Q3B", "results.json", 0)):
        d = P.run_dir_of(start, root)
        d.mkdir(parents=True)
        (d / name).write_text("{}\n")
        (d / "explore.json").write_text('{"written": "by the first job"}\n')
        before = _state(d / "explore.json")
        assert P.cmd_eval(["--start", start, "--root", str(root)], R, dest, repo, pdir, TP.SMALL) == rc_want
        assert _state(d / "explore.json") == before and not P.lock_path(start, root).exists()
    assert calls == []
    assert P.finished_result(tmp_path / "none") is None


def test_perc_eval_normal_run_still_writes_explore(tmp_path, in_job):
    repo, pdir, dest = TP.built_set(tmp_path)
    root, seen = tmp_path / "R", []
    R = TP._fake_run_vpd(seen, dest)
    d = P.run_dir_of("E2", root)
    d.mkdir(parents=True)  # an earlier cut run: a dir without results -> runs and writes explore.json
    assert P.finished_result(d) is None
    assert P.cmd_eval(["--start", "E2", "--root", str(root)], R, dest, repo, pdir, TP.SMALL) == 0
    assert len(seen) == 1 and json.loads((d / "explore.json").read_text())["rc"] == 0


# ------------------------------------------------------------------------------- 2. wb_learn483.sh: `part`
def _func_text(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.M | re.S)
    assert m, name
    return m.group(0)


STEP_FUNCS = ("floors", "init_check", "check_pilot_adapter", "train", "merged_evals", "tf_step", "evalrun")


def test_learn483_part_helper_and_no_bare_step_calls():
    text = L483.read_text()
    assert _func_text(text, "part") == _func_text(VARIANTS.read_text(), "part")  # identical helper
    steps = text.split('vpd_step setup x "$PY_EVAL"', 1)[1].split("df -h / /tmp /dev/shm", 1)[0]
    bare = [ln for ln in steps.splitlines() if ln.strip().split(" ")[0] in STEP_FUNCS]
    assert bare == [], bare  # every block of the step list runs through `part`
    assert "|| true" not in "\n".join(ln for ln in steps.splitlines() if ln.strip().startswith("part "))
    r = subprocess.run(["bash", "-n", str(L483)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


FAKE_PY = r"""#!/usr/bin/env bash
# fake PY_EVAL / PY_Q3 / PY_TRAIN of the non-dry payload test: log the call, answer it, touch nothing else
printf '%s\n' "$*" >> "$FAKE_LOG"
a=" $* "
case $a in
  *" vpd.walkerbench.learn483 init-check "*) exit "${FAKE_INIT_RC:-0}" ;;
  *" vpd.walkerbench.pilot adapter-ok "*) exit "${FAKE_ADAPTER_RC:-1}" ;;
  *" vpd.walkerbench.learn483 parity-ok "*) exit 0 ;;
esac
exit 0
"""

_KNOBS = ("VPD_THREADS", "VPD_JOB_DEADLINE_S", "VPD_MIN_STEP_S", "VPD_MERGE_DIR", "VPD_RERUN_STALE",
          "VPD_RERUN_INVALID", "VPD_DRYRUN", "RUN", "VPD_X483_ROOT", "VPD_X483_PILOT", "VPD_WORKTREE")


def _l483_tree(tmp_path: Path):
    """A temp code tree with a byte copy of wb_learn483.sh and a common_env.sh shim (the real prelude, then fake
    interpreters and VPD_OUT = tmp_path/OUT holding the payload's input files)."""
    jobs = tmp_path / "code" / "vpd" / "jobs"
    jobs.mkdir(parents=True)
    shutil.copyfile(L483, jobs / "wb_learn483.sh")
    fakepy = tmp_path / "fakepy"
    fakepy.write_text(FAKE_PY)
    fakepy.chmod(0o755)
    data = tmp_path / "OUT" / "walkerbench" / "tiers" / "seed1001" / "demos_all.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("{}\n")
    (tmp_path / "P" / "adapters" / "0W").mkdir(parents=True)
    (tmp_path / "P" / "adapters" / "0W" / "adapter.pt").write_bytes(b"0W")
    q = shlex.quote
    (jobs / "common_env.sh").write_text(
        "# test shim: the real prelude, then fake interpreters and a temp OUT\n"
        f"source {q(str(JOBS_DIR / 'common_env.sh'))}\n"
        f"export VPD_OUT={q(str(tmp_path / 'OUT'))} PY_EVAL={q(str(fakepy))} PY_Q3={q(str(fakepy))} "
        f"PY_TRAIN={q(str(fakepy))}\n")
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update({"VPD_WORKTREE": str(tmp_path / "code"), "VPD_X483_ROOT": str(tmp_path / "X"),
                "VPD_X483_PILOT": str(tmp_path / "P"),
                "VPD_MERGE_DIR": str(tmp_path / "merged"),  # not /tmp|/dev/shm vpd_merged: nothing is removed
                "FAKE_LOG": str(tmp_path / "fake.log")})
    return jobs / "wb_learn483.sh", env


def _steps(run: Path) -> list:
    return [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("arm", ["A", "B"])
def test_learn483_failed_init_check_does_not_end_the_payload(tmp_path, arm):
    """init-check refuses (rc 1): train_<arm> returns 1 (train_483X:no_init).  Before wave 1c that bare call ended the
    payload under `set -e`; now the merge / probe / raw-eval steps after it still run, FAILS lists init_check and the
    training, there is no RUN/DONE and the exit code is 1.  (A VPD_DRYRUN=1 run cannot fail a step: `x` only prints,
    so the failure path runs non-dry on fake interpreters, as test_wb_variants does.)"""
    payload, env = _l483_tree(tmp_path)
    run = tmp_path / "jobs" / f"wb-x483{arm.lower()}-202611100001"
    env.update({"FAKE_INIT_RC": "1", "FAKE_ADAPTER_RC": "1"})
    res = subprocess.run(["bash", str(payload), arm, str(run)], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path), timeout=300)
    assert res.returncode == 1, res.stderr + res.stdout
    steps = _steps(run)
    names = [s["name"] for s in steps]
    rc = {s["name"]: s["rc"] for s in steps}
    t = f"train_483{arm}:no_init"
    assert rc["init_check"] == 1 and rc[t] == 1 and names.index("init_check") < names.index(t)
    after = names[names.index(t) + 1:]
    assert f"merge_483{arm}:no_adapter" in after  # the next block ran
    m = re.search(r"failed steps: (.*)$", res.stdout.strip().splitlines()[-1])
    assert m, res.stdout
    fails = m.group(1).split()
    assert "init_check" in fails and f"train_483{arm}" in fails and f"merge_483{arm}" in fails
    if arm == "A":
        raw = ["eval_raw_s2val_640_empty", "eval_raw_s2in50_640_empty", "eval_raw_s2test_640_off",
               "eval_raw_s2test_1280_off"]
        assert [n for n in after if n.startswith("eval_raw")] == raw and all(rc[n] == 0 for n in raw)
        assert rc["tf_A"] == 0 and rc["floors_s2val"] == 0 and rc["floors_s2in50"] == 0
        assert {"pilot_adapter_0W", "pilot_adapter_50", "pilot_adapter_200"} <= set(fails)
    else:
        assert rc["tf_B:no_adapter"] == 1 and "tf_B" in fails
    assert not (run / "DONE").exists()
    log = (tmp_path / "fake.log").read_text()
    assert "n2_train_vpd.py" not in log and "merge.py" not in log  # no training, no merge
    assert not (tmp_path / "merged").exists()


def test_learn483_dryrun_still_passes(tmp_path):
    """The dry run of both arms is unchanged by `part`: every step passes and DONE is printed."""
    for arm in ("A", "B"):
        run = tmp_path / f"run{arm}"
        env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
        env.update({"VPD_DRYRUN": "1", "VPD_MERGE_DIR": "/dev/shm/vpd_merged", "VPD_X483_ROOT": str(tmp_path / "X"),
                    "VPD_X483_PILOT": str(tmp_path / "P")})
        res = subprocess.run(["bash", str(L483), arm, str(run)], capture_output=True, text=True, env=env,
                             cwd=str(tmp_path), timeout=300)
        assert res.returncode == 0, res.stderr + res.stdout
        assert f"[dry] date -u > {run}/DONE" in res.stdout and "failed steps: none" in res.stdout
        assert all(s["rc"] == 0 for s in _steps(run))
        assert not any(s["name"].endswith(":failed") for s in _steps(run))


# ------------------------------------------------------------------------------- 3. e3vs code_meta in a snapshot
def test_e3vs_code_meta_worktree_head(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    import vpd.e3vs as E

    src = tmp_path / "f.py"
    src.write_text("x = 1\n")
    commit = "c0ffee" * 6 + "abcd"
    snap = tmp_path / "RUN" / "code"
    (snap / "vpd").mkdir(parents=True)
    (snap / "SNAPSHOT.json").write_text(json.dumps({"format": 1, "commit": commit, "utc": "x", "files": {}}))
    monkeypatch.setattr(paths, "WORKTREE", snap)
    m = E.code_meta(src)
    assert m["worktree_head"] == commit and m["worktree_head_source"] == "snapshot"
    assert list(m) == ["worktree_head", "worktree_head_source", "files"]
    assert m["files"] == {str(src): E.sha256_file(src)}
    # a live tree without .git and without SNAPSHOT.json: None, no source key (unchanged behaviour)
    live = tmp_path / "live"
    (live / "vpd").mkdir(parents=True)
    monkeypatch.setattr(paths, "WORKTREE", live)
    assert E.code_meta(src) == {"worktree_head": None, "files": {str(src): E.sha256_file(src)}}
    # an unreadable marker is not a snapshot: None, never an exception
    (live / "SNAPSHOT.json").write_text("not json")
    assert E.code_meta()["worktree_head"] is None
    # a git checkout: HEAD from .git, as before (no source key)
    gitwt = tmp_path / "wt"
    (gitwt / ".git").mkdir(parents=True)
    (gitwt / ".git" / "HEAD").write_text("d" * 40 + "\n")
    (gitwt / "SNAPSHOT.json").write_text(json.dumps({"format": 1, "commit": commit, "files": {}}))
    monkeypatch.setattr(paths, "WORKTREE", gitwt)
    inside = gitwt / "vpd" / "a.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("y = 2\n")
    assert E.code_meta(inside) == {"worktree_head": "d" * 40, "files": {"vpd/a.py": E.sha256_file(inside)}}


# ------------------------------------------------------------------------------- 4. sco: re-check with start_time
def _end(run: Path) -> dict:
    ends = [e for e in ledger.read() if e["run"] == str(run.resolve()) and e["event"] == "end"]
    assert len(ends) == 1, ends
    return ends[0]


def _delete_event(run: Path) -> dict:
    ds = [e for e in ledger.read() if e["run"] == str(run.resolve()) and e["event"] == "delete"]
    assert len(ds) == 1, ds
    return ds[0]


def test_force_delete_recheck_with_start_time_keeps_the_estimate(base, fake, monkeypatch):
    """Both observes saw QUEUEING without start_time; the forced delete reports rc 1; the re-check then sees STOPPED
    WITH a start_time (the job got cards in between): not never_started, the estimate stays charged.  The same
    re-check without a start_time is still accepted (0.0, never_started).  The delete event records the parsed rc."""
    clock = OPS.Clock()
    runs = [OPS.prep(base, f"vpd-tvr-rs{i}", est=0.1) for i in range(2)]
    pts = [OPS.do_submit(r, clock) for r in runs]
    fake.force_delete_rc = 1
    t0 = OPS.T0
    started = {pts[0]: t0, pts[1]: None}

    def delete_then_stopped(run):
        out = fake.force_delete(run)  # rc 1: the job stays
        pt = json.loads((Path(run) / "job" / "submission.json").read_text())["job_id"]
        fake.set(pt, "STOPPED", started[pt], t0 + datetime.timedelta(minutes=3))
        return out

    monkeypatch.setattr(sco.sco_job, "delete", delete_then_stopped)
    out = sco.delete(runs[0], force_live=True, sleep=clock.sleep)
    assert "delete rc=1" in out
    end = _end(runs[0])
    assert end["gph"] is None and end["gph_source"] == "unknown_charged_est"
    assert end["state"] == "FORCED_FROM_QUEUEING" and end["start_time"] is None
    assert end["delete_check"] == f"delete rc=1; re-checked after the delete: STOPPED with start_time {OPS.iso(t0)}"
    assert _delete_event(runs[0])["rc"] == 1
    sco.delete(runs[1], force_live=True, sleep=clock.sleep)
    end = _end(runs[1])
    assert end["gph"] == 0.0 and end["gph_source"] == "never_started"
    assert end["delete_check"] == "re-checked after the delete: STOPPED"
    assert _delete_event(runs[1])["rc"] == 1
    assert abs(ledger.summarise(ledger.read())["by_gate"]["tvr"]["spent"] - 0.1) < 1e-9  # run 0's estimate only


def test_delete_rc_parsing():
    assert sco.delete_rc("pt-x: delete rc=0 ok") == 0
    assert sco.delete_rc("pt-x: delete rc=-9 ") == -9
    assert sco.delete_rc("pt-x: stopped and removed") is None and sco.delete_rc(None) is None


def test_forced_delete_event_rc_null_without_rc(base, fake):
    clock = OPS.Clock()
    run = OPS.prep(base, "vpd-tvr-rn0", est=0.1)
    OPS.do_submit(run, clock)
    fake.force_delete_out = "{pt}: stopped and removed"
    sco.delete(run, force_live=True, sleep=clock.sleep)
    assert _delete_event(run)["rc"] is None and _end(run)["gph_source"] == "never_started"


# ------------------------------------------------------------------------------- 5. README coverage
SECTION2_FILES = (  # CONTRACT_wave2 section 2, every package (paths relative to vpd/)
    "tvr/tvr_text.py", "tvr/tasks.py", "tvr/kin_env.py", "tvr/bfs.py", "tvr/reach.py", "tvr/floors.py",
    "tvr/demos.py", "tests/tvr_fakes.py", "tests/test_tvr_data.py",
    "tvr/thor_worker.py", "tvr/run_tvr.py", "tvr/score_tvr.py", "tvr/parity_tvr.py", "tvr/fake_server.py",
    "tvr/render_parity.py", "tests/test_tvr_eval.py",
    "tvr/pilot_tvr.py", "tvr/gate_tvr.py", "tvr/recheck_tvr.py", "tvr/merge_parity.py", "jobs/tvr_data.sh",
    "jobs/tvr_cal.sh", "jobs/tvr_cal400.sh", "jobs/tvr_mech_smoke.sh", "jobs/tvr_train.sh", "jobs/tvr_eval.sh",
    "tests/test_tvr_jobs.py", "tests/test_tvr_integration.py",
    "PREREG_TVR_PILOT.md", "tvr/pilot_cfg.py", "common/power_sim.py", "tests/test_tvr_prereg.py",
    "tests/test_power_sim.py",
    "walkerbench/perc.py", "jobs/wb_learn483.sh", "sco.py", "README.md", "tests/test_wave1c.py",
)
WAVE1B_ITEMS = ("`wb_perc.sh RUN [START ...]`", "perc_eval.lock", "VPD_PERC_TAKEOVER", "VPD_PERC_TEST_ALLOW_LIVE",
                "probe_lvp recommend", "`EXPLORATORY_WB_LEARN483.md`", "`walkerbench/learn483.py`",
                "`jobs/wb_learn483.sh`", "`tests/test_learn483.py`")
OUT_TVR = ("tvr/split/dev200/", "tvr/houses/", "tvr/reach/v1/", "tvr/replay/sft1600_lvp/", "tvr/demos/sft1600_lvp/",
           "tvr/tiers/v1/", "tvr/bfs/", "tvr/floors/", "tvr/runs/explore/cal/Q3B/", "tvr/runs/smoke/",
           "tvr/runs/pilot/<起点>/", "tvr/runs/pilot/")


def test_readme_covers_wave2_files_and_wave1b_items():
    text = README.read_text(encoding="utf-8")
    assert [f for f in SECTION2_FILES if f"`{f}`" not in text] == []
    assert [s for s in WAVE1B_ITEMS if s not in text] == []
    assert [d for d in OUT_TVR if f"`{d}" not in text] == []
    # the snapshot section describes the code_meta fallback instead of "worktree_head stays null"
    assert "worktree_head_source" in text and "仍是 null" not in text
    # every wave-2 row sits in the vpd/ table (a row starts with "| `<path>`")
    rows = [ln for ln in text.splitlines() if ln.startswith("| `")]
    for f in SECTION2_FILES:
        if f in ("walkerbench/perc.py", "sco.py", "README.md", "jobs/wb_learn483.sh"):
            continue
        assert any(f"`{f}`" in ln.split("|")[1] for ln in rows), f
