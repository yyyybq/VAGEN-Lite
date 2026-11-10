"""Tests of the code snapshots of vpd sco jobs (DECISIONS G1, 2026-11-10): vpd.snapshot, vpd.sco prepare with and
without --no-snapshot, the payload and launch.sh rewrites, vpd.paths.WORKTREE under VPD_WORKTREE, the relative
sourcing of vpd/jobs/common_env.sh, and the E3VS prereg guard inside a snapshot.

Hermetic: a throw-away git repo (vpd/ with copies of __init__.py, paths.py, snapshot.py and jobs/common_env.sh, plus
downstream/ and eval_external/ stubs) under eval_external_assets/vpd/scratch/snapshot/pytest_<rand>/ (env
VPD_SNAPSHOT_TEST_DIR overrides), removed at teardown.  The live worktree is never archived; sco is never called
(sco_job.submit is not used; prepare only writes files); launch.sh runs once with a fake nvidia-smi.
    cd $WT && OMP_NUM_THREADS=1 PY_Q3 -B -m pytest -q -p no:cacheprovider vpd/tests/test_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from vpd import paths, sco, snapshot

sco_job = sco.sco_job
JOBS = paths.VPD_PKG / "jobs"
PREREG = "vpd/PREREG_E3VS_PILOT.md"
GIT_ENV_KEYS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_CONFIG")


@pytest.fixture
def base():
    root = Path(os.environ.get("VPD_SNAPSHOT_TEST_DIR") or paths.scratch("snapshot"))
    d = root / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _clean_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("VPD_") and k not in GIT_ENV_KEYS}
    env.update(extra)
    return env


def git(repo: Path, *args: str) -> str:
    env = _clean_env(HOME=str(repo.parent), GIT_CONFIG_NOSYSTEM="1")
    r = subprocess.run(["git", "-c", "user.name=vpd-test", "-c", "user.email=vpd-test@example.invalid",
                        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, (args, r.stdout, r.stderr)
    return r.stdout.strip()


PAYLOAD_SH = """#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
cd "$VPD_WORKTREE"
echo "W=$VPD_WORKTREE"
echo "P=$PYTHONPATH"
echo "D=$(pwd)"
"$TEST_PY" -B -c 'from vpd import paths; print("PW=" + str(paths.WORKTREE))'
echo "PRE=$(vpd_snapshot_file vpd/PREREG_E3VS_PILOT.md)"
"""


@pytest.fixture
def repo(base):
    """A git repo shaped like the worktree: three commits; the prereg was last changed by the second (c2)."""
    r = base / "wt"
    (r / "vpd" / "jobs").mkdir(parents=True)
    (r / "vpd" / "tvr").mkdir()
    for f in ("__init__.py", "paths.py", "snapshot.py"):
        shutil.copy2(paths.VPD_PKG / f, r / "vpd" / f)
    shutil.copy2(JOBS / "common_env.sh", r / "vpd" / "jobs" / "common_env.sh")
    (r / "vpd" / "jobs" / "payload.sh").write_text(PAYLOAD_SH)
    (r / "vpd" / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    (r / PREREG).write_text("# E3VS prereg v1\n")
    (r / "vpd" / "tvr" / "SMOKE_CRITERIA.md").write_text("# criteria\n")
    (r / "downstream").mkdir()
    (r / "downstream" / "tool.py").write_text("X = 1\n")
    (r / "eval_external").mkdir()
    (r / "eval_external" / "common.py").write_text("Y = 2\n")
    (r / "docs").mkdir()
    (r / "docs" / "notes.md").write_text("outside the snapshot trees\n")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "c1")
    (r / PREREG).write_text("# E3VS prereg v2 (frozen)\n")
    git(r, "commit", "-q", "-am", "c2: freeze the prereg")
    (r / "downstream" / "tool.py").write_text("X = 3\n")
    git(r, "commit", "-q", "-am", "c3: unrelated change")
    assert C2(r) != HEAD(r)
    return r


def C2(repo: Path) -> str:
    """The commit that last changed the prereg (the second of the fixture's three)."""
    return git(repo, "log", "-1", "--format=%H", "--", PREREG)


def HEAD(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def at_repo(repo, monkeypatch):
    """vpd.sco prepare pointed at the temp repo: sco.WORKTREE, sco_job.WORKTREE and launch.sh's cd."""
    orig = sco_job.launch_script
    monkeypatch.setattr(sco, "WORKTREE", repo)
    monkeypatch.setattr(sco_job, "WORKTREE", repo)
    monkeypatch.setattr(sco_job, "launch_script",
                        lambda job, payload, worktree=None, as_root=False: orig(job, payload, worktree=repo,
                                                                               as_root=as_root))
    monkeypatch.setattr(sco_job, "code_state", lambda wt: {"commit": "fake", "branch": "fake",
                                                           "dirty_downstream_files": []})
    monkeypatch.setattr(sco, "vpd_code_state", lambda: {"commit": "fake", "vpd_status": [],
                                                        "vpd_tree_sha256": "0" * 64, "vpd_files": 0})
    return repo


def tracked(repo: Path) -> list:
    return sorted(git(repo, "ls-files", "--", *snapshot.TREES).splitlines())


# --- vpd.snapshot.create ---------------------------------------------------------------------------------------

def test_clean_tree_snapshot_contents(base, repo):
    dest = base / "run" / "code"
    rec = snapshot.create(repo, dest)
    files = sorted(str(p.relative_to(dest).as_posix()) for p in dest.rglob("*") if p.is_file())
    assert files == sorted(tracked(repo) + [snapshot.MARKER])  # docs/ (outside the trees) is not in it
    assert not (dest / ".git").exists() and not (dest / "docs").exists()
    assert (dest / "downstream" / "tool.py").read_text() == "X = 3\n" and (dest / PREREG).read_text().endswith(
        "v2 (frozen)\n")
    on_disk = json.loads((dest / snapshot.MARKER).read_text())
    assert on_disk == rec
    tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", HEAD(repo), *snapshot.TREES],
                         capture_output=True, check=True).stdout
    assert rec["commit"] == HEAD(repo) == rec["head_after"] and rec["format"] == 1
    assert rec["trees"] == list(snapshot.TREES) and rec["n_files"] == len(tracked(repo)) == len(rec["files"])
    assert rec["tar_sha256"] == hashlib.sha256(tar).hexdigest() and rec["repo"] == str(repo)
    assert rec["files"][PREREG] == git(repo, "rev-parse", f"HEAD:{PREREG}")
    # the frozen documents carry the commit that last changed them (c2, not HEAD)
    assert set(rec["pinned"]) == {PREREG, "vpd/tvr/SMOKE_CRITERIA.md"}
    assert rec["pinned"][PREREG] == {"blob": rec["files"][PREREG], "last_commit": C2(repo)}
    assert snapshot.summary(rec)["pinned"][PREREG] == C2(repo) and "files" not in snapshot.summary(rec)
    v = snapshot.verify(dest)
    assert v["ok"] and v["checked"] == rec["n_files"] and v["commit"] == HEAD(repo)
    # verify sees a changed, a missing and an extra file (bytecode caches are ignored)
    (dest / "vpd" / "__pycache__").mkdir()
    (dest / "vpd" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    assert snapshot.verify(dest)["ok"]
    (dest / "downstream" / "tool.py").write_text("X = 4\n")
    (dest / "eval_external" / "common.py").unlink()
    (dest / "vpd" / "new.py").write_text("")
    v = snapshot.verify(dest)
    assert not v["ok"] and v["changed"] == ["downstream/tool.py"] and v["missing"] == ["eval_external/common.py"]
    assert v["extra"] == ["vpd/new.py"]
    with pytest.raises(snapshot.SnapshotError, match="exists"):
        snapshot.create(repo, dest)


def test_dirty_tree_refused(base, repo, at_repo):
    payload = ["bash", str(repo / "vpd" / "jobs" / "payload.sh")]
    # a tracked change
    (repo / "eval_external" / "common.py").write_text("Y = 9\n")
    with pytest.raises(sco.Refused, match="SNAPSHOT REFUSED.*eval_external/common.py"):
        sco.prepare(base / "r1", "vpd-wb-dirty", payload, est_gph=0.1)
    assert not (base / "r1" / "code").exists() and not (base / "r1" / "code.partial").exists()
    assert not (base / "r1" / "job" / "launch.sh").exists()
    (repo / "eval_external" / "common.py").write_text("Y = 2\n")  # back to the committed bytes
    # an untracked file under vpd/
    (repo / "vpd" / "jobs" / "new_job.sh").write_text("true\n")
    with pytest.raises(snapshot.SnapshotError, match=r"\?\? vpd/jobs/new_job.sh"):
        snapshot.create(repo, base / "r2" / "code")
    assert sco.main(["prepare", "--run", str(base / "r2"), "--name", "vpd-wb-dirty", "--", *payload]) == 2
    (repo / "vpd" / "jobs" / "new_job.sh").unlink()
    # ignored files and files outside the three trees do not count
    (repo / "vpd" / "__pycache__").mkdir()
    (repo / "vpd" / "__pycache__" / "paths.cpython-312.pyc").write_bytes(b"\0")
    (repo / "docs" / "scratch.md").write_text("untracked, outside the trees\n")
    (repo / "docs" / "notes.md").write_text("changed, outside the trees\n")
    rec = snapshot.create(repo, base / "r3" / "code")
    assert rec["commit"] == HEAD(repo) and not (base / "r3" / "code" / "vpd" / "__pycache__").exists()


# --- vpd.sco prepare -------------------------------------------------------------------------------------------

def test_prepare_payload_and_launch_rewrite(base, at_repo):
    repo = at_repo
    run = base / "jobs" / "vpd-wb-snap"
    payload = ["env", "VPD_X=1", "timeout", "-k", "60", "600", "bash", str(repo / "vpd" / "jobs" / "payload.sh"),
               str(run / "out"), str(repo)]
    launch = sco.prepare(run, "vpd-wb-snap", payload, est_gph=0.2)
    code = run.resolve() / "code"
    assert json.loads((code / snapshot.MARKER).read_text())["commit"] == HEAD(repo)
    want = payload[:7] + [str(code / "vpd" / "jobs" / "payload.sh"), str(run / "out"), str(code)]
    prepared = json.loads((run / "job" / "prepared.json").read_text())
    assert prepared["payload"] == want and prepared["payload_as_given"] == payload
    assert prepared["worktree"] == str(code) and prepared["source_worktree"] == str(repo)
    assert prepared["code_snapshot"]["commit"] == HEAD(repo) and prepared["code_snapshot"]["n_files"] == len(
        tracked(repo))
    assert prepared["code_snapshot"]["path"] == str(code / snapshot.MARKER)
    meta = json.loads((run / "job" / "vpd.json").read_text())
    assert meta["snapshot"]["commit"] == HEAD(repo) and meta["no_snapshot"] is False
    assert meta["snapshot"]["pinned"][PREREG] == C2(repo) and meta["code_dir"] == str(code)
    assert meta["payload_rewrites"] == [[payload[7], want[7]], [payload[9], want[9]]]
    text = launch.read_text()
    lines = text.splitlines()
    assert f"cd {repo}" not in lines and lines.count(f"cd {code} && [[ -f SNAPSHOT.json ]] || "
                                                     f"{{ echo \"[launch] no code snapshot at \"{code} >&2; "
                                                     f"exit 97; }}") == 1
    assert f"export VPD_WORKTREE={code} PYTHONPATH={code}" in lines
    assert text.index("export VPD_WORKTREE") < text.index("exec > >(tee")
    # the payload line (sco_job quotes it) is the rewritten one; the only worktree path left is the owner wrapper
    assert " ".join(want) in text
    assert text.replace(str(repo / sco_job.RUN_AS_OWNER), "").count(str(repo)) == 0
    assert sco_job.create_command(run / "job", prepared)[-1] == f"--command=bash {run / 'job'}/launch.sh"


def test_launch_runs_the_snapshot(base, at_repo):
    """launch.sh end to end (as_root form: no owner wrapper; a fake nvidia-smi): the payload sees RUN/code."""
    repo = at_repo
    run = base / "jobs" / "vpd-wb-e2e"
    launch = sco.prepare(run, "vpd-wb-e2e", ["bash", str(repo / "vpd" / "jobs" / "payload.sh")], as_root=True,
                         est_gph=0.1)
    code = run.resolve() / "code"
    fake_bin = base / "bin"
    fake_bin.mkdir()
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\necho fake-gpu\n")
    (fake_bin / "nvidia-smi").chmod(0o755)
    # the worktree changes after prepare: the job must not see it
    (repo / "vpd" / "jobs" / "payload.sh").write_text("echo LIVE\n")
    env = _clean_env(PATH=f"{fake_bin}:{os.environ['PATH']}", TEST_PY=sys.executable, PYTHONPATH="/nonexistent")
    r = subprocess.run(["bash", str(launch)], capture_output=True, text=True, env=env, timeout=120)
    log = (run / "job" / "worker.log").read_text()
    assert (run / "job" / "exit_code").read_text().strip() == "0", log + r.stderr
    for line in (f"W={code}", f"P={code}", f"D={code}", f"PW={code}", f"PRE={C2(repo)}"):
        assert line in log.splitlines(), (line, log)
    assert "LIVE" not in log and "checked by git_show" in log
    # a launch.sh whose snapshot is gone stops before the payload
    shutil.move(str(code), str(run / "code_moved"))
    r = subprocess.run(["bash", str(launch)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 97 and "no code snapshot" in r.stderr


def test_prepare_refusals_with_snapshot(base, at_repo):
    repo = at_repo
    # an arg that only contains the worktree path, an arg outside the snapshot, a RUN inside the worktree
    cases = [(["bash", "-c", f"bash {repo}/vpd/jobs/payload.sh"], "does not start with it"),
             (["env", f"VPD_COMMON_ENV={repo}/vpd/jobs/common_env.sh", "true"], "does not start with it"),
             (["bash", str(repo / "docs" / "notes.md")], "not in the code snapshot"),
             (["bash", str(repo / "vpd" / "jobs" / "__pycache__" / "x.cpython-312.pyc")], "not in the code snapshot")]
    (repo / "vpd" / "jobs" / "__pycache__").mkdir()
    (repo / "vpd" / "jobs" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")  # ignored: tree stays clean
    for i, (payload, msg) in enumerate(cases):
        run = base / "jobs" / f"bad{i}"
        with pytest.raises(ValueError, match=msg):
            sco.prepare(run, "vpd-wb-bad", payload, est_gph=0.1)
        assert not (run / "code").exists() and not (run / "job" / "vpd.json").exists()
    with pytest.raises(ValueError, match="inside the worktree"):
        sco.prepare(repo / "vpd" / "run", "vpd-wb-in", ["true"], est_gph=0.1)
    # a submitted RUN is never re-prepared, and neither is a prepared one
    run = base / "jobs" / "done"
    sco.prepare(run, "vpd-wb-done", ["true"], est_gph=0.1)
    with pytest.raises(ValueError, match="prepared before"):
        sco.prepare(run, "vpd-wb-done", ["true"], est_gph=0.1)
    (run / "job" / "submission.json").write_text('{"job_id": "pt-x"}')
    with pytest.raises(FileExistsError):
        sco.prepare(run, "vpd-wb-done", ["true"], est_gph=0.1)


def test_no_snapshot(base, at_repo):
    repo = at_repo
    (repo / "vpd" / "jobs" / "uncommitted.sh").write_text("true\n")  # a dirty tree does not matter here
    run = base / "jobs" / "live"
    payload = ["bash", str(repo / "vpd" / "jobs" / "payload.sh")]
    launch = sco.prepare(run, "vpd-wb-live", payload, est_gph=0.1, snapshot_code=False)
    assert not (run / "code").exists()
    lines = launch.read_text().splitlines()
    assert f"cd {repo}" in lines and not any("VPD_WORKTREE" in ln for ln in lines)
    prepared = json.loads((run / "job" / "prepared.json").read_text())
    assert prepared["payload"] == payload and prepared["worktree"] == str(repo) and prepared["no_snapshot"] is True
    assert prepared["code_snapshot"] is None and "payload_as_given" not in prepared
    meta = json.loads((run / "job" / "vpd.json").read_text())
    assert meta["no_snapshot"] is True and meta["snapshot"] is None and meta["payload_rewrites"] == []
    assert sco.main(["prepare", "--run", str(base / "jobs" / "live2"), "--name", "vpd-wb-live2", "--no-snapshot",
                     "--", *payload]) == 0
    assert json.loads((base / "jobs" / "live2" / "job" / "vpd.json").read_text())["no_snapshot"] is True


def test_vpd_code_state_inside_a_snapshot(base, repo, monkeypatch):
    """In a job (no .git, no git binary) vpd.sco.vpd_code_state takes the commit from SNAPSHOT.json."""
    dest = base / "run" / "code"
    snapshot.create(repo, dest)
    monkeypatch.setattr(sco, "WORKTREE", dest)
    st = sco.vpd_code_state()
    assert st["commit"] == HEAD(repo) and st["vpd_status"] == [] and st["snapshot"]["mode"] == "snapshot"
    assert st["vpd_files"] == sum(1 for f in tracked(repo) if f.startswith("vpd/"))
    assert snapshot.snapshot_state(repo)["mode"] == "live"


# --- in-job checks: vpd_snapshot_file and the E3VS prereg guard ---------------------------------------------------

def _source_and_run(tree: Path, script: str, **extra) -> subprocess.CompletedProcess:
    env = _clean_env(**extra)
    return subprocess.run(["bash", "-c", f"source {tree / 'vpd/jobs/common_env.sh'}\n{script}"], capture_output=True,
                          text=True, env=env, timeout=120)


def test_snapshot_file_check(base, repo, monkeypatch):
    monkeypatch.delenv("VPD_SNAPSHOT_GIT", raising=False)
    code = base / "run" / "code"
    snapshot.create(repo, code)
    r = _source_and_run(code, f"vpd_snapshot_file {PREREG}")
    assert r.returncode == 0 and r.stdout.strip() == C2(repo) and "git_show" in r.stderr, r.stderr
    # no git binary (the job image): the record made at prepare time
    r = _source_and_run(code, f"vpd_snapshot_file {PREREG}", VPD_SNAPSHOT_GIT="")
    assert r.returncode == 0 and r.stdout.strip() == C2(repo) and "snapshot_record" in r.stderr, r.stderr
    assert snapshot.committed_file(PREREG, code) == (C2(repo), "git_show")
    # a file that is not pinned: fine with git, refused without
    assert snapshot.committed_file("downstream/tool.py", code)[0] == HEAD(repo)
    with monkeypatch.context() as m:
        m.setenv("VPD_SNAPSHOT_GIT", "")
        with pytest.raises(snapshot.SnapshotError, match="no git here"):
            snapshot.committed_file("downstream/tool.py", code)
    with pytest.raises(snapshot.SnapshotError, match="not in the code snapshot"):
        snapshot.committed_file("docs/notes.md", code)
    with pytest.raises(snapshot.SnapshotError, match="relative"):
        snapshot.committed_file("../wt/vpd/paths.py", code)
    # the job's copy changed after the snapshot: refused, with and without git
    (code / PREREG).write_text("# edited in the job\n")
    for extra in ({}, {"VPD_SNAPSHOT_GIT": ""}):
        r = _source_and_run(code, f"vpd_snapshot_file {PREREG}", **extra)
        assert r.returncode == 1 and r.stdout == "" and "changed after the snapshot" in r.stderr
    with pytest.raises(snapshot.SnapshotError, match="not a code snapshot"):
        snapshot.committed_file(PREREG, repo)
    assert snapshot.main(["verify", "--root", str(code)]) == 1


def _guard_fn(script: str, name: str) -> str:
    t = (JOBS / script).read_text()
    m = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", t)
    assert m, (script, name)
    return m.group(0)


def test_e3vs_prereg_guard_snapshot_and_live(base, repo):
    """e3vs_prep.sh's prereg_guard (the real function text): from a snapshot it writes the commit that last changed
    the prereg (as git log -1 did) and refuses an edited copy; from a git worktree the old git check still runs."""
    fn = _guard_fn("e3vs_prep.sh", "prereg_guard")
    code = base / "run" / "code"
    snapshot.create(repo, code)

    def guard(tree, run, **extra):
        run.mkdir(parents=True, exist_ok=True)
        return _source_and_run(tree, f"DRY=0\nRUN={run}\n{fn}prereg_guard\n", **extra)

    for i, extra in enumerate(({}, {"VPD_SNAPSHOT_GIT": ""})):
        r = guard(code, base / f"g{i}", **extra)
        assert r.returncode == 0, r.stderr
        assert (base / f"g{i}" / "prereg_commit.txt").read_text().strip() == C2(repo)
    (code / PREREG).write_text("# edited in the job\n")
    r = guard(code, base / "g_bad")
    assert r.returncode == 1 and "not committed or has local changes" in r.stderr
    assert not (base / "g_bad" / "prereg_commit.txt").exists()
    # live worktree (no SNAPSHOT.json): git log / git status as before
    r = guard(repo, base / "g_live")
    assert r.returncode == 0 and (base / "g_live" / "prereg_commit.txt").read_text().strip() == C2(repo), r.stderr
    (repo / PREREG).write_text("# local edit\n")
    r = guard(repo, base / "g_live_dirty")
    assert r.returncode == 1 and not (base / "g_live_dirty" / "prereg_commit.txt").exists()


def test_e3vs_start_guard_uses_the_same_check():
    """e3vs_start.sh's guard runs the same snapshot / live prereg block as e3vs_prep.sh's prereg_guard."""
    def block(script, name):
        fn = _guard_fn(script, name)
        return fn[fn.index("  if [[ -e $VPD_WORKTREE/SNAPSHOT.json ]]; then"):fn.index("  if [[ -z $c || -n $s ]]")]

    a, b = block("e3vs_prep.sh", "prereg_guard"), block("e3vs_start.sh", "guard")
    assert a == b and "vpd_snapshot_file vpd/PREREG_E3VS_PILOT.md" in a and "status --porcelain" in a


# --- paths.WORKTREE and common_env.sh ---------------------------------------------------------------------------

PATHS_PROBE = ("from vpd import paths; print(paths.WORKTREE); print(paths.VPD_PKG); print(paths.DOWNSTREAM); "
               "print(paths.EVAL_EXTERNAL); print(paths.STARTS_JSON); print(paths.GIT_WORKTREE); print(paths.OUT)")


def _probe(py, worktree_env, pythonpath=None):
    """PATHS_PROBE with `vpd` imported from pythonpath (default: the tree under test; also the cwd, which -c puts
    first on sys.path)."""
    tree = Path(pythonpath or paths.WORKTREE)
    env = _clean_env(PYTHONPATH=str(tree))
    if worktree_env is not None:
        env["VPD_WORKTREE"] = worktree_env
    return subprocess.run([str(py), "-B", "-c", PATHS_PROBE], capture_output=True, text=True, env=env, timeout=120,
                          cwd=str(tree))


def test_paths_worktree_env_override(base, repo):
    code = base / "run" / "code"
    snapshot.create(repo, code)
    r = _probe(sys.executable, str(code))
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [str(code), str(code / "vpd"), str(code / "downstream"),
                                     str(code / "eval_external"), str(code / "vpd" / "starts.json"),
                                     str(paths.GIT_WORKTREE), str(paths.OUT)]
    # the snapshot's own copy of paths.py (PYTHONPATH = RUN/code, as in a job) agrees
    r2 = _probe(sys.executable, str(code), pythonpath=code)
    assert r2.returncode == 0 and r2.stdout == r.stdout, r2.stderr
    # unset or empty: the git worktree
    for v in (None, ""):
        r = _probe(sys.executable, v)
        assert r.returncode == 0 and r.stdout.splitlines()[0] == str(paths.GIT_WORKTREE), r.stderr
    # not absolute, or no vpd/: refused at import
    for bad in ("relative/tree", str(base / "nowhere"), str(repo / "docs")):
        r = _probe(sys.executable, bad)
        assert r.returncode != 0 and "VPD_WORKTREE" in r.stderr
    assert paths.GIT_WORKTREE == paths.USER_ROOT / "VAGEN-Lite-vpd"


def test_paths_and_snapshot_import_on_py310(base, repo):
    """vpd.paths and vpd.snapshot are stdlib-only and import under the tvr env's python 3.10 (skipped without it)."""
    py = paths.TVR_ENV / "bin" / "python"
    if not py.exists():
        pytest.skip(f"no {py}")
    code = base / "run" / "code"
    snapshot.create(repo, code)
    env = _clean_env(PYTHONPATH=str(code), VPD_WORKTREE=str(code), VPD_SNAPSHOT_GIT="")
    r = subprocess.run([str(py), "-B", "-c", "import sys; from vpd import paths, snapshot; "
                        "print(sys.version_info[:2] >= (3, 10), paths.WORKTREE, snapshot.verify()['ok'], "
                        f"snapshot.committed_file({PREREG!r})[1])"], capture_output=True, text=True, env=env,
                       timeout=120, cwd=str(code))
    assert r.returncode == 0 and r.stdout.split() == ["True", str(code), "True", "snapshot_record"], r.stderr


def test_common_env_relative_sourcing(base):
    """A copied tree sources its own common_env.sh: VPD_WORKTREE / PYTHONPATH = that tree (vpd/jobs and vpd/tvr
    forms); a set VPD_WORKTREE is kept; one without vpd/ stops the payload."""
    tree = base / "copy"
    (tree / "vpd" / "jobs").mkdir(parents=True)
    (tree / "vpd" / "tvr").mkdir()
    shutil.copy2(JOBS / "common_env.sh", tree / "vpd" / "jobs" / "common_env.sh")
    show = 'echo "W=$VPD_WORKTREE P=$PYTHONPATH D=$(pwd)"\n'
    (tree / "vpd" / "jobs" / "x.sh").write_text(
        'source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"\ncd "$VPD_WORKTREE"\n' + show)
    (tree / "vpd" / "tvr" / "y.sh").write_text(
        'source "${VPD_COMMON_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../jobs" && pwd)/common_env.sh}"\n'
        'cd "$VPD_WORKTREE"\n' + show)
    want = f"W={tree} P={tree} D={tree}\n"
    for s in ("vpd/jobs/x.sh", "vpd/tvr/y.sh"):
        r = subprocess.run(["bash", str(tree / s)], capture_output=True, text=True, env=_clean_env(), timeout=60,
                           cwd=str(base))
        assert r.returncode == 0 and r.stdout == want, (s, r.stdout, r.stderr)
        # relative invocation from elsewhere
        r = subprocess.run(["bash", os.path.relpath(tree / s, base)], capture_output=True, text=True,
                           env=_clean_env(), timeout=60, cwd=str(base))
        assert r.returncode == 0 and r.stdout == want, (s, r.stdout, r.stderr)
    other = base / "other"
    (other / "vpd").mkdir(parents=True)
    r = subprocess.run(["bash", str(tree / "vpd/jobs/x.sh")], capture_output=True, text=True, timeout=60,
                       env=_clean_env(VPD_WORKTREE=str(other)))
    assert r.returncode == 0 and r.stdout == f"W={other} P={other} D={other}\n", r.stderr
    for bad in (str(base / "nowhere"), "relative"):
        r = subprocess.run(["bash", str(tree / "vpd/jobs/x.sh")], capture_output=True, text=True, timeout=60,
                           env=_clean_env(VPD_WORKTREE=bad))
        assert r.returncode == 2 and "VPD_WORKTREE" in r.stderr and r.stdout == ""


NOT_PAYLOADS = {"common_env.sh", "preflight.sh", "stage_tvr_login.sh"}  # the prelude and two login-node tools
SRC_JOBS = 'source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"'
SRC_TVR = ('source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../jobs" && pwd)/common_env.sh"',
           'source "${VPD_COMMON_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../jobs" && pwd)/common_env.sh}"')


def _code_lines(text: str) -> list:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_job_scripts_follow_the_conventions():
    """Every sco payload (vpd/jobs/*.sh, vpd/tvr/*.sh) sources the common_env.sh of its own tree, and no shell
    script hard-codes the worktree outside comments (usage comments keep the /mnt/.../VAGEN-Lite-vpd/vpd/... form
    that vpd.sco prepare rewrites to RUN/code).  vpd/jobs payloads also cd into $VPD_WORKTREE."""
    jobs = sorted(JOBS.glob("*.sh"))
    tvr = sorted((paths.VPD_PKG / "tvr").glob("*.sh"))
    assert {"wb_smoke.sh", "e3vs_prep.sh", "common_env.sh"} <= {p.name for p in jobs}
    assert {"tvr_smoke.sh", "debug_job.sh"} <= {p.name for p in tvr}
    for f in jobs + tvr:
        text = f.read_text()
        lines = text.splitlines()
        assert not [ln for ln in _code_lines(text) if "VAGEN-Lite-vpd" in ln], f
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)
        if f.name in NOT_PAYLOADS:
            continue
        src = [ln for ln in lines if ln.startswith("source ") and "common_env.sh" in ln]
        if f.parent == JOBS:
            assert src == [SRC_JOBS], (f, src)
            assert any(ln.startswith('cd "$VPD_WORKTREE"') for ln in lines), f
        else:
            assert len(src) == 1 and src[0] in SRC_TVR, (f, src)
