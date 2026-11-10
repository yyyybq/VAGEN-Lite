"""Tests of the git facts recorded inside a code snapshot (DECISIONS G1 follow-up, 2026-11-10, wave 1b) and of the
ply_gaussian_loader record of the E3VS renderer.

In a snapshot job RUN/code has no .git and the job image has no git, so the facts come from RUN/code/SNAPSHOT.json:
  - vpd.walkerbench.run_vpd.write_run_manifest: extra.vpd.git commit = the snapshot's, dirty False, read
    "SNAPSHOT.json", so smoke_vpd's manifest_fields check (its real expression, taken from smoke_vpd.py by ast)
    passes;
  - vpd.train.n2_train_vpd.git_head: git_head = the snapshot's commit, vpd_dirty False, git_read "SNAPSHOT.json";
  - vpd.e3vs.run.git_facts: commit = the snapshot's, prereg_commit from $RUN/prereg_commit.txt when the job wrote it.
A live worktree keeps the old facts.  vpd.e3vs.render.loader_facts records the ply_gaussian_loader module that was
imported (path, sha256, in_worktree), and SplatRenderer.meta carries it.

Hermetic: fake snapshot roots (copies of the eval CODE_FILES plus a SNAPSHOT.json record) under
eval_external_assets/vpd/scratch/snapshot_facts/pytest_<rand>/ (env VPD_SNAPSHOT_FACTS_TEST_DIR overrides), removed at
teardown; no GPU, no network, no sco; the live worktree is only read (git rev-parse / status).
    cd $WT && OMP_NUM_THREADS=1 PY_Q3 -B -m pytest -q -p no:cacheprovider vpd/tests/test_snapshot_facts.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from eval_external.common.manifest import git_state  # noqa: E402
from vpd import paths, snapshot  # noqa: E402

LIVE = paths.WORKTREE  # the tree under test (the git worktree when the suite runs from it)
COMMIT = "c0ffee" + "0" * 34


@pytest.fixture
def base():
    root = Path(os.environ.get("VPD_SNAPSHOT_FACTS_TEST_DIR") or paths.scratch("snapshot_facts"))
    d = root / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def fake_snapshot(base: Path, files=(), commit: str = COMMIT) -> Path:
    """RUN/code shaped like a snapshot: copies of `files` from the tree under test and a SNAPSHOT.json record."""
    root = base / "run" / "code"
    (root / "vpd").mkdir(parents=True)
    for rel in files:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LIVE / rel, dst)
    rec = {"format": snapshot.FORMAT, "commit": commit, "utc": "2026-11-10T00:00:00Z", "trees": list(snapshot.TREES),
           "extra_files": list(snapshot.EXTRA_FILES), "n_files": len(files), "tar_sha256": "0" * 64,
           "repo": str(paths.GIT_WORKTREE), "branch": "viewpoint-downstream", "head_after": commit,
           "files": {rel: snapshot.blob_id((root / rel).read_bytes()) for rel in files}, "pinned": {}}
    (root / snapshot.MARKER).write_text(json.dumps(rec, indent=1) + "\n")
    return root


def live_head() -> str | None:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(LIVE), "rev-parse", "HEAD"], capture_output=True,
                       text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


# --- run_vpd manifest + smoke_vpd manifest_fields -----------------------------------------------------------------

def smoke_manifest_fields():
    """The manifest_fields expression of vpd/walkerbench/smoke_vpd.py, compiled from its source (not a copy)."""
    src = (paths.VPD_PKG / "walkerbench" / "smoke_vpd.py").read_text()
    hits = [n.value for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript)
            and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "checks"
            and isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value == "manifest_fields"]
    assert len(hits) == 1, "smoke_vpd.py: expected one checks['manifest_fields'] = ... assignment"
    text = ast.unparse(hits[0])
    assert 'man["extra"]["vpd"]["git"].get("commit") is not None' in text.replace("'", '"'), text
    code = compile(ast.Expression(hits[0]), "smoke_vpd.manifest_fields", "eval")

    def check(man: dict, args, size) -> bool:
        return bool(eval(code, {"man": man, "args": args, "size": size, "con": man["prompt"]["constrained"]}))

    return check


def write_manifest(run_dir: Path) -> dict:
    from vpd.walkerbench import run_vpd as R

    opts = R.EvalOptions(constrained="empty", history_turns=2, max_tokens=512)
    R.write_run_manifest(run_dir, task_set="standard", n_tasks=6, diff="--- a\n+++ b\n", render_size=(640, 400),
                         opts=opts, server={"fake": True}, models=[{"code": "fake"}], workers=1, temperature=0.0,
                         model_dir=None, adapter=None, run_name="smoke__fake", lean=True, limit=6, lenient=False)
    return json.loads((run_dir / "manifest.json").read_text())


SMOKE_ARGS = SimpleNamespace(max_tokens=512, history_turns=2, constrained="empty")


def test_run_vpd_manifest_in_a_snapshot_passes_the_smoke_check(base, monkeypatch):
    from vpd.walkerbench import run_vpd as R

    root = fake_snapshot(base, R.CODE_FILES)
    assert git_state(root)["commit"] is None  # what a snapshot job's git_state reads: the reason for the fallback
    check = smoke_manifest_fields()
    monkeypatch.setattr(paths, "WORKTREE", root)
    man = write_manifest(base / "snap_run")
    g = man["extra"]["vpd"]["git"]
    assert g == {"path": str(root), "commit": COMMIT, "dirty": False, "read": snapshot.MARKER}
    assert set(man["extra"]["vpd"]["code_sha256"]) == set(R.CODE_FILES)
    assert all(man["extra"]["vpd"]["code_sha256"].values())  # hashed under RUN/code: the same bytes as live
    for rel, sha in man["extra"]["vpd"]["code_sha256"].items():
        assert sha == hashlib.sha256((LIVE / rel).read_bytes()).hexdigest(), rel
    assert check(man, SMOKE_ARGS, (640, 400))
    # an unreadable SNAPSHOT.json is recorded and the check fails (commit stays as git_state read it: None)
    (root / snapshot.MARKER).write_text("{not json")
    man = write_manifest(base / "bad_run")
    assert man["extra"]["vpd"]["git"]["commit"] is None and "unreadable" in man["extra"]["vpd"]["git"]["snapshot_error"]
    assert not check(man, SMOKE_ARGS, (640, 400))


def test_run_vpd_manifest_live_unchanged(base):
    """No SNAPSHOT.json: extra.vpd.git is exactly git_state(paths.WORKTREE), as before."""
    if (LIVE / snapshot.MARKER).exists():
        pytest.skip("the suite itself runs from a code snapshot")
    man = write_manifest(base / "live_run")
    g = man["extra"]["vpd"]["git"]
    assert g == git_state(LIVE) and g["commit"] is not None and g.get("read") != snapshot.MARKER
    assert smoke_manifest_fields()(man, SMOKE_ARGS, (640, 400))


# --- n2_train_vpd.git_head ----------------------------------------------------------------------------------------

def test_n2_git_head_snapshot_and_live(base):
    from vpd.train import n2_train_vpd as T

    root = fake_snapshot(base)
    assert T.git_head(root) == {"git_head": COMMIT, "vpd_dirty": False, "git_read": snapshot.MARKER}
    # not a snapshot and not a repo: None facts, as before (no git_read key)
    plain = base / "plain"
    (plain / "vpd").mkdir(parents=True)
    assert T.git_head(plain) == {"git_head": None, "vpd_dirty": None}
    # an unreadable record never fails a finished training: the live path (None here)
    (root / snapshot.MARKER).write_text("[]")
    assert T.git_head(root) == {"git_head": None, "vpd_dirty": None}
    # the live worktree: git as before
    if not (LIVE / snapshot.MARKER).exists():
        g = T.git_head(LIVE)
        assert set(g) == {"git_head", "vpd_dirty"} and g["git_head"] == live_head() and isinstance(g["vpd_dirty"], bool)
    # the fallback lives in a CODE_FILES file: its fingerprint is what meta.json records
    assert "vpd/train/n2_train_vpd.py" in T.CODE_FILES


# --- e3vs run.git_facts -------------------------------------------------------------------------------------------

def test_e3vs_git_facts_snapshot_and_live(base, monkeypatch):
    from vpd.e3vs import run as E

    root = fake_snapshot(base)
    run = root.parent
    monkeypatch.setattr(paths, "WORKTREE", root)
    monkeypatch.delenv("RUN", raising=False)
    want = {"commit": COMMIT, "vpd_status": [], "prereg_commit": None, "read": snapshot.MARKER,
            "prereg_commit_from": None}
    assert E.git_facts() == want  # RUN unset
    monkeypatch.setenv("RUN", str(run))
    assert E.git_facts() == want  # the guard has not written prereg_commit.txt
    (run / "prereg_commit.txt").write_text("not a commit\n")
    assert E.git_facts()["prereg_commit"] is None
    pre = "ab" * 20
    (run / "prereg_commit.txt").write_text(pre + "\n")  # as e3vs_prep.sh / e3vs_start.sh write it
    assert E.git_facts() == dict(want, prereg_commit=pre, prereg_commit_from=str(run / "prereg_commit.txt"))
    # the live worktree: git as before (no "read" key)
    monkeypatch.setattr(paths, "WORKTREE", LIVE)
    if not (LIVE / snapshot.MARKER).exists():
        g = E.git_facts()
        assert set(g) == {"commit", "vpd_status", "prereg_commit"} and g["commit"] == live_head()
        assert isinstance(g["vpd_status"], list)


# --- e3vs render: which ply_gaussian_loader ran ---------------------------------------------------------------------

def test_loader_facts_record_the_imported_module(base, monkeypatch):
    from vpd.e3vs import render as RD

    root = fake_snapshot(base)
    (root / "ply_gaussian_loader.py").write_text("X = 1\n")
    other = base / "site-packages" / "ply_gaussian_loader.py"
    other.parent.mkdir()
    other.write_text("X = 2\n")
    monkeypatch.setattr(paths, "WORKTREE", root)
    monkeypatch.setattr(RD, "_LOADER_FACTS", {})
    for f, inside in ((root / "ply_gaussian_loader.py", True), (other, False)):
        mod = types.ModuleType("ply_gaussian_loader")
        mod.__file__ = str(f)
        monkeypatch.setitem(sys.modules, "ply_gaussian_loader", mod)
        got = RD.loader_facts()
        assert got == {"module": "ply_gaussian_loader", "path": str(f.resolve()),
                       "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "in_worktree": inside}
    # parity records it next to its code_meta (parity itself needs a GPU)
    src = (paths.VPD_PKG / "e3vs" / "render.py").read_text()
    assert 'res["code"]["ply_gaussian_loader"] = loader_facts()' in src


def _write_ply(path: Path, n: int = 50):
    import numpy as np
    from plyfile import PlyData, PlyElement

    rng = np.random.default_rng(0)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
             "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(n, dtype=[(k, "f4") for k in names])
    for k in names:
        arr[k] = rng.normal(size=n).astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def test_splat_renderer_meta_names_the_loader(base):
    """SplatRenderer (CPU load only, no rasterisation) records the loader module it imported."""
    pytest.importorskip("torch")
    pytest.importorskip("plyfile")
    pytest.importorskip("ply_gaussian_loader")
    from vpd.e3vs import render as RD

    ply = base / "tiny.ply"
    _write_ply(ply)
    r = RD.SplatRenderer(ply=ply, device="cpu")
    import ply_gaussian_loader as mod

    p = Path(mod.__file__).resolve()
    lf = r.meta["ply_gaussian_loader"]
    assert lf["path"] == str(p) and lf["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert lf["in_worktree"] == p.is_relative_to(paths.WORKTREE.resolve())
    json.dumps(r.meta)  # goes into occupancy npz meta / SFT manifests as JSON
