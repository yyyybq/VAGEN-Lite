"""Code snapshots of the VPD sco jobs (DECISIONS G1, 2026-11-10): a job runs a frozen copy of the code, never the
live worktree, so code edits made while it runs (three lines share the worktree) cannot change what it executes.

`vpd.sco prepare` calls create(WORKTREE, RUN/code).  It refuses unless
    git status --porcelain --untracked-files=all -- vpd downstream eval_external ply_gaussian_loader.py
is empty (ignored files such as __pycache__ do not count), extracts
`git archive <HEAD> vpd downstream eval_external ply_gaussian_loader.py` into RUN/code, checks every extracted file
against its git blob id, and writes RUN/code/SNAPSHOT.json:
    {"format": 1, "commit", "utc", "trees", "extra_files", "n_files", "tar_sha256", "repo" (the worktree it came
     from), "branch", "head_after", "files": {relpath: git blob id}, "pinned": {relpath: {"blob", "last_commit"}}}
EXTRA_FILES are tracked files at the worktree root that job code imports: ply_gaussian_loader.py
(vpd/e3vs/render.py `from ply_gaussian_loader import PLYGaussianLoader`).  Live jobs (cwd = PYTHONPATH = the
worktree) imported the worktree's copy; with it in the snapshot, a snapshot job (PYTHONPATH = RUN/code) imports the
same bytes instead of falling through to the eval-external site-packages copy (review_infra, 2026-11-10).  A commit
without one of them is refused.  "extra_files" is additive (format stays 1): a record without the key has none.
"pinned" covers the frozen documents (vpd/**/PREREG_*.md, vpd/**/SMOKE_CRITERIA.md); last_commit is
`git log -1 --format=%H <commit> -- relpath`, recorded at prepare time because the job image has no git binary
(downstream/sco_job.py; the in-job git facts of every vpd job so far are None).

In a job (launch.sh exports VPD_WORKTREE = PYTHONPATH = RUN/code; vpd.paths.WORKTREE follows):
    python -m vpd.snapshot file vpd/PREREG_E3VS_PILOT.md   # print the commit that last changed the file; exit 1
                                                           # unless the job's copy is the committed file
    python -m vpd.snapshot verify                          # every recorded file unchanged, none missing or extra
    python -m vpd.snapshot state                           # JSON: the snapshot record summary, or "live"
`file` compares the job's bytes with the recorded blob id, and, when a git binary is available (login node; env
VPD_SNAPSHOT_GIT overrides the executable, "" disables it), also with `git -C <repo> show <commit>:<path>` and
`git log -1 <commit> -- <path>` of the real worktree (readable from jobs).  Without git it uses the recorded
last_commit.  Code fingerprints (n2_train_vpd CODE_FILES, run_vpd code_sha256) are unchanged by snapshots: the bytes
are the same, only the root differs.
Stdlib only; importable from py3.10 (tvr env) and py3.12.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vpd import paths

TREES = ("vpd", "downstream", "eval_external")
EXTRA_FILES = ("ply_gaussian_loader.py",)  # tracked root files that job code imports (module docstring)
MARKER = "SNAPSHOT.json"
FORMAT = 1
PINNED_RE = re.compile(r"^vpd/(?:.+/)?(?:PREREG_[^/]*\.md|SMOKE_CRITERIA\.md)$")
GIT_TIMEOUT_S = 120
_SKIP_DIRS = {"__pycache__", ".pytest_cache"}


class SnapshotError(ValueError):
    """A snapshot was refused, or a snapshot check failed; the message says why."""


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def blob_id(data: bytes, hex_len: int = 40) -> str:
    """git's object id of a blob with these bytes (sha1; sha256 for a 64-hex object format)."""
    h = hashlib.sha256() if hex_len == 64 else hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def git_exe() -> Optional[str]:
    """The git executable: env VPD_SNAPSHOT_GIT when set ("" = none), else git on PATH (None in the job image)."""
    v = os.environ.get("VPD_SNAPSHOT_GIT")
    if v is not None:
        return v or None
    return shutil.which("git")


def _git(repo, *args: str, exe: Optional[str] = None, raw: bool = False):
    exe = exe or git_exe()
    if not exe:
        raise SnapshotError("no git executable")
    cmd = [exe, "--no-optional-locks", "-c", "safe.directory=*", "-C", str(repo), *args]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SnapshotError(f"git {' '.join(args[:2])} in {repo}: {e!r}") from None
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace").strip()[:400]
        raise SnapshotError(f"git {' '.join(args[:2])} in {repo} failed (rc {r.returncode}): {err}")
    return r.stdout if raw else r.stdout.decode()


# --- create (login node, vpd.sco prepare) ---------------------------------------------------------------------

def where(trees=TREES, extra_files=EXTRA_FILES) -> str:
    """'vpd/, downstream/, eval_external/, ply_gaussian_loader.py': what a snapshot holds, for messages."""
    return ", ".join([t + "/" for t in trees] + list(extra_files))


def dirty_files(worktree, trees=TREES, extra_files=EXTRA_FILES) -> List[str]:
    """Porcelain lines of tracked changes and untracked, non-ignored files under the trees and of the extra files."""
    out = _git(worktree, "status", "--porcelain", "--untracked-files=all", "--", *trees, *extra_files)
    return [ln for ln in out.splitlines() if ln.strip()]


def _ls_tree(worktree, commit: str, trees, extra_files=()) -> Dict[str, str]:
    out = _git(worktree, "ls-tree", "-r", "-z", "--full-tree", commit, "--", *trees, *extra_files)
    files, odd = {}, []
    for ent in out.split("\0"):
        if not ent:
            continue
        meta, path = ent.split("\t", 1)
        mode, kind, obj = meta.split()
        if kind != "blob" or mode not in ("100644", "100755"):
            odd.append(f"{mode} {kind} {path}")
            continue
        files[path] = obj
    if odd:
        raise SnapshotError("SNAPSHOT REFUSED: symlinks and submodules are not supported in a code snapshot: "
                            + "; ".join(odd[:10]))
    missing = [t for t in trees if not any(p.startswith(t + "/") for p in files)]
    if missing:
        raise SnapshotError(f"SNAPSHOT REFUSED: {commit[:12]} has no files under {missing}")
    missing = [f for f in extra_files if f not in files]
    if missing:
        raise SnapshotError(f"SNAPSHOT REFUSED: {commit[:12]} has no tracked file {missing} (EXTRA_FILES)")
    return files


def _extract(tar_bytes: bytes, dest: Path, trees, extra_files=()) -> int:
    """Extract the archive into dest (data filter where available); returns the number of regular files."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        members = tf.getmembers()
        for m in members:
            parts = Path(m.name).parts
            inside = bool(parts) and (parts[0] in trees or (m.name in extra_files and m.isfile()))
            if (not inside or ".." in parts or Path(m.name).is_absolute()
                    or not (m.isfile() or m.isdir())):
                raise SnapshotError(f"unexpected archive member {m.name!r} (type {m.type!r})")
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, members=members, filter="data")
        else:  # older 3.10 builds without extraction filters; the members were checked above
            tf.extractall(dest, members=members)
        return sum(1 for m in members if m.isfile())


def create(worktree, dest, trees=TREES, extra_files=EXTRA_FILES) -> dict:
    """Snapshot `trees` and `extra_files` of the worktree's HEAD into dest (which must not exist); returns the
    SNAPSHOT.json record.  Raises SnapshotError (dirty tree, no git, odd entries, a missing extra file, bytes that
    do not match their blob ids)."""
    worktree, dest = Path(worktree), Path(dest)
    trees, extra_files = tuple(trees), tuple(extra_files)
    partial = dest.with_name(dest.name + ".partial")
    for p in (dest, partial):
        if p.exists() or p.is_symlink():
            raise SnapshotError(f"{p} exists: prepare a new run directory")
    top = _git(worktree, "rev-parse", "--show-toplevel").strip()
    if Path(top).resolve() != worktree.resolve():
        raise SnapshotError(f"{worktree} is not the top level of its git worktree ({top})")
    commit = _git(worktree, "rev-parse", "--verify", "HEAD^{commit}").strip()
    dirty = dirty_files(worktree, trees, extra_files)
    if dirty:
        more = f" ... ({len(dirty)} in all)" if len(dirty) > 20 else ""
        raise SnapshotError(f"SNAPSHOT REFUSED: uncommitted or untracked files in {where(trees, extra_files)} of "
                            f"{worktree}; commit them first (--no-snapshot is for tests and emergencies only): "
                            + "; ".join(dirty[:20]) + more)
    files = _ls_tree(worktree, commit, trees, extra_files)
    tar_bytes = _git(worktree, "archive", "--format=tar", commit, *trees, *extra_files, raw=True)
    rec = {"format": FORMAT, "commit": commit, "utc": utc(), "trees": list(trees), "extra_files": list(extra_files),
           "n_files": None, "tar_sha256": hashlib.sha256(tar_bytes).hexdigest(), "repo": str(worktree),
           "branch": None, "head_after": None, "files": files, "pinned": {}}
    partial.mkdir(parents=True)
    try:
        n = _extract(tar_bytes, partial, trees, extra_files)
        if n != len(files):
            raise SnapshotError(f"git archive gave {n} files, git ls-tree {len(files)}")
        bad = [rel for rel, obj in files.items()
               if not (partial / rel).is_file() or blob_id((partial / rel).read_bytes(), len(obj)) != obj]
        if bad:
            raise SnapshotError(f"extracted bytes differ from their git blobs (export attributes?): {bad[:10]}")
        rec["n_files"] = n
        for rel, obj in sorted(files.items()):
            if PINNED_RE.match(rel):
                last = _git(worktree, "log", "-1", "--format=%H", commit, "--", rel).strip()
                rec["pinned"][rel] = {"blob": obj, "last_commit": last or None}
        try:
            rec["branch"] = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() or None
            rec["head_after"] = _git(worktree, "rev-parse", "HEAD").strip() or None  # a commit made meanwhile
        except SnapshotError:
            pass
        (partial / MARKER).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        os.rename(partial, dest)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return rec


def summary(rec: dict) -> dict:
    """The part of a SNAPSHOT.json record that goes into vpd.json / prepared.json (no per-file map)."""
    out = {k: rec.get(k) for k in ("format", "commit", "utc", "trees", "n_files", "tar_sha256", "repo", "branch",
                                   "head_after")}
    out["extra_files"] = list(rec.get("extra_files") or [])
    out["pinned"] = {rel: p.get("last_commit") for rel, p in (rec.get("pinned") or {}).items()}
    return out


# --- in a job -------------------------------------------------------------------------------------------------

def _root(root) -> Path:
    return Path(root) if root is not None else paths.WORKTREE


def read_marker(root=None) -> Optional[dict]:
    """The SNAPSHOT.json record of root (default paths.WORKTREE), None when root is not a code snapshot."""
    p = _root(root) / MARKER
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise SnapshotError(f"{p}: unreadable ({e!r})") from None
    if not isinstance(rec, dict) or rec.get("format") != FORMAT or not rec.get("commit") \
            or not isinstance(rec.get("files"), dict):
        raise SnapshotError(f"{p}: not a vpd code snapshot record")
    return rec


def snapshot_state(root=None) -> dict:
    """{"mode": "snapshot", "root", <summary>} for a code snapshot, else {"mode": "live", "root"}."""
    root = _root(root)
    rec = read_marker(root)
    if rec is None:
        return {"mode": "live", "root": str(root)}
    return {"mode": "snapshot", "root": str(root), **summary(rec)}


def _walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for f in filenames:
            if f.endswith(".pyc"):
                continue
            yield Path(dirpath, f).relative_to(root).as_posix()


def verify(root=None) -> dict:
    """{"ok", "root", "commit", "checked", "changed", "missing", "extra"} of a code snapshot against its record
    (every recorded file, the EXTRA_FILES at the snapshot root included; any other file is "extra")."""
    root = _root(root)
    rec = read_marker(root)
    if rec is None:
        raise SnapshotError(f"{root} has no {MARKER} (not a code snapshot)")
    changed, missing = [], []
    for rel, obj in sorted(rec["files"].items()):
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif blob_id(p.read_bytes(), len(obj)) != obj:
            changed.append(rel)
    extra = sorted(r for r in _walk_files(root) if r != MARKER and r not in rec["files"])
    return {"ok": not (changed or missing or extra), "root": str(root), "commit": rec["commit"],
            "checked": len(rec["files"]), "changed": changed, "missing": missing, "extra": extra}


def committed_file(rel: str, root=None) -> Tuple[str, str]:
    """(commit that last changed rel, method) when the job's copy of rel is the committed file of the snapshot's
    commit; raises SnapshotError otherwise.  method: "git_show" (checked against the real worktree's git) or
    "snapshot_record" (no git here: the blob id and last_commit recorded at prepare time)."""
    root = _root(root)
    rec = read_marker(root)
    if rec is None:
        raise SnapshotError(f"{root} has no {MARKER} (not a code snapshot)")
    rp = Path(rel)
    if rp.is_absolute() or ".." in rp.parts:
        raise SnapshotError(f"{rel!r}: give the path relative to the worktree root")
    rel = rp.as_posix()
    commit = rec["commit"]
    obj = rec["files"].get(rel)
    if obj is None:
        raise SnapshotError(f"{rel} is not in the code snapshot of {commit[:12]}")
    p = root / rel
    try:
        data = p.read_bytes()
    except OSError as e:
        raise SnapshotError(f"{p}: {e!r}") from None
    if blob_id(data, len(obj)) != obj:
        raise SnapshotError(f"{p} differs from {rel} at {commit[:12]} (changed after the snapshot)")
    pin = (rec.get("pinned") or {}).get(rel) or {}
    exe, repo = git_exe(), rec.get("repo")
    if exe and repo and Path(repo).is_dir():
        committed = _git(repo, "show", f"{commit}:{rel}", exe=exe, raw=True)
        if committed != data:
            raise SnapshotError(f"{p} differs from `git show {commit[:12]}:{rel}` in {repo}")
        last = _git(repo, "log", "-1", "--format=%H", commit, "--", rel, exe=exe).strip()
        if not last:
            raise SnapshotError(f"{rel}: no commit of {repo} changed it up to {commit[:12]}")
        if pin.get("last_commit") and pin["last_commit"] != last:
            raise SnapshotError(f"{rel}: git says last changed in {last}, {MARKER} says {pin['last_commit']}")
        return last, "git_show"
    if not pin.get("last_commit") or pin.get("blob") != obj:
        raise SnapshotError(f"no git here and {MARKER} records no last commit of {rel}")
    return pin["last_commit"], "snapshot_record"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="vpd code snapshots (DECISIONS G1)")
    ap.add_argument("action", choices=["file", "verify", "state"])
    ap.add_argument("path", nargs="?", help="file: the path relative to the worktree root")
    ap.add_argument("--root", type=Path, default=None, help="default: vpd.paths.WORKTREE (VPD_WORKTREE)")
    args = ap.parse_args(argv)
    try:
        if args.action == "file":
            if not args.path:
                ap.error("file needs a path")
            last, how = committed_file(args.path, args.root)
            print(f"[vpd.snapshot] {args.path}: committed file, last changed in {last} (checked by {how})",
                  file=sys.stderr)
            print(last)
        elif args.action == "verify":
            res = verify(args.root)
            print(json.dumps(res, indent=1))
            return 0 if res["ok"] else 1
        else:
            print(json.dumps(snapshot_state(args.root), indent=1))
    except SnapshotError as e:
        print(f"[vpd.snapshot] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
