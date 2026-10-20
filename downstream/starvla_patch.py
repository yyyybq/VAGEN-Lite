#!/usr/bin/env python3
"""The StarVLA working-tree changes as one patch file, and the check that the checkout still equals base + patch.

    python downstream/starvla_patch.py export   # write downstream/patches/starvla_<commit>_working.patch
    python downstream/starvla_patch.py check    # commit, patch sha256, `git apply --reverse --check` (JSON)

The nested checkout third_party/starVLA is not in the parent repository's git.  `export` diffs the tracked files
(`git diff HEAD`) and adds every untracked file outside playground/ as a new-file diff (`git diff --no-index`),
without touching the nested repository's index.  `check` is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

STARVLA = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA")
PATCH_DIR = Path(__file__).resolve().parent / "patches"
EXCLUDE = ("playground/",)


def git(*args, cwd=STARVLA, check=True):
    res = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=cwd, capture_output=True, text=True)
    if check and res.returncode not in (0, 1):  # diff --no-index returns 1 when files differ
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res


def untracked(repo: Path = STARVLA):
    out = git("--no-optional-locks", "status", "--porcelain", "--untracked-files=all", cwd=repo).stdout
    files = [line[3:] for line in out.splitlines() if line.startswith("?? ")]
    return sorted(f for f in files if not f.startswith(EXCLUDE) and "__pycache__" not in f and not f.endswith(".pyc")
                  and ".egg-info/" not in f)


def build_patch(repo: Path = STARVLA) -> str:
    parts = [git("--no-optional-locks", "diff", "HEAD", "--binary", cwd=repo).stdout]
    for rel in untracked(repo):
        diff = git("--no-optional-locks", "diff", "--no-index", "--binary", "--", "/dev/null", rel, cwd=repo).stdout
        parts.append(diff)
    return "".join(parts)


def head(repo: Path = STARVLA) -> str:
    return git("rev-parse", "HEAD", cwd=repo).stdout.strip()


def patch_path(commit: str) -> Path:
    return PATCH_DIR / f"starvla_{commit[:7]}_working.patch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(repo: Path = STARVLA) -> dict:
    commit = head(repo)
    path = patch_path(commit)
    if not path.is_file():
        return {"ok": False, "commit": commit, "error": f"no patch file {path}"}
    res = subprocess.run(["git", "apply", "--reverse", "--check", str(path)], cwd=repo, capture_output=True, text=True)
    current = build_patch(repo)
    same = current == path.read_text()
    return {"ok": res.returncode == 0 and same, "commit": commit, "patch": str(path), "patch_sha256": sha256(path),
            "reverse_check": res.returncode == 0, "reverse_check_stderr": res.stderr.strip()[:500],
            "working_tree_equals_patch": same}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["export", "check"])
    ap.add_argument("--repo", type=Path, default=STARVLA)
    args = ap.parse_args(argv)
    if args.action == "export":
        commit = head(args.repo)
        PATCH_DIR.mkdir(parents=True, exist_ok=True)
        path = patch_path(commit)
        path.write_text(build_patch(args.repo))
        print(json.dumps({"commit": commit, "patch": str(path), "sha256": sha256(path),
                          "untracked_included": untracked(args.repo)}, indent=1))
        return 0
    result = check(args.repo)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
