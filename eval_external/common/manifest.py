"""The manifest every eval_external run writes next to its raw predictions (manifest.json).

It records: the models (code, origin, weight hash, view, chat template hash), the benchmark (name, source,
revision, number of items), the backend (name and commit), the subset seed, the prompt template, the number of
frames, the decoding parameters, the command line, the eval_external commit and the UTC date.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent


def _read_head(path: Path) -> str | None:
    """HEAD's commit read from the .git files (the sco image has no git binary); handles worktrees and
    packed refs."""
    dot = Path(path) / ".git"
    if dot.is_file():
        gitdir = Path(dot.read_text().split("gitdir:", 1)[1].strip())
        gitdir = gitdir if gitdir.is_absolute() else (Path(path) / gitdir)
    elif dot.is_dir():
        gitdir = dot
    else:
        return None
    common = gitdir
    if (gitdir / "commondir").exists():
        common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    for base in (gitdir, common):
        if (base / ref).exists():
            return (base / ref).read_text().strip()
    packed = common / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return None


def git_state(path: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"path": str(path), "commit": _read_head(path), "dirty": None, "read": ".git files (no git binary)"}
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"path": str(path), "commit": commit, "dirty": bool(status)}


def write_manifest(run_dir: Path, *, models: list[dict], benchmark: dict, backend: dict, prompt: dict,
                   frames: int | None, decoding: dict, subset: dict | None = None, extra: dict | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "command": sys.argv,
        "eval_external": git_state(EVAL_ROOT),
        "models": models,
        "benchmark": benchmark,
        "backend": backend,
        "subset": subset or {"kind": "full", "seed": None},
        "prompt": prompt,
        "frames": frames,
        "decoding": decoding,
        "env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE", "LMUData", "CUDA_VISIBLE_DEVICES")},
    }
    if extra:
        out["extra"] = extra
    path = run_dir / "manifest.json"
    with open(path, "w", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path
