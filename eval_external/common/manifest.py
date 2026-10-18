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


def git_state(path: Path) -> dict:
    def run(*args: str) -> str:
        r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    return {"path": str(path), "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


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
