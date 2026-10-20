#!/usr/bin/env python3
"""Versions and import status of the packages the downstream line needs, in one Python environment (CPU, read-only).

    /path/to/env/bin/python downstream/inventory/env_probe.py --out env.json [--timeout 240]

Distribution versions come from package metadata (no import).  Each module is then imported in its own
subprocess with a timeout, so one hanging import (e.g. a renderer probing for a display) is reported as a
timeout instead of stopping the probe.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import subprocess
import sys
import time
from pathlib import Path

DISTS = ["torch", "torchvision", "transformers", "accelerate", "deepspeed", "flash-attn", "flash_attn",
         "qwen-vl-utils", "lerobot", "av", "decord", "mujoco", "robosuite", "robocasa", "ray", "hydra-core",
         "omegaconf", "rlinf", "starVLA", "numpy", "numba", "gymnasium", "PyOpenGL", "safetensors", "peft",
         "pyarrow", "pandas", "imageio", "opencv-python", "opencv-python-headless", "websockets", "msgpack", "mink",
         "robosuite_models", "mimicgen", "openpi", "jax"]
MODS = ["torch", "transformers", "flash_attn", "deepspeed", "accelerate", "mujoco", "robosuite", "robocasa",
        "robocasa.utils.dataset_registry_utils", "ray", "hydra", "rlinf", "starVLA", "lerobot", "av", "decord",
        "qwen_vl_utils", "gymnasium"]
CODE = ("import importlib,sys; m=importlib.import_module(sys.argv[1]); "
        "print(getattr(m,'__version__','') or '', getattr(m,'__file__','') or '')")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=240)
    args = ap.parse_args()
    out = {"python": sys.version.split()[0], "executable": sys.executable, "dists": {}, "imports": {}}
    for d in DISTS:
        try:
            out["dists"][d] = md.version(d)
        except Exception:
            pass
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for m in MODS:
        t0 = time.time()
        try:
            res = subprocess.run([sys.executable, "-c", CODE, m], capture_output=True, text=True, timeout=args.timeout)
            if res.returncode == 0:
                line = (res.stdout.strip().splitlines() or [""])[-1]
                out["imports"][m] = f"ok ({time.time() - t0:.0f}s) {line[:160]}"
            else:
                err = [x for x in res.stderr.strip().splitlines() if x.strip()]
                out["imports"][m] = "FAIL " + (err[-1][:200] if err else f"rc={res.returncode}")
        except subprocess.TimeoutExpired:
            out["imports"][m] = f"TIMEOUT after {args.timeout:.0f}s"
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
