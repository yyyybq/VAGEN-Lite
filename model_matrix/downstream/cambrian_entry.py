"""Run a StarVLA script with Cambrian-S registered as a backbone (model_matrix/downstream/starvla_cambrian.py), the
script itself unchanged -- the downstream line's train entry, load test and StarVLA's policy server:

    python model_matrix/downstream/cambrian_entry.py SCRIPT [ARGS...]
    accelerate launch ... model_matrix/downstream/cambrian_entry.py SCRIPT [ARGS...]

As ``python SCRIPT ARGS`` would: the script's folder first on the path and ``sys.argv`` = [SCRIPT, ARGS...].  The
worktree (for vagen.models and model_matrix) follows; the stand-ins for two packages the Cambrian source imports and the
StarVLA env lacks (shims/: ezcolorlog, open_clip) go last, so installed packages win.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIMS = Path(__file__).resolve().with_name("shims")
CAMBRIAN_SRC = "/mnt/umm/users/yinbaiqiao/cambrian-s"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = str(Path(sys.argv[1]).resolve())
    sys.argv = [script, *sys.argv[2:]]
    drop = {"", str(ROOT), str(Path(__file__).resolve().parent)}  # not this wrapper's own folder
    sys.path[:] = [str(Path(script).parent), str(ROOT)] + [p for p in sys.path if p not in drop] + [str(SHIMS)]
    os.environ.setdefault("CAMBRIAN_SRC", CAMBRIAN_SRC)
    from model_matrix.downstream.starvla_cambrian import register

    register()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
