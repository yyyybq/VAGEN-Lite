#!/usr/bin/env python3
"""N series: corrupted camera-A renders (found 2026-11-03 while building N2's demonstrations).

Some simulator renders come out as colour noise (stripes of unrelated colours) instead of the kitchen: in N2's
demonstration generation 419 of 5,914 path frames so far, and in N1's evaluation every frame of some episodes
(14 of 300 instances in at least one model's job; N1's second set 6, all in the Qwen3 jobs).  The same poses render
cleanly in a fresh process, so it is a state of the process or the job, not of the scene; the jobs' simulator
processes share one Mesa shader cache (sim_env.sh: MESA_SHADER_CACHE_DIR=/tmp/ds-mesa-$UID), which is the suspected
cause (not verified).  A corrupted frame is told by its colour noise: the mean, over horizontally adjacent pixels, of
how differently the three channels change (chroma()); real renders, marble textures included, stay under 1, the
corrupted ones seen so far are 33 to 132.  CORRUPT = 8.

    python downstream/n/n_frames.py scan --eval EVAL_DIR [EVAL_DIR ...] --out REPORT.json

scan: every OUT/<model>/frames/<instance>.png strip (one square tile per turn) of n_episode.py evaluations: per
  instance and model the corrupted tiles of all tiles, and the instances with any.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

CORRUPT = 8.0


def chroma(rgb) -> float:
    """Mean over horizontally adjacent pixels of (|dR - dG| + |dG - dB|) / 2, on 0..255 values."""
    import numpy as np

    a = np.asarray(rgb, dtype=float)
    d = np.diff(a[..., :3], axis=1)
    return float((np.abs(d[..., 0] - d[..., 1]) + np.abs(d[..., 1] - d[..., 2])).mean() / 2)


def image(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def corrupted(path, threshold: float = CORRUPT) -> bool:
    return chroma(image(path)) > threshold


def tiles(rgb) -> list:
    """The square tiles of a frame strip (n_episode.strip_image: one tile per turn, side = the strip's height)."""
    import numpy as np

    a = np.asarray(rgb)
    side = a.shape[0]
    return [a[:, k * side:(k + 1) * side] for k in range(a.shape[1] // side)]


def scan(evals, threshold: float = CORRUPT) -> dict:
    out = {"threshold": threshold, "evals": {}}
    for ev in evals:
        per = collections.defaultdict(dict)
        n_tiles = n_bad = 0
        for p in sorted(Path(ev).glob("*/frames/*.png")):
            model, inst = p.parent.parent.name, p.stem
            c = [chroma(t) for t in tiles(image(p))]
            bad = sum(x > threshold for x in c)
            n_tiles, n_bad = n_tiles + len(c), n_bad + bad
            if bad:
                per[inst][model] = {"corrupted": bad, "tiles": len(c), "first": next(i for i, x in enumerate(c)
                                                                                      if x > threshold)}
        out["evals"][str(ev)] = {"tiles": n_tiles, "corrupted_tiles": n_bad, "instances": dict(per),
                                 "episodes": sum(len(v) for v in per.values())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--eval", nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rep = scan(args.eval)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    for ev, r in rep["evals"].items():
        print(f"{ev}: {r['corrupted_tiles']} of {r['tiles']} tiles corrupted, {r['episodes']} episodes in "
              f"{len(r['instances'])} instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
