#!/usr/bin/env python3
"""Corrupted episodes out (2026-11-03): the episodes of an n_episode.py folder whose camera-A frames are colour noise
(n_frames.py) are moved aside, so that the resumable runner (run_n_episodes.sh) plays them again.

    python downstream/n/n_repair.py drop --out OUT [--threshold 8]

drop: every episode of OUT/episodes.jsonl whose frame strip OUT/frames/<instance>.png has a corrupted tile leaves
  episodes.jsonl for OUT/episodes_corrupted.jsonl (appended, with its corrupted / total tiles and the time), and its
  strip moves to OUT/frames_corrupted/<instance>.<n>.png (n counts earlier moves of the same instance).  Episodes
  without a strip stay.  Nothing is deleted.  Prints {"dropped": k, "kept": m}.
11-03: the corruption showed only in the real jobs (12 simulator processes at once), not in a diagnostic job with 4
(pt-88mtzres); run_n2_repair.sh plays the dropped episodes again with fewer at once.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_frames as F  # noqa: E402


def drop(out: Path, threshold: float = F.CORRUPT) -> dict:
    out = Path(out)
    path = out / "episodes.jsonl"
    if not path.is_file():
        return {"dropped": 0, "kept": 0}
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    keep, gone = [], []
    for r in rows:
        strip = out / "frames" / f"{r.get('instance_id')}.png"
        tiles = [F.chroma(t) for t in F.tiles(F.image(strip))] if strip.is_file() else []
        bad = sum(c > threshold for c in tiles)
        if bad:
            gone.append((r, strip, bad, len(tiles)))
        else:
            keep.append(r)
    if gone:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (out / "frames_corrupted").mkdir(exist_ok=True)
        with open(out / "episodes_corrupted.jsonl", "a") as f:
            for r, strip, bad, n in gone:
                f.write(json.dumps(dict(r, corrupted_tiles=bad, tiles=n, dropped_utc=now), default=str) + "\n")
                k = 0
                while (out / "frames_corrupted" / f"{strip.stem}.{k}.png").exists():
                    k += 1
                os.replace(strip, out / "frames_corrupted" / f"{strip.stem}.{k}.png")
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in keep))
        os.replace(tmp, path)
    return {"dropped": len(gone), "kept": len(keep)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drop")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--threshold", type=float, default=F.CORRUPT)
    args = ap.parse_args(argv)
    print(json.dumps(drop(args.out, args.threshold)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
