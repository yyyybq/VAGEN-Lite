#!/usr/bin/env python3
"""Bad frames the frame scan misses (the user's decision 5 of 2026-11-06, option 甲; found in the check rounds of the
tier g-d report): a frame that differs from what every other episode of the same instance and the same camera setting
rendered at the same pose.  The renders are deterministic (healthy frames at one pose agree to within a fraction of a
grey level), so a frame far from the consensus at its pose is a bad render: colour noise or near-black from the start,
or a frame that stopped updating.  This also catches episodes whose moves were all blocked (pose unchanged), which the
earlier "pose moved but the frame did not" rule could not.  Read only; no simulator, no model.

    python downstream/n/n_stale.py scan --root RUNS_N --out OUT [--robot-camera n3_20261103/e/] [--folders GLOB ...]

Episodes: every DIR/episodes.jsonl under --root that has DIR/frames/<instance>.png (one TILE px tile per turn, the
frame the model saw at that turn, at the turn's pose_before).  Camera setting: a folder whose path relative to --root
starts with a --robot-camera prefix renders from the robot's own camera (N3 tier e); every other folder renders
camera A, so they are compared together.  For each (camera setting, instance, pose rounded to 1e-3 m / rad) with at
least MIN_REF other tiles: the consensus is the most common coarse signature (16 x 16 block means, binned by 8 grey
levels); a tile is bad when its mean absolute difference (grey) to a consensus tile exceeds BAD_DIFF and at least
MIN_REF other tiles share the consensus.  An episode is flagged when any of its tiles is bad.  Error rows are skipped.
Writes OUT/stale.jsonl (one line per flagged episode: folder, instance, success, the bad turns, whether the start
frame is bad, the largest difference), OUT/summary.json (per folder: episodes, checked tiles, tiles without a
reference, flagged) and OUT/instances.txt (the flagged instances).
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

TILE, BAD_DIFF, MIN_REF, POSE_GRID = 128, 20.0, 2, 1e-3


def read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def tiles_of(png: Path, size: int = TILE) -> list:
    from PIL import Image

    img = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
    h = img.shape[0] if img.shape[0] else size
    return [img[:, k * h:(k + 1) * h] for k in range(img.shape[1] // h)]


def signature(tile: np.ndarray) -> bytes:
    h, w = tile.shape
    b = tile[: h - h % 16, : w - w % 16].reshape(16, (h - h % 16) // 16, 16, (w - w % 16) // 16).mean(axis=(1, 3))
    return (b // 8).astype(np.uint8).tobytes()


def pose_key(pose) -> tuple:
    return tuple(int(round(float(v) / POSE_GRID)) for v in pose[:3])


def camera_of(rel: str, robot_prefixes) -> str:
    return "robot" if any(rel.startswith(p) for p in robot_prefixes) else "a"


def episodes(root: Path, robot_prefixes, folders=None):
    """{(camera, instance): [(folder, row, tiles)]}, loaded lazily per instance by the caller (see scan)."""
    index = collections.defaultdict(list)
    for ep in sorted(root.rglob("episodes.jsonl")):
        folder = ep.parent
        rel = str(folder.relative_to(root)).replace("\\", "/") + "/"
        if not (folder / "frames").is_dir():
            continue
        if folders and not any(fnmatch.fnmatch(rel, g) for g in folders):
            continue
        for r in read_jsonl(ep):
            if r.get("error"):
                continue
            iid = r.get("instance_id") or r.get("scene_id")
            png = folder / "frames" / f"{iid}.png"
            if png.exists():
                index[(camera_of(rel, robot_prefixes), iid)].append((rel.rstrip("/"), r, png))
    return index


def judge_instance(items) -> list:
    """items: [(folder, row, tiles)] of one camera setting and instance.  Returns per item a dict of its bad turns."""
    at = collections.defaultdict(list)  # pose key -> [(item index, turn, tile, signature)]
    for n, (_, row, tiles) in enumerate(items):
        poses = [t.get("pose_before") for t in row.get("turn_log") or []]
        for k, tile in enumerate(tiles[: len(poses)]):
            if poses[k] is not None:
                at[pose_key(poses[k])].append((n, k, tile, signature(tile)))
    out = [{"bad": [], "checked": 0, "no_ref": 0, "max_diff": 0.0} for _ in items]
    for entries in at.values():
        for n, k, tile, sig in entries:
            others = collections.Counter(e[3] for e in entries if e[0] != n)
            if not others:
                out[n]["no_ref"] += 1
                continue
            cons, support = others.most_common(1)[0]
            if support < MIN_REF:
                out[n]["no_ref"] += 1
                continue
            ref = next(e[2] for e in entries if e[3] == cons and e[0] != n)
            d = float(np.abs(tile - ref).mean())
            out[n]["checked"] += 1
            out[n]["max_diff"] = max(out[n]["max_diff"], d)
            if d > BAD_DIFF:
                out[n]["bad"].append(k)
    return out


def scan(root: Path, out: Path, robot_prefixes, folders=None) -> dict:
    index = episodes(root, robot_prefixes, folders)
    flagged, summary = [], collections.defaultdict(lambda: {"episodes": 0, "checked_tiles": 0, "no_ref_tiles": 0,
                                                            "flagged": 0})
    for (cam, iid), items in sorted(index.items()):
        loaded = [(f, r, tiles_of(p)) for f, r, p in items]
        res = judge_instance(loaded)
        for (folder, row, _), v in zip(loaded, res):
            s = summary[folder]
            s["episodes"] += 1
            s["checked_tiles"] += v["checked"]
            s["no_ref_tiles"] += v["no_ref"]
            if v["bad"]:
                s["flagged"] += 1
                flagged.append({"folder": folder, "instance_id": iid, "camera": cam, "success": row.get("success"),
                                "bad_turns": [k + 1 for k in v["bad"]], "start_bad": 0 in v["bad"],
                                "turns": row.get("turns"), "max_diff": round(v["max_diff"], 1)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "stale.jsonl").write_text("".join(json.dumps(x) + "\n" for x in flagged))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "instances.txt").write_text("".join(i + "\n" for i in sorted({x["instance_id"] for x in flagged})))
    return {"folders": len(summary), "episodes": sum(s["episodes"] for s in summary.values()), "flagged": len(flagged),
            "instances": len({x["instance_id"] for x in flagged})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--robot-camera", nargs="*", default=["n3_20261103/e/"],
                   help="folder prefixes (relative to --root) rendered from the robot's own camera")
    s.add_argument("--folders", nargs="*", default=None, help="only folders matching these globs (relative, with /)")
    args = ap.parse_args(argv)
    print(json.dumps(scan(args.root, args.out, args.robot_camera, args.folders)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
