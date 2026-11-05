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

2026-11-05 (the user's plan 甲 after the 11-04 rescan, docs/downstream/reports/FRAMES_RESCAN_20261104.md): three more
ways a render breaks, told per tile from its grey image (the mean of R, G and B; tile_rules()):
- near_black: grey mean under NEAR_BLACK (10), a black or almost black frame;
- flip: the tile is the previous tile upside down, often in a frozen render: its mean grey difference to the previous
  tile's vertical mirror is under FLIP_RATIO (0.6) of the plain difference, and the plain difference is over FLIP_MIN
  (20);
- stripes: horizontal stripes, the mean |vertical grey step| over STRIPES (2.5) times the mean |horizontal grey step|.
RULES says which of them count as broken and which only as hints to look at, by kind of strip:
- the N series' PNG strips: near_black is broken; flip and stripes are hints, since adjacent N turns are different
  views (in the rescan 5 of 23 flips and 17 of 51 stripes were real; most other stripes were close-ups of plank walls
  and shelves);
- the closed-loop JPEG thumbnails: near_black and flip are broken (21 of 21 flips were real); stripes are not used,
  since the other two rules caught their 4 real ones.
A blank rule (grey std under 2) is not used: its 3 real ones were near-black too, the other 7 were a cabinet door or
a panel in front of the camera.  chroma, corrupted and CORRUPT are as before (N2's demonstrations, N3's generation and
n_repair.py use them).

    python downstream/n/n_frames.py scan --eval EVAL_DIR [EVAL_DIR ...] --out REPORT.json
    python downstream/n/n_frames.py thumbs --eval EPISODES_DIR [EPISODES_DIR ...] --out REPORT.json

scan: every OUT/<model>/frames/<instance>.png strip (one square tile per turn) of n_episode.py evaluations: per
  instance and model the corrupted tiles of all tiles, and the instances with any.  Since 11-05 also per evaluation
  "broken" (instance -> model -> rule -> tiles, counted from 0; n_report.py --exclude leaves these instances out like
  the corrupted ones) and "hints" (the same for flip and stripes: look at them before leaving any out).
thumbs: the example strips EPISODES_DIR/frames/<task>_s<seed>.jpg of closed_loop.py (frames side by side, a band of
  step labels and the outcome bar below, THUMB_BAND px): per folder every episode with a broken tile, with the
  control steps of those tiles, its result row of EPISODES_DIR/episodes.jsonl (status and success; none when the
  episode has no row) and whether it is on the fixed exclusion list.  The episodes are only recorded and reported:
  the exclusion list (downstream/eval/excluded_episodes.json) stays as it is and nothing is played again.
  run_eval_ckpt.sh runs it after every evaluation (OUT/frames_scan.json).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

CORRUPT = 8.0
NEAR_BLACK = 10.0
FLIP_MIN, FLIP_RATIO = 20.0, 0.6
STRIPES = 2.5
RULES = {"png": {"broken": ["near_black"], "hints": ["flip", "stripes"]},
         "jpg": {"broken": ["near_black", "flip"], "hints": []}}
THUMB_BAND = 18  # closed_loop.FrameRecorder.save: a 14 px step label and a 4 px outcome bar under the frames
EXCLUDED = Path(__file__).resolve().parents[1] / "eval/excluded_episodes.json"
THUMB_NAME = re.compile(r"^(?P<task>.+)_s(?P<seed>\d+)$")


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


def thumb_tiles(rgb, band: int = THUMB_BAND) -> list:
    """The square frames of a closed-loop example strip (side = its height less the band); [] for another shape."""
    import numpy as np

    a = np.asarray(rgb)
    side = a.shape[0] - band
    if side <= 0 or a.shape[1] % side:
        return []
    return [a[:side, k * side:(k + 1) * side] for k in range(a.shape[1] // side)]


def tile_rules(tile_list) -> list:
    """Per tile, in order, the names of the 11-05 rules it trips (near_black, flip, stripes)."""
    import numpy as np

    out, prev = [], None
    for t in tile_list:
        g = np.asarray(t, dtype=np.float32)[..., :3].mean(axis=2)
        names = []
        if g.mean() < NEAR_BLACK:
            names.append("near_black")
        if prev is not None:
            plain = float(np.abs(g - prev).mean())
            if plain > FLIP_MIN and float(np.abs(g - prev[::-1]).mean()) < FLIP_RATIO * plain:
                names.append("flip")
        h = float(np.abs(np.diff(g, axis=1)).mean())
        if h > 0 and float(np.abs(np.diff(g, axis=0)).mean()) / h > STRIPES:
            names.append("stripes")
        out.append(names)
        prev = g
    return out


def defects(tile_list, kind: str) -> dict:
    """{"broken": {rule: [tile, ...]}, "hints": {rule: [tile, ...]}} of a strip's tiles under RULES[kind] (tiles
    counted from 0; a rule without a tile left out)."""
    out = {"broken": {}, "hints": {}}
    for i, names in enumerate(tile_rules(tile_list)):
        for n in names:
            for part in ("broken", "hints"):
                if n in RULES[kind][part]:
                    out[part].setdefault(n, []).append(i)
    return out


def scan(evals, threshold: float = CORRUPT) -> dict:
    out = {"threshold": threshold, "rules": RULES["png"], "evals": {}}
    for ev in evals:
        per = collections.defaultdict(dict)
        found = {"broken": collections.defaultdict(dict), "hints": collections.defaultdict(dict)}
        n_tiles = n_bad = 0
        for p in sorted(Path(ev).glob("*/frames/*.png")):
            model, inst = p.parent.parent.name, p.stem
            ts = tiles(image(p))
            c = [chroma(t) for t in ts]
            bad = sum(x > threshold for x in c)
            n_tiles, n_bad = n_tiles + len(c), n_bad + bad
            if bad:
                per[inst][model] = {"corrupted": bad, "tiles": len(c), "first": next(i for i, x in enumerate(c)
                                                                                      if x > threshold)}
            for part, rules in defects(ts, "png").items():
                if rules:
                    found[part][inst][model] = dict(rules, tiles=len(ts))
        out["evals"][str(ev)] = {"tiles": n_tiles, "corrupted_tiles": n_bad, "instances": dict(per),
                                 "episodes": sum(len(v) for v in per.values()),
                                 "broken": dict(found["broken"]), "hints": dict(found["hints"])}
    return out


def episode_rows(folder) -> dict:
    """(task, seed) -> its row of folder/episodes.jsonl: the last complete one, else the last one."""
    rows, path = {}, Path(folder) / "episodes.jsonl"
    for line in path.read_text().splitlines() if path.is_file() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        key = (r.get("task"), r.get("seed"))
        if r.get("status") == "complete" or rows.get(key, {}).get("status") != "complete":
            rows[key] = r
    return rows


def excluded_keys(path: Path = EXCLUDED) -> set:
    try:
        return {tuple(x) for x in json.loads(Path(path).read_text())["episodes"]}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def thumbs(folders, excluded_path: Path = EXCLUDED) -> dict:
    out = {"rules": RULES["jpg"]["broken"], "folders": {}}
    excl = excluded_keys(excluded_path)
    for folder in map(Path, folders):
        rows = episode_rows(folder)
        strips = sorted((folder / "frames").glob("*.jpg"))
        rec = {"strips": len(strips), "tiles": 0, "unreadable": [], "flagged": []}
        for p in strips:
            try:
                ts = thumb_tiles(image(p))
            except Exception:  # e.g. a strip cut short when its worker died
                ts = []
            if not ts:
                rec["unreadable"].append(p.name)
                continue
            rec["tiles"] += len(ts)
            broken = defects(ts, "jpg")["broken"]
            if not broken:
                continue
            m = THUMB_NAME.match(p.stem)
            task, seed = (m["task"], int(m["seed"])) if m else (p.stem, None)
            row = rows.get((task, seed), {})
            steps = row.get("frame_steps") or []
            at = sorted({i for v in broken.values() for i in v})
            rec["flagged"].append({"episode": p.stem, "task": task, "seed": seed, "tiles": len(ts), "broken": broken,
                                   "steps": [steps[i] for i in at if i < len(steps)], "status": row.get("status"),
                                   "success": row.get("success"), "excluded": (task, seed) in excl})
        out["folders"][str(folder)] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--eval", nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("thumbs")
    t.add_argument("--eval", nargs="+", required=True, help="closed_loop.py output folders (episodes.jsonl, frames/)")
    t.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "thumbs":
        rep = thumbs(args.eval)
        args.out.write_text(json.dumps(rep, indent=1) + "\n")
        for folder, r in rep["folders"].items():
            print(f"{folder}: {len(r['flagged'])} of {r['strips']} episodes with a broken frame "
                  f"({', '.join(rep['rules'])}), {len(r['unreadable'])} strips unreadable")
            for f in r["flagged"]:
                result = "no result" if f["status"] is None else f"{f['status']}, success {f['success']}"
                print(f"  {f['episode']}: {', '.join(f'{k} {len(v)}' for k, v in f['broken'].items())} of "
                      f"{f['tiles']} tiles at steps {f['steps']}; {result}" + ("; excluded" if f["excluded"] else ""))
        return 0
    rep = scan(args.eval)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    for ev, r in rep["evals"].items():
        counts = {part: dict(collections.Counter(k for inst in r[part].values() for m in inst.values() for k in m
                                                 if k != "tiles")) for part in ("broken", "hints")}
        print(f"{ev}: {r['corrupted_tiles']} of {r['tiles']} tiles corrupted, {r['episodes']} episodes in "
              f"{len(r['instances'])} instances; broken {counts['broken']} in {len(r['broken'])} instances, "
              f"hints {counts['hints']} in {len(r['hints'])} instances (episodes per rule)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
