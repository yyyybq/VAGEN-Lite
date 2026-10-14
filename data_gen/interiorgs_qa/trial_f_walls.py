"""Trial F (CPU): which local box axis points away from the wall an object stands against?

For every box of the ten trial categories, each of the four horizontal local directions d (+e1, -e1, +e2,
-e2 of ``trial_f_axes.edge_frame``) is tested: is there a structure wall parallel to the box face on the -d
side, within ``BACK_GAP_M`` of it and overlapping it along the wall?  An object with its back to a wall has
exactly one such d, and d is then its front under the "back against the wall" rule.  If the corner order
carries the model's own frame, one local direction should win for every category.

    python -m data_gen.interiorgs_qa.trial_f_walls --scenes-file docs/interiorgs_qa/audit/train_scenes_20.txt \
        --out exps/interiorgs_qa_trials/trial_f_walls.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10.referring import display_label
from data_gen.interiorgs_v10.scene import load_scene

from .trial_f_axes import TRIAL_CATEGORIES, corners, edge_frame

BACK_GAP_M = 0.30          # box face to wall centre line, minus half the wall thickness
PARALLEL_COS = 0.17        # wall direction within ~10 deg of the face
MIN_OVERLAP_FRACTION = 0.5  # share of the face width that lies along the wall
LOCAL_DIRS = ("+e1", "-e1", "+e2", "-e2")


def category_of(label: str) -> Optional[str]:
    disp = display_label(label).lower()
    for c in TRIAL_CATEGORIES:
        if disp == c.lower() or disp.endswith(" " + c.lower()):
            return c
    return None


def local_dirs(f: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, float, float]]:
    """name -> (unit horizontal direction, half extent along it, half width across it)."""
    out = {}
    for name, e, other in (("e1", f["e1"], f["e2"]), ("e2", f["e2"], f["e1"])):
        h = np.array([e[0], e[1]], float)
        n = float(np.linalg.norm(h))
        if n < 1e-6:
            return {}
        u = h / n
        w = float(np.linalg.norm(other[:2]))
        out["+" + name] = (u, n / 2.0, w / 2.0)
        out["-" + name] = (-u, n / 2.0, w / 2.0)
    return out


def wall_behind(center_xy: np.ndarray, d: np.ndarray, half_d: float, half_w: float,
                walls: Sequence[Tuple[np.ndarray, np.ndarray, float]]) -> Optional[float]:
    """Gap from the box face on the -d side to the nearest parallel wall behind it, or None."""
    back = center_xy - d * half_d
    side = np.array([-d[1], d[0]])
    best = None
    for a, b, thick in walls:
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-6:
            continue
        s = seg / L
        if abs(float(s @ d)) > PARALLEL_COS:
            continue
        # distance of the wall line behind the face (positive = behind it, on the -d side)
        dist = -float((a - back) @ d)
        gap = dist - thick / 2.0
        if gap < -0.15 or gap > BACK_GAP_M:
            continue
        # overlap of the face [back - half_w side, back + half_w side] with the segment, along the wall
        t0 = float((back - half_w * side - a) @ s)
        t1 = float((back + half_w * side - a) @ s)
        lo, hi = max(0.0, min(t0, t1)), min(L, max(t0, t1))
        if hi - lo < MIN_OVERLAP_FRACTION * 2.0 * half_w:
            continue
        best = gap if best is None else min(best, gap)
    return best


def scene_walls(scene_dir: Path, sign: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    s = json.loads((scene_dir / "structure.json").read_text())
    out = []
    for w in s.get("walls") or []:
        loc = w.get("location") or []
        if len(loc) != 2:
            continue
        a = np.array([float(loc[0][0]), sign * float(loc[0][1])])
        b = np.array([float(loc[1][0]), sign * float(loc[1][1])])
        out.append((a, b, float(w.get("thickness") or 0.0)))
    return out


def analyse_scene(scene_dir: Path) -> List[Dict[str, Any]]:
    scene = load_scene(scene_dir, scene_dir.name)
    sign = float(scene.convention_v2.get("applied_sign", 1.0))
    walls = scene_walls(scene_dir, sign)
    rows = []
    for item in json.loads((scene_dir / "labels.json").read_text()):
        if len(item.get("bounding_box") or []) != 8:
            continue
        cat = category_of(item.get("label", ""))
        if cat is None:
            continue
        xyz = corners(item)
        f = edge_frame(xyz)
        if not f["ok"] or abs(float(f["e3"][2])) < 0.9 * float(np.linalg.norm(f["e3"])):
            continue
        dirs = local_dirs(f)
        if not dirs:
            continue
        c = xyz.mean(0)[:2]
        backed = {name: wall_behind(c, u, hd, hw, walls) for name, (u, hd, hw) in dirs.items()}
        rows.append({"scene": scene_dir.name, "ins_id": str(item.get("ins_id")), "label": item.get("label"),
                     "category": cat, "handedness": f["handedness"],
                     "backed": {k: (None if v is None else round(v, 3)) for k, v in backed.items()}})
    return rows


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cat in TRIAL_CATEGORIES:
        rs = [r for r in rows if r["category"] == cat]
        one = [r for r in rs if sum(v is not None for v in r["backed"].values()) == 1]
        win = Counter(next(k for k, v in r["backed"].items() if v is not None) for r in one)
        by_hand = defaultdict(Counter)
        for r in one:
            by_hand[str(r["handedness"])][next(k for k, v in r["backed"].items() if v is not None)] += 1
        out[cat] = {"boxes": len(rs), "exactly_one_wall_behind": len(one),
                    "front_direction_if_back_to_wall": dict(win),
                    "by_handedness": {k: dict(v) for k, v in by_hand.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_meta")
    ap.add_argument("--scenes-file", default=None)
    ap.add_argument("--sample", type=int, default=0, help="random sample of this many scenes (seed 20261014)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.meta_root)
    if args.scenes_file:
        names = [l.strip() for l in Path(args.scenes_file).read_text().splitlines() if l.strip()]
    else:
        names = sorted(p.name for p in root.iterdir() if (p / "labels.json").is_file())
    if args.sample:
        names = sorted(random.Random(20261014).sample(names, min(args.sample, len(names))))
    rows: List[Dict[str, Any]] = []
    for n in names:
        try:
            rows += analyse_scene(root / n)
        except Exception as exc:  # report and continue
            print("skip", n, type(exc).__name__, exc)
    summary = summarise(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"scenes": names, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
