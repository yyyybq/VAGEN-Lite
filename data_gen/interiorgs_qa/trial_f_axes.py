"""Trial F, step 1 (CPU only): do the eight corners of a labels.json box carry the object's own frame?

If the corner order were just "min corner first" the edge p1 - p0 of an axis-aligned box would always point
the same world direction.  If it follows the object's model transform, it points along +x, -x, +y or -y
depending on how the object was placed, and boxes of one category show several world directions.

    python -m data_gen.interiorgs_qa.trial_f_axes --meta-root /mnt/umm/users/yinbaiqiao/InteriorGS_meta \
        --out exps/interiorgs_qa_trials/trial_f_axes.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from data_gen.interiorgs_v10.referring import display_label

# The ten categories of the trial (user prompt, section 7 F.2), as display-label phrases.
TRIAL_CATEGORIES = ("bed", "sofa", "chair", "cabinet", "desk", "TV", "toilet", "refrigerator", "wardrobe",
                    "washbasin")


def corners(item: Dict[str, Any]) -> np.ndarray:
    return np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in item["bounding_box"]], float)


def edge_frame(xyz: np.ndarray) -> Dict[str, Any]:
    """Edges e1 = p1 - p0, e2 = p3 - p0, e3 = p4 - p0, whether the 8 corners follow that box pattern, and
    the handedness of (e1, e2, e3)."""
    e1, e2, e3 = xyz[1] - xyz[0], xyz[3] - xyz[0], xyz[4] - xyz[0]
    pred = np.array([xyz[0], xyz[0] + e1, xyz[0] + e1 + e2, xyz[0] + e2,
                     xyz[0] + e3, xyz[0] + e1 + e3, xyz[0] + e1 + e2 + e3, xyz[0] + e2 + e3])
    lens = np.array([np.linalg.norm(e) for e in (e1, e2, e3)])
    ok = bool(np.max(np.abs(pred - xyz)) <= 1e-4 and lens.min() > 1e-4)
    if ok:
        cosines = [abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b)) for a, b in ((e1, e2), (e1, e3), (e2, e3))]
        ok = max(cosines) <= 1e-3
    hand = float(np.sign(np.cross(e1, e2) @ e3)) if ok else 0.0
    return {"ok": ok, "e1": e1, "e2": e2, "e3": e3, "handedness": hand}


def world_direction(v: np.ndarray) -> str:
    """Nearest of +x, -x, +y, -y, +z, -z."""
    k = int(np.argmax(np.abs(v)))
    return ("+" if v[k] > 0 else "-") + "xyz"[k]


def yaw_deg(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0]))) % 360.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_meta")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.meta_root)
    counts = Counter()
    e1_dir_axis_aligned = Counter()
    e2_dir_axis_aligned = Counter()
    e3_dir = Counter()
    handed = Counter()
    by_cat: Dict[str, Counter] = defaultdict(Counter)       # display label -> e1 world direction (axis-aligned)
    by_cat_yaw: Dict[str, List[float]] = defaultdict(list)  # display label -> yaw of the horizontal edge e1
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    scenes = sorted(p for p in root.iterdir() if (p / "labels.json").is_file())
    for sd in scenes:
        for item in json.loads((sd / "labels.json").read_text()):
            bb = item.get("bounding_box") or []
            if len(bb) != 8:
                counts["no_8_corner_box"] += 1
                continue
            xyz = corners(item)
            f = edge_frame(xyz)
            if not f["ok"]:
                counts["not_a_box_pattern"] += 1
                continue
            counts["boxes"] += 1
            lo, hi = xyz.min(0), xyz.max(0)
            aligned = bool(np.all(np.minimum(np.abs(xyz - lo), np.abs(xyz - hi)) <= 1e-6))
            counts["axis_aligned" if aligned else "rotated"] += 1
            handed[f["handedness"]] += 1
            e3_dir[world_direction(f["e3"])] += 1
            disp = display_label(item.get("label", ""))
            matched = [c for c in TRIAL_CATEGORIES if disp.lower() == c.lower() or disp.lower().endswith(" " + c.lower())]
            # p0 is the min corner exactly when e1, e2, e3 all point to positive world axes
            p0_is_min = bool(np.allclose(xyz[0], lo, atol=1e-6))
            counts["p0_is_min_corner" if p0_is_min else "p0_not_min_corner"] += 1
            if aligned:
                e1_dir_axis_aligned[world_direction(f["e1"])] += 1
                e2_dir_axis_aligned[world_direction(f["e2"])] += 1
            for c in matched:
                by_cat[c][world_direction(f["e1"])] += 1
                by_cat_yaw[c].append(yaw_deg(f["e1"]))
                if len(examples[c]) < 3:
                    examples[c].append({"scene": sd.name, "ins_id": item.get("ins_id"), "label": item.get("label"),
                                        "e1": f["e1"].round(3).tolist(), "e2": f["e2"].round(3).tolist(),
                                        "e3": f["e3"].round(3).tolist()})
    yaw_hist = {c: dict(Counter(int(round(y / 45.0)) * 45 % 360 for y in ys)) for c, ys in by_cat_yaw.items()}
    out = {
        "scenes": len(scenes), "counts": dict(counts),
        "e1_direction_axis_aligned": dict(e1_dir_axis_aligned),
        "e2_direction_axis_aligned": dict(e2_dir_axis_aligned),
        "e3_direction": dict(e3_dir), "handedness": {str(k): v for k, v in handed.items()},
        "trial_categories_e1_direction": {c: dict(v) for c, v in by_cat.items()},
        "trial_categories_e1_yaw_45deg_bins": yaw_hist,
        "examples": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("counts", "e1_direction_axis_aligned", "e2_direction_axis_aligned",
                                           "e3_direction", "handedness", "trial_categories_e1_direction")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
