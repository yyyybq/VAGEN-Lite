"""Label errors found by size (2026-10-14, round 6): instances whose box is far from the usual size of their
category become no-task objects (never a target or a reference object; they still collide and occlude).

The round-5 check page found boxes whose label names something else: a 2.15 m tall "dishwasher" (a curtain),
a 1.3 m "pinecone" (an armchair), a "stool" 2 m above the floor (a wall shelf), a 1.9 m tall "wall-mounted
air conditioner".  Rule, per display label (``referring.display_label``) with at least ``MIN_COUNT``
instances over all scenes, on three sizes (height, longer and shorter horizontal side):

* a size more than ``RATIO`` times the category median and above its ``HI_PCT`` percentile, or less than
  1/``RATIO`` of the median and below its ``LO_PCT`` percentile; or
* a floor category (90% of its instances stand within ``FLOOR_P90_M`` of the floor) whose box starts more
  than ``FLOOR_MAX_BOTTOM_M`` above the floor.

Only labels that may be a target or a reference object (``objects.semantic_exclusion_reason``) are listed.

    python -m data_gen.interiorgs_v10.label_outliers --meta-root /mnt/umm/users/yinbaiqiao/InteriorGS_meta \\
        --base data_gen/interiorgs_v10/configs/no_task_objects_20261013.json \\
        --out data_gen/interiorgs_v10/configs/no_task_objects_20261014.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RATIO = 3.0
LO_PCT, HI_PCT = 2.0, 98.0
MIN_COUNT = 20
FLOOR_P90_M = 0.3
FLOOR_MAX_BOTTOM_M = 1.2
SIZES = ("height", "longer side", "shorter side")

Row = Tuple[str, str, str, float, float, float, float]  # scene, id, display label, h, long, short, bottom


def size_row(scene_id: str, obj: Any) -> Row:
    from .referring import display_label

    d = np.asarray(obj.aabb_max, float) - np.asarray(obj.aabb_min, float)
    return (str(scene_id), str(obj.id), display_label(obj.label), float(d[2]), float(max(d[0], d[1])),
            float(min(d[0], d[1])), float(obj.aabb_min[2]))


def flag_outliers(rows: Iterable[Row]) -> Dict[Tuple[str, str], List[str]]:
    """(scene, id) -> reasons, for the instances the size rule calls label errors."""
    by_label: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_label[r[2]].append(r)
    out: Dict[Tuple[str, str], List[str]] = {}
    for label, group in by_label.items():
        if len(group) < MIN_COUNT:
            continue
        sizes = np.array([[r[3], r[4], r[5]] for r in group], float)
        bottoms = np.array([r[6] for r in group], float)
        med = np.median(sizes, axis=0)
        lo = np.percentile(sizes, LO_PCT, axis=0)
        hi = np.percentile(sizes, HI_PCT, axis=0)
        floor_category = float(np.percentile(bottoms, 90)) <= FLOOR_P90_M
        for r, s, z in zip(group, sizes, bottoms):
            why = []
            for k, name in enumerate(SIZES):
                if med[k] <= 0.01:
                    continue
                x = max(float(s[k]), 1e-3)
                if x > RATIO * med[k] and x > hi[k]:
                    why.append(f"{name} {x:.2f} m, {x / med[k]:.1f}x the usual {med[k]:.2f} m")
                elif x < med[k] / RATIO and x < lo[k]:
                    why.append(f"{name} {x:.2f} m, 1/{med[k] / x:.1f} of the usual {med[k]:.2f} m")
            if floor_category and z > FLOOR_MAX_BOTTOM_M:
                why.append(f"starts {z:.2f} m above the floor; {label} stands on the floor")
            if why:
                out[(r[0], r[1])] = why
    return out


def eligible(label: str, preset: str = "v10_r6") -> bool:
    """Labels that may be a target or a reference object under ``preset``."""
    from .objects import semantic_exclusion_reason

    return semantic_exclusion_reason(label, preset) is None


def build(meta_root: Path, base: Optional[Dict[str, Any]], preset: str = "v10_r6",
          scene_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    from .scene import parse_labels

    rows: List[Row] = []
    labels: Dict[Tuple[str, str], str] = {}
    dirs = sorted(p for p in Path(meta_root).iterdir() if (p / "labels.json").is_file())
    if scene_ids is not None:
        keep = set(scene_ids)
        dirs = [p for p in dirs if p.name in keep]
    for d in dirs:
        for o in parse_labels(json.loads((d / "labels.json").read_text(encoding="utf-8"))):
            rows.append(size_row(d.name, o))
            labels[(d.name, str(o.id))] = o.label
    flags = {k: v for k, v in flag_outliers(rows).items() if eligible(labels[k], preset)}
    scenes: Dict[str, List[Dict[str, Any]]] = {s: list(v) for s, v in ((base or {}).get("scenes") or {}).items()}
    already = {(s, str(o["ins_id"])) for s, v in scenes.items() for o in v}
    added = 0
    for (sid, oid), why in sorted(flags.items()):
        if (sid, oid) in already:
            continue
        scenes.setdefault(sid, []).append({"ins_id": oid, "label": labels[(sid, oid)], "kind": "size_outlier",
                                           "reason": "; ".join(why)})
        added += 1
    kinds = Counter(o.get("kind") for v in scenes.values() for o in v)
    return {
        "version": "interiorgs_no_task_objects_v1",
        "created": "2026-10-14",
        "method": ("Round 6 (data_gen/interiorgs_v10/STATUS.md, decision of 2026-10-14): the 2026-10-13 list "
                   "plus kind=size_outlier, boxes whose size is far from the usual size of their category "
                   "(data_gen/interiorgs_v10/label_outliers.py: more than 3x the median and beyond the 98th "
                   "percentile, or below 1/3 and the 2nd percentile, of height or a horizontal side, over "
                   f"categories with at least {MIN_COUNT} instances; or a floor category starting more than "
                   f"{FLOOR_MAX_BOTTOM_M} m above the floor), restricted to labels that may be a target or a "
                   "reference object.  The boxes keep colliding and occluding."),
        "source": "configs/no_task_objects_20261013.json + label_outliers.py",
        "counts": {"objects": sum(len(v) for v in scenes.values()), "scenes": len(scenes),
                   **{str(k): n for k, n in sorted(kinds.items())}, "size_outlier_added": added,
                   "instances_checked": len(rows)},
        "scenes": {s: scenes[s] for s in sorted(scenes)},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--meta-root", required=True)
    ap.add_argument("--base", default=None, help="no-task list to extend")
    ap.add_argument("--preset", default="v10_r6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    base = json.loads(Path(args.base).read_text(encoding="utf-8")) if args.base else None
    data = build(Path(args.meta_root), base, args.preset)
    Path(args.out).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data["counts"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
