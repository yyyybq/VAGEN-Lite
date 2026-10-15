"""Synthetic InteriorGS-format scenes for unit tests (no real assets needed).

The files mimic the real InteriorGS layout that existing repo parsers read:
  * labels.json   : list of {"ins_id", "label", "bounding_box": [8 x {"x","y","z"}]}
  * structure.json: {"rooms": [{"profile": [[x, y], ...], "name": str}],
                     "holes": [{"type": "DOOR", "profile": [[x, y], ...]}]}
Room/hole Y values are written with a per-scene ``structure_y_sign`` so that the
loader has to calibrate the convention, as with real InteriorGS exports.
A ``_truth.json`` sidecar (tests only) records the true sign and other facts.

Usage:
    from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes
    paths = write_synthetic_scenes(tmp_dir)   # {"two_rooms": Path, ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _box(ins_id: str, label: str, xmin: float, ymin: float, zmin: float,
         xmax: float, ymax: float, zmax: float) -> dict:
    corners = []
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            for z in (zmin, zmax):
                corners.append({"x": float(x), "y": float(y), "z": float(z)})
    return {"ins_id": str(ins_id), "label": label, "bounding_box": corners}


def _walls_for_rect(start_id: int, x0: float, y0: float, x1: float, y1: float,
                    t: float = 0.1, h: float = 2.8) -> List[dict]:
    """Thin 'wall' boxes along a rectangle edge set (real scenes contain these)."""
    return [
        _box(str(start_id), "wall", x0 - t, y0 - t, 0.0, x1 + t, y0, h),
        _box(str(start_id + 1), "wall", x0 - t, y1, 0.0, x1 + t, y1 + t, h),
        _box(str(start_id + 2), "wall", x0 - t, y0, 0.0, x0, y1, h),
        _box(str(start_id + 3), "wall", x1, y0, 0.0, x1 + t, y1, h),
    ]


def _write(scene_dir: Path, labels: List[dict], rooms: List[Tuple[str, Sequence[Sequence[float]]]],
           holes: List[Sequence[Sequence[float]]], y_sign: float, truth: dict,
           walls: Optional[List[Tuple[Sequence[float], Sequence[float], float]]] = None) -> Path:
    """``walls``: optional structure.json ``walls`` list as ((x0, y0), (x1, y1), thickness) centre lines, 2.8 m
    high (real scenes have one; without it the loaders fall back to the room outlines)."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    structure = {
        "rooms": [
            {"name": name, "profile": [[float(x), float(y_sign * y)] for x, y in poly]}
            for name, poly in rooms
        ],
        "holes": [
            {"type": "DOOR", "profile": [[float(x), float(y_sign * y)] for x, y in poly]}
            for poly in holes
        ],
    }
    if walls is not None:
        structure["walls"] = [{"thickness": float(t), "height": 2.8,
                               "location": [[float(a[0]), float(y_sign * a[1])], [float(b[0]), float(y_sign * b[1])]]}
                              for a, b, t in walls]
    (scene_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    (scene_dir / "structure.json").write_text(json.dumps(structure, indent=1))
    truth = dict(truth)
    truth["structure_y_sign"] = y_sign
    (scene_dir / "_truth.json").write_text(json.dumps(truth, indent=1))
    return scene_dir


def two_rooms(root: Path, walls: bool = False) -> Path:
    """Living room [0,6]x[0,5] + bedroom [6,10]x[0,5], door at x=6, y in [2,3].

    Duplicates: 2 dining chairs, 2 bedside tables.  Structure stored with the
    historical -y convention.  ``walls``: also a structure.json ``walls`` list (0.2 m thick: the outer
    rectangle and the partition at x = 6 with the door), as real scenes have.
    """
    labels = [
        _box("1", "multi person sofa", 0.3, 1.5, 0.0, 1.2, 3.7, 0.9),
        _box("2", "tv cabinet", 5.3, 1.6, 0.0, 5.8, 3.4, 0.55),
        _box("3", "tv", 5.55, 2.0, 0.6, 5.65, 3.0, 1.3),
        _box("4", "tea table", 2.4, 2.1, 0.0, 3.4, 2.9, 0.45),
        _box("5", "dining table", 2.5, 4.0, 0.0, 3.7, 4.8, 0.75),
        _box("6", "dining chair", 2.6, 3.45, 0.0, 3.1, 3.95, 0.95),
        _box("7", "dining chair", 3.2, 3.45, 0.0, 3.7, 3.95, 0.95),
        _box("8", "floor lamp", 0.2, 0.2, 0.0, 0.5, 0.5, 1.7),
        _box("9", "rug", 1.8, 1.5, 0.0, 4.2, 3.6, 0.02),
        _box("10", "double bed", 7.8, 1.2, 0.0, 9.8, 3.4, 0.6),
        _box("11", "bedside table", 7.3, 0.6, 0.0, 7.75, 1.05, 0.55),
        _box("12", "bedside table", 7.3, 3.55, 0.0, 7.75, 4.0, 0.55),
        _box("13", "wardrobe", 6.2, 4.3, 0.0, 8.2, 4.9, 2.2),
        _box("14", "desk", 9.2, 4.1, 0.0, 9.9, 4.9, 0.75),
        _box("15", "ceiling", 0.0, 0.0, 2.75, 10.0, 5.0, 2.8),
        _box("16", "floor", 0.0, 0.0, -0.02, 10.0, 5.0, 0.0),
    ]
    labels += _walls_for_rect(100, 0.0, 0.0, 6.0, 5.0)
    labels += _walls_for_rect(200, 6.0, 0.0, 10.0, 5.0)
    rooms = [
        ("living room", [(0, 0), (6, 0), (6, 5), (0, 5)]),
        ("bedroom", [(6, 0), (10, 0), (10, 5), (6, 5)]),
    ]
    holes = [[(5.95, 2.0), (6.05, 2.0), (6.05, 3.0), (5.95, 3.0)]]
    truth = {
        "duplicate_labels": {"dining chair": ["6", "7"], "bedside table": ["11", "12"]},
        "rooms": ["living room", "bedroom"],
    }
    if walls:
        wall_list = [((0, 0), (10, 0), 0.2), ((10, 0), (10, 5), 0.2), ((10, 5), (0, 5), 0.2), ((0, 5), (0, 0), 0.2),
                     ((6, 0), (6, 5), 0.2)]
        return _write(root / "synth_two_rooms_walls", labels, rooms, holes, y_sign=-1.0, truth=truth, walls=wall_list)
    return _write(root / "synth_two_rooms", labels, rooms, holes, y_sign=-1.0, truth=truth)


def symmetric_ambiguous(root: Path) -> Path:
    """L-shaped room straddling y=0: label-centre counting cannot pick the Y sign.

    The room is [0,6]x[-2.5,2.5] minus the notch x in [3,6], y in [1,2.5].
    Every object centre lies in |y| < 1 or x < 3, so both mirror images contain
    all centres.  Wall boxes along the true notch edge break the tie.
    Structure stored with the raw +y convention.
    """
    poly = [(0, -2.5), (6, -2.5), (6, 1.0), (3, 1.0), (3, 2.5), (0, 2.5)]
    labels = [
        _box("1", "sofa", 0.3, -0.8, 0.0, 1.1, 0.8, 0.85),
        _box("2", "tv cabinet", 5.3, -0.7, 0.0, 5.8, 0.7, 0.5),
        _box("3", "tea table", 2.3, -0.4, 0.0, 3.1, 0.4, 0.45),
        _box("4", "armchair", 1.8, -2.2, 0.0, 2.6, -1.4, 0.9),
        _box("5", "armchair", 1.8, 1.4, 0.0, 2.6, 2.2, 0.9),
        _box("6", "bookshelf", 4.2, -2.4, 0.0, 5.4, -2.0, 1.9),
        # wall boxes that only exist on the true (unmirrored) notch edges
        _box("50", "wall", 3.0, 1.0, 0.0, 6.0, 1.1, 2.8),
        _box("51", "wall", 2.9, 1.0, 0.0, 3.0, 2.5, 2.8),
        _box("52", "wall", 0.0, 2.5, 0.0, 3.0, 2.6, 2.8),
        _box("53", "wall", 0.0, -2.6, 0.0, 6.0, -2.5, 2.8),
    ]
    truth = {"ambiguous_by_centre_count": True}
    return _write(root / "synth_symmetric_ambiguous", labels, [("living room", poly)], [], y_sign=1.0,
                  truth=truth)


def dense_duplicates(root: Path) -> Path:
    """Kitchen-dining room with four identical chairs and two cabinets."""
    labels = [
        _box("1", "dining table", 2.0, 2.0, 0.0, 3.6, 2.9, 0.75),
        _box("2", "dining chair", 2.1, 1.4, 0.0, 2.6, 1.9, 0.95),
        _box("3", "dining chair", 3.0, 1.4, 0.0, 3.5, 1.9, 0.95),
        _box("4", "dining chair", 2.1, 3.0, 0.0, 2.6, 3.5, 0.95),
        _box("5", "dining chair", 3.0, 3.0, 0.0, 3.5, 3.5, 0.95),
        _box("6", "refrigerator", 0.1, 4.2, 0.0, 0.8, 4.9, 1.9),
        _box("7", "cabinet", 1.5, 4.4, 0.0, 2.7, 4.9, 0.9),
        _box("8", "cabinet", 4.6, 0.1, 0.0, 5.8, 0.6, 0.9),
        _box("9", "cooking bench", 3.0, 4.4, 0.0, 4.8, 4.9, 0.9),
        _box("10", "plant", 5.4, 4.4, 0.0, 5.9, 4.9, 1.2),
    ]
    labels += _walls_for_rect(100, 0.0, 0.0, 6.0, 5.0)
    truth = {"duplicate_labels": {"dining chair": ["2", "3", "4", "5"], "cabinet": ["7", "8"]}}
    return _write(root / "synth_dense_duplicates", labels, [("kitchen", [(0, 0), (6, 0), (6, 5), (0, 5)])], [],
                  y_sign=-1.0, truth=truth)


def write_synthetic_scenes(root: Path, with_walls: bool = False) -> Dict[str, Path]:
    """``with_walls``: also "two_rooms_walls", the two-room scene with a structure.json walls list (round 10)."""
    root = Path(root)
    out = {
        "two_rooms": two_rooms(root),
        "symmetric_ambiguous": symmetric_ambiguous(root),
        "dense_duplicates": dense_duplicates(root),
    }
    if with_walls:
        out["two_rooms_walls"] = two_rooms(root, walls=True)
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys
    out = write_synthetic_scenes(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_scenes"))
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=1))
