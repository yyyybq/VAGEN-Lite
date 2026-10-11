"""structure.json <-> labels.json Y-sign calibration, version 2.

Problem
-------
InteriorGS ``structure.json`` room/hole/wall coordinates are stored either in
label/world coordinates (sign +1) or with Y negated (sign -1), scene by scene.
The runtime ``CollisionDetector`` (v1 rule, ``interiorgs_structure_label_alignment_v1``)
counts label-box centres inside the room polygons under both signs and freezes
the sign only when >= 20 label centres exist and the winner leads by >= 10 %.
On the 1,000-scene InteriorGS set this left 117 scenes "ambiguous".  Typical
causes: small scenes (< 20 labels) and scenes whose room union is (nearly)
mirror-symmetric about y = 0, so that every centre lies inside both
interpretations.

Evidence types (each computed under sign +1 and sign -1)
-------------------------------------------------------
Every type yields per-unit preferences d in [-1, 1] (positive favours +1);
``plus`` / ``minus`` are the summed positive / negative parts, ``net`` their
difference, ``purity = |net| / (plus + minus)``.

  label_centres       (a) furniture centres inside some room polygon (unit = object).
                          The exact v1 statistic over ALL labels is reported
                          separately under ``v1`` (it also counts wall /
                          floor / ceiling boxes, whose centres sit on or
                          outside polygon edges and bias the count).
  corner_containment  (b) fraction of the 4 (slightly shrunk) footprint corners of
                          each furniture box inside some room (unit = object).
  edge_crossing       (e) a room edge cuts through the (shrunk) furniture footprint;
                          a crossing under one sign is a penalty for that sign.
  wall_alignment      (c) thin wall / door / window / partition boxes: fraction of
                          their centre line lying within thickness/2 + 0.15 m of a
                          PARALLEL room-polygon edge (or structure wall line).
  openings            (d) door / window boxes within 0.3 m of a DOOR / WINDOW hole.
  room_semantics      (f) unambiguous room-typed objects (bed, toilet, stove, ...)
                          inside a room whose name has the same type (weight 0.5).

Types (a), (b) and (e) look at the same furniture boxes, so they form one
*family* ("containment"); (c), (d) and (f) are separate families.  Within a
family the support for a sign is the MAX over its types (no double counting);
across families supports add up.

Decision rule (conservative; constants below)
---------------------------------------------
  R0  If the v1 rule freezes the scene, v2 returns exactly v1's sign with
      status "frozen" (rule "v1_frozen").  v2 therefore never disagrees with a
      v1-frozen scene.  v2's own verdict is still computed and stored under
      ``v2_own``; ``conflict_with_v1`` is set when v2's own rule would have frozen
      the opposite sign (reported by the CLI for manual review).
  R1  P = sum over families of max-type plus support, M likewise for minus.
      winner = sign with the larger total; F = max(P, M), A = min(P, M).
  R2  F - A >= MIN_NET_SUPPORT (3 units), else "insufficient_evidence".
  R3  A <= MAX_COUNTER_RATIO * F (20 %), else "counter_evidence".
  R4  No single type may lean against the winner with net >= CONFLICT_NET
      (1.5 units) at purity >= CONFLICT_PURITY (0.6), else "conflicting_evidence".
  R5  Fit: with >= 3 furniture boxes the winner must hold >= 60 % of their
      footprint corners inside rooms OR (>= 3 thin wall boxes) align >= 60 % of
      the wall centre lines, else "no_sign_fits" (both signs bad, e.g. the
      structure is not registered to the labels at all).  The wall family is
      ignored when its leaning sign aligns < 30 % of the thin wall boxes.
  Only when R2-R5 hold is the scene frozen by v2 (rule "v2_evidence").

Integration fact (IMPORTANT)
----------------------------
At train time the RL environment does NOT run this module: its
``CollisionDetector`` recomputes the sign with the v1 rule.  For a scene that
v1 calls ambiguous but v2 freezes, the env would raise "R1 collision
convention is not frozen" (canonical rows) or silently use v1's fallback
sign.  The generator must therefore emit the v2 sign as an explicit override
and the env must be constructed with
``CollisionDetector(structure_y_sign_overrides={scene_id: sign})``.
``env_overrides(records)`` returns exactly that dict (by default only for
scenes that need it, i.e. v2-frozen and v1-not-frozen).  NOTE: as of
2026-10-10 ``vagen/envs/active_spatial/env.py`` builds its detector through
``create_collision_detector({...})`` WITHOUT forwarding
``structure_y_sign_overrides`` -- the env config needs that plumbing before
v2-rescued scenes can be trained on.  ``scene.load_scene`` passes the same
override to its embedded detector, so generator and env agree.

CLI
---
    python -m data_gen.interiorgs_v10.coordinates --scenes-root DIR [--out report.json]
    python -m data_gen.interiorgs_v10.coordinates --benchmark 400 [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .fastgeom import point_segment_distance_clip, points_in_any_polygon, points_room_index

VERSION = "interiorgs_structure_label_alignment_v2"
V1_VERSION = "interiorgs_structure_label_alignment_v1"
V1_MIN_EVIDENCE = 20
V1_MIN_RELATIVE_MARGIN = 0.10

# v2 decision constants (see module docstring)
MIN_NET_SUPPORT = 3.0
MAX_COUNTER_RATIO = 0.2
CONFLICT_NET = 1.5
CONFLICT_PURITY = 0.6
MIN_CORNER_FIT = 0.6
MIN_WALL_FIT = 0.3
MIN_FIT_OBJECTS = 3

# evidence geometry constants
CORNER_SHRINK_M = 0.05        # (b) footprint corners pulled inwards by min(0.05, 25% extent)
CROSS_SHRINK_M = 0.10         # (e) footprint shrunk by 0.10 m before the crossing test ...
CROSS_MIN_EXTENT_M = 0.30     # ... and only boxes >= 0.30 m in both footprint axes take part
                              # (thin wall-hugging items -- paintings, panels -- routinely
                              # penetrate the wall line under the TRUE sign)
WALL_MAX_THICKNESS_M = 0.6    # (c) "thin" wall-like box
WALL_MIN_LENGTH_M = 0.4
WALL_ALIGN_TOL_M = 0.15
WALL_PARALLEL_TAN = 0.2       # |cross-axis| <= 0.2 * |axis| (~11 deg)
WALL_SAMPLE_STEP_M = 0.1
OPENING_TOL_M = 0.3           # (d) gap between a door/window box and a hole footprint
SEMANTIC_WEIGHT = 0.5         # (f)
FURNITURE_MAX_EXTENT_M = 6.0
FURNITURE_MAX_AREA_M2 = 12.0

FAMILIES: Dict[str, Tuple[str, ...]] = {
    "containment": ("label_centres", "corner_containment", "edge_crossing"),
    "walls": ("wall_alignment",),
    "openings": ("openings",),
    "semantics": ("room_semantics",),
}

# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------
_WALL_CORE = {"wall", "walls", "partition", "door", "doors", "doorframe", "doorway", "window", "windows"}
_DOOR_TOKENS = {"door", "doors", "doorframe", "doorway"}
_WINDOW_TOKENS = {"window", "windows"}
# A label is wall-like only if EVERY token is in this whitelist (e.g. "sliding door",
# "glass partition", "window frame"); "wall cabinet", "shower room partition",
# "door curtain" are not.
_WALL_ALLOWED = _WALL_CORE | {
    "glass", "sliding", "entrance", "entry", "main", "front", "back", "interior", "exterior", "french",
    "bay", "frame", "wooden", "wood", "single", "double", "swing", "folding", "pocket", "inner", "outer",
    "room", "bearing", "load", "solid", "hinged", "casement", "fixed",
}
_STRUCTURAL_LABELS = {
    "floor", "ceiling", "room", "niche", "floor mold", "ceiling mold", "baseboard", "skirting",
    "skirting board", "beam", "column", "pillar", "stair", "stairs", "staircase", "railing", "roof",
    "ground", "curtain", "curtains", "blind", "blinds", "window curtain", "other",
}
_ROOM_CATEGORY_PHRASES = {
    "bedroom": ("bedroom", "bed room", "master room", "guest room", "kids room", "children room",
                "nursery", "dormitory", "bedchamber"),
    "bathroom": ("bathroom", "bath room", "toilet", "washroom", "restroom", "lavatory", "wc",
                 "shower room", "powder room"),
    "kitchen": ("kitchen",),
}
_OBJECT_CATEGORY_TOKENS = {
    "bedroom": {"bed", "bedside", "crib"},
    "bathroom": {"toilet", "bathtub", "urinal", "bidet", "shower"},
    "kitchen": {"stove", "cooktop", "hob", "oven", "dishwasher", "cooker", "kitchen"},
}
_OBJECT_CATEGORY_LABELS = {"range hood": "kitchen", "cooking bench": "kitchen", "extractor hood": "kitchen"}
_OBJECT_CATEGORY_BLOCK = {"sofa", "pet", "dog", "cat", "bunk"}


def _tokens(label: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(label).lower()) if t]


def label_kind(label: str) -> str:
    """'wall_like' | 'structural' | 'furniture' for a normalised label string."""
    norm = " ".join(_tokens(label))
    toks = set(norm.split())
    if toks and (toks & _WALL_CORE) and toks <= _WALL_ALLOWED:
        return "wall_like"
    if norm in _STRUCTURAL_LABELS or not toks:
        return "structural"
    if toks & _WALL_CORE:  # wall-mounted / in-wall items: neither furniture nor walls
        return "structural"
    return "furniture"


def opening_kind(label: str) -> Optional[str]:
    toks = set(_tokens(label))
    if toks & _DOOR_TOKENS:
        return "door"
    if toks & _WINDOW_TOKENS:
        return "window"
    return None


def object_category(label: str) -> Optional[str]:
    norm = " ".join(_tokens(label))
    if norm in _OBJECT_CATEGORY_LABELS:
        return _OBJECT_CATEGORY_LABELS[norm]
    toks = set(norm.split())
    if toks & _OBJECT_CATEGORY_BLOCK:
        return None
    for cat, keys in _OBJECT_CATEGORY_TOKENS.items():
        if toks & keys:
            return cat
    return None


def room_category(name: str) -> Optional[str]:
    norm = " ".join(_tokens(name))
    if not norm:
        return None
    toks = set(norm.split())
    padded = f" {norm} "
    for cat, phrases in _ROOM_CATEGORY_PHRASES.items():
        for ph in phrases:
            if (" " in ph and f" {ph} " in padded) or ph in toks:
                return cat
    return None


def room_name(room: Mapping[str, Any]) -> str:
    for key in ("name", "type", "room_type", "label", "category"):
        v = room.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_boxes(labels: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every labels.json entry with exactly 8 corners (the v1 convention)."""
    out: List[Dict[str, Any]] = []
    for item in labels or []:
        if not isinstance(item, Mapping):
            continue
        bbox = item.get("bounding_box") or []
        if len(bbox) != 8:
            continue
        try:
            pts = np.array([[c["x"], c["y"], c["z"]] for c in bbox], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        label = str(item.get("label") or "").lower()
        out.append({"label": label, "min": pts.min(axis=0), "max": pts.max(axis=0)})
    return out


def room_polygons(structure: Mapping[str, Any], sign: float) -> List[np.ndarray]:
    polys = []
    for room in (structure or {}).get("rooms", []) or []:
        prof = room.get("profile", []) if isinstance(room, Mapping) else []
        if prof and len(prof) >= 3:
            polys.append(np.asarray([[p[0], sign * p[1]] for p in prof], dtype=np.float64))
    return polys


def _room_names(structure: Mapping[str, Any]) -> List[str]:
    return [room_name(room) for room in (structure or {}).get("rooms", []) or []
            if isinstance(room, Mapping) and len(room.get("profile", []) or []) >= 3]


def hole_rects(structure: Mapping[str, Any], sign: float) -> List[Tuple[str, np.ndarray]]:
    out = []
    for hole in (structure or {}).get("holes", []) or []:
        if not isinstance(hole, Mapping):
            continue
        kind = str(hole.get("type", "")).upper()
        kind = "door" if "DOOR" in kind else "window" if "WINDOW" in kind else None
        prof = hole.get("profile", []) or []
        if kind is None or len(prof) < 2:
            continue
        try:
            xy = np.asarray([[float(p[0]), sign * float(p[1])] for p in prof], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            continue
        out.append((kind, np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])))
    return out


def structure_wall_lines(structure: Mapping[str, Any], sign: float) -> np.ndarray:
    segs = []
    for wall in (structure or {}).get("walls", []) or []:
        loc = wall.get("location") if isinstance(wall, Mapping) else None
        try:
            if loc and len(loc) == 2 and len(loc[0]) >= 2 and len(loc[1]) >= 2:
                segs.append([float(loc[0][0]), sign * float(loc[0][1]), float(loc[1][0]), sign * float(loc[1][1])])
        except (TypeError, ValueError):
            continue
    return np.asarray(segs, dtype=np.float64).reshape(-1, 4)


def _poly_edges(polys: Sequence[np.ndarray]) -> np.ndarray:
    if not polys:
        return np.zeros((0, 4))
    return np.concatenate([np.concatenate([p, np.roll(p, -1, axis=0)], axis=1) for p in polys], axis=0)


# ---------------------------------------------------------------------------
# v1 replica (exactly CollisionDetector._select_structure_y_sign)
# ---------------------------------------------------------------------------
def v1_record(labels: Sequence[Mapping[str, Any]], structure: Mapping[str, Any]) -> Dict[str, Any]:
    boxes = parse_boxes(labels)
    centres = np.array([(b["min"][:2] + b["max"][:2]) / 2.0 for b in boxes], dtype=np.float64).reshape(-1, 2)
    scores = {}
    for sign in (1.0, -1.0):
        inside = points_in_any_polygon(centres, room_polygons(structure, sign))
        scores[f"{sign:+.0f}"] = int(np.count_nonzero(inside))
    pos, neg = scores["+1"], scores["-1"]
    best = max(pos, neg)
    margin = abs(pos - neg) / max(best, 1)
    frozen = len(centres) >= V1_MIN_EVIDENCE and pos != neg and margin >= V1_MIN_RELATIVE_MARGIN
    return {
        "version": V1_VERSION,
        "status": "frozen" if frozen else "ambiguous",
        "sign": 1.0 if pos > neg else -1.0,
        "alignment_scores": scores,
        "evidence_label_centers": int(len(centres)),
        "relative_margin": float(margin),
    }


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
def _summ(d: np.ndarray, weight: float = 1.0) -> Dict[str, float]:
    d = np.asarray(d, dtype=float) * weight
    plus = float(np.clip(d, 0, None).sum())
    minus = float(np.clip(-d, 0, None).sum())
    mass = plus + minus
    return {"plus": plus, "minus": minus, "net": plus - minus, "units": int(len(d)),
            "decisive": int(np.count_nonzero(np.abs(d) > 1e-12)),
            "purity": float(abs(plus - minus) / mass) if mass > 0 else 0.0, "weight": weight}


def _segments_hit_rects(edges: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """(R,) bool: some edge (E,4) intersects the closed rect (R,4: xmin,ymin,xmax,ymax). Liang-Barsky."""
    if len(edges) == 0 or len(rects) == 0:
        return np.zeros(len(rects), dtype=bool)
    x0 = edges[None, :, 0]
    y0 = edges[None, :, 1]
    dx = edges[None, :, 2] - x0
    dy = edges[None, :, 3] - y0
    t0 = np.zeros((len(rects), len(edges)))
    t1 = np.ones((len(rects), len(edges)))
    ok = np.ones((len(rects), len(edges)), dtype=bool)
    xmin, ymin, xmax, ymax = (rects[:, i:i + 1] for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
            p = np.broadcast_to(p, t0.shape)
            q = np.broadcast_to(q, t0.shape)
            zero = p == 0
            ok &= ~(zero & (q < 0))
            r = np.where(zero, 0.0, q / np.where(zero, 1.0, p))
            t0 = np.where(~zero & (p < 0), np.maximum(t0, r), t0)
            t1 = np.where(~zero & (p > 0), np.minimum(t1, r), t1)
    return np.any(ok & (t0 <= t1), axis=1)


def _rect_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean gap between rect sets a (A,4) and b (B,4)."""
    gx = np.maximum(0.0, np.maximum(a[:, None, 0] - b[None, :, 2], b[None, :, 0] - a[:, None, 2]))
    gy = np.maximum(0.0, np.maximum(a[:, None, 1] - b[None, :, 3], b[None, :, 1] - a[:, None, 3]))
    return np.sqrt(gx * gx + gy * gy)


def compute_evidence(labels: Sequence[Mapping[str, Any]], structure: Mapping[str, Any]) -> Dict[str, Any]:
    boxes = parse_boxes(labels)
    furn, walls = [], []
    for b in boxes:
        kind = label_kind(b["label"])
        ext = b["max"][:2] - b["min"][:2]
        if kind == "furniture" and max(ext) <= FURNITURE_MAX_EXTENT_M and ext[0] * ext[1] <= FURNITURE_MAX_AREA_M2:
            furn.append(b)
        elif kind == "wall_like":
            walls.append(b)
    fmin = np.array([b["min"][:2] for b in furn]).reshape(-1, 2)
    fmax = np.array([b["max"][:2] for b in furn]).reshape(-1, 2)
    fext = fmax - fmin
    centres = (fmin + fmax) / 2.0
    s_c = np.minimum(CORNER_SHRINK_M, 0.25 * fext)
    cmin, cmax = fmin + s_c, fmax - s_c
    corners = np.stack([np.stack([cmin[:, 0], cmin[:, 1]], 1), np.stack([cmax[:, 0], cmin[:, 1]], 1),
                        np.stack([cmin[:, 0], cmax[:, 1]], 1), np.stack([cmax[:, 0], cmax[:, 1]], 1)], axis=1)
    cross_ok = (fext.min(axis=1) >= CROSS_MIN_EXTENT_M) if len(furn) else np.zeros(0, dtype=bool)
    cross_rects = np.concatenate([fmin + CROSS_SHRINK_M, fmax - CROSS_SHRINK_M], axis=1)[cross_ok]

    # thin wall-like boxes -> centre-line samples
    wall_items = []
    for b in walls:
        ext = b["max"][:2] - b["min"][:2]
        axis = 0 if ext[0] >= ext[1] else 1
        L, t = float(ext[axis]), float(ext[1 - axis])
        if t > WALL_MAX_THICKNESS_M or L < WALL_MIN_LENGTH_M or L < 1.5 * t:
            continue
        n = int(min(200, max(3, math.ceil(L / WALL_SAMPLE_STEP_M))))
        along = b["min"][axis] + (np.arange(n) + 0.5) / n * L
        mid = 0.5 * (b["min"][1 - axis] + b["max"][1 - axis])
        pts = np.zeros((n, 2))
        pts[:, axis] = along
        pts[:, 1 - axis] = mid
        wall_items.append((axis, t, pts))

    open_boxes = [(opening_kind(b["label"]), np.array([b["min"][0], b["min"][1], b["max"][0], b["max"][1]]))
                  for b in walls if opening_kind(b["label"])]
    names = _room_names(structure)
    room_cats = [room_category(nm) for nm in names]
    sem_idx = [i for i, b in enumerate(furn) if object_category(b["label"])]
    sem_cat = [object_category(furn[i]["label"]) for i in sem_idx]

    per_sign: Dict[str, Dict[str, Any]] = {}
    for sign in (1.0, -1.0):
        key = f"{sign:+.0f}"
        polys = room_polygons(structure, sign)
        edges = _poly_edges(polys)
        r: Dict[str, Any] = {}
        r["centre_in"] = points_in_any_polygon(centres, polys) if len(furn) else np.zeros(0, bool)
        if len(furn):
            cin = points_in_any_polygon(corners.reshape(-1, 2), polys).reshape(-1, 4)
            r["corner_frac"] = cin.mean(axis=1)
        else:
            r["corner_frac"] = np.zeros(0)
        r["crossing"] = _segments_hit_rects(edges, cross_rects).astype(float) if len(cross_rects) else np.zeros(0)
        # (c) wall alignment against polygon edges + structure wall lines
        lines = np.concatenate([edges, structure_wall_lines(structure, sign)], axis=0)
        aligned = []
        for axis, t, pts in wall_items:
            if len(lines) == 0:
                aligned.append(0.0)
                continue
            d_ax = np.abs(lines[:, 2 + axis] - lines[:, axis])
            d_cr = np.abs(lines[:, 3 - axis] - lines[:, 1 - axis])
            par = (d_ax > 1e-6) & (d_cr <= WALL_PARALLEL_TAN * d_ax)
            if not np.any(par):
                aligned.append(0.0)
                continue
            L = lines[par]
            dist, _ = point_segment_distance_clip(pts[:, 0:1], pts[:, 1:2], L[None, :, 0], L[None, :, 1],
                                                  L[None, :, 2], L[None, :, 3])
            aligned.append(float(np.mean(dist.min(axis=1) <= 0.5 * t + WALL_ALIGN_TOL_M)))
        r["wall_aligned"] = np.asarray(aligned, dtype=float)
        # (d) openings
        holes = hole_rects(structure, sign)
        matched = []
        for kind, rect in open_boxes:
            same = [h for k, h in holes if k == kind]
            if same:
                matched.append(float(_rect_gap(rect[None, :], np.array(same)).min() <= OPENING_TOL_M))
            else:
                matched.append(0.0)
        r["opening_matched"] = np.asarray(matched, dtype=float)
        # (f) semantics
        if sem_idx and any(room_cats):
            ridx = points_room_index(centres[sem_idx], polys)
            r["semantic_match"] = np.array([float(ri >= 0 and room_cats[ri] == c) for ri, c in zip(ridx, sem_cat)])
        else:
            r["semantic_match"] = np.zeros(0)
        per_sign[key] = r

    P, M = per_sign["+1"], per_sign["-1"]
    types: Dict[str, Dict[str, Any]] = {}
    types["label_centres"] = _summ(P["centre_in"].astype(float) - M["centre_in"].astype(float))
    types["label_centres"]["score"] = {"+1": int(P["centre_in"].sum()), "-1": int(M["centre_in"].sum())}
    types["corner_containment"] = _summ(P["corner_frac"] - M["corner_frac"])
    types["corner_containment"]["score"] = {"+1": float(P["corner_frac"].sum()), "-1": float(M["corner_frac"].sum())}
    types["edge_crossing"] = _summ(M["crossing"] - P["crossing"])
    types["edge_crossing"]["score"] = {"+1": -float(P["crossing"].sum()), "-1": -float(M["crossing"].sum())}
    types["wall_alignment"] = _summ(P["wall_aligned"] - M["wall_aligned"])
    types["wall_alignment"]["score"] = {"+1": float(P["wall_aligned"].sum()), "-1": float(M["wall_aligned"].sum())}
    types["openings"] = _summ(P["opening_matched"] - M["opening_matched"])
    types["openings"]["score"] = {"+1": float(P["opening_matched"].sum()), "-1": float(M["opening_matched"].sum())}
    types["room_semantics"] = _summ(P["semantic_match"] - M["semantic_match"], SEMANTIC_WEIGHT)
    types["room_semantics"]["score"] = {"+1": float(P["semantic_match"].sum()), "-1": float(M["semantic_match"].sum())}

    n_f = len(furn)
    fits = {
        "n_furniture": n_f,
        "n_thin_walls": len(wall_items),
        "corner_fit": {k: (float(per_sign[k]["corner_frac"].mean()) if n_f else None) for k in ("+1", "-1")},
        "wall_fit": {k: (float(per_sign[k]["wall_aligned"].mean()) if wall_items else None) for k in ("+1", "-1")},
    }
    return {"types": types, "fits": fits}


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
def _decide(types: Dict[str, Dict[str, Any]], fits: Dict[str, Any]) -> Dict[str, Any]:
    notes: List[str] = []
    excluded = set()
    wall = types["wall_alignment"]
    if wall["plus"] + wall["minus"] > 0 and fits["wall_fit"]["+1"] is not None:
        lean = "+1" if wall["net"] >= 0 else "-1"
        if fits["wall_fit"][lean] < MIN_WALL_FIT:
            excluded.add("wall_alignment")
            notes.append(f"wall_alignment_ignored_fit_{fits['wall_fit'][lean]:.2f}")
    fam: Dict[str, Dict[str, float]] = {}
    for name, members in FAMILIES.items():
        use = [m for m in members if m not in excluded]
        fam[name] = {"plus": max([types[m]["plus"] for m in use] or [0.0]),
                     "minus": max([types[m]["minus"] for m in use] or [0.0])}
    P = sum(f["plus"] for f in fam.values())
    M = sum(f["minus"] for f in fam.values())
    out: Dict[str, Any] = {"support_plus": P, "support_minus": M, "families": fam, "notes": notes,
                           "excluded_types": sorted(excluded)}
    if P == M:
        out.update(status="ambiguous", sign=None, reason="no_evidence" if P == 0 else "tie")
        return out
    winner = 1.0 if P > M else -1.0
    F, A = max(P, M), min(P, M)
    out["sign"] = winner
    out["support"] = F
    out["counter"] = A
    if F - A < MIN_NET_SUPPORT:
        out.update(status="ambiguous", reason="insufficient_evidence")
        return out
    if A > MAX_COUNTER_RATIO * F:
        out.update(status="ambiguous", reason="counter_evidence")
        return out
    for name, t in types.items():
        if name in excluded:
            continue
        opp = -t["net"] * winner
        if opp >= CONFLICT_NET and t["purity"] >= CONFLICT_PURITY:
            out.update(status="ambiguous", reason=f"conflicting_evidence:{name}")
            return out
    wkey = f"{winner:+.0f}"
    corner_ok = fits["n_furniture"] >= MIN_FIT_OBJECTS and fits["corner_fit"][wkey] >= MIN_CORNER_FIT
    wall_ok = (fits["n_thin_walls"] >= MIN_FIT_OBJECTS and "wall_alignment" not in excluded
               and fits["wall_fit"][wkey] >= MIN_CORNER_FIT)
    corner_bad = fits["n_furniture"] >= MIN_FIT_OBJECTS and not corner_ok
    if corner_bad and not wall_ok:
        out.update(status="ambiguous", reason="no_sign_fits")
        return out
    out.update(status="frozen", reason="v2_evidence_concordant")
    return out


def calibrate_structure_y_sign(labels: Sequence[Mapping[str, Any]], structure: Mapping[str, Any]) -> Dict[str, Any]:
    """Calibrate the structure.json Y sign for one scene (see module docstring).

    Returns {sign, status, version, evidence, reason, rule, v1, v2_own,
    conflict_with_v1, env_override_required, thresholds}.
    ``sign`` is always +1.0/-1.0; for ambiguous scenes it is v1's operational
    fallback and must not be used.
    """
    structure = structure if isinstance(structure, Mapping) else {}
    v1 = v1_record(labels, structure)
    ev = compute_evidence(labels, structure)
    own = _decide(ev["types"], ev["fits"])
    conflict = bool(v1["status"] == "frozen" and own["status"] == "frozen" and own["sign"] != v1["sign"])
    if v1["status"] == "frozen":
        sign, status, rule = v1["sign"], "frozen", "v1_frozen"
        reason = "v1_frozen" + (";v2_own_evidence_disagrees" if conflict else "")
    elif own["status"] == "frozen":
        sign, status, rule, reason = float(own["sign"]), "frozen", "v2_evidence", own["reason"]
    else:
        sign, status, rule, reason = v1["sign"], "ambiguous", "v2_ambiguous", own["reason"]
    evidence = dict(ev["types"])
    evidence["fits"] = ev["fits"]
    evidence["v1_label_centres_all"] = {"score": dict(v1["alignment_scores"]),
                                        "units": v1["evidence_label_centers"],
                                        "relative_margin": v1["relative_margin"]}
    return {
        "version": VERSION,
        "sign": float(sign),
        "status": status,
        "reason": reason,
        "rule": rule,
        "evidence": evidence,
        "v1": v1,
        "v2_own": own,
        "conflict_with_v1": conflict,
        # Recommended admission flag: frozen AND v2's own evidence does not
        # contradict v1.  Scenes with conflict_with_v1 keep v1's sign (the env
        # will use it) but are likely mirrored; quarantine or review them.
        "usable": bool(status == "frozen" and not conflict),
        "env_override_required": bool(status == "frozen" and v1["status"] != "frozen"),
        "thresholds": {
            "min_net_support": MIN_NET_SUPPORT, "max_counter_ratio": MAX_COUNTER_RATIO,
            "conflict_net": CONFLICT_NET, "conflict_purity": CONFLICT_PURITY,
            "min_corner_fit": MIN_CORNER_FIT, "min_wall_fit": MIN_WALL_FIT,
            "v1_min_evidence": V1_MIN_EVIDENCE, "v1_min_relative_margin": V1_MIN_RELATIVE_MARGIN,
        },
    }


def env_overrides(records: Union[Mapping[str, Mapping[str, Any]], Iterable[Mapping[str, Any]]],
                  include_v1_frozen: bool = False) -> Dict[str, float]:
    """``structure_y_sign_overrides`` for the runtime CollisionDetector.

    ``records`` is {scene_id: record} or an iterable of records carrying a
    ``scene_id`` key.  By default only scenes that NEED an override (v2 frozen,
    v1 not frozen) are returned; ``include_v1_frozen=True`` also pins the
    v1-frozen scenes (same sign as v1, so harmless but explicit).
    """
    items: Iterable[Tuple[str, Mapping[str, Any]]]
    if isinstance(records, Mapping):
        items = records.items()
    else:
        items = ((str(r.get("scene_id")), r) for r in records)
    out: Dict[str, float] = {}
    for sid, rec in items:
        if rec.get("status") != "frozen" or not sid or sid == "None":
            continue
        if rec.get("env_override_required") or include_v1_frozen:
            out[str(sid)] = float(rec["sign"])
    return dict(sorted(out.items()))


def calibrate_scene_dir(scene_dir: Union[str, Path], scene_id: Optional[str] = None) -> Dict[str, Any]:
    scene_dir = Path(scene_dir)
    labels = json.loads((scene_dir / "labels.json").read_text(encoding="utf-8"))
    sp = scene_dir / "structure.json"
    structure = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    rec = calibrate_structure_y_sign(labels, structure)
    rec["scene_id"] = scene_id or scene_dir.name
    rec["structure_present"] = sp.exists()
    return rec


# ---------------------------------------------------------------------------
# Synthetic benchmark (random layouts with known sign)
# ---------------------------------------------------------------------------
_ROOM_FURNITURE = {
    "bedroom": [("double bed", (1.6, 2.1, 0.6)), ("single bed", (1.0, 2.0, 0.55)), ("wardrobe", (1.8, 0.6, 2.2)),
                ("bedside table", (0.45, 0.45, 0.55)), ("desk", (1.2, 0.6, 0.75)), ("dresser", (1.0, 0.5, 0.9)),
                ("chair", (0.5, 0.5, 0.9))],
    "living room": [("multi person sofa", (2.2, 0.9, 0.85)), ("armchair", (0.8, 0.8, 0.9)),
                    ("tea table", (1.0, 0.6, 0.45)), ("tv cabinet", (1.8, 0.45, 0.55)),
                    ("bookshelf", (1.0, 0.35, 1.9)), ("plant", (0.4, 0.4, 1.2)), ("floor lamp", (0.35, 0.35, 1.7)),
                    ("rug", (2.0, 1.5, 0.02))],
    "kitchen": [("cooking bench", (1.8, 0.6, 0.9)), ("refrigerator", (0.7, 0.7, 1.9)), ("cabinet", (1.2, 0.5, 0.9)),
                ("stove", (0.6, 0.6, 0.9)), ("dining table", (1.4, 0.9, 0.75)), ("dining chair", (0.5, 0.5, 0.95))],
    "bathroom": [("toilet", (0.4, 0.7, 0.8)), ("basin cabinet", (0.8, 0.5, 0.85)), ("bathtub", (1.6, 0.75, 0.6)),
                 ("towel rack", (0.6, 0.2, 1.2))],
    "study": [("desk", (1.4, 0.7, 0.75)), ("office chair", (0.6, 0.6, 1.1)), ("bookshelf", (1.2, 0.35, 2.0)),
              ("sofa", (1.8, 0.8, 0.85))],
}
_ROOM_TYPES = list(_ROOM_FURNITURE)
_SMALL_ITEMS = ["vase", "book", "cup", "decoration", "pillow", "storage box", "potted plant", "bottle"]


def _box(ins_id: str, label: str, xmin, ymin, zmin, xmax, ymax, zmax) -> Dict[str, Any]:
    corners = [{"x": float(x), "y": float(y), "z": float(z)}
               for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)]
    return {"ins_id": str(ins_id), "label": label, "bounding_box": corners}


def _bsp(rect, n_rooms, rng, min_size=2.2):
    rooms = [tuple(rect)]
    for _ in range(60):
        if len(rooms) >= n_rooms:
            break
        idx = int(np.argmax([(r[2] - r[0]) * (r[3] - r[1]) for r in rooms]))
        x0, y0, x1, y1 = rooms[idx]
        w, h = x1 - x0, y1 - y0
        axis = 0 if w >= h else 1
        if rng.random() < 0.25:
            axis = 1 - axis
        span = w if axis == 0 else h
        if span < 2 * min_size:
            axis = 1 - axis
            span = w if axis == 0 else h
            if span < 2 * min_size:
                continue
        lo = (x0 if axis == 0 else y0) + min_size
        hi = (x1 if axis == 0 else y1) - min_size
        cut = round(float(rng.uniform(lo, hi)) / 0.05) * 0.05
        if axis == 0:
            a, b = (x0, y0, cut, y1), (cut, y0, x1, y1)
        else:
            a, b = (x0, y0, x1, cut), (x0, cut, x1, y1)
        rooms[idx:idx + 1] = [a, b]
    return rooms


def _shared_edges(rooms):
    """Pairs (i, j, axis, coord, lo, hi): rooms i, j touch along a line."""
    out = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            for axis in (0, 1):
                lo_i, hi_i = (a[0], a[2]) if axis == 0 else (a[1], a[3])
                lo_j, hi_j = (b[0], b[2]) if axis == 0 else (b[1], b[3])
                o_lo_i, o_hi_i = (a[1], a[3]) if axis == 0 else (a[0], a[2])
                o_lo_j, o_hi_j = (b[1], b[3]) if axis == 0 else (b[0], b[2])
                for c in (hi_i,) if abs(hi_i - lo_j) < 1e-9 else ((lo_i,) if abs(lo_i - hi_j) < 1e-9 else ()):
                    lo, hi = max(o_lo_i, o_lo_j), min(o_hi_i, o_hi_j)
                    if hi - lo > 1e-6:
                        out.append((i, j, axis, c, lo, hi))
    return out


def _rect_edges(r):
    x0, y0, x1, y1 = r
    # (axis of the wall line normal, coord, lo, hi, outward sign)
    return [(1, y0, x0, x1, -1), (1, y1, x0, x1, 1), (0, x0, y0, y1, -1), (0, x1, y0, y1, 1)]


def _in_rects(pts, rects):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    ok = np.zeros(len(pts), dtype=bool)
    for x0, y0, x1, y1 in rects:
        ok |= (pts[:, 0] >= x0 - 1e-9) & (pts[:, 0] <= x1 + 1e-9) & (pts[:, 1] >= y0 - 1e-9) & (pts[:, 1] <= y1 + 1e-9)
    return ok


def synthetic_layout(rng: np.random.Generator, kind: str) -> Tuple[List[dict], dict, dict]:
    """One random InteriorGS-like layout with a known structure Y sign.

    kinds: generic, hull_symmetric (room union symmetric about y=0, partitions
    random), notch (a corner room removed; furniture only where both mirror
    images have floor), sparse (1-2 rooms, 3-6 objects), fully_symmetric
    (structure exactly mirror symmetric -> sign not identifiable).
    """
    sign = float(rng.choice([-1.0, 1.0]))
    W = float(rng.uniform(6, 16))
    H = float(rng.uniform(5, 12))
    if kind == "sparse":
        W, H = float(rng.uniform(3.5, 8)), float(rng.uniform(3, 6))
    x0 = round(float(rng.uniform(-12, 8)), 2)
    if kind in ("hull_symmetric", "fully_symmetric") or (kind in ("notch", "wall_heavy") and rng.random() < 0.5):
        y0 = -H / 2
    else:
        y0 = round(float(rng.uniform(-H - 2, 6)), 2)
    if kind == "fully_symmetric":
        half = _bsp((x0, 0.0, x0 + W, H / 2), int(rng.integers(1, 4)), rng, min_size=1.8)
        rooms = []
        for r in half:
            mir = (r[0], -r[3], r[2], -r[1])
            if abs(r[1]) < 1e-9 and rng.random() < 0.5:
                rooms.append((r[0], -r[3], r[2], r[3]))
            else:
                rooms += [r, mir]
    else:
        n_rooms = int(rng.integers(1, 3)) if kind == "sparse" else int(rng.integers(1, 7))
        rooms = _bsp((x0, y0, x0 + W, y0 + H), n_rooms, rng)
        if kind == "notch" and len(rooms) >= 3:
            corner = (x0 + W, y0 + H) if rng.random() < 0.5 else (x0, y0 + H)
            cand = [i for i, r in enumerate(rooms)
                    if (abs(r[2] - corner[0]) < 1e-9 or abs(r[0] - corner[0]) < 1e-9) and abs(r[3] - corner[1]) < 1e-9]
            if cand:
                rooms.pop(cand[0])
    rtypes = []
    for i, r in enumerate(rooms):
        rtypes.append(_ROOM_TYPES[int(rng.integers(len(_ROOM_TYPES)))])
    if kind == "fully_symmetric":  # mirrored rooms get the same name
        for i, r in enumerate(rooms):
            for j in range(i):
                if abs(rooms[j][1] + r[3]) < 1e-9 and abs(rooms[j][3] + r[1]) < 1e-9 and abs(rooms[j][0] - r[0]) < 1e-9:
                    rtypes[i] = rtypes[j]
    t = float(rng.uniform(0.08, 0.24))
    gapped = bool(rng.random() < 0.5) or kind == "wall_heavy"
    outward = bool(rng.random() < 0.5)
    walls_on = rng.random() < (0.7 if kind == "sparse" else 0.92) or kind == "wall_heavy"
    floor_polys = [(r[0] + t / 2, r[1] + t / 2, r[2] - t / 2, r[3] - t / 2) if gapped else r for r in rooms]
    mirror_floor = [(r[0], -r[3], r[2], -r[1]) for r in floor_polys]
    labels: List[dict] = []
    nid = [0]

    def add(label, *coords):
        nid[0] += 1
        labels.append(_box(str(nid[0]), label, *coords))

    # walls (unique line pieces)
    seen = set()
    shared = _shared_edges(rooms)
    if walls_on:
        for r in rooms:
            for ax, c, lo, hi, out in _rect_edges(r):
                k = (ax, round(c, 4), round(lo, 4), round(hi, 4))
                if k in seen or rng.random() < 0.12:
                    continue
                seen.add(k)
                interior = any(e[2] == ax and abs(e[3] - c) < 1e-9 and e[4] < hi - 1e-6 and e[5] > lo + 1e-6
                               for e in shared)
                off = out * t / 2 if (outward and not interior and not gapped) else 0.0
                c0, c1 = c + off - t / 2, c + off + t / 2
                hgt = float(rng.uniform(2.6, 2.9))
                if ax == 1:
                    add("wall", lo - t / 2, c0, 0.0, hi + t / 2, c1, hgt)
                else:
                    add("wall", c0, lo - t / 2, 0.0, c1, hi + t / 2, hgt)
    # doors on shared edges, windows on exterior edges
    holes = []
    if kind != "fully_symmetric":
        for (i, j, axis, c, lo, hi) in shared:
            if hi - lo < 1.4 or rng.random() > 0.8:
                continue
            wd = float(rng.uniform(0.75, 1.0))
            p = float(rng.uniform(lo + 0.2, hi - 0.2 - wd))
            if axis == 0:   # wall line x = c
                rect = (c - t / 2, p, c + t / 2, p + wd)
            else:
                rect = (p, c - t / 2, p + wd, c + t / 2)
            holes.append(("DOOR", rect))
            if rng.random() < 0.75:
                add("door", rect[0], rect[1], 0.0, rect[2], rect[3], 2.1)
        for r in rooms:
            for ax, c, lo, hi, out in _rect_edges(r):
                interior = any(e[2] == ax and abs(e[3] - c) < 1e-9 and e[4] < hi - 1e-6 and e[5] > lo + 1e-6
                               for e in shared)
                if interior or hi - lo < 1.6 or rng.random() > 0.35:
                    continue
                wd = float(rng.uniform(0.8, min(2.0, hi - lo - 0.4)))
                p = float(rng.uniform(lo + 0.2, hi - 0.2 - wd))
                rect = (p, c - t / 2, p + wd, c + t / 2) if ax == 1 else (c - t / 2, p, c + t / 2, p + wd)
                holes.append(("WINDOW", rect))
                if rng.random() < 0.7:
                    add("window", rect[0], rect[1], 0.9, rect[2], rect[3], 2.1)
    # furniture (+ small items on top and wall decorations, as in real scenes)
    restrict = kind == "notch"
    dense = kind not in ("sparse", "wall_heavy")
    placed: List[Tuple[float, float, float, float]] = []
    n_total = int(rng.integers(3, 7)) if kind == "sparse" else None
    for ri in range(len(rooms)):
        fr = floor_polys[ri]
        if n_total is not None:
            k = max(1, n_total // len(rooms))
        elif kind == "wall_heavy":
            k = int(rng.integers(1, 5))
        else:
            k = int(rng.integers(2, 10))
        catalogue = _ROOM_FURNITURE[rtypes[ri]]
        for _ in range(k):
            label, (a, b, hz) = catalogue[int(rng.integers(len(catalogue)))]
            for _try in range(25):
                dx, dy = a * rng.uniform(0.8, 1.2), b * rng.uniform(0.8, 1.2)
                if rng.random() < 0.5:
                    dx, dy = dy, dx
                if dx > fr[2] - fr[0] - 0.1 or dy > fr[3] - fr[1] - 0.1:
                    continue
                if rng.random() < 0.65:   # against a wall
                    side = int(rng.integers(4))
                    gap = float(rng.uniform(0.0, 0.05))
                    if rng.random() < 0.08:
                        gap = -float(rng.uniform(0.0, 0.08))  # label noise: penetrates the wall
                    xa = float(rng.uniform(fr[0], fr[2] - dx))
                    ya = float(rng.uniform(fr[1], fr[3] - dy))
                    if side == 0:
                        ya = fr[1] + gap
                    elif side == 1:
                        ya = fr[3] - dy - gap
                    elif side == 2:
                        xa = fr[0] + gap
                    else:
                        xa = fr[2] - dx - gap
                else:
                    if fr[2] - fr[0] - dx < 0.6 or fr[3] - fr[1] - dy < 0.6:
                        continue
                    xa = float(rng.uniform(fr[0] + 0.3, fr[2] - dx - 0.3))
                    ya = float(rng.uniform(fr[1] + 0.3, fr[3] - dy - 0.3))
                box = (xa, ya, xa + dx, ya + dy)
                if any(not (box[2] <= p[0] or box[0] >= p[2] or box[3] <= p[1] or box[1] >= p[3]) for p in placed):
                    continue
                if restrict:
                    cs = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
                    if not _in_rects(cs, mirror_floor).all():
                        continue
                placed.append(box)
                top = hz * rng.uniform(0.9, 1.1)
                add(label, box[0], box[1], 0.0, box[2], box[3], top)
                if dense and rng.random() < 0.6:
                    for _s in range(int(rng.integers(1, 4))):
                        sw, sd = float(rng.uniform(0.08, 0.35)), float(rng.uniform(0.08, 0.35))
                        if sw >= dx or sd >= dy:
                            continue
                        sx = float(rng.uniform(box[0], box[2] - sw))
                        sy = float(rng.uniform(box[1], box[3] - sd))
                        small = _SMALL_ITEMS[int(rng.integers(len(_SMALL_ITEMS)))]
                        add(small, sx, sy, top, sx + sw, sy + sd, top + float(rng.uniform(0.05, 0.4)))
                break
        if dense:
            for _d in range(int(rng.integers(0, 4))):
                side = int(rng.integers(4))
                w = float(rng.uniform(0.4, 1.2))
                dep = float(rng.uniform(0.02, 0.06))
                gap = float(rng.uniform(-0.02, 0.02))
                if side < 2:
                    if fr[2] - fr[0] < w + 0.2:
                        continue
                    xa = float(rng.uniform(fr[0] + 0.1, fr[2] - w - 0.1))
                    ya = fr[1] + gap if side == 0 else fr[3] - dep - gap
                    rect = (xa, ya, xa + w, ya + dep)
                else:
                    if fr[3] - fr[1] < w + 0.2:
                        continue
                    ya = float(rng.uniform(fr[1] + 0.1, fr[3] - w - 0.1))
                    xa = fr[0] + gap if side == 2 else fr[2] - dep - gap
                    rect = (xa, ya, xa + dep, ya + w)
                if restrict and not _in_rects([(rect[0], rect[1]), (rect[2], rect[3])], mirror_floor).all():
                    continue
                zb = float(rng.uniform(1.1, 1.6))
                add("decorative painting", rect[0], rect[1], zb, rect[2], rect[3], zb + float(rng.uniform(0.3, 0.8)))
        if dense and rng.random() < 0.5:
            cx, cy = (fr[0] + fr[2]) / 2, (fr[1] + fr[3]) / 2
            add("ceiling lamp", cx - 0.3, cy - 0.3, 2.5, cx + 0.3, cy + 0.3, 2.75)
    if kind not in ("sparse",) and rng.random() < 0.15:   # a few objects outside every room
        for _ in range(int(rng.integers(1, 3))):
            ox = x0 + W + float(rng.uniform(0.5, 2.0))
            oy = y0 + float(rng.uniform(0, H))
            add("plant", ox, oy, 0.0, ox + 0.4, oy + 0.4, 1.0)
    if kind != "sparse" and rng.random() < 0.7:
        add("floor", x0, y0, -0.02, x0 + W, y0 + H, 0.0)
        add("ceiling", x0, y0, 2.75, x0 + W, y0 + H, 2.8)
    order = rng.permutation(len(labels))
    labels = [labels[i] for i in order]
    structure = {
        "rooms": [{"name": rtypes[i], "profile": [[p[0], sign * p[1]] for p in
                                                  ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))]}
                  for i, r in enumerate(floor_polys)],
        "holes": [{"type": ht, "profile": [[p[0], sign * p[1]] for p in
                                           ((h[0], h[1]), (h[2], h[1]), (h[2], h[3]), (h[0], h[3]))]}
                  for ht, h in holes],
    }
    # identifiability: how far does the mirrored structure move any room edge?
    mirrored = [(r[0], -r[3], r[2], -r[1]) for r in floor_polys]
    shift = 0.0
    for r in floor_polys:
        best = min(max(abs(r[0] - m[0]), abs(r[1] - m[1]), abs(r[2] - m[2]), abs(r[3] - m[3])) for m in mirrored)
        shift = max(shift, best)
    names_sym = all(
        any(max(abs(r[0] - m[0]), abs(r[1] - m[1]), abs(r[2] - m[2]), abs(r[3] - m[3])) < 1e-6
            and rtypes[i] == rtypes[j] for j, m in enumerate(mirrored))
        for i, r in enumerate(floor_polys))
    if kind == "fully_symmetric" or (shift < 1e-6 and names_sym and not holes):
        identifiable = "no"
    elif shift < 0.5:
        identifiable = "weak"
    else:
        identifiable = "yes"
    truth = {"sign": sign, "kind": kind, "identifiable": identifiable, "mirror_shift_m": shift,
             "n_labels": len(labels), "n_rooms": len(rooms), "gapped": gapped, "walls_labelled": bool(walls_on)}
    return labels, structure, truth


BENCHMARK_MIX = (("generic", 0.35), ("hull_symmetric", 0.2), ("notch", 0.17), ("sparse", 0.18),
                 ("fully_symmetric", 0.1))
# Stress mix: few furniture boxes per room, walls/doors/windows labelled and room
# polygons inset by half a wall thickness ("gapped").  Wall/door/window centres
# then sit outside every room under the TRUE sign but often inside a mirrored
# room, which biases the v1 all-label centre count towards the WRONG sign.
STRESS_MIX = (("wall_heavy", 1.0),)


def _outcome(status: str, sign: Optional[float], truth: Mapping[str, Any]) -> str:
    if truth["identifiable"] == "no":
        return "ambiguous" if status != "frozen" else "frozen"
    if status != "frozen":
        return "ambiguous"
    return "frozen_correct" if sign == truth["sign"] else "frozen_wrong"


def run_benchmark(n: int = 400, seed: int = 0, unrelated: int = 50,
                  mix: Sequence[Tuple[str, float]] = BENCHMARK_MIX) -> Dict[str, Any]:
    """Calibrate ``n`` random layouts with known sign; return confusion counts.

    Outcomes per identifiability class of the layout:
      yes  (mirroring moves some room edge by >= 0.5 m) and
      weak (< 0.5 m, or only holes / room names differ):
             frozen_correct | frozen_wrong | ambiguous
      no   (structure exactly mirror symmetric): ambiguous | frozen (harmless)
    ``final`` is the calibrate_structure_y_sign decision (v1-frozen scenes keep
    v1's sign, rule R0); ``v2_own`` is the v2 evidence rule alone; ``v1`` the
    runtime rule.  ``unrelated`` pairs labels of one layout with the structure
    of another (no valid sign exists; freezing there is a false alarm).
    """
    rng = np.random.default_rng(seed)
    kinds = [k for k, _ in mix]
    probs = np.array([p for _, p in mix], dtype=float)
    conf = {name: {"yes": Counter(), "weak": Counter(), "no": Counter()} for name in ("final", "v2_own", "v1")}
    by_kind: Dict[str, Counter] = {k: Counter() for k in kinds}
    conflicts = Counter()
    rows = []
    for i in range(n):
        kind = kinds[int(rng.choice(len(kinds), p=probs / probs.sum()))]
        labels, structure, truth = synthetic_layout(rng, kind)
        rec = calibrate_structure_y_sign(labels, structure)
        outs = {
            "final": _outcome(rec["status"], rec["sign"], truth),
            "v2_own": _outcome(rec["v2_own"]["status"], rec["v2_own"].get("sign"), truth),
            "v1": _outcome(rec["v1"]["status"], rec["v1"]["sign"], truth),
        }
        for name, o in outs.items():
            conf[name][truth["identifiable"]][o] += 1
        by_kind[kind][outs["final"]] += 1
        if rec["conflict_with_v1"]:
            conflicts["conflict_flagged_v1_" + ("wrong" if rec["v1"]["sign"] != truth["sign"] else "right")] += 1
        rows.append({"i": i, "truth": truth, "status": rec["status"], "sign": rec["sign"], "rule": rec["rule"],
                     "reason": rec["reason"], "v2_own_reason": rec["v2_own"]["reason"], **{f"outcome_{k}": v for k, v in outs.items()}})
    unrel = {"final": Counter(), "v2_own": Counter(), "v1": Counter()}
    for _ in range(unrelated):
        la, _, _ = synthetic_layout(rng, "generic")
        _, sb, _ = synthetic_layout(rng, "generic")
        rec = calibrate_structure_y_sign(la, sb)
        unrel["final"][rec["status"]] += 1
        unrel["v2_own"][rec["v2_own"]["status"]] += 1
        unrel["v1"][rec["v1"]["status"]] += 1

    def acc(c: Counter) -> Optional[float]:
        tot = sum(c.values())
        return (c["frozen_correct"] / tot) if tot else None

    return {
        "n": n, "seed": seed, "mix": [list(m) for m in mix],
        "confusion": {name: {k: dict(v) for k, v in c.items()} for name, c in conf.items()},
        "accuracy_identifiable": {name: acc(c["yes"]) for name, c in conf.items()},
        "frozen_wrong": {name: c["yes"]["frozen_wrong"] + c["weak"]["frozen_wrong"] for name, c in conf.items()},
        "by_kind_final": {k: dict(v) for k, v in by_kind.items()},
        "conflict_with_v1": dict(conflicts),
        "unrelated_structure": {k: dict(v) for k, v in unrel.items()},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _find_scene_dirs(root: Path) -> List[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "labels.json" in filenames:
            out.append(Path(dirpath))
            dirnames[:] = []
        dirnames.sort()
    return sorted(out)


def calibrate_root(root: Union[str, Path]) -> Dict[str, Any]:
    root = Path(root)
    records: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for d in _find_scene_dirs(root):
        sid = d.name
        try:
            records[sid] = calibrate_scene_dir(d, sid)
        except Exception as exc:  # pragma: no cover - reported, never fatal
            errors[sid] = f"{type(exc).__name__}: {exc}"
    v1_counts = Counter(r["v1"]["status"] for r in records.values())
    v2_counts = Counter(r["status"] for r in records.values())
    transitions = Counter(f"{r['v1']['status']}->{r['status']}" for r in records.values())
    own_vs_v1 = Counter()
    disagreements = []
    for sid, r in sorted(records.items()):
        own = r["v2_own"]
        if r["v1"]["status"] == "frozen":
            own_vs_v1["agree" if own["status"] == "frozen" and own["sign"] == r["v1"]["sign"] else
                      "conflict" if own["status"] == "frozen" else "abstain"] += 1
        if r["status"] != r["v1"]["status"] or r["sign"] != r["v1"]["sign"] or r["conflict_with_v1"]:
            disagreements.append({
                "scene_id": sid, "v1_status": r["v1"]["status"], "v1_sign": r["v1"]["sign"],
                "v1_scores": r["v1"]["alignment_scores"], "v2_status": r["status"], "v2_sign": r["sign"],
                "v2_rule": r["rule"], "v2_reason": r["reason"], "conflict_with_v1": r["conflict_with_v1"],
                "support": own.get("support"), "counter": own.get("counter"),
                "families": own.get("families"),
            })
    ambiguous_reasons = Counter(r["reason"] for r in records.values() if r["status"] == "ambiguous")
    return {
        "version": VERSION,
        "scenes_root": str(root),
        "n_scenes": len(records),
        "v1_status_counts": dict(v1_counts),
        "v2_status_counts": dict(v2_counts),
        "v1_to_v2": dict(transitions),
        "v2_own_vs_v1_frozen": dict(own_vs_v1),
        "v2_ambiguous_reasons": dict(ambiguous_reasons),
        "disagreements": disagreements,
        "structure_y_sign_overrides": env_overrides(records),
        "errors": errors,
        "records": records,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes-root", type=Path, help="directory searched for scene dirs containing labels.json")
    ap.add_argument("--out", type=Path, help="write the full JSON report here")
    ap.add_argument("--benchmark", type=int, default=0, help="run the synthetic benchmark with N layouts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stress", action="store_true", help="benchmark the wall-heavy stress mix instead")
    args = ap.parse_args(argv)
    if not args.scenes_root and not args.benchmark:
        ap.error("give --scenes-root or --benchmark")
    report: Dict[str, Any] = {}
    if args.benchmark:
        bench = run_benchmark(args.benchmark, args.seed, mix=STRESS_MIX if args.stress else BENCHMARK_MIX)
        report["benchmark"] = bench
        print(json.dumps({k: v for k, v in bench.items() if k != "rows"}, indent=1))
    if args.scenes_root:
        rep = calibrate_root(args.scenes_root)
        report["calibration"] = rep
        summary = {k: rep[k] for k in ("n_scenes", "v1_status_counts", "v2_status_counts", "v1_to_v2",
                                       "v2_own_vs_v1_frozen", "v2_ambiguous_reasons")}
        summary["n_disagreements"] = len(rep["disagreements"])
        summary["n_overrides"] = len(rep["structure_y_sign_overrides"])
        summary["errors"] = len(rep["errors"])
        print(json.dumps(summary, indent=1))
        for d in rep["disagreements"]:
            print(f"  {d['scene_id']}: v1={d['v1_status']}/{d['v1_sign']:+.0f} {d['v1_scores']} -> "
                  f"v2={d['v2_status']}/{d['v2_sign']:+.0f} ({d['v2_reason']}) support={d['support']} "
                  f"counter={d['counter']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_jsonable(report), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
