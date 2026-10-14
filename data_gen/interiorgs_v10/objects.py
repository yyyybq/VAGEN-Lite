"""Candidate objects, single targets and object pairs for the v10 generator.

Public API
----------
    cfg = get_preset("v10_relaxed", max_pairs_per_scene=80)   # frozen ObjectConfig
    objs, f1   = candidate_objects(scene, cfg)                # every obj has obj.ref
    singles, f2 = select_singles(scene, objs, cfg)
    pairs, f3   = select_pairs(scene, objs, cfg, max_pairs=None, seed=0)
    ordered     = expand_ordered(pairs)                       # (A,B) and (B,A)

Order of operations inside ``candidate_objects``: semantic exclusion ->
single-object geometry -> (v9 only) category uniqueness -> referring
expressions (``referring.assign_referring_expressions``) -> drop objects whose
ref is None.  Returned objects are shallow copies of ``scene.objects`` entries
(same id / label / arrays / room_index) carrying ``ref`` and ``ref_method``;
``scene.objects`` is never mutated, so several presets can run on one Scene.

Presets
-------
``v9_strict``  reconstructs the v9 object stage from the legacy
``ObjectSelectionConfig`` defaults (data_gen/active_spatial_pipeline):
exact-label blacklist, per-axis dims in [0.1, 3.0] m, volume >= 0.1 m^3,
shortest/longest edge >= 0.1, category unique among the filter survivors of
the whole scene (the measured v9 rule; the legacy ObjectSelector's
"unique in scene OR in room" is available as ``unique_scope="scene_or_room"``),
the legacy pair rules (same room, dynamic AABB-gap band, longest-edge ratio
<= 5 and difference <= 4 m, 80 deg angular span) and the first 12 admissible
pairs in labels.json combination order.

``v10_relaxed``  duplicates allowed via referring expressions, documented
semantic exclusions (``semantic_exclusion_reason``), broader size range, the
v10 pair rules below, and a deterministic stratified budget of 60 pairs.

v10_relaxed thresholds (rationale)
----------------------------------
min_dim_component  0.02 m   drops only degenerate / zero-thickness boxes.
min_longest_edge   0.35 m   >= ~15 px at 3 m in the 256^2 H1 render (fx = 128).
max_dim_component  3.5 m    keeps L-sofas / long counters; larger boxes are
                            built-ins whose AABB centre is often off the object.
min_volume         0.02 m^3 v9's 0.1 dropped stools, plants, microwaves, TVs.
min_height         0.15 m   floor-flat items are seen at grazing angles from
                            1.5 m and the floor is only visible beyond ~2 m.
min_aspect_ratio   0.04     drops sheet-like boxes only (v9's 0.1 dropped TVs).
max_bottom_z       2.0 m    ceiling-mounted items (projectors, fans, AC units)
                            sit above the pitch-0 view band.
require_in_room    True     poses must be inside rooms; an object outside every
                            room (beyond 0.5 m) cannot anchor a same-room pair.
pair centre distance [0.6, 5.0] m  >= 0.6 m gives the 12 px Projective margin
                            from <= 6 m away; <= 5 m lets both fit the 90 deg
                            HFOV from ~3 m back (0.6 x distance).
pair size ratio    <= 6     (longest edges) keeps the smaller target >= ~20 px
                            when the larger one fills the view.
footprint overlap  < 0.5    of the smaller footprint, see ``pair_rejection``.
max_pairs_per_scene 60      stratified over centre-distance bands
                            [0.6,1.5) [1.5,2.5) [2.5,3.5) [3.5,5] with
                            category-pair / object diversity.

Doors and windows are excluded (as targets and as anchors): their boxes are
thin slabs centred on the wall line, so their centre often falls outside every
room polygon or in the wrong room; the runtime collision detector lists them
with walls (WALL_LABELS); window panes reconstruct poorly in Gaussian splats
and door leaves are in arbitrary open states, which hurts RGB identifiability;
and a scene usually has several of each, which cannot be told apart without
viewpoint words.  ``exclude_doors_windows=False`` re-enables them.

CLI
---
    python -m data_gen.interiorgs_v10.objects --scenes-root DIR [--out stats.json]
"""

from __future__ import annotations

import argparse
import bisect
import copy
import dataclasses
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .contract import Funnel, Obj
from .referring import (
    DEFAULT_REFERRING,
    ReferringConfig,
    Resolver,
    assign_referring_expressions,
    display_label,
    effective_room_index,
    same_category,
)

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
# Verbatim copy of ObjectSelectionConfig().blacklist (legacy v9 defaults).
# tests/test_objects.py asserts it still equals the legacy definition.
LEGACY_BLACKLIST: FrozenSet[str] = frozenset({
    "wall", "floor", "ceiling", "room",
    "carpet", "rug",
    "chandelier", "ceiling lamp", "spotlight", "lamp", "light",
    "downlights", "wall lamp", "table lamp", "strip light", "track light",
    "linear lamp", "decorative pendant",
    "other", "curtain", "bread", "cigar", "wine", "fresh food", "pen",
    "medicine bottle", "toiletries", "chocolate", "paper",
    "book", "boxed food", "bagged food", "medicine box",
    "vegetable", "fruit", "drinks", "canned food",
})

# v10 semantic exclusions.  Matching uses the label's tokens (split on
# non-alphanumerics) and its *head noun* (last token, ignoring a trailing
# "combination"/"set"/"group"), so "wall lamp" is a light but "wall cabinet"
# is a cabinet, and "mat" does not catch "mattress".
DOOR_WINDOW_TOKENS = frozenset({"door", "doors", "doorway", "doorframe", "window", "windows", "gate"})
STRUCTURAL_HEADS = frozenset({
    "wall", "walls", "floor", "floors", "ceiling", "ceilings", "room", "niche",
    "mold", "mould", "molding", "moulding", "skirting", "baseboard", "beam", "beams",
    "column", "columns", "pillar", "pillars", "stairs", "stair", "staircase", "steps",
    "railing", "railings", "handrail", "handrails", "balustrade", "partition",
    "partitions", "bulkhead", "opening", "hole", "platform", "pool",
})
STRUCTURAL_EXACT = frozenset({"floor mold", "suspended ceiling", "wall design combination"})
FLOOR_COVERING_HEADS = frozenset({"rug", "rugs", "carpet", "carpets", "mat", "mats",
                                  "doormat", "doormats", "placemat", "placemats"})
LIGHT_HEADS = frozenset({"light", "lights", "lamp", "lamps", "chandelier", "chandeliers",
                         "downlight", "downlights", "spotlight", "spotlights", "pendant",
                         "pendants", "sconce", "sconces", "lighting", "bulb"})
CURTAIN_HEADS = frozenset({"curtain", "curtains", "blind", "blinds", "drape", "drapes",
                           "screening", "valance"})
GENERIC_OR_WALL_FIXTURE_EXACT = frozenset({
    "other", "others", "unknown", "object", "objects", "misc", "miscellaneous", "item",
    "items", "thing", "things", "none", "null", "background",
    "decoration combination", "trappings", "landscape combination", "bathroom hardware",
    "switch panel", "switch", "socket", "outlet", "power outlet", "tuyere", "vent",
    "air vent", "thermostat", "smoke detector",
})
SOFT_FURNISHING_HEADS = frozenset({
    "blanket", "blankets", "quilt", "quilts", "towel", "towels", "scarf", "scarves",
    "clothes", "clothing", "cushion", "cushions", "pillow", "pillows", "bedding",
    "duvet", "sheet", "sheets", "yarn", "net",
})
SMALL_CLUTTER_EXACT = frozenset({
    "bread", "cigar", "wine", "fresh food", "pen", "medicine bottle", "toiletries",
    "chocolate", "paper", "book", "books", "boxed food", "bagged food", "medicine box",
    "vegetable", "fruit", "drinks", "canned food", "tooth brush", "toothbrush",
})
# Free-standing lamps are furniture-scale and were frequent v9 targets
# (only mounted fixtures and table lamps were blacklisted).
DEFAULT_KEEP_LABELS = frozenset({"floor lamp"})

_HEAD_SUFFIXES = frozenset({"combination", "combinations", "set", "group"})


def _tokens(label: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (label or "").lower()) if t]


def _head(tokens: Sequence[str]) -> str:
    toks = list(tokens)
    while len(toks) > 1 and toks[-1] in _HEAD_SUFFIXES:
        toks.pop()
    return toks[-1] if toks else ""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ObjectConfig:
    """All thresholds of the object / single / pair stages.  Defaults are the
    ``v10_relaxed`` values; ``None`` disables a numeric filter."""

    name: str = "v10_relaxed"
    # semantic ------------------------------------------------------------
    semantic_rules: str = "v10"                  # "v10" | "v9_blacklist" (exact labels only)
    blacklist: FrozenSet[str] = LEGACY_BLACKLIST  # exact labels, applied by both rule sets
    keep_labels: FrozenSet[str] = DEFAULT_KEEP_LABELS
    exclude_doors_windows: bool = True
    # single-object geometry ---------------------------------------------
    min_dim_component: Optional[float] = 0.02
    min_longest_edge: Optional[float] = 0.35
    max_dim_component: Optional[float] = 3.5
    min_volume: Optional[float] = 0.02
    min_height: Optional[float] = 0.15
    min_aspect_ratio: Optional[float] = 0.04
    max_bottom_z: Optional[float] = 2.0
    require_in_room: bool = True
    # identity -------------------------------------------------------------
    unique_scope: str = "none"                   # "none" | "scene" | "scene_or_room"
    ref_policy: str = "spatial"                  # "spatial" | "v9_label"
    referring: ReferringConfig = DEFAULT_REFERRING
    room_source: str = "scene"                   # "scene" | "raw_structure" (v9 rules only)
    # pairs ----------------------------------------------------------------
    pair_rules: str = "v10"                      # "v10" | "legacy"
    pair_room_rule: str = "same"                 # "same" | "any"
    pair_allow_adjacent_rooms: bool = False
    adjacent_room_max_gap_m: float = 0.35
    pair_center_dist_min_m: float = 0.6
    pair_center_dist_max_m: float = 5.0
    pair_size_ratio_max: float = 6.0
    pair_exclude_same_category: bool = True
    pair_max_footprint_overlap: Optional[float] = 0.5
    pair_selection: str = "stratified"           # "stratified" | "first"
    max_pairs_per_scene: Optional[int] = 60
    distance_band_edges_m: Tuple[float, ...] = (1.5, 2.5, 3.5)
    # legacy pair rules (ObjectSelectionConfig defaults) ---------------------
    legacy_require_same_room: bool = True
    legacy_min_pair_dist: float = 0.1
    legacy_max_pair_dist: float = 6.0
    legacy_dyn_min_mult: float = 0.1
    legacy_dyn_max_mult: float = 5.0
    legacy_max_pair_dim_ratio: float = 5.0
    legacy_max_pair_dim_diff: float = 4.0
    legacy_pair_fov_deg: float = 90.0
    legacy_pair_fov_margin_deg: float = 5.0
    legacy_min_viewing_distance_mult: float = 0.8
    # singles ----------------------------------------------------------------
    single_min_longest_edge: Optional[float] = None
    max_singles_per_scene: Optional[int] = 40
    max_singles_per_category: Optional[int] = 3

    def to_json(self) -> Dict[str, Any]:
        def conv(v: Any) -> Any:
            if isinstance(v, (set, frozenset)):
                return sorted(v)
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v
        return conv(dataclasses.asdict(self))


V10_RELAXED = ObjectConfig()
V9_STRICT = ObjectConfig(
    name="v9_strict",
    semantic_rules="v9_blacklist",
    exclude_doors_windows=False,
    keep_labels=frozenset(),
    min_dim_component=0.1,
    min_longest_edge=None,
    max_dim_component=3.0,
    min_volume=0.1,
    min_height=None,
    min_aspect_ratio=0.1,
    max_bottom_z=None,
    require_in_room=False,
    unique_scope="scene",
    ref_policy="v9_label",
    pair_rules="legacy",
    pair_selection="first",
    max_pairs_per_scene=12,
    max_singles_per_scene=None,
    max_singles_per_category=None,
)
# Diagnostic only (refs are NOT unique): v9 with the uniqueness rule dropped.
V9_NO_UNIQUE = dataclasses.replace(V9_STRICT, name="v9_no_unique", unique_scope="none")

# 2026-10-14: referring rules v2 (names only when needed, rooms inferred from their furniture, references
# large, unique in their room and apart from the target; referring.py) and two labels the pilot showed to be
# unusable as targets or references: "wall panels" is part of the wall, "edge combination" names unrelated
# objects (a plant, a sofa corner).
V10_R5 = dataclasses.replace(
    V10_RELAXED, name="v10_r5", referring=ReferringConfig(rules="v2"),
    blacklist=LEGACY_BLACKLIST | frozenset({"wall panels", "wall panel", "edge combination"}))

# 2026-10-14, round 6: referring rules v3 (a target that is the only one of its kind in the start's room is
# just "the X"; stricter room types; referring.py module docstring).
V10_R6 = dataclasses.replace(V10_R5, name="v10_r6", referring=ReferringConfig(rules="v3"))

PRESETS: Dict[str, ObjectConfig] = {c.name: c for c in (V9_STRICT, V10_RELAXED, V9_NO_UNIQUE, V10_R5, V10_R6)}

PresetLike = Union[str, ObjectConfig]


def get_preset(preset: PresetLike = "v10_relaxed", **overrides: Any) -> ObjectConfig:
    if isinstance(preset, ObjectConfig):
        cfg = preset
    else:
        try:
            cfg = PRESETS[str(preset)]
        except KeyError:
            raise ValueError(f"unknown object preset {preset!r}; known: {sorted(PRESETS)}") from None
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


# ---------------------------------------------------------------------------
# Single-object rules
# ---------------------------------------------------------------------------
def semantic_exclusion_reason(label: str, cfg: PresetLike = "v10_relaxed") -> Optional[str]:
    """Stable reason string if the label is never a target/anchor, else None."""
    cfg = get_preset(cfg)
    lab = (label or "").strip().lower()
    if cfg.semantic_rules == "v9_blacklist":
        return "blacklist" if lab in cfg.blacklist else None
    toks = _tokens(lab)
    norm = " ".join(toks)
    if not norm:
        return "empty_label"
    if lab in cfg.keep_labels or norm in cfg.keep_labels:
        return None
    head = _head(toks)
    if cfg.exclude_doors_windows and any(t in DOOR_WINDOW_TOKENS for t in toks):
        return "door_window"
    if norm in STRUCTURAL_EXACT or head in STRUCTURAL_HEADS:
        return "structural"
    if head in FLOOR_COVERING_HEADS:
        return "floor_covering"
    if head in LIGHT_HEADS:
        return "light_fixture"
    if head in CURTAIN_HEADS:
        return "curtain_blind"
    if norm in GENERIC_OR_WALL_FIXTURE_EXACT:
        return "generic_or_wall_fixture"
    if head in SOFT_FURNISHING_HEADS:
        return "soft_furnishing"
    if norm in SMALL_CLUTTER_EXACT:
        return "small_clutter"
    if lab in cfg.blacklist or norm in cfg.blacklist:
        return "blacklist"
    return None


def _finite(o: Obj) -> bool:
    return bool(np.all(np.isfinite(o.aabb_min)) and np.all(np.isfinite(o.aabb_max)))


def single_object_rejection(obj: Obj, cfg: PresetLike, room_index: Optional[int] = None,
                            scene_has_rooms: bool = False) -> Optional[str]:
    """First failing single-object rule (legacy order: semantic, dims, volume,
    aspect), or None.  ``room_index`` is the effective room used by
    ``require_in_room``."""
    cfg = get_preset(cfg)
    if not _finite(obj):
        return "non_finite_bbox"
    sem = semantic_exclusion_reason(obj.label, cfg)
    if sem is not None:
        return sem
    d = np.asarray(obj.dims, dtype=float)
    if cfg.min_dim_component is not None and bool(np.any(d < cfg.min_dim_component)):
        return "dim_too_small"
    if cfg.min_longest_edge is not None and float(d.max()) < cfg.min_longest_edge:
        return "longest_edge_too_small"
    if cfg.max_dim_component is not None and bool(np.any(d > cfg.max_dim_component)):
        return "dim_too_large"
    if cfg.min_volume is not None and float(np.prod(d)) < cfg.min_volume:
        return "volume_too_small"
    if cfg.min_height is not None and float(d[2]) < cfg.min_height:
        return "too_low"
    if cfg.min_aspect_ratio is not None and float(d.min()) / max(float(d.max()), 1e-12) < cfg.min_aspect_ratio:
        return "too_flat"
    if cfg.max_bottom_z is not None and float(obj.aabb_min[2]) > cfg.max_bottom_z:
        return "ceiling_mounted"
    if cfg.require_in_room and scene_has_rooms and room_index is None:
        return "outside_rooms"
    return None


def _raw_structure_room_indices(scene: Any, objs: Iterable[Obj]) -> Dict[str, Optional[int]]:
    """Legacy ObjectSelector room assignment: raw structure.json polygons,
    WITHOUT the per-scene Y-sign calibration."""
    from .scene import point_in_polygon

    polys: List[np.ndarray] = []
    path = Path(getattr(scene, "scene_dir", "") or "") / "structure.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    items = data.get("rooms", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for room in items:
        if not isinstance(room, dict):
            continue
        for key in ("profile", "polygon", "floor", "boundary"):
            if key in room:
                poly = room[key]
                if isinstance(poly, list) and len(poly) >= 3:
                    polys.append(np.asarray([[float(p[0]), float(p[1])] for p in poly], dtype=float))
                break
    out: Dict[str, Optional[int]] = {}
    for o in objs:
        c = o.center
        out[o.id] = next((i for i, poly in enumerate(polys)
                          if point_in_polygon(float(c[0]), float(c[1]), poly)), None)
    return out


def _room_map(scene: Any, objs: Sequence[Obj], cfg: ObjectConfig) -> Dict[str, Optional[int]]:
    if cfg.room_source == "raw_structure":
        return _raw_structure_room_indices(scene, objs)
    if cfg.room_source != "scene":
        raise ValueError(f"unknown room_source {cfg.room_source!r}")
    if cfg.pair_rules == "legacy":  # legacy: centre-in-polygon only, no fallback
        return {o.id: (None if o.room_index is None else int(o.room_index)) for o in objs}
    rooms = list(getattr(scene, "rooms", []) or [])
    return {o.id: effective_room_index(rooms, o, cfg.referring.room_fallback_max_dist_m) for o in objs}


def _runtime_ignored_labels() -> Optional[FrozenSet[str]]:
    try:
        from vagen.envs.active_spatial.collision_detector import CollisionDetector
        return frozenset(CollisionDetector.IGNORED_LABELS)
    except Exception:  # pragma: no cover - diagnostic only
        return None


def candidate_objects(scene: Any, preset: PresetLike = "v10_relaxed") -> Tuple[List[Obj], Funnel]:
    """Target-eligible objects of ``scene`` with a unique referring expression.

    Funnel stage ``objects``: ``input``, ``reject_<reason>`` per exclusion,
    ``pass_single_filters`` (checkpoint), ``kept``.  Referring stages are
    merged in (see referring.py).  ``objects.diag`` holds non-rejecting notes.
    """
    cfg = get_preset(preset)
    funnel = Funnel()
    S = "objects"
    all_objs = list(scene.objects)
    rooms = list(getattr(scene, "rooms", []) or [])
    funnel.add(S, "input", len(all_objs))
    room_map = _room_map(scene, all_objs, cfg)
    # no-task list (configs/no_task_objects_*.json): still collide and occlude, never a target or reference
    no_task = {str(i) for i in (getattr(scene, "no_task_object_ids", None) or ())}

    passed: List[Obj] = []
    for o in all_objs:
        if o.id in no_task:
            funnel.add(S, "reject_no_task_list")
            continue
        reason = single_object_rejection(o, cfg, room_map.get(o.id), bool(rooms))
        if reason is not None:
            funnel.add(S, "reject_" + reason)
            continue
        passed.append(o)
    funnel.add(S, "pass_single_filters", len(passed))

    if cfg.unique_scope == "scene":
        counts = Counter(o.label for o in passed)
        unique = [o for o in passed if counts[o.label] == 1]
        funnel.add(S, "reject_not_unique_in_scene", len(passed) - len(unique))
        passed = unique
    elif cfg.unique_scope == "scene_or_room":
        counts = Counter(o.label for o in passed)
        room_counts = Counter((room_map.get(o.id), o.label) for o in passed)
        unique = [o for o in passed if counts[o.label] == 1 or room_counts[(room_map.get(o.id), o.label)] == 1]
        funnel.add(S, "reject_not_unique_in_scene_or_room", len(passed) - len(unique))
        passed = unique
    elif cfg.unique_scope != "none":
        raise ValueError(f"unknown unique_scope {cfg.unique_scope!r}")

    out: List[Obj] = []
    for o in passed:
        c = copy.copy(o)
        c.ref, c.ref_method, c.ref_extra = None, None, None
        out.append(c)

    if cfg.ref_policy == "v9_label":
        resolver = Resolver(scene, all_objs, cfg.referring) if out else None
        for c in out:
            c.ref, c.ref_method = f"the {display_label(c.label)}", "v9_label"
            funnel.add("referring", "v9_label")
            if resolver is not None and len(resolver(c.ref)) != 1:
                # diagnostic only: v9 checked raw-label uniqueness among filter
                # survivors, not what a reader sees (all instances, synonyms)
                funnel.add("referring", "v9_ref_ambiguous_vs_all_instances")
    elif cfg.ref_policy == "spatial":
        funnel.merge(assign_referring_expressions(
            scene, out, cfg.referring, pool=all_objs,
            anchor_ok=lambda a: semantic_exclusion_reason(a.label, cfg) is None,
            reference_ok=lambda a: a.id not in no_task))
    else:
        raise ValueError(f"unknown ref_policy {cfg.ref_policy!r}")

    kept = [c for c in out if c.ref]
    if len(kept) != len(out):
        funnel.add(S, "reject_no_ref", len(out) - len(kept))
    funnel.add(S, "kept", len(kept))

    ignored = _runtime_ignored_labels()
    if ignored:
        n = sum(1 for c in kept if any(ig in c.label for ig in ignored))
        if n:
            funnel.add("objects.diag", "kept_but_runtime_collision_ignores_label", n)
    return kept, funnel


# ---------------------------------------------------------------------------
# Singles
# ---------------------------------------------------------------------------
def _hkey(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def select_singles(scene: Any, objs: Sequence[Obj], preset: PresetLike = "v10_relaxed",
                   seed: int = 0) -> Tuple[List[Obj], Funnel]:
    """Single-object targets: candidates (with ref), optional size floor, then
    a deterministic round-robin over display categories with a per-category
    cap and a per-scene budget (v9_strict: all candidates, no caps)."""
    cfg = get_preset(preset)
    funnel = Funnel()
    S = "singles"
    funnel.add(S, "input", len(objs))
    pool: List[Obj] = []
    for o in objs:
        if not o.ref:
            funnel.add(S, "reject_no_ref")
        elif cfg.single_min_longest_edge is not None and float(np.max(o.dims)) < cfg.single_min_longest_edge:
            funnel.add(S, "reject_too_small_for_single")
        else:
            pool.append(o)
    if cfg.max_singles_per_scene is None and cfg.max_singles_per_category is None:
        funnel.add(S, "selected", len(pool))
        return list(pool), funnel

    scene_id = getattr(scene, "scene_id", "")
    groups: Dict[str, List[Obj]] = {}
    for o in pool:
        groups.setdefault(display_label(o.label).lower(), []).append(o)
    order = sorted(groups, key=lambda k: _hkey(seed, scene_id, "cat", k))
    for k in order:
        groups[k].sort(key=lambda o: _hkey(seed, scene_id, "obj", o.id))
    cap = cfg.max_singles_per_category
    budget = cfg.max_singles_per_scene
    selected: List[Obj] = []
    rnd = 0
    while True:
        progressed = False
        for k in order:
            if rnd < len(groups[k]) and (cap is None or rnd < cap):
                if budget is not None and len(selected) >= budget:
                    break
                selected.append(groups[k][rnd])
                progressed = True
        rnd += 1
        if not progressed or (budget is not None and len(selected) >= budget):
            break
    capped = sum(max(0, len(v) - cap) for v in groups.values()) if cap is not None else 0
    funnel.add(S, "selected", len(selected))
    if capped:
        funnel.add(S, "reject_category_cap", capped)
    over = len(pool) - capped - len(selected)
    if over > 0:
        funnel.add(S, "reject_over_budget", over)
    return selected, funnel


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------
def aabb_min_distance(a: Obj, b: Obj) -> float:
    """3D gap between two AABBs (legacy ObjectSelector.aabb_min_distance)."""
    d = np.maximum(0.0, np.maximum(a.aabb_min - b.aabb_max, b.aabb_min - a.aabb_max))
    return float(np.sqrt(float(d @ d)))


def center_distance_xy(a: Obj, b: Obj) -> float:
    return float(np.linalg.norm(a.center[:2] - b.center[:2]))


def footprint_overlap_fraction(a: Obj, b: Obj) -> float:
    """Area of the XY footprint intersection over the smaller footprint."""
    w = min(float(a.aabb_max[0]), float(b.aabb_max[0])) - max(float(a.aabb_min[0]), float(b.aabb_min[0]))
    h = min(float(a.aabb_max[1]), float(b.aabb_max[1])) - max(float(a.aabb_min[1]), float(b.aabb_min[1]))
    if w <= 0 or h <= 0:
        return 0.0
    area_a = float(a.dims[0] * a.dims[1])
    area_b = float(b.dims[0] * b.dims[1])
    return float(w * h / max(min(area_a, area_b), 1e-9))


def _legacy_pair_rejection(a: Obj, b: Obj, cfg: ObjectConfig,
                           ra: Optional[int], rb: Optional[int]) -> Optional[str]:
    """ObjectSelector.filter_object_pair with ObjectSelectionConfig fields."""
    if cfg.legacy_require_same_room and ra is not None and rb is not None and ra != rb:
        return "different_rooms"
    dist = aabb_min_distance(a, b)
    max_a, max_b = float(np.max(a.dims)), float(np.max(b.dims))
    avg = (max_a + max_b) / 2.0
    dyn_min = max(cfg.legacy_min_pair_dist, cfg.legacy_dyn_min_mult * avg)
    dyn_max = min(cfg.legacy_max_pair_dist, cfg.legacy_dyn_max_mult * avg)
    if dist < dyn_min:
        return "too_close"
    if dist > dyn_max:
        return "too_far"
    if max(max_a, max_b) / (min(max_a, max_b) + 1e-9) > cfg.legacy_max_pair_dim_ratio:
        return "size_ratio"
    if abs(max_a - max_b) > cfg.legacy_max_pair_dim_diff:
        return "size_diff"
    ha, hb = float(a.dims[2]), float(b.dims[2])
    if abs(ha - hb) > 2 * max(ha, hb):
        return "height_diff"
    d_ab = center_distance_xy(a, b)
    min_view = max(max(max_a, max_b) * cfg.legacy_min_viewing_distance_mult, 0.8)
    span = math.degrees(2 * math.atan2(d_ab / 2, min_view))
    if span > cfg.legacy_pair_fov_deg - 2 * cfg.legacy_pair_fov_margin_deg:
        return "angular_span_too_large"
    return None


def _v10_pair_rejection(a: Obj, b: Obj, cfg: ObjectConfig, ra: Optional[int], rb: Optional[int],
                        has_rooms: bool, adjacent: FrozenSet[FrozenSet[int]]) -> Optional[str]:
    if cfg.pair_exclude_same_category and same_category(a.label, b.label):
        return "same_category"
    if cfg.pair_room_rule == "same" and has_rooms:
        if ra is None or rb is None:
            return "room_unknown"
        if ra != rb and not (cfg.pair_allow_adjacent_rooms and frozenset((ra, rb)) in adjacent):
            return "different_rooms"
    elif cfg.pair_room_rule not in ("same", "any"):
        raise ValueError(f"unknown pair_room_rule {cfg.pair_room_rule!r}")
    if cfg.pair_max_footprint_overlap is not None and \
            footprint_overlap_fraction(a, b) >= cfg.pair_max_footprint_overlap:
        return "stacked_or_contained"
    dc = center_distance_xy(a, b)
    if dc < cfg.pair_center_dist_min_m:
        return "too_close"
    if dc > cfg.pair_center_dist_max_m:
        return "too_far"
    la, lb = float(np.max(a.dims)), float(np.max(b.dims))
    if max(la, lb) / max(min(la, lb), 1e-9) > cfg.pair_size_ratio_max:
        return "size_ratio"
    return None


def pair_rejection(a: Obj, b: Obj, cfg: PresetLike, ra: Optional[int] = None, rb: Optional[int] = None,
                   has_rooms: bool = True,
                   adjacent: FrozenSet[FrozenSet[int]] = frozenset()) -> Optional[str]:
    """First failing pair rule or None.

    v10 rules, in order: ``same_category`` (two instances of one category make
    the instruction hinge entirely on the referring expression and the two
    look identical in the frame); ``room_unknown`` / ``different_rooms``
    (default same room; adjacent rooms optional); ``stacked_or_contained`` -
    the smaller XY footprint lies >= 50% inside the larger one (TV on its
    cabinet, vase on a counter, chair tucked under a table).  Rejected only
    because the relation degenerates: the canonical Projective gate compares
    projected bbox-centre u with a 12 px margin, and for such pairs the
    centre offset comes only from parallax inside the larger silhouette, so
    success needs grazing views and "A appears left of B" is not what a person
    reads off the image (A is *on/in* B); FOV inclusion is trivially implied.
    Partial overlaps (< 50%) keep separable centres and are retained.  Then
    the centre-distance band and the longest-edge ratio cap.
    """
    cfg = get_preset(cfg)
    if cfg.pair_rules == "legacy":
        return _legacy_pair_rejection(a, b, cfg, ra, rb)
    if cfg.pair_rules == "v10":
        return _v10_pair_rejection(a, b, cfg, ra, rb, has_rooms, adjacent)
    raise ValueError(f"unknown pair_rules {cfg.pair_rules!r}")


def _polygon_gap(p: np.ndarray, q: np.ndarray) -> float:
    from .referring import _point_polygon_distance

    best = math.inf
    for (x, y) in np.asarray(p, dtype=float):
        best = min(best, _point_polygon_distance(float(x), float(y), q))
    for (x, y) in np.asarray(q, dtype=float):
        best = min(best, _point_polygon_distance(float(x), float(y), p))
    return best


def adjacent_rooms(scene: Any, max_gap: float) -> FrozenSet[FrozenSet[int]]:
    """Room index pairs whose boundaries come within ``max_gap`` (a wall)."""
    rooms = list(getattr(scene, "rooms", []) or [])
    out = set()
    for r1, r2 in combinations(rooms, 2):
        if _polygon_gap(r1.polygon, r2.polygon) <= max_gap:
            out.add(frozenset((int(r1.index), int(r2.index))))
    return frozenset(out)


def _stratified(pairs: Sequence[Tuple[Obj, Obj]], cfg: ObjectConfig, budget: int,
                seed: int, scene_id: str) -> List[Tuple[Obj, Obj]]:
    edges = sorted(cfg.distance_band_edges_m)
    bands: List[List[Tuple[str, Tuple[Obj, Obj]]]] = [[] for _ in range(len(edges) + 1)]
    for a, b in pairs:
        ids = sorted((a.id, b.id))
        band = bisect.bisect_right(edges, center_distance_xy(a, b))
        bands[band].append((_hkey(seed, scene_id, ids[0], ids[1]), (a, b)))
    for band in bands:
        band.sort(key=lambda kp: kp[0])
    cat_use: Counter = Counter()
    obj_use: Counter = Counter()
    lab_use: Counter = Counter()
    selected: List[Tuple[Obj, Obj]] = []

    def cats(p: Tuple[Obj, Obj]) -> Tuple[str, str]:
        x, y = sorted((display_label(p[0].label).lower(), display_label(p[1].label).lower()))
        return x, y

    while len(selected) < budget and any(bands):
        for band in bands:
            if not band or len(selected) >= budget:
                continue
            best = min(range(len(band)), key=lambda i: (
                cat_use[cats(band[i][1])],
                max(obj_use[band[i][1][0].id], obj_use[band[i][1][1].id]),
                sum(lab_use[c] for c in cats(band[i][1])),
                band[i][0]))
            _, p = band.pop(best)
            selected.append(p)
            cat_use[cats(p)] += 1
            obj_use[p[0].id] += 1
            obj_use[p[1].id] += 1
            for c in cats(p):
                lab_use[c] += 1
    return selected


def select_pairs(scene: Any, objs: Sequence[Obj], preset: PresetLike = "v10_relaxed",
                 max_pairs: Optional[int] = None, seed: int = 0) -> Tuple[List[Tuple[Obj, Obj]], Funnel]:
    """Unordered admissible pairs (each once, in candidate order); callers
    expand to (A,B)/(B,A) with ``expand_ordered`` where order matters.

    ``max_pairs`` overrides the preset budget (``None`` = preset value).
    v9 ("first"): the first N admissible pairs in combination order (legacy).
    v10 ("stratified"): round-robin over centre-distance bands, inside a band
    the pair whose category pair / objects / labels were used least, ties
    broken by sha256(seed, scene_id, ids) - deterministic and independent of
    input order.  Funnel stage ``pairs``: ``input_objects``,
    ``input_combinations``, ``reject_<reason>``, ``admissible``, ``selected``,
    ``over_budget``.
    """
    cfg = get_preset(preset)
    funnel = Funnel()
    S = "pairs"
    with_ref = [o for o in objs if o.ref]
    funnel.add(S, "input_objects", len(objs))
    if len(with_ref) != len(objs):
        funnel.add(S, "reject_object_without_ref", len(objs) - len(with_ref))
    n = len(with_ref)
    funnel.add(S, "input_combinations", n * (n - 1) // 2)
    room_map = _room_map(scene, with_ref, cfg)
    has_rooms = bool(getattr(scene, "rooms", None))
    adjacent = adjacent_rooms(scene, cfg.adjacent_room_max_gap_m) \
        if (cfg.pair_rules == "v10" and cfg.pair_allow_adjacent_rooms) else frozenset()

    admissible: List[Tuple[Obj, Obj]] = []
    for a, b in combinations(with_ref, 2):
        reason = pair_rejection(a, b, cfg, room_map.get(a.id), room_map.get(b.id), has_rooms, adjacent)
        if reason is not None:
            funnel.add(S, "reject_" + reason)
        else:
            admissible.append((a, b))
    funnel.add(S, "admissible", len(admissible))

    budget = cfg.max_pairs_per_scene if max_pairs is None else max_pairs
    if budget is None or budget >= len(admissible):
        selected = list(admissible)
    elif cfg.pair_selection == "first":
        selected = admissible[: max(0, int(budget))]
    elif cfg.pair_selection == "stratified":
        selected = _stratified(admissible, cfg, max(0, int(budget)), seed, getattr(scene, "scene_id", ""))
    else:
        raise ValueError(f"unknown pair_selection {cfg.pair_selection!r}")
    funnel.add(S, "selected", len(selected))
    if len(selected) < len(admissible):
        funnel.add(S, "over_budget", len(admissible) - len(selected))
    return selected, funnel


def expand_ordered(pairs: Iterable[Tuple[Obj, Obj]]) -> List[Tuple[Obj, Obj]]:
    """(A,B) -> [(A,B), (B,A)] for order-sensitive families (Projective)."""
    out: List[Tuple[Obj, Obj]] = []
    for a, b in pairs:
        out.append((a, b))
        out.append((b, a))
    return out


# ---------------------------------------------------------------------------
# CLI: per-scene statistics under several presets
# ---------------------------------------------------------------------------
def scene_stats(scene: Any, presets: Sequence[str], seed: int = 0) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"convention": _jsonable(getattr(scene, "convention", {}) or {}),
                           "n_rooms": len(getattr(scene, "rooms", []) or []),
                           "n_labels": len(scene.objects), "presets": {}}
    for name in presets:
        cfg = get_preset(name)
        objs, f_obj = candidate_objects(scene, cfg)
        singles, f_single = select_singles(scene, objs, cfg, seed=seed)
        pairs, f_pair = select_pairs(scene, objs, cfg, seed=seed)
        funnel = Funnel()
        for f in (f_obj, f_single, f_pair):
            funnel.merge(f)
        rec["presets"][name] = {
            "objects": len(objs),
            "pass_single_filters": f_obj.counts.get("objects", {}).get("pass_single_filters", 0),
            "singles": len(singles),
            "pairs_admissible": f_pair.counts.get("pairs", {}).get("admissible", 0),
            "pairs": len(pairs),
            "ref_methods": dict(Counter(o.ref_method for o in objs)),
            "funnel": funnel.to_dict(),
        }
    return rec


def _jsonable(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except TypeError:
        return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def summarize(per_scene: Dict[str, Dict[str, Any]], presets: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ok = {k: v for k, v in per_scene.items() if "error" not in v}
    for name in presets:
        tot = Counter()
        funnel = Funnel()
        methods = Counter()
        lt2 = no_pair = no_pair_lt2 = 0
        for rec in ok.values():
            p = rec["presets"][name]
            for key in ("objects", "pass_single_filters", "singles", "pairs_admissible", "pairs"):
                tot[key] += int(p[key])
            for stage, reasons in p["funnel"].items():
                for reason, n in reasons.items():
                    funnel.add(stage, reason, n)
            methods.update(p["ref_methods"])
            if p["objects"] < 2:
                lt2 += 1
            if p["pairs_admissible"] == 0:
                no_pair += 1
                no_pair_lt2 += int(p["objects"] < 2)
        out[name] = {
            "scenes": len(ok),
            "totals": dict(tot),
            "scenes_with_lt2_objects": lt2,
            "scenes_without_admissible_pair": no_pair,
            "scenes_without_admissible_pair_lt2_objects": no_pair_lt2,
            "scenes_without_admissible_pair_pair_rules": no_pair - no_pair_lt2,
            "scenes_with_ge1_single": sum(1 for r in ok.values() if r["presets"][name]["singles"] > 0),
            "ref_methods": dict(methods),
            "funnel": funnel.to_dict(),
            "config": get_preset(name).to_json(),
        }
    out["scenes_total"] = len(per_scene)
    out["scenes_failed_to_load"] = len(per_scene) - len(ok)
    out["convention_status"] = dict(Counter(str((r.get("convention") or {}).get("status"))
                                            for r in ok.values()))
    return out


def _iter_scene_dirs(root: Path, only: Optional[Sequence[str]], limit: Optional[int]) -> List[Path]:
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "labels.json").is_file())
    if only:
        wanted = set(only)
        dirs = [p for p in dirs if p.name in wanted]
    return dirs[:limit] if limit else dirs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes-root", required=True, type=Path,
                    help="directory whose sub-directories hold labels.json (+ structure.json)")
    ap.add_argument("--out", type=Path, default=None, help="write per-scene + summary JSON here")
    ap.add_argument("--presets", default="v9_strict,v10_relaxed",
                    help=f"comma separated, known: {','.join(sorted(PRESETS))}")
    ap.add_argument("--scene", action="append", default=None, help="restrict to scene id (repeatable)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from .scene import load_scene

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    for p in presets:
        get_preset(p)
    per_scene: Dict[str, Dict[str, Any]] = {}
    dirs = _iter_scene_dirs(args.scenes_root, args.scene, args.limit)
    t0 = time.time()
    for k, d in enumerate(dirs):
        try:
            scene = load_scene(d, d.name)
            per_scene[d.name] = scene_stats(scene, presets, seed=args.seed)
        except Exception as exc:  # keep going; failures are counted
            per_scene[d.name] = {"error": f"{type(exc).__name__}: {exc}"}
        if (k + 1) % 50 == 0:
            print(f"[objects] {k + 1}/{len(dirs)} scenes, {time.time() - t0:.0f}s", file=sys.stderr)
    summary = summarize(per_scene, presets)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "per_scene": per_scene}, indent=1, sort_keys=True))
    print(f"scenes: {summary['scenes_total']} (load failures {summary['scenes_failed_to_load']})")
    for name in presets:
        s = summary[name]
        t = s["totals"]
        print(f"[{name}] pass_single_filters={t.get('pass_single_filters', 0)} objects={t.get('objects', 0)} "
              f"singles={t.get('singles', 0)} pairs_admissible={t.get('pairs_admissible', 0)} "
              f"pairs_selected={t.get('pairs', 0)} | scenes <2 objects={s['scenes_with_lt2_objects']} "
              f"no admissible pair={s['scenes_without_admissible_pair']} "
              f"(<2 objects {s['scenes_without_admissible_pair_lt2_objects']}, "
              f"pair rules {s['scenes_without_admissible_pair_pair_rules']})")
        print(f"    ref methods: {s['ref_methods']}")
        print(f"    object rejections: { {k: v for k, v in s['funnel'].get('objects', {}).items() if k.startswith('reject_')} }")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
