"""Viewpoint-free referring expressions for InteriorGS objects (v10).

User decision (SPEC.md, binding #2): duplicate categories are allowed and are
disambiguated with spatial referring expressions; an instance is excluded only
when no unambiguous expression exists.  Tasks are about an agent MOVING its
camera, so expressions never use viewer-relative words (left / right / front /
behind / near / far ...): such words change meaning as the agent moves.

Semantics (shared by ``assign_referring_expressions`` and ``resolve``)
---------------------------------------------------------------------
Every object has a *display label* (``display_label(obj.label)``); ``Obj.label``
itself stays raw.  A category phrase X *matches* an object o iff
``disp(o) == X`` or ``disp(o)`` ends with ``" " + X`` (case-insensitive), i.e.
a generic noun also covers its compounds: "the table" matches dining / side /
coffee tables, "the cabinet" matches TV cabinets.  Matching is evaluated over
ALL labels.json instances of the scene (``scene.objects``), including those
dropped by size filters, because a person looking at the image sees them too.

Expressions, tried in priority order for each instance:

1. ``"the X"``                       X matches exactly one object in the scene.
2. ``"the X in|on the R"``           R is a coarse, visually recognisable room
                                    type (bedroom, kitchen, ...); exactly one
                                    X-object lies in a room of type R.  Rooms
                                    whose name maps to no known type are
                                    unusable ("room_unnamed").
3. ``"the X next to the A"``         A is an anchor whose phrase matches exactly
                                    one object and is not the same category as
                                    X.  With d(o) = XY box gap(o, A) +
                                    ``anchor_dist_floor_m`` over all X-objects:
                                    the instance with the smallest d must have
                                    raw gap <= ``anchor_max_gap_m`` and every
                                    other X-object must have d >= ratio * d_min.
                                    ``resolve`` returns {o : d(o) < ratio*d_min}.
                                    Anchors are tried nearest first; generation
                                    also requires anchor and target to share a
                                    room (when both are known).
4. ``"the larger|smaller X"`` /      exactly two X-objects in the scene with
   ``"the taller|shorter X"``        volume ratio (else height ratio) >= 1.5;
                                    by default both must share a room so the
                                    comparison is visible.

Otherwise ``obj.ref = None`` and the reason is counted in the Funnel.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contract import Funnel, Obj

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# Viewer-relative words that must never appear in an expression.
VIEWPOINT_WORDS = frozenset({
    "left", "right", "leftmost", "rightmost", "front", "behind", "back", "ahead",
    "foreground", "background", "near", "nearer", "nearest", "far", "farther",
    "farthest", "further", "closer", "closest", "facing", "visible", "hidden",
})

# Raw InteriorGS label -> natural display text.  Merges true synonyms so that
# uniqueness is judged on what a reader would call the object (a "teatable"
# and a "tea table" are both "the coffee table").
DISPLAY_LABELS: Dict[str, str] = {
    "teatable": "coffee table",
    "tea table": "coffee table",
    "tea table combination": "coffee table",
    "multi person sofa": "sofa",
    "washing_machine": "washing machine",
    "smoke lampblack machine": "range hood",
    "hood": "range hood",
    "tv": "TV",
    "tv cabinet": "TV cabinet",
    "tv cabinet combination": "TV cabinet",
    "tv stand": "TV stand",
    "basin cabinet": "vanity cabinet",
    "basin": "washbasin",
    "hand sink combination": "washbasin",
    "vertical air conditioning": "standing air conditioner",
    "sideboards": "sideboard",
    "laundry cabinets": "laundry cabinet",
    "bathroom cabinets": "bathroom cabinet",
    "gas stoves": "gas stove",
    "green plants": "plant",
    "green plant combination": "plant",
    "landscape plants": "plant",
    "fig": "plant",
    "toy_animals": "stuffed animal",
    "coffee_maker": "coffee maker",
    "tooth_brush": "toothbrush",
    "bed combination": "bed",
    "functional bed": "bed",
    "bookcase": "bookshelf",
    "bookcase combination": "bookshelf",
    "wardrobe combination": "wardrobe",
    "display cabinet combination": "display cabinet",
    "storage rack combination": "storage rack",
    "bathtub combination": "bathtub",
    "makeup table": "dressing table",
    "makeup table combination": "dressing table",
    "study desk and bookshelf combination": "desk with bookshelf",
    "the mahjong table": "mahjong table",
    "squatting toilets": "squat toilet",
    "cooking bench": "kitchen counter",
    "integrated cooktop": "cooktop",
    "microwave oven": "microwave",
    "pet nest": "pet bed",
    "sterilizer": "dish sterilizer",
    "landscape stone": "decorative stone",
}

# Coarse, visually recognisable room types.  Keys are searched (in order) in the
# room name with everything but letters / CJK removed, so "MasterBedroom",
# "master_bedroom_2" and "主卧" all become "bedroom".  Numbered or functional
# distinctions ("second bedroom") are deliberately dropped: a viewer cannot
# tell which bedroom is the "second" one.
ROOM_TYPE_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("livingdining", "living room"),
    ("diningliving", "living room"),
    ("bathroom", "bathroom"),
    ("restroom", "bathroom"),
    ("washroom", "bathroom"),
    ("toilet", "bathroom"),
    ("lavatory", "bathroom"),
    ("bedroom", "bedroom"),
    ("kidsroom", "bedroom"),
    ("kidroom", "bedroom"),
    ("childroom", "bedroom"),
    ("childrenroom", "bedroom"),
    ("elderlyroom", "bedroom"),
    ("elderroom", "bedroom"),
    ("nannyroom", "bedroom"),
    ("guestroom", "bedroom"),
    ("babyroom", "bedroom"),
    ("kitchen", "kitchen"),
    ("diningroom", "dining room"),
    ("dining", "dining room"),
    ("livingroom", "living room"),
    ("lounge", "living room"),
    ("sittingroom", "living room"),
    ("library", "study"),
    ("study", "study"),
    ("office", "study"),
    ("balcony", "balcony"),
    ("terrace", "balcony"),
    ("cloakroom", "walk-in closet"),
    ("closet", "walk-in closet"),
    ("laundry", "laundry room"),
    ("storageroom", "storage room"),
    ("storage", "storage room"),
    ("hallway", "hallway"),
    ("corridor", "hallway"),
    ("aisle", "hallway"),
    ("entrance", "hallway"),
    ("foyer", "hallway"),
    ("garage", "garage"),
    ("gym", "gym"),
    # Chinese room names (Manycore / Kujiale exports)
    ("客餐厅", "living room"),
    ("客厅", "living room"),
    ("起居室", "living room"),
    ("餐厅", "dining room"),
    ("厨房", "kitchen"),
    ("卫生间", "bathroom"),
    ("洗手间", "bathroom"),
    ("浴室", "bathroom"),
    ("主卫", "bathroom"),
    ("次卫", "bathroom"),
    ("卧室", "bedroom"),
    ("主卧", "bedroom"),
    ("次卧", "bedroom"),
    ("儿童房", "bedroom"),
    ("老人房", "bedroom"),
    ("客房", "bedroom"),
    ("书房", "study"),
    ("阳台", "balcony"),
    ("走廊", "hallway"),
    ("过道", "hallway"),
    ("玄关", "hallway"),
    ("衣帽间", "walk-in closet"),
    ("储物间", "storage room"),
    ("储藏室", "storage room"),
    ("洗衣房", "laundry room"),
)
ROOM_TYPES = frozenset(t for _, t in ROOM_TYPE_KEYWORDS)
ROOM_PREPOSITION = {"balcony": "on"}  # default "in"

_SIZE_WORDS = {"larger": ("volume", True), "smaller": ("volume", False),
               "taller": ("height", True), "shorter": ("height", False)}
_MARKERS = (" next to the ", " in the ", " on the ")

METHOD_UNIQUE = "unique_label"
METHOD_ROOM = "room"
METHOD_ANCHOR = "anchor"
METHOD_SIZE = "size"


@dataclasses.dataclass(frozen=True)
class ReferringConfig:
    """Thresholds of the referring-expression generator (all sweepable)."""

    methods: Tuple[str, ...] = (METHOD_UNIQUE, METHOD_ROOM, METHOD_ANCHOR, METHOD_SIZE)
    # anchor ("next to") ------------------------------------------------------
    anchor_max_gap_m: float = 1.5        # target-anchor XY box gap ("within ~1.5 m")
    anchor_margin_ratio: float = 1.5     # 2nd-closest / closest same-category distance
    anchor_dist_floor_m: float = 0.25    # slack added to gaps so touching instances tie
    anchor_min_longest_edge_m: float = 0.3  # anchors must be visually substantial
    anchor_max_bottom_z_m: float = 2.0   # no ceiling-mounted anchors
    anchor_require_same_room: bool = True
    # size comparatives -------------------------------------------------------
    size_ratio: float = 1.5
    size_requires_same_room: bool = True
    # rooms -------------------------------------------------------------------
    room_fallback_max_dist_m: float = 0.5  # object centre just outside a polygon -> nearest room


DEFAULT_REFERRING = ReferringConfig()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"[_\s]+", " ", (text or "").strip().lower()).strip()


def display_label(label: str) -> str:
    """Natural display text for a raw InteriorGS label (Obj.label stays raw)."""
    raw = (label or "").strip().lower()
    if raw in DISPLAY_LABELS:
        return DISPLAY_LABELS[raw]
    s = _norm(raw)
    if s.startswith("the "):
        s = s[4:]
    if s in DISPLAY_LABELS:
        return DISPLAY_LABELS[s]
    if s.endswith(" combination") and len(s) > len(" combination"):
        s = s[: -len(" combination")].strip()
        if s in DISPLAY_LABELS:
            return DISPLAY_LABELS[s]
    return re.sub(r"\btv\b", "TV", s)


def category_matches(phrase: str, obj_display: str) -> bool:
    """True if the category phrase would be read as covering this object."""
    p = phrase.strip().lower()
    d = obj_display.strip().lower()
    return bool(p) and (d == p or d.endswith(" " + p))


def same_category(label_a: str, label_b: str) -> bool:
    """Raw labels whose display texts are equal or hypernym-related."""
    da, db = display_label(label_a), display_label(label_b)
    return category_matches(da, db) or category_matches(db, da)


def has_viewpoint_word(text: str) -> bool:
    return any(tok in VIEWPOINT_WORDS for tok in re.findall(r"[a-z]+", text.lower()))


def room_type(name: Optional[str]) -> Optional[str]:
    """Coarse room type of a structure.json room name, or None if unusable."""
    if not name:
        return None
    compact = re.sub(r"[^a-z一-鿿]", "", str(name).lower())
    if not compact:
        return None
    for key, rtype in ROOM_TYPE_KEYWORDS:
        if key in compact:
            return rtype
    return None


def _label_usable(disp: str) -> bool:
    """A display label can head an expression only if it is non-empty, has no
    viewpoint word and cannot be confused with the expression grammar."""
    low = disp.strip().lower()
    if not low or has_viewpoint_word(low):
        return False
    if low.split()[0] in _SIZE_WORDS:
        return False
    padded = f" {low} "
    return not any(m in padded for m in _MARKERS)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def xy_gap(a: Obj, b: Obj) -> float:
    """Euclidean gap between the XY footprints of two boxes (0 if overlapping)."""
    dx = max(0.0, float(a.aabb_min[0] - b.aabb_max[0]), float(b.aabb_min[0] - a.aabb_max[0]))
    dy = max(0.0, float(a.aabb_min[1] - b.aabb_max[1]), float(b.aabb_min[1] - a.aabb_max[1]))
    return math.hypot(dx, dy)


def _volume(o: Obj) -> float:
    return float(np.prod(np.maximum(o.dims, 0.0)))


def _height(o: Obj) -> float:
    return float(max(o.dims[2], 0.0))


def _point_polygon_distance(x: float, y: float, poly: np.ndarray) -> float:
    """Distance from (x, y) to the polygon boundary (vectorised over edges)."""
    a = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(a) == 0:
        return math.inf
    b = np.roll(a, -1, axis=0)
    ab = b - a
    p = np.array([x, y], dtype=float)
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom < 1e-12, 0.0, np.einsum("ij,ij->i", p - a, ab) / np.maximum(denom, 1e-12))
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(p - proj, axis=1)))


def effective_room_index(rooms: Sequence[Any], obj: Obj, max_dist: float) -> Optional[int]:
    """``obj.room_index`` if set (scene.py: centre inside a room polygon), else
    the nearest room whose boundary is within ``max_dist`` of the centre
    (objects pushed against a wall often have their centre just outside)."""
    if obj.room_index is not None:
        return int(obj.room_index)
    if not rooms or max_dist <= 0:
        return None
    c = obj.center
    best, best_d = None, math.inf
    for r in rooms:
        d = _point_polygon_distance(float(c[0]), float(c[1]), r.polygon)
        if d < best_d:
            best, best_d = int(r.index), d
    return best if best_d <= max_dist else None


class _SceneIndex:
    """Per-scene caches: display labels, room types, category matches."""

    def __init__(self, scene: Any, pool: Sequence[Obj], cfg: ReferringConfig) -> None:
        self.scene = scene
        self.cfg = cfg
        self.pool = list(pool)
        self.disp = [display_label(o.label) for o in self.pool]
        self.rooms = list(getattr(scene, "rooms", []) or [])
        self.rooms_by_index = {int(r.index): r for r in self.rooms}
        self._room_cache: Dict[int, Optional[int]] = {}
        self._match_cache: Dict[str, List[int]] = {}

    def room_of(self, i: int) -> Optional[int]:
        """Effective room index of pool[i] (lazy, cached)."""
        if i not in self._room_cache:
            self._room_cache[i] = effective_room_index(self.rooms, self.pool[i],
                                                       self.cfg.room_fallback_max_dist_m)
        return self._room_cache[i]

    def rtype_of(self, i: int) -> Optional[str]:
        idx = self.room_of(i)
        if idx is None or idx not in self.rooms_by_index:
            return None
        return room_type(getattr(self.rooms_by_index[idx], "name", None))

    def matches(self, phrase: str) -> List[int]:
        key = phrase.strip().lower()
        if key not in self._match_cache:
            self._match_cache[key] = [i for i, d in enumerate(self.disp) if category_matches(key, d)]
        return self._match_cache[key]

    def anchor_distances(self, members: Sequence[int], anchor: int) -> List[float]:
        a = self.pool[anchor]
        floor = max(float(self.cfg.anchor_dist_floor_m), 1e-3)  # keeps ratio tests strict for ties
        return [xy_gap(self.pool[i], a) + floor for i in members]


def _ensure_in_pool(pool: List[Obj], objs: Sequence[Obj]) -> Dict[str, int]:
    """id -> pool index; objs missing from the pool are appended (they are
    instances too).  Callers pass copies of scene objects, matched by id."""
    index: Dict[str, int] = {}
    for i, o in enumerate(pool):
        index.setdefault(o.id, i)
    for o in objs:
        if o.id not in index:
            index[o.id] = len(pool)
            pool.append(o)
    return index


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------
def default_anchor_ok(obj: Obj) -> bool:
    """Anchor eligibility when the caller passes no ``anchor_ok``: the same
    semantic exclusions as v10 targets (structure, fixtures, doors, windows,
    textiles, clutter)."""
    from .objects import get_preset, semantic_exclusion_reason  # lazy: avoid import cycle

    return semantic_exclusion_reason(obj.label, get_preset("v10_relaxed")) is None


def assign_referring_expressions(scene: Any, objs: Sequence[Obj],
                                 config: Optional[ReferringConfig] = None,
                                 pool: Optional[Sequence[Obj]] = None,
                                 anchor_ok: Optional[Callable[[Obj], bool]] = None,
                                 reference_ok: Optional[Callable[[Obj], bool]] = None) -> Funnel:
    """Set ``obj.ref`` / ``obj.ref_method`` for every obj in ``objs``.

    ``pool`` defaults to ``scene.objects`` (ALL instances) and is the universe
    against which uniqueness is checked.  ``anchor_ok`` restricts which pool
    objects may serve as anchors (default: v10 semantic filter).
    ``reference_ok`` (default: any) must hold for every object an expression
    refers to besides the target: the anchor and the other object of a size
    comparison (e.g. objects on the no-task list).
    Funnel stages: ``referring`` (method used, or ``no_ref``) and
    ``referring.no_ref_detail`` (``<method>:<why>`` for every method tried on
    an instance that ended without a ref).
    """
    cfg = config or DEFAULT_REFERRING
    funnel = Funnel()
    pool = list(scene.objects if pool is None else pool)
    id_index = _ensure_in_pool(pool, objs)
    idx = _SceneIndex(scene, pool, cfg)
    anchor_ok = anchor_ok or default_anchor_ok
    reference_ok = reference_ok or (lambda _o: True)

    # Anchors: substantial, not ceiling-mounted, unambiguous on their own.
    anchors: List[int] = []
    for j, a in enumerate(pool):
        disp = idx.disp[j]
        if not _label_usable(disp) or not anchor_ok(a) or not reference_ok(a):
            continue
        if not np.all(np.isfinite(a.aabb_min)) or not np.all(np.isfinite(a.aabb_max)):
            continue
        if float(np.max(a.dims)) < cfg.anchor_min_longest_edge_m:
            continue
        if float(a.aabb_min[2]) > cfg.anchor_max_bottom_z_m:
            continue
        if idx.matches(disp) == [j]:
            anchors.append(j)

    for obj in objs:
        obj.ref, obj.ref_method = None, None
        t = id_index[obj.id]
        disp = idx.disp[t]
        if not _label_usable(disp):
            funnel.add("referring", "no_ref")
            funnel.add("referring.no_ref_detail", "label_unusable")
            continue
        members = idx.matches(disp)
        details: List[str] = []
        ref: Optional[Tuple[str, str]] = None
        for method in cfg.methods:
            if method == METHOD_UNIQUE:
                if members == [t]:
                    ref = (f"the {disp}", METHOD_UNIQUE)
                else:
                    details.append("unique:duplicate_category")
            elif method == METHOD_ROOM:
                ref = _try_room(idx, t, disp, members, details)
            elif method == METHOD_ANCHOR:
                ref = _try_anchor(idx, t, disp, members, anchors, details)
            elif method == METHOD_SIZE:
                ref = _try_size(idx, t, disp, members, details, reference_ok)
            else:
                raise ValueError(f"unknown referring method {method!r}")
            if ref is not None:
                break
        if ref is None:
            funnel.add("referring", "no_ref")
            for d in details:
                funnel.add("referring.no_ref_detail", d)
            continue
        text, method_used = ref
        if has_viewpoint_word(text):  # cannot happen: labels are pre-checked
            raise AssertionError(f"viewpoint word in referring expression {text!r}")
        obj.ref, obj.ref_method = text, method_used
        funnel.add("referring", method_used)
    return funnel


def _try_room(idx: _SceneIndex, t: int, disp: str, members: Sequence[int],
              details: List[str]) -> Optional[Tuple[str, str]]:
    if idx.room_of(t) is None:
        details.append("room:outside_rooms")
        return None
    rtype = idx.rtype_of(t)
    if rtype is None:
        details.append("room:room_unnamed")
        return None
    in_type = [i for i in members if idx.rtype_of(i) == rtype]
    if in_type != [t]:
        details.append("room:not_unique_in_room_type")
        return None
    prep = ROOM_PREPOSITION.get(rtype, "in")
    return (f"the {disp} {prep} the {rtype}", METHOD_ROOM)


def _try_anchor(idx: _SceneIndex, t: int, disp: str, members: Sequence[int],
                anchors: Sequence[int], details: List[str]) -> Optional[Tuple[str, str]]:
    cfg = idx.cfg
    target = idx.pool[t]
    cand = []
    for j in anchors:
        if j == t or same_category(idx.pool[j].label, target.label):
            continue
        gap = xy_gap(target, idx.pool[j])
        if gap <= cfg.anchor_max_gap_m:
            cand.append((gap, j))
    if not cand:
        details.append("anchor:none_within_range")
        return None
    cand.sort(key=lambda gj: (gj[0], gj[1]))
    reason = "anchor:margin_too_small"
    members = list(members)
    for _, j in cand:
        if cfg.anchor_require_same_room:
            rt, ra = idx.room_of(t), idx.room_of(j)
            if rt is not None and ra is not None and rt != ra:
                reason = "anchor:other_room"
                continue
        dists = idx.anchor_distances(members, j)
        d_t = dists[members.index(t)]
        if all(d >= cfg.anchor_margin_ratio * d_t for i, d in zip(members, dists) if i != t):
            return (f"the {disp} next to the {idx.disp[j]}", METHOD_ANCHOR)
        reason = "anchor:margin_too_small"
    details.append(reason)
    return None


def _try_size(idx: _SceneIndex, t: int, disp: str, members: Sequence[int],
              details: List[str], reference_ok: Optional[Callable[[Obj], bool]] = None) -> Optional[Tuple[str, str]]:
    cfg = idx.cfg
    if len(members) != 2:
        details.append("size:not_exactly_two")
        return None
    other = members[0] if members[1] == t else members[1]
    if reference_ok is not None and not reference_ok(idx.pool[other]):
        details.append("size:other_not_referable")
        return None
    if cfg.size_requires_same_room:
        rt, ro = idx.room_of(t), idx.room_of(other)
        if rt is None or ro is None or rt != ro:
            details.append("size:other_room")
            return None
    a, b = idx.pool[t], idx.pool[other]
    for kind, fn in (("volume", _volume), ("height", _height)):
        va, vb = fn(a), fn(b)
        lo, hi = min(va, vb), max(va, vb)
        if lo > 0 and hi / lo >= cfg.size_ratio:
            if kind == "volume":
                word = "larger" if va > vb else "smaller"
            else:
                word = "taller" if va > vb else "shorter"
            return (f"the {word} {disp}", METHOD_SIZE)
    details.append("size:ratio_below_threshold")
    return None


# ---------------------------------------------------------------------------
# Resolution (inverse; used by tests and audits)
# ---------------------------------------------------------------------------
_RE_SIZE = re.compile(r"^the (larger|smaller|taller|shorter) (.+)$")
_RE_ANCHOR = re.compile(r"^the (.+?) next to the (.+)$")
_RE_ROOM = re.compile(r"^the (.+) (in|on) the (.+)$")
_RE_PLAIN = re.compile(r"^the (.+)$")


class Resolver:
    """Reusable ``resolve`` for many expressions over one scene/pool."""

    def __init__(self, scene: Any, objs_all: Optional[Sequence[Obj]] = None,
                 config: Optional[ReferringConfig] = None) -> None:
        self.cfg = config or DEFAULT_REFERRING
        self.pool = list(scene.objects if objs_all is None else objs_all)
        self.idx = _SceneIndex(scene, self.pool, self.cfg)

    def __call__(self, ref_text: str) -> List[Obj]:
        return [self.pool[i] for i in self._indices((ref_text or "").strip().lower())]

    def _indices(self, low: str) -> List[int]:
        idx, cfg = self.idx, self.cfg
        m = _RE_SIZE.match(low)
        if m:
            members = idx.matches(m.group(2))
            if len(members) != 2:
                return list(members)
            kind, want_big = _SIZE_WORDS[m.group(1)]
            fn = _volume if kind == "volume" else _height
            a, b = members
            va, vb = fn(self.pool[a]), fn(self.pool[b])
            lo, hi = min(va, vb), max(va, vb)
            if lo <= 0 or hi / lo < cfg.size_ratio:
                return [a, b]
            big, small = (a, b) if va > vb else (b, a)
            return [big if want_big else small]

        m = _RE_ANCHOR.match(low)
        if m:
            members = idx.matches(m.group(1))
            out: List[int] = []
            for j in idx.matches(m.group(2)):
                if not members:
                    break
                dists = idx.anchor_distances(members, j)
                k = int(np.argmin(dists))
                if not xy_gap(self.pool[members[k]], self.pool[j]) <= cfg.anchor_max_gap_m:
                    continue  # nothing of that category is next to this anchor
                d_min = dists[k]
                out.extend(i for i, d in zip(members, dists)
                           if d < cfg.anchor_margin_ratio * d_min or i == members[k])
            return sorted(set(out))

        m = _RE_ROOM.match(low)
        if m and m.group(3) in ROOM_TYPES:
            rtype = m.group(3)
            return [i for i in idx.matches(m.group(1)) if idx.rtype_of(i) == rtype]

        m = _RE_PLAIN.match(low)
        if m:
            return list(idx.matches(m.group(1)))
        return []


def resolve(scene: Any, objs_all: Optional[Sequence[Obj]], ref_text: str,
            config: Optional[ReferringConfig] = None) -> List[Obj]:
    """All objects of ``objs_all`` (default ``scene.objects``) that a reader
    could take ``ref_text`` to denote, under the module semantics.  An emitted
    expression is valid iff this returns exactly one object."""
    return Resolver(scene, objs_all, config)(ref_text)


__all__ = [
    "DEFAULT_REFERRING",
    "DISPLAY_LABELS",
    "ROOM_TYPES",
    "ReferringConfig",
    "Resolver",
    "VIEWPOINT_WORDS",
    "assign_referring_expressions",
    "category_matches",
    "display_label",
    "effective_room_index",
    "has_viewpoint_word",
    "resolve",
    "room_type",
    "same_category",
    "xy_gap",
]
