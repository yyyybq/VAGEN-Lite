"""Shared state of one scene's question generation: views, names, counters, question assembly."""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10.referring import category_matches

from .geom import OBB, box_distance
from .qa import Question, build_variants, count_tokens, qid
from .views import ViewBank
from .visible import ViewAnalysis


@dataclasses.dataclass
class Combo:
    """A set of objects (and cameras) satisfying a group's conditions before rendering."""

    group: str
    kind: str
    objects: Tuple[str, ...]
    shots: List[List[np.ndarray]]        # candidate camera sets, best first; each set = one image per entry
    info: Dict[str, Any] = dataclasses.field(default_factory=dict)


class GenCtx:
    def __init__(self, ctx, bank: ViewBank, split: str, seed: int = 20261014) -> None:
        self.ctx = ctx
        self.bank = bank
        self.split = split
        self.seed = seed
        self.rng = random.Random(f"{seed}|{ctx.scene_id}")
        self.questions: List[Question] = []
        self.stats: Dict[str, Counter] = defaultdict(Counter)   # group -> counters
        self.fail: Dict[str, Counter] = defaultdict(Counter)    # group -> rejection reasons
        self.census = False     # True: generators drop the caps they use to save time (combo census)

    def cap(self, n: int, census: Optional[int] = None) -> int:
        """A generator's search cap: ``n`` when generating; in census mode ``census`` (a larger sample whose
        count is scaled up in the report) or no cap at all."""
        if not self.census:
            return n
        return 10 ** 9 if census is None else census

    # ------------------------------------------------------------------ names
    def name_single(self, va: ViewAnalysis, oid: str) -> Optional[Tuple[str, str]]:
        """Single-image name: "the X" when no other X shows in this image; else the v10_r5 scene name if
        its reference object shows clearly too.  (name, method) or None."""
        disp = self.ctx.disp[oid]
        if va.name_unique_in_image(oid):
            return f"the {disp}", "unique_in_image"
        ref = self.ctx.scene_ref(oid)
        anchor = self.ctx.scene_ref_anchor(oid)
        if ref and (anchor is None or va.obj(anchor).recognisable(0.5)):
            return ref, "scene_v10_r5"
        return None

    def name_scene(self, oid: str) -> Optional[str]:
        """Multi-image / video name: unique in the whole scene (v10_r5 rules)."""
        return self.ctx.scene_ref(oid)

    def distinct_names(self, oids: Sequence[str]) -> bool:
        """No two objects whose display names could be confused (equal or one a hypernym of the other)."""
        d = [self.ctx.disp[o] for o in oids]
        for i, j in itertools.combinations(range(len(d)), 2):
            if category_matches(d[i], d[j]) or category_matches(d[j], d[i]):
                return False
        return True

    def obj_entry(self, oid: str, name: str, role: str, method: str = "") -> Dict[str, Any]:
        o = self.ctx.by_id[oid]
        return {"id": oid, "label": o.label, "display": self.ctx.disp[oid], "name": name, "role": role,
                "name_method": method, "obb": self.ctx.obb[oid].to_json()}

    # ------------------------------------------------------------------ question assembly
    def emit(self, group: str, subtasks: Sequence[str], benchmark: str, media: str, view_ids: Sequence[str],
             images: Sequence[str], families: Dict[str, Any], fields: Dict[str, Any],
             option_keys: Optional[Sequence[Any]], answer: Any, answer_type: str, gt: Dict[str, Any],
             objects: List[Dict[str, Any]], review: Optional[Dict[str, Any]] = None,
             extra: Optional[Dict[str, Any]] = None, keep_orig_order: Optional[Sequence[Any]] = None) -> Question:
        q_id = qid(self.ctx.scene_id, group, ",".join(subtasks), ",".join(view_ids), str(answer),
                   ",".join(o["id"] for o in objects), len(self.questions))
        variants = build_variants(families, fields, option_keys, answer, q_id, keep_orig_order,
                                  numeric=(answer_type == "numeric"))
        q = Question(qid=q_id, group=group, subtasks=list(subtasks), benchmark=benchmark,
                     scene_id=self.ctx.scene_id, split=self.split, media=media, views=list(view_ids),
                     variants=variants, answer=answer, answer_type=answer_type, gt=gt, objects=objects,
                     review=review or {}, extra=dict(extra or {}, images=list(images)))
        q.tokens = {"512": count_tokens(q, 512), "256": count_tokens(q, 256)}
        self.questions.append(q)
        self.stats[group]["questions"] += 1
        for s in subtasks:
            self.stats[group]["subtask:" + s] += 1
        return q

    def count(self, group: str, key: str, n: int = 1) -> None:
        self.stats[group][key] += n

    def reject(self, group: str, reason: str) -> None:
        self.fail[group][reason] += 1


# =============================================================================
# Helpers shared by groups
# =============================================================================
# answer balancing for yes/no wordings: the counts live for the whole process (one render run walks many scenes)
_BALANCE: Counter = Counter()


def balanced_choice(rng: random.Random, key: str, cands: Sequence[Tuple[str, str]],
                    max_lead: Optional[int] = None) -> Optional[Tuple[str, str]]:
    """Pick one (wording, yes/no answer) candidate so that, over the run, each wording is answered "yes" about as
    often as "no": prefer the candidate whose wording is furthest behind on its answer, then the least used
    wording, ties at random. With ``max_lead``, return None (skip the question) when every candidate's answer
    already leads its wording by more than that (e.g. a pair that can only be asked with answer "no").
    Counts live for the whole process (one render run walks many scenes)."""
    def score(c: Tuple[str, str]) -> Tuple[int, int]:
        w, ans = c
        other = "no" if ans == "yes" else "yes"
        return (_BALANCE[(key, w, ans)] - _BALANCE[(key, w, other)],
                _BALANCE[(key, w, "yes")] + _BALANCE[(key, w, "no")])
    cands = list(cands)
    low = min(score(c) for c in cands)
    if max_lead is not None and low[0] > max_lead:
        return None
    pick = rng.choice([c for c in cands if score(c) == low])
    _BALANCE[(key, pick[0], pick[1])] += 1
    return pick


def bare(name: str) -> str:
    return name[4:] if name.startswith("the ") else name


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def pairs_near(ctx, pool: Sequence[str], dmin: float = 0.5, dmax: float = 4.5, same_room: bool = True,
               limit: Optional[int] = None, rng: Optional[random.Random] = None) -> List[Tuple[str, str]]:
    out = []
    for a, b in itertools.combinations(sorted(pool), 2):
        if same_room and ctx.room_of_obj.get(a) is not None and ctx.room_of_obj.get(a) != ctx.room_of_obj.get(b):
            continue
        d = float(np.linalg.norm(ctx.obb[a].center[:2] - ctx.obb[b].center[:2]))
        if dmin <= d <= dmax:
            out.append((a, b))
    if rng is not None:
        rng.shuffle(out)
    return out[:limit] if limit else out


def single_pool(ctx, max_longest: float = 3.0) -> List[str]:
    """Objects that may be the subject of a single-image question: the v10 single-object filters (size,
    semantics, no-task list), not a repeated box, not huge."""
    return sorted(o for o in ctx.eligible if o not in ctx.duplicates and ctx.obb[o].longest_edge <= max_longest)


def scene_pool(ctx, max_longest: float = 3.5) -> List[str]:
    """Objects with a scene-unique v10_r5 name (multi-image and video questions)."""
    return sorted(o for o in ctx.named if o in ctx.obb and o not in ctx.duplicates
                  and ctx.obb[o].longest_edge <= max_longest)


def distance(ctx, a: str, b: str) -> float:
    return box_distance(ctx.obb[a], ctx.obb[b])


def bbox_gap_x(b1, b2) -> float:
    """Horizontal gap from box 1's right edge to box 2's left edge (positive: 1 entirely left of 2)."""
    return float(b2[0] - b1[2])


def bbox_gap_y(b1, b2) -> float:
    return float(b2[1] - b1[3])


def overlap_1d(a0, a1, b0, b1) -> float:
    return float(min(a1, b1) - max(a0, b0))


def simple_named_pool(ctx, max_longest: float = 3.5) -> List[str]:
    """Scene-unique objects whose name needs no reference object ("the X" or "the X in the R"): MindCube-
    and ViewSpatial-style questions put names inside spatial phrases ("is there a X behind the Y"), where a
    "next to" description would read as part of the relation."""
    return [o for o in scene_pool(ctx, max_longest) if ctx.named[o].ref_method in ("unique_label", "room")]
