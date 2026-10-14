"""What a start pose decides about a task's text, and the occlusion-alignment start rule (2026-10-14, round 6).

Referring rules v3 (referring.py) describe a target that is the only one of its kind in its room as just
"the X" (``ref_method`` "room_local").  That holds only for a start inside that room from which no other X
can be recognised; ``start_refs`` decides, per start, between "the X" and the room form "the X in the R"
(``ref_outside``), or rejects the start when neither works.  The success rule does not depend on the text:
neither form uses a reference object.  ``apply_refs`` writes the chosen expressions into a task row and
rebuilds its description.

Occlusion alignment ("hide A behind B"): a start where A is already hidden looks solved to a reader even when
the judge disagrees (A partly out of frame, hidden by something else), so with ``oa_start_target_visible``
the hidden target must be clearly visible at the start (``oa_target_visible``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .referring import METHOD_ROOM, METHOD_ROOM_LOCAL, display_label


def target_entries(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((item.get("target_object") or {}).get("objects") or [])


def has_room_local(item: Dict[str, Any]) -> bool:
    return any(e.get("ref_method") == METHOD_ROOM_LOCAL for e in target_entries(item))


def start_refs(scene: Any, item: Dict[str, Any], c2w: Any,
               obs_fn: Callable[[Any, Any, Any], Dict[str, Any]]) -> Optional[Dict[str, Dict[str, str]]]:
    """{target id: {"ref", "ref_method"}} for the room_local targets of ``item`` at start ``c2w``; {} if it has
    none; None if some target can be described neither by its name nor by its room from this start.

    "the X" needs the start inside the target's room (``ref_room``), no other X (``ref_confusers``)
    recognisable from the start (``obs_fn``: the start-frame rule) and no other target of the task called X."""
    entries = target_entries(item)
    local = [e for e in entries if e.get("ref_method") == METHOD_ROOM_LOCAL]
    if not local:
        return {}
    c2w = np.asarray(c2w, float)
    room = scene.room_of(float(c2w[0, 3]), float(c2w[1, 3]))
    names = [display_label(str(e.get("label", ""))) for e in entries]
    by_id = {str(o.id): o for o in getattr(scene, "objects", [])}
    out: Dict[str, Dict[str, str]] = {}
    for e in local:
        name_ok = (room is not None and int(room) == int(e["ref_room"])
                   and names.count(display_label(str(e.get("label", "")))) == 1)
        if name_ok:
            for cid in e.get("ref_confusers") or []:
                other = by_id.get(str(cid))
                if other is not None and obs_fn(scene, c2w, other).get("identifiable"):
                    name_ok = False
                    break
        if name_ok:
            out[str(e["id"])] = {"ref": str(e["ref"]), "ref_method": METHOD_ROOM_LOCAL}
        elif e.get("ref_outside"):
            out[str(e["id"])] = {"ref": str(e["ref_outside"]), "ref_method": METHOD_ROOM}
        else:
            return None
    return out


def describe(item: Dict[str, Any]) -> str:
    """The task description from the target entries' ``ref`` (the builders' own wording)."""
    from . import contract as C
    from .sampling import fov_description, projective_description

    task_type = str(item.get("task_type"))
    objs = target_entries(item)
    if task_type == "projective_relations":
        relation = item["target_region"]["params"]["relation"]
        return projective_description(objs[0]["ref"], objs[1]["ref"], relation)
    if task_type == "fov_inclusion" and item.get("canonical_task_metric_version") == C.CANONICAL_TASK_METRIC_V1:
        return fov_description(objs[0]["ref"], objs[1]["ref"])
    from .families.canonical_v2 import FAMILIES

    return FAMILIES[task_type].describe(item)


def apply_refs(row: Dict[str, Any], refs: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, Any]:
    """Write the start's expressions (``start_refs``) into the target entries of ``row`` and rebuild its
    description when one changed.  Returns ``row``."""
    if not refs:
        return row
    changed = False
    for e in target_entries(row):
        new = refs.get(str(e.get("id")))
        if new is None:
            continue
        changed |= e.get("ref") != new["ref"]
        e["ref"], e["ref_method"] = new["ref"], new["ref_method"]
    if changed:
        row["task_description"] = describe(row)
    return row


def oa_target_visible(item: Dict[str, Any], observability: Dict[str, Any]) -> bool:
    """Occlusion alignment: the target to hide is recognisable in the start frame (``observability`` is the
    start's ``initial_observability``).  Always True for other families."""
    if item.get("task_type") != "occlusion_alignment":
        return True
    params = (item.get("target_region") or {}).get("params") or {}
    hidden = params.get("occluded", params.get("object_a_id"))
    if hidden is None:
        entries = target_entries(item)
        hidden = entries[0].get("id") if entries else None
    per = (observability.get("targets") or {}).get(str(hidden)) or {}
    return bool(per.get("identifiable"))
