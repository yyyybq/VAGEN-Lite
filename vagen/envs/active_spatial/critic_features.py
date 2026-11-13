"""The env's ground truth for the privileged critic (trainer.critic_source "features", vagen/feature_critic.py; the
user's supplement of 2026-11-13, gate 1 of PREREG_PPO2.md).  Off by default (env critic_features false): nothing here
runs and the env's info is as before.

With critic_features on, reset() and step() add info["critic_features"]: the truth at the pose the next observation is
rendered from (the pose the model sees before its next answer), read by the agent loop as the state of that answer's
turn.  Nothing here changes a reward, the success rule or termination.

    d          the remaining minimum number of primitive actions to success from the pose, from the task's distance
               table (<tables_dir>/<task_id>.npz, data_gen.interiorgs_v10.distance_table), the 26 neighbour cells'
               smallest when the pose's own cell is missing (distance_table_near); None when neither has one
    near       d came from a neighbour cell
    table      the task has a table
    bearing    the sector of the task's object A from the camera's floor-projected facing direction (front / right /
               back / left, data_gen.interiorgs_v10.grounding_probe.truth); None when the camera stands inside the
               object's footprint or has no angle to its centre
    condition  absolute_positioning: the horizontal footprint distance against the required one (too_close / ok /
               too_far, the success rule's tolerance, grounding_probe.truth); None for the other families
    family     the task family (task_type)
    phi        the env's potential (shaping score, prev_potential_score) at the pose: for the critic's optional read-out
               term (a dense reward's return-to-go carries -scale * phi), not an MLP input

A failure while computing them (say a table that cannot be read) is caught (safe()): info["critic_features"] is None
and info["critic_features_error"] says why, the step's reward, success and end stay the env's; the trainer then
refuses the turn loudly.  A missing tables directory is a configuration error and raises at reset.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

VERSION = 1
_CACHE_SIZE = 8  # a table is 9-21 MB loaded; an agent-loop worker runs about 2 tasks an update, rarely the same again
_TABLES: "OrderedDict[tuple, Optional[Dict[str, Any]]]" = OrderedDict()  # (root, task_id) -> table, shared per process


def table_for(root: str, task_id: str) -> Optional[Dict[str, Any]]:
    """The task's distance table (read-only, cached per process; None when the file is missing)."""
    from data_gen.interiorgs_v10.distance_table import load, table_path

    key = (str(root), str(task_id))
    if key in _TABLES:
        _TABLES.move_to_end(key)
        return _TABLES[key]
    path = table_path(Path(root), str(task_id))
    table = load(path) if path.is_file() else None
    _TABLES[key] = table
    while len(_TABLES) > _CACHE_SIZE:
        _TABLES.popitem(last=False)
    return table


def features(item: Mapping[str, Any], c2w: Any, tables_dir: Optional[str], phi: Optional[float] = None
             ) -> Dict[str, Any]:
    """The ground truth of a pose (c2w: the env's view_engine.get_pose()); phi, the env's potential (shaping score)
    at the pose, is carried along for the critic's read-out (not an MLP input)."""
    from data_gen.interiorgs_v10.distance_table_near import lookup_near
    from data_gen.interiorgs_v10.grounding_probe import truth

    check_dir(tables_dir)
    table = table_for(tables_dir, str(item.get("task_id")))
    found = lookup_near(table, c2w) if table is not None else {"distance": None, "near": False}
    gt = truth(dict(item), c2w)
    inside = float(gt.get("distance_m", 1.0)) <= 0.0
    return {"v": VERSION,
            "d": None if found["distance"] is None else int(found["distance"]),
            "near": bool(found["near"]), "table": table is not None,
            "bearing": None if inside else gt.get("bearing"),
            "condition": gt.get("condition"),
            "family": str(item.get("task_type") or ""),
            "phi": None if phi is None else float(phi)}


def check_dir(tables_dir: Optional[str]) -> None:
    """A configuration error, raised as it is (the env calls it outside the guard of safe())."""
    if not tables_dir:
        raise ValueError("critic_features needs critic_feature_tables_dir (the tasks' distance tables)")


def safe(item: Mapping[str, Any], c2w: Any, tables_dir: Optional[str], phi: Optional[float] = None
         ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(features, None), or (None, the error) when computing them fails: the env's step must not fail on the critic's
    account (the agent loop would end the episode with reward 0); the trainer refuses a turn without features."""
    check_dir(tables_dir)
    try:
        return features(item, c2w, tables_dir, phi), None
    except Exception as exc:  # noqa: BLE001 - recorded in info["critic_features_error"]
        return None, f"{type(exc).__name__}: {exc}"
