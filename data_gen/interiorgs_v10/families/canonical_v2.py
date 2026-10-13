"""Canonical v2 metric dispatcher and family registry (generator side).

``score_canonical_task_v2(item, c2w)``
    * ``canonical_spatial_task_h1_v1`` rows -> frozen runtime
      ``score_canonical_task`` (unchanged).
    * ``canonical_spatial_task_h1_v2`` rows:
        projective_relations -> the frozen v1 ``canonical_projective`` on
        ``score_observation`` (bit-identical result dict, including
        ``metric_version`` = v1 because the judge *is* the v1 judge);
        the families, fov_inclusion included (its own v2 judge since
        2026-10-13) -> ``judges_v2.score_v2_family``.

The runtime env has the same dispatch (``vagen/envs/active_spatial/canonical_task_metrics.py``,
see ENV_INTEGRATION.md); tests check that both paths agree exactly.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..contract import CANONICAL_TASK_METRIC_V1, CANONICAL_TASK_METRIC_V2
from . import (
    absolute_positioning,
    centering,
    equidistance,
    fov_inclusion,
    occlusion_alignment,
    screen_occupancy,
    size_distance_invariance,
)
from . import judges_v2 as J

V1_TASK_TYPES = frozenset(("projective_relations", "fov_inclusion"))
NEW_TASK_TYPES = J.NEW_TASK_TYPES
SUPPORTED_TASK_TYPES_V2 = V1_TASK_TYPES | NEW_TASK_TYPES

# task_type -> module with build_item, judge, sampler (= success_pose_candidates),
# describe, tuple_ok, param_menu, necessary_mask, anchor_xy, GATES_DOC, ...
FAMILIES = {
    absolute_positioning.TASK_TYPE: absolute_positioning,
    equidistance.TASK_TYPE: equidistance,
    occlusion_alignment.TASK_TYPE: occlusion_alignment,
    screen_occupancy.TASK_TYPE: screen_occupancy,
    size_distance_invariance.TASK_TYPE: size_distance_invariance,
    centering.TASK_TYPE: centering,
    fov_inclusion.TASK_TYPE: fov_inclusion,
}
# Families whose success regions / semantics are the most robust; the others
# are implemented and tested but flagged as riskier (see GATES_DOC / report).
RECOMMENDED_FAMILIES = ("absolute_positioning", "equidistance", "screen_occupancy", "occlusion_alignment",
                        "fov_inclusion")

if frozenset(FAMILIES) != NEW_TASK_TYPES:  # pragma: no cover
    raise RuntimeError("family registry and judges_v2.NEW_TASK_TYPES disagree")


def score_canonical_task_v2(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    from vagen.envs.active_spatial import canonical_task_metrics as v1

    version = item.get("canonical_task_metric_version")
    pose = np.asarray(c2w, dtype=float)
    if version == CANONICAL_TASK_METRIC_V1:
        return v1.score_canonical_task(item, pose)
    if version != CANONICAL_TASK_METRIC_V2:
        raise ValueError("item does not request a canonical metric backend (v1 or v2)")
    task_type = str(item.get("task_type") or "")
    if task_type == "projective_relations":
        return v1.canonical_projective(v1.score_observation(item, pose))
    if task_type in NEW_TASK_TYPES:  # fov_inclusion included: a v2 family since 2026-10-13
        return J.score_v2_family(item, pose)
    raise ValueError(f"canonical v2 metric is not defined for task_type={task_type!r}")


def uses_canonical_backend_v2(item: Dict[str, Any] | None) -> bool:
    """Generator-side mirror of the patched runtime ``uses_canonical_backend``."""
    if not item or not item.get("canonical_task_metric_version"):
        return False
    version = item["canonical_task_metric_version"]
    supported = {CANONICAL_TASK_METRIC_V1: V1_TASK_TYPES, CANONICAL_TASK_METRIC_V2: SUPPORTED_TASK_TYPES_V2}
    if version not in supported:
        raise ValueError("unknown canonical_task_metric_version; refusing legacy fallback")
    if item.get("task_type") not in supported[version]:
        raise ValueError("canonical metric version does not support this task type")
    return True


def build_item(task_type: str, scene_id: str, objs: Any, params: Dict[str, Any] | None = None,
               init_c2w: Any = None) -> Dict[str, Any]:
    return FAMILIES[task_type].build_item(scene_id, objs, params or {}, init_c2w)


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return score_canonical_task_v2(item, c2w)
