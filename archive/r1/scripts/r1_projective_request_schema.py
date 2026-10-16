#!/usr/bin/env python3
"""Frozen Projective half-plane fields required by R1 layout search."""

from __future__ import annotations

import math
from typing import Any, Sequence


REQUIRED_PROJECTIVE_PARAMS = (
    "boundary_point",
    "boundary_direction",
    "normal",
    "object_a_center",
    "object_b_center",
    "relation",
)


def half_plane_geometry(
    object_a_center: Sequence[float],
    object_b_center: Sequence[float],
    relation: str,
) -> dict[str, Any]:
    """Match ``TaskGenerator._generate_projective_relation`` exactly."""
    if relation not in {"left", "right"}:
        raise ValueError(f"unsupported projective relation: {relation!r}")
    ax, ay = float(object_a_center[0]), float(object_a_center[1])
    bx, by = float(object_b_center[0]), float(object_b_center[1])
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        raise ValueError("projective object centers must have distinct XY coordinates")
    direction = [dx / norm, dy / norm]
    # Frozen camera convention: left uses the clockwise A->B normal.
    normal = [direction[1], -direction[0]] if relation == "left" else [-direction[1], direction[0]]
    return {
        "boundary_point": [(ax + bx) / 2.0, (ay + by) / 2.0],
        "boundary_direction": direction,
        "normal": normal,
    }


def validate_projective_params(params: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_PROJECTIVE_PARAMS if key not in params]
    if missing:
        raise ValueError(f"projective request missing required params: {missing}")
