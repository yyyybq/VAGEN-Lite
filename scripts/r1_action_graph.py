"""Formal forward transition and forward-validated reverse-edge helpers.

The Active Spatial action graph is defined by ``ViewManipulator.step``.  A
mathematical inverse is only a proposal: every reverse edge is admitted only
when replaying the corresponding *forward* action reaches the current state
key under the same runtime transition and collision convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from r1_reachability_audit import state_key
from vagen.envs.active_spatial.utils import ViewManipulator


ACTIONS = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
)
TRANSLATION_ACTIONS = frozenset(ACTIONS[:4])
INVERSE_ACTION = {
    "move_forward": "move_backward",
    "move_backward": "move_forward",
    "move_left": "move_right",
    "move_right": "move_left",
    "turn_left": "turn_right",
    "turn_right": "turn_left",
}


def forward_transition(pose: np.ndarray, action: str) -> np.ndarray:
    engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
    engine.reset(np.asarray(pose, dtype=float))
    return engine.step(action)


def pose_error(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return {
        "matrix_max_abs": float(np.max(np.abs(difference))),
        "translation_l2": float(np.linalg.norm(difference[:3, 3])),
        "rotation_fro": float(np.linalg.norm(difference[:3, :3])),
    }


def forward_validated_predecessors(
    current_pose: np.ndarray,
    detector: Any | None,
) -> list[dict[str, Any]]:
    """Return legal predecessor proposals for all formal forward actions.

    ``current_pose`` is the desired next state.  A proposal has three stages:
    inverse proposal, formal forward replay, and forward-direction collision
    check.  This deliberately does *not* apply target-only projective
    constraints; callers own state-quality policy.
    """
    current = np.asarray(current_pose, dtype=float)
    current_key = state_key(current)
    rows: list[dict[str, Any]] = []
    for action in ACTIONS:
        predecessor = forward_transition(current, INVERSE_ACTION[action])
        replay = forward_transition(predecessor, action)
        replay_error = pose_error(replay, current)
        state_key_match = state_key(replay) == current_key
        collision = None
        collision_reject = False
        if action in TRANSLATION_ACTIONS and detector is not None:
            collision = detector.check_collision(
                current[:3, 3], previous_position=predecessor[:3, 3]
            )
            collision_reject = bool(collision.has_collision)
        reason = None
        if not state_key_match:
            reason = "forward_state_mismatch"
        elif collision_reject:
            reason = "collision_reject"
        rows.append({
            "forward_action": action,
            "inverse_proposal_action": INVERSE_ACTION[action],
            "predecessor": predecessor,
            "forward_replay": replay,
            "forward_state_key_match": state_key_match,
            "forward_pose_error": replay_error,
            "collision": {
                "has_collision": collision_reject,
                "type": collision.collision_type if collision is not None else None,
            },
            "rejection_reason": reason,
        })
    return rows
