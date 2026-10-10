"""
SFT Conversation Formatter.

Converts a Trajectory (sequence of TrajStep objects) + rendered images into
the multi-turn conversation format used for VLM supervised fine-tuning.

Output format (per trajectory)
-------------------------------
{
    "id": "sft_000001",
    "source_item_idx": 42,
    "scene_id": "0267_840790",
    "task_type": "absolute_positioning",
    "task_description": "Navigate to be 1.5m from the sofa...",
    "trajectory_steps": 8,
    "total_actions": 18,
    "initial_score": 0.12,
    "final_score": 0.97,
    "success": true,
    "conversations": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "[Initial Observation]:\\n<image>\\n...", "image_path": "images/..."},
        {"role": "assistant", "content": "<think>...</think>\\n<action>move_forward|turn_left|</action>"},
        {"role": "user",      "content": "[Observation]:\\n<image>\\n...", "image_path": "images/..."},
        {"role": "assistant", "content": "<think>...</think>\\n<action>done|</action>"}
    ]
}
"""

import hashlib
import json
import numpy as np
from typing import Any, Dict, List, Optional
from PIL import Image

from .path_finder import TrajStep, Trajectory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _c2w_to_pose_str(c2w: np.ndarray) -> str:
    """Format a 4x4 c2w matrix as a human-readable 6-DoF pose string."""
    from vagen.envs.active_spatial.utils import c2w_extrinsic_to_se3, format_pose6_deg
    return format_pose6_deg(c2w_extrinsic_to_se3(c2w))


def _build_task_description(item: Dict[str, Any]) -> str:
    """Extract or reconstruct the task description from a pipeline item."""
    desc = item.get("task_description") or item.get("description") or ""
    if desc:
        return desc
    obj = item.get("object_label", "object")
    preset = item.get("preset", "front")
    dist = item.get("distance")
    if dist:
        return f"Move the camera to the {preset} view of the {obj}, about {dist:.2f} meters away."
    return f"Move the camera to the {preset} view of the {obj}."


# ---------------------------------------------------------------------------
# Think-block generation
# ---------------------------------------------------------------------------

_ACTION_VERB: Dict[str, str] = {
    "move_forward":  "move closer",
    "move_backward": "move away",
    "move_left":     "move left",
    "move_right":    "move right",
    "turn_left":     "rotate left",
    "turn_right":    "rotate right",
    "look_up":       "tilt upward",
    "look_down":     "tilt downward",
}


def _make_think(
    step: TrajStep,
    include_scores: bool = False,
    task_type: str = "",
) -> str:
    """Generate a short but meaningful reasoning block for one LLM turn.

    Args:
        step: The trajectory step to describe.
        include_scores: If True, embed numerical potential-field scores in the
            text.  Keep False (default) to avoid leaking oracle information into
            the SFT supervision signal.
    """
    delta = step.score_after - step.score_before

    if include_scores:
        score_info = (
            f"Current score: {step.score_before:.3f} "
            f"(position: {step.pos_score_before:.3f}, "
            f"orientation: {step.ori_score_before:.3f}). "
        )
    else:
        score_info = ""

    if task_type == "projective_relations":
        focus = "keeping both objects in frame while establishing the requested image relation"
    elif task_type == "fov_inclusion":
        focus = "keeping both target objects fully and safely inside the frame"
    elif step.pos_score_before < 0.7:
        focus = "positioning (reaching the target region)"
    else:
        focus = "orientation (facing the target correctly)"

    # Describe the action sequence more accurately
    if not step.actions:
        action_desc = "take no action"
    elif len(step.actions) == 1:
        verb = _ACTION_VERB.get(step.actions[0], step.actions[0].replace("_", " "))
        action_desc = verb
    else:
        # Summarise repeated actions (e.g. ["turn_left"]*3 → "rotate left 3 times")
        # and list distinct action groups
        groups: List[str] = []
        i = 0
        while i < len(step.actions):
            act = step.actions[i]
            count = 1
            while i + count < len(step.actions) and step.actions[i + count] == act:
                count += 1
            verb = _ACTION_VERB.get(act, act.replace("_", " "))
            groups.append(f"{verb}" if count == 1 else f"{verb} {count} times")
            i += count
        if len(groups) == 1:
            action_desc = groups[0]
        else:
            action_desc = ", then ".join(groups[:-1]) + f", and then {groups[-1]}"

    # Beam search may intentionally take a step that lowers the immediate score
    # to escape a local plateau and reach a better position later.  Describing
    # this as "improve by -0.167" is contradictory and confuses training.
    # Instead, frame it as a deliberate repositioning move.
    if delta < -1e-6:
        think = (
            f"{score_info}"
            f"The main challenge is {focus}. "
            f"I need to reposition to find a better approach angle; "
            f"I will {action_desc}."
        )
    else:
        if include_scores and delta > 1e-6:
            improvement = f" This should improve the score by {delta:+.3f} to {step.score_after:.3f}."
        else:
            improvement = ""
        think = (
            f"{score_info}"
            f"The main challenge is {focus}. "
            f"I will {action_desc} to make progress.{improvement}"
        )
    return think


def _make_done_think(trajectory: Trajectory, include_scores: bool = False) -> str:
    """Generate thinking for the final 'done' action.

    Args:
        include_scores: If True, embed the final numerical score.  The hard-
            coded success threshold is intentionally omitted even in this mode
            to avoid leaking internal oracle constants into training data.
    """
    if include_scores:
        return (
            f"My current score is {trajectory.final_score:.3f}. "
            f"I have successfully reached the target position and orientation. "
            f"I will issue 'done' to complete the task."
        )
    return (
        "I have reached the target position and orientation. "
        "I will issue 'done' to complete the task."
    )


def _trajectory_trace(item: Dict[str, Any], trajectory: Trajectory) -> List[Dict[str, Any]]:
    """Build frame-aligned, non-policy metadata for reward/path auditing."""
    from vagen.envs.active_spatial.canonical_task_metrics import (
        GATE_ALIGNED_SHAPING_SCORE_VERSION,
        score_canonical_task,
        uses_canonical_backend,
    )

    if trajectory.steps:
        poses = [trajectory.steps[0].c2w_before]
    else:
        poses = [np.asarray(item["init_camera"]["extrinsics"], dtype=float)]
    poses.extend(step.c2w_after for step in trajectory.steps)
    scores = [float(trajectory.initial_score)]
    scores.extend(float(step.score_after) for step in trajectory.steps)

    trace: List[Dict[str, Any]] = []
    canonical = uses_canonical_backend(item)
    for frame_index, (pose, score) in enumerate(zip(poses, scores)):
        pose = np.asarray(pose, dtype=float)
        step = trajectory.steps[frame_index - 1] if frame_index else None
        row: Dict[str, Any] = {
            "frame_index": frame_index,
            "actions_from_previous": list(step.actions) if step else [],
            "pose_c2w": pose.tolist(),
            "position_xyz": pose[:3, 3].tolist(),
            "forward_xyz": pose[:3, 2].tolist(),
            "score": score,
            "delta_score": None if frame_index == 0 else score - scores[frame_index - 1],
            "position_score": None if step is None else float(step.pos_score_after),
            "orientation_score": None if step is None else float(step.ori_score_after),
        }
        if canonical:
            metric = score_canonical_task(item, pose)
            shaping = float(metric.get("shaping_score", metric["score"]))
            if not np.isclose(shaping, score, atol=1e-9, rtol=0.0):
                raise ValueError(
                    f"canonical trace/search score mismatch at frame {frame_index}: "
                    f"{shaping} != {score}"
                )
            row.update({
                "score_backend": metric["metric_version"],
                "score_version": metric.get(
                    "shaping_score_version", GATE_ALIGNED_SHAPING_SCORE_VERSION
                ),
                "canonical_success": bool(metric["success"]),
                "historical_visual_score": float(metric["score"]),
                "gates": dict(metric.get("gates") or {}),
                "shaping_components": dict(metric.get("shaping_components") or {}),
                "inside_frame_fraction_min": metric.get("inside_frame_fraction_min"),
                "relation_margin_px": metric.get("relation_margin_px"),
                "center_margin_px": metric.get("center_margin_px"),
            })
        else:
            row.update({
                "score_backend": "spatial_potential_field",
                "score_version": "legacy_total_score",
                "canonical_success": None,
            })
        if step is not None and frame_index - 1 < len(trajectory.runtime_reward_traces):
            reward_trace = trajectory.runtime_reward_traces[frame_index - 1]
            row["runtime_reward_trace"] = reward_trace
            row["runtime_reward"] = (
                (reward_trace.get("accounting") or {}).get("actual_total_reward")
                if reward_trace else None
            )
        trace.append(row)
    return trace


def _primitive_trace(
    item: Dict[str, Any],
    trajectory: Trajectory,
    step_translation: float,
    step_rotation_deg: float,
) -> List[Dict[str, Any]]:
    """Expand bundled turns into atomic poses for score-guidance diagnostics.

    RGB remains frame/turn aligned.  This trace exists so a five-action policy
    response still exposes the score and XY path after every primitive action.
    """
    from .path_finder import simulate_action
    from vagen.envs.active_spatial.canonical_task_metrics import (
        GATE_ALIGNED_SHAPING_SCORE_VERSION,
        score_canonical_task,
        uses_canonical_backend,
    )

    if not uses_canonical_backend(item):
        return []
    pose = (
        np.asarray(trajectory.steps[0].c2w_before, dtype=float)
        if trajectory.steps
        else np.asarray(item["init_camera"]["extrinsics"], dtype=float)
    )

    def metric_row(
        primitive_index: int,
        turn_index: int,
        action: Optional[str],
        current_pose: np.ndarray,
    ) -> Dict[str, Any]:
        metric = score_canonical_task(item, current_pose)
        return {
            "primitive_index": primitive_index,
            "turn_index": turn_index,
            "action_from_previous": action,
            "pose_c2w": current_pose.tolist(),
            "position_xyz": current_pose[:3, 3].tolist(),
            "forward_xyz": current_pose[:3, 2].tolist(),
            "score": float(metric.get("shaping_score", metric["score"])),
            "score_backend": metric["metric_version"],
            "score_version": metric.get(
                "shaping_score_version", GATE_ALIGNED_SHAPING_SCORE_VERSION
            ),
            "canonical_success": bool(metric["success"]),
            "historical_visual_score": float(metric["score"]),
            "gates": dict(metric.get("gates") or {}),
            "shaping_components": dict(metric.get("shaping_components") or {}),
            "inside_frame_fraction_min": metric.get("inside_frame_fraction_min"),
            "relation_margin_px": metric.get("relation_margin_px"),
            "center_margin_px": metric.get("center_margin_px"),
        }

    rows = [metric_row(0, 0, None, pose)]
    primitive_index = 0
    for turn_index, step in enumerate(trajectory.steps):
        for action in step.actions:
            pose = simulate_action(
                pose,
                action,
                step_translation=step_translation,
                step_rotation_deg=step_rotation_deg,
            )
            primitive_index += 1
            rows.append(metric_row(primitive_index, turn_index, action, pose))
        if not np.allclose(pose, step.c2w_after, atol=1e-6, rtol=0.0):
            raise ValueError(f"primitive expansion drifted at turn {turn_index}")
    return rows


# ---------------------------------------------------------------------------
# Public formatter
# ---------------------------------------------------------------------------

def format_trajectory(
    item: Dict[str, Any],
    trajectory: Trajectory,
    image_paths: List[str],
    sft_id: str,
    prompt_format: str = "free_think",
    add_think: bool = True,
    include_scores: bool = False,
    force_no_done: bool = False,
    step_translation: float = 0.3,
    step_rotation_deg: float = 30.0,
    action_space: str = "legacy",
    enable_explicit_done: bool = True,
    max_actions_per_step: int = 5,
) -> Dict[str, Any]:
    """Convert a Trajectory + image paths into an SFT conversation record.

    Args:
        item: Original pipeline JSONL item (provides task description, scene_id, …).
        trajectory: Found trajectory (list of TrajStep objects).
        image_paths: Relative paths to saved images, one per step (step 0 = init view,
                     step k = view after step k-1's actions).
                     Length should be len(trajectory.steps) + 1.
        sft_id: Unique string identifier for this record.
        prompt_format: One of 'free_think', 'no_think', 'grounding', 'worldmodeling'.
        add_think: Whether to include <think> blocks.
        include_scores: Whether to include numerical scores in thinking.
        force_no_done: If True, do NOT append the final 'done' assistant turn even
            when trajectory.success is True.  Use for partial trajectories that did
            not reach the goal — ending the conversation without 'done' teaches the
            model to keep navigating rather than declare premature success.

    Returns:
        A dict with 'conversations', 'image_paths', and metadata.
    """
    from vagen.envs.active_spatial.prompt import (
        system_prompt,
        init_observation_template,
        action_template,
        format_prompt as format_prompt_map,
    )

    task_description = _build_task_description(item)
    scene_id = item.get("scene_id", "")
    task_type = item.get("task_type", "")

    # Retrieve the format-specific prompt builder
    fmt_fn = format_prompt_map.get(prompt_format, format_prompt_map["free_think"])

    conversations: List[Dict[str, Any]] = []

    # ── System message ───────────────────────────────────────────────────────
    sys_text = system_prompt(
        format=prompt_format,
        step_translation=step_translation,
        step_rotation_deg=step_rotation_deg,
        action_space=action_space,
        enable_explicit_done=enable_explicit_done,
        max_actions_per_step=max_actions_per_step,
    )
    conversations.append({"role": "system", "content": sys_text})

    # ── Initial observation (user turn 0) ────────────────────────────────────
    init_c2w = (
        trajectory.steps[0].c2w_before
        if trajectory.steps
        else np.array(item["init_camera"]["extrinsics"])
    )
    init_pose_str = _c2w_to_pose_str(init_c2w)
    init_obs = init_observation_template(
        observation=f"<image>\nCurrent camera pose: {init_pose_str}",
        task_prompt=task_description,
    )
    init_obs += "\n" + fmt_fn(max_actions_per_step=max_actions_per_step, action_sep="|", add_example=True)

    init_turn: Dict[str, Any] = {"role": "user", "content": init_obs}
    if image_paths:
        init_turn["image_path"] = image_paths[0]
    conversations.append(init_turn)

    # ── Interleaved (assistant → user) turns ────────────────────────────────
    for i, step in enumerate(trajectory.steps):
        actions_blob = "|".join(step.actions) + "|"

        # Assistant response
        if add_think and prompt_format in ("free_think", "grounding", "worldmodeling",
                                           "grounding_worldmodeling"):
            think_text = _make_think(
                step, include_scores=include_scores, task_type=task_type
            )
            response = f"<think>{think_text}</think>\n<action>{actions_blob}</action>"
        else:
            response = f"<action>{actions_blob}</action>"

        conversations.append({"role": "assistant", "content": response})

        # Next user observation (image after this step's actions).  Do not add
        # an observation after the final movement unless it is followed by an
        # explicit ``done`` label.  Auto-terminated R1 episodes otherwise end
        # in an unpaired user turn, which multimodal SFT loaders may reject (or
        # retain as pure prompt tokens with no supervised response).  The final
        # rendered frame remains in ``image_paths`` for trajectory auditing and
        # visualization.
        needs_next_action = i + 1 < len(trajectory.steps)
        needs_explicit_done = (
            i + 1 == len(trajectory.steps)
            and trajectory.success
            and not force_no_done
            and enable_explicit_done
        )
        if not (needs_next_action or needs_explicit_done):
            continue

        img_idx = i + 1
        pose_str = _c2w_to_pose_str(step.c2w_after)
        obs_text = action_template(
            observation=f"<image>\nCurrent camera pose: {pose_str}",
            env_feedback="Action executed.",
            task_prompt=task_description,
        )
        obs_text += "\n" + fmt_fn(max_actions_per_step=max_actions_per_step, action_sep="|", add_example=False)

        user_turn: Dict[str, Any] = {"role": "user", "content": obs_text}
        if img_idx < len(image_paths):
            user_turn["image_path"] = image_paths[img_idx]
        conversations.append(user_turn)

    # ── Final "done" assistant turn ──────────────────────────────────────────
    if trajectory.success and not force_no_done and enable_explicit_done:
        if add_think and prompt_format in ("free_think", "grounding", "worldmodeling",
                                           "grounding_worldmodeling"):
            done_think = _make_done_think(trajectory, include_scores=include_scores)
            done_response = f"<think>{done_think}</think>\n<action>done|</action>"
        else:
            done_response = "<action>done|</action>"
        conversations.append({"role": "assistant", "content": done_response})

    # ── Assemble record ──────────────────────────────────────────────────────
    trace = _trajectory_trace(item, trajectory)
    primitive_trace = _primitive_trace(
        item, trajectory, step_translation, step_rotation_deg
    )
    conversation_image_paths = [
        turn["image_path"]
        for turn in conversations
        if turn.get("role") == "user" and turn.get("image_path")
    ]
    score_versions = sorted({str(row["score_version"]) for row in trace})
    audit = item.get("_sft_audit") or {}
    difficulty = audit.get("difficulty") or {}
    reference_actions = list(audit.get("actions_audit_only") or [])
    optimality_reference = None
    if audit:
        lower = difficulty.get("certified_lower_bound")
        upper = difficulty.get("certificate_upper_bound")
        lower_complete = bool(difficulty.get("lower_bound_complete"))
        optimality_reference = {
            "reference_actions": reference_actions,
            "reference_action_count": len(reference_actions),
            "certified_lower_bound": lower,
            "certificate_upper_bound": upper,
            "lower_bound_complete": lower_complete,
            "shortest_length_certified": bool(
                lower_complete and lower is not None and upper is not None
                and int(lower) == int(upper)
            ),
            "generated_action_gap_to_upper_bound": (
                trajectory.total_actions - int(upper) if upper is not None else None
            ),
        }
    record = {
        "id": sft_id,
        "source_item_idx": trajectory.item_idx,
        "scene_id": scene_id,
        "task_type": task_type,
        "task_description": task_description,
        "trajectory_steps": len(trajectory.steps),
        "total_actions": trajectory.total_actions,
        "initial_score": round(trajectory.initial_score, 6),
        "final_score": round(trajectory.final_score, 6),
        "success": trajectory.success,
        "source_task_id": item.get("task_id"),
        "source_task_sha256": hashlib.sha256(json.dumps({k: v for k, v in item.items() if k != "_sft_audit"}, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "split": item.get("split", "train"),
        "source_versions": {key: item.get(key) for key in ("camera_model_version", "canonical_task_metric_version", "action_protocol_version")},
        "score_contract": {
            "search_score_versions": score_versions,
            "success_source": (
                "canonical_gates"
                if any(row.get("canonical_success") is not None for row in trace)
                else "score_threshold"
            ),
        },
        "trajectory_trace": trace,
        "primitive_trajectory_trace": primitive_trace,
        "terminal_reward_trace": trajectory.terminal_reward_trace,
        "optimality_reference": optimality_reference,
        "conversations": conversations,
        "conversation_image_paths": conversation_image_paths,
        "image_paths": image_paths,
    }
    return record
