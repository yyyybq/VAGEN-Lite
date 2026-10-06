"""System and observation prompts for RoboCasa VLM control.

Uses Active Spatial tags so weights trained on Active Spatial transfer
format priors:

    <think>...</think><action>...</action>
"""

from __future__ import annotations

from typing import Optional, Sequence

VALID_FORMATS = ("free_think", "no_think")

ACTION_FORMAT_DOC = """\
Action space (12-D Panda-Omron OSC, each component in [-1, 1]):
  [eef_dx, eef_dy, eef_dz, eef_droll, eef_dpitch, eef_dyaw,
   gripper_close, base_x, base_y, base_yaw, torso, control_mode]

Meaning:
  eef_dx, eef_dy, eef_dz     end-effector translation
  eef_droll, eef_dpitch, eef_dyaw   end-effector rotation
  gripper_close              1 closes the gripper, 0 opens it
  base_x, base_y, base_yaw   mobile base planar motion
  torso                      torso height
  control_mode               0 = arm, 1 = base (HybridMobileBase)

Emit one or more 12-D vectors inside a single <action> block, one vector
per line. They will be executed in order. Example:
<action>
[0.0200, 0.0000, -0.0100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
</action>
"""


def _format_instruction(format_name: str, max_actions_per_step: int) -> str:
    if format_name == "no_think":
        return (
            f"You may emit up to {max_actions_per_step} 12-D action vectors "
            "per turn. Respond with:\n<action>...</action>"
        )
    return (
        f"You may emit up to {max_actions_per_step} 12-D action vectors "
        "per turn. First reason, then act. Respond with:\n"
        "<think>...</think><action>...</action>"
    )


def system_prompt(
    format_name: str = "free_think",
    max_actions_per_step: int = 8,
    include_wrist_image: bool = False,
) -> str:
    """Static system prompt describing the kitchen VLA setup."""
    if format_name not in VALID_FORMATS:
        raise ValueError(f"Unknown format {format_name!r}. Valid: {VALID_FORMATS}")

    cameras = (
        "You see the third-person agent-view camera and the wrist (eye-in-hand) camera."
        if include_wrist_image
        else "You see the third-person agent-view camera of the kitchen."
    )
    return (
        "You are a vision-language-action policy controlling a Panda arm on an "
        "Omron mobile base inside a RoboCasa kitchen.\n"
        f"{cameras} Follow the language instruction and complete the household "
        "manipulation task.\n\n"
        f"{ACTION_FORMAT_DOC}\n"
        f"{_format_instruction(format_name, max_actions_per_step)}"
    )


def init_observation_template(
    observation: str,
    instruction: str,
    extra: str = "",
) -> str:
    """First-turn user observation (image placeholder + language)."""
    parts = [
        observation,
        f"Task: {instruction.strip() or '(no language instruction)'}",
    ]
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def action_template(
    observation: str,
    instruction: str,
    valid_actions: Optional[Sequence[str]] = None,
    extra: str = "",
) -> str:
    """Follow-up observation after a step."""
    parts = [
        observation,
        f"Task: {instruction.strip() or '(no language instruction)'}",
    ]
    if valid_actions:
        parts.append(f"Previous executed actions: {len(valid_actions)}")
    if extra:
        parts.append(extra)
    return "\n".join(parts)
