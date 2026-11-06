# Active Spatial Intelligence Environment Prompts
# This file defines system prompts and format configurations for the active spatial navigation task.

from typing import Sequence

FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><action>...</action>",
        "example": """<think>I can see I'm looking at a room with furniture. The target is to reach the front view of the chair. I need to move forward one step.</think><action>move_forward|</action>"""
    },
    "no_think": {
        "description": "You should provide only your action.",
        "format": "<action>...</action>",
        "example": """<action>move_forward|</action>"""
    },
    "grounding": {
        "description": "You should first describe what you observe, then reason about the actions needed, and finally provide your action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>",
        "example": """<think><observation>I'm in a living room. There's a sofa in front of me, a coffee table to my left, and the target chair is visible in the distance to my right. I appear to be viewing from an oblique angle.</observation><reasoning>To reach the front view of the chair, I need to turn right first to face it directly.</reasoning></think><action>turn_right|</action>"""
    },
    "worldmodeling": {
        "description": "You should first reason about your actions and predict the expected outcome, then provide your action.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": """<think><reasoning>The target is to reach the front view of the chair. Currently I can see the chair from the side. I need to move forward to get closer.</reasoning><prediction>After moving forward, I should be closer to the chair and can then adjust my angle.</prediction></think><action>move_forward|</action>"""
    },
    "grounding_worldmodeling": {
        "description": "You should describe your observation, reason about actions, predict the outcome, then provide your action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": """<think><observation>I'm in a bedroom looking at a desk with a lamp. The target chair is behind me based on the task description.</observation><reasoning>I need to turn around to locate the chair first.</reasoning><prediction>After turning right, I expect to see the chair and can then approach it directly.</prediction></think><action>turn_right|</action>"""
    },
    "free_think_fwd_first": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><action>...</action>",
        "example": """<think>The target object appears to be ahead and to the right. I will move forward to close the distance first, then adjust orientation.</think><action>move_forward|move_forward|turn_right|</action>"""
    }
}

# Round 12 freeze (the user's 3.1.1 of 2026-10-22): what the format line shows between the action tags.
# "placeholder" (default) is the text every model up to round 12 trained with, <action>action1|action2|...|</action>;
# RL runs copied those names into their answers (cluster_runs/FORMAT_DRIFT_R12.md).  "action_names" shows real
# action names, <action>move_forward|turn_left|</action> (move_forward alone when one action is allowed per step);
# both exist in every action space.  The system prompt's "Response format" line shows the same.
# Group G of matrix v3 ("point first, then act"; the user's instruction of 2026-11-06, part 4): a short targets
# segment before the action, <targets>A: (x,y); B: (x,y)</targets><action>move_forward|turn_left|</action>: where each
# object the task names is in the current image (A, B, ... as the format block lists them, in the task text's words),
# or "none" for an object that is not clearly visible; the action part shows real action names as "action_names".
# "targets_pixel": pixels of the 256 x 256 image; "targets_norm1000": 0 to 1000 across the image (Qwen-VL's grounding
# convention).  Only with prompt_format no_think; pointing.py reads and judges the segment.
TARGET_FORMATS = {"targets_pixel": "pixel", "targets_norm1000": "norm1000"}  # format example -> coordinates
FORMAT_EXAMPLES = ("placeholder", "action_names") + tuple(TARGET_FORMATS)
SLOT_LETTERS = ("A", "B", "C", "D")
_TARGET_TEXT = {
    "targets_pixel": {"where": "pixels of the 256 x 256 image", "example": "(120,88)",
                      "point": "(x,y) is a pixel on the object in the 256 x 256 image, x from the left edge and y "
                               "from the top edge"},
    "targets_norm1000": {"where": "0 to 1000 across the image", "example": "(469,344)",
                         "point": "(x,y) is a point on the object, x and y from 0 to 1000 across the image, x from "
                                  "the left edge and y from the top edge"},
}


def action_format_line(action_sep: str = "|", max_actions: int = 5, format_example: str = "placeholder",
                       n_slots: int = 2) -> str:
    """The format line of the action tags for the format example setting (a targets format: the targets segment for
    ``n_slots`` objects first)."""
    if format_example == "placeholder":
        return f"<action>action1{action_sep}action2{action_sep}...{action_sep}</action>"
    if format_example not in FORMAT_EXAMPLES:
        raise ValueError(f"format example must be one of {FORMAT_EXAMPLES}, not {format_example!r}")
    names = ("move_forward", "turn_left")[: 1 if int(max_actions) <= 1 else 2]
    line = "<action>" + "".join(n + action_sep for n in names) + "</action>"
    return targets_line(n_slots) + line if format_example in TARGET_FORMATS else line


def targets_line(n_slots: int = 2) -> str:
    """The targets segment of the format line: one "(x,y)" per object the task names."""
    return "<targets>" + "; ".join(f"{s}: (x,y)" for s in SLOT_LETTERS[: max(1, int(n_slots))]) + "</targets>"


def targets_example(format_example: str, n_slots: int = 2, action_sep: str = "|") -> str:
    """The example answer of a targets format: A at a point, the other objects not in view."""
    entries = ([f"A: {_TARGET_TEXT[format_example]['example']}"]
               + [f"{s}: none" for s in SLOT_LETTERS[1: max(1, int(n_slots))]])
    return "<targets>" + "; ".join(entries) + f"</targets><action>move_forward{action_sep}</action>"


def targets_description(format_example: str) -> str:
    """The system prompt's description of a targets format."""
    return ("Before your action, give the position of each object the task names, in the order the format block "
            f"lists them: {_TARGET_TEXT[format_example]['point']}; write none for an object you cannot see clearly.")


def targets_format_block(format_example: str, max_actions: int = 5, action_sep: str = "|", add_example: bool = True,
                         target_refs: Sequence[str] = ()) -> str:
    """A user turn's format block of a targets format: the format line for the task's objects, which object each
    letter is (the task text's words, pointing.target_refs) and, on the first turn, the example."""
    refs = [str(r) for r in target_refs][: len(SLOT_LETTERS)]
    n = len(refs) or 2
    prompt = (f"Respond with the targets ({_TARGET_TEXT[format_example]['where']}), then the action:\n"
              f"{action_format_line(action_sep, max_actions, format_example, n_slots=n)}\n")
    if refs:
        prompt += "; ".join(f"{s} is {r}" for s, r in zip(SLOT_LETTERS, refs)) + ".\n"
    prompt += f"\nYou can take up to {max_actions} actions per step.\n"
    if add_example:
        prompt += "\nExample: " + targets_example(format_example, n, action_sep)
    return prompt


def _response_format(format_config, action_sep: str, max_actions: int, format_example: str) -> str:
    """The system prompt's "Response format" text: the format's own (placeholder) or with real action names (a
    targets format: the targets segment first; prompt_format no_think only)."""
    text = format_config["format"]
    if format_example == "placeholder":
        return text
    if format_example in TARGET_FORMATS and text != "<action>...</action>":
        raise ValueError(f"the {format_example} format example goes with prompt_format no_think only")
    return text.replace("<action>...</action>", action_format_line(action_sep, max_actions, format_example))


# Action descriptions for the agent.
# NOTE: kept as a legacy single-action constant for backward compatibility
# with any external code that imports it. The runtime system prompt now
# uses `_action_descriptions(max_actions)` below so the instruction matches
# the actual `max_actions_per_step` configured for the environment.
ACTION_DESCRIPTIONS = """
Available actions (output EXACTLY ONE action per response, end with |):
- move_forward: Move the camera forward by a fixed step
- move_backward: Move the camera backward by a fixed step  
- turn_left: Rotate the camera left (yaw) by a fixed angle
- turn_right: Rotate the camera right (yaw) by a fixed angle
- look_up: Tilt the camera upward (pitch) by a fixed angle
- look_down: Tilt the camera downward (pitch) by a fixed angle
- done: Signal that you have reached the target pose (terminates episode)
"""


def _action_descriptions(max_actions: int = 1, action_sep: str = "|",
                         action_space: str = "legacy",
                         enable_done: bool = True) -> str:
    """Action list whose header is parameterised by `max_actions_per_step`.

    For `max_actions == 1` and `action_space == "legacy"` and `enable_done`
    True, the wording is byte-equivalent to the legacy `ACTION_DESCRIPTIONS`
    constant, so older experiments (e.g. v11) that rely on the previous
    prompt remain unchanged.

    action_space:
        "legacy" -> forward/back + turn_left/right + look_up/down (+ done)
        "strafe" -> forward/back + left/right + turn_left/right (+ done)
                    (no pitch; lateral translation instead)
    """
    if max_actions <= 1:
        header = "Available actions (output EXACTLY ONE action per response, end with |):"
    else:
        header = (
            f"Available actions (output up to {max_actions} actions per response, "
            f"separated by '{action_sep}', end with '{action_sep}'):"
        )

    if action_space == "strafe":
        body_lines = [
            "- move_forward: Move the camera forward by a fixed step",
            "- move_backward: Move the camera backward by a fixed step",
            "- move_left: Strafe the camera left (lateral translation) by a fixed step",
            "- move_right: Strafe the camera right (lateral translation) by a fixed step",
            "- turn_left: Rotate the camera left (yaw) by a fixed angle",
            "- turn_right: Rotate the camera right (yaw) by a fixed angle",
        ]
    else:  # legacy
        body_lines = [
            "- move_forward: Move the camera forward by a fixed step",
            "- move_backward: Move the camera backward by a fixed step",
            "- turn_left: Rotate the camera left (yaw) by a fixed angle",
            "- turn_right: Rotate the camera right (yaw) by a fixed angle",
            "- look_up: Tilt the camera upward (pitch) by a fixed angle",
            "- look_down: Tilt the camera downward (pitch) by a fixed angle",
        ]
    if enable_done:
        body_lines.append("- done: Signal that you have reached the target pose (terminates episode)")
    body = "\n".join(body_lines)
    return f"\n{header}\n{body}\n"


def _actual_reward_lines(kwargs) -> str:
    """The reward terms the env applies, from its settings (defaults: the v10 S0 config).  Round 9: the legacy
    lines promise a progress reward that S0 does not give."""
    def num(key, default):
        return float(kwargs.get(key, default))

    lines = []
    n_invalid = int(kwargs.get("max_consecutive_invalid_actions", 3))
    lines.append(f"- Invalid format (no action can be read): {num('invalid_format_penalty', -0.1):g}"
                 + (f"; {n_invalid} in a row end the episode" if n_invalid > 0 else ""))
    if num("format_reward", 0.0) > 0.0:
        lines.append(f"- Format correct: +{num('format_reward', 0.0):g}")
    if kwargs.get("enable_step_penalty", True) and num("step_penalty", -0.01) != 0.0:
        lines.append(f"- Each response (step): {num('step_penalty', -0.01):g}")
    n_blocked = int(kwargs.get("max_consecutive_collisions", 3))
    lines.append(f"- Each blocked move (the camera stays where it was): {num('collision_penalty', -0.15):g}"
                 + (f"; {n_blocked} blocked moves in a row end the episode "
                    f"({num('consecutive_collision_penalty', -0.5):g} more)" if n_blocked > 0 else ""))
    if kwargs.get("enable_low_info_frame_check", True):
        n_low = int(kwargs.get("max_consecutive_low_info_frames", 3))
        lines.append(f"- Nearly uniform view (e.g. facing a wall): {num('low_info_frame_penalty', -0.2):g}"
                     + (f"; {n_low} in a row end the episode" if n_low > 0 else ""))
    lines.append(f"- Reaching the requested view: +{num('success_reward', 1.0):g}, and the episode ends")
    lines.append("- There is no reward for partial progress")
    return "\n".join(lines)


def system_prompt(**kwargs):
    """Generate the system prompt for the Active Spatial environment.

    ``reward_text``: "legacy" (default, the text R1 and v10 trained with) or "actual" (round 9, next data version:
    the reward terms the env really applies, from the env settings passed in)."""
    selected_format = kwargs.get("format", "free_think")
    step_translation = kwargs.get("step_translation", 0.1)
    step_rotation_deg = kwargs.get("step_rotation_deg", 5.0)
    # New: thread `max_actions_per_step` and reward scalars from env config.
    # Defaults preserve the legacy single-action text/rewards exactly so any
    # caller that does not pass these kwargs (older experiments, eval scripts)
    # produces a prompt identical to the previous version.
    max_actions = int(kwargs.get("max_actions_per_step", 1))
    action_sep = kwargs.get("action_sep", "|")
    format_reward = float(kwargs.get("format_reward", 0.2))
    invalid_format_penalty = float(kwargs.get("invalid_format_penalty", -0.1))
    success_reward = float(kwargs.get("success_reward", 1.0))
    task_type = str(kwargs.get("task_type", "") or "")
    action_space = kwargs.get("action_space", "legacy")
    enable_done = bool(kwargs.get("enable_explicit_done", True))

    format_config = FORMAT_CONFIGS.get(selected_format, FORMAT_CONFIGS["free_think"])
    format_example = str(kwargs.get("format_example", "placeholder"))
    if format_example in TARGET_FORMATS:  # group G of matrix v3: the targets segment's description and example
        format_config = dict(format_config, description=targets_description(format_example),
                             example=targets_example(format_example, 2, action_sep))
    response_format = _response_format(format_config, action_sep, max_actions, format_example)

    actions_block = _action_descriptions(
        max_actions=max_actions,
        action_sep=action_sep,
        action_space=action_space,
        enable_done=enable_done,
    )

    if max_actions <= 1:
        hint2 = "Output exactly ONE action per response; you will receive feedback after each action"
    else:
        hint2 = (
            f"You may output up to {max_actions} actions per response "
            f"(separated by '{action_sep}'); feedback is provided after each step"
        )

    if format_reward > 0.0:
        format_reward_line = f"- Format correct: +{format_reward:g}"
    else:
        format_reward_line = (
            "- Valid format is required but gives no positive reward; "
            f"invalid format: {invalid_format_penalty:g}"
        )

    if task_type == "projective_relations":
        task_strategy = (
            "For projective left/right tasks, left and right describe horizontal ordering "
            "in the rendered image; they do not map directly to turn_left/turn_right. "
            "When both objects are already visible, first try lateral translation "
            "(move_left/move_right), and rotate only as needed to keep both objects visible. "
            "Do not alternate repeated inverse turns. If collision feedback says a translation "
            "is blocked, do not repeat it; back off or rotate before choosing a different path."
        )
        progress_reward_line = (
            "- Gate-aligned projective progress: reward follows bbox inside-frame fraction "
            "and a smooth image-relation margin target centered at 12 px; an off-frame object "
            "cannot receive high potential"
        )
        success_reward_line = (
            f"- Canonical projective success (both objects visible and in frame, with the "
            f"requested relation and margin): +{success_reward:g}"
        )
    else:
        task_strategy = (
            "For ordinary target-view tasks, rotate to face the relevant object or view "
            "direction before translating. For relation tasks such as occlusion, follow the "
            "task-specific relation: the correct viewpoint may require moving around an "
            "occluder rather than directly approaching the hidden target."
        )
        progress_reward_line = (
            "- Progress toward target pose: continuous reward based on distance and "
            "orientation improvement"
        )
        success_reward_line = f"- Reaching target pose: +{success_reward:g}"

    if str(kwargs.get("reward_text", "legacy")) == "actual":
        reward_lines = _actual_reward_lines(kwargs)
    else:
        reward_lines = f"{format_reward_line}\n{progress_reward_line}\n{success_reward_line}"

    base_prompt = f"""You are a spatial navigation agent in a 3D indoor environment. Your task is to navigate a camera to reach a specific target view of an object.
{actions_block}
Step sizes: translation = {step_translation:.2f} meters, rotation = {step_rotation_deg:.1f} degrees.

{format_config['description']}

Response format: {response_format}

Example:
{format_config['example']}

Rewards:
{reward_lines}

Hints:
1. Pay attention to the target object and the requested view (front, back, left, right, etc.)
2. {hint2}
3. Consider both position and orientation when navigating
4. Look around if you're unsure of the target location
5. Strategy: {task_strategy}
"""

    # Override hints for forward-first strategy (designed for models that over-rotate)
    if selected_format == "free_think_fwd_first":
        base_prompt = f"""You are a spatial navigation agent in a 3D indoor environment. Your task is to navigate a camera to reach a specific target view of an object.
{actions_block}
Step sizes: translation = {step_translation:.2f} meters, rotation = {step_rotation_deg:.1f} degrees.
ACTION NAMES (exact spelling required, lowercase with underscores): move_forward, move_backward, move_left, move_right, turn_left, turn_right

{format_config['description']}

Response format: {response_format}

Example:
{format_config['example']}

Rewards:
{reward_lines}

Hints:
1. Pay attention to the target object and the requested view (front, back, left, right, etc.)
2. {hint2}
3. Consider both position and orientation when navigating
4. Strategy: {task_strategy}
5. AVOID spinning in circles: if the reward is not improving after repeated turns, switch to move_forward, move_backward, or move_left/move_right to explore a new position.
"""

    return base_prompt


def init_observation_template(**kwargs):
    """Generate the initial observation prompt template."""
    observation = kwargs.get("observation", "")
    task_prompt = kwargs.get("task_prompt", "Navigate to the target view.")
    spatial_prior = kwargs.get("spatial_prior", "")  # Multi-frame spatial prior text
    
    template = ""
    
    # Add spatial prior section if provided
    if spatial_prior:
        template += f"""[Spatial Context]:
{spatial_prior}

"""
    
    template += f"""[Initial Observation]:
{observation}
Task: {task_prompt}

Navigate to reach the specified view of the target object. Use the available actions to position and orient the camera correctly.
"""
    return template


def action_template(**kwargs):
    """Generate the action observation template for subsequent steps."""
    observation = kwargs.get("observation", "")
    env_feedback = kwargs.get("env_feedback", "")
    task_prompt = kwargs.get("task_prompt", "")
    
    template = f"""[Observation]:
{observation}
"""
    if env_feedback:
        template += f"Environment Feedback: {env_feedback}\n"
    if task_prompt:
        template += f"Task: {task_prompt}\n"
    
    return template


# Format prompt functions for different prompt formats
def format_prompt_free_think(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    action_line = action_format_line(action_sep, max_actions, kwargs.get("format_example", "placeholder"))
    
    prompt = f"""Respond in the following format:
<think>Your reasoning process</think>
{action_line}

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['free_think']['example']}"
    return prompt


def format_prompt_no_think(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    if kwargs.get("format_example") in TARGET_FORMATS:  # group G of matrix v3: the targets segment first
        return targets_format_block(kwargs["format_example"], max_actions, action_sep, add_example,
                                    kwargs.get("target_refs") or ())
    action_line = action_format_line(action_sep, max_actions, kwargs.get("format_example", "placeholder"))
    
    prompt = f"""Respond with only the action:
{action_line}

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['no_think']['example']}"
    return prompt


def format_prompt_grounding(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    action_line = action_format_line(action_sep, max_actions, kwargs.get("format_example", "placeholder"))
    
    prompt = f"""Respond in the following format:
<think><observation>What you see</observation><reasoning>Your reasoning</reasoning></think>
{action_line}

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['grounding']['example']}"
    return prompt


def format_prompt_worldmodeling(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    action_line = action_format_line(action_sep, max_actions, kwargs.get("format_example", "placeholder"))
    
    prompt = f"""Respond in the following format:
<think><reasoning>Your reasoning</reasoning><prediction>Expected outcome</prediction></think>
{action_line}

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['worldmodeling']['example']}"
    return prompt


def format_prompt_grounding_worldmodeling(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    action_line = action_format_line(action_sep, max_actions, kwargs.get("format_example", "placeholder"))
    
    prompt = f"""Respond in the following format:
<think><observation>What you see</observation><reasoning>Your reasoning</reasoning><prediction>Expected outcome</prediction></think>
{action_line}

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['grounding_worldmodeling']['example']}"
    return prompt


# Mapping of prompt formats to their corresponding format functions
format_prompt = {
    "free_think": format_prompt_free_think,
    "no_think": format_prompt_no_think,
    "grounding": format_prompt_grounding,
    "worldmodeling": format_prompt_worldmodeling,
    "grounding_worldmodeling": format_prompt_grounding_worldmodeling,
}
