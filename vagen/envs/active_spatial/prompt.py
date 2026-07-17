# Active Spatial Intelligence Environment Prompts
# This file defines system prompts and format configurations for the active spatial navigation task.

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


def system_prompt(**kwargs):
    """Generate the system prompt for the Active Spatial environment."""
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
    success_reward = float(kwargs.get("success_reward", 1.0))
    action_space = kwargs.get("action_space", "legacy")
    enable_done = bool(kwargs.get("enable_explicit_done", True))

    format_config = FORMAT_CONFIGS.get(selected_format, FORMAT_CONFIGS["free_think"])

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

    base_prompt = f"""You are a spatial navigation agent in a 3D indoor environment. Your task is to navigate a camera to reach a specific target view of an object.
{actions_block}
Step sizes: translation = {step_translation:.2f} meters, rotation = {step_rotation_deg:.1f} degrees.

{format_config['description']}

Response format: {format_config['format']}

Example:
{format_config['example']}

Rewards:
- Format correct: +{format_reward:g}
- Progress toward target pose: continuous reward based on distance and orientation improvement
- Reaching target pose: +{success_reward:g}

Hints:
1. Pay attention to the target object and the requested view (front, back, left, right, etc.)
2. {hint2}
3. Consider both position and orientation when navigating
4. Look around if you're unsure of the target location
5. Strategy: For ordinary target-view tasks, rotate to face the relevant object or view direction before translating. For relation tasks such as occlusion, follow the task-specific relation: the correct viewpoint may require moving around an occluder rather than directly approaching the hidden target.
"""

    # Override hints for forward-first strategy (designed for models that over-rotate)
    if selected_format == "free_think_fwd_first":
        base_prompt = f"""You are a spatial navigation agent in a 3D indoor environment. Your task is to navigate a camera to reach a specific target view of an object.
{actions_block}
Step sizes: translation = {step_translation:.2f} meters, rotation = {step_rotation_deg:.1f} degrees.
ACTION NAMES (exact spelling required, lowercase with underscores): move_forward, move_backward, move_left, move_right, turn_left, turn_right

{format_config['description']}

Response format: {format_config['format']}

Example:
{format_config['example']}

Rewards:
- Format correct: +{format_reward:g}
- Progress toward target pose: continuous reward based on distance and orientation improvement
- Reaching target pose: +{success_reward:g}

Hints:
1. Pay attention to the target object and the requested view (front, back, left, right, etc.)
2. {hint2}
3. Consider both position and orientation when navigating
4. Strategy: DEFAULT to move_forward for ordinary target-view tasks, but follow task-specific relation instructions for occlusion, centering, projective, and size tasks. After at most 2 consecutive turns, try a translation action unless the relation requires continued alignment.
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
    
    template = f"""[Observation]:
{observation}
"""
    if env_feedback:
        template += f"Environment Feedback: {env_feedback}\n"
    
    return template


# Format prompt functions for different prompt formats
def format_prompt_free_think(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think>Your reasoning process</think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['free_think']['example']}"
    return prompt


def format_prompt_no_think(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond with only the action:
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['no_think']['example']}"
    return prompt


def format_prompt_grounding(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think><observation>What you see</observation><reasoning>Your reasoning</reasoning></think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['grounding']['example']}"
    return prompt


def format_prompt_worldmodeling(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think><reasoning>Your reasoning</reasoning><prediction>Expected outcome</prediction></think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['worldmodeling']['example']}"
    return prompt


def format_prompt_grounding_worldmodeling(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think><observation>What you see</observation><reasoning>Your reasoning</reasoning><prediction>Expected outcome</prediction></think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

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
