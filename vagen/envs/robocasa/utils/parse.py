"""Parse VLM responses for RoboCasa.

Supported prompt formats:
  - free_think: requires <think>...</think> and <action>...</action>
  - no_think:   requires <action>...</action> only

One VLM turn may emit several 12-D vectors inside <action>; they are
executed sequentially up to ``max_actions``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from vagen.envs.robocasa.utils.actions import parse_action_block

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

VALID_PARSE_FORMATS = ("free_think", "no_think")


def parse_response(
    action_str: str,
    prompt_format: str = "free_think",
    max_actions: int = 8,
) -> Dict[str, Any]:
    """Extract think text and 12-D actions from a model response.

    Returns:
        dict with keys:
            llm_raw_response, actions (list[list[float]]),
            format_correct (bool), think (str, optional)
    """
    if prompt_format not in VALID_PARSE_FORMATS:
        raise ValueError(
            f"Unknown prompt_format {prompt_format!r}. Valid: {VALID_PARSE_FORMATS}"
        )

    text = action_str or ""
    result: Dict[str, Any] = {
        "llm_raw_response": text,
        "actions": [],
        "format_correct": False,
        "think": "",
    }

    think_match = _THINK_RE.search(text)
    action_matches = list(_ACTION_RE.finditer(text))

    if think_match:
        result["think"] = think_match.group(1).strip()

    action_bodies = [m.group(1) for m in action_matches]
    parsed: List[List[float]] = []
    for body in action_bodies:
        parsed.extend(parse_action_block(body, max_actions=None))

    if max_actions is not None and max_actions > 0:
        parsed = parsed[: int(max_actions)]
    result["actions"] = parsed

    has_action_tag = len(action_matches) > 0
    if prompt_format == "free_think":
        result["format_correct"] = bool(think_match) and has_action_tag
    else:
        result["format_correct"] = has_action_tag

    return result
