"""
Compatibility shims for legacy VAGEN base classes.

These stubs allow active_spatial env.py and env_config.py (originally written
for the old VAGEN codebase) to work inside VAGEN-Lite without any path tricks.
"""

import re
import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from PIL import Image


# ─── BaseEnvConfig ────────────────────────────────────────────────────────────

@dataclass
class BaseEnvConfig(ABC):
    format_reward: float = 0.5
    image_placeholder: str = "<image>"
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>"]
    )
    action_sep: str = ","

    @abstractmethod
    def config_id(self) -> str:
        """Config identifier used by wandb and rollout manager."""
        pass

    def get(self, key, default=None):
        return getattr(self, key, default)

    def generate_seeds(self, size, seed=0, n_candidate: int = 20000) -> list:
        random.seed(seed)
        seeds = random.sample(range(0, n_candidate + size), size)
        return seeds


# ─── BaseEnv ──────────────────────────────────────────────────────────────────

class BaseEnv(ABC):
    @abstractmethod
    def step(self, llm_raw_response):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self, seed=None):
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    def compute_reward(self) -> float:
        return 0.0


# ─── context_utils ────────────────────────────────────────────────────────────

def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL RGB image."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array, mode='RGB')
    raise ValueError(
        f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB)."
    )


# ─── parse_utils ──────────────────────────────────────────────────────────────

def _parse_freethink(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    response = response.replace("<image>", "")
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    extraction_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None

    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for tok in special_token_list:
                action_content = action_content.replace(tok, "").strip()
                think_content = think_content.replace(tok, "").strip()
        actions = [a.strip() for a in action_content.split(action_sep) if a.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = (
        "<think>" + think_content.strip() + "</think>"
        + "<answer>" + action_content.strip() + "</answer>"
    )
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def _parse_no_think(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    response = response.replace("<image>", "")
    strict_pattern = r'^\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    extraction_pattern = r'<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None

    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        think_content = ""
        action_content = match.group(1)
        if special_token_list is not None:
            for tok in special_token_list:
                action_content = action_content.replace(tok, "").strip()
        actions = [a.strip() for a in action_content.split(action_sep) if a.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


PARSE_FUNC_MAP = {
    "free_think": _parse_freethink,
    "no_think": _parse_no_think,
    "grounding": _parse_freethink,
    "worldmodeling": _parse_freethink,
    "grounding_worldmodeling": _parse_freethink,
    "grounding_structured": _parse_freethink,
    "worldmodeling_structured": _parse_freethink,
    "grounding_worldmodeling_structured": _parse_freethink,
    "grounding_symbolic": _parse_freethink,
    "worldmodeling_symbolic": _parse_freethink,
    "grounding_worldmodeling_symbolic": _parse_freethink,
}
