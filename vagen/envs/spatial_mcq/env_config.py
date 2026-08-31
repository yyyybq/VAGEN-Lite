from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpatialMCQEnvConfig:
    """Single-turn spatial multiple-choice QA env (EASI-style aux)."""

    env_name: str = "spatial_mcq"
    jsonl_path: str = ""
    image_root: str = ""  # prepended to relative image paths
    answer_reward: float = 1.0
    format_reward: float = 0.0
    invalid_format_penalty: float = -0.1
    prompt_format: str = "free_think"
    image_placeholder: str = "<image>"
    # Cap aux vision tokens so prompt fits max_model_len (A1: 4096+384).
    max_images: int = 1
    max_image_side: int = 784  # downsample long side; 0 disables
    max_actions_per_step: int = 1
    action_sep: str = "|"
    # Reward scale used as unique A1 variable when mixed with navigation.
    # Kept here for logging; mix ratio in train.yaml is the primary knob.
    aux_coef: float = 0.05
