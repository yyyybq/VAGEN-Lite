"""
Evaluation Configuration for Active Spatial Navigation
======================================================

Defines all configurable parameters for running evaluation,
including environment, model, and metrics settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class EvalEnvConfig:
    """Environment configuration for evaluation."""
    jsonl_path: str = ""
    render_backend: Optional[str] = "local"
    gs_root: str = ""
    gpu_device: Optional[int] = 4
    image_width: int = 512
    image_height: int = 512
    step_translation: float = 0.2
    step_rotation_deg: float = 10.0
    
    # Potential field
    enable_potential_field: bool = True
    potential_field_position_weight: float = 0.7
    potential_field_orientation_weight: float = 0.3
    potential_field_reward_scale: float = 1.0
    success_score_threshold: float = 0.85
    
    # Collision
    enable_collision_detection: bool = True
    collision_camera_radius: float = 0.15
    collision_floor_height: float = 0.3
    collision_ceiling_height: float = 2.5
    collision_penalty: float = -0.15
    
    # Visibility
    enable_visibility_check: bool = True
    fov_horizontal: float = 60.0
    fov_vertical: float = 60.0
    
    # Prompt
    prompt_format: str = "free_think"
    max_actions_per_step: int = 5
    action_sep: str = "|"
    image_placeholder: str = "<image>"
    max_episode_steps: int = 50
    format_reward: float = 0.05
    success_reward: float = 1.0
    max_distance: float = 5.0


@dataclass
class EvalModelConfig:
    """Model configuration for evaluation."""
    provider: str = "vllm"                  # vllm, openai, claude, etc.
    model_name: str = ""                     # HF model ID or API model name
    checkpoint_path: Optional[str] = None    # Path to trained checkpoint
    temperature: float = 0.1                 # Lower temp for eval (less random)
    top_p: float = 0.95
    max_tokens: int = 512
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    
    # For API models
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""
    # Evaluation settings
    eval_name: str = "active_spatial_eval"
    output_dir: str = "evaluation/outputs"
    max_steps_per_episode: int = 20          # Max LLM turns per episode
    num_eval_episodes: Optional[int] = None  # None = use all test data
    seed_offset: int = 0                     # Offset for seed selection
    
    # Agent type
    agent_type: str = "model"                # "model", "random", "heuristic", "frozen"
    
    # Environment
    env: EvalEnvConfig = field(default_factory=EvalEnvConfig)
    
    # Model (only needed if agent_type == "model" or "frozen")
    model: EvalModelConfig = field(default_factory=EvalModelConfig)
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "active-spatial-eval"
    wandb_entity: Optional[str] = None
    save_trajectories: bool = True           # Save full episode trajectories
    save_images: bool = False                # Save rendered images per step
    verbose: bool = False
    
    # Evaluation subsets
    task_types: Optional[List[str]] = None   # None = all tasks; or list of specific types
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvalConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)
        
        env_cfg = EvalEnvConfig(**raw.pop("env", {}))
        model_cfg = EvalModelConfig(**raw.pop("model", {}))
        return cls(env=env_cfg, model=model_cfg, **raw)
    
    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        import dataclasses
        d = dataclasses.asdict(self)
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)
