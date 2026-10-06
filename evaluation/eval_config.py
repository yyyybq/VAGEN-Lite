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
import hashlib
import json


from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def evaluation_fingerprint(raw):
    """Invalidate old results when data, effective config or success protocol changes."""
    values = dict(raw)
    env = EvalEnvConfig(**values.pop("env", {}))
    model = EvalModelConfig(**values.pop("model", {}))
    normalized = EvalConfig(env=env, model=model, **values).to_dict()
    for key in ("eval_name", "output_dir", "verbose", "use_wandb", "wandb_project", "wandb_entity"):
        normalized.pop(key, None)
    normalized["model"].pop("api_key", None)
    hashes = {}
    contract_path = env.dataset_contract_path if env.require_verified_dataset else None
    for path in (env.jsonl_path, normalized.get("training_jsonl_path"), contract_path):
        if path:
            hashes[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = {"revision": "active_spatial_audit_v2", "config": normalized, "files": hashes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class EvalEnvConfig(ActiveSpatialEnvConfig):
    """The runtime schema is also the evaluation schema; no duplicate defaults."""


@dataclass
class EvalModelConfig:
    """Model configuration for evaluation."""
    provider: str = "vllm"                  # vllm, openai, openai_responses, claude, gemini
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
    reasoning_effort: Optional[str] = None  # low|medium|high|xhigh|max for GPT-5/GPT-6
    max_retries: int = 6
    request_timeout: float = 180.0
    service_tier: Optional[str] = None  # flex|standard|priority


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""
    # Evaluation settings
    eval_name: str = "active_spatial_eval"
    output_dir: str = "evaluation/outputs"
    max_steps_per_episode: int = 20          # Max LLM turns per episode
    num_eval_episodes: Optional[int] = None  # None = use all test data
    seed_offset: int = 0                     # Offset for seed selection
    training_jsonl_path: Optional[str] = None
    split_role: Optional[str] = None
    protocol_differences: Dict[str, Any] = field(default_factory=dict)
    
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
