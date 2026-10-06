"""Observation/action contract shared by offline and closed-loop StarVLA eval.

No MuJoCo or StarVLA imports: this runs in either isolated Python environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils.actions import GYM_SLICES, STARVLA_STATE_KEYS, STARVLA_STATE_KEY_DIMS

CAMERAS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


@dataclass(frozen=True)
class PolicyContract:
    cameras: tuple[str, ...]
    image_size: tuple[int, int]
    include_state: bool
    horizon: int

    @classmethod
    def from_config(cls, config):
        data = config["datasets"]["vla_data"]
        framework = config["framework"]
        if framework["name"] != "QwenOFT":
            raise ValueError("This adapter supports the PandaOmron QwenOFT contract only")
        if int(framework["action_model"]["action_dim"]) != 12:
            raise ValueError("PandaOmron requires 12 action dimensions")
        mix = data.get("mixture_spec")
        if mix and any(item[2] != "panda_omron_robocasa365" for item in mix):
            raise ValueError("Mixed embodiments need their own evaluation adapter")
        if not mix and not data["data_mix"].startswith("robocasa365_"):
            raise ValueError("Cannot infer the camera contract for this dataset")
        # The registered PandaOmron DataConfig uses all three cameras in this order.
        cameras = tuple(data.get("camera_keys", CAMERAS))
        if cameras != CAMERAS:
            raise ValueError("camera_keys differs from PandaOmron training DataConfig")
        size = tuple(int(v) for v in data["obs_image_size"])
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("obs_image_size must contain two positive dimensions")
        action = framework["action_model"]
        horizon = int(action.get("action_horizon", int(action.get("future_action_window_size", 15)) + 1))
        if horizon <= 0:
            raise ValueError("action_horizon must be positive")
        return cls(cameras, size, data.get("include_state", False) not in (False, "False", "false", None), horizon)

    def example(self, observation):
        images = []
        for key in self.cameras:
            arr = np.asarray(observation[key])
            if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                raise ValueError(f"{key}: expected HWC uint8 RGB, got {arr.shape}/{arr.dtype}")
            # Match LeRobotSingleDataset._pack_sample (PIL RGB default: bicubic).
            images.append(np.asarray(Image.fromarray(arr).resize(self.image_size)))
        language = observation["annotation.human.task_description"]
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Environment must supply a nonempty task instruction")
        example = {"image": images, "lang": language}
        if self.include_state:
            # Training transforms sin/cos PER named key before concatenation.
            parts = []
            for key in STARVLA_STATE_KEYS:
                value = np.asarray(observation[key], dtype=np.float32).reshape(-1)
                if value.size != STARVLA_STATE_KEY_DIMS[key] or not np.isfinite(value).all():
                    raise ValueError(f"Invalid proprioception for {key}")
                parts.append(np.concatenate([np.sin(value), np.cos(value)]))
            example["state"] = np.concatenate(parts)[None]
        return example

    def actions(self, value):
        actions = np.asarray(value, dtype=np.float32)
        if actions.shape != (1, self.horizon, 12) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite (1,{self.horizon},12) actions, got {actions.shape}")
        return actions[0]


def gym_action(vector, action_space):
    """Clip already unnormalized outputs to the environment's controller limits."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (12,) or not np.isfinite(arr).all():
        raise ValueError("Invalid 12-D action")
    return {
        key: np.clip(arr[start:end], action_space[key].low, action_space[key].high)
        for key, (start, end) in GYM_SLICES.items()
    }


def checkpoint_config(ckpt):
    import yaml

    path = Path(ckpt).resolve(strict=True)
    for root in (path.parent, path.parent.parent):
        config = root / "config.full.yaml"
        if not config.is_file():
            config = root / "config.yaml"
        if config.is_file():
            return yaml.safe_load(config.read_text()), root
    raise FileNotFoundError(f"No saved config next to {path}")


def verify_server(metadata, ckpt, contract):
    served = metadata.get("ckpt_path")
    if not served or Path(served).resolve() != Path(ckpt).resolve():
        raise ValueError(f"Server is serving a different checkpoint: {served}")
    if metadata.get("action_chunk_size") != contract.horizon:
        raise ValueError("Server action horizon differs from checkpoint config")
    if list(metadata.get("action_keys", [])) != list(GYM_SLICES):
        raise ValueError("Server action order differs from PandaOmron")
    if tuple(metadata.get("training_obs_image_size", [])) != contract.image_size:
        raise ValueError("Server image size differs from checkpoint config")
