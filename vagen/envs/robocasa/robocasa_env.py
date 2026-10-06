"""RoboCasa kitchen environment (GymImageEnv async interface).

The VLM emits text actions in Active Spatial tags:

    <think>...</think><action>[12-D] ...</action>

Each 12-D vector is converted to the official gym action dict from
``robocasa.wrappers.gym_wrapper.PandaOmronKeyConverter`` and executed
sequentially (action_horizon).

RoboCasa / MuJoCo imports are lazy so action unit tests and SFT
conversion can run without the policy venv.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.robocasa.utils.actions import format_action_text, to_gym_action
from vagen.envs.robocasa.utils.parse import parse_response
from vagen.envs.robocasa.utils.prompt import (
    VALID_FORMATS,
    action_template,
    init_observation_template,
    system_prompt,
)
from vagen.envs.robocasa.utils.tasks import get_horizon

LOGGER = logging.getLogger(__name__)

PRIMARY_IMAGE_KEY = "video.robot0_agentview_left"
WRIST_IMAGE_KEY = "video.robot0_eye_in_hand"
LANGUAGE_KEY = "annotation.human.task_description"


@dataclass
class RoboCasaEnvConfig:
    env_name: str = "robocasa"
    env_id: str = "PickPlaceCounterToCabinet"
    split: str = "target"  # pretrain | target
    render_mode: str = "vision"  # vision | text
    prompt_format: str = "free_think"
    action_horizon: int = 8
    max_steps: int = 0  # 0 = use task horizon fallback
    max_actions_per_step: int = 0  # 0 = action_horizon
    include_wrist_image: bool = False
    obj_registries: Optional[List[str]] = None
    layout_and_style_ids: Optional[Any] = None
    format_reward: float = 0.1
    success_reward: float = 10.0
    image_placeholder: str = "<image>"
    gpu_device: int = 0
    enable_render: Optional[bool] = None  # None = render_mode == vision


class RoboCasaEnv(GymImageEnv):
    """Async RoboCasa environment implementing the GymImageEnv interface."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        valid = {f.name for f in fields(RoboCasaEnvConfig)}
        self.cfg = RoboCasaEnvConfig(**{k: v for k, v in env_config.items() if k in valid})
        if self.cfg.prompt_format not in VALID_FORMATS:
            raise ValueError(
                f"Unknown prompt_format: {self.cfg.prompt_format}. Valid: {VALID_FORMATS}"
            )
        if self.cfg.max_actions_per_step <= 0:
            self.cfg.max_actions_per_step = self.cfg.action_horizon
        if self.cfg.max_steps <= 0:
            self.cfg.max_steps = get_horizon(self.cfg.env_id)
        if self.cfg.enable_render is None:
            self.cfg.enable_render = self.cfg.render_mode == "vision"

        self._env = None
        self._language: str = ""
        self._last_obs: Dict[str, Any] = {}
        self._last_info: Dict[str, Any] = {}
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._t0: float = 0.0

    # ------------------------------------------------------------------
    # Lazy gym.make (must happen in a worker thread; uses EGL / GPU)
    # ------------------------------------------------------------------

    def _ensure_env(self):
        if self._env is not None:
            return
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        if self.cfg.gpu_device is not None:
            os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(self.cfg.gpu_device))

        import gymnasium as gym  # lazy: policy venv only
        import robocasa  # noqa: F401  # registers robocasa/* gym ids

        make_kwargs: Dict[str, Any] = {
            "split": self.cfg.split,
            "enable_render": bool(self.cfg.enable_render),
        }
        if self.cfg.obj_registries is not None:
            make_kwargs["obj_registries"] = self.cfg.obj_registries
        if self.cfg.layout_and_style_ids is not None:
            make_kwargs["layout_and_style_ids"] = self.cfg.layout_and_style_ids

        gym_id = self.cfg.env_id
        if not str(gym_id).startswith("robocasa/"):
            gym_id = f"robocasa/{gym_id}"
        LOGGER.info("Creating %s split=%s render=%s", gym_id, self.cfg.split, self.cfg.enable_render)
        # disable_env_checker: official unmap_action compares gripper/control
        # as scalars and slices base_motion with [..., k]. The checker would
        # also reject 0-d floats for Box(shape=(1,)).
        try:
            self._env = gym.make(gym_id, disable_env_checker=True, **make_kwargs)
        except TypeError:
            self._env = gym.make(gym_id, **make_kwargs)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _extract_language(self, obs: Dict[str, Any]) -> str:
        lang = ""
        if isinstance(obs, dict):
            lang = obs.get(LANGUAGE_KEY) or obs.get("language") or ""
        if not lang:
            inner = getattr(self._env, "unwrapped", self._env)
            getter = getattr(inner, "get_ep_meta", None)
            if callable(getter):
                try:
                    lang = (getter() or {}).get("lang", "") or ""
                except Exception:
                    lang = ""
        return str(lang)

    def _numpy_to_pil(self, arr: Any):
        from PIL import Image

        if arr is None:
            return None
        try:
            import numpy as np
        except ImportError:
            if isinstance(arr, Image.Image):
                return arr.convert("RGB")
            return None
        if isinstance(arr, Image.Image):
            return arr.convert("RGB")
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.asarray(arr)
            except Exception:
                return None
        if arr.ndim == 3 and arr.shape[-1] == 4:
            return Image.fromarray(arr.astype("uint8"), mode="RGBA").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return Image.fromarray(arr.astype("uint8"), mode="RGB")
        return None

    def _collect_images(self, obs: Dict[str, Any]) -> List[Any]:
        images: List[Any] = []
        if not isinstance(obs, dict):
            return images
        primary = self._numpy_to_pil(obs.get(PRIMARY_IMAGE_KEY))
        if primary is not None:
            images.append(primary)
        if self.cfg.include_wrist_image:
            wrist = self._numpy_to_pil(obs.get(WRIST_IMAGE_KEY))
            if wrist is not None:
                images.append(wrist)
        return images

    def _image_placeholders(self, n_images: int) -> str:
        ph = self.cfg.image_placeholder
        if n_images <= 0:
            return ""
        if n_images == 1:
            return ph
        labels = ["Agent view:", "Wrist camera:"]
        parts = []
        for i in range(n_images):
            label = labels[i] if i < len(labels) else f"Camera {i}:"
            parts.append(f"{label}\n{ph}")
        return "\n".join(parts)

    def _render_obs(self, init: bool, valid_actions: Optional[List[str]] = None) -> Dict[str, Any]:
        images = self._collect_images(self._last_obs) if self.cfg.render_mode == "vision" else []
        obs_visual = self._image_placeholders(len(images))
        if init:
            obs_str = init_observation_template(obs_visual, self._language)
        else:
            obs_str = action_template(
                obs_visual,
                self._language,
                valid_actions=valid_actions,
            )
        obs: Dict[str, Any] = {"obs_str": obs_str}
        if images:
            obs["multi_modal_input"] = {self.cfg.image_placeholder: images}
        return obs

    # ------------------------------------------------------------------
    # Sync methods (run in thread)
    # ------------------------------------------------------------------

    def _sync_reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_env()
        reset_out = self._env.reset(seed=int(seed))
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
        self._last_obs = obs if isinstance(obs, dict) else {}
        self._last_info = dict(info or {})
        self._language = self._extract_language(self._last_obs)
        self._last_info.setdefault("success", False)
        self._last_info["language"] = self._language
        self._step_count = 0
        self._total_reward = 0.0
        self._t0 = time.time()
        return self._render_obs(init=True), {"success": False, "language": self._language}

    def _to_env_action(self, gym_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert list slices to numpy so PandaOmronKeyConverter.unmap_action works.

        Official unmap uses ``action.base_motion[..., 0:3]`` and
        ``action.gripper_close < 0.5``. Python lists raise TypeError.
        """
        import numpy as np

        out: Dict[str, Any] = {}
        for key, val in gym_action.items():
            arr = np.asarray(val, dtype=np.float32)
            if key in ("action.gripper_close", "action.control_mode"):
                out[key] = np.float32(arr.reshape(-1)[0])
            else:
                out[key] = arr
        return out

    def _gym_step(self, gym_action: Dict[str, Any]):
        """Call gym.step and normalize 4-tuple / 5-tuple returns."""
        step_out = self._env.step(self._to_env_action(gym_action))
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
            return obs, float(reward or 0.0), done, dict(info or {})
        if isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done, info = step_out
            return obs, float(reward or 0.0), bool(done), dict(info or {})
        raise RuntimeError(f"Unexpected gym.step return: {type(step_out)}")

    def _sync_step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            action_str,
            prompt_format=self.cfg.prompt_format,
            max_actions=self.cfg.max_actions_per_step,
        )
        valid_actions: List[str] = []
        info: Dict[str, Any] = {**parsed}
        metrics = {
            "turn_metrics": {
                "action_is_valid": False,
                "action_is_effective": False,
                "format_correct": parsed["format_correct"],
            },
            "traj_metrics": {"success": False},
        }

        success = bool(self._last_info.get("success", False))
        env_reward = 0.0
        done = False

        for vec in parsed["actions"]:
            try:
                gym_action = to_gym_action(vec)
            except ValueError:
                break
            obs, step_reward, step_done, step_info = self._gym_step(gym_action)
            self._last_obs = obs if isinstance(obs, dict) else {}
            self._last_info = step_info
            self._language = self._extract_language(self._last_obs) or self._language
            self._step_count += 1
            env_reward += float(step_reward)
            valid_actions.append(format_action_text(vec))
            success = bool(step_info.get("success", False)) or success
            done = bool(step_done) or success
            if done or self._step_count >= self.cfg.max_steps:
                done = True
                break

        if self._step_count >= self.cfg.max_steps:
            done = True

        metrics["turn_metrics"]["action_is_valid"] = (
            len(valid_actions) > 0 and len(valid_actions) == len(parsed["actions"])
        )
        metrics["turn_metrics"]["action_is_effective"] = len(valid_actions) > 0
        metrics["traj_metrics"]["success"] = bool(success)

        reward = 0.0
        if parsed["format_correct"] and valid_actions:
            reward += float(self.cfg.format_reward)
        if success:
            reward += float(self.cfg.success_reward)

        info["metrics"] = metrics
        info["success"] = bool(success)
        info["env_step"] = self._step_count
        info["env_reward"] = env_reward
        info["language"] = self._language
        info["episode_elapsed_seconds"] = time.time() - self._t0
        info["n_executed"] = len(valid_actions)
        self._total_reward += reward

        obs = self._render_obs(init=False, valid_actions=valid_actions)
        return obs, reward, bool(done), info

    def _sync_close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:
                LOGGER.warning("RoboCasa env close failed: %s", exc)
            self._env = None

    def _sync_system_prompt(self) -> Dict[str, Any]:
        return {
            "obs_str": system_prompt(
                format_name=self.cfg.prompt_format,
                max_actions_per_step=self.cfg.max_actions_per_step,
                include_wrist_image=self.cfg.include_wrist_image,
            )
        }

    # ------------------------------------------------------------------
    # Async interface (GymImageEnv)
    # ------------------------------------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._sync_system_prompt)

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_reset, seed)

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_step, action_str)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_close)


if __name__ == "__main__":
    import fire

    async def _run(
        env_id: str = "PickPlaceCounterToCabinet",
        split: str = "target",
        prompt_format: str = "free_think",
        seed: int = 0,
    ):
        env = RoboCasaEnv({"env_id": env_id, "split": split, "prompt_format": prompt_format})
        print((await env.system_prompt())["obs_str"])
        obs, info = await env.reset(seed)
        print(obs["obs_str"])
        print("info", info)
        await env.close()

    fire.Fire(lambda **kw: asyncio.run(_run(**kw)))
