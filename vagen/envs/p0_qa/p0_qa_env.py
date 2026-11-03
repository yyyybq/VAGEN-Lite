"""Single-turn P0 question answering as an RL env (the Qwen3 main line's QA RL pilot, the user's instruction of
2026-11-03, section 2).

One episode is one question of a rows file (data_gen.interiorgs_v10.qa_rl rows): the prompt and its images exactly as
the P0 SFT shows them (images first, then the question; no system prompt, as the SFT records have none), one answer,
done.  The reward is data_gen.interiorgs_v10.qa_rl.reward: the benchmark's own scoring (multiple choice 1 / 0,
VSI-Bench's numeric answers by MRA); an answer its extraction cannot read scores 0 and is reported as not valid
(turn_metrics.action_is_valid), which the agent loop counts as an unparseable answer (transition/invalid_action_rate,
the chain's 10% line).  Success (the group filter's mixed outcome): score 1 (multiple choice), MRA >= 0.5 (numeric).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from .env_config import P0QAEnvConfig

NUMERIC_SUCCESS = 0.5


class P0QAEnv(GymImageEnv):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        valid = {f.name for f in dataclass_fields(P0QAEnvConfig)}
        self.config = P0QAEnvConfig(**{k: v for k, v in env_config.items() if k in valid})
        path = Path(self.config.jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"P0QA rows not found: {path}")
        self._rows: List[Dict[str, Any]] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not self._rows:
            raise ValueError(f"P0QA rows file is empty: {path}")
        self._cur: Optional[Dict[str, Any]] = None

    def _obs(self, row: Dict[str, Any]) -> Dict[str, Any]:
        ph = self.config.image_placeholder
        if row["prompt"].count(ph) != len(row["images"]):
            raise ValueError(f"{row['qid']}: {row['prompt'].count(ph)} placeholders for {len(row['images'])} images")
        images = [Image.open(p).convert("RGB") for p in row["images"]]
        return {"obs_str": row["prompt"], "multi_modal_input": {ph: images}} if images else {"obs_str": row["prompt"]}

    async def system_prompt(self) -> Dict[str, Any]:
        return {}  # the P0 SFT records have no system message; an empty one would add tokens the SFT never saw

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._cur = self._rows[int(seed) % len(self._rows)]
        obs = await asyncio.to_thread(self._obs, self._cur)
        return obs, {"success": False, "task_type": self._cur["benchmark"], "scene_id": self._cur["scene_id"],
                     "task_id": self._cur["qid"]}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        from data_gen.interiorgs_v10.qa_rl import reward

        assert self._cur is not None, "step before reset"
        value, readable = reward(self._cur, action_str or "")
        success = value >= (NUMERIC_SUCCESS if self._cur["answer_type"] == "numeric" else 1.0)
        info = {"success": success, "qid": self._cur["qid"], "benchmark": self._cur["benchmark"], "score": value,
                "readable": readable,
                "metrics": {"turn_metrics": {"action_is_valid": readable, "action_is_effective": success},
                            "traj_metrics": {"success": success}}}
        return {"obs_str": "done."}, float(value), True, info

    async def close(self) -> None:
        self._cur = None
