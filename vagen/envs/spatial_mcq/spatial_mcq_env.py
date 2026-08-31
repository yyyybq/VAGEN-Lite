"""Single-turn spatial MCQ environment for A1 aux mixture training."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from .env_config import SpatialMCQEnvConfig

_ANSWER_RE = re.compile(
    r"<\s*answer\s*>\s*([A-Da-d])(?:\s*[\.\:\)]?\s*[^\n<]*)?\s*<\s*/\s*answer\s*>",
    re.IGNORECASE | re.DOTALL,
)
_LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def _extract_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    # fallback: last standalone A-D
    hits = _LETTER_RE.findall(text)
    return hits[-1].upper() if hits else None


class SpatialMCQEnv(GymImageEnv):
    """One-step MCQ: show images + question, score letter answer."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        valid = {f.name for f in dataclass_fields(SpatialMCQEnvConfig)}
        filtered = {k: v for k, v in env_config.items() if k in valid}
        self.config = SpatialMCQEnvConfig(**filtered)
        path = Path(self.config.jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"SpatialMCQ jsonl not found: {path}")
        self._items: List[Dict[str, Any]] = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._items.append(json.loads(line))
        if not self._items:
            raise ValueError(f"SpatialMCQ jsonl is empty: {path}")
        self._cur: Optional[Dict[str, Any]] = None
        self._image_root = Path(self.config.image_root) if self.config.image_root else Path()

    def _resolve_images(self, item: Dict[str, Any]) -> List[Image.Image]:
        paths = item.get("image_paths") or item.get("images") or []
        max_images = max(1, int(getattr(self.config, "max_images", 1) or 1))
        paths = list(paths)[:max_images]
        max_side = int(getattr(self.config, "max_image_side", 0) or 0)
        imgs: List[Image.Image] = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = self._image_root / pp
            im = Image.open(pp).convert("RGB")
            if max_side > 0:
                w, h = im.size
                scale = max(w, h) / float(max_side)
                if scale > 1.0:
                    im = im.resize((max(1, int(w / scale)), max(1, int(h / scale))), Image.Resampling.BILINEAR)
            imgs.append(im)
        if not imgs:
            raise FileNotFoundError(f"No images for item id={item.get('id')}")
        return imgs

    def _build_obs(self, item: Dict[str, Any]) -> Dict[str, Any]:
        imgs = self._resolve_images(item)
        ph = self.config.image_placeholder
        placeholders = " ".join([ph] * len(imgs))
        # SITE/jsonl questions often already embed "<image>" tokens. We inject
        # exactly len(imgs) placeholders above, so strip any pre-existing ones
        # from the body to keep #images == #<image>.
        question = str(item.get("question", "")).replace(ph, "").strip()
        options = item.get("options") or []
        if options:
            opt_text = "\n".join(str(o).replace(ph, "") for o in options)
            body = f"{question}\n{opt_text}".strip()
        else:
            body = question
        obs_str = (
            f"[Spatial Aux MCQ]\n{placeholders}\n\n{body}\n\n"
            "Reply with a single choice letter inside <answer>...</answer>, "
            "e.g. <answer>A</answer>."
        )
        return {
            "obs_str": obs_str,
            "multi_modal_input": {ph: imgs},
        }

    async def system_prompt(self) -> Dict[str, Any]:
        return {
            "obs_str": (
                "You are a spatial reasoning assistant. Answer the multiple-choice "
                "question about the image(s) with one letter in <answer></answer>."
            )
        }

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        idx = int(seed) % len(self._items)
        self._cur = self._items[idx]
        obs = await asyncio.to_thread(self._build_obs, self._cur)
        info = {
            "success": False,
            "mcq_id": self._cur.get("id", idx),
            "source": self._cur.get("source", "spatial_mcq"),
            "gold": str(self._cur.get("answer", "")).upper()[:1],
        }
        return obs, info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self._cur is not None
        gold = str(self._cur.get("answer", "")).strip().upper()[:1]
        pred = _extract_letter(action_str or "")
        format_ok = pred is not None
        correct = bool(pred and gold and pred == gold)
        reward = 0.0
        if format_ok:
            reward += float(self.config.format_reward)
        else:
            reward += float(self.config.invalid_format_penalty)
        if correct:
            reward += float(self.config.answer_reward) * float(self.config.aux_coef)
        info = {
            "success": correct,
            "format_correct": format_ok,
            "pred": pred,
            "gold": gold,
            "metrics": {
                "turn_metrics": {
                    "action_is_valid": format_ok,
                    "action_is_effective": correct,
                },
                "traj_metrics": {"success": correct},
            },
        }
        # Terminal empty obs
        obs = {"obs_str": "done."}
        return obs, float(reward), True, info

    async def close(self) -> None:
        self._cur = None
