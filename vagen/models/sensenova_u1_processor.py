"""
Processor wrapper for SenseNova-U1 in VAGEN-Lite.

SenseNova-U1 does not ship a HuggingFace ``AutoProcessor``.  Its reference
VQA path preprocesses each image with ``load_image_native`` and expands every
``<image>`` placeholder into::

    <img><IMG_CONTEXT> * num_patch_tokens</img>

where ``num_patch_tokens = grid_h * grid_w * downsample_ratio**2``.  This
wrapper gives VAGEN the same behavior for agent-loop tokenization and FSDP
logprob computation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase


SENSENOVA_U1_SRC = os.environ.get("SENSENOVA_U1_SRC", "/nas/baiqiao/SenseNova-U1/src")
if SENSENOVA_U1_SRC not in sys.path:
    sys.path.insert(0, SENSENOVA_U1_SRC)

from sensenova_u1.models.neo_unify.utils import load_image_native  # noqa: E402


IMAGE_TOKEN = "<image>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class SenseNovaU1ImageProcessor:
    """Sentinel object used by agent_loop_no_concat.py for model detection."""

    pass


def _normalize_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(Path(image))


class SenseNovaU1ProcessorWrapper:
    """Minimal tokenizer + image-preprocessor shim for SenseNova-U1."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        patch_size: int = 16,
        downsample_ratio: float = 0.5,
        min_pixels: int = 512 * 512,
        max_pixels: int = 2048 * 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_processor = SenseNovaU1ImageProcessor()

        for token in (IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN):
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token], special_tokens=True)

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ):
        flat_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    btype = block.get("type")
                    if btype == "image":
                        parts.append(IMAGE_TOKEN + "\n")
                    elif btype == "text":
                        parts.append(block.get("text", ""))
                new_msg = dict(msg)
                new_msg["content"] = "".join(parts)
                flat_messages.append(new_msg)
            else:
                flat_messages.append(msg)

        return self.tokenizer.apply_chat_template(
            flat_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def preprocess_images(self, images: List[Image.Image]) -> dict[str, torch.Tensor]:
        pixel_values = []
        grid_hw = []
        max_pixels = min(self.max_pixels, (4096 * 4096) // max(1, len(images)))
        for image in images:
            cur_pixel_values, cur_grid_hw = load_image_native(
                _normalize_image(image),
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
                min_pixels=self.min_pixels,
                max_pixels=max_pixels,
                upscale=False,
            )
            pixel_values.append(cur_pixel_values)
            grid_hw.append(cur_grid_hw)

        if not pixel_values:
            return {
                "pixel_values": torch.zeros(0, 3 * self.patch_size * self.patch_size),
                "grid_hw": torch.zeros(0, 2, dtype=torch.long),
            }

        return {
            "pixel_values": torch.cat(pixel_values, dim=0),
            "grid_hw": torch.cat(grid_hw, dim=0).long(),
        }

    def expand_image_tokens(self, token_ids: list[int], grid_hw: torch.Tensor) -> list[int]:
        expanded = list(token_ids)
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_token_patterns = [
            self.tokenizer.encode(IMAGE_TOKEN + "\n", add_special_tokens=False),
            self.tokenizer.encode(IMAGE_TOKEN, add_special_tokens=False),
        ]
        for i in range(grid_hw.shape[0]):
            num_patch_token = int(
                grid_hw[i, 0].item()
                * grid_hw[i, 1].item()
                * (self.downsample_ratio ** 2)
            )
            image_token_ids = (
                [self.img_start_token_id]
                + [self.img_context_token_id] * num_patch_token
                + [self.img_end_token_id]
            )
            idx = -1
            span_len = 0
            if image_token_id is not None:
                try:
                    idx = expanded.index(image_token_id)
                    span_len = 1
                except ValueError:
                    pass
            if idx < 0:
                for image_ids in image_token_patterns:
                    idx = _find_subsequence(expanded, image_ids)
                    if idx >= 0:
                        span_len = len(image_ids)
                        break
            if idx < 0:
                break
            expanded = expanded[:idx] + image_token_ids + expanded[idx + span_len :]
        return expanded

    def __call__(
        self,
        text: Union[str, List[str]],
        images=None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict:
        if isinstance(text, str):
            text = [text]

        mm = self.preprocess_images(images or [])
        all_input_ids = []
        for t in text:
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            all_input_ids.append(self.expand_image_tokens(token_ids, mm["grid_hw"]))

        if return_tensors != "pt":
            return {"input_ids": all_input_ids, **mm}

        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        padded_ids = []
        attention_masks = []
        for ids in all_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            **mm,
        }


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    max_start = len(values) - len(needle)
    for idx in range(max_start + 1):
        if values[idx : idx + len(needle)] == needle:
            return idx
    return -1
