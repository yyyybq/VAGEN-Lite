"""
vagen/models/cambrian_processor.py

Processor wrapper for Cambrian-S in VAGEN-Lite.

Used in two places:
  1. gym_agent_loop.py  – builds tokenized prompts with <image> placeholders
  2. agent_loop_no_concat.py – detects Cambrian branch and provides SigLIP
     preprocessing for FSDP pixel_values

Token design
------------
We add '<image>' as a new special token to the Qwen2 tokenizer (gets ID 151665
after the reserved slots).  In the prompt, each image is represented as a
single '<image>' token.

In gym_agent_loop.py, CambrianProcessorWrapper.__call__() is invoked to
tokenize the full multi-turn prompt.  It returns input_ids where each '<image>'
occurrence is replaced by TOKENS_PER_IMAGE copies of IMAGE_TOKEN_INDEX (-200).
This pre-expansion is what the FSDP adapter (cambrian_register.py) expects.

For vLLM inference, gym_agent_loop.py sends compact prompt_ids (with single
'<image>' tokens) via server_manager.generate().  The Cambrian vLLM model
(cambrian_vllm.py) expands them back to visual feature positions internally.
"""

import os
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_TOKEN: str = "<image>"
IMAGE_TOKEN_INDEX: int = -200          # sentinel used by FSDP adapter
TOKENS_PER_IMAGE: int = 756            # si_side_len * (si_side_len + 1) = 27 * 28
SIGLIP_MODEL: str = "google/siglip2-so400m-patch14-384"
SIGLIP_SIZE: int = 384


# ---------------------------------------------------------------------------
# Dummy image processor class (name is checked in agent_loop_no_concat.py)
# ---------------------------------------------------------------------------
class CambrianSiglipImageProcessor:
    """Sentinel class – __class__.__name__ is used to detect the Cambrian branch."""
    pass


# ---------------------------------------------------------------------------
# SigLIP image preprocessing helper
# ---------------------------------------------------------------------------
def _expand2square(pil_img: Image.Image, background_color) -> Image.Image:
    """Pad image to square with background fill, then resize to SIGLIP_SIZE."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess_siglip_images(
    images: List[Image.Image],
    siglip_processor,
) -> torch.Tensor:
    """
    Preprocess a list of PIL images for SigLIP.
    Returns pixel_values of shape (N, 3, SIGLIP_SIZE, SIGLIP_SIZE).
    """
    # SigLIP2 uses mean=std=[0.5, 0.5, 0.5]
    bg_color = tuple(int(x * 255) for x in (0.5, 0.5, 0.5))
    processed = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = _expand2square(img, bg_color)
        img = img.resize((SIGLIP_SIZE, SIGLIP_SIZE), Image.BICUBIC)
        result = siglip_processor(images=img, return_tensors="pt")
        processed.append(result["pixel_values"].squeeze(0))  # (3, H, W)
    return torch.stack(processed)  # (N, 3, H, W)


# ---------------------------------------------------------------------------
# Main processor wrapper
# ---------------------------------------------------------------------------
class CambrianProcessorWrapper:
    """
    Minimal processor wrapper for Cambrian-S in VAGEN-Lite.

    Responsibilities:
      - Add '<image>' as a special token to the Qwen2 tokenizer.
      - apply_chat_template(): delegates to the tokenizer.
      - __call__(text, images): tokenizes text; expands each '<image>' token
        to TOKENS_PER_IMAGE copies of IMAGE_TOKEN_INDEX=-200 so that the
        resulting input_ids match the FSDP adapter's expectation.
      - preprocess_images(): converts PIL images to SigLIP pixel_values
        for the FSDP training forward pass.
      - image_processor: sentinel object whose class name triggers the
        Cambrian branch in agent_loop_no_concat.py.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        siglip_model_name: str = SIGLIP_MODEL,
    ):
        self.tokenizer = tokenizer

        # Add '<image>' as a special token (gets a new vocab ID, e.g. 151665)
        if IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_tokens([IMAGE_TOKEN], special_tokens=True)
        self.image_token_id: int = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        # Sentinel object for branch detection in agent_loop
        self.image_processor = CambrianSiglipImageProcessor()

        # Load SigLIP image processor (cached after first load)
        self._siglip_processor = None
        self._siglip_model_name = siglip_model_name

    def _get_siglip_processor(self):
        if self._siglip_processor is None:
            self._siglip_processor = AutoImageProcessor.from_pretrained(
                self._siglip_model_name
            )
        return self._siglip_processor

    # ------------------------------------------------------------------
    # apply_chat_template – used in gym_agent_loop.py
    # ------------------------------------------------------------------
    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        # Qwen2 chat template expects str content, but agent_loop passes
        # multimodal list content: [{"type":"text","text":"..."},{"type":"image"}].
        # Flatten to a plain string with '<image>' placeholders.
        flat_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    btype = block.get("type")
                    if btype == "image":
                        parts.append(IMAGE_TOKEN)
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

    # ------------------------------------------------------------------
    # __call__ – tokenize text, keeping <image> as a single token
    # ------------------------------------------------------------------
    def __call__(
        self,
        text: Union[str, List[str]],
        images=None,          # PIL images (passed by gym_agent_loop but ignored here)
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict:
        """
        Tokenize text with '<image>' as a single special token per image.

        Returns input_ids where each '<image>' in the text becomes a single
        image_token_id (151665).  This compact form is sent to vLLM for rollout
        (vLLM's CambrianMultiModalProcessor expands to 756 <|image_pad|> internally).

        In agent_loop_no_concat.py's Cambrian branch, these single <image> tokens
        are expanded to IMAGE_TOKEN_INDEX=-200 blocks (×756) for the FSDP actor.

        Note: images argument is intentionally unused here.  PIL images are
        stored in agent_data.image_data and processed separately via
        preprocess_images() for the FSDP forward pass pixel_values.
        """
        if isinstance(text, str):
            text = [text]

        all_input_ids = []
        for t in text:
            # Tokenize with the special <image> token → single ID per image
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            all_input_ids.append(token_ids)

        if return_tensors == "pt":
            # Pad to same length within this batch
            max_len = max(len(ids) for ids in all_input_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            attention_masks = []
            padded_ids = []
            for ids in all_input_ids:
                pad_len = max_len - len(ids)
                padded_ids.append([pad_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        else:
            return {"input_ids": all_input_ids}

    # ------------------------------------------------------------------
    # preprocess_images – called in agent_loop_no_concat.py Cambrian branch
    # ------------------------------------------------------------------
    def preprocess_images(
        self,
        images: List[Image.Image],
    ) -> dict:
        """
        Convert PIL images to SigLIP pixel_values for the FSDP forward pass.
        Returns {"pixel_values": tensor of shape (N, 3, 384, 384)}.
        """
        siglip = self._get_siglip_processor()
        pixel_values = preprocess_siglip_images(images, siglip)
        return {"pixel_values": pixel_values}
