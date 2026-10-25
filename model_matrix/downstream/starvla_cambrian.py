"""Cambrian-S as StarVLA's vision-language backbone (the model line, step 2, the user's section 4 of 2026-10-24): an
interface with the methods QwenOFT uses, registered at run time (``register``), StarVLA's own code unchanged.

QwenOFT asks of a backbone: ``model.config.hidden_size`` (the action head's width; Cambrian-S: 3584, as Qwen2.5-VL-7B),
``processor.tokenizer`` (the action token "🔍" = 146663: Cambrian-S's tokenizer is Qwen2.5-VL-7B's, every id the
same), ``build_qwenvl_inputs(images, instructions)`` and a forward whose ``hidden_states[-1]`` lines up with
``input_ids``.  This interface gives them with Cambrian-S's own classes: the repository's adapter
(vagen/models/cambrian_register.py, the Cambrian source's model) for the weights, the SigLIP2 tower and projector,
SigLIP2's preprocessing (vagen.models.cambrian_processor), and Cambrian-S's chat format (conv_qwen: ChatML with the
system prompt "You are a helpful assistant.").

Several camera views, the Cambrian way (its "miv" form, cambrian/model/cambrian_arch.py and train_fsdp.py: several
images or video frames stand behind one <image>; each frame's 27 x 27 projected features are resized bilinearly to
miv_token_len = 64 = 8 x 8, with the learned newline after each row): 72 positions per view.  Framework option
``framework.qwenvl.cambrian_views: si`` gives each view the single-image form instead (756 positions: 27 x 27 and
the newlines), Cambrian-S's form for one picture.  Gradient checkpointing (the language model and the SigLIP2
tower) is on unless ``framework.qwenvl.enable_gradient_checkpointing: false`` (StarVLA's own switch, off by default
for its other backbones): the same gradients, recomputed.

Positions follow the attention mask (left padding).  The language model's hidden states are returned (no LM head:
QwenOFT reads only the last hidden state).  The language model runs through Transformers' own Qwen2Model.forward
(the Cambrian model's parent class: the same layers and rotary embedding): the Cambrian source's loop takes
``layer_outputs[0]``, which since Transformers 4.54 (a decoder layer returns its hidden states) is the batch's first
sample -- right for the batch of one that RL (remove-padding) and SFT use, wrong for StarVLA's batches
(model_matrix/tests/test_starvla_cambrian.py checks both on a small model).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_TOKEN_INDEX = -200  # the adapter's image positions
SYSTEM = "You are a helpful assistant."  # Cambrian-S's conv_qwen
VIEW_FORMS = {"miv": 8, "si": 27}  # grid side per view: 8 x 8 (+ 8 newlines) = 72, 27 x 27 (+ 27) = 756


def is_cambrian(path) -> bool:
    try:
        return json.loads((Path(path) / "config.json").read_text()).get("model_type") == "cambrian_qwen"
    except (OSError, ValueError, TypeError):
        return False


def tokens_per_view(form: str) -> int:
    side = VIEW_FORMS[form]
    return side * (side + 1)


def view_features(features: torch.Tensor, newline: torch.Tensor, side: int) -> torch.Tensor:
    """(views, 729, D) projected features -> (views, side * (side + 1), D): the grid resized bilinearly to side x side
    (as cambrian_arch for miv; unchanged at 27), the newline after each row."""
    n, length, d = features.shape
    full = int(round(length ** 0.5))
    grid = features.unflatten(1, (full, full)).permute(0, 3, 1, 2)  # (views, D, 27, 27)
    if side != full:
        grid = F.interpolate(grid, size=(side, side), mode="bilinear", align_corners=False).type_as(features)
    rows = grid.permute(0, 2, 3, 1)  # (views, side, side, D)
    nl = newline.to(rows.dtype).view(1, 1, 1, d).expand(n, side, 1, d)
    return torch.cat([rows, nl], dim=2).reshape(n, side * (side + 1), d)


def language_model_forward(inner, embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    """The Cambrian model's language model on a batch: Transformers' Qwen2Model.forward (see the module docstring),
    positions from the attention mask (left padding).  Returns last_hidden_state and hidden_states."""
    from transformers import Qwen2Model
    from transformers.modeling_outputs import BaseModelOutputWithPast

    position_ids = None
    if attention_mask is not None:
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    out = Qwen2Model.forward(inner, inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids,
                             use_cache=False, output_hidden_states=True, return_dict=True)
    return BaseModelOutputWithPast(last_hidden_state=out.last_hidden_state, hidden_states=out.hidden_states)


class CambrianVLInterface(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        from transformers import AutoImageProcessor, AutoTokenizer

        from vagen.models import cambrian_register as R
        from vagen.models.cambrian_processor import SIGLIP_CACHE, SIGLIP_MODEL

        qwenvl = config.framework.get("qwenvl", {})
        path = qwenvl.get("base_vlm")
        self.form = str(qwenvl.get("cambrian_views", "miv"))
        if self.form not in VIEW_FORMS:
            raise ValueError(f"framework.qwenvl.cambrian_views must be one of {sorted(VIEW_FORMS)}")
        attn = qwenvl.get("attn_implementation", "sdpa")
        if attn == "flash_attention_2":
            from starVLA.model.tools import has_flash_attn

            if not has_flash_attn():
                attn = "sdpa"
        self.model = R.CambrianForCausalLMAdapter.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                                   attn_implementation=attn)
        if qwenvl.get("enable_gradient_checkpointing", True):
            # StarVLA's switch (its Gemma4 and MiniCPM-V interfaces, off there by default), on by default here: the
            # recipe's 8 samples x 3 views per card are 24 SigLIP2 images of 729 patches each, whose activations (26
            # trainable layers) do not fit next to the language model's; the first 8-card smoke ran out of memory in
            # its first step (pt-xmljmr8z, 2026-10-25).  Recomputing them in the backward pass gives the same gradients
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.padding_side = "left"
        image_processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=SIGLIP_CACHE, local_files_only=True)
        self.processor = SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)
        self.config = config

    # ----------------------------------------------------------------------------------------------- inputs
    def prompt_ids(self, prompt: str, views: int) -> List[int]:
        tok = self.processor.tokenizer
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": "<image>\n" + prompt}],
                                       tokenize=False, add_generation_prompt=True)
        before, _, after = text.partition("<image>")
        n = views * tokens_per_view(self.form)
        return (tok.encode(before, add_special_tokens=False) + [IMAGE_TOKEN_INDEX] * n
                + tok.encode(after, add_special_tokens=False))

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        from vagen.models.cambrian_processor import preprocess_siglip_images

        if solutions is not None:
            raise NotImplementedError("Cambrian backbone: no language targets (QwenOFT trains only the action head)")
        if len(images) != len(instructions):
            raise AssertionError("Images and instructions must have the same length")
        cot = self.config.datasets.vla_data.get("CoT_prompt", None) if "CoT_prompt" in self.config.datasets.vla_data \
            else None
        rows, counts = [], []
        for imgs, instruction in zip(images, instructions):
            prompt = cot.replace("{instruction}", instruction) if cot else instruction
            rows.append(self.prompt_ids(prompt, len(imgs)))
            counts.append(len(imgs))
        pad = self.processor.tokenizer.pad_token_id
        width = max(len(r) for r in rows)
        input_ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], dtype=torch.long)
        attention_mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], dtype=torch.long)
        frames = [im for imgs in images for im in imgs]
        pixel_values = preprocess_siglip_images(frames, self.processor.image_processor) if frames else None
        device = next(self.model.parameters()).device
        out = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
               "views_per_sample": torch.tensor(counts, dtype=torch.long)}
        if pixel_values is not None:
            out["pixel_values"] = pixel_values.to(device)
        return out

    # ---------------------------------------------------------------------------------------------- forward
    def embed(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor]) -> torch.Tensor:
        inner = self.model.get_model()
        embeds = inner.embed_tokens(input_ids.clamp(min=0))
        where = input_ids == IMAGE_TOKEN_INDEX
        if pixel_values is None or not bool(where.any()):
            return embeds
        features = self.model.encode_images([pixel_values.to(embeds.dtype)])[0]  # (views, 729, 1152)
        projector = inner.mm_projector
        features = projector(features.to(projector[0].weight.dtype))  # (views, 729, D)
        features = view_features(features, inner.image_newline, VIEW_FORMS[self.form])
        flat = features.reshape(-1, features.shape[-1]).to(embeds.dtype)
        if flat.shape[0] != int(where.sum()):
            raise ValueError(f"{int(where.sum())} image positions, {flat.shape[0]} view features")
        embeds = embeds.clone()
        embeds[where] = flat  # rows in order: sample by sample, view by view, row by row
        return embeds

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, output_hidden_states=True,
                return_dict=True, views_per_sample=None, **kwargs):
        embeds = self.embed(input_ids, pixel_values)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return language_model_forward(self.model.get_model(), embeds, attention_mask)

    def generate(self, **kwargs):
        raise NotImplementedError("Cambrian backbone: QwenOFT does not generate text")


def register() -> None:
    """StarVLA's get_vlm_model (and QwenOFT's own reference to it) gives CambrianVLInterface for a Cambrian-S start,
    StarVLA's interface for every other start."""
    import starVLA.model.modules.vlm as vlm

    if getattr(vlm.get_vlm_model, "_model_line_cambrian", False):
        return
    original = vlm.get_vlm_model

    def get_vlm_model(config):
        if is_cambrian(config.framework.qwenvl.base_vlm):
            return CambrianVLInterface(config)
        return original(config)

    get_vlm_model._model_line_cambrian = True
    get_vlm_model._original = original
    vlm.get_vlm_model = get_vlm_model
    import starVLA.model.framework.VLM4A.QwenOFT as oft

    oft.get_vlm_model = get_vlm_model
