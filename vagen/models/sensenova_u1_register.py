"""
Registration module for SenseNova-U1 in VAGEN-Lite's FSDP training path.

The upstream SenseNova-U1 examples use ``model.chat()`` / ``model.generate()``
for inference and leave ``NEOChatModel.forward()`` disabled.  PPO needs
forward passes for actor logprobs, reference logprobs, and critic values.  This
module is imported through VERL's ``external_lib`` hook and registers adapter
classes that enable those training-time forwards.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification
from transformers.modeling_outputs import CausalLMOutputWithPast, TokenClassifierOutput


SENSENOVA_U1_SRC = os.environ.get("SENSENOVA_U1_SRC", "/nas/baiqiao/SenseNova-U1/src")
if SENSENOVA_U1_SRC not in sys.path:
    sys.path.insert(0, SENSENOVA_U1_SRC)

import sensenova_u1  # noqa: E402,F401 - registers AutoConfig/AutoModel
from sensenova_u1.models.neo_unify import NEOChatConfig, NEOChatModel  # noqa: E402


IMG_CONTEXT_TOKEN_ID = 151669
IMG_START_TOKEN_ID = 151670


class SenseNovaU1ForCausalLMAdapter(NEOChatModel):
    """SenseNova-U1 with a PPO-compatible forward path."""

    def _build_indexes(
        self,
        input_ids: torch.LongTensor,
        grid_hw: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
    ) -> torch.LongTensor:
        if position_ids is not None and position_ids.dim() == 3:
            return position_ids
        indexes = [
            self.get_thw_indexes(input_ids[b], grid_hw=grid_hw)
            for b in range(input_ids.shape[0])
        ]
        return torch.stack(indexes, dim=0)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        grid_hw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if input_ids is None:
            raise ValueError("input_ids is required for SenseNovaU1ForCausalLMAdapter.forward().")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.img_context_token_id = self.img_context_token_id or IMG_CONTEXT_TOKEN_ID
        self.img_start_token_id = self.img_start_token_id or IMG_START_TOKEN_ID

        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        if pixel_values is not None and pixel_values.numel() > 0:
            if grid_hw is None:
                raise ValueError("grid_hw is required when pixel_values are provided.")
            vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)

            bsz, seq_len, hidden = input_embeds.shape
            flat_embeds = input_embeds.reshape(bsz * seq_len, hidden)
            flat_ids = input_ids.reshape(bsz * seq_len)
            selected = flat_ids == self.img_context_token_id
            expected = int(selected.sum().item())
            actual = int(vit_embeds.numel() // hidden)
            if expected != actual:
                raise ValueError(
                    "SenseNova-U1 image-token mismatch: "
                    f"input has {expected} <IMG_CONTEXT> tokens but vision encoder produced {actual} embeddings."
                )
            flat_embeds[selected] = vit_embeds.reshape(-1, hidden).to(flat_embeds.dtype)
            input_embeds = flat_embeds.reshape(bsz, seq_len, hidden)

        indexes = self._build_indexes(input_ids, grid_hw=grid_hw, position_ids=position_ids)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + tuple(v for v in (outputs.past_key_values, outputs.hidden_states, outputs.attentions) if v is not None)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SenseNovaU1ForTokenClassification(SenseNovaU1ForCausalLMAdapter):
    """Token-classification/value-head wrapper for VERL's PPO critic."""

    def __init__(self, config: NEOChatConfig):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 1)
        hidden_size = config.llm_config.hidden_size
        self.score = nn.Linear(hidden_size, self.num_labels, bias=False)

    def forward(self, labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None, **kwargs):
        kwargs["output_hidden_states"] = True
        outputs = super().forward(labels=None, return_dict=True, **kwargs)
        hidden_states = outputs.hidden_states
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[-1]
        logits = self.score(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoModelForCausalLM.register(NEOChatConfig, SenseNovaU1ForCausalLMAdapter, exist_ok=True)
AutoModelForTokenClassification.register(
    NEOChatConfig,
    SenseNovaU1ForTokenClassification,
    exist_ok=True,
)

print(
    "[sensenova_u1_register] Registered SenseNovaU1ForCausalLMAdapter and "
    f"SenseNovaU1ForTokenClassification (SENSENOVA_U1_SRC={SENSENOVA_U1_SRC})"
)
