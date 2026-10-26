"""
Registration module for SenseNova-U1 in VAGEN-Lite's FSDP training path.

Phase-1 Plan B: open-loop next-frame flow-matching aux via a split
prefix (und) + gen forward, matching official ``it2i_generate`` / ``_t2i_predict_v``.
The HF decoder raises on mixed und/gen sequences, so gen tokens are NOT packed
into the PPO token sequence; gen targets arrive as ``u1_gen_*`` multimodal fields.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification
from transformers.modeling_outputs import CausalLMOutputWithPast, TokenClassifierOutput


SENSENOVA_U1_SRC = os.environ.get("SENSENOVA_U1_SRC", "/mnt/umm/users/yinbaiqiao/SenseNova-U1/src")
if SENSENOVA_U1_SRC not in sys.path:
    sys.path.insert(0, SENSENOVA_U1_SRC)

import sensenova_u1  # noqa: E402,F401
from sensenova_u1.models.neo_unify import NEOChatConfig, NEOChatModel  # noqa: E402
from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask  # noqa: E402


IMG_CONTEXT_TOKEN_ID = 151669
IMG_START_TOKEN_ID = 151670


class SenseNovaU1ForCausalLMAdapter(NEOChatModel):
    """SenseNova-U1 with PPO forward + open-loop FM aux (split und/gen)."""

    def _build_indexes(
        self,
        input_ids: torch.LongTensor,
        grid_hw: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
    ) -> torch.LongTensor:
        if input_ids.shape[0] != 1:
            raise ValueError(
                "SenseNovaU1ForCausalLMAdapter currently requires micro-batch size 1 "
                f"(got batch={input_ids.shape[0]})."
            )
        if position_ids is not None and position_ids.dim() == 3:
            if position_ids.shape[0] == 3 and position_ids.shape[1] == input_ids.shape[1]:
                return position_ids
            if position_ids.shape[0] == 1 and position_ids.shape[1] == 3:
                return position_ids[0]
        return self.get_thw_indexes(input_ids[0], grid_hw=grid_hw)

    def _build_block_attention_mask(
        self,
        indexes: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> dict:
        """Build MoT block-causal mask; keep pad query rows finite.

        Padding must NOT zero-out entire query rows. Eager softmax on an
        all-(-inf) row yields NaN; those NaNs sit in V and leak into every
        valid token on the next layer via ``0 * nan``. Official U1 inference
        usually has no pad; PPO left/right-pads to max length, so we:
          1) mask padded keys for all queries;
          2) give padded queries a self-attention sink only.
        """
        block = create_block_causal_mask(indexes[0])
        if attention_mask is not None and torch.is_tensor(attention_mask):
            valid = attention_mask[0].bool()
            if (~valid).any():
                block = block.clone()
                # Mask padded keys (columns).
                block[:, :, :, ~valid] = float("-inf")
                # Pad queries: self-only sink so softmax stays finite.
                block[:, :, ~valid, :] = float("-inf")
                pad_idx = torch.where(~valid)[0]
                block[:, :, pad_idx, pad_idx] = 0.0
        return {"full_attention": block}

    def _prepare_single_gen_target(self, pixel_values: torch.Tensor, grid_hw: torch.LongTensor):
        """Noise one gen image; return noisy patches + FM targets (merged tokens)."""
        patch_size = int(self.patch_size)
        merge_size = int(round(1 / self.downsample_ratio))
        patch_size_after_downsample = patch_size * merge_size
        t_eps = float(getattr(self.config, "t_eps", 0.05))
        p_mean = float(getattr(self.config, "P_mean", -0.8))
        p_std = float(getattr(self.config, "P_std", 0.8))
        timestep_shift = float(getattr(self.config, "timestep_shift", 1.0))

        cur_h = int(grid_hw[0, 0].item())
        cur_w = int(grid_hw[0, 1].item())
        cur_n = cur_h * cur_w
        assert pixel_values.shape[0] == cur_n, (pixel_values.shape, cur_n)

        und_mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        und_std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)

        cur = pixel_values.clone().view(-1, 3, patch_size, patch_size)
        cur = (cur * und_std + und_mean).clamp(0, 1).view(-1, 3 * patch_size * patch_size)
        cur = (cur - 0.5) * 2

        image_seq_len = cur_n // (merge_size ** 2)
        noise_scale = float(self.noise_scale)
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt(image_seq_len / base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, float(self.noise_scale_max_value))

        cur_noise = torch.randn_like(cur) * noise_scale
        u = torch.normal(mean=0.0, std=1.0, size=(1,), device=pixel_values.device) * p_std + p_mean
        t = (1 / (1 + torch.exp(-u))).to(dtype=pixel_values.dtype, device=pixel_values.device)
        t = self._apply_time_schedule(t, image_seq_len, timestep_shift)

        t_expanded = t.expand(cur_n)
        t_merged = t.expand(image_seq_len)
        cur_z_patch = t_expanded.view(-1, 1) * cur + (1 - t_expanded.view(-1, 1)) * cur_noise

        cur_x_m = cur.view(cur_h // merge_size, merge_size, cur_w // merge_size, merge_size, 3, patch_size, patch_size)
        cur_x_m = torch.einsum("h a w b c i j -> h w a i b j c", cur_x_m).contiguous()
        cur_x_m = cur_x_m.view(-1, patch_size_after_downsample ** 2 * 3)

        cur_z_m = cur_z_patch.view(cur_h // merge_size, merge_size, cur_w // merge_size, merge_size, 3, patch_size, patch_size)
        cur_z_m = torch.einsum("h a w b c i j -> h w a i b j c", cur_z_m).contiguous()
        cur_z_m = cur_z_m.view(-1, patch_size_after_downsample ** 2 * 3)

        image_gen_v = (cur_x_m - cur_z_m) / (1 - t_merged.view(-1, 1)).clamp_min(t_eps)
        ns = torch.full_like(t_merged, noise_scale / float(self.noise_scale_max_value))
        return cur_z_patch, cur_z_m, image_gen_v, t_merged, ns

    def _predict_x_from_hidden(self, gen_hidden: torch.Tensor, image_gen_t: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        if self.use_pixel_head:
            hs = gen_hidden.view(1, token_h, token_w, -1)
            img_2d = torch.einsum("b h w c -> b c h w", hs)
            smoothed = self.fm_modules["fm_head"](img_2d)
            smoothed = smoothed.view(1, 3, token_h, self.patch_size * int(1 / self.downsample_ratio), token_w, self.patch_size * int(1 / self.downsample_ratio))
            smoothed = torch.einsum("b c h p w q -> b h w p q c", smoothed)
            merge = int(1 / self.downsample_ratio)
            return smoothed.contiguous().view(token_h * token_w, (self.patch_size * merge) ** 2 * 3)
        if self.use_deep_fm_head:
            return self.fm_modules["fm_head"](gen_hidden, image_gen_t)
        return self.fm_modules["fm_head"](gen_hidden)

    def _detach_past_key_values(self, past):
        """Detach KV leaves so FM gen path does not backprop through und prefix."""
        if past is None:
            return None
        try:
            if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
                past.key_cache = [k.detach() if torch.is_tensor(k) else k for k in past.key_cache]
                past.value_cache = [v.detach() if torch.is_tensor(v) else v for v in past.value_cache]
                return past
            if isinstance(past, (tuple, list)):
                return tuple(
                    tuple(t.detach() if torch.is_tensor(t) else t for t in layer)
                    if isinstance(layer, (tuple, list))
                    else (layer.detach() if torch.is_tensor(layer) else layer)
                    for layer in past
                )
        except Exception:
            pass
        return past

    def _compute_fm_aux_split(
        self,
        indexes: torch.LongTensor,
        u1_gen_pixel_values: torch.Tensor,
        u1_gen_grid_hw: torch.LongTensor,
        past_key_values=None,
    ) -> torch.Tensor:
        """Reuse und KV from main forward + gen-only forward + velocity MSE.

        Avoids a second full und forward (OOM under colocated FSDP+vLLM).
        """
        merge_size = int(round(1 / self.downsample_ratio))
        token_h = int(u1_gen_grid_hw[0, 0].item()) // merge_size
        token_w = int(u1_gen_grid_hw[0, 1].item()) // merge_size
        n_gen = token_h * token_w
        t_eps = float(getattr(self.config, "t_eps", 0.05))

        noisy_patches, image_gen_z, image_gen_v, image_gen_t, image_gen_ns = self._prepare_single_gen_target(
            u1_gen_pixel_values, u1_gen_grid_hw
        )

        past = self._detach_past_key_values(past_key_values)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gen_vit = self.extract_feature(noisy_patches, gen_model=True, grid_hw=u1_gen_grid_hw)
        t_emb = self.fm_modules["timestep_embedder"](image_gen_t)
        if self.add_noise_scale_embedding:
            t_emb = t_emb + self.fm_modules["noise_scale_embedder"](image_gen_ns)
        gen_embeds = (gen_vit + t_emb.to(dtype=gen_vit.dtype)).unsqueeze(0)  # (1, N, H)

        indexes_gen = self._build_t2i_image_indexes(
            token_h, token_w, int(indexes[0].max().item()) + 1, device=gen_embeds.device
        )
        gen_ind = torch.ones((1, n_gen), dtype=torch.bool, device=gen_embeds.device)
        gen_attn = {"full_attention": None}

        gen_out = self.language_model.model(
            inputs_embeds=gen_embeds,
            indexes=indexes_gen,
            attention_mask=gen_attn,
            past_key_values=past,
            use_cache=False,
            image_gen_indicators=gen_ind,
            update_cache=False,
        )
        gen_hidden = gen_out.last_hidden_state[0]  # (N, H)

        x_pred = self._predict_x_from_hidden(gen_hidden, image_gen_t, token_h, token_w)
        v_pred = (x_pred - image_gen_z) / (1 - image_gen_t.view(-1, 1)).clamp_min(t_eps)
        per_tok = F.mse_loss(v_pred.float(), image_gen_v.float(), reduction="none").mean(dim=1)
        # Official training uses 1/sqrt(seq_len) weights; single-image => mean is enough.
        return per_tok.mean()

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
        u1_gen_pixel_values: Optional[torch.FloatTensor] = None,
        u1_gen_grid_hw: Optional[torch.LongTensor] = None,
        u1_gen_valid: Optional[torch.Tensor] = None,
        image_for_gen_flags: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # FSDP-safe entry for the second FM pass. Calling
        # ``module.compute_pending_fm_aux_loss()`` on the unwrapped module
        # bypasses the root FSDP pre-forward all-gather and can expose
        # zero-sized resharded storage. Routing the pending pass through the
        # wrapper's regular forward keeps the full FSDP materialize/reshard
        # lifecycle around every FM parameter access.
        if kwargs.pop("_u1_compute_pending_only", False):
            return self.compute_pending_fm_aux_loss()

        kwargs.pop("image_flags", None)
        kwargs.pop("nfp_pixel_values", None)
        kwargs.pop("nfp_loss_mask", None)
        kwargs.pop("image_for_gen_flags", None)

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
        attn = self._build_block_attention_mask(indexes, attention_mask)

        # Decide FM participation before the und forward so we can request KV
        # cache once and reuse it (avoids a second full und pass / OOM).
        local_valid = False
        if u1_gen_valid is not None:
            local_valid = bool(torch.as_tensor(u1_gen_valid).view(-1)[0].item())
        has_gen_pixels = (
            u1_gen_pixel_values is not None
            and u1_gen_grid_hw is not None
            and torch.as_tensor(u1_gen_pixel_values).numel() > 0
        )
        local_valid = bool(local_valid and has_gen_pixels)

        any_valid = local_valid
        if self.training and torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = torch.tensor(
                [1.0 if local_valid else 0.0],
                device=input_ids.device,
                dtype=torch.float32,
            )
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            any_valid = bool(flag.item() > 0.0)

        want_fm = bool(self.training and any_valid)
        # Plan B FM (memory-safe):
        #   - und forward with use_cache when FM needed (condition gen on prefix)
        #   - DETACH und KV so FM does not backprop through the full und sequence
        #     (avoids OOM under colocated FSDP+vLLM + grad checkpoint)
        #   - real grads through gen Vit / mot_gen / fm_head
        # Override: U1_FM_USE_UND_KV=0 → gen-only FM (still with grads)
        #           U1_FM_BACKPROP=0 → metrics-only (legacy smoke path)
        fm_use_und_kv = os.environ.get("U1_FM_USE_UND_KV", "0") != "0"
        fm_backprop = os.environ.get("U1_FM_BACKPROP", "1") != "0"
        und_use_cache = bool(use_cache) or bool(want_fm and fm_use_und_kv and fm_backprop)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            indexes=indexes,
            attention_mask=attn,
            past_key_values=past_key_values,
            use_cache=und_use_cache,
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

        # Open-loop FM aux must be collective-safe under FSDP: every rank must
        # execute the same module sequence. Mixing u1_gen_valid across ranks
        # (non-terminal vs terminal turns in one PPO batch) previously caused
        # NCCL desync (some ranks ALLGATHER in gen forward, others wait on
        # ALLREDUCE).
        # Clear any stale two-pass FM stash from a previous micro-batch.
        self._u1_pending_fm = None
        if want_fm:
            if local_valid:
                gen_pv = u1_gen_pixel_values.to(device=input_embeds.device, dtype=input_embeds.dtype)
                gen_hw = u1_gen_grid_hw.to(device=input_ids.device)
            else:
                # Dummy 32x32 grid / zero patches so gen Vit+LM still run.
                gen_hw = torch.tensor([[32, 32]], device=input_ids.device, dtype=torch.long)
                n_tok = int(32 * 32)  # patch tokens before vision merge inside extract_feature
                patch = 3 * 16 * 16
                gen_pv = torch.zeros(n_tok, patch, device=input_embeds.device, dtype=input_embeds.dtype)

            past_for_fm = None
            if fm_use_und_kv:
                # Detach: condition gen on und prefix without und activation grads.
                past_for_fm = self._detach_past_key_values(outputs.past_key_values)

            scale = 1.0 if local_valid else 0.0
            if fm_backprop:
                # Two-pass Plan B: do NOT attach FM to the und/PG graph.
                # Actor runs PG backward first (frees und activations), then
                # compute_pending_fm_aux_loss() for a gen-only backward.
                # CPU clones so PG/FSDP backward cannot invalidate storage views.
                self._u1_pending_fm = {
                    "indexes": indexes.detach().to("cpu").contiguous().clone(),
                    "gen_pv": gen_pv.detach().to("cpu").contiguous().clone(),
                    "gen_hw": gen_hw.detach().to("cpu").contiguous().clone(),
                    "past": None,  # und KV not kept across twopass for memory/safety
                    "scale": scale,
                    "und_kv": False,
                    "device": str(input_embeds.device),
                    "dtype": str(input_embeds.dtype).replace("torch.", ""),
                }
            else:
                with torch.no_grad():
                    fm_val = self._compute_fm_aux_split(
                        indexes=indexes,
                        u1_gen_pixel_values=gen_pv,
                        u1_gen_grid_hw=gen_hw,
                        past_key_values=past_for_fm,
                    )
                if not torch.isfinite(fm_val).all():
                    fm_val = torch.zeros((), device=input_embeds.device, dtype=input_embeds.dtype)
                    if not getattr(self, "_vagen_image_gen_nan_logged", False):
                        print("[sensenova_u1_register] image_gen FM aux produced non-finite loss; zeroed")
                        self._vagen_image_gen_nan_logged = True
                fm_loss = fm_val.detach().to(dtype=input_embeds.dtype) * scale + (input_embeds.sum() * 0)
                loss = fm_loss if loss is None else (loss + fm_loss)
                if local_valid and not getattr(self, "_vagen_image_gen_loss_logged", False):
                    print(
                        "[sensenova_u1_register] image_gen FM aux active "
                        f"(backprop=0 und_kv={int(fm_use_und_kv)}) "
                        f"loss={float(fm_loss.detach().item()):.6f}"
                    )
                    self._vagen_image_gen_loss_logged = True

        if not return_dict:
            output = (logits,) + tuple(
                v for v in (outputs.past_key_values, outputs.hidden_states, outputs.attentions) if v is not None
            )
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def compute_pending_fm_aux_loss(self) -> Optional[torch.Tensor]:
        """Run stashed gen-only FM after PG backward (two-pass Plan B).

        Returns a scalar loss with graph through gen Vit / mot_gen / fm_head,
        or None if nothing was stashed. Always clears the stash. Collective-safe:
        every rank that stashed (all ranks when any_valid) must call this.
        """
        pending = getattr(self, "_u1_pending_fm", None)
        self._u1_pending_fm = None
        if not pending:
            return None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = torch.device(pending.get("device", "cuda"))
        # dtype may be "bfloat16" / "float16"
        dt = getattr(torch, pending.get("dtype", "bfloat16"), torch.bfloat16)
        indexes = pending["indexes"].to(device=device)
        gen_pv = pending["gen_pv"].to(device=device, dtype=dt)
        gen_hw = pending["gen_hw"].to(device=device)
        fm_val = self._compute_fm_aux_split(
            indexes=indexes,
            u1_gen_pixel_values=gen_pv,
            u1_gen_grid_hw=gen_hw,
            past_key_values=None,
        )
        if not torch.isfinite(fm_val).all():
            fm_val = torch.zeros((), device=fm_val.device, dtype=fm_val.dtype)
            if not getattr(self, "_vagen_image_gen_nan_logged", False):
                print("[sensenova_u1_register] image_gen FM aux produced non-finite loss; zeroed")
                self._vagen_image_gen_nan_logged = True
        scale = float(pending["scale"])
        fm_loss = fm_val * scale
        if scale > 0.0 and not getattr(self, "_vagen_image_gen_loss_logged", False):
            print(
                "[sensenova_u1_register] image_gen FM aux active "
                f"(backprop=1 twopass=1 und_kv={int(pending.get('und_kv', False))}) "
                f"loss={float(fm_loss.detach().item()):.6f}"
            )
            self._vagen_image_gen_loss_logged = True
        return fm_loss



class SenseNovaU1ForTokenClassification(SenseNovaU1ForCausalLMAdapter):
    """Token-classification/value-head wrapper for VERL's PPO critic."""

    def __init__(self, config: NEOChatConfig):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 1)
        hidden_size = config.llm_config.hidden_size
        self.score = nn.Linear(hidden_size, self.num_labels, bias=False)

    def forward(self, labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("u1_gen_pixel_values", None)
        kwargs.pop("u1_gen_grid_hw", None)
        kwargs.pop("u1_gen_valid", None)
        kwargs.pop("image_for_gen_flags", None)
        kwargs["output_hidden_states"] = True
        # Force eval-style und-only path (no FM) even if module.training.
        was_training = self.training
        self.training = False
        try:
            outputs = SenseNovaU1ForCausalLMAdapter.forward(self, labels=None, return_dict=True, **kwargs)
        finally:
            self.training = was_training
        hidden_states = outputs.hidden_states
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[-1]
        elif hidden_states is None:
            # Qwen3ForCausalLM always fills hidden_states with last_hidden_state.
            raise RuntimeError("Critic forward missing hidden_states")
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


def _register_u1_auto_models(config_cls, label: str) -> None:
    """Register adapter/critic against a concrete NEOChatConfig class object."""
    if config_cls is None:
        return
    AutoModelForCausalLM.register(config_cls, SenseNovaU1ForCausalLMAdapter, exist_ok=True)
    AutoModelForTokenClassification.register(
        config_cls,
        SenseNovaU1ForTokenClassification,
        exist_ok=True,
    )
    print(f"[sensenova_u1_register] Registered AutoModels for {label} ({config_cls})")


_register_u1_auto_models(NEOChatConfig, "package NEOChatConfig")
try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _model_dir = os.environ.get(
        "SENSENOVA_U1_MODEL_PATH",
        "/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT",
    )
    _DynCfg = get_class_from_dynamic_module("configuration_neo_chat.NEOChatConfig", _model_dir)
    _register_u1_auto_models(_DynCfg, "dynamic_module NEOChatConfig")
except Exception as _e:
    print(f"[sensenova_u1_register] dynamic_module config register skip: {_e}")

print(
    "[sensenova_u1_register] Registered SenseNovaU1ForCausalLMAdapter and "
    f"SenseNovaU1ForTokenClassification (SENSENOVA_U1_SRC={SENSENOVA_U1_SRC})"
)


# Dense U1: FSDP wrap must not reference MoE-only layer class names.
SenseNovaU1ForCausalLMAdapter._no_split_modules = ["Qwen3DecoderLayer", "NEOVisionModel"]
SenseNovaU1ForTokenClassification._no_split_modules = ["Qwen3DecoderLayer", "NEOVisionModel"]


# trust_remote_code auto_map loads NEOChatModel from transformers_modules (a different
# class object than the package NEOChatModel). Promote adapter methods onto both.
_ADAPTER_METHODS = (
    "forward",
    "_build_indexes",
    "_build_block_attention_mask",
    "_prepare_single_gen_target",
    "_predict_x_from_hidden",
    "_detach_past_key_values",
    "_compute_fm_aux_split",
    "compute_pending_fm_aux_loss",
)


def _promote_adapter_onto(cls, label: str) -> None:
    if not isinstance(cls, type):
        return
    for _name in _ADAPTER_METHODS:
        setattr(cls, _name, getattr(SenseNovaU1ForCausalLMAdapter, _name))
    cls._no_split_modules = ["Qwen3DecoderLayer", "NEOVisionModel"]
    print(f"[sensenova_u1_register] Promoted adapter methods onto {label}")


_promote_adapter_onto(NEOChatModel, "package NEOChatModel")
for _modname, _mod in list(sys.modules.items()):
    if _mod is None or "neo_chat" not in _modname.lower():
        continue
    _cls = getattr(_mod, "NEOChatModel", None)
    if _cls is not None and getattr(_cls, "__name__", "") == "NEOChatModel":
        _promote_adapter_onto(_cls, _modname)
try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _model_dir = os.environ.get(
        "SENSENOVA_U1_MODEL_PATH",
        "/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT",
    )
    _Dyn = get_class_from_dynamic_module("modeling_neo_chat.NEOChatModel", _model_dir)
    _promote_adapter_onto(_Dyn, "dynamic_module NEOChatModel")
except Exception as _e:
    print(f"[sensenova_u1_register] dynamic_module promote skip: {_e}")
