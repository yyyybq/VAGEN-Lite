"""
vagen/models/cambrian_register.py

Registration module for Cambrian-S in VAGEN-Lite's FSDP training pipeline.

Imported by verl's `external_lib` mechanism before any model is loaded:
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register
    critic.model.external_lib=vagen.models.cambrian_register

What this file does:
  1. Adds the cambrian-s source tree to sys.path.
  2. Imports CambrianQwenForCausalLM, triggering AutoConfig / AutoModel registration.
  3. Defines CambrianForCausalLMAdapter – a GPU-compatible subclass with:
       a. _embed_multimodal_batch() – batched SigLIP encoding without XLA ops.
       b. forward() – bypasses XLA-only prepare_inputs_labels_for_multimodal.
       c. generate() – accepts pixel_values from verl multi_modal_inputs.
  4. Re-registers the adapter so AutoModelForCausalLM.from_pretrained() returns it.
  5. Defines CambrianForTokenClassification for the PPO critic head.

Token convention
----------------
The FSDP actor receives input_ids where each image is represented as
TOKENS_PER_IMAGE (756) consecutive IMAGE_TOKEN_INDEX (-200) tokens.
This expansion is performed in CambrianProcessorWrapper.__call__() and in
agent_loop_no_concat.py.

The IMAGE_TOKEN_INDEX=-200 sentinel is never a valid vocabulary ID;
input_ids.clamp(min=0) is used before any embedding lookup.
"""

import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast, TokenClassifierOutput

# ---------------------------------------------------------------------------
# 1.  Make cambrian-s importable
# ---------------------------------------------------------------------------
CAMBRIAN_SRC = os.environ.get(
    "CAMBRIAN_SRC",
    "/scratch/by2593/project/Active_Spatial/cambrian-s",
)
if CAMBRIAN_SRC not in sys.path:
    sys.path.insert(0, CAMBRIAN_SRC)

# Side-effects: registers CambrianQwenConfig with AutoConfig and
# CambrianQwenForCausalLM with AutoModelForCausalLM.
from cambrian.model.language_model.cambrian_qwen2 import (  # noqa: E402
    CambrianQwenConfig,
    CambrianQwenForCausalLM,
)

IMAGE_TOKEN_INDEX: int = -200       # pre-expanded sentinel in input_ids
TOKENS_PER_IMAGE: int = 756         # 27 * 28 (si_side_len * (si_side_len+1) with newlines)


# ---------------------------------------------------------------------------
# 2.  GPU-compatible adapter for FSDP actor / critic
# ---------------------------------------------------------------------------
class CambrianForCausalLMAdapter(CambrianQwenForCausalLM):
    """
    Wraps CambrianQwenForCausalLM for FSDP + verl RL training.

    Key fixes vs. base class
    -------------------------
    * _embed_multimodal_batch() replaces XLA-only prepare_inputs_labels_for_multimodal.
    * forward() bypasses the base-class dispatch to that XLA method.
    * generate() pre-computes inputs_embeds before calling Qwen2ForCausalLM.generate.
    * __init__() force-loads delay_load vision towers and converts to nn.ModuleList.
    * _tied_weights_keys = [] preserves lm_head != embed_tokens (tie_word_embeddings=False).
    """

    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)

        # Fix missing _attn_implementation on inner CambrianQwenModel
        attn_impl = getattr(config, "_attn_implementation", "eager")
        inner_model = getattr(self, "model", None)
        if inner_model is not None and not hasattr(inner_model, "_attn_implementation"):
            inner_model._attn_implementation = attn_impl

        # Force-load delay_load vision towers and convert list → nn.ModuleList
        if inner_model is not None:
            vt_list = getattr(inner_model, "vision_tower_aux_list", None)
            if vt_list is not None:
                for vt in vt_list:
                    if not getattr(vt, "is_loaded", True):
                        print(
                            f"[cambrian_register] Force-loading vision tower: "
                            f"{vt.vision_tower_name}"
                        )
                        vt.load_model()
                    vt.requires_grad_(True)
                if not isinstance(vt_list, nn.ModuleList):
                    inner_model.vision_tower_aux_list = nn.ModuleList(vt_list)
                    print("[cambrian_register] Converted vision_tower_aux_list → nn.ModuleList")

    # ------------------------------------------------------------------
    # Core image embedding
    # ------------------------------------------------------------------
    def _embed_multimodal_batch(
        self,
        input_ids: torch.Tensor,     # (bs, seq_len) with IMAGE_TOKEN_INDEX=-200 blocks
        pixel_values: torch.Tensor,  # (total_images, C, H, W)
    ) -> torch.Tensor:               # inputs_embeds (bs, seq_len, hidden_dim)
        """
        GPU-compatible image embedding.

        1. Encode all images through SigLIP (single batched call).
        2. Project through mm_projector.
        3. Append image_newline tokens (one per row of the 27×27 grid → 27×28=756 tokens).
        4. Embed text tokens (clamp -200 → 0 to avoid OOB).
        5. Scatter visual features into -200 positions in-place.
        """
        cfg = self.config
        si_token_len: int = cfg.si_token_len           # 729
        si_side_len: int = int(si_token_len ** 0.5)    # 27
        mm_use_newline: bool = getattr(cfg, "mm_use_im_newline_token", True)
        tokens_per_image: int = (
            si_side_len * (si_side_len + 1) if mm_use_newline else si_token_len
        )  # 756 with newlines

        batch_size = input_ids.shape[0]

        # 1. Encode -------------------------------------------------------
        image_aux_features_list = self.encode_images([pixel_values])
        image_features = image_aux_features_list[0]  # (N, si_token_len, vis_dim)

        # 2. Project -------------------------------------------------------
        proj_dtype = self.get_model().mm_projector[0].weight.dtype
        image_features = self.get_model().mm_projector(
            image_features.to(proj_dtype)
        ).to(pixel_values.dtype)  # (N, si_token_len, hidden_dim)

        # 2b. MIV features (NFP-specific) ----------------------------------
        # Cambrian-S projects images to 27×27 (SI), then also keeps a coarser
        # 8×8 (miv_side_len × miv_side_len) view of the same projected features.
        # These are injected at the first miv_token_len positions of each image
        # block, giving the LM a spatial overview alongside the full-resolution
        # SI tokens that follow.
        miv_features = None
        miv_token_len: int = getattr(cfg, 'miv_token_len', 0)  # 64
        if getattr(cfg, 'nfp_head', False) and miv_token_len > 0:
            miv_side_len = int(miv_token_len ** 0.5)  # 8
            total_n, _, _hidden = image_features.shape  # (N, 729, 3584)
            # Reshape to (N, 3584, 27, 27) for bilinear interpolation
            _feat_bchw = (
                image_features
                .view(total_n, si_side_len, si_side_len, _hidden)
                .permute(0, 3, 1, 2)
                .float()
            )
            _miv_bchw = F.interpolate(
                _feat_bchw,
                size=(miv_side_len, miv_side_len),
                mode='bilinear',
                align_corners=False,
            )
            miv_features = (
                _miv_bchw
                .permute(0, 2, 3, 1)
                .reshape(total_n, miv_token_len, _hidden)
                .to(image_features.dtype)
            )  # (N, 64, hidden_dim)

        # 3. Newline tokens ------------------------------------------------
        if mm_use_newline:
            total_imgs, _, hidden_dim = image_features.shape
            image_features = image_features.view(
                total_imgs, si_side_len, si_side_len, hidden_dim
            )
            newline = self.get_model().image_newline.to(image_features.dtype)
            newline_exp = newline.view(1, 1, 1, hidden_dim).expand(
                total_imgs, si_side_len, 1, hidden_dim
            )
            image_features = torch.cat([image_features, newline_exp], dim=2)
            image_features = image_features.view(total_imgs, -1, hidden_dim)
            # shape: (N, 756, hidden_dim)

        assert image_features.shape[1] == tokens_per_image, (
            f"image_features has {image_features.shape[1]} tokens/image but "
            f"input_ids was pre-expanded to {tokens_per_image} IMAGE_TOKEN_INDEX tokens."
        )

        # 4. Text embeddings (clamp -200 → 0) ------------------------------
        token_embeds = self.get_model().embed_tokens(
            input_ids.clamp(min=0)
        )  # (bs, seq_len, hidden_dim)

        # 5. Scatter -------------------------------------------------------
        img_global_idx = 0
        for b in range(batch_size):
            positions = (input_ids[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            n_img_tokens = positions.numel()
            if n_img_tokens % tokens_per_image != 0:
                raise ValueError(
                    f"Sample {b}: {n_img_tokens} IMAGE_TOKEN_INDEX tokens, "
                    f"not divisible by tokens_per_image={tokens_per_image}."
                )
            n_images = n_img_tokens // tokens_per_image
            for img_i in range(n_images):
                start = positions[img_i * tokens_per_image].item()
                # Write projected SI features (full 27×27+newlines = 756 tokens)
                token_embeds[b, start : start + tokens_per_image] = (
                    image_features[img_global_idx].to(token_embeds.dtype)
                )
                # MIV injection: overwrite first miv_token_len positions with the
                # coarser 8×8 projected overview (faithfully following Cambrian-S).
                if miv_features is not None and miv_token_len > 0:
                    token_embeds[b, start : start + miv_token_len] = (
                        miv_features[img_global_idx].to(token_embeds.dtype)
                    )
                img_global_idx += 1

        return token_embeds  # (bs, seq_len, hidden_dim)

    # ------------------------------------------------------------------
    # NFP auxiliary loss
    # ------------------------------------------------------------------
    def _compute_nfp_loss(
        self,
        hidden_states: torch.Tensor,     # (bs, seq_len, hidden_dim)
        nfp_pixel_values: torch.Tensor,  # (bs, C, H, W)  next-frame images
        nfp_loss_mask: torch.Tensor,     # (bs, seq_len) float, 1 at NFP positions
    ) -> torch.Tensor:
        """
        Compute the NFP auxiliary loss following the Cambrian-S design:
          1. Encode next-frame images through SigLIP (raw 1152-dim features, 729 patches).
          2. Interpolate to 8×8 = 64 tokens (miv_side_len × miv_side_len).
          3. Build nfp_tgt_embeds (bs, seq_len, 1152) with targets at NFP positions.
          4. nfp_head(hidden_states) → (bs, seq_len, 1152) predictions.
          5. Masked MSE + cosine loss weighted by nfp_mse_loss_weight / nfp_cosine_loss_weight.

        NFP positions are the first miv_token_len (=64) tokens of each image block,
        identified by nfp_loss_mask == 1.  Terminal turns set nfp_loss_mask = 0
        everywhere, so their (dummy) nfp_pixel_values contribute zero loss.
        """
        cfg = self.config
        miv_token_len: int = getattr(cfg, 'miv_token_len', 64)   # 64
        miv_side_len: int = int(miv_token_len ** 0.5)             # 8
        mm_hidden_size: int = 1152                                 # SigLIP output dim

        bs, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        hs_dtype = hidden_states.dtype

        # 1. Encode next-frame images through SigLIP (no mm_projector) --------
        nfp_raw = self.encode_images([nfp_pixel_values])[0]  # (bs, 729, 1152)
        feature_side_len = int(nfp_raw.shape[1] ** 0.5)     # 27

        # 2. Interpolate to miv_side_len × miv_side_len -----------------------
        nfp_bchw = (
            nfp_raw
            .view(bs, feature_side_len, feature_side_len, mm_hidden_size)
            .permute(0, 3, 1, 2)
            .float()
        )  # (bs, 1152, 27, 27)
        nfp_miv_bchw = F.interpolate(
            nfp_bchw,
            size=(miv_side_len, miv_side_len),
            mode='bilinear',
            align_corners=False,
        )
        nfp_target_grid = (
            nfp_miv_bchw
            .permute(0, 2, 3, 1)
            .reshape(bs, miv_token_len, mm_hidden_size)
            .to(hs_dtype)
        )  # (bs, 64, 1152)

        # 3. Build nfp_tgt_embeds (bs, seq_len, 1152) -------------------------
        nfp_tgt_embeds = torch.zeros(
            bs, seq_len, mm_hidden_size, device=device, dtype=hs_dtype
        )
        for b in range(bs):
            positions = (nfp_loss_mask[b] > 0.5).nonzero(as_tuple=True)[0]
            n_tokens = positions.numel()
            if n_tokens == 0:
                continue
            # Handle n_images * miv_token_len positions (RL: typically 1 image per turn)
            n_imgs = max(1, n_tokens // miv_token_len)
            for img_i in range(n_imgs):
                pos_slice = positions[img_i * miv_token_len : (img_i + 1) * miv_token_len]
                nfp_tgt_embeds[b, pos_slice, :] = nfp_target_grid[b]

        # 4. NFP predictions via nfp_head -------------------------------------
        nfp_outputs = self.model.nfp_head(hidden_states)  # (bs, seq_len, 1152)

        # 5. Masked MSE + cosine loss -----------------------------------------
        nfp_mse, nfp_cos = self.nfp_loss(nfp_outputs, nfp_tgt_embeds, nfp_loss_mask)
        nfp_mse_w = getattr(cfg, 'nfp_mse_loss_weight', 0.1)
        nfp_cos_w = getattr(cfg, 'nfp_cosine_loss_weight', 0.1)
        return nfp_mse_w * nfp_mse + nfp_cos_w * nfp_cos

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        nfp_pixel_values: Optional[torch.FloatTensor] = None,  # (bs, C, H, W) next-frame
        nfp_loss_mask: Optional[torch.FloatTensor] = None,     # (bs, seq_len) float
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if getattr(self.config, "connector_only", True) is False:
            raise NotImplementedError(
                "CambrianForCausalLMAdapter only supports connector_only=True."
            )

        # Build inputs_embeds
        if inputs_embeds is None:
            has_image_tokens = (
                input_ids is not None
                and (input_ids == IMAGE_TOKEN_INDEX).any()
            )
            if pixel_values is not None and has_image_tokens:
                inputs_embeds = self._embed_multimodal_batch(input_ids, pixel_values)
                input_ids = None
            elif input_ids is not None:
                inputs_embeds = self.get_model().embed_tokens(input_ids.clamp(min=0))
                input_ids = None

        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        lm_loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            lm_loss = CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        # NFP auxiliary loss (Cambrian-S Next Frame Prediction head) ----------
        # Computed only during training when nfp inputs are provided.
        nfp_aux_loss = None
        if (
            nfp_pixel_values is not None
            and nfp_loss_mask is not None
            and getattr(self.config, 'nfp_head', False)
            and self.training
        ):
            nfp_aux_loss = self._compute_nfp_loss(
                hidden_states,
                nfp_pixel_values,
                nfp_loss_mask.to(hidden_states.dtype),
            )

        # Combine: in RL training labels=None so lm_loss=None; nfp_aux_loss
        # is returned in the `loss` field for dp_actor to read and accumulate.
        combined_loss = lm_loss if lm_loss is not None else nfp_aux_loss

        if not return_dict:
            out = (logits,) + outputs[1:]
            return (combined_loss,) + out if combined_loss is not None else out

        return CausalLMOutputWithPast(
            loss=combined_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Pre-compute inputs_embeds then call Qwen2ForCausalLM.generate().
        verl's HF rollout calls: model.generate(input_ids=…, **multi_modal_inputs)
        where multi_modal_inputs = {"pixel_values": tensor}.
        """
        kwargs.pop("nfp_pixel_values", None)
        kwargs.pop("nfp_loss_masks", None)
        kwargs.pop("image_sizes", None)

        if pixel_values is not None and input_ids is not None:
            if (input_ids == IMAGE_TOKEN_INDEX).any():
                inputs_embeds = self._embed_multimodal_batch(input_ids, pixel_values)
            else:
                inputs_embeds = self.get_model().embed_tokens(input_ids.clamp(min=0))
        elif input_ids is not None:
            inputs_embeds = self.get_model().embed_tokens(input_ids.clamp(min=0))
        else:
            inputs_embeds = kwargs.pop("inputs_embeds", None)
            if inputs_embeds is None:
                raise ValueError("generate() requires either input_ids or inputs_embeds.")

        safe_input_ids = input_ids.clamp(min=0) if input_ids is not None else None

        from transformers import Qwen2ForCausalLM
        return Qwen2ForCausalLM.generate(
            self,
            input_ids=safe_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        kwargs.pop("pixel_values", None)
        from transformers import Qwen2ForCausalLM
        return Qwen2ForCausalLM.prepare_inputs_for_generation(
            self, input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 3.  Re-register adapter
# ---------------------------------------------------------------------------
AutoModelForCausalLM.register(CambrianQwenConfig, CambrianForCausalLMAdapter, exist_ok=True)


# ---------------------------------------------------------------------------
# 4.  Critic model
# ---------------------------------------------------------------------------
class CambrianForTokenClassification(CambrianQwenForCausalLM):
    """
    Cambrian-S with a token classification head for the PPO critic.
    Architecture: dropout → Linear(hidden_size, num_labels).
    """

    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = getattr(config, "num_labels", 1)

        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, "hidden_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = 0.0
        if isinstance(classifier_dropout, str):
            classifier_dropout = float(classifier_dropout)

        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, self.num_labels)

        if hasattr(self, "lm_head"):
            del self.lm_head

        attn_impl = getattr(config, "_attn_implementation", "eager")
        inner_model = getattr(self, "model", None)
        if inner_model is not None and not hasattr(inner_model, "_attn_implementation"):
            inner_model._attn_implementation = attn_impl

        if inner_model is not None:
            vt_list = getattr(inner_model, "vision_tower_aux_list", None)
            if vt_list is not None:
                for vt in vt_list:
                    if not getattr(vt, "is_loaded", True):
                        vt.load_model()
                    vt.requires_grad_(True)
                if not isinstance(vt_list, nn.ModuleList):
                    inner_model.vision_tower_aux_list = nn.ModuleList(vt_list)

        self.post_init()

    def get_input_embeddings(self):
        return self.get_model().embed_tokens

    def set_input_embeddings(self, value):
        self.get_model().embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            has_image_tokens = (
                input_ids is not None
                and (input_ids == IMAGE_TOKEN_INDEX).any()
            )
            if pixel_values is not None and has_image_tokens:
                inputs_embeds = CambrianForCausalLMAdapter._embed_multimodal_batch(
                    self, input_ids, pixel_values
                )
                input_ids = None
            elif input_ids is not None:
                inputs_embeds = self.get_model().embed_tokens(input_ids.clamp(min=0))
                input_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.dropout(outputs[0])
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss, MSELoss
            loss = (
                MSELoss()(logits.squeeze(), labels.squeeze())
                if self.num_labels == 1
                else CrossEntropyLoss()(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            )

        if not return_dict:
            out = (logits,) + outputs[2:]
            return ((loss,) + out) if loss is not None else out

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoModelForTokenClassification.register(
    CambrianQwenConfig, CambrianForTokenClassification, exist_ok=True
)

print(
    f"[cambrian_register] Registered CambrianForCausalLMAdapter and "
    f"CambrianForTokenClassification (CAMBRIAN_SRC={CAMBRIAN_SRC})"
)

# Register vLLM model for rollout inference (imported by vLLM rollout process via external_lib)
try:
    import vagen.models.cambrian_vllm  # noqa: F401 — registers CambrianVLLMForCausalLM
except Exception:
    pass
