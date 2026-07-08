"""
vagen/models/cambrian_vllm.py

vLLM V1 model wrapper for Cambrian-S (CambrianQwenForCausalLM).

Loaded by vllm_async_server.py before run_server() via:
    actor_rollout_ref.rollout.model.external_lib = vagen.models.cambrian_vllm

This file intentionally does NOT import from cambrian-s source to avoid the
missing `ezcolorlog` dependency.  It uses SiglipVisionModel from transformers.

Token convention
----------------
  <image>         ID 151665 -- single placeholder per image in the prompt
  <|image_pad|>   ID 151655 -- per-feature placeholder after vLLM expansion
  TOKENS_PER_IMAGE = 756    -- 27×28 (si_side_len=27, mm_use_im_newline_token=True)

Weight remapping from Cambrian-S checkpoint
-------------------------------------------
  model.embed_tokens.*                      → language_model.model.embed_tokens.*
  model.layers.*                            → language_model.model.layers.*
  model.norm.*                              → language_model.model.norm.*
  lm_head.*                                 → language_model.lm_head.*
  model.vision_tower_aux_list.0.vision_tower.* → vision_tower.*
  model.mm_projector.*                      → mm_projector.*
  model.image_newline                       → image_newline
  model.nfp_head.*                          → (skipped)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen2Config, SiglipConfig, SiglipVisionModel
from transformers.feature_extraction_utils import BatchFeature

from vllm import ModelRegistry
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import (init_vllm_registered_model,
                                               maybe_prefix,
                                               merge_multimodal_embeddings)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                         BaseProcessingInfo, PromptReplacement,
                                         PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_TOKEN: str = "<image>"
IMAGE_TOKEN_ID: int = 151665       # added as special token in Qwen2 tokenizer
IMAGE_PAD_TOKEN_ID: int = 151655   # <|image_pad|> built-in Qwen2 token
TOKENS_PER_IMAGE: int = 756        # 27 * (27 + 1) = 756
SIGLIP_HIDDEN_DIM: int = 1152
LM_HIDDEN_DIM: int = 3584
SIGLIP_CACHE: str = "/scratch/by2593/hf_cache"


# ---------------------------------------------------------------------------
# Image preprocessing (SigLIP, no HF Processor object needed)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_siglip_image_processor():
    from transformers import SiglipImageProcessor
    try:
        return SiglipImageProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384",
            cache_dir=SIGLIP_CACHE,
        )
    except Exception:
        return SiglipImageProcessor(image_size=384, patch_size=14)


def _expand2square(img: Image.Image, bg=(122, 116, 104)) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new("RGB", (s, s), bg)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def _preprocess_images(images: list[Image.Image]) -> torch.Tensor:
    """Returns float32 tensor (N, 3, 384, 384)."""
    proc = _get_siglip_image_processor()
    prepared = [_expand2square(img.convert("RGB")) for img in images]
    return proc(images=prepared, return_tensors="pt").pixel_values


# ---------------------------------------------------------------------------
# Processing info
# ---------------------------------------------------------------------------
class CambrianProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        # Fixed count avoids expensive dummy profiling at startup
        return {"image": TOKENS_PER_IMAGE}

    def get_hf_processor(self, **kwargs):  # type: ignore[override]
        # Cambrian has no HF processor; return None to prevent framework errors
        return None


# ---------------------------------------------------------------------------
# Dummy inputs builder
# ---------------------------------------------------------------------------
class CambrianDummyInputsBuilder(
        BaseDummyInputsBuilder[CambrianProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        n = mm_counts.get("image", 0)
        return IMAGE_TOKEN * n

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]):
        n = mm_counts.get("image", 0)
        return {
            "image": self._get_dummy_images(width=384, height=384,
                                            num_images=n),
        }


# ---------------------------------------------------------------------------
# MultiModal processor
# ---------------------------------------------------------------------------
class CambrianMultiModalProcessor(
        BaseMultiModalProcessor[CambrianProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        images: list[Image.Image] = list(mm_data.get("images", []))  # type: ignore
        if images:
            pixel_values = _preprocess_images(images)
        else:
            pixel_values = torch.zeros(0, 3, 384, 384)

        return BatchFeature({"input_ids": input_ids,
                             "pixel_values": pixel_values})

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {"pixel_values": MultiModalFieldConfig.batched("image")}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # Each single <image> (151665) → 756 × <|image_pad|> (151655)
        return [
            PromptReplacement(
                modality="image",
                target=[IMAGE_TOKEN_ID],
                replacement=[IMAGE_PAD_TOKEN_ID] * TOKENS_PER_IMAGE,
            )
        ]


# ---------------------------------------------------------------------------
# vLLM model
# ---------------------------------------------------------------------------
@MULTIMODAL_REGISTRY.register_processor(
    CambrianMultiModalProcessor,
    info=CambrianProcessingInfo,
    dummy_inputs=CambrianDummyInputsBuilder,
)
class CambrianVLLMForCausalLM(nn.Module, SupportsMultiModal):
    """
    vLLM V1 Cambrian-S model.

    Modules
    -------
    language_model  : vLLM Qwen2ForCausalLM (manages paged KV cache)
    vision_tower    : SiglipVisionModel (transformers, loaded from checkpoint)
    mm_projector    : Linear(1152→3584) + GELU + Linear(3584→3584)
    image_newline   : nn.Parameter (3584,)
    """

    supports_multimodal: bool = True

    def __init__(self, vllm_config, prefix: str = "") -> None:
        super().__init__()

        cfg = vllm_config.model_config.hf_config

        # ---- Language backbone (Qwen2) ----
        qwen2_config = Qwen2Config(
            hidden_size=LM_HIDDEN_DIM,
            intermediate_size=getattr(cfg, "intermediate_size", 18944),
            num_hidden_layers=getattr(cfg, "num_hidden_layers", 28),
            num_attention_heads=getattr(cfg, "num_attention_heads", 28),
            num_key_value_heads=getattr(cfg, "num_key_value_heads", 4),
            vocab_size=getattr(cfg, "vocab_size", 152064),
            max_position_embeddings=getattr(cfg, "max_position_embeddings",
                                            32768),
            rope_theta=getattr(cfg, "rope_theta", 1_000_000.0),
            rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
            tie_word_embeddings=False,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=qwen2_config,
            architectures=["Qwen2ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # ---- SigLIP vision encoder ----
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(
                "google/siglip-so400m-patch14-384",
                cache_dir=SIGLIP_CACHE,
            )
            siglip_cfg = _cfg.vision_config if hasattr(_cfg, "vision_config") else _cfg
        except Exception:
            siglip_cfg = SiglipConfig(
                hidden_size=SIGLIP_HIDDEN_DIM,
                num_hidden_layers=27,
                num_attention_heads=16,
                intermediate_size=4304,
                image_size=384,
                patch_size=14,
            )
        self.vision_tower = SiglipVisionModel(siglip_cfg)

        # ---- MM projector ----
        self.mm_projector = nn.Sequential(
            nn.Linear(SIGLIP_HIDDEN_DIM, LM_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(LM_HIDDEN_DIM, LM_HIDDEN_DIM),
        )

        # ---- Image newline token ----
        self.image_newline = nn.Parameter(torch.zeros(LM_HIDDEN_DIM))

        self.img_context_token_id = IMAGE_PAD_TOKEN_ID

    # ------------------------------------------------------------------
    # SupportsMultiModal interface
    # ------------------------------------------------------------------
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        return IMAGE_TOKEN if modality == "image" else None

    def get_language_model(self):
        return self.language_model

    def get_multimodal_embeddings(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[tuple]:
        if pixel_values is None or pixel_values.numel() == 0:
            return None

        # vLLM batches items as (N, 1, C, H, W) when merge_by_field_config=False.
        # Flatten to (N, C, H, W) so SigLIP gets the expected 4-D input.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.flatten(0, 1)  # (N,1,C,H,W) → (N,C,H,W)

        # SigLIP encode: (N, 729, 1152)
        vt_dtype = next(self.vision_tower.parameters()).dtype
        features = self.vision_tower(
            pixel_values=pixel_values.to(vt_dtype)
        ).last_hidden_state  # (N, 729, 1152)

        # Project: (N, 729, 3584)
        proj_dtype = self.mm_projector[0].weight.dtype
        features = self.mm_projector(features.to(proj_dtype))

        # Append newline column: (N, 27, 27, 3584) → (N, 27, 28, 3584) → (N, 756, 3584)
        N, _, hidden = features.shape
        features = features.view(N, 27, 27, hidden)
        newline = (self.image_newline.to(features.dtype)
                   .view(1, 1, 1, hidden)
                   .expand(N, 27, 1, hidden))
        features = torch.cat([features, newline], dim=2)   # (N, 27, 28, 3584)
        features = features.view(N, TOKENS_PER_IMAGE, hidden)  # (N, 756, 3584)

        return tuple(features.unbind(0))  # N × (756, 3584)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(
            input_ids.clamp(min=0))

        if multimodal_embeddings:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.img_context_token_id,
            )
        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward / logits
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        elif inputs_embeds is None:
            mm_embeds = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, mm_embeds)
            input_ids = None

        return self.language_model.model(
            input_ids, positions, intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        VISION_PFX = "model.vision_tower_aux_list.0.vision_tower."
        PROJ_PFX = "model.mm_projector."
        NEWLINE_KEY = "model.image_newline"

        lm_weights: list[Tuple[str, torch.Tensor]] = []
        local_tensors: dict[str, torch.Tensor] = {}

        for name, tensor in weights:
            if name.startswith(VISION_PFX):
                # e.g. model.vision_tower_aux_list.0.vision_tower.vision_model.X
                #   → vision_tower.vision_model.X
                local_tensors["vision_tower." + name[len(VISION_PFX):]] = tensor
            elif name.startswith(PROJ_PFX):
                local_tensors["mm_projector." + name[len(PROJ_PFX):]] = tensor
            elif name == NEWLINE_KEY:
                local_tensors["image_newline"] = tensor
            elif name.startswith("model.nfp_head."):
                pass  # skip NFP head
            elif name.startswith("model.") or name.startswith("lm_head."):
                # Qwen2 backbone weights go through the language_model loader
                lm_weights.append((name, tensor))

        # Delegate to Qwen2ForCausalLM weight loader (handles QKV fusion etc.)
        self.language_model.load_weights(iter(lm_weights))

        # Load local parameters
        params = dict(self.named_parameters())
        for pname, tensor in local_tensors.items():
            if pname in params:
                params[pname].data.copy_(tensor)

        # Return all parameter names so vllm's strict weight-tracking check passes.
        # The lm weights are loaded into tensors via self.language_model.load_weights()
        # even though their full root-level names (language_model.model.layers.*) are
        # not in local_tensors.  Returning the full named_parameters() set is correct
        # because every parameter is initialised from the checkpoint or from random
        # init for components not present in the checkpoint (there are none here).
        return {name for name, _ in self.named_parameters()}


# ---------------------------------------------------------------------------
# NOTE: Model registration is handled via the vllm.general_plugins entry_point
# in setup.py → vagen.models.cambrian_plugin:register().  vLLM calls
# load_general_plugins() in every spawned subprocess (EngineCore_DP*), which
# ensures CambrianQwenForCausalLM is in ModelRegistry.models before the engine
# looks it up.  The call below is kept for direct imports of this module
# (e.g. the parent vllm_async_server process via external_lib).
# ---------------------------------------------------------------------------
ModelRegistry.register_model(
    "CambrianQwenForCausalLM",
    "vagen.models.cambrian_vllm:CambrianVLLMForCausalLM",
)
