"""
vLLM V1 wrapper for SenseNova-U1 (NEOChatModel).

Phase-1 rollout (understanding / text decode):
  - NEOVision embeddings for image tokens
  - vLLM Qwen3ForCausalLM backbone with MoT-und attention patches:
      * split q/k RMSNorm (t half + hw half)
      * 3D RoPE: t uses rope_theta, h/w use rope_theta_hw
      * thw indexes matching NEOChatModel.get_thw_indexes
"""

from __future__ import annotations

import os
import sys
import types
from functools import lru_cache
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, Qwen3Config
from transformers.feature_extraction_utils import BatchFeature as HFBatchFeature

from vllm import ModelRegistry
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsMRoPE
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

SENSENOVA_U1_SRC = os.environ.get(
    "SENSENOVA_U1_SRC", "/mnt/umm/users/yinbaiqiao/SenseNova-U1/src"
)
if SENSENOVA_U1_SRC not in sys.path:
    sys.path.insert(0, SENSENOVA_U1_SRC)

from sensenova_u1.models.neo_unify.modeling_neo_vit import (  # noqa: E402
    NEOVisionModel,
    build_abs_positions_from_grid_hw,
)
from sensenova_u1.models.neo_unify.utils import load_image_native  # noqa: E402

# Special tokens (present in U1 tokenizer vocab)
IMAGE_TOKEN = "<image>"
IMG_START_TOKEN_ID = 151670
IMG_END_TOKEN_ID = 151671
IMG_CONTEXT_TOKEN_ID = 151669

# Active-Spatial smoke renders 512x512 → grid 32x32 → 256 context tokens
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 0.5
TOKENS_PER_IMAGE = 256  # grid_h * grid_w * downsample^2
SEQ_TOKENS_PER_IMAGE = TOKENS_PER_IMAGE + 2  # <img> + context*N + </img>
MIN_PIXELS = 512 * 512
MAX_PIXELS = 512 * 512
TOKEN_GRID_HW = (16, 16)  # after downsample


@lru_cache(maxsize=1)
def _vision_config_from_env():
    from transformers import AutoConfig

    path = os.environ.get(
        "SENSENOVA_U1_MODEL_PATH",
        "/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT",
    )
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return cfg.vision_config


def _preprocess_images(images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pixel_values (N, P, C*ps*ps) and grid_hw (N, 2) for fixed 512^2."""
    pvs = []
    ghws = []
    for image in images:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        # Force 512x512 for stable token count in smoke.
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.BICUBIC)
        pv, ghw = load_image_native(
            image,
            patch_size=PATCH_SIZE,
            downsample_ratio=DOWNSAMPLE_RATIO,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            upscale=False,
        )
        pvs.append(pv)
        ghws.append(ghw)
    if not pvs:
        return (
            torch.zeros(0, 3 * PATCH_SIZE * PATCH_SIZE),
            torch.zeros(0, 2, dtype=torch.long),
        )
    return torch.stack(pvs, dim=0), torch.cat(ghws, dim=0).long()


def compute_u1_thw_indexes(
    input_ids: torch.Tensor,
    grid_hw_tokens: Optional[torch.Tensor] = None,
    img_context_token_id: int = IMG_CONTEXT_TOKEN_ID,
    img_start_token_id: int = IMG_START_TOKEN_ID,
) -> torch.Tensor:
    """Match NEOChatModel.get_thw_indexes (1D input_ids → [3, L])."""
    assert input_ids.dim() == 1
    device = input_ids.device
    img_start_shift = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            (input_ids == img_start_token_id).long(),
        ],
        dim=0,
    )[:-1]
    not_img_token = (input_ids != img_context_token_id).long()
    t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
    h_indexes = torch.zeros_like(t_indexes)
    w_indexes = torch.zeros_like(t_indexes)

    if grid_hw_tokens is not None and grid_hw_tokens.numel() > 0:
        selected = input_ids == img_context_token_id
        if int(selected.sum().item()) > 0:
            abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                grid_hw_tokens.to(device=device, dtype=torch.long),
                device=device,
            )
            h_indexes[selected] = abs_pos_h.to(dtype=t_indexes.dtype)
            w_indexes[selected] = abs_pos_w.to(dtype=t_indexes.dtype)
    return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)


class U1MotRotaryEmbedding(nn.Module):
    """U1 MoT-und RoPE: t(half, θ) + h(quarter, θ_hw) + w(quarter, θ_hw)."""

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        rope_theta: float,
        rope_theta_hw: float,
        max_position_hw: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self.t_dim = head_dim // 2
        self.hw_dim = head_dim // 4
        self.rope_t = get_rope(
            self.t_dim,
            rotary_dim=self.t_dim,
            max_position=max_position,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        self.rope_h = get_rope(
            self.hw_dim,
            rotary_dim=self.hw_dim,
            max_position=max_position_hw,
            base=rope_theta_hw,
            is_neox_style=True,
            dtype=dtype,
        )
        self.rope_w = get_rope(
            self.hw_dim,
            rotary_dim=self.hw_dim,
            max_position=max_position_hw,
            base=rope_theta_hw,
            is_neox_style=True,
            dtype=dtype,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert key is not None
        if positions.ndim == 1:
            pos_t = positions
            pos_h = torch.zeros_like(positions)
            pos_w = torch.zeros_like(positions)
        else:
            assert positions.ndim == 2 and positions.size(0) == 3
            pos_t, pos_h, pos_w = positions[0], positions[1], positions[2]
            # Decode path (stock M-RoPE next-pos) sets t=h=w. U1 text wants h=w=0.
            text_like = (pos_h == pos_t) & (pos_w == pos_t)
            if bool(text_like.any()):
                pos_h = torch.where(text_like, torch.zeros_like(pos_h), pos_h)
                pos_w = torch.where(text_like, torch.zeros_like(pos_w), pos_w)

        q_shape, k_shape = query.shape, key.shape
        q = query.view(-1, q_shape[-1] // self.head_dim, self.head_dim)
        k = key.view(-1, k_shape[-1] // self.head_dim, self.head_dim)
        n_tok = q.shape[0]

        q_t, q_h, q_w = (
            q[..., : self.t_dim],
            q[..., self.t_dim : self.t_dim + self.hw_dim],
            q[..., self.t_dim + self.hw_dim :],
        )
        k_t, k_h, k_w = (
            k[..., : self.t_dim],
            k[..., self.t_dim : self.t_dim + self.hw_dim],
            k[..., self.t_dim + self.hw_dim :],
        )

        q_t, k_t = self.rope_t(pos_t, q_t.reshape(n_tok, -1), k_t.reshape(n_tok, -1))
        q_h, k_h = self.rope_h(pos_h, q_h.reshape(n_tok, -1), k_h.reshape(n_tok, -1))
        q_w, k_w = self.rope_w(pos_w, q_w.reshape(n_tok, -1), k_w.reshape(n_tok, -1))

        q_t = q_t.view(n_tok, -1, self.t_dim)
        q_h = q_h.view(n_tok, -1, self.hw_dim)
        q_w = q_w.view(n_tok, -1, self.hw_dim)
        k_t = k_t.view(n_tok, -1, self.t_dim)
        k_h = k_h.view(n_tok, -1, self.hw_dim)
        k_w = k_w.view(n_tok, -1, self.hw_dim)

        query = torch.cat([q_t, q_h, q_w], dim=-1).reshape(q_shape)
        key = torch.cat([k_t, k_h, k_w], dim=-1).reshape(k_shape)
        return query, key


def _mot_und_attn_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)

    q_t, q_hw = q_by_head[..., : self.head_dim // 2], q_by_head[..., self.head_dim // 2 :]
    k_t, k_hw = k_by_head[..., : self.head_dim // 2], k_by_head[..., self.head_dim // 2 :]
    q_t = self.q_norm_t(q_t)
    q_hw = self.q_norm_hw(q_hw)
    k_t = self.k_norm_t(k_t)
    k_hw = self.k_norm_hw(k_hw)
    q = torch.cat([q_t, q_hw], dim=-1).view(q.shape)
    k = torch.cat([k_t, k_hw], dim=-1).view(k.shape)

    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def install_u1_mot_und_attention(
    language_model: nn.Module,
    *,
    head_dim: int,
    rope_theta: float,
    rope_theta_hw: float,
    max_position: int,
    max_position_hw: int,
    rms_norm_eps: float,
) -> int:
    """Replace vanilla Qwen3 q/k-norm + 1D RoPE with MoT-und split norm + 3D RoPE."""
    model = getattr(language_model, "model", None)
    if model is None or not hasattr(model, "layers"):
        print("[sensenova_u1_vllm] install MoT: no layers found", flush=True)
        return 0

    dtype = torch.get_default_dtype()
    n = 0
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        # Half-dim norms (overwrite full-head norms used by stock Qwen3).
        attn.q_norm_t = RMSNorm(head_dim // 2, eps=rms_norm_eps)
        attn.q_norm_hw = RMSNorm(head_dim // 2, eps=rms_norm_eps)
        attn.k_norm_t = RMSNorm(head_dim // 2, eps=rms_norm_eps)
        attn.k_norm_hw = RMSNorm(head_dim // 2, eps=rms_norm_eps)
        attn.rotary_emb = U1MotRotaryEmbedding(
            head_dim=head_dim,
            max_position=max_position,
            rope_theta=rope_theta,
            rope_theta_hw=rope_theta_hw,
            max_position_hw=max_position_hw,
            dtype=dtype,
        )
        attn.forward = types.MethodType(_mot_und_attn_forward, attn)
        n += 1
    print(
        f"[sensenova_u1_vllm] Installed MoT-und attention on {n} layers "
        f"(head_dim={head_dim}, θ={rope_theta}, θ_hw={rope_theta_hw})",
        flush=True,
    )
    return n


class U1ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        return {"image": SEQ_TOKENS_PER_IMAGE}

    def get_hf_processor(self, **kwargs):  # type: ignore[override]
        return None


class U1DummyInputsBuilder(BaseDummyInputsBuilder[U1ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]):
        n = mm_counts.get("image", 0)
        return {
            "image": self._get_dummy_images(width=512, height=512, num_images=n),
        }


class U1MultiModalProcessor(BaseMultiModalProcessor[U1ProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> HFBatchFeature:
        tokenizer = self.info.get_tokenizer()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        images: list[Image.Image] = list(mm_data.get("images", []) or [])  # type: ignore
        if images:
            pixel_values, grid_hw = _preprocess_images(images)
            # Token-grid thw for vLLM M-RoPE init (t=1, h, w after downsample).
            image_grid_thw = torch.stack(
                [
                    torch.tensor(
                        [1, int(g[0].item() * DOWNSAMPLE_RATIO), int(g[1].item() * DOWNSAMPLE_RATIO)],
                        dtype=torch.long,
                    )
                    for g in grid_hw
                ],
                dim=0,
            )
        else:
            pixel_values = torch.zeros(0, 3 * PATCH_SIZE * PATCH_SIZE)
            grid_hw = torch.zeros(0, 2, dtype=torch.long)
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        return BatchFeature(
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "grid_hw": grid_hw,
                "image_grid_thw": image_grid_thw,
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: HFBatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "grid_hw": MultiModalFieldConfig.batched("image"),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        repl = (
            [IMG_START_TOKEN_ID]
            + [IMG_CONTEXT_TOKEN_ID] * TOKENS_PER_IMAGE
            + [IMG_END_TOKEN_ID]
        )
        return [
            PromptReplacement(
                modality="image",
                target=IMAGE_TOKEN,
                replacement=repl,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    U1MultiModalProcessor,
    info=U1ProcessingInfo,
    dummy_inputs=U1DummyInputsBuilder,
)
class SenseNovaU1VLLMForCausalLM(nn.Module, SupportsMultiModal, SupportsMRoPE):
    """vLLM wrapper: Qwen3 LM + NEOVision + MoT-und attention/RoPE."""

    supports_multimodal: bool = True
    supports_mrope: bool = True

    def __init__(self, vllm_config, prefix: str = "") -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        llm_cfg = getattr(cfg, "llm_config", cfg)

        # Ensure runner uses 3D M-RoPE positions (uses_mrope reads hf_config.rope_scaling).
        if getattr(cfg, "rope_scaling", None) is None:
            head_dim = int(getattr(llm_cfg, "head_dim", 128))
            cfg.rope_scaling = {
                "rope_type": "default",
                "mrope_section": [head_dim // 4, head_dim // 8, head_dim // 8],
            }

        head_dim = int(getattr(llm_cfg, "head_dim", 128))
        rope_theta = float(getattr(llm_cfg, "rope_theta", 5_000_000.0))
        rope_theta_hw = float(getattr(llm_cfg, "rope_theta_hw", 10_000.0))
        max_position = int(getattr(llm_cfg, "max_position_embeddings", 262144))
        max_position_hw = int(getattr(llm_cfg, "max_position_embeddings_hw", 10000))
        rms_eps = float(getattr(llm_cfg, "rms_norm_eps", 1e-6))

        qwen3_config = Qwen3Config(
            hidden_size=getattr(llm_cfg, "hidden_size", 4096),
            intermediate_size=getattr(llm_cfg, "intermediate_size", 12288),
            num_hidden_layers=getattr(llm_cfg, "num_hidden_layers", 42),
            num_attention_heads=getattr(llm_cfg, "num_attention_heads", 32),
            num_key_value_heads=getattr(llm_cfg, "num_key_value_heads", 8),
            vocab_size=getattr(llm_cfg, "vocab_size", 151936),
            max_position_embeddings=max_position,
            rope_theta=rope_theta,
            rms_norm_eps=rms_eps,
            head_dim=head_dim,
            tie_word_embeddings=False,
            rope_scaling=None,  # MoT rotary installed below; not stock MRotaryEmbedding
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=qwen3_config,
            architectures=["Qwen3ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )

        install_u1_mot_und_attention(
            self.language_model,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_theta_hw=rope_theta_hw,
            max_position=max_position,
            max_position_hw=max_position_hw,
            rms_norm_eps=rms_eps,
        )

        vision_cfg = getattr(cfg, "vision_config", None) or _vision_config_from_env()
        self.vision_model = NEOVisionModel(vision_cfg)
        self.img_context_token_id = IMG_CONTEXT_TOKEN_ID
        self.img_start_token_id = IMG_START_TOKEN_ID
        self.hidden_size = qwen3_config.hidden_size
        self.head_dim = head_dim
        self.downsample_ratio = DOWNSAMPLE_RATIO

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        return IMAGE_TOKEN if modality == "image" else None

    def get_language_model(self):
        return self.language_model

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Optional[Union[list[list[int]], torch.Tensor]] = None,
        video_grid_thw: Optional[Union[list[list[int]], torch.Tensor]] = None,
        second_per_grid_ts: Optional[list[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Build U1 thw indexes for the prompt (matches get_thw_indexes)."""
        input_ids = torch.tensor(input_tokens, dtype=torch.long)

        grids: List[List[int]] = []
        if image_grid_thw is not None:
            if torch.is_tensor(image_grid_thw):
                for row in image_grid_thw.tolist():
                    if isinstance(row[0], list):
                        for r in row:
                            grids.append([int(r[1]), int(r[2])])
                    else:
                        grids.append([int(row[1]), int(row[2])])
            else:
                for row in image_grid_thw:
                    grids.append([int(row[1]), int(row[2])])

        n_img_ctx = int((input_ids == self.img_context_token_id).sum().item())
        if n_img_ctx > 0 and not grids:
            # Fallback: fixed smoke grid 16x16 per image.
            n_img = max(1, n_img_ctx // TOKENS_PER_IMAGE)
            grids = [list(TOKEN_GRID_HW) for _ in range(n_img)]

        grid_hw_tokens = (
            torch.tensor(grids, dtype=torch.long) if grids else None
        )
        indexes = compute_u1_thw_indexes(
            input_ids,
            grid_hw_tokens=grid_hw_tokens,
            img_context_token_id=self.img_context_token_id,
            img_start_token_id=self.img_start_token_id,
        )
        if seq_len is None:
            seq_len = len(input_tokens)
        indexes = indexes[:, context_len:seq_len]
        mrope_position_delta = int(indexes[0].max().item() + 1 - len(input_tokens))
        return indexes, mrope_position_delta

    def get_multimodal_embeddings(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        grid_hw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if pixel_values is None or (
            torch.is_tensor(pixel_values) and pixel_values.numel() == 0
        ):
            return None

        if pixel_values.dim() == 4:
            pixel_values = pixel_values.squeeze(1)
        if grid_hw is not None and grid_hw.dim() == 3:
            grid_hw = grid_hw.squeeze(1)

        device = next(self.vision_model.parameters()).device
        dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        if grid_hw is None:
            n = pixel_values.shape[0]
            grid_hw = torch.tensor(
                [[32, 32]] * n, device=device, dtype=torch.long
            )
        else:
            grid_hw = grid_hw.to(device=device, dtype=torch.long)

        n = pixel_values.shape[0]
        flat_pv = pixel_values.reshape(-1, pixel_values.shape[-1])
        vit = self.vision_model(
            pixel_values=flat_pv, grid_hw=grid_hw, return_dict=True
        ).last_hidden_state

        outs = []
        offset = 0
        for i in range(n):
            h, w = int(grid_hw[i, 0]), int(grid_hw[i, 1])
            ntok = int(h * w * (DOWNSAMPLE_RATIO**2))
            outs.append(vit[offset : offset + ntok])
            offset += ntok
        return tuple(outs)

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        inputs_embeds = self.language_model.get_input_embeddings(
            input_ids.clamp(min=0)
        )
        if multimodal_embeddings:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.img_context_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
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
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map U1 MoT und checkpoint → vLLM Qwen3 + NEOVision + split norms."""
        vision_buf: dict[str, torch.Tensor] = {}
        n_skip = 0
        n_lm = 0

        def _map_name(name: str) -> Optional[str]:
            n = name
            if n.startswith("model.language_model."):
                n = n[len("model.") :]
            if n.startswith("language_model."):
                n = n[len("language_model.") :]
            if n.startswith("model.") or n.startswith("lm_head."):
                return n
            if n.startswith("layers.") or n.startswith("embed_tokens") or n.startswith("norm."):
                return "model." + n
            if n == "lm_head.weight" or n.startswith("lm_head."):
                return n
            return None

        def _iter_lm():
            nonlocal n_skip, n_lm
            for name, tensor in weights:
                if "mot_gen" in name or name.startswith("fm_modules."):
                    n_skip += 1
                    del tensor
                    continue
                if name.startswith("vision_model."):
                    local = name[len("vision_model.") :]
                    vision_buf[local] = tensor.detach().to("cpu", copy=True)
                    del tensor
                    continue

                n = _map_name(name)
                if n is None:
                    n_skip += 1
                    del tensor
                    continue

                # Split half-norms → q_norm_t / q_norm_hw (do NOT concatenate).
                for kind in ("q_norm", "k_norm"):
                    # Exact suffix match: "...self_attn.q_norm.weight" → "...q_norm_t.weight"
                    if n.endswith(f".{kind}_hw.weight"):
                        # already correct name style for our modules
                        break
                    if n.endswith(f".{kind}.weight"):
                        n = n[: -len(f"{kind}.weight")] + f"{kind}_t.weight"
                        break
                n_lm += 1
                yield (n, tensor)

        try:
            self.language_model.load_weights(_iter_lm())
        except Exception as e:
            print(f"[sensenova_u1_vllm] language_model.load_weights warn: {e}", flush=True)

        if vision_buf:
            missing, unexpected = self.vision_model.load_state_dict(
                vision_buf, strict=False
            )
            print(
                f"[sensenova_u1_vllm] vision load missing={len(missing)} "
                f"unexpected={len(unexpected)} skip={n_skip} lm={n_lm}",
                flush=True,
            )
        else:
            print(
                f"[sensenova_u1_vllm] load_weights done skip={n_skip} lm={n_lm}",
                flush=True,
            )

        return {name for name, _ in self.named_parameters()}


# Direct-import registration (parent process / external_lib).
ModelRegistry.register_model(
    "NEOChatModel",
    "vagen.models.sensenova_u1_vllm:SenseNovaU1VLLMForCausalLM",
)
ModelRegistry.register_model(
    "SenseNovaU1ForCausalLMAdapter",
    "vagen.models.sensenova_u1_vllm:SenseNovaU1VLLMForCausalLM",
)
