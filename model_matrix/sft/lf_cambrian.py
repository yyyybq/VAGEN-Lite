"""Cambrian-S in the bundled LLaMA-Factory (ca75f1e, which has no Cambrian support): the model line, step 2 of
2026-10-24 -- registered from this file, nothing installed (as model_matrix/sft/lf_qwen3vl.py for Qwen3-VL).

- Template "cambrian_qwen": the Qwen2.5 chat layout (<|im_start|>role\\n...<|im_end|>\\n, the model's own chat
  template; the conversations always carry their system prompt) with the CambrianPlugin below.
- CambrianPlugin: each "<image>" of a message becomes 756 <|image_pad|> tokens (151655; SigLIP2's 27 x 27 patches
  and a newline per row) and the frames go through Cambrian-S's own preprocessing, the RL path's function
  (vagen.models.cambrian_processor.preprocess_siglip_images: padded to a square, 384 x 384 bicubic, SigLIP2's
  normalisation) -- the expanded sequence the RL trainer sees (vagen/agent_loop/cambrian_concat.py).
- CambrianSFTProcessor, registered with AutoProcessor for Cambrian's config: what LLaMA-Factory asks of a processor
  (the tokenizer and an image processor: SigLIP2's, from the line's cache); it saves nothing of its own, so the
  checkpoints keep no preprocessor file that verl or vLLM would take for the model's processor.
- The model class: the repository's adapter (vagen/models/cambrian_register.py) under the same class name (the saved
  config names an architecture the vLLM plugin knows), with image_token_alias = 151655 (the adapter's image
  positions) and Transformers' causal-LM loss over the accumulated batch (num_items_in_batch), as Qwen2.5-VL's own
  forward computes it under the Trainer; the adapter alone averages per call.
- Composite model "cambrian_qwen" (what freeze_vision_tower freezes): the SigLIP2 tower
  (model.vision_tower_aux_list); the projector (model.mm_projector) and the row newline train, as the round-11
  recipe trains Qwen2.5-VL's merger.

``register()`` is idempotent and must run in every process before LLaMA-Factory reads the recipe
(model_matrix.sft.lf_launcher does it).
"""
from __future__ import annotations

TEMPLATE = "cambrian_qwen"
IMAGE_PAD = "<|image_pad|>"
IMAGE_PAD_ID = 151655
TOKENS_PER_IMAGE = 756


def plugin_class():
    from dataclasses import dataclass

    from copy import deepcopy

    from llamafactory.data.mm_plugin import BasePlugin
    from llamafactory.extras.constants import IMAGE_PLACEHOLDER

    @dataclass
    class CambrianPlugin(BasePlugin):
        def process_messages(self, messages, images, videos, audios, processor):
            self._validate_input(processor, images, videos, audios)
            self._validate_messages(messages, images, videos, audios)
            messages = deepcopy(messages)
            for message in messages:
                message["content"] = message["content"].replace(IMAGE_PLACEHOLDER, self.image_token * TOKENS_PER_IMAGE)
            return messages

        def get_mm_inputs(self, images, videos, audios, imglens, vidlens, audlens, batch_ids, processor):
            self._validate_input(processor, images, videos, audios)
            if not images:
                return {}
            from vagen.models.cambrian_processor import preprocess_siglip_images

            frames = self._regularize_images(images, image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                                             image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32))["images"]
            return {"pixel_values": preprocess_siglip_images(frames, processor.image_processor)}

    return CambrianPlugin


class CambrianSFTProcessor:
    """The tokenizer and SigLIP2's image processor (LLaMA-Factory's processor checks); saves nothing."""

    def __init__(self, tokenizer, image_processor) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        import os

        from transformers import AutoImageProcessor, AutoTokenizer

        from vagen.models.cambrian_processor import SIGLIP_CACHE, SIGLIP_MODEL

        keep = {k: v for k, v in kwargs.items() if k in ("trust_remote_code", "cache_dir", "revision", "token")}
        image_processor = AutoImageProcessor.from_pretrained(
            SIGLIP_MODEL, cache_dir=SIGLIP_CACHE, local_files_only=os.environ.get("VAGEN_ALLOW_HF_DOWNLOAD", "0") != "1")
        return cls(AutoTokenizer.from_pretrained(path, **keep), image_processor)

    def save_pretrained(self, *args, **kwargs) -> None:
        return None  # LLaMA-Factory saves the tokenizer itself


def model_class():
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from vagen.models import cambrian_register as R

    class CambrianForCausalLMAdapter(R.CambrianForCausalLMAdapter):
        def __init__(self, config):
            config.image_token_alias = IMAGE_PAD_ID
            super().__init__(config)

        def forward(self, input_ids=None, labels=None, **kwargs):
            num_items = kwargs.pop("num_items_in_batch", None)
            kwargs["return_dict"] = True
            out = super().forward(input_ids=input_ids, labels=None, **kwargs)
            if labels is None:
                return out
            loss = ForCausalLMLoss(out.logits, labels, self.config.vocab_size, num_items_in_batch=num_items)
            return CausalLMOutputWithPast(loss=loss, logits=out.logits, past_key_values=out.past_key_values,
                                          hidden_states=out.hidden_states, attentions=out.attentions)

    return CambrianForCausalLMAdapter


_REGISTERED = {}


def register() -> None:
    if _REGISTERED:
        return
    from llamafactory.data.formatter import FunctionFormatter, StringFormatter, ToolFormatter
    from llamafactory.data.mm_plugin import PLUGINS
    from llamafactory.data.template import TEMPLATES, register_template
    from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, _register_composite_model
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vagen.models import cambrian_register as R

    plugin = plugin_class()
    PLUGINS.setdefault("cambrian", plugin)
    if TEMPLATE not in TEMPLATES:
        register_template(
            name=TEMPLATE,
            format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
            format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
            format_observation=StringFormatter(
                slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n"
                       "<|im_start|>assistant\n"]),
            format_tools=ToolFormatter(tool_format="qwen"),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=plugin(image_token=IMAGE_PAD, video_token=None, audio_token=None),
        )
    if TEMPLATE not in COMPOSITE_MODELS:
        _register_composite_model(
            model_type=TEMPLATE,
            projector_key="model.mm_projector",
            vision_model_keys=["model.vision_tower_aux_list"],
            language_model_keys=["model.layers", "model.embed_tokens", "model.norm", "lm_head"],
        )
    AutoProcessor.register(R.CambrianQwenConfig, CambrianSFTProcessor, exist_ok=True)
    AutoModelForCausalLM.register(R.CambrianQwenConfig, model_class(), exist_ok=True)
    _REGISTERED["done"] = True
