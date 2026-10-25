"""Qwen3-VL in the bundled LLaMA-Factory (ca75f1e, which predates Qwen3-VL): the user's decision 2 of 2026-10-24 --
register the chat template and the vision-tower freeze rule from this file, install nothing.

- Template "qwen3_vl_nothink": upstream LLaMA-Factory's template of the same name (Qwen3-VL Instruct, no thinking
  section): the qwen2_vl layout (<|im_start|>role\\n...<|im_end|>\\n) without its default system prompt.  The
  multimodal plugin is the bundled Qwen2-VL one: for images it asks the processor's image processor for the patch grid
  and writes <|vision_start|> + grid / merge_size**2 x <|image_pad|> + <|vision_end|>, which is what Qwen3-VL's own
  processor writes (model_matrix/sft/check_lf_qwen3vl.py checks the tokens one by one).  Upstream's own Qwen3-VL
  plugin differs only for videos (timestamps), which these SFT data have none of.
- Composite model "qwen3_vl" (what freeze_vision_tower freezes): upstream's rule -- the vision encoder is pos_embed,
  patch_embed, blocks and the deepstack mergers; the merger (the projector) trains, as the round-11 recipe trains
  Qwen2.5-VL's merger (freeze_multi_modal_projector false).

``register()`` is idempotent and must run in every process before LLaMA-Factory reads the recipe
(model_matrix.sft.lf_launcher does it).
"""
from __future__ import annotations

TEMPLATE = "qwen3_vl_nothink"


def transformers_compat() -> None:
    """Names the bundled LLaMA-Factory imports that the Qwen3-VL runtime's Transformers (4.57.6) no longer has:
    ``transformers.utils.is_jieba_available`` (llamafactory/train/sft/metric.py; renamed is_rjieba_available), given
    back with its old meaning, whether the jieba package is installed.  The first smoke SFT (2026-10-25 05:31, job
    pt-45vp0qry) stopped at that import."""
    import importlib.util

    import transformers.utils as tu

    if not hasattr(tu, "is_jieba_available"):
        tu.is_jieba_available = lambda: importlib.util.find_spec("jieba") is not None


def register() -> None:
    transformers_compat()  # before any LLaMA-Factory module is imported
    from llamafactory.data.formatter import FunctionFormatter, StringFormatter, ToolFormatter
    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import TEMPLATES, register_template
    from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, _register_composite_model

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
            mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
        )
    if "qwen3_vl" not in COMPOSITE_MODELS:
        _register_composite_model(
            model_type="qwen3_vl",
            projector_key="visual.merger",
            vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
            language_model_keys=["language_model", "lm_head"],
            lora_conflict_keys=["patch_embed"],
        )
