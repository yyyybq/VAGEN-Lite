"""Qwen3-VL in the bundled LLaMA-Factory (ca75f1e, which predates Qwen3-VL): the user's decision 2 of 2026-10-24 --
register the chat template and the vision-tower freeze rule from this file, install nothing.

- Template "qwen3_vl_nothink": upstream LLaMA-Factory's template of the same name (Qwen3-VL Instruct, no thinking
  section): the qwen2_vl layout (<|im_start|>role\\n...<|im_end|>\\n) without its default system prompt.  For images
  the multimodal plugin is the bundled Qwen2-VL one: it asks the processor's image processor for the patch grid and
  writes <|vision_start|> + grid / merge_size**2 x <|image_pad|> + <|vision_end|>, which is what Qwen3-VL's own
  processor writes (model_matrix/sft/check_lf_qwen3vl.py checks the tokens one by one).
- Plugin "qwen3_vl" (the user's decision of 2026-11-01, for P0's video conversations; images unchanged): the bundled
  Qwen2-VL plugin writes a video as one <|vision_start|> block of all its <|video_pad|> tokens and passes the frames
  through the image processor; Qwen3-VL's processor writes one block per temporal patch (two frames), each after its
  time ("<1.2 seconds>"), from its video processor, and Qwen3-VL's M-RoPE index expects that layout (one vision block
  a temporal patch).  qwen3_vl_plugin does what the processor does, as upstream LLaMA-Factory's Qwen3-VL plugin: the
  video processor for the pixels and the grid (frames as given, do_sample_frames false), the processor's own
  _calculate_timestamps for the times, from the frame indices and the fps LLaMA-Factory gives the video (data
  argument video_fps, default 2.0, for a list of frames); an odd frame count is padded with its last frame and its
  last index, as both do.  model_matrix/sft/check_lf_qwen3vl_video.py checks the tokens and the pixels against the
  processor.
- Composite model "qwen3_vl" (what freeze_vision_tower freezes): upstream's rule -- the vision encoder is pos_embed,
  patch_embed, blocks and the deepstack mergers; the merger (the projector) trains, as the round-11 recipe trains
  Qwen2.5-VL's merger (freeze_multi_modal_projector false).

``register()`` is idempotent and must run in every process before LLaMA-Factory reads the recipe
(model_matrix.sft.lf_launcher does it).
"""
from __future__ import annotations

TEMPLATE = "qwen3_vl_nothink"
PLUGIN = "qwen3_vl"


def transformers_compat() -> None:
    """Names the bundled LLaMA-Factory imports that the Qwen3-VL runtime's Transformers (4.57.6) no longer has:
    ``transformers.utils.is_jieba_available`` (llamafactory/train/sft/metric.py; renamed is_rjieba_available), given
    back with its old meaning, whether the jieba package is installed.  The first smoke SFT (2026-10-25 05:31, job
    pt-45vp0qry) stopped at that import."""
    import importlib.util

    import transformers.utils as tu

    if not hasattr(tu, "is_jieba_available"):
        tu.is_jieba_available = lambda: importlib.util.find_spec("jieba") is not None


def qwen3_vl_plugin() -> type:
    """The bundled Qwen2-VL plugin with Qwen3-VL's video layout (see the module docstring)."""
    from copy import deepcopy

    from llamafactory.data.mm_plugin import Qwen2VLPlugin, _check_video_is_nested_images
    from llamafactory.extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER

    class Qwen3VLPlugin(Qwen2VLPlugin):
        def _video_frames(self, videos, processor):
            """The frames as the Qwen2-VL plugin reads them, and per video the metadata the processor uses."""
            data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            metadata = []
            for video, frames, fps in zip(videos, data["videos"], data["fps_per_video"]):
                n = len(video) if _check_video_is_nested_images(video) else len(frames)
                metadata.append({"fps": fps, "frames_indices": list(range(n)) + [n - 1] * (len(frames) - n),
                                 "total_num_frames": len(frames)})
            return data["videos"], metadata

        def _video_inputs(self, videos, processor):
            frames, metadata = self._video_frames(videos, processor)
            out = processor.video_processor(videos=frames, video_metadata=metadata, do_sample_frames=False,
                                            return_tensors="pt")
            out.pop("video_metadata", None)  # not a model input
            return out, metadata

        def _get_mm_inputs(self, images, videos, audios, processor):
            mm_inputs = super()._get_mm_inputs(images, [], audios, processor)
            if len(videos) != 0:
                mm_inputs.update(self._video_inputs(videos, processor)[0])
            return mm_inputs

        def process_messages(self, messages, images, videos, audios, processor):
            self._validate_input(processor, images, videos, audios)
            self._validate_messages(messages, images, videos, audios)
            messages = deepcopy(messages)
            image_merge = getattr(processor.image_processor, "merge_size") ** 2
            image_grid_thw, video_grid_thw, times = [None] * len(images), [None] * len(videos), []
            if self.expand_mm_tokens:
                image_grid_thw = super()._get_mm_inputs(images, [], audios, processor).get("image_grid_thw", [])
                if len(videos) != 0:
                    out, metadata = self._video_inputs(videos, processor)
                    video_grid_thw = out["video_grid_thw"]
                    times = [processor._calculate_timestamps(list(m["frames_indices"]), m["fps"],
                                                             processor.video_processor.merge_size) for m in metadata]
            n_image = n_video = 0
            for message in messages:
                content = message["content"]
                while IMAGE_PLACEHOLDER in content:
                    seqlen = image_grid_thw[n_image].prod() // image_merge if self.expand_mm_tokens else 1
                    content = content.replace(
                        IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * seqlen}<|vision_end|>", 1)
                    n_image += 1
                while VIDEO_PLACEHOLDER in content:
                    if self.expand_mm_tokens:
                        grid = video_grid_thw[n_video]
                        seqlen = grid[1:].prod() // processor.video_processor.merge_size ** 2
                        layout = "".join(f"<{times[n_video][t]:.1f} seconds><|vision_start|>"
                                         f"{self.video_token * seqlen}<|vision_end|>" for t in range(int(grid[0])))
                    else:
                        layout = f"<|vision_start|>{self.video_token}<|vision_end|>"
                    content = content.replace(VIDEO_PLACEHOLDER, layout, 1)
                    n_video += 1
                message["content"] = content
            return messages

    return Qwen3VLPlugin


def register() -> None:
    transformers_compat()  # before any LLaMA-Factory module is imported
    from llamafactory.data.formatter import FunctionFormatter, StringFormatter, ToolFormatter
    from llamafactory.data.mm_plugin import PLUGINS, get_mm_plugin, register_mm_plugin
    from llamafactory.data.template import TEMPLATES, register_template
    from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, _register_composite_model

    if PLUGIN not in PLUGINS:
        register_mm_plugin(PLUGIN, qwen3_vl_plugin())
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
            mm_plugin=get_mm_plugin(name=PLUGIN, image_token="<|image_pad|>", video_token="<|video_pad|>"),
        )
    if "qwen3_vl" not in COMPOSITE_MODELS:
        _register_composite_model(
            model_type="qwen3_vl",
            projector_key="visual.merger",
            vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
            language_model_keys=["language_model", "lm_head"],
            lora_conflict_keys=["patch_embed"],
        )
