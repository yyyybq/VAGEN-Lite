"""q38: LLaMA-Factory main's SFT encoding of Qwen3.8-27B against the model's own chat template, CPU only
(port of model_matrix/sft/check_lf_qwen3vl.py and, for videos, model_matrix/sft/check_lf_qwen3vl_video.py;
SFT_OPTIONS.md 2.3 and 7, TEMPLATE_CHECK.md, SMOKE_PLAN 2b).

Modes (the folder is one written by vagen.q38.sft.lf_data, whose build_manifest.json maps each record back to its line
of the 8B source file):

  tokens    a seeded sample of records (--media video / image: only records with a video / with images and no video).
            LLaMA-Factory (template qwen3_8, enable_thinking false, mask_history false, SupervisedDatasetProcessor as
            the SFT job runs it, with the media arguments of the folder's train.yaml: image_max_pixels,
            video_max_pixels, video_fps, ...) encodes the converted record; Qwen3.8's processor (AutoProcessor, as the
            RL agent loop and the P0 evaluation load it) encodes the SOURCE record (not stripped: what the RL agent
            loop renders, the template trims it) with ``apply_chat_template(enable_thinking=False)``, the images where
            the <image> placeholders stand, as content blocks (gym_agent_loop_v1.attach_images), and each <video> as a
            video block of its frames with the metadata the P0 evaluation gives a list of frames
            (data_gen/interiorgs_qa/evaluate.py video_input: frames as given, indices 0..n-1, fps video_fps,
            do_sample_frames false).  Per record:
              ids_identical          LF ids == processor ids, one by one;
              labels                 the supervised tokens decode to every (stripped) answer + "<|im_end|>\\n" and
                                     nothing else, one span per assistant turn, labels == ids there;
              think_block            the 4 tokens before every span are <think> \\n\\n </think> \\n\\n (248068 271
                                     248069 271) and not supervised (TEMPLATE_CHECK 4);
              generation_prefix      for every assistant turn the processor's generation prompt (add_generation_prompt,
                                     enable_thinking False) of the turns before it is a token prefix of the full ids
                                     and ends where LF's span of that turn starts (what RL's prompt is);
              image tokens           <|image_pad|> runs per image == t*h*w/merge**2 of the processor's grid;
              video tokens           <|video_pad|> runs == h*w/merge**2 per temporal patch (t per video) of the
                                     processor's video grid, and the "<x.x seconds>" texts in LF's ids == the times of
                                     the temporal patches (frame indices 0..n-1, an odd count padded with its last
                                     index, mean of each pair / fps; lf_qwen3vl.py's rule for the 8B);
              outer_video_markers    (reported, not a pass condition) per video whether the per-frame blocks sit
                                     inside one more <|vision_start|> ... <|vision_end|> pair, in LF's ids and in the
                                     processor's (transformers >= 5.10 expands only the bare <|video_pad|> and keeps
                                     the template's pair, vLLM 0.28 follows it: qwen3_vl.py _expands_only_video_token);
              no_reasoning_text      no "Reasoning effort" text, no "<think>" outside the empty blocks;
            and for the first --pixels records the pixel values and grids of LF's mm plugin (images and videos) == the
            processor's, and each video's pixels from its frames as one uint8 array (the P0 evaluation's input) ==
            LF's; for the first --positions records the M-RoPE position ids of LF's SFT collator (model_type qwen3_5,
            a weightless model on the meta device) == the model's own get_rope_index on the processor's inputs.
  lengths   every record of every dataset of the folder: LF's untruncated length (cutoff 10**9), how many exceed the
            recipe's cutoff_len (16384), quantiles, the longest records.
  pipeline  LLaMA-Factory's own loader (get_train_args on the folder's train.yaml + get_dataset, max_samples N): the
            datasets library of the q38 env (5.0.1, DISABLE_VERSION_CHECK=1) loads, converts and tokenizes the first N
            records of each split; they must equal the direct encoding of those records, and the images / videos the
            loader keeps for the collator must be the record's.
  fsdp      a weightless model (meta device, bf16 as loaded), LLaMA-Factory's freeze rule (freeze_vision_tower true,
            freeze_multi_modal_projector false) and fp32 upcast of trainable weights; every class of the accelerate
            config's fsdp_transformer_layer_cls_to_wrap exists in the model, and every FSDP1 unit (the nearest wrapped
            ancestor of each parameter, or the root) holds one dtype; frozen / trainable parameter counts.
  p0_eval   the P0 evaluation's video prompts: a seeded sample of the video requests of a predictions.jsonl that
            data_gen.interiorgs_qa.evaluate wrote (it stores vLLM's prompt length per request), each rebuilt as its
            run_vllm builds it (the selection and its build_requests, the video block then the question, the chat
            template with the model type's kwargs, the frames as one uint8 array with video_input's metadata) and
            encoded by Qwen3.8's processor; next to it LLaMA-Factory's encoding of the same frames and question in
            the training layout (export_lf.py: "<video>\\n" + question, the answer as the assistant turn).  Per request:
              prompt_tokens_equal_vllm   the processor's prompt length == vLLM's prompt_tokens;
              video_part_identical       the ids from the user turn's start to the last <|vision_end|> == LF's
                                         ("_with_outer_markers": LF's for "<|vision_start|><video><|vision_end|>",
                                         lf_data's proposed --video-markers outer);
              tail                       the ids after the last <|vision_end|> up to the answer, both sides (export_lf
                                         puts "\\n" before the question, evaluate.py does not: data_gen/interiorgs_qa
                                         export_lf.py:119, evaluate.py:163-164, both from before the q38 line);
              mrope_vllm_identical       vLLM's M-RoPE positions of the eval prompt (the static
                                         Qwen3VLForConditionalGeneration._get_mrope_input_positions, which vLLM 0.28's
                                         Qwen3.5 model inherits) == the model's get_rope_index on the processor's inputs;
              video_pixels_identical     the processor's pixel values of the eval input == LF's.

    cd VAGEN-Lite-q38 && DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=1 ... PYTHONPATH=$PWD nice -n 19 \\
      /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m vagen.q38.sft.check_lf_qwen35 tokens \\
      --lf-dir LF --model MODEL --dataset trajectory_train --n 200 --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

IGNORE = -100
THINK_IDS = [248068, 271, 248069, 271]  # <think> \n\n </think> \n\n (TEMPLATE_CHECK 4)
ASSISTANT_HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE = "qwen3_8"
CUTOFF = 16384
VIDEO_FPS = 2.0  # LLaMA-Factory's video_fps default, which the 8B recipes keep; evaluate.py --video-fps default
# the media arguments of a folder's train.yaml that LLaMA-Factory's ModelArguments take (the 8B recipes set
# image_max_pixels / video_max_pixels 65536 and keep the rest at LF's defaults)
LF_MEDIA_KEYS = ("image_max_pixels", "image_min_pixels", "video_max_pixels", "video_min_pixels", "video_fps",
                 "video_maxlen")
SECONDS = re.compile(r"<([0-9]+\.[0-9]) seconds>")
VIDEO_RUN = re.compile(r"(<\|vision_start\|>)?((?:<[0-9]+\.[0-9] seconds><\|vision_start\|>(?:<\|video_pad\|>)+"
                       r"<\|vision_end\|>)+)(<\|vision_end\|>)?")


# ----------------------------------------------------------------------------------------------------------- inputs
def read_manifest(lf_dir: Path) -> Dict[str, Any]:
    return json.loads((lf_dir / "build_manifest.json").read_text())


def dataset_file(lf_dir: Path, name: str) -> Path:
    info = json.loads((lf_dir / "dataset_info.json").read_text())
    return lf_dir / info[name]["file_name"]


def read_lines(path: Path) -> List[str]:
    return [l for l in path.read_text().splitlines() if l.strip()]


def source_records(manifest: Dict[str, Any], name: str, lf_lines: int) -> Tuple[Path, List[int]]:
    """(the 8B source file, its line number for each line of the converted file)."""
    d = manifest["datasets"][name]
    sel = d["selected_lines"]
    lines = list(range(d["source_records"])) if sel == "all" else list(sel)
    if len(lines) != lf_lines:
        raise SystemExit(f"{name}: the manifest lists {len(lines)} records, the file has {lf_lines}")
    return Path(d["source"]), lines


def lf_media_args(lf_dir: Path) -> Dict[str, Any]:
    """The media arguments of the folder's train.yaml (LF_MEDIA_KEYS), {} without a train.yaml."""
    p = lf_dir / "train.yaml"
    if not p.is_file():
        return {}
    import yaml

    cfg = yaml.safe_load(p.read_text()) or {}
    return {k: cfg[k] for k in LF_MEDIA_KEYS if k in cfg}


def media_kind(rec: Dict[str, Any]) -> str:
    return "video" if rec.get("videos") else "image" if rec.get("images") else "text"


# ----------------------------------------------------------------------------------------------------------- LF side
class LFEncoder:
    """LLaMA-Factory's tokenizer, processor, qwen3_8 template and SupervisedDatasetProcessor, as run_sft builds them."""

    def __init__(self, model: str, cutoff_len: int = CUTOFF, enable_thinking: bool = False,
                 media_args: Optional[Dict[str, Any]] = None):
        from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.model import load_tokenizer

        self.media_args = dict(media_args or {})
        self.model_args = ModelArguments(model_name_or_path=model, **self.media_args)
        tm = load_tokenizer(self.model_args)
        self.tokenizer, self.processor = tm["tokenizer"], tm["processor"]
        self.data_args = DataArguments(template=TEMPLATE, cutoff_len=cutoff_len, mask_history=False,
                                       enable_thinking=enable_thinking)
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.proc = SupervisedDatasetProcessor(template=self.template, tokenizer=self.tokenizer,
                                               processor=self.processor, data_args=self.data_args)
        self.video_fps = float(self.model_args.video_fps)
        self._hf_processor = None

    @property
    def hf_processor(self):
        """Qwen3.8's processor as AutoProcessor loads it (the RL agent loop's and the P0 evaluation's), not LF's
        patched copy: the reference side."""
        if self._hf_processor is None:
            from transformers import AutoProcessor

            self._hf_processor = AutoProcessor.from_pretrained(self.model_args.model_name_or_path)
        return self._hf_processor

    def encode(self, rec: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        msgs = rec["messages"]
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
        turns = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
        return self.proc._encode_data_example(turns[:-1], turns[-1:], system, None, list(rec.get("images") or []),
                                              videos_of(rec), [])

    def mm_inputs(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """The pixel values and grids the SFT collator gets from the qwen3_vl plugin (mm_plugin._get_mm_inputs)."""
        return self.template.mm_plugin._get_mm_inputs(list(rec.get("images") or []), videos_of(rec), [], self.processor)


def videos_of(rec: Dict[str, Any]) -> List[List[str]]:
    """A record's videos as new lists: LF's Qwen2-VL _regularize_videos pads an odd frame list in place (appends its
    last frame), which would change the record for the next encoding."""
    return [list(v) for v in rec.get("videos") or []]


# ----------------------------------------------------------------------------------------------------------- HF side
def video_metadata(n: int, fps: float) -> Dict[str, Any]:
    """The metadata the P0 evaluation gives a video of n frames (data_gen/interiorgs_qa/evaluate.py video_input, without
    its do_sample_frames switch, which goes to the processor call)."""
    return {"fps": fps, "frames_indices": list(range(n)), "total_num_frames": n, "duration": n / fps}


def expected_seconds(n: int, fps: float, temporal: int = 2) -> List[str]:
    """The "<x.x seconds>" texts of a video of n frames given as frames: indices 0..n-1, an odd count padded with its
    last index, the mean of each temporal patch's first and last / fps (lf_qwen3vl.py's rule for the 8B)."""
    idx = list(range(n)) + [n - 1] * (-n % temporal)
    return [f"{(idx[i] + idx[i + temporal - 1]) / 2 / fps:.1f}" for i in range(0, len(idx), temporal)]


def to_blocks_av(messages: List[Dict[str, Any]], image_paths: List[str], video_paths: List[List[str]],
                 video_fps: float = VIDEO_FPS):
    """Messages with each user turn's <image> / <video> placeholders as {"type": "image"} / {"type": "video"} blocks,
    the PIL images in order, the videos (lists of PIL frames, as given) and their metadata (video_metadata)."""
    from PIL import Image

    out, images, videos, metadata = [], [], [], []
    it_img, it_vid = iter(image_paths), iter(video_paths)
    for m in messages:
        if m["role"] != "user":
            out.append({"role": m["role"], "content": m["content"]})
            continue
        content = []
        for seg in re.split(r"(<image>|<video>)", m["content"]):
            if seg == "<image>":
                content.append({"type": "image"})
                images.append(Image.open(next(it_img)).convert("RGB"))
            elif seg == "<video>":
                frames = next(it_vid, None)
                if frames is None:
                    raise ValueError("a <video> placeholder without its video")
                content.append({"type": "video"})
                videos.append([Image.open(f).convert("RGB") for f in frames])
                metadata.append(video_metadata(len(frames), video_fps))
            elif seg:
                content.append({"type": "text", "text": seg})
        out.append({"role": "user", "content": content})
    return out, images, videos, metadata


def to_blocks(messages: List[Dict[str, Any]], image_paths: List[str]):
    """Messages with each user turn's <image> placeholders as {"type": "image"} blocks (the RL agent loop's layout),
    and the PIL images in order."""
    out, images, _, _ = to_blocks_av(messages, image_paths, [])
    return out, images


def hf_encode(processor, messages, images, add_generation_prompt: bool = False, videos=None, metadata=None):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         enable_thinking=False)
    kwargs: Dict[str, Any] = {}
    if videos:
        kwargs.update(videos=videos, video_metadata=metadata, do_sample_frames=False)
    out = processor(text=[text], images=images or None, return_tensors="pt", **kwargs)
    return text, out


def spans(labels: List[int]) -> List[List[int]]:
    runs, start = [], None
    for i, v in enumerate(list(labels) + [IGNORE]):
        if v != IGNORE and start is None:
            start = i
        elif v == IGNORE and start is not None:
            runs.append([start, i])
            start = None
    return runs


def image_runs(ids: List[int], pad: int) -> List[int]:
    return [len(m) for m in re.findall(r"P+", "".join("P" if t == pad else "." for t in ids))]


def outer_video_markers(text: str) -> List[bool]:
    """Per video of a decoded sequence (a run of per-frame blocks "<x.x seconds><|vision_start|> pads <|vision_end|>"):
    whether one more <|vision_start|> opens the run and one more <|vision_end|> closes it."""
    return [bool(m.group(1) and m.group(3)) for m in VIDEO_RUN.finditer(text)]


def check_record(enc: LFEncoder, conv: Dict[str, Any], src: Dict[str, Any], prefix: bool = True) -> Dict[str, Any]:
    tok, processor = enc.tokenizer, enc.hf_processor
    ids, labels = enc.encode(conv)
    src_images, src_videos = list(src.get("images") or []), list(src.get("videos") or [])
    blocks, images, videos, vmeta = to_blocks_av(src["messages"], src_images, src_videos, enc.video_fps)
    hf_text, out = hf_encode(processor, blocks, images, videos=videos, metadata=vmeta)
    hf_ids = out["input_ids"][0].tolist()
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    vpad = tok.convert_tokens_to_ids("<|video_pad|>")
    merge = processor.image_processor.merge_size
    grids = out["image_grid_thw"].tolist() if "image_grid_thw" in out else []
    want_per_image = [t * h * w // (merge * merge) for t, h, w in grids]
    vmerge = processor.video_processor.merge_size
    temporal = processor.video_processor.temporal_patch_size
    vgrids = out["video_grid_thw"].tolist() if "video_grid_thw" in out else []
    want_video_runs = [h * w // (vmerge * vmerge) for t, h, w in vgrids for _ in range(t)]
    want_seconds = [s for v in src_videos for s in expected_seconds(len(v), enc.video_fps, temporal)]
    first_diff = next((i for i, (a, b) in enumerate(zip(ids, hf_ids)) if a != b), None)
    if first_diff is None and len(ids) != len(hf_ids):
        first_diff = min(len(ids), len(hf_ids))
    answers = [m["content"].strip() for m in src["messages"] if m["role"] == "assistant"]
    sp = spans(labels)
    supervised = [t for t, lab in zip(ids, labels) if lab != IGNORE]
    think_ok = len(sp) == len(answers) and all(
        ids[s - 4:s] == THINK_IDS and all(l == IGNORE for l in labels[s - 4:s])
        and tok.decode(ids[s - 7:s]) == ASSISTANT_HEAD for s, _ in sp)
    span_text_ok = len(sp) == len(answers) and all(
        tok.decode(ids[s:e]) == a + "<|im_end|>\n" for (s, e), a in zip(sp, answers))
    lf_decoded = tok.decode(ids)
    r: Dict[str, Any] = {
        "assistant_turns": len(answers), "images": len(images), "videos": len(videos),
        "frames_per_video": [len(v) for v in src_videos], "tokens_lf": len(ids), "tokens_hf": len(hf_ids),
        "ids_identical": ids == hf_ids, "first_difference": first_diff,
        "image_tokens_per_image_lf": image_runs(ids, pad), "image_tokens_per_image_processor": want_per_image,
        "supervised_tokens": len(supervised), "supervised_spans": len(sp),
        "supervised_text_is_answers": tok.decode(supervised) == "".join(a + "<|im_end|>\n" for a in answers),
        "spans_are_answers": span_text_ok,
        "labels_equal_ids_where_supervised": all(lab == t for t, lab in zip(ids, labels) if lab != IGNORE),
        "think_block": think_ok,
        "no_reasoning_text": "Reasoning effort" not in hf_text
                             and hf_text.count("<think>") == len(answers) and lf_decoded.count("<think>") == len(answers),
    }
    r["image_counts_ok"] = r["image_tokens_per_image_lf"] == want_per_image
    if src_videos:
        lf_runs, lf_seconds = image_runs(ids, vpad), SECONDS.findall(lf_decoded)
        r.update(video_grid_processor=vgrids, video_tokens_lf=sum(lf_runs), video_blocks_lf=len(lf_runs),
                 video_counts_ok=lf_runs == want_video_runs, seconds_lf=lf_seconds,
                 seconds_ok=lf_seconds == want_seconds and SECONDS.findall(tok.decode(hf_ids)) == want_seconds,
                 outer_video_markers_lf=outer_video_markers(lf_decoded),
                 outer_video_markers_processor=outer_video_markers(tok.decode(hf_ids)))
    if prefix:
        ok, ends = True, []
        msgs = src["messages"]
        a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for k, i in enumerate(a_idx):
            n_img = sum(m["content"].count("<image>") for m in msgs[:i] if m["role"] == "user")
            n_vid = sum(m["content"].count("<video>") for m in msgs[:i] if m["role"] == "user")
            pb, pim, pvid, pmeta = to_blocks_av(msgs[:i], src_images[:n_img], src_videos[:n_vid], enc.video_fps)
            _, pout = hf_encode(processor, pb, pim, add_generation_prompt=True, videos=pvid, metadata=pmeta)
            p_ids = pout["input_ids"][0].tolist()
            ends.append(len(p_ids))
            ok = ok and p_ids == hf_ids[: len(p_ids)] and k < len(sp) and sp[k][0] == len(p_ids)
        r["generation_prefix"] = ok
        r["generation_prompt_ends"] = ends
    else:
        r["generation_prefix"] = None
    if first_diff is not None:
        r["context"] = {"lf": tok.decode(ids[max(0, first_diff - 20):first_diff + 20]),
                        "hf": tok.decode(hf_ids[max(0, first_diff - 20):first_diff + 20])}
    r["ok"] = bool(r["ids_identical"] and r["image_counts_ok"] and r["supervised_text_is_answers"] and span_text_ok
                   and r["labels_equal_ids_where_supervised"] and think_ok and r["no_reasoning_text"]
                   and r["generation_prefix"] is not False and r.get("video_counts_ok", True)
                   and r.get("seconds_ok", True))
    r["_hf"] = (out, hf_ids)
    return r


def _same(a, b) -> bool:
    import torch

    return a is not None and b is not None and a.shape == b.shape and bool(torch.equal(a, b))


def eval_video_array(frames: List[str]):
    """A video as the P0 evaluation hands it to vLLM: its frames as one uint8 array (n, H, W, 3) (evaluate.py run_vllm)."""
    import numpy as np
    from PIL import Image

    return np.stack([np.array(Image.open(f).convert("RGB")) for f in frames])


def check_pixels(enc: LFEncoder, conv: Dict[str, Any], hf_out) -> Dict[str, Any]:
    lf = enc.mm_inputs(conv)
    r: Dict[str, Any] = {}
    if conv.get("images"):
        r.update(pixels_identical=_same(lf.get("pixel_values"), hf_out.get("pixel_values")),
                 grid_identical=_same(lf.get("image_grid_thw"), hf_out.get("image_grid_thw")),
                 pixel_shape=list(lf["pixel_values"].shape))
    if conv.get("videos"):
        r.update(video_pixels_identical=_same(lf.get("pixel_values_videos"), hf_out.get("pixel_values_videos")),
                 video_grid_identical=_same(lf.get("video_grid_thw"), hf_out.get("video_grid_thw")),
                 video_pixel_shape=list(lf["pixel_values_videos"].shape))
        vids = list(conv["videos"])
        ev = enc.hf_processor.video_processor(videos=[eval_video_array(v) for v in vids],
                                              video_metadata=[video_metadata(len(v), enc.video_fps) for v in vids],
                                              do_sample_frames=False, return_tensors="pt")
        r["video_pixels_eval_identical"] = _same(ev["pixel_values_videos"], lf.get("pixel_values_videos")) \
            and _same(ev["video_grid_thw"], lf.get("video_grid_thw"))
    r["pixels_ok"] = all(v for k, v in r.items() if k.endswith("_identical"))
    return r


_META_MODEL = None


def meta_model(model: str):
    """Qwen3_5ForConditionalGeneration on the meta device (no weights), bf16 as LF loads it."""
    global _META_MODEL
    if _META_MODEL is None:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText

        config = AutoConfig.from_pretrained(model)
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                _META_MODEL = AutoModelForImageTextToText.from_config(config)
        finally:
            torch.set_default_dtype(old)
    return _META_MODEL


def check_positions(enc: LFEncoder, model: str, ids: List[int], labels: List[int], conv: Dict[str, Any], hf_out) -> Dict[str, Any]:
    """M-RoPE positions of LF's SFT collator (as run_sft builds it) vs the model's own get_rope_index on the
    processor's inputs (what the forward pass computes when no position_ids are given)."""
    import torch
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask

    m = meta_model(model)
    coll = SFTDataCollatorWith4DAttentionMask(template=enc.template, model=m, pad_to_multiple_of=8,
                                              label_pad_token_id=IGNORE, block_diag_attn=False, neat_packing=False,
                                              attn_implementation="sdpa", compute_dtype=torch.bfloat16,
                                              tokenizer=enc.tokenizer, processor=enc.processor)
    feats = [{"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
              "images": list(conv.get("images") or []), "videos": videos_of(conv), "audios": []}]
    batch = coll(feats)
    lf_pos = batch["position_ids"][..., : len(ids)]
    ref_pos, _ = m.model.get_rope_index(input_ids=hf_out["input_ids"], mm_token_type_ids=hf_out["mm_token_type_ids"],
                                        image_grid_thw=hf_out.get("image_grid_thw"),
                                        video_grid_thw=hf_out.get("video_grid_thw"),
                                        attention_mask=hf_out["attention_mask"])
    # LF passes 4 rows (transformers issue 39400): row 0 is a copy of the temporal row and becomes the text model's
    # text_position_ids, which only reach create_causal_mask; rows 1-3 are the M-RoPE positions.  The model's own
    # forward (no position_ids) computes the 3 rows with get_rope_index and no text row.
    mrope = lf_pos[1:] if lf_pos.shape[0] == 4 else lf_pos
    same = bool(mrope.shape == ref_pos.shape and torch.equal(mrope.cpu(), ref_pos.cpu()))
    row0_is_t = bool(lf_pos.shape[0] == 4 and torch.equal(lf_pos[0], lf_pos[1]))
    # the text row changes nothing in the mask while an attention_mask is given (packed-sequence detection runs only
    # without one, transformers masking_utils): the causal mask with LF's text row == the mask without it
    from transformers.masking_utils import create_causal_mask

    cfg = m.config.get_text_config()
    cfg._attn_implementation = "sdpa"
    emb = torch.zeros(1, len(ids), 8, dtype=torch.bfloat16)
    am = batch["attention_mask"][:, : len(ids)]
    with_row = create_causal_mask(config=cfg, inputs_embeds=emb, attention_mask=am, past_key_values=None,
                                  position_ids=lf_pos[0])
    without = create_causal_mask(config=cfg, inputs_embeds=emb, attention_mask=am, past_key_values=None, position_ids=None)
    mask_same = (with_row is None and without is None) or (
        with_row is not None and without is not None and torch.equal(with_row, without))
    return {"positions_identical": same and row0_is_t and bool(mask_same), "mrope_rows_identical": same,
            "text_row_is_temporal_row": row0_is_t, "causal_mask_unchanged_by_text_row": bool(mask_same),
            "position_shape": list(lf_pos.shape), "collator_keys": sorted(batch.keys())}


def mode_tokens(a) -> Dict[str, Any]:
    lf_dir = Path(a.lf_dir)
    manifest = read_manifest(lf_dir)
    conv_lines = read_lines(dataset_file(lf_dir, a.dataset))
    src_path, src_idx = source_records(manifest, a.dataset, len(conv_lines))
    src_lines = read_lines(src_path)
    pool = range(len(conv_lines)) if a.media == "any" else [
        i for i, l in enumerate(conv_lines) if media_kind(json.loads(l)) == a.media]
    pick = sorted(random.Random(a.seed).sample(list(pool), min(a.n, len(pool))))
    enc = LFEncoder(a.model, media_args=lf_media_args(lf_dir))
    results, t0 = [], time.time()
    for j, i in enumerate(pick):
        conv, src = json.loads(conv_lines[i]), json.loads(src_lines[src_idx[i]])
        r = check_record(enc, conv, src, prefix=not a.no_prefix)
        out, hf_ids = r.pop("_hf")
        if j < a.pixels:
            r.update(check_pixels(enc, conv, out))
            r["ok"] = r["ok"] and r["pixels_ok"]
        if j < a.positions:
            ids, labels = enc.encode(conv)
            r.update(check_positions(enc, a.model, ids, labels, conv, out))
            r["ok"] = r["ok"] and r["positions_identical"]
        r.update(line=i + 1, source_line=src_idx[i] + 1)
        results.append(r)
        if (j + 1) % 25 == 0:
            print(f"[tokens] {j + 1}/{len(pick)} ok so far {sum(x['ok'] for x in results)} ({time.time() - t0:.0f} s)", flush=True)
    keys = ("ids_identical", "image_counts_ok", "supervised_text_is_answers", "spans_are_answers",
            "labels_equal_ids_where_supervised", "think_block", "no_reasoning_text", "generation_prefix",
            "pixels_identical", "grid_identical", "positions_identical", "video_counts_ok", "seconds_ok",
            "video_pixels_identical", "video_grid_identical", "video_pixels_eval_identical")
    tally = {k: [sum(1 for r in results if r.get(k) is True), sum(1 for r in results if k in r and r[k] is not None)] for k in keys}
    markers = {side: sorted({str(r[f"outer_video_markers_{side}"]) for r in results if f"outer_video_markers_{side}" in r})
               for side in ("lf", "processor")}
    return {"mode": "tokens", "lf_dir": str(lf_dir), "dataset": a.dataset, "model": a.model, "seed": a.seed,
            "media": a.media, "lf_media_args": enc.media_args, "video_fps": enc.video_fps,
            "records_checked": len(results), "records_ok": sum(r["ok"] for r in results), "tally_pass_of_checked": tally,
            "outer_video_markers": markers,
            "seconds": round(time.time() - t0, 1), "failures": [r for r in results if not r["ok"]][:20],
            "results": results, "ok": all(r["ok"] for r in results) and len(results) >= min(a.n, len(pool))}


def mode_lengths(a) -> Dict[str, Any]:
    lf_dir = Path(a.lf_dir)
    info = json.loads((lf_dir / "dataset_info.json").read_text())
    enc = LFEncoder(a.model, cutoff_len=10 ** 9, media_args=lf_media_args(lf_dir))
    report: Dict[str, Any] = {"mode": "lengths", "lf_dir": str(lf_dir), "model": a.model, "cutoff_len": a.cutoff,
                              "lf_media_args": enc.media_args, "datasets": {}}
    t0 = time.time()
    for name in info:
        lines = read_lines(lf_dir / info[name]["file_name"])
        if a.limit:
            lines = lines[: a.limit]
        lens, sup, rows = [], [], []
        for i, l in enumerate(lines):
            rec = json.loads(l)
            ids, labels = enc.encode(rec)
            lens.append(len(ids))
            sup.append(sum(1 for x in labels if x != IGNORE))
            rows.append((len(ids), i + 1, len(rec.get("images") or []), sum(1 for m in rec["messages"] if m["role"] == "assistant"),
                         len(rec.get("videos") or [])))
            if (i + 1) % 1000 == 0:
                print(f"[lengths] {name} {i + 1}/{len(lines)} ({time.time() - t0:.0f} s)", flush=True)
        s = sorted(lens)
        q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))] if s else None
        report["datasets"][name] = {
            "records": len(lens), "over_cutoff": sum(1 for x in lens if x > a.cutoff), "max": max(lens) if lens else None,
            "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "mean": round(sum(lens) / len(lens), 1) if lens else None,
            "supervised_tokens_total": sum(sup), "tokens_total": sum(lens),
            "records_with_video": sum(1 for row in rows if row[4]),
            "longest": [{"tokens": t, "line": ln, "images": im, "assistant_turns": at, "videos": vd}
                        for t, ln, im, at, vd in sorted(rows, reverse=True)[:10]]}
    report["seconds"] = round(time.time() - t0, 1)
    report["ok"] = all(d["over_cutoff"] == 0 for d in report["datasets"].values())
    return report


def mode_pipeline(a) -> Dict[str, Any]:
    """LF's own loader on the folder's train.yaml (CPU: bf16 off, no output written but the cache folder given)."""
    import yaml
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    lf_dir = Path(a.lf_dir)
    cfg = yaml.safe_load((lf_dir / "train.yaml").read_text())
    cfg.update(model_name_or_path=a.model, max_samples=a.n, bf16=False, output_dir=a.scratch, overwrite_cache=True,
               preprocessing_num_workers=None, report_to="none")
    cfg.pop("flash_attn", None)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(cfg)
    tm = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tm["tokenizer"], data_args)
    t0 = time.time()
    dm = get_dataset(template, model_args, data_args, training_args, stage="sft", **tm)
    enc = LFEncoder(a.model, media_args=lf_media_args(lf_dir))
    import datasets as hf_datasets

    out: Dict[str, Any] = {"mode": "pipeline", "lf_dir": str(lf_dir), "datasets_version": hf_datasets.__version__,
                           "template_enable_thinking": template.enable_thinking,
                           "template_preserve_thinking": template.preserve_thinking,
                           "lf_media_args": enc.media_args, "splits": {}}
    for split, names in (("train_dataset", cfg["dataset"]), ("eval_dataset", cfg.get("eval_dataset"))):
        got = dm.get(split)
        if got is None or names is None:
            continue
        names = [n.strip() for n in str(names).split(",") if n.strip()]
        # several datasets: LF concatenates them in the order given (mix_strategy concat), each cut to max_samples;
        # eval_on_each_dataset gives the eval split as {name: dataset}
        parts = [(f"{split}/{k}", [k], v) for k, v in got.items()] if isinstance(got, dict) else [(split, names, got)]
        for key, part_names, ds in parts:
            lines: List[str] = []
            for name in part_names:
                lines += read_lines(dataset_file(lf_dir, name))[: a.n]
            same = same_media = videos = 0
            for i in range(min(len(ds), len(lines))):
                rec, row = json.loads(lines[i]), ds[i]
                ids, labels = enc.encode(rec)
                same += int(list(row["input_ids"]) == ids and list(row["labels"]) == labels)
                same_media += int(list(row.get("images") or []) == list(rec.get("images") or [])
                                  and [list(v) for v in (row.get("videos") or [])] == [list(v) for v in (rec.get("videos") or [])])
                videos += int(bool(rec.get("videos")))
            out["splits"][key] = {"dataset": ",".join(part_names), "rows": len(ds), "expected_rows": len(lines),
                                  "identical_to_direct_encoding": same, "media_identical": same_media,
                                  "rows_with_video": videos, "columns": list(ds.column_names)}
    out["seconds"] = round(time.time() - t0, 1)
    out["ok"] = bool(out["splits"]) and all(s["rows"] == s["expected_rows"] and s["identical_to_direct_encoding"] == s["rows"]
                                            and s["media_identical"] == s["rows"] for s in out["splits"].values()) \
        and template.enable_thinking is False
    return out


def wrap_classes(accelerate_yaml: Path) -> List[str]:
    import yaml

    cfg = yaml.safe_load(accelerate_yaml.read_text())
    return [c.strip() for c in cfg["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"].split(",") if c.strip()]


def mode_fsdp(a) -> Dict[str, Any]:
    import torch
    from llamafactory.model.adapter import _setup_full_tuning
    from llamafactory.model.model_utils.visual import get_forbidden_modules
    from llamafactory.hparams import FinetuningArguments

    m = meta_model(a.model)
    fa = FinetuningArguments(finetuning_type="full", freeze_vision_tower=True, freeze_multi_modal_projector=False)
    forbidden = sorted(get_forbidden_modules(m.config, fa))
    _setup_full_tuning(m, fa, is_trainable=True, cast_trainable_params_to_fp32=True)
    classes = wrap_classes(Path(a.accelerate_config))
    present = {c: 0 for c in classes}
    owner: Dict[str, str] = {}
    for name, mod in m.named_modules():
        if type(mod).__name__ in present:
            present[type(mod).__name__] += 1
    units: Dict[str, set] = {}
    wrapped = [(n, mod) for n, mod in m.named_modules() if type(mod).__name__ in present]
    wrapped_names = sorted((n for n, _ in wrapped), key=len, reverse=True)
    for pname, p in m.named_parameters():
        unit = next((w for w in wrapped_names if w and (pname.startswith(w + "."))), "<root>")
        owner[pname] = unit
        units.setdefault(unit, set()).add(str(p.dtype))
    mixed = {u: sorted(d) for u, d in units.items() if len(d) > 1}
    frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    trainable_dtypes = sorted({str(p.dtype) for p in m.parameters() if p.requires_grad})
    frozen_dtypes = sorted({str(p.dtype) for p in m.parameters() if not p.requires_grad})
    merger = [n for n, p in m.named_parameters() if ".visual.merger." in n]
    root_params = sorted(n for n, u in owner.items() if u == "<root>")
    return {"mode": "fsdp", "model": a.model, "accelerate_config": a.accelerate_config, "wrap_classes": present,
            "missing_classes": [c for c, k in present.items() if k == 0], "forbidden_modules": forbidden,
            "units": len(units), "mixed_dtype_units": mixed, "root_params": root_params,
            "frozen_params": frozen, "trainable_params": trainable, "trainable_dtypes": trainable_dtypes,
            "frozen_dtypes": frozen_dtypes, "merger_trainable": all(dict(m.named_parameters())[n].requires_grad for n in merger) and bool(merger),
            "keep_in_fp32_modules": getattr(m, "_keep_in_fp32_modules", None),
            "ok": not mixed and all(k > 0 for k in present.values()) and trainable_dtypes == ["torch.float32"]
                  and frozen_dtypes == ["torch.bfloat16"] and bool(merger)}


# ----------------------------------------------------------------------------------------------------------- P0 eval
def vllm_mrope(input_ids: List[int], video_grid: List[int], config):
    """vLLM 0.28's M-RoPE positions (3, L) of a prompt whose only medium is one video (the P0 evaluation's video
    prompts), from the static function its Qwen3.5 model inherits from Qwen3VLForConditionalGeneration.  The item's
    offset only starts _iter_mm_grid_hw's search for the frame blocks, so the prompt's first <|vision_start|> (the
    video's, outer marker or first frame) gives what vLLM's placeholder offset gives."""
    import torch
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    feats = [SimpleNamespace(modality="video", mm_position=SimpleNamespace(offset=input_ids.index(config.vision_start_token_id)),
                             data={"video_grid_thw": SimpleNamespace(data=torch.tensor(video_grid))})]
    pos, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(input_tokens=input_ids, mm_features=feats,
                                                                         config=config)
    return pos


def last_index(ids: List[int], tok_id: int) -> int:
    return len(ids) - 1 - ids[::-1].index(tok_id)


def mode_p0_eval(a) -> Dict[str, Any]:
    import torch
    from data_gen.interiorgs_qa import evaluate as ev

    pred_dir = Path(a.predictions).parent
    done = json.loads((pred_dir / "done.json").read_text()) if (pred_dir / "done.json").is_file() else {}
    meta = done.get("meta", {})
    selection = a.selection or done.get("selection")
    model_type = done.get("model_type") or meta.get("model_type") or "qwen3_5"
    fps = float(meta.get("video_fps", VIDEO_FPS))
    res = int(str(done.get("res") or meta.get("resolutions") or 256).split(",")[0])
    preds = {}
    for l in read_lines(Path(a.predictions)):
        p = json.loads(l)
        if p["media"] == "video":
            preds[(p["qid"], p["family"], p["rotation"], p["res"])] = p
    reqs = [r for r in ev.build_requests(ev.load_selection(selection), meta.get("families", ["orig", "h1"]), res)
            if r["media"] == "video" and (r["qid"], r["family"], r["rotation"], r["res"]) in preds]
    pick = sorted(random.Random(a.seed).sample(range(len(reqs)), min(a.n, len(reqs))))
    enc = LFEncoder(a.model, media_args={"video_fps": fps, **({} if a.lf_dir is None else lf_media_args(Path(a.lf_dir)))})
    proc, tok = enc.hf_processor, enc.tokenizer
    chat_kwargs = ev.MODEL_TYPE_CHAT_KWARGS.get(model_type, {})
    ve = tok.convert_tokens_to_ids("<|vision_end|>")
    m = meta_model(a.model)
    vllm_error = None
    results, t0 = [], time.time()
    for j in pick:
        r = reqs[j]
        p = preds[(r["qid"], r["family"], r["rotation"], r["res"])]
        # the evaluation's prompt (evaluate.run_vllm)
        content = [{"type": "video"}, {"type": "text", "text": r["text"]}]
        prompt = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                          add_generation_prompt=True, **chat_kwargs)
        frames, vmeta = ev.video_input(eval_video_array(r["files"]), model_type, fps)
        vmeta = dict(vmeta)
        do_sample = vmeta.pop("do_sample_frames", False)
        out = proc(text=[prompt], videos=[frames], video_metadata=[vmeta], do_sample_frames=do_sample, return_tensors="pt")
        e_ids = out["input_ids"][0].tolist()
        # the training layout of the same frames and question (export_lf.py, lf_data's strip rule), through LF; and
        # the same with the template's markers around the placeholder (lf_data --video-markers outer, if approved)
        def lf_prompt_of(placeholder: str):
            rec = {"messages": [{"role": "user", "content": (placeholder + "\n" + r["text"]).strip()},
                                {"role": "assistant", "content": str(r["answer"]).strip()}],
                   "videos": [list(r["files"])]}
            ids, labels = enc.encode(rec)
            return rec, ids[: next(i for i, x in enumerate(labels) if x != IGNORE)]

        rec, lf_prompt = lf_prompt_of("<video>")
        _, lf_wrapped = lf_prompt_of("<|vision_start|><video><|vision_end|>")
        le, ee, we = last_index(lf_prompt, ve), last_index(e_ids, ve), last_index(lf_wrapped, ve)
        row: Dict[str, Any] = {
            "qid": r["qid"], "family": r["family"], "rotation": r["rotation"], "frames": len(r["files"]),
            "prompt_tokens_vllm": p["prompt_tokens"], "prompt_tokens_processor": len(e_ids),
            "prompt_tokens_lf_layout": len(lf_prompt),
            "prompt_tokens_equal_vllm": p["prompt_tokens"] == len(e_ids),
            "video_part_identical": lf_prompt[: le + 1] == e_ids[: ee + 1],
            "video_part_identical_with_outer_markers": lf_wrapped[: we + 1] == e_ids[: ee + 1],
            "video_part_tokens": [le + 1, ee + 1],
            "outer_video_markers": [outer_video_markers(tok.decode(lf_prompt)), outer_video_markers(tok.decode(e_ids))],
            "tail_lf": tok.decode(lf_prompt[le + 1: le + 4]), "tail_eval": tok.decode(e_ids[ee + 1: ee + 4]),
            "tail_lf_is_newline_then_eval": lf_prompt[le + 1:] == tok.encode("\n", add_special_tokens=False) + e_ids[ee + 1:],
        }
        if row["video_part_identical"] is False:
            d = next((i for i, (x, y) in enumerate(zip(lf_prompt, e_ids)) if x != y), min(len(lf_prompt), len(e_ids)))
            row["first_difference"] = d
            row["context"] = {"lf": tok.decode(lf_prompt[max(0, d - 6): d + 6]), "eval": tok.decode(e_ids[max(0, d - 6): d + 6])}
        ref, _ = m.model.get_rope_index(input_ids=out["input_ids"], mm_token_type_ids=out["mm_token_type_ids"],
                                        video_grid_thw=out["video_grid_thw"], attention_mask=out["attention_mask"])
        try:
            vp = vllm_mrope(e_ids, out["video_grid_thw"][0].tolist(), m.config)
            row["mrope_vllm_identical"] = bool(torch.equal(vp.long(), ref[:, 0, :].long()))
        except Exception as exc:  # vLLM not importable on this host: reported, not a pass
            vllm_error = f"{type(exc).__name__}: {exc}"[:300]
            row["mrope_vllm_identical"] = None
        lf_mm = enc.mm_inputs(rec)
        row["video_pixels_identical"] = _same(out["pixel_values_videos"], lf_mm.get("pixel_values_videos")) \
            and _same(out["video_grid_thw"], lf_mm.get("video_grid_thw"))
        results.append(row)
    keys = ("prompt_tokens_equal_vllm", "video_part_identical", "video_part_identical_with_outer_markers",
            "tail_lf_is_newline_then_eval", "mrope_vllm_identical", "video_pixels_identical")
    tally = {k: [sum(1 for x in results if x.get(k) is True), sum(1 for x in results if x.get(k) is not None)] for k in keys}
    return {"mode": "p0_eval", "predictions": a.predictions, "selection": selection, "model": a.model,
            "model_type": model_type, "chat_template_kwargs": chat_kwargs, "video_fps": fps, "res": res,
            "video_requests": len(reqs), "checked": len(results), "tally_pass_of_checked": tally,
            "vllm_import_error": vllm_error, "seconds": round(time.time() - t0, 1), "results": results,
            # the pass conditions: what the model sees at test time is what the processor builds, and the video
            # (tokens, positions, pixels) is LF's; the layout differences are reported in the results
            "ok": bool(results) and all(x["prompt_tokens_equal_vllm"] and x["mrope_vllm_identical"] is not False
                                        and x["video_pixels_identical"] for x in results)}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--lf-dir", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--dataset", default="trajectory_train")
    t.add_argument("--n", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--pixels", type=int, default=20)
    t.add_argument("--positions", type=int, default=20)
    t.add_argument("--media", default="any", choices=("any", "video", "image", "text"),
                   help="sample only records with a video / with images and no video / with neither")
    t.add_argument("--no-prefix", action="store_true")
    t.add_argument("--out", required=True)
    l = sub.add_parser("lengths")
    l.add_argument("--lf-dir", required=True)
    l.add_argument("--model", required=True)
    l.add_argument("--cutoff", type=int, default=CUTOFF)
    l.add_argument("--limit", type=int, default=0)
    l.add_argument("--out", required=True)
    p = sub.add_parser("pipeline")
    p.add_argument("--lf-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--scratch", required=True, help="LF's output_dir for get_train_args (nothing is trained)")
    p.add_argument("--out", required=True)
    f = sub.add_parser("fsdp")
    f.add_argument("--model", required=True)
    f.add_argument("--accelerate-config", required=True)
    f.add_argument("--out", required=True)
    e = sub.add_parser("p0_eval")
    e.add_argument("--predictions", required=True, help="predictions.jsonl of data_gen.interiorgs_qa.evaluate (done.json beside it)")
    e.add_argument("--selection", default=None, help="default: done.json's selection")
    e.add_argument("--model", required=True)
    e.add_argument("--lf-dir", default=None, help="a folder whose train.yaml gives LF's media arguments")
    e.add_argument("--n", type=int, default=32)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = {"tokens": mode_tokens, "lengths": mode_lengths, "pipeline": mode_pipeline, "fsdp": mode_fsdp,
              "p0_eval": mode_p0_eval}[a.mode](a)
    report["env"] = {"DISABLE_VERSION_CHECK": os.environ.get("DISABLE_VERSION_CHECK")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    brief = {k: v for k, v in report.items() if k not in ("results", "failures", "root_params")}
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=str)[:6000])
    print("ok", report["ok"])


if __name__ == "__main__":
    main(sys.argv[1:])
