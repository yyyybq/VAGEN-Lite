"""The registered Qwen3-VL plugin's video layout against Qwen3-VL's own processor (the user's decision of 2026-11-01,
P0's video conversations), CPU only.  For LLaMA-Factory sharegpt records (messages, images, videos: lists of frame
paths, as P0's lf_256/qa_train.jsonl):

1. LLaMA-Factory (bundled ca75f1e + model_matrix.sft.lf_qwen3vl) encodes the conversation as the SFT job does
   (SupervisedDatasetProcessor), and Qwen3-VL's processor encodes it with its own chat template, the same frames and
   the metadata the plugin gives them (fps = LLaMA-Factory's video_fps, frame indices 0..n-1); the token ids must be
   the same, one by one (so the "<t seconds>" texts and the vision blocks too).
2. The pixels and grids the plugin hands the model (pixel_values_videos, video_grid_thw; images likewise) equal the
   processor's.
3. Only the assistant turns are supervised (all of them: mask_history false, as the SFT job).

    .venv-qwen3vl-runtime/bin/python -m model_matrix.sft.check_lf_qwen3vl_video --model QWEN3VL --records LF.jsonl \
        --line N [--line N ...] [--video-fps 2.0] --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

IGNORE = -100


def lf_side(model: str, rec: Dict[str, Any], video_fps: float):
    from model_matrix.sft.lf_qwen3vl import TEMPLATE, register

    register()
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    tm = load_tokenizer(ModelArguments(model_name_or_path=model, video_fps=video_fps))
    data_args = DataArguments(template=TEMPLATE, cutoff_len=16384, mask_history=False)
    template = get_template_and_fix_tokenizer(tm["tokenizer"], data_args)
    proc = SupervisedDatasetProcessor(template=template, tokenizer=tm["tokenizer"], processor=tm["processor"],
                                      data_args=data_args)
    msgs = rec["messages"]
    system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
    turns = [m for m in msgs if m["role"] != "system"]
    images, videos = list(rec.get("images") or []), list(rec.get("videos") or [])
    ids, labels = proc._encode_data_example(turns[:-1], turns[-1:], system, None, images, videos, [])
    mm = template.mm_plugin._get_mm_inputs(images, videos, [], tm["processor"])
    return ids, labels, mm, tm["tokenizer"], tm["processor"]


def hf_side(processor, rec: Dict[str, Any], video_fps: float):
    from PIL import Image

    images = iter(rec.get("images") or [])
    videos = iter(rec.get("videos") or [])
    messages, pil_images, pil_videos, metadata = [], [], [], []
    for m in rec["messages"]:
        if m["role"] != "user":
            messages.append({"role": m["role"], "content": m["content"]})
            continue
        content = []
        for seg in re.split(r"(<image>|<video>)", m["content"]):
            if seg == "<image>":
                content.append({"type": "image"})
                pil_images.append(Image.open(next(images)).convert("RGB"))
            elif seg == "<video>":
                content.append({"type": "video"})
                frames = [Image.open(f).convert("RGB") for f in next(videos)]
                n = len(frames)
                if n % 2:
                    frames.append(frames[-1])
                pil_videos.append(frames)
                metadata.append({"fps": video_fps, "frames_indices": list(range(n)) + [n - 1] * (len(frames) - n),
                                 "total_num_frames": len(frames)})
            elif seg:
                content.append({"type": "text", "text": seg})
        messages.append({"role": "user", "content": content})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    kwargs = {"images": pil_images or None, "videos": pil_videos or None, "return_tensors": "pt"}
    if pil_videos:
        kwargs["videos_kwargs"] = {"video_metadata": metadata, "do_sample_frames": False}
    out = processor(text=[text], **kwargs)
    return out, text


def same(a, b) -> bool:
    import torch

    if a is None or b is None:
        return a is None and b is None
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return a.shape == b.shape and bool(torch.allclose(a.float(), b.float(), atol=1e-5))


def check(model: str, rec: Dict[str, Any], line: int, video_fps: float) -> Dict[str, Any]:
    ids, labels, mm, tok, processor = lf_side(model, rec, video_fps)
    out, hf_text = hf_side(processor, rec, video_fps)
    hf_ids = out["input_ids"][0].tolist()
    first_diff = next((i for i, (a, b) in enumerate(zip(ids, hf_ids)) if a != b), None)
    if first_diff is None and len(ids) != len(hf_ids):
        first_diff = min(len(ids), len(hf_ids))
    supervised = [t for t, lab in zip(ids, labels) if lab != IGNORE]
    answers = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]  # every turn (mask_history false)
    decoded = tok.decode(ids, skip_special_tokens=False)
    for t in ("<|image_pad|>", "<|video_pad|>"):
        while t * 2 in decoded:
            decoded = decoded.replace(t * 2, t)
    result = {
        "line": line, "images": len(rec.get("images") or []), "videos": len(rec.get("videos") or []),
        "frames": [len(v) for v in rec.get("videos") or []],
        "tokens_lf": len(ids), "tokens_hf": len(hf_ids), "ids_identical": ids == hf_ids, "first_difference": first_diff,
        "vision_blocks": sum(1 for t in ids if t == tok.convert_tokens_to_ids("<|vision_start|>")),
        "seconds_texts": re.findall(r"<[0-9.]+ seconds>", decoded),
        "video_grid_lf": mm.get("video_grid_thw").tolist() if mm.get("video_grid_thw") is not None else None,
        "video_grid_hf": out.get("video_grid_thw").tolist() if out.get("video_grid_thw") is not None else None,
        "video_pixels_equal": same(mm.get("pixel_values_videos"), out.get("pixel_values_videos")),
        "image_pixels_equal": same(mm.get("pixel_values"), out.get("pixel_values")),
        "assistant_turns": len(answers),
        "supervised_is_answer": tok.decode(supervised) == "".join(a + "<|im_end|>\n" for a in answers),
        "decoded_lf": decoded[:900],
    }
    if first_diff is not None:
        result["context"] = {"lf": tok.decode(ids[max(0, first_diff - 20):first_diff + 20]),
                             "hf": tok.decode(hf_ids[max(0, first_diff - 20):first_diff + 20])}
    result["ok"] = bool(result["ids_identical"] and result["video_grid_lf"] == result["video_grid_hf"]
                        and result["video_pixels_equal"] and result["image_pixels_equal"]
                        and result["supervised_is_answer"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", required=True, help="LLaMA-Factory sharegpt jsonl (messages, images, videos)")
    ap.add_argument("--line", type=int, action="append", required=True, help="0-based line number; repeatable")
    ap.add_argument("--video-fps", type=float, default=2.0, help="LLaMA-Factory's video_fps (its default 2.0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = set(a.line)
    recs = {i: json.loads(l) for i, l in enumerate(open(a.records)) if i in want}
    results = [check(a.model, recs[i], i, a.video_fps) for i in a.line]
    report = {"model": a.model, "records": a.records, "video_fps": a.video_fps, "results": results,
              "ok": all(r["ok"] for r in results)}
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for r in results:
        print(json.dumps({k: r[k] for k in ("line", "ok", "ids_identical", "tokens_lf", "vision_blocks", "frames",
                                            "video_pixels_equal", "image_pixels_equal", "supervised_is_answer")}))
    print("ok", report["ok"])


if __name__ == "__main__":
    main()
