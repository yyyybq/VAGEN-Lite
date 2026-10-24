"""The registered Cambrian-S template and image plugin against the RL path (the user's step 2, sections 2.3 and 3.1 of
2026-10-24), CPU only:

1. LLaMA-Factory (bundled ca75f1e + model_matrix.sft.lf_cambrian) encodes one SFT conversation as the SFT job does
   (SupervisedDatasetProcessor, mask_history false); the reference is the same conversation through Cambrian-S's own
   chat template with one compact <image> per frame (CambrianProcessorWrapper, as the RL agent loop and the evaluator
   build it), expanded as the RL trainer expands it (vagen.agent_loop.cambrian_concat: 756 <|image_pad|> per frame).
   The token ids must be the same, one by one.
2. Only the assistant turns are supervised: the label tokens decode to every assistant answer followed by
   "<|im_end|>\\n", and nothing else.
3. The frames: 756 <|image_pad|> per frame, one run per frame; the plugin's pixel values equal the RL processor's
   (CambrianProcessorWrapper.preprocess_images) for the same frames.

    python -m model_matrix.sft.check_lf_cambrian --model CAMBRIAN --records SFT.jsonl --record-id ID [...] --out R.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from model_matrix.sft.check_lf_qwen3vl import IGNORE, load_records, spans


def lf_encode(model: str, rec: Dict[str, Any], cutoff_len: int = 16384):
    from model_matrix.sft.lf_cambrian import TEMPLATE, register

    register()
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    tm = load_tokenizer(ModelArguments(model_name_or_path=model))
    data_args = DataArguments(template=TEMPLATE, cutoff_len=cutoff_len, mask_history=False)
    template = get_template_and_fix_tokenizer(tm["tokenizer"], data_args)
    proc = SupervisedDatasetProcessor(template=template, tokenizer=tm["tokenizer"], processor=tm["processor"],
                                      data_args=data_args)
    conv = rec["conversations"]
    system = conv[0]["content"] if conv[0]["role"] == "system" else None
    turns = [{"role": m["role"], "content": m["content"]} for m in conv if m["role"] != "system"]
    images = list(rec["conversation_image_paths"])
    input_ids, labels = proc._encode_data_example(turns[:-1], turns[-1:], system, None, images, [], [])
    mm = template.mm_plugin.get_mm_inputs(images, [], [], [len(images)], [0], [0], [input_ids], tm["processor"])
    return input_ids, labels, tm["tokenizer"], tm["processor"], mm


def rl_encode(model: str, rec: Dict[str, Any]):
    """The conversation as the RL path builds it: compact ids from the model's chat template, then the trainer's
    expansion; and the RL processor's pixel values."""
    from PIL import Image
    from transformers import AutoTokenizer

    from vagen.agent_loop.cambrian_concat import IMAGE_PAD_ID, IMAGE_TOKEN_ID, TOKENS_PER_IMAGE, _expand
    from vagen.models.cambrian_processor import CambrianProcessorWrapper

    wrap = CambrianProcessorWrapper(AutoTokenizer.from_pretrained(model))
    messages, images = [], []
    paths = iter(rec["conversation_image_paths"])
    for m in rec["conversations"]:
        if m["role"] != "user":
            messages.append({"role": m["role"], "content": m["content"]})
            continue
        content = []
        for seg in re.split(r"(<image>)", m["content"]):
            if seg == "<image>":
                content.append({"type": "image"})
                images.append(Image.open(next(paths)).convert("RGB"))
            elif seg:
                content.append({"type": "text", "text": seg})
        messages.append({"role": "user", "content": content})
    text = wrap.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    compact = wrap(text=[text], return_tensors=None)["input_ids"][0]
    expanded, _, _ = _expand(compact, IMAGE_TOKEN_ID, IMAGE_PAD_ID, TOKENS_PER_IMAGE)
    return expanded, wrap.preprocess_images(images)["pixel_values"], text


def check(model: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    from model_matrix.sft.lf_cambrian import IMAGE_PAD_ID, TOKENS_PER_IMAGE

    ids, labels, tok, processor, mm = lf_encode(model, rec)
    rl_ids, rl_pixels, rl_text = rl_encode(model, rec)
    runs = [len(m) for m in re.findall(r"(?:P)+", "".join("P" if t == IMAGE_PAD_ID else "." for t in ids))]
    frames = len(rec["conversation_image_paths"])
    supervised = [t for t, lab in zip(ids, labels) if lab != IGNORE]
    answers = [m["content"] for m in rec["conversations"] if m["role"] == "assistant"]
    expected = "".join(a + "<|im_end|>\n" for a in answers)
    first_diff = next((i for i, (a, b) in enumerate(zip(ids, rl_ids)) if a != b), None)
    if first_diff is None and len(ids) != len(rl_ids):
        first_diff = min(len(ids), len(rl_ids))
    pixels = mm.get("pixel_values")
    result = {
        "record": rec["id"], "assistant_turns": len(answers), "frames": frames,
        "tokens_lf": len(ids), "tokens_rl": len(rl_ids), "ids_identical": ids == rl_ids, "first_difference": first_diff,
        "image_tokens_per_frame_lf": runs, "image_counts_ok": runs == [TOKENS_PER_IMAGE] * frames,
        "pixel_values_shape": list(pixels.shape) if pixels is not None else None,
        "pixel_values_equal_rl": bool(pixels is not None and pixels.shape == rl_pixels.shape
                                      and bool((pixels == rl_pixels).all())),
        "supervised_tokens": len(supervised), "supervised_text_is_answers": tok.decode(supervised) == expected,
        "supervised_spans": [{"from": s, "to": e, "text": tok.decode(ids[s:e])} for s, e in spans(labels)],
        "masked_tokens": sum(1 for lab in labels if lab == IGNORE),
        "labels_equal_ids_where_supervised": all(lab == t for t, lab in zip(ids, labels) if lab != IGNORE),
        "think_in_template": "<think>" in rl_text,
        "processor": type(processor).__name__,
    }
    if first_diff is not None:
        result["context"] = {"lf": tok.decode(ids[max(0, first_diff - 20):first_diff + 20]),
                             "rl": tok.decode(rl_ids[max(0, first_diff - 20):first_diff + 20])}
    result["ok"] = bool(result["ids_identical"] and result["image_counts_ok"] and result["pixel_values_equal_rl"]
                        and result["supervised_text_is_answers"] and result["labels_equal_ids_where_supervised"]
                        and not result["think_in_template"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", action="append", required=True, help="SFT conversations (jsonl); repeatable")
    ap.add_argument("--record-id", action="append", required=True, help="one per --records, in order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.records) != len(a.record_id):
        raise SystemExit("give one --record-id per --records")
    results = [check(a.model, load_records(p, [i])[0]) for p, i in zip(a.records, a.record_id)]
    report = {"model": a.model, "results": results, "ok": all(r["ok"] for r in results)}
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for r in results:
        print(json.dumps({k: r[k] for k in ("record", "ok", "ids_identical", "tokens_lf", "frames", "pixel_values_equal_rl",
                                            "supervised_tokens", "supervised_text_is_answers")}))
    print("ok", report["ok"])


if __name__ == "__main__":
    main()
