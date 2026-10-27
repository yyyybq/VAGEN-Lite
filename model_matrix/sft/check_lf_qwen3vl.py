"""The registered Qwen3-VL template against Qwen3-VL's own chat template (the user's step 2.3 of 2026-10-24), CPU only:

1. LLaMA-Factory (bundled ca75f1e + model_matrix.sft.lf_qwen3vl) encodes one SFT conversation as the SFT job does
   (SupervisedDatasetProcessor, mask_history false), and Qwen3-VL's processor encodes the same conversation with its
   own chat template; the token ids must be the same, one by one.
2. Only the assistant turns are supervised: the label tokens decode to every assistant answer followed by
   "<|im_end|>\\n", and nothing else.
3. The image placeholders: <|image_pad|> tokens per image equal the processor's grid (t * h * w / merge_size**2).

    .venv-qwen3vl-runtime/bin/python -m model_matrix.sft.check_lf_qwen3vl --model QWEN3VL --records SFT.jsonl \
        --record-id ID [--record-id ID ...] --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

IGNORE = -100


def load_records(path: str, ids: List[str]) -> List[Dict[str, Any]]:
    want, found = set(ids), {}
    for line in open(path):
        rec = json.loads(line)
        if rec["id"] in want:
            found[rec["id"]] = rec
    missing = want - set(found)
    if missing:
        raise SystemExit(f"not in {path}: {sorted(missing)}")
    return [found[i] for i in ids]


def lf_encode(model: str, rec: Dict[str, Any], cutoff_len: int = 16384):
    """input_ids and labels as the SFT job builds them (prepare_active_spatial_sft.py writes the conversation as
    sharegpt messages with the images in order; the system turn is LLaMA-Factory's system)."""
    from model_matrix.sft.lf_qwen3vl import TEMPLATE, register

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
    return input_ids, labels, tm["tokenizer"]


def hf_encode(model: str, rec: Dict[str, Any]):
    """The same conversation through Qwen3-VL's processor and chat template (images where "<image>" stands, as the
    RL agent loop and the evaluator place them)."""
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model)
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    out = processor(text=[text], images=images, return_tensors="pt")
    return out["input_ids"][0].tolist(), out["image_grid_thw"].tolist(), processor, text


def spans(labels: List[int]):
    """[start, end) runs of supervised positions."""
    runs, start = [], None
    for i, v in enumerate(labels + [IGNORE]):
        if v != IGNORE and start is None:
            start = i
        elif v == IGNORE and start is not None:
            runs.append([start, i])
            start = None
    return runs


def check(model: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    ids, labels, tok = lf_encode(model, rec)
    hf_ids, grids, processor, hf_text = hf_encode(model, rec)
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    merge = processor.image_processor.merge_size
    want_per_image = [t * h * w // (merge * merge) for t, h, w in grids]
    # LF's image runs, image by image
    lf_runs = [len(m) for m in re.findall(r"(?:P)+", "".join("P" if t == pad else "." for t in ids))]
    supervised = [t for t, lab in zip(ids, labels) if lab != IGNORE]
    answers = [m["content"] for m in rec["conversations"] if m["role"] == "assistant"]
    expected = "".join(a + "<|im_end|>\n" for a in answers)
    first_diff = next((i for i, (a, b) in enumerate(zip(ids, hf_ids)) if a != b), None)
    if first_diff is None and len(ids) != len(hf_ids):
        first_diff = min(len(ids), len(hf_ids))
    mask_view = []
    for s, e in spans(labels):
        mask_view.append({"from": s, "to": e, "text": tok.decode(ids[s:e])})
    result = {
        "record": rec["id"], "assistant_turns": len(answers), "images": len(grids),
        "tokens_lf": len(ids), "tokens_hf": len(hf_ids), "ids_identical": ids == hf_ids, "first_difference": first_diff,
        "image_tokens_per_image_lf": lf_runs, "image_tokens_per_image_processor": want_per_image,
        "image_tokens_hf": sum(1 for t in hf_ids if t == pad), "image_counts_ok": lf_runs == want_per_image,
        "supervised_tokens": len(supervised), "supervised_text_is_answers": tok.decode(supervised) == expected,
        "supervised_spans": mask_view,
        "masked_tokens": sum(1 for lab in labels if lab == IGNORE),
        "labels_equal_ids_where_supervised": all(lab == t for t, lab in zip(ids, labels) if lab != IGNORE),
        "think_in_template": "<think>" in hf_text,
    }
    if first_diff is not None:
        result["context"] = {"lf": tok.decode(ids[max(0, first_diff - 20):first_diff + 20]),
                             "hf": tok.decode(hf_ids[max(0, first_diff - 20):first_diff + 20])}
    result["ok"] = bool(result["ids_identical"] and result["image_counts_ok"] and result["supervised_text_is_answers"]
                        and result["labels_equal_ids_where_supervised"] and not result["think_in_template"])
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
        print(json.dumps({k: r[k] for k in ("record", "ok", "ids_identical", "tokens_lf", "image_tokens_per_image_lf",
                                            "supervised_tokens", "supervised_text_is_answers")}))
    print("ok", report["ok"])


if __name__ == "__main__":
    main()
