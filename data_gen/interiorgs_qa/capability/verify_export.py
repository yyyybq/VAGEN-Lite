"""Verify a QA export using the real, CPU-only LLaMA-Factory SFT preprocessor.

This loads the tokenizer/image processor, never model weights. A passing result
checks input formatting and supervision, not visual truth or human approval.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import traceback


IGNORE_INDEX = -100
PRIVATE_MARKERS = (
    "PRIVATE_GEOMETRY_SENTINEL_NOT_FOR_MODEL", "camera_position", "c2w",
    "world_aabb", "answer_key", "evidence_zh", "occlusion_fraction",
    "depth_supported_blocker_coverage", "target_box_in_frame_fraction",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_records(records):
    """Validate the export boundary before importing any training dependencies."""
    require(bool(records), "The export is empty; zero samples cannot pass.")
    for i, row in enumerate(records):
        require(set(row) == {"messages", "images", "videos"}, f"row {i}: unexpected/private fields")
        messages = row["messages"]
        require(isinstance(messages, list) and len(messages) == 2, f"row {i}: expected one user/assistant pair")
        require(all(set(m) == {"role", "content"} for m in messages), f"row {i}: extra message fields")
        require([m["role"] for m in messages] == ["user", "assistant"], f"row {i}: incorrect roles")
        user, answer = (m["content"] for m in messages)
        require(isinstance(user, str) and re.fullmatch(r"[A-Z]", str(answer)), f"row {i}: answer must be one option letter")
        paths = row["images"]
        require(isinstance(paths, list) and 1 <= len(paths) <= 2, f"row {i}: expected one or two ordered images")
        require(all(isinstance(p, str) and Path(p).is_absolute() for p in paths), f"row {i}: image paths must be absolute")
        require(all("overlay" not in p and "inspection_" not in p for p in paths), f"row {i}: inspection images are forbidden")
        require(row["videos"] == [], f"row {i}: videos must be empty")
        require(user.count("<image>") == len(paths), f"row {i}: image placeholders do not match image count")
        require(user.startswith("<image>\n" * len(paths)), f"row {i}: image placeholders must precede the question in order")
        require(not any(re.search(r"\b" + re.escape(s) + r"\b", user) for s in PRIVATE_MARKERS), f"row {i}: private evidence marker in user text")
        choices = re.findall(r"^([A-Z])\.\s+.+$", user, flags=re.MULTILINE)
        require(2 <= len(choices) <= 4 and choices == list("ABCD"[:len(choices)]), f"row {i}: choices must be consecutive A-D (2 to 4 choices)")
        require(answer in choices, f"row {i}: answer does not reference an offered option")


def verify_public_projection(records, questions):
    """Compare with only public question/options/images and the assistant label."""
    require(len(records) == len(questions), "questions.jsonl/export length mismatch")
    for i, (record, q) in enumerate(zip(records, questions)):
        text = "<image>\n" * len(q["images"]) + q["question"] + "\n" + "\n".join(
            f"{chr(65 + j)}. {option}" for j, option in enumerate(q["options"]))
        expected = {"messages": [{"role": "user", "content": text},
                                 {"role": "assistant", "content": q["answer"]}],
                    "images": q["images"], "videos": []}
        require(record == expected, f"row {i}: differs from public question/options/image projection")


def verify_image_bindings(questions, image_evidence):
    """Bind each ordered RGB input to its original rendering receipt hash."""
    for i, question in enumerate(questions):
        expected = question.get("source", {}).get("image_sha256")
        actual = [image_evidence[path]["sha256"] for path in question["images"]]
        require(expected == actual, f"row {i}: ordered image hashes differ from rendering source")


def verify_encoded(example, original, tokenizer):
    """Inspect actual training labels, including all image and prompt tokens."""
    ids, labels = example["input_ids"], example["labels"]
    require(len(ids) == len(labels), "Input/label lengths differ")
    supervised = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
    require(bool(supervised), "No supervised answer tokens")
    require(supervised == list(range(supervised[0], len(ids))), "Supervision must be a contiguous answer suffix")
    require(all(labels[i] == ids[i] for i in supervised), "Supervised labels do not match input tokens")
    decoded = tokenizer.decode([labels[i] for i in supervised], skip_special_tokens=False)
    answer = original["messages"][1]["content"]
    require(decoded in {answer + "<|im_end|>", answer + "<|im_end|>\n"}, "Supervision contains more than answer plus template end")
    require(example["images"] == original["images"], "Image order changed in loader")
    require(not example.get("videos"), "Unexpected video input")
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    positions = [i for i, token in enumerate(ids) if token == image_pad]
    require(bool(positions), "Image tokens are absent")
    require(all(labels[i] == IGNORE_INDEX for i in positions), "An image token is supervised")
    full = tokenizer.decode(ids, skip_special_tokens=False)
    require(full.count("<|vision_start|>") == len(original["images"]), "Encoded image count changed")
    require(not any(re.search(r"\b" + re.escape(s) + r"\b", full) for s in PRIVATE_MARKERS), "Private evidence in encoded input")
    return {"total_input_tokens": len(ids), "ignored_label_tokens": labels.count(IGNORE_INDEX),
            "supervised_token_count": len(supervised), "supervised_positions": supervised,
            "supervised_token_ids": [labels[i] for i in supervised], "supervised_decoded": decoded,
            "visual_placeholder_tokens": len(positions), "visual_tokens_all_ignored": True,
            "user_system_and_image_tokens_all_ignored": True, "image_order_preserved_in_loader": True,
            "videos_empty": True}


def run(dataset_path, model_path, out):
    """Run the actual loader/collator once for every exported sample."""
    dataset_path, model_path, out = Path(dataset_path).resolve(), Path(model_path).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = {"ok": False, "dataset": str(dataset_path), "model_tokenizer_path": str(model_path),
            "model_weights_loaded": False, "gpu_used": False, "training_run": False,
            "predictions_generated": False, "visual_truth_checked": False, "human_approval_granted": False}
    try:
        result = _run(dataset_path, model_path, out)
        result = {**base, **result, "ok": True}
    except Exception as error:
        result = {**base, "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc()}
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        raise
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _run(dataset_path, model_path, out):
    records = read_rows(dataset_path)
    validate_records(records)
    dataset_sha = sha256(dataset_path)
    questions_path = dataset_path.with_name("questions.jsonl")
    questions_sha = None
    questions = None
    if questions_path.exists():
        questions = read_rows(questions_path)
        verify_public_projection(records, questions)
        questions_sha = sha256(questions_path)
    from PIL import Image
    image_evidence = {}
    for row in records:
        for path in row["images"]:
            if path not in image_evidence:
                with Image.open(path) as im:
                    im.load()
                    require(im.size == (256, 256), f"Image dimensions are not 256x256: {path}")
                    image_evidence[path] = {"sha256": sha256(path), "size": list(im.size), "mode": im.mode}
    if questions is not None:
        verify_image_bindings(questions, image_evidence)
    for key, value in {"CUDA_VISIBLE_DEVICES": "", "DISABLE_VERSION_CHECK": "1", "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2",
                       "HF_DATASETS_CACHE": str(out / "datasets_cache"), "HF_MODULES_CACHE": str(out / "modules_cache"),
                       "TMPDIR": str(out / "tmp")}.items():
        os.environ[key] = value
    (out / "tmp").mkdir()
    import copy
    import torch
    import transformers
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    require(not torch.cuda.is_initialized(), "CUDA was already initialized; use a fresh CPU process")
    def forbidden_model_load(*args, **kwargs):
        raise RuntimeError("Model weight loading is forbidden in export verification")
    previous_model_loader = transformers.PreTrainedModel.__dict__["from_pretrained"]
    transformers.PreTrainedModel.from_pretrained = forbidden_model_load
    try:
        (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        info = {"qa_verify": {"file_name": "records.jsonl", "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images", "videos": "videos"},
                "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}}
        (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
        cfg = dict(model_name_or_path=str(model_path), stage="sft", finetuning_type="full", template="qwen2_vl",
                   dataset_dir=str(out), dataset="qa_verify", output_dir=str(out / "unused_model_output"),
                   do_train=False, do_eval=False, do_predict=False, use_cpu=True, bf16=False, fp16=False,
                   report_to="none", overwrite_cache=True, preprocessing_num_workers=None, preprocessing_batch_size=2,
                   train_on_prompt=False, mask_history=True, packing=False, cutoff_len=16384,
                   image_max_pixels=65536, video_max_pixels=65536)
        (out / "preprocess_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        model_args, data_args, training_args, _, _ = get_train_args(cfg)
        require(not training_args.do_train and str(training_args.device) == "cpu", "Expected CPU preprocessing only")
        tok = load_tokenizer(model_args)
        tokenizer, processor = tok["tokenizer"], tok["processor"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        dataset = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok)["train_dataset"]
        require(len(dataset) == len(records), "Training loader silently dropped or added records")
        collator = MultiModalDataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, template=template,
                    processor=processor, padding=True, label_pad_token_id=IGNORE_INDEX)
        checks = []
        with (out / "actual_preprocessed_tokens.jsonl").open("w") as tokens:
            for index, (example, original) in enumerate(zip(dataset, records)):
                check = verify_encoded(example, original, tokenizer)
                batch = collator([copy.deepcopy(example)])
                require(all(v.device.type == "cpu" for v in batch.values() if isinstance(v, torch.Tensor)), "Non-CPU collator output")
                require(batch["labels"].tolist()[0] == example["labels"], "Collator changed supervision")
                singles = [template.mm_plugin.get_mm_inputs([path], [], [], [1], [0], [0], [example["input_ids"]], processor) for path in original["images"]]
                require(torch.equal(batch["pixel_values"], torch.cat([x["pixel_values"] for x in singles], dim=0)), "Collator image patch order/content changed")
                require(torch.equal(batch["image_grid_thw"], torch.cat([x["image_grid_thw"] for x in singles], dim=0)), "Collator image grid order changed")
                check.update(record_index=index, images=original["images"], image_sha256=[image_evidence[p]["sha256"] for p in original["images"]],
                             image_patch_order_preserved_in_actual_collator=True, collator_cpu_only=True)
                checks.append(check)
                tokens.write(json.dumps({"record_index": index, "input_ids": example["input_ids"], "labels": example["labels"]}) + "\n")
        require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
        require(sha256(dataset_path) == dataset_sha, "Dataset changed while checking")
        if questions_sha:
            require(sha256(questions_path) == questions_sha, "Questions changed while checking")
        require(all(sha256(p) == e["sha256"] for p, e in image_evidence.items()), "An input image changed while checking")
        source_files = [Path(__file__), Path(inspect.getfile(get_dataset)), Path(inspect.getfile(MultiModalDataCollatorForSeq2Seq)),
                        Path(inspect.getfile(get_dataset)).parent / "processor/supervised.py"]
        return {"implementation": "actual LLaMA-Factory SFT get_dataset + MultiModalDataCollatorForSeq2Seq(model=None)",
                "dataset_sha256": dataset_sha, "questions_sha256": questions_sha,
                "public_projection_checked": questions_sha is not None, "private_field_marker_guard_passed": True,
                "render_source_image_sha256_checked": questions_sha is not None,
                "sample_count": len(records), "input_image_occurrences": sum(len(r["images"]) for r in records),
                "unique_image_paths": len(image_evidence), "unique_image_sha256": len({e["sha256"] for e in image_evidence.values()}),
                "total_input_tokens": sum(r["total_input_tokens"] for r in checks),
                "total_supervised_tokens": sum(r["supervised_token_count"] for r in checks),
                "train_on_prompt": False, "mask_history": True, "template": data_args.template,
                "records": checks, "images": image_evidence,
                "source_code_sha256": {str(p.resolve()): sha256(p) for p in source_files},
                "library_versions": {n: importlib.metadata.version(n) for n in ["transformers", "torch", "datasets", "llamafactory"]},
                "limitations": ["No model forward/backward, model-specific rotary positions, optimizer or training was run.",
                                "Private-field guards and public projection do not establish visual correctness or human approval."]}
    finally:
        transformers.PreTrainedModel.from_pretrained = previous_model_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Must be a new directory")
    args = parser.parse_args()
    result = run(args.dataset, args.model, args.out)
    print(json.dumps({"ok": result["ok"], "sample_count": result["sample_count"], "result": str(args.out / "result.json")}))


if __name__ == "__main__":
    main()
