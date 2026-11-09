#!/usr/bin/env python3
"""Read-only data-layer supervision audit using local tokenizer/processor.

No model class, weights, forward pass, optimizer or trainer is loaded. The
repository's real Qwen2-VL template and SupervisedDatasetProcessor produce
input_ids/labels. Training-entrypoint compatibility is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--model", required=True, help="Local checkpoint metadata/tokenizer/processor only")
    parser.add_argument("--repo-root", default="/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
    parser.add_argument("--output", help="New audit directory; default export/supervision_validation")
    parser.add_argument("--cutoff-len", type=int, default=16384)
    args = parser.parse_args()
    export, model = Path(args.export).resolve(strict=True), Path(args.model).resolve(strict=True)
    out = Path(args.output).resolve() if args.output else export / "supervision_validation"
    if out.exists():
        raise FileExistsError(f"Use a new audit directory: {out}")
    out.mkdir(parents=True)
    report = {
        "schema_version": "active_spatial_supervision_mask_audit_v1", "status": "running",
        "model_loaded": False, "model_forward": False, "backward": False, "training_run": False,
        "source_export": str(export), "local_model_metadata": str(model),
        "scope": "Actual data preprocessing and assistant-only labels; no model/trainer or data collator execution",
        "data_settings": {"template": "qwen2_vl", "train_on_prompt": False, "mask_history": False,
                          "cutoff_len": args.cutoff_len, "packing": False},
        "image_token_count": None, "records": 0, "datasets": {}, "training_entrypoint": {"status": "not_checked"},
    }
    try:
        # This process only uses CPU image preprocessing; never probes/allocates CUDA explicitly.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.dont_write_bytecode = True
        lf = Path(args.repo_root) / "third_party/LLaMA-Factory/src"
        sys.path.insert(0, str(lf))
        from transformers import AutoTokenizer, AutoProcessor
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

        config = json.loads((model / "config.json").read_text())
        if config.get("model_type") != "qwen2_5_vl":
            raise ValueError("This audit supports the pinned Qwen2.5-VL recipe only")
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False,
                                                  use_fast=True, padding_side="right")
        processor = AutoProcessor.from_pretrained(str(model), local_files_only=True, trust_remote_code=False, use_fast=True)
        processor.image_max_pixels = 768 * 768
        processor.image_min_pixels = 32 * 32
        settings = SimpleNamespace(template="qwen2_vl", cutoff_len=args.cutoff_len, train_on_prompt=False,
                                   mask_history=False, tool_format=None, default_system=None, enable_thinking=True)
        template = get_template_and_fix_tokenizer(tokenizer, settings)
        loader = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=processor, data_args=settings)
        if template.efficient_eos:
            raise ValueError("Unexpected Qwen2-VL template efficient_eos setting")
        report["runtime_versions"] = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "torch")}
        report["source_hashes"] = {str(path): sha(path) for path in [
            lf / "llamafactory/data/template.py", lf / "llamafactory/data/processor/supervised.py",
            lf / "llamafactory/data/mm_plugin.py", model / "tokenizer.json", model / "preprocessor_config.json"]}
        report["processor"] = {"class": type(processor).__name__, "image_processor": type(processor.image_processor).__name__,
                               "image_max_pixels": getattr(processor, "image_max_pixels", 768 * 768),
                               "image_min_pixels": getattr(processor, "image_min_pixels", 32 * 32),
                               "merge_size": processor.image_processor.merge_size}
        info = json.loads((export / "dataset_info.json").read_text())
        totals = Counter()
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        with (out / "tokenized_records.jsonl").open("x") as audit_stream:
            for dataset, entry in info.items():
                counts = Counter(examples=0, input_tokens=0, image_pad_tokens=0, assistant_turns=0,
                                 assistant_content_tokens=0, effective_supervised_tokens=0,
                                 prompt_or_image_masked_tokens=0, qa_content_one_token_examples=0)
                source = export / entry["file_name"]
                for index, row in enumerate(read_rows(source)):
                    if set(row) != {"messages", "images"}:
                        raise ValueError("Unexpected model input metadata")
                    messages = row["messages"]
                    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
                    turns = messages[1:] if system is not None else messages
                    if not turns or len(turns) % 2 or any(turn["role"] != ("user" if j % 2 == 0 else "assistant") for j, turn in enumerate(turns)):
                        raise ValueError("Invalid conversation order")
                    inputs, labels = loader._encode_data_example(
                        prompt=turns[:-1], response=turns[-1:], system=system, tools=None,
                        images=row["images"], videos=[], audios=[])
                    # Independently assemble expected masking from the template pairs.
                    expanded = template.mm_plugin.process_messages(turns, row["images"], [], [], processor)
                    pairs = template.encode_multiturn(tokenizer, expanded, system, None)
                    expected_inputs, expected_labels, spans = [], [], []
                    for turn_index, (source_ids, target_ids) in enumerate(pairs):
                        start = len(expected_inputs) + len(source_ids)
                        content = turns[2 * turn_index + 1]["content"]
                        content_ids = tokenizer.encode(content, add_special_tokens=False)
                        if target_ids[:len(content_ids)] != content_ids:
                            raise AssertionError("Assistant content token IDs differ from target prefix")
                        expected_inputs += source_ids + target_ids
                        expected_labels += [-100] * len(source_ids) + target_ids
                        spans.append({"assistant_turn": turn_index, "content": content,
                                      "content_token_ids": content_ids,
                                      "content_label_positions_zero_based": list(range(start, start + len(content_ids))),
                                      "content_causal_logit_positions_zero_based": list(range(start - 1, start + len(content_ids) - 1)),
                                      "full_target_positions_zero_based": list(range(start, start + len(target_ids))),
                                      "full_target_token_ids": target_ids,
                                      "template_closure_token_ids": target_ids[len(content_ids):],
                                      "template_closure_text": tokenizer.decode(target_ids[len(content_ids):], skip_special_tokens=False)})
                    if inputs != expected_inputs or labels != expected_labels:
                        raise AssertionError("Truncation or assistant-mask mismatch; sample not approved for this recipe")
                    if len(inputs) != len(labels) or not any(label != -100 for label in labels):
                        raise AssertionError("Missing supervision")
                    if any(label != -100 for token, label in zip(inputs, labels) if token == image_pad_id):
                        raise AssertionError("Image token is supervised")
                    is_qa = dataset.startswith("qa_") or dataset.startswith("candidate_qa_")
                    if is_qa and (len(spans) != 1 or spans[0]["content"] not in {"Yes", "No"}):
                        raise AssertionError("QA record is not a single Yes/No target")
                    supervised_positions = [j for j, label in enumerate(labels) if label != -100]
                    record = {"dataset": dataset, "record_index": index,
                              "source_record_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                              "input_ids": inputs, "labels": labels,
                              "loss_mask": [int(label != -100) for label in labels],
                              "supervised_label_positions_zero_based": supervised_positions,
                              "assistant_spans": spans, "image_pad_tokens": inputs.count(image_pad_id),
                              "prompt_masked": True, "image_tokens_masked": True, "truncated": False}
                    audit_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    counts.update(examples=1, input_tokens=len(inputs), image_pad_tokens=inputs.count(image_pad_id),
                                  assistant_turns=len(spans), assistant_content_tokens=sum(len(span["content_token_ids"]) for span in spans),
                                  effective_supervised_tokens=len(supervised_positions), prompt_or_image_masked_tokens=labels.count(-100),
                                  qa_content_one_token_examples=int(is_qa and len(spans[0]["content_token_ids"]) == 1))
                report["datasets"][dataset] = {**counts, "source_sha256": sha(source)}
                totals.update(counts)
        # Import compatibility is a separate check. Do not disable its version guard.
        try:
            from llamafactory.hparams import DataArguments  # noqa: F401
            report["training_entrypoint"] = {"status": "hparams_import_passed", "training_not_run": True}
        except Exception as exc:
            report["training_entrypoint"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}",
                                              "note": "Data modules were audited directly; this does not certify trainer compatibility"}
        report.update(status="PASS", records=totals["examples"], image_token_count=totals["image_pad_tokens"], totals=dict(totals),
                      mask_positions="Actual expanded-image token sequence, zero-based; causal logits predict label position i from i-1",
                      label_scope="Assistant text plus the exact template closure, usually <|im_end|> and newline; all source and image tokens use -100",
                      no_truncation=True, assistant_only_mask_verified=True)
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", image_token_count=None)
        save(out / "summary.json", report)
        print(json.dumps({"output": str(out), "status": report["status"], "error": report["error"]}, ensure_ascii=False))
        raise
    save(out / "summary.json", report)
    print(json.dumps({"output": str(out), "status": report["status"], "totals": report["totals"],
                      "training_entrypoint": report["training_entrypoint"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
