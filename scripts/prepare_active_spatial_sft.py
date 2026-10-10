#!/usr/bin/env python3
"""Prepare scene-disjoint trajectory/QA datasets and a LLaMA-Factory recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_gen.active_spatial_sft.convert_to_qwen25vl_sft import (
    _convert_record,
    _validate_record,
)


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def file_hash(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_messages(record):
    messages, images = record["messages"], record["images"]
    turns = messages[1:] if messages and messages[0]["role"] == "system" else messages
    if not turns or len(turns) % 2:
        raise ValueError("Expected complete user/assistant pairs")
    for i, turn in enumerate(turns):
        if turn["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Invalid conversation alternation")
        if not isinstance(turn["content"], str) or not turn["content"].strip():
            raise ValueError("Empty or non-text message")
    if sum(t["content"].count("<image>") for t in messages) != len(images):
        raise ValueError("Image placeholder count differs from image count")
    for path in images:
        with Image.open(path) as img:
            img.verify()


def qa_record(row, base, *, expected_split="train"):
    """Return observable single-view QA only; private fields never enter prompts."""
    if row.get("split") != expected_split:
        raise ValueError(
            "QA input must be a training bank; do not repurpose held-out labels"
        )
    if row.get("task_type") == "screen_occupancy":
        return None, "screen_occupancy_unverified"
    obs = row.get("public_observation", {})
    if row.get("qa_sft_allowed") is not True:
        return None, "qa_sft_disallowed"
    if row.get("label_validity") != "valid" or row.get("private_answer") not in {
        "Yes",
        "No",
    }:
        return None, "invalid_label"
    if row.get("observability_validity") != "valid" or not obs.get("image_path"):
        return None, "unobservable"
    if (
        row.get("task_type") == "delta_control"
        or obs.get("history")
        or obs.get("observation_config") != "single_image"
    ):
        return None, "history_not_supported"
    image = Path(obs["image_path"])
    if not image.is_absolute():
        image = base / image
    question = row.get("question")
    if not isinstance(question, str) or not question.strip() or "<image>" in question:
        raise ValueError("QA question must be nonempty text without image placeholders")
    result = {
        "messages": [
            {"role": "user", "content": "<image>\n" + question},
            {"role": "assistant", "content": row["private_answer"]},
        ],
        "images": [str(image.resolve(strict=True))],
    }
    validate_messages(result)
    return result, None


def prepare(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    if not 0 <= args.qa_probability <= 1 or not 0 < args.val_fraction < 1:
        raise ValueError("Invalid QA probability or validation fraction")
    model = Path(args.model).resolve(strict=True)
    model_type = json.loads((model / "config.json").read_text())["model_type"]
    # The bundled LLaMA-Factory has qwen2_vl but no qwen3_vl template.
    if model_type != "qwen2_5_vl":
        raise ValueError(
            "This bundled SFT recipe supports Qwen2.5-VL; Qwen3-VL needs a validated LLaMA-Factory upgrade"
        )
    from active_spatial_release_contract import scene_partitions, load_registry, bind_derived
    frozen = json.loads(Path(args.split_manifest).read_text()) if args.split_manifest else None
    strict = bool(frozen and frozen.get("version") == 2)
    lookup = scene_partitions(frozen) if frozen else {}
    registry = load_registry(getattr(args, "task_registry", None), frozen) if strict else {}
    samples, excluded, sources, seen, source_scenes = [], Counter(), [], set(), set()
    for kind, paths in (("trajectory", args.trajectory), ("qa", args.qa_bank)):
        for path_text in paths:
            path = Path(path_text).resolve(strict=True)
            sources.append({"kind": kind, "path": str(path), "sha256": file_hash(path)})
            for row in read_rows(path):
                scene = row.get("scene_id")
                sid = row.get("id" if kind == "trajectory" else "sample_id")
                if not scene or not sid:
                    raise ValueError(f"Missing scene or record ID in {path}")
                source_scenes.add(str(scene))
                identity = (kind, str(sid))
                if identity in seen:
                    raise ValueError(f"Duplicate source record: {identity}")
                seen.add(identity)
                parent = bind_derived(row, kind, registry, lookup) if strict else None
                if kind == "qa":
                    if strict and row.get("semantic_review", {}).get("status") != "PASS":
                        excluded["semantic_review_missing"] += 1
                        continue
                    record, reason = qa_record(row, path.parent, expected_split=parent["split"] if strict else "train")
                    if reason:
                        excluded[reason] += 1
                        continue
                else:
                    if not strict and row.get("split", "train") != "train":
                        raise ValueError("Trajectory input includes held-out records")
                    if row.get("success") is not True:
                        excluded["unsuccessful_trajectory"] += 1
                        continue
                    if (
                        row.get("score_contract", {}).get("success_source")
                        != "canonical_gates"
                    ):
                        raise ValueError(
                            "Use R1 canonical trajectories; legacy score labels need a separate recipe"
                        )
                    _validate_record(row, path.parent)
                    record = _convert_record(row, path.parent, strip_think=True)
                    validate_messages(record)
                samples.append(
                    {
                        "kind": kind,
                        "scene": str(scene),
                        "id": str(sid),
                        "source": str(path),
                        "record": record,
                        "parent_task_id": parent["task_id"] if parent else None,
                        "source_task_sha256": parent["source_task_sha256"] if parent else None,
                    }
                )
    scenes = sorted(source_scenes)
    if args.split_manifest:
        split = json.loads(Path(args.split_manifest).read_text())
        train_scenes, val_scenes = set(split["train_scenes"]), set(split["val_scenes"])
        test_scenes = set(split.get("test_scenes", []))
        if train_scenes & val_scenes or set(scenes) - (train_scenes | val_scenes | test_scenes):
            raise ValueError("Overlapping or unknown scenes in split manifest")
    else:
        if len(scenes) < 2:
            raise ValueError(
                "Need >=2 scenes for scene-disjoint validation; never split paired views randomly"
            )
        test_scenes = set()
        random.Random(args.seed).shuffle(scenes)
        count = min(len(scenes) - 1, max(1, round(len(scenes) * args.val_fraction)))
        val_scenes, train_scenes = set(scenes[:count]), set(scenes[count:])
    active = [
        k
        for k, p in (
            ("trajectory", 1 - args.qa_probability),
            ("qa", args.qa_probability),
        )
        if p > 0
    ]
    buckets = {
        (kind, split): [] for kind in ("trajectory", "qa") for split in ("train", "val", "test")
    }
    audit = []
    for sample in samples:
        split = "test" if sample["scene"] in test_scenes else "val" if sample["scene"] in val_scenes else "train"
        buckets[sample["kind"], split].append(sample["record"])
        audit.append(
            {k: v for k, v in sample.items() if k != "record"} | {"split": split}
        )
    for kind in active:
        for split in ("train", "val"):
            if not buckets[kind, split]:
                raise ValueError(
                    f"Empty {kind}/{split}; choose a split with coverage of both data sources"
                )
    out.mkdir(parents=True)
    info = {}
    for (kind, split), records in buckets.items():
        name = f"{kind}_{split}"
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        info[name] = {
            "file_name": f"{name}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    cfg = {
        "model_name_or_path": str(model),
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "full",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": False,
        "dataset_dir": str(out),
        "dataset": ",".join(k + "_train" for k in active),
        "eval_dataset": ",".join(k + "_val" for k in active),
        "val_size": 0.0,
        "template": "qwen2_vl",
        "cutoff_len": args.cutoff_len,
        "mask_history": False,
        "output_dir": str(out / "model"),
        "overwrite_output_dir": False,
        "seed": args.seed,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": True,
        "flash_attn": "sdpa",
        "logging_steps": 5,
        "save_steps": 200,
        "eval_strategy": "steps",
        "eval_steps": 200,
        "report_to": "none",
    }
    if len(active) == 2:
        cfg.update(
            mix_strategy="interleave_over",
            interleave_probs=f"{1 - args.qa_probability},{args.qa_probability}",
        )
    (out / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = {
        **(frozen if strict else {}),
        "version": 2 if strict else 1,
        "seed": args.seed,
        "split_unit": "scene",
        "trajectory_prompt_format": "no_think",
        "train_scenes": sorted(train_scenes),
        "val_scenes": sorted(val_scenes),
        "test_scenes": sorted(test_scenes),
        "qa_probability": args.qa_probability,
        "sources": sources,
        "counts": {f"{k}/{s}": len(v) for (k, s), v in buckets.items()},
        "excluded": dict(excluded),
        "records": audit,
        "qa_label_counts": {
            s: dict(Counter(r["messages"][-1]["content"] for r in buckets["qa", s]))
            for s in ("train", "val", "test")
        },
        "note": "Sampling probability counts examples, not supervised tokens. External test scenes must be excluded upstream.",
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trajectory",
        action="append",
        default=[],
        help="Internal R1 sft_data.jsonl, retaining scene IDs",
    )
    p.add_argument(
        "--qa-bank",
        action="append",
        default=[],
        help="Rendered training manifest.jsonl",
    )
    p.add_argument("--model", required=True, help="Local Qwen2.5-VL HF checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--qa-probability", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--split-manifest",
        help="Reuse a previously frozen scene split across ablations",
    )
    p.add_argument("--task-registry", help="Required raw task registry for release-v2 splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cutoff-len", type=int, default=16384)
    args = p.parse_args()
    if args.cutoff_len <= 0:
        p.error("cutoff-len must be positive")
    result = prepare(args)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("counts", "excluded", "train_scenes", "val_scenes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
