#!/usr/bin/env python3
"""Prepare episode-disjoint RoboCasa data and StarVLA configs from HF actors.

Uses small metadata copies and per-episode symlinks; never copies model tensors
or writes into the source checkpoint/dataset. Run with the starVLA interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

ROBOT = "panda_omron_robocasa365"


def json_file(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_checkpoint(path):
    """Validate complete HF weights, tokenizer and vision processor on CPU."""
    from safetensors import safe_open
    from transformers import AutoProcessor

    root = Path(path).resolve(strict=True)
    config = json_file(root / "config.json")
    if config.get("model_type") not in {"qwen2_5_vl", "qwen3_vl"}:
        raise ValueError("Transfer currently supports Qwen2.5-VL / Qwen3-VL HF actors; other backbones need adapters")
    if (root / "adapter_config.json").exists():
        raise ValueError("Merge the LoRA adapter into its original base before transfer")
    index = root / "model.safetensors.index.json"
    mapping = json_file(index)["weight_map"] if index.is_file() else None
    shards = sorted(set(mapping.values())) if mapping else ["model.safetensors"]
    inventory, shapes = [], {}
    for name in shards:
        shard = root / name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"Incomplete HF export: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if mapping and not {k for k, v in mapping.items() if v == name}.issubset(keys):
                raise ValueError(f"Index refers to missing tensors in {shard}")
            for key in keys:
                if key.endswith("embed_tokens.weight") or key == "lm_head.weight":
                    shapes[key] = list(handle.get_slice(key).get_shape())
        inventory.append({"name": name, "bytes": shard.stat().st_size, "mtime_ns": shard.stat().st_mtime_ns})
    processor = AutoProcessor.from_pretrained(str(root), local_files_only=True)
    tokenizer = processor.tokenizer
    token = tokenizer.encode("🔍", add_special_tokens=False)
    if len(token) != 1 or tokenizer.encode("🔍" * 16, add_special_tokens=False) != token * 16:
        raise ValueError("QwenOFT action query must tokenize atomically; prepare a separate resized checkpoint")
    text_cfg = config.get("text_config", config)
    embeddings = [v for k, v in shapes.items() if k.endswith("embed_tokens.weight")]
    if len(embeddings) != 1 or embeddings[0][0] < len(tokenizer):
        raise ValueError("Tokenizer and input embedding size mismatch")
    if embeddings[0] != [text_cfg["vocab_size"], text_cfg["hidden_size"]]:
        raise ValueError("Embedding dimensions differ from model config")
    return {
        "path": str(root), "model_type": config["model_type"],
        "hidden_size": text_cfg["hidden_size"], "config_sha256": fingerprint(root / "config.json"),
        "action_token_id": token[0], "weight_files": inventory,
        "weight_identity": "file size and mtime; not a full tensor hash",
        "processor_files": {p.name: fingerprint(p) for p in root.glob("*.json") if p.stat().st_size < 30000000},
    }


def partition_episodes(episodes, fraction, seed):
    if not 0 < fraction < 1 or len(episodes) < 2:
        raise ValueError("Need at least two episodes and 0 < val_fraction < 1")
    ids = [int(e["episode_index"]) for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode IDs")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    count = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    return sorted(shuffled[count:]), sorted(shuffled[:count])


def make_split(src, dst, episodes, ids):
    from check_starvla_navigatekitchen import check_lerobot

    check_lerobot(src)
    info = json_file(src / "meta/info.json")
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError("Only per-episode LeRobot v2 data is supported")
    if info["features"]["action"]["shape"] != [12] or info["features"]["observation.state"]["shape"] != [16]:
        raise ValueError("Dataset must use PandaOmron 12-D action / 16-D state")
    dst.mkdir(parents=True, exist_ok=False)
    (dst / "meta").mkdir()
    selected = [e for e in episodes if int(e["episode_index"]) in set(ids)]
    for name in ("modality.json", "tasks.jsonl", "embodiment.json"):
        if (src / "meta" / name).is_file():
            shutil.copy2(src / "meta" / name, dst / "meta" / name)
    # No full-dataset statistics/cache files are copied. StarVLA recomputes
    # normalization exclusively from the linked training parquet files.
    for e in selected:
        idx = int(e["episode_index"])
        rel = info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
        source = (src / rel).resolve(strict=True)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        for key in info["features"]:
            if key.startswith("observation.images."):
                video_rel = info["video_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx, video_key=key)
                if not (src / video_rel).is_file():
                    raise FileNotFoundError(src / video_rel)
    (dst / "videos").symlink_to((src / "videos").resolve(strict=True), target_is_directory=True)
    (dst / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in selected))
    info["total_episodes"] = len(selected)
    info["total_frames"] = sum(e["length"] for e in selected)
    info["total_videos"] = len(selected) * sum(k.startswith("observation.images.") for k in info["features"])
    # StarVLA uses explicit episodes.jsonl IDs; these are sparse original IDs.
    info.pop("splits", None)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")


def prepare_data(specs, out, fraction, seed):
    manifest = {"seed": seed, "val_fraction": fraction, "split_unit": "episode", "datasets": []}
    for alias, path in specs:
        src = Path(path).resolve(strict=True)
        episodes = [json.loads(line) for line in (src / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        train, val = partition_episodes(episodes, fraction, seed)
        for split, ids in (("train", train), ("val", val)):
            make_split(src, out / "data" / split / alias / "lerobot", episodes, ids)
        manifest["datasets"].append({
            "name": alias, "source": str(src), "train_ids": train, "val_ids": val,
            "source_info_sha256": fingerprint(src / "meta/info.json"),
            "source_episodes_sha256": fingerprint(src / "meta/episodes.jsonl"),
            "frames": {split: sum(e["length"] for e in episodes if e["episode_index"] in ids)
                       for split, ids in (("train", train), ("val", val))},
        })
    return manifest


def parse_spec(value):
    alias, sep, path = value.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
        raise argparse.ArgumentTypeError("Use NAME=/absolute/path (NAME: letters, digits, _ or -)")
    return alias, path


def main():
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", action="append", type=parse_spec, required=True)
    p.add_argument("--source", action="append", type=parse_spec, default=[], help="base=/hf/path, sft=/hf/path, sft_rl=/hf/path; labels are user-declared provenance")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()
    if args.steps <= 0 or args.batch <= 0:
        p.error("steps and batch must be positive")
    for specs in (args.dataset, args.source):
        if len({name for name, _ in specs}) != len(specs):
            p.error("Duplicate names")
    out = args.out.resolve()
    if out.exists():
        p.error("Output already exists; use a new directory to preserve the experiment")
    sources = {name: inspect_checkpoint(path) for name, path in args.source}
    if len({(s["model_type"], s["hidden_size"]) for s in sources.values()}) > 1:
        p.error("Controlled comparisons require the same backbone family and size")
    out.mkdir(parents=True)
    manifest = prepare_data(args.dataset, out, args.val_fraction, args.seed)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    template = root / "examples/train/robocasa/starvla_qwenoft_navigatekitchen_smoke.yaml"
    for name, source in sources.items():
        cfg = yaml.safe_load(template.read_text())
        cfg.update(run_id=name, run_root_dir=str(out / "runs"), seed=args.seed)
        cfg["framework"]["qwenvl"].update(base_vlm=source["path"], attn_implementation="sdpa")
        data = cfg["datasets"]["vla_data"]
        data.update(data_root_dir=str(out / "data/train"), data_mix="robocasa365_transfer",
                    mixture_spec=[[d["name"] + "/lerobot", 1.0, ROBOT] for d in manifest["datasets"]],
                    video_backend="torchvision_av", per_device_batch_size=args.batch,
                    num_workers=4, drop_incomplete_action_chunks=True)
        trainer = cfg["trainer"]
        trainer.update(max_train_steps=args.steps, num_warmup_steps=min(100, args.steps // 10),
                       save_interval=min(1000, args.steps), eval_interval=min(1000, args.steps), logging_frequency=10)
        trainer["learning_rate"] = {"base": 1e-5, "qwen_vl_interface": 1e-6, "action_model": 1e-4}
        trainer["scheduler_specific_kwargs"] = {"min_lr": 1e-7}
        cfg["transfer"] = {"source_label": name, "source_manifest": str(out / "sources.json"),
                           "split_manifest": str(out / "split_manifest.json"),
                           "note": "trainer mse_score is training-batch diagnostic; use heldout evaluator"}
        (out / (name + ".yaml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps({"out": str(out), "sources": list(sources), "datasets": [
        {"name": d["name"], "train": len(d["train_ids"]), "val": len(d["val_ids"])} for d in manifest["datasets"]]}, indent=2))


if __name__ == "__main__":
    main()
