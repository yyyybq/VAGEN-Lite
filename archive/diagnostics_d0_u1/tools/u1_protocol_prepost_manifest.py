#!/usr/bin/env python3
"""Create the immutable 64-sample Phase-4B protocol evaluation manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types_clean_layout_render.jsonl"
DEFAULT_OUT = ROOT / "exps/vagen_active_spatial/protocol_only_prepost/protocol_eval_manifest.jsonl"
DEFAULT_MODEL = Path("/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT")
DEFAULT_RENDERER = "http://10.119.27.237:8768/render"
QUOTAS = {
    "absolute_positioning": 10,
    "centering": 4,
    "delta_control": 10,
    "equidistance": 10,
    "fov_inclusion": 10,
    "occlusion_alignment": 10,
    "projective_relations": 10,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evenly_spaced(items: list[tuple[int, dict]], count: int) -> list[tuple[int, dict]]:
    if len(items) < count:
        raise ValueError(f"requested {count} unique items from only {len(items)}")
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise AssertionError(f"non-unique evenly spaced indices: {indices}")
    return [items[i] for i in indices]


def model_identity(model: Path) -> dict:
    selected = []
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        path = model / name
        if path.exists():
            selected.append({"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    shards = sorted(model.glob("*.safetensors"))
    shard_inventory = [{"name": p.name, "size": p.stat().st_size} for p in shards]
    identity_payload = {"path": str(model), "files": selected, "shards": shard_inventory}
    identity_payload["identity_sha256"] = sha256_bytes(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    return identity_payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--renderer-url", default=DEFAULT_RENDERER)
    args = ap.parse_args()

    raw_lines = [line for line in args.data.read_text().splitlines() if line.strip()]
    # Evaluation uses the complete clean-layout file so all four centering
    # examples are represented. Training remains unchanged at seeds [0, 2372).
    train_lines = raw_lines[:2372]
    rows = [json.loads(line) for line in raw_lines]
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for seed, row in enumerate(rows):
        grouped[str(row.get("task_type", "unknown"))].append((seed, row))

    if set(grouped) != set(QUOTAS):
        raise RuntimeError(f"task types changed: data={sorted(grouped)} expected={sorted(QUOTAS)}")

    identity = model_identity(args.model)
    prompt_sources = [
        ROOT / "vagen/envs/active_spatial/prompt.py",
        ROOT / "vagen/envs/active_spatial/env.py",
        ROOT / "vagen/models/sensenova_u1_processor.py",
    ]
    prompt_hash = sha256_bytes(b"".join(path.read_bytes() for path in prompt_sources))
    training_hash = sha256_bytes(("\n".join(train_lines) + "\n").encode())
    evaluation_pool_hash = sha256_bytes(("\n".join(raw_lines) + "\n").encode())

    selected: list[tuple[int, dict]] = []
    for task_type in sorted(QUOTAS):
        selected.extend(evenly_spaced(grouped[task_type], QUOTAS[task_type]))
    # Fixed stable order: interleave task types by original seed. This preserves
    # a deterministic order while preventing long same-task runs per eval shard.
    selected.sort(key=lambda item: item[0])

    manifest = []
    for order, (seed, row) in enumerate(selected):
        scene = str(row.get("scene_id"))
        obj = str(row.get("object_label"))
        preset = str(row.get("preset"))
        sample_id = f"protocol64-{order:03d}-seed{seed:04d}-{row['task_type']}"
        manifest.append(
            {
                "manifest_version": "phase4b_protocol64_v1",
                "sample_id": sample_id,
                "order": order,
                "dataset_index": seed,
                "seed": seed,
                "scene_id": scene,
                "task_id": f"{scene}/{obj}/{preset}/{seed}",
                "task_type": row["task_type"],
                "object_label": obj,
                "preset": preset,
                "renderer_url": args.renderer_url,
                "prompt_format": "free_think_fwd_first",
                "prompt_template_version": f"active_spatial_u1@sha256:{prompt_hash}",
                "max_output_length": 512,
                "temperature": 0.8,
                "top_p": 0.92,
                "generation_seed": 202608290000 + order,
                "reference_model_checkpoint_path": str(args.model),
                "reference_model_checkpoint_identity": identity["identity_sha256"],
                "data_path": str(args.data),
                "training_split_sha256": training_hash,
                "evaluation_pool_sha256": evaluation_pool_hash,
                "evaluation_task_filter": {"include": "ALL", "exclude": []},
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest)
    args.out.write_text(content, encoding="utf-8")
    manifest_sha = sha256_bytes(content.encode())
    report = {
        "status": "PASS",
        "manifest": str(args.out),
        "manifest_sha256": manifest_sha,
        "n": len(manifest),
        "task_counts": dict(sorted(Counter(x["task_type"] for x in manifest).items())),
        "unique_seeds": len({x["seed"] for x in manifest}),
        "unique_scenes": len({x["scene_id"] for x in manifest}),
        "source_train_rows": len(train_lines),
        "source_evaluation_pool_rows": len(rows),
        "selection": "all 4 centering rows; 10 evenly-spaced unique rows for each other task type",
        "base_model_identity": identity,
        "prompt_template_sha256": prompt_hash,
        "training_split_sha256": training_hash,
        "evaluation_pool_sha256": evaluation_pool_hash,
        "evaluation_task_filter": {"include": "ALL", "exclude": []},
        "decoding": {"max_output_length": 512, "temperature": 0.8, "top_p": 0.92},
        "renderer_url": args.renderer_url,
    }
    (args.out.parent / "protocol_eval_manifest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
