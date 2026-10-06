#!/usr/bin/env python3
"""Refuse launch if prepared source weights or the data split have drifted."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from prepare_starvla_transfer import inspect_checkpoint, fingerprint


def check(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    transfer = cfg["transfer"]
    saved = json.loads(Path(transfer["source_manifest"]).read_text())[transfer["source_label"]]
    actual = inspect_checkpoint(cfg["framework"]["qwenvl"]["base_vlm"])
    if saved != actual:
        raise ValueError("Source checkpoint/processor differs from the preparation manifest")
    manifest = json.loads(Path(transfer["split_manifest"]).read_text())
    data = cfg["datasets"]["vla_data"]
    expected_spec = [[d["name"] + "/lerobot", 1.0, "panda_omron_robocasa365"] for d in manifest["datasets"]]
    if data["mixture_spec"] != expected_spec:
        raise ValueError("Training mixture differs from split manifest")
    for dataset in manifest["datasets"]:
        train, val = set(dataset["train_ids"]), set(dataset["val_ids"])
        if not train or not val or train & val:
            raise ValueError("Empty or overlapping episode split")
        root = Path(data["data_root_dir"]) / dataset["name"] / "lerobot"
        ids = {json.loads(line)["episode_index"] for line in (root / "meta/episodes.jsonl").read_text().splitlines()}
        files = {int(p.stem.split("_")[-1]) for p in root.glob("data/*/*.parquet")}
        if ids != train or files != train:
            raise ValueError("Training episodes/parquet files differ from split manifest")
        source = Path(dataset["source"])
        if (fingerprint(source / "meta/info.json") != dataset["source_info_sha256"] or
                fingerprint(source / "meta/episodes.jsonl") != dataset["source_episodes_sha256"]):
            raise ValueError("Source dataset metadata changed after preparation")
        for file in root.glob("data/*/*.parquet"):
            if file.resolve(strict=True) != (source / file.relative_to(root)).resolve(strict=True):
                raise ValueError(f"Training link points outside the declared source: {file}")
    run = Path(cfg["run_root_dir"]) / cfg["run_id"]
    if run.exists():
        raise ValueError(f"Run already exists: {run}; choose a fresh run_id")
    return {"ok": True, "source": actual["path"], "model_type": actual["model_type"],
            "source_label": transfer["source_label"], "datasets": [d["name"] for d in manifest["datasets"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    print(json.dumps(check(parser.parse_args().config), indent=2))
