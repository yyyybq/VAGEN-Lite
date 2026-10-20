#!/usr/bin/env python3
"""Turn an episode manifest (subsets.py) into a StarVLA training directory.  CPU only; never writes to the source.

    python downstream/demos/prepare_data.py --manifest downstream/demos/manifests/step2_smoke.json --out RUN/data

Layout (same as scripts/prepare_starvla_transfer.py, whose make_split it reuses):
  OUT/train/<Task>/lerobot/   per-episode links to the training parquet files, a link to the source videos, and a
                              writable meta/ (StarVLA writes its stats / step-index caches there)
  OUT/split_manifest.json     name, source, train_ids, val_ids, source hashes per task (read by
                              scripts/eval_starvla_heldout.py and by the training entry's checks)
  OUT/mixture_spec.json       [[<Task>/lerobot, 1.0, panda_omron_robocasa365], ...] for the training config
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "scripts"))
ROBOT = "panda_omron_robocasa365"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mixture_spec(manifest: dict) -> list:
    return [[row["task"] + "/lerobot", 1.0, ROBOT] for row in manifest["tasks"]]


def prepare(manifest_path: Path, out: Path) -> dict:
    from prepare_starvla_transfer import make_split  # reuses the existing, tested per-episode link builder

    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    manifest = json.loads(manifest_path.read_text())
    if out.exists():
        raise FileExistsError(f"{out} exists; use a new directory")
    datasets = []
    for row in manifest["tasks"]:
        source = Path(row["source"])
        if sha256(source / "meta/info.json") != row["source_info_sha256"] or \
                sha256(source / "meta/episodes.jsonl") != row["source_episodes_sha256"]:
            raise ValueError(f"source metadata of {row['task']} changed since the manifest was written")
        eps = [json.loads(x) for x in (source / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
        make_split(source, out / "train" / row["task"] / "lerobot", eps, row["train_ids"])
        datasets.append({"name": row["task"], "source": str(source), "train_ids": row["train_ids"],
                         "val_ids": row["val_ids"], "source_info_sha256": row["source_info_sha256"],
                         "source_episodes_sha256": row["source_episodes_sha256"], "frames": row["frames"]})
    split = {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "split_unit": "episode",
             "rule": manifest["rule"], "datasets": datasets}
    (out / "split_manifest.json").write_text(json.dumps(split, indent=1) + "\n")
    (out / "mixture_spec.json").write_text(json.dumps(mixture_spec(manifest)) + "\n")
    return {"out": str(out), "tasks": len(datasets), "train_episodes": sum(len(d["train_ids"]) for d in datasets),
            "train_frames": sum(d["frames"]["train"] for d in datasets),
            "val_episodes": sum(len(d["val_ids"]) for d in datasets)}


def check(out: Path) -> dict:
    """The training directory still matches its manifest: episode files = train_ids, links point at the source."""
    split = json.loads((out / "split_manifest.json").read_text())
    problems = []
    for d in split["datasets"]:
        root = out / "train" / d["name"] / "lerobot"
        files = {int(p.stem.split("_")[-1]) for p in root.glob("data/*/*.parquet")}
        listed = {json.loads(x)["episode_index"] for x in (root / "meta/episodes.jsonl").read_text().splitlines() if x.strip()}
        if files != set(d["train_ids"]) or listed != set(d["train_ids"]):
            problems.append(f"{d['name']}: episode files / episodes.jsonl differ from train_ids")
        if set(d["train_ids"]) & set(d["val_ids"]):
            problems.append(f"{d['name']}: train / val overlap")
        source = Path(d["source"])
        if sha256(source / "meta/info.json") != d["source_info_sha256"]:
            problems.append(f"{d['name']}: source info.json changed")
    return {"ok": not problems, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="only check an existing directory")
    args = ap.parse_args(argv)
    result = check(args.out) if args.check else prepare(args.manifest, args.out)
    print(json.dumps(result, indent=1))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
