#!/usr/bin/env python3
"""Write a manifest.json for a dataset, model-weight or result directory (cleanup task, 2026-10).

    python scripts/cluster/make_manifest.py --kind dataset --name interiorgs_v10_tasks_r10 \
        --path data/releases/interiorgs_v10_tasks_r10 --note "round 10, thick walls"

The manifest records: name, kind, resolved path (and the symlink if --path is one), size, creation time,
the code commit found in the directory (code_commit.txt / code_commit_post.txt) and the repo commit at
manifest time, per-file SHA-256 of the small key files, and kind-specific counts:
  dataset: line counts of every *.jsonl at depth <= 2 and the per-split scene counts of split_manifest.json
  model:   model_type / hidden_size from config.json, number and total bytes of *.safetensors shards
  result:  the markdown / json files present
Only stdlib; safe to run on the login node (no model loading).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HASH_LIMIT = 2 * 1024 ** 3  # do not hash files larger than 2 GB (weights are identified by size + index)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def du_kb(path: Path) -> int:
    try:
        return int(subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=True).stdout.split()[0])
    except Exception:
        return -1


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def read_commit(d: Path) -> str | None:
    for name in ("code_commit_post.txt", "code_commit.txt"):
        p = d / name
        if p.is_file():
            return p.read_text().strip().split()[0]
    return None


def count_lines(p: Path) -> int:
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
    return n


def dataset_info(d: Path) -> dict:
    info: dict = {"jsonl_line_counts": {}, "splits": None}
    for p in sorted(d.rglob("*.jsonl")):
        if len(p.relative_to(d).parts) <= 2 and p.stat().st_size < 8 * 1024 ** 3:
            info["jsonl_line_counts"][str(p.relative_to(d))] = count_lines(p)
    sm = d / "split_manifest.json"
    if sm.is_file():
        try:
            m = json.loads(sm.read_text())
            scenes = m.get("scenes") or m.get("scene_split") or m
            if isinstance(scenes, dict):
                counts: dict = {}
                for scene, split in scenes.items():
                    if isinstance(split, str):
                        counts[split] = counts.get(split, 0) + 1
                info["splits"] = counts or {"raw_keys": list(m.keys())[:10]}
        except Exception as e:  # keep going; the manifest must never fail on an odd file
            info["splits"] = f"unreadable: {e}"
    return info


def model_info(d: Path) -> dict:
    info: dict = {}
    cfg = d / "config.json"
    if cfg.is_file():
        try:
            c = json.loads(cfg.read_text())
            text = c.get("text_config", c)
            info.update(model_type=c.get("model_type"), hidden_size=text.get("hidden_size"), num_hidden_layers=text.get("num_hidden_layers"))
        except Exception as e:
            info["config_error"] = str(e)
    shards = sorted(d.glob("*.safetensors"))
    info["safetensors_shards"] = len(shards)
    info["safetensors_bytes"] = sum(p.stat().st_size for p in shards)
    pts = sorted(d.rglob("*.pt"))
    if pts:
        info["pt_files"] = len(pts)
        info["pt_bytes"] = sum(p.stat().st_size for p in pts)
    return info


def result_info(d: Path) -> dict:
    files = [str(p.relative_to(d)) for p in sorted(d.rglob("*")) if p.is_file() and p.suffix in (".md", ".json", ".csv") and len(p.relative_to(d).parts) <= 2]
    return {"report_files": files[:50]}


def key_files(d: Path, kind: str) -> list[Path]:
    pats = {
        "dataset": ["*.jsonl", "*.json", "output/*.jsonl", "output/*.json", "sft/*.jsonl", "sft/*.json", "*.txt"],
        "model": ["config.json", "*.index.json", "generation_config.json", "tokenizer_config.json", "all_results.json", "train_results.json", "eval_results.json", "config.yaml", "dataset_statistics.json"],
        "result": ["*.md", "*.json", "*.csv", "*/*.json", "*/*.md"],
    }[kind]
    out: list[Path] = []
    for pat in pats:
        out += [p for p in d.glob(pat) if p.is_file()]
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("dataset", "model", "result"), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--source", default=None, help="run directory the entry was produced in (for provenance)")
    ap.add_argument("--note", default="")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None, help="manifest path (default: <resolved path>/manifest.json)")
    args = ap.parse_args()

    link = Path(args.path)
    target = link.resolve()
    if not target.is_dir():
        print(f"[error] not a directory: {target}", file=sys.stderr)
        return 2
    source = Path(args.source).resolve() if args.source else None
    files = key_files(target, args.kind)
    hashes = {str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": sha256(p) if p.stat().st_size <= HASH_LIMIT else None} for p in files}
    manifest = {
        "manifest_version": 1,
        "name": args.name,
        "kind": args.kind,
        "path": str(target),
        "link": str(link) if link.is_symlink() else None,
        "size_KB": du_kb(target),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max([p.stat().st_mtime for p in files] + [target.stat().st_mtime]))),
        "code_commit": read_commit(target) or (read_commit(source) if source else None),
        "source_run": str(source) if source else None,
        "manifest_written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_repo_commit": git_head(Path(args.repo)),
        "note": args.note,
        "key_files": hashes,
    }
    manifest.update({"dataset": dataset_info(target)} if args.kind == "dataset" else {"model": model_info(target)} if args.kind == "model" else {"result": result_info(target)})
    out = Path(args.out) if args.out else target / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "key_files"}, ensure_ascii=False))
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
