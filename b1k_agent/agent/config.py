"""Config loading, overrides, hashing and the run manifest (red line 6).

A run config is one YAML file, optionally with ``base: <other.yaml>`` inheritance and dotted
command-line overrides (``modules.plan.enabled=true``). The resolved config is hashed
(sha256 of canonical JSON) and written into the run manifest together with the code commit,
BEHAVIOR-1K / openpi commits, checkpoint, prompt versions, instances and seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
from typing import Any, Mapping, Optional, Sequence

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deep_merge(base: dict, over: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str, overrides: Sequence[str] = ()) -> dict:
    path = os.path.abspath(path)
    cfg = load_yaml(path)
    chain = [path]
    while "base" in cfg:
        base_path = cfg.pop("base")
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(chain[-1]), base_path)
        base_path = os.path.abspath(base_path)
        if base_path in chain:
            raise ValueError(f"config base cycle: {chain + [base_path]}")
        chain.append(base_path)
        cfg = _deep_merge(load_yaml(base_path), cfg)
    for ov in overrides:
        apply_override(cfg, ov)
    cfg.setdefault("_meta", {})["config_chain"] = [os.path.relpath(p, REPO_ROOT) for p in chain]
    return cfg


def apply_override(cfg: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must be key=value, got {override!r}")
    key, raw = override.split("=", 1)
    value = yaml.safe_load(raw)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {key}: {p} is not a mapping")
    node[parts[-1]] = value


def config_hash(cfg: Mapping) -> str:
    clean = {k: v for k, v in cfg.items() if k != "_meta"}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(cfg: Mapping, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, Mapping) or p not in node:
            return default
        node = node[p]
    return node


def git_info(path: str) -> dict:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(["git", "-C", path, *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    if not path or not os.path.isdir(path):
        return {"path": path, "commit": None}
    status = run("status", "--porcelain")
    return {
        "path": path,
        "commit": run("rev-parse", "HEAD"),
        "describe": run("describe", "--tags", "--always"),
        "dirty": bool(status) if status is not None else None,
    }


def file_sha256(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(cfg: Mapping, *, modules_desc: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    ckpt_dir = get(cfg, "executor.params.checkpoint_dir")
    norm_stats = []
    if ckpt_dir and os.path.isdir(ckpt_dir):
        for root, _, files in os.walk(os.path.join(ckpt_dir, "assets")):
            for fn in files:
                if fn == "norm_stats.json":
                    p = os.path.join(root, fn)
                    norm_stats.append({"path": p, "sha256": file_sha256(p)})
    manifest = {
        "config_hash": config_hash(cfg),
        "config": cfg,
        "code": git_info(REPO_ROOT),
        "behavior_1k": git_info(get(cfg, "paths.b1k_root", "")),
        "openpi": git_info(get(cfg, "paths.openpi_root", "")),
        "checkpoint": {"dir": ckpt_dir, "norm_stats": norm_stats},
        "modules": modules_desc or {},
        "seed": get(cfg, "run.seed"),
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if extra:
        manifest.update(extra)
    return manifest
