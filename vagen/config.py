"""Unified configuration of the repository (cleanup, phase 2, 2026-10).

Cluster paths live in ``configs/paths.yaml`` (two profiles: ``cluster`` and ``local``, chosen with the environment
variable ``VAGEN_PROFILE``, default ``cluster``) and the sco job settings in ``configs/cluster.yaml``.

    from vagen import config
    config.path("scenes_meta")         # -> "/mnt/umm/users/yinbaiqiao/InteriorGS_meta"
    config.cluster("sco.image")        # -> "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"

Shell scripts:  python -m vagen.config get paths.scenes_meta   /   python -m vagen.config get cluster.sco.image

``VAGEN_CONFIG_DIR`` points at another configs directory (default: ``configs/`` next to the ``vagen`` package).
Only PyYAML is needed; the module never imports anything else from the repository.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("cluster", "local")


def config_dir() -> Path:
    return Path(os.environ.get("VAGEN_CONFIG_DIR") or REPO_ROOT / "configs")


def profile() -> str:
    name = os.environ.get("VAGEN_PROFILE", "cluster")
    if name not in PROFILES:
        raise ValueError(f"VAGEN_PROFILE must be one of {PROFILES}, not {name!r}")
    return name


@lru_cache(maxsize=None)
def _load(name: str, directory: str) -> dict:
    import yaml

    path = Path(directory) / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing (set VAGEN_CONFIG_DIR to the repository's configs/ directory)")
    return yaml.safe_load(path.read_text()) or {}


def _dig(data: Any, dotted: str, where: str) -> Any:
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{where}: no key {dotted!r}")
        node = node[part]
    return node


def path(key: str) -> str:
    """A path of configs/paths.yaml in the current profile; an unset (null) key is an error that names it."""
    value = _dig(_load("paths.yaml", str(config_dir())), f"profiles.{profile()}.{key}", "configs/paths.yaml")
    if value is None:
        raise KeyError(f"configs/paths.yaml: {key!r} is not set in profile {profile()!r}")
    return str(value)


def cluster(key: str) -> Any:
    """A setting of configs/cluster.yaml, e.g. cluster("sco.image")."""
    return _dig(_load("cluster.yaml", str(config_dir())), key, "configs/cluster.yaml")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "get" or not args[1].startswith(("paths.", "cluster.")):
        print("usage: python -m vagen.config get paths.<key> | cluster.<dotted key>", file=sys.stderr)
        return 2
    kind, key = args[1].split(".", 1)
    value = path(key) if kind == "paths" else cluster(key)
    print(" ".join(map(str, value)) if isinstance(value, (list, tuple)) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
