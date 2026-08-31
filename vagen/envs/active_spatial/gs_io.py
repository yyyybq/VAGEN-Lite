"""AOSS-aware InteriorGS I/O with local PLY disk cache.

``gs_root`` may be a local path or a remote URL such as
``fj:s3://baiqiao/InteriorGS``. JSON metadata is read as bytes; PLY files are
materialized under ``AOSS_PLY_CACHE_DIR`` (default ``/tmp/aoss_ply_cache``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Any, List, Optional

_CLIENT = None
_CLIENT_LOCK = threading.Lock()
_CLIENT_PID = None
_REMOTE_RE = re.compile(r"^(?:[A-Za-z0-9_-]+:)?s3://")

PLY_CANDIDATES = (
    "3dgs_compressed.ply",
    "gaussian.ply",
)


def is_remote(path: Optional[str]) -> bool:
    return bool(path) and bool(_REMOTE_RE.match(str(path)))


def default_conf_path() -> str:
    return os.environ.get(
        "AOSS_CONF",
        "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf",
    )


def get_client(conf_path: Optional[str] = None):
    global _CLIENT, _CLIENT_PID
    pid = os.getpid()
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT_PID != pid:
            from aoss_client.client import Client

            _CLIENT = Client(conf_path or default_conf_path())
            _CLIENT_PID = pid
        return _CLIENT


def join_uri(root: str, *parts: str) -> str:
    root = str(root).rstrip("/")
    bits = [root]
    for p in parts:
        if p is None or p == "":
            continue
        bits.append(str(p).lstrip("/"))
    return "/".join(bits)


def ply_cache_dir() -> str:
    root = os.environ.get(
        "AOSS_PLY_CACHE_DIR",
        os.path.join(os.environ.get("TMPDIR", "/tmp"), "aoss_ply_cache"),
    )
    os.makedirs(root, exist_ok=True)
    return root


def exists(path: str) -> bool:
    if is_remote(path):
        try:
            return bool(get_client().contains(path))
        except Exception:
            return False
    return os.path.exists(path)


def read_bytes(path: str) -> bytes:
    if is_remote(path):
        data = get_client().get(path)
        if data is None:
            raise FileNotFoundError(path)
        return data if isinstance(data, (bytes, bytearray)) else bytes(data)
    with open(path, "rb") as f:
        return f.read()


def read_json(path: str) -> Any:
    raw = read_bytes(path)
    return json.loads(raw.decode("utf-8"))


def scene_file(gs_root: str, scene_id: str, name: str) -> str:
    return join_uri(gs_root, scene_id, name)


def load_scene_json(gs_root: str, scene_id: str, name: str) -> Optional[Any]:
    path = scene_file(gs_root, scene_id, name)
    if not exists(path):
        return None
    return read_json(path)


def resolve_ply(gs_root: str, scene_id: str) -> str:
    """Return a local filesystem path to the scene PLY (cached if remote)."""
    candidates: List[str] = [
        join_uri(gs_root, scene_id, name) for name in PLY_CANDIDATES
    ]
    candidates.append(join_uri(gs_root, f"{scene_id}.ply"))

    for cand in candidates:
        if not exists(cand):
            continue
        if not is_remote(cand):
            return cand
        # Materialize remote PLY once per URL.
        digest = hashlib.sha1(cand.encode("utf-8")).hexdigest()[:16]
        local_dir = os.path.join(ply_cache_dir(), scene_id)
        os.makedirs(local_dir, exist_ok=True)
        local = os.path.join(local_dir, f"{digest}_{os.path.basename(cand)}")
        if os.path.exists(local) and os.path.getsize(local) > 0:
            return local
        tmp = local + ".partial"
        data = read_bytes(cand)
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, local)
        return local

    raise FileNotFoundError(
        f"Could not find PLY for scene {scene_id} under {gs_root}. Tried: {candidates}"
    )
