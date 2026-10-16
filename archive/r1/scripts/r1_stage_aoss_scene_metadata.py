#!/usr/bin/env python3
"""Stage only InteriorGS labels/structure metadata for train-only scenes.

Credentials are read at runtime and never written to output.  This deliberately
does not download Gaussian assets; the metadata cache is sufficient for static
object-pair enumeration and is resumable per scene.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "r1_aoss_scene_metadata_stage_v1_20260928"
AMBIGUOUS = {"0059_839917", "0265_840795", "0270_840784", "0314_840535", "0328_840489", "0349_840373"}
FILES = ("labels.json", "structure.json")
METADATA_INCLUDE_PATTERN = r"(labels\.json|structure\.json)$"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def credentials(path: Path) -> tuple[str, str]:
    parser = configparser.RawConfigParser()
    parser.read(path)
    access = parser.get("fj", "access_key", fallback="").strip()
    secret = parser.get("fj", "secret_key", fallback="").strip()
    if not access or not secret:
        raise RuntimeError(f"unable to read AOSS credentials from {path}")
    return access, secret


def redact(value: str, access: str, secret: str) -> str:
    value = value.replace(access, "<redacted>").replace(secret, "<redacted>")
    return re.sub(r"s3://[^/@:\s]+:[^/@\s]+@", "s3://<redacted>@", value)


def metadata_sync_command(
    ads_cli: Path,
    source_dir: str,
    destination_dir: Path,
    *,
    threads: int = 2,
    listers: int = 1,
) -> list[str]:
    """Build the metadata-only ads-cli sync command.

    ads-cli v1.9 requires directory copies to use trailing slashes on both
    operands.  The include regex keeps this operation metadata-only instead of
    staging the complete InteriorGS scene.
    """
    return [
        str(ads_cli),
        "--quiet",
        "--threads",
        str(threads),
        "--listers",
        str(listers),
        "--conntimeout",
        "60",
        "--timeout",
        "300",
        "--include",
        METADATA_INCLUDE_PATTERN,
        "sync",
        source_dir.rstrip("/") + "/",
        str(destination_dir).rstrip("/") + "/",
    ]


def validate(scene_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "valid", "files": {}, "errors": []}
    for name in FILES:
        path = scene_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            result["errors"].append(f"missing_or_empty:{name}")
            continue
        try:
            value = json.loads(path.read_text())
            if name == "labels.json" and (not isinstance(value, list) or not value):
                result["errors"].append("labels_empty_or_not_list")
            if name == "structure.json" and (not isinstance(value, dict) or not isinstance(value.get("rooms"), list) or not value["rooms"]):
                result["errors"].append("structure_has_no_rooms")
        except Exception as error:
            result["errors"].append(f"invalid_json:{name}:{type(error).__name__}")
        result["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if result["errors"]:
        result["status"] = "invalid"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-train", type=Path, required=True)
    parser.add_argument("--canonical-eval", type=Path, required=True)
    parser.add_argument("--local-action-parents", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--conf", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"))
    parser.add_argument("--ads-cli", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/aoss/ads-cli"))
    parser.add_argument("--endpoint", default="aoss-internal.cn-fz-01.fjscmsapi-oss.com")
    parser.add_argument("--bucket", default="baiqiao")
    parser.add_argument("--prefix", default="InteriorGS")
    parser.add_argument("--process-timeout", type=int, default=600)
    args = parser.parse_args()

    old = read_jsonl(args.old_train)
    dev = read_jsonl(args.canonical_eval)
    local = read_jsonl(args.local_action_parents)
    eval_scenes = {str(row.get("scene_id")) for row in dev + local if row.get("scene_id")}
    train_scenes = sorted({str(row["scene_id"]) for row in old} - eval_scenes - AMBIGUOUS)
    access, secret = credentials(args.conf)
    if args.ledger.is_file():
        ledger = json.loads(args.ledger.read_text())
        if ledger.get("scenes") != train_scenes:
            raise RuntimeError("resume scene scope mismatch")
    else:
        ledger = {
            "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "scenes": train_scenes, "eval_scenes_excluded": sorted(eval_scenes),
            "ambiguous_scenes_excluded": sorted(AMBIGUOUS), "records": {},
            "inputs": {str(path): sha256(path) for path in (args.old_train, args.canonical_eval, args.local_action_parents)},
        }
        atomic_json(args.ledger, ledger)

    ready_root = args.cache_root / "ready"
    staging_root = args.cache_root / ".staging"
    ready_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    for position, scene in enumerate(train_scenes, start=1):
        ready = ready_root / scene
        if ready.is_dir() and validate(ready)["status"] == "valid":
            result = validate(ready)
            ledger["records"][scene] = {"status": "ready", "validation": result, "resumed": True}
            atomic_json(args.ledger, ledger)
            print(json.dumps({"position": position, "total": len(train_scenes), "scene_id": scene, "status": "resume_ready"}), flush=True)
            continue
        if ready.exists():
            raise RuntimeError(f"existing metadata cache is invalid; refusing overwrite: {ready}")
        staging = Path(tempfile.mkdtemp(prefix=f"{scene}.", dir=staging_root))
        try:
            source = f"s3://{access}:{secret}@{args.bucket}.{args.endpoint}/{args.prefix.strip('/')}/{scene}/"
            command = metadata_sync_command(args.ads_cli, source, staging)
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.process_timeout, check=False)
            if completed.returncode:
                safe = redact(completed.stdout + completed.stderr, access, secret)[-2000:]
                raise RuntimeError(f"AOSS metadata sync failed for {scene}: rc={completed.returncode}: {safe}")
            result = validate(staging)
            if result["status"] != "valid":
                raise RuntimeError(f"metadata validation failed for {scene}: {result['errors']}")
            os.replace(staging, ready)
            ledger["records"][scene] = {"status": "ready", "validation": result, "resumed": False}
        except Exception as error:
            ledger["records"][scene] = {"status": "failed", "reason": redact(str(error), access, secret)}
            atomic_json(args.ledger, ledger)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        atomic_json(args.ledger, ledger)
        print(json.dumps({"position": position, "total": len(train_scenes), "scene_id": scene, "status": "ready"}), flush=True)
    ledger["completed_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["ready_scenes"] = sum(row.get("status") == "ready" for row in ledger["records"].values())
    ledger["accounting_closed"] = ledger["ready_scenes"] == len(train_scenes)
    atomic_json(args.ledger, ledger)
    print(json.dumps({"scenes": len(train_scenes), "ready": ledger["ready_scenes"], "accounting_closed": ledger["accounting_closed"]}, indent=2))


if __name__ == "__main__":
    main()
