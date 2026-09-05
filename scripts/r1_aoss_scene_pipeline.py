#!/usr/bin/env python3
"""Build and maintain a resumable per-scene InteriorGS AOSS cache ledger.

Credentials are read at runtime from the existing AOSS config. They are never
written to the ledger, logs, reports, or stdout.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = REPO_ROOT / "scripts/r1_sources_v2.json"
DEFAULT_CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
DEFAULT_ADS_CLI = Path("/mnt/umm/users/yinbaiqiao/aoss/ads-cli")
DEFAULT_ENDPOINT = "aoss-internal.cn-fz-01.fjscmsapi-oss.com"
DEFAULT_BUCKET = "baiqiao"
DEFAULT_PREFIX = "InteriorGS"
REQUIRED_FILES = ("structure.json", "labels.json", "3dgs_compressed.ply")
OPTIONAL_FILES = ("occupancy.json", "occupancy.png")
TASKS = ("projective_relations", "fov_inclusion")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.chmod(temp_name, 0o644)
    os.replace(temp_name, path)


@contextmanager
def ledger_lock(path: Path):
    """Serialize ledger read-modify-write transactions across windows/nodes."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_sources(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for split, value in raw.items():
        source = Path(value)
        if not source.is_absolute():
            source = REPO_ROOT / source
        if not source.is_file():
            raise FileNotFoundError(f"missing source manifest for {split}: {source}")
        result[str(split)] = source.resolve()
    return result


def public_uri(bucket: str, prefix: str, scene_id: str) -> str:
    return f"s3://{bucket}/{prefix.strip('/')}/{scene_id}/"


def build_ledger(sources_path: Path, output: Path, tsv_output: Path | None, bucket: str, prefix: str) -> dict[str, Any]:
    sources = resolve_sources(sources_path)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_summary: dict[str, dict[str, Any]] = {}
    source_hashes = {}
    for split, source in sources.items():
        rows = read_jsonl(source)
        source_hashes[split] = sha256(source)
        scenes = set()
        task_counts = Counter()
        for row in rows:
            scene_id = str(row.get("scene_id") or "")
            if not scene_id:
                raise ValueError(f"row without scene_id in {split}")
            scenes.add(scene_id)
            scene_splits[scene_id].add(split)
            task = str(row.get("task_type") or "unknown")
            task_counts[task] += 1
            if task in TASKS:
                scene_counts[scene_id][f"{split}:{task}"] += 1
        split_summary[split] = {
            "source": str(source),
            "source_sha256": source_hashes[split],
            "rows": len(rows),
            "scenes": len(scenes),
            "projective_rows": task_counts["projective_relations"],
            "fov_rows": task_counts["fov_inclusion"],
        }
    entries = []
    for scene_id in sorted(scene_splits):
        entries.append(
            {
                "scene_id": scene_id,
                "splits": sorted(scene_splits[scene_id]),
                "split_task_counts": dict(sorted(scene_counts[scene_id].items())),
                "aoss_uri": public_uri(bucket, prefix, scene_id),
                "required_files": list(REQUIRED_FILES),
                "optional_files": list(OPTIONAL_FILES),
                "download_status": "pending",
                "asset_validation": {"status": "pending"},
                "renderer_smoke_status": "pending",
                "processing_status": "pending",
                "attempts": [],
            }
        )
    ledger = {
        "schema_version": "r1_aoss_scene_ledger_v1",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "bucket": bucket,
        "prefix": prefix.strip("/"),
        "required_scene_count": len(entries),
        "source_manifests": split_summary,
        "scenes": entries,
    }
    atomic_json(output, ledger)
    if tsv_output:
        tsv_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=tsv_output.parent, prefix=f".{tsv_output.name}.", delete=False, newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("scene_id", "split", "aoss_uri", "required_files", "download_status", "asset_validation", "renderer_smoke_status"),
                delimiter="\t",
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "scene_id": entry["scene_id"],
                        "split": ",".join(entry["splits"]),
                        "aoss_uri": entry["aoss_uri"],
                        "required_files": ",".join(entry["required_files"]),
                        "download_status": entry["download_status"],
                        "asset_validation": entry["asset_validation"]["status"],
                        "renderer_smoke_status": entry["renderer_smoke_status"],
                    }
                )
            temp_name = handle.name
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, tsv_output)
    return ledger


def load_ledger(path: Path) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != "r1_aoss_scene_ledger_v1":
        raise ValueError(f"unsupported ledger schema: {ledger.get('schema_version')}")
    return ledger


def scene_entry(ledger: dict[str, Any], scene_id: str) -> dict[str, Any]:
    for entry in ledger["scenes"]:
        if entry["scene_id"] == scene_id:
            return entry
    raise KeyError(f"scene not present in ledger: {scene_id}")


def validate_scene(path: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    empty = [name for name in REQUIRED_FILES if (path / name).is_file() and (path / name).stat().st_size <= 0]
    errors = []
    if not missing and not empty:
        try:
            structure = json.loads((path / "structure.json").read_text(encoding="utf-8"))
            if not isinstance(structure.get("rooms"), list) or not structure["rooms"]:
                errors.append("structure_has_no_rooms")
        except Exception as exc:
            errors.append(f"invalid_structure_json:{type(exc).__name__}")
        try:
            labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))
            if not isinstance(labels, list) or not labels:
                errors.append("labels_empty_or_not_list")
        except Exception as exc:
            errors.append(f"invalid_labels_json:{type(exc).__name__}")
        try:
            with (path / "3dgs_compressed.ply").open("rb") as handle:
                if handle.read(3) != b"ply":
                    errors.append("invalid_ply_header")
            if (path / "3dgs_compressed.ply").stat().st_size < 1024 * 1024:
                errors.append("ply_too_small")
        except Exception as exc:
            errors.append(f"invalid_ply:{type(exc).__name__}")
    file_names = [name for name in (*REQUIRED_FILES, *OPTIONAL_FILES) if (path / name).is_file()]
    files = {
        name: {"bytes": (path / name).stat().st_size, "sha256": sha256(path / name)}
        for name in file_names
    }
    return {
        "status": "valid" if not missing and not empty and not errors else "invalid",
        "validated_at": utc_now(),
        "missing": missing,
        "empty": empty,
        "errors": errors,
        "files": files,
    }


def credentials(conf: Path) -> tuple[str, str]:
    parser = configparser.RawConfigParser()
    parser.read(conf)
    access_key = parser.get("fj", "access_key", fallback="").strip()
    secret_key = parser.get("fj", "secret_key", fallback="").strip()
    if not access_key or not secret_key:
        raise RuntimeError(f"unable to read fj credentials from {conf}")
    return access_key, secret_key


def redact(value: str, access_key: str, secret_key: str) -> str:
    cleaned = value.replace(access_key, "<redacted>").replace(secret_key, "<redacted>")
    return re.sub(r"s3://[^/@:\s]+:[^/@\s]+@", "s3://<redacted>@", cleaned)


def update_attempt(entry: dict[str, Any], operation: str, status: str, reason: str | None = None) -> None:
    attempt = {"timestamp": utc_now(), "operation": operation, "status": status}
    if reason:
        attempt["reason"] = reason
    entry.setdefault("attempts", []).append(attempt)


def _stage_one_locked(args: argparse.Namespace) -> dict[str, Any]:
    ledger = load_ledger(args.ledger)
    entry = scene_entry(ledger, args.scene_id)
    ready = args.cache_root / "ready" / args.scene_id
    if ready.is_dir():
        validation = validate_scene(ready)
        if validation["status"] == "valid":
            entry["download_status"] = "ready"
            entry["asset_validation"] = validation
            entry["local_cache"] = str(ready)
            update_attempt(entry, "resume_validate", "success")
            ledger["updated_at"] = utc_now()
            atomic_json(args.ledger, ledger)
            return entry
        raise RuntimeError(f"existing ready cache is invalid; refusing overwrite: {ready}")
    staging_root = args.cache_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{args.scene_id}.", dir=staging_root))
    access_key, secret_key = credentials(args.conf)
    source = (
        f"s3://{access_key}:{secret_key}@{args.bucket}.{args.endpoint}/"
        f"{args.prefix.strip('/')}/{args.scene_id}/"
    )
    command = [
        str(args.ads_cli),
        "--quiet",
        "--threads",
        str(args.threads),
        "--listers",
        str(args.listers),
        "--conntimeout",
        str(args.connect_timeout),
        "--timeout",
        str(args.io_timeout),
        "sync",
        source,
        f"{staging}/",
    ]
    entry["download_status"] = "downloading"
    update_attempt(entry, "download", "started")
    ledger["updated_at"] = utc_now()
    atomic_json(args.ledger, ledger)
    log_path = args.log_dir / f"{args.scene_id}.download.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=args.process_timeout, check=False)
        safe_log = redact(result.stdout + result.stderr, access_key, secret_key)
        log_path.write_text(safe_log, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"ads-cli exited {result.returncode}; see {log_path}")
        validation = validate_scene(staging)
        if validation["status"] != "valid":
            raise RuntimeError(f"asset validation failed: {validation['missing'] + validation['empty'] + validation['errors']}")
        ready.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, ready)
        entry["download_status"] = "ready"
        entry["asset_validation"] = validation
        entry["local_cache"] = str(ready)
        entry["download_log"] = str(log_path)
        update_attempt(entry, "download", "success")
    except Exception as exc:
        entry["download_status"] = "failed"
        entry["asset_validation"] = {"status": "failed", "reason": str(exc)}
        update_attempt(entry, "download", "failed", str(exc))
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        ledger["updated_at"] = utc_now()
        atomic_json(args.ledger, ledger)
    return entry


def stage_one(args: argparse.Namespace) -> dict[str, Any]:
    with ledger_lock(args.ledger):
        return _stage_one_locked(args)


def _mark_locked(args: argparse.Namespace) -> dict[str, Any]:
    ledger = load_ledger(args.ledger)
    entry = scene_entry(ledger, args.scene_id)
    field = "renderer_smoke_status" if args.kind == "renderer-smoke" else "processing_status"
    entry[field] = args.status
    entry[f"{field}_updated_at"] = utc_now()
    if args.evidence:
        entry[f"{field}_evidence"] = str(args.evidence)
    update_attempt(entry, args.kind, args.status, args.reason)
    ledger["updated_at"] = utc_now()
    atomic_json(args.ledger, ledger)
    return entry


def mark(args: argparse.Namespace) -> dict[str, Any]:
    with ledger_lock(args.ledger):
        return _mark_locked(args)


def _cleanup_one_locked(args: argparse.Namespace) -> dict[str, Any]:
    ledger = load_ledger(args.ledger)
    entry = scene_entry(ledger, args.scene_id)
    ready = args.cache_root / "ready" / args.scene_id
    if entry.get("asset_validation", {}).get("status") != "valid":
        raise RuntimeError("refusing cleanup before successful asset validation")
    if entry.get("processing_status") != "success":
        raise RuntimeError("refusing cleanup before processing_status=success")
    if entry.get("renderer_smoke_status") != "success":
        raise RuntimeError("refusing cleanup before renderer_smoke_status=success")
    if not args.evidence.is_file():
        raise RuntimeError(f"refusing cleanup without result evidence: {args.evidence}")
    if ready.is_dir():
        shutil.rmtree(ready)
    entry["download_status"] = "cleaned"
    entry["local_cache"] = None
    entry["cleanup_evidence"] = str(args.evidence.resolve())
    update_attempt(entry, "cleanup", "success")
    ledger["updated_at"] = utc_now()
    atomic_json(args.ledger, ledger)
    return entry


def cleanup_one(args: argparse.Namespace) -> dict[str, Any]:
    with ledger_lock(args.ledger):
        return _cleanup_one_locked(args)


def print_entry(entry: dict[str, Any]) -> None:
    safe = {
        key: entry.get(key)
        for key in (
            "scene_id",
            "splits",
            "aoss_uri",
            "download_status",
            "asset_validation",
            "renderer_smoke_status",
            "processing_status",
            "local_cache",
        )
    }
    print(json.dumps(safe, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-ledger")
    build.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--tsv-output", type=Path)
    build.add_argument("--bucket", default=DEFAULT_BUCKET)
    build.add_argument("--prefix", default=DEFAULT_PREFIX)

    stage = sub.add_parser("stage-one")
    stage.add_argument("--ledger", type=Path, required=True)
    stage.add_argument("--scene-id", required=True)
    stage.add_argument("--cache-root", type=Path, required=True)
    stage.add_argument("--log-dir", type=Path, required=True)
    stage.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    stage.add_argument("--ads-cli", type=Path, default=DEFAULT_ADS_CLI)
    stage.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    stage.add_argument("--bucket", default=DEFAULT_BUCKET)
    stage.add_argument("--prefix", default=DEFAULT_PREFIX)
    stage.add_argument("--threads", type=int, default=8)
    stage.add_argument("--listers", type=int, default=4)
    stage.add_argument("--connect-timeout", type=int, default=120)
    stage.add_argument("--io-timeout", type=int, default=600)
    stage.add_argument("--process-timeout", type=int, default=1800)

    marker = sub.add_parser("mark")
    marker.add_argument("--ledger", type=Path, required=True)
    marker.add_argument("--scene-id", required=True)
    marker.add_argument("--kind", choices=("renderer-smoke", "processing"), required=True)
    marker.add_argument("--status", choices=("pending", "success", "failed", "blocked"), required=True)
    marker.add_argument("--evidence", type=Path)
    marker.add_argument("--reason")

    cleanup = sub.add_parser("cleanup-one")
    cleanup.add_argument("--ledger", type=Path, required=True)
    cleanup.add_argument("--scene-id", required=True)
    cleanup.add_argument("--cache-root", type=Path, required=True)
    cleanup.add_argument("--evidence", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-ledger":
        ledger = build_ledger(args.sources, args.output, args.tsv_output, args.bucket, args.prefix)
        print(json.dumps({"required_scene_count": ledger["required_scene_count"], "output": str(args.output)}, indent=2))
    elif args.command == "stage-one":
        print_entry(stage_one(args))
    elif args.command == "mark":
        print_entry(mark(args))
    elif args.command == "cleanup-one":
        print_entry(cleanup_one(args))


if __name__ == "__main__":
    main()
