#!/usr/bin/env python3
"""Create a blinded, stratified FOV RGB review pack without making labels.

The output deliberately separates a reviewer-facing manifest from internal
provenance.  The former contains no heuristic verdict, score, or rejection
reason.  It is an annotation *request*, never a substitute for human labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VERSION = "r1_fov_blinded_manual_review_pack_v1_20260928"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def stable_rank(row: dict[str, Any]) -> str:
    key = "|".join(str(row.get(name, "")) for name in ("population", "scene_id", "source_row_index", "task_id", "image_root"))
    return hashlib.sha256(key.encode()).hexdigest()


def choose_stratified(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Round-robin deterministic strata so common scenes never dominate."""
    by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row.get("scene_id")), str(row.get("category_pair")), str(row.get("reason_group")))].append(row)
    for values in by_stratum.values():
        values.sort(key=stable_rank)
    result: list[dict[str, Any]] = []
    while len(result) < count:
        advanced = False
        for key in sorted(by_stratum):
            if by_stratum[key] and len(result) < count:
                result.append(by_stratum[key].pop(0))
                advanced = True
        if not advanced:
            break
    return result


def category_pair(item: dict[str, Any]) -> str:
    objects = item.get("target_object", {}).get("objects") or []
    return "--".join(sorted(str(obj.get("label") or "unknown") for obj in objects))


def pilot_cases(candidates: Path, reachability: Path) -> list[dict[str, Any]]:
    reaches = {str(row.get("task_id")): row for row in read_jsonl(reachability)}
    image_root = reachability.parent / "path_rgb"
    cases: list[dict[str, Any]] = []
    for item in read_jsonl(candidates):
        task_id = str(item.get("task_id"))
        path = image_root / task_id
        if not (path / "step_00_initial.png").is_file():
            raise RuntimeError(f"missing frozen pilot RGB for {task_id}")
        cases.append({
            "population": "historical_manual_positive", "known_historical_label": "PASS",
            "scene_id": str(item.get("scene_id")), "source_row_index": None,
            "task_id": task_id, "category_pair": category_pair(item), "reason_group": "manual_positive",
            "path_steps": reaches.get(task_id, {}).get("steps"), "image_root": str(path),
        })
    if len(cases) != 15:
        raise RuntimeError(f"expected 15 historical manual positives, got {len(cases)}")
    return cases


def audit_cases(root: Path, population: str, require_pass: bool | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for manifest in sorted(root.glob("*/official_observability_v2/observability_manifest.jsonl")):
        for audit in read_jsonl(manifest):
            if require_pass is not None and bool(audit.get("passed")) != require_pass:
                continue
            image_root = audit.get("contact_sheet") or audit.get("frame_images", [None])[0]
            if not image_root:
                # v2 stores the contact sheet under the task directory even
                # when the manifest only has the directory fields.
                candidate = manifest.parent / "path_rgb" / str(audit.get("task_id"))
                image_root = str(candidate)
            image_path = Path(image_root)
            if image_path.is_file():
                image_path = image_path.parent
            if not (image_path / "step_00_initial.png").is_file():
                continue
            cases.append({
                "population": population, "known_historical_label": None,
                "scene_id": str(audit.get("scene_id")), "source_row_index": audit.get("source_row_index"),
                "task_id": str(audit.get("task_id")), "category_pair": "unknown",
                "reason_group": "v2_pass" if audit.get("passed") else "v2_reject:" + "+".join(sorted(audit.get("reasons") or ["unknown"])),
                "path_steps": audit.get("path_steps"), "image_root": str(image_path),
            })
    return cases


def legacy_cases(manifest: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for audit in read_jsonl(manifest):
        image = Path(str(audit.get("contact_sheet") or ""))
        image_root = image.parent if image.is_file() else image
        if not (image_root / "step_00_initial.png").is_file():
            continue
        cases.append({
            "population": "legacy_projective_style_screen", "known_historical_label": None,
            "scene_id": str(audit.get("scene_id")), "source_row_index": audit.get("source_row_index"),
            "task_id": str(audit.get("task_id")), "category_pair": "unknown",
            "reason_group": "legacy_pass" if audit.get("passed") else "legacy_reject:" + "+".join(sorted(audit.get("reasons") or ["unknown"])),
            "path_steps": audit.get("path_steps"), "image_root": str(image_root),
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-positive-candidates", type=Path, required=True)
    parser.add_argument("--manual-positive-reachability", type=Path, required=True)
    parser.add_argument("--salvage-fov-shards", type=Path, required=True)
    parser.add_argument("--legacy-observability", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provisional-count", type=int, default=35)
    parser.add_argument("--legacy-reject-count", type=int, default=40)
    parser.add_argument("--v2-reject-count", type=int, default=15)
    args = parser.parse_args()

    positives = pilot_cases(args.manual_positive_candidates, args.manual_positive_reachability)
    provisional = audit_cases(args.salvage_fov_shards, "salvage_v2_provisional_pass", True)
    v2_rejects = audit_cases(args.salvage_fov_shards, "salvage_v2_reject", False)
    legacy = legacy_cases(args.legacy_observability)
    legacy_rejects = [row for row in legacy if row["reason_group"] != "legacy_pass"]

    selected = positives + choose_stratified(provisional, args.provisional_count) + choose_stratified(legacy_rejects, args.legacy_reject_count) + choose_stratified(v2_rejects, args.v2_reject_count)
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selected:
        key = (str(row.get("scene_id")), str(row.get("source_row_index")), str(row.get("image_root")))
        unique.setdefault(key, row)
    selected = list(unique.values())
    selected.sort(key=stable_rank)
    if not 80 <= len(selected) <= 120:
        raise RuntimeError(f"review pack must be 80--120 deduplicated cases, got {len(selected)}")

    metadata, blinded = [], []
    for index, row in enumerate(selected, start=1):
        case_id = f"FOV-R1-{index:03d}"
        frames = sorted(str(path) for path in Path(row["image_root"]).glob("step_*.png"))
        metadata.append({**row, "case_id": case_id, "frame_paths": frames, "version": VERSION})
        blinded.append({"case_id": case_id, "review_frames": frames, "review_instruction": "Judge visual solvability from RGB only."})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "review_manifest_blinded.jsonl", blinded)
    write_jsonl(args.output_dir / "case_metadata_internal.jsonl", metadata)
    with (args.output_dir / "annotations_template.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "verdict", "optional_reason", "reviewer_id", "reviewed_utc"])
        writer.writeheader()
        for row in blinded:
            writer.writerow({"case_id": row["case_id"], "verdict": "", "optional_reason": "", "reviewer_id": "", "reviewed_utc": ""})
    protocol = """# FOV RGB manual review protocol\n\nFor each blinded case, inspect the supplied official RGB frames only.  Record:\n\n- `PASS`: both designated target objects and the FOV-inclusion task are visually discernible;\n- `FAIL`: RGB makes either target or the inclusion relation not visually solvable;\n- `UNCERTAIN`: insufficient confidence.\n\nDo **not** judge canonical success, action optimality, collision, annotations, or program output.  Those remain formal runtime/metric decisions.  Do not infer a verdict from filenames or metadata.\n"""
    (args.output_dir / "REVIEW_PROTOCOL.md").write_text(protocol)
    summary = {"version": VERSION, "cases": len(metadata), "by_population": dict(sorted(Counter(row["population"] for row in metadata).items())), "by_reason_group": dict(sorted(Counter(row["reason_group"] for row in metadata).items())), "human_labels_present": False, "annotation_values": ["PASS", "FAIL", "UNCERTAIN"], "inputs": {name: str(getattr(args, name)) for name in ("manual_positive_candidates", "manual_positive_reachability", "salvage_fov_shards", "legacy_observability")}}
    write_json(args.output_dir / "review_pack_summary.json", summary)
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.output_dir / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
