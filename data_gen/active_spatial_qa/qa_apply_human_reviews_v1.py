#!/usr/bin/env python3
"""Apply a user's downloaded review JSON to a new, immutable QA manifest.

This tool never creates human decisions. It verifies the source manifest and
reviewed image bytes, preserves the supplied reviews, and changes eligibility
only according to explicit, complete reviews. No model or rendering is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

SCHEMA = "active_spatial_human_review_application_v1"
REVIEW_SCHEMA = "active_spatial_human_review_v1"
PENDING = "待审"
ACCEPT = "通过"
REJECT = {"坏图", "目标不明确", "题意不明确", "标签错误", "证据不足"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_rows(records, name):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name}: expected a nonempty list")
    result = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"{name}: record is not an object")
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in result:
            raise ValueError(f"{name}: missing or duplicate sample_id: {sid!r}")
        result[sid] = row
    return result


def goal_id(row):
    return str(row.get("goal_id") or row.get("parent_goal_id") or row.get("source_task_id") or "")


def validate_review(review, source_row):
    sid = source_row["sample_id"]
    if review.get("scene_id") != source_row.get("scene_id") or review.get("goal_id") != goal_id(source_row):
        raise ValueError(f"{sid}: review scene/goal identity mismatch")
    status = review.get("status")
    if status not in {PENDING, ACCEPT, *REJECT}:
        raise ValueError(f"{sid}: unsupported human review label: {status!r}")
    if review.get("reviewer_type") not in {None, "human"}:
        raise ValueError(f"{sid}: a nonhuman review cannot authorize release")
    if status != PENDING:
        for key in ("reviewer", "reviewed_at"):
            if not isinstance(review.get(key), str) or not review[key].strip():
                raise ValueError(f"{sid}: {status} requires nonempty {key}")
    if review.get("reason") is not None and not isinstance(review["reason"], str):
        raise ValueError(f"{sid}: reason must be text or empty")
    if review.get("reviewed_at"):
        if not isinstance(review["reviewed_at"], str):
            raise ValueError(f"{sid}: reviewed_at must be a timestamp string")
        try:
            timestamp = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{sid}: invalid reviewed_at timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"{sid}: reviewed_at must include a timezone")
    return status


def check_image(row, base, cache):
    sid = row["sample_id"]
    obs = row.get("public_observation", {})
    name = obs.get("image_path")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{sid}: reviewed RGB path is missing")
    path = Path(name)
    path = (path if path.is_absolute() else base / path).resolve(strict=True)
    expected_values = [value for value in (
        row.get("render", {}).get("sha256"), row.get("image_sha256"),
        obs.get("sha256"), obs.get("image_sha256")) if value is not None]
    if not expected_values or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in expected_values):
        raise ValueError(f"{sid}: original reviewed RGB SHA256 is missing or malformed")
    expected = {value.lower() for value in expected_values}
    if len(expected) != 1:
        raise ValueError(f"{sid}: conflicting RGB SHA256 fields")
    if str(path) not in cache:
        actual = sha(path)
        with Image.open(path) as image:
            image.load()
            cache[str(path)] = {"path": str(path), "sha256": actual, "size": list(image.size), "mode": image.mode}
    evidence = cache[str(path)]
    if evidence["sha256"] != next(iter(expected)):
        raise ValueError(f"{sid}: reviewed RGB SHA256 changed")
    return evidence


def load_candidate_gate():
    path = Path(__file__).resolve().with_name("qa_production_export_review.py")
    if not path.is_file():
        raise FileNotFoundError(f"Install the matching export tool beside this script: {path}")
    spec = importlib.util.spec_from_file_location("paired_qa_release_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.candidate_reasons, {"path": str(path), "sha256": sha(path)}


def release_reasons(row, candidate_gate, image):
    reasons = list(candidate_gate(row))
    if row.get("geometry", {}).get("reasons"):
        reasons.append("geometry_contains_failure_reasons")
    render = row.get("render", {})
    if render.get("ok") is not True:
        reasons.append("render_not_verified_successful")
    if render.get("stats", {}).get("nonblank") is not True:
        reasons.append("render_nonblank_not_verified")
    if image["size"] != [256, 256]:
        reasons.append("image_not_256_square_for_frozen_contract")
    desired = row.get("private_answer") == "Yes"
    metrics = row.get("metrics", {})
    for name in ("active", "independent"):
        success = metrics.get(name, {}).get("success")
        if type(success) is not bool or success != desired:
            reasons.append(name + "_label_agreement_missing")
    if metrics.get("independent", {}).get("admission", {}).get("geometry_admissible") is not True:
        reasons.append("independent_geometry_admission_missing")
    return sorted(set(reasons))


def apply(manifest, reviews_file, output):
    source, reviews_file = Path(manifest).resolve(strict=True), Path(reviews_file).resolve(strict=True)
    output = Path(output).resolve()
    audit_path = output.with_name(output.stem + ".review_application.json")
    raw_reviews_path = output.with_name(output.stem + ".human_reviews.json")
    for path in (output, audit_path, raw_reviews_path):
        if path in {source, reviews_file} or path.exists():
            raise FileExistsError(f"Refusing to overwrite an input or existing output: {path}")
    source_bytes, reviews_bytes = source.read_bytes(), reviews_file.read_bytes()
    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    source_index = index_rows(source_rows, "manifest")
    reviews = json.loads(reviews_bytes)
    if reviews.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Unexpected review JSON schema_version")
    source_sha, reviews_sha = hashlib.sha256(source_bytes).hexdigest(), hashlib.sha256(reviews_bytes).hexdigest()
    if reviews.get("manifest_sha256") != source_sha:
        raise ValueError("Review manifest_sha256 does not match the exact input manifest")
    review_index = index_rows(reviews.get("records"), "reviews")
    if source_index.keys() != review_index.keys():
        raise ValueError(f"Review sample IDs must exactly cover the manifest; missing={sorted(source_index.keys() - review_index.keys())}, extra={sorted(review_index.keys() - source_index.keys())}")
    candidate_gate, gate_identity = load_candidate_gate()
    scene_splits, cache, validated, counts = {}, {}, [], Counter()
    for row in source_rows:
        sid = row["sample_id"]
        scene, split = row.get("scene_id"), row.get("split")
        if not isinstance(scene, str) or not scene or split not in {"train", "val", "test"}:
            raise ValueError(f"{sid}: missing scene or frozen split")
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError("A scene crosses source partitions")
        scene_splits[scene] = split
        review = review_index[sid]
        status = validate_review(review, row)
        image = check_image(row, source.parent, cache)
        reasons = release_reasons(row, candidate_gate, image)
        if status == ACCEPT and reasons:
            raise ValueError(f"{sid}: human acceptance cannot override failed candidate/automatic checks: {reasons}")
        validated.append((row, review, status, image, reasons))
        counts[status] += 1
    applied_at = datetime.now(timezone.utc).isoformat()
    result, audit = [], []
    for original, review, status, image, reasons in validated:
        row = copy.deepcopy(original)
        original_candidate = original.get("candidate_qa_allowed", original.get("qa_sft_allowed")) is True
        approved = status == ACCEPT
        row["candidate_qa_allowed"] = original_candidate if status == PENDING else approved
        row["qa_sft_allowed"] = approved
        row["human_review"] = {**copy.deepcopy(review), "reviewer_type": "human"}
        row["semantic_review"] = {
            "status": "PASS" if approved else "pending" if status == PENDING else "FAIL",
            "review_label": status, "reviewer_type": "human",
            "reviewer": review.get("reviewer"), "reason": review.get("reason"),
            "reviewed_at": review.get("reviewed_at"),
        }
        row["human_review_application"] = {
            "schema_version": SCHEMA, "applied_at": applied_at,
            "source_manifest": str(source), "source_manifest_sha256": source_sha,
            "review_file": str(reviews_file), "review_file_sha256": reviews_sha,
            "preserved_review_copy": str(raw_reviews_path),
            "reviewed_image_sha256": image["sha256"], "automatic_gate": gate_identity,
            "previous_candidate_qa_allowed": original_candidate,
            "previous_qa_sft_allowed": original.get("qa_sft_allowed") is True,
            "previous_human_review": copy.deepcopy(original.get("human_review")),
        }
        # Changing output directory must not change which RGB the model sees.
        row["public_observation"]["image_path"] = image["path"]
        result.append(row)
        audit.append({"sample_id": row["sample_id"], "review_status": status, "image": image,
                      "candidate_qa_allowed": row["candidate_qa_allowed"], "qa_sft_allowed": approved,
                      "automatic_admission_reasons": reasons})
    # Serialize and recheck immutable input bytes before creating any outputs.
    rendered = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in result)
    if sha(source) != source_sha or sha(reviews_file) != reviews_sha:
        raise ValueError("An input file changed during validation")
    for image in cache.values():
        if sha(image["path"]) != image["sha256"]:
            raise ValueError("A reviewed RGB file changed during validation")
    summary = {
        "schema_version": SCHEMA, "applied_at": applied_at,
        "source_manifest": str(source), "source_manifest_sha256": source_sha,
        "review_file": str(reviews_file), "review_file_sha256": reviews_sha,
        "output_manifest": str(output), "output_manifest_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "preserved_human_reviews": str(raw_reviews_path), "records": len(result), "review_counts": dict(counts),
        "candidate_qa_allowed": sum(row["candidate_qa_allowed"] for row in result),
        "qa_sft_allowed": sum(row["qa_sft_allowed"] for row in result),
        "reason_required": False,
        "reason_analysis_performed_by_this_tool": False,
        "reasons_for_analysis": [{"sample_id": r["sample_id"], "status": r["status"], "reason": r["reason"]} for r in review_index.values() if isinstance(r.get("reason"), str) and r["reason"].strip()],
        "human_review_authorship": "User-supplied local review records; this utility does not authenticate a person's identity or create decisions",
        "no_rendering": True, "no_model_loaded": True, "no_training": True, "records_audit": audit,
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, audit_path, raw_reviews_path):
        if path.exists():
            raise FileExistsError(f"Output appeared during validation: {path}")
    created = []
    try:
        for path, data in ((output, rendered.encode()), (audit_path, summary_text.encode()), (raw_reviews_path, reviews_bytes)):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="The exact assistant-reviewed manifest used to build review.html")
    parser.add_argument("--reviews", required=True, help="User-downloaded human_review JSON")
    parser.add_argument("--output", required=True, help="A new version's manifest JSONL; existing files are refused")
    args = parser.parse_args()
    result = apply(args.manifest, args.reviews, args.output)
    print(json.dumps({key: result[key] for key in ("output_manifest", "records", "review_counts", "candidate_qa_allowed", "qa_sft_allowed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
