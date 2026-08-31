#!/usr/bin/env python3
"""Merge ViewSpatial shard samples and write an EASI-compatible results JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

N_SAMPLES = 5712
CKPTS = [
    "qwen_pretrain_baseline",
    "v46_step200",
    "v50_step150",
    "cambrian_pretrain_baseline",
    "c8_step300",
]
SUBMETRICS = [
    "camera_perspective_relative_direction_accuracy",
    "camera_perspective_object_view_orientation_accuracy",
    "person_perspective_relative_direction_accuracy",
    "person_perspective_object_view_orientation_accuracy",
    "person_perspective_scene_simulation_relative_direction_accuracy",
]
TARGET_TYPES = {
    "camera_perspective_relative_direction_accuracy": "Camera perspective - Relative Direction",
    "camera_perspective_object_view_orientation_accuracy": "Camera perspective - Object View Orientation",
    "person_perspective_relative_direction_accuracy": "Person perspective - Relative Direction",
    "person_perspective_object_view_orientation_accuracy": "Person perspective - Object View Orientation",
    "person_perspective_scene_simulation_relative_direction_accuracy": "Person perspective - Scene Simulation Relative Direction",
}


def find_sanitized_dir(ckpt_dir: Path) -> Path:
    # Prefer existing non-shard model dirs used by prior EASI runs.
    cands = [
        p
        for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name not in {"shards", "shard_logs"} and not p.name.startswith(".")
    ]
    # Keep dirs that already contain *_results.json
    with_results = [p for p in cands if any(p.glob("*_results.json"))]
    if with_results:
        return sorted(with_results)[0]
    if cands:
        return sorted(cands)[0]
    d = ckpt_dir / "viewspatial_merged"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_shard_samples(ckpt_dir: Path) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    shard_root = ckpt_dir / "shards"
    files = sorted(shard_root.glob("viewspatial_*/**/*samples_viewspatial.jsonl"))
    if not files:
        files = sorted(shard_root.rglob("*samples_viewspatial.jsonl"))
    for path in files:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                doc_id = int(item["doc_id"])
                by_id[doc_id] = item
    return by_id


def aggregate(samples: dict[int, dict]) -> dict[str, float]:
    overall_scores = []
    typed: dict[str, list[float]] = defaultdict(list)
    for item in samples.values():
        oa = item.get("overall_accuracy")
        if isinstance(oa, dict) and "score" in oa:
            overall_scores.append(float(oa["score"]))
        qtype = None
        for metric in SUBMETRICS:
            blob = item.get(metric)
            if isinstance(blob, dict):
                qtype = blob.get("question_type") or qtype
        if qtype is None:
            continue
        for metric, target in TARGET_TYPES.items():
            blob = item.get(metric)
            if not isinstance(blob, dict):
                continue
            if blob.get("question_type") == target or blob.get("target_type") == target:
                typed[metric].append(float(blob.get("score", 0.0)))
    out = {
        "overall_accuracy": (sum(overall_scores) / len(overall_scores)) if overall_scores else None,
    }
    for metric in SUBMETRICS:
        vals = typed.get(metric) or []
        out[metric] = (sum(vals) / len(vals)) if vals else None
    return out


def write_outputs(ckpt_dir: Path, samples: dict[int, dict], metrics: dict[str, float]) -> Path:
    model_dir = find_sanitized_dir(ckpt_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    samples_path = model_dir / f"{stamp}_samples_viewspatial.jsonl"
    results_path = model_dir / f"{stamp}_results.json"
    with samples_path.open("w") as fh:
        for doc_id in sorted(samples):
            fh.write(json.dumps(samples[doc_id], ensure_ascii=False) + "\n")
    results = {
        "results": {
            "viewspatial": {
                "alias": "viewspatial",
                **{f"{k},none": v for k, v in metrics.items()},
            }
        },
        "n-samples": {"viewspatial": {"original": N_SAMPLES, "effective": len(samples)}},
        "config": {"merged_from_shards": True, "task": "viewspatial"},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[write] {results_path} n={len(samples)} metrics={metrics}")
    return results_path


def rebuild_easi_payload(ckpt_dir: Path) -> None:
    # Soft rebuild via easi runner extract path if available; else patch scores.
    easi_path = ckpt_dir / "easi_results.json"
    payload = {}
    if easi_path.exists():
        try:
            payload = json.loads(easi_path.read_text())
        except json.JSONDecodeError:
            payload = {}
    # Prefer reading through EASI helper
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/EASI/scripts/submissions"))
        from backends.lmmseval import LmmsEvalBackend

        backend = LmmsEvalBackend()
        # discover sanitized model dir
        model_dir = find_sanitized_dir(ckpt_dir)
        scores = backend.extract_scores(model_dir, model_name="model", benchmarks={"viewspatial": "viewspatial"})
        vs = scores.get("viewspatial")
        payload.setdefault("scores", {})
        payload.setdefault("subScores", {})
        if vs is not None:
            payload["scores"]["viewspatial"] = vs.overall
            if vs.sub_scores:
                payload["subScores"]["viewspatial"] = vs.sub_scores
            easi_path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"[easi] {easi_path} viewspatial={vs.overall}")
            return
    except Exception as exc:
        print(f"[warn] EASI extract failed: {exc}")
    # fallback from newest results file that actually contains viewspatial
    model_dir = find_sanitized_dir(ckpt_dir)
    candidates = []
    for path in model_dir.glob("*_results.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "viewspatial" in (data.get("results") or {}):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no viewspatial *_results.json under {model_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    metrics = json.loads(latest.read_text())["results"]["viewspatial"]
    overall = metrics.get("overall_accuracy,none")
    payload.setdefault("scores", {})
    payload["scores"]["viewspatial"] = None if overall is None else round(float(overall) * 100, 4)
    subs = {}
    for k in SUBMETRICS:
        v = metrics.get(f"{k},none")
        subs[k] = None if v is None else round(float(v) * 100, 4)
    payload.setdefault("subScores", {})
    payload["subScores"]["viewspatial"] = subs
    easi_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[easi-fallback] {easi_path} viewspatial={payload['scores']['viewspatial']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("exps/unified_eval_runs/easi_results"))
    parser.add_argument("--expect", type=int, default=N_SAMPLES)
    parser.add_argument("--ckpts", default=",".join(CKPTS))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    failed = 0
    for name in [x.strip() for x in args.ckpts.split(",") if x.strip()]:
        ckpt_dir = out / name
        samples = load_shard_samples(ckpt_dir)
        print(f"[{name}] loaded {len(samples)}/{args.expect} unique doc_ids")
        if len(samples) < args.expect:
            missing = sorted(set(range(args.expect)) - set(samples))
            print(f"[{name}] missing {len(missing)} e.g. {missing[:10]}")
            failed += 1
            continue
        metrics = aggregate(samples)
        write_outputs(ckpt_dir, samples, metrics)
        rebuild_easi_payload(ckpt_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
