"""Training data for InteriorGS v10 (user decisions of 2026-10-14).

SFT, first version: the formal tasks of train (training data) and val (validation data), one multi-turn
conversation per task along the shortest path the task certifies, actions only, no QA mixed in; test is
not used.  The conversations are built by ``release.build_sft_record`` (the RL env's system prompt, the
"Current camera pose" line in every user turn, at most three actions per turn).

    # CPU: frames to render (turn starts and the final state), published rows, task registry, split manifest
    python -m data_gen.interiorgs_v10.training_prep sft-frames --run OUT --split-manifest SPLIT --out SFT
    # GPU (one card, zoetrope): render the frames
    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot SFT --gs-root PLY_ROOT   (then submit / status / delete)
    # CPU: the conversations
    python -m data_gen.interiorgs_v10.training_prep sft-records --dir SFT
    # CPU: LLaMA-Factory datasets and recipe (no QA; trajectories are not model-reviewed this round)
    python scripts/prepare_active_spatial_sft.py --trajectory SFT/sft_trajectories.jsonl \\
        --split-manifest SFT/split_manifest.json --task-registry SFT/task_registry.jsonl \\
        --qa-probability 0 --allow-unreviewed-trajectories --model MODEL --out LF
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SFT_SPLITS = ("train", "val")


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def sft_states(row: Dict[str, Any], policy: Dict[str, Any]) -> List[int]:
    """State indices whose frames a conversation needs: every turn start and the final state."""
    from .admission import turn_boundaries

    return sorted(set(turn_boundaries(row, policy)))


def sft_frames(rows: Sequence[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Frames in the render job's format (pilot.py render), plain images only (no overlay)."""
    from .pilot import camera_item
    from .qa_states import certificate_poses

    out = []
    for row in rows:
        poses = certificate_poses(row)
        for k in sft_states(row, policy):
            out.append({"frame_id": f"{row['task_id']}__s{k}", "kind": "sft", "task_id": row["task_id"],
                        "scene_id": row["scene_id"], "state_id": f"s{k}", "c2w": np.asarray(poses[k]).tolist(),
                        "camera_item": camera_item(row), "overlay": {}})
    return out


def cmd_sft_frames(args: argparse.Namespace) -> None:
    from .admission import load_policy
    from .release import finalize_task_row, registry_row, split_lookup

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.split_manifest).read_text())
    lookup = split_lookup(manifest)
    splits = tuple(s for s in args.splits.split(",") if s)
    policy = load_policy()
    rows, skipped = [], Counter()
    for raw in _jsonl(Path(args.run) / "tasks.jsonl"):
        split = lookup.get(str(raw["scene_id"]))
        if split is None:
            skipped["scene_not_in_manifest"] += 1
            continue
        if split not in splits:
            skipped[f"split_{split}"] += 1
            continue
        rows.append(finalize_task_row(raw, split))
    if args.limit:
        rows = rows[: args.limit]
    frames = sft_frames(rows, policy)
    _write_jsonl(out / "rows.jsonl", rows)
    _write_jsonl(out / "frames.jsonl", frames)
    _write_jsonl(out / "task_registry.jsonl",
                 [registry_row(r, {"sft": True, "rl": r["split"] == "train"}) for r in rows])
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    report = {"tasks": len(rows), "by_split": dict(Counter(r["split"] for r in rows)),
              "by_family": dict(Counter(r["task_type"] for r in rows)), "frames": len(frames),
              "scenes": len({r["scene_id"] for r in rows}), "skipped": dict(skipped)}
    (out / "sft_frames_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_sft_records(args: argparse.Namespace) -> None:
    from .admission import load_policy
    from .pilot import safe_name
    from .release import build_sft_record

    d = Path(args.dir).resolve()
    policy = load_policy()
    rows = _jsonl(d / "rows.jsonl")
    images: Dict[str, str] = {}
    for f in _jsonl(d / "frames.jsonl"):
        path = d / "frames" / "plain" / safe_name(f["frame_id"])
        if path.is_file():
            images[f"{f['task_id']}:{f['state_id']}"] = str(path)
    records, reasons = [], Counter()
    for row in rows:
        rec, reason = build_sft_record(row, images, policy)
        if rec is None:
            reasons[reason] += 1
            continue
        records.append(rec)
    _write_jsonl(d / "sft_trajectories.jsonl", records)
    report = {"tasks": len(rows), "records": len(records), "dropped": dict(reasons),
              "by_split": dict(Counter(r["split"] for r in records)),
              "turns": dict(Counter(r["trajectory_steps"] for r in records)),
              "images": len(images)}
    (d / "sft_records_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("sft-frames")
    f.add_argument("--run", required=True, help="run_all output directory (tasks.jsonl = formal tasks)")
    f.add_argument("--split-manifest", required=True)
    f.add_argument("--splits", default=",".join(SFT_SPLITS))
    f.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke tests)")
    f.add_argument("--out", required=True)
    r = sub.add_parser("sft-records")
    r.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    {"sft-frames": cmd_sft_frames, "sft-records": cmd_sft_records}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
