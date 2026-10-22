#!/usr/bin/env python3
"""Compare the two runs of the resume test (downstream/train/resume_test.sh): A trained without a break, B trained,
killed, and resumed from its full state (twice).  Both log every step (loss, learning rates) and a hash of every
batch's actions and instructions on rank 0.

    python3 downstream/train/resume_check.py ROOT          # writes ROOT/resume_check.json, prints it
    python3 downstream/train/resume_check.py ROOT A B      # runs ROOT/A and ROOT/B -> ROOT/resume_check_B.json

Checks:
  * the kill took effect (kill_effective): B's first launch stopped after the first resume's state and before the
    second's, so the steps in between were trained twice; the segment launch ended at its stop step (segment_stopped);
  * batches: for every batch number, B's hash (the last launch that drew it) equals A's -> the data position was
    restored (same samples in the same order after the resume);
  * learning rates: equal at every step -> the schedule continued;
  * losses: |B - A| per step after the resume, next to the same difference before the kill (both runs fresh from the
    same seed: what the GPUs' non-determinism alone gives);
  * full state: size on disk, save and load seconds (full_state_log.jsonl, launches.jsonl).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def launches_of_metrics(path: Path) -> list:
    """The metric rows split at the resume markers: [[rows of launch 1], [rows of launch 2], ...]."""
    out = [[]]
    for r in rows(path):
        if "resumed_from_step" in r:
            out.append([])
        elif "step" in r:
            out[-1].append(r)
    return out


def last_by(items, key) -> dict:
    out = {}
    for r in items:
        out[r[key]] = r
    return out


def stats(diffs: list) -> dict:
    if not diffs:
        return {"n": 0}
    a = [abs(x) for x in diffs]
    return {"n": len(a), "mean_abs": sum(a) / len(a), "max_abs": max(a)}


def check(root: Path, a_name: str = "RESUME-A", b_name: str = "RESUME-B") -> dict:
    a_run, b_run = root / a_name / "run", root / b_name / "run"
    a = last_by(launches_of_metrics(a_run / "metrics.jsonl")[0], "step")
    b_launches = launches_of_metrics(b_run / "metrics.jsonl")
    b = last_by([r for launch in b_launches for r in launch], "step")  # later launches win
    resumed_at = [r["resumed_from_step"] for r in rows(b_run / "metrics.jsonl") if "resumed_from_step" in r]
    first_resume = resumed_at[0] if resumed_at else None
    killed_at = max((r["step"] for r in b_launches[0]), default=None)

    da = last_by(rows(a_run / "batch_digests.jsonl"), "batch")
    db_all = rows(b_run / "batch_digests.jsonl")
    db = last_by(db_all, "batch")
    common = sorted(set(da) & set(db))
    batch_mismatch = [n for n in common if da[n]["digest"] != db[n]["digest"]]
    # batches drawn twice by B (before the kill and again after the resume) must hash the same both times
    seen, redrawn_mismatch = {}, []
    for r in db_all:
        if r["batch"] in seen and seen[r["batch"]] != r["digest"]:
            redrawn_mismatch.append(r["batch"])
        seen[r["batch"]] = r["digest"]

    steps = sorted(set(a) & set(b))
    lr_mismatch = [s for s in steps if a[s].get("lr") != b[s].get("lr")]
    before = [b_launches[0][i] for i in range(len(b_launches[0]))]
    before_diffs = [r["action_dit_loss"] - a[r["step"]]["action_dit_loss"] for r in before
                    if r["step"] in a and (first_resume is None or r["step"] <= first_resume)]
    after_diffs = [b[s]["action_dit_loss"] - a[s]["action_dit_loss"] for s in steps
                   if first_resume is not None and s > first_resume]
    mean_loss = sum(a[s]["action_dit_loss"] for s in steps) / max(1, len(steps))

    states = rows(a_run / "full_state_log.jsonl") + rows(b_run / "full_state_log.jsonl")
    loads = [r.get("resumed_from") for r in rows(b_run / "launches.jsonl") if r.get("resumed_from")]
    out = {
        "a_steps": len(a), "b_steps": len(b), "b_launches": len(b_launches), "b_killed_after_step": killed_at,
        "b_resumed_from": resumed_at,
        "batches_compared": len(common), "batch_mismatches": batch_mismatch[:20], "batch_mismatch_count": len(batch_mismatch),
        "redrawn_batches": len(db_all) - len(db), "redrawn_mismatches": redrawn_mismatch[:20],
        "lr_steps_compared": len(steps), "lr_mismatch_steps": lr_mismatch[:20], "lr_mismatch_count": len(lr_mismatch),
        "loss_diff_before_kill": stats(before_diffs), "loss_diff_after_resume": stats(after_diffs),
        "mean_loss": mean_loss,
        "full_state": [{k: r.get(k) for k in ("completed_steps", "bytes", "save_seconds", "removed")} for r in states],
        "loads": [{k: x.get(k) for k in ("step", "batches", "skip", "load_seconds", "lr_after_load", "lr_saved")}
                  for x in loads],
    }
    out["data_continues"] = bool(common) and not batch_mismatch and not redrawn_mismatch
    out["schedule_continues"] = bool(steps) and not lr_mismatch and all(x["lr_after_load"] == x["lr_saved"] for x in out["loads"])
    # the kill took effect: the first launch stopped before the second full state, and the first resume went back to
    # an earlier state (the steps in between were trained again); the segment launch ended at its stop step
    seg_last = max((r["step"] for r in b_launches[1]), default=None) if len(b_launches) > 1 else None
    out["second_launch_last_step"] = seg_last
    out["kill_effective"] = bool(resumed_at) and killed_at is not None and resumed_at[0] < killed_at and (
        len(resumed_at) < 2 or killed_at < resumed_at[1])
    out["segment_stopped"] = len(resumed_at) >= 2 and seg_last == resumed_at[1]
    eps = root / "eval_P96/out/episodes"
    if (eps / "episodes.jsonl").is_file():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from full_state import speed_of

        try:
            out["eval_96"] = speed_of(eps)
        except SystemExit as exc:
            out["eval_96"] = {"error": str(exc)}
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0])
    names = argv[1:3] if len(argv) >= 3 else []
    out = check(root, *names)
    (root / (f"resume_check_{names[1]}.json" if names else "resume_check.json")).write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["data_continues"] and out["schedule_continues"] else 1


if __name__ == "__main__":
    sys.exit(main())
