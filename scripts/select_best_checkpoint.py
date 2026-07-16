#!/usr/bin/env python3
"""Select the best validation checkpoint and emit early-stop signals.

Rules used by default:
- best checkpoint: highest ID mean@N (ID_m4 with val_n=4), among steps with a checkpoint
- stop signal: latest train entropy > 1.0, or ID_m4 drops at two consecutive val points

The script writes early_stop_decision.json under the experiment directory and
updates checkpoints/best_auto as a symlink to the selected checkpoint when possible.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


DEFAULT_EXPDIR = Path("exps/vagen_active_spatial")


def parse_float(raw: str) -> float:
    if raw.startswith("np.float64(") and raw.endswith(")"):
        raw = raw[len("np.float64("):-1]
    return float(raw)


def parse_train_entropy(train_log: Path) -> list[tuple[int, float]]:
    if not train_log.is_file():
        return []
    rows = []
    value_pat = r"(np\.float64\([0-9.e+\-]+\)|[0-9.e+\-]+)"
    step_pat = re.compile(r"training/global_step:(\d+)")
    entropy_pat = re.compile(r"(?<![/\w])actor/entropy:" + value_pat)
    with train_log.open() as handle:
        for line in handle:
            step_match = step_pat.search(line)
            entropy_match = entropy_pat.search(line)
            if not step_match or not entropy_match:
                continue
            rows.append((int(step_match.group(1)), parse_float(entropy_match.group(1))))
    rows.sort(key=lambda item: item[0])
    return rows


def parse_val_file(path: Path, n_id: int, n_ood: int, val_n: int) -> dict:
    with path.open() as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    id_entries = lines[: n_id * val_n]
    ood_entries = lines[n_id * val_n : (n_id + n_ood) * val_n]

    def aggregate(entries: list[dict]) -> tuple[float | None, float | None, float | None]:
        best_values = []
        mean_values = []
        worst_values = []
        for index in range(0, len(entries), val_n):
            chunk = entries[index : index + val_n]
            if not chunk:
                continue
            successes = [float(row.get("traj_success", 0)) for row in chunk]
            best_values.append(max(successes))
            mean_values.append(sum(successes) / len(successes))
            worst_values.append(min(successes))
        if not best_values:
            return None, None, None
        return (
            sum(best_values) / len(best_values),
            sum(mean_values) / len(mean_values),
            sum(worst_values) / len(worst_values),
        )

    id_b, id_m, id_w = aggregate(id_entries)
    ood_b, ood_m, ood_w = aggregate(ood_entries)
    return {
        "id_b4": id_b,
        "id_m4": id_m,
        "id_w4": id_w,
        "ood_b4": ood_b,
        "ood_m4": ood_m,
        "ood_w4": ood_w,
        "total": len(lines),
    }


def parse_validation(exp_dir: Path, n_id: int, n_ood: int, val_n: int) -> list[dict]:
    val_dir = exp_dir / "validation"
    if not val_dir.is_dir():
        return []
    rows = []
    for path in sorted(val_dir.glob("*.jsonl"), key=lambda item: int(item.stem)):
        metrics = parse_val_file(path, n_id=n_id, n_ood=n_ood, val_n=val_n)
        metrics["step"] = int(path.stem)
        rows.append(metrics)
    return rows


def checkpoint_for_step(exp_dir: Path, step: int) -> Path:
    return exp_dir / "checkpoints" / f"global_step_{step}"


def choose_best_checkpoint(exp_dir: Path, val_rows: list[dict]) -> dict | None:
    candidates = []
    for row in val_rows:
        checkpoint = checkpoint_for_step(exp_dir, int(row["step"]))
        id_m4 = row.get("id_m4")
        if id_m4 is None or not checkpoint.is_dir():
            continue
        candidates.append((float(id_m4), float(row.get("id_b4") or 0), int(row["step"]), row, checkpoint))
    if not candidates:
        return None
    _, _, _, row, checkpoint = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {"row": row, "checkpoint": checkpoint}


def has_two_consecutive_drops(val_rows: list[dict], metric: str) -> bool:
    metric_rows = [row for row in val_rows if row.get(metric) is not None]
    if len(metric_rows) < 3:
        return False
    last_three = metric_rows[-3:]
    return float(last_three[1][metric]) < float(last_three[0][metric]) and float(last_three[2][metric]) < float(last_three[1][metric])


def update_best_symlink(best_link: Path, checkpoint: Path) -> str:
    if best_link.is_symlink() or not best_link.exists():
        if best_link.is_symlink():
            best_link.unlink()
        relative_target = os.path.relpath(checkpoint, start=best_link.parent)
        best_link.symlink_to(relative_target)
        return "updated"
    return "skipped_existing_non_symlink"


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Select best checkpoint and emit early-stop signals.")
    parser.add_argument("--exp", required=True, help="Experiment name under --expdir")
    parser.add_argument("--expdir", default=str(DEFAULT_EXPDIR), help="Experiment directory prefix")
    parser.add_argument("--n_id", type=int, default=18, help="Number of ID validation envs")
    parser.add_argument("--n_ood", type=int, default=25, help="Number of OOD validation envs")
    parser.add_argument("--val_n", type=int, default=4, help="Validation samples per env")
    parser.add_argument("--metric", default="id_m4", help="Validation metric used for drop detection")
    parser.add_argument("--entropy_threshold", type=float, default=1.0, help="Stop when latest entropy exceeds this value")
    parser.add_argument("--no_symlink", action="store_true", help="Do not update checkpoints/best_auto")
    parser.add_argument("--exit_nonzero_on_stop", action="store_true", help="Return exit code 2 when should_stop is true")
    args = parser.parse_args()

    exp_dir = Path(args.expdir) / args.exp
    val_rows = parse_validation(exp_dir, n_id=args.n_id, n_ood=args.n_ood, val_n=args.val_n)
    entropy_rows = parse_train_entropy(exp_dir / "train.log")
    best = choose_best_checkpoint(exp_dir, val_rows)

    reasons = []
    latest_entropy = entropy_rows[-1] if entropy_rows else None
    if latest_entropy and latest_entropy[1] > args.entropy_threshold:
        reasons.append(f"entropy>{args.entropy_threshold:g} at step {latest_entropy[0]} ({latest_entropy[1]:.4f})")
    if has_two_consecutive_drops(val_rows, args.metric):
        reasons.append(f"two_consecutive_{args.metric}_drops")

    symlink_status = "disabled"
    best_step = None
    best_checkpoint = None
    best_metrics = None
    if best:
        best_step = int(best["row"]["step"])
        best_checkpoint = str(best["checkpoint"])
        best_metrics = {key: finite_or_none(best["row"].get(key)) for key in ("id_b4", "id_m4", "id_w4", "ood_b4", "ood_m4", "ood_w4")}
        if not args.no_symlink:
            symlink_status = update_best_symlink(exp_dir / "checkpoints" / "best_auto", best["checkpoint"])

    latest_val = val_rows[-1] if val_rows else None
    decision = {
        "experiment": args.exp,
        "should_stop": bool(reasons),
        "reasons": reasons,
        "best_step": best_step,
        "best_checkpoint": best_checkpoint,
        "best_metrics": best_metrics,
        "latest_val": latest_val,
        "latest_entropy": {"step": latest_entropy[0], "value": latest_entropy[1]} if latest_entropy else None,
        "symlink_status": symlink_status,
    }

    output_path = exp_dir / "early_stop_decision.json"
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))

    if decision["should_stop"] and args.exit_nonzero_on_stop:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())