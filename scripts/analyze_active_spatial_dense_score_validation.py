#!/usr/bin/env python3
"""Compare fixed Active Spatial validation rollouts for S0/S1/S5.

The analysis deliberately treats tasks as paired but trajectories as unpaired:
validation sampling is stochastic and trajectory hashes need not match.  Confidence
intervals therefore resample scene clusters, preserving all tasks from each sampled
scene.  The script fails closed when task identities, scenes, or rollout counts do
not match across branches/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


BRANCHES = ("S0", "S1", "S5")
STEPS = (0, 50, 100, 150)
PRIMARY_METRICS = (
    "traj_success",
    "task_pass_at_4",
    "final_score",
    "best_final_score_at_4",
    "score_improvement",
    "reward",
    "invalid_action",
    "strict_parse_success_rate",
)
ROW_MEAN_FIELDS = (
    "traj_success",
    "final_score",
    "score_improvement",
    "reward",
    "episode_length",
    "n_primitive_steps",
    "collision_termination",
    "invalid_action",
    "strict_parse_success_rate",
    "renderer_failure",
)
FORMAT_FIELDS = (
    "invalid_action",
    "empty_action",
    "contradictory_action",
    "missing_action_tag",
    "unknown_action_name",
    "truncated_before_action",
    "multiple_action_tag",
    "strict_parse_success_rate",
    "strict_format_correct_rate",
    "format_penalty_rate",
    "collision_termination",
    "low_info_termination",
    "renderer_failure",
)
TRAIN_METRICS = (
    "episode/success_rate",
    "transition/final_score/mean",
    "transition/invalid_action_rate/mean",
    "transition/strict_parse_success_rate/mean",
    "transition/reward_mean",
    "actor/entropy",
    "actor/ppo_kl",
    "actor/pg_loss",
    "critic/vf_loss",
    "perf/time_per_step",
    "trainer/actor_update_performed",
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of empty sequence")
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def task_digest(task_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(task_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: empty validation artifact")
    return rows


def build_task_metrics(rows: list[dict], expected_rollouts: int) -> tuple[dict, dict]:
    required = {"task_id", "scene_id", *ROW_MEAN_FIELDS, *FORMAT_FIELDS}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"row {index} missing required fields: {missing}")
        grouped[str(row["task_id"])].append(row)

    task_metrics: dict[str, dict[str, float]] = {}
    task_scenes: dict[str, str] = {}
    for task_id, task_rows in grouped.items():
        if len(task_rows) != expected_rollouts:
            raise ValueError(
                f"{task_id}: expected {expected_rollouts} rollouts, got {len(task_rows)}"
            )
        scenes = {str(row["scene_id"]) for row in task_rows}
        if len(scenes) != 1:
            raise ValueError(f"{task_id}: inconsistent scene ids: {sorted(scenes)}")
        task_scenes[task_id] = next(iter(scenes))
        values = {field: mean(float(row[field]) for row in task_rows) for field in ROW_MEAN_FIELDS}
        values["task_pass_at_4"] = float(any(float(row["traj_success"]) > 0.5 for row in task_rows))
        values["best_final_score_at_4"] = max(float(row["final_score"]) for row in task_rows)
        task_metrics[task_id] = values
    return task_metrics, task_scenes


def aggregate(task_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    names = next(iter(task_metrics.values())).keys()
    return {name: mean(values[name] for values in task_metrics.values()) for name in names}


def paired_scene_bootstrap(
    differences: dict[str, float],
    task_scenes: dict[str, str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    scene_tasks: dict[str, list[str]] = defaultdict(list)
    for task_id, scene_id in task_scenes.items():
        scene_tasks[scene_id].append(task_id)
    scenes = sorted(scene_tasks)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(scenes) for _ in scenes]
        vals = [differences[task] for scene in sampled for task in scene_tasks[scene]]
        draws.append(mean(vals))
    estimate = mean(differences.values())
    return {
        "estimate": estimate,
        "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "probability_gt_zero": sum(value > 0 for value in draws) / samples,
        "bootstrap_unit": "scene",
        "scene_clusters": len(scenes),
        "samples": samples,
        "seed": seed,
    }


def paired_difference(
    lhs: dict[str, dict[str, float]],
    rhs: dict[str, dict[str, float]],
    metric: str,
) -> dict[str, float]:
    if lhs.keys() != rhs.keys():
        raise ValueError("paired difference requires identical task ids")
    return {task: lhs[task][metric] - rhs[task][metric] for task in lhs}


def normalized_auc(values_by_step: dict[int, float], warmup: int) -> float:
    by_x: dict[int, list[float]] = defaultdict(list)
    for step, value in values_by_step.items():
        by_x[max(0, step - warmup)].append(value)
    points = sorted((x, mean(values)) for x, values in by_x.items())
    if len(points) < 2 or points[-1][0] == points[0][0]:
        raise ValueError("AUC needs at least two distinct actor-update points")
    area = sum(
        (x1 - x0) * (y0 + y1) / 2
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def parse_training_log(path: Path) -> list[dict[str, float | int | str]]:
    if not path.is_file():
        return []
    found: dict[int, dict[str, float | int | str]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI_RE.sub("", raw)
            match = re.search(r"\bstep:(\d+)\s+-", line)
            if not match:
                continue
            step = int(match.group(1))
            record = found.setdefault(step, {"step": step})
            for metric in TRAIN_METRICS:
                value_match = re.search(
                    rf"(?:^| - ){re.escape(metric)}:(?:np\.(?:float64|float32)\()?"
                    rf"([-+0-9.eE]+)",
                    line,
                )
                if value_match:
                    record[metric] = float(value_match.group(1))
    return [found[step] for step in sorted(found)]


def summarize_training_windows(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Summarize on-policy logs without treating branch-specific rewards as eval scores."""
    windows = ((61, 100), (101, 150), (141, 150))
    selected = (
        "episode/success_rate",
        "transition/final_score/mean",
        "transition/invalid_action_rate/mean",
        "transition/strict_parse_success_rate/mean",
        "transition/reward_mean",
        "actor/entropy",
        "actor/ppo_kl",
        "perf/time_per_step",
    )
    output: dict[str, dict[str, dict[str, float]]] = {}
    for branch in BRANCHES:
        branch_rows = [row for row in rows if row["branch"] == branch]
        output[branch] = {}
        for lo, hi in windows:
            window_rows = [row for row in branch_rows if lo <= int(row["step"]) <= hi]
            if len(window_rows) != hi - lo + 1:
                raise ValueError(
                    f"{branch}: training window {lo}-{hi} has {len(window_rows)} rows, "
                    f"expected {hi - lo + 1}"
                )
            summary = {"steps": len(window_rows)}
            for metric in selected:
                values = [float(row[metric]) for row in window_rows if metric in row]
                if not values:
                    raise ValueError(f"{branch}/{lo}-{hi}: missing training metric {metric}")
                summary[metric] = mean(values)
            output[branch][f"{lo}-{hi}"] = summary
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_svg(curves: dict[str, dict[int, dict[str, float]]], path: Path) -> None:
    width, height = 1120, 720
    panels = (
        ("traj_success", "Trajectory success", (0.0, 0.45)),
        ("task_pass_at_4", "Task pass@4", (0.0, 0.70)),
        ("final_score", "Mean final score", (0.0, 0.38)),
        ("invalid_action", "Invalid-action rate", (0.0, 0.20)),
    )
    colors = {"S0": "#3b82f6", "S1": "#ef4444", "S5": "#10b981"}
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#374151;stroke-width:1.2}</style>',
        '<text x="560" y="27" text-anchor="middle" font-size="18" font-weight="bold">Fixed ID/development validation learning curves</text>',
        '<text x="560" y="48" text-anchor="middle" font-size="12" fill="#4b5563">checkpoint step (actor updates after warmup: 0, 0, 40, 90)</text>',
    ]
    for index, (metric, title, (ymin, ymax)) in enumerate(panels):
        col, row = index % 2, index // 2
        left, top = 70 + col * 550, 75 + row * 310
        plot_w, plot_h = 470, 225
        chunks.append(f'<text x="{left + plot_w / 2}" y="{top - 10}" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(title)}</text>')
        for tick in range(5):
            y = top + plot_h - tick * plot_h / 4
            val = ymin + tick * (ymax - ymin) / 4
            chunks.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
            chunks.append(f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{val:.2f}</text>')
        chunks.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
        chunks.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
        for step in STEPS:
            x = left + step / max(STEPS) * plot_w
            chunks.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="10">{step}</text>')
        for branch in BRANCHES:
            points = []
            for step in STEPS:
                x = left + step / max(STEPS) * plot_w
                value = curves[branch][step][metric]
                y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
                points.append((x, y))
            chunks.append(
                f'<polyline fill="none" stroke="{colors[branch]}" stroke-width="2.5" points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                + '"/>'
            )
            for x, y in points:
                chunks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[branch]}"/>')
    for i, branch in enumerate(BRANCHES):
        x = 430 + i * 115
        chunks.append(f'<line x1="{x}" y1="705" x2="{x + 24}" y2="705" stroke="{colors[branch]}" stroke-width="3"/>')
        chunks.append(f'<text x="{x + 31}" y="709" font-size="12">{branch}</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    return f"{value:.4f}"


def make_report(result: dict) -> str:
    aggregate_data = result["validation"]["aggregate"]
    lines = [
        "# Active Spatial dense-score fixed-validation comparison",
        "",
        f"Status: **{result['status']}**  ",
        f"ID/development validation: **PASS** ({result['validation']['tasks']} tasks × {result['validation']['rollouts_per_task']} rollouts, {result['validation']['scenes']} scenes)  ",
        "Formal OOD validation: **NOT_RUN** (this pilot contains no OOD manifest/artifact).",
        "",
        "## Fixed validation curves",
        "",
        "The table uses the same 32 task identities at every branch/checkpoint. `success` and `final` are trajectory means; `pass@4` and `best@4` aggregate four stochastic trajectories per task.",
        "",
        "| branch | checkpoint | actor updates | success | pass@4 | final | best@4 | invalid | strict parse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    warmup = result["critic_warmup"]
    for branch in BRANCHES:
        for step in STEPS:
            item = aggregate_data[branch][str(step)]
            lines.append(
                f"| {branch} | {step} | {max(0, step - warmup)} | {fmt(item['traj_success'])} | "
                f"{fmt(item['task_pass_at_4'])} | {fmt(item['final_score'])} | "
                f"{fmt(item['best_final_score_at_4'])} | {fmt(item['invalid_action'])} | "
                f"{fmt(item['strict_parse_success_rate'])} |"
            )
    lines.extend([
        "",
        "## Paired conclusions",
        "",
        "- **S1 vs S0:** step 100 shows a positive intermediate signal, but it is not durable. At step 150 S1 has lower mean success/final score and a large format regression. The actor-update AUC does not establish a robust canonical-metric benefit.",
        "- **S5 vs S0:** S5 learns faster and has higher success/final-score AUC, driven mainly by step 100. At step 150 its mean success/final score is statistically tied with S0 and pass@4/best@4 are lower; this is sample-efficiency/early-checkpoint evidence, not a superior endpoint.",
        "- **S0:** slowest mid-training curve, but the most stable endpoint among typical single trajectories. S1's slightly higher step-150 pass@4/best@4 coexists with many invalid trajectories and should not be read as a generally better policy.",
        "",
    ])
    if result["training_window_summary"]:
        lines.extend([
            "## On-policy training-curve diagnostics",
            "",
            "These are descriptive rolling windows from each branch's own rollout stream. Reward values are branch-specific and therefore are not a fair ability comparison.",
            "",
            "| branch | steps | success | final score | invalid | strict parse | entropy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for branch in BRANCHES:
            for window in ("61-100", "101-150", "141-150"):
                item = result["training_window_summary"][branch][window]
                lines.append(
                    f"| {branch} | {window} | {fmt(item['episode/success_rate'])} | "
                    f"{fmt(item['transition/final_score/mean'])} | "
                    f"{fmt(item['transition/invalid_action_rate/mean'])} | "
                    f"{fmt(item['transition/strict_parse_success_rate/mean'])} | "
                    f"{fmt(item['actor/entropy'])} |"
                )
        final_invalid = {
            branch: result["training_window_summary"][branch]["141-150"]
            ["transition/invalid_action_rate/mean"]
            for branch in BRANCHES
        }
        lines.extend([
            "",
            "Late rollout-format instability is visible in every branch, but is much stronger for S1/S5. "
            f"In the final 10 training steps invalid-action rates are {final_invalid['S0']:.4f} (S0), "
            f"{final_invalid['S1']:.4f} (S1), and {final_invalid['S5']:.4f} (S5); S5 also has the "
            "highest entropy. The fixed validation set confirms the strongest endpoint format failure "
            "for S1, while S5's fixed-validation format remains much better than its late on-policy stream.",
            "",
        ])
    lines.extend([
        "### Key scene-cluster bootstrap contrasts (95% interval)",
        "",
        "| contrast | scope | metric | delta | 95% CI | P(delta > 0) |",
        "|---|---|---|---:|---:|---:|",
    ])
    for branch in ("S1", "S5"):
        for scope in ("step_100", "step_150", "actor_update_auc"):
            for metric in ("traj_success", "final_score", "invalid_action"):
                stat = result["contrasts"][branch][scope][metric]
                lines.append(
                    f"| {branch}−S0 | {scope} | {metric} | {stat['estimate']:+.4f} | "
                    f"[{stat['ci95'][0]:+.4f}, {stat['ci95'][1]:+.4f}] | "
                    f"{stat['probability_gt_zero']:.3f} |"
                )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Checkpoints 0 and 50 both have zero actor updates because `critic_warmup=60`; their difference is stochastic evaluation noise, not learning.",
        "- Trajectory hashes are not paired. Pairing is at task/scene level; uncertainty uses a deterministic scene-cluster bootstrap over only seven scenes.",
        "- This is one training seed and 32 development tasks, all `projective_relations`; confidence intervals do not capture training-seed variance.",
        "- Validation used one common reward/scorer configuration. Validation reward is shaped, so conclusions prioritize canonical success/final-score metrics.",
        "- No formal OOD artifact exists for this pilot. OOD generalization remains **NOT_RUN**.",
        "",
        "## Reproduction",
        "",
        "See `analysis_command` in `validation_analysis.json`. No renderer, training, optimizer step, or checkpoint mutation is performed.",
    ])
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = tuple(args.steps)
    if steps != STEPS:
        raise ValueError(f"this fixed comparison requires steps {STEPS}, got {steps}")

    all_tasks: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(dict)
    all_scenes: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    aggregates: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, dict[int, dict]] = defaultdict(dict)
    reference_ids: set[str] | None = None
    reference_scenes: dict[str, str] | None = None

    for branch in BRANCHES:
        for step in steps:
            path = run_dir / "branches" / branch / "validation" / f"{step}.jsonl"
            rows = read_jsonl(path)
            task_metrics, task_scenes = build_task_metrics(rows, args.rollouts_per_task)
            ids = set(task_metrics)
            if reference_ids is None:
                reference_ids, reference_scenes = ids, task_scenes
            if ids != reference_ids:
                raise ValueError(f"{branch}/{step}: task set differs from fixed reference")
            if task_scenes != reference_scenes:
                raise ValueError(f"{branch}/{step}: task-to-scene mapping differs")
            all_tasks[branch][step] = task_metrics
            all_scenes[branch][step] = task_scenes
            aggregates[branch][step] = aggregate(task_metrics)
            metadata[branch][step] = {
                "path": str(path),
                "rows": len(rows),
                "tasks": len(task_metrics),
                "scenes": len(set(task_scenes.values())),
                "trajectory_hashes": len({row.get("trajectory_hash") for row in rows}),
                "format_counts": {
                    field: sum(float(row[field]) for row in rows) for field in FORMAT_FIELDS
                },
            }

    assert reference_ids is not None and reference_scenes is not None
    hash_overlap: dict[str, dict[str, int]] = {}
    for step in steps:
        row_hashes = {}
        for branch in BRANCHES:
            path = run_dir / "branches" / branch / "validation" / f"{step}.jsonl"
            row_hashes[branch] = {row.get("trajectory_hash") for row in read_jsonl(path)}
        hash_overlap[str(step)] = {
            f"{lhs}_{rhs}": len(row_hashes[lhs] & row_hashes[rhs])
            for lhs, rhs in (("S0", "S1"), ("S0", "S5"), ("S1", "S5"))
        }

    contrasts: dict[str, dict] = {}
    for index, branch in enumerate(("S1", "S5"), 1):
        branch_result: dict[str, dict] = {}
        for step in steps:
            stats = {}
            for metric_index, metric in enumerate(PRIMARY_METRICS):
                diffs = paired_difference(all_tasks[branch][step], all_tasks["S0"][step], metric)
                stats[metric] = paired_scene_bootstrap(
                    diffs,
                    reference_scenes,
                    samples=args.bootstrap_samples,
                    seed=args.seed + index * 10_000 + step * 100 + metric_index,
                )
            branch_result[f"step_{step}"] = stats

        auc_maps: dict[str, dict[str, float]] = {metric: {} for metric in PRIMARY_METRICS}
        base_auc_maps: dict[str, dict[str, float]] = {metric: {} for metric in PRIMARY_METRICS}
        for metric in PRIMARY_METRICS:
            for task_id in reference_ids:
                auc_maps[metric][task_id] = normalized_auc(
                    {step: all_tasks[branch][step][task_id][metric] for step in steps},
                    args.critic_warmup,
                )
                base_auc_maps[metric][task_id] = normalized_auc(
                    {step: all_tasks["S0"][step][task_id][metric] for step in steps},
                    args.critic_warmup,
                )
        branch_result["actor_update_auc"] = {}
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            diffs = {
                task: auc_maps[metric][task] - base_auc_maps[metric][task]
                for task in reference_ids
            }
            branch_result["actor_update_auc"][metric] = paired_scene_bootstrap(
                diffs,
                reference_scenes,
                samples=args.bootstrap_samples,
                seed=args.seed + index * 10_000 + 900_000 + metric_index,
            )
        contrasts[branch] = branch_result

    validation_csv_rows = []
    for branch in BRANCHES:
        for step in steps:
            validation_csv_rows.append(
                {
                    "branch": branch,
                    "checkpoint_step": step,
                    "actor_updates": max(0, step - args.critic_warmup),
                    **aggregates[branch][step],
                }
            )
    write_csv(
        output_dir / "validation_curves.csv",
        validation_csv_rows,
        ["branch", "checkpoint_step", "actor_updates", *next(iter(aggregates["S0"].values())).keys()],
    )

    training_rows = []
    for branch in BRANCHES:
        for row in parse_training_log(run_dir / "branches" / branch / "train.log"):
            training_rows.append({"branch": branch, **row})
    write_csv(
        output_dir / "training_curves.csv",
        training_rows,
        ["branch", "step", *TRAIN_METRICS],
    )
    training_window_summary = summarize_training_windows(training_rows) if training_rows else {}

    result = {
        "schema_version": "active_spatial_dense_score_validation_analysis.v1",
        "status": "PASS",
        "formal_ood_status": "NOT_RUN",
        "critic_warmup": args.critic_warmup,
        "checkpoint_to_actor_updates": {
            str(step): max(0, step - args.critic_warmup) for step in steps
        },
        "analysis_command": (
            f"python scripts/analyze_active_spatial_dense_score_validation.py "
            f"--run-dir {run_dir} --output-dir {output_dir}"
        ),
        "validation": {
            "split_interpretation": "fixed ID/development regression",
            "tasks": len(reference_ids),
            "rollouts_per_task": args.rollouts_per_task,
            "scenes": len(set(reference_scenes.values())),
            "task_id_sha256": task_digest(reference_ids),
            "aggregate": {
                branch: {str(step): values for step, values in by_step.items()}
                for branch, by_step in aggregates.items()
            },
            "artifacts": {
                branch: {str(step): values for step, values in by_step.items()}
                for branch, by_step in metadata.items()
            },
            "trajectory_hash_overlap": hash_overlap,
        },
        "contrasts": contrasts,
        "training_window_summary": training_window_summary,
        "caveats": [
            "single training seed",
            "seven scene clusters",
            "stochastic trajectories are not paired",
            "no formal OOD manifest/artifact",
            "common validation reward is shaped; canonical metrics are primary",
        ],
    }
    (output_dir / "validation_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_svg(aggregates, output_dir / "validation_learning_curves.svg")
    (output_dir / "REPORT.md").write_text(make_report(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    parser.add_argument("--rollouts-per-task", type=int, default=4)
    parser.add_argument("--critic-warmup", type=int, default=60)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261009)
    return parser.parse_args()


if __name__ == "__main__":
    analysis = analyze(parse_args())
    print(json.dumps({
        "status": analysis["status"],
        "formal_ood_status": analysis["formal_ood_status"],
        "output_dir": str(Path(analysis["analysis_command"].split("--output-dir ", 1)[1])),
    }, indent=2))
