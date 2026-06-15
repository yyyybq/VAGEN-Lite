"""
Per-Task Metrics Module for Active Spatial Evaluation
=====================================================

Computes fine-grained metrics broken down by:
- Task type (9 types)
- Episode outcome (success/failure/timeout)
- Efficiency (SPL, step count, score trajectory shape)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json


# All 9 task types
ALL_TASK_TYPES = [
    "absolute_positioning",
    "delta_control",
    "equidistance",
    "projective_relations",
    "centering",
    "occlusion_alignment",
    "fov_inclusion",
    "size_distance_invariance",
    "screen_occupancy",
]

TASK_CATEGORIES = {
    "metric_distance": ["absolute_positioning", "delta_control", "equidistance"],
    "projective_relation": ["projective_relations", "centering", "occlusion_alignment"],
    "view_perspective": ["fov_inclusion", "size_distance_invariance", "screen_occupancy"],
}


@dataclass
class EpisodeRecord:
    """Record of a single evaluation episode."""
    episode_id: int
    seed: int
    task_type: str
    scene_id: str
    object_label: str
    preset: str
    
    # Outcome
    success: bool = False
    done_by_agent: bool = False        # Agent issued "done" action
    auto_terminated: bool = False      # Score >= threshold
    timed_out: bool = False            # Hit max steps
    
    # Scores
    initial_score: float = 0.0
    final_score: float = 0.0
    max_score: float = 0.0
    score_trajectory: List[float] = field(default_factory=list)
    
    # Efficiency
    num_steps: int = 0                 # Total primitive actions
    num_turns: int = 0                 # Total LLM turns
    total_reward: float = 0.0
    
    # Collisions
    total_collisions: int = 0
    
    # Actions
    action_validity_rate: float = 0.0
    action_effectiveness_rate: float = 0.0
    actions_taken: List[str] = field(default_factory=list)
    
    # Timing
    episode_time_seconds: float = 0.0
    
    # Raw trajectory (optional, for analysis)
    trajectory: Optional[List[Dict]] = None


@dataclass
class TaskMetrics:
    """Aggregated metrics for a specific task type."""
    task_type: str
    num_episodes: int = 0
    
    # Success
    success_rate: float = 0.0
    completion_rate: float = 0.0       # Agent issued "done"
    timeout_rate: float = 0.0
    
    # Scores
    mean_final_score: float = 0.0
    std_final_score: float = 0.0
    mean_initial_score: float = 0.0
    mean_score_improvement: float = 0.0   # final - initial
    mean_max_score: float = 0.0
    
    # Efficiency
    mean_steps: float = 0.0
    mean_turns: float = 0.0
    spl: float = 0.0                    # Success weighted by Path Length
    mean_reward: float = 0.0
    
    # Collisions
    mean_collisions: float = 0.0
    
    # Actions
    mean_action_validity: float = 0.0
    mean_action_effectiveness: float = 0.0
    
    # Score trajectory shape
    monotonic_improvement_rate: float = 0.0   # % of episodes with mostly increasing score


def compute_spl(episodes: List[EpisodeRecord], optimal_steps: Optional[Dict[int, int]] = None) -> float:
    """
    Compute Success weighted by Path Length.
    
    SPL = (1/N) * Σ S_i * (l_i / max(p_i, l_i))
    
    Where S_i = success, l_i = optimal path length, p_i = actual path length.
    If optimal steps unknown, use a heuristic based on initial distance.
    """
    if not episodes:
        return 0.0
    
    spl_sum = 0.0
    for ep in episodes:
        if ep.success:
            if optimal_steps and ep.episode_id in optimal_steps:
                l_i = optimal_steps[ep.episode_id]
            else:
                # Heuristic: assume optimal is roughly score_needed / step_improvement
                # A very rough estimate: 5 steps minimum
                l_i = max(5, int(ep.num_steps * 0.3))
            
            p_i = max(ep.num_steps, 1)
            spl_sum += l_i / max(p_i, l_i)
    
    return spl_sum / len(episodes)


def compute_monotonic_rate(score_trajectory: List[float], tolerance: float = 0.02) -> bool:
    """Check if a score trajectory is mostly monotonically increasing."""
    if len(score_trajectory) < 2:
        return True
    
    regressions = 0
    for i in range(1, len(score_trajectory)):
        if score_trajectory[i] < score_trajectory[i-1] - tolerance:
            regressions += 1
    
    regression_rate = regressions / (len(score_trajectory) - 1)
    return regression_rate < 0.3  # Less than 30% regressions = "monotonic"


def compute_task_metrics(episodes: List[EpisodeRecord]) -> TaskMetrics:
    """Compute aggregated metrics for a group of episodes (same task type)."""
    if not episodes:
        return TaskMetrics(task_type="unknown")
    
    task_type = episodes[0].task_type
    n = len(episodes)
    
    successes = [e for e in episodes if e.success]
    done_by_agent = [e for e in episodes if e.done_by_agent]
    timed_outs = [e for e in episodes if e.timed_out]
    
    final_scores = [e.final_score for e in episodes]
    initial_scores = [e.initial_score for e in episodes]
    improvements = [e.final_score - e.initial_score for e in episodes]
    max_scores = [e.max_score for e in episodes]
    
    steps = [e.num_steps for e in episodes]
    turns = [e.num_turns for e in episodes]
    rewards = [e.total_reward for e in episodes]
    collisions = [e.total_collisions for e in episodes]
    
    validities = [e.action_validity_rate for e in episodes]
    effectivenesses = [e.action_effectiveness_rate for e in episodes]
    
    mono_count = sum(1 for e in episodes if compute_monotonic_rate(e.score_trajectory))
    
    return TaskMetrics(
        task_type=task_type,
        num_episodes=n,
        success_rate=len(successes) / n,
        completion_rate=len(done_by_agent) / n,
        timeout_rate=len(timed_outs) / n,
        mean_final_score=float(np.mean(final_scores)),
        std_final_score=float(np.std(final_scores)),
        mean_initial_score=float(np.mean(initial_scores)),
        mean_score_improvement=float(np.mean(improvements)),
        mean_max_score=float(np.mean(max_scores)),
        mean_steps=float(np.mean(steps)),
        mean_turns=float(np.mean(turns)),
        spl=compute_spl(episodes),
        mean_reward=float(np.mean(rewards)),
        mean_collisions=float(np.mean(collisions)),
        mean_action_validity=float(np.mean(validities)) if validities else 0.0,
        mean_action_effectiveness=float(np.mean(effectivenesses)) if effectivenesses else 0.0,
        monotonic_improvement_rate=mono_count / n,
    )


def compute_all_metrics(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Returns a dictionary with:
    - "overall": Aggregated across all episodes
    - "by_task_type": Per-task-type breakdown
    - "by_category": Per-category (metric_distance, projective_relation, view_perspective)
    - "summary_table": Formatted table data for display
    """
    results = {}
    
    # Overall
    results["overall"] = _task_metrics_to_dict(compute_task_metrics(episodes))
    results["overall"]["task_type"] = "ALL"
    results["overall"]["num_task_types_tested"] = len(set(e.task_type for e in episodes))
    
    # By task type
    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep.task_type].append(ep)
    
    results["by_task_type"] = {}
    for task_type in ALL_TASK_TYPES:
        if task_type in by_type:
            results["by_task_type"][task_type] = _task_metrics_to_dict(
                compute_task_metrics(by_type[task_type])
            )
        else:
            results["by_task_type"][task_type] = {"num_episodes": 0, "note": "no test data"}
    
    # By category
    results["by_category"] = {}
    for cat_name, cat_types in TASK_CATEGORIES.items():
        cat_episodes = [e for e in episodes if e.task_type in cat_types]
        if cat_episodes:
            results["by_category"][cat_name] = _task_metrics_to_dict(
                compute_task_metrics(cat_episodes)
            )
        else:
            results["by_category"][cat_name] = {"num_episodes": 0}
    
    # Summary table
    results["summary_table"] = _build_summary_table(results)
    
    return results


def _task_metrics_to_dict(tm: TaskMetrics) -> Dict[str, Any]:
    """Convert TaskMetrics dataclass to dictionary."""
    import dataclasses
    return dataclasses.asdict(tm)


def _build_summary_table(results: Dict) -> List[Dict[str, Any]]:
    """Build a summary table for display."""
    rows = []
    
    # Overall row
    overall = results["overall"]
    rows.append({
        "task": "ALL",
        "n": overall["num_episodes"],
        "success%": f"{overall['success_rate']*100:.1f}",
        "final_score": f"{overall['mean_final_score']:.3f}",
        "improvement": f"{overall['mean_score_improvement']:.3f}",
        "spl": f"{overall['spl']:.3f}",
        "steps": f"{overall['mean_steps']:.1f}",
        "collisions": f"{overall['mean_collisions']:.1f}",
        "monotonic%": f"{overall['monotonic_improvement_rate']*100:.1f}",
    })
    
    rows.append({"task": "---"})  # Separator
    
    # Per-task rows
    for task_type in ALL_TASK_TYPES:
        tm = results["by_task_type"].get(task_type, {})
        if tm.get("num_episodes", 0) > 0:
            rows.append({
                "task": task_type,
                "n": tm["num_episodes"],
                "success%": f"{tm['success_rate']*100:.1f}",
                "final_score": f"{tm['mean_final_score']:.3f}",
                "improvement": f"{tm['mean_score_improvement']:.3f}",
                "spl": f"{tm['spl']:.3f}",
                "steps": f"{tm['mean_steps']:.1f}",
                "collisions": f"{tm['mean_collisions']:.1f}",
                "monotonic%": f"{tm['monotonic_improvement_rate']*100:.1f}",
            })
    
    return rows


def format_results_table(results: Dict) -> str:
    """Format results as a printable table."""
    table = results["summary_table"]
    if not table:
        return "No results to display."
    
    # Column widths
    headers = ["task", "n", "success%", "final_score", "improvement", "spl", "steps", "collisions", "monotonic%"]
    col_widths = {h: max(len(h), max(len(str(row.get(h, ""))) for row in table)) for h in headers}
    
    # Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    
    lines = [header_line, sep_line]
    for row in table:
        if row.get("task") == "---":
            lines.append(sep_line)
        else:
            lines.append(" | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))
    
    return "\n".join(lines)


def save_results(results: Dict, output_path: str, episodes: Optional[List[EpisodeRecord]] = None):
    """Save full evaluation results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "metrics": results,
        "formatted_table": format_results_table(results),
    }
    
    # Include per-episode details (without full trajectory to save space)
    if episodes:
        output["episodes"] = []
        for ep in episodes:
            ep_dict = {
                "episode_id": ep.episode_id,
                "seed": ep.seed,
                "task_type": ep.task_type,
                "scene_id": ep.scene_id,
                "object_label": ep.object_label,
                "success": ep.success,
                "initial_score": ep.initial_score,
                "final_score": ep.final_score,
                "max_score": ep.max_score,
                "num_steps": ep.num_steps,
                "num_turns": ep.num_turns,
                "total_reward": ep.total_reward,
                "total_collisions": ep.total_collisions,
                "score_trajectory": ep.score_trajectory,
            }
            output["episodes"].append(ep_dict)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Also save a human-readable summary
    summary_path = output_path.replace(".json", "_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Active Spatial Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(format_results_table(results))
        f.write("\n\n")
        
        # Category summary
        f.write("Category Breakdown:\n")
        f.write("-" * 40 + "\n")
        for cat, cat_data in results.get("by_category", {}).items():
            if cat_data.get("num_episodes", 0) > 0:
                f.write(f"  {cat}: success={cat_data['success_rate']*100:.1f}%, "
                       f"score={cat_data['mean_final_score']:.3f}, "
                       f"n={cat_data['num_episodes']}\n")
        f.write("\n")
