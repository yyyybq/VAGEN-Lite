#!/usr/bin/env python3
"""
Compare evaluation results from multiple agents.

Usage:
    python evaluation/compare.py \
        evaluation/outputs/eval_random/results_random.json \
        evaluation/outputs/eval_heuristic/results_heuristic.json \
        evaluation/outputs/eval_model/results_model.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_results(path: str) -> Dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def print_comparison_table(all_results: Dict[str, Dict]):
    """Print a side-by-side comparison table."""
    agent_names = list(all_results.keys())
    
    # Overall metrics to compare
    key_metrics = [
        ("success_rate", "Success Rate", lambda x: f"{x*100:.1f}%"),
        ("mean_final_score", "Final Score", lambda x: f"{x:.3f}"),
        ("mean_score_improvement", "Score Improv.", lambda x: f"{x:+.3f}"),
        ("spl", "SPL", lambda x: f"{x:.3f}"),
        ("mean_steps", "Avg Steps", lambda x: f"{x:.1f}"),
        ("mean_turns", "Avg Turns", lambda x: f"{x:.1f}"),
        ("mean_collisions", "Avg Collisions", lambda x: f"{x:.1f}"),
        ("mean_action_validity", "Action Valid%", lambda x: f"{x*100:.1f}%"),
        ("monotonic_improvement_rate", "Monotonic%", lambda x: f"{x*100:.1f}%"),
    ]
    
    # Print header
    col_width = 16
    metric_width = 18
    
    print("\n" + "=" * (metric_width + col_width * len(agent_names) + 4))
    print("AGENT COMPARISON — Overall Metrics")
    print("=" * (metric_width + col_width * len(agent_names) + 4))
    
    header = "Metric".ljust(metric_width) + " | " + " | ".join(n.center(col_width) for n in agent_names)
    print(header)
    print("-" * len(header))
    
    for key, label, fmt in key_metrics:
        row = label.ljust(metric_width) + " | "
        values = []
        for agent in agent_names:
            overall = all_results[agent].get("metrics", {}).get("overall", {})
            val = overall.get(key, None)
            if val is not None:
                values.append(fmt(val))
            else:
                values.append("N/A")
        row += " | ".join(v.center(col_width) for v in values)
        print(row)
    
    # Per-task type comparison (success rate only)
    print("\n" + "=" * (metric_width + col_width * len(agent_names) + 4))
    print("PER-TASK SUCCESS RATE")
    print("=" * (metric_width + col_width * len(agent_names) + 4))
    
    header = "Task Type".ljust(metric_width) + " | " + " | ".join(n.center(col_width) for n in agent_names)
    print(header)
    print("-" * len(header))
    
    all_task_types = set()
    for agent_data in all_results.values():
        by_type = agent_data.get("metrics", {}).get("by_task_type", {})
        all_task_types.update(by_type.keys())
    
    for task_type in sorted(all_task_types):
        row = task_type[:metric_width].ljust(metric_width) + " | "
        values = []
        for agent in agent_names:
            task_data = all_results[agent].get("metrics", {}).get("by_task_type", {}).get(task_type, {})
            n = task_data.get("num_episodes", 0)
            if n > 0:
                sr = task_data.get("success_rate", 0)
                values.append(f"{sr*100:.1f}% (n={n})")
            else:
                values.append("—")
        row += " | ".join(v.center(col_width) for v in values)
        print(row)
    
    # Per-category comparison
    print("\n" + "=" * (metric_width + col_width * len(agent_names) + 4))
    print("PER-CATEGORY SUCCESS RATE")
    print("=" * (metric_width + col_width * len(agent_names) + 4))
    
    header = "Category".ljust(metric_width) + " | " + " | ".join(n.center(col_width) for n in agent_names)
    print(header)
    print("-" * len(header))
    
    for cat in ["metric_distance", "projective_relation", "view_perspective"]:
        row = cat.ljust(metric_width) + " | "
        values = []
        for agent in agent_names:
            cat_data = all_results[agent].get("metrics", {}).get("by_category", {}).get(cat, {})
            n = cat_data.get("num_episodes", 0)
            if n > 0:
                sr = cat_data.get("success_rate", 0)
                score = cat_data.get("mean_final_score", 0)
                values.append(f"{sr*100:.1f}% / {score:.3f}")
            else:
                values.append("—")
        row += " | ".join(v.center(col_width) for v in values)
        print(row)
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("result_files", nargs="+", help="Paths to result JSON files")
    parser.add_argument("--output", type=str, default=None, help="Save comparison to file")
    args = parser.parse_args()
    
    all_results = {}
    for path in args.result_files:
        name = Path(path).parent.name  # Use parent dir name as agent name
        if name.startswith("eval_"):
            name = name[5:]  # Remove "eval_" prefix
        data = load_results(path)
        all_results[name] = data
        print(f"Loaded: {name} ({path})")
    
    print_comparison_table(all_results)
    
    if args.output:
        # Save comparison data
        comparison = {
            "agents": list(all_results.keys()),
            "results": {name: data.get("metrics", {}) for name, data in all_results.items()},
        }
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
