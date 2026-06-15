"""
Generate paths for diverse task types from a specific scene.

This script selects a few samples from each task type to ensure all 9 task types work correctly.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from path_generator import PathGenerator, PathConfig, generate_paths_for_scene


def select_diverse_samples(jsonl_path: Path, scene_id: str, samples_per_task: int = 2):
    """
    Select a diverse set of samples covering all task types.
    
    Returns:
        List of (data_idx, item) tuples
    """
    task_samples = defaultdict(list)
    
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            if item.get('scene_id') == scene_id:
                task_type = item.get('task_type', 'unknown')
                if len(task_samples[task_type]) < samples_per_task:
                    task_samples[task_type].append((idx, item))
    
    # Flatten and sort by task type
    selected = []
    for task_type in sorted(task_samples.keys()):
        selected.extend(task_samples[task_type])
    
    print(f"Selected {len(selected)} samples from {len(task_samples)} task types:")
    for task_type, samples in sorted(task_samples.items()):
        print(f"  {task_type}: {len(samples)} samples")
    
    return selected


def generate_diverse_paths(
    jsonl_path: Path,
    scene_id: str,
    output_path: Path,
    samples_per_task: int = 2,
    config: PathConfig = None
):
    """Generate paths for diverse samples."""
    config = config or PathConfig()
    generator = PathGenerator(config)
    
    # Select diverse samples
    samples = select_diverse_samples(jsonl_path, scene_id, samples_per_task)
    
    # Generate paths
    generated_paths = []
    for i, (data_idx, item) in enumerate(samples):
        task_type = item.get('task_type', 'unknown')
        print(f"\n[{i+1}/{len(samples)}] {task_type}: {item.get('task_description', '')[:50]}...")
        
        path = generator.generate_path(item, data_idx)
        generated_paths.append(path)
        
        status = "✓ Success" if path.success else "✗ Not converged"
        print(f"  {status} - reward: {path.final_reward:.4f}, steps: {path.total_steps}")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for path in generated_paths:
            path_dict = {
                'data_idx': path.data_idx,
                'scene_id': path.scene_id,
                'task_type': path.task_type,
                'task_description': path.task_description,
                'object_label': path.object_label,
                'init_camera': path.init_camera,
                'target_region': path.target_region,
                'sample_target': path.sample_target,
                'path': path.path,
                'final_reward': path.final_reward,
                'success': path.success,
                'total_steps': path.total_steps,
            }
            f.write(json.dumps(path_dict) + '\n')
    
    print(f"\nSaved {len(generated_paths)} paths to {output_path}")
    
    # Summary
    by_task = defaultdict(list)
    for path in generated_paths:
        by_task[path.task_type].append(path)
    
    print("\nSummary by task type:")
    for task_type in sorted(by_task.keys()):
        paths = by_task[task_type]
        success = sum(1 for p in paths if p.success)
        avg_reward = sum(p.final_reward for p in paths) / len(paths)
        print(f"  {task_type}: {success}/{len(paths)} success, avg reward: {avg_reward:.4f}")
    
    return generated_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diverse paths for all task types")
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--samples_per_task", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--min_steps", type=int, default=3, 
                        help="Minimum steps required for meaningful paths")
    parser.add_argument("--min_distance", type=float, default=0.3,
                        help="Minimum distance to target (meters)")
    parser.add_argument("--min_yaw_offset", type=float, default=15.0,
                        help="Minimum yaw offset to target (degrees)")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    config = PathConfig(
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        min_distance=args.min_distance,
        min_yaw_offset_deg=args.min_yaw_offset,
        verbose=args.verbose,
    )
    
    generate_diverse_paths(
        jsonl_path=Path(args.jsonl_path),
        scene_id=args.scene_id,
        output_path=Path(args.output_path),
        samples_per_task=args.samples_per_task,
        config=config,
    )
