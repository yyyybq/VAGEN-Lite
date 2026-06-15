#!/usr/bin/env python3
"""
Active Spatial Perception Dataset Generation Script

This script runs the complete pipeline to generate training data for active spatial navigation tasks.

Usage:
    python run_pipeline.py \
        --scenes_root /path/to/InteriorGS \
        --output_dir /path/to/output \
        --num_cameras 5

Example with specific scenes:
    python run_pipeline.py \
        --scenes_root /path/to/InteriorGS \
        --output_dir /path/to/output \
        --scenes 0267_840790 0002_839955

Example with single scene_id:
    python run_pipeline.py \
        --scenes_root /path/to/InteriorGS \
        --output_dir /path/to/output \
        --scene_id 0267_840790
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add current directory and parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig, ObjectSelectionConfig, CameraSamplingConfig, TaskConfig
from pipeline import ActiveSpatialPipeline


# All 9 task types
ALL_TASK_TYPES = [
    'absolute_positioning',
    'delta_control', 
    'equidistance',
    'projective_relations',
    'centering',
    'occlusion_alignment',
    'fov_inclusion',
    'size_distance_invariance',
    'screen_occupancy'
]


def generate_statistics_figure(data_items, output_path, scene_id=None):
    """Generate a bar chart showing the count of each task type."""
    
    # Count task types
    task_counts = Counter(item.get('task_type', 'unknown') for item in data_items)
    
    # Ensure all 9 task types are included (even if count is 0)
    all_counts = {task: task_counts.get(task, 0) for task in ALL_TASK_TYPES}
    
    # Prepare data for plotting
    tasks = list(all_counts.keys())
    counts = list(all_counts.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color scheme - different colors for different task categories
    colors = [
        '#2ecc71',  # absolute_positioning - green
        '#3498db',  # delta_control - blue
        '#9b59b6',  # equidistance - purple
        '#e74c3c',  # projective_relations - red
        '#f39c12',  # centering - orange
        '#1abc9c',  # occlusion_alignment - teal
        '#e91e63',  # fov_inclusion - pink
        '#00bcd4',  # size_distance_invariance - cyan
        '#ff9800',  # screen_occupancy - amber
    ]
    
    # Create bars
    bars = ax.bar(range(len(tasks)), counts, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    # Format task names for display (replace underscores with newlines for readability)
    task_labels = [task.replace('_', '\n') for task in tasks]
    
    # Set labels and title
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(task_labels, fontsize=10, ha='center')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task Type', fontsize=14, fontweight='bold')
    
    title = 'Active Spatial Dataset - Task Distribution'
    if scene_id:
        title += f'\n(Scene: {scene_id})'
    title += f'\nTotal: {sum(counts)} samples'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Add some padding on top for the count labels
    max_count = max(counts) if counts else 1
    ax.set_ylim(top=max_count * 1.15)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Statistics figure saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Active Spatial Perception Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--scenes_root', required=True, type=str,
                        help='Root directory containing scene folders with labels.json')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Output directory for generated dataset')
    
    # Scene selection
    parser.add_argument('--scenes', nargs='+', type=str, default=None,
                        help='Specific scene names to process (default: all scenes)')
    parser.add_argument('--scene_id', type=str, default=None,
                        help='Process only this single scene by ID (e.g., 0267_840790). '
                             'This is a shortcut for --scenes with a single scene.')
    
    # Camera sampling
    parser.add_argument('--num_cameras', type=int, default=5,
                        help='Number of camera poses to sample per object/pair')
    
    # Task selection
    parser.add_argument('--tasks', nargs='+', type=str, default=None,
                        help='Tasks to enable (default: all). Options: '
                             'absolute_positioning, delta_control, equidistance, '
                             'projective_relations, centering, occlusion_alignment, '
                             'fov_inclusion, size_distance_invariance, screen_occupancy')
    
    # Processing options
    parser.add_argument('--no_intermediate', action='store_true',
                        help='Do not save intermediate per-scene results')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Load configuration from JSON file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle scene_id as a shortcut for --scenes with single scene
    scene_id = args.scene_id
    scenes_list = args.scenes
    if scene_id:
        # scene_id takes precedence over --scenes
        scenes_list = [scene_id]
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PipelineConfig.from_dict(config_dict)
    else:
        # Build config from arguments
        object_config = ObjectSelectionConfig()
        
        camera_config = CameraSamplingConfig(
            num_cameras_per_item=args.num_cameras
        )
        
        task_config = TaskConfig()
        if args.tasks:
            task_config.enabled_tasks = args.tasks
        
        config = PipelineConfig(
            scenes_root=args.scenes_root,
            output_dir=args.output_dir,
            scene_list=scenes_list,
            save_intermediate=not args.no_intermediate,
            object_selection=object_config,
            camera_sampling=camera_config,
            task_config=task_config,
        )
    
    # Override paths if provided in args
    if args.scenes_root:
        config.scenes_root = args.scenes_root
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Print configuration
    print("=" * 60)
    print("Active Spatial Perception Dataset Generation")
    print("=" * 60)
    print(f"Scenes root: {config.scenes_root}")
    print(f"Output directory: {config.output_dir}")
    if scene_id:
        print(f"Processing single scene: {scene_id}")
    elif scenes_list:
        print(f"Processing scenes: {scenes_list}")
    else:
        print("Processing all scenes")
    print(f"Cameras per object: {config.camera_sampling.num_cameras_per_item}")
    print(f"Enabled tasks: {config.task_config.enabled_tasks}")
    print("=" * 60)
    print()
    
    # Run pipeline
    pipeline = ActiveSpatialPipeline(config)
    data_items = pipeline.run(verbose=args.verbose, scene_id=scene_id)
    
    # Generate statistics figure
    output_dir = Path(config.output_dir)
    if scene_id:
        figure_path = output_dir / f'task_statistics_{scene_id}.png'
    else:
        figure_path = output_dir / 'task_statistics.png'
    generate_statistics_figure(data_items, figure_path, scene_id=scene_id)
    
    print()
    print(f"Generated {len(data_items)} data items")
    if scene_id:
        print(f"Dataset saved to: {Path(config.output_dir) / f'train_data_{scene_id}.jsonl'}")
    else:
        print(f"Dataset saved to: {Path(config.output_dir) / 'train_data.jsonl'}")


if __name__ == '__main__':
    main()
