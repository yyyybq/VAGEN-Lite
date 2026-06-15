#!/usr/bin/env python3
"""
Active Spatial Evaluation CLI
==============================

Main entry point for running evaluations.

Usage:
    # Run with YAML config
    python evaluation/run_eval.py --config evaluation/configs/eval_random.yaml
    
    # Run baseline evaluations
    python evaluation/run_eval.py --agent random --jsonl /path/to/test_data.jsonl
    python evaluation/run_eval.py --agent heuristic --jsonl /path/to/test_data.jsonl
    
    # Run with trained model checkpoint
    python evaluation/run_eval.py --agent model \
        --jsonl /path/to/test_data.jsonl \
        --checkpoint /path/to/checkpoint \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct
    
    # Run with API model (frozen baseline)
    python evaluation/run_eval.py --agent model \
        --jsonl /path/to/test_data.jsonl \
        --provider openai --model-name gpt-4o
    
    # Filter specific task types
    python evaluation/run_eval.py --agent random \
        --jsonl /path/to/test_data.jsonl \
        --task-types absolute_positioning screen_occupancy
    
    # Compare multiple agents
    python evaluation/compare.py \
        results/random/results_random.json \
        results/heuristic/results_heuristic.json \
        results/model/results_model.json
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_config import EvalConfig, EvalEnvConfig, EvalModelConfig
from evaluation.eval_runner import EvalRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Spatial Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Config file (overrides all other args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML evaluation config file")
    
    # Agent
    parser.add_argument("--agent", type=str, default="random",
                        choices=["random", "heuristic", "constant", "model", "frozen"],
                        help="Agent type for evaluation")
    
    # Environment
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to JSONL test data file")
    parser.add_argument("--gs-root", type=str, 
                        default="/scratch/by2593/project/Active_Spatial/InteriorGS",
                        help="Path to 3D Gaussian Splatting data root")
    parser.add_argument("--render-backend", type=str, default="local",
                        choices=["local", "client", "none"],
                        help="Rendering backend")
    parser.add_argument("--gpu-device", type=int, default=4,
                        help="GPU device for rendering")
    parser.add_argument("--success-threshold", type=float, default=0.85,
                        help="Score threshold for success")
    
    # Model (if agent is model/frozen)
    parser.add_argument("--provider", type=str, default="vllm",
                        help="Model provider (vllm, openai, claude, gemini)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Model name or HuggingFace ID")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    
    # Evaluation
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max number of episodes to evaluate (None = all)")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max LLM turns per episode")
    parser.add_argument("--task-types", nargs="+", default=None,
                        help="Filter for specific task types")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Seed offset for episode selection")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--eval-name", type=str, default=None,
                        help="Name for this evaluation run")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to WandB")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save full episode trajectories")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def build_config_from_args(args) -> EvalConfig:
    """Build EvalConfig from CLI arguments."""
    
    # Default JSONL path
    default_jsonl = "/scratch/by2593/project/Active_Spatial/VAGEN/data_gen/active_spatial_pipeline/output/train_data_0267_840790_balanced.jsonl"
    jsonl_path = args.jsonl or default_jsonl
    
    # Default output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = args.eval_name or f"eval_{args.agent}_{timestamp}"
    output_dir = args.output_dir or f"evaluation/outputs/{eval_name}"
    
    env_config = EvalEnvConfig(
        jsonl_path=jsonl_path,
        render_backend=args.render_backend if args.render_backend != "none" else None,
        gs_root=args.gs_root,
        gpu_device=args.gpu_device,
        success_score_threshold=args.success_threshold,
    )
    
    model_config = EvalModelConfig(
        provider=args.provider,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        temperature=args.temperature,
        tensor_parallel_size=args.tp,
    )
    
    config = EvalConfig(
        eval_name=eval_name,
        output_dir=output_dir,
        max_steps_per_episode=args.max_turns,
        num_eval_episodes=args.max_episodes,
        seed_offset=args.seed_offset,
        agent_type=args.agent,
        env=env_config,
        model=model_config,
        use_wandb=args.wandb,
        save_trajectories=args.save_trajectories,
        task_types=args.task_types,
        verbose=args.verbose,
    )
    
    return config


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        config = EvalConfig.from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = build_config_from_args(args)
    
    # Set verbose env var
    if config.verbose:
        os.environ["ACTIVE_SPATIAL_ENV_VERBOSE"] = "1"
    
    # Run evaluation
    runner = EvalRunner(config)
    results = runner.run()
    
    return results


if __name__ == "__main__":
    main()
