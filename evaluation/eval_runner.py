"""
Standalone Evaluation Runner for Active Spatial
================================================

Drives the ActiveSpatialEnv directly (no HTTP server required).
Supports both baseline agents and VLM model agents.

Usage:
    # With baseline agent
    runner = EvalRunner(config)
    results = runner.run()
    
    # The runner handles:
    # 1. Loading test data from JSONL
    # 2. Creating environment instances
    # 3. Running episodes with the specified agent
    # 4. Collecting per-episode metrics
    # 5. Computing aggregated statistics
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_config import EvalConfig, EvalEnvConfig
from evaluation.metrics import (
    EpisodeRecord, compute_all_metrics, format_results_table, save_results
)
from evaluation.agents import BaseAgent, create_agent


def load_test_episodes(jsonl_path: str, task_types: Optional[List[str]] = None,
                       max_episodes: Optional[int] = None,
                       include_task_types: Optional[List[str]] = None,
                       exclude_task_types: Optional[List[str]] = None) -> List[Dict]:
    """
    Load test episodes from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL data file
        task_types: Optional filter for specific task types
        max_episodes: Maximum number of episodes to load
        
    Returns:
        List of episode data dicts
    """
    episodes = []
    include_set = set(include_task_types or [])
    exclude_set = set(exclude_task_types or [])
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            # Filter by task type if specified
            item_task = item.get("task_type", "unknown")
            if task_types:
                if item_task not in task_types:
                    continue
            if include_set and item_task not in include_set:
                continue
            if exclude_set and item_task in exclude_set:
                continue
            
            item["_line_index"] = i
            item["_filtered_index"] = len(episodes)
            episodes.append(item)
            
            if max_episodes and len(episodes) >= max_episodes:
                break
    
    return episodes


class EvalRunner:
    """
    Main evaluation runner.
    
    Creates an ActiveSpatialEnv and runs evaluation episodes
    with a specified agent (baseline or VLM model).
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.env = None
        self.agent = None
        self.episode_records: List[EpisodeRecord] = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.to_yaml(str(self.output_dir / "eval_config.yaml"))
    
    def _create_env(self):
        """Create the environment instance."""
        from vagen.envs.active_spatial.env import ActiveSpatialEnv
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
        
        env_cfg = self.config.env
        include_task_types = env_cfg.include_task_types or self.config.task_types
        from dataclasses import asdict
        values = asdict(env_cfg)
        # Intersect the two filters identically in loader and environment.
        if env_cfg.include_task_types and self.config.task_types:
            include_task_types = sorted(set(env_cfg.include_task_types) & set(self.config.task_types))
            if not include_task_types:
                raise ValueError("task type filters have an empty intersection")
        values["include_task_types"] = include_task_types
        env_config = ActiveSpatialEnvConfig(**values)
        self.env = ActiveSpatialEnv(env_config)
    
    def _create_agent(self) -> BaseAgent:
        """Create the evaluation agent."""
        agent_type = self.config.agent_type
        
        if agent_type in ("random", "heuristic", "constant"):
            agent_config = {
                "seed": 42,
                "max_actions_per_step": self.config.env.max_actions_per_step,
                "step_translation": self.config.env.step_translation,
                "step_rotation_deg": self.config.env.step_rotation_deg,
            }
            return create_agent(agent_type, agent_config)
        
        elif agent_type in ("model", "frozen"):
            return self._create_model_agent()
        
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
    
    def _create_model_agent(self) -> BaseAgent:
        """Create a VLM model agent for evaluation."""
        from evaluation.model_agent import ModelAgent
        return ModelAgent(self.config.model, history_window_size=self.config.env.history_window_size)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.
        
        Returns:
            Dictionary with all evaluation metrics and results.
        """
        print("=" * 70)
        print(f"Active Spatial Evaluation: {self.config.eval_name}")
        print(f"Agent: {self.config.agent_type}")
        print("=" * 70)
        
        # Create environment and agent
        print("\n[1/4] Initializing environment...")
        self._create_env()
        
        print("[2/4] Creating agent...")
        self.agent = self._create_agent()
        
        # Load test data
        print("[3/4] Loading test episodes...")
        test_episodes = load_test_episodes(
            self.config.env.jsonl_path,
            task_types=self.config.task_types,
            max_episodes=None,
            include_task_types=self.config.env.include_task_types,
            exclude_task_types=self.config.env.exclude_task_types,
        )
        if self.config.training_jsonl_path:
            from data_gen.active_spatial_pipeline.splits import read_rows, id_candidates, assert_disjoint
            training = read_rows(self.config.training_jsonl_path)
            if self.config.split_role == "id":
                before = len(test_episodes)
                test_episodes = id_candidates(training, test_episodes)
                print(f"  ID protocol retained {len(test_episodes)}/{before} candidate episodes")
            assert_disjoint(training, test_episodes)
        if self.config.seed_offset < 0:
            raise ValueError("seed_offset must be nonnegative")
        test_episodes = test_episodes[self.config.seed_offset:]
        if self.config.num_eval_episodes is not None:
            test_episodes = test_episodes[:self.config.num_eval_episodes]
        if not test_episodes:
            raise ValueError("no eligible evaluation episodes")
        print(f"  Loaded {len(test_episodes)} episodes")
        
        # Show task distribution
        task_dist = defaultdict(int)
        for ep in test_episodes:
            task_dist[ep.get("task_type", "unknown")] += 1
        print("  Task distribution:")
        for task, count in sorted(task_dist.items()):
            print(f"    {task}: {count}")
        
        # Run episodes
        print(f"\n[4/4] Running evaluation ({len(test_episodes)} episodes)...")
        self.episode_records = []
        
        pbar = tqdm(enumerate(test_episodes), total=len(test_episodes), desc="Evaluating")
        for ep_idx, ep_data in pbar:
            seed = ep_data.get("_filtered_index", ep_idx)
            record = self._run_single_episode(ep_idx, ep_data, seed)
            self.episode_records.append(record)
            
            # Update progress bar
            completed = ep_idx + 1
            successes = sum(1 for r in self.episode_records if r.success)
            pbar.set_postfix({
                "success": f"{successes}/{completed}",
                "rate": f"{successes/completed*100:.1f}%",
                "avg_score": f"{np.mean([r.final_score for r in self.episode_records]):.3f}",
            })
        
        # Compute metrics
        print("\nComputing metrics...")
        results = compute_all_metrics(self.episode_records)
        from evaluation.eval_config import evaluation_fingerprint
        results["protocol_fingerprint"] = evaluation_fingerprint(self.config.to_dict())
        results["protocol_differences"] = self.config.protocol_differences
        
        # Display results
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(format_results_table(results))
        
        # Save results
        output_path = str(self.output_dir / f"results_{self.config.agent_type}.json")
        save_results(results, output_path, self.episode_records)
        print(f"\nResults saved to: {output_path}")
        
        # Wandb logging
        if self.config.use_wandb:
            self._log_to_wandb(results)
        
        # Cleanup
        if self.env:
            self.env.close()
        
        return results
    
    def _run_single_episode(self, ep_idx: int, ep_data: Dict, seed: int) -> EpisodeRecord:
        """Run a single evaluation episode and return the record."""
        task_type = ep_data.get("task_type", "unknown")
        scene_id = ep_data.get("scene_id", "unknown")
        object_label = ep_data.get("object_label", "unknown")
        preset = ep_data.get("preset", "unknown")
        
        # Reset environment
        obs, info = self.env.reset(seed=seed)
        self.agent.reset()
        
        # Set system prompt for model agents
        if hasattr(self.agent, 'set_system_prompt'):
            system_prompt = self.env.system_prompt()
            self.agent.set_system_prompt(system_prompt)
        
        # Build info for heuristic agent (with oracle geometry)
        agent_info = {
            "task_prompt": info.get("task_prompt"),
            "current_pose": info.get("current_pose"),
            "current_potential_score": info.get("initial_potential_score", 0.0),
            "current_region_metrics": info.get("initial_region_metrics", {}),
            "current_pose_matrix": self.env.view_engine.get_pose() if hasattr(self.env, 'view_engine') else None,
            "task_info": {
                "task_type": task_type,
                "target_region": ep_data.get("target_region", {}),
                "object_label": object_label,
                "preset": preset,
            },
        }
        
        # Episode tracking
        initial_score = info.get("initial_potential_score", 0.0)
        score_trajectory = [initial_score]
        total_reward = 0.0
        total_collisions = 0
        valid_actions = 0
        effective_actions = 0
        total_action_turns = 0
        all_actions = []
        trajectory = [] if self.config.save_trajectories else None
        
        traj_metrics = {}
        done = False
        done_by_agent = False
        auto_terminated = False
        num_turns = 0
        start_time = time.time()
        
        for turn in range(self.config.max_steps_per_episode):
            if done:
                break
            
            # Agent generates action
            action_str = self.agent.act(obs, agent_info)
            num_turns += 1
            
            # Step environment
            obs, reward, done, step_info = self.env.step(action_str)
            total_reward += reward
            
            # Track metrics
            metrics = step_info.get("metrics", {})
            turn_metrics = metrics.get("turn_metrics", {})
            traj_metrics = metrics.get("traj_metrics", {})
            
            if turn_metrics.get("action_is_valid", False):
                valid_actions += 1
            if turn_metrics.get("action_is_effective", False):
                effective_actions += 1
            total_action_turns += 1
            
            collisions = turn_metrics.get("collision_count", 0)
            total_collisions += collisions
            
            # Track score
            current_score = step_info.get("current_potential_score", 
                             turn_metrics.get("potential_score", 0.0))
            current_region_metrics = step_info.get("current_region_metrics", {})
            score_trajectory.append(current_score)
            
            # Track actions
            parsed_actions = step_info.get("actions", [])
            all_actions.extend(parsed_actions)
            
            # Check termination reason
            if step_info.get("auto_terminated", False):
                auto_terminated = True
            if "done" in [a.lower() for a in parsed_actions]:
                done_by_agent = True
            
            # Update agent info for next turn
            agent_info.update({
                "current_pose": step_info.get("current_pose"),
                "current_potential_score": current_score,
                "current_region_metrics": current_region_metrics,
                "current_pose_matrix": getattr(self.env, 'view_engine', None) and self.env.view_engine.get_pose(),
            })
            
            # Save trajectory step
            if trajectory is not None:
                trajectory.append({
                    "turn": turn,
                    "action_str": action_str,
                    "reward": reward,
                    "score": current_score,
                    "region_metrics": current_region_metrics,
                    "done": done,
                    "collisions": collisions,
                })
        
        episode_time = time.time() - start_time
        
        # Determine final state
        final_score = score_trajectory[-1] if score_trajectory else 0.0
        max_score = max(score_trajectory) if score_trajectory else 0.0
        # Scores are shaping/diagnostic values, not a second success predicate.
        # In particular canonical FOV/Projective gates can fail at high scores.
        success = bool(traj_metrics.get("success", False))
        timed_out = not done and num_turns >= self.config.max_steps_per_episode
        
        return EpisodeRecord(
            episode_id=ep_idx,
            seed=seed,
            task_type=task_type,
            scene_id=scene_id,
            object_label=object_label,
            preset=preset,
            success=success,
            done_by_agent=done_by_agent,
            auto_terminated=auto_terminated,
            timed_out=timed_out,
            initial_score=initial_score,
            final_score=final_score,
            max_score=max_score,
            score_trajectory=score_trajectory,
            num_steps=self.env._current_step,
            num_turns=num_turns,
            total_reward=total_reward,
            total_collisions=total_collisions,
            action_validity_rate=valid_actions / max(total_action_turns, 1),
            action_effectiveness_rate=effective_actions / max(total_action_turns, 1),
            actions_taken=all_actions,
            episode_time_seconds=episode_time,
            trajectory=trajectory,
        )
    
    def _log_to_wandb(self, results: Dict):
        """Log results to Weights & Biases."""
        try:
            import wandb
            
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"{self.config.eval_name}_{self.config.agent_type}",
                config=self.config.to_dict(),
            )
            
            # Log overall metrics
            overall = results["overall"]
            for k, v in overall.items():
                if isinstance(v, (int, float)):
                    wandb.log({f"eval/overall/{k}": v})
            
            # Log per-task metrics
            for task_type, tm in results.get("by_task_type", {}).items():
                if isinstance(tm, dict) and tm.get("num_episodes", 0) > 0:
                    for k, v in tm.items():
                        if isinstance(v, (int, float)):
                            wandb.log({f"eval/{task_type}/{k}": v})
            
            # Log score distributions
            if self.episode_records:
                final_scores = [r.final_score for r in self.episode_records]
                wandb.log({"eval/score_distribution": wandb.Histogram(final_scores)})
            
            wandb.finish()
            print("Results logged to WandB.")
            
        except Exception as e:
            print(f"Warning: Failed to log to WandB: {e}")
