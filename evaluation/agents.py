"""
Baseline Agents for Active Spatial Evaluation
==============================================

Provides baseline agents for comparison:
1. RandomAgent: Takes random valid actions
2. HeuristicAgent: Uses oracle geometric info to navigate optimally
3. FrozenAgent: Uses a VLM without RL training (zero-shot)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod


VALID_ACTIONS = ["move_forward", "move_backward", "turn_left", "turn_right", "look_up", "look_down"]


class BaseAgent(ABC):
    """Base class for evaluation agents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "base"
    
    @abstractmethod
    def act(self, observation: Dict[str, Any], info: Dict[str, Any] = None) -> str:
        """
        Generate an action string given an observation.
        
        Args:
            observation: {"obs_str": str, "multi_modal_data": dict}
            info: Additional info from env (e.g., current_pose, task info)
            
        Returns:
            Action string in the expected format, e.g.:
            "<think>...</think><action>move_forward|turn_left|</action>"
        """
        pass
    
    def reset(self):
        """Reset agent state for a new episode."""
        pass


class RandomAgent(BaseAgent):
    """
    Random baseline agent.
    Takes random valid actions each turn.
    Provides a lower bound on performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "random"
        self.rng = np.random.RandomState(config.get("seed", 42) if config else 42)
        self.max_actions_per_step = config.get("max_actions_per_step", 5) if config else 5
        self.done_probability = config.get("done_probability", 0.05) if config else 0.05
        self.step_count = 0
        self.max_steps_before_done = config.get("max_steps_before_done", 40) if config else 40
    
    def act(self, observation: Dict[str, Any], info: Dict[str, Any] = None) -> str:
        self.step_count += 1
        
        # Chance to say "done" increases over time
        if self.step_count >= self.max_steps_before_done:
            actions = ["done"]
        elif self.rng.random() < self.done_probability:
            actions = ["done"]
        else:
            n_actions = self.rng.randint(1, self.max_actions_per_step + 1)
            actions = [self.rng.choice(VALID_ACTIONS) for _ in range(n_actions)]
        
        action_str = "|".join(actions) + "|"
        return f"<think>Random exploration.</think><action>{action_str}</action>"
    
    def reset(self):
        self.step_count = 0


class HeuristicAgent(BaseAgent):
    """
    Oracle heuristic agent.
    Uses ground-truth task geometry (target_region, object positions) 
    to compute the optimal movement direction.
    Provides an approximate upper bound on performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "heuristic"
        self.step_translation = config.get("step_translation", 0.2) if config else 0.2
        self.step_rotation_deg = config.get("step_rotation_deg", 10.0) if config else 10.0
        self.step_count = 0
        self.prev_score = 0.0
        self.stagnation_count = 0
    
    def act(self, observation: Dict[str, Any], info: Dict[str, Any] = None) -> str:
        self.step_count += 1
        
        if info is None:
            return self._random_fallback()
        
        # Get current state
        current_pose = info.get("current_pose")
        current_score = info.get("current_potential_score", 0.0)
        task_info = info.get("task_info", {})
        target_region = task_info.get("target_region", {})
        
        if current_pose is None or not target_region:
            return self._random_fallback()
        
        # Extract current position and forward direction
        current_pos = np.array(current_pose[:3]) if len(current_pose) >= 3 else np.zeros(3)
        
        # Get sample target point from target region
        sample_point = target_region.get("sample_point")
        sample_forward = target_region.get("sample_forward")
        
        if sample_point is None:
            return self._random_fallback()
        
        target_pos = np.array(sample_point)
        
        # Compute direction to target
        direction = target_pos - current_pos
        distance_to_target = np.linalg.norm(direction[:2])  # 2D distance
        
        # Check if we're close enough
        if current_score >= 0.85:
            return "<think>Score is high enough, task complete.</think><action>done|</action>"
        
        # Detect stagnation
        if abs(current_score - self.prev_score) < 0.01:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        self.prev_score = current_score
        
        # Build action sequence
        actions = self._compute_navigation_actions(
            current_pos=current_pos,
            target_pos=target_pos,
            sample_forward=np.array(sample_forward) if sample_forward else None,
            info=info,
        )
        
        if not actions:
            actions = ["move_forward"]
        
        action_str = "|".join(actions) + "|"
        return f"<think>Heuristic: navigating toward target. Distance: {distance_to_target:.2f}m, Score: {current_score:.3f}</think><action>{action_str}</action>"
    
    def _compute_navigation_actions(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        sample_forward: Optional[np.ndarray],
        info: Dict,
    ) -> List[str]:
        """Compute a sequence of actions to move toward target."""
        actions = []
        
        # Get camera forward from current_pose (from extrinsics)
        current_pose_matrix = info.get("current_pose_matrix")
        if current_pose_matrix is not None:
            cam_forward = -np.array(current_pose_matrix)[:3, 2]  # -Z is forward
        else:
            # Approximate from SE3 pose
            cam_forward = np.array([1, 0, 0])  # Default
        
        # Direction to target in 2D
        direction = target_pos[:2] - current_pos[:2]
        dist = np.linalg.norm(direction)
        
        if dist < 0.05:
            # Very close, focus on orientation
            if sample_forward is not None:
                actions.extend(self._align_orientation(cam_forward, sample_forward))
            return actions if actions else ["done"]
        
        # Normalize direction
        direction_norm = direction / max(dist, 1e-6)
        
        # Compute angle between camera forward (2D) and target direction
        cam_forward_2d = cam_forward[:2]
        cam_forward_2d_norm = cam_forward_2d / max(np.linalg.norm(cam_forward_2d), 1e-6)
        
        # Cross product for turn direction, dot product for angle
        cross = cam_forward_2d_norm[0] * direction_norm[1] - cam_forward_2d_norm[1] * direction_norm[0]
        dot = np.dot(cam_forward_2d_norm, direction_norm)
        angle_deg = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        
        # Turn toward target first
        rotation_step = self.step_rotation_deg
        if angle_deg > rotation_step * 0.5:
            num_turns = min(3, int(angle_deg / rotation_step))
            turn_action = "turn_right" if cross > 0 else "turn_left"
            actions.extend([turn_action] * max(1, num_turns))
        
        # Move forward if roughly facing target
        if angle_deg < 60:
            num_steps = min(3, max(1, int(dist / self.step_translation)))
            actions.extend(["move_forward"] * num_steps)
        
        # Limit total actions
        return actions[:5]
    
    def _align_orientation(self, current_forward: np.ndarray, target_forward: np.ndarray) -> List[str]:
        """Generate actions to align camera orientation with target forward."""
        actions = []
        
        cf2d = current_forward[:2] / max(np.linalg.norm(current_forward[:2]), 1e-6)
        tf2d = target_forward[:2] / max(np.linalg.norm(target_forward[:2]), 1e-6)
        
        cross = cf2d[0] * tf2d[1] - cf2d[1] * tf2d[0]
        dot = np.dot(cf2d, tf2d)
        angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        
        if angle > self.step_rotation_deg * 0.5:
            turn = "turn_right" if cross > 0 else "turn_left"
            n = min(3, max(1, int(angle / self.step_rotation_deg)))
            actions.extend([turn] * n)
        
        # Pitch alignment  
        if len(current_forward) >= 3 and len(target_forward) >= 3:
            pitch_diff = target_forward[2] - current_forward[2]
            if abs(pitch_diff) > 0.1:
                pitch_action = "look_up" if pitch_diff > 0 else "look_down"
                actions.append(pitch_action)
        
        return actions
    
    def _random_fallback(self) -> str:
        actions = ["move_forward", "turn_left"]
        action_str = "|".join(actions) + "|"
        return f"<think>Heuristic fallback: no target info available.</think><action>{action_str}</action>"
    
    def reset(self):
        self.step_count = 0
        self.prev_score = 0.0
        self.stagnation_count = 0


class ConstantAgent(BaseAgent):
    """
    Always takes the same action. Useful for sanity checking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "constant"
        self.action = config.get("action", "move_forward") if config else "move_forward"
    
    def act(self, observation: Dict[str, Any], info: Dict[str, Any] = None) -> str:
        return f"<think>Constant agent.</think><action>{self.action}|</action>"


def create_agent(agent_type: str, config: Dict[str, Any] = None) -> BaseAgent:
    """Factory function to create agents."""
    agents = {
        "random": RandomAgent,
        "heuristic": HeuristicAgent,
        "constant": ConstantAgent,
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    
    return agents[agent_type](config)
