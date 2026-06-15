"""
Active Spatial SFT Data Generation Package

Generates supervised fine-tuning (SFT) data by finding optimal camera trajectories
from initial positions to target regions, using the same scoring system as the RL environment.
"""

from .config import SFTGenerationConfig
from .path_finder import find_trajectory, simulate_action, Trajectory, TrajStep
from .sft_generator import SFTDataGenerator

__all__ = [
    "SFTGenerationConfig",
    "find_trajectory",
    "simulate_action",
    "Trajectory",
    "TrajStep",
    "SFTDataGenerator",
]
