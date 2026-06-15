"""
Configuration for SFT data generation from active spatial navigation tasks.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTGenerationConfig:
    """
    Configuration for generating SFT training data from active spatial navigation.

    The pipeline:
    1. Reads items from a pipeline JSONL (produced by active_spatial_pipeline)
    2. For each item: uses greedy search on the SpatialPotentialField to find
       the optimal camera trajectory from init_camera to target_region
    3. Replays the trajectory, rendering images at each step
    4. Formats each trajectory as a multi-turn conversation (SFT format)
    5. Saves to a new JSONL file with image references
    """

    # ──────────────────────────── Input ────────────────────────────
    jsonl_path: str = ""
    """Path to the pipeline JSONL file (output of active_spatial_pipeline)."""

    gs_root: str = ""
    """Root directory containing Gaussian Splatting scene folders
    (each scene folder is named by scene_id)."""

    # ──────────────────────────── Output ───────────────────────────
    output_dir: str = ""
    """Directory where the SFT JSONL and images will be saved."""

    output_name: str = "sft_data"
    """Base name for output files (without extension)."""

    # ──────────────────────────── Rendering ────────────────────────
    render_backend: str = "local"
    """Rendering backend: 'local' (GPU direct), 'client' (WebSocket server), or 'none'."""

    client_url: str = "ws://127.0.0.1:8777/render/interiorgs"
    """WebSocket URL for client rendering backend."""

    client_origin: Optional[str] = None
    """Optional Origin header for WebSocket connection."""

    gpu_device: Optional[int] = None
    """GPU device ID for local rendering. None = auto-detect from CUDA_VISIBLE_DEVICES."""

    image_width: int = 512
    """Width of rendered images in pixels."""

    image_height: int = 512
    """Height of rendered images in pixels."""

    save_images: bool = True
    """If True, save images to disk and reference by path. If False, embed as base64."""

    image_format: str = "jpg"
    """Image file format: 'jpg' or 'png'."""

    image_quality: int = 85
    """JPEG quality (1-95). Only used when image_format='jpg'."""

    # ──────────────────────────── Camera ───────────────────────────
    step_translation: float = 0.3
    """Translation step size in meters. Must match the RL env's step_translation."""

    step_rotation_deg: float = 30.0
    """Rotation step size in degrees. Must match the RL env's step_rotation_deg."""

    # ──────────────────────────── Path Finding ─────────────────────
    success_threshold: float = 0.95
    """Potential field score threshold for declaring success (same as RL env)."""

    max_total_actions: int = 200
    """Maximum total number of individual actions per trajectory."""

    max_actions_per_turn: int = 5
    """Maximum actions bundled into one LLM turn (same as RL env's max_actions_per_step).

    Design choice: 5 (default, matches RL eval format) vs 1 (dense, per-action feedback).

    max_actions_per_turn=5 (sparse):
        - Matches RL inference format exactly.
        - Shorter conversation sequences (avg 4 turns vs 17+ turns).
        - Model must plan multiple steps ahead from a single image.

    max_actions_per_turn=1 (dense):
        - Model sees the visual consequence of every individual action.
        - Richer training signal per observation; easier to learn cause-effect.
        - Sequences are ~4× longer; each sample costs more tokens.
        - Does NOT match RL eval format; use --max_actions_per_turn 5 at inference.

    Use 1 for initial SFT warmup (rich per-step signal), then fine-tune with 5
    to recover the multi-action bundling behavior expected by the RL environment.
    """

    min_improvement: float = 0.005
    """Base minimum score improvement required to accept a turn's result.
    When adaptive_min_improvement=True this is scaled down near success_threshold."""

    plateau_tolerance: int = 5
    """Number of consecutive no-improvement turns before giving up."""

    beam_width: int = 3
    """Beam width for action search within each LLM turn.
    1 = original greedy behaviour (fast, may miss paths requiring short-term score drops).
    3+ = beam search (tolerates short-term drops, e.g. 'turn 180° then advance').
    Recommended: 3 for SFT data generation (3x compute per turn, much higher success rate)."""

    adaptive_min_improvement: bool = True
    """If True, scale min_improvement down proportionally to remaining headroom as the
    score approaches success_threshold. Prevents premature termination in the final
    approach phase where each step can only gain a small fraction of the threshold."""

    use_guided_search: bool = True
    """If True (default), use geometry-guided path planning:
    Phase 1: navigate directly to target_region["sample_point"] (distance minimisation).
    Phase 2: rotate to face the object centre (angle minimisation).
    Phase 3: beam-search fine-tune to overcome discretisation residual.
    Falls back to pure beam/greedy search when target geometry is unavailable.
    Set to False to use purely score-based beam/greedy search."""

    # ──────────────────────────── Potential Field ──────────────────
    position_weight: float = 0.7
    """Weight for the position component of the potential field score."""

    orientation_weight: float = 0.3
    """Weight for the orientation component of the potential field score."""

    max_distance: float = 5.0
    """Maximum distance used in potential field normalization."""

    # ──────────────────────────── Collision Detection ──────────────
    enable_collision_detection: bool = True
    """Whether to use collision detection when finding paths."""

    collision_camera_radius: float = 0.15
    """Camera collision sphere radius in meters."""

    collision_floor_height: float = 0.3
    """Minimum camera Z height in meters."""

    collision_ceiling_height: float = 2.5
    """Maximum camera Z height in meters."""

    collision_safety_margin: float = 0.05
    """Extra safety margin around objects for collision detection."""

    # ──────────────────────────── Data Selection ───────────────────
    max_items: int = -1
    """Maximum number of JSONL items to process. -1 = process all."""

    start_idx: int = 0
    """Start from this item index (for resuming or parallel processing)."""

    end_idx: int = -1
    """End at this item index (exclusive). -1 = process until max_items or end of file."""

    skip_failed: bool = True
    """If True, skip items where no valid path is found. If False, raise an error."""

    min_trajectory_steps: int = 2
    """Minimum number of LLM turns required for a trajectory to be kept."""

    max_trajectory_steps: int = 50
    """Maximum number of LLM turns allowed in a trajectory."""

    only_successful: bool = True
    """If True (default), only save trajectories that reach the success threshold.

    Setting this to False includes ALL trajectories regardless of final score.
    Those without a valid path will still be skipped by skip_failed=True.
    Near-miss trajectories (score > partial_success_min_score) can be included
    WITHOUT a 'done' action to provide recovery-navigation examples.
    See also: partial_success_min_score, include_goal_reached_examples.
    """

    partial_success_min_score: float = 0.0
    """When only_successful=False, include trajectories with final score >= this value.
    Set to e.g. 0.5 to include near-misses while discarding very poor paths."""

    include_goal_reached_examples: bool = False
    """If True, scan each pipeline item for the INITIAL camera pose: if its score is
    already >= success_threshold, emit a one-turn example where the model immediately
    issues 'done'.  This teaches the model to recognize task completion from visual
    input rather than counting steps.

    Background (the 'stopping-boundary' problem)
    --------------------------------------------
    Without this, 'done' always appears as the very last assistant turn in a
    long successful trajectory.  The model may learn to associate 'done' with
    "sequence end" rather than "task is visually complete".  At inference time
    this causes the model to keep navigating even after reaching the target.
    Adding immediate-done examples grounds the decision in visual content.
    """

    # ──────────────────────────── Format ───────────────────────────
    prompt_format: str = "free_think"
    """Prompt format style: 'free_think', 'no_think', 'grounding', 'worldmodeling'."""

    add_think: bool = True
    """Whether to include <think> reasoning blocks in the assistant responses."""

    include_score_in_think: bool = False
    """Whether to include numerical potential-field scores in the <think> reasoning.
    Defaults to False: oracle scores are not available at inference time and
    including them teaches the model to hallucinate numbers rather than reason
    from visual observations.  Set True only for debugging / ablation studies."""

    # ──────────────────────────── Misc ─────────────────────────────
    verbose: bool = False
    """Print detailed progress information."""

    seed: int = 42
    """Random seed (currently unused, for future use)."""
