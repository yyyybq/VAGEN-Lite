# active_spatial_sft — SFT Data Generator

Generates **Supervised Fine-Tuning (SFT)** training data for the active-spatial
camera-navigation task by finding optimal camera trajectories and rendering the
corresponding image sequences.

---

## Overview

The [active_spatial_pipeline](../active_spatial_pipeline/) already produces:
- A 3D scene (`scene_id` → Gaussian Splatting `.ply`)
- An initial camera pose (`init_camera` → 4 × 4 c2w matrix)
- A **target region** + scoring rules (`SpatialPotentialField`) that give every
  camera pose a score ∈ [0, 1]

What it does **not** produce is a sequence of actions that reaches that target —
the RL environment discovers those through self-play.

This package bridges the gap:

```
Pipeline JSONL item
  (init_camera + target_region + task_type)
         │
         ▼
  GreedyPathFinder          ← greedy hill-climbing on SpatialPotentialField
  (path_finder.py)
         │  trajectory: [TrajStep, ...]
         ▼
  UnifiedRenderGS           ← render one image per step
         │  [PIL.Image, ...]
         ▼
  SFTFormatter              ← build multi-turn conversation
  (sft_formatter.py)
         │
         ▼
  Output JSONL + images/
```

Each output record is a **multi-turn conversation** in the same format the RL
environment uses (system → user/assistant alternation), ready for VLM SFT.

---

## File Structure

```
active_spatial_sft/
├── __init__.py
├── config.py               # SFTGenerationConfig dataclass (all hyperparameters)
├── path_finder.py          # Greedy trajectory search on potential field
├── sft_formatter.py        # Convert Trajectory → SFT conversation record
├── sft_generator.py        # Main generator class (orchestrates the pipeline)
├── run_sft_generation.py   # CLI entry point
├── test_path_finder.py     # Sanity-check script (no rendering required)
└── README.md               # This file
```

---

## Quick Start

### 1. Test path-finding (no GPU / render server needed)

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN/data_gen/active_spatial_sft

python test_path_finder.py \
    --jsonl_path /path/to/pipeline_output.jsonl \
    --num_items 10 \
    --verbose
```

Expected output:
```
Item 0 [SUCCESS]: steps=7, actions=21, score 0.1234 → 0.9612
  → 8 assistant turns in conversation
...
Summary: 8/10 trajectories succeeded.
```

### 2. Full generation with local rendering

```bash
python run_sft_generation.py \
    --jsonl_path /path/to/pipeline_output.jsonl \
    --gs_root    /path/to/gaussian_scenes \
    --output_dir /path/to/sft_output \
    --render_backend local \
    --gpu_device 0 \
    --max_items 1000 \
    --verbose
```

### 3. Full generation with remote render server

```bash
# First start the render server on a GPU node:
#   cd /scratch/.../VAGEN/vagen/env/active_spatial && bash start_ray_server.sh

python run_sft_generation.py \
    --jsonl_path  /path/to/pipeline_output.jsonl \
    --gs_root     /path/to/gaussian_scenes \
    --output_dir  /path/to/sft_output \
    --render_backend client \
    --client_url  ws://localhost:8777/render/interiorgs \
    --max_items   5000
```

### 4. Path-finding only (skip rendering, no images)

```bash
python run_sft_generation.py \
    --jsonl_path /path/to/pipeline_output.jsonl \
    --output_dir /path/to/sft_output \
    --render_backend none \
    --max_items 1000
```
*(Records will have empty `image_paths`. Useful for validating trajectories.)*

---

## Output Format

### Directory structure

```
sft_output/
├── sft_data.jsonl          # One record per trajectory
├── sft_data_stats.json     # Summary statistics
└── images/
    ├── sft_000000_step00.jpg   # Initial view for item 0
    ├── sft_000000_step01.jpg   # View after step 0's actions
    ├── sft_000000_step02.jpg   # …
    └── …
```

### JSONL record schema

```json
{
  "id": "sft_000001",
  "source_item_idx": 42,
  "scene_id": "0267_840790",
  "task_type": "absolute_positioning",
  "task_description": "Navigate to be 1.5m from the sofa...",
  "trajectory_steps": 8,
  "total_actions": 21,
  "initial_score": 0.1234,
  "final_score": 0.9612,
  "success": true,
  "image_paths": [
    "images/sft_000001_step00.jpg",
    "images/sft_000001_step01.jpg",
    "..."
  ],
  "conversations": [
    {
      "role": "system",
      "content": "You are a spatial navigation agent..."
    },
    {
      "role": "user",
      "content": "[Initial Observation]:\n<image>\nCurrent camera pose: [...]\nTask: ...",
      "image_path": "images/sft_000001_step00.jpg"
    },
    {
      "role": "assistant",
      "content": "<think>Current score: 0.123 (position: 0.089, orientation: 0.201). The main challenge is positioning. I will move closer to make progress. Expected score after these actions: 0.312 (improvement: +0.189).</think>\n<action>move_forward|move_forward|turn_left|</action>"
    },
    {
      "role": "user",
      "content": "[Observation]:\n<image>\nCurrent camera pose: [...]\nEnvironment Feedback: Action executed.",
      "image_path": "images/sft_000001_step01.jpg"
    },
    "...",
    {
      "role": "assistant",
      "content": "<think>My current score is 0.961, which meets the success threshold of 0.95. I will issue 'done' to complete the task.</think>\n<action>done|</action>"
    }
  ]
}
```

Each `"user"` turn that requires an image has an `"image_path"` field pointing to
the corresponding file in `images/`.

---

## Path-Finding Algorithm

The **greedy hill-climber** (`path_finder.find_trajectory`) works as follows:

1. Start from `init_c2w` (initial camera pose).
2. **Each LLM turn**: try all 6 movement actions (`move_forward`, `move_backward`,
   `turn_left`, `turn_right`, `look_up`, `look_down`) by simulating them on a copy
   of the c2w matrix.
3. Select the action with the highest potential-field score improvement.
4. Repeat up to `max_actions_per_turn` times per turn (packing multiple actions
   per assistant response, just like the RL agent does).
5. If no action improves the score by at least `min_improvement`, increment a
   plateau counter; after `plateau_tolerance` consecutive turns, try rotation-only
   "escape" moves.  If still stuck, terminate.
6. Terminate early when `score ≥ success_threshold` (default 0.95).

The simulation mirrors `ViewManipulator.step()` exactly, so replaying the action
sequence in the RL environment produces identical camera poses.

---

## Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `success_threshold` | `0.95` | Score to declare success (matches RL env) |
| `max_actions_per_turn` | `5` | Actions per LLM turn (matches RL env) |
| `min_improvement` | `0.005` | Minimum score gain to accept an action |
| `plateau_tolerance` | `5` | Turns without improvement before stopping |
| `only_successful` | `True` | Only save trajectories that succeed |
| `prompt_format` | `free_think` | Conversation format (matches RL training) |
| `add_think` | `True` | Include `<think>` reasoning blocks |
| `render_backend` | `local` | `local` / `client` / `none` |

All parameters mirror the RL environment defaults to ensure SFT data is
in-distribution with the RL rollouts.

---

## Using SFT Data for Training

The output JSONL can be used directly with standard VLM SFT frameworks.
Each record's `conversations` list is a standard system/user/assistant chat,
and the images are referenced by `image_path` in each user turn.

To convert to a specific training framework format, post-process the JSONL:

```python
import json
from pathlib import Path

output_dir = Path("/path/to/sft_output")
with open(output_dir / "sft_data.jsonl") as f:
    for line in f:
        record = json.loads(line)
        # record["conversations"] – list of {"role", "content", "image_path"?}
        # record["image_paths"]   – all image paths for this trajectory
        # ...
```
