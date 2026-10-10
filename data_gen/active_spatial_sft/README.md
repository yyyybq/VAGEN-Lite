# active_spatial_sft — SFT Data Generator

Generates **Supervised Fine-Tuning (SFT)** training data for the active-spatial
camera-navigation task by finding score-guided camera trajectories and rendering
the corresponding image sequences.

> **R1 note (2026-10):** use `run_r1_sft_pipeline.py` for canonical R1 data.
> The older generic command below remains available for legacy multi-task data,
> but its defaults are not the current R1 protocol.

## Current R1 workflow

The R1 entry point binds generation to an authoritative environment YAML and
uses the same `canonical_gate_aligned_shaping_v1` potential as the runtime
reward. It also replays every planned trajectory through the real environment,
rejects collision/pose/render mismatches, validates the QwenVL multimodal
record, and produces score/path dashboards.

Example: apply the latest gate-aligned reward protocol to the complete frozen
210-row clean Projective corpus (without modifying either frozen input):

```bash
export R1_RENDER_URL=http://RENDER_HOST:PORT/render

python data_gen/active_spatial_sft/run_r1_sft_pipeline.py \
  --env-yaml \
    exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005/frozen/train.yaml \
  --jsonl-path \
    exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1/train.jsonl \
  --output-dir outputs/active_spatial_sft_r1_gate_aligned_20261008 \
  --qwen-format parquet \
  --also-no-think \
  --beam-width 16 \
  --verbose
```

Use a new output directory for every run. The generator refuses to overwrite
an existing SFT JSONL or report.

The important outputs are:

| Artifact | Purpose |
| --- | --- |
| `sft_data.jsonl` | Auditable internal records, conversations, frame paths, canonical gates, atomic pose/score trace, and runtime reward trace. |
| `qwen_vl_sft.parquet` (or `.jsonl`) | Strictly validated `messages` + absolute `images` records for QwenVL SFT. Oracle scores and certificates are not placed in messages. |
| `dataset_info.json` | LLaMA-Factory registration (`active_spatial_r1_sft`, plus the no-think variant when requested). |
| `generation_manifest.json` | Input/YAML/output hashes, resolved generation settings, and score version. |
| `primitive_images/` | Real renderer frames at the initial pose and after every primitive action; kept out of QwenVL messages. |
| `visualization/index.html` | Per-trajectory primitive RGB frames, action-level score curve, and top-down camera path. |
| `visualization/score_guidance_summary.json` | Success, score monotonicity, final-best rate, path directness, and certified shortest-length gap. |

For auto-terminated R1 episodes, the last rendered frame is retained in the
internal record and dashboards but is not appended as an unmatched final user
turn in the QwenVL conversation. Thus every exported training conversation ends
with a supervised assistant action.

By default the R1 runner uses **score-only beam search**. This is intentional:
it tests whether the current reward landscape itself can guide the camera from
the supplied initial view. `--guided-search` additionally uses the hidden
`sample_point`/`sample_forward`; use that only to maximize SFT yield, not to
evaluate the score system.

The audit manifest is auto-discovered beside the input as `audit_only.jsonl`.
For rows with a complete and equal certified lower/upper bound, the dashboard
can state whether the generated trajectory matches the certified shortest
action count. For all other rows it reports score-guided success and path
efficiency but does **not** claim global optimality.

### QwenVL fine-tuning handoff

For LLaMA-Factory, point `dataset_dir` to the output directory and select the
generated dataset name:

```yaml
dataset_dir: outputs/active_spatial_sft_r1_gate_aligned_20261008
dataset: active_spatial_r1_sft
template: qwen2_vl
```

Set `model_name_or_path`, training output, batch size, and distributed strategy
in the training config for the intended QwenVL checkpoint. Do not reuse the
hard-coded historical paths in `lf_qwen25vl_3b_sft.yaml` without overriding
them.

### What is guaranteed

- Search score equals the current canonical runtime shaping score.
- Canonical success still comes from boolean gates, not a numeric score cutoff.
- Published labels come from real environment replay, not search-only poses.
- Every user image placeholder has exactly one existing image before QwenVL
  export succeeds.
- Atomic score/path traces are metadata only and are excluded from model
  messages.

The full SFT publication step requires a live GPU renderer. Geometry-only
search can run on CPU, but it is evidence about the score landscape—not valid
multimodal SFT data.

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
  Score-guided path finder  ← discrete beam search on the active score
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
├── path_finder.py          # Score-guided trajectory search
├── sft_formatter.py        # Convert Trajectory → SFT conversation record
├── sft_generator.py        # Main generator class (orchestrates the pipeline)
├── run_sft_generation.py   # CLI entry point
├── run_r1_sft_pipeline.py  # R1 YAML-bound generation + QwenVL export
├── visualize_score_guidance.py # RGB/score/path dashboards
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

### 4. Legacy path-finding diagnostic (no rendering)

```bash
python test_path_finder.py \
    --jsonl_path /path/to/pipeline_output.jsonl \
    --num_items 10
```

This legacy diagnostic does not publish SFT records. The R1 runner deliberately
requires real rendered frames before it will emit QwenVL training data.

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
