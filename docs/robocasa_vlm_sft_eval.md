# RoboCasa VLM SFT and evaluation

Two parts, kept separate from kitchen Python 3.13 and from Diffusion Policy training:

1. **Evaluation.** Plug RoboCasa into `vagen.evaluate` the same way as PrimitiveSkill.
2. **SFT.** Fine-tune original VLM weights *or* an Active Spatial HF actor on
   RoboCasa LeRobot demos via LLaMA-Factory.

## Why a remote server

RoboCasa + MuJoCo + `PandaOmronKeyConverter` live in the **policy venv
(Python 3.10)**. VAGEN-Lite eval / kitchen stacks often use **Python 3.13**.
Mixing those interpreters breaks EGL / gym registration.

Recommended path:

```
policy venv  -->  python -m vagen.envs.robocasa.serve --port 8010
VAGEN python -->  RemoteEnv client (examples/evaluate/robocasa/config.yaml)
```

In-process eval (`name: RoboCasa`) is supported only when the eval process
itself is the policy venv (`run_eval_local.sh`).

## Action format

VLM actions are **text**. Official RoboCasa gym actions are a 12-D
Panda-Omron OSC dict from `robocasa.wrappers.gym_wrapper.PandaOmronKeyConverter`.

Canonical order:

```
[eef_dx, eef_dy, eef_dz, eef_droll, eef_dpitch, eef_dyaw,
 gripper_close, base_x, base_y, base_yaw, torso, control_mode]
```

Gym mapping:

| slice | gym key |
| --- | --- |
| `[0:3]` | `action.end_effector_position` |
| `[3:6]` | `action.end_effector_rotation` |
| `[6:7]` | `action.gripper_close` |
| `[7:11]` | `action.base_motion` |
| `[11:12]` | `action.control_mode` |

The model uses Active Spatial tags so format priors transfer:

```
<think>...</think><action>
[0.02, 0.00, -0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0]
</action>
```

One VLM turn may emit several 12-D vectors (`action_horizon`); they are
executed sequentially. Eval sets `concat_multi_turn: false` to match
single-turn SFT.

Observations:

- primary image: `video.robot0_agentview_left`
- optional wrist: `video.robot0_eye_in_hand`
- language: `annotation.human.task_description` (from `env.get_ep_meta()['lang']`)
- success: `info['success']`

## Evaluate

```bash
# 1) server (policy venv)
bash examples/evaluate/robocasa/run_server.sh

# 2) client (VAGEN python)
python scripts/robocasa_eval.py \
  --config examples/evaluate/robocasa/config.yaml \
  --task PickPlaceCounterToCabinet \
  --backend openai \
  --model gpt-4.1-mini
```

Full atomic seen split:

```bash
python scripts/robocasa_eval.py --task_set atomic_seen --backend openai
```

Local (policy venv only):

```bash
bash examples/evaluate/robocasa/run_eval_local.sh
```

## SFT from base vs Active Spatial

Same recipe. Only `--model-path` changes.

```bash
# Base Qwen2.5-VL-3B-Instruct
bash examples/train/robocasa/run_sft.sh \
  --model-path "$QWEN_PATH" \
  --data-root /path/to/lerobot_robocasa \
  --task PickPlaceCounterToCabinet \
  --gpus 0,1,2,3

# Active Spatial HF actor directory
# (must contain config.json + *.safetensors + tokenizer files)
bash examples/train/robocasa/run_sft.sh \
  --model-path "$ACTIVE_SPATIAL_HF_CKPT" \
  --data-root /path/to/lerobot_robocasa \
  --task-set atomic_seen \
  --gpus 0,1,2,3
```

`run_sft.sh` converts LeRobot v2 (`meta/info.json`, `data/*.parquet`,
`videos/`) to sharegpt parquet, then launches LLaMA-Factory full SFT with
`freeze_vision_tower=true` and `template=qwen2_vl`.

## Layout

| path | role |
| --- | --- |
| `vagen/envs/robocasa/` | GymImageEnv, handler, GymService serve |
| `examples/evaluate/robocasa/` | Remote + local eval configs |
| `data_gen/robocasa_sft/` | LeRobot → SFT + LLaMA-Factory |
| `scripts/robocasa_eval.py` | First-class eval CLI |

## StarVLA NavigateKitchen (action_dim=12)

LLaMA-Factory 12-D text SFT is only a smoke path. The VLA route is StarVLA
QwenOFT (`third_party/starVLA`) with the official PandaOmron contract:

| tensor | dim | order |
| --- | --- | --- |
| packed LeRobot `action` | 12 | `base_motion(4) + control_mode(1) + eef_pos(3) + eef_rot(3) + gripper(1)` |
| StarVLA / gym action | 12 | `eef_pos(3) + eef_rot(3) + gripper(1) + base_motion(4) + control_mode(1)` |
| state | 16 | `base_pos(3) + base_rot(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)` |

Named keys in `PandaOmronRoboCasa365DataConfig` match gym dict names. The
loader slices the packed column with `meta/modality.json`, so the existing
LeRobot dump is used as-is.

Local NavigateKitchen lives under **pretrain** (the official StarVLA registry
only listed target/human):

```
<datasets>/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot
```

Mixture: `robocasa365_navigate_kitchen_pretrain_human`.

```bash
# Verify 12-D packing + mixture + overlay, no training
CHECK_ONLY=1 bash examples/train/robocasa/run_starvla_navigatekitchen.sh

# Train inside the starVLA env (not kitchen 3.13, not the policy venv)
bash examples/train/robocasa/run_starvla_navigatekitchen.sh
```

YAML: `third_party/starVLA/examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_navigatekitchen.yaml`.

## Active Spatial → low-level mobile manipulation transfer

The completed `starvla_qwenoft_NavigateKitchen_20261001_025134` run used the
Qwen3-VL-4B-Instruct-Action backbone, **not an Active Spatial SFT/RL actor**.
Its 100k-step completion and successful offline inference validate the original
pipeline; they do not establish transfer from Active Spatial. NavigateKitchen
alone does not teach grasping, placing or long-horizon mobile manipulation.

Use the same backbone family/size for three controlled initializations:
`base`, `sft`, and `sft_rl`. Keep robot demonstrations, episode split, seeds,
action head initialization, optimizer budget and evaluation episodes identical.
Qwen2.5-VL weights cannot be inserted into Qwen3-VL. Cambrian / SenseNova need
separate StarVLA backbone adapters and are not supported by this entry point.

The action head predicts normalized PandaOmron OSC controller commands, not
joint torques: end-effector deltas, gripper, base motion, torso and mode. It is
trained with demonstration action supervision. The inherited VLM weights carry
the Active Spatial initialization; the new MLP head starts from random weights.
The effect of that initialization must be measured, not assumed.

### Prepare a reproducible experiment

Run in the `starVLA` Python environment. Each source must be a complete, readable
HF export with original tokenizer/processor and all weight shards. An
`actor/huggingface` directory with only config files is insufficient; first use
the matching verl model merger to export the FSDP actor. LoRA-only exports must
be merged with their original base. The preparation tool fails on these cases
instead of silently loading a base model.

For an FSDP actor that has not yet been exported, use the Python environment
matching that RL run (including its PyTorch version):

```bash
PYTHONPATH="$PWD/verl${PYTHONPATH:+:$PYTHONPATH}" "$RL_PY" -m verl.model_merger merge \
  --backend fsdp --local_dir "$RL_ACTOR_DIR" --target_dir "$SFT_RL_HF"
```

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python
DATA=/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets

"$PY" scripts/prepare_starvla_transfer.py \
  --dataset NavigateKitchen=playground/Datasets/robocasa365/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot \
  --dataset PickPlaceCounterToCabinet="$DATA/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot" \
  --source base="$BASE_HF" --source sft="$SFT_HF" --source sft_rl="$SFT_RL_HF" \
  --out playground/Experiments/starvla_transfer/comparison_v1

CHECK_ONLY=1 bash examples/train/robocasa/run_starvla_transfer.sh \
  playground/Experiments/starvla_transfer/comparison_v1/sft_rl.yaml
NUM_GPUS=8 bash examples/train/robocasa/run_starvla_transfer.sh \
  playground/Experiments/starvla_transfer/comparison_v1/sft_rl.yaml
```

Preparation creates per-episode symlinks, copies metadata, holds out 10% of
episodes, and leaves original datasets untouched. Train and validation frames
cannot overlap across an episode. Only training parquet files contribute to
StarVLA normalization. Video directories are shared read-only by convention;
the training metadata contains only selected training episode IDs. New configs
drop incomplete 16-step action chunks. The source manifest records config and
processor hashes plus weight file size/mtime (not full weight hashes); source
labels are user-declared provenance, not proof of prior SFT/RL training.

The prepared two-task pilot at
`playground/Experiments/starvla_transfer/pilot_20261002` contains 453/50
NavigateKitchen train/validation episodes and 452/50 PickPlaceCounterToCabinet
episodes. Its `base.yaml` uses Qwen2.5-VL-3B and is a preparation/dataloader
validation artifact; it is not an SFT/RL transfer result. Add composite tasks
using further `--dataset NAME=/path/to/lerobot` arguments in a new experiment.
All datasets in this recipe must use the same PandaOmron contract. Do not mix
other embodiments by padding or relabeling their action columns.

### Evaluate action fitting, execution and retained spatial ability

1. **Held-out actions.** The trainer's `mse_score` samples its training iterator.
   Use the separate evaluator below for unseen trajectories; it reuses the
   checkpoint's training statistics and reports raw action MAE/MSE per task,
   dimension and action group. Its manifest check rejects old all-data runs.
2. **Closed-loop success.** Evaluate seeded episodes, separately for pretrain
   and target scene distributions. Training on target demonstrations means
   target evaluation is not zero-shot scene transfer. A random episode holdout
   also does not by itself guarantee unseen layouts or objects.
3. **Active Spatial retention.** Export the fine-tuned VLM and evaluate the same
   frozen Active Spatial ID/OOD suites as before robot action training. This
   detects whether action fine-tuning erased the spatial capability.

```bash
"$PY" scripts/eval_starvla_heldout.py --ckpt "$CKPT" \
  --out playground/Experiments/starvla_transfer/heldout_v1

# terminal 1: StarVLA policy server
CKPT="$CKPT" bash examples/evaluate/robocasa/run_starvla_mobile.sh server
# terminal 2: isolated RoboCasa simulator
CKPT="$CKPT" bash examples/evaluate/robocasa/run_starvla_mobile.sh client \
  --task NavigateKitchen --task PickPlaceCounterToCabinet \
  --split target --episodes 50 --seed 42 --max-steps 500 --execute-steps 8 \
  --video --out playground/Experiments/starvla_transfer/closed_loop_v1

"$PY" scripts/export_starvla_backbone.py --ckpt "$CKPT" \
  --out playground/Pretrained_models/spatial_after_robot_sft
```

`run_starvla_mobile.sh` is the transfer evaluation entry point. It reads the
saved training configuration, supplies left/right/wrist cameras in that order,
uses the training PIL resize, and omits state when `include_state=false`.
For state-conditioned runs it applies the same per-key sin/cos transform as
training. It verifies server checkpoint identity and action ordering, rejects
nonfinite outputs, clips unnormalized actions to the environment's bounds,
replans every `execute_steps`, and stops an action chunk immediately on success
or termination. Reports include seeds, instructions, backend, step counts,
latency, per-task success and tracebacks for incomplete runs.

### Rendering and runtime isolation

Policy inference runs in `starVLA`; MuJoCo runs in the RoboCasa policy venv.
Do not merge either environment into the kitchen Python 3.13 environment.
Run the renderer preflight before loading a policy:

```bash
bash examples/evaluate/robocasa/run_starvla_mobile.sh preflight --out /absolute/new/render_check
```

The current Ubuntu container lacks the EGL dispatcher. The project-local
overlay below downloads Ubuntu packages without a system install. NVIDIA EGL
still requires working graphics driver/device exposure on the worker. When that
is unavailable, OSMesa provides CPU rendering while policy inference stays on
GPU. Record the backend and expect slower simulation.
The OSMesa entry also defaults `NUMBA_DISABLE_JIT=1`: this container's llvmlite
optimizer segfaulted during kitchen placement with JIT enabled. Reports record
this setting. A successful small-scene renderer preflight alone is not proof
that a full kitchen episode can execute.

```bash
bash examples/evaluate/robocasa/setup_render_runtime.sh
"$PY" -m pip install --no-deps --target playground/Runtime/robocasa_client \
  -r examples/evaluate/robocasa/client_requirements.txt
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  bash examples/evaluate/robocasa/run_starvla_mobile.sh preflight --out /absolute/new/osmesa_check
```

`sco_starvla_mobile_smoke_entry.sh` runs a bounded two-episode test of the old
NavigateKitchen baseline and records EGL/OSMesa preflight outcomes. This is a
plumbing test, not a statistical estimate of navigation performance.

Verified on 2026-10-02: job `pt-f6ctnhzt` completed both target-split episodes
using one H800 for inference and OSMesa with JIT disabled for simulation.
Seeds 42/43 requested coffee-machine/sink navigation. Each executed 200 control
steps and 25 policy calls; neither succeeded (0/2). Videos and the complete
report are in
`playground/Experiments/starvla_transfer/closed_loop_smoke_20261002T210042Z/eval/`.
This resolves the execution-path blocker, not the policy-quality problem.

CPU validation: 10 transfer tests plus 11 existing action tests passed. The
real two-task dataloader produced three camera views and `(16,12)` actions;
training normalization round-tripped raw actions, and held-out samples from
both tasks decoded successfully. Logs are in the prepared pilot directory.
The new held-out model scoring and full backbone export still require an
actual transfer checkpoint; only their data/extraction helpers have been tested.

The integration also changes the nested `third_party/starVLA` checkout:
config-based local Qwen backbone selection, strict Qwen3 weight shape loading,
honoring the attention backend, and explicit mixture specs shared by training
and normalization. Preserve those changes with the nested checkout when moving
this experiment; the parent repository ignores `third_party/`.
The new integration changes are also captured in
`examples/train/robocasa/patches/active_spatial_transfer.patch` for review and
reapplication to the matching StarVLA revision (`4507931`).

Completion criteria for the research goal: real SFT/RL source selected and
verified; matched action fine-tuning runs completed; held-out action errors and
closed-loop navigation/manipulation/composite-task success measured; Active
Spatial retention compared. Deployment on another robot additionally requires
that robot's camera/calibration, controller/action adapter, action-labelled data
and hardware validation. PandaOmron simulation checkpoints alone do not provide
that cross-embodiment deployment.

Upstream references: [StarVLA](https://github.com/starVLA/starVLA),
[RoboCasa dataset overview](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html),
[RoboCasa benchmarking splits](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html).
