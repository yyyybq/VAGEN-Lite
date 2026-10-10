# Train a VLM on RoboCasa demos

完整中文训练路线：[Active Spatial → 混合 SFT → 环境 RL → StarVLA → RLinf](../../../docs/active_spatial_end_to_end.md)。

从训练后的 StarVLA policy 继续 RoboCasa365 RL，使用
`scripts/prepare_rlinf_robocasa.py` 和 `run_rlinf_starvla.sh`；详细步骤见
[StarVLA / RLinf 交接](../../../docs/active_spatial_pipeline_robot.md)。

This is a thin wrapper around `data_gen/robocasa_sft/run_sft.sh`.

```bash
# Original VLM weights
bash examples/train/robocasa/run_sft.sh \
  --model-path "$QWEN_PATH" \
  --data-root /path/to/lerobot_robocasa \
  --task PickPlaceCounterToCabinet \
  --gpus 0,1

# Active Spatial trained HF actor (config.json + safetensors + tokenizer)
bash examples/train/robocasa/run_sft.sh \
  --model-path "$ACTIVE_SPATIAL_HF_CKPT" \
  --data-root /path/to/lerobot_robocasa \
  --task-set atomic_seen \
  --gpus 0,1,2,3
```

See `data_gen/robocasa_sft/README.md` and `docs/robocasa_vlm_sft_eval.md`.

## StarVLA (NavigateKitchen, action_dim=12)

LLaMA-Factory above is a text-12D smoke path. The intended VLA route is
StarVLA-OFT with the official PandaOmron 12-D head on the existing LeRobot
layout. Local NavigateKitchen is **pretrain**, not target:

```
datasets/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot
```

```bash
# Check packed LeRobot 12-D vs StarVLA concat order, then make a writable overlay.
CHECK_ONLY=1 bash examples/train/robocasa/run_starvla_navigatekitchen.sh

# 2-step smoke (vagen-lite, 1x GPU, Qwen2.5-VL-3B, action_dim=12)
bash examples/train/robocasa/sco_starvla_nk_smoke_entry.sh

# Official 8x H800 train via SCO (starVLA env, Qwen3-VL-4B, DeepSpeed, flash-attn)
bash examples/train/robocasa/submit_starvla_nk_sco.sh
```

Checkpoints go to `third_party/starVLA/playground/Checkpoints/`.
The official stack is `~/.conda/envs/starVLA` (Python 3.10, torch 2.6+cu124).
1x H800 ZeRO-2 full-finetune OOMs; use the 8-GPU SCO job.

`action_dim=12` is gym/eval order (`eef + rot + gripper + base + mode`).
The parquet column is packed `base + mode + eef + rot + gripper`; StarVLA
reassembles it through `meta/modality.json`.

## Transfer an Active Spatial HF actor

Use `scripts/prepare_starvla_transfer.py` followed by
`run_starvla_transfer.sh <prepared-config.yaml>`. This path validates the actual
HF actor, freezes a trajectory-disjoint train/validation split, and supports
multiple PandaOmron tasks. The existing Qwen3-VL-4B NavigateKitchen run did not
initialize from an Active Spatial actor.

See `docs/robocasa_vlm_sft_eval.md`, section **Active Spatial → low-level mobile
manipulation transfer**, for matched base/SFT/SFT+RL experiments, held-out action
evaluation, closed-loop execution, and VLM export for spatial retention checks.
