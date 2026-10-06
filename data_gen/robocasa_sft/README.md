# RoboCasa VLM SFT

Fine-tune **original VLM weights** (Qwen2.5-VL-3B-Instruct) **or** an
**Active Spatial HF actor checkpoint** on RoboCasa LeRobot demos.

The conversation format matches eval:

```
system: kitchen VLA + 12-D action spec
user:   <image> + Task: ...
assistant: <think>...</think><action>[12-D] ...</action>
```

## Convert + train

```bash
# Base Qwen
bash data_gen/robocasa_sft/run_sft.sh \
  --model-path "$QWEN_PATH" \
  --data-root /path/to/lerobot_robocasa \
  --task PickPlaceCounterToCabinet \
  --gpus 0,1,2,3

# Active Spatial HF actor (config.json + safetensors + tokenizer)
bash data_gen/robocasa_sft/run_sft.sh \
  --model-path "$ACTIVE_SPATIAL_HF_CKPT" \
  --data-root /path/to/lerobot_robocasa \
  --task-set atomic_seen \
  --gpus 0,1,2,3
```

`--model-path` is required and is passed through to LLaMA-Factory as
`model_name_or_path`. Full FT, `freeze_vision_tower=true`, `template=qwen2_vl`.

Convert only:

```bash
python data_gen/robocasa_sft/convert_lerobot_to_sft.py \
  --data-root /path/to/lerobot_robocasa \
  --task PickPlaceCounterToCabinet \
  --split target \
  --stride 4 \
  --horizon 8 \
  --output-dir ./outputs/robocasa_sft
```
