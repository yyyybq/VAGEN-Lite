#!/usr/bin/env bash
# T2's short run (the user's decision of 2026-10-27; an 8-card sco job): train EXP (a LoRA experiment of a few hundred
# steps, made with make_experiment.py --lora ...) to its end in one launch, with the same environment as
# run_segmented.sh, logging GPU memory and use every 10 s (EXP/gpu.csv); no evaluation.  Then lora_check.py compares
# the merged weights-only checkpoint with the full state at the first save point and records speed and memory
# (EXP/lora_check.json).  Usage: run_t2_short.sh EXP PLAIN_CKPT
set -uo pipefail
EXP="${1:?Usage: run_t2_short.sh EXP PLAIN_CKPT}"
PLAIN="${2:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
log() { echo "[t2-short] $* $(date -u +%FT%TZ)" | tee -a "$EXP/segments.log"; }
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 10000 >> "$EXP/gpu.csv" &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null' EXIT
log "train $EXP"
(
  export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
  export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
  source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
  set +e
  export PYTHONPATH="$WT:$STARVLA_ROOT"
  cd "$STARVLA_ROOT"
  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
    "$WT/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" >> "$EXP/train.log" 2>&1
)
rc=$?
log "train rc=$rc"
(
  source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
  export PYTHONPATH="$WT:$STARVLA_ROOT"
  python "$WT/downstream/train/lora_check.py" --exp "$EXP" --plain "$PLAIN"
)
crc=$?
log "check rc=$crc $(head -c 600 "$EXP/lora_check.json" 2>/dev/null | tr -d '\n')"
(( rc )) && exit $rc
exit $crc
