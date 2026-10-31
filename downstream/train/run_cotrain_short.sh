#!/usr/bin/env bash
# C1's short run (the user's decision of 2026-10-30; an 8-card sco job): train EXP (made with make_experiment.py
# --cotrain lambda=0.1,diag_every=10,skip_final_save=1 and a few hundred steps) to its end in one launch through
# downstream/train/cotrain_entry.py, with the same environment as run_segmented.sh, logging GPU memory and use every
# 10 s (EXP/gpu.csv); no checkpoint, no evaluation.  Then cotrain_check.py writes EXP/cotrain_check.json (the five
# items of the design).  Usage: run_cotrain_short.sh EXP
set -uo pipefail
EXP="${1:?Usage: run_cotrain_short.sh EXP}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
log() { echo "[c1-short] $* $(date -u +%FT%TZ)" | tee -a "$EXP/segments.log"; }
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 10000 >> "$EXP/gpu.csv" &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null' EXIT
log "train $EXP"
(
  export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
  export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
  # 2026-10-31: the first short run ran out of memory with 4.4 GiB reserved but unallocated (fragmentation)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
  set +e
  export PYTHONPATH="$WT:$STARVLA_ROOT"
  cd "$STARVLA_ROOT"
  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
    "$WT/downstream/train/cotrain_entry.py" --config_yaml "$EXP/config.yaml" >> "$EXP/train.log" 2>&1
)
rc=$?
log "train rc=$rc"
python3 "$WT/downstream/train/cotrain_check.py" --exp "$EXP" > "$EXP/cotrain_check.log" 2>&1
crc=$?
log "check rc=$crc $(head -c 600 "$EXP/cotrain_check.json" 2>/dev/null | tr -d '\n')"
(( rc )) && exit $rc
exit $crc
