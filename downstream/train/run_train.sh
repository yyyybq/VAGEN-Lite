#!/usr/bin/env bash
# One StarVLA training run of the downstream line (an 8-card sco job).  EXP is a directory made by make_experiment.py.
#   checks: the start weights unchanged since locked, the prepared data = its manifest, the experiment record exists
#   train:  accelerate (DeepSpeed ZeRO-2, 8 processes) -> downstream/train/train_entry.py --config_yaml EXP/config.yaml
# Writes only under EXP (EXP/run: StarVLA output, EXP/train.log, EXP/stage_times.tsv).  Usage: run_train.sh EXP
set -uo pipefail
EXP="${1:?Usage: run_train.sh EXP_DIR}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
source "$WT/examples/train/robocasa/setup_starvla_env.sh"
set +e
export PYTHONPATH="$WT:$STARVLA_ROOT"
read -r START DATA < <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["start"]["name"], d["data"]["dir"])' "$EXP/experiment.json")
echo "[train] exp=$EXP start=$START data=$DATA $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
python "$WT/downstream/verify_starts.py" verify --starts "$START" > "$EXP/verify_starts_at_start.json" \
  || { echo "[train] start $START changed since it was locked: stop"; cat "$EXP/verify_starts_at_start.json"; exit 3; }
python "$WT/downstream/demos/prepare_data.py" --out "$DATA" --check > "$EXP/data_check_at_start.json" \
  || { echo "[train] prepared data differs from its manifest: stop"; exit 3; }
[[ -e "$EXP/run/final_model" ]] && { echo "[train] $EXP/run already finished: stop"; exit 4; }
cd "$STARVLA_ROOT"
t0=$(date +%s)
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
  "$WT/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" >> "$EXP/train.log" 2>&1
rc=$?
echo -e "train\t$(date -u -d @$t0 +%FT%TZ)\t$(( $(date +%s) - t0 ))\t$rc" >> "$EXP/stage_times.tsv"
echo "[train] rc=$rc $(( $(date +%s) - t0 ))s; checkpoints: $(ls "$EXP/run/checkpoints" 2>/dev/null | tr '\n' ' ')"
exit $rc
