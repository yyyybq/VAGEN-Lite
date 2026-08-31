#!/usr/bin/env bash
# Run the five current_main checkpoints in parallel on GPUs 0-4.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
OUT_DIR="${EASI_OUTPUT_DIR:-$ROOT/exps/unified_eval_runs/easi_results}"

if [[ "${1:-}" == "--status" ]]; then
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  ps -ef | awk '/easi_eval.py|run_easi_eval.py|lmms_eval/ && !/awk/ {print}'
  exit 0
fi

BENCHMARKS="${1:-easi_8}"
[[ $# -gt 0 ]] && shift
mkdir -p "$OUT_DIR"
CKPTS=(qwen_pretrain_baseline v46_step200 v50_step150 cambrian_pretrain_baseline c8_step300)
PIDS=()
for gpu in 0 1 2 3 4; do
  ckpt="${CKPTS[$gpu]}"
  log="$OUT_DIR/worker_gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/easi_ds_gpu${gpu}}" \
  CAMBRIAN_SRC="${CAMBRIAN_SRC:-/mnt/umm/users/yinbaiqiao/cambrian-s}" \
  TOKENIZERS_PARALLELISM=false \
  nohup "$PYTHON" -u "$SCRIPT_DIR/easi_eval.py" \
    --ckpts "$ckpt" --benchmarks "$BENCHMARKS" --gpu "$gpu" \
    --output_dir "$OUT_DIR" "$@" >"$log" 2>&1 &
  PIDS+=("$!")
  echo "GPU $gpu: $ckpt (PID $!)"
done
printf 'Logs: %s/worker_gpu{0..4}.log\n' "$OUT_DIR"
printf 'PIDs: %s\n' "${PIDS[*]}"
