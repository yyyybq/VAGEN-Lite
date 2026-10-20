#!/usr/bin/env bash
# Step 2 of the downstream line, one 8-card job (docs/downstream/reports/STEP2_SMOKE_*.md):
#   0 checks: start weights unchanged, prepared data = manifest      4 seed check: same seed -> same initial state
#   1 seed evidence: CPU dry runs (RS s1, B s1, RS s2)                5 closed loop: 1 policy server (GPU 0), the same
#   2 SFT: StarVLA via downstream/train/train_entry.py, 8 GPUs          24 episodes with 3, 12 and 24 parallel
#   3 held-out action error (scripts/eval_starvla_heldout.py, 1 GPU)     simulator processes, GPU utilisation sampled
# Usage: run_step2.sh EXP_DIR DATA_DIR.  Writes only under EXP_DIR.
set -uo pipefail
EXP="${1:?Usage: run_step2.sh EXP_DIR DATA_DIR}"
DATA="${2:?Usage: run_step2.sh EXP_DIR DATA_DIR}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
STARVLA_ROOT="$MAIN/third_party/starVLA"
ROBOCASA365_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365
SIM_PY="$ROBOCASA365_ROOT/policies/diffusion_policy/.venv/bin/python"
TASKS="NavigateKitchen OpenStandMixerHead CloseToasterOvenDoor"
SEEDS="1000-1007"
OUT="$EXP/step2"
CKPT="$EXP/run/final_model/pytorch_model.pt"
mkdir -p "$OUT/seed"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
TIMES="$OUT/stage_times.tsv"
echo -e "stage\tstart_utc\tseconds\trc" > "$TIMES"
stage() {  # stage NAME COMMAND...: run, time, record
  local name=$1; shift
  local t0=$(date +%s) s0=$(date -u +%FT%TZ)
  echo "[step2] >>> $name $s0"
  "$@"
  local rc=$?
  echo -e "$name\t$s0\t$(( $(date +%s) - t0 ))\t$rc" >> "$TIMES"
  echo "[step2] <<< $name rc=$rc $(( $(date +%s) - t0 ))s"
  return $rc
}
SERVER_PID=""; MON_PID=""
cleanup() {
  [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null
  [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
  rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"
}
trap cleanup EXIT

source "$WT/examples/train/robocasa/setup_starvla_env.sh"
set +e
export PYTHONPATH="$WT:$STARVLA_ROOT"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
echo "[step2] cpu quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"

# ---- 0 checks -----------------------------------------------------------------------------------------------
stage verify_starts python "$WT/downstream/verify_starts.py" verify --starts B RS > "$OUT/verify_starts.json" \
  || { echo "[step2] a start changed since it was locked: stop"; exit 3; }
stage data_check python "$WT/downstream/demos/prepare_data.py" --out "$DATA" --check > "$OUT/data_check.json" \
  || { echo "[step2] prepared data differs from its manifest: stop"; exit 3; }

# ---- 1 seed evidence (CPU) ----------------------------------------------------------------------------------
B_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["B"]["path"])' "$WT/downstream/starts.json")
stage dry_RS_s1 env CUDA_VISIBLE_DEVICES= python "$WT/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" \
  --dry-run "$OUT/seed/RS_s1.json" --world 8 --batches 20
stage dry_B_s1 env CUDA_VISIBLE_DEVICES= python "$WT/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" \
  "framework.qwenvl.base_vlm=$B_PATH" --dry-run "$OUT/seed/B_s1.json" --world 8 --batches 20
stage dry_RS_s2 env CUDA_VISIBLE_DEVICES= python "$WT/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" \
  seed=2 --dry-run "$OUT/seed/RS_s2.json" --world 8 --batches 20

# ---- 2 SFT (8 GPUs) -------------------------------------------------------------------------------------------
cd "$STARVLA_ROOT"
stage train accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
  "$WT/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" > "$OUT/train.log" 2>&1
cd "$WT"
[[ -f "$CKPT" ]] || { echo "[step2] no final checkpoint: stop"; tail -50 "$OUT/train.log"; exit 4; }
ls -la "$CKPT"

# ---- 3 held-out action error (1 GPU) -------------------------------------------------------------------------
stage heldout env CUDA_VISIBLE_DEVICES=0 python "$WT/scripts/eval_starvla_heldout.py" --ckpt "$CKPT" \
  --out "$OUT/heldout" --episodes-per-task 20 --frames-per-episode 3 > "$OUT/heldout.log" 2>&1

# ---- simulator environment -------------------------------------------------------------------------------------
SIM_PYTHONPATH="$MAIN/playground/Runtime/robocasa_client:$WT:$STARVLA_ROOT:$ROBOCASA365_ROOT/robocasa:$ROBOCASA365_ROOT/robomimic"
EGL_LIB_DIR="$MAIN/playground/Runtime/egl/usr/lib/x86_64-linux-gnu"
sim() {  # sim THREADS COMMAND...: run in the simulator Python environment with OSMesa
  local threads=$1; shift
  env PYTHONPATH="$SIM_PYTHONPATH" LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 LP_NUM_THREADS="$threads" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMBA_CACHE_DIR="/tmp/ds-numba-$UID" MESA_SHADER_CACHE_DIR="/tmp/ds-mesa-$UID" "$@"
}
mkdir -p "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"

# ---- 4 seed check ---------------------------------------------------------------------------------------------
stage seed_check sim 4 timeout 1800 "$SIM_PY" "$WT/downstream/eval/seed_check.py" --tasks $TASKS --seed 1000 \
  --out "$OUT/seed_check.json" > "$OUT/seed_check.log" 2>&1

# ---- 5 closed loop -------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 CKPT="$CKPT" PORT=5678 bash "$WT/examples/evaluate/robocasa/run_starvla_mobile.sh" server \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do
  python -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",5678)))' && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[step2] policy server died"; tail -40 "$OUT/server.log"; exit 5; }
  sleep 5
done
echo "[step2] policy server ready after ~$((i * 5))s"
echo -e "server_ready\t-\t$((i * 5))\t0" >> "$TIMES"
CPUS=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 112)
for P in 3 12 24; do
  THREADS=$(( (CPUS - 8) / P )); (( THREADS < 1 )) && THREADS=1
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -i 0 -lms 2000 \
    > "$OUT/gpu_util_P$P.csv" 2>/dev/null &
  MON_PID=$!
  stage "eval_P$P" sim "$THREADS" timeout 10800 "$SIM_PY" "$WT/downstream/eval/closed_loop.py" --ckpt "$CKPT" \
    --servers 127.0.0.1:5678 --tasks $TASKS --seeds "$SEEDS" --workers "$P" --out "$OUT/eval_P$P" \
    > "$OUT/eval_P$P.log" 2>&1
  kill "$MON_PID" 2>/dev/null; MON_PID=""
  echo "[step2] P=$P threads/process=$THREADS: $(cat "$OUT/eval_P$P/summary.json" 2>/dev/null | tr -d '\n' | cut -c1-300)"
done
echo "[step2] done $(date -u +%T)"
