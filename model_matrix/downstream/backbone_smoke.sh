#!/usr/bin/env bash
# The StarVLA backbone acceptance of the model line, part 2 (the user's 4.4 of 2026-10-24): the downstream line's
# minimal scheme (its step 2: three tasks x 100 demonstrations) for one start -- StarVLA SFT on 8 cards for the
# experiment's steps, then a few closed-loop episodes with one policy server.  Everything of the downstream line is
# used read only, by path (its train entry, closed-loop evaluator, data); writes only under EXP.  An 8-card sco job.
#
#   bash backbone_smoke.sh EXP        (EXP made by python -m model_matrix.downstream.make_config)
set -uo pipefail
EXP=${1:?experiment folder}
WT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models
DS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream
MAIN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
STARVLA_ROOT=$MAIN/third_party/starVLA
ROBOCASA365_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365
SIM_PY=$ROBOCASA365_ROOT/policies/diffusion_policy/.venv/bin/python
TASKS="NavigateKitchen OpenStandMixerHead CloseToasterOvenDoor"
SEEDS=${SMOKE_SEEDS:-1000-1003}
WORKERS=${SMOKE_WORKERS:-12}
CKPT=$EXP/run/final_model/pytorch_model.pt
OWNER=$(stat -c %u:%g "${EXP}")
SERVER_PID=""; MON_PID=""
cleanup() {
  [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null
  [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
  rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"
  chown -R "${OWNER}" "${EXP}" 2>/dev/null || true
}
trap cleanup EXIT
exec > >(tee -a "${EXP}/smoke.log") 2>&1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
source "${WT}/examples/train/robocasa/setup_starvla_env.sh"
set +e
export PYTHONPATH="${DS}:${STARVLA_ROOT}"
TIMES="${EXP}/stage_times.tsv"
[[ -f "$TIMES" ]] || echo -e "stage\tstart_utc\tseconds\trc" > "$TIMES"
stage() {
  local name=$1; shift
  local t0=$(date +%s) s0=$(date -u +%FT%TZ)
  echo "[smoke] >>> $name $s0"
  "$@"
  local rc=$?
  echo -e "$name\t$s0\t$(( $(date +%s) - t0 ))\t$rc" >> "$TIMES"
  echo "[smoke] <<< $name rc=$rc $(( $(date +%s) - t0 ))s"
  return $rc
}
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 5000 > "${EXP}/gpu.csv" &
MON_PID=$!

# ---- StarVLA SFT, 8 GPUs (the downstream line's train entry and DeepSpeed ZeRO-2 config) --------------------------
cd "$STARVLA_ROOT"
stage train accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
  "$DS/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" > "$EXP/train.log" 2>&1
cd "$WT"
[[ -f "$CKPT" ]] || { echo "[smoke] no final checkpoint"; tail -60 "$EXP/train.log"; exit 4; }

# ---- a few closed-loop episodes, one policy server -----------------------------------------------------------------
SIM_PYTHONPATH="$MAIN/playground/Runtime/robocasa_client:$DS:$STARVLA_ROOT:$ROBOCASA365_ROOT/robocasa:$ROBOCASA365_ROOT/robomimic"
EGL_LIB_DIR="$MAIN/playground/Runtime/egl/usr/lib/x86_64-linux-gnu"
sim() {
  local threads=$1; shift
  env PYTHONPATH="$SIM_PYTHONPATH" LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 LP_NUM_THREADS="$threads" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMBA_CACHE_DIR="/tmp/ds-numba-$UID" MESA_SHADER_CACHE_DIR="/tmp/ds-mesa-$UID" "$@"
}
mkdir -p "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"
CUDA_VISIBLE_DEVICES=0 CKPT="$CKPT" PORT=5678 bash "$DS/examples/evaluate/robocasa/run_starvla_mobile.sh" server \
  --idle_timeout -1 > "$EXP/server.log" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 180); do
  python -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",5678)))' && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[smoke] policy server died"; tail -40 "$EXP/server.log"; exit 5; }
  sleep 5
done
echo -e "server_ready\t-\t$((i * 5))\t0" >> "$TIMES"
CPUS=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 112)
THREADS=$(( (CPUS - 8) / WORKERS )); (( THREADS < 1 )) && THREADS=1
stage eval sim "$THREADS" timeout 10800 "$SIM_PY" "$DS/downstream/eval/closed_loop.py" --ckpt "$CKPT" \
  --servers 127.0.0.1:5678 --tasks $TASKS --seeds "$SEEDS" --workers "$WORKERS" --out "$EXP/smoke_eval" \
  > "$EXP/eval.log" 2>&1
echo "[smoke] $(cat "$EXP/smoke_eval/summary.json" 2>/dev/null | tr -d '\n' | cut -c1-400)"
echo "[smoke] done $(date -u +%FT%TZ)"
