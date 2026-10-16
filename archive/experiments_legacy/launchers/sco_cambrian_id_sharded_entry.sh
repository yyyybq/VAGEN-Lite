#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
TASK=${SCO_CAMBRIAN_ID_TASK:?set SCO_CAMBRIAN_ID_TASK=c4|c8_50|c8_100|c8_1000}
case "$TASK" in
 c4) EXP=cambrian_c4_clean_d0pass_20260822_h800_r5_full; STEP=960; SWEEP=cambrian_c4_clean_id_zoetrope_sharded_20260905;;
 c8_50) EXP=cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full; STEP=50; SWEEP=cambrian_c8_clean_id_zoetrope_sharded_20260905_s50;;
 c8_100) EXP=cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full; STEP=100; SWEEP=cambrian_c8_clean_id_zoetrope_sharded_20260905_s100;;
 c8_1000) EXP=cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full; STEP=1000; SWEEP=cambrian_c8_clean_id_zoetrope_sharded_20260905_s1000;;
 *) echo "unknown task $TASK" >&2; exit 2;; esac
 cd "$ROOT"; export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}"
 export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
 export VAGEN_CAMBRIAN_MAX_MODEL_LEN=32768 VAGEN_GSPLAT_PREBUILT=1 TORCH_EXTENSIONS_DIR=/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer
 export VAGEN_ENV_ROOT=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite CC=$VAGEN_ENV_ROOT/bin/x86_64-conda-linux-gnu-gcc CXX=$VAGEN_ENV_ROOT/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=$VAGEN_ENV_ROOT/bin/x86_64-conda-linux-gnu-g++
 export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
 WORK=/tmp/cambrian_id_sharded_${TASK}_$(hostname); rm -rf "$WORK"; mkdir -p "$WORK"
 "$PY" scripts/prepare_cambrian_id_shards.py --base-config evaluation/sweeps/active_spatial/recovery_logs/test_suites_local_cache.yaml --input-jsonl data_gen/active_spatial_pipeline/output_v2/test_id_v1.jsonl --out-dir "$WORK" --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS --shards 8
 "$PY" scripts/active_spatial_eval_parallel.py --suite-config "$WORK/suite_config.yaml" --exp-root exps/vagen_active_spatial --out-root evaluation/sweeps/active_spatial --sweep-name "$SWEEP" --exps "$EXP" --steps "$STEP" --suites id_shard0,id_shard1,id_shard2,id_shard3,id_shard4,id_shard5,id_shard6,id_shard7 --agents model --gpus 0,1,2,3,4,5,6,7 --val-n 4 --max-attempts 1 --gpu-memory-utilization 0.7
 "$PY" scripts/aggregate_active_spatial_shards.py --sweep-root evaluation/sweeps/active_spatial/"$SWEEP" --experiment "$EXP" --step "$STEP"
