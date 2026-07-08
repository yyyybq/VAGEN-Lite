#!/bin/bash
# launch_parallel_probe.sh — 5-GPU parallel EASI probe
#
# Distributes checkpoint evaluations across 5 GPUs (0-4).
# Each GPU runs one easi_probe.py instance sequentially over its assigned
# checkpoints and all benchmarks in the given group.
#
# Usage:
#   bash scripts/launch_parallel_probe.sh                  # default: core5 benches
#   bash scripts/launch_parallel_probe.sh extended          # 7 benches
#   bash scripts/launch_parallel_probe.sh core5 --rerun    # force re-eval
#
# Logs:
#   easi_probe_results/worker_gpu{N}.log   (per-GPU orchestrator log)
#   easi_probe_results/{ckpt}/{bench}/lmms_eval.log  (per eval detail log)
#
# Monitor:
#   bash scripts/launch_parallel_probe.sh --status
#   python scripts/probe_table.py

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/scratch/by2593/miniconda3/envs/vagen-lite/bin/python"
PROBE="$SCRIPT_DIR/easi_probe.py"
OUT_DIR="$PROJ_DIR/easi_probe_results"
mkdir -p "$OUT_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
BENCH_GROUP="${1:-core5}"
EXTRA_ARGS="${2:-}"

# ── Status mode ──────────────────────────────────────────────────────────────
if [[ "$1" == "--status" ]]; then
    echo "=== Running processes ==="
    ps aux | grep "easi_probe\|lmms_eval" | grep -v grep | \
        awk '{print $1, $11, $12, $13, $14, $15, $16}'

    echo ""
    echo "=== Completed evaluations ==="
    find "$OUT_DIR" -name "*_results.json" 2>/dev/null | sort | while read f; do
        ckpt=$(echo "$f" | sed "s|$OUT_DIR/||" | cut -d/ -f1)
        bench=$(echo "$f" | sed "s|$OUT_DIR/||" | cut -d/ -f2)
        printf "  %-25s  %s\n" "$ckpt" "$bench"
    done

    echo ""
    echo "=== In-progress (lmms_eval.log last line) ==="
    for log in "$OUT_DIR"/*/*/lmms_eval.log; do
        [[ -f "$log" ]] || continue
        ckpt=$(echo "$log" | sed "s|$OUT_DIR/||" | cut -d/ -f1)
        bench=$(echo "$log" | sed "s|$OUT_DIR/||" | cut -d/ -f2)
        last=$(tail -1 "$log" 2>/dev/null)
        [[ -z "$last" ]] && continue
        printf "  %-20s %-20s %s\n" "$ckpt" "$bench" "$last"
    done
    exit 0
fi

# ── GPU → Checkpoint assignment ───────────────────────────────────────────────
# 10 checkpoints spread across 5 GPUs (2 per GPU)
# GPU 0 gets the fastest/smallest evaluations first (base + v33_step300)
declare -A GPU_CKPTS=(
    [0]="base,v33_step300"
    [1]="v26_step160,v34_step150"
    [2]="v31_step150,v35_step250"
    [3]="v33_step150,v36_step300"
    [4]="v28_step150"
)

echo "======================================================================"
echo "  EASI Parallel Probe — 5 GPUs × benchmarks=${BENCH_GROUP}"
echo "  Output: $OUT_DIR"
echo "======================================================================"
for gpu in 0 1 2 3 4; do
    echo "  GPU $gpu: ${GPU_CKPTS[$gpu]}"
done
echo "======================================================================"
echo ""

# ── Kill any leftover processes ───────────────────────────────────────────────
pkill -f "easi_probe.py.*--gpu [01234]" 2>/dev/null || true
sleep 1

# ── Launch one worker per GPU ────────────────────────────────────────────────
PIDS=()
for gpu in 0 1 2 3 4; do
    ckpts="${GPU_CKPTS[$gpu]}"
    log="$OUT_DIR/worker_gpu${gpu}.log"
    echo "  Launching GPU $gpu → checkpoints: $ckpts"
    nohup $PYTHON -u "$PROBE" \
        --ckpts "$ckpts" \
        --benchmarks "$BENCH_GROUP" \
        --gpu "$gpu" \
        --output_dir "$OUT_DIR" \
        $EXTRA_ARGS \
        > "$log" 2>&1 &
    PIDS+=($!)
    sleep 0.5  # stagger starts to avoid HF cache collisions
done

echo ""
echo "  Started PIDs: ${PIDS[*]}"
echo ""
echo "  Monitor with:"
echo "    bash scripts/launch_parallel_probe.sh --status"
echo "    python scripts/probe_table.py"
echo ""
echo "  Worker logs:"
for gpu in 0 1 2 3 4; do
    echo "    GPU $gpu: $OUT_DIR/worker_gpu${gpu}.log"
done
echo ""
echo "  To show live progress of a GPU:"
echo "    tail -f $OUT_DIR/worker_gpu0.log"
