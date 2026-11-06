#!/usr/bin/env bash
# Group G data (the user's instruction of 2026-11-06, part 4; no training): the pointing pool's candidates
# (data_gen.transition_qa.pointing_qa raw, the round-11 SFT trajectories' frames) and D's action trajectories in the
# two targets formats (data_gen.sft_mixture.targets_convert), as a one-card sco job, CPU only, 13 processes on the
# job's 14 vCPUs (CLAUDE.md: data making goes to sco).  Outputs in G/pg and G/p2t, the log in RUN/logs.
set -uo pipefail
RUN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/g_prep_20261106/data_run
G=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/g_prep_20261106
SFT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft
D=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D
P=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
OWNER=$(stat -c %u:%g "$RUN")
mkdir -p "$RUN/logs" "$G/pg" "$G/p2t"
trap 'status=$?; printf "{\"exit_status\":%s,\"ended_utc\":\"%s\"}\n" "$status" "$(date -u +%FT%TZ)" > "$RUN/logs/exit.json"; chown -R "$OWNER" "$RUN" "$G" 2>/dev/null || true' EXIT
exec > >(tee -a "$RUN/logs/g_data-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix
export PYTHONPATH=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
quiet() { sed -u '/CollisionDetector\|VisibilityChecker\|Warning/d'; }
date -u; nproc
"$P" -m data_gen.transition_qa.pointing_qa raw --sft-dir "$SFT" --out "$G/pg" --workers 13 2>&1 | quiet || exit 1
for split in train val; do
  "$P" -m data_gen.sft_mixture.targets_convert --rows "$SFT/rows.jsonl" --frames "$SFT/frames.jsonl" \
    --in "$D/P2q_$split.jsonl" --out-dir "$G/p2t" --split "$split" --workers 13 2>&1 | quiet || exit 1
done
date -u
# the SFT conversation file the chain checks the start model's format against (rl-config --sft-trajectories)
"$P" -m data_gen.sft_mixture.targets_convert --rows "$SFT/rows.jsonl" --frames "$SFT/frames.jsonl" \
  --in /mnt/umm/users/yinbaiqiao/models_runs/sft_v10_newformat_20261024/sft_trajectories.jsonl --out-dir "$G/p2t" \
  --split all --out-name sft_trajectories.jsonl --workers 13 2>&1 | quiet || exit 1
date -u
