#!/usr/bin/env bash
# Golden outputs for the cleanup equivalence check (CPU only; the login node is fine, about 15 minutes).
#   ROOT=<checkout to test> GOUT=<new output dir> bash scripts/cluster/golden_outputs.sh
# G1 generates 2 scenes with the round-10 generator config; G2 replays 5 tasks per family in the env;
# G3 builds the scene split and the SFT frame lists; G4 rebuilds the round-8 evaluation table on a copy.
# Compare two runs with scripts/cluster/compare_golden.py (wall-time fields are ignored).
set -uo pipefail
ROOT=${ROOT:?set ROOT to the checkout to test}
G=$(realpath -m "${GOUT:?set GOUT to a new output directory}")
DATA=${DATA:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}          # checkout that holds exps/ (frozen R1 tasks, round-8 eval)
SCENES=${SCENES:-/mnt/umm/users/yinbaiqiao/InteriorGS_meta}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}
cd "$ROOT"
export PYTHONPATH=$PWD:$PWD/verl CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$G"
git rev-parse HEAD > "$G/commit.txt"
CFG=data_gen/interiorgs_v10/configs/generation_r10_20261015.json
echo "== G1 generate 2 scenes $(date -u +%T)"
$PY -m data_gen.interiorgs_v10.run_all --scenes-root "$SCENES" --out "$G/gen" --workers 2 --limit 2 --expected-scenes 2 --config "$CFG" > "$G/g1_run_all.log" 2>&1; echo "G1 rc=$?"
$PY -m data_gen.interiorgs_v10.run_all --out "$G/gen" --aggregate >> "$G/g1_run_all.log" 2>&1; echo "G1 aggregate rc=$?"
echo "== G2 replay $(date -u +%T)"
$PY -m data_gen.interiorgs_v10.replay_env --run "$G/gen" --scenes-root "$SCENES" --per-family 5 \
  --env-template data_gen/interiorgs_v10/configs/rl_env_v10_S0.yaml \
  --no-task-objects data_gen/interiorgs_v10/configs/no_task_objects_20261014.json \
  --json "$G/g2_replay.json" > "$G/g2_replay.log" 2>&1; echo "G2 rc=$?"
echo "== G3 split + sft frames $(date -u +%T)"
$PY -m data_gen.interiorgs_v10.make_split --run "$G/gen" \
  --groups data_gen/interiorgs_v10/cluster_runs/same_source_groups_20261012.json \
  --r1-frozen "$DATA/exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1" \
  --out "$G/gen/split_manifest.json" > "$G/g3_split.log" 2>&1; echo "G3 split rc=$?"
$PY -m data_gen.interiorgs_v10.training_prep sft-frames --run "$G/gen" --split-manifest "$G/gen/split_manifest.json" --out "$G/sft" > "$G/g3_sft_frames.log" 2>&1; echo "G3 sft-frames rc=$?"
echo "== G4 evaluation table on a copy $(date -u +%T)"
rm -rf "$G/eval2_copy"; cp -r "$DATA/exps/vagen_active_spatial/interiorgs_v10_r8_20261014/eval2" "$G/eval2_copy" && rm -f "$G/eval2_copy/table.md" "$G/eval2_copy/table.json"
$PY -m data_gen.interiorgs_v10.evaluation table --eval "$G/eval2_copy" > "$G/g4_table.log" 2>&1; echo "G4 rc=$?"
echo "== hashes $(date -u +%T)"
(cd "$G" && find gen sft eval2_copy/table.md eval2_copy/table.json g2_replay.json -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" -o -name "*.md" \) 2>/dev/null | sort | xargs sha256sum > SHA256SUMS.txt)
wc -l "$G/SHA256SUMS.txt"
echo "done $(date -u +%T)"
