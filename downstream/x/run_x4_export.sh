#!/usr/bin/env bash
# X4 (the user's decision 6 of 2026-10-26: four HF copies allowed, about 66 GB), a one-card job: the VLM of each E1
# final checkpoint (DS-<start>-100-s1/run/final_model, read by path) exported with the main checkout's
# scripts/export_starvla_backbone.py into HF/E1-<start>-200k (the action head is not exported).  An export is done
# when its starvla_export.json exists (the script writes it last); a half-written folder is removed and redone.
# C1 (the user's decisions 5 of 2026-11-02 and 5 of 2026-11-03: one copy of C1's VLM): with NAME and CKPT, that one
# checkpoint into HF/NAME instead.
# Usage: run_x4_export.sh HF [NAME CKPT]
set -uo pipefail
HF="${1:?Usage: run_x4_export.sh HF_DIR [NAME CKPT]}"
MAIN="${X4_MAIN:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"  # X4_* only change for the tests
RUNS="${X4_RUNS:-/mnt/umm/users/yinbaiqiao/downstream_runs/experiments}"
PY="${X4_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
mkdir -p "$HF"
log() { echo "[x4-export] $* $(date -u +%FT%TZ)" | tee -a "$HF/export.log"; }
JOBS=()  # label|folder|checkpoint
if [[ -n ${2:-} ]]; then
  JOBS=("$2|$2|${3:?Usage: run_x4_export.sh HF_DIR [NAME CKPT]}")
else
  for s in B S RS RD; do JOBS+=("$s|E1-$s-200k|$RUNS/DS-$s-100-s1/run/final_model/pytorch_model.pt"); done
fi
rc_all=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r s name ckpt <<< "$job"
  out="$HF/$name"
  if [[ -f "$out/starvla_export.json" ]]; then log "$s: done before"; continue; fi
  [[ -e "$out" ]] && { rm -rf "$out"; log "$s: removed a half-written export"; }
  t0=$(date +%s)
  (cd "$MAIN" && PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/export_starvla_backbone.py --ckpt "$ckpt" --out "$out") \
    >> "$HF/export.log" 2>&1
  rc=$?
  log "$s: rc=$rc $(( $(date +%s) - t0 )) s $(du -sh "$out" 2>/dev/null | cut -f1)"
  (( rc )) && rc_all=$rc
done
exit $rc_all
