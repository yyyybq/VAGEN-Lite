#!/usr/bin/env bash
# X4 (the user's decision 6 of 2026-10-26: four HF copies allowed, about 66 GB), a one-card job: the VLM of each E1
# final checkpoint (DS-<start>-100-s1/run/final_model, read by path) exported with the main checkout's
# scripts/export_starvla_backbone.py into HF/E1-<start>-200k (the action head is not exported).  An export is done
# when its starvla_export.json exists (the script writes it last); a half-written folder is removed and redone.
# Usage: run_x4_export.sh HF
set -uo pipefail
HF="${1:?Usage: run_x4_export.sh HF_DIR}"
MAIN="${X4_MAIN:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"  # X4_* only change for the tests
RUNS="${X4_RUNS:-/mnt/umm/users/yinbaiqiao/downstream_runs/experiments}"
PY="${X4_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
mkdir -p "$HF"
log() { echo "[x4-export] $* $(date -u +%FT%TZ)" | tee -a "$HF/export.log"; }
rc_all=0
for s in B S RS RD; do
  out="$HF/E1-$s-200k"
  if [[ -f "$out/starvla_export.json" ]]; then log "$s: done before"; continue; fi
  [[ -e "$out" ]] && { rm -rf "$out"; log "$s: removed a half-written export"; }
  t0=$(date +%s)
  (cd "$MAIN" && PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/export_starvla_backbone.py \
     --ckpt "$RUNS/DS-$s-100-s1/run/final_model/pytorch_model.pt" --out "$out") >> "$HF/export.log" 2>&1
  rc=$?
  log "$s: rc=$rc $(( $(date +%s) - t0 )) s $(du -sh "$out" 2>/dev/null | cut -f1)"
  (( rc )) && rc_all=$rc
done
exit $rc_all
