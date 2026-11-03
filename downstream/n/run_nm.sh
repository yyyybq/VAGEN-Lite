#!/usr/bin/env bash
# Matrix v3 starts the N1 way (the user, 2026-11-03 evening), one one-card sco job per start: run_n_episodes.sh (N1's
# runner, frozen weights) with the start's own validation config (nm_starts.py val-env) and WORKERS episodes at once;
# then every frame is scanned for colour noise (OUT/frames_scan.json), the corrupted episodes are moved aside
# (n_repair.py drop; nothing deleted) and played again with REPAIR_WORKERS at once, and the frames scanned again
# (OUT/frames_scan_repaired.json), as N3 did.  Resumable.  Writes only under OUT.
# Usage: run_nm.sh OUT INSTANCES NAME [WORKERS] [REPAIR_WORKERS]   (NAME: an entry of downstream/n/nm_starts.json)
#   Runs as the container's root like N1's Qwen3 models; run_n_episodes.sh and this script give OUT back to
#   OWNER_UID:OWNER_GID (default 20325).  The simulator runs without the Mesa shader cache (as N2 and N3).
#   NM_STARTS: another starts file than nm_starts.json (tests; the model server reads only nm_starts.json);
#   NM_RUNNER: another episode runner than run_n_episodes.sh (tests).
set -uo pipefail
OUT="${1:?Usage: run_nm.sh OUT INSTANCES NAME [WORKERS] [REPAIR_WORKERS]}"; INST="${2:?}"; NAME="${3:?}"
WORKERS="${4:-12}"; REPAIR_WORKERS="${5:-4}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PY="${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
RUNNER="${NM_RUNNER:-$WT/downstream/n/run_n_episodes.sh}"
cd "$WT"
mkdir -p "$OUT"
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
trap 'if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$OUT"; fi' EXIT
log() { echo "[nm] $* $(date -u +%FT%TZ)" | tee -a "$OUT/nm.log"; }
STARTS=()
[[ -n ${NM_STARTS:-} ]] && STARTS=(--starts "$NM_STARTS")
N_VAL_ENV="$("$VLM_PY" downstream/n/nm_starts.py val-env "$NAME" ${STARTS[@]+"${STARTS[@]}"})" \
  || { log "$NAME is not in the starts"; exit 2; }
[[ -f $N_VAL_ENV ]] || { log "$NAME: no validation config $N_VAL_ENV"; exit 2; }
export N_VAL_ENV MESA_SHADER_CACHE_DISABLE=true MESA_GLSL_CACHE_DISABLE=true
scan() {
  mkdir -p "$OUT/.scan/$NAME" && ln -sfn "$OUT/frames" "$OUT/.scan/$NAME/frames"
  "$VLM_PY" downstream/n/n_frames.py scan --eval "$OUT/.scan" --out "$1" >> "$OUT/nm.log" 2>&1
  rm -rf "$OUT/.scan"
}
count() { wc -l < "$OUT/episodes.jsonl" 2>/dev/null || echo 0; }
log "$NAME: prompt $N_VAL_ENV; $WORKERS at once"
bash "$RUNNER" "$OUT" "$INST" "$NAME" "$WORKERS" 0
rc=$?
log "first pass rc=$rc: $(count) episodes"
scan "$OUT/frames_scan.json"
res="$("$VLM_PY" downstream/n/n_repair.py drop --out "$OUT")"
log "drop: $res"
if [[ $res != *'"dropped": 0'* ]]; then
  bash "$RUNNER" "$OUT" "$INST" "$NAME" "$REPAIR_WORKERS" 0
  rc=$?
  log "replay rc=$rc: $(count) episodes"
  scan "$OUT/frames_scan_repaired.json"
fi
want=$(grep -c . "$INST")
log "done: $(count) of $want episodes, $(grep -c '"success": true' "$OUT/episodes.jsonl" 2>/dev/null) successes"
[[ $rc -eq 0 && $(count) -eq $want ]] || exit 1
