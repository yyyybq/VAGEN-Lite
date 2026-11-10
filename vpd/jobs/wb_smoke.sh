#!/usr/bin/env bash
# sco payload of job vpd-wb-smoke (as_root: Q3B blobs are root-only; one H800):
#     env VPD_SMOKE_DEADLINE_S=<outer-180> timeout -k 60 <outer> bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/wb_smoke.sh RUN
# outer = floor(est_high*3600) - 300 (pilot.py job_timeouts; plan est_high 1.0 -> 3300 s, soft deadline 3120 s), so
# the job never holds the card past the GPU-h charged for it (DECISIONS B2).
# RUN = the job's run dir (OUT/jobs/wb-smoke-<stamp>), also taken from $RUN.  Outputs go to
# SMOKE = OUT/walkerbench/runs/smoke/<basename RUN>/ (one dir per smoke run, so a rerun never collides).
# Steps (vpd_step -> RUN/steps.jsonl; a failing step does NOT stop the job, so the verdict is always written):
#   dry_200        PY_TRAIN n2_train_vpd.py --dry-run on tiers/seed1001/demos_200 (CPU): tokens per conversation for
#                  the cost model + the parity hashes of the first 3 demos (--expect-grid 24x40)
#   parity_hf      PY_Q3 parity_check.py on that dry run (HF processor parity, G0.3a / B16)
#   render_parity  PY_EVAL pilot render-parity: one pano/heading at 640x400 and 1280x800 from PY_EVAL and PY_Q3 (B17)
#   vllm_parity    PY_Q3 pilot vllm-parity: live vLLM usage.prompt_tokens vs the training tokenization (B5); run
#                  early (one server start) so the soft deadline cannot drop this required check
#   train_probe    (a) PY_TRAIN wb_train_probe.py --start Q3S --n 8 --load-only Q3RD (B10)
#   train_0W       (b) PY_TRAIN n2_train_vpd.py Q3S on warmup20 --epochs auto --seed 1001 -> SMOKE/adapters/Q3S_0W
#                  (the smoke's own copy: every pilot job retrains its 0W, DECISIONS B7)
#   merge_0W       (c) PY_Q3 merge.py into $VPD_MERGE_DIR (job-local /tmp if >= 40 GB free, else /dev/shm); its
#                  vpd_merge.json is copied to SMOKE/merge/
#   eval_a..e      (d) PY_Q3 -m vpd.walkerbench.run_vpd --task-set s2test --limit 6: a Q3B 640 empty; b Q3B 1280
#                  empty; c merged Q3S-0W 640 empty; d Q3B 640 off; e Q3B 640 empty --history-turns 2
#   rm_merged      (e) remove the merged folder
#   verdict        (f) PY_EVAL pilot smoke-verdict -> SMOKE/smoke_verdict.json (PASS / FAIL + measurements)
# A soft deadline (VPD_SMOKE_DEADLINE_S, default 3120 s after start, 180 s below the outer timeout) caps every GPU
# step with `timeout`, keeping 180 s for rm_merged + verdict.  Env: VPD_DRYRUN=1 prints the commands (RUN only).
# Test hook honoured ONLY with VPD_DRYRUN=1: VPD_WB_ROOT (instead of OUT/walkerbench); a real job always uses OUT.
# Threads: VPD_THREADS=2 for everything but vLLM (4, set by VLLMServer), 3 loader workers, 8 eval workers (B9).

RUN=${1:-${RUN:-}}
if [[ -z $RUN ]]; then
  echo "usage: wb_smoke.sh RUN" >&2
  exit 2
fi
export RUN
export VPD_THREADS=${VPD_THREADS:-2}
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

DRY=${VPD_DRYRUN:-0}
T0=$(date +%s)
DEADLINE=$((T0 + ${VPD_SMOKE_DEADLINE_S:-3120}))
RESERVE=180
WB=$VPD_OUT/walkerbench
if [[ $DRY == 1 && -n ${VPD_WB_ROOT:-} ]]; then
  WB=$VPD_WB_ROOT   # test hook (dry run only)
fi
SMOKE=$WB/runs/smoke/$(basename "$RUN")
TIER200=$WB/tiers/seed1001/demos_200.jsonl
WARM=$WB/warmup/warmup20.jsonl
FAILS=()

mkdir -p "$RUN"
cd "$VPD_WORKTREE"
if [[ $DRY != 1 ]]; then
  mkdir -p "$SMOKE"
  vpd_chown_trap "$RUN" "$SMOKE"
  # pilot prepare creates runs/ and runs/smoke as the login user; should they still be missing (created here, as
  # root), hand them back too.  Not recursive: runs/pilot may hold live pilot dirs.
  if [[ $(id -u) -eq 0 ]]; then
    vpd_on_exit "chown ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q ' "$WB/runs" "$WB/runs/smoke")|| true"
  fi
fi

x() {  # run, or print under VPD_DRYRUN=1
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*"
  else
    "$@"
  fi
}

left() { echo $((DEADLINE - $(date +%s) - RESERVE)); }

# step NAME CMD...: vpd_step under the soft deadline; a failure is recorded and the job goes on
step() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t < 60)); then
    echo "[wb_smoke] $name skipped: ${t}s left before the deadline" >&2
    vpd_step "$name:no_time" false || true
    FAILS+=("$name")
    return 1
  fi
  if vpd_step "$name" x timeout -k 30 "$t" "$@"; then
    return 0
  fi
  FAILS+=("$name")
  return 1
}

safe_rm_merged() {  # only job-local merged folders
  local d=$1
  case $d in
    /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) x rm -rf -- "$d" ;;
    *) echo "[wb_smoke] refusing to remove $d" >&2; return 1 ;;
  esac
}

pick_merge_dir() {  # /tmp if >= 40 GB free, else /dev/shm (DECISIONS B11); empty if neither
  local base avail
  for base in /tmp /dev/shm; do
    avail=$(df -P -B1G "$base" 2>/dev/null | awk 'NR==2{print $4}')
    if [[ -n $avail && $avail -ge 40 ]]; then
      echo "$base/vpd_merged"
      return 0
    fi
  done
  echo ""
}

echo "[wb_smoke] start $(date -u +%FT%TZ) RUN=$RUN SMOKE=$SMOKE dry=$DRY deadline=$((DEADLINE - T0))s"
vpd_node_facts "$RUN/node.txt"
for f in "$TIER200" "$WARM" "$WB/demos/s2train_640x400/DONE"; do
  if [[ ! -f $f ]]; then
    echo "[wb_smoke] missing $f: run job wb-demos first" >&2
    [[ $DRY == 1 ]] || exit 2
  fi
done
export VPD_MERGE_DIR=${VPD_MERGE_DIR:-$(pick_merge_dir)}
echo "[wb_smoke] VPD_MERGE_DIR=${VPD_MERGE_DIR:-<none with 40 GB free>}"
MERGED=${VPD_MERGE_DIR:+$VPD_MERGE_DIR/Q3S__smoke_0W}
if [[ -n ${MERGED:-} && $DRY != 1 ]]; then  # a timeout must not leave 16 GB on the node's /tmp or /dev/shm
  vpd_on_exit "case '$MERGED' in /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) rm -rf -- '$MERGED';; esac"
fi

# --- CPU checks first (cheap, independent of the GPU steps) ---
step dry_200 "$PY_TRAIN" vpd/train/n2_train_vpd.py --start Q3S --data "$TIER200" --dry-run --expect-grid 24x40 \
  --threads "$VPD_THREADS" --out "$SMOKE/dry_200" || true
step parity_hf "$PY_Q3" vpd/train/parity_check.py --dry-run-json "$SMOKE/dry_200/dry_run.json" \
  --threads "$VPD_THREADS" || true
step render_parity "$PY_EVAL" -m vpd.walkerbench.pilot render-parity --out "$SMOKE/render_parity.json" || true

# --- B5 live vLLM prompt-token parity on the dry run's first demos (one raw-Q3S server start; early, so the
#     soft deadline never drops this required check) ---
step vllm_parity "$PY_Q3" -m vpd.walkerbench.pilot vllm-parity --start Q3S \
  --dry-run-json "$SMOKE/dry_200/dry_run.json" --out "$SMOKE/vllm_parity.json" || true

# --- (a) train probe, (b) Q3S 0W ---
step train_probe "$PY_TRAIN" vpd/train/wb_train_probe.py --start Q3S --n 8 --load-only Q3RD --threads "$VPD_THREADS" \
  --out "$SMOKE/train_probe" || true
OK0W=1
step train_0W "$PY_TRAIN" vpd/train/n2_train_vpd.py --start Q3S --data "$WARM" --epochs auto --seed 1001 --phase 0W \
  --threads "$VPD_THREADS" --num-workers 3 --out "$SMOKE/adapters/Q3S_0W" || OK0W=0

# --- (d) evals a, b, d, e on raw Q3B ---
EVROOT=$SMOKE/evals
ev() {  # KEY MODEL RES MODE K [extra args...]
  local key=$1 model=$2 res=$3 mode=$4 k=$5; shift 5
  local size=640x400 tier=raw
  [[ $res == 1280 ]] && size=1280x800
  [[ $key == c ]] && tier=0W
  local name="smoke__${model}__${tier}__${res}__${mode}"
  ((k > 0)) && name="${name}__k$k"
  step "eval_$key" "$PY_Q3" -m vpd.walkerbench.run_vpd --model "$model" --task-set s2test --limit 6 \
    --render-size "$size" --constrained "$mode" --history-turns "$k" --workers 8 \
    --run-name "$name" --out-root "$EVROOT" "$@" || true
}
ev a Q3B 640 empty 0
ev b Q3B 1280 empty 0
ev d Q3B 640 off 0
ev e Q3B 640 empty 2

# --- (c) merge round trip + eval c on the merged Q3S 0W, then (e) remove it ---
if ((OK0W)) && [[ -n ${MERGED:-} ]]; then
  safe_rm_merged "$MERGED" || true
  if step merge_0W "$PY_Q3" vpd/train/merge.py --start Q3S --adapter "$SMOKE/adapters/Q3S_0W" --out "$MERGED"; then
    x mkdir -p "$SMOKE/merge"
    x cp "$MERGED/vpd_merge.json" "$SMOKE/merge/vpd_merge.json" || true
    ev c Q3S 640 empty 0 --model-dir "$MERGED" --adapter-meta "$SMOKE/adapters/Q3S_0W/meta.json"
  fi
  vpd_step rm_merged safe_rm_merged "$MERGED" || FAILS+=(rm_merged)
else
  echo "[wb_smoke] merge_0W / eval_c skipped (0W ok=$OK0W, merge dir '${MERGED:-}')" >&2
  vpd_step merge_0W:skipped false || true
  FAILS+=(merge_0W eval_c)
fi

# --- (f) verdict (always) ---
df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
vpd_step verdict x "$PY_EVAL" -m vpd.walkerbench.pilot smoke-verdict --smoke "$SMOKE" || FAILS+=(verdict)
if [[ $DRY == 1 ]]; then
  echo "[dry] date -u > $RUN/DONE"
else
  date -u +%FT%TZ > "$RUN/DONE"
fi
echo "[wb_smoke] done $(date -u +%FT%TZ); failed steps: ${FAILS[*]:-none}"
((${#FAILS[@]} == 0))
