#!/usr/bin/env bash
# sco payload of job vpd-e3vs-prep (non-root; one card for gsplat; PY_Q3 = eval-external-q3, which has gsplat 1.5.3,
# plyfile and ply_gaussian_loader through its system site packages):
#     timeout -k 120 6900 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/e3vs_prep.sh RUN
# Prepared by the orchestrator only after the user's OK, the PREREG commit, the repo and the splat download:
#     python -m vpd.sco prepare --run OUT/jobs/e3vs-prep-<yyyymmddHHMM> --name vpd-e3vs-prep --est-gph 2.0 -- \
#         timeout -k 120 6900 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/e3vs_prep.sh \
#         OUT/jobs/e3vs-prep-<yyyymmddHHMM>
# RUN is a NEW timestamped dir for every submission (README section 3: OUT/jobs/<job>-<yyyymmddHHMM>/).  vpd.sco
# submit refuses a RUN that was submitted before ("already submitted ... prepare a new run directory", sco.py), so
# a job cut by its timeout is resubmitted into a new RUN; it resumes from its OUTPUTS under OUT/e3vs (see RESUMES
# below), not from the RUN dir.  With more headroom (diagnosis 2026-11-09: the 6720 s soft deadline may be tight):
#     ... --est-gph 3.0 -- env VPD_JOB_DEADLINE_S=10320 timeout -k 120 10500 bash .../vpd/jobs/e3vs_prep.sh RUN
# Wall-clock cap below the reservation (est-gph 2.0 = 7200 s): the outer timeout 6900 + its 120 s kill window end
# 180 s before it (headroom for container start / teardown, which run outside the payload but are charged), and a
# soft job deadline VPD_JOB_DEADLINE_S (default 6720 s after start): every step runs under
# `timeout -k 30 <seconds left before the deadline - 60>`, a step with < VPD_MIN_STEP_S (60) left is not started.
# Rule if --est-gph changes: outer = est*3600 - 300, soft deadline = outer - 180.
# Steps (vpd_step -> RUN/steps.jsonl; a SIGTERM / SIGINT / SIGHUP adds a line {name, rc 143/130/129, signal} for the
# step it cut; the first failure ends the job, later steps are not run: each step needs the one before).  The job
# RESUMES: a resubmission skips every step whose outputs exist and are valid for the current inputs:
#   guard         vpd/PREREG_E3VS_PILOT.md committed and clean (PREREG header); its commit -> RUN/prereg_commit.txt
#   inputs        render --check-inputs: pinned repo at OUT/e3vs/repo/E3VS-BENCH + all 99 PLYs
#   gsplat_probe  render --probe -> OUT/e3vs/parity/gsplat_probe_prep.json (B18: JIT or not, seconds; always run)
#   parity        render --parity $VPD_PARITY_SCENE -> OUT/e3vs/parity/parity_<scene>.json; min PSNR >= 40 or stop
#                 (skipped when that file has pass true)
#   occupancy     occupancy build --scenes all --validate --resume -> OUT/e3vs/occ/<scene>.npz + validation.json;
#                 pass or stop (a validation failure is handed to the user: PREREG section 5 allows one threshold
#                 change).  Skipped when `occupancy check --scenes all` passes (validation pass, every npz
#                 unchanged); --resume keeps the scenes already built with the same settings.
#                 KEY = validation.json grids_sha256, the identity of the validated grids.
#                 The iPhone convention is --convention auto = the parity file's iPhone 'best'; when the parity
#                 run's iPhone part errored (no 'best'), the build refuses and the job stops (to the user: rerun
#                 parity, or choose --convention explicitly).  validation.json records iphone_convention.
#   bfs_<split>   bfs_demos --split S --procs $VPD_PREP_PROCS --collides vpd.e3vs.occupancy:load_collider --tag occ
#                 (1 thread per process) -> OUT/e3vs/bfs/occ/<split>/  (train: demo_pool.jsonl; val/test: opt_len)
#   floors        floors --collides vpd.e3vs.occupancy:load_collider --tag occ -> OUT/e3vs/floors/floors_occ.json
#                 (the frozen gate floor = its gate_floor.rate, DECISIONS B8)
#   sft           make_sft all --seed 1001 --tiers 100 --overwrite -> OUT/e3vs/sft/{warmup20.jsonl,
#                 order_seed1001.json, seed1001/demos_100.jsonl, images/}
#                 bfs_*, floors and sft each write OUT/e3vs/prep_state/<step> = KEY when they succeed and are
#                 skipped when that stamp equals the current KEY and their output exists (new grids = rerun).
#                 The gate (summary gate, E0) requires prep_state/floors and prep_state/sft == grids_sha256.
#   DONE          OUT/e3vs/PREP_DONE and RUN/DONE
# Env: VPD_DRYRUN=1 prints the commands (guard and inputs are evaluated but do not stop the dry run);
#      VPD_PREP_PROCS (default 12), VPD_PARITY_SCENE (default 00dd871005, the first val scene),
#      VPD_JOB_DEADLINE_S (default 6720), VPD_MIN_STEP_S (default 60).
#      Test hooks honoured ONLY with VPD_DRYRUN=1: VPD_E3VS_ROOT (instead of OUT/e3vs), VPD_PREP_SCENES (instead
#      of all, for the occupancy check).
# gsplat JIT (B18): CUDA_HOME / PATH / CC / CXX point at the eval-external env (nvcc + conda gcc);
# TORCH_EXTENSIONS_DIR = OUT/e3vs/torch_ext so the start jobs can copy the built extension instead of rebuilding.
# A SIGTERM / SIGINT / SIGHUP (sco delete, the outer timeout) is forwarded to the running step's process group;
# the EXIT handlers then run.

RUN=${1:-${RUN:-}}
if [[ -z $RUN ]]; then
  echo "usage: e3vs_prep.sh RUN" >&2
  exit 2
fi
export RUN
# shellcheck source=common_env.sh
source /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/common_env.sh
unset VPD_E3VS_REPO   # production reads the pinned checkout at OUT/e3vs/repo/E3VS-BENCH only

DRY=${VPD_DRYRUN:-0}
PROCS=${VPD_PREP_PROCS:-12}
SCENE=${VPD_PARITY_SCENE:-00dd871005}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-6720}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
RESERVE=60
[[ $PROCS =~ ^[1-9][0-9]*$ ]] || { echo "VPD_PREP_PROCS must be a positive integer" >&2; exit 2; }
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[0-9]+$ ]] || { echo "VPD_MIN_STEP_S must be a non-negative integer" >&2; exit 2; }
DEADLINE=$((T0 + DEADLINE_S))
E3VS=$VPD_OUT/e3vs
OCC_SCENES=all
if [[ $DRY == 1 ]]; then
  if [[ -n ${VPD_E3VS_ROOT:-} ]]; then
    E3VS=$VPD_E3VS_ROOT
    export VPD_E3VS_OCC_DIR=$E3VS/occ
  fi
  OCC_SCENES=${VPD_PREP_SCENES:-all}
else
  unset VPD_E3VS_OCC_DIR   # production: the grids at OUT/e3vs/occ only
fi
STATE=$E3VS/prep_state
PY=$PY_Q3
COLL=vpd.e3vs.occupancy:load_collider
ENV_PREFIX=$(dirname "$(dirname "$PY_EVAL")")
KEY=

mkdir -p "$RUN"
cd "$VPD_WORKTREE"

# >>> timed-run helpers (identical text in e3vs_prep.sh and e3vs_start.sh; tested by test_e3vs_loop.py)
VPD_CHILD=
VPD_CUR_STEP=
VPD_CUR_START=
VPD_CUR_T0=0
left() { echo $((DEADLINE - $(date +%s) - RESERVE)); }
step() {  # NAME CMD...: vpd_step, remembering the running step so that on_term can record the cut
  local rc=0
  VPD_CUR_STEP=$1
  VPD_CUR_START=$(date -u +%FT%TZ)
  VPD_CUR_T0=$(date +%s)
  vpd_step "$@" || rc=$?
  VPD_CUR_STEP=
  return "$rc"
}
tx() {  # run CMD under the job deadline (own process group via timeout; TERM is forwarded by on_term)
  local t rc=0
  t=$(left)
  if ((t < MIN_STEP_S)); then
    echo "[e3vs] not started, ${t}s left before the job deadline: $*" >&2
    return 124
  fi
  timeout -k 30 "$t" "$@" &
  VPD_CHILD=$!
  wait "$VPD_CHILD" || rc=$?
  VPD_CHILD=
  return "$rc"
}
on_term() {  # forward to the running step, wait for it, record it in steps.jsonl, exit (the EXIT handlers run)
  local sig=$1 rc
  case $sig in INT) rc=130 ;; HUP) rc=129 ;; *) rc=143 ;; esac
  echo "[e3vs] SIG$sig received $(date -u +%FT%TZ); stopping the running step ${VPD_CUR_STEP:-<none>}" >&2
  if [[ -n $VPD_CHILD ]]; then
    kill -TERM "$VPD_CHILD" 2>/dev/null || true
    wait "$VPD_CHILD" 2>/dev/null || true
  fi
  if [[ -n $VPD_CUR_STEP && -n ${RUN:-} ]]; then
    printf '{"name":"%s","start":"%s","end":"%s","secs":%d,"rc":%d,"signal":"%s"}\n' "$VPD_CUR_STEP" \
      "$VPD_CUR_START" "$(date -u +%FT%TZ)" $(($(date +%s) - VPD_CUR_T0)) "$rc" "$sig" >> "$RUN/steps.jsonl" || true
  fi
  exit "$rc"
}
trap 'on_term TERM' TERM
trap 'on_term INT' INT
trap 'on_term HUP' HUP
# <<< timed-run helpers

x() {  # run under the deadline, or print under VPD_DRYRUN=1
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*"
  else
    tx "$@"
  fi
}

skip() { echo "[e3vs_prep] $1: outputs exist and are current, skipped"; return 0; }

stamped() {  # STEP FILE...: the step's stamp equals KEY and its outputs exist
  local step=$1 f; shift
  [[ -n $KEY && -s $STATE/$step && $(cat "$STATE/$step") == "$KEY" ]] || return 1
  for f in "$@"; do [[ -s $f ]] || return 1; done
  return 0
}

stamp() {  # STEP
  [[ $DRY == 1 ]] && { echo "[dry] stamp $STATE/$1 = $KEY"; return 0; }
  mkdir -p "$STATE"
  printf '%s\n' "$KEY" > "$STATE/$1.tmp" && mv -f "$STATE/$1.tmp" "$STATE/$1"
}

json_pass() {  # FILE: exit 0 when the JSON file has "pass": true
  [[ -s $1 ]] && "$PY" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") is True else 1)' "$1"
}

prereg_guard() {
  local c s
  c=$(git -c safe.directory='*' -C "$VPD_WORKTREE" log -1 --format=%H -- vpd/PREREG_E3VS_PILOT.md 2>/dev/null || true)
  s=$(git --no-optional-locks -c safe.directory='*' -C "$VPD_WORKTREE" status --porcelain -- vpd/PREREG_E3VS_PILOT.md 2>/dev/null || echo "??")
  if [[ -z $c || -n $s ]]; then
    echo "[e3vs] vpd/PREREG_E3VS_PILOT.md is not committed or has local changes (commit '${c}', status '${s}')" >&2
    [[ $DRY == 1 ]] && { echo "[dry] guard would stop the job here"; return 0; }
    return 1
  fi
  echo "$c" > "$RUN/prereg_commit.txt"
  echo "[e3vs] PREREG_E3VS_PILOT.md at commit $c"
}

check_inputs() {
  if [[ $DRY == 1 ]]; then
    "$PY" -m vpd.e3vs.render --check-inputs || echo "[dry] inputs check would stop the job here"
    return 0
  fi
  tx "$PY" -m vpd.e3vs.render --check-inputs
}

gsplat_env() {
  export CUDA_HOME=${CUDA_HOME:-$ENV_PREFIX}
  export PATH="$ENV_PREFIX/bin:$PATH"
  [[ -x $ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc ]] && export CC=${CC:-$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc}
  [[ -x $ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++ ]] && export CXX=${CXX:-$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++}
  export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$E3VS/torch_ext}
  export MAX_JOBS=${MAX_JOBS:-10} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
  [[ $DRY == 1 ]] || mkdir -p "$TORCH_EXTENSIONS_DIR"
  return 0
}

parity_step() {
  if json_pass "$E3VS/parity/parity_$SCENE.json"; then skip parity; return 0; fi
  x "$PY" -m vpd.e3vs.render --parity "$SCENE"
}

occ_key() {  # prints validation.json grids_sha256 when the validated grids are all unchanged
  "$PY" -m vpd.e3vs.occupancy check --scenes "$OCC_SCENES" 2>/dev/null | tail -n 1
}

occupancy_step() {
  local k
  k=$(occ_key || true)
  if [[ -n $k ]]; then
    KEY=$k
    skip occupancy
    return 0
  fi
  x "$PY" -m vpd.e3vs.occupancy build --scenes all --validate --resume || return $?
  if [[ $DRY == 1 ]]; then
    KEY=dry
    return 0
  fi
  KEY=$(occ_key || true)
  [[ -n $KEY ]] || { echo "[e3vs] occupancy check failed after the build" >&2; return 1; }
}

bfs_split() {  # 1 thread per process, $PROCS processes
  local sp=$1
  if stamped "bfs_$sp" "$E3VS/bfs/occ/$sp/summary.json" "$E3VS/bfs/occ/$sp/opt_len.json"; then
    skip "bfs_$sp"
    return 0
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    x "$PY" -m vpd.e3vs.bfs_demos --split "$sp" --procs "$PROCS" --collides "$COLL" --tag occ || return $?
  stamp "bfs_$sp"
}

floors_step() {
  if stamped floors "$E3VS/floors/floors_occ.json"; then skip floors; return 0; fi
  x "$PY" -m vpd.e3vs.floors --collides "$COLL" --tag occ || return $?
  stamp floors
}

sft_step() {  # --overwrite: images rendered from an older demo pool are never reused
  if stamped sft "$E3VS/sft/warmup20.jsonl" "$E3VS/sft/seed1001/demos_100.jsonl" "$E3VS/sft/order_seed1001.json"; then
    skip sft
    return 0
  fi
  x "$PY" -m vpd.e3vs.make_sft all --seed 1001 --tiers 100 --overwrite || return $?
  stamp sft
}

echo "[e3vs_prep] start $(date -u +%FT%TZ) RUN=$RUN procs=$PROCS parity_scene=$SCENE dry=$DRY deadline=${DEADLINE_S}s"
vpd_node_facts "$RUN/node.txt"
gsplat_env
step guard prereg_guard
step inputs check_inputs
step gsplat_probe x "$PY" -m vpd.e3vs.render --probe --out "$E3VS/parity/gsplat_probe_prep.json"
step parity parity_step
step occupancy occupancy_step
for SP in train val test; do
  step "bfs_$SP" bfs_split "$SP"
done
step floors floors_step
step sft sft_step
if [[ $DRY == 1 ]]; then
  echo "[dry] touch $E3VS/PREP_DONE $RUN/DONE"
else
  date -u +%FT%TZ > "$E3VS/PREP_DONE"
  date -u +%FT%TZ > "$RUN/DONE"
fi
echo "[e3vs_prep] done $(date -u +%FT%TZ) after $(( $(date +%s) - T0 ))s"
