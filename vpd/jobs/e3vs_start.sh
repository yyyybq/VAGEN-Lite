#!/usr/bin/env bash
# sco payload of jobs vpd-e3vs-q3b / vpd-e3vs-q3s / vpd-e3vs-e2 (AS ROOT: the Q3B blobs are root-only; outputs are
# chowned back to 20325 at EXIT):
#     timeout -k 120 9780 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/e3vs_start.sh START RUN
# Prepared by the orchestrator only after vpd-e3vs-prep passed (parity, mesh validation) and the user's OK:
#     python -m vpd.sco prepare --run OUT/jobs/e3vs-q3s-<yyyymmddHHMM> --name vpd-e3vs-q3s --as-root --est-gph 2.8 -- \
#         timeout -k 120 9780 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/e3vs_start.sh Q3S \
#         OUT/jobs/e3vs-q3s-<yyyymmddHHMM>
# RUN is a NEW timestamped dir for every submission (README section 3: OUT/jobs/<job>-<yyyymmddHHMM>/): vpd.sco
# submit refuses a RUN that was submitted before (sco.py), so a cut job is resubmitted into a new RUN and resumes
# from its outputs under OUT/e3vs/runs/pilot/START (adapters, evals with a results.json), not from the RUN dir.
# Wall-clock cap below the reservation (est-gph 2.8 = 10080 s): the outer timeout 9780 + its 120 s kill window end
# 180 s before it (headroom for container start / teardown, which run outside the payload but are charged), and a
# soft job deadline VPD_JOB_DEADLINE_S (default 9600 s after start): every command runs under
# `timeout -k 30 <seconds left before the deadline - 120>`; a step with < VPD_MIN_STEP_S (60) left is not started.
# Rule if --est-gph changes: outer = est*3600 - 300, soft deadline = outer - 180.
# A = OUT/e3vs/runs/pilot/START/adapters, E = OUT/e3vs/runs/pilot/START/evals (PREREG section 12).
# Steps (vpd_step -> RUN/steps.jsonl; a SIGTERM / SIGINT / SIGHUP adds a line {name, rc 143/130/129, signal} for the
# step it cut).  guard and gsplat failing end the job.  A failed train / merge / eval is recorded and the job goes
# on with what does not depend on it (train_0W failed: no train_100, no 0W / 100 evals; train_100 or merge failed:
# that tier's eval is skipped); the job then exits 1 without RUN/DONE.  A step whose output exists is skipped, so a
# resubmitted job resumes: finished adapters and every eval with a results.json are not redone; each eval session
# that is redone pays its vLLM start (~0.03-0.05 GPU-h) and its 21 test PLY + mesh loads again.
#   guard       PREREG_E3VS_PILOT.md committed and clean (from a code snapshot RUN/code: the job's copy equals the file
#               of the snapshot's commit, vpd_snapshot_file); prep finished (OUT/e3vs/PREP_DONE), the mesh validation
#               (OUT/e3vs/mesh_val/validation.json) and every parity file pass, the SFT and floor files exist;
#               `mesh_collision check --scenes all` passes (same definition, every mesh unchanged, key == the
#               manifest's) and its mesh_set_sha256 == the prep stamps prep_state/floors and prep_state/sft (the
#               key J5 validated and built the floor and the demos with; any other mesh set is refused); a
#               job-local merge base with >= 40 GiB free (DECISIONS B11: /tmp/vpd_merged, else /dev/shm/vpd_merged;
#               chosen once, before any training).  open3d 0.19.0 (the official collision model) comes from O3D_DIR =
#               OUT/envs/e3vs-o3d, appended to PYTHONPATH only for collider_smoke and the run.py evals (not for
#               training, merges, or the vLLM server, from whose env run.py removes it); run.py re-checks every
#               test scene's mesh sha256.  Before it, step o3d_check (after the chown trap, recorded in steps.jsonl):
#               `mesh_collision o3d-check` refuses (exit 2) a module of O3D_DIR that PY_Q3 imports itself.
#   gsplat      copy OUT/e3vs/torch_ext (the prep job's build) to job-local $TORCH_EXTENSIONS_DIR, then the B18 probe
#               -> RUN/gsplat_probe.json (as root: does gsplat rebuild, how long); every eval copies it into its
#               manifest (run.py --gsplat-probe)
#   collider_smoke  one PY_Q3 process with the evals' PYTHONPATH imports torch and gsplat, loads the first test
#               scene's validated mesh (load_collider) and runs one official check on a 5 cm segment
#               (mesh_collision smoke); a failure ends the job before any training (after gsplat, so that gsplat
#               imports the copied build instead of compiling)
#   train_0W    PY_TRAIN n2_train_vpd.py --start S --data sft/warmup20.jsonl --out A/0W --epochs auto (12)
#               --seed 1001 --phase 0W (retrained in every start job: DECISIONS B7)
#   train_100   ... --data sft/seed1001/demos_100.jsonl --out A/100 --init-adapter A/0W --epochs auto (3)
#               --phase tier100
#   eval_raw    PY_Q3 -m vpd.e3vs.run --start S --tier raw --mode u   (the start's own weights / eval view)
#   merge_0W, eval_0W (--mode u), then the merged 0W folder is removed
#   merge_100, eval_100 (--mode u,c: ONE vLLM session and one pass over the scenes for both decoding modes, two
#               eval dirs pilot__S__100__u and pilot__S__100__c), then the merged 100 folder is removed
#   VPD_OPTIONAL=1 also runs mode c for raw and 0W, inside the same session as their u eval (descriptive; dropped
#               first when the budget is short)
#   DONE        RUN/DONE (only when no step failed)
# Only the modes without a results.json run; an unfinished eval dir is moved aside to <dir>.aborted_<utc>.
# Merged weights live on job-local disk only: MERGE_BASE = /tmp/vpd_merged if /tmp has >= 40 GiB available, else
# /dev/shm/vpd_merged if /dev/shm has, else none (the guard then stops the job before training).  VPD_MERGE_DIR may
# force one of the two (its free space is still checked).  merge.py gets --out MERGE_BASE/START__TAG, refuses
# /mnt/umm and re-checks the free space.  Merged folders are removed after their evals and at EXIT (also after
# SIGTERM / SIGINT / SIGHUP, which are forwarded to the running command's process group first), under both bases.
# Env: VPD_DRYRUN=1 (print only, "  # +o3d" marks the commands with O3D_DIR on PYTHONPATH; the guard and o3d-check
#      are evaluated but do not stop), VPD_OPTIONAL, VPD_MERGE_DIR,
#      VPD_E3VS_CONCURRENCY (default 24), VPD_LIMIT (episodes per eval; smoke only, never in the pilot),
#      VPD_JOB_DEADLINE_S (default 9600), VPD_MIN_STEP_S (default 60).
#      Test hooks honoured ONLY with VPD_DRYRUN=1: VPD_FAKE_FREE_GIB="/tmp=N,/dev/shm=M" (the free-space probe),
#      VPD_DRY_FAIL=STEP,STEP (the printed commands of those steps fail, to test the keep-going rule),
#      VPD_E3VS_ROOT (instead of OUT/e3vs: PREP_DONE, sft, floors, parity, prep_state, meshes, manifest, mesh_val,
#      runs; the guard's mesh-key comparison then runs against it).
# Cost model (estimate, unmeasured until the prep job reports PLY load times; GPU-h per start job):
#   setup + probe 0.05, train 0W 0.1, train 100 0.2-0.3, 2 CPU merges 0.1-0.2, raw/u 0.35-0.55 (never-stop raw
#   models: 377 x 25 decisions), 0W/u 0.25-0.45, 100/u+c one session 0.25-0.45; each eval adds a vLLM start
#   (~0.03-0.05) and 21 test PLY + mesh loads (once per eval thanks to the refcounted cache) -> 1.3-2.2 vs the
#   2.8 reservation.

START=${1:-}
RUN=${2:-${RUN:-}}
if [[ -z $START || -z $RUN ]]; then
  echo "usage: e3vs_start.sh START RUN" >&2
  exit 2
fi
[[ $START =~ ^(Q3B|Q3S|E2)$ ]] || { echo "e3vs_start.sh: the pilot starts are Q3B, Q3S, E2 (got '$START')" >&2; exit 2; }
export RUN
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
unset VPD_E3VS_REPO
unset VPD_E3VS_OCC_DIR VPD_E3VS_MESH_ROOT VPD_E3VS_MESH_MANIFEST VPD_E3VS_MESH_VAL_DIR  # OUT/e3vs only
# open3d for the official collision model (== vpd.e3vs.mesh_collision.O3D_DIR; test_e3vs_mesh_collision checks):
# NOT exported job-wide; collider_smoke and the run.py evals run with PYTHONPATH="$O3D_PP" (the code tree first).
O3D_DIR=$VPD_OUT/envs/e3vs-o3d
O3D_PP="$PYTHONPATH:$O3D_DIR"

DRY=${VPD_DRYRUN:-0}
OPTIONAL=${VPD_OPTIONAL:-0}
CONC=${VPD_E3VS_CONCURRENCY:-24}
LIMIT=${VPD_LIMIT:-}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-9600}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
RESERVE=120
MIN_FREE_GIB=40
SEED=1001
E3VS=$VPD_OUT/e3vs
if [[ $DRY == 1 && -n ${VPD_E3VS_ROOT:-} ]]; then  # test hook: the guard's checks against a test tree
  E3VS=$VPD_E3VS_ROOT
  export VPD_E3VS_MESH_ROOT=$E3VS/mesh_dl VPD_E3VS_MESH_MANIFEST=$E3VS/mesh_manifest.json
  export VPD_E3VS_MESH_VAL_DIR=$E3VS/mesh_val
fi
SFT=$E3VS/sft
PILOT=$E3VS/runs/pilot/$START
A=$PILOT/adapters
E=$PILOT/evals
ENV_PREFIX=$(dirname "$(dirname "$PY_EVAL")")
FAILS=()
[[ $CONC =~ ^[1-9][0-9]*$ ]] || { echo "VPD_E3VS_CONCURRENCY must be a positive integer" >&2; exit 2; }
[[ -z $LIMIT || $LIMIT =~ ^[1-9][0-9]*$ ]] || { echo "VPD_LIMIT must be a positive integer" >&2; exit 2; }
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[0-9]+$ ]] || { echo "VPD_MIN_STEP_S must be a non-negative integer" >&2; exit 2; }
DEADLINE=$((T0 + DEADLINE_S))

free_gib() {  # BASE -> GiB available on its filesystem (VPD_FAKE_FREE_GIB only under VPD_DRYRUN=1)
  local b=$1 kv
  if [[ $DRY == 1 && -n ${VPD_FAKE_FREE_GIB:-} ]]; then
    for kv in ${VPD_FAKE_FREE_GIB//,/ }; do
      [[ ${kv%%=*} == "$b" ]] && { echo "${kv#*=}"; return 0; }
    done
    echo 0
    return 0
  fi
  df -P -B1G "$b" 2>/dev/null | awk 'NR==2{print $4}' || true
}
has_room() {  # BASE: >= MIN_FREE_GIB available
  local g
  g=$(free_gib "$1")
  [[ $g =~ ^[0-9]+$ ]] && ((g >= MIN_FREE_GIB))
}
pick_merge_base() {  # DECISIONS B11: /tmp, else /dev/shm, else empty
  local b
  if [[ -n ${VPD_MERGE_DIR:-} ]]; then
    has_room "${VPD_MERGE_DIR%/vpd_merged}" && echo "$VPD_MERGE_DIR"
    return 0
  fi
  for b in /tmp /dev/shm; do
    if has_room "$b"; then
      echo "$b/vpd_merged"
      return 0
    fi
  done
  return 0
}
case ${VPD_MERGE_DIR:-} in
  ''|/tmp/vpd_merged|/dev/shm/vpd_merged) ;;
  *) echo "VPD_MERGE_DIR must be /tmp/vpd_merged or /dev/shm/vpd_merged (job-local disk)" >&2; exit 2 ;;
esac
MERGE_BASE=$(pick_merge_base)
FREE_FACTS="/tmp $(free_gib /tmp) GiB, /dev/shm $(free_gib /dev/shm) GiB available"

mkdir -p "$RUN"
[[ $DRY == 1 ]] || mkdir -p "$A" "$E"
cd "$VPD_WORKTREE"
vpd_chown_trap "$RUN" "$PILOT"
# a timeout or signal must not leave merged weights on the node: both bases, fixed names only
[[ $DRY == 1 ]] || vpd_on_exit "for d in /tmp/vpd_merged/${START}__0W /tmp/vpd_merged/${START}__100 /dev/shm/vpd_merged/${START}__0W /dev/shm/vpd_merged/${START}__100; do rm -rf -- \"\$d\"; done"

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

x() {  # run under the deadline, or print under VPD_DRYRUN=1 (VPD_DRY_FAIL=STEP,...: those steps' commands fail;
       # "  # +o3d": the command gets O3D_DIR on PYTHONPATH)
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*$([[ :$PYTHONPATH: == *":$O3D_DIR:"* ]] && echo "  # +o3d")"
    if [[ -n $VPD_CUR_STEP && ,${VPD_DRY_FAIL:-}, == *",$VPD_CUR_STEP,"* ]]; then
      echo "[dry] (test hook) step $VPD_CUR_STEP fails" >&2
      return 3
    fi
  else
    tx "$@"
  fi
}

o3d_check() {  # N3 (identical text in both payloads): a module of O3D_DIR that PY_Q3 imports itself is refused
  if "$PY_Q3" -m vpd.e3vs.mesh_collision o3d-check; then
    return 0
  fi
  [[ $DRY == 1 ]] && { echo "[dry] o3d-check would stop the job here"; return 0; }
  return 1
}

guard() {
  local c s bad=0
  if [[ -e $VPD_WORKTREE/SNAPSHOT.json ]]; then
    # code snapshot (vpd.sco prepare, DECISIONS G1): no .git here; the job's copy must be the committed file of the
    # snapshot's commit (blob id; git show against the real worktree when git exists); c = its last commit
    c=$(vpd_snapshot_file vpd/PREREG_E3VS_PILOT.md) || c=
    s=
  else
    c=$(git -c safe.directory='*' -C "$VPD_WORKTREE" log -1 --format=%H -- vpd/PREREG_E3VS_PILOT.md 2>/dev/null || true)
    s=$(git --no-optional-locks -c safe.directory='*' -C "$VPD_WORKTREE" status --porcelain -- vpd/PREREG_E3VS_PILOT.md 2>/dev/null || echo "??")
  fi
  if [[ -z $c || -n $s ]]; then
    echo "[e3vs] vpd/PREREG_E3VS_PILOT.md is not committed or has local changes" >&2
    bad=1
  else
    echo "$c" > "$RUN/prereg_commit.txt"
  fi
  for f in "$E3VS/PREP_DONE" "$SFT/warmup20.jsonl" "$SFT/seed$SEED/demos_100.jsonl" "$E3VS/floors/floors_mesh.json" \
           "$E3VS/mesh_val/validation.json"; do
    [[ -s $f ]] || { echo "[e3vs] missing $f" >&2; bad=1; }
  done
  if ! "$PY_Q3" - "$E3VS" <<'PY'
import json, sys
from pathlib import Path
e3 = Path(sys.argv[1])
v = e3 / "mesh_val" / "validation.json"
ok = v.exists() and json.loads(v.read_text()).get("pass") is True
par = sorted((e3 / "parity").glob("parity_*.json"))
ok = ok and bool(par) and all(json.loads(p.read_text()).get("pass") is True for p in par)
print(f"[e3vs] mesh validation + parity pass: {ok}")
sys.exit(0 if ok else 1)
PY
  then
    bad=1
  fi
  # the mesh set key: the current meshes are the validated ones, and the floor / demos were built on them
  local key st k
  key=$("$PY_Q3" -m vpd.e3vs.mesh_collision check --scenes all | tail -n 1) || key=
  if [[ -z $key ]]; then
    echo "[e3vs] mesh check failed (python -m vpd.e3vs.mesh_collision check --scenes all)" >&2
    bad=1
  else
    for st in floors sft; do
      k=$(cat "$E3VS/prep_state/$st" 2>/dev/null || true)
      if [[ $k != "$key" ]]; then
        echo "[e3vs] prep_state/$st '${k}' != the validated mesh set $key: refusing" >&2
        bad=1
      fi
    done
    if [[ $bad == 0 ]]; then
      echo "[e3vs] mesh set $key (validated; floor and SFT stamped with it)"
    fi
  fi
  # B11, before any GPU time is spent: a merge that is still needed has a job-local base with >= 40 GiB
  # shellcheck disable=SC2086
  if [[ -n $(todo_modes 0W $MODES_0W) || -n $(todo_modes 100 u c) ]]; then
    if [[ -z $MERGE_BASE ]]; then
      echo "[e3vs] no job-local disk with >= ${MIN_FREE_GIB} GiB free for the merges (${FREE_FACTS};" \
           "VPD_MERGE_DIR='${VPD_MERGE_DIR:-}'): stopping before training (DECISIONS B11)" >&2
      bad=1
    else
      echo "[e3vs] merge base $MERGE_BASE (${FREE_FACTS})"
    fi
  fi
  if [[ $bad != 0 ]]; then
    [[ $DRY == 1 ]] && { echo "[dry] guard would stop the job here"; return 0; }
    return 1
  fi
  return 0
}

gsplat_env() {
  export CUDA_HOME=${CUDA_HOME:-$ENV_PREFIX}
  # conda CUDA keeps headers / libs in targets/x86_64-linux/include and lib, not include / lib64 (the only dirs
  # torch cpp_extension adds): without these g++ missed cuda_runtime_api.h (J5 2026-11-12, DECISIONS G19a)
  [[ -d $CUDA_HOME/targets/x86_64-linux/include ]] && \
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export PATH="$ENV_PREFIX/bin:$PATH"
  [[ -x $ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc ]] && export CC=${CC:-$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc}
  [[ -x $ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++ ]] && export CXX=${CXX:-$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++}
  export TORCH_EXTENSIONS_DIR=${VPD_TORCH_EXT:-/tmp/vpd_torch_ext}
  export MAX_JOBS=${MAX_JOBS:-10} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
  return 0
}

gsplat_setup() {
  if [[ $DRY == 1 ]]; then
    echo "[dry] cp -a $E3VS/torch_ext/. $TORCH_EXTENSIONS_DIR/ ; $PY_Q3 -m vpd.e3vs.render --probe --out $RUN/gsplat_probe.json"
    return 0
  fi
  mkdir -p "$TORCH_EXTENSIONS_DIR"
  if [[ -d $E3VS/torch_ext ]]; then
    cp -a "$E3VS/torch_ext/." "$TORCH_EXTENSIONS_DIR/"
  fi
  tx "$PY_Q3" -m vpd.e3vs.render --probe --out "$RUN/gsplat_probe.json"
}

train() {  # TAG DATA [INIT]
  local tag=$1 data=$2 init=${3:-} phase
  phase=$([[ $tag == 0W ]] && echo 0W || echo "tier$tag")
  if [[ -s $A/$tag/adapter.pt && -s $A/$tag/meta.json ]]; then
    echo "[e3vs] adapter $tag exists, skipped"
    return 0
  fi
  local args=(--start "$START" --data "$data" --out "$A/$tag" --epochs auto --seed "$SEED" --phase "$phase" --threads 2)
  [[ -n $init ]] && args+=(--init-adapter "$init")
  x "$PY_TRAIN" vpd/train/n2_train_vpd.py "${args[@]}"
}

collider_smoke() {  # the evals' process setup, before any training (S3): fail fast
  PYTHONPATH="$O3D_PP" x "$PY_Q3" -m vpd.e3vs.mesh_collision smoke --split test --imports torch,gsplat
}

merged_dir() {  # TAG -> MERGE_BASE/START__TAG, only under the two job-local bases
  case $MERGE_BASE in
    /tmp/vpd_merged|/dev/shm/vpd_merged) echo "$MERGE_BASE/${START}__$1" ;;
    *) echo "[e3vs] no job-local merge base ('$MERGE_BASE', ${FREE_FACTS})" >&2; return 1 ;;
  esac
}

merge() {  # TAG
  local tag=$1 dst
  dst=$(merged_dir "$tag") || return 1
  if [[ $DRY == 1 ]]; then
    echo "[dry] clear $dst (rm -rf before the merge)"
    x "$PY_Q3" vpd/train/merge.py --start "$START" --adapter "$A/$tag" --out "$dst"
    return
  fi
  rm -rf -- "$dst"
  mkdir -p "$MERGE_BASE"
  tx "$PY_Q3" vpd/train/merge.py --start "$START" --adapter "$A/$tag" --out "$dst"
}

unmerge() {
  local dst
  dst=$(merged_dir "$1") || return 0  # nothing was merged
  x rm -rf -- "$dst"
}

need_eval() {  # TIER MODE: 0 when the eval still has to run
  [[ ! -s $E/pilot__${START}__$1__$2/results.json ]]
}

todo_modes() {  # TIER MODE... -> the comma list of the modes without results.json (empty: nothing to do)
  local tier=$1 m out=(); shift
  for m in "$@"; do need_eval "$tier" "$m" && out+=("$m"); done
  local IFS=,
  echo "${out[*]}"
}

evaluate() {  # TIER "MODE..." [ADAPTER_TAG]: one run.py session for the listed modes that are not done yet
  local tier=$1 tag=${3:-} modes m dir
  local -a want
  read -r -a want <<< "$2"
  modes=$(todo_modes "$tier" "${want[@]}")
  if [[ -z $modes ]]; then
    echo "[e3vs] $tier ${want[*]} done, skipped"
    return 0
  fi
  for m in ${modes//,/ }; do
    dir=$E/pilot__${START}__${tier}__$m
    if [[ $DRY != 1 && -e $dir ]]; then  # an unfinished earlier attempt: keep it aside, never overwrite
      mv -- "$dir" "$dir.aborted_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
  done
  local args=(--start "$START" --tier "$tier" --mode "$modes" --split test --out-root "$E" --concurrency "$CONC"
              --gsplat-probe "$RUN/gsplat_probe.json")
  [[ -n $LIMIT ]] && args+=(--limit "$LIMIT")
  [[ -n $tag ]] && args+=(--model-dir "$MERGE_BASE/${START}__$tag" --adapter-meta "$A/$tag/meta.json")
  PYTHONPATH="$O3D_PP" x "$PY_Q3" -m vpd.e3vs.run "${args[@]}"  # run.py drops O3D_DIR for its vLLM server
}

MODES_RAW="u"
MODES_0W="u"
if [[ $OPTIONAL == 1 ]]; then
  MODES_RAW="u c"
  MODES_0W="u c"
fi

merged_eval() {  # TAG "MODES": merge, one eval session, unmerge; a failed merge skips the eval (recorded in FAILS)
  local tag=$1 modes=$2
  if step "merge_$tag" merge "$tag"; then
    step "eval_$tag" evaluate "$tag" "$modes" "$tag" || FAILS+=("eval_$tag")
  else
    FAILS+=("merge_$tag")
  fi
  step "unmerge_$tag" unmerge "$tag" || FAILS+=("unmerge_$tag")
}

echo "[e3vs_start] $START start $(date -u +%FT%TZ) RUN=$RUN optional=$OPTIONAL dry=$DRY uid=$(id -u)" \
     "deadline=${DEADLINE_S}s merge_base=${MERGE_BASE:-<none>} (${FREE_FACTS})"
vpd_node_facts "$RUN/node.txt"
gsplat_env
step o3d_check o3d_check || exit 2  # after the chown trap (RUN, PILOT chowned at EXIT); in steps.jsonl
step guard guard          # a failure ends the job (set -e)
step gsplat gsplat_setup  # a failure ends the job: no eval can render without gsplat
step collider_smoke collider_smoke  # a failure ends the job: torch + gsplat + open3d and the test-scene mesh
OK_0W=1
OK_100=1
step train_0W train 0W "$SFT/warmup20.jsonl" || { FAILS+=(train_0W); OK_0W=0; OK_100=0; }
if ((OK_0W)); then
  step train_100 train 100 "$SFT/seed$SEED/demos_100.jsonl" "$A/0W" || { FAILS+=(train_100); OK_100=0; }
fi
step eval_raw evaluate raw "$MODES_RAW" || FAILS+=(eval_raw)
# shellcheck disable=SC2086
if ((OK_0W)) && [[ -n $(todo_modes 0W $MODES_0W) ]]; then
  merged_eval 0W "$MODES_0W"
fi
if ((OK_100)) && [[ -n $(todo_modes 100 u c) ]]; then
  merged_eval 100 "u c"
fi
df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
if ((${#FAILS[@]})); then
  echo "[e3vs_start] $START finished with failed steps: ${FAILS[*]} (no RUN/DONE; a resubmission redoes only" \
       "these) $(date -u +%FT%TZ) after $(( $(date +%s) - T0 ))s" >&2
  exit 1
fi
if [[ $DRY == 1 ]]; then
  echo "[dry] touch $RUN/DONE"
else
  date -u +%FT%TZ > "$RUN/DONE"
fi
echo "[e3vs_start] $START done $(date -u +%FT%TZ) after $(( $(date +%s) - T0 ))s"
