#!/usr/bin/env bash
# sco payload of job vpd-e3vs-cal (round 4, OUT/plan/ROUND4.md item 1.1 and sections A / B; AS ROOT: the Q3B blobs
# are root-only; outputs are chowned back to 20325 at EXIT):
#     timeout -k 120 12300 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/e3vs_cal.sh RUN
#     python -m vpd.sco prepare --run OUT/jobs/e3vs-cal-<yyyymmddHHMM> --name vpd-e3vs-cal --as-root --est-gph 3.5 \
#         -- timeout -k 120 12300 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/e3vs_cal.sh \
#         OUT/jobs/e3vs-cal-<yyyymmddHHMM>
# Q3B only, EXPLORATORY, val only: the demo tiers 300 and ALL (make_sft tier, ROUND4.md B), each trained from the v1
# Q3B 0W adapter (OUT/e3vs/runs/pilot/Q3B/adapters/0W) and evaluated once on val in mode u (unconstrained, the
# primary arm).  No test eval, no other start, no comparison.  Success definition, collision model, floor, prompt
# and episode files are the v1 ones (same run.py, make_sft prompt, mesh key).
# P = OUT/e3vs/runs/pilot_v2/Q3B: P/adapters/{300,ALL}, P/cal_val/pilot__Q3B__{300,ALL}__u (eval names as v1).
# Rule if --est-gph changes: outer = est*3600 - 300, soft deadline VPD_JOB_DEADLINE_S = outer - 180 (default 12120).
# RUN is a NEW timestamped dir for every submission; a resubmitted job resumes from the outputs (a tier's demos
# with a manifest and a current prep_state/sft_<T> stamp, adapters with adapter.pt + meta.json, evals with
# results.json are not redone).
# Steps (vpd_step -> RUN/steps.jsonl; a signal adds a line for the step it cut):
#   o3d_check, guard  prep finished (PREP_DONE), mesh validation + every parity file pass, the mesh key of
#               `mesh_collision check --scenes all` == prep_state/{floors,sft,bfs_train,bfs_val}; the order and the
#               pool exist and the order's pool_sha256 is the pool's; the 0W adapter: adapter.pt, meta start Q3B,
#               phase 0W, seed 1001, data_sha256 == sha256(sft/warmup20.jsonl); a job-local merge base (B11)
#   gsplat      copy OUT/e3vs/torch_ext, probe -> RUN/gsplat_probe.json
#   collider_smoke  mesh_collision smoke --split val --imports torch,gsplat
#   300 first (its val result alone decides n_top when it passes, ROUND4.md A.3), then ALL; per tier T:
#   sft_T       make_sft tier --seed 1001 --tiers T (renders only the images that do not exist yet; stamp
#               prep_state/sft_T = the mesh key)
#   train_T     PY_TRAIN n2_train_vpd.py --start Q3B --data sft/seed1001/demos_T.jsonl --out P/adapters/T
#               --init-adapter OUT/e3vs/runs/pilot/Q3B/adapters/0W --epochs auto (3) --seed 1001 --phase tierT
#   merge_T, eval_T (run.py --tier T --mode u --split val --out-root P/cal_val), unmerge_T
#   DONE        RUN/DONE (only when no step failed)
# A failed sft_T skips train_T and eval_T; a failed train_T or merge_T skips eval_T; the job goes on to the next
# tier and exits 1 without RUN/DONE.
# Env: VPD_DRYRUN=1 (print only; the guard and o3d-check are evaluated but do not stop), VPD_MERGE_DIR,
#      VPD_E3VS_CONCURRENCY (default 24), VPD_JOB_DEADLINE_S (default 12120), VPD_MIN_STEP_S (default 60),
#      VPD_CAL_TIERS (default "300 ALL"; a subset in that order, e.g. "ALL" to redo one tier).
# Cost model (from the v1 Q3B job, 2026-11-14: train_100 405 s for 423 records x 3 epochs, eval_100 u+c 323 s,
# sft 270 s): sft_300 ~0.05, train_300 ~0.3, sft_ALL ~0.2-0.3, train_ALL ~1.3-1.6, 2 merges ~0.04,
# 2 val evals (231 episodes, u) ~0.1-0.15, setup ~0.03 -> ~2.1-2.5 GPU-h vs the 3.5 reservation.

RUN=${1:-${RUN:-}}
if [[ -z $RUN ]]; then
  echo "usage: e3vs_cal.sh RUN" >&2
  exit 2
fi
export RUN
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
unset VPD_E3VS_REPO
unset VPD_E3VS_OCC_DIR VPD_E3VS_MESH_ROOT VPD_E3VS_MESH_MANIFEST VPD_E3VS_MESH_VAL_DIR  # OUT/e3vs only
O3D_DIR=$VPD_OUT/envs/e3vs-o3d
O3D_PP="$PYTHONPATH:$O3D_DIR"

START=Q3B
DRY=${VPD_DRYRUN:-0}
CONC=${VPD_E3VS_CONCURRENCY:-24}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-12120}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
RESERVE=120
MIN_FREE_GIB=40
SEED=1001
read -r -a CAL_TIERS <<< "${VPD_CAL_TIERS:-300 ALL}"
E3VS=$VPD_OUT/e3vs
SFT=$E3VS/sft
STATE=$E3VS/prep_state
POOL=$E3VS/bfs/mesh/train/demo_pool.jsonl
INIT=$E3VS/runs/pilot/Q3B/adapters/0W
V2=$E3VS/runs/pilot_v2
PILOT=$V2/$START
A=$PILOT/adapters
E=$PILOT/cal_val
ENV_PREFIX=$(dirname "$(dirname "$PY_EVAL")")
FAILS=()
KEY=
[[ $CONC =~ ^[1-9][0-9]*$ ]] || { echo "VPD_E3VS_CONCURRENCY must be a positive integer" >&2; exit 2; }
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[0-9]+$ ]] || { echo "VPD_MIN_STEP_S must be a non-negative integer" >&2; exit 2; }
for t in "${CAL_TIERS[@]}"; do
  [[ $t == 300 || $t == ALL ]] || { echo "VPD_CAL_TIERS: only 300 and ALL (got '$t')" >&2; exit 2; }
done
DEADLINE=$((T0 + DEADLINE_S))

free_gib() {  # BASE -> GiB available on its filesystem
  df -P -B1G "$1" 2>/dev/null | awk 'NR==2{print $4}' || true
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
# V2 itself too (the J6 jobs left runs/pilot root:root, DECISIONS G22d); the new demos / images / stamps
vpd_chown_trap "$RUN" "$V2" "$SFT/seed$SEED" "$SFT/images" "$STATE"
[[ $DRY == 1 ]] || vpd_on_exit "for d in /tmp/vpd_merged/${START}__300 /tmp/vpd_merged/${START}__ALL /dev/shm/vpd_merged/${START}__300 /dev/shm/vpd_merged/${START}__ALL; do rm -rf -- \"\$d\"; done"

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

x() {  # run under the deadline, or print under VPD_DRYRUN=1 ("  # +o3d": the command gets O3D_DIR on PYTHONPATH)
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*$([[ :$PYTHONPATH: == *":$O3D_DIR:"* ]] && echo "  # +o3d")"
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
  local bad=0 f st k
  for f in "$E3VS/PREP_DONE" "$SFT/warmup20.jsonl" "$SFT/order_seed$SEED.json" "$POOL" \
           "$E3VS/floors/floors_mesh.json" "$E3VS/mesh/validation.json" "$INIT/adapter.pt" "$INIT/meta.json"; do
    [[ -s $f ]] || { echo "[e3vs] missing $f" >&2; bad=1; }
  done
  if ! "$PY_Q3" - "$E3VS" "$INIT" "$SEED" <<'PY'
import hashlib, json, sys
from pathlib import Path
e3, init, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
v = e3 / "mesh" / "validation.json"
ok = v.exists() and json.loads(v.read_text()).get("pass") is True
par = sorted((e3 / "parity").glob("parity_*.json"))
ok = ok and bool(par) and all(json.loads(p.read_text()).get("pass") is True for p in par)
print(f"[e3vs] mesh validation + parity pass: {ok}")
order = json.loads((e3 / "sft" / f"order_seed{seed}.json").read_text())
pool = e3 / "bfs" / "mesh" / "train" / "demo_pool.jsonl"
o_ok = order.get("pool_sha256") == sha(pool) and order.get("seed") == seed
print(f"[e3vs] order_seed{seed}.json built from the current pool: {o_ok}")
m = json.loads((init / "meta.json").read_text())
want = {"start": "Q3B", "phase": "0W", "seed": seed, "data_sha256": sha(e3 / "sft" / "warmup20.jsonl")}
got = {k: m.get(k) for k in want}
a_ok = got == want
print(f"[e3vs] 0W adapter {init}: {got} -> {a_ok}")
sys.exit(0 if ok and o_ok and a_ok else 1)
PY
  then
    bad=1
  fi
  KEY=$("$PY_Q3" -m vpd.e3vs.mesh_collision check --scenes all | tail -n 1) || KEY=
  if [[ -z $KEY ]]; then
    echo "[e3vs] mesh check failed (python -m vpd.e3vs.mesh_collision check --scenes all)" >&2
    bad=1
  else
    for st in floors sft bfs_train bfs_val; do
      k=$(cat "$STATE/$st" 2>/dev/null || true)
      if [[ $k != "$KEY" ]]; then
        echo "[e3vs] prep_state/$st '${k}' != the validated mesh set $KEY: refusing" >&2
        bad=1
      fi
    done
    [[ $bad == 0 ]] && echo "[e3vs] mesh set $KEY (validated; floor, BFS and v1 SFT stamped with it)"
  fi
  if [[ -z $MERGE_BASE ]]; then
    echo "[e3vs] no job-local disk with >= ${MIN_FREE_GIB} GiB free for the merges (${FREE_FACTS};" \
         "VPD_MERGE_DIR='${VPD_MERGE_DIR:-}'): stopping before training (DECISIONS B11)" >&2
    bad=1
  else
    echo "[e3vs] merge base $MERGE_BASE (${FREE_FACTS})"
  fi
  if [[ $bad != 0 ]]; then
    [[ $DRY == 1 ]] && { echo "[dry] guard would stop the job here"; return 0; }
    return 1
  fi
  return 0
}

gsplat_env() {
  export CUDA_HOME=${CUDA_HOME:-$ENV_PREFIX}
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

collider_smoke() {
  PYTHONPATH="$O3D_PP" x "$PY_Q3" -m vpd.e3vs.mesh_collision smoke --split val --imports torch,gsplat
}

sft_tier() {  # T: demos_T.jsonl + manifest, stamped with the mesh key
  local t=$1
  if [[ -n $KEY && -s $STATE/sft_$t && $(cat "$STATE/sft_$t") == "$KEY" && -s $SFT/seed$SEED/demos_$t.jsonl && \
        -s $SFT/seed$SEED/demos_${t}_manifest.json ]]; then
    echo "[e3vs] sft $t: outputs exist and are current, skipped"
    return 0
  fi
  x "$PY_Q3" -m vpd.e3vs.make_sft tier --seed "$SEED" --tiers "$t" || return $?
  if [[ $DRY == 1 ]]; then
    echo "[dry] stamp $STATE/sft_$t = $KEY"
  else
    printf '%s\n' "$KEY" > "$STATE/sft_$t.tmp" && mv -f "$STATE/sft_$t.tmp" "$STATE/sft_$t"
  fi
}

train() {  # T
  local tag=$1
  if [[ -s $A/$tag/adapter.pt && -s $A/$tag/meta.json ]]; then
    echo "[e3vs] adapter $tag exists, skipped"
    return 0
  fi
  x "$PY_TRAIN" vpd/train/n2_train_vpd.py --start "$START" --data "$SFT/seed$SEED/demos_$tag.jsonl" --out "$A/$tag" \
    --init-adapter "$INIT" --epochs auto --seed "$SEED" --phase "tier$tag" --threads 2
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
  dst=$(merged_dir "$1") || return 0
  x rm -rf -- "$dst"
}

evaluate() {  # TAG: one run.py session, val, mode u
  local tag=$1 dir
  dir=$E/pilot__${START}__${tag}__u
  if [[ -s $dir/results.json ]]; then
    echo "[e3vs] val eval $tag done, skipped"
    return 0
  fi
  if [[ $DRY != 1 && -e $dir ]]; then  # an unfinished earlier attempt: keep it aside, never overwrite
    mv -- "$dir" "$dir.aborted_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  PYTHONPATH="$O3D_PP" x "$PY_Q3" -m vpd.e3vs.run --start "$START" --tier "$tag" --mode u --split val \
    --out-root "$E" --concurrency "$CONC" --gsplat-probe "$RUN/gsplat_probe.json" \
    --model-dir "$MERGE_BASE/${START}__$tag" --adapter-meta "$A/$tag/meta.json"
}

echo "[e3vs_cal] start $(date -u +%FT%TZ) RUN=$RUN tiers=${CAL_TIERS[*]} dry=$DRY uid=$(id -u)" \
     "deadline=${DEADLINE_S}s merge_base=${MERGE_BASE:-<none>} (${FREE_FACTS})"
vpd_node_facts "$RUN/node.txt"
gsplat_env
step o3d_check o3d_check || exit 2
step guard guard
step gsplat gsplat_setup
step collider_smoke collider_smoke
for T in "${CAL_TIERS[@]}"; do
  if ! step "sft_$T" sft_tier "$T"; then
    FAILS+=("sft_$T")
    continue
  fi
  if ! step "train_$T" train "$T"; then
    FAILS+=("train_$T")
    continue
  fi
  if [[ -s $E/pilot__${START}__${T}__u/results.json ]]; then
    echo "[e3vs] val eval $T done, skipped"
    continue
  fi
  if step "merge_$T" merge "$T"; then
    step "eval_$T" evaluate "$T" || FAILS+=("eval_$T")
  else
    FAILS+=("merge_$T")
  fi
  step "unmerge_$T" unmerge "$T" || FAILS+=("unmerge_$T")
done
df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
if ((${#FAILS[@]})); then
  echo "[e3vs_cal] finished with failed steps: ${FAILS[*]} (no RUN/DONE) $(date -u +%FT%TZ)" \
       "after $(( $(date +%s) - T0 ))s" >&2
  exit 1
fi
if [[ $DRY == 1 ]]; then
  echo "[dry] touch $RUN/DONE"
else
  date -u +%FT%TZ > "$RUN/DONE"
fi
echo "[e3vs_cal] done $(date -u +%FT%TZ) after $(( $(date +%s) - T0 ))s"
