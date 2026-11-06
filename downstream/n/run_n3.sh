#!/usr/bin/env bash
# N3 (the transfer ladder; the user's decision 2 of 2026-11-03), one one-card sco job per model: its tiers in order,
# each with run_n3_episodes.sh into N3/<tier>/<MODEL>:
#   c  N1's merged instances solvable under the upstream camera sphere (run_n3_recheck.sh's solvable.jsonl), with that
#      collision rule (N3_COLLISION=sphere); S, RS, RD, Q3S, Q3RS, Q3RD only (the design's six);
#   e  tier (e)'s instances (n3_gen.py scene with the robot's own camera; the camera comes with them), N1's collision
#      rule; all eight models.
#   f  (the user's task book of 2026-11-05, item 4) tier c's instances, N1's robot body but turning in place never
#      blocked (N3_COLLISION=robot_free_turn); the same six models as c.
#   gd, gc  (the user's task book of 2026-11-06, tier g) with dialogue history (N_HISTORY=1, n_episode.py --history):
#      gd = tier d, N1 as it is on its merged 600 instances (D_INSTANCES, default N2's instances_600.jsonl), N1's
#      robot body, all eight models; gc = tier c again (C_INSTANCES, the camera sphere), the six models of c.
#   dr, drc  (tier g's check, run alone before gd / gc) tier d again without history on DR_INSTANCES (a subset of
#      N1's 600, n_dr_check.py pick), all eight models; tier c again on DRC_INSTANCES (a subset of c's), the six;
#      n_dr_check.py compare then sets them against N1's and c's recorded episodes.
# Usage: run_n3.sh N3 MODEL C_INSTANCES E_INSTANCES [TIERS] [WORKERS]   (TIERS default "c e"; a tier the model is not
#   in is skipped).  The Q3 models get their own validation config (N_VAL_ENV, as in N1) and run as the container's
#   root; everything under N3/<tier>/<MODEL> is given back to its owner by run_n3_episodes.sh.
#   A tier may carry its own number of episodes at once ("c:4 e:12"; else WORKERS).  N3_DROP_CORRUPTED=1: before each
#   tier, the episodes with corrupted frames already in its folder are moved aside (n_repair.py drop; nothing deleted),
#   so the resumable runner plays them again (11-03: corrupted frames showed with 12 at once); since 11-05 the
#   episodes with a near-black frame too (n_repair.py --broken, the user's plan 甲).  A tier given twice ("f:12 f:4")
#   with N3_DROP_CORRUPTED=1 runs at 12, then plays the broken episodes again at 4.
set -uo pipefail
N3="${1:?Usage: run_n3.sh N3 MODEL C_INSTANCES E_INSTANCES [TIERS] [WORKERS]}"; MODEL="${2:?}"; CI="${3:?}"; EI="${4:?}"
TIERS="${5:-c e}"; WORKERS="${6:-12}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case $MODEL in
  Q3*) export N_VAL_ENV="${N_VAL_ENV_QWEN3:-/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/rl_q3d/val_env.yaml}" ;;
esac
mkdir -p "$N3"
log() { echo "[n3] $* $(date -u +%FT%TZ)" | tee -a "$N3/n3_$MODEL.log"; }
rc_all=0
for spec in $TIERS; do
  t="${spec%%:*}"; w="$WORKERS"
  [[ $spec == *:* ]] && w="${spec#*:}"
  case $t in
    c) [[ " S RS RD Q3S Q3RS Q3RD " == *" $MODEL "* ]] || { log "tier c: $MODEL is not in it"; continue; }
       INST="$CI"; COLL=sphere ;;
    e) INST="$EI"; COLL=robot ;;
    f) [[ " S RS RD Q3S Q3RS Q3RD " == *" $MODEL "* ]] || { log "tier f: $MODEL is not in it"; continue; }
       INST="$CI"; COLL=robot_free_turn ;;
    gd) INST="${D_INSTANCES:-/mnt/umm/users/yinbaiqiao/downstream_runs/n/n2_20261103/instances_600.jsonl}"; COLL=robot ;;
    gc) [[ " S RS RD Q3S Q3RS Q3RD " == *" $MODEL "* ]] || { log "tier gc: $MODEL is not in it"; continue; }
        INST="$CI"; COLL=sphere ;;
    dr) INST="${DR_INSTANCES:?tier dr needs DR_INSTANCES}"; COLL=robot ;;
    drc) [[ " S RS RD Q3S Q3RS Q3RD " == *" $MODEL "* ]] || { log "tier drc: $MODEL is not in it"; continue; }
         INST="${DRC_INSTANCES:?tier drc needs DRC_INSTANCES}"; COLL=sphere ;;
    *) log "unknown tier $t"; rc_all=2; continue ;;
  esac
  if [[ ${N3_DROP_CORRUPTED:-0} == 1 && -f $N3/$t/$MODEL/episodes.jsonl ]]; then
    log "tier $t: $("${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}" "$WT/downstream/n/n_repair.py" drop --out "$N3/$t/$MODEL" --broken)"
  fi
  HIST=0; [[ $t == g* ]] && HIST=1
  mkdir -p "$N3/$t"
  # a root job (the Q3 models) may make a tier folder first: give it and this job's log back to the owner, so the
  # owner's jobs can add their model folders (11-06; run_n3_episodes.sh gives back only N3/<tier>/<MODEL>)
  if [[ $(id -u) -eq 0 ]]; then chown "${OWNER_UID:-20325}:${OWNER_GID:-20325}" "$N3" "$N3/$t" "$N3/n3_$MODEL.log" 2>/dev/null; fi
  N_HISTORY=$HIST N3_COLLISION=$COLL bash "$WT/downstream/n/run_n3_episodes.sh" "$N3/$t/$MODEL" "$INST" "$MODEL" "$w" 0
  rc=$?
  log "tier $t: rc=$rc $(grep -c '"success": true' "$N3/$t/$MODEL/episodes.jsonl" 2>/dev/null) successes of $(wc -l < "$N3/$t/$MODEL/episodes.jsonl" 2>/dev/null)"
  (( rc )) && rc_all=$rc
done
exit $rc_all
