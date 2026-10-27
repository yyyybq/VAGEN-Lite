#!/bin/bash
# Reward line, step 2a on the login node (14 CPUs): the ten check tasks' depth-12 tables and every check.
#   A  tables with the main line's prefilter and with the strict one (10 processes each, at the same time)
#   B  every state the main line's prefilter skips judged too (14 processes): must hold no success
#   C  strict vs main-line prefilter tables: must be identical
#   D  the q3d / q3s0 training records replayed: every pose the env scores looked up (must all be found)
#   E  against the old tables, the goal-side tables and a re-search of poses outside the old tables
# Logs in reward_runs/logs/2a_*, results in reward_runs/analysis/2a_*; reward_runs/logs/2a_done when finished.
# Start: cd W; nohup bash reward/run_2a.sh > /mnt/umm/users/yinbaiqiao/reward_runs/logs/2a_run.log 2>&1 < /dev/null &
set -u
W=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-reward
RR=/mnt/umm/users/yinbaiqiao/reward_runs
E=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial
D13=$E/interiorgs_v10_r12_20261019/analysis/r13_distance
Q=/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
ROWS=$E/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl
GS=/mnt/umm/users/yinbaiqiao/InteriorGS_ply
T=$RR/tasks/calib10.txt
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$W" || exit 1
mkdir -p "$RR/logs" "$RR/analysis" "$RR/tables"
stamp() { echo "$(date -u +%FT%TZ) $*" >> "$RR/logs/2a_phases.log"; }

stamp "A start"
$PY -m reward.distance12 build --rows "$ROWS" --task-ids "$T" --out "$RR/tables/calib10_loose" --gs-root "$GS" \
    --prefilter loose --workers 10 > "$RR/logs/2a_build_loose.log" 2>&1 &
$PY -m reward.distance12 build --rows "$ROWS" --task-ids "$T" --out "$RR/tables/calib10_strict" --gs-root "$GS" \
    --prefilter strict --workers 10 > "$RR/logs/2a_build_strict.log" 2>&1 &
wait
stamp "A done"

stamp "B start"
$PY -m reward.distance12 check-prefilter --rows "$ROWS" --task-ids "$T" --gs-root "$GS" --workers 14 \
    > "$RR/analysis/2a_check_prefilter.jsonl" 2> "$RR/logs/2a_check_prefilter.err"
stamp "B done"

stamp "C start"
$PY -m reward.distance12 compare "$RR/tables/calib10_loose" "$RR/tables/calib10_strict" --task-ids "$T" \
    > "$RR/analysis/2a_compare_strict.json" 2> "$RR/logs/2a_compare.err"
stamp "C done"

stamp "D start"
for run in q3d q3s0; do
    $PY -m reward.replay_q3 --run "$Q/rl_run_$run" --env-yaml "$Q/rl_$run/train_env.yaml" --rows "$ROWS" \
        --task-ids "$T" --tables "$RR/tables/calib10_loose" --old-tables "$D13/reserve" \
        --out "$RR/analysis/2a_replay_$run" --workers 12 > "$RR/logs/2a_replay_$run.log" 2>&1
done
stamp "D done"

stamp "E start"
$PY -m reward.check_2a --rows "$ROWS" --task-ids "$T" --new "$RR/tables/calib10_loose" --old "$D13/reserve" \
    --goal "$D13/reserve_goal" --outside "$RR/analysis/2a_replay_q3d/outside_old.jsonl" \
    "$RR/analysis/2a_replay_q3s0/outside_old.jsonl" --gs-root "$GS" --out "$RR/analysis/2a_check.json" \
    --per-task 5 --workers 12 > "$RR/logs/2a_check.log" 2>&1
stamp "E done"
date -u +%FT%TZ > "$RR/logs/2a_done"
