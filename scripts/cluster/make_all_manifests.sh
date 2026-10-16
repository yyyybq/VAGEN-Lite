#!/usr/bin/env bash
# Write manifest.json for every fixed-location dataset, model and result entry (cleanup task, 2026-10).
# Run on the cluster; ROOT=<checkout> picks the checkout whose data/releases, models/, results/ get manifests. Idempotent; re-run after new entries are added.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# ROOT: the checkout that holds data/releases, models/ and results/ (default: the repo this script lives in)
ROOT=${ROOT:-$(cd "$HERE/../.." && pwd)}
cd "$ROOT"
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}
mk() { "$PY" "$HERE/make_manifest.py" --repo "$ROOT" "$@" | tail -1; }
# datasets
mk --kind dataset --name interiorgs_v10_tasks_r10 --path data/releases/interiorgs_v10_tasks_r10 --note "round 10 generator (thick walls, generation_r10_20261015.json): formal + reserve tasks, split_manifest, sft frame lists"
mk --kind dataset --name interiorgs_v10_tasks_r6b --path data/releases/interiorgs_v10_tasks_r6b --note "round 6b generator (generation_r6_20261014.json); data of the first SFT/RL (interiorgs_v10_train_20261014)"
mk --kind dataset --name interiorgs_v10_eval_c_278 --path data/releases/interiorgs_v10_eval_c_278 --source exps/vagen_active_spatial/interiorgs_v10_r7_20261014 --note "278 test tasks after the depth check (round 7); results of 10 models in results/interiorgs_v10_r8_eval2"
# models
mk --kind model --name interiorgs_v10_sft_r6b_qwen25vl7b --path models/interiorgs_v10_sft_r6b_qwen25vl7b --source exps/vagen_active_spatial/interiorgs_v10_train_20261014 --note "LLaMA-Factory full SFT of Qwen2.5-VL-7B on r6b train trajectories, 3 epochs (checkpoint-163/326/489); final = checkpoint-489"
for s in 50 100 150 175 200 225 250; do
  mk --kind model --name interiorgs_v10_rl_r6b_qwen25vl7b_step$s --path models/interiorgs_v10_rl_r6b_qwen25vl7b_step$s --source exps/vagen_active_spatial/interiorgs_v10_train_20261014/rl_run --note "PPO (no_concat_gae, S0 sparse) from the SFT model; HF actor export of global_step_$s; step 50 = SFT (critic warmup 60)"
done
mk --kind model --name interiorgs_v10_sft_r5_qwen25vl3b --path models/interiorgs_v10_sft_r5_qwen25vl3b --source exps/vagen_active_spatial/interiorgs_v10_r5_20261014/trial --note "round 5 one-card trial: Qwen2.5-VL-3B SFT; used by the round-10 one-card RL pilots"
for m in models/historical/*/; do n=$(basename "$m"); mk --kind model --name "$n" --path "$m" --note "historical weight (decision D10); see docs/reports/PHASE1_INVENTORY_20261015.md"; done
for m in models/base/*/; do n=$(basename "$m"); mk --kind model --name "$n" --path "$m" --note "base model (HF hub snapshot)"; done
# results
mk --kind result --name interiorgs_v10_r8_eval2 --path results/interiorgs_v10_r8_eval2 --source exps/vagen_active_spatial/interiorgs_v10_r8_20261014 --note "round 8: all 10 models on the 278-task eval set, with per-step trajectories; table.md"
mk --kind result --name interiorgs_v10_r9_eval_diag --path results/interiorgs_v10_r9_eval_diag --source exps/vagen_active_spatial/interiorgs_v10_r9_20261015 --note "round 9: diagnostic evaluation D1-D7 (rule switches) of SFT epoch 3"
echo "manifests written: $(find data/releases models results -maxdepth 2 -name manifest.json 2>/dev/null | wc -l)"
