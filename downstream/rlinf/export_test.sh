#!/usr/bin/env bash
# Export test for step 4 (the user's decision 5 of 2026-10-22): bring an RLinf actor back to the StarVLA format with
# scripts/export_rlinf_starvla.py and evaluate the exported policy with the downstream evaluation, to show that the
# RL results can be evaluated with the same 950-episode protocol.  By default the actor is the RLinf smoke's (2 PPO
# updates from STEP2-RS-s1's final checkpoint) and the evaluation is one shard of the fixed plan (EVAL_SHARD, 1/40:
# about 24 episodes) with 8 policy servers x 24 simulator processes: enough to see the server load the export and
# episodes run, not a result.
# Writes OUT/export/ (checkpoints/policy.pt + config, about 16 GB), OUT/eval/out/, OUT/export_test.json.
# Usage: export_test.sh OUT [RLINF_WEIGHTS SOURCE_STARVLA_PT]   (OUT/export must not exist)
set -uo pipefail
OUT="${1:?Usage: export_test.sh OUT [RLINF_WEIGHTS SOURCE_PT]}"
RUNS=/mnt/umm/users/yinbaiqiao/downstream_runs/experiments
WEIGHTS="${2:-$RUNS/RLINF-SMOKE-1/cfg/logs/robocasa365_starvla_ppo/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt}"
SOURCE="${3:-$RUNS/STEP2-RS-s1/run/final_model/pytorch_model.pt}"
EVAL_SHARD="${EVAL_SHARD:-1/40}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
mkdir -p "$OUT"
log() { echo "[export-test] $* $(date -u +%FT%TZ)" | tee -a "$OUT/export_test.log"; }
log "weights=$WEIGHTS source=$SOURCE"
t0=$(date +%s)
(
  source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
  set +e
  cd "$MAIN"
  python scripts/export_rlinf_starvla.py --weights "$WEIGHTS" --source-ckpt "$SOURCE" --out "$OUT/export"
) >> "$OUT/export_test.log" 2>&1
rc=$?
export_s=$(( $(date +%s) - t0 ))
log "export rc=$rc ${export_s}s; $(du -sh "$OUT/export" 2>/dev/null | cut -f1)"
eval_rc=-1
if [[ $rc -eq 0 ]]; then
  t0=$(date +%s)
  bash "$WT/downstream/eval/run_eval_ckpt.sh" "$OUT/export/checkpoints/policy.pt" "$OUT/eval/out" 8 24 "$EVAL_SHARD" \
    >> "$OUT/export_test.log" 2>&1
  eval_rc=$?
  log "evaluation rc=$eval_rc $(( $(date +%s) - t0 ))s"
fi
python3 - "$OUT" "$rc" "$eval_rc" "$export_s" <<'EOF'
import json, sys
from pathlib import Path
out, rc, eval_rc, secs = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
info = out / "export/rlinf_export.json"
summary = out / "eval/out/episodes/summary.json"
res = {"export_rc": rc, "export_seconds": secs, "eval_rc": eval_rc,
       "export": json.loads(info.read_text()) if info.is_file() else None,
       "eval": {k: v for k, v in json.loads(summary.read_text()).items() if k != "per_task"} if summary.is_file() else None}
res["ok"] = rc == 0 and eval_rc == 0 and bool(res["eval"]) and res["eval"].get("complete", 0) > 0
(out / "export_test.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("export_rc", "eval_rc", "ok")}))
EOF
