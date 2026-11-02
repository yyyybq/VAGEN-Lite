#!/usr/bin/env bash
# Reward line: start a training chain from this worktree (the main line's R12/scripts/start_chain_dr2.sh with the
# worktree as the code directory, no SFT takeover: the plan names rl_start_model).  VAGEN_LITE_WORKTREE makes the
# chain's sco jobs start here too; the plan's commit becomes the worktree's HEAD, and the worktree must be clean.
# The chain is a resident process on the login node: one thread per library and nice -n 19 (the user's login-node
# rule of 2026-10-28).
#   bash reward/start_chain_b.sh TR/plan_template.json
set -euo pipefail
C=$(cd "$(dirname "$0")/.." && pwd)  # the worktree this script is in (VAGEN-Lite-reward, or -reward-jsrl for direction 2)
cd "$C"
export PYTHONPATH=$C PYTHONDONTWRITEBYTECODE=1 VAGEN_LITE_WORKTREE=$C
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
PLAN_SRC=$1
TR=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['tr'])" "$PLAN_SRC")
mkdir -p "$TR"
HEAD=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no || true)" ]; then
  echo "worktree not clean:"; git status --porcelain --untracked-files=no | head; exit 1
fi
if [ ! -f "$TR/plan.json" ]; then
  sed "s/COMMIT_PLACEHOLDER/$HEAD/" "$PLAN_SRC" > "$TR/plan.json"
fi
$PY - "$TR/plan.json" <<'PY'
import json, sys
from pathlib import Path
p = json.load(open(sys.argv[1]))
assert "PLACEHOLDER" not in json.dumps(p), "a placeholder is left in the plan"
need = [*p["eval_sets"].values(), p["sft_dir"], p["rl"]["rows"], p["rl"]["exclude"], p["rl_start_model"]]
need += [r["rl"]["distance_tables"] for r in p["rl_runs"].values() if (r.get("rl") or {}).get("distance_tables")]
need += [r["rl"]["jsrl_guide"]["tables"] for r in p["rl_runs"].values() if (r.get("rl") or {}).get("jsrl_guide")]
missing = [n for n in need if not Path(n).exists()]
assert not missing, missing
print("plan ok, commit", p["commit"][:8], "job_prefix", p.get("job_prefix"), "pool", p.get("pool"),
      "run_pools", p.get("run_pools"), "runs", {n: (r.get("rl") or {}) for n, r in p["rl_runs"].items()})
PY
setsid nohup nice -n 19 $PY -m data_gen.interiorgs_v10.train_chain run --plan "$TR/plan.json" >> "$TR/chain.log" 2>&1 < /dev/null &
echo "chain pid $!"
sleep 5
$PY -m data_gen.interiorgs_v10.train_chain status --dir "$TR" | head -16
date -u
