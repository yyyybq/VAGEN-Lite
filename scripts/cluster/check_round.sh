#!/usr/bin/env bash
# One equivalence round of the cleanup (CPU only, login node, about 15 minutes):
#   ROOT=<checkout> OUT=<new dir> bash scripts/cluster/check_round.sh
# Writes OUT/{suiteA.xml,suiteA.log,suiteB.xml,suiteB.log,golden/,commit.txt}. Suite A = every test root
# (TEST_PATHS, default: the roots of the cleaned layout that exist in ROOT); suite B = scripts/check_active_spatial_cpu.sh.
# Golden outputs come from $TOOLS/golden_outputs.sh (default: the directory of this script).
# Compare two rounds with compare_round.py.
set -uo pipefail
TOOLS=${TOOLS:-$(cd "$(dirname "$0")" && pwd)}   # golden_outputs.sh / compare scripts (the same tools for both rounds)
ROOT=$(realpath "${ROOT:?checkout}"); OUT=$(realpath -m "${OUT:?output dir}")
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}
mkdir -p "$OUT"; cd "$ROOT"; git rev-parse HEAD > "$OUT/commit.txt"
DEFAULT_PATHS=""
for p in tests data_gen/interiorgs_v10/tests vagen/envs/robocasa/tests data_gen/active_spatial_pipeline/test_enhanced_validation.py; do
  [ -e "$p" ] && DEFAULT_PATHS="$DEFAULT_PATHS $p"
done
TEST_PATHS=${TEST_PATHS:-$DEFAULT_PATHS}
echo "$TEST_PATHS" > "$OUT/test_paths.txt"
export PYTHONPATH=$ROOT:$ROOT/verl CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=2 PYTHON=$PY
( ROOT=$ROOT GOUT=$OUT/golden bash "$TOOLS/golden_outputs.sh" > "$OUT/golden.log" 2>&1 ) &
GPID=$!
$PY -m pytest -q -rs -p no:cacheprovider --junitxml="$OUT/suiteA.xml" $TEST_PATHS > "$OUT/suiteA.log" 2>&1
echo "suite A rc=$?" >> "$OUT/rc.txt"
bash scripts/check_active_spatial_cpu.sh --junitxml="$OUT/suiteB.xml" > "$OUT/suiteB.log" 2>&1
echo "suite B rc=$?" >> "$OUT/rc.txt"
wait $GPID; echo "golden done" >> "$OUT/rc.txt"
tail -n 1 "$OUT/suiteA.log"; tail -n 1 "$OUT/suiteB.log"; tail -n 2 "$OUT/golden.log"
