#!/usr/bin/env bash
# Guarded reward-only candidate.  This file is inert unless all three values
# below are supplied by active_spatial_dense_score_launch_gate.py.
source "$(dirname "${BASH_SOURCE[0]}")/v50_7b_w3_format001_stable.sh"

: "${DENSE_SCORE_PREFLIGHT_REPORT:?launch gate must provide a PASS preflight report}"
: "${DENSE_SCORE_VARIANT:?launch gate must provide S0, S1, or S5}"
: "${DENSE_SCORE_ENV_CONFIG:?launch gate must provide the materialized environment config}"

"${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}" - "$DENSE_SCORE_PREFLIGHT_REPORT" "$DENSE_SCORE_VARIANT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
variant = sys.argv[2]
if report.get("status") != "PASS":
    raise SystemExit(f"BLOCKED: preflight status={report.get('status')}")
if variant not in {"S0", "S1", "S5"}:
    raise SystemExit(f"BLOCKED: unknown variant={variant}")
if report.get("config_gate", {}).get("status") != "PASS" or report.get("data_gate") != "PASS":
    raise SystemExit("BLOCKED: config_gate and data_gate must both PASS")
PY

ENV_CONFIG="$DENSE_SCORE_ENV_CONFIG"
EXPERIMENT_NAME="${DENSE_SCORE_EXPERIMENT_NAME:-dense_score_${DENSE_SCORE_VARIANT,,}_pilot}"
RESUME_MODE="disable"

# The gate exports explicit manifests and counts; no slicing/refill is allowed.
export ID_VAL_JSONL="${DENSE_SCORE_ID_MANIFEST:?missing explicit ID manifest}"
export ID_VAL_N_ENVS="${DENSE_SCORE_ID_COUNT:?missing explicit ID count}"
export OOD_VAL_JSONL="${DENSE_SCORE_OOD_MANIFEST:?missing explicit OOD manifest}"
export OOD_VAL_N_ENVS="${DENSE_SCORE_OOD_COUNT:?missing explicit OOD count}"
export TRAIN_EXCLUDE_TASK_TYPES=""
export ID_VAL_EXCLUDE_TASK_TYPES=""
