#!/usr/bin/env bash
set -euo pipefail
export ACT2QA_OUT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_qa/artifacts/act2qa_real_canary_strict_v46_retry_hf_20260930
export ACT2QA_BACKEND=transformers
exec bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_qa/sco_act2qa_strict_v46_infer_entry.sh
