#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905
SCOPE=${BASE}/r1_phase_a_projective_schema_retry_scope_v1_20260930
[[ $(sha256sum "${SCOPE}/fresh_sources.jsonl" | cut -d ' ' -f 1) == f2335d40d0c73a53b26278ecc7cf0b3762b403d58de0afabb08be18602daab7d ]]
exec /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_run_as_artifact_owner.sh env \
  R1_FRESH_EXPANSION_CODE_ARCHIVE=${BASE}/r1_train_pair_universe_v1_20260928/code_bf14b881_source_only.tar.gz \
  R1_FRESH_EXPANSION_CODE_ARCHIVE_SHA256=20db60ccdcefa7146e99032da0c1baecfffe212fc232efb8dfbd50f9cff6a432 \
  R1_FRESH_EXPANSION_COMMIT=bf14b881b4cb3f9c76374dc86fb82331249298dc \
  R1_FRESH_EXPANSION_SCOPE=${SCOPE} \
  R1_FRESH_EXPANSION_RUN=${BASE}/r1_phase_a_projective_schema_retry_v1_20260930 \
  bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_r1_fresh_canonical_expansion_entry.sh
