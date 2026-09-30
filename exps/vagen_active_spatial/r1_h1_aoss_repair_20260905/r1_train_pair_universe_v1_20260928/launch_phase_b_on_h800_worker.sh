#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905
CODE=${BASE}/r1_train_pair_universe_v1_20260928/code_bf14b881_source_only.tar.gz
RUN=${BASE}/r1_fullscale_canonical_expansion_phase_b_v2_20260930
test -s ${BASE}/r1_fullscale_canonical_expansion_phase_b_scope_v2_20260930/fresh_sources.jsonl
mkdir -p ${RUN}
exec /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_run_as_artifact_owner.sh env \
  R1_FRESH_EXPANSION_CODE_ARCHIVE=${CODE} \
  R1_FRESH_EXPANSION_CODE_ARCHIVE_SHA256=20db60ccdcefa7146e99032da0c1baecfffe212fc232efb8dfbd50f9cff6a432 \
  R1_FRESH_EXPANSION_COMMIT=bf14b881b4cb3f9c76374dc86fb82331249298dc \
  R1_FRESH_EXPANSION_SCOPE=${BASE}/r1_fullscale_canonical_expansion_phase_b_scope_v2_20260930 \
  R1_FRESH_EXPANSION_RUN=${RUN} \
  bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_r1_fresh_canonical_expansion_entry.sh
