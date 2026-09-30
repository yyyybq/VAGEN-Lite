#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905
CODE=${BASE}/r1_train_pair_universe_v1_20260928/code_0b2918df_source_only.tar.gz
RUN=${BASE}/r1_fullscale_canonical_expansion_phase_b_v1_20260928
test -s ${BASE}/r1_fullscale_canonical_expansion_phase_b_scope_v1_20260928/fresh_sources.jsonl
mkdir -p ${RUN}
exec /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_run_as_artifact_owner.sh env \
  R1_FRESH_EXPANSION_CODE_ARCHIVE=${CODE} \
  R1_FRESH_EXPANSION_CODE_ARCHIVE_SHA256=ab1d674b18862e0cabba0a6c8c0cfe9e065d3786d9fbbe59bebd42837ca20f43 \
  R1_FRESH_EXPANSION_COMMIT=0b2918df6e748ea29e110d2f79749bde4362a710 \
  R1_FRESH_EXPANSION_SCOPE=${BASE}/r1_fullscale_canonical_expansion_phase_b_scope_v1_20260928 \
  R1_FRESH_EXPANSION_RUN=${RUN} \
  bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_r1_fresh_canonical_expansion_entry.sh
