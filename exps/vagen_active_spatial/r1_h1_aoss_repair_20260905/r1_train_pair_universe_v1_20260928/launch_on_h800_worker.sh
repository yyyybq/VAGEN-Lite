#!/usr/bin/env bash
set -euo pipefail
RUN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_train_pair_universe_v1_20260928
# Keep the payload in the SCO foreground process.  The prior nohup/background
# wrapper made the control plane mark a still-running payload as completed.
exec env \
  R1_PAIR_UNIVERSE_CODE_ARCHIVE=${RUN}/code_0b2918df_source_only.tar.gz \
  R1_PAIR_UNIVERSE_CODE_ARCHIVE_SHA256=ab1d674b18862e0cabba0a6c8c0cfe9e065d3786d9fbbe59bebd42837ca20f43 \
  R1_PAIR_UNIVERSE_COMMIT=0b2918df6e748ea29e110d2f79749bde4362a710 \
  bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_r1_pair_universe_phase_b_entry.sh
