#!/usr/bin/env bash
# Submit official StarVLA NavigateKitchen train to SCO H800 (1 node x 8 GPU).
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
SUBMIT_SH="${SUBMIT_SH:-/mnt/umm/users/yinbaiqiao/submit.sh}"
JOB_NAME="${JOB_NAME:-starvla_nk_qwenoft_20261001_h800_8gpu}"
ENTRY="${ENTRY:-$ROOT/examples/train/robocasa/sco_starvla_nk_train_entry.sh}"

bash "${SUBMIT_SH}" h800 1 8 "${JOB_NAME}" "${ENTRY}"
