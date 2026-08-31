#!/usr/bin/env bash
# Resume the D0-correct v48 clean run from its latest complete checkpoint.
source "$(dirname "${BASH_SOURCE[0]}")/qwen_v48_clean_v1.sh"

RESUME_MODE="auto"
VAL_BEFORE_TRAIN="False"
