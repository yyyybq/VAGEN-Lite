#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
OUT="${ROOT}/exps/vagen_active_spatial/protocol_only_prepost"
cd "${ROOT}"

mkdir -p "${OUT}/sco_submission"
{
  echo "utc_start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname: $(hostname)"
  echo "git_head: $(git rev-parse HEAD)"
  echo "git_head_expected_from_submit_host: 70d281173fd62a87ad4704d7a663dba7241db647"
  echo "verl_head_expected_from_submit_host: 3fe0a29975e1b02ae2bd1dec249f7807dd7966f5"
  echo "git_status_porcelain_sha256: $(git status --porcelain=v1 | sha256sum | awk '{print $1}')"
  echo "protocol_env_sha256: $(sha256sum vagen/envs/active_spatial/env.py | awk '{print $1}')"
  echo "agent_loop_sha256: $(sha256sum vagen/agent_loop/gym_agent_loop_no_concat.py | awk '{print $1}')"
  echo "trainer_sha256: $(sha256sum vagen/ray_trainer.py | awk '{print $1}')"
  echo "dp_actor_sha256: $(sha256sum verl/verl/workers/actor/dp_actor.py | awk '{print $1}')"
  echo "fsdp_utils_sha256: $(sha256sum verl/verl/utils/fsdp_utils.py | awk '{print $1}')"
  echo "experiment_config_sha256: $(sha256sum examples/train/active_spatial/experiments/u1_protocol_only_prepost_32.sh | awk '{print $1}')"
  echo "training_data_sha256: $(sha256sum data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types_clean_layout_render.jsonl | awk '{print $1}')"
  echo "fixed_manifest_sha256: $(sha256sum exps/vagen_active_spatial/protocol_only_prepost/protocol_eval_manifest.jsonl | awk '{print $1}')"
  echo "base_model_config_sha256: $(sha256sum /mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT/config.json | awk '{print $1}')"
  echo "base_model_index_sha256: $(sha256sum /mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT/model.safetensors.index.json | awk '{print $1}')"
  echo "base_model_identity_sha256: 81db486b9984df5a7ce1a1fbf75099589dc640e313a3da157b128440af24f8c6"
  echo "fm_backprop: 0"
} > "${OUT}/sco_submission/runtime_identity.yaml"

exec bash examples/train/active_spatial/experiments/u1_protocol_only_prepost_32_launch.sh
