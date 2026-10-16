# Evaluation Checkpoint Recovery Audit (2026-08-31)

## Scope

The failed evaluation recovery jobs targeted Qwen v50 clean, Cambrian C8
clean, and Cambrian C4 clean.

## Initial Finding

The selected checkpoint directories initially contained only empty
`actor/huggingface` and `critic/huggingface` directories:

| Run | Requested steps | Current source state |
| --- | --- | --- |
| `qwen_v50_clean_d0pass_20260824_full` | 50, 100, 700 | no HF weights and no FSDP shards |
| `cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full` | 50, 100, 1000 | no HF weights and no FSDP shards |
| `cambrian_c4_clean_d0pass_20260822_h800_r5_full` | 960 | no HF weights and no FSDP shards |

The training logs show that exports existed during training. However, the
resolved configs had `trainer.max_actor_ckpt_to_keep=1`; checkpoint rotation
removed older complete checkpoints from the mounted run directories. The
configured `default_hdfs_dir` and `huggingface_hub.hf_save_freq` were null.

## AOSS Recovery

The local/HDFS conclusion did not include the separate AOSS archive. The
archive loop recorded successful uploads for every requested checkpoint:

| Run | Recovered steps | AOSS actor export status |
| --- | --- | --- |
| `qwen_v50_clean_d0pass_20260824_full` | 50, 100, 700 | restored and loadable |
| `cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full` | 50, 100, 1000 | restored and loadable |
| `cambrian_c4_clean_d0pass_20260822_h800_r5_full` | 960 | restored and loadable |

`scripts/restore_active_spatial_checkpoints_aoss.sh` restores only
`actor/huggingface` from the AOSS archive into each original checkpoint
directory. This is sufficient for inference evaluation and deliberately does
not restore FSDP, optimizer, critic, or other training-state files.

## Repairs

- `evaluation/model_agent.py` now detects Cambrian from the requested
  checkpoint path before requiring `config.json`, so a Cambrian path cannot
  silently follow the Qwen route.
- `examples/train/active_spatial/sco_active_spatial_eval_entry.sh` explicitly
  exports `CAMBRIAN_SRC` and adds it to `PYTHONPATH`, covering both navigation
  vLLM and EASI's `cambrians` backend.
- `scripts/active_spatial_eval_sweep.py` accepts only a complete HF export:
  `config.json` plus model weights or a model index.  Empty checkpoint shells
  are no longer discoverable as evaluation candidates.
- `examples/train/active_spatial/submit_clean_checkpoint_evals_sco.sh` checks
  the same condition before an H800 job is submitted.
- Canonical Qwen v50, Cambrian C8, and Cambrian C4 clean configs now retain 20
  complete checkpoints instead of the inherited one-checkpoint default.

## Verification

- Before recovery, the submission preflight correctly rejected empty HF
  export shells rather than allocating an H800 worker.
- After recovery, all seven target directories contain `config.json` and at
  least one safetensors model shard; they pass the same submission preflight.
- The evaluation checkpoint resolver now ignores incomplete shells and picks
  the restored HF export.
