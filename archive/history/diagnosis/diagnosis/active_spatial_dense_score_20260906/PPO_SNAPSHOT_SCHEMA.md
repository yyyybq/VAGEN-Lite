# Exact PPO batch snapshot schema

Status: `PASS` for writer/replayer synthetic fixtures; `BLOCKED_REAL_BATCH` for the existing archived v50 JSONL.

The format is `active_spatial_ppo_batch_snapshot_v1`. One immutable directory contains:

- `manifest.json`: schema version, SHA256 values, resolved-config identity, actor/critic/reference checkpoint path+content SHA256+step, field contract, and capture identity.
- `resolved_config.json`: the full resolved configuration, canonical JSON, SHA256 bound by the manifest.
- `payload.pt`: CPU tensors and plain metadata. Its SHA256 is bound by the manifest.

The writer refuses incomplete batches, an existing target, a missing config/checkpoint binding, a missing reward trace, missing terminations, or non-serializable metadata. `load_snapshot` verifies version and both hashes before returning data.

## Field contract

| Classification | Saved / reconstructed fields |
| --- | --- |
| Required for training replay | `input_ids`, `responses`, `attention_mask`, `response_mask`, `position_ids`, `old_log_probs`, `values`, `token_level_scores`, `token_level_rewards`, `advantages`, `returns`, `value_mask`; processed `multi_modal_inputs`; `group_idx`, `traj_idx`, `turn_idx`, `__last_turn__`; reward trace; terminations; immutable actor/critic/reference identities; resolved config. `ref_log_prob` and rollout logprobs are saved whenever present; the replayer fails if explicit KL is configured but reference logprobs are missing. |
| Deterministically reconstructed | `optimization_mask` is copied from the actual response mask; turn-token map derives from response mask and group/traj/turn IDs; historical total reward derives from the trace; raw historical GAE derives from scores/values/masks/IDs. Both reconstructed forms are retained/rechecked. |
| Audit metadata | task/episode/family IDs, action text, parsed primitive bundle, rollout/logprob temperatures, RNG/sampler identities, actual termination reason, and bootstrap context. |

`bootstrap_context` records raw next-observation text plus mode/size/pixel bytes for next images only while snapshot capture is armed. This is an immutable source for a later critic preprocessing/value pass; it is not silently treated as an already-computed bootstrap value.

## Capture boundary and default behavior

The hook is [ray_trainer.py](/mnt/umm/users/yinbaiqiao/VAGEN-Lite/vagen/ray_trainer.py:2798), after no-concat GAE, value-mask construction, and optional filter/balance, but before `update_critic` and `update_actor`. This captures the actual update batch, including post-filter duplicate padding if any. It is deliberately later than reward scoring and old-logprob/value computation.

It is disabled unless `VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR` is nonempty. In the disabled state it performs no validation, serialization, model forward, or filesystem write. When enabled, actor/critic/reference checkpoint paths and precomputed content SHA256 values are mandatory; the hook fails closed rather than binding a batch to a model name alone.

The action and bootstrap metadata is propagated by [gym_agent_loop_no_concat.py](/mnt/umm/users/yinbaiqiao/VAGEN-Lite/vagen/agent_loop/gym_agent_loop_no_concat.py:403). It changes neither reward, stopping, action parsing, scorer, nor PPO math.

## Offline replay contract

[active_spatial_offline_ppo_replay.py](/mnt/umm/users/yinbaiqiao/VAGEN-Lite/scripts/active_spatial_offline_ppo_replay.py) verifies the snapshot then recomputes historical/S0/S1/S5 reward and exact historical no-concat GAE. It reports raw advantage, independently whitened advantage, and fixed-historical-reference-whitened advantage. `--backward` additionally requires an explicit immutable-checkpoint adapter (`MODULE:FACTORY`) returning actor/critic and pure forward callables. It has no optimizer construction or `optimizer.step` path.

The adapter must load exactly the actor/critic/reference material identified in `manifest.json`; otherwise its result is `BLOCKED`, not a gradient result.
