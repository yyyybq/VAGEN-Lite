# Active Spatial dense-score pilot continuation

Status: `PASS_REAL_R5_BACKWARD`; `PASS_FINAL_STATIC_PREFLIGHT`; pilot `SUBMITTED_RUNNING`.

This report supersedes the September 29 launch decision only for the newly
bound R1 artifacts.  It does not overwrite the historical blocked report.

## What R1 now clears

The current R1 chain is machine-verifiably complete for entering the
dense-score launch gates:

- task-context one-update acceptance: `PASS`;
- 210/210 live runtime/RGB replay: `PASS`;
- gate-aligned eight-update training: `PASS`;
- fixed Dev32 paired evaluation: `PASS` with
  `expanded_training_allowed=true`.

The eight-update gate used a deliberately small 12-row training subset.  It is
evidence for task-context, PPO and reward/scorer compatibility, not the formal
dense training manifest.  The dense preflight therefore binds the separate
frozen 210-row Projective support set.

## Dense-specific preflight

`preflight_report.json` is `PASS`:

- train: 210 unique rows, 45 scenes, SHA256
  `effcf449a5b3d07850b01a095606fc0d84563c2c920f05ba2be3ff1d9c213b82`;
- development: 32 rows, 7 scenes, SHA256
  `bf87b320df448f4da2aa053b709f4b644c42b49781e4d2155eca812df54e1108`;
- train/development identity, semantic, full-content and scene overlap: all 0;
- initial canonical success: 0/210 train and 0/32 development;
- canonical metric, H1 camera and collision contracts: exact-match `PASS`;
- all four external R1 gates: `PASS`.

The first run failed closed because the old verifier expected obsolete camera
and collision field spellings.  The verifier now reads an exact versioned
contract from the candidate matrix.  It still rejects any mismatch; no
threshold or data gate was relaxed.

## Reward-only invariant

- `S0 -> S1`: only `enable_potential_shaping_reward: false -> true`.
- `S1 -> S5`: near application, visibility application, near bonus `0 -> 0.5`,
  and potential gamma `0.95 -> 0.99`.
- scorer, canonical success, termination, success reward, format reward,
  invalid penalty, model, optimizer and PPO recipe are unchanged.
- format reward is held at the repaired R1 common value `0.0`; it is not
  incorrectly changed to `0.2` in S5.

Resolved configs are in `resolved_configs/`.

## r5 production backward intervention

The 2026-10-06 recheck is still `REAL_HISTORICAL_REPLAY_PASS`.  The local
regression set reports 30 passed tests.  The optimizer-free reward/advantage
replay is also `PASS`, with exact historical no-concat GAE parity (max error
0) and `optimizer_step_called=false`.

The production-compatible adapter runs the original four-way FSDP actor and
critic forward paths and production dual-clip PPO, entropy, explicit low-var
KL, and clipped value-loss operators.  Zoetrope job `pt-egfqnh0t` succeeded.
On the real r5 fixed batch:

- historical no-concat GAE max error: `0`;
- actor old-logprob parity over 18,000 valid tokens: max/mean error `0/0`;
- critic saved-value parity over 18,000 valid tokens: max/mean error `0/0`;
- all S0/S1/S5 branch zero-grad checks: `true`;
- optimizer-step guard calls: `0`; checkpoint writes: `false`;
- actor and critic parameter shard SHA256 digests were unchanged;
- actor S1/S0 gradient cosine `0.9968045600`, difference norm `0.8143024402`;
- critic S1/S0 gradient cosine `0.9999953497`, difference norm `9.8661048879`.

This is fixed-experience update-signal evidence only, not evidence of improved
spatial capability.  The full machine report is
`exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/diagnostic/r5_production_backward.json`.

## Why dense-score depends on R1

The reward ablation does not change R1, but it consumes the task state that R1
defines.  Potential is computed from the canonical score and camera geometry;
success and termination use the same canonical gate; rollout observations use
the H1-rendered RGB and persistent public task context.  If any of these differ
between branches, `S1-S0` is no longer a reward-only intervention.  R1 is
therefore an input-validity gate, not a component of the dense reward itself.

## Final preflight and submissions

`final_static_preflight.json` is `PASS`; all recorded checks are true.  The
frozen configs retain PPO gamma `0.95`, critic warmup `60`, and the 700-step
scheduler horizon.  S0 to S1 has exactly one reward difference: enabling the
potential shaping application.  The jobs are:

- renderer: Zoetrope `pt-vzu8hq6b`, `N4lS.Iq.I80.1`, health `PASS`;
- serial S0/S1/S5 pilot: Zoetrope `pt-f8xpnizj`, `N4lS.Iq.I80.8`, running;
- 150 updates per branch, fresh common step-0 initialization, evaluation at
  0/50/100/150 and checkpoints at 50/100/150.

The renderer is a separate job and is not embedded in the trainer.

## Commands run

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  scripts/active_spatial_dense_score_preflight.py \
  --matrix examples/train/active_spatial/dense_score_ablation_r1_20261006.yaml \
  --manifest train=exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1/train.jsonl \
  --manifest id=exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1/eval_policy_rows.jsonl \
  --output-dir docs/diagnosis/active_spatial_dense_score_pilot_20261006

CUDA_VISIBLE_DEVICES='' /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  scripts/active_spatial_offline_ppo_replay.py \
  --snapshot exps/vagen_active_spatial/active_spatial_dense_score_diagnostic_real_batch_20260916_r5/snapshot/global_step_000001_pre_update \
  --output docs/diagnosis/active_spatial_dense_score_pilot_20261006/r5_reward_advantage_intervention.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/pytest -q -p no:cacheprovider \
  tests/active_spatial/test_dense_score_preflight.py \
  tests/active_spatial/test_ppo_snapshot_replay.py \
  tests/active_spatial/test_reward_trace.py \
  tests/active_spatial/test_gate_aligned_shaping.py
```

Earlier regression result: `30 passed`, `0 failed`.
Production-adapter safety tests: `4 passed`, `0 failed`.

Additional production commands and results:

```bash
# Real r5 backward (inside Zoetrope 4xH800 job pt-egfqnh0t)
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/active_spatial_r5_production_backward.py \
  --snapshot exps/vagen_active_spatial/active_spatial_dense_score_diagnostic_real_batch_20260916_r5/snapshot/global_step_000001_pre_update \
  --critic-checkpoint exps/vagen_active_spatial/active_spatial_dense_score_diagnostic_real_batch_20260916_r5/critic_initial_pre_update \
  --output exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/diagnostic/r5_production_backward.json

# Final static preflight
python scripts/active_spatial_dense_score_final_preflight.py \
  --run exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007 \
  --package exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/package_main \
  --output exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/final_static_preflight.json
```

Current GPU status: diagnostic `SUCCEEDED`; renderer `RUNNING`; pilot `RUNNING`.
