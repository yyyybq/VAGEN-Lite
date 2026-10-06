# Active Spatial Historical Validity

Date: 2026-08-18

> The validity findings below remain useful, but job/launch status is an August
> snapshot. Later D0-corrected runs and the separate R1 support-subset experiment
> are reconciled in the [2026-10-02 takeover audit](active_spatial_takeover_20261002/REPORT.md).

Gate: **INCONCLUSIVE — clean restart partially launched; Cambrian anchor still pending**

Reason: Qwen v46 real-batch impact is now established, both canonical clean acceptance smokes passed with real rollout and actor update, and the Qwen clean anchor has been launched on `10.119.31.101` using the jumpbox-local renderer. The Cambrian clean anchor was not launched because only one 8-GPU training node was reachable during this run.

## Bug Classes

Cambrian-specific bugs:

- MIV/train-rollout multimodal mismatch before D0.1 fixes.
- SigLIP/SigLIP2 processor and vision-path mismatch before D0.1 fixes.
- Cambrian vision feature layer mismatch before D0.5 fix.
- Cambrian wrapper/runtime mismatches localized through D0.8-D0.11.

Framework-wide PPO/logprob bugs:

- Historical major Qwen v46 resolved configs had `rollout.calculate_log_probs=False`, so vLLM behavior-policy sampled-token logprobs were not saved.
- Non-bypass trainer path recomputes PPO anchor `old_log_probs` with HF/FSDP. Without saved `rollout_log_probs` and enabled `rollout_is`, decoupled correction is not active.
- `bypass_mode=True` makes PPO ratio `exp(HF_current - vLLM_rollout)`. D0.15/D0.16 showed this gives material pre-update ratio drift when rollout and training runtimes differ numerically.
- Actor-side on-policy shortcuts can replace old logprobs with `current_log_prob.detach()` unless `actor.use_rollout_log_probs=True`.

## Historical Validity

| Family | Representative runs | Label | Reason |
|---|---|---:|---|
| Cambrian C1-C7 | C1-C7 scripts | INVALIDATED | Cambrian-specific multimodal train/rollout plumbing was not repaired. Useful as debugging history, not clean Cambrian-S RL evidence. |
| Cambrian C8/B5 | `c8_fwdfirst_rewscale_lfp_server_v19`, `b5_c8_wrapper_img25_actionvalid` | INVALIDATED | D0.1-D0.16 established Cambrian forward mismatches plus incorrect bypass old-logprob handling. |
| Qwen v26-v31 | `v26`-`v31` | UNKNOWN | Shared trainer risk exists, but these individual runs were not real-batch audited. |
| Qwen v42-v45 | `v42`-`v45` | UNKNOWN | Same shared trainer risk; not individually audited. |
| Qwen v46 baselines | `v46_7b_nodelta_w3`, `v46_baseline_qwen25vl_7b`, `v46_baseline_qwen25vl_3b` | AFFECTED | Static configs lacked behavior-logprob correction, and real-batch Qwen v46 audit measured nonzero HF-vLLM off-policy gap. Clean method comparison must be rerun. |
| Qwen v47-v50 | `v47`, `v48`, `v49`, `v50` | UNKNOWN | Shares trainer risk, but no per-run real-batch audit in this pass. |
| A1/A2 derivatives | `a1_v46_spatial_aux_*`, `a2_v46_arrival_stop` | AFFECTED | Derived from v46-style Qwen RL; treat method conclusions as needing clean rerun. |
| Pure pretrained / inference-only eval | Fixed checkpoint evaluation | VALID | PPO old-logprob bugs do not alter standalone evaluation of a fixed checkpoint. |
| D0 diagnostic smokes | D0.12/D0.16/D0.17 smokes | VALID | Valid as diagnostics only, not as full clean RL baselines. |
| SenseNova/U1 and other shared-trainer runs | `u1_*` | UNKNOWN | Needs model-specific audit. |

Historical eval numbers remain the actual measured performance of their checkpoints. The labels above are about clean training-method evidence and paper-level comparisons, not about whether an old checkpoint really produced a logged score.

## Qwen v46 Real-Batch Audit

Artifact:

```text
exps/vagen_active_spatial/d0_17_qwen_v46_logprob_audit_31_101_jumpbox_renderer/d0_17_qwen_audit/d0_16_decoupled_correction_audit_step1.json
```

Same batch definitions:

```text
A = exp(current_HF - rollout_vLLM)
B = exp(current_HF - old_HF)
C = exp(old_HF - rollout_vLLM)
```

| Split | Tokens | A mean abs dlogp | A \|ratio-1\|>5% | B mean abs dlogp | B \|ratio-1\|>5% | C ESS/N | C max |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_response | 609 | 0.019671 | 13.63% | 0.000000 | 0.00% | 0.998301 | 1.306247 |
| think_text | 469 | 0.024597 | 17.48% | 0.000000 | 0.00% | 0.997830 | 1.306247 |
| action_tag+name | 84 | 0.001841 | 0.00% | 0.000000 | 0.00% | 0.999955 | 1.020357 |
| action_name | 36 | 0.004067 | 0.00% | 0.000000 | 0.00% | 0.999898 | 1.020357 |

Identity check:

```text
max_abs_log_error(A, B*C) = 0
```

Interpretation:

- Qwen is not affected by Cambrian-specific MIV/processor/vision-layer bugs.
- Qwen v46 is affected by the shared old-logprob/correction issue as clean PPO method evidence.
- The measured Qwen off-policy gap is much smaller than Cambrian and concentrated in think tokens; action-token drift was clean in this batch.

## Renderer / Node Setup

Jumpbox renderer:

```text
host = 10.119.30.223
bounded smoke port = 8767
detached clean-anchor port = 8768
status = healthy
```

The earlier renderer confusion was a sandbox/device-namespace issue: ordinary sandbox commands did not see `/dev/nvidia*`, but unsandboxed `nvidia-smi` on the jumpbox sees the local H800. Training/audit jobs should run on SSH nodes; renderer should run on the jumpbox HTTP service unless a node-local renderer is intentionally requested.

Reachable training node in this pass:

```text
10.119.31.101
```

Other checked nodes were not reachable at the time of check: `10.119.20.47`, `10.119.16.197`, `10.119.28.235`.

## Canonical Clean Configs

Qwen clean:

- Script: `examples/train/active_spatial/experiments/qwen_active_spatial_clean_v1.sh`
- Base: `examples/train/active_spatial/experiments/v46_baseline_qwen25vl_7b.sh`
- Run name default: `qwen_v46_clean_d0pass_v1`
- Shared PPO fixes only:
  - `actor_rollout_ref.rollout.calculate_log_probs=True`
  - `actor_rollout_ref.rollout.logprob_temperature=1.0`
  - `actor_rollout_ref.actor.use_rollout_log_probs=True`
  - `algorithm.rollout_correction.bypass_mode=False`
  - `algorithm.rollout_correction.rollout_is=token`
  - `algorithm.rollout_correction.rollout_is_threshold=2.0`
  - `algorithm.rollout_correction.rollout_rs=null`
  - `algorithm.rollout_correction.use_policy_gradient=False`

Cambrian clean:

- Script: `examples/train/active_spatial/experiments/cambrian_active_spatial_clean_v1.sh`
- Base: `examples/train/active_spatial/experiments/b5_c8_wrapper_img25_actionvalid.sh`
- Run name default: `cambrian_c8_clean_d0pass_v1`
- Includes landed Cambrian processor/MIV/vision-layer/sync fixes plus the same shared PPO fixes above.

Launch helper:

```text
examples/train/active_spatial/launch_d0pass_clean_anchors.sh
```

## Acceptance Smoke

Qwen smoke:

- Run: `d0_17_qwen_clean_acceptance_smoke_actorupdate_31_101_jumpbox_renderer`
- Status: PASS
- Real rollout: PASS
- Correction: `bypass_mode=False`, `rollout_is=token`, threshold `2.0`
- Actor update: performed
- Grad norm: `7.6293959618`
- Pre-update PPO proximal ratio B: exactly `1.0`
- IS ESS/N: `0.998509` all response, `0.998172` think, `0.999912` action name

Cambrian smoke:

- Run: `d0_17_cambrian_clean_acceptance_smoke_actorupdate_31_101_jumpbox_renderer`
- Status: PASS
- Real rollout: PASS
- Correction: `bypass_mode=False`, `rollout_is=token`, threshold `2.0`
- Actor update: performed
- Grad norm: `8.0798578262`
- Pre-update PPO proximal ratio B: exactly `1.0`
- IS ESS/N: `0.988082` all response, `0.987831` think, `0.998090` action name

Cambrian smoke still showed nontrivial generated-action quality issues in the tiny batch. That is not a plumbing failure, but it should be watched during clean training.

## Clean RL Restart

Launched:

```text
run = qwen_v46_clean_d0pass_20260818_renderer8768
node = 10.119.31.101
renderer = http://10.119.30.223:8768
log = exps/vagen_active_spatial/qwen_v46_clean_d0pass_20260818_renderer8768/train.log
```

Confirmed launch config includes:

```text
rollout.calculate_log_probs=True
rollout.logprob_temperature=1.0
actor.use_rollout_log_probs=True
rollout_correction.bypass_mode=False
rollout_is=token
rollout_is_threshold=2.0
```

Not launched:

```text
cambrian_c8_clean_d0pass_*
```

Reason: no second reachable idle 8-GPU training node during this run. Do not run it concurrently on the same node unless resource pressure is explicitly accepted.

## Next Observation Priority

1. Watch Qwen clean first steps: reward, entropy, KL, PPO ratio, IS ESS/N, action distribution, timeout/collision.
2. Launch Cambrian clean anchor when a second idle 8-GPU node is reachable or after the Qwen anchor is stopped/completed.
3. Compare clean anchors against historical curves only after both runs have matching production-style logs.

### D0.17 Gate

```text
INCONCLUSIVE — Qwen impact audited and Qwen clean anchor launched; Cambrian clean anchor still pending
```
