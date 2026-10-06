# Active Spatial dense-score reward-only ablation handoff

Date: 2026-09-06  
Repository: `/mnt/umm/users/yinbaiqiao/VAGEN-Lite`  
Scope: audit, reward componentization, tests, data/config preflight, and historical-batch eligibility only.

## Outcome

| deliverable | status | result |
|---|---|---|
| Actual training/reward/GAE/optimizer audit | PASS | Audited current files and the clean v50 resolved run. |
| Reward-only switches and per-turn component ledger | PASS | Defaults preserve the old formulas; scorer/success/termination remain independent. |
| Reward behavior tests | FIXTURE_ONLY | 12 tests passed; no renderer was contacted. |
| S0/S1/S5 full resolved configs and diff gate | PASS | S1−S0 changes only the potential reward switch. |
| Formal split/canonical/data preflight | BLOCKED | Historical manifests are not eligible formal H1 data. |
| Real fixed-batch gradient diagnosis | BLOCKED_REAL_BATCH | Required batch tensors, reward terms, critic state, and identities are absent. |
| Pilot/training | NOT_RUN | Launch gate refuses the current BLOCKED report. |

No PPO training, SCO submission, renderer rollout/replay, regeneration, optimizer step, checkpoint write, full QA/probe, or CKA was run. Port 8877 and the live R1 repair process were not touched.

## 1. Actual code and resolved training audit

### Entry and resolved identity

The historical reference entry is `examples/train/active_spatial/experiments/qwen_v50_clean_v1.sh`, which sources `v50_7b_w3_format001_stable.sh` and is consumed by `examples/train/active_spatial/run_experiment.sh`. The latter maps `masked_gae` to `no_concat_gae` and launches `python -m vagen.main_ppo` in no-concat multi-turn mode.

Audited resolved config:

`exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/hydra_run/.hydra/config.yaml`  
SHA256: `76215864e7251a1ceb4452522e57f8e72fa6e6d705959fcba795e1397e18a693`

Key resolved settings:

- Actor and critic base: `Qwen/Qwen2.5-VL-7B-Instruct`; LoRA rank 0.
- PPO/GAE: `adv_estimator=no_concat_gae`, `gamma=0.95`, `lambda=0.95`.
- Actor: LR `5e-7`, AdamW weight decay `0.01`, one PPO epoch, entropy coefficient `0.001`, gradient clip `0.3`, explicit KL loss enabled with coefficient `0.3` and `low_var_kl`.
- Critic: LR `2e-5`, weight decay `0.01`, value clip `0.5`, warmup 60; PPO epochs inherit actor's value 1.
- Rollout: `n=4`, temperature `0.8`, rollout logprob temperature `1.0`, rollout logprobs enabled.
- `algorithm.use_kl_in_reward=false`: the adaptive `kl_ctrl.kl_coef=0.002` is not subtracted from token reward in this run. Actor KL loss is a separate loss term.

### Trainable parameters and optimizers

Actor `freeze_vision_tower=false`; critic does not set the option and therefore takes its false default. With LoRA rank 0, no PEFT-only restriction is introduced. The current FSDP workers call `build_optimizer(actor_module_fsdp.parameters(), ...)` and `build_optimizer(critic_module.parameters(), ...)`, so all parameters left with `requires_grad=true` enter their optimizer.

The run log reports the critic as 8,292,170,241 trainable parameters, including a 3,585-parameter value head. Its diagnostic says `vision=0`, but that bucket is based on name substrings such as `vision`; Qwen's module is named `visual`. It is not evidence of a frozen visual tower. Static configuration and the absence of the freeze call show it remains trainable.

### Reward and termination path before this change

The actual v50 environment recipe was:

- scorer enabled, position/orientation base weights `0.7/0.3`, dynamic weighting inside the scorer;
- potential mode `r = 1.0 * (0.99 * phi_t - phi_{t-1})`;
- success threshold `0.65`, auto termination on, success bonus `+5.0`;
- constant near bonus `+0.5` at `phi >= 0.55`;
- visibility checker enabled by dataclass default with scale `0.3`;
- format `+0.01`, invalid format/action `-0.2`, step penalty `-0.01` by default;
- collision, repeated-collision, premature-done, and low-information penalties are additional independent terms.

Reward is computed once per LLM turn after an action bundle, while `_current_step` and `max_episode_steps` count primitive actions. A bundle may contain up to five primitives. Format and collision terms can be applied on the same turn. A primitive-limit terminal turn computes a final score/success but skips `_calculate_pose_reward`, so the old runtime applies no terminal potential/near/visibility/step term on that turn. Reset computes initial phi but does not immediately terminate an initial-success episode.

### Turn reward to token reward and advantage

`gym_agent_loop_no_concat.py` emits one sample per LLM turn with the scalar environment reward. `agent_loop_no_concat.py` places that scalar on the final valid response token in `rm_scores`; all other response-token score entries are zero. Because `use_kl_in_reward=false`, `token_level_rewards` is copied from `token_level_scores`.

`no_concat_gae` then:

1. sums token scores to obtain the turn reward;
2. uses the critic value at the first valid response token as the turn value;
3. groups by `(group_idx, traj_idx)`, sorts by turn, and applies GAE with gamma/lambda `0.95/0.95`;
4. broadcasts each turn advantage to all valid response tokens, supervises the return only at the first valid token, and whitens over unique valid tokens.

The implementation initializes `next_value=0` for every emitted trajectory end and does not consume a terminal/truncation bootstrap mask. Consequently an environment terminal and an outer `max_turns` truncation are both zero-bootstrapped in the actual historical update path.

Matching S1 potential gamma to PPO gamma is necessary but is not enough to assert policy invariance. The current finite/truncated bootstrap, primitive-limit omission of terminal shaping, reset-time initial-success behavior, variable-size primitive bundles under a turn-level discount, and other non-potential reward terms must all be considered. This handoff makes no policy-invariance claim.

## 2. Reward-only implementation

The new switches in `ActiveSpatialEnvConfig` default to true:

- `enable_potential_shaping_reward`
- `enable_near_success_reward`
- `enable_visibility_shaping_reward`

They control only whether a computed shaping term is added. `enable_potential_field`, canonical backend selection, phi updates, success gates, auto termination, and rendering/collision behavior are unchanged. Thus S0 remains on the same scorer and task definition.

Every turn now has an `active_spatial_reward_trace_v1` record with:

- task/episode/turn ID, scene and task family;
- LLM-turn and primitive-action clocks;
- phi previous/current, position/orientation scores, dynamic weights, scorer backend;
- potential, near, visibility, success, format, invalid, collision, repeated collision, premature done, step, low-info, and legacy terms;
- for every term: `raw`, `scale`, `scaled`, `applied`, `enabled`, and event count;
- actual total, reconstructed total, and reconstruction error;
- success, terminated, truncated, and reason;
- an explicit empty clipping list, explicit invalid-reward assignment override, and separately named post-render/termination terms.

The full nested trace is retained in `extra_fields`; flattened scalar/string fields enter `reward_extra_info` and therefore archived per-turn data. The no-concat collector now takes the union of extra-info keys so an exception or mixed environment batch cannot fail merely because the first row has a different instrumentation schema. The outer agent loop explicitly marks `max_llm_turns` as truncation.

Files:

- `vagen/envs/active_spatial/reward_trace.py`
- `vagen/envs/active_spatial/env_config.py`
- `vagen/envs/active_spatial/env.py`
- `vagen/agent_loop/gym_agent_loop_no_concat.py`
- `vagen/agent_loop/agent_loop_no_concat.py`
- `tests/active_spatial/test_reward_trace.py`

## 3. Reward tests

Command:

```bash
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  -m pytest -q -p no:cacheprovider tests/active_spatial/test_reward_trace.py
```

Result: `12 passed` (latest run: `0.99s`).

The tests cover the legacy-v50 default formula, component reconstruction, shaping-off invariance, ordinary progress, near interval, success termination, initial success, primitive timeout, collision/repeated collision, invalid override, post-render low-info, multi-primitive bundles, and visibility-formula parity. Five representative traces had maximum absolute reconstruction error `0.0` at tolerance `1e-12`.

Status is `FIXTURE_ONLY`: these are deterministic environment lifecycle fixtures with mocked score/render/collision dependencies. They establish formula and accounting compatibility, not renderer or training behavior. Machine report: `reward_test_report.json`.

## 4. S0/S1/S5 and resolved-config gate

Source matrix: `examples/train/active_spatial/dense_score_ablation_candidates.yaml`.

- S0 Sparse: potential/near/visibility applied values are zero; the scorer and success bonus remain active. Format, invalid, collision, low-info, and step terms remain common.
- S1 Potential: identical to S0 except potential shaping is applied; potential gamma is `0.95`, equal to actual PPO gamma.
- S5 Legacy-reward reference: potential/near/visibility are applied and potential gamma is the verified historical `0.99`. Near bonus `0.5`, visibility scale `0.3`, success `5.0`, and other common rewards come from the actual baseline config/defaults.

S5 is only the old reward recipe reference. On repaired scorer/data/splits it is not a reproduction of historical training.

Full resolved products:

- `resolved_configs/S0.resolved.json`
- `resolved_configs/S1.resolved.json`
- `resolved_configs/S5.resolved.json`

The full pairwise diff is embedded in `preflight_report.json`. Excluding descriptive labels:

- S0 vs S1: only `environment.enable_potential_shaping_reward: false -> true`.
- S0 vs S5: potential/near/visibility switches plus potential gamma `0.95 -> 0.99`.
- S1 vs S5: near/visibility switches plus potential gamma `0.95 -> 0.99`.

All model, optimizer, rollout, PPO, scorer, success, termination, prompt, action, and renderer settings are identical among candidates.

## 5. Explicit split/canonical preflight

Tool: `scripts/active_spatial_dense_score_preflight.py`.

The run explicitly named all nine available historical source manifests; no slice or refill was used. It hashes every file, counts all rows/families/scenes, computes stable identity plus semantic/full-content fingerprints, applies split-specific scene rules, evaluates initial success with the current scorer, validates canonical/camera/collision fields, reads external gates, writes all three full resolved configs, and checks the reward-only diff invariant.

Result: `BLOCKED`. Report: `preflight_report.json`.

Important findings:

- Actual v50 train has 9,154 rows; its SHA256 is `3421a21ee9ac9c7489f672c3baa82d4c41df7e60adaf400fdef3409e30d45e20`. It is byte-identical to the R1 historical train source used by the preflight.
- Actual v50 ID has 19 rows and SHA256 `b4241abfff7e6fe3692b0bbb60375c7f6bf53a0fe2c6942a157fd80a0401de5f`; all 19 rows are exact train-content overlaps. ID scene sharing itself is allowed; exact sample overlap is the blocker.
- The historical base YAML placed `train_size=11135` and `test_size=22` inside `env_config`, while the launcher reads them from the outer `env1` object and therefore used defaults before later filtering/overrides. Those misplaced keys persisted in the generated effective `train.yaml`; the preflight removes them and requires named manifests.
- Every historical projective/FOV row lacks the required canonical metric, H1 camera, and frozen collision-convention fields. Train has 3,506 such visual rows, all 3,506 failing these version gates.
- Initial-success counts include train `1231/9154`, val-ID `3/19`, ID-test `1006/7790`, and validation proxy `3/25`. Full per-family counts for all OOD files are in the report.
- The umbrella `id_test` source contains exact rows from several OOD diagnostic manifests (for example all 316 `ood_scene` and all 200 `ood_instance` rows), so those files cannot simultaneously serve as disjoint ID/OOD evaluation manifests under this gate.
- The latest checked runtime-canary and projective-observability reports both end in `BLOCKED`; no complete post-repair formal train/ID/OOD manifests exist. Current R1 files/processes belong to the other window and were only observed.

Actual command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  scripts/active_spatial_dense_score_preflight.py \
  --matrix examples/train/active_spatial/dense_score_ablation_candidates.yaml \
  --manifest train=exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl \
  --manifest val_id=exps/vagen_active_spatial/a2_v46_arrival_stop/val_id_delta_boost.jsonl \
  --manifest id_test=data_gen/active_spatial_pipeline/output_100scenes/test.jsonl \
  --manifest validation_proxy=data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl \
  --manifest ood_scene=data_gen/active_spatial_pipeline/ood_splits/ood_scene.jsonl \
  --manifest ood_instance=data_gen/active_spatial_pipeline/ood_splits/ood_instance.jsonl \
  --manifest ood_category=data_gen/active_spatial_pipeline/ood_splits/ood_category.jsonl \
  --manifest ood_template=data_gen/active_spatial_pipeline/ood_splits/ood_template.jsonl \
  --manifest ood_geometry=data_gen/active_spatial_pipeline/ood_splits/ood_geometry.jsonl \
  --output-dir docs/diagnosis/active_spatial_dense_score_20260906
```

Exit code `2` is the designed BLOCKED result.

## 6. Historical rollout / gradient decision

Tool: `scripts/active_spatial_real_batch_preflight.py`.  
Report: `real_batch_preflight.json`.

The scan covered all 700 archived v50 rollout JSONL files and 267,592 rows. They contain text `input`/`output`, aggregate `score`, task identity, and episode-level final statistics. They do not contain the exact multimodal model tensor batch.

Missing requirements:

- exact model input token IDs and generated token IDs;
- images/multimodal input tensors and turn-token correspondence;
- reward components (only aggregate score is present);
- behavior old logprob and critic values;
- response/optimization masks;
- terminated/truncated semantics;
- content identity binding the rollout to a resolved config/checkpoint;
- a critic checkpoint: the inspected global-step-100 actor has seven safetensor shards, but its critic `huggingface` directory has zero files.

Therefore the status is `BLOCKED_REAL_BATCH`. No S0/S1/S5 reward or advantage was guessed, no synthetic fixture is reported as real evidence, no forward/backward was run, and no policy/KL/value loss or gradient norm/angle claim is made.

Actual command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  scripts/active_spatial_real_batch_preflight.py \
  --rollout-dir exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/rollout_data \
  --actor-checkpoint exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/checkpoints/global_step_100/actor/huggingface \
  --critic-checkpoint exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/checkpoints/global_step_100/critic/huggingface \
  --resolved-config exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/hydra_run/.hydra/config.yaml \
  --output docs/diagnosis/active_spatial_dense_score_20260906/real_batch_preflight.json
```

Exit code `2` is the designed `BLOCKED_REAL_BATCH` result.

## 7. Pilot launch gate and stop condition

The guarded candidate experiment is `examples/train/active_spatial/experiments/dense_score_reward_only_pilot.sh`; the generic launcher gained only absolute env-config path support. `scripts/active_spatial_dense_score_launch_gate.py` reads the machine report and refuses unless overall, config, and data gates are all `PASS`. It can materialize an explicit train config and print a command, but never executes it.

The minimum pilot may be considered only after:

1. complete formal train/ID/OOD manifests exist and are supplied explicitly;
2. canonical metric/camera/collision gates pass on every applicable row;
3. sample/fingerprint overlap and split-specific scene rules pass;
4. initial-success is zero (or an explicitly approved, consistently handled policy replaces this gate);
5. external runtime/observability/data gates report PASS;
6. the three newly resolved configs again pass the reward-only diff check;
7. unique experiment names/resources are selected without using or restarting 8877.

Command draft to ask the gate for S1 materialization; do not run until the report is regenerated as PASS:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  scripts/active_spatial_dense_score_launch_gate.py \
  --preflight-report docs/diagnosis/active_spatial_dense_score_20260906/preflight_report.json \
  --variant S1 \
  --experiment-name dense_score_s1_minpilot \
  --output-dir /tmp/dense_score_launch_material \
  --emit-command
```

Against the current report this was tested and returned `BLOCKED: preflight status=BLOCKED` with exit code 2. Training remains `NOT_RUN`.

## Worktree boundary

At the start of this work, R1 donor/allocation/regeneration scripts and a new ID JSONL/sharding scripts were already dirty or untracked. During this work additional R1 files continued changing in another window. None were edited, stashed, reset, deleted, or used as formal repaired data here. This delivery touches only the reward/agent-loop minimum integration, its independent configs/scripts/tests, the absolute-path compatibility line, and this diagnosis directory.

## 8. Exact batch snapshot and offline replay (2026-09-06 follow-up)

| deliverable | status | result |
| --- | --- | --- |
| Exact pre-update snapshot hook | PASS | Disabled by default; captures the post-GAE/post-filter batch before either critic or actor update. |
| Versioned schema and hash validation | PASS | Config, payload, checkpoint content identities, masks, values/logprobs, reward traces, actions and bootstrap source are bound and fail closed. |
| Offline no-concat reward/GAE replay | FIXTURE_ONLY | Historical/S0/S1/S5 reward and both whitening views reconstruct on synthetic batches. |
| Offline backward diagnosis | FIXTURE_ONLY | Separate policy/KL/value loss and clean branch backward are tested without an optimizer. |
| Historical real-batch diagnosis | BLOCKED_REAL_BATCH | Existing archive still lacks the required exact tensors/components/critic identity. |
| Future one-batch capture draft | NOT_RUN | Requires a regenerated formal data gate `PASS`; it stops before either optimizer step. |

Actual PPO route: `agent_loop_no_concat.py` builds `rm_scores` on the final valid response token; `ray_trainer.py` computes/recomputes old logprob, critic values, `token_level_scores`, reward, no-concat GAE and `value_mask`; `ray_trainer.py:2798-2809` is the new exact boundary; then `dp_critic.py` and `dp_actor.py` perform their respective backward/step loops. The hook is after an optional training filter so its snapshot corresponds to the actual update batch, not merely a pre-filter rollout batch.

Files delivered in this follow-up:

- `vagen/utils/active_spatial_ppo_snapshot.py` — schema, strict writer/reader and disabled-by-default hook.
- `vagen/utils/active_spatial_ppo_replay.py` — exact historical no-concat GAE, S0/S1/S5 reward recomposition, whitening views and optimizer-free gradient diagnosis.
- `scripts/active_spatial_offline_ppo_replay.py` — immutable snapshot CLI; backward needs an explicit exact-checkpoint adapter.
- `PPO_SNAPSHOT_SCHEMA.md`, `GAE_BOUNDARY_DECISION.md`, `snapshot_replay_test_report.json`, and `SNAPSHOT_CAPTURE_COMMAND_DRAFT.sh` in this directory.

The snapshot hook requires `VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR` plus immutable actor/critic/reference checkpoint path and SHA256 variables. It is otherwise a no-op. When the optional stop flag is set it returns before `update_critic`/`update_actor`, so it does not call `optimizer.step`.

Latest deterministic command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/active_spatial/test_ppo_snapshot_replay.py \
  tests/active_spatial/test_reward_trace.py
```

Result: `19 passed, 2 warnings, 8.00s`; see `snapshot_replay_test_report.json`. This run uses only a CPU synthetic model/batch. It did not call a renderer, collect a rollout, run training, submit SCO, update a checkpoint, or execute `optimizer.step`.
