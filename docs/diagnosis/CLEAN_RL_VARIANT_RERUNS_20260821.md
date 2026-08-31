# Clean RL Variant Reruns (2026-08-21)

## Scope

The historical Active Spatial RL runs used the pre-D0 PPO/logprob plumbing and
cannot be treated as clean method comparisons. The first clean anchors remain:

| Family | SCO job | Run | State at variant submission |
| --- | --- | --- | --- |
| Qwen v46 | `pt-3oav3ziu` | `qwen_v46_clean_d0pass_20260821_full_r6` | RUNNING |
| Cambrian C8/B5 | `pt-bjmg7q4e` | `cambrian_c8_clean_d0pass_20260821_full_r4` | RUNNING |

The first rerun wave covers four orthogonal historical recipe choices. It does
not introduce new hyperparameters or architecture changes.

## Frozen correctness baseline

- Repository base commit: `70d281173fd62a87ad4704d7a663dba7241db647`.
- The mounted working tree contains the existing D0 fixes and clean launch
  files; it was deliberately not reset because these changes are not all in
  the base commit.
- Every variant enables raw T=1 rollout logprobs and framework-native
  decoupled rollout correction:
  - `rollout.calculate_log_probs=True`
  - `rollout.logprob_temperature=1.0`
  - `actor.use_rollout_log_probs=True`
  - `rollout_correction.bypass_mode=False`
  - token IS with threshold `2.0`, RS disabled
  - `rollout_correction.use_policy_gradient=False`
- Renderer: `http://10.119.30.223:8768`. Direct `/health` passed immediately
  before submission. The final single-job service has 32 workers,
  `max_inflight=32`, and a 300-second admission timeout. SCO entries set
  `NO_PROXY`, retry transient render contention, and require a real `/render`
  preflight. The C4 client uses a 900-second request timeout.
- Resource per job: one worker node, 8 H800-80GB, 112 vCPU, 1920 GiB RAM,
  SCO pool `h800`, priority `HIGHEST`.

## First rerun wave

The first submission accidentally targeted the `zoetrope` pool. Those jobs
never received workers (`start_time=null`) and were stopped before resubmission:

`pt-wh7uhqr8`, `pt-p7h7tuel`, `pt-7yjkkmf6`, and `pt-cmork6a2` are all
`SUSPENDED` and contain no training run.

The corrected submissions target the SCO `h800` pool:

| Variant | SCO job | Canonical config | Historical variable isolated | State at 2026-08-21 18:51 UTC |
| --- | --- | --- | --- | --- |
| Qwen v47 clean | `pt-zqsr6wi6` | `examples/train/active_spatial/experiments/qwen_v47_clean_v1.sh` | v46 W3 with KL loss `0.40` instead of `0.30` | RUNNING; renderer health passed |
| Qwen v48 clean | `pt-t0g0yvg3` | `examples/train/active_spatial/experiments/qwen_v48_clean_v1.sh` | v46 with observation window `1` instead of `3` | RUNNING; renderer health passed |
| Qwen v50 clean | `pt-9kz91ia1` | `examples/train/active_spatial/experiments/qwen_v50_clean_v1.sh` | W3, response length `160`, entropy `0.001`, format reward `0.01` | RUNNING; renderer health passed |
| Cambrian C4 clean | `pt-36x6ya7u` | `examples/train/active_spatial/experiments/cambrian_c4_clean_v1.sh` | forward-first prompt with the original high-variance reward | RUNNING r5; step 1 complete |

Run names and persistent run roots are:

| Variant | Run name | Persistent root |
| --- | --- | --- |
| Qwen v47 | `qwen_v47_clean_d0pass_20260821_h800_r1_full` | `exps/vagen_active_spatial/qwen_v47_clean_d0pass_20260821_h800_r1_full/` |
| Qwen v48 | `qwen_v48_clean_d0pass_20260821_h800_r1_full` | `exps/vagen_active_spatial/qwen_v48_clean_d0pass_20260821_h800_r1_full/` |
| Qwen v50 | `qwen_v50_clean_d0pass_20260821_h800_r1_full` | `exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260821_h800_r1_full/` |
| Cambrian C4 | `cambrian_c4_clean_d0pass_20260822_h800_r5_full` | `exps/vagen_active_spatial/cambrian_c4_clean_d0pass_20260822_h800_r5_full/` |

## Recipe fidelity

Qwen v47 and v48 retain the historical Qwen2.5-VL-7B recipe: actor LR
`5e-7`, 700 steps, train batch 12, PPO minibatch 8, entropy `0.005`, 12 turns,
response length 384, and save/eval cadence 50. Their only recipe-level changes
relative to v46 are respectively KL loss and window size.

Qwen v50 retains actor LR `5e-7`, KL loss `0.30`, W3, 12 turns, train batch
12, PPO minibatch 8, and 700 steps. Its historical stability changes are kept
together: response length 160, entropy `0.001`, and the format-0.01 environment.

Cambrian C4 retains the historical forward-first prompt, actor LR `5e-7`, KL
loss `0.20`, entropy `0.005`, W1, 12 turns, response length 384, train batch 8,
and 2000-step target. The infrastructure is adapted from four to eight GPUs;
the PPO minibatch is correspondingly 8 so the per-rank sample topology remains
one. It uses the current Cambrian-S-7B-LFP checkpoint path and all landed
Cambrian D0 correctness fixes.

## Cambrian C4 r1 restart

The first H800 C4 job, `pt-qbnzeipy`, passed renderer preflight and its complete
one-step acceptance smoke. The pre-update proximal ratio was exactly 1, token
IS ESS/N was approximately 0.987, `A = B * C` held, and the actor update
completed. The subsequent full run failed before step 1 while creating the
vLLM KV cache: the historical `gpu_memory_utilization=0.20` combined with
`max_num_batched_tokens=18000` left no cache blocks after model/activation
allocation.

The r2 restart changes only the rollout runtime memory budget to the value
already proven by the running Cambrian C8 clean anchor:
`gpu_memory_utilization=0.35`. Training LR, KL, entropy, reward, prompt,
batching, response length, and architecture are unchanged. The replacement is
`pt-teu1jrqt` / `cambrian_c4_clean_d0pass_20260821_h800_r2_full`.

The apparent r2 idle period was not a model or collective hang. Its renderer
preflight spent about 7,384 seconds in each of three retry chains because the
old jumpbox render service admitted disconnected requests into its process-pool
queue. It exited with status 5 after only one successful transition. The
renderer was restarted with bounded admission, and new real renders returned.

The r3 replacement, `pt-5tcfanon`, deliberately reused the already-passed r1
acceptance gate. It proved that `gpu_memory_utilization=0.35` fixes the vLLM
startup issue: all four HTTP servers started and validation began. Validation
then exposed a separate launch-path bug. Historical C4 still referenced its
old `/scratch/by2593/...` JSONLs, but SCO mounts only `/mnt/umm`. ActiveSpatial
silently fell back to synthetic `scene_test` when the JSONL was absent, and the
real renderer correctly returned HTTP 500 because no such PLY exists.

The r4 replacement, `pt-scl78yx1`, uses the new SCO path-only environment
config `env_config_v24_100scenes_fwdfirst_mnt.yaml`. It keeps the historical
reward, prompt, and split sizes unchanged while mapping the training data, OOD
data, and InteriorGS root to `/mnt/umm`. `run_experiment.sh` now fails before
model startup if the primary or configured OOD JSONL is missing. The generated
r4 YAMLs contain the real paths; startup reported train dataloader size 1398
and validation dataloader size 10 rather than the zero-line mock fallback.
All four vLLM servers then started with the corrected 0.35 memory budget.
Step-0 validation completed all 10 batches with zero environment exceptions,
wrote `validation/0.jsonl`, and entered the 2000-step training loop. During
that validation the r4 node received 1,776 successful real render responses;
bounded 503 responses were initially retried rather than entering the renderer
queue. The first training rollout then exhausted those retries. r4 failed at
03:29 UTC with `ActiveSpatial rendering failed: render server busy`; it never
completed training step 1.

The service inspection after r4 found 32 orphaned ProcessPool workers from an
older renderer restart plus the four workers owned by the active service. The
orphans held about 40 GiB of H800 memory, while the four-worker service exposed
only four admission slots to 32 concurrent rollout trajectories. The launcher
now `exec`s the Python service so terminating the service does not strand its
children. After explicit cleanup, the renderer was restarted with 32 workers
and 32 admission slots. A same-scene 32-way `reset + step` probe passed 32/32
in 34.7 seconds. A second probe spanning 29 real GS scenes passed 32/32 with
median 69.0 seconds, p95 90.7 seconds, maximum 125.6 seconds, zero 503s, and a
32.0 GiB renderer GPU-memory peak.

The r5 replacement is `pt-36x6ya7u` /
`cambrian_c4_clean_d0pass_20260822_h800_r5_full`. It uses one 8-H800 node and
the unchanged C4 clean recipe. All four vLLM servers started with
`gpu_memory_utilization=0.35`; step-0 validation completed and wrote a 10.98 MB
`validation/0.jsonl`. Training step 1 completed in 133.6 seconds. By that point
the r5 node had received 2,121 real renderer HTTP 200 responses and zero 503s,
and `transition/env_exception_rate` was zero. Token rollout IS ESS/N was
0.9819. The historical C4 `critic_warmup=60` correctly skipped the actor update
at step 1 while completing the critic update; this is expected recipe behavior,
not an update failure.

## Acceptance gate

Each job first runs the established production-topology one-step smoke. Long
training starts only after all of the following pass:

- renderer health and one real render;
- pre-update HF proximal ratio within the clean gate;
- `A = B * C` for rollout correction;
- token IS ESS/N at least 0.80;
- finite loss/gradient and a real actor update;
- actor-to-vLLM sync returns.

At 17:45 UTC, all four corrected jobs had `resource_pool.name=h800`, one active
8-H800 replica, persistent run directories, and successful renderer `/health`
responses. The real-render and one-step acceptance gates were still in
progress; no long-training step is claimed yet.

## Deferred second wave

Do not launch the second wave until renderer load and first-wave acceptance are
visible. The next highest-value candidates are Qwen v49 or the v26-style
stability recipe, Cambrian C7 (`entropy=0.01`), and Cambrian A1/A2. They remain
useful, but are less orthogonal than the four first-wave controls and should not
compete with acceptance diagnosis if the shared renderer saturates.
