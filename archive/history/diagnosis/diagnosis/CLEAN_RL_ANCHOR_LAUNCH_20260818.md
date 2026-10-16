# Clean RL Anchor Launch (2026-08-19 UTC)

## Frozen launch state

- Local repository commit: `70d281173fd62a87ad4704d7a663dba7241db647`.
- The working tree was already dirty and contains the D0 diagnostics/configuration
  artifacts. It was not cleaned or reset. The SCO worker records its own
  `frozen_state.txt`; its mounted source view does not expose Git metadata, so
  worker-side Git provenance is explicitly recorded as unavailable rather than
  allowed to block training.
- SCO workflow reused: `/mnt/umm/users/yinbaiqiao/submit.sh`, using workspace
  `aigc`, `aec2=h800`, `N4lS.Iq.I80.8`, one worker node, the established
  `wc-dev:260617` image, and the existing `/mnt/umm` storage mount.
- Renderer: `http://10.119.30.223:8768` on the jumpbox. Both r5 workers wrote
  a successful `/health` response and real render-preflight image artifacts.

## Submitted anchors

| Anchor | SCO job | Resource | Run name | Canonical config | State at 2026-08-19 03:37 UTC |
| --- | --- | --- | --- | --- | --- |
| Qwen | `pt-4q9e18z4` | 1 node x 8 H800 | `qwen_v46_clean_d0pass_20260819_sco_r7` | `examples/train/active_spatial/experiments/qwen_active_spatial_clean_v1.sh` | STARTING; replacement for worker-terminated r6 |
| Cambrian-S | `pt-q2mugty5` | 1 node x 8 H800 | `cambrian_c8_clean_d0pass_20260819_sco_r6` | `examples/train/active_spatial/experiments/cambrian_active_spatial_clean_v1.sh` | RUNNING; renderer preflight in progress |

Per-run paths:

- Qwen logs: `exps/vagen_active_spatial/qwen_v46_clean_d0pass_20260819_sco_r7/sco/`
- Qwen checkpoints: `exps/vagen_active_spatial/qwen_v46_clean_d0pass_20260819_sco_r7/checkpoints/`
- Cambrian logs: `exps/vagen_active_spatial/cambrian_c8_clean_d0pass_20260819_sco_r6/sco/`
- Cambrian checkpoints: `exps/vagen_active_spatial/cambrian_c8_clean_d0pass_20260819_sco_r6/checkpoints/`

Each `sco/` directory receives `frozen_state.txt`, `renderer_health.json`, the
real-render preflight artifacts, `acceptance_smoke.log`,
`acceptance_gate_check.txt`, and (only after the gate) `train_full.log`.

## Intended differences from historical recipes

Qwen clean v1 inherits `v46_baseline_qwen25vl_7b` and changes only the D0
correctness plumbing: raw T=1 rollout logprobs are retained, HF/FSDP
pre-update logprobs are the PPO proximal anchor, and native detached token IS
rollout correction is enabled (`bypass_mode=False`, threshold `2.0`, RS off).

Cambrian clean v1 inherits `b5_c8_wrapper_img25_actionvalid`; it retains the
already-landed Cambrian processor, native vision path, selected vision layer,
MIV/multimodal merge, temperature, and actor-to-vLLM sync fixes, then applies
the same shared D0 rollout-logprob and decoupled correction settings. Neither
anchor changes LR, KL, reward, prompt, task sampling, or architecture.

## Acceptance contract

Before long training each job must pass: real rollout; pre-update proximal
ratio near one; `A = B * C`; healthy IS ESS/N; finite loss/grad; an actor
update; actor-to-vLLM sync; and stable renderer. The entry stops instead of
starting the full run if any item fails. At the time of this note, this gate
has not yet completed in the SCO jobs.

## Eight-GPU acceptance smoke result

The initial reduced-GPU smoke used batch size 4 against the inherited default
of eight agent-loop workers and was therefore structurally invalid. Bounded
replacement smokes used the production topology (`8` H800 GPUs, batch size
`8`, and `actor_rollout_ref.rollout.agent.num_workers=8`) and stopped before
full training.

| Anchor | SCO job | Result | PPO proximal B | IS ESS/N (all response) | Actor update |
| --- | --- | --- | --- | --- | --- |
| Qwen | `pt-sb8wgm82` | SUCCEEDED | exact 1.0; no token above 5% | 0.99863 | performed |
| Cambrian-S | `pt-4w3de2g6` | SUCCEEDED | exact 1.0; no token above 5% | 0.98954 | performed |

For both jobs the decoupled-correction identity `A = B * C` had zero maximum
log error. Renderer health and real render preflight also passed. These jobs
were smoke-only and did not start the long canonical anchors.

## Formal anchor submission (2026-08-20 UTC)

Following the successful bounded smoke, the two long-running clean anchors
were submitted as separate SCO jobs. Each requests one dedicated eight-H800
node and repeats the same eight-GPU acceptance gate before starting its full
historical recipe.

| Anchor | SCO job | Run name | State at submission |
| --- | --- | --- | --- |
| Qwen | `pt-um1o9e3k` | `qwen_v46_clean_d0pass_20260820_full` | STARTING |
| Cambrian-S | `pt-ormbj287` | `cambrian_c8_clean_d0pass_20260820_full` | STARTING |

## Submission incident

Earlier attempts (`pt-i9oga5x6`, `pt-vw0w23q8`, `pt-blsa7zbu`, `pt-ctz30ki8`,
`pt-10qu1nqh`, `pt-7cqw7yox`, `pt-hamfmlci`, `pt-7t5qj5d6`, `pt-rzgbwvfv`,
`pt-ugm7wlhh`, `pt-b3stwz8o`) are retained and do not contain a full training step. The first
root cause was worker-side provenance capture treating absent worker Git
metadata as fatal under `set -e`. The subsequent r5 smoke reached model
startup and exposed a launcher-only mismatch: its 2/4-GPU smoke left eight
CUDA devices visible, causing the agent loop to split batch size 4 into eight
chunks. r6 constrains visible GPUs only during the smoke, then restores all
eight devices before the full anchor. No training recipe or shared trainer
behavior was altered.

## Renderer retry and submission repair (2026-08-21 UTC)

The first formal attempts, Qwen `pt-um1o9e3k` and Cambrian-S
`pt-ormbj287`, are retained as failed starts. Their common primary error was
`RuntimeError: render server busy`: two eight-worker jobs briefly sent up to
sixteen reset requests to the single jumpbox renderer, which admits eight at a
time and returns HTTP 503 after a five-second admission wait. This is renderer
capacity contention, not a PPO, model, or D0-correctness failure.

For the r2 launch, the HTTP client now accepts an environment-configured
backoff. SCO anchor entries use `INTERIORGS_HTTP_RETRIES=6` and
`INTERIORGS_HTTP_BACKOFF=2`, so transient 503 responses are retried with
exponential backoff. `run_experiment.sh` now uses `pipefail`; an exception from
the Python trainer can no longer be masked by the final `tee` pipeline.

The legacy `/mnt/umm/users/yinbaiqiao/submit.sh` was found to be zero bytes on
2026-08-21. The clean-anchor submitter therefore now directly reuses the
existing verified SCO `jobs create` workflow (`aigc`, H800
`N4lS.Iq.I80.8`, `wc-dev:260617`, and the established `/mnt/umm` mount) and
records returned job IDs. The r2 tasks use fresh names and do not overwrite any
prior run:

| Anchor | SCO job | Run name | State at submission |
| --- | --- | --- | --- |
| Qwen | `pt-flnhy80j` | `qwen_v46_clean_d0pass_20260821_full_r2` | FAILED |
| Cambrian-S | `pt-avtoe5pu` | `cambrian_c8_clean_d0pass_20260821_full_r2` | FAILED |

Both r2 jobs later failed after startup. SCO removed their failed workers and
did not retain their stdout, so their exact terminal traceback cannot be
recovered. The next entry implementation persists `entry.log` and
`exit_status.txt` on the shared mount before renderer preflight. To remove the
remaining single-renderer contention variable, Qwen was restarted alone as
`pt-3sbxrar0` / `qwen_v46_clean_d0pass_20260821_full_r3`; Cambrian is held
until this one-step acceptance gate completes.

At the user's request, Cambrian-S was subsequently launched on the explicit
`zoetrope` pool as `pt-7pl8u5md`, run
`cambrian_c8_clean_d0pass_20260821_full_r3`. It requests one node with eight
H800 GPUs. Renderer health and the real renderer preflight passed; the job is
currently running its one-step acceptance smoke.
