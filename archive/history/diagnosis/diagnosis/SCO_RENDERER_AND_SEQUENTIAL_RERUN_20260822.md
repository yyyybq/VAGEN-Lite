# SCO Renderer and Sequential Clean Reruns (2026-08-22)

## Dedicated renderer

- Canonical SCO job: `pt-w5zt86tx`
- Job name: `active_spatial_renderer_8h800_20260822_r2`
- Resource: one node, eight H800-80GB GPUs
- Worker POD IP: `10.119.18.249`
- Endpoint: `http://10.119.18.249:8768`
- Endpoint file: `exps/vagen_active_spatial/sco_renderer/active_spatial_renderer_8h800_20260822_r2/endpoint.txt`
- Runtime: 32 render workers across GPUs 0-7, 64 admitted requests, 300-second admission timeout
- Shared gsplat cache: `/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer`

The renderer publishes its endpoint only after local health succeeds. The entry
now also performs a real reset/step warmup before endpoint publication on future
service starts. It deletes the endpoint file when the service exits, so training
entries cannot silently use a stale POD address.

Jumpbox-to-SCO real preflight passed after the initial gsplat warmup:

- health: PASS
- valid transitions: 3/3
- actions: move forward, turn left, turn right
- retries/timeouts: 0/0
- warm request latency: about 0.03-0.05 seconds
- first unseen-scene load: about 10 seconds

Artifact:

`exps/vagen_active_spatial/sco_renderer/active_spatial_renderer_8h800_20260822_r1/jumpbox_real_preflight_r2/phase3_data_gate_audit.json`

The original renderer job `pt-l48bh4le` proved SCO-to-SCO serving and passed
the Qwen acceptance workload, but GPU inspection found that all spawned workers
were bound to physical GPU 0. The cause was changing `CUDA_VISIBLE_DEVICES`
inside a spawned process after CUDA initialization and then selecting logical
device 0. It is retained temporarily only because stopping a running job needs
explicit authorization; it is not the canonical eight-GPU service.

The r2 launcher passes logical IDs 0-7 after applying the parent visibility
mask. Each worker now calls `torch.cuda.set_device(gpu_id)` and gives the same
device to `GaussianRenderer`. A nine-unique-scene probe verified resident render
processes on every physical GPU:

| GPU | Resident memory after probe |
|---:|---:|
| 0 | 2605 MiB (two scene workers) |
| 1 | 1189 MiB |
| 2 | 1233 MiB |
| 3 | 1295 MiB |
| 4 | 1249 MiB |
| 5 | 1197 MiB |
| 6 | 1199 MiB |
| 7 | 1193 MiB |

## Sequential training queue

Only one full training run should consume this renderer at a time. Do not submit
the queue as concurrent SCO jobs.

| Order | Run | Status |
|---:|---|---|
| 1 | `qwen_v46_clean_d0pass_20260822_sco_renderer_r1_full` | `pt-5htwavlt`; acceptance PASS, full run initializing against noncanonical r1 renderer; no full-run step/checkpoint at migration decision |
| 2 | clean Cambrian C8 replacement | pending v46 completion |
| 3 | clean Qwen v47 replacement | pending |
| 4 | clean Qwen v48 replacement | pending |
| 5 | clean Qwen v50 replacement | pending |

Cambrian C4 r5 (`pt-36x6ya7u`) remains a separate active run using the jumpbox
renderer. It was not stopped or replaced by this migration.

Qwen v46 acceptance evidence:

- training-node renderer preflight: PASS, 3/3 valid transitions, no retry/timeout
- actor update: performed
- proximal ratio B: exactly 1 before update
- `A = B * C` maximum log error: 0
- rollout IS ESS/N: 0.99836
- rollout IS range: 0.8144-1.2957, no high/low clipping
- finite actor loss and gradient norm
- eight trajectories, zero invalid actions

The SCO API exposed no dependency flag. Consequently, later jobs are submitted
only after the active full run exits; submitting all queue entries now would let
the scheduler run them concurrently and recreate renderer contention.

## Training endpoint discovery

Training entries may set:

```bash
ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/sco_renderer/active_spatial_renderer_8h800_20260822_r2/endpoint.txt
```

The shared SCO entry waits for this file, validates its URL, exports the resolved
host/port/protocol, and then requires `/health`, real renderer preflight, and the
one-step PPO acceptance gate before launching a full run.
