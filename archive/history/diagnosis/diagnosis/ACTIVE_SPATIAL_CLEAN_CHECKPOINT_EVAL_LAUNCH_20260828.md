# Active Spatial clean checkpoint evaluation launch (2026-08-28)

Snapshot time: `2026-08-28T11:30:00Z`.

## Scope

Every selected checkpoint is scheduled for the same acceptance contract:

1. ID navigation (`id_test`);
2. all configured OOD navigation suites;
3. the complete EASI-8 static spatial QA suite.

Each SCO job requests one node with eight H800-80GB GPUs. Navigation tasks are
balanced across all GPUs. QA checkpoints are sharded across all GPUs, with one
checkpoint evaluator per GPU.

## Submitted jobs

| Family | SCO job | Checkpoints | Navigation tasks | State at snapshot |
|---|---|---:|---:|---|
| Qwen v46 clean | `pt-riw6dhdp` | 14 (steps 50..700 every 50) | 112 reused from the completed r5 sweep | RUNNING (8 QA workers active) |
| Qwen v47 clean | `pt-w9uvobf6` | 3 (50, 100, 700) | 24 | RUNNING |
| Qwen v48 clean | `pt-c69w6mfw` | 3 (50, 100, 700) | 24 | STARTING |
| Qwen v50 clean | `pt-pcwkumg2` | 3 (50, 100, 700) | 24 | STARTING |
| Cambrian C8 clean | `pt-zc43cclg` | 3 (50, 100, 1000) | 24 | STARTING |
| Cambrian C4 clean | `pt-crjlsdzw` | 1 (960) | 8 | STARTING |

The four `STARTING` jobs have been accepted by SCO and are waiting for node
allocation; they had not produced worker artifacts at the snapshot time.

## Why v46 had no QA result

The successful v46 job `pt-uznwgpew` used the parallel navigation branch in
`sco_active_spatial_eval_entry.sh`. That branch invoked
`active_spatial_eval_parallel.py` and exited immediately after the 112
navigation tasks. It never invoked `easi_eval.py`, so the absence of QA was a
launch-scope omission, not a missing field in `results_model.json`.

An older single-checkpoint full-eval attempt did invoke EASI-8, but only
MindCube-Tiny completed. The other datasets were unavailable to the offline SCO
worker because the job did not point at the shared Hugging Face dataset cache.
That partial result is not a complete QA evaluation.

## Repair

- `sco_active_spatial_eval_entry.sh` now continues from parallel navigation to
  parallel EASI-8 instead of exiting early.
- Workers use the shared caches under
  `/mnt/umm/users/yinbaiqiao/.cache/huggingface` and run in explicit offline
  mode.
- `active_spatial_qa_parallel.py` records a per-checkpoint plan, progress log,
  aggregate `easi_summary.json`, and `qa_parallel_completion.json`.
- `easi_eval.py --no-summary` prevents concurrent checkpoint workers from
  racing on the aggregate summary file.
- The completeness contract is now ID + OOD + QA; navigation-only success is
  not reported as complete evaluation.

At the snapshot time, all eight v46 QA workers had loaded the cached VSI-Bench
and entered real model inference over 5,130 samples. No network, token, or
dataset-cache error had appeared.

## Output roots

All results are under `evaluation/sweeps/active_spatial/`:

- `qwen_v46_clean_all_ckpts_id_ood_qa_20260828`
- `qwen_v47_clean_all_ckpts_id_ood_qa_20260828`
- `qwen_v48_clean_all_ckpts_id_ood_qa_20260828`
- `qwen_v50_clean_all_ckpts_id_ood_qa_20260828`
- `cambrian_c8_clean_all_ckpts_id_ood_qa_20260828`
- `cambrian_c4_clean_all_ckpts_id_ood_qa_20260828`

Submission log:

`evaluation/sweeps/active_spatial/sco_submissions/clean_checkpoint_evals_20260828_112538.log`

## 2026-08-29 direct-node migration

The still-unallocated Qwen v48 and v50 SCO jobs were suspended to avoid
duplicate evaluation:

- v48 `pt-c69w6mfw`: `STARTING` -> `SUSPENDED`, zero allocated nodes;
- v50 `pt-pcwkumg2`: `STARTING` -> `SUSPENDED`, zero allocated nodes.

Both matrices were moved to the user-provided node `10.119.27.217` (eight
H800-80GB GPUs). Direct launcher PID: `10991`.

The node already hosted a small U1 protocol evaluation on GPUs 0-6 and an HTTP
renderer on GPU 7. Those processes were left untouched. The direct navigation
workers therefore use `gpu_memory_utilization=0.5` instead of `0.7`; this only
reduces vLLM KV-cache reservation and does not change model weights, prompts,
datasets, generation settings, or metrics.

The direct schedule is:

1. v48 navigation on GPUs 0-7;
2. v48 EASI-8 on GPUs 0-2 concurrently with v50 navigation on GPUs 3-7;
3. v50 EASI-8 on GPUs 0-2;
4. generate both consolidated analysis reports.

At launch validation, all eight v48 vLLM engines completed model load,
KV-cache allocation, and CUDA graph capture, then entered real navigation
episodes. No OOM or engine initialization failure was observed.

Direct runtime state and logs:

`evaluation/sweeps/active_spatial/direct_v48_v50_20260829/`
