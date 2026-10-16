# Active Spatial real PPO batch readiness — 2026-09-07

Status: `BLOCKED_DATA_GATE`  
Decision: no real rollout/capture, snapshot replay, backward, or pilot was run.

## Current R1 inputs read-only

| Artifact | SHA256 | Declared result |
| --- | --- | --- |
| `r1_formal_sources_9.json` | `d5ee015def79400795ea81008e42c17b263d280517ec47fde089011e86a9afa0` | source ledger only; it still names historical manifests, not a frozen post-R1 complete split set |
| `R1_RUNTIME_CANARY_GATE_REPORT_20260905.md` | `d14468ff27d43a4918d5a7b31aa9507f7978706b82e9bcaffb0e5e84a3be1c50` | `BLOCKED` |
| `R1_PROJECTIVE_OBSERVABILITY_GATE_REPORT_20260905.md` | `bfb4185e72ff9a68d0a328dd679845a3cbca7945c8061b05b8a0a537ccbc78a0` | `BLOCKED` |
| `R1_PROJECTIVE_PATH_FIRST_AND_FUNNEL_REPORT_20260906.md` | `48a19549e2521061d93cedd249823ac058e38f760e56720546f388d1cd85add9` | `BLOCKED — PROJECTIVE GENERATION STILL INADEQUATE` |
| collision ambiguity summary | `641a36025f12b9e3fcc276a651bc1674e4234ae366ef2b2b7032d362eebebe69` | 3 quarantined + 3 frozen scenes; 746 impacted rows |

The runtime report identifies canonical task version `canonical_spatial_task_h1_v1`, H1 camera handling (no second K scaling), and collision convention `interiorgs_structure_label_alignment_v1`. These are implementation/canary facts, not fields completed in the formal source rows. The observability gate explicitly says no post-observability final trainable manifest was emitted.

At inspection, `git status --short` contained no R1/canonical/donor script entry. This is only a worktree observation, not a substitute for the absent frozen formal split release; no R1 file was edited in this task.

## Re-run preflight

The exact current R1 source ledger paths were supplied explicitly to `scripts/active_spatial_dense_score_preflight.py`; no slicing, refill, substitute split, renderer, or rollout was used. Result:

| Gate | Result |
| --- | --- |
| S0/S1/S5 resolved-config/reward-only diff | `PASS` — S1−S0 remains only potential application |
| Formal data gate | `BLOCKED` |
| Runtime canary | `BLOCKED` |
| Projective/FOV observability | `BLOCKED` |
| Canonical/camera/collision fields in applicable rows | `BLOCKED` for all nine supplied manifests |
| Initial success | nonzero in formal-train candidate: `1231 / 9154` |
| Forbidden sample overlap | `BLOCKED` |

Forbidden exact/semantic/content intersections are unchanged: train↔val_id `19`; id_test↔ood_scene `316`; id_test↔ood_instance `200`; id_test↔ood_category `338`; id_test↔ood_geometry `335`.

New report: `preflight_report.json`, SHA256 `2fd04c7ec50c469460a178d019f47e7dd2d9d5b63956cbc3269018ff1cc8006d`.

Actual command (exit code `2` is the script's designed blocked result):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
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
  --output-dir docs/diagnosis/active_spatial_dense_score_real_batch_20260907
```

## Change relative to 2026-09-06 preflight

No eligibility improvement occurred. The old and new preflight have identical blocker categories and identical nine source paths; the old report SHA is `3cf054a94e79dc871dd4884fa4c517b3403f06bd289aea4f548e4f9cc287a894`. The new R1 9/6 diagnostics add evidence that projective generation remains inadequate; they do not create frozen formal train/ID/OOD manifests or supersede the two formal `BLOCKED` gates.

## Stop decision

The minimal remaining blocker is a complete, frozen post-R1 formal train/ID/OOD manifest set that passes runtime and observability gates, carries canonical/H1/collision fields on every applicable row, has no forbidden fingerprint overlap, and has zero initial-success rows (or a formally frozen alternative policy). Until then a real PPO batch would necessarily use ineligible historical data, which is prohibited.

Downstream status: snapshot capture `NOT_RUN`; exact replay `NOT_RUN`; S0/S1/S5 reward/advantage diagnosis `NOT_RUN`; gradient adapter/diagnosis `NOT_RUN`; terminal-aware analysis `NOT_RUN`; pilot `NOT_RUN`.

Recommendation: `NOT_READY_FOR_PILOT: formal data gate remains BLOCKED (no frozen eligible manifests; external runtime/observability gates BLOCKED).`
