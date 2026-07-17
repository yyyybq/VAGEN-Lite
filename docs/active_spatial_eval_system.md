# Active Spatial evaluation system

> Created on 2026-07-08 for the VAGEN-Lite Active Spatial fork.

## 1. What this project is doing

VAGEN-Lite is a lightweight reimplementation of VAGEN on top of the VERL
agent-loop stack. The main framework trains multi-turn VLM agents with PPO/GRPO
in POMDP-style visual environments. The Active Spatial branch adds a 3D camera
navigation task: given rendered observations and a spatial instruction, the
model emits multi-action strings such as `move_forward|turn_left|...` and is
rewarded by geometric progress, visibility, collision/action validity, and
task success.

The Active Spatial code path has four important layers:

- `data_gen/active_spatial_pipeline/`: builds JSONL task datasets and splits.
- `vagen/envs/active_spatial/`: environment, prompts, scoring, collision and
  visibility checks.
- `examples/train/active_spatial/`: experiment configs and the common
  `run_experiment.sh` launcher. Each run materializes into
  `exps/vagen_active_spatial/<experiment>/` with `train.yaml`, `val.yaml`,
  `checkpoints/`, `validation/`, `rollout_data/`, and `train.log`.
- `evaluation/`: standalone evaluator that directly drives `ActiveSpatialEnv`
  and reports overall, per-task, and per-category metrics for random,
  heuristic, frozen, and checkpointed model agents.

## 2. Recent improvements in this fork

The recent Active Spatial work is not just a single model change; it is a
sequence of training-system and analysis improvements:

- v19-v20 introduced stronger reward designs: AND-gated success, threshold
  tuning, no-farm reward variants, dual position/orientation progress, and
  stability patches with higher KL, longer critic warmup, smaller value clip,
  and higher entropy.
- Validation was upgraded from a single fixed in-domain set to optional OOD
  validation through `OOD_VAL_JSONL` and multi-split OOD validation through
  `OOD_SPLITS_DIR`.
- Training now records `train/traj_success/*`, making train success curves
  comparable with validation success curves.
- v26-v28 established LR as the dominant stability lever: `5e-7` was more
  stable than `1e-6`; slow cosine from `1e-6` was ineffective; cosine from
  `5e-7` delayed but did not remove later degradation.
- `scripts/analyze_experiments.py` was expanded to parse validation curves,
  task-type curves, failure modes, and rollout behavior diversity.
- SFT support was moved to LLaMA-Factory because the local VERL SFT path did
  not pass VL image tensors through the trainer.
- `scripts/easi_probe.py` and `scripts/probe_registry.yaml` provide a separate
  offline spatial-QA probe over selected checkpoints, complementing embodied
  navigation evaluation.

The missing piece was a systematic post-training evaluator: every saved
checkpoint from every started experiment should be tested on the same named
test distributions, then compared against the validation curve at the same
training step.

## 3. Evaluation design

The core navigation layer is `scripts/active_spatial_eval_sweep.py`.
For production runs, use the top-level orchestrator
`scripts/active_spatial_full_eval.py`, which runs:

1. embodied navigation test sweep;
2. validation/test alignment through the navigation summary;
3. EASI static spatial-QA probe;
4. consolidated Markdown analysis report.

It treats the experiment directory as the source of truth:

```text
exps/vagen_active_spatial/<experiment>/
  train.yaml
  val.yaml
  checkpoints/global_step_<N>/actor/huggingface/
  validation/<N>.jsonl
  rollout_data/<N>.jsonl
  train.log
```

For each selected experiment it:

1. discovers loadable actor checkpoints;
2. selects steps with `latest`, `best-val`, `all`, `every:N`,
   `range:start:end:stride`, or explicit step numbers;
3. expands the test matrix from
   `examples/evaluate/active_spatial/test_suites.yaml`;
4. writes one `eval_config.yaml` per checkpoint x suite x agent;
5. optionally runs `evaluation/run_eval.py --config ...`;
6. summarizes test metrics and joins validation metrics from
   `validation/<step>.jsonl`.

The test matrix is deliberately config-driven. The current template defines:

- `id_test`: canonical in-domain held-out test set;
- `ood_scene`: held-out scenes/layouts;
- `ood_object`: held-out target objects/categories;
- `ood_category`: target object categories absent from training;
- `ood_template`: paraphrased instructions with the same task semantics;
- `ood_geometry`: metric distances/scales outside the central training range;
- `ood_task_mix`: shifted task-type mix;
- `smoke`: three-episode infrastructure check.

The default matrix excludes `delta_control` and includes
`apparent_size_ordering`, where one object must appear larger than another while
both remain visible.

Older JSONL files can be converted to no-delta variants with:

```bash
python scripts/filter_active_spatial_jsonl.py \
  --input data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl \
  --output data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_no_delta.jsonl \
  --exclude delta_control
```

Before serious runs, make sure the JSONL files referenced by
`examples/evaluate/active_spatial/test_suites.yaml` exist. The OOD scene,
instance/object, category, template, and geometry splits can be regenerated with:

```bash
python scripts/gen_ood_splits.py \
  --train_jsonl data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl \
  --test_jsonl data_gen/active_spatial_pipeline/output_100scenes/test.jsonl \
  --out_dir data_gen/active_spatial_pipeline/ood_splits
```

If your canonical train/test files live elsewhere, pass those paths explicitly
and then update the suite YAML if needed.

Before interpreting RL success, run the oracle/discrete-planner audit on the
same JSONL. It checks the initial score, generator oracle score, and whether a
privileged discrete planner can reach the success criterion within the current
action-space budget:

```bash
python scripts/active_spatial_planner_audit.py \
  --input data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_no_ood.jsonl \
  --env-config examples/train/active_spatial/env_config_v24_100scenes.yaml \
  --max-steps 20 \
  --exclude-task-types delta_control \
  --output outputs/active_spatial_audit/train_planner_audit.jsonl \
  --summary-output outputs/active_spatial_audit/train_planner_audit_summary.json \
  --summary-csv outputs/active_spatial_audit/train_planner_audit_summary.csv
```

Each output row has a `planner_answer`:

- `solvable`: a discrete action sequence reached the configured success gate.
- `not_solvable_within_budget`: exhaustive quantized search finished without
  success.
- `not_proven_search_truncated`: beam/max-state budget was truncated, so this
  is not a proof of impossibility.

For stricter small-split checks, add `--exact --max-states <N>`. For large
splits, keep beam mode and treat `planner_solved=true` as a positive
reachability proof, while treating truncated failures as candidates for a larger
search budget.

The planner audit is region-aware. It still uses the current
`SpatialPotentialField` score and the configured success threshold for
`planner_solved`, but its beam-search heuristic no longer chases the arbitrary
`sample_point`. Each score summary now records:

- `distance_to_region`: distance/violation relative to the full task region.
- `region_score`: smooth score derived from `distance_to_region`.
- `sample_target_distance`: distance to the generator sample point, kept only as
  an auxiliary diagnostic.
- `sample_target_is_auxiliary`: true for region-valued tasks.
- `visual_bbox_overrode_score`: true when a visual-relation task used projected
  3D bbox metrics as the primary score.
- `visual_bbox_available`: whether the item had object bboxes and camera
  metadata sufficient for projected-bbox scoring.
- `visual_score`: projected-bbox relation score before score blending.

Use `distance_to_region` when diagnosing navigation geometry. Use
`sample_target_distance` only to inspect how far the sampled oracle point is
from the current pose.

Visual-relation tasks now use projected 3D AABB bboxes as the main reward/success
score when bbox metadata is available:

- `projective_relations`: A/B projected horizontal ordering, with both objects
  visible.
- `centering`: A projected near the midpoint between projected B/C, B/C
  sufficiently separated, all visible.
- `occlusion_alignment`: target projects into the frame, occluder projects in
  front of it, and their projected boxes overlap.
- `fov_inclusion`: both objects project into the frame, with a centering bonus.
- `size_distance_invariance`: projected bbox heights are equal and both objects
  visible.
- `apparent_size_ordering`: the requested object has the larger projected bbox
  height by the configured ratio.
- `screen_occupancy`: projected bbox height matches the requested screen
  occupancy band.

This is still an analytical projected-AABB metric, not a rendered instance-mask
occlusion metric. It removes the main center-point/sample-target shortcuts;
final paper-quality occlusion analysis can add rendered masks later.

For legacy centering JSONL rows that are missing B/C centers in
`target_region.params`, the audit script repairs those fields in memory from
`target_object.objects` and records the repair names in the per-row `repairs`
field. It does not rewrite the source JSONL.

The `fov_inclusion` margin fix affects newly generated data. Existing JSONL
files already contain their old annulus radii, so regenerate those splits before
using `fov_inclusion` results as final evidence.

The `projective_relations` left/right half-plane sign is also fixed for newly
generated data to match the active-spatial camera convention: local `+Z` is
visual forward and local `+X` is image right. Existing JSONL files may contain
sample target views whose visual left/right relation is inverted; the projected
bbox score and planner audit will expose those as low oracle/sample scores.

For `occlusion_alignment`, newly generated data records
`target_region.params.occluder_angular_width_deg` and filters out rows below
`task_config.occlusion_min_angular_width_deg` by default. When auditing legacy
occlusion splits, treat zero-action successes and low-angle occluders as data
quality issues rather than evidence of navigation ability.

SFT path finding uses the same `+Z` forward convention and projected-bbox score
metadata as the RL environment, so generated expert trajectories are aligned
with the reward used during training.

## 4. Standard experiment protocol

Run a smoke dry run first:

```bash
python scripts/active_spatial_eval_sweep.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps all \
  --steps latest \
  --suites smoke \
  --dry-run
```

Run the full three-layer stack for selected experiments:

```bash
python scripts/active_spatial_full_eval.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exp-root exps/vagen_active_spatial \
  --exps v38_masked_gae_6types_fastcosine,v40_masked_gae_6types_distinfo_w3 \
  --steps best-val,latest \
  --nav-suites all \
  --easi-benchmarks core5 \
  --run
```

The command is restart-safe. If it is interrupted, run the exact same command
again; completed navigation `results_<agent>.json` files and EASI result JSONs
are skipped unless `--rerun` is passed.

Run the smoke suite for real:

```bash
python scripts/active_spatial_eval_sweep.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps all \
  --steps latest \
  --suites smoke \
  --run
```

Evaluate every saved checkpoint for one experiment:

```bash
python scripts/active_spatial_eval_sweep.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps v38_masked_gae_6types_fastcosine \
  --steps all \
  --run
```

Evaluate only the most decision-relevant checkpoints for several experiments:

```bash
python scripts/active_spatial_eval_sweep.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps v33_grpo_rewscale_klhi,v34_grpo_rewscale_clip02,v38_masked_gae_6types_fastcosine \
  --steps best-val,latest \
  --run
```

If the experiment outputs live outside this clone, point the script there:

```bash
python scripts/active_spatial_eval_sweep.py \
  --exp-root /scratch/by2593/project/Active_Spatial/VAGEN-Lite/exps/vagen_active_spatial \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps 're:^(v33|v34|v38)_' \
  --steps best-val,latest \
  --run
```

## 5. Outputs and analysis

All outputs go to:

```text
evaluation/sweeps/active_spatial/<matrix-name>/
  manifest.jsonl
  summary.csv
  summary.md
  full_eval_plan.json
  analysis_report.md
  easi_probe_results/
  <experiment>/global_step_<N>/<suite>/model/
    eval_config.yaml
    results_model.json
    results_model_summary.txt
```

The key columns in `summary.csv` / `summary.md` are:

- `test_success_rate`;
- `test_mean_final_score`;
- `test_mean_score_improvement`;
- `test_spl`;
- `test_mean_steps`;
- `test_mean_collisions`;
- `val/overall_success_mean`;
- `test_minus_val_success_mean`.

Interpretation rules:

- Prefer checkpoints that are strong on `id_test` and do not lose much on
  `ood_scene` / `ood_object`.
- A high validation score but weak test score indicates validation overfitting,
  split leakage, or a too-small validation set.
- A high `id_test` score but weak `ood_geometry` score means the policy may be
  fragile to metric scale or distance distribution shifts.
- Compare `best-val` against `latest`; if latest regresses on test while val
  looked stable, the validation matrix is missing that failure mode.
- Use `results_model.json` per-task and per-category metrics to decide whether
  the loss is metric-distance, projective-relation, or view-perspective specific.

## 6. Baselines

For a full report, run baselines on the same suites:

```bash
python scripts/active_spatial_eval_sweep.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps v38_masked_gae_6types_fastcosine \
  --steps latest \
  --agents random,heuristic,model \
  --run
```

`random` is the floor. `heuristic` is a geometry-informed sanity upper bound.
The trained `model` should beat random by a wide margin and should approach the
heuristic trend on easier in-domain tasks.

## 7. Notes and caveats

- `evaluation/run_eval.py` loads one vLLM model per job. For many checkpoints,
  run the sweep on a GPU worker and expect it to be slow.
- The script skips existing `results_<agent>.json` unless `--rerun` is passed.
- `best-val` uses the validation JSONL at the same global step. If a checkpoint
  exists but no validation file exists at that step, it cannot be selected by
  `best-val` but can still be selected explicitly or by `latest/all`.
- The test-suite file uses repo-relative default paths. Treat it as the canonical
  schema and update paths only if your dataset lives outside this clone.
