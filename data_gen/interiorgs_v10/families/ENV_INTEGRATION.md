# v2 task families: what the runtime needs (workstream F)

Since 2026-10-13 the v2 support is part of the env itself (the former `canonical_task_metrics_v2.patch` and
`env_structure_override.patch` were merged into `vagen/envs/active_spatial/`). This file records what was
merged and what an integrator still has to check.

## 1. The v2 metric in the env

* `vagen/envs/active_spatial/canonical_task_metrics_v2.py` holds the v2 judges. It is the single source:
  the generator imports it (`families/judges_v2.py` is an alias of that module).
* `vagen/envs/active_spatial/canonical_task_metrics.py`:
  * **Constants.** `CANONICAL_TASK_METRIC_VERSION_V2`, `V2_FAMILY_TASK_TYPES`, `SUPPORTED_TASK_TYPES_V2` and a
    version-to-types map. `SUPPORTED_TASK_TYPES` is unchanged and remains the v1 set.
  * **`score_canonical_task`.** Accepts v1 and v2 rows: v1 rows run exactly the old code; v2
    `projective_relations` / `fov_inclusion` rows run the v1 judges; v2 family rows go to `score_v2_family`.
  * **`uses_canonical_backend`.** Accepts v2 rows for the families plus projective and fov. Unknown
    versions still raise; a v1 version on a new family still raises; rows without a version return `False`.
* `env.py` `reset`: rows whose `collision_convention.requires_env_override` is true load the scene with the
  calibrated `structure_y_sign` (scenes the v1 label-centre rule leaves ambiguous).
* `tests/test_families.py::test_env_runtime_scores_v2_rows_with_the_same_judges` checks the dispatch.

Every canonical consumer handles v2 rows with no further change, because they all route through `uses_canonical_backend` / `score_canonical_task`:

* `env.py`: reset (l.545, 636), `done` (l.935-944), max-step truncation (l.1047), auto-termination (l.1124), `_calculate_canonical_metric` (l.1381), `_calculate_current_score` (l.1395) and `_calculate_pose_reward` (l.1482).
* `runtime_render_camera_parameters`.
* `dataset_contract.validate_canonical_row`.
* SFT `path_finder` and `sft_formatter`.
* `active_spatial_qa/contract.py`.
* `scripts/r1_runtime_consistency.py`.

The reward potential is `metric["shaping_score"]`, success is `metric["success"]` (all gates pass), and `shaping_score >= 0.5` holds exactly when `success` is true.

## 2. Blocking bug in contract.py (lead; affects v1 rows too)

`contract.CAMERA_MODEL_VERSION = "canonical_h1_resize_v1"` is not accepted by:

* `env.runtime_render_camera_parameters` (l.111-121), which accepts `{"canonical_camera_h1_resize_v1", "canonical_h1_from_frozen_candidate_intrinsics_and_pose"}` and raises `ValueError` on the first render;
* `dataset_contract.validate_canonical_row`, which raises "formal data requires canonical H1 camera metadata";
* `scripts/active_spatial_dense_score_preflight.py`, whose `expected_camera` check fails.

Fix: in `contract.py`, set `CAMERA_MODEL_VERSION = CANONICAL_CAMERA_H1_RESIZE_V1`, which is `"canonical_camera_h1_resize_v1"`. The families' `build_item` and `sampling.projective_item` both read the constant, so they pick up the fix with no further change.

## 3. env.py changes (`vagen/envs/active_spatial/env.py`)

1. **Occlusion guidance (l.739-744, `_build_task_prompt`).** With the default `enable_occlusion_task_guidance=True`, every `occlusion_alignment` row gets two strategy lines appended. It also gets `"Occluder hint: <label> is to your front-left, distance 2.31 m"`, which is computed from the live pose. That is privileged information, it uses the raw label instead of the disambiguating `ref`, and it means the policy no longer sees the certified `task_description`. Make one of these changes:

   ```python
   if (item.get("task_type") == "occlusion_alignment"
           and self.config.get("enable_occlusion_task_guidance", True)
           and not uses_canonical_backend(item)):        # keep v2 rows' frozen instruction
   ```

   Alternatively, set `enable_occlusion_task_guidance: false` and `enable_occlusion_occluder_hint: false` in every v10 env config.

   The role order matches what the guidance code assumes: `objects[0]` / `object_a_*` is the hidden target and `objects[1]` / `object_b_*` is the occluder. The rows also carry `params.occluded` / `params.occluder` ids.

2. No other task-type-specific branch in `env.py` needs changing. `_get_target_pose`, `_get_target_center` and the legacy reward read `target_pose`, `sample_target`, `target_region.sample_point` and `params.object_center`. v2 rows deliberately carry none of these, so the legacy distance and visibility terms stay inert.

## 4. Prompt text (`vagen/envs/active_spatial/prompt.py::system_prompt`)

Only `projective_relations` gets canonical wording. Every other task type gets "Progress toward target pose: continuous reward based on distance and orientation improvement" and "Reaching target pose", which do not describe the v2 reward. Add a branch keyed on `task_type`, for example:

| task_type | strategy | progress line | success line |
| --- | --- | --- | --- |
| absolute_positioning | Distance is horizontal, to the object's nearest side; keep it in view while moving forward/backward. | Gate-aligned distance progress (0.5 at the tolerance) times in-frame fraction | Object in view and distance within max(0.25 m, 10%) |
| equidistance | Slide sideways to balance the two distances; keep both objects in view. | Gate-aligned distance-difference progress | Both in view, distances equal within max(0.25 m, 10%) |
| occlusion_alignment | Move to the far side of the occluder, then line it up with the target; do not walk straight at the target. | Occluder visibility times hidden-fraction/alignment progress | ≥ 85% of the target hidden behind the occluder, occluder in view, target in the view direction |
| screen_occupancy | Move closer or farther; keep the whole object, top to bottom, in view. | Gate-aligned occupancy progress | Height fills the requested fraction ±15% |
| size_distance_invariance | Approach the smaller-looking object or back away from the larger one. | Gate-aligned log-height-ratio progress | Apparent heights within a factor of 1.2, both fully visible |
| centering | Translate sideways until the object sits midway between the two flanks. | Gate-aligned image-midpoint progress | Within max(6 px, 15% of the flank separation) |

`env.system_prompt()` reads `self.current_item` when it is called. If the trainer builds the system prompt before `reset`, it gets the generic text. Pass the task type per episode or call it after `reset`.

## 5. Config requirements for v2 rows

* `enable_potential_field: true`. The canonical shaping score only becomes the reward inside the `enable_potential_field and potential_field is not None` branches (reset l.632, `_calculate_pose_reward` l.1480). With it off, success still comes from canonical gates but the reward falls back to the legacy pose reward.
* `enable_auto_termination: true` and `enable_explicit_done: false`, as in the R1 configs.
* `enable_visibility_check: false`, or `enable_visibility_shaping_reward: false`. The visibility checker is not part of the canonical metric.
* `near_success_threshold`, if used, must be `< 0.5`. `shaping_score ≥ 0.5` already means success. `success_score_threshold` is ignored for canonical rows.
* Select families with `include_task_types` / `exclude_task_types`.

## 6. Other consumers (metadata only, non-blocking)

* `scripts/active_spatial_dense_score_preflight.py:219` filters "visual rows" with `SUPPORTED_TASK_TYPES`, which is v1 only, and checks a single expected metric version. Use `SUPPORTED_TASK_TYPES_V2` and accept a per-row version.
* `data_gen/active_spatial_sft/sft_generator.py:861-866` writes `CANONICAL_TASK_METRIC_VERSION` and `GATE_ALIGNED_SHAPING_SCORE_VERSION` (v1) into the manifest `score_contract`. Record the row's version instead. Per-frame traces already use `metric["shaping_score_version"]`, which is `"canonical_gate_aligned_shaping_v2"` for family rows.
* `scripts/r1_repair_pipeline_tests.py:135` already expects `uses_canonical_backend(... "unknown")` to return `False`, but the current runtime raises (pre-existing; unchanged by the merge).

## 7. Admission notes (workstreams D and A)

* **Terminal identifiability.** Use `FAMILIES[t].TERMINAL_IDENTIFIABLE_ROLES`. For `occlusion_alignment` only the occluder (role 1) must be identifiable at the end, because the target is hidden by definition. Prefer initial states where the target (role 0, `INITIAL_PREFERRED_ROLES`) is identifiable.
* **Non-triviality.** Reject an initial state that already succeeds; that is the same rule as v1. `common.region_stats(...)["success_pose_fraction"]` is available for a triviality cap. In the synthetic scenes no family exceeds 7.1%.
* **Region size.** Some parameterisations leave a tiny region in small rooms, for example screen_occupancy 0.15 on a 0.45 m tea table at 0.09 m². Admit an item only if its best sampler candidate has `lattice_support >= 2`, or `region_stats(...)["expected_lattice_successes"] >= 5`.
