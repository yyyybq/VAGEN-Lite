# Active Spatial Task / Reward / Action-Space Audit

Date: 2026-07-13

This note audits the current active-spatial navigation tasks from the angle of
whether the policy can actually move to a high-scoring viewpoint under the
current reward and discrete action space.

## Scope

The main focus is the current no-delta 8-task setup:

- `absolute_positioning`
- `equidistance`
- `projective_relations`
- `centering`
- `occlusion_alignment`
- `fov_inclusion`
- `size_distance_invariance`
- `screen_occupancy`

`apparent_size_ordering` is listed separately because it was added later and is
not necessarily included in the runs that motivated this audit.

## Main Conclusion

The low success rate is very plausibly caused by the task/reward/action-space
definition, not only by model weakness. The issues are not uniform across tasks:

1. Some tasks contain code-level or scorer-level bugs.
2. Some tasks are geometrically correct but too thin or too precise for
   `0.3m` translation and `20-30deg` rotation.
3. Some reward definitions can give high total score to a partial solution that
   is near the geometric locus but does not actually satisfy the visual task.
4. The data pipeline checks region-in-room reachability, but it does not prove
   reachability under the exact discrete action set, collision settings, and
   episode length used by training.

The strongest immediate suspects are `fov_inclusion`, `centering`,
`occlusion_alignment`, `size_distance_invariance`, and `screen_occupancy`.

## Quick Sanity Check

I ran a synthetic oracle check: generate one task of each type from simple
synthetic 3D boxes, then score the generated `sample_point` and `sample_forward`
with the potential field. The oracle poses mostly score close to 1.0:

| task | oracle total | observation |
| --- | ---: | --- |
| `absolute_positioning` | 1.000 | generator and scorer align on sample pose |
| `equidistance` | 0.998 | generator and scorer align on sample pose |
| `projective_relations` | 0.996 | generator and scorer align on sample pose |
| `centering` | 1.000 | misleading: scorer did not receive real B/C centers |
| `occlusion_alignment` | 1.000 | geometric scorer aligns, but does not verify rendered occlusion |
| `fov_inclusion` | 0.998 | oracle can work, but radius formula is wrong |
| `size_distance_invariance` | 1.000 | oracle can work, but equality curve is thin |
| `screen_occupancy` | 1.000 | oracle can work, but exact ratio is action-sensitive |

Interpretation: most generated sample poses are internally scoreable, so the
problem is less "sample target always impossible" and more "many target regions
are brittle, reward is misaligned, and discrete-action reachability is not
validated."

## System-Level Issues

### 1. Action granularity is mismatched with task design

The initial-view difficulty comments still assume `0.1m` translation and `5deg`
rotation:

- `data_gen/active_spatial_pipeline/config.py`: `distance/0.1 + yaw_offset/5`
- `data_gen/active_spatial_pipeline/pipeline.py`: estimated steps also use
  `distance/0.1` and `yaw_offset/5`

But the environment defaults and most recent train configs use coarser control:

- `vagen/envs/active_spatial/env_config.py`: default `0.3m / 30deg`
- many v18-v24 configs: `0.3m / 20deg`

This creates two opposite problems:

- Difficulty accounting is wrong. A sample considered "12 steps away" may be
  only a few coarse actions away.
- Precision tasks become hard to finish. A circle, ray, or equality curve may
  require sub-`0.3m` correction, especially near object boundaries.

Recommendation: run a dedicated action ablation:

- `0.3m / 20deg` current baseline
- `0.2m / 10deg`
- `0.1m / 10deg`
- add optional fine actions: `move_forward_small=0.1`, `turn_small=10deg`

The first diagnostic should be not RL performance, but planner upper bound:
given the same start pose, can a discrete planner reach success in 20 steps?

### 2. "Reachability" is not action-space reachability

`validate_target_reachability()` only checks whether a target circle/curve has
some sampled points inside a room polygon. It does not check:

- collision-free path from init pose to a successful pose
- exact action set, such as legacy vs strafe
- step size and turn size
- max episode length
- whether the final orientation is attainable at the required resolution

Recommendation: add a dataset audit pass that runs a small BFS/A* or heuristic
planner under the exact env config. Each generated item should record:

- `oracle_score_at_sample`
- `best_discrete_planner_score`
- `planner_success_within_max_steps`
- `min_steps_to_success`
- `failure_reason`: no position, no orientation, collision, max steps, no FoV

If planner success is low, RL success cannot be expected to be high.

### 3. Total score can reward partial visual satisfaction

`_combine_scores()` dynamically blends position and orientation. Position can
remain dominant even near success, and FoV penalty only changes orientation, not
position. This is good for preserving gradients, but weak as a success criterion.

Failure pattern:

- The policy reaches the geometric locus.
- The target is partly or fully out of view.
- Position score stays high.
- Total score may still look decent.
- Training learns a shortcut: "go near the locus" rather than "make the visual
  relation true."

Recommendation:

- Keep dense shaping if useful, but make success gates task-specific:
  `position_ok AND orientation_ok AND visibility_ok AND relation_ok`.
- Apply this to explicit `done`, auto-termination, and max-step success. At the
  moment some branches still rely mainly on total score.

### 4. Reward scale can hide navigation signal

Older configs use a large `format_reward` such as `0.2`, while potential-field
progress is often a small delta. Some later configs set `format_reward=0`, which
is directionally better.

Recommendation:

- For diagnosis runs, set `format_reward=0`.
- Report separately:
  - geometric progress reward
  - visibility reward
  - success reward
  - format reward
  - collision penalty
- Do not compare task success across configs with different reward scales unless
  planner upper bound and score distributions are also logged.

### 5. Some tasks are weakly observable from single images

Tasks involving metric distance, equal apparent size, and occupancy require
absolute scale or memory of motion. If the model only gets a single rendered
view plus text, it may not infer metric distance reliably.

Recommendation:

- Evaluate single-step visual ambiguity with static probes.
- Add world-model/prediction heads or explicit observation metadata only in
  controlled ablations.
- Prefer inequality/range tasks over exact equality tasks until the action
  space and observation design are proven.

## Per-Task Audit

### `absolute_positioning`

Goal: stand at a requested distance from one object and face it.

What works:

- Generator and scorer are mostly aligned.
- The target is a circle, not a single point, so it is less brittle than delta
  control.

Risks:

- Target distances are clamped by object max dimension. The instruction may say
  `0.8m`, but large objects silently make the actual target farther.
- The score uses distance to object center and center-in-FoV, not rendered bbox
  visibility.
- With `0.3m` steps, some target radii have a coarse final quantization error.

Severity: medium.

Suggested fix:

- Keep this task as a control task.
- Log requested vs effective distance.
- Success should require distance band, object visible, and orientation toward
  the object.

### `equidistance`

Goal: stand where distances to A and B are equal.

What works:

- The geometric scorer matches the perpendicular-bisector idea.
- The valid set is a line, which is broader than a point but still thin.

Risks:

- Initial-position filtering often uses distance to one sampled point, not
  closeness to the whole valid line. A pose can be far from the sample but
  already near another valid part of the line.
- With coarse actions, reaching a thin line plus facing the midpoint can be
  difficult.
- The task says "equidistant", but visual evidence from a single frame may be
  ambiguous unless the scene gives strong scale cues.

Severity: medium.

Suggested fix:

- Use a tolerance band around the bisector for success.
- During data audit, compute planner best score to the whole line, not just the
  stored sample point.

### `projective_relations`

Goal: make A appear left/right of B.

What works:

- Generator samples the correct side of the A-B half-plane.
- Scorer includes a simplified image-plane left/right check.

Risks:

- "A is left of B in the rendered image" depends on camera orientation. The
  half-plane is only a proxy.
- The simplified relation check uses center points, not projected bboxes.
- If both objects are close together or one is occluded, the relation may be
  visually unclear.
- Lateral movement is important. Legacy action space lacks strafe; even strafe
  with `0.3m` may be coarse.

Severity: medium-high.

Suggested fix:

- Define success from projected/rendered 2D centers or bboxes.
- Reject examples where projected separation is too small.
- Use strafe or orbit/fine actions in all experiments involving this task.

### `centering`

Goal: make A appear centered between B and C.

Major bug:

- Generator stores `object_a_center` and `midpoint_bc`, but not
  `object_b_center` or `object_c_center`.
- Scorer falls back to `midpoint_bc` for both B and C. As a result, the FoV
  check does not actually verify B and C.

Other risks:

- Orientation only checks whether A is centered in the camera, not whether A is
  visually between B and C.
- The target is a ray, which is geometrically thin.
- A, B, C can be at very different depths or heights; center-based geometry may
  not match rendered appearance.

Severity: high.

Suggested fix:

- Store B and C centers in the target region.
- Score projected angular relation:
  - all A/B/C visible
  - A lies between B and C horizontally
  - A is near the midpoint of B and C in image coordinates
  - B/C separation is above a minimum threshold
- Add rendered-view validation for generated samples.

### `occlusion_alignment`

Goal: make A hidden behind B.

What works:

- Generator and scorer agree on a ray from A through B.

Risks:

- Scorer checks only geometric ray alignment and whether B is closer than A.
  It does not verify rendered occlusion or bbox overlap.
- FoV check only requires B to be visible, because A should be hidden. This is
  reasonable in spirit, but it does not distinguish "A hidden by B" from "A is
  simply outside the frame."
- The valid region is a thin ray and can be collision-prone around objects.
- Old prompt construction ignored `task_description` and fell back to generic
  preset text, so "A is hidden behind B" was not reliably shown to the model.
- Old initial-view filtering used distance to one sampled point on the ray; an
  initial camera already on the occlusion ray but far from that sampled point
  could pass filtering and produce zero-action successes.
- Roughly yaw-step-scale occluder angular width is a real action-resolution
  issue: if the occluder subtends less than one discrete turn, the policy can
  skip over the useful alignment interval.

Severity: high.

Suggested fix:

- Define success using projected bboxes or sampled rays:
  - B visible
  - A projected behind/overlapping B
  - A visibility ratio below threshold
  - A would be in FoV without occlusion, or at least its projected center is near
    B, to avoid "A outside frame" shortcuts
- Add a minimum occluder/occluded size and overlap threshold.

Implemented partial fixes:

- Runtime prompts now prefer `task_description` and add occlusion-specific
  guidance: the occluder must be between the camera and the hidden target, so
  the policy should not simply move straight toward the hidden object.
- Runtime prompts can include the occluder's relative direction and distance
  from the current camera pose.
- New data generation constrains the sampled occlusion segment so the occluder
  angular width is at least `task_config.occlusion_min_angular_width_deg`
  (default `25.0` degrees). The actual angle is recorded in
  `target_region.params.occluder_angular_width_deg`.
- Initial-view filtering now scores the occlusion ray geometry instead of only
  distance to the sampled point, reducing zero-action starts.
- The occlusion potential field now respects finite ray bounds and uses the
  optional angular-width gate for newly generated rows.

### `fov_inclusion`

Goal: find a view where both objects are visible.

Critical bug:

- The generator computes:
  `np.radians(fov_horizontal) * (1 - 2 * margin)`
- `margin` is configured in degrees, default `5.0`, so this becomes
  `radians(110) * -9` instead of `radians(100)`.
- Because `tan()` is periodic, this can produce plausible-looking but
  arbitrary radii. It is not a correct FoV calculation.

Risks:

- The scorer uses horizontal center angles; it does not ensure full bboxes are
  visible.
- The generator uses a distance annulus, but actual visibility also depends on
  orientation, object size, height, and occluders.

Severity: critical.

Suggested fix:

- Replace the formula with:
  `effective_fov_rad = np.radians(max(fov_horizontal - 2 * margin, 1.0))`
- Generate and score using bbox corners or rendered bbox coverage.
- Reject samples where either object is only barely inside the edge.

### `size_distance_invariance`

Goal: make A and B appear the same size.

What works:

- The Apollonius-circle construction is mathematically reasonable for a simple
  height-over-distance proxy.
- Oracle sample poses can score high.

Risks:

- Equal apparent size is an equality curve. With `0.3m` steps, high success can
  require landing close to a thin curve.
- If object sizes are nearly equal, the generator returns an `equidistance`
  task. This silently changes the task distribution.
- The scorer has a synthetic FoV workaround based on `sample_forward`; this can
  decouple success from both real objects being visible.
- The task uses object height over horizontal distance, not actual rendered bbox
  size.

Severity: high.

Suggested fix:

- Do not silently change task type. Either skip near-equal pairs or keep the
  same task type with an explicit bisector target.
- Convert exact equality to a tolerance band:
  `0.85 <= apparent_size_ratio <= 1.15`, or curriculum over tighter bands.
- Require both rendered bboxes visible.
- Report ratio error as the main metric, not only total score.

### `screen_occupancy`

Goal: make one object occupy a target fraction of the vertical FoV.

What works:

- Generator and scorer use the same height/distance approximation.
- Infeasible close-up ratios are partially filtered.

Risks:

- Exact occupancy is very sensitive to distance. `0.3m` steps can make some
  target ratios unreachable within a high threshold.
- Large ratios such as `0.7` often require getting very close, which collides
  with min-distance and collision constraints.
- The metric is based on object height and center distance, not rendered bbox
  area or height.
- Pitch matters, but the strafe action space removes `look_up/look_down`.

Severity: medium-high.

Suggested fix:

- Use rendered bbox height or area as the success metric.
- Convert exact ratios into bands, such as `target +/- 20% relative`.
- Temporarily drop the `0.7` ratio until planner upper-bound success is high.
- Decide whether pitch should exist. If pitch is removed, generate tasks that
  do not require vertical framing precision.

### `apparent_size_ordering` New Task

Goal: make A appear larger than B, or B larger than A.

Why it is promising:

- It is an inequality task, not an exact equality.
- The reward has a plateau once the desired ordering is clearly satisfied.
- It should be more robust to coarse actions than `size_distance_invariance`.

Remaining risks:

- It still uses height over horizontal distance rather than rendered bbox size.
- It still needs both objects visible.

Severity: low-medium.

Suggested fix:

- Keep it, but evaluate it separately from the older 8 tasks at first.
- Add rendered bbox validation when generating examples.

## Recommended Experiments

### E0. Dataset and planner audit before any RL

For every train/val/test item, compute:

- initial total/position/orientation/FoV score
- oracle sample total/position/orientation/FoV score
- best score reachable by a discrete planner in `max_steps`
- whether success is reachable by task-specific gates

This directly answers whether the task is learnable under the action space.

### E1. Action-space ablation

Run the same fixed dataset with:

- current `0.3m / 20deg` strafe
- `0.2m / 10deg` strafe
- `0.1m / 10deg` strafe
- current coarse actions plus optional fine actions

Primary metric: planner upper-bound success. RL should only be interpreted
after planner success is high.

### E2. Success-gate ablation

Compare:

- current total-score gate
- `position AND orientation`
- `position AND orientation AND visibility`
- task-specific visual relation gates

Set `format_reward=0` in all diagnostic runs.

### E3. Patch high-risk tasks one at a time

Patch order:

1. `fov_inclusion` margin formula
2. `centering` B/C center storage and projected centering scorer
3. `occlusion_alignment` rendered occlusion metric
4. `size_distance_invariance` tolerance band and no task-type fallback
5. `screen_occupancy` bbox-based band metric

Run planner audit and a short RL smoke run after each patch.

### E4. Task curriculum

Start from robust tasks:

1. `absolute_positioning`
2. `apparent_size_ordering`
3. relaxed `projective_relations`
4. relaxed `fov_inclusion`

Then add thin/equality tasks only after planner and heuristic policies can
solve them reliably.

## Immediate Fix List

P0:

- Fix `fov_inclusion` FoV margin formula. Implemented in
  `data_gen/active_spatial_pipeline/task_generator.py`.
- Fix `projective_relations` left/right half-plane sign to match the project
  camera convention (`+Z` forward, `+X` image right). Implemented in
  `data_gen/active_spatial_pipeline/task_generator.py`.
- Store real B/C centers for `centering`. Implemented in
  `data_gen/active_spatial_pipeline/task_generator.py`.
- Fix `occlusion_alignment` prompt/task-description usage, occluder guidance,
  angular-width filtering, initial-view ray scoring, and finite-ray scoring.
- Add oracle-score and discrete-planner audit scripts. Implemented as
  `scripts/active_spatial_planner_audit.py`.
- Align initial-view step estimates with the env action config.
- Align SFT path finding with the env scorer: `+Z` forward, current c2w pose,
  object bboxes, and camera intrinsics are passed into potential-field scoring.
- Use projected 3D AABB bbox metrics as the primary score for visual-relation
  tasks when bbox metadata is available. Implemented in
  `vagen/envs/active_spatial/visual_bbox_metrics.py` and wired through
  `SpatialPotentialField`, the env, and planner audit.
- Filter newly generated visual-relation items whose sampled target view fails
  the projected-bbox score. Implemented in
  `data_gen/active_spatial_pipeline/pipeline.py`.

P1:

- Make success gates task-specific and include visibility/relation booleans.
- Convert equality/ratio targets into tolerance bands.
- Upgrade projected AABB occlusion to rendered instance-mask occlusion if
  paper-quality occlusion measurement is required.
- Keep `format_reward=0` for diagnosis and main comparisons.

P2:

- Add fine-control actions or a two-stage coarse/fine action space.
- Add curriculum by target tolerance and action granularity.
- Add static visual probes for metric/scale ambiguity.
