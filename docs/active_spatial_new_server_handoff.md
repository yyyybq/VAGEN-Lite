# Active Spatial New Server Handoff

Date: 2026-07-17

This document is written for a new Codex/session with no prior context. The goal is to continue the VAGEN-Lite Active Spatial project on a new server without re-discovering the same engineering and experiment pitfalls.

## 0. Current Local State

Repository:

```bash
/nas/baiqiao/active_spatial/VAGEN-Lite
origin: https://github.com/yyyybq/VAGEN-Lite
branch: main
latest pulled commit: a0408e46 Reuse gsplat torch extension cache
```

GitHub updates have already been pulled locally with `git pull --rebase --autostash origin main`. The pull succeeded and the autostash was applied without conflicts.

Recent GitHub commits now present locally:

```text
a0408e46 Reuse gsplat torch extension cache
accd0882 Use local JIT cache for active spatial runs
97902a1c Add no-delta active spatial experiments
6f7de3c4 add new experiments and fix env/model code
```

Important: this worktree is still dirty. Many changes from the previous debugging/design rounds are local and not guaranteed to be on GitHub. A new server that only runs `git clone https://github.com/yyyybq/VAGEN-Lite` will miss those changes unless this worktree is transferred, patched, committed, or pushed.

Local modified/untracked files include:

```text
data_gen/active_spatial_pipeline/{config.py,pipeline.py,run_pipeline.py,task_generator.py}
data_gen/active_spatial_sft/{path_finder.py,sft_generator.py}
evaluation/{agents.py,eval_config.py,eval_runner.py,metrics.py}
scripts/{active_spatial_eval_report.py,active_spatial_eval_sweep.py,active_spatial_full_eval.py}
scripts/{active_spatial_planner_audit.py,filter_active_spatial_jsonl.py}
vagen/envs/active_spatial/{env.py,env_config.py,prompt.py,spatial_potential_field.py}
vagen/envs/active_spatial/{region_metrics.py,visual_bbox_metrics.py}
docs/{active_spatial_eval_system.md,active_spatial_task_reward_audit.md}
examples/evaluate/active_spatial/
```

Before moving to a new server, preserve these local changes. The safest options are:

```bash
# Option A: commit and push to a working branch
git checkout -b active-spatial-handoff
git add .
git commit -m "Active spatial eval and reward handoff"
git push origin active-spatial-handoff

# Option B: transfer the entire worktree directory as-is
rsync -a /nas/baiqiao/active_spatial/VAGEN-Lite NEW_SERVER:/nas/baiqiao/active_spatial/
```

Do not assume GitHub `main` alone contains the full current implementation.

## 1. What The Project Is Exploring

The Active Spatial part of VAGEN-Lite trains a vision-language-action model to actively move a camera in 3D indoor scenes until a spatial relation becomes true. The model receives visual observations and a task instruction, then emits discrete actions such as turning and moving forward.

The project is trying to answer whether RL can teach spatial viewpoint control beyond static spatial QA:

- Can the model move to viewpoints satisfying geometric/visual relations?
- Does RL improve general spatial capability, or only learn task/action shortcuts?
- Can validation performance predict test performance across ID and OOD distributions?
- Which failures come from model learning, and which come from task/reward/action-space design?

Current task families include absolute positioning, projective relations, equidistance, FoV inclusion, occlusion alignment, centering, screen occupancy, size/equal-size style relations, and planned apparent-size ordering. `delta_control` is currently excluded because its reward/action/odometry setting has been unreliable.

## 2. Major Local Changes From Previous Rounds

### Evaluation System

A systematic eval framework was added around:

```text
docs/active_spatial_eval_system.md
examples/evaluate/active_spatial/
scripts/active_spatial_full_eval.py
scripts/active_spatial_eval_sweep.py
scripts/active_spatial_eval_report.py
scripts/gen_ood_splits.py
scripts/easi_probe.py
```

The intended evaluation has three layers:

1. Embodied navigation tests: ID, OOD scene, OOD object/category, OOD template, OOD geometry, hard delta. Every checkpoint should run the same matrix.
2. Training/validation alignment: compare validation and test at the same step; check whether best-val is also best-test and whether latest degrades.
3. Capability probes: EASI, MindCube, SPAR, MMSI, ViewSpatial style static spatial QA to test whether RL improves general spatial reasoning or only task-local action behavior.

### Region-Aware Metrics

`sample_target` is now treated as an auxiliary diagnostic target, not the definition of success. This matters because many tasks have a valid region, line, ray, half-plane, ring, or image relation rather than a single correct camera point.

Local additions:

```text
vagen/envs/active_spatial/region_metrics.py
scripts/active_spatial_planner_audit.py
```

The planner/audit should ask: "Can this task be solved within 20 steps under the current action space and success gate?" It should not only ask whether the agent can reach `sample_target`.

### Projected-BBox Visual Scoring

A local projected-bbox scoring layer was added:

```text
vagen/envs/active_spatial/visual_bbox_metrics.py
vagen/envs/active_spatial/spatial_potential_field.py
vagen/envs/active_spatial/env.py
vagen/envs/active_spatial/env_config.py
```

For visual relation tasks, the intended success/reward semantics should use projected 3D object boxes in the current image:

- Is the target visible/in-frame?
- Is its projected area large enough?
- Is its image center left/right/centered as requested?
- Does another object occlude or overlap it in the intended way?
- Does apparent size/order/occupancy match the task?

This is a semantic shift from older geometry-only scoring. Old results and new results should not be compared as if they use the same reward.

### Occlusion Fixes

Occlusion had a hidden failure mode: the valid region is a ray behind the occluder, not "move toward the target object." The old prompt and generic navigation hints could push the model toward the wrong side.

Local fixes/intent:

- Include or prefer concrete `task_description`, e.g. "Position where wardrobe is hidden behind coffee_maker".
- Add occlusion-specific prompt guidance: position yourself so the occluder is between you and the target.
- Filter very narrow occlusion cases so the effective angular window is compatible with 20 degree turns.
- Avoid treating zero-action validation successes as evidence of navigation ability.

### Delta Control Exclusion

`delta_control` is currently skipped for training and validation unless specifically debugging it. The reason is not that it is uninteresting, but that the current reward/action/odometry setup often makes it look impossible or misleading.

Relevant places:

```text
vagen/envs/active_spatial/env_config.py
data_gen/active_spatial_pipeline/run_pipeline.py
examples/train/active_spatial/experiments/v42_nodelta_*.sh
```

## 3. Data Files And Counts

Files currently present from GitHub/local data:

```text
data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_6types.jsonl
data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_no_ood.jsonl
data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl
```

Observed line counts:

```text
train_100scenes_6types.jsonl        11157
train_100scenes_no_ood.jsonl        28662
val_ood_v2_centering.jsonl             25
```

`train_100scenes_6types.jsonl` task counts:

```text
absolute_positioning       2003
delta_control              2003
equidistance               1770
projective_relations       2003
occlusion_alignment        1875
fov_inclusion              1503
```

The file name says 6 types, but it still includes `delta_control`. The v42+ scripts exclude delta at runtime using `TRAIN_EXCLUDE_TASK_TYPES=delta_control` and `ID_VAL_EXCLUDE_TASK_TYPES=delta_control`, leaving about 9154 non-delta training samples.

Important caveat: some older JSONL samples are unhealthy under the new visual projected-bbox semantics. That does not mean the new scorer is unusable; it means the old data was generated for older criteria. For final runs under the new scoring, audit/filter/regenerate data instead of blindly using every old sample.

## 4. New GitHub Experiment Scripts

GitHub now includes these no-delta experiments:

```text
examples/train/active_spatial/experiments/v42_nodelta_w1.sh
examples/train/active_spatial/experiments/v42_nodelta_w3.sh
examples/train/active_spatial/experiments/v43_nodelta_w3_lr3e7.sh
examples/train/active_spatial/experiments/v44_nodelta_w3_fastcosine.sh
examples/train/active_spatial/experiments/v45_nodelta_w3_kl40.sh
```

Experiment intent:

- `v42_nodelta_w1`: no delta, window size 1.
- `v42_nodelta_w3`: no delta, window size 3. Main paired ablation against W=1.
- `v43_nodelta_w3_lr3e7`: same as W=3 but actor LR reduced from `5e-7` to `3e-7`.
- `v44_nodelta_w3_fastcosine`: W=3 with faster cosine decay and total steps 500.
- `v45_nodelta_w3_kl40`: W=3 with stronger KL coefficient `0.40`.

All are responses to prior instability: after early improvement, entropy often rises and validation degrades. The goal is to keep the useful early learning while delaying or preventing collapse.

## 5. Current Training Status

On this local server at the time of handoff:

- GitHub code is updated to `a0408e46`.
- Local worktree contains uncommitted active-spatial changes.
- No active `active_spatial`/VAGEN training process was found.
- `nvidia-smi` showed all 8 L40S GPUs idle.
- No local `exps/vagen_active_spatial` checkpoints/logs were found in this clone.

So the current blocker is not a running job; it is setup/state transfer:

1. Decide whether the next experiments use GitHub `main` only or the local visual/region-aware modified worktree.
2. Preserve/transfer/commit the local modifications if using the new reward/eval system.
3. Fix hardcoded paths on the new server.
4. Audit/filter/regenerate data before final training under the new visual reward semantics.

## 6. New Server Setup Notes

Several training scripts currently contain hardcoded paths from the previous server:

```text
/scratch/by2593/project/Active_Spatial/VAGEN-Lite
/scratch/by2593/project/Active_Spatial/InteriorGS
/scratch/by2593/miniconda3/envs/vagen-lite/bin/python
```

The current local path is:

```text
/nas/baiqiao/active_spatial/VAGEN-Lite
```

The Python previously used in this environment was:

```text
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python
```

On a new server, either create compatible symlinks or edit the hardcoded paths in:

```text
examples/train/active_spatial/run_experiment.sh
examples/train/active_spatial/env_config_v39_6types_distinfo.yaml
examples/train/active_spatial/experiments/*.sh
```

Path fix option A, if allowed:

```bash
mkdir -p /scratch/by2593/project
ln -s /nas/baiqiao/active_spatial /scratch/by2593/project/Active_Spatial
```

Path fix option B:

Edit `run_experiment.sh` and the YAML/scripts to point directly at the new repo, conda env, InteriorGS root, data JSONLs, and OOD val JSONL.

Also make sure cache directories are local and writable. Recent GitHub commits added local cache/JIT handling for gsplat/torch extensions; keep that behavior. Do not let extension caches fall back to a quota-limited `$HOME`.

## 7. Recommended Next Plan

### Step 1: Preserve Code State

Do this before starting any new server work:

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
git status --short
```

If continuing with the local region/visual scoring system, commit/push a branch or transfer the whole worktree. Do not start from GitHub `main` and assume the local changes are there.

### Step 2: Audit Data Under Current Success Gates

Run planner/score audit before final training with the local visual reward code. A small smoke audit is enough to catch path/import mistakes:

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python scripts/active_spatial_planner_audit.py \
  --input data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_6types.jsonl \
  --env-config examples/train/active_spatial/env_config_v39_6types_distinfo.yaml \
  --max-items 200 \
  --max-steps 20 \
  --beam-size 512 \
  --output outputs/active_spatial_audit/train_6types_audit.jsonl \
  --summary-output outputs/active_spatial_audit/train_6types_audit_summary.json
```

If many projected-bbox tasks fail at their own generated targets, regenerate/filter the data before treating the run as a final experiment.

### Step 3: Start No-Delta Baselines

Once paths and data are settled, start the paired v42 comparison:

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
nohup bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/v42_nodelta_w1.sh \
  > v42_nodelta_w1.log 2>&1 &

nohup bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/v42_nodelta_w3.sh \
  > v42_nodelta_w3.log 2>&1 &
```

If W=3 repeats the entropy explosion around the mid-run, continue with:

```text
v43_nodelta_w3_lr3e7
v44_nodelta_w3_fastcosine
v45_nodelta_w3_kl40
```

Do not run all variants blindly before checking early validation curves.

### Step 4: Select And Evaluate Checkpoints

Use the new checkpoint selector:

```bash
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python scripts/select_best_checkpoint.py \
  --exp v42_nodelta_w3 \
  --expdir exps/vagen_active_spatial \
  --n_id 18 \
  --n_ood 25 \
  --val_n 4
```

Then run the systematic eval suite from `examples/evaluate/active_spatial/` and analyze with `scripts/active_spatial_eval_report.py`.

## 8. Things Already Learned

### Training Stability

Prior experiments repeatedly showed this pattern:

- Early checkpoints can improve quickly.
- Later checkpoints often collapse through rising entropy and degraded validation.
- Stronger KL and lower LR help but do not automatically solve the issue.
- Best checkpoint selection is mandatory; latest checkpoint is often not best.

Historical lessons:

- `KL_LOSS_COEF=0.05` was too low and led to bad drift.
- Actor LR `1e-6` was too aggressive for long training.
- LR `5e-7` was better, but still could collapse later.
- LR `3e-7`, stronger KL, or faster cosine decay are plausible follow-ups.
- `v26`-style settings reached strong early validation, but long-run stability remained the bottleneck.

### Delta Control

`delta_control` often had near-zero success and appears to pollute multi-task training. The likely issue is a combination of reward design, action-space mismatch, and pose/depth/odometry interpretation. Keep it excluded unless specifically debugging that task.

### Occlusion

Occlusion is not a simple "approach target" task. The correct camera region is behind the occluder, so the occluder lies between camera and target. Old prompts and generic hints could actively encourage the wrong movement. Also, narrow occlusion angles can be smaller than the 20 degree rotation step.

### Sample Target

For region tasks, `sample_target` is only one sampled representative, not the task definition. Measuring only distance to `sample_target` can call a solvable policy bad or call an invalid visual state good. Use region-aware and projected-bbox metrics.

### Visual Rewards

For projective/centering/occlusion/size/occupancy tasks, final semantics should be image-based where possible. Geometry-only center/ray/half-plane scores are useful diagnostics, but projected visible bboxes are closer to what the model should learn.

## 9. Pitfalls Not To Repeat

- Do not assume GitHub `main` includes local uncommitted changes.
- Do not run final experiments on old JSONL visual-relation data without audit/filter/regeneration under the current scoring.
- Do not include `delta_control` in main training until its setting is fixed.
- Do not use `sample_target` distance as the only planner/eval criterion for region tasks.
- Do not compare old geometry-reward runs directly against new projected-bbox reward runs.
- Do not train long past entropy explosion. If entropy exceeds about 1.0 or ID validation drops twice in a row, select the best checkpoint and move on.
- Do not use very low KL such as `0.05`.
- Do not use high actor LR such as `1e-6` for long runs unless intentionally testing instability.
- Do not reintroduce the old camera-forward convention bug. The local fixes assume the active spatial action logic uses local `+Z` forward consistently.
- Do not let torch/gsplat/triton caches write into a quota-limited home directory.
- Do not reserve the rendering GPU with a memory-holder process; the GS renderer needs memory.
- Do not trust zero-action validation successes as navigation ability.
- Do not make success depend on "A not visible" when the intended task is "A visible but occluded/related to B"; visibility and bbox gates need to be explicit.

## 10. Quick Status Checklist For New Session

Run this checklist first:

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
git log --oneline -5
git status --short
nvidia-smi
ls -la exps/vagen_active_spatial || true
```

Expected from this handoff point:

```text
latest commit should include a0408e46
worktree should show local active_spatial modifications unless they were committed
GPUs may be idle
local exps directory may be absent
```

If `git status` fails with "dubious ownership", use a one-time safe-directory override:

```bash
git -c safe.directory=/nas/baiqiao/active_spatial/VAGEN-Lite status --short
```

Do not change global Git config unless the user wants that.

## 11. Short Version

The project is now at a fork point:

1. GitHub added practical no-delta training scripts and cache fixes.
2. Local work added a deeper correction: region-aware planner/audit and projected-bbox visual rewards.
3. The next meaningful run should either be a clean GitHub no-delta baseline or a clearly labeled new-reward run after data audit/regeneration.
4. The main immediate task for a new server is not model design; it is preserving the exact code state, fixing paths, auditing data, then starting v42 W=1/W=3 no-delta baselines with early-stop checkpoint selection.
