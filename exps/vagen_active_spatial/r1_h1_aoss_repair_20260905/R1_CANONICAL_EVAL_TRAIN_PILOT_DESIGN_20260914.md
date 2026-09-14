# R1 Canonical Evaluation and Minimal Training-Control Design

Date: 2026-09-14

This document is a design only. No checkpoint was restored, no policy rollout was run, and no training was started.

## Frozen task system

All comparisons must use the same `canonical_spatial_task_h1_v1` environment: H1 camera, six formal actions (0.3 m translation and 20 degree turn), the frozen collision convention, 12-step limit, 12 px Projective relation margin, canonical auto termination, and `projective_observability_v1_initial_keyframe`. Medium means a complete formal-runtime lower-bound search finds no success through depth 3 and an independently replayed certificate first succeeds at step 4, 5, or 6.

## Available inventory and isolation

The verified inventory before the bounded frontier experiment contains 96 unique source identities and 98 verified episodes. Only 64 source identities / 65 episodes have original split `train`; the other 32 source identities are validation, test, or OOD and remain evaluation-only. The evaluation identities span seven scenes, but all seven have participated in R1 development, so they are a development regression set rather than untouched generalization evidence.

The train-only inventory is sufficient for task-system and overfit/smoke validation, not a broad training claim. It spans ten scenes but is imbalanced: 37/64 source identities are bed--wardrobe category pairs, and the verified train episodes contain 46 step-4, 19 step-5, and no step-6 examples.

## Canonical evaluation-only comparison

Compare the original pretrained Qwen2.5-VL-7B-Instruct baseline and v46 at step 250 on one immutable canonical evaluation manifest. Use paired seeds, prompts, images, action parser, renderer, collision handling, termination, and scorer. Do not tune on these results.

Report overall and paired results by split, scene, relation, category pair, and certified difficulty:

- canonical success and first-success step;
- collision attempts, invalid actions, timeouts/truncations, and premature termination;
- action count/mix and pose/score trajectory;
- initial/terminal/path observability diagnostics;
- paired model disagreement and confidence intervals over source identities.

The current 32 evaluation identities may be used for a development regression comparison only. A later independent evaluation should freeze at least 60 new identities before model inspection: at least ten each for ID, scene OOD, instance OOD, category OOD, template OOD, and geometry OOD, spanning at least ten scenes not used in R1 selector or threshold development. The current inventory has no category-OOD Medium identity and cannot satisfy this condition.

The historical training log proves that v46 step 250 was once saved under `exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/checkpoints/global_step_250`, but its actor Hugging Face directory is currently empty. The checkpoint artifact and the pretrained model snapshot must therefore be restored/resolved and hashed before evaluation; paths must not be inferred from an empty directory.

## Minimal training control

Reuse the stable v46 training skeleton and its frozen optimizer, LR, KL, auxiliary-loss, task-mixture, and checkpoint policy. Do not add an LR/KL/auxiliary sweep. The comparison changes camera, task/runtime semantics, and repaired data together, so it must be reported as a system/data repair control, not a reward-only ablation.

For a controlled pilot, freeze at least 200 unique train source identities across at least 20 scenes, with each left/right relation represented by at least 80 identities, each first-success bucket 4/5/6 represented by at least 40 verified episodes, and no unordered category-pair family above 35 percent. These are predeclared pilot sufficiency criteria, not universal task definitions.

Against the current 64 train identities, the minimum deficits are 136 source identities, ten scenes, 44 left, 52 right, and 40 step-6 episodes, plus category-pair diversification. New data must preserve the same runtime/RGB evidence and failure accounting; unsupported source rows remain explicit rather than being silently removed.

## Execution order

1. Freeze the canonical specification, verified inventory, and development regression manifest with SHA256.
2. Restore and hash the pretrained and v46@250 model artifacts without running them.
3. Run the paired 32-identity development evaluation to verify functional compatibility only.
4. Build and freeze the untouched 60-identity evaluation set.
5. Fill the train-only coverage deficits, then run one stable-skeleton pilot against its unchanged control.

No broad generalization or training-readiness claim follows from the present small, development-exposed inventory.
