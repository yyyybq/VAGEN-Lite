# R1 Task-System Freeze and Bounded Projection-Frontier A/B

Date: 2026-09-14

## Outcome

This round closes the task-system specification and verified Medium inventory, and runs exactly one bounded projection-aware frontier variant.

1. **Task system and accepted data: freezeable.** Canonical/runtime/data-quality scopes are explicit and internally consistent. The deduplicated inventory contains 96 verified source identities and 98 verified episodes; all 98 have complete Medium lower-bound evidence, independent runtime PASS, and official RGB PASS.
2. **Projection-aware frontier variant: not supported.** It recovered 0/16 frozen reverse-cap failures and preserved only 3/4 generator controls. Its cost was acceptable, but it failed the predeclared recovery and regression gates. No cap/seed/threshold adjustment or follow-on selector was made.
3. **Model mainline: functional evaluation design is ready, training/generalization evidence is not.** The 32 evaluation identities are development-exposed; the 64 train identities are small and category/step imbalanced. The v46@250 files also need restoration and hashing before any evaluation run.

No 75-shortcut rerun, 788-row rerun, 100-scene regeneration, checkpoint restore, policy rollout, or training was performed.

## Recovered provenance

- Repository state first inspected at `7c7e8b92185e...`. That commit added the prior report and review artifacts.
- The prior SCO worker actually used frozen code `58636eb5fbf110221a1f5d5d5fca7577d1580a53`; its `run_environment.json` recorded the short form `58636eb5c6`. Current HEAD was not substituted for that experiment.
- All prior v3/reverse-diagnostic files match their original checksum except `bootstrap.log`. The log was appended after the original checksum command. A read-only post-close manifest now validates all 253 files.
- The sole new search variant is commit `26af4280cea86e0f10ab8d663607b40a296ac6da`. Its immutable worker archive SHA256 is `640b5a53837a0648b922c88643b75cd5808ac919ca68a60ed30ffc4f95e97a3e`; the archived v4 selector is byte-identical to the commit.

## Canonical constraint scope and `inside_frame`

| Scope | Frozen rule |
|---|---|
| Formal runtime state/action | Six actions; 0.3 m translation; 20 degree turn; formal collision; canonical success after every legal action; 12-step cap. RGB, 0.5 m wall-quality clearance, and source `min_distance` do not define runtime legality. |
| Certificate-generator state subset | Frozen collision convention, labeled room membership, same room, object collision free, and at least 0.5 m wall clearance. This is a stricter data-quality subset, not a new runtime transition. |
| Initial only | Same object pair, canonical failure, absolute pitch at most 30 degrees, two projectable/in-front/visible objects, each area ratio at least `1e-4`, each Projective bbox inside fraction at least 0.50, then official initial dual-target RGB. |
| Terminal success | Any canonical-success state with both objects valid, requested left/right relation true, and relation margin at least 12 px. The state need not equal a sampled target pose. |
| Path observation quality | Official RGB checks initial and terminal dual-target recognizability, no more than two consecutive wall-like frames, and 0.5 m wall clearance on every saved pose. Intermediate frames need not all contain both complete targets. |
| Medium certification | Complete formal forward enumeration finds no canonical success through depth 3; independently replayed certificate first succeeds at step 4, 5, or 6. Reverse depth and old planner proxies are not difficulty labels. |

Projective `inside_frame` is the clipped projected bbox area divided by the raw projected bbox area, at least **0.50 for each object**. It is neither center-only nor full-bbox inclusion. FOV remains distinct at 0.95 plus a 5 percent center margin. The official RGB heuristic's 0.25 patch-inside threshold is also distinct and does not replace the generator's frozen 0.50 Projective initial prefilter. No FOV-only constraint was found in the Projective reverse diagnostic.

Target `min_distance`, target radius/orientation, and equality to a sampled target pose are not applied to predecessor/intermediate states or the Medium lower-bound search.

## Verified Medium inventory

The previous 75 sources / 77 episodes were combined with the independent five-scene screen and v3 A/B, then deduplicated by stable source and episode fingerprints.

| Evidence bundle | RGB-pass input records | Net interpretation |
|---|---:|---|
| Existing development canary | 77 episodes / 75 sources | Base inventory |
| Independent v2 screen | 20 | Twenty new train sources |
| Independent v3 A/B | 6 | Five duplicate positive-control episodes plus one new train recovery |
| **Deduplicated total** | **103 input records** | **96 sources / 98 episodes** |

| Split | Unique sources | Episodes |
|---|---:|---:|
| train | 64 | 65 |
| id_test | 16 | 17 |
| ood_geometry | 1 | 1 |
| ood_instance | 5 | 5 |
| ood_scene | 6 | 6 |
| ood_template | 1 | 1 |
| validation_proxy | 3 | 3 |
| **Total** | **96** | **98** |

All 392 evidence references across 16 unique files exist and match their recorded SHA256. All 98 episodes are runtime PASS, official RGB PASS, and complete lower-bound-4 certificates. First success is step 4 for 71 episodes and step 5 for 27; there is no step-6 episode.

Only the 64 original `train` source identities are train-eligible. The other 32 remain evaluation-only. All seven evaluation scenes have participated in R1 development, so they form a development regression set rather than an untouched final test set.

Train inventory limitations are material:

- ten scenes;
- left/right sources = 36/28;
- train episode first success step 4/5/6 = 46/19/0;
- 37/64 train sources are bed--wardrobe category pairs.

The formal historical evidence remains 73 old RGB-pass one-step rows. The earlier `183` count is unsupported by formal first-success artifacts and remains unknown.

## Frozen projection-frontier experiment

### Inputs and budgets

- Same frozen 16 reverse-cap failures plus four known positive controls.
- Same source rows, success seeds, H1 camera, metric, 12 px margin, collision/layout, action graph, observability, and Medium definition.
- Success seed cap 12; expansion cap 512 per seed; candidate cap 48 per seed; lower-bound cap 100,000.
- Same worker and timing basis for baseline v2 and v4.
- Same-pair only; no replacement.

The new deterministic order remains breadth-first by depth, then ranks already-legal frontier states by object count, in-front count, visible count, minimum bbox inside fraction, minimum clipped bbox area, existing camera-forward alignment to the pair midpoint, and stable state key. It adds no action, camera rotation, snapping, RGB signal, filtering, state-key change, or transition change.

### Generation and validation result

| Stage | Baseline v2 | Projection-frontier v4 |
|---|---:|---:|
| Frozen requests | 20 | 20 |
| Difficulty-certified | 4 | 3 |
| Frozen failure sources recovered | 0/16 | 0/16 |
| Positive controls generated | 4/4 | 3/4 |
| Runtime PASS | 4/4 | 3/3 |
| Official RGB PASS | 4/4 | 3/3 |
| Manual RGB PASS | 4/4 | 3/3 |

All 16 baseline failures were `requested_bucket_not_found_within_budget`. Under v4, 15 remained not found and `id_test:6648` produced three initial candidates and one five-step upper-bound candidate, but exhaustive formal search found a three-step shortcut (`move_forward, move_left, move_backward`), so it was correctly rejected.

The regressed control is `id_test:1030` in `0226_840298`. Baseline collected 576 initials and 22 eligible Medium candidates; the 19th shortcut probe was the complete lower-bound-4 accepted candidate. V4 collected 576 initials and 25 eligible candidates, but all 25 had one- or two-step shortcuts (11 and 14 respectively). Under the frozen candidate cap, projection ordering displaced the previously valid candidate set. This is a generator-search regression, not runtime, collision, renderer, or observability drift; the stored baseline certificate independently remains runtime/RGB PASS.

The failure subset had exactly the same 139 success seeds and 71,168 reverse expansions under both versions. Baseline found zero initial candidate among the 16. V4 found only the three `id_test:6648` initials, with no certified recovery. Sorting legal states by projection quality therefore did not address the dominant reachable-but-ineligible-initial bottleneck.

### RGB review

Official renderer produced seven contact sheets: four baseline controls and three v4 controls. All seven passed the frozen heuristic and manual review. The target pairs are recognizable, paths are not wall-dominated, camera views are not extreme, and motion is natural. This validates the three generated v4 candidates, but there is no new failure-source episode to add to the frozen inventory.

### Cost

- Scheduler wait: 50m17s; worker `RUNNING`: 16m47s (`21:55:07Z` to `22:11:54Z`).
- Structural baseline diagnostic: 9m06.4s.
- Baseline selector phase: 145.0s; v4 selector phase: 148.9s.
- Sum of same-source per-row generation time: 561.11s baseline, 575.60s v4; ratio `1.02583`, below the 1.2 limit.
- Runtime materialization/replay: about 5.4s per variant.
- Official RGB: 81.2s baseline, 55.9s v4; different candidate counts explain the phase difference.
- One H800 was allocated; the serial official renderer used GPU 0. CPU search and renderer ran in the same worker. Renderer exited cleanly and SCO job `pt-txco1a7c` finished `SUCCEEDED`, exit 0.

## Strict frontier gate

| Criterion | Result |
|---|---|
| At least 4/16 failures recovered | **FAIL: 0/16** |
| Recovery spans at least three scenes | **FAIL: 0 scenes** |
| Four controls preserved | **FAIL: 3/4** |
| Every accepted candidate runtime/RGB passes | PASS: 3/3 |
| Cost no more than 1.2x baseline | PASS: 1.026x |

**Decision: `frontier_variant_evidence_insufficient`.** The v4 ordering remains a versioned negative experiment and is not promoted into the frozen generator or inventory. Per the predeclared stop rule, no budget increase, seed increase, `inside_frame` change, or second selector variant was attempted.

## Freezeable support range

### Freeze now

- H1 camera, `canonical_spatial_task_h1_v1`, formal action/collision semantics, 12 px Projective margin, Medium proof definition, and `projective_observability_v1_initial_keyframe`.
- Canonical constraint-scope document and exact component hashes.
- Verified 96-source / 98-episode inventory with train/evaluation isolation and complete lineage/evidence SHA256.
- All existing failure accounting; absence of a generated Medium remains `not found within frozen budget`, never intrinsic unreachability.
- The historical 73-row formal one-step evidence count; `183` remains unsupported/unknown.

### Optional, not promoted

- Projection-frontier v4. It is useful as negative evidence that projection-quality ordering alone is insufficient, but it failed recovery and control gates.
- Further reverse-search optimization. It is no longer an R1 completion requirement; the current Easy/Medium support subsets remain valid even where historical rows do not yield Medium.

### Still unverified

- Generalization to scenes untouched by all R1 selector and threshold work.
- Category-balanced and step-6 Medium generation.
- Recovery of the upstream 298 no-safe-success-region rows or the broader 304 reverse-cap population; finite-budget misses remain explicit.
- Actual v46@250 evaluation in the canonical environment. The historical log shows that step 250 was saved, but the present actor Hugging Face directory is empty.

## Minimal canonical evaluation and training-control design

### Evaluation-only first

Select exactly one deterministic episode per each of the 32 evaluation-only source identities for a frozen development regression manifest. Compare pretrained Qwen2.5-VL-7B-Instruct and v46@250 with paired seeds in the same canonical environment. Record overall and per split/scene/category/difficulty success, first-success step, collision attempts, invalid action, timeout/truncation, premature termination, action mix, pose/score trajectories, and observability/path diagnostics.

This 32-source comparison is a functional development evaluation only. Before model inspection, separately freeze at least 60 identities over at least ten untouched scenes: at least ten each for ID, scene OOD, instance OOD, category OOD, template OOD, and geometry OOD. The present Medium inventory has no category-OOD identity and none of its evaluation scenes is untouched.

Before any evaluation command, restore/resolve and hash both model artifacts. The expected v46 directory exists but contains no current actor weights; the earlier local pretrained cache snapshot is also not currently present. Empty paths are not valid model provenance.

### Minimal training control

Reuse the v46 stable training skeleton exactly; do not add LR/KL/auxiliary sweeps. Because the comparison jointly repairs camera, data, and task/runtime semantics, report it as a system/data repair control rather than a reward-only ablation.

For a controlled pilot, predeclare at least 200 unique train sources across at least 20 scenes, at least 80 identities for each left/right relation, at least 40 verified episodes for each first-success step 4/5/6, and no unordered category-pair family above 35 percent. These are pilot sufficiency criteria, not task semantics.

Relative to the current inventory, the minimum deficits are 136 train sources, ten scenes, 44 left, 52 right, 40 step-6 episodes, and substantial category-pair diversification. Until those are filled, the current 64 train sources support smoke/overfit and task-system validation only.

## Artifacts and SHA256

- `R1_CANONICAL_CONSTRAINT_SCOPE_20260914.json`: `126e21c41b4475b8f18874770b8c3e92131eda4a6da1b21d5b8302a207419007`
- Unified inventory: `3d048d1e7825a1656623d4203ae03d8f75dc6ea9af609c92a34c7a3ba0903cfd`
- A/B `run_config.json`: `f6f179cdbfc894e201429914aaf2c73d38fb7cc63f48000ba4044e976be5a645`
- A/B `aggregate_summary.json`: `bc174ba9d83236b0104c9f4d82a0c5ae208f3f91eef9bac6dd89c82870be0d41`
- v4 selector ledger: `7de67b17930737c9e3d433ed9b04db5184560647f06a088747c7c3bfd1340946`
- v4 runtime replay: `b78bcfcb23cc58443e2ccddf487306873fadacad115ccd2aa73d4b2efce468e4`
- v4 official RGB manifest: `9a994636ad09fa0f490d6d7d90da5829a1c7e55b1f2f5214af438e8e4c3785c8`
- Manual RGB review: `49a9de6d16a7c96086efc21fa7817074c26f880d6f025c01086429984d84d7c9`
- A/B post-close 124-file checksum manifest: `0b7af35e5845949d076589ad30fa71b07e80d7464317fc5a9156980e3aac946f`
- Prior v3/reverse-diagnostic post-close 253-file checksum manifest: `c738b7d5bfd63751cff521a62cb03bc0ee012a5ae294a99a48ebdeaf90178e84`
- Minimal evaluation/training-control design: `4e951aaf4cebd7695634ef4550232dad6e90990f6149f1704064bc460379bcf9`

Both post-close checksum manifests validate without a mismatch. Original experiment directories were not rewritten.

## Next recommendation

End the reverse-frontier search branch here. The next bounded mainline step is to restore and hash the pretrained and v46@250 model artifacts, freeze a one-episode-per-source 32-identity development evaluation manifest from the evaluation-only inventory, and run the paired canonical **evaluation-only** comparison. Do not train until the stated train-only coverage deficits are filled, and do not claim final generalization until a scene-disjoint untouched evaluation set is frozen.
