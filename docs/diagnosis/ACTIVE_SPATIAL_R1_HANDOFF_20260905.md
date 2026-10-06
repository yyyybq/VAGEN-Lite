# Active Spatial R1 Handoff (2026-09-05)

> Historical repair snapshot. Later frozen Projective data and a support-subset
> experiment now exist; see the [2026-10-02 takeover audit](active_spatial_takeover_20261002/REPORT.md).
> This does not retroactively clear the full-corpus or dense-score pilot gates below.

## 2026-10-03 task-context repair acceptance (in progress)

- Historical R1 renderer `pt-t7bb75de` expired its 48-hour lease at 02:35 UTC; trainer `pt-kbvfmyih` failed on renderer connection errors at 02:36 UTC. Latest rollout: 213; latest complete checkpoint: 200. Preserve the original run as a **missing-task-context fault baseline**; do not resume or hot-patch it.
- User authorized a new isolated repair, one-update acceptance, then paired evaluation. New run: `exps/vagen_active_spatial/R1-task-context-v1-acceptance-20261003`. Its package is derived from the frozen v7 archive with explicit minimal patches; unrelated dirty workspace changes, reward changes and VERL changes are excluded.
- Both environment observation paths now repeat the public task each turn. The PPO inference boundary checks the exact image-expanded token IDs. Old full turns may be evicted; a current observation that cannot fit fails closed rather than slicing task/image tokens. Generic evaluation and the canonical dev runner enforce task retention too.
- Final local regression: all 72 `tests/active_spatial` tests passed (23.51 seconds), including the real Qwen processor checking image expansion, whole-turn eviction and rejection of an undersized current-only prompt, SFT task retention, actual evaluation token checks and six paired-report failure tests.
- Frozen acceptance: 12 train episodes from one existing train scene, no overlap with the fixed 32-task development evaluation's seven scenes. One actual actor/critic PPO update, finite gradients, changed tensors, complete export and real checkpoint reload must pass. Every actual rollout input must retain its task; action parsing and strict-format rates must each be at least 95%.
- Renderer `pt-goa1zokk` uses one H800. Runtime replay passed 12/12 with pixel-identical initial RGB and task retention throughout certificate execution. Only after this PASS, one-update trainer `pt-uls7be48` was submitted. Worker leases are bounded; the training entry never launches formal 250-step training. Base/step-1 paired dev evaluation is gated on successful acceptance, and is not an independent generalization or training-benefit claim.
- Latest evidence and status belong in the new run directory; GPU acceptance and paired results were **not yet complete at this entry's creation**. Do not infer PASS from submission or CPU tests.

## Recovered State

- Source repository: `/mnt/umm/users/yinbaiqiao/VAGEN-Lite`; R1 was developed from HEAD `5c582b936c6422fdfc6c2d89172aaa34cf6cccd5` and integrated after the independent evaluation-recovery commit `fa55c439`.
- The source repository's pre-existing evaluation-recovery changes were verified and committed separately. No reset, clean, stash, source-manifest overwrite, training, checkpoint restore, or v46 replay was performed during R1 integration.
- H0 (unscaled native intrinsics at resized resolution) remains rejected by real RGB. H1 is native-to-render resize scaling.

## H1 Integration

- `build_canonical_camera` now requires an explicit `camera_model_version`; every R1 caller passes `canonical_camera_h1_resize_v1` explicitly.
- Generator rows and mappings record `camera_model_version`, `canonical_task_metric_version`, and `generator_version`.
- Deterministic camera tests: 10/10 passed. Layout/version tests: 7/7 passed.
- Real-render regression on scene `0267_840790`: native render resized to 256 versus direct H1 256 render MAE `4.9125518798828125`, RMSE `11.819524765014648`; threshold 6, PASS.

## FOV Root Cause and Repair Pilot

- Root cause: **B4 multiple issues**. The historical score has a `0.7` floor when both objects are visible while the threshold is `0.65`, so truncated views can pass; the original initial-pose generation also commonly places both objects in view.
- Historical H1 audit: 1503 total, target success 1502, initial success 1503, truncated 835, H0/H1 disagreement 1159.
- Versioned layout-gated pilot on the only asset-backed scene: 28 source rows, 15 accepted, 13 explicit failures.
- Accepted rows: target success 15/15, initial success 0/15, initial and target constraint success 15/15. Minimum target center margin is `67.3163 px`; minimum inside-frame fraction is `0.96103`.
- Independent repository layout audit kept 15/15. It reports two non-filtering `collision_after_one_step` diagnostics.

## Projective Failure Resolution

- The old target pose is on the wrong relation side; canonical truth is now evaluated from the final H1 projection, not legacy region metadata alone.
- The v4 generator adds bounded target search, explicit 12 px margin, initial-state repair, room/wall/object gates, lineage, retry accounting, and an explicit failure manifest.
- Layout-gated pilot on scene `0267_840790`: 69 source rows, 14 accepted, 55 explicit failures.
- Accepted rows: target success 14/14, initial success 0/14, initial and target constraint success 14/14. Minimum relation margin is `19.7770 px`.
- Failure classification: 45 bounded layout candidates exhausted; 10 source pairs are infeasible under the 12 px margin and in-frame gates.
- Independent repository layout audit kept 14/14. It reports two non-filtering `collision_after_one_step` diagnostics.

## Projective Full Regeneration

- The older projection-only v3 diagnostic produced 1919/2003 train repairs and 89 unresolved rows across train/ID/OOD. It is superseded and is **not formally usable**, because it did not enforce layout gates.
- Full v4 regeneration correctly requires `--gs-root` and will not silently fall back to projection-only mode.
- Full v4 files were not generated because the available GS root contains only scene `0267_840790`. Coverage is 69/2003 train projective rows and 28/1503 train FOV rows; validation ID and all five OOD projective splits have zero asset coverage.

## Real-Render Validation

- The designated renderer node `10.119.31.101` was used with one H800 and the official HTTP renderer. The temporary renderer was stopped and all eight GPUs returned to zero memory use.
- Camera H1 regression passed as described above.
- Existing shared projective script rendered 24 paired old/new rows: old canonical relation 0/24, new 24/24. Existing FOV audit rendered 28 target overlays. Across 52 images, minimum RGB standard deviation was `30.3843`, mean `63.2458`; all were nonblank.
- These RGB sheets validate the camera convention and expose the legacy relation/FOV problems. They do **not** validate the final layout-gated v4/v2 repaired rows, because transfer of those artifacts to the node was not authorized.

## Files/Artifacts and SHA256

- Durable artifact root: `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_layout_repair_20260905`.
- Complete 40-file inventory: `sha256_inventory.json`, SHA256 `d8a4e625fa994d26932bd1eba70024fbf01fec3c844c8057143cdd956221ccad`.
- Asset coverage: `asset_coverage.json`, SHA256 `a17826a1b83014367c46a3f70e6576e1aeb0118ee1a5ccde8a7a3f3ab3431822`.
- Projective v4 pilot: `projective_v4_final/r1_projective_repair_pilot_h1_v4_layout_gated.jsonl`, SHA256 `5cb23e53185d443ef3fc7e6323633bdcdf169ac97729a81716be7a18cfea569c`.
- FOV v2 pilot: `fov_v2_final/r1_fov_repair_pilot_h1_v2_layout_gated.jsonl`, SHA256 `77fc7815e4bd243257c1d1990be86c24bbb3e9f44c1bbcaad0318dfd781d7e9a`.
- Camera metrics: `real_render/metrics.json`, SHA256 `dd18ecfa4872649c69eda20b86bc71569cb74030913d5019e4071094d9bf39e4`.
- Projective RGB sheet: `real_render/paired_contact_sheet.png`, SHA256 `d10faafd26d802989a16dc6192b12713537a40f6be1b3c0e8272ce78499359cb`.
- FOV RGB sheet: `real_render/contact_sheet.png`, SHA256 `4a2d00541b47816e630ee94d4f459b6a87c7a305404a6ff8358b23bfb7148bcf`.

## Remaining Risks

- The designated node lacks the GS assets needed for train, ID, validation, and five OOD split coverage.
- Final layout-gated repaired rows have not received paired real-render validation.
- Pilot coverage is only one scene, and 55/69 projective plus 13/28 FOV rows remain explicit failures under strict gates.
- `collision_after_one_step` remains a diagnostic rather than a hard rejection gate.
- Occlusion still has no true visible-fraction oracle and remains on the existing projection/depth proxy.

## Gate Decision

**BLOCKED — DATA/CAMERA/METRIC REPAIR INCOMPLETE**

## Exact Next Step

Mount or provide the complete GS `structure.json`, `labels.json`, and render assets for every scene referenced by train, validation ID, and the five OOD manifests on `10.119.31.101`. Then run layout-gated v4 projective and v2 FOV regeneration per split, perform paired real-render sampling of the generated rows, and re-run canonical/layout/failure/SHA audits. Do not start v46 shadow replay or training until those checks pass.
