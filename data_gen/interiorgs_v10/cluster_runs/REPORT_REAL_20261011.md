# v10 generator on real InteriorGS (fujian, 2026-10-11)

Outputs: `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_real_20261011/`
(`calibration_report.json`, `object_stats.json`, `pilot/`).

## 0. Unblocking data access

* Hugging Face was not needed. The full dataset is in AOSS at `fj:s3://baiqiao/InteriorGS`
  (1,000 scene folders). `fetch_interiorgs_metadata.py --source aoss` (new) pulled only
  `labels.json`, `structure.json`, `occupancy.json` and `occupancy.png` for every scene into
  `/mnt/umm/users/yinbaiqiao/InteriorGS_meta`: 4,000/4,000 files, 329 MB, about 30 s, no PLY.
* The path in the handoff, `/mnt/umm/users/yinbaiqiao/InteriorGS`, holds only 100 scenes, owned
  by root with mode `drwx------`, so user `yinbaiqiao` cannot read them.
* AOSS hangs through the login node's `http_proxy`. The internal endpoint answers directly, so
  the fetcher drops the proxy variables.
* `aoss_client` is installed only in the `probe-spatial` and `masked_visual_actions` envs, not in
  `vagen-lite`. The fetch runs with `probe-spatial`; the generator itself needs only the local files.
* On the Windows host, `ssh fujian` fails with "Failed to connect to new control master"
  (ControlMaster is set in `~/.ssh/config`). `ssh -o ControlMaster=no -o ControlPath=none fujian` works.

## 1. Tests

`python -m unittest discover -s data_gen/interiorgs_v10/tests -t .` on the cluster (Python 3.12.13,
numpy 2.2.4): **92 passed** in 58 s.

## 2. Real data format vs `scene.py`

| field | assumption | real data |
| --- | --- | --- |
| labels.json | list of `{ins_id, label, bounding_box: 8×{x,y,z}}`, Z up | matches. 28 to 19,300 instances per scene (median 294); 712 distinct labels |
| structure.json | `rooms[].profile`, `holes[].type` | top-level keys are `rooms`, `holes`, `walls`; holes are WINDOW 6,273 / DOOR 5,716 / HOLE 885 / OPENING 784 |
| room names | `name` / `type` / `room_type` | **missing**: rooms have only `profile` (6,607 rooms). Room-based referring expressions never apply (`room:room_unnamed`) |

1 to 20 rooms per scene (median 7). All 1,000 scenes parse; no load failures.

## 3. Audits (all 1,000 scenes)

Coordinate convention:

| | frozen | ambiguous |
| --- | --- | --- |
| v1 | 883 | 117 (same as the v1 measurement) |
| v2 | **987** | **13** (insufficient_evidence 6, counter_evidence 7) |

v1 → v2: frozen→frozen 883, ambiguous→frozen 104, ambiguous→ambiguous 13. Of the 883 v1-frozen
scenes, v2's own evidence agrees on 855 and abstains on 28; **0 conflicts**. All 104 recovered scenes
are frozen at sign +1, and each needs `patches/env_structure_override.patch` in the RL env.
Ambiguous scenes: 0024_840020, 0037_839879, 0040_839882, 0067_839906, 0194_840128, 0407_840140,
0420_840007, 0432_839899, 0459_840528, 0462_840762, 0617_841196, 0632_839937, 0826_841622.

Objects:

| preset | candidate objects | pairs selected | scenes with <2 objects | scenes with no admissible pair |
| --- | --- | --- | --- | --- |
| v9_strict | 10,952 | 4,922 | 67 | 182 |
| v10_relaxed | **20,950** | **36,249** | **46** (0 objects: 33, 1 object: 13) | **54** |

Referring methods under v10_relaxed: anchor 10,883, unique_label 9,427, size 640, room 0.
In the 46 scenes the loss is almost entirely `reject_no_ref`. These are non-residential venues
(kart track, cinema with 94 identical sofas, banquet halls with 126–140 chairs, gym, amusement park,
car showroom) where every object has duplicates and no unique anchor lies within range. This is a
property of the data, not a bug.

**Ceiling on scene coverage**: 1000 − 13 ambiguous − 33 with 0 objects + 2 counted in both = **956**.

## 4. Pilot (first 30 scenes, 8 workers, login node)

| | value |
| --- | --- |
| scenes with tasks | **29 / 30** (the one without is 0024_840020, convention ambiguous) |
| scenes at 14 tasks | 29 |
| tasks | **406**: projective 187, absolute_positioning 92, screen_occupancy 78, equidistance 49 |
| d* | 1:29, 2:51, 3:36, 4:84, 5:49, 6:24, 7:117, 8:16 |
| time per scene | mean 110 s, median 115 s, max 293 s (budget 600 s); 561 s wall for 30 scenes at 8 workers |

Funnel highlights:
* Projective items: 850 built (780 more dropped over budget); 335 got an endpoint. The ones
  without failed on occlusion 417, no candidate pose 66, inside_frame 20, margin 12. The
  generator-side AABB occlusion gate remains the main blocker, as on synthetic scenes.
  It does not limit yield: every usable scene reached its target.
* New families with ≥1 endpoint: absolute_positioning 124/208, screen_occupancy 125/208,
  equidistance 53/202, occlusion_alignment 23/174. occlusion_alignment produced 0 episodes,
  because the scene quota fills from the other families first.
* Initial states: 394 accepted. Rejections: diversity_too_close 122,751,
  no_target_identifiable 106,753, already_success 43,355.

**Projection for 1,000 scenes**: about 940 scenes × 14 ≈ 13,000 tasks, against targets of
≥ 900 scenes and ≥ 10,000 tasks. Compute is about 30 core-hours, so about 2 h wall on one
`N4lS.Iq.I80.1` worker (14 vCPU, the smallest spec; zoetrope has no CPU-only spec).

## 5. Fixes in this round

* `fetch_interiorgs_metadata.py`: new `--source aoss` backend (proxy-free, resumable, reports
  missing labels/structure).
* `sco_interiorgs_v10.py`: `SCENES_ROOT` pointed at the 100-scene unreadable directory. It now
  defaults to `InteriorGS_meta`, is overridable with `--scenes-root`, and `prepare` refuses a
  root with no `*/labels.json`.
* README and CLUSTER_RUN_HANDOFF: data path, AOSS route, ssh flags.

No generator thresholds were changed: the pilot met its targets without retuning.

## 6. Full run (sco, approved by the user)

| tag | job | result | cause and fix |
| --- | --- | --- | --- |
| 20261011 | pt-wx2ptic0 | "SUCCEEDED" in 2 min, but **all 1000 scenes `error`** and `workers=1` | (a) the source archive contained only `vagen/envs/active_spatial`, whose `__init__` imports `vagen.envs.gym_image_env` → `ModuleNotFoundError` in `load_scene`. The pilot ran from the full repo, so it did not catch this. Fix: archive the whole `vagen/` package (7 MB). (b) `nproc` ran after `export OMP_NUM_THREADS=1`; GNU nproc honours that variable and returned 1. Fix: count CPUs before the export. |
| 20261011b | pt-767pwueo | stopped after ~6 min (SUSPENDED) | inside the job `nproc` = 128 (whole host), but I80.1 grants a CFS quota of 14 vCPU, so 127 workers ran about 9× slower. Scenes that normally finish in ~20 s took 170 s, and heavier ones would have run into the 600 s wall-clock scene budget. Fix: cap workers by `/sys/fs/cgroup/cpu.max`, and pass `--workers 13` explicitly for this run. |
| 20261011c | pt-snyv3zp4 | **SUCCEEDED** 09:25 → 11:02 UTC (1 h 37 min), `cpus=14 workers=13` | 1 scene crashed (`KeyError` in the occlusion memo, fixed below) and was rerun on the login pod |

`launch.sh` now also loads one scene from the packaged source before the audits (set -e aborts
the job if that fails), exits non-zero when no scene produced tasks, and chowns the run directory
back to the submitting user on exit. The job runs as root, so the 20261011 and 20261011b output
directories are root-owned and cannot be removed by `yinbaiqiao`.

Note: the login pod is limited to 14 CPUs as well (same `cpu.max`), even though `nproc` says 128.

### Results (all 1,000 scenes)

Output: `exps/vagen_active_spatial/interiorgs_v10_gen_20261011c/output/` (`tasks.jsonl` 40 MB,
`summary.json`, `per_scene.csv`, `env_structure_overrides.json`, `scenes/<id>.json`).

| | result | target |
| --- | --- | --- |
| scenes with tasks | **938 / 1000 (93.8%)** | ≥ 900 |
| tasks | **13,109** | ≥ 10,000 |
| scenes at 14 tasks | 932 (6 more have 5–13) | |

* By family: projective_relations 6,632, absolute_positioning 2,730, screen_occupancy 2,317,
  equidistance 1,430. occlusion_alignment has 0 (524 of its items have endpoints, but the
  14-task quota fills from the other families first).
* d*: 1: 949, 2: 1,779, 3: 1,074, 4: 2,753, 5: 1,476, 6: 757, 7: 3,754, 8: 567.
  v1 had only d* = 1.
* Time per scene (ok): mean 77 s, median 67 s, p95 157 s, max 631 s; 98 items were skipped
  by the scene time budget across all scenes.
* The 62 scenes without tasks:
  * 13 have an ambiguous convention.
  * 31 have no candidate object.
  * 16 have objects but no item reached an endpoint (1–22 objects; mostly the occlusion gate).
  * 2 have endpoints but no legal initial state.
* 104 scenes need the env override (`env_structure_overrides.json`); see "Before training".

### Bug found by the full run

`endpoints._blocked_rows` cleared its per-target occlusion memo once it held more than 200,000
positions, and did so before reading back the rows of the current call. The first call to cross
the threshold therefore raised `KeyError` (scene 0463_840754, 1 of 1000). The fix reads the
rows first and trims afterwards; the limit is now `_OCCL_MEMO_MAX`. New regression test
`test_occlusion_memo_reset_keeps_results` reproduces the `KeyError` on the old code and passes on the
new one (suite: 93 tests OK). The fix changes only cache lifetime, not values, so the other 999
scene results stand. 0463_840754 was rerun with the fix and gave 14 tasks; the error record is kept as
`output/0463_840754.error_before_memo_fix.json`.

## Open items (not blocking the 10k / 900 targets)

* Room types could be inferred from contents (bed → bedroom, toilet → bathroom, ...) to bring back
  the `room` referring method; that needs a visual check before use.
* Re-tune or GS-verify the occlusion gate. It would help the projective share and occlusion_alignment.
* GPU stage (render + VLM review) and `admission.admit` have not run on real data yet.
