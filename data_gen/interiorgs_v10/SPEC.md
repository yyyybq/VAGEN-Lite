# InteriorGS v10 task generator — shared spec

## Goal

Rebuild the InteriorGS Active Spatial data generator so that **≥ 90% of the 1,000
InteriorGS scenes (≥ 900) yield usable tasks and the dataset holds ≥ 10,000 RL
tasks** (each RL task also yields an SFT trajectory and several QA states).
The v1 release admitted only 90 scenes / 100 tasks.

## What went wrong in v1 (measured on the cluster, 2026-10-10)

```
1000 scenes
 ├─ 117  coordinate convention "ambiguous" → never generated
 └─ 883
     ├─   8  per-scene 180 s timeout
     ├─ 140  no admissible object pair  (87: <2 objects after filters, 53: pair rules)
     ├─ 501  pairs exist but no candidate (498 found NO admissible endpoint)
     └─ 234 scenes / 319 candidates
          → runtime replay 230 / 314 → same-source isolation 211 / 288
          → camera legality 211 / 286 → RGB semantic review 90 / 100
```
* Endpoint search: 4,914 failed pair searches; 4,910 had no admissible endpoint
  (75 no legal pose; 4,835 scored but failed the joint gate). Gate-level counts
  were never logged.
* Object filter: 40,711 objects pass single-object geometry filters, only 7,560
  survive "category unique in the whole scene". 84 of the 87 "<2 objects"
  scenes recover by dropping uniqueness alone.
* v9 endpoint sampler: ≤ 12 pairs/scene, ≤ 64 endpoint positions/pair, yaw = face
  pair midpoint ± 10°, reverse search ≤ 256 expansions depth ≤ 8, keeps the FIRST
  path found, stops at 2 candidates/scene, no regeneration after review failure.
  Result: every admitted trajectory is a single action.
* Semantic review required both targets identifiable in EVERY frame; QA states
  were admitted only if their parent RL task was admitted.

## Decisions taken by the user (binding)

1. Initial frame of RL/SFT episodes: **at least one target must be identifiable**;
   the other may be cropped, occluded or out of view. Terminal frame: both
   targets identifiable. Initial state must be legal and not already successful.
2. Duplicate categories are allowed: **disambiguate with spatial referring
   expressions** (room, nearest unique anchor, size, ...). Exclude an instance only
   if no unambiguous expression exists.
3. **Expand task families** beyond Projective. New families need canonical
   judges first (registered under `canonical_spatial_task_h1_v2`; v1 Projective /
   FOV behaviour must stay bit-identical).
4. Scene data will come either from Hugging Face (`spatialverse/InteriorGS`,
   metadata only: labels.json, structure.json, occupancy) or from the cluster copy
   at `/mnt/umm/users/yinbaiqiao/InteriorGS`. Code must not assume either path.

## Conventions

See `contract.py` (camera, action protocol, Obj, Funnel, task-row keys) and
`scene.py` (Scene API). Camera height 1.5 m, pitch 0, H1 intrinsics, 256×256,
6 strafe actions, 0.3 m / 20°, ≤ 12 primitive steps. Use
`sampling.projective_item / fov_item / score` for the frozen v1 metric.

Geometry notes worth knowing:
* H1 effective intrinsics at 256×256: fx = 128, fy ≈ 170.7 → HFOV 90°, VFOV ≈ 73.7°.
  With pitch 0 at 1.5 m, the floor is visible only beyond ≈ 2.0 m, so low objects
  closer than ~2 m lose their lower part and can fail `inside_frame ≥ 0.5`.
* `_project_object` uses the 8 AABB corners; corners behind the camera are dropped.

## Module ownership (one owner per file; do not edit files you do not own)

| file(s) | owner workstream |
| --- | --- |
| contract.py, sampling.py, SPEC.md, testing/synthetic_scene.py | lead (read-only for others; ask lead for changes via your report) |
| scene.py, coordinates.py, scene_identity.py, fastgeom.py | G — scene geometry |
| objects.py, referring.py | O — objects & referring expressions |
| endpoints.py, occlusion.py | E — endpoint sampling (Projective + FOV) |
| families/*.py, canonical_v2_patch.md | F — new task families |
| difficulty.py, initial_states.py | D — initial states & difficulty |
| admission.py, release.py, qa_states.py, configs/*.json | A — admission, QA, release |
| run_scene.py, run_all.py, sco_*.sh | lead (integration) |

Tests: `tests/test_<your_module>.py`, stdlib `unittest` (pytest is NOT
installed), run with `python3 -m pytest tests` (the generator's tests are in `tests/datagen/` since the 2026-10 cleanup)
from the repo root. Use the synthetic scenes; never require network or GPUs.
Code must run on Python 3.10 (cluster) — avoid 3.11+ only syntax/APIs.
Every rejection must be counted in a `Funnel` with a stable reason string.
