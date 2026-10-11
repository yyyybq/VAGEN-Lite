# InteriorGS v10 task generator

Goal: ≥ 900 of 1,000 InteriorGS scenes usable and ≥ 10,000 RL tasks (each with
an SFT trajectory and QA states). v1 admitted 90 scenes / 100 one-step tasks.
`SPEC.md` records the v1 funnel, the binding decisions and module ownership.

## Pipeline

```
scene dir (labels.json, structure.json)
  └ scene.load_scene ── coordinates.calibrate_structure_y_sign (v2: wall/door/corner evidence)
      └ objects.candidate_objects (v10_relaxed) ── referring.assign_referring_expressions
          ├ objects.select_pairs ─ Projective items (A left/right of B)          [frozen v1 judge]
          └ families.* tuple/param menus ─ absolute / screen_occupancy /
                                           equidistance / occlusion ... items  [canonical v2 judges]
              └ endpoints.sample_endpoints (Projective/FOV, gate-instrumented, occlusion gate)
                families.<f>.success_pose_candidates (new families)
                  └ initial_states.episodes_for_scene
                        reverse exploration → difficulty.certify_d_star (exact BFS, runtime collision rule)
                        initial: legal, not success, ≥ 1 target identifiable; terminal: both identifiable
                        quotas over d* bands 1 / 2–3 / 4–6 / 7–8
                      └ run_scene._task_row  →  run_all (process pool, resumable, sharded)
                          └ GPU stage on the cluster: render + VLM review (admission.review_requests)
                              └ admission.admit (RL / SFT / QA decided separately)
                                  └ release.build_release (scene-group splits, SFT export, QA, registry,
                                                           regeneration plan for short scenes)
```

## Run

```bash
# metadata only (labels/structure/occupancy, no PLY). On fujian the full 1,000-scene mirror is
# already at /mnt/umm/users/yinbaiqiao/InteriorGS_meta, fetched from AOSS with the probe-spatial
# env (it has aoss_client); /mnt/umm/users/yinbaiqiao/InteriorGS holds only 100 root-only scenes.
.conda/envs/probe-spatial/bin/python -m data_gen.interiorgs_v10.fetch_interiorgs_metadata \
    --source aoss --out /mnt/umm/users/yinbaiqiao/InteriorGS_meta --workers 16
python -m data_gen.interiorgs_v10.fetch_interiorgs_metadata --out /data/InteriorGS_meta --list-only  # Hugging Face

# audits that need no rendering
python -m data_gen.interiorgs_v10.coordinates --scenes-root ROOT --out calibration_report.json
python -m data_gen.interiorgs_v10.objects     --scenes-root ROOT --out object_stats.json

# generation (resumable; --shard/--num-shards to split across jobs)
python -m data_gen.interiorgs_v10.run_all --scenes-root ROOT --out OUT --workers 32 --expected-scenes 1000
python -m data_gen.interiorgs_v10.run_all --out OUT --aggregate      # summary.json, per_scene.csv, tasks.jsonl

# optional features (all off by default; off == the 20261011c generation)
#   --reserve-k 16                       reserve episodes per scene -> tasks_reserve.jsonl
#   --family-quotas "projective_relations:2:6,absolute_positioning:2:4,..."   per-scene family min:max
#   --terminal-identifiable family       occlusion_alignment: only the occluder must be identifiable at the end
#   --motion-model horizon_locked --pitch-deg -15   EXPERIMENT ONLY: the RL env implements r1 (pitch 0) only
#   --diagnostics                        per-item initial-state search reports in the scene summaries

# read-only diagnostics of a finished run, and run-vs-run comparison
python -m data_gen.interiorgs_v10.diagnose_run --out OUT [--scenes-root ROOT --r1-frozen DIR --workers 10]
python -m data_gen.interiorgs_v10.diagnose_run --compare base=OUT1 new=OUT2 --reference OUT1
# Projective / FOV items with a success pose at fixed pitches (0, -10, -15, -20, -30)
python -m data_gen.interiorgs_v10.pitch_audit --scenes-root ROOT --only S1 S2 --workers 10

# cluster job (CPU generation; prepare does not submit)
python data_gen/interiorgs_v10/sco_interiorgs_v10.py prepare --tag <tag>
python data_gen/interiorgs_v10/sco_interiorgs_v10.py submit  --tag <tag>

# tests (stdlib unittest)
python -m unittest discover -s data_gen/interiorgs_v10/tests -t .
```

## Before training on v10 rows

1. Apply `families/canonical_task_metrics_v2.patch` (new families in the runtime metric) and
   follow `families/ENV_INTEGRATION.md`.
2. Apply `patches/env_structure_override.patch` so rows whose scene was resolved only by the
   v2 coordinate calibration (`collision_convention.requires_env_override`) load with that sign.
3. The env evaluates success at the end of each assistant turn (env.py 1113–1161), not after
   every primitive; certificates end exactly at success and SFT turns are chunked so that the
   last turn ends there.
4. The SFT export omits the "Current camera pose" text the RL prompt shows; align one side
   before mixing SFT and RL.

## Status (2026-10-11)

* All modules unit-tested on synthetic scenes (`tests/`), including exactness checks of the
  vectorised geometry against the runtime CollisionDetector and of d* against brute force.
* Real InteriorGS, all 1,000 scenes (fujian, sco job pt-snyv3zp4): **938 scenes with tasks,
  13,109 tasks**, d* 1–8. Calibration v2 freezes 987 scenes (13 ambiguous). Output in
  `exps/vagen_active_spatial/interiorgs_v10_gen_20261011c/output/`; details in
  `cluster_runs/REPORT_REAL_20261011.md`. Rendering + VLM review (GPU stage) has not run yet.
* Real `structure.json` rooms carry only `profile` (no name/type), so room-based referring
  expressions never apply.
* `fetch_interiorgs_metadata.py --source aoss` is tested on the cluster; the Hugging Face path is
  still untested.
