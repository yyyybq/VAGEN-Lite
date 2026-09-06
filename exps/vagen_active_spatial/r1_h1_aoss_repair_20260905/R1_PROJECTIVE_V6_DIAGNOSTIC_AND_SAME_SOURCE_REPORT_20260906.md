# R1 Projective v6 Diagnostic and Same-Source Report — 2026-09-06

## Donor lifecycle and affected rows

The donor allocator v2 records reserved -> validation_pending -> committed and releases generation failure, planner unverified, timeout, and RGB rejection. Scheduler recovery is owner/attempt based and retains pending validation rather than blindly releasing it. The five allocator lifecycle tests and 18 repair-pipeline tests pass.

The historical isolated ledger contained 10 reservations. The final comparison identifies 6 stale owners (the timeout/released rows) and 4 live replacement owners. Stage events show the parallel donor conflict affected id_test:6634, which skipped candidates from source rows 6602 and 6606. A fixed owner-order diagnostic (id_test:6629 before id_test:6634) generated and official-rendered both rows successfully. New ledger uniqueness is true; no cross-split donor was introduced.

## 16-row diagnostic

Frozen selection: 5 prior accepted positives, row id_test:6629 wall-clearance negative, 5 generation failures, and 5 generation timeouts. Generation closed all 16 rows: 6 hard failures, 5 wall-clock timeouts, 2 strict same-pair repairs, and 3 replacements. All 5 generated candidates reached runtime/reachability. Five timeouts stopped at planner_start; no row was relabeled unreachable.

Stage accounting: 16 candidate searches, 1,383 initial states evaluated, 6 geometry-prefilter passes, 1,024 target points evaluated, and 2 donor candidate-conflict skips. The targeted donor-order follow-up gave 2/2 generation, reachability, runtime, and official RGB passes. The saved positive reachability certificates independently replayed 5/5.

## Targeted implementation changes

- Existing wall-clearance/layout validation is passed into planner state expansion; no frozen threshold was changed.
- Stage events now persist candidate, initial, planner, retry, cost, and timeout stage data.
- Donor lock acquisition retries transient EAGAIN; scheduler recovery is explicit.
- Renderer/process errors are not RGB verdicts and keep validation_pending; a missing audit manifest cannot release a donor.
- Selection, source mapping, job outputs, donor ledger, and audit outputs are versioned.

## 71-row same-source rerun

The frozen 71-row serial rerun used the complete nine-manifest source map, one worker, 180-second generation timeout, existing 2k/25k planner budgets, and no 250k escalation. Accounting is closed:

| stage | count |
|---|---:|
| source rows | 71 |
| generation completed | 61 |
| generation timeout / unverified | 10 |
| hard generation failure | 55 |
| strict same-pair repair | 2 |
| replacement | 4 |
| reachable and runtime-consistent | 6 |
| official RGB observability pass | 6 |
| final accepted | 6 |

Compared with aggregate_v5, the only net recovery is id_test:6629 (old final false -> new final true; path upper bound 8 -> 11). There are no final-row regressions. All six new poses have matching official-render evidence. The old bad frames remain historical; they were not reused as new evidence.

## Difficulty and coverage

The accepted set is still small (6/71 = 8.45%). Replacement rows remain systematically easier in several dimensions: accepted repaired translation values include 1.8–5.08 m, yaw 0–126°, and three replacement paths are one-step; source planner proxies for replacement rows are mostly 12. The paired artifact shows bbox and relation margins also shift upward for replacements. Therefore this run demonstrates an explainable donor/scheduling recovery, not solved difficulty preservation.

## Real RGB/path review

The six generation candidates were rendered serially through dedicated renderer 8876; all six passed projective_observability_v1. No renderer error or continuous-near-wall rejection remains in this final audit. Renderer 8877 was not touched.

## Files and hashes

Core artifacts:

- .../projective_v6_small_sample_v1/diagnostic16_selection.json
- .../diagnostic16_v1/positive_certificate_replay.json
- .../diagnostic16_v1/run_v5/diagnostic_stage_stats.json
- .../diagnostic_donor_pair_v1/official_render_audit_results.json
- .../full71_same_source_v3/aggregate_final/summary.json
- .../full71_same_source_v3/aggregate_final/sample_manifest.jsonl
- .../full71_same_source_v3/row_comparison.json
- .../full71_same_source_v3/difficulty_comparison.json
- .../full71_same_source_v3/donor_impact.json

SHA256: final summary 8ab20295df6b336408ac8ba4a2edacb9a2ec15996f2570381d77f48f845557eb; final manifest 1b2e2031b0e3f8ed402bd4d8a42a2aec25444dc743440a32e519023d8e219d8e; donor impact 6f3dc71d1f15c0c36c45cfa81897d0f27c4fcc54268f679d58db7117162a27f3; diagnostic selection 96868a3d711bcfdafed9f5b8e5086c26ef65026faeb1248a6206ffaeb3862237.

## Remaining risks and gate

The 71-row generation bottleneck remains dominated by hard layout/collision failures (55) and planner-start timeouts (10). Replacement difficulty matching is not yet adequate, and the requested full 10-scene canary has not been rerun. This is not sufficient evidence for 100-scene regeneration.

**BLOCKED — projective coverage/difficulty and the full 10-scene canary remain incomplete.**

Exact next step: use the persisted stage counters to improve joint initial/target candidate construction and difficulty-bucket donor matching, then rerun the same frozen 71-row comparison (not full regeneration) before reconsidering the 10-scene canary.
