# Active Spatial dense-score S0/S1/S5 pilot

Final status: `PILOT_BLOCKED_INFRA`  
Blocker class: `R1_FORMAL_TRAINING_DATA_GATE`

No renderer or trainer job was submitted. No rollout, fixed-batch backward,
optimizer step, checkpoint creation, S0/S1/S5 training, or canonical model
evaluation was run.

## Blocking decision

The latest completed formal R1 training gate is
`r1_clean_training_corpus_v1_20260919/minimal_pilot_readiness_gate.json`. It
contains `passed: false`. Its companion report explicitly concludes:
`BLOCKED — do not submit clean retraining`.

The provisional repaired corpus has 241 unique sources across 21 scenes and
passes the left/right counts, but fails three frozen pilot criteria:

| Criterion | Required | Observed | Result |
| --- | ---: | ---: | --- |
| first-success step 5 | at least 40 | 34 | FAIL |
| first-success step 6 | at least 40 | 5 | FAIL |
| largest category-pair fraction | at most 0.35 | 0.377593 | FAIL |

The 5,889-row mixed manifest is explicitly described by the frozen R1 report
as an audit artifact rather than a safe clean-retraining manifest because it
contains 5,648 unchanged historical non-canonical rows and only 241 repaired
canonical rows.

## Why the 2026-09-28 artifacts do not clear the gate

The newest R1 material is a submission package for Phase A, pair-universe
enumeration, and Phase B. It is not a completed frozen corpus:

- Phase A contains only a code archive, worker launch script, and submission
  request; it contains no generated output or completion ledger.
- Pair-universe contains only code archives and worker launch scripts.
- `r1_fullscale_canonical_expansion_phase_b_scope_v1_20260928/` is absent.
- `r1_fullscale_canonical_expansion_phase_b_v1_20260928/` is absent.
- Phase-A `fresh_sources.jsonl` is a generation request and is forbidden from
  being substituted for a formal training manifest.

Using the September 19 candidate, the old historical manifests, or the
September 28 request scope would violate the explicit no-fallback rule.

## Gates that were checked

- r5 historical replay: PASS; the machine report still says
  `REAL_HISTORICAL_REPLAY_PASS`.
- custom replay numeric turn ordering: PASS; UUID group IDs are factorized and
  trajectory/turn IDs remain numeric.
- r5 regression test: PASS, `12 passed` in 29.90 seconds.
- provisional train/evaluation identity isolation: PASS for the September 19
  candidate only; canonical development source overlap is empty, and permanent
  local-action source and scene overlap are empty.
- frozen canonical contract: PASS at artifact/code level: H1,
  `canonical_spatial_task_h1_v1`, six actions, 0.3 m translation, 20 degree
  turns, 12 primitive steps, formal collision, and success checks after every
  legal action.
- renderer/native-K cross-task health, resolved-config diff, common step-0,
  fixed-batch gradient intervention, training, and evaluation: NOT_RUN because
  the formal train-manifest gate failed first.

## Job and initialization identities

| Item | Status | Identity |
| --- | --- | --- |
| 1xH800 Zoetrope renderer | NOT_RUN | no job ID, endpoint, or service identity |
| 8xH800 Zoetrope trainer | NOT_RUN | no job ID |
| COMMON_STEP0_INIT | NOT_RUN | no checkpoint or optimizer state created |
| S0/S1/S5 resolved configs | NOT_RUN | candidate matrix only |
| r5 fixed-batch gradients | NOT_RUN | no backward performed |
| S0/S1/S5 150-step pilots | NOT_RUN | no checkpoints |
| canonical evaluations | NOT_RUN | no model results |

The complete frozen-path and SHA ledger is in `EXPERIMENT_INPUT_LEDGER.json`.
The machine-readable gate decision is in `PREFLIGHT_GATE_REPORT.json`.

## Unblock condition

The pilot may be reconsidered only after R1 produces a new formal train
manifest with a versioned PASS gate, frozen SHA256 and lineage, and the same
evaluation exclusions. At that point the remaining renderer, native-K,
reward-only resolved-config, common-step0, and cross-task health gates must run
before either GPU job enters rollout/training.

This round stops at the required hard gate and does not submit the pending R1
data-expansion packages on behalf of the separate R1/canonical window.
