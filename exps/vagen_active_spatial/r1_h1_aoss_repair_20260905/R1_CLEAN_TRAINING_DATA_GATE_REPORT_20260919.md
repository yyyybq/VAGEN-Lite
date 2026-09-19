# R1 Clean Retraining Data Gate

Date: 2026-09-19  
Scope: repaired canonical data only; no first-action supervision, action-specific finetuning, policy replay, or training.

## What was frozen

The local-action 60-parent / 120-state protocol remains permanently
evaluation-only and is only loaded as an exclusion set.  It has zero source
and scene overlap with the provisional clean corpus.  The 32-source canonical
development regression set also has zero source overlap.

The pre-existing verified train inventory contained 141 unique Projective
source identities across 11 scenes.  It did not meet the frozen minimal
training-control criteria.

The expansion scope was frozen before execution:

- 10 new non-ambiguous train scenes, disjoint from the local-action scenes;
- 593 same-pair Projective Medium-v2 source requests;
- 255 same-pair FOV min-4 source requests;
- identical H1 camera, canonical runtime/scorer, 12-action budget, collision
  convention, Projective 12 px margin, and existing selector budgets.

## H800 expansion result

H800 job `pt-htke09xv` (`h800`, `N4lS.Iq.I80.1`) ran from
2026-09-18T09:02:24Z through 10:25:32Z.  The injected code archive was commit
`1d35cf54162b8879915d91cbc277881f03a6814b`, SHA256
`9abdb9195687622eaf23c06c6912cc6baf95e6577c959957dc4496ecf59b34dc`.

Projective:

- 593/593 source accounting closed;
- 88 difficulty-certified candidates;
- 88/88 independent runtime replay PASS;
- 88/88 official `projective_observability_v1` PASS;
- all are same-pair and their first success is 4--6 actions.

FOV:

- 255/255 source accounting closed;
- 161 same-pair reachable candidates, 93 hard failures, one planner-unverified;
- 161/161 independent replay PASS, including complete no-success-through-depth-3 checks;
- a conservative shared-RGB screen accepted 12/161 and flagged 149 primarily
  for an initial target whose clipped/raw bbox fraction was far below the
  Projective RGB calibration floor.

The shared-RGB screen is not a new FOV success definition and its FOV
calibration has not been established from the original manually reviewed FOV
positives.  The 149 rows are therefore recorded as **not eligible pending
FOV-specific RGB calibration**, not as semantic FOV failures and not as a
reason to relax a threshold.  The accepted 12 have matched runtime/RGB path
evidence.  For example, their contact sheets include 4--10 action paths in
seven new scenes.

## Provisional repaired corpus

The strict evidence merge contains 241 unique train source identities / 241
episodes in 21 scenes:

| Task | Verified episodes |
| --- | ---: |
| Projective | 229 |
| FOV | 12 |
| Total repaired canonical target tasks | 241 |

Frozen pilot criteria have the following observed values:

| Criterion | Required | Observed |
| --- | ---: | ---: |
| Unique sources | 200 | 241 |
| Scenes | 20 | 21 |
| Projective left | 80 | 125 |
| Projective right | 80 | 104 |
| First-success step 4 | 40 | 142 |
| First-success step 5 | 40 | 34 |
| First-success step 6 | 40 | 5 |
| Largest unordered category pair | <=35% | 37.76% (`bed--wardrobe`) |

The gate is **not passed**.  In addition, replacing original v46 Projective
and FOV rows with the verified corpus would retain only 229/2003 (11.43%)
Projective and 12/1503 (0.80%) FOV rows.  The resulting full manifest has
5,889 rows: 5,648 unchanged non-canonical historical rows plus 241 repaired
canonical rows.  It is a useful audit artifact, not a safe clean-retraining
manifest because its task mixture differs substantially from v46.

## Decision

**BLOCKED — do not submit clean retraining.**  No training job, checkpoint,
or first-action/action-specific finetuning job was created.

The immediate data-only next step is to calibrate an FOV-specific official RGB
quality audit against the already manually accepted FOV positives and negative
examples, then regenerate only candidates whose initial state keeps both
objects genuinely observable while FOV failure is due to incomplete framing.
In parallel, collect more verified Projective 5--6-step same-pair examples
from scenes not used for selector development, with category-pair balancing
reported rather than achieved by duplication or sampling tricks.  Re-run the
same corpus merger and gate before any training submission.

## Key artifacts

- `r1_clean_training_inventory_audit_v1_20260918/`
- `r1_clean_training_expansion_scope_v1_20260918/`
- `r1_clean_training_expansion_v1_20260918/`
- `r1_clean_training_corpus_v1_20260919/`

The final corpus SHA256 manifest is
`r1_clean_training_corpus_v1_20260919/SHA256SUMS`.
