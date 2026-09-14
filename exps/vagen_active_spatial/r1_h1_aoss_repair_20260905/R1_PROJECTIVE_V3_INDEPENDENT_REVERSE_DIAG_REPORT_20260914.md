# R1 Projective v3 Independent Validation and Reverse-Cap Diagnosis

Date: 2026-09-14

## Execution and frozen state

- SCO task `pt-2bvzd2mx`, pool `zoetrope`, one H800 used.
- Worker `pt-9406b4ca1451426fb5c5b62ad1c23d3f-worker-0`.
- `2026-09-14T04:06:30Z` to `04:27:28Z`; `SUCCEEDED`, exit 0.
- Frozen commit `58636eb5fbf110221a1f5d5d5fca7577d1580a53` was injected into the gitless worker.
- Official renderer stopped cleanly. H1, metric, 12 px, collision/layout, action space, Medium, observability, and FOV were unchanged.

## Verified Medium inventory

The 72 v2 RGB-passing sources plus three distinct v3 additions give 75 unique verified sources and 77 episodes; two cross-bundle duplicate episodes retain lineage but are not double-counted as sources.

| Split | Unique sources | Episodes |
|---|---:|---:|
| train | 43 | 44 |
| id_test | 16 | 17 |
| ood_geometry | 1 | 1 |
| ood_instance | 5 | 5 |
| ood_scene | 6 | 6 |
| ood_template | 1 | 1 |
| validation_proxy | 3 | 3 |
| **Total** | **75** | **77** |

Only the 43 train sources are potential training inventory. The remaining 32 validation/test/OOD sources stay isolated. The formal old audit supports 84 one-step certificates among 284 reachable rows and 73 one-step rows among 225 RGB-pass rows. The former `183` claim has no formal first-success support and remains unknown. Thirty verified Medium sources overlap the old one-step RGB-pass set.

## Independent v2 screen

Five train scenes outside the original ten-scene design set were frozen before running.

| Scene | Sources | Certified/runtime | RGB pass | Shortcut | Not found |
|---|---:|---:|---:|---:|---:|
| 0012_840878 | 26 | 7 | 7 | 4 | 15 |
| 0015_840888 | 26 | 2 | 2 | 4 | 20 |
| 0016_840873 | 26 | 5 | 3 | 7 | 14 |
| 0019_840447 | 25 | 5 | 5 | 2 | 18 |
| 0020_840256 | 25 | 3 | 3 | 2 | 20 |
| **Total** | **128** | **22** | **20** | **19** | **87** |

The screen produced only 19 shortcuts, below the planned 24–32, so all 19 were used without expanding the screen. Both v2 RGB failures were quality rejections, not renderer errors.

## Frozen v2/v3 shortcut A/B

The v3 run used 19 shortcuts plus five distinct-scene RGB-pass controls: 24 requests, eight certified, 8/8 runtime PASS, and 6/8 RGB PASS. Controls were 5/5 generator rediscovery plus runtime/RGB PASS. Three former shortcuts became certified/runtime-pass, but only one passed RGB.

The sole recovery is `train:1308` in `0019_840447`, same pair (`77:wardrobe`, `149:stairs`). Its certificate is `turn_left, turn_left, move_right, move_right, move_backward`; the complete lower bound is four and first success is step five. RGB shows both targets, no wall-like run, and a natural trajectory. `train:1085` and `train:1120` remain frozen RGB rejects because their initial dual targets are not discernible; manual review agrees.

Recovery therefore spans one scene, not the required three. Cost and controls pass, but the independent scalability gate fails. v3 should not be applied to the original 75 shortcut rows on this evidence.

## Reverse-cap diagnosis

The frozen set contains 16 cap failures plus four known-certificate controls. All 139 failure seeds reached 512 expansions.

- Primary classes: A=13, B=3; evidence tags A=13, B=16.
- All 16 reached depth 4–6 states but found zero eligible initial; 13 also retained a shallow frontier at cap.
- No C: cross-seed duplicate fraction median `0.09868`, max `0.20455`; repeated seed coverage is not dominant.
- No D: zero forward-state mismatches and max same-key pose alias error about `2.66e-15`.
- No E among sampled failures.

The failure searches admitted many depth-4-to-6 states. Initial-projection rejects totaled 42,565; `inside_frame` occurred in every projection reject, with relation/margin/visibility/min-area overlapping. Layout and collision rejects were 20,060 and 9,010. The main evidence is therefore missing dual-target-projectable initials among reached states; incomplete shallow coverage is secondary. Increasing the cap or sharing visited states alone is unlikely to fix the dominant issue.

## Cost and conclusions

- SCO wall: 20m58s; reverse diagnostic about 8m01s; v2 screen 6m03s; v2 RGB 2m27s; v3 generation 1m54s; v3 RGB 2m11s.
- Same-source v3/v2 generator elapsed ratio: `1.09794`, inside the frozen 1.2 tolerance.
- Per-source elapsed sums are diagnostic CPU accounting, not parallel wall time.

Conclusions:

1. The 75-source inventory is reliable; only 43 are train identities and evaluation identities remain isolated.
2. v3 shortcut recovery is not independently scalable: 1/19, one scene, despite preserved controls and acceptable cost.
3. No action-graph/state-key bug was found. The sampled bottleneck is initial projection/inside-frame feasibility in the reached 4–6-step state set.

## Exact next bounded experiment

Do not rerun the original 75 shortcuts with v3. If generation work continues, test one versioned projection-quality/camera-orientation-ranked depth-4-to-6 frontier variant on the same 16 failures plus four controls. Keep seeds, 512 per-seed cap, candidate limit, lower-bound budget, runtime transitions, and final gates unchanged; never filter legal intermediate states.

Expand only if it recovers at least 4/16 across at least three scenes, preserves 4/4 controls, passes independent runtime and official RGB for every accepted row, and costs at most 1.2x the same-source baseline. Otherwise retain verified Easy/Medium support subsets and keep unsupported sources explicit. No 788-row rerun, 100-scene regeneration, replay, or training was started.
