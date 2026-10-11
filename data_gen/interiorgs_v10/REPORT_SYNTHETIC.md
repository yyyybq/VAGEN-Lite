# v10 generator — synthetic-scene results (2026-10-11)

Real InteriorGS was not reachable from the development sandbox, so every number
below comes from synthetic scenes (`testing/synthetic_scene.py` and
`coordinates.synthetic_layout`). They validate mechanics and relative gains,
not real-data yield.

## End-to-end: 24 random layouts (`run_all`, target 14 tasks/scene)

| outcome | scenes | notes |
| --- | --- | --- |
| reached 14 tasks | 17 | 13/14 "generic" layouts, 3/4 "sparse", 1/3 "hull_symmetric" |
| some tasks (8) | 1 | sparse layout with 4 objects |
| no candidate objects | 2 | layouts with 0–2 non-structural objects (synthetic artefact) |
| structure convention unusable | 4 | deliberately adversarial notch / hull-symmetric layouts |

246 tasks; by family: projective 96, absolute_positioning 64, screen_occupancy 50,
equidistance 36. Exact d* (shortest action count, certified by BFS):
1:19, 2:38, 3:19, 4:65, 5:17, 6:21, 7:60, 8:7 — v1 had only d* = 1.
Mean 75 s per scene on one core (endpoint search + certification).

## Endpoint search: v9-like vs v10 (2,838 Projective/FOV items, all pairs of 5 scenes)

| sampler | items with ≥ 1 endpoint | endpoints | s/item |
| --- | --- | --- | --- |
| v9-like (≤ 64 positions, yaw = face midpoint ± 10°) | 1,433 (50.5%) | 10,143 | 0.032 |
| v10 (free-floor grid, analytic yaw interval, neighbour-room fallback) | 1,768 (62.3%) | 13,004 | 0.034 |

Items still without an endpoint under v10 (1,070): occlusion gate 830,
no candidate pose 77, inside_frame 108, margin 47.
Pose-level sole failing gate under v10: occlusion 7.0 M, inside_frame 0.37 M,
margin 0.22 M — the generator-side occlusion gate (each target ≥ 50% visible,
ray-cast against object AABBs and walls) is the dominant blocker. AABB occluders
are conservative (a table's box hides everything behind it), so this gate should
be re-tuned on real scenes (lower threshold, exclude open furniture from
occluders, or verify with GS depth on the cluster).

Hypothetical, not reachable under the current motion model (translations follow
the camera axis, so pitch must stay 0):

| fixed pitch | items with ≥ 1 success pose |
| --- | --- |
| 0° | 1,701 |
| −15° | 2,136 (+26%) |
| −30° | 2,088 |

A horizon-locked motion model (yaw about world Z, translation in the floor plane,
fixed downward pitch) would be a protocol change for the RL env, not a generator
change.

## Coordinate calibration v2 (400 random layouts with known sign)

v2 on its own resolved 318/325 and 322/327 identifiable layouts with 0 wrong
freezes; v1 froze 11 and 14 layouts with the wrong sign. On a wall-heavy stress
set v1 was wrong on 68/198, v2 on 0.
