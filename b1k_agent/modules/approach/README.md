# M-approach: standing position and approach adjustment

## Purpose

Before a manipulation subtask, move the base to a distance and heading from which the arms can reach the target,
using only legal inputs: depth (RGB-D wrapper), the camera poses relative to the base that the evaluator sends
(`cam_rel_poses`), proprioception and the RGB images. No prior maps, object poses or segmentation.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

- **planned**: no real implementation exists.
- The `null` impl is interface-verified (mock tests): `tests/test_loop_matrix.py` switch matrix.

## Source paper

Cracking Mobile Manipulation, arXiv 2610.19600. Per `docs/PAPERS.md` (read from the PDF in the authors'
repository): target grounding plus depth back-projection, reach thresholds of about 0.75-1.0 m (to be re-estimated
for R1Pro), a local XY shift of at most 0.25 m, A* on a 0.05 m depth occupancy grid. **Not verified by us**: arXiv
was unreachable from the environment that wrote this README, and the authors' code is "coming soon".

## Interface

```python
class Approach(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def propose(self, obs: Observation, state: StateView, subtask: Subtask) -> Optional[AdjustPlan]: ...
```

`AdjustPlan(actions (K, 23) | instruction, reason, source, verify_after)`.

## When the loop calls it

`agent/loop.py::_plan`, once per new subtask, right after it starts and before the first executor call. Actions are
queued as an "adjust" sequence, an instruction overrides the subtask's for `loop.check_interval_steps`, and M-check
runs when the sequence finishes (`loop.check_after_scripted`). Only if this returns `None` is
`memory.propose_adjustment` asked.

## Implementations

| impl | description |
|---|---|
| `null` | returns `None`: the loop behaves as if the module did not exist |

## Config

```yaml
modules:
  approach: {enabled: false}      # default; a future impl, e.g. impl: depth_reach (name not decided), sets params
```

## Known failure modes

Expected, from the design (nothing has run):

- Depth is only sent by `RGBDFullResWrapper`; with `DefaultWrapper` (224 RGB) this module has no depth at all.
- Grounding the wrong object, or depth noise at range, gives a wrong standing point.
- Base commands are velocity commands without global odometry (2026 proprio has no global pose), so motions drift.
- Approach motions cost budget and can collide.

## Enable condition

Only when the failure distribution (Step 6/7, `eval/failures.py`) shows "arrived but cannot reach / wrong standing
position" (到达与站位, failure type `out_of_reach`) as a dominant category.
