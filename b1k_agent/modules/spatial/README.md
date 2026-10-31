# M-spatial: spatial memory

## Purpose

Keep a map-like memory built only from legal observations (RGB, depth, `cam_rel_poses`, proprioception) within one
episode, and give the planner a short text summary (where things were seen, which rooms were visited).

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

- **planned**: no real implementation exists.
- The `null` impl is interface-verified (mock tests).

## Source paper

SERF, arXiv 2606.12956, and Spatial Grafting, arXiv 2609.35249. Per `docs/PAPERS.md`, SERF's released evaluation uses
inputs the 2026 track forbids (`seg_instance`, world camera poses, prebuilt per-instance maps, a patched BDDL goal),
so it is not usable as is; Spatial Grafting has no code found. **Not verified by us**: arXiv was unreachable from the
environment that wrote this README.

## Interface

```python
class SpatialMemory(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def update(self, obs: Observation, state: StateView) -> Optional[str]: ...
```

A non-empty return becomes `state.spatial_summary`, shown in M-plan's prompt.

## When the loop calls it

At the start of every planning round (`agent/loop.py::_plan`), before M-plan.

## Implementations

| impl | description |
|---|---|
| `null` | returns `None` |

## Config

```yaml
modules:
  spatial: {enabled: false}
```

## Known failure modes

Expected, from the design:

- No global pose: positions come from integrating base velocity and drift.
- Object association without instance labels is error-prone; a wrong summary misleads the planner.

## Enable condition

Only when failures point to spatial memory or geometric precision (e.g. a target forgotten after turning away, the
wrong room). Prior maps and instance labels are never legal inputs.
