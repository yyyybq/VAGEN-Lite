# M-explore: online exploration

## Purpose

Propose an exploration subtask (look around, visit a room) when the planner has nothing to offer, e.g. because the
target is not in view. Off by default.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

- **planned**: no real implementation exists.
- The `null` impl is interface-verified (mock tests).

## Source paper

RAX ("Scaling Online Exploration for Robot Agents"), arXiv 2610.22496. Per `docs/PAPERS.md` its code is Apache-2.0
with a BEHAVIOR branch, and tools inside its wrapper (cuRobo, IK) need a legality check. **Not verified by us**:
arXiv was unreachable from the environment that wrote this README.

## Interface

```python
class Explorer(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def propose(self, obs: Observation, state: StateView) -> Optional[Subtask]: ...
```

## When the loop calls it

In `agent/loop.py::_plan`, only when M-plan returned `None`. A proposed subtask then goes through M-goal's
regression-risk veto like any planner candidate.

## Implementations

| impl | description |
|---|---|
| `null` | returns `None` |

## Config

```yaml
modules:
  explore: {enabled: false}       # default and required until the rules below are verified
```

## Known failure modes

Expected, from the design:

- Exploration spends the same `max_steps x 1 s` budget as task progress.
- Wandering away can disturb satisfied goal conditions.

## Enable condition

Off by default. Before any use, verify and write down (task spec section 5): the exploration budget, test-time agent
participation, and that no information crosses instances or episodes (EpisodeState is per episode; cross-episode
online learning stays off). Tools it calls must not read simulator truth.
