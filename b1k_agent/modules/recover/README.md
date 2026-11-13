# M-recover: bounded recovery

## Purpose

React to a failed subtask with a bounded, logged recovery. Two kinds of primitive:

- scripted, from proprioception only: retract the arms to a safe pose, back up, rotate in place to look again;
- instruction-type (a text instruction for the executor), only for skills the catalog marks `available` AND
  `verified` by the Step-5 diagnosis.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

| impl | state |
|---|---|
| `null` | interface-verified (mock tests) |
| `scripted` | interface-verified (mock tests): `tests/test_units.py::test_scripted_recovery_primitives`, `::test_instruction_recovery_requires_verified_skill`. The primitive magnitudes (speeds, durations, safe pose) are untested on the simulated R1Pro |

Nothing is simulator-run; no full-eval results.

## Source paper

Cracking Mobile Manipulation, arXiv 2610.19600: diagnose failures first, then add targeted recovery skills built
from RGB-D and proprioception. **Not verified by us**: arXiv was unreachable from the environment that wrote this
README; `docs/PAPERS.md` lists what was read from the PDF in the authors' repository.

## Interface

```python
class Recovery(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def propose(self, obs: Observation, state: StateView, failure_type: str) -> Optional[RecoveryPlan]: ...
```

`RecoveryPlan(kind, primitive, actions, instruction, duration_steps, reason, source)` with `kind` one of `SCRIPTED`
(`actions` (K, 23)), `INSTRUCTION`, `REPLAN`, `GIVE_UP`.

## When the loop calls it

`agent/loop.py::_recover`, after M-check returns `FAILED` (with its failure type) or when the subtask's `max_steps`
ran out without `DONE` (`timeout`). The loop first counts a retry; past `loop.max_retries_per_subtask` it ends the
subtask and replans (at most `loop.max_replans`, then the full-task fallback) without asking this module.

- `SCRIPTED`: the executor is interrupted, the actions are queued, and M-check runs when they finish.
- `INSTRUCTION`: the instruction overrides the subtask's for `duration_steps`.
- `REPLAN`: the subtask is abandoned and M-plan is asked.
- `GIVE_UP`: the retry counter is set to its maximum.
- `None`: keep executing.

## Implementations

| impl | description |
|---|---|
| `null` | no recovery |
| `scripted` | playbook failure type -> primitives (`retract_arms`, `back_up`, `rotate_in_place`), at most `max_attempts_per_type` per type; instruction primitives only with `instruction_recovery: true` and a verified skill |

## Config

```yaml
modules:
  recover: {enabled: true, impl: scripted,
            params: {retract_steps: 60, back_up_steps: 45, back_up_speed: 0.3, rotate_steps: 60, rotate_speed: 0.4,
                     max_attempts_per_type: 3, instruction_recovery: false, instruction_skills: {}}}
```

## Known failure modes

- Base motions are blind (no collision checking beyond what the cameras would show a checker afterwards).
- Retracting the arms can drop a held object; the safe pose (all arm joints 0, the R1Pro reset pose) may be a poor
  pose near furniture.
- Recoveries spend budget and leave the executor in states it may not have seen in training.
- A false `FAILED` from M-check triggers needless recoveries.

## Enable condition

Step 7 (minimal M-recover; ladder level L4). Always compared with `configs/C_budget_matched.yaml` (same retries and
primitives with a rule checker, no strong model) so that a gain is not just extra retries.
