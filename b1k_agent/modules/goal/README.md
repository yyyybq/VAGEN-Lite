# M-goal: goal management against the BDDL goal

## Purpose

Aim directly at the official Q-score, cheaply (no model calls): order the BDDL subgoals (easy before hard, enabling
before dependent), protect subgoals already estimated as satisfied (veto subtasks that would touch their objects
without targeting them), and switch to a low-risk mode when the time budget is nearly used up instead of moving
randomly. Satisfaction estimates come from M-check, never from simulator truth.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

| impl | state |
|---|---|
| `null` | interface-verified (mock tests) |
| `bddl` | interface-verified (mock tests): `tests/test_units.py::test_goal_order_and_regression_risk`; the BDDL reader is checked on real `problem0.bddl` files (`tests/test_units.py::test_bddl_*`) |

Nothing is simulator-run; no full-eval results.

## Source paper

None from the task spec's list. The design follows the official scoring code: the Q-score counts goal literals that
were false at the start and are true at the end, maximised over goal options
(`OmniGibson/omnigibson/metrics/task_metric.py`; `docs/PROTOCOL.md` section 8).

## Interface

```python
class GoalManager(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def order(self, goal_spec: Optional[GoalSpec], state: StateView) -> list[str]: ...      # subgoal ids
    def regression_risk(self, plan: Subtask, state: StateView) -> float: ...                # 0..1
    def low_risk_mode(self, state: StateView) -> bool: ...
```

## When the loop calls it

- `low_risk_mode` on every step; once true, `state.low_risk` stays set.
- `order` at the start of every planning round (`_plan`); the result is `state.subgoal_order`, shown to M-plan.
- `regression_risk` for every candidate subtask in `_plan`: above `loop.regression_risk_threshold` the candidate is
  rejected (kept in `state.rejected_candidates`) and the planner asked again, at most `loop.max_rejections_per_round`
  times. In low-risk mode it is also checked before each executor call, and a risky subtask gets a hold action.

## Implementations

| impl | description |
|---|---|
| `null` | no order, no veto, never low-risk |
| `bddl` | order by (estimated satisfied last, open-before-inside, close containers last, difficulty class); risk = share of satisfied subgoals whose objects the subtask touches without targeting; low-risk when less than `low_risk_fraction` of the budget (or of `max_steps`) remains |

## Config

```yaml
modules:
  goal: {enabled: true, impl: bddl, params: {low_risk_fraction: 0.05, full_task_risk: 0.0}}
```

## Known failure modes

- Wrong M-check estimates mis-order subgoals or veto useful subtasks.
- Quantified goals (`forall`, `exists`, `forn`) are approximated; the risk only looks at shared object names.
- With only the full-task skill, ordering is prompt context and the risk is the constant `full_task_risk`.
- Holding still in low-risk mode can forfeit progress that a last attempt would have made.

## Enable condition

Ladder level L3. Cheap, so it is tried early; it matters when the failure distribution shows predicate regressions
(task spec 6.7) or episodes that end while the robot is still moving objects.
