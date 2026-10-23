# M-check: progress checking

## Purpose

Judge the running subtask from legal observations: done / in_progress / failed / unknown, with a confidence, the
visible evidence, a failure type, and the system's estimates of subgoal satisfaction and of what each gripper holds.
These estimates (never simulator truth) feed M-goal, M-plan and M-recover through `EpisodeState`.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

| impl | state |
|---|---|
| `null` | interface-verified (mock tests) |
| `rule` | interface-verified (mock tests): `tests/test_units.py::test_rule_checker_detects_stall`, `tests/test_loop_matrix.py::test_check_rule_impl_runs` |
| `llm` | interface-verified (mock tests) with a `scripted` provider: `tests/test_units.py::test_checker_downgrades_low_confidence_and_handles_errors`. Never called against a real endpoint |

Nothing is simulator-run; no full-eval results.

## Source paper

MobiAgent, arXiv 2610.03476: the judge decides complete / incomplete / error from head and wrist frames and must not
infer success from elapsed time, the task name or an earlier verdict. **Not verified by us**: arXiv was unreachable
from the environment that wrote this README; see `docs/PAPERS.md` for what was checked in the authors' code.

## Interface

```python
class Checker(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def judge(self, obs: Observation, state: StateView) -> Judgment: ...
```

`Judgment(status, confidence, evidence, failure_type, subgoal_updates, holding_updates, source)`. `failure_type` is
one of `agent.interfaces.FAILURE_TYPES` (no_progress, grasp_failed, object_dropped, out_of_reach, wrong_object,
collision, subgoal_regressed, timeout, unknown). The loop writes `subgoal_updates` to `state.subgoal_estimates` and
`holding_updates` to `state.holding`.

## When the loop calls it

At a decision point while a subtask runs (`agent/loop.py::_decide`):

- every `loop.check_interval_steps` control steps (150 = 5 s at 30 Hz);
- right after a scripted recovery or an adjustment sequence finishes (`loop.check_after_scripted`);
- when the subtask's own `max_steps` is reached.

`DONE` closes the subtask and plans the next one; `FAILED` calls M-recover with the failure type; an exhausted subtask
budget without `DONE` calls M-recover with `timeout`. Provider-backed impls are skipped (treated as `UNKNOWN`) when
less than `loop.provider_min_remaining_s` of budget remains.

## Implementations

| impl | provider | description |
|---|---|---|
| `null` | no | always `UNKNOWN` ("keep executing") |
| `rule` | no | proprioception only: no joint/base motion for `stall_window_steps` -> `FAILED(no_progress)`; holding estimated from finger width when the gripper is commanded closed; never claims `DONE` |
| `llm` | yes | strong-model judgment with `configs/prompts/check/<version>.txt`; `DONE` / `FAILED` below `min_confidence` become `UNKNOWN`; provider errors -> `UNKNOWN` |
| `distill:<name>` | no | planned, from `modules/distill` |

## Config

```yaml
modules:
  check: {enabled: true, impl: llm, provider: astra, prompt_version: v1,
          params: {cameras: [head, left_wrist, right_wrist], min_confidence: 0.6, max_tokens: 1024, timeout_s: 60}}
# budget-matched control (configs/C_budget_matched.yaml)
  check: {enabled: true, impl: rule, params: {stall_window_steps: 240, joint_motion_eps: 0.02, base_speed_eps: 0.02}}
```

## Known failure modes

- A false `DONE` ends a subtask early; a shaky verdict is downgraded, which can also mean `DONE` is never declared.
- Success that is not visible (occlusion, small state changes such as a radio display) is judged unknown or wrong.
- A long check interval misses regressions; a short one spends budget on model calls.
- The rule checker reports a stall while the executor legitimately pauses; a closed gripper is not a grasp.

## Enable condition

Step 7 minimal closed loop (ladder level L2). The strong-model impl is later replaced by M-distill once enough
labelled calls exist.
