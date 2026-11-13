# M-plan: high-level planning

## Purpose

Choose the next subtask: one skill from the executor's skill catalog plus its arguments, from legal inputs only
(cameras, the BDDL task definition, the task prompt, the skill catalog, the episode history and the system's own
estimates). The executor instruction is rendered from the chosen skill's template, so the model cannot send the
executor an instruction for a skill that does not exist; `__full_task__` renders to the exact trained task prompt.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

| impl | state |
|---|---|
| `null` | interface-verified (mock tests): L0 passthrough identity, `tests/test_loop_matrix.py` |
| `llm` | interface-verified (mock tests) with a `scripted` provider (`tests/test_units.py::test_planner_only_uses_available_skills_and_renders_templates`, loop matrix). Never called against a real endpoint (Astra or the Claude substitute) |

Nothing is simulator-run; no full-eval results.

## Source paper

MobiAgent, arXiv 2610.03476: one subtask per planner call, with the orchestrator (our loop) owning retry budgets and
termination. **Not verified by us**: arXiv was unreachable from the environment that wrote this README, so the paper
text was not read; what was checked from the authors' code and from search snippets is in `docs/PAPERS.md`. Treat
the design pointers as hypotheses.

## Interface

`agent/interfaces.py`:

```python
class Planner(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def next_subtask(self, obs: Observation, state: StateView) -> Optional[Subtask]: ...
```

- Reads: `obs.rgb` (cameras in `params.cameras`), `ctx.task_instruction`, `ctx.goal_spec`, `ctx.skills` (only
  `available` ones), and from the state view `summary_text()`, `holding`, `subgoal_estimates`, `subgoal_order` (M-goal),
  `retrieved` (M-memory), `spatial_summary` (M-spatial), `rejected_candidates` (M-goal veto), `low_risk`,
  `remaining_budget_s()`.
- Returns a `Subtask` (skill, instruction, args, target_subgoals, max_steps, source, evidence) or `None` = "nothing left
  to plan" (the planner believes the task is complete).

## When the loop calls it

`agent/loop.py::_plan`:

- at the first step of an episode (no subtask yet);
- after a `DONE` judgment from M-check;
- after a subtask used up `loop.max_retries_per_subtask` (a replan, at most `loop.max_replans`);
- after a `REPLAN` recovery.

Each round may call it up to `loop.max_rejections_per_round + 1` times when M-goal vetoes the candidate
(`regression_risk > loop.regression_risk_threshold`). With less than `loop.provider_min_remaining_s` of budget left,
the loop skips the call and uses the full-task subtask (`loop.fallback.budget`). A `None` goes to M-explore; if that
also returns `None`, the plan is exhausted and `loop.on_plan_exhausted` decides (`full_task` keeps running the task
prompt, `hold` stops).

## Implementations

`modules/plan/__init__.py` IMPLS:

| impl | provider | description |
|---|---|---|
| `null` | no | the whole task as one subtask with the executor's trained prompt, once |
| `llm` | yes | strong-model skill selection with `configs/prompts/plan/<version>.txt`; provider errors or invalid replies fall back to the full-task subtask (`plan.llm.fallback`, logged) |

## Config

```yaml
modules:
  plan:
    enabled: true
    impl: llm
    provider: astra            # or claude (substitute; label results claude-substitute)
    prompt_version: v1
    params: {cameras: [head], max_tokens: 1024, timeout_s: 60, default_max_steps: 900, min_max_steps: 60,
             allow_task_complete: true}
```

Ablation: `enabled: false` (the null planner), everything else unchanged.

## Known failure modes

- With the current catalog (`configs/skills/pi05_turning_on_radio.yaml`) only `__full_task__` is available, so the
  planner can only re-issue the task prompt or declare the task complete. Its effect is limited to stopping and
  re-planning decisions until the executor has verified skills.
- A false "task complete" stops planning; with `on_plan_exhausted: hold` the robot then idles with unsatisfied goals.
- Provider latency is charged to the `max_steps x 1 s` budget (only the first query is free).
- Arguments naming objects that are not visible or not in the task still render into the template.
- Sub-instructions are only meaningful if the executor responds to them (Step-5 diagnosis, `scripts/skill_probe.py`).

## Enable condition

Step 7 (task spec): the minimal closed loop M-plan + M-check + minimal M-recover, compared with L0 and the
budget-matched control `C_budget_matched` on the same Dev set. Ladder level L1.
