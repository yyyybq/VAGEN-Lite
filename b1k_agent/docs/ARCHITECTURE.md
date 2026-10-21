# Architecture

Status words used here: **code written**, **interface-verified** (mock tests pass), **simulator-run**, **full-eval results**. As of 2026-10-21, the whole system is **code written + interface-verified**. Nothing has run in the simulator yet (STATUS.md).

## Process layout

```
┌──────────── behavior conda env (GPU) ────────────┐
│ python -m omnigibson.eval.eval  (OFFICIAL, as is) │  termination, Q-score, video, metrics JSON
└──────────────┬───────────────────────────────────┘
               │ websocket: msgpack obs (B,…) → {"action": (B,23)}; {"reset": True}; /healthz
┌──────────────▼───────────── agent venv / openpi venv (CPU) ─────────────┐
│ agent/server.py   AgentPolicy: one EpisodeLoop + module set + executor    │
│                   connection per logical env; resend-idempotent;          │
│                   act() runs in a worker thread (pings/health keep going) │
│ agent/loop.py     step-driven loop (the evaluator owns time)              │
│ modules/*         plan / check / goal / recover / approach / memory /     │
│                   score / spatial / explore (null by default)             │
│ providers/*       strong-model clients (Astra = openai_compat, Claude =   │
│                   anthropic SDK, scripted/replay for tests)               │
└──────────────┬──────────────────────────────────────────────────────────┘
               │ websocket: same obs dict (+ "__prompt__", "__flush__", "__return_chunk__")
┌──────────────▼──────────── openpi venv (GPU) ─────────────┐
│ executor/openpi_server/serve_b1k_prompted.py               │  openpi B1KPolicyWrapper per connection,
│   (openpi imported as a library, its tree untouched)       │  robot_r1 → robot key renaming
└────────────────────────────────────────────────────────────┘
```

Why a server instead of a wrapper inside the evaluator:

- **Scoring stays official.** The evaluator, its wrappers and its metrics run unchanged; we cannot change termination, scoring or time stepping.
- **Truth isolation is physical.** The agent process only receives what the evaluator sends: whitelisted RGB, depth, proprio, `cam_rel_poses` and `task_id`. It has no handle on the simulator.
- **Same channel as a submission.** A real submission is also a websocket policy server (Docker or IP-based, `docs/challenge/submission.md`).

## Time

- The official loop is synchronous: the simulator waits for every reply (`evaluator.py _step_fn`). We do not change that.
- The 2026 rules add a wall-clock budget, implemented on branch `2026/eval`:
  - The episode budget is `max_steps × 1 s`, counted from after the first query, which is uncharged up to 600 s.
  - Each query is capped at 600 s.
  - An over-budget rollout scores 0.
- Our handling:
  - `EpisodeState.t_budget_start` starts at the end of the first `step()`.
  - The loop skips provider-backed module calls when less than `loop.provider_min_remaining_s` remains.
  - Every call's latency is logged.
  - The aggregator applies the budget rule as a derived column.

## The loop (`agent/loop.py`)

```
step(obs):
  state.observe(obs)
  goal.low_risk_mode()?  → state.low_risk
  queue empty and a decision is due?   (no subtask | check interval | scripted motion finished | subtask step budget)
      no subtask → _plan:  goal.order → memory.retrieve → spatial.update → plan.next_subtask
                            (→ explore.propose if None) → goal.regression_risk veto (bounded)
                            → start subtask → approach.propose | memory.propose_adjustment
      else       → check.judge → DONE: close + plan | FAILED: _recover | over budget: _recover("timeout")
  _recover: bounded retries → recover.propose → SCRIPTED (queue actions) | INSTRUCTION (override prompt)
            | REPLAN | GIVE_UP; past retries → replan; past replans → full-task fallback
  queue empty → executor.act(obs, current_instruction, horizon) → queue open-loop actions
  emit queue.pop()
```

- With every slot null, the planner returns the full-task subtask once, the checker always says UNKNOWN, and all other modules are no-ops. The loop then forwards every step to the executor with the trained task prompt, so the executor server sees exactly the official request: configuration L0.

## Data contracts

- `agent/interfaces.py`: Observation, ActionChunk, SkillSpec, GoalSpec/Subgoal, EpisodeContext, Subtask, Judgment, RecoveryPlan, Experience, AdjustPlan, and one Protocol per slot. Stable; any change here is a protocol change.
- `agent/state.py`: `EpisodeState` holds system estimates only, and every field is documented. Modules get a read-only `StateView` plus a private scratch dict.
- `agent/fields.py`: the evaluator-key whitelist and the three data-path categories (policy_observation, training_annotation, evaluation_record).
- `agent/registry.py`: config → module objects. A disabled slot gets its null impl. A cross-package impl is written `"<package>:<impl>"`.
- `configs/skills/*.yaml`: what the current executor can do. Only `available` skills reach the planner, and instruction-type recovery needs `verified`.

## Logs (red line 6)

- `<local>/<run_id>/run_manifest.json`, also copied to the NAS: config + hash, code/BEHAVIOR-1K/openpi commits, checkpoint dir and norm-stats hashes, module impls and prompt-file hashes, seed, tags (task/instance), host, GPU.
- `<local>/<run_id>/episodes/<episode_id>/`:
  - `events.jsonl`: decisions, executor calls, provider calls with prompt text, image refs, raw replies, usage, latency and cost;
  - `steps.jsonl`: the action and its source for every step;
  - `frames/`;
  - `summary.json`, also copied to the NAS.
- Official outputs (`json/`, `videos/`) stay untouched in the run's `eval_output/`. Derived tables are built by `eval/aggregate.py`.
