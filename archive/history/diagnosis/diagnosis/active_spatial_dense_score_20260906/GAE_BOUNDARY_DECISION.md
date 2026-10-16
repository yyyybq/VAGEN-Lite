# no-concat GAE boundary decision report

Status: `PASS` for code-path audit and synthetic fixtures. No production semantic change is made in this round.

## A. Historical mode (implemented and replayed)

`vagen/custom_advantage/no_concat_gae.py:303` initializes `next_value = 0.0` for every `(group_idx, traj_idx)` trajectory and never reads `terminated` or `truncated`. At lines 306–324 it therefore applies a zero bootstrap to both environment terminals and LLM-turn time limits. It uses first valid response-token value, sums per-turn token rewards, broadcasts each turn advantage to valid response tokens, writes one return at the first valid token, and whitens unique rows before duplicate expansion (lines 247–336).

The exact replay implementation is [active_spatial_ppo_replay.py](/mnt/umm/users/yinbaiqiao/VAGEN-Lite/vagen/utils/active_spatial_ppo_replay.py:43). It intentionally retains this historical behavior. No `terminated`/`truncated` repair is silently applied.

## B. Terminal-aware candidate (not applied)

Candidate desired boundary:

```python
if terminal:
    next_value = 0.0
elif truncated:
    next_value = critic(next_state)  # exact captured post-action next state
else:
    next_value = next turn's first-token value
```

This requires a new explicit `bootstrap_value` tensor computed from the same immutable critic checkpoint and the captured next-state preprocessing input. It cannot be inferred from the final turn's current-state value. The new snapshot stores enough source context for a later implementation, but the generic replayer reports terminal-aware analysis `BLOCKED` unless that exact value/preprocessing adapter is supplied.

Candidate patch location is the `next_value = 0.0` initialization and trajectory reverse loop in `no_concat_gae.py:303-324`; it must take explicit `terminated`, `truncated`, `bootstrap_values`, and a convention/version flag. That patch is intentionally only documented here, not applied to formal training.

## Boundary cases audited

| Case | Current behavior | Candidate implication |
| --- | --- | --- |
| Primitive limit | `env.py:980-1055` marks `truncated_max_primitive_steps`, obtains final score/success, then skips `_calculate_pose_reward` because `done` is true. Thus final shaping is absent; current GAE also zero-bootstraps. | Preserve reward omission as a separate reward semantic; only consider bootstrap if the limit is defined as time-limit truncation and next-state value is exact. |
| `max_llm_turns` | `gym_agent_loop_no_concat.py:380-392` turns outer limit into `truncated=True`, reason `max_llm_turns`; historical GAE ignores it. | Candidate can bootstrap only from stored post-action state, not a dummy NFP frame. |
| Initial success | Reset computes phi but does not terminate immediately; the first generated turn remains a normal episode turn. | No special bootstrap unless a future task-definition change adds reset-time terminal transitions. |
| Variable primitive bundle | Reward/GAE is one LLM turn while environment clock counts 0..5 primitive actions per turn. | Gamma is turn-level today, so potential-policy-invariance and time-limit interpretation are not primitive-time invariants. |
| Collision/invalid/explicit done | Recorded as terminal reasons and reward components; historical GAE zero-bootstraps all final trajectories. | True terminal remains zero bootstrap. |

The synthetic fixture includes a terminated row, a `max_llm_turns` truncated row, mixed response lengths, and a two-primitive action bundle. It establishes serializer/replay boundary handling only, not a critic estimate from a real batch.
