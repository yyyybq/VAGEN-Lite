# M-memory: experience library, retrieval and handover bridge

## Purpose

Two stages, built in this order:

1. **Retrieval ("RAG")**: a library of experiences from the demonstrations and from successful Dev-period
   trajectories, retrieved by skill, object semantics and visual state, handed to M-plan / M-check as evidence. Every
   entry records its provenance (dataset, version, episode, frame range, or Dev run id).
2. **State-adjustment bridge**: before handing over to the next skill, retrieve candidate states from which that skill
   is known to succeed, score them locally, generate a short adjustment with legal perception and control, and check
   the result afterwards. Absolute joint trajectories are never replayed across scenes.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

- **planned**: no real implementation exists.
- The `null` impl is interface-verified (mock tests); the library record type `data/schema.py` `ExperienceRecord` is
  code written and unit-tested.

## Source paper

RoboHarness, arXiv 2607.18060 (memory-driven orchestration of heterogeneous robot policies; capability descriptions,
experience retrieval, Memory Bridge). Per `docs/PAPERS.md` no official code was found, and its geometric bridge must be
re-checked against our observation limits. **Not verified by us**: arXiv was unreachable from the environment that
wrote this README.

## Interface

```python
class Memory(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def retrieve(self, obs: Observation, state: StateView, k: int) -> list[Experience]: ...
    def propose_adjustment(self, obs: Observation, state: StateView, next_subtask: Subtask) -> Optional[AdjustPlan]: ...
```

`Experience(exp_id, provenance, skill, objects, summary, outcome, payload)`; the loop stores the list in
`state.retrieved`, which M-plan's prompt shows as "experiences".

## When the loop calls it

- `retrieve`: at the start of every planning round (`_plan`), with `k = loop.retrieve_k`.
- `propose_adjustment`: once per new subtask, only when M-approach returned `None`; handled like an approach
  adjustment (queued actions or an instruction override, then a check).

## Implementations

| impl | description |
|---|---|
| `null` | `retrieve -> []`, `propose_adjustment -> None` |

## Config

```yaml
modules:
  memory: {enabled: false}        # default; a retrieval impl will take a frozen library path and k in params
```

## Known failure modes

Expected, from the design:

- Irrelevant or misleading experiences steer the planner; longer prompts cost latency.
- Leakage: Report instances (public_test 301-310) and hidden instances must never enter the library
  (`data/schema.py` REPORT_INSTANCE_IDS / HIDDEN_INSTANCE_IDS checks).
- Bridge adjustments computed in one scene do not transfer to another.

## Enable condition

Retrieval first. The bridge only after handover failures (技能切换与交接) are evident in the failure distribution. For
Report evaluation the library and parameters are frozen; only the per-episode task memory changes within an episode.
