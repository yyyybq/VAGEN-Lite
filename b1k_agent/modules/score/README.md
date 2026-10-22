# M-score: action and stage scoring

## Purpose

Learn, from mixed-quality data, a score for action candidates and for stage success, and at inference pick the best
of several executor candidates. A predicted score is never called the official Q-score. Comparisons with the plain
executor use the same number of candidates and the same inference budget.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

- **planned**: no real implementation, no training data, no model.
- The `null` impl exists and is built by the registry, but **the loop does not call `score` yet**: `agent/loop.py`
  asks the executor for one chunk per call. Multi-candidate sampling (one executor call returning N candidates, or N
  calls) must be added to the loop first; that is a loop change, not a module change.

## Source paper

ForesightFlow ("Potential-Guided Flow Matching for VLA Policy Improvement"), arXiv 2606.04968: joint learning of action
and stage-success scores from mixed-quality trajectories. Per `docs/PAPERS.md` no code was found. **Not verified by
us**: arXiv was unreachable from the environment that wrote this README.

## Interface

```python
class Scorer(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def score(self, obs: Observation, state: StateView, candidates: Sequence[ActionChunk]) -> list[float]: ...
```

## When the loop calls it

Not at all today (see Status). The intended trigger is each executor call made while the module is enabled.

## Implementations

| impl | description |
|---|---|
| `null` | 0.0 for every candidate |

## Config

```yaml
modules:
  score: {enabled: false}
```

## Known failure modes

Expected, from the design:

- Scores are overconfident outside the training distribution; selection can exploit scorer errors.
- N candidates multiply executor inference time, which the 1 s/step budget charges.

## Enable condition

Later (task spec section 5): only in a training round that chooses the scorer as its one trainable component (7.4),
after the loop supports multiple candidates.
