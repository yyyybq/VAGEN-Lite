# M-distill: small-model checker / switcher

Not a loop slot: `modules/distill/__init__.py` has `IMPLS = {}`. When an implementation exists it plugs into the
`check` slot as a cross-package impl, `impl: "distill:<name>"` (`agent/registry.py` resolves it and requires the
ImplSpec's slot to be `check`).

## Purpose

Train a small VLM on the strong model's judgments made from legal observations, and use it instead of the
strong-model checker to cut API cost and latency. Start with capabilities that have clear boundaries: "is the subtask
done", "switch to the next subtask". Simulator truth may verify the labels offline; the small model's inputs contain
only legal fields.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

| part | state |
|---|---|
| small model, training, `distill:<name>` impl | planned |
| logging of every strong-model call (inputs, outputs, evidence, usage) to `events.jsonl` | interface-verified (mock tests) |
| `data/schema.py` `DistillRecord`, `data/distill.py` export from run logs | code written and unit-tested on synthetic logs; no real calls exported yet |
| `diagnostics/label_distill.py` (offline truth labels for the records) | code written and unit-tested on synthetic truth; real labels need replay truth from the simulator |

## Source paper

StageGuard, arXiv 2609.20791: a strong teacher builds stage-transition supervision, then a light checker is trained.
Per `docs/PAPERS.md` no code was found. **Not verified by us**: arXiv was unreachable from the environment that wrote
this README.

## Interface

It will implement `Checker` (`agent/interfaces.py`): `judge(obs, state) -> Judgment` with the same statuses,
confidence, visible evidence and failure types as `check.llm`, so the two can be swapped with one config line.

## When the loop calls it

Exactly when M-check is called (see `modules/check/README.md`), once configured under the `check` slot.

## Implementations

None (`IMPLS = {}`). A reference to `distill:<name>` fails with `KeyError: modules.distill has no impl ...`.

## Config

```yaml
modules:
  check: {enabled: true, impl: "distill:small_vlm", params: {...}}   # once the impl exists; the name is not decided
```

## Known failure modes

Expected, from the design:

- Teacher label noise is learned too; truth-based verification must stay offline (a label, never an input).
- Distribution shift between the Dev tasks it was trained on and other tasks.
- Poor calibration makes the `min_confidence` downgrade unreliable.

## Enable condition

Replaces the strong-model checker only after enough labelled calls exist and a Dev comparison against `check.llm`
under equal conditions (same tasks, instances, seeds and budget) shows it holds up. It is one trainable component of a
training round (task spec 7.4).
