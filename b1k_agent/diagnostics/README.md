# diagnostics/

The only package allowed to read simulator ground truth. Everything here runs **offline, after the
episodes**, and nothing here is fed back into a running agent.

## The isolation rule

- Runtime code (`agent/`, `modules/`, `executor/`, `providers/`, `eval/`) never imports `diagnostics/`, nor any
  simulator package (`omnigibson`, `isaacsim`, `omni`, `pxr`, `bddl`, `carb`). `tests/test_import_guards.py`
  enforces this with an AST scan. `data/` follows the same rule (`tests/test_data_diag_guards.py`), so runtime
  modules may read `data/` records.
- Simulator imports exist only in `replay_truth.py`, inside `main()`. The module itself, and every other file
  here, imports and runs without the simulator (`tests/test_data_diag_guards.py` checks this).
- Truth outputs are `training_annotation` data (`agent/fields.py`). They may become labels
  (`label_distill.py`) or offline statistics (`join.py`), never policy inputs.
- `diagnostics/` may import `eval/` and `data/` (one-way). `eval/` and `data/` never import `diagnostics/`.

## Scripts

| File | What it does | Needs |
|---|---|---|
| `replay_truth.py` | Replays one episode's executed actions (`<local run>/episodes/<ep>/steps.jsonl`, field `a`, 23 floats per step) from the same task instance through a subclass of the official `BatchedEvaluator`. Env setup, instance loading (`load_batch`: TRO state, settling, reset) and stepping are all inherited. It records BDDL goal-literal truth (`BehaviorTask.get_goal_option_satisfaction`, task success, running Q via `compute_q_score`) after reset, every `--every` steps, at every agent decision / model-call step, and at the end. Output: a truth JSONL (`truth.py` format; written as `<out>.partial` and renamed only once complete) plus the replay's own official-format result JSON (refused if it would overwrite the official one), and a footer saying whether the replay reproduced the official outcome. The env wrapper defaults to the episode's `eval.launch` tag, `OMNIGIBSON_APPDATA_PATH` to `~/.omnigibson_appdata` as in `eval/launch.py`; pass `--robot-config` again if the original run used one. | behavior conda env on 105, one free GPU (`eval/gpu.py` rule; `--gpu auto` exits non-zero when none is free, `--wait-gpu` polls) |
| `truth.py` | The truth JSONL format: writer, reader (`load_truth`, `TruthTrack.at / regressions / first_true_steps`), canonical literal texts, and a mirror of the official `compute_q_score`. | pure Python |
| `join.py` | Joins official result JSONs, agent episode summaries and `events.jsonl` (and truth, when given) into per-episode diagnostic records. Only decisions before the official end count (batched runs keep querying a finished env). Each record says whether the checker's last DONE/FAILED verdict agreed with official success (unknown for a 2026/eval `failure_reason` record, whose success is forced to false), lists the planner's "task complete" claims, counts fallbacks (and fallbacks before a success), and, with truth, adds replay consistency, predicate regressions and the maximum running Q. Matching reuses `eval/aggregate.py`; timelines reuse `eval/failures.py`. | pure Python |
| `label_distill.py` | Fills `truth_label`, `truth_source`, `correct` and `truth_detail` of DistillRecords (`data/distill.py` export) from truth. It checks each checker verdict's "done" claim against the truth of the subtask's target subgoals (or of the whole goal), and each subgoal estimate against its literal. | pure Python |

## Commands (105)

```bash
# 1. truth for one episode (behavior env, free GPU). B1K_ENV_SETUP / the /nas/baiqiao/b1k_runs helpers may wrap it.
[ -n "$B1K_ENV_SETUP" ] && source "$B1K_ENV_SETUP"
cd /nas/baiqiao/b1k_agent && PYTHONPATH=. conda run -p /nas/baiqiao/envs/behavior --no-capture-output \
  python -m diagnostics.replay_truth --episode-dir ~/b1k_local_logs/<run>/episodes/<ep> \
    --official-json /nas/baiqiao/b1k_runs/logs/<run>/eval_output/json/<task>_<iid>_0.json \
    --out /nas/baiqiao/b1k_runs/logs/<run>/truth/<ep>.truth.jsonl
#    add --dry-run first: it checks the inputs and prints the plan without importing the simulator

# 2. per-episode diagnostics (any python with the agent deps)
python -m diagnostics.join /nas/baiqiao/b1k_runs/logs/<run> --local-root ~/b1k_local_logs \
  --truth /nas/baiqiao/b1k_runs/logs/<run>/truth/<ep>.truth.jsonl --out diag.jsonl --md diag.md

# 3. distillation labels
python -m data.distill export ~/b1k_local_logs/<run> --out distill.jsonl
python -m diagnostics.label_distill distill.jsonl --truth .../<ep>.truth.jsonl --out distill_labelled.jsonl
```

## Known limits

- The simulator is not deterministic, and `agent/recorder.py` rounds logged actions to 5 decimals. A replay can
  therefore diverge from the original episode. The footer's `consistent` flags this, and `label_distill.py`
  skips divergent episodes unless `--allow-divergent` is given (and always skips a truth track without a footer).
  For a 2026/eval failure record (`failure_reason`) only the step count can be compared, so `consistent` is
  `null` when the steps match.
- An episode whose agent step failed (`agent/server.py` fallback action) has a gap in `steps.jsonl` (the fallback
  action is not logged) and cannot be replayed; `replay_truth.py` refuses it.
- Truth is sampled (every N steps plus decision steps). A literal that flips between two samples is not seen as a
  regression.
- A batched original run (`--num-envs > 1`) is replayed alone. In batched runs the agent keeps stepping an env
  the evaluator has already finished, so `steps.jsonl` can run past the official episode. `--official-json` limits
  the replay to the official `steps`.
- `replay_truth.py` mirrors the v3.9.3-post2 evaluator (the tree on 105). On the `2026/eval` worktree, check the
  `BatchedEvaluator` signatures before using it.
- Quantified goal subgoals that `agent/bddl.py` leaves unexpanded cannot be mapped to grounded literals. Labels
  that depend on them are `unknown`.

## Status

- `truth.py`, `join.py`, `label_distill.py`: code written, unit-tested on SYNTHETIC inputs
  (`tests/test_data_diag_*.py`). Not yet run on real outputs.
- `replay_truth.py`: code written, tested only with a fake evaluator (mock). **NOT RUN**: it has never touched a
  simulator, and nothing has run on 105.
