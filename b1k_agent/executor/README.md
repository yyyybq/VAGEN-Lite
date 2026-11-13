# M-exec: executors

## Purpose

The action base: an off-the-shelf visuomotor policy behind one fixed interface, so that π0.5 (now), GR00T N1.7 or a
later checkpoint can be swapped by configuration. Also home of the skill catalogs (what the current executor can do)
and of our π0.5 server with per-request prompt override.

```
official eval.py --ws--> agent/server.py (loop) --ws--> executor/openpi_server/serve_b1k_prompted.py (openpi venv, GPU)
```

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

| part | state |
|---|---|
| `openpi_ws.py` client (`OpenPIWebsocketExecutor`) | interface-verified (mock tests): `tests/test_protocol.py` against a fake prompt-capable server (prompt override, flush, chunk mode, multi-task) |
| `openpi_server/serve_b1k_prompted.py` | code written; never run (needs the openpi venv and a GPU on 105) |
| `openpi_server/keys.py` (robot_r1 -> robot key renaming) | interface-verified (mock tests): `tests/test_protocol.py::test_rename_keys` |
| `hold` executor | interface-verified (mock tests); for pipeline smoke tests only, never for results |
| GR00T N1.7, Comet / RLC 50-task, Galaxea G0.5 adapters | planned |

Nothing is simulator-run; no full-eval results.

## Source paper

- Baseline: the official 2026 π0.5 route, `wensi-ai/openpi` branch `behavior` @ `0cc8e355` (`scripts/b1k/serve_b1k.py`,
  `src/openpi/shared/eval_b1k_wrapper.py`), and the official `turning_on_radio` checkpoint
  (`docs/challenge/baselines.md`). These were read in code; `docs/PROTOCOL.md` sections 10 and 12 have the details.
- Alternative backbone: Galaxea G0.5, arXiv 2608.11739. Per `docs/PAPERS.md` it has open weights under a
  non-commercial license and no BEHAVIOR checkpoint or entry point. **Not verified by us**: arXiv was unreachable from
  the environment that wrote this README.

## Interface

```python
class Executor(Protocol):            # agent/interfaces.py
    name: str
    def reset(self, ctx: EpisodeContext) -> None: ...
    def act(self, obs: Observation, instruction: str, horizon: int) -> ActionChunk: ...
    def interrupt(self) -> None: ...  # drop buffered actions; the next act() re-infers
    def close(self) -> None: ...
```

`ActionChunk(actions (H, 23), instruction, source, infer_ms, open_loop_steps, meta)`; the 23-D R1Pro action layout is
`agent/fields.py` `ACTION_SLICES`.

Modes of `openpi_ws`:

- `passthrough` (default): the evaluator's observation is forwarded unchanged and ONE action comes back per step; the
  server's `B1KPolicyWrapper` applies the official chunking rule (`receding_horizon`, `action_horizon 16`). With the
  trained task prompt this is the official baseline action stream (configuration L0).
- `chunk`: the raw model chunk (`__return_chunk__: H`) comes back and the loop executes `open_loop_steps` actions open
  loop. Used by `scripts/skill_probe.py`.

Wire additions (all optional; absent = byte-for-byte serve_b1k behaviour): `"__prompt__"` (instruction override),
`"__flush__"` (drop buffered actions), `"__return_chunk__"`. Connect-time metadata adds `default_prompt`, `prompts`,
`control_mode`, `action_horizon`, `model_action_horizon`, `checkpoint_dir`, `train_config`, `norm_stats_repo_id`.

## When the loop calls it

- `reset` at the start of each episode; one executor connection per logical environment.
- `act(obs, state.current_instruction(), loop.exec_horizon)` whenever the action queue is empty after any decision
  (`agent/loop.py::_fill_from_executor`).
- `interrupt` on every subtask change, scripted recovery or adjustment.
- On an agent error, `agent/server.py` asks the executor directly with the full task prompt (then a hold action).

## Implementations

| impl | description |
|---|---|
| `openpi_ws` | π0.5 served by `openpi_server/serve_b1k_prompted.py` |
| `hold` | base 0, joints held at their current positions; smoke tests only |

## Config

```yaml
executor:
  impl: openpi_ws
  skill_catalog: configs/skills/pi05_turning_on_radio.yaml
  params: {host: 127.0.0.1, port: 8000, mode: passthrough, allow_prompt_override: true, recv_timeout_s: 120,
           checkpoint_dir: /nas/baiqiao/checkpoints/pi05_turn_on_the_radio}
```

Server (openpi venv, a free GPU; `--repo-id` is the asset id under `<ckpt>/assets/`, see
`scripts/inspect_checkpoint.py`):

```bash
cd /nas/baiqiao/openpi && CUDA_VISIBLE_DEVICES=<free> XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 .venv/bin/python \
  /nas/baiqiao/b1k_agent/executor/openpi_server/serve_b1k_prompted.py --robot b1k/R1Pro --task b1k/turning_on_radio \
  --repo-id <asset-id> --policy.config pi05_b1k --policy.dir /nas/baiqiao/checkpoints/pi05_turn_on_the_radio \
  --control-mode receding_horizon --action-horizon 16 --port 8000 --rename-robot-from robot_r1
```

`python -m eval.launch` starts it with these arguments.

## Known failure modes

- Robot-name mismatch: the openpi R1Pro config reads `robot::...` keys, the official evaluator sends `robot_r1::...`;
  `--rename-robot-from robot_r1` rewrites key names only.
- Prompt mismatch (unverified): the server prompts with the natural-language task text, while the official data
  pipeline may have trained on the snake_case task name (`configs/tasks/README.md`).
- The official checkpoint covers `turning_on_radio` only; `--task b1k/<other>` raises a `KeyError` in openpi's
  `TASK_REGISTRY`.
- The norm-stats asset id (`--repo-id`) must match `<ckpt>/assets/<id>/norm_stats.json`.
- One model per server: inference is serialized across connections, so parallel environments slow each other down.
- A policy trained on full-task prompts may ignore other instructions: check with `scripts/skill_probe.py` before
  marking any catalog skill available.

## Enable condition

Always on: it is the action base. The choice of executor follows Step 5: it must download, load, and run through the
official interface on 105, in that order. Classes are never interchanged (generic π0.5 pretrain, 2025 50-task
adapted, per-task, 2026 100-task), even when they share a name.
