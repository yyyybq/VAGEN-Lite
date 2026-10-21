# CLAUDE.md — b1k_agent

Read this first in every session. Full task spec: `docs/TASK_SPEC_20261022.md`. Current state: `STATUS.md`.
Evaluation rules with sources: `docs/PROTOCOL.md`.

## What this is

A performance-oriented BEHAVIOR-1K 2026 Challenge (100 tasks) baseline. It has three parts:

- a strong-model high-level loop (`agent/`, `modules/`);
- an off-the-shelf action executor, π0.5 now (`executor/`);
- the OFFICIAL evaluator, unchanged.

Our agent is a websocket policy server. The official `python -m omnigibson.eval.eval` connects to it, and it forwards observations to the executor server. Termination, scoring and stepping are 100% official.

```
official eval.py --ws--> agent/server.py (loop + modules) --ws--> executor/openpi_server/serve_b1k_prompted.py (pi0.5)
```

No result is submitted or published without the user's explicit authorization.

## Where things run

- Server **105** has everything: `/nas/baiqiao/BEHAVIOR-1K` (v3.9.3-post2, a8247a8c), the conda env `/nas/baiqiao/envs/behavior`, the openpi venv `/nas/baiqiao/openpi/.venv`, the checkpoint `/nas/baiqiao/checkpoints/pi05_turn_on_the_radio`, and the user's scripts in `/nas/baiqiao/b1k_runs/` (reuse them, don't duplicate).
- If the shell cannot see `/nas/baiqiao/BEHAVIOR-1K` or `nvidia-smi`, you are NOT on 105:
  - use `ssh 105 '...'` (check the alias and passwordless login first);
  - otherwise do code, data tools and tests on recorded logs only;
  - never install Isaac Sim elsewhere.
- Do not reinstall or upgrade the existing envs.
  - The agent server runs in the openpi venv (same pins) or in a NEW venv (`scripts/setup_agent_env.sh`).
  - Record any new dependency with its exact pip command and the reason (`docs/ENVIRONMENT.md`).
- Do not modify the BEHAVIOR-1K or openpi working trees. Use a separate git worktree or a patch file, and note the upstream commit. The `2026/eval` evaluator goes in its own worktree.

## Commands (run from the repo root)

```bash
# tests (local, no simulator): python>=3.11 with numpy<2, pyyaml, websockets, msgpack, pillow, httpx, pytest
python -m pytest -q

# executor server (openpi venv, on a free GPU); --repo-id = the asset id under <ckpt>/assets/
cd /nas/baiqiao/openpi && CUDA_VISIBLE_DEVICES=<free> XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 .venv/bin/python \
  /nas/baiqiao/b1k_agent/executor/openpi_server/serve_b1k_prompted.py --robot b1k/R1Pro --task b1k/turning_on_radio \
  --repo-id <asset-id> --policy.config pi05_b1k --policy.dir /nas/baiqiao/checkpoints/pi05_turn_on_the_radio \
  --control_mode receding_horizon --action_horizon 16 --port 8000 --rename_robot_from robot_r1

# agent server (L0 = executor only)
python -m agent.server --config configs/L0_exec_only.yaml --port 8100 --max-steps <official max_steps> --tag task=turning_on_radio

# official evaluator (behavior env, free GPU), pointed at the agent server
python -m omnigibson.eval.eval --task-name turning_on_radio --mode public_test --instance-indices 10 \
  --host 127.0.0.1 --port 8100 --output-dir <run>/eval_output --write-video

# all of the above, with GPU checks, logs and status files
python -m eval.launch --config configs/L0_exec_only.yaml --task turning_on_radio --instances 10 --dry-run

# tables come from logs only
python -m eval.aggregate <run_dir>...
```

## Modularity red lines (each has a test in `tests/`)

1. Every module has a null impl and is off by default. With only the executor, the loop runs a full episode (`configs/L0_exec_only.yaml`).
2. Modules never import each other. They exchange data only through documented `EpisodeState` fields (`agent/state.py`) and interface return values (`agent/interfaces.py`). New data means a new documented state field first.
3. Modules don't change the loop's control flow. `agent/loop.py` branches only on interface return values.
4. Runtime code (agent, modules, executor, providers, eval) never imports `diagnostics/` or simulator APIs (`tests/test_import_guards.py`). The agent only sees whitelisted policy observations (`agent/fields.py`).
5. Prompts live in `configs/prompts/<module>/<version>.txt` and are never shared between modules.
6. Every run records its config and hash, code commit, checkpoint, prompt versions, instances and seed (`run_manifest.json`). Tables are generated from logs, never typed.
7. Switch matrix: each module on/off runs a mock episode (`tests/test_loop_matrix.py`).
8. Truth isolation: adding or changing truth fields in observations leaves decisions unchanged (`tests/test_loop_matrix.py::test_truth_isolation`).
9. A new module needs a README (purpose, paper, I/O, triggers, failures), a null impl, unit tests, and one real-episode on/off comparison before it merges.

## GPU discipline

- Check `nvidia-smi` before every launch. Use only free GPUs (memory under 2 GB, utilization under 10%) and set `CUDA_VISIBLE_DEVICES` explicitly (`eval/gpu.py`).
- Never kill, renice or signal anyone else's processes. Stop only the PIDs you started.
- If no GPU is free, do GPU-free work and write the wait reason in STATUS.md.
- Long jobs run under tmux or nohup. Logs go to `/nas/baiqiao/b1k_runs/logs/<run_id>/` (summaries) and local disk (per-step files). Poll status files; never block in the foreground.

## Experiment discipline

- Tune only on Dev: public_test indices 10-19 (IDs 311-320) and train instances.
- Report is public_test indices 0-9 (IDs 301-310). Run it with frozen configs only, one rollout per instance, no re-runs to pick the best.
- Infrastructure errors may be re-run at most twice on the same (task, instance, seed), and every attempt is recorded. Policy failures are never re-run.
- Comparisons hold task, instance, seed and budget fixed and change exactly one switch (`python -m eval.ablation check A B`). Include the budget-matched "no planner, same retries" control.
- Budget: the 2026 rule is `max_steps × 1 s` per rollout, with the first query free (≤600 s) and each query ≤600 s. Over budget scores 0.
- Five status words, always distinguished: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results. Mock or synthetic data never counts as a result.
- Secrets come only from environment variables (`ASTRA_BASE_URL`, `ASTRA_API_KEY`, `ASTRA_MODEL`, `ANTHROPIC_API_KEY`). Never log them.
- "Astra" is the user's preferred provider name, not a verified model id. Runs with the Claude substitute must be labelled as such.
