# b1k_agent

A performance-oriented baseline for the **BEHAVIOR-1K 2026 Challenge** (100 tasks). A strong model does the high-level scheduling, an off-the-shelf action executor (π0.5 now) does the low-level control, and checking, recovery, goal management and memory are pluggable modules. Every module is behind one config switch and has a null implementation.

Scores come only from the **official evaluator**, which we use unchanged. Our agent is the websocket policy server it connects to.

> **Status (2026-10-21):** code written, all tests passing on mock/synthetic data. **Nothing has run in the simulator yet.** See `STATUS.md`. No results exist; none are published.

Documents:
- `docs/TASK_SPEC_20261022.md`: the task spec.
- `docs/PROTOCOL.md`: verified evaluation rules and our protocol.
- `docs/ARCHITECTURE.md`: design and data flow.
- `docs/RUNBOOK_105.md`: Steps 1–4 on server 105.
- `CLAUDE.md`: rules for future sessions.

## Layout

```
agent/        loop, interfaces, EpisodeState, obs whitelist, BDDL reader, registry, recorder, websocket server
modules/      one package per trick: plan, check, goal, recover, approach, memory, score, spatial, explore
              (+ skill, distill data/impl packages); each has null.py, impls, README.md
executor/     executor clients (openpi_ws, hold) + openpi_server/serve_b1k_prompted.py (runs in the openpi venv)
providers/    strong-model clients: openai_compat (Astra), anthropic (Claude, labelled substitute), scripted/replay
eval/         launch official runs, batches, aggregation, ablation configs, failure records
data/         schemas: skill segments, experiences, distillation records, training rounds
diagnostics/  the ONLY place allowed to read simulator truth (offline analysis / labels)
configs/      base + ladder (L0..L4), Step-7 loop (S7), budget-matched control (C), prompts, skill catalogs, batches
scripts/      server-105 operations: takeover inventory, checkpoint / dataset inspection, throughput, skill probe
tests/        red lines, protocol, units (mock / synthetic only)
```

## Environment

Do not reinstall the existing envs (`docs/RUNBOOK_105.md` §0). Three processes are involved:

| Process | Environment |
|---|---|
| Simulator and official evaluator | `conda activate /nas/baiqiao/envs/behavior` (Isaac Sim 5.1 + OmniGibson, BEHAVIOR-1K v3.9.3-post2) |
| Executor server | `/nas/baiqiao/openpi/.venv` |
| Agent server and tools | any Python ≥ 3.11 with `numpy<2 pyyaml websockets msgpack pillow httpx` — the openpi venv already has the exact versions (`PYTHONPATH=/nas/baiqiao/b1k_agent`), or create a new venv with `scripts/setup_agent_env.sh` |

Strong-model credentials come from environment variables only:
- `ASTRA_BASE_URL`, `ASTRA_API_KEY`, `ASTRA_MODEL`: Astra, via an OpenAI-compatible endpoint. Nothing about Astra is assumed; probe it first.
- `ANTHROPIC_API_KEY`: Claude substitute. Install with `pip install anthropic` into the agent venv only.

## Commands

Run everything from the repo root.

```bash
# Tests (no simulator needed)
python -m pytest -q

# Probe a strong-model endpoint: text, JSON schema, vision, latency (keys never printed)
python -m providers.probe --config configs/S7_min_loop.yaml --provider astra --out docs/provider_probe_astra.json

# One official episode through the agent (Step 3; picks a free GPU, starts executor + agent + evaluator,
# writes status.json / commands.sh / logs under /nas/baiqiao/b1k_runs/logs/<run_id>/)
python -m eval.launch --config configs/L0_exec_only.yaml --task turning_on_radio --mode public_test \
    --instances 10 --wrapper default --seed 0 --write-video --run-id s3_L0_radio_311 --detach
python -m eval.launch status /nas/baiqiao/b1k_runs/logs/s3_L0_radio_311
#   --direct  : evaluator -> executor server with no agent in between (official baseline reference)
#   --dry-run : print the commands only

# Batches (infrastructure errors re-run at most twice; policy failures never)
python -m eval.batch configs/batches/step3_turning_on_radio.yaml

# Tables, from logs only (per episode / per task / per config, task-stratified bootstrap CI)
python -m eval.aggregate /nas/baiqiao/b1k_runs/logs/<run_id> ... --out results/<name>

# Ablation configs: forward ladder and leave-one-out; each pair differs in exactly one module switch
python -m eval.ablation ladder --base configs/L0_exec_only.yaml --steps plan,check,goal,recover --from configs/L4_recover.yaml --out configs/ladder/
python -m eval.ablation loo --final configs/L4_recover.yaml --out configs/loo/
python -m eval.ablation check configs/L2_plan_check.yaml configs/L3_goal.yaml

# Failure records (task spec 6.7): skeletons from decision logs, labelled afterwards
python -m eval.failures skeleton /nas/baiqiao/b1k_runs/logs/<run_id> --out failures.jsonl
python -m eval.failures summarize failures.jsonl
```

Manual start, without `eval.launch` (for example inside the user's own `b1k_runs` scripts):

```bash
# 1) executor server (openpi venv); --repo_id = the asset id under <ckpt>/assets/
cd /nas/baiqiao/openpi && CUDA_VISIBLE_DEVICES=<free gpu> XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 .venv/bin/python \
  /nas/baiqiao/b1k_agent/executor/openpi_server/serve_b1k_prompted.py --robot b1k/R1Pro --task b1k/turning_on_radio \
  --repo_id <asset-id> --policy.config pi05_b1k --policy.dir /nas/baiqiao/checkpoints/pi05_turn_on_the_radio \
  --control_mode receding_horizon --action_horizon 16 --port 8000 --rename_robot_from robot_r1
# 2) agent server
cd /nas/baiqiao/b1k_agent && PYTHONPATH=. /nas/baiqiao/openpi/.venv/bin/python -m agent.server \
  --config configs/L0_exec_only.yaml --port 8100 --max-steps <task max_steps> --tag task=turning_on_radio --tag instance=311
# 3) official evaluator (behavior env, free GPU)
cd /nas/baiqiao/BEHAVIOR-1K && python -m omnigibson.eval.eval --task-name turning_on_radio --mode public_test \
  --instance-indices 10 --host 127.0.0.1 --port 8100 --output-dir <run_dir>/eval_output --write-video
```

## Configurations

| Config | Modules on | Purpose |
|---|---|---|
| `L0_exec_only` | none | executor only; must equal the official baseline action stream |
| `L1_plan` | plan.llm | ladder step 1 |
| `L2_plan_check` | + check.llm | ladder step 2 |
| `L3_goal` | + goal.bddl | ladder step 3 |
| `L4_recover` | + recover.scripted | ladder step 4 |
| `S7_min_loop` | plan.llm + check.llm + recover.scripted | Step-7 minimal closed loop |
| `C_budget_matched` | check.rule + recover.scripted | "no commander, same retries" control (spec 6.3) |

Results are reported only from Dev while tuning; Report instances are run once, on frozen configs (`docs/PROTOCOL.md` §4).
