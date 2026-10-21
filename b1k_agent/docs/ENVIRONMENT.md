# Environments and dependencies

Rule (task spec §2): do not reinstall or upgrade existing environments. Any new dependency is recorded here with its exact command and the reason, and goes into a NEW environment.

## Existing on server 105 (do not modify)

These facts come from the hand-off notes. They have not been re-verified from this cloud session; Step 1 records the versions with `scripts/takeover_check.sh`.

| Env | Path | Used for |
|---|---|---|
| behavior (conda) | `/nas/baiqiao/envs/behavior` | Isaac Sim 5.1 + OmniGibson (BEHAVIOR-1K v3.9.3-post2), the official evaluator |
| openpi (uv venv) | `/nas/baiqiao/openpi/.venv` | π0.5 serving; our `executor/openpi_server/serve_b1k_prompted.py` |

Notes on the behavior env (hand-off):

- A newer libstdc++ (GLIBCXX_3.4.30) is preloaded inside the env, and only that library.
- `warp-lang==1.12.0` is installed separately, because `import omnigibson` without `--primitives` needs it.
- `pyarrow==21.0.0` is pinned.
- Required environment: `OMNIGIBSON_HEADLESS=1`, `OMNI_KIT_ACCEPT_EULA=YES`; appdata lives in `~/.omnigibson_appdata`.
- Global git `safe.directory` entries exist for the BEHAVIOR-1K and openpi checkouts, because files on NFS show as owned by `nobody`.

If the evaluator needs extra environment setup beyond the env's activate hooks, put it in a file and point `B1K_ENV_SETUP` at it. `eval/launch.py` sources it before starting the evaluator. Reuse what the scripts in `/nas/baiqiao/b1k_runs/` already do.

## Agent server and tools

The agent server needs Python ≥ 3.11 with:

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| websockets | 15.0.1 |
| msgpack | 1.1.0 |
| pyyaml | 6.0.2 |
| pillow | 11.2.1 |
| httpx | 0.28.1 |

Option A (no install): run it with the openpi venv's interpreter, `PYTHONPATH=/nas/baiqiao/b1k_agent /nas/baiqiao/openpi/.venv/bin/python -m agent.server ...`. These exact versions come from openpi's `uv.lock` at `0cc8e355`.

Option B (new env, recommended once you want pytest or the Claude provider) creates `/nas/baiqiao/envs/b1k_agent`:

```bash
bash scripts/setup_agent_env.sh                    # uv venv -p 3.11 + the pins above + pytest==8.3.4
bash scripts/setup_agent_env.sh --with-anthropic   # additionally: pip install anthropic
```

| New package | Command | Why |
|---|---|---|
| pytest 8.3.4 | `pip install pytest==8.3.4` (new env only) | run `tests/` on 105 |
| anthropic | `pip install anthropic` (new env only) | `providers/anthropic_sdk.py`, the Claude substitute provider. Its results must be labelled "claude-substitute", not Astra |

Local tests in the cloud session ran on Python 3.11.15 with the pins above, using `uv venv` in a scratch directory.

## Secrets

Secrets come from environment variables only and are never written to configs or logs:

| Variable | Used for |
|---|---|
| `ASTRA_BASE_URL`, `ASTRA_API_KEY`, `ASTRA_MODEL` | the "astra" provider (OpenAI-compatible endpoint; nothing assumed) |
| `ANTHROPIC_API_KEY` | the Claude substitute (other SDK credential sources also work) |
