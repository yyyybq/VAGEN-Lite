# RoboCasa evaluation

Two ways to evaluate a VLM on RoboCasa kitchens:

1. **Remote (recommended).** Start the policy-venv server, then run VAGEN-Lite
   eval against `RemoteEnv`. This keeps kitchen Python 3.13 off the MuJoCo /
   RoboCasa 3.10 stack.
2. **Local / in-process.** `name: RoboCasa` in `config_local.yaml`. The eval
   process itself must be the policy venv.

## Remote smoke

```bash
# terminal 1 — policy venv (Python 3.10)
bash examples/evaluate/robocasa/run_server.sh

# terminal 2 — VAGEN python
export OPENAI_API_KEY=...
bash examples/evaluate/robocasa/run_eval.sh
```

Or via the first-class CLI:

```bash
python scripts/robocasa_eval.py \
  --config examples/evaluate/robocasa/config.yaml \
  --task PickPlaceCounterToCabinet \
  --backend openai \
  --model gpt-4.1-mini
```

Expand to the full atomic seen split:

```bash
python scripts/robocasa_eval.py --task_set atomic_seen --backend openai
```

## Local smoke

```bash
bash examples/evaluate/robocasa/run_eval_local.sh
```

`concat_multi_turn` is `false` so each turn is `system + current image`,
matching single-turn SFT.

Action format: `<think>...</think><action>[12-D]...</action>`.

## StarVLA continuous-action transfer

For StarVLA checkpoints use `run_starvla_mobile.sh server|client|preflight`.
This entry reads the checkpoint's training contract, preserves all three camera
views, and supplies state only if it was used in training. It records seeded
closed-loop results and runtime failures. `scripts/eval_starvla_heldout.py`
separately evaluates action errors on reserved demonstration episodes.

Full commands and rendering setup: `docs/robocasa_vlm_sft_eval.md`.
