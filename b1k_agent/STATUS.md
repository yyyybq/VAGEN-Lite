# STATUS

Last updated: 2026-10-22. Work so far comes from the cloud session on branch `claude/b1k-agent-baseline` of github.com/yyyybq/VAGEN-Lite (draft PR #3); see `docs/RUNBOOK_105.md` §0 for turning it into a standalone repo.

## Current stage

Done, where server 105 is not needed:
- **Step 1 desk part:** official rules verified from source (`docs/PROTOCOL.md`, evidence in `docs/research/`).
- **Framework:** code written and tested on mocks.
- **Tooling for Steps 1–7:** written and tested on mocks.

**Blocked:** Step 1 on 105 (watcher, scripts, versions, checkpoint, datasets) through Step 7. This session runs in a cloud container: no `ssh`, no `/nas`, no GPU, and behavior.stanford.edu / huggingface.co / arxiv.org are blocked by the network policy.

## Five states

### 1. Planned (no code yet)
- M-approach (depth grounding + base adjustment; Cracking-paper recipe, `docs/PAPERS.md`).
- M-memory retrieval and bridge.
- M-score, which also needs multi-candidate executors.
- M-distill small checker. Its data export and labelling are written.
- M-spatial and M-explore.
- M-skill executor (fine-tuning on the 2026 skill annotations).
- A joint plan+check call.
- Before/after frame comparison in M-check.
- Executor adapters for Comet / RLC / GR00T.
- A chunk-returning agent server (`--replay-action-chunk-size`).

### 2. Code written, needs 105 to exercise
- `executor/openpi_server/serve_b1k_prompted.py` (openpi venv).
- `diagnostics/replay_truth.py` (behavior env).
- `scripts/takeover_check.sh`, `inspect_checkpoint.py`, `inventory_datasets.py`, `measure_throughput.py`, `gpu_sampler.sh`, `skill_probe.py`, `record_obs_fixture.py`, `setup_agent_env.sh`.
- `eval/launch.py` / `eval/batch.py` real process management.
- `providers/openai_compat.py` (Astra) and `providers/anthropic_sdk.py` (Claude substitute): never called against a real endpoint.

### 3. Interface-verified (mock / synthetic tests; `python -m pytest -q` → 381 passed)
- **Agent server ↔ an official-style client:** `/healthz`, metadata, reset without reply, batched obs, the 2026/eval reconnect resend.
- **Executor client ↔ a fake prompt-capable openpi server:** override, flush, chunk mode, multi-task.
- **Loop:**
  - L0 passthrough identity;
  - every slot on alone;
  - leave-one-out;
  - truth isolation (added or changed truth keys → identical decisions);
  - read-only StateView;
  - prompt isolation;
  - import guards;
  - batched envs with separate state;
  - the budget regression.
- **Modules with scripted providers:** plan.llm, check.llm, check.rule, goal.bddl, recover.scripted.
- **BDDL reader** on the 100 task files.
- **Task metadata from the snapshots:** 100 tasks, exact `max_steps`, NL instructions.
- **eval:** launch/batch with fake processes; aggregate/ablation/failures on synthetic official JSONs.
- **data / diagnostics** on synthetic logs.

### 4. Simulator-run
- **None.**

### 5. Full-eval results
- **None.**

## Verified facts that shape the plan (details and sources: `docs/PROTOCOL.md`, `docs/research/`)

- **Versions:**
  - The tag `v3.9.3` does not exist.
  - Final hidden tests use `v3.9.3-post1`; 105 has `v3.9.3-post2` (custom-robot fix only).
  - The 2026 timing rules exist only on branch `2026/eval` (`38d2c9fb`): `max_steps` × 1 s after a free first query, 600 s per query, over budget = 0.
- **Instances:** Report = public_test 0–9 (IDs 301–310). Dev = 10–19 (311–320) plus train IDs (0–299 for tasks 0–49, 1–300 for tasks 50–99).
- **Steps:** exact `max_steps` for every task (`configs/tasks/`); turning_on_radio = 3224.
- **Q-score:** literals false at start and true at the final step, divided by the option length, maximised over options; 1.0 on success. Initially-true literals cap partial credit.
- **openpi key mismatch:** the fork reads `robot::` keys; the evaluator sends `robot_r1::`. `serve_b1k_prompted.py --rename_robot_from robot_r1` fixes it. Check how the `b1k_runs` scripts handle it.
- **Radio checkpoint prompt:** the π0.5 radio checkpoint was probably trained on "turning on radio" (snake_case task name) but is served with the NL instruction. Prompt swapping is not expected to steer it.
- **Executors:**
  - No official multi-task executor exists.
  - Community options: Comet π0.5 (tasks 0–49, NL prompts), RLC (tasks 0–49, task-ID only), GR00T N1.7 100-task (unverified).
  - No released executor is trained on skill language.
- **"Astra"** is most likely OpenAI GPT-6 Astra (`gpt-6-astra`), per search snippets only. Verify with `GET /v1/models` and `python -m providers.probe`.

## Blocks and minimal conditions to clear them

| Block | Minimal condition |
|---|---|
| No access to server 105 (all of Steps 1–7 that run anything) | Run `docs/RUNBOOK_105.md` on 105, or start a Claude Code session there (`claude remote-control` in `/nas/baiqiao/b1k_agent`) |
| Smoke-test watcher state unknown | `scripts/takeover_check.sh` on 105 (read-only) |
| GPUs were all busy at hand-off | `eval/gpu.py` takes only GPUs with < 2 GB used and < 10% utilization; wait otherwise; never touch others' processes |
| Astra access | On 105, set `ASTRA_BASE_URL`, `ASTRA_API_KEY`, `ASTRA_MODEL`, then run `python -m providers.probe --config configs/S7_min_loop.yaml --provider astra`. Its latency decides how often M-plan / M-check can run inside the 1 s/step budget |
| Multi-task executor | On 105: can HF be reached? Download, load and run candidates through the official interface (PROTOCOL §12); then the skill probe |

## Next steps (expected gain / cost / dependency)

1. **105 Steps 1–3** (`docs/RUNBOOK_105.md`): the takeover report; the smoke result; the official `--direct` baseline episode and the L0 passthrough episode on Dev instance 311, with both wrappers. Nothing else counts until a real episode exists.
2. **Step 4:** throughput and determinism → fills PROTOCOL §11 and sets every later experiment size.
3. **Probe Astra** (or label a substitute).
4. **Step 5:** executor candidates; skill probe (`configs/skills/probe_turning_on_radio.txt`).
5. **Step 6:** L0 on 5 Dev tasks (PROTOCOL §13; choose set A/B/C by executor coverage) → failure distribution (`eval.failures`).
6. **Step 7:** `S7_min_loop` vs `L0_exec_only` vs `C_budget_matched` on the same Dev set → the first comparison table (`eval.aggregate --compare`).
