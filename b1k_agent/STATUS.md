# STATUS

Last updated: 2026-10-21 (session `claude/b1k-agent-baseline`, cloud container).

## Current stage

Step 1 (takeover) is done only for the parts that do not need server 105:
- official rules verified from source;
- `docs/PROTOCOL.md` first version written;
- framework code written and tested on mocks.

Steps 1 (on 105) to 4 are **blocked**: this session runs in a cloud container. It has no `ssh`, no `/nas`, no GPU, and the network policy blocks behavior.stanford.edu / huggingface.co / arxiv.org. Next actions on 105: `docs/RUNBOOK_105.md`.

## Five states

### 1. Planned (no code yet)
- M-approach (depth-based distance to target).
- M-memory retrieval (RAG) and bridge.
- M-score.
- M-distill small checker.
- M-spatial.
- M-explore.
- M-skill executor.
- A joint plan+check call (one model call, two interface returns).
- GR00T executor adapter.
- Comet / RLC executor adapters.

### 2. Code written (not yet exercised beyond unit tests)
- `executor/openpi_server/serve_b1k_prompted.py`: needs the openpi venv; never run.
- `providers/openai_compat.py` (Astra) and `providers/anthropic_sdk.py` (Claude substitute): never called against a real endpoint.
- `diagnostics/replay_truth.py`: needs the simulator.
- `scripts/*` for 105.

### 3. Interface-verified (mock / synthetic tests pass)
- **Agent server ↔ an official-style client**: `/healthz`, metadata, reset, batched obs, the reconnect resend.
- **Executor client ↔ a fake prompt-capable openpi server**: prompt override, flush, chunk mode.
- **Loop:**
  - L0 passthrough identity;
  - each module on/off;
  - leave-one-out;
  - truth isolation;
  - batched envs with separate state;
  - the budget regression.
- **Modules with mock providers:** plan.llm, check.llm / check.rule, goal.bddl, recover.scripted.
- **BDDL reader:** checked on real task files (the 100 2026 tasks all have problem0.bddl).
- **Config inheritance and hashing.**

### 4. Simulator-run
- **None.**

### 5. Full-eval results
- **None.**

## Verified facts (see docs/PROTOCOL.md for sources)
- **Version:** the BEHAVIOR-1K tag `v3.9.3` does not exist. The real tags are `v3.9.3-post1` (`bd049de3`, used for final hidden tests) and `v3.9.3-post2` (`a8247a8c`, the one on 105, custom-robot fix).
- **Timing rules:** the 2026 rules (max_steps × 1 s budget after a free first query, 600 s per query, over budget = 0) exist only on branch `2026/eval` (`38d2c9fb`).
- **Instances:** Report = public_test indices 0–9 (IDs 301–310); Dev = 10–19 (IDs 311–320).
- **Robot-name mismatch:** the openpi fork expects `robot::` keys; the official evaluator sends `robot_r1::`. Our serving script renames keys.
- **No official multi-task executor exists.** The official checkpoints are `turning_on_radio` only (π0.5 and GR00T N1.7). The multi-task options are community 2025 50-task weights (Comet π0.5 with text prompts, RLC task-ID only) and a community 100-task GR00T N1.7 fine-tune (unverified).
- **No released executor is trained on skill-level language.**

## Blocks and minimal conditions to clear them

| Block | Minimal condition |
|---|---|
| No access to server 105 (Steps 1–4, any simulator run) | Run the runbook on 105 yourself, or start a Claude Code session there (`claude remote-control` in `/nas/baiqiao/b1k_agent`), or give this environment ssh access |
| Smoke-test watcher state unknown | `scripts/takeover_check.sh` on 105 |
| GPUs busy (8 GPUs used by egor at hand-off) | Wait; `eval/gpu.py` only takes GPUs with < 2 GB used and < 10% utilization; never touch others' processes |
| Astra endpoint unknown | Set `ASTRA_BASE_URL`, `ASTRA_API_KEY`, `ASTRA_MODEL` on 105 and run `python -m providers.probe ...`; a Claude substitute is wired but its results must be labelled |
| Multi-task executor | Check on 105 whether HF downloads work; inspect candidates (PROTOCOL §12); pick one that loads and runs through the official interface |
| Web pages, HF, arXiv unreachable here | Read the docs from the repo; the papers listed in the spec are NOT verified (see `docs/PAPERS.md` once written) |

## Next steps (ordered by expected gain / cost / dependency)

1. On 105, follow `docs/RUNBOOK_105.md`:
   - Step 1 takeover script, including reading the `b1k_runs` scripts;
   - Step 2 smoke result;
   - Step 3: `--direct` baseline episode and the L0 passthrough episode on Dev instance 311, both wrappers;
   - Step 4: throughput and determinism.
   - **Gives the action base and official numbers; nothing else counts until then.**
2. Probe the Astra endpoint (latency decides the check interval within the 1 s/step budget).
3. Step 5:
   - download and load the executor candidates;
   - run the skill-instruction diagnosis (`scripts/skill_probe.py`).
4. Step 6:
   - L0 on 5 Dev tasks (from the executor's covered tasks);
   - failure distribution (`eval.failures`).
5. Step 7:
   - `S7_min_loop` vs `L0_exec_only` vs `C_budget_matched` on the same Dev set: the first comparison table.
