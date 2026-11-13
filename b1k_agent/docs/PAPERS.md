# Reference papers: what was verified and what we take from them

Check date: 2026-10-22. arxiv.org and huggingface.co were unreachable from the session that wrote this, so evidence comes from:

- **[repo]:** the authors' GitHub code, cloned read-only;
- **[pdf]:** the paper PDF shipped in an author repo;
- **[search]:** web-search snippets only. Treat these as unverified.

Numbers from different papers are **not comparable**: different task sets, metrics, evaluators and budgets. Several papers did not use the official 2026 evaluator.

| arXiv | Paper (verified title) | Code / weights | BEHAVIOR-1K evidence | Our module | Verdict for our rules |
|---|---|---|---|---|---|
| 2610.03476 | **MobiAgent**: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation (HKU / Astribot / SUSTech) | code [repo `kaiknower/MobiAgent` ffce244]; no BEHAVIOR weights | 4 tasks, 65.0% vs 42.5% (π0.5-TA) [search]; own 224² env, **not** the official evaluator [repo] | M-plan, M-check, M-skill | Closest template for our loop. Adopted (see below) |
| 2610.12089 | **ManiUnit**: A Manipulation Skill Dataset and Benchmark for Long-Horizon Tasks | code NOT FOUND | built on BEHAVIOR-1K annotations; skill policy 78.7% vs task policy 49.3% local success; planner + skills 4% → 18% full success [search] | M-skill | Evidence that skill-segment training is needed for instruction-following executors |
| 2606.04968 | Potential-Guided Flow Matching for VLA Policy Improvement (method "**ForesightFlow**") | code NOT FOUND | 5 BEHAVIOR tasks, numbers not seen | M-score | Later |
| 2607.18060 | **RoboHarness**: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning (Huawei Noah's Ark et al.) | official code NOT FOUND (repo `BinceQu/RoboHarness` is a different paper, 2610.14018) | 6 tasks, SR 0.43 / progress 0.66 [search] | M-memory | Later; re-check the geometric Memory Bridge against observation limits |
| 2610.19600 | **Cracking Mobile Manipulation**: From Failure Diagnosis to Agentic Robot System (PKU / DAMO et al.) | PDF in repo; code "coming soon" | 5 tasks from the 2026 challenge: 37/100 vs 14/100 (harness VLA) vs 7/100 (π0.5) [pdf] | M-approach, M-recover | Failure-first method matches our protocol. Its geometric skills use RGB-D + proprio, which are legal |
| 2609.20791 | **StageGuard**: Learning Stage Transitions … via Agentic Distillation (Huawei Noah's Ark) | code NOT FOUND | evaluated on BEHAVIOR-1K; ">90%" unverified | M-distill | Later: strong-model stage labels → small checker |
| 2606.12956 | **SERF**: Spatiotemporal Environment and Robot Feature Map … | code + weights [repo] | 3 tasks, 44.0–58.7% [search] | M-spatial | **Its evaluation uses inputs the 2026 track forbids** (`seg_instance`, world camera poses, prebuilt per-instance maps, a patched BDDL goal) [repo]. Not usable as is |
| 2609.35249 | **Spatial Grafting**: Grounding 3D Features for Flow-Matching Robot Policies | code NOT FOUND | 6 tasks [search] | M-spatial (features) | Later |
| 2610.22496 | Scaling Online Exploration for Robot Agents (**RAX**; Harvard / NVIDIA / UCB / MIT / Stanford) | code, Apache-2.0, with a BEHAVIOR branch [repo] | 6 tasks, Q 46.5 vs 8.2 [search]; pins v3.9.3-post1 | M-explore (default off) | Tools such as cuRobo/IK inside a wrapper need a legality check. Its task catalog gave us the exact `max_steps` table (`configs/tasks/`) |
| 2608.11739 | **G0.5**: One Autoregressive Stream for Robot Reasoning and Action (Galaxea) | code + open weights, **non-commercial** license [repo] | README claims 0.3136; no BEHAVIOR checkpoint or entry point | M-exec alternative | Not usable until a BEHAVIOR checkpoint exists |

## What we adopt now, and where

**MobiAgent planner/judge contract** [repo `src/mobiagent/execution/`]:
- **One subtask per planner call**, with an explicit success check and failure cues. Our `plan.llm` returns one `Subtask` per call.
- **The orchestrator, not the LLM, owns retry budgets and termination.** Our loop holds `max_retries_per_subtask` and `max_replans`, and the evaluator alone terminates.
- **The judge returns `complete / incomplete / error` from head and wrist frames.** "Do not infer success from elapsed time, a task name, or an earlier verdict." That matches our `check.llm` rules: visible evidence only, low confidence → unknown.
- Still to add: before/after frame comparison (the previous attempt's last frames), and `success_check` / `failure_cues` fields on `Subtask`. These need an interface change.

**Cracking Mobile Manipulation** [pdf]:
- Failure-diagnosis first, then targeted interventions. This is exactly our Step 6 → module order.
- Its approach and recovery skills use only RGB-D and proprio, so they are candidates for M-approach:
  - grounding plus depth back-projection;
  - reach thresholds of about 0.75–1.0 m (to be re-estimated for R1Pro);
  - a local XY shift of at most 0.25 m;
  - A* on a 0.05 m depth occupancy grid.
- "Gripper closed" is not proof of a grasp. Our `check.rule` estimates holding from finger width, not from the command.

**ManiUnit / MobiAgent skill training:** executors trained on full-task prompts are not instruction-following skill executors. Skill-segment training (the 2026 demo annotations: 31 skill types) is the route if M-skill is pursued. This is consistent with the Step-5 diagnosis requirement.

## The "Astra" model

- The only API-served model named Astra we found is **OpenAI GPT-6 Astra**, model id `gpt-6-astra` (announced 2026-09-03), per search snippets of https://developers.openai.com/api/docs/models/gpt-6-astra and https://openai.com/index/gpt-6-astra/.
- Per the same snippets: text and image input, structured outputs supported, Chat Completions and Responses endpoints, about $10 / $50 per 1M input / output tokens.
- The rollout started limited, and general availability for our account is unknown.
- **Unverified.** Confirm the id with `GET /v1/models` and capabilities with `providers/probe.py` before any run. No dated snapshot id was found; do not guess one.
- Other "Astra"s, which are not API model ids: Google's Project Astra (research prototype) and ByteDance Astra (robot navigation, arXiv 2506.06205).
