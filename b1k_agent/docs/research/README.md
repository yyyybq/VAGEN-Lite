# Research notes (2026-10-21/22)

Source-grounded notes written by research subagents from:
- local clones of BEHAVIOR-1K (v3.9.3-post2 `a8247a8c`, main `7d6718cb`, 2026/eval `38d2c9fb`, group-time-budget `ad934aa6`);
- the openpi fork (`behavior` `0cc8e355`);
- third-party repos (RAX, Isaac-GR00T fork, RLC, Comet, MobiAgent, ...);
- web-search snippets.

Every claim carries a citation and an evidence label (code / docs / search snippet / inferred). `R9_critic.md` cross-checks R1–R8 and lists errata; read it before relying on any single report. Two corrections to keep in mind:
- R6's `updates.md` line numbers are wrong;
- exact `max_steps` values supersede the estimates in R1–R6.

`docs/PROTOCOL.md` is the condensed, maintained version. These notes are the evidence behind it. Paths like `/home/user/...` and `/tmp/claude-0/...` refer to the cloud session that wrote them; the clones are not part of this repo.

| File | Topic |
|---|---|
| R1_official_rules.md | rules as stated in `docs/challenge/*.md` at several refs |
| R2_evaluator_code.md | official evaluator code at v3.9.3-post2 (obs/action/protocol/termination/outputs) |
| R3_eval_branch_diff.md | what `2026/eval` adds (timing rules, multi-port, failure records) |
| R4_openpi_b1k.md | openpi fork B1K integration (π0.5 config, transforms, serving, key mismatch) |
| R5_checkpoints.md | executor checkpoint candidates (official, 2025 community, 100-task community) |
| R6_tasks_dataset.md, R6_task_table_full.md | the 100 tasks, instances, dataset, annotations, Dev proposal |
| R7_qscore_aggregation.md | Q-score and aggregation exactly as implemented |
| R8_papers_astra.md | reference papers and the "Astra" model |
| R9_critic.md | cross-check, gap fills (exact max_steps, tasks.jsonl, training prompt), errata |
