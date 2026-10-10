# Active Spatial paired Yes/No QA

端到端训练说明见 [数据与混合 SFT](../../docs/active_spatial_pipeline_data.md)。
生成训练 bank 时提供 `--scene-root` 和 `--env-yaml`：前者用于合法室内相机检查，
后者绑定与轨迹/RL 一致的评分配置。缺少几何上下文时标签会被撤回。
渲染后的 bank 可用 `scripts/prepare_active_spatial_sft.py` 与 R1 轨迹按场景共同划分。

This adapter reuses `vagen.envs.active_spatial.canonical_task_metrics` and
`SpatialPotentialField`; it does not define a second evaluator.  A parent goal
is expanded into multiple pose states and each label is the exact Active
success predicate result.  Samples without an image rendered for that pose are
kept for oracle/audit checks but marked `invalid_missing_render` and excluded
from an observable QA test bank.

Generate a resumable, versioned bank:

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m \
  data_gen.active_spatial_qa.generate_paired_qa \
  --input <pipeline-or-state-jsonl> --output-dir <qa_v1> --split train --limit 100
```

Evaluate frozen predictions (format failures remain in the denominator):

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m \
  data_gen.active_spatial_qa.qa_eval --bank <qa_v1/manifest.jsonl> \
  --predictions <predictions.jsonl> --require-observable --output <qa_eval.json>
```

`contact_sheet.html`, `task_contract.json`, `summary.json`, and `errors.jsonl`
are emitted beside the manifest.  `run_cross_canary.py` writes the first
Base/QA-trained/Act-trained matrix with explicit `NOT_RUN` cells until frozen
model predictions and Active rollouts are supplied.

Render exact-pose states with the existing UnifiedRenderGS HTTP/client/local
backend (failed samples are retained in `errors.jsonl`):

```bash
python -m data_gen.active_spatial_qa.render_paired_bank \
  --bank <qa_v1/manifest.jsonl> --output-dir <qa_v2_rgb> \
  --backend http --renderer-url http://<renderer>:8768
```

Before model evaluation, run the environment-side predicate check:

```bash
python -m data_gen.active_spatial_qa.runtime_compare \
  --bank <qa_v1/manifest.jsonl> --output <runtime_compare.json>
```

Frozen VLM inference is provided by `model_qa_eval.py`; it refuses to run when
the common observable-ID set is empty and emits `EMPTY_BLOCKED` instead.
