# Active Spatial checkpoint evaluation

This directory contains the test distribution matrix used by
`scripts/active_spatial_eval_sweep.py`. The default matrix excludes
`delta_control` and includes `apparent_size_ordering`.

Filter older JSONL files into no-delta train/test files when needed:

```bash
python scripts/filter_active_spatial_jsonl.py \
  --input data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl \
  --output data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_no_delta.jsonl \
  --exclude delta_control
```

Prepare the OOD split files if they are not already present:

```bash
python scripts/gen_ood_splits.py \
  --train_jsonl data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl \
  --test_jsonl data_gen/active_spatial_pipeline/output_100scenes/test.jsonl \
  --out_dir data_gen/active_spatial_pipeline/ood_splits
```

Quick dry run:

```bash
python scripts/active_spatial_eval_sweep.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps all \
  --steps latest \
  --suites smoke \
  --dry-run
```

Run the latest checkpoint of every started experiment on the full matrix:

```bash
python scripts/active_spatial_eval_sweep.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exps all \
  --steps latest \
  --run
```

Run the full three-layer evaluation stack:

```bash
python scripts/active_spatial_full_eval.py \
  --suite-config examples/evaluate/active_spatial/test_suites.yaml \
  --exp-root exps/vagen_active_spatial \
  --exps all \
  --steps best-val,latest \
  --run
```

An evaluation is complete only when all three result families exist for every
selected checkpoint:

- ID navigation (`id_test`);
- OOD navigation (all configured `ood_*` suites);
- static spatial QA (the complete EASI-8 suite).

For an 8-GPU SCO worker, `sco_active_spatial_eval_entry.sh` runs navigation
with `active_spatial_eval_parallel.py` and then runs checkpoint-level QA with
`active_spatial_qa_parallel.py`. Check both `parallel_completion.json` and
`qa_parallel_completion.json`; a navigation-only completion is not a complete
model evaluation.

Outputs are written under `evaluation/sweeps/active_spatial/<matrix-name>/`,
including generated eval configs, per-run `results_model.json`, `summary.csv`,
`summary.md`, `manifest.jsonl`, `easi_results/`, and `analysis_report.md`.

## GPT-6 Astra ID / OOD

Use the API runner instead of the checkpoint sweep. It writes stratified
slices, then calls `evaluation/run_eval.py` with `provider: openai_responses`.

```bash
export OPENAI_API_KEY=...

# configs + slices only
python scripts/run_gpt6_id_ood_eval.py --mode smoke --dry-run

# 3 ID episodes
python scripts/run_gpt6_id_ood_eval.py --mode smoke --run

# small ID + every OOD axis
python scripts/run_gpt6_id_ood_eval.py --mode canary --run

# protocol-sized subsample (80 ID, 40 / OOD axis)
python scripts/run_gpt6_id_ood_eval.py --mode standard --run
```

Suite file: `examples/evaluate/active_spatial/test_suites_gpt6.yaml`.

### Audit v2 protocol

Checkpoint sweeps inherit the complete runtime environment configuration, including
action preset, done policy, movement sizes, image resolution, success gates and
observation window. Suite-wide defaults no longer overwrite checkpoint settings.
Declare intentional per-suite changes in `protocol_overrides`; differences are
saved with results. Success comes exclusively from environment `traj_metrics.success`,
never from a second scalar-score threshold in the evaluator.

The supplied suites now use `ood_splits_v2` for corrected ID/OOD partitions. ID is
restricted to seen scenes, tasks and atomic categories; instance OOD means new
instances in seen scenes; unseen combinations are separate from unseen categories.
Geometry thresholds are computed from the supplied training manifest. These new
splits do not retroactively certify the legacy camera/metric protocol.

Training and evaluation share bounded observation history (default 1). Evaluation
offsets select rows once without changing their environment indices. Results carry
a protocol/data fingerprint; old incompatible results require a new sweep directory
or an explicit rerun. Old and corrected success rates are not directly comparable.

QA accuracy and balanced accuracy include missing/invalid outputs as errors; ambiguous
Yes/No outputs fail parsing. SaPaVe unwraps nested target objects and reports unknown
visibility when projection evidence is unavailable.

Runbook: `docs/gpt6_id_ood_eval.md`.
Outputs: `evaluation/sweeps/active_spatial/gpt6_astra_id_ood_<mode>_<date>/`.
