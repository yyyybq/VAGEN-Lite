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

Outputs are written under `evaluation/sweeps/active_spatial/<matrix-name>/`,
including generated eval configs, per-run `results_model.json`, `summary.csv`,
`summary.md`, `manifest.jsonl`, `easi_probe_results/`, and `analysis_report.md`.
