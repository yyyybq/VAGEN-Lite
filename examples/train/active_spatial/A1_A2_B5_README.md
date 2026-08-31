# A1 / A2 / B5 Experiment Pack

Prepared from current_main EASI-8 analysis (v46 nav↑ / EASI flat; c8 weak).

## Unique variables

| Exp | Base | Unique variable | Success criteria |
|-----|------|-----------------|------------------|
| **A1** | `v46_7b_nodelta_w3_rerun` | `SPATIAL_AUX_RATIO` ∈ {0.05, 0.10} (MCQ mix fraction; answer reward coef fixed at 1.0) via SITE MCQ mix | EASI macro ≥ +1.5pp; ID400/OOD drop ≤ 2pp |
| **A2** | v46 | `enable_explicit_done=true`, `enable_auto_termination=false`, `premature_done_penalty=-2.0` | lower `stop_fail` / timeout; success↑ |
| **B5** | c8 v19 | train `limit_images=25` + `max_model_len=16384`; stronger format/invalid penalties | if ID400 still ≪ v46 → close Cambrian line |

## Files

- Env YAMLs: `env_config_a1_spatial_mcq.yaml`, `env_config_a2_v46_arrival_stop.yaml`, `env_config_b5_c8_actionvalid.yaml`
- Experiments: `experiments/a1_v46_spatial_aux_w005.sh`, `a1_v46_spatial_aux_w010.sh`, `a2_v46_arrival_stop.sh`, `b5_c8_wrapper_img25_actionvalid.sh`
- Aux env: `vagen/envs/spatial_mcq/` (registered as `SpatialMCQ`)
- Data prep: `scripts/prep_spatial_aux_mcq.py`
- Launcher: `launch_a1_a2_b5.sh`
- `run_experiment.sh` supports `AUX_ENV_CONFIG` + `SPATIAL_AUX_RATIO`
- Eval overrides: `VAGEN_CAMBRIAN_LIMIT_IMAGES`, `VAGEN_CAMBRIAN_MAX_MODEL_LEN`

## Launch

```bash
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite

# 1) build aux MCQ data (SITE image, ~2000 items)
bash examples/train/active_spatial/launch_a1_a2_b5.sh prep

# 2) dry-run
bash examples/train/active_spatial/launch_a1_a2_b5.sh dry-run

# 3) start one arm (needs free 8-GPU node + render service for A1/A2)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_GPU_HOLDER=false \
  bash examples/train/active_spatial/launch_a1_a2_b5.sh run a1_w005

# A2 / B5
bash examples/train/active_spatial/launch_a1_a2_b5.sh run a2
bash examples/train/active_spatial/launch_a1_a2_b5.sh run b5
```

A1/A2 expect remote GS render at `10.119.18.163:8767` (same as v46). Override `RENDER_HOST` in the experiment script if needed.

## Eval notes

- Navigation: same suites as current_main (`id_test_stratified400` + OOD splits).
- EASI-8: only required for A1 decision; A2/B5 can use nav metrics first.
- B5 eval:
  ```bash
  export VAGEN_CAMBRIAN_LIMIT_IMAGES=25
  export VAGEN_CAMBRIAN_MAX_MODEL_LEN=16384
  ```

## Control

A1 control arm is existing `v46_7b_nodelta_w3_rerun` (aux ratio = 0). Do not retrain a zero-aux clone unless you need a matched seed.
