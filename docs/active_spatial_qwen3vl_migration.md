# Active Spatial: Qwen3-VL migration

The existing recipe is Qwen2.5-VL-specific.  Use a **separate environment**
for Qwen3-VL; do not overwrite the validated Qwen2.5 environment.

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m venv \
  --system-site-packages .venv-qwen3vl-runtime
.venv-qwen3vl-runtime/bin/python -m pip install --no-deps 'transformers==4.57.6'
.venv-qwen3vl-runtime/bin/python scripts/check_qwen3_vl_env.py
```

Do not install an unconstrained latest Transformers: v5 removed
`AutoModelForVision2Seq`, which this VAGEN/verl revision imports. This runtime
reuses the known-good Torch 2.8.0, vLLM 0.11.0, and SGLang 0.5.2 from
`vagen-lite`, while isolating Transformers 4.57.6.

The shared Zoetrope checkpoint is under
`/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`.
The check is intentionally offline and does not download weights. To run the
one-update smoke:

```bash
QWEN3_VL_MODEL=/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/qwen3vl_active_spatial_smoke.sh
```

Qwen3-VL's processor emits `mm_token_type_ids`; unlike Qwen2.5-VL it computes
multimodal RoPE inside the model.  The no-concat agent loop therefore preserves
all processor metadata and does not call `verl.models.transformers.qwen2_vl`.
The smoke must reach one actor update with finite loss and one renderer turn
before launching a long run.
