# Model Backbone Integration Guide

Date: 2026-07-20

This note explains what model backbones VAGEN-Lite can currently train, how Cambrian-S is wired into the framework, and what needs to be done when adding a new model family.

For the SenseNova-U1 Active Spatial integration and experiment checklist, see
[`docs/sensenova_u1_active_spatial.md`](sensenova_u1_active_spatial.md).

## 1. Current Training Support

VAGEN-Lite is model-agnostic only up to a point. The RL loop itself can use any HuggingFace/vLLM-compatible model, but multimodal training also depends on image processor behavior, rollout-server support, log-prob support, and FSDP actor/critic loading.

### Directly Supported / Lowest Friction

These are the easiest path because the model is already supported by HuggingFace Transformers, the processor is loadable through `AutoProcessor`, and vLLM/verl already knows how to handle the model family.

| Model family | Examples in repo | Expected work |
|---|---|---|
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct` | Usually just set `MODEL_PATH`, adjust GPU/TP/batch. |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct`, old active-spatial warmer for `Qwen/Qwen3-VL-8B-Instruct` | Should work if the installed Transformers/vLLM/verl versions support the exact checkpoint. Smoke test first. |
| Qwen text-only | `Qwen/Qwen3-0.6B` in Sokoban text LoRA example | Text-only tasks only, not Active Spatial vision tasks. |

For Active Spatial, the most used path is:

```bash
examples/train/active_spatial/run_experiment.sh
examples/train/active_spatial/experiments/v42_nodelta_w1.sh
examples/train/active_spatial/experiments/v42_nodelta_w3.sh
```

These default to Qwen2.5-VL style training. To switch to another standard supported Qwen-VL checkpoint, create a new experiment file and override:

```bash
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
TP_SIZE="4"
GPU_MEM_UTIL="0.35"
MAX_PROMPT_LENGTH="3072"
MAX_RESPONSE_LENGTH="384"
TRAIN_BATCH_SIZE="8"
VAL_BATCH_SIZE="4"
PPO_MINI_BATCH_SIZE="4"
```

Tune GPU memory, TP size, batch size, and prompt length conservatively first.

### Supported With Custom Adapter: Cambrian-S 7B

Cambrian-S is not vendored into this repo, but VAGEN-Lite already has adapter code for it:

```text
vagen/models/cambrian_register.py
vagen/models/cambrian_processor.py
vagen/models/cambrian_plugin.py
vagen/models/cambrian_vllm.py
```

Active Spatial Cambrian experiment scripts already exist:

```text
examples/train/active_spatial/experiments/c1_groupadv_100scenes.sh
examples/train/active_spatial/experiments/c2_nfp_groupadv.sh
examples/train/active_spatial/experiments/c3_lr5e7_cambrian.sh
examples/train/active_spatial/experiments/c4_fwdfirst.sh
examples/train/active_spatial/experiments/c5_entropy_hi.sh
examples/train/active_spatial/experiments/c6_no_think.sh
examples/train/active_spatial/experiments/c7_fwdfirst_ehi.sh
examples/train/active_spatial/experiments/c8_fwdfirst_rewscale.sh
```

The experiment history and diagnosis are in:

```text
docs/cambrian_iteration.md
docs/v26_v27_analysis.md
```

Those documents are mainly experiment-analysis notes. This file is the operational integration guide.

## 2. How Cambrian-S Is Integrated

Cambrian-S needs extra work because it is not a plain Qwen2.5-VL HuggingFace model:

1. The external Cambrian-S source tree must be importable.
2. Transformers `AutoConfig` / `AutoModelForCausalLM` must know `cambrian_qwen`.
3. FSDP actor training needs a GPU-compatible forward path for Cambrian multimodal embeddings.
4. The PPO critic needs a token-classification/value-head compatible wrapper.
5. vLLM rollout workers need a custom model and multimodal processor.
6. The agent loop must convert environment images into Cambrian SigLIP pixel values and expand image tokens correctly.

The current adapter split is:

| File | Purpose |
|---|---|
| `vagen/models/cambrian_register.py` | FSDP/Transformers registration, actor adapter, critic adapter. Imported through `external_lib`. |
| `vagen/models/cambrian_processor.py` | Minimal processor wrapper. Adds `<image>`, preprocesses SigLIP images, marks Cambrian branch. |
| `vagen/models/cambrian_plugin.py` | vLLM plugin entry point. Registers Cambrian model in vLLM subprocesses. |
| `vagen/models/cambrian_vllm.py` | vLLM model wrapper and multimodal prompt/image expansion. |
| `vagen/agent_loop/agent_loop_no_concat.py` | Detects `CambrianSiglipImageProcessor`, expands image tokens to `-200` blocks, attaches `pixel_values`. |
| `setup.py` | Registers `vllm.general_plugins` entry point for Cambrian. Requires `pip install -e .`. |

## 3. Cambrian-S Setup On A New Server

### Step 1: Prepare External Cambrian-S Project

Clone or copy the existing Cambrian-S project outside this repo, for example:

```bash
/nas/baiqiao/active_spatial/cambrian-s
```

The adapter defaults to:

```text
/scratch/by2593/project/Active_Spatial/cambrian-s
```

If your path is different, export:

```bash
export CAMBRIAN_SRC=/nas/baiqiao/active_spatial/cambrian-s
```

Do this before launching training.

### Step 2: Prepare Cambrian-S Checkpoint

The old scripts expect:

```text
/scratch/by2593/hf_cache/cambrian-s-7b
```

On a new server, either place/symlink the checkpoint there or edit the experiment script:

```bash
MODEL_PATH="/nas/baiqiao/hf_cache/cambrian-s-7b"
```

The checkpoint must include the Cambrian config/tokenizer/model files expected by the external Cambrian-S code.

### Step 3: Install This Repo As Editable

The vLLM Cambrian plugin is exposed through `setup.py` entry points. Install the repo in the environment used for training:

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
pip install -e .
```

This is important. Without the editable install, vLLM subprocesses may not load `vagen.models.cambrian_plugin`.

### Step 4: Use A Cambrian Experiment Script

The safest current Cambrian starting point is the fwd-first/reward-scaled branch, because earlier Cambrian experiments showed that the default "rotate first" prompt caused strong `turn_left` bias.

Recommended script to inspect first:

```text
examples/train/active_spatial/experiments/c8_fwdfirst_rewscale.sh
```

Core Cambrian-specific settings:

```bash
MODEL_PATH="/path/to/cambrian-s-7b"
TP_SIZE=2
GPU_MEM_UTIL=0.20

EXTRA_OVERRIDES="\
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.external_lib=vagen.models.cambrian_register \
    critic.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False"
```

`use_remove_padding=False` is required for the Cambrian/NFP image-token layout used here.

Launch pattern:

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
export CAMBRIAN_SRC=/nas/baiqiao/active_spatial/cambrian-s
nohup bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/c8_fwdfirst_rewscale.sh \
  > c8_fwdfirst_rewscale.log 2>&1 &
```

Before doing a full run, make a tiny smoke experiment by copying the script and reducing:

```bash
TOTAL_STEPS=1
SAVE_FREQ=1
TEST_FREQ=1
TRAIN_BATCH_SIZE=1
VAL_BATCH_SIZE=1
N_TRAJECTORY=1
PPO_MINI_BATCH_SIZE=1
```

The smoke run should verify:

- tokenizer loads
- Cambrian processor wrapper is used when `AutoProcessor` is absent
- vLLM rollout starts and accepts images
- FSDP actor forward receives `pixel_values`
- critic/value path initializes
- validation JSONL is written

## 4. Cambrian-S Experiment Lessons

See `docs/cambrian_iteration.md` for details. Short version:

- Cambrian-S can be launched in this framework, but it was not a drop-in win.
- Early c1/c3 runs had strong `turn_left` bias and nontrivial no-action-tag rate.
- Lower LR alone did not fix it.
- `fwd-first` prompt was the most effective intervention; c4 reached much better move-forward behavior and better validation.
- `no_think` was worse and unstable.
- Later Cambrian runs should start from fwd-first plus lower-variance reward settings, not from the original free-think prompt.

Recommended current Cambrian baseline:

```text
c8_fwdfirst_rewscale.sh
```

But if the Active Spatial reward/data has changed since those scripts were written, label the run clearly and audit the data first.

## 5. Adding A New Model Family

New models fall into three levels.

### Level 0: Same Family, Standard HuggingFace/vLLM Support

Examples: another Qwen2.5-VL checkpoint, a compatible Qwen3-VL checkpoint.

Usually enough:

1. Create a new experiment script under `examples/train/active_spatial/experiments/`.
2. Set `MODEL_PATH`.
3. Adjust `TP_SIZE`, `GPU_MEM_UTIL`, batch sizes, prompt/response lengths.
4. If needed, set:

```bash
EXTRA_OVERRIDES="\
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.trust_remote_code=True"
```

5. Smoke run for 1 step before long training.

No code changes should be needed if the processor returns Qwen2-VL style fields such as `pixel_values` and `image_grid_thw`.

### Level 1: HuggingFace Loads, But vLLM/verl Needs Help

This is the common case for newer or less standard VLMs.

You may need:

- `external_lib` module to register `AutoConfig`, `AutoModelForCausalLM`, and critic/value classes.
- vLLM `ModelRegistry.register_model(...)` plugin.
- A custom multimodal processor for vLLM rollout.
- A custom image processor wrapper if `AutoProcessor` is absent or incompatible.
- Agent-loop code to convert environment PIL images into the model's expected `multi_modal_inputs`.

Use Cambrian as the template:

```text
vagen/models/cambrian_register.py
vagen/models/cambrian_processor.py
vagen/models/cambrian_plugin.py
vagen/models/cambrian_vllm.py
```

Then add experiment overrides:

```bash
EXTRA_OVERRIDES="\
    actor_rollout_ref.model.external_lib=vagen.models.<new_model>_register \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.external_lib=vagen.models.<new_model>_register \
    critic.model.trust_remote_code=True"
```

### Level 2: Custom Multimodal Token Semantics

Cambrian-S is Level 2: the image token in the prompt is not enough; it must be expanded into hundreds of internal visual feature positions, and the FSDP actor must scatter image embeddings into text embeddings.

For a Level 2 model, implement or verify:

1. Prompt image-token convention for rollout.
2. Training image-token convention for FSDP actor.
3. Conversion between compact rollout tokens and expanded training tokens.
4. `position_ids` logic after expansion.
5. `response_mask` alignment after expansion.
6. `multi_modal_inputs` shape and dtype.
7. Critic/value forward path.
8. vLLM rollout path.
9. Checkpoint saving/loading in HuggingFace format.

This is doable, but not a one-line config change.

## 6. Minimal New-Model Checklist

Before investing in a long run, answer these questions:

1. Does `AutoTokenizer.from_pretrained(MODEL_PATH)` work?
2. Does `AutoProcessor.from_pretrained(MODEL_PATH)` work? If not, do we have a wrapper?
3. Does vLLM support this architecture? If not, do we have a vLLM plugin?
4. Can the FSDP actor compute logprobs with images?
5. Can the critic/value model initialize?
6. Are image tokens, `position_ids`, and `response_mask` aligned after multi-turn concatenation?
7. Does rollout generation accept the same image format as training logprob?
8. Does `trainer.replace_image_tokens_for_logging` produce readable logs?
9. Can a 1-step smoke run save validation JSONL?
10. Does the model obey the required `<action>...</action>` output format before RL?

The last point is important: Cambrian showed that a stronger visual backbone can still fail if the action-format prior is weak.

## 7. Practical Recommendation

For the next round of Active Spatial training:

1. Use Qwen2.5-VL-3B or 7B for the cleanest baseline.
2. Use Cambrian-S only after confirming external source/checkpoint/plugin setup on the server.
3. For Cambrian-S, start from `c8_fwdfirst_rewscale.sh` style settings rather than c1/c3.
4. For any new model, first do frozen rollout/action-format audit, then 1-step PPO smoke, then short 50-step diagnostic, then full run.

The framework can support new models, but convenience depends on how standard the model is. Qwen-family models are easy. Cambrian-style custom VLMs require adapter work across processor, rollout, FSDP actor, critic, and agent-loop token handling.
