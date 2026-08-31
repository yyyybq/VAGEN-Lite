# D0 Cambrian-S / LFP Plumbing Audit

Date: 2026-08-09

Scope: Cambrian-S C-series focus, especially C8, and the latest B5 attempt under
`examples/train/active_spatial`. This audit is D0 only. No D1 experiment and no new
RL training run were started.

Gate decision after D0.1: **FAIL**

Primary reason: D0.1 static forward-parity tracing found a concrete implementation
mismatch in the action-policy trunk. The FSDP training adapter builds multimodal
embeddings with the Cambrian 64-token MIV overwrite, while the vLLM rollout/eval
wrapper builds only projected 27x27 SI features plus newline tokens. This means
the synced vLLM behavior policy is not guaranteed to be the same policy as the FSDP
actor used for logprob/training, even before considering the auxiliary `nfp_head`.

The requested GPU logprob parity probe and one-mini-batch gradient/update audit were
attempted but could not be run from this shell: local `nvidia-smi` cannot communicate
with the NVIDIA driver, `srun/sinfo` are unavailable, and read-only SSH probes to
known GPU nodes failed with `No route to host`. No D1 experiment and no full RL
training run were started.

## A. What Was Tested

This audit checked the Cambrian-S C-series plumbing rather than running another
hyperparameter search.

Tested questions:

1. Which model was actually loaded by C8 and B5.
2. Whether C8/B5 used plain Cambrian-S or Cambrian-S-7B-LFP.
3. Whether the LFP/NFP head exists in the checkpoint.
4. Whether the LFP/NFP head is called in actor training, rollout/server generation,
   and evaluation generation.
5. Whether training, rollout, and eval use consistent Cambrian architecture paths.
6. Whether action-token learning signal exists: response/action tokens in the PPO
   loss, nonzero advantages, policy ratio/KL/clip/gradient behavior.
7. Whether `N_TRAJECTORY=1` degenerates the advantage estimator.
8. Whether existing C8/B5 logs show NFP loss activity, gradients, and nonzero PPO
   learning signal.

Files inspected:

- `examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_lfp_server.sh`
- `examples/train/active_spatial/experiments/b5_c8_wrapper_img25_actionvalid.sh`
- `examples/train/active_spatial/run_experiment.sh`
- `exps/vagen_active_spatial/b5_c8_wrapper_img25_actionvalid/hydra_run/.hydra/overrides.yaml`
- `exps/vagen_active_spatial/c8_fwdfirst_rewscale_lfp_server_v19/train.log`
- `exps/vagen_active_spatial/b5_c8_wrapper_img25_actionvalid/train.log`
- `vagen/models/cambrian_register.py`
- `vagen/models/cambrian_vllm.py`
- `vagen/models/cambrian_processor.py`
- `vagen/models/cambrian_plugin.py`
- `vagen/agent_loop/agent_loop_no_concat.py`
- `vagen/custom_advantage/no_concat_gae.py`
- `evaluation/model_agent.py`
- `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP/config.json`
- `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP/model.safetensors`
- `/mnt/umm/users/yinbaiqiao/hf_cache/cambrian-s-7b/config.json`

## B. What Was Changed

Only this diagnosis document was added:

- `docs/diagnosis/D0_CAMBRIAN_PLUMBING_AUDIT.md`

No training script, model code, rollout code, evaluation code, or experiment config
was modified. No new RL training job was launched.

## C. Sanity Checks

### C8 and B5 model identity

C8 script:

- Experiment: `c8_fwdfirst_rewscale_lfp_server_v19`
- Model path: `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP`
- Env config: `env_config_v24_100scenes_fwdfirst_rewscale_server.yaml`
- `N_TRAJECTORY=1`
- actor LR `5e-7`, critic LR `2e-6`
- critic warmup `60`
- `actor_rollout_ref.actor.nfp_loss_coef=1.0`
- `vagen.models.cambrian_register` loaded as external library
- `actor_rollout_ref.model.trust_remote_code=True`
- final `actor_rollout_ref.model.use_remove_padding=False`

B5 script:

- Experiment: `b5_c8_wrapper_img25_actionvalid`
- Model path: `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP`
- Env config: `env_config_b5_c8_actionvalid.yaml`
- remote render host: `10.119.30.223:8767`
- `N_TRAJECTORY=1`
- `actor_rollout_ref.actor.nfp_loss_coef=1.0`
- `actor_rollout_ref.rollout.limit_images=25`
- `actor_rollout_ref.rollout.max_model_len=16384`
- `RESUME_MODE=auto`

B5 Hydra overrides confirm:

- `actor_rollout_ref.model.path=/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP`
- `algorithm.adv_estimator=no_concat_gae`
- `actor_rollout_ref.rollout.n=1`
- `actor_rollout_ref.rollout.limit_images=25`
- `actor_rollout_ref.rollout.max_model_len=16384`
- `actor_rollout_ref.model.use_remove_padding=False`
- `actor_rollout_ref.model.external_lib=vagen.models.cambrian_register`

Conclusion: C8 and B5 were configured to use **Cambrian-S-7B-LFP**, not the plain
local `cambrian-s-7b` directory.

### Checkpoint config

`Cambrian-S-7B-LFP/config.json`:

- `model_type=cambrian_qwen`
- `architectures=["CambrianQwenForCausalLM"]`
- `_name_or_path=Qwen/Qwen2.5-7B-Instruct`
- `mm_projector_type=mlp2x_gelu`
- `connector_only=True`
- `nfp_head=True`
- `miv_token_len=64`
- `si_token_len=729`
- `mm_use_im_newline_token=True`
- `nfp_mse_loss_weight=0.1`
- `nfp_cosine_loss_weight=0.1`
- hidden size `3584`
- layers `28`
- vocab size `152064`

Plain `/mnt/umm/users/yinbaiqiao/hf_cache/cambrian-s-7b/config.json` has the same
base Cambrian-Qwen family, but `nfp_head` is absent/none. The LFP run path is
therefore not the plain model.

### Parameter inventory from safetensors

Measured from `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP/model.safetensors`:

| Group | Params | Evidence |
|---|---:|---|
| Total | 8,047,327,392 | all tensors |
| Vision tower | 397,747,744 | `model.vision_tower_aux_list.0.vision_tower.*` |
| Connector/projector | 16,984,576 | `model.mm_projector.*`, `model.image_newline` |
| LFP/NFP head | 16,978,560 | `model.nfp_head.*` |
| LLM embed/norm | 545,000,960 | embeddings and final norm |
| LLM early layers 0-8 | 2,097,520,128 | `model.layers.0` through `8` |
| LLM middle layers 9-18 | 2,330,577,920 | `model.layers.9` through `18` |
| LLM late layers 19-27 | 2,097,520,128 | `model.layers.19` through `27` |
| LM head | 544,997,376 | `lm_head.weight` |

Representative LFP/NFP tensors:

- `model.nfp_head.0.weight`: `[3584, 3584]`
- `model.nfp_head.2.weight`: `[1152, 3584]`

### Fresh mini-batch limitation

The D0 plan requested a real mini-batch forward/backward plus gradient norm,
optimizer membership, and update norm by group. This local session cannot execute
that check because `nvidia-smi` reports that it cannot communicate with the NVIDIA
driver. A CPU run is not a realistic substitute for this 7B multimodal model.

Therefore these items are not fully proven by a fresh D0 run:

- per-group `requires_grad` from a live instantiated FSDP actor
- per-group optimizer membership from the live trainer optimizer
- per-group gradient norm from one new mini-batch
- per-group update norm after one optimizer step

Existing code and logs still provide partial evidence:

- `vagen/models/cambrian_register.py` explicitly sets the loaded vision tower to
  `requires_grad_(True)`.
- No freeze override was found in C8/B5 configs.
- C8 and B5 logs show nonzero actor gradient norms after critic warmup.
- C8 and B5 logs show active NFP auxiliary loss during actor training.

## D. Results

### Training forward path

Actor/ref/critic registration comes from `vagen.models.cambrian_register`.

For actor training:

- model class: `CambrianForCausalLMAdapter`
- base Cambrian class: `CambrianQwenForCausalLM`
- compact `<image>` tokens are expanded in `vagen/agent_loop/agent_loop_no_concat.py`
  to Cambrian image blocks of 756 tokens.
- image preprocessing uses `CambrianProcessorWrapper`.
- pixel values are shaped as `[1, 3, 384, 384]` in sanity manifests.
- `_embed_multimodal_batch` calls `encode_images`, `mm_projector`, and inserts image
  newline tokens.
- if `config.nfp_head` and `miv_token_len > 0`, the first 64 positions of each image
  block are overwritten with MIV features.
- if `nfp_pixel_values`, `nfp_loss_mask`, `config.nfp_head`, and `self.training` are
  present, `_compute_nfp_loss` runs `self.model.nfp_head(hidden_states)`.
- the model returns `nfp_aux_loss`, and the actor loss consumes it through
  `actor_rollout_ref.actor.nfp_loss_coef=1.0`.

Direct log evidence:

- C8 contains `[cambrian_register] nfp loss active hidden=(1,3187,3584) target=(1,3187,1152) mask_tokens=64.0`.
- B5 contains `[cambrian_register] nfp loss active hidden=(1,3187,3584) target=(1,3187,1152)`.
- Terminal/invalid next-frame cases can have `mask_tokens=0.0`; nonterminal cases show
  active 64-token masks.

Conclusion: the LFP/NFP head is active in the FSDP actor training path when valid
next-frame targets exist.

### Rollout/server forward path

Rollout uses vLLM:

- registered by `vagen/models/cambrian_plugin.py`
- model class: `CambrianVLLMForCausalLM`
- implementation file: `vagen/models/cambrian_vllm.py`
- loads vision tower, mm projector, image newline, language model, and LM head
- explicitly skips `model.nfp_head.*` weights
- does not call the NFP head during generation
- does not implement the same 64-token MIV overwrite found in the FSDP training
  adapter

Conclusion: the rollout server generates actions with Cambrian multimodal features,
but not with the LFP/NFP head and not with the same MIV-injected image embedding path
used by actor training.

### Evaluation forward path

`evaluation/model_agent.py` detects Cambrian checkpoints and uses the same vLLM
Cambrian implementation:

- imports `vagen.models.cambrian_plugin`
- default Cambrian eval `max_model_len=16384`
- default Cambrian eval `limit_images=25`
- generation sends prompt plus `multi_modal_data`
- no NFP fields are passed
- no NFP head is called

Conclusion: evaluation matches rollout more closely than training, because both use
vLLM and omit the NFP/MIV training path.

### Processor and model-path consistency

Observed consistency:

- C8/B5 train path, rollout path, and eval path all target the same
  `Cambrian-S-7B-LFP` checkpoint.
- `CambrianProcessorWrapper` is used in B5 agent-loop logs.
- vision sanity records show one image, 756 expanded image tokens, and finite
  `[1, 3, 384, 384]` pixel values.

Observed risks:

- training processor wrapper defaults to `google/siglip2-so400m-patch14-384`.
- vLLM Cambrian path uses `google/siglip-so400m-patch14-384`.
- image size and normalization appear compatible, but the processor family and
  resampling behavior differ.
- B5 config requested `actor_rollout_ref.rollout.max_model_len=16384`, but the B5
  train log's vLLM engine line reports `max_seq_len=2432`. This may be a logging or
  propagation issue, but it is not clean.

### Action learning signal

The C8 and B5 training logs show nonzero PPO learning signal after critic warmup.

Selected C8 metrics:

| Step | Success | Invalid action | NFP loss | PPO KL | PG loss | Grad norm | Advantage range |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 0.000 | 0.052 | 0.007676 | 0.000629 | -0.06578 | 8.9617 | -1.625 to 3.453 |
| 150 | 0.125 | 0.011 | 0.003459 | present | present | 5.0938 | -0.711 to 4.906 |
| 200 | 0.375 | 0.031 | 0.002750 | present | present | 7.5418 | -0.668 to 3.875 |
| 300 | 0.375 | 0.069 | 0.001900 | present | present | 7.1429 | -0.809 to 3.016 |
| 304 | 0.500 | 0.148 | 0.001822 | present | present | 7.9971 | -0.844 to 3.094 |

Selected B5 metrics:

| Step | Success | Invalid action | NFP loss | PPO KL | PG loss | Grad norm | Advantage range |
|---:|---:|---:|---:|---:|---:|---:|---|
| 101 | 0.125 | 0.044 | 0.008712 | 0.00633 | 0.05758 | 10.018 | -1.164 to 3.844 |
| 150 | 0.125 | 0.089 | 0.004107 | present | present | 4.9529 | -0.879 to 7.688 |
| 200 | 0.500 | 0.000 | 0.002898 | present | present | 7.5590 | -0.801 to 2.953 |
| 250 | 0.125 | 0.051 | 0.003104 | present | present | 5.7679 | -1.219 to 6.188 |
| 278 | 0.125 | 0.037 | 0.001853 | present | present | 6.0286 | -2.047 to 5.938 |

Conclusion: action-token policy learning is not dead. PPO updates, KL, policy loss,
gradient norm, and nonzero advantages are present.

### N=1 advantage behavior

C8/B5 use:

- `N_TRAJECTORY=1`
- run script maps `ADV_ESTIMATOR="masked_gae"` to `algorithm.adv_estimator=no_concat_gae`
- `no_concat_gae` is implemented in `vagen/custom_advantage/no_concat_gae.py`

`no_concat_gae` behavior:

- uses token-level scores, critic values, response masks, `group_idx`, `traj_idx`,
  and `turn_idx`
- sums token rewards into turn rewards
- anchors each turn on the first valid response-token value
- computes GAE over turns within each `(group_idx, traj_idx)` sequence
- broadcasts the resulting advantage to valid response tokens
- whitens advantages with `masked_whiten`

Conclusion: `N_TRAJECTORY=1` does not degenerate the estimator to a zero-advantage
GRPO-style comparison, because this path is critic/GAE based rather than
same-prompt multi-sample normalization. However, `N=1` still reduces exploration and
removes multi-sample contrast for a given prompt.

### B5 run status

B5 resumed from:

- `exps/vagen_active_spatial/b5_c8_wrapper_img25_actionvalid/checkpoints/global_step_100`

B5 later stopped because remote rendering failed:

- `RuntimeError: ActiveSpatial rendering failed: All connection attempts failed`

So B5 did not complete a clean 1000-step run. Existing checkpoints include at least
steps 50, 100, and 250.

Known B5 external navigation signal at step 100:

| Split | Success | Timeout | Mean action validity |
|---|---:|---:|---:|
| ID400 | 0.110 | 0.3175 | 0.852 |
| OOD scene | 0.1203 | not listed here | 0.887 |
| OOD instance | 0.185 | not listed here | 0.902 |
| OOD geometry | 0.1493 | not listed here | 0.924 |
| OOD template | 0.1465 | not listed here | 0.900 |
| OOD v2 centering | 0.240 | not listed here | 0.892 |

Existing summary notes place B5 EASI macro around 44.6, close to Cambrian baseline
44.4 and below C8 at step 300 around 45.4.

## E. Interpretation

### Findings with strong evidence

1. C8 and B5 are not plain Cambrian-S runs. They load
   `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP`, whose config has
   `nfp_head=True`.

2. The LFP/NFP head exists in the checkpoint. The safetensors file contains
   `model.nfp_head.*` parameters totaling 16,978,560 parameters.

3. The FSDP actor training path calls the NFP head when next-frame targets and
   masks are available. C8/B5 logs directly show active NFP loss.

4. The rollout and evaluation vLLM path does not load or call the NFP head.
   `vagen/models/cambrian_vllm.py` explicitly skips `model.nfp_head.*`.

5. The action-token PPO signal is not dead. Existing C8/B5 logs show nonzero
   advantages, policy loss, KL, clip fraction, NFP loss, and actor gradient norm.

6. `N_TRAJECTORY=1` does not mathematically collapse the selected advantage
   estimator, because C8/B5 use `no_concat_gae`, not a same-prompt GRPO estimator.

### Suspected plumbing problem

The largest issue found by D0 is not simply "NFP head unused." It is more specific:

- training actor forward uses LFP/MIV image-token modification when `nfp_head=True`
- rollout/eval generation uses a separate vLLM Cambrian implementation
- vLLM skips NFP weights and appears not to reproduce the training adapter's
  64-token MIV overwrite

That means rollout/eval can generate actions under a different visual-token
representation than the representation used by the actor training/logprob path.
This is a serious consistency risk for policy optimization.

This can explain weak or noisy C8/B5 gains even when PPO metrics look alive:

- rollout samples actions from one visual embedding path
- training computes logprobs and gradients under a different visual embedding path
- the auxiliary NFP loss can update representations used by training but not used in
  the same way by the rollout server

This is not yet proven as the only cause of the C-series/B5 behavior, but it is
strong enough that D1 should not start before resolving or explicitly ablating it.

### Additional concerns

1. Processor mismatch:
   training uses a SigLIP2 processor wrapper, while vLLM uses a SigLIP processor.
   Shape and normalization are compatible, but exact preprocessing may differ.

2. B5 context-length mismatch:
   B5 config requests `max_model_len=16384`, while the visible vLLM engine log reports
   `max_seq_len=2432`. This needs a direct server-side config print or one-shot probe.

3. The per-group trainability/update audit is incomplete:
   existing logs show global actor gradient norms, but not group-wise gradient and
   update norms for vision tower, connector, LLM layers, LM head, and LFP head.

## F. D0.1 Train / Rollout Forward Parity

D0.1 target: determine whether the same Cambrian-S-7B-LFP checkpoint, same
observation, same prompt, and deterministic generation settings implement the same
action policy in:

- FSDP/HF training adapter: `CambrianForCausalLMAdapter`
- rollout/eval wrapper: `CambrianVLLMForCausalLM`

The answer from static tracing is **no**. The policy trunk is not identical.

This is not primarily about whether `nfp_head` is called during inference. It is
valid for `nfp_head` to be an auxiliary-only training head. The parity failure is
that `config.nfp_head=True` also changes the image-token embeddings used by the
FSDP actor through MIV injection, and the vLLM behavior-policy wrapper does not
implement the same image-token construction.

### Static code parity table

| Item | FSDP training adapter | vLLM rollout/eval wrapper | Status |
|---|---|---|---|
| Checkpoint | C8/B5 use `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP` | rollout/eval point at same model path | IDENTICAL |
| Tokenizer during RL rollout | `CambrianProcessorWrapper` adds compact `<image>` token id 151665 before sending prompt ids | vLLM receives `TokensPrompt(prompt_token_ids=...)`; target replacement also expects id 151665 | IDENTICAL for RL rollout |
| Eval tokenizer path | evaluation uses plain `AutoTokenizer.from_pretrained` and prompt string | vLLM processor expects image token id 151665 | NOT VERIFIED; likely risk because plain tokenizer encodes `<image>` as `[27, 1805, 29]` unless token is added |
| Chat template during RL rollout | `processor.apply_chat_template(..., add_generation_prompt=True)` | rollout receives the resulting token ids directly | IDENTICAL for the prompt ids sent during RL |
| Generation prefix during RL rollout | assistant generation prompt comes from same chat template | vLLM continues from the provided prompt ids | IDENTICAL for RL rollout |
| Image ordering | `sys_images + cur_images` is stored in `multi_modal_data={"image": turn_images}` | same list is passed as `image_data` to vLLM server | IDENTICAL by code path |
| Image token placement | compact `<image>` later expanded to 756 `IMAGE_TOKEN_INDEX=-200` sentinels in training postprocess | compact id 151665 is replaced by 756 `<|image_pad|>` id 151655 positions | INTENDED IDENTICAL |
| Multimodal replacement location | replacement occurs at every expanded `-200` position | `merge_multimodal_embeddings` replaces every image pad token | INTENDED IDENTICAL |
| Processor model name | `google/siglip2-so400m-patch14-384` in `CambrianProcessorWrapper` | `google/siglip-so400m-patch14-384` in `cambrian_vllm.py` | DIFFERENT |
| Padding color before resize | `(127,127,127)` from mean 0.5 in training helper | `(122,116,104)` in vLLM helper | DIFFERENT for non-square images; Active Spatial sanity images are square `[256,256]`, so usually not active |
| Resize/crop implementation | manual square pad, manual BICUBIC resize to 384, then HF image processor | square pad, then `SiglipImageProcessor` does its own resize/normalize | DIFFERENT / NOT FULLY QUANTIFIED |
| Pixel tensor shape | sanity logs show `[1,3,384,384]` float32 | vLLM helper returns `[N,3,384,384]` float32 | IDENTICAL shape |
| Vision tower weights | checkpoint `model.vision_tower_aux_list.0.vision_tower.*` loaded by Cambrian source model | remapped to `vision_tower.*` | INTENDED IDENTICAL |
| Vision feature layer | FSDP calls Cambrian `encode_images([pixel_values])`; observed/commented shape `(N,729,1152)` | vLLM uses `SiglipVisionModel(...).last_hidden_state`, shape `(N,729,1152)` | NOT VERIFIED exact layer equivalence |
| Multimodal projector | checkpoint `model.mm_projector.*` | remapped to `mm_projector.*` | INTENDED IDENTICAL |
| Image newline | checkpoint `model.image_newline` inserted after each 27-token row | remapped to `image_newline` and inserted after each row | IDENTICAL before MIV overwrite |
| 64-token MIV overwrite | if `config.nfp_head` and `miv_token_len>0`, overwrite first 64 image positions with 8x8 interpolated projected features | not implemented; first 64 positions remain ordinary flattened SI/newline sequence | **DIFFERENT** |
| `nfp_head` inference call | not called in `eval()` because auxiliary loss requires `self.training` | weights skipped and not called | Acceptable difference; not the policy-parity issue |
| Legal action logits/logprobs | requires GPU run | requires GPU run | NOT VERIFIED numerically |
| `exp(logp_train-logp_rollout)` at synced weights | requires GPU run | requires GPU run | NOT VERIFIED numerically |

### Confirmed forward mismatch

The FSDP policy-forward embedding path in `CambrianForCausalLMAdapter` is:

1. encode image to 729 SigLIP features
2. apply `mm_projector`
3. append image-newline tokens, producing 756 image tokens
4. embed text tokens
5. scatter visual features into `-200` image positions
6. if `nfp_head=True`, overwrite the first 64 positions of each image block with
   bilinearly interpolated 8x8 MIV features

The vLLM policy-forward embedding path in `CambrianVLLMForCausalLM` is:

1. preprocess image
2. run `SiglipVisionModel(...).last_hidden_state`
3. apply `mm_projector`
4. append image-newline tokens, producing 756 image tokens
5. merge those features into `<|image_pad|>` positions

The vLLM path stops before step 6. It never overwrites the first 64 image tokens
with MIV features.

This necessarily creates a different `inputs_embeds` tensor for image-containing
prompts when `nfp_head=True` and `miv_token_len=64`, except in the accidental
measure-zero case where the 8x8 interpolated features exactly equal the first 64
tokens of the flattened 27x28 SI/newline sequence. A local CPU toy check of the two
tensor transformations produced nonzero L2 difference for the first 64 tokens, as
expected.

Therefore D0.1 finds a real forward implementation mismatch before any optimizer
step or PPO update.

### GPU parity probe status

Requested probe:

- fixed 4-16 real Active Spatial observations
- same checkpoint
- same prompt/image/history
- deterministic settings
- compare FSDP/HF `eval()+no_grad` against vLLM
- record processed image stats, token positions, visual features, final legal-action
  logits, action-token logprobs, top-1 agreement, ranking correlation, and
  `exp(logp_train-logp_rollout)`

Status: **not executable from this shell**.

Attempts:

- local `nvidia-smi` failed: cannot communicate with NVIDIA driver
- `srun` and `sinfo` are not available
- read-only SSH `nvidia-smi` probes to `10.119.21.155`, `10.119.21.185`, and
  `10.119.21.237` failed with `No route to host`

No GPU process, no vLLM server, and no RL job was started.

### Mini-batch gradient/update audit status

Still not executable from this shell for the same GPU-access reason.

The missing numerical evidence remains:

- per-group live `requires_grad` count from the instantiated actor
- optimizer membership by group
- one-mini-batch grad norm by group
- one optimizer-step update norm by group

Groups still required:

- vision tower
- mm projector
- MIV / related multimodal modules
- early LLM
- middle LLM
- late LLM
- LM head
- `nfp_head`

### D0.1 answer to the key question

Question:

> Did previous C8/B5 possibly use a vLLM behavior policy for rollout that was
> inconsistent with the training actor?

Answer:

**Yes. Based on static execution-path evidence, this is not just possible; the
image-containing forward path is implemented differently.**

C8/B5 vLLM rollout likely sampled actions from a behavior policy whose visual
embeddings did not include the FSDP actor's 64-token MIV overwrite. The FSDP actor
used for training/logprob then evaluated those trajectories under a different
multimodal embedding construction. The expected synced-weight PPO ratio
`exp(logp_train - logp_rollout)` is therefore not guaranteed to concentrate near 1.
The exact magnitude still requires the GPU parity probe, but the forward mismatch
itself is already established.

## G. Gate Decision

Decision: **FAIL**

D1 should not start yet.

Reason:

- identity is resolved
- NFP/LFP training usage is proven
- rollout/eval non-usage is proven
- action learning signal is present
- `N=1` advantage collapse is ruled out for `no_concat_gae`
- D0.1 found a concrete train/rollout forward mismatch in the policy trunk:
  FSDP actor applies 64-token MIV overwrite, vLLM behavior policy does not
- the required live mini-batch group-wise gradient/update audit still could not be
  run in this environment because no reachable GPU is available

Minimum follow-up required before any D1 experiment:

1. On a GPU node, run a one-mini-batch plumbing probe that instantiates the actual
   C8/B5 actor path and records `requires_grad`, optimizer membership, gradient norm,
   and one-step update norm for:
   vision tower, connector/projector, early LLM, middle LLM, late LLM, LM head, and
   LFP/NFP head.

2. Fix or ablate the parity issue with the smallest possible change:
   first align processor/preprocessing if needed, then align or disable MIV overwrite,
   then verify multimodal token handling.

3. On the same prompt/image, rerun parity and compare the training adapter's image
   embeddings/logits against the vLLM rollout path. Specifically verify whether the
   first 64 image tokens and legal-action logits now match closely enough.

4. Print the effective vLLM `max_model_len` from the live B5 server process and
   resolve why the log shows `max_seq_len=2432` despite the Hydra override
   `max_model_len=16384`.

5. Either make rollout/eval reproduce the FSDP training adapter's MIV path, or run a
   deliberate ablation that disables training-side MIV injection/NFP auxiliary loss so
   rollout and training are matched.

Until those checks are complete, the safest interpretation is:

> C8/B5 did load Cambrian-S-7B-LFP and did train with active NFP auxiliary loss, but
> the policy rollout/evaluation path did not implement the same image-token embedding
> construction as the FSDP training actor. This is a concrete plumbing failure in
> train/rollout forward parity and must be fixed or ablated before D1.

## H. D0.2 Repair And Verification

D0.2 used the GPU node `root@10.119.24.168` only for plumbing probes. No D1 run and
no full RL training run were started.

### Root cause repaired

The concrete policy mismatch found in D0.1 was real:

- FSDP `CambrianForCausalLMAdapter` applied the 64-token MIV overwrite before
  scattering visual embeddings into the LLM input.
- vLLM `CambrianVLLMForCausalLM` appended newline-expanded SI visual tokens but did
  not apply the same MIV overwrite.

D0.2 also found a second risky mismatch in the vLLM wrapper: its hand-built SigLIP
vision config instantiated 27 layers plus a head, while the
`Cambrian-S-7B-LFP` checkpoint contains 26 vision layers and no SigLIP head. The
old loader returned all parameter names to vLLM even when local vision/projector
parameters were unmatched or missing, which could hide randomly initialized rollout
parameters.

### Code changes

Changed files:

- `vagen/models/cambrian_miv.py`: new shared Cambrian visual-token helper for
  27x27 SI projection, newline append, 8x8 MIV construction, and first-64 overwrite.
- `vagen/models/cambrian_register.py`: FSDP adapter now uses the shared helper for
  visual embedding construction.
- `vagen/models/cambrian_vllm.py`: rollout wrapper now uses the same helper, aligns
  preprocessing with `CambrianProcessorWrapper`, builds the 26-layer/no-head SigLIP
  vision config, and fails fast on unmatched/missing local vision/projector/newline
  weights.
- `vagen/models/cambrian_processor.py`: SigLIP processor load now uses the shared
  local cache/offline behavior.
- `tools/d0_2_cambrian_parity_probe.py`: added processor, MIV, HF logprob, vLLM
  logprob, compare, and gradient-audit probes.
- `tools/d0_2_vllm_runtime_probe.py`: added live vLLM runtime config probe.

### Processor parity

Output: `docs/diagnosis/d0_2_processor_parity.json`

Eight real Active Spatial images from B5 `vision_sanity` were tested. These images
are square 256x256, so the previous and repaired paths happened to produce identical
pixels even though the processor model names differed.

| Metric | Result |
|---|---:|
| image count | 8 |
| training pixel shape | `[8,3,384,384]` |
| repaired vLLM pixel shape | `[8,3,384,384]` |
| after-vs-training max abs | 0.0 |
| after-vs-training mean abs | 0.0 |
| after-vs-training cosine | 0.9999665 |
| old-vs-training max abs on these square samples | 0.0 |

Interpretation: preprocessing is now code-aligned. The earlier
SigLIP-vs-SigLIP2/resample mismatch did not numerically affect this square-image
probe, but it remained a latent risk for non-square inputs and has been removed.

### MIV parity

Outputs:

- `docs/diagnosis/d0_2_miv_synthetic.json`
- `docs/diagnosis/d0_2_miv_real.json`

The old/no-MIV path is materially different from the MIV path:

| Probe | First-64 max abs | First-64 mean abs | First-64 cosine | Full visual mean abs |
|---|---:|---:|---:|---:|
| synthetic projected features, no-MIV vs MIV | 5.9481 | 0.9562 | -0.0006 | 0.0809 |
| real vision/projector output, no-MIV vs MIV | 17.3438 | 1.0186 | 0.4292 | 0.0862 |

The shared helper self-check is exact:

| Probe | First-64 max abs | Full visual max abs |
|---|---:|---:|
| synthetic helper vs helper | 0.0 | 0.0 |
| real helper vs helper | 0.0 | 0.0 |

Real vision/projector tensor stats:

- pixels: `[8,3,384,384]`
- raw vision: `[8,729,1152]`, bf16
- projected: `[8,729,3584]`, bf16
- MIV: `[8,64,3584]`

Interpretation: the D0.1 MIV mismatch was large enough to change the policy input
embeddings. The repaired vLLM and FSDP code paths now share the same implementation
before the LLM sees visual tokens.

### End-to-end action logprob parity

Outputs:

- `docs/diagnosis/d0_2_hf_logprob.json`
- `docs/diagnosis/d0_2_vllm_logprob.json`
- `docs/diagnosis/d0_2_logprob_compare.json`

Four fixed B5 rollout samples from `rollout_data/100.jsonl` were replayed with the
same checkpoint, same restored image placeholder, same prompt text, same image, and
deterministic settings. The rollout JSON text had image tokens removed for logging;
the probe restored one `<image>` at the observation location. Compact image token
position was 470 in all four samples.

Action-level parity after repair:

| Metric | Result |
|---|---:|
| legal-action top-1 agreement | 1.0 |
| legal-action ranking correlation | mean 1.0, std 0.0 |
| first-action ratio mean | 1.00314 |
| first-action ratio median | 1.00068 |
| first-action ratio std | 0.00348 |
| first-action ratio min/max | 1.00067 / 1.00806 |
| first-action logprob delta mean | 0.00313 |
| first-action logprob delta min/max | 0.00067 / 0.00803 |

The full natural-language response logprob ratio was not concentrated near 1:

| Metric | Result |
|---|---:|
| full-response ratio mean | 0.50646 |
| full-response ratio median | 0.38966 |
| full-response ratio min/max | 0.02963 / 1.21689 |

Interpretation: for the action policy token actually used by Active Spatial, the
synced train/rollout policy is now very close. The full-response ratio is less useful
as a gate because small HF-vs-vLLM numerical differences accumulate over 70-137
natural-language tokens.

### Live vLLM runtime config

Outputs:

- `docs/diagnosis/d0_2_vllm_runtime_config.json`
- `docs/diagnosis/d0_2_vllm_runtime_config_tp2_16384.json`

Live B5-shaped probe:

| Field | Value |
|---|---|
| model | `/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP` |
| dtype | `torch.bfloat16` |
| tensor parallel | 2 |
| requested max model len | 16384 |
| effective max model len | 16384 |
| limit images argument | `{'image': 25}` in vLLM startup log |

The live D0.2 probe showed `max_seq_len=16384` when the B5 override is passed.
Therefore the historical B5 `max_seq_len=2432` line was not reproduced here; it was
likely from an earlier/different launch context or a previous config propagation
problem, but the exact historical source was not traced.

### Gradient/update audit

Output: `docs/diagnosis/d0_2_grad_audit_single_gpu.json`

This was a single-GPU real `CambrianForCausalLMAdapter` backward audit on one B5
sample. It used the real checkpoint, real prompt/image, expanded 756-image-token HF
input, a policy-gradient-like response-token loss, an NFP auxiliary loss with a
64-token MIV mask, and one manual SGD-style update with `lr=1e-7`.

Important limitation: this is **not** a full Ray/FSDP AdamW shard optimizer audit.
It verifies gradient reachability and updateability in the real adapter path, but it
does not prove live FSDP optimizer membership/resync.

Loss and PPO-like diagnostics:

| Metric | Value |
|---|---:|
| policy PG-like loss | 1.00474 |
| NFP aux loss | 0.17383 |
| total audited loss | 1.17857 |
| KL | 0.0 |
| advantage mean/std | 1.0 / 0.0 |
| policy ratio min/mean/max | 1.0 / 1.0 / 1.0 |
| clip fraction | 0.0 |
| response mask tokens | 105 |

Per-group gradient/update stats:

| Group | Requires-grad params | Params with grad | Grad norm | Manual update norm |
|---|---:|---:|---:|---:|
| vision tower | 397,747,744 | 397,745,440 | 78.6864 | 7.8686e-06 |
| mm projector | 16,980,992 | 16,980,992 | 1.4664 | 1.4664e-07 |
| MIV/image newline | 3,584 | 3,584 | 0.0658 | 6.5752e-09 |
| LLM early | 2,097,520,128 | 2,097,520,128 | 5.9627 | 5.9627e-07 |
| LLM middle | 2,330,577,920 | 2,330,577,920 | 9.2390 | 9.2390e-07 |
| LLM late | 2,097,520,128 | 2,097,520,128 | 8.5712 | 8.5712e-07 |
| LM head | 544,997,376 | 544,997,376 | 12.8852 | 1.2885e-06 |
| nfp_head | 16,978,560 | 16,978,560 | 0.4704 | 4.7037e-08 |

Interpretation: gradients are reachable through the vision tower, projector,
image-newline/MIV-related parameter, all LLM depth groups, LM head, and `nfp_head`.
The `nfp_head` is trainable under an NFP loss. The remaining missing evidence is the
actual Ray/FSDP AdamW optimizer/resync path.

### Minimal RL health check

Not run.

Reason: D0.2 action parity and single-GPU gradient reachability passed, but the
requested gradient audit was not the exact C8/B5 Ray/FSDP AdamW topology. Starting a
rollout/update/resync smoke before that exact optimizer path is verified would blur
the boundary between plumbing audit and a new training run.

## I. D0.2 Gate Decision

Decision: **INCONCLUSIVE**

The original D0.1 FAIL root cause has been repaired in code, and the most important
action-policy evidence is now positive:

- processor path is code-aligned
- MIV construction is shared by FSDP and vLLM
- vLLM local weight loading now fails fast on vision/projector/newline mismatch
- first-action train/rollout synced-policy ratio is tightly concentrated around 1
- legal-action top-1 and ranking match on the fixed probe
- single-GPU adapter gradient reachability is healthy across all required module
  groups, including `nfp_head`

However, D0.2 should not be marked PASS yet because two requested pieces of evidence
remain absent:

1. a live C8/B5 Ray/FSDP AdamW one-mini-batch audit with true optimizer membership,
   sharded update norms, and weight-sync/resync behavior;
2. a minimal rollout/update/resync health check after that exact optimizer audit.

Answer to the key question after D0.2:

> Were previous C8/B5 possibly using a vLLM behavior policy inconsistent with the
> training actor?

Yes. D0.1 established that historical C8/B5 vLLM rollout lacked the training
actor's 64-token MIV overwrite, so those rollouts could have been generated under a
different visual embedding policy. D0.2 repairs that mismatch and shows repaired
action-token parity on a small fixed probe, but historical C8/B5 samples were
collected before this repair.

D1 should still not start until the exact Ray/FSDP optimizer and weight-resync path
is checked with one bounded mini-step.

## J. D0.3 Exact FSDP Update / Weight-Sync / Resync Smoke

Date: 2026-08-09

Scope: D0.3 only. No D1, no validation benchmark, no EASI/OOD run, and no full RL
training. The only successful production run was a single B5-topology PPO step with
`TOTAL_STEPS=1`, `MAX_TURNS=1`, `MAX_RESPONSE_LENGTH=128`,
`actor_rollout_ref.rollout.calculate_log_probs=True`, checkpoint
`/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP`, and a temporary local HTTP
renderer on `127.0.0.1:18767`.

### 1. Plain Cambrian audit

Plain Cambrian-S checkpoint:
`/mnt/umm/users/yinbaiqiao/hf_cache/cambrian-s-7b`

Static config:

| Item | Result |
|---|---|
| model_type | `cambrian_qwen` |
| architecture | `CambrianQwenForCausalLM` |
| vision tower config | `google/siglip2-so400m-patch14-384` |
| vision select layer | `-2` |
| MIV token length | `64` |
| SI token length | `729` |
| image newline | enabled |
| nfp_head | absent / not enabled |

Plain checkpoint tensor audit found a 26-layer/no-head SigLIP2 vision tower. The
LFP checkpoint also has a 26-layer/no-head vision tower. Therefore the old vLLM
27-layer + head wrapper risk was not limited to LFP/C8/B5; plain Cambrian-S C-series
baselines that used the old wrapper may also have been affected. The repaired wrapper
is shape-compatible with the plain checkpoint, but D0.3 did not run a separate plain
Cambrian generation smoke.

### 2. Exact FSDP optimizer audit

The exact production path was used:

- `vagen/ray_trainer.py` recomputed actor `old_log_probs` via
  `actor_rollout_wg.compute_log_prob(batch)` before actor update.
- `vagen/ray_trainer.py` called the production actor update path
  `actor_rollout_wg.update_actor(batch)`.
- `verl/verl/workers/actor/dp_actor.py` executed the real optimizer path:
  gradient clipping followed by `self.actor_optimizer.step()`.

Audit files:
`exps/vagen_active_spatial/d0_3_b5_exact_fsdp_smoke/d0_3_smoke/optimizer_audit_rank*.json`

Observed on all 8 FSDP ranks:

| Rank | Grad clip norm | Trainable+optimizer params | Params with grad | Update norm |
|---:|---:|---:|---:|---:|
| 0 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0135580 |
| 1 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0129435 |
| 2 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0130833 |
| 3 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0147141 |
| 4 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0146090 |
| 5 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0134948 |
| 6 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0133835 |
| 7 | 13.748991 | 1,005,915,924 | 1,005,915,924 | 0.0133786 |

Important limitation: B5 uses `actor_rollout_ref.actor.fsdp_config.use_orig_params=false`.
The exact FSDP audit therefore exposes parameters as `fsdp_flat_mixed`; this confirms
real optimizer membership, gradients, and updates, but it cannot honestly separate
vision tower / mm projector / LLM depth / LM head / `nfp_head` at module granularity.
D0.2's single-GPU audit remains the module-level gradient evidence.

### 3. Pre-update parity

This is the critical D0.3 failure.

Before any actor optimizer step, the production vLLM rollout logprobs and FSDP actor
recomputed `old_log_probs` were compared on the real response mask:

File:
`exps/vagen_active_spatial/d0_3_b5_exact_fsdp_smoke/d0_3_smoke/pre_update_response_mask_parity_step1.json`

| Metric | Value |
|---|---:|
| response tokens | 880 |
| delta logp mean | 0.092956 |
| delta logp median | 0.085177 |
| delta logp std | 0.426834 |
| delta logp max abs | 8.036354 |
| exp(delta) mean | 1.148328 |
| exp(delta) median | 1.088910 |
| exp(delta) p5 / p95 | 0.734865 / 1.651172 |
| exp(delta) p1 / p99 | 0.367997 / 2.029384 |
| frac `abs(ratio-1)>0.01` | 0.793182 |
| frac `abs(ratio-1)>0.05` | 0.670455 |

Console metrics independently agree:

| Metric | Value |
|---|---:|
| `training/rollout_probs_diff_valid` | 1 |
| `training/rollout_probs_diff_mean` | 0.100288 |
| `training/rollout_probs_diff_max` | 0.999555 |
| `training/rollout_actor_probs_pearson_corr` | 0.922370 |

Interpretation: under a synced checkpoint and before optimizer update,
`exp(logp_train - logp_rollout)` is not concentrated near 1. This is a production
path policy/logprob mismatch. It may be due to policy forward differences, vLLM
logprob extraction/temperature semantics, response-token alignment, or another
rollout/logprob plumbing issue, but it is not a PASS-compatible result.

### 4. Weight sync audit

Static production path:

- `verl/verl/workers/fsdp_workers.py::rollout_mode()` loads/offloads the FSDP actor,
  gets `actor_module_fsdp.state_dict()`, converts keys, and calls
  `await self.rollout.update_weights(...)`.
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py::update_weights()`
  is the vLLM weight-loading entry point.

Runtime evidence:

- The B5 run used vLLM `load_format=dummy`; therefore generation depended on FSDP
  actor-to-vLLM weight sync.
- The run reached generation, old-logprob recompute, critic update, actor update,
  and post-update resync without a missing-weight or shape-load failure.
- `d0_3/post_update_resync_called=1.0` and
  `post_update_resync_step1.json` confirm the diagnostic resync hook called the
  production `rollout_mode()` after actor update.

Limitation: no `vllm_sync_audit_rank*.json` tensor-sample files were produced in the
final run, so tensor-level before/after vLLM weight equality is **not verified**.

### 5. Post-sync parity

Post-update production resync was called successfully:

```json
{
  "global_step": 1,
  "tag": "post_update_resync",
  "used_production_rollout_mode": true
}
```

The timed metric was `timing_s/d0_3_post_update_resync=4.479668`, and the training
metric contained `d0_3/post_update_resync_called=1.0`.

However, D0.3 did not perform a second post-sync generation/logprob recompute on the
same samples. Post-sync policy parity is therefore **not verified**. Because the
pre-update parity already fails, this does not change the Gate decision.

### 6. Minimal RL health smoke

The bounded one-step smoke completed:

| Metric | Value |
|---|---:|
| `trainer/actor_update_performed` | 1.0 |
| `trainer/actor_update_skipped_by_warmup` | 0.0 |
| `actor/pg_loss` | -0.117676 |
| `actor/grad_norm` | 13.748991 |
| `actor/nfp_loss_active` | 1.0 |
| `actor/nfp_loss` | 0.0 |
| `critic/vf_loss` | 19.6953125 |
| `transition/invalid_action_rate/mean` | 0.375 |
| `transition/missing_action_tag_rate/mean` | 0.25 |
| `response_length/clip_ratio` | 0.375 |
| `num_turns/mean` | 1.0 |

This confirms the exact C8/B5 Ray/FSDP/vLLM topology can run one real mini-batch
through rollout, FSDP logprob recompute, critic update, actor optimizer step, and
post-update resync. It also shows the policy is still unhealthy at the plumbing level:
rollout and FSDP recomputed logprobs disagree before the update.

### 7. Remaining risks

- The exact production path still has a substantial train-vs-rollout logprob mismatch
  before optimizer step.
- The source is not isolated yet. Next minimal ablations should focus on vLLM returned
  logprob extraction/alignment, temperature normalization, action-token-only masking,
  and image-token/MIV placement in the exact async rollout path.
- Tensor-level vLLM weight-sync audit did not write files, so exact post-load equality
  is not verified even though production sync/resync calls completed.
- Module-level exact FSDP optimizer grouping is blocked by `use_orig_params=false`;
  only flat-shard membership/update is proven in D0.3.
- The old remote renderer `10.119.30.223:8767` was down during D0.3
  (`connection refused`); the successful smoke used a temporary local HTTP renderer.

### 8. Gate decision

Decision: **FAIL**

Reason: D0.3 has clear evidence that, in the exact B5 Ray/FSDP/vLLM production path,
the behavior policy logprobs returned by rollout and the FSDP actor recomputed
logprobs are systematically different before the actor optimizer step. The expected
`exp(logp_train - logp_rollout)` concentration near 1 is not present.

Answer to the key question after D0.3:

> Were previous C8/B5 possibly using a vLLM behavior policy inconsistent with the
> training actor?

Yes. D0.1/D0.2 already showed historical C8/B5 had a real forward-path mismatch
around MIV/SigLIP handling before repair. D0.3 now adds that even after the D0.2
repair, the exact production Ray/FSDP/vLLM path still produces mismatched rollout
vs actor recomputed logprobs before the optimizer step. Therefore D1 must not start.
The next work should remain D0 plumbing localization and repair, not hyperparameter
search or full RL training.

## K. D0.4 Production Logprob Localization

Date: 2026-08-09

Scope: D0.4 only. No D1, no validation benchmark, no EASI/OOD run, and no full RL
training run were started. The probe used a bounded one-step B5-style production
path with 8 real Active Spatial rollouts, `MAX_TURNS=1`, `MAX_RESPONSE_LENGTH=128`,
and `TOTAL_STEPS=1`.

Run script:

- `examples/train/active_spatial/experiments/d0_4_b5_logprob_localization.sh`

Artifact directory:

- `exps/vagen_active_spatial/d0_4_b5_logprob_localization/d0_4/`

Artifacts:

- `d0_4_fixed_batch_manifest.json`
- `d0_4_pre_fix_parity.json`
- `d0_4_post_fix_parity.json`
- `d0_4_token_alignment.csv`
- `d0_4_top_mismatch_tokens.json`
- `vllm_raw_logprobs.jsonl`

### 1. Root cause localization

D0.4 localized one concrete production mismatch:

- vLLM rollout samples with `temperature=0.6`, `top_p=0.9`.
- vLLM's returned sampled-token logprobs are raw model logprobs for the sampled
  token, not `log_softmax(logits / 0.6)` logprobs.
- The production FSDP `compute_log_prob` path had been reusing
  `actor_rollout_ref.rollout.temperature` as the actor-side logprob temperature.
  In B5 this made recomputed `old_log_probs` use `log_softmax(logits / 0.6)`.

This is a PPO plumbing bug: rollout records the behavior-policy sampled-token
logprob under one semantics, while the training batch recomputes old logprobs under
a sharper temperature-scaled semantics.

The fix separates sampling temperature from logprob recompute temperature:

- `actor_rollout_ref.rollout.temperature` remains the vLLM sampling temperature.
- `actor_rollout_ref.rollout.logprob_temperature` controls FSDP actor-side logprob
  recomputation and defaults to `1.0`.
- D0.4 runs with `temperature=0.6` and `logprob_temperature=1.0`.

Modified code paths:

- `verl/verl/workers/fsdp_workers.py`: use `logprob_temperature` for
  `actor.compute_log_prob`, and keep an optional D0.4 legacy comparison at sampling
  temperature.
- `verl/verl/workers/config/rollout.py` and
  `verl/verl/trainer/config/rollout/rollout.yaml`: add the new config field.
- `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`: dump raw vLLM
  sampled-token logprob records for D0.4.
- `vagen/agent_loop/gym_agent_loop_no_concat.py`: pass a stable D0.4 request id into
  the training batch.
- `vagen/ray_trainer.py`: write token alignment, parity, and mismatch artifacts.
- `verl/verl/trainer/constants_ppo.py`: propagate D0.4 diagnostic env vars into Ray.

### 2. Token alignment audit

The production batch contained 8 requests and 957 valid response tokens.

Strict matching between `d0_4_token_alignment.csv` and `vllm_raw_logprobs.jsonl`:

| Check | Value |
|---|---:|
| active CSV response tokens | 957 |
| vLLM raw request records | 8 |
| matched request/index/token/logprob rows | 957 |
| missing request ids | 0 |
| out-of-range token indices | 0 |
| token-id/logprob mismatches | 0 |

Shift ablation rules out a simple off-by-one token alignment bug:

| Comparison | Pearson | Mean abs delta logp | Median abs delta logp | p95 abs delta logp |
|---|---:|---:|---:|---:|
| same index | 0.909070 | 0.108650 | 0.046175 | 0.373120 |
| FSDP `k-1` | 0.191947 | 0.617650 | 0.442020 | 1.831308 |
| FSDP `k+1` | 0.203711 | 0.618741 | 0.441258 | 1.820756 |

Conclusion: the remaining mismatch is not caused by reading the wrong generated
token from vLLM output or by a one-token response shift.

### 3. Logprob semantics audit

Before the fix, D0.4 reproduces the D0.3 failure pattern when FSDP recomputes
old logprobs at sampling temperature `0.6`:

| Metric | Pre-fix semantics |
|---|---:|
| response tokens | 957 |
| delta logp mean | 0.087355 |
| delta logp median | 0.094154 |
| delta logp std | 0.528555 |
| delta logp max abs | 11.226469 |
| mean abs delta logp | 0.242491 |
| median abs delta logp | 0.163163 |
| p95 abs delta logp | 0.686461 |
| `exp(delta)` mean | 1.162239 |
| `exp(delta)` median | 1.098729 |
| `exp(delta)` p5 / p95 | 0.703107 / 1.726636 |
| `exp(delta)` p1 / p99 | 0.224934 / 2.115702 |
| frac `abs(ratio-1)>0.01` | 0.801463 |
| frac `abs(ratio-1)>0.05` | 0.691745 |
| Pearson | 0.794220 |

After recomputing FSDP old logprobs at raw temperature `1.0`, parity improves
substantially:

| Metric | Post-fix semantics |
|---|---:|
| response tokens | 957 |
| delta logp mean | -0.061334 |
| delta logp median | -0.010110 |
| delta logp std | 0.311156 |
| delta logp max abs | 7.231855 |
| mean abs delta logp | 0.108650 |
| median abs delta logp | 0.046175 |
| p95 abs delta logp | 0.373120 |
| `exp(delta)` mean | 0.962968 |
| `exp(delta)` median | 0.989940 |
| `exp(delta)` p5 / p95 | 0.716316 / 1.160258 |
| `exp(delta)` p1 / p99 | 0.433697 / 1.356107 |
| frac `abs(ratio-1)>0.01` | 0.709509 |
| frac `abs(ratio-1)>0.05` | 0.475444 |
| Pearson | 0.909070 |

Conclusion: temperature semantics are a real root cause of the D0.3 production
logprob mismatch, but not the only remaining source of all-response-token drift.

### 4. Production input parity

The fixed batch manifest records:

| Field | Value |
|---|---|
| batch size | 8 |
| input shape | `[8, 2931]` |
| attention mask shape | `[8, 2931]` |
| position ids shape | `[8, 2931]` |
| response shape | `[8, 128]` |
| rollout sampling temperature | `0.6` |
| FSDP logprob temperature | `1.0` |
| pad token id | `151643` |
| eos token id | `151645` |

Each sample includes `pixel_values`, `nfp_pixel_values`, and `nfp_loss_mask` in
`multi_modal_inputs`. The pixel tensors have shape `[1,3,384,384]`, finite float32
values, and matching `pixel_values` / `nfp_pixel_values` digests for this one-turn
probe.

The D0.4 run also produced rollout-side vision sanity dumps with square
`[256,256,3]` RGB observations, normalized `[1,3,384,384]` tensors, one compact
image token, and 756 expanded image tokens. This matches the repaired Cambrian path
used since D0.2.

### 5. Before/after metrics

The fix improves every main aggregate except that a long tail remains:

| Metric | Pre-fix | Post-fix |
|---|---:|---:|
| mean abs delta logp | 0.242491 | 0.108650 |
| median abs delta logp | 0.163163 | 0.046175 |
| p95 abs delta logp | 0.686461 | 0.373120 |
| `exp(delta)` mean | 1.162239 | 0.962968 |
| `exp(delta)` median | 1.098729 | 0.989940 |
| frac `abs(ratio-1)>0.05` | 0.691745 | 0.475444 |
| Pearson | 0.794220 | 0.909070 |

Top post-fix mismatches are concentrated in free-form `<think>` text, not in the
action name tokens. The largest observed outlier was token `" for"` in think text:
vLLM logprob `-0.989939`, FSDP raw-T1 logprob `-8.221794`, delta `-7.231855`.

### 6. Action-token-specific parity

Action-token parity is much healthier than all-response parity after the fix.

For `action_tag + action_name` tokens:

| Metric | Pre-fix | Post-fix |
|---|---:|---:|
| token count | 45 | 45 |
| mean abs delta logp | 0.031545 | 0.014119 |
| median abs delta logp | 0.003096 | 0.001432 |
| max abs delta logp | 0.223016 | 0.213006 |
| `exp(delta)` mean | 1.018839 | 0.988950 |
| `exp(delta)` median | 1.002898 | 0.999329 |
| frac `abs(ratio-1)>0.01` | 0.355556 | 0.177778 |
| frac `abs(ratio-1)>0.05` | 0.222222 | 0.088889 |

For `action_name` tokens only:

| Metric | Pre-fix | Post-fix |
|---|---:|---:|
| token count | 18 | 18 |
| mean abs delta logp | 0.057320 | 0.016032 |
| median abs delta logp | 0.035927 | 0.003967 |
| max abs delta logp | 0.223016 | 0.184564 |
| `exp(delta)` mean | 1.036465 | 0.988856 |
| `exp(delta)` median | 1.030008 | 0.998775 |
| frac `abs(ratio-1)>0.05` | 0.388889 | 0.055556 |

Conclusion: the action policy tokens now look close under the exact production
rollout/recompute path. The residual drift is mainly in long generated reasoning
text.

### 7. Post-update/resync parity

D0.4 completed one exact production actor update:

| Metric | Value |
|---|---:|
| `trainer/actor_update_performed` | 1.0 |
| `actor/grad_norm` | 14.417351 |
| `actor/pg_loss` | -0.000488 |
| `actor/kl_loss` | 0.052326 |
| `actor/nfp_loss_active` | 1.0 |
| `critic/vf_loss` | 31.796875 |

D0.3 already verified that the post-update production resync hook is called, and
D0.4 did not introduce a new weight-sync path. However, D0.4 still did not run a
second same-sample post-update generation/logprob recompute after resync. Therefore
post-update same-sample parity remains **not verified**.

### 8. Remaining risks

- The all-response-token parity is improved but not PASS-clean. Nearly half of valid
  response tokens still have `abs(exp(delta)-1)>0.05`, and the largest think-text
  outlier is large.
- PPO still trains over the full response mask, not just final action-name tokens.
  Good action-token parity is encouraging, but it does not fully eliminate the
  optimization risk from reasoning-token old-logprob drift.
- Tensor-level post-load vLLM weight equality remains unverified.
- Same-sample post-update/resync parity remains unverified.
- Module-level exact FSDP optimizer grouping is still limited by flat FSDP shards in
  the production B5 topology.

### 9. Gate decision

Decision: **FAIL**

Reason: D0.4 found and fixed a clear production bug in old-logprob temperature
semantics, and action-token-specific parity is now close. However, the exact
production path still has nontrivial all-response-token rollout-vs-FSDP logprob
drift, and post-update same-sample resync parity is still not verified. Because PPO
uses the full response mask, this is not yet a clean D0 PASS.

Answer to the key question after D0.4:

> Were previous C8/B5 possibly using a vLLM behavior policy inconsistent with the
> training actor?

Yes. The evidence is now stronger than D0.3:

1. Historical C8/B5 had the earlier D0.1 MIV/SigLIP forward-path mismatch before the
   D0.2 repair.
2. Historical C8/B5 also recomputed FSDP `old_log_probs` under
   `temperature=0.6`, while production vLLM returned raw sampled-token logprobs.

So C8/B5 could have generated rollouts from a vLLM behavior policy whose returned
logprob semantics did not match the FSDP actor old-logprob semantics used by PPO.
D1 should still not start until the residual all-response drift and post-update
same-sample parity are resolved or explicitly bounded.

## L. D0.5 Residual Think-Token Logprob Drift

Date: 2026-08-10

Scope: D0.5 only. No D1 experiment, no full RL training, no reward/task/prompt
change, and no new hyperparameter search was started. The run was a bounded
one-step smoke probe:

- script: `examples/train/active_spatial/experiments/d0_5_b5_vision_select_logprob.sh`
- artifacts: `exps/vagen_active_spatial/d0_5_b5_vision_select_logprob/d0_5/`
- summary artifacts:
  - `docs/diagnosis/d0_5_before_position_buckets.json`
  - `docs/diagnosis/d0_5_before_token_type_summary.json`
  - `docs/diagnosis/d0_5_before_loss_contribution.json`
  - `docs/diagnosis/d0_5_after_position_buckets.json`
  - `docs/diagnosis/d0_5_after_token_type_summary.json`
  - `docs/diagnosis/d0_5_after_loss_contribution.json`

### 1. Weight-sync equality

Status: **NOT VERIFIED**.

D0.5 did not complete a tensor-level equality audit between the FSDP actor shards and
the vLLM worker weights after load/sync. The production run used the existing
dummy-load plus sync path and completed generation/logprob recomputation, but that is
not a substitute for exact per-module tensor comparison. This remains a hard blocker
for D0 PASS.

### 2. Vision/multimodal equality

Status: **DIFFERENT before D0.5; partially repaired in code; dynamic equality still
not fully proven**.

D0.5 found a concrete static forward mismatch:

- FSDP/HF Cambrian vision tower uses `hidden_states[mm_vision_select_layer]`.
- The LFP config has `mm_vision_select_layer=-2`.
- The vLLM wrapper was using `last_hidden_state`.

The upstream Cambrian tower stores the requested layer and selects it directly:

```python
self.select_layer = args.mm_vision_select_layer
image_features = image_forward_outs.hidden_states[self.select_layer]
```

The vLLM wrapper was patched to request hidden states and select the same config
layer:

```python
vision_outputs = self.vision_tower(pixel_values=pixel_values.to(vt_dtype), output_hidden_states=True)
select_layer = int(getattr(self.hf_config, "mm_vision_select_layer", -1))
features = vision_outputs.hidden_states[select_layer]
```

This is a real production-path fix, but D0.5 did not dump paired FSDP/vLLM vision
features from the exact same sample to prove tensor equality after the repair.

### 3. Position dependence

Status: **no evidence for monotonically growing position/KV-cache drift**.

Offline bucketing over D0.4 and D0.5 token-alignment CSVs shows weak negative
correlation between response position and absolute logprob delta:

| Run | Tokens | corr(position, abs delta logp) |
|---|---:|---:|
| D0.4 post temperature fix | 957 | -0.036771 |
| D0.5 after vision-layer repair | 921 | -0.049004 |

D0.5 bucket examples:

| Response pos bucket | Tokens | Mean abs delta | Median abs delta | Max abs delta |
|---|---:|---:|---:|---:|
| 0-15 | 128 | 0.087844 | 0.037545 | 0.672933 |
| 16-31 | 128 | 0.197783 | 0.074426 | 2.827775 |
| 80-95 | 112 | 0.134451 | 0.052283 | 5.178907 |
| 112-127 | 79 | 0.058249 | 0.007823 | 0.987761 |

The largest residuals are localized outliers in think text, not a smooth late-token
accumulation pattern.

### 4. Full vs incremental matrix

Status: **NOT VERIFIED**.

D0.5 did not implement the requested forced fixed-sequence scoring matrix:

| Condition | Status |
|---|---|
| FSDP full forward | Existing production recompute only |
| FSDP incremental / past-key-values | Not run |
| vLLM prefill+decode | Existing production generation only |
| vLLM forced scoring with identical generated sequence | Not run |

The existing same-index shift ablation still rules out a simple response-token
off-by-one bug, but it does not prove full-vs-incremental equivalence.

### 5. Batch/position/mode/backend audit

Status: **PARTIAL**.

Known D0.5 production settings:

| Item | Value / status |
|---|---|
| batch | 8 prompts, 921 valid response tokens |
| response length cap | 128 |
| rollout sampling temperature | 0.6 |
| FSDP logprob temperature | 1.0 |
| vLLM attention backend | `TORCH_SDPA` |
| Transformers attention implementation | `eager` |
| vLLM dtype | bf16 |
| vLLM prefix caching | enabled |
| FSDP adapter mode for old logprob | production recompute path |
| dropout/mode ablation | not run |
| batch=1 ablation | not run |

Position dependence was audited offline and does not point to KV-cache accumulation.
Mode/backend parity is still not fully isolated.

### 6. Top mismatch explanation

Status: **PARTIAL**.

D0.5 top residual mismatches remain in free-form `<think>` text:

| Token | Type | vLLM logp | FSDP raw-T1 logp | Delta |
|---|---|---:|---:|---:|
| `" around"` | think_text | -0.202066 | -5.380974 | -5.178907 |
| `" wardrobe"` | think_text | -1.538276 | -4.366051 | -2.827775 |
| `" plant"` | think_text | -1.104052 | -3.455593 | -2.351541 |
| `" the"` | think_text | -0.768625 | -2.854136 | -2.085511 |

D0.5 still only has sampled-token logprob alignment, not top-10 vocabulary logits
from both sides. Therefore the exact rank-level explanation for the top mismatch is
not yet proven.

### 7. PPO loss impact

Status: **PARTIAL; risk remains because PPO trains over the full response mask**.

Offline proxy after D0.5:

| Token group | Tokens | Mean abs delta | Median ratio | frac abs(ratio-1)>0.05 |
|---|---:|---:|---:|---:|
| all response | 921 | 0.111270 | 0.984164 | 0.472313 |
| think_text | 795 | 0.120834 | 0.978906 | 0.511950 |
| action_tag + action_name | 62 | 0.019601 | 0.998535 | 0.112903 |
| action_name only | 26 | 0.023477 | 0.998483 | 0.153846 |

Action-token parity is much healthier than all-response parity, but the training
objective still consumes full-response old logprobs. The CSV artifacts do not include
advantages/current-policy logprobs, so D0.5 cannot compute signed per-token PG/KL
loss contribution.

The one-step production smoke still performed an actor update:

| Metric | Value |
|---|---:|
| `trainer/actor_update_performed` | 1.0 |
| `actor/pg_loss` | -0.185486 |
| `actor/kl_loss` | 0.048678 |
| `actor/grad_norm` | 18.903093 |
| `actor/nfp_loss_active` | 1.0 |

### 8. Before/after parity

The D0.5 vision-layer repair did not resolve the remaining post-temperature-fix
all-response drift.

| Metric | D0.4 post temp fix | D0.5 after vision-layer repair |
|---|---:|---:|
| response tokens | 957 | 921 |
| mean abs delta logp | 0.108650 | 0.111270 |
| median abs delta logp | 0.046175 | 0.044414 |
| p95 abs delta logp | 0.373120 | 0.392264 |
| max abs delta logp | 7.231855 | 5.178907 |
| ratio mean | 0.962968 | 0.955721 |
| ratio median | 0.989940 | 0.984164 |
| frac abs(ratio-1)>0.05 | 0.475444 | 0.472313 |
| Pearson | 0.909070 | 0.924458 |

Within the D0.5 run, the old pre-fix temperature semantics are again much worse than
the corrected raw-T1 semantics:

| Metric | D0.5 pre-fix temp semantics | D0.5 post-fix raw-T1 semantics |
|---|---:|---:|
| mean abs delta logp | 0.229509 | 0.111270 |
| median abs delta logp | 0.143272 | 0.044414 |
| p95 abs delta logp | 0.682447 | 0.392264 |
| max abs delta logp | 7.541082 | 5.178907 |
| ratio median | 1.070362 | 0.984164 |
| frac abs(ratio-1)>0.05 | 0.666667 | 0.472313 |
| Pearson | 0.816348 | 0.924458 |

### 9. Post-update/resync parity

Status: **NOT VERIFIED**.

D0.5 completed one actor update, but it did not run a second same-sample
post-update/resync forced parity probe. The existing resync hook is exercised by the
production trainer, but same-sample post-update equality remains an evidence gap.

### 10. Gate decision

Decision after D0.5: **FAIL**.

Reasons:

1. A real additional train/rollout forward mismatch was found: FSDP selected
   `mm_vision_select_layer=-2`, while vLLM used `last_hidden_state`.
2. The mismatch was patched in the vLLM wrapper, but after the patch the production
   one-step probe still has nontrivial all-response logprob drift:
   mean abs delta `0.111270`, median ratio `0.984164`, and
   `47.23%` of response tokens with `abs(ratio-1)>0.05`.
3. Tensor-level weight-sync equality, paired vision-feature equality,
   full-vs-incremental forced scoring, and post-update/resync same-sample parity are
   still not verified.
4. Because PPO still trains over the full response mask, healthy action-token parity
   alone is not enough for D0 PASS.

Updated answer to the key question:

Yes, previous C8/B5 could have been using a vLLM behavior path inconsistent with the
FSDP training actor. The evidence now includes three independent production-relevant
issues:

1. D0.1: historical MIV/processor/vision-config mismatch before D0.2 repair.
2. D0.4: vLLM returned raw sampled-token logprobs while FSDP old logprobs were
   recomputed at sampling temperature `0.6`.
3. D0.5: FSDP selected the configured SigLIP hidden layer `-2`, while the vLLM
   wrapper used the last hidden state before the D0.5 patch.

D1 remains blocked. The next D0-only step should be a forced scorer matrix with exact
weight equality and paired hidden-feature dumps, not a new RL run.

## M. D0.6 Forced Scoring Matrix / Exact Weight Equality

Date: 2026-08-12

Scope: D0.6 only. No D1 experiment, no C8/B5 rerun, no navigation evaluation, no
reward/prompt/objective change, and no action-only PPO change.

### What was attempted

A bounded D0.6 one-step probe was launched on the same B5/Cambrian production path:

- script: `examples/train/active_spatial/experiments/d0_6_b5_forced_scoring_matrix.sh`
- experiment: `d0_6_b5_forced_scoring_matrix`
- rollout sampling temperature: `0.6`
- FSDP logprob temperature: `1.0`
- `total_training_steps=1`
- `critic_warmup=999`, so actor update is skipped
- validation/checkpoint save disabled
- renderer reused at `http://127.0.0.1:18767`

The run produced normal D0.4-style production artifacts under:

- `exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/d0_6/`

However, no `d0_6_weight_equality*.json` was written.

### Important correction

The first D0.6 weight-equality hook was incorrectly placed in
`FSDPVLLMShardingManager.__enter__`. B5 uses async vLLM rollout, whose real sync path
is:

```text
ActorRolloutRefWorker.rollout_mode()
→ self.rollout.update_weights(...)
→ vLLMAsyncRollout.update_weights()
→ model.load_weights(weights)
```

Therefore the completed D0.6 one-step artifact did not actually execute the
tensor-level equality check.

The diagnostic hook has now been moved/added to the actual async update path:

- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/verl/workers/sharding_manager/fsdp_vllm.py`
- `verl/verl/trainer/constants_ppo.py`

The new hook caches only the requested source tensors/shards, not the full 8B
checkpoint. For `embed_tokens` and `lm_head`, it stores only the local TP row shard.

### Existing D0.6 production logprob result

The completed one-step production probe still shows the same broad pattern:

| Metric | D0.6 value |
|---|---:|
| response tokens | 878 |
| mean abs delta logp | 0.114605 |
| median abs delta logp | 0.035906 |
| p95 abs delta logp | 0.382577 |
| max abs delta logp | 10.731915 |
| ratio median | 0.991498 |
| frac `abs(ratio-1)>0.05` | 0.419134 |
| Pearson | 0.844631 |

Token-type split:

| Token group | Tokens | Mean abs delta | Ratio median | frac `abs(ratio-1)>0.05` |
|---|---:|---:|---:|---:|
| think_text | 658 | 0.119801 | 0.991114 | 0.430091 |
| action_name | 18 | 0.012547 | 0.999436 | 0.055556 |
| action_tag | 30 | 0.070680 | 0.999166 | 0.100000 |
| eos | 5 | 0.000485 | 0.999521 | 0.000000 |

The largest residual remains a free-form think token:

| Token | Type | vLLM logp | FSDP raw-T1 logp | Delta |
|---|---|---:|---:|---:|
| `"0"` | think_text | -0.004349 | -10.736263 | -10.731914 |
| `" for"` | think_text | -0.298954 | -3.873804 | -3.574850 |
| `" wardrobe"` | think_text | -1.336227 | -4.446444 | -3.110216 |

This run does not localize the root cause because the requested A/B/C/D forced
scoring matrix was not yet executed.

### Current evidence status

| Required D0.6 item | Status |
|---|---|
| fixed D0.5/D0.6 samples with exact production arrays | instrumentation added, not rerun |
| tensor-level FSDP/vLLM weight equality | instrumentation added to real async path, not rerun |
| paired vision/multimodal feature equality | not executed |
| A. FSDP full-sequence scoring | existing production recompute only |
| B. HF incremental teacher-forced scoring | not executed |
| C. vLLM production incremental logprob | available from production artifacts |
| D. vLLM fixed-sequence forced scoring | not executed / availability not verified |
| A/B/C/D pairwise matrix | not executed |
| batch=1 ablation | not executed |
| train/eval/dropout audit | not executed |
| top-k logits analysis | not executed |
| PPO signed loss impact | not executed |
| post-update/resync parity | not executed |

### Infrastructure blocker

After adding the corrected async-path instrumentation, the specified GPU node became
unreachable from this session:

```text
ssh root@10.119.24.168 → No route to host
```

The rerun needed to produce `d0_6_weight_equality.json`,
`d0_6_fixed_sequence_manifest.json`, paired feature equality, and the forced scoring
matrix could therefore not be completed in this turn.

### Gate decision after D0.6 attempt

Decision: **INCONCLUSIVE for D0.6; overall D0 remains FAIL**.

Reasons:

1. The required tensor-level production weight equality was not obtained.
2. The required paired vision/multimodal feature equality was not obtained.
3. The A/B/C/D forced scoring matrix was not run.
4. The existing D0.6 production artifact still shows nontrivial all-response
   residual drift, mostly in think text.

D1 remains blocked. The next action should be to rerun the same bounded D0.6 script
after a GPU node is reachable, then stop immediately if weight equality fails.

## N. D0.6 Continuation: GPU Reachability and Instrumentation Readiness

Date: 2026-08-13

Scope remains D0.6 only. No D1 experiment, no C8/B5 rerun, no navigation evaluation,
and no full RL training.

### GPU reachability

The original node is still unreachable:

```text
ssh root@10.119.24.168 -> No route to host
```

The replacement 8-GPU node was attempted next:

```text
ssh root@10.119.30.117 -> Connection refused
```

The current local container also cannot run the probe:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

Therefore the corrected async-path D0.6 probe still has not produced new GPU
evidence.

### Instrumentation validation

Static readiness checks passed:

```text
bash -n examples/train/active_spatial/experiments/d0_6_b5_forced_scoring_matrix.sh
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python -m py_compile \
  verl/verl/workers/sharding_manager/fsdp_vllm.py \
  verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py \
  vagen/ray_trainer.py \
  verl/verl/trainer/constants_ppo.py
```

The corrected async hook is present in:

```text
verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
```

It wraps the actual production async `update_weights()` path and writes
`d0_6_weight_equality_rank*_tp*_pid*.json` after `model.load_weights(...)`.

The tensor list covers the requested policy-relevant groups:

| Group | Covered tensors |
|---|---|
| vision tower | encoder layer 0 q, layer 13 q, layer 25 q, post-layernorm |
| mm projector | `mm_projector.0/2` weight and bias |
| multimodal | `image_newline` |
| LLM embeddings/head | TP shard of `embed_tokens`, TP shard of `lm_head` |
| LLM attention | layers 0/14/27 q/k/v packed shard, o-proj column shard |
| LLM MLP | layers 0/14/27 gate/up packed shard, down-proj column shard |
| LLM norm | final norm |

For each compared tensor the artifact records shape, dtype, source/target checksum,
max abs diff, mean abs diff, L2, cosine, and representative sample values.

### Current D0.6 status

No new `d0_6_weight_equality*.json`, `d0_6_fixed_sequence_manifest.json`,
`d0_6_vision_feature_equality.json`, or A/B/C/D scoring matrix was generated in this
continuation because no GPU node was reachable.

Decision remains: **INCONCLUSIVE for D0.6 evidence; overall D0 remains FAIL**.

D1 remains blocked.

## D0.8 — Prefill / Position / Attention Backend Divergence Localization

Date: 2026-08-15

Gate after D0.8: **FAIL — vLLM prefill/runtime root cause remains**

D0.8 did not start renderer, ActiveSpatial, GymAgentLoop, PPO/Ray training,
critic, optimizer step, C8/B5 rerun, or navigation evaluation.  It used the
renderer-free fixed request replay from D0.7b, focusing on sample 5 and three
positions:

| Label | Response index | Token | Absolute position | Role |
|---|---:|---|---:|---|
| P0 | 0 | `<th` | 1414 | first response control |
| P1 | 73 | ` required` | 1487 | large think-token outlier |
| P2 | 87 | `move` | 1501 | healthy action-token control |

Artifacts produced:

- `docs/diagnosis/d0_8_runtime_truth.json`
- `docs/diagnosis/d0_8_prefill_chunks.json`
- `docs/diagnosis/d0_8_single_vs_multi_chunk.json`
- `docs/diagnosis/d0_8_sequence_layout.json`
- `docs/diagnosis/d0_8_position_semantics.json`
- `docs/diagnosis/d0_8_input_embedding_parity.json`
- `docs/diagnosis/d0_8_raw_logits.json`
- `docs/diagnosis/d0_8_backend_matrix.json`
- `docs/diagnosis/d0_8_before_after_parity.json`
- `docs/diagnosis/d0_8_hf_probe.json`
- `docs/diagnosis/d0_8_vllm_forced_baseline_repeat.json`
- `docs/diagnosis/d0_8_vllm_forced_multi_1024.json`
- `docs/diagnosis/d0_8_vllm_forced_after_native_siglip.json`
- `docs/diagnosis/d0_8_vllm_forced_after_native_siglip_logits.json`
- raw worker captures under `docs/diagnosis/d0_8_runs/`

### D0.8.1 Runtime Truth

The requested node was usable:

```text
10.119.28.231
hostname: pt-f548cd278c8e43a5ac7e86ce8138c286-worker-0
GPU: 8x H800
```

The installed vLLM runtime was:

| Item | Value |
|---|---|
| vLLM version | 0.11.0 |
| engine | V1 |
| model dtype | bf16 |
| tensor parallel | 2 |
| `enforce_eager` | true |
| kv cache dtype | auto |
| prefix cache | enabled in baseline |
| chunked prefill | enabled |
| requested backend | `TORCH_SDPA` |
| actual backend | Flash Attention |

Source audit showed why the TORCH_SDPA request did not take effect:

| Source | Finding |
|---|---|
| `vllm/attention/selector.py:181-200` | environment/backend override is parsed |
| `vllm/platforms/cuda.py:300-368` | the V1 CUDA path has no TORCH_SDPA branch; SM80+ falls through to FlashAttention |

So D0.8 did **not** obtain a true vLLM V1 TORCH_SDPA matched-backend run.  The
production vLLM runtime for these probes was Flash Attention.

### D0.8.2 Determinism and Chunked Prefill

Same fixed request, same seed, same runtime, repeated twice:

| Token | Repeat 1 logp/rank | Repeat 2 logp/rank | Result |
|---|---:|---:|---|
| `<th` | -1.727953 / 2 | -1.727953 / 2 | exact |
| ` required` | -2.082076 / 3 | -2.082076 / 3 | exact |
| `move` | -0.000241 / 1 | -0.000241 / 1 | exact |

The default/baseline worker capture was a single prefill call per TP worker:

```text
tokens = 1514
positions = 0..1513
contains P0/P1/P2
```

The `max_num_batched_tokens=1024` run produced true multi-chunk prefill:

```text
chunk 1: 1024 tokens, positions 0..1023
chunk 2: 490 tokens, positions 1024..1513, contains P0/P1/P2
```

Single vs multi-chunk scoring:

| Token | HF B logp/rank | Single vLLM logp/rank | Multi vLLM logp/rank | Multi - single |
|---|---:|---:|---:|---:|
| `<th` | -2.301685 / 2 | -1.727953 / 2 | -1.630633 / 2 | +0.097321 |
| ` required` | -6.588740 / 21 | -2.082076 / 3 | -2.096742 / 3 | -0.014665 |
| `move` | -0.000218 / 1 | -0.000241 / 1 | -0.000245 / 1 | -0.000004 |

Conclusion: chunking changes small numeric details, but P1 remains vLLM-like
with rank 3 and is nowhere near HF B rank 21.  Chunked prefill is not the root
cause of the residual distribution mismatch.

### D0.8.3 Sequence and Position Semantics

HF and vLLM sequence layout match after removing HF left padding:

| Item | HF | vLLM |
|---|---:|---:|
| expanded prompt length | 1414 | 1414 |
| response length | 100 | 100 |
| visual span | [470, 1226) | [470, 1226) |
| visual token count | 756 | 756 |
| P0 absolute position | 1414 | 1414 |
| P1 absolute position | 1487 | 1487 |
| P2 absolute position | 1501 | 1501 |

HF valid-token `position_ids` and vLLM worker positions are contiguous `0..1513`.
The image block occupies positions `470..1225`, and text resumes at 1226 in both
paths.  This rules out prompt expansion, image span, left padding removal, and
basic position-id construction as the immediate source of B/C divergence.

### D0.8.4 Layer-0 Input Embedding Parity and Repair

Before D0.8 repair, text and newline positions matched exactly, but projected
visual-token positions did not:

| Position | Meaning | max abs diff | mean abs diff | cosine |
|---:|---|---:|---:|---:|
| 470 | visual offset 0 | 0.056641 | 0.011393 | 0.999561 |
| 501 | visual offset 31 | 0.119141 | 0.025538 | 0.998928 |
| 533 | visual offset 63 | 0.125000 | 0.021625 | 0.999730 |
| 534 | visual offset 64 | 0.164062 | 0.026801 | 0.999736 |
| 600 | visual offset 130 | 0.156250 | 0.033276 | 0.998040 |
| 1000 | visual offset 530 | 0.218750 | 0.039328 | 0.998753 |
| 1225 | newline/end | 0.000000 | 0.000000 | 1.000000 |
| 1226 | first text after visual | 0.000000 | 0.000000 | 1.000000 |
| 1414 | P0 | 0.000000 | 0.000000 | 1.000000 |
| 1487 | P1 | 0.000000 | 0.000000 | 1.000000 |
| 1501 | P2 | 0.000000 | 0.000000 | 1.000000 |

The concrete cause was that `CambrianVLLMForCausalLM` instantiated
`transformers.SiglipVisionModel`, while the training adapter uses Cambrian's
native SigLIP implementation from:

```text
/mnt/umm/users/yinbaiqiao/cambrian-s/cambrian/model/multimodal_encoder/llava_next_siglip_encoder.py
```

Repair applied in `vagen/models/cambrian_vllm.py`:

- prefer Cambrian native `SigLipVisionConfig` / `SigLipVisionModel` from
  `CAMBRIAN_SRC`;
- keep Transformers SigLIP only as fallback;
- neutralize the optional vision `head` with `nn.Identity()` when present;
- retain the existing selected-layer, projector, newline, and MIV handling.

After this repair, selected layer-0 input embeddings match exactly for all
checked visual/text/control positions:

| Checked positions | max abs diff | mean abs diff | cosine |
|---|---:|---:|---:|
| 470, 501, 533, 534, 600, 1000, 1225, 1226, 1414, 1487, 1501 | 0.0 | 0.0 | approximately 1.0 |

This repairs the remaining input-prefix mismatch found in D0.8, but it does not
repair final token distribution parity.

### D0.8.5 Raw Logits After Input Parity Repair

After exact layer-0 input parity, vLLM still differs from HF incremental at raw
pre-softmax logits:

| Token | HF raw/logp/rank | vLLM raw/logp/rank | Interpretation |
|---|---:|---:|---|
| `<th` | 15.3125 / -2.301685 / 2 | 15.7500 / -1.602257 / 2 | same rank, different distribution |
| ` required` | 10.8750 / -6.588740 / 21 | 15.1875 / -2.152102 / 3 | large distribution change |
| `move` | 27.1250 / -0.000218 / 1 | 26.6250 / -0.000197 / 1 | action control remains clean |

Top-token behavior for the outlier P1:

| Path | top-1 token | top-1 logp | target ` required` rank/logp |
|---|---|---:|---:|
| HF incremental B | ` sequence` | -0.213740 | 21 / -6.588740 |
| vLLM fixed replay | ` should` | -1.589603 | 3 / -2.152102 |

This proves the residual mismatch is not a vLLM API logprob extraction issue.
The sampled-token raw logit itself and the vocabulary distribution are different.

### D0.8.6 Hypotheses Status

| Hypothesis | Status | Evidence |
|---|---|---|
| nondeterministic vLLM replay | ruled out | repeated fixed request gave identical selected logprobs |
| renderer/env/PPO/optimizer caused C | ruled out by D0.7b | direct fixed vLLM replay reproduced production-like C |
| prefix cache caused mismatch | strongly downgraded by D0.7b | prefix cache ON/OFF had zero selected-token delta |
| chunked prefill caused mismatch | downgraded | true 1024+490 chunk run kept P1 vLLM-like |
| sequence expansion mismatch | ruled out | expanded length, image span, response positions match |
| position-id mismatch | ruled out for checked case | both paths use contiguous valid-token positions 0..1513 |
| visual tower implementation mismatch | confirmed and repaired | Transformers SigLIP visual embeddings differed; Cambrian native SigLIP restores exact input parity |
| logprob extraction-only bug | ruled out | raw logits already differ |
| matched TORCH_SDPA backend test | not available | installed vLLM V1 CUDA path falls through to FlashAttention |
| remaining root | unresolved | with exact layer-0 input, divergence is inside vLLM Qwen2 transformer/FlashAttention runtime or vLLM model execution semantics |

### D0.8.7 Gate

D0.8 found and fixed one real plumbing bug: vLLM visual tower implementation
mismatch.  However, after the repair:

1. FSDP/HF and vLLM have matching fixed request sequence layout.
2. FSDP/HF and vLLM have matching position semantics.
3. FSDP/HF and vLLM have exact checked layer-0 input embedding parity.
4. The healthy action token remains clean.
5. The large free-form think-token outlier remains a raw-logit distribution
   mismatch.

Therefore the D0.8 gate is:

```text
FAIL — vLLM prefill/runtime root cause remains
```

D1 remains blocked.  No post-update resync test was run because pre-update
runtime parity is still not explained.

## P. D0.7b Fixed-Request Production vLLM Replay

Date: 2026-08-15

Scope remained D0.7b only. This run did not start renderer, ActiveSpatial env,
Ray trainer, PPO, optimizer, critic, navigation evaluation, or any C8/B5 rerun.
It directly replayed the fixed D0.6 prompt/image/response artifacts into the
production Cambrian vLLM wrapper:

```text
scripts/d0_7b_fixed_vllm_replay.py
```

### Fixed inputs

The D0.6 manifest had expanded FSDP prompt ids, where one visual placeholder was
represented as `-200 x 756`. For vLLM replay this was converted back to the
compact form by removing left padding and collapsing each visual block to token
id `151665`.

Recovered exact image PNGs:

```text
sample 3: exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/pid216624_sample_000_pre_processor.png
sample 5: exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/pid216636_sample_000_pre_processor.png
```

Prompt geometry:

| Sample | compact prompt len | expanded prompt len | compact image token position |
|---|---:|---:|---:|
| 3 | 657 | 1412 | 470 |
| 5 | 659 | 1414 | 470 |

Important indexing correction: vLLM `prompt_logprobs` for this multimodal request
must be read at:

```text
expanded_prompt_len + response_idx
```

not at `compact_prompt_len + response_idx`. Reading the compact index lands
inside the expanded visual/prompt region and produces bogus forced-scoring
records.

### Runtime

The run used node `10.119.28.231`, TP=2, bf16, eager mode, temperature 0.6 for
generation and temperature 1.0 for forced scoring. FlashInfer sampling was
disabled, so vLLM reported the PyTorch-native top-p/top-k sampler.

Caveat: even with `VLLM_ATTENTION_BACKEND=TORCH_SDPA`, the vLLM V1 workers
reported:

```text
Using Flash Attention backend on V1 engine.
```

Therefore this replay is a fixed-request production-wrapper replay, but its
attention backend is not confirmed to match the earlier D0.6 production log that
reported TORCH_SDPA.

### C-prime versus historical C

Forced scoring on the same fixed response sequence reproduced historical vLLM C
much more closely than HF B on the selected diagnostic tokens:

```text
mean abs(direct forced vLLM - historical C) = 0.04274759140521796
max  abs(direct forced vLLM - historical C) = 0.17248773574829102
mean abs(direct forced vLLM - historical B) = 0.6390546442972581
```

Direct stochastic generation did not reproduce the historical sampled response
sequence beyond an early prefix, so generation replay is not used as the decisive
fixed-sequence evidence. The decisive evidence is vLLM forced scoring on the
historical token sequence.

Key token comparison:

| Sample | idx | token | HF B | historical C | direct vLLM forced | vLLM rank |
|---:|---:|---|---:|---:|---:|---:|
| 3 | 0 | `<` | -0.329762 | -0.489088 | -0.474437 | 1 |
| 3 | 17 | ` decorated` | -0.972311 | -0.647216 | -0.678113 | 1 |
| 3 | 117 | `move` | -0.000359 | -0.000329 | -0.000346 | 1 |
| 5 | 0 | `<th` | -2.301748 | -1.684820 | -1.727953 | 2 |
| 5 | 73 | ` required` | -6.588775 | -2.127776 | -2.082076 | 3 |
| 5 | 87 | `move` | -0.000218 | -0.000220 | -0.000241 | 1 |

Conclusion: renderer/env are not required to reproduce the main B-vs-C mismatch.
The residual appears when the fixed request is sent directly to the production
Cambrian vLLM wrapper.

### Top mismatch logits

For sample 5 idx 73 token ` required`, HF B and direct vLLM do not merely assign
a different logprob to the sampled token; their distributions/rankings differ.

HF B from D0.6:

```text
rank 1: " sequence"  -0.213740
rank 2: " '"         -3.338740
rank 3: " should"    -3.526240
rank 4: " to"        -3.838740
rank 5: " will"      -4.276240
target " required":  -6.588775, rank 21
```

Direct vLLM forced scoring:

```text
rank 1: " should"    -1.582076
rank 2: " I"         -1.707076
rank 3: " required"  -2.082076
rank 4: " will"      -2.332076
rank 5: " needed"    -2.644576
```

This confirms a real next-token distribution/ranking mismatch, not a simple
sampled-token logprob extraction error.

### Chunked prefill and prefix cache ablations

Artifacts:

```text
docs/diagnosis/d0_7b_direct_replay_parity.json
docs/diagnosis/d0_7b_direct_replay_parity_off_on.json
docs/diagnosis/d0_7b_direct_replay_parity_on_off.json
docs/diagnosis/d0_7b_hf_vs_vllm_topk.json
docs/diagnosis/d0_7b_hf_vs_vllm_topk_off_on.json
docs/diagnosis/d0_7b_hf_vs_vllm_topk_on_off.json
docs/diagnosis/d0_7b_vllm_raw_topk.json
docs/diagnosis/d0_7b_vllm_raw_topk_off_on.json
docs/diagnosis/d0_7b_vllm_raw_topk_on_off.json
docs/diagnosis/d0_7b_fixed_replay_summary.json
```

Chunked-prefill OFF attempt:

```text
requested enable_chunked_prefill=False
observed chunked_prefill_enabled=True
forced target logprobs: identical to ON for all selected tokens
```

This is not a valid OFF ablation in the current vLLM V1 path; the engine forced
chunked prefill enabled.

Prefix-cache OFF:

```text
observed enable_prefix_caching=False
forced target logprobs: identical to prefix-cache ON for all selected tokens
```

Prefix caching is therefore ruled out for the selected fixed-request mismatch.

### D0.7b decision

Decision: **FAIL for D0.7b; overall D0 remains FAIL**.

What is now resolved:

1. Fixed-request direct vLLM forced scoring reproduces the historical vLLM-side
   mismatch on the top outlier.
2. Renderer, ActiveSpatial env, reward code, PPO objective, optimizer, critic,
   and action masking are not necessary to produce the mismatch.
3. Prefix caching is not the cause on these fixed requests.
4. Action token parity remains clean on the selected action-name token.

What remains unresolved:

1. The root cause is still inside the vLLM model/runtime computation path or
   backend, because HF full/incremental remain close while vLLM forced scoring is
   distributionally different.
2. The requested chunked-prefill OFF ablation could not be made effective through
   the current vLLM V1 constructor path.
3. The direct replay observed Flash Attention despite requesting TORCH_SDPA, so a
   backend-matched probe remains the next bounded diagnostic if D0 continues.

D1 remains blocked.

## D0.7 Production vLLM Runtime Divergence

Date: 2026-08-14

Scope:

- Continue D0 only.
- Do not enter D1.
- Do not start full RL training.
- Do not change LR/KL/entropy/reward/prompt/action mask/task sampling/PPO objective.

### First divergent token

Artifact:

```text
docs/diagnosis/d0_7_first_divergent_token.json
```

Using the existing D0.6 token matrix and threshold `abs(B-C) > 0.1`, both
selected samples already diverge at the first response token:

| sample | selection | idx | token | token id | B HF incremental | C production vLLM | B-C |
|---:|---|---:|---|---:|---:|---:|---:|
| 3 | normal sample | 0 | `<` | 27 | -0.329762 | -0.489088 | +0.159326 |
| 5 | top mismatch sample | 0 | `<th` | 13708 | -2.301748 | -1.684820 | -0.616929 |

Interpretation: the residual B-vs-C mismatch is present at the first generated
response token. This downgrades a simple cumulative KV-cache numerical drift
hypothesis and points earlier in the path: production prefill, multimodal prefix,
position handling, vLLM runtime/backend, or logprob extraction.

### Production vLLM top-logprob capture hook

Code was instrumented in:

```text
verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
```

The hook is controlled by:

```text
VAGEN_D0_7_DIR
VAGEN_D0_7_TOP_LOGPROBS
```

Behavior:

- If rollout requests token logprobs, `VAGEN_D0_7_TOP_LOGPROBS=N` asks vLLM for
  top-N token logprobs instead of only sampled-token logprob.
- The async server writes `production_vllm_top_logprobs.jsonl` containing
  request id, effective sampling params, token ids, sampled raw logprob,
  sampled rank, decoded sampled token, and top logprobs per generated position.

Static vLLM sampler inspection found that vLLM V1 default
`logprobs_mode="raw_logprobs"` computes token logprobs from the raw logits before
temperature/top-p processors. Therefore, once the hook runs, equality between
`stored_token_output_logprob` and the sampled entry in `top_logprobs` will test
the returned-logprob extraction path. It does not by itself expose raw lm-head
logits or layer hidden states.

### Renderer preflight and production smoke

Artifacts:

```text
docs/diagnosis/d0_7_renderer_preflight.json
docs/diagnosis/d0_7_production_smoke_attempt.json
```

The bounded production smoke on `10.119.28.231` reached production vLLM server
initialization with:

```text
CambrianVLLMForCausalLM
dtype=torch.bfloat16
VLLM_ATTENTION_BACKEND=TORCH_SDPA
tensor_parallel_size=2
enforce_eager=True
enable_prefix_caching=True
enable_chunked_prefill=True
FlashInfer sampler disabled; PyTorch-native top-p/top-k sampler
```

It failed before any vLLM generation because ActiveSpatial reset could not render
the first observation:

```text
RuntimeError: ActiveSpatial rendering failed:
  Server error '500 Internal Server Error' for url 'http://127.0.0.1:18767/render'
```

Renderer service logs showed the real render failure:

```text
gsplat: No CUDA toolkit found. gsplat will be disabled.
AttributeError: 'NoneType' object has no attribute 'CameraModelType'
```

After restarting the renderer with `CUDA_HOME`, the next preflight exposed a
second environment issue:

```text
gcc: No such file or directory
nvcc fatal: Failed to preprocess host compiler properties
```

After adding conda `CC`/`CXX`, a single-job gsplat JIT preflight remained running
for several minutes and was interrupted to avoid leaving an unbounded compile
task. No successful real render artifact was produced.

Cleanup:

- Stopped exact renderer PIDs created for this diagnosis.
- GPU memory after cleanup was approximately 4 MiB used on each GPU.

### D0.7 executed/missing matrix

| Required D0.7 item | Status |
|---|---|
| reuse D0.6 fixed samples | partially done from existing D0.6 token CSV |
| first divergent token analysis | **DONE**; divergence starts at response token 0 for samples 3 and 5 |
| production vLLM top-logprob hook | **DONE**; code path instrumented and py_compile passed |
| production vLLM top-logprob artifact | not produced; render failed before generation |
| raw lm-head logits before sampler | not captured |
| raw logits vs returned logprob | not verified dynamically |
| first divergent layer localization | not executed |
| prefix-cache/chunked-prefill/backend ablation | not executed |
| top mismatch token logits explanation | only D0.6 HF/vLLM logprob/rank evidence available; no new vLLM top-logprob artifact |
| final B-vs-C parity | not executed |
| post-sync parity | not executed |

### D0.7 gate

Decision: **INCONCLUSIVE for D0.7 continuation; overall D0 remains FAIL**.

Why not PASS:

1. Raw production vLLM logits/logprobs were not dynamically captured.
2. The first divergent layer/source was not localized.
3. Final B-vs-C parity was not re-run.
4. Post-sync parity was not run.

Why not a new model-policy FAIL:

1. This attempt failed before production vLLM generation.
2. The new evidence is an infrastructure/rendering block, not a newly measured
   policy-forward mismatch.

Current best answer to the D0.7 question:

- Existing D0.6 evidence remains: `A ~= B` and `B != C`, so residual mismatch
  still points to the production vLLM runtime path rather than FSDP full-sequence
  scoring or weight sync.
- D0.7 strengthens that this is not simple cumulative KV drift, because the
  historical B-vs-C mismatch is already visible at response token 0.
- The exact mechanism remains unproven because renderer/gsplat preflight blocked
  the production top-logprob/logit capture.

D1 remains blocked.

## P. D0.6b on 10.119.28.231: Renderer Preflight, Fixed Sequence, A/B/C Matrix

Date: 2026-08-14

Scope remained D0.6b only. No D1 experiment, no C8/B5 rerun, no navigation
evaluation, and no full RL training were started. The production smoke used
`VAGEN_D0_6_STOP_AFTER_LOGPROB=1`, so it stopped immediately after rollout
logprobs and FSDP old-logprob capture, before ref/critic/value/advantage or any
optimizer step.

### Gate 0 renderer preflight

The independent HTTP renderer on `127.0.0.1:18767` passed health, direct render,
environment reset, and one-step probes. Artifacts:

```text
docs/diagnosis/d0_6b_renderer_preflight.json
docs/diagnosis/d0_6b_renderer_preflight_render.json
docs/diagnosis/d0_6b_renderer_preflight_env.json
docs/diagnosis/d0_6b_renderer_preflight_min_render.png
```

Direct render used scene `0003_839989`, produced `256x256x3` RGB, and sha256:

```text
05613f87416a55a07410b47de0c99e54765e1556d0122427808c0a674f47b0e4
```

### Fixed production sequence capture

The bounded production smoke completed successfully and generated:

```text
docs/diagnosis/d0_6_fixed_sequence_manifest.json
exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/d0_6/vllm_raw_logprobs.jsonl
```

Captured samples:

| sample | selection | prompt tensor len | active response tokens | production A-vs-C mean abs delta | max abs delta |
|---:|---|---:|---:|---:|---:|
| 3 | normal_sample | 2803 | 127 | 0.052500 | 0.356804 |
| 5 | top_mismatch_sample | 2803 | 100 | 0.121047 | 4.277603 |

The server-side compact prompt lengths were 657 and 659 before vLLM multimodal
expansion. This is consistent with the FSDP expanded prompt: after removing left
padding and expanding one compact image token to 756 visual positions, response
positions start at 1412/1414. Therefore the compact-vs-expanded length difference
is not by itself a parity failure.

### Weight sync regression status

No new weight sync regression was run in D0.6b because D0.6 had already compared
the real production actor -> vLLM sync at tensor level. The current evidence still
uses the existing D0.6 result:

```text
docs/diagnosis/d0_6_weight_equality.json
```

Policy-relevant sampled tensors matched exactly after casting FSDP source tensors
to the vLLM bf16 runtime dtype:

```text
bad tensors after source cast to vLLM dtype: 0
max abs diff after source cast to target dtype: 0.0
```

### Vision and multimodal equality

The standalone paired probe now confirms the repaired Cambrian visual path:

```text
docs/diagnosis/d0_6_vision_feature_equality.json
```

For both fixed samples:

| Comparison | result |
|---|---|
| vLLM preprocess vs FSDP processor | max abs diff 0.0 |
| effective selected hidden vs `encode_images()` | max abs diff 0.0 |
| MIV helper consistency | max abs diff 0.0 |
| final 756 visual embeddings vs FSDP scatter | max abs diff 0.0 |

The artifact also preserves the old config-layer mismatch evidence:

```text
config mm_vision_select_layer=-2 on a 26-layer truncated tower:
sample 3 mean abs hidden diff 0.644266, max 8.5
sample 5 mean abs hidden diff 0.635278, max 8.125
```

The repaired effective layer is `-1` for the vLLM wrapper's 26-layer tower, matching
the FSDP `LOVSiglipVisionTower` behavior.

### Forced scoring matrix

Artifacts:

```text
docs/diagnosis/d0_6_scoring_matrix.json
docs/diagnosis/d0_6_token_level_matrix.csv
docs/diagnosis/d0_6_top_logits_cases.json
```

Paths:

```text
A = production FSDP/HF full-sequence old_log_prob
A' = standalone HF full-sequence reconstruction
B = standalone HF teacher-forced incremental KV-cache scoring
C = production vLLM incremental generation sampled-token logprob
D = vLLM native forced scoring, not available in this run
```

Aggregate active-token matrix:

| Pair / token type | n | mean abs delta | median abs delta | p95 abs delta | max abs delta | ratio median | frac \|ratio-1\|>5% | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A vs A' all | 227 | 0.000001 | 0.000000 | 0.000001 | 0.000002 | 1.000000 | 0.000 | 1.000000 | 1.000000 |
| A vs B all | 227 | 0.018991 | 0.006441 | 0.075803 | 0.183396 | 0.999967 | 0.119 | 0.998955 | 0.999187 |
| A vs C all | 227 | 0.082697 | 0.027880 | 0.286521 | 4.277603 | 0.993883 | 0.370 | 0.920964 | 0.993606 |
| B vs C all | 227 | 0.081520 | 0.030681 | 0.223010 | 4.460999 | 0.992358 | 0.361 | 0.916214 | 0.993466 |
| A vs C think_text | 186 | 0.090800 | 0.036937 | 0.286521 | 4.277603 | 0.988407 | 0.403 | 0.911429 | 0.992865 |
| A vs C action_tag | 12 | 0.001909 | 0.000764 | 0.006136 | 0.007420 | 1.000003 | 0.000 | 0.983081 | 0.951049 |
| A vs C action_name | 15 | 0.052591 | 0.004462 | 0.179664 | 0.386397 | 0.999621 | 0.333 | 0.918498 | 0.992857 |

Interpretation by requested cases: this is **Case C**.

```text
A ~= B
B != C
```

FSDP full-sequence scoring and HF incremental scoring are close. The remaining
large residual is in production vLLM generation/logprob behavior, concentrated in
free-form think tokens, with action tags clean and action names mostly clean but
not perfectly clean.

### Conditional ablations

Batch=1 vs production batch was not run because A vs B does not indicate a
production FSDP full-sequence/padding failure.

Train/eval/dropout audit:

```text
standalone HF model.training = false
attention_dropout = 0.0
hidden_dropout = null
```

Dropout is ruled out for the observed parity failure.

Position drift remains low priority:

| sample | A vs C corr(position, abs delta) | B vs C corr(position, abs delta) |
|---:|---:|---:|
| 3 | -0.1134 | -0.0694 |
| 5 | 0.1076 | 0.1124 |

Legacy layer ablation:

```text
docs/diagnosis/d0_6_legacy_layer_ablation.json
```

Forcing standalone HF `encode_images()` back to config layer `-2` did not make it
match production C:

| Pair / token type | n | mean abs delta | p95 abs delta | max abs delta | Pearson |
|---|---:|---:|---:|---:|---:|
| legacy-layer full vs C all | 227 | 0.115573 | 0.366278 | 4.082544 | 0.911797 |
| A vs legacy-layer full all | 227 | 0.075728 | 0.262778 | 1.709991 | 0.976369 |

The top sample-5 outlier `" required"` remains far from C under both repaired and
legacy standalone scoring:

```text
A full: -6.405378
B incremental: -6.588775
legacy-layer full: -6.210320
C production vLLM: -2.127776
```

Therefore the remaining C mismatch is not explained by the already fixed
effective-layer issue.

### vLLM native forced scoring availability

Artifact:

```text
docs/diagnosis/d0_6_vllm_forced_scoring.json
```

D was not produced. Two bounded offline vLLM prompt-logprob attempts failed before
scoring:

1. default offline vLLM selected FlashInfer sampler during profile and failed
   because `nvcc` is absent;
2. with `VLLM_USE_FLASHINFER_SAMPLER=0`, sampler warmup passed, but vLLM warmup
   imported `deep_gemm` and failed because `CUDA_HOME` is absent.

No D numbers are fabricated.

### Top mismatch logits

For the largest outlier:

```text
sample 5, response idx 73, decoded token " required"
A full logprob = -6.405378, rank 19
B incremental logprob = -6.588775, rank 21
C production vLLM logprob = -2.127776, vLLM rank 3
```

A/B top candidates were stable and semantically different:

```text
top A/B token: " sequence"
other high tokens: " '", " should", " to", " will", " order"
```

This is not a local A/B normalization bug. Production vLLM assigned a much higher
rank/probability to the sampled token in that state than the HF policy did.

### PPO impact status

The bounded capture stopped before advantage computation by design. Therefore the
requested signed PPO loss-impact audit cannot be computed from this D0.6b batch
without another bounded capture that includes advantages but still avoids optimizer
step. No PPO-impact artifact was generated.

### Repair made in this continuation

`vagen/models/cambrian_vllm.py` now maps the checkpoint config's
`mm_vision_select_layer=-2` to effective `-1` when the vLLM wrapper has the
already-truncated 26-layer SigLIP tower. This makes the standalone paired visual
path exactly match FSDP for processed pixels, selected hidden layer, projected
features, MIV/newline construction, and final 756 visual embeddings.

Additional bounded diagnostic scripts added:

```text
scripts/d0_6_hf_incremental_probe.py
scripts/d0_6_legacy_layer_ablation.py
scripts/d0_6_vllm_forced_scoring.py
```

### Gate decision after D0.6b

Decision: **FAIL — root cause remains in production vLLM path**.

Resolved:

1. Renderer infrastructure works on `10.119.28.231`.
2. Fixed production sequences were captured before optimizer step.
3. FSDP full vs standalone HF full reconstruction is exact to about 1e-6.
4. HF full vs HF incremental is close enough to rule out the main FSDP
   full-sequence recompute hypothesis.
5. Standalone paired vision/multimodal construction is numerically identical after
   the effective-layer repair.
6. The old `-2` layer hypothesis no longer explains the production residual.
7. Action tags are clean; action names are much cleaner than think text but still
   have small residual outliers.

Still failing / missing:

1. Production C vLLM logprobs remain systematically different from A/B on
   free-form think tokens.
2. The largest outlier is a true distribution/ranking difference in production C,
   not an A/B logits normalization issue.
3. vLLM native forced scoring D could not be obtained because offline vLLM
   initialization is blocked by missing CUDA toolchain components.
4. PPO signed loss impact and post-update sync parity were not run.

D1 remains blocked.

## O. D0.6 Continuation on 10.119.17.247: Production Weight Equality A-Gate

Date: 2026-08-13

Scope remained D0.6 only. The run used the bounded diagnostic script:

```text
examples/train/active_spatial/experiments/d0_6_b5_forced_scoring_matrix.sh
```

No D1 experiment, no C8/B5 rerun, no navigation evaluation, and no full RL
training were started.

### Node and run status

The replacement GPU node was reachable:

```text
hostname = pt-4cd01d05323e4f9d87b9ad1338f73f2c-worker-0
GPU = 8 x NVIDIA H800
```

The production D0.6 run reached the real async rollout weight sync path and wrote
the tensor-level equality artifacts. After that, the run failed during ActiveSpatial
environment reset because the configured remote renderer endpoint was not accepting
connections:

```text
RuntimeError: ActiveSpatial rendering failed: All connection attempts failed
```

Therefore this continuation produced weight equality evidence only. It did not
produce a new `d0_6_fixed_sequence_manifest.json`, paired vision feature equality,
or an A/B/C/D forced-scoring matrix.

### Instrumentation fixes during this continuation

The first GPU attempt exposed an instrumentation bug in the comparator: the source
tensor was on CPU while the target tensor was still CUDA. `_d0_6_pair_stats()` was
fixed to compare CPU tensors.

The second attempt exposed an overly strict equality criterion: FSDP source tensors
were gathered as fp32, while vLLM production tensors are stored as bf16. This made
raw exact equality fail for all tensors even though the differences matched bf16
rounding. Those strict fp32-vs-bf16 artifacts were preserved under:

```text
exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/d0_6/failed_strict_dtype_only_20260813_0400/
```

The comparator was then updated to record both:

```text
strict_fp32_vs_bf16 equality
source-cast-to-target-dtype equality
```

The D0.6 abort gate now fails only when the FSDP tensor cast to the actual vLLM
runtime dtype differs from the vLLM tensor.

### Production tensor equality result

New artifacts:

```text
exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/d0_6/d0_6_weight_equality_rank{0..7}_tp{0..1}_pid*.json
docs/diagnosis/d0_6_weight_equality.json
```

Result:

```text
rank files: 16 total = two bounded attempts x 8 ranks
compared tensors per rank: 27
missing tensors: 0 on every rank
bad tensors after source cast to vLLM dtype: 0 on every rank
strict fp32-vs-bf16 exact equality: false, expected dtype representation difference
source cast to vLLM bf16 exact equality: true on every rank
max raw fp32-vs-bf16 abs diff: 0.013660430908203125
max raw fp32-vs-bf16 mean abs diff: 0.0015766506548970938
max abs diff after source cast to target dtype: 0.0
```

Interpretation: for the sampled policy-relevant tensors, the real production
FSDP actor -> vLLM sync path is weight-content correct at the vLLM runtime dtype.
The previous strict failure was not evidence of stale or wrong vLLM weights; it was
fp32 master/source representation compared against bf16 runtime storage.

Covered groups:

| Group | Tensor coverage |
|---|---|
| vision tower | encoder layer 0 q-proj, layer 13 q-proj, layer 25 q-proj, post-layernorm |
| mm projector | `mm_projector.0/2` weight and bias |
| multimodal | `image_newline` |
| LLM embeddings/head | TP shard of `embed_tokens`, TP shard of `lm_head` |
| LLM attention | layers 0/14/27 q/k/v packed TP shard, o-proj column shard |
| LLM MLP | layers 0/14/27 gate/up packed TP shard, down-proj column shard |
| LLM norm | final norm |

### Runtime path confirmations from this run

The logs confirmed that the production paths were active:

```text
FSDP actor: CambrianForCausalLMAdapter
vLLM rollout: vagen.models.cambrian_vllm:CambrianVLLMForCausalLM
vision tower: google/siglip2-so400m-patch14-384-interp729
mm_vision_select_layer: -2
miv_token_len: 64
attention_dropout: 0.0
FSDP config dtype: bfloat16
FSDP model_dtype: fp32
vLLM dtype: bfloat16
vLLM attention/backend: V1 eager, prefix caching enabled, chunked prefill enabled, kv_cache_dtype=auto
rollout sampling temperature: 0.6
logprob temperature: 1.0
```

The logs also emitted the slow image processor warning. For this run it means the
checkpoint-saved slow processor was used; it remains a reproducibility item for
future transformer versions, not an observed D0.6 mismatch.

### Renderer fallback attempt

Because all checked HTTP renderer endpoints were unavailable:

```text
127.0.0.1:{8767,8777,18767} -> unavailable
10.119.30.223:{8767,8777,18767} -> unavailable
10.119.18.163:{8767,8777,18767} -> unavailable
10.119.21.67:{8767,8777,18767} -> unavailable
```

a second bounded D0.6 run was attempted with:

```text
D0_6_RENDER_MODE=local
```

Local prerequisites existed:

```text
/mnt/umm/users/yinbaiqiao/InteriorGS exists
from gsplat.rendering import rasterization succeeds
```

This run again reached production vLLM initialization and wrote a second complete
set of weight equality files. These also passed source-cast-to-vLLM-dtype exact
equality on all 8 ranks.

It then failed during ActiveSpatial reset because local rendering inside the agent
workers could not use CUDA and the concurrent gsplat lazy extension path failed:

```text
[GaussianRenderer] WARNING: CUDA not available, using CPU (very slow!)
ImportError: cannot import name 'csrc' from 'gsplat'
RuntimeError: ActiveSpatial rendering failed:
  [Errno 2] No such file or directory:
  /mnt/umm/users/yinbaiqiao/.cache/torch_extensions_yinbaiqiao/gsplat_cuda
```

This is infrastructure/rendering evidence only. It does not change the policy
parity conclusion, and it did not produce valid fixed samples.

### D0.6 matrix status after this continuation

| Required D0.6 item | Status |
|---|---|
| fixed D0.5/D0.6 samples with exact production arrays | not produced in this continuation; remote renderer unavailable and local renderer failed before rollout samples completed |
| tensor-level FSDP/vLLM weight equality | **PASS at vLLM runtime dtype in two bounded attempts** |
| paired vision/multimodal feature equality | not executed |
| A. FSDP full-sequence scoring | not newly executed |
| B. HF incremental teacher-forced scoring | not executed |
| C. vLLM production incremental logprob | not newly produced; older D0.5/D0.6-like artifact still available |
| D. vLLM fixed-sequence forced scoring | not executed / availability not verified |
| A/B/C/D pairwise matrix | not executed |
| batch=1 ablation | not executed |
| train/eval/dropout audit | partially static only; dropout is 0.0 in printed config |
| top-k logits analysis | not executed |
| PPO signed loss impact | not executed |
| post-update/resync parity | not executed |

### Gate decision after this continuation

Decision: **INCONCLUSIVE for D0.6 continuation; overall D0 remains FAIL**.

What is now resolved:

1. Production actor -> vLLM policy-relevant weights are synchronized correctly at
   vLLM bf16 runtime dtype.
2. Historical residual logprob mismatch is no longer supported by a simple
   "wrong synced weights" hypothesis.

What remains missing:

1. Paired vision/multimodal feature equality for the same real image.
2. Fixed-sequence A/B/C/D scoring matrix.
3. Top mismatch token logits analysis.
4. PPO loss-impact audit for any residual all-response drift.
5. Post-update resync parity, which should only run after pre-update parity is
   explained.

D1 remains blocked.

## Latest Status After D0.8

Latest gate: **FAIL — vLLM prefill/runtime root cause remains**.

The D0.8 section above is the newest completed diagnosis.  It repaired a real
vLLM-side visual tower implementation mismatch by switching the wrapper from
Transformers `SiglipVisionModel` to Cambrian native `SigLipVisionModel` when
`CAMBRIAN_SRC` is available.  After that repair, the fixed sample's checked
layer-0 input embeddings, sequence layout, visual span, and position IDs match
HF/FSDP exactly, but vLLM still produces different raw pre-softmax logits for the
large think-token outlier ` required`.

D1 remains blocked.  No renderer, env rollout, PPO/Ray training, optimizer step,
C8/B5 rerun, navigation evaluation, or post-update resync test was started.

## D0.9 — First Divergent Transformer Layer

Date: 2026-08-15

Gate after D0.9: **FAIL — divergence localized but not fixed**

D0.9 stayed within the renderer-free fixed-request diagnosis.  It did not start
renderer, ActiveSpatial, GymAgentLoop, PPO/Ray training, critic, optimizer step,
C8/B5 rerun, navigation evaluation, D1, or post-update resync.

Fixed case:

| Label | Response idx | Token | Prediction position | Target position |
|---|---:|---|---:|---:|
| P0 | 0 | `<th` | 1413 | 1414 |
| P1 | 73 | ` required` | 1486 | 1487 |
| P2 | 87 | `move` | 1500 | 1501 |

Artifacts:

- `docs/diagnosis/d0_9_layer_divergence.json`
- `docs/diagnosis/d0_9_first_divergent_layer.json`
- `docs/diagnosis/d0_9_first_layer_breakdown.json`
- `docs/diagnosis/d0_9_attention_metadata.json`
- `docs/diagnosis/d0_9_before_after.json`
- `docs/diagnosis/d0_9_hf_layer_probe_coarse.json`
- `docs/diagnosis/d0_9_hf_layer_probe_breakdown_l0.json`
- `docs/diagnosis/d0_9_vllm_probe_coarse_no_compile.json`
- `docs/diagnosis/d0_9_vllm_probe_breakdown_l0.json`
- raw captures under `docs/diagnosis/d0_9_runs/`

Runtime notes:

```text
vLLM 0.11.0
V1
bf16
TP=2
actual attention backend: Flash Attention on V1
```

The node lacked `nvcc`, CUDA_HOME for DeepGEMM warmup, and a C compiler for a
TorchInductor helper.  To get the bounded fixed replay to execute, the following
non-policy warmup/compile blockers were disabled:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_SKIP_DEEP_GEMM_WARMUP=1
VLLM_USE_DEEP_GEMM=0
TORCH_COMPILE_DISABLE=1
```

These changes do not switch the vLLM attention backend; logs still showed
`Using Flash Attention backend on V1 engine`.

### D0.9.1 Layer Search

Coarse comparison used the same fixed prompt/image/response and captured only
P0/P1/P2 prediction-position hidden states.

Result:

```text
last matching layer = layer0_input
first divergent layer = layer_0_output
```

Layer-0 input remained exact:

| Position | max abs diff | mean abs diff | cosine | L2 |
|---:|---:|---:|---:|---:|
| 1413 | 0.0 | 0.0 | ~1.0 | 0.0 |
| 1486 | 0.0 | 0.0 | ~1.0 | 0.0 |
| 1500 | 0.0 | 0.0 | ~1.0 | 0.0 |

Layer-0 output diverged:

| Position | max abs diff | mean abs diff | cosine | L2 |
|---:|---:|---:|---:|---:|
| 1413 | 0.046875 | 0.005662 | 0.999232 | 0.429433 |
| 1486 | 1.195312 | 0.076747 | 0.923021 | 6.368046 |
| 1500 | 0.626953 | 0.052987 | 0.939936 | 4.279336 |

### D0.9.2 Layer-0 Breakdown

First divergent tensor inside layer 0:

```text
attention_pre_o
```

Summary across P0/P1/P2:

| Tensor | max mean abs diff | max abs diff | min cosine | Status |
|---|---:|---:|---:|---|
| input RMSNorm | 0.0 | 0.0 | ~1.0 | match |
| Q | 0.0 | 0.0 | 0.9999996 | match |
| K | 1.63e-9 | 8.34e-7 | 0.9999999 | match |
| V | 3.73e-9 | 1.91e-6 | 0.9999999 | match |
| Q after RoPE | 0.0 | 0.0 | 1.0 | match |
| K after RoPE | 0.0 | 0.0 | 0.9999999 | match |
| attention pre-o | 0.015890 | 0.707153 | 0.904287 | **diverge** |
| attention output after o-proj | 0.032804 | 0.496094 | 0.941008 | diverge |
| post-attention residual | 0.032806 | 0.496094 | 0.942941 | diverge |
| post-attention RMSNorm | 0.061212 | 0.814453 | 0.916276 | diverge |
| MLP output | 0.063677 | 0.714844 | 0.885536 | diverge |
| post-MLP residual | 0.076747 | 1.195312 | 0.923021 | diverge |

Per-position attention aggregation mismatch:

| Position | Tensor | max abs diff | mean abs diff | cosine |
|---:|---|---:|---:|---:|
| 1413 | attention pre-o | 0.055176 | 0.003094 | 0.998561 |
| 1486 | attention pre-o | 0.660543 | 0.015890 | 0.904287 |
| 1500 | attention pre-o | 0.707153 | 0.010219 | 0.936030 |

Interpretation:

1. RMSNorm is ruled out for the first divergence.
2. Q/K/V linear and TP execution for the current query position are ruled out.
3. RoPE position semantics for the current query/key are ruled out.
4. The first observed mismatch happens when attention aggregates over visible
   keys/values, before the output projection.

This points to:

```text
attention backend / attention mask / attention metadata / KV layout or visible
context aggregation
```

It does not point to MLP or LM head as the primary source.

### D0.9.3 Attention Metadata

For the fixed causal sequence, recorded/inferred visibility is:

| Label | Prediction position | RoPE position | Visible key range | Visible keys |
|---|---:|---:|---|---:|
| P0 | 1413 | 1413 | [0, 1413] | 1414 |
| P1 | 1486 | 1486 | [0, 1486] | 1487 |
| P2 | 1500 | 1500 | [0, 1500] | 1501 |

HF incremental and vLLM prefill both use the same absolute prediction positions
for these tokens.  Because current-position Q/K/V and RoPE match, the remaining
attention-side suspects are the actual aggregation over the causal prefix:
FlashAttention behavior, attention metadata, mask interpretation, KV/cache layout,
or non-target prefix K/V handling.

### D0.9.4 Before / After

No repair was applied in D0.9, so before/after is unchanged:

| Token | HF logp/rank/raw | vLLM logp/rank | Status |
|---|---:|---:|---|
| `<th` | -2.301685 / 2 / 15.3125 | -1.602315 / 2 | mismatch |
| ` required` | -6.588740 / 21 / 10.8750 | -2.152150 / 3 | mismatch remains |
| `move` | -0.000218 / 1 / 27.1250 | -0.000197 / 1 | action control clean |

### D0.9.5 Gate

```text
FAIL — divergence localized but not fixed
```

D1 remains blocked.  Post-update parity was not run because pre-update
transformer/runtime parity is still not fixed.

## D0.10 — Attention Root-Cause Isolation

Scope:

```text
fixed sample 5 only
P0 prediction position = 1413
P1 prediction position = 1486 (" required")
P2 prediction position = 1500 ("move")
no renderer/env/PPO/post-sync/D1
```

Runtime:

```text
node = 10.119.28.231
vLLM = 0.11.0 V1
dtype = bf16
TP = 2
actual backend = FlashAttention
```

The same non-policy startup bypasses as D0.9 were used:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_SKIP_DEEP_GEMM_WARMUP=1
VLLM_USE_DEEP_GEMM=0
TORCH_COMPILE_DISABLE=1
```

These do not change the vLLM attention backend; worker logs again printed
`Using Flash Attention backend on V1 engine`.

### D0.10.1 Gate 0 — Instrumentation Neutrality

HF original vs HF instrumented:

| Token | raw logit delta | logprob delta | rank |
|---|---:|---:|---:|
| `<th` | 0.0 | 0.0 | 2 -> 2 |
| ` required` | 0.0 | 0.0 | 21 -> 21 |
| `move` | 0.0 | 0.0 | 1 -> 1 |

vLLM original vs vLLM instrumented:

| Token | raw logit delta | logprob delta | rank |
|---|---:|---:|---:|
| `<th` | 0.0 | 0.0 | 2 -> 2 |
| ` required` | 0.0 | 0.0 | 3 -> 3 |
| `move` | 0.0 | 0.0 | 1 -> 1 |

Result:

```text
hooks are neutral for P0/P1/P2 raw logit, logprob, and rank
```

### D0.10.2 Gate 1 — Full Prefix K/V Parity

Layer 0 full-prefix K/V was compared for P0 and P1, after RoPE for K.

Strict bit-level-style threshold:

```text
per-position mean_abs <= 1e-3
and max_abs <= 1e-2
and cosine >= 0.99999
```

Effective threshold:

```text
full-prefix q/k/v mean_abs <= 1e-6
and <= 1 strict outlier per query
```

Result:

```text
strict match = false
effective match = true
```

The only strict outlier for both P0 and P1 is:

| Prefix position | Token | Tensor | max abs | mean abs | cosine |
|---:|---|---|---:|---:|---:|
| 425 | ` A` | K after RoPE | 0.015625 | 2.478e-4 | 0.9999999 |
| 425 | ` A` | V | 0.001953 | 7.606e-5 | 0.9999997 |

Full-prefix summaries:

| Query | Prefix len | K max abs | K mean abs | V max abs | V mean abs |
|---:|---:|---:|---:|---:|---:|
| 1413 | 1414 | 0.015625 | 1.834e-7 | 0.001953 | 5.379e-8 |
| 1486 | 1487 | 0.015625 | 2.423e-7 | 0.003906 | 8.492e-8 |

Interpretation:

```text
full-prefix K/V are not bit-exact under the strict max threshold, but the
observed mismatch is a single sparse bf16-scale outlier in the text prefix.
It is far too small/sparse to explain the P1 attention_pre_o delta
(mean_abs ~0.01589, max_abs ~0.66054).
```

Therefore prefix K/V mismatch is not treated as the material root cause.

### D0.10.3 Gate 2 — Future-Suffix Invariance

Two same-length future-suffix edits were tested in vLLM with hooks enabled:

1. P0 variant: keep response idx 0 target token fixed, modify idx > 0.
2. P1 variant: keep response idx 73 target token fixed, modify idx > 73.

Invariant positions:

| Variant | Checked token | raw logit delta | logprob delta | rank |
|---|---|---:|---:|---:|
| P0 suffix | `<th` | 0.0 | 0.0 | 2 -> 2 |
| P1 suffix | `<th` | 0.0 | 0.0 | 2 -> 2 |
| P1 suffix | ` required` | 0.0 | 0.0 | 3 -> 3 |

Positions whose past context was intentionally changed did change strongly, as
expected.

Result:

```text
no future-suffix leakage was observed for P0 or P1
```

This rules down a simple future-token leakage / causal-mask failure in vLLM for
the checked positions.

### D0.10.4 Gate 3 — Reference Attention Reconstruction

Using the effective full-prefix K/V parity, standard PyTorch reference attention
was computed from the same layer-0 Q/K/V:

```text
softmax(QK^T / sqrt(head_dim)) V
```

Reference attention compared to HF/vLLM attention_pre_o:

| Query | ref vs HF mean abs | ref vs HF max abs | ref vs vLLM mean abs | ref vs vLLM max abs | Closer |
|---:|---:|---:|---:|---:|---|
| 1413 | 0.003095 | 0.055493 | 0.000065 | 0.003391 | vLLM |
| 1486 | 0.015902 | 0.660136 | 0.000081 | 0.004021 | vLLM |

HF vs vLLM attention_pre_o remains:

| Query | mean abs | max abs | cosine |
|---:|---:|---:|---:|
| 1413 | 0.003094 | 0.055176 | 0.998561 |
| 1486 | 0.015890 | 0.660543 | 0.904287 |

Interpretation:

```text
Reference attention matches vLLM, not HF.
```

So the D0.9 attention_pre_o divergence is not explained by:

```text
vLLM FlashAttention numerical/runtime kernel error
vLLM causal future leakage
vLLM current-token Q/K/V or RoPE
material prefix K/V mismatch
instrumentation side effects
```

The evidence instead points to the HF/FSDP-side attention execution path used by
this incremental probe, especially attention mask / cache / metadata semantics.

### D0.10.5 Before / After

No repair was applied in D0.10.

The key P1 mismatch remains:

| Path | Token | raw logit | logprob | rank |
|---|---|---:|---:|---:|
| HF | ` required` | 10.8750 | -6.588740 | 21 |
| vLLM | ` required` | 15.1875 | -2.152102 | 3 |

### D0.10.6 Artifacts

Generated:

- `docs/diagnosis/d0_10_hook_neutrality.json`
- `docs/diagnosis/d0_10_prefix_kv_parity.json`
- `docs/diagnosis/d0_10_suffix_invariance.json`
- `docs/diagnosis/d0_10_reference_attention.json`
- `docs/diagnosis/d0_10_before_after.json`
- `docs/diagnosis/d0_10_hf_original.json`
- `docs/diagnosis/d0_10_hf_instrumented.json`
- `docs/diagnosis/d0_10_hf_prefix_kv.json`
- `docs/diagnosis/d0_10_vllm_probe_original.json`
- `docs/diagnosis/d0_10_vllm_probe_instrumented_prefix.json`
- `docs/diagnosis/d0_10_vllm_probe_suffix_p0_after_target.json`
- `docs/diagnosis/d0_10_vllm_probe_suffix_p1_after_target.json`
- raw worker captures under `docs/diagnosis/d0_10_runs/`

### D0.10.7 Gate

```text
FAIL — attention root cause localized but not fixed
```

Root cause status:

```text
localized to HF/FSDP-side attention execution semantics in the fixed incremental
probe, most likely attention mask / cache / metadata handling.
```

D1 remains blocked.  No PPO, renderer/env, post-sync, C8/B5 rerun, or navigation
evaluation was executed.

## D0.11 — Attention Numerical Precision Closure

Scope:

```text
fixed sample 5 only
P0 prediction position = 1413
P1 prediction position = 1486 (" required")
no renderer/env/PPO/post-sync/D1
```

Question:

```text
Is the HF-vLLM attention_pre_o mismatch mainly caused by BF16 eager attention
versus FlashAttention / higher-precision fused attention numerical semantics?
```

### D0.11.1 Prepared Probe

Added:

```text
scripts/d0_11_precision_reference.py
```

The script reads existing D0.10 artifacts:

```text
docs/diagnosis/d0_10_hf_prefix_kv.json
docs/diagnosis/d0_10_runs/instrumented_prefix/worker/d0_10_pid*.jsonl
```

It computes, from the same saved layer-0 Q/K/V:

```text
R32:
  Q/K/V float32
  matmul float32
  softmax float32
  output float32

Rbf16:
  Q/K/V cast to bfloat16
  BF16 QK matmul
  scale, then cast scores to bfloat16
  softmax(dtype=float32)
  cast softmax probabilities to bfloat16
  BF16 probability-V matmul
```

The intended output is:

```text
docs/diagnosis/d0_11_precision_reference.json
```

If the precision hypothesis passes, the next bounded step would be the requested
P1-only A/B/C closure:

```text
A = HF/FSDP full-sequence attention_pre_o
B = HF incremental attention_pre_o
C = vLLM attention_pre_o
R32
Rbf16
```

### D0.11.2 Precision Reference

The first requested node (`10.119.28.231`) was unavailable, but the replacement
H800 node `10.119.28.235` was available and used for the GPU BF16 reference.

Result:

```text
precision hypothesis = PASS
```

P0/P1 comparison:

| Query | R32 closer to | R32 vs HF mean abs | R32 vs vLLM mean abs | Rbf16 closer to | Rbf16 vs HF mean abs | Rbf16 vs vLLM mean abs |
|---:|---|---:|---:|---|---:|---:|
| 1413 | vLLM | 0.003095 | 0.000065 | HF | 0.000001 | 0.003095 |
| 1486 | vLLM | 0.015902 | 0.000081 | HF | 0.000000 | 0.015890 |

Interpretation:

```text
R32 ~= vLLM
Rbf16 ~= HF
```

This directly explains the D0.9/D0.10 layer0 attention_pre_o mismatch as an
attention numerical execution-semantics difference, not a material cache/mask
or prefix-K/V mismatch.

### D0.11.3 P1 A/B/C Closure

Because the precision hypothesis passed, a P1-only A/B/C closure was run:

```text
A = HF/FSDP-style full-sequence layer0 attention_pre_o
B = HF incremental layer0 attention_pre_o
C = vLLM layer0 attention_pre_o
R32 = FP32 reference attention
Rbf16 = BF16 eager-flow reference attention
```

P1 (`" required"`, prediction position 1486):

| Pair | mean abs | max abs | cosine |
|---|---:|---:|---:|
| A HF full vs B HF incremental | 0.00000264 | 0.003906 | 0.9999997 |
| A HF full vs Rbf16 | 0.00000264 | 0.003906 | 0.9999997 |
| B HF incremental vs Rbf16 | 0.00000000 | 0.000000 | 1.0000001 |
| C vLLM vs R32 | 0.00008073 | 0.004021 | 0.9999991 |
| B HF incremental vs C vLLM | 0.01589022 | 0.660543 | 0.9042875 |
| R32 vs Rbf16 | 0.01590231 | 0.660136 | 0.9043517 |

Closure:

```text
A ~= B ~= Rbf16
C ~= R32
between-group gap is material
```

Therefore:

```text
rollout vLLM FlashAttention / higher-precision attention path and HF/FSDP eager
BF16 recompute path produce materially different attention numerics under the
same synced weights and same input.
```

### D0.11.4 Artifacts

Generated:

- `docs/diagnosis/d0_11_precision_reference.json`
- `docs/diagnosis/d0_11_abc_attention_matrix.json`
- `docs/diagnosis/d0_11_hf_full_attention_p1.json`
- `docs/diagnosis/d0_11_conclusion.json`
- `scripts/d0_11_precision_reference.py`
- `scripts/d0_11_hf_full_attention_probe.py`
- `scripts/d0_11_abc_attention_matrix.py`

Validation:

```text
D0.11 scripts py_compile passed
D0.11 JSON artifacts parse successfully
```

### D0.11.5 Gate

```text
PASS — attention numerical mismatch explained
```

Cache/mask follow-up is not required for this D0.11 question.  D1 is still not
entered in this turn; no PPO, renderer/env, post-update parity, C8/B5 rerun, or
navigation evaluation was executed.

## D0.12 PPO Logprob Path Repair

Goal:

```text
old_log_probs consumed by PPO must represent the actual vLLM rollout behavior
policy, not the numerically different HF/FSDP eager-BF16 recompute path.
```

### D0.12.1 Old Logprob Flow

Static audit:

```text
vLLM generation
  -> sampled-token logprobs are produced only when
     actor_rollout_ref.rollout.calculate_log_probs=True
  -> stored as rollout_log_probs

trainer old_log_prob stage
  -> if algorithm.rollout_correction.bypass_mode=True:
       old_log_probs = rollout_log_probs
  -> else:
       old_log_probs are recomputed by FSDP/HF compute_log_prob()

actor PPO update
  -> if actor.use_rollout_log_probs=True:
       consumes model_inputs["old_log_probs"]
  -> else, when on_policy:
       overwrites old_log_prob with current log_prob.detach()

vanilla PPO loss
  -> ratio = exp(log_prob - old_log_prob)
  -> mask = response_mask, so think/action tokens both enter the ratio
```

Relevant code locations:

- `vagen/agent_loop/agent_loop_no_concat.py:421-425`, `:543-546`, `:913-914`
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:399-446`
- `vagen/ray_trainer.py:2088-2114`
- `verl/verl/trainer/ppo/rollout_corr_helper.py:930-937`
- `verl/verl/workers/actor/dp_actor.py:541-570`
- `verl/verl/trainer/ppo/core_algos.py:938-963`

Historical B5/C8 config did not enable `calculate_log_probs`,
`rollout_correction.bypass_mode`, or `actor.use_rollout_log_probs`. Therefore
historical PPO did not reliably use the vLLM behavior-policy denominator. In
the common one-epoch one-minibatch case it could even be replaced inside actor
update by `log_prob.detach()` from the HF/FSDP current forward.

### D0.12.2 Repair

Minimal repair was applied to the B5/C8 wrapper only:

```text
examples/train/active_spatial/experiments/b5_c8_wrapper_img25_actionvalid.sh

+actor_rollout_ref.actor.use_rollout_log_probs=True
actor_rollout_ref.rollout.calculate_log_probs=True
actor_rollout_ref.rollout.logprob_temperature=1.0
algorithm.rollout_correction.bypass_mode=True
```

This preserves vLLM rollout raw sampled-token logprobs, maps them to response
tokens, sets trainer `old_log_probs = rollout_log_probs`, and prevents the actor
on-policy shortcut from replacing them with HF/FSDP `log_prob.detach()`.

No PPO objective, reward, KL, entropy, LR, GAE, action mask, architecture, or
prompt change was made.

### D0.12.3 Fixed-Sample Validation

Using the D0.6 fixed samples (`sample 3` normal, `sample 5` top mismatch),
historical consumed-old path is represented by `A_prod_fsdp_full`, while
behavior policy is `C_vllm_production_incremental`.

Before repair:

| Group | Tokens | mean \|delta logp\| | median | p95 | max | ratio median | frac \|ratio-1\| > 5% | Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_response | 227 | 0.082697 | 0.027880 | 0.266220 | 4.277603 | 0.993883 | 0.370044 | 0.920964 |
| think_text | 186 | 0.090800 | 0.037146 | 0.269603 | 4.277603 | 0.988565 | 0.403226 | 0.911429 |
| action_tag+name | 27 | 0.030066 | 0.001056 | 0.149056 | 0.386397 | 0.999946 | 0.185185 | 0.931436 |
| action_name | 15 | 0.052591 | 0.004462 | 0.241684 | 0.386397 | 0.999621 | 0.333333 | 0.918498 |

After repair, PPO-consumed `old_log_probs` is the same tensor as
`rollout_log_probs`, so fixed-sample parity is exact:

```text
mean |delta logp| = 0.0
median = 0.0
p95 = 0.0
max = 0.0
ratio median = 1.0
frac |ratio-1| > 5% = 0.0
Pearson = 1.0
```

This holds for all_response, think_text, action_tag+name, and action_name.

### D0.12.4 Bounded PPO Smoke

A D0.12 one-step smoke script was added:

```text
examples/train/active_spatial/experiments/d0_12_b5_logprob_plumbing_smoke.sh
```

It sets:

```text
TOTAL_STEPS=1
VAL_BEFORE_TRAIN=False
TEST_FREQ=-1
SAVE_FREQ=-1
CRITIC_WARMUP=0
```

The smoke was rerun on `10.119.28.235` with a renderer started on the same
node:

```text
renderer = http://127.0.0.1:8767/render
```

Confirmed:

```text
Hydra accepted repair overrides
actor.use_rollout_log_probs=True
rollout.calculate_log_probs=True
rollout.logprob_temperature=1.0
algorithm.rollout_correction.bypass_mode=True
Ray started
FSDP actor/ref/critic initialized
vLLM async HTTP servers started
training progress entered 0/1
local renderer processed real /render requests
rollout batch reached actor update with Pad 0 samples
```

The earlier renderer blocker is therefore ruled out for this attempt. The run
reached a real actor update and wrote per-rank optimizer audit files:

| Rank | requires_grad params | optimizer params | params with grad | grad norm | update norm |
|---:|---:|---:|---:|---:|---:|
| 0 | 1005915924 | 1005915924 | 1005915924 | 0.075039 | 0.013483 |
| 1 | 1005915924 | 1005915924 | 1005915924 | 0.029711 | 0.012845 |
| 2 | 1005915924 | 1005915924 | 1005915924 | 0.030380 | 0.012992 |
| 3 | 1005915924 | 1005915924 | 1005915924 | 0.244095 | 0.014648 |
| 4 | 1005915924 | 1005915924 | 1005915924 | 0.063059 | 0.014560 |
| 5 | 1005915924 | 1005915924 | 1005915924 | 0.128938 | 0.013466 |
| 6 | 1005915924 | 1005915924 | 1005915924 | 0.034717 | 0.013328 |
| 7 | 1005915924 | 1005915924 | 1005915924 | 0.034265 | 0.013314 |

Because FSDP is running with `use_orig_params=false`, this audit is exposed as
`fsdp_flat_mixed` rather than separated vision/projector/LLM groups. It still
proves that the real batch reached forward/backward/optimizer membership and
nonzero parameter update on all 8 ranks.

The bounded command timed out after this point while Ray workers were still in
`actor_rollout_update_actor`. No
`post_update_resync_step*.json` was produced, so the post-update actor -> vLLM
sync hook and same-sample post-sync parity were not executed.

Cleanup:

```text
local renderer stopped
D0.12 Ray actor-update worker PIDs 162860-162867 were killed
no D0.12 renderer process remained on 127.0.0.1:8767
```

Most killed Ray workers became defunct while awaiting parent reaping; one PID
briefly remained in kernel `D` state, but no D0.12 GPU process was reported by
the final GPU-process query.

### D0.12.5 Artifacts

Generated:

- `docs/diagnosis/d0_12_logprob_flow.json`
- `docs/diagnosis/d0_12_old_logprob_parity.json`
- `docs/diagnosis/d0_12_ppo_smoke.json`

Not generated because the real one-update closure did not reach post-sync:

- `docs/diagnosis/d0_12_post_sync_parity.json`

Validation:

```text
D0.12 JSON artifacts parse successfully
D0.12 smoke script bash -n passed
```

### D0.12.6 Gate

```text
INCONCLUSIVE — old-logprob plumbing repaired and fixed-sample parity is exact.
The local-renderer smoke reached rollout, PPO old-logprob bypass, and a real
FSDP optimizer step, but the run timed out before actor update returned and
before post-update actor -> vLLM sync / same-sample post-sync parity could be
verified.
```

D1 is not entered. C8/B5 is not rerun. No full RL training or navigation
evaluation was launched.

## D0.13 — Actor Update Return + Post-Sync Closure

### D0.13.1 Timeline Instrumentation

Minimal env-gated timeline markers were added around the real actor-update path:

- trainer side: before/after `actor_rollout_wg.update_actor(batch)`, actor
  metric reduction, and diagnostic post-update resync;
- FSDP worker side: model/optimizer load, sharding-manager entry, policy update,
  DataProto output, FSDP model offload, FSDP optimizer offload, and worker return;
- actor side: backward, grad clip, optimizer step, optimizer audit, metrics
  reduce, and worker return.

The markers are enabled only when `VAGEN_D0_13_TIMELINE_DIR` is set.

Artifacts:

- `docs/diagnosis/d0_13_update_actor_timeline.json`
- `docs/diagnosis/d0_13_post_sync_closure.json`

### D0.13.2 One-Step Return Result

The D0.12 bounded smoke was rerun on `10.119.28.235` with the local HTTP
renderer at:

```text
http://127.0.0.1:8767/render
```

The run completed `TOTAL_STEPS=1`:

```text
Training Progress: 100%|...| 1/1 [12:19<00:00, 739.77s/it]
trainer/actor_update_performed = 1.0
d0_3/post_update_resync_called = 1.0
timing_s/update_actor = 178.771537
timing_s/d0_3_post_update_resync = 6.490961
actor/grad_norm = 10.509290
```

The previous D0.12 conclusion that `update_actor(batch)` did not return is
superseded by these markers:

```text
trainer before_update_actor_call  -> after_update_actor_call
trainer before_post_update_resync -> after_post_update_resync
all 8 FSDP workers reached worker_returned
all 8 actor ranks reached before_worker_return
```

### D0.13.3 Hang Localization

There was no evidence of a distributed collective hang, DataProto serialization
hang, Ray future hang, or metrics-reduction mismatch.

The apparent stall was dominated by post-step offload/sync work that had little
console output. The slowest region was FSDP optimizer CPU offload:

| Region | Min seconds | Max seconds |
|---|---:|---:|
| actor backward | 1.353732 | 1.375202 |
| actor optimizer step | 0.022440 | 0.044325 |
| actor metrics reduce | 0.002784 | 0.003586 |
| FSDP `update_policy` wrapper | 30.594120 | 39.609004 |
| FSDP model offload | 0.207281 | 25.598317 |
| FSDP optimizer offload | 105.682799 | 137.966808 |

Root cause:

```text
not an update_actor hang;
bounded smoke was slow/silent after optimizer.step, especially during FSDP
optimizer offload, then actor -> vLLM diagnostic resync completed.
```

### D0.13.4 Optimizer Update Evidence

The real batch again produced nonzero gradient and update norms on all 8 ranks:

| Rank | grad_clip_norm | grad norm | update norm |
|---:|---:|---:|---:|
| 0 | 10.509290 | 0.064870 | 0.013354 |
| 1 | 10.509290 | 0.027330 | 0.012688 |
| 2 | 10.509290 | 0.026157 | 0.012836 |
| 3 | 10.509290 | 0.263971 | 0.014526 |
| 4 | 10.509290 | 0.059556 | 0.014444 |
| 5 | 10.509290 | 0.098007 | 0.013341 |
| 6 | 10.509290 | 0.028099 | 0.013205 |
| 7 | 10.509290 | 0.027171 | 0.013192 |

Because FSDP uses `use_orig_params=false`, these are still exposed as
`fsdp_flat_mixed` instead of module-level groups.

### D0.13.5 Actor -> vLLM Sync Closure

The trainer-side post-update resync hook executed and returned:

```json
{
  "global_step": 1,
  "tag": "post_update_resync",
  "used_production_rollout_mode": true
}
```

However, this D0.13 run did not enable `VAGEN_D0_6_DIR`, so the existing
tensor-level vLLM sync audit did not write post-update
`d0_6_weight_equality*.json` files. Therefore:

```text
sync invocation: verified
sync return: verified
post-sync tensor equality: not verified in this run
```

### D0.13.6 Same-Sample Post-Sync Parity

No fixed sample 5 updated-FSDP-vs-updated-production-vLLM parity probe was run
before the one-step process shut down. This remains the only intended D0.13
closure item not completed.

An immediate bounded rerun with:

```text
VAGEN_D0_6_DIR=<d0_13_post_sync_weight_audit_dir>
VAGEN_D0_6_ABORT_ON_WEIGHT_MISMATCH=1
VAGEN_D0_13_TIMELINE_DIR=<fresh_d0_13_timeline_dir>
```

was prepared conceptually, but could not be launched from this session because
the next SSH request to `10.119.28.235` was blocked by the platform approval
usage limit.

### D0.13.7 Gate

```text
INCONCLUSIVE — update_actor return and post-update resync invocation are
verified; the earlier actor-update hang hypothesis is ruled out. D0 is still not
closed because post-sync tensor equality and same-sample post-sync parity were
not executed.
```

D1 is not entered. C8/B5 is not rerun. No full RL training, PPO continuation,
post-sync parity rerun, or navigation evaluation was launched after this point.

## D0.14 — Final Post-Sync Closure

### D0.14.1 Bounded One-Step Closure

Ran the already validated D0.12/D0.13 bounded smoke on `10.119.28.235` with a
local HTTP renderer:

```text
renderer = http://127.0.0.1:8767/render
TOTAL_STEPS=1
VAL_BEFORE_TRAIN=False
TEST_FREQ=-1
SAVE_FREQ=-1
```

Additional D0.14 env:

```text
VAGEN_D0_6_DIR=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/d0_12_b5_logprob_plumbing_smoke/d0_12_smoke/d0_14_post_sync_weight_audit_run1
VAGEN_D0_6_ABORT_ON_WEIGHT_MISMATCH=1
VAGEN_D0_13_TIMELINE_DIR=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/d0_12_b5_logprob_plumbing_smoke/d0_12_smoke/d0_14_timeline_run1
```

The run completed:

```text
Training Progress: 100%|...| 1/1 [09:00<00:00, 540.68s/it]
trainer/actor_update_performed = 1.0
actor/grad_norm = 7.680732
d0_3/post_update_resync_called = 1.0
timing_s/update_actor = 79.533154
timing_s/d0_3_post_update_resync = 73.397758
```

The longer resync time is expected because D0.6 tensor equality audit was enabled
inside the post-update sync path.

### D0.14.2 Post-Sync Tensor Equality

Artifact:

```text
docs/diagnosis/d0_14_post_sync_weight_equality.json
```

The real post-update actor -> vLLM sync wrote 8 rank-local audit files:

```text
d0_6_weight_equality_rank0_tp0_pid236006.json
d0_6_weight_equality_rank1_tp1_pid236007.json
d0_6_weight_equality_rank2_tp0_pid236009.json
d0_6_weight_equality_rank3_tp1_pid236010.json
d0_6_weight_equality_rank4_tp0_pid236011.json
d0_6_weight_equality_rank5_tp1_pid236008.json
d0_6_weight_equality_rank6_tp0_pid236012.json
d0_6_weight_equality_rank7_tp1_pid236013.json
```

Summary:

| Item | Result |
|---|---:|
| audit files | 8 |
| compared tensors per rank | 27 |
| total tensor records | 216 |
| missing tensors | 0 |
| bad tensors | 0 |
| all ranks equal after source cast to target dtype | true |
| strict equality without dtype cast | false |

The strict no-cast comparison is false only because FSDP source tensors are
`torch.float32` while vLLM runtime tensors are `torch.bfloat16`. After casting
source -> target runtime dtype, every checked tensor matches exactly:

| Component | Tensor records | Bad | Source dtype | Target dtype | max raw diff | max diff after source cast |
|---|---:|---:|---|---|---:|---:|
| vision | 32 | 0 | fp32 | bf16 | 0.007775 | 0.0 |
| projector | 32 | 0 | fp32 | bf16 | 0.000942 | 0.0 |
| image_newline / MIV-related | 8 | 0 | fp32 | bf16 | 0.000233 | 0.0 |
| LLM TP shards | 136 | 0 | fp32 | bf16 | 0.013661 | 0.0 |
| lm_head | 8 | 0 | fp32 | bf16 | 0.001413 | 0.0 |

Post-sync tensor equality:

```text
PASS
```

No new actor -> vLLM sync bug was found.

### D0.14.3 Same-Sample Post-Sync Parity

Artifact:

```text
docs/diagnosis/d0_14_same_sample_post_sync_parity.json
```

Status:

```text
NOT EXECUTED
```

Reason:

```text
The bounded trainer path currently has no existing post-sync fixed-response vLLM
forced-scoring API. The one-step process completed and then shut down without
saving updated weights, so updated FSDP/HF vs updated production vLLM fixed
sample 5 logprob parity cannot be reconstructed after process exit.
```

Available post-sync evidence is strong but not identical to the requested
fixed-sample parity:

```text
one-step completed
optimizer.step completed
post-update actor -> vLLM resync called and returned
post-sync sampled policy-relevant tensor equality PASS
```

The known pre-update numerical baseline remains the D0.11 conclusion:

```text
HF/FSDP eager BF16 attention != vLLM FlashAttention / FP32-like attention
```

D0.14 did not produce evidence that this mismatch worsened after sync; however,
it also did not directly score fixed sample 5 after the update.

### D0.14.4 Cleanup

The D0.14 local renderer was stopped after the run:

```text
curl http://127.0.0.1:8767/health -> connection refused
```

No D0.14 training/vLLM process was intentionally left running.

### D0.14.5 Gate

```text
INCONCLUSIVE — post-sync tensor equality is verified and PASS, but the requested
same-sample post-sync fixed-response parity is still missing because the current
production trainer path lacks a post-sync vLLM forced-scoring hook.
```

D1 is not entered. C8/B5 is not rerun. No full RL training, PPO continuation, or
navigation evaluation was launched after this point.

## D0.15 — Pre-Update PPO Ratio + Rollout Correction Audit

Artifacts:

```text
docs/diagnosis/d0_15_rollout_correction_audit.json
docs/diagnosis/d0_15_pre_update_ratio_audit.json
```

### D0.15.1 Current PPO Old-Logprob Flow

The current repaired B5/C8 plumbing is:

```text
vLLM generation
-> rollout_log_probs
-> bypass_mode=True apply_rollout_correction()
-> old_log_probs = rollout_log_probs
-> actor update computes HF/FSDP current log_prob
-> vanilla PPO ratio = exp(current_hf_logp - old_logp)
```

Code anchors:

| Stage | Code | Finding |
|---|---|---|
| rollout requests sampled logprobs | `vagen/agent_loop/agent_loop_no_concat.py:421-425` | `logprobs=config.calculate_log_probs` |
| rollout stores sampled logprobs | `vagen/agent_loop/agent_loop_no_concat.py:543-546` | response logprobs are padded to response length |
| bypass old-logprob path | `vagen/ray_trainer.py:2277-2281`, `verl/verl/trainer/ppo/rollout_corr_helper.py:909-937` | `old_log_probs = rollout_log_probs` |
| actor current policy path | `verl/verl/workers/actor/dp_actor.py:591-601` | HF/FSDP `log_prob` is recomputed before loss |
| PPO ratio | `verl/verl/trainer/ppo/core_algos.py:888-970` | `ratio = exp(log_prob - old_log_prob)` |

Therefore, with the current D0.12 repair:

```text
old_log_prob == vLLM behavior logprob
current_log_prob == HF/FSDP training-policy logprob
pre-update PPO ratio == exp(HF_current_logp - vLLM_rollout_logp)
```

This is the correct behavior-policy denominator, but it is not a guarantee that
the pre-update PPO proximal ratio is near 1 when the training and rollout
runtimes implement numerically different policies.

### D0.15.2 Pre-Update Ratio Audit Status

I added an env-gated bounded audit in `vagen/ray_trainer.py`:

```text
VAGEN_D0_15_DIR=<dir>
VAGEN_D0_15_STOP_AFTER_AUDIT=1
```

The audit insertion point is after rollout/reward/old-logprob/advantage/filter
construction and before critic or actor optimizer updates. It computes HF/FSDP
`compute_log_prob(batch)` on the same real rollout batch, compares it to vLLM
`rollout_log_probs`, and writes split stats for:

```text
all_response
think_text
action_tag+name
action_name
```

The fresh real-batch audit did not run in this turn because every attempted GPU
node was unreachable from the current host:

```text
10.119.28.235 -> No route to host
10.119.28.231 -> No route to host
10.119.17.247 -> No route to host
10.119.30.117 -> No route to host
```

Prior evidence is not substituted as the D0.15 result, but it sets expectation:

| Source | Scope | Key evidence |
|---|---|---|
| `docs/diagnosis/d0_12_old_logprob_parity.json` | fixed D0 samples | before repair all-response mean abs delta 0.082697; think_text 0.090800; action_name 0.052591 |
| `docs/diagnosis/d0_14_post_sync_weight_equality.json` | real one-step aggregate | `rollout_corr/log_ppl_abs_diff=0.040675`, `ppl_ratio=1.041738` |

These support that the HF/FSDP current-vs-vLLM rollout gap remains material, but
the exact D0.15 real-batch think/action split still requires a reachable GPU
node.

### D0.15.3 Rollout Correction Semantics

`rollout_correction` is explicitly designed for rollout/training policy
mismatch. The helper describes off-policy gaps including implementation/runtime
mismatch, and the normal decoupled path computes:

```text
log_ratio = HF_old_log_prob - vLLM_rollout_log_prob
rollout_is_weights = exp(log_ratio), truncated and detached
optional rejection mask / veto
```

Code anchors:

| Mode | Code | Semantics |
|---|---|---|
| bypass | `verl/verl/trainer/ppo/rollout_corr_helper.py:909-937` | skip old-logprob recompute; use `rollout_log_probs` as PPO `old_log_probs` |
| decoupled correction | `verl/verl/trainer/ppo/rollout_corr_helper.py:809-866` | compute IS weights/rejection from HF old vs vLLM rollout |
| IS weights | `verl/verl/trainer/ppo/rollout_corr_helper.py:319-428` | `exp(training_logp - rollout_logp)`, thresholded, detached |
| PPO loss application | `verl/verl/trainer/ppo/core_algos.py:966-967` | multiply policy loss by `rollout_is_weights` when present |
| pure PG mode | `verl/verl/trainer/ppo/core_algos.py:1555-1664` | no PPO clipping; uses IS/RS corrected policy-gradient loss |

Important distinction:

```text
bypass_mode=True
```

is not the same as decoupled rollout correction. It makes old_logprob equal to
the behavior policy, but PPO's pre-update ratio becomes the cross-runtime ratio
`HF_current / vLLM_rollout`.

In contrast:

```text
bypass_mode=False + rollout_is enabled
```

uses HF/FSDP recompute as the PPO proximal anchor, so the first pre-update PPO
ratio is near 1 by construction, while the rollout/training mismatch is handled
as a separate detached IS/rejection correction.

### D0.15.4 Recommendation

Current bypass is useful for preserving behavior-policy logprob semantics, but
it is not sufficient final handling for the D0.11 runtime mismatch:

```text
HF/FSDP eager BF16 attention != vLLM FlashAttention / FP32-like attention
```

Recommended final direction:

```text
Enable decoupled rollout_correction with an explicit rollout_is/RS policy,
or align the HF/FSDP and vLLM attention backend so pre-update ratio is naturally near 1.
```

Do not keep the current bypass-only configuration as the final D0 answer if the
fresh pre-update split confirms the expected material ratio drift.

### D0.15.5 Gate

```text
FAIL — pre-update PPO ratio remains expected to be materially inconsistent under
the current bypass-only config; rollout_correction is the existing mechanism for
this mismatch, but it is not enabled in the decoupled IS/RS sense. The fresh
real-batch split audit is instrumented but was blocked by GPU node connectivity.
```

D1 is not entered. No full training, PPO continuation, C8/B5 rerun, or navigation
evaluation was launched.

## D0.16 — Decoupled Rollout Correction Closure

Artifact:

```text
docs/diagnosis/d0_16_decoupled_rollout_correction_flow.json
```

### D0.16.1 Code-Level Tensor Flow

The non-bypass code path matches the intended decoupled correction formula:

```text
behavior policy:
  rollout_log_probs = vLLM sampled-token raw logprob

PPO proximal anchor:
  old_log_probs = HF/FSDP pre-update recompute

PPO ratio:
  B = exp(current_HF - old_HF)

off-policy correction:
  C = exp(old_HF - rollout_vLLM)

diagnostic bypass ratio:
  A = exp(current_HF - rollout_vLLM)

identity:
  A = B * C
```

Code anchors:

| Stage | Code | Finding |
|---|---|---|
| vLLM behavior logprob | `vagen/agent_loop/agent_loop_no_concat.py:421-425`, `vagen/agent_loop/agent_loop_no_concat.py:543-546` | rollout sampled-token logprobs are stored as `rollout_log_probs` |
| HF/FSDP old anchor | `vagen/ray_trainer.py:2270-2298` | non-bypass path calls `actor_rollout_wg.compute_log_prob(batch)` and unions result as `old_log_probs` |
| correction construction | `vagen/ray_trainer.py:2686-2698`, `verl/verl/trainer/ppo/rollout_corr_helper.py:809-866` | `compute_rollout_correction_and_add_to_batch()` uses `old_log_probs` vs `rollout_log_probs` |
| IS weights | `verl/verl/trainer/ppo/rollout_corr_helper.py:319-428` | supported modes are `token` and `sequence`; weights are thresholded and detached |
| RS/veto | `verl/verl/trainer/ppo/rollout_corr_helper.py:546-697` | optional; default disabled |
| actor loss | `verl/verl/workers/actor/dp_actor.py:591-626`, `verl/verl/trainer/ppo/core_algos.py:888-970` | current HF/FSDP logprob is recomputed; vanilla PPO multiplies `pg_losses` by `rollout_is_weights` once if present |

Important configuration caveat:

```text
actor_rollout_ref.actor.use_rollout_log_probs=True
```

should remain enabled in the decoupled test. Otherwise, for a single minibatch
and single PPO epoch, the actor's on-policy shortcut can replace `old_log_prob`
with `log_prob.detach()`, bypassing the trainer-provided HF/FSDP old anchor.

### D0.16.2 Minimum Native Config To Test

The minimal framework-native decoupled correction config is:

```text
algorithm.rollout_correction.bypass_mode=False
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
algorithm.rollout_correction.rollout_rs=null
algorithm.rollout_correction.use_policy_gradient=False
actor_rollout_ref.actor.use_rollout_log_probs=True
actor_rollout_ref.rollout.calculate_log_probs=True
actor_rollout_ref.rollout.logprob_temperature=1.0
```

This keeps PPO clipping anchored to HF/FSDP `old_log_probs`, while adding a
detached token-level `old_HF / rollout_vLLM` IS correction weight to the policy
objective.

### D0.16.3 Runtime Audit Instrumentation

I added a D0.16 env-gated audit in `vagen/ray_trainer.py`:

```text
VAGEN_D0_16_DIR=<dir>
VAGEN_D0_16_STOP_AFTER_AUDIT=1
```

It records, on a single real rollout batch before optimizer update:

```text
A = exp(current_HF - rollout_vLLM)
B = exp(current_HF - old_HF)
C = exp(old_HF - rollout_vLLM)
A ~= B*C
```

with split stats for:

```text
all_response
think_text
action_tag+name
action_name
```

and IS diagnostics:

```text
ESS/N
clip fraction
RS acceptance fraction
rollout_is_weights tensor stats
```

### D0.16.4 Runtime Status

Runtime node:

```text
10.119.16.197
```

Artifacts:

```text
docs/diagnosis/d0_16_decoupled_correction_closure.json
exps/vagen_active_spatial/d0_16_b5_decoupled_rollout_correction_smoke/
  d0_16_actor_only_smoke_run5_2gpu_b8_foreachfalse/
    d0_16_decoupled_correction_audit_step1.json
  d0_16_timeline_actor_only_smoke_run5_2gpu_b8_foreachfalse/
  d0_16_smoke/post_update_resync_step1.json
```

Run configuration:

```text
TOTAL_STEPS=1
local HTTP renderer at http://127.0.0.1:8767
no validation
no checkpoint save
critic.enable=False
algorithm.rollout_correction.bypass_mode=False
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
algorithm.rollout_correction.rollout_rs=null
actor_rollout_ref.actor.use_rollout_log_probs=True
actor_rollout_ref.rollout.calculate_log_probs=True
actor_rollout_ref.rollout.logprob_temperature=1.0
```

On the 2-GPU bounded smoke, plain torch AdamW foreach mode produced a transient
actor optimizer OOM. The successful diagnostic run used the existing optimizer
override path:

```text
actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}
```

This changes only the AdamW execution implementation for the bounded smoke. It
does not change PPO objective, rollout correction math, reward, KL, LR,
sampling, prompt, or model architecture.

Pre-update A/B/C matrix on the same real rollout batch:

| Split | Tokens | A median | A p5/p95 | A \|ratio-1\|>5% | B median | B \|ratio-1\|>5% | IS ESS/N | IS min/max | IS clip frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_response | 941 | 0.98836 | 0.80542 / 1.11655 | 40.60% | 1.00000 | 0.00% | 0.98543 | 0.10517 / 1.88933 | 0.00% |
| think_text | 823 | 0.98174 | 0.80258 / 1.10980 | 44.23% | 1.00000 | 0.00% | 0.98424 | 0.10517 / 1.88933 | 0.00% |
| action_tag+name | 63 | 0.99967 | 0.97196 / 1.03245 | 6.35% | 1.00000 | 0.00% | 0.99902 | 0.84917 / 1.13673 | 0.00% |
| action_name | 30 | 0.99853 | 0.95113 / 1.04365 | 10.00% | 1.00000 | 0.00% | 0.99820 | 0.84917 / 1.13673 | 0.00% |

Where:

```text
A = exp(current_HF - rollout_vLLM)
B = exp(current_HF - old_HF)
C = exp(old_HF - rollout_vLLM)
```

The identity check passed exactly on all splits:

```text
max_abs_log_error(A, B*C) = 0
max_abs_ratio_error(A, B*C) = 0
RS acceptance fraction = 1.0
```

Interpretation:

```text
bypass ratio A is materially off-policy, especially on think tokens
PPO proximal ratio B is exactly 1.0 before optimizer update
the rollout-vs-HF mismatch is isolated into detached IS weights C
IS weights are healthy under token-level threshold=2.0
```

The one-step actor smoke completed:

```text
actor update returned: yes
grad_norm: 7.8679704666
actor pg_loss: -0.0095384826
actor total_loss: -0.0073149872
loss/grad finite: yes
post-update actor->vLLM resync called: yes
```

Timeline evidence:

```text
both ranks reached after_backward
both ranks reached after_optimizer_step
both ranks reached before_worker_return
both FSDP workers reached worker_returned
trainer reached after_update_actor_call
trainer reached after_post_update_resync
```

### D0.16.5 Gate

```text
PASS — decoupled rollout correction correctly handles vLLM/HF policy mismatch
```

Recommendation:

```text
Use framework-native decoupled rollout correction:
  bypass_mode=False
  rollout_is=token
  rollout_is_threshold=2.0
  rollout_rs=null initially
  actor.use_rollout_log_probs=True
  rollout.calculate_log_probs=True
  rollout.logprob_temperature=1.0

Do not use bypass_mode=True for Cambrian-S vLLM/HF mixed-runtime PPO.
Attention backend alignment is no longer required to make PPO ratio plumbing
correct, although it remains a possible future engineering cleanup.
```

D0.16 is closed. D1 is not entered in this run. No full training, backend work,
vision/MIV/sync re-audit, C8/B5 rerun, or navigation evaluation was launched.

## D0.17 — Active Spatial Historical Validity Audit

D0.17 stops Cambrian-specific plumbing diagnosis and records the historical
blast radius plus clean restart plan.

Artifacts:

```text
docs/diagnosis/ACTIVE_SPATIAL_HISTORICAL_VALIDITY.md
docs/diagnosis/active_spatial_historical_validity.json
examples/train/active_spatial/experiments/d0_17_qwen_v46_logprob_audit.sh
examples/train/active_spatial/experiments/qwen_active_spatial_clean_v1.sh
examples/train/active_spatial/experiments/cambrian_active_spatial_clean_v1.sh
examples/train/active_spatial/experiments/d0_17_qwen_clean_acceptance_smoke.sh
examples/train/active_spatial/experiments/d0_17_cambrian_clean_acceptance_smoke.sh
examples/train/active_spatial/launch_d0pass_clean_anchors.sh
```

Static findings:

```text
Cambrian C1-C8/B5: INVALIDATED for clean Cambrian-S RL method comparison.
Qwen v46: static exposure found, but real-batch impact not yet established.
Pure pretrained / inference-only eval: remains VALID as fixed-checkpoint behavior.
```

Qwen v46 static evidence:

```text
historical rollout.calculate_log_probs=False
historical rollout_correction.bypass_mode=False
historical rollout_is=None
```

Therefore historical Qwen v46 did not use the explicit Cambrian/B5 bypass bug,
but also did not preserve vLLM behavior logprobs for decoupled correction. A
real-batch A/B/C audit is required before marking Qwen history invalidated.

Canonical clean configs are frozen:

```text
qwen_active_spatial_clean_v1:
  base = v46_baseline_qwen25vl_7b
  run_name = qwen_v46_clean_d0pass_v1
  shared PPO fixes only

cambrian_active_spatial_clean_v1:
  base = b5_c8_wrapper_img25_actionvalid
  run_name = cambrian_c8_clean_d0pass_v1
  landed Cambrian fixes + shared PPO fixes
```

GPU attempts:

```text
10.119.16.197 -> No route to host
10.119.28.235 -> No route to host
10.119.28.231 -> No route to host
10.119.20.47 -> reachable; Qwen v46 audit run5 reached Ray/FSDP/vLLM/AgentLoop
                 and real renderer requests, then failed before a valid batch
                 because gsplat_cuda.so was not built. Concurrent HTTP renderer
                 JIT hit build-directory/lock races; single-process precompile
                 then exposed missing cuda_runtime_api.h in the C++ include path.
                 Afterward SSH degraded to Connection refused/timeout and then
                 No route to host. 10.119.16.197 also remained No route to host.
local Codex container / 10.119.30.223 -> usable as renderer host when commands
                 run unsandboxed. Ordinary sandbox commands lack /dev/nvidia*
                 and report torch CUDA unavailable, but unsandboxed nvidia-smi
                 sees one NVIDIA H800. Jumpbox-local renderer at
                 10.119.30.223:8767 compiled gsplat_cuda.so successfully
                 in 782.55s and passed real reset/step preflight:
                 exps/vagen_active_spatial/d0_17_jumpbox_renderer_preflight_rerun/phase3_data_gate_audit.json
10.119.17.247 -> No route to host after renderer PASS
10.119.30.117 -> No route to host after renderer PASS
```

Because the Qwen v46 real-batch audit, acceptance smokes, and clean anchor
launches are still blocked by unreachable SSH training nodes rather than
completed evidence, D0.17 cannot claim PASS.

### D0.17 Update — Qwen Audit + Clean Restart

The earlier renderer blocker was corrected by running the HTTP renderer on the
jumpbox instead of the SSH training node. The jumpbox does have an H800 visible
to unsandboxed commands; the earlier `torch.cuda`/`nvidia-smi` failures were
from the Codex sandbox/device namespace. A detached renderer is healthy at:

```text
http://10.119.30.223:8768
```

Using `10.119.31.101`, the Qwen v46 real-batch audit completed:

```text
artifact = exps/vagen_active_spatial/d0_17_qwen_v46_logprob_audit_31_101_jumpbox_renderer/d0_17_qwen_audit/d0_16_decoupled_correction_audit_step1.json
A = exp(current_HF - rollout_vLLM)
B = exp(current_HF - old_HF)
C = exp(old_HF - rollout_vLLM)
max_abs_log_error(A, B*C) = 0
```

Qwen v46 impact:

| Split | Tokens | A mean abs dlogp | A \|ratio-1\|>5% | B mean abs dlogp | B \|ratio-1\|>5% | C ESS/N |
|---|---:|---:|---:|---:|---:|---:|
| all_response | 609 | 0.019671 | 13.63% | 0.000000 | 0.00% | 0.998301 |
| think_text | 469 | 0.024597 | 17.48% | 0.000000 | 0.00% | 0.997830 |
| action_tag+name | 84 | 0.001841 | 0.00% | 0.000000 | 0.00% | 0.999955 |
| action_name | 36 | 0.004067 | 0.00% | 0.000000 | 0.00% | 0.999898 |

Interpretation:

```text
Qwen is not affected by Cambrian-specific MIV/processor/vision-layer bugs.
Qwen v46 historical RL is affected as clean PPO method evidence because
behavior-logprob correction was not active and a real HF-vLLM gap exists.
The Qwen gap is much smaller than Cambrian and mostly think-token concentrated.
```

Clean acceptance smokes completed:

```text
Qwen smoke:
  run = d0_17_qwen_clean_acceptance_smoke_actorupdate_31_101_jumpbox_renderer
  status = PASS
  actor_update_performed = 1
  grad_norm = 7.6293959618
  proximal ratio B = 1.0 before update

Cambrian smoke:
  run = d0_17_cambrian_clean_acceptance_smoke_actorupdate_31_101_jumpbox_renderer
  status = PASS
  actor_update_performed = 1
  grad_norm = 8.0798578262
  proximal ratio B = 1.0 before update
```

Clean restart:

```text
launched = qwen_v46_clean_d0pass_20260818_renderer8768
node = 10.119.31.101
renderer = http://10.119.30.223:8768
log = exps/vagen_active_spatial/qwen_v46_clean_d0pass_20260818_renderer8768/train.log

not launched = cambrian_c8_clean_d0pass_*
reason = no second reachable idle 8-GPU training node during this pass
```

### D0.17 Gate

```text
INCONCLUSIVE — Qwen impact audited and Qwen clean anchor launched; Cambrian clean anchor still pending
```
