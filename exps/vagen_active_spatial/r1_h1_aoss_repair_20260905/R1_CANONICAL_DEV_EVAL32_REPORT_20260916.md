# R1 Canonical 32-Source Development Evaluation

Date: 2026-09-16

## Executive conclusion

The corrected evaluation completed 64/64 policy episodes with no infrastructure
errors: 32 for the pinned Qwen2.5-VL-7B-Instruct pretrained snapshot and 32 for
v46_baseline_qwen25vl_7b at global step 250. Both models achieved 0/32 canonical
success. All 32 paired outcomes were therefore both-fail; there was no
pretrained-only or v46-only success.

This is a valid floor result on a development-exposed, certified-Medium
regression set. It establishes model/environment functional compatibility and
shows that v46@250 does not improve canonical Medium success on this set. It
does not prove that the models are generally equivalent, that v46 training was
harmful, or that R1 training is effective. One source is 3.125 percentage
points, and this set is neither independent nor a final generalization test.

An earlier 64-episode run under policy_eval is invalid for model comparison.
That run rendered the policy observation with the native 640x480 camera
intrinsics at 256x256 while scoring with the H1 camera. Its apparent 2/32 versus
0/32 outcome must not be used. The corrected run is
policy_eval_h1_render_v2.

## Camera/runtime correction and validation

The pre-correction environment passed the row-native intrinsics directly to the
official renderer:

- native K: fx=320, fy=320, cx=320, cy=240 at 640x480;
- canonical H1 K at 256x256: fx=128, fy=170.6666667, cx=128, cy=128.

The scorer and history already used canonical H1 intrinsics, so the old policy
RGB and the task metric described different cameras. Commit
d940372a5d710fddb6275ee2b91866777fbf138a adds a versioned runtime render-camera
helper. For canonical R1 rows it reconstructs H1 from immutable native
intrinsics on every render, preventing repeated scaling. Historical rows retain
the legacy native-camera path.

Regression evidence:

- runtime render-camera, H1 camera utility, legacy compatibility and
  no-double-scaling tests passed;
- 18 repair-pipeline tests passed;
- 13 frozen-evaluation tests passed;
- both models passed the same four-source smoke gate;
- all 64 evaluated initial frames matched their frozen official-render evidence
  pixel-for-pixel: MAE=0, P99 absolute error=0 and maximum absolute error=0;
- initial pose error was 0 for every episode;
- all 64 initial states remained canonical-fail and consistent with the
  certified lower bound.

## Model restoration and load evidence

### Pretrained

- identity: Qwen2.5-VL-7B-Instruct, revision
  cc594898137f460bfe9f0759e9844b3ce807cfb5 (the historical v46 log records
  the same upstream snapshot);
- local path:
  r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898;
- 16 files, five safetensor shards, 729 tensors, 16,595,981,281 bytes;
- stored tensor dtype: BF16;
- model manifest SHA256:
  46f05ffcc6127a4caa9a3e8c11ddf298b9a5263c8680afe6b5d017ea91702c8b;
- processor: Qwen2_5_VLProcessor; tokenizer: Qwen2TokenizerFast;
- actual vLLM dtype: torch.bfloat16; no quantization;
- load smoke passed.

The per-file hashes freeze the bytes used in this run. They are newly computed
for R1 and must not be described as a comparison against a historical checksum.
The credential-free restoration form is:

    snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        local_dir=".../model_restore/qwen25vl7b_pretrained_cc594898",
    )

### v46 step 250

- identity: v46_baseline_qwen25vl_7b, global_step_250;
- historical training records locate the checkpoint at
  exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/checkpoints/global_step_250;
- restored from the existing AOSS upload record into
  r1_canonical_dev_eval32_20260915/model_restore/v46_step250_hf;
- the AOSS sync ledger records 19/19 files copied, 30.906 GiB and zero failures;
- 19 files, seven safetensor shards, 729 tensors, 33,184,687,614 bytes;
- stored tensor dtype: F32;
- model manifest SHA256:
  1fae9d370ade47ed79970cddbb27edab38c513526a29508bd1d3e7796e23374f;
- processor: Qwen2_5_VLProcessor; tokenizer: Qwen2TokenizerFast;
- vLLM auto mode explicitly logged the F32-to-BF16 load conversion; actual
  inference dtype was torch.bfloat16 for parity with pretrained, with no
  quantization;
- load smoke passed.

As with pretrained, the new per-file hashes freeze the restored bytes used here;
they do not by themselves prove equality to an unavailable older checksum.
No other checkpoint step or substitute model was used.

The v46 artifact was already in Hugging Face layout, so no weight conversion was
performed. Restoration used the existing AOSS client configuration and a
recursive copy into model_restore/v46_step250_hf, followed by
sha256sum -c v46_step250_hf_SHA256SUMS. The AOSS sync output is retained in
v46_step250_aoss_sync_v2.log without credentials. The raw source-prefix
invocation was not persisted in the versioned artifact; this is a provenance
limitation, and the report relies on the historical global_step_250 identity,
the 19/19 sync ledger, the frozen per-file hashes and the successful load
rather than claiming an independently verified historical checksum.

## Frozen 32-source manifest and protocol

Selection was deterministic: one episode per non-train source, choosing the
lexicographically smallest episode fingerprint. Certificate actions, terminal
pose, planner output and audit evidence paths were not exposed to either
policy.

Inventory:

| Split | Sources |
|---|---:|
| id_test | 16 |
| ood_scene | 6 |
| ood_instance | 5 |
| validation_proxy | 3 |
| ood_geometry | 1 |
| ood_template | 1 |
| Total | 32 |

The sources span seven scenes: 0014_841007 (1), 0226_840298 (8),
0229_840306 (3), 0240_840881 (11), 0276_840780 (1), 0361_840315 (3), and
0367_840260 (5). All have participated in R1 development, so the role is
development_regression and the generalization claim is none. There is no
category-OOD Medium source; that missing axis is not reported as a zero.

All tasks are certified Medium: complete formal-runtime search excludes every
success through depth 3, and the independently replayed certificate first
succeeds at step 4 (24 sources) or step 5 (8 sources).

Frozen runtime:

- canonical_spatial_task_h1_v1, H1 camera and 12px relation margin;
- six formal actions, 0.3m translation, 20 degree yaw;
- frozen per-scene collision convention;
- at most 12 primitive actions, canonical auto termination;
- same prompt, no-concat current-observation history, parser and paired
  per-turn seed;
- sampling temperature 0.8, top_p 0.95, max 384 tokens, stop at closing action
  tag;
- official InteriorGS renderer at 256x256.

Input SHA256:

- development_regression_manifest.jsonl:
  78a7a8f2a7c94bd0f007e89271f33d6a84ffa607a65e895613b43f8825b06b1d;
- policy_input_rows.jsonl:
  bf87b320df448f4da2aa053b709f4b644c42b49781e4d2155eca812df54e1108;
- eval_protocol.json:
  1024a6640c993e11d84a66d61d9062149b2f4f0329a6ab4aabe79c243399665f.

## Execution environment

- SCO job: pt-4gejc6k2 (r1-canonical-dev32-h1-r4), state SUCCEEDED;
- resource pool: zoetrope;
- resource: one NVIDIA H800 80GB, 14 CPU, 240GB RAM;
- worker: pt-ce5b1907d06345baa5c0ca1959ac3428-worker-0;
- start/completion: 2026-09-16 11:19:11 to 11:51:56 UTC;
- Python 3.12.13;
- official renderer port 8898, InteriorGS root
  /mnt/umm/users/yinbaiqiao/InteriorGS;
- injected code commit: d940372a5d710fddb6275ee2b91866777fbf138a;
- code archive SHA256:
  f6787de50fe7edab03853d4e4ca29fb82e91a3fb8b038974396c693f8aa2627f.

The requested environment set TORCH_SDPA and disabled FlashInfer sampling. vLLM
0.11.0 logged the Flash Attention backend for model attention and the
PyTorch-native top-p/top-k sampler. Both models used the same worker and actual
backend. This discrepancy between requested attention-backend environment and
vLLM's selected backend is retained as a reproducibility caveat, not hidden.

## 64-episode accounting and paired result

| Metric | Pretrained | v46@250 |
|---|---:|---:|
| Requested / complete | 32 / 32 | 32 / 32 |
| Canonical success | 0 | 0 |
| Success rate | 0.0% | 0.0% |
| Infrastructure errors | 0 | 0 |
| Primitive actions | 384 | 384 |
| Timeout at 12 actions | 32 | 32 |
| Collision attempts | 4 | 4 |
| Invalid model turns | 0 | 2 |
| Total model turns | 307 | 317 |
| Episode inference/rollout time | 259.22s | 258.62s |

Paired outcomes: both fail 32, both succeed 0, pretrained-only 0, v46-only 0.
Every split, scene, category pair and the step-4/step-5 difficulty strata
therefore also has zero successes for both models.

The two v46 invalid outputs occurred on ood_instance:170 and ood_scene:98. One
inserted an unsupported numeric token after turn_left; one emitted a long
sequence of x tokens. Both were recorded as model behavior, consumed no
substitute expert action and remained in the denominator.

Collision attempts were sparse and tied. Pretrained collisions occurred on
three sources (1+1+2 attempts); v46 collisions occurred on three sources
(1+1+2 attempts). Collision was therefore not the principal reason for the
zero-success result.

## Action and trajectory analysis

Action counts:

| Action | Pretrained | v46@250 |
|---|---:|---:|
| move_forward | 77 | 73 |
| move_backward | 1 | 5 |
| move_left | 9 | 21 |
| move_right | 14 | 10 |
| turn_left | 156 | 118 |
| turn_right | 127 | 157 |

Turns account for 283/384 (73.7%) pretrained actions and 275/384 (71.6%) v46
actions. The first action matched the audit certificate on only 3/32 pretrained
sources and 2/32 v46 sources. This certificate comparison is an offline
diagnostic; certificates were not policy inputs.

Both policies sometimes improved the canonical score without completing all
gates:

- pretrained improved over its initial score at least once on 23/32 sources;
- v46 improved on 25/32;
- pretrained mean best-minus-initial score was 0.0821;
- v46 mean best-minus-initial score was 0.0934.

At the final state, pretrained retained both targets visible on 11/32 sources
and inside-frame on 1/32; v46 retained both targets visible on 17/32 and
inside-frame on 2/32. Across the whole trajectory, pretrained reached the
relation-margin gate on one source and v46 on two, but never while all other
success gates also held.

Representative near misses:

- v46, id_test:6597 reached canonical score 0.8673 and relation margin 44.96px,
  but inside_frame=false: the policy created the requested ordering while
  cropping a target out of frame;
- v46, id_test:1095 reached score 0.7596 and margin 18.57px, again with
  inside_frame=false;
- pretrained, id_test:303 reached score 0.6413 and margin 19.32px, also with
  inside_frame=false.

Manual RGB review of these initial and near-miss frames confirms that the H1
initial views are meaningful and consistent with the frozen evidence. The
near-miss failures are real: the policy turns/translates until one target is
partly or fully lost rather than satisfying the relation with both targets in
frame. The evidence does not support blaming renderer emptiness, a scorer
termination mismatch, or an initial-state difficulty violation.

Because every episode timed out, the primary empirical class is
observation/action-selection failure under the 12-step budget. Two v46 parsing
failures and the tied collision attempts are secondary. The current logs do not
justify assigning a more specific internal causal label such as object
recognition failure.

## Artifact integrity

The top-level SHA256SUMS verified every listed ledger, episode JSON, RGB frame,
model-load record, log and paired artifact successfully after renderer
shutdown. Selected hashes:

- pretrained episode ledger:
  e04bcd4b6918863da497ca669583dac4cdb98a4e29ab75b7df2761bf44382e6e;
- v46 episode ledger:
  3671f1bbc70a6ff195094b44a4dce9adc3fb74c28728808fdba82defedc81806;
- paired_results.jsonl:
  a13dbd49997cf91b2f306178a5082dd4e4a134080b57d600aadd8ff3283bb6d5;
- paired_summary.json:
  b4e5e4daca083571170139299511e3595cd5321aae493ee4781d095112536948;
- full evaluation SHA256SUMS file:
  85d525d6004ff10664b4c2d2d7a23fe1a003cd656bc957bae682757170106ede.

Primary artifact directory:

exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/
r1_canonical_dev_eval32_20260915/policy_eval_h1_render_v2/

The invalid camera-mismatched run remains read-only under policy_eval for audit
lineage and must not be merged with the corrected results.

## Decision and exact next step

1. Model/environment functional evaluation: COMPLETE. Both exact model
   artifacts loaded, all 64 policy episodes closed, H1 RGB/pose/lower-bound
   checks passed, and no infrastructure error occurred.
2. v46 relative to pretrained on this development set: NO DETECTABLE
   IMPROVEMENT AT THE SUCCESS LEVEL. Both are 0/32, so the comparison is
   floor-limited; small trajectory-score differences are diagnostics, not
   evidence of policy superiority.
3. Generalization/training conclusion: NOT ESTABLISHED. These seven scenes were
   exposed during R1 development, the category-OOD Medium axis is absent, and
   no R1 training was run.

The next authorized model-side milestone should follow the already frozen
design: build and freeze an untouched 60-source canonical evaluation set with
at least ten identities for each of ID, scene OOD, instance OOD, category OOD,
template OOD and geometry OOD, spanning at least ten scenes not used for R1
selector/threshold development. Run the same paired evaluation without tuning
the prompt or metric on the result. This report does not start that generation.

Training must remain blocked until the separate train-only inventory reaches
the predeclared pilot minimum: 200 unique sources across at least 20 scenes,
left/right at least 80 each, step-4/5/6 at least 40 each, and no unordered
category-pair family above 35%. Relative to the frozen pre-evaluation inventory,
the documented deficits remain 136 unique train sources, ten scenes, 44 left,
52 right, 40 step-6 episodes and category diversification. Any later training
comparison is a system/data repair control, not a reward-only ablation.
