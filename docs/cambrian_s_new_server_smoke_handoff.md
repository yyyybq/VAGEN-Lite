# Cambrian-S 新服务器 Smoke 训练交接记录

Date: 2026-07-24

本文记录当前工作会话中对 VAGEN-Lite 项目完成的 Cambrian-S 新服务器配置、代码修复、问题排查和 smoke 实验结果。目标是让没有参与本次调试的开发者可以继续接手。

## 1. 项目目标与当前背景

VAGEN-Lite 是一个面向多模态强化学习训练的框架。本次工作目标是在新的 8 卡训练节点 `10.119.16.151` 上配置并跑通 Cambrian-S 7B 的 Active Spatial 训练 smoke test。

本次开始时，项目已有 Cambrian-S 适配代码，但新服务器还没有完整配置：Cambrian 外部源码、checkpoint、SigLIP vision tower cache、Active Spatial 数据路径、训练脚本和若干 Transformers/vLLM 兼容问题都需要验证。会话主要围绕 Cambrian-S 的 FSDP actor、vLLM rollout、图像 token 展开、scene 路径、分布式 batch 配置和 1-step PPO smoke run 展开。

## 2. 已完成的代码改动

### `vagen/models/cambrian_vllm.py`

修改区域：Cambrian vLLM 模型包装与 SigLIP vision config 处理。

修改前问题：

```text
AttributeError: 'SiglipConfig' object has no attribute 'num_channels'
```

原因是 `AutoConfig` 加载 SigLIP 时返回的配置结构里 `num_channels` 可能位于 nested `vision_config` 中，而不是顶层 `SiglipConfig`。

已做修改：在 Cambrian vLLM 适配中优先提取 nested `vision_config`，并在 fallback `SiglipConfig` 中补 `num_channels=3`。

影响：只影响 Cambrian-S 的 vLLM rollout worker 初始化和多模态图像处理路径。

验证：后续 rerun 中 vLLM HTTP server 成功启动，Cambrian plugin 成功注册，未再出现 `num_channels` 错误。

### `vagen/models/cambrian_register.py`

修改区域：`CambrianForCausalLMAdapter.__init__()`、`_patch_qwen2_decoder_layers()`、`forward()`。

问题 1：新版 Transformers 的 `Qwen2DecoderLayer` attention forward 期望 `position_embeddings=(cos, sin)`，Cambrian 外部实现未传入，报错：

```text
TypeError: cannot unpack non-iterable NoneType object
```

修改：新增 `_patch_qwen2_decoder_layers(inner_model)`，在 Cambrian actor/ref/critic 初始化时包装每层 Qwen2 decoder layer forward，动态构造 `position_ids` 和 RoPE `position_embeddings`。

关键逻辑：

```python
position_embeddings = _rotary_emb(hidden_states, position_ids)
kwargs["position_embeddings"] = position_embeddings
```

问题 2：某些 FSDP / gradient checkpointing 路径中 `hidden_states` 是 2D `(seq_len, hidden_dim)`，RoPE 计算期望 3D，曾报：

```text
RuntimeError: The size of tensor a (28) must match the size of tensor b (128)
```

修改：检测 `hidden_states.dim() == 2` 时临时 `unsqueeze(0)`，并规范 `position_ids` 为 `(batch, seq)`，支持 1D、转置和单 batch expand。

问题 3：最初把 decoder output squeeze 回 2D，导致 actor logprob 的 `dp_actor.py` 期望 3D logits 时失败：

```text
IndexError: too many indices for tensor of dimension 2
```

修改：移除 decoder shim 中对输出的 squeeze，保持 batch 维。

问题 4：Cambrian 外层 `forward()` 仍可能从 `self.model(...)` 得到 2D hidden，进而产生 2D logits。

修改：在 `lm_head` 前统一补 batch 维：

```python
hidden_states = outputs[0]
if hidden_states.dim() == 2:
    hidden_states = hidden_states.unsqueeze(0)
logits = self.lm_head(hidden_states).float()
```

影响：影响 Cambrian-S FSDP actor/ref/critic 的 forward、logprob 和 checkpoint 保存路径；不影响 Qwen 标准模型。

验证：`python3 -m py_compile vagen/models/cambrian_register.py` 通过；最终 rerun14 完成 1-step PPO，actor old logprob/ref logprob 均完成，checkpoint 成功保存。

### `examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_smoke_server.sh`

状态：创建并多次调整 smoke 专用实验脚本，基于 `c8_fwdfirst_rewscale.sh`。

主要配置：

```bash
MODEL_PATH="/mnt/umm/users/yinbaiqiao/hf_cache/cambrian-s-7b"
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=4
VAL_BATCH_SIZE=1
N_TRAJECTORY=1
TOTAL_STEPS=1
SAVE_FREQ=1
TEST_FREQ=100
VAL_BEFORE_TRAIN="False"
VAL_N=1
GPU_MEM_UTIL=0.35
```

修改原因：

- batch size 需要能被 actor worker / chunk 逻辑整除；
- `gpu_memory_utilization=0.20` 不足，vLLM cache block 初始化失败；
- 关闭训练前 validation，加快 debug；
- 保留 step 后 validation 和 checkpoint，用于验证完整链路。

验证：rerun14 成功完成 1 step，并写出 rollout、validation 和 checkpoint。

### `run_cambrian_smoke_server.sh`

状态：创建/修改 Cambrian smoke launcher。

主要作用：

- 设置 `CAMBRIAN_SRC=/mnt/umm/users/yinbaiqiao/cambrian-s`；
- 设置 HuggingFace cache 到 `/mnt/umm/users/yinbaiqiao/.cache/huggingface`；
- 设置 offline 相关环境变量；
- 默认暴露 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`；
- 启动 smoke experiment。

修改原因：

- 新服务器 `.bashrc` 非交互 shell 不会自动激活 conda；
- rendering config 使用 GPU 4，若只暴露 0-3 会触发 invalid device ordinal；
- vLLM plugin 依赖 editable install 和正确环境变量。

验证：最终通过该 launcher 启动 rerun14。

### `examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_smoke_server.yaml`

状态：创建 smoke 专用 env config。

主要改动：

- `gs_root` 从旧路径 `/scratch/by2593/project/Active_Spatial/InteriorGS` 改为 `/mnt/umm/users/yinbaiqiao/InteriorGS`；
- `test_size=1`；
- smoke 实验使用有效 validation scene 文件。

原因：新服务器存在 `/mnt/umm/users/yinbaiqiao/InteriorGS`，旧 scratch 路径不可用；`scene_test` 不是新数据集里的有效 scene。

验证：后续不再出现 PLY/scene path 错误，Active Spatial 环境能渲染并产生 `ENV_DEBUG`。

## 3. 发现的 Bug 和问题

### 已确认 Bug

1. `ModuleNotFoundError: No module named 'cambrian'`
   - 原因：外部 Cambrian-S 源码未在服务器上准备/导入。
   - 解决：下载/放置 Cambrian-S 到 `/mnt/umm/users/yinbaiqiao/cambrian-s`，设置 `CAMBRIAN_SRC`。
   - 状态：已解决。

2. 缺少依赖
   - 现象：缺少 `ezcolorlog`、`open_clip`、`diffusers`。
   - 解决：在 `/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite` 中安装 `ezcolorlog`、`open_clip_torch`、`diffusers`。
   - 状态：已解决。

3. SigLIP offline cache 缺失
   - 日志：`We couldn't connect to 'https://hf-mirror.com' ... couldn't find them in cached files`
   - 原因：`HF_HUB_OFFLINE=1` 时本地没有 SigLIP vision tower。
   - 解决：下载 `google/siglip2-so400m-patch14-384` 和 `google/siglip-so400m-patch14-384` 到本地 HF cache。
   - 状态：已解决。

4. vLLM SigLIP config 不兼容
   - 日志：`AttributeError: 'SiglipConfig' object has no attribute 'num_channels'`
   - 原因：Cambrian vLLM adapter 没有处理 nested `vision_config`。
   - 解决：修改 `vagen/models/cambrian_vllm.py`。
   - 状态：已解决。

5. vLLM cache memory 不足
   - 日志：`ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization`
   - 解决：`GPU_MEM_UTIL=0.20` 提高到 `0.35`。
   - 状态：已解决。

6. scene path 错误
   - 日志：`FileNotFoundError: Could not find PLY for scene scene_test under /scratch/...`
   - 原因：env config 仍指向旧 scratch 路径，且使用无效 scene。
   - 解决：创建 smoke env config，改 `gs_root` 到 `/mnt/umm/users/yinbaiqiao/InteriorGS`。
   - 状态：已解决。

7. rendering GPU 不可见
   - 日志：`CUDA error: invalid device ordinal`
   - 原因：只暴露 `CUDA_VISIBLE_DEVICES=0,1,2,3`，但 env/rendering 使用 GPU 4。
   - 解决：launcher 改为暴露 0-7。
   - 状态：已解决。

8. batch / chunk 不整除
   - 日志：`AssertionError: only support equal chunk. Got size of DataProto 4 and chunk 8`
   - 原因：`TRAIN_BATCH_SIZE=4` 与 worker/chunk 要求不匹配。
   - 解决：`TRAIN_BATCH_SIZE=8`，`PPO_MINI_BATCH_SIZE=4`。
   - 状态：已解决。

9. Qwen2 `position_embeddings=None`
   - 日志：`TypeError: cannot unpack non-iterable NoneType object`
   - 原因：Cambrian 外部 Qwen2 forward 与新版 Transformers API 不兼容。
   - 解决：`_patch_qwen2_decoder_layers()` 动态补 RoPE embeddings。
   - 状态：已解决。

10. RoPE shape mismatch
    - 日志：`RuntimeError: The size of tensor a (28) must match the size of tensor b (128)`
    - 原因：某些路径 hidden 是 2D，RoPE 位置维度处理错误。
    - 解决：2D hidden 临时补 batch 维，并规范 `position_ids`。
    - 状态：已解决。

11. actor logits 2D
    - 日志：`IndexError: too many indices for tensor of dimension 2`
    - 原因：Cambrian 外层 forward 可能返回 2D logits，但 `dp_actor` 期望 `(bsz, seq, vocab)`。
    - 解决：在 `CambrianForCausalLMAdapter.forward()` 中对 2D hidden 补 batch 维。
    - 状态：已解决；rerun14 成功通过。

### 环境/基础设施问题

- workspace 本地读取部分文件曾受权限或 ignore 限制影响，后续主要通过 SSH 命令在训练节点检查。
- `.bashrc` 非交互 shell 提前 return，conda 需要显式 source 或使用完整环境。
- `flash_attn` 有 GLIBC mismatch，日志显示 fallback 到 eager/native PyTorch attention。未阻塞 smoke，但会影响性能。
- 训练末尾 `wandb` atexit 出现 `BrokenPipeError`。发生在 1-step 完成、validation/checkpoint 写出之后，属于退出清理阶段，不影响本次 smoke 结果。

### 尚需注意的风险

- 当前修改针对 Cambrian-S 单样本/expanded image token 路径验证过，长训练、多 batch、多图、多轮更长上下文仍需进一步验证。
- final validation 只有 `VAL_N=1`，只能证明链路跑通，不能证明性能。
- `flash_attn` fallback 会显著影响速度，完整训练前建议处理或接受性能损失。

## 4. 配置与运行方式的变化

### 路径

- Cambrian source：`/mnt/umm/users/yinbaiqiao/cambrian-s`
- Cambrian checkpoint：`/mnt/umm/users/yinbaiqiao/hf_cache/cambrian-s-7b`
- Active Spatial `gs_root`：`/mnt/umm/users/yinbaiqiao/InteriorGS`

### 环境变量

```bash
CAMBRIAN_SRC=/mnt/umm/users/yinbaiqiao/cambrian-s
HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WANDB_MODE=offline
RAY_DEDUP_LOGS=0
```

影响：保证 Cambrian 外部源码、HF cache、rendering GPU 和 vLLM worker 都能在训练节点正常工作。

### 训练参数

smoke 关键参数：

```text
TOTAL_STEPS=1
SAVE_FREQ=1
TEST_FREQ=100
VAL_BEFORE_TRAIN=False
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=4
N_TRAJECTORY=1
VAL_N=1
TP_SIZE=2
GPU_MEM_UTIL=0.35
```

训练/算法沿用 c8 风格：

```text
actor_rollout_ref.model.external_lib=vagen.models.cambrian_register
critic.model.external_lib=vagen.models.cambrian_register
trust_remote_code=True
use_remove_padding=False
actor lr=5e-7
kl_loss_coef=0.20
kl_loss_type=low_var_kl
entropy_coeff=0.005
grad_clip=0.3
adv_estimator=no_concat_gae
gamma=0.95
lam=0.95
```

注意：日志显示 `algorithm.adv_estimator != gae` 时 critic 被 disable 的 warning，但后续仍记录了 critic/rewards/advantages 类指标。当前没有进一步确认该 warning 对训练语义的影响。

## 5. 数据与评估流程的修改

实际修改：

- 创建 smoke 专用 env config，修正 `gs_root`；
- 使用有效 validation JSONL，避免 `scene_test` 无效 scene；
- `test_size=1`，`VAL_N=1`，只做 smoke 级验证。

检查/观察：

- Active Spatial 环境能正常初始化、渲染并输出 `ENV_DEBUG`；
- agent 输出 action format 有效，例如 `move_forward`、`turn_left` 等；
- validation JSONL 成功写出：`validation/1.jsonl`，大小约 `710930` bytes；
- rollout JSONL 成功写出：`rollout_data/1.jsonl`，大小约 `232195` bytes。

没有修改：

- 数据过滤逻辑；
- metric 计算逻辑；
- train/val/test split 生成逻辑；
- 结果汇总脚本；
- 可视化工具。

是否影响旧结论：本次只修正新服务器路径和 Cambrian 兼容性，不足以推翻已有 Cambrian 实验结论。但若旧实验也运行在新版 Transformers 环境中，则 Qwen2 RoPE/logits 修复会影响它们能否正常运行。

## 6. 已执行的实验

### Cambrian smoke prechecks

目的：确认 tokenizer、processor fallback、external_lib、checkpoint、SigLIP cache、vLLM plugin 是否可用。

结果：逐步暴露并修复了缺源码、缺依赖、缺 SigLIP cache、`num_channels` 等问题。

状态：已完成。

### `c8_fwdfirst_rewscale_smoke_server` 多轮 rerun

目录：

```text
exps/vagen_active_spatial/c8_fwdfirst_rewscale_smoke_server/
```

模型：

```text
/mnt/umm/users/yinbaiqiao/hf_cache/cambrian-s-7b
```

外部源码：

```text
/mnt/umm/users/yinbaiqiao/cambrian-s
```

数据：Active Spatial smoke config，`gs_root=/mnt/umm/users/yinbaiqiao/InteriorGS`，validation size 1。

失败阶段汇总：

- 早期：依赖、SigLIP cache、scene path、GPU ordinal、batch/chunk；
- rerun12：RoPE shape 问题修复后进入 logprob，但失败于 2D logits；
- rerun13：移除 decoder squeeze 后仍失败于外层 2D logits；
- rerun14：外层 logits 维度修复后成功完成。

### 最终成功实验：rerun14

启动：

```bash
nohup bash ./run_cambrian_smoke_server.sh > cambrian_smoke_server_rerun14.log 2>&1 &
```

结果：

- 训练完成：`Training Progress: 100%|...| 1/1`
- 训练耗时：日志显示约 `592.24s/it`
- 写出 rollout：`rollout_data/1.jsonl`
- 写出 validation：`validation/1.jsonl`
- 保存 checkpoint：`checkpoints/global_step_1/actor/`
- 保存 HF config/tokenizer：`checkpoints/global_step_1/actor/huggingface/`
- 保存 4 个 FSDP rank 的 model/optim/extra_state 分片。

可信度：作为 smoke test 可信，证明链路可跑通；不作为模型质量结论。

## 7. 实验结果与主要发现

最终 rerun14 关键指标来自 `train.log`：

```text
training/global_step: 1
actor/entropy: 0.9899821877479553
train/traj_success/mean: 0.046875
train/traj_success/sum: 3.0
train/traj_success/count: 64
val-core/active_spatial/reward/mean@1: -1.1175401210784912
val-aux/active_spatial/traj_success/mean@1: 0.0
response_length/mean: 128.0625
prompt_length/mean: 1345.9375
response/aborted_ratio: 0.0
perf/total_num_tokens: 94336
perf/throughput: 93.41965165319327
timing_s/gen: 163.38
timing_s/old_log_prob: 37.46
timing_s/ref: 51.52
timing_s/testing: 236.79
timing_s/save_checkpoint: 102.81
```

主要发现：

- Cambrian-S 在当前服务器和当前环境中可以完成完整 1-step PPO smoke；
- vLLM rollout、FSDP actor logprob、ref logprob、validation、checkpoint 保存都已经跑通；
- `flash_attn` fallback 后性能偏保守，完整训练速度可能较慢；
- validation 样本量为 1，性能指标只能作为链路验证，不能作为模型能力结论；
- `wandb` 退出 BrokenPipe 发生在完成后，不影响本次结果文件可信度。

## 8. 对项目设计的分析与改进思路

已实现：

- Cambrian-S Qwen2 decoder API shim；
- Cambrian-S 2D hidden/logits 兼容；
- Cambrian vLLM SigLIP config 修复；
- 新服务器 smoke launcher 和 smoke env config；
- 1-step smoke 验证闭环。

值得尝试：

- 处理 `flash_attn` GLIBC mismatch，提高完整训练速度；
- 将 Cambrian smoke config 固化为文档化入口，避免依赖手工环境变量；
- 增加 Cambrian adapter 的单元/集成测试：processor、image token expansion、actor forward、vLLM init；
- 做 50-step diagnostic run，观察 reward/action-format 是否稳定；
- 增加 frozen rollout/action-format audit，确认 Cambrian 初始策略是否满足 `<action>...</action>`；
- 对 Cambrian 长训练继续使用 `c8_fwdfirst_rewscale.sh` 风格，而不是早期自由思考 prompt。

暂时缺少证据：

- 当前修复是否覆盖多图输入；
- 长上下文、多轮 concat、更大 batch 是否有新的 position/mask 边界问题；
- 当前 validation 指标是否能代表真实泛化能力。

## 9. 当前项目状态

当前代码在训练节点上可以跑通 Cambrian-S 1-step smoke。

推荐配置：

- 使用 `run_cambrian_smoke_server.sh`；
- 实验脚本用 `c8_fwdfirst_rewscale_smoke_server.sh`；
- 基线长跑从 `c8_fwdfirst_rewscale.sh` 风格继续扩展。

可信结果：

- 新服务器环境已配置到能训练 Cambrian-S；
- vLLM rollout 和 FSDP actor/ref logprob 已跑通；
- validation JSONL、rollout JSONL、checkpoint 写出可信。

需要重新验证：

- 50-step 或更长短跑；
- 完整 validation set；
- 若切回更大 batch、更多 trajectory 或不同 prompt，需要重新 smoke。

当前最严重阻塞：

- 无明确 P0 阻塞；1-step smoke 已完成；
- 性能风险来自 `flash_attn` fallback 和 Cambrian 8B 训练较慢。

技术债务：

- Cambrian adapter shim 是兼容性补丁，建议后续加测试和注释文档；
- `wandb` offline 退出 BrokenPipe 可清理，但不阻塞；
- 训练节点上的改动尚未确认是否已同步回主工作区或纳入版本管理。

## 10. 待办事项

### P0：必须立即处理

1. 确认远程代码改动纳入版本管理或同步回主仓库。
   - 原因：当前关键修复在训练节点工作区。
   - 完成标准：`cambrian_register.py`、`cambrian_vllm.py`、smoke scripts/env config 都可在主开发环境复现。
   - 验证：重新运行 `py_compile` 和 1-step smoke。

2. 清理/确认 `wandb` 退出 BrokenPipe。
   - 原因：虽然不影响结果，但可能干扰自动化判断。
   - 推荐：smoke 使用纯 console logger 或更彻底 offline/disabled wandb。
   - 完成标准：训练结束无 atexit traceback。
   - 验证：再跑 1-step smoke。

### P1：下一轮实验前应完成

1. 运行 50-step diagnostic。
   - 原因：1-step 只能证明链路，不证明训练稳定性。
   - 完成标准：无 NaN/OOM，reward/action-format/entropy 可解释。
   - 验证：保存 checkpoint、rollout、validation，并检查 action format。

2. 扩大 validation。
   - 原因：当前 `VAL_N=1` 不足以评估性能。
   - 完成标准：使用稳定 val split，输出可比较指标。
   - 验证：对比 Qwen2.5-VL baseline 或旧 Cambrian c8。

3. 处理 `flash_attn` fallback 或记录性能预期。
   - 原因：当前 eager/native attention 可能显著降低训练速度。
   - 完成标准：要么修复 GLIBC/flash_attn，要么在实验记录中明确性能限制。
   - 验证：比较 `timing_s/old_log_prob`、`timing_s/ref`、throughput。

### P2：研究增强项

1. Cambrian frozen rollout/action-format audit。
   - 原因：历史 Cambrian 有 action prior 问题。
   - 完成标准：统计 no-action-tag、动作分布、turn bias。
   - 验证：固定模型 rollout，不训练。

2. Cambrian adapter 测试。
   - 原因：当前修复覆盖关键路径，但缺自动化。
   - 完成标准：最小 processor、vLLM init、actor forward logprob 测试。
   - 验证：CI 或本地 smoke test。

3. 做 Qwen2.5-VL 与 Cambrian-S 短跑对比。
   - 原因：判断 Cambrian 视觉 backbone 是否带来收益。
   - 完成标准：相同数据、reward、prompt、step 数。
   - 验证：50-step 或更长 diagnostic 指标对齐。

## 11. 关键文件索引

| 文件或目录 | 本次作用 | 是否修改 | 主要改动或内容 | 当前状态 |
| ----- | ---- | ---: | ------- | ---- |
| `vagen/models/cambrian_register.py` | Cambrian FSDP actor/ref/critic 适配 | 是 | Qwen2 RoPE shim、2D hidden/logits batch 维修复 | rerun14 已验证 |
| `vagen/models/cambrian_vllm.py` | Cambrian vLLM rollout 适配 | 是 | SigLIP nested `vision_config` / `num_channels` 修复 | vLLM 已启动验证 |
| `vagen/models/cambrian_processor.py` | Cambrian processor wrapper | 未确认修改 | `<image>` token、SigLIP preprocessing 路径被使用 | 参与 smoke |
| `vagen/models/cambrian_plugin.py` | vLLM plugin entry | 未确认修改 | 注册 Cambrian vLLM model | 参与 smoke |
| `vagen/agent_loop/agent_loop_no_concat.py` | Active Spatial agent loop | 未确认修改 | Cambrian image token expansion / pixel_values 路径 | 参与 smoke |
| `examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_smoke_server.sh` | smoke 实验入口 | 是 | batch、step、model path、GPU mem、val 设置 | rerun14 已验证 |
| `run_cambrian_smoke_server.sh` | smoke launcher | 是 | 环境变量、CUDA_VISIBLE_DEVICES、HF cache、CAMBRIAN_SRC | rerun14 已验证 |
| `examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_smoke_server.yaml` | smoke env config | 是 | `gs_root` 改到新服务器 InteriorGS，val size 缩小 | 已验证 |
| `/mnt/umm/users/yinbaiqiao/cambrian-s` | Cambrian 外部源码 | 新增/准备 | 提供 `cambrian` import | 已验证 |
| `/mnt/umm/users/yinbaiqiao/hf_cache/cambrian-s-7b` | Cambrian checkpoint | 使用 | 模型权重/tokenizer/config | 已验证 |
| `/mnt/umm/users/yinbaiqiao/.cache/huggingface` | HF/SigLIP cache | 使用/补齐 | SigLIP/SigLIP2 vision tower cache | 已验证 |
| `exps/vagen_active_spatial/c8_fwdfirst_rewscale_smoke_server/train.log` | 主训练日志 | 生成 | rerun14 完整日志和指标 | 成功 |
| `exps/vagen_active_spatial/c8_fwdfirst_rewscale_smoke_server/validation/1.jsonl` | validation 输出 | 生成 | 1-step 后 validation generations | 成功 |
| `exps/vagen_active_spatial/c8_fwdfirst_rewscale_smoke_server/rollout_data/1.jsonl` | rollout 输出 | 生成 | 训练 rollout generations | 成功 |
| `exps/vagen_active_spatial/c8_fwdfirst_rewscale_smoke_server/checkpoints/global_step_1/actor/` | checkpoint | 生成 | FSDP actor model/optim/extra/hf files | 成功 |
| `cambrian_smoke_server_rerun14.log` | launcher 日志 | 生成 | 最终成功 rerun 日志 | 成功 |

## 12. 一页式总结

本次工作完成了 Cambrian-S 在新 8 卡服务器上的配置、调试和 1-step PPO smoke 验证。主要修复包括：Cambrian 外部源码和依赖配置、SigLIP cache、vLLM SigLIP config、Active Spatial 数据路径、GPU 可见性、batch/chunk 设置，以及 Cambrian-S 与新版 Transformers Qwen2 decoder 的 RoPE/`position_embeddings` 兼容问题。

代码层面最关键的改动在 `vagen/models/cambrian_register.py` 和 `vagen/models/cambrian_vllm.py`：前者修复 FSDP actor/ref logprob 中的 RoPE 和 2D logits 问题，后者修复 vLLM SigLIP config 问题。配置层面新增/调整了 smoke experiment、launcher 和 smoke env config。

最终 rerun14 成功完成 `1/1` training step，写出 `validation/1.jsonl`、`rollout_data/1.jsonl`，并保存 `checkpoints/global_step_1/actor/`。这说明 Cambrian-S 的 tokenizer/processor、vLLM rollout、FSDP actor/ref logprob、validation 和 checkpoint 保存链路已经跑通。

当前结果只证明 smoke 可运行，不证明模型效果。下一步最应该做的是：把远程修复同步进正式代码管理，清理 wandb 退出噪音，然后运行 50-step diagnostic 和更完整 validation，对比 Qwen2.5-VL baseline。
