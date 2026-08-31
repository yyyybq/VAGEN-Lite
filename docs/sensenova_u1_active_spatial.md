# SenseNova-U1 Active Spatial 实验接入指南

创建时间: 2026-07-25

本文记录如何把 SenseNova-U1 接入 VAGEN-Lite 的 Active Spatial 训练框架，以及后续在新机器上配置、smoke、诊断实验的步骤。

## 1. Active Spatial 在 VAGEN-Lite 中如何训练

训练入口是：

```bash
examples/train/active_spatial/run_experiment.sh
```

它做四件事：

1. 读取 `examples/train/active_spatial/experiments/*.sh` 中的实验变量。
2. 将旧式 `env_config_*.yaml` 转成 VAGEN-Lite 的 `train.yaml` / `val.yaml`。
3. 调用 `python -m vagen.main_ppo --config-name=vagen_multiturn`。
4. 使用 async vLLM rollout + `agent_loop_no_concat` 与环境多轮交互。

环境主体在：

```text
vagen/envs/active_spatial/env.py
vagen/envs/active_spatial/active_spatial_env.py
vagen/envs/active_spatial/prompt.py
```

每一轮 observation 由当前渲染图像和文字反馈组成，返回为：

```python
{
    "obs_str": "...",
    "multi_modal_input": {"<image>": [PIL.Image, ...]},
}
```

模型生成原始文本后，环境只解析 `<action>...</action>` 中的动作。奖励来自格式、进度势能、碰撞/无效动作、成功等规则；`<think>` 内的预测文字本身不被环境直接执行，但会进入训练序列并影响策略学习。

## 2. Cambrian-S 每步如何同时输出动作和预测

Cambrian-S 没有单独的“预测 API”。它每步同时输出预测和动作，靠的是 prompt / SFT / RL 的文本格式约束：

```text
<think>当前状态分析、预计下一步会提高/降低 score 或 progress</think>
<action>move_forward|</action>
```

相关位置：

```text
vagen/envs/active_spatial/prompt.py
data_gen/active_spatial_sft/output_100scenes_5k/sft_data_part2.jsonl
```

Cambrian-S 的特殊性主要在模型接入层：

```text
vagen/models/cambrian_register.py
vagen/models/cambrian_processor.py
vagen/models/cambrian_plugin.py
vagen/models/cambrian_vllm.py
vagen/agent_loop/agent_loop_no_concat.py
```

它把 rollout 侧的紧凑 `<image>` token，在训练/logprob 侧展开成 756 个 Cambrian 内部图像 token，并把 SigLIP pixel values 放入 `multi_modal_inputs`。

## 3. SenseNova-U1 接入内容

本仓库新增了 U1 的 VAGEN 接入层：

```text
vagen/models/sensenova_u1_processor.py
vagen/models/sensenova_u1_register.py
vagen/models/sensenova_u1_plugin.py
examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_smoke.sh
```

U1 与 Cambrian 类似，也不是标准 `AutoProcessor` 路径。U1 的图像理解流程是：

1. 用 U1 repo 的 `load_image_native()` 将 PIL 图像 patchify。
2. 得到 `pixel_values` 和 `grid_hw`。
3. 将文本 `<image>` 展开为 `<img><IMG_CONTEXT>*N</img>`。
4. forward 中根据 `grid_hw` 生成 THW 位置索引。
5. 使用 U1 原生 `extract_feature()` 得到视觉 embedding 并写入 `<IMG_CONTEXT>` token 位置。

`vagen.models.sensenova_u1_register` 还补了 U1 upstream 没有开放的训练 forward，使 PPO actor/ref logprob 和 critic 初始化有可调用路径。

## 4. 已准备好的本地资源

当前服务器上已存在：

```text
/nas/baiqiao/SenseNova-U1
/nas/baiqiao/models/SenseNova-U1-8B-MoT-SFT
```

U1 官方 full-parameter 微调 smoke 已在 U1 repo 中跑通过 1 step：

```bash
cd /nas/baiqiao/SenseNova-U1/training
conda run -n vllm bash training/shell/train_u1/8B_local_smoke.sh
```

日志：

```text
/nas/baiqiao/SenseNova-U1/training/RUN/sensenovau1_8b_local_smoke/07-24-15.08.35/logs/main_dp=0_wp=0_pp=0
```

## 5. 环境配置步骤

### Step 1: 安装 VAGEN-Lite

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
pip install -e .
```

安装后，vLLM 子进程才能加载 `vagen.models.*_plugin` entry point。

本机推荐训练解释器：

```bash
export PYTHON=/data/baiqiao/miniconda3/envs/vagen/bin/python
```

`run_experiment.sh` 已支持 `PYTHON` 环境变量覆盖；如果迁移到别的 conda 环境，只需要改这个变量。

### Step 2: 配置 U1 源码路径

```bash
export SENSENOVA_U1_SRC=/nas/baiqiao/SenseNova-U1/src
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH}"
export SENSENOVA_U1_MODEL_PATH=/nas/baiqiao/models/SenseNova-U1-8B-MoT-SFT
export ACTIVE_SPATIAL_ROOT=/nas/baiqiao/active_spatial
```

`ACTIVE_SPATIAL_ROOT` 用来把旧脚本中的 `/scratch/by2593/project/Active_Spatial/...` 自动重映射到本机路径。

### Step 3: 先跑 U1 VQA 单样本

确认 U1 原生理解路径可用：

```bash
cd /nas/baiqiao/SenseNova-U1
python examples/vqa/inference.py \
  --model_path /nas/baiqiao/models/SenseNova-U1-8B-MoT-SFT \
  --image examples/vqa/data/images/demo_1.jpg \
  --question "Describe the image briefly." \
  --max_new_tokens 64 \
  --attn_backend sdpa
```

如果这里失败，先修 U1 repo / 权重 / CUDA 环境，不要进入 PPO。

### Step 4: 跑 Active Spatial 1-step smoke

```bash
cd /nas/baiqiao/active_spatial/VAGEN-Lite
bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_smoke.sh
```

这个脚本默认：

```text
TOTAL_STEPS=1
TRAIN_BATCH_SIZE=1
VAL_BATCH_SIZE=1
N_TRAJECTORY=1
MAX_TURNS=2
```

### Step 5: 检查 smoke 日志

重点搜索：

```bash
rg -n "sensenova_u1_register|SenseNova-U1|neo_chat|unsupported|NotImplemented|IMG_CONTEXT|image-token mismatch|validation" exps/vagen_active_spatial/u1_fwdfirst_rewscale_smoke/train.log
```

期望看到：

```text
[sensenova_u1_register] Registered SenseNovaU1ForCausalLMAdapter ...
```

如果卡在 vLLM unsupported architecture，这是 rollout 侧缺 U1 vLLM wrapper，不是环境或 FSDP adapter 问题。下一步需要实现 U1 的 `SupportsMultiModal` vLLM model wrapper，类似 `vagen/models/cambrian_vllm.py`。

## 6. 从 smoke 到正式实验

smoke 通过后，复制脚本：

```bash
cp examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_smoke.sh \
   examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_50step.sh
```

建议先改成：

```bash
TOTAL_STEPS="50"
SAVE_FREQ="10"
TEST_FREQ="10"
MAX_TURNS="12"
TRAIN_BATCH_SIZE="4"
VAL_BATCH_SIZE="2"
N_TRAJECTORY="2"
PPO_MINI_BATCH_SIZE="2"
```

50 step 重点看：

```text
action tag 合规率
move_forward / turn_left / turn_right 分布
invalid_format_penalty 占比
score / success / entropy 是否稳定
```

Cambrian 历史显示，新模型最容易出现“无 `<action>` 标签”和“反复旋转”两类冷启动问题。U1 也应先做冻结 rollout/action-format audit，再扩大 PPO。

## 7. 当前限制

本次接入已经覆盖：

```text
U1 源码注册
U1 processor wrapper
U1 actor/ref forward adapter
U1 critic token-classification adapter
agent_loop_no_concat 中的 U1 图像 token 展开
Active Spatial smoke 脚本
```

仍需通过 smoke 验证或继续补齐：

```text
vLLM 对 NEOChatModel 的 rollout 支持
U1 多 GPU TP 下的 KV cache/位置索引行为
长轨迹下动态图像 token 对 max_prompt_length/max_num_batched_tokens 的压力
critic value head 保存/恢复
```

如果 vLLM 不支持 U1，优先级最高的是实现 `vagen/models/sensenova_u1_vllm.py`，把 U1 的 `<img><IMG_CONTEXT>` prompt expansion、`pixel_values/grid_hw` 预处理、`extract_feature()` embedding merge 搬到 vLLM `SupportsMultiModal` 接口里。

## 6. Plan B phase-1 (open-loop FM aux)

Locked design:

- Open-loop: predicted frames are for aux loss / viz only; env obs always from renderer.
- Rollout: text-only `<action>...</action>` (no image generation at sample time).
- Train aux: split und-prefix + gen forward (HF U1 decoder does not allow mixed und/gen in one pass).
  Next-frame GT arrives as `u1_gen_pixel_values` / `u1_gen_grid_hw` / `u1_gen_valid` in `multi_modal_inputs`.
- Coefficient: `actor_rollout_ref.actor.image_gen_loss_coef=0.1`.
- Smoke: `examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke.sh`
  Use free GPUs only, e.g. `CUDA_VISIBLE_DEVICES=5,6,7`.

Unit-tested on GPU5: FM aux forward+backward OK (`loss≈0.07` on synthetic frames).

