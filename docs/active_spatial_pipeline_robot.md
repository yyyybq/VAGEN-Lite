# 第 3 阶段：StarVLA 动作学习与 RLinf 在线 RL

这一阶段把 Active Spatial HF actor 变成 RoboCasa365 的连续动作 policy，再在机器人环境中继续优化。[上一步：Active Spatial RL](active_spatial_pipeline_rl.md) · [返回总览](active_spatial_end_to_end.md)。

## 先用机器人示范训练动作头

主入口是 `prepare_starvla_transfer.py` 和 `run_starvla_transfer.sh`。`examples/train/robocasa/run_sft.sh` 是把 12 维动作写成文本的 LLaMA-Factory 路线，不是本流程的连续动作头。

```bash
export ROBO_DATA=/absolute/path/to/robocasa365/datasets
"$STARVLA_PY" scripts/prepare_starvla_transfer.py \
  --dataset NavigateKitchen="$ROBO_DATA/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot" \
  --dataset PickPlaceCounterToCabinet="$ROBO_DATA/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot" \
  --source base="$BASE_HF" --source mix_sft="$SFT_HF" --source mix_sft_rl="$SFT_RL_HF" \
  --seed 42 --val-fraction 0.1 --steps 10000 --batch 2 \
  --out "$RUN/robot_sft"

CHECK_ONLY=1 bash examples/train/robocasa/run_starvla_transfer.sh "$RUN/robot_sft/mix_sft_rl.yaml"
NUM_GPUS=8 bash examples/train/robocasa/run_starvla_transfer.sh "$RUN/robot_sft/mix_sft_rl.yaml"
```

路径应指向各任务真正存在的 LeRobot v2 目录。这里同时示范 pretrain 导航和 target 操作数据；如果使用 target 示范训练，target 评测不能称为 zero-shot。要严格研究场景迁移，应给所有分支统一选择 pretrain 示范，另冻结未见 layout/object 测试集。

准备器验证完整 HF 权重、模型家族/尺寸、processor 和 `🔍` action query 的单 token 条件，按 episode 划分 train/val，建立只读源数据的链接和可写 metadata overlay。统计量只从训练 episode 计算。`base`、`mix_sft` 等只是来源标签，不会自动证明这些模型做过对应训练，必须保留前两阶段的 manifest。

分别启动其他来源的 YAML，以得到相同训练预算的对照。一个 StarVLA checkpoint 的可迁移单位是整个 run 所需的文件：

```text
robot_sft/runs/mix_sft_rl/
├── config.yaml
├── config.full.yaml                 # 若该训练器版本生成
├── dataset_statistics.json
└── checkpoints/*.pt 或 final_model/pytorch_model.pt
```

`config.yaml` 中的 `framework.qwenvl.base_vlm` 也要保持可访问。模型加载器会先构建 VLM，再加载 `.pt` 参数。只复制一个 `.pt` 文件不够；`export_starvla_backbone.py` 会去掉动作头，也不能拿它替代完整 policy 进入机器人 RL。

## 核对机器人接口

| 项目 | 本地 PandaOmron QwenOFT 合同 |
| --- | --- |
| 模型动作维度 | 12 |
| 顺序 | eef position 3、eef rotation 3、gripper 1、base motion 4、control mode 1 |
| LeRobot 原始 packed action | base 4、mode 1、eef 3、rotation 3、gripper 1；通过 modality.json 重排 |
| 相机输入顺序 | left、right、wrist |
| 图像 | RoboCasa 原始渲染 256；按训练配置进行 PIL resize，常见 224 |
| 动作归一化 | 12 维全部 continuous min/max，包括 gripper 和 mode |
| action horizon | 从 checkpoint 读取；本地训练为 16 |
| state | 本流程 `include_state=false`；RLinf 的 OFT 路径也不消费 state |
| statistics key | 从 `dataset_statistics.json` 读取；本地真实 checkpoint 为 `new_embodiment` |

12 维是 OSC 控制器命令，不是关节力矩。RoboCasa 与 LIBERO 的夹爪映射不同；不能保留 RLinf 的 `policy_setup: libero`，也不能保留 7D、8-step 或 `franka` 默认值。

## 先验证动作拟合和执行

```bash
export CKPT="$RUN/robot_sft/runs/mix_sft_rl/final_model/pytorch_model.pt"
"$STARVLA_PY" scripts/eval_starvla_heldout.py --ckpt "$CKPT" --out "$RUN/robot_heldout"

# 分别在 policy server 和 RoboCasa 客户端环境/终端启动
CKPT="$CKPT" bash examples/evaluate/robocasa/run_starvla_mobile.sh server
CKPT="$CKPT" bash examples/evaluate/robocasa/run_starvla_mobile.sh client \
  --task NavigateKitchen --task PickPlaceCounterToCabinet --split target \
  --episodes 50 --seed 1042 --max-steps 512 --execute-steps 16 \
  --video --out "$RUN/robot_closed_loop_before_rl"
```

服务器运行的是长驻进程，两条启动命令应在不同终端执行。先按 [现有渲染说明](robocasa_vlm_sft_eval.md) 运行 `run_starvla_mobile.sh preflight`；本机已有 OSMesa/EGL 隔离方式。

held-out action error、真实闭环成功率、空间能力保持要分别报告。StarVLA trainer 的 `mse_score` 来自训练 batch，不是 held-out 验证。较小动作误差不保证任务成功。

## 准备 RLinf 的实际连接配置

本地 `third_party/RLinf` 已有 StarVLA adapter 和 RoboCasa365 环境，现有主示例分别是 LIBERO+StarVLA 和 RoboCasa+OpenPI。本次增加父仓库入口，把两者按真实 checkpoint 合同连接。

先确保独立 RLinf 环境能导入本地 RLinf、StarVLA 和 RoboCasa365。父仓库忽略 `third_party/`；新机器必须保留匹配 revision 和本地修改。本次 RLinf 修复单独保存在 `examples/train/robocasa/patches/rlinf_robocasa_starvla.patch`，基于 `c70606f08cdca259b8dec03d4430926b5b8fac9d`：

```bash
# 新机器且补丁尚未应用时执行；当前工作区已经应用
git -C third_party/RLinf apply --check "$ROOT/examples/train/robocasa/patches/rlinf_robocasa_starvla.patch"
git -C third_party/RLinf apply "$ROOT/examples/train/robocasa/patches/rlinf_robocasa_starvla.patch"
```

StarVLA 的既有迁移补丁是 `examples/train/robocasa/patches/active_spatial_transfer.patch`，基于 `4507931a625c844404c4a76128fb116536d8ca7c`。不要在已应用或不同 revision 上盲目重复应用；父仓库的 CPU 测试和真实 dataloader 检查应在迁移后重跑。

```bash
"$ACTIVE_PY" scripts/prepare_rlinf_robocasa.py prepare \
  --ckpt "$CKPT" --task NavigateKitchen --task PickPlaceCounterToCabinet \
  --train-split pretrain --eval-split target \
  --steps 2 --num-envs 8 --episode-steps 512 --out "$RUN/robot_rl_smoke"

export RLINF_PY=/absolute/path/to/rlinf/environment/bin/python
CHECK_ONLY=1 bash examples/train/robocasa/run_rlinf_starvla.sh "$RUN/robot_rl_smoke/train.yaml"
bash examples/train/robocasa/run_rlinf_starvla.sh "$RUN/robot_rl_smoke/train.yaml"
```

准备器读取 `.pt` 对应的 `config.yaml`、统计量、训练 image size 和 horizon，生成新的 PPO recipe 与 `handoff.json`，绑定 actor/rollout 到同一 `.pt`。有多个 statistics key 时显式指定 `--unnorm-key`；不会猜 `franka`。输入是 metadata-only HF 目录、错误动作维度、混合 embodiment、state-conditioned OFT 或冲突 horizon 时会拒绝。

新 recipe 使用 GAE＋actor/critic loss，增加 value head，学习率起始值为 `1e-6`；RLinf wrapper 在连续动作均值周围加 Gaussian exploration，并缓存对应的 normalized action/logprob 供 PPO 回放。它不是继续训练文本 next-token loss。

`CHECK_ONLY=1` 验证 checkpoint/config 身份和 Hydra defaults 展开，不启动 GPU 或模拟器。正式启动才会验证实际模型、Ray、环境、梯度、optimizer 和 checkpoint；按机器调整并重新生成实验配置。该入口固定校验已准备的配置，不接受静默修改后的 YAML。

## 短程通过后再扩大预算

至少验证一次完整 rollout/update/save/reload，并检查以下行为：

* reset 后的三相机图像方向和顺序与示范一致，指令来自环境真实 episode。
* 模型输出 `(B,16,12)`，actor/rollout logprob 回放一致且 finite，梯度和参数确实改变。
* base/torso/mode 未被关闭，夹爪没有经过 LIBERO 的二值翻转。
* 训练和评测使用不同 seeds；对照分支使用同一组评测 seeds。
* 动作 horizon 与执行步数一致。本 recipe 每次执行整个 chunk；独立闭环评测应使用相同 `--execute-steps 16`。

`episode_horizon_source: max_episode_steps` 明确使用 512 步预算，便于短程检查；这不是官方完整任务 horizon 的成绩。正式报告应根据任务统一更长预算或使用完整 benchmark horizon，并保证 rollout 足够长。不要让短程截断和模型失败混在同一个成功率结论中。

RLinf 恢复使用其自己的 `runner.resume_dir` 和 `checkpoints/global_step_N/actor`，不是把该目录传给 StarVLA 的 `.pt` loader。需要修改预算或恢复训练时，应使用独立、审查过的 RLinf 配置并保留原 handoff。

本 recipe 显式开启 `save_full_model_weights`。RL 训练结束后，可以从完整 actor 导出供确定性评测使用的 StarVLA policy：

```bash
"$RLINF_PY" scripts/export_rlinf_starvla.py \
  --weights /absolute/rlinf/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt \
  --source-ckpt "$CKPT" --out "$RUN/after_robot_rl_policy"
export RL_POLICY="$RUN/after_robot_rl_policy/checkpoints/policy.pt"
"$STARVLA_PY" scripts/export_starvla_backbone.py \
  --ckpt "$RL_POLICY" --out "$RUN/after_robot_rl_hf"
```

导出器逐项核对原 StarVLA 参数名和形状，保留训练后的 VLM 与动作头，剥离 RLinf 的 `starvla_model.` 前缀。value head、Gaussian log-std 和 optimizer 不进入部署 policy，因此只用于确定性动作均值评测；恢复 RL 仍用原 checkpoint。DCP shards、LoRA 或额外 wrapper 不会被自动猜测转换。将 `$RL_POLICY` 交给上面的 held-out 和闭环入口，将 `after_robot_rl_hf` 交给空间评测。

StarVLA SFT 后的空间保持评测已可执行：

```bash
"$STARVLA_PY" scripts/export_starvla_backbone.py --ckpt "$CKPT" --out "$RUN/after_robot_sft_hf"
```

将导出的 HF 目录交给上一页相同空间评测配置，只换 checkpoint 和输出目录；同时保留完整 StarVLA policy 供机器人使用。最终验收见 [审计清单](active_spatial_pipeline_audit.md)。
