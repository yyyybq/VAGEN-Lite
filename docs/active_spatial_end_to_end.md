# Active Spatial → RoboCasa：完整训练路线

这组文档把本仓库的任务生成、轨迹 SFT、QA 混合 SFT、Active Spatial RL、StarVLA 动作 SFT 和 RLinf 机器人 RL 接成一条可追踪的流程。入口、配置和限制依据 **2026-10-09 本地代码**核对；训练效果仍需用真实训练和评测确认。

按以下顺序阅读和执行：

1. [生成任务、轨迹与 QA，并完成混合 SFT](active_spatial_pipeline_data.md)。
2. [从 SFT 权重继续 Active Spatial RL，导出 HF actor](active_spatial_pipeline_rl.md)。
3. [用 StarVLA 学习 RoboCasa 低层动作，再接 RLinf](active_spatial_pipeline_robot.md)。
4. [检查结果、修复清单和验收标准](active_spatial_pipeline_audit.md)。

## 训练过程中到底传递什么

```text
冻结的 Active Spatial 任务、场景、相机和成功条件
        ├─ 搜索 + 真实环境回放 + 渲染 ─→ 轨迹对话
        └─ 多个相机状态 + 同一成功判据 + 渲染 ─→ Yes/No QA
                              ↓ 共用场景划分
Qwen2.5-VL base ─→ 轨迹/QA 混合 SFT ─→ 完整 HF VLM
                                          ↓ VAGEN/verl PPO
                                 Active Spatial HF actor
                                          ↓ 初始化 StarVLA 的 VLM
RoboCasa LeRobot demos ─────────→ QwenOFT 动作 SFT
                                          ↓ 完整 VLM + 12D 动作头 + 统计量
                                 RLinf / RoboCasa365 PPO
                                          ↓
                        机器人闭环评测 + 空间能力保持评测
```

你的总体理解成立：先训练空间观察和决策能力，再用机器人示范训练一个动作头，最后在机器人环境里继续 RL。需要补充的是，**这不是同一组数据直接贯穿所有 trainer**。

| 阶段 | 输入 | 输出 | 优化目标 |
| --- | --- | --- | --- |
| 任务生成 | 场景、目标物体、任务规则 | 任务 JSONL、初始相机、目标条件 | 不训练模型 |
| 轨迹生成 | 任务 JSONL、权威环境 YAML、渲染器 | 多轮图像与离散动作对话 | 搜索与回放标签 |
| QA 生成 | 同一任务的多个相机状态 | 图像、问题、Yes/No、私有审计字段 | 同一成功判据的监督标签 |
| 混合 SFT | `messages` + `images` | HF VLM | assistant 文本 token 的交叉熵 |
| Active Spatial RL | 原始任务 JSONL、环境、HF VLM | HF actor / verl checkpoint | 环境成功奖励和 shaping |
| StarVLA SFT | RoboCasa 图像、语言、连续动作示范；空间 HF actor | StarVLA `.pt` + 配置 + normalization stats | QwenOFT 连续动作 L1 回归 |
| RLinf RL | 完整 StarVLA policy；RoboCasa365 在线环境 | RLinf 训练 checkpoint | 连续动作 policy 的 PPO |

Active Spatial 的 `move_forward`、`move_left` 等是相机控制语言；RoboCasa 的输出是 PandaOmron 控制器的 12 维连续命令。不能把离散轨迹中的动作字符串当作机器人动作标签。StarVLA 阶段复用的是 VLM 权重，动作头从示范学习。

## 本次主线选择

主线采用 **Qwen2.5-VL，同一模型规模贯穿所有阶段**，以及当前 R1 canonical 数据。原因是本地 LLaMA-Factory 提供 `qwen2_vl` 模板，本地 StarVLA 已支持 Qwen2.5-VL HF actor。StarVLA 单独也支持 Qwen3-VL，但当前捆绑的 LLaMA-Factory 未注册 `qwen3_vl`，不能把整个流程的模型路径直接替换成 Qwen3-VL。Cambrian 和 SenseNova 也不能直接加载到 QwenOFT。

当前 canonical 成功判据覆盖 `projective_relations`、`fov_inclusion`；当前已冻结的 clean R1 主线主要是 Projective。旧版多任务数据仍可用于独立实验，但不能把旧分数阈值、旧相机协议、旧奖励曲线混进 R1 并统称为同一实验。

新混合入口将轨迹转换成 `no_think`，去掉可能携带 oracle 分数的 think 内容；后续 RL 准备入口同步设置 `prompt_format: no_think`。QA 保持单图 Yes/No。这是这组新实验的明确协议，不追溯改变已有历史实验。

## 用哪些对照验证迁移

建议至少比较下列来源，保持模型家族与规模、机器人示范、episode 划分、动作头随机种子和训练预算相同。

| 来源标签 | 空间阶段 | 后续阶段 |
| --- | --- | --- |
| `base` | 原始 VLM | 相同 RoboCasa SFT → 相同 RLinf RL |
| `act_sft` | 仅轨迹 SFT | 同上 |
| `qa_sft` | 仅 QA SFT | 同上 |
| `mix_sft` | 轨迹＋QA SFT | 同上 |
| `mix_sft_rl` | 混合 SFT＋Active Spatial RL | 同上 |

QA 与轨迹是两个监督分支，不强制要求先 QA、再轨迹。默认顺序是混合 SFT → Active Spatial RL。若先 RL 再做 QA SFT，这是另一个实验分支，需要保留各自 checkpoint 并检查空间行为是否遗忘。

已有 `starvla_qwenoft_NavigateKitchen_20261001_025134` 使用 Qwen3-VL-4B-Instruct-Action，**不是从 Active Spatial 训练过的权重开始**。它可用于验证接口，不能作为空间能力迁移成功的证据。NavigateKitchen 单任务也不能代表抓取、放置和复合任务。

## 其他模型路径

主线用 **Qwen2.5-VL** 贯穿混合 SFT、Active Spatial RL 和 StarVLA。仓库里另外两条 Active Spatial RL 模型路径已经接好 adapter，但**不能直接进入当前 QwenOFT / RLinf 动作头**：

| 模型 | 仓库入口 | 外部依赖（不随 Git 分发） |
| --- | --- | --- |
| Cambrian-S 7B | [接入说明](model_backbone_integration.md)、[新机器 smoke](cambrian_s_new_server_smoke_handoff.md)、[实验记录](cambrian_iteration.md)、`vagen/models/cambrian_*.py`、`examples/train/active_spatial/experiments/c8_fwdfirst_rewscale.sh` | 设置 `CAMBRIAN_SRC` 指向外部 Cambrian-S 源码，并准备独立 checkpoint |
| SenseNova-U1 | [接入说明](sensenova_u1_active_spatial.md)、`vagen/models/sensenova_u1_*.py`、`vagen/models/u1_neo_compat.py`、[1-step smoke](../examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_smoke.sh)、[50-step 诊断](../examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_50step.sh) | 设置 `SENSENOVA_U1_SRC` / `SENSENOVA_U1_MODEL_PATH` |

StarVLA 与 RLinf 也不在父仓库里：clone 到 `third_party/starVLA` 和 `third_party/RLinf`（该目录被 `.gitignore`），再按 [机器人交接](active_spatial_pipeline_robot.md) 打补丁。

## 运行环境分开管理

| 工作 | 环境要求 |
| --- | --- |
| Active 数据生成、VAGEN PPO、HF 导出 | 与该次训练匹配的 `vagen-lite` Python、PyTorch、verl |
| 混合 SFT | 有本地 LLaMA-Factory、Qwen2.5-VL 支持的环境 |
| StarVLA 动作 SFT / policy server | 独立 `starVLA` 环境 |
| RoboCasa 闭环客户端 | 已能运行 RoboCasa365/MuJoCo 的 policy 环境 |
| RLinf 在线 RL | 独立、同时支持 RLinf＋本地 StarVLA＋RoboCasa365 的环境 |

`PYTHONPATH` 指向本地代码不等于依赖已安装。现有 RoboCasa 客户端可跨进程运行；RLinf 的环境 worker 也有自己的运行环境要求。不要直接向已有生产训练环境升级 torch/transformers。

以下各页命令默认从仓库根目录执行，路径变量使用绝对路径。每次准备都使用新输出目录，保留各阶段 manifest、模型配置、数据划分和评测结果。整个流程没有自动启动全部训练的脚本：每阶段先验证输入与输出，再启动对应 trainer，便于在不同 GPU 和模拟器环境间交接。
