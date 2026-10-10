# 全流程审计与验收

这页记录衔接检查发现的问题、实际修改以及尚需真实 GPU/模拟器验证的边界。[返回总览](active_spatial_end_to_end.md)。检查日期：2026-10-09。

## 结论

数据生成、环境 RL、StarVLA SFT、RLinf StarVLA 模型和 RoboCasa365 环境都有代码基础，但此前没有完整约定数据划分、prompt、动作归一化和 checkpoint 交接。现在补上了可复用入口，并修复了可在代码和 CPU 检查中复现的问题。**目前不能宣称空间训练已带来机器人迁移收益，也不能宣称新链路已完成 GPU 端到端训练。**

## 已修复的问题

| 问题 | 影响 | 本次处理 |
| --- | --- | --- |
| QA manifest 没有统一的混合 SFT 入口 | 容易使用无图、无有效标签或带审计字段的数据 | `prepare_active_spatial_sft.py` 筛选监督、校验图片、导出 LLaMA-Factory 数据和配置 |
| QA 与轨迹分别随机划分 | 相同场景/相机状态可跨训练和验证 | 两分支共享 scene split；消融复用 manifest |
| SFT 验证场景被 RL 再次训练 | 后续验证失去独立性 | `prepare_active_spatial_rl.py` 用相同场景 manifest 拆分原始任务 |
| 历史环境 YAML 未启用正式数据准入 | 新训练可能使用未经真实回放验证的数据 | 准备入口强制检查任务/协议绑定的回放证书，两个子集均启用运行时校验 |
| 旧 RL 脚本绑定 pretrained 模型和历史目录 | 名义上 SFT→RL，实际可能从 base 重启 | 新入口替换 actor/critic、输出目录和 schedule，禁用旧任务恢复 |
| action-only SFT 后使用 free-think RL prompt | 监督与运行协议变化 | manifest 记录 `no_think`，RL 环境同步使用 |
| QA runtime compare 丢失 canonical 版本 | 复核可能退回旧分数阈值 | 恢复版本、camera metadata 和评分配置；不一致/空复核返回失败退出码 |
| QA 渲染直接使用原生 K 和另一输出分辨率 | 图像与 canonical 标签的相机几何不一致 | 共用环境 `runtime_render_camera_parameters` 的 H1 resize |
| QA 生成沿用默认评分参数 | 与环境 YAML 阈值不一致 | 支持 `--env-yaml`，保存并复用 `evaluation_config` |
| RLinf 示例默认 LIBERO 7D/8-step/franka | 不能直接用于 PandaOmron 12D/16-step | 从真实 StarVLA checkpoint 生成 RoboCasa365 PPO recipe |
| 把机器人类型当作 normalization key | 本机 key 实际是 `new_embodiment` | 自动读取唯一 key，多 key 时必须指定 |
| 通用反归一化二值化夹爪并裁剪 normalized 动作 | 不符合 RoboCasa365 全维 min/max 逆变换 | `policy_setup=robocasa365` 使用连续逆变换，保留 LIBERO 行为 |
| RLinf checkpoint 与 StarVLA `.pt` 不能互换 | RL 后无法直接复用现有评测 | 保存 full actor；`export_rlinf_starvla.py` 核对 keys/shapes 后导出 deterministic policy，再复用 HF 导出 |
| `third_party` 不受父仓库跟踪 | 搬机器可能丢失修复 | 父仓库保存 RLinf patch，文档记录双方 revision/补丁 |

## 已执行的验证

CPU 回归覆盖混合数据场景隔离、审计字段排除、失效 QA/图片拒绝、SFT→PPO 权重与协议继承、canonical QA camera/判据、RoboCasa 动作、RLinf 配置和参数提取，以及既有 StarVLA 和 RoboCasa 测试：

本次结果：主仓库 **56 passed**；RLinf 针对 RoboCasa/LIBERO 动作的回归 **2 passed**。新增入口通过 Ruff 检查。扩展回归还修正了 QA 测试中的模块别名 mock，使同一测试在整体收集时也能正确隔离真实渲染调用。

```bash
PYTHONPATH="$PWD" /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m pytest -q \
  tests/active_spatial/test_training_handoff.py \
  data_gen/active_spatial_qa/test_qa_contract.py \
  data_gen/active_spatial_qa/test_position_pipeline.py \
  data_gen/active_spatial_qa/test_all_no_diagnosis.py \
  tests/test_starvla_transfer.py vagen/envs/robocasa/tests/test_actions.py \
  tests/active_spatial/test_r1_sft_pipeline.py

PYTHONPATH="$PWD/third_party/RLinf:$PWD/third_party/starVLA" \
  third_party/RLinf/.venv/bin/python -m pytest -q \
  third_party/RLinf/tests/unit_tests/test_models.py \
  -k 'starvla_robocasa or starvla_env_actions_keep'
```

另使用现有 `starvla_qwenoft_NavigateKitchen_20261001_025134/final_model/pytorch_model.pt` 完成真实 metadata 检查、RLinf Hydra 配置展开，并将 StarVLA 原生 `PolicyNormProcessor.unapply_actions` 与修复后的 RLinf 逆变换比较：随机 `(16,12)` 输入含超出 `[-1,1]` 的值，最大绝对误差 **0.0**。该模型是既有 Qwen3-VL baseline，只作为接口检查来源。

独立 RLinf 环境可导入本地 StarVLA adapter 和 framework。当前会话 CUDA 初始化返回 error 304，`torch.cuda.is_available()` 为 false；本次没有启动 GPU rollout、backward 或 optimizer，没有新生成大规模 RGB 语料、启动长程 SFT/RL 或提交集群任务。

## 每阶段需要的真实验收证据

| 阶段 | 通过条件 | 不能替代它的结果 |
| --- | --- | --- |
| 轨迹生成 | 图片与 pose 对应；真实环境回放通过；canonical gates 成功 | 几何搜索分数高 |
| QA | 合法 camera、有效 RGB、判据复核一致、合理标签分布、可见证据充分 | manifest 行数很多 |
| 混合 SFT | 无交叉泄漏，轨迹未不当截断，完整模型保存并评估 | loss 下降 |
| Active RL | 真 rollout/backward/update/save/reload；独立任务成功率 | 脚本启动正常 |
| StarVLA SFT | 指定空间 actor 初始化；episode holdout；动作拟合和真实闭环 | base 导航训练完成 |
| RLinf | normalization parity、forward/logprob replay、更新、保存/恢复、环境成功率 | YAML 能展开 |
| 迁移收益 | 同预算同数据的对照、方差、空间能力保持 | 单 checkpoint 成功案例 |

## 尚需真实实验确认的边界

* R1 canonical 当前只覆盖 Projective/FOV。旧六类/十类任务、旧 reward 和成绩单独标记。
* “有图且非空”不证明 QA 可以从图中判断。需抽样检查目标辨识、可见性与标签；history-dependent QA 当前被混合入口排除。
* `screen_occupancy` 保持 `qa_sft_allowed=false`，已有角度/像素监督冲突没有绕过。
* 本地 LLaMA-Factory 未提供 Qwen3-VL 模板。Qwen3-VL、Cambrian、SenseNova 的全链路需要相应 SFT/backbone 适配。
* RoboCasa episode holdout 不保证 layout/object holdout；用 target 示范训练后的 target 成绩不是 zero-shot。
* 初始化 manifest 记录路径、配置/hash、权重 size/mtime，不是整个大模型的 tensor 内容哈希。归档时保留不可变模型或完整 SHA256 清单。
* RLinf PPO recipe 是连接起点，GPU 数、batch、512 步 horizon 和资源需短程实验确认。旧导航闭环 0/2 只证明能执行，不证明质量。
* full-weights 导出已覆盖 CPU 参数提取测试；新训练结束后还需真实 checkpoint round-trip 和推理一致性测试。导出 policy 去除了探索方差/value head，用于确定性评测，不能替代 RL 恢复文件。

保留生成 manifest、各阶段配置、完整 checkpoint、数据划分和评测 JSON。各阶段证据齐全后，再把该分支记为“完整链路通过”。
