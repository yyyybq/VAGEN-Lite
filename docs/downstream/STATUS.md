# 下游具身任务线状态（唯一的状态文档）

最后更新：2026-10-20（第一步完成）。任务书分四步，每步做完停下来等确认。

## 1. 现在在哪一步

- **第一步（盘点与核实）已完成，停下等你确认。** 报告：[reports/STEP1_INVENTORY_20261020.md](reports/STEP1_INVENTORY_20261020.md)，证据在 [reports/step1/](reports/step1/)。
- 第一步的唯一 GPU 用量：1 卡 sco 任务 `pt-405nyxoy`（加载测试 + 渲染检查），已完成（退出码 0），卡已释放。
- **下一步等什么**：你对报告末尾 D1–D8 的决定；第二步（连通试跑，一个 8 卡任务）在你确认后才提交。

## 2. 谁在跑

| sco 任务 | 内容 | 状态 | 目录 |
| --- | --- | --- | --- |
| `pt-405nyxoy` | 第一步加载测试 + 渲染检查（1 卡 H800） | 已完成（10-20 13:30–13:35 UTC，退出码 0）；现在没有本线的任务在跑 | `/mnt/umm/users/yinbaiqiao/downstream_runs/step1_load_test_20261020/` |

## 3. 代码与位置

- 代码与文档：集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream`，分支 `downstream`（从 main `59053083` 开出）；只新增 `downstream/`、`docs/downstream/`。
- 大文件（实验产物）：`/mnt/umm/users/yinbaiqiao/downstream_runs/`，不进 git。
- 只读引用（不改）：StarVLA `VAGEN-Lite/third_party/starVLA`（`4507931` + 本地改动）、RLinf `VAGEN-Lite/third_party/RLinf`（`c70606f` + 补丁）、RoboCasa 1.0.1 `probe_spatial/robocasa365/robocasa`、渲染库覆盖层 `VAGEN-Lite/playground/Runtime/egl`。

## 4. 权重与数据路径

起点登记在 [downstream/starts.json](../../downstream/starts.json)。集群路径前缀 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/`。

| 名字 | 用途 | 路径 | 格式 / 精度 |
| --- | --- | --- | --- |
| B | 原始 Qwen2.5-VL-7B-Instruct（`cc594898`） | `interiorgs_v10_r11_20261016/inputs/base_7b` | 完整 HF，bf16 |
| S | 第十一轮动作 SFT 第 9 个 epoch | `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386` | 完整 HF，bf16 + fp32 混合（配置 bf16） |
| RS | P-S0 第 300 次 | `interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300/actor/huggingface` | 完整 HF，fp32 |
| RD | P-D 第 150 次 | `interiorgs_v10_train_r11p_20261017/rl_run_pd/checkpoints/global_step_150/actor/huggingface` | 完整 HF，fp32 |
| R10S5（备用） | 第十轮 SFT 第 5 个 epoch | `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770` | 完整 HF，bf16 + fp32 混合 |
| R1（备用） | R1 最终模型 | `R1-clean-Projective-v0/formal/checkpoints/global_step_250/actor/huggingface` | 完整 HF，fp32 |

| 数据 | 路径 | 状态 |
| --- | --- | --- |
| RoboCasa365 target 50 任务人工示教（LeRobot v2.1） | `/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets/v1.0/target/` | 50/50 齐全，25,307 条示教，58 GB |
| RoboCasa365 pretrain 示教 | `.../datasets/v1.0/pretrain/` | 人工 293/300 齐全（缺 4、不完整 3），合成 57/60 |
| RoboCasa 1.0.1 资产 | `/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/robocasa/robocasa/models/assets/` | 6 类都在 |

## 5. 实验表

| 编号 | 步骤 | 起点 | 卡数 | 预计时间 | 状态 | 结果路径 |
| --- | --- | --- | --- | --- | --- | --- |
| STEP1-LOAD | 1 | B / S / RS / RD | 1 | 实测约 5 分钟（排队约 62 分钟） | 已完成，四个起点都通过 | `downstream_runs/step1_load_test_20261020/out/` |

第二步（连通试跑）和第三步（36 个 8 卡任务）的表在你确认第一步之后再填。

## 6. 等你决定的事

见第一步报告末尾：D1 加载精度、D2 种子、D3 第二步规模、D4 第三步数据范围、D5 第三步步数、D6 第三步评测规模、D7 RLinf 环境修补、D8 代码与产物位置。
