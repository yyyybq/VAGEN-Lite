# 下游具身任务线状态（唯一的状态文档）

最后更新：2026-10-21 05:50 UTC（第二步进行中）。任务书分四步，每步做完停下来等确认。

> **2026-10-20 21:04 – 10-21 约 05:43 UTC：`/mnt/umm/users/yinbaiqiao` 写满**（报 No space left，`df` 显示整盘仍有 4.0 TB，疑似个人配额；训练主线会话负责清理）。期间提交的两个任务一启动就因写不了文件而失败（约 3 秒，记录已删），恢复后已重新提交。写满时本文件在 worktree 里被清空，已从 git 恢复。

## 1. 现在在哪一步

- 第一步（盘点与核实）已完成并经你确认（决定 D1–D8 记在第二步报告开头）。报告：[reports/STEP1_INVENTORY_20261020.md](reports/STEP1_INVENTORY_20261020.md)。
- **第二步（连通试跑）进行中**：
  - 8 卡任务 `pt-189qz570`（20:30–20:56 UTC）完成了：起点 / 数据校验、种子证据（同种子换起点哈希相同，换种子都变）、200 步训练（损失 2.285 → 约 0.13）、留出集动作误差、种子检查（同种子两次重置初始状态相同）。
  - 闭环评测卡住：策略服务读的 `run/config.yaml` 缺 `obs_image_size`（StarVLA 只存训练时读过的配置键，200 步的短训练读不到），仿真进程过不了服务契约检查，进程池无限重启。已手动停掉任务（约 26 分钟 8 卡）。代码已修（`be43f6b2`），这次的检查点用 `fill_serving_config.py` 从 `config.full.yaml` 补齐键（原文件备份为 `config.yaml.before_fill`）。
  - 只重跑闭环评测：8 卡任务 `pt-4qac93lo`（zoetrope 池）在排队。
  - D7 环境已建好（1.9 GB，导入检查全部通过）；RLinf 2 次更新冒烟 `pt-a7r4q7s5`（zoetrope 池）在排队。
- **下一步**：评测重跑满足成功标准后写第二步报告（含第三步完整实验表）。D5 的四个 20 万步任务（按每 2.5 万步存检查点约 540 GB）**等你看过清理后的空间余量、定下检查点间隔再提交**；四个实验目录因写满没建成，提交前重建。

## 2. 谁在跑

| sco 任务 | 内容 | 状态 | 目录 |
| --- | --- | --- | --- |
| `pt-4qac93lo` | 第二步闭环评测重跑（8 卡，zoetrope 池；3 / 12 / 24 个并行仿真进程） | 排队中（10-21 05:47 UTC 提交） | `downstream_runs/experiments/STEP2-RS-s1/step2/eval2/` |
| `pt-a7r4q7s5` | RLinf 2 次更新冒烟（8 卡，zoetrope 池，D7 环境） | 排队中（10-21 05:47 UTC 提交） | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| `pt-9ruo5yrs`、`pt-g0ydp76l` | 上面两个任务的第一次提交 | 10-20 21:04 一启动就因写满失败（约 3 秒），记录已删 | 同上 |
| `pt-189qz570` | 第二步连通试跑 | 10-20 20:56 手动停止（闭环评测卡住，见上），记录已删 | `downstream_runs/experiments/STEP2-RS-s1/` |
| `pt-405nyxoy` | 第一步加载测试 + 渲染检查（1 卡） | 已完成，记录已删 | `downstream_runs/step1_load_test_20261020/` |

登录节点上没有本线的后台进程。

## 3. 代码与位置

- 代码与文档：集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream`，分支 `downstream`（从 main `59053083` 开出，已推送）；只新增 `downstream/`、`docs/downstream/`。worktree 里 `third_party`、`playground` 是指向主检出同名目录的链接（两者都被 .gitignore 忽略），只读使用。
- 大文件（实验产物）：`/mnt/umm/users/yinbaiqiao/downstream_runs/`（`data/` 训练数据目录、`experiments/<编号>/` 实验），不进 git。
- StarVLA：`4507931` + 补丁 [downstream/patches/starvla_4507931_working.patch](../../downstream/patches/starvla_4507931_working.patch)（sha256 `b039782d…`，`git apply --reverse --check` 通过）；每个实验的 `experiment.json` 记提交号和补丁哈希。
- D7 新环境：`/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365`（叠在 starVLA 环境上的 venv，1.9 GB；包清单 `downstream/envs/`，激活 `source downstream/envs/activate_rlinf.sh <ENV>`）。

## 4. 权重与数据路径

起点登记在 [downstream/starts.json](../../downstream/starts.json)，锁定值在 [downstream/starts_lock.json](../../downstream/starts_lock.json)（每次提交前校验）。集群路径前缀 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/`。

| 名字 | 用途 | 路径 | 格式 / 精度 |
| --- | --- | --- | --- |
| B | 原始 Qwen2.5-VL-7B-Instruct（`cc594898`） | `interiorgs_v10_r11_20261016/inputs/base_7b` | 完整 HF，bf16 |
| S | 第十一轮动作 SFT 第 9 个 epoch | `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386` | 完整 HF，bf16 + fp32 混合 |
| RS | P-S0 第 300 次 | `interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300/actor/huggingface` | 完整 HF，fp32 |
| RD | P-D 第 150 次 | `interiorgs_v10_train_r11p_20261017/rl_run_pd/checkpoints/global_step_150/actor/huggingface` | 完整 HF，fp32 |
| R10S5（备用） | 第十轮 SFT 第 5 个 epoch | `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770` | 完整 HF |
| R1（备用） | R1 最终模型 | `R1-clean-Projective-v0/formal/checkpoints/global_step_250/actor/huggingface` | 完整 HF |

训练时所有起点统一按 bf16 加载（D1）。

| 数据 | 清单 | 规模 |
| --- | --- | --- |
| 第二步 | [downstream/demos/manifests/step2_smoke.json](../../downstream/demos/manifests/step2_smoke.json) | 3 个任务各前 100 条示教（训练）+ 后 20 条（留出），训练 44,369 帧 |
| 第三步 100% | [target50_100.json](../../downstream/demos/manifests/target50_100.json) | 50 个任务，25,307 条示教，14,957,899 帧 |
| 第三步 30% | [target50_030.json](../../downstream/demos/manifests/target50_030.json) | 每任务 150 条（RoboCasa 官方 `150_demos`），7,500 条，4,442,662 帧 |
| 第三步 10% | [target50_010.json](../../downstream/demos/manifests/target50_010.json) | 每任务 50 条（官方 `50_demos`），2,500 条，1,494,106 帧；10% ⊂ 30% ⊂ 100% |
| 原始示教 | `/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets/v1.0/target/` | 50/50 齐全，58 GB |

## 5. 实验表

| 编号 | 步骤 | 起点 | 卡数 | 预计时间 | 状态 | 结果路径 |
| --- | --- | --- | --- | --- | --- | --- |
| STEP1-LOAD | 1 | B / S / RS / RD | 1 | 实测约 5 分钟 | 已完成，四个起点都通过 | `downstream_runs/step1_load_test_20261020/out/` |
| STEP2-RS-s1 | 2 | RS | 8 | 训练实测 7 分钟（200 步 2 分 56 秒） | 训练、留出集、种子检查完成；闭环评测重跑排队中（`pt-4qac93lo`） | `downstream_runs/experiments/STEP2-RS-s1/` |
| RLINF-SMOKE-1 | 2（D7） | STEP2-RS-s1 的检查点 | 8 | 估计 30–60 分钟 | 排队中（`pt-a7r4q7s5`） | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| DS-{B,S,RS,RD}-100-s1 | 3（D5 先行） | 四个起点 | 各 8 | 按实测每步 0.75 秒估计各约 42 小时 | 等第二步评测通过和你定检查点间隔 | `downstream_runs/experiments/DS-*-100-s1/` |

## 6. 等你决定的事

- D5 的检查点间隔：每 2.5 万步（约 540 GB，原计划）还是每 5 万步（约 270 GB）——等你看过清理后的空间余量再定。
- 第二步报告会附第三步完整实验表等你确认。
