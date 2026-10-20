# 下游具身任务线状态（唯一的状态文档）

最后更新：2026-10-20 17:10 UTC（第二步进行中）。任务书分四步，每步做完停下来等确认。

## 1. 现在在哪一步

- 第一步（盘点与核实）已完成并经你确认（决定 D1–D8 记在第二步报告开头）。报告：[reports/STEP1_INVENTORY_20261020.md](reports/STEP1_INVENTORY_20261020.md)。
- **第二步（连通试跑）进行中**：8 卡任务 `pt-189qz570` 已提交，在排队。起点 RS（P-S0 第 300 次），种子 1，3 个任务各 100 条示教，200 步，之后 15+ 回合闭环评测和 3 / 12 / 24 个并行仿真进程的效率实测。
- 同时在做：D7 的"StarVLA + RoboCasa365"环境（先列包和总大小）；第三步 100% 数据目录（25,307 条示教的链接）。
- **下一步**：第二步通过（四条成功标准）后，**不等确认**立即提交 D5 的四个 20 万步任务 DS-{B,S,RS,RD}-100-s1，并启动检查点评测的自动提交；然后写第二步报告，含第三步完整实验表，等你确认后才提交第三步。

## 2. 谁在跑

| sco 任务 | 内容 | 状态 | 目录 |
| --- | --- | --- | --- |
| `pt-189qz570` | 第二步连通试跑（8 卡 H800，h800 池） | 排队中（16:30 UTC 提交） | `/mnt/umm/users/yinbaiqiao/downstream_runs/experiments/STEP2-RS-s1/` |
| `pt-405nyxoy` | 第一步加载测试 + 渲染检查（1 卡） | 已完成（10-20 13:30–13:35 UTC，退出码 0），任务记录已删除 | `/mnt/umm/users/yinbaiqiao/downstream_runs/step1_load_test_20261020/` |

登录节点上的后台进程（本线自己的）：100% 数据目录准备（`downstream_runs/logs/prepare_target50_100.log`）、D7 的 pip 试运行（`downstream_runs/logs/d7_pip_dryrun.log`）。

## 3. 代码与位置

- 代码与文档：集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream`，分支 `downstream`（从 main `59053083` 开出，已推送）；只新增 `downstream/`、`docs/downstream/`。worktree 里 `third_party`、`playground` 是指向主检出同名目录的链接（两者都被 .gitignore 忽略），只读使用。
- 大文件（实验产物）：`/mnt/umm/users/yinbaiqiao/downstream_runs/`（`data/` 训练数据目录、`experiments/<编号>/` 实验），不进 git。
- StarVLA：`4507931` + 补丁 [downstream/patches/starvla_4507931_working.patch](../../downstream/patches/starvla_4507931_working.patch)（sha256 `b039782d…`，`git apply --reverse --check` 通过）；每个实验的 `experiment.json` 记提交号和补丁哈希。
- D7 新环境：`/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365`（叠在 starVLA 环境上的 venv，尚未装包）。

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
| STEP2-RS-s1 | 2 | RS | 8 | 估计约 1–1.5 小时 | 排队中（`pt-189qz570`） | `downstream_runs/experiments/STEP2-RS-s1/` |
| DS-{B,S,RS,RD}-100-s1 | 3（D5 先行） | 四个起点 | 各 8 | 估计各约 56 小时（第二步实测后更新） | 第二步通过后提交 | `downstream_runs/experiments/DS-*-100-s1/` |

## 6. 等你决定的事

暂无新的。第二步报告会附第三步完整实验表等你确认。
