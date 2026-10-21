# 下游具身任务线状态（唯一的状态文档）

最后更新：2026-10-21 08:10 UTC（第二步已通过，报告待交；D5 四个任务排队中）。任务书分四步，每步做完停下来等确认。

## 1. 现在在哪一步

- 第一步（盘点与核实）：已完成，并经你确认（决定 D1–D8 记在第二步报告开头）。报告：[reports/STEP1_INVENTORY_20261020.md](reports/STEP1_INVENTORY_20261020.md)。
- **第二步（连通试跑）：已通过，四个成功标准都达到。**
  - 训练：损失 2.285 → 0.125，全程有限。
  - 评测：检查点由策略服务加载；15 个回合成功 5 个。
  - 记录齐全。
  - 报告：[reports/STEP2_SMOKE_20261020.md](reports/STEP2_SMOKE_20261020.md)（草稿，等两项结果后定稿：34 个任务的重置扫描、RLinf 重试）。
- **D5 四个任务已提交（10-21 07:05 UTC），在排队。** 四份配置只有起点路径不同，每 2.5 万步存一次检查点。检查点的自动评测进程已在登录节点启动。
- **新发现，已修正，待你认可**：原版 RoboCasa 的同一评测种子在不同进程里可能重置出不同场景。第二步 24 对里有 2 对出现两个版本，其中一对机器人起点差 1.65 米。原因是 `Counter.get_reset_regions` 用 `list(set(...))` 去重，顺序随内存地址变化。评测进程里已换成按出现顺序去重的副本（`1bd9ee5f`，RoboCasa 文件未改）；修正后 4 对 × 4 个新进程全部逐位相同。全部 34 个任务的扫描在跑（`pt-cb0kvvf0`）。
- **下一步**：
  - 扫描和 RLinf 重试有结果后，给第二步报告定稿，交给你；
  - 第三步的 32 个新任务等你确认实验表后再提交。

## 2. 谁在跑

| sco 任务 | 内容 | 状态 | 目录 |
| --- | --- | --- | --- |
| `pt-au0xzxrw` | DS-B-100-s1（8 卡，h800） | 排队中（10-21 07:05 提交） | `downstream_runs/experiments/DS-B-100-s1/` |
| `pt-uatwo4vx` | DS-S-100-s1（8 卡，h800） | 排队中（07:05 提交） | `downstream_runs/experiments/DS-S-100-s1/` |
| `pt-10jkijs8` | DS-RS-100-s1（8 卡，zoetrope） | 排队中（07:05 提交） | `downstream_runs/experiments/DS-RS-100-s1/` |
| `pt-2z4mnq4y` | DS-RD-100-s1（8 卡，h800） | 排队中（07:05 提交） | `downstream_runs/experiments/DS-RD-100-s1/` |
| `pt-00o9ha1p` | RLinf 2 次更新试跑，第三次（8 卡，zoetrope；加 `actor.fsdp_config.use_orig_params=true`） | 排队中（07:08 提交） | `downstream_runs/experiments/RLINF-SMOKE-1/retry2/` |
| `pt-cb0kvvf0` | 34 个评测任务的重置扫描（1 卡，h800；原版和固定区域顺序各一遍） | 运行中（07:46 开始） | `downstream_runs/checks/reset_sweep_20261021/` |
| `pt-4qac93lo` | 第二步闭环评测重跑（8 卡，zoetrope） | 已完成，退出码 0（06:28–06:50） | `downstream_runs/experiments/STEP2-RS-s1/step2/eval2/` |
| `pt-a7r4q7s5` | RLinf 试跑，第二次 | 失败，退出码 255：第一次训练更新时价值头在 FSDP 下取不到参数 | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| `pt-189qz570` | 第二步连通试跑 | 10-20 20:56 手动停止（闭环评测卡死，已修），记录已删 | `downstream_runs/experiments/STEP2-RS-s1/` |
| `pt-9ruo5yrs`、`pt-g0ydp76l` | 第二步评测、RLinf 试跑的第一次提交 | 10-20 21:04 遇上磁盘写满，启动约 3 秒即失败，记录已删 | 同上 |
| `pt-405nyxoy` | 第一步加载测试 + 渲染检查（1 卡） | 已完成，记录已删 | `downstream_runs/step1_load_test_20261020/` |

登录节点上本线的后台进程：
- `downstream/eval/watch.py`（PID 3692130，10-21 07:46 启动）：每 15 分钟检查 D5 四个实验。新检查点存完 10 分钟不变后，就提交一个 8 卡评测任务（1 个策略服务、24 个仿真进程、D6 固定评测）。看板写在 `downstream_runs/BOARD_D5.md`。它从不删除文件。停止：`kill 3692130`。

## 3. 代码与位置

- 代码与文档：集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream`，分支 `downstream`（从 main `59053083` 开出，已推送）。只新增 `downstream/` 和 `docs/downstream/`。worktree 里的 `third_party`、`playground` 是指向主检出同名目录的链接（都被 .gitignore 忽略），只读使用。
- 大文件（实验产物）：`/mnt/umm/users/yinbaiqiao/downstream_runs/`，不进 git。
  - `data/`：训练数据目录（只建链接）；
  - `experiments/<编号>/`：实验；
  - `checks/`：一次性检查；
  - `logs/`：登录节点上的日志。
- StarVLA：`4507931` + 补丁 [downstream/patches/starvla_4507931_working.patch](../../downstream/patches/starvla_4507931_working.patch)（sha256 `b039782d…`，`git apply --reverse --check` 通过）。每个实验的 `experiment.json` 记提交号和补丁哈希。
- 评测：[downstream/eval/eval_d6.json](../../downstream/eval/eval_d6.json)（34 个已见任务 × 种子 1000–1019）。
  - 运行：`downstream/eval/run_eval_ckpt.sh CKPT OUT 服务数 仿真进程数`；
  - 仿真环境：`downstream/eval/sim_env.sh`（OSMesa）；
  - 台面区域顺序默认固定（`closed_loop.py --no-stable-regions` 可关）。
- D7 新环境：`/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365`。它是叠在 starVLA 环境上的 venv，1.9 GB；包清单在 `downstream/envs/`，激活用 `source downstream/envs/activate_rlinf.sh <ENV>`。

## 4. 权重与数据路径

起点登记在 [downstream/starts.json](../../downstream/starts.json)，锁定值在 [downstream/starts_lock.json](../../downstream/starts_lock.json)（每次建实验和每个训练任务开头都校验）。集群路径前缀 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/`。

| 名字 | 用途 | 路径 | 格式 / 精度 |
| --- | --- | --- | --- |
| B | 原始 Qwen2.5-VL-7B-Instruct（`cc594898`） | `interiorgs_v10_r11_20261016/inputs/base_7b` | 完整 HF，bf16 |
| S | 第十一轮动作 SFT 第 9 个 epoch | `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386` | 完整 HF，bf16 + fp32 混合 |
| RS | P-S0 第 300 次 | `interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300/actor/huggingface` | 完整 HF，fp32 |
| RD | P-D 第 150 次 | `interiorgs_v10_train_r11p_20261017/rl_run_pd/checkpoints/global_step_150/actor/huggingface` | 完整 HF，fp32 |
| R10S5（备用） | 第十轮 SFT 第 5 个 epoch | `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770` | 完整 HF |
| R1（备用） | R1 最终模型 | `R1-clean-Projective-v0/formal/checkpoints/global_step_250/actor/huggingface` | 完整 HF |

训练时所有起点统一按 bf16 加载（D1）。RS 和 RD 的索引文件 sha256 相同、分片大小也相同（结构一样），现有锁定分不出两者。加强方案见第二步报告"需要你决定的事"第 5 条。

| 数据 | 清单 | 规模 |
| --- | --- | --- |
| 第二步 | [downstream/demos/manifests/step2_smoke.json](../../downstream/demos/manifests/step2_smoke.json) | 3 个任务各前 100 条示教（训练）+ 后 20 条（留出），训练 44,369 帧 |
| 第三步 100% | [target50_100.json](../../downstream/demos/manifests/target50_100.json) | 50 个任务，25,307 条示教，14,957,899 帧 |
| 第三步 30% | [target50_030.json](../../downstream/demos/manifests/target50_030.json) | 每任务 150 条（RoboCasa 官方 `150_demos`），7,500 条，4,442,662 帧 |
| 第三步 10% | [target50_010.json](../../downstream/demos/manifests/target50_010.json) | 每任务 50 条（官方 `50_demos`），2,500 条，1,494,106 帧；10% ⊂ 30% ⊂ 100% |
| 原始示教 | `/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets/v1.0/target/` | 50 个任务齐全，58 GB |

## 5. 实验表

| 编号 | 步骤 | 起点 | 卡数 | 时间 | 状态 | 结果路径 |
| --- | --- | --- | --- | --- | --- | --- |
| STEP1-LOAD | 1 | B / S / RS / RD | 1 | 实测约 5 分钟 | 已完成，四个起点都通过 | `downstream_runs/step1_load_test_20261020/out/` |
| STEP2-RS-s1 | 2 | RS | 8 | 训练实测 434 秒（200 步，每步 0.722 秒）；评测 24 回合在 P = 24 时用 171 秒 | **通过** | `downstream_runs/experiments/STEP2-RS-s1/` |
| RLINF-SMOKE-1 | 2（D7） | STEP2-RS-s1 的检查点 | 8 | 第二次实测 692 秒后失败 | 第三次排队中（`pt-00o9ha1p`） | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| DS-{B,S,RS,RD}-100-s1 | 3（D5 先行） | 四个起点 | 各 8 | 按实测估计各约 40.2 小时；每个检查点评测约 2.4 小时（8 卡，上限） | 排队中（07:05 提交） | `downstream_runs/experiments/DS-*-100-s1/` |
| 第三步其余 32 个 | 3 | 四个起点 × 10% / 30% / 100% × 种子 1–3 | 各 8 | 同上 | 等你确认（实验表在第二步报告 6.4 节） | — |

## 6. 等你决定的事

- （已定）D5 每 2.5 万步存一个检查点（10-21）。
- 第二步报告"需要你决定的事"里的 6 条：
  1. 评测时固定台面区域顺序；
  2. 第三步实验表（复用 D5 的 4 个，新跑 32 个）；
  3. 第三步只存最后一个检查点；
  4. 量一次评测本身的噪声；
  5. 加强起点校验；
  6. 3 个边界空间任务。
