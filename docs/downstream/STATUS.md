# 下游具身任务线状态（唯一的状态文档）

最后更新：2026-10-22 09:35 UTC（按你 10-22 的要求，下游任务撤到只剩 RLinf 试跑，其余等密集奖励第一批上机器后再提；评测改为只在已有任务的机器上启动的 1 卡分片）。任务书分四步，每步做完停下来等确认。

## 1. 现在在哪一步

- **第一步（盘点与核实）**：已完成并经你确认。报告 [reports/STEP1_INVENTORY_20261020.md](reports/STEP1_INVENTORY_20261020.md)。
- **第二步（连通试跑）**：已完成，10-21 经你确认通过。报告 [reports/STEP2_SMOKE_20261020.md](reports/STEP2_SMOKE_20261020.md)。RLinf 那部分还没通过（见下）。
- **第三步：暂停**（10-22 09:08 UTC 按你的要求撤下）：
  - 停掉了 9 个在跑的训练：
    - D5 四个，分别到第 113,700–118,800 步，已存第 2.5 万–10 万步检查点；
    - DS-B-100-s2 / s3、DS-S-100-s2 / s3、DS-RS-100-s2，分别到第 55,700–72,400 步，没有检查点。
  - 撤下了 3 个排队的训练（DS-RS-100-s3、DS-RD-100-s2 / s3）和 18 个排队的评测。没有一个评测开始过，所以至今没有任何成功率。
  - 什么都没删：
    - 停下的输出改名为 `run_stopped_20261022/`，训练日志改名为 `train_stopped_20261022.log`；
    - 撤下的任务记录改名为 `job_cancelled_20261022/`；
    - 目录都在 `downstream_runs/experiments/DS-*/` 下。
  - 30% / 10% 的 24 个实验目录已建好（没提交过）。
  - 重新提交的时机：密集奖励第一批（训练链 `interiorgs_v10_train_r12k_20261021`）的三个训练任务都开始以后。三个任务是 S0-keep `pt-ma8luzju`（h800，10-22 09:09 开始）、D-keep-1.77 `pt-pt8wne8e`（zoetrope，排队）、D-keep-1.0 `pt-xg0rku15`（zoetrope，排队）。
  - 重新提交时 D5 怎么接，需要你决定（第 6 节）。
- **评测**：按 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)（950 回合、台面区域顺序固定、边界 3 个任务单列）。
  - 10-22 起改为每个检查点 8 个 1 卡分片，每片约 119 回合、12 个仿真进程。
  - **只在已有别的任务、还有空卡的机器上启动**：
    - 提交前，用 `sco acp jobs get-workers` 查各机器在跑什么，只往"有别的任务、还有空卡"的机器的空卡数里提交；
    - 分片开始后 2 分钟内再查一次：如果它所在的机器上只有我们的分片，说明占了空闲机器，立刻删掉，30 分钟后重提。
    - 方法和训练线的渲染服务搬家（`renderer_repack`）相同。
    - 局限：只看得到本工作区（aigc）的任务。
  - 10-22 09:25 时两个池里有任务的机器都是 8 张卡占满，所以现在不会有分片被提交。
  - P = 48 / 96 的并行度测试不再适用（1 卡分片每片只有 14 个 vCPU）。DS-B 第一个检查点评两遍的噪声测量保留，第二遍也是一套 8 个分片。
- **起点校验已加强**（10-21 11:13）：锁定文件加了每个分片的修改时间和首尾各 4 MiB 的 sha256，RS 和 RD 现在能分开。
- **RLinf 两次更新的试跑还没通过**：
  - 第三次（`pt-00o9ha1p`，10-21 19:00–19:10）：价值头的问题已解决，到第一次训练更新的反向传播时显存不够。8 张卡显存峰值 75.1–75.8 GiB（每卡容量 79.18 GiB）；反向还要再申请 15.69 GiB，正好是整个 7B 模型 bf16 的大小。原因判断：StarVLA 的包装没被 RLinf 自动按层切分，整个模型是一个 FSDP 单元。
  - 第四次：加了按层切分（Qwen2_5_VLDecoderLayer、Qwen2_5_VLVisionBlock）和梯度检查点，只用命令行覆盖。10-21 19:19 提交到 zoetrope（`pt-jnetqxe5`）；10-22 09:15 为了不排在密集奖励 D-keep 两个训练任务前面，改提交到 h800（`pt-wytuck3x`，排队中）。
  - 通过后单独报告，并开始写第四步方案。

## 2. 谁在跑

| sco 任务 | 内容 | 状态 | 目录 |
| --- | --- | --- | --- |
| `pt-wytuck3x` | RLinf 2 次更新试跑，第四次（8 卡，h800；use_orig_params、按层切分、梯度检查点） | 排队中（10-22 09:15 提交） | `downstream_runs/experiments/RLINF-SMOKE-1/retry3/` |
| 29 个（见上） | 第三步训练和 D5 评测 | 10-22 09:08 按你的要求撤下 | `downstream_runs/experiments/DS-*/` |
| `pt-00o9ha1p`、`pt-a7r4q7s5` | RLinf 试跑第三、二次 | 退出码 255（显存不够；价值头取不到参数）；记录已删 | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| `pt-cb0kvvf0` | 34 个评测任务的重置扫描（1 卡） | 已完成：原版 89 / 102 对可复现，固定顺序后 102 / 102；记录已删 | `downstream_runs/checks/reset_sweep_20261021/` |
| 更早的任务 | 第一、二步 | 已结束，记录已删（见第二步报告） | — |

**登录节点上本线的后台进程**：
- **看板** `downstream/eval/watch.py`：10-22 09:25 起，PID 1201348，代码 `5cf505a0`，启动参数 `--hold-until-running pt-ma8luzju pt-pt8wne8e pt-xg0rku15`。
  - 每 15 分钟一轮，每 2 分钟做一次分片落点检查。
  - 在密集奖励第一批三个训练任务都开始之前，不提交任何任务。
  - 每轮读一次配额剩余空间：剩余少于 1 TB 或读不出时，只报警、不提交。
  - 新检查点按上面的规则提交 1 卡分片评测。
  - 按门槛提交第三步其余 24 个。
  - 有新评测完成时重建结果报告。
  - 看板 `downstream_runs/BOARD.md`，日志 `downstream_runs/logs/watch.log`，状态 `downstream_runs/WATCH_STATE.json`。
  - 用 setsid + nohup 运行，ssh 断开后仍会继续。它除了删掉落在空闲机器上的分片任务，不删任何东西。
  - 启动、停止和查看（在集群 worktree 里；代码更新后要先 stop 再 start）：

```bash
bash downstream/eval/watch.sh start --hold-until-running pt-ma8luzju pt-pt8wne8e pt-xg0rku15
```

```bash
bash downstream/eval/watch.sh stop
```

```bash
bash downstream/eval/watch.sh status
```

## 3. 代码与位置

- **代码与文档**：分支 `downstream`（从 main `59053083` 开出）。只新增 `downstream/` 和 `docs/downstream/`。
- **同步方式（10-21 起）**：本地 worktree `C:\Users\user\Desktop\code\VAGEN-Lite-downstream` 提交 → push → 集群 `git pull --ff-only`。不再用 tar 覆盖。只在集群上才能生成的文件（如 `starts_lock.json`），在集群上提交、push，再在本地 pull。
- **集群 worktree**：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream`。里面的 `third_party`、`playground` 是指向主检出同名目录的链接（都被 .gitignore 忽略），只读使用。
- **大文件**（实验产物）在 `/mnt/umm/users/yinbaiqiao/downstream_runs/`，不进 git：
  - `data/target50_{100,030,010}`：训练数据目录，只建链接；
  - `experiments/<编号>/`：每个实验；
  - `checks/`：一次性检查；
  - `logs/`：登录节点上的日志；
  - 看板和看板状态。
- **训练任务的建立和提交**：`downstream/train/runs.py make|submit|list`（第三步统一用它：20 万步，只存最后一个检查点）。
- **StarVLA**：`4507931` + 补丁 [downstream/patches/starvla_4507931_working.patch](../../downstream/patches/starvla_4507931_working.patch)（sha256 `b039782d…`，`git apply --reverse --check` 通过）。每个实验的 `experiment.json` 记提交号和补丁哈希。
- **评测**：[EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)，计划文件 [downstream/eval/eval_d6.json](../../downstream/eval/eval_d6.json)，运行脚本 `downstream/eval/run_eval_ckpt.sh CKPT OUT 服务数 进程数[,进程数…]`。
- **剩余空间**：`python3 downstream/quota.py` 读 `sco afs dir-quota list` 整张表里我们这一行（`-d` 参数返回的记录不对，不用它）。
- **结果报告**（10-21 起）：`downstream/report/`。
  - 内容：
    - 收集训练损失和全部评测；
    - 配对统计：Wilson 区间、逐任务 McNemar 精确检验加 Holm 校正、跨任务 Wilcoxon 检验、种子均值 ± 标准差；
    - 8 张图：训练损失、检查点曲线、样本效率曲线、逐任务热力图、逐任务配对差、结果一致性、成功所用步数、评测噪声；
    - 指标说明；
    - 按固定规则挑的范例。范例画面来自每个评测回合记下的 8 帧画面条（`frames/<任务>_s<种子>.jpg`，10-21 起记录）。
  - 生成：看板在有新评测完成时自动重建到 `downstream_runs/report/`（`index.html`、`results.json`、`figures/`）。手动生成用下面的命令（vagen-lite 环境，登录节点 CPU，几秒钟）：

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m downstream.report.build --out /mnt/umm/users/yinbaiqiao/downstream_runs/report
```

  - 网页版（私有 Artifact，我在有新结果时重新发布）：https://claude.ai/artifact/Qa1oNasyCN5pSkjg2LXhBq
  - 版式预览用的模拟数据在 `downstream_runs/checks/report_sim_20261021/`（目录里有 `SIMULATED` 标记文件，页面带"模拟数据"横幅），与真实实验分开。
- **D7 新环境**：`/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365`。它是叠在 starVLA 环境上的 venv，1.9 GB；包清单在 `downstream/envs/`，激活用 `source downstream/envs/activate_rlinf.sh <ENV>`。

## 4. 权重与数据路径

起点登记在 [downstream/starts.json](../../downstream/starts.json)，锁定值在 [downstream/starts_lock.json](../../downstream/starts_lock.json)。每次建实验和每个训练任务开头都校验一次，变了就停，不换路径。集群路径前缀 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/`。

| 名字 | 用途 | 路径 | 格式 / 精度 |
| --- | --- | --- | --- |
| B | 原始 Qwen2.5-VL-7B-Instruct（`cc594898`） | `interiorgs_v10_r11_20261016/inputs/base_7b` | 完整 HF，bf16 |
| S | 第十一轮动作 SFT 第 9 个 epoch | `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386` | 完整 HF，bf16 + fp32 混合 |
| RS | P-S0 第 300 次 | `interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300/actor/huggingface` | 完整 HF，fp32 |
| RD | P-D 第 150 次 | `interiorgs_v10_train_r11p_20261017/rl_run_pd/checkpoints/global_step_150/actor/huggingface` | 完整 HF，fp32 |
| R10S5（备用） | 第十轮 SFT 第 5 个 epoch | `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770` | 完整 HF |
| R1（备用） | R1 最终模型 | `R1-clean-Projective-v0/formal/checkpoints/global_step_250/actor/huggingface` | 完整 HF |

- 训练时所有起点统一按 bf16 加载（D1）。
- 锁定内容：
  - 分片大小；
  - 索引和 config.json 的 sha256；
  - 10-21 起，还有每个分片的修改时间和首尾各 4 MiB 的 sha256。

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
| STEP2-RS-s1 | 2 | RS | 8 | 训练实测 434 秒（200 步） | **通过**（10-21 确认） | `downstream_runs/experiments/STEP2-RS-s1/` |
| RLINF-SMOKE-1 | 2（D7） | STEP2-RS-s1 的检查点 | 8 | 第二次 692 秒、第三次 646 秒后失败（环境交互 41 秒跑通，第一次训练更新没完成） | 第四次排队中（`pt-wytuck3x`，h800） | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| DS-{B,S,RS,RD}-100-s1 | 3（D5） | 四个起点 | 各 8 | 训练约 42.0 小时；评测改为 8 个 1 卡分片 | 10-22 09:08 撤下（到第 113,700–118,800 步，有第 2.5 万–10 万步检查点）；等密集奖励第一批上机器后重提，怎么接待你决定 | `downstream_runs/experiments/DS-*-100-s1/` |
| DS-{B,S,RS,RD}-100-s{2,3} | 3 | 四个起点 | 各 8 | 同上，只评最后一个检查点 | 10-22 09:08 撤下（5 个在跑的到第 55,700–72,400 步，没有检查点；3 个没开始）；等密集奖励第一批上机器后从头重提 | `downstream_runs/experiments/DS-*-100-s{2,3}/` |
| DS-{B,S,RS,RD}-{030,010}-s{1,2,3} | 3 | 四个起点 | 各 8 | 同上 | 实验目录已建好；等门槛（DS-B-100-s1 第 2.5 万步评测成功回合 > 0）且密集奖励第一批上机器后由看板提交 | `downstream_runs/experiments/DS-*-{030,010}-s*/` |

## 6. 决定记录与待定

**10-21 你对第二步报告的决定**：
1. 固定台面区域顺序，作为本线所有评测的固定设置。已执行，见 EVAL_PROTOCOL.md。
2. 先提交 100% 数据、种子 2 和 3 的 8 个任务；其余 24 个按门槛由看板提交；D5 的 4 个复用。已执行。
3. 评测规模改为 950 回合。已执行；还没有需要补评的检查点。
4. DS-B 第一个检查点评两遍。看板会自动提交。
5. 加强起点校验。已执行。
6. 3 个边界任务单独列出。看板和协议里已单列。
7. 第三步新任务只存最后一个检查点。已执行。
8. D5 的中间检查点先留着；看板每次提交前检查剩余空间，少于 1 TB 只报警不提交。已执行。

**补充要求**：
- P = 48、96 的并行度测试：在 DS-B 第一个检查点的第二遍评测里做。
- 代码同步改用 git。已改。
- 看板改成断开后仍运行，启动命令见第 2 节。已改。
- RLinf：第三次已报告（显存不够），第四次排队中，结果出来后单独报告。

**待定**：暂无。门槛不过（成功回合 = 0）时会停下来等你决定。
