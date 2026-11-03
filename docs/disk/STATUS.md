# 磁盘配额线状态（唯一的状态文档）

最后更新：2026-11-03 07:45 UTC（配额表 21 TB、剩 1 TB，停止提交标志 02:24 起有效；一天涨了约 5 TB，主要是 P1 矩阵 v3 的 12 个 RL 和 2 个 SFT（`VAGEN-Lite-matrix` +3.60 TB）和奖励线（+0.77 TB）；第四轮及以前的记录见第 4 节）。

这条线管 `/mnt/umm/users/yinbaiqiao`（个人目录，配额 22 TB）：量、预测、提方案、按批准的清单执行，默认只读。

- 报告：[reports/](reports/)。第一步报告是 [reports/DISK_INVENTORY_20261026.md](reports/DISK_INVENTORY_20261026.md)；第一到三批的执行记录是 [reports/CLEANUP_20261026.md](reports/CLEANUP_20261026.md)。
- 保留规则：[RETENTION_POLICY.md](RETENTION_POLICY.md)；"磁盘紧张就不提交"约定：[STOP_FLAG_PROPOSAL.md](STOP_FLAG_PROPOSAL.md)（都已批）；给各线的转告：[RELAY_20261026.md](RELAY_20261026.md)。
- 脚本：仓库根目录 `disk/`，说明见 [README.md](README.md)。批次说明（批准了删什么）：[batches/](batches/)。
- worktree：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-disk`，本地分支 `disk`，从 `origin/main` `00363333` 开出，不跟踪远端，不推送。
- 原始结果和记录：`/mnt/umm/users/yinbaiqiao/disk_runs/`（批次记录在 `disk_runs/batches/<批次>/`，核查记录在 `disk_runs/audit/<批次>/`）。

## 1. 现在在哪一步

- 第一步（盘点与预测）：完成，你 10-26 已答复。
- 第二步（按批准的清单清理、保留规则、标志文件方案）：
  - 第一、二、三批已执行，共腾出 3,864.0 GB（第三批是 SenseNova-U1 冒烟检查点，root 任务 `pt-1r0uuww1` 删除）。
  - 保留规则和标志约定你已批，转告的话在 [RELAY_20261026.md](RELAY_20261026.md)。
  - 你批了但核查没过的 0.54 TB 交回下游线，0.06 TB 等主线确认。
- 第 6 项（probe_spatial、SenseNova-U1、Consistency 纳入这条线）：盘点完成，报告 [reports/SCOPE6_INVENTORY_20261026.md](reports/SCOPE6_INVENTORY_20261026.md)。
  - 3.6 TB 里约 1.81 TB 是公开下载（决定 3 不删），约 1.68 TB 是 probe_spatial 发布要用的，约 0.26 TB 是 root 属主的派生特征。
  - 唯一候选是 SenseNova-U1 的冒烟检查点 292.1 GB（root 属主），你批了，已作为第三批删除。
- 第三步（常驻监控）：10-26 07:37 起在跑；09:19 起带停止提交标志（`--stop-flag`）。
- 第二轮清理（10-27，你说"继续管理配额，Qwen2.5 已舍弃，只留必要的"）：报告 [reports/RECLAIM_20261027.md](reports/RECLAIM_20261027.md)，你四项都选甲，19:27–20:03 做完（执行记录在报告第 5 节）。
  - 模型线交来的清单 1,236.4 GB 已核对、核查通过，分 b4（模型线自己的 634.0 GB）、b5（Qwen3-VL 的 602.4 GB，归主线）。
  - Qwen2.5-VL 推荐删 2,154 GB（续训状态、最后但不是最好的、第十轮以前的旧权重、3B 试跑），每个实验的最好检查点和下游起点都留。
  - b4–b7 共 1,150 个文件、3,517.9 GB，都先上传 AOSS（0 个不符）、核查通过后删除；转告的话在 [RELAY_20261027.md](RELAY_20261027.md)。

## 2. 汇报表

以后每次汇报都用这张表，行列不变，只改数字和时间；没有变化不汇报。剩余 = 22 TB − 配额表读数（你的决定 4）。

| 项 | 数值 | 时间（UTC） | 说明 |
| --- | --- | --- | --- |
| 配额 | 22 TB | — | 按你的决定 4 |
| 已用（配额表） | 21 TB | 11-03 07:30 | 两位有效数字，±0.5 TB；11-02 05:58 删完第四轮是 16 TB |
| 剩余 | 1 TB | 11-03 07:30 | 22 − 配额表读数；低于 1.5 TB，停止提交标志 11-03 02:24 起有效 |
| 已用（文件系统统计） | 15.29 TB | 11-03 07:09 | 只看变化趋势；比配额表少 5.7 TB（10-31 以前少 2–3 TB，原因不明） |
| 本步已腾出 | 16.79 TB | 11-02 05:57 | 按各批 `deleted.tsv` 逐文件加总，b1–b24 共 16,788.8 GB：第一轮 b1–b3 3,864.0 GB；第二轮 b4–b7 3,517.9 GB；第三轮 b8、b9 2,109.5 GB；b10 754.5 GB；b11–b14 2,063.8 GB；b15 839.9 GB；b16 327.9 GB；b17 114.2 GB；b18 16.3 GB；第四轮 b19–b22 1,637.1 GB、b23–b24 1,543.7 GB；10-28 拉回 136.0 GB 不扣。以前报的累计数多了 0.01 TB（b1–b17 实为 13.59 TB，不是 13.60） |
| 7 天内最低剩余（情形 A / B） | 5.07 / 4.01 TB | 预测于 10-27 00:00 | 第二轮删除之前的预测；删了 3.52 TB 后最低剩余相应上移，不会碰 3 TB 线 |
| 监控 | 在跑，PID 3214666，带停止提交标志 | 10-26 09:19 起 | 见第 6 节 |

## 3. 你的决定（10-26）

| 编号 | 决定 | 执行 |
| --- | --- | --- |
| 1 | 甲：批第一批清理清单 | 5.2.1、5.2.4、5.2.5 在第一批；5.2.6 在第二批；5.2.3（E0 检查点）因下游看板会读，交给下游线处理 |
| 2 | 甲："满 7 天即可删"提前删 | 主线组合链续训状态在第一批；RL-SPEED-1/2 在第二批；`evals_src` 因下游看板每轮都会读、RLINF-SMOKE-1 是下游导出测试脚本的默认输入，交给下游线处理 |
| 3 | 公开下载的模型和数据集不删 | "立刻可删"清单（0.15 TB）作废；RoboCasa365 的 `pretrain` 数据同样不删 |
| 4 | 不问管理员，按 22 TB 用 | 剩余按 22 − 配额表读数算；文件系统统计只看趋势 |
| 5 | 甲：现在开监控 | 07:37 已开（第 6 节）；停止提交标志在第二轮批了以后 09:19 加上 |
| 6 | probe_spatial、SenseNova-U1、Consistency 纳入这条线 | 盘点完成（`reports/SCOPE6_INVENTORY_20261026.md`）；其中两项在第二轮定了 |
| 7 | （10-22 报错）没有答复 | 待你告诉在哪看到的 |

10-26 第二轮（清理报告和第 6 项盘点之后），六项都选甲：

| 编号 | 决定 | 执行 |
| --- | --- | --- |
| 保留规则 | 批，转给主线、模型线、下游 | 转告的话写在 `RELAY_20261026.md`，由你转达 |
| 标志约定 | 批：1.5 TB 写、3 TB 删，评测不拦 | 监控 09:19 起带 `--stop-flag`；各线接入由你转达 |
| 交回下游的 0.54 TB | 转告下游，改好代码后它自己删或叫磁盘线删 | 写在 `RELAY_20261026.md` |
| 主线 3B 试跑 | 问主线，不要就由磁盘线归档后删 | 写在 `RELAY_20261026.md` |
| SenseNova-U1 冒烟检查点 | 先上传 AOSS 核对，再用 1 卡 root sco 任务删 | 已完成：第三批，root 任务 `pt-1r0uuww1`，09:24 删完 292.1 GB |
| InsScene-15K 的 root 本地特征镜像 | 不动 | — |

另外：下游线 07:14–07:16 按你在那边给的决定，自己删了 11 份已训完实验的全状态（1,296.8 GB，记录 `downstream_runs/logs/delete_full_states_20261026.txt`，没有上传 AOSS），也就是我清单里的 5.2.2，从清单划掉。

## 4. 清理批次

| 批次 | 内容 | 文件数 | 大小 | 上传 AOSS | 核查 | 删除 | 腾出 |
| --- | --- | ---: | ---: | --- | --- | --- | ---: |
| b1_20261026 | pilot r3 分片 1,692.0 GB；主线 9 个组合运行的续训状态 895.5 GB；R1 taskfix 第 50–200 步 165.9 GB；整理归档区 6 个子目录 223.5 GB | 10,583 | 2,976.9 GB | 08:20 完成，0 个不符 | 通过（`disk_runs/audit/b1/CONCLUSION.md`） | 08:24 完成 | 2,976.9 GB |
| b2_20261026 | RL-SPEED-1、RL-SPEED-2 的检查点 471.8 GB；StarVLA NavigateKitchen 检查点 123.3 GB | 105 | 595.0 GB | 08:27 完成，0 个不符 | 通过（`disk_runs/audit/b2/CONCLUSION.md`） | 08:28 完成 | 595.0 GB |
| 交给下游 | E1 权重副本 134.8 GB、E0 检查点 269.6 GB、RLINF-SMOKE-1 检查点 134.8 GB | — | 539.2 GB | — | 下游代码会读，未通过 | — | — |
| 两次 3B 试跑 | 62.6 GB，已并入 b6_20261027（第二轮决定 3） | — | — | — | — | — | — |
| b3_20261026 | SenseNova-U1 冒烟检查点 `SenseNova-U1/training/RUN/sensenovau1_8b_it2i_smoke3/`（root 属主） | 87 | 292.1 GB | 09:23 完成，0 个不符 | 通过（`disk_runs/audit/b3/CONCLUSION.md`） | 09:24 由 1 卡 root 任务 `pt-1r0uuww1` 删除（`disk/submit_root_delete.sh`） | 292.1 GB |
| AOSS 测速对象 | `disk_line_speedtest_20261026/base7b_shard1.safetensors` | 1 | 3.9 GB | — | — | 08:30 已删（记录 `disk_runs/logs/aoss_speedtest_object_deleted.txt`） | 不占配额 |
| b4_20261027 | 模型线清单里模型线自己的：Cambrian SFT 没选中的 epoch、冒烟 SFT、Cambrian RL 续训状态、下游冒烟 | 265 | 634.0 GB | 10-27 19:27 完成，0 个不符 | 通过（`disk_runs/audit/b45/CONCLUSION.md`） | 19:29 | 634.0 GB |
| b5_20261027 | 模型线清单里 Qwen3-VL 的（归主线，你直接批）：SFT 没选中的 epoch、冒烟、RL 续训状态、下游冒烟 | 261 | 602.4 GB | 19:35 完成，0 个不符 | 通过（同上） | 19:36 | 602.4 GB |
| b5 拉回（10-28） | 模型线转达你的第 29 条（甲）：Qwen3 SFT `checkpoint-154/308/462/616`（第 1–4 个 epoch）从 AOSS 拉回原路径，`disk/restore_from_aoss.py`，2 个文件 × 2 线程 | 84 | 136.0 GB | 拉回 19:48–19:52，逐文件核对大小 0 个不符（记录 `disk_runs/batches/b5_20261027/restored_20261028T195226Z.tsv`） | — | 说明文件 `lf_q3formal/model/DELETED_TO_AOSS_20261027.txt` 已补一段 | −136.0 GB |
| b8_20261028 | 下游已训完、已评完实验的续训状态：X2 1% 四个、3% 的 B、S、RS 的全状态，X1 四个的全状态，E3-B、E3-RS 的 RLinf 续训检查点（第三轮决定 1） | 249 | 1,066.2 GB | 10-28 22:22–23:45，0 个不符（2 × 2 线程） | 通过（`reports/CLEANUP3_20261028.md` 第 1 节） | 23:45 | 1,066.2 GB |
| b9_20261028 | 跑完的 Qwen3 / Cambrian RL 最后检查点的续训状态：主线 7 个、模型线 3 个（第三轮决定 2） | 250 | 1,043.3 GB | 23:45–01:14，0 个不符 | 通过（同上） | 10-29 01:14 | 1,043.3 GB |
| b10_20261029 | 模型线第二份清单（`VAGEN-Lite-models/docs/models/reports/DISK_RECLAIM_20261029.md` 第 1 表）：DS-Q3B、DS-Q3S、DS-Q3RD、DS-Q3RS、DS-CSB 的 17.5 万步全状态，8 份已评完的 `evals_src` 副本（R3、R4） | 137 | 754.5 GB | 10-29 21:47–22:43，0 个不符（2 × 2 线程） | 通过：五个训练任务都 SUCCEEDED，评测点在 `watch.json` 都记着已评完，只删 `evals_src/steps_*`、留 `watch.json` | 22:43 | 754.5 GB |
| b11–b14（10-30 盘点，10-31 执行） | b14 下游已训完已评完的 T5 四个、X2-RD 3%、XE、T2 三个的全状态和 E3-S / RD 的 RLinf 续训检查点 915.1；b13 主线 8 个跑完的 Qwen3 RL 续训状态 841.7；b12 三个停下的 Cambrian RL 臂续训状态 289.1；b11 DS-Q3S 17.5 万步副本 17.9 | 486 | 2,063.8 GB | 一个 1 卡 sco 任务 `pt-rpmx7y80`（`disk/submit_archive_delete_job.sh`，8 × 8 线程）04:39–04:59，四批都 0 个不符 | 通过（运行都已结束、评完；没有计划从这些检查点续训） | 04:48 / 04:55 / 04:58 / 04:59 | 2,063.8 GB |
| b15_20261031 | 主线第 96 条（用户 10-31"甲，交给磁盘线先归档再删"）：没设 retain 的 Qwen3 链（q3dck、q3s2dck、q3s3、q3weak）22 个中间检查点，格式 SFT 的 checkpoint-14 和最终权重（`lf_q3fmt/model` 顶层）；清单 `interiorgs_v10_q3_20261027/monitor/cleanup_q3_dryrun_20261031.txt` | 491 | 839.9 GB | 1 卡 sco 任务 `pt-2cj8kq8n` 07:20–07:28，0 个不符 | 通过（`disk_runs/audit/b15/CONCLUSION.md`：22 个检查点没有任何地方引用；最终权重只有自己的 train.yaml 当输出目录提到；q3weak 的起点是 checkpoint-7） | 07:28 | 839.9 GB |
| b16_20261031 | 模型线第 1d 表（它的决定 42 = 甲，你在这边也批了）：DS-CSS、DS-CSRS 的 17.5 万步全状态，DS-CSS 17.5 万、DS-CSRS 和 DS-CSRD 15 万 / 17.5 万、DS-Q3RD 17.5 万步的已评完副本；留着 DS-CSS 15 万步副本（决定 39）、DS-CSRD 的全状态（还在训）、各最终权重 | 66 | 327.9 GB | 1 卡 sco 任务 `pt-okfifo6x` 12:53–12:56，0 个不符 | 通过（训练任务 SUCCEEDED，这些点的评测任务都 SUCCEEDED，watch.json 记着已评完） | 12:56 | 327.9 GB |
| b17_20261031 | 模型线第 1e 表（它的决定 44 = 甲，你 11-01 在这边确认）：DS-CSRD 的 17.5 万步全状态；最终权重留着 | 21 | 114.2 GB | 1 卡 sco 任务 `pt-o2hssdb6` 07:13–07:15，0 个不符 | 通过（训练任务、JPEG 评测都 SUCCEEDED，没有进程或任务脚本读它） | 07:15 | 114.2 GB |
| b18_20261101 | 模型线第 1f 表（它的决定 45 = 甲，你 11-01 在这边确认）：DS-CSS 15 万步的已评完副本 `evals_src/steps_150000`（R4）；`watch.json` 留着 | 4 | 16.3 GB | 1 卡 sco 任务 `pt-xai4243f` 21:55–21:56，0 个不符 | 通过（原评测任务 pt-327497s5 和 JPEG 重评 pt-2oaccrlf、pt-vbdyiybe 都已不存在，`watch.json` 记着 10-30 08:18 评完，没有进程读它；不碰别删清单） | 21:56 | 16.3 GB |
| b19–b22（11-02，第四轮第一类，你"1甲"；报告 [reports/CLEANUP4_20261102.md](reports/CLEANUP4_20261102.md)） | b19 主线 q3abl 6 个、q3pd5 3 个运行最后检查点的续训状态 946.9；b20 Cambrian cs2d 第 100 次的续训状态 96.4；b21 下游 V4 复核两条（DS-Q3V4S3、DS-Q3DCKS2）的全状态和已评完副本 321.7；b22 q3c3sft 8 个没选中的 SFT epoch 272.1 | 476 | 1,637.1 GB | 先交 h800（pt-kgbse1k1），按你 11-02 改投 zoetrope 的指示在没开跑时删掉；zoetrope 上 1 卡 sco 任务 `pt-emznzyg9` 04:35–04:51，四批都 0 个不符 | 通过（`disk_runs/audit/b1922/CONCLUSION.md`：引用这些检查点的只读 `actor/huggingface`；没有在跑的任务或进程读它们；不碰别删清单） | 04:38 / 04:47 / 04:50 / 04:51 | 1,637.1 GB |
| b23–b24（11-02，第四轮第二类，你"按乙来执行"后"1甲 2甲"；报告第 3.1 节） | b23 Qwen2.5 留下的权重，依赖清单以外的 33 个检查点（含 r13c3sft 的 SFT checkpoint-420）1,093.8；b24 Cambrian 14 个 RL 检查点 449.9。留着：依赖清单里的全部 Qwen2.5 权重、Cambrian 登记的两个下游起点（cs0 第 100 次、crd 第 50 次）和 SFT 选中的 checkpoint-770 | 996 | 1,543.7 GB | zoetrope 上 1 卡 sco 任务 `pt-5v3eh67l` 05:41–05:57，两批都 0 个不符 | 通过（`disk_runs/audit/b2324/CONCLUSION.md`：引用的都是已完成的评测链、训练链和已结束的下游实验；没有在跑的任务或进程读它们；STATUS 里没有标"在用"的；不碰别删清单） | 05:46 / 05:57 | 1,543.7 GB |
| b6_20261027 | Qwen2.5-VL：第十三轮 12 个续训状态 1,194.1；13 个最后但不是最好的权重 431.4；两次 3B 试跑 62.6 | 444 | 1,688.1 GB | 19:55 完成，0 个不符 | 通过（`disk_runs/audit/b67/CONCLUSION.md`） | 19:56 | 1,688.1 GB |
| b7_20261027 | 第十轮以前的 Qwen2.5-VL 旧权重 464.6（含登记区条目）；四个八月的旧 Cambrian 128.8 | 180 | 593.4 GB | 20:03 完成，0 个不符 | 通过（同上） | 20:03 | 593.4 GB |

## 5. 预测

10-27 00:00 按主线第 13 轮第二批和模型线正式 RL 重算（输入 `reports/forecast_plan_20261026d.json`，起点 10-26 23:47 配额表 16 TB）。重算命令：`python disk/forecast.py docs/disk/reports/forecast_plan_20261026d.json --out docs/disk/reports/forecast_20261026d`。

![按第 13 轮第二批和模型线正式 RL 重算的剩余空间预测](reports/forecast_20261026d/forecast.png)

- 主线：
  - 第一批 r13d、r13c1、r13c2 的 10 个 RL 已跑完 7 个，剩下 3 个凌晨 1–2 点跑完，都还没按 `retain` 删。现在 2.86 TB，评测完共删约 1.33 TB。
  - r13c3sft 的 SFT 已删到 32 GB。
  - 第二批新开 r13c3（2 个 RL）、r13b（2 个 RL），每个到第 300 次约 298.7 GB，评完各删 132.8 GB。
- 模型线：Qwen3-VL、Cambrian-S 的正式 RL 已在跑（4 个运行），plan 没设 `retain`，跑完中间检查点都留着，四个合计约 1.2 TB。
- 情形 A：最低剩余 5.07 TB，在 10-27 08:47 左右（第二批都写满、还没评测删减），之后约 5.1 TB。
- 情形 B（A 加上主线评测和删减晚 6 小时，再加 10-27 一批 4 个 RL）：最低 4.01 TB。
- 都不会碰 3 TB 线。

上一版（10-26 14:00，`forecast_plan_20261026c.json`，起点 13 TB）：

- 主线第 13 轮 11:28–11:44 开跑（`exps/vagen_active_spatial/interiorgs_v10_train_r13{d,c1,c2,c3sft}_20261026`）：
  - 10 个 8 卡 RL，每个跑到第 300 次更新，每 50 次存一个检查点，跑完每个约 298.7 GB；评测完按 `retain` 每个删 132.8 GB；
  - 另有 1 个 7B SFT。
- 情形 A（在跑的和已批的）：最低剩余 5.63 TB，在今晚 22:00 左右，也就是 RL 都跑完、还没评测删减的时候；之后回到约 6.5 TB。
- 情形 B（再加 Qwen3 的 RL、r13c3sft 之后可能的 RL）：最低 5.61 TB。
- 如果评测拖延、`retain` 没按时删，最低点再低约 1.3 TB，仍在 3 TB 以上。
- 第 13 轮结束后，10 个运行最后一个检查点的续训状态约 1 TB 仍会留着（保留规则 R1 要主线自动删，见转告）。
- 上一版（`forecast_plan_20261026b.json`，起点 08:44 的 12 TB）没算第 13 轮，最低是 8.33 / 7.36 TB。

## 6. 监控

在登录节点后台跑，每 30 分钟一次：

- 读配额表和文件系统统计，追加到 `docs/disk/usage.csv`；
- 剩余低于 3 TB 写 `disk_runs/WARN_DISK_LOW` 并记事件，低于 1.5 TB 再记一次，回到 3 TB 以上时删掉这个警告文件；
- 每天一次按一级目录记大小，写 `docs/disk/daily_top.csv` 和 `docs/disk/daily_growth.md`（哪条线涨得最快）；
- 事件记在 `disk_runs/monitor_events.jsonl`。

10-26 09:19 起带 `--stop-flag`：剩余低于 1.5 TB 写 `disk_runs/flags/STOP_SUBMIT_DISK_LOW`，回到 3 TB 以上删掉（你已批的约定，见 `STOP_FLAG_PROPOSAL.md`）。本机的监看（每 5 分钟读一次事件、有变化就报你）按你 10-27 的话已停；集群上的监控照常运行，事件和读数仍记在 `disk_runs/monitor_events.jsonl` 和 `docs/disk/usage.csv`。

断了用下面的命令重启（在 worktree 里执行）：

```bash
bash disk/monitor.sh start --stop-flag
```

```bash
bash disk/monitor.sh status
```

```bash
bash disk/monitor.sh stop
```

## 7. 要转告的事（不改其他线的文件）

给各线的完整转告（可以直接转发）：第一轮在 [RELAY_20261026.md](RELAY_20261026.md)，第二轮在 [RELAY_20261027.md](RELAY_20261027.md)，下面是第一轮的要点。
- 下游线（10-31 05:1x 回话）：b14 核对无误，要用的起点（DS-{B,S,RS,RD}-030-s1 的 run/final_model）和 E3 的导出都在。以后下游训完、评完的实验按 R3 由磁盘线处理：磁盘线定期查出清单交你批，批了再归档、删除；下游自己的脚本不删文件。
- 主线（10-31 回话，更正磁盘线）：b13 的三条链 q3s45、q3s0k、q3c3 都设了 `retain: best_and_last`，中间检查点评完已自动清掉（progress.json 的 retained 有 4、2、2 条）；磁盘线 10-31 告诉主线"这几条链没设 retain"是错的，b13 本身只删了最后检查点的续训状态，没错。真正没设 retain 的是 10-27、10-28 的 q3、q3s2dck、q3s3、q3dck、q3weak、q3fmt，它们的中间检查点在主线自己的清单里（主线第 85 条，31 项、782.2 GiB，`interiorgs_v10_q3_20261027/monitor/cleanup_q3_dryrun_20261031.txt`），删不删、要不要归档等你在主线定，定了主线告诉磁盘线。q3abl 设了 retain。
- P1 线（转移题线，11-01 交接文档，你转来）：它依赖、请别删的路径见第 8 节和 [PROTECTED_PATHS.tsv](PROTECTED_PATHS.tsv)；它的工作树 `VAGEN-Lite-data` 别切分支、提交或改文件。
- （已了结：JPEG 重评做完，DS-CSS 15 万步副本 11-01 作为 b18 删除；DS-CSB 的最终权重按规则一直留着。）原来的记录——**暂不列进待删清单（模型线 10-31 告知，评完它再说）**：`models_runs/downstream/experiments/DS-CSS-100-s1/evals_src/steps_150000/`（16.3 GB，评测任务 pt-2oaccrlf；评完后可按 R4 处理）、`DS-CSB-100-s1/run/final_model/`（16.3 GB，评测任务 pt-vbdyiybe）。模型线第 39 条：在闭环里重评，每帧先压成 JPEG 质量 80。

- 下游线：
  - 磁盘线已删 RL-SPEED-1、RL-SPEED-2 的检查点（先传了 AOSS，目录里留了说明文件）。
  - `evals_src`（134.8 GB）、E0 检查点（269.6 GB）、RLINF-SMOKE-1 检查点（134.8 GB）你也批了，但下游代码会读，所以磁盘线没动：
    - 看板 `copy_weights` 只看 `evals_src/…/source.json` 在不在，不看评测是否完成，副本删了会把这个点显示成"等全状态"；
    - 看板 `stage0_row` 按 `run_stopped_20261022/checkpoints` 里的文件列阶段 0 那几行；
    - `downstream/rlinf/export_test.sh` 的默认输入是 RLINF-SMOKE-1 第 2 步的权重。
  - 请下游线改好这几处后自己删，或者告诉磁盘线可以删。
- 主线：
  - 9 个组合运行最后一个检查点的续训状态已上传 AOSS 后删除，每个目录里写了 `RESUME_STATE_DELETED`，做法和你 10-25 的清理相同。
  - pilot r3 的分片、R1 taskfix 第 50–200 步也已上传后删除。
  - 两次 3B 试跑（62.6 GB）要不要删，等主线确认。
- 整理窗口：
  - `archive/cluster/` 下 6 个子目录（`interiorgs_v10_rounds`、`dense_score_pilot`、`r1`、`logs`、`data_intermediate`、`diagnostics_d0_u1`）已上传 AOSS 后删除，`archive/cluster/DELETED_TO_AOSS_20261026.txt` 写了去向，请更新 `archive/README.md`。
  - `VAGEN-Lite/models/` 下 8 个登记链接已断（10-25 主线清理删了目标）。
- 模型线：保留规则 R1、R2（见 `RETENTION_POLICY.md`）：正式 RL 的 plan 设 `"retain": "best_and_last"`；SFT 选完 epoch 后删没选中的，`model/` 下的最终模型只留一份。

## 8. 本线的规则和教训

- 默认只读。删除只按你批过的清单做，清单外一个文件都不删。
- 删除的流程：
  1. 先上传 AOSS 并逐文件核对大小，清单本地和 AOSS 各一份（`disk/archive_delete.py archive`）；
  2. 用整理窗口的 `scripts/cluster/audit_archived_paths.sh` 加补查，确认没人读；
  3. 删前再核对文件没变，打印完整列表和总大小再删，留逐文件记录（`disk/archive_delete.py delete --yes`）。
- 不碰的：依赖清单和各线 STATUS 里标为在用的目录、root 属主的、别人名下的、在跑的任务。不移动目录。代码会读的，即使在批准的清单里也先不删，交回那条线。
- **其他线依赖的路径，删前先问你（你 11-01 的规定）**：表在 [PROTECTED_PATHS.tsv](PROTECTED_PATHS.tsv)，来源是 P1 线（转移题线）交接文档"给其他窗口的注意事项"一节。
  - 内容：模型线的 `models_runs/sft_v10_newformat_20261024`、`models_runs/chains/m_qwen3vl8b_formal_*`、`models_runs/eval_sets`；主线的 r12o `lf_format`、r12 `format_sft_v2a`、r11 `sft`、`VAGEN-Lite-r12dense`、`.venv-qwen3vl-runtime`；QA 线的 `interiorgs_qa_lf_20261015/sample_256`；P1 自己的工作树 `VAGEN-Lite-data`（含 `exps/c0_restore_20261029`，66 GB，你 10-29 说先留着）。
  - 哪个批次涉及这些路径，先单独问你，即使那条线已经批了。`archive_delete.py` 的 `plan` 和 `delete` 碰到这些路径就拒绝，你同意后把你的原话写进批次说明的 `protected_ok` 才能跑。
  - 11-01 对过以前的记录：b5、b9 删过 `m_qwen3vl8b_formal_*` 里的东西，都是在这张表出来以前、按当时批的清单删的。删的是 C0 四个 RL 检查点（q3s0 第 150 步，q3d、q3s2s0、q3s2d 第 300 步）的续训状态，每个 105.2 GB；还有 SFT 没选中的 epoch，其中 154、308、462、616 在 10-28 已取回。P1 要用的都还在：四个 RL 检查点的 HF 权重（各 35.1 GB）、SFT `checkpoint-1232`（34.0 GB）、对话和 `train.yaml`。别的行没有批次碰过；b18 也不碰。
- **sco 任务投哪个池（你 11-02 的指示，主实验转告）**：上午定的是一律投 zoetrope；19:08 起放宽，排队中的任务可以转到 h800。磁盘线以后的上传、删除任务，提交前看 h800 和 zoetrope 哪个排队少就投哪个（`bash disk/submit_archive_delete_job.sh POOL BATCH ...`），拿不准就问你。
- 不用 GPU。你批过的 root 属主目录删除，用 1 卡 root 任务（`disk/submit_root_delete.sh`，先查 `PAUSE_SUBMISSIONS`），删完删掉任务记录。
- **登录节点的核数上限（你 10-28 的规定，用户级 `CLAUDE.md`；同日补充放宽）**：账号配额 14 个核，所有窗口共用。
  - 常驻监控 `monitor.py`：用 `nice -n 19`、单线程起（`monitor.sh` 已改）；10-26 起在跑的那个已调成 `nice` 19。
  - 扫描（`xtree.py`、`walk.py`、`freeable.py`）、写死路径核查、`quota.py` 可以在登录节点上跑，但：
    - 每个任务最多 4 个核（进程数 × 每个进程的线程数）；
    - 线程环境变量设好（`OMP_NUM_THREADS` 等，加 `TOKENIZERS_PARALLELISM=false`）；
    - 开跑前用 `/sys/fs/cgroup/cpu.stat` 看已用核数，"已用 + 这个任务"不超过 12 个核；
    - 超过几分钟的加 `nice -n 19`。
  - 上传 AOSS：`archive_delete.py archive` 默认 `--files 8 --threads 8`（最多 64 个线程），在登录节点上要降到 4 个核以内，比如 `--files 2 --threads 2`，速度会明显变慢。大批量（几百 GB 以上）改用 1 卡 sco 任务（14 个 vCPU），先写好提交脚本（照 `submit_root_delete.sh`），跑完删掉任务。
  - 10-26、10-27 两轮的上传（8 个文件 × 8 线程）是在这条规定之前跑的。
- 递归扩展属性由服务器现算：一次只发一个查询；绝不对 `/mnt/umm` 或 `/mnt/umm/users` 求和。
- AOSS 的 `list_objects_v2` 翻页不可靠，列对象一律用 v1 的 `Marker` / `NextMarker`。
- 通过 ssh 传给集群的命令里不能有英文撇号（会截断单引号）；带文字的文件先在本地写好再传。
