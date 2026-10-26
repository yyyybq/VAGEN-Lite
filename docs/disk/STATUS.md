# 磁盘配额线状态（唯一的状态文档）

最后更新：2026-10-26 07:20 UTC。

这条线管 `/mnt/umm/users/yinbaiqiao`（个人目录，配额 22 TB）：量、预测、提方案、按批准的清单执行，默认只读。

- 报告：[reports/](reports/)，最新的是第一步报告 [reports/DISK_INVENTORY_20261026.md](reports/DISK_INVENTORY_20261026.md)。
- 脚本：仓库根目录 `disk/`，说明见 [README.md](README.md)。
- worktree：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-disk`，本地分支 `disk`，从 `origin/main` `00363333` 开出，不跟踪远端，不推送。
- 原始扫描结果：`/mnt/umm/users/yinbaiqiao/disk_runs/`。

## 1. 现在在哪一步

**第一步（盘点与预测）做完，停下等你批**：报告末尾"需要你决定的事"共 7 项。第二步（清理、保留规则）和第三步（常驻监控）都还没开始。

## 2. 汇报表

以后每次汇报都用这张表，行列不变，只改数字和时间；没有变化不汇报。

| 项 | 数值 | 时间（UTC） | 来源和误差 |
| --- | --- | --- | --- |
| 配额 | 21.99 TB | 10-26 07:00 | 配额表"22 TB"，按 20 TiB 算（推断，待管理员确认） |
| 已用（配额表） | 16 TB | 10-26 07:00 | 两位有效数字，即 15.5–16.5 TB；配额按它算 |
| 已用（文件系统统计） | 14.23 TB | 10-26 06:57 | `quarkfs.dir.rfbytes`，精确到字节，但比配额表少 1.6–2.3 TB，原因未查清 |
| 剩余（保守） | 5.49 TB | 10-26 07:00 | 21.99 − （配额表读数 + 0.5）；报警按这个 |
| 剩余（估计） | 5.5–6.2 TB | 10-26 07:00 | 21.99 − （文件系统统计 + 差额 1.58–2.27） |
| 过去 1 小时净增 | 0.31 TB | 06:06–06:57 | 文件系统统计之差，全部来自模型线 |
| 7 天内最低剩余：情形 A / B | 3.25 / 2.16 TB | 预测于 10-26 07:00 | 报告第 4 节；B 在 10-27 02:00 跌破 3 TB |
| 已批清单执行 | 0 / 0 TB | — | 还没有批准的清单 |
| 监控 | 未启动 | — | 等决定 5 |

## 3. 各线占用（10-26 06:34–07:03，文件系统统计）

| 线 | 占用（TB） | 主要目录 | 现在写不写 |
| --- | ---: | --- | --- |
| 主线 | 2.95 | `VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_*` | 不写（无任务） |
| 主线旧产物（pilot r3、R1） | 2.20 | 同上目录下 `dense_score_reward_only_pilot_parallel_r3_20261009`、`R1-*`、`r1_h1_aoss_repair_*` | 不写 |
| 整理窗口和主 checkout 其他 | 0.76 | `VAGEN-Lite/{archive,models,third_party,data_gen,playground}`、`VAGEN-Lite/exps/{unified_eval_runs,interiorgs_qa_*}` | 不写 |
| 下游 | 2.63（+ 借用 0.54） | `downstream_runs/`（借用 `probe_spatial/robocasa365`） | 写（2 个训练、3 个评测） |
| 模型线 | 0.92 | `models_runs/` | 写，最快（两个正式 SFT、两个下游、两条测试链） |
| 评测线 | 0.16 | `eval_external_assets/` | 不写 |
| 数据线 | 0.004 | `VAGEN-Lite-data` | 不写 |
| probe_spatial 项目（不在列出的线里） | 3.13 | `probe_spatial/`、`InsScene-15K/` | 34 个任务今早刚结束 |
| 归属不明 | 0.47 | `SenseNova-U1/`、`Consistency/` | 不写 |
| 共用（环境、缓存、场景数据、worktree） | 0.43 | `.cache`、`.conda`、`InteriorGS_ply` 等 | — |
| 合计 | 14.23 | | |

## 4. 预测

见报告第 4 节和下图（输入 `reports/forecast_plan_20261026.json`，重算：`python disk/forecast.py docs/disk/reports/forecast_plan_20261026.json --out docs/disk/reports/forecast_20261026`）。

![剩余空间预测](reports/forecast_20261026/forecast.png)

## 5. 清单执行情况

| 清单 | 内容 | 大小 | 状态 |
| --- | --- | ---: | --- |
| 5.1 立刻可删 | 评测线的公开模型和数据集下载 | 0.15 TB | 等批 |
| 5.1b 满 7 天即可删 | 主线组合链续训状态、下游 RL 测试检查点、下游 E1 权重副本 | 1.64 TB | 等批（10-29 起陆续到期） |
| 5.2 归档后可删 | pilot r3 分片、下游全状态、E0 检查点、R1、整理归档区、StarVLA NavigateKitchen、主线 3B 试跑 | 3.84 TB | 等批；5.2.2 等下游线确认，5.2.7 等主线确认 |
| 整理窗口移交项 | D7b → 5.2.1；D9 → 5.2.6；其余（D5、D6、D7a、D7c、D8、D10、冒烟残留）已由整理窗口做完 | — | — |
| AOSS 测速对象 | `fj:s3://baiqiao/disk_line_speedtest_20261026/base7b_shard1.safetensors` | 3.9 GB | 列进第一批 |

## 6. 监控

还没启动（等决定 5）。启动、停止、查看的命令写在这里，断了按这里重启。

## 7. 要转告的事（不改其他线的文件）

- 整理窗口：`VAGEN-Lite/models/` 下 8 个登记链接已断（`interiorgs_v10_rl_r6b_qwen25vl7b_step*` 7 个、`interiorgs_v10_sft_r6b_qwen25vl7b`），目标在 10-25 主线清理时删掉了。
- 模型线：正式 SFT 选完 epoch 后，没选中的 epoch 和 `model/` 下多存的最终模型约 0.65 TB 可清。
- 下游线：5.1b、5.2.2、5.2.3 涉及它的目录，执行前要它确认。
- 主线：5.1b（组合链续训状态）、5.2.7（3B 试跑）涉及它的目录。

## 8. 本线的规则和教训

- 默认只读。删除只按你批过的清单做，清单外一个文件都不删。删前先归档到 AOSS 并逐文件核对大小（清单本地和 AOSS 各一份），再用整理窗口的 `scripts/cluster/audit_archived_paths.sh` 确认没人读，最后打印完整列表和总大小后再删。
- 不碰的：依赖清单和各线 STATUS 里标为在用的目录、root 属主的、别人名下的、在跑的任务。不移动目录。
- 不用 GPU，不提交 sco 任务（测 AOSS 上传要用 sco 时先问）。
- 递归扩展属性由服务器现算：一次只发一个查询；绝不对 `/mnt/umm` 或 `/mnt/umm/users` 求和。10-26 05:55 误对整卷发过一次，文件服务器慢了约 40 分钟；查询进程卡在不可中断状态，要等服务器算完才退出。
- AOSS 的 `list_objects_v2` 翻页不可靠（`IsTruncated` 为真却没有续页令牌），列对象一律用 v1 的 `Marker` / `NextMarker`（`disk/aoss_check.py`）。
