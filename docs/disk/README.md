# 磁盘配额线（docs/disk）

这条线负责 `/mnt/umm/users/yinbaiqiao`（个人目录）的配额：量、预测、提方案、按批准的清单执行。默认只读。

- 状态：[STATUS.md](STATUS.md)（唯一的状态文档）
- 报告：[reports/](reports/)
- 脚本：仓库根目录 `disk/`（只新增文件，分支 `disk`，worktree `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-disk`）
- 大文件和原始扫描结果：`/mnt/umm/users/yinbaiqiao/disk_runs/`（不进 git）

| 脚本 | 做什么 | 写什么 |
| --- | --- | --- |
| `disk/quota.py` | 配额、已用、剩余三个数 | 不写 |
| `disk/xtree.py` | 用文件系统的递归扩展属性取目录大小（不逐个文件 stat） | `--out` 指定的 TSV |
| `disk/walk.py` | 一次完整遍历（等于 du，外加时间、大文件、清理记录） | `--out` 目录 |
| `disk/classify.py` | 把 xtree 结果按线、按类别汇总，找重复的大文件 | `--out` 目录 |
| `disk/forecast.py` | 按计划任务算 7 天剩余空间曲线 | `--out` 目录 |
| `disk/aoss_check.py` | AOSS 桶的用量、归档清单核对、上传下载测速 | 只有 `upload` 子命令会在 AOSS 上建对象 |
| `disk/run_inventory.sh` | 依次跑 xtree、walk、du | `disk_runs/raw/inventory_<时间>/` |
| `disk/archive_delete.py` | 一个批准批次的归档和删除：`plan`（列完整清单和总大小）、`archive`（上传 AOSS、逐文件核对大小、清单本地和 AOSS 各一份）、`delete --yes`（删前再核对、打印清单、删、逐文件记录、写说明文件） | `disk_runs/batches/<批次>/`；AOSS `vagen_lite_archive/` |
| `disk/protected.py` | 其他线依赖、删前要先问你的路径（[PROTECTED_PATHS.tsv](PROTECTED_PATHS.tsv)）：`archive_delete.py` 的 `plan` 和 `delete` 碰到就拒绝，除非批次说明里写了 `protected_ok`（你的原话）；`records` 子命令拿所有批次的删除、恢复记录和没执行的计划对这张表 | 不写 |
| `disk/monitor.py`、`disk/monitor.sh` | 常驻监控：每 30 分钟记配额表和文件系统统计，低于 3 TB 写警告文件，每天按一级目录记增量 | `docs/disk/usage.csv`、`daily_top.csv`、`daily_growth.md`；`disk_runs/monitor_*` |

批次说明（批准了删什么、留什么、说明文件写在哪）放在 `docs/disk/batches/<批次>.json`。

注意（2026-10-26 的教训）：递归扩展属性由服务器现算，代价和子树里的目录数成正比。一次只跑一个；绝不对 `/mnt/umm` 或 `/mnt/umm/users` 求和（约 1.8 亿个条目，会让文件服务器对所有人变慢，进程卡在不可中断状态直到算完）。
