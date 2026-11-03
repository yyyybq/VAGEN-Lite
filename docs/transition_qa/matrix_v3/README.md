# 矩阵第三版的分析结果（副本，2026-11-03）

原件在集成分支的工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/` 下，`exps/` 不在 git 里，所以在这里留一份。结论和读法见 `../REPORT.md` 开头"矩阵第三版的结论"，过程见 `../PROGRESS.md` 12.28–12.45。

| 文件 | 内容 | 原件 |
| --- | --- | --- |
| `main_31.{json,md}` | 预登记 3.1 主指标：D − C、D − C0、E − C0 | `analysis_20261103/` |
| `rest.{json,md}` | 3.2 按族、3.3 探针、3.4 失败拆解、SFT 阶段、第 5 节"稳不稳" | `analysis_20261103/` |
| `probe/compare_*.{json,md}` | 3.3 的五个探针比较（`qa_probe.compare` 的输出） | `analysis_20261103/probe/` |
| `matched.{json,md}` | 探索性：D 取和 C 相同的更新次数；各检查点的测试集成功题数 | `analysis_20261103/` |
| `card_hours.json` | 每个作业的卡时（190 个作业）和按组的合计 | `analysis_20261103/` |
| `handover_main_20261103.{md,json}` | 交给主线的：动作成功率两种口径的定义和数，权重的只读路径 | 本分支生成 |
| `accept_20261103/` | 补充工作 A.2：三个 SFT 起点的采样验收（`accept_summary.json` 是主线 `accept_summary.py` 的输出，`table.md` 是评测表，`plan.json` 是只做评测的链） | `accept_20261103/`、`eval_accept_reserve200/table.md` |
| `scripts/m3_main31.py`、`m3_prereg_rest.py`、`m3_matched.py`、`m3_cardhours.py` | 生成上面这些文件的脚本 | `watch/` |
| `scripts/m3_accept.py`、`m3_cells.py`、`m3_handover.py` | A.2 的验收链、三个新格子的值守脚本、交接文件 | `watch/` |

脚本里的路径写死在集群上（`VAGEN-Lite-matrix`、`VAGEN-Lite-data`），要在 `exps/matrix_v3` 下运行，命令见 `../REPORT.md` 第 5 节。它们只读评测和探针结果，不提交任务。
