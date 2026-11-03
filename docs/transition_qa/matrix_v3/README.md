# 矩阵第三版的分析结果（副本，2026-11-03）

原件在集成分支的工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/` 下，`exps/` 不在 git 里，所以在这里留一份。结论和读法见 `../REPORT.md` 开头"矩阵第三版的结论"，过程见 `../PROGRESS.md` 12.28–12.39。

| 文件 | 内容 | 原件 |
| --- | --- | --- |
| `main_31.{json,md}` | 预登记 3.1 主指标：D − C、D − C0、E − C0 | `analysis_20261103/` |
| `rest.{json,md}` | 3.2 按族、3.3 探针、3.4 失败拆解、SFT 阶段、第 5 节"稳不稳" | `analysis_20261103/` |
| `probe/compare_*.{json,md}` | 3.3 的五个探针比较（`qa_probe.compare` 的输出） | `analysis_20261103/probe/` |
| `matched.{json,md}` | 探索性：D 取和 C 相同的更新次数；各检查点的测试集成功题数 | `analysis_20261103/` |
| `card_hours.json` | 每个作业的卡时（190 个作业）和按组的合计 | `analysis_20261103/` |
| `scripts/m3_main31.py`、`m3_prereg_rest.py`、`m3_matched.py`、`m3_cardhours.py` | 生成上面这些文件的脚本 | `watch/` |

脚本里的路径写死在集群上（`VAGEN-Lite-matrix`、`VAGEN-Lite-data`），要在 `exps/matrix_v3` 下运行，命令见 `../REPORT.md` 第 5 节。它们只读评测和探针结果，不提交任务。
