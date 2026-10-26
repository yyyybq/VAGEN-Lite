# 整理任务：你对第一阶段方案的决定（2026-10-16）

对应 [PHASE1_INVENTORY_20261015.md](PHASE1_INVENTORY_20261015.md) 第 8 节的 15 项。"推荐"指报告里标推荐的选项。

| 编号 | 决定 | 执行要点 |
| --- | --- | --- |
| D1 | (b) 重写 git 历史，**放最后做**，并保存新旧提交号对照表 | 整理分支合并后做；重写前在集群留镜像；对照表写入 `docs/reports/COMMIT_MAP_<日期>.tsv`，文档里引用的提交号同时换成标签 |
| D2 | 推荐：上游 VAGEN 环境与示例移到 `archive/upstream/` | |
| D3 | 推荐：轨迹评测统一到 v10 `evaluation.py`，旧评测栈归档 | `evaluation/metrics.py` 里被 `train_chain.py` 用到的部分先并入保留代码 |
| D4 | 冒烟之后（2026-10-16 晚）定为 **(a)**：Cambrian-S、SenseNova-U1、Qwen3-VL 三条旁支的代码和文档归档，`.venv-qwen3vl` 留在磁盘；实验矩阵的起点模型只有 Qwen2.5-VL | 冒烟结果见 [SMOKE_DOWNSTREAM_20261016.md](SMOKE_DOWNSTREAM_20261016.md)。文档已归档到 `archive/side_models/`；`vagen/models/` 要连同核心训练循环里导入它的分支一起移走，放到第十一轮结束后的代码搬动批次 |
| 主线尺寸 | **7B**（Qwen2.5-VL-7B-Instruct）；3B 只用于单卡试跑 | 第三、四阶段的默认配置用 7B。RLinf 现有配置接的是 Qwen3-VL-4B 底座的 StarVLA 策略，要换成 Qwen2.5-VL-7B 的策略需要一次 8 卡 StarVLA 训练，属于正式实验，等你安排 |
| 冒烟残留 | `archive/cluster/_pending_delete/`（45 个条目，230 GB）**删除** | 2026-10-16 已删；逐文件记录 `archive/cluster/DELETED_pending_delete_20261016.tsv` |
| D5 | 推荐：文本 12 维动作路线代码归档，49 GiB smoke 检查点删除 | |
| D6 | 推荐：导出 `R1-clean-Projective-v0-taskfix` 第 250 步 HF actor 后删除全部分片；冻结数据、发布 json、报告归档 | |
| D7a | 推荐：`interiorgs_v10_train_20261014/rl_run` 只留 7 个 HF actor 和日志；按新增第 3 条，第 250 步的续训状态也保留 | |
| D7b | **先出结论报告，不删** | 见 [DENSE_SCORE_PILOT_R3_CONCLUSION_20261016.md](DENSE_SCORE_PILOT_R3_CONCLUSION_20261016.md)。2026-10-26 由磁盘线先传 AOSS 再删（用户 10-26 的决定），记录见 `archive/README.md` |
| D7c | 推荐：第五轮试跑只留 3B SFT 模型，删 RL 试跑分片 | |
| D8 | 推荐：`cjepa`、`spatial-training-room`、`ele` 移到 `~/` | 不重装可编辑包；CLUSTER.md 记录 |
| D9 | 推荐：StarVLA NavigateKitchen 检查点保留到下游冒烟后 | 2026-10-26 由磁盘线先传 AOSS 再删；示例脚本指向的目录现在是空的，要用先从 AOSS 恢复 |
| D10 | 推荐：历史主线权重登记进 `models/historical/` | |
| D11 | **"自有 QA" = `interiorgs_qa`**，含三部分：① 仿 8 个 benchmark 的 51 个子任务（训练和测试都有）；② 专门为训练造的简单 Yes/No QA；③ 专门为测试造的 RPA（Read / Perception / Action）QA | `data_gen/active_spatial_qa` 的成功谓词 Yes/No 转换器是 RPA 的 Read 测试（COVERAGE.md 第 1 节），**不归档**；三部分在 DATA.md / EVAL.md 里分开写 |
| D12 | 推荐：删除 mkdocs / readthedocs 配置，README 重写 | |
| D13 | 推荐：`data/releases/`、`models/`、`results/` 在仓库目录下，清单入 git | |
| D14 | (a) 现在就做不与训练链冲突的部分 | 代码搬动等第十轮训练链结束 |
| D15 | 推荐：另一会话的 QA 代码完工前原位不动 | |

另加三件：

1. **先做 RLinf、StarVLA、EASI 的单卡冒烟**，核查支持的模型尺寸后再定 D4 和主线用 3B 还是 7B。
2. **OOD 轨迹的两个轴**：没见过的物体类别；换一种说法的指令。
3. **实验检查点只留可直接加载的权重（HF 格式），续训用状态只留最后一步。** 这条写进 TRAINING.md 和训练链的保存策略。
