# 任务 B 的归档索引（2026-11-01）

任务 B（真实剩余步数做进度分）10-30 收口，结论是"不涨"。要引用请用 `B_SUMMARY.md`。这一页只列 B 留下的东西在哪，文件没挪。

## 报告（`docs/reward/`）

| 文件 | 内容 |
| --- | --- |
| `PREREG_B.md` | 预注册的比较口径、"涨"的判据、补种子的规则 |
| `reports/STEP1_ASSETS_20261027.md` | 资产核对 |
| `reports/STEP2_TABLES_20261027.md` | 深度 12 距离表：标定、粗筛、代价 |
| `reports/STEP3_RUNS_20261029.md` | 第一、二波的配置核对和前两次更新的查表核对 |
| `reports/STEP4_WAVE1_20261029.md` | 第一波（O、S0）结果 |
| `reports/STEP4_WAVE2_20261030.md` | 第二波（G、O-keep）结果 |
| `reports/STEP4_B_RESULTS_20261030.md` | 总报告（含种子 2） |
| `B_SUMMARY.md` | 给主线引用的一节 |
| `reports/DRIFT_S0_20261101.md` | S0 走样的机制和全部 34 个 Qwen3 运行的走样表（B 之后的分析） |

## 数据（`/mnt/umm/users/yinbaiqiao/reward_runs/`）

| 路径 | 内容 |
| --- | --- |
| `tasks/rows_b3000.jsonl` | 两波和种子 2 共用的 RL 题目文件（3,000 道有表的后备题） |
| `tables/b3000`、`tables/eval705` | 深度 12 距离表。eval705 只读，给主线用。AOSS 副本在 `fj:s3://baiqiao/vagen_lite_archive/reward_runs/tables/`，11-01 上传，大小逐个核过 |
| `runs/rw_b_w1`、`rw_b_w2`、`rw_b_s2` | 三条训练链：plan、PROGRESS、RL 日志、训练记录（rollout_data） |
| `eval_sets/eval_val_v2`、`eval_test_v2` | 评测结果 `rw_<臂>_<更新次数>`：每个运行每 50 次一个检查点的结果都在（rw_s0 到第 200 次，其余到第 300 次） |
| `analysis/2*`–`analysis/6*` | 各步的核对和分析输出 |
| `deletions/*.tsv` | 三次删除的逐项清单（10-29、10-30、11-01） |

## 留下的检查点（只剩 `actor/huggingface` 的权重）

| 运行 | 检查点 | 说明 |
| --- | --- | --- |
| rw_s0 | 第 150、200 次 | 150 是验证集最好的；200 是被停前最后一个，续训状态已删 |
| rw_o | 第 200、300 次 | 200 最好；300 最后，续训状态已删 |
| rw_g | 第 250、300 次 | 同上 |
| rw_ok | 第 250、300 次 | 同上 |
| rw_s2s0 | 第 300 次 | 最好也是最后，续训状态已删 |
| rw_s2ok | 第 250、300 次 | 250 最好；300 最后，续训状态已删 |

第二阶段和 rw_s0、rw_g 比的是它们第 50 / 100 / 150 次的评测结果，结果都在，不需要这些检查点。

## 代码（分支 `reward-design`）

- B 用的工具在 `reward/`：distance12、replay_q3、make_plans_b、compare_b、two_seed_b、miss_shift、fail_end_d、unique_signal 等。
- 每个文件开头都写了用法。
