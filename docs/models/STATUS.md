# 模型线状态（唯一的状态文档）

最后更新：2026-10-24 18:55 UTC。

模型线：把 Qwen3-VL-8B-Instruct 和 Cambrian-S（7B 为主）接进统一框架，在主线最高的 setting 上跑主实验和下游实验。报告在 [reports/](reports/)，实验登记表 `MATRIX.md` 第三步建。

## 1. 现在在哪一步

- **第一步（盘点与核实）已完成，停下来等你确认。** 报告：[reports/STEP1_INVENTORY_20261024.md](reports/STEP1_INVENTORY_20261024.md)。
- 等你定报告末尾"需要你决定的事"的 8 项。最要紧的是第 1 项（Cambrian-S 的权重从哪来）和第 2、3 项（Qwen3-VL 的 SFT 工具和运行环境），它们决定第二步能不能开始。
- 第二步（接入与冒烟）在你确认后开始。

## 2. 谁在跑

**训练**

| 编号 | 内容 | 状态 | sco 任务 | 更新（UTC） | 卡在哪 |
| --- | --- | --- | --- | --- | --- |
| M-qwen3vl8b-sft | Qwen3-VL-8B 的 SFT | 未提交 | — | 10-24 19:00 | 等第二步（等决定 2、3） |
| M-qwen3vl8b-s0 / -d | Qwen3-VL-8B 的 RL 两臂 | 未提交 | — | 10-24 19:00 | 等 SFT |
| M-cambrians7b-sft | Cambrian-S-7B 的 SFT | 未提交 | — | 10-24 19:00 | 等权重（决定 1）和代码（决定 4） |
| M-cambrians7b-s0 / -d | Cambrian-S-7B 的 RL 两臂 | 未提交 | — | 10-24 19:00 | 同上 |
| M-qwen25vl7b-sft / -s0 / -d | Qwen2.5-VL-7B 参照行（新格式重跑） | 未提交 | — | 10-24 19:00 | 排在两个新模型之后 |
| DS-Q3B-100-s1 | 下游：原始 Qwen3-VL-8B | 未提交 | — | 10-24 19:00 | 等第二步的 StarVLA 加载测试 |
| DS-Q3S / Q3RS / Q3RD-100-s1 | 下游：Qwen3-VL 的 SFT、RL 两臂 | 未提交 | — | 10-24 19:00 | 等第三步的检查点 |
| DS-CS*-100-s1 | 下游：Cambrian 四个起点 | 不能做 | — | 10-24 19:00 | StarVLA 没有 Cambrian 接口（决定 5） |

**评测**

| 编号 | 内容 | 状态 | sco 任务 | 更新（UTC） | 卡在哪 |
| --- | --- | --- | --- | --- | --- |
| STEP1-LOAD | 1 卡加载测试（Qwen3-VL、Cambrian，HF 和 vLLM） | 已完成 | `pt-kqhed0xd`（18:19–18:28）、`pt-gm3kn6l8`（18:29–18:35，vLLM 补跑） | 10-24 18:35 | — |
| M-qwen3vl8b-base | Qwen3-VL-8B 零样本 705 道（温度 0.8） | 未开始 | — | 10-24 19:00 | 第二步 |
| M-cambrians7b-base | Cambrian-S-7B 零样本 705 道 | 未开始 | — | 10-24 19:00 | 等权重 |

现在没有在跑或排队的任务。

## 3. 权重和数据

| 内容 | 路径 | 说明 |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | `/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 只读；分片只有 root 能读（sco 任务里读得到） |
| Cambrian-S-7B | **集群上没有** | 要从外面拷，32.13 GB；另要 SigLIP2-so400m-patch14-384，4.58 GB |
| Cambrian 外部源码 | `/mnt/umm/users/yinbaiqiao/cambrian-s` | 只读 |
| 8 月 RL 训过的 Cambrian 检查点 | `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/models/historical/cambrian_*` | 只用来测代码，不是起点 |
| Qwen2.5-VL-7B 原始 | `E/interiorgs_v10_r11_20261016/inputs/base_7b` | 参照行起点 |
| SFT 数据来源（C0） | `E/interiorgs_v10_r11_20261016/sft/sft_trajectories.jsonl` | 旧格式；第二步按新格式重导出到本线目录 |
| RL 题 | `E/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl`，排除 `E/interiorgs_v10_r10_20261015/depth_check/depth_check_summary.json` | 后备题 10,798 道 |
| 评测集 | `E/interiorgs_v10_r12_20261019/eval_val_v2`（413）、`eval_test_v2`（292） | 结果名带模型前缀，提交前查重名 |
| 密集变体汇总 | `E/interiorgs_v10_r12_20261019/analysis/variants/variants.json` | 提交 RL 时再读一次 |

`E` = `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial`。

## 4. 实验表

| 编号 | 模型 | 起点 | 内容 | 状态 | 结果 |
| --- | --- | --- | --- | --- | --- |
| M-qwen3vl8b-base | Qwen3-VL-8B | 原始 | 零样本 | 未开始 | — |
| M-qwen3vl8b-sft | Qwen3-VL-8B | 原始 | SFT（C0 的数据来源和配方，新格式） | 未开始 | — |
| M-qwen3vl8b-s0 | Qwen3-VL-8B | 本线 SFT | RL 稀疏臂（`v10_p1_S0`） | 未开始 | — |
| M-qwen3vl8b-d | Qwen3-VL-8B | 本线 SFT | RL 密集臂（提交时按汇总选，现在是 V4） | 未开始 | — |
| M-cambrians7b-base / -sft / -s0 / -d | Cambrian-S-7B | 同上 | 同上 | 等权重 | — |
| M-qwen25vl7b-sft / -s0 / -d | Qwen2.5-VL-7B | 同上 | 同上（参照行） | 未开始 | — |

## 5. 代码和输出

- worktree：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models`，本地分支 `models`，从 `origin/main` `00363333` 开出，不跟踪远端。代码整理期间不推 main，本地小步提交。
- 本线只新增：`model_matrix/`（代码和测试）、`docs/models/`（状态、报告、登记表）。
- 大文件和任务输出：`/mnt/umm/users/yinbaiqiao/models_runs/`。
- 整理窗口问到下面这些目录，一律回答"不能动"：
  - `VAGEN-Lite-models`、`models_runs/`、`cambrian-s/`；
  - `VAGEN-Lite/.venv-qwen3vl-runtime`；
  - `VAGEN-Lite/models/historical/cambrian_*`；
  - 共享目录里的 Qwen3-VL-8B。

## 6. 规则

- GPU：bq_tp_mmxbf16 和 bq-ready-* 最高，不动、不问；其次主线的密集奖励实验和 A 组；本线和下游同级，谁先排队谁先上，冲突时记在这里并告诉你；其他扫描最后。
- 暂停文件 `/mnt/umm/users/yinbaiqiao/downstream_runs/PAUSE_SUBMISSIONS` 存在时本线不提交新任务。
- 从集群外拷东西、下载大文件、安装新环境、调用付费接口、删除文件、转换或导出权重产生新副本，都先问你。
