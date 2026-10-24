# 模型线状态（唯一的状态文档）

最后更新：2026-10-24 19:45 UTC。

模型线：把 Qwen3-VL-8B-Instruct 和 Cambrian-S（7B 为主）接进统一框架，在主线最高的 setting 上跑主实验和下游实验。报告在 [reports/](reports/)，实验登记表 `MATRIX.md` 第三步建。

## 1. 现在在哪一步

- 第一步（盘点与核实）已完成，你 10-24 确认。报告：[reports/STEP1_INVENTORY_20261024.md](reports/STEP1_INVENTORY_20261024.md)。
- **第二步（接入与冒烟）进行中**（10-24 19:40 起）。顺序：Qwen3-VL 接入 → Qwen3-VL 冒烟 → Qwen3-VL 的 StarVLA 骨干 → Cambrian-S 代码（权重到了再冒烟）→ Cambrian-S 的 StarVLA 骨干。下载、CPU 上的代码和 GPU 冒烟并行。
- 报告写到 `reports/STEP2_SMOKE_<日期>.md`。

## 1a. 你的决定（10-24，第一步报告之后）

1. **Cambrian-S 权重**：同意在登录节点经 hf-mirror 下载 Cambrian-S-7B（32.13 GB）和 SigLIP2-so400m-patch14-384（4.58 GB），放到共享模型目录下本线的子目录（`/mnt/umm/shared_model/vagen-lite-models/hub`，HF 缓存的目录结构）；下载后核对 sha256（来源没有就核对大小和分片清单），记录来源地址、修订号、时间。只下这两样，再要下别的先问。
2. **Qwen3-VL 的 SFT**：在现有的 LLaMA-Factory（`ca75f1e`）里用本线新文件登记 Qwen3-VL 的对话模板和多模态插件，不装新包，在 `.venv-qwen3vl-runtime` 里跑。
3. **Qwen3-VL 的 RL 和评测**用 `.venv-qwen3vl-runtime`；主环境一个包都不改。
4. **Cambrian-S 全做**：SFT、两个 RL 臂、评测、下游四个起点。
5. **StarVLA 适配 Cambrian-S**（新文件加开关，用 Cambrian-S 自己的模型类和处理器；估代价不超过 4 天就直接做）。
6. **KL 停止线用主线标准的 0.2**，和 C0、V4 一致（简报里的 1.0 是 A 组的规则，不用）。
7. **V4 的 keep 门槛沿用 0.3318、下限 1.0**。另外用新模型的 SFT 模型回放全失败题，算出它自己的前 20% 门槛，写进报告但不用；逐次记录补进来的题数，到第 50 次平均每次不到 1 道就停下来报告。
8. **Qwen3-VL 权重的 sha256** 放进第二步第一个 sco 任务核对。
9. **之前定的继续有效**：每个模型 RL 两臂（S0 对照、密集为主）；密集变体在提交 RL 时按主线最新汇总取最高（现在是 V4）；SFT 用新格式；下游四个起点都做，原始模型的起点骨干一通就提；和下游同级；Qwen2.5-VL 新格式参照行排最后。
10. **Cambrian-S 每帧 756 个 token 的开销先量再定**：主实验按估算做；下游若一个起点的训练时间超过 Qwen2.5-VL 同样任务（约 42 小时）的 3 倍，停下来报告再提。

执行记录：

- dense-batch2 的 6 个提交按顺序拣进 `models`（10-24 19:41），没有冲突：`aa6451e8→7fe42c0d`、`ec7682ec→f7bce387`、`991cc92d→db55eba1`、`f8ba48c5→cd812b25`、`24f1272e→21294b24`、`418927c4→0fbc32b9`。
- worktree 的 `verl` 子模块从主 checkout 的 verl 本地克隆，检出 `a0ae85c`（没联网）。

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
