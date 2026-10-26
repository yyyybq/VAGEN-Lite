# VAGEN-Lite 状态（唯一的状态文档）

最后更新：2026-10-26（整理任务第二阶段，冻结期代码搬动三批做完）。各条工作线的细节在各自的文档里，这里只汇总"现在在哪一步、谁在跑、下一步等什么"。

## 1. 整理任务（把仓库整理成可发布版本，把 7 步流程穿起来）

- **第一阶段（盘点与方案）** 已完成：[docs/reports/PHASE1_INVENTORY_20261015.md](docs/reports/PHASE1_INVENTORY_20261015.md)。你的决定记录在 [docs/reports/DECISIONS_20261016.md](docs/reports/DECISIONS_20261016.md)。
- **第二阶段（整理）** 在分支 `cleanup/20261016` 上做，独立检出 `~/VAGEN-Lite-cleanup-wt`。分支的代码停在 `9af4796c`（之后只有报告和 STATUS 的提交），已推送，包含 main 的全部提交，**还没有合并回 main**。
  - **10 月 16 日：**
    - 第一批归档 807 个无人引用的文件，提交 `52eb7944`；
    - D4 的文档部分，提交 `efc6928e`；
    - 整理前后行为对照，见 [CLEANUP_EQUIVALENCE_20261016.md](docs/reports/CLEANUP_EQUIVALENCE_20261016.md)；
    - 两次归档事故的处理和 17 条兼容链接，见 [CLEANUP_ARCHIVE_AUDIT_20261016.md](docs/reports/CLEANUP_ARCHIVE_AUDIT_20261016.md)；
    - 集群数据归档、压缩和清单。
  - **10 月 26 日冻结期代码搬动**，见 [FREEZE_CODE_MOVES_20261026.md](docs/reports/FREEZE_CODE_MOVES_20261026.md)：
    - 先合并 main 的第十二轮结束版本 `00363333`；
    - 第一批：统一配置，`configs/` 和 `vagen/config.py`；
    - 第二批：移走 SenseNova-U1 适配代码；
    - 第三批：测试重组到 `tests/<领域>/`。
    - **每批之后两套测试和金标准都和 main 等价。** 最后一轮的套件 A 共 479 条，478 通过、1 跳过；套件 B 73 条全部通过；金标准 17 个文件里 14 个逐字节相同，3 个只差耗时。
  - 单卡冒烟：[docs/reports/SMOKE_DOWNSTREAM_20261016.md](docs/reports/SMOKE_DOWNSTREAM_20261016.md)。EASI 3B/7B 通过；StarVLA 3B/7B 单卡通过，要绕开 DeepSpeed；RLinf 没有通过，原因见报告。
- **等你决定：**
  - Cambrian-S、Qwen3-VL 的适配代码留不留。D4 选的是 a，但模型线从 10 月 24 日起在用它们。
  - 什么时候把整理分支合并回 main，以及什么时候更新共享检出。
- **还没做：**
  - `data_gen/interiorgs_v10` 的训练和评测代码拆到 `training/`、`evaluation/`；
  - D3 旧评测栈、旧 EASI 分片脚本、旧 SFT 生成器、`sapave` 归档，同时撤掉兼容链接；
  - 数据和权重从原目录真正搬到固定位置；
  - 重装 vagen-lite 环境后删掉 SenseNova-U1 的插件空壳；
  - git 历史重写（D1，最后做）。
- **规则：** 以后每次移动集群数据之前，先跑 `scripts/cluster/audit_archived_paths.sh`。

## 2. 正在运行（2026-10-26 12:40 UTC 前后）

| 任务 | 从哪里运行代码 | 和整理的关系 |
| --- | --- | --- |
| 模型线的训练链（Qwen3-VL-8B、Cambrian-S-7B） | 副本 `~/VAGEN-Lite-models`（分支 `models`） | 不受整理分支影响；合并回来时 `job_sco.py` 有 2 处冲突 |
| 密集奖励第二批的训练链和距离表 | 副本 `~/VAGEN-Lite-r12dense`（分支 `dense-batch2`） | 不受影响；整理不能动这个副本，也不能动它读取的主检出路径（见 `docs/TRAINING_LINE_DEPENDENCIES.md`） |
| 第十二轮的守护进程和看板 `batch_board` | 共享检出 `~/VAGEN-Lite` | 更新共享检出时要先停看板循环 |

## 3. 各工作线

| 工作线 | 状态文档 | 一句话状态 |
| --- | --- | --- |
| InteriorGS v10 任务生成与训练（轨迹 → SFT → RL → 评测） | [data_gen/interiorgs_v10/STATUS.md](data_gen/interiorgs_v10/STATUS.md) | 第十二轮批次已结束，冻结期；密集奖励第二批从单独的副本跑 |
| 模型线（Qwen3-VL-8B、Cambrian-S-7B 作骨干） | `models` 分支的 `docs/models/STATUS.md` | 正式 SFT 和对数概率诊断在跑 |
| 磁盘线 | `disk` 分支的 `docs/disk/` | 10-26 定了保留规则；`archive/cluster/` 六个子目录已移到 AOSS |
| InteriorGS 空间问答（自有 QA，D11）：① 仿 8 个 benchmark 的 51 个子任务；② 训练用简单 Yes/No QA；③ 测试用 RPA（Read / Perception / Action） | [docs/interiorgs_qa/PROGRESS.md](docs/interiorgs_qa/PROGRESS.md) | 见该文档；`data_gen/active_spatial_qa` 的 Yes/No 转换器是 RPA 的 Read 测试 |
| R1（45 场景 210 题 Projective 修复战役） | [archive/history/diagnosis/active_spatial_takeover_20261002/REPORT.md](archive/history/diagnosis/active_spatial_takeover_20261002/REPORT.md) | 已被 v10 取代；代码与产物已归档 |
| 下游 StarVLA / RLinf / RoboCasa365 | [docs/active_spatial_pipeline_robot.md](docs/active_spatial_pipeline_robot.md)；`downstream` 分支 | 单卡冒烟见第 1 节 |

## 4. 固定数据与权重

固定位置在集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/` 下，每一项都有 `manifest.json`，汇总在 `configs/{datasets,models,results}.yaml`。目前是指向原目录的符号链接，原目录不动。

| 内容 | 固定位置 | 原目录 |
| --- | --- | --- |
| 第十轮任务数据集（厚墙） | `data/releases/interiorgs_v10_tasks_r10` | `exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015/` |
| 第六轮 b 任务数据集（第一次 SFT / RL 所用） | `data/releases/interiorgs_v10_tasks_r6b` | `exps/vagen_active_spatial/interiorgs_v10_gen_r6b_20261014/` |
| 278 题评测集 | `data/releases/interiorgs_v10_eval_c_278` | `exps/vagen_active_spatial/interiorgs_v10_r7_20261014/eval_c/` |
| 第五轮 3B SFT 模型 | `models/interiorgs_v10_sft_r5_qwen25vl3b` | `exps/vagen_active_spatial/interiorgs_v10_r5_20261014/trial/` |
| 历史权重（D10） | `models/historical/`，共 16 个 | — |
| 基座权重 | `models/base/Qwen2.5-VL-{3B,7B}-Instruct`，清单在旁边的 `.manifest.json` | `~/.cache/huggingface/hub/` |
| 第八轮全部模型评测、第九轮诊断评测 | `results/interiorgs_v10_r8_eval2`、`results/interiorgs_v10_r9_eval_diag` | `exps/vagen_active_spatial/interiorgs_v10_r{8,9}_…/` |
| 场景数据 | — | `~/InteriorGS_meta`（元数据）、`~/InteriorGS_ply`（3DGS） |

第六轮 b 的 SFT 7B 和 7 个 RL 检查点已在 10 月 25 日的主线清理中删除，删前没有归档；撤下记录在 `configs/removed_entries.yaml`，它们的评测结果仍在 `results/interiorgs_v10_r8_eval2`。第十轮到第十二轮训练出的模型还没进固定位置，等那条线收尾后补登记。
