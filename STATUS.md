# VAGEN-Lite 状态（唯一的状态文档）

最后更新：2026-10-16 晚（整理任务第二阶段进行中）。各条工作线的细节仍在各自文档里，这里只汇总"现在在哪一步、谁在跑、下一步等什么"。

## 1. 整理任务（把仓库整理成可发布版本，把 7 步流程穿起来）

- **第一阶段（盘点与方案）** 已完成：[docs/reports/PHASE1_INVENTORY_20261015.md](docs/reports/PHASE1_INVENTORY_20261015.md)。你的决定记录在 [docs/reports/DECISIONS_20261016.md](docs/reports/DECISIONS_20261016.md)。
- **第二阶段（整理）** 2026-10-16 开始，只做不与训练链冲突的部分。
  - 标签 `pre-cleanup-20261016`（main `3c1d59ed`）已推送。整理在分支 `cleanup/20261016` 上做，用独立检出 `~/VAGEN-Lite-cleanup-wt`；共享目录 `~/VAGEN-Lite` 留在 main，不动。分支已合并 main 到第十一轮的 `8134f102`。
  - 第一批归档 `52eb7944`：807 个无人引用的文件移入 `archive/`，索引 [archive/README.md](archive/README.md)。
  - 第二批 `efc6928e`（D4 选 a 的文档部分）：5 份旁支模型文档和 Qwen3-VL 环境检查移入 `archive/side_models/`。`vagen/models/` 要连同核心训练循环里导入它的分支一起移走，放到代码搬动批次。
  - **两次事故，已处理**：集群归档时移走了两个仍被在用代码读取的目录。第十一轮训练脚本写死的 gsplat 编译缓存，已由第十一轮会话移到仓库外修好；自有 QA 流水线读的第六轮生成目录，已原样还原。全面核查后补了 17 条兼容链接。见 [docs/reports/CLEANUP_ARCHIVE_AUDIT_20261016.md](docs/reports/CLEANUP_ARCHIVE_AUDIT_20261016.md)。
  - **整理前后行为一致**：两套测试无新增失败，固定输出 17 个文件里 14 个逐字节相同、3 个只差耗时字段。见 [docs/reports/CLEANUP_EQUIVALENCE_20261016.md](docs/reports/CLEANUP_EQUIVALENCE_20261016.md)。
  - 集群整理（都有清单，可追溯）：
    - root 属主文件已全部收回；
    - 实验残留、旧结果、日志移入 `archive/cluster/`，清单 `MANIFEST_moves_20261016.tsv`；
    - 其中小文件多的类目按条目压成 `.tar.gz`，122 个条目从 41.6 GB 压到 19.8 GB，清单 `MANIFEST_compressed_20261016.tsv`；
    - D6 / D7a / D7c 的权重导出与分片删除已完成，记录 `exps/_cleanup_ckpt_20261016/MANIFEST_checkpoints_20261016.tsv`；
    - 固定位置 `data/releases/`、`models/`、`results/` 已建好（目前是指向原目录的符号链接），32 份 `manifest.json` 已写出（`scripts/cluster/make_all_manifests.sh`）。
  - dense-score pilot r3 结论报告：[docs/reports/DENSE_SCORE_PILOT_R3_CONCLUSION_20261016.md](docs/reports/DENSE_SCORE_PILOT_R3_CONCLUSION_20261016.md)（D7b：只报告、不删）。
  - 单卡冒烟：[docs/reports/SMOKE_DOWNSTREAM_20261016.md](docs/reports/SMOKE_DOWNSTREAM_20261016.md)。EASI 3B/7B 通过；StarVLA 3B/7B 单卡通过（要绕开 DeepSpeed）；RLinf 已解决 Python 环境和 git 两个问题，卡在计算节点的渲染库，第五次任务用项目内的渲染库覆盖层重试。
- **已决定（2026-10-16 晚）**：D4 选 a；主线用 Qwen2.5-VL-7B，3B 只用于单卡试跑；`archive/cluster/_pending_delete/` 的 230 GB 冒烟残留已删除，逐文件记录 `archive/cluster/DELETED_pending_delete_20261016.tsv`。
- **还没做**：代码搬动（data_gen / training / evaluation / downstream 拆分，含 `vagen/models/`，等第十一轮结束），每批之后撤掉对应的兼容链接；统一配置；数据与权重从原目录真正搬到固定位置；git 历史重写（D1，最后做）。以后每次移动集群数据前，先跑 `scripts/cluster/audit_archived_paths.sh`。

## 2. 正在运行（2026-10-16 晚）

| 任务 | 位置 | 状态 |
| --- | --- | --- |
| 第十一轮（另一个会话负责） | `exps/vagen_active_spatial/interiorgs_v10_r11_20261016/` | 登录节点上目前没有训练链进程；整理不碰这个目录和第十轮的目录 |
| 整理任务的 2 个单卡 sco 任务：渲染诊断 `pt-a3vjvtyk`、RLinf 第五次 `pt-65515vh8` | `exps/_cleanup_smoke_20261016/{gl_diag,rlinf5}/` | 已提交；zoetrope 池整体停在"启动中"，所有人的任务都没开始 |

## 3. 各工作线

| 工作线 | 状态文档 | 一句话状态 |
| --- | --- | --- |
| InteriorGS v10 任务生成与训练（轨迹 → SFT → RL → 评测） | [data_gen/interiorgs_v10/STATUS.md](data_gen/interiorgs_v10/STATUS.md) | 第十一轮：第十轮最终模型定为 SFT 正式题组第 5 个 epoch；验证集扩到约 400 题；多个 RL 同时跑 |
| InteriorGS 空间问答（自有 QA，D11）：① 仿 8 个 benchmark 的 51 个子任务；② 训练用简单 Yes/No QA；③ 测试用 RPA（Read / Perception / Action） | [docs/interiorgs_qa/PROGRESS.md](docs/interiorgs_qa/PROGRESS.md) | 230 道样题在标注；全量生成等标注结果；`data_gen/active_spatial_qa` 的 Yes/No 转换器是 RPA 的 Read 测试 |
| R1（45 场景 210 题 Projective 修复战役） | [archive/history/diagnosis/active_spatial_takeover_20261002/REPORT.md](archive/history/diagnosis/active_spatial_takeover_20261002/REPORT.md) | 已被 v10 取代；代码与产物已归档 |
| 下游 StarVLA / RLinf / RoboCasa365 | [docs/active_spatial_pipeline_robot.md](docs/active_spatial_pipeline_robot.md) | 单卡冒烟见第 1 节 |

## 4. 固定数据与权重

固定位置在集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/` 下，每一项都有 `manifest.json`，记录来源、大小、代码提交和关键文件的 SHA-256。目前是指向原目录的符号链接，原目录不动；真正搬动等训练链结束。

| 内容 | 固定位置 | 原目录 |
| --- | --- | --- |
| 第十轮任务数据集（厚墙） | `data/releases/interiorgs_v10_tasks_r10` | `exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015/` |
| 第六轮 b 任务数据集（第一次 SFT / RL 所用） | `data/releases/interiorgs_v10_tasks_r6b` | `exps/vagen_active_spatial/interiorgs_v10_gen_r6b_20261014/` |
| 278 题评测集 | `data/releases/interiorgs_v10_eval_c_278` | `exps/vagen_active_spatial/interiorgs_v10_r7_20261014/eval_c/` |
| SFT 7B 与 RL 7 个检查点（只剩 HF 权重，第 250 步保留续训状态） | `models/interiorgs_v10_sft_r6b_qwen25vl7b`、`models/interiorgs_v10_rl_r6b_qwen25vl7b_step{50,…,250}` | `exps/vagen_active_spatial/interiorgs_v10_train_20261014/` |
| 第五轮 3B SFT 模型 | `models/interiorgs_v10_sft_r5_qwen25vl3b` | `exps/vagen_active_spatial/interiorgs_v10_r5_20261014/trial/` |
| 历史权重（D10） | `models/historical/` | — |
| 基座权重 | `models/base/Qwen2.5-VL-{3B,7B}-Instruct`（清单在旁边的 `.manifest.json`） | `~/.cache/huggingface/hub/` |
| 第八轮全部模型评测、第九轮诊断评测 | `results/interiorgs_v10_r8_eval2`、`results/interiorgs_v10_r9_eval_diag` | `exps/vagen_active_spatial/interiorgs_v10_r{8,9}_…/` |
| 场景数据 | — | `~/InteriorGS_meta`（元数据）、`~/InteriorGS_ply`（3DGS） |
| RoboCasa 源码（从登录节点 `/var/tmp` 复制，供 sco 节点上的 RLinf 用） | — | `third_party/robocasa` |

第十轮、第十一轮训练出的模型还没进固定位置，等那条线收尾后补登记。
