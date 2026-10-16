# VAGEN-Lite 状态（唯一的状态文档）

最后更新：2026-10-16（整理任务第二阶段进行中）。各条工作线的细节状态仍在各自文档里，这里只汇总"现在在哪一步、谁在跑、下一步等什么"。

## 1. 整理任务（把仓库整理成可发布版本，把 7 步流程穿起来）

- **第一阶段（盘点与方案）** 已完成：[docs/reports/PHASE1_INVENTORY_20261015.md](docs/reports/PHASE1_INVENTORY_20261015.md)。你的决定记录在 [docs/reports/DECISIONS_20261016.md](docs/reports/DECISIONS_20261016.md)。
- **第二阶段（整理）** 2026-10-16 开始，只做不与第十轮训练链冲突的部分：
  - 已打标签 `pre-cleanup-20261016`（对应 main `3c1d59ed`）并推送；整理在分支 `cleanup/20261016` 上做，用独立 worktree `~/VAGEN-Lite-cleanup-wt`，共享目录 `~/VAGEN-Lite` 仍在 main、不动。
  - 整理前基线：全套 pytest **395 通过、1 跳过**（15 分钟）；CPU 回归集 **77 通过**；金标准输出（2 个场景出题、回放、split 与 SFT 帧表、评测表）在集群 `exps/_cleanup_golden_20261016/`（`SHA256SUMS.txt` 17 个文件）。
  - 第一批归档已提交（`52eb7944`）：807 个文件移入 `archive/`，索引 [archive/README.md](archive/README.md)。
  - 集群：root 属主文件已全部收回（129,458 个 → 0）；实验残留、旧结果、日志正在移入 `archive/cluster/`（清单 `archive/cluster/MANIFEST_moves_20261016.tsv`）；D6 / D7a / D7c 的权重导出与分片删除在后台执行（记录 `exps/_cleanup_ckpt_20261016/MANIFEST_checkpoints_20261016.tsv`）。
  - dense-score pilot r3 的结论报告：[docs/reports/DENSE_SCORE_PILOT_R3_CONCLUSION_20261016.md](docs/reports/DENSE_SCORE_PILOT_R3_CONCLUSION_20261016.md)（D7b：只报告、不删）。
  - 单卡冒烟（决定 D4 和 3B/7B 之前要做的）：EASI `pt-f9j3tmf1`、StarVLA 3B `pt-c8ukfb85`、StarVLA 7B `pt-13kh2el1`、RLinf `pt-ignjv4xu`，产物在 `exps/_cleanup_smoke_20261016/<名称>/`。
- **还没做**：代码搬动（data_gen / training / evaluation / downstream 的拆分，等第十轮训练链结束）、统一配置、数据与权重清单、大目录压缩、git 历史重写（D1，最后做）。

## 2. 正在运行（2026-10-16 07:00 UTC 前后）

| 任务 | 位置 | 状态 |
| --- | --- | --- |
| 第十轮训练链 `train_chain run`（登录节点进程 983896） | `exps/vagen_active_spatial/interiorgs_v10_train_r10_20261015/` | SFT 两组之后进入评测 / 挑检查点 / RL；进度看该目录 `PROGRESS.md`。整理不碰它和 `interiorgs_v10_r10_20261015/` |
| 整理任务的 4 个单卡冒烟 sco 任务 | `exps/_cleanup_smoke_20261016/` | 见上节 |
| 整理任务的后台脚本（登录节点） | `/tmp/psv/cleanup_inv/archive_moves.log`、`exps/_cleanup_ckpt_20261016/d6_d7.log` | 数据搬移、权重导出与删除 |

## 3. 各工作线

| 工作线 | 状态文档 | 一句话状态 |
| --- | --- | --- |
| InteriorGS v10 任务生成与训练（轨迹 → SFT → RL → 评测） | [data_gen/interiorgs_v10/STATUS.md](data_gen/interiorgs_v10/STATUS.md) | 第十轮（厚墙、新默认规则）在跑；第八轮结论：RL 250 次未超过 SFT 第 3 epoch（21.6% 对 23.4%，278 题） |
| InteriorGS 空间问答（自有 QA，D11）：① 仿 8 个 benchmark 的 51 个子任务；② 训练用简单 Yes/No QA；③ 测试用 RPA（Read / Perception / Action） | [docs/interiorgs_qa/PROGRESS.md](docs/interiorgs_qa/PROGRESS.md) | 230 道样题在标注；全量生成（B4）等标注结果；`data_gen/active_spatial_qa` 的 Yes/No 转换器是 RPA 的 Read 测试 |
| R1（45 场景 210 题 Projective 修复战役） | [archive/history/diagnosis/active_spatial_takeover_20261002/REPORT.md](archive/history/diagnosis/active_spatial_takeover_20261002/REPORT.md) | 已被 v10 取代；代码与产物归档中 |
| 下游 StarVLA / RLinf / RoboCasa365 | [docs/active_spatial_pipeline_robot.md](docs/active_spatial_pipeline_robot.md) | 单卡冒烟进行中（见第 1 节） |

## 4. 固定数据与权重（整理前的位置）

| 内容 | 路径（集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/`） |
| --- | --- |
| 当前任务数据集（第十轮，厚墙） | `exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015/`（`output/tasks.jsonl`、`split_manifest.json`） |
| 上一版任务数据集（第一次 SFT / RL 所用） | `exps/vagen_active_spatial/interiorgs_v10_gen_r6b_20261014/` |
| 评测集（278 题 test） | `exps/vagen_active_spatial/interiorgs_v10_r7_20261014/eval_c/tasks.jsonl` |
| SFT 7B（3 个 epoch）与 RL 7 个检查点（整理后只剩 HF 权重，第 250 步保留续训状态） | `exps/vagen_active_spatial/interiorgs_v10_train_20261014/{lf/model,rl_run/checkpoints}` |
| 第五轮 3B SFT 模型（第十轮试跑在用） | `exps/vagen_active_spatial/interiorgs_v10_r5_20261014/trial/sft_trial_run/model` |
| 全部模型评测结果与轨迹浏览页 | `exps/vagen_active_spatial/interiorgs_v10_r8_20261014/eval2/`、`interiorgs_v10_r9_20261015/` |
| 场景数据 | `~/InteriorGS_meta`（元数据）、`~/InteriorGS_ply`（3DGS） |
| 基座权重 | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-{7B,3B}-Instruct` |
| RoboCasa 源码（从登录节点 `/var/tmp` 复制而来，供 RLinf 环境在 sco 节点上使用） | `third_party/robocasa` |

整理后这些会搬到 `data/releases/`、`models/`、`results/` 并带清单（决定 D13），搬动等第十轮训练链结束。
