# VAGEN-Lite 状态（唯一的状态文档）

最后更新：2026-10-15（整理任务第一阶段结束）。各条工作线的细节状态仍在各自文档里，这里只汇总"现在在哪一步、谁在跑、下一步等什么"。

## 1. 整理任务（把仓库整理成可发布版本，把 7 步流程穿起来）

- **阶段**：第一阶段（盘点与方案）已完成，**停下等你确认**。报告：[docs/reports/PHASE1_INVENTORY_20261015.md](docs/reports/PHASE1_INVENTORY_20261015.md)，附录在 `docs/reports/phase1/`。
- **没有动任何东西**：未移动、归档、删除文件，未打标签，未改代码。
- **等你决定的 15 项**见报告第 8 节；最影响后续的是 D1（git 历史重写）、D3（评测栈二选一）、D6 / D7（约 4.5 TB 训练分片删不删）、D11（哪一套算"自有 QA"）。
- **第二阶段（整理）** 待确认后开始；代码搬动必须等第十轮训练链结束。

## 2. 正在运行（2026-10-15 17:51 UTC 查询）

| 任务 | 位置 | 状态 |
| --- | --- | --- |
| 第十轮训练链 `train_chain run`（登录节点进程 983896） | `exps/vagen_active_spatial/interiorgs_v10_train_r10_20261015/` | SFT formal 已到 770/770 步；SFT formal_reserve 8 卡任务 `pt-k9lxp6la` 在跑；评测任务 `pt-ggblr0u2`、`pt-ivl9ager` 在跑。后续：按验证集挑检查点 → RL（S0，一道题一个序列）→ 评测。进度看该目录的 `PROGRESS.md` |
| 第十轮带历史的两种 RL 做法试跑 | `exps/vagen_active_spatial/interiorgs_v10_r10_20261015/pilot_{concat,no_concat}` | 产物在，结论待 `REPORT_R10_20261015.md` 填写 |

## 3. 各工作线

| 工作线 | 状态文档 | 一句话状态 |
| --- | --- | --- |
| InteriorGS v10 任务生成与训练（轨迹 → SFT → RL → 评测） | [data_gen/interiorgs_v10/STATUS.md](data_gen/interiorgs_v10/STATUS.md) | 第十轮（厚墙、新默认规则）在跑；第八轮结论：RL 250 次未超过 SFT 第 3 epoch（21.6% 对 23.4%，278 题） |
| InteriorGS 空间问答第二版（仿 8 个 benchmark 出题） | [docs/interiorgs_qa/PROGRESS.md](docs/interiorgs_qa/PROGRESS.md) | 230 道样题在标注；全量生成（B4）等标注结果 |
| R1 配对 Yes/No QA 生产（`data_gen/active_spatial_qa/` 未跟踪脚本） | 无（另一会话） | 生产 v1–v3 产物 31 GiB 在 `artifacts/`；整理方案里待定 D6 / D11 |
| R1（45 场景 210 题 Projective 修复战役） | [docs/diagnosis/active_spatial_takeover_20261002/REPORT.md](docs/diagnosis/active_spatial_takeover_20261002/REPORT.md) | 已被 v10 取代；产物待归档 |
| 下游 StarVLA / RLinf / RoboCasa365 | [docs/active_spatial_pipeline_robot.md](docs/active_spatial_pipeline_robot.md) | 接口与补丁就绪；StarVLA 只跑过与本项目权重无关的 NavigateKitchen；RLinf 从未训练 |

## 4. 固定数据与权重（整理前的位置）

| 内容 | 路径（集群 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/`） |
| --- | --- |
| 当前任务数据集（第十轮，厚墙） | `exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015/`（`output/tasks.jsonl`、`split_manifest.json`） |
| 上一版任务数据集（第一次 SFT / RL 所用） | `exps/vagen_active_spatial/interiorgs_v10_gen_r6b_20261014/` |
| 评测集（278 题 test） | `exps/vagen_active_spatial/interiorgs_v10_r7_20261014/eval_c/tasks.jsonl` |
| SFT 7B（3 个 epoch）与 RL 7 个检查点 | `exps/vagen_active_spatial/interiorgs_v10_train_20261014/{lf/model,rl_run/checkpoints}` |
| 全部模型评测结果与轨迹浏览页 | `exps/vagen_active_spatial/interiorgs_v10_r8_20261014/eval2/`、`interiorgs_v10_r9_20261015/` |
| 场景数据 | `~/InteriorGS_meta`（元数据）、`~/InteriorGS_ply`（3DGS） |
| 基座权重 | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-{7B,3B}-Instruct` |

整理后这些会搬到 `data/releases/`、`models/`、`results/` 并带清单（方案见报告第 5 节，待确认）。
