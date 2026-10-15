# 第一阶段报告：VAGEN-Lite 盘点与整理方案（2026-10-15）

本报告只盘点、不动手。没有移动、归档、删除任何文件，也没有提交代码；只新增了本报告、附录表和仓库根目录的 `STATUS.md`。

## 结论（先看这里）

1. **仓库里真正在用的核心代码只占小头。** 代码类文件（.py / .sh / .yaml / .json，不含图片）共 1,096 个。其中 122 个是环境、训练框架、下游接口（建议原位保留），约 100 个是当前 v10 数据与训练链、下游入口、评测入口（建议整理进 data_gen / training / evaluation / downstream），39 个是另一会话的 QA 出题代码，54 个是测试。其余约 780 个没有被当前流程引用：明确可归档的约 490 个（历史实验配置 246、R1 修复战役脚本 135、诊断探针 57、旧生成器与旧 SFT 生成器 40、分析脚本 4 等），需要你拍板的 289 个（上游 VAGEN 环境与示例 172、旧评测栈 46、旁支模型 11、文本动作路线 8、R1 配对 QA 52）。
2. **git 仓库被数据撑大。** 42,084 个被跟踪文件里有 39,720 张 jpg（旧 SFT 轨迹图片，403 MB）和 553 个旧生成器产物（465 MB）；代码本身不到 20 MB。clone 一次要拉 236 MB 的 pack。要变成"可发布"的仓库，这些必须出 git（是否重写历史需要你决定，见决定 D1）。
3. **集群上 6.5 TiB 里约 5.5 TiB 是可以压缩或删除的训练分片。** `exps/` 占 6.2 TiB：R1 训练分片 1.2 TiB、dense-score pilot 1.7 TiB、v10 第一次 RL 的 verl 分片 1.35 TiB、第五轮单卡试跑 0.26 TiB，加上 root 属主的 smoke 残留 0.22 TiB。真正要长期保留的是：当前数据集（r10，6.7 GB）、上一版数据集（r6b，3.5 GB）、SFT / RL 的 HuggingFace 权重（约 230 GB）、评测集与评测结果（约 3 GB）。
4. **7 步流程里，第 1–4 步和第 6 步的"ID 轨迹"有可运行的代码和真实结果；第 5 步（实验矩阵）完全没有；第 6 步的 EASI 自 8 月后没再跑、OOD 对 v10 数据没有定义；第 4 步的"QA 上 RL"只有一个单步选择题环境雏形；第 7 步 StarVLA 只跑过一次与本项目权重无关的 NavigateKitchen 训练，RLinf 从未跑过训练。** 缺口详见第 4 节。
5. **有三条线正在进行，整理时不能碰：** 第十轮训练链（登录节点进程 + 1 个 8 卡 SFT 任务在跑、另一组 SFT 已完成 + 2 个单卡评测任务）、InteriorGS QA 第二版（等你标注）、active_spatial_qa 生产脚本（27 个未跟踪文件）。代码搬动必须等第十轮训练链结束。
6. 建议的目录结构基本按你给的方案，两处改动：保留 Python 包名 `vagen` 和 `data_gen`（理由见第 5 节），评测代码统一到 v10 的 `evaluation.py`，旧评测栈归档（决定 D3）。
7. 工作量：第二阶段约 5–6 个工作日，第三阶段约 8–10 个工作日（含 GPU 冒烟和下游环境核验），第四阶段约 3 个工作日。详见第 7 节。
8. **需要你决定的事项有 15 项**（第 8 节），其中最影响后续的是 D1（git 历史）、D3（评测栈二选一）、D6/D7（4 TB 训练分片删不删）、D11（哪一套是"自有 QA"）。

术语：本报告里"流程步骤"指任务说明里的 7 步（1 环境与评分、2 数据生成、3 SFT、4 RL、5 实验矩阵、6 标准化测评、7 下游）。"R1"指 2026 年 8–10 月在 45 个场景、210 道 Projective 题上做的数据修复与训练战役，已被 10 月 11 日起的 "interiorgs_v10"（1,000 个场景、6 个任务族的新生成器）取代。"v10"即这个新生成器及其训练链。"verl 分片"指 PPO 训练时按 GPU 切开保存的 actor / critic / optimizer 状态，只用来续训；"HF 权重"指可直接加载推理的 HuggingFace 格式权重。

## 0. 盘点方法与边界

- 代码：对 GitHub `main`（提交 `f53247c7`，2026-10-15）的全部 42,084 个被跟踪文件和 27 个未跟踪文件做了统计；对 1,096 个代码类文件逐个计算了最后提交日期、提交次数、被多少个代码文件和文档引用（按路径、模块名、`import`、`python -m`、`python path.py` 五种写法匹配，语料是仓库里全部 1,282 个文本文件）。逐文件结果在附录 [phase1/code_files.tsv](phase1/code_files.tsv)。
- 集群：对 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite` 做了 `du`，对 `exps/` 的 308 个一级条目逐个记录了大小、修改日期、属主、检查点数量、是否含 HF 权重。结果在附录 [phase1/exps_inventory.tsv](phase1/exps_inventory.tsv)。
- 文档：103 个被跟踪的 md 文件逐个看了开头、日期和被引用情况；另查了 docx / pdf（只有 19 个 pdf 图，没有 docx）。
- 引用统计的局限：只能发现文本里写出来的引用；通过配置动态拼出的路径、注释里的提法会被算成"文档引用"或漏掉。因此"无人引用"只是归档的必要条件，第二阶段搬动前还会用 `pytest` 和冒烟测试复核。
- 时间：集群是 UTC，本报告写的日期以提交记录为准。

### 不碰的东西（整理期间）

| 项目 | 现状 | 依据 |
| --- | --- | --- |
| 第十轮训练链 `exps/vagen_active_spatial/interiorgs_v10_train_r10_20261015/`（268 GB）和 `interiorgs_v10_r10_20261015/`（407 GB，两个单卡 RL 试跑） | 登录节点进程 `train_chain run`（pid 983896）在跑；sco 任务 `pt-k9lxp6la`（8 卡 SFT）、`pt-ggblr0u2`、`pt-ivl9ager`（单卡评测）RUNNING（2026-10-15 17:51 UTC 查询） | `data_gen/interiorgs_v10/STATUS.md` 第 3 节；第七轮决定"训练期间原目录只加文档、不改代码" |
| InteriorGS QA 第二版：`data_gen/interiorgs_qa/`、`docs/interiorgs_qa/`、`exps/interiorgs_qa_*`（15 个目录，6.5 GB）、`data_gen/sapave`（它的 sco 打包清单里引用） | 另一会话负责，等你交回 230 道样题标注后做 B1–B5 | `docs/interiorgs_qa/PROGRESS.md` |
| `data_gen/active_spatial_qa/` 的 27 个未跟踪文件（`production_v1–v3.py`、`qa_apply_*`、`sco_production_*`，2026-10-11 至 10-14 修改）和 `data_gen/active_spatial_qa/artifacts/`（31.4 GiB） | QA 生产脚本，留给负责 QA 的会话 | v10 STATUS 第七轮第 4 条 |

## 1. 代码盘点

### 1.1 总体数字

| 项目 | 数量 | 说明 |
| --- | --- | --- |
| 被跟踪文件 | 42,084 | 其中 39,720 jpg、416 png、132 mp4 都在 `data_gen/` 下 |
| 代码类文件（py/sh/yaml/json，不含 `data_gen/*/output*`） | 1,096 | 本节分类的对象 |
| 未跟踪文件 | 27 | 全在 `data_gen/active_spatial_qa/`（另一会话） |
| 提交数 | 385 | 2025-12 至 2026-10；2026 年 4 月前的 180 次是上游 VAGEN 作者的提交，6 月起的 205 次是本项目 |
| git pack 大小 | 236 MB（集群）；本机 clone `.git` 287 MB | 图片和产物占了绝大部分 |
| 子模块 | 1：`verl` → `yyyybq/verl` 分支 `vagen-lite`，提交 `a0ae85c7`（2026-10-14），工作区无改动 | 复现需同时记录子模块提交 |

### 1.2 按目录的盘点

"步骤"列是流程的 7 步；"引用"列写的是被代码引用的文件数 / 组内文件数（引用统计见 0 节）；"去向"是第 5 节方案里的位置。

| 目录 / 文件 | 文件数 | 步骤 | 最后提交 | 引用 | 分类 | 建议去向 |
| --- | ---: | --- | --- | --- | --- | --- |
| `vagen/envs/active_spatial/` | 37 | 1 环境与评分 | 2026-10-15 | 37/37 | 核心 | 保留 |
| `vagen/main_ppo.py`、`ray_trainer.py`、`agent_loop/`、`custom_advantage/`、`custom_filter/`、`custom_metric/`、`gym_agent_dataset.py`、`utils/`、`configs/`、`envs/{registry,gym_*}.py` | 34 | 4 RL 框架 | 2026-10-14 | 全部 | 核心 | 保留 |
| `vagen/interiorgs_v10_ppo.py` | 1 | 4 RL | 2026-10-15 | 被 `sco_interiorgs_v10_train.sh` 调用 | 核心 | 保留 |
| `vagen/envs/spatial_mcq/` | 3 | 4 RL（QA 上） | 2026-08-31 | 被 A1 实验配置引用 | 核心（雏形） | 保留，第三阶段改造 |
| `vagen/envs/robocasa/`、`envs_remote/`、`evaluate/`、`server/`、`rollout/` | 46 | 7 下游 / 框架 | 2026-10-06 | 45/46 | 核心 | 保留 |
| `vagen/envs/{sokoban,frozenlake,navigation,primitive_skill,spatial_gym}/`、`vagen/inference/` | 116 | 无（上游 VAGEN） | 2026-06-14 前 | 组内互引，流程不引用 | 第三方/上游 | 待定 D2（建议 `archive/upstream/`） |
| `vagen/models/`（Cambrian-S、SenseNova-U1 适配） | 11 | 无（旁支模型） | 2026-08-31 | 被旧 launcher 引用 | 旁支 | 待定 D4 |
| `vagen/r1_clean_projective_ppo.py`、`vagen/dense_score_reward_only_pilot.py` | 2 | 无 | 2026-10-13 | 被 R1 / pilot 入口引用 | 一次性 | `archive/r1/`、`archive/dense_score_pilot/` |
| `data_gen/interiorgs_v10/`（代码 45 个 + 配置 13 + 族定义 12 + 测试 42） | 112 | 1、2、3、4、6 | 2026-10-15 | 全部被训练链引用 | 核心 | 拆到 `data_gen/interiorgs_v10/`（出题）、`training/`（train_chain、training_prep、job_sco）、`evaluation/trajectory/`（evaluation、eval_browser、fail_analysis、rl_checks） |
| `data_gen/interiorgs_v10/cluster_runs/`（328 个文件：12 份轮次报告 + 审计 json/图） | 328 | 文档 | 2026-10-15 | — | 报告 | `docs/reports/history/interiorgs_v10/` |
| `data_gen/interiorgs_qa/` | 51 | 2 QA 数据 | 2026-10-15 | 另一会话 | 核心 | 保留原位，完工后再移 |
| `data_gen/active_spatial_qa/`（26 跟踪 + 27 未跟踪） | 53 | 2 QA（R1 配对） | 2026-10-10 | 只被自身和 CPU 测试引用 | R1 线 | 待定 D6 / D11 |
| `data_gen/active_spatial_pipeline/`（21 个代码 + 532 个产物） | 553 | 无（v1 旧生成器） | 2026-10-06 | 被旧评测栈和 2 个保留脚本引用（`splits.read_rows`） | 被取代 | `archive/legacy_pipeline_v1/`，`read_rows` 复制到保留脚本 |
| `data_gen/active_spatial_sft/`（19 个代码 + 39,874 张图片） | 39,893 | 无（旧 / R1 SFT 生成器） | 2026-10-10 | `convert_to_qwen25vl_sft.py` 被 `prepare_active_spatial_sft.py` 和 v10 `evaluation.py` 导入 | 被取代 | 代码 `archive/legacy_sft/`；被导入的函数并入 `training/sft/`；图片出 git（D1） |
| `data_gen/robocasa_sft/` | 5 | 7（文本 12 维动作路线） | 2026-10-06 | 被 `examples/train/robocasa/run_sft.sh` 引用 | 备选路线 | 待定 D5 |
| `data_gen/sapave/` | 3 | 6（旧评测分类表） | 2026-10-06 | 被旧评测套件引用 | 旧评测栈 | 待定 D3 |
| `scripts/r1_*.py`、`r1_*.json` | 135 | 无（R1 一次性） | 2026-08-31 至 10-13 | 61 个只被同系列或 sco 入口引用，74 个无引用 | 一次性 | `archive/r1/scripts/` |
| `scripts/d0_*.py`、`tools/` | 37 | 无（D0 / U1 诊断） | 2026-08-31 | 14 个无任何引用 | 一次性 | `archive/diagnostics_d0_u1/` |
| `scripts/*dense_score*`、`prepare_active_spatial_dense_score_pilot.py` | 6 | 无（pilot 门禁） | 2026-10-13 | 2 个无引用 | 一次性 | `archive/dense_score_pilot/` |
| `scripts/{active_spatial_eval_sweep,full_eval,eval_parallel,eval_report,qa_parallel,aggregate_*_shards,record_one_ckpt_results,select_best_checkpoint,analyze_experiments,filter_*_jsonl,monitor_*}`、`run_viewspatial_sharded`、`run_site_sharded`、`merge_viewspatial_shards`、`build_sapave_splits`、`run_gpt6_id_ood_eval` | 16 | 6（旧评测栈） | 2026-10-06 | 互引为主；`eval_sweep` 被文档引用 5 次 | 旧评测栈 | 待定 D3 |
| `evaluation/`（EvalRunner、随机 / 启发式 agent、API 模型适配） | 19 | 6（旧评测栈） | 2026-10-06 | `metrics.py` 被 v10 `train_chain.py` 引用 1 次 | 旧评测栈 | 待定 D3（`metrics.py` 的用到部分并入 `evaluation/trajectory/`） |
| `scripts/{easi_eval.py,easi_registry.yaml,easi_table.py,launch_parallel_easi.sh}` | 4 | 6 EASI | 2026-08-31 | `easi_eval.py` 被 `interiorgs_qa/qa.py`、`scoring.py` 引用（复用提示词） | 核心 | `evaluation/easi/` |
| `scripts/gen_ood_splits.py` | 1 | 6 OOD | 2026-10-06 | 被旧 R1 脚本引用 | 核心（需改造） | `evaluation/ood/` |
| `scripts/prep_spatial_aux_mcq.py` | 1 | 4 RL（QA 上） | 2026-08-31 | 被 A1 launcher 引用 | 核心（雏形） | `training/rl_qa/` |
| `scripts/{prepare_active_spatial_sft,prepare_active_spatial_rl,verify_active_spatial_dataset,active_spatial_release_contract}.py` | 4 | 2、3、4 | 2026-10-15 | `prepare_sft` 被 11 个代码文件引用；`release_contract` 被 `release.py` 加载 | 核心 | `training/sft/`、`training/rl/`、`data_gen/common/` |
| `scripts/{prepare_starvla_transfer,check_starvla_*,export_starvla_backbone,export_rlinf_starvla,prepare_rlinf_robocasa,eval_starvla_*,smoke_starvla_dataloader,robocasa_eval}.py` | 10 | 7 下游 | 2026-10-10 | 9/10 被入口脚本或测试引用 | 核心 | `downstream/` |
| `scripts/{active_spatial_offline_ppo_replay,r5_replay_forensics,real_batch_preflight,diagnostic_real_batch_gate,r1_rgb_runtime_preflight,r5_production_backward,renderer_microbenchmark,compare_renderer_microbenchmarks,prepare_cambrian_id_shards,planner_audit}.py` | 10 | 无（诊断） | 2026-10-10 | 只被诊断报告或同系列引用 | 一次性 | `archive/diagnostics/` |
| `scripts/{visualize_*,analyze_active_spatial_training,build_baseline_trained_analysis}.py` | 4 | 无（分析画图） | 2026-10-06 | 无引用 | 一次性 | `archive/analysis/` |
| `scripts/check_active_spatial_cpu.sh`、`restore_active_spatial_checkpoints_aoss.sh`、`check_qwen3_vl_env.py` | 3 | 测试 / 集群 | 2026-10-10 | 被文档引用 | 工具 | `tests/`、`scripts/cluster/` |
| `examples/train/active_spatial/experiments/`（v1–v50、c1–c8、u1、d0、a1/a2/b5、clean、dense 等实验配置） | 130 | 无（历史实验） | 2026-10-10 | 97 个无代码引用 | 实验脚本（历史） | `archive/experiments_legacy/experiments/`，建索引 |
| `examples/train/active_spatial/env_config_*.yaml` | 53 | 无（历史环境配置） | 2026-10-13 | 只被历史实验引用；v10 用 `data_gen/interiorgs_v10/configs/rl_env_v10_*.yaml` | 历史 | `archive/experiments_legacy/env_configs/` |
| `examples/train/active_spatial/`其余（`run_experiment.sh` 旧通用 launcher、`launch_*.sh`、`submit_*.sh`、`sco_*_entry.sh`、GPU 占位脚本等） | 63 | 无（历史启动器） | 2026-10-14 | `run_experiment.sh` 被 101 个历史实验引用，v10 训练链不用 | 历史 | `archive/experiments_legacy/launchers/` |
| `examples/train/active_spatial/{sco_interiorgs_v10_train.sh,accelerate_fsdp_qwen25vl.yaml}` | 2 | 3、4 | 2026-10-14 | 被 v10 训练链调用 | 核心 | `training/` |
| `examples/train/active_spatial/{start_gs_render_http_service,start_gs_render_server,start_active_spatial_env_server}.sh`、`sco_run_as_artifact_owner.sh` | 4 | 1 渲染 / 集群 | 2026-09-30 | 被 19 个脚本引用 | 核心 | `scripts/render/`、`scripts/cluster/` |
| `examples/train/robocasa/`、`examples/evaluate/robocasa/` | 22 | 7 下游 | 2026-10-10 | 互引 + 文档 | 核心 | `downstream/`（其中 LLaMA-Factory 文本路线 3 个文件待定 D5） |
| `examples/evaluate/active_spatial/`（旧评测套件 yaml、R1 评测入口） | 8 | 6（旧评测栈） | 2026-10-06 | 4 个无引用 | 旧评测栈 | 待定 D3 |
| `examples/{train,evaluate}/{frozenlake,navigation,primitive_skill,sokoban,spatial_gym}/` | 56 | 无（上游示例） | 2026-04-09 前 | 只被上游 README 引用 | 上游 | 待定 D2 |
| `tests/`（22 个文件，154 个用例） | 22 | 测试 | 2026-10-15 | — | 测试 | 12 个随保留代码进 `tests/<模块>/`；10 个（`test_r1_*`、`test_dense_score_*`、`test_gate_aligned_shaping`、`test_r5_production_backward`、`test_renderer_microbenchmark`、`test_sapave_taxonomy`、`test_r1_legacy_configs`）随对应代码归档 |
| `reports/`（3 份 9 月组会报告 + 图表，含 2 份"plus10pp"派生情景） | 74 | 文档 | 2026-10-06 | — | 历史报告 | `archive/history/reports/` |
| `exps/`（被跟踪的 57 个 R1 报告、哈希清单、启动脚本） | 57 | 无（R1 产物） | 2026-10-01 | — | 历史 | `archive/r1/artifacts/` |
| 根目录 `oracle_replay_*.csv`（8 个，v18/v19 oracle 回放） | 8 | 无 | 2026-07-17 | 无 | 历史 | `archive/experiments_legacy/oracle_replay/` |
| 根目录 `ply_gaussian_loader.py` | 1 | 1 渲染 | 2026-10-10 | 被 `gs_render_local.py` 和 3 个 sco 打包清单引用 | 核心 | 移入 `vagen/envs/active_spatial/render/` |
| 根目录 `mkdocs.yml`、`.readthedocs.yaml`、`setup.py`、`README.md` | 4 | — | 2026-10-10 | — | 上游文档站 / 打包 | `setup.py` 保留；文档站待定 D12；README 重写 |
| 集群未跟踪：根目录 `run_cambrian_*.sh`（5）、root 属主的 `c8_*.log` / `cambrian_smoke_*.log/.pid`（48 个，125 MiB）、`v15_per5action_*`（3 个，40 MiB） | 56 | 无 | 2026-07 | — | 一次性 | `archive/cluster_logs/` |

说明：
- 上表里"核心"合计 122 个原位保留 + 约 100 个需要搬动的 v10 / 训练 / 评测 / 下游文件 + 39 个另一会话的 QA 代码 + 54 个测试；"一次性 / 历史 / 旧评测栈 / 上游 / 待定"合计约 780 个，占 71%。
- 同一件事有多个实现的地方：(a) 轨迹评测：v10 `evaluation.py`（当前，逐步记录轨迹，带第九轮开关）与旧 `evaluation/eval_runner.py` + `scripts/active_spatial_eval_sweep.py`（v46 时代，读旧 100 场景数据）——建议留前者（决定 D3）；(b) SFT 数据生成：v10 `release.py` + `training_prep.py`（当前）与 `data_gen/active_spatial_sft/run_r1_sft_pipeline.py`（R1）——留前者；(c) 任务生成：v10 与 `data_gen/active_spatial_pipeline/run_pipeline.py`（v1）——留前者；(d) RL 入口：`vagen/interiorgs_v10_ppo.py`（当前）与 `run_experiment.sh` + `vagen/r1_clean_projective_ppo.py`——留前者；(e) sco 提交：`data_gen/interiorgs_v10/{job_sco,shard_sco,pilot_sco}.py`、`data_gen/interiorgs_qa/sco.py`（复用 pilot_sco 常量）、`scripts/r1_submit_*.py`、十几个 `sco_*_entry.sh`——第二阶段统一成 `scripts/cluster/sco.py` 一个实现，前两者作为调用方保留。

## 2. 集群文件盘点

集群目录 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite` 共 6.52 TiB；`/mnt/umm` 文件系统共 100 TB，已用 86 TB（86%），剩 15 TB。

### 2.1 顶层目录占用

| 目录 | 大小 | 内容 | 属主 | 建议 |
| --- | ---: | --- | --- | --- |
| `exps/` | 6,371 GiB | 实验产物，见 2.2 | 混合（root 属主一级条目 177 个，546 GB） | 按 2.2 处理 |
| `third_party/` | 110 GiB | `starVLA/playground/Checkpoints` 100.5 GiB（1 次 NavigateKitchen 训练）、`Pretrained_models` 8.3 GiB、RLinf 1.2 GiB、EASI / lmms-eval / LLaMA-Factory 源码 | 本人 | 保留（D9） |
| `outputs/` | 48.9 GiB | `robocasa_sft_smoke/20260930_164450/ckpt` 48.9 GiB（LLaMA-Factory 文本动作路线的 smoke 检查点）；其余 21 个 `2026-xx-xx/` 是 Hydra 日志，共 10 MiB | root | smoke 检查点删除候选（D5）；Hydra 日志归档 |
| `data_gen/` | 32.4 GiB | `active_spatial_qa/artifacts` 31.4 GiB（另一会话的 QA 生产产物）；旧生成器 / 旧 SFT 的 `output_*` 约 1 GiB | 本人 | artifacts 不碰；`output_*` 归档 |
| `cjepa/`、`spatial-training-room/`、`ele/` | 25.1 + 20.2 + 0.14 GiB | 三个有各自 GitHub 远端的独立仓库，嵌套在本仓库里；仓库代码没有任何引用（`spatial_training_room` 以可编辑方式装在 vagen-lite 环境里） | 本人 | 待定 D8（建议移到 `~/`） |
| `playground/` | 14.7 GiB | `Checkpoints/`（StarVLA NavigateKitchen 评测与 smoke，14.3 GiB）、`Datasets/robocasa365/v1.0`（0.39 GiB） | 本人（4 个 root 文件） | 并入 `downstream/` 固定数据位置（D9、D13） |
| `data/` | 10.3 GiB | `releases/interiorgs1000_active_spatial_v1`（R1 v1 发布：assets 9.7 GiB、production_v9 等） | 本人 | `archive/r1/`（D6） |
| `.venv-qwen3vl/`、`.venv-qwen3vl-runtime/` | 7.3 + 0.13 GiB | Qwen3-VL 的隔离运行环境 | 本人 | 待定 D4 |
| `evaluation/sweeps/`、`evaluation/outputs/` | 4.1 GiB | 8 月 clean 系列的 ID/OOD/EASI 评测矩阵结果（MASTER_RESULTS 的来源） | 本人 | `archive/results_legacy/`（压缩） |
| `docs/diagnosis/`（未跟踪 json / runs） | 2.3 GiB | D0 诊断的中间 json（`d0_8_runs` 2.06 GiB） | 本人 | `archive/diagnostics_d0_u1/` |
| 根目录散落文件 | 0.17 GiB | 48 个 root 属主的 7 月 Cambrian 日志 / pid、3 个 v15 崩溃日志、5 个 `run_cambrian_*.sh`、8 个 oracle csv | root / 本人 | `archive/cluster_logs/` |
| `wandb/`、`easi_results/`、`easi_probe_results/`、`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`vagen.egg-info/` | < 0.1 GiB | 离线 wandb（7 条）、8 月 EASI 小结、空目录、缓存 | 部分 root | wandb / easi_results 归档；其余删除 |

仓库外但流程依赖的数据（不在整理范围，写进 DATA.md）：`~/InteriorGS_meta`（1,000 个场景的 labels / structure / occupancy，0.32 GiB）、`~/InteriorGS_ply`（1,000 个场景的压缩 3DGS，40.8 GiB）、`~/InteriorGS`（root 属主，只有 100 个场景，不可读）、`~/.cache/huggingface/hub` 里的 Qwen2.5-VL-7B（15.5 GiB）与 3B（7.0 GiB）、`~/models/RLinf-Pi0-RoboCasa`（6.5 GiB）、`~/robocasa365/robocasa-main`（RoboCasa365 源码）。

### 2.2 `exps/` 逐类清单

308 个一级条目的逐条表在附录 [phase1/exps_inventory.tsv](phase1/exps_inventory.tsv)。按建议去向汇总：

| 类别 | 大小 | 条目数 | 代表目录 | 建议去向 |
| --- | ---: | ---: | --- | --- |
| 正在运行（第十轮） | 675 GB | 2 | `interiorgs_v10_train_r10_20261015`（两组 SFT 各 178 GB）、`interiorgs_v10_r10_20261015`（两个单卡 RL 试跑各 203 GB） | 不碰；本轮结束后按下面规则处理 |
| 最终数据集 | 6.7 + 3.5 GB | 2 | `interiorgs_v10_gen_r10_20261015`（当前，厚墙，13,091 正式题 + 14,433 后备题）、`interiorgs_v10_gen_r6b_20261014`（上一版，第一次 SFT/RL 所用） | 固定位置 `data/releases/`，带清单 |
| 评测集与评测结果 | 2.5 GB | 3 | `interiorgs_v10_r7_20261014/eval_c`（278 题评测集）、`interiorgs_v10_r8_20261014/eval2`（10 个模型的评测与表）、`interiorgs_v10_r9_20261015/eval_diag`（诊断评测） | `data/releases/`（评测集）、`results/`（结果） |
| 模型权重（v10 第一次训练） | 1,572 GB | 1 | `interiorgs_v10_train_20261014`：`lf/model` SFT 7B 及 3 个 epoch（119 GiB）；`rl_run/checkpoints` 7 个 global_step（1,453 GiB，其中 HF actor 7×约 16 GB，其余是 verl 分片） | HF 权重 → `models/`；verl 分片 1.35 TiB 待定 D7a |
| 模型权重（历史主线） | 62 GB | 1 | `unified_eval_runs/frozen_ckpts`（v46@200、v50@150，旧协议 7B RL） | `models/historical/`（D10） |
| R1 训练产物 | 1,238 GB | 1 | `R1-clean-Projective-v0-taskfix`：`formal/` 5 个 global_step 的 verl 分片（1,020 GiB，无 HF 导出）、`smoke/` 216 GiB | 待定 D6 |
| dense-score pilot 产物 | 1,762 GB | 1 | `dense_score_reward_only_pilot_parallel_r3_20261009/branches/`（S0/S1/S5 三支的训练分片） | 待定 D7b |
| 第五轮单卡试跑 | 268 GB | 1 | `interiorgs_v10_r5_20261014/trial`（3B SFT 模型 + RL 试跑；第十轮试跑复用其 3B SFT） | 3B SFT → `models/`；其余待定 D7c |
| 8 月 clean 系列（有 HF 权重，MASTER_RESULTS 引用） | 216 GB | 3 | `qwen_v50_clean_d0pass_20260824_full`（3 个 HF）、`cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full`（3 个）、`cambrian_c4_..._r5_full`（1 个） | HF → `models/historical/`；其余压缩归档（D10） |
| 8 月 clean 系列（仅日志 / 验证记录） | 4.2 GB | 39 | `qwen_v46/v47/v48_clean_*`、`cambrian_c8_*_r2/r3/r4`，大多 root 属主 | `archive/clean_runs_aug/`（压缩） |
| root 属主 smoke 残留 | 229 GB | 45 | `qwen3vl_active_spatial_smoke_h800_20261002_metricfix`（229 GB，1 个检查点）、8 个其他 qwen3vl smoke、20 个 `*_acceptance_smoke`、`d0_17_*` smoke | 删除候选（见第 6 节删除清单） |
| v10 各轮审计中间结果 | 107 GB | 5 | `interiorgs_v10_r6_20261014`（102 GiB，含 `backup_original_dir_20261014`）、`r4`、`cabinet_audit`、`diag`、`real` | `archive/interiorgs_v10_rounds/`（压缩） |
| 作废 / 中间数据集 | 2.9 GB | 8 | `interiorgs_v10_gen_20261011{,b,c}`、`gen_20261012_final`、`gen_20261014`、`gen_r6_20261014`（STATUS 明确作废） | `archive/data_intermediate/`（压缩） |
| R1 中间产物 | 100 GB | 11 | `r1_h1_aoss_repair_20260905`（66 GB）、`R1-clean-Projective-v0`（35 GB，含发布 json、冻结数据、1 个检查点）、`R1-gate-aligned-*`、`R1-task-context-*`、`R1-experiment-tracking` | `archive/r1/`（压缩；冻结数据与发布 json 不压缩） |
| dense-score 中间产物 | 94 GB | 9 | `active_spatial_dense_score_diagnostic_real_batch_20260916_r2/r4/r5`（各 31 GB，root）、`dense_score_reward_only_pilot_*20261007`、renderer microbenchmark | `archive/dense_score_pilot/`（压缩） |
| 7–8 月旧实验（检查点已删，只剩日志和验证记录） | 18 GB | 27 | `c8_fwdfirst_rewscale_lfp_server_v*`、`v46_baseline_qwen25vl_{3b,7b}`、`a1_*`、`a2_*`、`b5_*`、`protocol_only_prepost`、`u1_*` | `archive/experiments_legacy_runs/`（压缩） |
| 旧协议评测结果 | 2.1 GB | 25 | `unified_eval_runs/` 除 frozen_ckpts 外（EASI 结果、one_ckpt 评测、`MASTER_RESULTS.md`） | `archive/results_legacy/`；`MASTER_RESULTS.md` 的表并入 EXPERIMENTS.md 历史节 |
| 散落日志与任务记录 | 0.6 GB | 97 | `*.log`、`sco_renderer/`、`render_logs/`、`sco_submissions/`、`launch_overrides/` | `archive/cluster_logs/`（压缩） |
| 另一会话（QA） | 6.5 GB | 15 | `exps/interiorgs_qa_*` | 不碰 |

root 属主文件：`exps/` 一级条目里 177 个属 root（sco 任务以 root 运行写出的），`outputs/` 下 621 个，`wandb/` 46 个，根目录 52 个。本用户不能直接移动或删除它们；第二阶段用现有的 `sco_run_as_artifact_owner.sh`（提交一个 sco 任务执行 `chown`）先改回属主，再归档。

## 3. 文档盘点

103 个被跟踪的 md 文件按组列出。"判断"分四档：**有效**（内容与当前代码一致）、**部分**（主体有效，个别章节过时）、**过时**（描述的代码或状态已被取代）、**历史**（当时正确的记录，不再更新）。

| 文档 | 日期 | 判断 | 建议 |
| --- | --- | --- | --- |
| `README.md` | 2026-10-10 | 过时：主体是上游 VAGEN 的首页，本项目只有开头三行入口 | 重写（第三阶段） |
| `docs/active_spatial.md`（项目入口、代码导航表） | 2026-10-10 | 部分：代码入口表基本准确；"当前主线是 R1"的判断已被 v10 取代 | 并入 README / PIPELINE.md 后归档 |
| `docs/active_spatial_end_to_end.md` + `active_spatial_pipeline_{data,rl,robot,audit}.md` | 2026-10-10 | 部分：流程逻辑、SFT→RL→StarVLA→RLinf 的接口约定、已修复问题清单有效；数据和环境部分以 R1 语料为例，已被 v10 取代；"本机路径示例"都指向 R1 目录 | 作为 PIPELINE / TRAINING / DOWNSTREAM 的底稿，重写后归档 |
| `docs/active_spatial_eval_system.md` | 2026-07-08（08-31 更新） | 过时：描述旧评测栈；其中 7 个 OOD 轴的定义可参考 | 随 D3 归档；OOD 轴定义引用到 EVAL.md |
| `docs/active_spatial_task_reward_audit.md` | 2026-07-13 | 过时：旧 8 任务奖励审计 | 归档；写 SCORE.md 时参考 |
| `docs/active_spatial_remote_env.md` | 2026-08-31 | 有效（RemoteEnv 服务） | 并入 CLUSTER.md / TRAINING.md |
| `docs/active_spatial_new_server_handoff.md` | 2026-07-17 | 历史（文首已注明） | `archive/history/docs/` |
| `docs/active_spatial_qwen3vl_migration.md`、`model_backbone_integration.md`、`cambrian_iteration.md`、`cambrian_s_new_server_smoke_handoff.md`、`sensenova_u1_active_spatial.md` | 2026-06 至 10-10 | 旁支模型（Qwen3-VL / Cambrian-S / SenseNova-U1） | 待定 D4，建议归档 |
| `docs/gpt6_id_ood_eval.md` | 2026-10-06 | 依赖旧评测栈 | 待定 D3 |
| `docs/robocasa_vlm_sft_eval.md` | 2026-10-06 | 部分：RoboCasa 远程评测、StarVLA 迁移章节有效；LLaMA-Factory 文本路线待定 D5 | 并入 DOWNSTREAM.md |
| `docs/v19_v20_progress.md`、`docs/v26_v27_analysis.md` | 2026-05/06 | 历史实验记录（v19–v28、c1/c2） | `archive/history/docs/` |
| `docs/{index,quickstart,configuration,custom-environment,custom-filter,custom-metric,issues}.md` | 2026-02/03 | 上游 VAGEN 文档 | 待定 D2 / D12 |
| `docs/diagnosis/`（22 份：D0 审计 4,448 行、历史有效性、R1 交接、clean 启动记录、dense-score 各报告、10-02 接管审计） | 2026-08 至 10-10 | 历史；10-02 接管审计的"证据边界"表仍是对 8 月结果最准确的说明 | `archive/history/diagnosis/`；接管审计结论摘要写进 EXPERIMENTS.md 历史节 |
| `docs/interiorgs_qa/{COVERAGE,PROGRESS,REPORT_20261014,REPORT_20261015}.md` | 2026-10-14/15 | 有效（另一会话维护） | 保留；完工后并入 DATA.md / EVAL.md |
| `data_gen/interiorgs_v10/STATUS.md` | 2026-10-15 | 有效（v10 唯一状态文档，到第十轮） | 第三阶段并入根 `STATUS.md` 与 DATA.md / SCORE.md |
| `data_gen/interiorgs_v10/SPEC.md`、`families/ENV_INTEGRATION.md` | 2026-10-11/13 | 有效（约定、模块归属、族与环境的对接） | 并入 DATA.md / SCORE.md |
| `data_gen/interiorgs_v10/README.md` | 2026-10-13 | 部分："Status"段落停在 10-12 | 并入 DATA.md |
| `data_gen/interiorgs_v10/CLUSTER_RUN_HANDOFF.md`、`REPORT_SYNTHETIC.md` | 2026-10-11 | 历史 | `docs/reports/history/interiorgs_v10/` |
| `data_gen/interiorgs_v10/cluster_runs/`（12 份轮次报告 REPORT_REAL…REPORT_R10、GPU_PLAN、CABINET_AUDIT） | 2026-10-11 至 10-15 | 历史（R10 结论待训练结束后填） | `docs/reports/history/interiorgs_v10/`，建索引 |
| `data_gen/active_spatial_pipeline/{README（1,651 行）,CORNER_COUNTING_USAGE,IMPLEMENTATION_SUMMARY,ood_splits/ood_splits_summary}.md` | 2026-06/10 | 过时（v1 生成器） | 随代码归档 |
| `data_gen/active_spatial_sft/{README,SFT_SCORE_BUG_ANALYSIS}.md` | 2026-05/10 | 过时 | 随代码归档 |
| `data_gen/active_spatial_qa/README.md`、`data_gen/robocasa_sft/README.md` | 2026-10 | 待定 D6 / D5 | 随决定处理 |
| `evaluation/README.md`、`examples/evaluate/active_spatial/README.md`、`examples/train/active_spatial/A1_A2_B5_README.md` | 2026-10-06 / 08-31 | 旧评测栈 / 历史实验 | 随 D3 归档 |
| `examples/train/robocasa/README.md`、`examples/evaluate/robocasa/README.md` | 2026-10-10 | 有效 | 并入 DOWNSTREAM.md |
| `vagen/envs/{navigation,primitive_skill,spatial_gym}/README.md`、`vagen/inference/README.md` | 2026-03/06 | 上游 | 待定 D2 |
| `vagen/envs_remote/README.md`、`vagen/evaluate/README.md` | 2026-03 | 有效（框架说明） | 保留 |
| `reports/*/GROUP_MEETING_REPORT_CN.md`、`SLIDE_OUTLINE_CN.md`（3 组，其中 2 组是"plus10pp"加分派生情景） | 2026-09-06 | 历史；派生情景不能当实测 | `archive/history/reports/`，索引里注明派生 |
| `exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/*.md`（8 份 R1 报告） | 2026-09 | 历史 | `archive/r1/` |
| 集群未跟踪：`exps/unified_eval_runs/MASTER_RESULTS.md`（旧协议成绩账本） | 2026-10-07 | 有效作为历史结果表 | 表格并入 EXPERIMENTS.md 历史节；原件归档 |
| pdf：`reports/*/figures/*.pdf` 19 个 | 2026-09/10 | 报告配图 | 随报告归档 |

应合并的文档（第三阶段写新文档时）：`active_spatial_end_to_end` 系列 5 份 + `active_spatial.md` + `robocasa_vlm_sft_eval.md` + 两个 robocasa README → PIPELINE.md / TRAINING.md / DOWNSTREAM.md；`interiorgs_v10/{STATUS,SPEC,README}` + `families/ENV_INTEGRATION.md` + `interiorgs_qa/COVERAGE.md` → DATA.md / SCORE.md；`active_spatial_remote_env.md` + 各 sco 入口里的常量 → CLUSTER.md。

## 4. 流程缺口（对照 7 步）

| 步骤 | 现在有什么（代码与证据） | 缺什么 |
| --- | --- | --- |
| 1 环境与评分 | 环境 `vagen/envs/active_spatial/env.py`（2,221 行）；成功判定：Projective / FOV 的 v1 判据 `canonical_task_metrics.py`（R1 冻结）、6 个任务族的 v2 判据 `canonical_task_metrics_v2.py` + `data_gen/interiorgs_v10/families/*.py`；可见规则 `target_visibility.py`（face_visible_v3）、厚墙碰撞 `collision_detector.py`；奖励：`env.py compute_reward`（S0 稀疏）、`spatial_potential_field.py` + `reward_trace.py`（密集 shaping，S1/S5 候选）；当前默认规则在 `data_gen/interiorgs_v10/configs/rl_env_v10_S0.yaml`（每个动作检查成功、撞墙不结束、如实反馈、带历史、上限 12）。测试：`test_reward_trace`、`test_gate_aligned_shaping`、v10 `test_success_rules_r5`、`test_env_switches`、`test_walls_r10` 等 | 没有一份文档把 6 个族的条件、公式、门槛和奖励各分支写全（SCORE.md 要新写）；密集奖励 S1/S5 只在 `examples/.../dense_score_ablation_*.yaml` 里有配置，pilot r3（10-09，1.7 TiB）是否跑完、结果如何没有报告（第二阶段核实）；旧的 `success-threshold` 等参数散在 5 个地方 |
| 2 数据生成 | 轨迹：v10 生成器（`run_all` → `release` → `training_prep`），第十轮数据 938 场景 / 13,091 正式题，split 按场景 800/100/100，每轮有 `code_commit.txt` 和 `split_manifest.json`；SFT 轨迹由训练链渲染导出。QA：(a) `interiorgs_qa`：仿 8 个 benchmark 的 51 个子任务出题（单图 / 多图 / 视频），230 道样题在标注，题库未全量生成，有 `sampling_plan` 分训练 / 评测；(b) `active_spatial_qa`：与 R1 任务配对的 Yes/No QA（生产 v1–v3，31 GiB 产物，另一会话） | 统一的数据集清单格式（版本、来源、命令、条数、split、校验和）没有；两条 QA 线哪一条算"自有 QA"要定（D11）；"主流 benchmark 复现"即 `interiorgs_qa`，其全量生成（B4）等标注结果 |
| 3 SFT | LLaMA-Factory（`third_party/`，2025-06 版本，关掉版本检查）；`prepare_active_spatial_sft.py` 支持轨迹与 QA 按概率混合（`--qa-probability`）、按场景划分；训练链 `train_chain.py sft`（8 卡 sco，Qwen2.5-VL-7B，每 epoch 存）；已完成 r6b 数据 3 个 epoch，第十轮两组 SFT 在跑；`interiorgs_qa/export_lf.py` 导出同一格式，7B 单卡前向反向通过 | 轨迹 + QA 混合比例作为实验因素的配置入口（现在要手工拼命令）；只用 QA 的 SFT 没跑过；SFT 产物清单 |
| 4 RL | 环境里 RL：`vagen/interiorgs_v10_ppo.py` + `sco_interiorgs_v10_train.sh rl`（PPO，no_concat_gae，S0 稀疏，1+8 卡）；已跑到 250 次更新（评测成功率 21.6%，与 SFT 第 3 epoch 的 23.4% 无显著差别）；带历史的 concat / no_concat 两种做法在试跑。QA 上 RL：`vagen/envs/spatial_mcq/`（单步选择题环境，答案字母匹配给分）+ `prep_spatial_aux_mcq.py`（从 SITE 抽 MCQ），只在 8 月 A1 实验里作为辅助环境按比例混入过（`AUX_ENV_CONFIG` + `SPATIAL_AUX_RATIO`，走旧 launcher）；密集奖励：配置存在，无成功训练 | QA 环境不认 `interiorgs_qa` 的题格式（jsonl：question / options / answer / image_paths，多图与视频）和自由文本答案（`interiorgs_qa/scoring.py` 已有打分，可复用）；辅助环境混入没有接到 v10 入口；"先后 / 混合"两种组合方式没有配置；密集奖励需先核实 pilot 结论 |
| 5 实验矩阵 | 只有 R1 的 `R1-experiment-tracking/experiment_registry.json`（2 个实验）和旧协议账本 `MASTER_RESULTS.md`；v10 各轮结果写在轮次报告里 | 全部缺：登记表、ID 规则、一条命令启动、汇总出表、误差范围、分批计划（第四阶段） |
| 6 标准化测评 | 自有 QA：`interiorgs_qa/evaluate.py`（vLLM 推理、按组 / benchmark 汇总、95% 区间），基础模型在 240 题上跑过。EASI：`scripts/easi_eval.py` + `third_party/EASI`（v0.2.2）+ `lmms-eval`，8 月 9 日前跑过 12 个检查点的 EASI-8，结果在 `MASTER_RESULTS.md`；登记表里 30 个检查点路径多已删除。ID 轨迹：v10 `evaluation.py` 在 test 的 278/279 题上评测，带每步轨迹和浏览页，已评 10 个模型。OOD：`gen_ood_splits.py` 按场景 / 实例 / 类别 / 模板 / 几何五轴切分，只认旧 100 场景数据的字段 | EASI 自 8 月后未跑，环境（vagen-lite：vllm 0.11、transformers 4.56）变过，要重新冒烟；OOD 对 v10 没有定义和数据（建议轴：未见场景已有、未见物体类别、未见任务族、d\* 超出训练范围、改写指令）；没有"一条命令跑完四类测评出一张表"；测试集没有版本号 |
| 7 下游 | StarVLA：`third_party/starVLA`（`starVLA_dev` 分支 4507931 + 本地补丁 `active_spatial_transfer.patch`），conda 环境 `starVLA`（Python 3.10，torch 2.6），`prepare_starvla_transfer.py` 能从本项目 HF 权重出发并有 10 个测试；已跑过一次 NavigateKitchen 8 卡训练（起点是 Qwen3-VL-4B-Action，**不是本项目权重**，100 GiB 检查点在 `third_party/starVLA/playground`）。RLinf：`third_party/RLinf`（c70606f + 补丁 `rlinf_robocasa_starvla.patch`），自带 `.venv`（robosuite、mujoco、lerobot、openpi 已装），`prepare_rlinf_robocasa.py` 能生成 PPO 配置并通过 CPU 测试；RoboCasa365 数据 `playground/Datasets/robocasa365/v1.0`（0.39 GiB，疑似只有部分任务）、源码 `~/robocasa365`、Pi0 权重已下载。评测：`scripts/robocasa_eval.py`、`run_starvla_mobile.sh`，闭环跑过 2 条（"0/2 只证明能执行"） | RLinf 从未启动过 GPU 训练（10-09 审计时 CUDA 不可用）；RoboCasa365 模拟器的运行环境（文档说的"policy venv Python 3.10"）具体在哪个环境没有确认；StarVLA 从本项目权重出发的 SFT 没跑过；robocasa365 数据集完整性未核实；这一步的冒烟要一张卡 + 模拟器，第三阶段单独核验 |

专项核查（任务书第 2.4 条点名的四项）：

- **EASI**：有代码、有第三方库、有历史结果；环境是否仍能跑需要冒烟；登记表要重建（现有 30 条里大多指向已删检查点）。
- **OOD 轨迹**：有旧数据的切分器，没有 v10 的定义与数据；需要在第三阶段定义并生成（CPU 任务，约半天）。
- **QA 上的 RL**：有单步 MCQ 环境雏形和旧 launcher 的混入机制；要新写一个读 `interiorgs_qa` 格式、用 `scoring.py` 给分的环境适配器，并接到 `interiorgs_v10_ppo.py`（约 1–2 天）。
- **StarVLA / RLinf 在 robocasa365**：代码、补丁、环境、权重都在；StarVLA 训练能跑（已有一次），RLinf 训练从未跑过；需要单卡冒烟核验环境，不能假设。

## 5. 新目录结构方案

按你的建议，两处改动并说明理由：

1. **保留 Python 包名 `vagen` 和 `data_gen`**。`vagen` 是 `setup.py` 安装的包名，verl 的 Hydra 配置、`python -m vagen.main_ppo` 等都用它；`data_gen.interiorgs_v10.*` 在 300 多处 import、3 个 sco 打包清单和另一会话的 `interiorgs_qa/sco.py` 里出现。改名只是好看，代价是全量替换和打断正在跑的链；建议 `data_gen/` 就作为你方案里的 `datagen/`。
2. **`evaluation/` 名字复用**：现在的 `evaluation/` 是旧评测栈；先按 D3 归档，再把 v10 的评测代码放进去。

```
VAGEN-Lite/
├── vagen/                      # 环境、任务定义、成功判定、分数（步骤 1）+ RL 训练框架（步骤 4）
│   ├── envs/active_spatial/    #   环境、6 个族的判据、可见规则、碰撞、奖励、渲染（含 ply_gaussian_loader）
│   ├── envs/spatial_mcq/       #   QA 上 RL 的环境（第三阶段改造为读 interiorgs_qa 格式）
│   ├── envs/robocasa/ envs_remote/ evaluate/ server/ rollout/
│   └── main_ppo.py ray_trainer.py interiorgs_v10_ppo.py agent_loop/ custom_*/ utils/ configs/
├── data_gen/                   # 数据生成（步骤 2）
│   ├── interiorgs_v10/         #   任务出题、回放、split、SFT 导出（families/ configs/ 在内）
│   ├── interiorgs_qa/          #   QA 出题（另一会话完工后接管）
│   └── common/                 #   verify_dataset.py、release_contract.py、manifest 工具
├── training/                   # SFT 与 RL 的配置与启动（步骤 3、4）
│   ├── train_chain.py training_prep.py job_sco.py sco_train.sh
│   ├── sft/  (prepare_sft.py、accelerate_fsdp_qwen25vl.yaml、LLaMA-Factory 模板)
│   ├── rl/   (prepare_rl.py、rl_env_v10_S0.yaml 等环境模板、ppo 模板)
│   └── rl_qa/ (prep_mcq.py、QA 环境配置)
├── evaluation/                 # 标准化测评（步骤 6）
│   ├── trajectory/  (evaluation.py、eval_browser.py、fail_analysis.py、rl_checks.py)
│   ├── easi/        (easi_eval.py、registry.yaml、easi_table.py、launch_parallel_easi.sh)
│   ├── qa/          (调用 data_gen/interiorgs_qa/evaluate.py 的封装)
│   ├── ood/         (gen_ood_splits.py 改造版 + OOD 定义)
│   └── run_all.py   (一条命令：四类测评 → 一张表)
├── downstream/                 # StarVLA / RLinf / RoboCasa365（步骤 7）
│   ├── starvla/ rlinf/ robocasa_eval/ patches/
├── experiments/                # 实验矩阵（步骤 5）：registry.yaml、configs/<ID>/、launch.py、aggregate.py、batches/
├── scripts/                    # 集群工具：cluster/（sco.py 统一提交、chown、aoss、同步）、render/（渲染与环境服务）
├── configs/                    # 统一配置：paths.yaml（本机 / 集群两套路径）、models.yaml、datasets.yaml、cluster.yaml
├── docs/                       # 第 4 节的 11 份文档 + reports/（阶段报告、history/ 轮次报告索引）
├── tests/                      # envs/ datagen/ training/ evaluation/ downstream/ regression/（金标准比对）
├── archive/                    # README.md 索引 + r1/ experiments_legacy/ upstream/ diagnostics_d0_u1/ dense_score_pilot/
│                               #   eval_legacy/ legacy_pipeline_v1/ legacy_sft/ side_models/ analysis/ history/{docs,reports,diagnosis}
├── third_party/                # 不入 git；EASI、lmms-eval、LLaMA-Factory、starVLA、RLinf，版本号与补丁记在 downstream/patches 和 CLUSTER.md
├── verl/                       # 子模块
└── （集群上，不入 git）data/releases/<数据集>_<版本>/  models/<名称>/  results/<实验ID>/  各带 manifest.json；manifest 的副本进 git 的 configs/datasets.yaml、configs/models.yaml
```

统一配置的做法：所有绝对路径（`/mnt/umm/...`、conda 环境、sco 的 workspace / pool / 镜像 / 挂载）集中到 `configs/paths.yaml` 和 `configs/cluster.yaml`，代码里通过一个 `vagen.config` 模块读取，本机用 `VAGEN_PROFILE=local` 切换；现在这些常量散在 `pilot_sco.py`、`sco.py`、`easi_registry.yaml`、`run_experiment.sh`、各 `sco_*_entry.sh` 和文档里。

实验 ID 的初步设想（第四阶段细化）：`E<三位序号>_<起点>_<SFT 数据>_<RL>_<奖励>_<规模>_s<种子>`，例如 `E012_base_mix50_env_sparse_full_s1`；登记表记因素取值、配置目录、数据版本、代码提交号、状态、结果路径。

## 6. 删除清单与归档清单

两份清单分开列。删除清单里的东西都不在 git 里，删前仍会先移到 `archive/_pending_delete/` 并写清单，由你最终确认。

### 6.1 删除清单（候选，共约 0.5 TiB + git 内图片）

| 项目 | 大小 | 为什么可以删 |
| --- | ---: | --- |
| `exps/vagen_active_spatial/qwen3vl_active_spatial_smoke_h800_20261002_metricfix/` 及 8 个 `qwen3vl_*smoke*`（root） | 229 GB | Qwen3-VL 一步更新的工程 smoke，接管审计已记录结论；1 个检查点无评测价值 |
| `exps/vagen_active_spatial/*_acceptance_smoke/`（20 个）、`d0_17_*smoke*`、`qwen_v46_clean_d0pass_20260819_8gpu_smoke` 等（root） | < 1 GB | 8 月 clean 启动前的验收 smoke，只有日志 |
| `outputs/robocasa_sft_smoke/20260930_164450/ckpt/`（root） | 48.9 GiB | LLaMA-Factory 文本动作路线的 smoke 检查点（随 D5） |
| `easi_probe_results/`（空）、`__pycache__/`（root）、`.pytest_cache/`、`.ruff_cache/`、`vagen.egg-info/`、`data_gen/*/__pycache__` | < 0.1 GiB | 缓存 |
| git 里的 `data_gen/active_spatial_sft/output_*/`（39,874 张图 + jsonl）与 `data_gen/active_spatial_pipeline/output*/` 产物 | 403 + 465 MB（blob） | 数据不该进 git；磁盘副本随代码归档（是否重写历史见 D1） |
| 仓库根目录 48 个 root 属主的 `c8_*.log` / `cambrian_smoke_*.log/.pid` | 125 MiB | 7 月 Cambrian 调试日志；建议先归档压缩再删（列在归档清单也可） |

待你决定后可能进入删除清单的大项（见第 8 节）：R1 训练分片 1.2 TiB（D6）、v10 第一次 RL 的 verl 分片 1.35 TiB（D7a）、dense-score pilot 分片 1.7 TiB（D7b）、第五轮试跑 0.26 TiB（D7c）。四项合计 4.5 TiB，是集群占用的 70%。

### 6.2 归档清单

git 内（移到 `archive/`，保留历史，建索引）：

| 来源 | 文件数 | 归档位置 |
| --- | ---: | --- |
| `scripts/r1_*`、`vagen/r1_clean_projective_ppo.py`、`exps/`（57 个被跟踪文件）、`examples/evaluate/active_spatial/r1_*.sh`、`examples/train/active_spatial/sco_r1_*.sh`、`tests/test_r1_*`、`tests/active_spatial/test_r1_legacy_configs.py`、`docs/diagnosis/ACTIVE_SPATIAL_R1_HANDOFF_20260905.md`、`data/releases/interiorgs1000_active_spatial_v1`（集群） | ≈ 220 | `archive/r1/` |
| `examples/train/active_spatial/{experiments,env_config_*,launch_*,submit_*,sco_*_entry.sh,gpu_*,run_experiment.sh,train_*,val_*,...}`、根目录 `oracle_replay_*.csv`、`A1_A2_B5_README.md` | ≈ 254 | `archive/experiments_legacy/`（每个实验一行索引：名称、日期、变量、结果位置） |
| `scripts/d0_*`、`tools/`、`docs/diagnosis/D0_CAMBRIAN_PLUMBING_AUDIT.md` 等 D0 文档、`docs/diagnosis/` 未跟踪 json | 37 + 文档 | `archive/diagnostics_d0_u1/` |
| `scripts/*dense_score*`、`vagen/dense_score_reward_only_pilot.py`、`tests/active_spatial/test_dense_score_*`、`test_gate_aligned_shaping.py`、`docs/diagnosis/active_spatial_dense_score_*` | ≈ 15 | `archive/dense_score_pilot/` |
| `scripts/{active_spatial_offline_ppo_replay,...}` 10 个诊断脚本、`tests/active_spatial/test_r5_production_backward.py`、`test_renderer_microbenchmark.py` | 12 | `archive/diagnostics/` |
| `scripts/{visualize_*,analyze_active_spatial_training,build_baseline_trained_analysis}.py` | 4 | `archive/analysis/` |
| `data_gen/active_spatial_pipeline/`（代码与文档） | 25 | `archive/legacy_pipeline_v1/` |
| `data_gen/active_spatial_sft/`（代码与文档；`convert_to_qwen25vl_sft.py` 里被引用的函数先并入 `training/sft/`） | 21 | `archive/legacy_sft/` |
| `reports/`、`docs/{v19_v20_progress,v26_v27_analysis,active_spatial_new_server_handoff,active_spatial_task_reward_audit,active_spatial_eval_system}.md`、`docs/diagnosis/` 其余报告、`data_gen/interiorgs_v10/cluster_runs/*.md`、`CLUSTER_RUN_HANDOFF.md`、`REPORT_SYNTHETIC.md` | ≈ 110 | `archive/history/{reports,docs,diagnosis}/` 与 `docs/reports/history/`（轮次报告留在 docs 下便于引用） |
| 待定项（D2 上游 206 个文件、D3 旧评测栈约 60 个、D4 旁支模型约 35 个、D5 文本路线 8 个、D6 配对 QA 53 个） | ≈ 360 | `archive/upstream/`、`archive/eval_legacy/`、`archive/side_models/`、`archive/legacy_text_action/`、`archive/paired_qa_r1/` |

集群（移到 `VAGEN-Lite/archive/` 下，大目录用 `tar + zstd` 压缩，每个写一行清单：原路径、大小、日期、内容、压缩后路径、校验和）：

| 来源 | 大小 | 归档位置 |
| --- | ---: | --- |
| 8 月 clean 系列 39 个日志目录 + 3 个带 HF 权重的目录（HF 权重另存 `models/historical/`） | 220 GB → 压缩后约 100 GB | `archive/clean_runs_aug/` |
| R1 中间产物 11 个目录（冻结数据、发布 json 不压缩） | 100 GB | `archive/r1/` |
| dense-score 中间产物 9 个目录 | 94 GB | `archive/dense_score_pilot/` |
| v10 各轮审计中间结果 5 个目录、作废数据集 8 个目录 | 110 GB | `archive/interiorgs_v10_rounds/`、`archive/data_intermediate/` |
| 7–8 月旧实验 27 个目录（仅日志） | 18 GB | `archive/experiments_legacy_runs/` |
| 旧协议评测结果（`unified_eval_runs` 25 个条目、`evaluation/sweeps`、`evaluation/outputs`、`easi_results`、`wandb`） | 6 GB | `archive/results_legacy/` |
| 散落日志 97 条、根目录日志、`outputs/2026-*` Hydra 日志、`docs/diagnosis` 未跟踪 json | 3 GB | `archive/cluster_logs/`、`archive/diagnostics_d0_u1/` |
| `data_gen/active_spatial_pipeline/output_100scenes`、`active_spatial_sft/output_*`、`sapave/output` | 1 GB | `archive/data_legacy/` |

固定位置（整理后，写进 DATA.md / TRAINING.md / EVAL.md）：`data/releases/`（r10、r6b 数据集；278 题评测集；之后的 OOD 集、QA 题库）、`models/`（SFT ep1–3、RL 7 个 HF actor、3B SFT、历史 v46@200 / v50@150 等）、`results/`（r8 eval2、r9 eval_diag、之后每个实验 ID 一个目录）。

## 7. 工作量估计

按"一个人全职做"的工作日估计；集群排队、压缩大目录、等你确认的时间另计。

| 阶段 | 工作 | 估计 | 前提 / 风险 |
| --- | --- | ---: | --- |
| 第二阶段：整理 | 打标签、建分支、记录两套测试基线和 6 组金标准输出 | 0.5 天 | 金标准：固定 2 个场景出题、20 道题回放、5 道题 SFT 导出、1 个场景 QA 出题（需单卡）、固定轨迹文件评分、固定预测文件 EASI 评分 |
| | 代码搬动与 import 修改（约 90 个保留文件搬家、约 880 个归档、`data_gen/active_spatial_sft` 两个被引用函数并入、sco 提交统一为一个实现） | 1.5 天 | **必须等第十轮训练链结束**（训练链每个任务打包当前工作树；STATUS 规定训练期间不改代码），预计 1–2 天后 |
| | 统一配置（`configs/paths.yaml` 等，替换约 25 处写死路径） | 1 天 | 另一会话的 `interiorgs_qa/sco.py` 引用 `pilot_sco` 常量，改法要和它商量 |
| | 数据 / 权重 / 结果清单（manifest 格式、生成工具、给现有 4 个数据集与约 15 个权重补清单） | 1 天 | |
| | 测试重组 + 回归测试（金标准比对） | 1 天 | 全套 pytest 17 分钟 |
| | 历史文档打包与索引 | 0.5 天 | |
| | 集群归档（chown 的 sco 任务、搬移、压缩约 0.6 TB、写清单） | 1 天人工 + 1–2 天后台 | 压缩在登录节点跑，受 14 核限制 |
| | 合计 | **5–6 天** | |
| 第三阶段：穿流程、写文档 | 11 份文档（README、PIPELINE、SCORE、DATA、TRAINING、EVAL、DOWNSTREAM、EXPERIMENTS、CLUSTER、CHANGELOG、archive/README） | 3 天 | SCORE.md 要从 6 个族的代码里逐条抄公式和门槛 |
| | 步骤 1–4、6（ID 轨迹、自有 QA）的冒烟：2 个场景出题 → 几十道题 → SFT 几十步 → RL 两次更新 → 评测几道题 | 1.5 天 | 每步一个单卡 sco 任务；排队时间不计 |
| | EASI 环境修复与冒烟（登记表重建、lmms-eval 依赖） | 0.5–1 天 | 环境可能要单独 venv |
| | OOD 定义 + 生成器改造 + 冒烟 | 1 天 | |
| | QA 上 RL：环境适配器（读 interiorgs_qa 格式、`scoring.py` 给分）+ 接入 v10 入口 + 两次更新冒烟 | 1.5–2 天 | 新代码 |
| | 下游：StarVLA 从本项目权重出发的冒烟、RLinf 单卡冒烟、RoboCasa365 环境核验 | 1.5–2 天 | 不确定性最大；跑不通就如实记录缺什么 |
| | 合计 | **8–10 天** | |
| 第四阶段：实验框架 | 登记表、ID 规则、`launch.py`、`aggregate.py`（含置信区间）、分批计划 | 2.5–3 天 | 依赖第三阶段每步命令稳定 |

## 8. 需要你决定的事项

每项给出选项和代价；标"推荐"的是我的建议。

| 编号 | 事项 | 选项与代价 |
| --- | --- | --- |
| D1 | git 历史里的约 4 万个图片 / 视频文件和旧产物（共 868 MB blob，pack 236 MB） | (a) 只从当前树移除，历史保留：clone 仍 240 MB，零风险。(b) **推荐**：整理分支合并后用 `git filter-repo` 重写历史：clone 降到约 40 MB；代价是强制推送、所有克隆要重新 clone、旧提交号全部变化（文档里引用的提交号要换成标签）；重写前在集群留一份镜像作为归档。 |
| D2 | 上游 VAGEN 的环境、示例与文档（sokoban / frozenlake / navigation / primitive_skill / spatial_gym、`vagen/inference`、7 份上游文档，共 206 个文件） | (a) **推荐**：移到 `archive/upstream/`，README 说明本仓库源自 VAGEN-Lite 并给上游链接；代价：以后要跑上游环境得从归档拿。(b) 原位保留：仓库继续混着不相关代码。 |
| D3 | 两套轨迹评测代码 | (a) **推荐**：统一到 v10 `evaluation.py`，旧评测栈（`evaluation/`、`scripts/*eval_sweep*` 等约 60 个文件、`sapave`、GPT-6 runner）归档；代价：要评 API 模型（GPT-6）时需给 `evaluation.py` 加一个 provider，约 1 天。(b) 两套都留：文档和测试都要维护两份。 |
| D4 | 旁支模型路径：Cambrian-S、SenseNova-U1、Qwen3-VL（`vagen/models` 11 个文件、`tools/u1_*` 16 个、5 份文档、`.venv-qwen3vl` 7.3 GiB、`exps/u1_*`） | (a) **推荐**：代码和文档归档、venv 保留在磁盘；实验矩阵的"起点模型"只有 Qwen2.5-VL。(b) 保留：每份文档都要为它们留章节。 |
| D5 | LLaMA-Factory"文本写 12 维动作"的 RoboCasa 路线（`data_gen/robocasa_sft` 等 8 个文件 + 49 GiB smoke 检查点） | (a) **推荐**：代码归档、检查点删除；下游只走 StarVLA 连续动作头。(b) 保留作为对照路线。 |
| D6 | R1 线的产物：`R1-clean-Projective-v0-taskfix` 的 verl 分片 1.2 TiB（无 HF 导出）、`R1-clean-Projective-v0` 1 个检查点、`data/releases/..._v1` 10 GiB | (a) **推荐**：导出 taskfix 最后一步（step 250）的 HF actor（约 16 GB）后删除全部分片；冻结数据、发布 json、报告归档。(b) 全部压缩归档：约 0.6 TiB。(c) 全删。 |
| D7a | v10 第一次 RL（`interiorgs_v10_train_20261014/rl_run`）7 个 global_step 的 verl 分片 1.35 TiB | (a) **推荐**：保留 7 个 HF actor（约 112 GB）和日志，删除 critic / optimizer 分片；代价：不能从这些步续训。(b) 只保留第 250 步的分片（约 190 GB）。(c) 全留。 |
| D7b | dense-score pilot r3（10-09，1.7 TiB，S0/S1/S5 三支）| 先核实是否跑完、有无结论（第二阶段第一件事）。(a) **推荐**：有 HF actor 则各留一个，其余删；无结论则只留日志和配置。(b) 全留。 |
| D7c | 第五轮单卡试跑 `interiorgs_v10_r5_20261014/trial` 0.26 TiB | (a) **推荐**：只留 3B SFT 模型（第十轮试跑在用），删 RL 试跑分片。(b) 全留。 |
| D8 | 嵌套的三个独立仓库 `cjepa`、`spatial-training-room`、`ele`（45 GiB） | (a) **推荐**：移到 `~/`；`spatial_training_room` 的可编辑安装需要重新 `pip install -e`。(b) 原位不动，写进 `.gitignore`（已忽略）并在 CLUSTER.md 说明。 |
| D9 | StarVLA 的 NavigateKitchen 检查点 100 GiB（起点 Qwen3-VL-4B，只能验接口） | (a) **推荐**：保留到第三阶段下游冒烟用本项目权重重跑一次后再删。(b) 现在删。 |
| D10 | 历史主线权重 v46@200、v50@150（62 GB）与 8 月 clean 系列 HF 权重（约 200 GB） | (a) **推荐**：登记进 `models/historical/`，实验矩阵不用它们。(b) 删除：MASTER_RESULTS 的成绩就无法复现。 |
| D11 | "自有 QA"指哪一套 | (a) **推荐**：`interiorgs_qa`（仿 8 个 benchmark 的 51 个子任务，训练和测评都用它），`active_spatial_qa` 配对 QA 归档。(b) 两套都算：配对 QA 作为"同一判据"的诊断集保留在 `data_gen/paired_qa/`。(c) 只要配对 QA。 |
| D12 | 上游文档站（mkdocs / readthedocs）和 README 里的上游内容 | (a) **推荐**：删除文档站配置，README 重写、只保留出处说明。(b) 保留文档站并改成本项目内容：多维护一套导航。 |
| D13 | 数据、权重、结果的固定位置 | (a) **推荐**：仓库目录下 `data/releases/`、`models/`、`results/`（不入 git，清单入 git）。(b) 放到 `~/VAGEN-data/` 之类的仓库外目录：sco 任务挂载不变，但路径配置多一层。 |
| D14 | 第二阶段的开工时机 | (a) **推荐**：现在就做不与训练链冲突的部分（打标签、集群归档、文档打包、金标准准备），代码搬动等第十轮结束。(b) 全部等第十轮结束。 |
| D15 | 另一会话的两块 QA 代码（`interiorgs_qa` 和 `active_spatial_qa` 未跟踪脚本）由谁搬 | (a) **推荐**：它们完工前原位不动，第二阶段只在 `configs/paths.yaml` 里给它们留位置；完工后由我按本方案搬。(b) 现在就搬，需要和那个会话协调 import。 |

## 9. 本阶段完成了什么

- 对 42,084 个被跟踪文件、1,096 个代码类文件、308 个实验目录、103 份文档做了盘点，附录三张表可逐条查。
- 给出目录方案、删除清单、归档清单、7 步缺口和工作量估计。
- 查清了四个专项（EASI、OOD、QA 上 RL、StarVLA / RLinf）各自有什么、缺什么。
- 记录了不能碰的三条线和集群上正在运行的任务。
- 没有移动、归档、删除任何东西；没有打标签（打标签属于第二阶段第一步）。

附录：[phase1/code_files.tsv](phase1/code_files.tsv)（1,096 行：路径、步骤、最后提交、提交次数、代码引用数、文档引用数、引用者、建议去向、理由）、[phase1/exps_inventory.tsv](phase1/exps_inventory.tsv)（308 行：路径、大小、日期、属主、检查点数、HF 权重数、类别、建议去向）、[phase1/md_docs.tsv](phase1/md_docs.tsv)（103 行：路径、首次 / 最后提交、提交次数、大小）、[phase1/disk_usage.tsv](phase1/disk_usage.tsv)（集群顶层与 exps 的 du 原始数据）。
