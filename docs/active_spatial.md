# Active Spatial 项目入口

Active Spatial 研究的是：VLM 能否根据当前图像、任务文本与相机状态，主动移动到满足空间关系的视角；这种交互训练能否迁移到静态空间问答。底座是 VAGEN-Lite 的多轮 agent loop 与 VERL PPO。主要实验使用 Qwen2.5-VL；Cambrian 与 Qwen3-VL 是另外的模型路径。

当前应先收束 **R1 canonical 数据、运行中训练及固定评估集的证据链**。历史导航分数、PPO 管线通过、canonical 任务成功、静态 QA 迁移是四件不同的事。具体状态、缺陷与下一步见 [2026 年 10 月 2 日接管核验](diagnosis/active_spatial_takeover_20261002/REPORT.md)。运行状态只在报告记录的查询时刻有效。

## 五分钟阅读顺序

1. [接管结论与有限计划](diagnosis/active_spatial_takeover_20261002/REPORT.md)：先了解什么已完成、什么仍不能声称。
2. [唯一成绩账本](../exps/unified_eval_runs/MASTER_RESULTS.md)：保留旧协议成绩，并单列本轮回收的 D0 修复后结果。该路径是本机产物，不保证普通 clone 包含。
3. [历史有效性审计](diagnosis/ACTIVE_SPATIAL_HISTORICAL_VALIDITY.md)：理解 Cambrian 实现问题与共享 PPO logprob 问题；其中作业状态是 8 月快照。
4. 按下表进入要维护的代码；7 月 [服务器交接](active_spatial_new_server_handoff.md) 与 9 月初 [R1 交接](diagnosis/ACTIVE_SPATIAL_R1_HANDOFF_20260905.md) 用于追溯，不作为当前任务状态。

## 代码与入口

| 层次 | 主要入口 | 维护时要核对的约束 |
| --- | --- | --- |
| 通用训练 | [run_experiment.sh](../examples/train/active_spatial/run_experiment.sh) → [main_ppo.py](../vagen/main_ppo.py) → [ray_trainer.py](../vagen/ray_trainer.py) | shell 会物化 train/val manifest，并启动训练；不是只读检查命令。实验变量在 `examples/train/active_spatial/experiments/`。 |
| 当前 R1 专用训练 | [冻结数据与历史 recipe](../scripts/r1_freeze_clean_projective_v0.py)、[打包](../scripts/r1_package_clean_projective_v0.py)、[worker 入口](../examples/train/active_spatial/sco_r1_clean_projective_entry.sh)、[固定终点适配器](../vagen/r1_clean_projective_ppo.py) | 已冻结 210 条 Projective；scheduler horizon 700，终点 250。运行包独立于当前脏工作区。不能用通用 launcher 直接代替或重建本次运行。 |
| 数据生成与切分 | [pipeline README](../data_gen/active_spatial_pipeline/README.md)、[run_pipeline.py](../data_gen/active_spatial_pipeline/run_pipeline.py)、[splits.py](../data_gen/active_spatial_pipeline/splits.py)、[gen_ood_splits.py](../scripts/gen_ood_splits.py) | 生成候选不等于可训练；ID/OOD 要记录训练参考集、hash、内容与 source 身份。`ood_splits_v2` 只修正切分，仍是 legacy 数据。 |
| canonical 数据准入 | [dataset_contract.py](../vagen/envs/active_spatial/dataset_contract.py)、[verify_active_spatial_dataset.py](../scripts/verify_active_spatial_dataset.py) | 绑定整行内容和运行协议。验证器调用真实 renderer；本轮没有运行。新 launcher 的 sidecar 合同与旧冻结 R1 的 preflight 是不同入口。 |
| 环境与成功定义 | [env.py](../vagen/envs/active_spatial/env.py)、[env_config.py](../vagen/envs/active_spatial/env_config.py)、[canonical_camera.py](../vagen/envs/active_spatial/canonical_camera.py)、[canonical_task_metrics.py](../vagen/envs/active_spatial/canonical_task_metrics.py) | canonical H1 当前只覆盖 Projective/FOV；相机 native K 只缩放一次。canonical 成功 gate 与 dense score 不等价。 |
| 奖励、可见性与碰撞 | [spatial_potential_field.py](../vagen/envs/active_spatial/spatial_potential_field.py)、[reward_trace.py](../vagen/envs/active_spatial/reward_trace.py)、[collision_detector.py](../vagen/envs/active_spatial/collision_detector.py)、[visual_bbox_metrics.py](../vagen/envs/active_spatial/visual_bbox_metrics.py) | reward 是学习信号，不能替代成功布尔值；投影 bbox 也不能直接证明真实遮挡可见比例。 |
| 渲染 | [unified_renderer.py](../vagen/envs/active_spatial/render/unified_renderer.py)、[远程环境说明](active_spatial_remote_env.md) | `local`、HTTP 和 client 都可能用 GPU；只有读取已有 RGB 与统计属于本轮离线核验。 |
| policy 输入与轨迹 | [gym_agent_dataset.py](../vagen/gym_agent_dataset.py)、[gym_agent_loop_no_concat.py](../vagen/agent_loop/gym_agent_loop_no_concat.py)、[agent_loop_no_concat.py](../vagen/agent_loop/agent_loop_no_concat.py)、[observation_history.py](../vagen/utils/observation_history.py) | seed 显式枚举；图像与文本回合一起裁剪。当前工作区 `WINDOW_SIZE` 才显式连到 `history_window_size`；旧 W1/W3 名称不能证明实际观察历史。 |
| PPO 与离线重放 | [no_concat_gae.py](../vagen/custom_advantage/no_concat_gae.py)、[snapshot](../vagen/utils/active_spatial_ppo_snapshot.py)、[replay](../vagen/utils/active_spatial_ppo_replay.py)、[r5 forensic CLI](../scripts/active_spatial_r5_replay_forensics.py) | UUID group、数值 turn 顺序、padding 去重、whitening、HF old logprob 与 rollout logprob 都要对齐。离线重放通过不代表 reward 干预有效。 |
| Active Spatial 离线评估 | [evaluation/run_eval.py](../evaluation/run_eval.py)、[eval_runner.py](../evaluation/eval_runner.py)、[sweep](../scripts/active_spatial_eval_sweep.py)、[suite README](../examples/evaluate/active_spatial/README.md) | 当前评估读取环境成功值，继承 checkpoint 协议；suite 改协议需显式声明。`vagen/evaluate/` 是框架通用入口，不能与此处混用。 |
| 静态 QA 与迁移诊断 | [paired QA README](../data_gen/active_spatial_qa/README.md)、[contract.py](../data_gen/active_spatial_qa/contract.py)、[qa_eval.py](../data_gen/active_spatial_qa/qa_eval.py)、[easi_eval.py](../scripts/easi_eval.py) | paired QA 是同一任务谓词的局部诊断，EASI-8 是外部静态评估；两者分母与证据范围不同。invalid 答案留在分母中。 |
| SFT | [SFT README](../data_gen/active_spatial_sft/README.md)、[sft_generator.py](../data_gen/active_spatial_sft/sft_generator.py) | 当前实现会在环境中重放轨迹；搜索分数、旧导出文件与新 runtime 验证不能混为一谈。 |
| 模型适配与旁支 | [模型接入](model_backbone_integration.md)、[Qwen3-VL 迁移](active_spatial_qwen3vl_migration.md) | `data_gen/robocasa_sft`、`vagen/envs/robocasa`、`ele/`、`cjepa/`、`spatial-training-room/` 是相邻工作，暂不并入当前 canonical RL 主线。 |

`verl/` 是带本地改动的子模块；复现需要根仓库、子模块版本及实际 source archive，不能只记根仓库 commit。`exps/`、`evaluation/sweeps/`、`outputs/`、`wandb/` 是产物；`reports/*plus10pp*` 是明确标注的派生情景，不能当作实测成绩。

## 不启动 GPU 的维护入口

在仓库根目录运行：

```bash
bash scripts/check_active_spatial_cpu.sh --junitxml=/tmp/active_spatial_cpu.xml
```

脚本优先选择工作区旁已有的 `.conda/envs/vagen-lite/bin/python`，也可用 `PYTHON=/path/to/python` 指定。系统 `python3` 在本次环境没有 numpy、pytest 等依赖；本轮没有安装依赖。检查限定在列出的 CPU 测试，关闭模型网络访问并隐藏 CUDA；不 source 训练脚本，不连接 renderer。

最终复测 **70 项全部通过、无跳过**。此前发现的 QA 空标签计数失败在工作区同期修订后消失；本轮恢复了重复命名所覆盖的测试，并重新验证。各阶段结果见 [CPU 记录](diagnosis/active_spatial_takeover_20261002/cpu_checks.json)。CPU 全绿不代表真实 renderer、模型长期稳定性或 QA 数据语义已经通过。

## 结果应当保存在哪里

导航 ID/OOD 与 EASI 结果回填 [MASTER_RESULTS.md](../exps/unified_eval_runs/MASTER_RESULTS.md)，保留明确的协议与分母；EASI 未齐八项不填 Macro。接管审计、缺陷复现和本轮检查证据保存在 [本次审计目录](diagnosis/active_spatial_takeover_20261002/REPORT.md)。未来实验继续使用独立版本目录，不覆盖冻结输入或历史产物。
