# archive/ 索引

整理任务把不再被流程引用的代码、配置、文档搬到这里，**只移动、不改内容**，git 历史完整保留（可用 `git log --follow` 追溯）。每个子目录一节；表里的「最后提交」是搬动前该文件最后一次改动的日期，「取代它的」指现在该用什么。决定依据见 [docs/reports/DECISIONS_20261016.md](../docs/reports/DECISIONS_20261016.md)。

| 子目录 | 内容 | 取代它的 |
| --- | --- | --- |
| experiments_legacy/ | 2026-05 至 10 月的 PPO 实验配置（v1–v50、c1–c8、u1、d0、a1/a2/b5、clean 系列）、历史环境配置 env_config_*.yaml、旧 launcher 与 sco 入口、GPU 占位脚本、v18/v19 oracle 回放结果 | training/（v10 训练链）、experiments/registry.yaml |
| r1/ | 2026-08 至 10 月 R1 修复战役的 133 个一次性脚本、R1 专用 PPO 适配器、R1 评测入口、exps 里被跟踪的 R1 报告与哈希清单 | data_gen/interiorgs_v10 与 evaluation/ |
| diagnostics_d0_u1/ | D0（PPO logprob 链路）与 U1（SenseNova）诊断探针 | 无（问题已在 8 月修复） |
| diagnostics/ | 离线 PPO 重放、渲染微基准、真实 batch 门禁等一次性诊断 | 无 |
| dense_score_pilot/ | dense-score S0/S1/S5 pilot 的门禁与分析脚本、适配器、测试 | 结论见 docs/reports/DENSE_SCORE_PILOT_R3_CONCLUSION_20261016.md |
| analysis/ | 画图与训练曲线分析脚本 | experiments/aggregate.py（第四阶段） |
| upstream/ | 上游 VAGEN-Lite 的 sokoban / frozenlake / navigation / primitive_skill / spatial_gym 环境、vagen/inference、示例与文档 | 上游仓库 mll-lab-nu/VAGEN |
| legacy_text_action/ | LLaMA-Factory 把 12 维动作写成文本的 RoboCasa 路线 | downstream/（StarVLA 连续动作头） |
| history/docs、history/diagnosis、history/reports | 历史交接文档、诊断报告、9 月组会报告（含两份标明为 plus10pp 的派生情景，不是实测） | docs/EXPERIMENTS.md 历史节 |
| tests/ | 只对归档代码有意义的测试 | tests/ |
| side_models/ | Cambrian-S、SenseNova-U1、Qwen3-VL 三条旁支模型的文档和 Qwen3-VL 环境检查脚本（决定 D4 选 a）。适配代码 `vagen/models/` 还在原位：核心训练循环里有按模型类型才走到的分支会导入它，要连同这些分支一起移走，放在第十一轮结束后的代码搬动批次。`.venv-qwen3vl` 留在集群磁盘上，4 个 Cambrian 历史权重留在 `models/historical/`（D10） | 主线只用 Qwen2.5-VL，正式实验用 7B |
| cluster/（只在集群，不入 git） | 实验残留、旧评测结果、日志，见 cluster/MANIFEST_moves_20261016.tsv | — |

**2026-10-26 更新（磁盘线）：** `archive/cluster/` 下的 `interiorgs_v10_rounds`、`dense_score_pilot`、`r1`、`logs`、`data_intermediate`、`diagnostics_d0_u1` 六个子目录已上传 AOSS 后从集群删除，按用户 10-26 的决定 1 选 A。AOSS 路径是 `fj:s3://baiqiao/vagen_lite_archive/` 加上它们在 `/mnt/umm/users/yinbaiqiao` 下的相对路径，记录在 `archive/cluster/DELETED_TO_AOSS_20261026.txt` 和 `/mnt/umm/users/yinbaiqiao/disk_runs/batches/b1_20261026/`。兼容链接指向的 `clean_runs_aug`、`results_legacy`、`experiments_legacy_runs` 还在集群上。决定 D7b（dense-score pilot r3）和 D9（StarVLA NavigateKitchen 检查点）也由磁盘线先传 AOSS 再删；NavigateKitchen 示例脚本指向的目录现在是空的，要跑先从 AOSS 恢复。

下面这段写于 10-16，其中前四个类目的压缩包已随上述目录一起移到 AOSS。

集群 `archive/cluster/` 里小文件多的类目（logs、results_legacy、data_intermediate、experiments_legacy_runs、r1、diagnostics_d0_u1、misc）已按条目压成 `<条目>.tar.gz`，原目录在压缩成功后删除。122 个条目从 41.6 GB 压到 19.8 GB，每条的压缩前后大小和 SHA-256 见 `cluster/MANIFEST_compressed_20261016.tsv`。还原一条：`tar -xzf <条目>.tar.gz -C <所在目录>`。存权重的大目录（clean_runs_aug、dense_score_pilot、interiorgs_v10_rounds、_pending_delete）没有压缩：内容主要是浮点张量，压不了多少还很费时。

## 逐文件清单（第一批，2026-10-16）

| 原路径 | 现位置 | 最后提交 | 说明 |
| --- | --- | --- | --- |
| data_gen/robocasa_sft/convert_lerobot_to_sft.py | archive/legacy_text_action/ | 2026-10-06 | LLaMA-Factory 文本 12 维路线 |
| data_gen/robocasa_sft/lf_dataset_info.json | archive/legacy_text_action/ | 2026-10-06 | LLaMA-Factory 文本 12 维路线 |
| data_gen/robocasa_sft/lf_qwen25vl_sft.yaml | archive/legacy_text_action/ | 2026-10-06 | LLaMA-Factory 文本 12 维路线 |
| data_gen/robocasa_sft/run_sft.sh | archive/legacy_text_action/ | 2026-10-06 | LLaMA-Factory 文本 12 维路线 |
| examples/evaluate/active_spatial/r1_canonical_dev_eval32_floor_sco_entry.sh | archive/r1/eval_entries/ | 2026-09-16 | 旧评测矩阵套件 |
| examples/evaluate/active_spatial/r1_canonical_dev_eval32_sco_entry.sh | archive/r1/eval_entries/ | 2026-09-16 | 旧评测矩阵套件 |
| examples/evaluate/active_spatial/r1_independent_local_action_eval60_sco_entry.sh | archive/r1/eval_entries/ | 2026-09-16 | 旧评测矩阵套件 |
| examples/evaluate/active_spatial/run_v48_v50_direct_8gpu.sh | archive/r1/eval_entries/ | 2026-08-31 | 旧评测矩阵套件 |
| examples/train/active_spatial/check_h800_status.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/dense_score_ablation_candidates.yaml | archive/experiments_legacy/launchers/ | 2026-10-13 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/dense_score_ablation_r1_20261006.yaml | archive/experiments_legacy/launchers/ | 2026-10-13 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/env_config.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_5action_256.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_5action_suc50.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_a1_spatial_mcq.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_a2_v46_arrival_stop.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_b5_c8_actionvalid.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_balanced.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_diagnostic_r1_real_batch.yaml | archive/experiments_legacy/env_configs/ | 2026-10-13 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_h800_7b_6types.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_h800_7b_6types_v50_format001.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_h800_7b_v26_100scenes.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_remote_render.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_sapave.yaml | archive/experiments_legacy/env_configs/ | 2026-10-06 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_sapave_val.yaml | archive/experiments_legacy/env_configs/ | 2026-10-06 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_single_action.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_single_action_256.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v17.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v17_action_only.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v17_easy.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v17_reward_only.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_absolute.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_andgate.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_dual.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_dual2.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_potential.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_potential2.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_sigmoid.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_sigmoid2.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v18_sigmoid_dual.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_dual.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_no_autoterm.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_no_farm.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_no_farm_dual.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_succ10.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_thr70_andgate.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_thr85_andgate.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v19_thr85_no_autoterm.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v2.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v22_3scenes.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_c7.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_fwdfirst.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_mnt.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_server.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_smoke_server.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_lm.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_no_think.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v24_100scenes_rewscale.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v39_6types_distinfo.yaml | archive/experiments_legacy/env_configs/ | 2026-07-16 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_v46_baseline_6types.yaml | archive/experiments_legacy/env_configs/ | 2026-08-31 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/env_config_with_spatial_prior.yaml | archive/experiments_legacy/env_configs/ | 2026-06-14 | 历史环境配置；v10 用 datagen 的 rl_env_v10_*.yaml |
| examples/train/active_spatial/experiments/a1_v46_spatial_aux_w005.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/a1_v46_spatial_aux_w010.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/a2_v46_arrival_stop.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/b5_c8_wrapper_img25_actionvalid.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c1_groupadv_100scenes.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c2_nfp_groupadv.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c3_lr5e7_cambrian.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c4_fwdfirst.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c5_entropy_hi.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c6_no_think.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c7_fwdfirst_ehi.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c8_fwdfirst_rewscale.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_lfp_debug_10_119_30_148.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_lfp_server.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_lfp_smoke_server.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_lfp_smoke_server_2gpu.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_smoke_server.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/cambrian_active_spatial_clean_v1.sh | archive/experiments_legacy/experiments/ | 2026-09-05 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/cambrian_c4_clean_v1.sh | archive/experiments_legacy/experiments/ | 2026-09-05 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_12_b5_logprob_plumbing_smoke.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_16_b5_decoupled_rollout_correction_smoke.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_17_cambrian_clean_acceptance_smoke.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_17_qwen_clean_acceptance_smoke.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_17_qwen_v46_logprob_audit.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_3_b5_exact_fsdp_smoke.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_4_b5_logprob_localization.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_5_b5_vision_select_logprob.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/d0_6_b5_forced_scoring_matrix.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/dense_score_reward_only_pilot.sh | archive/experiments_legacy/experiments/ | 2026-10-06 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/diagnostic_r1_real_batch_no_update.sh | archive/experiments_legacy/experiments/ | 2026-10-06 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/qwen3vl_active_spatial_smoke.sh | archive/experiments_legacy/experiments/ | 2026-10-06 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/qwen_active_spatial_clean_v1.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/qwen_v47_clean_v1.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/qwen_v48_clean_v1.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/qwen_v48_clean_v1_resume.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/qwen_v50_clean_v1.sh | archive/experiments_legacy/experiments/ | 2026-09-05 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_50step.sh | archive/experiments_legacy/experiments/ | 2026-10-10 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_short.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_short_launch.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke_launch.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke_remote_launch.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_smoke.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_protocol_only_prepost_32.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/u1_protocol_only_prepost_32_launch.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v10_per_action_dense.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v11_max_turns50.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v11_v3.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v12_entropy_fix.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v12_n2_max_turns50.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v12_n2_max_turns50_v3.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v13_gamma99_window3.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v13_gamma99_window3_v3.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v14_n4_gamma99.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v14_n4_gamma99_v3.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v15_per5action.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v15_per5action_v3_test.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v16_update15.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v17.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v17_action_only.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v17_easy.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v17_n4.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v17_resp512.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v17_reward_only.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_absolute.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_andgate.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_dual.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_dual2.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_potential.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_potential2.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_sigmoid.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_sigmoid2.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v18_sigmoid_dual.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_dual.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_no_autoterm.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_no_farm.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_no_farm_dual.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_succ10.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_thr70_andgate.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_thr70_klhi.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_thr85_andgate.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v19_thr85_no_autoterm.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v1_baseline.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v20_sft_no_autoterm.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v20_sft_thr85_andgate.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v20_winner.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v21_groupadv.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v21_lrdecay.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v21_no_farm_only_stab.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v22_3scenes.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v22_cosine_no_farm.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v22_kl_ablation.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v23_groupadv_3scenes.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v23_groupadv_3scenes_7types.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v23_groupadv_no_farm.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v24_groupadv_100scenes.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v25_groupadv_100scenes_cosine.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v25_groupadv_100scenes_klhi.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v26_klhi_lr5e7.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v26_klhi_lrdecay.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v27_from_v25ckpt.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v28_klhi_lr5e7_cosine.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v29_grpo.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v29_grpo_resume.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v2_entropy_fix.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v30_rew_scale.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v30_rew_scale_resume.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v31_grpo_rewscale.sh | archive/experiments_legacy/experiments/ | 2026-07-08 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v3_bilevel_gae.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v42_nodelta_w1.sh | archive/experiments_legacy/experiments/ | 2026-07-16 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v42_nodelta_w3.sh | archive/experiments_legacy/experiments/ | 2026-07-16 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v43_nodelta_w3_lr3e7.sh | archive/experiments_legacy/experiments/ | 2026-07-16 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v44_nodelta_w3_fastcosine.sh | archive/experiments_legacy/experiments/ | 2026-07-16 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v45_nodelta_w3_kl40.sh | archive/experiments_legacy/experiments/ | 2026-07-16 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v46_7b_nodelta_w3.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v46_7b_nodelta_w3_rerun.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v46_baseline_qwen25vl_3b.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v46_baseline_qwen25vl_3b_resume150.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v46_baseline_qwen25vl_7b.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v47_7b_nodelta_w3_kl40.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v48_7b_nodelta_w1.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v48_7b_nodelta_w1_resume_skipval.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v49_7b_v26_repro.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v4_stronger_critic.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v50_7b_w3_format001_stable.sh | archive/experiments_legacy/experiments/ | 2026-08-31 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v5_entropy_balanced.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v6_antientropy_collapse.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v7_action_diversity.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v8_sampling_coherence.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/experiments/v9_stable_diversity.sh | archive/experiments_legacy/experiments/ | 2026-06-14 | v1–v50/c*/u1/d0 历史实验配置 |
| examples/train/active_spatial/gpu_holder.py | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/gpu_warmer.py | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/gpu_warmer_adaptive.py | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/gpu_warmer_streams.py | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/install_render_deps.sh | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_a1_a2_b5.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_d0pass_clean_anchors.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_h800_experiments.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_node_155.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_node_185.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_node_237.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_v12_8gpu_local.sh | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/launch_v46_baselines.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/parse_val_ood.py | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/run_experiment.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/run_warmer.sh | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/run_warmer_qwen3vl8b.sh | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_active_spatial_eval_entry.sh | archive/experiments_legacy/launchers/ | 2026-09-05 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_active_spatial_renderer_microbenchmark.sh | archive/experiments_legacy/launchers/ | 2026-10-10 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_cambrian_8gpu_smoke_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_cambrian_c4_clean_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_cambrian_clean_anchor_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_cambrian_id_sharded_entry.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_clean_anchor_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_dense_score_reward_only_parallel.sh | archive/experiments_legacy/launchers/ | 2026-10-10 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_dense_score_reward_only_pilot.sh | archive/experiments_legacy/launchers/ | 2026-10-10 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_diagnostic_r1_real_batch_entry.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_qwen3vl_smoke_entry.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_qwen_8gpu_smoke_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_qwen_clean_anchor_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_qwen_v47_clean_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_qwen_v48_clean_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_qwen_v48_resume_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_qwen_v50_clean_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_clean_projective_entry.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_clean_training_expansion_entry.sh | archive/experiments_legacy/launchers/ | 2026-09-18 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_fresh_canonical_expansion_entry.sh | archive/experiments_legacy/launchers/ | 2026-09-28 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_gate_aligned_eval.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_gate_aligned_pilot.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_old_v46_target_salvage_entry.sh | archive/experiments_legacy/launchers/ | 2026-09-25 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_pair_universe_phase_b_entry.sh | archive/experiments_legacy/launchers/ | 2026-09-28 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_projection_frontier_ab_entry.sh | archive/experiments_legacy/launchers/ | 2026-09-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_sft_generation.sh | archive/experiments_legacy/launchers/ | 2026-10-10 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_task_context_acceptance.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_task_context_eval.sh | archive/experiments_legacy/launchers/ | 2026-10-06 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r1_v3_holdout_reverse_diag_entry.sh | archive/experiments_legacy/launchers/ | 2026-09-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_r5_production_backward.sh | archive/experiments_legacy/launchers/ | 2026-10-10 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_renderer_8gpu_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/sco_u1_protocol_only_prepost_entry.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/submit_8gpu_clean_smokes_sco.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/submit_clean_anchors_sco.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/submit_clean_checkpoint_evals_sco.sh | archive/experiments_legacy/launchers/ | 2026-09-05 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/submit_renderer_8gpu_sco.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/submit_u1_protocol_only_prepost_sco.sh | archive/experiments_legacy/launchers/ | 2026-08-31 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/train_active_spatial.yaml | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/train_active_spatial_local.yaml | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/train_local_8gpu.sh | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/train_ppo_qwen25vl7b.sh | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/val_active_spatial.yaml | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/active_spatial/val_active_spatial_local.yaml | archive/experiments_legacy/launchers/ | 2026-06-14 | 旧 launcher / sco 入口 / GPU 占位 |
| examples/train/robocasa/lf_sft_smoke.yaml | archive/legacy_text_action/ | 2026-10-06 | LLaMA-Factory 文本 12 维路线 |
| examples/train/robocasa/run_sft.sh | archive/legacy_text_action/ | 2026-10-06 | LLaMA-Factory 文本 12 维路线 |
| examples/train/robocasa/sco_sft_smoke_entry.sh | archive/legacy_text_action/ | 2026-10-06 | LLaMA-Factory 文本 12 维路线 |
| scripts/active_spatial_dense_score_final_preflight.py | archive/dense_score_pilot/ | 2026-10-10 | dense-score pilot 门禁脚本 |
| scripts/active_spatial_dense_score_launch_gate.py | archive/dense_score_pilot/ | 2026-10-06 | dense-score pilot 门禁脚本 |
| scripts/active_spatial_dense_score_parallel_preflight.py | archive/dense_score_pilot/ | 2026-10-10 | dense-score pilot 门禁脚本 |
| scripts/active_spatial_dense_score_preflight.py | archive/dense_score_pilot/ | 2026-10-13 | dense-score pilot 门禁脚本 |
| scripts/active_spatial_diagnostic_real_batch_gate.py | archive/diagnostics/ | 2026-10-06 | 一次性诊断 |
| scripts/active_spatial_offline_ppo_replay.py | archive/diagnostics/ | 2026-10-06 | 一次性诊断 |
| scripts/active_spatial_planner_audit.py | archive/diagnostics/ | 2026-07-17 | 一次性诊断 |
| scripts/active_spatial_r1_rgb_runtime_preflight.py | archive/diagnostics/ | 2026-10-06 | 一次性诊断 |
| scripts/active_spatial_r5_production_backward.py | archive/diagnostics/ | 2026-10-10 | 一次性诊断 |
| scripts/active_spatial_r5_replay_forensics.py | archive/diagnostics/ | 2026-10-06 | 一次性诊断 |
| scripts/active_spatial_real_batch_preflight.py | archive/diagnostics/ | 2026-10-06 | 一次性诊断 |
| scripts/active_spatial_renderer_microbenchmark.py | archive/diagnostics/ | 2026-10-10 | 一次性诊断 |
| scripts/analyze_active_spatial_dense_score_validation.py | archive/dense_score_pilot/ | 2026-10-10 | dense-score pilot 门禁脚本 |
| scripts/analyze_active_spatial_training.py | archive/analysis/ | 2026-10-06 | 画图/分析脚本，无引用 |
| scripts/build_baseline_trained_analysis.py | archive/analysis/ | 2026-08-31 | 画图/分析脚本，无引用 |
| scripts/compare_active_spatial_renderer_microbenchmarks.py | archive/diagnostics/ | 2026-10-10 | 一次性诊断 |
| scripts/d0_10_compare_attention.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_10_hf_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_10_vllm_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_11_abc_attention_matrix.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_11_hf_full_attention_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_11_precision_reference.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_6_hf_incremental_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_6_legacy_layer_ablation.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_6_vllm_forced_scoring.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_7b_fixed_vllm_replay.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_8_fixed_vllm_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_8_hf_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_9_compare_layers.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_9_hf_layer_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/d0_9_vllm_layer_probe.py | archive/diagnostics_d0_u1/scripts/ | 2026-08-31 | D0 / U1 诊断探针 |
| scripts/prepare_active_spatial_dense_score_pilot.py | archive/dense_score_pilot/ | 2026-10-10 | dense-score pilot 门禁脚本 |
| scripts/prepare_cambrian_id_shards.py | archive/diagnostics/ | 2026-10-06 | 一次性诊断 |
| scripts/r1_action_graph.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_aoss_scene_pipeline_tests.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_asset_coverage_audit.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_audit_clean_training_inventory.py | archive/r1/scripts/ | 2026-09-18 | R1 修复战役脚本（9 月） |
| scripts/r1_audit_full_clean_target_inventory.py | archive/r1/scripts/ | 2026-09-28 | R1 修复战役脚本（9 月） |
| scripts/r1_audit_projective_small_sample.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_build_fov_blinded_review_site.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_build_sha256_inventory.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_camera_utility_tests.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_canary_difficulty_audit.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_canonical_dev_eval32_floor_tests.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_canonical_dev_eval32_tests.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_canonical_tasks.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_clean_projective_runtime_preflight.py | archive/r1/scripts/ | 2026-10-13 | R1 修复战役脚本（9 月） |
| scripts/r1_clean_projective_smoke_gate.py | archive/r1/scripts/ | 2026-10-01 | R1 修复战役脚本（9 月） |
| scripts/r1_collect_historical_projective_positives.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_collision_ambiguity_resolver.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_collision_convention_audit.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_compare_gate_aligned_pilot.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_compare_task_context_eval.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_donor_allocator.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_donor_allocator_tests.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_donor_impact_report.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_enumerate_train_pair_universe.py | archive/r1/scripts/ | 2026-09-28 | R1 修复战役脚本（9 月） |
| scripts/r1_extend_verified_medium_inventory.py | archive/r1/scripts/ | 2026-09-14 | R1 修复战役脚本（9 月） |
| scripts/r1_failure_taxonomy.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_finalize_clean_training_corpus.py | archive/r1/scripts/ | 2026-09-18 | R1 修复战役脚本（9 月） |
| scripts/r1_finalize_observability_canary.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_fov_h1_audit.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_fov_local_perturbation_audit.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_fov_observability.py | archive/r1/scripts/ | 2026-09-19 | R1 修复战役脚本（9 月） |
| scripts/r1_fov_observability_audit.py | archive/r1/scripts/ | 2026-09-18 | R1 修复战役脚本（9 月） |
| scripts/r1_fov_observability_calibrate.py | archive/r1/scripts/ | 2026-09-19 | R1 修复战役脚本（9 月） |
| scripts/r1_fov_observability_tests.py | archive/r1/scripts/ | 2026-09-19 | R1 修复战役脚本（9 月） |
| scripts/r1_fov_observability_v2_audit.py | archive/r1/scripts/ | 2026-09-19 | R1 修复战役脚本（9 月） |
| scripts/r1_freeze_canonical_dev_eval32.py | archive/r1/scripts/ | 2026-09-15 | R1 修复战役脚本（9 月） |
| scripts/r1_freeze_clean_projective_v0.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_freeze_independent_local_action_scope.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_freeze_independent_v3_holdout.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_freeze_projective_canary10.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_freeze_projective_selection.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_freeze_verified_medium_inventory.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_full_regeneration.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_gate_aligned_reward_replay.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_independent_local_action_tests.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_make_fov_h0_h1_overlay_comparison.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_materialize_difficulty_conditioned.py | archive/r1/scripts/ | 2026-09-08 | R1 修复战役脚本（9 月） |
| scripts/r1_materialize_path_first_certificates.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_materialize_required_manifest_fields.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_canary_shards.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_fresh_canonical_expansion_shards.py | archive/r1/scripts/ | 2026-09-28 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_index_shards.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_medium_selector_retries.py | archive/r1/scripts/ | 2026-09-10 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_observability_retry_index_shards.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_observability_retry_shards.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_observability_shards.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_old_v46_target_salvage_shards.py | archive/r1/scripts/ | 2026-09-19 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_projective_path_first_chunks.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_merge_projective_small_sample.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_package_clean_projective_v0.py | archive/r1/scripts/ | 2026-10-01 | R1 修复战役脚本（9 月） |
| scripts/r1_package_gate_aligned_eval.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_package_gate_aligned_eval_v2.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_package_gate_aligned_pilot.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_package_task_context_eval.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_parallel_observability_retry.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_parallel_observability_retry_index_shards.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_canonical_dev_eval32_floor_diagnostics.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_clean_training_expansion.py | archive/r1/scripts/ | 2026-09-18 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_fov_manual_review_pack.py | archive/r1/scripts/ | 2026-09-28 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_fullscale_canonical_expansion.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_independent_local_action_eval.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_old_v46_target_salvage_scope.py | archive/r1/scripts/ | 2026-09-19 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_phase_a_projective_schema_retry.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_prepare_phase_b_from_pair_universe.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_action_graph_audit.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_canary10_aggregate.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_convention_audit.py | archive/r1/scripts/ | 2026-08-31 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_difficulty_conditioned.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_difficulty_donor_audit.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_failure_analysis.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_funnel_report.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_medium_selector_v2.py | archive/r1/scripts/ | 2026-09-11 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_medium_selector_v3_depth_banded.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_medium_selector_v4_projection_frontier.py | archive/r1/scripts/ | 2026-09-14 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_observability.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_observability_audit.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_observability_keyframe_recheck.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_observability_retry.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_path_first_gate_aggregate.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_path_first_prototype.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_regeneration_audit.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_repair_pilot_h1.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_request_schema.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_request_schema_tests.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_same_source71_comparison.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_projective_sources.json | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_publish_sft_pages.py | archive/r1/scripts/ | 2026-10-10 | R1 修复战役脚本（9 月） |
| scripts/r1_reachability_audit.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_real_render_validation.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_reconcile_donor_ledger.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_reconcile_expansion_inventory.py | archive/r1/scripts/ | 2026-09-30 | R1 修复战役脚本（9 月） |
| scripts/r1_renderer_scene_smoke.py | archive/r1/scripts/ | 2026-09-19 | R1 修复战役脚本（9 月） |
| scripts/r1_repair_gate_aligned_packages.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_repair_pilot_audit.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_repair_pipeline.py | archive/r1/scripts/ | 2026-09-07 | R1 修复战役脚本（9 月） |
| scripts/r1_repair_pipeline_tests.py | archive/r1/scripts/ | 2026-09-05 | R1 修复战役脚本（9 月） |
| scripts/r1_replay_difficulty_candidates.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_replay_fov_certificates.py | archive/r1/scripts/ | 2026-09-18 | R1 修复战役脚本（9 月） |
| scripts/r1_replay_projective_certificates.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_reverse_cap_structure_diagnostic.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_run_canonical_dev_eval32.py | archive/r1/scripts/ | 2026-10-13 | R1 修复战役脚本（9 月） |
| scripts/r1_run_canonical_dev_eval32_floor_diagnostics.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_run_independent_local_action_eval.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_run_medium_selector_scene_shards.py | archive/r1/scripts/ | 2026-09-14 | R1 修复战役脚本（9 月） |
| scripts/r1_run_projective_small_sample.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_runtime_consistency.py | archive/r1/scripts/ | 2026-10-13 | R1 修复战役脚本（9 月） |
| scripts/r1_runtime_render_camera_tests.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_select_holdout_v3_ab.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_select_projective_diagnostics.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_select_projective_small_sample.py | archive/r1/scripts/ | 2026-09-06 | R1 修复战役脚本（9 月） |
| scripts/r1_stage_aoss_scene_metadata.py | archive/r1/scripts/ | 2026-09-29 | R1 修复战役脚本（9 月） |
| scripts/r1_stage_aoss_scene_metadata_tests.py | archive/r1/scripts/ | 2026-09-29 | R1 修复战役脚本（9 月） |
| scripts/r1_submit_clean_projective_after_preflight.py | archive/r1/scripts/ | 2026-10-01 | R1 修复战役脚本（9 月） |
| scripts/r1_submit_gate_aligned_eval.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_submit_gate_aligned_pilot.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_submit_sft_generation.py | archive/r1/scripts/ | 2026-10-10 | R1 修复战役脚本（9 月） |
| scripts/r1_summarize_canonical_dev_eval32.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_summarize_canonical_dev_eval32_floor_diagnostics.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_summarize_independent_local_action_eval.py | archive/r1/scripts/ | 2026-09-16 | R1 修复战役脚本（9 月） |
| scripts/r1_summarize_projection_frontier_ab.py | archive/r1/scripts/ | 2026-09-14 | R1 修复战役脚本（9 月） |
| scripts/r1_summarize_v3_holdout_reverse_diag.py | archive/r1/scripts/ | 2026-09-13 | R1 修复战役脚本（9 月） |
| scripts/r1_task_context_acceptance.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/r1_vllm_eval_preflight.py | archive/r1/scripts/ | 2026-10-06 | R1 修复战役脚本（9 月） |
| scripts/visualize_spatial_potential_field.py | archive/analysis/ | 2026-10-06 | 画图/分析脚本，无引用 |
| scripts/visualize_trajectories.py | archive/analysis/ | 2026-07-17 | 画图/分析脚本，无引用 |
| tests/active_spatial/test_dense_score_preflight.py | archive/tests/ | 2026-10-10 | 只对 R1/pilot/旧评测有意义 |
| tests/active_spatial/test_dense_score_validation_analysis.py | archive/tests/ | 2026-10-10 | 只对 R1/pilot/旧评测有意义 |
| tests/active_spatial/test_r1_legacy_configs.py | archive/tests/ | 2026-10-13 | 只对 R1/pilot/旧评测有意义 |
| tests/active_spatial/test_r1_sft_pipeline.py | archive/tests/ | 2026-10-10 | 只对 R1/pilot/旧评测有意义 |
| tests/active_spatial/test_r5_production_backward.py | archive/tests/ | 2026-10-10 | 只对 R1/pilot/旧评测有意义 |
| tests/active_spatial/test_renderer_microbenchmark.py | archive/tests/ | 2026-10-10 | 只对 R1/pilot/旧评测有意义 |
| tests/test_r1_clean_projective_v0.py | archive/tests/ | 2026-09-30 | 只对 R1/pilot/旧评测有意义 |
| tests/test_r1_projective_projection_frontier.py | archive/tests/ | 2026-09-14 | 只对 R1/pilot/旧评测有意义 |
| tools/d0_2_cambrian_parity_probe.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/d0_2_vllm_runtime_probe.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/d0_5_summarize_logprob_artifacts.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/debug_u1_training_batch.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/oracle_replay.py | archive/diagnostics_d0_u1/tools/ | 2026-06-14 | D0 / U1 诊断探针 |
| tools/run_u1_protocol_milestone_eval.sh | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/test_fsdp_mixed_grad_clip.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_actor_one_batch_overfit.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_critic_one_batch_overfit.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_critic_phase2b.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_fm_backprop_ab_gate.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_fm_fixed_overfit_gate.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_fm_forward_only_gate.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_fm_fsdp_empty_storage_gate.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_fm_phase3.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_fm_save_reload_gate.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_phase3_data_gate_http.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_protocol_live_gate.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_protocol_prepost_checkpoint_audit.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_protocol_prepost_compare.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_protocol_prepost_eval.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| tools/u1_protocol_prepost_manifest.py | archive/diagnostics_d0_u1/tools/ | 2026-08-31 | D0 / U1 诊断探针 |
| vagen/dense_score_reward_only_pilot.py | archive/dense_score_pilot/ | 2026-10-13 | dense-score pilot 适配器 |
| vagen/r1_clean_projective_ppo.py | archive/r1/ | 2026-10-13 | R1 专用 PPO 适配器 |
| docs/{v19_v20_progress,v26_v27_analysis,active_spatial_new_server_handoff,active_spatial_task_reward_audit,active_spatial_eval_system}.md | history/docs/ | 2026-05 至 10-06 | 历史实验记录与交接 |
| docs/diagnosis/（22 份 md） | history/diagnosis/ | 2026-08 至 10-10 | D0 审计、历史有效性、R1 交接、dense-score 报告、10-02 接管审计 |
| reports/（4 组） | history/reports/ | 2026-10-06 | 组会报告与图表 |
| exps/vagen_active_spatial/{R1-clean-Projective-v0,r1_h1_aoss_repair_20260905}（被跟踪的 57 个文件） | r1/exps_artifacts/ | 2026-09 至 10-01 | R1 报告、哈希清单、启动脚本 |
| vagen/envs/{sokoban,frozenlake,navigation,primitive_skill,spatial_gym}、vagen/inference、examples/{train,evaluate}/{同名} 、docs/{index,quickstart,configuration,custom-*,issues}.md | upstream/ | 2026-02 至 06 | 上游 VAGEN |
| mkdocs.yml、.readthedocs.yaml、docs/requirements.txt | 已删除（决定 D12） | 2026-10-10 | 上游文档站配置 |

## 逐文件清单（第二批，2026-10-16，决定 D4 的文档部分）

| 原路径 | 现位置 | 最后提交 | 说明 |
| --- | --- | --- | --- |
| docs/active_spatial_qwen3vl_migration.md | archive/side_models/docs/ | 2026-10-06 | Qwen3-VL 迁移说明 |
| docs/model_backbone_integration.md | archive/side_models/docs/ | 2026-08-31 | Qwen / Cambrian-S 接入说明 |
| docs/cambrian_iteration.md | archive/side_models/docs/ | 2026-06-14 | Cambrian-S 实验记录 |
| docs/cambrian_s_new_server_smoke_handoff.md | archive/side_models/docs/ | 2026-08-31 | Cambrian-S 新机器 smoke 交接 |
| docs/sensenova_u1_active_spatial.md | archive/side_models/docs/ | 2026-10-10 | SenseNova-U1 接入说明 |
| scripts/check_qwen3_vl_env.py | archive/side_models/scripts/ | 2026-10-06 | Qwen3-VL 环境检查，只被上面的迁移说明引用 |

README、`docs/active_spatial.md`、`docs/active_spatial_end_to_end.md` 里指向这些文档的链接已改到归档位置。

## 逐文件清单（第三批，2026-10-26，冻结期代码搬动第二批：SenseNova-U1 适配代码）

| 原路径 | 现位置 | 最后提交 | 说明 |
| --- | --- | --- | --- |
| vagen/models/sensenova_u1_plugin.py | archive/side_models/sensenova_u1/ | 2026-08-31 | vLLM 插件；原位置留一个什么都不注册的同名空壳，见下 |
| vagen/models/sensenova_u1_processor.py | archive/side_models/sensenova_u1/ | 2026-08-31 | 图像处理与占位符展开 |
| vagen/models/sensenova_u1_register.py | archive/side_models/sensenova_u1/ | 2026-08-31 | Transformers 注册 |
| vagen/models/sensenova_u1_vllm.py | archive/side_models/sensenova_u1/ | 2026-08-31 | vLLM 模型类 |
| vagen/models/u1_neo_compat.py | archive/side_models/sensenova_u1/ | 2026-08-31 | NEO 兼容层 |

同一批从保留代码里删掉了只在 SenseNova-U1 检查点上才会执行的部分：

- `vagen/agent_loop/agent_loop_no_concat.py`：`model_type == "neo_chat"` 时的处理器选择、多模态后处理里的 SenseNova-U1 分支和它专用的 `_expand_sensenova_u1_ids_and_mask`；
- `vagen/agent_loop/gym_agent_loop_no_concat.py`：SenseNova-U1 的紧凑占位符分词，原来 `else` 里的两行原样保留；
- `vagen/models/cambrian_plugin.py`：顺带注册的两个 SenseNova-U1 模型类；
- `setup.py`：`sensenova_u1` 这个 vLLM 插件入口。

**空壳的用处：** vagen-lite 环境是按旧的 `setup.py` 装的，环境里的插件登记还指向 `vagen.models.sensenova_u1_plugin`，vLLM 在每个进程里都会导入它。原位置的空壳让这次导入照常成功、什么都不注册。重装环境（`pip install -e .`）之后插件登记就没有了，那时可以删掉空壳。

Cambrian-S 和 Qwen3-VL 的适配代码没有动。决定 D4 选的是 a，但 10 月 24 日起的模型线（`models` 分支）正在用它们训练 Qwen3-VL-8B 和 Cambrian-S-7B，去留要你重新定。
