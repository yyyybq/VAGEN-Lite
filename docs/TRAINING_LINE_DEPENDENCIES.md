# InteriorGS v10 训练线：运行时依赖和可归档目录

2026-10-19（第十二轮）写，给代码整理和实验矩阵交接用。

**规则：** 第一部分列出的目录和文件，是训练链和评测任务运行时要读或要写的。整理时不能移动、改名或删除；路径写死在代码、脚本、plan 或训练链的进度文件里，移走后正在跑或以后重跑的任务会在启动时失败。第十一轮就出过一次：整理任务把 gsplat 编译结果移进 `archive/`，见 `data_gen/interiorgs_v10/cluster_runs/REPORT_R11_20261016.md` 第九节。第二部分列出的目录，本线运行时已经不用，可以归档，但报告里引用了其中的结果，归档时要保留可读的副本和清单。

以下路径都在集群 fujian 上。`E` 指 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial`。

## 一、不能移动

### 1. 代码

集群上唯一的代码目录是 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite`。`data_gen/interiorgs_v10/job_sco.py` 的 `WORKTREE` 写死了这个路径，每个 sco 任务都从这里启动；训练链开始时会核对它和 plan 里的代码提交号是否一致。

| 路径 | 用途 |
| --- | --- |
| `data_gen/interiorgs_v10/` | 训练链 `train_chain.py`、sco 任务 `job_sco.py`、SFT / RL 配置 `training_prep.py`、评测 `evaluation.py`、SFT 导出 `release.py`、回放 `replay_env.py`、失败分析 `fail_analysis.py`、浏览页 `eval_browser.py`、渲染 `pilot_sco.py`、深度核对 `depth_check.py` |
| `data_gen/interiorgs_v10/configs/` | `rl_env_v10_S0.yaml`（RL 和评测的环境规则）、`ppo_template_v10.yaml`（PPO 模板）、`rl_standards.json`（命名的标准 RL 配置）、`admission_v2.json`（SFT 导出策略）、`generation_r10_20261015.json`（出题配置）、`no_task_objects_20261014.json` |
| `vagen/` | 环境（`envs/active_spatial/`）、多回合生成（`agent_loop/`）、训练器（`ray_trainer.py`、`main_ppo.py`、`interiorgs_v10_ppo.py`）、只用有成有败的题的过滤器（`custom_filter/filter.py`）、`configs/agent.yaml` |
| `verl/` | git 子模块，`yyyybq/verl` 的 `vagen-lite` 分支，提交 `a0ae85c` |
| `scripts/prepare_active_spatial_sft.py` | 生成 LLaMA-Factory 的 SFT 数据和配方 |
| `examples/train/active_spatial/sco_interiorgs_v10_train.sh` | 每个 sco 任务的入口（SFT、RL、渲染服务、评测） |
| `examples/train/active_spatial/start_gs_render_http_service.sh`、`accelerate_fsdp_qwen25vl.yaml` | 渲染服务的启动脚本、SFT 的 accelerate 配置 |
| `third_party/LLaMA-Factory/` | SFT 用的 LLaMA-Factory。conda 环境里的 `llamafactory` 指向这里（可编辑安装） |

`third_party/RLinf/` 只有测试 `tests/active_spatial/test_training_handoff.py` 用到，本线训练不用。缺了它，那两项测试会跳过。

### 2. 运行环境

| 路径 | 用途 |
| --- | --- |
| `/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite` | Python 环境（torch、vLLM、gsplat、ray），也提供编译器和 CUDA |
| `/mnt/umm/users/yinbaiqiao/.sco/bin/sco` | 提交和管理 sco 任务 |
| `/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007` | 渲染服务的 gsplat 编译结果。每个 RL 和评测任务启动渲染服务时复制一份；路径写在 `sco_interiorgs_v10_train.sh` 里 |

### 3. 数据

| 路径 | 用途 |
| --- | --- |
| `/mnt/umm/users/yinbaiqiao/InteriorGS_ply` | 3D 高斯场景，渲染服务读取 |
| `/mnt/umm/users/yinbaiqiao/InteriorGS_meta` | 场景的墙和物体标注（`labels.json`、`structure.json`），环境的碰撞和可见判定、回放、SFT 导出都读 |
| `E/interiorgs_v10_gen_r10_20261015/` | 第十轮出的题：`output/tasks.jsonl`（正式题）、`output/tasks_reserve.jsonl`（后备题）、`split_manifest.json`、`sft/`（SFT 的帧） |
| `E/interiorgs_v10_r10_20261015/depth_check/depth_check_summary.json` | 深度核对标出的题。RL 配置和评测集都按它排除题目 |
| `E/interiorgs_v10_r11_20261016/sft/` | 第十一轮的 SFT 数据（对话、帧、`rows.jsonl`、任务登记表），也是第十一、十二轮训练链的 `sft_dir` |
| `E/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl` | RL 训练题：train 场景的后备题 |
| `E/interiorgs_v10_r11_20261016/eval_val400/`、`eval_test/`、`eval_train200/` | v1 评测集：398 道验证题、279 道评测题、198 道 SFT 训练题。里面有 `tasks.jsonl` 和各模型的结果 |
| `E/interiorgs_v10_r12_20261019/eval_val_v2/`、`eval_test_v2/`、`eval_val_v2_oa/`、`eval_test_v2_oa/` | v2 评测集（遮挡族补了后备题），以后用这一套 |

### 4. 模型权重

| 路径 | 内容 |
| --- | --- |
| `E/interiorgs_v10_r11_20261016/inputs/` | 原始 Qwen2.5-VL-7B（`base_7b`）、R1 最终模型、第十轮最终模型（`r10_sft_e5`）、3B 试跑模型：都是硬链接，原目录被移走也不受影响。第十、十一轮 plan 引用的 PPO 模板 `ppo_template_S0.yaml` 现在只剩这一份（仓库里的 `configs/ppo_template_v10.yaml` 是它加了说明的副本） |
| `E/interiorgs_v10_train_r11_20261016/lf_formal/model/` | 第十一轮 SFT 每个 epoch 的检查点。`checkpoint-1386`（第 9 个 epoch）是第十一、十二轮 RL 的起点；`checkpoint-924` 是第 6 个 epoch |
| `E/interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300` | **第十一轮的最终模型**（P-S0 第 300 次） |
| `E/interiorgs_v10_train_r11p_20261017/rl_run_*/`、`E/interiorgs_v10_train_r11_20261016/rl_run_*/` | 第十一轮其余 RL 的检查点（浏览页和对照评测用） |
| `E/interiorgs_v10_train_r12*_20261019/rl_run_*/` | 第十二轮成对 RL 的检查点 |

### 5. 训练链目录

`E/interiorgs_v10_train_r11_20261016/`、`E/interiorgs_v10_train_r11p_20261017/`、`E/interiorgs_v10_train_r12a_20261019/`、`E/interiorgs_v10_train_r12b_20261019/`、`E/interiorgs_v10_train_r12e_20261019/` 都要保留。里面的 `plan.json` 和 `progress.json` 记录了用到的代码提交号和每一步的状态；训练链从断点重跑要读它们，第十二轮的启动脚本也从第十一轮链的 `progress.json` 复制 SFT 步骤。

### 6. 分析和浏览页

报告引用的结果都在这些目录里：

- `E/interiorgs_v10_r11_20261016/` 下的 `analysis/`、`browser/`、`diag_sampling/`、`scripts/`；
- `E/interiorgs_v10_r12_20261019/` 下的 `analysis/`、`scripts/`；
- 本机的 `exps/interiorgs_v10_eval_browser_r11_20261017/`。

## 二、可以归档（本线运行时不再用）

| 路径 | 是什么 |
| --- | --- |
| `E/interiorgs_v10_train_r11_20261016/rl_attempts/` | 第十一轮 RL 的两次失败尝试：在正式题上第 43 次停掉；第 25 次信号检查没通过 |
| `E/interiorgs_v10_train_r12a_20261019/attempt0_rollout_seed/` | 第十二轮第一对的第一次启动，在模型初始化时失败（采样配置不认 seed） |
| `E/interiorgs_v10_r11_20261016/pilot_*`、`pkg_pilot_*`、`diag_runs/`、`render_x/` | 第十一轮的 3B 单卡试跑、诊断任务和渲染任务的运行目录（结论已写进报告） |
| `E/interiorgs_v10_train_r10_20261015/` | 第十轮训练。要用的第十轮最终模型已经硬链接到 `interiorgs_v10_r11_20261016/inputs/r10_sft_e5` |
| `E/interiorgs_v10_train_20261014/` | 第六到九轮训练（旧的墙、旧规则）。**注意：** QA 线的 `data_gen/interiorgs_qa/export_lf.py` 和 `configs/sampling_plan_20261015.json` 引用了其中的 `lf/train.yaml`，归档前要问 QA 线 |
| `E/interiorgs_v10_r5_20261014/`、`r7_20261014/`、`r8_20261014/`、`r9_20261015/` | 第五、七到九轮的评测和诊断（结果在各轮报告里）。3B 试跑模型已经硬链接到 `interiorgs_v10_r11_20261016/inputs/qwen25vl3b_sft_trial_r5` |
| `E/interiorgs_v10_r10_20261015/` 里除 `depth_check/` 以外的内容 | 第十轮的评测和分析；评测集已由第十一轮的 v1 和第十二轮的 v2 取代 |
| `E/interiorgs_v10_gen_r6b_20261014/`、`E/interiorgs_v10_gen_20261014/` | 第六轮和第五轮出的题（旧的墙），已由第十轮的数据取代 |

归档前请再查一次：`grep -r <目录名>`，范围是 `data_gen/`、`examples/`、`scripts/` 和上面第 5 节各训练链的 `plan.json`。确认没有任何地方还引用它，再移动，并在 `archive/` 的清单里记下原路径。
