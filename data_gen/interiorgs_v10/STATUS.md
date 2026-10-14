# InteriorGS v10 任务生成器：状态

这是 v10 生成器唯一的状态文档。每轮工作结束时更新。最后更新：2026-10-14（第六轮）。

## 1. 现在可以用的数据

**用这一份：** `exps/vagen_active_spatial/interiorgs_v10_gen_r6b_20261014/`（集群 fujian，仓库目录下）。
- 内容：
  - `output/tasks.jsonl` 是正式题，`output/tasks_reserve.jsonl` 是后备题；
  - `split_manifest.json` 记录每个场景分到哪一份；
  - `sft/` 是 SFT 的帧列表、发布用的题目和任务登记表（渲染由训练链做）。
- 生成配置：`configs/generation_r6_20261014.json`。代码：main `9215427e`。
- 规模：
  - 1,000 个场景里 938 个有题，正式题 13,094 道，后备题 14,443 道；
  - train / val / test 是 800 / 100 / 100 个场景，正式题 10,410 / 1,358 / 1,326 道。
- 检查：抽样回放 1,535 道（6 个族都有），全部恰好在 d\* 成功、无碰撞。
- **配环境的默认设置用**（`r1_legacy_success` 关闭，可见规则 `face_visible_v3`）。没有经过模型审核。
- 详见 `cluster_runs/REPORT_R6_20261014.md`。

**旧数据：**
- `interiorgs_v10_gen_r6_20261014/`：本轮第一次重跑，有 6.5% 的题参照物是另一个目标，作废。
- `interiorgs_v10_gen_20261014/`：第五轮的规则（描述 v2、旧的墙），不要再用来训练。
- `interiorgs_v10_gen_20261012_final/full/`：旧规则，只能在打开 `r1_legacy_success` 的环境里用。
- 20261011c：不要再用，projective 题约一半 d\* 偏大。
- 作废的运行：`interiorgs_v10_gen_20261012_final/full_aborted_no_runtime_check/`；`interiorgs_v10_r4_20261013/sample_r4_aborted_verify_bug/`。

**人工标注：**
- 第四轮试点：`exps/interiorgs_v10_pilot_labels_20261013/interiorgs_pilot_labels_20261013_1606.json`。
- 第五轮检查页：`exps/interiorgs_v10_check_labels_20261014/interiorgs_check_labels_20261014_0858.json`（40 道里 16 道有问题，第六轮按它改了规则）。
- 第六轮人工检查（100 道 val / test 题）：本机 `exps/interiorgs_v10_audit_site_20261014/index.html`，结果放到 `exps/interiorgs_v10_audit_labels_20261014/`。

**深度核对：** val / test 2,684 道里 199 道（7.4%）终点目标被渲染出的表面挡住，抽看 10 道约一半是真的、一半是标注框放偏。清单在 `interiorgs_v10_r6_20261014/depth_check_b/`。没有据此改数据。

## 2. 已定的决定

"题"（任务）指：一个场景、要找的目标物体、要达到的画面条件（成功判据），以及相机的起点位置和朝向。
"最少步数" d\* 指从起点到满足成功判据，最少需要几个动作（前进、后退、左移、右移，每步 0.3 m；左转、右转，每步 20°）。
"参照物"指描述目标时借用的物体（例如"沙发旁边的边几"里的沙发）。

| 日期 | 决定 |
| --- | --- |
| 2026-10-10 | 起点画面里至少一个目标能被认出来；终点画面里所有目标都能被认出来；起点必须合法且尚未成功。（例外见 2026-10-12 第 2 条；"能认出"的定义见 2026-10-13 第 2 条和 2026-10-14 第 1 条） |
| 2026-10-10 | 同类物体可以出现多个，用描述区分；描述不唯一的物体不出题。（描述规则见 2026-10-14 第 3、4 条和第六轮第 1、4 条） |
| 2026-10-10 | 题型扩展到 projective 以外，新题型用 v2 判据（`canonical_spatial_task_h1_v2`）；projective / FOV 的 v1 判据保持逐位不变（R1 用的就是它们）。 |
| 2026-10-12 | **俯仰保持水平。** `horizon_locked` 运动模型只用于实验。 |
| 2026-10-12 | **occlusion_alignment 的终点只要求遮挡物能被认出来**，目标必须在画面内并被挡住（目标 8 个角都在相机前方、至少 80% 的轮廓在画面内、其中至少 85% 被挡住）。审核时这个族改问"目标是否被遮挡物挡住"。 |
| 2026-10-12 | **按族配额，软上限。** 每个场景：projective 2–6、absolute_positioning 2–4、screen_occupancy 2–4、equidistance 1–3、occlusion_alignment 1–2、fov_inclusion 1–3。别的族都出不了题时，允许超过上限把 14 道补满。 |
| 2026-10-12 | **后备题。** 每个场景最多 16 道，只有正式题被刷掉的场景才用。 |
| 2026-10-12 | **split。** R1 的 45 个训练场景固定进 train，7 个评测场景固定进 test，其余按场景数 80/10/10；同源组必须进同一份。 |
| 2026-10-12 | **时间预算按工作量计**，600 个单位；"找终点"阶段的低估接受。 |
| 2026-10-12 | **最少步数 d\* 必须和环境一致**（`runtime_success_check` 继续开启）。 |
| 2026-10-13 | **1. 环境改动合入并作为默认**：新题型判据、坐标方向覆盖；OA 的两个提示开关默认关闭；SFT 导出带"当前相机位姿"一行。 |
| 2026-10-13 | **2. 可见规则**：成功画面里每个必须认出的目标至少有一个面清楚可见；环境和生成器用同一份代码（`vagen/envs/active_spatial/target_visibility.py`）；OA 只要求遮挡物可见；起点"至少一个能认出"用同一定义；重叠不到 0.10 m 不算遮挡；R1 旧规则开关 `r1_legacy_success` 保留，默认关闭。 |
| 2026-10-13 | **3. 不出题清单**：318 个合并大框和 726 个无法确定的框不当目标、不当参照物，碰撞和遮挡按原框算（`configs/no_task_objects_20261013.json`）。 |
| 2026-10-13 | **4. 短场景保持短。** |
| 2026-10-13 | **5. FOV 作为正式族**（`fov_inclusion_h1_v2`，两个物体至少 95% 在画面内且都可见），配额 1–3。 |
| 2026-10-14 | **1. 可见门槛按人工标注定：一个面露出至少 0.4，露出部分至少占画面 0.3%。** 重叠余量保持 0.10 m。 |
| 2026-10-14 | **2. 左右题要有 45° 限制。** 成功画面的相机视角下，地面上 B 中心指向 A 中心的线和相机左右方向的夹角不超过 45°。环境和生成器都按这条规则。 |
| 2026-10-14 | **3. 只在需要区分时才加描述**（第六轮第 1 条改为以起点所在房间为准）。房间类型从家具推断，InteriorGS 的房间没有名字。参照物不能是目标的一部分，也不能和目标重叠、压在目标上或被目标压着。 |
| 2026-10-14 | **4. 参照物要选好。** 最大的面至少 0.25 平方米；在它所在的房间里唯一，且整个描述在全场景只指向一个物体；在成功画面里和目标一起满足可见规则（题目里记为 `ref_anchor`）。"wall panels"和"edge combination"不再当目标或参照物。 |
| 2026-10-14 | **5. R1 的训练和评测保持旧的成功规则**（`r1_legacy_success: true`）。 |
| 2026-10-14 | **6. 训练：先 SFT 再 RL。** SFT 只用 train 正式题的最短路径轨迹（val 正式题做验证），不混 QA，不用 test。RL 用 dense-score S0（稀疏奖励），先训 150 次更新。不做训练数据的模型审核。LLaMA-Factory 保持现状（关掉它的版本检查）。106 个有题场景没有 FOV 题，接受。 |
| 2026-10-14 第六轮 | **1. 描述以起点所在房间为准**（描述规则 v3，对象预设 `v10_r6`）。目标在整个场景唯一：只写名字。不唯一但在它的房间里唯一：起点在这个房间、且从起点认不出另一个同类物体时只写名字；起点在别的房间时用"in the 房间"（房间类型分得开时），分不开就不用这个起点。同一房间有多个：参照物或大小比较。 |
| 2026-10-14 第六轮 | **2. 可见规则 `face_visible_v3`：** 墙用 structure.json 的墙列表（每面墙从地面到它的高度，门的范围内只保留门上方部分，窗户算墙），窗帘挡视线。碰撞不变。 |
| 2026-10-14 第六轮 | **3. 不出题清单 20261014**：在 20261013 的基础上，加上尺寸明显不符合同类物体的标注框（`label_outliers.py`），共 11,211 个。 |
| 2026-10-14 第六轮 | **4. 参照物不能是同一题的另一个目标**（第一次重跑里 6.5%，超过 2% 的线）：换成其他描述，换不了就不出这道题。 |
| 2026-10-14 第六轮 | **5. structure.json 没有墙的场景不出题；房间里同时有两类房间的东西就不给房间类型；OA 起点必须能看清要藏的目标。** |
| 2026-10-14 第六轮 | **6. 集群只用原目录** `/mnt/umm/users/yinbaiqiao/VAGEN-Lite`（已更新到 main 并逐个核对）。VAGEN-Lite-r4 不再使用；`VAGEN-Lite-r4-envs/before` 保留作 R1 回归对照。 |
| 2026-10-14 第六轮 | **7. 检查点：** SFT 每个 epoch 存一次；RL 每 25 次更新存一次，每 50 次的都留，其他只留最新 2 个。 |

## 3. 现在在做的事

**正式训练已启动（2026-10-14 15:29 UTC），自动接续：** 渲染 SFT 画面（1 卡）→ SFT 数据 → SFT 7B（8 卡，2–3 小时）→ RL S0 150 次（1 + 8 卡，7–8 小时）。

随时看进度：

```bash
ssh -o ControlMaster=no fujian 'cat /mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_train_20261014/PROGRESS.md'
```

训练完后：按 `cluster_runs/REPORT_R6_20261014.md` 第五节跑评测（`evaluation.py`，从 test 固定抽 280 道，在 `interiorgs_v10_r6_20261014/eval_b/`）。

## 4. 待决事项

按影响大小排序；推荐项放在第一个。详见 `cluster_runs/REPORT_R6_20261014.md` 第八节。

1. **RL 到 150 次时要不要接着训到 250 次**：(a) **推荐**：验证成功率仍在明显上升就接着训，约 5 小时；(b) 停在 150。
2. **看新的人工检查页**（100 道题，约 30 分钟）。
3. **深度核对标出的题**：(a) **推荐**：等检查页的结果再定，可先把门槛调得更严以减少误报；(b) 现在就核对 train 并出清单（1 卡约 15 分钟）。
4. **墙的厚度**（相机在门洞里时两侧墙垛挡不住视线）：(a) **推荐**：下一轮把墙改成两面；(b) 不改。
5. **左右 45° 收紧到 40°**：(a) **推荐**：不改；(b) 收紧并重新出题。
6. **起点要求两个目标都能认出**：(a) **推荐**：不改；(b) 改。
7. **VAGEN-Lite-r4**：(a) **推荐**：删（里面没有输出），`VAGEN-Lite-r4-envs/` 保留；(b) 留着。
8. **原目录里剩下的 27 个未管理文件**：(a) **推荐**：QA 脚本交给负责 QA 的会话处理，第三轮的 7 个补丁文件删掉（已备份）；(b) 都留着。
9. **训练期间原目录只加文档、不改代码**（另一个会话也在这里提交）：(a) **推荐**：是；(b) 不限制。
10. **verl 子模块里那处训练要用、但没提交的改动**：(a) **推荐**：提交到 verl 分支；(b) 保持现状（已备份）。
11. **模型审核**：训练数据未审核；之后要审核按 `GPU_PLAN_20261012.md`（2–8 GPU·小时）。

## 5. 怎么复现

| 用途 | 命令 / 文件 |
| --- | --- |
| 本轮配置 | `configs/generation_r6_20261014.json`（对象预设 `v10_r6`、不出题清单 20261014、`require_structure_walls`、`oa_start_target_visible`、`forbid_other_target_reference`） |
| 第五轮配置 | `configs/generation_r5_20261014.json`（可见规则已是 v3，不再逐字复现第五轮） |
| 上一轮最终配置 | `configs/generation_final_20261012.json`（在本轮代码上仍逐字复现） |
| 分片出题 | `shard_sco.py prepare / submit / status / delete --out OUT`（`--job-dir` 再开一组任务） |
| 不出题清单 | `label_outliers.py --meta-root … --base configs/no_task_objects_20261013.json --out …` |
| 生成 split | `make_split.py --run OUT/output --groups cluster_runs/same_source_groups_20261012.json --r1-frozen …/frozen_v1 --out OUT/split_manifest.json` |
| 在环境里回放 | `replay_env.py --run OUT --scenes-root … --no-task-objects configs/no_task_objects_20261014.json` |
| R1 回归 | `r1_regression.py plan / trace / compare` |
| 检查页 / 人工检查 | `pilot.py check`、`pilot.py audit`、`pilot.py topdown`、渲染 `pilot_sco.py`、网站 `pilot_site.py --mode check / audit` |
| 深度核对 | `depth_check.py prepare`，`pilot_sco.py prepare --command "depth_check run"`，`depth_check.py report` |
| 训练 | `train_chain.py run --plan TR/plan.json`（渲染 → SFT 数据 → SFT → RL 配置 → RL）；单个任务用 `job_sco.py` 跑 `sco_interiorgs_v10_train.sh` 的各个模式 |
| 接着训 | `sco_interiorgs_v10_train.sh rl RUN RL 250`（resume_mode auto，从最新检查点接着训）；`rl-resume-check` 是单卡验证 |
| 评测 | `evaluation.py select / run / table`；单卡任务 `sco_interiorgs_v10_train.sh eval RUN EVAL NAME MODEL [PROMPT_FORMAT]` |

报告（都在 `cluster_runs/`）：
- `REPORT_REAL_20261011.md`：首次全量；
- `REPORT_DIAG_20261012.md`：上 GPU 前的诊断；
- `CABINET_AUDIT_20261012.md`：标注框核查；
- `REPORT_FULLRUN_20261012.md`：第三轮；
- `GPU_PLAN_20261012.md`：GPU 阶段计划；
- `REPORT_R4_20261013.md`：第四轮；
- `REPORT_R5_20261014.md`：第五轮；
- `REPORT_R6_20261014.md`：本轮。
