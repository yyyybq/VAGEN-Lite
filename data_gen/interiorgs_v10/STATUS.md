# InteriorGS v10 任务生成器：状态

这是 v10 生成器唯一的状态文档。每轮工作结束时更新。最后更新：2026-10-14（第五轮）。

## 1. 现在可以用的数据

**用这一份：** `exps/vagen_active_spatial/interiorgs_v10_gen_20261014/`（集群 fujian，仓库目录下）。
- 内容：`output/tasks.jsonl` 是正式题，`output/tasks_reserve.jsonl` 是后备题，`split_manifest.json` 记录每个场景分到哪一份；`sft/` 是 SFT 要渲染的帧列表、发布用的题目、任务登记表（还没有渲染）。
- 生成配置：`configs/generation_r5_20261014.json`。代码：main `35585b88`。
- 规模：
  - 1,000 个场景里 942 个有题，正式题 13,158 道，后备题 14,600 道；
  - train / val / test 分别是 800 / 100 / 100 个场景、10,477 / 1,351 / 1,330 道正式题。
- 检查：抽样回放 1,537 道（6 个族都有），全部恰好在 d\* 成功、无碰撞；100 个样本场景上 1,116 道 projective 和 FOV 题按环境规则重算 d\*，全部一致。
- **配环境的默认设置用**（`r1_legacy_success` 关闭）。没有经过模型审核。
- 详见 `cluster_runs/REPORT_R5_20261014.md`。

**旧数据：**
- `interiorgs_v10_gen_20261012_final/full/`：旧规则，只能在打开 `r1_legacy_success` 的环境里用，不要再用来训练 v10。
- 20261011c：不要再用，projective 题约一半 d\* 偏大。
- 作废的运行：`interiorgs_v10_gen_20261012_final/full_aborted_no_runtime_check/`；`interiorgs_v10_r4_20261013/sample_r4_aborted_verify_bug/`。

**人工标注：**
- 第四轮试点的结果：`exps/interiorgs_v10_pilot_labels_20261013/interiorgs_pilot_labels_20261013_1606.json`（208 个门槛例子标了 207 个，任务标了 8 道）。
- 第五轮检查页：本机 `exps/interiorgs_v10_check_site_20261014/index.html`，结果放到 `exps/interiorgs_v10_check_labels_20261014/`。

## 2. 已定的决定

"题"（任务）指：一个场景、要找的目标物体、要达到的画面条件（成功判据），以及相机的起点位置和朝向。
"最少步数" d\* 指从起点到满足成功判据，最少需要几个动作（前进、后退、左移、右移，每步 0.3 m；左转、右转，每步 20°）。
"参照物"指描述目标时借用的物体（例如"沙发旁边的边几"里的沙发）。

| 日期 | 决定 |
| --- | --- |
| 2026-10-10 | 起点画面里至少一个目标能被认出来；终点画面里所有目标都能被认出来；起点必须合法且尚未成功。（例外见 2026-10-12 第 2 条；"能认出"的定义见 2026-10-13 第 2 条和 2026-10-14 第 1 条） |
| 2026-10-10 | 同类物体可以出现多个，用描述区分；描述不唯一的物体不出题。（描述规则见 2026-10-14 第 3、4 条） |
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
| 2026-10-14 | **1. 可见门槛按人工标注定：一个面露出至少 0.4，露出部分至少占画面 0.3%。** 重叠余量保持 0.10 m。规则版本 `face_visible_v2`。 |
| 2026-10-14 | **2. 左右题要有 45° 限制。** 成功画面的相机视角下，地面上 B 中心指向 A 中心的线和相机左右方向的夹角不超过 45°。环境和生成器都按这条规则。没有前后题，本轮不加。 |
| 2026-10-14 | **3. 只在需要区分时才加描述。** 按顺序：整个场景唯一只写名字；否则用房间（房间类型从家具推断，InteriorGS 的房间没有名字），前提是房间类型能区分开；再否则用参照物或大小比较；都不行就不出题。参照物不能是目标的一部分，也不能和目标重叠、压在目标上或被目标压着。 |
| 2026-10-14 | **4. 参照物要选好。** 最大的面至少 0.25 平方米（按尺寸，不写类别名单）；在它所在的房间里唯一，且整个描述在全场景只指向一个物体；在成功画面里和目标一起满足可见规则（题目里记为 `ref_anchor`）。"wall panels"和"edge combination"不再当目标或参照物。 |
| 2026-10-14 | **5. R1 的训练和评测保持旧的成功规则**（`r1_legacy_success: true`），`examples/train/active_spatial/` 下的 R1 配置、R1 启动器和评测脚本都已设置。 |
| 2026-10-14 | **6. 训练：先 SFT 再 RL。** SFT 第一版只用 train 正式题的最短路径轨迹（val 正式题做验证），不混 QA，不用 test。RL 沿用 dense-score 试验的设置，环境用默认的新规则。训练从单独目录 `VAGEN-Lite-r4` 运行；不动集群仓库目录和挂起的 `pt-0nmy49z4`。 |

## 3. 训练前还要做的事（按顺序）

1. **你看完检查页，确认三个问题改好了**（约 15 分钟）。
2. **渲染 SFT 画面**：1 张卡，约 1 小时（39,746 帧）。
3. **生成 SFT 对话和 LLaMA-Factory 数据**：CPU，约 10 分钟。
4. **SFT**：Qwen2.5-VL-7B，8 张卡，约 2–3 小时。
5. **RL**：从 SFT 模型开始，渲染 1 张卡 + 训练 8 张卡，150 次更新约 7–8 小时。

命令见 `cluster_runs/REPORT_R5_20261014.md` 第六节第 5 部分。SFT 和 RL 都已在单卡上用 3B 模型试跑通过。

## 4. 待决事项

按影响大小排序；推荐项放在第一个。详见 `cluster_runs/REPORT_R5_20261014.md` 第八节。

1. **检查页的结论**（训练启动前唯一要你做的事）。
2. **RL 奖励分支**：(a) **推荐** S0（稀疏，终点最稳定）；(b) S5（前期快）。只改环境配置。
3. **RL 跑多久**：(a) **推荐** 150 次更新，约 7–8 小时；(b) 250 次，约 12–13 小时。
4. **标注框以外的遮挡物**（装饰墙、坡屋顶等）：(a) **推荐** GPU 阶段用渲染深度核对终点画面，剔掉终点目标被遮住的题，约半天代码加 1 GPU·小时；(b) 不处理。
5. **参照物恰好是同一题的另一个目标**：(a) **推荐** 保留；(b) 禁止，需重跑全量。
6. **106 个有题场景没有 FOV 题**：(a) **推荐** 接受；(b) 放宽到 90% 在画面内，未经人工标定。
7. **模型审核**：本轮训练数据未审核；之后要审核按 `GPU_PLAN_20261012.md`（2–8 GPU·小时）。
8. **以前留下的挂起任务** `pt-0nmy49z4`、`pt-zgpey3na`、`pt-f8xpnizj`：(a) **推荐** 删除两个 10-07 的任务，`pt-0nmy49z4` 由你决定；(b) 保留。
9. **LLaMA-Factory 版本**：(a) **推荐** 保持现状（关掉它的版本检查，试跑正常）；(b) 换支持 transformers 4.56 的版本，要改环境。
10. **集群仓库目录**仍是第四轮之前的代码（因为 `pt-0nmy49z4`），训练用 `VAGEN-Lite-r4`。

## 5. 怎么复现

| 用途 | 命令 / 文件 |
| --- | --- |
| 本轮配置 | `configs/generation_r5_20261014.json`（`visibility_rule: env`、对象预设 `v10_r5`） |
| 第四轮配置 | `configs/generation_r4_20261013.json`（`face_v1` 现在等于 `env`，环境规则已变，不再逐字复现第四轮） |
| 上一轮最终配置 | `configs/generation_final_20261012.json`（在本轮代码上仍逐字复现） |
| 分片出题 | `shard_sco.py prepare / submit / status / delete --out OUT`（`--job-dir` 再开一组任务） |
| 只拉元数据 / 3DGS 文件 | `fetch_interiorgs_metadata.py --source aoss`；全部 3DGS 文件已在 `/mnt/umm/users/yinbaiqiao/InteriorGS_ply` |
| 生成 split | `make_split.py --run OUT/output --groups cluster_runs/same_source_groups_20261012.json --r1-frozen …/frozen_v1 --out OUT/split_manifest.json` |
| 在环境里回放 | `replay_env.py --run OUT --scenes-root … --no-task-objects configs/no_task_objects_20261013.json` |
| R1 回归 | `r1_regression.py plan / trace / compare` |
| 检查页 / 标注网站 | `pilot.py check`（或 `select`）、`pilot.py topdown`、渲染 `pilot_sco.py`、网站 `pilot_site.py --mode check` |
| SFT 数据 | `training_prep.py sft-frames / sft-records`，然后 `scripts/prepare_active_spatial_sft.py` |
| RL 配置 | `training_prep.py rl-config`（环境模板 `configs/rl_env_v10_S0.yaml`） |
| 训练任务 | `job_sco.py prepare / submit / status / delete`，跑 `examples/train/active_spatial/sco_interiorgs_v10_train.sh` 的各个模式 |

报告（都在 `cluster_runs/`）：
- `REPORT_REAL_20261011.md`：首次全量；
- `REPORT_DIAG_20261012.md`：上 GPU 前的诊断；
- `CABINET_AUDIT_20261012.md`：标注框核查；
- `REPORT_FULLRUN_20261012.md`：第三轮；
- `GPU_PLAN_20261012.md`：GPU 阶段计划；
- `REPORT_R4_20261013.md`：第四轮；
- `REPORT_R5_20261014.md`：本轮。
