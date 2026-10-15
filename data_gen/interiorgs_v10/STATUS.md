# InteriorGS v10 任务生成器：状态

这是 v10 生成器唯一的状态文档。每轮工作结束时更新。最后更新：2026-10-15（第九轮）。

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

**训练出的模型（`exps/vagen_active_spatial/interiorgs_v10_train_20261014/`）：**
- SFT：`lf/model`（Qwen2.5-VL-7B，3 个 epoch，最后的验证损失 0.166）；每个 epoch 的模型在 `lf/model/checkpoint-163 / -326 / -489`。
- RL（S0）：训到第 250 次，`rl_run/checkpoints/global_step_*`（留下第 50、100、150、175、200、225、250 次；`actor/huggingface` 是可直接加载的权重）。前 60 次更新只训练价值网络（`critic_warmup: 60`），第 50 次就是 SFT 模型。
- 评测集：`interiorgs_v10_r7_20261014/eval_c/tasks.jsonl`（278 道 test 题，已去掉深度核对标出的题）。全部模型的评测结果（带每一步的轨迹）：`interiorgs_v10_r8_20261014/eval2/`，表 `eval2/table.md`。
- **评测结果（成功率）：** 原始 7B 5.0%，R1 9.0%，SFT 第 1/2/3 个 epoch 10.1% / 18.7% / 23.4%，RL 第 100/150/200/250 次 21.6% / 20.1% / 24.5% / 21.6%。RL 和 SFT 第 3 个 epoch 的差别在随机波动之内。
- **评测轨迹浏览页：** 本机 `exps/interiorgs_v10_eval_browser_r9_20261015/index.html`（双击打开；第八轮的 10 个模型加第九轮的诊断评测 D1–D7）；集群 `interiorgs_v10_r9_20261015/browser/`。第八轮那一份 `exps/interiorgs_v10_eval_browser_20261015/` 仍可用。
- **诊断评测（第九轮，SFT 第 3 个 epoch、278 道 test 题，不训练只换规则）：** 现在的规则 23.4%；每个动作检查成功 24.1%；上限 20 24.1%；撞墙不结束 24.1%；撞墙不结束 + 如实反馈 26.6%；带历史 28.4%；五个开关全开 34.9%（逐题 10 对 42，p < 0.001）。结果在 `interiorgs_v10_r9_20261015/eval_diag/`。
- `lf/model` 的所有者已改回你（第九轮）；`checkpoint-489` 和最终模型逐字节相同。

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
| 2026-10-14 第七轮 | **1. RL 到 150 次时**：进度文件判"验证成功率仍在明显上升"就从检查点接着训到 250 次，否则停在 150 次（自动程序 `interiorgs_v10_r7_20261014/scripts/watch_r7.py` 执行）。 |
| 2026-10-14 第七轮 | **2. 评测集去掉深度核对标出的题**，按同族、同最少步数段从 test 补（278 道）；训练数据不变。 |
| 2026-10-14 第七轮 | **3. 墙改成有厚度：下一轮做**（要重新出题）。左右 45° 不改；起点只要求认出至少一个目标，不改。 |
| 2026-10-14 第七轮 | **4. VAGEN-Lite-r4 已删除**，VAGEN-Lite-r4-envs/ 保留。原目录里第三轮的 7 个补丁文件训练后删除；20 个 QA 生产脚本留给负责 QA 的会话；verl 子模块的改动训练后提交到 verl 分支。 |
| 2026-10-14 第七轮 | **5. 训练期间原目录只加文档、不改代码。** |
| 2026-10-14 第八轮 | **1. 150 次后的判断改用修正后的验证成功率**：第 150 次比之前最好的一次低 10 个百分点以上才停，否则接着训到 250 次（第七轮的"仍在明显上升"不再用；`train_chain.py` 的 `keep_training`）。本次：38.7% 对 39.7%，接着训到了 250 次。 |
| 2026-10-14 第八轮 | **2. 评测所有模型**（原始 7B、SFT 每个 epoch、RL 每 50 次、R1），评测记录每一步的轨迹；做评测轨迹浏览页（`eval_browser.py`）。 |
| 2026-10-15 第八轮 | **3. 收尾已做**：代码推到 main；verl 的改动提交到 `yyyybq/verl` `vagen-lite`（`a0ae85c`）并更新子模块版本号；第三轮的 7 个补丁文件已删除。 |
| 2026-10-15 第九轮 | **1. 环境加 5 个开关，默认都关**（R1 和现有训练不变）：每个动作检查成功（`success_check_each_action`）、上限可设（`max_episode_steps` / `max_turns`）、撞墙不结束（`max_consecutive_collisions: 0`）、碰撞如实反馈（`collision_feedback_detail`）、带历史（评测 `--history`，格式和 SFT 逐字一致）。 |
| 2026-10-15 第九轮 | **2. 只给下一版用的两个文字开关**：系统提示的奖励说明按实际规则写（`prompt_reward_text: actual`）、位姿的 −0.00 写成 0.00（`pose_normalize_negative_zero`）；环境和 SFT 导出各有同名设置，要一起打开。 |
| 2026-10-15 第九轮 | **3. `lf/model` 所有者改回你；`patches/` 已删。** |
| 2026-10-15 第十轮 | **1. 默认规则**：每个动作检查成功、撞墙不结束、碰撞如实反馈、带历史默认打开，动作和回合上限保持 12（`configs/rl_env_v10_S0.yaml`、评测默认 `--history`；第六至九轮的模板留作 `rl_env_v10_S0_r6.yaml`）。R1 和环境的代码默认值不变。 |
| 2026-10-15 第十轮 | **2. RL 带历史的做法**：先各做一次单卡试跑（"每个回合单独一个样本" `--history-mode no_concat`、"一道题一个序列" `--history-mode concat`），比较速度、奖励是否正常、训练是否稳定；"一道题一个序列"正常就用它，否则用前者。单卡放不下 7B 的 PPO（第五轮同样原因），试跑用第五轮的 3B SFT 模型。 |
| 2026-10-15 第十轮 | **3. 文字首尾空白**：SFT 导出照环境的原文（`prepare_active_spatial_sft.py --keep-text`），不再去掉。 |
| 2026-10-15 第十轮 | **4. 系统提示的奖励说明改成实际规则（`prompt_reward_text: actual`），位姿 −0.00 统一成 0.00**；环境模板和 SFT 导出策略（`admission_v2.json` 的 `sft.export`）一起打开。 |
| 2026-10-15 第十轮 | **5. 深度核对也对 train 做**（正式题和后备题一起），标出的题从训练数据里去掉（`sft-frames --exclude`），val、test 的评测集抽题时也去掉（`evaluation.py select --exclude`）。 |
| 2026-10-15 第十轮 | **6. "撞到之后怎么办"的样本，只做便宜的版本**：在最短走法的某个回合前插入一个会被挡住的动作（用环境自己的碰撞检测挑），接上环境的如实反馈，再继续原来的最短走法（`release.blocked_insert`，比例 `sft.export.blocked_turn_share`）。 |
| 2026-10-15 第十轮 | **7. 后退占比本轮不改。** |
| 2026-10-15 第十轮 | **8. 挑模型只看验证集**：固定 200 道左右的 val 题（按族和最少步数分层，固定随机种子），每题一条不采样的轨迹；SFT 挑 epoch、RL 挑检查点都按它（`train_chain` 的 `sft_eval` / `pick_sft`，RL 的 `--val-tasks --greedy-val`）；test 只记录。 |
| 2026-10-15 第十轮 | **9. 墙改成有厚度**（第七轮第 3 条的执行）：`wall_model: thick`，生成器和环境用同一套墙板（门和落地的洞口是通道，窗户不是）。 |

## 3. 现在在做的事

**第十轮训练链在跑（2026-10-15 14:26 UTC 起，代码 `c212b0a0`）：** `exps/vagen_active_spatial/interiorgs_v10_train_r10_20261015/`，看 `PROGRESS.md` 或 `python -m data_gen.interiorgs_v10.train_chain status --dir <该目录>`。步骤：两组 SFT（只用正式题 / 加后备题，各 5 个 epoch，8 卡 pt-bjcfifv5 / pt-k9lxp6la）→ 每个 epoch 在 193 道验证题和 279 道评测题上评测 → 按验证集挑检查点 → RL（S0，一道题一个序列，`critic_warmup` 10，150 次以内，每 25 次验证）→ 评测。新数据 `interiorgs_v10_gen_r10_20261015/`（厚墙），详见 `cluster_runs/REPORT_R10_20261015.md`（训练部分待填）。

第九轮的诊断评测已结束（2026-10-15 04:40 UTC），sco 任务都已删除。详见 `cluster_runs/REPORT_R9_20261015.md`。

- **诊断评测的结论：** 现在的失败里有相当一部分是环境规则和输入造成的。同一个 SFT 模型，五个开关全开从 23.4% 升到 34.9%；主要来自带历史（+5.0 个百分点）和"撞墙不结束 + 如实反馈"（+3.2，p = 0.012）。上限 20 在全开时没有带来任何成功（成功全在 12 个动作以内）。
- **SFT 导出去掉了每段文字的首尾空白**，RL 和评测没有去掉；单独改这一点几乎没有影响（23.7%），下一版统一。
- 第八轮的结论仍然成立：**RL 没有提高评测成功率**（SFT 第 3 个 epoch 23.4%，RL 第 250 次 21.6%，逐题比较 p = 0.53）；修正后的验证成功率从第 75 次起在 36%–41% 之间不动。RL 让成功的题走得更省（动作数 / d\* 1.89 → 1.47），转向更多。
- **最常见的失败是一直后退撞墙被结束**（RL 第 250 次 278 道里 79 道），而环境只在一个回合的动作全被挡住时才提示碰撞，这 79 道里只有 23 道收到过提示。
- 验证补齐的 bug 已修（`a807609f`），以后训练的验证记录就是对的；本次训练的修正值用 `interiorgs_v10_r8_20261014/scripts/watch_r8.py --dry` 重算。

## 4. 待决事项

按影响大小排序；推荐项放在第一个。详见 `cluster_runs/REPORT_R9_20261015.md` 第六节（下一次训练的完整方案在第三节第 3 部分）。

1. **下一版默认规则**：(a) **推荐**：每个动作检查成功、撞墙不结束、碰撞如实反馈、带历史默认打开，上限保持 12（同一个 SFT 模型预计 34.9%）；(b) 只开前三个（不带历史，约 26.6%，RL 速度不变）；(c) 再把上限放宽到 20（成功题不增加，每道题多走约 5 个回合）。
2. **RL 带历史的做法**：(a) **推荐**：先单卡试跑比较"每个回合单独一个样本"和"一道题一个序列"（各约 35 分钟）；(b) 直接用现在的做法（窗口 12），150 次更新约 13–14 小时（1 + 8 卡）；(c) 只保留最近 4 个回合，约 10 小时，但和 SFT 不一致。
3. **文字首尾空白**：(a) **推荐**：SFT 导出不再去掉；(b) 环境和 RL 也去掉。和重新出题一起做。
4. **"撞到之后怎么办"的 SFT 样本**：(a) **推荐**：下一版加入（如实反馈的文字才能进入训练数据，生成器开发约 1–2 天）；(b) 不加。
5. **`critic_warmup`**：下一次训练设成 10（你已定）。
6. **新的人工检查页**（你在标）和**深度核对标出的训练题**（等检查页结果再定）。
7. **模型审核**：训练数据未审核；之后要审核按 `GPU_PLAN_20261012.md`（2–8 GPU·小时）。

## 5. 下一版数据的改动清单（先不执行）

"重新出题"指重跑整个生成器（10 个单卡 CPU 任务并行约 40 分钟），之后重新渲染 SFT 画面、重训 SFT 和 RL。原因和证据见 `REPORT_R8_20261015.md` 第五、七、八节和 `REPORT_R9_20261015.md`；按这份清单的完整训练方案见 `REPORT_R9_20261015.md` 第三节第 3 部分。

| # | 改什么 | 为什么 | 代价 |
| --- | --- | --- | --- |
| 1 | 墙改成有厚度（已定） | 墙按中线算，门洞两侧的墙垛挡不住视线，碰撞也按中线判 | 重新出题，之后全部重训 |
| 2 | 深度核对标出的题（等检查页） | val/test 7.4% 的题终点目标被渲染出的表面挡住，约一半是真的 | 删题：train 也跑一遍深度核对（约 12 分钟单卡），再重训；或先调严门槛 |
| 3 | 位姿一行用环境的回放打印（已改，`518fc7da`），−0.00 写成 0.00（开关已加，`pose_normalize_negative_zero`） | SFT 里 4.9% 的后续回合和环境只差 "−0.00" 的负号，数值相同 | 无；下一版在环境和 SFT 导出里一起打开 |
| 4 | 环境默认规则：每个动作检查成功、撞墙不结束、碰撞如实反馈、带历史（开关已加，第九轮） | 同一个 SFT 模型 23.4% → 34.9%（第九轮诊断评测） | 带历史时 RL 变慢（见待决事项第 2 条）；碰撞扣分变多 |
| 5 | SFT 加入"被挡住后怎么办"的样本 | 最短走法从不碰撞；评测里约 28% 的题因一直后退撞墙被结束 | 生成器要能造出碰撞后绕开的轨迹；SFT 数据变多 |
| 6 | 后退占 37% | 起点从终点倒推，很多最短走法以后退开头；模型看不到身后 | 按动作平衡 SFT 数据（简单），或改起点生成（重新出题） |
| 7 | SFT 和 RL 的输入统一为带历史（第九轮改为这个方向，不再拆单回合样本） | 带历史 +5.0 个百分点；SFT 本来就是多回合 | RL 每次更新变慢，取决于做法（待决事项第 2 条） |
| 8 | 系统提示里关于奖励的一句（开关已加，`prompt_reward_text: actual`） | 写着按进度给奖励，S0 没有 | 无；下一版在环境和 SFT 导出里一起打开 |
| 9 | 上限保持 12（第九轮改：不再建议放宽到 20） | 全开时没有一道成功需要 12 个以上的动作 | 无 |
| 10 | 文字首尾空白统一（第九轮发现） | SFT 导出去掉了，RL 和评测没有去掉 | 改导出一处；影响很小（第九轮 D7） |
| 11 | `rl-config` 按 `reserve` 标记只取正式题（第十轮发现） | 第十轮 `sft-frames --include-reserve` 后 rows.jsonl 含后备题，RL 训练题变成 20,600 道（含后备题 10,798） | 改一处过滤；第十轮 RL 已经用了两类题 |
| 12 | 提前评测和训练链的评测步不能写同一个结果目录（第十轮教训） | 两个任务同时追加写 results.jsonl，一行损坏 | 训练链到评测步前确认 done.json 已存在，或者不提前评 |

## 6. 怎么复现

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
| 评测 | `evaluation.py select / run / table / replace / compare`（v2 起每道题记录轨迹）；单卡任务 `sco_interiorgs_v10_train.sh eval RUN EVAL NAME MODEL [PROMPT_FORMAT\|- [选项 ...]]` |
| 诊断评测的开关（第九轮） | `evaluation.py run` 的 `--check-each-action`、`--max-actions N --max-turns N`、`--no-collision-end`、`--collision-feedback`、`--history`（隐含 `--sft-text`）、`--sft-text`；逐题比较 `evaluation.py compare --base DIR --other DIR...` |
| 评测轨迹浏览页 | `eval_browser.py --eval EVAL [EVAL2 ...] --out OUT [--label NAME=TEXT] [--left NAME --right NAME]`，双击 `OUT/index.html` |
| 在 RL 环境里核对 SFT 对话、最短走法、简单策略 | `rl_checks.py run / report` |
| 训练链接着训、评测 | `train_chain.py` 的 `rl_extend`（plan 里的 `extend_to`）和 `evaluate`（plan 里的 `eval`）两步 |

报告（都在 `cluster_runs/`）：
- `REPORT_REAL_20261011.md`：首次全量；
- `REPORT_DIAG_20261012.md`：上 GPU 前的诊断；
- `CABINET_AUDIT_20261012.md`：标注框核查；
- `REPORT_FULLRUN_20261012.md`：第三轮；
- `GPU_PLAN_20261012.md`：GPU 阶段计划；
- `REPORT_R4_20261013.md`：第四轮；
- `REPORT_R5_20261014.md`：第五轮；
- `REPORT_R6_20261014.md`：第六轮；
- `REPORT_R7_20261014.md`：第七轮（训练期间的核对）；
- `REPORT_R8_20261015.md`：第八轮（150 次后的判断、全部模型的评测、轨迹浏览页、失败分析、收尾）；
- `REPORT_R9_20261015.md`：本轮（环境开关、诊断评测、下一次训练的方案）。
