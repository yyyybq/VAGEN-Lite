# 环境能力题：2026-10-16 进度

结论：任务定义、可达状态来源、审核与训练格式入口已实现并验证；两项真实深度渲染作业仍等待资源，尚未交付本轮能力样题或基座结果。

## 工作位置与边界

- 独立分支：`capability-qa/20261016`；独立工作目录：`/mnt/umm/users/yinbaiqiao/probe_spatial/.worktrees/VAGEN-Lite-capability-qa-20261016`。
- 整理分支 `cleanup/20261016` 尚未合入 main；不修改共享 VAGEN-Lite 工作目录，不推 main。新分支已基于远端 main `168d443b`。
- 代码放 `data_gen/interiorgs_qa/capability/`；本轮产物放 `exps/interiorgs_capability_qa_20261016/`。旧 production_v3 和既有 benchmark 样题原样保留。
- GPU 仅允许 zoetrope 的 SCO 单卡作业，用于必要渲染与 Qwen2.5-VL-7B 有图/无图检查；不训练、不以模型答案筛选样题。
- 用户审核前，所有能力题均待人工审核，正式训练放行数量为零。

## 已核实

- 读取根 STATUS、v10 STATUS、interiorgs_qa PROGRESS；它们部分描述是历史状态，以实际路径和数据为准。
- 当前第十轮正式任务和场景划分位于共享目录 `exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015/`；仅作只读来源。
- 正式任务包含六类；真实成功条件、当前训练去除名单、轨迹帧绑定已核查，定义及数量见 DESIGN.md。
- 现有 QA 代码支持多图，但默认相机采样、旧物体描述和可见性不能未经适配直接用于本任务。

## 下一步与交付检查

1. 固定数据与代码来源；写六类任务定义、能力题规则、模糊区间和待定事项。
2. 从四个正式 train 场景的可达状态制作样题，保存每个候选的通过/排除理由及几何依据；真实深度不足时不造标签。
3. 新增功能测试通过后发布私有审核页，跑有图/无图基座检查并清理本轮单卡作业；报告图片预算估计，等待用户审核。

本轮用户完整要求：`/mnt/umm/users/yinbaiqiao/.codex/attachments/e5b2d1b0-a19f-4ec5-b513-19dd7799cf2c/pasted-text.txt`。

## 2026-10-16 实际进展更新

- 独立工作分支已基于远端 main `168d443b9b9b91ed696d8182319a7f26c38103f6`；共享工作目录未改，整理分支尚未合入。
- 六类正式训练任务的真实条件、数量及每类两条原指令已写入同目录 DESIGN.md。占画面高度的样题继续暂停；先覆盖其余五类。
- 本批源目录 `exps/interiorgs_capability_qa_20261016/` 已绑定四个训练场景、56 道正式任务、1104 条状态记录；其中 211 张是已有轨迹图片（155 张原训练输入、56 张终点观察），其余是按环境动作得到的合法扩展。5 个终点虽满足运行时位置条件，未满足额外的墙距筛选，未作为样题输入。
- 首轮几何检查只有 106 个单图草案，尚未核对真实深度；部分语义答案缺失，不能按 100 道成品汇报。已沿同一正式任务的合法动作路径补充177个相机位置，不增加后备物体组合。
- 已提交 zoetrope 单卡 H800 作业 `pt-7bzew4mi`；当前调度状态 STARTING，worker Pending，尚无运行日志或新渲染结果。提交记录在 `jobs/depth_v1/`；不得重复提交该批。
- 新审核页和有图／无图基座检查入口已实现并通过各自 CPU 测试；未真实模型推理、未发布网页、未人工审核，训练放行仍为 0 道。
- 本轮所有新增几何、深度、选样和网页字段均与旧 production_v3 隔离；没有修改环境或 v10 出题器。

## 当前恢复点（2026-10-16 21:23 UTC）

- 首轮作业 `pt-7bzew4mi` 与只渲染177个新相机的补充作业 `pt-zzymdg47` 均为 STARTING，开始时间为空。状态原文已存各自 jobs 子目录；没有运行日志或真实深度产物，不能报告完成。
- 本次已持续核对首轮作业约63分钟、补充作业约45分钟；两者worker均为Pending且未分配主机。953个首轮相机和177个新增相机都仍等待渲染；这些是渲染请求数量，不是合格题数量。保留原作业，不重复提交、不停止等待中的作业。最新状态汇总为 jobs/latest_status.json。
- 额外沿原任务路径检查12,087个动作后的状态，保存177个新相机；与原1104条源状态的相机集合无交集，完整动作复算与56条正式任务原文校验均一致。证据在 extension_v2/verification.json。500个几何候选只是搜索目标，本次未达到；不得降低质量补数。
- 使用当前含窗帘和门洞信息的可见性接口重算，首轮有效几何文件为 geometry_v3/geometry.jsonl；geometry_v1/v2 仅保留过程记录，不用于最终选样。
- 对称远近问法的对象提及顺序按固定种子随机一次，真实近物和原任务角色不变。原任务中某一角色从未更近的覆盖缺口仍记录，不把提及顺序平衡当作补齐空间状态。
- 已实际调用当前训练加载器和批次整理代码完成双图CPU测试：图片顺序正确，监督仅为答案字母与结束标记；没有加载权重或更新参数。最终待审文件生成后还需全量复查。证据在 cpu_evidence/。
- 新私有审核站点已登记，项目编号 `appgprj_6ad2899bcd548191b9a6d0fdac51e17e`，配置在 site_source/.openai/hosting.json。尚未发布，不把预期网址当可用网页。
- 新增功能共118项CPU测试通过，包括“较小／较大”描述必须看见比较对象的筛选；基座5个权重文件与指定Qwen2.5-VL-7B-Instruct缓存快照的完整内容哈希一致。这些结果不代替GPU渲染或模型推理。
- 本轮已完成的能力样题为0道、人工审核放行为0道；私有审核网页尚未发布，基座检查尚未提交。当前动作轨迹训练一轮实际读取25,243次图片；需要多少场景才能匹配这个图片预算，必须等真实图像筛选后的产量再估算，尚未验证。

渲染结束后的固定顺序：读取两个实际 render_manifest，执行 pipeline combine（原root + extension_v2、首轮geometry_v3）到 combined_v1；再执行 pipeline finalize（root与geometry均为combined_v1）到 review_v1；全量运行 verify_export；发布新私有站点；用 evaluate 在 zoetrope 单卡跑基座有图／无图检查。所有任务只在确认终态并保存输出后清理。待人审训练放行仍为0道。

## 渲染完成后的准确命令（尚未执行）

先检查两个作业的实际结束状态、日志和逐图失败记录。以下命令不重新渲染；输出目录必须尚不存在，失败记录保留在原目录，修复后使用新版本目录。

```bash
cd /mnt/umm/users/yinbaiqiao/probe_spatial/.worktrees/VAGEN-Lite-capability-qa-20261016
QA_PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
QA_OUT=$PWD/exps/interiorgs_capability_qa_20261016
QA_BASE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898
export PYTHONDONTWRITEBYTECODE=1
"$QA_PY" -m data_gen.interiorgs_qa.capability.pipeline combine --root "$QA_OUT" --extension "$QA_OUT/extension_v2" --geometry "$QA_OUT/geometry_v3" --out "$QA_OUT/combined_v1"
"$QA_PY" -m data_gen.interiorgs_qa.capability.pipeline finalize --root "$QA_OUT/combined_v1" --geometry "$QA_OUT/combined_v1" --out "$QA_OUT/review_v1"
"$QA_PY" -m data_gen.interiorgs_qa.capability.verify_export --dataset "$QA_OUT/review_v1/pending_training_format.jsonl" --model "$QA_BASE" --out "$QA_OUT/review_v1/loading_verification"
```

上述全量加载和图片绑定检查通过后，才提交已冻结准备的基座检查；它会对同一份题分别做有图和纯文字推理，不训练。

```bash
"$QA_PY" -m data_gen.interiorgs_qa.sco submit --job "$QA_OUT/jobs/base_check_v1" --name interiorgs-capability-basecheck-20261016
```

网站继续使用已登记的同一个私有项目，发布 review_v1/review_site 的实际内容；不得另外建项目或上传未验证的占位题。所有审核状态保持待人工审，备注可选，非空备注进入逐条分析清单。
