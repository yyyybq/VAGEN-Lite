# InteriorGS 空间问答：进度

本文件是这项工作的交接记录。每完成一个部分就更新，并提交、推送到 GitHub main。
任务说明：用户 2026-10-14 的提示词（构造式出题、样题审核网页），2026-10-15 的第二轮提示词（对报告第九节的决定；A1–A4；交回标注后的 B1–B5）。

## 1. 做完了什么

**第一轮（2026-10-14）**，详见 [`REPORT_20261014.md`](REPORT_20261014.md)：
- 出题代码 `data_gen/interiorgs_qa/`（单图 S1–S7、多图 M1–M11、视频 V1–V6）；样题 240 道（单图 70、多图 110、视频 60）；
  CPU 构造式普查、指代普查、统计扩量运行；盲答基线发现的 5 处答案偏差已修正并重跑。
- COVERAGE.md 第二版（92 个子任务编号全部有着落）。

**第二轮（2026-10-15）**，详见 [`REPORT_20261015.md`](REPORT_20261015.md)：
- 标注网页发布到 claude.ai：https://claude.ai/artifact/5RY1E9BeCYwSvaaesoAgKJ （240 道样题，标注存在页面的数据库里，
  集合 `labels`，每题一个文档；用 ArtifactData 读取）。生成：`site_artifact.py` + `site_artifact_template.html`。
- 用户决定落实：试验 F、G 和颜色试做不做，13 个子任务、BLINK 朝向类关系、MMSI 颜色改为"不做（用户决定）"
  （COVERAGE.md、REPORT_20261014.md 已改）；每张图记录俯仰角（`extra.pitch_deg`）。
- A1 训练格式：`export_lf.py`（Act-SFT 同一 LLaMA-Factory 格式，视频按多轮对话），`lf_tokens.py`（真实预处理计数）；
  单卡训练试跑（7B 前向反向、3B 全套配置）通过。产物 `exps/interiorgs_qa_lf_20261015/`。
- A2 评测：`evaluate.py`（vLLM 推理、scoring.py 打分、按组 / benchmark / 措辞组汇总，带题数和 95% 区间）；
  基础模型在 240 道样题上跑完：`exps/interiorgs_qa_eval_20261015/base_sample/`。
- A3 全量流水线：渲染前的同名物体 CPU 预筛（`views.same_name_visible`）、题库模式（`pipeline bank`，分片、断点续跑，
  不做答案平衡，物体上限）、`pipeline bank-merge`、抽样脚本 `sample.py`（预算、组间配比、答案上限、物体上限，
  训练集和评测集一起抽）；配置 `configs/bank_plan_20261015.json`、`configs/sampling_plan_20261015.json`。
  5 个 train 场景试跑：生成 → 抽样 → 导出 → 真实预处理全部通过（`exps/interiorgs_qa_bank_trial_20261015/`；
  加快预筛后的计时重跑 `exps/interiorgs_qa_bank_trial_20261015b/`，平均每场景 265 秒）。
- A4 评测题数估计：`estimate_eval.py`，结果在报告第五节。

- 2026-10-15 用户对第二轮报告第六节 5 项的决定已写进代码和文档（REPORT_20261015.md 第六节开头）：
  原格式题用 `scripts/easi_eval.py` 实际发出的提示（`qa.py` 第 2 版排版，`render()` 转换旧题），留出措辞统一结尾，
  训练答案按说明写（`answer_target()`）；稀少组可出自视频场景（`rare_groups_everywhere`）；物体上限 6/2/3；
  题库每场景每子任务 3 道；评测主结果按 benchmark 官方方式合并并附区间，M6、M10、M11、V6 只进附录（`evaluate.py`）。
  存储：`/mnt/umm` 剩 8.2 TB（共 100 TB，已用 92%），题库约 90 GB，全量前再查一次。
- 2026-10-15 标注中（用户已标 62 道）：用户决定平行 / 垂直题（3D-11，S7 组）全都不要。标注网页更新到第 3 版：去掉 S7 的 10 道
  （4c41759d3d076b4b、62d4c9537ccc6b35、71a8a1c2d2863896、7e5c38135e354c65、8834226a8c063faa、8d64eb9ee8ce2cc3、
  9abe2fd3ad2e7c06、b32a59035c0e8dfd、ba7f3a9f7ba47903、ef58af7dda2a9203），共 230 道；其他题、图片包和已有标注都没动，
  页面按题号记住当前位置，用户原来停在 S7 时会接到 M1 第一题。已标的 3 道 S7（前三个题号）记录保留，B1 汇总时不用。
  题库计划去掉 S7；COVERAGE.md 里 3D-11 改为"不做（用户决定）"（能做由 52 项变为 51 项）。
- 2026-10-19 标注中（数据库里已有 87 条标注，含 3 条 S7）：用户决定先用一个已知方向定指南针、再问另一个方向的 MMSI 题
  （"P 在 Q 的东北方，R 在 Q 的哪个方向"）全都不要：物体–物体（M2）、物体–区域、区域–区域（M3）。标注网页更新到第 4 版：去掉 9 道
  （物体–物体 6e6c4018e8eb0def、b36ee42843afcb81、c557da4b021db112；物体–区域 1d9cd1808b1dfbda、29d45aa3014f8196、
  2f2d79b45a52f614；区域–区域 12d6d2ac2668a733、2f2fde8c597353ac、836911d19a9ce5f2），共 221 道；只换了 M2、M3 两个图片包，
  其他题和已有标注都没动，用户停在物体–区域第一题时会接到 M4 第一题。已标的 3 道物体–物体（前三个题号）记录保留，B1 汇总时不用。
  题库计划去掉这三个子任务（`site_20261014.sh` 也排除它们）；COVERAGE.md 里改为"不做（用户决定）"。
- 2026-10-19 标注中：用户反馈图上画了圆圈和字母的题切到 256 后字母看不清。原因：SCO 节点没有系统 DejaVu 字体，标记退回 PIL 的 10 像素默认字体；
  256 图又是把 512 标记图整体缩小。修正：`draw._font` 先用共享环境里 matplotlib 自带的 DejaVuSans-Bold；`ViewBank.save_marked` 先缩到 256 再画标记，
  尺寸见 `draw.mark_style`（512：圆半径 10、字号 23；256：半径 7、字号 18）；标签避开图边、别的圆圈和标签。
  已出的题用 `redraw_marks.py` 按题目记录里的点和框重画（样题两个 run：23 道、33 张图的 512 和 256 都重画，旧图备份在
  `exps/interiorgs_qa_sample_20261014/marks_backup_20261019/`）。网页第 5 版只换了 S3、S6、M4 三个图片包，题号和标注不变。
  涉及的 23 道：S3 相对深度 659ff1b8764f896e、c5bfe5dc8db20245、db9a310711f43c36；S6 选框 17063f0398ba64ea、2d275b4fd33d551a、324bdb8ae9a3ad0f、49b3b3bbcdbc60d0、73a5510e95a30560、7afeebb34649763a、89688634c5177620、b6ad7bccac308708、b9b71639419b74e4、e31708d5ec93f4a3；M4 点对应 0de912d6d2d13a84、1216ea829f811aa3、43d146d6cceb3cd8、4d915fc478917c39、52f63cdfc6109d11、8f2673a087d185e0、a871d34c5956e657、b2615a011be56b8f、d4d6da01eab15247、e662f8dba4a825c9。
  其中 15 道用户在重画前已标（S3、S6 的 13 道和 M4 的 0de912d6d2d13a84、1216ea829f811aa3），B1 汇总时这些题的"256 下能否认出"按旧图记录，
  单独列出。题库试跑目录（`exps/interiorgs_qa_bank_trial_20261015*`）里的标记图没有重画，全量生成用新代码。

- 2026-10-27 用户结束标注："检查了快 100 道都没有问题……认为生成没问题"，要求做 scaling，生成最终版训练集和测试集，
  并且和转换过来的 benchmark task 严格对应。B1：数据库里网页第 5 版的题有 87 道已标（S1–S6、M1–M3 全部，M4 6 道），
  标出问题的 3 道都是单题：M4 1216ea829f811aa3（256 下字母看不清，已在第 5 版修好）、S3 2fc3520ccdbda7fb（画面里有两处花和花瓶，
  有歧义）、S5 ae1a2f9ce61d6fa0（一处花在墙后，框落在墙上，答案判错）。没有整组性问题，按用户决定不做 B2、B3 的校准，阈值不变。
- 2026-10-27 benchmark 任务严格对应：`tasks.py` 列出有题的 46 个任务（COVERAGE.md 里能做的 48 项去掉 MindCube 的两个设置标签），
  每个任务记下 benchmark 和 benchmark 自己的类别名（VSI `question_type`、MMSI `question_type`（lmms-eval 写法）、MindCube 题号前缀、
  ViewSpatial `question_type`、EmbSpatial `relation`（下方是 `under`）、3DSRBench `category`、BLINK 子集），取自缓存的 benchmark 数据和
  lmms-eval 任务代码。`tasks.check` 要求每道题只属于一个任务，且 benchmark、组、原格式排版、答案类型都和任务一致；
  `sample.py` 遇到任何不一致就停，输出的每一行带上任务；`export_lf.py` 的审计行带任务；`export_test.py` 按 benchmark 分文件写评测集，
  每题带 benchmark 的类别；`evaluate.py` 另按 benchmark 自己的类别出表。`docs/interiorgs_qa/audit/check_tasks.py` 核对任务表和
  COVERAGE 判断一致，并检查已生成的题：样题、试跑题库和新题库已完成的场景共 880 道全部通过（只有用户去掉的四个任务的旧题不计）。
- 2026-10-27 全量题库开始生成（B4）：`exps/interiorgs_qa_bank_20261027/`，1000 个场景（test 100、val 100、train 800，按这个顺序，
  场景表 `configs/bank_scenes_20261027.txt`），题库计划 `bank_plan_20261015.json`。资源池忙：8 个分片只有 3 个拿到卡，
  另外 5 个排队的删掉，改成 5 个"认领"模式的任务（`pipeline bank --claim`：共用一张场景表，谁先拿到卡谁领下一个没人做的场景，
  `scenes/<场景>/claim.json`，3 小时没完成的认领可被接手），共用分片 3–7 的 625 个场景（`configs/bank_scenes_20261027_claim.txt`）。
  存储：`/mnt/umm` 剩 21 TB。

- 2026-10-28 全量题库完成：1000 个场景全部完成、0 个出错，59,095 道题（train 46,948、val 6,001、test 6,146），89 GB，
  平均每场景 171 秒；`check_tasks.py` 全部通过。但 VSI-direction-hard 一道都没有：每段视频只出一道题，取剩余配额最多的子任务，
  配额都是 999 时总取列表里第一个（easy），而一个场景只有 1–2 段视频；同样原因相对距离（46 道训练题）和房间面积偏少。
  修正（`video.py` 的 `each_subtask`）：每段视频对每个还有配额的子任务各出一道（VSI 本来就对一段视频问多种题）。
  只重跑视频组 V1–V6：`exps/interiorgs_qa_bank_20261028_video/`（8 个认领任务 `bankvideo_20261028_c0`–`c7`，1000 个场景）；
  抽样时用 `sample.py --exclude` 去掉旧题库的 V1–V6。前 28 个场景：方向题 easy 9、medium 11、hard 10，每段视频 1–9 道题。
  最终抽样用 `configs/sampling_plan_20261028_final.json`：同 `sampling_plan_20261015.json`，只是训练预算按 Act-SFT 的
  24,564 次图片输入整数取用，不再按"有题的 train 场景占比"缩小（那是试跑用的设置）。

- 2026-10-28 视频组重跑完成：1000 个场景 0 出错，6,518 道视频题（第一遍 3,661），12.9 张卡·小时，19 GB；VSI 10 个任务都有题
  （方向 hard 训练题库 502）。
- 2026-10-28 B5 完成，最终数据 `exps/interiorgs_qa_final_20261028/`，报告 [`REPORT_20261028.md`](REPORT_20261028.md)：
  训练 12,238 道题（10,879 条记录，图片输入 23,841 次，视频对话 262 段），val 5,412 道，test 5,590 道（场景 785 / 99 / 99）；
  46 个任务在训练和 test 里都有题；11 个任务 test 少于 30 道（MindCube Around / Among 为主）；val 里 MindCube around-q2 为 0。
  过程中又修了两处：抽样改为各任务轮流取题（`sample.interleave`；原来按任务依次取，Among 四种题共用 5 个物体，物体上限被前面的任务
  用光，among-q5 训练题为 0）；训练文件里视频对话均匀插入（`export_lf.spread_videos`；原来视频都在末尾，LLaMA-Factory 分块推断列类型
  时报 "Couldn't cast … to null"，训练会读不进来）。实测 token（`lf_tokens.py`，无截断）：256 每轮 251 万（视觉 159 万），
  512 每轮 728 万（视觉 637 万）。

- 2026-10-28 训练和 test 集评测完成（报告第 6 节）：训练 pt-vkamp7w5 38 分钟（510 步，loss 2.11→0.04，val loss 0.18 左右不变），
  模型 `exps/interiorgs_qa_final_20261028/lf_256/model`；评测 `exps/interiorgs_qa_eval_20261028/`（基座 pt-mw58pnxb、QA-SFT pt-a2qmufge），
  对照表 `compare_base_vs_sft256_test.md`（新模块 `compare.py`）。原格式 VSI 0.32→0.48，其余 benchmark +18 到 +47 个百分点；
  VSI 相对方向没有学好（easy 原格式 53%→39%）。

## 2. 正在运行什么

没有（2026-10-30 第 1、2 轮 checkpoint 的 EASI-8 完成，报告第 8 节：EASI-8 平均 46.2 / 48.1 / 48.3；SITE 在第 1 轮就降 4.7 分，
少训几轮不能解决；`easi_job.py` 对 SITE 视频默认打开 `--stack-dump`（不带的 5 次全停、带的 5 次全完成））。下面是当时的记录。

用户 2026-10-29 要求：在 EASI-8 上评测 QA-SFT 的 checkpoint-170（第 1 轮）和 checkpoint-340（第 2 轮），看少训几轮能不能保住提高、减少 SITE 的退步。
18 个单卡任务（2 个 checkpoint × 9 个 lmms-eval 任务，SITE 拆成图片和视频），`exps/interiorgs_qa_jobs/easi1_qasft_ck{170,340}_*_20261030`，
结果在 `exps/interiorgs_qa_easi_20261028/qasft256_ck{170,340}/`。同一 checkpoint 的任务按 `--start-delay` 错开启动秒数，避免结果文件同名覆盖。
预计约 3 小时（SITE 视频和 VSI 最慢）。

以下是 EASI-8 三模型评测时的记录（2026-10-29 已完成）。

用户 2026-10-28 要求：用 `scripts/easi_eval.py` 在 8 个原 benchmark（EASI-8）上评测 QA-SFT，提交成 SCO 任务，和基座、Act-SFT 对比。

- 新的提交方式：`python -m data_gen.interiorgs_qa.sco prepare --job JOB --gpus 8 -- easi_job --name NAME --model DIR --out OUT`
  （`easi_job.py` 调用不改动的 `scripts/easi_eval.py`，离线数据缓存同 `sco_active_spatial_eval_entry.sh`；8 张卡数据并行）。
- 三个模型，同一份代码、同一天跑，结果在 `exps/interiorgs_qa_easi_20261028/<名字>/easi_results.json`：
  QA-SFT `qasft256_e3`（`lf_256/model`，作业 pt-cmds7w2p）；基座 `base7b`（`qwen25vl7b_pretrained_cc594898`，pt-bm5obgyk）；
  Act-SFT `actsft_r11_e9`（第十一轮 SFT 第 9 个 epoch `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386`，pt-t1hy337m）。
  选 Act-SFT 的理由：和训练预算对齐的第八轮 3 个 epoch 模型（`interiorgs_v10_train_20261014/lf/model`）已被清理删除，只剩训练数据；
  现存的是第十轮第 5 个 epoch（checkpoint-770）和第十一轮第 9 个 epoch（checkpoint-1386，v10 线目前最好的 SFT、后续 RL 的起点）。
  基座 8 月有过一次 EASI 结果（`easi_results/qwen_pretrain_baseline`，同一个 cc594898），这次在同一份代码下重跑，旧结果只作核对。
- 冒烟测试：单卡、每个 benchmark 3 道题（pt-46lie0ib，`exps/interiorgs_qa_easi_20261028_smoke`）。
- 2026-10-29 进展：
  - 冒烟测试发现 5 个 benchmark（MMSI、ViewSpatial、SITE、BLINK、3DSRBench）的数据不在离线缓存里（缓存里有同一版本的处理后数据，
    原始 Hub 快照被清掉了）。经跳板机代理下载到 `~/.cache/huggingface/hub`（共约 86 GB，SITE 65 GB；4 线程、nice 19），已全部下完。
  - 8 卡任务在 SITE 缺数据时出错后挂住（一个进程报错，其余 7 个在 NCCL 里一直等），QA-SFT 从 19:27、基座从 04:02 空占 8 张卡；
    Act-SFT 的 8 卡任务一直 SUSPENDED。三个都删了（已完成的 VSI、MMSI、MindCube、ViewSpatial 保留）。
  - 按用户建议改成单卡任务（每个模型 × 每个 benchmark 一个，zoetrope 空卡上立即开跑）：13 个全部成功。
  - SITE 的图片和视频要从 15 个 zip 解到 `~/.cache/huggingface/sitebench/data`；那个目录 8 月被不完整地解过一次，lmms-eval 见目录
    存在就不再解，所以 SITE 一直缺文件。`site_extract.py`（单卡 SCO 任务，目录是 root 的）把缺的文件补全；第一次因多进程同时建
    目录报错，已修，重跑中（pt-x1o2gcb4）。完成后三个模型的 SITE 各一个单卡任务。
  - 同一模型的几个任务同一秒启动时，lmms-eval 的结果文件同名互相覆盖：Act-SFT 的 MindCube 结果被 BLINK 覆盖，单独重跑（pt-xfrj7xgd）。
  - Act-SFT 在 MindCube-Tiny 只有 4.1%：它答 `<answer>选项文字</answer>`，官方提取要选项字母，几乎全判错；其他 benchmark 格式正常。
  - 对照表 `exps/interiorgs_qa_easi_20261028/compare_partial.md`（`easi_compare.py`，用 EASI 自己的打分代码）。
- 2026-10-29 EASI-8 完成（报告第 7 节）：SITE 视频在全量上正常跑（约 3 小时 / 模型；此前三次全量停在开头的原因没有复现）。
  EASI-8 平均 QA-SFT 48.3、基座 43.1（与 8 月逐项相同）、Act-SFT 37.3；QA-SFT 在 VSI +11.4、MMSI +6.0、MindCube +19.7、ViewSpatial +12.2，
  BLINK / 3DSRBench / EmbSpatial 基本不变，SITE −6.2。Act-SFT 的 MindCube 4.1 是格式问题（按文字匹配 38.1）。对照表 `exps/interiorgs_qa_easi_20261028/compare_easi8.md`。

以下是 2026-10-28 训练时的记录。

用户 2026-10-28 要求：把 lf_256 的训练提交成 SCO 任务（Act-SFT 配方，8 卡），训完用 test 集评测。

- 训练：`exps/interiorgs_qa_train_20261028`（`job_sco.py prepare --gpus 8 --pool zoetrope -- sft …/lf_256`，作业 pt-vkamp7w5，
  10:09 UTC 提交）。配方与 Act-SFT 只差数据路径（另两项像素上限在 256 下等于默认值）；模型写到 `lf_256/model`。
  Act-SFT 自己的训练在 8 卡上约 38 分钟；这里 val 4,956 条（Act-SFT 1,358 条），每 200 步评一次，预计 40–70 分钟。
- 基座模型在同一 test 集上的评测（对照用）：`exps/interiorgs_qa_jobs/eval_base_test_20261028`（单卡，作业 pt-mw58pnxb），
  结果 `exps/interiorgs_qa_eval_20261028/base_test_256/`。
- 训练完成后：用 `data_gen.interiorgs_qa.sco` 提交单卡评测，模型 `lf_256/model`，`--selection …/selection/test.jsonl --res 256
  --families orig,h1`，结果 `exps/interiorgs_qa_eval_20261028/sft256_test_256/`。

## 3. 下一步

**用户决定（2026-10-28，报告第 5 节）**：训练用 256，和 Act-SFT 一致（`exps/interiorgs_qa_final_20261028/lf_256/train.yaml`）；
更大的训练集等主对照结果出来再说，现在不导出（题库够抽约 3.9 倍；导出时抽样程序要加"视频场景也出单图、多图题"的选项）。
下一步由用户安排：训练（Act-SFT 配方，8 卡，用 SCO 任务）；评测 `evaluate.py --selection …/selection/test.jsonl --res 256`（也走 SCO）。
跳板机上不跑重的 CPU 计算，一律提交成 SCO 任务（用户 2026-10-28）。

以下 B1–B5 已完成，留作记录。

**B5（全量完成后）**：`bank-merge` → `sample.py`（`sampling_plan_20261015.json`；训练集按 Act-SFT 每轮图片输入次数 24,564 抽，
评测集是 val / test 场景里的全部题，受答案上限和物体上限约束）→ `export_lf.py`（256 和 512 两份训练数据，带 val）→
`export_test.py`（val / test 按 benchmark 分文件）→ `lf_tokens.py` 核对 token → 报告：每个 benchmark 任务在题库、训练、val、test
里的题数，test 没有题的任务单列。

以下是标注前的计划，留作记录。

**标注期间的规矩（用户 2026-10-15）**：不改标注网页、不改这 240 道样题。用户中途报告某组有整组性问题时：改代码，
只重新生成那一组的样题，替换网页里那一组（其他组和已有标注不动），并告诉用户替换了哪几道（旧题号 → 新题号）。

**用户交回标注以后（B1–B5，计划）：**

- **B1 读标注**：只算网页第 4 版上的 221 道（去掉的 S7 和三个指南针子任务的旧标注不用）；
  每组按五项（能否认出物体、能否认出同一物体、标准答案是否正确、有没有歧义、256 下能否认出）分别算合格率，
  附题数和 95% 区间；列出整组性的问题（某组多数题同一项不合格）和集中出错的物体类别。
- **B2 校准和修正**：
  - M8（Around）的遮挡阈值：按"标准答案是否正确"和"有没有歧义"两项，比较远处物体可见比例和被挡比例不同的题，定阈值；
  - "能认出"的各项阈值（可见部分至少占画面 0.4%、露出至少一半、投影至少 60–70% 在画面内）：按"能否认出"一项，
    看哪些阈值附近的题被判不合格，调整；
  - 两条放宽规则（"只出现在一张图里"允许 0.2% 边角、M3 共享 1 个物体）：按"能否认出同一物体"一项定去留；
  - 标错的物体类别：加进不出题清单或改名；
  - 分辨率：按"256 下能否认出"一项，定训练用 256 还是 512。
- **B3 合格率低的组**：改代码，只重新渲染这些组的样题，发布到同一个网页给用户复核。
- **B4 用户确认后全量生成**：800 个 train、100 个 val、100 个 test 场景，`pipeline bank` 分片到多张卡（每片一个
  SCO 单卡任务），完成后 `bank-merge`。
- **B5 抽样、导出、汇报**：按第一节的决定（`sampling_plan`）抽出训练集和评测集，`export_lf.py` 导出，汇报每组题数、
  答案分布、图片输入次数和 token 数（`lf_tokens.py`）。

## 4. 约定

- 代码：`data_gen/interiorgs_qa/`（进 Git）。文档：`docs/interiorgs_qa/`（进 Git）。
- 运行产物（图片、深度、网页、题库、导出数据）：`exps/interiorgs_qa_*`（git 忽略）。
- 同一目录里另一条工作线（v10 训练）也在提交；本任务只用 `git commit -- <自己的路径>` 提交自己的文件，不碰别人的未提交改动；
  不改 Act-SFT 的代码和数据（只调用 `job_sco.py` 和 `sco_interiorgs_v10_train.sh` 的单卡试跑模式）。
- 共享缓存 `~/.cache/huggingface/datasets` 里有 root 拥有的文件：本任务的 CPU 预处理用 `HF_DATASETS_CACHE=/tmp/psv/hf_datasets_cache`。
