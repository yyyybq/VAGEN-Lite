# 数据契约：P1 转移题和混合 SFT 的行

本文件规定 P1 题库（`data_gen/transition_qa/`）和混合 SFT 构建器（`data_gen/sft_mixture/`）写出的每一行有哪些字段、取什么值、按什么规则平衡和截断，以及"没出题"的原因码。代码和本文件不一致时以代码为准，并改本文件。题型的逐个说明见 `P1_QTYPES.md`，数字见 `P1_STATS.md`。

**要点**

- **一行 = 一道题**（P1、P0），或**一整条动作轨迹**（P2）。每行都是 LLaMA-Factory 的 sharegpt 格式（`messages` + `images`），另带元数据字段。训练只读 `messages`、`images`、`videos`。
- **场景不串**：P1 的 train 行只来自 `split_manifest.json` 的 train 场景，test 和 probe 行只来自 val 场景，test 场景一概不用。构建时逐行检查，有一行不符就拒绝写文件。
- **答案平衡在先，比例上限在后**：先按题类平衡答案，再按三类题的规则截断，最后单场景上限。截断都按（题类、答案、是否变化）分层抽，平衡不会被破坏。
- **"P1 / P2" 和 "v10_p1 / v10_p2" 无关**：前者是数据池（转移题 / 动作轨迹），后者是 RL 修正包的名字（`data_gen/interiorgs_v10/configs/rl_standards.json`）。

## 1. 术语

| 术语 | 含义 |
| --- | --- |
| 池（pool） | 混合 SFT 的数据来源。**P0**：仿 8 个 benchmark 的 QA 样题（`exps/interiorgs_qa_lf_20261015/sample_256`，230 道，177 条记录）；**P1**：本题库的转移题；**P2**：第十一轮正式 SFT 的动作轨迹（`interiorgs_v10_train_r11_20261016/lf_formal`，train 9,802 条） |
| 任务族（family） | 题目所属任务的类型，6 个：`projective_relations` 投影（A 在 B 左边 / 右边）、`equidistance` 等距、`absolute_positioning` 绝对距离、`occlusion_alignment` 遮挡、`fov_inclusion` 视野（两个都在画面里）、`screen_occupancy` 屏幕占比 |
| 大类（qclass） | `state` 状态题（问当前画面）、`forward` 前向题（问"做了某个动作之后"）、`inverse` 逆向题（给前后两张图，问做了什么动作） |
| 题类（qtype） | 15 个，见第 4 节；名字 = 量 + `_now` / `_after` / `_between` |
| 状态 | 轨迹上的一个相机位姿和它的渲染图（SFT 轨迹每回合开始一张，终点一张） |
| 锚点状态 | 拿来出题的状态。2026-10-19 起：**每条轨迹的每个不同的帧**（同一帧出现两次时取后一次，那一回合是真正移动的）；2026-10-18 的题库每条轨迹最多 3 个（终点、起点、随机一个中间帧） |
| 内部状态 | 目标区域内部、离成功边界较远的成功位姿（`interior.py`），另渲染，只出 `goal_now`（答 Yes） |
| 门 | 成功判据里的一项条件；全部通过才算成功（环境的 `score_with_visibility`） |
| 变化（changed） | 前向题里，被问的量在动作前后变化超过门槛（第 6.2 节），或答案变了 |
| 翻转（flip） | 前向题里，动作后的答案和"现在"的答案不同；"现在"那一侧不出题时为空 |

## 2. 文件和划分

题库目录（2026-10-19 起）：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261019/`

| 文件 | 内容 | 划分 |
| --- | --- | --- |
| `p1_train.jsonl` | 训练题 | 只有 train 场景（745 个） |
| `p1_test.jsonl` | 测试题；混合 SFT 的 P1 验证集从这里分层抽 300 道 | 只有 val 场景（97 个），不含探针题 |
| `p1_probe.jsonl` | 探针集（诊断 3）：每个题类 40 道 | 只有 val 场景；第十二轮 v2 验证集的场景（就是全部 97 个 val 场景） |
| `stats.json` | 构建统计：候选数、每一步去掉多少、上限 | |
| `probe_stats.json` | 探针集统计：每个题类的候选数、按答案分的数量、任务数 | |
| `config.json` | 出题门槛（`questions.DEFAULT_CONFIG`） | |
| `raw/<场景>.jsonl` / `.json` | 每个场景的候选题和"没出题"的计数（`funnel`）、一致性检查（`checks`） | |
| `raw/<场景>__oa15.*` | 第十二轮 v2 验证集新增的 15 道遮挡题的候选题（`extra_oa15/` 渲染的最短路径帧） | val |
| `raw/zz_interior.*` | 内部状态的 `goal_now` 候选题（图在 2026-10-18 题库的 `interior/`） | train |

- `split` 字段只取 `train` 或 `val`，等于该行场景在 `interiorgs_v10_r11_20261016/sft/split_manifest.json` 里的划分。
- `generate build` 拒绝：train 行的场景不在 train 场景里、val 行的场景不在 val 场景里、train 和 test / probe 有共同的场景。
- 混合 SFT 构建器（`sft_mixture/build.py`）再查一次：训练行来自 train 场景，验证行来自 val 场景（P0 样题是配置里写明的例外：它的验证行来自 train 场景），同一条不能既训练又验证；每张图都必须是存在的绝对路径。

## 3. 行格式

### 3.1 P1 题库的行（`p1_*.jsonl`）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `messages` | list | `[{"role": "user", "content": 题目}, {"role": "assistant", "content": 答案}]`。题目以 `<image>\n`（逆向题 `First image: <image>\nSecond image: <image>\n`）开头；答案是选项字母（`A`–`D`）或 `Yes` / `No`，**只有答案，没有思考文字** |
| `images` | list[str] | 绝对路径，数量和 `<image>` 标记相同 |
| `sample_id` | str | `p1_` + 12 位十六进制，由（任务、状态、题类、动作、次数）决定；同一道题在重建后 id 不变 |
| `pool` | str | 固定 `P1` |
| `qclass` | str | `state` / `forward` / `inverse` |
| `qtype` | str | 第 4 节的 15 个之一 |
| `family` | str | 第 1 节的 6 个之一（逆向题、goal 题也带所属任务的族） |
| `task_id` | str | 出题用的任务（`v10_…`），即 SFT 轨迹的 `source_task_id` |
| `scene` | str | InteriorGS 场景号 |
| `split` | str | `train` / `val` |
| `state_id` | str | 图的状态：`s<k>`（轨迹第 k 个原子动作之后）、`x<k>`（第十一轮"被挡后先离开"的新路径）、`i0`（内部状态） |
| `state_role` | str | `start` / `middle` / `terminal` / `interior` |
| `state_index` | int | 状态在轨迹图片序列里的位置；内部状态为 -1 |
| `state_id_2` | str | 逆向题的第二张图的状态；其他题为空 |
| `options` | list[str] | 选项文字，顺序同题目里的 (A)(B)…；是非题为空 |
| `answer` | str | 正确选项字母或 `Yes` / `No`，等于 `messages[1].content` |
| `answer_value` | str | 答案的含义：`yes` / `no`、`left` / `right`、`a` / `b`（哪个更近）、`1.5`（米）、`0.4`（画面高度比例）、`full` / `partial` / `none`、`both` / `only_a` / `only_b` / `neither`、`turn_leftx2`（动作 × 次数） |
| `action`、`k` | str、int | 前向题、逆向题的动作和次数（0.3 m、20° 一次）；状态题为空、0 |
| `changed` | bool / null | 前向题：被问的量是否变化；其他为 null |
| `flip` | bool / null | 前向题：答案是否和现在不同；现在不出题时为 null |
| `audit` | str（JSON） | 出题时测得的量：`quantity`（状态题）、`quantity_before` / `quantity_after`（前向题）、`potential` / `failed_gates`（goal 题）、`interior`（内部状态的偏移）。核对页的"过滤检查"就是从它读的 |

### 3.2 混合 SFT 的行（`exps/sft_mixture_*/<组>/P<n>_{train,val}.jsonl`）

所有池同一个格式（`build.row`）：

| 字段 | 含义 |
| --- | --- |
| `messages`、`images`、`videos` | 原样复制自来源，不合并、不改写（P2 的对话和图与主线 `trajectory_train.jsonl` 逐条相同、顺序相同） |
| `pool` | `P0` / `P1` / `P2` |
| `source_id` | P0：`p0_<题号>`；P1：`sample_id`；P2：任务号 |
| `scene` | 场景号（P2 从帧文件名找到任务，再从第十一轮 `rows.jsonl` 找场景） |
| `stratum` | 分层抽样的层：P0 题组（`S1`–`S8`），P1 题类，P2 任务族 |
| `meta` | JSON：P0 `{groups, kind, qids, questions}`，P1 `{family, qclass, qtype, task_id, questions}`，P2 `{family, task_id, turns}` |

LLaMA-Factory 只读 `messages`、`images`、`videos`（`dataset_info.json` 的 `columns`）；每个池单独一个数据集，`eval_on_each_dataset: true` 给出每个池的验证损失。

## 4. 题类

| 大类 | 题类 | 族 | 选项 | 随机猜 |
| --- | --- | --- | --- | --- |
| 状态 | `goal_now` | 全部 | Yes / No | 50% |
| 状态 | `left_right_now` | 投影 | 2（左 / 右，顺序固定） | 50% |
| 状态 | `closer_now` | 等距 | 2（两个物体，位置轮流） | 50% |
| 状态 | `distance_bin_now` | 绝对距离 | 4（0.5 m 一档，从小到大） | 25% |
| 状态 | `occluded_now` | 遮挡 | 3（完全 / 部分 / 没有，位置轮流） | 33% |
| 状态 | `both_in_view_now` | 视野 | 4（都在 / 只有 A / 只有 B / 都不在，位置轮流） | 25% |
| 状态 | `height_bin_now` | 屏幕占比 | 4（10% 一档，从小到大） | 25% |
| 前向 | 上面 7 个的 `_after` 版本 | 同上 | 同上 | 同上 |
| 逆向 | `action_between` | 全部 | 4（动作 × 次数，位置轮流） | 25% |

每个题类的题目原文、答案函数、出题门槛见 `P1_QTYPES.md`。

## 5. 答案平衡

按（划分、题类）分别做，顺序是：

1. **选项位置无含义的题类**（`closer`、`occluded`、`both_in_view`、`action_between`）：先按答案值平衡（下一条），文件写出前把正确选项放到轮流的位置（按 `sample_id` 排序后 A、B、C、D 循环），字母完全均匀。
2. **答案值平衡**：每个答案组最多是最少那组的 **1.15 倍**（两个选项时偏差 ≤ 3.5 个百分点）。是非题、左右题、距离和高度题按字母分组；遮挡、视野按答案值分组；占比不到 5% 的答案值不参与（整组保留，不拿它压别的组）。
3. **前向题的两条上限**和答案平衡交替做，直到都不再去掉题（最多 8 轮），见第 6.1 节。
4. **字母上限**：选项位置有含义的题类（是非、左右、距离、高度），每个字母最多是最少者的 **1.10 倍**（两个选项 ≤ 2.4 个百分点，四个 ≤ 1.8）。
   - 2026-10-19 起这一步放在三类题的上限**之前**，所以上限在写出的文件上严格成立；之后的分层截断按字母分层，字母占比不变。
   - 2026-10-18 的题库把它放在最后，结果逆向题实际占 15.3%。
   - 探针集不做这一步，它按第 6.4 节轮流抽。

## 6. 上限和比例

### 6.1 前向题（每个题类单独）

- **没变化的题** ≤ 10%（被问的量在动作前后几乎不变，例如转身后到物体的距离）。
- **答案翻转的比例**控制在 35%–65%，只在"翻转"和"不翻转"都至少占 10% 的题类上做；`left_right_after` 几乎从不翻转（转身不改变两个物体的左右顺序），不做，比例照实报告。
- 每个锚点状态、每个题类最多 3 道（不同的动作 × 次数，分散到不同的（答案、是否变化）组）；`goal_after` 的"否"每个锚点最多检查 2 个。

### 6.2 "变化"的门槛

| 量 | 门槛 |
| --- | --- |
| 左右（横坐标差） | 10 px |
| 哪个更近（距离差） | 0.10 m |
| 距离 | 0.15 m |
| 画面高度比例 | 0.03 |
| 遮挡（被挡比例或在画面内比例） | 0.10 |
| 视野（在画面内比例） | 0.10 |
| goal（势能） | 0.02，或成功与否变了 |

### 6.3 三类题的规则（用户 2026-10-19）

| 规则 | 现在 | 2026-10-18 的题库 |
| --- | --- | --- |
| 状态题 | **全部保留**（答案平衡之后） | 按 50 / 35 / 15 抽 |
| `goal_now` | 不设上限（它是状态题） | 最多占状态题一半 |
| 前向题 | **≤ 状态题的 1.2 倍** | 按 50 / 35 / 15 抽 |
| `goal_after` | 最多占前向题一半（先做） | 同左 |
| 逆向题 | **≤ 题库的 15%** | 按 50 / 35 / 15 抽 |
| 单个场景 | ≤ 8% | 同左 |
| 锚点状态 | 每条轨迹全部可用帧 | 每条轨迹最多 3 个 |

截断都按（题类、答案、是否变化）分层抽：每层取相同的比例，剩下的名额给余数最大的层，所以上限是硬的（不会多一道）。

### 6.4 探针集（诊断 3，用户 2026-10-19）

- **每个题类 40 道**，15 个题类共 600 道。用户说的"500 道、每个题类至少 40 道"在 15 个题类下做不到（至少 600），按每类 40 道做。
- 候选：第十二轮 v2 验证集的场景（97 个 val 场景）里的全部候选题，加 15 道新增遮挡题的最短路径帧。
- 同一题类里：答案轮流（位置无含义的题类按答案值轮流，之后字母循环；其他按字母轮流）；前向题再按"翻转 / 不翻转"轮流；没变化的前向题最多 4 道（10%）；每轮先从候选最少的答案组抽；同一答案组里任务族轮流、同一族里任务轮流；整个题类里先抽还没被抽过的任务。
- 探针题不进测试集（`build` 按 `sample_id` 去掉），两份不重叠。
- 不够 40 道的题类照实报告，不补。

## 7. "没出题"的原因码

候选阶段（`raw`）每个状态、每个题类的判定结果计在 `raw/<场景>.json` 的 `funnel` 里，键是 `<题类>:<原因>`。

| 原因码 | 含义 | 用在 | 门槛 |
| --- | --- | --- | --- |
| `emitted` | 出题了 | 全部 | |
| `emitted_interior` | 内部状态出了 `goal_now` | `goal_now` | |
| `near_boundary_yes` / `near_boundary_no` | 在成功边界上：位姿挪 ±0.05 m、转 ±3° 后结论会变（答"是" / "否"的） | `goal_now`、`goal_after` | |
| `near_boundary` | 测得的量离判定阈值太近 | 族题 | 左右：横坐标差 < 20 px；哪个更近：距离差 < 较远者的 5%；距离：离最近选项 > 0.15 m；高度：离最近选项 > 0.03；遮挡：被挡比例在 0.05–0.25 或 0.70–0.85；视野：在画面内比例在 0.30–0.95，或在画面里但看不清 |
| `not_visible` | 被问的物体在当前画面里看不清（不满足 face_visible_v3） | 状态题 | 包围盒至少一个面露出 ≥ 40%，露出部分 ≥ 画面 0.3% |
| `not_visible_now` | 前向题：被问的物体在当前画面（模型看到的那张）里看不清 | 前向题 | 同上 |
| `behind_camera` | 物体中心在相机后方，或 8 个角不全在前方（高度题） | 左右、高度 | |
| `out_of_frame` | 动作后物体在画面内的比例 < 0.5 | `left_right_after` | 0.5 |
| `beyond_45deg` | B→A 连线和相机左右轴的夹角 > 45°（按任务定义是前后关系，不是左右） | 左右 | 45° |
| `below_options` / `above_options` | 测得的量超出选项能表示的范围 | 距离（最小 0.5 m）、高度（0.1–0.9） | |
| `cut_off_vertically` | 物体上下不完整 | 高度 | 竖直方向在画面内 < 0.95 |
| `occluder_not_visible` | 遮挡物 B 看不清 | `occluded_now` | |
| `occluder_out_of_view` | 动作后遮挡物 B 不在视野里 | `occluded_after` | |
| `target_behind_camera` | 目标 A 的 8 个角不全在相机前方 | 遮挡 | |
| `target_out_of_frame` | 目标 A 的轮廓在画面内的比例 < 0.80（出画不算被挡，任务的门） | 遮挡 | 0.80 |
| `target_not_visible` | 答"没有 / 部分"时目标 A 看不清（防止 A 其实被墙挡住） | 遮挡 | |
| `forward:blocked_move` | 前向题的候选动作有一步被碰撞规则（厚墙）挡住 | 前向题（按动作计，不分题类） | |
| `blocked_turn` | 逆向题：这一回合被挡住了，前后两张图相同 | `action_between` | |
| `mixed_actions` | 逆向题：这一回合有不止一种动作 | `action_between` | |
| `pose_mismatch` | 逆向题：前一帧做这个动作 k 次得不到后一帧（误差 > 0.001） | `action_between` | |

构建阶段（`build`）去掉的题记在 `stats.json`：

| 键 | 含义 |
| --- | --- |
| `balance.<题类>.dropped.answer_balance` | 答案平衡去掉 |
| `balance.<题类>.dropped.unchanged` | 前向题没变化的超过 10% |
| `balance.<题类>.dropped.flip_band` | 前向题翻转比例超出 35%–65% |
| `classes.goal_cap_dropped.forward` | `goal_after` 超过前向题一半 |
| `classes.caps.forward` / `classes.caps.inverse` | 前向题、逆向题的上限（`classes.<大类>.available` → `kept`） |
| `scene_cap_dropped` | 单场景超过 8% |
| `final_letter_cap_dropped` | 文件最后的字母平衡 |
| `probe_excluded` | 测试集里去掉的探针题数 |

## 8. 一致性检查

`raw` 对每条轨迹断言：终点状态成功，终点之前的状态都不成功（最短路径上不会提前成功；环境满足就结束）。结果在 `checks`：`terminal_not_success`、`success_before_terminal` 出现任何一次，或成功判定和势能（≥ 0.5 当且仅当成功）不一致（`ConsistencyError`），就停下不出题。逆向题另用 `contract.apply_action` 复核每一对帧（`pose_mismatch`）。

## 9. 版本

| 题库 | 目录 | 锚点 | 三类题规则 | 探针集 |
| --- | --- | --- | --- | --- |
| 2026-10-18 | `exps/transition_qa_20261018` | 每条轨迹最多 3 个 | 50 / 35 / 15，goal 题各占所在大类一半 | 500 道，从 val 平衡后的题里按题类轮流抽（遮挡题只有 20 道） |
| 2026-10-19 | `exps/transition_qa_20261019` | 全部可用帧 | 第 6.3 节 | 600 道，每类 40 道，第 6.4 节 |
