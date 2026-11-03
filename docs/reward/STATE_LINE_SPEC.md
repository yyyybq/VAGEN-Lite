# 感知陈述的格式（STATE_LINE_SPEC）

主线写，2026-11-03。依据用户 11-03 的指示第 3 节："动作前一行固定格式的状态陈述，字段从生成器和环境的真值算，和进度分三组用同一套函数，距离分档和任务的距离要求对齐"。交奖励线确认。

- 第 2 版（11-03 晚）按奖励线的核对意见改了四处：代码分在两个提交、face 和 clear 不完全等价、整项比的计分口径、三分位量的是到底面轮廓的距离。另改了几处小处。
- 用户 11-03 定了沿用这个格式和奖励线已训的起点。

## 结论

- **建议沿用奖励线已经实现、已经训过的 `<observe>` 格式，不另起一套。**
  - 它的四个字段和用户例子里的四个字段一一对应。真值从环境自己的成功判定算，三组的划分和进度分组是同一个划分。
  - 带这种陈述的 SFT 数据（11,095 段对话）和 10 个 epoch 的起点都已经有了，只差采样验收和 EASI-8。
- **和用户例子有三处写法不同**，见第 3 节。都不影响用途：
  1. 位置按任务里的每个物体分别写；
  2. 正对写 yes / no；
  3. "还差哪个条件"写成一个集合，可以同时差几组。
- 如果要严格照例子的写法，要重做数据、重训起点：约 45 卡时、1–1.5 天，奖励线第一轮也要跟着等。这一项列在主线报告的待决事项里。

## 1. 格式

每个回答轮在动作之前写一行陈述：

```
<observe>where=A:left,B:in_view; face=A:no,B:yes; dist=far; missing=frame,clear</observe>
<action>turn_left|</action>
```

动作照提示的写法，每个动作后面跟一个竖线。

- 陈述描述的是模型这一轮看到的那一帧，也就是动作之前的位姿。
- 四个字段用 `; ` 分开，顺序固定。字段名和取值都是小写，物体字母大写。
- 读不出（没有 `<observe>`，或某个字段缺失、写错）的字段算错。动作照环境原来的规则读，陈述不影响动作是否有效。

| 字段 | 写法 | 取值 | 含义 |
| --- | --- | --- | --- |
| `where` | 每个物体一个：`A:值,B:值` | `in_view` / `left` / `right` / `behind` | 物体包围盒的中心在相机前方、投影落在画面内，就是 `in_view`。不在画面里时按相机坐标系的水平方位角分：偏离正前方超过 135° 是 `behind`，否则按左右分 `left` / `right`。遮挡题只写被挡的目标 A。 |
| `face` | 成功条件要求"清晰可见"的每个物体一个：`A:yes` | `yes` / `no` | 环境的正对规则（face_visible_v3）判这个物体此刻是否清晰可见。遮挡题写遮挡物 B。 |
| `dist` | 一个值 | `near` / `mid` / `far` | 相机到物体 A 底面轮廓的水平距离（`footprint_distance`），不是到物体中心的距离。绝对距离题和任务要求的距离对齐：比"要求 − 容差"近是 `near`，在容差内是 `mid`，比"要求 + 容差"远是 `far`；容差 = max(0.25 m, 0.1 × 要求的距离)。提示里只写"比要求近 / 在要求处 / 比要求远"，不写容差的数值。其余题按固定分档：< 1.6 m 是 `near`，> 2.6 m 是 `far`，中间是 `mid`，这两条边界写在提示里。 |
| `missing` | 一组名字，按 frame、clear、relation 的顺序 | 三组的任意组合，成功时写 `none` | 当前位姿还没满足的成功条件属于哪几组。三组的含义见第 2 节。 |

物体字母按任务描述里出现的角色顺序（canonical_task_metrics_v2 的角色）：

| 题型 | `where` 写哪些物体 | `face` 写哪些物体 |
| --- | --- | --- |
| absolute_positioning、screen_occupancy | A | A |
| equidistance、fov_inclusion、projective_relations | A、B | A、B |
| occlusion_alignment | A（被挡的目标） | B（遮挡物） |
| size_distance_invariance | A、B | A、B |
| centering | A、B、C | A、B、C |

最后两个题型代码里有物体表，但 v10 的数据和评测里没有这两类，没有在数据上验过。

## 2. 真值从哪里来

- **代码分在奖励线的两个提交上：**
  - reward-design 2b7820ea（工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-reward`）：
    - 真值函数 `vagen/statement_truth.py`；
    - observe 提示（`vagen/envs/active_spatial/prompt.py`）；
    - SFT 数据生成器 `reward/build_statement_sft.py`。
  - reward-stmt c51a2c66（工作树 `VAGEN-Lite-reward-stmt`，还没并进 reward-design）：
    - 环境开关 `statement_record`：每个回答的陈述对照模型这一轮看到的帧打分；
    - 陈述的 token 范围 `statement_token_count`；
    - 估计器的陈述项；
    - rl-config 的 `--statement`；
    - 打分工具 `reward/statement_eval.py`（不渲染重放，逐字段打分）。
  - 这两个工作树都只读。主线要用它们的代码时，按提交号另建自己的工作树：采样验收用的 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-q3stmt` 就停在 c51a2c66。
- `statement_truth.py` 里都是纯函数，只用位姿和场景几何，不渲染：
  - `truth(item, c2w, occ)`：算出四个字段，同时返回成功与否和没满足的条件；
  - `format_statement`：写出标签；
  - `parse_statement`：从回答里读出四个字段；
  - `score_statement`：逐字段判对错。
- **成功判定：** 用环境自己的 `target_visibility.score_with_visibility(item, c2w, occ)`。遮挡体和环境一样，从场景的碰撞体建 `OccluderSet`，所以陈述的真值和环境判成功永远一致。
- **三组的划分和进度分组是同一个划分。** 两边都以 `target_visibility.FRAME_CONDITIONS` 为界：
  - **frame**：物体存在、在前方、可见、面积够大、在画面内，以及遮挡题里遮挡物的这几项。进度分组里叫 frame。
  - **clear**：每个 `*_clearly_visible` 条件和 `required_targets_found`，也就是正对规则。进度分组里叫 face。
  - **relation**：其余条件，包括距离、等距、左右前后关系、留白、画面占比，以及左右题的 45° 规则。进度分组里叫 relation。
  - 两边用的是不同的函数：陈述用 `statement_truth.gate_group`，判的是"条件满足没有"；进度用 `progress_groups`，算的是"离满足还差多少"。划分本身相同，奖励线对着代码核过：
    - `progress_groups` 的 frame_met 就是 FRAME_CONDITIONS 那些条件；
    - face_met = vis["visible"]，即目标找到了，且所有目标和参照物都清晰可见，正好是 clear 组的全部条件；
    - relation_met 是其余条件加 45° 规则，正好是 relation 组。
  - 主线会补一个逐帧核对，在 SFT 数据的全部帧上，比 `missing` 和 (not frame_met, not face_met, not relation_met)，结果写回这一页。
- **`face` 字段和 clear 组不完全等价。**
  - `face` 只写任务目标（REQUIRED_VISIBLE_SLOTS）。
  - clear 组还包括 `required_targets_found`，以及题面里参照物（ref_anchor）的清晰可见，例如"沙发旁边的椅子"里的沙发。
  - SFT 的 28,654 个回答帧里，有 1,831 帧（6.4%）`face` 全是 yes，`missing` 却含 clear；反过来的情况是 0 帧。
- **距离分档：**
  - 绝对距离题的容差和环境的成功判定是同一个式子（canonical_task_metrics_v2）：max(0.25 m, 0.1 × 要求的距离)。
  - 其余题的 1.6 / 2.6 m，取自 C0 SFT 数据 28,654 个回答帧的三分位数（1.59 / 2.60 m，取整）。量的是相机到物体 A 底面轮廓的水平距离。
  - 提示里写明了这两条边界。

## 3. 和用户例子的对照

| 用户的例子 | 这里的写法 | 为什么这样写 |
| --- | --- | --- |
| `<state>…</state>` | `<observe>…</observe>` | 只是标签名不同。已训的起点用 observe，改名就要重训。 |
| `target=in_view\|left\|right\|behind`，一个值 | `where=A:…,B:…`，每个物体一个值 | 两物体题（等距、视野、投影关系）的成败取决于两个物体各自在哪儿，一个值说不清。 |
| `face=visible\|hidden` | `face=A:yes,B:no` | 意思相同；也按物体分开写，因为成功要求每个指定物体都清晰可见。 |
| `dist=near\|mid\|far` | 相同 | 绝对距离题和任务要求对齐，和例子的要求一致。 |
| `missing=frame\|face\|relation\|none`，一个值 | `missing=frame,clear,relation` 的子集，成功时写 `none` | SFT 数据的 28,654 个回答帧里，70% 同时差两组以上，例如"frame,clear,relation"有 11,547 帧，"frame,relation"有 4,048 帧。只写一个值会丢信息，取哪一个也要另定规则。clear 大体就是例子里的 face，但还包括参照物的清晰可见（见第 2 节）。 |

## 4. 已有的数据和起点（奖励线，11-03）

- **数据：** `reward_runs/sft_statement/sft_trajectories.jsonl`，11,095 段对话：train 9,802 段，val 1,293 段，共 28,654 个回答轮，每轮都有陈述。
  - 来源是 C0 的 Qwen3 新格式动作轨迹（`models_runs/sft_v10_newformat_20261024`），一段不少。
  - 陈述平均 88.6 个字符。
- **起点：** 链 `reward_runs/runs/rw_st_sft`，按 C0 的配方训了 10 个 epoch，10 个检查点都在 `lf_q3stmt/model/checkpoint-{154..1540}`。
  - 链按验证集挑了第 5 个 epoch（checkpoint-770），用的是没筛场景的 413 道验证集。
  - 动作成功率和 C0 e8 分不出：两个集合合起来 43.7% 对 44.3%。
  - 陈述全部读得出。四项平均准确率约 62%，"还差哪个条件"一项约 44%。
- **还缺：**
  1. 每个 epoch 的采样验收：200 道后备题 × 4 次、温度 0.8，看熵、读不出、陈述逐字段准确率；
  2. 在过线的 epoch 里，按筛过场景的验证集 382 道重挑；
  3. 挑出的起点的 EASI-8；
  4. 200 条人工抽查页面。
  - 主线负责补这几项。

## 5. 验收时报的数

- **动作：** 读不出的回合比例（≤ 1%）、熵（≤ 0.06）、成功率。用 `data_gen/interiorgs_v10/accept_summary.py`。
- **陈述：**
  - 读得出的比例；
  - 四个字段各自的准确率，以及四项平均。每个字段整项比：`where`、`face` 要每个物体都对，这一项才算对；`missing` 要整组相同。奖励线的 62% 和固定答参照都是这个口径。按物体逐个比会更高，两个口径不能混。
  - 成功局和失败局分开报，第 1 回合和第 4 回合以后分开报。
  - 参照线：每个题型固定答验证集上最常见的值能得多少（奖励线 11-03 算过：52.7% / 53.0%）。
