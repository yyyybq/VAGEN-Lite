# P1 题型目录

15 个题类，每个一节：问什么、模型看到什么、题目原文、选项、答案用哪个函数算、什么情况下不出题。字段和平衡、上限规则见 `DATA_CONTRACT.md`，题数和例题见 `P1_STATS.md`。

**共同点**

- 答案都由环境或生成器现有的函数算，和环境对同一个位姿的判定一致（`PROGRESS.md` 第 1.3 节的对照表）。缩写：`TV` = `vagen/envs/active_spatial/target_visibility.py`，`M1` / `M2` = `canonical_task_metrics.py` / `canonical_task_metrics_v2.py`，`VB` = `visual_bbox_metrics.py`。
- 图是 256×256 的渲染图，相机是判据用的同一个 `canonical_camera`。
- 物体的称呼用任务原文里的指代（`qa_states.ref_text`，例如 "the smaller cabinet"）；A 是任务的第一个物体，B 是第二个。遮挡题里 A 是要藏的目标，B 是遮挡物。
- "清楚可见" = 可见规则 face_visible_v3：包围盒至少一个面露出 ≥ 40%，露出部分 ≥ 画面 0.3%。被问的物体在模型看到的那张图里必须清楚可见（"都在画面里吗"除外，它问的就是这个）。
- 动作：前进、后退、左移、右移每次 0.3 m，左转、右转每次 20°。前向题 k = 1–3 次，逆向题 k = 1–5 次（一个回合最多 5 个动作）。前向题的动作每一步都要合法（厚墙碰撞规则），有一步被挡就不出题。
- 前向题的条件句：`If the camera <动作> from the current view, ...`，例如 `turns left by 40 degrees`、`moves sideways to the right by 0.6 meters`。模型只看到当前一张图，看不到动作之后的画面。
- 每道题以 `Answer Yes or No.` 或 `Answer with the option's letter.` 结尾；训练目标只有答案本身。

## 状态题（看当前一张图）

### `goal_now`：当前是否已满足任务（所有族）

- 题目：`Task: <任务原文>\nIn the current view, is the task's requirement already met? Answer Yes or No.`
- 答案：`TV.score_with_visibility`，就是环境的成功判定（标准判据 + 可见规则 + 左右 45°）。
- 不出题：在成功边界上，也就是位姿挪 ±0.05 m 或转 ±3° 后结论会变。
- 状态来源：轨迹上的帧答 No（终点之前都没成功），终点帧和内部状态答 Yes。
  - 内部状态：终点附近 0.45 m、15° 内挪 ±0.10 m、转 ±6° 仍成功的位姿，另渲染了 7,528 张。
  - 终点帧大多贴着成功边界，56.6% 被边界过滤去掉。
- 平衡：Yes / No 各约一半。2026-10-19 起它是状态题，全部保留，不再限它占状态题的一半。2026-10-20 起在**每个任务族里**各约一半（题干里的任务原文就说明了族；之前只按任务族猜就能答对测试集 65.3%）。

### `left_right_now`：A 在 B 的左边还是右边（投影）

- 题目：`In the current view, is A to the left or to the right of B?`
- 选项：(A) to the left、(B) to the right，顺序固定。
- 答案：两个中心在图上的横坐标差，用 `VB` 的投影，加 `TV.lateral_gate`。两个物体都在相机前方时，横坐标差的符号就是地面上的左右。
- 不出题：
  - A、B 有一个看不清，或中心在相机后方；
  - B→A 连线和相机左右轴夹角 > 45°（按任务定义是前后关系）；
  - 横坐标差 < 20 px（成功门 12 px 加 8 px 缓冲）。

### `closer_now`：A 和 B 哪个离你更近（等距）

- 题目：`In the current view, which is closer to you (horizontal distance to its nearest side): A or B?`
- 选项：两个物体，正确的那个位置轮流。
- 答案：`M2.footprint_distance`，就是等距判据用的距离：相机到包围盒地面投影最近边的水平距离。
- 不出题：A、B 有一个看不清；两个距离之差 < 较远者的 5%。

### `distance_bin_now`：离 A 大约几米（绝对距离）

- 题目：`In the current view, about how far are you from A, measured horizontally to its nearest side?`
- 选项：4 个相邻的 0.5 m 档（例如 1.5 / 2.0 / 2.5 / 3.0 米），从小到大排。正确档放在第几个，由构建时的窗口分配决定（2026-10-20 起）：同一组 4 个选项里 A–D 一样多，所以"看最小的选项猜字母"答不出（之前测试集 44.0%，随机 25%）。见 `DATA_CONTRACT.md` 第 5 节。
- 答案：同上 `M2.footprint_distance`，取最近的档。
- 不出题：A 看不清；离最近的档超过 0.15 m；最近的档小于 0.5 m。

### `occluded_now`：A 现在被 B 挡住了吗（遮挡）

- 题目：`In the current view, is A hidden behind B?`
- 选项：fully / partly / not hidden behind B，正确的位置轮流。
- 答案：`M2.occlusion_silhouette`，就是 OA 判据本身：在 A 朝相机的面上采样，看视线是否先穿过 B 的包围盒，按投影面积加权。
  - 完全：被挡比例 ≥ 0.85（任务的门）；
  - 没有：≤ 0.05；
  - 部分：0.25–0.70。
- 不出题：
  - B 看不清；
  - A 的 8 个角不全在相机前方，或 A 的轮廓在画面内的比例 < 0.80（出画不算被挡）；
  - 答"没有 / 部分"时 A 看不清；
  - 被挡比例落在两档之间。
- 数量：遮挡族只占正式题的 2.8%，这是题最少的题类。探针集另用第十二轮新增的 15 道 val 遮挡题的最短路径帧补；2026-10-20 起 val 场景的遮挡题全部进探针集，测试集没有这一类。
- **已知局限：遮挡物按实心的包围盒算**（用户核对第 18 题，2026-10-20）。环境的遮挡判据和本题用同一个函数，都把遮挡物当成实心的长方体。内部空的、镂空的或形状不规则的物体（敞开的衣柜、人、显示器、画架、枯枝等），框挡住了目标，画面里却看得见目标，答案就和画面不符。题库里遮挡物是衣柜的占遮挡候选题的 11.8%，人 5.8%，显示器、电脑、画架、望远镜、枯枝、水龙头等细或不规则的物体合计约 15%。要改，得改环境的判据（主线），或者用渲染的深度 / 不透明度复核（要渲染）；题库没有改。2026-10-20 先抽 30 道（衣柜、人、细或不规则各 10 道）给用户核，量出错误率再定（`PROGRESS.md` 第 12.9 节）。

### `both_in_view_now`：A 和 B 现在都在画面里吗（视野）

- 题目：`In the current view, are A and B both inside the view?`
- 选项：both are / only A / only B / neither，正确的位置轮流。
- 答案：每个物体分别用视野判据的门：
  - "在"：8 个角都在前方、在画面内的比例 ≥ 0.95（`M2.FOV_FULL_INSIDE_MIN`）、清楚可见；
  - "不在"：中心在相机后方、投影不可见，或在画面内 ≤ 0.30。
- 不出题：有一个物体介于"在"和"不在"之间（包括在画面里但看不清）。

### `height_bin_now`：A 的高度大约占画面高度的几成（屏幕占比）

- 题目：`In the current view, about what fraction of the image height does A fill, from its top to its bottom?`
- 选项：4 个相邻的 10% 档，从小到大排，范围 10%–90%；正确档的位置同 `distance_bin_now`，构建时按窗口分配。
- 答案：`M2.judge_screen_occupancy` 的 occupancy，即包围盒投影高度 / 图像高度。
- 不出题：A 看不清；8 个角不全在前方；上下不完整（竖直方向在画面内 < 0.95）；离最近的档超过 0.03。

## 前向题（看当前一张图加一句动作）

7 个题类和状态题一一对应：`goal_after`、`left_right_after`、`closer_after`、`distance_bin_after`、`occluded_after`、`both_in_view_after`、`height_bin_after`。

- **答案**：`contract.apply_action` 做 k 次得到动作后的位姿，在那个位姿上用对应状态题的同一个函数重算。
- **题目**：状态题的题目把 `In the current view,` 换成 `If the camera <动作> from the current view,`，动词改成将来时（`will A be ...`、`which will be closer`）。`goal_after` 是 `Task: <任务原文>\nIf the camera <动作> from the current view, will the task's requirement be met?`。
- **两张图的要求不同**：
  - 当前那张图：被问的物体要清楚可见；
  - 动作后的位姿：只要求测得的量离阈值够远，物体可以出画（例如"右转 40° 后 B 还在画面里吗"答"只有 A"）。
  - 例外：`left_right_after` 要求动作后两个物体在画面内都 ≥ 0.5，`occluded_after` 要求动作后遮挡物还在视野里。
- **上限和平衡**：
  - 每个锚点、每个题类最多 3 道，分散到不同的（答案、是否变化）组；
  - 没变化的 ≤ 10%；
  - 答案翻转比例 35%–65%（两种都常见时）；
  - 2026-10-20 起，是非、左右、遮挡、视野的答案在**同一个动作 × 次数、同一个任务族**里平衡。之前只凭题干里的任务和动作就能答对测试集 `goal_after` 71.9%、`both_in_view_after` 39.2%（随机 50%、25%），例如"后退"之后多半答"都在画面里"；
  - `goal_after` 最多占前向题的一半；
  - 前向题合计 ≤ 状态题的 1.2 倍。
- **左右题几乎不翻转**：`left_right_after` 的答案基本等于现在的答案。转身不改变前方两个物体的左右顺序，0.9 m 以内的平移也很少改变。这是物理事实，这一类题照实保留，翻转比例另报。

## 逆向题（看前后两张图）

### `action_between`：从第一张图到第二张图，相机做了什么动作（所有族）

- 题目：`First image: <image>\nSecond image: <image>\nThe first image was taken right before a camera action and the second image right after it. What did the camera do?`
  - 题干不带任务原文（用户 2026-10-18）。
- 选项（2026-10-20 起）：**2 × 2**，{真实动作 a，配对动作 b} × {真实次数 k，相邻次数 k′}，例如真实是"后退 0.3 m"时：后退 0.3 m、后退 0.6 m、前进 0.3 m、前进 0.6 m。正确的位置轮流。
  - 配对动作 b 是对称的：前进只和后退配；左转配右转或左移，左移配右移或左转（按题号二选一），右侧同理。所以每一对动作里，两个都可能是答案。
  - k′ = k − 1 或 k + 1（按题号二选一，限在 1–5）。
  - 每个动作、每个次数在选项里都正好出现两次，"出现两次的那个"不再指向答案；构建时还在每个（动作对、次数对）里把 4 种答案配平（`DATA_CONTRACT.md` 第 5 节），只看选项文字的规则都在 25% 左右。
  - 2026-10-18 / 19 的旧选项是：同一动作的相邻次数、反方向同样次数、同侧另一类同样次数。正确答案总是唯一一个"动作出现两次、次数又最常见"的选项，只看文字就能 100% 答对（`DIAG3_PROBE.md`）。
- 答案：SFT 轨迹相邻两帧之间那个回合的动作，用 `contract.apply_action` 复核前一帧做 k 次得到后一帧（误差 ≤ 0.001）。
- 不出题：
  - 这一回合被挡住了（两张图相同）；
  - 回合里有不止一种动作；
  - 复核不符。
- 上限：逆向题合计 ≤ 题库的 15%。2026-10-20 的题库配平后只有 8,409 道（9.6%），没碰到上限。
- 偏向：动作来自最短路径，不是随机动作。train 的 22,947 对可用帧对里，后退 7,394、右转 4,414、左转 4,176、左移 2,643、右移 2,488、前进 1,832（`PROGRESS.md` 第 1.5 节）。
