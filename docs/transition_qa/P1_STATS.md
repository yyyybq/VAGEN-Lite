# P1 转移题：统计

由 `python -m data_gen.transition_qa.report` 从 `exps/transition_qa_20261018` 生成。题库怎么来的、各题类的答案用哪个函数算，见 `PROGRESS.md` 第 1.3 节和第 2 节。

## 1. 题数和三类题的比例

| 文件 | 题数 | 状态题 (a) | 前向题 (b) | 逆向题 (c) |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | 35039 | 17467（49.9%） | 12216（34.9%） | 5356（15.3%） |
| p1_test.jsonl | 4073 | 2056（50.5%） | 1354（33.2%） | 663（16.3%） |
| p1_probe.jsonl | 500 | 230（46.0%） | 233（46.6%） | 37（7.4%） |

目标比例 50% / 35% / 15%。三类题按这个比例从平衡后的题里抽，受最少的那一类限制。

- train 场景：前向题 (b)可用 45335、留下 12498；逆向题 (c)可用 17015、留下 5356；状态题 (a)可用 17854、留下 17854
  - "任务要求满足了吗"两类题（goal_now / goal_after）各自最多占本类的一半，因此去掉：前向题 (b) 15277 道，状态题 (a) 16333 道
- val 场景：前向题 (b)可用 5407、留下 1634；逆向题 (c)可用 2259、留下 700；状态题 (a)可用 2334、留下 2334
  - "任务要求满足了吗"两类题（goal_now / goal_after）各自最多占本类的一半，因此去掉：前向题 (b) 2383 道，状态题 (a) 101 道

## 2. 按任务族 × 题类 × 划分的题数

**p1_train.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 993 | 0 | 0 | 0 | 0 | 2522 | 2123 | 1727 | 2031 | 0 | 0 | 0 | 0 | 0 | 0 | 9396 |
| equidistance | 758 | 0 | 0 | 754 | 1151 | 0 | 0 | 665 | 1121 | 0 | 0 | 0 | 0 | 0 | 0 | 4449 |
| fov_inclusion | 762 | 860 | 1320 | 0 | 0 | 0 | 0 | 555 | 1148 | 0 | 0 | 0 | 0 | 0 | 0 | 4645 |
| occlusion_alignment | 168 | 0 | 0 | 0 | 0 | 0 | 0 | 206 | 253 | 0 | 0 | 0 | 0 | 162 | 207 | 996 |
| projective_relations | 1661 | 0 | 0 | 0 | 0 | 0 | 0 | 1649 | 2366 | 0 | 0 | 856 | 3231 | 0 | 0 | 9763 |
| screen_occupancy | 1014 | 0 | 0 | 0 | 0 | 0 | 0 | 1302 | 1800 | 958 | 716 | 0 | 0 | 0 | 0 | 5790 |
| 合计 | 5356 | 860 | 1320 | 754 | 1151 | 2522 | 2123 | 6104 | 8719 | 958 | 716 | 856 | 3231 | 162 | 207 | 35039 |

**p1_test.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 132 | 0 | 0 | 0 | 0 | 337 | 325 | 227 | 307 | 0 | 0 | 0 | 0 | 0 | 0 | 1328 |
| equidistance | 108 | 0 | 0 | 73 | 111 | 0 | 0 | 77 | 112 | 0 | 0 | 0 | 0 | 0 | 0 | 481 |
| fov_inclusion | 99 | 54 | 130 | 0 | 0 | 0 | 0 | 61 | 140 | 0 | 0 | 0 | 0 | 0 | 0 | 484 |
| occlusion_alignment | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 42 |
| projective_relations | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 219 | 280 | 0 | 0 | 67 | 317 | 0 | 0 | 1075 |
| screen_occupancy | 119 | 0 | 0 | 0 | 0 | 0 | 0 | 164 | 246 | 63 | 71 | 0 | 0 | 0 | 0 | 663 |
| 合计 | 663 | 54 | 130 | 73 | 111 | 337 | 325 | 760 | 1102 | 63 | 71 | 67 | 317 | 0 | 0 | 4073 |

**p1_probe.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 4 | 0 | 0 | 0 | 0 | 37 | 37 | 13 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 103 |
| equidistance | 7 | 0 | 0 | 37 | 37 | 0 | 0 | 3 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 89 |
| fov_inclusion | 8 | 37 | 37 | 0 | 0 | 0 | 0 | 4 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 93 |
| occlusion_alignment | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 11 | 9 | 21 |
| projective_relations | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 8 | 0 | 0 | 37 | 36 | 0 | 0 | 98 |
| screen_occupancy | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 5 | 37 | 37 | 0 | 0 | 0 | 0 | 96 |
| 合计 | 37 | 37 | 37 | 37 | 37 | 37 | 37 | 37 | 37 | 37 | 37 | 37 | 36 | 11 | 9 | 500 |

## 3. 答案分布

"字母"是正确选项的字母；"偏差"是各字母占比和均匀分布之差的最大值（验收线 5 个百分点）。"答案值"是选项的含义（例如 left / right、full / partial / none）。

**p1_train.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 5356 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | move_backwardx1 565，turn_rightx1 459，move_backwardx2 445，turn_leftx1 423，move_rightx1 338，move_leftx1 338 |
| both_in_view_after | 860 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | neither 232，both 214，only_a 209，only_b 205 |
| both_in_view_now | 1320 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_a 341，both 341，only_b 341，neither 297 |
| closer_after | 754 | A 50.0% / B 50.0% | 0.0 | a 380，b 374 |
| closer_now | 1151 | A 50.0% / B 50.0% | 0.0 | b 592，a 559 |
| distance_bin_after | 2522 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 2.5 568，2.0 539，3.0 426，1.5 340，3.5 235，1.0 147 |
| distance_bin_now | 2123 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 2.5 661，3.0 375，2.0 361，1.5 279，3.5 168，1.0 128 |
| goal_after | 6104 | Yes 47.6% / No 52.4% | 2.4 | no 3197，yes 2907 |
| goal_now | 8719 | Yes 47.6% / No 52.4% | 2.4 | no 4567，yes 4152 |
| height_bin_after | 958 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 0.3 268，0.2 171，0.4 143，0.5 91，0.6 87，0.8 73 |
| height_bin_now | 716 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 0.3 205，0.4 178，0.2 139，0.6 60，0.5 42，0.7 39 |
| left_right_after | 856 | A 52.3% / B 47.7% | 2.3 | left 448，right 408 |
| left_right_now | 3231 | A 52.4% / B 47.6% | 2.4 | left 1692，right 1539 |
| occluded_after | 162 | A 33.3% / B 33.3% / C 33.3% | 0.0 | full 63，none 50，partial 49 |
| occluded_now | 207 | A 33.3% / B 33.3% / C 33.3% | 0.0 | full 72，none 72，partial 63 |

**p1_test.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 663 | A 25.0% / B 24.9% / C 25.0% / D 25.0% | 0.1 | move_backwardx1 77，turn_leftx1 65，move_leftx1 47，move_rightx1 46，turn_rightx1 46，move_backwardx3 40 |
| both_in_view_after | 54 | A 25.9% / B 24.1% / C 24.1% / D 25.9% | 0.9 | only_b 21，neither 17，only_a 10，both 6 |
| both_in_view_now | 130 | A 24.6% / B 25.4% / C 25.4% / D 24.6% | 0.4 | both 35，only_b 34，neither 31，only_a 30 |
| closer_after | 73 | A 49.3% / B 50.7% | 0.7 | a 39，b 34 |
| closer_now | 111 | A 50.5% / B 49.5% | 0.5 | a 56，b 55 |
| distance_bin_after | 337 | A 25.5% / B 25.5% / C 25.5% / D 23.4% | 1.6 | 2.5 74，2.0 74，3.0 51，1.5 44，3.5 36，4.0 21 |
| distance_bin_now | 325 | A 26.5% / B 24.6% / C 24.3% / D 24.6% | 1.5 | 2.5 90，3.0 71，1.5 49，2.0 45，3.5 25，4.0 23 |
| goal_after | 760 | Yes 47.6% / No 52.4% | 2.4 | no 398，yes 362 |
| goal_now | 1102 | Yes 47.6% / No 52.4% | 2.4 | no 577，yes 525 |
| height_bin_after | 63 | A 25.4% / B 25.4% / C 25.4% / D 23.8% | 1.2 | 0.3 16，0.2 15，0.4 13，0.7 5，0.5 5，0.6 4 |
| height_bin_now | 71 | A 25.4% / B 25.4% / C 25.4% / D 23.9% | 1.1 | 0.3 22，0.2 16，0.4 13，0.1 5，0.5 4，0.8 4 |
| left_right_after | 67 | A 52.2% / B 47.8% | 2.2 | left 35，right 32 |
| left_right_now | 317 | A 52.4% / B 47.6% | 2.4 | left 166，right 151 |

**p1_probe.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 37 | A 24.3% / B 24.3% / C 27.0% / D 24.3% | 2.0 | turn_rightx1 5，move_backwardx1 4，move_backwardx2 4，move_leftx1 3，move_forwardx4 3，turn_leftx1 2 |
| both_in_view_after | 37 | A 24.3% / B 24.3% / C 24.3% / D 27.0% | 2.0 | only_a 12，neither 9，only_b 8，both 8 |
| both_in_view_now | 37 | A 24.3% / B 27.0% / C 24.3% / D 24.3% | 2.0 | only_a 13，only_b 9，both 8，neither 7 |
| closer_after | 37 | A 48.6% / B 51.4% | 1.4 | a 19，b 18 |
| closer_now | 37 | A 51.4% / B 48.6% | 1.4 | b 21，a 16 |
| distance_bin_after | 37 | A 27.0% / B 24.3% / C 24.3% / D 24.3% | 2.0 | 2.5 9，3.5 7，2.0 7，3.0 4，4.0 4，1.5 3 |
| distance_bin_now | 37 | A 27.0% / B 24.3% / C 24.3% / D 24.3% | 2.0 | 2.5 14，2.0 5，1.5 5，3.0 4，3.5 3，4.5 2 |
| goal_after | 37 | Yes 48.6% / No 51.4% | 1.4 | no 19，yes 18 |
| goal_now | 37 | Yes 48.6% / No 51.4% | 1.4 | no 19，yes 18 |
| height_bin_after | 37 | A 27.0% / B 24.3% / C 24.3% / D 24.3% | 2.0 | 0.2 13，0.3 8，0.4 5，0.5 4，0.8 3，0.6 2 |
| height_bin_now | 37 | A 27.0% / B 24.3% / C 24.3% / D 24.3% | 2.0 | 0.3 9，0.4 8，0.2 7，0.7 4，0.6 3，0.5 3 |
| left_right_after | 37 | A 51.4% / B 48.6% | 1.4 | left 19，right 18 |
| left_right_now | 36 | A 50.0% / B 50.0% | 0.0 | left 18，right 18 |
| occluded_after | 11 | A 27.3% / B 36.4% / C 36.4% | 6.1 | full 5，partial 4，none 2 |
| occluded_now | 9 | A 33.3% / B 33.3% / C 33.3% | 0.0 | full 3，none 3，partial 3 |

训练集和测试集里字母偏差的最大值：2.4 个百分点（探针集只有 500 道、每类约 30 道，不计入；它的偏差见上表）。

## 4. 前向题：没变化的题、答案和现在不同的题

"没变化"：被问的量在动作前后的变化小于门槛（例如转身后到物体的距离不变）；上限 10%。"答案变了"：动作后的答案和现在的答案不同；控制在 35%–65%，防止模型学成"把现在的答案反过来"。

| 文件 | 题类 | 题数 | 没变化 | 答案变了 |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | both_in_view_after | 860 | 10.0% | 64.4% |
| p1_train.jsonl | closer_after | 754 | 9.9% | 44.8% |
| p1_train.jsonl | distance_bin_after | 2522 | 10.0% | 64.2% |
| p1_train.jsonl | goal_after | 6104 | 9.7% | 50.3% |
| p1_train.jsonl | height_bin_after | 958 | 9.6% | 65.2% |
| p1_train.jsonl | left_right_after | 856 | 9.9% | 0.0% |
| p1_train.jsonl | occluded_after | 162 | 8.0% | 56.0% |
| p1_test.jsonl | both_in_view_after | 54 | 13.0% | 46.4% |
| p1_test.jsonl | closer_after | 73 | 9.6% | 47.7% |
| p1_test.jsonl | distance_bin_after | 337 | 9.5% | 69.2% |
| p1_test.jsonl | goal_after | 760 | 9.2% | 53.9% |
| p1_test.jsonl | height_bin_after | 63 | 12.7% | 52.4% |
| p1_test.jsonl | left_right_after | 67 | 6.0% | 0.0% |
| p1_probe.jsonl | both_in_view_after | 37 | 5.4% | 82.4% |
| p1_probe.jsonl | closer_after | 37 | 10.8% | 47.6% |
| p1_probe.jsonl | distance_bin_after | 37 | 10.8% | 50.0% |
| p1_probe.jsonl | goal_after | 37 | 18.9% | 45.9% |
| p1_probe.jsonl | height_bin_after | 37 | 8.1% | 70.6% |
| p1_probe.jsonl | left_right_after | 37 | 18.9% | 0.0% |
| p1_probe.jsonl | occluded_after | 11 | 9.1% | 71.4% |

## 5. 过滤：没出的题和原因

每条轨迹最多取 3 个状态（终点一定取）。原始候选题 209400 道（训练和 val 场景合计），下表是在候选阶段就没出的题（按题类和原因）；之后的平衡、比例、场景上限去掉的数量见第 6 节。

| 题类 | 出了 | 没出（原因 数量） |
| --- | --- | --- |
| action_between | 19274 |  |
| both_in_view_after | 10474 | near_boundary 29835 |
| both_in_view_now | 2767 | near_boundary 1330 |
| closer_after | 4831 | not_visible_now 38854，near_boundary 7296 |
| closer_now | 1299 | not_visible 2323，near_boundary 591 |
| distance_bin_after | 17029 | near_boundary 38923，not_visible_now 3668，below_options 236 |
| distance_bin_now | 3090 | near_boundary 2686，not_visible 228，below_options 6 |
| forward | 0 | blocked_move 58041 |
| goal_after | 95555 | near_boundary_yes 20586，near_boundary_no 2280 |
| goal_now | 31245 | near_boundary_yes 6284，near_boundary_no 368 |
| height_bin_after | 10288 | cut_off_vertically 33711，near_boundary 18525，behind_camera 14567，not_visible_now 2376，above_options 52，below_options 2 |
| height_bin_now | 2001 | cut_off_vertically 2032，near_boundary 1666，not_visible 153，behind_camera 74，above_options 3 |
| left_right_after | 6062 | not_visible_now 76763，out_of_frame 41312，beyond_45deg 6058，behind_camera 3962，near_boundary 855 |
| left_right_now | 3758 | not_visible 4706，beyond_45deg 670，near_boundary 135 |
| occluded_after | 1233 | not_visible_now 3070，target_out_of_frame 1769，target_not_visible 1733，occluder_out_of_view 1576，near_boundary 596，target_behind_camera 52 |
| occluded_now | 494 | occluder_not_visible 196，target_not_visible 55，target_out_of_frame 53，near_boundary 52，target_behind_camera 1 |

- 锚点状态 30369 个；一致性检查：{"trajectories": 11095}（`terminal_not_success`、`success_before_terminal` 都没有出现就是 0）。

原因的含义：`not_visible` 被问物体在当前画面里不满足可见规则；`near_boundary` 离判定阈值太近；`beyond_45deg` 左右题里两物体的连线和相机左右轴夹角超过 45°；`blocked_move` 动作会被碰撞规则挡住；`behind_camera` / `out_of_frame` / `cut_off_vertically` 物体在相机后方、出画、上下不完整；`blocked_turn` 逆向题的那个回合被挡住了（两张图一样）。

## 6. 平衡和上限去掉的题

**train 场景**

| 题类 | 平衡前 | 去掉（原因） | 平衡后 |
| --- | --- | --- | --- |
| action_between | 17015 | {"answer_balance": 0} | 17015 |
| both_in_view_after | 9362 | {"flip_band": 1095, "unchanged": 1893, "answer_balance": 3251} | 3123 |
| both_in_view_now | 2464 | {"answer_balance": 1144} | 1320 |
| closer_after | 4265 | {"flip_band": 832, "unchanged": 697, "answer_balance": 0} | 2736 |
| closer_now | 1151 | {"answer_balance": 0} | 1151 |
| distance_bin_after | 14918 | {"flip_band": 1572, "unchanged": 1849, "answer_balance": 2063} | 9434 |
| distance_bin_now | 2705 | {"answer_balance": 507} | 2198 |
| goal_after | 84295 | {"flip_band": 15493, "unchanged": 23817, "answer_balance": 7040} | 37945 |
| goal_now | 28444 | {"answer_balance": 3184} | 25260 |
| height_bin_after | 9033 | {"flip_band": 0, "unchanged": 2025, "answer_balance": 3409} | 3599 |
| height_bin_now | 1741 | {"answer_balance": 998} | 743 |
| left_right_after | 5361 | {"flip_band": 0, "unchanged": 2167, "answer_balance": 10} | 3184 |
| left_right_now | 3322 | {"answer_balance": 14} | 3308 |
| occluded_after | 1151 | {"flip_band": 62, "unchanged": 348, "answer_balance": 149} | 592 |
| occluded_now | 457 | {"answer_balance": 250} | 207 |

场景上限（单个场景不超过 8%）去掉：{}

**val 场景**

| 题类 | 平衡前 | 去掉（原因） | 平衡后 |
| --- | --- | --- | --- |
| action_between | 2259 | {"answer_balance": 0} | 2259 |
| both_in_view_after | 1112 | {"flip_band": 104, "unchanged": 226, "answer_balance": 480} | 302 |
| both_in_view_now | 303 | {"answer_balance": 136} | 167 |
| closer_after | 566 | {"flip_band": 114, "unchanged": 88, "answer_balance": 0} | 364 |
| closer_now | 148 | {"answer_balance": 0} | 148 |
| distance_bin_after | 2111 | {"flip_band": 225, "unchanged": 274, "answer_balance": 316} | 1296 |
| distance_bin_now | 385 | {"answer_balance": 18} | 367 |
| goal_after | 11260 | {"flip_band": 1980, "unchanged": 3265, "answer_balance": 929} | 5086 |
| goal_now | 2801 | {"answer_balance": 1533} | 1268 |
| height_bin_after | 1255 | {"flip_band": 6, "unchanged": 304, "answer_balance": 589} | 356 |
| height_bin_now | 260 | {"answer_balance": 147} | 113 |
| left_right_after | 701 | {"flip_band": 0, "unchanged": 279, "answer_balance": 72} | 350 |
| left_right_now | 436 | {"answer_balance": 73} | 363 |
| occluded_after | 82 | {"flip_band": 19, "unchanged": 20, "answer_balance": 8} | 35 |
| occluded_now | 37 | {"answer_balance": 28} | 9 |

场景上限（单个场景不超过 8%）去掉：{}

## 7. 每类两道原题

**action_between**（absolute_positioning，start，`p1_7d753bb0889c`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera moved backward by 0.9 meters
(B) the camera moved forward by 0.6 meters
(C) the camera moved backward by 0.6 meters
(D) the camera moved sideways to the right by 0.6 meters
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_e7e4ad67215982dce49cb377__s0.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_e7e4ad67215982dce49cb377__s2.png`

**action_between**（projective_relations，start，`p1_3a07af733046`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera turned left by 40 degrees
(B) the camera turned left by 20 degrees
(C) the camera turned right by 20 degrees
(D) the camera moved sideways to the left by 0.3 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b407615d617c581ac0fc1e27__s0.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b407615d617c581ac0fc1e27__s1.png`

**both_in_view_after**（fov_inclusion，start，`p1_7b4b1fb78188`）

```
<image>
If the camera turns right by 60 degrees from the current view, will the wardrobe and the plant next to the reception both be inside the view?
(A) only the wardrobe
(B) only the plant next to the reception
(C) neither
(D) both are
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_de86c6f37472b9eb27fe94d5__s0.png`

**both_in_view_after**（fov_inclusion，middle，`p1_c67f30bfe4e2`）

```
<image>
If the camera turns left by 40 degrees from the current view, will the sofa next to the reception and the display cabinet next to the wardrobe both be inside the view?
(A) neither
(B) only the display cabinet next to the wardrobe
(C) only the sofa next to the reception
(D) both are
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_2a8e802247044760ce577e1c__s3.png`

**both_in_view_now**（fov_inclusion，start，`p1_116e39890eb4`）

```
<image>
In the current view, are the TV cabinet and the stool both inside the view?
(A) only the TV cabinet
(B) both are
(C) neither
(D) only the stool
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_f323d20c12f4971682eea3e6__s0.png`

**both_in_view_now**（fov_inclusion，middle，`p1_1bb1713911af`）

```
<image>
In the current view, are the dining table and the sofa both inside the view?
(A) only the dining table
(B) only the sofa
(C) both are
(D) neither
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_026938a451f49442baa50283__x2.png`

**closer_after**（equidistance，terminal，`p1_b6a969c13222`）

```
<image>
If the camera moves sideways to the left by 0.9 meters from the current view, which will be closer to you (horizontal distance to its nearest side): the wardrobe next to the plant or the sofa?
(A) the wardrobe next to the plant
(B) the sofa
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_f15ed61b2079324c94425f03__s8.png`

**closer_after**（equidistance，terminal，`p1_2439c76b8328`）

```
<image>
If the camera moves sideways to the left by 0.6 meters from the current view, which will be closer to you (horizontal distance to its nearest side): the armchair next to the sofa or the dining table?
(A) the dining table
(B) the armchair next to the sofa
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_ea828d2615921225584a6784__s7.png`

**closer_now**（equidistance，middle，`p1_a68dc84c7d63`）

```
<image>
In the current view, which is closer to you (horizontal distance to its nearest side): the sofa or the TV cabinet?
(A) the sofa
(B) the TV cabinet
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_c47f7939f27de2a662634f22__s3.png`

**closer_now**（equidistance，terminal，`p1_18d0bc05a1f1`）

```
<image>
In the current view, which is closer to you (horizontal distance to its nearest side): the side table or the floor lamp?
(A) the side table
(B) the floor lamp
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_2b154bf91033c784b6b8a580__s7.png`

**distance_bin_after**（absolute_positioning，terminal，`p1_d2f1b447cd31`）

```
<image>
If the camera moves backward by 0.9 meters from the current view, about how far will you be from the desk, measured horizontally to its nearest side?
(A) about 3.5 meters
(B) about 4.0 meters
(C) about 4.5 meters
(D) about 5.0 meters
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b56aea8521c13be9a041d7ce__s2.png`

**distance_bin_after**（absolute_positioning，terminal，`p1_587fbb31f583`）

```
<image>
If the camera turns left by 60 degrees from the current view, about how far will you be from the refrigerator, measured horizontally to its nearest side?
(A) about 1.0 meters
(B) about 1.5 meters
(C) about 2.0 meters
(D) about 2.5 meters
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_f87f44510284f00f94e66dd9__s7.png`

**distance_bin_now**（absolute_positioning，middle，`p1_133710746a12`）

```
<image>
In the current view, about how far are you from the bedside table, measured horizontally to its nearest side?
(A) about 0.5 meters
(B) about 1.0 meters
(C) about 1.5 meters
(D) about 2.0 meters
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_dad36b63d299ec14f033b377__x1.png`

**distance_bin_now**（absolute_positioning，start，`p1_28c2b6843356`）

```
<image>
In the current view, about how far are you from the kitchen equipment next to the ice making machine, measured horizontally to its nearest side?
(A) about 4.0 meters
(B) about 4.5 meters
(C) about 5.0 meters
(D) about 5.5 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_3d275b7835d140f3991e93d2__s0.png`

**goal_after**（screen_occupancy，terminal，`p1_9445b059b7ee`）

```
<image>
Task: Move so that the coffee table is fully visible from top to bottom and its height fills about 25% of the image height.
If the camera moves backward by 0.3 meters from the current view, will the task's requirement be met? Answer Yes or No.
答案：Yes
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_c9d2227d1fad1e339999df96__s4.png`

**goal_after**（fov_inclusion，terminal，`p1_8efa06e5ab9f`）

```
<image>
Task: Move so that both the side table and the notebook computer are fully in your view.
If the camera turns right by 60 degrees from the current view, will the task's requirement be met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_a37883204fef0246c51e1034__s3.png`

**goal_now**（absolute_positioning，start，`p1_220882b51f37`）

```
<image>
Task: Move to a spot about 3.0 meters away from the plant (measured to its nearest side), keeping the plant in view.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_43893c5c80adb4143fc22ee6__s0.png`

**goal_now**（fov_inclusion，middle，`p1_7369772726fa`）

```
<image>
Task: Move so that both the wardrobe next to the dining table and the sofa are fully in your view.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_0adac6318fa503e2dc657a37__s2.png`

**height_bin_after**（screen_occupancy，terminal，`p1_17bd3f9f5f36`）

```
<image>
If the camera turns left by 20 degrees from the current view, about what fraction of the image height will the chair fill, from its top to its bottom?
(A) about 20%
(B) about 30%
(C) about 40%
(D) about 50%
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_5d1df059a3d3061b14ba85e7__s4.png`

**height_bin_after**（screen_occupancy，middle，`p1_999b7ffd7783`）

```
<image>
If the camera moves backward by 0.9 meters from the current view, about what fraction of the image height will the flowers next to the armchair fill, from its top to its bottom?
(A) about 10%
(B) about 20%
(C) about 30%
(D) about 40%
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_6ed6825356e86d6c3d9ca14f__s1.png`

**height_bin_now**（screen_occupancy，start，`p1_9909caebe6b7`）

```
<image>
In the current view, about what fraction of the image height does the sofa fill, from its top to its bottom?
(A) about 20%
(B) about 30%
(C) about 40%
(D) about 50%
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_c0cb59f56d8273f759cb111f__s0.png`

**height_bin_now**（screen_occupancy，terminal，`p1_18e73b54beed`）

```
<image>
In the current view, about what fraction of the image height does the sofa fill, from its top to its bottom?
(A) about 10%
(B) about 20%
(C) about 30%
(D) about 40%
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_4a56b0142d1ac3543b7ca28c__x3.png`

**left_right_after**（projective_relations，middle，`p1_fd50877ed8bd`）

```
<image>
If the camera moves sideways to the left by 0.6 meters from the current view, will the desk calendar be to the left or to the right of the bed?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_d8300ce035ce5c3ea3b784da__s3.png`

**left_right_after**（projective_relations，terminal，`p1_aab09ea34e4a`）

```
<image>
If the camera moves backward by 0.3 meters from the current view, will the table be to the left or to the right of the TV?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_8e62feeed41c1617e383c363__s6.png`

**left_right_now**（projective_relations，terminal，`p1_25b9b266f1d1`）

```
<image>
In the current view, is the wardrobe next to the dining table to the left or to the right of the table next to the armchair?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_aa256f655dcf39f960946ade__s5.png`

**left_right_now**（projective_relations，terminal，`p1_464f905aaa55`）

```
<image>
In the current view, is the wardrobe to the left or to the right of the desk?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_7e7114088c0949e612fe0127__x4.png`

**occluded_after**（occlusion_alignment，middle，`p1_fe9c68f16348`）

```
<image>
If the camera moves forward by 0.6 meters from the current view, will the bed be hidden behind the wardrobe?
(A) fully hidden behind the wardrobe
(B) partly hidden behind the wardrobe
(C) not hidden behind the wardrobe
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_d4fc5f63188d248f6e4ec4ae__s2.png`

**occluded_after**（occlusion_alignment，terminal，`p1_fc90ae15d987`）

```
<image>
If the camera moves sideways to the right by 0.6 meters from the current view, will the flag be hidden behind the cabinet?
(A) partly hidden behind the cabinet
(B) fully hidden behind the cabinet
(C) not hidden behind the cabinet
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_2f62dee2afc8576e473112c7__s1.png`

**occluded_now**（occlusion_alignment，terminal，`p1_bbbafd51f62d`）

```
<image>
In the current view, is the coffee table hidden behind the shoe cabinet?
(A) not hidden behind the shoe cabinet
(B) fully hidden behind the shoe cabinet
(C) partly hidden behind the shoe cabinet
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_1c921dddb59338d0666a1e23__s4.png`

**occluded_now**（occlusion_alignment，start，`p1_13dfcf5b2787`）

```
<image>
In the current view, is the plant next to the coffee table hidden behind the TV cabinet?
(A) partly hidden behind the TV cabinet
(B) fully hidden behind the TV cabinet
(C) not hidden behind the TV cabinet
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_3826e7b6f3f45347673c01b6__s0.png`

