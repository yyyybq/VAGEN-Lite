# P1 转移题：统计

由 `python -m data_gen.transition_qa.report` 从 `exps/transition_qa_20261019` 生成。题库怎么来的、各题类的答案用哪个函数算，见 `PROGRESS.md` 第 1.3 节和第 2 节。

## 1. 题数和三类题的比例

| 文件 | 题数 | 状态题 (a) | 前向题 (b) | 逆向题 (c) |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | 90101 | 34812（38.6%） | 41774（46.4%） | 13515（15.0%） |
| p1_test.jsonl | 6035 | 2332（38.6%） | 2798（46.4%） | 905（15.0%） |
| p1_probe.jsonl | 600 | 280（46.7%） | 280（46.7%） | 40（6.7%） |

三类题的规则（用户 2026-10-19）：状态题全部保留（答案平衡之后）；前向题最多是状态题的 1.2 倍；逆向题最多占题库的 15%；"动作后任务要求会满足吗"（goal_after）最多占前向题的一半；单个场景最多 8%。超出的按（题类、答案、是否变化）分层抽。

- train 场景：前向题 (b)可用 62385、留下 41774；逆向题 (c)可用 24223、留下 13515；状态题 (a)可用 34812、留下 34812
  - 上限：前向题 41774 道，逆向题 13515 道
  - goal 上限去掉：前向题 (b) 10527 道
- val 场景：前向题 (b)可用 7958、留下 2798；逆向题 (c)可用 3265、留下 905；状态题 (a)可用 2332、留下 2332
  - 上限：前向题 2798 道，逆向题 905 道
  - goal 上限去掉：前向题 (b) 2012 道
- 测试集去掉了探针集的 600 道题（两份不重叠）。
- 探针集：600 道，每个题类 40 道，来自 97 个 val 场景；每个题类都够。

## 2. 按任务族 × 题类 × 划分的题数

**p1_train.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 2173 | 0 | 0 | 0 | 0 | 8022 | 2397 | 5881 | 5398 | 0 | 0 | 0 | 0 | 0 | 0 | 23871 |
| equidistance | 2233 | 0 | 0 | 2548 | 1444 | 0 | 0 | 2465 | 3342 | 0 | 0 | 0 | 0 | 0 | 0 | 12032 |
| fov_inclusion | 1960 | 3812 | 1711 | 0 | 0 | 0 | 0 | 1952 | 3277 | 0 | 0 | 0 | 0 | 0 | 0 | 12712 |
| occlusion_alignment | 376 | 0 | 0 | 0 | 0 | 0 | 0 | 674 | 752 | 0 | 0 | 0 | 0 | 577 | 267 | 2646 |
| projective_relations | 4524 | 0 | 0 | 0 | 0 | 0 | 0 | 5679 | 7038 | 0 | 0 | 2751 | 3515 | 0 | 0 | 23507 |
| screen_occupancy | 2249 | 0 | 0 | 0 | 0 | 0 | 0 | 4236 | 4865 | 3177 | 806 | 0 | 0 | 0 | 0 | 15333 |
| 合计 | 13515 | 3812 | 1711 | 2548 | 1444 | 8022 | 2397 | 20887 | 24672 | 3177 | 806 | 2751 | 3515 | 577 | 267 | 90101 |

**p1_test.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 146 | 0 | 0 | 0 | 0 | 627 | 366 | 403 | 284 | 0 | 0 | 0 | 0 | 0 | 0 | 1826 |
| equidistance | 159 | 0 | 0 | 179 | 160 | 0 | 0 | 166 | 158 | 0 | 0 | 0 | 0 | 0 | 0 | 822 |
| fov_inclusion | 108 | 253 | 159 | 0 | 0 | 0 | 0 | 101 | 149 | 0 | 0 | 0 | 0 | 0 | 0 | 770 |
| occlusion_alignment | 47 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 33 | 0 | 0 | 0 | 0 | 23 | 15 | 171 |
| projective_relations | 298 | 0 | 0 | 0 | 0 | 0 | 0 | 360 | 333 | 0 | 0 | 166 | 354 | 0 | 0 | 1511 |
| screen_occupancy | 147 | 0 | 0 | 0 | 0 | 0 | 0 | 316 | 250 | 151 | 71 | 0 | 0 | 0 | 0 | 935 |
| 合计 | 905 | 253 | 159 | 179 | 160 | 627 | 366 | 1399 | 1207 | 151 | 71 | 166 | 354 | 23 | 15 | 6035 |

**p1_probe.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 25 | 0 | 0 | 0 | 0 | 40 | 40 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 121 |
| equidistance | 9 | 0 | 0 | 40 | 40 | 0 | 0 | 6 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 103 |
| fov_inclusion | 3 | 40 | 40 | 0 | 0 | 0 | 0 | 5 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 94 |
| occlusion_alignment | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 6 | 0 | 0 | 0 | 0 | 40 | 40 | 97 |
| projective_relations | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 6 | 0 | 0 | 40 | 40 | 0 | 0 | 93 |
| screen_occupancy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 40 | 40 | 0 | 0 | 0 | 0 | 92 |
| 合计 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 600 |

## 3. 答案分布

"字母"是正确选项的字母；"偏差"是各字母占比和均匀分布之差的最大值（验收线 5 个百分点）。"答案值"是选项的含义（例如 left / right、full / partial / none）。

**p1_train.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 13515 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | move_backwardx1 1616，turn_rightx1 1291，turn_leftx1 1274，move_backwardx2 1007，move_leftx1 905，move_rightx1 899 |
| both_in_view_after | 3812 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_a 995，neither 980，only_b 972，both 865 |
| both_in_view_now | 1711 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_a 442，only_b 442，both 442，neither 385 |
| closer_after | 2548 | A 50.0% / B 50.0% | 0.0 | b 1277，a 1271 |
| closer_now | 1444 | A 50.0% / B 50.0% | 0.0 | b 734，a 710 |
| distance_bin_after | 8022 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 2.0 1694，2.5 1660，3.0 1300，1.5 1106，3.5 797，1.0 570 |
| distance_bin_now | 2397 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 2.5 704，2.0 430，3.0 388，1.5 359，3.5 185，1.0 155 |
| goal_after | 20887 | Yes 47.6% / No 52.4% | 2.4 | no 10941，yes 9946 |
| goal_now | 24672 | Yes 47.6% / No 52.4% | 2.4 | no 12923，yes 11749 |
| height_bin_after | 3177 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 0.3 872，0.2 637，0.4 483，0.5 312，0.6 268，0.7 211 |
| height_bin_now | 806 | A 25.6% / B 25.6% / C 25.6% / D 23.3% | 1.7 | 0.3 226，0.4 189，0.2 139，0.6 68，0.5 58，0.7 50 |
| left_right_after | 2751 | A 52.4% / B 47.6% | 2.4 | left 1441，right 1310 |
| left_right_now | 3515 | A 52.4% / B 47.6% | 2.4 | left 1841，right 1674 |
| occluded_after | 577 | A 33.3% / B 33.4% / C 33.3% | 0.1 | partial 208，full 197，none 172 |
| occluded_now | 267 | A 33.3% / B 33.3% / C 33.3% | 0.0 | full 93，none 93，partial 81 |

**p1_test.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 905 | A 25.0% / B 25.0% / C 25.1% / D 25.0% | 0.1 | move_backwardx1 117，turn_leftx1 97，turn_rightx1 93，move_backwardx2 78，move_leftx1 66，move_rightx1 58 |
| both_in_view_after | 253 | A 24.9% / B 24.9% / C 24.9% / D 25.3% | 0.3 | both 69，only_b 64，neither 61，only_a 59 |
| both_in_view_now | 159 | A 24.5% / B 25.2% / C 25.2% / D 25.2% | 0.5 | only_a 41，both 41，only_b 41，neither 36 |
| closer_after | 179 | A 49.7% / B 50.3% | 0.3 | b 98，a 81 |
| closer_now | 160 | A 50.0% / B 50.0% | 0.0 | b 84，a 76 |
| distance_bin_after | 627 | A 25.2% / B 25.7% / C 25.7% / D 23.4% | 1.6 | 2.5 136，2.0 124，3.0 109，1.5 78，3.5 60，4.0 41 |
| distance_bin_now | 366 | A 26.2% / B 24.9% / C 24.0% / D 24.9% | 1.2 | 2.5 101，3.0 66，1.5 60，2.0 55，3.5 33，4.0 26 |
| goal_after | 1399 | Yes 47.7% / No 52.3% | 2.3 | no 732，yes 667 |
| goal_now | 1207 | Yes 47.6% / No 52.4% | 2.4 | no 632，yes 575 |
| height_bin_after | 151 | A 25.2% / B 25.8% / C 25.8% / D 23.2% | 1.8 | 0.3 45，0.2 28，0.4 19，0.6 17，0.1 13，0.5 12 |
| height_bin_now | 71 | A 25.4% / B 25.4% / C 25.4% / D 23.9% | 1.1 | 0.2 19，0.4 15，0.3 14，0.6 6，0.1 5，0.5 5 |
| left_right_after | 166 | A 52.4% / B 47.6% | 2.4 | left 87，right 79 |
| left_right_now | 354 | A 52.3% / B 47.7% | 2.3 | left 185，right 169 |
| occluded_after | 23 | A 30.4% / B 34.8% / C 34.8% | 2.9 | full 9，partial 7，none 7 |
| occluded_now | 15 | A 33.3% / B 33.3% / C 33.3% | 0.0 | full 8，partial 7 |

**p1_probe.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | turn_leftx4 2，move_leftx5 2，move_leftx3 2，move_rightx5 2，move_forwardx5 2，turn_rightx5 2 |
| both_in_view_after | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_a 11，neither 10，only_b 10，both 9 |
| both_in_view_now | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_a 10，neither 10，only_b 10，both 10 |
| closer_after | 40 | A 50.0% / B 50.0% | 0.0 | a 20，b 20 |
| closer_now | 40 | A 50.0% / B 50.0% | 0.0 | b 20，a 20 |
| distance_bin_after | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | 3.0 8，2.5 7，2.0 7，1.5 5，4.0 3，3.5 3 |
| distance_bin_now | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | 2.5 10，3.0 8，2.0 6，1.5 4，4.0 4，1.0 3 |
| goal_after | 40 | Yes 50.0% / No 50.0% | 0.0 | no 20，yes 20 |
| goal_now | 40 | Yes 50.0% / No 50.0% | 0.0 | no 20，yes 20 |
| height_bin_after | 40 | A 22.5% / B 22.5% / C 27.5% / D 27.5% | 2.5 | 0.3 11，0.2 7，0.5 6，0.7 5，0.6 4，0.4 3 |
| height_bin_now | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | 0.3 12，0.4 10，0.2 6，0.5 5，0.7 3，0.8 3 |
| left_right_after | 40 | A 50.0% / B 50.0% | 0.0 | right 20，left 20 |
| left_right_now | 40 | A 50.0% / B 50.0% | 0.0 | left 20，right 20 |
| occluded_after | 40 | A 32.5% / B 35.0% / C 32.5% | 1.7 | partial 15，full 14，none 11 |
| occluded_now | 40 | A 32.5% / B 32.5% / C 35.0% | 1.7 | partial 17，full 16，none 7 |

训练集和测试集里字母偏差的最大值：2.9 个百分点（探针集每类只有约 40 道，不计入；它的偏差见上表）。

## 4. 前向题：没变化的题、答案和现在不同的题

"没变化"：被问的量在动作前后的变化小于门槛（例如转身后到物体的距离不变）；上限 10%。"答案变了"：动作后的答案和现在的答案不同；控制在 35%–65%，防止模型学成"把现在的答案反过来"。

| 文件 | 题类 | 题数 | 没变化 | 答案变了 |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | both_in_view_after | 3812 | 10.0% | 64.9% |
| p1_train.jsonl | closer_after | 2548 | 10.0% | 45.6% |
| p1_train.jsonl | distance_bin_after | 8022 | 10.0% | 65.0% |
| p1_train.jsonl | goal_after | 20887 | 9.6% | 53.0% |
| p1_train.jsonl | height_bin_after | 3177 | 9.6% | 64.2% |
| p1_train.jsonl | left_right_after | 2751 | 10.0% | 0.0% |
| p1_train.jsonl | occluded_after | 577 | 9.5% | 59.4% |
| p1_test.jsonl | both_in_view_after | 253 | 9.9% | 66.4% |
| p1_test.jsonl | closer_after | 179 | 10.1% | 49.5% |
| p1_test.jsonl | distance_bin_after | 627 | 10.0% | 64.3% |
| p1_test.jsonl | goal_after | 1399 | 9.6% | 51.8% |
| p1_test.jsonl | height_bin_after | 151 | 9.9% | 62.9% |
| p1_test.jsonl | left_right_after | 166 | 9.6% | 0.0% |
| p1_test.jsonl | occluded_after | 23 | 8.7% | 50.0% |
| p1_probe.jsonl | both_in_view_after | 40 | 10.0% | 52.0% |
| p1_probe.jsonl | closer_after | 40 | 10.0% | 53.8% |
| p1_probe.jsonl | distance_bin_after | 40 | 10.0% | 42.9% |
| p1_probe.jsonl | goal_after | 40 | 10.0% | 50.0% |
| p1_probe.jsonl | height_bin_after | 40 | 10.0% | 51.9% |
| p1_probe.jsonl | left_right_after | 40 | 10.0% | 0.0% |
| p1_probe.jsonl | occluded_after | 40 | 10.0% | 60.0% |

## 5. 过滤：没出的题和原因

每条轨迹的每个不同的帧都当锚点状态（2026-10-18 的题库每条轨迹最多 3 个）。原始候选题 254085 道（训练和 val 场景合计），下表是在候选阶段就没出的题（按题类和原因）；之后的平衡、比例、场景上限去掉的数量见第 6 节。

| 题类 | 出了 | 没出（原因 数量） |
| --- | --- | --- |
| action_between | 27528 |  |
| both_in_view_after | 13850 | near_boundary 39954 |
| both_in_view_now | 3389 | near_boundary 2068 |
| closer_after | 5808 | not_visible_now 62314，near_boundary 7967 |
| closer_now | 1644 | not_visible 3743，near_boundary 640 |
| distance_bin_after | 18825 | near_boundary 42642，not_visible_now 5684，below_options 311 |
| distance_bin_now | 3465 | near_boundary 2931，not_visible 367，below_options 8 |
| forward | 0 | blocked_move 76203 |
| goal_after | 113661 | near_boundary_yes 23071，near_boundary_no 2420 |
| goal_now | 39404 | near_boundary_yes 6294，near_boundary_no 468 |
| height_bin_after | 11281 | cut_off_vertically 39469，near_boundary 19967，behind_camera 16361，not_visible_now 4098，above_options 59，below_options 2 |
| height_bin_now | 2186 | cut_off_vertically 2489，near_boundary 1771，not_visible 273，behind_camera 77，above_options 3 |
| left_right_after | 6637 | not_visible_now 113675，out_of_frame 50279，beyond_45deg 8025，behind_camera 4571，near_boundary 912 |
| left_right_now | 4158 | not_visible 7027，beyond_45deg 1066，near_boundary 144 |
| occluded_after | 1635 | not_visible_now 4612，target_out_of_frame 2435，target_not_visible 2371，occluder_out_of_view 2294，near_boundary 773，target_behind_camera 85 |
| occluded_now | 614 | occluder_not_visible 300，target_not_visible 114，target_out_of_frame 89，near_boundary 70，target_behind_camera 2 |

- 锚点状态 38638 个；一致性检查：{"trajectories": 11110}（`terminal_not_success`、`success_before_terminal` 都没有出现就是 0）。

原因的含义：`not_visible` 被问物体在当前画面里不满足可见规则；`near_boundary` 离判定阈值太近；`beyond_45deg` 左右题里两物体的连线和相机左右轴夹角超过 45°；`blocked_move` 动作会被碰撞规则挡住；`behind_camera` / `out_of_frame` / `cut_off_vertically` 物体在相机后方、出画、上下不完整；`blocked_turn` 逆向题的那个回合被挡住了（两张图一样）。

## 6. 平衡和上限去掉的题

**train 场景**

| 题类 | 平衡前 | 去掉（原因） | 平衡后 |
| --- | --- | --- | --- |
| action_between | 24223 | {"answer_balance": 0} | 24223 |
| both_in_view_after | 12332 | {"flip_band": 928, "unchanged": 2188, "answer_balance": 4483} | 4733 |
| both_in_view_now | 3004 | {"answer_balance": 1293} | 1711 |
| closer_after | 5096 | {"flip_band": 1142, "unchanged": 792, "answer_balance": 0} | 3162 |
| closer_now | 1444 | {"answer_balance": 0} | 1444 |
| distance_bin_after | 16474 | {"flip_band": 1648, "unchanged": 2040, "answer_balance": 2565} | 10221 |
| distance_bin_now | 3028 | {"answer_balance": 547} | 2481 |
| goal_after | 100047 | {"flip_band": 27070, "unchanged": 26327, "answer_balance": 9326} | 37324 |
| goal_now | 35559 | {"answer_balance": 10299} | 25260 |
| height_bin_after | 9908 | {"flip_band": 0, "unchanged": 2238, "answer_balance": 3587} | 4083 |
| height_bin_now | 1898 | {"answer_balance": 1062} | 836 |
| left_right_after | 5870 | {"flip_band": 0, "unchanged": 2324, "answer_balance": 51} | 3495 |
| left_right_now | 3673 | {"answer_balance": 74} | 3599 |
| occluded_after | 1436 | {"flip_band": 89, "unchanged": 419, "answer_balance": 210} | 718 |
| occluded_now | 537 | {"answer_balance": 270} | 267 |

场景上限（单个场景不超过 8%）去掉：{}

**val 场景**

| 题类 | 平衡前 | 去掉（原因） | 平衡后 |
| --- | --- | --- | --- |
| action_between | 3265 | {"answer_balance": 0} | 3265 |
| both_in_view_after | 1478 | {"flip_band": 107, "unchanged": 285, "answer_balance": 546} | 540 |
| both_in_view_now | 345 | {"answer_balance": 186} | 159 |
| closer_after | 672 | {"flip_band": 186, "unchanged": 106, "answer_balance": 0} | 380 |
| closer_now | 160 | {"answer_balance": 0} | 160 |
| distance_bin_after | 2311 | {"flip_band": 272, "unchanged": 312, "answer_balance": 373} | 1354 |
| distance_bin_now | 397 | {"answer_balance": 26} | 371 |
| goal_after | 13574 | {"flip_band": 3646, "unchanged": 3573, "answer_balance": 1251} | 5104 |
| goal_now | 3805 | {"answer_balance": 2569} | 1236 |
| height_bin_after | 1333 | {"flip_band": 7, "unchanged": 351, "answer_balance": 645} | 330 |
| height_bin_now | 248 | {"answer_balance": 174} | 74 |
| left_right_after | 727 | {"flip_band": 0, "unchanged": 295, "answer_balance": 71} | 361 |
| left_right_now | 445 | {"answer_balance": 82} | 363 |
| occluded_after | 159 | {"flip_band": 46, "unchanged": 39, "answer_balance": 26} | 48 |
| occluded_now | 37 | {"answer_balance": 22} | 15 |

场景上限（单个场景不超过 8%）去掉：{}

## 7. 每类两道原题

**action_between**（projective_relations，start，`p1_65e1d95c8cda`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera moved sideways to the right by 1.2 meters
(B) the camera moved sideways to the left by 0.9 meters
(C) the camera turned right by 60 degrees
(D) the camera moved sideways to the right by 0.9 meters
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_6c18f2681f04ddbfec5b1bdb__s0.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_6c18f2681f04ddbfec5b1bdb__s3.png`

**action_between**（projective_relations，start，`p1_2eea2e8d4cdf`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera moved sideways to the left by 0.6 meters
(B) the camera moved sideways to the left by 0.3 meters
(C) the camera moved sideways to the right by 0.3 meters
(D) the camera turned left by 20 degrees
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_a3910b05a73e89724a95c343__s0.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_a3910b05a73e89724a95c343__s1.png`

**both_in_view_after**（fov_inclusion，terminal，`p1_6ab695ffb4d2`）

```
<image>
If the camera turns right by 40 degrees from the current view, will the smaller pet bed and the entrance hall cabinet next to the dining table both be inside the view?
(A) only the entrance hall cabinet next to the dining table
(B) both are
(C) neither
(D) only the smaller pet bed
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_8a10dd7eba7fb34d421d4cb0__s7.png`

**both_in_view_after**（fov_inclusion，middle，`p1_b1ec9455341c`）

```
<image>
If the camera moves forward by 0.9 meters from the current view, will the side table and the wardrobe both be inside the view?
(A) only the wardrobe
(B) neither
(C) both are
(D) only the side table
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_634e793b9f519b7ba9094013__s1.png`

**both_in_view_now**（fov_inclusion，middle，`p1_0e8295b60186`）

```
<image>
In the current view, are the water bar and the table next to the wardrobe both inside the view?
(A) only the table next to the wardrobe
(B) neither
(C) both are
(D) only the water bar
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_3716ae1f34710cf6dcae7b10__s2.png`

**both_in_view_now**（fov_inclusion，middle，`p1_155930cabd52`）

```
<image>
In the current view, are the TV and the sofa both inside the view?
(A) both are
(B) only the TV
(C) only the sofa
(D) neither
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_8ef97eebeab0d2d3286a2482__s1.png`

**closer_after**（equidistance，terminal，`p1_db70e9661ef9`）

```
<image>
If the camera turns left by 60 degrees from the current view, which will be closer to you (horizontal distance to its nearest side): the coffee table or the floor lamp?
(A) the coffee table
(B) the floor lamp
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_2b5641a2ae6c8c2de44dae80__s7.png`

**closer_after**（equidistance，terminal，`p1_260b7241e042`）

```
<image>
If the camera moves sideways to the right by 0.6 meters from the current view, which will be closer to you (horizontal distance to its nearest side): the floor lamp or the side table?
(A) the side table
(B) the floor lamp
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_9b19a2a1c6bd739aa76621ae__s4.png`

**closer_now**（equidistance，terminal，`p1_81e98a8bcd3a`）

```
<image>
In the current view, which is closer to you (horizontal distance to its nearest side): the flowers or the dining table?
(A) the flowers
(B) the dining table
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_d29074e3919374e301af6077__s4.png`

**closer_now**（equidistance，middle，`p1_d449a1657649`）

```
<image>
In the current view, which is closer to you (horizontal distance to its nearest side): the toy or the armchair?
(A) the armchair
(B) the toy
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_31fee75af52f61f97d54d73b__s2.png`

**distance_bin_after**（absolute_positioning，terminal，`p1_0fdf566eefb0`）

```
<image>
If the camera moves sideways to the right by 0.9 meters from the current view, about how far will you be from the coffee table, measured horizontally to its nearest side?
(A) about 1.5 meters
(B) about 2.0 meters
(C) about 2.5 meters
(D) about 3.0 meters
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_09ce417fe1543e300aedf173__s3.png`

**distance_bin_after**（absolute_positioning，middle，`p1_ee061966b7eb`）

```
<image>
If the camera moves forward by 0.3 meters from the current view, about how far will you be from the chair next to the side table, measured horizontally to its nearest side?
(A) about 1.5 meters
(B) about 2.0 meters
(C) about 2.5 meters
(D) about 3.0 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_5ecc248fa899b03306fee038__s6.png`

**distance_bin_now**（absolute_positioning，start，`p1_df3a4edfb37c`）

```
<image>
In the current view, about how far are you from the bed, measured horizontally to its nearest side?
(A) about 0.5 meters
(B) about 1.0 meters
(C) about 1.5 meters
(D) about 2.0 meters
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_2e27c33fba722cd17f4d8430__s0.png`

**distance_bin_now**（absolute_positioning，start，`p1_5b409c7cc52d`）

```
<image>
In the current view, about how far are you from the reception, measured horizontally to its nearest side?
(A) about 1.0 meters
(B) about 1.5 meters
(C) about 2.0 meters
(D) about 2.5 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_61aa08bb2e7c51632d3ac523__s0.png`

**goal_after**（occlusion_alignment，terminal，`p1_0f2778020ad4`）

```
<image>
Task: Move so that the coffee table is hidden behind the sofa from your viewpoint, while the sofa stays in view.
If the camera turns right by 60 degrees from the current view, will the task's requirement be met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_39c651ebf36899362cbd9d4a__s3.png`

**goal_after**（projective_relations，middle，`p1_22a39e3a11e7`）

```
<image>
Task: Move so that the smaller wardrobe appears to the right of the reception in your view.
If the camera turns right by 20 degrees from the current view, will the task's requirement be met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b489b8e6cd203ded3825f392__s2.png`

**goal_now**（absolute_positioning，start，`p1_92cc9c6ba8e4`）

```
<image>
Task: Move to a spot about 2.5 meters away from the plant next to the dining table (measured to its nearest side), keeping the plant next to the dining table in view.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_e04df14c36dd146c82c4f1fc__s0.png`

**goal_now**（fov_inclusion，interior，`p1_8d92b4c5beb8`）

```
<image>
Task: Move so that both the TV and the sofa are fully in your view.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：Yes
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261018/interior/frames/plain/v10_bdac8f56c10f2f059107c6cf__i0.png`

**height_bin_after**（screen_occupancy，middle，`p1_1853775a0e03`）

```
<image>
If the camera turns right by 60 degrees from the current view, about what fraction of the image height will the fireplace fill, from its top to its bottom?
(A) about 50%
(B) about 60%
(C) about 70%
(D) about 80%
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_fc66a4c98daf469dccb52d52__s1.png`

**height_bin_after**（screen_occupancy，middle，`p1_5007ef53ad7b`）

```
<image>
If the camera turns right by 20 degrees from the current view, about what fraction of the image height will the washing machine fill, from its top to its bottom?
(A) about 30%
(B) about 40%
(C) about 50%
(D) about 60%
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_01ddaaa34e37134c4f228645__s1.png`

**height_bin_now**（screen_occupancy，terminal，`p1_1e04a59f5333`）

```
<image>
In the current view, about what fraction of the image height does the ornament fill, from its top to its bottom?
(A) about 20%
(B) about 30%
(C) about 40%
(D) about 50%
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_4ebeae214d5f8449237e07ce__x3.png`

**height_bin_now**（screen_occupancy，start，`p1_aafad4f58446`）

```
<image>
In the current view, about what fraction of the image height does the wardrobe next to the coffee table fill, from its top to its bottom?
(A) about 10%
(B) about 20%
(C) about 30%
(D) about 40%
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_ca508f3abdcfea030c4e9d84__s0.png`

**left_right_after**（projective_relations，terminal，`p1_a59add50e485`）

```
<image>
If the camera moves forward by 0.3 meters from the current view, will the wardrobe be to the left or to the right of the bed?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_5ed2c2b979a0a02d489edc6d__s5.png`

**left_right_after**（projective_relations，terminal，`p1_16c969278df3`）

```
<image>
If the camera moves sideways to the right by 0.6 meters from the current view, will the armchair be to the left or to the right of the coffee table?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b439b01e2efdbedcfaa606c5__s3.png`

**left_right_now**（projective_relations，terminal，`p1_f6261dfaf4eb`）

```
<image>
In the current view, is the wardrobe next to the floor lamp to the left or to the right of the chair next to the floor lamp?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_36620dfaad12e747cb503e65__s5.png`

**left_right_now**（projective_relations，terminal，`p1_a7941f96d1fd`）

```
<image>
In the current view, is the floor lamp to the left or to the right of the table?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_5dc8322cb1ebc54fe8c86437__s4.png`

**occluded_after**（occlusion_alignment，terminal，`p1_33a9dea304c7`）

```
<image>
If the camera moves forward by 0.6 meters from the current view, will the table next to the drum kit be hidden behind the larger coffee maker?
(A) not hidden behind the larger coffee maker
(B) partly hidden behind the larger coffee maker
(C) fully hidden behind the larger coffee maker
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_7816ca65144f58360fa4cdde__s7.png`

**occluded_after**（occlusion_alignment，middle，`p1_64417073f7f9`）

```
<image>
If the camera moves sideways to the right by 0.9 meters from the current view, will the chair next to the wardrobe be hidden behind the water bar?
(A) fully hidden behind the water bar
(B) not hidden behind the water bar
(C) partly hidden behind the water bar
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_e1a98254896167a6eda9769e__s3.png`

**occluded_now**（occlusion_alignment，middle，`p1_25717f0296b7`）

```
<image>
In the current view, is the vanity cabinet hidden behind the shower head?
(A) not hidden behind the shower head
(B) partly hidden behind the shower head
(C) fully hidden behind the shower head
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_96b0792817a178036885cedd__s1.png`

**occluded_now**（occlusion_alignment，start，`p1_c9eeebb0f894`）

```
<image>
In the current view, is the sofa next to the wardrobe hidden behind the pos next to the shelf?
(A) partly hidden behind the pos next to the shelf
(B) not hidden behind the pos next to the shelf
(C) fully hidden behind the pos next to the shelf
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_127b757bddac30cf9e84e0c7__s0.png`

