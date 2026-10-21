# P1 转移题：统计

由 `python -m data_gen.transition_qa.report` 从 `exps/transition_qa_20261019b` 生成。题库怎么来的、各题类的答案用哪个函数算，见 `PROGRESS.md` 第 1.3 节和第 2 节。

## 1. 题数和三类题的比例

| 文件 | 题数 | 状态题 (a) | 前向题 (b) | 逆向题 (c) |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | 87437 | 35922（41.1%） | 43106（49.3%） | 8409（9.6%） |
| p1_test.jsonl | 5985 | 2313（38.6%） | 2775（46.4%） | 897（15.0%） |
| p1_probe.jsonl | 952 | 441（46.3%） | 471（49.5%） | 40（4.2%） |

三类题的规则（用户 2026-10-19）：状态题全部保留（答案平衡之后）；前向题最多是状态题的 1.2 倍；逆向题最多占题库的 15%；"动作后任务要求会满足吗"（goal_after）最多占前向题的一半；单个场景最多 8%。超出的按（题类、答案、是否变化）分层抽。

- train 场景：前向题 (b)可用 54398、留下 43106；逆向题 (c)可用 8409、留下 8409；状态题 (a)可用 35922、留下 35922
  - 上限：前向题 43106 道，逆向题 13946 道
- val 场景：前向题 (b)可用 6697、留下 2775；逆向题 (c)可用 1015、留下 897；状态题 (a)可用 2313、留下 2313
  - 上限：前向题 2775 道，逆向题 897 道
- 测试集去掉了探针集的 952 道题（两份不重叠）。
- 探针集：952 道，每个题类 40 道，其中 goal_after 200 道、goal_now 200 道、occluded_after 全部可用的 71 道、occluded_now 全部可用的 41 道（预注册要的题类），来自 97 个 val 场景；其余题类都够；occluded_after 另有 37 道"没变化"的超过 10% 上限，没取。

## 2. 按任务族 × 题类 × 划分的题数

**p1_train.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 1310 | 0 | 0 | 0 | 0 | 9640 | 2686 | 6998 | 6909 | 0 | 0 | 0 | 0 | 0 | 0 | 27543 |
| equidistance | 1477 | 0 | 0 | 2497 | 1444 | 0 | 0 | 1709 | 2577 | 0 | 0 | 0 | 0 | 0 | 0 | 9704 |
| fov_inclusion | 1245 | 3314 | 2459 | 0 | 0 | 0 | 0 | 1550 | 2981 | 0 | 0 | 0 | 0 | 0 | 0 | 11549 |
| occlusion_alignment | 238 | 0 | 0 | 0 | 0 | 0 | 0 | 731 | 711 | 0 | 0 | 0 | 0 | 355 | 227 | 2262 |
| projective_relations | 2828 | 0 | 0 | 0 | 0 | 0 | 0 | 4699 | 5724 | 0 | 0 | 2645 | 3515 | 0 | 0 | 19411 |
| screen_occupancy | 1311 | 0 | 0 | 0 | 0 | 0 | 0 | 5083 | 5770 | 3885 | 919 | 0 | 0 | 0 | 0 | 16968 |
| 合计 | 8409 | 3314 | 2459 | 2497 | 1444 | 9640 | 2686 | 20770 | 24672 | 3885 | 919 | 2645 | 3515 | 355 | 227 | 87437 |

**p1_test.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 140 | 0 | 0 | 0 | 0 | 694 | 361 | 510 | 386 | 0 | 0 | 0 | 0 | 0 | 0 | 2091 |
| equidistance | 144 | 0 | 0 | 160 | 160 | 0 | 0 | 83 | 57 | 0 | 0 | 0 | 0 | 0 | 0 | 604 |
| fov_inclusion | 122 | 172 | 252 | 0 | 0 | 0 | 0 | 88 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 730 |
| occlusion_alignment | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 41 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 76 |
| projective_relations | 294 | 0 | 0 | 0 | 0 | 0 | 0 | 322 | 221 | 0 | 0 | 139 | 354 | 0 | 0 | 1330 |
| screen_occupancy | 164 | 0 | 0 | 0 | 0 | 0 | 0 | 336 | 277 | 230 | 147 | 0 | 0 | 0 | 0 | 1154 |
| 合计 | 897 | 172 | 252 | 160 | 160 | 694 | 361 | 1380 | 1039 | 230 | 147 | 139 | 354 | 0 | 0 | 5985 |

**p1_probe.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 25 | 0 | 0 | 0 | 0 | 40 | 40 | 37 | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 180 |
| equidistance | 9 | 0 | 0 | 40 | 40 | 0 | 0 | 34 | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 160 |
| fov_inclusion | 2 | 40 | 40 | 0 | 0 | 0 | 0 | 34 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 152 |
| occlusion_alignment | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 19 | 0 | 0 | 0 | 0 | 71 | 41 | 157 |
| projective_relations | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 36 | 0 | 0 | 40 | 40 | 0 | 0 | 154 |
| screen_occupancy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 34 | 40 | 40 | 0 | 0 | 0 | 0 | 149 |
| 合计 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 200 | 200 | 40 | 40 | 40 | 40 | 71 | 41 | 952 |

## 3. 答案分布

"字母"是正确选项的字母；"偏差"是各字母占比和均匀分布之差的最大值（验收线 5 个百分点）。"答案值"是选项的含义（例如 left / right、full / partial / none）。

**p1_train.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 8409 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | turn_leftx2 730，turn_rightx2 712，move_leftx2 521，turn_rightx1 506，turn_leftx1 499，move_rightx2 475 |
| both_in_view_after | 3314 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_b 968，only_a 932，neither 836，both 578 |
| both_in_view_now | 2459 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | both 770，only_b 672，only_a 632，neither 385 |
| closer_after | 2497 | A 50.0% / B 50.0% | 0.0 | b 1275，a 1222 |
| closer_now | 1444 | A 50.0% / B 50.0% | 0.0 | b 734，a 710 |
| distance_bin_after | 9640 | A 24.8% / B 25.2% / C 25.2% / D 24.8% | 0.2 | 2.0 2060，2.5 1937，3.0 1548，1.5 1456，3.5 914，1.0 745 |
| distance_bin_now | 2686 | A 24.8% / B 24.3% / C 26.2% / D 24.6% | 1.2 | 2.5 669，2.0 532，3.0 493，1.5 468，3.5 195，1.0 152 |
| goal_after | 20770 | Yes 49.7% / No 50.3% | 0.3 | no 10454，yes 10316 |
| goal_now | 24672 | Yes 47.6% / No 52.4% | 2.4 | no 12923，yes 11749 |
| height_bin_after | 3885 | A 25.6% / B 25.8% / C 25.0% / D 23.5% | 1.5 | 0.3 932，0.2 923，0.4 819，0.5 511，0.1 334，0.7 101 |
| height_bin_now | 919 | A 24.4% / B 25.8% / C 25.8% / D 24.0% | 1.0 | 0.2 223，0.4 223，0.3 223，0.5 106，0.1 89，0.7 14 |
| left_right_after | 2645 | A 52.1% / B 47.9% | 2.1 | left 1378，right 1267 |
| left_right_now | 3515 | A 52.4% / B 47.6% | 2.4 | left 1841，right 1674 |
| occluded_after | 355 | A 33.2% / B 33.5% / C 33.2% | 0.2 | none 121，full 118，partial 116 |
| occluded_now | 227 | A 33.5% / B 33.0% / C 33.5% | 0.3 | full 79，none 79，partial 69 |

**p1_test.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 897 | A 25.0% / B 25.0% / C 25.1% / D 25.0% | 0.1 | turn_rightx2 74，turn_leftx2 74，move_backwardx2 59，turn_rightx1 56，move_forwardx2 56，move_rightx2 53 |
| both_in_view_after | 172 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_b 51，neither 49，only_a 44，both 28 |
| both_in_view_now | 252 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | only_a 72，both 72，only_b 72，neither 36 |
| closer_after | 160 | A 50.0% / B 50.0% | 0.0 | b 81，a 79 |
| closer_now | 160 | A 50.0% / B 50.0% | 0.0 | b 84，a 76 |
| distance_bin_after | 694 | A 24.8% / B 25.2% / C 25.1% / D 24.9% | 0.2 | 2.5 145，2.0 126，3.0 114，1.5 86，3.5 73，1.0 53 |
| distance_bin_now | 361 | A 25.5% / B 24.1% / C 26.3% / D 24.1% | 1.3 | 2.5 90，3.0 76，1.5 61，2.0 59，3.5 30，4.0 28 |
| goal_after | 1380 | Yes 50.0% / No 50.0% | 0.0 | yes 690，no 690 |
| goal_now | 1039 | Yes 47.6% / No 52.4% | 2.4 | no 544，yes 495 |
| height_bin_after | 230 | A 24.8% / B 26.1% / C 25.2% / D 23.9% | 1.1 | 0.2 53，0.3 50，0.4 45，0.1 29，0.5 24，0.8 12 |
| height_bin_now | 147 | A 24.5% / B 25.9% / C 25.9% / D 23.8% | 1.2 | 0.3 35，0.2 34，0.4 31，0.1 22，0.5 12，0.6 4 |
| left_right_after | 139 | A 52.5% / B 47.5% | 2.5 | left 73，right 66 |
| left_right_now | 354 | A 52.3% / B 47.7% | 2.3 | left 185，right 169 |

**p1_probe.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | move_leftx5 2，move_rightx5 2，turn_leftx4 2，move_forwardx5 2，turn_rightx4 2，move_leftx4 2 |
| both_in_view_after | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | neither 11，only_b 10，only_a 10，both 9 |
| both_in_view_now | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | both 10，only_a 10，neither 10，only_b 10 |
| closer_after | 40 | A 50.0% / B 50.0% | 0.0 | b 20，a 20 |
| closer_now | 40 | A 50.0% / B 50.0% | 0.0 | a 20，b 20 |
| distance_bin_after | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | 2.5 8，2.0 8，3.5 6，1.5 5，3.0 4，1.0 3 |
| distance_bin_now | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | 2.5 10，2.0 9，1.5 8，3.0 5，4.0 3，1.0 2 |
| goal_after | 200 | Yes 50.0% / No 50.0% | 0.0 | no 100，yes 100 |
| goal_now | 200 | Yes 50.0% / No 50.0% | 0.0 | no 100，yes 100 |
| height_bin_after | 40 | A 27.5% / B 22.5% / C 25.0% / D 25.0% | 2.5 | 0.3 10，0.4 8，0.2 7，0.1 6，0.6 4，0.5 3 |
| height_bin_now | 40 | A 25.0% / B 25.0% / C 25.0% / D 25.0% | 0.0 | 0.2 10，0.4 9，0.3 9，0.5 5，0.1 4，0.8 2 |
| left_right_after | 40 | A 50.0% / B 50.0% | 0.0 | right 20，left 20 |
| left_right_now | 40 | A 50.0% / B 50.0% | 0.0 | right 20，left 20 |
| occluded_after | 71 | A 32.4% / B 33.8% / C 33.8% | 0.9 | none 27，full 22，partial 22 |
| occluded_now | 41 | A 34.1% / B 31.7% / C 34.1% | 1.6 | full 19，partial 15，none 7 |

训练集和测试集里字母偏差的最大值：2.5 个百分点（探针集每类只有约 40 道，不计入；它的偏差见上表）。

## 4. 前向题：没变化的题、答案和现在不同的题

"没变化"：被问的量在动作前后的变化小于门槛（例如转身后到物体的距离不变）；上限 10%。"答案变了"：动作后的答案和现在的答案不同；控制在 35%–65%，防止模型学成"把现在的答案反过来"。

| 文件 | 题类 | 题数 | 没变化 | 答案变了 |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | both_in_view_after | 3314 | 10.0% | 65.4% |
| p1_train.jsonl | closer_after | 2497 | 10.0% | 45.5% |
| p1_train.jsonl | distance_bin_after | 9640 | 10.0% | 65.0% |
| p1_train.jsonl | goal_after | 20770 | 10.0% | 52.8% |
| p1_train.jsonl | height_bin_after | 3885 | 10.1% | 64.8% |
| p1_train.jsonl | left_right_after | 2645 | 10.0% | 0.0% |
| p1_train.jsonl | occluded_after | 355 | 9.0% | 61.1% |
| p1_test.jsonl | both_in_view_after | 172 | 10.5% | 66.1% |
| p1_test.jsonl | closer_after | 160 | 10.0% | 48.5% |
| p1_test.jsonl | distance_bin_after | 694 | 9.9% | 61.7% |
| p1_test.jsonl | goal_after | 1380 | 9.8% | 53.8% |
| p1_test.jsonl | height_bin_after | 230 | 8.7% | 66.0% |
| p1_test.jsonl | left_right_after | 139 | 8.6% | 0.0% |
| p1_probe.jsonl | both_in_view_after | 40 | 10.0% | 51.9% |
| p1_probe.jsonl | closer_after | 40 | 5.0% | 50.0% |
| p1_probe.jsonl | distance_bin_after | 40 | 10.0% | 42.9% |
| p1_probe.jsonl | goal_after | 200 | 10.0% | 50.0% |
| p1_probe.jsonl | height_bin_after | 40 | 10.0% | 46.4% |
| p1_probe.jsonl | left_right_after | 40 | 10.0% | 0.0% |
| p1_probe.jsonl | occluded_after | 71 | 9.9% | 55.8% |

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

**渲染深度核对遮挡题**（2026-10-21，`occ_depth.py`）：答"完全挡住"的要求目标在渲染图里被挡 ≥ 0.70，答"部分挡住"的 ≥ 0.10，否则在平衡之前去掉；答"没有"的不核对。

| 题类 / 答案 | 核对 | 去掉 | 没有深度结果（也去掉） |
| --- | --- | --- | --- |
| occluded_after/full | 829 | 372 | 0 |
| occluded_after/partial | 472 | 108 | 0 |
| occluded_now/full | 408 | 193 | 0 |
| occluded_now/partial | 105 | 21 | 0 |

**train 场景**

| 题类 | 平衡前 | 去掉（原因） | 平衡后 |
| --- | --- | --- | --- |
| action_between | 24223 | {"answer_balance": 15814} | 8409 |
| both_in_view_after | 12332 | {"flip_band": 526, "unchanged": 2217, "answer_balance": 5406} | 4183 |
| both_in_view_now | 3004 | {"answer_balance": 545} | 2459 |
| closer_after | 5096 | {"flip_band": 1142, "unchanged": 803} | 3151 |
| closer_now | 1444 | {} | 1444 |
| distance_bin_after | 16474 | {"flip_band": 2019, "unchanged": 2106, "answer_balance": 185} | 12164 |
| distance_bin_now | 3028 | {"answer_balance": 342} | 2686 |
| goal_after | 100047 | {"flip_band": 27070, "unchanged": 26499, "answer_balance": 20267} | 26211 |
| goal_now | 35559 | {"answer_balance": 10454} | 25105 |
| height_bin_after | 9908 | {"flip_band": 0, "unchanged": 2240, "answer_balance": 2712} | 4956 |
| height_bin_now | 1898 | {"answer_balance": 979} | 919 |
| left_right_after | 5870 | {"flip_band": 0, "unchanged": 2331, "answer_balance": 201} | 3338 |
| left_right_now | 3673 | {"answer_balance": 74} | 3599 |
| occluded_after | 1047 | {"flip_band": 0, "unchanged": 328, "answer_balance": 271} | 448 |
| occluded_now | 359 | {"answer_balance": 132} | 227 |

场景上限（单个场景不超过 8%）去掉：{}

**val 场景**

| 题类 | 平衡前 | 去掉（原因） | 平衡后 |
| --- | --- | --- | --- |
| action_between | 3265 | {"answer_balance": 2250} | 1015 |
| both_in_view_after | 1478 | {"flip_band": 97, "unchanged": 285, "answer_balance": 685} | 411 |
| both_in_view_now | 345 | {"answer_balance": 93} | 252 |
| closer_after | 672 | {"flip_band": 184, "unchanged": 102} | 386 |
| closer_now | 160 | {} | 160 |
| distance_bin_after | 2311 | {"flip_band": 327, "unchanged": 307, "answer_balance": 0} | 1677 |
| distance_bin_now | 397 | {"answer_balance": 34} | 363 |
| goal_after | 13414 | {"flip_band": 3714, "unchanged": 3502, "answer_balance": 2868} | 3330 |
| goal_now | 3645 | {"answer_balance": 2584} | 1061 |
| height_bin_after | 1333 | {"flip_band": 0, "unchanged": 351, "answer_balance": 424} | 558 |
| height_bin_now | 248 | {"answer_balance": 100} | 148 |
| left_right_after | 727 | {"flip_band": 0, "unchanged": 295, "answer_balance": 97} | 335 |
| left_right_now | 445 | {"answer_balance": 82} | 363 |
| occluded_after | 37 | {"flip_band": 0, "unchanged": 37, "answer_balance": 0} | 0 |

场景上限（单个场景不超过 8%）去掉：{}

## 7. 只看文字能答对多少

同一道题不给图、只给题干和选项时的准确率，和随机猜比；明显高于随机猜，说明题目文字泄露了答案。"学出来的规则"：只用训练集文字学到的最强一条（`shortcuts.py`，例如同一组选项在训练集里最常对的那个）；"原版 7B 不看图"：原版 Qwen2.5-VL-7B 只拿到题干和选项（`qa_probe run --no-images`，题前一句"看不到图也请选最可能的答案"，不采样），读不出的算错。

**p1_test.jsonl**

| 题类 | 题数 | 随机猜 | 学出来的规则（哪条） | base_7b_blind 不看图 | base_7b_blind 读不出 |
| --- | --- | --- | --- | --- | --- |
| action_between | 897 | 25.0% | 25.5%（learned_options） | 20.1% | 18.3% |
| both_in_view_after | 172 | 25.0% | 30.2%（learned_action） | 18.6% | 13.4% |
| both_in_view_now | 252 | 25.0% | 28.6%（learned_action） | 19.4% | 0.0% |
| closer_after | 160 | 50.0% | 48.1%（learned_options） | 16.2% | 68.8% |
| closer_now | 160 | 50.0% | 52.8%（learned_options） | 53.8% | 0.6% |
| distance_bin_after | 694 | 25.0% | 24.4%（learned_options） | 22.0% | 12.1% |
| distance_bin_now | 361 | 25.0% | 26.6%（first_option） | 13.9% | 35.7% |
| goal_after | 1380 | 50.0% | 52.8%（learned_action） | 50.0% | 0.0% |
| goal_now | 1039 | 50.0% | 52.4%（learned_action） | 52.4% | 0.0% |
| height_bin_after | 230 | 25.0% | 28.3%（learned_options） | 23.5% | 14.3% |
| height_bin_now | 147 | 25.0% | 25.9%（learned_options） | 16.3% | 0.7% |
| left_right_after | 139 | 50.0% | 52.5%（learned_options） | 46.0% | 8.6% |
| left_right_now | 354 | 50.0% | 52.3%（learned_options） | 51.4% | 0.8% |
| 全部 | 5985 | 38.5% |  | 35.7% | 9.4% |

**p1_probe.jsonl**

| 题类 | 题数 | 随机猜 | 学出来的规则（哪条） | base_7b_blind 不看图 | base_7b_blind 读不出 |
| --- | --- | --- | --- | --- | --- |
| action_between | 40 | 25.0% | 25.0%（inverse_frequency） | 15.0% | 37.5% |
| both_in_view_after | 40 | 25.0% | 37.5%（learned_action） | 17.5% | 12.5% |
| both_in_view_now | 40 | 25.0% | 30.6%（learned_options） | 27.5% | 0.0% |
| closer_after | 40 | 50.0% | 47.5%（learned_options） | 15.0% | 62.5% |
| closer_now | 40 | 50.0% | 50.0%（learned_options） | 50.0% | 2.5% |
| distance_bin_after | 40 | 25.0% | 25.0%（first_option） | 27.5% | 15.0% |
| distance_bin_now | 40 | 25.0% | 27.5%（first_option） | 15.0% | 32.5% |
| goal_after | 200 | 50.0% | 49.5%（learned_action） | 50.0% | 0.0% |
| goal_now | 200 | 50.0% | 50.0%（learned_action） | 50.0% | 0.0% |
| height_bin_after | 40 | 25.0% | 30.0%（first_option） | 22.5% | 15.0% |
| height_bin_now | 40 | 25.0% | 25.0%（learned_options） | 15.0% | 2.5% |
| left_right_after | 40 | 50.0% | 60.0%（learned_action） | 37.5% | 10.0% |
| left_right_now | 40 | 50.0% | 50.0%（learned_options） | 50.0% | 0.0% |
| occluded_after | 71 | 33.3% | 44.6%（learned_options） | 39.4% | 1.4% |
| occluded_now | 41 | 33.3% | 46.3%（learned_action） | 14.6% | 0.0% |
| 全部 | 952 | 40.7% |  | 36.9% | 8.1% |

## 8. 每类两道原题

**action_between**（projective_relations，middle，`p1_a2671a893ea2`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera turned right by 40 degrees
(B) the camera turned right by 60 degrees
(C) the camera turned left by 60 degrees
(D) the camera turned left by 40 degrees
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b8d5043fb4db68d8810f1a4b__x1.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b8d5043fb4db68d8810f1a4b__x3.png`

**action_between**（projective_relations，start，`p1_4cacc490461f`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera turned right by 60 degrees
(B) the camera turned left by 60 degrees
(C) the camera turned right by 40 degrees
(D) the camera turned left by 40 degrees
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_6e87c8e766cf0fa996bda6f8__s0.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_6e87c8e766cf0fa996bda6f8__s3.png`

**both_in_view_after**（fov_inclusion，start，`p1_7d65240ef275`）

```
<image>
If the camera turns right by 60 degrees from the current view, will the taller wardrobe and the TV both be inside the view?
(A) only the TV
(B) only the taller wardrobe
(C) neither
(D) both are
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_32dad472ac291aa8257d5ec6__s0.png`

**both_in_view_after**（fov_inclusion，terminal，`p1_cdb953fe1f1e`）

```
<image>
If the camera turns right by 40 degrees from the current view, will the smaller coffee table and the sofa both be inside the view?
(A) only the smaller coffee table
(B) only the sofa
(C) both are
(D) neither
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_5dcd80b77457afa32007da99__s7.png`

**both_in_view_now**（fov_inclusion，terminal，`p1_1516152f7a04`）

```
<image>
In the current view, are the chair and the decorative painting both inside the view?
(A) only the decorative painting
(B) only the chair
(C) both are
(D) neither
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_bec8f0bc2455eb0c816fd0f2__s3.png`

**both_in_view_now**（fov_inclusion，middle，`p1_1f3f81d71bee`）

```
<image>
In the current view, are the stool and the wardrobe both inside the view?
(A) both are
(B) only the wardrobe
(C) neither
(D) only the stool
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_ab58285c6b791594a4ab4153__s5.png`

**closer_after**（equidistance，terminal，`p1_df9ad6ea5b41`）

```
<image>
If the camera moves forward by 0.6 meters from the current view, which will be closer to you (horizontal distance to its nearest side): the stool or the bedside table?
(A) the bedside table
(B) the stool
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_86b67d9b42d8ed55c4d10cce__s4.png`

**closer_after**（equidistance，terminal，`p1_279b5f987ce0`）

```
<image>
If the camera moves sideways to the right by 0.9 meters from the current view, which will be closer to you (horizontal distance to its nearest side): the cabinet or the coffee table?
(A) the coffee table
(B) the cabinet
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_a26074d73bf81aedb19b10e2__s4.png`

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

**distance_bin_after**（absolute_positioning，terminal，`p1_193e691392e1`）

```
<image>
If the camera moves backward by 0.3 meters from the current view, about how far will you be from the water bar, measured horizontally to its nearest side?
(A) about 1.0 meters
(B) about 1.5 meters
(C) about 2.0 meters
(D) about 2.5 meters
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_0b28109a5dfc8a01b5719381__s4.png`

**distance_bin_after**（absolute_positioning，terminal，`p1_dc7027a503f5`）

```
<image>
If the camera moves sideways to the left by 0.3 meters from the current view, about how far will you be from the trash can, measured horizontally to its nearest side?
(A) about 2.5 meters
(B) about 3.0 meters
(C) about 3.5 meters
(D) about 4.0 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_bca944b9b26144b4cae0190c__s1.png`

**distance_bin_now**（absolute_positioning，middle，`p1_518a0680ce56`）

```
<image>
In the current view, about how far are you from the sofa next to the TV, measured horizontally to its nearest side?
(A) about 2.5 meters
(B) about 3.0 meters
(C) about 3.5 meters
(D) about 4.0 meters
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_0a03029bcb22c9a4b3624fe5__s4.png`

**distance_bin_now**（absolute_positioning，terminal，`p1_0f592bc6f62a`）

```
<image>
In the current view, about how far are you from the monitor, measured horizontally to its nearest side?
(A) about 1.0 meters
(B) about 1.5 meters
(C) about 2.0 meters
(D) about 2.5 meters
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_ad44b6e59c5b02755203f85f__s4.png`

**goal_after**（equidistance，terminal，`p1_2252abc0ec96`）

```
<image>
Task: Move to a spot that is equally far from the dishwasher and the electric cooker (measured to their nearest sides), with both of them in view.
If the camera moves sideways to the left by 0.3 meters from the current view, will the task's requirement be met? Answer Yes or No.
答案：Yes
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_482cd9940abb9c4ce2c21e8b__s6.png`

**goal_after**（screen_occupancy，terminal，`p1_aee4cd67abe0`）

```
<image>
Task: Move so that the dining table is fully visible from top to bottom and its height fills about 15% of the image height.
If the camera moves backward by 0.3 meters from the current view, will the task's requirement be met? Answer Yes or No.
答案：Yes
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_24c247c6b11da7019ea9b7ee__x2.png`

**goal_now**（projective_relations，middle，`p1_8e24cbbaee91`）

```
<image>
Task: Move so that the wardrobe appears to the right of the chair next to the projection screen in your view.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_a99120089e90853a93ead160__s1.png`

**goal_now**（screen_occupancy，interior，`p1_167c21ea8664`）

```
<image>
Task: Move so that the easel is fully visible from top to bottom and its height fills about 60% of the image height.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：Yes
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261018/interior/frames/plain/v10_6e6289d7f3552a322928cfba__i0.png`

**height_bin_after**（screen_occupancy，start，`p1_3f204df4378c`）

```
<image>
If the camera moves backward by 0.9 meters from the current view, about what fraction of the image height will the notebook computer fill, from its top to its bottom?
(A) about 10%
(B) about 20%
(C) about 30%
(D) about 40%
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_bdd87e29666b91b98cdcbbd0__s0.png`

**height_bin_after**（screen_occupancy，middle，`p1_17f6be69307a`）

```
<image>
If the camera turns left by 20 degrees from the current view, about what fraction of the image height will the fireplace fill, from its top to its bottom?
(A) about 20%
(B) about 30%
(C) about 40%
(D) about 50%
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_fc66a4c98daf469dccb52d52__s2.png`

**height_bin_now**（screen_occupancy，middle，`p1_93c1b19c2c03`）

```
<image>
In the current view, about what fraction of the image height does the score stand fill, from its top to its bottom?
(A) about 10%
(B) about 20%
(C) about 30%
(D) about 40%
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_a14b34bf99d9df2a08a77b34__s2.png`

**height_bin_now**（screen_occupancy，middle，`p1_71512324c3c5`）

```
<image>
In the current view, about what fraction of the image height does the dining table fill, from its top to its bottom?
(A) about 10%
(B) about 20%
(C) about 30%
(D) about 40%
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_06d0a965230ba8efe4598dcf__s3.png`

**left_right_after**（projective_relations，terminal，`p1_17719b5a9952`）

```
<image>
If the camera moves backward by 0.3 meters from the current view, will the flowers be to the left or to the right of the smaller side table?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_10184bdf7e64e7c7c9a7bc85__s7.png`

**left_right_after**（projective_relations，terminal，`p1_deaac429e1da`）

```
<image>
If the camera moves forward by 0.6 meters from the current view, will the water dispenser be to the left or to the right of the storage rack next to the sofa?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_040ff2a68a74ea575851d148__s2.png`

**left_right_now**（projective_relations，terminal，`p1_23a982333325`）

```
<image>
In the current view, is the plant to the left or to the right of the side table?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_7c49668c7a4a38748e60d0a5__s5.png`

**left_right_now**（projective_relations，terminal，`p1_3f95c904df83`）

```
<image>
In the current view, is the sofa to the left or to the right of the side table?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_9343f438cf57805a21887d88__x1.png`

**occluded_after**（occlusion_alignment，terminal，`p1_e622f0e0f405`）

```
<image>
If the camera moves sideways to the left by 0.9 meters from the current view, will the stool next to the floor lamp be hidden behind the sofa?
(A) not hidden behind the sofa
(B) partly hidden behind the sofa
(C) fully hidden behind the sofa
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_b70bfa335843e19ee291552b__s4.png`

**occluded_after**（occlusion_alignment，middle，`p1_e54734cba5c2`）

```
<image>
If the camera moves forward by 0.3 meters from the current view, will the bedside table be hidden behind the wardrobe?
(A) partly hidden behind the wardrobe
(B) not hidden behind the wardrobe
(C) fully hidden behind the wardrobe
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_25a608876df74330d293d746__x1.png`

**occluded_now**（occlusion_alignment，start，`p1_ad900ec668c8`）

```
<image>
In the current view, is the camera next to the wardrobe hidden behind the microphone?
(A) partly hidden behind the microphone
(B) not hidden behind the microphone
(C) fully hidden behind the microphone
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_436b17f6849b8340193dd5c5__s0.png`

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

