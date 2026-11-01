# P1 转移题：统计

由 `python -m data_gen.transition_qa.report` 从 `exps/transition_qa_20261019b` 生成。题库怎么来的、各题类的答案用哪个函数算，见 `PROGRESS.md` 第 1.3 节和第 2 节。

## 1. 题数和三类题的比例

| 文件 | 题数 | 状态题 (a) | 前向题 (b) | 逆向题 (c) |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | 78786 | 32498（41.2%） | 38420（48.8%） | 7868（10.0%） |
| p1_test.jsonl | 5313 | 2101（39.5%） | 2404（45.2%） | 808（15.2%） |
| p1_probe.jsonl | 838 | 391（46.7%） | 413（49.3%） | 34（4.1%） |

三类题的规则（用户 2026-10-19）：状态题全部保留（答案平衡之后）；前向题最多是状态题的 1.2 倍；逆向题最多占题库的 15%；"动作后任务要求会满足吗"（goal_after）最多占前向题的一半；单个场景最多 8%。超出的按（题类、答案、是否变化）分层抽。

- train 场景：前向题 (b)可用 54398、留下 43106；逆向题 (c)可用 8409、留下 8409；状态题 (a)可用 35922、留下 35922
  - 上限：前向题 43106 道，逆向题 13946 道
- val 场景：前向题 (b)可用 6697、留下 2775；逆向题 (c)可用 1015、留下 897；状态题 (a)可用 2313、留下 2313
  - 上限：前向题 2775 道，逆向题 897 道
- 测试集去掉了探针集的 838 道题（两份不重叠），以及人工核对从探针去掉的 35 道。
- 场景排除（scene_exclusions_20261029.json，用户 2026-10-29）：128 个场景的题在抽样、配平之后去掉（p1_train.jsonl 6726 道、p1_test.jsonl 636 道，暂扣之前），其余题逐题不变。
- p1_train.jsonl 暂扣 1925 道（occlusion_alignment/goal_after 702、occlusion_alignment/goal_now 680、occlusion_alignment/occluded_after 332、occlusion_alignment/occluded_now 211），在 p1_train_held.jsonl：答案来自遮挡的框判据（遮挡题和遮挡任务的 goal 题），用户核对探针时发现这个判据的标签大多不对（2026-10-23、10-27）。
- p1_test.jsonl 暂扣 36 道（occlusion_alignment/goal_after 34、occlusion_alignment/goal_now 2），在 p1_test_held.jsonl：答案来自遮挡的框判据（遮挡题和遮挡任务的 goal 题），用户核对探针时发现这个判据的标签大多不对（2026-10-23、10-27）。
- 探针集：838 道，每个题类 40 道（去掉排除场景后 30–40 道），其中 goal_after 200 道（人工核对、去掉排除场景后留 171 道）、goal_now 200 道（人工核对、去掉排除场景后留 176 道）、occluded_after 全部可用的 71 道（人工核对、去掉排除场景后留 58 道）、occluded_now 全部可用的 41 道（人工核对、去掉排除场景后留 33 道）（预注册要的题类），来自 88 个 val 场景；其余题类都够；occluded_after 另有 37 道"没变化"的超过 10% 上限，没取。
- 探针集的人工核对（probe_label_overrides.json）：40 道按核对的答案重标（full → partial 4、no → yes 2、partial → none 29、yes → no 5），35 道不该出的题去掉；2026-10-23 核对答“完全 / 部分挡住”的 78 道遮挡题：对 27、错 51，重标 33、去掉 18；2026-10-28 核对遮挡任务的 44 道 goal 题：对 20、错 23、有歧义 1，重标 7、去掉 17。
- 探针集去掉了排除场景的 79 道（action_between 6、both_in_view_after 1、both_in_view_now 3、closer_after 2、closer_now 2、distance_bin_after 3、distance_bin_now 5、goal_after 18、goal_now 18、height_bin_after 10、height_bin_now 5、left_right_now 3、occluded_after 2、occluded_now 1），在人工核对之后。

## 2. 按任务族 × 题类 × 划分的题数

**p1_train.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 1189 | 0 | 0 | 0 | 0 | 8763 | 2446 | 6277 | 6253 | 0 | 0 | 0 | 0 | 0 | 0 | 24928 |
| equidistance | 1428 | 0 | 0 | 2385 | 1388 | 0 | 0 | 1624 | 2465 | 0 | 0 | 0 | 0 | 0 | 0 | 9290 |
| fov_inclusion | 1194 | 3184 | 2347 | 0 | 0 | 0 | 0 | 1484 | 2843 | 0 | 0 | 0 | 0 | 0 | 0 | 11052 |
| occlusion_alignment | 223 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 223 |
| projective_relations | 2660 | 0 | 0 | 0 | 0 | 0 | 0 | 4356 | 5416 | 0 | 0 | 2509 | 3339 | 0 | 0 | 18280 |
| screen_occupancy | 1174 | 0 | 0 | 0 | 0 | 0 | 0 | 4451 | 5200 | 3387 | 801 | 0 | 0 | 0 | 0 | 15013 |
| 合计 | 7868 | 3184 | 2347 | 2385 | 1388 | 8763 | 2446 | 18192 | 22177 | 3387 | 801 | 2509 | 3339 | 0 | 0 | 78786 |

**p1_test.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 123 | 0 | 0 | 0 | 0 | 587 | 319 | 429 | 334 | 0 | 0 | 0 | 0 | 0 | 0 | 1792 |
| equidistance | 139 | 0 | 0 | 154 | 154 | 0 | 0 | 78 | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 579 |
| fov_inclusion | 111 | 162 | 239 | 0 | 0 | 0 | 0 | 85 | 93 | 0 | 0 | 0 | 0 | 0 | 0 | 690 |
| occlusion_alignment | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 |
| projective_relations | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 309 | 212 | 0 | 0 | 130 | 339 | 0 | 0 | 1265 |
| screen_occupancy | 132 | 0 | 0 | 0 | 0 | 0 | 0 | 281 | 239 | 189 | 118 | 0 | 0 | 0 | 0 | 959 |
| 合计 | 808 | 162 | 239 | 154 | 154 | 587 | 319 | 1182 | 932 | 189 | 118 | 130 | 339 | 0 | 0 | 5313 |

**p1_probe.jsonl**

| 任务族 | action_between | both_in_view_after | both_in_view_now | closer_after | closer_now | distance_bin_after | distance_bin_now | goal_after | goal_now | height_bin_after | height_bin_now | left_right_after | left_right_now | occluded_after | occluded_now | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute_positioning | 21 | 0 | 0 | 0 | 0 | 37 | 35 | 33 | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 158 |
| equidistance | 9 | 0 | 0 | 38 | 38 | 0 | 0 | 33 | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 152 |
| fov_inclusion | 1 | 39 | 37 | 0 | 0 | 0 | 0 | 33 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 143 |
| occlusion_alignment | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 13 | 0 | 0 | 0 | 0 | 58 | 33 | 116 |
| projective_relations | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 36 | 0 | 0 | 40 | 37 | 0 | 0 | 148 |
| screen_occupancy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 28 | 30 | 35 | 0 | 0 | 0 | 0 | 121 |
| 合计 | 34 | 39 | 37 | 38 | 38 | 37 | 35 | 171 | 176 | 30 | 35 | 40 | 37 | 58 | 33 | 838 |

## 3. 答案分布

"字母"是正确选项的字母；"偏差"是各字母占比和均匀分布之差的最大值（验收线 5 个百分点）。"答案值"是选项的含义（例如 left / right、full / partial / none）。

**p1_train.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 7868 | A 24.8% / B 25.0% / C 25.0% / D 25.1% | 0.2 | turn_leftx2 682，turn_rightx2 665，move_leftx2 485，turn_leftx1 476，turn_rightx1 475，move_rightx2 452 |
| both_in_view_after | 3184 | A 25.2% / B 24.9% / C 25.0% / D 25.0% | 0.2 | only_b 924，only_a 901，neither 808，both 551 |
| both_in_view_now | 2347 | A 25.2% / B 25.0% / C 24.9% / D 24.9% | 0.2 | both 732，only_b 642，only_a 602，neither 371 |
| closer_after | 2385 | A 49.9% / B 50.1% | 0.1 | b 1214，a 1171 |
| closer_now | 1388 | A 50.2% / B 49.8% | 0.2 | b 698，a 690 |
| distance_bin_after | 8763 | A 25.3% / B 25.5% / C 24.9% / D 24.3% | 0.7 | 2.0 1903，2.5 1754，3.0 1389，1.5 1360，3.5 792，1.0 699 |
| distance_bin_now | 2446 | A 25.8% / B 24.5% / C 25.9% / D 23.8% | 1.2 | 2.5 615，2.0 493，1.5 442，3.0 432，3.5 165，1.0 142 |
| goal_after | 18192 | Yes 49.1% / No 50.9% | 0.9 | no 9256，yes 8936 |
| goal_now | 22177 | Yes 47.3% / No 52.7% | 2.7 | no 11695，yes 10482 |
| height_bin_after | 3387 | A 26.0% / B 26.2% / C 24.4% / D 23.4% | 1.6 | 0.2 816，0.3 809，0.4 687，0.5 442，0.1 298，0.8 92 |
| height_bin_now | 801 | A 24.1% / B 27.0% / C 26.1% / D 22.8% | 2.2 | 0.3 201，0.2 198，0.4 187，0.5 86，0.1 77，0.7 14 |
| left_right_after | 2509 | A 51.7% / B 48.3% | 1.7 | left 1297，right 1212 |
| left_right_now | 3339 | A 52.4% / B 47.6% | 2.4 | left 1749，right 1590 |

**p1_test.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 808 | A 25.0% / B 25.1% / C 24.6% / D 25.2% | 0.4 | turn_leftx2 68，turn_rightx2 66，move_backwardx2 52，turn_rightx1 50，move_rightx2 49，move_forwardx2 45 |
| both_in_view_after | 162 | A 25.9% / B 22.8% / C 26.5% / D 24.7% | 2.2 | only_b 48，neither 45，only_a 42，both 27 |
| both_in_view_now | 239 | A 25.1% / B 24.7% / C 24.7% / D 25.5% | 0.5 | only_b 70，both 68，only_a 66，neither 35 |
| closer_after | 154 | A 50.0% / B 50.0% | 0.0 | b 78，a 76 |
| closer_now | 154 | A 50.0% / B 50.0% | 0.0 | b 81，a 73 |
| distance_bin_after | 587 | A 25.9% / B 25.6% / C 25.2% / D 23.3% | 1.7 | 2.5 120，2.0 110，3.0 93，1.5 79，3.5 54，1.0 50 |
| distance_bin_now | 319 | A 26.3% / B 23.5% / C 25.4% / D 24.8% | 1.5 | 2.5 82，3.0 66，1.5 56，2.0 51，4.0 25，3.5 23 |
| goal_after | 1182 | Yes 49.8% / No 50.2% | 0.2 | no 593，yes 589 |
| goal_now | 932 | Yes 47.4% / No 52.6% | 2.6 | no 490，yes 442 |
| height_bin_after | 189 | A 25.4% / B 27.0% / C 23.3% / D 24.3% | 2.0 | 0.2 45，0.3 40，0.4 31，0.1 25，0.5 22，0.8 12 |
| height_bin_now | 118 | A 26.3% / B 24.6% / C 23.7% / D 25.4% | 1.3 | 0.2 26，0.4 25，0.3 24，0.1 21，0.5 9，0.6 4 |
| left_right_after | 130 | A 51.5% / B 48.5% | 1.5 | left 67，right 63 |
| left_right_now | 339 | A 52.2% / B 47.8% | 2.2 | left 177，right 162 |

**p1_probe.jsonl**

| 题类 | 题数 | 字母 | 偏差（百分点） | 答案值（前 6） |
| --- | --- | --- | --- | --- |
| action_between | 34 | A 17.6% / B 29.4% / C 26.5% / D 26.5% | 7.4 | move_leftx5 2，move_rightx5 2，turn_leftx4 2，turn_rightx4 2，move_leftx4 2，move_rightx3 2 |
| both_in_view_after | 39 | A 25.6% / B 23.1% / C 25.6% / D 25.6% | 1.9 | only_b 10，neither 10，only_a 10，both 9 |
| both_in_view_now | 37 | A 24.3% / B 27.0% / C 27.0% / D 21.6% | 3.4 | only_b 10，both 9，only_a 9，neither 9 |
| closer_after | 38 | A 47.4% / B 52.6% | 2.6 | b 19，a 19 |
| closer_now | 38 | A 50.0% / B 50.0% | 0.0 | a 20，b 18 |
| distance_bin_after | 37 | A 24.3% / B 24.3% / C 24.3% / D 27.0% | 2.0 | 2.0 8，2.5 7，3.5 6，3.0 4，1.0 3，1.5 3 |
| distance_bin_now | 35 | A 22.9% / B 25.7% / C 25.7% / D 25.7% | 2.1 | 2.5 9，2.0 8，1.5 7，3.0 4，4.0 3，1.0 2 |
| goal_after | 171 | Yes 50.3% / No 49.7% | 0.3 | yes 86，no 85 |
| goal_now | 176 | Yes 49.4% / No 50.6% | 0.6 | no 89，yes 87 |
| height_bin_after | 30 | A 30.0% / B 23.3% / C 26.7% / D 20.0% | 5.0 | 0.3 8，0.1 5，0.2 5，0.4 5，0.6 3，0.5 2 |
| height_bin_now | 35 | A 28.6% / B 25.7% / C 17.1% / D 28.6% | 7.9 | 0.2 9，0.4 7，0.3 7，0.5 5，0.1 4，0.8 2 |
| left_right_after | 40 | A 50.0% / B 50.0% | 0.0 | right 20，left 20 |
| left_right_now | 37 | A 45.9% / B 54.1% | 4.1 | right 20，left 17 |
| occluded_after | 58 | A 37.9% / B 25.9% / C 36.2% | 7.5 | none 41，full 12，partial 5 |
| occluded_now | 33 | A 21.2% / B 36.4% / C 42.4% | 12.1 | none 19，full 10，partial 4 |

训练集和测试集里字母偏差的最大值：2.7 个百分点（探针集每类只有约 40 道，不计入；它的偏差见上表）。

## 4. 前向题：没变化的题、答案和现在不同的题

"没变化"：被问的量在动作前后的变化小于门槛（例如转身后到物体的距离不变）；上限 10%。"答案变了"：动作后的答案和现在的答案不同；控制在 35%–65%，防止模型学成"把现在的答案反过来"。

| 文件 | 题类 | 题数 | 没变化 | 答案变了 |
| --- | --- | --- | --- | --- |
| p1_train.jsonl | both_in_view_after | 3184 | 10.0% | 65.2% |
| p1_train.jsonl | closer_after | 2385 | 9.9% | 45.5% |
| p1_train.jsonl | distance_bin_after | 8763 | 10.1% | 64.8% |
| p1_train.jsonl | goal_after | 18192 | 10.3% | 53.2% |
| p1_train.jsonl | height_bin_after | 3387 | 10.4% | 64.0% |
| p1_train.jsonl | left_right_after | 2509 | 9.8% | 0.0% |
| p1_test.jsonl | both_in_view_after | 162 | 11.1% | 67.3% |
| p1_test.jsonl | closer_after | 154 | 10.4% | 46.7% |
| p1_test.jsonl | distance_bin_after | 587 | 10.4% | 61.9% |
| p1_test.jsonl | goal_after | 1182 | 10.1% | 54.1% |
| p1_test.jsonl | height_bin_after | 189 | 9.0% | 66.2% |
| p1_test.jsonl | left_right_after | 130 | 9.2% | 0.0% |
| p1_probe.jsonl | both_in_view_after | 39 | 10.3% | 50.0% |
| p1_probe.jsonl | closer_after | 38 | 5.3% | 51.9% |
| p1_probe.jsonl | distance_bin_after | 37 | 10.8% | 42.9% |
| p1_probe.jsonl | goal_after | 171 | 11.1% | 50.9% |
| p1_probe.jsonl | height_bin_after | 30 | 13.3% | 40.0% |
| p1_probe.jsonl | left_right_after | 40 | 10.0% | 0.0% |
| p1_probe.jsonl | occluded_after | 58 | 10.3% | 54.3% |

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
| action_between | 808 | 25.0% | 25.4%（learned_options） | 20.1% | 18.3% |
| both_in_view_after | 162 | 25.0% | 25.6%（learned_options） | 18.6% | 13.4% |
| both_in_view_now | 239 | 25.0% | 28.5%（learned_action） | 19.4% | 0.0% |
| closer_after | 154 | 50.0% | 48.1%（learned_options） | 16.2% | 68.8% |
| closer_now | 154 | 50.0% | 52.9%（learned_options） | 53.8% | 0.6% |
| distance_bin_after | 587 | 25.0% | 24.2%（learned_options） | 22.0% | 12.1% |
| distance_bin_now | 319 | 25.0% | 27.3%（first_option） | 13.9% | 35.7% |
| goal_after | 1182 | 50.0% | 55.0%（learned_action） | 50.0% | 0.0% |
| goal_now | 932 | 50.0% | 52.6%（learned_action） | 52.4% | 0.0% |
| height_bin_after | 189 | 25.0% | 29.0%（learned_options） | 23.5% | 14.3% |
| height_bin_now | 118 | 25.0% | 24.2%（learned_options） | 16.3% | 0.7% |
| left_right_after | 130 | 50.0% | 51.5%（learned_options） | 46.0% | 8.6% |
| left_right_now | 339 | 50.0% | 52.2%（learned_options） | 51.4% | 0.8% |
| 全部 | 5313 | 38.6% |  | 35.7% | 9.4% |

**p1_probe.jsonl**

| 题类 | 题数 | 随机猜 | 学出来的规则（哪条） | base_7b_blind 不看图 | base_7b_blind 读不出 |
| --- | --- | --- | --- | --- | --- |
| action_between | 34 | 25.0% | 25.0%（inverse_frequency） | 11.8% | 38.2% |
| both_in_view_after | 39 | 25.0% | 32.1%（learned_options） | 17.9% | 12.8% |
| both_in_view_now | 37 | 25.0% | 31.1%（learned_options） | 24.3% | 0.0% |
| closer_after | 38 | 50.0% | 47.4%（learned_options） | 15.8% | 63.2% |
| closer_now | 38 | 50.0% | 50.0%（learned_options） | 50.0% | 2.6% |
| distance_bin_after | 37 | 25.0% | 24.3%（first_option） | 27.0% | 13.5% |
| distance_bin_now | 35 | 25.0% | 25.7%（first_option） | 14.3% | 31.4% |
| goal_after | 171 | 50.0% | 49.7%（learned_action） | 49.7% | 0.0% |
| goal_now | 176 | 50.0% | 49.1%（learned_action） | 50.6% | 0.0% |
| height_bin_after | 30 | 25.0% | 30.0%（first_option） | 26.7% | 16.7% |
| height_bin_now | 35 | 25.0% | 25.7%（learned_options） | 17.1% | 2.9% |
| left_right_after | 40 | 50.0% | 60.0%（learned_action） | 37.5% | 10.0% |
| left_right_now | 37 | 50.0% | 45.9%（learned_options） | 48.6% | 0.0% |
| occluded_after | 58 | 33.3% | 42.2%（option_order） | 70.7% | 1.7% |
| occluded_now | 33 | 33.3% | 33.3%（learned_options） | 54.5% | 0.0% |
| 全部 | 838 | 40.8% |  | 40.6% | 8.4% |

## 8. 每类两道原题

**action_between**（screen_occupancy，start，`p1_578eb8debff3`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera moved sideways to the right by 0.6 meters
(B) the camera turned right by 20 degrees
(C) the camera turned right by 40 degrees
(D) the camera moved sideways to the right by 0.3 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_597822da44dffdcb581bbbb7__s0.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_597822da44dffdcb581bbbb7__s1.png`

**action_between**（equidistance，middle，`p1_fc8d2a48cfee`）

```
First image: <image>
Second image: <image>
The first image was taken right before a camera action and the second image right after it. What did the camera do?
(A) the camera turned left by 40 degrees
(B) the camera turned left by 60 degrees
(C) the camera moved sideways to the left by 0.9 meters
(D) the camera moved sideways to the left by 0.6 meters
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_f3aa775150b6eba81cca4f55__s4.png`；`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_f3aa775150b6eba81cca4f55__s7.png`

**both_in_view_after**（fov_inclusion，start，`p1_314ac70dc13b`）

```
<image>
If the camera turns left by 60 degrees from the current view, will the stool and the armchair both be inside the view?
(A) only the armchair
(B) neither
(C) only the stool
(D) both are
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_0f1e095e9d0de14129e90c5a__s0.png`

**both_in_view_after**（fov_inclusion，middle，`p1_83a1e7d4a7ba`）

```
<image>
If the camera turns left by 20 degrees from the current view, will the pet cage and the sofa both be inside the view?
(A) neither
(B) both are
(C) only the sofa
(D) only the pet cage
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_70f7323184a5a50628e20d61__s1.png`

**both_in_view_now**（fov_inclusion，start，`p1_15f28e344a2b`）

```
<image>
In the current view, are the plant next to the dining table and the plant next to the sofa both inside the view?
(A) only the plant next to the dining table
(B) neither
(C) only the plant next to the sofa
(D) both are
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_71a14ed8c404ef5e93adfee4__s0.png`

**both_in_view_now**（fov_inclusion，terminal，`p1_2087c3deacc4`）

```
<image>
In the current view, are the standing air conditioner and the coffee table both inside the view?
(A) only the coffee table
(B) neither
(C) only the standing air conditioner
(D) both are
Answer with the option's letter.
答案：D
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_028850891e0456e6791c5c61__s1.png`

**closer_after**（equidistance，middle，`p1_ebcd33b12bd7`）

```
<image>
If the camera moves forward by 0.3 meters from the current view, which will be closer to you (horizontal distance to its nearest side): the sofa next to the dining table or the TV?
(A) the sofa next to the dining table
(B) the TV
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_cf0398c0b611b6897d6e3adc__s4.png`

**closer_after**（equidistance，terminal，`p1_294e3367e61f`）

```
<image>
If the camera moves sideways to the left by 0.6 meters from the current view, which will be closer to you (horizontal distance to its nearest side): the dining table or the side table?
(A) the dining table
(B) the side table
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_ef02342a33a27898e0083605__s7.png`

**closer_now**（equidistance，terminal，`p1_87a287faf049`）

```
<image>
In the current view, which is closer to you (horizontal distance to its nearest side): the TV cabinet or the desk?
(A) the desk
(B) the TV cabinet
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_a18145e37f385a9c78c70840__s7.png`

**closer_now**（equidistance，middle，`p1_dd9c7442609b`）

```
<image>
In the current view, which is closer to you (horizontal distance to its nearest side): the wardrobe or the toilet?
(A) the toilet
(B) the wardrobe
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_7ce5585867d52bbd807d82ed__s7.png`

**distance_bin_after**（absolute_positioning，terminal，`p1_1beabbf58d61`）

```
<image>
If the camera moves forward by 0.6 meters from the current view, about how far will you be from the floor lamp, measured horizontally to its nearest side?
(A) about 1.5 meters
(B) about 2.0 meters
(C) about 2.5 meters
(D) about 3.0 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_4a40101ec2340f8a6b03c66b__s1.png`

**distance_bin_after**（absolute_positioning，middle，`p1_f30e5fc734d8`）

```
<image>
If the camera moves backward by 0.9 meters from the current view, about how far will you be from the bed, measured horizontally to its nearest side?
(A) about 2.5 meters
(B) about 3.0 meters
(C) about 3.5 meters
(D) about 4.0 meters
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_2079d09871289a3c1dfa3163__s1.png`

**distance_bin_now**（absolute_positioning，start，`p1_5844810e7993`）

```
<image>
In the current view, about how far are you from the wardrobe next to the coffee table, measured horizontally to its nearest side?
(A) about 1.5 meters
(B) about 2.0 meters
(C) about 2.5 meters
(D) about 3.0 meters
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_00894e78eebf730861805d54__s0.png`

**distance_bin_now**（absolute_positioning，middle，`p1_10a2a9af8807`）

```
<image>
In the current view, about how far are you from the sofa, measured horizontally to its nearest side?
(A) about 1.0 meters
(B) about 1.5 meters
(C) about 2.0 meters
(D) about 2.5 meters
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_d3c2956bb77391201f747dbb__s5.png`

**goal_after**（absolute_positioning，start，`p1_2726e1d6172d`）

```
<image>
Task: Move to a spot about 2.5 meters away from the floor lamp (measured to its nearest side), keeping the floor lamp in view.
If the camera moves backward by 0.6 meters from the current view, will the task's requirement be met? Answer Yes or No.
答案：Yes
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_079b6118f235ec0822fb1f72__s0.png`

**goal_after**（screen_occupancy，terminal，`p1_c7688f64c137`）

```
<image>
Task: Move so that the sofa next to the luminous character is fully visible from top to bottom and its height fills about 40% of the image height.
If the camera turns left by 60 degrees from the current view, will the task's requirement be met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_41d28bbaa231b2761fc96b4d__s2.png`

**goal_now**（projective_relations，middle，`p1_9ef8ac4dd1c6`）

```
<image>
Task: Move so that the water dispenser appears to the left of the floor lamp in your view.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：No
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_41b65bca1acbd50eb6f62805__s2.png`

**goal_now**（absolute_positioning，interior，`p1_193399c00769`）

```
<image>
Task: Move to a spot about 2.5 meters away from the storage rack next to the bar counter (measured to its nearest side), keeping the storage rack next to the bar counter in view.
In the current view, is the task's requirement already met? Answer Yes or No.
答案：Yes
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261018/interior/frames/plain/v10_bd747ba3733a58fb85abb393__i0.png`

**height_bin_after**（screen_occupancy，start，`p1_4a53135b058b`）

```
<image>
If the camera moves sideways to the right by 0.6 meters from the current view, about what fraction of the image height will the reception fill, from its top to its bottom?
(A) about 20%
(B) about 30%
(C) about 40%
(D) about 50%
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_7795eae665b67384b5f7c96b__s0.png`

**height_bin_after**（screen_occupancy，middle，`p1_1abe1e9dc6bc`）

```
<image>
If the camera moves backward by 0.9 meters from the current view, about what fraction of the image height will the display stand fill, from its top to its bottom?
(A) about 10%
(B) about 20%
(C) about 30%
(D) about 40%
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_5336751ac4b78b4bc11da75c__s2.png`

**height_bin_now**（screen_occupancy，terminal，`p1_a8feeca8f477`）

```
<image>
In the current view, about what fraction of the image height does the range hood fill, from its top to its bottom?
(A) about 20%
(B) about 30%
(C) about 40%
(D) about 50%
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_2b008e8edfe6185eb678ef75__s4.png`

**height_bin_now**（screen_occupancy，middle，`p1_7f290bcc603c`）

```
<image>
In the current view, about what fraction of the image height does the fireplace fill, from its top to its bottom?
(A) about 20%
(B) about 30%
(C) about 40%
(D) about 50%
Answer with the option's letter.
答案：C
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_fc66a4c98daf469dccb52d52__s1.png`

**left_right_after**（projective_relations，terminal，`p1_18f7c93f1823`）

```
<image>
If the camera turns right by 20 degrees from the current view, will the wardrobe be to the left or to the right of the wrap?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：A
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_456e963b7e6460a794825400__s2.png`

**left_right_after**（projective_relations，middle，`p1_ea862730e991`）

```
<image>
If the camera turns left by 20 degrees from the current view, will the computer be to the left or to the right of the sofa?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_8ee84124782077c997869e9e__s4.png`

**left_right_now**（projective_relations，terminal，`p1_24fb2fc978ab`）

```
<image>
In the current view, is the washbasin to the left or to the right of the pan?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_dd91a2186d72229753ccc400__x3.png`

**left_right_now**（projective_relations，middle，`p1_42c47d2131a6`）

```
<image>
In the current view, is the chair next to the TV cabinet to the left or to the right of the plant next to the sofa?
(A) to the left
(B) to the right
Answer with the option's letter.
答案：B
```
图：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain/v10_1bcabc11d962c144cd452efb__x1.png`

