**与 ours-RL-r11 逐题比**（只有它做对 / 只有 ours-RL-r11 做对；p 为精确 McNemar 双侧检验）

| 运行 | 成功 / 题数 | 差（点） | 只有它对 | 只有参照对 | p |
|---|---|---|---|---|---|
| ours-RL-r11__shuffle | 109 / 279 | -2.9 | 13 | 21 | 0.2295 |
| ours-RL-r11__drop | 68 / 279 | -17.6 | 12 | 61 | <0.0001 |
| ours-RL-r11__replace | 88 / 279 | -10.4 | 17 | 46 | 0.0003 |

**按任务类型的成功数 / 题数**

| 运行 | absolute_positioning | equidistance | fov_inclusion | occlusion_alignment | projective_relations | screen_occupancy |
|---|---|---|---|---|---|---|
| ours-RL-r11__shuffle | 27/50 | 10/50 | 20/50 | 5/29 | 21/50 | 26/50 |
| ours-RL-r11__drop | 22/50 | 6/50 | 10/50 | 2/29 | 13/50 | 15/50 |
| ours-RL-r11__replace | 24/50 | 7/50 | 17/50 | 4/29 | 15/50 | 21/50 |
