**与 ours-RL-r11-dense 逐题比**（只有它做对 / 只有 ours-RL-r11-dense 做对；p 为精确 McNemar 双侧检验）

| 运行 | 成功 / 题数 | 差（点） | 只有它对 | 只有参照对 | p |
|---|---|---|---|---|---|
| ours-RL-r11-dense__shuffle | 122 / 279 | -2.2 | 17 | 23 | 0.4296 |
| ours-RL-r11-dense__drop | 82 / 279 | -16.5 | 16 | 62 | <0.0001 |
| ours-RL-r11-dense__replace | 100 / 279 | -10.0 | 16 | 44 | 0.0004 |

**按任务类型的成功数 / 题数**

| 运行 | absolute_positioning | equidistance | fov_inclusion | occlusion_alignment | projective_relations | screen_occupancy |
|---|---|---|---|---|---|---|
| ours-RL-r11-dense__shuffle | 34/50 | 11/50 | 18/50 | 6/29 | 19/50 | 34/50 |
| ours-RL-r11-dense__drop | 19/50 | 8/50 | 16/50 | 5/29 | 15/50 | 19/50 |
| ours-RL-r11-dense__replace | 28/50 | 10/50 | 21/50 | 2/29 | 19/50 | 20/50 |
