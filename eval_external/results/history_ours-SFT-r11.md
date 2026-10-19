**与 ours-SFT-r11 逐题比**（只有它做对 / 只有 ours-SFT-r11 做对；p 为精确 McNemar 双侧检验）

| 运行 | 成功 / 题数 | 差（点） | 只有它对 | 只有参照对 | p |
|---|---|---|---|---|---|
| ours-SFT-r11__shuffle | 102 / 279 | -1.8 | 15 | 20 | 0.4996 |
| ours-SFT-r11__drop | 84 / 279 | -8.2 | 24 | 47 | 0.0086 |
| ours-SFT-r11__replace | 88 / 279 | -6.8 | 25 | 44 | 0.0295 |

**按任务类型的成功数 / 题数**

| 运行 | absolute_positioning | equidistance | fov_inclusion | occlusion_alignment | projective_relations | screen_occupancy |
|---|---|---|---|---|---|---|
| ours-SFT-r11__shuffle | 25/50 | 8/50 | 17/50 | 10/29 | 15/50 | 27/50 |
| ours-SFT-r11__drop | 23/50 | 7/50 | 14/50 | 3/29 | 11/50 | 26/50 |
| ours-SFT-r11__replace | 26/50 | 5/50 | 16/50 | 3/29 | 15/50 | 23/50 |
