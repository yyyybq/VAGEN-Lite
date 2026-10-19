**与 ours-SFT 逐题比**（只有它做对 / 只有 ours-SFT 做对；p 为精确 McNemar 双侧检验）

| 运行 | 成功 / 题数 | 差（点） | 只有它对 | 只有参照对 | p |
|---|---|---|---|---|---|
| ours-SFT__shuffle | 116 / 279 | +1.4 | 13 | 9 | 0.5235 |
| ours-SFT__drop | 95 / 279 | -6.1 | 13 | 30 | 0.0137 |
| ours-SFT__replace | 89 / 279 | -8.2 | 22 | 45 | 0.0067 |

**按任务类型的成功数 / 题数**

| 运行 | absolute_positioning | equidistance | fov_inclusion | occlusion_alignment | projective_relations | screen_occupancy |
|---|---|---|---|---|---|---|
| ours-SFT__shuffle | 33/50 | 11/50 | 19/50 | 8/29 | 15/50 | 30/50 |
| ours-SFT__drop | 26/50 | 7/50 | 17/50 | 2/29 | 19/50 | 24/50 |
| ours-SFT__replace | 26/50 | 5/50 | 19/50 | 1/29 | 17/50 | 21/50 |
