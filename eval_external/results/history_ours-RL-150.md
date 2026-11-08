**与 ours-RL-150 逐题比**（只有它做对 / 只有 ours-RL-150 做对；p 为精确 McNemar 双侧检验）

| 运行 | 成功 / 题数 | 差（点） | 只有它对 | 只有参照对 | p |
|---|---|---|---|---|---|
| ours-RL-150__shuffle | 116 / 279 | -2.9 | 7 | 15 | 0.1338 |
| ours-RL-150__drop | 97 / 279 | -9.7 | 12 | 39 | 0.0002 |
| ours-RL-150__replace | 105 / 279 | -6.8 | 20 | 39 | 0.0183 |

**按任务类型的成功数 / 题数**

| 运行 | absolute_positioning | equidistance | fov_inclusion | occlusion_alignment | projective_relations | screen_occupancy |
|---|---|---|---|---|---|---|
| ours-RL-150__shuffle | 31/50 | 11/50 | 19/50 | 5/29 | 18/50 | 32/50 |
| ours-RL-150__drop | 23/50 | 11/50 | 17/50 | 3/29 | 17/50 | 26/50 |
| ours-RL-150__replace | 24/50 | 10/50 | 21/50 | 7/29 | 20/50 | 23/50 |
