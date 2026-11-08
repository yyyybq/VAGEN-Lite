**与 ours-SFT-r11 逐题比**（只有它做对 / 只有 ours-SFT-r11 做对；p 为精确 McNemar 双侧检验）

| 运行 | 成功 / 题数 | 差（点） | 只有它对 | 只有参照对 | p |
|---|---|---|---|---|---|
| ours-RL-r11 | 117 / 279 | +3.6 | 35 | 25 | 0.2451 |
| ours-RL-r11-dense | 128 / 279 | +7.5 | 41 | 20 | 0.0099 |

**按任务类型的成功数 / 题数**

| 运行 | absolute_positioning | equidistance | fov_inclusion | occlusion_alignment | projective_relations | screen_occupancy |
|---|---|---|---|---|---|---|
| ours-RL-r11 | 30/50 | 14/50 | 21/50 | 5/29 | 19/50 | 28/50 |
| ours-RL-r11-dense | 31/50 | 12/50 | 22/50 | 5/29 | 21/50 | 37/50 |
