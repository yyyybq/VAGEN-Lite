# 矩阵第三版：D 取和 C 相同的更新次数（探索性，不在预登记里）

Matrix v3, exploratory (not in the pre-registration): D - C with D taken at the same RL update as each C RL's
checkpoint of 3.1 (m3cs1s0 50, m3cs1d 0 = the SFT picks, m3cs2s0 100, m3cs2d 50), so the difference does not
include "how many updates ran"; and the test success of every evaluated checkpoint by update (0 = the SFT start).
Same statistics as 3.1 (m3_main31.py): pairs by seed and side, task by task; sign test pooled; 2,000 bootstrap draws.
Read only; writes OUT/matched.{json,md}.

| 配对 | 更新次数 | D − C |
| --- | --- | --- |
| 种子 1 S0 | 50 | -3.8 |
| 种子 1 V4 | 0 | +1.4 |
| 种子 2 S0 | 100 | -2.4 |
| 种子 2 V4 | 50 | +2.7 |

- 种子 1 -1.20，种子 2 +0.17；四个合起来 -0.51（95% 区间 -3.5 到 +2.2）；只有 D 做对 140，只有 C 做对 146，p = 0.768
- D 第 300 次对这几个 D 检查点：四个合起来 +5.05（+2.5 到 +7.8），p = 0.000

## 各检查点的测试集成功题数（292 道）

| 组 | 更新次数 | 各 RL |
| --- | --- | --- |
| C | 0 | SFT 起点 113 |
| C | 50 | m3cs1s0 154，m3cs2s0 139，m3cs2d 129 |
| C | 100 | m3cs2s0 130 |
| D | 0 | SFT 起点 117 |
| D | 50 | m3ds1s0 143，m3ds1d 130，m3ds2s0 137，m3ds2d 137 |
| D | 100 | m3ds1s0 142，m3ds1d 140，m3ds2s0 123，m3ds2d 136 |
| D | 150 | m3ds1s0 133，m3ds1d 145，m3ds2s0 132，m3ds2d 148 |
| D | 200 | m3ds1s0 137，m3ds1d 150，m3ds2s0 131，m3ds2d 144 |
| D | 250 | m3ds1s0 146，m3ds1d 148，m3ds2s0 131，m3ds2d 142 |
| D | 300 | m3ds1s0 151，m3ds1d 147，m3ds2s0 140，m3ds2d 141 |
| E | 0 | SFT 起点 131 |
| E | 50 | m3es1s0 142，m3es1d 132，m3es2s0 125，m3es2d 119 |
| E | 100 | m3es1s0 136，m3es1d 131，m3es2s0 131，m3es2d 139 |
| E | 150 | m3es1s0 137，m3es1d 140，m3es2s0 148，m3es2d 136 |
| E | 200 | m3es1s0 136，m3es1d 136，m3es2s0 145，m3es2d 143 |
| E | 250 | m3es1s0 141，m3es2d 147 |
| E | 300 | m3es1s0 146，m3es2d 139 |
