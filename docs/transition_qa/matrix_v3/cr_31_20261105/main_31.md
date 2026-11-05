# 矩阵第三版：Cr 和 C、D 比（按 3.1 的方法，探索性）

Matrix v3, Cr (pre-registration note 9, exploratory): Cr's RL against C and D by the method of 3.1 (m3_main31.py:
the same pairs, sign test, bootstrap draws and verdict rule) at each RL's pre-registered checkpoint (update 300, or the
last checkpoint of a run stopped early): m3crs1s0_250 (stopped at 270), m3crs1d_250 (stopped at 286), m3crs2s0_300,
m3crs2d_200 (stopped at 223).  Exploratory: it changes no H1-H3 verdict and does not replace C's result.  Read only;
writes OUT.

## Cr − C（探索性）

| 配对 | 差（百分点） | 只有 Cr 做对 | 只有 C 做对 | p |
| --- | --- | --- | --- | --- |
| 种子 1 S0 | -0.7 | 35 | 37 | 0.906 |
| 种子 1 V4 | +12.0 | 59 | 24 | 0.000 |
| 种子 2 S0 | +5.8 | 44 | 27 | 0.057 |
| 种子 2 V4 | +4.5 | 43 | 30 | 0.160 |

- 种子 1：+5.65（95% 区间 +1.5 到 +9.6）；种子 2：+5.14（+0.9 到 +9.2）；同号：是
- 四个合起来：+5.39 个百分点（+2.4 到 +8.2）；只有 Cr 做对 181，只有 C 做对 118，符号检验 p = 0.000
- 判定：有差异（p < 0.05：是；差 ≥ 5：是；同号：是）

## Cr − D（探索性）

| 配对 | 差（百分点） | 只有 Cr 做对 | 只有 D 做对 | p |
| --- | --- | --- | --- | --- |
| 种子 1 S0 | +0.3 | 35 | 34 | 1.000 |
| 种子 1 V4 | +0.3 | 30 | 29 | 1.000 |
| 种子 2 S0 | +2.4 | 35 | 28 | 0.450 |
| 种子 2 V4 | +0.3 | 39 | 38 | 1.000 |

- 种子 1：+0.34（95% 区间 -3.6 到 +4.1）；种子 2：+1.37（-2.6 到 +5.1）；同号：是
- 四个合起来：+0.86 个百分点（-2.1 到 +3.8）；只有 Cr 做对 139，只有 D 做对 129，符号检验 p = 0.583
- 判定：没有差异（p < 0.05：否；差 ≥ 5：否；同号：是）

