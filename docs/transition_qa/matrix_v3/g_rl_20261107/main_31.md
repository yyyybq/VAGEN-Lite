# 预登记第四版：G 和 D、C 比（3.1）

Matrix v3, group G (PREREG_G_vs_D.md, the fourth pre-registration; the user calls this design G-text from 2026-11-07):
G against D (H1, main) and C (H2, secondary) by the method of 3.1 (m3_main31.py: the same pairs, sign test, bootstrap
draws and verdict rule) at each RL's pre-registered checkpoint (update 300, or the last checkpoint of a run stopped
early): m3gs1s0_200 (stopped at 229), m3gs1d_250 (stopped at 262), m3gs2s0_300, m3gs2d_250 (stopped at 265); all
three stops by the Qwen3-VL image-token crash rule.  Read only; writes OUT.

## G − D（主比较）

| 配对 | 差（百分点） | 只有 G 做对 | 只有 D 做对 | p |
| --- | --- | --- | --- | --- |
| 种子 1 S0 | -3.1 | 31 | 40 | 0.343 |
| 种子 1 V4 | -4.8 | 27 | 41 | 0.114 |
| 种子 2 S0 | +3.4 | 45 | 35 | 0.314 |
| 种子 2 V4 | -1.4 | 29 | 33 | 0.704 |

- 种子 1：-3.94（95% 区间 -8.1 到 +0.3）；种子 2：+1.03（-3.2 到 +5.3）；同号：否
- 四个合起来：-1.46 个百分点（-4.6 到 +1.9）；只有 G 做对 132，只有 D 做对 149，符号检验 p = 0.340
- 判定：没有差异（p < 0.05：否；差 ≥ 5：否；同号：否）

## G − C（次要比较）

| 配对 | 差（百分点） | 只有 G 做对 | 只有 C 做对 | p |
| --- | --- | --- | --- | --- |
| 种子 1 S0 | -4.1 | 33 | 45 | 0.213 |
| 种子 1 V4 | +6.8 | 49 | 29 | 0.031 |
| 种子 2 S0 | +6.8 | 54 | 34 | 0.042 |
| 种子 2 V4 | +2.7 | 40 | 32 | 0.410 |

- 种子 1：+1.37（95% 区间 -3.1 到 +5.7）；种子 2：+4.79（+0.2 到 +9.8）；同号：是
- 四个合起来：+3.08 个百分点（-0.4 到 +6.6）；只有 G 做对 176，只有 C 做对 140，符号检验 p = 0.049
- 判定：没有差异（p < 0.05：是；差 ≥ 5：否；同号：是）

