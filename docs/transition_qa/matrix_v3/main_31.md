# 矩阵第三版：主指标（预登记 3.1）

Matrix v3, pre-registration 3.1 (main metric): action success on eval_test_v2_screened (292 tasks) at each RL's
pre-registered checkpoint (section 2: update 300; a stopped RL its last checkpoint; m3cs1d, stopped at 47 before its
first checkpoint, uses C's picked SFT: the user, 2026-11-03, "用 SFT 起点").  Pairs: same seed, same side, task by
task (matched by task_id).  Pooled = the four pairs together (4 x 292); each seed's difference = the mean of its S0 and
V4 pairs.  Test: exact two-sided sign test on the tasks only one side solved, pooled; 95% CI from 2,000 bootstrap
draws of the 292 tasks (the same draws for every pair).  Verdict: pooled p < 0.05 and pooled difference >= 5 pp and
the two seeds' differences of the same sign (both > 0 or both < 0).  D - C is the main comparison, D - C0 secondary,
E - C0 exploratory (pre-registration note 4).  Read only; writes OUT.

## D − C（主比较）

| 配对 | 差（百分点） | 只有 D 做对 | 只有 C 做对 | p |
| --- | --- | --- | --- | --- |
| 种子 1 S0 | -1.0 | 32 | 35 | 0.807 |
| 种子 1 V4 | +11.6 | 52 | 18 | 0.000 |
| 种子 2 S0 | +3.4 | 42 | 32 | 0.295 |
| 种子 2 V4 | +4.1 | 38 | 26 | 0.169 |

- 种子 1：+5.31（95% 区间 +1.2 到 +9.4）；种子 2：+3.77（-0.2 到 +7.9）；同号：是
- 四个合起来：+4.54 个百分点（+1.8 到 +7.4）；只有 D 做对 164，只有 C 做对 111，符号检验 p = 0.002
- 判定：没有差异（p < 0.05：是；差 ≥ 5：否；同号：是）

## D − C0（次要比较）

| 配对 | 差（百分点） | 只有 D 做对 | 只有 C0 做对 | p |
| --- | --- | --- | --- | --- |
| 种子 1 S0 | +9.2 | 49 | 22 | 0.002 |
| 种子 1 V4 | +1.4 | 34 | 30 | 0.708 |
| 种子 2 S0 | +1.0 | 33 | 30 | 0.801 |
| 种子 2 V4 | +1.7 | 34 | 29 | 0.615 |

- 种子 1：+5.31（95% 区间 +1.2 到 +9.6）；种子 2：+1.37（-2.4 到 +5.3）；同号：是
- 四个合起来：+3.34 个百分点（+0.5 到 +6.4）；只有 D 做对 150，只有 C0 做对 111，符号检验 p = 0.018
- 判定：没有差异（p < 0.05：是；差 ≥ 5：否；同号：是）

## E − C0（探索性）

| 配对 | 差（百分点） | 只有 E 做对 | 只有 C0 做对 | p |
| --- | --- | --- | --- | --- |
| 种子 1 S0 | +7.5 | 42 | 20 | 0.007 |
| 种子 1 V4 | -2.4 | 27 | 34 | 0.443 |
| 种子 2 S0 | +2.7 | 33 | 25 | 0.358 |
| 种子 2 V4 | +1.0 | 31 | 28 | 0.795 |

- 种子 1：+2.57（95% 区间 -0.7 到 +6.2）；种子 2：+1.88（-2.0 到 +5.7）；同号：是
- 四个合起来：+2.23 个百分点（-0.3 到 +5.0）；只有 E 做对 133，只有 C0 做对 107，符号检验 p = 0.106
- 判定：没有差异（p < 0.05：否；差 ≥ 5：否；同号：是）

