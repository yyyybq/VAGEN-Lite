# 矩阵第三版补充格：交主线的结果（2026-11-04；11-05 加了 Cr 的 RL；11-06 加了训练任务崩溃的记录）

主线 11-04 要的：补充格 SFT 挑点的测试集成功率、探针 goal_now / goal_after、d、采样验收；RL 的检查点路径。口径同 `handover_main_20261103`（测试集 292 道、验证集 382 道，都去掉边缘场景）。

## SFT 挑点

| 格 | 挑点 | 测试 % | 验证 % | 探针 goal_now / goal_after % | d 平均（中位），失败局 | 采样验收：熵、读不出、过没过 |
| --- | --- | --- | --- | --- | --- | --- |
| B' | `sft_m3b_e10` | 32.2 | 31.9 | 55.1 / 55.0 | 5.20（5），198 | 0.045、0.17%、过 |
| D_noP0 | `sft_m3dn_e8` | 42.1 | 40.3 | 72.7 / 64.3 | 6.25（6），169 | 0.032、0.00%、过 |
| E2 | `sft_m3e2_e10` | 9.2 | 9.4 | 72.7 / 66.7 | 8.81（9），265 | 0.022、0.04%、过 |
| Cr | `sft_m3cr_e7` | 40.4 | 40.6 | 55.7 / 53.8 | 5.76（5），174 | 0.053、0.00%、过 |

- 探针：B'、D_noP0、Cr 的挑点是 RL 链诊断步骤里的 `start`（RL 的起点就是这个挑点）。E2 没有 RL 链，探针单独跑（sco_probe `pt-ostoj75m`，结果在 `probe_cells_20261104/probe/`）；Cr 接 RL 之前也在那里单独跑过一次，和链里的 `start` 逐题一致。
- d：失败局终点离成功还差几步，`reward.fail_end_steps`，设置同主线 `failend_m3_20261103`。逐局结果：SFT 挑点在 `handover_cells_20261104/failend_test.json`，RL 的预登记检查点在 `failend_rl_test.json`（B′、D_noP0）和 `failend_cr_rl_test.json`（Cr，11-05）。

## RL（跑完的链）

| 格 | 种子 | 侧 | 跑了几次更新 | 预登记口径：检查点 | 测试 | d | 按验证集挑：检查点 | 验证 | 测试 | 权重在盘上 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B' | 20261019 | S0 | 300 | `m3bs1s0_300` | 43.5 | 5.39 | `m3bs1s0_200` | 43.2 | 43.2 | 是 |
| B' | 20261019 | V4 | 300 | `m3bs1d_300` | 45.2 | 5.40 | `m3bs1d_300` | 41.1 | 45.2 | 是 |
| D_noP0 | 20261019 | S0 | 300 | `m3dns1s0_300` | 52.1 | 5.38 | `m3dns1s0_150` | 47.9 | 49.7 | 是 |
| D_noP0 | 20261019 | V4 | 300 | `m3dns1d_300` | 49.0 | 5.46 | `m3dns1d_250` | 47.1 | 49.7 | 是 |
| D_noP0 | 20261021 | S0 | 300 | `m3dns2s0_300` | 47.3 | 5.50 | `m3dns2s0_300` | 46.6 | 47.3 | 是 |
| D_noP0 | 20261021 | V4 | 300 | `m3dns2d_300` | 51.7 | 5.65 | `m3dns2d_300` | 49.2 | 51.7 | 是 |
| Cr | 20261019 | S0 | 270（停下：读不出的回答超过 10%） | `m3crs1s0_250` | 52.1 | 6.21 | `m3crs1s0_150` | 50.0 | 55.1 | 是 |
| Cr | 20261019 | V4 | 286（停下：读不出的回答超过 10%） | `m3crs1d_250` | 50.7 | 5.19 | `m3crs1d_150` | 48.2 | 49.0 | 是 |
| Cr | 20261021 | S0 | 300 | `m3crs2s0_300` | 50.3 | 5.46 | `m3crs2s0_250` | 47.6 | 47.6 | 是 |
| Cr | 20261021 | V4 | 223（停下：训练进程连续两次崩溃（Qwen3-VL 图像 token）） | `m3crs2d_200` | 48.6 | 5.51 | `m3crs2d_100` | 46.3 | 49.0 | 是 |

停下的原因（训练链原文）：
- `m3crs1s0`（2026-11-05T01:00:41+00:00）：unparseable answers 0.101 > 0.1 (mean of updates 261-270)
- `m3crs1d`（2026-11-05T01:29:09+00:00）：unparseable answers 0.116 > 0.1 (mean of updates 277-286)
- `m3crs2d`（2026-11-05T01:55:50+00:00）：rule 91: 2 trainer attempts in a row ended with the Qwen3-VL image-token crash (last: RuntimeError: shape mismatch: value tensor of shape [3, 4820] cannot be broadcast to indexing result of shape [3, 4771])

合起来（测试集成功率 %；d 和探针是各运行预登记检查点的平均）：

| 格 | 运行数 | 预登记口径 | 按验证集挑 | d | 探针 goal_now / goal_after % | 300 次前被停下 | 齐了没有 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B' | 2 | 44.3 | 44.2 | 5.40 | 54.5 / 57.3 | 0/2 | 齐 |
| D_noP0 | 4 | 50.0 | 49.6 | 5.50 | 72.4 / 61.8 | 0/4 | 齐 |
| Cr | 4 | 50.4 | 50.2 | 5.60 | 55.8 / 54.2 | 3/4 | 齐 |

## 训练任务崩溃（11-06 加；矩阵第三版全部 RL，各链 `progress.json` 的 `trainer_crashes`）

22 个 RL 里，Qwen3-VL 图像 token 的形状错误一共 8 次，出在 5 个 RL 上，全是 C 配方（C、Cr）；被图像 token 崩溃线（连续 2 次就停）停下的是其中 3 个。11-05 说的"3 个 RL、都在种子 20261021"只数了被这条线停下的（`PROGRESS.md` 12.55 更正）。

| 链 | RL | 崩溃次数（其中图像 token） | 时间（UTC） | 结局 |
| --- | --- | --- | --- | --- |
| `m3_c_s20261019` | `m3cs1s0` | 1（1） | 11-02 20:34 | 熵连续 5 次超过 0.3 |
| `m3_c_s20261019` | `m3cs1d` | 1（1） | 11-02 21:06 | 熵连续 5 次超过 0.3 |
| `m3_c_s20261021` | `m3cs2s0` | 2（2） | 11-02 22:16、11-03 01:10 | 训练进程连续两次崩溃（Qwen3-VL 图像 token） |
| `m3_c_s20261021` | `m3cs2d` | 2（2） | 11-02 21:25、11-03 01:00 | 训练进程连续两次崩溃（Qwen3-VL 图像 token） |
| `m3_cr_s20261021` | `m3crs2d` | 2（2） | 11-05 01:14、11-05 01:55 | 训练进程连续两次崩溃（Qwen3-VL 图像 token） |
| `m3_dn_s20261021` | `m3dns2d` | 1（0） | 11-04 11:09 | 跑满 300 次 |

其余 RL 没有崩溃记录。

## 权重（只读）

- SFT B' sft_m3b_e10：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/B/model/checkpoint-1870`
- SFT D_noP0 sft_m3dn_e8：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D_noP0/model/checkpoint-2272`
- SFT E2 sft_m3e2_e10：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/E2/model/checkpoint-3010`
- SFT Cr sft_m3cr_e7：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/Cr/model/checkpoint-2107`
- RL `m3bs1d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_b_s20261019/rl_run_m3bs1d/checkpoints/global_step_300/actor/huggingface`
- RL `m3bs1s0_200`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_b_s20261019/rl_run_m3bs1s0/checkpoints/global_step_200/actor/huggingface`
- RL `m3bs1s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_b_s20261019/rl_run_m3bs1s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3crs1d_150`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261019/rl_run_m3crs1d/checkpoints/global_step_150/actor/huggingface`
- RL `m3crs1d_250`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261019/rl_run_m3crs1d/checkpoints/global_step_250/actor/huggingface`
- RL `m3crs1s0_150`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261019/rl_run_m3crs1s0/checkpoints/global_step_150/actor/huggingface`
- RL `m3crs1s0_250`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261019/rl_run_m3crs1s0/checkpoints/global_step_250/actor/huggingface`
- RL `m3crs2d_100`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261021/rl_run_m3crs2d/checkpoints/global_step_100/actor/huggingface`
- RL `m3crs2d_200`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261021/rl_run_m3crs2d/checkpoints/global_step_200/actor/huggingface`
- RL `m3crs2s0_250`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261021/rl_run_m3crs2s0/checkpoints/global_step_250/actor/huggingface`
- RL `m3crs2s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_cr_s20261021/rl_run_m3crs2s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns1d_250`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1d/checkpoints/global_step_250/actor/huggingface`
- RL `m3dns1d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1d/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns1s0_150`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1s0/checkpoints/global_step_150/actor/huggingface`
- RL `m3dns1s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns2d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261021/rl_run_m3dns2d/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns2s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261021/rl_run_m3dns2s0/checkpoints/global_step_300/actor/huggingface`
