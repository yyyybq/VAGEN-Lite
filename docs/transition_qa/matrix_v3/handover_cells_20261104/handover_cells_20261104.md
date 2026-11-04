# 矩阵第三版补充格：交主线的结果（2026-11-04）

主线 11-04 要的：补充格 SFT 挑点的测试集成功率、探针 goal_now / goal_after、d、采样验收；RL 的检查点路径。口径同 `handover_main_20261103`（测试集 292 道、验证集 382 道，都去掉边缘场景）。

## SFT 挑点

| 格 | 挑点 | 测试 % | 验证 % | 探针 goal_now / goal_after % | d 平均（中位），失败局 | 采样验收：熵、读不出、过没过 |
| --- | --- | --- | --- | --- | --- | --- |
| B' | `sft_m3b_e10` | 32.2 | 31.9 | 55.1 / 55.0 | 5.20（5），198 | 0.045、0.17%、过 |
| D_noP0 | `sft_m3dn_e8` | 42.1 | 40.3 | 72.7 / 64.3 | 6.25（6），169 | 0.032、0.00%、过 |
| E2 | `sft_m3e2_e10` | 9.2 | 9.4 | 72.7 / 66.7 | 8.81（9），265 | 0.022、0.04%、过 |
| Cr | `sft_m3cr_e7` | 40.4 | 40.6 | 55.7 / 53.8 | 5.76（5），174 | 0.053、0.00%、过 |

- 探针：B′、D_noP0 的挑点是 RL 链诊断步骤里的 `start`（RL 的起点就是这个挑点）；E2、Cr 没有 RL 链，探针单独跑（sco_probe `pt-ostoj75m`，两个模型一个任务，结果在 `probe_cells_20261104/probe/`）。
- d：失败局终点离成功还差几步，`reward.fail_end_steps`，设置同主线 `failend_m3_20261103`。逐局结果：SFT 挑点在 `handover_cells_20261104/failend_test.json`，RL 的预登记检查点在 `failend_rl_test.json`。

## RL（跑完的链）

| 格 | 种子 | 侧 | 预登记口径：检查点 | 测试 | d | 按验证集挑：检查点 | 验证 | 测试 | 权重在盘上 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B' | 20261019 | S0 | `m3bs1s0_300` | 43.5 | 5.39 | `m3bs1s0_200` | 43.2 | 43.1 | 是 |
| B' | 20261019 | V4 | `m3bs1d_300` | 45.2 | 5.40 | `m3bs1d_300` | 41.1 | 45.2 | 是 |
| D_noP0 | 20261019 | S0 | `m3dns1s0_300` | 52.0 | 5.38 | `m3dns1s0_150` | 47.9 | 49.7 | 是 |
| D_noP0 | 20261019 | V4 | `m3dns1d_300` | 49.0 | 5.46 | `m3dns1d_250` | 47.1 | 49.7 | 是 |
| D_noP0 | 20261021 | S0 | `m3dns2s0_300` | 47.3 | 5.50 | `m3dns2s0_300` | 46.6 | 47.3 | 是 |
| D_noP0 | 20261021 | V4 | `m3dns2d_300` | 51.7 | 5.65 | `m3dns2d_300` | 49.2 | 51.7 | 是 |

合起来（测试集成功率 %；d 和探针是各运行预登记检查点的平均）：

| 格 | 运行数 | 预登记口径 | 按验证集挑 | d | 探针 goal_now / goal_after % | 300 次前被停下 | 齐了没有 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B' | 2 | 44.4 | 44.2 | 5.40 | 54.5 / 57.3 | 0/2 | 齐 |
| D_noP0 | 4 | 50.0 | 49.6 | 5.50 | 72.4 / 61.8 | 0/4 | 齐 |

## 权重（只读）

- SFT B' sft_m3b_e10：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/B/model/checkpoint-1870`
- SFT D_noP0 sft_m3dn_e8：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D_noP0/model/checkpoint-2272`
- SFT E2 sft_m3e2_e10：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/E2/model/checkpoint-3010`
- SFT Cr sft_m3cr_e7：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/Cr/model/checkpoint-2107`
- RL `m3bs1d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_b_s20261019/rl_run_m3bs1d/checkpoints/global_step_300/actor/huggingface`
- RL `m3bs1s0_200`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_b_s20261019/rl_run_m3bs1s0/checkpoints/global_step_200/actor/huggingface`
- RL `m3bs1s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_b_s20261019/rl_run_m3bs1s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns1d_250`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1d/checkpoints/global_step_250/actor/huggingface`
- RL `m3dns1d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1d/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns1s0_150`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1s0/checkpoints/global_step_150/actor/huggingface`
- RL `m3dns1s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261019/rl_run_m3dns1s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns2d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261021/rl_run_m3dns2d/checkpoints/global_step_300/actor/huggingface`
- RL `m3dns2s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_dn_s20261021/rl_run_m3dns2s0/checkpoints/global_step_300/actor/huggingface`
