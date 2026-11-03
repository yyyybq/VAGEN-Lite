# 矩阵第三版：交主线的结果（2026-11-03）

主线 11-03 要的：动作成功率两种口径、探针逐模型的数、失败拆解、权重的只读路径。用户 11-03 定：结果按预登记的格式交主线，汇总由主线写。

## 两种口径（测试集 292 道，去掉边缘场景）

- **预登记口径**（预登记第 2 节，主指标）：每个 RL 用第 300 次的检查点；被停止线停下的用最后一个检查点；第一个检查点之前就停下的（`m3cs1d`）用它的 SFT 起点。
- **按验证集挑**（主线 Qwen3 报告的做法；预登记里只作探索性）：每个 RL 在验证集 382 道上成功率最高的检查点（每 50 次更新一个，打平取更早的），报它在测试集上的成功率。

| 组 | 种子 | 侧 | 预登记口径：检查点 | 测试 | 按验证集挑：检查点 | 验证 | 测试 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C | 1 | S0 | `m3cs1s0_050` | 52.7 | `m3cs1s0_050` | 47.9 | 52.7 |
| C | 1 | V4 | `sft_m3c_e4` | 38.7 | `sft_m3c_e4` | 41.6 | 38.7 |
| C | 2 | S0 | `m3cs2s0_100` | 44.5 | `m3cs2s0_050` | 46.1 | 47.6 |
| C | 2 | V4 | `m3cs2d_050` | 44.2 | `m3cs2d_050` | 41.9 | 44.2 |
| D | 1 | S0 | `m3ds1s0_300` | 51.7 | `m3ds1s0_300` | 49.2 | 51.7 |
| D | 1 | V4 | `m3ds1d_300` | 50.3 | `m3ds1d_250` | 48.4 | 50.7 |
| D | 2 | S0 | `m3ds2s0_300` | 47.9 | `m3ds2s0_300` | 45.6 | 47.9 |
| D | 2 | V4 | `m3ds2d_300` | 48.3 | `m3ds2d_300` | 47.9 | 48.3 |
| E | 1 | S0 | `m3es1s0_300` | 50.0 | `m3es1s0_300` | 49.5 | 50.0 |
| E | 1 | V4 | `m3es1d_200` | 46.6 | `m3es1d_150` | 48.2 | 47.9 |
| E | 2 | S0 | `m3es2s0_200` | 49.7 | `m3es2s0_150` | 47.4 | 50.7 |
| E | 2 | V4 | `m3es2d_300` | 47.6 | `m3es2d_200` | 51.0 | 49.0 |
| C0 | 1 | S0 | `c0q3_s1s0_150` | 42.5 | `c0q3_s1s0_150` | 44.8 | 42.5 |
| C0 | 1 | V4 | `c0q3_s1d_300` | 49.0 | `c0q3_s1d_300` | 49.7 | 49.0 |
| C0 | 2 | S0 | `c0q3_s2s0_300` | 46.9 | `c0q3_s2s0_300` | 48.2 | 46.9 |
| C0 | 2 | V4 | `c0q3_s2d_300` | 46.6 | `c0q3_s2d_300` | 49.7 | 46.6 |

四个 RL 合起来（测试集成功率，%）：

| 组 | 预登记口径 | 按验证集挑 |
| --- | --- | --- |
| C | 45.0 | 45.8 |
| D | 49.6 | 49.7 |
| E | 48.5 | 49.4 |
| C0 | 46.2 | 46.2 |

- C0 在去掉边缘场景的两个题集上只评过预登记口径用的检查点和 SFT 第 8 个 epoch（`PROGRESS.md` 12.27），所以它的两种口径相同。模型线的 C0 报告是在原来的 `eval_val_v2` / `eval_test_v2` 上按验证集挑的，那是另一个题集。
- C 种子 1 的 V4（`m3cs1d`）在第 47 次停下，没有检查点，两种口径都用它的 SFT 起点（用户 11-03 定）。

## 探针（838 道；逐模型）

| 模型 | 全部 | `goal_now` | `goal_after` | 遮挡平衡准确率 | 读不出 |
| --- | --- | --- | --- | --- | --- |
| `sft_m3c_e4` | 51.1 | 56.8 | 50.3 | 34.0 | 0.0% |
| `m3cs1s0_050` | 53.5 | 55.7 | 52.0 | 35.3 | 0.0% |
| `m3cs2s0_100` | 51.1 | 53.4 | 52.6 | 32.5 | 0.0% |
| `m3cs2d_050` | 50.6 | 52.8 | 51.5 | 31.6 | 0.0% |
| `sft_m3d_e9` | 64.0 | 71.0 | 68.4 | 31.0 | 0.0% |
| `m3ds1s0_300` | 61.7 | 67.6 | 66.7 | 26.2 | 0.0% |
| `m3ds1d_300` | 63.5 | 70.5 | 69.6 | 37.2 | 0.0% |
| `m3ds2s0_300` | 64.2 | 71.0 | 69.6 | 33.2 | 0.0% |
| `m3ds2d_300` | 62.9 | 71.6 | 69.0 | 30.9 | 0.0% |
| `m3es1s0_300` | 46.8 | 52.8 | 48.5 | 43.5 | 0.0% |
| `m3es1d_200` | 48.3 | 54.0 | 50.3 | 46.4 | 0.2% |
| `m3es2s0_200` | 45.7 | 52.8 | 53.2 | 30.6 | 0.0% |
| `m3es2d_300` | 47.1 | 54.0 | 54.4 | 37.1 | 0.0% |
| `c0q3_sft_e8` | 46.1 | 56.2 | 50.9 | 35.6 | 0.0% |
| `c0q3_s1s0_150` | 46.8 | 57.4 | 52.6 | 33.1 | 0.1% |
| `c0q3_s1d_300` | 45.8 | 52.3 | 48.5 | 36.1 | 0.0% |
| `c0q3_s2s0_300` | 48.3 | 55.7 | 52.6 | 44.7 | 0.0% |
| `c0q3_s2d_300` | 47.1 | 53.4 | 53.8 | 40.4 | 0.1% |

比较和判定见 `rest.md` 第 3.3 节；逐题结果在各链的 `diag/probe_P1/<模型>.jsonl`（JSON 里的 `path`）。

## 失败拆解

- 预登记 3.4 的汇总：`rest.json` 的 `sets.test["3.4"]`（测试集）、`sets.val["3.4"]`（验证集），表在 `rest.md`。
- 每条链的拆解：`<链>/diag/failure_test.json`、`failure_val.json`（本文件 JSON 的 `failure_breakdown.per_chain`）。

## 权重（只读）

主线按路径只读，不要改、不要删。D_noP0、B′ 挑出 epoch 后再补。

- C (epoch 4)：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/C/model/checkpoint-1204`
- D (epoch 9)：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D/model/checkpoint-3987`
- C0 / E start (epoch 8)：`/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232`
- RL `c0q3_s1d_300`：`/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/rl_run_q3d/checkpoints/global_step_300/actor/huggingface`
- RL `c0q3_s1s0_150`：`/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/rl_run_q3s0/checkpoints/global_step_150/actor/huggingface`
- RL `c0q3_s2d_300`：`/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261027_s20261021/rl_run_q3s2d/checkpoints/global_step_300/actor/huggingface`
- RL `c0q3_s2s0_300`：`/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261027_s20261021/rl_run_q3s2s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3cs1s0_050`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_c_s20261019/rl_run_m3cs1s0/checkpoints/global_step_50/actor/huggingface`
- RL `m3cs2d_050`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_c_s20261021/rl_run_m3cs2d/checkpoints/global_step_50/actor/huggingface`
- RL `m3cs2s0_050`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_c_s20261021/rl_run_m3cs2s0/checkpoints/global_step_50/actor/huggingface`
- RL `m3cs2s0_100`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_c_s20261021/rl_run_m3cs2s0/checkpoints/global_step_100/actor/huggingface`
- RL `m3ds1d_250`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_d_s20261019/rl_run_m3ds1d/checkpoints/global_step_250/actor/huggingface`
- RL `m3ds1d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_d_s20261019/rl_run_m3ds1d/checkpoints/global_step_300/actor/huggingface`
- RL `m3ds1s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_d_s20261019/rl_run_m3ds1s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3ds2d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_d_s20261021/rl_run_m3ds2d/checkpoints/global_step_300/actor/huggingface`
- RL `m3ds2s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_d_s20261021/rl_run_m3ds2s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3es1d_150`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_e_s20261019/rl_run_m3es1d/checkpoints/global_step_150/actor/huggingface`
- RL `m3es1d_200`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_e_s20261019/rl_run_m3es1d/checkpoints/global_step_200/actor/huggingface`
- RL `m3es1s0_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_e_s20261019/rl_run_m3es1s0/checkpoints/global_step_300/actor/huggingface`
- RL `m3es2d_200`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_e_s20261021/rl_run_m3es2d/checkpoints/global_step_200/actor/huggingface`
- RL `m3es2d_300`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_e_s20261021/rl_run_m3es2d/checkpoints/global_step_300/actor/huggingface`
- RL `m3es2s0_150`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_e_s20261021/rl_run_m3es2s0/checkpoints/global_step_150/actor/huggingface`
- RL `m3es2s0_200`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_e_s20261021/rl_run_m3es2s0/checkpoints/global_step_200/actor/huggingface`
- RL `sft_m3c_e4`：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/C/model/checkpoint-1204`
- D_noP0 SFT (every epoch kept until the pick)：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D_noP0/model`
- B' SFT (every epoch kept until the pick)：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/B/model`
