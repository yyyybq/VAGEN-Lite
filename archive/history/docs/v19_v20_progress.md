# v19 → v20 实验进度与决策记录

> 文档生成时间: 2026-05-21
> 范围: VAGEN-Lite Active Spatial RL 训练 + SFT 下游管线

## 1. 阶段总览

本轮工作分两条线:
- **A. SFT 下游管线** (verl 不支持 VL SFT → 切换 LLaMA-Factory)
- **B. v19 RL 实验进度分析与 v20 候选实验设计**

---

## 2. SFT 下游管线 (新建)

| 文件 | 作用 |
|---|---|
| [data_gen/active_spatial_sft/convert_to_qwen25vl_sft.py](data_gen/active_spatial_sft/convert_to_qwen25vl_sft.py) | JSONL → JSONL/Parquet 转换；新增 `--to_parquet` flag (`pd.DataFrame.from_records(...).to_parquet`) |
| [data_gen/active_spatial_sft/lf_dataset_info.json](data_gen/active_spatial_sft/lf_dataset_info.json) | LLaMA-Factory dataset_info 注册项 |
| [data_gen/active_spatial_sft/lf_qwen25vl_3b_sft.yaml](data_gen/active_spatial_sft/lf_qwen25vl_3b_sft.yaml) | LF 训练配置 (full FT, freeze_vision_tower=true, FSDP, bf16, fa2) |
| [data_gen/active_spatial_sft/run_sft_qwen25vl_3b.sh](data_gen/active_spatial_sft/run_sft_qwen25vl_3b.sh) | 一键管线: 安装 LF → 生成 accelerate FSDP yaml → 运行 |

**关键决策**:
- verl 的 `MultiTurnSFTDataset` 仅支持纯文本，`fsdp_sft_trainer.py` 不会前向 `pixel_values` → 不能直接做 Qwen2.5-VL SFT，转用 LLaMA-Factory。
- `freeze_vision_tower=true` 与 Qwen-VL SFT-then-RL 经典 recipe 对齐。
- SFT ckpt 输出位置: `/scratch/by2593/project/Active_Spatial/VAGEN-Lite/checkpoints/sft_qwen25vl_3b_5k`，将作为 v20_sft_* 实验的 `MODEL_PATH`。

**5k 数据生成**: pid=2761914 仍在运行，最近进度 ~item 81/5000。日志: `data_gen/active_spatial_sft/run_100scenes_5k.log`。

---

## 3. v19 RL 实验进度分析

### 3.1 进程状态 (截至日志最近更新)

| 实验 | 日志最近更新 | 状态 |
|---|---|---|
| v19_clean | 0 min | 运行中 → **建议终止** |
| v19_thr70_andgate | 2 min | 运行中（val 已崩塌） |
| v19_no_farm | 1 min | 运行中（val 0.880, 仅 2 ckpt） |
| v19_succ10 | 0 min | 运行中（仅 1 ckpt） |
| v19_no_autoterm | 24 min | 已停（疑似挂掉）→ **建议终止** |
| v19_thr85_no_autoterm | ~5.5 h | 已停 |
| v19_dual | ~9 h | 已停 |
| v19_thr85_andgate | ~9.5 h | 已停 |

### 3.2 综合指标

| 实验 | 关键参数差异 | train peak | val best@4 序列 | val 峰值 | 结论 |
|---|---|---|---|---|---|
| **v19_thr70_andgate** | thr=0.70 / AND-gate / autoterm ON | +4.94 @ s67 | 0.654→**0.859**→0.742→0.689→0.600→0.036→0.243→0.000 | 0.859 | winner，但**后期严重崩塌** |
| **v19_dual** | thr70_andgate + dual progress (pos 0.3 / ori 0.7) | +4.78 @ s66 | 0.675→**0.808**→0.702 | 0.808 | 稳定，3 ckpt 已停 |
| **v19_no_farm** | thr70_andgate + format=0 / step_pen=-0.02 | +3.43 @ s12 | 0.560→**0.880** | **0.880** | 当前最高峰，2 ckpt |
| **v19_succ10** | thr70_andgate + success_reward 50→10 | +0.55 | 0.669 | 0.669 | 1 ckpt，太早 |
| **v19_thr85_andgate** | SFT 级稀疏 from-scratch | +2.85 @ s35 | 0.480→0.496→**0.623**→0.086→0.000→0.442 | 0.623 | from-scratch 起步太慢，需 SFT |
| **v19_clean** | v19 hyper baseline（无 reward 改） | +0.25 | 0.036→0.175→0.036→0→0.035→0.035 | 0.175 | 不学习 |
| **v19_no_autoterm** | autoterm OFF + 显式 done | +0.26 (entropy 5.75) | 多次 0 / 偶 0.107 | 0.107 | 失败：探索发散 |
| **v19_thr85_no_autoterm** | autoterm OFF + thr=0.85 | +0.07 (entropy 0.73) | 0.036→0.036→0→0 | 0.036 | 失败：模式崩溃 |

### 3.3 单变量假设验证结果

| 假设 | 来源实验 | 结论 |
|---|---|---|
| AND-gate + thr=0.70 优于 OR-gate + thr=0.45 | v19_thr70_andgate | ✅ 成立 |
| dual progress 解锁 ori 通道 | v19_dual | ✅ 成立 (val 0.808 稳定) |
| format_reward=0.05 是后期噪声/farm 源 | v19_no_farm | ✅ 早期成立 (0.880 峰值)，需更多 ckpt |
| autoterm OFF + agent 自决 done (from-scratch) | v19_no_autoterm / v19_thr85_no_autoterm | ❌ 已证伪 |
| thr=0.85 SFT 级 from-scratch 可行 | v19_thr85_andgate | ❌ 信号太稀 |
| success_reward 50→10 改善 cliff | v19_succ10 | ⏸ 数据不足 |

### 3.4 关键问题

- **v19_thr70_andgate 后期崩塌**: val best@4 从 step ~40 的 0.859 崩到 step 100+ 的 0.000，train score 仍 +4.94 → 经典 RL 失稳模式（critic 失稳 / KL 泄漏 / format-farm 入侵）。这正是 v20_winner / v19_thr70_klhi 设计目标。

---

## 4. v20 / 新增 v19 候选实验 (本轮新建)

### 4.1 三个新 sh 脚本 + 对应 yaml

| 实验 | sh 文件 | yaml | 与 winner 关系 |
|---|---|---|---|
| **v19_no_farm_dual** | [examples/train/active_spatial/experiments/v19_no_farm_dual.sh](examples/train/active_spatial/experiments/v19_no_farm_dual.sh) | [examples/train/active_spatial/env_config_v19_no_farm_dual.yaml](examples/train/active_spatial/env_config_v19_no_farm_dual.yaml) (新建) | 纯 yaml 三路合并 (thr70_andgate × no_farm × dual)；sh 沿用 thr70_andgate |
| **v20_winner** | [examples/train/active_spatial/experiments/v20_winner.sh](examples/train/active_spatial/experiments/v20_winner.sh) | 复用 `env_config_v19_no_farm_dual.yaml` | 同 yaml + sh 级稳定性补丁 (KL_LOSS_COEF 0.05→0.10, CRITIC_WARMUP 30→60, CLIPRANGE_VALUE 0.8→0.5, ENTROPY_COEFF 0.001→0.002) |
| **v19_thr70_klhi** | [examples/train/active_spatial/experiments/v19_thr70_klhi.sh](examples/train/active_spatial/experiments/v19_thr70_klhi.sh) | 复用 `env_config_v19_thr70_andgate.yaml` (不变) | sh-only 抢救，验证稳定补丁单独是否能修崩塌 |

#### 三者的因子设计

|  | yaml 三路合并 | sh 稳定补丁 |
|---|:---:|:---:|
| v19_thr70_andgate（基线） | ✗ | ✗ |
| v19_no_farm_dual | ✓ | ✗ |
| v19_thr70_klhi | ✗ | ✓ |
| v20_winner | ✓ | ✓ |

→ 这是 2×2 设计，能干净 attribute "yaml 合并" vs "sh 补丁" 两个改动各自的贡献。

### 4.2 SFT-warmstart 两个待启动实验

| 实验 | sh 文件 | yaml | 启动前置 |
|---|---|---|---|
| **v20_sft_thr85_andgate** | [examples/train/active_spatial/experiments/v20_sft_thr85_andgate.sh](examples/train/active_spatial/experiments/v20_sft_thr85_andgate.sh) | 复用 `env_config_v19_thr85_andgate.yaml` | 5k SFT 数据完成 + LF SFT 训练完成 |
| **v20_sft_no_autoterm** | [examples/train/active_spatial/experiments/v20_sft_no_autoterm.sh](examples/train/active_spatial/experiments/v20_sft_no_autoterm.sh) | 复用 `env_config_v19_thr85_no_autoterm.yaml` | 同上 |

两者通过 `MODEL_PATH=/scratch/by2593/project/Active_Spatial/VAGEN-Lite/checkpoints/sft_qwen25vl_3b_5k` 加载 SFT ckpt 作 actor 初始。

**启动前必检**:
1. `ls $SFT_CKPT/checkpoint-* | tail -1` 确认最终 step ckpt 路径（LF 默认按 step 分片）。
2. 若 LF 输出在 `checkpoint-N/` 下，应把 `MODEL_PATH` 指向该子目录或先合并到 `final/`。
3. ckpt 目录需含 `config.json` + tokenizer + `preprocessor_config.json` + safetensors。

---

## 5. 后续操作建议

### 5.1 立即可执行（无依赖）

| 优先级 | 动作 |
|---|---|
| ★★★ | 终止 `v19_clean` / `v19_no_autoterm` 进程，释放 2 个 GPU slot |
| ★★★ | slot-1 启动 **v20_winner**（最高 ROI 候选） |
| ★★★ | slot-2 启动 **v19_thr70_klhi**（隔离 sh 补丁贡献）；若 GPU 紧张可改 **v19_no_farm_dual** |
| ★★ | 保留 `v19_thr70_andgate` 再跑 1-2 个 ckpt 确认崩塌确认后再终止；保留 `v19_no_farm`、`v19_succ10` 跑满更多 val ckpt |

### 5.2 等 SFT 完成后启动

| 优先级 | 动作 |
|---|---|
| ★★ | 启动 **v20_sft_thr85_andgate** |
| ★ | 启动 **v20_sft_no_autoterm**（取决于 v20_sft_thr85_andgate 是否成功，若失败再决定是否跑） |

### 5.3 中期（取决于上述实验结果）

- 若 v20_winner 稳定 ≥ 0.88 → 作为 main candidate，进入更长训练 + 更多 val seeds 验证
- 若 v19_thr70_klhi 单独能修崩塌（且 yaml 合并无收益）→ 反过来证明 yaml 合并是冗余的，简化 v20 stack
- 若 SFT-warmstart 实验成功 → 主线切到 SFT→RL 两阶段，废弃 from-scratch SFT 级 thr 路线

---

## 6. 文件清单（本轮新建/修改）

### SFT 管线
- [data_gen/active_spatial_sft/convert_to_qwen25vl_sft.py](data_gen/active_spatial_sft/convert_to_qwen25vl_sft.py) (修改：+`--to_parquet`)
- [data_gen/active_spatial_sft/lf_dataset_info.json](data_gen/active_spatial_sft/lf_dataset_info.json) (新建)
- [data_gen/active_spatial_sft/lf_qwen25vl_3b_sft.yaml](data_gen/active_spatial_sft/lf_qwen25vl_3b_sft.yaml) (新建)
- [data_gen/active_spatial_sft/run_sft_qwen25vl_3b.sh](data_gen/active_spatial_sft/run_sft_qwen25vl_3b.sh) (新建, chmod +x)

### v19 / v20 实验
- [examples/train/active_spatial/env_config_v19_no_farm_dual.yaml](examples/train/active_spatial/env_config_v19_no_farm_dual.yaml) (新建)
- [examples/train/active_spatial/experiments/v19_no_farm_dual.sh](examples/train/active_spatial/experiments/v19_no_farm_dual.sh) (新建)
- [examples/train/active_spatial/experiments/v19_thr70_klhi.sh](examples/train/active_spatial/experiments/v19_thr70_klhi.sh) (新建)
- [examples/train/active_spatial/experiments/v20_winner.sh](examples/train/active_spatial/experiments/v20_winner.sh) (新建)
- [examples/train/active_spatial/experiments/v20_sft_thr85_andgate.sh](examples/train/active_spatial/experiments/v20_sft_thr85_andgate.sh) (新建)
- [examples/train/active_spatial/experiments/v20_sft_no_autoterm.sh](examples/train/active_spatial/experiments/v20_sft_no_autoterm.sh) (新建)

### 文档
- [docs/v19_v20_progress.md](docs/v19_v20_progress.md) (本文档)

---

## 7. 增量更新（本轮操作）

### 7.1 SFT 数据生成被 OOM 中断 → resume 续跑

**状态快照**
- 数据集：`data_gen/active_spatial_sft/output_100scenes_5k/`
  - 源样本：`sampled_5k.jsonl`（5000 条）
  - 已落盘：`sft_data.jsonl` = **1362 条**（处理过 source items `[0, 2185)`，部分因 `partial_success < 0.5` 被丢弃）
  - 末条 id：`sft_002079`
  - 渲染图：`images/`（11200 张）
- 中断原因：宿主进程被 OOM-killer 杀掉（用户确认与代码无关，是其他文件误删导致内存压力）。

**Resume 方案**
- 新脚本：[data_gen/active_spatial_sft/run_100scenes_5k_resume.sh](data_gen/active_spatial_sft/run_100scenes_5k_resume.sh)（chmod +x）
  - 复用 `sampled_5k.jsonl`、复用同一 `output_dir`、复用 GPU 4。
  - `--start_idx 2185` 从下一条源样本继续。
  - **关键**：`output_name` 改为 `sft_data_part2`（generator 用 `"w"` 打开 jsonl，必须用不同名避免覆盖 part1）。
  - 图片 ID `sft_NNNNNN_stepKK.jpg` 来自 `source_item_idx`，与 part1 天然不冲突。
- 启动：
  ```bash
  cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
    nohup bash data_gen/active_spatial_sft/run_100scenes_5k_resume.sh \
    > /tmp/sft_resume.log 2>&1 &
  ```
- 完成后合并：
  ```bash
  cat output_100scenes_5k/sft_data.jsonl output_100scenes_5k/sft_data_part2.jsonl \
      > output_100scenes_5k/sft_data_merged.jsonl
  ```
- **已启动**（2026-05-21）：PID 2761829，运行中，第一条样本 (item 2185) 已成功生成（score=0.9860）。

### 7.2 补 train-time 任务成功率（traj_success）

**问题**
- v19 之前的对比里只看了 `val-aux/active_spatial/traj_success/best@4/mean`（这就是 val task success rate）。
- 但 train 侧 wandb 完全没有任务成功率曲线：rollout (`gym_agent_loop_no_concat.py`) 每个 output 已经在 `reward_extra_info` 里 emit 了 `traj_success`，可是 [vagen/ray_trainer.py](vagen/ray_trainer.py) 的 `fit()` 在 train loop 没把它聚合成 metric。

**修复**
- 文件：[vagen/ray_trainer.py](vagen/ray_trainer.py)（约第 1474–1483 行，紧跟 `batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})` 之后插入）
- 同时覆盖 sync (`launch_reward_fn_async=False`) 与 async 两条路径（它们在 `compute_advantage` 之前汇合到同一处）。
- Diff（精简版）：
  ```python
  if reward_extra_infos_dict:
      batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

      # ── train-time task success rate (mirrors val-aux/.../traj_success) ──
      if "traj_success" in reward_extra_infos_dict and len(reward_extra_infos_dict["traj_success"]) > 0:
          _ts = np.asarray(reward_extra_infos_dict["traj_success"], dtype=np.float32)
          metrics["train/traj_success/mean"] = float(_ts.mean())
          metrics["train/traj_success/std"]  = float(_ts.std())
          metrics["train/traj_success/sum"]  = float(_ts.sum())
          metrics["train/traj_success/count"] = int(_ts.size)
  ```
- 新增 wandb / console metric key：
  - `train/traj_success/mean` — 当步 rollout group 内 task success 占比（与 val 端 `best@4` 不可直接比，分母为 n=1）
  - `train/traj_success/std`, `train/traj_success/sum`, `train/traj_success/count`
- 影响范围：
  - 仅对 **本次 patch 之后启动** 的训练生效。当前正在跑的 `v19_thr70_klhi` 是 patch 之前启动的，不会有新 metric；下一个起的 (`v20_winner` / `v19_thr70_andgate` 等) 会自动出现。
  - 零额外开销，无副作用（不修改 batch、advantage、loss）。

### 7.3 本轮新增文件清单（增量）

- [data_gen/active_spatial_sft/run_100scenes_5k_resume.sh](data_gen/active_spatial_sft/run_100scenes_5k_resume.sh) (新建, chmod +x) — SFT 续跑
- [vagen/ray_trainer.py](vagen/ray_trainer.py) (修改, +8 行) — train-time `traj_success` 聚合

---

## 8. v20_winner 进度复盘 & v21 候选实验设计

### 8.1 v20_winner 训练进展（截至本次复盘）

- 启动命令：`bash examples/train/active_spatial/run_experiment.sh examples/train/active_spatial/experiments/v20_winner.sh`
- yaml：`env_config_v19_no_farm_dual.yaml`（= no_farm 取消 farming + dual ori reward）
- 关键 sh 改动（相对 v19_no_farm 基线，全部"稳定性向"）：
  | 参数 | v19_no_farm | v20_winner | 作用 |
  |------|-------------|------------|------|
  | `ENTROPY_COEFF` | 0.001 | **0.002** | 抑制熵塌缩、保探索 |
  | `KL_LOSS_COEF` | 0.05 | **0.10** | 加大对 ref policy 的拉力，防漂移 |
  | `CRITIC_WARMUP` | 30 | **60** | 让 value 更稳后再用 advantage |
  | `CLIPRANGE_VALUE` | 0.8 | **0.5** | value loss 更保守，防 critic 跳变 |

- 跨实验 val 峰值对照（best@4 / tsucc_b4，按 step 0/20/40/...）：
  | 实验 | step=20 | step=40 | step=80 | step=120 | step=180 | 备注 |
  |------|---------|---------|---------|----------|----------|------|
  | v19_no_farm           | 39.41 / 0.703 | **47.45 / 0.880** | 44.55 / 0.819 | 39.93 / 0.731 | 23.61 / 0.422 | 峰值最高，崩盘最重 |
  | v19_thr70_andgate     | 41.10 / 0.749 | **46.59 / 0.859** | 43.42 / 0.789 | 31.04 / 0.567 | — | 早崩 (~step120) |
  | v19_dual              | 38.21 / 0.673 | **44.79 / 0.808** | 42.83 / 0.774 | 39.40 / 0.708 | — | 较稳但峰值低 |
  | v19_no_farm_dual      | **39.57 / 0.715** | — | — | — | — | 只有 2 个 ckpt |
  | **v20_winner**        | 40.86 / 0.733 | **44.95 / 0.819** | (still training) | — | — | 暂未崩，但峰值低于 v19_no_farm |

  **关键 finding**：所有能训得起来的配置都在 **step ≈ 40 出现 best@4 峰值**，之后开始衰减（衰减剧烈程度因配置而异）。这是横向最强的共同信号 → 当前 actor LR (`1e-6` 常数) 在 sweet spot 之后偏大。

- v20_winner 训练侧主要信号（前 ~50 step）：
  - `actor/entropy` 稳定在 0.44 ~ 0.57，未塌缩；
  - `response_length` 130 ~ 170，`response_length_clip_ratio` 0.15 ~ 0.27（偶有偏高，但不失控）；
  - `env_turns` 5.5 ~ 6.2（cap=12），探索深度合理；
  - `train/traj_success/mean`（新加的 metric）随训练上升 → patch 生效。

- 结论：v20_winner 的稳定性补丁有效（暂未观察到 v19_no_farm 那种 step120+ 的崩盘），但 **峰值 best@4 (44.95) 仍低于 v19_no_farm (47.45) 与 v19_thr70_andgate (46.59)**。怀疑 dual reward 与 no_farm 的奖励结构存在轻微冲突。

### 8.2 v21 候选实验设计（三路并行假设）

| 候选 | 攻击的问题 | 关键改动 | 期望结果 |
|------|------------|----------|----------|
| **v21_lrdecay** | "step40 峰值后衰减" 共同模式 | 在 v20_winner 之上加 `lr_scheduler_type=cosine`, `warmup=10%`, `min_lr_ratio=0.1`, `TOTAL_STEPS=500` | 峰值 step 略推后（~50-80），峰后衰减放缓 → best@4 峰值 ≥ v20_winner，并保持更久 |
| **v21_no_farm_only_stab** | "dual 是否值得？" 干净消融 | yaml 回到 `env_config_v19_no_farm.yaml`（去掉 dual），sh 保留 v20 的 4 个稳定性补丁 | 若 best@4 峰值回到 ≥ 0.86，说明 dual 拖低了峰值；否则说明 4 个 sh 补丁本身就压峰 |
| **v21_groupadv** | "best@4 vs mean@4 有 13pt gap 没被训练利用" | `N_TRAJECTORY=1→4`, `TRAIN_BATCH_SIZE=24→12`, `PPO_MINI_BATCH=12→8`；保持 PPO + no_concat_gae | 同 prompt 4 条 rollout 的方差进入 batch，advantage 估更稳，最优行为信号更强 → 峰值 best@4 提升 |

### 8.3 GRPO 可行性分析（v21_groupadv 附带）

verl 自带 GRPO (`compute_grpo_outcome_advantage`)：按 `non_tensor_batch["uid"]` 分组，对组内 `token_level_rewards.sum(-1)` 做 z-score，无 critic。理论上比 PPO 更适合"组内变异大"的设置。

但在 VAGEN-Lite 的 multi-turn no-concat 模式下直接切 GRPO 有 4 个需先验证的点：

1. **uid 语义**：no-concat 把同一轨迹的不同 turn 拆成不同 sample。若 `uid` 是 per-turn，GRPO 会把同轨迹的不同 turn 当组员归一，信号失真。需要确认 `uid` 是 per-trajectory（同 prompt 的 N 条轨迹各一个 uid，同轨迹的所有 turn 共享）。
2. **critic 关闭**：`need_critic` 在 `adv_estimator != gae` 时默认 disable，但 VAGEN-Lite 的 `no_concat_gae` 走的是另一条路径，切 GRPO 时需显式 `critic.enable=False`。
3. **`compute_value_mask` 旁路**：`ray_trainer.py:1524` 附近的 `compute_value_mask(batch)` 只在 `adv_estimator` 以 `no_concat_gae` 开头时触发，切 GRPO 会被绕过，需要确认 GRPO 分支的 `response_mask` 生效。
4. **token_level_rewards 内容**：当前 reward shaping（no_farm / dual）是 per-step 给，求和得到的轨迹总回报与 GRPO 期望兼容；但若 KL penalty 已经掺进 `token_level_rewards`，组内归一可能稀释主信号。

**结论**：GRPO 不作为 v21_groupadv 默认；默认走"PPO + no_concat_gae + N=4"安全路线。GRPO 配置以注释形式（`EXTRA_OVERRIDES`）放在 sh 末尾，待 1/2/3 的单元测试通过后，作为 `v21_groupadv_grpo` 独立跑。

### 8.4 工程改动

- [examples/train/active_spatial/run_experiment.sh](examples/train/active_spatial/run_experiment.sh) (修改, +1 行) — 在最终 hydra 命令尾追加 `${EXTRA_OVERRIDES:-} \`，允许实验 sh 通过 `export EXTRA_OVERRIDES=...` 注入额外 hydra override（用于 lr scheduler、GRPO 切换等"非通用变量"参数）。
- [examples/train/active_spatial/experiments/v21_lrdecay.sh](examples/train/active_spatial/experiments/v21_lrdecay.sh) (新建) — v20_winner + cosine lr decay。
- [examples/train/active_spatial/experiments/v21_no_farm_only_stab.sh](examples/train/active_spatial/experiments/v21_no_farm_only_stab.sh) (新建) — no_farm yaml + v20 的 4 个稳定性 sh 补丁，消融 dual。
- [examples/train/active_spatial/experiments/v21_groupadv.sh](examples/train/active_spatial/experiments/v21_groupadv.sh) (新建) — N_TRAJECTORY=4 + GRPO 可行性分析注释块。

### 8.5 推荐运行顺序

1. **先单独跑 v21_lrdecay**（只动 LR schedule，复用 v20_winner 全部，最低风险）→ 看 step 40~120 衰减是否真的能压住。
2. v21_lrdecay 跑过 step 80 后，并行启 **v21_no_farm_only_stab**（独立 yaml 路径，与 v21_lrdecay 不冲突）。
3. 资源允许时再启 **v21_groupadv**（N=4，rollout 成本最高，放最后）。
4. 都跑到 step 120 后看 best@4 / tsucc_b4 横向比较，决定 v22 的方向（lr×groupadv 复合、或换 GRPO 之类）。

---

## 9. v21 / v22 实验扩展与首次分析（2026-05-23 上午）

### 9.1 新增实验（v21_groupadv 之后追加 3 个 v22 + OOD 验证基础设施）

| 候选 | 攻击的问题 | 关键改动 |
|------|------------|----------|
| **v22_cosine_no_farm** | "cosine LR 在 no_farm 上是否更稳" | no_farm yaml + cosine LR (warmup 10%, min_lr 0.1) + OOD val + TOTAL_STEPS=500 |
| **v22_kl_ablation** | "KL=0.05 vs 0.10 能否再推峰值" | no_farm yaml + KL=0.05（v21_no_farm_stab 是 0.10），其他 3 个稳定性补丁不变 + OOD val |
| **v22_3scenes** | "单场景过拟合担忧" | 3 场景 (0267+0299+0342) 训练，752 task；OOD val + TEST_FREQ=30 + TOTAL_STEPS=1000 |

### 9.2 OOD 验证基础设施（Plan A）

- **数据**: [`val_ood_v1.jsonl`](data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl) — 19 个 OOD 任务来自 6 个未训练场景（0276/0367/0229/0240/0351/0266），任务类型分布与 in-domain 严格对齐：projective:8, equidistance:4, occlusion:4, size:3。
- **集成**: [`run_experiment.sh`](examples/train/active_spatial/run_experiment.sh) 增加 `OOD_VAL_JSONL` + `OOD_VAL_N_ENVS` 环境变量；val.yaml 自动追加第二个 env entry。
- **解析**: [`parse_val_ood.py`](examples/train/active_spatial/parse_val_ood.py) 按前 76 条 = in-domain，后 76 条 = OOD 切分。
- **开销**: 每次 val 增加 ~30s，0 训练侧影响。

### 9.3 3 场景课程数据集（Plan B）

- [`train_data_3scenes_v1.jsonl`](data_gen/active_spatial_pipeline/output_v2/train_data_3scenes_v1.jsonl): 771 行 = 752 训练 (259+258+235) + 19 in-domain val。
- [`env_config_v22_3scenes.yaml`](examples/train/active_spatial/env_config_v22_3scenes.yaml): 与 no_farm reward 一致；`train_size: 752, test_size: 19`。
- 训练场景与 OOD val 场景**不重叠**。

### 9.4 首次分析（截至 v21/v22 各跑 ~20-120 步）

| exp | step | ID b4 | ID m4 | OOD b4 | OOD m4 |
|---|---|---|---|---|---|
| v21_lrdecay | 20→120 | 0.947→0.789→0.895 | 0.553→0.540 | — | — |
| v21_no_farm_stab | 80 | **0.9474** | **0.6316** | — | — |
| v21_no_farm_stab | 100 | 0.6842（崩塌） | 0.5263 | — | — |
| v21_no_farm_stab | 120 | 0.8947（恢复） | 0.6053 | — | — |
| v21_groupadv | 20→40 | 0.8947→0.8421 | 0.5000 | — | — |
| v22_cosine_no_farm | 20→40 | 0.8947→0.8421 | 0.487→0.434 | 0.7368→0.8947 | 0.421→0.447 |
| v22_kl_ablation | 20 | 0.8947 | 0.5000 | 0.7895 | 0.4342 |
| v22_3scenes | 30 | 0.8421 | 0.5000 | **0.8947** | 0.3684 |

**初步结论**:
- v21_no_farm_stab step 80 达到 0.9474/0.6316，是当时所有实验的历史峰值。
- v21_no_farm_stab step 100 单步崩塌至 0.6842，step 120 恢复 → 推测振荡周期。
- v22_3scenes step 30 出现 OOD > ID（0.895 vs 0.842）→ 初步支持"多场景训练泛化更强"。
- v22_cosine_no_farm OOD 从 0.737→0.895 大幅上升 → 初步支持"模型在学通用导航"。

---

## 10. v21 / v22 扩展分析（更新版，2026-05-23 晚）

### 10.1 完整最新指标表

| exp | step | ID_b4 | ID_m4 | OOD_b4 | OOD_m4 |
|---|---|---|---|---|---|
| **v21_lrdecay** | 80 | 0.7895 | 0.5395 | — | — |
| | 120 | 0.8947 | 0.5395 | — | — |
| | **140** | **0.7368** | 0.5658 | — | — |
| **v21_no_farm_stab** | 80 | **0.9474** | **0.6316** | — | — |
| | 100 | 0.6842 | 0.5263 | — | — |
| | 120 | 0.8947 | 0.6053 | — | — |
| | 140 | 0.7895 | 0.4868 | — | — |
| | **160** | **0.8421** | **0.4868** | — | — |
| **v21_groupadv** | 60 | 0.8947 | 0.4737 | — | — |
| | 80 | 0.8421 | 0.5132 | — | — |
| | **100** | **0.8421** | **0.5789** ↑ | — | — |
| **v22_cosine_no_farm** | 40 | 0.8421 | 0.4342 | **0.8947** | 0.4474 |
| | 80 | 0.8421 | 0.5658 | 0.7368 | 0.3947 |
| | 120 | 0.8421 | 0.5263 | 0.6842 | 0.4079 |
| | **140** | **0.7895** | 0.5000 | **0.6316** | **0.2895** |
| **v22_kl_ablation** | 20 | 0.8947 | 0.5000 | 0.7895 | 0.4342 |
| | 40 | 0.8947 | 0.5395 | 0.8421 | 0.3553 |
| | **60** | **0.8947** | 0.4868 | **0.6842** | 0.3158 |
| **v22_3scenes** | 30 | 0.8421 | 0.5000 | **0.8947** | 0.3684 |
| | 60 | 0.8421 | 0.4737 | 0.8421 | 0.4079 |
| | 90 | 0.7895 | 0.4868 | 0.8421 | 0.4342 |
| | **120** | **0.7368** | 0.5263 | **0.7368** | **0.4605** |

### 10.2 上次待确认观察点的现状

| 待确认观察点（首次分析时） | 现在的答案 |
|---|---|
| v21_no_farm_stab step 140-160 会再崩塌还是恢复？ | **再次下降**（140=0.789, 160=0.842），振荡周期约 40 步，m4 整体下滑 |
| v22_cosine_no_farm step 60 OOD 是否稳定 ≥0.89？ | **否**，单调下滑 0.895→0.737→0.632，"OOD 上升=学到泛化"假设被否定 |
| v22_kl_ablation 低 KL 能否推过 0.9474？ | **否**，三个 ckpt 精确 0.8947，疑似"鞍点停滞" |
| v21_lrdecay 能否恢复 0.9474？ | **否**，step 140 进一步跌至 0.7368，cosine LR "底板效应"假设被推翻 |

### 10.3 新发现（更新分析）

**发现 1 — OOD 普遍随训练下降（关键负面信号）**

| exp | OOD 峰值 | 最新 OOD b4 |
|---|---|---|
| v22_cosine_no_farm | 0.8947 (s40) | 0.6316 (s140) |
| v22_kl_ablation | 0.8421 (s40) | 0.6842 (s60) |
| v22_3scenes | 0.8947 (s30) | 0.7368 (s120) |

→ 训练越深入，OOD 越退化，证实"过拟合到 in-domain"的隐忧；3 场景训练**未能**根本消除这一现象。

**发现 2 — v22_3scenes 的 b4↓ 与 m4↑ 矛盾**

step 30→120: b4 单调降 (0.842→0.737)，但 **OOD m4 从 0.368→0.460 持续上升**。说明 best-of-4 成功率下降，但每次 rollout 成功概率提高 → 训练在缩小 best/mean gap，但"运气上限"在下降。这是新的、未在 v21 系列出现的信号。

**发现 3 — v21_groupadv 是唯一 m4 正向上升的实验**

m4 轨迹：0.500→0.500→0.474→0.513→**0.579**（s100）。训练 tsucc=0.108（与 no_farm_stab 同级最高）。GroupAdvantage 通过同 prompt N=4 rollout 的相对优势计算，可能正在以最慢但最稳的方式提升平均成功率。**当前最有前景的实验**。

**发现 4 — v22_kl_ablation 的"鞍点停滞"现象**

val b4 在 step 20/40/60 精确 0.8947/0.8947/0.8947（标准差 0），但训练 tsucc=0.038（最低），entropy=0.579（中位）。KL=0.05 似乎把模型锁在固定策略上，既不崩塌也不改进。OOD 已在下降（0.842→0.684），说明虽然 ID 不变但模型仍在 in-domain 局部调整。

### 10.4 当前状态总览与去留决策

| 实验 | step | ID b4 趋势 | m4 趋势 | OOD 趋势 | 决策 |
|---|---|---|---|---|---|
| v21_lrdecay | 140/500 | ↓ 0.789→0.737 | 平 | n/a | ❌ **Kill**：3 次振荡均无法回到峰值；cosine LR 假设失败 |
| v21_no_farm_stab | 160/2000 | ↓ 振荡（0.947→0.842） | ↓ 0.632→0.487 | n/a | ⚠ **观察 step 200**，若再次出现 0.9474 则保留，否则 kill |
| v21_groupadv | 100/2000 | 稳 0.842-0.895 | ↑ 0.500→0.579 | n/a | ✅ **保留**：唯一正向趋势，N=4 信号需更长时间证实 |
| v22_cosine_no_farm | 140/500 | ↓ 0.842→0.789 | 平 | ↓ 0.895→0.632 | ❌ **Kill**：ID/OOD 双降，无任何优于 v21 的指标 |
| v22_kl_ablation | 60/2000 | 卡死 0.8947 | 平 | ↓ 0.842→0.684 | ⚠ **观察 step 100**：若 ID 仍卡死且 OOD 继续降，kill |
| v22_3scenes | 120/1000 | ↓ 0.842→0.737 | 平 | ↓ 0.895→0.737 | ⚠ **观察 step 180**：m4 ↑ 信号仍在；若 OOD 继续降则需要换更多场景（v23_10scenes） |

### 10.5 推荐 v23 候选实验（替换被 kill 的 slot）

| 候选 | 攻击的问题 | 关键改动 |
|------|------------|----------|
| **v23_groupadv_no_farm** | "GroupAdv + no_farm 组合是否更强" | v21_groupadv 的 N=4 + no_farm yaml + OOD val |
| **v23_groupadv_3scenes** | "GroupAdv + 多场景课程是否解决 OOD 退化" | v21_groupadv N=4 + 3 场景 env config + OOD val |
| **v23_entropy_bonus** | "提高 entropy 阻止过早专化" | no_farm + ENTROPY_COEFF=0.005 (vs 0.002) + OOD val |

---

## 11. v21 / v22 / v23 深度分析（2026-05-24 晚）

> 本节基于 6 个实验的 train.log 完整 val 曲线（通过脚本解析），覆盖 v21_no_farm_only_stab (step 279)、v21_groupadv (step 364)、v22_kl_ablation (step 300)、v22_3scenes (step 446)、v23_groupadv_3scenes (step 156)、v23_groupadv_no_farm (step 33，已死亡）。

### 11.1 完整 val 曲线（本次新增数据）

#### v21_no_farm_only_stab（gh108，step 279，无 OOD val）

| step | ID_b4 | ID_w4 | critic/score | 备注 |
|------|-------|-------|-------------|------|
| 20 | 0.652 | 0.187 | — | |
| 40 | 0.763 | 0.168 | | |
| 60 | 0.835 | 0.374 | | |
| 80 | **0.880** | 0.343 | | 全局峰值 |
| 100 | 0.646 | 0.385 | | 崩塌 |
| 120 | 0.847 | 0.307 | | 回弹 |
| 140 | 0.744 | 0.184 | | |
| 160 | 0.728 | 0.272 | | ← §10 决策点 |
| 180 | 0.527 | 0.052 | | |
| **200** | **0.350** | **0.008** | | ⚠ kill 触发 |
| 220 | 0.686 | 0.172 | | 短暂回弹 |
| 240 | 0.418 | 0.022 | | 再次下跌 |
| **260** | **0.227** | **0.005** | | 持续崩塌 |

- `critic/score`: 1.715 → 1.042（持续下降）；`resp_len`: 158.2 → 162.1（稳定）
- **结论**：step 200 b4=0.350 满足 §10 kill 触发条件，之后再无实质恢复。❌ 应在 step 200 时 Kill。

#### v21_groupadv（gh117，step 364，无 OOD val）

| step | ID_b4 | ID_w4 | 备注 |
|------|-------|-------|------|
| 20 | 0.794 | 0.209 | |
| 100 | 0.776 | 0.383 | ← §10 保留点 |
| 140 | 0.578 | 0.176 | 开始下滑 |
| 180 | **0.242** | 0.054 | 崩塌 |
| 260 | 0.627 | 0.358 | 部分回弹 |
| 280 | 0.625 | 0.358 | 峰值 |
| 340 | 0.348 | 0.268 | 再次下跌 |
| **360** | **0.259** | 0.075 | 低位震荡 |

- `critic/score`: 3.090 → 2.968（相对稳定）；`resp_len`: 145.6 → 130.0（缩短，策略固化信号）
- **结论**：step 180 崩塌至 0.242，部分回弹后 step 360 再次跌至 0.259。GroupAdv 单独在单一场景（no_farm_dual，259 envs）下无法阻止 PPO 崩塌。❌ Kill。

#### v22_kl_ablation（gh116，step 300，有 OOD val）

| step | ID_b4 | OOD_b4 | critic/score | resp_len | 备注 |
|------|-------|--------|-------------|---------|------|
| 20 | 0.777 | 0.714 | 3.142 | 157.7 | |
| 40 | 0.819 | 0.676 | | | 峰值 |
| 60 | 0.805 | 0.550 | | | ← §10 决策点 |
| 160 | 0.105 | 0.210 | | | 崩塌前兆 |
| **180** | **0.000** | **0.000** | | | **双路归零** |
| 240 | 0.715 | 0.225 | | | 短暂回弹 |
| **300** | **0.137** | **0.000** | **0.407** | **206.1** | 再次崩坏 |

- `critic/score`: 3.142 → **0.407**（灾难性下降）；`resp_len`: 157.7 → 206.1（+30%，格式崩塌信号）
- **结论**：KL=0.05 在 step 180 造成 ID/OOD 同时归零，是迄今最严重的崩塌。确认 **KL_LOSS_COEF≥0.10 是硬性底线**，0.05 不可用。❌ Kill（重要负向参照实验）。

#### v22_3scenes（gh119，step 446，有 OOD val）

| step | ID_b4 | OOD_b4 | critic/score | 备注 |
|------|-------|--------|-------------|------|
| 30 | 0.769 | 0.721 | 2.240 | |
| 120 | 0.677 | 0.659 | | ← §10 决策点 |
| 180 | 0.476 | 0.157 | | 阶段性崩塌 |
| **270** | **0.000** | **0.000** | | 归零 |
| 360 | 0.617 | **0.822** | | 强势回弹，OOD 创新高 |
| 420 | 0.610 | **0.827** | | OOD 维持高位 |
| 446 | — | — | 2.062 | critic 较稳 |

- `critic/score`: 2.240 → 2.062（仅小幅下降，与其他崩塌实验形成对比）；`resp_len`: 139.8 → 151.9（轻微上升，尚可接受）
- **结论**：极度震荡，但 critic 未崩。回弹后 OOD_b4 达 0.827 是所有实验迄今最高 OOD 值。"高方差但 critic 稳"的模式有别于其他实验的崩塌。⚠️ 继续观察至 step 480-500，关注是否能在 OOD>0.8 附近稳定。

#### v23_groupadv_3scenes（gh113，step 156，有 OOD val）—— **当前最优**

| step | ID_b4 | OOD_b4 | critic/score | 备注 |
|------|-------|--------|-------------|------|
| 30 | 0.695 | **0.789** | 2.171 | OOD 开局即高 |
| 60 | 0.784 | 0.717 | | |
| 90 | 0.714 | 0.738 | | |
| 120 | 0.646 | **0.870** | | OOD 创阶段新高 |
| **150** | **0.797** | **0.799** | **3.934** | ID 回升，critic ↑ |

与 v22_3scenes 同区间（step 30-150）对比：

| step | v22_3scenes OOD_b4 | v23_groupadv_3scenes OOD_b4 |
|------|-------------------|-----------------------------|
| 30 | 0.721 | **0.789** |
| 60 | 0.709 | **0.717** |
| 90 | 0.721 | **0.738** |
| 120 | 0.659 | **0.870** |
| 150 | 0.569 | **0.799** |

- `critic/score`: 2.171 → **3.934**（上升！是唯一正向增长的 critic 趋势）；`resp_len`: 147.7 → 161.3（正常范围）
- **结论**：GroupAdv (N=4) + 3scenes (752 envs) 完全满足预期。OOD_b4 全程 ≥0.71，无任何崩塌迹象；critic 持续增长是健康信号。✅ **最优实验，优先保留。**

#### v23_groupadv_no_farm（gh118，step 33，**已死亡**）

| step | ID_b4 | OOD_b4 | 备注 |
|------|-------|--------|------|
| 20 | 0.826 | 0.612 | 唯一数据点 |

- 最后日志时间：**2026-05-24 09:17**（距本次分析 >13 小时无更新）
- 进度条显示 `33/2000 [6:35:16<226:49:09]`——卡在 step 33 的 rollout 中途，进程已死（OOM 或节点故障）
- **结论**：无法评估假设（GroupAdv + no_farm 是否能叠加）。🔄 需重启，先确认 gh118 节点状态。

---

### 11.2 六实验横向对比

#### 11.2.1 峰值性能对比

| 实验 | 峰值 step | 峰值 ID_b4 | 峰值 OOD_b4 | 崩塌 step | 最终 step | 最终 ID_b4 | 状态 |
|------|-----------|-----------|------------|----------|----------|-----------|------|
| v21_no_farm_only_stab | 80 | **0.880** | — | ~200 | 279 | 0.227 | ❌ 崩塌 |
| v21_groupadv | 20 | 0.794 | — | ~180 | 364 | 0.259 | ❌ 崩塌 |
| v22_kl_ablation | 40 | 0.819 | 0.676 | **180** | 300 | 0.137 | ❌ 崩塌最重 |
| v22_3scenes | 30 | 0.769 | 0.827@420 | ~270 | 446 | 0.610 | ⚠️ 高振荡 |
| **v23_groupadv_3scenes** | 150 | **0.797** | **0.870** | 无 | **156** | **0.797** | ✅ 健康 |
| v23_groupadv_no_farm | 20 | 0.826 | 0.612 | — | 33 | — | 💀 进程死亡 |

#### 11.2.2 关键假设验证结果

| 假设 | 验证实验 | 结论 |
|------|---------|------|
| GroupAdv (N=4) 单独能防止单场景崩塌 | v21_groupadv | ❌ **证伪**：step 180 仍崩至 0.242 |
| KL=0.05 vs 0.10 能推高峰值 | v22_kl_ablation | ❌ **证伪**：KL=0.05 → 最严重崩塌（步 180 归零），确认 KL≥0.10 是底线 |
| GroupAdv + 3scenes 解决 OOD 退化 | v23_groupadv_3scenes | ✅ **初步成立**：OOD_b4 全程 ≥0.71，显著优于 v22_3scenes 同区间 |
| no_farm + stab 补丁能长期维持峰值 | v21_no_farm_only_stab | ❌ **证伪**：step 200 触发崩塌，步 260 跌至 0.227 |
| GroupAdv + no_farm 叠加两优点 | v23_groupadv_no_farm | ⏸ **无法评估**：进程死亡，仅 1 个数据点 |

---

### 11.3 PPO 崩塌机制总结

本轮实验验证了以下崩塌规律：

**崩塌的根本诱因**（按危险度排序）：
1. **KL_LOSS_COEF 过低**（最致命）：v22_kl_ablation 的 KL=0.05 → critic/score 3.142→0.407，步 180 完全归零。KL 约束不足时 actor 漂离 ref policy，advantage 分布偏移无法被 critic 跟上。
2. **环境多样性不足**：259 envs 的单场景实验（v21_no_farm_only_stab、v21_groupadv）无一幸免崩塌；752 envs 的 v22_3scenes 虽然震荡，但 critic 相对稳定（2.240→2.062）。
3. **GroupAdv 有必要但非充分条件**：N=4 组内归一化需配合足够的场景多样性才能有效。单场景 GroupAdv (v21_groupadv) 依然崩塌；多场景 GroupAdv (v23_groupadv_3scenes) 则稳定。

**崩塌前的 leading indicator**（按提前量排序）：
- `critic/score/mean` 下降（提前 ~40 step）：v22_kl_ablation 3.142→0.407 是最极端案例
- `response_length` 异常增长：v22_kl_ablation resp_len 157→206（+30%），对应格式探索崩塌
- `ID_w4`（worst@4）先于 `best@4` 跌至接近 0

---

### 11.4 更新后决策矩阵（2026-05-24）

| 实验 | 当前 step | 关键趋势 | 决策 | 释放 GPU |
|------|-----------|---------|------|---------|
| v21_no_farm_only_stab | 279/2000 | b4 崩至 0.227，critic ↓ | ❌ **Kill**（早在 step 200 应 kill） | gh108 |
| v21_groupadv | 364/2000 | step 180 崩至 0.242，360 再至 0.259 | ❌ **Kill** | gh117 |
| v22_kl_ablation | 300/2000 | KL=0.05 归零，critic=0.407，resp↑206 | ❌ **Kill**（负向参照留存记录） | gh116 |
| v22_3scenes | 446/1000 | 极振荡，OOD 回弹峰值 0.827，critic 稳 | ⚠️ **观察至 step 500**，关注 OOD 能否稳定 ≥0.7 | — |
| **v23_groupadv_3scenes** | **156/?** | **OOD_b4 全程≥0.71，critic ↑ 3.934** | ✅ **最优，重点保留** | — |
| v23_groupadv_no_farm | 33（死亡） | 进程死 >13h，无法评估 | 🔄 **重启**，需确认 gh118 | — |

---

### 11.5 推荐 v24 候选方向

基于本轮分析，**GroupAdv + 大场景多样性**是当前最有效的稳定组合。v24 应沿此方向继续推进：

| 候选 | 攻击的问题 | 关键改动 | 优先级 |
|------|------------|----------|--------|
| **重启 v23_groupadv_no_farm** | 假设尚未验证 | 原配置重启（gh118 确认可用） | ★★★ |
| **v24_groupadv_3scenes_longer** | v23_groupadv_3scenes 能否长期维持 OOD≥0.7 | 当前 exp 继续跑，观察 step 300-500 | ★★★（原地继续） |
| **v24_groupadv_6scenes** | 3→6 场景是否进一步提升 OOD | GroupAdv N=4 + 6 场景课程（~1500 envs）+ OOD val | ★★ |
| **v24_groupadv_3scenes_kl_hi** | KL=0.10 在 GroupAdv+3scenes 下是否过强 | v23 基础上 KL=0.15，观察是否推高峰值 | ★ |

---

## 12. v23 重启 & v24 新实验启动分析（2026-05-25）

> 本节记录 v23_groupadv_no_farm 重启、v23_groupadv_3scenes_7types 新实验、v24_groupadv_100scenes 新实验的数据准备、启动与早期分析。

### 12.1 数据准备修复（本轮核心工程工作）

#### 12.1.1 in-domain val 机制的根本缺陷与修复

**问题**：`run_experiment.sh` 使用 `seed=[train_size, train_size+test_size]` 索引 jsonl 末尾的 holdout 任务。早期的 7-type filtered jsonl 只有训练数据，`test_size` 的索引范围越界 → val 环境创建失败，in-domain val 一直是随机结果。

**修复**：将 holdout 任务（每 type 3 条，随机采样 seed=42）物理追加到 jsonl 末尾，同时更新 yaml 的 `train_size` / `test_size`。

#### 12.1.2 train_data_3scenes_7types.jsonl（3 场景，7 类型）

| 属性 | 值 |
|------|-----|
| 文件 | `data_gen/active_spatial_pipeline/output_v2/train_data_3scenes_7types.jsonl` |
| 总行数 | 497（479 训练 + 18 val） |
| Val 构成 | 6 有效 type × 3 = 18（centering=0，无数据） |
| 训练分布 | projective 134, absolute 129, occlusion 77, equidistance 67, delta 44, fov 28 |
| YAML | `env_config_v22_3scenes.yaml`（train_size: 479, test_size: 18） |

> **注意**：3 个训练场景中无 centering 数据（数据生成覆盖缺失），in-domain val 仅 6 类。

#### 12.1.3 train_100scenes_7types.jsonl（100 场景，7 类型）

| 属性 | 值 |
|------|-----|
| 文件 | `data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl` |
| 总行数 | 11211（11190 训练 + 21 val） |
| Val 构成 | 7 type × 3 = 21（seed=42 随机采样） |
| 训练分布（上限 2000/type）| absolute 2000, projective 2000, delta 2000, occlusion 1872, equidistance 1767, fov 1500, centering 51 |
| OOD 场景排除 | 6 个 OOD val 场景已从训练集移除 |
| YAML | `env_config_v24_100scenes.yaml`（train_size: 11190, test_size: 21） |

原始数据 absolute_positioning 占比 ~33%，已通过 2000/type 上限截断平衡。

---

### 12.2 三个实验配置

| 实验 | 配置文件 | 训练数据 | 超参关键点 | TOTAL_STEPS |
|------|---------|---------|-----------|-------------|
| **v23_groupadv_no_farm** | env_config_v19_no_farm.yaml | 1 场景, 259 envs | GroupAdv N=4, KL=0.10, critic_warmup=60 | 2000 |
| **v23_groupadv_3scenes_7types** | env_config_v22_3scenes.yaml | 3 场景, 479 envs, 7 类型 | 同上，TEST_FREQ=30 | 1000 |
| **v24_groupadv_100scenes** | env_config_v24_100scenes.yaml | 94 场景, 11190 envs, 7 类型 | 同上，TEST_FREQ=50 | 2000 |

所有实验共享 v20_winner 稳定性补丁：KL_LOSS_COEF=0.10, CRITIC_WARMUP=60, CLIPRANGE_VALUE=0.5, ENTROPY_COEFF=0.002, GRAD_CLIP=0.3。

---

### 12.3 早期训练指标（截至 2026-05-25）

| 实验 | 当前 step | traj_success | entropy | critic_score | resp_clip% | 步速 |
|------|---------|-------------|---------|-------------|-----------|------|
| v23_no_farm | 87 | 4.6–9.6% | **0.46** | 2.3–4.9 | **36.5%** | ~900s |
| v23_7types | 60 | 4.8% | 0.53 | 2.42 | 17.3% | ~893s |
| v24_100scenes | 58 | 4.1% | 0.53 | 2.08 | 19.6% | **604s** |

**关键发现**：
- v24_100scenes 步速快 33%（604s vs 900s），归因于更丰富的训练分布使 rollout 更多样，平均 episode 更短。
- v23_no_farm response_clip_ratio 高达 36.5%（vs 其他 ~18%），说明单场景实验中模型倾向生成超长轨迹（撞 384 token 上限），与之前 v19/v21 系列单场景实验的 farming 信号一致。
- v23_no_farm entropy=0.46 已是三组最低，提示策略固化速度更快。

---

### 12.4 Val 曲线（截至 2026-05-25）

| 实验 | step | ID_b4 | ID_m4 | OOD_b4 | OOD_m4 |
|------|------|-------|-------|--------|--------|
| v23_no_farm | 20 | 0.842 | 0.474 | 0.789 | 0.316 |
| v23_no_farm | 40 | 0.789 | 0.434 | **0.947** | 0.395 |
| v23_no_farm | 60 | 0.842 | 0.553 | 0.895 | 0.487 |
| v23_no_farm | 80 | **0.895** | **0.658** | 0.737 | 0.408 |
| v23_7types | 30 | 0.722 | 0.375 | 0.789 | 0.368 |
| v23_7types | 60 | 0.722 | 0.458 | 0.737 | 0.316 |
| v24_100scenes | 50 | 0.762 | 0.381 | 0.789 | 0.382 |

---

### 12.5 综合分析

#### v23_groupadv_no_farm（step 87）

- OOD_b4 在 step 40 峰值 0.947，到 step 80 已回落至 0.737——**与 v19/v21 系列单场景崩塌前信号高度吻合**。
- entropy 最低（0.46），response_clip 最高（36.5%）→ 策略正在固化 + farming 入侵初期信号。
- ID_b4 step 80 仍为 0.895，ID_m4 上升到 0.658，**短期 in-domain 仍有学习**；但以过往单场景实验的规律（v21_groupadv 在 step 180 崩至 0.242），**风险期约在 step 120-200**。
- 建议继续观察至 step 120，若 OOD_b4 < 0.6 或 entropy 跌至 0.35 以下则考虑 kill。

#### v23_groupadv_3scenes_7types（step 60）

- OOD_b4 从 step 30 的 0.789 降至 step 60 的 0.737，早期轻微退化但幅度小于 v22_cosine_no_farm（同区间 0.895→0.737）。
- ID_b4 flat at 0.722，critic warmup 刚完成（step 60），**后续可能出现上升拐点**。
- 7 类型 + 3 场景 = 479 envs，环境多样性介于 v23_no_farm（259）和 v24（11190）之间。
- 需等待 step 90-120 才能判断 GroupAdv + 7types 是否有独立贡献。

#### v24_groupadv_100scenes（step 58）

- 仅 1 个 val 检查点（step 50），OOD_b4=0.789，稳定无崩迹象。
- 步速最快、训练多样性最大、entropy 尚高（0.53）——**当前最稳健的实验**。
- critic_score 最低（2.08）是因为 11190 任务的多样性使 value 收敛更慢，属正常现象（对比 v23_no_farm 单场景 critic_score 飙升到 4.9）。
- centering 类型在训练数据中仅 51 条（受限于数据生成覆盖），是该实验的潜在短板。

---

### 12.6 v23_groupadv_3scenes（原版）的最新状态

> v23_groupadv_3scenes（使用旧的 `train_data_3scenes_v1.jsonl`，752 envs，6 类型）已停止于 step 156，数据见 §11.1。本轮启动的 **v23_groupadv_3scenes_7types** 是其 7 类型升级版，使用新的修复 jsonl（479 envs）。

---

### 12.7 当前实验状态总览（2026-05-25）

| 实验 | step | 趋势 | 主要风险 | 决策 |
|------|------|------|---------|------|
| **v24_groupadv_100scenes** | 58/2000 | 稳定，步速快 | 太早判断 | ✅ **重点保留，等 step 100-200** |
| **v23_groupadv_no_farm** | 87/2000 | OOD 已从峰值回落，entropy 低 | 单场景 farming/崩塌（step 120-200） | ⚠️ **观察至 step 120**，OOD<0.6 则 kill |
| **v23_groupadv_3scenes_7types** | 60/1000 | 平稳，OOD 轻微退化 | 太早判断，7type 覆盖窗口太短 | ⚠️ **等 step 90-120** 再判断 |

---

### 12.8 下一步计划（2026-05-25 时设定，已被 §13 更新覆盖）

- **step 90-120**：三个实验均出 val 结果，做横向对比（与 v23_groupadv_3scenes§11.1 同区间对比）。
- **重点关注**：v24 在 step 100 时的 OOD_b4 能否维持 ≥0.75（若是，则成为迄今最优的单实验配置）。
- **若 v23_no_farm 在 step 120 前 OOD 崩塌**：进一步证实"GroupAdv 需要多场景才能防崩塌"的假设，可考虑释放 slot 启动 v24_groupadv_6scenes 或增加 100scenes 实验步数。
- **7 类型数据集的价值**：v23_7types 和 v24 都引入了 centering 类型（v24 有 51 条 centering 训练数据），若 OOD val 包含 centering 任务则能直接测泛化；当前 `val_ood_v1.jsonl` 不含 centering，后续可考虑补充。

---

## 13. v23/v24 完整分析与 v25 候选方向（2026-05-27）

> 本节记录 v23_groupadv_no_farm（第三次启动）、v23_groupadv_3scenes_7types、v24_groupadv_100scenes 的完整 val 曲线分析与关键结论。

### 13.1 实验状态总览（2026-05-27）

| 实验 | 日志最终更新 | 最终 step | 状态 | 结论 |
|------|------------|---------|------|------|
| v23_groupadv_no_farm | 2026-05-26 09:32 | 120 | 💀 **进程死亡**（第三次，非训练崩塌） | 未崩塌，数据截断 |
| v23_groupadv_3scenes | 2026-05-25 01:17 | 180 | ✅ **正常终止**（entropy 爆炸后自然完成） | 已崩塌，数据完整 |
| v23_groupadv_3scenes_7types | 2026-05-27 01:19 | 180+ | ⚠️ **仍在运行**（entropy 爆炸，生成乱码） | **应立即 kill** |
| v24_groupadv_100scenes | 2026-05-27 00:35 | 150+ | ⚠️ **仍在运行**（entropy 爆炸，生成乱码） | **应立即 kill** |

> v23_groupadv_3scenes_7types 和 v24_groupadv_100scenes 的当前 rollout 日志均显示模型生成混合语言、符号乱码等严重格式崩溃特征，继续跑无意义。

---

### 13.2 完整 Val 指标表

#### v23_groupadv_no_farm（单场景 259 envs，第三次重启）

| step | entropy | train_succ% | train_rew | resp_clip% | ID_b4  | ID_m4  | OOD_b4 | OOD_m4 |
|------|---------|------------|-----------|-----------|--------|--------|--------|--------|
| 20   | 0.529   | 3.8        | 1.92      | 19.5      | 0.717  | 0.474  | 0.655  | 0.316  |
| 40   | 0.511   | 4.0        | 2.01      | 20.6      | 0.676  | 0.434  | 0.771  | 0.395  |
| 60   | 0.522   | 3.7        | 1.85      | 19.3      | 0.778  | 0.553  | 0.773  | 0.487  |
| 80   | 0.466   | 4.0        | 1.99      | **31.5**  | **0.819** | **0.658** | 0.623 | 0.408 |
| 100  | 0.594   | 5.2        | 2.61      | 24.8      | 0.685  | 0.500  | **0.828** | **0.553** |
| 120  | 0.517   | 5.6        | 2.81      | **37.4**  | 0.771  | 0.618  | 0.768  | 0.513  |

- entropy 全程稳定 0.47-0.59，无爆炸迹象。
- resp_clip 在 step 80 跳升至 31.5%，step 120 达 37.4% ← **farming 建立信号**。
- train_rew 从 1.92 缓升至 2.81，正向学习信号持续。
- ID_b4 峰值 0.819 @ step 80；OOD_b4 峰值 0.828 @ step 100。
- **进程死亡原因**：非训练崩塌（entropy 正常），疑似节点 OOM 或调度中断（v23_no_farm 已连续第三次被进程杀死而非训练失败）。

#### v23_groupadv_3scenes（3 场景 752 envs）

| step | entropy | train_succ% | train_rew | ID_b4  | ID_m4  | OOD_b4  | OOD_m4 |
|------|---------|------------|-----------|--------|--------|---------|--------|
| 30   | 0.500   | 4.4        | 2.22      | 0.695  | 0.421  | **0.789** | 0.421 |
| 60   | 0.549   | 3.2        | 1.62      | 0.784  | 0.513  | 0.717   | 0.461  |
| 90   | 0.542   | 9.7        | **4.92**  | 0.714  | 0.513  | 0.738   | 0.487  |
| 120  | 0.968 ↑ | **11.0**   | **5.55**  | 0.646  | 0.447  | **0.870** | **0.592** |
| 150  | 1.038   | 4.9        | 2.46      | **0.797** | **0.540** | 0.799 | 0.500 |
| 180  | **3.424** ❌ | 2.6  | 1.25      | 0.489  | 0.250  | **0.790** | 0.447 |

- Step 90-120 出现**奖励峰值期**：train_succ 9.7→11%，train_rew 4.92→5.55（疑似 reward hack）。
- Step 120 起 entropy 快速上升（0.97→1.04→3.42），与训练奖励崩塌同步。
- **关键异象**：step 180 entropy=3.42（策略崩溃），ID_b4 跌至 0.489，但 OOD_b4 仍维持 **0.790**！说明 OOD 场景对 farming/shortcut 的免疫力更强。

#### v23_groupadv_3scenes_7types（3 场景 479 envs，7 类型）

| step | entropy | train_succ% | train_rew | ID_b4  | ID_m4  | OOD_b4  | OOD_m4 |
|------|---------|------------|-----------|--------|--------|---------|--------|
| 30   | 0.502   | 4.4        | 2.20      | 0.642  | 0.375  | 0.658   | 0.368  |
| 60   | 0.534   | 4.8        | 2.42      | 0.666  | 0.458  | 0.612   | 0.316  |
| 90   | 0.549   | 7.0        | **3.53**  | **0.750** | **0.556** | **0.855** | **0.566** |
| 120  | 1.858 ↑ | 6.1        | 3.06      | 0.594  | 0.444  | 0.823   | 0.487  |
| 150  | 1.303   | 6.6        | 3.30      | 0.641  | 0.444  | 0.762   | 0.461  |
| 180  | **2.214** ❌ | 6.0  | 2.97      | 0.459  | 0.264  | 0.567   | 0.316  |

- 峰值在 step 90：ID_b4=0.750，OOD_b4=**0.855**。
- Step 120 起 entropy 爆炸（1.86），ID 崩塌，OOD_b4 从 0.855 退化至 0.567（相比 v23_3scenes 的 OOD 韧性更弱）。
- 7 类型覆盖 vs 6 类型：OOD 峰值略高（0.855 vs 0.870），但 entropy 爆炸更早（step 120 vs step 180）且 OOD 最终退化更严重（0.567 vs 0.790）。

#### v24_groupadv_100scenes（94 场景 11190 envs）

| step | entropy | train_succ% | train_rew | ID_b4  | ID_m4  | OOD_b4  | OOD_m4 |
|------|---------|------------|-----------|--------|--------|---------|--------|
| 50   | 0.518   | 3.5        | 1.73      | 0.671  | 0.381  | 0.679   | 0.382  |
| 100  | 0.718   | 4.9        | 2.46      | **0.776** | **0.524** | **0.717** | **0.421** |
| 150  | **2.114** ❌ | 3.6 | 1.75     | 0.402  | 0.226  | **0.709** | 0.421  |

- 仅 3 个 val 检查点，但信号清晰：step 100→150 entropy 从 0.718 直接爆炸至 2.114。
- ID_b4 从 0.776 崩至 0.402（-48%）；OOD_b4 仅从 0.717 降至 0.709（-1%）——**OOD 韧性极强**。
- 步速 604s/step（最快），这意味着同样的 wall time 内它的 step 数最少，在 step 150 就到 entropy 爆炸是偏早的。

---

### 13.3 四实验横向对比

#### 13.3.1 峰值性能对比

| 实验 | envs | 峰值 step | 峰值 ID_b4 | 峰值 OOD_b4 | entropy 爆炸 step | 最终 step 时 OOD_b4 | 状态 |
|------|------|-----------|-----------|------------|-----------------|-------------------|------|
| v23_no_farm | 259 | 80/100 | 0.819/0.828 | **0.828** | **无**（至 120） | — | 💀 进程死亡 |
| v23_3scenes | 752 | 120/150 | 0.797 | 0.870 | ~step 120 | **0.790**（step 180） | 完结 |
| v23_3scenes_7types | 479 | 90 | 0.750 | 0.855 | ~step 120 | 0.567（step 180） | ⚠️ kill |
| v24_100scenes | 11190 | 100 | 0.776 | 0.717 | ~step 150 | **0.709**（step 150） | ⚠️ kill |

#### 13.3.2 关键假设验证结果（v23/v24 轮次）

| 假设 | 验证实验 | 结论 |
|------|---------|------|
| GroupAdv + 多场景（3→94）越多越稳定 | v23_3scenes vs v24_100scenes | ❌ **部分证伪**：v24 entropy 爆炸更早（step 150 vs 180），更多场景≠更晚崩塌 |
| GroupAdv + 7 类型 vs 6 类型更泛化 | v23_3scenes vs v23_3scenes_7types | ❌ **反效果**：7 类型 entropy 更早爆炸，OOD 最终更低（0.567 vs 0.790） |
| GroupAdv + no_farm 单场景可避免 farming | v23_no_farm（step 120） | ⏸ **未验证**：resp_clip 在上升（37.4%），但崩塌尚未发生；进程死亡截断 |
| OOD 泛化在策略崩溃时仍保持韧性 | v23_3scenes（step 180），v24（step 150） | ✅ **强烈成立**：OOD_b4 在 ID_b4 崩 50% 后几乎不跌，OOD 场景对 shortcut 免疫 |

---

### 13.4 Entropy 爆炸机制深度分析

本轮实验揭示了 **GroupAdv + multi-scene** 配置下 entropy 爆炸的统一模式：

**阶段 1（step 0-80）：正常探索**  
entropy ~0.50，train_rew 缓慢上升，ID/OOD 指标稳步改善。

**阶段 2（step 80-120）：奖励峰值期（reward hack 信号）**  
train_rew 大幅跳升（如 v23_3scenes: 1.62 → 5.55），train_succ% 同步上升（3.2% → 11%）。这个 spike 表面上是"学到了"，但 advantage 分布此时已高度偏斜——model 找到了特定 trick（可能是利用 near_success_bonus 或格式奖励），使 GroupAdv 组内的高/低轨迹方差急剧扩大。

**阶段 3（step 100-150）：entropy 爆炸**  
entropy 从 0.5 突然跳至 1.0-2.2+，critic/score 先涨后崩，ID_b4 急跌。模型在 GroupAdv 的组内归一化下，一旦某些轨迹的回报异常高，gradient 被放大，导致 actor 偏离 ref policy，KL 约束（0.10）无法完全拦截。

**阶段 4（step 150-180）：策略崩溃**  
entropy 3.4+，生成乱码，格式完全失效。ID 任务成功率归零。但 OOD 场景由于没有被"memorized"的 shortcut 路径，反而能触发模型残余的通用导航策略。

**关键 leading indicators（优先级排序）**：
1. `train_rew` 急剧上升（奖励峰值）>奖励后期下跌 — 最早出现（提前 ~30-40 step）
2. `entropy` 超过 0.80 — 爆炸的直接前兆（提前 ~20-30 step）
3. `resp_clip_ratio` 持续上升超过 30% — farming 加剧信号（提前 ~20 step）
4. `ID_b4 下降而 OOD_b4 上升` — shortcut/memorization 信号（提前 ~10-20 step）

**修复方向**（优先级排序）：
1. **更激进的 KL 约束**：当前 0.10 不足以拦截 GroupAdv 放大的梯度。候选 0.20-0.30。
2. **entropy 正则化加强**：当前 ENTROPY_COEFF=0.002；可提升至 0.005-0.010。
3. **LR decay 在奖励峰值后**：cosine schedule 使 step 80+ 的 LR 降低，防止 actor 过激更新。
4. **奖励峰值 clip**：对 GroupAdv 组内高离群值（>2σ）做截断，防止异常高回报放大梯度。

---

### 13.5 OOD 韧性的理论解释

**现象**：v23_3scenes 在 entropy=3.42（策略基本随机）时，OOD_b4 仍为 0.790。

**解释**：
- In-domain 崩塌 = 模型学到了场景特定 shortcut（记住了起点→目标的特定路径），这些 shortcut 在 entropy 爆炸时被覆盖掉，导致 ID 失效。
- OOD 场景的 shortcut 从未被学到（训练中不出现），所以模型只能依赖通用导航能力。这个通用能力经过 180 步训练实际上保留得很好。
- **启示**：OOD_b4 是比 ID_b4 更稳健的训练信号。**训练早期（step 60 前）就看 OOD_b4 的趋势可能比 ID_b4 更有价值**。

---

### 13.6 更新后状态总览与决策矩阵（2026-05-27）

| 实验 | 最终 step | 峰值 ID_b4 | 峰值 OOD_b4 | 熵爆炸 | 结论 / 动作 |
|------|---------|-----------|------------|------|-----------|
| v23_no_farm | 120（死亡） | 0.819 | 0.828 | ❌ 无 | 💀 第三次被进程杀死，需 stable 环境重启 |
| v23_3scenes | 180（完结） | 0.797 | 0.870 | step ~120 | 数据已完整，OOD 韧性最强 |
| v23_3scenes_7types | 180（运行中） | 0.750 | 0.855 | step ~120 | ❌ **立即 kill**，entropy=2.21，乱码生成 |
| v24_100scenes | 150（运行中） | 0.776 | 0.717 | step ~150 | ❌ **立即 kill**，entropy=2.11，乱码生成 |

---

### 13.7 v25 候选实验设计

基于以上分析，v25 的核心问题是：**如何在 GroupAdv + 多场景下防止 step 100-150 的 entropy 爆炸**。

| 候选 | 攻击的问题 | 关键改动 | 期望 | 优先级 |
|------|------------|---------|------|--------|
| **v25_groupadv_100scenes_klhi** | KL=0.10 不足以拦截 GroupAdv 梯度放大 | v24 基础上 KL_LOSS_COEF 0.10→**0.20**，entropy_coeff 0.002→**0.005** | entropy 爆炸被推迟至 step 200+，峰值 OOD_b4 ≥0.75 | ★★★ |
| **v25_groupadv_100scenes_cosine** | step 80+ actor LR 仍是 1e-6，过激更新 | v24 + cosine LR decay（warmup 10%, min_lr 0.1, total_steps=500） | 奖励峰值后 update 幅度下降，entropy 稳定 | ★★★ |
| **v25_no_farm_100scenes** | no_farm 在单场景下稳定（step 120 entropy=0.52），是否在多场景也有效 | v24 的 100 场景 + v23_no_farm 的 no_farm yaml（format_reward=0, step_penalty=-0.02），N=4 | resp_clip 降低，entropy 稳定，OOD_b4 ≥0.75 | ★★ |
| **v25_groupadv_3scenes_earlyckpt** | v23_3scenes step 120 峰值 OOD_b4=0.870，可否以此为起点 SFT？ | 直接使用 v23_3scenes step 120 ckpt 做下游 SFT 或 BC，而非继续 RL | 利用 OOD 峰值权重做迁移，跳过 entropy 爆炸阶段 | ★★ |

**推荐启动顺序**：
1. kill v23_3scenes_7types 和 v24_100scenes，释放 GPU slot
2. 并行启动 v25_groupadv_100scenes_klhi（最直接修复）+ v25_groupadv_100scenes_cosine（正交方向）
3. 若 v25_klhi 在 step 150 的 OOD_b4 ≥ 0.75 且 entropy < 1.0，则为 v26 的 main candidate
4. v25_no_farm_100scenes 可在资源允许时作为第三个并行实验（验证 no_farm 配方的可迁移性）


---

## 14. v25 完整分析与 v26 实验设计（2026-05-29）

> 本节记录 v25_groupadv_100scenes_cosine 与 v25_groupadv_100scenes_klhi 的完整数据，分析失败原因并设计 v26 修复方向。

### 14.1 实验总览（截至 2026-05-29）

| 实验 | 配置关键点 | 最新 step | 状态 |
|------|-----------|---------|------|
| v25_groupadv_100scenes_cosine | KL=0.10, ent=0.002, LR=1e-6, cosine(warmup=10%, min=0.1, total=500) | 249/500 | 💀 **已崩塌**（stalled @ step 249，val 在 step 200 已崩） |
| v25_groupadv_100scenes_klhi | KL=0.20, ent=0.005, LR=1e-6（常数）, total=2000 | 317/2000 | ⚠️ **运行中**（无爆炸崩塌，但 mean@4 持续下滑） |

---

### 14.2 v25_groupadv_100scenes_cosine：完整 Val + Train 指标

#### 14.2.1 Val 曲线

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 50   | 0.8095 | 0.5357 | 0.2381 | 0.7895 | 0.3158 | 0.0526 |
| 100  | 0.7619 | 0.5595 | 0.3333 | 0.8421 | 0.5263 | 0.2632 |
| **150** | **0.8571** | **0.6548** | 0.3810 | 0.7895 | 0.4737 | 0.1579 |
| **200** | **0.2381** | **0.0952** | 0.0000 | **0.1053** | **0.0263** | 0.0000 |

峰值 step 150（ID_b4=0.857），step 200 彻底崩塌（ID -72%，OOD -87%）。

#### 14.2.2 Train 指标（按 40 步采样）

| step | entropy | t_succ% | crit_sc | resp_clip | grad_norm | actor_lr |
|------|---------|---------|---------|-----------|-----------|---------|
| 1-40 | 0.499–0.512 | 4.1–4.5 | 2.0–2.3 | ~0.21 | — | 0.00e+00 |
| 80   | 0.481 | 7.77 | 3.937 | 0.231 | 5.826 | **4.00e-07** |
| 120  | 0.580 | 6.25 | 3.156 | 0.180 | 5.170 | **9.99e-07** |
| **160** | **1.114** | 2.83 | 1.383 | 0.315 | 3.384 | **9.73e-07** |
| 200  | 2.881 | 0.97 | 0.392 | 0.110 | 3.222 | 9.14e-07 |
| 240  | 3.514 | 1.03 | 0.432 | 0.127 | 4.652 | 8.27e-07 |

#### 14.2.3 根本原因分析

**Cosine LR 未能防崩塌的机制**：

```
设计预期：
  warmup_steps_ratio=0.1 × TOTAL_STEPS=500 = 50 步 warmup → 步 50 到达 peak LR=1e-6
  → cosine 从步 50 开始衰减，步 80-120（奖励峰值期）LR 已显著下降 → 保护崩塌窗口

实际发生：
  critic_warmup=60（步 1-60 actor LR=0），消耗了全部 50 步 warmup budget
  → 步 60 之后 actor LR 才开始从 0 ramping
  → 步 80: LR=4.00e-07（warmup 仍在进行）
  → 步 120: LR=9.99e-07（峰值，此时 warmup 才刚结束）
  → 步 160: LR=9.73e-07（cosine decay 极慢，衰减仅 ~3%）
  → 步 160 entropy 爆炸时 LR 仍接近峰值，保护完全失效
```

**结论**：`cosine_lr + critic_warmup` 存在根本冲突。`lr_warmup_steps_ratio` 基于总步数计算，但实际 warmup 从 `critic_warmup` 结束后才生效，导致 cosine 调度整体延迟，在关键窗口（步 80-150）LR 仍处于 ~1e-6 的高位。

---

### 14.3 v25_groupadv_100scenes_klhi：完整 Val + Train 指标

#### 14.3.1 Val 曲线

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 50   | 0.7143 | 0.4762 | 0.0952 | 0.7895 | 0.4079 | 0.1053 |
| **100** | 0.8095 | 0.4643 | 0.1429 | **0.9474** | 0.4474 | 0.1579 |
| 150  | 0.7619 | **0.5000** | 0.2381 | 0.8947 | **0.5000** | 0.1579 |
| 200  | 0.7619 | 0.4643 | 0.0952 | 0.7895 | 0.3816 | 0.0526 |
| 250  | 0.7143 | 0.3333 | 0.0476 | 0.5789 | 0.2368 | 0.0000 |
| 300  | 0.6667 | 0.2381 | 0.0000 | 0.6842 | 0.3158 | 0.0000 |

**OOD_b4=0.9474（步 100）是迄今历史最高 OOD 指标**。无灾难性崩塌，但 mean@4 在步 150 后持续下滑。

#### 14.3.2 Train 指标（按 40 步采样）

| step | entropy | t_succ% | crit_sc | resp_clip | grad_norm | actor_lr |
|------|---------|---------|---------|-----------|-----------|---------|
| 1-40 | 0.521–0.551 | 4.1–4.4 | 2.0–2.2 | ~0.19 | — | 0.00e+00 |
| 80   | 0.585 | 8.33 | 4.229 | 0.255 | 5.267 | 1.00e-06 |
| 120  | 0.669 | 3.96 | 1.988 | 0.190 | 4.626 | 1.00e-06 |
| 160  | 0.916 | 2.33 | 1.144 | 0.153 | 4.271 | 1.00e-06 |
| 200  | 0.845 | 4.46 | 2.221 | 0.259 | 5.392 | 1.00e-06 |
| 240  | 1.282 | 1.83 | 0.874 | 0.159 | 5.432 | 1.00e-06 |
| 280  | 1.754 | 5.21 | 2.574 | 0.096 | 4.723 | 1.00e-06 |

entropy 缓慢爬升（0.55→1.75，步 1-280），无爆炸式跳升。critic/score 在步 80 达到峰值 4.23 后持续下降，说明 policy 正在从 in-domain shortcut 中"收益递减"。

#### 14.3.3 mean@4 持续下滑分析

**现象**：best@4（b4）维持 0.67-0.81，但 mean@4（m4）从步 150 的 0.50 降至步 300 的 0.24（ID）。

**机制**：高 b@4 + 低 m@4 = 策略对部分 seed 成功概率大幅分化。模型在某些初始状态下能稳定成功（形成 shortcut），但其他 seed 的成功率趋近于 0（ID_w4 在步 300 = 0.000）。KL=0.20 拦住了 entropy 爆炸，但无法阻止这种"赢家通吃"式的 advantage 分化。

---

### 14.4 v24 vs v25 峰值对比

| 实验 | 峰值 ID_b4 | @step | 峰值 OOD_b4 | @step | 峰值 ID_m4 | entropy 崩 | 状态 |
|------|-----------|-------|------------|-------|-----------|-----------|------|
| v24_groupadv_100scenes | 0.857 | 100 | 0.842 | 100 | 0.524 | step ~150 | ❌ 崩塌 |
| v25_cosine | 0.857 | 150 | 0.842 | 100 | **0.655** | step ~160 | ❌ 崩塌 |
| **v25_klhi** | 0.810 | 100 | **0.947** | 100 | 0.500 | 无（step 317+） | ⚠️ 运行中 |

- v25_cosine 将崩塌推迟了约 10 步（150 vs 100 峰值），但无本质区别。
- **v25_klhi 是首个无灾难性崩塌的 100scenes 实验**；OOD_b4=0.947 是历史最高。
- v25_klhi ID_m4 下滑是当前主要问题，但不同于 entropy 爆炸型崩塌，可能通过 LR 收敛修复。

---

### 14.5 假设验证结果（v25 轮次）

| 假设 | 验证实验 | 结论 |
|------|---------|------|
| Cosine LR + warmup 能防止步 80-150 的 entropy 爆炸 | v25_cosine | ❌ **证伪**：cosine warmup 被 critic_warmup=60 吞噬，LR 在步 160 崩塌时仍为 9.73e-07 |
| KL=0.20 + entropy=0.005 能防崩塌 | v25_klhi（步 317） | ✅ **初步成立**：无 entropy 爆炸，OOD 历史最高；mean@4 下滑是另一问题 |
| critic_warmup 与 lr_warmup_steps_ratio 存在冲突 | v25_cosine LR log | ✅ **证实**：actor_lr 在步 80 仅 4e-7，步 120 才达峰值，cosine 保护窗口完全错位 |

---

### 14.6 v26 实验设计（2026-05-29 创建）

基于 v25 分析，v26 的目标是解决 v25_klhi 的 mean@4 下滑问题，同时验证 cosine LR 的正确使用方式。

#### 14.6.1 v26_klhi_lrdecay.sh（cosine + 无 warmup）

| 参数 | 值 |
|------|-----|
| KL_LOSS_COEF | 0.20 |
| ENTROPY_COEFF | 0.005 |
| ACTOR_LR | 1e-6 |
| TOTAL_STEPS | 2000 |
| lr_scheduler_type | cosine |
| lr_warmup_steps_ratio | **0**（无 warmup，LR 从步 1 就开始衰减） |
| min_lr_ratio | 0.1 |

**设计理由**：v25_cosine 失败是因为 warmup 被 critic_warmup 吞噬。设置 `warmup_steps_ratio=0` 让 cosine 从训练开始就衰减，绕过 critic_warmup 冲突。即使步 1-60 actor 实际 LR=0（critic warmup），调度器内的计数器仍在推进，所以步 61 时 LR 已完成 ~3% 的 cosine 衰减，能更早产生约束效果。

#### 14.6.2 v26_klhi_lr5e7.sh（恒定 LR 减半）

| 参数 | 值 |
|------|-----|
| KL_LOSS_COEF | 0.20 |
| ENTROPY_COEFF | 0.005 |
| **ACTOR_LR** | **5e-7**（减半） |
| TOTAL_STEPS | 2000 |
| lr_scheduler_type | constant |

**设计理由**：最简单的单变量消融。所有历史崩塌均发生于 LR=1e-6；将 LR 减半是最直接的保守化策略，无需解决 warmup 冲突。

#### 14.6.3 2×2 因子矩阵

| 实验 | KL+ent 约束强 | LR 保守化 |
|------|-------------|---------|
| v24_groupadv_100scenes | ✗ | ✗ |
| v25_klhi | ✓ | ✗ |
| v26_klhi_lr5e7 | ✓ | ✓（恒定 5e-7） |
| v26_klhi_lrdecay | ✓ | ✓（cosine，步 61+ 衰减） |

---

### 14.7 操作建议与状态总览（2026-05-29）

| 实验 | 状态 | 建议 |
|------|------|------|
| v25_cosine（249/500） | 💀 已崩塌（步 200），stalled | ★★★ **立即终止**，释放 GPU slot |
| **v25_klhi**（317/2000） | ⚠️ 运行中，无崩但 m4↓ | ★★★ **继续运行至步 500+**，观察是否稳定 |
| **v26_klhi_lrdecay** | 脚本已创建（`experiments/v26_klhi_lrdecay.sh`） | ★★★ 启动（slot 1，优先） |
| **v26_klhi_lr5e7** | 脚本已创建（`experiments/v26_klhi_lr5e7.sh`） | ★★ 启动（slot 2，如有资源） |

**启动命令（在 VAGEN-Lite 根目录）**：
```bash
# 终止 v25_cosine（已崩）
# kill <PID>

# 启动 v26_klhi_lrdecay
nohup bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/v26_klhi_lrdecay.sh \
  > v26_klhi_lrdecay.log 2>&1 &
echo "v26_lrdecay PID: $!"

# 启动 v26_klhi_lr5e7
nohup bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/v26_klhi_lr5e7.sh \
  > v26_klhi_lr5e7.log 2>&1 &
echo "v26_lr5e7 PID: $!"
```

---

### 14.8 分析工具

新增可复用分析脚本：`scripts/analyze_experiments.py`

**常用命令**：
```bash
# 查看单实验 val 曲线
python3 scripts/analyze_experiments.py --exps v25_groupadv_100scenes_klhi

# 多实验对比 + train 指标
python3 scripts/analyze_experiments.py \
  --exps v25_groupadv_100scenes_klhi,v26_klhi_lrdecay,v26_klhi_lr5e7 \
  --train --summary --every 40

# 分析历史所有 v24/v25/v26
python3 scripts/analyze_experiments.py \
  --exps v24_groupadv_100scenes,v25_groupadv_100scenes_cosine,v25_groupadv_100scenes_klhi,v26_klhi_lrdecay,v26_klhi_lr5e7 \
  --summary
```

参数说明：`--expdir`（实验根目录，默认 `exps/vagen_active_spatial`）、`--n_id`/`--n_ood`/`--val_n`（val 结构，默认 21/19/4）、`--every`（train 指标采样间隔，默认 20）。
